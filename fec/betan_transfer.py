"""betaN transfer study: Thomson electron-pressure betaN (betaN_e) and a machine-agnostic
correction f = betaN / betaN_e, fitted on DIII-D TRAIN, evaluated on DIII-D VAL with OUR
predicted psi, applied zero-shot to MAST (inputs only, never MAST labels).

One per-frame function (`frame_record`) is used for every population; only the flux map differs:

  DIII-D TRAIN   EFIT psi from the shot npz (labels, allowed for fitting)
  DIII-D VAL     fec_work/models/d3d_v2/val_psi.npy (our model's prediction, mimics MAST)
                 (+ EFIT psi as a diagnostic, to separate mapping error from pressure error)
  MAST           fec_work/mast_v2/psi.npy (our label-free GS equilibria)

Geometry is label-free on every population: the magnetic axis is the strongest local maximum of
the axis-maximum flux phi inside a fixed axis box, the boundary is the scorer's closed-surface
bisection (`fusion_scoring.lcfs.extract_lcfs`) called WITHOUT envelope masks (as fec.q95_phys).

betaN_e = 200 mu0 <p_e>_V a / (B0 Ip[MA])   (percent m T / MA), B0 = vacuum field at
          DIII-D RCENTR = 1.6955 m (DIII-D EFIT convention) / MAST magnetic axis (EFIT++).

  v1   faithful reproduction of fec.mast_v1._scal_worker: 10 median bins in psiN < 1, np.interp
       (flat outside the covered bins), 65x65 grid-cell average inside the LCFS.
  v2   robust profile: non-increasing L1 isotonic fit of p_e (and n_e) against psiN using every
       channel with psiN <= 1.15 (SOL channels anchor the boundary), flat core, explicit edge
       rule beyond the outermost channel, 3x refined-grid volume average.

Run (root only; reads DIII-D TRAIN/VAL labels, never TEST, never MAST labels):

    python -B fec/betan_transfer.py records --pop train   # TRAIN sample, EFIT psi
    python -B fec/betan_transfer.py records --pop val     # VAL, predicted psi (+EFIT diag)
    python -B fec/betan_transfer.py records --pop mast    # MAST inputs + mast_v2 psi
    python -B fec/betan_transfer.py fit                   # TRAIN-only fits + CV selection
    python -B fec/betan_transfer.py evaluate              # VAL report (after freeze)
    python -B fec/betan_transfer.py mast                  # betan_v2.npy if accepted
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import SCORING, WORK  # noqa: E402

MU0 = 4e-7 * np.pi
QE = 1.602176634e-19
OUT = WORK / "betan_transfer"
D3D_RREF = 1.6955            # DIII-D EFIT betaN uses BCENTR at RCENTR
D3D_TF_TURNS = 144
MAST_TF_TURNS = 24
D3D_AXIS_BOX = (1.30, 2.20, -0.60, 0.60)
MAST_AXIS_BOX = (0.45, 1.30, -0.60, 0.60)   # = fec.machines.mast_setup axis_box
FINE = 3                     # v2 volume-average refinement of the 65x65 grid
PN_MAX_USE = 1.15            # v2 uses channels up to this psiN (SOL anchors the edge)
EDGE_RULES = ("flat", "lin0")
SALT_TRAIN_SAMPLE = 20261008

FIELDS = ("ok_geo", "Rax", "Zax", "a", "Rgeo", "kappa", "area", "vol", "flux_span",
          "B0", "Bgeo", "Ip", "n_ok", "n_in", "x_lo", "x_hi", "f_core_ext", "f_edge_ext",
          "pav_v1", "pav_v1f", "pav_flat", "pav_lin0", "ne_vol", "p0", "ne0", "te0",
          "n_outl", "inb_out", "li")


# ----------------------------------------------------------------------------- geometry
def _scoring():
    if str(SCORING) not in sys.path:
        sys.path.insert(0, str(SCORING))
    import lcfs  # noqa: E402
    import o_point  # noqa: E402
    return lcfs, o_point


def internal_inductance(phi, gR, gZ, C):
    """Scorer's li(2) functional (fusion_scoring.derive) on our own extracted contour."""
    if str(SCORING) not in sys.path:
        sys.path.insert(0, str(SCORING))
    import derive  # noqa: E402
    return float(derive.internal_inductance(phi, gR, gZ, "", C))


def box_mask(gR, gZ, box):
    RR, ZZ = np.meshgrid(gR, gZ)
    return (RR >= box[0]) & (RR <= box[1]) & (ZZ >= box[2]) & (ZZ <= box[3])


def geometry(phi, gR, gZ, axis_box):
    """Label-free axis/boundary of an axis-MAXIMUM flux map; None on failure."""
    from scipy.interpolate import RectBivariateSpline
    lcfs, op = _scoring()
    am = box_mask(gR, gZ, axis_box)
    C = lcfs.extract_lcfs(phi, gR, gZ, "", am, None, axis_sign=1.0)
    if C is None:
        return None
    Ra, Za, iz, ir, _ = op.find_o_point(phi, gR, gZ, am)
    sp = RectBivariateSpline(gZ, gR, phi)
    pax = max(float(sp(Za, Ra)[0, 0]), float(phi[iz, ir]))
    pb = float(np.mean(sp(C[:, 1], C[:, 0], grid=False)))
    if not (np.isfinite(pax) and np.isfinite(pb) and pax > pb):
        return None
    r, z = C[:, 0], C[:, 1]
    a = 0.5 * (r.max() - r.min())
    if not a > 0:
        return None
    area = 0.5 * abs(np.dot(r, np.roll(z, 1)) - np.dot(z, np.roll(r, 1)))
    return dict(C=C, sp=sp, Rax=Ra, Zax=Za, psi_ax=pax, psi_b=pb, a=a,
                Rgeo=0.5 * (r.max() + r.min()), kappa=(z.max() - z.min()) / (2 * a), area=area)


def chord_pn(geo, R, Z):
    """psiN at chord points (cubic spline of phi); NaN off-grid."""
    pn = (geo["psi_ax"] - geo["sp"](Z, R, grid=False)) / (geo["psi_ax"] - geo["psi_b"])
    return pn


def cells_inside(geo, gR, gZ, fine):
    """Points inside the LCFS polygon on the (refined) grid: (R, psiN). fine=1: 65x65 nodes."""
    from matplotlib.path import Path as MPath
    C = geo["C"]
    if fine == 1:
        rr, zz = gR, gZ
    else:
        dR, dZ = (gR[1] - gR[0]) / fine, (gZ[1] - gZ[0]) / fine
        r0, r1, z0, z1 = C[:, 0].min(), C[:, 0].max(), C[:, 1].min(), C[:, 1].max()
        rr = np.arange(r0 + 0.5 * dR, r1, dR)
        zz = np.arange(z0 + 0.5 * dZ, z1, dZ)
    RR, ZZ = np.meshgrid(rr, zz)
    m = MPath(C[::4]).contains_points(np.c_[RR.ravel(), ZZ.ravel()])   # 128-vertex polygon
    R, Z = RR.ravel()[m], ZZ.ravel()[m]
    if fine == 1:
        # v1 used the grid psi itself (not the spline) on the nodes
        return R, Z, None
    pn = np.clip(chord_pn(geo, R, Z), 0.0, 1.0)
    return R, Z, pn


# ----------------------------------------------------------------------------- profiles
def iso_dec_l1(y):
    """Non-increasing L1 isotonic regression (pool-adjacent-violators with block medians)."""
    y = np.asarray(y, dtype=np.float64)
    starts, vals = [], []
    for i, v in enumerate(y):
        starts.append(i)
        vals.append(v)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            s = starts[-2]
            starts.pop()
            vals.pop()
            vals[-1] = float(np.median(y[s:i + 1]))
    out = np.empty_like(y)
    ends = starts[1:] + [len(y)]
    for s, e, v in zip(starts, ends, vals):
        out[s:e] = v
    return out, np.array(starts), np.array(ends), np.array(vals)


def iso_profile(pn, v, edge="flat", x_hi=None):
    """Nodes (x, y) of a robust non-increasing profile v(psiN) on [0, 1].

    Node x = median psiN of each isotonic block; flat inside the innermost node (np.interp);
    beyond the outermost channel x_hi < 1 the edge rule applies: 'flat' holds the last value,
    'lin0' decays linearly to 0 at psiN = 1."""
    o = np.argsort(pn, kind="stable")
    x, y = pn[o], v[o]
    _, st, en, vals = iso_dec_l1(y)
    xn = np.array([np.median(x[s:e]) for s, e in zip(st, en)])
    yn = vals.copy()
    x_hi = float(x[-1]) if x_hi is None else x_hi
    if x_hi < 1.0:
        if x_hi > xn[-1]:
            xn = np.r_[xn, x_hi]
            yn = np.r_[yn, yn[-1]]
        if edge == "lin0":
            xn = np.r_[xn, 1.0]
            yn = np.r_[yn, 0.0]
    return xn, yn


def bin_profile_v1(pn, pe):
    """fec.mast_v1._scal_worker profile: (bin centres, medians) or None."""
    ins = pn < 1
    if ins.sum() < 6:
        return None
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(pn[ins], bins) - 1
    prof = np.array([np.median(pe[ins][idx == b]) if (idx == b).any() else np.nan
                     for b in range(10)])
    xb = 0.5 * (bins[1:] + bins[:-1])
    okb = np.isfinite(prof)
    if okb.sum() < 4:
        return None
    return xb[okb], prof[okb]


def rmean(v, R):
    return float(np.sum(v * R) / np.sum(R))


# ----------------------------------------------------------------------------- one frame
def frame_record(phi, gR, gZ, axis_box, Rch, Zch, te, ne, F, ip_MA, R_ref=None):
    """All betaN_e variants and transferable features for one frame (dict of floats).

    phi: flux with the axis at a MAXIMUM; te [eV], ne [m^-3] per channel (NaN = missing);
    F = R*B_phi vacuum [T m]; ip_MA = |Ip| [MA]; R_ref: fixed reference radius for B0
    (DIII-D) or None for the vacuum field at the magnetic axis (MAST EFIT++)."""
    rec = {k: np.nan for k in FIELDS}
    rec["ok_geo"] = 0.0
    try:
        geo = geometry(phi, gR, gZ, axis_box)
    except Exception:
        geo = None
    if geo is None:
        return rec
    rec.update(ok_geo=1.0, Rax=geo["Rax"], Zax=geo["Zax"], a=geo["a"], Rgeo=geo["Rgeo"],
               kappa=geo["kappa"], area=geo["area"], flux_span=geo["psi_ax"] - geo["psi_b"])
    try:
        rec["li"] = internal_inductance(phi, gR, gZ, geo["C"])
    except Exception:
        pass
    B0 = F / (R_ref if R_ref is not None else geo["Rax"])
    rec.update(B0=B0, Bgeo=F / geo["Rgeo"], Ip=ip_MA)
    Rf, _, pnf = cells_inside(geo, gR, gZ, FINE)
    if len(Rf) < 20:
        return rec
    dA = (gR[1] - gR[0]) * (gZ[1] - gZ[0]) / FINE ** 2
    rec["vol"] = float(2 * np.pi * np.sum(Rf) * dA)
    ok = np.isfinite(te) & np.isfinite(ne) & np.isfinite(Rch) & np.isfinite(Zch) & (te > 0) & (ne > 0)
    rec["n_ok"] = float(ok.sum())
    if ok.sum() < 10:
        return rec
    pn = chord_pn(geo, Rch[ok], Zch[ok])
    pe = te[ok] * ne[ok] * QE
    rec["n_in"] = float(np.sum(pn < 1))
    # ---- v1 (faithful): 65x65 grid nodes inside the LCFS, grid psi
    b1 = bin_profile_v1(pn, pe)
    if b1 is not None:
        Rc, Zc, _ = cells_inside(geo, gR, gZ, 1)
        if len(Rc) >= 5:
            ir = np.clip(np.round((Rc - gR[0]) / (gR[1] - gR[0])).astype(int), 0, len(gR) - 1)
            iz = np.clip(np.round((Zc - gZ[0]) / (gZ[1] - gZ[0])).astype(int), 0, len(gZ) - 1)
            pnc = (geo["psi_ax"] - phi[iz, ir]) / (geo["psi_ax"] - geo["psi_b"])
            rec["pav_v1"] = rmean(np.interp(pnc, *b1), Rc)
        rec["pav_v1f"] = rmean(np.interp(pnf, *b1), Rf)   # same profile, refined volume
    # ---- v2: robust isotonic profile with SOL anchor and edge rule
    use = np.isfinite(pn) & (pn <= PN_MAX_USE) & (te[ok] < 3e4) & (ne[ok] < 3e21)
    if use.sum() < 8 or np.sum(pn[use] < 1) < 6:
        return rec
    x, p, n = pn[use], pe[use], ne[ok][use]
    tt = te[ok][use]
    x_lo, x_hi = float(x.min()), float(x.max())
    rec.update(x_lo=x_lo, x_hi=x_hi,
               f_core_ext=rmean((pnf < x_lo).astype(float), Rf),
               f_edge_ext=rmean((pnf > x_hi).astype(float), Rf))
    for rule in EDGE_RULES:
        xn, yn = iso_profile(x, p, rule, x_hi)
        rec[f"pav_{rule}"] = rmean(np.interp(pnf, xn, yn), Rf)
        if rule == "flat":
            rec["p0"] = float(yn[0])
            fit = np.interp(x, xn, yn)
            rec["n_outl"] = float(np.sum(np.abs(np.log(np.maximum(p, 1e-30) /
                                                       np.maximum(fit, 1e-30))) > np.log(3.0)))
    xn, yn = iso_profile(x, n, "flat", x_hi)
    rec["ne_vol"] = rmean(np.interp(pnf, xn, yn), Rf)
    rec["ne0"] = float(yn[0])
    xt, yt = iso_profile(x, tt, "flat", x_hi)
    rec["te0"] = float(yt[0])
    # inboard/outboard consistency of the mapping (MAST midplane chord: both sides)
    Rk = Rch[ok][use]
    inb = Rk < geo["Rax"]
    if inb.sum() >= 5 and (~inb).sum() >= 5:
        xi, yi = iso_profile(x[inb], p[inb], "flat", None)
        xo, yo = iso_profile(x[~inb], p[~inb], "flat", None)
        g = np.linspace(max(x[inb].min(), x[~inb].min(), 0.05),
                        min(x[inb].max(), x[~inb].max(), 0.9), 9)
        if g[-1] > g[0]:
            rec["inb_out"] = float(np.mean(np.interp(g, xi, yi)) /
                                   max(np.mean(np.interp(g, xo, yo)), 1e-30))
    return rec


def betan_from_pav(pav, a, B0, Ip):
    """200 mu0 <p> a / (B0 Ip[MA]) in percent m T / MA."""
    return 200.0 * MU0 * pav * a / (np.abs(B0) * np.maximum(np.abs(Ip), 1e-6))


# ----------------------------------------------------------------------------- populations
def _shot_job(job):
    """One shot: job = (machine, npz path, frame indices, psi source, rows, sign[, dR])."""
    machine, path, fidx, psi_path, rows, sign = job[:6]
    dR = job[6] if len(job) > 6 else 0.0          # sensitivity only: shift chord radii
    z = np.load(path)
    gR, gZ = z["grid_R"], z["grid_Z"]
    Rch = np.r_[z["ts_core_R"], z["ts_edge_R"]].astype(float) + dR
    Zch = np.r_[z["ts_core_Z"], z["ts_edge_Z"]].astype(float)
    cur = z["cur"]
    if psi_path == "npz":
        psi_all = z["psi"]
    else:
        psi_all = np.load(psi_path, mmap_mode="r")
    out = []
    for j, k in enumerate(fidx):
        psi = np.asarray(psi_all[k] if psi_path == "npz" else psi_all[rows[j]], dtype=np.float64)
        phi = sign * psi
        te = np.r_[z["ts_core_te"][k], z["ts_edge_te"][k]].astype(float)
        ne = np.r_[z["ts_core_ne"][k], z["ts_edge_ne"][k]].astype(float)
        if machine == "DIII-D":
            bc, ip = cur[k, 19, 6], cur[k, 20, 6]          # magnetics_bcoil, plasma_current
            F = MU0 * D3D_TF_TURNS * abs(bc) * 1e3 / (2 * np.pi)
            r = frame_record(phi, gR, gZ, D3D_AXIS_BOX, Rch, Zch, te, ne, F, abs(ip) / 1e3,
                             R_ref=D3D_RREF)
        else:
            itf, ip = cur[k, 11, 6], cur[k, 13, 6]         # magnetics_tf_current, plasma_current
            F = MU0 * MAST_TF_TURNS * abs(itf) * 1e3 / (2 * np.pi)
            r = frame_record(phi, gR, gZ, MAST_AXIS_BOX, Rch, Zch, te, ne, F, abs(ip) / 1e3,
                             R_ref=None)
        # dynamics (inputs): Ip at -5 / +5 ms offsets, kA -> MA
        r["ip_m5"] = abs(float(cur[k, 20 if machine == "DIII-D" else 13, 4])) / 1e3
        r["ip_p5"] = abs(float(cur[k, 20 if machine == "DIII-D" else 13, 8])) / 1e3
        out.append(r)
    return out


def _check_channels(machine, channels):
    if machine == "DIII-D":
        assert channels[19] == "magnetics_bcoil" and channels[20] == "magnetics_plasma_current"
    else:
        assert channels[11] == "magnetics_tf_current" and channels[13] == "magnetics_plasma_current"


def build_records(pop, nproc=6, n_train_shots=1500, frames_per_shot=8):
    """pop in {'train', 'val', 'val_efit', 'mast'} -> OUT/records_<pop>.npz."""
    from fec.features import load_ds
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = "mast_public_test" if pop == "mast" else "diii_d_train"
    machine = "MAST" if pop == "mast" else "DIII-D"
    ds = load_ds(cfg)
    info = ds["info"]
    _check_channels(machine, info["channels"])
    order = info["order"]
    shot = np.asarray(ds["shot"])
    fidx = np.asarray(ds["fidx"])
    src = WORK / "shots" / cfg
    jobs, gl = [], []
    if pop == "train":
        splits = np.array(info["splits"])
        tr = np.flatnonzero(splits == "train")
        rng = np.random.default_rng(SALT_TRAIN_SAMPLE)
        pick = np.sort(rng.choice(tr, size=min(n_train_shots, len(tr)), replace=False))
        starts = np.searchsorted(shot, pick, side="left")
        ends = np.searchsorted(shot, pick, side="right")
        for s, a0, a1 in zip(pick, starts, ends):
            g = np.arange(a0, a1)
            g = np.sort(rng.choice(g, size=min(frames_per_shot, len(g)), replace=False))
            jobs.append(("DIII-D", str(src / f"{order[s]}.npz"), fidx[g].tolist(), "npz", None, -1.0))
            gl.append(g)
    elif pop in ("val", "val_efit"):
        vidx = np.load(WORK / "models" / "d3d_v2" / "val_idx.npy")
        splits = np.array(info["splits"])
        assert (splits[shot[vidx]] == "val").all()
        vpath = str(WORK / "models" / "d3d_v2" / "val_psi.npy")
        for s in np.unique(shot[vidx]):
            rows = np.flatnonzero(shot[vidx] == s)
            g = vidx[rows]
            psrc = "npz" if pop == "val_efit" else vpath
            jobs.append(("DIII-D", str(src / f"{order[s]}.npz"), fidx[g].tolist(), psrc,
                         rows.tolist(), -1.0))
            gl.append(g)
    elif pop == "mast":
        mpath = str(WORK / "mast_v2" / "psi.npy")
        N = len(shot)
        assert np.load(mpath, mmap_mode="r").shape[0] == N
        for s in np.unique(shot):
            g = np.flatnonzero(shot == s)
            jobs.append(("MAST", str(src / f"{order[s]}.npz"), fidx[g].tolist(), mpath,
                         g.tolist(), 1.0))
            gl.append(g)
    else:
        raise ValueError(pop)
    t0 = time.time()
    recs = []
    with Pool(nproc) as pool:
        for i, r in enumerate(pool.imap(_shot_job, jobs, chunksize=2)):
            recs.extend(r)
            if i % 100 == 0:
                print(pop, i, len(jobs), f"{time.time() - t0:.0f}s", flush=True)
    g = np.concatenate(gl)
    keys = list(recs[0].keys())
    arr = {k: np.array([r[k] for r in recs], dtype=np.float64) for k in keys}
    arr["gidx"] = g
    arr["shot"] = shot[g]
    arr["t_ms"] = np.asarray(ds["t_ms"])[g]
    if machine == "DIII-D":
        arr["betan"] = np.asarray(ds["scal"][g, 1], dtype=np.float64)
        arr["q95"] = np.asarray(ds["scal"][g, 0], dtype=np.float64)
    np.savez(OUT / f"records_{pop}.npz", **arr)
    print(pop, "frames", len(g), "geo ok", np.mean(arr["ok_geo"]),
          "v1 finite", np.isfinite(arr["pav_v1"]).mean(), f"{time.time() - t0:.0f}s", flush=True)
    return arr


# ----------------------------------------------------------------------------- features/models
IP_MIN = 0.05        # MA: frames below are a DIII-D Ip-input artefact (MAST minimum is 80 kA)
BASES = ("v1", "v1f", "flat", "lin0")
SOFT_L1_SCALE = 0.25
RANGE_PCT = (0.5, 99.5)


def load_records(pop):
    return dict(np.load(OUT / f"records_{pop}.npz"))


def base_betan(r, base, guard=True):
    """betaN_e of one profile/averaging variant; NaN where |Ip| < IP_MIN if guard."""
    b = betan_from_pav(r["pav_" + base], r["a"], r["B0"], r["Ip"])
    return np.where(r["Ip"] >= IP_MIN, b, np.nan) if guard else b


def features(r, base):
    """Log features (dimensionless unless noted) from records; dict name -> (N,) array.

    bNe   betaN_e itself                      bp    poloidal electron beta 2mu0<p_e>/<Bp>^2,
    fgw   <n_e>_V / n_Greenwald                     <Bp> = mu0 Ip / (2 pi a sqrt((1+k^2)/2))
    nus   electron collisionality nu*_e (q_cyl, <T_e>, <n_e>, eps; Zeff=1, lnL=17)
    peak  p_e(0)/<p_e>                        qcyl  5 a^2 B_geo (1+k^2)/2 / (R_geo Ip)
    te    <T_e> keV (DIMENSIONAL)             ne    <n_e> 1e19 (DIMENSIONAL)
    dlnip dIp/dt / Ip over +-5 ms [1/s] (time units: machine-specific, diagnostic only)
    lam   2 R_geo (R_axis - R_geo) / a^2 from the flux map (~ betap + li/2), not logged
    li    scorer li(2) of the flux map (linear); bpm = lam - li/2 (linear)"""
    pav = r["pav_" + base]
    a, Ip, kap, Rg, ne = r["a"], r["Ip"], r["kappa"], r["Rgeo"], r["ne_vol"]
    with np.errstate(all="ignore"):
        te = pav / (QE * ne) / 1e3
        shape2 = (1 + kap ** 2) / 2
        qcyl = 5 * a ** 2 * r["Bgeo"] * shape2 / (Rg * Ip)
        nus = 6.921e-18 * qcyl * Rg * ne * 17.0 / ((te * 1e3) ** 2 * (a / Rg) ** 1.5)
        Bp = MU0 * Ip * 1e6 / (2 * np.pi * a * np.sqrt(shape2))
        F = dict(bNe=base_betan(r, base), bp=2 * MU0 * pav / Bp ** 2,
                 fgw=(ne / 1e20) / (Ip / (np.pi * a ** 2)), nus=nus, peak=r["p0"] / pav,
                 qcyl=qcyl, te=te, ne=ne / 1e19)
        X = {k: np.log(np.where(v > 0, v, np.nan)) for k, v in F.items()}
        X["dlnip"] = (r["ip_p5"] - r["ip_m5"]) / 0.010 / Ip
        X["lam"] = 2 * Rg * (r["Rax"] - Rg) / a ** 2     # Shafranov-shift parameter (linear)
        if "li" in r:
            X["li"] = r["li"]                              # scorer li(2) of the flux map
            X["bpm"] = X["lam"] - 0.5 * r["li"]            # shift-based magnetic betap proxy
    return X


def fit_model(r, base, feats, mask):
    """betaN = betaN_e * exp(c0 + sum c_j z_j), z = standardised log features (TRAIN stats);
    robust (soft-L1) least squares on betaN residuals over TRAIN frames in `mask`."""
    from scipy.optimize import least_squares
    b = base_betan(r, base)
    X = features(r, base)
    y = r["betan"]
    m = mask & np.isfinite(b) & (b > 0) & np.isfinite(y) & (y > 0) & (y < 8)
    for f in feats:
        m &= np.isfinite(X[f])
    Z = np.stack([X[f][m] for f in feats], 1) if feats else np.zeros((m.sum(), 0))
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    lo, hi = np.percentile(Z, RANGE_PCT[0], axis=0), np.percentile(Z, RANGE_PCT[1], axis=0)
    Zs = (Z - mu) / sd
    bm, ym = b[m], y[m]

    def res(c):
        return bm * np.exp(c[0] + Zs @ c[1:]) - ym
    c0 = np.r_[np.log(np.median(ym / bm)), np.zeros(len(feats))]
    sol = least_squares(res, c0, loss="soft_l1", f_scale=SOFT_L1_SCALE)
    return dict(base=base, feats=list(feats), mu=mu.tolist(), sd=sd.tolist(), lo=lo.tolist(),
                hi=hi.tolist(), coef=sol.x.tolist(), n_fit=int(m.sum()))


def predict(model, r):
    """(betaN prediction, in-range mask). Features are clamped to the TRAIN 0.5-99.5% range."""
    b = base_betan(r, model["base"])
    c = np.asarray(model["coef"])
    if not model["feats"]:
        return b * np.exp(c[0]), np.isfinite(b)
    X = features(r, model["base"])
    Z = np.stack([X[f] for f in model["feats"]], 1)
    lo, hi = np.asarray(model["lo"]), np.asarray(model["hi"])
    inr = np.all((Z >= lo) & (Z <= hi), 1)
    Zs = (np.clip(Z, lo, hi) - np.asarray(model["mu"])) / np.asarray(model["sd"])
    return b * np.exp(c[0] + Zs @ c[1:]), inr & np.isfinite(b)


def r2(y, p):
    """Pooled R2 over frames with finite labels; non-finite predictions -> label mean."""
    ok = np.isfinite(y)
    y, p = y[ok], p[ok]
    p = np.where(np.isfinite(p), p, y.mean())
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def fill_by_shot(x, shot):
    """NaN -> shot median of finite values, else median of all finite (fec.build_submission)."""
    x = x.copy()
    glob = np.nanmedian(x)
    for s in np.unique(shot[~np.isfinite(x)]):
        m = shot == s
        x[m & ~np.isfinite(x)] = np.nanmedian(x[m]) if np.isfinite(x[m]).any() else glob
    return x


def postprocess(x, shot, k=5, clip=(0.0, 8.0)):
    """Production MAST post-processing (fec.build_submission): fill, shot running median, clip.
    Frames must be in time order within each shot."""
    from scipy.ndimage import median_filter
    y = fill_by_shot(x, shot)
    z = y.copy()
    for s in np.unique(shot):
        m = np.flatnonzero(shot == s)
        if len(m) >= 3:
            z[m] = median_filter(y[m], size=min(k, len(m) | 1), mode="nearest")
    return np.clip(z, *clip)


def shot_folds(shot, k=5, seed=0):
    u = np.unique(shot)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(u)
    fold_of = {s: i % k for i, s in enumerate(perm)}
    return np.array([fold_of[s] for s in shot])


def cv_r2(r, base, feats, k=5, seed=0):
    """Grouped-by-shot k-fold CV R2 on TRAIN records (raw frames, NaN -> fold median)."""
    fo = shot_folds(r["shot"], k, seed)
    pred = np.full(len(fo), np.nan)
    for i in range(k):
        mdl = fit_model(r, base, feats, fo != i)
        p, _ = predict(mdl, {kk: v[fo == i] for kk, v in r.items()})
        pred[fo == i] = np.where(np.isfinite(p), p, np.nanmedian(p))
    return r2(r["betan"], pred), pred


# ----------------------------------------------------------------------------- frozen protocol
# Frozen 2026-10-08 before any VAL evaluation (docs/research/BETAN_TRANSFER_20261008.md, sec. 2).
# Exploration that led here used DIII-D TRAIN records (EFIT psi) and MAST inputs only.
CANDIDATES = {
    "v1_const": ("v1", []),                  # current estimator with a TRAIN-fitted constant
    "const": ("lin0", []),                   # improved betaN_e, constant factor
    "fgw": ("lin0", ["fgw"]),
    "nus": ("lin0", ["nus"]),
    "bp": ("lin0", ["bp"]),
    "peak": ("lin0", ["peak"]),
    "bNe": ("lin0", ["bNe"]),
    "qcyl": ("lin0", ["qcyl"]),
    "lam": ("lin0", ["lam"]),
    "lam_fgw": ("lin0", ["lam", "fgw"]),
    "bpm_fgw": ("lin0", ["bpm", "fgw"]),
    "bp_fgw_peak": ("lin0", ["bp", "fgw", "peak"]),
    "bp_lam_fgw": ("lin0", ["bp", "lam", "fgw"]),
    "te_ne": ("lin0", ["te", "ne"]),         # dimensional: reported, never eligible
    "dlnip": ("lin0", ["dlnip"]),            # machine time units: reported, never eligible
}
NON_TRANSFERABLE = {"te", "ne", "dlnip"}
SELECT_BASE = "lin0"
MAX_FEATS = 3
PARSIMONY = 0.002        # fewest features within this TRAIN-CV R2 of the best eligible
MIN_INRANGE = 0.90       # joint feature in-range fraction over MAST frames with finite betaN_e
ACCEPT_GAIN = 0.02       # VAL post-processed R2 gain over the guarded 2.3 x v1 baseline
BASELINE_K = 2.3


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def select(cands):
    """Frozen TRAIN-only selection among eligible candidates (see CANDIDATES rules)."""
    elig = {n: c for n, c in cands.items()
            if c["model"]["base"] == SELECT_BASE and len(c["model"]["feats"]) <= MAX_FEATS
            and not (set(c["model"]["feats"]) & NON_TRANSFERABLE)
            and c["mast_inrange"] >= MIN_INRANGE}
    best = max(c["cv_r2"] for c in elig.values())
    near = [n for n, c in elig.items() if c["cv_r2"] >= best - PARSIMONY]
    return min(near, key=lambda n: (len(elig[n]["model"]["feats"]), -elig[n]["cv_r2"])), \
        sorted(elig)


def cmd_fit():
    """TRAIN-only: CV R2, full-TRAIN fit and MAST in-range per candidate; frozen selection."""
    r, m = load_records("train"), load_records("mast")
    cands = {}
    for name, (base, feats) in CANDIDATES.items():
        cv, _ = cv_r2(r, base, feats)
        mdl = fit_model(r, base, feats, np.ones(len(r["betan"]), bool))
        _, inr = predict(mdl, m)
        bm = np.isfinite(base_betan(m, base))
        cands[name] = dict(model=mdl, cv_r2=cv, mast_inrange=float(np.mean(inr[bm])),
                           f_at_mast_median=float(np.nanmedian(predict(mdl, m)[0][bm] /
                                                               base_betan(m, base)[bm])))
        print(f"{name:12s} CV R2 {cv:.4f}  MAST in-range {cands[name]['mast_inrange']:.3f}  "
              f"coef {np.round(mdl['coef'], 3).tolist()}", flush=True)
    selected, eligible = select(cands)
    sel = dict(selected=selected, eligible=eligible, candidates=cands,
               rules=dict(SELECT_BASE=SELECT_BASE, MAX_FEATS=MAX_FEATS, PARSIMONY=PARSIMONY,
                          MIN_INRANGE=MIN_INRANGE, ACCEPT_GAIN=ACCEPT_GAIN,
                          BASELINE_K=BASELINE_K, IP_MIN=IP_MIN, RANGE_PCT=RANGE_PCT,
                          NON_TRANSFERABLE=sorted(NON_TRANSFERABLE)),
               records_sha256={p: _sha256(OUT / f"records_{p}.npz") for p in ("train", "mast")})
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print("selected", selected, "eligible", eligible)
    return sel


def _diagnostics(v, y, shot):
    """Error-structure diagnostics on one labelled population (VAL labels; report only)."""
    d = {}
    b = base_betan(v, "lin0")
    ok = np.isfinite(b) & np.isfinite(y) & (y > 0.02) & (b > 0.005)
    lf = np.log(y[ok] / b[ok])
    u, inv = np.unique(shot[ok], return_inverse=True)
    sm = np.bincount(inv, lf) / np.bincount(inv)
    d["logf_sd"] = float(lf.std())
    d["logf_var_between_shot"] = float(np.var(sm[inv]))
    d["logf_var_within_shot"] = float(np.var(lf - sm[inv]))
    # oracles that use VAL labels (upper bounds, never deployable)
    k = np.sum(b[ok] * y[ok]) / np.sum(b[ok] ** 2)
    d["oracle_global_K"] = float(k)
    d["oracle_global_K_r2_raw"] = r2(y, fill_by_shot(k * b, shot))
    po = np.full(len(y), np.nan)
    po[np.flatnonzero(ok)] = b[ok] * np.exp(sm[inv])
    d["oracle_per_shot_factor_r2_raw"] = r2(y, fill_by_shot(np.where(ok, po, b * k), shot))
    res = (y - postprocess(BASELINE_K * base_betan(v, "v1"), shot)) ** 2
    res = np.sort(res[np.isfinite(res)])[::-1]
    d["baseline_ss_share_top1pct"] = float(res[:len(res) // 100].sum() / res.sum())
    d["baseline_ss_share_top5pct"] = float(res[:len(res) // 20].sum() / res.sum())
    d["n_ip_below_min"] = int(np.sum(v["Ip"] < IP_MIN))
    d["edge_extrap_frac"] = float(np.mean(v["f_edge_ext"] > 0))
    return d


def cmd_evaluate(pops=("val", "val_efit")):
    """VAL report for the frozen candidates; acceptance of the frozen selection."""
    sel = json.loads((OUT / "selection.json").read_text())
    rep = dict(selected=sel["selected"], selection_sha256=_sha256(OUT / "selection.json"))
    for pop in pops:
        f = OUT / f"records_{pop}.npz"
        if not f.exists():
            continue
        v = load_records(pop)
        y, shot = v["betan"], v["shot"]
        rows = {}

        def add(name, p):
            rows[name] = dict(r2_pp=r2(y, postprocess(p, shot)), r2_raw=r2(y, fill_by_shot(p, shot)),
                              finite=float(np.isfinite(p).mean()))
        add("baseline_2.3xv1_noguard", BASELINE_K * base_betan(v, "v1", guard=False))
        add("baseline_2.3xv1", BASELINE_K * base_betan(v, "v1"))
        add("2.3xlin0", BASELINE_K * base_betan(v, "lin0"))
        for name, c in sel["candidates"].items():
            add(name, predict(c["model"], v)[0])
        rep[pop] = dict(rows=rows, diagnostics=_diagnostics(v, y, shot), n=int(len(y)),
                        records_sha256=_sha256(f))
        for n, rr in rows.items():
            print(f"{pop:8s} {n:24s} R2 pp {rr['r2_pp']:.4f}  raw {rr['r2_raw']:.4f}  "
                  f"finite {rr['finite']:.4f}", flush=True)
    if "val" in rep:
        s = sel["selected"]
        gain = rep["val"]["rows"][s]["r2_pp"] - rep["val"]["rows"]["baseline_2.3xv1"]["r2_pp"]
        inr = sel["candidates"][s]["mast_inrange"]
        rep["acceptance"] = dict(selected=s, gain_vs_guarded_baseline=gain, mast_inrange=inr,
                                 gain_vs_unguarded_baseline=rep["val"]["rows"][s]["r2_pp"] -
                                 rep["val"]["rows"]["baseline_2.3xv1_noguard"]["r2_pp"],
                                 accepted=bool(gain >= ACCEPT_GAIN and inr >= MIN_INRANGE))
        print("acceptance", rep["acceptance"])
    (OUT / "val_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def mast_betan(model, records, shot_all):
    """MAST betaN for every frame of the config: model on records, production post-processing.

    records: build_records('mast') output (mast_v2 psi + Thomson + Ip/I_TF inputs only);
    shot_all: ds shot index of every frame (time-ordered within shot)."""
    p, inr = predict(model, records)
    full = np.full(len(shot_all), np.nan)
    full[records["gidx"]] = p
    return postprocess(full, np.asarray(shot_all)), inr


def cmd_mast(force=False):
    """Write WORK/mast_v2/betan_v2.npy only if the frozen acceptance rule passed on VAL."""
    from fec.features import load_ds
    rep = json.loads((OUT / "val_report.json").read_text())
    if not rep.get("acceptance", {}).get("accepted") and not force:
        print("NOT accepted: betan_v2.npy not written", rep.get("acceptance"))
        return None
    sel = json.loads((OUT / "selection.json").read_text())
    model = sel["candidates"][sel["selected"]]["model"]
    m = load_records("mast")
    ds = load_ds("mast_public_test")
    shot_all = np.asarray(ds["shot"])
    assert np.array_equal(shot_all[m["gidx"]], m["shot"])
    out, inr = mast_betan(model, m, shot_all)
    dst = WORK / "mast_v2" / "betan_v2.npy"
    np.save(dst, out.astype(np.float64))
    old = postprocess(BASELINE_K * base_betan(m, "v1", guard=False), m["shot"])
    info = dict(file=str(dst), sha256=_sha256(dst), n=int(len(out)), selected=sel["selected"],
                inrange=float(np.mean(inr[np.isfinite(base_betan(m, sel["candidates"][sel["selected"]]["model"]["base"]))])),
                median=float(np.median(out)), p5_p95=np.percentile(out, [5, 95]).tolist(),
                corr_vs_current_2p3xv1=float(np.corrcoef(out, old)[0, 1]),
                median_ratio_vs_current=float(np.median(out / np.maximum(old, 1e-6))))
    (OUT / "mast_betan_v2.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return out


def cmd_sensitivity(pop="mast", n_shots=60, dR=0.02, nproc=6):
    """Mapping robustness (inputs only on MAST): relative betaN_e change when every Thomson
    chord radius is shifted by +-dR (emulates an equilibrium radial misplacement)."""
    from fec.features import load_ds
    cfg = "mast_public_test" if pop == "mast" else "diii_d_train"
    ds = load_ds(cfg)
    info = ds["info"]
    shot = np.asarray(ds["shot"])
    rng = np.random.default_rng(7)
    if pop == "mast":
        pool_shots = np.unique(shot)
    else:
        pool_shots = np.flatnonzero(np.array(info["splits"]) == "train")
    pick = np.sort(rng.choice(pool_shots, n_shots, replace=False))
    res = {}
    for d in (-dR, 0.0, dR):
        jobs = []
        for s in pick:
            g = np.flatnonzero(shot == s)[::5]
            src = WORK / "shots" / cfg / f"{info['order'][s]}.npz"
            if pop == "mast":
                jobs.append(("MAST", str(src), ds["fidx"][g].tolist(), str(WORK / "mast_v2" / "psi.npy"),
                             g.tolist(), 1.0, d))
            else:
                jobs.append(("DIII-D", str(src), ds["fidx"][g].tolist(), "npz", None, -1.0, d))
        with Pool(nproc) as pool:
            recs = [x for part in pool.map(_shot_job, jobs) for x in part]
        res[d] = {k: np.array([x[k] for x in recs]) for k in recs[0]}
    out = {}
    for base in ("v1", "lin0"):
        b0 = base_betan(res[0.0], base)
        for d in (-dR, dR):
            rel = base_betan(res[d], base) / b0 - 1
            out[f"{base}_dR{d:+.2f}_median_abs_rel"] = float(np.nanmedian(np.abs(rel)))
            out[f"{base}_dR{d:+.2f}_p90_abs_rel"] = float(np.nanpercentile(np.abs(rel), 90))
    out["n_frames"] = int(len(b0))
    (OUT / f"sensitivity_{pop}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["records", "fit", "evaluate", "mast", "sensitivity"])
    ap.add_argument("--pop", default="train")
    ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--n-train-shots", type=int, default=1500)
    a = ap.parse_args()
    if a.cmd == "records":
        build_records(a.pop, a.nproc, a.n_train_shots)
    elif a.cmd == "fit":
        cmd_fit()
    elif a.cmd == "evaluate":
        cmd_evaluate()
    elif a.cmd == "mast":
        cmd_mast()
    else:
        cmd_sensitivity(a.pop, nproc=a.nproc)
