"""Scorer-shape functionals R_axis, kappa and V at the frozen TRUE geometry (li2ft arm S; R28 recipe A).

Agent li2ft, 2026-10-13. Pre-declaration: docs/research/FEC_COMPETITION_20261007.md, block "Pre-declaration, last-push
lines after Round 16" -> "Ch1, arm S" + its Amendment 1. Detail doc: docs/archive/fec/LI2FT_ARMS_SN_20261013.md.

The scorer (fusion_scoring) derives R_axis from the strongest interior local maximum of phi = AXIS_SIGN * psi with a
separable three-point parabolic sub-pixel step (o_point._subpixel), and kappa / V from the extracted 512-point LCFS
(derive._shape_from_contour: kappa = (Zmax - Zmin) / (Rmax - Rmin), V = 2 pi |M| with the polygon first moment M). Here,
for a predicted map phi_hat, with every discrete choice frozen at the TRUE map:
  * R_axis_hat = R[ir] + dR * 0.5 (a - c) / (a - 2 b + c), (a, b, c) = phi_hat at the true O-point cell (iz, ir-1), (iz, ir),
    (iz, ir+1): the scorer's own formula on the true branch. Masked (detached) when the true cell is not a local maximum of
    phi_hat (8 neighbours), the curvature is not negative or the step would be clipped (|d| >= 1), or the true branch
    itself is irregular.
  * the true scorer LCFS C (512 points; the li2 cache contour, built with the scorer's LCFS routine and the TRAIN-only
    envelope env_train) is displaced along the frozen true normals n = grad phi_true / |grad phi_true| by
        dn_i = -[(phi_hat(C_i) - phi_true(C_i)) - (b_hat - b_true)] / |grad phi_true(C_i)|,
    b = phi at the true ACTIVE point (the X-point saddle or the envelope contact that limits the true LCFS); this is the
    first-order level-set perturbation, exactly zero at phi_hat = phi_true. Masked (detached): frames whose active point
    is ambiguous (both or neither of X-point / envelope contact), points with |grad phi_true| below G_REL x the frame median
    (TRAIN-chosen) and points within X_EXCL cells of the X-point (displacement 0 there), frames with fewer than 90% valid
    points, frames leaving the trust region (more than 5% of the valid points with |dn| > 0.5 cell, decided without
    gradient), and for kappa frames whose extrema are tied (another LOCAL extremum of the same coordinate within
    1e-4 m of the global one, at an index distance > 3).
  * kappa_hat and V_hat from the displaced contour with the true extrema indices and the true orientation sign.
  L_shape = (0.20/7) * sum_f sum_frames mask_f (f_hat - f_true)^2 / Var_TRAIN(f) / B, f in {R_axis, kappa, V}.
The geometry cache is a CPU precompute from the float32 true maps (declared and uncapped, like the li2 cache).

  python -B fec/li2_shape.py build --li2_cache DIR --out DIR [--nproc 4]
  python -B fec/li2_shape.py check --li2_cache DIR --shape_cache DIR [--n 300]   (zero at the truth, scorer parity)
Run with FEC_WORK=<idx root>.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NPTS = 512
SIGN = -1.0                    # fusion_scoring.common.AXIS_SIGN["DIII-D"]: phi = -psi has the axis as a maximum
WC = 0.20 / 7                  # score weight of one consistency functional
# TRAIN-chosen thresholds (fixed before any arm-S number; fractions on TRAIN in meta.json)
TAU_X = 5e-3                   # |phi_saddle - b| / (phi_axis - b) below which the X-point is active
X_DIST = 1.5                   # saddle within this many cells of the contour
TAU_L = 0.02                   # min envelope value on the contour - 0.5 below which the envelope contact is active
G_REL = 0.25                   # |grad phi_true| below this fraction of the frame median: point excluded
X_EXCL = 2.0                   # points within this many cells of the X-point: excluded
MIN_VALID = 0.90               # fraction of valid contour points needed for kappa / V
EXT_TOL = 1e-4                 # extrema tie tolerance [m]
TRUST_CELL = 0.5               # trust region: |dn| <= 0.5 cell ...
TRUST_FRAC = 0.05              # ... on all but 5% of the valid points
FIELDS = ("ax_cell", "ax_ok", "r_ax", "kap", "vol", "msign", "ext", "kap_ok", "act_type", "act_rz", "b_act", "phiC",
          "g", "theta", "pt_ok", "kv_ok")


# ----------------------------------------------------------------------------- numpy: true geometry of one frame
def _bilin(field, R, Z, rq, zq):
    import derive as DV
    return DV._bilinear(field, R, Z, rq, zq)


def saddles(phi):
    """Interior grid saddles of phi (8-neighbour ring with >= 4 sign changes), Newton-refined on the 3x3 stencil (index
    units); returns a list of (z_idx, r_idx, phi_value)."""
    nz, nr = phi.shape
    c = phi[1:-1, 1:-1]
    ring = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    s = np.stack([np.sign(phi[1 + dz:nz - 1 + dz, 1 + dr:nr - 1 + dr] - c) for dz, dr in ring])
    ch = (s != np.roll(s, 1, axis=0)).sum(0)
    out = []
    for iz, ir in zip(*np.nonzero(ch >= 4)):
        z, r = iz + 1, ir + 1
        gz = 0.5 * (phi[z + 1, r] - phi[z - 1, r]); gr = 0.5 * (phi[z, r + 1] - phi[z, r - 1])
        hzz = phi[z + 1, r] - 2 * phi[z, r] + phi[z - 1, r]; hrr = phi[z, r + 1] - 2 * phi[z, r] + phi[z, r - 1]
        hzr = 0.25 * (phi[z + 1, r + 1] - phi[z + 1, r - 1] - phi[z - 1, r + 1] + phi[z - 1, r - 1])
        det = hzz * hrr - hzr * hzr
        if det >= 0:
            continue
        dz = -(hrr * gz - hzr * gr) / det; dr = -(hzz * gr - hzr * gz) / det
        if max(abs(dz), abs(dr)) > 1.0:
            continue
        val = phi[z, r] + gz * dz + gr * dr + 0.5 * (hzz * dz * dz + 2 * hzr * dz * dr + hrr * dr * dr)
        out.append((z + dz, r + dr, val))
    return out


def frame_geometry(psi, C, R, Z, mc, mf):
    """True geometry of one frame (float64 math). psi (65,65) true map, C (512,2) its scorer LCFS (li2 cache contour)."""
    import derive as DV
    from o_point import find_o_point
    phi = SIGN * np.asarray(psi, np.float64)
    C = np.asarray(C, np.float64)
    dRg, dZg = float(R[1] - R[0]), float(Z[1] - Z[0])
    cell = max(abs(dRg), abs(dZg))
    out = {}
    # axis: the scorer's O-point (env_train envelope as mask_coarse, as cons_project/xli) and its parabolic step
    Ra, Za, iz, ir, used_local = find_o_point(phi, R, Z, mc)
    a, b, c = (phi[iz, ir - 1], phi[iz, ir], phi[iz, ir + 1]) if 0 < ir < phi.shape[1] - 1 else (0.0, 0.0, 0.0)
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if den != 0 else 0.0
    out.update(ax_cell=np.array([iz, ir], np.int16), r_ax=np.float32(Ra),
               ax_ok=bool(used_local and 0 < ir < phi.shape[1] - 1 and 0 < iz < phi.shape[0] - 1 and den < 0
                          and abs(d) < 1.0))
    # contour shape (the scorer's functions) and frozen indices / orientation
    sh = DV._shape_from_contour(C)
    Rc, Zc = C[:, 0], C[:, 1]
    R2, Z2 = np.append(Rc, Rc[0]), np.append(Zc, Zc[0])
    M = (1.0 / 6.0) * np.sum((R2[:-1] + R2[1:]) * (R2[:-1] * Z2[1:] - R2[1:] * Z2[:-1]))
    ext = np.array([np.argmax(Rc), np.argmin(Rc), np.argmax(Zc), np.argmin(Zc)], np.int16)
    tie = False
    idx = np.arange(NPTS)
    for k, vals in enumerate((Rc, -Rc, Zc, -Zc)):
        # another LOCAL extremum of the same coordinate along the contour, within EXT_TOL of the global one
        loc = (vals >= np.roll(vals, 1)) & (vals >= np.roll(vals, -1))
        dist = np.abs((idx - ext[k] + NPTS // 2) % NPTS - NPTS // 2)
        if np.any(loc & (dist > 3) & (vals.max() - vals <= EXT_TOL)):
            tie = True
    out.update(kap=np.float32(sh["kappa"]), vol=np.float32(sh["volume"]), msign=np.int8(1 if M >= 0 else -1), ext=ext)
    # gradient of phi_true at the contour: |grad| and direction (frozen normals), phi_true at the contour
    gz, gr = np.gradient(phi, Z, R)
    gRc, gZc = _bilin(gr, R, Z, Rc, Zc), _bilin(gz, R, Z, Rc, Zc)
    g = np.hypot(gRc, gZc)
    phiC = _bilin(phi, R, Z, Rc, Zc)
    blev = float(np.median(phiC))
    depth = float(phi[iz, ir] - blev)
    # active point: X-point saddle at the LCFS level and/or envelope contact
    xs = [(z, r, v) for z, r, v in saddles(phi)]
    xact, xbest = False, None
    if xs and depth > 0:
        cz = (Zc - Z[0]) / dZg; cr = (Rc - R[0]) / dRg
        for z, r, v in xs:
            if abs(v - blev) / depth < TAU_X and np.min(np.hypot(cz - z, cr - r)) < X_DIST:
                if xbest is None or abs(v - blev) < abs(xbest[2] - blev):
                    xbest = (z, r, v)
        xact = xbest is not None
    venv = _bilin(np.asarray(mf, np.float64), R, Z, Rc, Zc)
    lact = bool(np.min(venv) - 0.5 < TAU_L)
    if xact and not lact:
        typ, rz = 1, (R[0] + xbest[1] * dRg, Z[0] + xbest[0] * dZg)
    elif lact and not xact:
        k = int(np.argmin(venv))
        typ, rz = 2, (Rc[k], Zc[k])
    else:
        typ, rz = 0, (np.nan, np.nan)
    rz = tuple(float(v) for v in np.asarray(rz, np.float32))     # the stored (float32) location, sampled identically
    b_act = float(_bilin(phi, R, Z, np.array([rz[0]]), np.array([rz[1]]))[0]) if typ else np.nan
    ok = np.isfinite(g) & (g >= G_REL * np.median(g))
    if typ == 1:
        ok &= np.hypot((Rc - rz[0]) / dRg, (Zc - rz[1]) / dZg) >= X_EXCL
    out.update(act_type=np.int8(typ), act_rz=np.array(rz, np.float32), b_act=np.float32(b_act),
               phiC=phiC.astype(np.float32), g=g.astype(np.float16), theta=np.arctan2(gZc, gRc).astype(np.float16),
               pt_ok=np.packbits(ok), kv_ok=bool(typ in (1, 2) and ok.mean() >= MIN_VALID and np.isfinite(b_act)),
               kap_ok=bool(not tie))
    out["_diag"] = {"xs": len(xs), "xact": xact, "lact": lact, "valid_frac": float(ok.mean()), "depth": depth,
                    "cell": cell}
    return out


# ----------------------------------------------------------------------------- cache build (CPU)
def _build_worker(args):
    key, fidx, pos, src, c_path = args
    from fec.li2_torch import env
    R, Z, mc, mf, _ = env()
    Cm = np.load(c_path, mmap_mode="r")
    with np.load(Path(src) / f"{key}.npz") as z:
        P = np.asarray(z["psi"][fidx], np.float32)
    res = []
    for k in range(len(fidx)):
        res.append(frame_geometry(P[k], np.asarray(Cm[pos[k]], np.float64), R, Z, mc, mf))
    return key, res


def cmd_build(a):
    from multiprocessing import Pool
    from fec.features import load_ds
    from fec.li2_torch import Li2Cache, env
    from fec.paths import WORK
    lc = Li2Cache(a.li2_cache)
    ds = load_ds("diii_d_train")
    n = len(lc.gidx)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R, Z, mc, mf, patched = env()
    shots = ds["shot"][lc.gidx]
    jobs, posmap = [], {}
    for s in np.unique(shots):
        p = np.flatnonzero(shots == s)
        p = p[lc.valid[p]]                                  # frames with a valid li2 target / contour
        if not len(p):
            continue
        key = ds["info"]["order"][int(s)]
        posmap[key] = p
        jobs.append((key, ds["fidx"][lc.gidx[p]], p, str(WORK / "shots" / "diii_d_train"), str(Path(a.li2_cache) / "C.npy")))
    arr = {"ax_cell": np.zeros((n, 2), np.int16), "ax_ok": np.zeros(n, bool), "r_ax": np.full(n, np.nan, np.float32),
           "kap": np.full(n, np.nan, np.float32), "vol": np.full(n, np.nan, np.float32), "msign": np.zeros(n, np.int8),
           "ext": np.zeros((n, 4), np.int16), "kap_ok": np.zeros(n, bool), "act_type": np.zeros(n, np.int8),
           "act_rz": np.full((n, 2), np.nan, np.float32), "b_act": np.full(n, np.nan, np.float32),
           "kv_ok": np.zeros(n, bool)}
    big = {"phiC": ((n, NPTS), np.float32), "g": ((n, NPTS), np.float16), "theta": ((n, NPTS), np.float16),
           "pt_ok": ((n, NPTS // 8), np.uint8)}
    mm = {k: np.lib.format.open_memmap(out / f"{k}.tmp.npy", "w+", dt, sh) for k, (sh, dt) in big.items()}
    diag = {"xs": 0, "xact": 0, "lact": 0, "valid_frac": 0.0, "frames": 0}
    t0 = time.time(); done = 0
    print(f"li2 shape cache: {n} rows ({int(lc.valid.sum())} with a contour), {len(jobs)} shots, nproc {a.nproc}, "
          f"fast helpers {patched}", flush=True)
    with Pool(a.nproc) as pool:
        for key, res in pool.imap_unordered(_build_worker, jobs, chunksize=2):
            p = posmap[key]
            for k, r in zip(p, res):
                for f in arr:
                    arr[f][k] = r[f]
                for f in mm:
                    mm[f][k] = r[f]
                dg = r["_diag"]
                diag["xs"] += dg["xs"] > 0; diag["xact"] += dg["xact"]; diag["lact"] += dg["lact"]
                diag["valid_frac"] += dg["valid_frac"]; diag["frames"] += 1
            done += 1
            if done % 500 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} shots {time.time() - t0:.0f}s", flush=True)
    keys = list(mm)
    for m in mm.values():
        m.flush()
    del m
    mm.clear()                                         # release the memory maps before renaming (Windows)
    for k in keys:
        (out / f"{k}.tmp.npy").replace(out / f"{k}.npy")
    for k, v in arr.items():
        np.save(out / f"{k}.npy", v)
    np.save(out / "gidx.npy", lc.gidx)
    nf = max(diag["frames"], 1)
    ok = lc.valid
    meta = {"n": int(n), "frames_with_contour": int(ok.sum()), "li2_cache": str(a.li2_cache),
            "li2_meta_built_utc": lc.meta.get("built_utc"),
            "thresholds": {"TAU_X": TAU_X, "X_DIST": X_DIST, "TAU_L": TAU_L, "G_REL": G_REL, "X_EXCL": X_EXCL,
                           "MIN_VALID": MIN_VALID, "EXT_TOL": EXT_TOL, "TRUST_CELL": TRUST_CELL, "TRUST_FRAC": TRUST_FRAC},
            "frac": {"ax_ok": float(arr["ax_ok"][ok].mean()), "act_X": float((arr["act_type"][ok] == 1).mean()),
                     "act_L": float((arr["act_type"][ok] == 2).mean()),
                     "act_ambiguous": float((arr["act_type"][ok] == 0).mean()), "kv_ok": float(arr["kv_ok"][ok].mean()),
                     "kap_ok_and_kv_ok": float((arr["kap_ok"] & arr["kv_ok"])[ok].mean()),
                     "frames_with_saddle": diag["xs"] / nf, "mean_valid_point_frac": diag["valid_frac"] / nf},
            "splits": lc.meta.get("splits"), "seconds": time.time() - t0,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("LI2_SHAPE_CACHE_DONE", json.dumps(meta["frac"]), flush=True)


class ShapeCache:
    def __init__(self, d):
        d = Path(d)
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text())
        self.gidx = np.load(d / "gidx.npy")
        self.a = {f: np.load(d / f"{f}.npy", mmap_mode="r" if f in ("phiC", "g", "theta", "pt_ok") else None)
                  for f in FIELDS}


# ----------------------------------------------------------------------------- torch: shape functionals
def _interp_index(x, xp):
    from fec.li2_torch import _interp_index as f
    return f(x, xp)


def point_weights(P, R_t, Z_t):
    """Bilinear (derive._bilinear) node indices / weights of points P (..., Q, 2) on the 65x65 grid."""
    fr = _interp_index(P[..., 0].contiguous(), R_t)
    fz = _interp_index(P[..., 1].contiguous(), Z_t)
    r0 = torch.floor(fr).long().clamp(0, 63)
    z0 = torch.floor(fz).long().clamp(0, 63)
    tr = fr - r0.to(fr.dtype); tz = fz - z0.to(fz.dtype)
    i00 = z0 * 65 + r0
    idx = torch.stack([i00, i00 + 1, i00 + 65, i00 + 66], -1)
    w = torch.stack([(1 - tr) * (1 - tz), tr * (1 - tz), (1 - tr) * tz, tr * tz], -1)
    return idx, w


def sample(phi_flat, idx, w):
    """phi_flat (M, B, 4225), idx/w (B, Q, 4) -> (M, B, Q)."""
    M = phi_flat.shape[0]
    B, Q = idx.shape[:2]
    v = torch.gather(phi_flat, -1, idx.reshape(1, B, Q * 4).expand(M, B, Q * 4)).view(M, B, Q, 4)
    return (v * w).sum(-1)


def shape_terms(phi, geo, grid):
    """phi (M, B, 65, 65) = AXIS_SIGN * psi_hat. geo: dict of batch tensors (B, ...) from the cache (C, phiC, g, n,
    pt_ok, act_rz, b_act, act_type, ext, msign, ax_cell, ax_ok, kv_ok, kap_ok). Returns predicted (M, B) r_ax, kap, vol
    and detached masks ax_m, kap_m, vol_m (M, B)."""
    M, B = phi.shape[:2]
    flat = phi.reshape(M, B, -1)
    R_t, Z_t = grid["R"], grid["Z"]
    dR = float(grid["R_np"][1] - grid["R_np"][0])
    cellmin = min(abs(dR), abs(float(grid["Z_np"][1] - grid["Z_np"][0])))
    iz, ir = geo["ax_cell"][:, 0].long(), geo["ax_cell"][:, 1].long()
    irc = ir.clamp(1, 63); izc = iz.clamp(1, 63)
    base = izc * 65 + irc
    nb = torch.stack([base - 1, base, base + 1], -1)                               # (B, 3): (iz, ir-1), (iz, ir), (iz, ir+1)
    abc = torch.gather(flat, -1, nb.view(1, B, 3).expand(M, B, 3))
    a, b, c = abc[..., 0], abc[..., 1], abc[..., 2]
    den = a - 2 * b + c
    with torch.no_grad():
        ring = torch.stack([base + dz * 65 + dr for dz in (-1, 0, 1) for dr in (-1, 0, 1) if dz or dr], -1)
        nbr = torch.gather(flat, -1, ring.view(1, B, 8).expand(M, B, 8))
        locmax = (b.unsqueeze(-1) >= nbr).all(-1)
        dd = 0.5 * (a - c) / torch.where(den == 0, torch.ones_like(den), den)
        ax_m = locmax & (den < 0) & (dd.abs() < 1.0) & geo["ax_ok"].view(1, B)
    safe = torch.where(ax_m, den, -torch.ones_like(den))
    r_ax = grid["R"][irc].view(1, B) + dR * 0.5 * (a - c) / safe
    # LCFS displacement along frozen true normals
    idx, w = point_weights(geo["C"], R_t, Z_t)
    phiC = sample(flat, idx, w)                                                      # (M, B, 512)
    aidx, aw = point_weights(torch.nan_to_num(geo["act_rz"]).unsqueeze(1), R_t, Z_t)
    bh = sample(flat, aidx, aw)[..., 0]                                              # (M, B)
    g = geo["g"].float().clamp_min(1e-12)
    dn = -((phiC - geo["phiC"].unsqueeze(0)) - (bh - torch.nan_to_num(geo["b_act"])).unsqueeze(-1)) / g.unsqueeze(0)
    pt = geo["pt_ok"].unsqueeze(0)
    dn = torch.where(pt, dn, torch.zeros_like(dn))
    with torch.no_grad():
        nvalid = pt.float().sum(-1).clamp_min(1)
        far = ((dn.abs() > TRUST_CELL * cellmin) & pt).float().sum(-1) / nvalid
        kv_m = geo["kv_ok"].view(1, B) & (far <= TRUST_FRAC)
    Ct = geo["C"].unsqueeze(0) + dn.unsqueeze(-1) * geo["n"].unsqueeze(0)            # (M, B, 512, 2)
    e = geo["ext"].long()
    Rt, Zt = Ct[..., 0], Ct[..., 1]
    ge = lambda X, j: torch.gather(X, -1, e[:, j].view(1, B, 1).expand(M, B, 1))[..., 0]  # noqa: E731
    kap = (ge(Zt, 2) - ge(Zt, 3)) / (ge(Rt, 0) - ge(Rt, 1))
    R2, Z2 = torch.roll(Rt, -1, -1), torch.roll(Zt, -1, -1)
    Mom = (1.0 / 6.0) * ((Rt + R2) * (Rt * Z2 - R2 * Zt)).sum(-1)
    vol = 2 * np.pi * geo["msign"].float().view(1, B) * Mom
    return {"r_ax": r_ax, "kap": kap, "vol": vol, "ax_m": ax_m, "kap_m": kv_m & geo["kap_ok"].view(1, B), "vol_m": kv_m}


def shape_loss(terms, tgt, var, B):
    """(M,) = (0.20/7) * sum_f sum_frames mask (f_hat - f_true)^2 / Var_TRAIN(f) / B."""
    out = 0.0
    for f, mk in (("r_ax", "ax_m"), ("kap", "kap_m"), ("vol", "vol_m")):
        err = torch.where(terms[mk], terms[f] - tgt[f].view(1, -1), torch.zeros_like(terms[f]))
        out = out + WC * err.square().sum(-1) / (var[f] * B)
    return out


def geo_batch(src, pos, dev):
    """Gather the cached geometry of cache positions pos (B,) (numpy) into device tensors (+ normals from theta)."""
    a = src
    t = lambda x, dt=None: torch.as_tensor(np.ascontiguousarray(x), device=dev, dtype=dt)  # noqa: E731
    th = np.asarray(a["theta"][pos], np.float32)
    ptb = np.unpackbits(np.asarray(a["pt_ok"][pos]), axis=-1)[:, :NPTS].astype(bool)
    return {"phiC": t(a["phiC"][pos]), "g": t(np.asarray(a["g"][pos], np.float32)),
            "n": t(np.stack([np.cos(th), np.sin(th)], -1)), "pt_ok": t(ptb), "act_rz": t(a["act_rz"][pos]),
            "b_act": t(a["b_act"][pos]), "ext": t(a["ext"][pos].astype(np.int64)), "msign": t(a["msign"][pos]),
            "ax_cell": t(a["ax_cell"][pos].astype(np.int64)), "ax_ok": t(a["ax_ok"][pos]), "kv_ok": t(a["kv_ok"][pos]),
            "kap_ok": t(a["kap_ok"][pos]), "r_ax": t(a["r_ax"][pos]), "kap": t(a["kap"][pos]), "vol": t(a["vol"][pos])}


def cmd_check(a):
    """At phi_hat = phi_true: the functionals equal the cached true values (R_axis = scorer, kappa/V = scorer contour)
    and dn = 0; prints the max deviations on n random frames with valid branches."""
    from fec.features import load_ds
    from fec.li2_torch import Li2Cache, Li2Grid, env
    from fec.paths import WORK
    lc, sc = Li2Cache(a.li2_cache), ShapeCache(a.shape_cache)
    assert np.array_equal(lc.gidx, sc.gidx)
    R, Z, mc, mf, _ = env()
    g = Li2Grid(R, Z, "cpu", torch.float64)
    grid = {"R": g.R, "Z": g.Z, "R_np": R, "Z_np": Z}
    ds = load_ds("diii_d_train")
    rng = np.random.default_rng(3)
    pos = np.sort(rng.choice(np.flatnonzero(lc.valid), a.n, replace=False))
    P = []
    for p in pos:
        gi = int(lc.gidx[p]); s = int(ds["shot"][gi])
        with np.load(WORK / "shots" / "diii_d_train" / f"{ds['info']['order'][s]}.npz") as z:
            P.append(np.asarray(z["psi"][ds["fidx"][gi]], np.float64))
    phi = SIGN * torch.as_tensor(np.stack(P)).unsqueeze(0)
    geo = geo_batch(sc.a, pos, "cpu")
    geo["C"] = torch.as_tensor(np.asarray(lc.C[pos], np.float64))
    for k in ("phiC", "g", "n", "act_rz", "b_act", "r_ax", "kap", "vol"):
        geo[k] = geo[k].double()
    tm = shape_terms(phi, geo, grid)
    rep = {}
    for f, mk in (("r_ax", "ax_m"), ("kap", "kap_m"), ("vol", "vol_m")):
        m = tm[mk][0].numpy()
        d = np.abs(tm[f][0].numpy() - geo[f].numpy())[m]
        rep[f] = {"frames_masked_in": int(m.sum()), "max_abs_dev_at_truth": float(d.max()) if len(d) else None}
    print("LI2_SHAPE_CHECK", json.dumps(rep), flush=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--li2_cache", required=True); b.add_argument("--out", required=True)
    b.add_argument("--nproc", type=int, default=4)
    c = sp.add_parser("check"); c.add_argument("--li2_cache", required=True); c.add_argument("--shape_cache", required=True)
    c.add_argument("--n", type=int, default=300)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        return cmd_build(a)
    cmd_check(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
