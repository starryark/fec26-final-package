"""DIII-D label-backed validation of the MAST physics machinery (fec.gs, fec.mast_select_v2/v3).

MAST equilibria cannot be checked against labels.  DIII-D has EFIT labels, so every MAST
component is re-run here on DIII-D inputs and scored with the official local functionals
(fec/score.py; the d3d envelope mask is used only inside scoring):

  TRAIN  calibration only (E-coil effective turns, filament convention, external-field
         correction, plasma region / limiter, default profile, separatrix Te);
  VAL    evaluation only, on a frozen random subset of whole shots (SEED_VAL);
  TEST   never read.

Conventions.  s = sign(Ip(t)); currents are canonicalised by s so Ip > 0 and the canonical flux
psi_c has its axis at a MAXIMUM (fec.gs convention).  DIII-D stores psi = -sign(Ip) psi_phys
(FEC_COMPETITION), hence psi_c = -psi_stored for every frame and predictions are written back
as psi_stored = -psi_c.  Coil currents: F-coils kA-turns, ECOILA kA per turn (x N_E effective
turns, fitted on TRAIN), Ip kA, bcoil kA (x 144 turns for F).

Variants (registry VARIANTS): label-current best cases (LJ_*), label-profile free-boundary
solves (LP_*), fixed-profile solves (FX_*), Thomson-constrained kinetic fits (KF_*), grid
selection by Thomson consistency (GR_*), label oracles (OR_*).  Study note:
docs/research/D3D_PHYSICS_VALIDATION_20261008.md.

    python -B fec/d3d_physics.py subsets
    python -B fec/d3d_physics.py prep --set cal|val
    python -B fec/d3d_physics.py geom
    python -B fec/d3d_physics.py extfit
    python -B fec/gpu_slot.py run --owner d3dphys -- python -B fec/d3d_physics.py run --family F
    python -B fec/d3d_physics.py score --variants A,B
    python -B fec/d3d_physics.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

MU0 = 4e-7 * math.pi
OUT = WORK / "d3d_phys"
CONFIG = "diii_d_train"
R0_D3D = 1.6955                      # DIII-D RCENTR (betaN B0 reference, d3d_scalar_physics)
N_TF = 144                           # DIII-D TF turns: F = mu0 * 144 * I_bcoil / (2 pi)
K_BETA = 2.3                         # EFIT betaN / Thomson electron betaN (DIII-D TRAIN fit)
F_COLS = [f"magnetics_F{i}{s}" for s in "AB" for i in range(1, 10)]
E_COL = "magnetics_ECOILA"
COILS = F_COLS + [E_COL]             # solver column order (+ optional correction modes)
IP_COL = "magnetics_plasma_current"
BT_COL = "magnetics_bcoil"
OFF0 = 6                             # offset index of t + 0 ms in the extracted current box means

# frozen subsets (drawn by `subsets` before any evaluation)
SEED_VAL, N_VAL_SHOTS, VAL_EVERY = 20261008, 40, 2
SEED_CAL, N_CAL_SHOTS, CAL_EVERY = 20261011, 400, 8
# vessel interior bounding box (inner wall R 1.016 m, outer midplane ~2.37 m, |Z| <= 1.37 m)
VESSEL_BOX = (1.016, 2.37, -1.37, 1.37)


# ============================================================================ subsets / data
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_subsets():
    from fec.features import load_ds
    ds = load_ds(CONFIG)
    info = ds["info"]
    sp = np.array(info["splits"])
    val = np.flatnonzero(sp == "val")
    tr = np.flatnonzero(sp == "train")
    vs = np.sort(np.random.default_rng(SEED_VAL).choice(val, N_VAL_SHOTS, replace=False))
    cs = np.sort(np.random.default_rng(SEED_CAL).choice(tr, N_CAL_SHOTS, replace=False))
    shot, fidx = ds["shot"], ds["fidx"]
    vf = np.flatnonzero(np.isin(shot, vs) & (fidx % VAL_EVERY == 0))
    cf = np.flatnonzero(np.isin(shot, cs) & (fidx % CAL_EVERY == 0))
    assert set(sp[shot[vf]]) == {"val"} and set(sp[shot[cf]]) == {"train"}
    out = {"seed_val": SEED_VAL, "n_val_shots": N_VAL_SHOTS, "val_every": VAL_EVERY,
           "seed_cal": SEED_CAL, "n_cal_shots": N_CAL_SHOTS, "cal_every": CAL_EVERY,
           "val_shots": [info["order"][i] for i in vs], "cal_shots": [info["order"][i] for i in cs],
           "val_frames": vf.tolist(), "cal_frames": cf.tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "subsets.json"
    if f.exists():
        raise SystemExit(f"{f} exists (frozen); delete deliberately to redraw")
    f.write_text(json.dumps(out))
    print(f"VAL {len(vs)} shots {len(vf)} frames; CAL {len(cs)} shots {len(cf)} frames; "
          f"sha256 {_sha(f)}")


def load_subsets():
    return json.loads((OUT / "subsets.json").read_text())


def prep(set_name):
    """Inputs (in_<set>.npz) and labels (lab_<set>.npz) for a frozen subset.

    Inputs: canonical currents, Ip, |bcoil|, Thomson chords.  Labels: canonical EFIT psi,
    scalars, LCFS polygon and the label node currents I = -Delta*psi dR dZ / (mu0 R) inside the
    label LCFS (5-point stencil, EFIT's own discretisation)."""
    from fec.features import load_ds
    sub = load_subsets()
    idx = np.asarray(sub[f"{set_name}_frames"])
    ds = load_ds(CONFIG)
    info = ds["info"]
    ch = info["channels"]
    cur = np.asarray(ds["cur"][idx], dtype=np.float64)           # (n, C, K) kA
    ip = ch.index(IP_COL)
    s = np.sign(cur[:, ip, OFF0]); s[s == 0] = 1.0
    cur_c = cur * s[:, None, None]
    shot = ds["shot"][idx]
    fidx = ds["fidx"][idx]
    n = len(idx)
    stems = [info["order"][k] for k in shot]
    # Thomson: pad per-shot chords to a common channel count
    NC, NE = 64, 32
    tv = {k: np.full((n, NC), np.nan, np.float32) for k in ("vR", "vZ", "vte", "vne")}
    tt = {k: np.full((n, NE), np.nan, np.float32) for k in ("tR", "tZ", "tte", "tne")}
    lab = {"psi": np.zeros((n, 65, 65), np.float32), "q95": np.zeros(n), "betan": np.zeros(n),
           "li": np.zeros(n), "raxis": np.zeros(n), "zaxis": np.zeros(n),
           "lcfs": np.full((n, 100, 2), np.nan, np.float32), "lcfs_n": np.zeros(n, int)}
    gR = gZ = None
    cache = {}
    for i in range(n):
        st = stems[i]
        if st not in cache:
            cache.clear()
            cache[st] = dict(np.load(WORK / "shots" / CONFIG / f"{st}.npz"))
        z = cache[st]
        k = int(fidx[i])
        gR, gZ = z["grid_R"], z["grid_Z"]
        nc, ne_ = z["ts_core_te"].shape[1], z["ts_edge_te"].shape[1]
        tv["vR"][i, :nc] = z["ts_core_R"]; tv["vZ"][i, :nc] = z["ts_core_Z"]
        tv["vte"][i, :nc] = z["ts_core_te"][k]; tv["vne"][i, :nc] = z["ts_core_ne"][k]
        tt["tR"][i, :ne_] = z["ts_edge_R"]; tt["tZ"][i, :ne_] = z["ts_edge_Z"]
        tt["tte"][i, :ne_] = z["ts_edge_te"][k]; tt["tne"][i, :ne_] = z["ts_edge_ne"][k]
        lab["psi"][i] = -z["psi"][k]
        for key in ("q95", "betan", "li", "raxis", "zaxis"):
            lab[key][i] = z[key][k]
        m = int(z["lcfs_n"][k])
        lab["lcfs"][i, :m, 0] = z["lcfs_r"][k, :m]; lab["lcfs"][i, :m, 1] = z["lcfs_z"][k, :m]
        lab["lcfs_n"][i] = m
    cols = [ch.index(c) for c in COILS]
    inp = dict(idx=idx, shot=shot, fidx=fidx, t_ms=ds["t_ms"][idx], sign=s,
               I=cur_c[:, cols, OFF0] * 1e3,                       # A(-turn), canonical
               Ip=cur_c[:, ip, OFF0] * 1e3,
               bt=np.abs(cur[:, ch.index(BT_COL), OFF0]),          # kA
               cur_c=cur_c.astype(np.float32), gR=gR, gZ=gZ, **tv, **tt)
    lab["Inode"] = label_node_currents(lab["psi"], lab["lcfs"], lab["lcfs_n"], gR, gZ)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"in_{set_name}.npz", **inp)
    np.savez(OUT / f"lab_{set_name}.npz", **lab)
    tot = lab["Inode"].reshape(n, -1).sum(1)
    print(f"{set_name}: {n} frames; label sum(I)/Ip median {np.median(tot / inp['Ip']):.4f} "
          f"p5/p95 {np.percentile(tot / inp['Ip'], 5):.4f}/{np.percentile(tot / inp['Ip'], 95):.4f}")


def load_set(set_name, labels=False):
    z = dict(np.load(OUT / f"in_{set_name}.npz"))
    if labels:
        z.update({"lab_" + k: v for k, v in np.load(OUT / f"lab_{set_name}.npz").items()})
    return z


# ============================================================================ label currents
def delta_star(psi, gR, gZ):
    """EFIT 5-point Delta* on interior nodes of psi (..., nz, nr); zero on the border."""
    psi = np.asarray(psi, np.float64)
    dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
    d = np.zeros_like(psi)
    c = psi[..., 1:-1, 1:-1]
    pr, pl = psi[..., 1:-1, 2:], psi[..., 1:-1, :-2]
    pu, pd = psi[..., 2:, 1:-1], psi[..., :-2, 1:-1]
    Ri = np.asarray(gR[1:-1], np.float64)
    d[..., 1:-1, 1:-1] = (pr - 2 * c + pl) / dR ** 2 - (pr - pl) / (2 * Ri * dR) + \
        (pu - 2 * c + pd) / dZ ** 2
    return d


def inside_polygon(poly, R, Z):
    from matplotlib.path import Path as MPath
    return MPath(poly).contains_points(np.c_[np.ravel(R), np.ravel(Z)]).reshape(np.shape(R))


def label_node_currents(psi_c, lcfs, lcfs_n, gR, gZ):
    """Node currents [A] = J dR dZ, J = -Delta*psi/(mu0 R), on nodes inside the label LCFS."""
    RR, ZZ = np.meshgrid(gR, gZ)
    dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
    out = np.zeros(psi_c.shape, np.float32)
    for i in range(len(psi_c)):
        m = int(lcfs_n[i])
        if m < 4:
            continue
        ins = inside_polygon(lcfs[i, :m], RR, ZZ)
        J = -delta_star(psi_c[i], gR, gZ) / (MU0 * RR)
        out[i] = np.where(ins, J * dR * dZ, 0.0)
    return out


# ============================================================================ geometry
def filaments(R, Z, W, H, a1, a2, conv, n=None):
    """Sub-filaments of an EFIT coil parallelogram under a shear convention.

    conv 'tan'  (fec.green.element_filaments): x += z / tan(a1), z += x * tan(a2);
         'cot'  : x += z / tan(a1), z += x / tan(a2)   (a2 read as the side angle from horizontal)
         'rect' : angles ignored."""
    nr = max(2, int(np.ceil(W / 0.01))) if n is None else n[0]
    nz = max(2, int(np.ceil(H / 0.01))) if n is None else n[1]
    u = (np.arange(nr) + 0.5) / nr - 0.5
    v = (np.arange(nz) + 0.5) / nz - 0.5
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x, z = uu * W, vv * H
    if conv != "rect":
        if a1 not in (0.0, 90.0) and a1 == a1:
            x = x + z / np.tan(np.deg2rad(a1))
        if a2 not in (0.0, 90.0) and a2 == a2:
            t2 = np.tan(np.deg2rad(a2))
            z = z + (x * t2 if conv == "tan" else x / t2)
    return (R + x).ravel(), (Z + z).ravel()


def coil_matrix(conv="cot"):
    """psi per A(-turn) on the 65x65 grid for COILS (ECOILA column per turn; x N_E later)."""
    from fec.geometry import machine_geometry
    from fec.green import greens_psi
    f = OUT / f"coilM_{conv}.npy"
    if f.exists():
        return np.load(f)
    g = machine_geometry(CONFIG)
    RR, ZZ = np.meshgrid(g["grid_R"], g["grid_Z"])
    pr, pz = RR.ravel(), ZZ.ravel()
    M = np.zeros((pr.size, len(COILS)))
    for (R, Z, W, H, a1, a2), c in zip(g["elements"], g["cols"]):
        if c not in COILS:
            continue
        fr, fz = filaments(R, Z, W, H, a1, a2, conv)
        M[:, COILS.index(c)] += greens_psi(pr[:, None], pz[:, None], fr[None], fz[None]).mean(1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(f, M)
    return M


def build_geometry():
    """Plasma region and limiter from TRAIN (cal) label LCFS points (no scorer mask).

    Polar envelope about (R0, 0) of all cal LCFS points (max per 2-degree bin, widened by a
    +-2-bin running max), + REGION_MARGIN for the current/X-point region, + LIM_MARGIN for the
    limiter polygon; both clipped to VESSEL_BOX.  Axis box: cal label axis 0.1-99.9% range +- 5 cm."""
    lab = np.load(OUT / "lab_cal.npz")
    lc, ln = lab["lcfs"], lab["lcfs_n"]
    pts = np.concatenate([lc[i, :int(ln[i])] for i in range(len(ln))])
    th = np.arctan2(pts[:, 1], pts[:, 0] - R0_D3D)
    rr = np.hypot(pts[:, 0] - R0_D3D, pts[:, 1])
    nb = 180
    b = ((th + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    env = np.zeros(nb)
    np.maximum.at(env, b, rr)
    env = np.max(np.stack([np.roll(env, k) for k in range(-2, 3)]), 0)
    thc = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi

    def poly(margin):
        R = R0_D3D + (env + margin) * np.cos(thc)
        Z = (env + margin) * np.sin(thc)
        R = np.clip(R, VESSEL_BOX[0], VESSEL_BOX[1]); Z = np.clip(Z, VESSEL_BOX[2], VESSEL_BOX[3])
        return np.c_[R, Z]
    ra, za = lab["raxis"], lab["zaxis"]
    geo = {"region": poly(REGION_MARGIN).tolist(), "limiter": poly(LIM_MARGIN).tolist(),
           "axis_box": [float(np.percentile(ra, 0.1) - 0.05), float(np.percentile(ra, 99.9) + 0.05),
                        float(np.percentile(za, 0.1) - 0.05), float(np.percentile(za, 99.9) + 0.05)],
           "lcfs_Rmin": float(pts[:, 0].min()), "lcfs_Rmax": float(pts[:, 0].max()),
           "lcfs_Zmin": float(pts[:, 1].min()), "lcfs_Zmax": float(pts[:, 1].max()),
           "n_frames": int(len(ln)), "region_margin": REGION_MARGIN, "lim_margin": LIM_MARGIN}
    (OUT / "geometry.json").write_text(json.dumps(geo))
    print(json.dumps({k: v for k, v in geo.items() if k not in ("region", "limiter")}))


REGION_MARGIN = 0.04
LIM_MARGIN = 0.015


def load_geometry():
    return json.loads((OUT / "geometry.json").read_text())


def region_mask(gR, gZ, geo=None):
    geo = geo or load_geometry()
    RR, ZZ = np.meshgrid(gR, gZ)
    return inside_polygon(np.asarray(geo["region"]), RR, ZZ)


def plasma_G(gR, gZ, geo=None):
    from fec.gs import plasma_green_matrix_cached
    return plasma_green_matrix_cached("d3dphys", gR, gZ, region_mask(gR, gZ, geo))


# ============================================================================ (A) external field
EXT_K = 24          # residual PCA modes of the learned external correction
DYN_OFFS = [k for k in range(11) if k != OFF0]
FEAT_CH = list(range(18)) + [18, 20]   # cur_c channels: F1A..F9B, ECOILA, Ip (bcoil skipped)


def ext_features(inp, kind):
    """Ridge features (canonical currents, kA): 'static' = 19 coils + Ip at t;
    'dynamic' = static + I(t+o) - I(t) for the other 10 extraction offsets (eddy drive)."""
    cur = inp["cur_c"].astype(np.float64)                   # (n, 21, 11) canonical kA
    X = [cur[:, FEAT_CH, OFF0]]
    if kind == "dynamic":
        X.append((cur[:, FEAT_CH][:, :, DYN_OFFS] -
                  cur[:, FEAT_CH, OFF0][:, :, None]).reshape(len(cur), -1))
    return np.concatenate(X, 1)


def plasma_flux(Inode, G, sidx, nchunk=2048):
    """Green's-function flux (n, Npix) of node currents (n, nz, nr) on the region nodes."""
    n = len(Inode)
    flat = Inode.reshape(n, -1)
    out = np.empty((n, G.shape[0]), np.float64)
    for a in range(0, n, nchunk):
        out[a:a + nchunk] = flat[a:a + nchunk][:, sidx].astype(np.float64) @ G.T
    return out


class FDPlasma:
    """EFIT-style plasma flux: 5-point Delta* solve with Dirichlet values from G on the border."""

    def __init__(self, gR, gZ, G, sidx):
        import scipy.sparse as sps
        from scipy.sparse.linalg import splu
        nz, nr = len(gZ), len(gR)
        dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
        self.nz, self.nr, self.G, self.sidx = nz, nr, G, sidx
        idx = -np.ones((nz, nr), int)
        inner = np.zeros((nz, nr), bool)
        inner[1:-1, 1:-1] = True
        idx[inner] = np.arange(inner.sum())
        rows, cols, vals = [], [], []
        self.bcoef = []                        # (row, border flat index, coefficient)
        for iz in range(1, nz - 1):
            for ir in range(1, nr - 1):
                k = idx[iz, ir]
                R = gR[ir]
                nb = {(iz, ir + 1): 1 / dR ** 2 - 1 / (2 * R * dR),
                      (iz, ir - 1): 1 / dR ** 2 + 1 / (2 * R * dR),
                      (iz + 1, ir): 1 / dZ ** 2, (iz - 1, ir): 1 / dZ ** 2}
                rows.append(k); cols.append(k); vals.append(-2 / dR ** 2 - 2 / dZ ** 2)
                for (a, b), cf in nb.items():
                    if inner[a, b]:
                        rows.append(k); cols.append(idx[a, b]); vals.append(cf)
                    else:
                        self.bcoef.append((k, a * nr + b, cf))
        n = int(inner.sum())
        self.lu = splu(sps.csc_matrix((vals, (rows, cols)), shape=(n, n)))
        self.inner = inner
        self.Rin = np.broadcast_to(np.asarray(gR)[None, :], (nz, nr))[inner]
        self.border = np.flatnonzero(~inner.ravel())
        self.dA = dR * dZ

    def __call__(self, Inode):
        """Inode (nz, nr) node currents -> plasma flux (nz*nr,)."""
        psi = np.zeros(self.nz * self.nr)
        I = np.asarray(Inode, np.float64).ravel()
        psi[self.border] = self.G[self.border] @ I[self.sidx]
        rhs = -MU0 * self.Rin * np.asarray(Inode, np.float64)[self.inner] / self.dA
        for k, j, cf in self.bcoef:
            rhs[k] -= cf * psi[j]
        out = psi.reshape(self.nz, self.nr)
        out[self.inner] = self.lu.solve(rhs)
        return out.ravel()


def _ridge_cv(X, Y, groups, alphas, nfold=5, seed=0):
    """Shot-grouped CV of ridge (standardised X, centred Y); returns best alpha and CV SSE."""
    ug = np.unique(groups)
    fold = np.random.default_rng(seed).permutation(len(ug)) % nfold
    fmap = dict(zip(ug.tolist(), fold.tolist()))
    f = np.array([fmap[g] for g in groups.tolist()])
    sse = np.zeros(len(alphas))
    for k in range(nfold):
        tr, te = f != k, f == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        ym = Y[tr].mean(0)
        Xt = (X[tr] - mu) / sd
        Xe = (X[te] - mu) / sd
        s2, V = _eigh(Xt.T @ Xt)                 # Gram eigendecomposition (no tall SVD)
        VtXtY = V.T @ (Xt.T @ (Y[tr] - ym))
        for j, a in enumerate(alphas):
            W = V @ (VtXtY / (s2 + a)[:, None])
            sse[j] += ((Y[te] - ym - Xe @ W) ** 2).sum()
    return alphas[int(np.argmin(sse))], sse


def _eigh(C):
    """Symmetric eigendecomposition via torch (this numpy/OpenBLAS build's dense LAPACK
    decompositions are pathologically slow on large matrices)."""
    import torch
    w, v = torch.linalg.eigh(torch.as_tensor(C, dtype=torch.float64))
    return w.numpy(), v.numpy()


def top_modes(Rm, k):
    """Top-k principal directions (k, Npix) and variance fractions of centred rows Rm."""
    C = Rm.T @ Rm
    w, v = _eigh(C)
    o = np.argsort(w)[::-1]
    w, v = w[o], v[:, o]
    return v[:, :k].T.copy(), w / w.sum()


def _ridge_fit(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    ym = Y.mean(0)
    Xt = (X - mu) / sd
    W = np.linalg.solve(Xt.T @ Xt + alpha * np.eye(X.shape[1]), Xt.T @ (Y - ym))
    return dict(mu=mu, sd=sd, ym=ym, W=W, alpha=np.float64(alpha))


def _ridge_pred(m, X):
    return ((X - m["mu"]) / m["sd"]) @ m["W"] + m["ym"]


def ext_fit():
    """(A) External-field model on TRAIN (cal) and its error budget on the VAL subset."""
    cal = load_set("cal", labels=True)
    val = load_set("val", labels=True)
    gR, gZ = cal["gR"], cal["gZ"]
    G, sidx = plasma_G(gR, gZ)
    rep = {}
    for nm, d in (("cal", cal), ("val", val)):
        flat = d["lab_Inode"].reshape(len(d["Ip"]), -1)
        outside = np.abs(np.delete(flat, sidx, axis=1)).sum(1) / np.abs(flat).sum(1).clip(1e-9)
        rep[f"{nm}_label_current_outside_region_max"] = float(outside.max())
        d["psi_pl"] = plasma_flux(d["lab_Inode"], G, sidx)
        d["y"] = d["lab_psi"].reshape(len(d["Ip"]), -1).astype(np.float64)
        print(nm, "plasma flux done", flush=True)
    # 1. filament convention + E-coil turns (pooled LS over cal pixels and frames)
    conv_res = {}
    sub = np.arange(0, len(cal["Ip"]), 16)           # gains regression on every 16th cal frame
    for conv in ("tan", "cot", "rect"):
        M = coil_matrix(conv)
        r0 = cal["y"] - cal["psi_pl"] - cal["I"][:, :18] @ M[:, :18].T
        mE = np.outer(cal["I"][:, 18], M[:, 18])
        NE = float((r0 * mE).sum() / (mE ** 2).sum())
        res = r0 - NE * mE
        A = np.stack([np.outer(cal["I"][sub, j], M[:, j]).ravel() for j in range(19)], 1)
        tgt = (cal["y"][sub] - cal["psi_pl"][sub]).ravel()
        gains = np.linalg.lstsq(A, tgt, rcond=None)[0]
        resg = tgt - A @ gains
        conv_res[conv] = dict(N_E=NE, rms_mWb=float(np.sqrt((res ** 2).mean()) * 1e3),
                              rms_free_gains_mWb=float(np.sqrt((resg ** 2).mean()) * 1e3),
                              gains=[round(float(x), 4) for x in gains])
        print(conv, json.dumps(conv_res[conv]), flush=True)
        del A, resg, r0, mE, res, tgt
    rep["conventions"] = conv_res
    conv = min(conv_res, key=lambda c: conv_res[c]["rms_mWb"])
    NE = conv_res[conv]["N_E"]
    M = coil_matrix(conv)
    rep["conv"], rep["N_E"] = conv, NE
    for d in (cal, val):
        d["r_F"] = d["y"] - d["psi_pl"] - d["I"][:, :18] @ M[:, :18].T       # F coils only
        d["r_A0"] = d["r_F"] - NE * np.outer(d["I"][:, 18], M[:, 18])        # + E-coil
    # 2. residual PCA on cal
    mu_r = cal["r_A0"].mean(0)
    modes, ev = top_modes(cal["r_A0"] - mu_r, EXT_K)             # (K, Npix)
    rep["pca_cum_var"] = [round(float(x), 5) for x in np.cumsum(ev)[:EXT_K]]
    print("pca", rep["pca_cum_var"][:8], flush=True)
    # 3. ridge: features -> mode coefficients (shot-grouped CV for alpha)
    models = {}
    for kind in ("static", "dynamic"):
        Xc = ext_features(cal, kind)
        Yc = (cal["r_A0"] - mu_r) @ modes.T
        alphas = np.logspace(-4, 3, 15) * len(Xc)
        a, _ = _ridge_cv(Xc, Yc, cal["shot"], alphas)
        models[kind] = _ridge_fit(Xc, Yc, a)
        print(kind, "alpha/n", a / len(Xc), flush=True)
    # 4. budget: SS of external residuals relative to SS_tot (pooled, as the scorer)
    for nm, d in (("cal", cal), ("val", val)):
        y = d["y"]
        sst = float(((y - y.mean()) ** 2).sum())
        ins = d["lab_Inode"].reshape(len(y), -1) != 0
        b = {"ss_tot": sst, "n": int(len(y))}
        arms = {"F_only": d["r_F"], "A0": d["r_A0"], "A0+const": d["r_A0"] - mu_r}
        for kind in ("static", "dynamic"):
            coef = _ridge_pred(models[kind], ext_features(d, kind))
            arms[f"A1_{kind}"] = d["r_A0"] - mu_r - coef @ modes
        arms["A0+pca_oracle"] = (d["r_A0"] - mu_r) - ((d["r_A0"] - mu_r) @ modes.T) @ modes
        for k, r in arms.items():
            b[k] = dict(r2_ceiling=1 - float((r ** 2).sum()) / sst,
                        rms_mWb=float(np.sqrt((r ** 2).mean()) * 1e3),
                        rms_in_plasma_mWb=float(np.sqrt((r[ins] ** 2).mean()) * 1e3),
                        rms_out_mWb=float(np.sqrt((r[~ins] ** 2).mean()) * 1e3))
        b["plasma_flux_rms_mWb"] = float(np.sqrt((d["psi_pl"] ** 2).mean()) * 1e3)
        b["psi_rms_about_mean_mWb"] = float(np.sqrt(sst / y.size) * 1e3)
        rep[f"budget_{nm}"] = b
        print(nm, json.dumps(b), flush=True)
    # 5. discretisation: EFIT-style FD plasma flux vs Green's-function flux (VAL sample)
    fd = FDPlasma(gR, gZ, G, sidx)
    pick = np.sort(np.random.default_rng(0).choice(len(val["Ip"]), 300, replace=False))
    dd = np.stack([fd(val["lab_Inode"][i]) - val["psi_pl"][i] for i in pick])
    rA0 = val["r_A0"][pick]
    rep["fd_minus_green"] = dict(rms_mWb=float(np.sqrt((dd ** 2).mean()) * 1e3),
                                 rA0_rms_mWb=float(np.sqrt((rA0 ** 2).mean()) * 1e3),
                                 rA0_minus_fd_rms_mWb=float(np.sqrt(((rA0 - dd) ** 2).mean()) * 1e3),
                                 frac_of_A0_ss=float((dd ** 2).sum() / (rA0 ** 2).sum()))
    print("FD check", json.dumps(rep["fd_minus_green"]), flush=True)
    np.savez(OUT / "extfit.npz", conv=conv, N_E=NE, mu_r=mu_r, modes=modes,
             **{f"{k}_{f}": v for k, m in models.items() for f, v in m.items()})
    (OUT / "extfit.json").write_text(json.dumps(rep, indent=1))


def load_ext(kind="dynamic"):
    """(conv, N_E, extra columns (Npix, K+1) = [mean residual, modes], coefficient predictor)."""
    z = np.load(OUT / "extfit.npz")
    m = {f: z[f"{kind}_{f}"] for f in ("mu", "sd", "ym", "W", "alpha")}
    cols = np.concatenate([z["mu_r"][:, None], z["modes"].T], 1)

    def coeffs(inp):
        return np.concatenate([np.ones((len(inp["Ip"]), 1)),
                               _ridge_pred(m, ext_features(inp, kind))], 1)
    return str(z["conv"]), float(z["N_E"]), cols, coeffs


# ============================================================================ solver
def field_matrix(field):
    """Solver external matrix and a per-frame current builder for field model A0 / A1."""
    conv, NE, cols, coeffs = load_ext("dynamic")
    M = coil_matrix(conv).copy()
    M[:, 18] *= NE
    if field == "A0":
        return M, (lambda inp: inp["I"])
    Mx = np.concatenate([M, cols], 1)
    return Mx, (lambda inp: np.concatenate([inp["I"], coeffs(inp)], 1))


def _make_solver_cls():
    import torch
    from fec.gs import MachineSpec
    from fec.mast_select_v3 import V3Solver

    class D3DSolver(V3Solver):
        """fec.gs.Solver numerics (via V3Solver._picard) on the polygonal DIII-D plasma region,
        with an optional polygon limiter (TRAIN LCFS envelope + LIM_MARGIN).

        solve(prof): prof (B,3) = FreeGS beta0, am, an (as fec.gs); prof (B,8) = label-profile
        polynomial J = sum_m a_m psiN^m R/R0 + c_m psiN^m R0/R (m = 0..3).  MAST callers pass
        Rinit = 0.9 m (MAST); an Rinit outside the region is replaced by R0."""

        def __init__(self, Mext, device="cuda", dtype=torch.float32, limiter=False, geo=None):
            from fec.geometry import machine_geometry
            torch.backends.cuda.matmul.allow_tf32 = False
            geo = geo or load_geometry()
            g = machine_geometry(CONFIG)
            gR, gZ = g["grid_R"], g["grid_Z"]
            box = region_mask(gR, gZ, geo)
            RR, ZZ = np.meshgrid(gR, gZ)
            a = geo["axis_box"]
            reg = np.asarray(geo["region"])
            self.spec = MachineSpec("d3dphys", gR, gZ,
                                    box=(reg[:, 0].min(), reg[:, 0].max(), reg[:, 1].min(),
                                         reg[:, 1].max()),
                                    axis_box=tuple(a), R0=R0_D3D, keepout=[])
            self.dev, self.dt = device, dtype
            self.nr, self.nz = len(gR), len(gZ)
            self.dR, self.dZ = gR[1] - gR[0], gZ[1] - gZ[0]
            abox = (RR >= a[0]) & (RR <= a[1]) & (ZZ >= a[2]) & (ZZ <= a[3]) & box
            G, sidx = plasma_G(gR, gZ, geo)
            t = lambda x: torch.as_tensor(x, device=device, dtype=dtype)  # noqa: E731
            self.G = t(G)
            self.sidx = torch.as_tensor(sidx, device=device)
            self.Mext = t(Mext)
            self.RR, self.ZZ = t(RR), t(ZZ)
            self.box = torch.as_tensor(box, device=device)
            self.abox = torch.as_tensor(abox, device=device)
            self.Rs = self.RR.reshape(-1)[self.sidx]
            self.Zs = self.ZZ.reshape(-1)[self.sidx]
            self.limiter = None
            self.last_limited = None
            if limiter:
                self.set_poly_limiter(np.asarray(geo["limiter"]))

        def set_poly_limiter(self, poly, step=0.02):
            pts = []
            for (r1, z1), (r2, z2) in zip(poly, np.roll(poly, -1, 0)):
                n = max(1, int(np.ceil(np.hypot(r2 - r1, z2 - z1) / step)))
                tt = np.arange(n) / n
                pts.append(np.c_[r1 + tt * (r2 - r1), z1 + tt * (z2 - z1)])
            P = np.concatenate(pts)
            gR, gZ = self.spec.gR, self.spec.gZ
            fr = (P[:, 0] - gR[0]) / self.dR
            fz = (P[:, 1] - gZ[0]) / self.dZ
            j = np.clip(np.floor(fr).astype(int), 0, self.nr - 2)
            i = np.clip(np.floor(fz).astype(int), 0, self.nz - 2)
            wr, wz = fr - j, fz - i
            idx = np.stack([i * self.nr + j, i * self.nr + j + 1, (i + 1) * self.nr + j,
                            (i + 1) * self.nr + j + 1], 1)
            w = np.stack([(1 - wr) * (1 - wz), wr * (1 - wz), (1 - wr) * wz, wr * wz], 1)
            self.lim_idx = torch.as_tensor(idx, device=self.dev)
            self.lim_w = torch.as_tensor(w, device=self.dev, dtype=self.dt)
            self.lim_P = torch.as_tensor(P, device=self.dev, dtype=self.dt)
            self.limiter = {"n": len(P)}

        def boundary(self, psi, psi_ax, Rax, Zax, xp, final=False):
            """As V3Solver.boundary with a polygon limiter (points beyond an X-point or above
            the axis flux ignored)."""
            (fu, vu, Rxu, Zxu), (fl, vl, Rxl, Zxl) = xp
            neg = torch.full_like(vu, -1e30)
            psi_b = torch.maximum(torch.where(fu, vu, neg), torch.where(fl, vl, neg))
            limited = torch.zeros_like(fu)
            if self.limiter is not None:
                B = psi.shape[0]
                pl = (psi.reshape(B, -1)[:, self.lim_idx] * self.lim_w).sum(-1)    # (B, L)
                LR, LZ = self.lim_P[:, 0], self.lim_P[:, 1]
                ok = torch.ones_like(pl, dtype=torch.bool)
                for (found, Rx, Zx) in ((fu, Rxu, Zxu), (fl, Rxl, Zxl)):
                    dRx = (Rx - Rax)[:, None]
                    dZx = (Zx - Zax)[:, None]
                    proj = (LR[None] - Rax[:, None]) * dRx + (LZ[None] - Zax[:, None]) * dZx
                    ok &= torch.where(found[:, None], proj < dRx ** 2 + dZx ** 2,
                                      torch.ones_like(ok))
                ok &= pl < psi_ax[:, None]
                plim = torch.where(ok, pl, torch.full_like(pl, -1e30)).max(1).values
                limited = (plim > psi_b) & (plim > -1e29)
                psi_b = torch.where(limited, plim, psi_b)
            if not final:
                nox = ~(fu | fl) & ~limited
                if nox.any():
                    psi_b = torch.where(nox, psi_ax - 0.05 * psi_ax.abs().clamp_min(1e-3), psi_b)
            return psi_b, limited

        def solve(self, Iext, Ip, prof, Zt, n_iter=150, omega=0.4, kc=0.05, I0=None, c0=None,
                  Rinit=None, fix_z=True, verbose=False):
            prof = torch.as_tensor(np.asarray(prof, np.float64) if not torch.is_tensor(prof)
                                   else prof, device=self.dev, dtype=self.dt)
            if Rinit is not None:
                Rinit = np.where(np.asarray(Rinit, float) < self.spec.box[0] + 0.3,
                                 self.spec.R0, Rinit)
            Rs, R0 = self.Rs, self.spec.R0
            if prof.shape[1] == 3:
                beta0, am, an = prof[:, 0:1], prof[:, 1:2], prof[:, 2:3]

                def shape_fn(pn, ms):
                    return (beta0 * Rs[None] / R0 + (1 - beta0) * R0 / Rs[None]) * \
                        (1 - pn ** am).clamp_min(0) ** an * ms
            else:
                a, c = prof[:, 0:4], prof[:, 4:8]

                def shape_fn(pn, ms):
                    pw = torch.stack([torch.ones_like(pn), pn, pn ** 2, pn ** 3], 1)  # (B,4,Ns)
                    return (torch.einsum("bm,bmn->bn", a, pw) * Rs[None] / R0 +
                            torch.einsum("bm,bmn->bn", c, pw) * R0 / Rs[None]).clamp_min(0) * ms
            return self._picard(Iext, Ip, shape_fn, Zt, n_iter, omega, kc, I0, c0, Rinit, fix_z)

        def final_boundary(self, psi):
            """(psi_ax, psi_b, Rax, Zax, limited) of arbitrary maps psi (B,nz,nr) (final=True)."""
            psi = torch.as_tensor(psi, device=self.dev, dtype=self.dt)
            psi_ax, Rax, Zax = self.find_axis(psi)
            xp = self.find_xpoints(psi, Rax, Zax, psi_ax)
            psi_b, lim = self.boundary(psi, psi_ax, Rax, Zax, xp, final=True)
            return psi_ax, psi_b, Rax, Zax, lim

    return D3DSolver


_SOLVER_CLS = None


def make_solver(field="A1", limiter=False, device="cuda"):
    global _SOLVER_CLS
    if _SOLVER_CLS is None:
        _SOLVER_CLS = _make_solver_cls()
    M, cur = field_matrix(field)
    return _SOLVER_CLS(M, device=device, limiter=limiter), cur


def make_bare_solver(device="cpu", limiter=False):
    """Solver geometry only (zero external matrix): label-geometry checks."""
    global _SOLVER_CLS
    if _SOLVER_CLS is None:
        _SOLVER_CLS = _make_solver_cls()
    return _SOLVER_CLS(np.zeros((65 * 65, 1)), device=device, limiter=limiter)


# ============================================================================ label profile
def label_flux_bounds(psi_c, lcfs, lcfs_n, raxis, zaxis, gR, gZ):
    """(psi_ax, psi_b) of label maps: bicubic value at the EFIT axis and mean on the EFIT LCFS."""
    from scipy.interpolate import RectBivariateSpline
    n = len(psi_c)
    pa, pb = np.zeros(n), np.zeros(n)
    for i in range(n):
        sp = RectBivariateSpline(gZ, gR, psi_c[i].astype(np.float64))
        pa[i] = sp(zaxis[i], raxis[i])[0, 0]
        m = int(lcfs_n[i])
        pb[i] = np.mean(sp(lcfs[i, :m, 1], lcfs[i, :m, 0], grid=False))
    return pa, pb


def fit_label_profile(lab, gR, gZ):
    """Per-frame LS fit of label node currents with the 8-term polynomial profile basis
    (D3DSolver prof (B,8)); returns coef (n,8) and the fit's relative residual."""
    RR, ZZ = np.meshgrid(gR, gZ)
    Rf = RR.ravel()
    pa, pb = label_flux_bounds(lab["psi"], lab["lcfs"], lab["lcfs_n"], lab["raxis"],
                               lab["zaxis"], gR, gZ)
    n = len(pa)
    coef = np.zeros((n, 8))
    rel = np.full(n, np.nan)
    for i in range(n):
        I = lab["Inode"][i].ravel().astype(np.float64)
        m = I != 0
        if m.sum() < 10:
            coef[i] = [1, 0, -1, 0, 0, 0, 0, 0]
            continue
        x = np.clip((pa[i] - lab["psi"][i].ravel()[m]) / (pa[i] - pb[i]), 0, 1)
        P = np.stack([x ** k for k in range(4)], 1)
        A = np.concatenate([P * (Rf[m] / R0_D3D)[:, None], P * (R0_D3D / Rf[m])[:, None]], 1)
        sol = np.linalg.lstsq(A, I[m], rcond=None)[0]
        s = sol / max(abs(sol).max(), 1e-30)
        coef[i] = s
        rel[i] = np.linalg.norm(A @ sol - I[m]) / np.linalg.norm(I[m])
    return coef, rel, pa, pb


# ============================================================================ Thomson (DIII-D)
TS_LEVELS = np.linspace(0.1, 0.9, 5)       # fractions of the overlapping Te range


def _prep_chord(x, te, med):
    """Finite positive samples sorted by position x, median-filtered (size med)."""
    from scipy.ndimage import median_filter
    ok = np.isfinite(x) & np.isfinite(te) & (te > 0)
    if ok.sum() < 3:
        return None
    o = np.argsort(x[ok], kind="stable")
    xs, ts = x[ok][o], te[ok][o]
    if med > 1 and len(ts) >= med:
        ts = median_filter(ts, size=med, mode="nearest")
    return xs, ts


def te_pairs_d3d(vR, vZ, vte, tR, tZ, tte, levels=TS_LEVELS, tmin=50.0, n_in=2):
    """Equal-Te point pairs (P,2,2) [pair, side, (R,Z)] on the DIII-D Thomson chords.

    Vertical core chord (R ~ 1.94 m, Z 0.03-0.83 m): monotone (cummin) branch walking up from
    its Te peak.  Tangential chord (Z ~ -0.06 m, R 1.67-2.06 m): branch walking out from its Te
    peak and, if channels lie inboard of the peak, the inboard branch.  Te is a flux function,
    so equal-Te points on the two outboard branches share psi (vertical-tangential pairs at
    `levels` of the overlapping Te range) and so do inboard/outboard tangential points (up to
    n_in pairs).  Rows are NaN where unavailable."""
    P = len(levels) + n_in
    out = np.full((P, 2, 2), np.nan)
    v = _prep_chord(vZ, vte, 5)
    t = _prep_chord(tR, tte, 3)
    if v is None or t is None or len(v[0]) < 10 or len(t[0]) < 4:
        return out
    Rv = float(np.nanmedian(vR))
    zv, tv = v
    rt, ttv = t
    okt = np.isfinite(tR) & np.isfinite(tZ)
    ot = np.argsort(tR[okt], kind="stable")
    zt_of = lambda r: np.interp(r, tR[okt][ot], tZ[okt][ot])  # noqa: E731
    pv = int(np.argmax(tv))
    xv, ev = zv[pv:], np.minimum.accumulate(tv[pv:])
    pk = int(np.argmax(ttv))
    xo, eo = rt[pk:], np.minimum.accumulate(ttv[pk:])
    if ev[0] < tmin or eo[0] < tmin:
        return out
    lo, hi = max(ev[-1], eo[-1], tmin), min(ev[0], eo[0])
    if hi > lo * 1.05 and len(xv) >= 3 and len(xo) >= 2:
        for k, f in enumerate(levels):
            T = lo + f * (hi - lo)
            z = np.interp(-T, -ev, xv)          # -ev non-decreasing along xv
            r = np.interp(-T, -eo, xo)
            out[k] = [[Rv, z], [r, zt_of(r)]]
    if n_in and pk >= 1:
        xi, ei = rt[:pk + 1][::-1], np.minimum.accumulate(ttv[:pk + 1][::-1])
        lo2, hi2 = max(ei[-1], eo[-1], tmin), min(ei[0], eo[0])
        if hi2 > lo2 * 1.05:
            for k, f in enumerate(np.linspace(0.25, 0.75, n_in)):
                T = lo2 + f * (hi2 - lo2)
                ri = np.interp(-T, -ei, xi)
                ro = np.interp(-T, -eo, xo)
                out[len(levels) + k] = [[ri, zt_of(ri)], [ro, zt_of(ro)]]
    return out


def all_pairs(inp, **kw):
    return np.stack([te_pairs_d3d(inp["vR"][i], inp["vZ"][i], inp["vte"][i], inp["tR"][i],
                                  inp["tZ"][i], inp["tte"][i], **kw) for i in range(len(inp["Ip"]))])


MM_LEVELS = np.linspace(0.05, 0.85, 17)


def _psin_branch(x, pn, v, med):
    """Branch of a chord walking away from its psiN minimum: (psiN cummax, values)."""
    from scipy.ndimage import median_filter
    ok = np.isfinite(x) & np.isfinite(pn) & np.isfinite(v) & (v > 0)
    if ok.sum() < 3:
        return None
    o = np.argsort(x[ok], kind="stable")
    x, pn, v = x[ok][o], pn[ok][o], v[ok][o]
    if med > 1 and len(v) >= med:
        v = median_filter(v, size=med, mode="nearest")
    k = int(np.argmin(pn))
    p, vv = np.maximum.accumulate(pn[k:]), v[k:]
    keep = p < 1.0
    p, vv = p[keep], vv[keep]
    if len(p) < 2:
        return None
    u, ix = np.unique(p, return_index=True)
    return u, vv[ix]


def ts_mismatch(vZ, pnv, vv, tR, pnt, tv, levels=MM_LEVELS, min_levels=3):
    """DIII-D Thomson consistency of an equilibrium (label-free; analogue of symmetry_mm).

    Profiles mapped to psiN: vertical chord branch above its psiN minimum vs tangential chord
    branch outboard of its psiN minimum; mean |V_vert(g) - V_tang(g)| / p95(V) over psiN
    levels g covered by both branches.  NaN if fewer than min_levels levels are covered."""
    a = _psin_branch(vZ, pnv, vv, 5)
    b = _psin_branch(tR, pnt, tv, 3)
    if a is None or b is None:
        return np.nan
    g = levels[(levels >= max(a[0][0], b[0][0])) & (levels <= min(a[0][-1], b[0][-1]))]
    if len(g) < min_levels:
        return np.nan
    sc = np.nanpercentile(np.r_[vv[np.isfinite(vv)], tv[np.isfinite(tv)]], 95)
    if not sc > 0:
        return np.nan
    return float(np.mean(np.abs(np.interp(g, *a) - np.interp(g, *b))) / sc)


def points_psi(psi, gR, gZ, pts):
    """Bilinear psi (n, Np) at points pts (n, Np, 2) (NaN points -> NaN); numpy, CPU."""
    n, nz, nr = psi.shape
    dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
    bad = ~np.isfinite(pts).all(-1)
    p = np.where(bad[..., None], [[gR[nr // 2], gZ[nz // 2]]], pts)
    fr, fz = (p[..., 0] - gR[0]) / dR, (p[..., 1] - gZ[0]) / dZ
    j = np.clip(np.floor(fr).astype(int), 0, nr - 2)
    i = np.clip(np.floor(fz).astype(int), 0, nz - 2)
    a, b = fz - i, fr - j
    bi = np.arange(n)[:, None]
    v = (psi[bi, i, j] * (1 - a) * (1 - b) + psi[bi, i + 1, j] * a * (1 - b) +
         psi[bi, i, j + 1] * (1 - a) * b + psi[bi, i + 1, j + 1] * a * b)
    return np.where(bad, np.nan, v)


def mismatch_frames(inp, psi, psi_ax, psi_b, which="te"):
    """Per-frame Thomson consistency (Te or ne) of maps psi (n,nz,nr) with bounds."""
    gR, gZ = inp["gR"], inp["gZ"]
    n = len(psi)
    vp = np.stack([np.c_[inp["vR"][i], inp["vZ"][i]] for i in range(n)])
    tp = np.stack([np.c_[inp["tR"][i], inp["tZ"][i]] for i in range(n)])
    den = (psi_ax - psi_b)[:, None]
    with np.errstate(all="ignore"):
        pnv = (psi_ax[:, None] - points_psi(psi, gR, gZ, vp)) / den
        pnt = (psi_ax[:, None] - points_psi(psi, gR, gZ, tp)) / den
    vk, tk = ("vte", "tte") if which == "te" else ("vne", "tne")
    out = np.full(n, np.nan)
    for i in range(n):
        if not (np.isfinite(den[i, 0]) and den[i, 0] > 0 and psi_b[i] > -1e20):
            continue
        out[i] = ts_mismatch(inp["vZ"][i], pnv[i], inp[vk][i], inp["tR"][i], pnt[i], inp[tk][i])
    return out


# ============================================================================ kinetic fit
def _pt_index(S, pts):
    """Bilinear node indices / weights (..., 4) of points (..., 2) on the solver grid."""
    gR, gZ = S.spec.gR, S.spec.gZ
    fr = (pts[..., 0] - gR[0]) / S.dR
    fz = (pts[..., 1] - gZ[0]) / S.dZ
    j = np.clip(np.floor(fr).astype(int), 0, S.nr - 2)
    i = np.clip(np.floor(fz).astype(int), 0, S.nz - 2)
    wr, wz = fr - j, fz - i
    idx = np.stack([i * S.nr + j, i * S.nr + j + 1, (i + 1) * S.nr + j, (i + 1) * S.nr + j + 1], -1)
    w = np.stack([(1 - wr) * (1 - wz), wr * (1 - wz), (1 - wr) * wz, wr * wz], -1)
    return idx, w


def kfit_d3d(S, Iext, Ip, Zt, pairs, n_iter=120, omega=0.3, kc=0.05, I0=None, c0=None,
             prior=None, w_prior=(0.3, 0.3, 0.3, 0.3, 1.0), lm=5.0, pair_tol=0.02, dv_max=2.0,
             fit_dv=True, nbasis=4):
    """fec.mast_select_v2.kfit2 (via the V3 boundary, so limiter-aware) with equal-Te pairs at
    arbitrary (R,Z) points: pairs (B,P,2,2) from te_pairs_d3d; NaN rows ignored.  Same model:
    4 non-hollow basis currents scaled to carry Ip + uniform vertical-field nuisance dv;
    LM damping, priors, positivity, pair rows from the vertically re-centred basis.
    nbasis=6: + R/R0 x(1-x), R0/R x(1-x) (fec.mast_select_v3.kfit_basis_args)."""
    import torch
    from fec.mast_select_v2 import KF_DV, KF_PRIOR
    from fec.mast_select_v3 import kfit_basis_args, kfit_basis_stack
    prior = KF_PRIOR if prior is None else prior
    nb = int(nbasis)
    prior, w_prior = kfit_basis_args(nb, prior, w_prior)
    K = nb + 1
    dev, dt = S.dev, S.dt
    t = lambda x: torch.as_tensor(x, device=dev, dtype=dt)  # noqa: E731
    Iext, Ip, Zt = t(Iext), t(Ip), t(Zt)
    B = Iext.shape[0]
    nz, nr = S.nz, S.nr
    Npix = nz * nr
    psi_ext = (Iext @ S.Mext.T).reshape(B, nz, nr)
    Rs, Zs = S.Rs, S.Zs
    R0 = S.spec.R0
    pairs = np.asarray(pairs, float)
    pvalid = t(np.isfinite(pairs).all((-1, -2)).astype(np.float32))           # (B,P)
    pts = np.where(np.isfinite(pairs), pairs, np.array([R0, 0.0]))
    idx_np, w_np = _pt_index(S, pts)                                           # (B,P,2,4)
    idx = torch.as_tensor(idx_np, device=dev)
    w = t(w_np)
    Gpt = None
    for q in range(4):
        term = S.G[idx[..., q]] * w[..., q, None]                              # (B,P,2,Ns)
        Gpt = term if Gpt is None else Gpt + term
    Gd = Gpt[:, :, 0] - Gpt[:, :, 1]
    del Gpt, term
    pr = t(pts[..., 0])
    r2d = (pr[..., 0] ** 2 - pr[..., 1] ** 2) / 2 * KF_DV

    def ptval(field, side):
        f = field.reshape(B, -1)
        ii = idx[:, :, side].reshape(B, -1)
        return (f.gather(1, ii).reshape(idx[:, :, side].shape) * w[:, :, side]).sum(-1)

    if I0 is None:
        g = torch.exp(-((Rs[None] - R0) ** 2 + (Zs[None] - Zt[:, None]) ** 2) / 0.05)
        I = Ip[:, None] * g / g.sum(1, keepdim=True)
    else:
        I = t(I0)
    c = t(c0) if c0 is not None else torch.zeros(B, device=dev, dtype=dt)
    pu = t(prior)
    u = pu.expand(B, K).clone()
    if not fit_dv:
        w_prior = tuple(w_prior[:nb]) + (1e4,)
    pw = t(w_prior)
    fb = (S.RR ** 2)[None] * (S.ZZ[None] - Zt[:, None, None])
    vfield = (S.RR ** 2 / 2 * KF_DV)[None]
    prev_mask = None
    dI = torch.zeros(B, device=dev, dtype=dt)
    for it in range(n_iter):
        psi = psi_ext + (I @ S.G.T).reshape(B, nz, nr) + c[:, None, None] * fb + \
            u[:, nb, None, None] * vfield
        psi_ax, Rax, Zax = S.find_axis(psi)
        xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
        psi_b, _ = S.boundary(psi, psi_ax, Rax, Zax, xp)
        seedm = prev_mask if (it % 10 != 0) else None
        mask, psiN = S.core_mask(psi, psi_ax, psi_b, Rax, Zax, xp, seed_mask=seedm)
        prev_mask = mask
        pn = psiN.reshape(B, -1)[:, S.sidx].clamp(0, 1)
        ms = mask.reshape(B, -1)[:, S.sidx].to(dt)
        basis = kfit_basis_stack(Rs, R0, pn, ms, nb)
        basis = basis / basis.sum(2, keepdim=True).clamp_min(1e-12) * Ip[:, None, None]
        mix = torch.einsum("bk,bkn->bn", u[:, :nb], basis)
        delta = Zt - (mix * Zs[None]).sum(1) / Ip
        full = torch.zeros(B * nb, Npix, device=dev, dtype=dt)
        full[:, S.sidx] = basis.reshape(B * nb, -1)
        full = S._shiftZ(full.reshape(B * nb, nz, nr), delta.repeat_interleave(nb))
        basis = full.reshape(B, nb, Npix)[:, :, S.sidx]
        phi = (psi_ax - psi_b).abs().clamp_min(1e-4)
        ext = psi_ext + c[:, None, None] * fb
        A_p = torch.cat([torch.einsum("bpn,bkn->bpk", Gd, basis), r2d[..., None]], 2)
        b_p = -(ptval(ext, 0) - ptval(ext, 1))
        sc = (pvalid / (phi[:, None] * pair_tol))[..., None]
        A_ip = torch.cat([torch.ones(B, 1, nb, device=dev, dtype=dt),
                          torch.zeros(B, 1, 1, device=dev, dtype=dt)], 2)
        A = torch.cat([A_ip * 100.0, A_p * sc, torch.diag_embed(pw).expand(B, K, K),
                       lm * torch.eye(K, device=dev, dtype=dt).expand(B, K, K)], 1)
        bb = torch.cat([torch.full((B, 1), 100.0, device=dev, dtype=dt), b_p * sc[..., 0],
                        (pw * pu).expand(B, K), lm * u], 1)
        A64 = torch.nan_to_num(A.double())
        b64 = torch.nan_to_num(bb.double())
        AtA = A64.transpose(1, 2) @ A64 + 1e-9 * torch.eye(K, device=dev, dtype=torch.float64)
        unew = torch.linalg.solve(AtA, (A64.transpose(1, 2) @ b64[..., None])[..., 0]).to(dt)
        bad = ~torch.isfinite(unew).all(1)
        unew = torch.where(bad[:, None], u, unew)
        cu = unew[:, :nb].clamp_min(0)
        cu = cu / cu.sum(1, keepdim=True).clamp_min(1e-6)
        unew = torch.cat([cu, unew[:, nb:].clamp(-dv_max, dv_max)], 1)
        u = (1 - omega) * u + omega * unew
        Inew = torch.einsum("bk,bkn->bn", u[:, :nb], basis)
        Inew = Ip[:, None] * Inew / Inew.sum(1, keepdim=True).clamp_min(1e-30)
        Zc = (Inew * Zs[None]).sum(1) / Ip
        c = c + kc * (Zt - Zc)
        dI = (Inew - I).abs().sum(1) / Ip
        I = (1 - omega) * I + omega * Inew
    psi = psi_ext + (I @ S.G.T).reshape(B, nz, nr) + c[:, None, None] * fb + \
        u[:, nb, None, None] * vfield
    psi_ax, Rax, Zax = S.find_axis(psi)
    xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
    psi_b, limited = S.boundary(psi, psi_ax, Rax, Zax, xp, final=True)
    res = ((ptval(psi, 0) - ptval(psi, 1)) / (psi_ax - psi_b).abs()[:, None]) * pvalid
    return dict(psi=psi, I=I, c=c, u=u, psi_ax=psi_ax, Rax=Rax, Zax=Zax, psi_b=psi_b, dI=dI,
                pair_res=res, dv=u[:, nb] * KF_DV, limited=limited)


# ============================================================================ kinetic profile
QE = 1.602176634e-19
KP_NQ = 21          # psiN table nodes for Te(psiN), ne(psiN)
KP_NS = 16          # flux shells for the FF' solve


def trapped_fraction(eps):
    """Sauter et al. 1999 effective trapped fraction fit (circular approximation)."""
    eps = eps.clamp(0, 0.95)
    return 1 - (1 - eps) ** 2 / (torch_sqrt(1 - eps ** 2) * (1 + 1.46 * torch_sqrt(eps)))


def torch_sqrt(x):
    return x.clamp_min(0) ** 0.5


def kp_chords(inp, sl=slice(None)):
    """Pre-smoothed Thomson samples for the kinetic profile: points (B,N,2), Te [eV],
    ne [m^-3], valid (B,N) from both DIII-D chords (median filter along each chord)."""
    from scipy.ndimage import median_filter
    vR, vZ, vte, vne = (inp[k][sl] for k in ("vR", "vZ", "vte", "vne"))
    tR, tZ, tte, tne = (inp[k][sl] for k in ("tR", "tZ", "tte", "tne"))
    B = len(vR)
    pts = np.concatenate([np.stack([vR, vZ], -1), np.stack([tR, tZ], -1)], 1).astype(np.float64)
    te = np.full(pts.shape[:2], np.nan)
    ne = np.full(pts.shape[:2], np.nan)
    nv = vR.shape[1]
    for i in range(B):
        for (x, T, N, off, med) in ((vZ[i], vte[i], vne[i], 0, 5), (tR[i], tte[i], tne[i], nv, 3)):
            ok = np.flatnonzero(np.isfinite(x) & np.isfinite(T) & np.isfinite(N) & (T > 0) & (N > 0))
            if len(ok) < 3:
                continue
            o = ok[np.argsort(x[ok], kind="stable")]
            te[i, off + o] = median_filter(T[o].astype(np.float64), size=min(med, len(o)), mode="nearest")
            ne[i, off + o] = median_filter(N[o].astype(np.float64), size=min(med, len(o)), mode="nearest")
    valid = np.isfinite(te) & np.isfinite(ne) & np.isfinite(pts).all(-1)
    return dict(pts=np.where(np.isfinite(pts), pts, np.array([R0_D3D, 0.0])), te=np.nan_to_num(te),
                ne=np.nan_to_num(ne), valid=valid)


def _ffill_tables(s, cnt):
    """Bin means (B,NQ) with empty bins filled from the nearest inner bin (outer edge) or the
    first filled bin (axis side); returns (table, any_valid)."""
    import torch
    B, NQ = s.shape
    ar = torch.arange(NQ, device=s.device).expand(B, NQ)
    vq = cnt > 0
    last = torch.cummax(torch.where(vq, ar, torch.full_like(ar, -1)), 1).values
    first = torch.argmax(vq.to(torch.int8), 1, keepdim=True).expand(B, NQ)
    idx = torch.where(last < 0, first, last)
    mean = s / cnt.clamp_min(1)
    return mean.gather(1, idx), vq.any(1)


def kp_chord_tensors(S, ts):
    import torch
    idx_np, w_np = _pt_index(S, ts["pts"])
    t = lambda x: torch.as_tensor(x, device=S.dev, dtype=S.dt)  # noqa: E731
    return dict(cidx=torch.as_tensor(idx_np, device=S.dev), cw=t(w_np), te=t(ts["te"]),
                ne=t(ts["ne"] * 1e-19), valid=torch.as_tensor(ts["valid"], device=S.dev))


def kp_current(S, pn, ms, psiN_flat, psi_ax, psi_b, ch, alpha=1.0, K=K_BETA, texp=1.5,
               neo=True, c_bs=0.0, fallback=(0.05, 2.0, 2.0), anchor=True):
    """Kinetic-profile pieces on an equilibrium (see solve_kinetic); kp_assemble -> currents.

    pn/ms: psiN (clamped) and core mask on the solver source nodes; psiN_flat (B,Npix)
    full-grid psiN used to map the Thomson samples; ch from kp_chord_tensors."""
    import torch
    dev, dt = S.dev, S.dt
    B = pn.shape[0]
    Rs, R0 = S.Rs, S.spec.R0
    cidx, cw = ch["cidx"], ch["cw"]
    pc = (psiN_flat.gather(1, cidx.reshape(B, -1)).reshape(cidx.shape) * cw).sum(-1)   # (B,N)
    ok = ch["valid"] & torch.isfinite(pc) & (pc >= 0) & (pc < 1)
    # cloud-in-cell binning: each sample splits its weight between the two nearest nodes
    fq = torch.nan_to_num(pc).clamp(0, 1) * (KP_NQ - 1)
    q0 = fq.floor().long().clamp(0, KP_NQ - 2)
    f1 = (fq - q0.to(dt)) * ok.to(dt)
    f0 = ok.to(dt) - f1
    z0 = torch.zeros(B, KP_NQ, device=dev, dtype=dt)

    def cic(v):
        return z0.scatter_add(1, q0, v * f0).scatter_add(1, q0 + 1, v * f1)
    cnt = cic(torch.ones_like(pc))
    Tq, has = _ffill_tables(cic(ch["te"]), cnt)
    Nq, _ = _ffill_tables(cic(ch["ne"]), cnt)
    has = has & (ok.sum(1) >= 5)
    if anchor:
        # axis value = measured chord peak (the tangential chord crosses the axis region);
        # bins inside the first sampled bin are interpolated linearly from it (not
        # back-filled), which removes the position <-> pressure mapping runaway
        vq = cnt > 0
        first = torch.argmax(vq.to(torch.int8), 1)
        xq = torch.arange(KP_NQ, device=dev, dtype=dt)[None]
        fpos = first[:, None].to(dt).clamp_min(1)
        for tab, raw in ((Tq, ch["te"]), (Nq, ch["ne"])):
            peak = torch.where(ch["valid"], raw, torch.zeros_like(raw)).max(1).values
            v1 = tab.gather(1, first[:, None])
            lin = peak[:, None] + (v1 - peak[:, None]) * (xq / fpos)
            tab.copy_(torch.where(xq < first[:, None].to(dt), lin, tab))
    Tq = torch.cummin(Tq, 1).values.clamp_min(1.0)
    Nq = torch.cummin(Nq, 1).values.clamp_min(1e-3)
    pq = K * QE * Tq * Nq * 1e19                                                # Pa
    h = 1.0 / (KP_NQ - 1)
    dpq = torch.zeros_like(pq)
    dpq[:, 1:-1] = (pq[:, 2:] - pq[:, :-2]) / (2 * h)
    dpq[:, 0] = (pq[:, 1] - pq[:, 0]) / h
    dpq[:, -1] = (pq[:, -1] - pq[:, -2]) / h
    dpq = torch.cat([dpq[:, :1], (dpq[:, :-2] + dpq[:, 1:-1] + dpq[:, 2:]) / 3, dpq[:, -1:]], 1)
    dpq = dpq.clamp_max(0)                                                      # dp/dpsiN <= 0

    def lerp(tab, x):
        fx = x * (KP_NQ - 1)
        i0 = fx.floor().long().clamp(0, KP_NQ - 2)
        f = fx - i0.to(dt)
        return tab.gather(1, i0) * (1 - f) + tab.gather(1, i0 + 1) * f
    phi = (psi_ax - psi_b).clamp_min(1e-6)[:, None]
    Jp = alpha * Rs[None] * (-lerp(dpq, pn) / phi) * ms
    sb = (pn * KP_NS).floor().long().clamp(0, KP_NS - 1)
    z = torch.zeros(B, KP_NS, device=dev, dtype=dt)
    nb = z.scatter_add(1, sb, ms)
    sJp = z.scatter_add(1, sb, Jp)
    sinvR = z.scatter_add(1, sb, ms / Rs[None])
    RsB = Rs[None].expand(B, -1)
    Rmin = torch.full_like(z, 1e3).scatter_reduce(1, sb, torch.where(ms > 0, RsB, RsB * 0 + 1e3),
                                                 reduce="amin", include_self=True)
    Rmax = torch.zeros_like(z).scatter_reduce(1, sb, torch.where(ms > 0, RsB, RsB * 0 - 1e3),
                                              reduce="amax", include_self=True)
    eps = torch.where(nb > 0, (Rmax - Rmin) / (Rmax + Rmin).clamp_min(1e-6), torch.zeros_like(z))
    ft = trapped_fraction(eps)
    xs = ((torch.arange(KP_NS, device=dev, dtype=dt) + 0.5) / KP_NS).expand(B, -1)
    sig = lerp(Tq, xs) ** texp
    if neo:
        sig = sig * (1 - 1.36 * ft + 0.59 * ft ** 2 - 0.23 * ft ** 3).clamp_min(0.02)
    jbs = torch.zeros_like(z)
    L31 = 1.7 * ft - 0.95 * ft ** 2 + 0.15 * ft ** 3 + 0.1 * ft ** 4
    jbs1 = R0 * L31 * (-lerp(dpq, xs) / phi)                 # bootstrap per unit c_bs [A/m^2]
    if c_bs:
        jbs = alpha * c_bs * jbs1
    return dict(Jp=Jp, sb=sb, nb=nb, sJp=sJp, sinvR=sinvR, sig=sig, jbs=jbs, jbs1=jbs1,
                tot_sig=(sig * nb).sum(1).clamp_min(1e-30), has=has, Tq=Tq, pq=pq, eps=eps,
                fallback=fallback)


def kp_assemble(S, k, Ip, pn, ms):
    """Node currents (B,Ns) [A] from kp_current: lam sets the total Ip, FF' per shell."""
    import torch
    Rs, R0 = S.Rs, S.spec.R0
    dA = S.dR * S.dZ
    nb, sig, jbs = k["nb"], k["sig"], k["jbs"]
    lam = ((Ip / dA - (jbs * nb).sum(1)) / k["tot_sig"]).clamp_min(0.05 * Ip / dA / k["tot_sig"])
    g = torch.where(nb > 0, ((lam[:, None] * sig + jbs) * nb - k["sJp"]) /
                    k["sinvR"].clamp_min(1e-30), torch.zeros_like(nb))
    J = k["Jp"] + g.gather(1, k["sb"]) / Rs[None] * ms
    b0, am, an = k["fallback"]
    Jf = (b0 * Rs[None] / R0 + (1 - b0) * R0 / Rs[None]) * (1 - pn ** am).clamp_min(0) ** an * ms
    J = torch.where(k["has"][:, None], J, Jf)
    return Ip[:, None] * J / J.sum(1, keepdim=True).clamp_min(1e-30), lam


def solve_kinetic(S, Iext, Ip, Zt, ts, n_iter=120, omega=0.3, kc=0.05, I0=None, c0=None,
                  K=K_BETA, texp=1.5, neo=True, c_bs=0.0, ramp=30, fallback=(0.05, 2.0, 2.0),
                  anchor=True):
    """Kinetic-profile free-boundary equilibrium (coil currents fixed, Picard as fec.gs).

    Each iteration maps the Thomson samples onto the current psiN (bilinear), bins them into
    Te(psiN), ne(psiN) tables (KP_NQ nodes, empty bins filled, monotonised outward) and builds
      p = K e ne Te,   J_p = R p'(psi)      (p' = -dp/dpsiN / (psi_ax - psi_b) > 0),
      target shell-mean current density  j_t = lam * sigma(psiN) + j_bs(psiN),
      sigma = Te^texp * [neo: Sauter collisionless sigma_neo/sigma_Sp(f_t(eps))],
      j_bs = c_bs * R0 * L31(f_t) * p'(psi)    (Sauter L31 pressure term; c_bs = 0 off),
      eps per shell = (Rmax - Rmin)/(Rmax + Rmin) of its nodes,
    and FF' per shell (g/R, g = FF'/mu0) so that the shell current equals j_t x shell area;
    lam makes the total Ip.  J_p (and j_bs) are ramped in over `ramp` iterations.  Frames
    without usable Thomson inside psiN < 1 use the FreeGS profile `fallback`.  Vertical
    re-centring at Zt with radial-field feedback exactly as Solver.solve."""
    import torch
    dev, dt = S.dev, S.dt
    t = lambda x: torch.as_tensor(x, device=dev, dtype=dt)  # noqa: E731
    Iext, Ip, Zt = t(Iext), t(Ip), t(Zt)
    B = Iext.shape[0]
    nz, nr = S.nz, S.nr
    Npix = nz * nr
    psi_ext = (Iext @ S.Mext.T).reshape(B, nz, nr)
    Rs, Zs = S.Rs, S.Zs
    R0 = S.spec.R0
    ch = kp_chord_tensors(S, ts)
    if I0 is None:
        g0 = torch.exp(-((Rs[None] - R0) ** 2 + (Zs[None] - Zt[:, None]) ** 2) / 0.05)
        I = Ip[:, None] * g0 / g0.sum(1, keepdim=True)
    else:
        I = t(I0)
    c = t(c0) if c0 is not None else torch.zeros(B, device=dev, dtype=dt)
    fb = (S.RR ** 2)[None] * (S.ZZ[None] - Zt[:, None, None])
    prev_mask = None
    dI = torch.zeros(B, device=dev, dtype=dt)
    k = None
    for it in range(n_iter):
        psi = psi_ext + (I @ S.G.T).reshape(B, nz, nr) + c[:, None, None] * fb
        psi_ax, Rax, Zax = S.find_axis(psi)
        xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
        psi_b, _ = S.boundary(psi, psi_ax, Rax, Zax, xp)
        seedm = prev_mask if (it % 10 != 0) else None
        mask, psiN = S.core_mask(psi, psi_ax, psi_b, Rax, Zax, xp, seed_mask=seedm)
        prev_mask = mask
        pn = psiN.reshape(B, -1)[:, S.sidx].clamp(0, 1)
        ms = mask.reshape(B, -1)[:, S.sidx].to(dt)
        alpha = min(1.0, (it + 1) / ramp) if ramp else 1.0
        k = kp_current(S, pn, ms, psiN.reshape(B, -1), psi_ax, psi_b, ch, alpha=alpha, K=K,
                       texp=texp, neo=neo, c_bs=c_bs, fallback=fallback, anchor=anchor)
        Inew, lam = kp_assemble(S, k, Ip, pn, ms)
        Zc_new = (Inew * Zs[None]).sum(1) / Ip
        delta = Zt - Zc_new
        full = torch.zeros(B, Npix, device=dev, dtype=dt)
        full[:, S.sidx] = Inew
        full = S._shiftZ(full.reshape(B, nz, nr), delta).reshape(B, Npix)
        Inew = full[:, S.sidx]
        Inew = Ip[:, None] * Inew / Inew.sum(1, keepdim=True).clamp_min(1e-30)
        c = c + kc * delta
        dI = (Inew - I).abs().sum(1) / Ip
        I = (1 - omega) * I + omega * Inew
    psi = psi_ext + (I @ S.G.T).reshape(B, nz, nr) + c[:, None, None] * fb
    psi_ax, Rax, Zax = S.find_axis(psi)
    xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
    psi_b, limited = S.boundary(psi, psi_ax, Rax, Zax, xp, final=True)
    dA = S.dR * S.dZ
    return dict(psi=psi, I=I, c=c, psi_ax=psi_ax, Rax=Rax, Zax=Zax, psi_b=psi_b, dI=dI,
                limited=limited, has=k["has"], pp_frac=k["sJp"].sum(1) * dA / Ip,
                bs_frac=(k["jbs"] * k["nb"]).sum(1) * dA / Ip)


PHASE_FLAT = 0.9    # flat-top: measured Ip >= 0.9 x the shot's max (frames of whole shots)


def phase_of(inp):
    """'ramp_up' / 'flat' / 'ramp_down' per frame from the measured Ip trace of its shot:
    flat-top spans the first..last frame with Ip >= PHASE_FLAT x shot max; frames before /
    after it are ramp-up / ramp-down.  (dIp/dt over +-5 ms is noise-dominated on DIII-D.)"""
    ip = inp["cur_c"][:, 20, OFF0].astype(np.float64)
    shot = np.asarray(inp["shot"])
    ph = np.full(len(ip), "flat", dtype=object)
    for s in np.unique(shot):
        ix = np.flatnonzero(shot == s)
        ix = ix[np.argsort(inp["t_ms"][ix], kind="stable")] if "t_ms" in inp else ix
        top = np.flatnonzero(ip[ix] >= PHASE_FLAT * ip[ix].max())
        ph[ix[:top[0]]] = "ramp_up"
        ph[ix[top[-1] + 1:]] = "ramp_down"
    return ph


KP_CHECK = [("FX default (0.05,2,2)", None),
            ("KP Spitzer Te^1.5", dict(neo=False, c_bs=0.0)),
            ("KP neo", dict(neo=True, c_bs=0.0))] + \
    [(f"KP neo + bs c={c:g}", dict(neo=True, c_bs=c)) for c in (0.5, 1.0, 1.5, 2.0, 3.0)]


def kp_check(set_name="cal", chunk=1024):
    """Profile models evaluated at the TRUE (label) geometry: node currents built on the
    label psiN with the same Ip as EFIT, compared with the label node currents.  Reports the
    flux error G (I_model - I_label) as an R2psi ceiling and the relative current error, per
    phase.  CPU; on TRAIN (cal) this calibrates c_bs, on VAL it is a diagnostic."""
    import torch
    from fec.mast_select_v2 import FFZ_PROF
    inp = load_set(set_name, labels=True)
    gR, gZ = inp["gR"], inp["gZ"]
    S = make_bare_solver(device="cpu")
    n = len(inp["Ip"])
    pa, pb = label_flux_bounds(inp["lab_psi"], inp["lab_lcfs"], inp["lab_lcfs_n"],
                               inp["lab_raxis"], inp["lab_zaxis"], gR, gZ)
    ts = kp_chords(inp)
    y = inp["lab_psi"].reshape(n, -1).astype(np.float64)
    sst = float(((y - y.mean()) ** 2).sum())
    sidx = S.sidx.numpy()
    Gn = S.G.double()
    ph = phase_of(inp)
    res = {name: dict(ss=np.zeros(n), rel=np.zeros(n)) for name, _ in KP_CHECK}
    t0 = time.time()
    for a in range(0, n, chunk):
        sl = slice(a, min(n, a + chunk))
        B = sl.stop - sl.start
        Il = torch.as_tensor(inp["lab_Inode"][sl].reshape(B, -1)[:, sidx], dtype=torch.float32)
        ip = Il.sum(1)
        psi = torch.as_tensor(inp["lab_psi"][sl].reshape(B, -1), dtype=torch.float32)
        pat, pbt = torch.as_tensor(pa[sl], dtype=torch.float32), torch.as_tensor(pb[sl], dtype=torch.float32)
        psiN = (pat[:, None] - psi) / (pat - pbt)[:, None]
        pn = psiN[:, sidx].clamp(0, 1)
        ms = (Il != 0).to(torch.float32)
        ch = kp_chord_tensors(S, {k: v[sl] for k, v in ts.items()})
        for name, cfg in KP_CHECK:
            if cfg is None:
                b0, am, an = FFZ_PROF
                Rs, R0 = S.Rs, S.spec.R0
                J = (b0 * Rs[None] / R0 + (1 - b0) * R0 / Rs[None]) * \
                    (1 - pn ** am).clamp_min(0) ** an * ms
                Im = ip[:, None] * J / J.sum(1, keepdim=True)
            else:
                k = kp_current(S, pn, ms, psiN, pat, pbt, ch, **cfg)
                Im, _ = kp_assemble(S, k, ip, pn, ms)
            dI = (Im - Il).double()
            dpsi = dI @ Gn.T
            res[name]["ss"][sl] = (dpsi ** 2).sum(1).numpy()
            res[name]["rel"][sl] = (dI.norm(dim=1) / Il.double().norm(dim=1)).numpy()
        print(f"kp_check {sl.stop}/{n} {time.time() - t0:.0f}s", flush=True)
    rep = {"set": set_name, "n": n, "ss_tot": sst, "phases": {}}
    for p in ("all", "ramp_up", "flat", "ramp_down"):
        m = np.ones(n, bool) if p == "all" else (ph == p)
        sstp = float(((y[m] - y.mean()) ** 2).sum())
        rep["phases"][p] = {"n": int(m.sum()), **{
            name: dict(r2_ceiling=1 - float(r["ss"][m].sum()) / sstp,
                       rel_current_err_median=float(np.median(r["rel"][m])))
            for name, r in res.items()}}
    (OUT / f"kp_check_{set_name}.json").write_text(json.dumps(rep, indent=1))
    for p, d in rep["phases"].items():
        print(p, d["n"], " | ".join(f"{k}: {v['r2_ceiling']:.5f}/{v['rel_current_err_median']:.3f}"
                                    for k, v in d.items() if k != "n"))


# ============================================================================ runs
PRED = OUT / "pred"
CONV_TOL = 1e-2
META_KEYS = ("psi_ax", "psi_b", "Rax", "Zax", "c", "dI")
GRID_KEYS = ("psi_ax", "psi_b", "Rax", "Zax", "dI")


def label_centroid_z(lab, gZ):
    """Oracle height: Z of the label current centroid (the solver's Zt definition)."""
    I = lab["Inode"].astype(np.float64)
    tot = I.sum((1, 2))
    return (I.sum(2) * np.asarray(gZ)[None]).sum(1) / np.where(tot != 0, tot, 1.0)


class Store:
    """Per-variant accumulator over whole-shot chunks -> PRED/<name>.npz (psi_c float32)."""

    def __init__(self, n):
        self.n = n
        self.v = {}

    def put(self, name, sl, psi, **meta):
        d = self.v.setdefault(name, {"psi": np.zeros((self.n, 65, 65), np.float32)})
        d["psi"][sl] = psi
        for k, x in meta.items():
            x = np.asarray(x)
            if k not in d:
                if x.dtype.kind == "f":
                    d[k] = np.full((self.n,) + x.shape[1:], np.nan)
                else:
                    d[k] = np.zeros((self.n,) + x.shape[1:], x.dtype)
            d[k][sl] = x

    def save(self, idx):
        PRED.mkdir(parents=True, exist_ok=True)
        for name, d in self.v.items():
            np.savez(PRED / f"{name}.npz", idx=idx, **d)
            print("saved", name, flush=True)


def _res(o):
    r = {k: o[k].detach().cpu().numpy() for k in META_KEYS + ("limited",) if k in o}
    r["psi"] = o["psi"].detach().cpu().numpy()
    return r


def _put(st, name, sl, r, **extra):
    st.put(name, sl, r["psi"], **{k: r[k] for k in META_KEYS + ("limited",) if k in r}, **extra)


def _subset(inp, n):
    N = len(inp["Ip"]) if "Ip" in inp else len(inp["psi"])
    return {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim and len(v) == N else v)
            for k, v in inp.items()}


def run_family(family, set_name="val", max_frames=1024, limit=0):
    """GPU families.  F (A1 field): fixed default profile, kfit_d3d and the 12-shape grid at
    the v2 force-free height, plus kfit/fixed at the oracle, zero and v3-robust heights.
    F0 (A0 field) / Flim (A1 + limiter): fixed + kfit at the force-free height.  LP / LP0 /
    LPlim: label-profile solves (ffz; LP also oracle, zero and robust heights).  All solves
    warm-start from the Zt = 0 solution o0 as in select_v2.  Saves PRED/<variant>.npz."""
    import torch
    from fec.mast_select_v2 import FFZ_PROF, V2_GRID, force_free_z, shot_chunks
    from fec.mast_select_v3 import robust_zt
    inp = _subset(load_set(set_name), limit or len(load_set(set_name)["Ip"]))
    n = len(inp["Ip"])
    shot = inp["shot"]
    if family.startswith("KP"):
        return run_kp(inp, set_name, family, max_frames)
    if family == "IPO":
        return run_ipo(inp, set_name, max_frames)
    field = "A0" if family.endswith("0") else "A1"
    lim = family.endswith("lim")
    S, curf = make_solver(field, limiter=lim)
    Iall = curf(inp)
    Ip = np.maximum(inp["Ip"], 1e5)            # measured Ip; floor 100 kA (input artefacts)
    st = Store(n)
    tag = field + ("_lim" if lim else "")
    lp = family.startswith("LP")
    full = family in ("LP", "F")
    if lp or full:
        lab = _subset(dict(np.load(OUT / f"lab_{set_name}.npz")), n)
        zor = label_centroid_z(lab, inp["gZ"])
    if lp:
        coef = fit_label_profile(lab, inp["gR"], inp["gZ"])[0]
    else:
        pairs = all_pairs(inp)
    grid = None
    if family == "F":
        grid = {"psi16": np.zeros((len(V2_GRID), n, 65, 65), np.float16),
                **{k: np.full((len(V2_GRID), n), np.nan) for k in GRID_KEYS}}
    P = "LP" if lp else "FX"
    t0 = time.time()
    for sl in shot_chunks(shot, max_frames):
        B = sl.stop - sl.start
        I, ip = Iall[sl], Ip[sl]
        prof = coef[sl] if lp else np.tile(FFZ_PROF, (B, 1))
        o0, Zs = force_free_z(S, I, ip, prof, Rinit=R0_D3D)
        clip = np.abs(Zs) >= 0.3 - 1e-9
        heights = {"ffz": Zs}
        if full:
            _put(st, f"{P}_{tag}_z0", sl, _res(o0), Zt=np.zeros(B))
            rz = robust_zt(S, I, ip, prof, shot[sl], temporal=True)
            heights.update(zor=zor[sl], rob=rz["Zs"])
        for hn, Zh in heights.items():
            if hn == "rob":
                o = S.solve(I, ip, prof, Zh, n_iter=100, I0=rz["I"], c0=rz["c"])
                extra = dict(zflag=rz["zflag"])
            else:
                o = S.solve(I, ip, prof, Zh, n_iter=100, I0=o0["I"], c0=o0["c"])
                extra = dict(zclip=clip) if hn == "ffz" else {}
            _put(st, f"{P}_{tag}_{hn}", sl, _res(o), Zt=Zh, **extra)
            del o
        if not lp:
            if full:
                heights["z0"] = np.zeros(B)
            npairs = np.isfinite(pairs[sl]).all((-1, -2)).sum(1)
            for hn, Zh in heights.items():
                ok = kfit_d3d(S, I, ip, Zh, pairs[sl], I0=o0["I"], c0=o0["c"])
                r = _res(ok)
                _put(st, f"KF_{tag}_{hn}", sl, r, Zt=Zh, npairs=npairs,
                     dv=ok["dv"].cpu().numpy(), pair_res=ok["pair_res"].cpu().numpy())
                del ok
        if grid is not None:
            for k, combo in enumerate(V2_GRID):
                og = S.solve(I, ip, np.tile(combo, (B, 1)), Zs, n_iter=100, I0=o0["I"],
                             c0=o0["c"])
                grid["psi16"][k, sl] = og["psi"].cpu().numpy().astype(np.float16)
                for kk in GRID_KEYS:
                    grid[kk][k, sl] = og[kk].cpu().numpy()
                del og
        del o0
        torch.cuda.empty_cache()
        print(f"{family} {sl.stop}/{n} {time.time() - t0:.0f}s clip {clip.mean():.3f}", flush=True)
    st.save(inp["idx"])
    if grid is not None:
        PRED.mkdir(parents=True, exist_ok=True)
        np.savez(PRED / f"_grid12_{tag}_{set_name}.npz", idx=inp["idx"], **grid)
        print("saved grid12", flush=True)


KP_TEXP_CAL = 0.56   # EFIT shell current density ~ Te^alpha: TRAIN (cal) median alpha (probe log)
KP_TEXP_KF = 1.0     # best label-free KP start on TRAIN probes: kfit start, Te^1.0


def kp_calibrated_cbs():
    """Bootstrap coefficient chosen on TRAIN (kp_check_cal.json, all phases, best ceiling)."""
    rep = json.loads((OUT / "kp_check_cal.json").read_text())["phases"]["all"]
    best = max((k for k in rep if k.startswith("KP neo")), key=lambda k: rep[k]["r2_ceiling"])
    return float(best.split("c=")[1]) if "c=" in best else 0.0


def run_kp(inp, set_name, family, max_frames=1024):
    """KP family (A1 field), kinetic-profile solves at the v2 force-free height (FFZ_PROF
    secant, as select_v2):  sp = Spitzer Te^1.5;  nb = Te^1.5 x Sauter sigma_neo + bootstrap
    (c_bs from TRAIN kp_check);  cal = Te^KP_TEXP_CAL (TRAIN-calibrated EFIT exponent); all
    three warm-started from the Zt = 0 default solution o0;  kf = Te^KP_TEXP_KF warm-started
    from kfit_d3d (best label-free start on TRAIN probes).  Anchored tables (kp_current)."""
    import torch
    from fec.mast_select_v2 import FFZ_PROF, force_free_z, shot_chunks
    n = len(inp["Ip"])
    shot = inp["shot"]
    S, curf = make_solver("A1")
    Iall = curf(inp)
    Ip = np.maximum(inp["Ip"], 1e5)
    cbs = kp_calibrated_cbs()
    cfgs = {"sp": dict(neo=False, c_bs=0.0, texp=1.5), "nb": dict(neo=True, c_bs=cbs, texp=1.5),
            "cal": dict(neo=False, c_bs=0.0, texp=KP_TEXP_CAL)}
    if family == "KPbs":           # re-run of nb after the TRAIN re-check with final tables
        cfgs = {f"nb{cbs:g}": dict(neo=True, c_bs=cbs, texp=1.5)}
    pairs = all_pairs(inp)
    st = Store(n)
    t0 = time.time()
    print("KP c_bs (TRAIN) =", cbs, flush=True)
    for sl in shot_chunks(shot, max_frames):
        B = sl.stop - sl.start
        I, ip = Iall[sl], Ip[sl]
        ts = kp_chords(inp, sl)
        o0, Zs = force_free_z(S, I, ip, np.tile(FFZ_PROF, (B, 1)), Rinit=R0_D3D)
        for cn, cfg in cfgs.items():
            o = solve_kinetic(S, I, ip, Zs, ts, I0=o0["I"], c0=o0["c"], **cfg)
            _put(st, f"KP_A1_ffz_{cn}", sl, _res(o), Zt=Zs, has=o["has"].cpu().numpy(),
                 pp_frac=o["pp_frac"].cpu().numpy(), bs_frac=o["bs_frac"].cpu().numpy())
            del o
        if family != "KPbs":
            ok = kfit_d3d(S, I, ip, Zs, pairs[sl], I0=o0["I"], c0=o0["c"])
            o = solve_kinetic(S, I, ip, Zs, ts, I0=ok["I"], c0=ok["c"], neo=False, texp=KP_TEXP_KF)
            _put(st, "KP_A1_ffz_kf", sl, _res(o), Zt=Zs, has=o["has"].cpu().numpy(),
                 pp_frac=o["pp_frac"].cpu().numpy())
            del o, ok
        del o0
        torch.cuda.empty_cache()
        print(f"{family} {sl.stop}/{n} {time.time() - t0:.0f}s", flush=True)
    st.save(inp["idx"])


def run_ipo(inp, set_name, max_frames=1024):
    """Oracle-Ip diagnostic (A1): FX / KF / LP at the force-free height with Ip = EFIT's total
    label current instead of the measured Ip (sizes the measured-Ip time-base error)."""
    import torch
    from fec.mast_select_v2 import FFZ_PROF, force_free_z, shot_chunks
    n = len(inp["Ip"])
    S, curf = make_solver("A1")
    Iall = curf(inp)
    lab = _subset(dict(np.load(OUT / f"lab_{set_name}.npz")), n)
    Ip = np.maximum(lab["Inode"].reshape(n, -1).sum(1).astype(np.float64), 1e5)
    coef = fit_label_profile(lab, inp["gR"], inp["gZ"])[0]
    pairs = all_pairs(inp)
    st = Store(n)
    t0 = time.time()
    for sl in shot_chunks(inp["shot"], max_frames):
        B = sl.stop - sl.start
        I, ip = Iall[sl], Ip[sl]
        prof = np.tile(FFZ_PROF, (B, 1))
        o0, Zs = force_free_z(S, I, ip, prof, Rinit=R0_D3D)
        _put(st, "FX_A1_ffz_ipo", sl, _res(S.solve(I, ip, prof, Zs, n_iter=100, I0=o0["I"],
                                                     c0=o0["c"])), Zt=Zs)
        ok = kfit_d3d(S, I, ip, Zs, pairs[sl], I0=o0["I"], c0=o0["c"])
        _put(st, "KF_A1_ffz_ipo", sl, _res(ok), Zt=Zs)
        del ok
        o0l, Zl = force_free_z(S, I, ip, coef[sl], Rinit=R0_D3D)
        _put(st, "LP_A1_ffz_ipo", sl, _res(S.solve(I, ip, coef[sl], Zl, n_iter=100, I0=o0l["I"],
                                                     c0=o0l["c"])), Zt=Zl)
        del o0, o0l
        torch.cuda.empty_cache()
        print(f"IPO {sl.stop}/{n} {time.time() - t0:.0f}s", flush=True)
    st.save(inp["idx"])


def shape_check(set_name="cal", chunk=1024):
    """FreeGS shapes (V2_GRID + FFZ_PROF) at the TRUE (label) geometry: flux error of
    G (I_shape - I_label) with EFIT's Ip.  On TRAIN it picks the calibrated fixed shape."""
    import torch
    from fec.mast_select_v2 import FFZ_PROF, V2_GRID
    shapes = [tuple(FFZ_PROF)] + [tuple(s) for s in V2_GRID]
    inp = load_set(set_name, labels=True)
    S = make_bare_solver(device="cpu")
    n = len(inp["Ip"])
    pa, pb = label_flux_bounds(inp["lab_psi"], inp["lab_lcfs"], inp["lab_lcfs_n"],
                               inp["lab_raxis"], inp["lab_zaxis"], inp["gR"], inp["gZ"])
    y = inp["lab_psi"].reshape(n, -1).astype(np.float64)
    sst = float(((y - y.mean()) ** 2).sum())
    sidx = S.sidx.numpy()
    Gn = S.G.double()
    Rs, R0 = S.Rs, S.spec.R0
    ss = np.zeros((len(shapes), n))
    for a in range(0, n, chunk):
        sl = slice(a, min(n, a + chunk))
        B = sl.stop - sl.start
        Il = torch.as_tensor(inp["lab_Inode"][sl].reshape(B, -1)[:, sidx], dtype=torch.float32)
        psi = torch.as_tensor(inp["lab_psi"][sl].reshape(B, -1), dtype=torch.float32)
        pat = torch.as_tensor(pa[sl], dtype=torch.float32)
        pbt = torch.as_tensor(pb[sl], dtype=torch.float32)
        pn = ((pat[:, None] - psi) / (pat - pbt)[:, None])[:, sidx].clamp(0, 1)
        ms = (Il != 0).to(torch.float32)
        for k, (b0, am, an) in enumerate(shapes):
            J = (b0 * Rs[None] / R0 + (1 - b0) * R0 / Rs[None]) * (1 - pn ** am).clamp_min(0) ** an * ms
            dI = (Il.sum(1, keepdim=True) * J / J.sum(1, keepdim=True) - Il).double()
            ss[k, sl] = ((dI @ Gn.T) ** 2).sum(1).numpy()
    rep = {"set": set_name, "shapes": [list(s) for s in shapes],
           "r2_ceiling": [1 - float(s.sum()) / sst for s in ss]}
    rep["best"] = rep["shapes"][int(np.argmax(rep["r2_ceiling"]))]
    (OUT / f"shape_check_{set_name}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


def _nearest_fill(choice, ok, shot):
    from fec.mast_select_v2 import nearest_in_shot
    return nearest_in_shot(choice, ok, shot)[0]


def derive_grid_variants(set_name="val", tag="A1"):
    """GR (Te / Te+ne consistency argmin over V2_GRID), FXcal (TRAIN-chosen shape), OR (label
    oracle), V2 (select_v2 composite: kfit where >= 3 pairs and converged, else GR Te+ne)."""
    from fec.mast_select_v2 import V2_FALLBACK, V2_GRID
    inp = load_set(set_name)
    g = np.load(PRED / f"_grid12_{tag}_{set_name}.npz")
    K, n = g["psi_ax"].shape
    mm_te = np.full((K, n), np.nan)
    mm_ne = np.full((K, n), np.nan)
    for k in range(K):
        psi = g["psi16"][k].astype(np.float64)
        ok = g["dI"][k] < CONV_TOL
        mm_te[k] = np.where(ok, mismatch_frames(inp, psi, g["psi_ax"][k], g["psi_b"][k], "te"), np.nan)
        mm_ne[k] = np.where(ok, mismatch_frames(inp, psi, g["psi_ax"][k], g["psi_b"][k], "ne"), np.nan)
    lab = np.load(OUT / f"lab_{set_name}.npz")
    y = lab["psi"].reshape(n, -1).astype(np.float64)
    sse = np.stack([((g["psi16"][k].reshape(n, -1).astype(np.float64) - y) ** 2).sum(1)
                    for k in range(K)])
    shot = inp["shot"]
    st = Store(n)
    sl = slice(0, n)
    best_cal = json.loads((OUT / "shape_check_cal.json").read_text())["best"]
    grid = [list(map(float, s)) for s in V2_GRID]
    choices = {}
    for nm, sc in (("GRte", mm_te), ("GR", mm_te + mm_ne)):
        okf = np.isfinite(sc).any(0)
        ch = np.where(okf, np.nanargmin(np.where(np.isfinite(sc), sc, np.inf), 0), V2_FALLBACK)
        choices[nm] = _nearest_fill(ch, okf, shot)
    choices["OR"] = np.argmin(sse, 0)
    if best_cal in grid:
        choices["FXcal"] = np.full(n, grid.index(best_cal))
    for nm, ch in choices.items():
        psi = g["psi16"][ch, np.arange(n)].astype(np.float32)
        st.put(f"{nm}_{tag}_ffz", sl, psi, choice=ch,
               **{k: g[k][ch, np.arange(n)] for k in GRID_KEYS})
    # select_v2 composite on DIII-D
    kf = load_pred(f"KF_{tag}_ffz")
    use = (kf["npairs"] >= 3) & (kf["dI"] < CONV_TOL)
    grc = choices["GR"]
    psi = np.where(use[:, None, None], kf["psi"], g["psi16"][grc, np.arange(n)].astype(np.float32))
    st.put(f"V2_{tag}_ffz", sl, psi, kfit_used=use,
           **{k: np.where(use, kf[k], g[k][grc, np.arange(n)]) for k in GRID_KEYS})
    st.save(inp["idx"])
    # premise test: does lower Thomson mismatch mean lower psi error across shapes?
    from scipy.stats import spearmanr
    rho = []
    for i in range(n):
        a, b = mm_te[:, i] + mm_ne[:, i], sse[:, i]
        ok = np.isfinite(a)
        if ok.sum() >= 6:
            rho.append(spearmanr(a[ok], b[ok]).statistic)
    rho = np.asarray(rho)
    ssr = lambda c: float(sse[c, np.arange(n)].sum())  # noqa: E731
    rep = {"n": n, "spearman_mm_vs_sse_median": float(np.nanmedian(rho)),
           "spearman_frac_positive": float(np.nanmean(rho > 0)),
           "sse_ratio_GR_over_OR": ssr(choices["GR"]) / ssr(choices["OR"]),
           "sse_ratio_GRte_over_OR": ssr(choices["GRte"]) / ssr(choices["OR"]),
           "sse_ratio_default_over_OR": ssr(np.full(n, V2_FALLBACK)) / ssr(choices["OR"]),
           "kfit_used_frac": float(use.mean()),
           "choice_hist_OR": np.bincount(choices["OR"], minlength=K).tolist(),
           "choice_hist_GR": np.bincount(choices["GR"], minlength=K).tolist()}
    if "FXcal" in choices:
        rep["sse_ratio_FXcal_over_OR"] = ssr(choices["FXcal"]) / ssr(choices["OR"])
    (OUT / f"grid_premise_{set_name}.json").write_text(json.dumps(rep, indent=1))
    np.savez(OUT / f"grid_mm_{set_name}.npz", mm_te=mm_te, mm_ne=mm_ne, sse=sse)
    print(json.dumps(rep))


def label_variants(set_name="val"):
    """LJ_A0 / LJ_A1: external model + Green's flux of the label node currents (no solve)."""
    inp = load_set(set_name, labels=True)
    gR, gZ = inp["gR"], inp["gZ"]
    G, sidx = plasma_G(gR, gZ)
    psi_pl = plasma_flux(inp["lab_Inode"], G, sidx)
    st = Store(len(psi_pl))
    sl = slice(0, len(psi_pl))
    for field in ("A0", "A1"):
        M, curf = field_matrix(field)
        psi = (psi_pl + curf(inp) @ M.T).reshape(-1, 65, 65).astype(np.float32)
        st.put(f"LJ_{field}", sl, psi)
    st.save(inp["idx"])


# ============================================================================ evaluation
EVAL = OUT / "eval"
REF_TAG = "d3dphys_e2"          # WORK/refs/<REF_TAG>/<stem>.pkl (subset frames only)


def _shot_slices(shot):
    b = np.flatnonzero(np.diff(shot) != 0) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(shot)]
    return [slice(s, e) for s, e in zip(st, en)]


def gt_refs(set_name="val", nproc=8):
    """Reference contours / derived scalars per frame via fec.score (scoring-only mask)."""
    from multiprocessing import Pool
    from fec import score as SC
    inp = load_set(set_name)
    lab = np.load(OUT / f"lab_{set_name}.npz")
    psi_st = -lab["psi"]                                     # stored convention
    sls = _shot_slices(inp["shot"])
    stems = load_subsets()[f"{set_name}_shots"]
    info = json.loads((WORK / "ds" / CONFIG / "info.json").read_text())
    keys = [f"{REF_TAG}/{info['order'][inp['shot'][s.start]]}" for s in sls]
    assert all(k.split("/")[1] in stems for k in keys)
    with Pool(nproc) as pool:
        list(pool.imap_unordered(SC._ref_worker, [(k, psi_st[s]) for k, s in zip(keys, sls)]))
    n = len(inp["Ip"])
    contours = [None] * n
    cons = np.full((n, 7), np.nan)
    cmask = np.zeros((n, 7), bool)
    rgeo = np.full(n, np.nan)
    for k, s in zip(keys, sls):
        c, cg, cm, rg = SC._load_ref(k)
        for j, i in enumerate(range(s.start, s.stop)):
            contours[i] = c[j]
        cons[s], cmask[s], rgeo[s] = cg, cm, rg
    return dict(keys=keys, slices=sls, contours=contours, cons=cons, cmask=cmask, rgeo=rgeo,
                q95=lab["q95"], betan=lab["betan"], psi_st=psi_st)


def _betan_e(psi, pa, pb, ra, za, F, ip_kA, Rc, Zc, te, ne, gR, gZ):
    """Thomson electron betaN on a flux map (d3d_scalar_physics convention, B0 = F/R0)."""
    from matplotlib.path import Path as MPath
    from fec.eqscalars import surface_contour
    from fec.kinetic import chord_psiN
    c = surface_contour((pa - psi) / (pa - pb), gR, gZ, 0.995, ra, za)
    if c is None:
        return np.nan
    lr, lz = c
    ok = np.isfinite(te) & np.isfinite(ne) & np.isfinite(Rc) & np.isfinite(Zc) & (te > 0) & (ne > 0)
    if ok.sum() < 10:
        return np.nan
    pn = chord_psiN(psi, pa, pb, gR, gZ, Rc[ok], Zc[ok])
    pe = te[ok] * ne[ok] * QE
    ins = pn < 1
    if ins.sum() < 6:
        return np.nan
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(pn[ins], bins) - 1
    prof = np.array([np.median(pe[ins][idx == b]) if (idx == b).any() else np.nan for b in range(10)])
    xb = 0.5 * (bins[1:] + bins[:-1])
    okb = np.isfinite(prof)
    if okb.sum() < 4:
        return np.nan
    RR, ZZ = np.meshgrid(gR, gZ)
    m = MPath(np.c_[lr, lz]).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
    if m.sum() < 5:
        return np.nan
    pav = np.sum(np.interp((pa - psi[m]) / (pa - pb), xb[okb], prof[okb]) * RR[m]) / np.sum(RR[m])
    B0 = F / R0_D3D
    a = 0.5 * (lr.max() - lr.min())
    return 100 * 2 * MU0 * pav / B0 ** 2 * a * B0 / (abs(ip_kA) / 1e3)


def _frame_worker(args):
    """Official per-frame functionals (starter local_score.score_shot, one frame) + scalars."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fec import score as SC  # noqa: F401  (puts the starter scorer on sys.path)
    from contour import symmetric_hausdorff
    from derive import derive_frame
    from lcfs import extract_lcfs_with_sign
    from fec.eqscalars import q_at
    (psi_c, gt_st, ref_c, need, rgeo, pa, pb, ra, za, F, ip, Rc, Zc, te, ne, gR, gZ) = args
    R, Z, mc, mf = SC._mask()
    pred = -psi_c
    nf = ~np.isfinite(pred)                 # official: non-finite pixels -> fold mean psi
    pz = np.where(nf, 0.0, pred)
    out = {"ss_p": float(((gt_st - pz) ** 2)[~nf].sum()), "ss_m": float(((gt_st + pz) ** 2)[~nf].sum()),
           "nf_n": int(nf.sum()), "nf_s1": float(gt_st[nf].sum()), "nf_s2": float((gt_st[nf] ** 2).sum()),
           "d": np.nan, "lfail": False, "cons": np.full(7, np.nan), "q95": np.nan, "betan": np.nan}
    ex, _ = extract_lcfs_with_sign(pred, R, Z, "DIII-D", mc, mf, n_iter=22, axis_sign=-1.0)
    if ref_c is not None:
        if ex is None:
            out["d"], out["lfail"] = 1.0, True
        else:
            out["d"] = min(1.0, symmetric_hausdorff(ex, np.asarray(ref_c, float)) / rgeo
                           if rgeo > 0 else 1.0)
    if need.any():
        vals = derive_frame(pred, R, Z, "DIII-D", mc, mf, contour=ex, with_li=bool(need[6]),
                            axis_sign=-1.0)
        out["cons"] = np.array([vals[k] for k in ("R_axis", "Z_axis", "kappa", "tri_top",
                                                  "tri_bot", "volume", "li")], float)
    if np.isfinite(pa) and np.isfinite(pb) and pb > -1e20 and pa > pb:
        try:
            out["q95"] = q_at(psi_c, gR, gZ, pa, pb, ra, za, F, 0.95)
            out["betan"] = K_BETA * _betan_e(psi_c, pa, pb, ra, za, F, ip, Rc, Zc, te, ne, gR, gZ)
        except Exception:  # noqa: BLE001
            pass
    return out


def load_pred(name):
    z = dict(np.load(PRED / f"{name}.npz"))
    if "psi_ax" not in z:
        import torch
        S = make_bare_solver(device="cpu")
        pa, pb, ra, za = [], [], [], []
        for a in range(0, len(z["psi"]), 512):
            o = S.final_boundary(torch.as_tensor(z["psi"][a:a + 512]))
            pa.append(o[0].numpy()); pb.append(o[1].numpy())
            ra.append(o[2].numpy()); za.append(o[3].numpy())
        z.update(psi_ax=np.concatenate(pa), psi_b=np.concatenate(pb), Rax=np.concatenate(ra),
                 Zax=np.concatenate(za))
    return z


def evaluate(name, set_name="val", nproc=8, gt=None):
    """Per-frame official functionals of a saved variant -> EVAL/<name>.npz."""
    from multiprocessing import Pool
    inp = load_set(set_name)
    gt = gt or gt_refs(set_name)
    z = load_pred(name)
    assert np.array_equal(z["idx"], inp["idx"])
    n = len(inp["Ip"])
    F = MU0 * N_TF * inp["bt"] * 1e3 / (2 * np.pi)
    gR, gZ = inp["gR"], inp["gZ"]

    def jobs():
        for i in range(n):
            Rc = np.r_[inp["vR"][i], inp["tR"][i]].astype(float)
            Zc = np.r_[inp["vZ"][i], inp["tZ"][i]].astype(float)
            te = np.r_[inp["vte"][i], inp["tte"][i]].astype(float)
            ne = np.r_[inp["vne"][i], inp["tne"][i]].astype(float)
            yield (z["psi"][i].astype(np.float64), gt["psi_st"][i].astype(np.float64),
                   gt["contours"][i], gt["cmask"][i], gt["rgeo"][i], float(z["psi_ax"][i]),
                   float(z["psi_b"][i]), float(z["Rax"][i]), float(z["Zax"][i]), float(F[i]),
                   float(max(inp["Ip"][i], 1e5) / 1e3), Rc, Zc, te, ne, gR, gZ)   # solver Ip floor
    t0 = time.time()
    with Pool(nproc) as pool:
        rows = list(pool.imap(_frame_worker, jobs(), chunksize=16))
    ev = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    lab = np.load(OUT / f"lab_{set_name}.npz")
    ins = lab["Inode"].reshape(n, -1) != 0
    dpsi = (z["psi"].reshape(n, -1) - lab["psi"].reshape(n, -1)).astype(np.float64)
    ev["ss_in"] = (np.where(ins, np.nan_to_num(dpsi), 0) ** 2).sum(1)
    ev["n_in"] = ins.sum(1)
    ev["mm_te"] = mismatch_frames(inp, z["psi"].astype(np.float64), z["psi_ax"], z["psi_b"], "te")
    ev["mm_ne"] = mismatch_frames(inp, z["psi"].astype(np.float64), z["psi_ax"], z["psi_b"], "ne")
    for k in ("Rax", "Zax", "psi_ax", "psi_b", "Zt", "dI", "limited", "zclip", "npairs", "dv"):
        if k in z:
            ev["meta_" + k] = z[k]
    EVAL.mkdir(parents=True, exist_ok=True)
    np.savez(EVAL / f"{name}.npz", **ev)
    print(f"evaluated {name} in {time.time() - t0:.0f}s", flush=True)
    return ev


def aggregate(ev, gt, mask=None, shot=None):
    """Official composite over a frame subset (pooled means of the subset, as score.py)."""
    n = len(ev["ss_p"])
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    g = gt["psi_st"][m].astype(np.float64)
    sst = float(((g - g.mean()) ** 2).sum())
    ssp, ssm = float(ev["ss_p"][m].sum()), float(ev["ss_m"][m].sum())
    if "nf_n" in ev:
        mu = g.mean()
        extra = float((ev["nf_s2"][m] - 2 * mu * ev["nf_s1"][m] + mu ** 2 * ev["nf_n"][m]).sum())
        ssp, ssm = ssp + extra, ssm + extra
    sign = 1 if ssp <= ssm else -1
    r2psi = 1 - min(ssp, ssm) / sst
    r2s = []
    for k_gt, k_p in (("q95", "q95"), ("betan", "betan")):
        y = gt[k_gt][m]
        p = ev[k_p][m]
        ok = np.isfinite(y)
        mu = y[ok].mean()
        pp = np.where(np.isfinite(p[ok]), p[ok], mu)
        r2s.append(1 - ((y[ok] - pp) ** 2).sum() / ((y[ok] - mu) ** 2).sum())
    r2qb = float(np.mean(r2s))
    ds = []
    for s in np.unique(shot[m]):
        dd = ev["d"][m & (shot == s)]
        dd = dd[np.isfinite(dd)]
        if len(dd):
            ds.append(dd.mean())
    dl = float(np.mean(ds)) if ds else 1.0
    r2c = np.full(7, np.nan)
    for j in range(7):
        cm = gt["cmask"][m][:, j]
        y = gt["cons"][m][cm, j]
        if len(y) < 2:
            continue
        mu = y.mean()
        p = ev["cons"][m][cm, j]
        p = np.where(np.isfinite(p), p, mu)
        tot = ((y - mu) ** 2).sum()
        if tot > 0:
            r2c[j] = 1 - ((y - p) ** 2).sum() / tot
    cons = float(np.mean(np.maximum(0, r2c[np.isfinite(r2c)])))
    S = 0.55 * max(0, r2psi) + 0.15 * max(0, r2qb) + 0.10 * (1 - min(1, dl)) + 0.20 * cons
    rms_in = float(np.sqrt(ev["ss_in"][m].sum() / max(ev["n_in"][m].sum(), 1)) * 1e3)
    return dict(n=int(m.sum()), S=S, r2psi=r2psi, r2qb=r2qb, r2q95=float(r2s[0]),
                r2betan=float(r2s[1]), dlcfs=dl, cons=cons, sign=sign,
                r2cons=dict(zip(("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume",
                                 "li"), [None if not np.isfinite(x) else float(x) for x in r2c])),
                rms_in_mWb=rms_in, lcfs_fail=float(np.mean(ev["lfail"][m])),
                mm_te_med=float(np.nanmedian(ev["mm_te"][m])),
                mm_ne_med=float(np.nanmedian(ev["mm_ne"][m])))


def official(name, set_name="val", nproc=8):
    """Cross-check: fec.score.score (official local scorer) on the whole subset."""
    from fec import score as SC
    inp = load_set(set_name)
    gt = gt_refs(set_name)
    ev = np.load(EVAL / f"{name}.npz")
    z = load_pred(name)
    gts, preds = [], []
    for s in gt["slices"]:
        gts.append({"psi": gt["psi_st"][s].astype(np.float32), "q95": gt["q95"][s],
                    "betaN": gt["betan"][s]})
        preds.append({"psirz": (-z["psi"][s]).astype(np.float64), "q95": ev["q95"][s],
                      "betaN": ev["betan"][s]})
    res = SC.score(gt["keys"], gts, preds, nproc=nproc, verbose=False)
    mine = aggregate(dict(ev), gt, shot=inp["shot"])
    out = {"official": {k: res[k] for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")},
           "aggregate": {k: mine[k] for k in ("S", "r2psi", "r2qb", "dlcfs", "cons")},
           "official_cons_each": res["r2_cons_each"]}
    print(json.dumps(out, indent=1))
    return out


DESC = {
    "LJ_A0": "label current + A0 field (no solve)",
    "LJ_A1": "label current + A1 field (no solve)",
    "LP_A1_ffz": "label profile, free-boundary, v2 force-free height",
    "LP_A1_zor": "label profile, oracle height",
    "LP_A1_z0": "label profile, Zt = 0",
    "LP_A1_rob": "label profile, v3 robust height",
    "LP_A0_ffz": "label profile, A0 field",
    "LP_A1_lim_ffz": "label profile + polygon limiter",
    "FX_A1_ffz": "fixed default (0.05,2,2), ffz",
    "FX_A1_zor": "fixed default, oracle height",
    "FX_A1_z0": "fixed default, Zt = 0",
    "FX_A1_rob": "fixed default, v3 robust height",
    "FX_A0_ffz": "fixed default, A0 field",
    "FX_A1_lim_ffz": "fixed default + limiter",
    "FXcal_A1_ffz": "TRAIN-calibrated fixed shape (0.5,1,1)",
    "KF_A1_ffz": "kfit_d3d (Thomson pairs), ffz",
    "KF_A1_zor": "kfit_d3d, oracle height",
    "KF_A1_z0": "kfit_d3d, Zt = 0",
    "KF_A1_rob": "kfit_d3d, v3 robust height",
    "KF_A0_ffz": "kfit_d3d, A0 field",
    "KF_A1_lim_ffz": "kfit_d3d + limiter",
    "GRte_A1_ffz": "12-shape grid, Te-consistency argmin",
    "GR_A1_ffz": "12-shape grid, Te+ne-consistency argmin",
    "V2_A1_ffz": "select_v2 analogue (kfit if >=3 pairs else grid Te+ne)",
    "OR_A1_ffz": "12-shape grid, label-oracle argmin",
    "KP_A1_ffz_sp": "kinetic profile, Spitzer Te^1.5",
    "KP_A1_ffz_nb": "kinetic profile, neo + bootstrap (c_bs TRAIN)",
    "KP_A1_ffz_cal": "kinetic profile, Te^0.56 (TRAIN EFIT exponent)",
    "KP_A1_ffz_kf": "kinetic profile Te^1.0 from kfit start",
    "KP_A1_ffz_nb2": "kinetic profile, neo + bootstrap c_bs=2 (final TRAIN re-check)",
    "FX_A1_ffz_ipo": "fixed default, oracle Ip (EFIT total current)",
    "KF_A1_ffz_ipo": "kfit_d3d, oracle Ip",
    "LP_A1_ffz_ipo": "label profile, oracle Ip",
}
PHASES = ("ramp_up", "flat", "ramp_down")


def report(set_name="val", names=None):
    """Tables of the official composite per variant (all frames and per phase) + diagnostics."""
    inp = load_set(set_name)
    lab = np.load(OUT / f"lab_{set_name}.npz")
    gt = gt_refs(set_name)
    ph = phase_of(inp)
    shot = inp["shot"]
    names = names or [k for k in DESC if (EVAL / f"{k}.npz").exists()]
    zor = label_centroid_z({"Inode": lab["Inode"]}, inp["gZ"])
    rows = {}
    for nm in names:
        ev = dict(np.load(EVAL / f"{nm}.npz"))
        r = {"all": aggregate(ev, gt, shot=shot)}
        for p in PHASES:
            r[p] = aggregate(ev, gt, mask=ph == p, shot=shot)
        if "meta_Rax" in ev:
            r["dRax_med_cm"] = float(np.nanmedian(np.abs(ev["meta_Rax"] - lab["raxis"])) * 100)
            r["dZax_med_cm"] = float(np.nanmedian(np.abs(ev["meta_Zax"] - lab["zaxis"])) * 100)
            r["dRax_bias_cm"] = float(np.nanmedian(ev["meta_Rax"] - lab["raxis"]) * 100)
        if "meta_Zt" in ev:
            dz = np.abs(ev["meta_Zt"] - zor)
            r["dZt_med_cm"] = float(np.nanmedian(dz) * 100)
            r["dZt_p90_cm"] = float(np.nanpercentile(dz, 90) * 100)
        if "meta_dI" in ev:
            r["conv"] = float(np.nanmean(ev["meta_dI"] < CONV_TOL))
        if "meta_limited" in ev:
            r["limited"] = float(np.mean(ev["meta_limited"]))
        if "meta_psi_b" in ev:
            r["no_xpt"] = float(np.mean(ev["meta_psi_b"] < -1e20))
        rows[nm] = r
    (OUT / f"report_{set_name}.json").write_text(json.dumps(rows, indent=1))
    cs = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li")
    f = lambda x: "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"  # noqa: E731
    print("| variant | description | S | R2psi | R2q95 | R2betaN | D_LCFS | cons | " +
          " | ".join(cs) + " | rms_in mWb | dRax cm | dZax cm | Te mm |")
    print("|" + "---|" * (14 + len(cs)))
    for nm, r in rows.items():
        a = r["all"]
        print(f"| {nm} | {DESC.get(nm, '')} | {a['S']:.4f} | {a['r2psi']:.5f} | {f(a['r2q95'])} | "
              f"{f(a['r2betan'])} | {a['dlcfs']:.4f} | {a['cons']:.4f} | " +
              " | ".join(f(a["r2cons"][c]) for c in cs) +
              f" | {a['rms_in_mWb']:.1f} | {r.get('dRax_med_cm', float('nan')):.1f} | "
              f"{r.get('dZax_med_cm', float('nan')):.1f} | {a['mm_te_med']:.3f} |")
    print()
    print("| variant | " + " | ".join(f"{p} S / R2psi / D_LCFS / cons" for p in PHASES) + " |")
    print("|---|---|---|---|")
    for nm, r in rows.items():
        print(f"| {nm} | " + " | ".join(
            f"{r[p]['S']:.4f} / {r[p]['r2psi']:.4f} / {r[p]['dlcfs']:.4f} / {r[p]['cons']:.4f}"
            for p in PHASES) + " |")
    print()
    print("| variant | dZt vs oracle med/p90 cm | dRax bias cm | conv | limited | no X-point |")
    print("|---|---|---|---|---|---|")
    for nm, r in rows.items():
        print(f"| {nm} | {r.get('dZt_med_cm', float('nan')):.2f} / {r.get('dZt_p90_cm', float('nan')):.2f} | "
              f"{r.get('dRax_bias_cm', float('nan')):+.1f} | {r.get('conv', float('nan')):.3f} | "
              f"{r.get('limited', float('nan')):.3f} | {r.get('no_xpt', float('nan')):.3f} |")
    print("phase counts", {p: int((ph == p).sum()) for p in PHASES})
    return rows


def evaluate_many(names, set_name="val", nproc=8):
    gt = gt_refs(set_name)
    for nm in names:
        if (EVAL / f"{nm}.npz").exists():
            print("exists", nm)
            continue
        evaluate(nm, set_name, nproc, gt)


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--set", default="val")
    ap.add_argument("--family", default="F")
    ap.add_argument("--names", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "run":
        run_family(a.family, a.set, limit=a.limit)
    elif a.cmd == "lj":
        label_variants(a.set)
    elif a.cmd == "eval":
        evaluate_many([x for x in a.names.split(",") if x], a.set, a.nproc)
    elif a.cmd == "report":
        report(a.set, [x for x in a.names.split(",") if x] or None)
    elif a.cmd == "official":
        official(a.names, a.set, a.nproc)
    elif a.cmd == "subsets":
        make_subsets()
    elif a.cmd == "prep":
        prep(a.set)
    elif a.cmd == "geom":
        build_geometry()
    elif a.cmd == "extfit":
        ext_fit()
    elif a.cmd == "kpcheck":
        kp_check(a.set)
    elif a.cmd == "shapecheck":
        shape_check(a.set)
    elif a.cmd == "gridvars":
        derive_grid_variants(a.set)


if __name__ == "__main__":
    main()
