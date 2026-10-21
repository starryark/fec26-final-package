"""Verbatim copies of the research-tree helpers the NRS-L cascade stage (fec/mast_nrsl.py) needs and the v11 package does
not ship (family doc amendment 68: "fill_in_shot and meta_from_maps enter as verbatim copies in a new module, each with a
test against its source. The vetoes enter the same way").  Each function / constant below is the exact source text of:

  FROZEN_SHA_PREFIX, fill_in_shot                          fec/licond_d3d.py   (licond fill rule; frozen.json prefix)
  meta_from_maps                                           fec/amort_real.py   (axis / boundary meta of a map batch)
  MU0, _GEO, _mast_geo, delta_star, dist_to_polygon,       fec/xfer_eval.py    (amendment-64 label-free vetoes: FAIR
  _g3_frame                                                                      wall O-point search, LCFS, map-Ip)

tests/test_fec_nrsl_pkg.py compares each against its source module where that module exists (the research tree).
CD_SHA256: the accepted SHA-256 of fec/current_diffusion.py for the li target's frozen-code assert: the frozen copy
(research tree, = the file the C57 calibration froze) and the package's copy, whose only difference is the one docstring
line the package build scrubs (line 77: the outputs path "C:/.../fec_work/cdiff/" -> "<FEC_WORK>/cdiff/").
No label, scorer mask or envelope is read by anything here.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from fec.paths import SCORING, WORK  # noqa: F401

CD_SHA256 = {"source": "bf7f71ce16f5a08e4596afedd994eb74d3ef4e8ae077b62b18962b30d3827868",
             "package": "34a9cb6f7fcaef7132b8dbd2bdc91bfa18c3cd2d15a646c8e1f6b86030e68246"}


# ---- verbatim from fec/licond_d3d.py: FROZEN_SHA_PREFIX
FROZEN_SHA_PREFIX = "2098a4d9"


# ---- verbatim from fec/licond_d3d.py: fill_in_shot
def fill_in_shot(x, shot, t, default):
    """Fill NaNs by the nearest-in-time finite value of the same shot; shots with none -> default."""
    x = np.asarray(x, float).copy()
    filled = np.zeros(len(x), bool)
    for s in np.unique(shot):
        ii = np.flatnonzero(shot == s)
        ok = np.isfinite(x[ii])
        bad = ii[~ok]
        if not len(bad):
            continue
        if ok.any():
            good = ii[ok]
            j = np.abs(np.asarray(t)[bad][:, None] - np.asarray(t)[good][None, :]).argmin(1)
            x[bad] = x[good[j]]
        else:
            x[bad] = default
        filled[bad] = True
    return x, filled


# ---- verbatim from fec/amort_real.py: meta_from_maps
def meta_from_maps(psi, device="cuda", bs=4096):
    """Axis/boundary meta with the physics candidates' routines: fec.mast_select_v3.V3Solver
    (limiter=None, as cand_zsc_z0 / sub09) find_axis, find_xpoints, boundary(final=True);
    frames without any X-point get fec.gs's in-loop fallback psi_ax - 0.05|psi_ax| (counted)."""
    import torch
    from fec.machines import mast_setup
    from fec.mast_select_v3 import V3Solver
    spec, M = mast_setup()
    S = V3Solver(spec, M, device=device, limiter=None)
    N = len(psi)
    meta = {k: np.zeros(N) for k in ("psi_ax", "psi_b", "Rax", "Zax", "xfound", "x_upper",
                                     "x_lower", "imb", "limited")}
    for a in range(0, N, bs):
        e = min(N, a + bs)
        with torch.no_grad():
            ps = torch.as_tensor(np.asarray(psi[a:e]), device=S.dev, dtype=S.dt)
            pa, Ra, Za = S.find_axis(ps)
            xp = S.find_xpoints(ps, Ra, Za, pa)
            pb, lim = S.boundary(ps, pa, Ra, Za, xp, final=True)
        (fu, vu, _, _), (fl, vl, _, _) = xp
        fu, fl, vu, vl = [t.cpu().numpy() for t in (fu, fl, vu, vl)]
        pa, pb, Ra, Za = [t.cpu().numpy().astype(np.float64) for t in (pa, pb, Ra, Za)]
        nox = ~(fu | fl)
        pb = np.where(nox, pa - 0.05 * np.maximum(np.abs(pa), 1e-3), pb)
        meta["psi_ax"][a:e], meta["psi_b"][a:e], meta["Rax"][a:e], meta["Zax"][a:e] = pa, pb, Ra, Za
        meta["xfound"][a:e] = ~nox
        meta["x_upper"][a:e], meta["x_lower"][a:e] = fu, fl
        meta["imb"][a:e] = np.where(fu & fl, (vu - vl) / np.maximum(pa - pb, 1e-6), np.nan)
        meta["limited"][a:e] = lim.cpu().numpy()
    return meta


# ---- verbatim from fec/xfer_eval.py: MU0
MU0 = 4e-7 * math.pi


# ---- verbatim from fec/xfer_eval.py: _GEO
_GEO = {}


# ---- verbatim from fec/xfer_eval.py: _mast_geo
def _mast_geo():
    if not _GEO:
        from fec.geometry import machine_geometry
        g = machine_geometry("mast_public_test")
        z = np.load(WORK / "geometry" / "mast_limiter_fairmast.npz")
        gR, gZ = np.asarray(g["grid_R"], float), np.asarray(g["grid_Z"], float)
        wall = np.c_[z["R"], z["Z"]].astype(float)
        from matplotlib.path import Path as MPath
        RR, ZZ = np.meshgrid(gR, gZ)
        mc = MPath(wall).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
        _GEO.update(gR=gR, gZ=gZ, wall=wall, mc=mc)
    return _GEO


# ---- verbatim from fec/xfer_eval.py: delta_star
def delta_star(psi, gR, gZ):
    dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
    d = np.zeros_like(psi)
    c = psi[1:-1, 1:-1]
    pr, pl, pu, pd = psi[1:-1, 2:], psi[1:-1, :-2], psi[2:, 1:-1], psi[:-2, 1:-1]
    Ri = gR[1:-1][None, :]
    d[1:-1, 1:-1] = (pr - 2 * c + pl) / dR ** 2 - (pr - pl) / (2 * Ri * dR) + (pu - 2 * c + pd) / dZ ** 2
    return d


# ---- verbatim from fec/xfer_eval.py: dist_to_polygon
def dist_to_polygon(pts, poly):
    """Min distance of points (n, 2) to the closed polygon's edges."""
    a = poly
    b = np.roll(poly, -1, 0)
    ab = b - a
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.sqrt(((pts[:, None, :] - proj) ** 2).sum(-1)).min(1)


# ---- verbatim from fec/xfer_eval.py: _g3_frame
def _g3_frame(args):
    """O-point search restricted to grid nodes inside the public FAIR-MAST wall polygon (wallmask=True; the scorer's
    own envelope mask is never used); without it the scorer's O-point finder picks the P2 coil flux maximum at
    (0.5, +-1.75) on ~5% of S1's P2-on frames.  LCFS validity is not wall-restricted (mask_f None)."""
    psi, Ip, wallmask = args[:3]
    wall_lcfs = len(args) > 3 and bool(args[3])
    if str(SCORING) not in sys.path:
        sys.path.insert(0, str(SCORING))
    from derive import derive_frame
    from lcfs import extract_lcfs
    from matplotlib.path import Path as MPath
    geo = _mast_geo()
    gR, gZ, wall = geo["gR"], geo["gZ"], geo["wall"]
    mc = geo["mc"] if wallmask else None
    psi = np.asarray(psi, np.float64)
    out = dict(found=False, inside=False, ip_err=np.nan, R_axis=np.nan, Z_axis=np.nan, kappa=np.nan,
               Rin=np.nan, Rout=np.nan)
    try:
        # wall_lcfs (informational only, root 07:35 UTC): LCFS validity restricted to the public FAIR wall polygon
        C = extract_lcfs(psi, gR, gZ, "MAST", mask_coarse=mc,
                         mask_f=geo["mc"].astype(np.float64) if wall_lcfs else None)
    except Exception:
        C = None
    try:
        d = derive_frame(psi, gR, gZ, "MAST", mask_coarse=mc, contour=C, with_li=False)
        out["R_axis"], out["Z_axis"], out["kappa"] = d["R_axis"], d["Z_axis"], d.get("kappa", np.nan)
    except Exception:
        pass
    if C is None or len(C) < 8:
        return out
    out["found"] = True
    wp = MPath(wall)
    ins = wp.contains_points(C)
    if (~ins).any():
        ins[~ins] = dist_to_polygon(C[~ins], wall) <= 0.01
    out["inside"] = bool(ins.all())
    RR, ZZ = np.meshgrid(gR, gZ)
    m = MPath(C).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
    J = -delta_star(psi, gR, gZ) / (MU0 * RR)
    Itot = float((J * m).sum() * (gR[1] - gR[0]) * (gZ[1] - gZ[0]))
    out["ip_err"] = abs(Itot - Ip) / Ip
    near = np.abs(C[:, 1]) < 0.05
    if near.any():
        out["Rin"], out["Rout"] = float(C[near, 0].min()), float(C[near, 0].max())
    return out
