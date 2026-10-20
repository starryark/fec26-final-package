"""Label-free kinetic consistency of an equilibrium with midplane Thomson profiles.

Te and ne are flux functions: on the Thomson chord (MAST: horizontal, Z=0) the
inboard and outboard branches must coincide when expressed in psiN of the true
equilibrium. The mismatch therefore scores candidate equilibria without labels.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def chord_psiN(psi, psi_ax, psi_b, gR, gZ, R, Z):
    """Bilinear psiN at points (R,Z) for one frame."""
    dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
    fr = np.clip((R - gR[0]) / dR, 0, len(gR) - 1.001)
    fz = np.clip((Z - gZ[0]) / dZ, 0, len(gZ) - 1.001)
    i, j = fz.astype(int), fr.astype(int)
    a, b = fz - i, fr - j
    p = (psi[i, j] * (1 - a) * (1 - b) + psi[i + 1, j] * a * (1 - b) +
         psi[i, j + 1] * (1 - a) * b + psi[i + 1, j + 1] * a * b)
    return (psi_ax - p) / (psi_ax - psi_b)


def symmetry_mismatch(pn, R, prof, Rax, grid=np.linspace(0.05, 0.85, 17)):
    """Mean |inboard - outboard| of a flux-function profile over psiN grid (relative).

    pn: psiN at chord points R; prof: measured values (NaN = missing).
    Returns (mismatch, n_points) ; NaN if insufficient coverage.
    """
    ok = np.isfinite(prof) & np.isfinite(pn)
    if ok.sum() < 20:
        return np.nan, 0
    R, pn, v = R[ok], pn[ok], prof[ok]
    o = np.argsort(R)
    R, pn, v = R[o], pn[o], v[o]
    v = median_filter(v, size=5, mode="nearest")
    scale = np.nanpercentile(v, 95)
    if not scale > 0:
        return np.nan, 0
    inb = R < Rax
    out = ~inb
    res = []
    for side in (inb, out):
        s_pn, s_v = pn[side], v[side]
        if side is inb:
            s_pn, s_v = s_pn[::-1], s_v[::-1]   # increasing psiN away from axis
        # keep monotone part inside plasma
        keep = s_pn < 1.0
        s_pn, s_v = s_pn[keep], s_v[keep]
        if len(s_pn) < 5:
            return np.nan, 0
        m = np.maximum.accumulate(s_pn)
        u, idx = np.unique(m, return_index=True)
        res.append((u, s_v[idx]))
    (pi, vi), (po, vo) = res
    g = grid[(grid >= max(pi[0], po[0])) & (grid <= min(pi[-1], po[-1]))]
    if len(g) < 5:
        return np.nan, len(g)
    di = np.interp(g, pi, vi) - np.interp(g, po, vo)
    return float(np.mean(np.abs(di)) / scale), len(g)
