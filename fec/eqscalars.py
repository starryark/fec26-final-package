"""Scalars from a solved equilibrium on the 65x65 grid (physical psi, axis = maximum).

q(psiN) = F/(2 pi) * contour-integral dl / (R |grad psi|)   (psi per radian)
F^2(psi) = F_vac^2 + 2 * int FF' dpsi  (FreeGS-style profile from fec.gs)
Volume-averaged quantities by summing grid cells inside the LCFS (psiN<1 core mask).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage import measure

MU0 = 4e-7 * math.pi


def surface_contour(psiN, gR, gZ, level, Rax, Zax):
    """Closed contour of psiN=level enclosing the axis (R,Z arrays) or None."""
    cs = measure.find_contours(psiN, level)
    best, bd = None, 1e9
    dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
    for c in cs:
        if len(c) < 8:
            continue
        Z = gZ[0] + c[:, 0] * dZ
        R = gR[0] + c[:, 1] * dR
        closed = np.hypot(R[0] - R[-1], Z[0] - Z[-1]) < 2 * max(dR, dZ)
        if not closed:
            continue
        # contains axis? winding test
        ang = np.unwrap(np.arctan2(Z - Zax, R - Rax))
        if abs(ang[-1] - ang[0]) < 5.0:
            continue
        d = np.mean(np.hypot(R - Rax, Z - Zax))
        if d < bd:
            best, bd = (R, Z), d
    return best


def q_at(psi, gR, gZ, psi_ax, psi_b, Rax, Zax, F, level=0.95, spline=None):
    psiN = (psi_ax - psi) / (psi_ax - psi_b)
    c = surface_contour(psiN, gR, gZ, level, Rax, Zax)
    if c is None:
        return np.nan
    R, Z = c
    if spline is None:
        spline = RectBivariateSpline(gZ, gR, psi, kx=3, ky=3)
    pR = spline(Z, R, dy=1, grid=False)
    pZ = spline(Z, R, dx=1, grid=False)
    g = np.hypot(pR, pZ)
    Rm = 0.5 * (R[1:] + R[:-1]); gm = 0.5 * (g[1:] + g[:-1])
    dl = np.hypot(np.diff(R), np.diff(Z))
    return abs(F) / (2 * math.pi) * np.sum(dl / (Rm * np.maximum(gm, 1e-9)))


def profile_integrals(psiN_nodes, beta0, am, an, n=200):
    """int_x^1 g(s) ds for g=(1-s^am)^an evaluated at x=psiN_nodes (vectorised table)."""
    s = np.linspace(0, 1, n + 1)
    g = np.clip(1 - s ** am, 0, None) ** an
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(s))])
    tail = cum[-1] - cum   # int_s^1
    return np.interp(np.clip(psiN_nodes, 0, 1), s, tail)
