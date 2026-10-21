"""Axisymmetric Green's functions (poloidal flux per radian, psi = R*A_phi).

psi(R,Z) = G(R,Z; Rc,Zc) * I   [Wb/rad] for a toroidal filament current I [A] at (Rc,Zc).
Positive I flows in +phi; then psi > 0 near the filament.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk

MU0 = 4e-7 * np.pi


def greens_psi(R, Z, Rc, Zc):
    """Flux per radian at (R,Z) from a unit filament at (Rc,Zc); broadcasting arrays."""
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    Rc = np.asarray(Rc, dtype=np.float64)
    Zc = np.asarray(Zc, dtype=np.float64)
    denom = (R + Rc) ** 2 + (Z - Zc) ** 2
    m = 4.0 * R * Rc / denom  # k^2
    m = np.clip(m, 1e-14, 1.0 - 1e-12)
    k = np.sqrt(m)
    K = ellipk(m)
    E = ellipe(m)
    return MU0 / (2.0 * np.pi) * np.sqrt(R * Rc) * ((2.0 - m) * K - 2.0 * E) / k


def greens_BrBz(R, Z, Rc, Zc):
    """(Br, Bz) at (R,Z) from a unit filament at (Rc,Zc) [T/A]."""
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    dZ = Z - Zc
    a2 = (R + Rc) ** 2 + dZ ** 2
    b2 = (R - Rc) ** 2 + dZ ** 2
    m = np.clip(4.0 * R * Rc / a2, 1e-14, 1.0 - 1e-12)
    K = ellipk(m)
    E = ellipe(m)
    a = np.sqrt(a2)
    pref = MU0 / (2.0 * np.pi)
    Br = pref * dZ / (np.maximum(R, 1e-9) * a) * (-K + (Rc ** 2 + R ** 2 + dZ ** 2) / b2 * E)
    Bz = pref / a * (K + (Rc ** 2 - R ** 2 - dZ ** 2) / b2 * E)
    return Br, Bz


def element_filaments(R, Z, W, H, a1=0.0, a2=0.0, n=None):
    """Sub-filament centres (and equal weights) covering a rectangle/parallelogram.

    EFIT convention: a1 shears the vertical sides (x shifted by z/tan(a1)),
    a2 shears the horizontal sides (z shifted by x*tan(a2)); 0 (or 90) means none.
    """
    if n is None:
        nr = max(2, int(np.ceil(W / 0.01)))
        nz = max(2, int(np.ceil(H / 0.01)))
    else:
        nr, nz = n
    u = (np.arange(nr) + 0.5) / nr - 0.5  # in [-0.5,0.5]
    v = (np.arange(nz) + 0.5) / nz - 0.5
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = uu * W
    z = vv * H
    if a1 not in (0.0, 90.0) and a1 == a1:
        x = x + z / np.tan(np.deg2rad(a1))
    if a2 not in (0.0, 90.0) and a2 == a2:
        z = z + x * np.tan(np.deg2rad(a2))
    fr = (R + x).ravel()
    fz = (Z + z).ravel()
    w = np.full(fr.size, 1.0 / fr.size)
    return fr, fz, w


def coil_psi_matrix(gR, gZ, elements, cols, col_order, n=None):
    """psi on grid (len(gZ)*len(gR), row-major Z,R) per unit current (A) of each column.

    elements: list of (R,Z,W,H,a1,a2); cols: per-element input column name.
    Each element carries current I_col / n_elements(col) (ampere-turns split over turns),
    i.e. for MAST P-coils total ampere-turns are shared across their turns. For per-turn
    currents (solenoid) multiply the result column by n_elements yourself.
    Returns M (Npix, len(col_order)) and the element count per column.
    """
    RR, ZZ = np.meshgrid(gR, gZ)  # (nz, nr)
    pr, pz = RR.ravel(), ZZ.ravel()
    M = np.zeros((pr.size, len(col_order)))
    counts = np.zeros(len(col_order), dtype=int)
    idx = {c: i for i, c in enumerate(col_order)}
    for (R, Z, W, H, a1, a2), c in zip(elements, cols):
        if c not in idx:
            continue
        counts[idx[c]] += 1
    for (R, Z, W, H, a1, a2), c in zip(elements, cols):
        if c not in idx:
            continue
        fr, fz, w = element_filaments(R, Z, W, H, a1, a2, n=n)
        g = greens_psi(pr[:, None], pz[:, None], fr[None, :], fz[None, :]) @ w
        M[:, idx[c]] += g / counts[idx[c]]
    return M, counts
