"""Faster, semantics-identical versions of three pure-Python scorer helpers (d3dcons agent).

Used ONLY inside d3dcons refinement code (fec/cons_project.py) to compute derived scalars of
candidate psi maps; the official evaluation (fec/score.py, fec/cons_frames.py) never imports
this module. `patch()` rebinds, in the current process only:
  lcfs.is_closed            (np.allclose(c[0], c[-1], atol=1e-6) on two floats)
  lcfs.encloses_point       (even-odd ray cast, vectorised; identical arithmetic incl. 1e-30)
  derive._points_in_polygon (even-odd ray cast over all edges, vectorised in edge chunks)
`check()` verifies bit-identical derived scalars against the unpatched scorer on VAL frames.
No scorer file is modified.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_ORIG = {}


def is_closed(contour_rc, tol=1e-6):
    a0, a1 = float(contour_rc[0, 0]), float(contour_rc[0, 1])
    b0, b1 = float(contour_rc[-1, 0]), float(contour_rc[-1, 1])
    # numpy.allclose(a, b, rtol=1e-05, atol=tol): |a - b| <= atol + rtol * |b|
    return abs(a0 - b0) <= tol + 1e-05 * abs(b0) and abs(a1 - b1) <= tol + 1e-05 * abs(b1)


def encloses_point(contour_rc, row, col):
    y, x = row, col
    yi = contour_rc[:, 0]; xi = contour_rc[:, 1]
    yj = np.roll(yi, 1); xj = np.roll(xi, 1)          # j = i - 1 (j = n - 1 for i = 0)
    c1 = (yi > y) != (yj > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        c2 = x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
    return bool(np.count_nonzero(c1 & c2) % 2)


@njit(cache=True)
def _pip(rq, zq, r1, z1, r2, z2):
    inside = np.zeros(rq.shape[0], np.bool_)
    for e in range(r1.shape[0]):
        za = z1[e]; zb = z2[e]
        if za == zb:
            continue
        for q in range(rq.shape[0]):
            if (za > zq[q]) != (zb > zq[q]):
                xint = r1[e] + (zq[q] - za) * (r2[e] - r1[e]) / (zb - za)
                if rq[q] < xint:
                    inside[q] = not inside[q]
    return inside


def points_in_polygon(rq, zq, C):
    r1 = np.ascontiguousarray(C[:, 0], dtype=np.float64); z1 = np.ascontiguousarray(C[:, 1], dtype=np.float64)
    r2 = np.roll(r1, -1); z2 = np.roll(z1, -1)
    rr = np.ascontiguousarray(rq, dtype=np.float64).ravel(); zz = np.ascontiguousarray(zq, dtype=np.float64).ravel()
    return _pip(rr, zz, r1, z1, r2, z2).reshape(np.shape(rq))


def patch(contours=True):
    import derive
    import lcfs
    if contours:
        from fec.cons_fastcontour import patch as _pc
        _pc()
    if not _ORIG:
        _ORIG.update(is_closed=lcfs.is_closed, encloses_point=lcfs.encloses_point,
                     pip=derive._points_in_polygon)
    lcfs.is_closed = is_closed
    lcfs.encloses_point = encloses_point
    derive._points_in_polygon = points_in_polygon


def unpatch():
    import derive
    import lcfs
    from fec.cons_fastcontour import unpatch as _uc
    _uc()
    if _ORIG:
        lcfs.is_closed = _ORIG["is_closed"]; lcfs.encloses_point = _ORIG["encloses_point"]
        derive._points_in_polygon = _ORIG["pip"]


def check(n=300, seed=0):
    import sys
    import time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fec.cons_project import derive_env
    from fec.paths import WORK
    psi = np.load(WORK / "models" / "d3d_b_ix_C2_fp32" / "val_psi.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(psi), n, replace=False))
    P = np.asarray(psi[sel], np.float64)
    unpatch(); t0 = time.time(); A = np.array([derive_env(p) for p in P]); ta = time.time() - t0
    patch(); t0 = time.time(); B = np.array([derive_env(p) for p in P]); tb = time.time() - t0
    same = (A == B) | (np.isnan(A) & np.isnan(B))
    print(f"{n} frames: identical {same.all()} (mismatch cells {int((~same).sum())}); "
          f"orig {ta / n * 1e3:.1f} ms/frame, patched {tb / n * 1e3:.1f} ms/frame")
    return same.all()


if __name__ == "__main__":
    check()
