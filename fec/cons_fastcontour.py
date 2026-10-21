"""Numba re-implementation of skimage.measure.find_contours (default options: no mask,
fully_connected='low', positive_orientation='low') for the d3dcons refinement code.

Segment generation follows skimage's marching-squares case table; assembly follows
skimage's `_assemble_contours` (starts/ends maps, append / appendleft / join-by-creation-order
/ close), with points identified by grid-edge id instead of float tuples (identical unless a
grid value equals the level exactly). `check()` compares against skimage on VAL psi frames at
random levels and at the levels the LCFS bisection visits; `patch()` installs it as
lcfs.find_contours in the current process only (scorer files untouched; the official
evaluation never imports this module).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _frac(a, b, level):
    if b == a:
        return 0.0
    return (level - a) / (b - a)


@njit(cache=True)
def _segments(arr, level):
    nr, nc = arr.shape
    nh = nr * (nc - 1)
    maxseg = 2 * (nr - 1) * (nc - 1)
    seg_id = np.empty((maxseg, 2), np.int64)
    seg_xy = np.empty((maxseg, 4), np.float64)
    n = 0
    for r0 in range(nr - 1):
        for c0 in range(nc - 1):
            r1 = r0 + 1; c1 = c0 + 1
            ul = arr[r0, c0]; ur = arr[r0, c1]; ll = arr[r1, c0]; lr = arr[r1, c1]
            if np.isnan(ul) or np.isnan(ur) or np.isnan(ll) or np.isnan(lr):
                continue
            case = 0
            if ul > level:
                case += 1
            if ur > level:
                case += 2
            if ll > level:
                case += 4
            if lr > level:
                case += 8
            if case == 0 or case == 15:
                continue
            # point ids/coords: top, bottom, left, right
            t_id = r0 * (nc - 1) + c0; t_r = float(r0); t_c = c0 + _frac(ul, ur, level)
            b_id = r1 * (nc - 1) + c0; b_r = float(r1); b_c = c0 + _frac(ll, lr, level)
            l_id = nh + r0 * nc + c0; l_r = r0 + _frac(ul, ll, level); l_c = float(c0)
            g_id = nh + r0 * nc + c1; g_r = r0 + _frac(ur, lr, level); g_c = float(c1)
            # (from, to) pairs per case; 6 and 9 emit two segments (vertex_connect_high=False)
            for k in range(2):
                f = -1; tt = -1
                if case == 1 and k == 0:
                    f = 0; tt = 2
                elif case == 2 and k == 0:
                    f = 3; tt = 0
                elif case == 3 and k == 0:
                    f = 3; tt = 2
                elif case == 4 and k == 0:
                    f = 2; tt = 1
                elif case == 5 and k == 0:
                    f = 0; tt = 1
                elif case == 6:
                    if k == 0:
                        f = 3; tt = 0
                    else:
                        f = 2; tt = 1
                elif case == 7 and k == 0:
                    f = 3; tt = 1
                elif case == 8 and k == 0:
                    f = 1; tt = 3
                elif case == 9:
                    if k == 0:
                        f = 0; tt = 2
                    else:
                        f = 1; tt = 3
                elif case == 10 and k == 0:
                    f = 1; tt = 0
                elif case == 11 and k == 0:
                    f = 1; tt = 2
                elif case == 12 and k == 0:
                    f = 2; tt = 3
                elif case == 13 and k == 0:
                    f = 0; tt = 3
                elif case == 14 and k == 0:
                    f = 2; tt = 0
                if f < 0:
                    continue
                for side in range(2):
                    p = f if side == 0 else tt
                    if p == 0:
                        pid = t_id; pr = t_r; pc = t_c
                    elif p == 1:
                        pid = b_id; pr = b_r; pc = b_c
                    elif p == 2:
                        pid = l_id; pr = l_r; pc = l_c
                    else:
                        pid = g_id; pr = g_r; pc = g_c
                    seg_id[n, side] = pid
                    seg_xy[n, 2 * side] = pr; seg_xy[n, 2 * side + 1] = pc
                n += 1
    return seg_id[:n], seg_xy[:n]


@njit(cache=True)
def _assemble(seg_id, seg_xy, npts):
    nseg = seg_id.shape[0]
    starts = -np.ones(npts, np.int64)      # point id -> contour
    ends = -np.ones(npts, np.int64)
    maxn = 2 * nseg + 2
    nxt = -np.ones(maxn, np.int64)
    nr_ = np.empty(maxn, np.float64); nc_ = np.empty(maxn, np.float64); npid = np.empty(maxn, np.int64)
    hnode = np.empty(nseg + 1, np.int64); tnode = np.empty(nseg + 1, np.int64)
    alive = np.zeros(nseg + 1, np.bool_); length = np.zeros(nseg + 1, np.int64)
    nn = 0; nc = 0
    for s in range(nseg):
        fp = seg_id[s, 0]; tp = seg_id[s, 1]
        fr = seg_xy[s, 0]; fc = seg_xy[s, 1]; tr = seg_xy[s, 2]; tc = seg_xy[s, 3]
        if fr == tr and fc == tc:
            continue
        tail = starts[tp]; starts[tp] = -1
        head = ends[fp]; ends[fp] = -1
        if tail >= 0 and head >= 0:
            if tail == head:
                nr_[nn] = tr; nc_[nn] = tc; npid[nn] = tp
                nxt[tnode[head]] = nn; tnode[head] = nn; nn += 1; length[head] += 1
            else:
                if tail > head:                       # creation number == contour index
                    nxt[tnode[head]] = hnode[tail]; tnode[head] = tnode[tail]
                    length[head] += length[tail]; alive[tail] = False
                    starts[npid[hnode[head]]] = head; ends[npid[tnode[head]]] = head
                else:
                    nxt[tnode[head]] = hnode[tail]; hnode[tail] = hnode[head]
                    length[tail] += length[head]
                    starts[npid[hnode[head]]] = -1
                    alive[head] = False
                    starts[npid[hnode[tail]]] = tail; ends[npid[tnode[tail]]] = tail
        elif tail < 0 and head < 0:
            c = nc; nc += 1
            nr_[nn] = fr; nc_[nn] = fc; npid[nn] = fp
            nr_[nn + 1] = tr; nc_[nn + 1] = tc; npid[nn + 1] = tp
            nxt[nn] = nn + 1
            hnode[c] = nn; tnode[c] = nn + 1; nn += 2
            alive[c] = True; length[c] = 2
            starts[fp] = c; ends[tp] = c
        elif head < 0:
            nr_[nn] = fr; nc_[nn] = fc; npid[nn] = fp
            nxt[nn] = hnode[tail]; hnode[tail] = nn; nn += 1; length[tail] += 1
            starts[fp] = tail
        else:
            nr_[nn] = tr; nc_[nn] = tc; npid[nn] = tp
            nxt[tnode[head]] = nn; tnode[head] = nn; nn += 1; length[head] += 1
            ends[tp] = head
    # flatten alive contours in creation order
    tot = 0
    for c in range(nc):
        if alive[c]:
            tot += length[c]
    out = np.empty((tot, 2), np.float64)
    offs = np.zeros(nc + 1, np.int64)
    k = 0; m = 0
    for c in range(nc):
        if not alive[c]:
            continue
        node = hnode[c]
        for _ in range(length[c]):
            out[k, 0] = nr_[node]; out[k, 1] = nc_[node]; k += 1
            node = nxt[node]
        m += 1
        offs[m] = k
    return out, offs[:m + 1]


def find_contours(arr, level):
    a = np.ascontiguousarray(arr, dtype=np.float64)
    sid, sxy = _segments(a, float(level))
    npts = a.shape[0] * (a.shape[1] - 1) + (a.shape[0] - 1) * a.shape[1]
    out, offs = _assemble(sid, sxy, npts)
    return [out[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]


_ORIG = {}


def patch():
    import lcfs
    if "fc" not in _ORIG:
        _ORIG["fc"] = lcfs.find_contours
    lcfs.find_contours = find_contours


def unpatch():
    import lcfs
    if "fc" in _ORIG:
        lcfs.find_contours = _ORIG["fc"]


def check(n=200, seed=0):
    import sys
    import time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from skimage.measure import find_contours as skfc
    from fec.paths import WORK
    psi = np.load(WORK / "models" / "d3d_b_ix_C2_fp32" / "val_psi.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(psi), n, replace=False))
    bad = 0; tot = 0; t_sk = t_nb = 0.0
    find_contours(np.asarray(psi[0], np.float64), -0.1)          # compile
    for i in sel:
        phi = -np.asarray(psi[i], np.float64)
        for lev in np.r_[rng.uniform(phi.min(), phi.max(), 8),
                         phi.max() - (phi.max() - phi.min()) * 2.0 ** -np.arange(1, 12)]:
            t0 = time.perf_counter(); A = skfc(phi, lev); t_sk += time.perf_counter() - t0
            t0 = time.perf_counter(); B = find_contours(phi, lev); t_nb += time.perf_counter() - t0
            tot += 1
            if len(A) != len(B) or any(not np.array_equal(a, b) for a, b in zip(A, B)):
                bad += 1
    print(f"{tot} (frame, level) cases: mismatches {bad}; skimage {t_sk / tot * 1e3:.2f} ms, "
          f"numba {t_nb / tot * 1e3:.3f} ms per call")
    return bad == 0


if __name__ == "__main__":
    check()
