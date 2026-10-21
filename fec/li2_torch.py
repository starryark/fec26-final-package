"""Differentiable scorer li(2) at a FIXED contour (torch) and the TRAIN target/contour cache of the li2ft line.

Agent li2ft, 2026-10-12. Pre-declaration: docs/research/FEC_COMPETITION_20261007.md, block "Pre-declaration, DIII-D
li(2)-loss fine-tune li2ft" (R23 rank 1). Detail doc: docs/archive/fec/D3D_LI2FT_20261012.md.

The scorer (fusion_scoring/derive.py::internal_inductance) computes, for a map psi and an LCFS polygon C (512 points),
    li(2) = (num / V) / (G / lp)^2,
    num = sum_{inside} Bp^2 dV,  V = sum_{inside} dV,  dV = 2 pi R mean(dR) mean(dZ),
    Bp = |np.gradient(psi, Z, R)| / R,  G = sum_i 0.5 (Bp(C_i) + Bp(C_{i+1})) |C_{i+1} - C_i|  (Bp bilinear at C),
    lp = sum_i |C_{i+1} - C_i|,  inside = even-odd ray cast of the grid nodes in C.
With C (hence the inside mask, the bilinear weights and the segment lengths) held FIXED, li(2) is a smooth function of
psi: `li2()` below evaluates it batched in torch (any device/dtype) with autograd. Held at the TRUE map's scorer
contour it is the training term of li2ft; the contour is a training-time target structure only, never an input.

np.gradient on the (possibly float-non-uniform) grid is reproduced exactly by fec.train_d3d_bx.fd_matrix (second-
order non-uniform central differences inside, first-order one-sided at the edges); np.interp + floor/clip of
derive._bilinear by `contour_weights`. In float64 the result equals derive.internal_inductance to ~1e-15 relative
(tests/test_fec_li2_torch.py, real TRAIN frames).

Cache (`build`): for every frame of the requested splits (default TRAIN only; VAL/TEST rows are never read), from the
FLOAT32 per-shot label cache WORK/shots/diii_d_train/<stem>.npz (the ds psi.npy cache is float16):
    C   = the scorer's own LCFS routine on the true map, reference-builder call (lcfs.extract_lcfs, 512 points,
          22 bisection steps, axis_sign None), with the TRAIN-only envelope WORK/ds_b/diii_d_train/env_train.npz
          (mask_coarse = env, mask_f = env as float; the label-free convention of fec/cons_project.py and fec/xli.py).
          The scorer's own mask (fusion_scoring/masks/d3d_envelope.npz) is NOT used: modelling never reads scorer masks.
    y   = derive.internal_inductance(psi_true, C) with the scorer's plausibility clamp (0.1 <= li <= 5, else invalid).
    inside = the scorer's inside mask of C on the 65x65 grid (bit-packed, 529 bytes).
Files in --out: gidx.npy (n,) int64 global ds frame index (sorted), y.npy (n,) float64 (nan = invalid), valid.npy
(n,) bool, C.npy (n,512,2) float32 (zeros when invalid), inside.npy (n,529) uint8, meta.json.

  python -B fec/li2_torch.py build --out DIR [--nproc 4] [--splits train] [--limit_shots N]
  python -B fec/li2_torch.py check --cache DIR [--n 300]       torch li2 at the cached contour vs y (float64, CPU)
  python -B fec/li2_torch.py compare --a DIR --b DIR           two builds of the cache agree (pod vs local)
  python -B fec/li2_torch.py merge --inputs DIR1,DIR2 --out DIR  disjoint caches -> one (all-data: train + val,test)
--splits train,val,test (all labelled rows; the all-data Final fine-tune, li2ft Amendment 3) needs the one-time TEST
evaluation marker test_eval_b/DONE (FEC_WORK or the TRAIN work root), as train_d3d_b --alldata.
Run with FEC_WORK=<idx root> (<FEC_IDX_WORK> locally, <POD_WORKSPACE>/fec_work_idx on a pod).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NPTS = 512
NGRID = 65
MASK_BYTES = (NGRID * NGRID + 7) // 8          # 529
LI_RANGE = (0.1, 5.0)                          # derive.PLAUSIBLE["li"]


# ----------------------------------------------------------------------------- torch: li(2) at a fixed contour
def fd_matrix(x):
    """np.gradient(f, x) along axis 0 as a matrix (= fec.train_d3d_bx.fd_matrix; imported lazily there to avoid an
    import cycle: train_d3d_bx imports this module in its fine-tune mode)."""
    from fec.train_d3d_bx import fd_matrix as _fd
    return _fd(x)


class Li2Grid:
    """Constants of li2() on the (Z rows, R columns) grid, on one device/dtype."""

    def __init__(self, R, Z, device="cpu", dtype=torch.float64):
        R = np.asarray(R, np.float64); Z = np.asarray(Z, np.float64)
        assert R.shape == (NGRID,) and Z.shape == (NGRID,)
        self.R_np, self.Z_np = R, Z
        self.device, self.dtype = device, dtype
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)  # noqa: E731
        self.GZ = t(fd_matrix(Z))                          # d/dZ along rows
        self.GRt = t(fd_matrix(R).T)                       # d/dR along columns (right-multiplied)
        self.invR2 = t(1.0 / R ** 2)
        dR = float(np.mean(np.diff(R))); dZ = float(np.mean(np.diff(Z)))
        RR = np.broadcast_to(R[None, :], (NGRID, NGRID))
        self.dV = t(2.0 * np.pi * RR * dR * dZ)            # derive.internal_inductance's dV
        self.R = t(R); self.Z = t(Z)
        self.ar = torch.arange(NGRID, device=device)


def bp2(psi, g):
    """Bp^2 = |np.gradient(psi, Z, R)|^2 / R^2 on (..., 65, 65)."""
    return (torch.matmul(g.GZ, psi).square() + torch.matmul(psi, g.GRt).square()) * g.invR2


def _interp_index(x, xp):
    """np.interp(x, xp, arange(n)) for increasing xp (torch, same dtype as x)."""
    n = xp.shape[0]
    j = (torch.searchsorted(xp, x.contiguous(), right=True) - 1).clamp(0, n - 2)
    f = j.to(x.dtype) + (x - xp[j]) / (xp[j + 1] - xp[j])
    f = torch.where(x <= xp[0], torch.zeros_like(f), f)
    return torch.where(x >= xp[-1], torch.full_like(f, n - 1), f)


def contour_weights(C, g):
    """Fixed-contour constants of derive._bilinear + the segment lengths. C (..., P, 2) physical (R, Z).
    Returns idx (..., P, 4) flat node indices, w (..., P, 4) bilinear weights, seg (..., P) |C_{i+1} - C_i| (closing
    segment included), lp (...,)."""
    C = C.to(g.dtype)
    fr = _interp_index(C[..., 0], g.R)
    fz = _interp_index(C[..., 1], g.Z)
    r0 = torch.floor(fr).long().clamp(0, NGRID - 2)
    z0 = torch.floor(fz).long().clamp(0, NGRID - 2)
    tr = fr - r0.to(fr.dtype); tz = fz - z0.to(fz.dtype)
    i00 = z0 * NGRID + r0
    idx = torch.stack([i00, i00 + 1, i00 + NGRID, i00 + NGRID + 1], -1)
    w = torch.stack([(1 - tr) * (1 - tz), tr * (1 - tz), (1 - tr) * tz, tr * tz], -1)
    seg = torch.sqrt((torch.roll(C, -1, dims=-2) - C).square().sum(-1))
    return idx, w, seg, seg.sum(-1)


def li2(psi, inside, cw, g, eps=0.0, parts=False):
    """Scorer li(2) of psi (..., 65, 65) at a fixed contour.

    inside: (..., 65, 65) bool/float, broadcastable to psi (the scorer's inside mask of the fixed contour);
    cw: contour_weights(C, g) with leading dims broadcastable to psi's (e.g. (B,...) for psi (M, B, ...));
    eps: Bp^2 floor before the square root (0 = exact; > 0 keeps the gradient finite at a node with Bp = 0).
    Returns li (...) [, dict(num, V, G, lp)]. Non-finite where V = 0 or G = 0 (the caller masks those frames)."""
    idx, w, seg, lp = cw
    B2 = bp2(psi, g)
    ins = inside.to(B2.dtype)
    num = (B2 * ins * g.dV).sum((-2, -1))
    V = (ins * g.dV).sum((-2, -1))                                              # broadcasts with num
    Bp = torch.sqrt(B2.clamp_min(eps) if eps > 0 else B2).flatten(-2)          # (..., 4225)
    lead = Bp.shape[:-1]
    ii = idx.expand(lead + idx.shape[-2:]).reshape(lead + (-1,))
    Bc = (torch.gather(Bp, -1, ii).view(lead + idx.shape[-2:]) * w).sum(-1)    # (..., P)
    G = (0.5 * (Bc + torch.roll(Bc, -1, dims=-1)) * seg).sum(-1)
    li = (num / V) / (G / lp).square()
    if parts:
        return li, {"num": num, "V": V, "G": G, "lp": lp.expand_as(G)}
    return li


def unpack_inside(packed):
    """(..., 529) uint8 (np.packbits, big bit order) -> (..., 65, 65) bool (torch)."""
    sh = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> sh) & 1
    return bits.reshape(packed.shape[:-1] + (-1,))[..., :NGRID * NGRID].view(packed.shape[:-1] + (NGRID, NGRID)).bool()


# ----------------------------------------------------------------------------- numpy: scorer pieces (cache build)
_ENV = None


def env(fast=True):
    """(R, Z, mask_coarse, mask_f) of the TRAIN-only envelope (env_train.npz), as fec/cons_project.env(); `fast`
    applies the bit-identical vectorised scorer helpers (fec/cons_fast.py, needs numba) when available."""
    global _ENV
    if _ENV is None:
        from fec import score as SC  # noqa: F401  (puts the starter + fusion_scoring on sys.path)
        from fec.paths import WORK
        patched = False
        if fast:
            try:
                from fec.cons_fast import patch
                patch(); patched = True
            except Exception as ex:                    # numba missing: the unpatched official helpers (slower)
                print(f"li2_torch: cons_fast unavailable ({type(ex).__name__}); unpatched scorer helpers", flush=True)
        e = np.load(WORK / "ds_b" / "diii_d_train" / "env_train.npz")
        mc = e["env"].astype(bool)
        _ENV = (e["grid_R"].astype(np.float64), e["grid_Z"].astype(np.float64), mc, mc.astype(np.float64), patched)
    return _ENV


def frame_target(psi, R, Z, mc, mf):
    """Scorer contour, inside mask and li(2) of one TRUE frame (float64 math). Returns (C (512,2) float64 | None,
    inside (65,65) bool | None, y float (nan if invalid), reason str)."""
    from fec import score as SC
    import derive as DV
    p = np.asarray(psi, np.float64)
    C = SC.LS.extract_lcfs(p, R, Z, "DIII-D", mc, mf, n_points=NPTS)
    if C is None or len(C) < 8:
        return None, None, np.nan, "no_contour"
    RR, ZZ = np.meshgrid(R, Z)
    inside = DV._points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
    li = DV.internal_inductance(p, R, Z, "DIII-D", C)
    if not np.isfinite(li):
        return C, inside, np.nan, "li_nan"
    if not (LI_RANGE[0] <= li <= LI_RANGE[1]):
        return C, inside, np.nan, "li_clamp"
    return C, inside, float(li), "ok"


def _build_worker(args):
    key, fidx, src = args
    R, Z, mc, mf, _ = env()
    with np.load(Path(src) / f"{key}.npz") as z:
        P = np.asarray(z["psi"][fidx], np.float32)
        gR, gZ = z["grid_R"], z["grid_Z"]
    assert np.allclose(gR, R) and np.allclose(gZ, Z), key
    n = len(fidx)
    C = np.zeros((n, NPTS, 2), np.float32); ins = np.zeros((n, MASK_BYTES), np.uint8)
    y = np.full(n, np.nan); why = []
    for k in range(n):
        c, m, v, r = frame_target(P[k], R, Z, mc, mf)
        why.append(r)
        if r == "ok":
            C[k] = c; ins[k] = np.packbits(m.ravel()); y[k] = v
    return key, C, ins, y, why


def _content_md5(arrs):
    h = hashlib.md5()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}|{a.shape}|".encode()); h.update(a.tobytes())
    return h.hexdigest()


def test_done():
    """The one-time internal-TEST evaluation marker (train_d3d_b --alldata's condition): WORK/test_eval_b/DONE, or the
    TRAIN work root's (fec.paths.TRAIN_WORK; on this machine the marker lives in fec_work, not in the idx root)."""
    from fec.paths import TRAIN_WORK, WORK
    return any((r / "test_eval_b" / "DONE").exists() for r in (WORK, TRAIN_WORK))


def cmd_build(a):
    from multiprocessing import Pool
    from fec.features import load_ds
    from fec.paths import WORK
    want = set(a.splits.split(","))
    if not want <= {"train", "val", "test"}:
        raise SystemExit(f"--splits {a.splits}")
    if "test" in want and not test_done():
        raise SystemExit("--splits with test refused: test_eval_b/DONE missing in FEC_WORK and in the TRAIN work "
                         "root (as train_d3d_b --alldata)")
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])
    fsplit = splits[ds["shot"]]
    gidx = np.flatnonzero(np.isin(fsplit, sorted(want)))       # only rows of the requested splits are read
    shots = ds["shot"][gidx]
    us = np.unique(shots)
    if a.limit_shots:
        us = np.random.default_rng(0).choice(us, a.limit_shots, replace=False)
        us.sort()
        keep = np.isin(shots, us)
        gidx, shots = gidx[keep], shots[keep]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = len(gidx)
    R, Z, mc, mf, patched = env()
    src = str(WORK / "shots" / "diii_d_train")
    print(f"li2 cache: {n} frames of {len(us)} shots (splits {sorted(want)}) -> {out}; nproc {a.nproc}; "
          f"fast helpers {patched}", flush=True)
    Cm = np.lib.format.open_memmap(out / "C.tmp.npy", "w+", np.float32, (n, NPTS, 2))
    Im = np.lib.format.open_memmap(out / "inside.tmp.npy", "w+", np.uint8, (n, MASK_BYTES))
    y = np.full(n, np.nan)
    reasons = {}
    pos = {}
    jobs = []
    for s in us:
        j = np.flatnonzero(shots == s)
        key = ds["info"]["order"][int(s)]
        pos[key] = j
        jobs.append((key, ds["fidx"][gidx[j]], src))
    t0 = time.time(); done = 0
    with Pool(a.nproc) as pool:
        for key, C, ins, yy, why in pool.imap_unordered(_build_worker, jobs, chunksize=2):
            j = pos[key]
            Cm[j] = C; Im[j] = ins; y[j] = yy
            for r in why:
                reasons[r] = reasons.get(r, 0) + 1
            done += 1
            if done % 250 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)} shots, {el:.0f}s, eta {el / done * (len(jobs) - done):.0f}s, "
                      f"{reasons}", flush=True)
    Cm.flush(); Im.flush()
    del Cm, Im
    (out / "C.tmp.npy").replace(out / "C.npy")
    (out / "inside.tmp.npy").replace(out / "inside.npy")
    np.save(out / "gidx.npy", gidx.astype(np.int64))
    np.save(out / "y.npy", y)
    np.save(out / "valid.npy", np.isfinite(y))
    e = np.load(WORK / "ds_b" / "diii_d_train" / "env_train.npz")
    ok = np.isfinite(y)
    meta = {"n": int(n), "n_shots": int(len(us)), "splits": sorted(want), "n_valid": int(ok.sum()), "reasons": reasons,
            "y_mean": float(np.mean(y[ok])), "y_std": float(np.std(y[ok])), "y_gt_1.7": int((y[ok] > 1.7).sum()),
            "y_max": float(np.max(y[ok])),
            "contour": "lcfs.extract_lcfs(psi_true_float32, R, Z, 'DIII-D', env, env_f, n_points=512) (reference-"
                       "builder call); li = derive.internal_inductance + the scorer clamp [0.1, 5]",
            "envelope": "WORK/ds_b/diii_d_train/env_train.npz (TRAIN-only; no scorer mask)",
            "env_content_md5": _content_md5([e["env"], e["grid_R"], e["grid_Z"]]),
            "labels": "float32 per-shot cache WORK/shots/diii_d_train/<stem>.npz",
            "grid_R": R.tolist(), "grid_Z": Z.tolist(), "fast_helpers": patched,
            "numpy": np.__version__, "seconds": time.time() - t0, "limit_shots": a.limit_shots,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("LI2_CACHE_DONE", json.dumps({k: meta[k] for k in ("n", "n_valid", "reasons", "y_std", "y_gt_1.7",
                                                              "seconds")}), flush=True)


# ----------------------------------------------------------------------------- cache access
class Li2Cache:
    """Memory-mapped cache (see module doc)."""

    def __init__(self, d):
        d = Path(d)
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text())
        self.gidx = np.load(d / "gidx.npy")
        self.y = np.load(d / "y.npy")
        self.valid = np.load(d / "valid.npy")
        self.C = np.load(d / "C.npy", mmap_mode="r")
        self.inside = np.load(d / "inside.npy", mmap_mode="r")
        assert len(self.gidx) == len(self.y) == len(self.C) == len(self.inside)

    def positions(self, global_idx):
        """Cache position of each global ds frame index, -1 if not cached."""
        gi = np.asarray(global_idx)
        p = np.searchsorted(self.gidx, gi)
        p = np.clip(p, 0, len(self.gidx) - 1)
        return np.where(self.gidx[p] == gi, p, -1)


def cmd_check(a):
    """torch li2 (float64, CPU) at the cached contour (float32-stored and the float64 re-extraction) vs y."""
    from fec.features import load_ds
    from fec.paths import WORK
    c = Li2Cache(a.cache)
    R, Z, mc, mf, _ = env()
    g = Li2Grid(R, Z)
    ds = load_ds("diii_d_train")
    rng = np.random.default_rng(1)
    ok = np.flatnonzero(c.valid)
    hi = ok[c.y[ok] > 1.7]
    pick = np.unique(np.r_[rng.choice(ok, min(a.n, len(ok)), replace=False), hi[:min(len(hi), a.n // 3)]])
    rel32, rel64 = [], []
    src = WORK / "shots" / "diii_d_train"
    for p in pick:
        gi = int(c.gidx[p]); s = int(ds["shot"][gi])
        with np.load(src / f"{ds['info']['order'][s]}.npz") as z:
            psi = np.asarray(z["psi"][ds["fidx"][gi]], np.float64)
        ins = unpack_inside(torch.as_tensor(np.asarray(c.inside[p]))).numpy()
        C32 = np.asarray(c.C[p], np.float64)
        v32 = float(li2(torch.as_tensor(psi), torch.as_tensor(ins), contour_weights(torch.as_tensor(C32), g), g))
        C64, ins64, y64, _ = frame_target(psi, R, Z, mc, mf)
        assert (ins64 == ins).all()
        v64 = float(li2(torch.as_tensor(psi), torch.as_tensor(ins64), contour_weights(torch.as_tensor(C64), g), g))
        rel32.append(abs(v32 / c.y[p] - 1)); rel64.append(abs(v64 / c.y[p] - 1))
    rep = {"frames": int(len(pick)), "max_rel_err_C64": float(np.max(rel64)), "max_rel_err_C32": float(np.max(rel32)),
           "median_rel_err_C32": float(np.median(rel32))}
    print("LI2_CHECK", json.dumps(rep), flush=True)
    return rep


def cmd_compare(a):
    """Two builds of the cache (e.g. pod vs local): same frames, valid flags, masks; |dy| and |dC| bounds."""
    A, B = Li2Cache(a.a), Li2Cache(a.b)
    rep = {"gidx_equal": bool(np.array_equal(A.gidx, B.gidx))}
    if rep["gidx_equal"]:
        rep["valid_equal"] = bool(np.array_equal(A.valid, B.valid))
        both = A.valid & B.valid
        rep["max_abs_dy"] = float(np.max(np.abs(A.y[both] - B.y[both]))) if both.any() else 0.0
        nd = 0; mx = 0.0
        for i in range(0, len(A.gidx), 50000):
            nd += int((np.asarray(A.inside[i:i + 50000]) != np.asarray(B.inside[i:i + 50000])).any(1).sum())
            mx = max(mx, float(np.abs(np.asarray(A.C[i:i + 50000]) - np.asarray(B.C[i:i + 50000])).max()))
        rep["inside_frames_differ"] = nd
        rep["max_abs_dC"] = mx
    rep["ok"] = bool(rep["gidx_equal"] and rep.get("valid_equal") and rep.get("max_abs_dy", 1) <= 1e-9
                     and rep.get("inside_frames_differ", 1) == 0)
    print("LI2_COMPARE", json.dumps(rep), flush=True)
    return 0 if rep["ok"] else 3


def cmd_merge(a):
    """Merge disjoint caches (e.g. the TRAIN build + a VAL/TEST build) into one cache sorted by global frame index. Every
    frame's entry is computed independently, so the result equals a single build over the union of the splits."""
    ins = [Li2Cache(d) for d in a.inputs.split(",")]
    m0 = ins[0].meta
    for c in ins[1:]:
        for k in ("grid_R", "grid_Z", "env_content_md5", "contour", "envelope", "labels"):
            if c.meta.get(k) != m0.get(k):
                raise SystemExit(f"cannot merge: meta {k} differs ({c.dir} vs {ins[0].dir})")
    gidx = np.concatenate([c.gidx for c in ins])
    if len(np.unique(gidx)) != len(gidx):
        raise SystemExit("cannot merge: the caches overlap")
    order = np.argsort(gidx, kind="stable")
    n = len(gidx)
    dest = np.empty(n, np.int64)
    dest[order] = np.arange(n)                         # output position of each concatenated row
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Cm = np.lib.format.open_memmap(out / "C.tmp.npy", "w+", np.float32, (n, NPTS, 2))
    Im = np.lib.format.open_memmap(out / "inside.tmp.npy", "w+", np.uint8, (n, MASK_BYTES))
    y = np.full(n, np.nan)
    off = 0
    for c in ins:
        k = len(c.gidx)
        for i in range(0, k, 50000):
            j = dest[off + i: off + min(k, i + 50000)]
            Cm[j] = np.asarray(c.C[i:i + 50000])
            Im[j] = np.asarray(c.inside[i:i + 50000])
        y[dest[off:off + k]] = c.y
        off += k
    Cm.flush(); Im.flush()
    del Cm, Im
    (out / "C.tmp.npy").replace(out / "C.npy")
    (out / "inside.tmp.npy").replace(out / "inside.npy")
    gs = gidx[order]
    np.save(out / "gidx.npy", gs.astype(np.int64))
    np.save(out / "y.npy", y)
    ok = np.isfinite(y)
    np.save(out / "valid.npy", ok)
    reasons = {}
    for c in ins:
        for r, v in c.meta.get("reasons", {}).items():
            reasons[r] = reasons.get(r, 0) + v
    meta = dict(m0)
    meta.update(n=int(n), n_shots=int(sum(c.meta.get("n_shots", 0) for c in ins)),
                splits=sorted({s_ for c in ins for s_ in c.meta.get("splits", [])}), n_valid=int(ok.sum()),
                reasons=reasons, y_mean=float(np.mean(y[ok])), y_std=float(np.std(y[ok])),
                **{"y_gt_1.7": int((y[ok] > 1.7).sum())}, y_max=float(np.max(y[ok])),
                merged_from=[{"dir": str(c.dir), "n": int(len(c.gidx)), "splits": c.meta.get("splits"),
                              "built_utc": c.meta.get("built_utc")} for c in ins],
                built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("LI2_MERGE_DONE", json.dumps({k: meta[k] for k in ("n", "n_valid", "splits", "reasons", "y_std")}),
          flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--out", required=True)
    b.add_argument("--nproc", type=int, default=4); b.add_argument("--splits", default="train")
    b.add_argument("--limit_shots", type=int, default=0)
    c = sp.add_parser("check"); c.add_argument("--cache", required=True); c.add_argument("--n", type=int, default=300)
    m = sp.add_parser("compare"); m.add_argument("--a", required=True); m.add_argument("--b", required=True)
    g = sp.add_parser("merge"); g.add_argument("--inputs", required=True, help="comma-separated cache dirs")
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        return cmd_merge(a)
    if a.cmd == "build":
        return cmd_build(a)
    if a.cmd == "check":
        cmd_check(a); return 0
    return cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
