"""Per-frame official-functional evaluation of DIII-D VAL psi predictions (d3dcons agent).

For a psi array aligned with a VAL index (N,65,65 in val_idx order) this computes, per frame,
exactly what fec/score.py / local_score.score_shot computes inside its shot workers -- the
predicted LCFS (scorer mask, axis_sign = -1, i.e. psi_sign = +1), D_LCFS = Hausdorff / rgeo
against the cached reference contour, and the seven derived scalars -- and saves them so that
any subset / combination can be re-aggregated without re-running contour extraction.

Only VAL references are read (keys = shots present in the val index; refs cached by score.py).
The scorer mask is used here ONLY for evaluation, never for modelling.

  python -B fec/cons_frames.py --psi <val_psi.npy> --out <frames.npz> [--idx <val_idx.npy>]
         [--every k] [--nproc 6] [--fp16] [--fast]
  (--fast: numba/vectorised scorer helpers from fec/cons_fast.py, verified bit-identical;
   the gate evaluation uses the unpatched official path)
  python -B fec/cons_frames.py --agg <frames.npz> [--psi <val_psi.npy> --scal <val_scal.npy>]

`aggregate()` reproduces finalize_machine (pooled R2 per derived scalar about the VAL GT mean,
nan predictions mean-substituted, D_LCFS = mean over shots of the per-shot frame mean).
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
from fec import score as SC  # noqa: E402
from fec.features import load_ds  # noqa: E402
from fec.paths import WORK  # noqa: E402

LS = SC.LS
NAMES = list(LS.CONS_SCALARS)
DEFAULT_IDX = WORK / "models" / "d3d_b_ix_C2_fp32" / "val_idx.npy"


def frame_scalars(psi, R, Z, mc, mf, ref_c=None, rgeo=np.nan):
    """(d, cons7) of one predicted frame with the scorer's own extraction (eval only)."""
    ex, _ = LS.extract_lcfs_with_sign(psi, R, Z, "DIII-D", mc, mf, n_iter=LS.N_ITER,
                                      axis_sign=-1.0)
    d = np.nan
    if ref_c is not None:
        d = 1.0 if ex is None else min(1.0, LS.symmetric_hausdorff(
            ex, np.asarray(ref_c, dtype=np.float64)) / rgeo)
    vals = LS.derive_frame(psi, R, Z, "DIII-D", mc, mf, contour=ex, with_li=True, axis_sign=-1.0)
    return d, np.array([vals[n] for n in NAMES], np.float64), ex


def _worker(args):
    key, psi_pred, fast = args
    if fast:                                   # bit-identical vectorised helpers (cons_fast.check)
        from fec.cons_fast import patch
        patch()
    R, Z, mc, mf = SC._mask()
    contours, cons_gt, cmask, rgeo = SC._load_ref(key)
    T = psi_pred.shape[0]
    d = np.full(T, np.nan)
    cp = np.full((T, 7), np.nan)
    for k in range(T):
        d[k], cp[k], _ = frame_scalars(psi_pred[k], R, Z, mc, mf, contours[k], rgeo)
    return key, d, np.where(cmask, cons_gt, np.nan), cp


def run(psi_path, out, idx_path=DEFAULT_IDX, every=1, nproc=6, fp16=False, fast=False):
    psi = np.load(psi_path, mmap_mode="r")
    idx = np.load(idx_path)
    assert psi.shape[0] == len(idx)
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])
    shots = ds["shot"][idx]
    assert (splits[np.unique(shots)] == "val").all(), "only VAL shots may be evaluated"
    order = ds["info"]["order"]
    jobs, pos = [], []
    for s in np.unique(shots)[::every]:
        j = np.flatnonzero(shots == s)
        p = np.asarray(psi[j], dtype=np.float32)
        if fp16:
            p = p.astype(np.float16)
        jobs.append((order[s], p.astype(np.float64), fast))
        pos.append(j)
    t0 = time.time()
    with Pool(nproc) as pool:
        res = pool.map(_worker, jobs, chunksize=1)
    P = np.concatenate(pos)
    np.savez(out, pos=P, gidx=idx[P], shot=shots[P],
             dlcfs=np.concatenate([r[1] for r in res]),
             cons_gt=np.concatenate([r[2] for r in res]),
             cons_pred=np.concatenate([r[3] for r in res]))
    print(f"{len(P)} frames / {len(jobs)} shots in {time.time() - t0:.0f}s -> {out}", flush=True)
    return out


def r2_cons(cg, cp):
    """Pooled R2 per derived scalar exactly as finalize_machine (nan pred -> GT mean)."""
    out = np.full(7, np.nan)
    for j in range(7):
        ok = np.isfinite(cg[:, j])
        g = cg[ok, j]
        mu = g.mean()
        p = np.where(np.isfinite(cp[ok, j]), cp[ok, j], mu)
        out[j] = 1.0 - ((g - p) ** 2).sum() / ((g - mu) ** 2).sum()
    return out


def dlcfs(shot, d):
    vals = [np.nanmean(d[shot == s]) for s in np.unique(shot) if np.isfinite(d[shot == s]).any()]
    return float(np.mean(vals))


def aggregate(fr, psi=None, scal=None, idx_path=DEFAULT_IDX, verbose=True):
    """Composite S from a frames npz (dict-like) + optional psi/scal arrays (full VAL order)."""
    r2c = r2_cons(fr["cons_gt"], fr["cons_pred"])
    cons = float(np.mean(np.maximum(0.0, r2c)))
    dl = dlcfs(fr["shot"], fr["dlcfs"])
    out = {"consistency": cons, "dlcfs": dl, "r2_cons_each": dict(zip(NAMES, r2c.tolist()))}
    if psi is not None:
        ds = load_ds("diii_d_train")
        idx = np.load(idx_path)
        pos = np.sort(fr["pos"])
        sse = tot_s = tot_ss = 0.0; n = 0
        for i in range(0, len(pos), 4000):
            p = pos[i:i + 4000]
            g = np.asarray(ds["psi"][idx[p]], np.float64)
            q = np.asarray(psi[p], np.float64)
            sse += float(((g - q) ** 2).sum()); tot_s += g.sum(); tot_ss += (g ** 2).sum()
            n += g.size
        r2p = 1 - sse / (tot_ss - tot_s ** 2 / n)
        r2qb = np.mean([1 - ((ds["scal"][idx[pos], j] - scal[pos, j]) ** 2).sum()
                        / ((ds["scal"][idx[pos], j] - ds["scal"][idx[pos], j].mean()) ** 2).sum()
                        for j in (0, 1)])
        out.update(r2_psi=r2p, r2_qb=float(r2qb),
                   S=.55 * r2p + .15 * r2qb + .10 * (1 - dl) + .20 * cons)
    if verbose:
        print(" ".join(f"{k}={v:.6f}" for k, v in out.items() if not isinstance(v, dict)))
        print("   ", " ".join(f"{k}={v:.4f}" for k, v in out["r2_cons_each"].items()))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi"); ap.add_argument("--out"); ap.add_argument("--idx", default=str(DEFAULT_IDX))
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--agg")
    ap.add_argument("--fast", action="store_true", help="use fec/cons_fast.py patched helpers")
    ap.add_argument("--scal")
    a = ap.parse_args()
    if a.agg:
        fr = np.load(a.agg)
        psi = np.load(a.psi, mmap_mode="r") if a.psi else None
        scal = np.load(a.scal) if a.scal else None
        aggregate(fr, psi, scal, a.idx)
    else:
        run(a.psi, a.out, a.idx, a.every, a.nproc, a.fp16, a.fast)
        aggregate(np.load(a.out))
