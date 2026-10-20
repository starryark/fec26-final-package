"""Local DIII-D composite scorer: the starter's local_score functions, parallel + cached refs.

Mirrors local_score.main(): pooled R2 psi (one global sign), pooled R2 q95/betaN,
shot-mean D_LCFS, consistency over seven derived scalars.  Reference contours/scalars
for each held-out shot are cached under WORK/refs/<tag>/<stem>.pkl.
"""
from __future__ import annotations

import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import STARTER, WORK  # noqa: E402

sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "fusion_scoring"))
import local_score as LS  # noqa: E402
from common import N_CONS, N_SCALARS, PSI_SIGNS, SCALARS  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

_MASK = None


def _mask():
    global _MASK
    if _MASK is None:
        m = np.load(STARTER / "fusion_scoring" / "masks" / "d3d_envelope.npz")
        R, Z = m["grid_R"], m["grid_Z"]
        mc = m["mask_coarse"].astype(bool)
        _MASK = (R, Z, mc, mc.astype(np.float64))
    return _MASK


def _ref_worker(args):
    key, psi_gt = args
    f = WORK / "refs" / f"{key}.pkl"
    if f.exists():
        return key
    R, Z, mc, mf = _mask()
    ref = LS.build_reference(psi_gt, R, Z, mc, mf)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(ref, fh)
    tmp.replace(f)
    return key


def _load_ref(key):
    with open(WORK / "refs" / f"{key}.pkl", "rb") as fh:
        return pickle.load(fh)


def _score_worker(args):
    key, gt, pred, psi_sign, means = args
    R, Z, mc, mf = _mask()
    ref = _load_ref(key)
    return LS.score_shot(gt, ref, pred, R, Z, mc, mf, psi_sign, means)


def score(keys, gts, preds, nproc=6, verbose=True):
    """keys: unique cache keys per shot; gts: list of dict(psi,q95,betaN); preds likewise
    (psirz,q95,betaN). Returns finalize_machine result dict."""
    with Pool(nproc) as pool:
        list(pool.imap_unordered(_ref_worker, [(k, g["psi"]) for k, g in zip(keys, gts)],
                                 chunksize=2))
    refs = [_load_ref(k) for k in keys]
    psi_sum = psi_sumsq = psi_n = 0.0
    scal_sum, scal_sumsq, scal_n = np.zeros(N_SCALARS), np.zeros(N_SCALARS), np.zeros(N_SCALARS)
    cons_sum, cons_sumsq, cons_n = np.zeros(N_CONS), np.zeros(N_CONS), np.zeros(N_CONS)
    for g, ref in zip(gts, refs):
        p = g["psi"].astype(np.float64)
        psi_sum += float(p.sum()); psi_sumsq += float((p ** 2).sum()); psi_n += p.size
        for j, name in enumerate(SCALARS):
            v = np.asarray(g[name], dtype=np.float64)
            v = v[np.isfinite(v)]
            scal_sum[j] += v.sum(); scal_sumsq[j] += (v ** 2).sum(); scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum(); cons_sumsq[j] += (v ** 2).sum(); cons_n[j] += v.size
    ref_stats = dict(psi_sum=psi_sum, psi_sumsq=psi_sumsq, psi_n=psi_n, scal_sum=scal_sum,
                     scal_sumsq=scal_sumsq, scal_n=scal_n, cons_sum=cons_sum,
                     cons_sumsq=cons_sumsq, cons_n=cons_n)
    mean_psi = psi_sum / psi_n
    means = (mean_psi, np.where(scal_n > 0, scal_sum / np.maximum(scal_n, 1), 0.0),
             np.where(cons_n > 0, cons_sum / np.maximum(cons_n, 1), 0.0))
    totals = {s: 0.0 for s in PSI_SIGNS}
    for g, p in zip(gts, preds):
        rr, _, _ = LS.psi_residuals(g["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
    acc = Accum()
    acc.psi_sign = psi_sign
    jobs = [(k, g, p, psi_sign, means) for k, g, p in zip(keys, gts, preds)]
    with Pool(nproc) as pool:
        for part in pool.imap(_score_worker, jobs, chunksize=2):
            acc.add(part)
    res = finalize_machine(acc, ref_stats)
    if verbose:
        print_result(res)
    return res


def print_result(res):
    print(f"S={res['S']:.5f}  R2psi={res['r2_psi']:.6f}  R2qb={res['r2_qb']:.5f} "
          f"D_LCFS={res['dlcfs']:.5f}  cons={res['consistency']:.5f}")
    print("   ", "  ".join(f"{k}={v:.4f}" for k, v in res.get("r2_each", {}).items()
                           if v is not None),
          " | ", "  ".join(f"{k}={v:.4f}" for k, v in res["r2_cons_each"].items()
                           if v is not None))
