"""All-row mtask aux target cache of the Final config final_MTfull (family doc amendment 39; agent mtpkg 2026-10-15).

The "MT" members (docs/archive/fec/D3D_MTPROD_20261014.md, D3D_MTF2_20261014.md) are --alldata (train + val + test rows
of the frozen whole-shot split; the TEST split was evaluated once, 2026-10-08), and fec/train_d3d_bx.py's aux_context
needs a target row for every frame it trains on.  fec/mtask_aux.py build refuses li2 caches with TEST rows (a guard of
the TRAIN-only mtask gate), so production assembled its cache (fec_work/mtprod/aux_all, aux md5 93fe30ea..., gidx md5
581929b9...) from the gate cache's TRAIN + VAL rows and recomputed TEST rows.  A clean rebuild has no gate cache: this
module computes EVERY row from the all-data li(2) cache (train.py --stage li2cache: <idx>/li2/cache_all, the contours
the li2ft fine-tune trains on) with fec/mtask_aux.py's own per-shot `_worker`, i.e. the same per-frame arithmetic
(shape from the li2 cache's scorer contour; flux-surface-averaged J_phi of the float32 label).  The targets are
training-only (never model inputs, never used at inference).  On the research machine the result is byte-identical to
production's cache (tests/test_fec_mtfull_pkg.py; the package detail doc docs/archive/fec/FINAL_MTFULL_PKG_20261015.md).

  FEC_WORK=<idx work root> python -B fec/mtask_aux_all.py build --li2 <li2 cache_all> --out <dir> [--nproc 8]
  FEC_WORK=<idx work root> python -B fec/mtask_aux.py check --aux <dir> --li2 <li2 cache_all> --n 300

Files (as fec/mtask_aux.py build): gidx.npy (n,) int64 = every diii_d_train ds frame in order, aux.npy (n, 20) float32
(NaN = masked), meta.json (written last: names, heads, per-split counts / NaN counts / mean / sd, aux_md5, gidx_md5,
source li2 cache).  Marker: MTASK_AUX_ALL_DONE {json}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fec import mtask_aux as MA  # noqa: E402


def shot_jobs(ds, gl, li2_dir, src, R, Z, pos=None):
    """MA._worker jobs, one per shot, for the li2-cache positions `pos` (default: every row): (key, fidx, positions,
    li2 dir, label dir, R, Z) and {key: positions}."""
    pos = np.arange(len(gl)) if pos is None else np.asarray(pos)
    g = gl[pos]
    shots = ds["shot"][g]
    jobs, posd = [], {}
    for s in np.unique(shots):
        j = np.flatnonzero(shots == s)
        key = ds["info"]["order"][int(s)]
        posd[key] = pos[j]
        jobs.append((key, ds["fidx"][g[j]], pos[j], str(li2_dir), str(src), R, Z))
    return jobs, posd


def compute(jobs, posd, n, nproc, log_every=500):
    """(n, 20) float64 aux rows (NaN where no job wrote) from MA._worker, nproc processes (1 = in-process)."""
    A = np.full((n, len(MA.NAMES)), np.nan)
    t1, done = time.time(), 0
    if nproc <= 1:
        it = map(MA._worker, jobs)
        pool = None
    else:
        from multiprocessing import Pool
        pool = Pool(nproc)
        it = pool.imap_unordered(MA._worker, jobs, chunksize=4)
    try:
        for key, rows in it:
            A[posd[key]] = rows
            done += 1
            if done % log_every == 0 or done == len(jobs):
                el = time.time() - t1
                print(f"  {done}/{len(jobs)} shots {el:.0f}s eta {el / done * (len(jobs) - done):.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return A


def write_cache(out, G, A, splits_of_rows, extra):
    """gidx.npy (int64), aux.npy (float32), then meta.json (last, with the md5s)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    G = np.asarray(G, np.int64)
    A = np.asarray(A, np.float32)
    np.save(out / "gidx.npy", G)
    np.save(out / "aux.npy", A)
    stats = {}
    for sp in sorted(set(splits_of_rows)):
        m = splits_of_rows == sp
        X = A[m].astype(np.float64)
        stats[sp] = {"n": int(m.sum()), "n_nan": np.isnan(X).sum(0).tolist(),
                     "mean": np.nanmean(X, 0).tolist(), "sd": np.nanstd(X, 0).tolist()}
    meta = {"names": list(MA.NAMES), "heads": MA.HEADS, "definition": MA.__doc__.split("Files written")[0].strip(),
            "labels": "float32 per-shot cache WORK/shots/diii_d_train/<stem>.npz", "mu0": MA.MU0,
            "axis_sign": MA.AXIS_SIGN, "n": int(len(G)), "stats": stats,
            "aux_md5": MA._md5_file(out / "aux.npy"), "gidx_md5": MA._md5_file(out / "gidx.npy"),
            "numpy": np.__version__, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta.update(extra)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def cmd_build(a):
    from fec.features import load_ds
    from fec.paths import WORK
    t0 = time.time()
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])
    e = np.load(WORK / "ds_b" / "diii_d_train" / "env_train.npz")
    R = e["grid_R"].astype(np.float64)
    Z = e["grid_Z"].astype(np.float64)
    src = WORK / "shots" / "diii_d_train"
    li2 = Path(a.li2)
    lm = json.loads((li2 / "meta.json").read_text())
    if sorted(lm["splits"]) != ["test", "train", "val"]:
        raise SystemExit(f"{li2}: the all-row aux cache needs the all-data li2 cache (splits train,val,test), "
                         f"got {lm['splits']}")
    gl = np.load(li2 / "gidx.npy")
    if not np.array_equal(gl, np.arange(len(ds["shot"]))):
        raise SystemExit(f"{li2}: gidx must hold every diii_d_train ds frame in order ({len(gl)} rows, "
                         f"{len(ds['shot'])} frames)")
    jobs, posd = shot_jobs(ds, gl, li2, src, R, Z)
    print(f"mtask_aux_all: {len(gl)} rows, {len(jobs)} shots, nproc {a.nproc}", flush=True)
    A = compute(jobs, posd, len(gl), a.nproc)
    fs = splits[ds["shot"][gl]]
    meta = write_cache(a.out, gl, A, fs, {
        "production": "final_MTfull package (agent mtpkg, amendment 39): every labelled diii_d_train frame "
                      "(train+val+test) from the all-data li2 cache with fec/mtask_aux._worker",
        "sources": {"li2": li2.name, "li2_splits": lm["splits"], "li2_n": lm.get("n"),
                    "li2_built_utc": lm.get("built_utc"), "li2_env_content_md5": lm.get("env_content_md5"),
                    "li2_gidx_md5": MA._md5_file(li2 / "gidx.npy")},
        "seconds": time.time() - t0})
    print("MTASK_AUX_ALL_DONE", json.dumps({"n": meta["n"], "seconds": round(meta["seconds"]),
                                            "aux_md5": meta["aux_md5"], "gidx_md5": meta["gidx_md5"],
                                            "nan_frac": {sp: float(np.mean(np.array(v["n_nan"]) / max(v["n"], 1)))
                                                         for sp, v in meta["stats"].items()}}), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--li2", required=True, help="the all-data li2 cache dir (fec/li2_torch.py build --splits "
                                                "train,val,test)")
    b.add_argument("--out", required=True)
    b.add_argument("--nproc", type=int, default=4)
    a = ap.parse_args(argv)
    return {"build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
