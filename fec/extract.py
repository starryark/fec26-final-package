"""Parquet -> per-shot aligned arrays.

Inputs only are read on test configs (the test parquet has no target columns anyway).
On diii_d_train the EFIT targets are stored separately as supervision (never as inputs).

Per shot npz (WORK/shots/<config>/<stem>.npz):
  t            (T,)            efit_times [ms]
  cur          (T, C, K)       box means (+-HALF ms) of each current channel at t + OFFSETS[k]
  ts_core_te/ne (T, Nc), ts_core_R/Z (Nc,)   Thomson core, time-interpolated (NaN = missing)
  ts_edge_te/ne (T, Ne), ts_edge_R/Z (Ne,)
  grid_R, grid_Z (65,)
  [train] psi (T,65,65) f32, q95, betan, li, raxis, zaxis (T,), lcfs_r/z (T,N), lcfs_n (T,)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import DATA, HF_DATA, WORK, shot_files  # noqa: E402

OFFSETS = np.array([-100, -50, -20, -10, -5, -2, 0, 2, 5, 10, 20], dtype=np.float64)
HALF = 1.0  # ms half window for box means
TS_MAX_GAP = 25.0  # ms: Thomson sample must be this close to the EFIT time

D3D_CH = [f"magnetics_F{i}{s}" for s in "AB" for i in range(1, 10)] + [
    "magnetics_ECOILA", "magnetics_bcoil", "magnetics_plasma_current"]
MAST_CH = [f"magnetics_p{i}{s}_current" for i in range(2, 7) for s in "lu"] + [
    "magnetics_sol_current", "magnetics_tf_current", "magnetics_efps_current",
    "magnetics_plasma_current"]
TARGETS = ["efit_psirz", "efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis",
           "efit_lcfs_n", "efit_lcfs_r", "efit_lcfs_z"]


def _np(col):
    """Flatten a single-row nested list column to a numpy array of leaf values + shape."""
    c = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    arr = c
    dims = []
    while pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        offs = arr.offsets.to_numpy()
        dims.append(offs)
        arr = arr.flatten()
    return arr, dims


def col1d(t, name):
    arr, _ = _np(t.column(name))
    return arr.to_numpy(zero_copy_only=False).astype(np.float64)


def col2d(t, name):
    arr, dims = _np(t.column(name))
    v = arr.to_numpy(zero_copy_only=False).astype(np.float64)
    outer = dims[0]
    n = outer[1] - outer[0]
    if n == 0:
        return np.zeros((0, 0))
    return v.reshape(n, -1)


def col3d(t, name):
    arr, dims = _np(t.column(name))
    v = arr.to_numpy(zero_copy_only=False)
    n = dims[0][1] - dims[0][0]
    return v.reshape(n, 65, 65)


def box_means(ts, s, centers, half=HALF):
    """Mean of finite samples of s within [c-half, c+half]; linear interp fallback."""
    ok = np.isfinite(s) & np.isfinite(ts)
    ts, s = ts[ok], s[ok]
    out = np.full(centers.shape, np.nan)
    if ts.size == 0:
        return out
    order = np.argsort(ts, kind="stable")
    ts, s = ts[order], s[order]
    cs = np.concatenate([[0.0], np.cumsum(s)])
    i0 = np.searchsorted(ts, centers - half, side="left")
    i1 = np.searchsorted(ts, centers + half, side="right")
    n = i1 - i0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[i1] - cs[i0]) / n
    miss = n == 0
    if miss.any():
        out[miss] = np.interp(centers[miss], ts, s)
    return out


def ts_interp(times, prof, t):
    """Per-channel time interpolation of a Thomson profile; zeros/NaN are missing."""
    T, N = len(t), prof.shape[1] if prof.ndim == 2 else 0
    out = np.full((T, N), np.nan, dtype=np.float32)
    if N == 0 or times.size == 0:
        return out
    order = np.argsort(times, kind="stable")
    times, prof = times[order], prof[order]
    for j in range(N):
        v = prof[:, j]
        ok = np.isfinite(v) & (v > 0) & np.isfinite(times)
        if ok.sum() == 0:
            continue
        tv, vv = times[ok], v[ok]
        val = np.interp(t, tv, vv)
        k = np.clip(np.searchsorted(tv, t), 1, len(tv) - 1) if len(tv) > 1 else np.zeros(T, int)
        if len(tv) > 1:
            gap = np.minimum(np.abs(t - tv[k - 1]), np.abs(tv[k] - t))
        else:
            gap = np.abs(t - tv[0])
        val[gap > TS_MAX_GAP] = np.nan
        out[:, j] = val
    return out


def d3d_ip_times(ipt, ip, mag_t0):
    a = np.abs(ip)
    thr = 0.10 * np.nanpercentile(a, 99.5)
    breakdown = float(ipt[int(np.argmax(a > thr))])
    if -500.0 <= breakdown <= 1000.0:
        return ipt
    return float(mag_t0) + (ipt - ipt[0])


IP_TIMEBASE = "idx"   # "idx": fec.ip_timebase rule (ACCEPT with conditions: research/reviews/fec_ip_timebase_review_20261008.md); "starter": data_fixes logic. Extract into FRESH dirs: existing per-shot outputs are skipped.


def extract_one(args):
    path, config, outdir = args[:3]
    ip_tb = args[3] if len(args) > 3 else IP_TIMEBASE
    path = Path(path)
    out = Path(outdir) / (path.stem + ".npz")
    if out.exists():
        return path.stem, "skip"
    t = pq.read_table(path)
    names = set(t.column_names)
    machine = t.column("source")[0].as_py()
    et = col1d(t, "efit_times")
    mt = col1d(t, "magnetics_time")
    chans = D3D_CH if machine == "DIII-D" else MAST_CH
    q = (et[:, None] + OFFSETS[None, :]).ravel()
    cur = np.full((len(et), len(chans), len(OFFSETS)), np.nan, dtype=np.float32)
    for ci, ch in enumerate(chans):
        s = col1d(t, ch)
        if ch == "magnetics_plasma_current" and machine == "DIII-D":
            if ip_tb == "idx":
                from fec.ip_timebase import idx_ip_times
                ts = idx_ip_times(col1d(t, "magnetics_plasma_current_times"), mt[0])
            else:
                ts = d3d_ip_times(col1d(t, "magnetics_plasma_current_times"), s, mt[0])
        else:
            ts = mt
        cur[:, ci, :] = box_means(ts, s, q).reshape(len(et), len(OFFSETS))
    rec = {"t": et, "cur": cur, "grid_R": col1d(t, "efit_grid_R"),
           "grid_Z": col1d(t, "efit_grid_Z")}
    # Thomson
    cname = t.column("thomson_chord_name")[0].as_py()
    cR = col1d(t, "thomson_chord_R")
    cZ = col1d(t, "thomson_chord_Z")
    ctimes = col1d(t, "thomson_core_times")
    cte, cne = col2d(t, "thomson_core_Te"), col2d(t, "thomson_core_ne")
    etimes = col1d(t, "thomson_edge_times")
    ete, ene = col2d(t, "thomson_edge_Te"), col2d(t, "thomson_edge_ne")
    if machine == "MAST":
        coreR = col1d(t, "thomson_core_R")
        if cte.shape[1] == coreR.size + 1:
            cte, cne = cte[:, 1:], cne[:, 1:]
        coreZ = np.zeros_like(coreR)
        edgeR = col1d(t, "thomson_edge_spatial")
        edgeZ = np.zeros_like(edgeR)
    else:
        core_idx = [i for i, n in enumerate(cname) if n.startswith("TS_core")]
        edge_idx = [i for i, n in enumerate(cname) if n.startswith("TS_tangential")]
        coreR, coreZ = cR[core_idx], cZ[core_idx]
        edgeR, edgeZ = cR[edge_idx], cZ[edge_idx]
        if coreR.size != cte.shape[1]:
            coreR = np.full(cte.shape[1], np.nan); coreZ = np.full(cte.shape[1], np.nan)
        if edgeR.size != ete.shape[1]:
            edgeR = np.full(ete.shape[1], np.nan); edgeZ = np.full(ete.shape[1], np.nan)
    rec.update(ts_core_te=ts_interp(ctimes, cte, et), ts_core_ne=ts_interp(ctimes, cne, et),
               ts_core_R=coreR, ts_core_Z=coreZ,
               ts_edge_te=ts_interp(etimes, ete, et), ts_edge_ne=ts_interp(etimes, ene, et),
               ts_edge_R=edgeR, ts_edge_Z=edgeZ)
    if "efit_psirz" in names:
        rec["psi"] = col3d(t, "efit_psirz").astype(np.float32)
        rec["q95"] = col1d(t, "efit_q95")
        rec["betan"] = col1d(t, "efit_beta_n")
        rec["li"] = col1d(t, "efit_li")
        rec["raxis"] = col1d(t, "efit_r_axis")
        rec["zaxis"] = col1d(t, "efit_z_axis")
        rec["lcfs_n"] = col1d(t, "efit_lcfs_n")
        rec["lcfs_r"] = col2d(t, "efit_lcfs_r").astype(np.float32)
        rec["lcfs_z"] = col2d(t, "efit_lcfs_z").astype(np.float32)
    tmp = out.with_suffix(".part.npz")
    np.savez(tmp, **rec)
    tmp.replace(out)
    return path.stem, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ip-timebase", choices=["idx", "starter"], default=IP_TIMEBASE,
                    help="DIII-D Ip time axis: idx (long records +8192 ms) or the starter fix")
    a = ap.parse_args()
    files = shot_files(a.config)
    if a.limit:
        files = files[: a.limit]
    if not files:          # fail instead of "done 0" + exit 0 (pkgfix D2: an empty extract let train.py go on)
        raise SystemExit(f"extract {a.config}: 0 parquet files (data root {DATA / a.config}, HF mirror "
                         f"{HF_DATA / a.config}); nothing written")
    outdir = WORK / "shots" / a.config
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"config": a.config, "n": len(files), "offsets_ms": OFFSETS.tolist(), "half_ms": HALF,
            "ip_timebase": a.ip_timebase,
            "channels": D3D_CH if a.config.startswith("diii") else MAST_CH,
            "order": [f.stem for f in files]}
    (outdir / "_meta.json").write_text(json.dumps(meta, indent=1))
    t0 = time.time()
    jobs = [(str(f), a.config, str(outdir), a.ip_timebase) for f in files]
    done = 0
    with Pool(a.workers) as pool:
        for stem, st in pool.imap_unordered(extract_one, jobs, chunksize=4):
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(jobs)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
