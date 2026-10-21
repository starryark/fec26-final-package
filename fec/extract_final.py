"""fec/extract.py with the organisers' Thomson time rule for the Final inputs (agent fpkg 2026-10-12).

Starter issue #12 (organiser comment 2026-10-11): per Thomson system, keep a sample only if its
time is later than EVERY earlier sample's (running maximum); the same mask applies to Te and ne;
DIII-D zeros are missing.  No filtering of t <= 0 and no truncation at the first backward step
(the Final MAST case has valid samples after the bad one).  fec/extract.ts_interp instead sorts the
samples by time and keeps out-of-order ones.

This module runs fec.extract unchanged except that ts_interp first applies the running-maximum
mask (`ts_interp_runmax`); everything else (box means, Ip time axis, shapes, the output layout,
zeros/NaN = missing, the 25 ms gap rule) is fec.extract's own code.  fec/run_final.py uses it for
the PRIVATE configs by default (`--ts-rule auto`); the public configs keep fec/extract.py, the
features every model was trained and validated on.  Public-data effect: FINAL_PACKAGE_20261012
(counts of shots and frames where the two rules differ).

    python -B fec/extract_final.py <config> [--workers 6] [--ip-timebase idx|starter] [--limit N]
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
import fec.extract as EX  # noqa: E402
from fec.paths import WORK, shot_files  # noqa: E402

RULE = "running_max"
_ORIG = EX.ts_interp


def running_max_keep(times):
    """True where the sample time is later than every earlier sample time (NaN never kept;
    NaN times do not raise the running maximum)."""
    t = np.asarray(times, dtype=np.float64)
    if t.size == 0:
        return np.zeros(0, bool)
    prev = np.fmax.accumulate(np.r_[-np.inf, t[:-1]])
    return np.isfinite(t) & (t > prev)


def ts_interp_runmax(times, prof, t):
    """fec.extract.ts_interp on the running-maximum subset of the samples."""
    times = np.asarray(times, dtype=np.float64)
    prof = np.asarray(prof)
    if prof.ndim == 2 and times.size and prof.shape[0] == times.size:
        keep = running_max_keep(times)
        return _ORIG(times[keep], prof[keep], t)
    return _ORIG(times, prof, t)


def _patch():
    EX.ts_interp = ts_interp_runmax


def extract_one(args):
    """fec.extract.extract_one with the running-maximum Thomson rule (works in spawned pool
    workers: the patch is applied in the worker before the shot is processed)."""
    _patch()
    return EX.extract_one(args)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ip-timebase", choices=["idx", "starter"], default=EX.IP_TIMEBASE)
    a = ap.parse_args(argv)
    files = shot_files(a.config)
    if a.limit:
        files = files[: a.limit]
    if not files:          # fail instead of "done 0" + exit 0 (pkgfix open risk 1, as fec/extract.py D2)
        raise SystemExit(f"extract_final {a.config}: 0 parquet files; nothing written")
    outdir = WORK / "shots" / a.config
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"config": a.config, "n": len(files), "offsets_ms": EX.OFFSETS.tolist(),
            "half_ms": EX.HALF, "ip_timebase": a.ip_timebase,
            "channels": EX.D3D_CH if a.config.startswith("diii") else EX.MAST_CH,
            "order": [f.stem for f in files], "thomson_rule": RULE,
            "code": "fec/extract_final.py (fec/extract.py + issue #12 running-maximum rule)"}
    (outdir / "_meta.json").write_text(json.dumps(meta, indent=1))
    t0 = time.time()
    jobs = [(str(f), a.config, str(outdir), a.ip_timebase) for f in files]
    done = 0
    with Pool(a.workers) as pool:
        for stem, st in pool.imap_unordered(extract_one, jobs, chunksize=4):
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(jobs)} in {time.time() - t0:.0f}s ({RULE})", flush=True)


if __name__ == "__main__":
    main()
