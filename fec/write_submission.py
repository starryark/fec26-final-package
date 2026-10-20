"""Write challenge npz files from per-frame predictions in HF stream order.

Keys shot_{i:04d}_psirz (T,65,65) float16, shot_{i:04d}_q95 / _betaN (T,) float32,
i = index in the lexically sorted parquet list (verified == HF stream order, 2026-10-07).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.features import load_ds  # noqa: E402


def write(config, psi, q95, betaN, out_path, psi_dtype=np.float16):
    ds = load_ds(config)
    shot = ds["shot"]; fidx = ds["fidx"]
    n_shots = len(ds["info"]["order"])
    assert len(shot) == len(psi) == len(q95) == len(betaN)
    preds = {}
    bounds = np.searchsorted(shot, np.arange(n_shots + 1))
    assert np.all(np.diff(shot) >= 0)
    for i in range(n_shots):
        a, b = bounds[i], bounds[i + 1]
        assert np.array_equal(fidx[a:b], np.arange(b - a)), i
        p = np.asarray(psi[a:b], dtype=np.float32)
        q = np.asarray(q95[a:b], dtype=np.float32)
        bn = np.asarray(betaN[a:b], dtype=np.float32)
        if not (np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(bn).all()):
            raise ValueError(f"non-finite prediction in shot {i}")
        preds[f"shot_{i:04d}_psirz"] = p.astype(psi_dtype)
        preds[f"shot_{i:04d}_q95"] = q
        preds[f"shot_{i:04d}_betaN"] = bn
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **preds)
    meta = {"config": config, "n_shots": n_shots, "n_frames": int(len(psi)),
            "psi_dtype": np.dtype(psi_dtype).name}
    out_path.with_suffix(".json").write_text(json.dumps(meta))
    return out_path
