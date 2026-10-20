"""Submission builder of the Final candidates with FROZEN DIII-D TRAIN clip ranges (agent fpkg
2026-10-12, docs/archive/fec/FINAL_PACKAGE_20261012.md).

fec/build_submission.py computes the q95/betaN clip ranges from the DIII-D TRAIN labels at build
time (fec.build_submission.train_ranges: 0.01/99.99 percentiles of the TRAIN-split frames), so an
inference-only rebuild would need the 13 GB TRAIN cache for two numbers.  This builder is the same
arithmetic -- it calls build_submission's own helper functions and fec.write_submission.write --
with the ranges read from a frozen JSON (fec/final_train_ranges.json, written by `freeze` from the
TRAIN labels and checked against train_ranges() whenever the TRAIN cache is present).

    python -B fec/final_build.py build --name OUT_DIR --d3d PREDS_DIR --mast MAST_DIR
        [--split private_test] [--mast-betan betan_v2.npy] [--psi-dtype float32]
        [--ranges fec/final_train_ranges.json]
    python -B fec/final_build.py freeze [--out fec/final_train_ranges.json]   (needs TRAIN ds)
    python -B fec/final_build.py check  [--ranges ...]                        (needs TRAIN ds)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

RANGES = Path(__file__).resolve().parent / "final_train_ranges.json"


def load_ranges(path=None):
    r = json.loads(Path(path or RANGES).read_text())
    return {"q95": tuple(float(x) for x in r["q95"]), "betaN": tuple(float(x) for x in r["betaN"])}


def freeze(out=None):
    """Write the TRAIN clip ranges (fec.build_submission.train_ranges) + provenance."""
    from fec.build_submission import train_ranges
    from fec.features import load_ds
    rng = train_ranges()
    ds = load_ds("diii_d_train")
    sp = np.array(ds["info"]["splits"])[ds["shot"]] == "train"
    rec = dict(q95=list(rng["q95"]), betaN=list(rng["betaN"]),
               rule="q95: TRAIN-frame 0.01/99.99 percentiles; betaN: [0, 99.99 percentile] "
                    "(fec.build_submission.train_ranges)",
               n_train_frames=int(sp.sum()),
               scal_sha256=hashlib.sha256(np.ascontiguousarray(ds["scal"]).tobytes()).hexdigest(),
               info_sha256=hashlib.sha256((WORK / "ds" / "diii_d_train" / "info.json")
                                          .read_bytes()).hexdigest())
    Path(out or RANGES).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))
    return rec


def check(path=None):
    from fec.build_submission import train_ranges
    a, b = load_ranges(path), train_ranges()
    ok = all(tuple(a[k]) == tuple(b[k]) for k in ("q95", "betaN"))
    print(json.dumps(dict(ok=ok, frozen=a, recomputed=b)))
    return ok


def build(name, d3d, mast, split="public_test", mast_betan="betan_v2.npy", psi_dtype="float32",
          ranges=None, k_beta=2.3):
    """fec.build_submission.main with frozen ranges (same helpers, same order of operations)."""
    from fec.build_submission import (fill_nan_by_shot, interp_bad_by_shot, q95_robust,
                                      shot_running_median)
    from fec.features import load_ds
    from fec.write_submission import write
    rng = load_ranges(ranges)
    out = Path(name) if Path(name).is_absolute() else WORK / "submissions" / name
    out.mkdir(parents=True, exist_ok=True)
    pd = Path(d3d) if Path(d3d).is_absolute() else WORK / "preds" / d3d
    dcfg, mcfg = f"diii_d_{split}", f"mast_{split}"
    psi = np.load(pd / f"{dcfg}_psi.npy", mmap_mode="r")
    sc = np.load(pd / f"{dcfg}_scal.npy")
    q = np.clip(sc[:, 0], *rng["q95"]); b = np.clip(sc[:, 1], *rng["betaN"])
    write(dcfg, psi, q, b, out / f"{dcfg}.npz", psi_dtype=np.dtype(psi_dtype))
    md = Path(mast) if Path(mast).is_absolute() else WORK / mast
    ds = load_ds(mcfg)
    mpsi = np.load(md / "psi.npy", mmap_mode="r")
    ms = dict(np.load(md / "scalars.npz"))
    mq = q95_robust(ms["q95"], ms["q90"], ds["shot"]) if "q90" in ms else ms["q95"].copy()
    rg = ms["Rgeo"]
    rmed = np.array([np.nanmedian(rg[ds["shot"] == s]) for s in range(ds["shot"].max() + 1)])
    dip = np.nan_to_num(rg, nan=0) < rmed[ds["shot"]] - 0.06
    if "blim_active" in ms:
        dip &= ~(np.nan_to_num(ms["blim_active"]) > 0.5)
    bad = ~np.isfinite(mq) | (mq > 15) | dip
    mq = interp_bad_by_shot(mq, bad, ds["t_ms"], ds["shot"])
    mq = shot_running_median(mq, ds["shot"])
    if mast_betan:
        mb = fill_nan_by_shot(np.load(md / mast_betan).astype(np.float64), ds["shot"])
    else:
        mb = shot_running_median(fill_nan_by_shot(k_beta * ms["betaN_e"], ds["shot"]), ds["shot"])
    mq = np.clip(mq, 1.5, 20.0); mb = np.clip(mb, 0.0, 8.0)
    write(mcfg, mpsi, mq, mb, out / f"{mcfg}.npz", psi_dtype=np.dtype(psi_dtype))
    info = {"d3d": str(d3d), "mast": str(mast), "k_beta": k_beta, "mast_betan": mast_betan,
            "psi_dtype": psi_dtype,
            "ranges": {k: list(v) for k, v in rng.items()},
            "mast_q95_nan_frac": float(np.mean(~np.isfinite(ms["q95"]))),
            "mast_betaN_nan_frac": float(np.mean(~np.isfinite(ms["betaN_e"]))),
            "ranges_source": str(Path(ranges or RANGES).name), "code": "fec/final_build.py"}
    (out / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--name", required=True)
    p.add_argument("--d3d", required=True)
    p.add_argument("--mast", required=True)
    p.add_argument("--split", default="public_test", choices=["public_test", "private_test"])
    p.add_argument("--mast-betan", default="betan_v2.npy")
    p.add_argument("--psi-dtype", default="float32", choices=["float16", "float32"])
    p.add_argument("--ranges", default=None)
    p = sub.add_parser("freeze")
    p.add_argument("--out", default=None)
    p = sub.add_parser("check")
    p.add_argument("--ranges", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.name, a.d3d, a.mast, a.split, a.mast_betan, a.psi_dtype, a.ranges)
    elif a.cmd == "freeze":
        freeze(a.out)
    else:
        return 0 if check(a.ranges) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
