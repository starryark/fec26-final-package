"""Assemble submission npz files from DIII-D model predictions and MAST physics outputs.

Scalars are clipped to the DIII-D TRAIN label range (q95, betaN); MAST NaN scalars fall back
to the shot median, then to the fold median of finite predictions (inputs-only statistics).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.features import load_ds  # noqa: E402
from fec.paths import WORK  # noqa: E402
from fec.write_submission import write  # noqa: E402


def train_ranges():
    ds = load_ds("diii_d_train")
    sp = np.array(ds["info"]["splits"])[ds["shot"]] == "train"
    sc = ds["scal"][sp]
    return {"q95": (float(np.nanpercentile(sc[:, 0], 0.01)), float(np.nanpercentile(sc[:, 0], 99.99))),
            "betaN": (0.0, float(np.nanpercentile(sc[:, 1], 99.99)))}


def fill_nan_by_shot(x, shot):
    x = x.copy()
    glob = np.nanmedian(x)
    for s in np.unique(shot[~np.isfinite(x)]):
        m = shot == s
        med = np.nanmedian(x[m]) if np.isfinite(x[m]).any() else glob
        x[m & ~np.isfinite(x)] = med
    return x


def shot_running_median(x, shot, k=5):
    """Per-shot running median (window k frames) to suppress isolated contour/Thomson spikes."""
    from scipy.ndimage import median_filter
    y = x.copy()
    for s in np.unique(shot):
        m = np.flatnonzero(shot == s)
        if len(m) >= 3:
            y[m] = median_filter(x[m], size=min(k, len(m) | 1), mode="nearest")
    return y


def q95_robust(q95, q90, shot, tol=1.6):
    """Contour-failure guard: q95/q90 is a stable per-shot shaping ratio (~1.1-1.3). Frames whose
    ratio deviates > tol x from the shot median (or with non-finite q95) use q90 * median ratio."""
    q = q95.copy()
    ratio = q95 / q90
    glob = np.nanmedian(ratio)
    for s in np.unique(shot):
        m = shot == s
        r = np.nanmedian(ratio[m]) if np.isfinite(ratio[m]).any() else glob
        bad = m & np.isfinite(q90) & (~np.isfinite(ratio) |
                                      (np.abs(np.log(np.maximum(ratio, 1e-9) / r)) > np.log(tol)))
        q[bad] = r * q90[bad]
    return q


def interp_bad_by_shot(x, bad, t, shot):
    """Replace flagged frames by linear interpolation in time over the shot's good frames."""
    x = x.copy()
    glob = np.nanmedian(x[~bad])
    for s in np.unique(shot[bad]):
        m = np.flatnonzero(shot == s)
        g = m[~bad[m]]
        b = m[bad[m]]
        x[b] = np.interp(t[b], t[g], x[g]) if len(g) >= 1 else glob
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--d3d", required=True, help="preds dir name under fec_work/preds")
    ap.add_argument("--mast", required=True, help="dir under fec_work with psi.npy, scalars.npz")
    ap.add_argument("--k-beta", type=float, default=2.3)
    ap.add_argument("--split", default="public_test", choices=["public_test", "private_test"])
    ap.add_argument("--psi-dtype", default="float16", choices=["float16", "float32"],
                    help="psirz storage dtype (float32: +0.00013 DIII-D VAL S over float16, 2x size)")
    ap.add_argument("--mast-betan", default=None,
                    help="file (under the --mast dir) holding final MAST betaN per frame, used as-is "
                         "(e.g. betan_v2.npy from fec/betan_transfer.py); default 2.3 x betaN_e")
    a = ap.parse_args()
    rng = train_ranges()
    out = WORK / "submissions" / a.name
    out.mkdir(parents=True, exist_ok=True)
    # DIII-D
    pd = WORK / "preds" / a.d3d
    dcfg, mcfg = f"diii_d_{a.split}", f"mast_{a.split}"
    psi = np.load(pd / f"{dcfg}_psi.npy", mmap_mode="r")
    sc = np.load(pd / f"{dcfg}_scal.npy")
    q = np.clip(sc[:, 0], *rng["q95"]); b = np.clip(sc[:, 1], *rng["betaN"])
    write(dcfg, psi, q, b, out / f"{dcfg}.npz", psi_dtype=np.dtype(a.psi_dtype))
    # MAST
    md = WORK / a.mast
    ds = load_ds(mcfg)
    mpsi = np.load(md / "psi.npy", mmap_mode="r")
    ms = dict(np.load(md / "scalars.npz"))
    mq = q95_robust(ms["q95"], ms["q90"], ds["shot"]) if "q90" in ms else ms["q95"].copy()
    # geometry flag: boundary pressed inboard (R_geo >6 cm below the shot median) or q95>15
    rg = ms["Rgeo"]
    rmed = np.array([np.nanmedian(rg[ds["shot"] == s]) for s in range(ds["shot"].max() + 1)])
    dip = np.nan_to_num(rg, nan=0) < rmed[ds["shot"]] - 0.06
    if "blim_active" in ms:   # limiter-bounded frames: R_geo dips are physical, keep their q95
        dip &= ~(np.nan_to_num(ms["blim_active"]) > 0.5)
    bad = ~np.isfinite(mq) | (mq > 15) | dip
    mq = interp_bad_by_shot(mq, bad, ds["t_ms"], ds["shot"])
    mq = shot_running_median(mq, ds["shot"])
    if a.mast_betan:
        mb = fill_nan_by_shot(np.load(md / a.mast_betan).astype(np.float64), ds["shot"])
    else:
        mb = shot_running_median(fill_nan_by_shot(a.k_beta * ms["betaN_e"], ds["shot"]), ds["shot"])
    # MAST (spherical tokamak) may legitimately exceed DIII-D's betaN range: physical bounds only
    mq = np.clip(mq, 1.5, 20.0); mb = np.clip(mb, 0.0, 8.0)
    write(mcfg, mpsi, mq, mb, out / f"{mcfg}.npz", psi_dtype=np.dtype(a.psi_dtype))
    info = {"d3d": a.d3d, "mast": a.mast, "k_beta": a.k_beta, "mast_betan": a.mast_betan,
            "psi_dtype": a.psi_dtype,
            "ranges": rng,
            "mast_q95_nan_frac": float(np.mean(~np.isfinite(ms["q95"]))),
            "mast_betaN_nan_frac": float(np.mean(~np.isfinite(ms["betaN_e"])))}
    (out / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
