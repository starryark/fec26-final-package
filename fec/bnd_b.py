"""Per-frame flux levels + a TRAIN-only envelope for the shape-aware loss.

Uses only labels of 'train'/'val' shots of diii_d_train (WORK/shots npz); rows of 'test'
shots are never opened and stay NaN, except with --alldata (below: all rows).  No scorer asset
(masks) is used anywhere.

  bnd2.npy  (N,2) float32: psi_a = stored psi bilinearly sampled at EFIT's own axis
            (efit_r_axis, efit_z_axis); psi_b = median of stored psi sampled on EFIT's own
            LCFS polygon (efit_lcfs_r/z).
  bnd_x.npy (N,4) float32: (R,Z) of the lowest and the highest point of EFIT's LCFS polygon.
  env_train.npz: grid_R, grid_Z (from the shot files) and env (65,65) bool = grid cells inside
            any EFIT LCFS of a 'train'-split frame (every ENV_STRIDE-th frame), dilated by
            ENV_DILATE cells.  'val' shots do not contribute to the envelope.
These are loss weights only, never model inputs.

`python -B fec/bnd_b.py --alldata` (Final-phase only, AFTER the one-time internal-TEST evaluation;
refuses unless WORK/test_eval_b/DONE exists) also processes 'test' shots and writes
bnd2_all.npy / bnd_x_all.npy (all rows); the envelope env_train.npz is left unchanged.
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

CFG = "diii_d_train"
ENV_STRIDE = 10
ENV_DILATE = 3


def _bilinear(psi, R, Z, rq, zq):
    fr = np.interp(rq, R, np.arange(len(R)))
    fz = np.interp(zq, Z, np.arange(len(Z)))
    r0 = np.clip(np.floor(fr).astype(int), 0, len(R) - 2)
    z0 = np.clip(np.floor(fz).astype(int), 0, len(Z) - 2)
    tr = fr - r0
    tz = fz - z0
    return (psi[..., z0, r0] * (1 - tr) * (1 - tz) + psi[..., z0, r0 + 1] * tr * (1 - tz)
            + psi[..., z0 + 1, r0] * (1 - tr) * tz + psi[..., z0 + 1, r0 + 1] * tr * tz)


def _inside(rq, zq, r1, z1):
    """Even-odd ray cast of points (rq,zq) against the closed polygon (r1,z1)."""
    r2 = np.roll(r1, -1); z2 = np.roll(z1, -1)
    inside = np.zeros(rq.shape, bool)
    for e in range(len(r1)):
        za, zb = z1[e], z2[e]
        if za == zb:
            continue
        cross = (za > zq) != (zb > zq)
        xint = r1[e] + (zq - za) * (r2[e] - r1[e]) / (zb - za)
        inside ^= cross & (rq < xint)
    return inside


def _one(args):
    stem, is_train = args
    with np.load(WORK / "shots" / CFG / f"{stem}.npz") as z:
        psi = z["psi"].astype(np.float64)
        R, Z = z["grid_R"], z["grid_Z"]
        lr, lz, ln = z["lcfs_r"], z["lcfs_z"], z["lcfs_n"]
        ra, za = z["raxis"], z["zaxis"]
    T = psi.shape[0]
    out = np.full((T, 6), np.nan, np.float32)
    RR, ZZ = np.meshgrid(R, Z)
    env = np.zeros(RR.shape, bool)
    for k in range(T):
        if np.isfinite(ra[k]) and np.isfinite(za[k]):
            out[k, 0] = _bilinear(psi[k], R, Z, np.array([ra[k]]), np.array([za[k]]))[0]
        n = int(ln[k]) if np.isfinite(ln[k]) else 0
        r, zz = lr[k, :n].astype(np.float64), lz[k, :n].astype(np.float64)
        ok = np.isfinite(r) & np.isfinite(zz) & (r > 0)
        if ok.sum() < 8:
            continue
        r, zz = r[ok], zz[ok]
        out[k, 1] = np.median(_bilinear(psi[k], R, Z, r, zz))
        lo, hi = int(np.argmin(zz)), int(np.argmax(zz))
        out[k, 2:6] = [r[lo], zz[lo], r[hi], zz[hi]]
        if is_train and k % ENV_STRIDE == 0:
            env |= _inside(RR, ZZ, r, zz)
    return stem, out, env, R, Z


def _dilate(m, cells):
    m = m.copy()
    for _ in range(cells):
        e = m.copy()
        e[1:, :] |= m[:-1, :]; e[:-1, :] |= m[1:, :]
        e[:, 1:] |= m[:, :-1]; e[:, :-1] |= m[:, 1:]
        m = e
    return m


def main(nproc=6, alldata=False):
    d = WORK / "ds" / CFG
    info = json.loads((d / "info.json").read_text())
    shot = np.load(d / "shot.npy")
    order, splits = info["order"], info["splits"]
    out_dir = WORK / "ds_b" / CFG
    out_dir.mkdir(parents=True, exist_ok=True)
    bnd = np.full((len(shot), 6), np.nan, np.float32)
    starts = np.searchsorted(shot, np.arange(len(order)))
    ends = np.searchsorted(shot, np.arange(len(order)), side="right")
    if alldata and not (WORK / "test_eval_b" / "DONE").exists():
        raise SystemExit("--alldata refused: WORK/test_eval_b/DONE missing")
    keep = ("train", "val", "test") if alldata else ("train", "val")
    todo = [i for i in range(len(order)) if splits[i] in keep]
    pos = {order[i]: i for i in todo}
    env = None; grid = None
    t0 = time.time()
    jobs = [(order[i], splits[i] == "train") for i in todo]
    with Pool(nproc) as pool:
        for n, (stem, arr, e, R, Z) in enumerate(pool.imap_unordered(_one, jobs, chunksize=8)):
            i = pos[stem]
            assert ends[i] - starts[i] == len(arr), stem
            bnd[starts[i]:ends[i]] = arr
            env = e if env is None else env | e
            if grid is None:
                grid = (R, Z)
            assert np.allclose(grid[0], R) and np.allclose(grid[1], Z), stem
            if n % 500 == 0:
                print(n, f"{time.time() - t0:.0f}s", flush=True)
    if alldata:
        np.save(out_dir / "bnd2_all.npy", bnd[:, :2])
        np.save(out_dir / "bnd_x_all.npy", bnd[:, 2:6])
        print("done (alldata)", f"{time.time() - t0:.0f}s", "finite frac",
              float(np.isfinite(bnd[:, :2]).all(1).mean()), flush=True)
        return
    np.save(out_dir / "bnd2.npy", bnd[:, :2])
    np.save(out_dir / "bnd_x.npy", bnd[:, 2:6])
    envd = _dilate(env, ENV_DILATE)
    np.savez(out_dir / "env_train.npz", grid_R=grid[0], grid_Z=grid[1], env=envd,
             env_undilated=env, stride=ENV_STRIDE, dilate=ENV_DILATE)
    ok = np.isfinite(bnd[:, :2]).all(1)
    print("done", f"{time.time() - t0:.0f}s", "finite (psi_a,psi_b) frac over train/val rows",
          ok.sum() / max(1, sum(ends[i] - starts[i] for i in todo)),
          "env cells", int(env.sum()), "dilated", int(envd.sum()), flush=True)


if __name__ == "__main__":
    main(alldata="--alldata" in sys.argv[1:])
