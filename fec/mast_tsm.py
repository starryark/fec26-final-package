"""Within-shot temporal smoothing of a MAST candidate's psi maps (post-processing node `tsm`).

Owner mastpost, 2026-10-10 (docs/research/MAST_PIPELINE_20261009.md sec. 8).  Exact port of root's
scratch fec_work/root_var/tsm_maps.py, which made sub29 (sub09 zsc maps + sigma 30 ms; Dev MAST
0.7644 vs 0.7396) and sub30 (sub23 avg6 + sigma 30 ms):

* Gaussian in t_ms within each shot (a contiguous run of one shot id); the weights are
  renormalised per row, so frames near the shot edges use a one-sided window;
* float64 tensordot over the shot's frames, one float32 rounding at the store;
* production meta (psi_ax, psi_b, Rax, Zax, limited, method = 1) recomputed on the smoothed maps
  with V3Solver(mast_setup(), limiter=None) find_axis / find_xpoints / boundary(final=True), in
  batches of whole shots of >= 2048 frames (the scratch batching: the float32 GPU solver depends on
  the batch shape at the ULP level, so the batches are part of the reproduction);
* scalars.npz, the betaN file, betan_records.npz and the betaN json are COPIED unchanged from the
  source node: on the DIII-D testbed, recomputing q95/betaN on smoothed maps hurt while the
  psi-derived terms improved (FINAL_GENERALIZATION_20261009 / testbed re-check 2026-10-09).

Inputs: the source node's psi.npy and the fold's shot / t_ms (fec.features.load_ds(config)).
No MAST label, demo parquet or scorer mask is read.

    python -B fec/mast_tsm.py --config mast_public_test|mast_private_test --src DIR --out DIR \
        --sigma-ms 30 [--device cuda|cpu] [--n N] [--betan betan_v2.npy]

The pipeline node (fec/mast_pipeline.py recipe key "post": [{"kind": "tsm", ...}]) runs this CLI
under fec/gpu_slot.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIGS = ("mast_public_test", "mast_private_test")
META_KEYS = ("psi_ax", "psi_b", "Rax", "Zax", "limited")
BATCH = 2048                     # tsm_maps.py: flush the meta batch once >= 2048 frames are queued
KIND = "temporal smoothing of psi"


def copy_names(betan="betan_v2.npy"):
    """Files copied unchanged from the source node (scalars of the raw maps, frozen betaN)."""
    return ("scalars.npz", betan, "betan_records.npz", Path(betan).stem + ".json")


def _sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def shot_segments(shot):
    """(starts, ends) of the contiguous runs of equal shot id (tsm_maps.py)."""
    shot = np.asarray(shot)
    n = len(shot)
    b = np.flatnonzero(np.diff(shot) != 0) + 1
    return np.r_[0, b], np.r_[b, n]


def gaussian_weights(t, sigma_ms):
    """Row-normalised Gaussian weights (n, n) of one shot's frame times (float64)."""
    tt = np.asarray(t, dtype=np.float64)
    w = np.exp(-0.5 * ((tt[:, None] - tt[None, :]) / float(sigma_ms)) ** 2)
    w /= w.sum(1, keepdims=True)
    return w


def solver_meta_fn(device="cuda"):
    """Production-format meta of a float32 map batch: V3Solver axis, X-points and boundary flux
    (no solver limiter, final=True) -- the tsm_maps.py flush()."""
    import torch
    from fec.machines import mast_setup
    from fec.mast_select_v3 import V3Solver
    spec, M = mast_setup()
    S = V3Solver(spec, M, device=device, limiter=None)

    def fn(x):
        with torch.no_grad():
            ps = torch.as_tensor(x, device=S.dev, dtype=S.dt)
            pa, Ra, Za = S.find_axis(ps)
            xp = S.find_xpoints(ps, Ra, Za, pa)
            pb, lim = S.boundary(ps, pa, Ra, Za, xp, final=True)
        return {k: v.detach().cpu().numpy() for k, v in zip(META_KEYS, (pa, pb, Ra, Za, lim))}
    return fn


def smooth_maps(src_dir, out_dir, shot, t_ms, sigma_ms, meta_fn=None, device="cuda",
                batch=BATCH, betan="betan_v2.npy", config=None, n=None, log=print):
    """Smooth src_dir/psi.npy within shots (Gaussian sigma_ms in t_ms) into out_dir/psi.npy,
    recompute meta.npz on the smoothed maps, copy the scalar/betaN files of src_dir unchanged.

    shot / t_ms: the fold's per-frame arrays (at least as long as the maps; the maps are a frame
    prefix of the fold).  meta_fn(x float32 (B,65,65)) -> {psi_ax, psi_b, Rax, Zax, limited};
    default the V3Solver on `device`.  Returns a summary dict (also written to config.json)."""
    src, out = Path(src_dir), Path(out_dir)
    sig = float(sigma_ms)
    if not (np.isfinite(sig) and sig > 0):
        raise ValueError(f"sigma_ms must be a positive number, not {sigma_ms!r}")
    if src.resolve() == out.resolve():
        raise ValueError("out_dir must differ from src_dir")
    copies = copy_names(betan)
    miss = [f for f in copies if not (src / f).exists()]
    if miss:
        raise FileNotFoundError(f"{src} lacks {miss}: finish the source node first (stage 2 + "
                                f"betaN); tsm copies its scalars and betaN unchanged")
    scfg = {}
    if (src / "config.json").exists():
        try:
            scfg = json.loads((src / "config.json").read_text())
        except ValueError:
            scfg = {}
    if config and scfg.get("config") not in (None, config):
        raise ValueError(f"{src} was made for {scfg.get('config')}, not {config}")
    P = np.load(src / "psi.npy", mmap_mode="r")
    N = P.shape[0]
    if n is not None and N != int(n):
        raise ValueError(f"{src} has {N} frames, the run has {n}")
    shot, t_all = np.asarray(shot), np.asarray(t_ms)
    if len(shot) < N or len(t_all) < N:
        raise ValueError(f"fold arrays ({len(shot)}, {len(t_all)} frames) shorter than the "
                         f"{N} maps")
    shot = shot[:N]
    t = t_all[:N].astype(np.float64)
    if not np.isfinite(t).all():
        raise ValueError("t_ms has non-finite entries")
    solver_device = None if meta_fn is not None else device
    meta_fn = meta_fn or solver_meta_fn(device)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "psi.part.npy"
    psi_out = np.lib.format.open_memmap(tmp, "w+", np.float32, P.shape)
    meta = {k: np.full(N, np.nan) for k in META_KEYS}
    meta["method"] = np.ones(N)
    starts, ends = shot_segments(shot)
    t0 = time.time()
    buf_idx, buf = [], []

    def flush():
        if not buf_idx:
            return
        idx = np.concatenate(buf_idx)
        x = np.concatenate(buf).astype(np.float32)
        r = meta_fn(x)
        for k in META_KEYS:
            meta[k][idx] = r[k]
        buf_idx.clear()
        buf.clear()

    n_acc, n_batches = 0, 0
    for a, e in zip(starts, ends):
        w = gaussian_weights(t[a:e], sig)
        x = np.tensordot(w, np.asarray(P[a:e], dtype=np.float64), axes=1).astype(np.float32)
        psi_out[a:e] = x
        buf_idx.append(np.arange(a, e))
        buf.append(x)
        n_acc += e - a
        if n_acc >= batch:
            flush()
            n_acc, n_batches = 0, n_batches + 1
            log(f"{e}/{N} {time.time() - t0:.0f}s")
    if buf_idx:
        n_batches += 1
    flush()
    psi_out.flush()
    del psi_out
    os.replace(tmp, out / "psi.npy")
    np.savez(out / "meta.npz", **meta)
    copied = {}
    for f in copies:
        shutil.copy2(src / f, out / f)
        h = _sha256(out / f)
        if h != _sha256(src / f):
            raise RuntimeError(f"copy of {src / f} differs from its source")
        copied[f] = h
    info = dict(kind=KIND, config=config or scfg.get("config"), N=int(N), src=str(src),
                sigma_ms=sig, shots=int(len(starts)), batch=int(batch), meta_batches=n_batches,
                device=solver_device,
                scalars="copied from src (raw maps)", copied=copied,
                code="fec/mast_tsm.py (port of fec_work/root_var/tsm_maps.py)",
                seconds=round(time.time() - t0, 1))
    (out / "config.json").write_text(json.dumps(info, indent=1))
    log("TSM_DONE")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, choices=CONFIGS)
    ap.add_argument("--src", required=True, help="finished source node (psi, meta, scalars, betaN)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-ms", type=float, required=True)
    ap.add_argument("--device", default="cuda", help="V3Solver device of the meta recompute")
    ap.add_argument("--n", type=int, default=None, help="expected frame count of the maps")
    ap.add_argument("--betan", default="betan_v2.npy")
    ap.add_argument("--batch", type=int, default=BATCH)
    a = ap.parse_args(argv)
    from fec.features import load_ds
    ds = load_ds(a.config)
    smooth_maps(a.src, a.out, ds["shot"], ds["t_ms"], a.sigma_ms, device=a.device,
                batch=a.batch, betan=a.betan, config=a.config, n=a.n,
                log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
