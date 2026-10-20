"""Consolidate per-shot npz into frame-level arrays + frozen whole-shot split.

Split (frozen 2026-10-07 before any validation): VAL/TEST are drawn only from DIII-D train
shots never used by earlier studies (exposed list = stems cited in research/ outside
catalog). u = sha256(stem + SALT) -> [0,1): u<0.06 VAL, 0.06<=u<0.12 TEST, else TRAIN.

    python -B fec/dataset.py <config> [--psi-dtype float16|float32]

--psi-dtype (labels-bearing configs only; default float16 = every existing cache, byte-identical):
float32 stores the per-shot float32 psi labels unrounded in ds/<config>/psi.npy (2x the size;
agent k256 2026-10-12, gate K256f32 in docs/research/FEC_COMPETITION_20261007.md). Every other
output file is byte-identical between the two; a float32 build adds "psi_dtype" to info.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

SALT = "fec-2026-10-07-split"
CORE_ZG = np.linspace(0.0, 0.95, 20)      # DIII-D vertical core chord Z grid
EDGE_RG_D3D = np.linspace(1.67, 2.27, 12)  # DIII-D tangential (edge) chord R grid
MAST_CORE_RG = np.linspace(0.24, 1.50, 43)
MAST_EDGE_RG = np.linspace(1.31, 1.48, 9)


def split_of(stem, exposed):
    if stem in exposed:
        return "train"
    u = int(hashlib.sha256((stem + SALT).encode()).hexdigest()[:12], 16) / 16 ** 12
    return "val" if u < 0.06 else ("test" if u < 0.12 else "train")


def profile_on_grid(pos, vals, grid):
    """vals (T,N) at positions pos (N,) -> (T,len(grid)) linear interp; NaN outside/missing."""
    T = vals.shape[0]
    out = np.full((T, len(grid)), np.nan, dtype=np.float32)
    ok_pos = np.isfinite(pos)
    if ok_pos.sum() < 2:
        return out
    order = np.argsort(pos[ok_pos])
    p = pos[ok_pos][order]
    v = vals[:, ok_pos][:, order]
    for k in range(T):
        m = np.isfinite(v[k])
        if m.sum() < 2:
            continue
        pk, vk = p[m], v[k][m]
        o = np.interp(grid, pk, vk)
        o[(grid < pk[0] - 0.03) | (grid > pk[-1] + 0.03)] = np.nan
        out[k] = o
    return out


def thomson_features(z, machine):
    """Fixed-size Thomson profile features: Te [keV], ne [1e19 m^-3] on fixed grids."""
    if machine == "DIII-D":
        cpos, cg = z["ts_core_Z"], CORE_ZG
        epos, eg = z["ts_edge_R"], EDGE_RG_D3D
    else:
        cpos, cg = z["ts_core_R"], MAST_CORE_RG
        epos, eg = z["ts_edge_R"], MAST_EDGE_RG
    te_c = profile_on_grid(cpos, z["ts_core_te"].astype(np.float64) / 1e3, cg)
    ne_c = profile_on_grid(cpos, z["ts_core_ne"].astype(np.float64) / 1e19, cg)
    te_e = profile_on_grid(epos, z["ts_edge_te"].astype(np.float64) / 1e3, eg)
    ne_e = profile_on_grid(epos, z["ts_edge_ne"].astype(np.float64) / 1e19, eg)
    return np.concatenate([te_c, ne_c, te_e, ne_e], axis=1)


PSI_DTYPES = {"float16": np.float16, "float32": np.float32}


def build(config, with_targets, psi_dtype="float16"):
    psi_dt = PSI_DTYPES[psi_dtype]
    src = WORK / "shots" / config
    meta = json.loads((src / "_meta.json").read_text())
    order = meta["order"]
    machine = "DIII-D" if config.startswith("diii") else "MAST"
    exposed = set(json.loads((WORK / "exposed_train_stems.json").read_text())) \
        if config == "diii_d_train" else set()
    out = WORK / "ds" / config
    out.mkdir(parents=True, exist_ok=True)
    # first pass: sizes
    Ts = []
    for stem in order:
        with np.load(src / f"{stem}.npz") as z:
            Ts.append(len(z["t"]))
    N = int(np.sum(Ts))
    print(config, "frames", N, flush=True)
    cur = np.lib.format.open_memmap(out / "cur.npy", "w+", np.float32,
                                    (N, len(meta["channels"]), len(meta["offsets_ms"])))
    nts = (len(CORE_ZG) * 2 + len(EDGE_RG_D3D) * 2) if machine == "DIII-D" else \
        (len(MAST_CORE_RG) * 2 + len(MAST_EDGE_RG) * 2)
    ts = np.lib.format.open_memmap(out / "ts.npy", "w+", np.float32, (N, nts))
    t_ms = np.zeros(N, np.float64)
    shot = np.zeros(N, np.int32)
    fidx = np.zeros(N, np.int32)
    if with_targets:
        psi = np.lib.format.open_memmap(out / "psi.npy", "w+", psi_dt, (N, 65, 65))
        scal = np.zeros((N, 5), np.float32)  # q95, betan, li, raxis, zaxis
    splits = []
    i0 = 0
    t0 = time.time()
    for si, stem in enumerate(order):
        with np.load(src / f"{stem}.npz") as z:
            T = len(z["t"])
            sl = slice(i0, i0 + T)
            cur[sl] = z["cur"]
            ts[sl] = thomson_features(z, machine)
            t_ms[sl] = z["t"]
            shot[sl] = si
            fidx[sl] = np.arange(T)
            if with_targets:
                psi[sl] = z["psi"].astype(psi_dt)
                scal[sl] = np.stack([z["q95"], z["betan"], z["li"], z["raxis"], z["zaxis"]], 1)
        splits.append(split_of(stem[9:], exposed) if config == "diii_d_train" else "test")
        i0 += T
        if si % 500 == 0:
            print(si, f"{time.time() - t0:.0f}s", flush=True)
    cur.flush(); ts.flush()
    np.save(out / "t_ms.npy", t_ms)
    np.save(out / "shot.npy", shot)
    np.save(out / "fidx.npy", fidx)
    if with_targets:
        psi.flush()
        np.save(out / "scal.npy", scal)
    info = {"config": config, "N": N, "order": order, "splits": splits,
            "channels": meta["channels"], "offsets_ms": meta["offsets_ms"], "salt": SALT,
            "n_split": {k: splits.count(k) for k in set(splits)}}
    if with_targets and psi_dtype != "float16":
        info["psi_dtype"] = psi_dtype          # float16 builds keep the historical info.json bytes
    (out / "info.json").write_text(json.dumps(info))
    print("done", info["n_split"], f"{time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="frame-level arrays + frozen split for one config")
    ap.add_argument("config")
    ap.add_argument("--psi-dtype", dest="psi_dtype", choices=sorted(PSI_DTYPES), default="float16",
                    help="dtype of the stored psi labels (labels-bearing configs; default float16)")
    a = ap.parse_args(argv)
    with_targets = a.config == "diii_d_train"
    if a.psi_dtype != "float16" and not with_targets:
        raise SystemExit(f"--psi-dtype applies to the labels-bearing config only (diii_d_train), not {a.config}")
    build(a.config, with_targets, a.psi_dtype)


if __name__ == "__main__":
    main()
