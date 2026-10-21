"""mast_solver2 (DESIGN §8.4): the frozen label-free li estimator (licond C57 path) on our BASE maps of one MAST fold.
fec.current_diffusion MAST job (_mast_job / read_mast, frozen variant, sha-checked) + frozen TRAIN 'alone' calibration
+ licond fill rule.  Meta: fec.amort_real.meta_from_maps numerics.  Geometry records: fec.betan_transfer.geometry with
MAST_AXIS_BOX (the definition inside our stage-2 betan_records).  Inputs only; no label, scorer or envelope.

    python -B ms2_mast_li.py --fold-dir outm/public --config mast_public_test [--nproc 32] [--device cuda]
-> fold-dir/li.npz (li = target used by stage L), fold-dir/base_meta.npz, fold-dir/li.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SYN_RANGE = (0.5724211035924996, 1.8478958463634254)   # licond/mast/m0_check.json syn TRAIN p0.5-p99.5
CLAMP = (0.625, 1.665)                                  # fec.licond_mast.CLAMP


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _geo(args):
    psis, gR, gZ = args
    from fec import betan_transfer as BT
    out = np.full((len(psis), 3), np.nan)
    for i, phi in enumerate(psis):
        try:
            g = BT.geometry(np.asarray(phi, np.float64), gR, gZ, BT.MAST_AXIS_BOX)
        except Exception:  # noqa: BLE001
            g = None
        if g is not None:
            out[i] = (g["a"], g["kappa"], g["Rgeo"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from fec import betap_li as BL
    from fec import current_diffusion as CD
    from fec.nrsl_copies import CD_SHA256, meta_from_maps
    from fec.nrsl_copies import FROZEN_SHA_PREFIX, fill_in_shot
    from fec.paths import WORK, shot_files
    t0 = time.time()
    fz = CD.OUT / "frozen.json"
    assert _sha(fz).startswith(FROZEN_SHA_PREFIX)
    assert _sha(CD.__file__) in CD_SHA256.values()
    fr_ = json.loads(fz.read_text())
    v = fr_["selected_variant"]
    d = Path(a.fold_dir)
    z = np.load(d / "inputs.npz")
    shot, t_ms, tf, stems = z["shot"], z["t_ms"].astype(float), np.abs(z["tf"]), [str(s) for s in z["stems"]]
    psi_path = str(d / "base.npy")
    P = np.load(psi_path, mmap_mode="r")
    n = P.shape[0]
    meta = meta_from_maps(np.asarray(P), device=a.device)
    np.savez(d / "base_meta.npz", **meta)
    gR = np.load(d / "grid.npz")["gR"]; gZ = np.load(d / "grid.npz")["gZ"]
    chunks = [slice(s, min(n, s + 512)) for s in range(0, n, 512)]
    with Pool(a.nproc) as pool:
        geo = np.concatenate(pool.map(_geo, [(np.asarray(P[c], np.float64), gR, gZ) for c in chunks]))
    print(f"meta+geo {time.time() - t0:.0f}s geo finite {np.isfinite(geo).all(1).mean():.4f}", flush=True)
    jobs, slices, parq = [], {}, {p.stem: p for p in shot_files(a.config)}
    for s in np.unique(shot):
        rows = np.flatnonzero(shot == s)
        assert np.all(np.diff(rows) == 1)
        stem = stems[int(s)]
        slices[stem] = slice(int(rows[0]), int(rows[-1]) + 1)
        F = CD.MU0 * CD.MAST_TF_TURNS * float(np.nanmedian(tf[rows])) * 1e3 / (2 * np.pi)
        jobs.append((stem, str(parq[stem]), rows.tolist(), psi_path,
                     meta["psi_ax"][rows], meta["psi_b"][rows], meta["Rax"][rows], meta["Zax"][rows],
                     geo[rows], t_ms[rows], F, {}))
    results = []
    with Pool(a.nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(CD._mast_job, jobs, chunksize=1)):
            results.append(r)
            if i % 200 == 0:
                print(f"cd {i + 1}/{len(jobs)} {time.time() - t0:.0f}s {r[2].get('status')}", flush=True)
    V = len(CD.variants(CD.ZEFFS, CD.ICS))
    arr, infos = CD._collect(n, V, results, slices)
    li_cyl = arr["li"][:, v]
    li_est = BL.predict_reg(fr_["models"]["alone"], {"li_cd": li_cyl})[0]
    default = float(json.loads((WORK / "licond" / "sigma.json").read_text())["train_est_median"])
    li_in, filled = fill_in_shot(li_est, shot, t_ms, default)
    np.savez(d / "li.npz", li=li_in, li_est_raw=li_est, filled=filled, li_cyl=li_cyl, geo=geo)
    st = [i.get("status") for i in infos.values()]
    fin = np.isfinite(li_est)
    rep = dict(config=a.config, n=int(n), coverage=float(fin.mean()), filled=float(filled.mean()),
               status={s: st.count(s) for s in set(st)},
               frac_inrange_syn=float(np.mean((li_in >= SYN_RANGE[0]) & (li_in <= SYN_RANGE[1]))),
               frac_clamp=float(np.mean(fin & ((li_est <= CLAMP[0] + 1e-9) | (li_est >= CLAMP[1] - 1e-9)))),
               li_pct_5_50_95=np.percentile(li_in, [5, 50, 95]).round(4).tolist(),
               li_cyl_pct_5_50_95=np.nanpercentile(li_cyl, [5, 50, 95]).round(4).tolist(),
               frozen_variant=int(v), seconds=time.time() - t0)
    (d / "li.json").write_text(json.dumps(rep, indent=1))
    print("MLI_DONE " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
