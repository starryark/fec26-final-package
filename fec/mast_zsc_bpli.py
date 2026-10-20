"""MAST self-consistent height (fec.mast_zsc) with the beta_p / li constraint rows of
fec.betap_li (inputs only, no labels; root 2026-10-09).

docs/research/BETAP_LI_CONSTRAINTS_20261008.md: TRAIN-fitted, machine-agnostic regressions of
EFIT beta_p (CV R2 0.87) and li (0.58) from Thomson / time features, added as weighted rows
(selected KC8: w_bp = w_li = 8) to the Thomson-pair kinetic fit, lifted every DIII-D VAL headline
term (S +0.0087, li consistency -0.73 -> +0.11, axis bias halved) but missed the +0.020 gate, so
its MAST driver (select_v3 based) was frozen and never run.  This module ports the same rows into
the zsc secant: every kfit3(cfix=True) call becomes kfit_bpli(cfix=True, cons=targets).  Frames
without a finite target get no rows (identical to plain zsc there).

    # 1) targets from an existing stage-1 map set (features use our own maps: lam, bpe, ...)
    python -B fec/mast_zsc_bpli.py targets --base <stage-1 dir> --out <dir>
    # 2) stage 1 (production format psi.npy + meta.npz); then stage 2 + betaN as usual
    python -B fec/gpu_slot.py run --owner <me> -- python -B fec/mast_zsc_bpli.py stage1 --out <dir> \
        [--passive c1_r1|none] [--sol-frac 0.5]
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec import betap_li as BL  # noqa: E402


def mast_targets(psi, recs_extra, frames, nproc=4):
    """beta_p / li targets and sigmas: copy of the frozen driver's mast_targets
    (fec_work/betap_li/frozen/mast_select_bpli_frozen.py, sha256 4355b136...)."""
    sel = json.loads((BL.OUT / "selection.json").read_text())
    jobs, order = {}, []
    for i, (path, k) in enumerate(frames):
        jobs.setdefault(path, []).append((k, i))
    payload = [("mast_arrays", (path, [k for k, _ in v], [psi[i] for _, i in v]))
               for path, v in jobs.items()]
    for v in jobs.values():
        order.extend(i for _, i in v)
    if nproc > 1 and len(payload) > 1:
        with Pool(min(nproc, 4, len(payload))) as pool:
            recs = [r for part in pool.map(BL._record_job, payload) for r in part]
    else:
        recs = [r for p in payload for r in BL._record_job(p)]
    B = len(frames)
    arr = {k: np.full(B, np.nan) for k in recs[0]}
    for j, i in enumerate(order):
        for k in arr:
            arr[k][i] = recs[j][k]
    arr.update(recs_extra)
    X = BL.feature_table(arr)
    out = {}
    for kind in ("li", "bp"):
        p, inr = BL.predict_reg(sel[kind]["model"], X)
        out[f"{kind}_t"] = p
        out[f"{kind}_sig"] = (np.full(B, sel[kind]["sigma"]) if kind == "li"
                              else p * sel[kind]["sigma"])
        out[f"{kind}_inrange"] = inr
    return out


def compute_targets(base, out_dir, config="mast_public_test", chunk=4096, nproc=4):
    from fec.features import load_ds
    from fec.machines import MAST_CH
    from fec.paths import WORK
    ds = load_ds(config)
    info = ds["info"]
    P = np.load(Path(base) / "psi.npy", mmap_mode="r")
    N = P.shape[0]
    cur_all = np.asarray(ds["cur"][:N], dtype=np.float64)
    ipc = MAST_CH.index("magnetics_plasma_current")
    shot = np.asarray(ds["shot"][:N])
    shot_max = BL.shot_ipmax(config, shot)
    keys = ("bp_t", "bp_sig", "li_t", "li_sig", "bp_inrange", "li_inrange")
    res = {k: np.full(N, np.nan) for k in keys}
    t0 = time.time()
    for a in range(0, N, chunk):
        sl = slice(a, min(N, a + chunk))
        frames = [(str(WORK / "shots" / config / f"{info['order'][ds['shot'][g]]}.npz"),
                   int(ds["fidx"][g])) for g in range(sl.start, sl.stop)]
        s0 = np.sign(cur_all[sl, ipc, 6])
        s0[s0 == 0] = 1
        extra = dict(ip_off=cur_all[sl, ipc, :] * s0[:, None],
                     t_ms=np.asarray(ds["t_ms"][sl], np.float64), ipmax=shot_max[sl])
        tg = mast_targets(np.asarray(P[sl], dtype=np.float64), extra, frames, nproc=nproc)
        for k in keys:
            res[k][sl] = tg[k]
        print(f"targets {sl.stop}/{N} {time.time() - t0:.0f}s", flush=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "bpli_targets.npz", base=str(base), **res)
    print("TARGETS_DONE li_t p5/50/95", np.nanpercentile(res["li_t"], [5, 50, 95]).round(3),
          "bp_t", np.nanpercentile(res["bp_t"], [5, 50, 95]).round(3),
          "finite", float(np.isfinite(res["li_t"]).mean()), flush=True)


def stage1(out_dir, config="mast_public_test", passive="c1_r1", sol_frac=None, weights=None,
           max_frames=1024, limit=0, targets=None, heights_from=None):
    """fec.mast_zsc.stage1_zsc (z0 fallback) with kfit_bpli(cfix=True, cons) in the secant."""
    from fec.features import load_ds
    from fec.machines import MAST_CH, MAST_COILS, mast_setup
    from fec.mast_select_v2 import V2_KFIT, shot_chunks
    from fec.mast_select_v3 import V3Solver
    from fec.mast_v1 import thomson_frame
    from fec.mast_zsc import META_KEYS, ZSC_DEFAULT, select_zsc
    from fec.paths import WORK
    out_dir = Path(out_dir)
    tg = dict(np.load(Path(targets) if targets else out_dir / "bpli_targets.npz"))
    heights = None
    if heights_from:          # final heights of a previous stage-1 run (meta Zt)
        heights = np.asarray(np.load(Path(heights_from) / "meta.npz")["Zt"], float)
        heights = np.where(np.isfinite(heights), heights, 0.0)
    wts = dict(BL.VARIANTS[BL.SELECTED]) if weights is None else weights
    ds = load_ds(config)
    info = ds["info"]
    N = len(ds["t_ms"]) if not limit else limit
    cur = np.asarray(ds["cur"][:N], dtype=np.float64)[:, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    spec, M = mast_setup() if sol_frac is None else mast_setup(sol_fraction=sol_frac)
    Ipas = None
    if passive != "none":
        from fec.passive import PassiveModel
        Ipas = np.load(WORK / "passive" / "frames" / config / passive / "I.npy", mmap_mode="r")
        M = PassiveModel().augment(M)
    S = V3Solver(spec, M)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(config=config, passive=passive, zmode="root_t+zsc+bpli", sol_frac=sol_frac,
               zsc=dict(ZSC_DEFAULT), weights=wts, variant=BL.SELECTED,
               targets_base=str(tg.get("base")), heights_from=str(heights_from), N=N,
               code="fec/mast_zsc_bpli.py")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    psi_out = np.lib.format.open_memmap(out_dir / "psi.npy", "w+", np.float32, (N, 65, 65))
    meta = {k: np.full(N, np.nan) for k in META_KEYS}
    shot = np.asarray(ds["shot"][:N])
    kw = {k: v for k, v in V2_KFIT.items() if k != "levels"}
    t0 = time.time()
    cache = {}
    for sl in shot_chunks(shot, max_frames):
        I = cur[sl][:, ci] * 1e3
        if Ipas is not None:
            I = np.concatenate([I, np.asarray(Ipas[sl], dtype=np.float64)], 1)
        Ip = np.maximum(cur[sl, MAST_CH.index("magnetics_plasma_current")] * 1e3, 5e4)
        Rl, tel, nel = [], [], []
        for g in range(sl.start, sl.stop):
            stem = info["order"][ds["shot"][g]]
            if stem not in cache:
                cache.clear()
                cache[stem] = dict(np.load(WORK / "shots" / config / f"{stem}.npz"))
            R, te, ne = thomson_frame(cache[stem], ds["fidx"][g])
            Rl.append(R)
            tel.append(te)
            nel.append(ne)
        nch = max(len(r) for r in Rl)

        def pad(L):
            return np.stack([np.pad(np.asarray(x, float), (0, nch - len(x)),
                                    constant_values=np.nan) for x in L])
        cons = {k: np.asarray(tg[k][sl], float) for k in ("bp_t", "bp_sig", "li_t", "li_sig")}
        cons.update(wts)

        def kf(S_, Ix_, Ip_, Z_, pairs_, I0=None, c0=None, n_iter=None, **_):
            k2 = dict(kw)
            if n_iter:
                k2["n_iter"] = n_iter
            return BL.kfit_bpli(S_, Ix_, Ip_, Z_, pairs_, I0=I0, c0=c0, cfix=True, rinit=0.9,
                                cons=cons, **k2)
        if heights is None:      # full zsc secant with the constrained fit (slow: ~7 solves)
            o = select_zsc(S, I, Ip, pad(Rl), pad(tel), pad(nel), shot[sl],
                           zsc=dict(fallback="z0"), kfit_fn=kf)
        else:                    # fast: one constrained solve at the given zsc heights
            o = select_zsc(S, I, Ip, pad(Rl), pad(tel), pad(nel), shot[sl],
                           zsc=dict(zfix="given"), kfit_fn=kf, Zgiven=heights[sl],
                           zflag_given=np.full(sl.stop - sl.start, -1))
        psi_out[sl] = o["psi"]
        for k in META_KEYS:
            if k in o:
                meta[k][sl] = o[k]
        np.savez(out_dir / "meta_partial.npz", done=np.array(sl.stop), **meta)
        print(f"{sl.stop}/{N} {time.time() - t0:.0f}s kfit {np.mean(o['method']):.2f} "
              f"zsc_conv {np.mean(o['zsc_conv']):.3f} |Z| med {np.median(np.abs(o['Zax'])):.3f}",
              flush=True)
        del o
        torch.cuda.empty_cache()
    psi_out.flush()
    np.savez(out_dir / "meta.npz", **meta)
    (out_dir / "meta_partial.npz").unlink(missing_ok=True)
    print("DONE", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["targets", "stage1"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--passive", default="c1_r1")
    ap.add_argument("--sol-frac", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--heights-from", default=None, help="fast mode: one constrained solve at the "
                    "meta Zt heights of this stage-1 dir (e.g. the plain zsc run)")
    a = ap.parse_args()
    if a.cmd == "targets":
        compute_targets(a.base, a.out, nproc=a.nproc)
    else:
        stage1(a.out, passive=a.passive, sol_frac=a.sol_frac, limit=a.limit, targets=a.targets,
               heights_from=a.heights_from)
