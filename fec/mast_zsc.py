"""MAST vertical position, self-consistent version (inputs only, no labels).

Forensic finding (docs/research/MAST_VERTICAL_FORENSIC_20261008.md): the production selector
(fec.mast_select_v3.select_v3, zmode root_t, kfit_warm zs) fixes the plasma height Zt from the
force balance of a FIXED default profile (FFZ_PROF = (0.05, 2, 2), no dv field), then refits the
profile (kfit3) at that frozen height with the radial-field feedback c frozen (cfix=False), so
the final equilibrium is not force-free with its own profile.  On MAST the external field has a
near-zero decay index, so the FFZ_PROF force law is flat/non-monotone on ~15-20% of frames and
robust_zt then takes a far root (|Z| ~ 0.1-0.3 m).  The synthetic twin shows the consequence:
production Z_axis R2 = -2.9 (BP world) with truth Z std 2.4 cm.

select_zsc = select_v3 (root_t, warm start zs, same grid fallback) plus a self-consistent height:
the force balance is re-solved with the Thomson-fitted profile itself.  For each frame,
kfit3(..., cfix=True) at height Z returns the radial-field feedback c(Z) that the FITTED
profile needs to stay at Z; a damped secant on Z drives c -> 0 (n_sec steps, step <= step_max,
|Z| <= zmax), keeping the iterate with the smallest |c|.  Policy `fallback`:
  "best"  always the best iterate;
  "z0"    frames whose secant did not converge (|c_best| > ctol) take the kfit3 solution at Z = 0
          -- a vertically centred plasma is the MAST default (DND, up/down series-wound PF
          pairs);
  "prod"  frames that did not converge keep the production (robust_zt) height.

  "z0" note: the final kfit3 solve is run at Z = 0 for those frames (same call as the rest).

Drivers (production format psi.npy + meta.npz; then stage 2 + betaN as fec/run_final.py, e.g.
fec_work/mastz/mz_finish_cand.py <dir>):
    # full: every frame (robust_zt start, zsc on all kfit frames), ~4 h on the shared GPU
    python -B fec/gpu_slot.py run --owner <me> -- python -B fec/mast_zsc.py --out <dir> --fallback z0
    # hybrid: production maps kept on zflag-0 frames, zsc on zflag 1/2/3 frames only (~40 min)
    python -B fec/gpu_slot.py run --owner <me> -- python -B fec/mast_zsc.py --out <dir> \
        --hybrid-prod fec_work/final_public_test/mast/loff --fallback z0
Twin / DIII-D validation: docs/research/MAST_VERTICAL_FORENSIC_20261008.md; CPU tests:
tests/test_fec_mast_zsc.py (synthetic force law).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.mast_select_v2 import (FFZ_PROF, V2_FALLBACK, V2_GRID, V2_KFIT, nearest_in_shot,  # noqa: E402
                                prep_profiles, select_stacked, symmetry_mm, to_dev, LEVELS_V1)
from fec.mast_select_v3 import kfit3, robust_zt  # noqa: E402

ZSC_DEFAULT = dict(n_sec=4, dz=0.03, step_max=0.08, zmax=0.3, ctol_rel=0.1, fallback="z0",
                   it_sec=None, start="prod", smooth_h=0, zfix=None)


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _shot_running_median(z, shot, h):
    out = z.copy()
    for s_ in np.unique(shot):
        ix = np.flatnonzero(shot == s_)
        for k in range(len(ix)):
            out[ix[k]] = np.median(z[ix[max(0, k - h):k + h + 1]])
    return out


def zsc_refine(S, Ix, Ip, Z0, pairs, I0, c0, kw, n_sec=4, dz=0.03, step_max=0.08, zmax=0.3,
               ctol_rel=0.1, fallback="z0", it_sec=None, shot=None, smooth_h=0, kfit_fn=None):
    """Self-consistent force-free height for the Thomson-fitted profile (module doc).

    Returns (o, Zf, info): o = kfit3 output at the final heights (cfix=True, so o['c'] is the
    remaining radial-field feedback), Zf (B,), info dict of per-frame diagnostics."""
    B = len(Ip)
    tt = lambda x: torch.as_tensor(np.asarray(x), device=S.dev)  # noqa: E731
    if kfit_fn is None:      # MAST: kfit3 with the feedback driven by the force imbalance
        def kf(*a, **k):
            return kfit3(*a, cfix=True, **k)
    else:                    # e.g. a cfix variant of fec.d3d_physics.kfit_d3d (DIII-D proxy)
        kf = kfit_fn
    za = np.asarray(Z0, float).copy()
    oa = kf(S, Ix, Ip, za, pairs, I0=I0, c0=c0, **kw)
    ca = _np(oa["c"]).astype(float)
    # scale for the convergence test: |c| change over dz of the first secant pair
    zb = za + np.where(za > 0, -dz, dz)
    kws = dict(kw, n_iter=it_sec) if it_sec else kw      # warm-started secant solves
    ob = kf(S, Ix, Ip, zb, pairs, I0=oa["I"], c0=oa["c"], **kws)
    cb = _np(ob["c"]).astype(float)
    slope0 = (cb - ca) / (zb - za)
    best_z = np.where(np.abs(cb) < np.abs(ca), zb, za)
    best_c = np.minimum(np.abs(ca), np.abs(cb))
    state = {"I": torch.where(tt(np.abs(cb) < np.abs(ca))[:, None], ob["I"], oa["I"]),
             "c": torch.where(tt(np.abs(cb) < np.abs(ca)), ob["c"], oa["c"])}
    zs_hist = [za.copy(), zb.copy()]
    for k in range(n_sec):
        den = cb - ca
        with np.errstate(all="ignore"):
            zn = zb - cb * (zb - za) / den
        bad = ~np.isfinite(zn) | (np.abs(den) < 1e-14)
        zn = np.where(bad, zb, zn)
        zn = zb + np.clip(zn - zb, -step_max, step_max)
        zn = np.clip(zn, -zmax, zmax)
        on = kf(S, Ix, Ip, zn, pairs, I0=state["I"], c0=state["c"], **kws)
        cn = _np(on["c"]).astype(float)
        better = np.abs(cn) < best_c
        best_z = np.where(better, zn, best_z)
        best_c = np.where(better, np.abs(cn), best_c)
        bt = tt(better)
        state = {"I": torch.where(bt[:, None], on["I"], state["I"]),
                 "c": torch.where(bt, on["c"], state["c"])}
        za, ca, zb, cb = zb, cb, zn, cn
        zs_hist.append(zn.copy())
        del on
    # converged: |c| reduced to ctol_rel of the first-pair scale, or below an absolute
    # floor (2e-5 ~ 2 mm of displacement at the typical slope 1e-2 per m)
    scale = np.maximum(np.abs(slope0) * dz, np.abs(_np(oa["c"])))
    conv = best_c <= np.maximum(ctol_rel * scale, 2e-5)
    Zf = best_z.copy()
    if fallback == "z0":
        Zf = np.where(conv, best_z, 0.0)
    elif fallback == "prod":
        Zf = np.where(conv, best_z, np.asarray(Z0, float))
    Zraw = Zf.copy()
    if smooth_h and shot is not None:
        # feedback-held position evolves slowly: per-shot running median over +-smooth_h
        # frames (5 ms EFIT spacing) of the chosen heights
        Zf = _shot_running_median(Zf, np.asarray(shot), int(smooth_h))
    # final solve at the chosen heights, warm-started from the best state
    o = kf(S, Ix, Ip, Zf, pairs, I0=state["I"], c0=state["c"], **kw)
    info = dict(zsc_conv=conv, zsc_best=best_z, zsc_cbest=best_c, zsc_slope0=slope0, zsc_raw=Zraw,
                zsc_c_final=_np(o["c"]), zsc_hist=np.stack(zs_hist, 1))
    return o, Zf, info


def select_zsc(S, I, Ip, R, te, ne, shot, grid=None, kfit_kw=None, w_ne=1.0, n_iter=100,
               min_pairs=3, dv_col=False, zsc=None, Zgiven=None, zflag_given=None, kfit_fn=None):
    """fec.mast_select_v3.select_v3(zmode='root_t', kfit_warm='zs', no limiter) with the
    kfit3 equilibrium replaced by the self-consistent height (zsc_refine).  Grid-fallback
    frames (too few Thomson pairs or kfit not converged) are unchanged from select_v3."""
    from fec.kefit import LEVELS, te_pairs
    zsc = dict(ZSC_DEFAULT, **(zsc or {}))
    dev = S.dev
    B = len(Ip)
    shot = np.asarray(shot)
    grid = V2_GRID if grid is None else grid
    kw = dict(V2_KFIT, **(kfit_kw or {}))
    levels_k = kw.pop("levels", LEVELS)
    Ix = np.concatenate([I, np.zeros((B, 1))], 1) if dv_col else np.asarray(I, float)
    prof = np.tile(FFZ_PROF, (B, 1))
    if Zgiven is None:
        zst = robust_zt(S, Ix, Ip, prof, shot, temporal=True)
    else:   # hybrid: start from given (production) heights; FFZ warm state solved there
        Zg = np.asarray(Zgiven, float)
        og = S.solve(Ix, Ip, prof, Zg, n_iter=150, Rinit=np.full(B, 0.9))
        zst = dict(Zs=Zg.copy(), I=og["I"], c=og["c"], o0=og, Zv=np.full(B, np.nan),
                   zflag=np.asarray(zflag_given if zflag_given is not None else
                                    np.full(B, -1), int))
    Zs = zst["Zs"]
    wI, wc = zst["I"], zst["c"]
    keys = ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI")
    gR = torch.as_tensor(S.spec.gR, device=dev, dtype=torch.float32)
    iz0 = int(np.argmin(np.abs(S.spec.gZ)))
    lev = torch.as_tensor(LEVELS_V1, device=dev, dtype=torch.float32)
    out = dict(Zt=Zs.copy(), method=np.zeros(B, int), score=np.full(B, np.nan),
               grid_choice=np.full(B, -1), grid_score=np.full(B, np.nan),
               inherited=np.zeros(B, bool), kfit_dv=np.full(B, np.nan),
               kfit_score=np.full(B, np.nan), kfit_npairs=np.zeros(B, int),
               zflag=zst["zflag"], Zv=zst["Zv"], limited=np.zeros(B, bool), Zprod=Zs.copy())
    pairs = np.stack([te_pairs(R[i], te[i], levels=levels_k) for i in range(B)])
    o_prod = kfit3(S, Ix, Ip, Zs, pairs, I0=wI, c0=wc, **kw)          # == select_v3 kfit
    start = zsc.pop("start")
    zfix = zsc.pop("zfix", None)
    if zfix is not None:     # fixed-height variant: kfit3 with the feedback holding Z = zfix
        # zfix = "given": hold each frame at its Zgiven height (e.g. a previous zsc run's heights)
        Zf = np.asarray(Zs, float).copy() if isinstance(zfix, str) else np.full(B, float(zfix))
        if kfit_fn is not None:
            o = kfit_fn(S, Ix, Ip, Zf, pairs, I0=o_prod["I"], c0=o_prod["c"])
        else:
            o = kfit3(S, Ix, Ip, Zf, pairs, I0=o_prod["I"], c0=o_prod["c"], cfix=True, **kw)
        cf = _np(o["c"])
        info = dict(zsc_conv=np.ones(B, bool), zsc_best=Zf.copy(), zsc_cbest=np.abs(cf),
                    zsc_slope0=np.full(B, np.nan), zsc_raw=Zf.copy(), zsc_c_final=cf)
    elif start == "zero" and Zgiven is not None:
        raise ValueError("start='zero' needs the robust_zt Zt=0 state (Zgiven=None)")
    elif start == "zero":    # secant from the mid-plane, warm start = robust_zt's Zt=0 solve
        o, Zf, info = zsc_refine(S, Ix, Ip, np.zeros(B), pairs, zst["o0"]["I"], zst["o0"]["c"],
                                 kw, shot=shot, **zsc)
    else:                    # secant from the production (robust_zt) height
        # kfit_fn (e.g. fec.mast_zsc_bpli: kfit_bpli with beta_p / li rows, cfix=True) replaces
        # kfit3(cfix=True) inside the secant; None keeps kfit3
        o, Zf, info = zsc_refine(S, Ix, Ip, Zs, pairs, o_prod["I"], o_prod["c"], kw, shot=shot,
                                 kfit_fn=kfit_fn, **zsc)
    pte = to_dev(prep_profiles(R, te), dev)
    pne = to_dev(prep_profiles(R, ne), dev)
    mk = symmetry_mm(o["psi"], o["psi_ax"], o["psi_b"], gR, iz0, pte, lev)[0] + \
        w_ne * symmetry_mm(o["psi"], o["psi_ax"], o["psi_b"], gR, iz0, pne, lev)[0]
    conv = (o["dI"] < 1e-2).cpu().numpy()
    mk = np.where(conv, mk.cpu().numpy(), np.nan)
    npairs = np.isfinite(pairs).all(-1).sum(1)
    for k in keys:
        out[k] = o[k].cpu().numpy()
    if "limited" in o:       # limiter runs (--limiter-cc): keep the solver's limited flag (review 2026-10-09)
        out["limited"] = o["limited"].cpu().numpy().astype(bool)
    out.update(method=np.ones(B, int), score=mk, kfit_dv=o["dv"].cpu().numpy(), kfit_score=mk,
               kfit_npairs=npairs, Zt=Zf, Zax_prod=o_prod["Zax"].cpu().numpy(),
               psi_prod=o_prod["psi"].cpu().numpy(), **info)
    need_grid = ~((npairs >= min_pairs) & conv)
    del o, o_prod
    if need_grid.any():
        sub = np.flatnonzero(need_grid)
        st = torch.as_tensor(sub, device=dev)
        g = select_stacked(S, Ix[sub], Ip[sub], R[sub], te[sub], ne[sub], grid, Zs[sub],
                           wI[st], wc[st], w_ne=w_ne, n_iter=n_iter)
        okg = np.isfinite(g["score"]).any(0)
        ch = np.where(okg, g["choice"], V2_FALLBACK)
        ch, inh = nearest_in_shot(ch, okg, shot[sub])
        fix = np.flatnonzero(~okg)
        if len(fix):
            ft = torch.as_tensor(sub[fix], device=dev)
            prof_f = np.array([grid[int(c)] for c in ch[fix]], float)
            os_ = S.solve(Ix[sub[fix]], Ip[sub[fix]], prof_f, Zs[sub[fix]], n_iter=n_iter,
                          I0=wI[ft], c0=wc[ft])
            for k in keys:
                g[k][fix] = os_[k].cpu().numpy()
        J = g["score"][ch, np.arange(len(sub))]
        out["grid_choice"][sub] = ch
        out["grid_score"][sub] = J
        out["inherited"][sub] = inh
        for k in keys:
            out[k][sub] = g[k]
        out["method"][sub] = 0
        out["score"][sub] = J
        out["Zt"][sub] = Zs[sub]
    return out


META_KEYS = ("psi_ax", "psi_b", "Rax", "Zax", "Zt", "c", "dI", "method", "score", "grid_choice",
             "grid_score", "inherited", "kfit_dv", "kfit_score", "kfit_npairs", "zflag", "Zv",
             "limited", "Zprod", "Zax_prod", "zsc_conv", "zsc_best", "zsc_cbest", "zsc_slope0",
             "zsc_c_final", "zsc_raw")


# Official MAST wall (FAIR-MAST level-2 `wall` limiter_r/z, fec_work/geometry/mast_limiter_fairmast.npz):
# centre-column armour at R = 0.195 m for |Z| <= 1.084 m (root 2026-10-09)
LIMITER_WALL_CC = dict(R=0.195, zmax=1.084)


def given_heights(zsc, heights, sl, n):
    """(Zgiven, zsc dict) for one chunk: a per-frame heights array (e.g. denoised zsc heights,
    fec_work/root_var/zden_heights.py) holds each frame at heights[sl] (zfix='given'); otherwise
    a constant zfix or None (unchanged default behaviour)."""
    z = dict(zsc or {})
    if heights is not None:
        z["zfix"] = "given"
        return np.asarray(heights[sl], float).copy(), z
    return (None if z.get("zfix") is None else np.full(n, float(z["zfix"]))), z


def stage1_zsc(out_dir, limit=0, max_frames=1024, config="mast_public_test", passive="c1_r1",
               zsc=None, p6=1.0, sol_frac=None, limiter=False, nbasis=4, heights=None, heights_from=None):
    """Production-format driver (psi.npy + meta.npz, as fec.mast_select_v3.stage1_v3 with
    zmode root_t, kfit_warm zs, limiter off) using select_zsc.  p6 scales the P6 columns
    (sensitivity only; the C1 passive currents are then NOT recomputed).  nbasis=6: kfit3 with the
    two extra broad/hollow bases (docs/research/KFIT_BROAD_BASIS_20261009.md); 4 = unchanged.
    heights: optional (N,) per-frame plasma heights to hold (zfix='given'); None = unchanged."""
    import json
    import time
    from fec.features import load_ds
    from fec.machines import MAST_CH, MAST_COILS, mast_setup
    from fec.mast_select_v2 import shot_chunks
    from fec.mast_select_v3 import V3Solver
    from fec.mast_v1 import thomson_frame
    from fec.paths import WORK
    ds = load_ds(config)
    info = ds["info"]
    N = len(ds["t_ms"]) if not limit else limit
    cur = np.asarray(ds["cur"][:N], dtype=np.float64)[:, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    # sol_frac: solenoid current per element as a fraction of the feed current (default
    # fec.machines.MAST_SOL_TURN_FRACTION = 0.5); 0.65 = Thomson-calibrated (root 2026-10-09)
    spec, M = mast_setup() if sol_frac is None else mast_setup(sol_fraction=sol_frac)
    if p6 != 1.0:
        for c in ("magnetics_p6l_current", "magnetics_p6u_current"):
            M[:, MAST_COILS.index(c)] *= p6
    Ipas = None
    if passive != "none":
        from fec.passive import PassiveModel
        Ipas = np.load(WORK / "passive" / "frames" / config / passive / "I.npy", mmap_mode="r")
        M = PassiveModel().augment(M)
    S = V3Solver(spec, M, limiter=LIMITER_WALL_CC if limiter else None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(config=config, passive=passive, zmode="root_t+zsc", limiter="on" if limiter else "off",
               kfit_warm="zs", zsc=dict(ZSC_DEFAULT, **(zsc or {})), p6=p6, sol_frac=sol_frac, N=N,
               code="fec/mast_zsc.py", **({"nbasis": int(nbasis)} if nbasis != 4 else {}),
               **({"heights_from": str(heights_from)} if heights is not None else {}))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    psi_out = np.lib.format.open_memmap(out_dir / "psi.npy", "w+", np.float32, (N, 65, 65))
    prod_out = np.lib.format.open_memmap(out_dir / "psi_prodz.npy", "w+", np.float16,
                                         (N, 65, 65))
    meta = {k: np.full(N, np.nan) for k in META_KEYS}
    shot = np.asarray(ds["shot"][:N])
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
            Rl.append(R); tel.append(te); nel.append(ne)
        nch = max(len(r) for r in Rl)

        def pad(L):
            return np.stack([np.pad(np.asarray(x, float), (0, nch - len(x)),
                                    constant_values=np.nan) for x in L])
        zg, zsc_l = given_heights(zsc, heights, sl, len(Ip))
        o = select_zsc(S, I, Ip, pad(Rl), pad(tel), pad(nel), shot[sl], zsc=zsc_l,
                       Zgiven=zg, zflag_given=None if zg is None else np.full(len(Ip), -1),
                       kfit_kw=None if nbasis == 4 else dict(nbasis=int(nbasis)))
        psi_out[sl] = o["psi"]
        prod_out[sl] = o["psi_prod"].astype(np.float16)
        for k in META_KEYS:
            if k in o:
                meta[k][sl] = o[k]
        np.savez(out_dir / "meta_partial.npz", done=np.array(sl.stop), **meta)
        print(f"{sl.stop}/{N} {time.time() - t0:.0f}s kfit {np.mean(o['method']):.2f} "
              f"zsc_conv {np.mean(o['zsc_conv']):.3f} |Z| med {np.median(np.abs(o['Zax'])):.3f} "
              f"(prod {np.median(np.abs(o['Zax_prod'])):.3f})", flush=True)
        del o
        torch.cuda.empty_cache()
    psi_out.flush(); prod_out.flush()
    np.savez(out_dir / "meta.npz", **meta)
    (out_dir / "meta_partial.npz").unlink(missing_ok=True)
    print("DONE", f"{time.time() - t0:.0f}s", flush=True)


def stage1_hybrid(out_dir, prod_dir, config="mast_public_test", passive="c1_r1", zsc=None,
                  flags=(1, 2, 3), batch=1024):
    """Hybrid candidate: production maps (prod_dir psi.npy/meta.npz) for frames whose
    production height was a verified secant root (zflag 0); frames with zflag in `flags`
    are re-solved with select_zsc started from the production height (no robust_zt).
    Writes a complete psi.npy + meta.npz in production format plus zsc diagnostics."""
    import json
    import time
    from fec.features import load_ds
    from fec.machines import MAST_CH, MAST_COILS, mast_setup
    from fec.mast_select_v3 import V3Solver
    from fec.mast_v1 import thomson_frame
    from fec.paths import WORK
    prod_dir, out_dir = Path(prod_dir), Path(out_dir)
    ds = load_ds(config)
    info = ds["info"]
    pm = dict(np.load(prod_dir / "meta.npz"))
    P = np.load(prod_dir / "psi.npy", mmap_mode="r")
    N = P.shape[0]
    zf = pm["zflag"].astype(int)
    sel = np.flatnonzero(np.isin(zf, list(flags)))
    cur = np.asarray(ds["cur"][:N], dtype=np.float64)[:, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    spec, M = mast_setup()
    Ipas = None
    if passive != "none":
        from fec.passive import PassiveModel
        Ipas = np.load(WORK / "passive" / "frames" / config / passive / "I.npy", mmap_mode="r")
        M = PassiveModel().augment(M)
    S = V3Solver(spec, M)
    out_dir.mkdir(parents=True, exist_ok=True)
    zcfg = dict(ZSC_DEFAULT, **(zsc or {}))
    cfg = dict(config=config, passive=passive, zmode="hybrid: production root_t heights for "
               f"zflag 0, zsc for zflags {list(flags)}", limiter="off", kfit_warm="zs",
               zsc=zcfg, prod_dir=str(prod_dir), n_resolved=int(len(sel)), N=int(N),
               code="fec/mast_zsc.py stage1_hybrid")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    psi_out = np.lib.format.open_memmap(out_dir / "psi.npy", "w+", np.float32, (N, 65, 65))
    for a in range(0, N, 8192):
        psi_out[a:a + 8192] = P[a:a + 8192]
    meta = {k: np.asarray(v, float).copy() for k, v in pm.items()}
    for k in META_KEYS:
        if k not in meta:
            meta[k] = np.full(N, np.nan)
    meta["resolved"] = np.zeros(N)
    shot = np.asarray(ds["shot"][:N])
    t0 = time.time()
    cache = {}
    for b0 in range(0, len(sel), batch):
        g = sel[b0:b0 + batch]
        I = cur[g][:, ci] * 1e3
        if Ipas is not None:
            I = np.concatenate([I, np.asarray(Ipas[g], dtype=np.float64)], 1)
        Ip = np.maximum(cur[g, MAST_CH.index("magnetics_plasma_current")] * 1e3, 5e4)
        Rl, tel, nel = [], [], []
        for k in g:
            stem = info["order"][ds["shot"][k]]
            if stem not in cache:
                cache.clear()
                cache[stem] = dict(np.load(WORK / "shots" / config / f"{stem}.npz"))
            R, te, ne = thomson_frame(cache[stem], ds["fidx"][k])
            Rl.append(R); tel.append(te); nel.append(ne)
        nch = max(len(r) for r in Rl)

        def pad(L):
            return np.stack([np.pad(np.asarray(x, float), (0, nch - len(x)),
                                    constant_values=np.nan) for x in L])
        o = select_zsc(S, I, Ip, pad(Rl), pad(tel), pad(nel), shot[g], zsc=dict(zcfg),
                       Zgiven=pm["Zt"][g], zflag_given=zf[g])
        psi_out[g] = o["psi"]
        for k in META_KEYS:
            if k in o and np.ndim(o[k]) == 1:
                meta[k][g] = o[k]
        meta["zflag"][g] = zf[g]
        meta["Zv"][g] = pm["Zv"][g]
        meta["resolved"][g] = 1
        print(f"{b0 + len(g)}/{len(sel)} {time.time() - t0:.0f}s zsc_conv "
              f"{np.mean(o['zsc_conv']):.3f} |Z| med {np.median(np.abs(o['Zax'])):.3f} "
              f"(prod {np.median(np.abs(pm['Zax'][g])):.3f})", flush=True)
        del o
        torch.cuda.empty_cache()
    psi_out.flush()
    np.savez(out_dir / "meta.npz", **meta)
    print("DONE", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hybrid-prod", default=None, help="production dir: run stage1_hybrid "
                    "(re-solve only frames with zflag != 0 there)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default="mast_public_test")
    ap.add_argument("--passive", default="c1_r1")
    ap.add_argument("--fallback", default=ZSC_DEFAULT["fallback"], choices=["best", "z0", "prod"])
    ap.add_argument("--p6", type=float, default=1.0)
    ap.add_argument("--start", default=ZSC_DEFAULT["start"], choices=["prod", "zero"])
    ap.add_argument("--smooth-h", type=int, default=0, help="per-shot running median of the "
                    "heights over +-h frames before the final solve (0 = off)")
    ap.add_argument("--limiter-cc", action="store_true", help="EFIT-style centre-column limiter "
                    "(official wall R = 0.195 m, |Z| <= 1.084 m) in every Picard iteration")
    ap.add_argument("--sol-frac", type=float, default=None, help="solenoid current fraction per "
                    "element (default fec.machines 0.5)")
    ap.add_argument("--zfix", type=float, default=None, help="fixed plasma height for every frame "
                    "(feedback holds Z = zfix; no force-free search), e.g. 0 = DND mid-plane")
    ap.add_argument("--nbasis", type=int, default=4, choices=[4, 6], help="kfit3 current bases: "
                    "4 (default) or 6 (+ R/R0 x(1-x), R0/R x(1-x); KFIT_BROAD_BASIS_20261009)")
    ap.add_argument("--heights-from", default=None, help=".npy of per-frame plasma heights to hold "
                    "(zfix='given'), e.g. denoised zsc heights; default off")
    ap.add_argument("--it-sec", type=int, default=0, help="kfit3 iterations of the warm-started "
                    "secant solves (0 = same as the first solve, 120)")
    a = ap.parse_args()
    if a.hybrid_prod:
        stage1_hybrid(a.out, a.hybrid_prod, config=a.config, passive=a.passive,
                      zsc=dict(fallback=a.fallback, it_sec=a.it_sec or None, start="prod"))
        raise SystemExit(0)
    stage1_zsc(a.out, limit=a.limit, config=a.config, passive=a.passive,
               zsc=dict(fallback=a.fallback, it_sec=a.it_sec or None, start=a.start,
                        smooth_h=a.smooth_h, zfix=a.zfix),
               p6=a.p6, sol_frac=a.sol_frac, limiter=a.limiter_cc, nbasis=a.nbasis,
               heights=None if a.heights_from is None else np.load(a.heights_from),
               heights_from=a.heights_from)
