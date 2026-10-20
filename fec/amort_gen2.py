"""Generator v2 (post-G0) for the amortized MAST psi network (AMORTIZED_MAST_20261010).

Root's post-G0 brief (deep-research round 3 triage rows 1, 5, 6, 8, 9, 11). Every "truth" is
still our own free-boundary GS solution driven by REAL mast_public_test INPUTS only (coil
currents, Ip, their recorded history, Thomson channel radii and missing pattern); no MAST label,
demo parquet or envelope mask is opened, and no prior width below is fitted to MAST scores.

Differences from fec/amort_gen.py (v1, G0):
  1. Posterior-mean training: n_lat independent HIDDEN latent draws per real frame (J shape,
     dv, radial-field height perturbation b_r, passive resistance scale), each with its own
     Thomson realisations; MSE training then targets E[psi | inputs].
  2. Thomson realism (synth_thomson2): shot-level radial registration offset dR_shot
     (sd 5 mm, Student-t3 tails, clipped at 10 mm) + 1 mm channel jitter; laser chord height
     Z_chord ~ U[0, 15] mm (engineering: ~14.5 mm above the midplane); per-shot correlated
     calibration scales c_T, c_n (lognormal sd 7.5%) + 3% core/edge-system factor;
     heteroscedastic channel noise (relative noise grows as sqrt(T_ref/T)); edge/SOL-only
     model discrepancy (smooth multiplicative ripple for psiN > 0.85, random SOL decay
     lengths); contiguous block dropout on top of the real missing pattern.
  3. Short causal state: passive currents of the C1 circuit model (fec/passive.py) driven by
     the REAL coil/Ip history of the frame, with the resistance scale r ~ logU[0.5, 2] per
     segment (quadratic interpolation in log2 r of the cached c1_r0.5 / c1_r1 / c1_r2
     per-frame currents); the network sees coil currents, Ip, their 5/20 ms backward
     derivatives and P6U-P6L with its derivatives (fec/amort_net2.py).
  4. Radial-field height perturbation: psi_br = -b_r R^2 Z / R0 (vacuum field, Br = b_r at
     R0), b_r per segment (+-0.6 mT; a CPU smoke test gave a median 3.5 cm height shift for
     2 mT with 30 cm tails, so the range was cut to ~1 cm median before any training);
     the force-free secant then moves the plasma height.

Prior sets: TRAIN (= B family J shape: a2 = a1, n2 = n1) and the shifted twin test C2
(independent FF' shape, wider dv/b_r/registration spreads).  Twin-v2 test truths are written
in fec.twin_mast's file format under FEC_WORK/amort/twin2 (never into FEC_WORK/twin).

CLI:
  python -B fec/gpu_slot.py run --owner amort -- python -B fec/amort_gen2.py gen --tag v2main --every 2 --n-lat 3
  python -B fec/gpu_slot.py run --owner amort -- python -B fec/amort_gen2.py twin --scen B2,C2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

OUT = WORK / "amort"
TWIN2 = OUT / "twin2"
PASS_DIR = WORK / "passive" / "frames" / "mast_public_test"
N_CORE = 130          # thomson_frame concatenates 130 core then 16 edge channels
R0 = 0.85

PRIOR_TRAIN = dict(gen_ff=False, dv_max=0.010, br_max=0.0006, r_lo=0.5, r_hi=2.0,
                   dR_sd=0.005, dR_max=0.010, jit_sd=0.001, z_lo=0.0, z_hi=0.015,
                   cal_sd=0.075, sys_sd=0.03, edge_amp=0.25, p_block=0.15)
PRIORS = {
    "TRAIN": PRIOR_TRAIN,
    "B2": PRIOR_TRAIN,
    # shifted test prior: out-of-family FF' + wider unmodelled-field and registration spreads
    "C2": dict(PRIOR_TRAIN, gen_ff=True, dv_max=0.015, br_max=0.0009, dR_sd=0.0075,
               dR_max=0.015),
}


# ============================================================================ machine
def make_solver2(device="cuda"):
    """TwinSolver with external columns [coils (12), passive pieces (386), dv, b_r]."""
    from fec.machines import mast_setup
    from fec.passive import PassiveModel
    from fec.twin_mast import TwinSolver
    spec, M = mast_setup()
    pm = PassiveModel()
    RR, ZZ = np.meshgrid(spec.gR, spec.gZ)
    Mx = np.concatenate([M, pm.Gg, (RR ** 2 / 2).reshape(-1, 1),
                         (-RR ** 2 * ZZ / R0).reshape(-1, 1)], 1)
    return TwinSolver(spec, Mx, device=device), spec, M, pm


def passive_currents(g, r):
    """C1 piece currents (B, Np) for global frames g at resistance scale r (B,), quadratic
    interpolation in x = log2 r through the cached r = 0.5, 1, 2 integrations."""
    Im, I0, Ip_ = [np.asarray(np.load(PASS_DIR / f"c1_r{s}" / "I.npy", mmap_mode="r")[g],
                              np.float64) for s in ("0.5", "1", "2")]
    x = np.log2(np.asarray(r, float))[:, None]
    return I0 + x * (Ip_ - Im) / 2 + x ** 2 * (Ip_ + Im - 2 * I0) / 2


def causal_state(g):
    """Backward 5/20 ms derivatives [kA/ms] of the 12 MAST_COILS columns (incl. Ip) and
    P6U-P6L [kA] with its derivatives, from fec_work/ds cur offsets (-20, -5, 0 ms)."""
    from fec.features import load_ds
    from fec.machines import MAST_CH, MAST_COILS
    ds = load_ds("mast_public_test")
    offs = ds["info"]["offsets_ms"]
    k0, k5, k20 = offs.index(0.0), offs.index(-5.0), offs.index(-20.0)
    cur = np.asarray(ds["cur"][np.asarray(g)], np.float64)
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    c0, c5, c20 = (np.nan_to_num(cur[:, ci, k]) for k in (k0, k5, k20))
    u, l = MAST_COILS.index("magnetics_p6u_current"), MAST_COILS.index("magnetics_p6l_current")
    p6 = lambda c: c[:, u] - c[:, l]
    return np.concatenate([(c0 - c5) / 5.0, (c0 - c20) / 20.0,
                           np.stack([p6(c0), (p6(c0) - p6(c5)) / 5.0,
                                     (p6(c0) - p6(c20)) / 20.0], 1)], 1)


# ============================================================================ truth
def latent_draw(fr, prior, seed):
    """Per-segment smooth hidden latents: J shape (B family or independent FF'), dv, b_r,
    passive resistance scale r (constant per segment)."""
    from fec.twin_mast import smooth_track
    rng = np.random.default_rng(seed)
    sh = fr["shot"]                                      # segment ids
    b0 = smooth_track(rng, sh, 0.0, 0.7)
    a1 = smooth_track(rng, sh, 1.0, 3.0)
    n1 = smooth_track(rng, sh, 0.5, 3.0)
    if prior["gen_ff"]:
        a2 = smooth_track(rng, sh, 1.0, 3.0)
        n2 = smooth_track(rng, sh, 0.5, 3.0)
    else:
        a2, n2 = a1.copy(), n1.copy()
    dv = smooth_track(rng, sh, -prior["dv_max"], prior["dv_max"], jitter=0.1)
    br = smooth_track(rng, sh, -prior["br_max"], prior["br_max"], jitter=0.1)
    useg, inv = np.unique(sh, return_inverse=True)
    r = np.exp(rng.uniform(np.log(prior["r_lo"]), np.log(prior["r_hi"]), len(useg)))[inv]
    return np.stack([b0, a1, n1, a2, n2], 1), dv, br, r


def ext_currents(fr, dv, br, r):
    return np.concatenate([fr["I"], passive_currents(fr["g"], r), dv[:, None], br[:, None]], 1)


def solve_truth2(S, Ix, Ip, prof, n_sec=3):
    """Force-free truth (as fec.twin_mast.solve_truth) for generic external columns Ix."""
    B = len(Ip)
    o0 = S.solve_gen(Ix, Ip, prof, np.zeros(B), n_iter=200, Rinit=np.full(B, 0.9))
    z0, c0 = np.zeros(B), o0["c"].cpu().numpy()
    o1 = S.solve_gen(Ix, Ip, prof, np.full(B, 0.05), n_iter=80, I0=o0["I"], c0=o0["c"])
    z1, c1 = np.full(B, 0.05), o1["c"].cpu().numpy()
    best = o0
    for k in range(n_sec):
        den = np.where(np.abs(c1 - c0) > 1e-12, c1 - c0, np.nan)
        z2 = np.nan_to_num(np.clip(z1 - c1 * (z1 - z0) / den, -0.4, 0.4), nan=z1)
        o2 = S.solve_gen(Ix, Ip, prof, z2, n_iter=120 if k < n_sec - 1 else 200,
                         I0=best["I"], c0=best["c"])
        z0, c0, z1, c1 = z1, c1, z2, o2["c"].cpu().numpy()
        best = o2
    t = lambda x: x.detach().cpu().numpy()
    return dict(psi=t(best["psi"]).astype(np.float32), psi_ax=t(best["psi_ax"]),
                psi_b=t(best["psi_b"]), Rax=t(best["Rax"]), Zax=t(best["Zax"]),
                c=t(best["c"]), dI=t(best["dI"]), Zt=z1,
                xfound=(t(best["xu"][0]) | t(best["xl"][0])))


def screen_select2(S, fr, prior, K, n_screen, seed):
    """K latent draws per segment, keep the one whose Z=0 screening axis follows the real
    Te-peak radius best (fec.amort_gen.screen_select with the v2 latents)."""
    from fec.twin_mast import real_te_peak
    B = len(fr["Ip"])
    pk = real_te_peak(fr)
    cands = []
    for k in range(K):
        prof, dv, br, r = latent_draw(fr, prior, seed + k)
        Ix = ext_currents(fr, dv, br, r)
        o = S.solve_gen(Ix, fr["Ip"], prof, np.zeros(B), n_iter=n_screen,
                        Rinit=np.full(B, 0.9))
        Rax = o["Rax"].cpu().numpy()
        ok = (o["dI"].cpu().numpy() < 1e-2) & (o["xu"][0] | o["xl"][0]).cpu().numpy() & \
            (Rax > 0.47) & (Rax < 1.28)
        cands.append((prof, dv, br, r, Rax, ok))
    sh = fr["shot"]
    sel = np.zeros(B, int)
    for s_ in np.unique(sh):
        ix = np.flatnonzero(sh == s_)
        cost = []
        for c in cands:
            d = np.abs(c[4][ix] - pk[ix])
            d = np.where(np.isfinite(d), d, 0.05)
            cost.append(np.mean(np.where(c[5][ix], d, 0.5)))
        sel[ix] = int(np.argmin(cost))
    rr = np.arange(B)
    pick = lambda j: np.stack([c[j] for c in cands])[sel, rr]
    return pick(0), pick(1), pick(2), pick(3), sel, pk


# ============================================================================ Thomson v2
def synth_thomson2(fr, truth, gR, gZ, prior, seed):
    """Synthetic Te/ne on the NOMINAL channel radii with the v2 realism layers (see module
    doc).  fr["shot"] = segment ids (profile tracks), fr["shot_real"] = real shot (nuisances
    that are shot-level: registration, chord height, calibration, channel noise factors)."""
    from fec.kinetic import chord_psiN
    from fec.twin_mast import smooth_track
    rng = np.random.default_rng(seed)
    sh = fr["shot"]
    shr = fr.get("shot_real", sh)
    B, nch = fr["R"].shape
    te0 = smooth_track(rng, sh, 300.0, 1500.0)
    ta = smooth_track(rng, sh, 1.0, 4.0)
    tb = smooth_track(rng, sh, 0.8, 2.5)
    tsep = smooth_track(rng, sh, 5.0, 40.0)
    ped = smooth_track(rng, sh, 0.0, 1.0) > 0.6
    ne0 = smooth_track(rng, sh, 1.5e19, 5e19)
    nc = smooth_track(rng, sh, 2.0, 8.0)
    nd = smooth_track(rng, sh, 0.3, 1.2)
    lamT = smooth_track(rng, sh, 0.02, 0.08)
    lamN = smooth_track(rng, sh, 0.03, 0.10)
    us, inv = np.unique(shr, return_inverse=True)
    ns = len(us)
    dR = np.clip(prior["dR_sd"] * rng.standard_t(3, ns), -prior["dR_max"], prior["dR_max"])[inv]
    zc = rng.uniform(prior["z_lo"], prior["z_hi"], ns)[inv]
    jit = rng.normal(0, prior["jit_sd"], (ns, nch))[inv]
    cT = np.exp(prior["cal_sd"] * rng.standard_normal(ns))[inv]
    cN = np.exp(prior["cal_sd"] * rng.standard_normal(ns))[inv]
    sysT = np.exp(prior["sys_sd"] * rng.standard_normal((ns, 2)))[inv]
    sysN = np.exp(prior["sys_sd"] * rng.standard_normal((ns, 2)))[inv]
    nfac = np.exp(0.3 * rng.standard_normal((ns, nch)))[inv]        # per-channel noise level
    eA = rng.uniform(0, prior["edge_amp"], (ns, 2))[inv]
    eK = rng.uniform(0.5, 2.0, (ns, 2))[inv]
    eP = rng.uniform(0, 2 * np.pi, (ns, 2))[inv]
    sig = rng.uniform(0.03, 0.08, B)
    is_edge_sys = np.arange(nch) >= N_CORE
    te = np.full((B, nch), np.nan); ne = np.full((B, nch), np.nan)
    for i in range(B):
        R = fr["R"][i]
        ok = np.isfinite(R)
        pn = np.full(nch, np.nan)
        Rm = R[ok] + dR[i] + jit[i][ok]
        pn[ok] = chord_psiN(truth["psi"][i].astype(np.float64), truth["psi_ax"][i],
                            truth["psi_b"][i], gR, gZ, Rm, np.full(ok.sum(), zc[i]))
        x = np.clip(pn, 0, 1)
        core = (1 - x ** ta[i]) ** tb[i]
        if ped[i]:
            pedf = 0.5 * (1 - np.tanh((pn - 0.95) / 0.02))
            core = 0.6 * core + 0.4 * pedf * np.clip(1 - x ** 8, 0, 1) ** 0.3
        inside = pn < 1
        sol_d = np.clip(pn - 1, 0, None)
        t = np.where(inside, tsep[i] + (te0[i] - tsep[i]) * core,
                     tsep[i] * np.exp(-sol_d / lamT[i]))
        n = np.where(inside, 0.1 * ne0[i] + 0.9 * ne0[i] * (1 - x ** nc[i]) ** nd[i],
                     0.1 * ne0[i] * np.exp(-sol_d / lamN[i]))
        # edge/SOL-only model discrepancy (smooth ripple in psiN beyond 0.85)
        w = np.clip((pn - 0.85) / 0.15, 0, 1)
        t = t * (1 + w * eA[i, 0] * np.sin(2 * np.pi * eK[i, 0] * (pn - 0.85) / 0.3 + eP[i, 0]))
        n = n * (1 + w * eA[i, 1] * np.sin(2 * np.pi * eK[i, 1] * (pn - 0.85) / 0.3 + eP[i, 1]))
        # calibration (shot-level, correlated across channels; core/edge system factor)
        t = t * cT[i] * np.where(is_edge_sys, sysT[i, 1], sysT[i, 0])
        n = n * cN[i] * np.where(is_edge_sys, sysN[i, 1], sysN[i, 0])
        # heteroscedastic noise
        sT = np.sqrt(sig[i] ** 2 + (0.06 * np.sqrt(50.0 / np.maximum(t, 1.0))) ** 2) * nfac[i]
        sN = np.sqrt(sig[i] ** 2 + (0.06 * np.sqrt(5e18 / np.maximum(n, 1e17))) ** 2) * nfac[i]
        t = t * (1 + np.minimum(sT, 0.5) * rng.standard_normal(nch)) + \
            3.0 * rng.standard_normal(nch)
        n = n * (1 + np.minimum(sN, 0.5) * rng.standard_normal(nch))
        out = rng.random(nch) < 0.02
        t[out] = rng.uniform(0, 1.2 * te0[i], out.sum())
        out = rng.random(nch) < 0.02
        n[out] = rng.uniform(0, 1.2 * ne0[i], out.sum())
        # missing: real pattern + 3% random + contiguous block (in R order)
        blk = np.zeros(nch, bool)
        if rng.random() < prior["p_block"]:
            order = np.argsort(np.where(ok, R, np.inf))[:ok.sum()]
            L = int(rng.integers(4, 26))
            u_ = rng.random()
            st = 0 if u_ < 0.25 else (len(order) - L if u_ < 0.5 else
                                      int(rng.integers(0, max(1, len(order) - L))))
            blk[order[max(0, st):st + L]] = True
        miss = ~np.isfinite(fr["te_real"][i]) | (rng.random(nch) < 0.03) | ~ok | blk
        miss_n = ~np.isfinite(fr["ne_real"][i]) | (rng.random(nch) < 0.03) | ~ok | blk
        te[i] = np.where(miss, np.nan, np.maximum(t, 1.0))
        ne[i] = np.where(miss_n, np.nan, np.maximum(n, 1e17))
    return dict(te=te, ne=ne, dR=dR, zc=zc, cT=cT, cN=cN)


# ============================================================================ drivers
def gen_chunk(S, spec, M, fr, prior, K, n_screen, n_real, seed):
    """One latent draw (screened) for the frames in fr -> dict of arrays to save."""
    import torch
    from fec.amort_gen import midplane_rin_rout
    from fec.twin_mast import truth_valid
    B = len(fr["Ip"])
    if S.dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    prof, dv, br, r, sel, pk = screen_select2(S, fr, prior, K, n_screen, seed)
    t1 = time.time()
    Ix = ext_currents(fr, dv, br, r)
    tr = solve_truth2(S, Ix, fr["Ip"], prof)
    t2 = time.time()
    valid = truth_valid(tr)
    tes, nes, regs = [], [], []
    for k in range(n_real):
        ts = synth_thomson2(fr, tr, spec.gR, spec.gZ, prior, seed=seed + 7919 * (k + 1))
        tes.append(ts["te"].astype(np.float32)); nes.append(ts["ne"].astype(np.float32))
        regs.append(np.stack([ts["dR"], ts["zc"], ts["cT"], ts["cN"]], 1))
    Rin, Rout = midplane_rin_rout(tr["psi"], tr["psi_ax"], tr["psi_b"], spec.gR, spec.gZ)
    with np.errstate(invalid="ignore"):
        real = valid & np.isfinite(Rout) & (Rout <= 1.55) & (Rin >= 0.20)
    psi_coil = (np.asarray(fr["I"], float) @ M.T).reshape(B, 65, 65).astype(np.float32)
    out = dict(prof=prof, dv=dv, br=br, rscale=r, track=sel, te_peak_real=pk,
               te=np.stack(tes), ne=np.stack(nes), nuis=np.stack(regs),
               valid=valid, realistic=real, Rin=Rin, Rout=Rout, psi_coil=psi_coil, **tr)
    tm = (t1 - t0, t2 - t1, time.time() - t2)
    return out, tm


def gen(tag, every=2, n_lat=3, chunk=2000, K=4, n_screen=100, n_real=2, seed=0):
    """Training set v2: frames of the non-twin shots (every `every`-th frame per shot),
    n_lat independent latent draws each."""
    from fec.amort_gen import take, train_frames
    fr_all = train_frames()
    keep = np.zeros(len(fr_all["g"]), bool)
    for s in np.unique(fr_all["shot"]):
        ix = np.flatnonzero(fr_all["shot"] == s)
        keep[ix[::every]] = True
    fr_all = take(fr_all, np.flatnonzero(keep))
    d = OUT / f"gen_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    segs = np.unique(fr_all["seg"])
    rng.shuffle(segs)
    cnt = np.bincount(fr_all["seg"], minlength=segs.max() + 1)
    groups, cur, n = [], [], 0
    for s in segs:
        if n + cnt[s] > chunk and cur:
            groups.append(cur); cur, n = [], 0
        cur.append(s); n += cnt[s]
    if cur:
        groups.append(cur)
    jobs = [(lat, ci) for ci in range(len(groups)) for lat in range(n_lat)]   # ci-major
    print(f"gen2 {tag}: {len(fr_all['g'])} real frames x {n_lat} latents, {len(groups)} chunks"
          f" per latent, K={K}, n_real={n_real}", flush=True)
    S, spec, M, pm = make_solver2()
    t_all = time.time(); n_done = 0
    for lat, ci in jobs:
        f = d / f"chunk_l{lat}_{ci:04d}.npz"
        if f.exists():
            continue
        ix = np.flatnonzero(np.isin(fr_all["seg"], groups[ci]))
        fr = take(fr_all, ix)
        fr["shot_real"] = fr["shot"]
        fr["shot"] = fr["seg"]
        sd = 1_000_003 * (seed + 1) + 10_007 * lat + 101 * ci
        out, tm = gen_chunk(S, spec, M, fr, PRIORS["TRAIN"], K, n_screen, n_real, sd)
        np.savez(f, g=fr["g"], shot=fr["shot_real"], seg=fr["seg"], fidx=fr["fidx"],
                 I=fr["I"], Ip=fr["Ip"], Itf=fr["Itf"], R=fr["R"].astype(np.float32),
                 lat=np.full(len(ix), lat), **out)
        n_done += len(ix)
        print(f"lat {lat} chunk {ci}: B {len(ix)} screen {tm[0]:.0f}s truth {tm[1]:.0f}s ts "
              f"{tm[2]:.0f}s | valid {out['valid'].mean():.3f} realistic "
              f"{out['realistic'].mean():.3f} | Zax sd {np.std(out['Zax']) * 100:.1f}cm | "
              f"total {n_done} truths {time.time() - t_all:.0f}s", flush=True)
    print(f"GEN2_DONE {tag} truths {n_done} wall {time.time() - t_all:.0f}s", flush=True)


def build_twin2(scen, seed=0, K=4, n_screen=100, fr=None, out_dir=None, device="cuda",
                n_real=1):
    """Twin-v2 test truths on the twin's 1000 frames (fec_work/twin/truth_B.npz inputs),
    written in fec.twin_mast format under FEC_WORK/amort/twin2 (+ x_* extras)."""
    import fec.twin_mast as tm
    if fr is None:
        tm.OUT = WORK / "twin"
        fr, _, _, _, _ = tm.load_truth("B")              # inputs only (fr_*), read-only
    fr = dict(fr)
    fr["shot_real"] = fr["shot"]                         # twin windows: one segment per shot
    out_dir = TWIN2 if out_dir is None else Path(out_dir)
    S, spec, M, pm = make_solver2(device=device)
    prior = PRIORS[scen]
    sd = 777 + 1000 * (ord(scen[0]) - 64) + seed
    out, tmg = gen_chunk(S, spec, M, fr, prior, K, n_screen, n_real, sd)
    tr = {k: out[k] for k in ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI", "Zt",
                              "xfound", "psi_coil")}
    tr["track"] = out["track"]; tr["te_peak_real"] = out["te_peak_real"]
    ts = dict(te=out["te"][0].astype(np.float64), ne=out["ne"][0].astype(np.float64))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"truth_{scen}.npz", prof=out["prof"], dv=out["dv"], sol_fraction=0.5,
             **{f"fr_{k}": v for k, v in fr.items() if k != "shot_real"},
             **{f"tr_{k}": v for k, v in tr.items()}, **{f"ts_{k}": v for k, v in ts.items()},
             x_br=out["br"], x_rscale=out["rscale"], x_nuis=out["nuis"][0])
    print(f"twin2 {scen}: valid {out['valid'].mean():.3f} realistic {out['realistic'].mean():.3f}"
          f" Zax sd {np.std(out['Zax']) * 100:.1f}cm Rax pct "
          f"{np.percentile(out['Rax'], [5, 50, 95]).round(3).tolist()} ({sum(tmg):.0f}s)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "twin"])
    ap.add_argument("--tag", default="v2main")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--n-lat", type=int, default=3)
    ap.add_argument("--n-real", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scen", default="B2,C2")
    a = ap.parse_args()
    if a.cmd == "gen":
        gen(a.tag, a.every, a.n_lat, a.chunk, a.K, n_real=a.n_real, seed=a.seed)
    else:
        for s in a.scen.split(","):
            build_twin2(s, seed=a.seed, K=a.K)


if __name__ == "__main__":
    main()
