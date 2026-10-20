"""Generator v4 (post-G0, sub40 candidate): Thomson calibrated to the REAL MAST input distribution.

Root 2026-10-10 (AMORTIZED_MAST_20261010 §v4): the synthetic ne collapse at the LCFS gave the net
an artificially perfect boundary marker (deep-research triage row 8) and real Thomson shapes were
out of distribution.  v4 re-synthesises Thomson on STORED truths (no new GS solves):
  * synth_thomson4: Te = Tsep + (Te0-Tsep)(1-x^ta)^tb (+ optional pedestal); ne = ne0 [f_sep +
    (1-f_sep) shape(x)], shape = (1-x^nc)^nd with an optional hollow core (1 - h exp(-(x/0.4)^2),
    renormalised); SOL (psiN >= 1): ne = ne0 f_sep exp(-d_R/lam_n), Te = Tsep exp(-d_R/lam_T) with
    d_R the R distance to the chord's nearest LCFS crossing; f_sep, lam_n, lam_T drawn PER REAL
    SHOT; v2 systematics kept (registration, chord height, calibration, heteroscedastic noise,
    edge ripple, block dropout).
  * calibration: hyper-ranges by random search + local refinement minimising the mean KS distance
    between synthetic and real distributions of 10 per-frame INPUT statistics on the same
    non-twin frames (inputs only; never twin shots, labels or scores).

  python -B fec/amort_gen4.py calib --bank v3main --n 1500           (CPU)
  python -B fec/amort_gen4.py resynth --src v3main --tag v4th        (CPU, after calib)
  python -B fec/amort_gen4.py twin --scen B2,C2                      (-> twin2/truth_{B4,C4}.npz)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.amort_gen2 import N_CORE, PRIORS  # noqa: E402
from fec.paths import WORK  # noqa: E402

OUT = WORK / "amort"
HP_FILE = OUT / "v4_hp.json"
# search bounds of the hyper-range parameters (lo, hi)
BOUNDS = dict(te_lo=(150, 500), te_hi=(800, 3000), ta_lo=(0.8, 2.0), ta_hi=(2.0, 6.0),
              tb_lo=(0.5, 1.5), tb_hi=(1.5, 4.0), p_ped=(0.0, 0.6),
              ne_lo=(0.8e19, 2.5e19), ne_hi=(3e19, 8e19), nc_lo=(1.0, 4.0), nc_hi=(4.0, 12.0),
              nd_lo=(0.1, 0.6), nd_hi=(0.6, 2.0), p_hollow=(0.0, 0.5), h_hi=(0.3, 1.0),
              fs_lo=(0.02, 0.2), fs_hi=(0.2, 0.8), ln_lo=(0.01, 0.05), ln_hi=(0.05, 0.4),
              ts_lo=(2.0, 20.0), ts_hi=(20.0, 80.0), lt_lo=(0.005, 0.03), lt_hi=(0.03, 0.2))
STATS = ("te_pk", "ne_pk", "ne_hollow", "te_w", "ne_w", "ne_edge", "te_edge", "ne_inb", "te_inb",
         "dR_nemax")
XS = np.linspace(0, 1, 201)


# ============================================================================ statistics
def input_stats(R, te, ne):
    """Per-frame label-free Thomson statistics (NaN where < 40 usable channels)."""
    from scipy.ndimage import median_filter
    B = len(R)
    out = {k: np.full(B, np.nan) for k in STATS}
    for i in range(B):
        r, t, n = R[i], te[i], ne[i]
        ok = np.isfinite(r) & np.isfinite(t) & np.isfinite(n) & (r > 0.2) & (r < 1.55)
        if ok.sum() < 40:
            continue
        o = np.argsort(r[ok]); rr = r[ok][o]
        tt = median_filter(t[ok][o], 7, mode="nearest")
        nn = median_filter(n[ok][o], 7, mode="nearest")
        it, inn = int(np.argmax(tt)), int(np.argmax(nn))
        if tt[it] <= 0 or nn[inn] <= 0:
            continue
        out["te_pk"][i] = tt[it]; out["ne_pk"][i] = nn[inn]
        out["ne_hollow"][i] = nn[it] / nn[inn]
        out["te_w"][i] = np.ptp(rr[tt >= 0.5 * tt[it]])
        out["ne_w"][i] = np.ptp(rr[nn >= 0.5 * nn[inn]])
        e, b = rr > 1.3, rr < 0.3
        if e.any():
            out["ne_edge"][i] = nn[e].mean() / nn[inn]; out["te_edge"][i] = tt[e].mean() / tt[it]
        if b.any():
            out["ne_inb"][i] = nn[b].mean() / nn[inn]; out["te_inb"][i] = tt[b].mean() / tt[it]
        out["dR_nemax"][i] = rr[inn] - rr[it]
    return out


def ks(a, b):
    a = np.sort(a[np.isfinite(a)]); b = np.sort(b[np.isfinite(b)])
    if len(a) < 10 or len(b) < 10:
        return 1.0
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, "right") / len(a) -
                               np.searchsorted(b, x, "right") / len(b))))


# ============================================================================ Thomson v4
def _logu(rng, sh, lo, hi):
    from fec.twin_mast import smooth_track
    return np.exp(smooth_track(rng, sh, np.log(lo), np.log(hi)))


def lcfs_dist(Rm, pn):
    """R distance of each chord point to the nearest psiN = 1 crossing along the chord."""
    o = np.argsort(Rm)
    r, p = Rm[o], pn[o] - 1.0
    s = np.flatnonzero(np.sign(p[:-1]) != np.sign(p[1:]))
    d = np.full(len(Rm), np.nan)
    if len(s) == 0:
        d[o] = np.where(p > 0, 0.3 * p, 0.0)
        return d
    rc = r[s] - p[s] * (r[s + 1] - r[s]) / (p[s + 1] - p[s])
    d[o] = np.min(np.abs(r[:, None] - rc[None]), 1)
    return d


def synth_thomson4(fr, truth, gR, gZ, hp, prior=None, seed=0, tsep_frame=None, lamt_frame=None):
    """Te/ne on the nominal channel radii (see module doc). fr["shot"] = segment ids (profile
    tracks), fr["shot_real"] = real shot (registration, chord height, calibration, SOL)."""
    from fec.kinetic import chord_psiN
    from fec.twin_mast import smooth_track
    prior = PRIORS["TRAIN"] if prior is None else prior
    rng = np.random.default_rng(seed)
    sh = fr["shot"]
    shr = fr.get("shot_real", sh)
    B, nch = fr["R"].shape
    te0 = _logu(rng, sh, hp["te_lo"], hp["te_hi"])
    ta = smooth_track(rng, sh, hp["ta_lo"], hp["ta_hi"])
    tb = smooth_track(rng, sh, hp["tb_lo"], hp["tb_hi"])
    ped = smooth_track(rng, sh, 0.0, 1.0) < hp["p_ped"]
    ne0 = _logu(rng, sh, hp["ne_lo"], hp["ne_hi"])
    nc = smooth_track(rng, sh, hp["nc_lo"], hp["nc_hi"])
    nd = smooth_track(rng, sh, hp["nd_lo"], hp["nd_hi"])
    hol = np.where(smooth_track(rng, sh, 0.0, 1.0) < hp["p_hollow"],
                   smooth_track(rng, sh, 0.0, hp["h_hi"]), 0.0)
    tsep = smooth_track(rng, sh, hp["ts_lo"], hp["ts_hi"])
    if tsep_frame is not None:               # v6: separatrix Te tied to the geometry offset delta
        tsep = np.asarray(tsep_frame, float)
    us, inv = np.unique(shr, return_inverse=True)
    ns = len(us)
    fsep = rng.uniform(hp["fs_lo"], hp["fs_hi"], ns)[inv]
    lamn = np.exp(rng.uniform(np.log(hp["ln_lo"]), np.log(hp["ln_hi"]), ns))[inv]
    lamt = np.exp(rng.uniform(np.log(hp["lt_lo"]), np.log(hp["lt_hi"]), ns))[inv]
    if lamt_frame is not None:
        lamt = np.asarray(lamt_frame, float)
    dR = np.clip(prior["dR_sd"] * rng.standard_t(3, ns), -prior["dR_max"], prior["dR_max"])[inv]
    zc = rng.uniform(prior["z_lo"], prior["z_hi"], ns)[inv]
    jit = rng.normal(0, prior["jit_sd"], (ns, nch))[inv]
    cT = np.exp(prior["cal_sd"] * rng.standard_normal(ns))[inv]
    cN = np.exp(prior["cal_sd"] * rng.standard_normal(ns))[inv]
    sysT = np.exp(prior["sys_sd"] * rng.standard_normal((ns, 2)))[inv]
    sysN = np.exp(prior["sys_sd"] * rng.standard_normal((ns, 2)))[inv]
    nfac = np.exp(0.3 * rng.standard_normal((ns, nch)))[inv]
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
        dl = np.zeros(nch)
        dl[ok] = lcfs_dist(Rm, pn[ok])
        x = np.clip(pn, 0, 1)
        core = (1 - x ** ta[i]) ** tb[i]
        if ped[i]:
            pedf = 0.5 * (1 - np.tanh((pn - 0.95) / 0.02))
            core = 0.6 * core + 0.4 * pedf * np.clip(1 - x ** 8, 0, 1) ** 0.3
        shp = lambda xx: (1 - xx ** nc[i]) ** nd[i] * (1 - hol[i] * np.exp(-(xx / 0.4) ** 2))
        nsh = shp(x) / max(shp(XS).max(), 1e-6)
        inside = pn < 1
        t = np.where(inside, tsep[i] + (te0[i] - tsep[i]) * core,
                     tsep[i] * np.exp(-dl / lamt[i]))
        n = np.where(inside, ne0[i] * (fsep[i] + (1 - fsep[i]) * nsh),
                     ne0[i] * fsep[i] * np.exp(-dl / lamn[i]))
        w = np.clip((pn - 0.85) / 0.15, 0, 1)
        t = t * (1 + w * eA[i, 0] * np.sin(2 * np.pi * eK[i, 0] * (pn - 0.85) / 0.3 + eP[i, 0]))
        n = n * (1 + w * eA[i, 1] * np.sin(2 * np.pi * eK[i, 1] * (pn - 0.85) / 0.3 + eP[i, 1]))
        t = t * cT[i] * np.where(is_edge_sys, sysT[i, 1], sysT[i, 0])
        n = n * cN[i] * np.where(is_edge_sys, sysN[i, 1], sysN[i, 0])
        sT = np.sqrt(sig[i] ** 2 + (0.06 * np.sqrt(50.0 / np.maximum(t, 1.0))) ** 2) * nfac[i]
        sN = np.sqrt(sig[i] ** 2 + (0.06 * np.sqrt(5e18 / np.maximum(n, 1e17))) ** 2) * nfac[i]
        t = t * (1 + np.minimum(sT, 0.5) * rng.standard_normal(nch)) + \
            3.0 * rng.standard_normal(nch)
        n = n * (1 + np.minimum(sN, 0.5) * rng.standard_normal(nch))
        out = rng.random(nch) < 0.02
        t[out] = rng.uniform(0, 1.2 * te0[i], out.sum())
        out = rng.random(nch) < 0.02
        n[out] = rng.uniform(0, 1.2 * ne0[i], out.sum())
        blk = np.zeros(nch, bool)
        if rng.random() < prior["p_block"]:
            order = np.argsort(np.where(ok, R, np.inf))[:ok.sum()]
            L = int(rng.integers(4, 26)); u_ = rng.random()
            st = 0 if u_ < 0.25 else (len(order) - L if u_ < 0.5 else
                                      int(rng.integers(0, max(1, len(order) - L))))
            blk[order[max(0, st):st + L]] = True
        miss = ~np.isfinite(fr["te_real"][i]) | (rng.random(nch) < 0.03) | ~ok | blk
        miss_n = ~np.isfinite(fr["ne_real"][i]) | (rng.random(nch) < 0.03) | ~ok | blk
        te[i] = np.where(miss, np.nan, np.maximum(t, 1.0))
        ne[i] = np.where(miss_n, np.nan, np.maximum(n, 1e17))
    return dict(te=te, ne=ne)


# ============================================================================ calibration
_BANK = {}


def _init(bank_file):
    z = np.load(bank_file)
    _BANK.update({k: z[k] for k in z.files})


def _score(args):
    hp, seed = args
    from fec.machines import mast_setup
    spec, _ = mast_setup()
    b = _BANK
    fr = dict(shot=b["seg"], shot_real=b["shot"], R=b["R"], te_real=b["te_real"],
              ne_real=b["ne_real"])
    ts = synth_thomson4(fr, dict(psi=b["psi"], psi_ax=b["psi_ax"], psi_b=b["psi_b"]),
                        spec.gR, spec.gZ, hp, seed=seed)
    st = input_stats(b["R"], ts["te"], ts["ne"])
    real = {k: b[f"real_{k}"] for k in STATS}
    d = {k: ks(st[k], real[k]) for k in STATS}
    return float(np.mean(list(d.values()))), d


def sample_hp(rng, center=None, scale=0.15):
    hp = {}
    for k, (lo, hi) in BOUNDS.items():
        if center is None:
            v = rng.uniform(lo, hi)
        else:
            v = np.clip(center[k] + scale * (hi - lo) * rng.standard_normal(), lo, hi)
        hp[k] = float(v)
    return hp


def make_bank(src, n, seed=0, realistic=True):
    """n valid (by default realistic: midplane LCFS inside the Thomson chord, the twin study's
    0.20 <= R_in, R_out <= 1.55 m) truths of a stored set + their REAL Thomson input statistics
    (same frames)."""
    from fec.amort_gen import frame_inputs, load_gen
    z = load_gen(src, keys=["g", "shot", "seg", "psi", "psi_ax", "psi_b", "R", "realistic"])
    rng = np.random.default_rng(seed)
    pool_ = np.flatnonzero(z["realistic"]) if realistic else np.arange(len(z["g"]))
    i = np.sort(rng.choice(pool_, min(n, len(pool_)), replace=False))
    fr = frame_inputs(z["g"][i])
    real = input_stats(fr["R"], fr["te_real"], fr["ne_real"])
    f = OUT / f"v4_bank_{src}_{n}{'_real' if realistic else ''}.npz"
    np.savez(f, g=z["g"][i], shot=z["shot"][i], seg=z["seg"][i], psi=z["psi"][i],
             psi_ax=z["psi_ax"][i], psi_b=z["psi_b"][i], R=fr["R"], te_real=fr["te_real"],
             ne_real=fr["ne_real"], **{f"real_{k}": v for k, v in real.items()})
    return f


def calibrate(src="v3main", n=1500, n_rand=240, n_ref=120, nproc=6, seed=0, realistic=True):
    t0 = time.time()
    bank = make_bank(src, n, seed, realistic=realistic)
    rng = np.random.default_rng(seed + 1)
    # reference: the v1/v2 synthetic Thomson model's distance on the same bank (for the record)
    cands = [sample_hp(rng) for _ in range(n_rand)]
    with Pool(nproc, initializer=_init, initargs=(str(bank),)) as pool:
        res = pool.map(_score, [(hp, 17 + j) for j, hp in enumerate(cands)], chunksize=2)
        best = int(np.argmin([r[0] for r in res]))
        print(f"random search: best mean KS {res[best][0]:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        c2 = [sample_hp(rng, cands[best], 0.08) for _ in range(n_ref)]
        res2 = pool.map(_score, [(hp, 5000 + j) for j, hp in enumerate(c2)], chunksize=2)
        b2 = int(np.argmin([r[0] for r in res2]))
        hp, (m, d) = (c2[b2], res2[b2]) if res2[b2][0] < res[best][0] else (cands[best], res[best])
        # fresh-seed confirmation of the chosen ranges
        mc, dc = pool.map(_score, [(hp, 99_991)])[0]
    rec = dict(hp=hp, mean_ks=m, ks=d, confirm_mean_ks=mc, confirm_ks=dc, bank=str(bank),
               src=src, bank_realistic=realistic,
               n=n, n_rand=n_rand, n_ref=n_ref, seconds=time.time() - t0)
    HP_FILE.write_text(json.dumps(rec, indent=1))
    print("CALIB_DONE", json.dumps({k: round(v, 3) for k, v in dc.items()}),
          f"mean {mc:.4f}", flush=True)
    return rec


# ============================================================================ truths (v4e)
EDGES = WORK / "mastedge" / "mast_edges.npz"   # label-free Thomson 40 eV crossings (inputs only)


def screen_select4(S, fr, prior, K, n_screen, seed, r40, ok40, per_frame=True):
    """v3 latents; per segment keep the draw minimising mean(|Rax - real Te-peak R| +
    |Rout - real Thomson 40 eV outboard crossing|) over its frames (Z=0 screening solves).
    Both targets are REAL INPUT features of the same frame (as the twin's Te-peak screening)."""
    from fec.amort_gen import midplane_rin_rout
    from fec.amort_gen2 import ext_currents
    from fec.amort_gen3 import latent_draw3
    from fec.twin_mast import real_te_peak
    B = len(fr["Ip"])
    pk = real_te_peak(fr)
    gR, gZ = S.spec.gR, S.spec.gZ
    cands = []
    for k in range(K):
        prof, dv, br, r = latent_draw3(fr, prior, seed + k)
        o = S.solve_gen(ext_currents(fr, dv, br, r), fr["Ip"], prof, np.zeros(B),
                        n_iter=n_screen, Rinit=np.full(B, 0.9))
        Rax = o["Rax"].cpu().numpy()
        ok = (o["dI"].cpu().numpy() < 1e-2) & (o["xu"][0] | o["xl"][0]).cpu().numpy() &             (Rax > 0.47) & (Rax < 1.28)
        _, Rout = midplane_rin_rout(o["psi"].cpu().numpy(), o["psi_ax"].cpu().numpy(),
                                    o["psi_b"].cpu().numpy(), gR, gZ)
        cands.append((prof, dv, br, r, Rax, ok, Rout))
    sh = fr["shot"]
    cost = []
    for c in cands:
        d1 = np.abs(c[4] - pk); d1 = np.where(np.isfinite(d1), d1, 0.05)
        d2 = np.where(ok40, np.abs(c[6] - r40), 0.05)
        d2 = np.where(np.isfinite(d2), d2, 0.3)
        cost.append(np.where(c[5], d1 + d2, 1.0))
    cost = np.stack(cost)                                   # (K, B)
    if per_frame:      # per-frame choice among the K smooth tracks (the net is per-frame)
        sel = np.argmin(cost, 0)
    else:
        sel = np.zeros(B, int)
        for s_ in np.unique(sh):
            ix = np.flatnonzero(sh == s_)
            sel[ix] = int(np.argmin(cost[:, ix].mean(1)))
    rr = np.arange(B)
    pick = lambda j: np.stack([c[j] for c in cands])[sel, rr]
    return pick(0), pick(1), pick(2), pick(3), sel, pk


def gen_truth(tag, every=2, n_lat=1, chunk=2000, K=16, n_screen=100, seed=0, device="cuda",
              edge_tol=0.05):
    """v4e truths: v3 physics (mixture, kappa <= 3, passive, b_r, dv) with dual screening (Te
    peak + 40 eV outboard edge).  Thomson is NOT synthesised here (resynth with the v4 model)."""
    import torch
    from fec.amort_gen import midplane_rin_rout, take, train_frames
    from fec.amort_gen2 import PRIORS as P2, ext_currents, solve_truth2
    from fec.amort_gen3 import KAPPA_MAX, lcfs_kappa, make_solver3
    from fec.twin_mast import truth_valid
    E = np.load(EDGES)
    fr_all = train_frames()
    keep = np.zeros(len(fr_all["g"]), bool)
    for s in np.unique(fr_all["shot"]):
        ix = np.flatnonzero(fr_all["shot"] == s)
        keep[ix[::every]] = True
    fr_all = take(fr_all, np.flatnonzero(keep))
    d = OUT / f"gen_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    segs = np.unique(fr_all["seg"]); rng.shuffle(segs)
    cnt = np.bincount(fr_all["seg"], minlength=segs.max() + 1)
    groups, cur, n = [], [], 0
    for s in segs:
        if n + cnt[s] > chunk and cur:
            groups.append(cur); cur, n = [], 0
        cur.append(s); n += cnt[s]
    if cur:
        groups.append(cur)
    jobs = [(lat, ci) for ci in range(len(groups)) for lat in range(n_lat)]
    print(f"gen4 truths {tag}: {len(fr_all['g'])} real frames x {n_lat}, {len(groups)} chunks, "
          f"K={K}", flush=True)
    S, spec, M, pm = make_solver3(device=device)
    t_all = time.time(); n_done = 0
    for lat, ci in jobs:
        f = d / f"chunk_l{lat}_{ci:04d}.npz"
        if f.exists():
            continue
        ix = np.flatnonzero(np.isin(fr_all["seg"], groups[ci]))
        fr = take(fr_all, ix)
        fr["shot_real"] = fr["shot"]; fr["shot"] = fr["seg"]
        r40 = E["R_out"][fr["g"]]; ok40 = E["st_out"][fr["g"]] == 1
        t0 = time.time()
        sd = 5_000_011 * (seed + 1) + 10_007 * lat + 101 * ci
        prof, dv, br, r, sel, pk = screen_select4(S, fr, P2["TRAIN"], K, n_screen, sd, r40, ok40,
                                                  per_frame=True)
        t1 = time.time()
        tr = solve_truth2(S, ext_currents(fr, dv, br, r), fr["Ip"], prof)
        valid_pre = truth_valid(tr)
        kappa = lcfs_kappa(tr["psi"], np.flatnonzero(valid_pre))
        kdrop = valid_pre & (kappa > KAPPA_MAX)
        valid = valid_pre & ~kdrop
        Rin, Rout = midplane_rin_rout(tr["psi"], tr["psi_ax"], tr["psi_b"], spec.gR, spec.gZ)
        # edge-consistency rejection: truth LCFS within edge_tol of the frame's real 40 eV crossing
        with np.errstate(invalid="ignore"):
            edge_ok = ~ok40 | (np.abs(Rout - r40) <= edge_tol)
        valid_kappa = valid
        valid = valid & edge_ok
        with np.errstate(invalid="ignore"):
            real = valid & np.isfinite(Rout) & (Rout <= 1.55) & (Rin >= 0.20)
        np.savez(f, valid_kappa=valid_kappa, edge_ok=edge_ok, g=fr["g"], shot=fr["shot_real"], seg=fr["seg"], fidx=fr["fidx"],
                 I=fr["I"], Ip=fr["Ip"], Itf=fr["Itf"], R=fr["R"].astype(np.float32),
                 lat=np.full(len(ix), lat), prof=prof, dv=dv, br=br, rscale=r, track=sel,
                 te_peak_real=pk, r40=r40, ok40=ok40, valid=valid, valid_pre=valid_pre,
                 kappa=kappa, kdrop=kdrop, realistic=real, Rin=Rin, Rout=Rout, **tr)
        n_done += len(ix)
        m = ok40 & valid_kappa
        print(f"lat {lat} chunk {ci}: B {len(ix)} screen {t1 - t0:.0f}s total {time.time() - t0:.0f}s"
              f" | valid {valid.mean():.3f} (before edge cut {valid_kappa.mean():.3f}) realistic "
              f"{real.mean():.3f} | |Rout-R40| med "
              f"{np.nanmedian(np.abs(Rout - r40)[m]) * 100:.1f}cm p90 "
              f"{np.nanpercentile(np.abs(Rout - r40)[m], 90) * 100:.1f}cm | {n_done} truths "
              f"{time.time() - t_all:.0f}s", flush=True)
    print(f"GEN4_DONE {tag} truths {n_done} wall {time.time() - t_all:.0f}s", flush=True)


# ============================================================================ re-synthesis
def resynth(src, tag, n_real=2, seed=0):
    """Thomson v4 for every frame of every chunk of a stored set -> gen_<tag>/<same name>.npz
    holding te, ne (n_real, B, C) only (truths stay in gen_<src>)."""
    from fec.machines import mast_setup
    hp = json.loads(HP_FILE.read_text())["hp"]
    spec, _ = mast_setup()
    d_in, d_out = OUT / f"gen_{src}", OUT / f"gen_{tag}"
    d_out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, f in enumerate(sorted(d_in.glob("chunk_*.npz"))):
        fo = d_out / f.name
        if fo.exists():
            continue
        z = np.load(f)
        fr = dict(shot=z["seg"], shot_real=z["shot"], R=z["R"].astype(np.float64))
        # real missing pattern from the frame inputs (not from any synthetic dropout)
        from fec.amort_gen import frame_inputs
        fi = frame_inputs(z["g"])
        fr["te_real"], fr["ne_real"] = fi["te_real"], fi["ne_real"]
        tr = dict(psi=z["psi"], psi_ax=z["psi_ax"], psi_b=z["psi_b"])
        tes, nes = [], []
        for r in range(n_real):
            ts = synth_thomson4(fr, tr, spec.gR, spec.gZ, hp,
                                seed=7_000_003 * (seed + 1) + 1009 * k + 31 * r)
            tes.append(ts["te"].astype(np.float32)); nes.append(ts["ne"].astype(np.float32))
        np.savez(fo, te=np.stack(tes), ne=np.stack(nes))
        print(f"resynth {f.name}: {len(z['g'])} frames ({time.time() - t0:.0f}s)", flush=True)
    print(f"RESYNTH_DONE {tag}", flush=True)


def build_twin4(scen_src, seed=0):
    """Twin truths (twin2 B2/C2, read) with Thomson v4 -> twin2/truth_{B4,C4}.npz."""
    from fec.amort_gen2 import TWIN2
    from fec.machines import mast_setup
    hp = json.loads(HP_FILE.read_text())["hp"]
    spec, _ = mast_setup()
    z = dict(np.load(TWIN2 / f"truth_{scen_src}.npz"))
    fr = {k[3:]: v for k, v in z.items() if k.startswith("fr_")}
    fr["shot_real"] = fr["shot"]
    tr = dict(psi=z["tr_psi"], psi_ax=z["tr_psi_ax"], psi_b=z["tr_psi_b"])
    ts = synth_thomson4(fr, tr, spec.gR, spec.gZ, hp,
                        prior=PRIORS["C2" if scen_src.startswith("C") else "B2"],
                        seed=4_444 + 7 * seed + ord(scen_src[0]))
    z["ts_te"], z["ts_ne"] = ts["te"], ts["ne"]
    dst = scen_src[0] + "4"
    np.savez(TWIN2 / f"truth_{dst}.npz", **z)
    print(f"twin4 {dst} from {scen_src}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "resynth", "twin", "truth"])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--n-lat", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--bank-all-valid", action="store_true")
    ap.add_argument("--bank", default="v3main")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--n-rand", type=int, default=240)
    ap.add_argument("--n-ref", type=int, default=120)
    ap.add_argument("--src", default="v3main")
    ap.add_argument("--tag", default="v4th")
    ap.add_argument("--scen", default="B2,C2")
    a = ap.parse_args()
    if a.cmd == "calib":
        calibrate(a.bank, a.n, a.n_rand, a.n_ref, realistic=not a.bank_all_valid)
    elif a.cmd == "resynth":
        resynth(a.src, a.tag)
    elif a.cmd == "truth":
        gen_truth(a.tag, a.every, a.n_lat, K=a.K, device=a.device)
    else:
        for s in a.scen.split(","):
            build_twin4(s)


if __name__ == "__main__":
    main()
