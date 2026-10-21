"""Synthetic-twin study of label-free MAST profile selection (no MAST labels anywhere).

Real MAST inputs (coil currents + Ip at offset index 6, Thomson channel radii and their real
missing-channel pattern) drive OUR free-boundary GS solver with random, time-smooth profile
parameters and an optional uniform vertical-field error dv (psi_dv = dv R^2/2) standing in for
unmodelled currents.  Synthetic Thomson Te/ne are flux functions of the truth psi with noise,
outliers and missing channels.  Selection procedures (v1 replica, v2 variants) see only the
synthetic Thomson + real currents; their equilibria are scored against the truth.

Caveat (inverse crime): truth and reconstruction share the solver, grid and coil model.  Errors
from the real coil/vessel model, Thomson calibration or non-GS physics are not represented,
except through dv and the out-of-family profile scenario.

CLI (GPU steps through fec/gpu_slot.py; results under FEC_WORK/twin):
  python -B fec/twin_mast.py truth --scen B          # 8 tracks/shot, matched to real Te peak
  python -B fec/twin_mast.py truth --scen A|C        # variants of B (dv=0 / out-of-family FF')
  python -B fec/twin_mast.py bank --scen B --tag p120   # 120-shape candidate bank
  python -B fec/twin_mast.py analyze --scen B --tag p120 --subset real|all
  python -B fec/twin_mast.py real --scen B --names "v1 real (6 combos);v2 pairs rule"
  python -B fec/twin_mast.py rescore --scen B --subset real|all
  python -B fec/twin_mast.py realcheck               # label-free v1 vs v2 on real inputs
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
from fec.paths import WORK  # noqa: E402

OUT = WORK / "twin"
MU0 = 4e-7 * np.pi
N_TF = 24
DV_MAX = 0.010      # tesla


# ============================================================================ solver subclass
def sol_fraction():
    """Solenoid turn fraction used by every twin run (fec.machines default, recorded)."""
    from fec.machines import MAST_SOL_TURN_FRACTION
    return float(MAST_SOL_TURN_FRACTION)


def make_solver(device="cuda", sol=None):
    """TwinSolver with the coil matrix augmented by a uniform-Bz column (psi = R^2/2 per T)."""
    from fec.machines import mast_setup
    spec, M = mast_setup(sol_fraction=sol_fraction() if sol is None else sol)
    RR, _ = np.meshgrid(spec.gR, spec.gZ)
    Mx = np.concatenate([M, (RR ** 2 / 2).reshape(-1, 1)], 1)
    return TwinSolver(spec, Mx, device=device), spec, M


def _twin_solver_cls():
    import torch
    from fec.gs import Solver

    class _TS(Solver):
        """Solver with a generalised two-shape profile (exactly fec.gs when shapes agree).

        prof (B,5) = beta0, a1, n1, a2, n2:
          J ~ beta0 R/R0 (1-psiN^a1)^n1 + (1-beta0) R0/R (1-psiN^a2)^n2
        """

        def solve_gen(self, Iext, Ip, prof, Zt, n_iter=150, omega=0.4, kc=0.05, I0=None,
                      c0=None, Rinit=None):
            dev, dt = self.dev, self.dt
            t = lambda x: torch.as_tensor(x, device=dev, dtype=dt)
            Iext, Ip, prof, Zt = t(Iext), t(Ip), t(prof), t(Zt)
            B = Iext.shape[0]
            psi_ext = (Iext @ self.Mext.T).reshape(B, self.nz, self.nr)
            Rs, Zs = self.Rs, self.Zs
            R0 = self.spec.R0
            if I0 is None:
                Rc = t(Rinit) if Rinit is not None else torch.full((B,), R0, device=dev,
                                                                   dtype=dt)
                w = torch.exp(-((Rs[None] - Rc[:, None]) ** 2 +
                                (Zs[None] - Zt[:, None]) ** 2) / 0.05)
                I = Ip[:, None] * w / w.sum(1, keepdim=True)
            else:
                I = t(I0)
            c = t(c0) if c0 is not None else torch.zeros(B, device=dev, dtype=dt)
            fb = (self.RR ** 2)[None] * (self.ZZ[None] - Zt[:, None, None])
            b0 = prof[:, 0:1]; a1 = prof[:, 1:2]; n1 = prof[:, 2:3]
            a2 = prof[:, 3:4]; n2 = prof[:, 4:5]
            Npix = self.nz * self.nr
            prev_mask = None
            for it in range(n_iter):
                psi = psi_ext + (I @ self.G.T).reshape(B, self.nz, self.nr) + \
                    c[:, None, None] * fb
                psi_ax, Rax, Zax = self.find_axis(psi)
                xp = self.find_xpoints(psi, Rax, Zax, psi_ax)
                (fu, vu, _, _), (fl, vl, _, _) = xp
                neg = torch.full_like(vu, -1e30)
                psi_b = torch.maximum(torch.where(fu, vu, neg), torch.where(fl, vl, neg))
                nox = ~(fu | fl)
                if nox.any():
                    psi_b = torch.where(nox, psi_ax - 0.05 * psi_ax.abs().clamp_min(1e-3),
                                        psi_b)
                seedm = prev_mask if (it % 10 != 0) else None
                mask, psiN = self.core_mask(psi, psi_ax, psi_b, Rax, Zax, xp, seed_mask=seedm)
                prev_mask = mask
                pn = psiN.reshape(B, -1)[:, self.sidx].clamp(0, 1)
                ms = mask.reshape(B, -1)[:, self.sidx].to(dt)
                shape = (b0 * Rs[None] / R0 * (1 - pn ** a1).clamp_min(0) ** n1 +
                         (1 - b0) * R0 / Rs[None] * (1 - pn ** a2).clamp_min(0) ** n2) * ms
                tot = shape.sum(1, keepdim=True).clamp_min(1e-30)
                Inew = Ip[:, None] * shape / tot
                Zc_new = (Inew * Zs[None]).sum(1) / Ip
                delta = Zt - Zc_new
                full = torch.zeros(B, Npix, device=dev, dtype=dt)
                full[:, self.sidx] = Inew
                full = self._shiftZ(full.reshape(B, self.nz, self.nr), delta).reshape(B, Npix)
                Inew = full[:, self.sidx]
                Inew = Ip[:, None] * Inew / Inew.sum(1, keepdim=True).clamp_min(1e-30)
                c = c + kc * delta
                dI = (Inew - I).abs().sum(1) / Ip
                I = (1 - omega) * I + omega * Inew
            psi = psi_ext + (I @ self.G.T).reshape(B, self.nz, self.nr) + c[:, None, None] * fb
            psi_ax, Rax, Zax = self.find_axis(psi)
            xp = self.find_xpoints(psi, Rax, Zax, psi_ax)
            (fu, vu, Rxu, Zxu), (fl, vl, Rxl, Zxl) = xp
            neg = torch.full_like(vu, -1e30)
            psi_b = torch.maximum(torch.where(fu, vu, neg), torch.where(fl, vl, neg))
            return dict(psi=psi, I=I, c=c, psi_ax=psi_ax, Rax=Rax, Zax=Zax, psi_b=psi_b,
                        xu=(fu, Rxu, Zxu, vu), xl=(fl, Rxl, Zxl, vl), dI=dI, delta=delta,
                        psi_ext=psi_ext)
    return _TS


def TwinSolver(spec, Mx, device="cuda"):
    return _twin_solver_cls()(spec, Mx, device=device)


# ============================================================================ frames
def load_frames(n_shots=40, win=25, seed=0):
    """~n_shots*win frames: contiguous windows of real MAST inputs (whole-shot blocks)."""
    from fec.features import load_ds
    from fec.machines import MAST_CH, MAST_COILS
    from fec.mast_v1 import thomson_frame
    ds = load_ds("mast_public_test")
    info = ds["info"]
    shot = ds["shot"]
    rng = np.random.default_rng(seed)
    nfr = np.bincount(shot, minlength=len(info["order"]))
    elig = np.flatnonzero(nfr >= win)
    shots = np.sort(rng.choice(elig, n_shots, replace=False))
    gs = []
    for s in shots:
        idx = np.flatnonzero(shot == s)
        idx = idx[np.argsort(ds["fidx"][idx])]
        st = rng.integers(0, len(idx) - win + 1)
        gs.append(idx[st:st + win])
    g = np.concatenate(gs)
    cur = np.asarray(ds["cur"][g], dtype=np.float64)[:, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    Rl, tel, nel = [], [], []
    cache = {}
    for k in g:
        stem = info["order"][shot[k]]
        if stem not in cache:
            cache.clear()
            cache[stem] = dict(np.load(WORK / "shots" / "mast_public_test" / f"{stem}.npz"))
        R, te, ne = thomson_frame(cache[stem], ds["fidx"][k])
        Rl.append(R); tel.append(te); nel.append(ne)
    nch = max(len(r) for r in Rl)
    pad = lambda L: np.stack([np.pad(np.asarray(x, float), (0, nch - len(x)),
                                     constant_values=np.nan) for x in L])
    R, te, ne = pad(Rl), pad(tel), pad(nel)
    return dict(g=g, shot=shot[g], fidx=ds["fidx"][g], t_ms=ds["t_ms"][g],
                I=cur[:, ci] * 1e3,
                Ip=np.maximum(cur[:, MAST_CH.index("magnetics_plasma_current")] * 1e3, 5e4),
                Itf=cur[:, MAST_CH.index("magnetics_tf_current")],
                R=R, te_real=te, ne_real=ne)


def smooth_track(rng, shot, lo, hi, jitter=0.25, knot=8):
    """Per-shot slowly varying random values in [lo,hi] (knots every `knot` frames)."""
    out = np.empty(len(shot))
    for s in np.unique(shot):
        ix = np.flatnonzero(shot == s)
        n = len(ix)
        base = rng.uniform(lo, hi)
        nk = n // knot + 2
        kv = np.clip(base + rng.uniform(-jitter, jitter, nk) * (hi - lo), lo, hi)
        x = np.arange(n) / knot
        out[ix] = np.interp(x, np.arange(nk), kv)
    return out


# ============================================================================ truth
SCEN = {
    # A: in-family profile, no unmodelled field.  B: in-family + dv.  C: out-of-family + dv.
    "A": dict(dv=False, gen=False),
    "B": dict(dv=True, gen=False),
    "C": dict(dv=True, gen=True),
}


def truth_params(fr, scen, seed=1):
    rng = np.random.default_rng(seed)
    sh = fr["shot"]
    b0 = smooth_track(rng, sh, 0.0, 0.7)
    a1 = smooth_track(rng, sh, 1.0, 3.0)
    n1 = smooth_track(rng, sh, 0.5, 3.0)
    if SCEN[scen]["gen"]:
        a2 = smooth_track(rng, sh, 1.0, 3.0)
        n2 = smooth_track(rng, sh, 0.5, 3.0)
    else:
        a2, n2 = a1.copy(), n1.copy()
    dv = smooth_track(rng, sh, -DV_MAX, DV_MAX, jitter=0.1) if SCEN[scen]["dv"] else \
        np.zeros(len(sh))
    return np.stack([b0, a1, n1, a2, n2], 1), dv


def real_te_peak(fr):
    """Radius of the (median-filtered) real Thomson Te maximum per frame (NaN if sparse)."""
    from scipy.ndimage import median_filter
    pk = np.full(len(fr["Ip"]), np.nan)
    for i in range(len(pk)):
        r, t = fr["R"][i], fr["te_real"][i]
        ok = np.isfinite(t) & np.isfinite(r) & (r > 0.3) & (r < 1.4)
        if ok.sum() < 30:
            continue
        o = np.argsort(r[ok])
        tt = median_filter(t[ok][o], 9, mode="nearest")
        if tt.max() > 100:
            pk[i] = r[ok][o][np.argmax(tt)]
    return pk


def solve_truth(S, fr, prof, dv, n_sec=3):
    """Force-free (c->0) truth equilibria by repeated secant on the target height Zt."""
    import torch
    B = len(dv)
    Ix = np.concatenate([fr["I"], dv[:, None]], 1)
    Ip = fr["Ip"]
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
    psi_coil = (torch.as_tensor(np.concatenate([fr["I"], np.zeros((B, 1))], 1),
                                device=S.dev, dtype=S.dt) @ S.Mext.T).reshape(B, 65, 65)
    return dict(psi=t(best["psi"]).astype(np.float32), psi_ax=t(best["psi_ax"]),
                psi_b=t(best["psi_b"]), Rax=t(best["Rax"]), Zax=t(best["Zax"]),
                c=t(best["c"]), dI=t(best["dI"]), Zt=z1,
                xfound=(t(best["xu"][0]) | t(best["xl"][0])),
                psi_coil=t(psi_coil).astype(np.float32))


def synth_thomson(fr, truth, gR, gZ, seed=2):
    """Te/ne on the real channel radii (Z=0) as flux functions of truth psiN + noise."""
    from fec.kinetic import chord_psiN
    rng = np.random.default_rng(seed)
    sh = fr["shot"]
    B, nch = fr["R"].shape
    te0 = smooth_track(rng, sh, 300.0, 1500.0)
    ta = smooth_track(rng, sh, 1.0, 4.0)
    tb = smooth_track(rng, sh, 0.8, 2.5)
    tsep = smooth_track(rng, sh, 5.0, 40.0)
    ped = smooth_track(rng, sh, 0.0, 1.0) > 0.6          # ~30% frames H-mode-like pedestal
    ne0 = smooth_track(rng, sh, 1.5e19, 5e19)
    nc = smooth_track(rng, sh, 2.0, 8.0)
    nd = smooth_track(rng, sh, 0.3, 1.2)
    sig = rng.uniform(0.05, 0.10, B)
    te = np.full((B, nch), np.nan); ne = np.full((B, nch), np.nan)
    pn_all = np.full((B, nch), np.nan)
    for i in range(B):
        R = fr["R"][i]
        ok = np.isfinite(R)
        pn = np.full(nch, np.nan)
        pn[ok] = chord_psiN(truth["psi"][i].astype(np.float64), truth["psi_ax"][i],
                            truth["psi_b"][i], gR, gZ, R[ok], np.zeros(ok.sum()))
        pn_all[i] = pn
        x = np.clip(pn, 0, 1)
        core = (1 - x ** ta[i]) ** tb[i]
        if ped[i]:
            pedf = 0.5 * (1 - np.tanh((pn - 0.95) / 0.02))
            core = 0.6 * core + 0.4 * pedf * np.clip(1 - x ** 8, 0, 1) ** 0.3
        inside = pn < 1
        sol = tsep[i] * np.exp(-np.clip(pn - 1, 0, None) / 0.04)
        t = np.where(inside, tsep[i] + (te0[i] - tsep[i]) * core, sol)
        n = np.where(inside, 0.1 * ne0[i] + 0.9 * ne0[i] * (1 - x ** nc[i]) ** nd[i],
                     0.1 * ne0[i] * np.exp(-np.clip(pn - 1, 0, None) / 0.06))
        t = t * (1 + sig[i] * rng.standard_normal(nch)) + 3.0 * rng.standard_normal(nch)
        n = n * (1 + sig[i] * rng.standard_normal(nch))
        out = rng.random(nch) < 0.02
        t[out] = rng.uniform(0, 1.2 * te0[i], out.sum())
        out = rng.random(nch) < 0.02
        n[out] = rng.uniform(0, 1.2 * ne0[i], out.sum())
        miss = ~np.isfinite(fr["te_real"][i]) | (rng.random(nch) < 0.03) | ~ok
        t[miss] = np.nan
        miss_n = ~np.isfinite(fr["ne_real"][i]) | (rng.random(nch) < 0.03) | ~ok
        n[miss_n] = np.nan
        te[i] = np.where(miss, np.nan, np.maximum(t, 1.0))
        ne[i] = np.where(miss_n, np.nan, np.maximum(n, 1e17))
    return dict(te=te, ne=ne, pn_true=pn_all, te0=te0, ped=ped)


def truth_path(scen):
    return OUT / f"truth_{scen}.npz"


def build_truth(scen, n_shots=40, win=25, K=8):
    """K random smooth parameter tracks per shot; keep per shot the track whose truth axis
    follows the REAL Thomson Te-peak radius best (realistic prior; inputs only)."""
    fr = load_frames(n_shots, win)
    sol = sol_fraction()
    S, spec, M = make_solver(sol=sol)
    pk = real_te_peak(fr)
    t0 = time.time()
    runs = []
    for k in range(K):
        prof, dv = truth_params(fr, scen, seed=100 * (ord(scen) - 64) + k)
        tr = solve_truth(S, fr, prof, dv)
        runs.append((prof, dv, tr))
        print(f"truth {scen} track {k}: {time.time() - t0:.0f}s dI p99 "
              f"{np.percentile(tr['dI'], 99):.1e} |c| p99 {np.percentile(np.abs(tr['c']), 99):.1e}"
              f" Rax med {np.median(tr['Rax']):.3f}", flush=True)
    sh = fr["shot"]
    sel = np.zeros(len(sh), int)
    for s_ in np.unique(sh):
        ix = np.flatnonzero(sh == s_)
        cost = []
        for k, (_, _, tr) in enumerate(runs):
            ok = truth_valid(tr)[ix]
            d = np.abs(tr["Rax"][ix] - pk[ix])
            d = np.where(np.isfinite(d), d, 0.05)
            cost.append(np.mean(np.where(ok, d, 0.5)))
        sel[ix] = int(np.argmin(cost))
    pick_ = lambda a: np.stack([a[sel[i]][i] for i in range(len(sel))])
    prof = pick_([r[0] for r in runs]); dv = pick_([r[1] for r in runs])
    tr = {key: pick_([r[2][key] for r in runs]) for key in runs[0][2]}
    tr["track"] = sel
    tr["te_peak_real"] = pk
    ts = synth_thomson(fr, tr, spec.gR, spec.gZ)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(truth_path(scen), prof=prof, dv=dv, sol_fraction=sol,
             **{f"fr_{k}": v for k, v in fr.items()},
             **{f"tr_{k}": v for k, v in tr.items()}, **{f"ts_{k}": v for k, v in ts.items()})
    d = np.abs(tr["Rax"] - pk)
    print(f"truth {scen} saved: valid {truth_valid(tr).mean():.3f}  |Rax-TePeak| med "
          f"{np.nanmedian(d):.3f} p90 {np.nanpercentile(d, 90):.3f}  Rax pct "
          f"{np.percentile(tr['Rax'], [5, 50, 95]).round(3).tolist()}", flush=True)


def build_truth_from(scen, base="B"):
    """Scenario variants on the SAME frames and profile tracks as `base` (one truth solve):
    A = base profiles with dv = 0; C = base beta0/(a1,n1)/dv with an independent smooth FF'
    shape (a2, n2) (out of the selection family)."""
    fr, trb, tsb, prof, dv = load_truth(base)
    sol = trb["sol_fraction"]
    S, spec, M = make_solver(sol=sol)
    prof = prof.copy(); dv = dv.copy()
    rng = np.random.default_rng(1000 + ord(scen))
    if scen == "A":
        dv[:] = 0.0
    elif scen == "C":
        prof[:, 3] = smooth_track(rng, fr["shot"], 1.0, 3.0)
        prof[:, 4] = smooth_track(rng, fr["shot"], 0.5, 3.0)
    else:
        raise ValueError(scen)
    t0 = time.time()
    tr = solve_truth(S, fr, prof, dv)
    tr["track"] = trb["track"]
    tr["te_peak_real"] = trb["te_peak_real"]
    ts = synth_thomson(fr, tr, spec.gR, spec.gZ)
    np.savez(truth_path(scen), prof=prof, dv=dv, sol_fraction=sol,
             **{f"fr_{k}": v for k, v in fr.items()},
             **{f"tr_{k}": v for k, v in tr.items()}, **{f"ts_{k}": v for k, v in ts.items()})
    print(f"truth {scen} from {base}: {time.time() - t0:.0f}s valid {truth_valid(tr).mean():.3f}"
          f" Rax pct {np.percentile(tr['Rax'], [5, 50, 95]).round(3).tolist()}", flush=True)
    truth_table(scen)


def load_truth(scen):
    z = np.load(truth_path(scen))
    fr = {k[3:]: z[k] for k in z.files if k.startswith("fr_")}
    tr = {k[3:]: z[k] for k in z.files if k.startswith("tr_")}
    ts = {k[3:]: z[k] for k in z.files if k.startswith("ts_")}
    tr["sol_fraction"] = float(z["sol_fraction"]) if "sol_fraction" in z.files else 1.0
    return fr, tr, ts, z["prof"], z["dv"]


def realistic(scen, tr=None):
    """Valid truths with MAST-like midplane boundary inside the Thomson chord coverage:
    0.20 m <= R_in (centre column) and R_out <= 1.55 m."""
    if tr is None:
        tr = load_truth(scen)[1]
    Q, _ = truth_table(scen)
    Rin, Rout = Q[:, QCOLS.index("Rin")], Q[:, QCOLS.index("Rout")]
    with np.errstate(invalid="ignore"):
        return truth_valid(tr) & np.isfinite(Rout) & (Rout <= 1.55) & (Rin >= 0.20)


def truth_valid(tr):
    """Usable truth: converged, diverted, force-free, axis not clamped at the axis-box edge."""
    return (tr["dI"] < 1e-3) & tr["xfound"] & (np.abs(tr["c"]) < 1e-5) & \
        np.isfinite(tr["psi_b"]) & (tr["psi_ax"] > tr["psi_b"]) & (tr["Rax"] < 1.28) & \
        (tr["Rax"] > 0.47)


# ============================================================================ metrics
def _scal_one(args):
    """Per-frame derived quantities for one psi map (same code for truth and prediction)."""
    from fec.eqscalars import q_at
    from fec.paths import SCORING
    if str(SCORING) not in sys.path:
        sys.path.insert(0, str(SCORING))
    from derive import derive_frame  # noqa
    from lcfs import extract_lcfs  # noqa
    psi, pa, pb, ra, za, F, gR, gZ, box = args
    out = dict(q95=np.nan, Rin=np.nan, Rout=np.nan, C=None)
    try:
        out["q95"] = q_at(psi, gR, gZ, pa, pb, ra, za, F, 0.95)
    except Exception:
        pass
    # midplane LCFS radii from psiN on the Z=0 row (walk out from the row minimum)
    iz0 = int(np.argmin(np.abs(gZ)))
    pn = (pa - psi[iz0]) / (pa - pb)
    Rf = np.linspace(gR[0], gR[-1], 2001)
    pf = np.interp(Rf, gR, pn)
    j0 = int(np.argmin(np.where((Rf > 0.3) & (Rf < 1.4), pf, np.inf)))
    if pf[j0] < 1:
        jo = np.flatnonzero(pf[j0:] >= 1)
        ji = np.flatnonzero(pf[:j0 + 1][::-1] >= 1)
        out["Rout"] = Rf[j0 + jo[0]] if len(jo) else np.nan
        out["Rin"] = Rf[j0 - ji[0]] if len(ji) else np.nan
    try:
        C = extract_lcfs(psi, gR, gZ, "MAST", box, box.astype(float), axis_sign=1.0)
        d = derive_frame(psi, gR, gZ, "MAST", box, box.astype(float), contour=C,
                         axis_sign=1.0)
        out.update(d)
        out["C"] = C
    except Exception:
        pass
    return out


def r2(y, p):
    ok = np.isfinite(y)
    y, p = y[ok], p[ok]
    p = np.where(np.isfinite(p), p, np.mean(y))
    tot = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - p) ** 2) / tot if tot > 0 else np.nan


# ============================================================================ candidate bank
BANK_B0 = [0.0, 0.05, 0.15, 0.3, 0.5, 0.7]
BANK_AM = [1.0, 1.5, 2.0, 3.0]
BANK_AN = [0.5, 1.0, 1.5, 2.0, 3.0]
LEVEL_SETS = {"v1": np.linspace(0.05, 0.85, 17), "e95": np.linspace(0.05, 0.95, 19),
              "c60": np.linspace(0.05, 0.60, 12), "e99": np.linspace(0.05, 0.99, 48)}


def bank_grid():
    return [(b, a, n) for b in BANK_B0 for a in BANK_AM for n in BANK_AN]


DV_PROFILES = [(b, a, n) for b in (0.0, 0.15, 0.3, 0.5)
               for (a, n) in ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (3.0, 1.0), (1.5, 2.0))]
DV_VALUES = (-0.010, -0.005, 0.0, 0.005, 0.010)


def bank_path(scen, tag):
    return OUT / f"bank_{scen}_{tag}"


def build_bank(scen, tag="p120", combos=None, dvs=(0.0,), n_iter=100, cpu_v1=True):
    """Solve every candidate (profile x dv) for every twin frame, warm-started like v1."""
    import torch
    from fec.kinetic import chord_psiN, symmetry_mismatch
    from fec.mast_select_v2 import force_free_z, prep_profiles, symmetry_mm, to_dev
    from fec.mast_v1 import COMBOS, DEFAULT
    fr, tr, ts, prof, dv = load_truth(scen)
    S, spec, M = make_solver(sol=tr["sol_fraction"])   # truth and candidates share the matrix
    combos = bank_grid() if combos is None else combos
    cand = [(c, d) for c in combos for d in dvs]
    K, B = len(cand), len(fr["Ip"])
    I = np.concatenate([fr["I"], np.zeros((B, 1))], 1)
    Ip = fr["Ip"]
    t0 = time.time()
    o0, Zs = force_free_z(S, I, Ip, np.tile(COMBOS[DEFAULT], (B, 1)))
    dev = S.dev
    pte = to_dev(prep_profiles(fr["R"], ts["te"]), dev)
    pne = to_dev(prep_profiles(fr["R"], ts["ne"]), dev)
    gR = torch.as_tensor(spec.gR, device=dev, dtype=torch.float32)
    iz0 = int(np.argmin(np.abs(spec.gZ)))
    lev = {k: torch.as_tensor(v, device=dev, dtype=torch.float32) for k, v in LEVEL_SETS.items()}
    ptrue = torch.as_tensor(tr["psi"], device=dev)
    d = bank_path(scen, tag)
    d.mkdir(parents=True, exist_ok=True)
    psi_mm = np.lib.format.open_memmap(d / "psi.npy", "w+", np.float32, (K, B, 65, 65))
    keys = ["psi_ax", "psi_b", "Rax", "Zax", "dI", "c", "sse", "mm_cpu_v1"] + \
        [f"mmte_{k}" for k in LEVEL_SETS] + [f"mmne_{k}" for k in LEVEL_SETS] + \
        ["mmte2_v1", "Rin1", "Rout1", "te_in1", "te_out1"]
    feat = {k: np.full((K, B), np.nan) for k in keys}
    v1set = {tuple(c) for c in COMBOS}
    one = torch.ones(1, device=dev)
    for k, (combo, dvk) in enumerate(cand):
        Ix = I.copy(); Ix[:, -1] = dvk
        o = S.solve(Ix, Ip, np.tile(combo, (B, 1)), Zs, n_iter=n_iter, I0=o0["I"], c0=o0["c"])
        psi = o["psi"]
        for nm in ("psi_ax", "psi_b", "Rax", "Zax", "dI", "c"):
            feat[nm][k] = o[nm].cpu().numpy()
        feat["sse"][k] = ((psi - ptrue) ** 2).sum((1, 2)).cpu().numpy()
        for ln, lv in lev.items():
            feat[f"mmte_{ln}"][k] = symmetry_mm(psi, o["psi_ax"], o["psi_b"], gR, iz0, pte,
                                                lv)[0].cpu().numpy()
            feat[f"mmne_{ln}"][k] = symmetry_mm(psi, o["psi_ax"], o["psi_b"], gR, iz0, pne,
                                                lv)[0].cpu().numpy()
        feat["mmte2_v1"][k] = symmetry_mm(psi, o["psi_ax"], o["psi_b"], gR, iz0, pte,
                                          lev["v1"], power=2)[0].cpu().numpy()
        # LCFS midplane radii and measured Te there (edge consistency features)
        _, Rin, Rout, okr = symmetry_mm(psi, o["psi_ax"], o["psi_b"], gR, iz0, pte,
                                        one * 1.0, min_levels=0)
        from fec.mast_select_v2 import sample_profile
        vi, oki = sample_profile(pte, Rin); vo, oko = sample_profile(pte, Rout)
        feat["Rin1"][k] = torch.where(okr[:, 0], Rin[:, 0], torch.nan).cpu().numpy()
        feat["Rout1"][k] = torch.where(okr[:, 0], Rout[:, 0], torch.nan).cpu().numpy()
        feat["te_in1"][k] = torch.where(oki[:, 0], vi[:, 0] / pte["scale"], torch.nan).cpu().numpy()
        feat["te_out1"][k] = torch.where(oko[:, 0], vo[:, 0] / pte["scale"], torch.nan).cpu().numpy()
        pn_ = psi.cpu().numpy()
        psi_mm[k] = pn_
        if cpu_v1 and tuple(combo) in v1set and dvk == 0.0:
            pa, pb, ra = feat["psi_ax"][k], feat["psi_b"][k], feat["Rax"][k]
            for i in range(B):
                R = fr["R"][i]; ok = np.isfinite(R)
                pn = chord_psiN(pn_[i], pa[i], pb[i], spec.gR, spec.gZ, R[ok], np.zeros(ok.sum()))
                feat["mm_cpu_v1"][k, i] = symmetry_mismatch(pn, R[ok], ts["te"][i][ok], ra[i])[0]
        if k % 10 == 0 or k == K - 1:
            print(f"bank {scen}/{tag} {k + 1}/{K} {time.time() - t0:.0f}s", flush=True)
    psi_mm.flush()
    np.savez(d / "feat.npz", cand_prof=np.array([c for c, _ in cand], float),
             cand_dv=np.array([dd for _, dd in cand], float), Zs=Zs,
             sol_fraction=tr["sol_fraction"], **feat)
    print(f"bank done {time.time() - t0:.0f}s", flush=True)


# ============================================================================ quantity tables
QCOLS = ["q95", "Rin", "Rout", "R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume",
         "li", "H", "Rgeo"]


def _qrow(args):
    """QCOLS vector for one psi map; H = Hausdorff to the reference contour (if given)."""
    from fec.paths import SCORING
    if str(SCORING) not in sys.path:
        sys.path.insert(0, str(SCORING))
    from contour import symmetric_hausdorff  # noqa
    from lcfs import major_radius  # noqa
    psi, pa, pb, ra, za, F, gR, gZ, box, Cref = args
    o = _scal_one((psi, pa, pb, ra, za, F, gR, gZ, box))
    row = [o.get(k, np.nan) for k in QCOLS[:10]]
    C = o["C"]
    H = np.nan
    if Cref is not None:
        H = 1e9 if C is None else symmetric_hausdorff(C, Cref)
    Rg = major_radius(C) if C is not None else np.nan
    return np.array(row + [H, Rg], float), C


def truth_table(scen, nproc=6):
    """Truth quantities + contours (cached)."""
    f = OUT / f"truthq_{scen}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return z["Q"], list(z["C"])
    fr, tr, ts, prof, dv = load_truth(scen)
    from fec.machines import mast_setup
    spec, _ = mast_setup()
    box = _box()
    F = MU0 * N_TF * np.abs(fr["Itf"]) * 1e3 / (2 * np.pi)
    jobs = [(tr["psi"][i].astype(np.float64), tr["psi_ax"][i], tr["psi_b"][i], tr["Rax"][i],
             tr["Zax"][i], F[i], spec.gR, spec.gZ, box, None) for i in range(len(F))]
    with Pool(nproc) as pool:
        res = pool.map(_qrow, jobs, chunksize=8)
    Q = np.stack([r[0] for r in res])
    C = [r[1] for r in res]
    Ca = np.empty(len(C), dtype=object)
    for i, c in enumerate(C):
        Ca[i] = c
    np.savez(f, Q=Q, C=Ca)
    return Q, C


def _box():
    from fec.machines import mast_setup
    spec, _ = mast_setup()
    RR, ZZ = np.meshgrid(spec.gR, spec.gZ)
    Rmin, Rmax, Zmin, Zmax = spec.box
    return (RR >= Rmin) & (RR <= Rmax) & (ZZ >= Zmin) & (ZZ <= Zmax)


class QTable:
    """Lazily filled (K,B,len(QCOLS)) quantity table of bank candidates (cached on disk)."""

    def __init__(self, scen, tag):
        self.scen, self.tag = scen, tag
        d = bank_path(scen, tag)
        self.fz = dict(np.load(d / "feat.npz"))
        self.psi = np.load(d / "psi.npy", mmap_mode="r")
        K, B = self.psi.shape[:2]
        self.fq = d / "qtab.npy"
        self.fd = d / "qdone.npy"
        self.Q = np.load(self.fq) if self.fq.exists() else np.full((K, B, len(QCOLS)), np.nan)
        self.done = np.load(self.fd) if self.fd.exists() else np.zeros((K, B), bool)
        self.Qtr, self.Ctr = truth_table(scen)
        fr, tr, ts, prof, dv = load_truth(scen)
        self.F = MU0 * N_TF * np.abs(fr["Itf"]) * 1e3 / (2 * np.pi)
        from fec.machines import mast_setup
        self.spec, _ = mast_setup()
        self.box = _box()

    def ensure(self, choice, valid, nproc=6):
        idx = [(int(choice[i]), int(i)) for i in np.flatnonzero(valid)
               if not self.done[int(choice[i]), i]]
        if not idx:
            return 0
        fz, sp = self.fz, self.spec
        jobs = [(np.asarray(self.psi[k, i], np.float64), fz["psi_ax"][k, i], fz["psi_b"][k, i],
                 fz["Rax"][k, i], fz["Zax"][k, i], self.F[i], sp.gR, sp.gZ, self.box,
                 self.Ctr[i]) for k, i in idx]
        with Pool(nproc) as pool:
            res = pool.map(_qrow, jobs, chunksize=8)
        for (k, i), r in zip(idx, res):
            self.Q[k, i] = r[0]
            self.done[k, i] = True
        np.save(self.fq, self.Q)
        np.save(self.fd, self.done)
        return len(idx)


def metrics_from_choice(choice, Qt, feat, Qtr, tr, shot, valid):
    """Twin metrics for per-frame candidate choice (B,) using precomputed tables."""
    v = np.flatnonzero(valid)
    ch = choice[v]
    return metrics_from_rows(Qt[ch, v], feat["sse"][ch, v], Qtr, tr, shot, valid)


def metrics_from_rows(q, sse_rows, Qtr, tr, shot, valid):
    """q (n_valid, QCOLS) and per-frame SSE (n_valid,) of a prediction on the valid frames."""
    v = np.flatnonzero(valid)
    pt = tr["psi"][v].astype(np.float64)
    sst = np.sum((pt - pt.mean()) ** 2)
    ptp = pt - tr["psi_coil"][v]
    sstp = np.sum((ptp - ptp.mean()) ** 2)
    sse = np.sum(sse_rows)
    out = dict(R2psi=1 - sse / sst, R2psi_pl=1 - sse / sstp)
    t = Qtr[v]
    c = {k: i for i, k in enumerate(QCOLS)}
    for k in ("R_axis", "Z_axis", "Rin", "Rout"):
        e = q[:, c[k]] - t[:, c[k]]
        out[f"{k}_mae_cm"] = 100 * np.nanmean(np.abs(e))
    out["R_axis_p90_cm"] = 100 * np.nanpercentile(np.abs(q[:, c["R_axis"]] - t[:, c["R_axis"]]), 90)
    out["q95_mape"] = 100 * np.nanmean(np.abs(q[:, c["q95"]] / t[:, c["q95"]] - 1))
    out["R2q95"] = r2(t[:, c["q95"]], q[:, c["q95"]])
    cons = [r2(t[:, c[k]], q[:, c[k]]) for k in
            ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li")]
    out["R2_li"] = cons[-1]
    out["cons"] = float(np.mean([max(0.0, x) for x in cons if np.isfinite(x)]))
    D = np.minimum(1.0, q[:, c["H"]] / t[:, c["Rgeo"]])
    dsh = {}
    for s_, dd in zip(shot[v], D):
        if np.isfinite(dd):
            dsh.setdefault(s_, []).append(dd)
    out["D_LCFS"] = float(np.mean([np.mean(x) for x in dsh.values()]))
    out["S_proxy"] = 0.55 * max(0, out["R2psi"]) + 0.15 * max(0, out["R2q95"]) + \
        0.10 * (1 - min(1, out["D_LCFS"])) + 0.20 * out["cons"]
    return out


# ============================================================================ strategies
def argmin_masked(score, allowed, default):
    """Per-frame argmin over allowed candidates; default where nothing finite."""
    sc = np.where(np.isfinite(score), score, np.inf)
    allowed = np.asarray(allowed, bool)
    if allowed.ndim == 1:
        allowed = allowed[:, None]
    sc = np.where(allowed, sc, np.inf)
    best = np.argmin(sc, 0)
    best[~np.isfinite(sc).any(0)] = default
    return best


def smooth_time(x, shot, h):
    """Centred nan-mean over +-h neighbouring frames of the same shot (frames time-ordered)."""
    if h <= 0:
        return x
    out = np.full_like(x, np.nan)
    for s_ in np.unique(shot):
        ix = np.flatnonzero(shot == s_)
        v = x[:, ix]
        n = v.shape[1]
        ok = np.isfinite(v)
        vz = np.where(ok, v, 0.0)
        cs = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(vz, 1)], 1)
        cn = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(ok, 1)], 1)
        lo = np.clip(np.arange(n) - h, 0, n); hi = np.clip(np.arange(n) + h + 1, 0, n)
        num = cs[:, hi] - cs[:, lo]; den = cn[:, hi] - cn[:, lo]
        sm = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        out[:, ix] = np.where(ok, sm, np.nan)     # a candidate must be valid at this frame
    return out


class Bank:
    def __init__(self, scen, tag, subset="all"):
        self.scen, self.tag = scen, tag
        self.q = QTable(scen, tag)
        self.f = self.q.fz
        self.fr, self.tr, self.ts, self.prof, self.dv = load_truth(scen)
        self.valid = truth_valid(self.tr) if subset == "all" else realistic(scen, self.tr)
        self.P = self.f["cand_prof"]
        self.cdv = self.f["cand_dv"]
        self.K, self.B = self.f["sse"].shape
        self.conv = self.f["dI"] < 1e-2

    def idx(self, combo, dv=0.0):
        d = np.abs(self.P - np.asarray(combo)[None]).sum(1) + np.abs(self.cdv - dv)
        k = int(np.argmin(d))
        assert d[k] < 1e-9, combo
        return k

    def subset(self, combos, dv=0.0):
        m = np.zeros(self.K, bool)
        for c in combos:
            m[self.idx(c, dv)] = True
        return m

    def mm(self, key):
        return np.where(self.conv, self.f[key], np.nan)

    def evaluate(self, choice):
        self.q.ensure(choice, self.valid)
        return metrics_from_choice(choice, self.q.Q, self.f, self.q.Qtr, self.tr,
                                   self.fr["shot"], self.valid)

    def evaluate_avg(self, W, nproc=6):
        """Weighted average of candidate maps per frame (W (K,B), columns sum to 1)."""
        v = np.flatnonzero(self.valid)
        sp, f = self.q.spec, self.f
        psi = np.zeros((len(v), 65, 65)); aux = np.zeros((len(v), 4))
        for j, i in enumerate(v):
            ks = np.flatnonzero(W[:, i] > 0)
            w = W[ks, i] / W[ks, i].sum()
            psi[j] = np.tensordot(w, np.asarray(self.q.psi[ks, i], np.float64), 1)
            aux[j] = [np.dot(w, f[n][ks, i]) for n in ("psi_ax", "psi_b", "Rax", "Zax")]
        sse = ((psi - self.tr["psi"][v]) ** 2).sum((1, 2))
        jobs = [(psi[j], *aux[j], self.q.F[i], sp.gR, sp.gZ, self.q.box, self.q.Ctr[i])
                for j, i in enumerate(v)]
        with Pool(nproc) as pool:
            res = pool.map(_qrow, jobs, chunksize=8)
        q = np.stack([r[0] for r in res])
        return metrics_from_rows(q, sse, self.q.Qtr, self.tr, self.fr["shot"], self.valid)


def fmt_table(rows, cols):
    head = "| method | " + " | ".join(cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [head, sep]
    for name, m in rows:
        cells = []
        for c in cols:
            v = m.get(c, np.nan)
            cells.append(f"{v:.5f}" if c.startswith(("R2", "S_", "cons", "D_")) else f"{v:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


COLS = ["S_proxy", "R2psi", "R2psi_pl", "R_axis_mae_cm", "Z_axis_mae_cm", "Rin_mae_cm",
        "Rout_mae_cm", "q95_mape", "R2q95", "D_LCFS", "cons", "R2_li"]


def bank_strategies(bk):
    """Named per-frame choices (B,) from bank features only (no truth except oracles)."""
    from fec.mast_v1 import COMBOS, DEFAULT
    allc = np.ones(bk.K, bool)
    v1m = bk.subset(COMBOS)
    d = bk.idx(COMBOS[DEFAULT])
    sh = bk.fr["shot"]
    te, ne = bk.mm("mmte_v1"), bk.mm("mmne_v1")
    S = {}
    S["default (0.05,2,2)"] = np.full(bk.B, d)
    S["v1: 6 combos, Te (CPU, production)"] = argmin_masked(bk.mm("mm_cpu_v1"), v1m, d)
    S["v1 combos, Te (GPU metric)"] = argmin_masked(te, v1m, d)
    S["v1 combos, Te, time-smooth h=2"] = argmin_masked(smooth_time(te, sh, 2), v1m, d)
    S["v1 combos, Te+ne"] = argmin_masked(te + ne, v1m, d)
    S["120 grid, Te"] = argmin_masked(te, allc, d)
    for lv in ("e95", "c60", "e99"):
        S[f"120 grid, Te levels {lv}"] = argmin_masked(bk.mm(f"mmte_{lv}"), allc, d)
    S["120 grid, Te^2"] = argmin_masked(bk.mm("mmte2_v1"), allc, d)
    S["120 grid, ne only"] = argmin_masked(ne, allc, d)
    for w in (0.5, 1.0):
        S[f"120 grid, Te+{w}ne"] = argmin_masked(te + w * ne, allc, d)
    for h in (1, 2, 3):
        S[f"120 grid, Te, time-smooth h={h}"] = argmin_masked(smooth_time(te, sh, h), allc, d)
    S["120 grid, Te+ne, time-smooth h=2"] = argmin_masked(smooth_time(te + ne, sh, 2), allc, d)
    edge = bk.f["te_in1"] + bk.f["te_out1"]
    for lam in (0.1, 0.3):
        S[f"120 grid, Te+{lam}*edgeTe"] = argmin_masked(te + lam * np.nan_to_num(edge, nan=1.0),
                                                       allc, d)
    S["ORACLE v1 combos (min psi SSE)"] = argmin_masked(bk.f["sse"], v1m, d)
    S["ORACLE 120 grid (min psi SSE)"] = argmin_masked(bk.f["sse"], allc, d)
    return S


def grid_coords(bk):
    """Index coordinates of every bank candidate on the (b0, am, an) grid (dv ignored)."""
    axes = [np.unique(bk.P[:, j]) for j in range(3)]
    co = np.stack([np.searchsorted(axes[j], bk.P[:, j]) for j in range(3)], 1)
    lut = {tuple(c): k for k, c in enumerate(co)}
    return axes, co, lut


def emulate_descent(bk, score, start_set, rounds=2, default=None, pattern="axis"):
    """Batched grid descent emulated on the bank.

    Stage 1: argmin of score over start_set (len(start_set) batched solves).
    Each round: evaluate the +-1 grid neighbours of every frame's current best along each axis
    (pattern 'axis': 6 batched solves/round; 'full': all 26 neighbours) and move to the best.
    Returns (choice (B,), batched-solve count).
    """
    axes, co, lut = grid_coords(bk)
    d = default if default is not None else bk.idx((0.05, 2, 2))
    m = np.zeros(bk.K, bool); m[list(start_set)] = True
    cur = argmin_masked(score, m, d)
    sc = np.where(np.isfinite(score), score, np.inf)
    n = len(start_set)
    if pattern == "axis":
        steps = [s for j in range(3) for s in ((j, -1), (j, 1))]
        offs = []
        for j, sg in steps:
            o = [0, 0, 0]; o[j] = sg; offs.append(tuple(o))
    else:
        offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                if (a, b, c) != (0, 0, 0)]
    B = bk.B
    for r in range(rounds):
        best = cur.copy()
        bval = sc[cur, np.arange(B)]
        for o in offs:
            nb = np.array([lut.get(tuple(co[cur[i]] + np.array(o)), -1) for i in range(B)])
            ok = nb >= 0
            val = np.where(ok, sc[np.where(ok, nb, 0), np.arange(B)], np.inf)
            imp = val < bval
            best[imp] = nb[imp]; bval[imp] = val[imp]
        n += len(offs)
        cur = best
    return cur, n


def analyze_bank(scen, tag, names=None, extra=None, subset="all"):
    bk = Bank(scen, tag, subset)
    S = bank_strategies(bk)
    if extra:
        S.update(extra(bk))
    rows = []
    for name, ch in S.items():
        if names and name not in names:
            continue
        t0 = time.time()
        m = bk.evaluate(ch)
        m["frac_default"] = float(np.mean(ch[bk.valid] == S["default (0.05,2,2)"][0]))
        rows.append((name, m))
        print(f"{name:45s} S {m['S_proxy']:.5f} R2psi {m['R2psi']:.6f} Rax {m['R_axis_mae_cm']:.2f}"
              f" q95 {m['q95_mape']:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
    res = {n: {k: float(v) for k, v in m.items()} for n, m in rows}
    (OUT / f"results_{scen}_{tag}_{subset}.json").write_text(json.dumps(res, indent=1))
    print(fmt_table(rows, COLS))
    return bk, S, rows


# ============================================================================ real procedures
def evaluate_pred(scen, pred, nproc=6, subset="all"):
    """Metrics of an arbitrary prediction dict (psi, psi_ax, psi_b, Rax, Zax for all frames)."""
    fr, tr, ts, prof, dv = load_truth(scen)
    valid = truth_valid(tr) if subset == "all" else realistic(scen, tr)
    Qtr, Ctr = truth_table(scen)
    from fec.machines import mast_setup
    spec, _ = mast_setup()
    box = _box()
    F = MU0 * N_TF * np.abs(fr["Itf"]) * 1e3 / (2 * np.pi)
    v = np.flatnonzero(valid)
    P = np.asarray(pred["psi"], np.float64)
    sse = ((P[v] - tr["psi"][v]) ** 2).sum((1, 2))
    jobs = [(P[i], float(pred["psi_ax"][i]), float(pred["psi_b"][i]), float(pred["Rax"][i]),
             float(pred["Zax"][i]), F[i], spec.gR, spec.gZ, box, Ctr[i]) for i in v]
    with Pool(nproc) as pool:
        res = pool.map(_qrow, jobs, chunksize=8)
    q = np.stack([r[0] for r in res])
    return metrics_from_rows(q, sse, Qtr, tr, fr["shot"], valid)


def run_real(scen, method, **kw):
    """Execute a selector exactly as production would (GPU solves) on the twin frames."""
    import torch
    from fec.mast_select_v2 import select_batch, refine_batch
    from fec.mast_v1 import COMBOS, DEFAULT
    fr, tr, ts, prof, dv = load_truth(scen)
    S, spec, M = make_solver(sol=tr["sol_fraction"])
    B = len(fr["Ip"])
    torch.cuda.synchronize(); t0 = time.time()
    cands = kw.get("cands", COMBOS)
    default = kw.get("default", cands.index(COMBOS[DEFAULT]) if COMBOS[DEFAULT] in cands else 0)
    out = select_batch(S, fr["I"], fr["Ip"], fr["R"], ts["te"], ts["ne"], fr["shot"], cands,
                       default, w_ne=kw.get("w_ne", 0.0), h=kw.get("h", 0),
                       n_iter=kw.get("n_iter", 100), dv_col=True,
                       resecant=kw.get("resecant", False), keep_all=True)
    n_solves = 2 + len(cands) + (2 if kw.get("resecant") else 0)
    if kw.get("refine"):
        Ix = np.concatenate([fr["I"], np.zeros((B, 1))], 1)
        prof0 = np.array([cands[int(c)] for c in out["choice"]], float)
        dev = S.dev
        st0 = {k: torch.as_tensor(out[k], device=dev) for k in
               ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI", "I")}
        mm0 = out["score"][out["choice"], np.arange(B)]
        prof1, mm1, st = refine_batch(S, Ix, fr["Ip"], fr["R"], ts["te"], prof0, None, None,
                                      out["Zt"], deltas=kw["refine"],
                                      n_iter=kw.get("refine_iter", 60), ne=ts["ne"],
                                      w_ne=kw.get("w_ne", 0.0), mm0=mm0, state0=st0)
        n_solves += 3 * len(kw["refine"])
        for k in ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI"):
            out[k] = st[k].cpu().numpy()
        out["prof"] = prof1
    torch.cuda.synchronize()
    out["gpu_s"] = time.time() - t0
    out["n_solves"] = n_solves
    return out


def run_kfit2(scen, levels=None, pair_src="te", S=None, o0Zs=None, **cfg):
    """Regularised kinetic fit (fec.mast_select_v2.kfit2) on the twin frames."""
    import torch
    from fec.kefit import LEVELS, te_pairs
    from fec.mast_select_v2 import force_free_z, kfit2
    from fec.mast_v1 import COMBOS, DEFAULT
    fr, tr, ts, prof, dv = load_truth(scen)
    if S is None:
        S, spec, M = make_solver(sol=tr["sol_fraction"])
    B = len(fr["Ip"])
    Ix = np.concatenate([fr["I"], np.zeros((B, 1))], 1)
    lv = LEVELS if levels is None else np.asarray(levels)
    srcs = {"te": [ts["te"]], "ne": [ts["ne"] / 1e17], "te+ne": [ts["te"], ts["ne"] / 1e17]}
    pairs = np.concatenate([np.stack([te_pairs(fr["R"][i], src[i], levels=lv)
                                      for i in range(B)]) for src in srcs[pair_src]], 1)
    torch.cuda.synchronize(); t0 = time.time()
    if o0Zs is None:
        o0Zs = force_free_z(S, Ix, fr["Ip"], np.tile(COMBOS[DEFAULT], (B, 1)))
    o0, Zs = o0Zs
    o = kfit2(S, Ix, fr["Ip"], Zs, pairs, I0=o0["I"], c0=o0["c"], **cfg)
    torch.cuda.synchronize()
    out = {k: o[k].cpu().numpy() for k in ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI",
                                             "u", "dv")}
    out["npairs"] = np.isfinite(pairs).all(-1).sum(1)
    out["gpu_s"] = time.time() - t0
    out["Zt"] = Zs
    return out, S, o0Zs


KF_LEV9 = np.round(np.arange(0.1, 0.95, 0.1), 2)
KF_LEV17 = np.round(np.arange(0.1, 0.91, 0.05), 2)
REAL = {
    # name: (runner, kwargs)
    "kfit2": ("kfit", dict()),
    "kfit2 lm5": ("kfit", dict(lm=5.0)),
    "kfit2 lm5 9 levels": ("kfit", dict(lm=5.0, levels=KF_LEV9)),
    "kfit2 lm5 no dv": ("kfit", dict(lm=5.0, fit_dv=False)),
    "kfit2 lm5 9 levels no dv": ("kfit", dict(lm=5.0, levels=KF_LEV9, fit_dv=False)),
    "kfit2 lm5 9 levels 200it": ("kfit", dict(lm=5.0, levels=KF_LEV9, n_iter=200)),
    "kfit2 lm5 17 levels": ("kfit", dict(lm=5.0, levels=KF_LEV17)),
    "kfit2 lm5 9 levels te+ne": ("kfit", dict(lm=5.0, levels=KF_LEV9, pair_src="te+ne")),
    "kfit2 lm5 17 levels te+ne": ("kfit", dict(lm=5.0, levels=KF_LEV17, pair_src="te+ne")),
    "kfit2 lm2 9 levels": ("kfit", dict(lm=2.0, levels=KF_LEV9)),
    "kfit2 lm5 9 levels tol0.01": ("kfit", dict(lm=5.0, levels=KF_LEV9, pair_tol=0.01)),
    "kfit2 lm5 9 levels prior1": ("kfit", dict(lm=5.0, levels=KF_LEV9,
                                               w_prior=(1.0, 1.0, 1.0, 1.0, 1.0))),
    "v2 hybrid": ("v2", dict(combine="score")),
    "v2 grid12 only": ("v2", dict(use_kfit=False)),
    "v2 hybrid kfit 9 levels": ("v2", dict(combine="score", kfit_kw=dict(levels=KF_LEV9))),
    "v2 hybrid it60": ("v2", dict(combine="score", n_iter=60, kfit_kw=dict(n_iter=80))),
    "v2 hybrid grid8": ("v2", dict(combine="score",
                                   grid=[(b, a, n) for b in (0.0, 0.15, 0.3, 0.5)
                                         for (a, n) in ((1.0, 1.0), (2.0, 2.0))])),
    "v2 pairs rule": ("v2", dict()),
    "v2 final": ("v2", dict()),          # same call after the seeded-kfit2/stacked-grid speedup
    "kfit2 lm5 final": ("v2", dict(use_grid=False)),
    "v2 pairs rule it60": ("v2", dict(n_iter=60, kfit_kw=dict(n_iter=80))),
    "default only": ("select", dict(cands=[(0.05, 2, 2)])),
    "grid12 Te+ne + parabolic refine b0,an": ("select", dict(
        cands=[(b, a, n) for b in (0.0, 0.15, 0.3, 0.5) for (a, n) in ((1, 1), (2, 2), (3, 1))],
        default=1, w_ne=1.0, refine=(("b0", 0.075), ("an", 0.5)), refine_iter=60)),
    "v1 real (6 combos)": ("select", dict()),
    "v1 real + resecant": ("select", dict(resecant=True)),
}


def run_v2(scen, S=None, **kw):
    """fec.mast_select_v2.select_v2 exactly as in production, on the twin frames."""
    import torch
    from fec.mast_select_v2 import select_v2
    fr, tr, ts, prof, dv = load_truth(scen)
    if S is None:
        S, spec, M = make_solver(sol=tr["sol_fraction"])
    torch.cuda.synchronize(); t0 = time.time()
    out = select_v2(S, fr["I"], fr["Ip"], fr["R"], ts["te"], ts["ne"], fr["shot"],
                    dv_col=True, **kw)
    torch.cuda.synchronize()
    out["gpu_s"] = time.time() - t0
    return out, S


def real_experiments(scen, names):
    """Run named real procedures on the twin (one GPU process), save preds + metrics."""
    d = OUT / f"preds_{scen}"
    d.mkdir(parents=True, exist_ok=True)
    S, o0Zs = None, None
    for name in names:
        kind, kw = REAL[name]
        if kind == "kfit":
            out, S, o0Zs = run_kfit2(scen, S=S, o0Zs=o0Zs, **kw)
        elif kind == "v2":
            out, S = run_v2(scen, S=S, **kw)
        else:
            out = run_real(scen, name, **kw)
        m = evaluate_pred(scen, out)
        mr = evaluate_pred(scen, out, subset="real")
        m.update({f"real_{k}": v for k, v in mr.items()})
        m["gpu_s"] = out["gpu_s"]
        if "npairs" in out:
            fr, tr, ts, prof, dv = load_truth(scen)
            v = truth_valid(tr)
            m["frac_ge3_pairs"] = float(np.mean(out["npairs"][v] >= 3))
            m["dv_corr"] = float(np.corrcoef(out["dv"][v], dv[v])[0, 1]) \
                if np.std(dv[v]) > 0 else float("nan")
            m["dv_mae_mT"] = float(1e3 * np.mean(np.abs(out["dv"][v] - dv[v])))
            m["dI_p99"] = float(np.percentile(out["dI"], 99))
        np.savez(d / (name.replace(" ", "_").replace("(", "").replace(")", "") + ".npz"),
                 **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
        with open(OUT / f"real_{scen}.jsonl", "a") as fh:
            fh.write(json.dumps({"name": name, **{k: float(v) for k, v in m.items()}}) + "\n")
        print(f"{name:35s} S {m['S_proxy']:.5f} R2psi {m['R2psi']:.6f} R2pl {m['R2psi_pl']:.4f} "
              f"Rax {m['R_axis_mae_cm']:.2f}cm q95 {m['q95_mape']:.1f}% D {m['D_LCFS']:.4f} "
              f"cons {m['cons']:.4f} gpu {m['gpu_s']:.0f}s | realistic S {m['real_S_proxy']:.5f} "
              f"R2psi {m['real_R2psi']:.5f} Rax {m['real_R_axis_mae_cm']:.2f}", flush=True)


def rescore_preds(scen, subset="real", names=None):
    """Metrics of every saved real-procedure prediction on a truth subset."""
    d = OUT / f"preds_{scen}"
    rows = []
    for f in sorted(d.glob("*.npz")):
        if names and f.stem not in names:
            continue
        z = dict(np.load(f))
        m = evaluate_pred(scen, z, subset=subset)
        rows.append((f.stem, m))
        print(f"{f.stem:40s} S {m['S_proxy']:.5f} R2psi {m['R2psi']:.5f} Rax "
              f"{m['R_axis_mae_cm']:.2f} Zax {m['Z_axis_mae_cm']:.2f} q95 {m['q95_mape']:.2f}",
              flush=True)
    res = {n: {k: float(v) for k, v in m.items()} for n, m in rows}
    (OUT / f"preds_{scen}_{subset}.json").write_text(json.dumps(res, indent=1))
    print(fmt_table(rows, COLS))
    return rows


def realdata_check(limit=1024, out_name="realcheck_v1_v2.npz"):
    """Label-free comparison on REAL MAST inputs (first `limit` frames): v1 replica vs
    select_v2.  ne symmetry is used by neither kfit2 nor v1 (Te only), so the ne mismatch of
    the returned equilibria is an independent consistency check.  No labels involved."""
    import torch
    from fec.features import load_ds
    from fec.gs import Solver
    from fec.machines import MAST_CH, MAST_COILS, mast_setup
    from fec.mast_select_v2 import (LEVELS_V1, prep_profiles, select_batch, select_v2,
                                    symmetry_mm, to_dev)
    from fec.mast_v1 import COMBOS, DEFAULT, thomson_frame
    ds = load_ds("mast_public_test")
    info = ds["info"]
    shot = np.asarray(ds["shot"][:limit])
    cur = np.asarray(ds["cur"][:limit], dtype=np.float64)[:, :, 6]
    I = cur[:, [MAST_CH.index(c) for c in MAST_COILS]] * 1e3
    Ip = np.maximum(cur[:, MAST_CH.index("magnetics_plasma_current")] * 1e3, 5e4)
    Rl, tel, nel, cache = [], [], [], {}
    for g in range(limit):
        stem = info["order"][ds["shot"][g]]
        if stem not in cache:
            cache.clear()
            cache[stem] = dict(np.load(WORK / "shots" / "mast_public_test" / f"{stem}.npz"))
        R, te, ne = thomson_frame(cache[stem], ds["fidx"][g])
        Rl.append(R); tel.append(te); nel.append(ne)
    nch = max(len(r) for r in Rl)
    pad = lambda L: np.stack([np.pad(np.asarray(x, float), (0, nch - len(x)),
                                     constant_values=np.nan) for x in L])
    R, te, ne = pad(Rl), pad(tel), pad(nel)
    spec, M = mast_setup()
    S = Solver(spec, M)
    res = {}
    torch.cuda.synchronize(); t0 = time.time()
    res["v1"] = select_batch(S, I, Ip, R, te, ne, shot, COMBOS, DEFAULT)
    torch.cuda.synchronize(); t1 = time.time()
    res["v2"] = select_v2(S, I, Ip, R, te, ne, shot)
    torch.cuda.synchronize(); t2 = time.time()
    gR = torch.as_tensor(spec.gR, device=S.dev, dtype=torch.float32)
    iz0 = int(np.argmin(np.abs(spec.gZ)))
    lev = torch.as_tensor(LEVELS_V1, device=S.dev, dtype=torch.float32)
    pte = to_dev(prep_profiles(R, te), S.dev); pne = to_dev(prep_profiles(R, ne), S.dev)
    save = {"t_v1": t1 - t0, "t_v2": t2 - t1}
    for k, o in res.items():
        t = lambda x: torch.as_tensor(x, device=S.dev, dtype=torch.float32)
        args = (t(o["psi"]), t(o["psi_ax"]), t(o["psi_b"]), gR, iz0)
        save[f"{k}_mm_te"] = symmetry_mm(*args, pte, lev)[0].cpu().numpy()
        save[f"{k}_mm_ne"] = symmetry_mm(*args, pne, lev)[0].cpu().numpy()
        for q in ("Rax", "Zax", "psi_ax", "psi_b"):
            save[f"{k}_{q}"] = np.asarray(o[q])
    save["v2_method"] = res["v2"]["method"]
    save["v2_kfit_dv"] = res["v2"]["kfit_dv"]
    np.savez(OUT / out_name, **save)
    for k in ("v1", "v2"):
        print(f"{k}: GPU {save['t_' + k]:.0f}s  median Te mm {np.nanmedian(save[k + '_mm_te']):.4f}"
              f"  median ne mm {np.nanmedian(save[k + '_mm_ne']):.4f}  R_axis med "
              f"{np.nanmedian(save[k + '_Rax']):.3f}", flush=True)
    both = np.isfinite(save["v1_mm_ne"]) & np.isfinite(save["v2_mm_ne"])
    print(f"frames v2 ne-mismatch < v1: {np.mean(save['v2_mm_ne'][both] < save['v1_mm_ne'][both]):.3f}"
          f"  Te: {np.mean(save['v2_mm_te'][both] < save['v1_mm_te'][both]):.3f}  kfit used "
          f"{np.mean(save['v2_method']):.3f}  fitted dv median {np.nanmedian(save['v2_kfit_dv']) * 1e3:.2f} mT",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["truth", "bank", "analyze", "real", "rescore",
                                    "realcheck"])
    ap.add_argument("--names", default="")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--scen", default="B")
    ap.add_argument("--tag", default="p120")
    a = ap.parse_args()
    if a.cmd == "truth":
        if a.scen in ("A", "C"):
            build_truth_from(a.scen)
        else:
            build_truth(a.scen)
    elif a.cmd == "bank":
        if a.tag == "dv100":
            build_bank(a.scen, a.tag, combos=DV_PROFILES, dvs=DV_VALUES)
        else:
            build_bank(a.scen, a.tag)
    elif a.cmd == "analyze":
        analyze_bank(a.scen, a.tag, subset=a.subset)
    elif a.cmd == "realcheck":
        realdata_check()
    elif a.cmd == "rescore":
        rescore_preds(a.scen, a.subset, [n for n in a.names.split(";") if n] or None)
    elif a.cmd == "real":
        real_experiments(a.scen, [n for n in a.names.split(";") if n])
