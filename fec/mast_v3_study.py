"""MAST v3 study: label-free tests of passive currents, vertical position and the centre-column
limiter on top of the production selector fec.mast_select_v2.select_v2 (no MAST labels).

Frozen evaluation subsets (written once by `subsets`, never re-drawn):
  E1  1500 random frames (seed 20261008) of mast_public_test, excluding frames already examined
      by earlier studies (first 1024 stream frames = realcheck/review, the 1000-frame passive
      study subset seed 20261007); sub-task 1 (passive A/B).
  E2  32 random whole shots (seed 20261009), excluding the 40 twin shots (fec.twin_mast seed 0)
      and the 19 realcheck shots; sub-tasks 2 and 3 (vertical position, limiter) and the
      combined v3 check.  Design probes use only shots outside E2.

CLI (GPU steps via fec/gpu_slot.py; outputs under FEC_WORK/v3):
  python -B fec/mast_v3_study.py subsets
  python -B fec/mast_v3_study.py real --set E1 --arms v2,v2_c1
  python -B fec/mast_v3_study.py twin_truth --scen BP|BL|BPL
  python -B fec/mast_v3_study.py twin --scen BP --arms v2,v2_c1
  python -B fec/mast_v3_study.py report --set E1 --arms v2,v2_c1
Docs: docs/research/MAST_V3_STUDY_20261008.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

OUT = WORK / "v3"
CONFIG = "mast_public_test"
SEED_E1, N_E1 = 20261008, 1500
SEED_E2, N_E2 = 20261009, 32
SOL0 = 0.5          # fec.machines.MAST_SOL_TURN_FRACTION (production default)


# ============================================================================ subsets
def _twin_shots():
    """Shots of the fec.twin_mast frames (40 shots, seed 0), recomputed exactly as load_frames."""
    from fec.features import load_ds
    ds = load_ds(CONFIG)
    shot = ds["shot"]
    rng = np.random.default_rng(0)
    nfr = np.bincount(shot, minlength=len(ds["info"]["order"]))
    elig = np.flatnonzero(nfr >= 25)
    return np.sort(rng.choice(elig, 40, replace=False))


def make_subsets():
    """Draw and freeze E1/E2 (refuses to overwrite an existing subsets.json)."""
    from fec.features import load_ds
    f = OUT / "subsets.json"
    if f.exists():
        raise SystemExit(f"{f} exists; subsets are frozen")
    ds = load_ds(CONFIG)
    shot = np.asarray(ds["shot"])
    N = len(shot)
    seen = np.zeros(N, bool)
    seen[:1024] = True                                               # realcheck / review
    seen[np.random.default_rng(20261007).choice(N, size=1000, replace=False)] = True  # passive
    pool = np.flatnonzero(~seen)
    e1 = np.sort(np.random.default_rng(SEED_E1).choice(pool, N_E1, replace=False))
    excl = np.union1d(_twin_shots(), np.unique(shot[:1024]))
    cand = np.setdiff1d(np.unique(shot), excl)
    e2 = np.sort(np.random.default_rng(SEED_E2).choice(cand, N_E2, replace=False))
    e2f = np.flatnonzero(np.isin(shot, e2))
    rec = dict(E1_frames=e1.tolist(), E2_shots=e2.tolist(), E2_frames=e2f.tolist(),
               exclude_shots=excl.tolist(), seeds=dict(E1=SEED_E1, E2=SEED_E2),
               note="frozen before any v3 evaluation; E1 excludes frames 0-1023 and the "
                    "seed-20261007 passive-study frames; E2 excludes twin + realcheck shots")
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec))
    print(f"E1 {len(e1)} frames from {len(np.unique(shot[e1]))} shots; E2 {len(e2)} shots "
          f"{len(e2f)} frames; excluded shots {len(excl)}")
    return rec


def load_subsets():
    return json.loads((OUT / "subsets.json").read_text())


def subset_frames(name):
    if name in ("E3", "E4"):
        return np.asarray(json.loads((OUT / f"subsets_{name}.json").read_text())[f"{name}_frames"],
                          int)
    s = load_subsets()
    return np.asarray(s[f"{name}_frames"], int)


# ============================================================================ inputs
def frame_inputs(idx, config=CONFIG):
    """Real inputs of global frames idx (sorted): coil A, Ip A, Thomson R/te/ne, phase."""
    from fec.features import load_ds
    from fec.machines import MAST_CH, MAST_COILS
    from fec.mast_v1 import thomson_frame
    from fec.passive_eval import _phase
    ds = load_ds(config)
    info = ds["info"]
    idx = np.asarray(idx, int)
    cur = np.asarray(ds["cur"][idx], dtype=np.float64)
    c6 = cur[:, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    Rl, tel, nel, cache = [], [], [], {}
    for g in idx:
        stem = info["order"][ds["shot"][g]]
        if stem not in cache:
            cache.clear()
            cache[stem] = dict(np.load(WORK / "shots" / config / f"{stem}.npz"))
        R, te, ne = thomson_frame(cache[stem], ds["fidx"][g])
        Rl.append(R); tel.append(te); nel.append(ne)
    nch = max(len(r) for r in Rl)
    pad = lambda L: np.stack([np.pad(np.asarray(x, float), (0, nch - len(x)),
                                     constant_values=np.nan) for x in L])
    ph, _, dIp = _phase(ds, idx)
    return dict(idx=idx, shot=np.asarray(ds["shot"][idx]), t_ms=ds["t_ms"][idx],
                I=c6[:, ci] * 1e3,
                Ip=np.maximum(c6[:, MAST_CH.index("magnetics_plasma_current")] * 1e3, 5e4),
                Itf=c6[:, MAST_CH.index("magnetics_tf_current")],
                R=pad(Rl), te=pad(tel), ne=pad(nel), phase=ph.astype(str), dIp=dIp)


def passive_currents(variant, idx, config=CONFIG, sol=SOL0, p6=1.0):
    if not variant:
        return None
    if variant == "c1conv":
        return conv_passive(idx, sol, p6, config).astype(np.float64)
    I = np.load(WORK / "passive" / "frames" / config / variant / "I.npy", mmap_mode="r")
    return np.asarray(I[np.asarray(idx, int)], dtype=np.float64)


def midplane_lcfs(psi, psi_ax, psi_b, gR, gZ):
    """Midplane crossings (R_in, R_out) of psiN = 1 (walk out from the row minimum in
    0.3-1.4 m; R_in = gR[0] if psiN < 1 down to the grid edge).  NaN when no boundary."""
    iz0 = int(np.argmin(np.abs(gZ)))
    B = len(psi_ax)
    Rin = np.full(B, np.nan); Rout = np.full(B, np.nan)
    Rf = np.linspace(gR[0], gR[-1], 1941)
    w = (Rf > 0.3) & (Rf < 1.4)
    for i in range(B):
        if not (np.isfinite(psi_b[i]) and psi_b[i] > -1e29 and psi_ax[i] > psi_b[i]):
            continue
        pf = np.interp(Rf, gR, (psi_ax[i] - np.asarray(psi[i][iz0], float)) /
                       (psi_ax[i] - psi_b[i]))
        j0 = int(np.argmin(np.where(w, pf, np.inf)))
        if pf[j0] >= 1:
            continue
        jo = np.flatnonzero(pf[j0:] >= 1); ji = np.flatnonzero(pf[:j0 + 1][::-1] >= 1)
        Rout[i] = Rf[j0 + jo[0]] if len(jo) else np.nan
        Rin[i] = Rf[j0 - ji[0]] if len(ji) else gR[0]
    return Rin, Rout


def score_mm(S, out, R, te, ne):
    """Te and ne symmetry mismatch (17 levels, GPU) of an equilibrium dict."""
    import torch
    from fec.mast_select_v2 import LEVELS_V1, prep_profiles, symmetry_mm, to_dev
    dev = S.dev
    t = lambda x: torch.as_tensor(np.asarray(x), device=dev, dtype=torch.float32)
    gR = t(S.spec.gR)
    iz0 = int(np.argmin(np.abs(S.spec.gZ)))
    lev = t(LEVELS_V1)
    args = (t(out["psi"]), t(out["psi_ax"]), t(out["psi_b"]), gR, iz0)
    mte = symmetry_mm(*args, to_dev(prep_profiles(R, te), dev), lev)[0].cpu().numpy()
    mne = symmetry_mm(*args, to_dev(prep_profiles(R, ne), dev), lev)[0].cpu().numpy()
    return mte, mne


# ============================================================================ coil conventions
P6_COLS = ("magnetics_p6l_current", "magnetics_p6u_current")


def conv_scale(sol=SOL0, p6=1.0):
    """Per-column multiplier of the MAST_COILS coil matrix for a convention (sol, p6)."""
    from fec.machines import MAST_COILS
    f = np.ones(len(MAST_COILS))
    f[MAST_COILS.index("magnetics_sol_current")] = sol / SOL0
    for c in P6_COLS:
        f[MAST_COILS.index(c)] = p6
    return f


def conv_matrix(sol=SOL0, p6=1.0):
    """(spec, M) for a convention: production matrix with scaled columns.  sol=1.0 equals
    fec.machines.mast_setup(sol_fraction=1.0) (the solenoid column is linear in the fraction)."""
    from fec.machines import mast_setup
    spec, M = mast_setup()
    return spec, M * conv_scale(sol, p6)[None, :]


def conv_passive(idx, sol=SOL0, p6=1.0, config=CONFIG):
    """C1 passive currents (N,386) [A] at global frames idx, integrated with the coil-to-vessel
    drive of convention (sol, p6) (fec.passive.c1_shot on the 0.5 ms input signals; same L, R,
    nominal plasma).  Cached under FEC_WORK/v3/passive_conv/."""
    import copy
    import hashlib
    from fec.features import load_ds
    from fec.passive import OUT as POUT, PassiveModel, c1_shot
    idx = np.asarray(idx, int)
    key = hashlib.sha1(idx.tobytes()).hexdigest()[:10]
    f = OUT / "passive_conv" / f"c1_s{sol:g}_p{p6:g}_{config}_{key}.npy"
    if f.exists():
        return np.load(f)
    base = PassiveModel()
    m = copy.copy(base)
    m.Mpc = base.Mpc * conv_scale(sol, p6)[None, :]
    modes = m.modes(1.0)
    ds = load_ds(config)
    order = ds["info"]["order"]
    out = np.zeros((len(idx), len(m.piece)))
    stems = [order[s_] for s_ in ds["shot"][idx]]
    for stem in dict.fromkeys(stems):
        k = np.flatnonzero(np.array(stems) == stem)
        z = np.load(POUT / "sig" / config / f"{stem}.npz")
        t, I = c1_shot(m, z, 1.0, modes)
        tf = ds["t_ms"][idx[k]]
        for j in range(I.shape[1]):
            out[k, j] = np.interp(tf, t, I[:, j])
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, out.astype(np.float32))
    return out.astype(np.float32)


# ============================================================================ arms
# name -> dict(passive variant | None, selector kind, selector kwargs, limiter on/off)
def _arm(passive=None, kind="v3", lim=False, sol=SOL0, p6=1.0, **kw):
    return dict(passive=passive, kind=kind, lim=lim, kw=kw, sol=sol, p6=p6)


ARMS = {
    "v2": _arm(kind="v2"),
    "v2_c1": _arm("c1_r1", kind="v2"),
    "v1": _arm(kind="v1"),                                   # diagnostic only (sub-task 3)
    "z_root": _arm(zmode="root", kfit_warm="zs"),
    "z_root_t": _arm(zmode="root_t", kfit_warm="o0"),
    "z_root_t_zs": _arm(zmode="root_t", kfit_warm="zs"),
    "lim": _arm(lim=True),
    "v3": _arm(lim=True, zmode="root_t", kfit_warm="zs"),
    "v3_zroot": _arm(lim=True, zmode="root", kfit_warm="zs"),
    "v3_zt_o0": _arm(lim=True, zmode="root_t", kfit_warm="o0"),
    "v3_c1": _arm("c1_r1", lim=True, zmode="root_t", kfit_warm="zs"),
}
for _s in (0.5, 1.0):
    for _p in (1.0, 4.0):
        for _c in ("n", "c"):
            ARMS[f"cv_s{int(round(_s * 10)):02d}_p{int(_p)}_{_c}"] = _arm(
                "c1conv" if _c == "c" else None, kind="v2", sol=_s, p6=_p)
CV_ARMS = [a for a in ARMS if a.startswith("cv_")]
# R6: coil-model arms on the verified-height solver (zmode root_t, warm start zs, no limiter)
for _p in (1.0, 4.0):
    for _c in ("n", "c"):
        ARMS[f"zv_s05_p{int(_p)}_{_c}"] = _arm("c1conv" if _c == "c" else None, sol=0.5, p6=_p,
                                               zmode="root_t", kfit_warm="zs")
ZV_ARMS = [a for a in ARMS if a.startswith("zv_")]
TWIN_ARMS = ARMS


def make_real_solver(passive, lim=False, sol=SOL0, p6=1.0, cache={}):
    from fec.mast_select_v3 import LIMITER_CC, V3Solver
    key = (passive, lim, sol, p6)
    if key not in cache:
        cache.clear()
        spec, M = conv_matrix(sol, p6)
        if passive:
            from fec.passive import PassiveModel
            M = PassiveModel().augment(M)
        cache[key] = V3Solver(spec, M, limiter=LIMITER_CC if lim else None)
    return cache[key]


def run_selector(kind, S, I, Ip, R, te, ne, shot, kw, dv_col=False):
    if kind == "v2":
        from fec.mast_select_v2 import select_v2
        return select_v2(S, I, Ip, R, te, ne, shot, dv_col=dv_col, **kw)
    if kind == "v1":
        from fec.mast_select_v2 import select_batch
        from fec.mast_v1 import COMBOS, DEFAULT
        return select_batch(S, I, Ip, R, te, ne, shot, COMBOS, DEFAULT, dv_col=dv_col)
    from fec.mast_select_v3 import select_v3
    return select_v3(S, I, Ip, R, te, ne, shot, dv_col=dv_col, **kw)


KEEP = ("psi_ax", "psi_b", "Rax", "Zax", "Zt", "c", "dI", "method", "kfit_dv", "kfit_npairs",
        "score", "limited", "zflag")


def run_real(set_name, arms, max_frames=1024, keep_psi=True):
    """Run named arms on a frozen real-input subset; one npz per arm."""
    import torch
    from fec.mast_select_v2 import shot_chunks
    idx = subset_frames(set_name)
    fr = frame_inputs(idx)
    for arm in arms:
        A = ARMS[arm]
        passive, kind, kw = A["passive"], A["kind"], A["kw"]
        S = make_real_solver(passive, A["lim"], A["sol"], A["p6"])
        Ipas = passive_currents(passive, idx, sol=A["sol"], p6=A["p6"])
        I = fr["I"] if Ipas is None else np.concatenate([fr["I"], Ipas], 1)
        res = {}
        t0 = time.time()
        for sl in shot_chunks(fr["shot"], max_frames):
            o = run_selector(kind, S, I[sl], fr["Ip"][sl], fr["R"][sl], fr["te"][sl],
                             fr["ne"][sl], fr["shot"][sl], kw)
            mte, mne = score_mm(S, o, fr["R"][sl], fr["te"][sl], fr["ne"][sl])
            Rin, Rout = midplane_lcfs(o["psi"], o["psi_ax"], o["psi_b"], S.spec.gR, S.spec.gZ)
            nsl = sl.stop - sl.start
            part = {k: np.asarray(o[k]) for k in KEEP
                    if k in o and np.ndim(o[k]) >= 1 and np.shape(o[k])[0] == nsl}
            part.update(mm_te=mte, mm_ne=mne, Rin=Rin, Rout=Rout)
            if keep_psi:
                part["psi"] = np.asarray(o["psi"], np.float32)
            for k, v in part.items():
                res.setdefault(k, []).append(v)
            torch.cuda.empty_cache()
            print(f"[{set_name}/{arm}] {sl.stop}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
        res = {k: np.concatenate(v) for k, v in res.items()}
        res.update(idx=idx, shot=fr["shot"], phase=fr["phase"], t_ms=fr["t_ms"],
                   gpu_s=np.array(time.time() - t0))
        np.savez(OUT / f"real_{set_name}_{arm}.npz", **res)
        print(f"[{set_name}/{arm}] done {time.time() - t0:.0f}s  Te mm med "
              f"{np.nanmedian(res['mm_te']):.4f}  ne mm med {np.nanmedian(res['mm_ne']):.4f}",
              flush=True)


# ============================================================================ statistics
def cluster_boot_median(d, shot, n=2000, seed=0, level=0.95):
    """Median of d and its shot-cluster bootstrap CI at `level` (finite d only)."""
    ok = np.isfinite(d)
    d, shot = d[ok], shot[ok]
    us, inv = np.unique(shot, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(us))]
    rng = np.random.default_rng(seed)
    meds = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(us), len(us))
        meds[b] = np.median(d[np.concatenate([groups[k] for k in pick])])
    a = 100 * (1 - level) / 2
    return float(np.median(d)), float(np.percentile(meds, a)), \
        float(np.percentile(meds, 100 - a)), int(ok.sum())


def fmt_ci(t):
    m, lo, hi, n = t
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={n})"


# ============================================================================ twin
TWIN = OUT / "twin"


class _TwinOut:
    """Temporarily point fec.twin_mast at another truth directory (its OUT global)."""

    def __init__(self, d):
        self.d = Path(d)

    def __enter__(self):
        import fec.twin_mast as tm
        self.old, tm.OUT = tm.OUT, self.d
        return tm

    def __exit__(self, *a):
        import fec.twin_mast as tm
        tm.OUT = self.old


# twin truth worlds of R5: name -> (solenoid fraction, P6 turns, passive variant)
# (W_s05_p1_n is scenario B and W_s05_p1_c equals BP; both are rebuilt only if requested)
WORLDS = {f"W{a[3:]}": (ARMS[a]["sol"], ARMS[a]["p6"], ARMS[a]["passive"]) for a in CV_ARMS}


def twin_dir(scen):
    return (WORK / "twin") if scen in ("A", "B", "C") else TWIN


def twin_solver(passive, limiter=None, sol=SOL0, p6=1.0):
    """Twin solver: coil matrix [+ C1 passive columns] + uniform-Bz column (last).
    (Scenario BP truths and the first BP/B reconstructions were built with fec.twin_mast's
    TwinSolver, whose solve/solve_gen numerics V3Solver reproduces when limiter is None.)"""
    from fec.machines import mast_setup
    import fec.twin_mast as tm
    spec, M = mast_setup(sol_fraction=tm.sol_fraction())
    M = M * conv_scale(sol, p6)[None, :]
    RR, _ = np.meshgrid(spec.gR, spec.gZ)
    cols = [M]
    if passive:
        from fec.passive import PassiveModel
        cols.append(PassiveModel().augment(np.zeros((M.shape[0], 0))))
    cols.append((RR ** 2 / 2).reshape(-1, 1))
    Mx = np.concatenate(cols, 1)
    from fec.mast_select_v3 import V3TwinSolver   # == TwinSolver numerics when limiter None
    return V3TwinSolver(spec, Mx, limiter=limiter)


def build_twin_truth(scen):
    """BP = scenario B + C1 passive flux; BL = B with the centre-column limiter;
    BPL = both.  Same frames, profile tracks and dv as scenario B."""
    import fec.twin_mast as tm
    with _TwinOut(WORK / "twin"):
        fr, trb, tsb, prof, dv = tm.load_truth("B")
    if scen in WORLDS:
        sol, p6, passive = WORLDS[scen]
        lim = False
    else:
        sol, p6 = SOL0, 1.0
        passive = "c1_r1" if "P" in scen[1:] else None
        lim = "L" in scen[1:]
    if lim:
        from fec.mast_select_v3 import LIMITER_CC
    S = twin_solver(passive, LIMITER_CC if lim else None, sol, p6)
    fr2 = dict(fr)
    Ipas = passive_currents(passive, fr["g"], sol=sol, p6=p6)
    if Ipas is not None:
        fr2["I"] = np.concatenate([fr["I"], Ipas], 1)
    t0 = time.time()
    tr = tm.solve_truth(S, fr2, prof, dv)
    if lim:
        tr["limited"] = S.last_limited
    tr["track"] = trb["track"]
    tr["te_peak_real"] = trb["te_peak_real"]
    spec = S.spec
    ts = tm.synth_thomson(fr, tr, spec.gR, spec.gZ)
    TWIN.mkdir(parents=True, exist_ok=True)
    np.savez(TWIN / f"truth_{scen}.npz", prof=prof, dv=dv, sol_fraction=trb["sol_fraction"],
             **{f"fr_{k}": v for k, v in fr.items()},
             **{f"tr_{k}": v for k, v in tr.items()}, **{f"ts_{k}": v for k, v in ts.items()})
    with _TwinOut(TWIN):
        tm.truth_table(scen)
    print(f"truth {scen}: {time.time() - t0:.0f}s valid {twin_valid(tr).mean():.3f} Rax pct "
          f"{np.percentile(tr['Rax'], [5, 50, 95]).round(3).tolist()}", flush=True)


def twin_valid(tr):
    """fec.twin_mast.truth_valid, except that limited truths need no X-point."""
    lim = tr.get("limited", np.zeros(len(tr["dI"]), bool))
    return (tr["dI"] < 1e-3) & (tr["xfound"] | lim) & (np.abs(tr["c"]) < 1e-5) & \
        np.isfinite(tr["psi_b"]) & (tr["psi_ax"] > tr["psi_b"]) & (tr["Rax"] < 1.28) & \
        (tr["Rax"] > 0.47)


def twin_realistic(scen, tr):
    """Valid truths whose midplane LCFS lies in 0.195 <= R_in, R_out <= 1.55 m (0.195 admits
    truths touching the R = 0.20 m limiter)."""
    import fec.twin_mast as tm
    with _TwinOut(twin_dir(scen)):
        Q, _ = tm.truth_table(scen)
    Rin, Rout = Q[:, tm.QCOLS.index("Rin")], Q[:, tm.QCOLS.index("Rout")]
    with np.errstate(invalid="ignore"):
        return twin_valid(tr) & np.isfinite(Rout) & (Rout <= 1.55) & (Rin >= 0.195)


def twin_eval(scen, pred, subset):
    """fec.twin_mast metrics of a prediction on a truth subset (boolean mask)."""
    import fec.twin_mast as tm
    from fec.machines import mast_setup
    from multiprocessing import Pool
    with _TwinOut(twin_dir(scen)):
        fr, tr, ts, prof, dv = tm.load_truth(scen)
        Qtr, Ctr = tm.truth_table(scen)
        spec, _ = mast_setup()
        box = tm._box()
        F = tm.MU0 * tm.N_TF * np.abs(fr["Itf"]) * 1e3 / (2 * np.pi)
        v = np.flatnonzero(subset)
        P = np.asarray(pred["psi"], np.float64)
        sse = ((P[v] - tr["psi"][v]) ** 2).sum((1, 2))
        jobs = [(P[i], float(pred["psi_ax"][i]), float(pred["psi_b"][i]), float(pred["Rax"][i]),
                 float(pred["Zax"][i]), F[i], spec.gR, spec.gZ, box, Ctr[i]) for i in v]
        with Pool(6) as pool:
            res = pool.map(tm._qrow, jobs, chunksize=8)
        q = np.stack([r[0] for r in res])
        return tm.metrics_from_rows(q, sse, Qtr, tr, fr["shot"], subset)


def load_twin(scen):
    import fec.twin_mast as tm
    with _TwinOut(twin_dir(scen)):
        return tm.load_truth(scen)


def run_twin(scen, arms):
    """Reconstruct twin truth `scen` with the named arms (production calls, dv column)."""
    import torch
    fr, tr, ts, prof, dv = load_twin(scen)
    TWIN.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        A = TWIN_ARMS[arm]
        passive, kind, kw = A["passive"], A["kind"], A["kw"]
        from fec.mast_select_v3 import LIMITER_CC
        S = twin_solver(passive, LIMITER_CC if A["lim"] else None, A["sol"], A["p6"])
        Ipas = passive_currents(passive, fr["g"], sol=A["sol"], p6=A["p6"])
        I = fr["I"] if Ipas is None else np.concatenate([fr["I"], Ipas], 1)
        torch.cuda.synchronize(); t0 = time.time()
        o = run_selector(kind, S, I, fr["Ip"], fr["R"], ts["te"], ts["ne"], fr["shot"], kw,
                         dv_col=True)
        torch.cuda.synchronize()
        o = {k: np.asarray(v) for k, v in o.items() if not isinstance(v, (dict, list))}
        o["gpu_s"] = np.array(time.time() - t0)
        o["mm_te"], o["mm_ne"] = score_mm(S, o, fr["R"], ts["te"], ts["ne"])
        o["Rin"], o["Rout"] = midplane_lcfs(o["psi"], o["psi_ax"], o["psi_b"], S.spec.gR,
                                            S.spec.gZ)
        np.savez(TWIN / f"pred_{scen}_{arm}.npz", **o)
        del S
        torch.cuda.empty_cache()
        print(f"[twin {scen}/{arm}] {float(o['gpu_s']):.0f}s Te mm {np.nanmedian(o['mm_te']):.4f}"
              f" ne mm {np.nanmedian(o['mm_ne']):.4f}", flush=True)


def twin_table(scen, arms):
    """Metrics rows (all-valid and realistic) of saved twin predictions."""
    fr, tr, ts, prof, dv = load_twin(scen)
    allv = twin_valid(tr)
    real = twin_realistic(scen, tr)
    rows = {}
    for arm in arms:
        p = dict(np.load(TWIN / f"pred_{scen}_{arm}.npz"))
        m = twin_eval(scen, p, allv)
        mr = twin_eval(scen, p, real)
        m.update({f"real_{k}": v for k, v in mr.items()})
        m["n_all"], m["n_real"] = int(allv.sum()), int(real.sum())
        if "limited" in tr and (allv & tr["limited"]).sum() > 20:
            ml = twin_eval(scen, p, allv & tr["limited"])
            m.update({f"limtruth_{k}": v for k, v in ml.items()})
            m["n_limtruth"] = int((allv & tr["limited"]).sum())
        m["mm_te_med"] = float(np.nanmedian(p["mm_te"][allv]))
        m["mm_ne_med"] = float(np.nanmedian(p["mm_ne"][allv]))
        m["zclip"] = float(np.mean(np.abs(p["Zt"]) >= 0.2999))
        rows[arm] = m
    (TWIN / f"table_{scen}.json").write_text(json.dumps(
        {a: {k: float(v) for k, v in m.items()} for a, m in rows.items()}, indent=1))
    return rows


TWIN_COLS = ("S_proxy", "R2psi", "R2psi_pl", "R_axis_mae_cm", "Z_axis_mae_cm", "q95_mape",
             "D_LCFS")


def twin_report(scen, arms, base, matched=None):
    """Twin rows (all-valid | realistic) and paired mismatch deltas arm - base (all-valid).
    matched: name of the arm that matches the truth world (R1 D_twin = other - matched)."""
    rows = twin_table(scen, arms)
    fr, tr, ts, prof, dv = load_twin(scen)
    allv = twin_valid(tr)
    P = {a: dict(np.load(TWIN / f"pred_{scen}_{a}.npz")) for a in arms}
    lines = [f"| arm | " + " | ".join(TWIN_COLS) + " | realistic " +
             " | ".join(TWIN_COLS) + " | Zt clip | median mm Te / ne | paired d_ne vs "
             f"{base} | paired d_Te vs {base} | kfit dv median mT |",
             "|---|" + "---|" * (2 * len(TWIN_COLS) + 5)]
    extra = {}
    for a in arms:
        m, p = rows[a], P[a]
        dne = cluster_boot_median(np.where(allv, p["mm_ne"] - P[base]["mm_ne"], np.nan), fr["shot"])
        dte = cluster_boot_median(np.where(allv, p["mm_te"] - P[base]["mm_te"], np.nan), fr["shot"])
        k1 = allv & (p["method"] == 1)
        dvm = 1e3 * float(np.nanmedian(p["kfit_dv"][k1])) if k1.any() else float("nan")
        extra[a] = dict(d_ne=dne, d_te=dte, dv_med_mT=dvm)
        f = lambda k: (f"{m[k]:.4f}" if k.startswith(("S_", "R2", "D_")) else f"{m[k]:.2f}")
        g = lambda k: (f"{m['real_' + k]:.4f}" if k.startswith(("S_", "R2", "D_"))
                       else f"{m['real_' + k]:.2f}")
        lines.append(f"| {a} | " + " | ".join(f(k) for k in TWIN_COLS) + " | " +
                     " | ".join(g(k) for k in TWIN_COLS) + f" | {m['zclip']:.3f} | "
                     f"{m['mm_te_med']:.4f} / {m['mm_ne_med']:.4f} | {fmt_ci(dne)} | "
                     f"{fmt_ci(dte)} | {dvm:+.1f} |")
    head = f"twin {scen}: all-valid n={rows[arms[0]]['n_all']}, realistic n={rows[arms[0]]['n_real']}" \
        f", truth dv median {1e3 * float(np.median(dv[allv])):+.1f} mT"
    if matched is not None:
        other = [a for a in arms if a != matched][0]
        for q in ("ne", "te"):
            extra[f"D_twin_{q}"] = cluster_boot_median(
                np.where(allv, P[other][f"mm_{q}"] - P[matched][f"mm_{q}"], np.nan), fr["shot"])
        head += (f"; D_twin (mismatched {other} - matched {matched}): ne "
                 f"{fmt_ci(extra['D_twin_ne'])}, Te {fmt_ci(extra['D_twin_te'])}")
    (TWIN / f"report_{scen}_{'_'.join(arms)}.json").write_text(json.dumps(
        dict(rows={a: {k: float(v) for k, v in r.items()} for a, r in rows.items()},
             extra=extra), indent=1))
    return head + "\n" + "\n".join(lines)


def real_scalars(set_name, arm, nproc=6):
    """Stage-2 q95 / R_geo (fec.mast_v1._scal_worker, CPU) of a saved real-subset arm."""
    from multiprocessing import Pool
    from fec.machines import mast_setup
    from fec.mast_v1 import _scal_worker
    f = OUT / f"scal_{set_name}_{arm}.npz"
    if f.exists():
        return dict(np.load(f))
    spec, _ = mast_setup()
    r = load_real(set_name, arm)
    fr = frame_inputs(r["idx"])
    jobs = [(r["psi"][i].astype(np.float64), r["psi_ax"][i], r["psi_b"][i], r["Rax"][i],
             r["Zax"][i], fr["Itf"][i], fr["Ip"][i], fr["R"][i], fr["te"][i], fr["ne"][i],
             spec.gR, spec.gZ) for i in range(len(r["idx"]))]
    with Pool(nproc) as pool:
        res = pool.map(_scal_worker, jobs, chunksize=16)
    out = {k: np.array([x[k] for x in res], float) for k in ("q95", "q90", "betaN_e", "Rgeo", "a")}
    np.savez(f, **out)
    return out


def shot_steps(x, shot):
    """|x[i+1]-x[i]| for consecutive frames of the same shot."""
    same = shot[1:] == shot[:-1]
    return np.abs(np.diff(x))[same]


def e2_metrics(set_name, arm, base="v2"):
    """Real-input diagnostics of sub-tasks 2/3 for one arm (vs base for the paired deltas)."""
    r = load_real(set_name, arm)
    sc = real_scalars(set_name, arm)
    shot = r["shot"]
    st = shot_steps(r["Zax"], shot)
    med = np.array([np.nanmedian(sc["Rgeo"][shot == s_]) for s_ in shot])
    m = dict(n=len(shot), zclip=float(np.mean(np.abs(r["Zt"]) >= 0.2999)),
             jump10=float(np.mean(st > 0.10)), n_jump10=int(np.sum(st > 0.10)),
             dZax_med_mm=1e3 * float(np.median(st)), dZax_p90_mm=1e3 * float(np.percentile(st, 90)),
             cross=float(np.mean(r["Rin"] < 0.195)),
             limited=float(np.mean(r["limited"])) if "limited" in r else 0.0,
             sentinel=int(np.sum(r["psi_b"] < -1e29)),
             q95_gt20=float(np.mean(sc["q95"] > 20)), q95_finite=float(np.mean(np.isfinite(sc["q95"]))),
             q95_med=float(np.nanmedian(sc["q95"])), q95_p95=float(np.nanpercentile(sc["q95"], 95)),
             rgeo_dip=float(np.mean((med - sc["Rgeo"]) > 0.06)),
             te_med=float(np.nanmedian(r["mm_te"])), ne_med=float(np.nanmedian(r["mm_ne"])),
             zflag=np.bincount(r["zflag"].astype(int), minlength=4).tolist() if "zflag" in r else None)
    if arm != base:
        b = load_real(set_name, base)
        cross_b = b["Rin"] < 0.195
        for q in ("ne", "te"):
            d = r[f"mm_{q}"] - b[f"mm_{q}"]
            m[f"d_{q}"] = cluster_boot_median(d, shot)
            m[f"d_{q}_basecross"] = cluster_boot_median(np.where(cross_b, d, np.nan), shot)
        m["dZt_abs_med_mm"] = 1e3 * float(np.median(np.abs(r["Zt"] - b["Zt"])))
    return m


def e2_report(set_name, arms, base="v2"):
    rows = {a: e2_metrics(set_name, a, base) for a in arms}
    head = ("| arm | Zt clip | steps >10 cm (n) | median / p90 step mm | R_in<0.195 | limited | "
            "sentinel | q95>20 | q95 p95 | R_geo dip | median Te / ne mm | d_ne vs base | "
            "d_Te vs base | d_ne on base-crossing frames | zflag 0/1/2/3 |\n|---|" + "---|" * 14)
    lines = [head]
    for a, m in rows.items():
        dne = fmt_ci(m["d_ne"]) if "d_ne" in m else "-"
        dte = fmt_ci(m["d_te"]) if "d_te" in m else "-"
        dnc = fmt_ci(m["d_ne_basecross"]) if "d_ne_basecross" in m else "-"
        lines.append(f"| {a} | {m['zclip']:.3f} | {m['jump10']:.3f} ({m['n_jump10']}) | "
                     f"{m['dZax_med_mm']:.1f} / {m['dZax_p90_mm']:.1f} | {m['cross']:.3f} | "
                     f"{m['limited']:.3f} | {m['sentinel']} | {m['q95_gt20']:.3f} | "
                     f"{m['q95_p95']:.1f} | {m['rgeo_dip']:.3f} | {m['te_med']:.4f} / "
                     f"{m['ne_med']:.4f} | {dne} | {dte} | {dnc} | {m['zflag']} |")
    (OUT / f"e2_report_{set_name}.json").write_text(json.dumps(rows, indent=1, default=float))
    return "\n".join(lines), rows


def r5_report(set_name="E3", base="cv_s05_p1_n", family=None):
    """Rule R5/R6 table: every coil-model arm vs the base on the real subset + twin checks."""
    family = family or CV_ARMS
    arms = [a for a in family if a != base]
    A = load_real(set_name, base)
    shot = A["shot"]
    bonf = 1 - 0.05 / len(arms)

    def stats(r):
        k = r["method"] == 1
        return dict(cross=float(np.mean(r["Rin"] < 0.195)),
                    clip=float(np.mean(np.abs(r["Zt"]) >= 0.2999)),
                    absdv=1e3 * float(np.nanmedian(np.abs(r["kfit_dv"][k]))),
                    dv=1e3 * float(np.nanmedian(r["kfit_dv"][k])), kfit=float(np.mean(k)),
                    te=float(np.nanmedian(r["mm_te"])), ne=float(np.nanmedian(r["mm_ne"])))
    base_s = stats(A)
    rows, lines = {}, []
    for a in arms:
        r = load_real(set_name, a)
        st_ = stats(r)
        dne = r["mm_ne"] - A["mm_ne"]
        dte = r["mm_te"] - A["mm_te"]
        m = dict(st_, d_ne=cluster_boot_median(dne, shot),
                 d_ne_bonf=cluster_boot_median(dne, shot, level=bonf),
                 d_te=cluster_boot_median(dte, shot),
                 d_ne_phase={ph: cluster_boot_median(np.where(A["phase"] == ph, dne, np.nan),
                                                     shot)
                             for ph in ("ramp_up", "flat", "ramp_down")})
        W = "W" + a[3:]                       # the arm's own twin truth world
        if W == "Ws05_p1_c":
            scen, p0, p1 = ("BP", "v2", "v2_c1") if a.startswith("cv_") else ("BP", base, a)
        else:
            scen, p0, p1 = W, base, a
        try:
            fr, tr, ts, prof, dv = load_twin(scen)
            allv = twin_valid(tr)
            P0 = dict(np.load(TWIN / f"pred_{scen}_{p0}.npz"))
            P1 = dict(np.load(TWIN / f"pred_{scen}_{p1}.npz"))
            v = np.flatnonzero(allv)
            sst = np.sum((tr["psi"][v] - tr["psi"][v].mean()) ** 2)
            r2 = lambda P: float(1 - np.sum((P["psi"][v] - tr["psi"][v]) ** 2) / sst)
            tw = dict(world=scen,
                      D_ne=cluster_boot_median(np.where(allv, P0["mm_ne"] - P1["mm_ne"], np.nan),
                                               fr["shot"]),
                      D_te=cluster_boot_median(np.where(allv, P0["mm_te"] - P1["mm_te"], np.nan),
                                               fr["shot"]),
                      R2psi_A0=r2(P0), R2psi_matched=r2(P1))
        except FileNotFoundError as e:
            tw = dict(missing=str(e))
        m["twin"] = tw
        chk = dict(i=m["d_te"][2] <= 0.002, ii=st_["cross"] <= base_s["cross"] + 0.01,
                   iii=st_["clip"] <= base_s["clip"] + 0.01,
                   iv=st_["absdv"] <= base_s["absdv"] + 0.5,
                   v=st_["kfit"] >= base_s["kfit"] - 0.01,
                   vi=bool("R2psi_A0" in tw and tw["R2psi_matched"] >= tw["R2psi_A0"]))
        m["checks"] = chk
        m["adopt"] = bool(m["d_ne_bonf"][2] < 0 and all(chk.values()))
        n_alt = (a[3:6] != "s05") + (a[7:9] != "p1") + (a[-1] != "n")
        m["reject"] = bool(n_alt == 1 and m["d_ne"][1] > 0 and "D_ne" in tw and tw["D_ne"][1] > 0)
        rows[a] = m
        tws = (f"{tw['world']}: D_ne {fmt_ci(tw['D_ne'])}, R2psi A0 {tw['R2psi_A0']:.4f} -> "
               f"matched {tw['R2psi_matched']:.4f}") if "D_ne" in tw else "missing"
        failed = "".join(f"({k})" for k, ok in chk.items() if not ok) or "none"
        verdict = "ADOPT" if m["adopt"] else ("REJECT" if m["reject"] else "inconclusive")
        lines.append(f"| {a} | {fmt_ci(m['d_ne'])} | {m['d_ne_bonf'][1]:+.4f}, "
                     f"{m['d_ne_bonf'][2]:+.4f} | {fmt_ci(m['d_te'])} | {st_['ne']:.4f} / "
                     f"{st_['te']:.4f} | {st_['cross']:.3f} | {st_['clip']:.3f} | "
                     f"{st_['dv']:+.1f} ({st_['absdv']:.1f}) | {st_['kfit']:.3f} | {failed} | "
                     f"{tws} | {verdict} |")
    head = (f"A0 {base}: ne / Te mm median {base_s['ne']:.4f} / {base_s['te']:.4f}, crossing "
            f"{base_s['cross']:.3f}, clip {base_s['clip']:.3f}, kfit dv {base_s['dv']:+.1f} mT "
            f"(|dv| {base_s['absdv']:.1f}), kfit used {base_s['kfit']:.3f}")
    table = (f"| arm | paired d_ne [95% CI] | Bonferroni {100 * bonf:.2f}% CI | paired d_Te [95% CI] | "
             "median ne / Te mm | crossing | Zt clip | kfit dv median (abs) mT | kfit used | "
             "failed checks | twin (own world) | R5 |")
    sep = "|---|" + "---|" * 11
    (OUT / f"r5_report_{set_name}_{base}.json").write_text(
        json.dumps(dict(base=base_s, rows=rows), indent=1, default=float))
    return "\n".join([head, table, sep] + lines), rows


def load_real(set_name, arm):
    return dict(np.load(OUT / f"real_{set_name}_{arm}.npz"))


def paired_report(set_name, base, arm, phases=True):
    """R1-style paired statistics of arm vs base on a real subset (text lines + dict)."""
    a, b = load_real(set_name, base), load_real(set_name, arm)
    shot = a["shot"]
    lines, res = [], {}
    groups = [("all", np.ones(len(shot), bool))]
    if phases:
        groups += [(p, a["phase"] == p) for p in ("ramp_up", "flat", "ramp_down")]
    for gname, gm in groups:
        row = {}
        for q in ("te", "ne"):
            d = np.where(gm, b[f"mm_{q}"] - a[f"mm_{q}"], np.nan)
            row[f"d_{q}"] = cluster_boot_median(d, shot)
            row[f"med_{q}"] = (float(np.nanmedian(a[f"mm_{q}"][gm])),
                               float(np.nanmedian(b[f"mm_{q}"][gm])))
            both = gm & np.isfinite(a[f"mm_{q}"]) & np.isfinite(b[f"mm_{q}"])
            row[f"frac_better_{q}"] = float(np.mean(b[f"mm_{q}"][both] < a[f"mm_{q}"][both]))
        k1 = gm & (a["method"] == 1) & (b["method"] == 1)
        row["dv_med_mT"] = (1e3 * float(np.nanmedian(a["kfit_dv"][k1])),
                            1e3 * float(np.nanmedian(b["kfit_dv"][k1])))
        row["absdv_med_mT"] = (1e3 * float(np.nanmedian(np.abs(a["kfit_dv"][k1]))),
                               1e3 * float(np.nanmedian(np.abs(b["kfit_dv"][k1]))))
        row["dv_bound"] = (float(np.mean(np.abs(a["kfit_dv"][k1]) > 0.0199)),
                           float(np.mean(np.abs(b["kfit_dv"][k1]) > 0.0199)))
        row["cc_contact"] = (float(np.mean(a["Rin"][gm] < 0.20)),
                             float(np.mean(b["Rin"][gm] < 0.20)))
        row["zclip"] = (float(np.mean(np.abs(a["Zt"][gm]) >= 0.2999)),
                        float(np.mean(np.abs(b["Zt"][gm]) >= 0.2999)))
        row["sentinel"] = (int(np.sum(a["psi_b"][gm] < -1e29)), int(np.sum(b["psi_b"][gm] < -1e29)))
        row["Rax_med"] = (float(np.nanmedian(a["Rax"][gm])), float(np.nanmedian(b["Rax"][gm])))
        row["n"] = int(gm.sum())
        res[gname] = row
        lines.append(f"| {gname} | {row['n']} | {row['med_ne'][0]:.4f} -> {row['med_ne'][1]:.4f} | "
                     f"{fmt_ci(row['d_ne'])} | {row['frac_better_ne']:.2f} | "
                     f"{row['med_te'][0]:.4f} -> {row['med_te'][1]:.4f} | {fmt_ci(row['d_te'])} | "
                     f"{row['dv_med_mT'][0]:+.1f} -> {row['dv_med_mT'][1]:+.1f} "
                     f"(abs {row['absdv_med_mT'][0]:.1f} -> {row['absdv_med_mT'][1]:.1f}) | "
                     f"{row['cc_contact'][0]:.3f} -> {row['cc_contact'][1]:.3f} | "
                     f"{row['zclip'][0]:.3f} -> {row['zclip'][1]:.3f} | "
                     f"{row['Rax_med'][0]:.3f} -> {row['Rax_med'][1]:.3f} |")
    head = ("| group | n | median ne mm | paired median d_ne [95% CI] | frac ne better | "
            "median Te mm | paired median d_Te [95% CI] | kfit dv median mT | "
            "R_in<0.20 | Zt clip | R_axis median |\n|---|---|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(lines), res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["subsets", "real", "twin_truth", "twin", "twin_table",
                                    "paired", "twin_report", "e2_report", "scalars", "r5"])
    ap.add_argument("--matched", default=None)
    ap.add_argument("--base", default="v2")
    ap.add_argument("--set", default="E1")
    ap.add_argument("--arms", default="v2,v2_c1")
    ap.add_argument("--scen", default="BP")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    if a.cmd == "subsets":
        make_subsets()
    elif a.cmd == "real":
        run_real(a.set, arms)
    elif a.cmd == "twin_truth":
        build_twin_truth(a.scen)
    elif a.cmd == "twin":
        run_twin(a.scen, arms)
    elif a.cmd == "twin_table":
        rows = twin_table(a.scen, arms)
        for k, m in rows.items():
            print(k, json.dumps({kk: round(float(v), 5) for kk, v in m.items()}))
    elif a.cmd == "r5":
        txt, rows = (r5_report(a.set, "zv_s05_p1_n", ZV_ARMS) if a.set == "E4"
                     else r5_report(a.set))
        print(txt)
        for k, m in rows.items():
            print(k, "d_ne by phase:", {ph: fmt_ci(t) for ph, t in m["d_ne_phase"].items()})
    elif a.cmd == "scalars":
        for arm in arms:
            real_scalars(a.set, arm)
    elif a.cmd == "e2_report":
        print(e2_report(a.set, arms, a.base)[0])
    elif a.cmd == "twin_report":
        print(twin_report(a.scen, arms, a.base, a.matched))
    elif a.cmd == "paired":
        for arm in arms:
            txt, res = paired_report(a.set, a.base, arm)
            (OUT / f"paired_{a.set}_{a.base}_vs_{arm}.json").write_text(json.dumps(res, indent=1))
            print(f"## {a.set}: {a.base} -> {arm}\n{txt}")
