"""Generator v6' (post-G0): geometry-targeted, population-level truth prior (root-approved 2026-10-10).

Every truth is our own GS solution on REAL mast_public_test inputs (coils, Ip, history, Thomson
radii/missing pattern).  No label, demo parquet or envelope mask is read.  Differences from v3/v4f:
  * NO per-frame screening.  Per segment a target outboard LCFS radius is drawn from the POOLED
    real Thomson 40 eV outboard-crossing distribution of the non-twin frames (never the frame's
    own value) plus an offset delta, and one profile-breadth latent (twin family: n1 = n2;
    NSTX-U-like polynomial family: beta0) is solved by secant on Z = 0 solves to hit it; then the
    force-free truth is solved.
  * delta is marginalised and tied to the synthetic separatrix temperature:
        T_sep ~ logU(20, 100) eV (round-5 physics: "tens to ~100 eV", non-MAST),
        delta = -lam_SOL ln(T_sep/40)  if T_sep >= 40 eV  (40 eV point in the SOL; lam_SOL ~ U(1, 3) cm),
        delta = +L_in  ln(40/T_sep)    if T_sep <  40 eV  (40 eV point inside; L_in ~ real local
                Te gradient length at 20-80 eV on non-twin inputs, clipped to its p10-p50),
    and the synthetic Thomson uses the same T_sep and SOL decay length (consistency).
  * Feasibility bounds only: vessel (R_in >= 0.20, R_out <= 1.55 m), kappa <= 3, and the secant
    must reach its target within TARGET_TOL (a property of the solve, not of the frame's data).
  * Solenoid turn fraction f is a parameter (env FEC_AMORT_SOLF / --f): coil matrix solenoid
    column scaled from f = 0.5; C1 passive currents recomputed at f (FEC_WORK/amort/passive_f<f>).
  * Thomson: fec.amort_gen4.synth_thomson4 with the frozen v4 profile hyper-ranges
    (FEC_WORK/amort/v4_hp.json) and the T_sep / lam_SOL above; v2 systematics.

  python -B fec/amort_gen6.py freeze                  (targets + delta prior -> v6_config.json)
  python -B fec/amort_gen6.py passive --f F           (only if F != 0.5; CPU)
  python -B fec/gpu_slot.py run --owner amort -- python -B fec/amort_gen6.py gen --f F
  python -B fec/gpu_slot.py run --owner amort -- python -B fec/amort_gen6.py twin --f F   (F != 0.5)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

OUT = WORK / "amort"
CFG = OUT / "v6_config.json"
PASS_CACHE = WORK / "passive" / "frames" / "mast_public_test"
EDGES = WORK / "mastedge" / "mast_edges.npz"
TARGET_TOL = 0.04         # m, the solve must land within this of the drawn target
N_SEC = 6                 # bisection steps on Z=0 solves (R_out is monotone in the control: CPU
                          # diag 2026-10-10, 99% of frames; 3 secant steps hit only 9%)
CTRL = {0: ("n1", 0.5, 3.0, 0.4), 1: ("b0", 0.0, 0.7, 0.15)}   # family -> latent, lo, hi, step


def set_f(f):
    os.environ["FEC_AMORT_SOLF"] = f"{float(f):g}"
    return float(f)


def pass_dir(f):
    return PASS_CACHE if abs(f - 0.5) < 1e-12 else OUT / f"passive_f{f:g}"


# ============================================================================ frozen config
def local_LT(R, te):
    """Outboard local Te gradient length [m] at 20-80 eV beyond the Te peak (NaN if not found)."""
    from scipy.ndimage import median_filter
    out = np.full(len(R), np.nan)
    for i in range(len(R)):
        ok = np.isfinite(R[i]) & np.isfinite(te[i]) & (R[i] > 0.2)
        if ok.sum() < 40:
            continue
        o = np.argsort(R[i][ok]); rr = R[i][ok][o]
        tt = median_filter(te[i][ok][o], 5, mode="nearest")
        ip = int(np.argmax(tt))
        if tt[ip] < 150:
            continue
        m = (np.arange(len(rr)) > ip) & (tt >= 20) & (tt <= 80)
        if m.sum() < 3:
            continue
        s = np.polyfit(rr[m], np.log(tt[m]), 1)[0]
        if s < 0:
            out[i] = -1.0 / s
    return out


def freeze(n_lt=8000, seed=0):
    """Pooled real 40 eV outboard crossings (non-twin frames) + L_in bounds -> v6_config.json."""
    from fec.amort_gen import train_frames
    fr = train_frames()
    E = np.load(EDGES)
    g = fr["g"]
    ok = E["st_out"][g] == 1
    pool = np.asarray(E["R_out"][g][ok], float)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(g), n_lt, replace=False)
    L = local_LT(fr["R"][i], fr["te_real"][i])
    L = L[np.isfinite(L)]
    cfg = dict(r40_pool_file=str(OUT / "v6_r40_pool.npy"), n_pool=int(len(pool)),
               r40_pct=np.percentile(pool, [5, 25, 50, 75, 95]).round(4).tolist(),
               tsep_eV=[20.0, 100.0], lam_sol_m=[0.01, 0.03],
               L_in_m=np.percentile(L, [10, 50]).round(4).tolist(), n_LT=int(len(L)),
               L_pct_m=np.percentile(L, [5, 10, 25, 50, 75, 90]).round(4).tolist(),
               target_tol_m=TARGET_TOL, n_sec=N_SEC, ctrl=CTRL, vessel=[0.20, 1.55], kappa_max=3.0,
               thomson_hp=str(OUT / "v4_hp.json"))
    np.save(OUT / "v6_r40_pool.npy", pool)
    d = draw_delta(np.random.default_rng(1), 200_000, cfg)[0]
    cfg["delta_pct_cm"] = (np.percentile(d, [1, 5, 25, 50, 75, 95, 99]) * 100).round(2).tolist()
    CFG.write_text(json.dumps(cfg, indent=1))
    print("V6_FROZEN", json.dumps({k: cfg[k] for k in ("r40_pct", "L_in_m", "delta_pct_cm")}),
          flush=True)
    return cfg


def draw_delta(rng, n, cfg):
    """(delta [m], T_sep [eV], lam_SOL [m]) per draw, see module doc."""
    t0, t1 = cfg["tsep_eV"]
    tsep = np.exp(rng.uniform(np.log(t0), np.log(t1), n))
    lam = rng.uniform(*cfg["lam_sol_m"], n)
    Lin = rng.uniform(*cfg["L_in_m"], n)
    delta = np.where(tsep >= 40.0, -lam * np.log(tsep / 40.0), Lin * np.log(40.0 / tsep))
    return delta, tsep, lam


# ============================================================================ machine at f
def passive_frames(f, rscales=("0.5", "1", "2")):
    """C1 per-frame passive currents at solenoid fraction f (solenoid drive column scaled from
    the f = 0.5 model; same integration as fec.passive.run_frames)."""
    from fec.machines import MAST_COILS
    from fec.passive import PassiveModel, c1_shot, frame_index
    model = PassiveModel()
    isol = MAST_COILS.index("magnetics_sol_current")
    model.Mpc = model.Mpc.copy()
    model.Mpc[:, isol] *= f / 0.5
    per, N, ds = frame_index("mast_public_test")
    sd = WORK / "passive" / "sig" / "mast_public_test"
    t0 = time.time()
    for rs in rscales:
        od = pass_dir(f) / f"c1_r{rs}"
        if (od / "I.npy").exists():
            continue
        od.mkdir(parents=True, exist_ok=True)
        I_out = np.lib.format.open_memmap(od / "I.part.npy", "w+", np.float32,
                                          (N, len(model.piece)))
        modes = model.modes(float(rs))
        for k, (stem, frl) in enumerate(per.items()):
            z = np.load(sd / f"{stem}.npz")
            t, I = c1_shot(model, z, float(rs), modes)
            gg = np.array([a for a, _ in frl]); tf = np.array([b for _, b in frl])
            for j in range(I.shape[1]):
                I_out[gg, j] = np.interp(tf, t, I[:, j])
        I_out.flush(); del I_out
        (od / "I.part.npy").replace(od / "I.npy")
        print(f"passive f={f:g} r={rs} ({time.time() - t0:.0f}s)", flush=True)


def passive_currents6(g, r, f):
    d = pass_dir(f)
    Im, I0, Ip_ = [np.asarray(np.load(d / f"c1_r{s}" / "I.npy", mmap_mode="r")[g], np.float64)
                   for s in ("0.5", "1", "2")]
    x = np.log2(np.asarray(r, float))[:, None]
    return I0 + x * (Ip_ - Im) / 2 + x ** 2 * (Ip_ + Im - 2 * I0) / 2


def make_solver6(f, device="cuda"):
    from fec.amort_gen2 import R0
    from fec.amort_gen3 import _v3_solver_cls
    from fec.amort_net import coil_matrix
    from fec.passive import PassiveModel
    spec, M = coil_matrix(f)
    Gg = PassiveModel().Gg
    RR, ZZ = np.meshgrid(spec.gR, spec.gZ)
    Mx = np.concatenate([M, Gg, (RR ** 2 / 2).reshape(-1, 1),
                         (-RR ** 2 * ZZ / R0).reshape(-1, 1)], 1)
    return _v3_solver_cls()(spec, Mx, device=device), spec, M


def ext6(fr, dv, br, r, f):
    return np.concatenate([fr["I"], passive_currents6(fr["g"], r, f), dv[:, None], br[:, None]], 1)


# ============================================================================ truths
def secant_step(x0, y0, x1, y1, target, lo, hi):
    """One vectorised secant update toward target, clipped; falls back to a fixed step."""
    den = y1 - y0
    with np.errstate(invalid="ignore", divide="ignore"):
        x2 = x1 + (target - y1) * (x1 - x0) / den
    bad = ~np.isfinite(x2) | (np.abs(den) < 1e-4)
    x2 = np.where(bad, x1 + np.sign(np.nan_to_num(target - y1)) * 0.1 * (hi - lo), x2)
    return np.clip(x2, lo, hi)


def gen_chunk6(S, spec, M, fr, cfg, pool, f, seed, n_real=2, n_screen=100):
    import torch
    from fec.amort_gen import midplane_rin_rout
    from fec.amort_gen2 import PRIORS
    from fec.amort_gen3 import KAPPA_MAX, lcfs_kappa, latent_draw3
    from fec.amort_gen4 import synth_thomson4
    from fec.amort_gen2 import solve_truth2
    from fec.twin_mast import truth_valid
    rng = np.random.default_rng(seed)
    B = len(fr["Ip"])
    prof, dv, br, r = latent_draw3(fr, PRIORS["TRAIN"], seed)
    us, inv = np.unique(fr["shot"], return_inverse=True)
    dl, tsep, lam = draw_delta(rng, len(us), cfg)
    target = rng.choice(pool, len(us)) + dl
    target, dl, tsep, lam = target[inv], dl[inv], tsep[inv], lam[inv]
    fam = (prof[:, 0] > 0.5).astype(int)
    col = np.where(fam == 0, 3, 1)                          # n1 (twin) or b0 (poly) column
    lo = np.where(fam == 0, CTRL[0][1], CTRL[1][1]); hi = np.where(fam == 0, CTRL[0][2], CTRL[1][2])
    stp = np.where(fam == 0, CTRL[0][3], CTRL[1][3])
    rr = np.arange(B)

    def setc(p, x):
        p = p.copy(); p[rr, col] = x
        p[:, 5] = np.where(fam == 0, p[:, 3], p[:, 5])      # n2 = n1 in the twin family
        return p

    Ix = ext6(fr, dv, br, r, f)
    st = None

    def rout(x):
        nonlocal st
        kw = dict(n_iter=n_screen) if st is None else dict(n_iter=n_screen // 2, I0=st["I"],
                                                            c0=st["c"])
        o = S.solve_gen(Ix, fr["Ip"], setc(prof, x), np.zeros(B), Rinit=np.full(B, 0.9), **kw)
        st = o
        _, Ro = midplane_rin_rout(o["psi"].cpu().numpy(), o["psi_ax"].cpu().numpy(),
                                  o["psi_b"].cpu().numpy(), spec.gR, spec.gZ)
        return np.where(np.isfinite(Ro), Ro, 2.2)
    t0 = time.time()
    # bisection on the breadth control (R_out increases with n1 / b0), keep the best evaluated
    a, b = lo.copy(), hi.copy()
    best, berr = prof[rr, col].copy(), np.full(B, np.inf)
    for _ in range(N_SEC):
        m = 0.5 * (a + b)
        y = rout(m)
        e = np.abs(y - target)
        upd = e < berr
        best = np.where(upd, m, best); berr = np.where(upd, e, berr)
        a = np.where(y < target, m, a); b = np.where(y < target, b, m)
    prof = setc(prof, best)
    t1 = time.time()
    tr = solve_truth2(S, Ix, fr["Ip"], prof)
    valid_pre = truth_valid(tr)
    kappa = lcfs_kappa(tr["psi"], np.flatnonzero(valid_pre))
    Rin, Rout = midplane_rin_rout(tr["psi"], tr["psi_ax"], tr["psi_b"], spec.gR, spec.gZ)
    with np.errstate(invalid="ignore"):
        vessel = np.isfinite(Rout) & (Rout <= cfg["vessel"][1]) & (Rin >= cfg["vessel"][0])
        hit = np.abs(Rout - target) <= TARGET_TOL
    valid = valid_pre & ~(kappa > KAPPA_MAX) & vessel & hit
    tes, nes = [], []
    for k in range(n_real):
        ts = synth_thomson4(fr, tr, spec.gR, spec.gZ, json.loads(Path(cfg["thomson_hp"]).read_text())["hp"],
                            seed=seed + 7919 * (k + 1), tsep_frame=tsep, lamt_frame=lam)
        tes.append(ts["te"].astype(np.float32)); nes.append(ts["ne"].astype(np.float32))
    psi_coil = (np.asarray(fr["I"], float) @ M.T).reshape(B, 65, 65).astype(np.float32)
    out = dict(prof=prof, dv=dv, br=br, rscale=r, target=target, delta=dl, tsep=tsep, lam_sol=lam,
               screen_err=berr,
               te=np.stack(tes), ne=np.stack(nes), valid=valid, valid_pre=valid_pre, kappa=kappa,
               vessel=vessel, hit=hit, realistic=valid, Rin=Rin, Rout=Rout, psi_coil=psi_coil, **tr)
    return out, (t1 - t0, time.time() - t1)


def gen(tag, f, every=2, n_lat=2, chunk=2000, seed=0, device="cuda"):
    from fec.amort_gen import take, train_frames
    cfg = json.loads(CFG.read_text())
    pool = np.load(cfg["r40_pool_file"])
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
    S, spec, M = make_solver6(f, device=device)
    print(f"gen6 {tag}: f={f:g}, {len(fr_all['g'])} real frames x {n_lat}, {len(groups)} chunks",
          flush=True)
    t_all = time.time(); n_done = 0
    for ci in range(len(groups)):
        for lat in range(n_lat):
            fo = d / f"chunk_l{lat}_{ci:04d}.npz"
            if fo.exists():
                continue
            ix = np.flatnonzero(np.isin(fr_all["seg"], groups[ci]))
            fr = take(fr_all, ix)
            fr["shot_real"] = fr["shot"]; fr["shot"] = fr["seg"]
            sd = 6_000_011 * (seed + 1) + 10_007 * lat + 101 * ci
            out, tm = gen_chunk6(S, spec, M, fr, cfg, pool, f, sd)
            np.savez(fo, g=fr["g"], shot=fr["shot_real"], seg=fr["seg"], fidx=fr["fidx"],
                     I=fr["I"], Ip=fr["Ip"], Itf=fr["Itf"], R=fr["R"].astype(np.float32),
                     lat=np.full(len(ix), lat), sol_f=f, **out)
            n_done += len(ix)
            v = out["valid"]
            print(f"lat {lat} chunk {ci}: B {len(ix)} secant {tm[0]:.0f}s truth+ts {tm[1]:.0f}s | "
                  f"valid {v.mean():.3f} (gs {out['valid_pre'].mean():.3f} vessel "
                  f"{out['vessel'].mean():.3f} hit {out['hit'].mean():.3f}) | R_out p10/50/90 "
                  f"{np.nanpercentile(out['Rout'][v], [10, 50, 90]).round(3).tolist() if v.any() else []}"
                  f" | {n_done} truths {time.time() - t_all:.0f}s", flush=True)
    print(f"GEN6_DONE {tag} truths {n_done} wall {time.time() - t_all:.0f}s", flush=True)


def build_twin_f(f, seed=0):
    """Deployment-matched twin sets at f != 0.5: twin2 priors B2/C2 (passive, b_r, dv, Thomson v2)
    solved at f on the twin's 1000 frames, then Thomson v4 -> twin2/truth_{B5,C5}.npz."""
    import fec.amort_gen2 as G2
    from fec.amort_gen4 import synth_thomson4
    import fec.twin_mast as tm
    G2.PASS_DIR = pass_dir(f)                       # amort_gen2.passive_currents reads this
    tm.OUT = WORK / "twin"
    fr, _, _, _, _ = tm.load_truth("B")
    fr = dict(fr); fr["shot_real"] = fr["shot"]
    from fec.amort_gen2 import R0
    from fec.amort_net import coil_matrix
    from fec.passive import PassiveModel
    from fec.twin_mast import TwinSolver
    spec, M = coil_matrix(f)
    RR, ZZ = np.meshgrid(spec.gR, spec.gZ)
    S = TwinSolver(spec, np.concatenate([M, PassiveModel().Gg, (RR ** 2 / 2).reshape(-1, 1),
                                         (-RR ** 2 * ZZ / R0).reshape(-1, 1)], 1))
    hp = json.loads((OUT / "v4_hp.json").read_text())["hp"]
    for src, dst in (("B2", "B5"), ("C2", "C5")):
        prior = G2.PRIORS[src]
        out, _ = G2.gen_chunk(S, spec, M, fr, prior, 4, 100, 1, 777 + 1000 * (ord(src[0]) - 64) + seed)
        tr = {k: out[k] for k in ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI", "Zt",
                                  "xfound", "psi_coil")}
        tr["track"] = out["track"]; tr["te_peak_real"] = out["te_peak_real"]
        ts = synth_thomson4(fr, tr, spec.gR, spec.gZ, hp, prior=prior,
                            seed=4_444 + 7 * seed + ord(src[0]))
        np.savez(OUT / "twin2" / f"truth_{dst}.npz", prof=out["prof"], dv=out["dv"], sol_fraction=f,
                 **{f"fr_{k}": v for k, v in fr.items() if k != "shot_real"},
                 **{f"tr_{k}": v for k, v in tr.items()}, ts_te=ts["te"], ts_ne=ts["ne"],
                 x_br=out["br"], x_rscale=out["rscale"])
        print(f"twin {dst} at f={f:g}: valid {out['valid'].mean():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "passive", "gen", "twin"])
    ap.add_argument("--f", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--n-lat", type=int, default=2)
    a = ap.parse_args()
    f = set_f(a.f)
    if a.cmd == "freeze":
        freeze()
    elif a.cmd == "passive":
        passive_frames(f)
    elif a.cmd == "gen":
        gen(a.tag or f"v6_f{f:g}", f, a.every, a.n_lat)
    else:
        build_twin_f(f)


if __name__ == "__main__":
    main()
