"""Generator v7 (post-G0): HYPOTHESISED EFIT++-like profile family "efitpp" + v6' targeting.

Framing (root, deep research round 7): the original-MAST EFM p'/FF' family is NOT established; the
2-dof edge-zero / soft-zero-slope family below is documented for MAST-U only (no inheritance
statement).  Disclosure: "profile family hypothesised from the published MAST-U EFIT++ configuration
(Kogan et al. 2022); not verified for MAST".  Two flavours with everything else identical:
  v7a  pure efitpp;   v7h  hedge: each segment 50/50 efitpp or the twin two-shape family
       (beta R/R0 (1-x^a)^n + (1-beta) R0/R (1-x^a)^n, a ~ U(1,3), geometry control n in [0.5, 3]).
Passive flux: from the C1 sectional induction model driven by the measured coil/Ip history
(fec/passive.py, randomised resistance scale), as MAST EFIT used modelled induced currents
(McArdle & Taylor 2008) - neither free nor zero.  Kinetic (Thomson) conditioning of beta is kept
WEAK (not routine in MAST EFIT, Berkery 2021): two damped updates beta *= (p_target/p)^0.25.


Configuration source (deep research round 6, triage 20261010d row 1): Kogan et al., UKAEA-CCFE-CP2341
p. 5 - routine magnetics-only MAST-U EFIT++ uses polynomial p'(psiN) and FF'(psiN) with 2 degrees of
freedom each, both zero at psiN = 1, edge first derivatives softly (weight 0.1) pulled to zero.
Old-MAST production configuration is unverified (MAST-U != MAST); no MAST label and no DIII-D
coefficient is used anywhere.

Family "efitpp" (x = psiN):
    J(R, x) ~ beta (R/R0) g_p(x) + (1 - beta) (R0/R) g_f(x),
    g(x; s) = c0 (1 - x^2) + c1 x (1 - x) with c0 = 1, c1 = -2 - s, i.e. g = (1 - x)(1 - (1 + s) x),
    normalised to max 1 on [0, 1]; s = g'(1) is the edge slope (s = 0 -> (1 - x)^2, the soft EFIT++
    target; s = -2 -> 1 - x^2; s < -2 -> off-axis rise; s > 0 -> small negative edge source, clamped
    to J >= 0 by the solver).  s_p, s_f are drawn per segment from a mixture: w.p. 1/2 "near zero"
    N(0, 0.3), w.p. 1/2 "wide" U(-5, 0.5) (the effective strength of the 0.1 weight vs the magnetics
    chi^2 is unknown, so the slope is not pinned).
    The pressure split beta ~ U(0.1, 0.6) is weakly pulled toward target <p> = f_tot * <p_e>
    with f_tot ~ U(1, 3) per segment (total/electron), <p_e> the R-weighted chord mean of n_e T_e over
    real Thomson channels with T_e > 40 eV of the same frame (inputs only), via two damped updates
    beta *= (p_target/p)^0.25 (p' = K g_p, K = Ip beta / (R0 sum(shape) dA)).
v7 = efitpp + v6' geometry targeting: per segment target R_out = pooled real 40 eV R_out + delta
(delta marginalised exactly as v6', fec/amort_gen6.draw_delta); beta is first weakly
conditioned (2 damped updates at the mid control), then the FF' edge slope s_f (efitpp) or n (twin
family) is the geometry control, bisected on force-free-tracked solves (the height is
re-estimated by one secant step per bisection step; Z=0-held targets moved ~18 cm once the
force-free height was solved for efitpp frames) (6 bisection steps on Z = 0 solves; R_out is non-decreasing in s_f for 100% of frames in the
CPU scan); force-free truth; vessel, kappa <= 3 and target
tolerance bounds; Thomson v4 model with the segment's T_sep / lam_SOL (as v6').

  python -B fec/amort_gen7.py scan                 (CPU diag: R_out vs s_f monotonicity)
  python -B fec/gpu_slot.py run --owner amort -- python -B fec/amort_gen7.py gen --f F
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

OUT = WORK / "amort"
XG = np.linspace(0.0, 1.0, 201)
S_NEAR_SD, S_WIDE = 0.3, (-5.0, 0.5)
SF_RANGE = (-6.0, 1.0)            # geometry-control range for s_f (bisection)
FTOT = (1.0, 3.0)
TARGET_TOL = 0.04
N_SEC = 6
N_BETA = 2                 # pressure-split updates before the geometry bisection
BETA_DAMP = 0.25           # WEAK kinetic conditioning (root, round 7): partial updates only
BETA0 = (0.1, 0.6)         # efitpp beta prior before the weak update
TN_RANGE = (0.5, 3.0)      # v7h twin-family geometry control n (= n1 = n2)
FF_IT = 40                 # iterations of the short force-free tracking solves
N_FF = 6                   # force-free height secant steps (3 left |c| >= 1e-5 in 82% of efitpp frames, CPU diag)


def g_edge(x, s):
    """Unnormalised efitpp basis g(x; s) = (1 - x)(1 - (1 + s) x) (numpy or torch)."""
    return (1 - x) * (1 - (1 + s) * x)


def g_max(s):
    """max over [0, 1] of g(x; s) clamped at 0 (normalisation), vectorised over s."""
    s = np.atleast_1d(np.asarray(s, float))
    return np.maximum(g_edge(XG[None, :], s[:, None]), 0).max(1)


def draw_slope(rng, n):
    near = rng.random(n) < 0.5
    return np.where(near, rng.normal(0, S_NEAR_SD, n), rng.uniform(*S_WIDE, n))


def _v7_solver_cls():
    import torch
    from fec.gs import Solver

    class _S7(Solver):
        """Solver with the efitpp profile and (v7h) the twin two-shape family.
        prof (B,8) = beta, s_p, s_f, gmax_p, gmax_f, fam, a, n;  fam 0 = efitpp,
        fam 1 = twin family beta R/R0 (1-x^a)^n + (1-beta) R0/R (1-x^a)^n (in-family)."""

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
            beta = prof[:, 0:1]; sp = prof[:, 1:2]; sf = prof[:, 2:3]
            gp = prof[:, 3:4]; gf = prof[:, 4:5]
            fam = prof[:, 5:6]; ta = prof[:, 6:7]; tn = prof[:, 7:8]
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
                eff = (beta * Rs[None] / R0 * (g_edge(pn, sp) / gp).clamp_min(0) +
                       (1 - beta) * R0 / Rs[None] * (g_edge(pn, sf) / gf).clamp_min(0))
                tw = (1 - pn ** ta).clamp_min(0) ** tn
                twin = beta * Rs[None] / R0 * tw + (1 - beta) * R0 / Rs[None] * tw
                shape = torch.where(fam > 0.5, twin, eff) * ms
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
                        psi_ext=psi_ext, shape_tot=tot[:, 0])
    return _S7


def make_solver7(f, device="cuda"):
    from fec.amort_gen2 import R0
    from fec.amort_net import coil_matrix
    from fec.passive import PassiveModel
    spec, M = coil_matrix(f)
    RR, ZZ = np.meshgrid(spec.gR, spec.gZ)
    Mx = np.concatenate([M, PassiveModel().Gg, (RR ** 2 / 2).reshape(-1, 1),
                         (-RR ** 2 * ZZ / R0).reshape(-1, 1)], 1)
    return _v7_solver_cls()(spec, Mx, device=device), spec, M


def make_prof(beta, sp, sf, fam=None, a=None, n=None):
    B = len(beta)
    fam = np.zeros(B) if fam is None else fam
    a = np.full(B, 2.0) if a is None else a
    n = np.full(B, 1.0) if n is None else n
    return np.stack([beta, sp, sf, g_max(sp), g_max(sf), fam, a, n], 1)


# ============================================================================ pressure
def chord_pe(R, te, ne):
    """R-weighted chord mean of n_e T_e [Pa] over channels with T_e > 40 eV (NaN if < 5)."""
    e = 1.602176634e-19
    out = np.full(len(R), np.nan)
    for i in range(len(R)):
        ok = np.isfinite(R[i]) & np.isfinite(te[i]) & np.isfinite(ne[i]) & (te[i] > 40)
        if ok.sum() >= 5:
            out[i] = np.sum(ne[i][ok] * te[i][ok] * e * R[i][ok]) / np.sum(R[i][ok])
    return out


def truth_pressure(S, o, prof, Ip):
    """R-weighted volume mean of the truth's total pressure [Pa] from its p' term:
    p' = K g_p(x), K = Ip beta / (R0 sum(shape) dA);  p(x) = K (psi_ax - psi_b) int_x^1 g_p."""
    import torch
    B = len(Ip)
    psi = o["psi"].reshape(B, -1)[:, S.sidx]
    pa, pb = o["psi_ax"][:, None], o["psi_b"][:, None]
    x = ((pa - psi) / (pa - pb).clamp_min(1e-9)).clamp(0, 2)
    inside = (x < 1).to(psi.dtype)
    sp = torch.as_tensor(prof[:, 1:2], device=S.dev, dtype=S.dt)
    gp = torch.as_tensor(prof[:, 3:4], device=S.dev, dtype=S.dt)
    beta = torch.as_tensor(prof[:, 0:1], device=S.dev, dtype=S.dt)
    xx = x.clamp(0, 1)
    # int_x^1 (1-u)(1-(1+s)u) du  (closed form)
    a = 1 + sp
    F = lambda u: u - (1 + a) * u ** 2 / 2 + a * u ** 3 / 3
    Gint = (F(torch.ones_like(xx)) - F(xx)) / gp
    dA = S.dR * S.dZ
    K = torch.as_tensor(Ip, device=S.dev, dtype=S.dt)[:, None] * beta / \
        (S.spec.R0 * o["shape_tot"][:, None] * dA)
    p = K * (pa - pb) * Gint * inside
    Rw = S.Rs[None] * inside
    return ((p * Rw).sum(1) / Rw.sum(1).clamp_min(1e-12)).cpu().numpy()


# ============================================================================ generation
def gen_chunk7(S, spec, M, fr, cfg, pool, f, seed, n_real=2, n_screen=100, p_twin=0.0):
    import torch  # noqa: F401
    from fec.amort_gen import midplane_rin_rout
    from fec.amort_gen2 import PRIORS, solve_truth2
    from fec.amort_gen3 import KAPPA_MAX, lcfs_kappa
    from fec.amort_gen4 import synth_thomson4
    from fec.amort_gen6 import draw_delta, ext6
    from fec.twin_mast import smooth_track, truth_valid
    rng = np.random.default_rng(seed)
    B = len(fr["Ip"])
    sh = fr["shot"]
    us, inv = np.unique(sh, return_inverse=True)
    pri = PRIORS["TRAIN"]
    dv = smooth_track(rng, sh, -pri["dv_max"], pri["dv_max"], jitter=0.1)
    br = smooth_track(rng, sh, -pri["br_max"], pri["br_max"], jitter=0.1)
    r = np.exp(rng.uniform(np.log(pri["r_lo"]), np.log(pri["r_hi"]), len(us)))[inv]
    sp = draw_slope(rng, len(us))[inv]
    ftot = rng.uniform(*FTOT, len(us))[inv]
    fam = (rng.random(len(us)) < p_twin).astype(float)[inv]       # v7h: 50/50 per segment
    beta = np.where(fam > 0.5, smooth_track(rng, sh, 0.0, 0.7),
                    rng.uniform(*BETA0, len(us))[inv])
    ta = smooth_track(rng, sh, 1.0, 3.0)
    dl, tsep, lam = draw_delta(rng, len(us), cfg)
    target = (rng.choice(pool, len(us)) + dl)[inv]
    dl, tsep, lam = dl[inv], tsep[inv], lam[inv]
    pe = chord_pe(fr["R"], fr["te_real"], fr["ne_real"])
    p_tgt = np.where(fam > 0.5, np.nan, ftot * pe)               # efitpp only; NaN = no update
    Ix = ext6(fr, dv, br, r, f)
    st = None
    twin = fam > 0.5
    lo = np.where(twin, TN_RANGE[0], SF_RANGE[0])
    hi = np.where(twin, TN_RANGE[1], SF_RANGE[1])

    def prof_of(ctrl, beta):
        return make_prof(beta, sp, np.where(twin, -1.0, ctrl), fam, ta,
                         np.where(twin, ctrl, 1.0))

    zt = np.zeros(B)

    def solve(ctrl, beta, ff=False):
        """Z=0-held solve (ff=False) or a force-free-tracked one: two short solves bracket the
        vertical feedback c at the current height estimate, one secant step moves it, and a
        third solve at the new height gives R_out (the height estimate is carried over)."""
        nonlocal st, zt
        prof = prof_of(ctrl, beta)
        kw = dict(n_iter=n_screen) if st is None else dict(n_iter=n_screen // 2, I0=st["I"],
                                                            c0=st["c"])
        if not ff:
            o = S.solve_gen(Ix, fr["Ip"], prof, np.zeros(B), Rinit=np.full(B, 0.9), **kw)
        else:
            o1 = S.solve_gen(Ix, fr["Ip"], prof, zt, Rinit=np.full(B, 0.9), **kw)
            c1 = o1["c"].cpu().numpy()
            o2 = S.solve_gen(Ix, fr["Ip"], prof, zt + 0.02, n_iter=FF_IT, I0=o1["I"], c0=o1["c"])
            c2 = o2["c"].cpu().numpy()
            den = np.where(np.abs(c2 - c1) > 1e-12, c2 - c1, np.nan)
            zt = np.nan_to_num(np.clip(zt - c1 * 0.02 / den, -0.4, 0.4), nan=zt)
            o = S.solve_gen(Ix, fr["Ip"], prof, zt, n_iter=FF_IT, I0=o1["I"], c0=o1["c"])
        st = o
        _, Ro = midplane_rin_rout(o["psi"].cpu().numpy(), o["psi_ax"].cpu().numpy(),
                                  o["psi_b"].cpu().numpy(), spec.gR, spec.gZ)
        return np.where(np.isfinite(Ro), Ro, 2.2), truth_pressure(S, o, prof, fr["Ip"])
    t0 = time.time()
    # stage A (WEAK kinetic conditioning, efitpp frames only): beta *= (p_target/p)^0.25, twice
    mid0 = 0.5 * (lo + hi)
    for _ in range(N_BETA):
        _, pt = solve(mid0, beta)
        ok = np.isfinite(p_tgt) & (pt > 0)
        beta = np.where(ok, np.clip(beta * (p_tgt / np.where(ok, pt, 1.0)) ** BETA_DAMP, 0.02,
                                    0.9), beta)
    # stage B: geometry - bisection on the control of each frame (R_out non-decreasing in it)
    a, b = lo.copy(), hi.copy()
    best, berr = mid0.copy(), np.full(B, np.inf)
    for k in range(N_SEC):
        m = 0.5 * (a + b)
        y, _ = solve(m, beta, ff=True)
        e = np.abs(y - target)
        upd = e < berr
        best = np.where(upd, m, best); berr = np.where(upd, e, berr)
        a = np.where(y < target, m, a); b = np.where(y < target, b, m)
    prof = prof_of(best, beta)
    t1 = time.time()
    tr = solve_truth2(S, Ix, fr["Ip"], prof, n_sec=N_FF)
    valid_pre = truth_valid(tr)
    kappa = lcfs_kappa(tr["psi"], np.flatnonzero(valid_pre))
    Rin, Rout = midplane_rin_rout(tr["psi"], tr["psi_ax"], tr["psi_b"], spec.gR, spec.gZ)
    with np.errstate(invalid="ignore"):
        vessel = np.isfinite(Rout) & (Rout <= cfg["vessel"][1]) & (Rin >= cfg["vessel"][0])
        hit = np.abs(Rout - target) <= TARGET_TOL
    valid = valid_pre & ~(kappa > KAPPA_MAX) & vessel & hit
    hp = json.loads(Path(cfg["thomson_hp"]).read_text())["hp"]
    tes, nes = [], []
    for k in range(n_real):
        ts = synth_thomson4(fr, tr, spec.gR, spec.gZ, hp, seed=seed + 7919 * (k + 1),
                            tsep_frame=tsep, lamt_frame=lam)
        tes.append(ts["te"].astype(np.float32)); nes.append(ts["ne"].astype(np.float32))
    psi_coil = (np.asarray(fr["I"], float) @ M.T).reshape(B, 65, 65).astype(np.float32)
    out = dict(prof=prof, dv=dv, br=br, rscale=r, target=target, delta=dl, tsep=tsep,
               lam_sol=lam, ftot=ftot, pe_chord=pe, fam=fam, screen_err=berr, te=np.stack(tes),
               ne=np.stack(nes), valid=valid, valid_pre=valid_pre, kappa=kappa, vessel=vessel,
               hit=hit, realistic=valid, Rin=Rin, Rout=Rout, psi_coil=psi_coil, **tr)
    return out, (t1 - t0, time.time() - t1)


FLAVOURS = {"a": 0.0, "h": 0.5}     # v7a: pure efitpp; v7h: 50/50 efitpp / twin family


def gen(tag, f, every=2, n_lat=2, chunk=2000, seed=0, device="cuda", max_chunks=0, flavour="a"):
    from fec.amort_gen import take, train_frames
    from fec.amort_gen6 import CFG
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
    if max_chunks:
        groups = groups[:max_chunks]
    S, spec, M = make_solver7(f, device=device)
    print(f"gen7 {tag}: f={f:g}, {int(sum(cnt[g_].sum() for g_ in groups))} real frames x {n_lat},"
          f" {len(groups)} chunks", flush=True)
    t_all = time.time(); n_done = 0
    for ci in range(len(groups)):
        for lat in range(n_lat):
            fo = d / f"chunk_l{lat}_{ci:04d}.npz"
            if fo.exists():
                continue
            ix = np.flatnonzero(np.isin(fr_all["seg"], groups[ci]))
            fr = take(fr_all, ix)
            fr["shot_real"] = fr["shot"]; fr["shot"] = fr["seg"]
            sd = 7_000_019 * (seed + 1) + 10_007 * lat + 101 * ci + (0 if flavour == "a" else 3)
            out, tm = gen_chunk7(S, spec, M, fr, cfg, pool, f, sd, p_twin=FLAVOURS[flavour])
            np.savez(fo, g=fr["g"], shot=fr["shot_real"], seg=fr["seg"], fidx=fr["fidx"],
                     I=fr["I"], Ip=fr["Ip"], Itf=fr["Itf"], R=fr["R"].astype(np.float32),
                     lat=np.full(len(ix), lat), sol_f=f, **out)
            n_done += len(ix)
            v = out["valid"]
            print(f"lat {lat} chunk {ci}: B {len(ix)} control {tm[0]:.0f}s truth+ts {tm[1]:.0f}s |"
                  f" valid {v.mean():.3f} (gs {out['valid_pre'].mean():.3f} vessel "
                  f"{out['vessel'].mean():.3f} hit {out['hit'].mean():.3f}) | beta p10/50/90 "
                  f"{np.percentile(out['prof'][:, 0], [10, 50, 90]).round(2).tolist()} | "
                  f"{n_done} truths {time.time() - t_all:.0f}s", flush=True)
    print(f"GEN7_DONE {tag} truths {n_done} wall {time.time() - t_all:.0f}s", flush=True)


def build_twin7(f, seed=0, device="cuda", flavour="a"):
    """In-family twin truths: the v7 recipe on the twin's 1000 frames (fec_work/twin/truth_B.npz
    inputs, read-only) -> FEC_WORK/amort/twin2/truth_B7.npz (fec.twin_mast format)."""
    import fec.twin_mast as tm
    from fec.amort_gen6 import CFG
    cfg = json.loads(CFG.read_text())
    pool = np.load(cfg["r40_pool_file"])
    tm.OUT = WORK / "twin"
    fr, _, _, _, _ = tm.load_truth("B")
    fr = dict(fr); fr["shot_real"] = fr["shot"]
    S, spec, M = make_solver7(f, device=device)
    out, _ = gen_chunk7(S, spec, M, fr, cfg, pool, f, seed=77_777 + seed + (0 if flavour == "a" else 5),
                        n_real=1, p_twin=FLAVOURS[flavour])
    dst = "B7" if flavour == "a" else "B8"
    tr = {k: out[k] for k in ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI", "Zt", "xfound",
                              "psi_coil")}
    (OUT / "twin2").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "twin2" / f"truth_{dst}.npz", prof=out["prof"], dv=out["dv"], sol_fraction=f,
             **{f"fr_{k}": v for k, v in fr.items() if k != "shot_real"},
             **{f"tr_{k}": v for k, v in tr.items()}, ts_te=out["te"][0].astype(np.float64),
             ts_ne=out["ne"][0].astype(np.float64), x_valid7=out["valid"])
    print(f"twin {dst} at f={f:g}: GS-valid {out['valid_pre'].mean():.3f}, v7-valid "
          f"{out['valid'].mean():.3f}", flush=True)


def scan(device="cpu"):
    """CPU diag: R_out response to the FF' edge slope s_f at fixed beta = 0.3, s_p = 0."""
    from fec.amort_gen import midplane_rin_rout, take, train_frames
    from fec.amort_gen2 import PRIORS
    from fec.amort_gen6 import ext6
    from fec.twin_mast import smooth_track
    fr_all = train_frames()
    segs = np.unique(fr_all["seg"])[7::500][:5]
    fr = take(fr_all, np.flatnonzero(np.isin(fr_all["seg"], segs)))
    fr["shot_real"] = fr["shot"]; fr["shot"] = fr["seg"]
    S, spec, M = make_solver7(0.5, device=device)
    B = len(fr["Ip"]); rng = np.random.default_rng(3)
    dv = smooth_track(rng, fr["shot"], -0.01, 0.01, jitter=0.1)
    Ix = ext6(fr, dv, np.zeros(B), np.ones(B), 0.5)
    R = []
    for s in (-6, -4, -2, -1, 0, 1):
        o = S.solve_gen(Ix, fr["Ip"], make_prof(np.full(B, 0.3), np.zeros(B), np.full(B, float(s))),
                        np.zeros(B), n_iter=150, Rinit=np.full(B, 0.9))
        _, Ro = midplane_rin_rout(o["psi"].cpu().numpy(), o["psi_ax"].cpu().numpy(),
                                  o["psi_b"].cpu().numpy(), spec.gR, spec.gZ)
        R.append(Ro)
        pt = truth_pressure(S, o, make_prof(np.full(B, 0.3), np.zeros(B), np.full(B, float(s))),
                            fr["Ip"])
        print(f"s_f {s:+d}: R_out median {np.nanmedian(Ro):.3f} p10/90 "
              f"{np.nanpercentile(Ro, [10, 90]).round(3).tolist()} frac>1.55 "
              f"{np.mean(~(Ro <= 1.55)):.2f} | <p> median {np.nanmedian(pt):.0f} Pa", flush=True)
    R = np.where(np.isfinite(np.array(R)), np.array(R), 2.5)
    d = np.diff(R, axis=0)
    print(f"monotone (non-increasing) frac {np.mean(np.all(d <= 0.01, 0)):.2f}, (non-decreasing) "
          f"{np.mean(np.all(d >= -0.01, 0)):.2f}; per-frame range median "
          f"{np.median(R.max(0) - R.min(0)) * 100:.1f} cm", flush=True)
    pe = chord_pe(fr["R"], fr["te_real"], fr["ne_real"])
    print(f"real chord <p_e> median {np.nanmedian(pe):.0f} Pa (finite {np.isfinite(pe).mean():.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "gen", "twin"])
    ap.add_argument("--f", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flavour", default="a", choices=["a", "h"])
    a = ap.parse_args()
    from fec.amort_gen6 import set_f
    f = set_f(a.f)
    if a.cmd == "scan":
        scan("cpu")
    elif a.cmd == "twin":
        build_twin7(f, device=a.device, flavour=a.flavour)
    else:
        gen(a.tag or f"v7{a.flavour}_f{f:g}", f, max_chunks=a.max_chunks, device=a.device,
            flavour=a.flavour)


if __name__ == "__main__":
    main()
