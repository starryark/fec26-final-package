"""Generator v3 (post-G0): generator v2 + an NSTX-U-like profile component in the J family.

Deep-research round 4 (docs/research/DEEP_RESEARCH_TRIAGE_20261010b.md row 2): NSTX-U EFIT01
uses 1 p' and 3 FF' polynomial basis functions.  v3 draws, per segment, the current-profile
family from a 50/50 mixture:
  fam 0  twin family (as v1/v2 TRAIN): J ~ b0 R/R0 (1-x^a1)^n1 + (1-b0) R0/R (1-x^a1)^n1
  fam 1  NSTX-U-like polynomial family (edge-zero EFIT polynomial bases, x = psiN):
         J ~ b0 R/R0 P(x) + (1-b0) R0/R F(x),
         P(x) = 1 - x                                   (1 p' coefficient)
         F(x) = c0 (1-x^3) + c1 (x-x^3) + c2 (x^2-x^3)  (3 FF' coefficients),
         c0 ~ U(0.05, 1), c1 ~ U(-1, 2), c2 ~ U(-2, 2) as smooth segment tracks, scaled so
         that max F = 1 on [0, 1]; where F < 0 somewhere, (c1, c2) are shrunk toward 0 until
         F >= 0 (c0 alone is non-negative), so hollow but non-negative FF' shapes are allowed.
The coefficient ranges are a-priori choices; nothing is fitted or reweighted toward MAST,
the scorer or the published ST envelopes (fec/amort_cover.py only REPORTS coverage).
Feasibility filter (root 2026-10-10): valid truths with scorer LCFS kappa > 3.0 are dropped
(KAPPA_MAX; non-MAST bound), recorded per chunk as valid_pre / kappa / kdrop.
Everything else (latent draws per frame, dv, b_r, passive C1 with random resistance,
Thomson v2 realism, Te-peak screening, force-free height) is fec/amort_gen2.py unchanged.

  python -B fec/gpu_slot.py run --owner amort -- python -B fec/amort_gen3.py gen --tag v3main --every 2 --n-lat 2
  python -B fec/amort_gen3.py probe --n 600 --device cpu       (coverage probe, CPU)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.amort_gen2 import (OUT, PRIORS, ext_currents, passive_currents, synth_thomson2,  # noqa
                            solve_truth2)

P_POLY = 0.5
XG = np.linspace(0.0, 1.0, 101)
# Physical-feasibility bound (root 2026-10-10): drop synthetic truths whose LCFS elongation
# exceeds 3.0.  "Feasibility bound: no ST plasma on record exceeds kappa ~ 3; non-MAST" (the
# most strongly shaped ST plasma on record is a transient NSTX one, round-4 triage row 5).
# A hard physical cut only: nothing inside kappa 1.4-3.0 is reweighted.
KAPPA_MAX = 3.0


def _kappa_one(args):
    """Scorer LCFS elongation (Zmax-Zmin)/(Rmax-Rmin) of one map (twin_mast._scal_one call)."""
    from fec.paths import SCORING
    if str(SCORING) not in sys.path:
        sys.path.insert(0, str(SCORING))
    from lcfs import extract_lcfs  # noqa
    psi, gR, gZ, box = args
    try:
        C = np.asarray(extract_lcfs(psi, gR, gZ, "MAST", box, box.astype(float), axis_sign=1.0))
        return float((C[:, 1].max() - C[:, 1].min()) / (C[:, 0].max() - C[:, 0].min()))
    except Exception:
        return np.nan


def lcfs_kappa(psi, idx, nproc=4):
    """kappa for maps psi[idx] (NaN elsewhere or where the scorer finds no LCFS)."""
    from multiprocessing import Pool
    from fec.machines import mast_setup
    from fec.twin_mast import _box
    spec, _ = mast_setup()
    box = _box()
    out = np.full(len(psi), np.nan)
    if len(idx):
        with Pool(nproc) as pool:
            out[idx] = pool.map(_kappa_one, [(np.asarray(psi[i], np.float64), spec.gR, spec.gZ,
                                              box) for i in idx], chunksize=16)
    return out


def _v3_solver_cls():
    import torch
    from fec.gs import Solver

    class _S3(Solver):
        """Solver with the v3 mixture profile.  prof (B,9) = fam, b0, a1, n1, a2, n2, c0, c1, c2
        (fam 0 = twin two-shape family, exactly fec.twin_mast's solve_gen; fam 1 = poly)."""

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
            fam = prof[:, 0:1]; b0 = prof[:, 1:2]; a1 = prof[:, 2:3]; n1 = prof[:, 3:4]
            a2 = prof[:, 4:5]; n2 = prof[:, 5:6]
            q0 = prof[:, 6:7]; q1 = prof[:, 7:8]; q2 = prof[:, 8:9]
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
                twin = (b0 * Rs[None] / R0 * (1 - pn ** a1).clamp_min(0) ** n1 +
                        (1 - b0) * R0 / Rs[None] * (1 - pn ** a2).clamp_min(0) ** n2)
                x3 = pn ** 3
                Fp = (q0 * (1 - x3) + q1 * (pn - x3) + q2 * (pn ** 2 - x3)).clamp_min(0)
                poly = b0 * Rs[None] / R0 * (1 - pn) + (1 - b0) * R0 / Rs[None] * Fp
                shape = torch.where(fam > 0.5, poly, twin) * ms
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
    return _S3


def make_solver3(device="cuda"):
    """v3 solver with v2's external columns [coils, passive pieces, dv, b_r]."""
    from fec.amort_gen2 import R0
    from fec.machines import mast_setup
    from fec.passive import PassiveModel
    spec, M = mast_setup()
    pm = PassiveModel()
    RR, ZZ = np.meshgrid(spec.gR, spec.gZ)
    Mx = np.concatenate([M, pm.Gg, (RR ** 2 / 2).reshape(-1, 1),
                         (-RR ** 2 * ZZ / R0).reshape(-1, 1)], 1)
    return _v3_solver_cls()(spec, Mx, device=device), spec, M, pm


def poly_F(c, x=XG):
    """F(x) for coefficient rows c (B,3) on x -> (B, len(x))."""
    x3 = x ** 3
    return c[:, :1] * (1 - x3) + c[:, 1:2] * (x - x3) + c[:, 2:3] * (x ** 2 - x3)


def poly_coeffs(c):
    """Non-negative, max-1 normalised FF' coefficients: shrink (c1, c2) where F < 0."""
    c = np.array(c, float)
    lo, hi = np.zeros(len(c)), np.ones(len(c))
    ok = poly_F(c).min(1) >= 0
    for _ in range(30):                       # bisection on the shrink factor
        mid = 0.5 * (lo + hi)
        cm = np.concatenate([c[:, :1], c[:, 1:] * mid[:, None]], 1)
        good = poly_F(cm).min(1) >= 0
        lo = np.where(good, mid, lo); hi = np.where(good, hi, mid)
    lam = np.where(ok, 1.0, lo)
    c = np.concatenate([c[:, :1], c[:, 1:] * lam[:, None]], 1)
    return c / poly_F(c).max(1, keepdims=True)


def latent_draw3(fr, prior, seed):
    """v2 latents (J shape, dv, b_r, r) + per-segment family and NSTX-U-like FF' tracks."""
    from fec.amort_gen2 import latent_draw
    from fec.twin_mast import smooth_track
    prof5, dv, br, r = latent_draw(fr, prior, seed)
    rng = np.random.default_rng(seed + 500_009)
    sh = fr["shot"]
    useg, inv = np.unique(sh, return_inverse=True)
    fam = (rng.random(len(useg)) < P_POLY).astype(float)[inv]
    cc = np.stack([smooth_track(rng, sh, 0.05, 1.0), smooth_track(rng, sh, -1.0, 2.0),
                   smooth_track(rng, sh, -2.0, 2.0)], 1)
    prof = np.concatenate([fam[:, None], prof5, poly_coeffs(cc)], 1)
    return prof, dv, br, r


def screen_select3(S, fr, prior, K, n_screen, seed):
    """fec.amort_gen2.screen_select2 with the v3 latents."""
    from fec.twin_mast import real_te_peak
    B = len(fr["Ip"])
    pk = real_te_peak(fr)
    cands = []
    for k in range(K):
        prof, dv, br, r = latent_draw3(fr, prior, seed + k)
        o = S.solve_gen(ext_currents(fr, dv, br, r), fr["Ip"], prof, np.zeros(B),
                        n_iter=n_screen, Rinit=np.full(B, 0.9))
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


def gen_chunk3(S, spec, M, fr, prior, K, n_screen, n_real, seed):
    """fec.amort_gen2.gen_chunk with the v3 latents (same output keys; prof is (B,9))."""
    import torch
    from fec.amort_gen import midplane_rin_rout
    from fec.twin_mast import truth_valid
    B = len(fr["Ip"])
    if S.dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    prof, dv, br, r, sel, pk = screen_select3(S, fr, prior, K, n_screen, seed)
    t1 = time.time()
    tr = solve_truth2(S, ext_currents(fr, dv, br, r), fr["Ip"], prof)
    t2 = time.time()
    valid_pre = truth_valid(tr)
    kappa = lcfs_kappa(tr["psi"], np.flatnonzero(valid_pre))
    kdrop = valid_pre & (kappa > KAPPA_MAX)
    valid = valid_pre & ~kdrop
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
               valid=valid, valid_pre=valid_pre, kappa=kappa, kdrop=kdrop, realistic=real,
               Rin=Rin, Rout=Rout, psi_coil=psi_coil, **tr)
    return out, (t1 - t0, t2 - t1, time.time() - t2)


def gen(tag, every=2, n_lat=2, chunk=2000, K=4, n_screen=100, n_real=2, seed=0,
        device="cuda", max_frames=0):
    """As fec.amort_gen2.gen (ci-major latent order) with the v3 profile mixture."""
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
    if max_frames:
        segs = segs[:int(np.searchsorted(np.cumsum(cnt[segs]), max_frames)) + 1]
    groups, cur, n = [], [], 0
    for s in segs:
        if n + cnt[s] > chunk and cur:
            groups.append(cur); cur, n = [], 0
        cur.append(s); n += cnt[s]
    if cur:
        groups.append(cur)
    jobs = [(lat, ci) for ci in range(len(groups)) for lat in range(n_lat)]
    print(f"gen3 {tag}: {int(sum(cnt[g].sum() for g in groups))} real frames x {n_lat} latents,"
          f" {len(groups)} chunks per latent, K={K}, n_real={n_real}, P_poly={P_POLY}",
          flush=True)
    S, spec, M, pm = make_solver3(device=device)
    t_all = time.time(); n_done = 0
    for lat, ci in jobs:
        f = d / f"chunk_l{lat}_{ci:04d}.npz"
        if f.exists():
            continue
        ix = np.flatnonzero(np.isin(fr_all["seg"], groups[ci]))
        fr = take(fr_all, ix)
        fr["shot_real"] = fr["shot"]
        fr["shot"] = fr["seg"]
        sd = 3_000_017 * (seed + 1) + 10_007 * lat + 101 * ci
        out, tm = gen_chunk3(S, spec, M, fr, PRIORS["TRAIN"], K, n_screen, n_real, sd)
        np.savez(f, g=fr["g"], shot=fr["shot_real"], seg=fr["seg"], fidx=fr["fidx"],
                 I=fr["I"], Ip=fr["Ip"], Itf=fr["Itf"], R=fr["R"].astype(np.float32),
                 lat=np.full(len(ix), lat), **out)
        n_done += len(ix)
        fam = out["prof"][:, 0] > 0.5
        vp, kd = out["valid_pre"], out["kdrop"]
        dr = lambda m: kd[m].sum() / max(vp[m].sum(), 1)
        print(f"lat {lat} chunk {ci}: B {len(ix)} screen {tm[0]:.0f}s truth {tm[1]:.0f}s | "
              f"valid {out['valid'].mean():.3f} (poly {out['valid'][fam].mean():.3f}, twin "
              f"{out['valid'][~fam].mean():.3f}) kappa>3 dropped poly {dr(fam):.3f} twin "
              f"{dr(~fam):.3f} realistic {out['realistic'].mean():.3f} | "
              f"poly share {fam.mean():.2f} | total {n_done} truths {time.time() - t_all:.0f}s",
              flush=True)
    print(f"GEN3_DONE {tag} truths {n_done} wall {time.time() - t_all:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "probe"])
    ap.add_argument("--tag", default="v3main")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--n-lat", type=int, default=2)
    ap.add_argument("--n-real", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd == "gen":
        gen(a.tag, a.every, a.n_lat, a.chunk, a.K, n_real=a.n_real, seed=a.seed,
            device=a.device)
    else:   # small coverage probe: one latent, few frames, distinct seed and tag
        gen("v3probe", every=2, n_lat=1, chunk=200, K=a.K, n_screen=60, n_real=1,
            seed=a.seed + 99, device=a.device, max_frames=a.n)


if __name__ == "__main__":
    main()
