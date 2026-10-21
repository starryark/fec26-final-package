"""MAST v3 label-free equilibria: select_v2 plus a verified force-free height, an optional
centre-column limiter boundary and optional passive-structure flux (inputs only, no labels).

Changes relative to fec.mast_select_v2.select_v2 (each switchable; all off == select_v2):
  zmode="root"/"root_t"  force-free height Zt: v2's one-step secant is kept only where a
        verification solve confirms it (next secant step moves < 2 cm, |Zt| <= 0.3 m);
        otherwise c(Zt) is scanned on -0.3..0.3 m (stacked batched solves), the sign change
        nearest a reference height is bracketed and refined (two regula-falsi solves).
        Reference: 0 m ("root") or the running median of verified heights of the same shot
        within +-10 frames ("root_t", temporal continuity).  No sign change -> reference.
  kfit_warm="zs"  kfit2 is warm-started from the force-free solution at the final Zt
        (v2 uses the Zt = 0 solution, whose radial-field feedback c then stays frozen inside
        kfit2, review finding F1).
  limiter=LIMITER_CC  plasma boundary psi_b = max(X-point flux, max flux on the centre-column
        armour R = 0.20 m, |Z| <= 1.0 m) as in EFIT: a separatrix may not pass through the
        column.  Limited frames need no X-point, so the psi_b = -1e30 sentinel (F2) vanishes
        wherever the limiter flux is below the axis flux.
  passive  the caller appends PassiveModel().augment(M) columns to the solver matrix and the
        per-frame C1 currents to I (stage1_v3 --passive c1_r1).

    python -B fec/mast_select_v3.py --out fec_work/mast_v3 [--limit N]
        [--config mast_public_test] [--passive none|c1_r1] [--zmode v2|root|root_t]
        [--limiter on|off] [--kfit_warm o0|zs]

Study and acceptance: docs/research/MAST_V3_STUDY_20261008.md (fec/mast_v3_study.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.gs import Solver  # noqa: E402
from fec.mast_select_v2 import (FFZ_PROF, LEVELS_V1, V2_FALLBACK, V2_GRID, V2_KFIT, KF_DV,  # noqa: E402
                                KF_PRIOR, force_free_z, nearest_in_shot, prep_profiles,
                                select_stacked, symmetry_mm, to_dev)

# Centre-column armour (inboard limiter) radius 0.20 m: Sykes 2002 "c/col graphite radius =
# 20cm" (docs/research/MAST_CONVENTIONS_RESEARCH_20261007.md section 6); straight section
# |Z| <= 1.0 m (the armour flares beyond |Z| ~ 1.1 m, fec/passive.py 'rodgr').
LIMITER_CC = dict(R=0.20, zmax=1.0)


class V3Solver(Solver):
    """fec.gs.Solver with an optional limiter in the plasma-boundary search.

    solve() reproduces Solver.solve exactly when limiter is None (same operations in the
    same order; only the per-iteration host sync of the convergence history is dropped).
    solve_gen() is the two-shape profile of fec.twin_mast (twin truths with the limiter).
    """

    def __init__(self, spec, Mext, device="cuda", dtype=torch.float32, limiter=None):
        super().__init__(spec, Mext, device=device, dtype=dtype)
        self.set_limiter(limiter)
        self.last_limited = None

    def set_limiter(self, limiter):
        self.limiter = limiter
        if limiter is None:
            return
        gR, gZ = self.spec.gR, self.spec.gZ
        R = float(limiter["R"])
        j = int(np.searchsorted(gR, R)) - 1
        self.lim_j = j
        self.lim_w = float((R - gR[j]) / (gR[j + 1] - gR[j]))
        rows = np.flatnonzero(np.abs(gZ) <= float(limiter["zmax"]) + 1e-9)
        self.lim_rows = torch.as_tensor(rows, device=self.dev)
        self.lim_R = R
        self.lim_Z = torch.as_tensor(gZ[rows], device=self.dev, dtype=self.dt)

    def boundary(self, psi, psi_ax, Rax, Zax, xp, final=False):
        """psi_b (B,) and limited (B,) bool.  X-point flux as fec.gs; with a limiter the
        larger of it and the limiter flux (limiter points beyond an X-point, i.e. in its
        divertor/private-flux side, are ignored exactly as core_mask cuts the plasma, and so
        are points whose flux exceeds the axis flux).
        final=False applies the in-loop 'no X-point' fallback of fec.gs.Solver.solve."""
        (fu, vu, Rxu, Zxu), (fl, vl, Rxl, Zxl) = xp
        neg = torch.full_like(vu, -1e30)
        psi_b = torch.maximum(torch.where(fu, vu, neg), torch.where(fl, vl, neg))
        limited = torch.zeros_like(fu)
        if self.limiter is not None:
            j, w = self.lim_j, self.lim_w
            pl = (1 - w) * psi[:, self.lim_rows, j] + w * psi[:, self.lim_rows, j + 1]
            ok = torch.ones_like(pl, dtype=torch.bool)
            for (found, Rx, Zx) in ((fu, Rxu, Zxu), (fl, Rxl, Zxl)):
                dRx = (Rx - Rax)[:, None]; dZx = (Zx - Zax)[:, None]
                proj = (self.lim_R - Rax[:, None]) * dRx + (self.lim_Z[None] - Zax[:, None]) * dZx
                ok &= torch.where(found[:, None], proj < dRx ** 2 + dZx ** 2,
                                  torch.ones_like(ok))
            ok &= pl < psi_ax[:, None]          # points above the axis flux are not on
            plim = torch.where(ok, pl, torch.full_like(pl, -1e30)).max(1).values  # its surfaces
            limited = (plim > psi_b) & (plim > -1e29)
            psi_b = torch.where(limited, plim, psi_b)
        if not final:
            nox = ~(fu | fl) & ~limited
            if nox.any():
                psi_b = torch.where(nox, psi_ax - 0.05 * psi_ax.abs().clamp_min(1e-3), psi_b)
        return psi_b, limited

    # ------------------------------------------------------------------ Picard loop
    def _picard(self, Iext, Ip, shape_fn, Zt, n_iter, omega, kc, I0, c0, Rinit, fix_z):
        dev, dt = self.dev, self.dt
        t = lambda x: torch.as_tensor(x, device=dev, dtype=dt)
        Iext, Ip, Zt = t(Iext), t(Ip), t(Zt)
        B = Iext.shape[0]
        psi_ext = (Iext @ self.Mext.T).reshape(B, self.nz, self.nr)
        Rs, Zs = self.Rs, self.Zs
        R0 = self.spec.R0
        if I0 is None:
            Rc = t(Rinit) if Rinit is not None else torch.full((B,), R0, device=dev, dtype=dt)
            w = torch.exp(-((Rs[None] - Rc[:, None]) ** 2 + (Zs[None] - Zt[:, None]) ** 2) / 0.05)
            I = Ip[:, None] * w / w.sum(1, keepdim=True)
        else:
            I = t(I0)
        c = t(c0) if c0 is not None else torch.zeros(B, device=dev, dtype=dt)
        fb = (self.RR ** 2)[None] * (self.ZZ[None] - Zt[:, None, None])
        Npix = self.nz * self.nr
        prev_mask = None
        for it in range(n_iter):
            psi = psi_ext + (I @ self.G.T).reshape(B, self.nz, self.nr) + c[:, None, None] * fb
            psi_ax, Rax, Zax = self.find_axis(psi)
            xp = self.find_xpoints(psi, Rax, Zax, psi_ax)
            psi_b, _ = self.boundary(psi, psi_ax, Rax, Zax, xp)
            seedm = prev_mask if (it % 10 != 0) else None
            mask, psiN = self.core_mask(psi, psi_ax, psi_b, Rax, Zax, xp, seed_mask=seedm)
            prev_mask = mask
            pn = psiN.reshape(B, -1)[:, self.sidx].clamp(0, 1)
            ms = mask.reshape(B, -1)[:, self.sidx].to(dt)
            shape = shape_fn(pn, ms)
            tot = shape.sum(1, keepdim=True).clamp_min(1e-30)
            Inew = Ip[:, None] * shape / tot
            Zc_new = (Inew * Zs[None]).sum(1) / Ip
            delta = Zt - Zc_new
            if fix_z:
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
        psi_b, limited = self.boundary(psi, psi_ax, Rax, Zax, xp, final=True)
        self.last_limited = limited.cpu().numpy()
        return dict(psi=psi, I=I, c=c, psi_ax=psi_ax, Rax=Rax, Zax=Zax, psi_b=psi_b,
                    xu=(fu, Rxu, Zxu, vu), xl=(fl, Rxl, Zxl, vl), dI=dI, delta=delta,
                    hist=[], psi_ext=psi_ext, limited=limited)

    def solve(self, Iext, Ip, prof, Zt, n_iter=150, omega=0.4, kc=0.05, I0=None, c0=None,
              Rinit=None, fix_z=True, verbose=False):
        """Same signature and numerics as fec.gs.Solver.solve (+ limiter, + 'limited')."""
        prof = torch.as_tensor(prof, device=self.dev, dtype=self.dt)
        beta0, am, an = prof[:, 0:1], prof[:, 1:2], prof[:, 2:3]
        Rs, R0 = self.Rs, self.spec.R0

        def shape_fn(pn, ms):
            return (beta0 * Rs[None] / R0 + (1 - beta0) * R0 / Rs[None]) * \
                (1 - pn ** am).clamp_min(0) ** an * ms
        return self._picard(Iext, Ip, shape_fn, Zt, n_iter, omega, kc, I0, c0, Rinit, fix_z)

    def solve_gen(self, Iext, Ip, prof, Zt, n_iter=150, omega=0.4, kc=0.05, I0=None, c0=None,
                  Rinit=None):
        """fec.twin_mast two-shape profile, prof (B,5) = beta0, a1, n1, a2, n2."""
        prof = torch.as_tensor(prof, device=self.dev, dtype=self.dt)
        b0, a1, n1 = prof[:, 0:1], prof[:, 1:2], prof[:, 2:3]
        a2, n2 = prof[:, 3:4], prof[:, 4:5]
        Rs, R0 = self.Rs, self.spec.R0

        def shape_fn(pn, ms):
            return (b0 * Rs[None] / R0 * (1 - pn ** a1).clamp_min(0) ** n1 +
                    (1 - b0) * R0 / Rs[None] * (1 - pn ** a2).clamp_min(0) ** n2) * ms
        return self._picard(Iext, Ip, shape_fn, Zt, n_iter, omega, kc, I0, c0, Rinit, True)


V3TwinSolver = V3Solver


# ----------------------------------------------------------------------------- kinetic fit
def kfit_basis_args(nbasis, prior, w_prior):
    """Basis count check and the prior / prior-weight vectors for nbasis 4 (unchanged) or 6.

    nbasis=6 appends R/R0 * x(1-x) and R0/R * x(1-x) (x = psiN) after the four kfit2 bases
    (docs/research/KFIT_BROAD_BASIS_20261009.md); 5-element prior / w_prior vectors (4 bases +
    dv) get prior target 0 and prior weight 0 inserted for the two new bases."""
    if nbasis not in (4, 6):
        raise ValueError(f"nbasis must be 4 or 6, not {nbasis!r}")
    if nbasis == 4:
        return prior, w_prior
    prior = np.asarray(prior, float)
    w_prior = tuple(float(x) for x in w_prior)
    if len(prior) == 5:
        prior = np.concatenate([prior[:4], [0.0, 0.0], prior[4:]])
    if len(w_prior) == 5:
        w_prior = w_prior[:4] + (0.0, 0.0) + w_prior[4:]
    if len(prior) != 7 or len(w_prior) != 7:
        raise ValueError("nbasis=6 needs 5- or 7-element prior / w_prior")
    return prior, w_prior


def kfit_basis_stack(Rs, R0, pn, ms, nbasis):
    """(B, nbasis, Ns) unnormalised kfit basis currents on the source nodes (nbasis 4 or 6)."""
    x1 = (1 - pn ** 2) * ms
    x2 = (1 - pn) ** 2 * ms
    if nbasis == 4:
        return torch.stack([Rs / R0 * x1, Rs / R0 * x2, R0 / Rs * x1, R0 / Rs * x2], 1)
    x3 = pn * (1 - pn) * ms
    return torch.stack([Rs / R0 * x1, Rs / R0 * x2, R0 / Rs * x1, R0 / Rs * x2,
                        Rs / R0 * x3, R0 / Rs * x3], 1)


def kfit3(S, Iext, Ip, Zt, pairs, n_iter=120, omega=0.3, kc=0.05, I0=None, c0=None,
          prior=KF_PRIOR, w_prior=(0.3, 0.3, 0.3, 0.3, 1.0), lm=2.0, pair_tol=0.02,
          dv_max=2.0, fit_dv=True, cfix=False, nbasis=4):
    """fec.mast_select_v2.kfit2 with the solver's boundary() (limiter) and, if cfix, the
    radial-field feedback driven by the pre-shift offset as in Solver.solve (F1).
    Identical to kfit2 for a V3Solver without limiter and cfix=False.
    nbasis=6: two extra non-negative broad/hollow bases (kfit_basis_args); u = 6 weights + dv."""
    nb = int(nbasis)
    prior, w_prior = kfit_basis_args(nb, prior, w_prior)
    K = nb + 1
    dev, dt = S.dev, S.dt
    t = lambda x: torch.as_tensor(x, device=dev, dtype=dt)
    Iext, Ip, Zt = t(Iext), t(Ip), t(Zt)
    B = Iext.shape[0]
    nz, nr = S.nz, S.nr
    Npix = nz * nr
    psi_ext = (Iext @ S.Mext.T).reshape(B, nz, nr)
    Rs, Zs = S.Rs, S.Zs
    R0 = S.spec.R0
    gR0 = S.spec.gR[0]
    iz0 = int(np.argmin(np.abs(S.spec.gZ)))
    pr = t(np.nan_to_num(pairs, nan=1.0))
    pvalid = t(np.isfinite(pairs).all(-1).astype(np.float32))
    fj = (pr - gR0) / S.dR
    j0 = torch.clamp(fj.floor().long(), 0, nr - 2)
    w = fj - j0.to(dt)
    rowG = S.G[iz0 * nr:(iz0 + 1) * nr]
    Gin = (1 - w[..., 0, None]) * rowG[j0[..., 0]] + w[..., 0, None] * rowG[j0[..., 0] + 1]
    Gout = (1 - w[..., 1, None]) * rowG[j0[..., 1]] + w[..., 1, None] * rowG[j0[..., 1] + 1]
    Gd = Gin - Gout
    r2d = (pr[..., 0] ** 2 - pr[..., 1] ** 2) / 2 * KF_DV

    def chord(field, k):
        row = field[:, iz0, :]
        a = row.gather(1, j0[..., k]); b = row.gather(1, j0[..., k] + 1)
        return (1 - w[..., k]) * a + w[..., k] * b

    if I0 is None:
        g = torch.exp(-((Rs[None] - 0.9) ** 2 + (Zs[None] - Zt[:, None]) ** 2) / 0.05)
        I = Ip[:, None] * g / g.sum(1, keepdim=True)
    else:
        I = t(I0)
    c = t(c0) if c0 is not None else torch.zeros(B, device=dev, dtype=dt)
    pu = t(prior)
    u = pu.expand(B, K).clone()
    if not fit_dv:
        w_prior = tuple(w_prior[:nb]) + (1e4,)
    pw = t(w_prior)
    fb = (S.RR ** 2)[None] * (S.ZZ[None] - Zt[:, None, None])
    vfield = (S.RR ** 2 / 2 * KF_DV)[None]
    prev_mask = None
    for it in range(n_iter):
        psi = psi_ext + (I @ S.G.T).reshape(B, nz, nr) + c[:, None, None] * fb + \
            u[:, nb, None, None] * vfield
        psi_ax, Rax, Zax = S.find_axis(psi)
        xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
        psi_b, _ = S.boundary(psi, psi_ax, Rax, Zax, xp)
        seedm = prev_mask if (it % 10 != 0) else None
        mask, psiN = S.core_mask(psi, psi_ax, psi_b, Rax, Zax, xp, seed_mask=seedm)
        prev_mask = mask
        pn = psiN.reshape(B, -1)[:, S.sidx].clamp(0, 1)
        ms = mask.reshape(B, -1)[:, S.sidx].to(dt)
        basis = kfit_basis_stack(Rs, R0, pn, ms, nb)
        basis = basis / basis.sum(2, keepdim=True).clamp_min(1e-12) * Ip[:, None, None]
        mix = torch.einsum("bk,bkn->bn", u[:, :nb], basis)
        delta = Zt - (mix * Zs[None]).sum(1) / Ip
        full = torch.zeros(B * nb, Npix, device=dev, dtype=dt)
        full[:, S.sidx] = basis.reshape(B * nb, -1)
        full = S._shiftZ(full.reshape(B * nb, nz, nr), delta.repeat_interleave(nb))
        basis = full.reshape(B, nb, Npix)[:, :, S.sidx]
        phi = (psi_ax - psi_b).abs().clamp_min(1e-4)
        ext = psi_ext + c[:, None, None] * fb
        A_p = torch.cat([torch.einsum("bpn,bkn->bpk", Gd, basis), r2d[..., None]], 2)
        b_p = -(chord(ext, 0) - chord(ext, 1))
        sc = (pvalid / (phi[:, None] * pair_tol))[..., None]
        A_ip = torch.cat([torch.ones(B, 1, nb, device=dev, dtype=dt),
                          torch.zeros(B, 1, 1, device=dev, dtype=dt)], 2)
        A = torch.cat([A_ip * 100.0, A_p * sc, torch.diag_embed(pw).expand(B, K, K),
                       lm * torch.eye(K, device=dev, dtype=dt).expand(B, K, K)], 1)
        bb = torch.cat([torch.full((B, 1), 100.0, device=dev, dtype=dt), b_p * sc[..., 0],
                        (pw * pu).expand(B, K), lm * u], 1)
        A64 = torch.nan_to_num(A.double()); b64 = torch.nan_to_num(bb.double())
        AtA = A64.transpose(1, 2) @ A64 + 1e-9 * torch.eye(K, device=dev, dtype=torch.float64)
        unew = torch.linalg.solve(AtA, (A64.transpose(1, 2) @ b64[..., None])[..., 0]).to(dt)
        bad = ~torch.isfinite(unew).all(1)
        unew = torch.where(bad[:, None], u, unew)
        cu = unew[:, :nb].clamp_min(0)
        cu = cu / cu.sum(1, keepdim=True).clamp_min(1e-6)
        unew = torch.cat([cu, unew[:, nb:].clamp(-dv_max, dv_max)], 1)
        u = (1 - omega) * u + omega * unew
        Inew = torch.einsum("bk,bkn->bn", u[:, :nb], basis)
        Inew = Ip[:, None] * Inew / Inew.sum(1, keepdim=True).clamp_min(1e-30)
        Zc = (Inew * Zs[None]).sum(1) / Ip
        c = c + kc * ((delta + Zt - Zc) if cfix else (Zt - Zc))
        dI = (Inew - I).abs().sum(1) / Ip
        I = (1 - omega) * I + omega * Inew
    psi = psi_ext + (I @ S.G.T).reshape(B, nz, nr) + c[:, None, None] * fb + \
        u[:, nb, None, None] * vfield
    psi_ax, Rax, Zax = S.find_axis(psi)
    xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
    psi_b, limited = S.boundary(psi, psi_ax, Rax, Zax, xp, final=True)
    res = ((chord(psi, 0) - chord(psi, 1)) / (psi_ax - psi_b).abs()[:, None]) * pvalid
    return dict(psi=psi, I=I, c=c, u=u, psi_ax=psi_ax, Rax=Rax, Zax=Zax, psi_b=psi_b, dI=dI,
                pair_res=res, dv=u[:, nb] * KF_DV, limited=limited)


# ----------------------------------------------------------------------------- vertical
ZGRID = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)
ZFLAGS = {0: "v2 secant verified", 1: "scan: single root", 2: "scan: root nearest reference",
          3: "no root in +-0.3 m: reference height"}


def running_ref(z, ok, shot, h=10):
    """Per frame: median of z over frames of the same shot within +-h positions with ok;
    NaN if none (frames time-ordered within a shot)."""
    out = np.full(len(z), np.nan)
    for s_ in np.unique(shot):
        ix = np.flatnonzero(shot == s_)
        good = ok[ix]
        zz = z[ix]
        for k in range(len(ix)):
            lo, hi = max(0, k - h), min(len(ix), k + h + 1)
            sel = good[lo:hi]
            if sel.any():
                out[ix[k]] = np.median(zz[lo:hi][sel])
    return out


def pick_root(zg, cg, okg, zref):
    """Roots of c(Z) from sign changes between adjacent valid scan points.
    zg (K,n) per-frame heights sorted along axis 0; cg (K,n) feedback; okg (K,n) converged;
    zref (n,).  Returns (zr, za, zb, ca, cb, nroots): the root nearest zref and its bracket."""
    K, n = cg.shape
    zg = np.broadcast_to(np.asarray(zg, float).reshape(K, -1), (K, n))
    zr = np.full(n, np.nan); za = np.full(n, np.nan); zb = np.full(n, np.nan)
    ca = np.full(n, np.nan); cb = np.full(n, np.nan)
    nroots = np.zeros(n, int)
    for i in range(n):
        best = np.inf
        for k in range(K - 1):
            if not (okg[k, i] and okg[k + 1, i]):
                continue
            c1, c2 = cg[k, i], cg[k + 1, i]
            if not (np.isfinite(c1) and np.isfinite(c2)) or np.sign(c1) == np.sign(c2):
                continue
            z1, z2 = zg[k, i], zg[k + 1, i]
            r = z1 - c1 * (z2 - z1) / (c2 - c1) if c2 != c1 else z1
            nroots[i] += 1
            if abs(r - zref[i]) < best:
                best = abs(r - zref[i])
                zr[i], za[i], zb[i], ca[i], cb[i] = r, z1, z2, c1, c2
    return zr, za, zb, ca, cb, nroots


def robust_zt(S, Ix, Ip, prof, shot, temporal=True, dz=0.05, zmax=0.3, tol=0.02,
              n_ref=60, conv_tol=1e-2, max_stack=2048, href=10):
    """Verified force-free height (module doc).  Returns dict(o0, Zs, I, c, zflag, nroots,
    Zv, zref): o0 = Zt=0 solution (as v2); (I, c) = solution state at the final Zs."""
    dev = S.dev
    B = len(Ip)
    shot = np.asarray(shot)
    # v2 secant, numerically identical to fec.mast_select_v2.force_free_z
    o0 = S.solve(Ix, Ip, prof, np.zeros(B), n_iter=150, Rinit=np.full(B, 0.9))
    o1 = S.solve(Ix, Ip, prof, np.full(B, dz), n_iter=60, I0=o0["I"], c0=o0["c"])
    c0, c1 = o0["c"].cpu().numpy(), o1["c"].cpu().numpy()
    den = np.where(np.abs(c1 - c0) > 1e-12, c1 - c0, np.nan)
    Zv = np.nan_to_num(np.clip(-c0 * dz / den, -zmax, zmax))
    conv0 = o0["dI"].cpu().numpy() < conv_tol
    conv1 = o1["dI"].cpu().numpy() < conv_tol
    # verification solve at Zv; next secant step against the farther of the two v2 points
    o2 = S.solve(Ix, Ip, prof, Zv, n_iter=n_ref, I0=o0["I"], c0=o0["c"])
    c2 = o2["c"].cpu().numpy()
    conv2 = o2["dI"].cpu().numpy() < conv_tol
    far0 = np.abs(Zv) >= np.abs(Zv - dz)
    Zf = np.where(far0, 0.0, dz); cf = np.where(far0, c0, c1)
    with np.errstate(all="ignore"):
        Z2 = Zv - c2 * (Zv - Zf) / (c2 - cf)
    ok = conv2 & np.isfinite(Z2) & (np.abs(Z2 - Zv) <= tol) & (np.abs(Zv) < zmax - 1e-9) &         np.isfinite(den)
    Zs = Zv.copy()
    zflag = np.zeros(B, int)
    nroots = np.ones(B, int)
    zref_all = running_ref(Zv, ok, shot, href) if temporal else np.full(B, np.nan)
    zref_all = np.where(np.isfinite(zref_all), zref_all, 0.0)
    Ist = o2["I"].clone(); cst = o2["c"].clone()
    fail = np.flatnonzero(~ok)
    if len(fail):
        zref = zref_all[fail]
        nf = len(fail)
        ft = torch.as_tensor(fail, device=dev)
        scan = np.array(ZGRID, float)
        cs = np.full((len(scan), nf), np.nan); oks = np.zeros((len(scan), nf), bool)
        per = max(1, max_stack // nf)
        for s0 in range(0, len(scan), per):
            ks = list(range(s0, min(len(scan), s0 + per)))
            m = len(ks)
            zt = np.concatenate([np.full(nf, scan[k]) for k in ks])
            o = S.solve(np.tile(Ix[fail], (m, 1)), np.tile(Ip[fail], m),
                        np.tile(prof[fail], (m, 1)), zt, n_iter=n_ref,
                        I0=o0["I"][ft].repeat(m, 1), c0=o0["c"][ft].repeat(m))
            cc = o["c"].cpu().numpy().reshape(m, nf)
            dd = o["dI"].cpu().numpy().reshape(m, nf)
            for j, k in enumerate(ks):
                cs[k] = cc[j]; oks[k] = dd[j] < conv_tol
            del o
        # per-frame point set: scan + (0, c0) + (dz, c1) + (Zv, c2) unless Zv duplicates
        zv = Zv[fail]
        dup = np.min(np.abs(np.concatenate([scan, [0.0, dz]])[:, None] - zv[None]), 0) < 0.01
        zg = np.concatenate([np.repeat(scan[:, None], nf, 1), np.zeros((1, nf)),
                             np.full((1, nf), dz), np.where(dup, np.nan, zv)[None]], 0)
        cg = np.concatenate([cs, c0[fail][None], c1[fail][None], c2[fail][None]], 0)
        okg = np.concatenate([oks, conv0[fail][None], conv1[fail][None],
                              (conv2[fail] & ~dup)[None]], 0)
        zsort = np.where(np.isfinite(zg), zg, np.inf)
        o_ = np.argsort(zsort, 0, kind="stable")
        zg = np.take_along_axis(zsort, o_, 0)
        cg = np.take_along_axis(cg, o_, 0)
        okg = np.take_along_axis(okg, o_, 0) & np.isfinite(zg)
        zr, za, zb, ca, cb, nr = pick_root(zg, cg, okg, zref)
        has = np.isfinite(zr)
        nroots[fail] = nr
        # regula falsi: solve at the interpolated root, shrink the bracket, re-interpolate,
        # and solve once more at the final height (its state warm-starts kfit2)
        zcur = np.where(has, zr, zref)
        o = S.solve(Ix[fail], Ip[fail], prof[fail], zcur, n_iter=n_ref, I0=o0["I"][ft],
                    c0=o0["c"][ft])
        cr = o["c"].cpu().numpy()
        upd = has & (o["dI"].cpu().numpy() < conv_tol) & np.isfinite(cr)
        left = upd & (np.sign(cr) == np.sign(ca))
        right = upd & ~left
        za = np.where(left, zcur, za); ca = np.where(left, cr, ca)
        zb = np.where(right, zcur, zb); cb = np.where(right, cr, cb)
        with np.errstate(all="ignore"):
            znew = za - ca * (zb - za) / (cb - ca)
        znew = np.where(upd & np.isfinite(znew), np.clip(znew, np.minimum(za, zb),
                                                          np.maximum(za, zb)), zcur)
        o = S.solve(Ix[fail], Ip[fail], prof[fail], znew, n_iter=n_ref, I0=o["I"], c0=o["c"])
        Zs[fail] = znew
        zflag[fail] = np.where(has, np.where(nr > 1, 2, 1), 3)
        Ist[ft] = o["I"]; cst[ft] = o["c"]
        del o
    return dict(o0=o0, Zs=Zs, I=Ist, c=cst, zflag=zflag, nroots=nroots, Zv=Zv,
                zref=zref_all)


# ----------------------------------------------------------------------------- selector
def select_v3(S, I, Ip, R, te, ne, shot, grid=None, kfit_kw=None, w_ne=1.0, n_iter=100,
              min_pairs=3, use_kfit=True, use_grid=True, dv_col=False, zmode="v2",
              kfit_warm="o0", cfix=False):
    """select_v2 (pairs rule) with the v3 switches (module doc).  S: V3Solver (limiter set on
    the solver) or any fec.gs.Solver when no limiter is wanted.  Extra outputs: zflag,
    limited, Zv (v2 secant height)."""
    from fec.kefit import LEVELS, te_pairs
    dev = S.dev
    B = len(Ip)
    shot = np.asarray(shot)
    grid = V2_GRID if grid is None else grid
    kw = dict(V2_KFIT, **(kfit_kw or {}))
    levels_k = kw.pop("levels", LEVELS)
    Ix = np.concatenate([I, np.zeros((B, 1))], 1) if dv_col else np.asarray(I, float)
    prof = np.tile(FFZ_PROF, (B, 1))
    if zmode == "v2":
        o0, Zs = force_free_z(S, Ix, Ip, prof)
        zst = dict(I=o0["I"], c=o0["c"], zflag=np.zeros(B, int), Zv=Zs.copy())
    else:
        zst = robust_zt(S, Ix, Ip, prof, shot, temporal=(zmode == "root_t"))
        o0, Zs = zst["o0"], zst["Zs"]
    if kfit_warm == "zs" and zmode != "v2":
        wI, wc = zst["I"], zst["c"]
    else:
        wI, wc = o0["I"], o0["c"]
    keys = ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI")
    gR = torch.as_tensor(S.spec.gR, device=dev, dtype=torch.float32)
    iz0 = int(np.argmin(np.abs(S.spec.gZ)))
    lev = torch.as_tensor(LEVELS_V1, device=dev, dtype=torch.float32)
    out = dict(Zt=Zs.copy(), method=np.zeros(B, int), score=np.full(B, np.nan),
               grid_choice=np.full(B, -1), grid_score=np.full(B, np.nan),
               inherited=np.zeros(B, bool), kfit_dv=np.full(B, np.nan),
               kfit_score=np.full(B, np.nan), kfit_npairs=np.zeros(B, int),
               zflag=zst["zflag"], Zv=zst["Zv"], limited=np.zeros(B, bool))
    need_grid = np.ones(B, bool)
    if use_kfit:
        pairs = np.stack([te_pairs(R[i], te[i], levels=levels_k) for i in range(B)])
        o = kfit3(S, Ix, Ip, Zs, pairs, I0=wI, c0=wc, cfix=cfix, **kw)
        pte = to_dev(prep_profiles(R, te), dev)
        pne = to_dev(prep_profiles(R, ne), dev)
        mk = symmetry_mm(o["psi"], o["psi_ax"], o["psi_b"], gR, iz0, pte, lev)[0] + \
            w_ne * symmetry_mm(o["psi"], o["psi_ax"], o["psi_b"], gR, iz0, pne, lev)[0]
        conv = (o["dI"] < 1e-2).cpu().numpy()
        mk = np.where(conv, mk.cpu().numpy(), np.nan)
        npairs = np.isfinite(pairs).all(-1).sum(1)
        for k in keys:
            out[k] = o[k].cpu().numpy()
        out.update(method=np.ones(B, int), score=mk, kfit_dv=o["dv"].cpu().numpy(),
                   kfit_score=mk, kfit_npairs=npairs, limited=o["limited"].cpu().numpy())
        need_grid = ~((npairs >= min_pairs) & conv)
        del o
    if use_grid and need_grid.any():
        sub = np.flatnonzero(need_grid) if use_kfit else np.arange(B)
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
            if k not in out:
                out[k] = np.zeros((B,) + g[k].shape[1:], g[k].dtype)
            out[k][sub] = g[k]
        out["method"][sub] = 0
        out["score"][sub] = J
        # limited flag of grid frames: recompute from their final maps
        if getattr(S, "limiter", None) is not None:
            t = lambda x: torch.as_tensor(np.asarray(x), device=dev, dtype=S.dt)
            psi = t(out["psi"][sub])
            pa, Ra, Za = S.find_axis(psi)
            xp = S.find_xpoints(psi, Ra, Za, pa)
            out["limited"][sub] = S.boundary(psi, pa, Ra, Za, xp, final=True)[1].cpu().numpy()
    return out


# ----------------------------------------------------------------------------- driver
V3_DEFAULT = dict(passive="none", zmode="v2", limiter="off", kfit_warm="o0")


def stage1_v3(out_dir, limit=0, max_frames=1024, config="mast_public_test", passive="none",
              zmode="v2", limiter="off", kfit_warm="o0"):
    """Production driver (same outputs as fec.mast_select_v2.stage1_v2 + zflag, Zv, limited):
    psi.npy + meta.npz (meta_partial.npz after every chunk)."""
    import json
    import time
    from fec.features import load_ds
    from fec.machines import MAST_CH, MAST_COILS, mast_setup
    from fec.mast_select_v2 import shot_chunks
    from fec.mast_v1 import thomson_frame
    from fec.paths import WORK
    ds = load_ds(config)
    info = ds["info"]
    N = len(ds["t_ms"]) if not limit else limit
    cur = np.asarray(ds["cur"][:N], dtype=np.float64)[:, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    spec, M = mast_setup()
    Ipas = None
    if passive != "none":
        from fec.passive import PassiveModel
        f = WORK / "passive" / "frames" / config / passive / "I.npy"
        if not f.exists():
            raise SystemExit(f"missing passive currents {f}: run fec/passive.py signals/frames "
                             f"--config {config} first")
        Ipas = np.load(f, mmap_mode="r")
        if Ipas.shape[0] != len(ds["t_ms"]):
            raise SystemExit(f"passive currents {Ipas.shape} do not match {config}")
        M = PassiveModel().augment(M)
    S = V3Solver(spec, M, limiter=LIMITER_CC if limiter == "on" else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(config=config, passive=passive, zmode=zmode, limiter=limiter,
               kfit_warm=kfit_warm, limiter_geom=LIMITER_CC, N=N)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    psi_out = np.lib.format.open_memmap(out_dir / "psi.npy", "w+", np.float32, (N, 65, 65))
    mk = ("psi_ax", "psi_b", "Rax", "Zax", "Zt", "c", "dI", "method", "score", "grid_choice",
          "grid_score", "inherited", "kfit_dv", "kfit_score", "kfit_npairs", "zflag", "Zv",
          "limited")
    meta = {k: np.full(N, np.nan) for k in mk}
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
        o = select_v3(S, I, Ip, pad(Rl), pad(tel), pad(nel), shot[sl], zmode=zmode,
                      kfit_warm=kfit_warm)
        psi_out[sl] = o["psi"]
        for k in mk:
            if k in o:
                meta[k][sl] = o[k]
        np.savez(out_dir / "meta_partial.npz", done=np.array(sl.stop), **meta)
        print(f"{sl.stop}/{N} {time.time() - t0:.0f}s kfit used {np.mean(o['method']):.2f} "
              f"score med {np.nanmedian(o['score']):.3f} zflag>0 {np.mean(o['zflag'] > 0):.3f} "
              f"limited {np.mean(o['limited']):.3f}", flush=True)
        torch.cuda.empty_cache()
    psi_out.flush()
    np.savez(out_dir / "meta.npz", **meta)
    (out_dir / "meta_partial.npz").unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fec.paths import WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORK / "mast_v3"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default="mast_public_test")
    ap.add_argument("--passive", default=V3_DEFAULT["passive"])
    ap.add_argument("--zmode", default=V3_DEFAULT["zmode"], choices=["v2", "root", "root_t"])
    ap.add_argument("--limiter", default=V3_DEFAULT["limiter"], choices=["on", "off"])
    ap.add_argument("--kfit_warm", default=V3_DEFAULT["kfit_warm"], choices=["o0", "zs"])
    a = ap.parse_args()
    stage1_v3(Path(a.out), limit=a.limit, config=a.config, passive=a.passive, zmode=a.zmode,
              limiter=a.limiter, kfit_warm=a.kfit_warm)
