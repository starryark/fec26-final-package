"""Kinetic EFIT for MAST: free-boundary equilibrium constrained by Thomson, not magnetics.

Per frame unknowns u = [a1, a2, b1, b2, dv]:
  J = sum_k a_k * R/R0 * x_k(psiN) + b_k * R0/R * x_k(psiN)   (each basis scaled to carry Ip)
  x_1 = 1 - psiN^2, x_2 = (1 - psiN)^2   (non-hollow, zero at the boundary)
  dv  = uniform vertical-field correction (units DV_SCALE tesla) for unmodelled currents.
Constraints (weighted least squares, Picard-iterated like EFIT):
  sum of basis currents = Ip; Thomson equal-Te pairs on the Z=0 chord share psi;
  weak priors on all coefficients.
Coil currents are fixed at their measured values; vertical position via rigid shift +
radial-field feedback at the force-free Zt (from fec.gs).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import median_filter, uniform_filter1d

from fec.gs import Solver

DV_SCALE = 0.01  # tesla per unit of u[4]
LEVELS = np.array([0.15, 0.3, 0.45, 0.6, 0.75])
PRIOR = np.array([0.2, 0.0, 0.8, 0.0, 0.0])
PRIOR_W = np.array([0.05, 0.05, 0.05, 0.05, 0.3])


def te_pairs(R, te, levels=LEVELS, rmin=0.236, rmax=1.51):
    """Inboard/outboard radii where smoothed Te crosses fractions of its peak."""
    out = np.full((len(levels), 2), np.nan)
    ok = np.isfinite(te) & np.isfinite(R) & (te > 0)
    if ok.sum() < 30:
        return out
    R, te = R[ok], te[ok]
    o = np.argsort(R)
    R, te = R[o], te[o]
    t = uniform_filter1d(median_filter(te, 7, mode="nearest"), 5, mode="nearest")
    core = (R > 0.3) & (R < 1.35)
    if core.sum() < 10:
        return out
    ip = np.flatnonzero(core)[np.argmax(t[core])]
    tmax = t[ip]
    if not tmax > 50:
        return out
    for li, f in enumerate(levels):
        lev = f * tmax
        j = ip
        while j > 0 and t[j] >= lev:
            j -= 1
        rin = np.nan
        if t[j] < lev and j < ip:
            rin = R[j] + (lev - t[j]) / (t[j + 1] - t[j]) * (R[j + 1] - R[j])
        j = ip
        while j < len(t) - 1 and t[j] >= lev:
            j += 1
        rout = np.nan
        if t[j] < lev and j > ip:
            rout = R[j - 1] + (t[j - 1] - lev) / (t[j - 1] - t[j]) * (R[j] - R[j - 1])
        if rin > rmin and rout < rmax:
            out[li] = (rin, rout)
    return out


class KineticSolver(Solver):
    def kfit(self, Iext, Ip, Zt, pairs, n_iter=150, omega=0.3, kc=0.05, I0=None, c0=None,
             prior=PRIOR, prior_w=PRIOR_W, w_pair=1.0, pair_tol=0.02, Rinit=None,
             u0=None, verbose=False):
        dev, dt = self.dev, self.dt
        t = lambda x: torch.as_tensor(x, device=dev, dtype=dt)
        Iext, Ip, Zt = t(Iext), t(Ip), t(Zt)
        B = Iext.shape[0]
        nz, nr = self.nz, self.nr
        Npix = nz * nr
        psi_ext = (Iext @ self.Mext.T).reshape(B, nz, nr)
        Rs, Zs = self.Rs, self.Zs
        R0 = self.spec.R0
        gR0 = self.spec.gR[0]
        iz0 = int(np.argmin(np.abs(self.spec.gZ)))
        # pair interpolation weights on the Z=0 row
        P = pairs.shape[1]
        pr = t(np.nan_to_num(pairs, nan=1.0))            # (B,P,2)
        pvalid = t(np.isfinite(pairs).all(-1).astype(np.float32))  # (B,P)
        fj = (pr - gR0) / self.dR
        j0 = torch.clamp(fj.floor().long(), 0, nr - 2)
        w = fj - j0.to(dt)
        rowG = self.G[iz0 * nr:(iz0 + 1) * nr]           # (nr, Ns)
        Gin = (1 - w[..., 0, None]) * rowG[j0[..., 0]] + w[..., 0, None] * rowG[j0[..., 0] + 1]
        Gout = (1 - w[..., 1, None]) * rowG[j0[..., 1]] + w[..., 1, None] * rowG[j0[..., 1] + 1]
        Gd = Gin - Gout                                   # (B,P,Ns)
        r2d = (pr[..., 0] ** 2 - pr[..., 1] ** 2) / 2 * DV_SCALE  # dv column

        def chord(field, k):
            row = field[:, iz0, :]                         # (B,nr)
            a = row.gather(1, j0[..., k]); b = row.gather(1, j0[..., k] + 1)
            return (1 - w[..., k]) * a + w[..., k] * b

        if I0 is None:
            Rc = t(Rinit) if Rinit is not None else torch.full((B,), R0, device=dev, dtype=dt)
            g = torch.exp(-((Rs[None] - Rc[:, None]) ** 2 + (Zs[None] - Zt[:, None]) ** 2) / 0.05)
            I = Ip[:, None] * g / g.sum(1, keepdim=True)
        else:
            I = t(I0)
        c = t(c0) if c0 is not None else torch.zeros(B, device=dev, dtype=dt)
        u = t(np.tile(prior if u0 is None else u0, (B, 1)) if np.ndim(u0) < 2 else u0)
        fb = (self.RR ** 2)[None] * (self.ZZ[None] - Zt[:, None, None])
        vfield = (self.RR ** 2 / 2 * DV_SCALE)[None]      # psi of unit dv
        pw = t(prior_w); pu = t(prior)
        for it in range(n_iter):
            psi = psi_ext + (I @ self.G.T).reshape(B, nz, nr) + c[:, None, None] * fb + \
                u[:, 4, None, None] * vfield
            psi_ax, Rax, Zax = self.find_axis(psi)
            xp = self.find_xpoints(psi, Rax, Zax, psi_ax)
            (fu, vu, _, _), (fl, vl, _, _) = xp
            neg = torch.full_like(vu, -1e30)
            psi_b = torch.maximum(torch.where(fu, vu, neg), torch.where(fl, vl, neg))
            nox = ~(fu | fl)
            if nox.any():
                psi_b = torch.where(nox, psi_ax - 0.05 * psi_ax.abs().clamp_min(1e-3), psi_b)
            mask, psiN = self.core_mask(psi, psi_ax, psi_b, Rax, Zax, xp)
            pn = psiN.reshape(B, -1)[:, self.sidx].clamp(0, 1)
            ms = mask.reshape(B, -1)[:, self.sidx].to(dt)
            x1 = (1 - pn ** 2) * ms
            x2 = (1 - pn) ** 2 * ms
            basis = torch.stack([Rs / R0 * x1, Rs / R0 * x2, R0 / Rs * x1, R0 / Rs * x2], 1)
            norm = basis.sum(2, keepdim=True).clamp_min(1e-12)
            basis = basis / norm * Ip[:, None, None]       # (B,4,Ns) amps per unit coeff
            # vertical re-centre of each basis to Zt (shift by centroid of the prior mix)
            # rows
            A_ip = torch.cat([torch.ones(B, 4, device=dev, dtype=dt),
                              torch.zeros(B, 1, device=dev, dtype=dt)], 1)[:, None, :]  # sum=1
            b_ip = torch.ones(B, 1, device=dev, dtype=dt)
            phi = (psi_ax - psi_b).abs().clamp_min(1e-4)
            ext_in = chord(psi_ext + c[:, None, None] * fb, 0)
            ext_out = chord(psi_ext + c[:, None, None] * fb, 1)
            A_p = torch.cat([torch.einsum("bpn,bkn->bpk", Gd, basis), r2d[..., None]], 2)
            b_p = -(ext_in - ext_out)
            sc = (pvalid * w_pair / (phi[:, None] * pair_tol))[..., None]
            A = torch.cat([A_ip * 100.0, A_p * sc,
                           torch.diag_embed(pw).expand(B, 5, 5)], 1)
            bb = torch.cat([b_ip * 100.0, b_p * sc[..., 0], (pw * pu).expand(B, 5)], 1)
            A64, b64 = A.double(), bb.double()
            A64 = torch.nan_to_num(A64); b64 = torch.nan_to_num(b64)
            AtA = A64.transpose(1, 2) @ A64 + 1e-9 * torch.eye(5, device=dev, dtype=torch.float64)
            Atb = (A64.transpose(1, 2) @ b64[..., None])[..., 0]
            unew = torch.linalg.solve(AtA, Atb).to(dt)
            bad = ~torch.isfinite(unew).all(1)
            unew = torch.where(bad[:, None], pu.expand(B, 5), unew)
            Inew = torch.einsum("bk,bkn->bn", unew[:, :4], basis)
            Zc_new = (Inew * Zs[None]).sum(1) / Ip
            delta = Zt - Zc_new
            full = torch.zeros(B, Npix, device=dev, dtype=dt)
            full[:, self.sidx] = Inew
            full = self._shiftZ(full.reshape(B, nz, nr), delta).reshape(B, Npix)
            Inew = full[:, self.sidx]
            Inew = Ip[:, None] * Inew / Inew.sum(1, keepdim=True).clamp_min(1e-30)
            okI = torch.isfinite(Inew).all(1) & torch.isfinite(delta)
            Inew = torch.where(okI[:, None], Inew, I)
            delta = torch.where(okI, delta, torch.zeros_like(delta))
            c = c + kc * delta
            dI = (Inew - I).abs().sum(1) / Ip
            I = (1 - omega) * I + omega * Inew
            u = torch.where(okI[:, None], (1 - omega) * u + omega * unew, u)
            if verbose and (it % 10 == 0 or it == n_iter - 1):
                print(it, f"dI {dI.max().item():.2e} med {dI.median().item():.1e} "
                      f"dv med {u[:, 4].median().item() * DV_SCALE * 1e3:.2f} mT")
        psi = psi_ext + (I @ self.G.T).reshape(B, nz, nr) + c[:, None, None] * fb + \
            u[:, 4, None, None] * vfield
        psi_ax, Rax, Zax = self.find_axis(psi)
        xp = self.find_xpoints(psi, Rax, Zax, psi_ax)
        (fu, vu, Rxu, Zxu), (fl, vl, Rxl, Zxl) = xp
        neg = torch.full_like(vu, -1e30)
        psi_b = torch.maximum(torch.where(fu, vu, neg), torch.where(fl, vl, neg))
        # pair residuals (flux, relative to phi)
        res = ((chord(psi, 0) - chord(psi, 1)) / (psi_ax - psi_b).abs()[:, None]) * pvalid
        return dict(psi=psi, I=I, c=c, u=u, psi_ax=psi_ax, Rax=Rax, Zax=Zax, psi_b=psi_b,
                    xu=(fu, Rxu, Zxu, vu), xl=(fl, Rxl, Zxl, vl), dI=dI, pair_res=res,
                    psi_ext=psi_ext)
