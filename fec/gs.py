"""Batched free-boundary Grad-Shafranov solver on the EFIT 65x65 grid (torch, GPU).

Plasma current lives on the grid nodes (as in EFIT): node current I_k = J_phi(R_k,Z_k) dR dZ.
psi = psi_ext(coil currents) + Gp @ I_plasma + c * psi_fb  (Wb/rad, physical sign: Ip > 0
gives an axis MAXIMUM). Profile (FreeGS-style):
    J_phi = lam * (beta0 R/R0 + (1-beta0) R0/R) * (1 - psiN^am)^an   inside the LCFS
lam is fixed by Ip.  Vertical position: proportional feedback field c*R^2*(Z-Zt) keeps the
current centroid at Zt; an outer secant search on Zt drives c -> 0 (force-free with the
given coil currents).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from fec.green import greens_psi

MU0 = 4e-7 * math.pi


def plasma_green_matrix(gR, gZ, src_mask, nsub=4):
    """G[target_node, src] = psi at node per amp in source cell (cell-averaged)."""
    RR, ZZ = np.meshgrid(gR, gZ)
    dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
    tr, tz = RR.ravel(), ZZ.ravel()
    sidx = np.flatnonzero(src_mask.ravel())
    sr, sz = tr[sidx], tz[sidx]
    off = (np.arange(nsub) + 0.5) / nsub - 0.5
    G = np.zeros((tr.size, sidx.size))
    for a in off:
        for b in off:
            G += greens_psi(tr[:, None], tz[:, None], (sr + a * dR)[None, :],
                            (sz + b * dZ)[None, :])
    return G / nsub ** 2, sidx


def plasma_green_matrix_cached(name, gR, gZ, src_mask):
    from fec.paths import WORK
    import hashlib
    key = hashlib.sha1(np.concatenate([gR, gZ, src_mask.ravel().astype(float)]).tobytes()
                       ).hexdigest()[:12]
    f = WORK / "geometry" / f"Gp_{name}_{key}.npz"
    if f.exists():
        z = np.load(f)
        return z["G"], z["sidx"]
    G, sidx = plasma_green_matrix(gR, gZ, src_mask)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, G=G, sidx=sidx)
    return G, sidx


@dataclass
class MachineSpec:
    name: str
    gR: np.ndarray
    gZ: np.ndarray
    box: tuple            # (Rmin, Rmax, Zmin, Zmax) allowed plasma region
    axis_box: tuple       # region where the magnetic axis may sit
    R0: float
    keepout: list         # [(R, Z, radius)] nodes excluded (in-vessel coils)


class Solver:
    def __init__(self, spec: MachineSpec, Mext: np.ndarray, device="cuda", dtype=torch.float32):
        torch.backends.cuda.matmul.allow_tf32 = False
        self.spec = spec
        self.dev = device
        self.dt = dtype
        gR, gZ = spec.gR, spec.gZ
        self.nr, self.nz = len(gR), len(gZ)
        RR, ZZ = np.meshgrid(gR, gZ)
        self.dR, self.dZ = gR[1] - gR[0], gZ[1] - gZ[0]
        Rmin, Rmax, Zmin, Zmax = spec.box
        box = (RR >= Rmin) & (RR <= Rmax) & (ZZ >= Zmin) & (ZZ <= Zmax)
        for (r, z, rad) in spec.keepout:
            box &= (RR - r) ** 2 + (ZZ - z) ** 2 > rad ** 2
        a = spec.axis_box
        abox = (RR >= a[0]) & (RR <= a[1]) & (ZZ >= a[2]) & (ZZ <= a[3]) & box
        G, sidx = plasma_green_matrix_cached(spec.name, gR, gZ, box)
        t = lambda x: torch.as_tensor(x, device=device, dtype=dtype)
        self.G = t(G)                      # (Npix, Ns)
        self.sidx = torch.as_tensor(sidx, device=device)
        self.Mext = t(Mext)                # (Npix, C)
        self.RR, self.ZZ = t(RR), t(ZZ)
        self.box = torch.as_tensor(box, device=device)
        self.abox = torch.as_tensor(abox, device=device)
        self.Rs = self.RR.reshape(-1)[self.sidx]
        self.Zs = self.ZZ.reshape(-1)[self.sidx]

    # ------------------------------------------------------------------ helpers
    def _grad(self, psi):
        """Central-difference derivatives on (B,nz,nr)."""
        p = psi
        dR, dZ = self.dR, self.dZ
        pR = torch.zeros_like(p); pZ = torch.zeros_like(p)
        pR[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) / (2 * dR)
        pZ[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) / (2 * dZ)
        pRR = torch.zeros_like(p); pZZ = torch.zeros_like(p); pRZ = torch.zeros_like(p)
        pRR[:, :, 1:-1] = (p[:, :, 2:] - 2 * p[:, :, 1:-1] + p[:, :, :-2]) / dR ** 2
        pZZ[:, 1:-1, :] = (p[:, 2:, :] - 2 * p[:, 1:-1, :] + p[:, :-2, :]) / dZ ** 2
        pRZ[:, 1:-1, 1:-1] = (p[:, 2:, 2:] - p[:, 2:, :-2] - p[:, :-2, 2:] + p[:, :-2, :-2]) \
            / (4 * dR * dZ)
        return pR, pZ, pRR, pZZ, pRZ

    def _quad_extremum(self, psi, idx):
        """Quadratic refinement of a stationary point near flat node index idx (B,)."""
        B = psi.shape[0]
        nz, nr = self.nz, self.nr
        iz = torch.clamp(idx // nr, 1, nz - 2)
        ir = torch.clamp(idx % nr, 1, nr - 2)
        b = torch.arange(B, device=psi.device)
        f = lambda dz, dr: psi[b, iz + dz, ir + dr]
        f0 = f(0, 0)
        fr = (f(0, 1) - f(0, -1)) / 2
        fz = (f(1, 0) - f(-1, 0)) / 2
        frr = f(0, 1) - 2 * f0 + f(0, -1)
        fzz = f(1, 0) - 2 * f0 + f(-1, 0)
        frz = (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / 4
        det = frr * fzz - frz ** 2
        safe = det.abs() > 1e-30
        det_s = torch.where(safe, det, torch.ones_like(det))
        dr = torch.where(safe, -(fzz * fr - frz * fz) / det_s, torch.zeros_like(det))
        dz = torch.where(safe, -(frr * fz - frz * fr) / det_s, torch.zeros_like(det))
        dr = dr.clamp(-1, 1); dz = dz.clamp(-1, 1)
        val = f0 + fr * dr + fz * dz + 0.5 * (frr * dr ** 2 + 2 * frz * dr * dz + fzz * dz ** 2)
        R = self.spec.gR[0] + (ir.to(psi.dtype) + dr) * self.dR
        Z = self.spec.gZ[0] + (iz.to(psi.dtype) + dz) * self.dZ
        return val, R, Z

    def find_axis(self, psi):
        B = psi.shape[0]
        neg = torch.finfo(psi.dtype).min
        p = torch.where(self.abox, psi, torch.full_like(psi, neg)).reshape(B, -1)
        idx = torch.argmax(p, dim=1)
        return self._quad_extremum(psi, idx)

    def find_xpoints(self, psi, Rax, Zax, psi_ax):
        """Best saddle above and below the axis inside the box (by psi closest to axis)."""
        B = psi.shape[0]
        pR, pZ, pRR, pZZ, pRZ = self._grad(psi)
        # sign changes of pR and pZ across the 2x2 quad at (i,j)..(i+1,j+1)
        def changes(g):
            a, b_, c, d = g[:, :-1, :-1], g[:, :-1, 1:], g[:, 1:, :-1], g[:, 1:, 1:]
            mn = torch.minimum(torch.minimum(a, b_), torch.minimum(c, d))
            mx = torch.maximum(torch.maximum(a, b_), torch.maximum(c, d))
            return (mn <= 0) & (mx >= 0)
        hdet = pRR * pZZ - pRZ ** 2
        hq = (hdet[:, :-1, :-1] + hdet[:, :-1, 1:] + hdet[:, 1:, :-1] + hdet[:, 1:, 1:]) / 4
        cand = changes(pR) & changes(pZ) & (hq < 0)
        boxq = self.box[:-1, :-1] & self.box[1:, 1:] & self.box[:-1, 1:] & self.box[1:, :-1]
        cand &= boxq[None]
        # node index of quad's lower-left; value = mean psi of quad
        pq = (psi[:, :-1, :-1] + psi[:, :-1, 1:] + psi[:, 1:, :-1] + psi[:, 1:, 1:]) / 4
        Zq = (self.ZZ[:-1, :-1] + self.ZZ[1:, 1:]) / 2
        neg = torch.finfo(psi.dtype).min
        out = []
        for upper in (True, False):
            side = (Zq[None] > Zax[:, None, None] + 0.15) if upper else \
                   (Zq[None] < Zax[:, None, None] - 0.15)
            m = cand & side & (pq < psi_ax[:, None, None])
            v = torch.where(m, pq, torch.full_like(pq, neg)).reshape(B, -1)
            best = torch.argmax(v, dim=1)
            found = m.reshape(B, -1).gather(1, best[:, None])[:, 0]
            iz = best // (self.nr - 1); ir = best % (self.nr - 1)
            # refine at the nearest node of the quad with smallest |grad|
            g2 = pR ** 2 + pZ ** 2
            b = torch.arange(B, device=psi.device)
            cands = torch.stack([iz * self.nr + ir, iz * self.nr + ir + 1,
                                 (iz + 1) * self.nr + ir, (iz + 1) * self.nr + ir + 1], 1)
            gv = g2.reshape(B, -1).gather(1, cands)
            node = cands.gather(1, torch.argmin(gv, 1, keepdim=True))[:, 0]
            val, R, Z = self._quad_extremum(psi, node)
            out.append((found, val, R, Z))
        return out

    def core_mask(self, psi, psi_ax, psi_b, Rax, Zax, xpts, seed_mask=None):
        """Nodes inside LCFS: psiN<1, inside box, on the axis side of X-points, connected."""
        B = psi.shape[0]
        psiN = (psi_ax[:, None, None] - psi) / (psi_ax - psi_b)[:, None, None]
        allowed = (psiN < 1.0) & self.box[None]
        RR, ZZ = self.RR[None], self.ZZ[None]
        for (found, val, Rx, Zx) in xpts:
            dRx = (Rx - Rax)[:, None, None]; dZx = (Zx - Zax)[:, None, None]
            proj = (RR - Rax[:, None, None]) * dRx + (ZZ - Zax[:, None, None]) * dZx
            cut = proj < (dRx ** 2 + dZx ** 2)
            allowed &= torch.where(found[:, None, None], cut, torch.ones_like(cut))
        # flood fill from axis node (4-connectivity)
        iz = torch.round((Zax - self.spec.gZ[0]) / self.dZ).long().clamp(0, self.nz - 1)
        ir = torch.round((Rax - self.spec.gR[0]) / self.dR).long().clamp(0, self.nr - 1)
        seed = torch.zeros_like(allowed)
        seed[torch.arange(B, device=psi.device), iz, ir] = True
        if seed_mask is not None:
            seed = seed | seed_mask
        m = (seed & allowed).to(psi.dtype)[:, None]
        al = allowed.to(psi.dtype)[:, None]
        k = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=psi.dtype,
                         device=psi.device)[None, None]
        for step in range(100):
            m2 = (F.conv2d(m, k, padding=1) > 0).to(psi.dtype) * al
            if step % 4 == 3 and torch.equal(m2, m):
                break
            m = m2
        return m[:, 0] > 0, psiN

    # ------------------------------------------------------------------ solve
    def _shiftZ(self, Ig, d):
        """Shift node-current grid Ig (B,nz,nr) by d (B,) metres in Z (linear, mass-conserving)."""
        B = Ig.shape[0]
        x = d / self.dZ
        n = torch.floor(x)
        f = (x - n)[:, None, None]
        n = n.long()
        iz = torch.arange(self.nz, device=Ig.device)[None, :] - n[:, None]  # source index
        def take(idx):
            ok = (idx >= 0) & (idx < self.nz)
            g = Ig.gather(1, idx.clamp(0, self.nz - 1)[:, :, None].expand(B, self.nz, self.nr))
            return g * ok[:, :, None]
        return (1 - f) * take(iz) + f * take(iz - 1)

    def solve(self, Iext, Ip, prof, Zt, n_iter=150, omega=0.4, kc=0.05, I0=None, c0=None,
              Rinit=None, fix_z=True, verbose=False):
        """Iext (B,C) [A]; Ip (B,) [A] (>0); prof (B,3) beta0, am, an; Zt (B,).

        fix_z: rigidly re-centre the current at Zt each iteration and integrate the
        radial-field feedback c until no shift is needed (equilibrium at Zt with field c).
        """
        dev, dt = self.dev, self.dt
        t = lambda x: torch.as_tensor(x, device=dev, dtype=dt)
        Iext, Ip, prof, Zt = t(Iext), t(Ip), t(prof), t(Zt)
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
        fb = (self.RR ** 2)[None] * (self.ZZ[None] - Zt[:, None, None])  # (B,nz,nr)
        beta0, am, an = prof[:, 0:1], prof[:, 1:2], prof[:, 2:3]
        hist = []
        Npix = self.nz * self.nr
        prev_mask = None
        for it in range(n_iter):
            psi = psi_ext + (I @ self.G.T).reshape(B, self.nz, self.nr) + c[:, None, None] * fb
            psi_ax, Rax, Zax = self.find_axis(psi)
            xp = self.find_xpoints(psi, Rax, Zax, psi_ax)
            (fu, vu, _, _), (fl, vl, _, _) = xp
            neg = torch.full_like(vu, -1e30)
            psi_b = torch.maximum(torch.where(fu, vu, neg), torch.where(fl, vl, neg))
            nox = ~(fu | fl)
            if nox.any():
                psi_b = torch.where(nox, psi_ax - 0.05 * psi_ax.abs().clamp_min(1e-3), psi_b)
            seedm = prev_mask if (it % 10 != 0) else None
            mask, psiN = self.core_mask(psi, psi_ax, psi_b, Rax, Zax, xp, seed_mask=seedm)
            prev_mask = mask
            pn = psiN.reshape(B, -1)[:, self.sidx].clamp(0, 1)
            ms = mask.reshape(B, -1)[:, self.sidx].to(dt)
            shape = (beta0 * Rs[None] / R0 + (1 - beta0) * R0 / Rs[None]) *                 (1 - pn ** am).clamp_min(0) ** an * ms
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
            if verbose and (it % 10 == 0 or it == n_iter - 1):
                print(it, f"dI {dI.max().item():.2e} |delta| {delta.abs().max().item():.2e} "
                      f"c {c.abs().max().item():.3e}")
            hist.append(dI.max().item())
        psi = psi_ext + (I @ self.G.T).reshape(B, self.nz, self.nr) + c[:, None, None] * fb
        psi_ax, Rax, Zax = self.find_axis(psi)
        xp = self.find_xpoints(psi, Rax, Zax, psi_ax)
        (fu, vu, Rxu, Zxu), (fl, vl, Rxl, Zxl) = xp
        neg = torch.full_like(vu, -1e30)
        psi_b = torch.maximum(torch.where(fu, vu, neg), torch.where(fl, vl, neg))
        return dict(psi=psi, I=I, c=c, psi_ax=psi_ax, Rax=Rax, Zax=Zax, psi_b=psi_b,
                    xu=(fu, Rxu, Zxu, vu), xl=(fl, Rxl, Zxl, vl), dI=dI, delta=delta,
                    hist=hist, psi_ext=psi_ext)
