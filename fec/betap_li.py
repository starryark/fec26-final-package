"""beta_p / l_i constraints for the Thomson-pair kinetic fit (kfit2), validated with labels on
DIII-D before any MAST port.  Study note: docs/research/BETAP_LI_CONSTRAINTS_20261008.md.

kfit2 (fec.mast_select_v2 / v3, DIII-D analogue fec.d3d_physics.kfit_d3d) fits 4 basis currents
(R/R0 and R0/R times (1 - psiN^2), (1 - psiN)^2) plus a uniform-Bz nuisance to equal-Te Thomson
pairs.  Thomson symmetry is saturated on DIII-D, so this module adds two scalar constraint rows
to the same LM system:

  beta_p(u) = 2 mu0 <p>_V / (mu0 Ip / lp)^2,  p from the R/R0 (pressure, J = R p') columns
  l_i(u)    = <Bp^2>_V / (mu0 Ip / lp)^2       (scorer / EFIT li(2) normalisation)

both evaluated inside the Picard loop on the current iterate's plasma (core mask, ray-cast LCFS
perimeter lp, Bp from the total flux of the basis mix), with targets from machine-agnostic
regressions fitted on DIII-D TRAIN only and weights 1/sigma from their TRAIN-CV residuals.

Evidence boundary: DIII-D TRAIN "cal" set (fec.d3d_physics subsets) for every fit and design
probe; the 40-shot VAL subset once, after the freeze; TEST never; MAST inputs and our own
label-free mast_v3_loff geometry for feature ranges only; MAST labels never.

    python -B fec/betap_li.py labels --set cal            # TRAIN beta_p / li targets (CPU)
    python -B fec/gpu_slot.py run --owner bpli -- python -B fec/betap_li.py stage1 --set cal
    python -B fec/betap_li.py features --pop cal|val|mast
    python -B fec/betap_li.py fit                         # TRAIN-only selection (+ MAST ranges)
    python -B fec/gpu_slot.py run --owner bpli -- python -B fec/betap_li.py stage2 --set cal
    python -B fec/betap_li.py score --set cal|val --names ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

MU0 = 4e-7 * math.pi
QE = 1.602176634e-19
OUT = WORK / "betap_li"

# ray-cast LCFS (in-loop perimeter): rays from the magnetic axis
RAY_NT = 64           # rays
RAY_DS = 0.01         # m, step along a ray
RAY_SMAX = 1.7        # m, longest ray (DIII-D and MAST plasmas are < 1.45 m from the axis)


# ============================================================================ in-loop functionals
def _bilinear_grid(S, field, Rq, Zq):
    """Bilinear samples of field (B, nz, nr) at (Rq, Zq) (B, ...); also the in-grid mask."""
    import torch
    B = field.shape[0]
    gR0, gZ0 = float(S.spec.gR[0]), float(S.spec.gZ[0])
    fr = (Rq - gR0) / S.dR
    fz = (Zq - gZ0) / S.dZ
    ing = (fr >= 0) & (fr <= S.nr - 1) & (fz >= 0) & (fz <= S.nz - 1)
    j = fr.floor().long().clamp(0, S.nr - 2)
    i = fz.floor().long().clamp(0, S.nz - 2)
    wr = (fr - j.to(fr.dtype)).clamp(0, 1)
    wz = (fz - i.to(fz.dtype)).clamp(0, 1)
    f = field.reshape(B, -1)
    shp = Rq.shape

    def g(ii, jj):
        return f.gather(1, (ii * S.nr + jj).reshape(B, -1)).reshape(shp)
    v = (g(i, j) * (1 - wr) * (1 - wz) + g(i, j + 1) * wr * (1 - wz) +
         g(i + 1, j) * (1 - wr) * wz + g(i + 1, j + 1) * wr * wz)
    return v, ing


def ray_boundary(S, psiN, Rax, Zax, mask=None, nt=RAY_NT, ds=RAY_DS, smax=RAY_SMAX):
    """LCFS polygon (B, nt, 2) and perimeter lp (B,) by ray casting psiN = 1 from the axis.

    Along each ray the first crossing of psiN >= 1 (or of the grid edge) is located by linear
    interpolation between samples.  With the core mask, nodes outside it that still have
    psiN < 1 (private-flux region beyond an X-point, disconnected islands) are reflected to
    2 - psiN > 1, so rays close at the X-point instead of leaking down a divertor leg (on the
    coarse grid the sampled psiN along a ray through the saddle can stay just below 1)."""
    import torch
    dev, dt = psiN.device, psiN.dtype
    B = psiN.shape[0]
    if mask is not None:
        psiN = torch.where(mask | (psiN >= 1), psiN, 2 - psiN)
    th = (torch.arange(nt, device=dev, dtype=dt) + 0.5) * (2 * math.pi / nt)
    ns = int(round(smax / ds))
    s = torch.arange(1, ns + 1, device=dev, dtype=dt) * ds
    Rq = Rax[:, None, None] + s[None, None, :] * torch.cos(th)[None, :, None]
    Zq = Zax[:, None, None] + s[None, None, :] * torch.sin(th)[None, :, None]
    v, ing = _bilinear_grid(S, psiN, Rq, Zq)
    v = torch.where(ing & torch.isfinite(v), v, torch.full_like(v, 2.0))
    v = torch.cat([v, torch.full_like(v[..., :1], 2.0)], -1)            # guaranteed crossing
    hit = v >= 1.0
    k = torch.argmax(hit.to(torch.int8), -1)                            # first crossing (B, nt)
    vk = v.gather(-1, k[..., None])[..., 0]
    vprev = torch.where(k > 0, v.gather(-1, (k - 1).clamp_min(0)[..., None])[..., 0],
                        torch.zeros_like(vk))
    frac = ((1.0 - vprev) / (vk - vprev).clamp_min(1e-9)).clamp(0, 1)
    sb = (k.to(dt) + frac) * ds                                         # s_k = (k+1) ds
    sb = sb.clamp(max=smax)
    P = torch.stack([Rax[:, None] + sb * torch.cos(th)[None], Zax[:, None] + sb * torch.sin(th)[None]], -1)
    lp = (P - P.roll(-1, 1)).norm(dim=-1).sum(1)
    return P, lp


def grad_psi(S, psi):
    """Central-difference dpsi/dR, dpsi/dZ (B, nz, nr); one-sided on the grid border."""
    import torch
    pR = torch.zeros_like(psi)
    pZ = torch.zeros_like(psi)
    pR[:, :, 1:-1] = (psi[:, :, 2:] - psi[:, :, :-2]) / (2 * S.dR)
    pR[:, :, 0] = (psi[:, :, 1] - psi[:, :, 0]) / S.dR
    pR[:, :, -1] = (psi[:, :, -1] - psi[:, :, -2]) / S.dR
    pZ[:, 1:-1, :] = (psi[:, 2:, :] - psi[:, :-2, :]) / (2 * S.dZ)
    pZ[:, 0, :] = (psi[:, 1, :] - psi[:, 0, :]) / S.dZ
    pZ[:, -1, :] = (psi[:, -1, :] - psi[:, -2, :]) / S.dZ
    return pR, pZ


def volume_weights(S, mask):
    """dV = 2 pi R dR dZ on core-mask nodes (B, nz, nr) and the plasma volume V (B,)."""
    w = mask.to(S.dt) * (2 * math.pi * S.dR * S.dZ) * S.RR[None]
    return w, w.sum((1, 2)).clamp_min(1e-12)


def li_of(S, psi, mask, Ip, lp):
    """li(2) = <Bp^2>_V / (mu0 Ip / lp)^2 of maps psi (B, nz, nr) on the core mask."""
    pR, pZ = grad_psi(S, psi)
    w, V = volume_weights(S, mask)
    W = ((pR ** 2 + pZ ** 2) / S.RR[None] ** 2 * w).sum((1, 2))
    return W / V / (MU0 * Ip / lp) ** 2


def pressure_integrals(pn_nodes, ms_nodes, Rn, ncol=2):
    """Volume averages <X_k(psiN)> (B, 2) of the pressure shape integrals of the two R/R0
    kfit columns x1 = 1 - s^2, x2 = (1 - s)^2: X_k(psiN) = int_psiN^1 x_k(s) ds; plus the
    normalisation sums S_k = sum (R/R0) x_k (B, 2) (as the kfit basis normalisation).
    ncol=3 (kfit nbasis=6): + the broad column x3 = s (1 - s), X3 = 1/6 - x^2/2 + x^3/3."""
    import torch
    x = pn_nodes.clamp(0, 1)
    X1 = 2.0 / 3.0 - x + x ** 3 / 3.0
    X2 = (1 - x) ** 3 / 3.0
    wv = ms_nodes * Rn[None]
    den = wv.sum(1).clamp_min(1e-12)
    if ncol == 3:
        X3 = 1.0 / 6.0 - x ** 2 / 2.0 + x ** 3 / 3.0
        return torch.stack([(X1 * wv).sum(1) / den, (X2 * wv).sum(1) / den,
                            (X3 * wv).sum(1) / den], 1)
    return torch.stack([(X1 * wv).sum(1) / den, (X2 * wv).sum(1) / den], 1)


def pressure_cols(nbasis):
    """Indices of the R/R0 (pressure, J = R p') columns of a kfit weight vector u."""
    return slice(0, 2) if nbasis == 4 else [0, 1, 4]


def betap_of_coeffs(S, u_p, pn_nodes, ms_nodes, Ip, phi, lp):
    """beta_p of a kfit mix: u_p (B, 2) weights of the R/R0 columns (each carrying Ip);
    (B, 3) for kfit nbasis=6 (columns x1, x2, x3 = s (1 - s)).

    p'(psi) of column k = Ip x_k / (R0 S_k dA) (J = R p'), p = phi int_psiN^1 p' ds,
    <p>_V over the core-mask source nodes, beta_p = 2 mu0 <p> / (mu0 Ip / lp)^2.
    Returns (beta_p (B,), dbeta_p/du_p (B, 2 or 3))."""
    import torch
    Rs, R0 = S.Rs, S.spec.R0
    dA = S.dR * S.dZ
    x = pn_nodes.clamp(0, 1)
    if u_p.shape[1] == 3:
        Sk = torch.stack([((Rs[None] / R0) * (1 - x ** 2) * ms_nodes).sum(1),
                          ((Rs[None] / R0) * (1 - x) ** 2 * ms_nodes).sum(1),
                          ((Rs[None] / R0) * x * (1 - x) * ms_nodes).sum(1)], 1).clamp_min(1e-12)
        Xv = pressure_integrals(pn_nodes, ms_nodes, Rs, ncol=3)
    else:
        Sk = torch.stack([((Rs[None] / R0) * (1 - x ** 2) * ms_nodes).sum(1),
                          ((Rs[None] / R0) * (1 - x) ** 2 * ms_nodes).sum(1)], 1).clamp_min(1e-12)
        Xv = pressure_integrals(pn_nodes, ms_nodes, Rs)
    amp = Ip[:, None] / (R0 * Sk * dA)                         # p' amplitude per unit u [A/m^3]
    pk = phi[:, None] * amp * Xv                               # <p> per unit u [Pa]
    g = 2 * MU0 * pk / (MU0 * Ip / lp)[:, None] ** 2
    return (g * u_p).sum(1), g


def plasma_state(S, psi):
    """Axis, X-points, boundary flux (solver boundary(), final=True), core mask and psiN of maps
    psi (B, nz, nr) with the axis at a maximum.  Frames without an X-point get the solver's
    in-loop fallback boundary (psi_ax - 5%), as inside kfit."""
    import torch
    psi_ax, Rax, Zax = S.find_axis(psi)
    xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
    psi_b, _ = S.boundary(psi, psi_ax, Rax, Zax, xp)
    mask, psiN = S.core_mask(psi, psi_ax, psi_b, Rax, Zax, xp)
    return dict(psi_ax=psi_ax, psi_b=psi_b, Rax=Rax, Zax=Zax, mask=mask, psiN=psiN,
                nox=~(xp[0][0] | xp[1][0]))


# ============================================================================ DIII-D labels
def _d3d():
    from fec import d3d_physics as dp
    return dp


def lcfs_geometry(lcfs, n):
    """(a, perimeter, Rgeo, kappa) of label LCFS polygons (n_frames, M, 2) with lengths n."""
    out = np.full((len(n), 4), np.nan)
    for i, m in enumerate(np.asarray(n, int)):
        if m < 8:
            continue
        P = np.asarray(lcfs[i, :m], float)
        r, z = P[:, 0], P[:, 1]
        a = 0.5 * (r.max() - r.min())
        lp = float(np.sum(np.hypot(*(P - np.roll(P, -1, 0)).T)))
        out[i] = a, lp, 0.5 * (r.max() + r.min()), (z.max() - z.min()) / (2 * a)
    return out


def label_targets(set_name="cal", chunk=512):
    """beta_p and li of the EFIT (label) equilibria, two ways each (TRAIN: targets; VAL: only
    after the freeze, as diagnostics).

    *_F: the in-loop functionals of this module applied to the EFIT map (solver axis /
         X-point / core mask, ray-cast perimeter, Ip = label current sum); beta_p from the
         R p' part of a per-frame LS fit of the label node currents with
         J = sum_m a_m psiN^m R/R0 + c_m psiN^m R0/R (m <= 3).
    *_E: EFIT label li; beta_p from the betaN label and the EFIT LCFS:
         beta_t = betaN/100 Ip[MA] / (a B0), beta_p = beta_t B0^2 lp^2 / (mu0 Ip)^2,
         B0 = F / 1.6955 m (DIII-D BCENTR convention), a, lp of the label LCFS polygon."""
    import torch
    dp = _d3d()
    inp = dp.load_set(set_name, labels=True)
    n = len(inp["Ip"])
    S = dp.make_bare_solver(device="cpu")
    S.dt = torch.float64
    for k in ("G", "RR", "ZZ", "Rs", "Zs"):
        setattr(S, k, getattr(S, k).double())
    sidx = S.sidx.numpy()
    R0 = S.spec.R0
    dA = S.dR * S.dZ
    Ilab = inp["lab_Inode"].reshape(n, -1).astype(np.float64)
    Ip_lab = Ilab.sum(1)
    res = {k: np.full(n, np.nan) for k in ("li_F", "betap_F", "lp_F", "V_F", "fit_rel",
                                           "phi_F", "pp_frac_F")}
    res["nox_F"] = np.zeros(n, bool)
    t0 = time.time()
    for a in range(0, n, chunk):
        sl = slice(a, min(n, a + chunk))
        B = sl.stop - sl.start
        psi = torch.as_tensor(inp["lab_psi"][sl], dtype=torch.float64)
        st = plasma_state(S, psi)
        P, lp = ray_boundary(S, st["psiN"], st["Rax"], st["Zax"], st["mask"])
        ip = torch.as_tensor(Ip_lab[sl])
        li = li_of(S, psi, st["mask"], ip, lp)
        _, V = volume_weights(S, st["mask"])
        pn = st["psiN"].reshape(B, -1)[:, sidx].clamp(0, 1).numpy()
        ms = st["mask"].reshape(B, -1)[:, sidx].numpy()
        Rn = S.Rs.numpy()
        phi = (st["psi_ax"] - st["psi_b"]).numpy()
        for j in range(B):
            i = sl.start + j
            I = Ilab[i, sidx]
            m = I != 0
            if m.sum() < 10 or not np.isfinite(phi[j]) or phi[j] <= 0:
                continue
            x = pn[j][m]
            Pw = np.stack([x ** k for k in range(4)], 1)
            A = np.concatenate([Pw * (Rn[m] / R0)[:, None], Pw * (R0 / Rn[m])[:, None]], 1)
            sol = np.linalg.lstsq(A, I[m], rcond=None)[0]
            res["fit_rel"][i] = np.linalg.norm(A @ sol - I[m]) / np.linalg.norm(I[m])
            res["pp_frac_F"][i] = float((A[:, :4] @ sol[:4]).sum() / I[m].sum())
            mm = ms[j] > 0
            xv = pn[j][mm]
            pint = sum(sol[k] * (1 - xv ** (k + 1)) / (k + 1) for k in range(4))
            pav = phi[j] / (R0 * dA) * float(np.sum(pint * Rn[mm]) / np.sum(Rn[mm]))
            res["betap_F"][i] = 2 * MU0 * pav / (MU0 * Ip_lab[i] / float(lp[j])) ** 2
        res["li_F"][sl] = li.numpy()
        res["lp_F"][sl] = lp.numpy()
        res["V_F"][sl] = V.numpy()
        res["phi_F"][sl] = phi
        res["nox_F"][sl] = st["nox"].numpy()
        print(f"labels {set_name} {sl.stop}/{n} {time.time() - t0:.0f}s", flush=True)
    geo = lcfs_geometry(inp["lab_lcfs"], inp["lab_lcfs_n"])
    a_E, lp_E = geo[:, 0], geo[:, 1]
    F = MU0 * dp.N_TF * inp["bt"] * 1e3 / (2 * np.pi)
    B0 = F / dp.R0_D3D
    with np.errstate(all="ignore"):
        beta_t = inp["lab_betan"] / 100 * (Ip_lab / 1e6) / (a_E * B0)
        res["betap_E"] = beta_t * B0 ** 2 * lp_E ** 2 / (MU0 * Ip_lab) ** 2
    res.update(li_E=inp["lab_li"].astype(np.float64), a_E=a_E, lp_E=lp_E, Rgeo_E=geo[:, 2],
               kappa_E=geo[:, 3], Ip_lab=Ip_lab, idx=inp["idx"], shot=inp["shot"],
               betan=inp["lab_betan"].astype(np.float64))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"targets_{set_name}.npz", **res)
    ok = np.isfinite(res["li_F"]) & np.isfinite(res["li_E"])
    okb = np.isfinite(res["betap_F"]) & np.isfinite(res["betap_E"]) & (res["betap_E"] > 0)
    summ = dict(n=n, li_corr=float(np.corrcoef(res["li_F"][ok], res["li_E"][ok])[0, 1]),
                li_ratio_med=float(np.median(res["li_F"][ok] / res["li_E"][ok])),
                betap_corr=float(np.corrcoef(res["betap_F"][okb], res["betap_E"][okb])[0, 1]),
                betap_ratio_med=float(np.median(res["betap_F"][okb] / res["betap_E"][okb])),
                lp_ratio_med=float(np.nanmedian(res["lp_F"] / lp_E)),
                fit_rel_med=float(np.nanmedian(res["fit_rel"])),
                nox_frac=float(res["nox_F"].mean()),
                li_E_pct=np.nanpercentile(res["li_E"], [1, 50, 99]).tolist(),
                betap_E_pct=np.nanpercentile(res["betap_E"], [1, 50, 99]).tolist(),
                betap_F_pct=np.nanpercentile(res["betap_F"], [1, 50, 99]).tolist())
    (OUT / f"targets_{set_name}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return res


# ============================================================================ constrained kfit
class PointPairs:
    """fec.d3d_physics.kfit_d3d pair rows: pairs (B, P, 2, 2) = [pair, side, (R, Z)] points,
    bilinear on the grid (NaN rows ignored).  Same operations as kfit_d3d."""

    def __init__(self, S, pairs, t):
        import torch
        from fec.d3d_physics import _pt_index
        from fec.mast_select_v2 import KF_DV
        dev = S.dev
        R0 = S.spec.R0
        pairs = np.asarray(pairs, float)
        self.pvalid = t(np.isfinite(pairs).all((-1, -2)).astype(np.float32))
        pts = np.where(np.isfinite(pairs), pairs, np.array([R0, 0.0]))
        idx_np, w_np = _pt_index(S, pts)
        self.idx = torch.as_tensor(idx_np, device=dev)
        self.w = t(w_np)
        Gpt = None
        for q in range(4):
            term = S.G[self.idx[..., q]] * self.w[..., q, None]
            Gpt = term if Gpt is None else Gpt + term
        self.Gd = Gpt[:, :, 0] - Gpt[:, :, 1]
        del Gpt, term
        pr = t(pts[..., 0])
        self.r2d = (pr[..., 0] ** 2 - pr[..., 1] ** 2) / 2 * KF_DV

    def val(self, field, side):
        B = field.shape[0]
        f = field.reshape(B, -1)
        ii = self.idx[:, :, side].reshape(B, -1)
        return (f.gather(1, ii).reshape(self.idx[:, :, side].shape) * self.w[:, :, side]).sum(-1)


class RowPairs:
    """fec.mast_select_v2.kfit2 / v3.kfit3 pair rows: pairs (B, P, 2) inboard/outboard radii
    on the grid row nearest Z = 0 (NaN rows ignored).  Same operations as kfit3."""

    def __init__(self, S, pairs, t):
        from fec.mast_select_v2 import KF_DV
        gR0 = S.spec.gR[0]
        self.iz0 = int(np.argmin(np.abs(S.spec.gZ)))
        nr = S.nr
        pr = t(np.nan_to_num(pairs, nan=1.0))
        self.pvalid = t(np.isfinite(pairs).all(-1).astype(np.float32))
        fj = (pr - gR0) / S.dR
        import torch
        self.j0 = torch.clamp(fj.floor().long(), 0, nr - 2)
        self.wj = fj - self.j0.to(S.dt)
        rowG = S.G[self.iz0 * nr:(self.iz0 + 1) * nr]
        j0, w = self.j0, self.wj
        Gin = (1 - w[..., 0, None]) * rowG[j0[..., 0]] + w[..., 0, None] * rowG[j0[..., 0] + 1]
        Gout = (1 - w[..., 1, None]) * rowG[j0[..., 1]] + w[..., 1, None] * rowG[j0[..., 1] + 1]
        self.Gd = Gin - Gout
        self.r2d = (pr[..., 0] ** 2 - pr[..., 1] ** 2) / 2 * KF_DV

    def val(self, field, k):
        row = field[:, self.iz0, :]
        a = row.gather(1, self.j0[..., k]); b = row.gather(1, self.j0[..., k] + 1)
        return (1 - self.wj[..., k]) * a + self.wj[..., k] * b


def constraint_rows(S, u, basis, ext, vfield, pn, ms, mask, psiN, Rax, Zax, Ip, phi, cons):
    """Linearised beta_p / li rows (B, 2, 5) and right-hand sides (B, 2) at the mix u.

    beta_p is linear in the R/R0 weights u[:, :2] for the frozen plasma (geometry of the
    iterate); li is quadratic in u through the flux of the basis mix
    psi(u) = ext + sum_k u_k G basis_k + u_4 vfield, Gauss-Newton linearised at u.
    Rows are scaled by w / sigma per frame; frames with a NaN target get zero rows.
    kfit nbasis=6 (u (B, 7), basis (B, 6, Ns)): rows (B, 2, 7), beta_p over the R/R0 columns
    {0, 1, 4}, li over all 6 bases + dv (u_6 vfield)."""
    import torch
    dev, dt = S.dev, S.dt
    B = u.shape[0]
    nb = u.shape[1] - 1
    nz, nr = S.nz, S.nr
    _, lp = ray_boundary(S, psiN, Rax, Zax, mask)
    rows = torch.zeros(B, 2, nb + 1, device=dev, dtype=dt)
    rhs = torch.zeros(B, 2, device=dev, dtype=dt)
    diag = {"lp": lp}
    if cons.get("w_bp", 0) > 0:
        pc = pressure_cols(nb)
        bp, g = betap_of_coeffs(S, u[:, pc], pn, ms, Ip, phi, lp)
        s = cons["bp_scale"]                                     # w / sigma (B,), 0 = off
        rows[:, 0, pc] = g * s[:, None]
        rhs[:, 0] = cons["bp_t"] * s
        diag["bp"] = bp
    if cons.get("w_li", 0) > 0:
        Psi = (basis.reshape(B * nb, -1) @ S.G.T).reshape(B, nb, nz, nr)
        psi_u = ext + torch.einsum("bk,bkzr->bzr", u[:, :nb], Psi) + u[:, nb, None, None] * vfield
        gR0, gZ0 = grad_psi(S, psi_u)
        gRk, gZk = grad_psi(S, torch.cat([Psi.reshape(B * nb, nz, nr), vfield], 0))
        gRv, gZv = gRk[B * nb:], gZk[B * nb:]
        gRk = torch.cat([gRk[:B * nb].reshape(B, nb, nz, nr), gRv.expand(B, 1, nz, nr)], 1)
        gZk = torch.cat([gZk[:B * nb].reshape(B, nb, nz, nr), gZv.expand(B, 1, nz, nr)], 1)
        w, V = volume_weights(S, mask)
        wr = w / S.RR[None] ** 2
        W0 = ((gR0 ** 2 + gZ0 ** 2) * wr).sum((1, 2))
        dW = 2 * ((gR0[:, None] * gRk + gZ0[:, None] * gZk) * wr[:, None]).sum((2, 3))   # (B,nb+1)
        nrm = V * (MU0 * Ip / lp) ** 2
        li0 = W0 / nrm
        d = dW / nrm[:, None]
        s = cons["li_scale"]
        rows[:, 1] = d * s[:, None]
        rhs[:, 1] = (cons["li_t"] - li0 + (d * u).sum(1)) * s
        diag["li"] = li0
    return rows, rhs, diag


def kfit_bpli(S, Iext, Ip, Zt, pairs, n_iter=120, omega=0.3, kc=0.05, I0=None, c0=None,
              prior=None, w_prior=(0.3, 0.3, 0.3, 0.3, 1.0), lm=5.0, pair_tol=0.02,
              dv_max=2.0, fit_dv=True, cfix=False, rinit=None, cons=None, nbasis=4):
    """kfit2 with optional beta_p / li constraint rows (module doc).

    pairs (B,P,2,2) points -> fec.d3d_physics.kfit_d3d numerics; (B,P,2) radii on the Z ~ 0
    row -> fec.mast_select_v3.kfit3 numerics (cfix as kfit3; rinit = initial current centre
    when I0 is None: R0 for kfit_d3d, 0.9 m for kfit2/kfit3).  With cons=None the loop is
    operation-for-operation kfit_d3d / kfit3 (the final beta_p / li diagnostics are computed
    after the loop and do not touch the solution).

    cons: dict(bp_t, bp_sig, li_t, li_sig (B,) targets / 1-sigma (NaN = no row for that
    frame), w_bp, w_li global weights).  Rows: w (value(u) - target) / sigma.
    nbasis=6: + R/R0 x(1-x), R0/R x(1-x) (fec.mast_select_v3.kfit_basis_args; beta_p row over
    the R/R0 columns {0, 1, 4}, li row over all columns)."""
    import torch
    from fec.mast_select_v2 import KF_DV, KF_PRIOR
    from fec.mast_select_v3 import kfit_basis_args, kfit_basis_stack
    prior = KF_PRIOR if prior is None else prior
    nb = int(nbasis)
    prior, w_prior = kfit_basis_args(nb, prior, w_prior)
    K = nb + 1
    dev, dt = S.dev, S.dt
    t = lambda x: torch.as_tensor(x, device=dev, dtype=dt)  # noqa: E731
    Iext, Ip, Zt = t(Iext), t(Ip), t(Zt)
    B = Iext.shape[0]
    nz, nr = S.nz, S.nr
    Npix = nz * nr
    psi_ext = (Iext @ S.Mext.T).reshape(B, nz, nr)
    Rs, Zs = S.Rs, S.Zs
    R0 = S.spec.R0
    pairs = np.asarray(pairs, float)
    pm = PointPairs(S, pairs, t) if pairs.ndim == 4 else RowPairs(S, pairs, t)
    Gd, r2d, pvalid = pm.Gd, pm.r2d, pm.pvalid
    if I0 is None:
        rc = R0 if rinit is None else rinit
        g = torch.exp(-((Rs[None] - rc) ** 2 + (Zs[None] - Zt[:, None]) ** 2) / 0.05)
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
    cs = None
    if cons is not None and (cons.get("w_bp", 0) > 0 or cons.get("w_li", 0) > 0):
        cs = dict(w_bp=cons.get("w_bp", 0), w_li=cons.get("w_li", 0))
        for k in ("bp", "li"):
            tg = np.asarray(cons.get(f"{k}_t", np.full(B, np.nan)), float)
            sg = np.asarray(cons.get(f"{k}_sig", np.full(B, np.nan)), float)
            ok = np.isfinite(tg) & np.isfinite(sg) & (sg > 0)
            cs[f"{k}_t"] = t(np.where(ok, tg, 0.0))
            cs[f"{k}_scale"] = t(np.where(ok, cs[f"w_{k}"] / np.where(ok, sg, 1.0), 0.0))
    prev_mask = None
    dI = torch.zeros(B, device=dev, dtype=dt)
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
        b_p = -(pm.val(ext, 0) - pm.val(ext, 1))
        sc = (pvalid / (phi[:, None] * pair_tol))[..., None]
        A_ip = torch.cat([torch.ones(B, 1, nb, device=dev, dtype=dt),
                          torch.zeros(B, 1, 1, device=dev, dtype=dt)], 2)
        A = torch.cat([A_ip * 100.0, A_p * sc, torch.diag_embed(pw).expand(B, K, K),
                       lm * torch.eye(K, device=dev, dtype=dt).expand(B, K, K)], 1)
        bb = torch.cat([torch.full((B, 1), 100.0, device=dev, dtype=dt), b_p * sc[..., 0],
                        (pw * pu).expand(B, K), lm * u], 1)
        if cs is not None:
            crow, crhs, _ = constraint_rows(S, u, basis, ext, vfield, pn, ms, mask, psiN, Rax,
                                            Zax, Ip, phi, cs)
            A = torch.cat([A, crow], 1)
            bb = torch.cat([bb, crhs], 1)
        A64 = torch.nan_to_num(A.double())
        b64 = torch.nan_to_num(bb.double())
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
    res = ((pm.val(psi, 0) - pm.val(psi, 1)) / (psi_ax - psi_b).abs()[:, None]) * pvalid
    out = dict(psi=psi, I=I, c=c, u=u, psi_ax=psi_ax, Rax=Rax, Zax=Zax, psi_b=psi_b, dI=dI,
               pair_res=res, dv=u[:, nb] * KF_DV, limited=limited)
    out.update(final_betap_li(S, psi, u, Ip))
    return out


def final_betap_li(S, psi, u, Ip):
    """Diagnostics of a finished fit: beta_p of the R/R0 weights and li of the map (in-loop
    definitions, in-loop boundary fallback for frames without an X-point)."""
    import torch
    with torch.no_grad():
        st = plasma_state(S, psi)
        B = psi.shape[0]
        pn = st["psiN"].reshape(B, -1)[:, S.sidx].clamp(0, 1)
        ms = st["mask"].reshape(B, -1)[:, S.sidx].to(S.dt)
        _, lp = ray_boundary(S, st["psiN"], st["Rax"], st["Zax"], st["mask"])
        phi = (st["psi_ax"] - st["psi_b"]).abs().clamp_min(1e-4)
        bp, _ = betap_of_coeffs(S, u[:, pressure_cols(u.shape[1] - 1)], pn, ms, Ip, phi, lp)
        li = li_of(S, psi, st["mask"], Ip, lp)
    return dict(fit_betap=bp, fit_li=li, fit_lp=lp)


# ============================================================================ DIII-D runs
BASE = "KB_A1_ffz"                      # plain kfit_d3d analogue (stage 1, no constraints)
REF_TAGS = {"cal": "betapli_cal_e2"}    # scorer reference cache for TRAIN cal frames


def bind_dirs(set_name):
    """Point fec.d3d_physics' prediction / evaluation stores at this study's directories."""
    dp = _d3d()
    dp.PRED = OUT / f"pred_{set_name}"
    dp.EVAL = OUT / f"eval_{set_name}"
    if set_name in REF_TAGS:
        dp.REF_TAG = REF_TAGS[set_name]
    return dp


def load_inputs(set_name, limit_shots=0):
    """Inputs of a frozen fec.d3d_physics subset; optionally the first limit_shots shots."""
    dp = _d3d()
    inp = dp.load_set(set_name)
    if limit_shots:
        n = len(inp["Ip"])
        keep = np.isin(inp["shot"], np.unique(inp["shot"])[:limit_shots])
        m = int(keep.sum())
        assert keep[:m].all()
        inp = {k: (v[:m] if isinstance(v, np.ndarray) and v.ndim and len(v) == n else v)
               for k, v in inp.items()}
    return inp


def run_stage1(set_name, max_frames=1024):
    """Stage 1 (label-free): v2 force-free height (FFZ_PROF secant, Rinit R0) and the plain
    kfit_d3d fit, exactly as fec.d3d_physics run_family F -> KF_A1_ffz.  Saves the KB variant
    and the warm-start state (Zt, Zt = 0 solution current / feedback) for stage 2."""
    import torch
    from fec.mast_select_v2 import FFZ_PROF, force_free_z, shot_chunks
    dp = bind_dirs(set_name)
    inp = load_inputs(set_name)
    n = len(inp["Ip"])
    S, curf = dp.make_solver("A1")
    Iall = curf(inp)
    Ip = np.maximum(inp["Ip"], 1e5)
    pairs = dp.all_pairs(inp)
    st = dp.Store(n)
    Ns = int(S.Rs.shape[0])
    warm = dict(I0=np.zeros((n, Ns), np.float32), c0=np.zeros(n, np.float32), Zt=np.zeros(n))
    t0 = time.time()
    for sl in shot_chunks(inp["shot"], max_frames):
        B = sl.stop - sl.start
        I, ip = Iall[sl], Ip[sl]
        o0, Zs = force_free_z(S, I, ip, np.tile(FFZ_PROF, (B, 1)), Rinit=dp.R0_D3D)
        warm["I0"][sl] = o0["I"].cpu().numpy()
        warm["c0"][sl] = o0["c"].cpu().numpy()
        warm["Zt"][sl] = Zs
        o = kfit_bpli(S, I, ip, Zs, pairs[sl], I0=o0["I"], c0=o0["c"])
        _store(dp, st, BASE, sl, o, Zs, pairs[sl])
        del o, o0
        torch.cuda.empty_cache()
        print(f"stage1 {set_name} {sl.stop}/{n} {time.time() - t0:.0f}s", flush=True)
    st.save(inp["idx"])
    np.savez(OUT / f"warm_{set_name}.npz", idx=inp["idx"], **warm)


def _store(dp, st, name, sl, o, Zs, pairs):
    r = dp._res(o)
    npairs = np.isfinite(pairs).all((-1, -2)).sum(1)
    dp._put(st, name, sl, r, Zt=Zs, npairs=npairs, dv=o["dv"].cpu().numpy(),
            u=o["u"].cpu().numpy(), fit_betap=o["fit_betap"].cpu().numpy(),
            fit_li=o["fit_li"].cpu().numpy(), fit_lp=o["fit_lp"].cpu().numpy())


# frozen stage-2 variants (design probes on TRAIN cal; module doc / study note sec. 3)
VARIANTS = {
    "KC_A1_ffz": dict(w_bp=1.0, w_li=1.0),       # both constraints, 1/sigma weights
    "KCb_A1_ffz": dict(w_bp=1.0, w_li=0.0),      # beta_p only
    "KCl_A1_ffz": dict(w_bp=0.0, w_li=1.0),      # li only
    "KC2_A1_ffz": dict(w_bp=2.0, w_li=2.0),      # both, doubled weight
    # second TRAIN probe (added after the first probe was monotone in the weight)
    "KC4_A1_ffz": dict(w_bp=4.0, w_li=4.0),
    "KC8_A1_ffz": dict(w_bp=8.0, w_li=8.0),
}


SELECTED = "KC8_A1_ffz"     # frozen TRAIN-probe choice (highest probe S), study note sec. 4.2


def run_stage2(set_name, names, max_frames=1024, limit_shots=0):
    """Stage 2: kfit_bpli with beta_p / li rows from OUT/targets_pred_<set>.npz, warm-started
    from the stage-1 state (same Zt, same Zt = 0 start as KB).  STAGE2_RULE: a variant is the
    constrained fit on frames with a finite target where it converged (dI < CONV_TOL) and the
    stage-1 KB fit elsewhere (fec.mast_select_bpli applies the same rule on MAST)."""
    import torch
    from fec.mast_select_v2 import shot_chunks
    dp = bind_dirs(set_name)
    inp = load_inputs(set_name, limit_shots)
    n = len(inp["Ip"])
    w = dict(np.load(OUT / f"warm_{set_name}.npz"))
    tp = dict(np.load(OUT / f"targets_pred_{set_name}.npz"))
    assert np.array_equal(w["idx"][:n], inp["idx"]) and np.array_equal(tp["idx"][:n], inp["idx"])
    kb = dict(np.load(dp.PRED / f"{BASE}.npz"))
    assert np.array_equal(kb["idx"][:n], inp["idx"])
    S, curf = dp.make_solver("A1")
    Iall = curf(inp)
    Ip = np.maximum(inp["Ip"], 1e5)
    pairs = dp.all_pairs(inp)
    st = dp.Store(n)
    t0 = time.time()
    for sl in shot_chunks(inp["shot"], max_frames):
        I, ip = Iall[sl], Ip[sl]
        Zs = w["Zt"][sl]
        for nm in names:
            cfg = VARIANTS[nm]
            cons = dict(bp_t=tp["bp_t"][sl], bp_sig=tp["bp_sig"][sl], li_t=tp["li_t"][sl],
                        li_sig=tp["li_sig"][sl], **cfg)
            o = kfit_bpli(S, I, ip, Zs, pairs[sl], I0=w["I0"][sl], c0=w["c0"][sl], cons=cons)
            r = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v) for k, v in o.items()}
            has = np.zeros(sl.stop - sl.start, bool)
            for k in ("bp", "li"):
                if cfg[f"w_{k}"] > 0:
                    has |= np.isfinite(tp[f"{k}_t"][sl]) & np.isfinite(tp[f"{k}_sig"][sl])
            use = has & (r["dI"] < dp.CONV_TOL)          # STAGE2_RULE (as the MAST driver)
            rec = dict(psi=r["psi"], u=r["u"], dv=r["dv"], fit_betap=r["fit_betap"],
                       fit_li=r["fit_li"], fit_lp=r["fit_lp"],
                       **{k: r[k] for k in dp.META_KEYS + ("limited",)})
            for k, v in rec.items():
                v0 = kb[k][sl]
                rec[k] = np.where(use.reshape((-1,) + (1,) * (np.ndim(v) - 1)), v, v0)
            npairs = np.isfinite(pairs[sl]).all((-1, -2)).sum(1)
            st.put(nm, sl, rec.pop("psi"), Zt=Zs, npairs=npairs, used_c=use,
                   conv_c=r["dI"] < dp.CONV_TOL, **rec)
            del o, r
        torch.cuda.empty_cache()
        print(f"stage2 {set_name} {sl.stop}/{n} {time.time() - t0:.0f}s", flush=True)
    sub = dict(st.v)
    st.save(inp["idx"])
    return sub


# ============================================================================ features
FEATS = ("bpe", "fgw", "peak_te", "peak_p", "qcyl", "lam", "li_own", "ramp", "tnorm", "ipfrac")
LOG_FEATS = {"bpe", "fgw", "peak_te", "peak_p", "qcyl"}
ETA_LNL = 17.0                          # Spitzer resistivity eta = 1.65e-9 lnL / Te[keV]^1.5


def _record_job(job):
    """One shot: per-frame fec.betan_transfer.frame_record on our own flux maps."""
    from fec import betan_transfer as BT
    kind, payload = job
    out = []
    if kind == "arrays":                # DIII-D: maps and Thomson passed in
        for (phi, gR, gZ, box, Rc, Zc, te, ne, F, ip, Rref) in payload:
            out.append(BT.frame_record(np.asarray(phi, np.float64), gR, gZ, box, Rc, Zc, te, ne,
                                       F, ip, R_ref=Rref))
        return out
    if kind == "mast_arrays":           # MAST: shot npz + maps passed in (production driver)
        path, fidx, maps = payload
        psi_all, rows = maps, range(len(fidx))
    else:                               # MAST: shot npz + rows of a psi.npy
        path, fidx, psi_path, rows = payload
        psi_all = np.load(psi_path, mmap_mode="r")
    with np.load(path) as zf:           # inputs only: grid, Thomson, currents (no labels)
        z = {k: zf[k] for k in ("grid_R", "grid_Z", "ts_core_R", "ts_edge_R", "ts_core_Z",
                                "ts_edge_Z", "ts_core_te", "ts_edge_te", "ts_core_ne",
                                "ts_edge_ne", "cur")}
    gR, gZ = z["grid_R"], z["grid_Z"]
    Rch = np.r_[z["ts_core_R"], z["ts_edge_R"]].astype(float)
    Zch = np.r_[z["ts_core_Z"], z["ts_edge_Z"]].astype(float)
    for k, row in zip(fidx, rows):
        te = np.r_[z["ts_core_te"][k], z["ts_edge_te"][k]].astype(float)
        ne = np.r_[z["ts_core_ne"][k], z["ts_edge_ne"][k]].astype(float)
        itf, ip = z["cur"][k, 11, 6], z["cur"][k, 13, 6]
        F = MU0 * BT.MAST_TF_TURNS * abs(itf) * 1e3 / (2 * np.pi)
        out.append(BT.frame_record(np.asarray(psi_all[row], np.float64), gR, gZ, BT.MAST_AXIS_BOX,
                                   Rch, Zch, te, ne, F, abs(ip) / 1e3, R_ref=None))
    return out


def build_records(pop, nproc=4, mast_every=4):
    """Feature records -> OUT/records_<pop>.npz.

    cal / val: DIII-D subset frames, our stage-1 KB map, subset Thomson, measured Ip, B0 at
    1.6955 m; mast: every mast_every-th frame of mast_public_test (whole shots kept in order),
    our label-free mast_v3_loff map, MAST inputs only (never labels).
    Memory: per-frame maps are views of ONE in-memory array (never index an NpzFile per frame:
    each access re-reads the whole array and the view pins it); pools <= 4 workers."""
    from fec import betan_transfer as BT
    from fec.features import load_ds
    dp = _d3d()
    t0 = time.time()
    if pop in ("cal", "val"):
        inp = dp.load_set(pop)
        z = dict(np.load(OUT / f"pred_{pop}" / f"{BASE}.npz"))    # read each array once
        assert np.array_equal(z["idx"], inp["idx"])
        n = len(inp["Ip"])
        F = MU0 * dp.N_TF * inp["bt"] * 1e3 / (2 * np.pi)
        gR, gZ = inp["gR"], inp["gZ"]
        jobs = []
        for sl in dp._shot_slices(inp["shot"]):
            pl = []
            for i in range(sl.start, sl.stop):
                Rc = np.r_[inp["vR"][i], inp["tR"][i]].astype(float)
                Zc = np.r_[inp["vZ"][i], inp["tZ"][i]].astype(float)
                te = np.r_[inp["vte"][i], inp["tte"][i]].astype(float)
                ne = np.r_[inp["vne"][i], inp["tne"][i]].astype(float)
                pl.append((z["psi"][i], gR, gZ, BT.D3D_AXIS_BOX, Rc, Zc, te, ne, float(F[i]),
                           float(max(inp["Ip"][i], 1e5)) / 1e6, BT.D3D_RREF))
            jobs.append(("arrays", pl))
        ipc = inp["cur_c"][:, 20, :].astype(np.float64)            # canonical kA, 11 offsets
        shot, t_ms, idx = inp["shot"], inp["t_ms"], inp["idx"]
    elif pop == "mast":
        cfg = "mast_public_test"
        ds = load_ds(cfg)
        info = ds["info"]
        shot_all = np.asarray(ds["shot"])
        g = np.flatnonzero(np.arange(len(shot_all)) % mast_every == 0)
        psi_path = str(WORK / "mast_v3_loff" / "psi.npy")
        assert np.load(psi_path, mmap_mode="r").shape[0] == len(shot_all)
        jobs = []
        for s in np.unique(shot_all[g]):
            gg = g[shot_all[g] == s]
            jobs.append(("mast", (str(WORK / "shots" / cfg / f"{info['order'][s]}.npz"),
                                  np.asarray(ds["fidx"])[gg].tolist(), psi_path, gg.tolist())))
        cur = np.asarray(ds["cur"])
        s0 = np.sign(cur[g, 13, 6]); s0[s0 == 0] = 1
        ipc = cur[g, 13, :].astype(np.float64) * s0[:, None]
        shot, t_ms, idx = shot_all[g], np.asarray(ds["t_ms"])[g], g
    else:
        raise ValueError(pop)
    recs = []
    with Pool(min(nproc, 4)) as pool:
        for i, r in enumerate(pool.imap(_record_job, jobs, chunksize=1)):
            recs.extend(r)
            if i % 50 == 0:
                print(f"records {pop} {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    arr = {k: np.array([r[k] for r in recs], dtype=np.float64) for k in recs[0]}
    assert len(arr["a"]) == len(shot)
    ipmax = shot_ipmax("mast_public_test" if pop == "mast" else dp.CONFIG, shot)
    arr.update(idx=np.asarray(idx), shot=np.asarray(shot), t_ms=np.asarray(t_ms, np.float64),
               ip_off=ipc, ipmax=ipmax)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"records_{pop}.npz", **arr)
    print(f"records {pop}: {len(shot)} frames, geometry ok {np.mean(arr['ok_geo']):.3f}, "
          f"lin0 finite {np.isfinite(arr['pav_lin0']).mean():.3f}, {time.time() - t0:.0f}s",
          flush=True)


def shot_ipmax(config, shot):
    """max |Ip(t + 0)| over ALL frames of each frame's shot in the config (inputs only)."""
    from fec.features import load_ds
    ds = load_ds(config)
    ch = ds["info"]["channels"]
    ipc = ch.index("magnetics_plasma_current")
    sh = np.asarray(ds["shot"])
    out = np.zeros(len(shot))
    for s_ in np.unique(shot):
        a0, a1 = np.searchsorted(sh, s_, "left"), np.searchsorted(sh, s_, "right")
        out[np.asarray(shot) == s_] = float(np.max(np.abs(np.asarray(ds["cur"][a0:a1, ipc, 6]))))
    return out


def feature_table(r):
    """Machine-agnostic features (dimensionless) from records; dict name -> (N,).

    bpe     poloidal electron beta 2 mu0 <p_e> / Bpa^2, Bpa = mu0 Ip / (2 pi a sqrt((1+k^2)/2))
    fgw     Greenwald fraction <n_e>_V / (Ip / pi a^2)
    peak_te Te(0) / <Te>, <Te> = <p_e> / (e <n_e>)      peak_p  p_e(0) / <p_e>
    qcyl    5 a^2 B_geo (1+k^2)/2 / (R_geo Ip)           lam     2 R_geo (R_axis - R_geo) / a^2
    li_own  scorer li(2) of our own (stage-1) map
    ramp    asinh(dln Ip/dt x tau_R), dIp/dt over -20..+20 ms, tau_R = mu0 a^2 / eta_Spitzer(<Te>)
    tnorm   ln(t / tau_R), t from the shot time origin (both machines: t = 0 near breakdown)
    ipfrac  Ip / max |Ip| over all frames of the shot (inputs)
    Logged: LOG_FEATS.  Profile quantities from the isotonic 'lin0' Thomson fit."""
    a, kap, Rg, Rax = r["a"], r["kappa"], r["Rgeo"], r["Rax"]
    pe, ne, Ip = r["pav_lin0"], r["ne_vol"], r["Ip"]
    with np.errstate(all="ignore"):
        te_kev = pe / (QE * ne) / 1e3
        shape2 = (1 + kap ** 2) / 2
        Bpa = MU0 * Ip * 1e6 / (2 * np.pi * a * np.sqrt(shape2))
        tau = MU0 * a ** 2 / (1.65e-9 * ETA_LNL / te_kev ** 1.5)
        ip = r["ip_off"]
        dlnip = (ip[:, 10] - ip[:, 2]) / 0.040 / ip[:, 6]
        F = dict(bpe=2 * MU0 * pe / Bpa ** 2, fgw=(ne / 1e20) / (Ip / (np.pi * a ** 2)),
                 peak_te=r["te0"] / (te_kev * 1e3), peak_p=r["p0"] / pe,
                 qcyl=5 * a ** 2 * r["Bgeo"] * shape2 / (Rg * Ip),
                 lam=2 * Rg * (Rax - Rg) / a ** 2, li_own=r["li"],
                 ramp=np.arcsinh(dlnip * tau), tnorm=np.log(r["t_ms"] / 1e3 / tau),
                 ipfrac=ip[:, 6] / r["ipmax"])
        X = {k: (np.log(np.where(v > 0, v, np.nan)) if k in LOG_FEATS else v)
             for k, v in F.items()}
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in X.items()}


# ============================================================================ regressions
RANGE_PCT = (0.5, 99.5)
SOFT = {"li": 0.1, "bp": 0.25}           # soft-L1 scales (li absolute, beta_p in ln units)
BP_FIT_RANGE = (0.02, 6.0)               # TRAIN beta_p labels used in the fit
LI_FIT_RANGE = (0.3, 3.0)
MAX_FEATS = 4
MIN_GAIN = 0.005                         # forward step must raise TRAIN-CV R2 by this much
MIN_INRANGE_ONE = 0.95                   # per-feature MAST in-range fraction to be eligible
MIN_INRANGE_JOINT = 0.90                 # joint MAST in-range fraction of the selected set
CV_K, CV_SEED = 5, 0


def _target(t, kind):
    y = t["li_E"] if kind == "li" else t["betap_E"]
    lo, hi = LI_FIT_RANGE if kind == "li" else BP_FIT_RANGE
    return np.where(np.isfinite(y) & (y > lo) & (y < hi), y, np.nan)


def fit_reg(X, y, feats, mask, kind):
    """Robust (soft-L1) linear model of li, or of ln beta_p, on standardised features clamped
    to the TRAIN 0.5-99.5% range."""
    from scipy.optimize import least_squares
    m = mask & np.isfinite(y)
    for f in feats:
        m &= np.isfinite(X[f])
    Z = np.stack([X[f][m] for f in feats], 1) if feats else np.zeros((int(m.sum()), 0))
    lo = np.percentile(Z, RANGE_PCT[0], axis=0) if feats else np.zeros(0)
    hi = np.percentile(Z, RANGE_PCT[1], axis=0) if feats else np.zeros(0)
    Zc = np.clip(Z, lo, hi)
    mu, sd = Zc.mean(0), Zc.std(0) + 1e-12
    Zs = (Zc - mu) / sd
    yy = y[m] if kind == "li" else np.log(y[m])
    A = np.c_[np.ones(len(yy)), Zs]
    c0 = np.linalg.lstsq(A, yy, rcond=None)[0]
    sol = least_squares(lambda c: A @ c - yy, c0, loss="soft_l1", f_scale=SOFT[kind])
    return dict(kind=kind, feats=list(feats), lo=lo.tolist(), hi=hi.tolist(), mu=mu.tolist(),
                sd=sd.tolist(), coef=sol.x.tolist(), n_fit=int(m.sum()))


def predict_reg(model, X):
    """(prediction, in-range mask); NaN where any feature is missing."""
    n = len(next(iter(X.values())))
    c = np.asarray(model["coef"])
    if not model["feats"]:
        v = np.full(n, c[0])
        return (v if model["kind"] == "li" else np.exp(v)), np.ones(n, bool)
    Z = np.stack([X[f] for f in model["feats"]], 1)
    lo, hi = np.asarray(model["lo"]), np.asarray(model["hi"])
    fin = np.isfinite(Z).all(1)
    inr = fin & np.all((Z >= lo) & (Z <= hi), 1)
    v = c[0] + ((np.clip(Z, lo, hi) - np.asarray(model["mu"])) / np.asarray(model["sd"])) @ c[1:]
    v = np.where(fin, v, np.nan)
    return (v if model["kind"] == "li" else np.exp(v)), inr


def _r2(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def _folds(shot):
    u = np.unique(shot)
    perm = np.random.default_rng(CV_SEED).permutation(u)
    f = {s: i % CV_K for i, s in enumerate(perm)}
    return np.array([f[s] for s in shot])


def cv_oof(X, y, feats, shot, kind):
    """Shot-grouped CV out-of-fold predictions and R2 (target scale, frames with both)."""
    fo = _folds(shot)
    p = np.full(len(y), np.nan)
    for k in range(CV_K):
        mdl = fit_reg(X, y, feats, fo != k, kind)
        p[fo == k] = predict_reg(mdl, {f: X[f][fo == k] for f in feats} if feats else
                                 {"_": np.zeros(int((fo == k).sum()))})[0]
    return p, _r2(y, p)


def _inrange(model, Xm):
    _, inr = predict_reg(model, Xm)
    fin = np.ones(len(inr), bool)
    for f in model["feats"]:
        fin &= np.isfinite(Xm[f])
    return float(inr[fin].mean()) if fin.any() else 0.0


def forward_select(X, y, shot, kind, Xm, log):
    """Frozen TRAIN-only forward selection (module constants): eligible features have MAST
    per-feature in-range >= MIN_INRANGE_ONE; greedy on CV R2; stop at MAX_FEATS or a gain
    < MIN_GAIN; then drop trailing features until the joint MAST in-range >= MIN_INRANGE_JOINT.
    CV R2 of a model is computed on the frames where all of the candidate's features exist
    and compared against the constant model on the same frames (coverage-fair)."""
    allm = np.ones(len(y), bool)
    one = {}
    for f in FEATS:
        mdl = fit_reg(X, y, [f], allm, kind)
        one[f] = _inrange(mdl, Xm)
    elig = [f for f in FEATS if one[f] >= MIN_INRANGE_ONE]
    log["mast_inrange_one"] = one
    log["eligible"] = elig
    sel, steps = [], []
    cur = cv_oof(X, y, [], shot, kind)[1]
    steps.append(dict(feats=[], cv_r2=cur))
    while len(sel) < MAX_FEATS:
        best = None
        for f in elig:
            if f in sel:
                continue
            fs = sel + [f]
            r2 = cv_oof(X, y, fs, shot, kind)[1]
            if best is None or r2 > best[1]:
                best = (f, r2)
        if best is None or best[1] - cur < MIN_GAIN:
            break
        sel.append(best[0])
        cur = best[1]
        steps.append(dict(feats=list(sel), cv_r2=cur))
        print(kind, "forward", sel, round(cur, 4), flush=True)
    while sel:
        joint = _inrange(fit_reg(X, y, sel, allm, kind), Xm)
        if joint >= MIN_INRANGE_JOINT:
            break
        sel = sel[:-1]
    log["steps"] = steps
    return sel


def cmd_fit():
    """TRAIN-only fits (cal records on our stage-1 maps, EFIT li / beta_p labels), MAST
    in-range from mast_v3_loff records; OOF targets for the cal design probes."""
    r = dict(np.load(OUT / "records_cal.npz"))
    t = dict(np.load(OUT / "targets_cal.npz"))
    assert np.array_equal(r["idx"], t["idx"])
    m = dict(np.load(OUT / "records_mast.npz"))
    X, Xm = feature_table(r), feature_table(m)
    shot = r["shot"]
    sel = {"records_sha256": {p: _sha(OUT / f"records_{p}.npz") for p in ("cal", "mast")},
           "targets_sha256": _sha(OUT / "targets_cal.npz"),
           "rules": dict(FEATS=FEATS, MAX_FEATS=MAX_FEATS, MIN_GAIN=MIN_GAIN,
                         MIN_INRANGE_ONE=MIN_INRANGE_ONE, MIN_INRANGE_JOINT=MIN_INRANGE_JOINT,
                         RANGE_PCT=RANGE_PCT, SOFT=SOFT, CV_K=CV_K, CV_SEED=CV_SEED)}
    oof = {"idx": r["idx"]}
    for kind in ("li", "bp"):
        y = _target(t, kind)
        log = {}
        fs = forward_select(X, y, shot, kind, Xm, log)
        mdl = fit_reg(X, y, fs, np.ones(len(y), bool), kind)
        p_oof, r2 = cv_oof(X, y, fs, shot, kind)
        res = (y - p_oof) if kind == "li" else (np.log(y) - np.log(p_oof))
        sig = float(np.sqrt(np.nanmean(res ** 2)))
        const_oof, r2c = cv_oof(X, y, [], shot, kind)
        pm, inr = predict_reg(mdl, Xm)
        sel[kind] = dict(model=mdl, cv_r2=r2, cv_r2_const=r2c, sigma=sig,
                         sigma_kind="absolute" if kind == "li" else "ln",
                         coverage_cal=float(np.isfinite(p_oof).mean()),
                         coverage_mast=float(np.isfinite(pm).mean()),
                         mast_inrange_joint=_inrange(mdl, Xm),
                         mast_pred_pct=np.nanpercentile(pm, [1, 5, 50, 95, 99]).tolist(),
                         cal_label_pct=np.nanpercentile(y, [1, 5, 50, 95, 99]).tolist(),
                         **log)
        oof[f"{kind}_t"] = p_oof
        oof[f"{kind}_sig"] = np.full(len(p_oof), sig) if kind == "li" else p_oof * sig
        print(kind, json.dumps({k: v for k, v in sel[kind].items() if k not in ("model",)},
                               default=str)[:900], flush=True)
    np.savez(OUT / "targets_pred_cal.npz", **{k: np.asarray(v) for k, v in oof.items()})
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    return sel


def predict_targets(pop):
    """Targets / sigmas from the frozen full-TRAIN models for a records population."""
    sel = json.loads((OUT / "selection.json").read_text())
    r = dict(np.load(OUT / f"records_{pop}.npz"))
    X = feature_table(r)
    out = {"idx": r["idx"]}
    for kind in ("li", "bp"):
        p, inr = predict_reg(sel[kind]["model"], X)
        out[f"{kind}_t"] = p
        out[f"{kind}_sig"] = np.full(len(p), sel[kind]["sigma"]) if kind == "li" else \
            p * sel[kind]["sigma"]
        out[f"{kind}_inrange"] = inr
    np.savez(OUT / f"targets_pred_{pop}.npz", **out)
    return out


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================ evaluation
def evaluate_variant(name, set_name, nproc=4, gt=None):
    """fec.d3d_physics.evaluate for a (shot-prefix) prediction of this study -> EVAL dir."""
    dp = bind_dirs(set_name)
    inp = dp.load_set(set_name)
    gt = gt or dp.gt_refs(set_name)
    z = dp.load_pred(name)
    m = len(z["idx"])
    assert np.array_equal(z["idx"], inp["idx"][:m])
    F = MU0 * dp.N_TF * inp["bt"] * 1e3 / (2 * np.pi)
    gR, gZ = inp["gR"], inp["gZ"]

    def jobs():
        for i in range(m):
            Rc = np.r_[inp["vR"][i], inp["tR"][i]].astype(float)
            Zc = np.r_[inp["vZ"][i], inp["tZ"][i]].astype(float)
            te = np.r_[inp["vte"][i], inp["tte"][i]].astype(float)
            ne = np.r_[inp["vne"][i], inp["tne"][i]].astype(float)
            yield (z["psi"][i].astype(np.float64), gt["psi_st"][i].astype(np.float64),
                   gt["contours"][i], gt["cmask"][i], gt["rgeo"][i], float(z["psi_ax"][i]),
                   float(z["psi_b"][i]), float(z["Rax"][i]), float(z["Zax"][i]), float(F[i]),
                   float(max(inp["Ip"][i], 1e5) / 1e3), Rc, Zc, te, ne, gR, gZ)
    t0 = time.time()
    with Pool(min(nproc, 4)) as pool:
        rows = list(pool.imap(dp._frame_worker, jobs(), chunksize=16))
    ev = {k: np.array([rr[k] for rr in rows]) for k in rows[0]}
    lab = np.load(dp.OUT / f"lab_{set_name}.npz")
    ins = lab["Inode"][:m].reshape(m, -1) != 0
    dpsi = (z["psi"].reshape(m, -1) - lab["psi"][:m].reshape(m, -1)).astype(np.float64)
    ev["ss_in"] = (np.where(ins, np.nan_to_num(dpsi), 0) ** 2).sum(1)
    ev["n_in"] = ins.sum(1)
    sub = {k: (v[:m] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(inp["Ip"]) else v)
           for k, v in inp.items()}
    ev["mm_te"] = dp.mismatch_frames(sub, z["psi"].astype(np.float64), z["psi_ax"], z["psi_b"], "te")
    ev["mm_ne"] = dp.mismatch_frames(sub, z["psi"].astype(np.float64), z["psi_ax"], z["psi_b"], "ne")
    for k in ("Rax", "Zax", "psi_ax", "psi_b", "Zt", "dI", "limited", "npairs", "dv",
              "fit_betap", "fit_li", "fit_lp", "used_c", "conv_c"):
        if k in z:
            ev["meta_" + k] = z[k]
    dp.EVAL.mkdir(parents=True, exist_ok=True)
    np.savez(dp.EVAL / f"{name}.npz", **ev)
    print(f"evaluated {name} ({set_name}, {m} frames) in {time.time() - t0:.0f}s", flush=True)
    return ev


def _gt_prefix(gt, m):
    out = {}
    for k, v in gt.items():
        if isinstance(v, np.ndarray) and v.ndim and len(v) >= m and k != "keys":
            out[k] = v[:m]
        elif k == "contours":
            out[k] = v[:m]
        else:
            out[k] = v
    return out


def summarize(set_name, names, out_name=None, limit_shots=0):
    """Official composite (fec.d3d_physics.aggregate) per variant, all frames and per phase,
    with per-scalar consistency R2 and beta_p / li diagnostics against the labels.
    limit_shots: score only the frames of the first shots (common prefix of all variants)."""
    dp = bind_dirs(set_name)
    inp = dp.load_set(set_name)
    limit_frames = int(np.isin(inp["shot"], np.unique(inp["shot"])[:limit_shots]).sum())         if limit_shots else 0
    gt = dp.gt_refs(set_name)
    ph = dp.phase_of(inp)
    tg = np.load(OUT / f"targets_{set_name}.npz") if (OUT / f"targets_{set_name}.npz").exists() else None
    rows = {}
    for nm in names:
        ev = dict(np.load(dp.EVAL / f"{nm}.npz"))
        m = len(ev["ss_p"]) if not limit_frames else min(len(ev["ss_p"]), limit_frames)
        ev = {k: (v[:m] if isinstance(v, np.ndarray) and v.ndim else v) for k, v in ev.items()}
        g = _gt_prefix(gt, m)
        shot = inp["shot"][:m]
        r = {"all": dp.aggregate(ev, g, shot=shot)}
        for p in dp.PHASES:
            r[p] = dp.aggregate(ev, g, mask=ph[:m] == p, shot=shot)
        r["conv"] = float(np.nanmean(ev["meta_dI"] < dp.CONV_TOL)) if "meta_dI" in ev else None
        for k in ("used_c", "conv_c"):
            if "meta_" + k in ev:
                r[k] = float(np.mean(ev["meta_" + k]))
        if "meta_Rax" in ev:
            lab = np.load(dp.OUT / f"lab_{set_name}.npz")
            r["dRax_bias_cm"] = float(np.nanmedian(ev["meta_Rax"] - lab["raxis"][:m]) * 100)
            r["dRax_med_cm"] = float(np.nanmedian(np.abs(ev["meta_Rax"] - lab["raxis"][:m])) * 100)
            r["dZax_med_cm"] = float(np.nanmedian(np.abs(ev["meta_Zax"] - lab["zaxis"][:m])) * 100)
        if tg is not None and "meta_fit_li" in ev:
            for k, lk in (("fit_li", "li_E"), ("fit_betap", "betap_E")):
                y, p = tg[lk][:m], ev["meta_" + k]
                r[k + "_r2_vs_label"] = _r2(y, p)
                r[k + "_median_ratio"] = float(np.nanmedian(p / y))
        rows[nm] = r
    if out_name:
        (OUT / out_name).write_text(json.dumps(rows, indent=1))
    cs = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li")
    f = lambda x: "-" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    print("| variant | n | S | R2psi | R2q95 | R2betaN | D_LCFS | cons | " + " | ".join(cs) +
          " | conv | li fit R2 | betap fit R2 | dRax bias cm | dRax med cm |")
    for nm, r in rows.items():
        a = r["all"]
        print(f"| {nm} | {a['n']} | {a['S']:.4f} | {a['r2psi']:.5f} | {f(a['r2q95'])} | "
              f"{f(a['r2betan'])} | {a['dlcfs']:.4f} | {a['cons']:.4f} | " +
              " | ".join(f(a["r2cons"][c]) for c in cs) +
              f" | {f(r.get('conv'))} | {f(r.get('fit_li_r2_vs_label'))} | "
              f"{f(r.get('fit_betap_r2_vs_label'))} | {f(r.get('dRax_bias_cm'))} | "
              f"{f(r.get('dRax_med_cm'))} |")
    for nm, r in rows.items():
        print(f"| {nm} | " + " | ".join(
            f"{p} {r[p]['S']:.4f}/{r[p]['r2psi']:.4f}/{r[p]['dlcfs']:.4f}/{r[p]['cons']:.4f}"
            for p in dp.PHASES) + " |")
    return rows


# ============================================================================ acceptance
ACCEPT = dict(min_S_gain=0.020, min_cons_gain=0.0, max_r2psi_loss=0.002)


def accept_val(selected):
    """Frozen VAL acceptance (study note sec. 3.6) of `selected` against KB, same run."""
    names = [BASE] + list(VARIANTS)
    rows = summarize("val", names, "summary_val.json")
    b, s = rows[BASE]["all"], rows[selected]["all"]
    d = dict(S_gain=s["S"] - b["S"], cons_gain=s["cons"] - b["cons"],
             r2psi_change=s["r2psi"] - b["r2psi"], dlcfs_change=s["dlcfs"] - b["dlcfs"])
    ok = (d["S_gain"] >= ACCEPT["min_S_gain"] and d["cons_gain"] > ACCEPT["min_cons_gain"]
          and d["r2psi_change"] >= -ACCEPT["max_r2psi_loss"])
    ref = OUT.parent / "d3d_phys" / "eval" / "KF_A1_ffz.npz"
    rep = dict(selected=selected, rule=ACCEPT, deltas=d, accepted=bool(ok),
               KB=b, selected_row=s,
               others={k: {kk: rows[k]["all"][kk] - b[kk] for kk in ("S", "r2psi", "dlcfs", "cons")}
                       for k in VARIANTS},
               selection_sha256=_sha(OUT / "selection.json"),
               targets_pred_val_sha256=_sha(OUT / "targets_pred_val.npz"),
               kf_reference_eval=str(ref))
    if ref.exists():
        dp = bind_dirs("val")
        ev = dict(np.load(ref))
        rep["KF_A1_ffz_reference"] = dp.aggregate(ev, dp.gt_refs("val"),
                                                  shot=dp.load_set("val")["shot"])
    (OUT / "val_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("selected", "deltas", "accepted")}, indent=1))
    return rep


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--set", default="cal")
    ap.add_argument("--pop", default="cal")
    ap.add_argument("--names", default="")
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "labels":
        label_targets(a.set)
    elif a.cmd == "stage1":
        run_stage1(a.set)
    elif a.cmd == "stage2":
        run_stage2(a.set, [x for x in a.names.split(",") if x], limit_shots=a.limit)
    elif a.cmd == "records":
        build_records(a.pop, a.nproc)
    elif a.cmd == "fit":
        cmd_fit()
    elif a.cmd == "predict":
        predict_targets(a.pop)
    elif a.cmd == "eval":
        dp = bind_dirs(a.set)
        gt = dp.gt_refs(a.set)
        for nm in [x for x in a.names.split(",") if x]:
            evaluate_variant(nm, a.set, a.nproc, gt)
    elif a.cmd == "accept":
        accept_val(a.names)
    elif a.cmd == "summary":
        summarize(a.set, [x for x in a.names.split(",") if x],
                  f"summary_{a.set}{'_lim' if a.limit else ''}.json", a.limit)
    else:
        raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
