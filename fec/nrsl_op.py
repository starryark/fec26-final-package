"""mast_solver operator NRS (net-anchored re-solve), DIII-D testbed implementation.

For each frame, given a base map psi_N (canonical, axis max) from the amortized net:
  1. node currents of the base map: I_k = -Delta* psi_N / (mu0 R) dR dZ on the base map's own core mask
     (solver core_mask from its axis / X-points), clamped >= 0;
  2. profile: LS fit of I_k with the 8-term basis sum_m a_m x^m R/R0 + c_m x^m R0/R (x = base psiN, m = 0..3),
     i.e. fec.d3d_physics.fit_label_profile applied to the NET's map (never a label);
  3. free-boundary re-solve in the PHYSICAL external field (A1 on DIII-D: coil currents + TRAIN vacuum model),
     NOT the map-implied field; Ip = measured (100 kA floor); height held at the base map's current-centroid Z
     (solver fix_z: rigid re-centring + radial-field feedback c); radial position optionally held at the base
     map's current-centroid R by a uniform-Bz column dv (the generator's own dv nuisance) with an integral
     feedback ('pin'), or left to force balance ('free');
  4. warm start from the base node currents; fixed iteration count (deterministic, no data-dependent stop);
  5. label-free fallback: keep the base map if the re-solve is not converged / has no X-point boundary /
     axis at the axis-box edge / centroid miss > 1 cm / (pin) |dv| at its 10 mT clamp.

Library + CLI (cal probe):  python -B tools/ms_op.py run --set cal --frames F.npy --base B.npz --out O.npz --rmode pin
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MU0 = 4e-7 * math.pi
DV_CLAMP = 0.010           # tesla
SP_LO, SP_HI = -0.9, 4.0    # additive pressure-like term weight (rfb 'bp2'), relative to the net profile
LSP_CLAMP = 3.0            # |log scale| of the pressure-like profile half (rfb 'bp')
K_G = 10.0                 # gamma per metre of R miss while the pressure actuator is saturated (soft li)
GAM_CLAMP = 4.0            # profile peaking exponent exp(-gam psiN) (li feedback)
LI_TOL = 0.03              # |li_fin - li_target| for the fallback (li arms)
CONV_TOL = 1e-2            # as the generator's validity rule
MISS_TOL = 0.01            # m, centroid miss for the fallback


def _D():
    from fec import d3d_physics as D
    return D


def make_solver(device="cpu", dtype=None):
    """D3DSolver on the A1 matrix + uniform-Bz column (psi = R^2/2 per tesla), as amort_d3d_gen.make_gen_solver."""
    import torch
    D = _D()
    M, curf = D.field_matrix("A1")
    if D._SOLVER_CLS is None:
        D._SOLVER_CLS = D._make_solver_cls()
    from fec.geometry import machine_geometry
    g = machine_geometry(D.CONFIG)
    RR, _ = np.meshgrid(g["grid_R"], g["grid_Z"])
    Mx = np.concatenate([M, (RR ** 2 / 2).reshape(-1, 1)], 1)
    S = D._SOLVER_CLS(Mx, device=device, dtype=dtype or torch.float64)
    return S, curf


def base_state(S, psi):
    """Core mask, psiN, axis, X-points of base maps (B,nz,nr) with the solver's own boundary logic (final)."""
    import torch
    psi = torch.as_tensor(psi, device=S.dev, dtype=S.dt)
    psi_ax, Rax, Zax = S.find_axis(psi)
    xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
    psi_b, _ = S.boundary(psi, psi_ax, Rax, Zax, xp, final=True)
    ok = (psi_b > -1e20) & (psi_ax > psi_b)
    psi_b_s = torch.where(ok, psi_b, psi_ax - 0.05 * psi_ax.abs().clamp_min(1e-3))
    mask, psiN = S.core_mask(psi, psi_ax, psi_b_s, Rax, Zax, xp)
    return dict(psi=psi, psi_ax=psi_ax, psi_b=psi_b, Rax=Rax, Zax=Zax, mask=mask & ok[:, None, None],
                psiN=psiN, ok=ok)


def node_currents(S, psi, mask):
    """(B, Ns) node currents I = -Delta* psi/(mu0 R) dR dZ on mask (clamped >= 0), numpy float64."""
    D = _D()
    gR, gZ = S.spec.gR, S.spec.gZ
    RR, _ = np.meshgrid(gR, gZ)
    ds = D.delta_star(np.asarray(psi, np.float64), gR, gZ)
    J = -ds / (MU0 * RR[None])
    I = np.where(np.asarray(mask), J * S.dR * S.dZ, 0.0)
    I = np.clip(I, 0.0, None).reshape(len(psi), -1)
    return I[:, np.asarray(S.sidx.cpu())]


def fit_profile(S, I, pn):
    """LS fit of node currents I (B,Ns) on base psiN pn (B,Ns) with the 8-term basis; rows scaled to max |coef| 1.
    Returns coef (B,8), relative residual (B,)."""
    Rs = np.asarray(S.Rs.cpu(), np.float64)
    R0 = S.spec.R0
    B = len(I)
    coef = np.tile(np.array([1, 0, -1, 0, 0, 0, 0, 0], float), (B, 1))
    rel = np.full(B, np.nan)
    for i in range(B):
        m = I[i] > 0
        if m.sum() < 10:
            continue
        x = np.clip(pn[i, m], 0, 1)
        P = np.stack([x ** k for k in range(4)], 1)
        A = np.concatenate([P * (Rs[m] / R0)[:, None], P * (R0 / Rs[m])[:, None]], 1)
        sol = np.linalg.lstsq(A, I[i, m], rcond=None)[0]
        rel[i] = np.linalg.norm(A @ sol - I[i, m]) / np.linalg.norm(I[i, m])
        coef[i] = sol / max(abs(sol).max(), 1e-30)
    return coef, rel


def _shiftR(S, Ig, d):
    """Shift node-current grid Ig (B,nz,nr) by d (B,) metres in R (linear, mass-conserving), as Solver._shiftZ."""
    import torch
    B = Ig.shape[0]
    x = d / S.dR
    n = torch.floor(x)
    f = (x - n)[:, None, None]
    n = n.long()
    ir = torch.arange(S.nr, device=Ig.device)[None, :] - n[:, None]

    def take(idx):
        ok = (idx >= 0) & (idx < S.nr)
        g = Ig.gather(2, idx.clamp(0, S.nr - 1)[:, None, :].expand(B, S.nz, S.nr))
        return g * ok[:, None, :]
    return (1 - f) * take(ir) + f * take(ir - 1)


def resolve(S, Iext, Ip, prof8, Zt, Rt=None, I0=None, n_iter=150, omega=0.4, kc=0.05, kr=0.02, li_target=None,
            k_li=2.0, li_start=10, dv_clamp=DV_CLAMP, rfb="dv", kp=3.0, beta0=None, soft_li=False):
    """rfb: radial feedback actuator when Rt is given: 'dv' (uniform Bz column) or 'bp' (log-scale of the pressure-like
    R/R0 half of the profile; dv stays 0)."""
    """Picard re-solve (V3Solver._picard numerics + optional R pin through the uniform-Bz column).
    Iext (B,44) A1 currents (dv column appended here), Ip (B,), prof8 (B,8), Zt/Rt (B,) targets of the current
    centroid; I0 (B,Ns) warm start.  kr: integral gain [T/m per MA per iteration]."""
    import torch
    dev, dt = S.dev, S.dt
    t = lambda x: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)  # noqa: E731
    B = len(Ip)
    Ipt, Ztt = t(Ip), t(Zt)
    dv = torch.zeros(B, device=dev, dtype=dt)
    Iext0 = t(np.concatenate([np.asarray(Iext, np.float64), np.zeros((B, 1))], 1))
    psi_ext0 = (Iext0 @ S.Mext.T).reshape(B, S.nz, S.nr)
    psi_dv = (S.Mext[:, -1]).reshape(S.nz, S.nr)
    Rs, Zs, R0 = S.Rs, S.Zs, S.spec.R0
    a, cc = t(prof8)[:, 0:4], t(prof8)[:, 4:8]

    gam = torch.zeros(B, device=dev, dtype=dt)
    lsp = torch.zeros(B, device=dev, dtype=dt)
    sp = torch.zeros(B, device=dev, dtype=dt)
    sat = torch.zeros(B, device=dev, dtype=torch.bool)
    dR_prev = torch.zeros(B, device=dev, dtype=dt)
    bet = t(beta0) if beta0 is not None else torch.full((B,), 0.5, device=dev, dtype=dt)
    lit = t(li_target) if li_target is not None else None

    def shape_fn(pn, ms):
        pw = torch.stack([torch.ones_like(pn), pn, pn ** 2, pn ** 3], 1)
        base = (torch.exp(lsp)[:, None] * torch.einsum("bm,bmn->bn", a, pw) * Rs[None] / R0 +
                torch.einsum("bm,bmn->bn", cc, pw) * R0 / Rs[None]).clamp_min(0) * ms
        if rfb == "beta":
            P = torch.einsum("bm,bmn->bn", a, pw).clamp_min(0) * (Rs[None] / R0) * ms
            Fq = torch.einsum("bm,bmn->bn", cc, pw).clamp_min(0) * (R0 / Rs[None]) * ms
            P = torch.where(P.sum(1, keepdim=True) > 1e-12 * Fq.sum(1, keepdim=True).clamp_min(1e-30), P,
                            (1 - pn) * (Rs[None] / R0) * ms)
            Fq = torch.where(Fq.sum(1, keepdim=True) > 1e-12 * P.sum(1, keepdim=True).clamp_min(1e-30), Fq,
                             (1 - pn) * (R0 / Rs[None]) * ms)
            base = bet[:, None] * P / P.sum(1, keepdim=True).clamp_min(1e-30) +                 (1 - bet[:, None]) * Fq / Fq.sum(1, keepdim=True).clamp_min(1e-30)
        if rfb == "bp2":
            bn = base / base.sum(1, keepdim=True).clamp_min(1e-30)
            pr = (Rs[None] / R0) * (1 - pn) * ms
            pr = pr / pr.sum(1, keepdim=True).clamp_min(1e-30)
            base = (bn + sp[:, None] * pr).clamp_min(0) * ms
        return base * torch.exp(-gam[:, None] * pn) if lit is not None else base
    if I0 is None:
        w = torch.exp(-((Rs[None] - R0) ** 2 + (Zs[None] - Ztt[:, None]) ** 2) / 0.05)
        I = Ipt[:, None] * w / w.sum(1, keepdim=True)
    else:
        I = t(I0)
        I = Ipt[:, None] * I / I.sum(1, keepdim=True).clamp_min(1e-30)
    c = torch.zeros(B, device=dev, dtype=dt)
    fb = (S.RR ** 2)[None] * (S.ZZ[None] - Ztt[:, None, None])
    pin = Rt is not None
    Rtt = t(Rt) if pin else None
    Npix = S.nz * S.nr
    prev_mask = None
    dI = torch.zeros(B, device=dev, dtype=dt)
    for it in range(n_iter):
        psi = psi_ext0 + dv[:, None, None] * psi_dv[None] + (I @ S.G.T).reshape(B, S.nz, S.nr) + \
            c[:, None, None] * fb
        psi_ax, Rax, Zax = S.find_axis(psi)
        xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
        psi_b, _ = S.boundary(psi, psi_ax, Rax, Zax, xp)
        seedm = prev_mask if (it % 10 != 0) else None
        mask, psiN = S.core_mask(psi, psi_ax, psi_b, Rax, Zax, xp, seed_mask=seedm)
        prev_mask = mask
        if lit is not None and it >= li_start:
            li_now = li_grid(S, psi, mask, Ipt)
            step = k_li * (lit - li_now)
            if soft_li:            # position priority: while the pressure actuator is saturated, gamma holds R
                step = torch.where(sat, K_G * dR_prev, step)
            gam = (gam + step).clamp(-GAM_CLAMP, GAM_CLAMP)
        pn = psiN.reshape(B, -1)[:, S.sidx].clamp(0, 1)
        ms = mask.reshape(B, -1)[:, S.sidx].to(dt)
        shape = shape_fn(pn, ms)
        tot = shape.sum(1, keepdim=True).clamp_min(1e-30)
        Inew = Ipt[:, None] * shape / tot
        full = torch.zeros(B, Npix, device=dev, dtype=dt)
        full[:, S.sidx] = Inew
        full = full.reshape(B, S.nz, S.nr)
        Zc_new = (Inew * Zs[None]).sum(1) / Ipt
        delta = Ztt - Zc_new
        full = S._shiftZ(full, delta)
        c = c + kc * delta
        if pin:
            Rc_new = (Inew * Rs[None]).sum(1) / Ipt
            dR = Rtt - Rc_new
            full = _shiftR(S, full, dR)
            if rfb == "dv":
                dv = (dv + kr * (Ipt / 1e6) * dR).clamp(-dv_clamp, dv_clamp)
            elif rfb == "beta":
                bet = (bet + kp * dR).clamp(0.0, 1.0)
            elif rfb == "bp":
                lsp = (lsp + kp * dR).clamp(-LSP_CLAMP, LSP_CLAMP)
            else:
                sp = (sp + kp * dR).clamp(SP_LO, SP_HI)
                sat = ((sp >= SP_HI - 1e-9) & (dR > 0)) | ((sp <= SP_LO + 1e-9) & (dR < 0))
            dR_prev = dR
        Inew = full.reshape(B, Npix)[:, S.sidx]
        Inew = Ipt[:, None] * Inew / Inew.sum(1, keepdim=True).clamp_min(1e-30)
        dI = (Inew - I).abs().sum(1) / Ipt
        I = (1 - omega) * I + omega * Inew
    psi = psi_ext0 + dv[:, None, None] * psi_dv[None] + (I @ S.G.T).reshape(B, S.nz, S.nr) + c[:, None, None] * fb
    psi_ax, Rax, Zax = S.find_axis(psi)
    xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
    psi_b, _ = S.boundary(psi, psi_ax, Rax, Zax, xp, final=True)
    psi_b_s = torch.where((psi_b > -1e20) & (psi_ax > psi_b), psi_b, psi_ax - 0.05 * psi_ax.abs().clamp_min(1e-3))
    mfin, _ = S.core_mask(psi, psi_ax, psi_b_s, Rax, Zax, xp)
    li_fin = li_grid(S, psi, mfin, Ipt)
    # natural (un-shifted) centroid miss at the last iterate
    Rc = (I * Rs[None]).sum(1) / Ipt
    Zc = (I * Zs[None]).sum(1) / Ipt
    tn = lambda x: x.detach().cpu().numpy()  # noqa: E731
    return dict(psi=tn(psi), psi_ax=tn(psi_ax), psi_b=tn(psi_b), Rax=tn(Rax), Zax=tn(Zax), dI=tn(dI),
                c=tn(c), dv=tn(dv), Rc=tn(Rc), Zc=tn(Zc), last_dZ=tn(delta), gam=tn(gam), li_fin=tn(li_fin), lsp=tn(lsp), sp=tn(sp), bet=tn(bet),
                last_dR=tn(dR) if pin else np.zeros(B))


def operator(S, Iext, Ip, psi_base, rmode="pin", n_iter=150, kr=0.02, li_target=None, k_li=2.0, dv_clamp=DV_CLAMP,
             kp=3.0):
    """rmode: 'pin' (R held by dv), 'pinp' (R held by the pressure-half scale), 'free'."""
    """NRS on a batch.  Returns out dict with psi (B,nz,nr) float64 and diagnostics, fallback flags."""
    st = base_state(S, psi_base)
    mask = st["mask"].cpu().numpy()
    I_b = node_currents(S, psi_base, mask)
    pn = st["psiN"].reshape(len(psi_base), -1)[:, S.sidx].clamp(0, 1).cpu().numpy()
    coef, rel = fit_profile(S, I_b, pn)
    Rs = np.asarray(S.Rs.cpu(), np.float64)
    Zs = np.asarray(S.Zs.cpu(), np.float64)
    Ib_sum = I_b.sum(1)
    good_base = st["ok"].cpu().numpy() & (Ib_sum > 0)
    Rt = np.where(good_base, (I_b * Rs).sum(1) / np.maximum(Ib_sum, 1e-30), S.spec.R0)
    Zt = np.where(good_base, (I_b * Zs).sum(1) / np.maximum(Ib_sum, 1e-30), 0.0)
    # pressure-like (R/R0) share of the base profile on the base mask (beta-split start value)
    x = np.clip(pn, 0, 1)
    pw = np.stack([x ** k for k in range(4)], 1)
    Pp = np.clip(np.einsum("bm,bmn->bn", coef[:, :4], pw), 0, None) * (Rs / S.spec.R0)[None] * (I_b > 0)
    Ff = np.clip(np.einsum("bm,bmn->bn", coef[:, 4:], pw), 0, None) * (S.spec.R0 / Rs)[None] * (I_b > 0)
    beta0 = np.clip(Pp.sum(1) / np.maximum(Pp.sum(1) + Ff.sum(1), 1e-30), 0.0, 1.0)
    I0 = np.where(good_base[:, None], I_b, 0.0)
    I0[~good_base] = 1.0
    o = resolve(S, Iext, Ip, coef, Zt, Rt if rmode in ("pin", "pinp", "pinb", "pinbs", "pinbeta") else None, I0=I0, n_iter=n_iter, kr=kr,
                li_target=li_target, k_li=k_li, dv_clamp=dv_clamp, rfb={"pinp": "bp", "pinb": "bp2", "pinbs": "bp2", "pinbeta": "beta"}.get(rmode, "dv"), kp=kp,
                beta0=beta0, soft_li=rmode == "pinbs")
    a = S.spec.axis_box
    edge = 0.005
    conv = o["dI"] < CONV_TOL
    xok = (o["psi_b"] > -1e20) & (o["psi_ax"] > o["psi_b"])
    inbox = (o["Rax"] > a[0] + edge) & (o["Rax"] < a[1] - edge) & (o["Zax"] > a[2] + edge) & (o["Zax"] < a[3] - edge)
    miss = np.abs(o["last_dZ"]) < MISS_TOL
    if rmode == "pin":
        miss &= (np.abs(o["last_dR"]) < MISS_TOL) & (np.abs(o["dv"]) < dv_clamp - 1e-6)
    if rmode == "pinp":
        miss &= (np.abs(o["last_dR"]) < MISS_TOL) & (np.abs(o["lsp"]) < LSP_CLAMP - 1e-6)
    if rmode == "pinbeta":
        miss &= (np.abs(o["last_dR"]) < MISS_TOL) & (o["bet"] > 1e-6) & (o["bet"] < 1 - 1e-6)
    if rmode == "pinb":
        miss &= (np.abs(o["last_dR"]) < MISS_TOL) & (o["sp"] > SP_LO + 1e-6) & (o["sp"] < SP_HI - 1e-6)
    if li_target is not None and rmode != "pinbs":
        miss &= (np.abs(o["li_fin"] - np.asarray(li_target)) < LI_TOL) & (np.abs(o["gam"]) < GAM_CLAMP - 1e-6)
    if rmode == "pinbs":
        miss &= (np.abs(o["last_dR"]) < MISS_TOL) & (np.abs(o["gam"]) < GAM_CLAMP - 1e-6)
    use = good_base & conv & xok & inbox & miss & np.isfinite(o["psi"]).all((1, 2))
    psi_out = np.where(use[:, None, None], o["psi"], np.asarray(psi_base, np.float64))
    o.update(beta0=beta0, psi_out=psi_out, use=use, conv=conv, xok=xok, inbox=inbox, miss_ok=miss, good_base=good_base,
             coef=coef, fit_rel=rel, Rt=Rt, Zt=Zt, Ib_sum=Ib_sum)
    return o


N_DIR = 8


def li_grid(S, psi, mask, Ip):
    """Grid li(2) = <Bp^2>_V / (mu0 Ip / lp)^2 (scorer definition): Bp from central differences on the core-mask
    nodes, dV = 2 pi R dR dZ, mu0 Ip from the solver's total current, lp = pi x mean projected width of the mask
    over 8 directions (Cauchy's perimeter formula for a convex section, cell extent added).  torch, (B,)."""
    import torch
    pR, pZ, _, _, _ = S._grad(psi)
    RR = S.RR[None]
    Bp2 = (pR ** 2 + pZ ** 2) / RR ** 2
    m = mask.to(psi.dtype)
    dV = 2 * math.pi * RR * S.dR * S.dZ
    V = (m * dV).sum((1, 2)).clamp_min(1e-12)
    bp2 = (m * Bp2 * dV).sum((1, 2)) / V
    big = 1e9
    widths = []
    for k in range(N_DIR):
        th = math.pi * k / N_DIR
        u = math.cos(th) * S.RR + math.sin(th) * S.ZZ
        hi = torch.where(mask, u[None], torch.full_like(psi, -big)).amax((1, 2))
        lo = torch.where(mask, u[None], torch.full_like(psi, big)).amin((1, 2))
        widths.append(hi - lo + abs(math.cos(th)) * S.dR + abs(math.sin(th)) * S.dZ)
    lp = math.pi * torch.stack(widths, 0).mean(0)
    return bp2 / (MU0 * Ip / lp) ** 2
