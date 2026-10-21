"""MAST v2 label-free equilibria: Thomson-constrained profile selection and kinetic fit.

Inputs only (coil currents, Ip, Thomson Te/ne on the Z=0 chord); no MAST labels.
Validated on the synthetic twin (fec/twin_mast.py, docs/research/MAST_TWIN_STUDY_20261007.md).

  select_v2   production function (one whole-shot batch): force-free height, regularised
              kinetic fit kfit2, 12-shape grid on frames lacking Te pairs
  stage1_v2   production driver -> psi.npy + meta.npz (same keys stage 2 of fec.mast_v1 reads)
  kfit2       convergent replacement of fec.kefit.KineticSolver.kfit
  select_batch / refine_batch / symmetry_mm   grid selection, parabolic refinement, GPU mismatch

    python -B fec/mast_select_v2.py --out fec_work/mast_v2 [--limit N]

The mismatch maps each psiN level g of a candidate equilibrium to its inboard/outboard
midplane radii R_in(g), R_out(g) (monotone envelope walking out from the row minimum, as in
fec.kinetic) and compares the measured, median-filtered profile sampled there:
    mm = mean_g |V(R_in(g)) - V(R_out(g))| / p95(V)
Profile preparation is candidate independent (done once per frame on CPU); each candidate
costs a few tensor ops on GPU, so tens of candidates per frame are affordable.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import median_filter

LEVELS_V1 = np.linspace(0.05, 0.85, 17)


def prep_profiles(R, V, size=5, min_n=20):
    """R, V (B,N) channel radii / values (NaN = missing) -> padded sorted filtered arrays."""
    B, N = R.shape
    Rs = np.full((B, N), np.inf)
    Vs = np.zeros((B, N))
    scale = np.full(B, np.nan)
    n = np.zeros(B, int)
    for i in range(B):
        ok = np.isfinite(R[i]) & np.isfinite(V[i])
        if ok.sum() < min_n:
            continue
        o = np.argsort(R[i][ok], kind="stable")
        r = R[i][ok][o]
        v = median_filter(V[i][ok][o], size=size, mode="nearest")
        sc = np.percentile(v, 95)
        if not sc > 0:
            continue
        Rs[i, :len(r)] = r
        Vs[i, :len(r)] = v
        scale[i] = sc
        n[i] = len(r)
    return dict(R=Rs, V=Vs, scale=scale, n=n)


def to_dev(prep, device, dtype=torch.float32):
    return {k: torch.as_tensor(v, device=device, dtype=(torch.long if k == "n" else dtype))
            for k, v in prep.items()}


def midplane_radii(psi, psi_ax, psi_b, gR, iz0, levels, rwin=(0.3, 1.4)):
    """Inboard/outboard radii (B,L) of psiN levels on grid row iz0; valid mask (B,L)."""
    B, nz, nr = psi.shape
    pn = (psi_ax[:, None] - psi[:, iz0, :]) / (psi_ax - psi_b)[:, None]
    Rg = gR
    win = (Rg > rwin[0]) & (Rg < rwin[1])
    jmin = torch.argmin(torch.where(win[None], pn, torch.full_like(pn, float("inf"))), 1)
    j = torch.arange(nr, device=psi.device)
    ninf = torch.full_like(pn, -float("inf"))
    lev = levels[None].expand(B, -1).contiguous()

    def cross(s, R):
        s = torch.cummax(s, 1).values
        k = torch.searchsorted(s.contiguous(), lev)          # first index with s >= level
        k1 = k.clamp(1, nr - 1)
        s0, s1 = s.gather(1, k1 - 1), s.gather(1, k1)
        ok = (k > 0) & (k < nr) & torch.isfinite(s0) & (s1 > s0)
        f = ((lev - s0) / torch.where(ok, s1 - s0, torch.ones_like(s0))).clamp(0, 1)
        Rc = R[k1 - 1] + f * (R[k1] - R[k1 - 1])
        return Rc, ok

    so = torch.where(j[None] >= jmin[:, None], pn, ninf)
    Rout, okout = cross(so, Rg)
    pf = pn.flip(1)
    si = torch.where(j[None] >= (nr - 1 - jmin)[:, None], pf, ninf)
    Rin, okin = cross(si, Rg.flip(0))
    return Rin, Rout, okin & okout


def sample_profile(pd, Rq):
    """Linear interpolation of prepared profiles at Rq (B,L); valid inside channel coverage."""
    Rk, V, n = pd["R"], pd["V"], pd["n"]
    N = Rk.shape[1]
    k = torch.searchsorted(Rk.contiguous(), Rq.contiguous())
    k1 = k.clamp(1, N - 1)
    r0, r1 = Rk.gather(1, k1 - 1), Rk.gather(1, k1)
    v0, v1 = V.gather(1, k1 - 1), V.gather(1, k1)
    ok = (k > 0) & (k < n[:, None])
    den = torch.where(ok, r1 - r0, torch.ones_like(r0)).clamp_min(1e-9)
    f = ((Rq - torch.where(ok, r0, Rq)) / den).clamp(0, 1)
    return v0 + f * (v1 - v0), ok


def symmetry_mm(psi, psi_ax, psi_b, gR, iz0, pd, levels, min_levels=5, power=1):
    """Batched symmetry mismatch (B,) [NaN if <min_levels covered levels]."""
    Rin, Rout, okr = midplane_radii(psi, psi_ax, psi_b, gR, iz0, levels)
    vi, oki = sample_profile(pd, Rin)
    vo, oko = sample_profile(pd, Rout)
    ok = okr & oki & oko
    d = (vi - vo).abs() ** power
    cnt = ok.sum(1)
    mm = (d * ok).sum(1) / cnt.clamp_min(1) / pd["scale"] ** power
    return torch.where(cnt >= min_levels, mm, torch.full_like(mm, float("nan"))), Rin, Rout, ok


def force_free_z(S, I, Ip, prof, n0=150, n1=60, dz=0.05, Rinit=0.9, solve=None):
    """v1's vertical secant: solves at Zt=0 and Zt=dz, linear estimate of the c=0 height."""
    solve = solve or S.solve
    B = len(Ip)
    o0 = solve(I, Ip, prof, np.zeros(B), n_iter=n0, Rinit=np.full(B, Rinit))
    o1 = solve(I, Ip, prof, np.full(B, dz), n_iter=n1, I0=o0["I"], c0=o0["c"])
    c0, c1 = o0["c"].cpu().numpy(), o1["c"].cpu().numpy()
    den = np.where(np.abs(c1 - c0) > 1e-12, c1 - c0, np.nan)
    Zs = np.nan_to_num(np.clip(-c0 * dz / den, -0.3, 0.3))
    return o0, Zs


# ----------------------------------------------------------------------------- selector
def shot_chunks(shot, max_frames=1024):
    """Contiguous slices of whole shots (frames sorted by shot, time) up to max_frames."""
    bounds = np.flatnonzero(np.diff(shot) != 0) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(shot)]])
    out, s0, e0 = [], None, None
    for s, e in zip(starts, ends):
        if s0 is None:
            s0, e0 = s, e
        elif e - s0 <= max_frames:
            e0 = e
        else:
            out.append(slice(s0, e0)); s0, e0 = s, e
    if s0 is not None:
        out.append(slice(s0, e0))
    return out


def smooth_rows(x, shot, h):
    """x (K,B): centred nan-mean over +-h frames of the same shot; NaN stays NaN."""
    if h <= 0:
        return x
    out = np.full_like(x, np.nan)
    for s_ in np.unique(shot):
        ix = np.flatnonzero(shot == s_)
        v = x[:, ix]
        n = v.shape[1]
        ok = np.isfinite(v)
        cs = np.concatenate([np.zeros((len(v), 1)), np.cumsum(np.where(ok, v, 0.0), 1)], 1)
        cn = np.concatenate([np.zeros((len(v), 1)), np.cumsum(ok, 1)], 1)
        lo = np.clip(np.arange(n) - h, 0, n); hi = np.clip(np.arange(n) + h + 1, 0, n)
        den = cn[:, hi] - cn[:, lo]
        sm = np.where(den > 0, (cs[:, hi] - cs[:, lo]) / np.maximum(den, 1), np.nan)
        out[:, ix] = np.where(ok, sm, np.nan)
    return out


def select_batch(S, I, Ip, R, te, ne, shot, cands, default, dvs=None, w_ne=0.0, h=0,
                 n_iter=100, levels=LEVELS_V1, resecant=False, dv_col=False, keep_all=False,
                 conv_tol=1e-2, o0Zs=None):
    """Label-free profile (and optional dv) selection for one whole-shot batch.

    S: fec.gs.Solver (with an extra uniform-Bz column in Mext when dv_col=True).
    I (B,C) coil amperes (without the dv column); Ip (B,); R/te/ne (B,N) Thomson chord data.
    cands: list of (beta0, am, an); default: index of the fallback candidate;
    dvs: optional per-candidate dv [T] (len(cands)); score = mm_Te + w_ne*mm_ne, optionally
    time-smoothed over +-h frames of the same shot; frames with no finite score -> default.
    resecant: re-run the force-free vertical secant for the chosen profile (2 extra solves).
    Returns dict(psi, psi_ax, psi_b, Rax, Zax, c, dI, Zt, choice, score(K,B)).
    """
    dev = S.dev
    B = len(Ip)
    K = len(cands)
    dvs = np.zeros(K) if dvs is None else np.asarray(dvs, float)
    Ix = np.concatenate([I, np.zeros((B, 1))], 1) if dv_col else np.asarray(I, float)
    gR = torch.as_tensor(S.spec.gR, device=dev, dtype=torch.float32)
    iz0 = int(np.argmin(np.abs(S.spec.gZ)))
    lev = torch.as_tensor(levels, device=dev, dtype=torch.float32)
    o0, Zs = o0Zs if o0Zs is not None else force_free_z(S, Ix, Ip,
                                                         np.tile(cands[default], (B, 1)))
    pte = to_dev(prep_profiles(R, te), dev)
    pne = to_dev(prep_profiles(R, ne), dev) if w_ne > 0 else None
    score = np.full((K, B), np.nan)
    keys = ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI")
    store = {}
    for k, combo in enumerate(cands):
        Ik = Ix.copy()
        if dv_col:
            Ik[:, -1] = dvs[k]
        o = S.solve(Ik, Ip, np.tile(combo, (B, 1)), Zs, n_iter=n_iter, I0=o0["I"], c0=o0["c"])
        mm = symmetry_mm(o["psi"], o["psi_ax"], o["psi_b"], gR, iz0, pte, lev)[0]
        if pne is not None:
            mm = mm + w_ne * symmetry_mm(o["psi"], o["psi_ax"], o["psi_b"], gR, iz0, pne,
                                         lev)[0]
        mm = torch.where(o["dI"] < conv_tol, mm, torch.full_like(mm, float("nan")))
        score[k] = mm.cpu().numpy()
        store[k] = {kk: o[kk].cpu() for kk in keys}
        if keep_all:
            store[k]["I"] = o["I"].cpu()
    sc = smooth_rows(score, np.asarray(shot), h)
    scf = np.where(np.isfinite(sc), sc, np.inf)
    choice = np.argmin(scf, 0)
    choice[~np.isfinite(scf).any(0)] = default
    out = {kk: torch.stack([store[int(choice[i])][kk][i] for i in range(B)]).numpy()
           for kk in keys}
    if keep_all:
        out["I"] = torch.stack([store[int(choice[i])]["I"][i] for i in range(B)]).numpy()
    out["Zt"] = Zs.copy()
    if resecant:
        prof = np.array([cands[int(c)] for c in choice], float)
        Ik = Ix.copy()
        if dv_col:
            Ik[:, -1] = dvs[choice]
        Ich = torch.as_tensor(out["I"], device=dev) if keep_all else o0["I"]
        cch = torch.as_tensor(out["c"], device=dev)
        o1 = S.solve(Ik, Ip, prof, Zs + 0.05, n_iter=60, I0=Ich, c0=cch)
        c0, c1 = out["c"], o1["c"].cpu().numpy()
        den = np.where(np.abs(c1 - c0) > 1e-12, c1 - c0, np.nan)
        Z2 = np.nan_to_num(np.clip(Zs - c0 * 0.05 / den, -0.3, 0.3), nan=0.0)
        Z2 = np.where(np.isfinite(den), Z2, Zs)
        o2 = S.solve(Ik, Ip, prof, Z2, n_iter=n_iter, I0=Ich, c0=cch)
        better = (o2["dI"].cpu().numpy() < conv_tol) & (np.abs(o2["c"].cpu().numpy()) <=
                                                       np.abs(out["c"]) + 1e-12)
        for kk in keys + (("I",) if keep_all else ()):
            v = o2[kk].cpu().numpy()
            out[kk] = np.where(better.reshape((-1,) + (1,) * (v.ndim - 1)), v, out[kk])
        out["Zt"] = np.where(better, Z2, Zs)
    out["choice"] = choice
    out["score"] = score
    return out


def refine_batch(S, Ix, Ip, R, te, prof0, I0, c0, Zs, deltas=(("b0", 0.1),),
                 bounds=((0.0, 0.8), (0.7, 4.0), (0.3, 4.0)), n_iter=60, levels=LEVELS_V1,
                 ne=None, w_ne=0.0, conv_tol=1e-2, pte=None, mm0=None, state0=None):
    """Batched coordinate-wise parabolic refinement of per-frame profile parameters.

    For each (parameter, delta): solve at p-delta and p+delta (per-frame profiles, warm-started
    from the current per-frame best current), fit a parabola through the three mismatch values,
    solve once more at the clamped vertex, keep the best of the evaluated points per frame.
    Cost: 3 batched solves per parameter.  Returns (prof, mm, state) with state = dict of the
    best per-frame solution tensors (psi, psi_ax, psi_b, Rax, Zax, c, dI, I).
    """
    dev = S.dev
    B = len(Ip)
    pidx = {"b0": 0, "am": 1, "an": 2}
    gR = torch.as_tensor(S.spec.gR, device=dev, dtype=torch.float32)
    iz0 = int(np.argmin(np.abs(S.spec.gZ)))
    lev = torch.as_tensor(levels, device=dev, dtype=torch.float32)
    pte = pte if pte is not None else to_dev(prep_profiles(R, te), dev)
    pne = to_dev(prep_profiles(R, ne), dev) if (ne is not None and w_ne > 0) else None
    keys = ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI", "I")

    def run(prof, Iw, cw):
        o = S.solve(Ix, Ip, prof, Zs, n_iter=n_iter, I0=Iw, c0=cw)
        mm = symmetry_mm(o["psi"], o["psi_ax"], o["psi_b"], gR, iz0, pte, lev)[0]
        if pne is not None:
            mm = mm + w_ne * symmetry_mm(o["psi"], o["psi_ax"], o["psi_b"], gR, iz0, pne,
                                         lev)[0]
        mm = torch.where(o["dI"] < conv_tol, mm, torch.full_like(mm, float("nan")))
        return mm.cpu().numpy(), {k: o[k] for k in keys}

    prof = np.asarray(prof0, float).copy()
    if state0 is None or mm0 is None:
        mm, st = run(prof, I0, c0)
    else:
        mm, st = np.asarray(mm0, float).copy(), {k: state0[k] for k in keys}
    inf = lambda x: np.where(np.isfinite(x), x, np.inf)

    def adopt(mask, prof_new, mm_new, st_new):
        m = torch.as_tensor(mask, device=dev)
        for k in keys:
            mk = m.reshape((-1,) + (1,) * (st[k].dim() - 1))
            st[k] = torch.where(mk, st_new[k], st[k])
        prof[mask] = prof_new[mask]
        mm[mask] = mm_new[mask]

    for name, d in deltas:
        j = pidx[name]
        lo, hi = bounds[j]
        base = prof.copy(); m0 = mm.copy()
        pm = base.copy(); pm[:, j] = np.clip(base[:, j] - d, lo, hi)
        pp = base.copy(); pp[:, j] = np.clip(base[:, j] + d, lo, hi)
        mmm, stm = run(pm, st["I"], st["c"])
        mmp, stp = run(pp, st["I"], st["c"])
        # parabola through (-d, mmm), (0, m0), (+d, mmp) in the actual clipped offsets
        xm = pm[:, j] - base[:, j]; xp = pp[:, j] - base[:, j]
        ym, y0, yp = inf(mmm), inf(m0), inf(mmp)
        ok3 = np.isfinite(ym) & np.isfinite(y0) & np.isfinite(yp) & (xm < 0) & (xp > 0)
        with np.errstate(all="ignore"):
            # divided differences
            a = ((yp - y0) / xp - (ym - y0) / xm) / (xp - xm)
            b = (yp - y0) / xp - a * xp
            xv = np.where(ok3 & (a > 0), -b / (2 * a), np.nan)
        xv = np.clip(np.nan_to_num(xv, nan=0.0), xm, xp)
        # adopt the better of the bracket points first
        better_m = ym < np.minimum(y0, yp)
        better_p = (yp < y0) & ~better_m
        adopt(better_m, pm, mmm, stm)
        adopt(better_p, pp, mmp, stp)
        pv = base.copy(); pv[:, j] = np.clip(base[:, j] + xv, lo, hi)
        need = ok3 & (np.abs(xv) > 1e-3 * d)
        if need.any():
            mmv, stv = run(pv, st["I"], st["c"])
            adopt(need & (inf(mmv) < inf(mm)), pv, mmv, stv)
    return prof, mm, st


# ----------------------------------------------------------------------------- kinetic fit v2
KF_PRIOR = np.array([0.2, 0.0, 0.8, 0.0, 0.0])     # basis mix (sum 1) + dv [units of KF_DV]
KF_DV = 0.01                                         # tesla per dv unit


def kfit2(S, Iext, Ip, Zt, pairs, n_iter=120, omega=0.3, kc=0.05, I0=None, c0=None,
          prior=KF_PRIOR, w_prior=(0.3, 0.3, 0.3, 0.3, 1.0), lm=2.0, pair_tol=0.02,
          dv_max=2.0, fit_dv=True, verbose=False):
    """Regularised Thomson-constrained free-boundary fit (fixes fec.kefit divergence).

    Same model as fec.kefit (4 non-hollow basis currents scaled to carry Ip + uniform dv field;
    equal-Te inboard/outboard pairs on the Z=0 chord must share psi), with
      * Levenberg-Marquardt rows lm*(u - u_prev) damping every Picard update,
      * priors strong enough to fix the near-null directions of the few pair rows,
      * positivity u[:4] >= 0 (clamp + renormalise to carry exactly Ip), |dv| <= dv_max units,
      * pair rows built from the vertically re-centred basis actually applied.
    pairs (B,P,2) radii from fec.kefit.te_pairs; NaN rows ignored.
    """
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
    u = pu.expand(B, 5).clone()
    if not fit_dv:
        w_prior = tuple(w_prior[:4]) + (1e4,)
    pw = t(w_prior)
    fb = (S.RR ** 2)[None] * (S.ZZ[None] - Zt[:, None, None])
    vfield = (S.RR ** 2 / 2 * KF_DV)[None]
    eye5 = torch.eye(5, device=dev, dtype=torch.float64)
    prev_mask = None
    for it in range(n_iter):
        psi = psi_ext + (I @ S.G.T).reshape(B, nz, nr) + c[:, None, None] * fb + \
            u[:, 4, None, None] * vfield
        psi_ax, Rax, Zax = S.find_axis(psi)
        xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
        (fu, vu, _, _), (fl, vl, _, _) = xp
        neg = torch.full_like(vu, -1e30)
        psi_b = torch.maximum(torch.where(fu, vu, neg), torch.where(fl, vl, neg))
        nox = ~(fu | fl)
        psi_b = torch.where(nox, psi_ax - 0.05 * psi_ax.abs().clamp_min(1e-3), psi_b)
        # seed the flood fill with the previous mask (re-grown from the axis every 10th
        # iteration), exactly as fec.gs.Solver.solve does
        seedm = prev_mask if (it % 10 != 0) else None
        mask, psiN = S.core_mask(psi, psi_ax, psi_b, Rax, Zax, xp, seed_mask=seedm)
        prev_mask = mask
        pn = psiN.reshape(B, -1)[:, S.sidx].clamp(0, 1)
        ms = mask.reshape(B, -1)[:, S.sidx].to(dt)
        x1 = (1 - pn ** 2) * ms
        x2 = (1 - pn) ** 2 * ms
        basis = torch.stack([Rs / R0 * x1, Rs / R0 * x2, R0 / Rs * x1, R0 / Rs * x2], 1)
        basis = basis / basis.sum(2, keepdim=True).clamp_min(1e-12) * Ip[:, None, None]
        # re-centre every basis function with the shift the current mix needs
        mix = torch.einsum("bk,bkn->bn", u[:, :4], basis)
        delta = Zt - (mix * Zs[None]).sum(1) / Ip
        full = torch.zeros(B * 4, Npix, device=dev, dtype=dt)
        full[:, S.sidx] = basis.reshape(B * 4, -1)
        full = S._shiftZ(full.reshape(B * 4, nz, nr), delta.repeat_interleave(4))
        basis = full.reshape(B, 4, Npix)[:, :, S.sidx]
        phi = (psi_ax - psi_b).abs().clamp_min(1e-4)
        ext = psi_ext + c[:, None, None] * fb
        A_p = torch.cat([torch.einsum("bpn,bkn->bpk", Gd, basis), r2d[..., None]], 2)
        b_p = -(chord(ext, 0) - chord(ext, 1))
        sc = (pvalid / (phi[:, None] * pair_tol))[..., None]
        A_ip = torch.cat([torch.ones(B, 1, 4, device=dev, dtype=dt),
                          torch.zeros(B, 1, 1, device=dev, dtype=dt)], 2)
        A = torch.cat([A_ip * 100.0, A_p * sc, torch.diag_embed(pw).expand(B, 5, 5),
                       lm * torch.eye(5, device=dev, dtype=dt).expand(B, 5, 5)], 1)
        bb = torch.cat([torch.full((B, 1), 100.0, device=dev, dtype=dt), b_p * sc[..., 0],
                        (pw * pu).expand(B, 5), lm * u], 1)
        A64 = torch.nan_to_num(A.double()); b64 = torch.nan_to_num(bb.double())
        AtA = A64.transpose(1, 2) @ A64 + 1e-9 * eye5
        unew = torch.linalg.solve(AtA, (A64.transpose(1, 2) @ b64[..., None])[..., 0]).to(dt)
        bad = ~torch.isfinite(unew).all(1)
        unew = torch.where(bad[:, None], u, unew)
        cu = unew[:, :4].clamp_min(0)
        cu = cu / cu.sum(1, keepdim=True).clamp_min(1e-6)
        unew = torch.cat([cu, unew[:, 4:].clamp(-dv_max, dv_max)], 1)
        u = (1 - omega) * u + omega * unew
        Inew = torch.einsum("bk,bkn->bn", u[:, :4], basis)
        Inew = Ip[:, None] * Inew / Inew.sum(1, keepdim=True).clamp_min(1e-30)
        Zc = (Inew * Zs[None]).sum(1) / Ip
        c = c + kc * (Zt - Zc)
        dI = (Inew - I).abs().sum(1) / Ip
        I = (1 - omega) * I + omega * Inew
        if verbose and (it % 20 == 0 or it == n_iter - 1):
            print(it, f"dI max {dI.max().item():.2e} med {dI.median().item():.1e} "
                  f"dv med {u[:, 4].median().item() * KF_DV * 1e3:.2f} mT")
    psi = psi_ext + (I @ S.G.T).reshape(B, nz, nr) + c[:, None, None] * fb + \
        u[:, 4, None, None] * vfield
    psi_ax, Rax, Zax = S.find_axis(psi)
    xp = S.find_xpoints(psi, Rax, Zax, psi_ax)
    (fu, vu, _, _), (fl, vl, _, _) = xp
    neg = torch.full_like(vu, -1e30)
    psi_b = torch.maximum(torch.where(fu, vu, neg), torch.where(fl, vl, neg))
    res = ((chord(psi, 0) - chord(psi, 1)) / (psi_ax - psi_b).abs()[:, None]) * pvalid
    return dict(psi=psi, I=I, c=c, u=u, psi_ax=psi_ax, Rax=Rax, Zax=Zax, psi_b=psi_b, dI=dI,
                pair_res=res, dv=u[:, 4] * KF_DV)


# ----------------------------------------------------------------------------- v2 selector
FFZ_PROF = (0.05, 2.0, 2.0)          # v1 default: profile used for the force-free height
V2_GRID = [(b, a, n) for b in (0.0, 0.15, 0.3, 0.5)
           for (a, n) in ((1.0, 1.0), (2.0, 2.0), (3.0, 1.0))]
V2_FALLBACK = V2_GRID.index((0.0, 2.0, 2.0))
V2_KFIT = dict(lm=5.0, n_iter=120)    # kfit2 settings (5 Te levels as fec.kefit.LEVELS)


def nearest_in_shot(choice, ok, shot):
    """Frames without a usable score inherit the choice of the nearest scored frame in the
    same shot (frames time-ordered); returns (choice, inherited mask)."""
    choice = choice.copy()
    inh = np.zeros(len(choice), bool)
    for s_ in np.unique(shot):
        ix = np.flatnonzero(shot == s_)
        good = ix[ok[ix]]
        if len(good) == 0 or len(good) == len(ix):
            continue
        for i in ix[~ok[ix]]:
            j = good[np.argmin(np.abs(good - i))]
            choice[i] = choice[j]
            inh[i] = True
    return choice, inh


def select_stacked(S, Ix, Ip, R, te, ne, grid, Zs, I0, c0, w_ne=1.0, n_iter=100,
                   levels=LEVELS_V1, conv_tol=1e-2, max_stack=1024):
    """Grid selection with (candidate x frame) stacked into batched solves of <= max_stack
    frames: identical per-frame numerics to select_batch, but a small sub-batch costs one
    solve instead of len(grid) (the solver's per-iteration latency dominates small batches).

    Ix (n,C) external currents (incl. any dv column); I0 (n,Ns) tensor, c0 (n,) tensor warm
    start; Zs (n,).  Returns dict(psi, psi_ax, psi_b, Rax, Zax, c, dI, score (K,n), choice).
    """
    dev = S.dev
    n, K = len(Ip), len(grid)
    gR = torch.as_tensor(S.spec.gR, device=dev, dtype=torch.float32)
    iz0 = int(np.argmin(np.abs(S.spec.gZ)))
    lev = torch.as_tensor(levels, device=dev, dtype=torch.float32)
    pte = to_dev(prep_profiles(R, te), dev)
    pne = to_dev(prep_profiles(R, ne), dev) if w_ne > 0 else None
    keys = ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI")
    score = np.full((K, n), np.nan)
    best = None
    bsc = np.full(n, np.inf)
    per = max(1, max_stack // max(n, 1))
    rep = lambda pd, m: {k: (v.repeat(m) if v.dim() == 1 else v.repeat(m, 1))
                         for k, v in pd.items()}
    for k0 in range(0, K, per):
        ks = list(range(k0, min(K, k0 + per)))
        m = len(ks)
        prof = np.repeat(np.asarray([grid[k] for k in ks], float), n, axis=0)
        o = S.solve(np.tile(Ix, (m, 1)), np.tile(Ip, m), prof, np.tile(Zs, m),
                    n_iter=n_iter, I0=I0.repeat(m, 1), c0=c0.repeat(m))
        mm = symmetry_mm(o["psi"], o["psi_ax"], o["psi_b"], gR, iz0, rep(pte, m), lev)[0]
        if pne is not None:
            mm = mm + w_ne * symmetry_mm(o["psi"], o["psi_ax"], o["psi_b"], gR, iz0,
                                         rep(pne, m), lev)[0]
        mm = torch.where(o["dI"] < conv_tol, mm, torch.full_like(mm, float("nan")))
        mm = mm.cpu().numpy().reshape(m, n)
        vals = {kk: o[kk].cpu().numpy().reshape((m, n) + tuple(o[kk].shape[1:])) for kk in keys}
        if best is None:
            best = {kk: v[0].copy() for kk, v in vals.items()}
        for j, k in enumerate(ks):
            score[k] = mm[j]
            better = np.isfinite(mm[j]) & (mm[j] < bsc)
            if better.any():
                for kk in keys:
                    best[kk][better] = vals[kk][j][better]
                bsc[better] = mm[j][better]
        del o
    choice = np.argmin(np.where(np.isfinite(score), score, np.inf), 0)
    out = dict(best)
    out["score"] = score
    out["choice"] = choice
    return out


def select_v2(S, I, Ip, R, te, ne, shot, grid=None, kfit_kw=None, w_ne=1.0, n_iter=100,
              combine="pairs", min_pairs=3, use_kfit=True, use_grid=True, dv_col=False):
    """MAST v2 label-free equilibrium for one whole-shot batch (frames sorted by shot, time).

    1. force-free height Zs from the v1 default profile (vertical secant, 2 solves);
    2. kfit2: regularised Thomson-pair kinetic fit (4 non-hollow basis currents + uniform
       vertical-field nuisance dv; Te pairs at 5 levels as fec.kefit), warm-started from 1;
    3. grid: 12 FreeGS-style shapes (beta0 x (am,an)), score = Te + w_ne*ne symmetry mismatch
       (17 psiN levels 0.05-0.85, GPU); frames without a finite score inherit the nearest
       scored frame's shape in the same shot (else V2_FALLBACK);
    4. combine="pairs" (default, best in the twin): keep kfit2 where it had >= min_pairs Te
       pairs and converged, otherwise the grid result; the grid is solved only on that
       sub-batch.  combine="score": grid on all frames, per frame the lower score wins.
    Cost per batch: 2 + ~2 (kfit2) + 12 x (fraction of frames needing the grid) solves of
    100 iterations.  Inputs only: coil currents, Ip, Thomson Te/ne on the Z=0 chord.
    dv_col: S.Mext carries an extra (zeroed) uniform-Bz column (twin solver).
    Returns a dict of per-frame arrays (method 1 = kfit2, 0 = grid).
    """
    from fec.kefit import LEVELS, te_pairs
    dev = S.dev
    B = len(Ip)
    shot = np.asarray(shot)
    grid = V2_GRID if grid is None else grid
    kw = dict(V2_KFIT, **(kfit_kw or {}))
    levels_k = kw.pop("levels", LEVELS)
    Ix = np.concatenate([I, np.zeros((B, 1))], 1) if dv_col else np.asarray(I, float)
    o0, Zs = force_free_z(S, Ix, Ip, np.tile(FFZ_PROF, (B, 1)))
    keys = ("psi", "psi_ax", "psi_b", "Rax", "Zax", "c", "dI")
    gR = torch.as_tensor(S.spec.gR, device=dev, dtype=torch.float32)
    iz0 = int(np.argmin(np.abs(S.spec.gZ)))
    lev = torch.as_tensor(LEVELS_V1, device=dev, dtype=torch.float32)
    out = dict(Zt=Zs.copy(), method=np.zeros(B, int), score=np.full(B, np.nan),
               grid_choice=np.full(B, -1), grid_score=np.full(B, np.nan),
               inherited=np.zeros(B, bool), kfit_dv=np.full(B, np.nan),
               kfit_score=np.full(B, np.nan), kfit_npairs=np.zeros(B, int))
    need_grid = np.ones(B, bool)
    if use_kfit:
        pairs = np.stack([te_pairs(R[i], te[i], levels=levels_k) for i in range(B)])
        o = kfit2(S, Ix, Ip, Zs, pairs, I0=o0["I"], c0=o0["c"], **kw)
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
                   kfit_score=mk, kfit_npairs=npairs)
        if combine == "pairs":
            need_grid = ~((npairs >= min_pairs) & conv)
        del o
    if use_grid and need_grid.any():
        sub = np.flatnonzero(need_grid) if use_kfit else np.arange(B)
        st = torch.as_tensor(sub, device=dev)
        g = select_stacked(S, Ix[sub], Ip[sub], R[sub], te[sub], ne[sub], grid, Zs[sub],
                           o0["I"][st], o0["c"][st], w_ne=w_ne, n_iter=n_iter)
        okg = np.isfinite(g["score"]).any(0)
        ch = np.where(okg, g["choice"], V2_FALLBACK)
        ch, inh = nearest_in_shot(ch, okg, shot[sub])
        fix = np.flatnonzero(~okg)
        if len(fix):
            # unscored frames: neighbour's shape (same shot) or the fallback shape
            ft = torch.as_tensor(sub[fix], device=dev)
            prof = np.array([grid[int(c)] for c in ch[fix]], float)
            os_ = S.solve(Ix[sub[fix]], Ip[sub[fix]], prof, Zs[sub[fix]], n_iter=n_iter,
                          I0=o0["I"][ft], c0=o0["c"][ft])
            for k in keys:
                g[k][fix] = os_[k].cpu().numpy()
        J = g["score"][ch, np.arange(len(sub))]
        out["grid_choice"][sub] = ch
        out["grid_score"][sub] = J
        out["inherited"][sub] = inh
        if use_kfit and combine == "score":
            mk = out["kfit_score"][sub]
            take = ~(np.isfinite(mk) & (~np.isfinite(J) | (mk < J)))
        else:
            take = np.ones(len(sub), bool)
        tsub = sub[take]
        for k in keys:
            if k not in out:
                out[k] = np.zeros((B,) + g[k].shape[1:], g[k].dtype)
            out[k][tsub] = g[k][take]
        out["method"][tsub] = 0
        out["score"][tsub] = J[take]
    return out


def stage1_v2(out_dir, limit=0, max_frames=1024, config="mast_public_test"):
    """Production driver (mirrors fec.mast_v1.stage1 outputs): psi.npy + meta.npz."""
    import time
    from fec.features import load_ds
    from fec.gs import Solver
    from fec.machines import MAST_CH, MAST_COILS, mast_setup
    from fec.mast_v1 import thomson_frame
    from fec.paths import WORK
    ds = load_ds(config)
    info = ds["info"]
    N = len(ds["t_ms"]) if not limit else limit
    cur = np.asarray(ds["cur"][:N], dtype=np.float64)[:, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    spec, M = mast_setup()
    S = Solver(spec, M)
    out_dir.mkdir(parents=True, exist_ok=True)
    psi_out = np.lib.format.open_memmap(out_dir / "psi.npy", "w+", np.float32, (N, 65, 65))
    mk = ("psi_ax", "psi_b", "Rax", "Zax", "Zt", "c", "dI", "method", "score", "grid_choice",
          "grid_score", "inherited", "kfit_dv", "kfit_score", "kfit_npairs")
    meta = {k: np.full(N, np.nan) for k in mk}
    shot = np.asarray(ds["shot"][:N])
    t0 = time.time()
    cache = {}
    for sl in shot_chunks(shot, max_frames):
        I = cur[sl][:, ci] * 1e3
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
        o = select_v2(S, I, Ip, pad(Rl), pad(tel), pad(nel), shot[sl])
        psi_out[sl] = o["psi"]
        for k in mk:
            if k in o:
                meta[k][sl] = o[k]
        print(f"{sl.stop}/{N} {time.time() - t0:.0f}s kfit used {np.mean(o['method']):.2f} "
              f"score med {np.nanmedian(o['score']):.3f}", flush=True)
        torch.cuda.empty_cache()
    psi_out.flush()
    np.savez(out_dir / "meta.npz", **meta)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fec.paths import WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORK / "mast_v2"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default="mast_public_test")
    a = ap.parse_args()
    stage1_v2(Path(a.out), limit=a.limit, config=a.config)
