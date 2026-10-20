"""mmk_op: member-mean shape finishing of an ensemble psi map (amendment 57 Stage 1, agent mmk1, 2026-10-19).

Self-contained operator for later packaging. The operator itself (weights, warps, bisection) is numpy/scipy only,
deterministic and offline. The shape functional it solves on is the challenge scorer's OWN LCFS extraction and shape
read-out (fusion_scoring lcfs.extract_lcfs_with_sign / derive.magnetic_axis / derive._shape_from_contour /
derive._clamp_to_nan; scikit-image marching squares as the scorer pins it), evaluated with the TRAIN-only envelope
(`env_train.npz` "env"), never the scorer mask. `scoring_dir` points at the starter's `fusion_scoring` directory.

Per frame (psi float32 (65,65), DIII-D convention: axis = minimum of psi, scorer axis_sign = -1):
  f(psi)   = (R_axis, kappa, tri_top, tri_bot, contour) with the TRAIN envelope (bit-identical to
             fec/cons_project.derive_env for kappa/tri_top/tri_bot/R_axis).
  weights  psi_N = (psi - psi[iz,ir]) / (psi_b - psi[iz,ir]); psi_b = mean bilinear psi on the env contour;
           w_psi = 1 (psi_N <= 1), 0.5(1 + cos(pi (psi_N - 1)/0.4)) (1 < psi_N < 1.4), 0 (psi_N >= 1.4), restricted
           to the 8-connected component of {psi_N < 1.4} that contains the O-point cell (zero elsewhere);
           w_Z = exp(-(Z - Z_in)^2 / 2 sigma^2) for R < R_ax, exp(-(Z - Z_out)^2 / 2 sigma^2) for R >= R_ax,
           sigma = 0.25 m, (R_in, Z_in) / (R_out, Z_out) = the env contour points of minimum / maximum R;
           w = w_psi * w_Z.  All weights are computed once from the operator's own input map.
  kappa    psi'(R,Z) = psi(R + (R - R_ax) w (1/c - 1), Z)  [= psi(R_ax + (R - R_ax)(1 + w(1/c - 1)), Z)],
           c by bisection on [0.9, 1.1] so that kappa_env(psi') = kappa0 + alpha (T_kappa - kappa0), alpha = 0.5.
  delta    (on the float32 kappa-edited map) psi'(R,Z) = psi(R - dR w g, Z), g = smoothstep(s) = 3s^2 - 2s^3,
           s = clip((R - R_ax)/(R_out - R_ax), 0, 1) for R >= R_ax, clip((R_ax - R)/(R_ax - R_in), 0, 1) for R < R_ax
           (g = 0 at the axis, g = 1 at and beyond each width extreme, C1 at both ends); dR by bisection on
           [-0.03, 0.03] m so that dm = (tri_top + tri_bot)/2 moves dm0 + alpha (T_delta - dm0).
  interp   row-wise (fixed Z) not-a-knot cubic spline of the source map along R (scipy CubicSpline coefficients,
           end-interval polynomials beyond the grid); nodes whose displacement is exactly 0 copy the source value, so
           c = 1 / dR = 0 is the exact identity.
  bisection standard: g(x) = f(psi'(x)) - target must be finite at both bracket ends with opposite signs (else skip
           "no admissible root"); <= 30 midpoint iterations (the first midpoint is the identity); converged iff
           |g| <= tol = max(1e-4 |T - f0|, 1e-6); not converged after 30 -> skip; NaN at a midpoint -> skip.
  re-check the solution map is cast to float32 and f is recomputed: NaN -> skip (extraction failure on the edited
           map); (f1 - f0)(T - f0) <= 0 -> skip (realised move of the wrong sign).
  skip     leaves that frame unchanged for that operator (bit-identical float32 input).

Status codes (per operator): see STATUS.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline

ALPHA = 0.5
SIGMA_Z = 0.25
PSIN_TAPER = (1.0, 1.4)
C_BRACKET = (0.9, 1.1)
DR_BRACKET = (-0.03, 0.03)
MAXIT = 30
RTOL, ATOL = 1e-4, 1e-6
N_ITER = 22                   # scorer's LCFS bisection depth (local_score.N_ITER)
MACHINE = "DIII-D"
AXIS_SIGN = -1.0

STATUS = {0: "ok", 1: "at_target",
          2: "extract_fail_orig",       # (c) env extraction failure on the operator's input map (or degenerate geometry)
          3: "no_root_end_nan",         # (a) no admissible root: bracket end not finite
          4: "no_root_no_sign_change",  # (a) no admissible root: g has the same sign at both bracket ends
          5: "no_root_no_conv",         # (a) no admissible root: tolerance not met in 30 iterations (jump)
          6: "no_root_mid_nan",         # (a) no admissible root: extraction failed at a bisection midpoint
          7: "wrong_sign",              # (b) realised move (float32 map) of the wrong sign (or zero)
          8: "extract_fail_edit",       # (c) env extraction failure on the float32 edited map
          9: "target_nan"}              # target not finite (member extraction failures; 0 frames on VAL)
SKIP_CLASS = {2: "c_extract", 3: "a_no_root", 4: "a_no_root", 5: "a_no_root", 6: "a_no_root", 7: "b_wrong_sign",
              8: "c_extract", 9: "target_nan"}


# ------------------------------------------------------------------------------------------------ functional
class EnvFunctional:
    """Scorer shape functional with the TRAIN envelope. __call__(psi64) -> dict (NaN shape on failure)."""

    def __init__(self, R, Z, env_mask, scoring_dir=None):
        if scoring_dir is not None and str(scoring_dir) not in sys.path:
            sys.path.insert(0, str(scoring_dir))
        import derive  # noqa: E402  (fusion_scoring)
        import lcfs    # noqa: E402
        self.derive, self.lcfs = derive, lcfs
        self.R = np.asarray(R, np.float64)
        self.Z = np.asarray(Z, np.float64)
        self.mc = np.asarray(env_mask, bool)
        self.mf = self.mc.astype(np.float64)
        self.ncalls = 0

    def __call__(self, psi):
        self.ncalls += 1
        psi = np.asarray(psi, np.float64)
        ex, _ = self.lcfs.extract_lcfs_with_sign(psi, self.R, self.Z, MACHINE, self.mc, self.mf, n_iter=N_ITER,
                                                 axis_sign=AXIS_SIGN)
        Ra, Za, iz, ir = self.derive.magnetic_axis(psi, self.R, self.Z, MACHINE, self.mc, AXIS_SIGN)
        out = {"R_axis": Ra, "Z_axis": Za, "iz": iz, "ir": ir, "contour": ex,
               "kappa": np.nan, "tri_top": np.nan, "tri_bot": np.nan, "volume": np.nan}
        if ex is not None and len(ex) >= 8:
            sh = self.derive._clamp_to_nan(dict(self.derive._shape_from_contour(ex)))
            out.update(kappa=sh["kappa"], tri_top=sh["tri_top"], tri_bot=sh["tri_bot"], volume=sh["volume"])
        return out

    @staticmethod
    def dmean(info):
        return 0.5 * (info["tri_top"] + info["tri_bot"])


def load_env(env_train):
    """env_train: path to env_train.npz, a mapping with grid_R/grid_Z/env, or a (R, Z, mask) tuple."""
    if isinstance(env_train, (str, Path)):
        e = np.load(env_train)
        return e["grid_R"].astype(np.float64), e["grid_Z"].astype(np.float64), e["env"].astype(bool)
    if isinstance(env_train, tuple):
        R, Z, m = env_train
        return np.asarray(R, np.float64), np.asarray(Z, np.float64), np.asarray(m, bool)
    return (np.asarray(env_train["grid_R"], np.float64), np.asarray(env_train["grid_Z"], np.float64),
            np.asarray(env_train["env"], bool))


# ------------------------------------------------------------------------------------------------ geometry
def taper(psiN):
    lo, hi = PSIN_TAPER
    x = np.clip((psiN - lo) / (hi - lo), 0.0, 1.0)
    w = 0.5 * (1.0 + np.cos(np.pi * x))
    return np.where(psiN <= lo, 1.0, np.where(psiN >= hi, 0.0, w))


def frame_weights(psi, info, R, Z):
    """Localisation weights of one map from its own env functional output; None if degenerate."""
    C = info["contour"]
    if C is None or len(C) < 8:
        return None
    iz, ir = int(info["iz"]), int(info["ir"])
    R_ax = float(info["R_axis"])
    psi_ax = float(psi[iz, ir])
    psi_b = float(np.mean(_bilinear(psi, R, Z, C[:, 0], C[:, 1])))
    den = psi_b - psi_ax
    if not (np.isfinite(R_ax) and np.isfinite(den)) or den == 0.0:
        return None
    psiN = (psi - psi_ax) / den
    i_in, i_out = int(np.argmin(C[:, 0])), int(np.argmax(C[:, 0]))
    R_in, Z_in = float(C[i_in, 0]), float(C[i_in, 1])
    R_out, Z_out = float(C[i_out, 0]), float(C[i_out, 1])
    if not (R_in < R_ax < R_out):
        return None
    lab, _ = ndimage.label(psiN < PSIN_TAPER[1], structure=np.ones((3, 3), int))
    comp = (lab == lab[iz, ir]) & (lab[iz, ir] > 0)
    wpsi = np.where(comp, taper(psiN), 0.0)
    RR, ZZ = np.meshgrid(R, Z)
    inb = RR < R_ax
    wz = np.where(inb, np.exp(-(ZZ - Z_in) ** 2 / (2 * SIGMA_Z ** 2)), np.exp(-(ZZ - Z_out) ** 2 / (2 * SIGMA_Z ** 2)))
    w = wpsi * wz
    s = np.where(inb, (R_ax - RR) / (R_ax - R_in), (RR - R_ax) / (R_out - R_ax))
    s = np.clip(s, 0.0, 1.0)
    g = s * s * (3.0 - 2.0 * s)
    return {"w": w, "g": g, "RR": RR, "R_ax": R_ax, "R_in": R_in, "R_out": R_out, "Z_in": Z_in, "Z_out": Z_out,
            "psi_ax": psi_ax, "psi_b": psi_b, "nw": int((w > 0).sum())}


def _bilinear(field, R, Z, rq, zq):
    """bilinear sample (same arithmetic as fusion_scoring derive._bilinear)"""
    nZ, nR = field.shape
    fr = np.interp(rq, R, np.arange(nR))
    fz = np.interp(zq, Z, np.arange(nZ))
    r0 = np.clip(np.floor(fr).astype(int), 0, nR - 2)
    z0 = np.clip(np.floor(fz).astype(int), 0, nZ - 2)
    tr = fr - r0
    tz = fz - z0
    return (field[z0, r0] * (1 - tr) * (1 - tz) + field[z0, r0 + 1] * tr * (1 - tz)
            + field[z0 + 1, r0] * (1 - tr) * tz + field[z0 + 1, r0 + 1] * tr * tz)


class RowSpline:
    """Not-a-knot cubic spline of each Z-row of psi along R (scipy CubicSpline coefficients), vectorised eval."""

    def __init__(self, psi, R):
        cs = CubicSpline(np.asarray(R, np.float64), np.asarray(psi, np.float64), axis=1)
        self.x = cs.x
        self.c = cs.c                         # (4, nR-1, nZ)

    def __call__(self, Rq, rows):
        k = np.clip(np.searchsorted(self.x, Rq, side="right") - 1, 0, len(self.x) - 2)
        t = Rq - self.x[k]
        c = self.c[:, k, rows]
        return ((c[0] * t + c[1]) * t + c[2]) * t + c[3]


def warp(psi, spline, RR, disp):
    """psi'(node) = spline(R_node + disp) where disp != 0; exact copy elsewhere."""
    out = np.array(psi, np.float64, copy=True)
    m = disp != 0.0
    if m.any():
        rows = np.nonzero(m)[0]
        out[m] = spline(RR[m] + disp[m], rows)
    return out


def kappa_disp(geo, c):
    return (geo["RR"] - geo["R_ax"]) * geo["w"] * (1.0 / c - 1.0)


def delta_disp(geo, dR):
    return -dR * geo["w"] * geo["g"]


# ------------------------------------------------------------------------------------------------ bisection
def bisect(F, lo, hi, x_id, f0, target, tol, maxit=MAXIT):
    """Standard bisection of g(x) = F(x)[0] - target on [lo, hi]. F(x) -> (value, map). x_id = identity point whose
    value f0 is known exactly (identity map). Returns (status, x, map, value, n_mid)."""
    def ev(x):
        if x == x_id:
            return f0, None
        return F(x)
    flo, plo = ev(lo)
    fhi, phi_ = ev(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        return 3, np.nan, None, np.nan, 0
    glo, ghi = flo - target, fhi - target
    if abs(glo) <= tol:
        return 0, lo, plo, flo, 0
    if abs(ghi) <= tol:
        return 0, hi, phi_, fhi, 0
    if np.sign(glo) == np.sign(ghi):
        return 4, np.nan, None, np.nan, 0
    for it in range(1, maxit + 1):
        mid = 0.5 * (lo + hi)
        fm, pm = ev(mid)
        if not np.isfinite(fm):
            return 6, np.nan, None, np.nan, it
        gm = fm - target
        if abs(gm) <= tol:
            if pm is None:                     # identity already within tolerance (handled before; safety)
                return 1, mid, None, fm, it
            return 0, mid, pm, fm, it
        if np.sign(gm) == np.sign(glo):
            lo, glo = mid, gm
        else:
            hi, ghi = mid, gm
    return 5, np.nan, None, np.nan, maxit


# ------------------------------------------------------------------------------------------------ one frame
FLAG_KEYS = ("k_status", "k_c", "k_f0", "k_T", "k_target", "k_solved", "k_real", "k_nmid",
             "d_status", "d_dR", "d_f0", "d_T", "d_target", "d_solved", "d_real", "d_nmid",
             "kappa_after_delta", "R_axis0", "R_axis_final", "n_calls", "seconds")


def _step(psi32, T, func, which):
    """One operator on a float32 map. Returns (psi32_out, flags dict)."""
    psi = psi32.astype(np.float64)
    fl = {"status": 2, "x": np.nan, "f0": np.nan, "T": float(T), "target": np.nan, "solved": np.nan,
          "real": np.nan, "nmid": 0}
    info0 = func(psi)
    f0 = info0["kappa"] if which == "k" else func.dmean(info0)
    fl["f0"] = float(f0)
    fl["info0"] = info0
    if not np.isfinite(f0):
        fl["status"] = 2
        return psi32, fl
    if not np.isfinite(T):
        fl["status"] = 9
        return psi32, fl
    target = f0 + ALPHA * (T - f0)
    tol = max(RTOL * abs(T - f0), ATOL)
    fl["target"] = float(target)
    if abs(target - f0) <= tol:
        fl["status"] = 1
        return psi32, fl
    geo = frame_weights(psi, info0, func.R, func.Z)
    if geo is None:
        fl["status"] = 2
        return psi32, fl
    spline = RowSpline(psi, func.R)

    if which == "k":
        def F(c):
            p = warp(psi, spline, geo["RR"], kappa_disp(geo, c))
            return func(p)["kappa"], p
        lo, hi, xid = C_BRACKET[0], C_BRACKET[1], 1.0
    else:
        def F(dR):
            p = warp(psi, spline, geo["RR"], delta_disp(geo, dR))
            return func.dmean(func(p)), p
        lo, hi, xid = DR_BRACKET[0], DR_BRACKET[1], 0.0
    st, x, p, fs, nmid = bisect(F, lo, hi, xid, f0, target, tol)
    fl["nmid"] = nmid
    if st != 0:
        fl["status"] = st
        return psi32, fl
    fl["x"], fl["solved"] = float(x), float(fs)
    p32 = p.astype(np.float32)
    info1 = func(p32.astype(np.float64))
    f1 = info1["kappa"] if which == "k" else func.dmean(info1)
    fl["real"] = float(f1)
    fl["info1"] = info1
    if not np.isfinite(f1):
        fl["status"] = 8
        return psi32, fl
    if (f1 - f0) * (T - f0) <= 0:
        fl["status"] = 7
        return psi32, fl
    fl["status"] = 0
    return p32, fl


def mmk_frame(psi32, T_kappa, T_delta, func):
    """kappa step, then delta step on the float32 kappa-edited map. Returns (psi_kappa32, psi_final32, flags)."""
    t0 = time.perf_counter()
    n0 = func.ncalls
    psi32 = np.asarray(psi32, np.float32).reshape(func.Z.size, func.R.size)
    pk, fk = _step(psi32, T_kappa, func, "k")
    pd, fd = _step(pk, T_delta, func, "d")
    fl = {"k_status": fk["status"], "k_c": fk["x"], "k_f0": fk["f0"], "k_T": fk["T"], "k_target": fk["target"],
          "k_solved": fk["solved"], "k_real": fk["real"], "k_nmid": fk["nmid"],
          "d_status": fd["status"], "d_dR": fd["x"], "d_f0": fd["f0"], "d_T": fd["T"], "d_target": fd["target"],
          "d_solved": fd["solved"], "d_real": fd["real"], "d_nmid": fd["nmid"]}
    # information only (no decision): kappa / R_axis of the final map (env)
    if fd["status"] == 0:
        fin = fd["info1"]
    else:
        fin = fd.get("info0")
    fl["kappa_after_delta"] = float(fin["kappa"]) if fin is not None else np.nan
    fl["R_axis_final"] = float(fin["R_axis"]) if fin is not None else np.nan
    fl["R_axis0"] = float(fk["info0"]["R_axis"]) if fk.get("info0") is not None else np.nan
    fl["n_calls"] = func.ncalls - n0
    fl["seconds"] = time.perf_counter() - t0
    return pk, pd, fl


# ------------------------------------------------------------------------------------------------ targets + API
def member_targets(member_psis, func):
    """T_kappa, T_delta = equal (NaN-ignoring) member means of kappa_env and (tri_top_env + tri_bot_env)/2.
    member_psis: (M, N, 65, 65) or (M, N, 4225)."""
    M, N = member_psis.shape[0], member_psis.shape[1]
    k = np.full((N, M), np.nan)                 # (frame, member) layout, reduced over axis 1
    d = np.full((N, M), np.nan)
    for m in range(M):
        for i in range(N):
            inf = func(np.asarray(member_psis[m, i], np.float64).reshape(func.Z.size, func.R.size))
            k[i, m] = inf["kappa"]
            d[i, m] = func.dmean(inf)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(k, 1), np.nanmean(d, 1)


def apply_mmk(psi_ens, member_psis, R, Z, env_train, scoring_dir=None, targets=None, return_kappa_only=False):
    """Member-mean shape finishing of an ensemble map.

    psi_ens: (N, 65, 65) or (N, 4225) float32 ensemble maps; member_psis: (M, N, ...) member maps (used only for the
    targets; may be None when `targets` = (T_kappa, T_delta) is given); R, Z: grid vectors; env_train: TRAIN envelope
    (path / mapping with grid_R, grid_Z, env / (R, Z, mask)); scoring_dir: the starter's fusion_scoring directory.
    Returns (psi_edited float32 like psi_ens, flags dict of (N,) arrays) [+ psi_kappa_only if return_kappa_only].
    """
    eR, eZ, mask = load_env(env_train)
    R = np.asarray(R, np.float64)
    Z = np.asarray(Z, np.float64)
    if not (np.allclose(eR, R, rtol=0, atol=1e-6) and np.allclose(eZ, Z, rtol=0, atol=1e-6)):
        raise ValueError("env_train grid differs from R, Z")
    func = EnvFunctional(R, Z, mask, scoring_dir)
    shp = np.shape(psi_ens)
    P = np.asarray(psi_ens, np.float32).reshape(shp[0], Z.size, R.size)
    if targets is None:
        Tk, Td = member_targets(np.asarray(member_psis), func)
    else:
        Tk, Td = (np.asarray(t, np.float64) for t in targets)
    out = np.empty_like(P)
    outk = np.empty_like(P) if return_kappa_only else None
    flags = {k: np.full(shp[0], np.nan) for k in FLAG_KEYS}
    for i in range(shp[0]):
        pk, pd, fl = mmk_frame(P[i], Tk[i], Td[i], func)
        out[i] = pd
        if outk is not None:
            outk[i] = pk
        for k in FLAG_KEYS:
            flags[k][i] = fl[k]
    flags["k_status"] = flags["k_status"].astype(int)
    flags["d_status"] = flags["d_status"].astype(int)
    res = (out.reshape(shp), flags)
    if return_kappa_only:
        res = res + (outk.reshape(shp),)
    return res
