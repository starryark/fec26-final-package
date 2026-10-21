"""1-D poloidal-flux (current) diffusion: a label-free l_i / current-profile model from the measured
plasma current Ip(t) and Thomson Te(R, t).  Plan, data boundary and PRE-DECLARED gates:
docs/research/CURRENT_DIFFUSION_PROBE_20261009.md (family `current-diffusion-li`).  CPU only.

Model (large aspect ratio, similar elliptic flux surfaces)
---------------------------------------------------------
State: the enclosed toroidal current I(rho, t), rho = r / a the normalised minor radius of nested
similar ellipses (horizontal half-width r, elongation kappa; rho = sqrt(A / A_lcfs) is the
area-equivalent coordinate).  Ampere (perimeter-integrated, B_p = |grad psi| / R0), Faraday and the
parallel Ohm's law E = eta j (no bootstrap / NBI / ECCD current) give

    dI/dt = f_k * rho / (mu0 a^2) * d/drho[ (eta / rho) dI/drho ],     f_k = (1 + kappa^2) / (2 kappa^2)

with I(0, t) = 0 and I(1, t) = Ip(t): the measured total current is imposed EXACTLY at every step
(== the Neumann condition on the flux, dpsi/drho|_1 = mu0 R0 Ip a / G(a)).  a(t), kappa(t), R0(t)
enter as scalars of the diffusivity; the advective (grid-motion) term of a moving boundary is
dropped (declared simplification).  Backward Euler (L-stable: eta spans 2-3 decades across rho),
finite volume on N = 64 cells, sub-steps <= DT_MAX between the native Thomson laser times (EFIT
output times are merged into the step grid, so outputs land exactly on them).  The variants
(Zeff x initial condition) are solved together as one block-tridiagonal LAPACK gtsv call.

Outputs per EFIT frame:
  li_cyl = 2 int_0^1 (I / Ip)^2 drho / rho   (= li(2) = <B_p^2>_V / (mu0 Ip / l_p)^2 for similar
           ellipses with l_p = 2 pi a sqrt((1 + kappa^2) / 2); uniform j -> 0.5, j ~ 1 - rho^2 -> 11/12)
  j(psiN) on 21 points, j / <j> (<j> = Ip / area), psiN(rho) = int_0^rho I / rho' / int_0^1 I / rho'
           (the diffused profile's own flux; NaN if I <= 0 anywhere inside, i.e. psi not monotone).

Resistivity: Sauter, Angioni & Lin-Liu, Phys. Plasmas 6, 2834 (1999) (+ 2002 erratum)
  sigma_Sp = 1.9012e4 Te^1.5 / (Z N(Z) lnL_e) [S/m, Te eV],  N(Z) = 0.58 + 0.74 / (0.76 + Z),
  lnL_e = 31.3 - ln(sqrt(n_e) / Te)
  sigma_neo / sigma_Sp = F33(X) = 1 - (1 + 0.36/Z) X + (0.59/Z) X^2 - (0.23/Z) X^3,
  X = f_t / [1 + (0.55 - 0.1 f_t) sqrt(nu*_e) + 0.45 (1 - f_t) nu*_e / Z^1.5],
  nu*_e = 6.921e-18 q R0 n_e Z lnL_e / (Te^2 eps^1.5),
  f_t = 1 - (1 - eps)^2 / (sqrt(1 - eps^2) (1 + 1.46 sqrt(eps)))   (= fec.d3d_physics.trapped_fraction)
  eps = rho a / R0 = rho / A.  Aspect-ratio dependence (collisionless, Z = 2): eta_neo / eta_Sp =
  1 / F33(f_t) = 2.6 at rho 0.5 and 4.2 at rho 0.9 for DIII-D (A = 2.7; f_t 0.58 / 0.74), but 4.6
  and 14.0 for MAST (A = 1.4; f_t 0.76 / 0.92): in the collisionless limit trapped particles raise
  MAST's resistivity far more.  Collisionality (nu*_e, with q from the previous step's diffused profile,
  q = 2 pi r^2 B0 (1 + kappa^2) / 2 / (mu0 R0 I), clipped to [0.5, 50], B0 = F / R0 from the TF /
  bcoil current) restores Spitzer at the cold edge.  Te floor TE_FLOOR (10 eV), n_e floor 1e18.
  Zeff is a flat nuisance in ZEFFS = (1.5, 2, 3).

Initial conditions at the first valid Thomson frame with Ip >= IP_START_FRAC x max Ip:
'flat' (j uniform, I = Ip rho^2) and 'te15' (j ~ Te^1.5).  Variant v = iz * 2 + ic.

Thomson mapping (R, Z) -> rho:
  * at map frames: OUR label-free maps (DIII-D testbed KB_A1_ffz maps of fec.betap_li; MAST
    fec_work/mastz/cand_zsc_z0/psi.npy); psiN at the chord by bilinear interpolation, then the map's
    own area function rho = sqrt(A(psiN) / A(1)) over the core region (4-connected psiN < 1
    component of the axis node, cut at the axis column's top / bottom LCFS crossings);
    between map frames each channel's rho is linear in time;
  * before the first map frame: MAST (midplane radial array) -> symmetric-profile estimate from the
    Thomson profile itself (centre = the R that makes the smoothed Te(R) most symmetric, edges = the
    inboard / outboard crossings of max(20 eV, 0.05 Te_peak), halves matched in normalised
    distance); DIII-D (vertical chord at R 1.94 m,
    Z 0.03-0.83, plus outboard tangential chords: no symmetric estimate is possible) -> the first
    map's channel rho is held.
  Te(rho), n_e(rho) per laser event: medians in NB rho bins, Te made non-increasing (weighted PAV),
  constant inside the innermost bin, linear to rho = 1 from the last two bins (clipped to
  [TE_FLOOR, last]).  Te is interpolated linearly in time between laser events; EFIT outputs farther
  than GAP_MS (25 ms) from the nearest valid laser event, before the start or after the last event
  + GAP_MS are NaN (no extrapolation); 'gap_hist' records the longest inter-event gap in a frame's
  history.

Evidence boundary: raw parquet INPUT columns only (Ip and its time axis, Thomson, TF / bcoil
current via the extracted ds, efit_times / grids as scaffolding); DIII-D labels (EFIT li and the
scorer li(2) of the EFIT psi) only in `probe` / `freeze` (TRAIN cal) and `evaluate` (VAL subset,
after the freeze); TEST never; MAST: never any efit_* value, demo parquet or scorer mask.

    python -B fec/current_diffusion.py d3d --set cal [--limit-shots 30] [--nproc 4]
    python -B fec/current_diffusion.py probe --limit-shots 30         # TRAIN bug / feasibility probe
    python -B fec/current_diffusion.py freeze                          # TRAIN-only choices -> frozen json
    python -B fec/current_diffusion.py d3d --set val
    python -B fec/current_diffusion.py evaluate                        # VAL subset, gate verdict
    python -B fec/current_diffusion.py mast [--limit-shots 20] [--nproc 4]
    python -B fec/current_diffusion.py mastreport
Outputs: <FEC_WORK>/cdiff/ (d3d_<set>.npz, mast.npz, frozen.json, *_report.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import DATA, WORK  # noqa: E402

MU0 = 4e-7 * math.pi
OUT = WORK / "cdiff"

# ----------------------------------------------------------------------------- frozen settings
N_RHO = 64                    # finite-volume cells on rho in [0, 1]
DT_MAX = 2.0                  # ms, longest backward-Euler sub-step
GAP_MS = 25.0                 # ms, no Te extrapolation beyond this distance from a laser event
TE_FLOOR = 10.0               # eV
NE_FLOOR = 1e18               # m^-3
ZEFFS = (1.5, 2.0, 3.0)
ICS = ("flat", "te15")
NOMINAL = (2.0, "flat")       # default variant before any TRAIN selection
IP_START_FRAC = 0.03          # start at the first valid laser event with Ip >= 3 % of max Ip ...
IP_START_MIN = 2e4            # ... and >= 20 kA
NB = 10                       # rho bins of the per-event profile fit
MIN_CH = 6                    # valid channels inside rho <= 1 for a valid laser event
MIN_BINS = 3
RHO_CORE_MAX = 0.5            # the innermost valid channel must reach rho <= this
Q_CLIP = (0.5, 50.0)
NPSI = 21                     # psiN points of the j(psiN) output
PSIN_OUT = np.linspace(0.0, 1.0, NPSI)
SYM_THR_ABS, SYM_THR_REL = 20.0, 0.05     # MAST pre-map Thomson edge threshold (eV, x Te peak)
D3D_TF_TURNS = 144
MAST_TF_TURNS = 24


def variants(zeffs=ZEFFS, ics=ICS):
    """[(Zeff, ic)] in the stored order v = iz * len(ics) + ic."""
    return [(z, c) for z in zeffs for c in ics]


def variant_index(zeff, ic, zeffs=ZEFFS, ics=ICS):
    return list(zeffs).index(zeff) * len(ics) + list(ics).index(ic)


# ============================================================================ resistivity
def trapped_fraction(eps):
    """Effective trapped fraction, circular fit (as fec.d3d_physics.trapped_fraction, numpy)."""
    e = np.clip(eps, 0.0, 0.95)
    return 1.0 - (1.0 - e) ** 2 / (np.sqrt(1.0 - e ** 2) * (1.0 + 1.46 * np.sqrt(e)))


def lnlambda_e(te_ev, ne):
    return 31.3 - np.log(np.sqrt(ne) / te_ev)


def spitzer_eta(te_ev, zeff, lnl):
    """Sauter 1999 Spitzer parallel resistivity [Ohm m]."""
    nz = 0.58 + 0.74 / (0.76 + zeff)
    return zeff * nz * lnl / (1.9012e4 * te_ev ** 1.5)


def f33(x, zeff):
    return 1.0 - (1.0 + 0.36 / zeff) * x + 0.59 / zeff * x ** 2 - 0.23 / zeff * x ** 3


def nustar_e(q, R0, ne, zeff, lnl, te_ev, eps):
    return 6.921e-18 * q * R0 * ne * zeff * lnl / (te_ev ** 2 * np.maximum(eps, 1e-3) ** 1.5)


def eta_neo(te_ev, ne, zeff, eps, q=None, R0=None):
    """Neoclassical parallel resistivity [Ohm m]; q = None -> collisionless (nu* = 0)."""
    te = np.maximum(te_ev, TE_FLOOR)
    n = np.maximum(ne, NE_FLOOR)
    lnl = lnlambda_e(te, n)
    ft = trapped_fraction(eps)
    if q is None:
        x = ft
    else:
        nu = nustar_e(q, R0, n, zeff, lnl, te, eps)
        x = ft / (1.0 + (0.55 - 0.1 * ft) * np.sqrt(nu) + 0.45 * (1.0 - ft) * nu / zeff ** 1.5)
    return spitzer_eta(te, zeff, lnl) / f33(x, zeff)


# ============================================================================ 1-D solver
class Grid:
    def __init__(self, n=N_RHO):
        self.n = n
        self.drho = 1.0 / n
        self.rn = np.linspace(0.0, 1.0, n + 1)            # nodes (I lives here)
        self.rf = (np.arange(n) + 0.5) / n                # faces (eta, j live here)


def f_kappa(kappa):
    return (1.0 + kappa ** 2) / (2.0 * kappa ** 2)


def be_step(g, I, ip_new, eta_f, a, kappa, dt_s):
    """One backward-Euler step.  I (V, n+1) [A], eta_f (V, n) [Ohm m] at faces, a [m], dt_s [s].
    Returns the new I with I[:, 0] = 0 and I[:, n] = ip_new exactly."""
    from scipy.linalg.lapack import dgtsv
    V, n = eta_f.shape
    C = f_kappa(kappa) / (MU0 * a * a)
    D = eta_f / g.rf                                       # (V, n): eta / rho at faces
    r = dt_s * C * g.rn[1:-1] / g.drho ** 2                # (n-1,)
    alpha = r * D[:, :-1]                                  # couples node i to i-1 (face i-1/2)
    beta = r * D[:, 1:]                                    # couples node i to i+1 (face i+1/2)
    d = (1.0 + alpha + beta).ravel()
    z = np.zeros((V, 1))
    dl = np.concatenate([-alpha[:, 1:], z], 1).ravel()[:-1]
    du = np.concatenate([-beta[:, :-1], z], 1).ravel()[:-1]
    rhs = I[:, 1:-1].copy()
    rhs[:, -1] += beta[:, -1] * ip_new
    x = dgtsv(dl, d, du, rhs.reshape(-1, 1))[3]
    out = np.empty_like(I)
    out[:, 0] = 0.0
    out[:, -1] = ip_new
    out[:, 1:-1] = x.reshape(V, n - 1)
    return out


def li_cyl(g, I):
    """2 int (I/Ip)^2 drho/rho (trapezoid; integrand -> 0 at the axis).  I (..., n+1)."""
    ip = I[..., -1:]
    with np.errstate(all="ignore"):
        f = (I[..., 1:] / ip) ** 2 / g.rn[1:]
        f0 = np.zeros(I.shape[:-1] + (1,))
        f = np.concatenate([f0, f], -1)
        return 2.0 * np.trapezoid(f, g.rn, axis=-1)


def j_norm_faces(g, I):
    """j / <j> at faces: (dI/drho) / (2 rho Ip)."""
    return np.diff(I, axis=-1) / g.drho / (2.0 * g.rf * I[..., -1:])


def j_of_psin(g, I, psin_out=PSIN_OUT):
    """(..., NPSI) j / <j> on psiN points of the profile's own flux; NaN if I <= 0 inside."""
    I2 = I.reshape(-1, I.shape[-1])
    out = np.full((I2.shape[0], len(psin_out)), np.nan)
    jf = j_norm_faces(g, I2)
    for k in range(I2.shape[0]):
        Ik = I2[k]
        if not (np.all(np.isfinite(Ik)) and np.all(Ik[1:] > 0)):
            continue
        dpsi = Ik[1:] / g.rn[1:]                        # dpsi/drho ~ I / rho (geometry cancels)
        dp = np.r_[0.0, dpsi]                           # I ~ c rho^2 -> I / rho -> 0 at the axis
        psi = np.concatenate([[0.0], np.cumsum(0.5 * (dp[1:] + dp[:-1]) * g.drho)])
        psin_f = np.interp(g.rf, g.rn, psi / psi[-1])
        xs = np.r_[0.0, psin_f, 1.0]
        # j at the axis: extrapolate the first two faces; at the edge: the last face
        j0 = jf[k, 0] + (jf[k, 0] - jf[k, 1]) * (g.rf[0] / (g.rf[1] - g.rf[0]))
        ys = np.r_[j0, jf[k], jf[k, -1]]
        out[k] = np.interp(psin_out, xs, ys)
    return out.reshape(I.shape[:-1] + (len(psin_out),))


def init_current(g, kind, ip, te_f):
    """Initial I (n+1,) for one IC: 'flat' (j uniform) or 'te15' (j ~ Te^1.5 at faces)."""
    if kind == "flat":
        return ip * g.rn ** 2
    if kind == "te15":
        j = np.maximum(te_f, TE_FLOOR) ** 1.5
        cum = np.concatenate([[0.0], np.cumsum(j * 2.0 * g.rf * g.drho)])
        return ip * cum / cum[-1]
    raise ValueError(kind)


def static_current(g, eta_f, ip=1.0):
    """Relaxed (uniform-E) profile j ~ 1 / eta at faces -> I at nodes (the static 'j ~ sigma'
    reference that ignores the history)."""
    cum = np.concatenate([[0.0], np.cumsum(2.0 * g.rf * g.drho / np.asarray(eta_f, float))])
    return ip * cum / cum[-1]


def q_faces(g, I, a, kappa, R0, B0):
    """Cylindrical (elongation-corrected) q at faces from I at nodes (V, n+1)."""
    If = 0.5 * (I[:, 1:] + I[:, :-1])
    r = g.rf * a
    with np.errstate(all="ignore"):
        q = 2.0 * np.pi * r ** 2 * B0 * 0.5 * (1.0 + kappa ** 2) / (MU0 * R0 * If)
    q = np.where(np.isfinite(q) & (q > 0), q, Q_CLIP[1])
    return np.clip(q, *Q_CLIP)


# ============================================================================ Thomson profiles
def pav_decreasing(y, w):
    """Weighted least-squares non-increasing fit (pool adjacent violators)."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(np.asarray(y, float), np.asarray(w, float)):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            wsum = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wsum
            c = cnt[-2] + cnt[-1]
            vals[-2:] = [v]; wts[-2:] = [wsum]; cnt[-2:] = [c]
    return np.repeat(vals, cnt)


def fit_profile(rho, val, rho_out, monotone=True, floor=TE_FLOOR, edge="linear"):
    """Profile on rho_out from channel (rho, value) pairs, or None if the event is not valid."""
    ok = np.isfinite(rho) & np.isfinite(val) & (val > 0) & (rho <= 1.0)
    if ok.sum() < MIN_CH:
        return None
    r, v = rho[ok], val[ok]
    if r.min() > RHO_CORE_MAX:
        return None
    b = np.minimum((r * NB).astype(int), NB - 1)
    xs, ys, ws = [], [], []
    for k in range(NB):
        m = b == k
        if m.any():
            xs.append(float(np.median(r[m]))); ys.append(float(np.median(v[m]))); ws.append(m.sum())
    if len(xs) < MIN_BINS:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)
    if monotone:
        ys = pav_decreasing(ys, ws)
    if xs[-1] < 1.0:
        if edge == "linear" and len(xs) >= 2 and xs[-1] > xs[-2]:
            y1 = ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (1.0 - xs[-1])
            y1 = float(np.clip(y1, floor, max(ys[-1], floor)))
        else:
            y1 = ys[-1]
        xs, ys = np.r_[xs, 1.0], np.r_[ys, y1]
    return np.maximum(np.interp(rho_out, xs, ys), floor)


def ts_symmetric(R, te, thr_abs=SYM_THR_ABS, thr_rel=SYM_THR_REL):
    """Midplane radial Thomson array -> (R_c, R_in, R_out) from the profile itself, or None.
    R_in / R_out: threshold crossings of the smoothed Te(R); R_c makes the two halves most alike
    as functions of the normalised distance from R_c."""
    from scipy.ndimage import median_filter
    ok = np.isfinite(R) & np.isfinite(te) & (te > 0)
    if ok.sum() < 10:
        return None
    o = np.argsort(R[ok])
    r, t = R[ok][o], median_filter(te[ok][o], size=5, mode="nearest")
    p = int(np.argmax(t))
    thr = max(thr_abs, thr_rel * t[p])
    if t[p] <= thr:
        return None
    i = p
    while i > 0 and t[i - 1] >= thr:
        i -= 1
    R_in = r[i] if i == 0 else r[i - 1] + (thr - t[i - 1]) / (t[i] - t[i - 1]) * (r[i] - r[i - 1])
    j = p
    while j < len(r) - 1 and t[j + 1] >= thr:
        j += 1
    R_out = r[j] if j == len(r) - 1 else r[j] + (t[j] - thr) / (t[j] - t[j + 1]) * (r[j + 1] - r[j])
    w = R_out - R_in
    if w < 0.3:
        return None
    # centre: the inboard and outboard halves match best in normalised distance (x = 0 at R_c,
    # 1 at the threshold crossing on each side; Shafranov-compressed outboard side allowed)
    x = np.linspace(0.0, 0.95, 20)
    best, Rc = np.inf, 0.5 * (R_in + R_out)
    for c in np.linspace(R_in + 0.25 * w, R_out - 0.25 * w, 81):
        m = np.mean((np.interp(c - x * (c - R_in), r, t) - np.interp(c + x * (R_out - c), r, t)) ** 2)
        if m < best:
            best, Rc = m, c
    if not (0.3 <= Rc <= 1.3):
        return None
    return float(Rc), float(R_in), float(R_out)


def rho_symmetric(R, sym):
    Rc, Rin, Rout = sym
    return np.where(R < Rc, (Rc - R) / (Rc - Rin), (R - Rc) / (Rout - Rc))


# ============================================================================ map -> channel rho
def bilinear(F, gR, gZ, Rq, Zq):
    fr = (np.asarray(Rq, float) - gR[0]) / (gR[1] - gR[0])
    fz = (np.asarray(Zq, float) - gZ[0]) / (gZ[1] - gZ[0])
    ok = np.isfinite(fr) & np.isfinite(fz) & (fr >= 0) & (fz >= 0) & (fr <= len(gR) - 1) & \
        (fz <= len(gZ) - 1)
    i = np.clip(np.floor(np.where(ok, fr, 0)).astype(int), 0, len(gR) - 2)
    k = np.clip(np.floor(np.where(ok, fz, 0)).astype(int), 0, len(gZ) - 2)
    u, v = np.where(ok, fr, 0) - i, np.where(ok, fz, 0) - k
    val = (F[k, i] * (1 - u) * (1 - v) + F[k, i + 1] * u * (1 - v) + F[k + 1, i] * (1 - u) * v +
           F[k + 1, i + 1] * u * v)
    return np.where(ok, val, np.nan)


def map_channel_rho(psi, psi_ax, psi_b, Rax, Zax, gR, gZ, Rch, Zch):
    """Area-equivalent rho of chord points on one flux map (any psi sign convention: psiN =
    (psi_ax - psi) / (psi_ax - psi_b)).  Returns (rho (n_ch,), geo dict) or (NaNs, None)."""
    from scipy.ndimage import label
    nan = np.full(len(Rch), np.nan)
    den = psi_ax - psi_b
    if not (np.isfinite(den) and den != 0 and np.isfinite(Rax) and np.isfinite(Zax)):
        return nan, None
    pn = (psi_ax - np.asarray(psi, float)) / den
    ir = int(np.argmin(np.abs(gR - Rax)))
    iz = int(np.argmin(np.abs(gZ - Zax)))
    if not pn[iz, ir] < 1:
        return nan, None
    lab, _ = label(pn < 1)                                  # 4-connectivity
    core = lab == lab[iz, ir]
    col = pn[:, ir] < 1
    top = iz
    while top + 1 < len(gZ) and col[top + 1]:
        top += 1
    bot = iz
    while bot - 1 >= 0 and col[bot - 1]:
        bot -= 1
    core[:bot] = False
    core[top + 1:] = False
    vals = np.sort(np.clip(pn[core], 0.0, 1.0))
    n = len(vals)
    if n < 20:
        return nan, None
    xs = np.r_[0.0, vals, 1.0]
    ys = np.r_[0.0, (np.arange(n) + 0.5) / n, 1.0]
    pch = bilinear(pn, gR, gZ, Rch, Zch)
    inside = pch < 1
    rho = np.where(inside, np.sqrt(np.interp(np.clip(pch, 0, 1), xs, ys)),
                   np.sqrt(np.maximum(pch, 1.0)))
    RR = np.broadcast_to(gR[None, :], pn.shape)[core]
    ZZ = np.broadcast_to(gZ[:, None], pn.shape)[core]
    dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
    a = 0.5 * (RR.max() - RR.min() + dR)
    geo = dict(a=a, kappa=0.5 * (ZZ.max() - ZZ.min() + dZ) / a, R0=0.5 * (RR.max() + RR.min()),
               area=n * dR * dZ)
    return np.where(np.isfinite(pch), rho, np.nan), geo


# ============================================================================ per-shot driver
def run_shot(sh, zeffs=ZEFFS, ics=ICS, n_rho=N_RHO, dt_max=DT_MAX, collisional=True, saw=1.0):
    """Diffuse one shot.  sh: dict with
      ip_t (ms), ip (A, canonical > 0 at flat-top); ev_t (n_ev,) ms; ch_R, ch_Z (n_ch,);
      te (n_ev, n_ch) eV, ne (n_ev, n_ch) m^-3 (NaN = missing); out_t (n_out,) ms (EFIT times);
      map_t (n_map,) ms; map_rho (n_map, n_ch); map_geo (n_map, 3) = a, kappa, R0;
      pre_map 'hold' | 'symmetric'; F (T m, R B_tor) or NaN.
    saw > 1: eta x saw on faces with q < 1 (sawtooth-mimicking flattening; 1 = off).
    Returns a dict of per-output arrays (li (n_out, V), jpsi (n_out, V, NPSI), ok, ...)."""
    g = Grid(n_rho)
    var = [(z, c) for z in zeffs for c in ics]
    V = len(var)
    zv = np.array([z for z, _ in var])[:, None]
    out_t = np.asarray(sh["out_t"], float)
    n_out = len(out_t)
    res = dict(li=np.full((n_out, V), np.nan), jpsi=np.full((n_out, V, NPSI), np.nan, np.float32),
               li_static=np.full(n_out, np.nan), ok=np.zeros(n_out, bool),
               t_rel=np.full(n_out, np.nan), dt_ev=np.full(n_out, np.nan),
               gap_hist=np.full(n_out, np.nan), a=np.full(n_out, np.nan),
               kappa=np.full(n_out, np.nan), te_core=np.full(n_out, np.nan),
               ip=np.full(n_out, np.nan), q0=np.full((n_out, V), np.nan))
    info = dict(n_ev=int(len(sh["ev_t"])), n_valid=0, t_start=np.nan, t_last=np.nan,
                n_sym=0, n_steps=0, status="ok")
    ev_t = np.asarray(sh["ev_t"], float)
    map_t = np.asarray(sh["map_t"], float)
    map_rho = np.asarray(sh["map_rho"], float)
    map_geo = np.asarray(sh["map_geo"], float)
    ch_R = np.asarray(sh["ch_R"], float)
    if len(ev_t) == 0 or len(map_t) == 0:
        info["status"] = "no_events" if len(ev_t) == 0 else "no_maps"
        return res, info
    # ---- channel rho at each event (time-linear between map frames, held outside)
    good_map = np.isfinite(map_geo).all(1)
    if not good_map.any():
        info["status"] = "no_good_maps"
        return res, info
    mt, mr, mg = map_t[good_map], map_rho[good_map], map_geo[good_map]
    rho_ev = np.full((len(ev_t), len(ch_R)), np.nan)
    for c in range(len(ch_R)):
        f = np.isfinite(mr[:, c])
        if f.any():
            rho_ev[:, c] = np.interp(ev_t, mt[f], mr[f, c])
    geo_ev = np.stack([np.interp(ev_t, mt, mg[:, k]) for k in range(3)], 1)
    if sh.get("pre_map") == "symmetric":
        for e in np.flatnonzero(ev_t < mt[0]):
            s = ts_symmetric(ch_R, sh["te"][e])
            if s is None:
                continue
            rho_ev[e] = rho_symmetric(ch_R, s)
            geo_ev[e] = (0.5 * (s[2] - s[1]), mg[0, 1], s[0])
            info["n_sym"] += 1
    # ---- profiles per event (Te at faces, n_e at faces)
    te_f = np.full((len(ev_t), n_rho), np.nan)
    ne_f = np.full((len(ev_t), n_rho), np.nan)
    for e in range(len(ev_t)):
        p = fit_profile(rho_ev[e], sh["te"][e], g.rf)
        if p is None:
            continue
        q = fit_profile(rho_ev[e], sh["ne"][e], g.rf, monotone=False, floor=NE_FLOOR, edge="hold")
        te_f[e] = p
        ne_f[e] = q if q is not None else 3e19
    valid = np.isfinite(te_f[:, 0])
    ip_t, ip = np.asarray(sh["ip_t"], float), np.asarray(sh["ip"], float)
    ipmax = float(np.nanmax(ip)) if len(ip) else np.nan
    ip_ev = np.interp(ev_t, ip_t, ip)
    start_ok = valid & (ip_ev >= max(IP_START_MIN, IP_START_FRAC * ipmax))
    info["n_valid"] = int(valid.sum())
    if not start_ok.any():
        info["status"] = "no_start"
        return res, info
    e0 = int(np.flatnonzero(start_ok)[0])
    ve = np.flatnonzero(valid & (np.arange(len(ev_t)) >= e0))
    tv, Tv, Nv, Gv = ev_t[ve], te_f[ve], ne_f[ve], geo_ev[ve]
    t0, t_last = tv[0], tv[-1]
    t_end = t_last + GAP_MS
    info.update(t_start=float(t0), t_last=float(t_last))
    # ---- step grid: valid events + output times, sub-steps <= dt_max
    want = (out_t >= t0) & (out_t <= t_end)
    knots = np.unique(np.r_[tv, out_t[want]])
    pts = [knots[:1]]
    for a_, b_ in zip(knots[:-1], knots[1:]):
        m = max(1, int(math.ceil((b_ - a_) / dt_max - 1e-9)))
        pts.append(np.r_[a_ + (b_ - a_) * np.arange(1, m) / m, b_])   # knots kept exact
    tg = np.concatenate(pts)
    # Te / n_e / geometry / Ip on the grid (time-linear between valid events, held after the last)
    k = np.clip(np.searchsorted(tv, tg, "right") - 1, 0, len(tv) - 1)
    k1 = np.minimum(k + 1, len(tv) - 1)
    with np.errstate(all="ignore"):
        w = np.where(k1 > k, (tg - tv[k]) / (tv[k1] - tv[k]), 0.0).clip(0, 1)[:, None]
    Tg = (1 - w) * Tv[k] + w * Tv[k1]
    Ng = (1 - w) * Nv[k] + w * Nv[k1]
    Gg = (1 - w) * Gv[k] + w * Gv[k1]
    Ig = np.interp(tg, ip_t, ip)
    F = float(sh.get("F", np.nan))
    use_q = collisional and np.isfinite(F) and F > 0
    # ---- initial conditions
    I = np.stack([init_current(g, c, Ig[0], Tg[0]) for _, c in var])
    out_idx = {}
    for i_o in np.flatnonzero(want):
        j = int(np.searchsorted(tg, out_t[i_o]))
        if j < len(tg) and abs(tg[j] - out_t[i_o]) < 1e-9:
            out_idx.setdefault(j, []).append(i_o)

    def record(j, Icur):
        for i_o in out_idx.get(j, ()):
            res["li"][i_o] = li_cyl(g, Icur)
            res["jpsi"][i_o] = j_of_psin(g, Icur)
            a_, kap_, R0_ = Gg[j]
            eps = g.rf * a_ / R0_
            res["li_static"][i_o] = li_cyl(g, static_current(g, eta_neo(Tg[j], Ng[j], 2.0, eps)))
            res["a"][i_o], res["kappa"][i_o] = a_, kap_
            res["te_core"][i_o] = Tg[j][0]
            res["ip"][i_o] = Ig[j]
            if np.isfinite(F) and F > 0:
                res["q0"][i_o] = q_faces(g, Icur, a_, kap_, R0_, F / R0_)[:, 0]
    record(0, I)
    for j in range(1, len(tg)):
        a_, kap_, R0_ = Gg[j]
        eps = g.rf * a_ / R0_
        q = q_faces(g, I, a_, kap_, R0_, F / R0_) if (use_q or saw > 1) else None
        if use_q:
            eta = eta_neo(Tg[j][None, :], Ng[j][None, :], zv, eps[None, :], q, R0_)
        else:
            eta = eta_neo(Tg[j][None, :], Ng[j][None, :], zv, eps[None, :])
        if saw > 1:
            eta = np.where(q < 1.0, eta * saw, eta)
        I = be_step(g, I, Ig[j], eta, a_, kap_, (tg[j] - tg[j - 1]) * 1e-3)
        record(j, I)
    info["n_steps"] = int(len(tg) - 1)
    # ---- validity: within GAP_MS of a valid event; history gaps
    gaps = np.diff(tv)
    for i_o in range(n_out):
        t = out_t[i_o]
        if not want[i_o]:
            continue
        d = float(np.min(np.abs(tv - t)))
        res["dt_ev"][i_o] = d
        res["t_rel"][i_o] = t - t0
        upto = np.searchsorted(tv, t, "right")
        res["gap_hist"][i_o] = float(gaps[:max(upto, 1)].max()) if len(gaps) else 0.0
        res["ok"][i_o] = d <= GAP_MS and np.isfinite(res["li"][i_o]).all()
    bad = ~res["ok"]
    res["q0"][bad] = np.nan
    res["li"][bad] = np.nan
    res["jpsi"][bad] = np.nan
    res["li_static"][bad] = np.nan
    return res, info


# ============================================================================ readers (inputs only)
def _col(t, name):
    from fec.extract import col1d
    return col1d(t, name)


def _col2(t, name):
    from fec.extract import col2d
    return col2d(t, name)


def _clean(a):
    a = np.asarray(a, float)
    return np.where(np.isfinite(a) & (a > 0), a, np.nan)


def _smooth_ip(t, ip, width_ms=2.0):
    o = np.argsort(t, kind="stable")
    t, ip = t[o], ip[o]
    ok = np.isfinite(t) & np.isfinite(ip)
    t, ip = t[ok], ip[ok]
    if len(t) > 3:
        dt = float(np.median(np.diff(t)))
        k = max(1, int(round(width_ms / max(dt, 1e-6))))
        if k > 1:
            ip = np.convolve(ip, np.ones(k) / k, mode="same")
    return t, ip


def _nearest_rows(t_from, t_to, tol):
    """Index into t_from for each t_to within tol (else -1)."""
    if len(t_from) == 0:
        return np.full(len(t_to), -1)
    o = np.argsort(t_from)
    s = t_from[o]
    j = np.clip(np.searchsorted(s, t_to), 1, max(len(s) - 1, 1))
    jl = np.clip(j - 1, 0, len(s) - 1)
    jr = np.clip(j, 0, len(s) - 1)
    pick = np.where(np.abs(s[jl] - t_to) <= np.abs(s[jr] - t_to), jl, jr)
    ok = np.abs(s[pick] - t_to) <= tol
    return np.where(ok, o[pick], -1)


def read_d3d(path):
    """DIII-D inputs: Ip on the idx time axis (fec.ip_timebase.idx_ip_times), Thomson core
    (vertical chord) + tangential chords (tangential Te attached to core laser times within 2 ms)."""
    import pyarrow.parquet as pq
    from fec.ip_timebase import idx_ip_times
    cols = ["magnetics_time", "magnetics_plasma_current", "magnetics_plasma_current_times",
            "thomson_core_times", "thomson_core_Te", "thomson_core_ne", "thomson_edge_times",
            "thomson_edge_Te", "thomson_edge_ne", "thomson_chord_name", "thomson_chord_R",
            "thomson_chord_Z", "efit_times"]
    t = pq.read_table(path, columns=cols)
    mt0 = float(t.column("magnetics_time")[0].as_py()[0])
    ipt = idx_ip_times(_col(t, "magnetics_plasma_current_times"), mt0)
    ip = _col(t, "magnetics_plasma_current") * 1e3
    ip_t, ip = _smooth_ip(ipt, ip)
    s = np.sign(ip[np.nanargmax(np.abs(ip))]) or 1.0
    names = t.column("thomson_chord_name")[0].as_py()
    cR, cZ = _col(t, "thomson_chord_R"), _col(t, "thomson_chord_Z")
    ci = [i for i, n in enumerate(names) if n.startswith("TS_core")]
    ti = [i for i, n in enumerate(names) if n.startswith("TS_tangential")]
    cte, cne = _col2(t, "thomson_core_Te"), _col2(t, "thomson_core_ne")
    ete, ene = _col2(t, "thomson_edge_Te"), _col2(t, "thomson_edge_ne")
    ct, et = _col(t, "thomson_core_times"), _col(t, "thomson_edge_times")
    if len(ci) != cte.shape[1] or len(ti) != ete.shape[1]:
        return None
    rows = _nearest_rows(et, ct, 2.0)
    tte = np.full((len(ct), len(ti)), np.nan)
    tne = np.full((len(ct), len(ti)), np.nan)
    m = rows >= 0
    if len(ti):
        tte[m], tne[m] = ete[rows[m]], ene[rows[m]]
    return dict(ip_t=ip_t, ip=ip * s, ev_t=ct, ch_R=np.r_[cR[ci], cR[ti]], ch_Z=np.r_[cZ[ci], cZ[ti]],
                te=_clean(np.c_[cte, tte]), ne=_clean(np.c_[cne, tne]),
                efit_t=_col(t, "efit_times"))


def read_mast(path):
    """MAST inputs: Ip on magnetics_time, Thomson core (first column dropped as in fec.extract) +
    edge (attached to core laser times within 1 ms), midplane Z = 0, grid (scaffolding)."""
    import pyarrow.parquet as pq
    cols = ["magnetics_time", "magnetics_plasma_current", "thomson_core_times", "thomson_core_Te",
            "thomson_core_ne", "thomson_core_R", "thomson_edge_times", "thomson_edge_Te",
            "thomson_edge_ne", "thomson_edge_spatial", "efit_times", "efit_grid_R", "efit_grid_Z"]
    t = pq.read_table(path, columns=cols)
    ip_t, ip = _smooth_ip(_col(t, "magnetics_time"), _col(t, "magnetics_plasma_current") * 1e3)
    s = np.sign(ip[np.nanargmax(np.abs(ip))]) or 1.0
    coreR = _col(t, "thomson_core_R")
    cte, cne = _col2(t, "thomson_core_Te"), _col2(t, "thomson_core_ne")
    if cte.shape[1] == coreR.size + 1:
        cte, cne = cte[:, 1:], cne[:, 1:]
    edgeR = _col(t, "thomson_edge_spatial")
    ete, ene = _col2(t, "thomson_edge_Te"), _col2(t, "thomson_edge_ne")
    ct, et = _col(t, "thomson_core_times"), _col(t, "thomson_edge_times")
    if cte.shape[1] != coreR.size:
        return None
    rows = _nearest_rows(et, ct, 1.0)
    tte = np.full((len(ct), edgeR.size), np.nan)
    tne = np.full((len(ct), edgeR.size), np.nan)
    m = rows >= 0
    if edgeR.size and ete.shape[1] == edgeR.size:
        tte[m], tne[m] = ete[rows[m]], ene[rows[m]]
    chR = np.r_[coreR, edgeR]
    return dict(ip_t=ip_t, ip=ip * s, ev_t=ct, ch_R=chR, ch_Z=np.zeros_like(chR),
                te=_clean(np.c_[cte, tte]), ne=_clean(np.c_[cne, tne]),
                efit_t=_col(t, "efit_times"), gR=_col(t, "efit_grid_R"), gZ=_col(t, "efit_grid_Z"))


# ============================================================================ jobs
def _maps_to_rho(sh, psis, pa, pb, ra, za, gR, gZ, geo_rec):
    n = len(pa)
    rho = np.full((n, len(sh["ch_R"])), np.nan)
    geo = np.full((n, 3), np.nan)
    for i in range(n):
        r, gm = map_channel_rho(psis[i], pa[i], pb[i], ra[i], za[i], gR, gZ, sh["ch_R"], sh["ch_Z"])
        rho[i] = r
        if gm is None:
            continue
        a, k, R0 = geo_rec[i] if geo_rec is not None else (np.nan,) * 3
        geo[i] = (a if np.isfinite(a) and a > 0.05 else gm["a"],
                  k if np.isfinite(k) and k > 0.5 else gm["kappa"],
                  R0 if np.isfinite(R0) and R0 > 0.1 else gm["R0"])
    return rho, geo


def _d3d_job(job):
    stem, path, out_t, psis, pa, pb, ra, za, geo_rec, gR, gZ, F, kw = job
    try:
        sh = read_d3d(path)
        if sh is None:
            return stem, None, dict(status="layout")
        rho, geo = _maps_to_rho(sh, psis, pa, pb, ra, za, gR, gZ, geo_rec)
        sh.update(out_t=out_t, map_t=out_t, map_rho=rho, map_geo=geo, pre_map="hold", F=F)
        res, info = run_shot(sh, **kw)
        return stem, res, info
    except Exception as e:  # noqa: BLE001
        return stem, None, dict(status=f"error {type(e).__name__}: {e}")


def _mast_job(job):
    stem, path, rows, psi_path, pa, pb, ra, za, geo_rec, out_t, F, kw = job
    try:
        sh = read_mast(path)
        if sh is None:
            return stem, None, dict(status="layout")
        P = np.load(psi_path, mmap_mode="r")
        psis = np.asarray(P[rows[0]:rows[-1] + 1], np.float64)[np.asarray(rows) - rows[0]]
        rho, geo = _maps_to_rho(sh, psis, pa, pb, ra, za, sh["gR"], sh["gZ"], geo_rec)
        sh.update(out_t=out_t, map_t=out_t, map_rho=rho, map_geo=geo, pre_map="symmetric", F=F)
        res, info = run_shot(sh, **kw)
        return stem, res, info
    except Exception as e:  # noqa: BLE001
        return stem, None, dict(status=f"error {type(e).__name__}: {e}")


def _collect(n, V, results, slices):
    arr = dict(li=np.full((n, V), np.nan), jpsi=np.full((n, V, NPSI), np.nan, np.float32),
               li_static=np.full(n, np.nan), ok=np.zeros(n, bool), t_rel=np.full(n, np.nan),
               dt_ev=np.full(n, np.nan), gap_hist=np.full(n, np.nan), a=np.full(n, np.nan),
               kappa=np.full(n, np.nan), te_core=np.full(n, np.nan), ip=np.full(n, np.nan),
               q0=np.full((n, V), np.nan))
    infos = {}
    for stem, res, info in results:
        infos[stem] = info
        if res is None:
            continue
        sl = slices[stem]
        for k in arr:
            arr[k][sl] = res[k]
    return arr, infos


def run_d3d(set_name="cal", limit_shots=0, nproc=4, out=None, tag="", **kw):
    """Diffusion on a frozen fec.d3d_physics subset (cal = TRAIN, val = VAL subset); outputs on
    the subset frames (aligned with in_<set>.npz idx).  Maps: fec.betap_li KB_A1_ffz (our own)."""
    from multiprocessing import Pool
    from fec import d3d_physics as dp
    from fec import betap_li as BL
    inp = dp.load_set(set_name)
    z = np.load(BL.OUT / f"pred_{set_name}" / f"{BL.BASE}.npz")
    assert np.array_equal(z["idx"], inp["idx"])
    rec = np.load(BL.OUT / f"records_{set_name}.npz")
    assert np.array_equal(rec["idx"], inp["idx"])
    info = json.loads((WORK / "ds" / dp.CONFIG / "info.json").read_text())
    sls = dp._shot_slices(inp["shot"])
    if limit_shots:
        sls = sls[:limit_shots]
    n = sls[-1].stop
    psi_all = z["psi"]
    jobs, slices = [], {}
    for sl in sls:
        stem = info["order"][int(inp["shot"][sl.start])]
        slices[stem] = sl
        bt = float(np.nanmedian(inp["bt"][sl]))
        F = MU0 * D3D_TF_TURNS * bt * 1e3 / (2 * np.pi)
        geo_rec = np.stack([rec["a"][sl], rec["kappa"][sl], rec["Rgeo"][sl]], 1)
        jobs.append((stem, str(DATA / dp.CONFIG / f"{stem}.parquet"), np.asarray(inp["t_ms"][sl], float),
                     np.asarray(psi_all[sl], np.float64), z["psi_ax"][sl], z["psi_b"][sl], z["Rax"][sl],
                     z["Zax"][sl], geo_rec, inp["gR"], inp["gZ"], F, kw))
    t0 = time.time()
    results = []
    with Pool(max(1, min(nproc, 6))) as pool:
        for i, r in enumerate(pool.imap_unordered(_d3d_job, jobs, chunksize=1)):
            results.append(r)
            if i % 25 == 0:
                print(f"d3d {set_name} {i + 1}/{len(jobs)} {time.time() - t0:.0f}s {r[0]} "
                      f"{r[2].get('status')}", flush=True)
    V = len(variants(kw.get("zeffs", ZEFFS), kw.get("ics", ICS)))
    arr, infos = _collect(n, V, results, slices)
    arr.update(idx=inp["idx"][:n], shot=inp["shot"][:n], t_ms=inp["t_ms"][:n])
    OUT.mkdir(parents=True, exist_ok=True)
    f = out or OUT / (f"d3d_{set_name}" + (f"_lim{limit_shots}" if limit_shots else "") + tag + ".npz")
    np.savez(f, **arr)
    Path(str(f).replace(".npz", "_info.json")).write_text(json.dumps(infos, indent=0, default=str))
    st = [v.get("status") for v in infos.values()]
    print(f"d3d {set_name}: {n} frames, ok {arr['ok'].mean():.3f}, status "
          f"{ {s: st.count(s) for s in set(st)} }, {time.time() - t0:.0f}s -> {f}", flush=True)
    return arr, infos


def run_mast(limit_shots=0, nproc=4, out=None, maps="cand_zsc_z0", **kw):
    """Diffusion on every mast_public_test shot (inputs only; our own label-free maps)."""
    from multiprocessing import Pool
    from fec.features import load_ds
    ds = load_ds("mast_public_test")
    info = ds["info"]
    shot = np.asarray(ds["shot"])
    t_ms = np.asarray(ds["t_ms"], float)
    mdir = WORK / "mastz" / maps
    meta = np.load(mdir / "meta.npz")
    br = np.load(mdir / "betan_records.npz")
    assert np.array_equal(br["gidx"], np.arange(len(shot)))
    ch = info["channels"]
    tf = np.abs(np.asarray(ds["cur"][:, ch.index("magnetics_tf_current"), 6], float))
    psi_path = str(mdir / "psi.npy")
    assert np.load(psi_path, mmap_mode="r").shape[0] == len(shot)
    sids = np.unique(shot)
    if limit_shots:
        sids = sids[:limit_shots]
    jobs, slices = [], {}
    for s in sids:
        rows = np.flatnonzero(shot == s)
        assert np.all(np.diff(rows) == 1)
        stem = info["order"][int(s)]
        slices[stem] = slice(int(rows[0]), int(rows[-1]) + 1)
        F = MU0 * MAST_TF_TURNS * float(np.nanmedian(tf[rows])) * 1e3 / (2 * np.pi)
        geo_rec = np.stack([br["a"][rows], br["kappa"][rows], br["Rgeo"][rows]], 1)
        jobs.append((stem, str(DATA / "mast_public_test" / f"{stem}.parquet"), rows.tolist(), psi_path,
                     meta["psi_ax"][rows], meta["psi_b"][rows], meta["Rax"][rows], meta["Zax"][rows],
                     geo_rec, t_ms[rows], F, kw))
    n = len(shot)
    t0 = time.time()
    results = []
    with Pool(max(1, min(nproc, 6))) as pool:
        for i, r in enumerate(pool.imap_unordered(_mast_job, jobs, chunksize=1)):
            results.append(r)
            if i % 50 == 0:
                print(f"mast {i + 1}/{len(jobs)} {time.time() - t0:.0f}s {r[0]} {r[2].get('status')}",
                      flush=True)
    V = len(variants(kw.get("zeffs", ZEFFS), kw.get("ics", ICS)))
    arr, infos = _collect(n, V, results, slices)
    covered = np.zeros(n, bool)
    for sl in slices.values():
        covered[sl] = True
    arr.update(idx=np.arange(n), shot=shot, t_ms=t_ms, covered=covered)
    OUT.mkdir(parents=True, exist_ok=True)
    f = out or OUT / ("mast" + (f"_lim{limit_shots}" if limit_shots else "") + ".npz")
    np.savez(f, **arr)
    Path(str(f).replace(".npz", "_info.json")).write_text(json.dumps(infos, indent=0, default=str))
    st = [v.get("status") for v in infos.values()]
    print(f"mast: {int(covered.sum())} frames of {len(sids)} shots, ok {arr['ok'][covered].mean():.3f}, "
          f"status { {s: st.count(s) for s in set(st)} }, {time.time() - t0:.0f}s -> {f}", flush=True)
    return arr, infos


# ============================================================================ analysis helpers
def corner_spread(li, zeffs=ZEFFS, ics=ICS, z_pair=(1.5, 3.0)):
    """Per-frame max - min of li over the corners {z_pair} x {ICs} (the frozen falsifier set)."""
    cols = [variant_index(z, c, zeffs, ics) for z in z_pair for c in ics]
    sub = li[:, cols]
    return np.nanmax(sub, 1) - np.nanmin(sub, 1)


def sensitivity(li, nominal_v, mask=None, zeffs=ZEFFS, ics=ICS):
    """RMS / median / p90 of the corner spread vs the across-frame SD of the nominal li."""
    m = np.isfinite(li).all(1) if mask is None else (mask & np.isfinite(li).all(1))
    if m.sum() < 3:
        return dict(n=int(m.sum()))
    sp = corner_spread(li[m], zeffs, ics)
    iz = {c: [variant_index(z, c, zeffs, ics) for z in (1.5, 3.0)] for c in ics}
    zsp = np.max([np.abs(li[m][:, a] - li[m][:, b]) for a, b in iz.values()], 0)
    icp = np.max([np.abs(li[m][:, variant_index(z, ics[0], zeffs, ics)] -
                         li[m][:, variant_index(z, ics[1], zeffs, ics)]) for z in (1.5, 3.0)], 0)
    sd = float(np.std(li[m][:, nominal_v]))
    rms = float(np.sqrt(np.mean(sp ** 2)))
    return dict(n=int(m.sum()), sd_nominal=sd, spread_rms=rms, spread_med=float(np.median(sp)),
                spread_p90=float(np.percentile(sp, 90)), zeff_rms=float(np.sqrt(np.mean(zsp ** 2))),
                ic_rms=float(np.sqrt(np.mean(icp ** 2))), ratio=rms / sd if sd > 0 else np.inf,
                passes=bool(rms < sd))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _labels(set_name):
    """EFIT li label (bpli bar target) and the scorer li(2) of the EFIT psi (refs cache)."""
    from fec import betap_li as BL
    t = np.load(BL.OUT / f"targets_{set_name}.npz")
    dp = BL.bind_dirs(set_name)
    gt = dp.gt_refs(set_name, nproc=4)
    return dict(idx=t["idx"], li_E=t["li_E"], li_scorer=gt["cons"][:, 6],
                li_scorer_ok=gt["cmask"][:, 6])


def _phase(set_name, n):
    from fec import d3d_physics as dp
    inp = dp.load_set(set_name)
    return dp.phase_of({k: v[:n] for k, v in inp.items() if isinstance(v, np.ndarray) and v.ndim and
                        len(v) == len(inp["Ip"])})


def _r2(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 3:
        return float("nan")
    y, p = y[ok], p[ok]
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def cmd_probe(limit_shots=30, nproc=4, tag="", **kw):
    """TRAIN probe (first cal shots): bug check + feasibility (in-sample a + b li_cyl R2, corner
    spread by phase).  Uses TRAIN labels only."""
    f = OUT / f"d3d_cal_lim{limit_shots}{tag}.npz"
    if not f.exists():
        run_d3d("cal", limit_shots=limit_shots, nproc=nproc, tag=tag, **kw)
    z = dict(np.load(f))
    n = len(z["idx"])
    from fec import betap_li as BL
    t = np.load(BL.OUT / "targets_cal.npz")
    assert np.array_equal(t["idx"][:n], z["idx"])
    y = BL._target({k: t[k][:n] for k in ("li_E",)}, "li")
    ph = _phase("cal", n)
    V = z["li"].shape[1]
    rep = dict(n=n, ok=float(z["ok"].mean()), variants=variants())
    rep["r2_insample"] = {}
    for v in range(V):
        x = z["li"][:, v]
        m = np.isfinite(x) & np.isfinite(y)
        A = np.c_[np.ones(m.sum()), x[m]]
        c = np.linalg.lstsq(A, y[m], rcond=None)[0]
        rep["r2_insample"][str(variants()[v])] = dict(
            r2=_r2(y[m], A @ c), corr=float(np.corrcoef(x[m], y[m])[0, 1]), coef=c.tolist(),
            by_phase={p: _r2(y[m][ph[m] == p], (A @ c)[ph[m] == p]) for p in ("ramp_up", "flat", "ramp_down")})
    xs = z["li_static"]
    m = np.isfinite(xs) & np.isfinite(y)
    rep["li_static_corr"] = float(np.corrcoef(xs[m], y[m])[0, 1]) if m.sum() > 3 else None
    nv = variant_index(*NOMINAL)
    rep["sensitivity"] = {p: sensitivity(z["li"], nv, ph == p) for p in ("ramp_up", "flat", "ramp_down")}
    rep["sensitivity"]["all"] = sensitivity(z["li"], nv)
    rep["li_pct"] = np.nanpercentile(z["li"][:, nv], [1, 50, 99]).tolist()
    rep["label_pct"] = np.nanpercentile(y, [1, 50, 99]).tolist()
    rep["q0_pct"] = np.nanpercentile(z["q0"][:, nv], [5, 25, 50, 75, 95]).tolist()
    rep["kw"] = kw
    (OUT / f"probe_cal_lim{limit_shots}{tag}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str)[:6000])
    return rep


# ============================================================================ freeze / evaluate
FREEZE_RULE = (
    "TRAIN cal only (400 shots, 11,211 frames; EFIT li label li_E in (0.3, 3) as in fec.betap_li). "
    "Nominal variant = the (Zeff, IC) with the highest 5-fold shot-grouped TRAIN-CV R2 (fec.betap_li "
    "cv_oof, seed 0) of the single-feature robust linear model li ~ a + b li_cyl (fec.betap_li fit_reg); "
    "ties within 0.002 -> NOMINAL (2.0, flat). Combined model = the frozen bpli li features "
    "(selection.json: tnorm, fgw, bpe, peak_te) + li_cd (nominal li_cyl), same fit_reg / folds. "
    "Gate (plan): pass if [(a) VAL R2(li alone) >= 0.517 OR (b) TRAIN-CV R2(combined) >= 0.605 AND "
    "VAL R2(combined) >= 0.547] AND VAL corner spread RMS over {Zeff 1.5, 3} x {flat, te15} < SD over "
    "frames of the nominal li_cyl. Gains over bpli alone on the same frames are reported beside (b). "
    "Labels for the gate: li_E (the bpli bar's target); scorer li(2) of the EFIT psi is secondary.")
BPLI_BAR = dict(cv=0.575, val=0.517, cv_comb=0.605, val_comb=0.547)


def _xy(set_name, darr):
    """bpli feature table of a records population + li_cd variants; li_E target; aligned idx."""
    from fec import betap_li as BL
    r = dict(np.load(BL.OUT / f"records_{set_name}.npz"))
    t = dict(np.load(BL.OUT / f"targets_{set_name}.npz"))
    assert np.array_equal(r["idx"], darr["idx"]) and np.array_equal(t["idx"], darr["idx"])
    X = BL.feature_table(r)
    for v in range(darr["li"].shape[1]):
        X[f"li_cd{v}"] = darr["li"][:, v]
    return X, BL._target(t, "li"), r["shot"]


def cmd_freeze():
    """TRAIN-only choices -> OUT/frozen.json (+ a frozen copy of this module)."""
    import shutil
    from fec import betap_li as BL
    fz = OUT / "frozen.json"
    if fz.exists():
        raise SystemExit(f"{fz} exists (frozen); delete deliberately to refreeze")
    d = dict(np.load(OUT / "d3d_cal.npz"))
    X, y, shot = _xy("cal", d)
    sel = json.loads((BL.OUT / "selection.json").read_text())
    fb = list(sel["li"]["model"]["feats"])
    ph = _phase("cal", len(y))
    var = variants()
    cv = {}
    for v in range(len(var)):
        cv[v] = BL.cv_oof(X, y, [f"li_cd{v}"], shot, "li")[1]
    nom = variant_index(*NOMINAL)
    best = max(cv, key=lambda v: cv[v])
    v_sel = best if cv[best] - cv[nom] > 0.002 else nom
    X["li_cd"] = X[f"li_cd{v_sel}"]
    have = np.isfinite(X["li_cd"])
    y_c = np.where(have, y, np.nan)                       # paired frames for the baseline
    p_alone, r_alone = BL.cv_oof(X, y_c, ["li_cd"], shot, "li")
    p_base, r_base = BL.cv_oof(X, y_c, fb, shot, "li")
    p_comb, r_comb = BL.cv_oof(X, y_c, fb + ["li_cd"], shot, "li")
    r_base_all = BL.cv_oof(X, y, fb, shot, "li")[1]
    byph = {p: dict(alone=_r2(y_c[ph == p], p_alone[ph == p]), bpli=_r2(y_c[ph == p], p_base[ph == p]),
                    comb=_r2(y_c[ph == p], p_comb[ph == p]), n=int(np.isfinite(y_c[ph == p]).sum()))
            for p in ("ramp_up", "flat", "ramp_down")}
    allm = np.ones(len(y), bool)
    m_alone = BL.fit_reg(X, y, ["li_cd"], allm, "li")
    m_comb = BL.fit_reg(X, y, fb + ["li_cd"], allm, "li")
    sens = {p: sensitivity(d["li"], v_sel, ph == p) for p in ("ramp_up", "flat", "ramp_down")}
    sens["all"] = sensitivity(d["li"], v_sel)
    froz = OUT / "frozen"
    froz.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), froz / "current_diffusion_frozen.py")
    rep = dict(rule=FREEZE_RULE, bar=BPLI_BAR, variants=var, nominal_default=NOMINAL,
               cv_r2_alone_by_variant={str(var[v]): cv[v] for v in cv}, selected_variant=int(v_sel),
               selected=var[v_sel], bpli_feats=fb,
               train_cv=dict(alone=r_alone, bpli_same_frames=r_base, combined=r_comb,
                             bpli_all_frames=r_base_all, gain_combined=r_comb - r_base,
                             coverage=float(have[np.isfinite(y)].mean()), by_phase=byph),
               models=dict(alone=m_alone, combined=m_comb), train_sensitivity=sens,
               sha256={"d3d_cal.npz": _sha(OUT / "d3d_cal.npz"),
                       "records_cal.npz": _sha(BL.OUT / "records_cal.npz"),
                       "targets_cal.npz": _sha(BL.OUT / "targets_cal.npz"),
                       "selection.json": _sha(BL.OUT / "selection.json"),
                       "current_diffusion.py": _sha(Path(__file__))},
               frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    fz.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("cv_r2_alone_by_variant", "selected", "train_cv",
                                          "train_sensitivity")}, indent=1, default=str))
    return rep


def cone_floor(jpsi, u, thr=1e-9):
    """Report-only: on frames where kfit's cone binds (both (1-x)^2 weights <= thr), is the diffused
    j(psiN) broader than the cone floor (1 - x^2)?  LS projection of j / <j> on {(1-x^2), (1-x)^2}:
    a negative (1-x)^2 coefficient = broader than every in-cone profile."""
    x = PSIN_OUT
    A = np.stack([1 - x ** 2, (1 - x) ** 2], 1)
    ok = np.isfinite(jpsi).all(1)
    coef = np.full((len(jpsi), 2), np.nan)
    if ok.any():
        coef[ok] = np.linalg.lstsq(A, jpsi[ok].T, rcond=None)[0].T
    binds = (u[:, 1] <= thr) & (u[:, 3] <= thr)
    broader = coef[:, 1] < 0
    out = {}
    for name, m in (("binds", binds & ok), ("free", ~binds & ok)):
        out[name] = dict(n=int(m.sum()), frac_broader=float(broader[m].mean()) if m.any() else None,
                         j0_med=float(np.median(jpsi[m, 0])) if m.any() else None,
                         j05_med=float(np.median(jpsi[m, 10])) if m.any() else None)
    return out


def cmd_evaluate():
    """VAL subset (40 shots, 4,750 frames), once, after the freeze: gate verdict -> val_report.json."""
    from fec import betap_li as BL
    fr = json.loads((OUT / "frozen.json").read_text())
    d = dict(np.load(OUT / "d3d_val.npz"))
    X, y, shot = _xy("val", d)
    v = fr["selected_variant"]
    X["li_cd"] = X[f"li_cd{v}"]
    have = np.isfinite(X["li_cd"])
    ph = _phase("val", len(y))
    p_alone = BL.predict_reg(fr["models"]["alone"], X)[0]
    p_comb = BL.predict_reg(fr["models"]["combined"], X)[0]
    tp = np.load(BL.OUT / "targets_pred_val.npz")
    assert np.array_equal(tp["idx"], d["idx"])
    p_bpli = tp["li_t"]
    y_c = np.where(have, y, np.nan)
    lab = _labels("val")
    ys = np.where(lab["li_scorer_ok"], lab["li_scorer"], np.nan)
    ys_c = np.where(have, ys, np.nan)

    def block(yy):
        m = np.isfinite(yy) & have
        return dict(alone=_r2(yy, p_alone), combined=_r2(yy, p_comb), bpli_same_frames=_r2(yy, p_bpli),
                    li_cyl_raw_corr=float(np.corrcoef(X["li_cd"][m], yy[m])[0, 1]) if m.sum() > 3 else None,
                    n=int(m.sum()))
    res = dict(label_li_E=block(y_c), label_scorer=block(ys_c),
               bpli_all_frames=_r2(y, p_bpli), coverage=float(have[np.isfinite(y)].mean()),
               by_phase={p: block(np.where(ph == p, y_c, np.nan)) for p in
                         ("ramp_up", "flat", "ramp_down")})
    sens = {p: sensitivity(d["li"], v, ph == p) for p in ("ramp_up", "flat", "ramp_down")}
    sens["all"] = sensitivity(d["li"], v)
    kb = np.load(BL.OUT / "pred_val" / f"{BL.BASE}.npz")
    assert np.array_equal(kb["idx"], d["idx"])
    cone = cone_floor(d["jpsi"][:, v].astype(float), kb["u"])
    tc = fr["train_cv"]
    pass_a = res["label_li_E"]["alone"] >= BPLI_BAR["val"]
    pass_b = tc["combined"] >= BPLI_BAR["cv_comb"] and res["label_li_E"]["combined"] >= BPLI_BAR["val_comb"]
    gate = dict(a_val_alone=res["label_li_E"]["alone"], a_pass=bool(pass_a),
                b_train_cv_combined=tc["combined"], b_val_combined=res["label_li_E"]["combined"],
                b_pass=bool(pass_b), b_gain_cv=tc["gain_combined"],
                b_gain_val=res["label_li_E"]["combined"] - res["label_li_E"]["bpli_same_frames"],
                sens_ratio=sens["all"]["ratio"], sens_pass=bool(sens["all"]["passes"]))
    gate["stage1_pass"] = bool((pass_a or pass_b) and gate["sens_pass"])
    rep = dict(frozen_sha=_sha(OUT / "frozen.json"), d3d_val_sha=_sha(OUT / "d3d_val.npz"),
               selected=fr["selected"], val=res, val_sensitivity=sens, cone_floor=cone, gate=gate)
    (OUT / "val_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return rep


def cmd_mastreport(f=None):
    """MAST, inputs only: li_cyl distribution, temporal smoothness, Zeff / IC sensitivity."""
    z = dict(np.load(f or OUT / "mast.npz"))
    fr = json.loads((OUT / "frozen.json").read_text()) if (OUT / "frozen.json").exists() else None
    v = fr["selected_variant"] if fr else variant_index(*NOMINAL)
    cov = z.get("covered", np.ones(len(z["ok"]), bool))
    li = z["li"][:, v]
    ok = cov & z["ok"]
    shot = z["shot"]
    d1, sd_shot = [], []
    for s in np.unique(shot[cov]):
        m = (shot == s) & ok
        x = li[m]
        if x.size >= 3:
            d1.append(np.diff(x))
            sd_shot.append(np.std(x))
    d1 = np.concatenate(d1) if d1 else np.zeros(0)
    rep = dict(frames=int(cov.sum()), ok_frac=float(ok[cov].mean()), variant=variants()[v],
               li_cyl_pct=np.nanpercentile(li[ok], [1, 5, 25, 50, 75, 95, 99]).tolist(),
               smooth=dict(frame_to_frame_rms=float(np.sqrt(np.mean(d1 ** 2))) if d1.size else None,
                           frame_to_frame_p95=float(np.percentile(np.abs(d1), 95)) if d1.size else None,
                           within_shot_sd_med=float(np.median(sd_shot)) if sd_shot else None,
                           across_frame_sd=float(np.nanstd(li[ok]))),
               sensitivity=sensitivity(z["li"], v, ok),
               q0_pct=np.nanpercentile(z["q0"][ok, v], [5, 50, 95]).tolist(),
               t_rel_pct=np.nanpercentile(z["t_rel"][ok], [1, 50, 99]).tolist(),
               gap_hist_gt25=float(np.mean(z["gap_hist"][ok] > GAP_MS)))
    if fr:
        from fec import betap_li as BL
        mal = fr["models"]["alone"]
        cal = BL.predict_reg(mal, {"li_cd": li})[0]
        lo, hi = mal["lo"][0], mal["hi"][0]
        rep["li_cal_pct"] = np.nanpercentile(cal[ok], [1, 5, 50, 95, 99]).tolist()
        rep["li_cyl_inrange_train"] = float(np.mean((li[ok] >= lo) & (li[ok] <= hi)))
    (OUT / "mast_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("d3d")
    a.add_argument("--set", default="cal", choices=["cal", "val"])
    a.add_argument("--limit-shots", type=int, default=0)
    a.add_argument("--nproc", type=int, default=4)
    a = sub.add_parser("probe")
    a.add_argument("--limit-shots", type=int, default=30)
    a.add_argument("--nproc", type=int, default=4)
    a.add_argument("--saw", type=float, default=1.0)
    a.add_argument("--tag", default="")
    a = sub.add_parser("mast")
    a.add_argument("--limit-shots", type=int, default=0)
    a.add_argument("--nproc", type=int, default=4)
    sub.add_parser("freeze")
    sub.add_parser("evaluate")
    a = sub.add_parser("mastreport")
    a.add_argument("--file", default=None)
    args = ap.parse_args()
    if args.cmd == "d3d":
        run_d3d(args.set, args.limit_shots, args.nproc)
    elif args.cmd == "probe":
        kw = {} if args.saw == 1.0 else dict(saw=args.saw)
        cmd_probe(args.limit_shots, args.nproc, tag=args.tag, **kw)
    elif args.cmd == "mast":
        run_mast(args.limit_shots, args.nproc)
    elif args.cmd == "freeze":
        cmd_freeze()
    elif args.cmd == "evaluate":
        cmd_evaluate()
    elif args.cmd == "mastreport":
        cmd_mastreport(args.file)


if __name__ == "__main__":
    main()
