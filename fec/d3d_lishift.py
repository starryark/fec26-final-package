"""DIII-D li shift toward current diffusion in the ramp-down gate (agent lishift, 2026-10-09).

Plan, freeze (section 4.0) and results: docs/research/D3D_LISHIFT_20261009.md.

In ramp-down frames (input-only gate g of fec/d3d_rdspec.py) the derived li of a predicted psi map
is moved toward the current-diffusion estimate li_cd (fec/current_diffusion.py, raw li_cyl of the
frozen variant (Z_eff 1.5, 'te15')) by a boundary-preserving interior flux relabelling

    phi = -psi (axis maximum),  x = (phi_ax - phi) / (phi_ax - phi_b)  inside the LCFS,
    phi' = phi_ax - (phi_ax - phi_b) * F(x),
    F(x) = x + eps * xc * u (1 - u)^2 (u = x / xc) for x < xc, F(x) = x for x >= xc,
    F(0) = 0, F(xc) = xc, F'(xc) = 1, F' >= 0.5 for eps in [-0.5, 1.5]   (doc amendment A1)
with xc = min x over the inside nodes that the scorer's Ampere loop integral reads (g_stencil).

The axis node value, the LCFS level (found by the starter's own bisection inside the TRAIN-only
envelope), every G-stencil node and every node outside the LCFS polygon are unchanged
(byte-identical), so G, the contour and the shape scalars are unchanged exactly; per frame, eps is
found by scan + bisection so that the env_train derived li equals
li* = clip(li_pred + beta * g * (li_cd - li_pred), 0.1, 4.9).

li_cd geometry comes from PREDICTED maps (never labels): betan_transfer.geometry on -psi ->
current_diffusion._maps_to_rho -> run_shot, Thomson / Ip from the raw parquet (inputs only).

    FEC_WORK=<FEC_IDX_WORK> python -B fec/d3d_lishift.py licd --config diii_d_train \
        --psi <psi.npy> --rows <rows.npy> --out <licd.npz> [--gated-only] [--nproc 6]
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

EPS_LO, EPS_HI = -0.5, 1.5        # F' >= 0.5 everywhere (monotone with margin)
N_SCAN = 9                        # scan points on [0, eps_bound]
LI_TOL, MAX_HALV = 5e-4, 30
LI_CAP_LO, LI_CAP_HI = 0.1, 4.9   # target clip (never push a frame out of the scorer's range)
LI_OK_LO, LI_OK_HI = 0.1, 5.0     # applicability: base li inside the scorer's plausible range
VARIANT = 1                       # frozen.json selected_variant = (Z_eff 1.5, 'te15')
BCOIL = "magnetics_bcoil"
OFF0 = 6


# ======================================================================== li_cd (inputs + predicted maps)
def _grid():
    from fec.d3d_rdspec import load_env
    R, Z, _, _ = load_env()
    return R, Z


def shot_licd(parquet, psi_ds, t_ms, bt_kA, gR, gZ):
    """Current-diffusion li for one shot on predicted maps.

    psi_ds: (n, 65, 65) predicted maps in the ds convention (axis minimum); t_ms: (n,) frame times
    (any order); bt_kA: |bcoil| (kA) per frame (median used). Returns (li (n, 6), ok (n,), info)."""
    from fec import betan_transfer as BT
    from fec import current_diffusion as CD
    t_ms = np.asarray(t_ms, float)
    o = np.argsort(t_ms, kind="stable")
    n = len(o)
    psis = -np.asarray(psi_ds, np.float64)[o]
    pa, pb, ra, za = (np.full(n, np.nan) for _ in range(4))
    geo = np.full((n, 3), np.nan)
    for k in range(n):
        try:
            gm = BT.geometry(psis[k], gR, gZ, BT.D3D_AXIS_BOX)
        except Exception:  # noqa: BLE001
            gm = None
        if gm is None:
            continue
        pa[k], pb[k], ra[k], za[k] = gm["psi_ax"], gm["psi_b"], gm["Rax"], gm["Zax"]
        geo[k] = gm["a"], gm["kappa"], gm["Rgeo"]
    li = np.full((n, len(CD.variants())), np.nan)
    ok = np.zeros(n, bool)
    sh = CD.read_d3d(str(parquet))
    if sh is None:
        return li, ok, {"status": "layout"}
    rho, geo2 = CD._maps_to_rho(sh, psis, pa, pb, ra, za, gR, gZ, geo)
    bt = float(np.nanmedian(np.abs(np.asarray(bt_kA, float))))
    F = CD.MU0 * CD.D3D_TF_TURNS * bt * 1e3 / (2 * np.pi)
    ts = t_ms[o]
    sh.update(out_t=ts, map_t=ts, map_rho=rho, map_geo=geo2, pre_map="hold", F=F)
    res, info = CD.run_shot(sh)
    li[o] = res["li"]
    ok[o] = res["ok"]
    info = dict(info, n_geo_fail=int((~np.isfinite(pa)).sum()))
    return li, ok, info


def _licd_job(job):
    stem, parquet, psi_path, pos, t_ms, bt, gR, gZ = job
    try:
        P = np.load(psi_path, mmap_mode="r")
        psi = np.asarray(P[np.asarray(pos)], np.float64)
        li, ok, info = shot_licd(parquet, psi, t_ms, bt, gR, gZ)
        return stem, pos, li, ok, info
    except Exception as e:  # noqa: BLE001
        return stem, pos, None, None, {"status": f"error {type(e).__name__}: {e}"}


def run_licd(config, psi_path, rows, shots=None, nproc=6, log_every=25):
    """li_cd for every psi row whose shot is in `shots` (default: all shots).

    psi_path: (N, 65, 65) predicted maps; rows: (N,) ds row of each psi row (config ds).
    Returns dict(li (N, 6) NaN elsewhere, ok (N,), done (N,) bool, info {stem: info})."""
    from multiprocessing import Pool
    from fec.features import load_ds
    from fec.paths import DATA
    ds = load_ds(config)
    rows = np.asarray(rows)
    shot_r = ds["shot"][rows]
    t_r = np.asarray(ds["t_ms"])[rows]
    bi = ds["info"]["channels"].index(BCOIL)
    gR, gZ = _grid()
    want = np.unique(shot_r) if shots is None else np.intersect1d(np.unique(shot_r), shots)
    jobs = []
    for s in want:
        pos = np.flatnonzero(shot_r == s)
        stem = ds["info"]["order"][int(s)]
        bt = np.asarray(ds["cur"][rows[pos], bi, OFF0], float)
        jobs.append((stem, str(DATA / config / f"{stem}.parquet"), str(psi_path), pos, t_r[pos], bt, gR, gZ))
    N = len(rows)
    out = {"li": np.full((N, 6), np.nan), "ok": np.zeros(N, bool), "done": np.zeros(N, bool)}
    infos = {}
    t0 = time.time()
    with Pool(max(1, min(nproc, 6))) as pool:
        for i, (stem, pos, li, ok, info) in enumerate(pool.imap_unordered(_licd_job, jobs, chunksize=1)):
            infos[stem] = info
            if li is not None:
                out["li"][pos] = li
                out["ok"][pos] = ok
                out["done"][pos] = True
            if i % log_every == 0:
                print(f"licd {config} {i + 1}/{len(jobs)} {time.time() - t0:.0f}s {stem} "
                      f"{info.get('status')}", flush=True)
    out["info"] = infos
    return out


def cmd_licd(a):
    from fec.features import load_ds
    from fec.d3d_rdspec import ds_gate
    rows = np.load(a.rows)
    ds = load_ds(a.config)
    shots = None
    if a.gated_only:
        g, _, _ = ds_gate(ds)
        shots = np.unique(ds["shot"][rows][g[rows] > 0])
    t0 = time.time()
    res = run_licd(a.config, a.psi, rows, shots, a.nproc)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, li=res["li"], ok=res["ok"], done=res["done"], rows=rows,
             variant=VARIANT, psi=str(a.psi), config=a.config)
    Path(str(out).replace(".npz", "_info.json")).write_text(json.dumps(res["info"], indent=0, default=str))
    st = [v.get("status") for v in res["info"].values()]
    print(f"LICD_DONE {out}: shots {len(st)} status { {s: st.count(s) for s in set(st)} }, rows done "
          f"{int(res['done'].sum())}, ok {int(res['ok'].sum())}, {time.time() - t0:.0f}s", flush=True)


# ======================================================================== transform (pure functions)
def F_of(x, eps):
    """F(x) = x + eps x (1 - x)^2: F(0) = 0, F(1) = 1, F'(1) = 1."""
    x = np.asarray(x, np.float64)
    return x + eps * x * (1.0 - x) ** 2


def dF_of(x, eps):
    x = np.asarray(x, np.float64)
    return 1.0 + eps * (1.0 - x) * (1.0 - 3.0 * x)


def _starter():
    from fec import score as SC   # noqa: F401  (puts the starter / fusion_scoring on sys.path)
    import lcfs as LC
    import derive as DV
    return LC, DV


def lcfs_geometry(psi, env, n_iter=22, n_points=512):
    """The starter's LCFS bisection (lcfs._extract_one_sign, axis_sign -1) on psi (ds convention)
    inside `env` = (R, Z, mask bool, mask float), also returning the axis node value and the LCFS
    level. Returns dict(C, phi_ax, phi_b, iz, ir, inside, x) or None (same failures as the starter)."""
    LC, DV = _starter()
    R, Z, mc, mf = env
    phi = -np.asarray(psi, np.float64).reshape(65, 65)
    if not np.isfinite(phi).all():
        return None
    mask = mc if (mc is not None and mc.any()) else None
    _, _, iz, ir, _ = LC.find_o_point(phi, R, Z, mask)
    axis_val = float(phi[iz, ir])
    border = np.concatenate([phi[0, :], phi[-1, :], phi[:, 0], phi[:, -1]])
    lo, hi = float(np.min(border)), axis_val - 1e-9
    if not hi > lo:
        return None
    best = LC._best_contour_at(phi, hi, iz, ir, mf)
    if best is None:
        return None
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        c = LC._best_contour_at(phi, mid, iz, ir, mf)
        if c is not None:
            hi, best = mid, c
        else:
            lo = mid
    C = LC.resample_closed(LC.grid_to_physical(best, R, Z), n_points)
    RR, ZZ = np.meshgrid(R, Z)
    inpoly = DV._points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
    den = axis_val - hi
    x = (axis_val - phi) / den
    inside = inpoly & (x >= 0.0) & (x < 1.0)
    band = g_stencil(C, R, Z)
    bx = x[band & inside]
    xc = float(bx.min()) if bx.size else 1.0
    change = inside & (x < xc)
    return dict(C=C, phi_ax=axis_val, phi_b=hi, iz=int(iz), ir=int(ir), inside=inside, x=x,
                band=band, xc=xc, change=change)


def g_stencil(C, R, Z):
    """Grid nodes that derive.internal_inductance's Ampere loop integral reads: the corners of the
    cells holding the contour points (bilinear) dilated by the np.gradient 4-neighbour stencil."""
    from scipy.ndimage import binary_dilation
    nR, nZ = len(R), len(Z)
    fr = np.interp(C[:, 0], R, np.arange(nR))
    fz = np.interp(C[:, 1], Z, np.arange(nZ))
    r0 = np.clip(np.floor(fr).astype(int), 0, nR - 2)
    z0 = np.clip(np.floor(fz).astype(int), 0, nZ - 2)
    m = np.zeros((nZ, nR), bool)
    for dz in (0, 1):
        for dr in (0, 1):
            m[z0 + dz, r0 + dr] = True
    return binary_dilation(m, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))


def Fc_of(x, eps, xc):
    """Compact-support relabelling (amendment A1): identity for x >= xc, else
    x + eps * xc * u (1 - u)^2 with u = x / xc."""
    x = np.asarray(x, np.float64)
    u = np.clip(x / xc, 0.0, 1.0)
    return np.where(x < xc, x + eps * xc * u * (1.0 - u) ** 2, x)


def relabel(psi, eps, geo):
    """float32 map: nodes in geo['change'] (inside the LCFS, x < x_c) relabelled with F_c(., eps);
    all other nodes are the input's float32 bytes."""
    base = np.asarray(psi, np.float32).reshape(65, 65)
    out = base.copy()
    m = geo["change"]
    if eps == 0.0 or not m.any():
        return out
    phi_new = geo["phi_ax"] - (geo["phi_ax"] - geo["phi_b"]) * Fc_of(geo["x"][m], eps, geo["xc"])
    out[m] = (-phi_new).astype(np.float32)
    return out


def ampere_G(psi, env, C):
    """mu0 Ip as derive.internal_inductance computes it (loop integral of Bp on the polygon C)."""
    _, DV = _starter()
    R, Z = env[0], env[1]
    Bp = DV._poloidal_field(np.asarray(psi, np.float64), R, Z, "DIII-D")
    b = DV._bilinear(Bp, R, Z, C[:, 0], C[:, 1])
    Cc = np.vstack([C, C[0]])
    seg = np.sqrt(np.sum(np.diff(Cc, axis=0) ** 2, axis=1))
    return float(np.sum(0.5 * (b + np.roll(b, -1)) * seg))


def env_li_fn(env):
    from fec.d3d_rdspec import frame_li
    return lambda m: frame_li(m, env)


def solve_eps(psi, li_target, li0, geo, li_fn, eps_lo=EPS_LO, eps_hi=EPS_HI, n_scan=N_SCAN,
              tol=LI_TOL, max_halv=MAX_HALV):
    """eps with li(relabel(psi, eps)) = li_target. Scan [0, bound] (n_scan points incl. 0) in the
    direction whose first step moves li toward the target, then bisect the first crossing bracket.
    Returns (eps, li, status) with status 'ok' | 'noop' | 'unreachable'; unreachable -> eps 0."""
    d0 = li0 - li_target
    if abs(d0) <= tol:
        return 0.0, li0, "noop"

    def ev(e):
        v, li = li_fn(relabel(psi, e, geo))
        return li if (v and np.isfinite(li)) else np.nan

    def toward(d):
        return np.isfinite(d) and (np.sign(d) != np.sign(d0) or abs(d) < abs(d0))

    # direction: the side whose first scan step moves li toward the target
    chosen = None
    for bound in (eps_hi, eps_lo):
        e1 = bound / (n_scan - 1)
        l1 = ev(e1)
        if toward(l1 - li_target):
            chosen = (bound, e1, l1)
            break
    if chosen is None:
        return 0.0, li0, "unreachable"
    bound, e1, l1 = chosen
    grid = np.linspace(0.0, bound, n_scan)
    prev_e, prev_d = 0.0, d0
    for k, e in enumerate(grid[1:]):
        li = l1 if k == 0 else ev(e)
        if not np.isfinite(li):
            break
        d = li - li_target
        if abs(d) <= tol:
            return float(e), float(li), "ok"
        if np.sign(d) != np.sign(prev_d):         # crossing in (prev_e, e): bisect
            a, b, da = prev_e, e, prev_d
            best = (abs(d), e, li)
            for _ in range(max_halv):
                m = 0.5 * (a + b)
                lm = ev(m)
                if not np.isfinite(lm):
                    break
                dm = lm - li_target
                if abs(dm) < best[0]:
                    best = (abs(dm), m, lm)
                if abs(dm) <= tol:
                    break
                if np.sign(dm) == np.sign(da):
                    a, da = m, dm
                else:
                    b = m
            if best[0] > 4 * tol:                 # li(eps) jumps across the bracket (contour change)
                return 0.0, li0, "unreachable"
            return float(best[1]), float(best[2]), "ok"
        prev_e, prev_d = e, d
    return 0.0, li0, "unreachable"


def target_li(li_pred, li_cd, g, beta):
    t = li_pred + beta * g * (li_cd - li_pred)
    return float(np.clip(t, LI_CAP_LO, LI_CAP_HI))


def shift_frame(psi, li_cd, g, beta, env, li_fn=None):
    """One frame: (map float32, info). Unchanged (input float32 bytes) unless applicable and solved."""
    li_fn = li_fn or env_li_fn(env)
    base = np.asarray(psi, np.float32).reshape(65, 65)
    info = {"status": "skip", "eps": 0.0}
    if not (g > 0 and np.isfinite(li_cd) and beta > 0):
        info["status"] = "not_gated" if not g > 0 else ("no_licd" if not np.isfinite(li_cd) else "beta0")
        return base.copy(), info
    v, li0 = li_fn(base)
    info["li_pred"] = float(li0)
    if not (v and np.isfinite(li0) and LI_OK_LO <= li0 <= LI_OK_HI):
        info["status"] = "out_of_range"
        return base.copy(), info
    geo = lcfs_geometry(base, env)
    if geo is None:
        info["status"] = "no_lcfs"
        return base.copy(), info
    lt = target_li(li0, li_cd, g, beta)
    eps, li, st = solve_eps(base, lt, li0, geo, li_fn)
    info.update(li_target=lt, eps=eps, li_final=li, status=st, xc=geo["xc"],
                n_change=int(geo["change"].sum()))
    if st != "ok":
        return base.copy(), info
    return relabel(base, eps, geo), info


def _shift_job(args):
    from fec.d3d_rdspec import load_env
    maps, licd, g, beta, env_path = args
    env = load_env(env_path)
    li_fn = env_li_fn(env)
    out, infos = [], []
    for m, lc, gg, b in zip(maps, licd, g, beta):
        o, inf = shift_frame(m, lc, gg, b, env, li_fn)
        out.append(o); infos.append(inf)
    return np.stack(out) if out else np.zeros((0, 65, 65), np.float32), infos


def lishift_rows(psi, li_cd, g, beta, env_path=None, nproc=6, chunk=40):
    """Shift every row of psi (n, 65, 65) with its li_cd, g and beta (scalar or (n,)).
    Pure function of (psi, li_cd, g, beta, envelope). Returns (float32 maps (n,65,65), infos)."""
    from multiprocessing import Pool
    from fec.d3d_rdspec import env_train_path
    n = len(psi)
    beta = np.broadcast_to(np.asarray(beta, np.float64), (n,))
    ep = str(env_path or env_train_path())
    jobs = [(np.asarray(psi[i:i + chunk], np.float32), np.asarray(li_cd[i:i + chunk], np.float64),
             np.asarray(g[i:i + chunk], np.float64), beta[i:i + chunk], ep) for i in range(0, n, chunk)]
    if nproc <= 1:
        res = [_shift_job(j) for j in jobs]
    else:
        with Pool(nproc) as pool:
            res = pool.map(_shift_job, jobs, chunksize=1)
    maps = np.concatenate([r[0] for r in res]) if res else np.zeros((0, 65, 65), np.float32)
    return maps, sum((r[1] for r in res), [])


def apply_shift(psi_all, rows, maps):
    """float32 copy of psi_all with `rows` replaced by `maps`; every other row byte-identical."""
    out = np.array(psi_all, dtype=np.float32, copy=True)
    if len(rows):
        out[np.asarray(rows)] = np.asarray(maps, np.float32)
    return out


# ======================================================================== ensemble spread (label-free)
def weighted_var(L_, w):
    """Unbiased (reliability-weighted, ddof-1 form) variance across members: L_ (n_mem, n), w (n_mem,)."""
    w = np.asarray(w, np.float64) / np.sum(w)
    m = np.tensordot(w, L_, axes=1)
    return np.tensordot(w, (L_ - m) ** 2, axes=1) / (1.0 - np.sum(w ** 2))


def beta_kalman(kappa, s2, sig2_cd):
    v = kappa * s2
    return float(v / (v + sig2_cd))


def cmd_members(a):
    """Member maps of an ensemble on given ds rows (GPU predict), their env_train li and the ensemble
    mean (for a reproduction check against the stored ensemble array)."""
    from fec.d3d_rdspec import predict_rows, derived_li
    tags = a.tags.split(",")
    w = np.array([float(x) for x in a.weights.split(",")]) if a.weights else np.ones(len(tags))
    rows = np.load(a.rows)
    t0 = time.time()
    maps = np.zeros((len(tags), len(rows), 65, 65), np.float32)
    for i, t in enumerate(tags):
        p, _ = predict_rows([t], [1.0], rows, a.config)
        maps[i] = p.reshape(-1, 65, 65).astype(np.float32)
        print(f"member {t}: {time.time() - t0:.0f}s", flush=True)
    mean = np.tensordot(w / w.sum(), maps.astype(np.float64), axes=1).astype(np.float32)
    li = np.full((len(tags), len(rows)), np.nan); valid = np.zeros((len(tags), len(rows)), bool)
    for i in range(len(tags)):
        valid[i], li[i] = derived_li(maps[i], nproc=a.nproc)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rows=rows, tags=np.array(tags), weights=w, li=li, valid=valid, mean=mean)
    if a.save_maps:
        np.save(str(out).replace(".npz", "_maps.npy"), maps)
    s2 = weighted_var(li, w)
    print(f"MEMBERS_DONE {out}: {len(tags)} members x {len(rows)} rows, mean s2 {np.nanmean(s2):.5f}, "
          f"{time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------- CLI
def get_parser():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("licd")
    c.add_argument("--config", default="diii_d_train")
    c.add_argument("--psi", required=True, help="(N,65,65) predicted maps (ds convention)")
    c.add_argument("--rows", required=True, help=".npy (N,) ds row of each psi row")
    c.add_argument("--out", required=True)
    c.add_argument("--gated-only", action="store_true", help="only shots with g > 0 rows")
    c.add_argument("--nproc", type=int, default=6)
    m = sp.add_parser("members")
    m.add_argument("--config", default="diii_d_train")
    m.add_argument("--tags", required=True)
    m.add_argument("--weights", default="")
    m.add_argument("--rows", required=True, help=".npy ds rows")
    m.add_argument("--out", required=True)
    m.add_argument("--save-maps", action="store_true")
    m.add_argument("--nproc", type=int, default=6)
    return ap


def main(argv=None):
    a = get_parser().parse_args(argv)
    return {"licd": cmd_licd, "members": cmd_members}[a.cmd](a)


if __name__ == "__main__":
    main()
