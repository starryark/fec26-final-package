"""MAST zero-shot v1: physics equilibria from measured coil currents + Ip, profile chosen
per frame by Thomson flux-function symmetry (inputs only, no MAST labels anywhere).

Stage 1 (GPU): force-free vertical position (secant on the radial-field feedback), then
forward solves for a grid of profile shapes; keep the shape minimising the inboard/outboard
Te mismatch on the midplane chord.  Stage 2 (CPU): q95 (contour integral, F from the TF
current with 24 turns) and betaN = K_BETA * electron-pressure betaN (K calibrated on DIII-D).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

COMBOS = [(0.05, 2, 1), (0.05, 1, 2), (0.05, 2, 2), (0.05, 3, 1), (0.3, 2, 1), (0.05, 1, 3),
          (0.3, 1, 2), (0.3, 1, 3)]
DEFAULT = COMBOS.index((0.05, 2, 2))
OUT = WORK / "mast_v1b"
CONFIG = "mast_public_test"   # mast_private_test in the Final phase (set by --config)


def thomson_frame(z, k):
    R = np.concatenate([z["ts_core_R"], z["ts_edge_R"]])
    te = np.concatenate([z["ts_core_te"][k], z["ts_edge_te"][k]]).astype(float)
    ne = np.concatenate([z["ts_core_ne"][k], z["ts_edge_ne"][k]]).astype(float)
    return R, te, ne


def stage1(chunk=1024, limit=0):
    import torch
    from fec.features import load_ds
    from fec.gs import Solver
    from fec.kinetic import chord_psiN, symmetry_mismatch
    from fec.machines import MAST_CH, MAST_COILS, mast_setup
    ds = load_ds(CONFIG)
    info = ds["info"]
    N = len(ds["t_ms"]) if not limit else limit
    cur = np.asarray(ds["cur"][:N], dtype=np.float64)[:, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    spec, M = mast_setup()
    S = Solver(spec, M)
    OUT.mkdir(parents=True, exist_ok=True)
    psi_out = np.lib.format.open_memmap(OUT / "psi.npy", "w+", np.float32, (N, 65, 65))
    meta = {k: np.zeros(N) for k in ("psi_ax", "psi_b", "Rax", "Zax", "Zt", "c", "combo",
                                     "mm", "mm_default", "dI")}
    cache = {}
    t0 = time.time()
    for s0 in range(0, N, chunk):
        sl = slice(s0, min(N, s0 + chunk))
        B = sl.stop - sl.start
        I = cur[sl][:, ci] * 1e3
        Ip = np.maximum(cur[sl, MAST_CH.index("magnetics_plasma_current")] * 1e3, 5e4)
        prof0 = np.tile(COMBOS[DEFAULT], (B, 1))
        o0 = S.solve(I, Ip, prof0, np.zeros(B), n_iter=150, Rinit=np.full(B, 0.9))
        o1 = S.solve(I, Ip, prof0, np.full(B, 0.05), n_iter=60, I0=o0["I"], c0=o0["c"])
        c0, c1 = o0["c"].cpu().numpy(), o1["c"].cpu().numpy()
        den = np.where(np.abs(c1 - c0) > 1e-12, c1 - c0, np.nan)
        Zs = np.nan_to_num(np.clip(-c0 * 0.05 / den, -0.3, 0.3))
        TS = []
        for g in range(sl.start, sl.stop):
            stem = info["order"][ds["shot"][g]]
            if stem not in cache:
                cache.clear()
                cache[stem] = dict(np.load(WORK / "shots" / CONFIG / f"{stem}.npz"))
            TS.append(thomson_frame(cache[stem], ds["fidx"][g]))
        best_mm = np.full(B, np.inf); best = np.full(B, DEFAULT); mm_def = np.full(B, np.nan)
        best_psi = None; keep = {}
        for cidx, combo in enumerate(COMBOS):
            o = S.solve(I, Ip, np.tile(combo, (B, 1)), Zs, n_iter=100, I0=o0["I"], c0=o0["c"])
            psi = o["psi"].cpu().numpy(); pa = o["psi_ax"].cpu().numpy()
            pb = o["psi_b"].cpu().numpy(); ra = o["Rax"].cpu().numpy()
            za = o["Zax"].cpu().numpy(); dI = o["dI"].cpu().numpy(); cc = o["c"].cpu().numpy()
            mm = np.full(B, np.nan)
            for i in range(B):
                R, te, _ = TS[i]
                pn = chord_psiN(psi[i], pa[i], pb[i], spec.gR, spec.gZ, R, np.zeros_like(R))
                mm[i] = symmetry_mismatch(pn, R, te, ra[i])[0]
            mm = np.where(dI < 1e-2, mm, np.nan)   # unconverged frames cannot win
            if cidx == DEFAULT:
                mm_def = mm.copy()
                keep["default"] = (psi, pa, pb, ra, za, cc, dI)
            if best_psi is None:
                best_psi = psi.copy(); bpa, bpb, bra, bza, bcc, bdI = pa.copy(), pb.copy(),                     ra.copy(), za.copy(), cc.copy(), dI.copy()
                best[:] = cidx
                best_mm = np.where(np.isfinite(mm), mm, np.inf)
                continue
            better = np.isfinite(mm) & (mm < best_mm)
            if better.any():
                best_psi[better] = psi[better]; bpa[better] = pa[better]; bpb[better] = pb[better]
                bra[better] = ra[better]; bza[better] = za[better]; bcc[better] = cc[better]
                bdI[better] = dI[better]; best[better] = cidx; best_mm[better] = mm[better]
        # frames where no combo produced a finite mismatch -> default solution
        nofit = ~np.isfinite(best_mm)
        if nofit.any():
            psi, pa, pb, ra, za, cc, dI = keep["default"]
            best_psi[nofit] = psi[nofit]; bpa[nofit] = pa[nofit]; bpb[nofit] = pb[nofit]
            bra[nofit] = ra[nofit]; bza[nofit] = za[nofit]; bcc[nofit] = cc[nofit]
            bdI[nofit] = dI[nofit]; best[nofit] = DEFAULT
        psi_out[sl] = best_psi
        for k, v in (("psi_ax", bpa), ("psi_b", bpb), ("Rax", bra), ("Zax", bza), ("Zt", Zs),
                     ("c", bcc), ("combo", best), ("mm", best_mm), ("mm_default", mm_def),
                     ("dI", bdI)):
            meta[k][sl] = v
        print(f"{sl.stop}/{N} {time.time() - t0:.0f}s  best-combo counts "
              f"{np.bincount(best, minlength=len(COMBOS)).tolist()} mm med "
              f"{np.nanmedian(best_mm):.3f} default {np.nanmedian(mm_def):.3f}", flush=True)
        torch.cuda.empty_cache()
    psi_out.flush()
    np.savez(OUT / "meta.npz", **meta)


# ----------------------------------------------------------------------------- stage 2
MU0 = 4e-7 * np.pi
K_BETA = 2.3       # EFIT betaN / Thomson-electron betaN on DIII-D (robust fit, R2~0.92)
N_TF = 24


def _scal_worker(args):
    from fec.eqscalars import q_at, surface_contour
    from fec.kinetic import chord_psiN
    from matplotlib.path import Path as MPath
    from scipy.interpolate import RectBivariateSpline
    (psi, pa, pb, ra, za, itf, ip, R, te, ne, gR, gZ) = args[:12]
    blim = args[12] if len(args) > 12 else False
    blim_active = 0.0
    if blim and np.isfinite(pa) and np.isfinite(pb):
        # boundary-only centre-column limiter (R=0.20 m armour, straight section |Z|<=1.0 m):
        # psi_b = max(psi_X, psi_lim) for the scalars only; the submitted psi map is unchanged
        j = int(np.searchsorted(gR, 0.20)) - 1
        w = (0.20 - gR[j]) / (gR[j + 1] - gR[j])
        rows = np.abs(gZ) <= 1.0 + 1e-9
        pl = (1 - w) * psi[rows, j] + w * psi[rows, j + 1]
        pl = pl[pl < pa]
        if pl.size and pl.max() > pb:
            pb = float(pl.max())
            blim_active = 1.0
    F = MU0 * N_TF * abs(itf) * 1e3 / (2 * np.pi)
    out = dict(q95=np.nan, q90=np.nan, betaN_e=np.nan, Rgeo=np.nan, a=np.nan,
               blim_active=blim_active)
    if not (np.isfinite(pa) and np.isfinite(pb) and pb > -1e20 and pa > pb):
        return out   # no X-point found (sentinel psi_b): scalars filled per shot later
    try:
        sp = RectBivariateSpline(gZ, gR, psi)
        out["q95"] = q_at(psi, gR, gZ, pa, pb, ra, za, F, 0.95, spline=sp)
        out["q90"] = q_at(psi, gR, gZ, pa, pb, ra, za, F, 0.90, spline=sp)
        psiN = (pa - psi) / (pa - pb)
        c = surface_contour(psiN, gR, gZ, 0.995, ra, za)
        if c is None:
            return out
        lr, lz = c
        Rgeo = 0.5 * (lr.max() + lr.min()); a = 0.5 * (lr.max() - lr.min())
        out["Rgeo"], out["a"] = Rgeo, a
        ok = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
        if ok.sum() < 10:
            return out
        pn = chord_psiN(psi, pa, pb, gR, gZ, R[ok], np.zeros(ok.sum()))
        pe = te[ok] * ne[ok] * 1.602e-19
        ins = pn < 1
        if ins.sum() < 6:
            return out
        bins = np.linspace(0, 1, 11); idx = np.digitize(pn[ins], bins) - 1
        prof = np.array([np.median(pe[ins][idx == b]) if (idx == b).any() else np.nan
                         for b in range(10)])
        xb = 0.5 * (bins[1:] + bins[:-1]); okb = np.isfinite(prof)
        if okb.sum() < 4:
            return out
        RR, ZZ = np.meshgrid(gR, gZ)
        m = MPath(np.c_[lr, lz]).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
        if m.sum() < 5:
            return out
        pcell = np.interp(psiN[m], xb[okb], prof[okb])
        pav = np.sum(pcell * RR[m]) / np.sum(RR[m])
        B0 = F / ra   # EFIT++ betaN uses the vacuum toroidal field at the magnetic axis
        out["betaN_e"] = 100 * 2 * MU0 * pav / B0 ** 2 * a * B0 / (abs(ip) / 1e6)
    except Exception:
        pass
    return out


def stage2(nproc=6, boundary_limiter=False):
    from fec.features import load_ds
    from fec.machines import MAST_CH
    from fec.paths import WORK as W
    ds = load_ds(CONFIG)
    info = ds["info"]
    psi = np.load(OUT / "psi.npy", mmap_mode="r")
    meta = dict(np.load(OUT / "meta.npz"))
    N = psi.shape[0]
    cur = np.asarray(ds["cur"][:N], dtype=np.float64)[:, :, 6]
    from fec.geometry import machine_geometry
    geo = machine_geometry(CONFIG)
    gR, gZ = geo["grid_R"], geo["grid_Z"]

    def jobs():
        cache = {}
        for g in range(N):
            stem = info["order"][ds["shot"][g]]
            if stem not in cache:
                cache.clear()
                cache[stem] = dict(np.load(W / "shots" / CONFIG / f"{stem}.npz"))
            R, te, ne = thomson_frame(cache[stem], ds["fidx"][g])
            yield (np.asarray(psi[g], dtype=np.float64), meta["psi_ax"][g], meta["psi_b"][g],
                   meta["Rax"][g], meta["Zax"][g], cur[g, MAST_CH.index("magnetics_tf_current")],
                   cur[g, MAST_CH.index("magnetics_plasma_current")] * 1e3, R, te, ne, gR, gZ,
                   boundary_limiter)
    res = {k: np.full(N, np.nan) for k in ("q95", "q90", "betaN_e", "Rgeo", "a", "blim_active")}
    t0 = time.time()
    with Pool(nproc) as pool:
        for g, o in enumerate(pool.imap(_scal_worker, jobs(), chunksize=64)):
            for k in res:
                res[k][g] = o[k]
            if g % 10000 == 0:
                print(g, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(OUT / "scalars.npz", **res)
    print("q95 finite", np.isfinite(res["q95"]).mean(), "betaN_e finite",
          np.isfinite(res["betaN_e"]).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["1", "2"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default=None, help="output dir name under fec_work")
    ap.add_argument("--nproc", type=int, default=6,
                    help="stage 2: CPU worker processes (default 6; fec/run_final.py and fec/mast_pipeline.py pass "
                         "their --workers when it is not 6)")
    ap.add_argument("--boundary-limiter", action="store_true",
                    help="stage 2: centre-column limiter applied to psi_b for q95/betaN only")
    a = ap.parse_args()
    CONFIG = a.config
    if a.out:
        OUT = WORK / a.out
    stage1(limit=a.limit) if a.stage == "1" else stage2(nproc=a.nproc, boundary_limiter=a.boundary_limiter)
