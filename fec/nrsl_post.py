"""mast_solver2 (DESIGN §8.1, 8.5, 8.7): frozen cascade (tools/ms_cascade.py) of the two MAST stage files, the primary
output base + W·(cascade − base) (shipped tsm30 weights, per shot), amendment-64 label-free vetoes on base /
un-smoothed / primary (fec.xfer_eval._g3_frame numerics + scorer li), and per-class statistics of our own maps.
No label, scorer mask/envelope or MAST score.

    python -B ms2_mast_post.py --fold-dir outm/public [--nproc 64]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
BARS = dict(lcfs=0.99, inside=0.85, iperr=0.03, zfrac=0.01)


def frame_eval(args):
    """fec.xfer_eval._g3_frame (wall-restricted O-point, unrestricted LCFS) + scorer li (derive_frame with_li)."""
    psi, Ip = args
    from fec import nrsl_copies as XE
    out = XE._g3_frame((psi, Ip, True, False))
    out["li"] = np.nan
    out["finite"] = bool(np.isfinite(psi).all())
    try:
        if str(XE.SCORING) not in sys.path:
            sys.path.insert(0, str(XE.SCORING))
        from derive import derive_frame
        from lcfs import extract_lcfs
        g = XE._mast_geo()
        p = np.asarray(psi, np.float64)
        C = extract_lcfs(p, g["gR"], g["gZ"], "MAST", mask_coarse=g["mc"], mask_f=None)
        d = derive_frame(p, g["gR"], g["gZ"], "MAST", mask_coarse=g["mc"], contour=C, with_li=True)
        out["li"] = float(d.get("li", np.nan))
    except Exception:  # noqa: BLE001
        pass
    return out


def run_eval(P, Ip, nproc):
    with Pool(nproc) as pool:
        rows = list(pool.imap(frame_eval, ((np.asarray(P[i], np.float64), float(Ip[i])) for i in range(len(P))),
                              chunksize=64))
    return {k: np.array([r[k] for r in rows], float) for k in rows[0]}


def veto(R, p2):
    found = R["found"] > 0.5
    off, on = np.abs(p2) < 20, np.abs(p2) > 100
    z = np.abs(R["Z_axis"])
    res = dict(lcfs_found=float(found.mean()), inside_wall=float((R["inside"] > 0.5).mean()),
               ip_err_median=float(np.nanmedian(R["ip_err"][found])),
               absZ_gt_0p3=float(np.mean(np.where(np.isfinite(z), z, 0) > 0.3)),
               absZ_gt_0p3_P2off=float(np.mean(np.where(np.isfinite(z[off]), z[off], 0) > 0.3)),
               absZ_gt_0p3_P2on=float(np.mean(np.where(np.isfinite(z[on]), z[on], 0) > 0.3)),
               nonfinite_frames=int((R["finite"] < 0.5).sum()))
    res["pass"] = dict(lcfs=res["lcfs_found"] >= BARS["lcfs"], inside=res["inside_wall"] >= BARS["inside"],
                       iperr=res["ip_err_median"] <= BARS["iperr"], zfrac=res["absZ_gt_0p3"] <= BARS["zfrac"],
                       finite=res["nonfinite_frames"] == 0)
    res["pass_all"] = bool(all(res["pass"].values()))
    return res


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, [5, 50, 95])] if len(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", required=True)
    ap.add_argument("--nproc", type=int, default=64)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the stage / output files (e.g. _cpu)")
    a = ap.parse_args()
    t0 = time.time()
    d = Path(a.fold_dir)
    z = np.load(d / "inputs.npz")
    shot, t_ms, Ip, p2 = z["shot"], z["t_ms"].astype(float), np.maximum(z["Ip"], 1.0), z["p2"]
    base = np.load(d / "base.npy", mmap_mode="r")
    N = base.shape[0]
    if not (d / "base.npz").exists():
        np.savez(d / "base.npz", frames=np.arange(N), psi=np.asarray(base))
    r = subprocess.run([sys.executable, "-B", str(HERE / "nrsl_cascade.py"), str(d / f"casc{a.tag}.npz"), str(d / "base.npz"),
                        str(d / f"stage_L{a.tag}.npz"), str(d / f"stage_P{a.tag}.npz")], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()[-500:], flush=True)
    C = np.load(d / f"casc{a.tag}.npz")
    casc, src = C["psi"], C["src"]
    # primary: base + W (casc - base), per shot (fec.mast_tsm.gaussian_weights, sigma 30 ms)
    from fec.mast_tsm import gaussian_weights, shot_segments
    prim = np.zeros_like(casc)
    starts, ends = shot_segments(shot)
    for s, e in zip(starts, ends):
        w = gaussian_weights(t_ms[s:e], 30.0)
        dlt = casc[s:e].astype(np.float64) - np.asarray(base[s:e], np.float64)
        prim[s:e] = (np.asarray(base[s:e], np.float64) + np.tensordot(w, dlt, axes=1)).astype(np.float32)
    np.save(d / f"primary{a.tag}.npy", prim)
    sha = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in
           (("base", np.asarray(base)), ("casc", casc), ("primary", prim))}
    # jitter diagnostic: RMS consecutive-frame dpsi within shots
    same = np.r_[shot[1:] == shot[:-1]]
    jit = {}
    for k, P in (("base", base), ("casc", casc), ("primary", prim)):
        acc, n = 0.0, 0
        for s0 in range(0, N - 1, 4096):
            s1 = min(N - 1, s0 + 4096)
            dd = (np.asarray(P[s0 + 1:s1 + 1], np.float64) - np.asarray(P[s0:s1], np.float64))[same[s0:s1]]
            acc += float((dd ** 2).sum()); n += dd.size
        jit[k] = float(np.sqrt(acc / max(n, 1)))
    rep = dict(fold=str(d), N=int(N), stage_counts={int(k): int((src == k).sum()) for k in (-1, 0, 1)},
               sha256=sha, jitter_rms=jit)
    print("CASC/PRIM", json.dumps(rep), flush=True)
    if a.skip_eval:
        (d / f"post{a.tag}.json").write_text(json.dumps(rep, indent=1))
        return
    E = {}
    for k, P in (("base", base), ("casc", casc), ("primary", prim)):
        E[k] = run_eval(P, Ip, a.nproc)
        np.savez(d / f"eval_{k}{a.tag}.npz", **E[k])
        rep[f"veto_{k}"] = veto(E[k], p2)
        print(f"VETO {k} {json.dumps(rep[f'veto_{k}'])} {time.time() - t0:.0f}s", flush=True)
    # classes (DESIGN 8.7)
    li = np.load(d / "li.npz")
    filled = li["filled"].astype(bool)
    ipm = z["Ip"] / 1e6
    dip = np.full(N, np.nan)
    for s, e in zip(starts, ends):
        tt, ii = t_ms[s:e], ipm[s:e]
        sm = np.array([ii[np.abs(tt - x) <= 10.0].mean() for x in tt])
        dip[s:e] = np.gradient(sm, tt / 1e3) if e - s > 1 else 0.0
    ter = np.nanpercentile(z["Ip"], [100 / 3, 200 / 3])
    # C27 P2-off runs (|p2u| + |p2l| < 20 kA-turn): parked = returns with the same sign, rev = opposite, end = to shot end
    cols = [str(c) for c in z["coils"]]
    p2u = z["Iext"][:, cols.index("magnetics_p2u_current")] / 1e3
    p2l = z["Iext"][:, cols.index("magnetics_p2l_current")] / 1e3
    off27 = (np.abs(p2u) + np.abs(p2l)) < 20
    parked, rev, endrun = np.zeros(N, bool), np.zeros(N, bool), np.zeros(N, bool)
    for s, e in zip(starts, ends):
        k = s
        while k < e:
            if not off27[k]:
                k += 1
                continue
            j = k
            while j < e and off27[j]:
                j += 1
            sb = np.sign(p2u[k - 1] + p2l[k - 1]) if k > s else 0.0
            if j >= e:
                endrun[k:j] = True
            elif sb != 0 and np.sign(p2u[j] + p2l[j]) == sb:
                parked[k:j] = True
            else:
                rev[k:j] = True
            k = j
    classes = dict(all=np.ones(N, bool), P2off=np.abs(p2) < 20, P2on=np.abs(p2) > 100,
                   P2off_parked_C27=parked, P2off_rev_C27=rev, P2off_end_C27=endrun,
                   P2other=(np.abs(p2) >= 20) & (np.abs(p2) <= 100),
                   rampup=dip > 0.5, flat=np.abs(dip) <= 0.5, rampdown=dip < -0.5,
                   Ip_lo=z["Ip"] < ter[0], Ip_mid=(z["Ip"] >= ter[0]) & (z["Ip"] < ter[1]), Ip_hi=z["Ip"] >= ter[1],
                   Ip_below_100kA=z["Ip"] < 1e5, noThomson_li_filled=filled)
    out = {}
    for name, m in classes.items():
        if not m.any():
            out[name] = dict(n=0)
            continue
        dR = E["primary"]["R_axis"][m] - E["base"]["R_axis"][m]
        dZ = E["primary"]["Z_axis"][m] - E["base"]["Z_axis"][m]
        dk = E["primary"]["kappa"][m] - E["base"]["kappa"][m]
        dl = E["primary"]["li"][m] - E["base"]["li"][m]
        dRu = E["casc"]["R_axis"][m] - E["base"]["R_axis"][m]
        out[name] = dict(n=int(m.sum()), stage_L=float(np.mean(src[m] == 0)), stage_P=float(np.mean(src[m] == 1)),
                         base=float(np.mean(src[m] == -1)),
                         dRax_cm=q(100 * dR), dZax_cm=q(100 * dZ), dkappa=q(dk), dli=q(dl),
                         frac_abs_dRax_gt3cm=float(np.mean(np.abs(np.nan_to_num(dR)) > 0.03)),
                         frac_dRax_pos=float(np.mean(dR[np.isfinite(dR) & (np.abs(dR) > 1e-4)] > 0)) if
                         np.any(np.isfinite(dR) & (np.abs(dR) > 1e-4)) else None,
                         unsmoothed_frac_abs_dRax_gt3cm=float(np.mean(np.abs(np.nan_to_num(dRu)) > 0.03)),
                         veto_primary=veto({k: v[m] for k, v in E["primary"].items()}, p2[m]),
                         veto_base=veto({k: v[m] for k, v in E["base"].items()}, p2[m]))
    rep["classes"] = out
    # amendment 67(a) axis veto: median signed dR_ax / dZ_ax in [-1, +1] cm and share |d| > 3 cm <= 5 %, per fold
    a67 = {}
    for k in ("primary", "casc"):
        row = {}
        for ax in ("R_axis", "Z_axis"):
            dd = E[k][ax] - E["base"][ax]
            f = np.isfinite(dd)
            med = float(np.median(dd[f])) if f.any() else np.nan
            big = float(np.mean(np.abs(dd[f]) > 0.03)) if f.any() else np.nan
            row[ax] = dict(median_cm=100 * med, frac_gt3cm=big, n_nan=int((~f).sum()),
                           pass_=bool(abs(med) <= 0.01 and big <= 0.05))
        row["pass"] = bool(row["R_axis"]["pass_"] and row["Z_axis"]["pass_"])
        a67[k] = row
    rep["axis_veto_a67"] = a67
    rep["seconds"] = time.time() - t0
    (d / f"post{a.tag}.json").write_text(json.dumps(rep, indent=1, default=float))
    print("POST_DONE", json.dumps({k: rep[k] for k in ("stage_counts", "jitter_rms")}),
          json.dumps({k: rep[f"veto_{k}"]["pass_all"] for k in ("base", "casc", "primary")}), flush=True)


if __name__ == "__main__":
    main()
