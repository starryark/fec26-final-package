"""Auxiliary training targets of the Ch1 family "mtask" (family doc amendment 22; agent mtask, 2026-10-13).

Detail doc: docs/archive/fec/D3D_MTASK_20261013.md. The targets are TRAINING-ONLY structures (like the li2ft caches):
they are never model inputs and never used at inference. Each column is computed from the LABEL psi of a frame:

  shape (4): kappa, tri_top, tri_bot, volume = fusion_scoring/derive.py::_shape_from_contour + derive._clamp_to_nan on
      the scorer's own LCFS of the true map. The contour is the li2 cache's (fec/li2_torch.py build: lcfs.extract_lcfs on
      the FLOAT32 per-shot label, reference-builder call, TRAIN-only envelope env_train; the scorer mask is never used),
      cast back to float64. Frames whose li2 cache row is invalid (li outside the scorer clamp; the cache zeroes C there)
      get NaN, i.e. they are masked out of the auxiliary loss.
  jphi (16): flux-surface-averaged toroidal current density <J_phi>(psiN_k), k = 0..15, psiN_k = (k + 0.5) / 16:
      J_phi = -Delta* psi / (mu0 R),  Delta* psi = d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2,
      second-order three-point finite differences on the (Z rows, R columns) grid (non-uniform form; exact for
      quadratics, O(h^2) otherwise; `delstar`), interior nodes only. psi is the float32 label (as stored, no sign flip).
      psiN = (psi - psi_axis) / (psi_bnd - psi_axis): psi_bnd = median of psi bilinearly sampled on the scorer contour C;
      psi_axis = the extremum of phi = AXIS_SIGN * psi (DIII-D: -1) over the nodes inside C, refined by the separable
      three-point parabola through its neighbours (`axis_value`). The flux-surface average at psiN_k is the
      volume-weighted kernel mean over the grid nodes inside C, sum(K J R) / sum(K R) with the triangular (linear-
      interpolation) kernel K = max(0, 1 - |16 psiN - (k + 0.5)|) of half-width 1/16 (dV = 2 pi R dR dZ; by the coarea
      formula a thin-shell volume average IS the flux-surface average <J> = oint J dl/B_p / oint dl/B_p). psiN is
      clipped to [0, 1]; the end points are one-sided. Empty kernels -> NaN (masked). On Solov'ev the 65x65 values
      match the contour-integral FSA to < 0.7% (a hard shell [k/16, (k+1)/16) aliases the node lattice: up to 5%).
      "inside C" = the li2 cache's inside mask (the scorer's even-odd ray cast of the grid nodes).

Files written by `build` (--out DIR): gidx.npy (n,) int64 sorted global ds frame index, aux.npy (n, 20) float32
(NaN = masked), meta.json (names, sources, counts, per-split mean/sd, timings).

  python -B fec/mtask_aux.py build --li2 DIR[,DIR2] --out DIR [--nproc 4] [--limit_shots N]
  python -B fec/mtask_aux.py check --aux DIR --li2 DIR [--n 200]   re-derive random rows with the unvectorised path
Run with FEC_WORK=<idx root> (<FEC_IDX_WORK> locally). The li2 caches name their own splits; this module
reads only the frames they contain (VAL rows only from a VAL-only cache; TEST rows are never requested).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SHAPE_NAMES = ("kappa", "tri_top", "tri_bot", "volume")
NJ = 16
JPHI_NAMES = tuple(f"jphi_{k:02d}" for k in range(NJ))
NAMES = SHAPE_NAMES + JPHI_NAMES
MU0 = 4e-7 * np.pi
AXIS_SIGN = -1.0                     # fusion_scoring.common.AXIS_SIGN["DIII-D"]
HEADS = {"shape": len(SHAPE_NAMES), "shape_jphi": len(NAMES)}


# ----------------------------------------------------------------------------- numerics (pure numpy, tested)
def _fd_coeffs(x):
    """Three-point non-uniform coefficients at interior nodes i = 1..n-2 of x:
    f'(x_i) ~ a1 f_{i-1} + b1 f_i + c1 f_{i+1};  f''(x_i) ~ a2 f_{i-1} + b2 f_i + c2 f_{i+1}."""
    x = np.asarray(x, np.float64)
    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]
    s = hm + hp
    a1 = -hp / (hm * s); b1 = (hp - hm) / (hm * hp); c1 = hm / (hp * s)
    a2 = 2.0 / (hm * s); b2 = -2.0 / (hm * hp); c2 = 2.0 / (hp * s)
    return (a1, b1, c1), (a2, b2, c2)


def delstar(psi, R, Z):
    """Delta* psi = psi_RR - psi_R / R + psi_ZZ for psi (..., nZ, nR) on the grid (Z rows, R columns).
    Three-point second-order differences (non-uniform form); the outermost ring of nodes is NaN."""
    p = np.asarray(psi, np.float64)
    R = np.asarray(R, np.float64); Z = np.asarray(Z, np.float64)
    (a1, b1, c1), (a2, b2, c2) = _fd_coeffs(R)
    (_, _, _), (az, bz, cz) = _fd_coeffs(Z)
    c = p[..., 1:-1, 1:-1]
    l, r = p[..., 1:-1, :-2], p[..., 1:-1, 2:]
    d, u = p[..., :-2, 1:-1], p[..., 2:, 1:-1]
    prr = a2 * l + b2 * c + c2 * r
    pr = a1 * l + b1 * c + c1 * r
    pzz = az[:, None] * d + bz[:, None] * c + cz[:, None] * u
    out = np.full(p.shape, np.nan)
    out[..., 1:-1, 1:-1] = prr - pr / R[1:-1] + pzz
    return out


def jphi(psi, R, Z):
    """J_phi = -Delta* psi / (mu0 R) on the grid (NaN on the outer ring)."""
    return -delstar(psi, R, Z) / (MU0 * np.asarray(R, np.float64))


def bilinear(field, R, Z, rq, zq):
    """fusion_scoring/derive.py::_bilinear (np.interp + floor/clip), copied for a scorer-free unit test."""
    nZ, nR = field.shape
    fr = np.interp(rq, R, np.arange(nR))
    fz = np.interp(zq, Z, np.arange(nZ))
    r0 = np.clip(np.floor(fr).astype(int), 0, nR - 2)
    z0 = np.clip(np.floor(fz).astype(int), 0, nZ - 2)
    tr = fr - r0
    tz = fz - z0
    return (field[z0, r0] * (1 - tr) * (1 - tz) + field[z0, r0 + 1] * tr * (1 - tz)
            + field[z0 + 1, r0] * (1 - tr) * tz + field[z0 + 1, r0 + 1] * tr * tz)


def axis_value(psi, inside, sign=AXIS_SIGN):
    """psi at the extremum of phi = sign * psi over the `inside` nodes, refined by the separable three-point parabola
    (phi_peak = b - (c - a)^2 / (8 (a - 2 b + c)) per direction, applied only where the curvature is negative)."""
    phi = sign * np.asarray(psi, np.float64)
    m = np.where(inside, phi, -np.inf)
    iz, ir = np.unravel_index(int(np.argmax(m)), m.shape)
    b = phi[iz, ir]
    peak = b
    nz, nr = phi.shape
    if 0 < ir < nr - 1:
        a, c = phi[iz, ir - 1], phi[iz, ir + 1]
        den = a - 2 * b + c
        if den < 0:
            peak -= (c - a) ** 2 / (8.0 * den)
    if 0 < iz < nz - 1:
        a, c = phi[iz - 1, ir], phi[iz + 1, ir]
        den = a - 2 * b + c
        if den < 0:
            peak -= (c - a) ** 2 / (8.0 * den)
    return sign * peak


def shell_average(f, w, psin, inside, nbin=NJ):
    """Volume-weighted triangular-kernel means of f over the inside nodes at psiN_k = (k + 0.5) / nbin: node weight
    w * max(0, 1 - |nbin * psiN - (k + 0.5)|) (linear-interpolation split between the two nearest centres; psiN clipped
    to [0, 1]). Nodes with non-finite f or psin are skipped; empty kernels -> NaN. Returns (nbin,)."""
    ok = inside & np.isfinite(f) & np.isfinite(psin)
    x = np.clip(psin[ok], 0.0, 1.0) * nbin - 0.5
    k0 = np.floor(x).astype(np.int64)
    fr = x - k0
    fw, ww = f[ok] * w[ok], w[ok]
    num = np.zeros(nbin + 2); den = np.zeros(nbin + 2)
    for k, c in ((k0, 1.0 - fr), (k0 + 1, fr)):          # bins -1 .. nbin, then dropped
        num += np.bincount(k + 1, weights=fw * c, minlength=nbin + 2)
        den += np.bincount(k + 1, weights=ww * c, minlength=nbin + 2)
    num, den = num[1:nbin + 1], den[1:nbin + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def jphi_profile(psi, inside, C, R, Z, nbin=NJ):
    """(nbin,) <J_phi> shells of one frame (see the module doc). psi (nZ,nR) label, inside (nZ,nR) bool, C (N,2)."""
    p = np.asarray(psi, np.float64)
    R = np.asarray(R, np.float64); Z = np.asarray(Z, np.float64)
    if inside.sum() < 4 or C is None or len(C) < 8:
        return np.full(nbin, np.nan)
    pb = float(np.median(bilinear(p, R, Z, C[:, 0], C[:, 1])))
    pa = float(axis_value(p, inside))
    if not np.isfinite(pa) or not np.isfinite(pb) or pb == pa:
        return np.full(nbin, np.nan)
    psin = (p - pa) / (pb - pa)
    J = jphi(p, R, Z)
    W = np.broadcast_to(R[None, :], p.shape)
    return shell_average(J, W, psin, inside, nbin)


def shape_row(C):
    """kappa, tri_top, tri_bot, volume of one contour via the scorer's own functions (NaN if absent / implausible)."""
    import derive as DV
    if C is None or len(C) < 8:
        return np.full(len(SHAPE_NAMES), np.nan)
    out = {k: np.nan for k in SHAPE_NAMES}
    out.update(DV._shape_from_contour(np.asarray(C, np.float64)))
    out = DV._clamp_to_nan(out)
    return np.array([out[k] for k in SHAPE_NAMES], np.float64)


# ----------------------------------------------------------------------------- cache build
def _scorer_path():
    from fec import score as SC  # noqa: F401  (puts the starter + fusion_scoring on sys.path)


def _unpack(packed):
    return np.unpackbits(np.asarray(packed, np.uint8), axis=-1)[..., :65 * 65].reshape(packed.shape[:-1] + (65, 65)) \
        .astype(bool)


def frame_targets(psi, C, inside_packed, valid, R, Z):
    """(20,) float64 aux row of one frame; NaN everywhere when the li2 row is invalid (C zeroed)."""
    if not valid:
        return np.full(len(NAMES), np.nan)
    C = np.asarray(C, np.float64)
    ins = _unpack(inside_packed)
    return np.concatenate([shape_row(C), jphi_profile(psi, ins, C, R, Z)])


def _worker(args):
    key, fidx, pos, li2_dir, src, R, Z = args
    _scorer_path()
    C = np.load(Path(li2_dir) / "C.npy", mmap_mode="r")
    I = np.load(Path(li2_dir) / "inside.npy", mmap_mode="r")
    V = np.load(Path(li2_dir) / "valid.npy", mmap_mode="r")
    with np.load(Path(src) / f"{key}.npz") as z:
        P = np.asarray(z["psi"][fidx], np.float32)
        gR, gZ = z["grid_R"], z["grid_Z"]
    assert np.allclose(gR, R) and np.allclose(gZ, Z), key
    out = np.empty((len(fidx), len(NAMES)), np.float64)
    for k in range(len(fidx)):
        out[k] = frame_targets(P[k], C[pos[k]], I[pos[k]], bool(V[pos[k]]), R, Z)
    return key, out


def _md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cmd_build(a):
    from multiprocessing import Pool
    from fec.features import load_ds
    from fec.paths import WORK
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])
    e = np.load(WORK / "ds_b" / "diii_d_train" / "env_train.npz")
    R = e["grid_R"].astype(np.float64); Z = e["grid_Z"].astype(np.float64)
    src = str(WORK / "shots" / "diii_d_train")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    parts, gall = [], []
    meta = {"names": list(NAMES), "heads": HEADS, "sources": [], "definition": __doc__.split("Files written")[0].strip(),
            "labels": "float32 per-shot cache WORK/shots/diii_d_train/<stem>.npz", "mu0": MU0, "axis_sign": AXIS_SIGN}
    t0 = time.time()
    for d in a.li2.split(","):
        d = Path(d)
        lm = json.loads((d / "meta.json").read_text())
        if "test" in lm["splits"]:
            raise SystemExit(f"{d}: li2 cache contains TEST rows; use a TRAIN-only or VAL-only cache")
        gidx = np.load(d / "gidx.npy")
        fsplit = splits[ds["shot"][gidx]]
        assert set(np.unique(fsplit)) <= set(lm["splits"]), (d, np.unique(fsplit))
        shots = ds["shot"][gidx]
        us = np.unique(shots)
        if a.limit_shots:
            us = np.sort(np.random.default_rng(0).choice(us, a.limit_shots, replace=False))
        jobs, posd = [], {}
        for s in us:
            j = np.flatnonzero(shots == s)
            key = ds["info"]["order"][int(s)]
            posd[key] = j
            jobs.append((key, ds["fidx"][gidx[j]], j, str(d), src, R, Z))
        keep = np.concatenate([posd[k] for k in posd]) if posd else np.zeros(0, int)
        keep.sort()
        A = np.full((len(gidx), len(NAMES)), np.nan)
        done = 0
        t1 = time.time()
        with Pool(a.nproc) as pool:
            for key, rows in pool.imap_unordered(_worker, jobs, chunksize=4):
                A[posd[key]] = rows
                done += 1
                if done % 500 == 0 or done == len(jobs):
                    el = time.time() - t1
                    print(f"  {d.name}: {done}/{len(jobs)} shots {el:.0f}s eta {el / done * (len(jobs) - done):.0f}s",
                          flush=True)
        parts.append(A[keep]); gall.append(gidx[keep])
        meta["sources"].append({"li2_dir": str(d), "splits": lm["splits"], "n": int(len(keep)), "n_shots": int(len(us)),
                                "li2_built_utc": lm.get("built_utc"), "li2_env_content_md5": lm.get("env_content_md5"),
                                "li2_gidx_md5": _md5_file(d / "gidx.npy"), "seconds": time.time() - t1})
    G = np.concatenate(gall); A = np.concatenate(parts)
    o = np.argsort(G, kind="stable")
    G, A = G[o], A[o].astype(np.float32)
    assert len(np.unique(G)) == len(G), "li2 caches overlap"
    np.save(out / "gidx.npy", G.astype(np.int64))
    np.save(out / "aux.npy", A)
    fs = splits[ds["shot"][G]]
    stats = {}
    for sp in sorted(set(fs)):
        m = fs == sp
        X = A[m].astype(np.float64)
        stats[sp] = {"n": int(m.sum()), "n_nan": np.isnan(X).sum(0).tolist(),
                     "mean": np.nanmean(X, 0).tolist(), "sd": np.nanstd(X, 0).tolist()}
    meta.update({"n": int(len(G)), "stats": stats, "seconds": time.time() - t0, "limit_shots": a.limit_shots,
                 "aux_md5": _md5_file(out / "aux.npy"), "gidx_md5": _md5_file(out / "gidx.npy"),
                 "numpy": np.__version__, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("MTASK_AUX_DONE", json.dumps({"n": meta["n"], "seconds": round(meta["seconds"]),
                                        "nan_frac": {sp: float(np.mean(np.array(v["n_nan"]) / max(v["n"], 1)))
                                                     for sp, v in stats.items()}}), flush=True)


def cmd_check(a):
    """Recompute random rows with the scorer's own unpatched derive path and an independent contour sample."""
    from fec.features import load_ds
    from fec.paths import WORK
    _scorer_path()
    ds = load_ds("diii_d_train")
    G = np.load(Path(a.aux) / "gidx.npy"); A = np.load(Path(a.aux) / "aux.npy")
    d = Path(a.li2)
    gl = np.load(d / "gidx.npy")
    C = np.load(d / "C.npy", mmap_mode="r"); I = np.load(d / "inside.npy", mmap_mode="r")
    V = np.load(d / "valid.npy")
    e = np.load(WORK / "ds_b" / "diii_d_train" / "env_train.npz")
    R = e["grid_R"].astype(np.float64); Z = e["grid_Z"].astype(np.float64)
    rng = np.random.default_rng(3)
    pick = np.sort(rng.choice(len(gl), min(a.n, len(gl)), replace=False))
    worst = 0.0
    for p in pick:
        g = gl[p]
        r = int(np.searchsorted(G, g)); assert G[r] == g
        key = ds["info"]["order"][int(ds["shot"][g])]
        with np.load(WORK / "shots" / "diii_d_train" / f"{key}.npz") as z:
            psi = np.asarray(z["psi"][ds["fidx"][g]], np.float32)
        ref = frame_targets(psi, C[p], I[p], bool(V[p]), R, Z)
        x = A[r].astype(np.float64)
        same = np.isnan(ref) == np.isnan(x)
        assert same.all(), (g, ref, x)
        f = np.isfinite(ref)
        if f.any():
            worst = max(worst, float(np.max(np.abs(ref[f] - x[f]) / (np.abs(ref[f]) + 1e-12))))
    print("MTASK_AUX_CHECK", json.dumps({"n": int(len(pick)), "max_rel_diff_vs_float32_store": worst}), flush=True)
    assert worst < 1e-6


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--li2", required=True, help="li2 cache dir(s), comma-separated (TRAIN-only / VAL-only)")
    b.add_argument("--out", required=True)
    b.add_argument("--nproc", type=int, default=4)
    b.add_argument("--limit_shots", type=int, default=0)
    c = sp.add_parser("check")
    c.add_argument("--aux", required=True)
    c.add_argument("--li2", required=True)
    c.add_argument("--n", type=int, default=200)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
