"""DIII-D q95 from predicted psi (contour integral) combined with the NN q95 head.

Study note: docs/research/D3D_Q95_HYBRID_20261008.md (pre-declared variants, gate, VAL result).

Physics q95 P (label-free; inputs = predicted psi, magnetics_bcoil, grid scaffold):
  phi = -psi (DIII-D stored convention: axis = minimum), O-point by the scorer's find_o_point
  restricted to the TRAIN axis box, LCFS by the scorer's bisection (fusion_scoring.lcfs) with the
  TRAIN label-LCFS polar envelope + 4 cm as mask_f (built from TRAIN shots only by `envelope`;
  the scorer's own envelope masks are never read), q95 = fec.eqscalars.q_at(level 0.95) with
  F = mu0 * 144 * |I_bcoil[kA] * 1e3| / (2 pi) at the 0 ms extraction offset.
Variants: H head, P (fallback H), M=(H+P)/2, G = P if |P/H-1|<=TAU else H, GM = M if ... else H
(all pre-declared); G1 = P if P/H-1 <= TAU else H (POST-HOC, one-sided; doc sec. 4).

    python -B fec/q95_hybrid.py envelope                       # TRAIN-only envelope json
    python -B fec/q95_hybrid.py val --model d3d_b_ix_C2_fp32 [--modes H,G] [--no-score]
    python -B fec/q95_hybrid.py apply --src <preds dir> --ds <ds root> --out <dir> --parts-only
    python -B fec/q95_hybrid.py apply --src <preds dir> --ds <ds root> --out <new preds dir> \
        --mode G1 [--parts <dir>/<config>_q95_parts.npz] [--config diii_d_public_test]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import SCORING, TRAIN_WORK, WORK  # noqa: E402

MU0 = 4e-7 * math.pi
TF_TURNS = 144                     # DIII-D toroidal-field coil turns (validated R2 0.9999)
TAU = 0.03                         # agreement gate |P/H-1| (pre-declared from TRAIN dev)
Q_RANGE = (1.0, 40.0)              # P outside -> failure
# TRAIN label axis 0.1-99.9% range +- 5 cm (fec/d3d_physics.build_geometry, TRAIN cal set)
AXIS_BOX = (1.5651086688041687, 1.9231273758411411, -0.2631645709276199, 0.26402235552668574)
R0_D3D = 1.6955
VESSEL_BOX = (1.016, 2.37, -1.37, 1.37)
ENV_MARGIN = 0.04
OUT_DIR = WORK / "q95h"
ENV_FILE = Path(__file__).resolve().parent / "q95_hybrid_envelope.json"
MODES = ("H", "P", "M", "G", "GM", "G1")   # G1 = POST-HOC one-sided gate (see doc sec. 4)


# ----------------------------------------------------------------------------- envelope
def build_envelope(every=4, margin=ENV_MARGIN, out=ENV_FILE):
    """Polar envelope about (R0, 0) of TRAIN-split label LCFS points (2-degree bins, +-2-bin
    running max) + margin, clipped to the vessel box. TRAIN shots of the frozen split only."""
    info = json.loads((TRAIN_WORK / "ds" / "diii_d_train" / "info.json").read_text())
    pts = []
    for stem, sp in zip(info["order"], info["splits"]):
        if sp != "train":
            continue
        z = np.load(TRAIN_WORK / "shots" / "diii_d_train" / f"{stem}.npz")
        n, lr, lz = z["lcfs_n"], z["lcfs_r"], z["lcfs_z"]
        for k in range(0, len(n), every):
            m = int(n[k])
            if m >= 3:
                pts.append(np.c_[lr[k, :m], lz[k, :m]])
    pts = np.concatenate(pts).astype(np.float64)
    pts = pts[np.isfinite(pts).all(1)]
    th = np.arctan2(pts[:, 1], pts[:, 0] - R0_D3D)
    rr = np.hypot(pts[:, 0] - R0_D3D, pts[:, 1])
    nb = 180
    b = ((th + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    env = np.zeros(nb)
    np.maximum.at(env, b, rr)
    env = np.max(np.stack([np.roll(env, k) for k in range(-2, 3)]), 0)
    thc = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
    R = np.clip(R0_D3D + (env + margin) * np.cos(thc), VESSEL_BOX[0], VESSEL_BOX[1])
    Z = np.clip((env + margin) * np.sin(thc), VESSEL_BOX[2], VESSEL_BOX[3])
    geo = {"polygon": np.c_[R, Z].tolist(), "margin": margin, "every": every,
           "n_points": int(len(pts)), "source": "TRAIN-split label LCFS, fec_work/shots/diii_d_train",
           "R0": R0_D3D, "vessel_box": VESSEL_BOX}
    Path(out).write_text(json.dumps(geo))
    print(json.dumps({k: v for k, v in geo.items() if k != "polygon"}))
    return geo


def _raster(poly, gR, gZ):
    from matplotlib.path import Path as MPath
    RR, ZZ = np.meshgrid(gR, gZ)
    inside = MPath(np.asarray(poly)).contains_points(np.c_[RR.ravel(), ZZ.ravel()])
    return inside.reshape(RR.shape).astype(np.float64)


# ----------------------------------------------------------------------------- physics q95
_W = {}


def _init(psi_path, gR, gZ, env_poly):
    sys.path.insert(0, str(SCORING))
    _W["psi"] = np.load(psi_path, mmap_mode="r")
    _W["gR"], _W["gZ"] = np.asarray(gR), np.asarray(gZ)
    _W["mask_f"] = _raster(env_poly, _W["gR"], _W["gZ"])
    RR, ZZ = np.meshgrid(_W["gR"], _W["gZ"])
    a = AXIS_BOX
    _W["axmask"] = (RR >= a[0]) & (RR <= a[1]) & (ZZ >= a[2]) & (ZZ <= a[3])


def q95_one(phi, gR, gZ, F, mask_f, axmask):
    """(q95, R_axis, Z_axis, psi_ax - psi_b, LCFS Rmin) for phi with the axis at a maximum."""
    from scipy.interpolate import RectBivariateSpline
    from lcfs import extract_lcfs
    from o_point import find_o_point
    from fec.eqscalars import q_at
    nan5 = (np.nan,) * 5
    try:
        c = extract_lcfs(phi, gR, gZ, "MAST", axmask, mask_f, axis_sign=1.0)
        if c is None:
            return nan5
        sp = RectBivariateSpline(gZ, gR, phi)
        Ra, Za, _, _, _ = find_o_point(phi, gR, gZ, axmask)
        pax = float(sp(Za, Ra)[0, 0])
        pb = float(np.mean(sp(c[:, 1], c[:, 0], grid=False)))
        q = q_at(phi, gR, gZ, pax, pb, Ra, Za, F, 0.95, spline=sp)
        return (float(q), Ra, Za, pax - pb, float(c[:, 0].min()))
    except Exception:
        return nan5


def _chunk(args):
    rows, F = args
    out = np.full((len(rows), 5), np.nan)
    for i, (j, f) in enumerate(zip(rows, F)):
        phi = -np.asarray(_W["psi"][j], dtype=np.float64)
        out[i] = q95_one(phi, _W["gR"], _W["gZ"], f, _W["mask_f"], _W["axmask"])
    return out


def bcoil_F(ds_root, rows):
    info = json.loads((Path(ds_root) / "info.json").read_text())
    ch = info["channels"].index("magnetics_bcoil")
    k0 = info["offsets_ms"].index(0.0)
    cur = np.load(Path(ds_root) / "cur.npy", mmap_mode="r")
    bc = np.asarray(cur[np.asarray(rows), ch, k0], dtype=np.float64)
    return MU0 * TF_TURNS * np.abs(bc) * 1e3 / (2 * math.pi), info


def grid(config):
    from fec.geometry import machine_geometry
    g = machine_geometry(config)
    return g["grid_R"], g["grid_Z"]


def physics_q95(psi_path, F, config, nproc=6, chunk=256):
    """P and diagnostics (N,5) for every row of the psi array at psi_path (row i uses F[i])."""
    env = json.loads(ENV_FILE.read_text())["polygon"]
    gR, gZ = grid(config)
    n = len(F)
    jobs = [(np.arange(s, min(s + chunk, n)), F[s:s + chunk]) for s in range(0, n, chunk)]
    t0 = time.time()
    res = []
    with Pool(nproc, initializer=_init, initargs=(str(psi_path), gR, gZ, env)) as pool:
        for i, r in enumerate(pool.imap(_chunk, jobs)):
            res.append(r)
            if i % 20 == 0:
                print(f"  physics q95 {min((i + 1) * chunk, n)}/{n}  {time.time() - t0:.0f}s",
                      flush=True)
    return np.concatenate(res)


def combine(H, P, mode, tau=TAU):
    """Variant q95 from head H and physics P (P failures -> H). Returns (q, used_P mask)."""
    H = np.asarray(H, dtype=np.float64)
    ok = np.isfinite(P) & (P >= Q_RANGE[0]) & (P <= Q_RANGE[1])
    Pf = np.where(ok, P, H)
    agree = ok & (np.abs(Pf / H - 1) <= tau)
    if mode == "H":
        return H.copy(), np.zeros_like(ok)
    if mode == "P":
        return Pf, ok
    if mode == "M":
        return 0.5 * (H + Pf), ok
    if mode == "G":
        return np.where(agree, Pf, H), agree
    if mode == "GM":
        return np.where(agree, 0.5 * (H + Pf), H), agree
    if mode == "G1":   # POST-HOC: fall back to H only when P exceeds H by > tau (TRAIN failure
        lo = ok & (Pf / H - 1 <= tau)   # mode of P is one-sided: EFIT boundary inside separatrix)
        return np.where(lo, Pf, H), lo
    raise ValueError(mode)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- apply (label-free)
def cmd_apply(a):
    src, out = Path(a.src), Path(a.out)
    cfg = a.config
    psi_path = src / f"{cfg}_psi.npy"
    scal = np.load(src / f"{cfg}_scal.npy")
    meta = json.loads((src / f"{cfg}_meta.json").read_text()) if (src / f"{cfg}_meta.json").exists() else {}
    n = len(scal)
    info = json.loads((Path(a.ds) / "info.json").read_text())
    assert info["config"] == cfg and info["N"] == n, (info["config"], info["N"], n)
    F, _ = bcoil_F(a.ds, np.arange(n))
    out.mkdir(parents=True, exist_ok=True)
    pfile = Path(a.parts) if a.parts else out / f"{cfg}_q95_parts.npz"
    if pfile.exists() and not a.recompute:
        z = np.load(pfile)
        assert len(z["P"]) == n and np.array_equal(z["H"], scal[:, 0]) and np.allclose(z["F"], F),             f"parts file {pfile} does not match source scal / ds"
        P = z["P"]
    else:
        pfile = out / f"{cfg}_q95_parts.npz"
        D = physics_q95(psi_path, F, cfg, nproc=a.nproc)
        P = D[:, 0]
        np.savez(pfile, P=P, R_axis=D[:, 1], Z_axis=D[:, 2], dpsi=D[:, 3], lcfs_Rmin=D[:, 4],
                 H=scal[:, 0], F=F)
    if a.parts_only:
        ok = np.isfinite(P) & (P >= Q_RANGE[0]) & (P <= Q_RANGE[1])
        rel = P[ok] / scal[ok, 0] - 1
        summ = {"parts": str(pfile), "n": int(n), "P_ok_frac": float(ok.mean()),
                "frac_|P/H-1|<=tau": float(np.mean(np.abs(rel) <= TAU) * ok.mean()),
                "frac_P/H-1<=tau": float(np.mean(rel <= TAU) * ok.mean()),
                "median_abs_rel_P_minus_H": float(np.median(np.abs(rel)))}
        (out / f"{cfg}_q95_parts_summary.json").write_text(json.dumps(summ, indent=1))
        print(json.dumps(summ, indent=1))
        return
    q, used = combine(scal[:, 0], P, a.mode)
    new = scal.copy()
    new[:, 0] = q.astype(scal.dtype)
    np.save(out / f"{cfg}_scal.npy", new)
    dst_psi = out / f"{cfg}_psi.npy"
    if not dst_psi.exists():
        shutil.copyfile(psi_path, dst_psi)
    ok = np.isfinite(P) & (P >= Q_RANGE[0]) & (P <= Q_RANGE[1])
    meta = dict(meta)
    meta["q95_hybrid"] = {
        "mode": a.mode, "tau": TAU, "source_dir": str(src), "ds": str(a.ds), "parts": str(pfile),
        "source_scal_sha256": sha256(src / f"{cfg}_scal.npy"),
        "envelope_sha256": sha256(ENV_FILE), "axis_box": AXIS_BOX, "tf_turns": TF_TURNS,
        "P_ok_frac": float(ok.mean()), "used_P_frac": float(used.mean()),
        "q95_changed_frac": float(np.mean(new[:, 0] != scal[:, 0])),
        "median_abs_rel_P_minus_H": float(np.nanmedian(np.abs(P[ok] / scal[ok, 0] - 1))),
        "doc": "docs/research/D3D_Q95_HYBRID_20261008.md"}
    (out / f"{cfg}_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["q95_hybrid"], indent=1))


# ----------------------------------------------------------------------------- VAL evaluation
def _phases(ip, shot):
    ph = np.ones(len(ip), dtype=int)       # 0 ramp-up, 1 flat-top, 2 ramp-down
    for s in np.unique(shot):
        j = np.flatnonzero(shot == s)
        a = np.abs(ip[j])
        ft = np.flatnonzero(a >= 0.9 * a.max())
        ph[j[:ft[0]]] = 0
        ph[j[ft[-1] + 1:]] = 2
    return ph


def _gt_boundary_class(args):
    """limited-like flag from VAL ground truth (diagnostic only): EFIT LCFS flux vs the GT psi's
    own mask-free separatrix flux."""
    stem, ks = args
    sys.path.insert(0, str(SCORING))
    from scipy.interpolate import RectBivariateSpline
    from lcfs import extract_lcfs
    z = np.load(TRAIN_WORK / "shots" / "diii_d_train" / f"{stem}.npz")
    gR, gZ = z["grid_R"], z["grid_Z"]
    out = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        try:
            phi = -z["psi"][k].astype(np.float64)
            sp = RectBivariateSpline(gZ, gR, phi)
            m = int(z["lcfs_n"][k])
            pbl = float(np.median(sp(z["lcfs_z"][k, :m], z["lcfs_r"][k, :m], grid=False)))
            pax = float(sp(z["zaxis"][k], z["raxis"][k])[0, 0])
            c = extract_lcfs(phi, gR, gZ, "MAST", None, None, axis_sign=1.0)
            pbf = float(np.mean(sp(c[:, 1], c[:, 0], grid=False)))
            out[i] = (pbl - pbf) / (pax - pbl)
        except Exception:
            pass
    return out


def cmd_val(a):
    from fec import score as SC
    from fec.features import load_ds
    m = TRAIN_WORK / "models" / a.model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = np.load(m / "val_idx.npy")
    sc = np.load(m / "val_scal.npy")
    ds_root = TRAIN_WORK / "ds" / "diii_d_train"
    F, info = bcoil_F(ds_root, idx)
    pfile = OUT_DIR / f"val_{a.model}_P.npz"
    if pfile.exists() and not a.recompute:
        D = np.load(pfile)["D"]
    else:
        D = physics_q95(m / "val_psi.npy", F, "diii_d_train", nproc=a.nproc)
        np.savez(pfile, D=D, idx=idx, F=F)
    P, H = D[:, 0], sc[:, 0].astype(np.float64)
    ds = load_ds("diii_d_train")
    shots = ds["shot"][idx]
    splits = np.array(info["splits"])
    assert set(splits[np.unique(shots)]) == {"val"}
    y = ds["scal"][idx, 0].astype(np.float64)
    r2 = lambda yy, pp: 1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)
    variants = {k: combine(H, P, k) for k in MODES}
    rep = {"model": a.model, "n_frames": int(len(idx)), "n_shots": int(len(np.unique(shots))),
           "P_ok_frac": float(np.mean(np.isfinite(P) & (P >= Q_RANGE[0]) & (P <= Q_RANGE[1]))),
           "q95_r2": {k: float(r2(y, v[0])) for k, v in variants.items()},
           "used_P_frac": {k: float(v[1].mean()) for k, v in variants.items()}}
    old = OUT_DIR / f"val_{a.model}_report.json"   # keep scores of modes run earlier
    rep["S"] = json.loads(old.read_text()).get("S", {}) if old.exists() else {}
    print(json.dumps(rep, indent=1), flush=True)
    # official scorer, psi float32 (as fec_work/scratch/s_eval_fp32psi.py)
    if not a.no_score:
        psi = np.load(m / "val_psi.npy", mmap_mode="r")
        order = ds["info"]["order"]
        keys, gts, preds, rows = [], [], [], []
        for s in np.unique(shots):
            j = np.flatnonzero(shots == s); g = idx[j]; keys.append(order[s]); rows.append(j)
            gts.append({"psi": np.asarray(ds["psi"][g], dtype=np.float32),
                        "q95": ds["scal"][g, 0].astype(np.float64),
                        "betaN": ds["scal"][g, 1].astype(np.float64)})
            preds.append({"psirz": np.asarray(psi[j], dtype=np.float32).astype(np.float64),
                          "q95": None, "betaN": sc[j, 1].astype(np.float64)})
        for k in a.modes.split(","):
            q = variants[k][0]
            for p, j in zip(preds, rows):
                p["q95"] = q[j].astype(np.float64)
            print(f"== scoring {k}", flush=True)
            res = SC.score(keys, gts, preds, nproc=a.nproc)
            rep["S"][k] = {kk: res[kk] for kk in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")}
            rep["S"][k]["r2_q95"] = res["r2_each"]["q95"]
            (OUT_DIR / f"val_{a.model}_report.json").write_text(json.dumps(rep, indent=1, default=float))
        del gts, preds
    # regime breakdown
    t = ds["t_ms"][idx]
    ip = np.asarray(ds["cur"][idx, info["channels"].index("magnetics_plasma_current"),
                              info["offsets_ms"].index(0.0)], dtype=np.float64)
    ph = _phases(ip, shots)
    lfile = OUT_DIR / f"val_{a.model}_limclass.npy"
    if lfile.exists():
        lim = np.load(lfile)
    else:
        fidx = ds["fidx"][idx]
        jobs = [(ds["info"]["order"][s], fidx[shots == s]) for s in np.unique(shots)]
        with Pool(a.nproc) as pool:
            parts = pool.map(_gt_boundary_class, jobs)
        lim = np.full(len(idx), np.nan)
        for s, part in zip(np.unique(shots), parts):
            lim[shots == s] = part
        np.save(lfile, lim)
    regimes = {
        "all": np.ones(len(idx), bool),
        "diverted": lim <= 0.02, "limited_like": lim > 0.02, "lim_class_nan": ~np.isfinite(lim),
        "q<3.5": y < 3.5, "3.5<=q<5": (y >= 3.5) & (y < 5), "5<=q<7": (y >= 5) & (y < 7), "q>=7": y >= 7,
        "ramp_up": ph == 0, "flat_top": ph == 1, "ramp_down": ph == 2,
        "gate_disagree": ~variants["G"][1],
    }
    sse = {k: np.sum((y - v[0]) ** 2) for k, v in variants.items()}
    br = {}
    for name, msk in regimes.items():
        e = {}
        for k, v in variants.items():
            d = (y[msk] - v[0][msk]) ** 2
            e[k] = {"rmse": float(np.sqrt(d.mean())) if msk.any() else None,
                    "sse_share": float(d.sum() / sse["H"])}
        br[name] = {"n": int(msk.sum()), **e}
    rep["regimes"] = br
    rep["frame_time_ms_median_by_phase"] = {str(p): float(np.median(t[ph == p])) for p in range(3) if (ph == p).any()}
    (OUT_DIR / f"val_{a.model}_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("envelope")
    e.add_argument("--every", type=int, default=4)
    v = sub.add_parser("val")
    v.add_argument("--model", default="d3d_b_ix_C2_fp32")
    v.add_argument("--nproc", type=int, default=6)
    v.add_argument("--modes", default="H,P,M,G,GM,G1")
    v.add_argument("--no-score", action="store_true")
    v.add_argument("--recompute", action="store_true")
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--ds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="diii_d_public_test")
    p.add_argument("--mode", required=True, choices=MODES,
                   help="H/P/M/G/GM pre-declared; G1 post-hoc (doc sec. 4). VAL gate: see doc")
    p.add_argument("--parts", default=None, help="reuse a precomputed <config>_q95_parts.npz")
    p.add_argument("--parts-only", action="store_true",
                   help="compute/save the physics parts in --out and stop (no preds dir)")
    p.add_argument("--nproc", type=int, default=6)
    p.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    if a.cmd == "envelope":
        build_envelope(every=a.every)
    elif a.cmd == "val":
        cmd_val(a)
    else:
        cmd_apply(a)


if __name__ == "__main__":
    main()
