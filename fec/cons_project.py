"""Test-time projection of an ensemble psi toward target derived scalars (d3dcons, M2/M3).

Per frame, with the ensemble coefficients c = sum_m w_m c_m (members share one PCA basis):
  * search subspace U = top-k eigenvectors of the members' weighted spread covariance in
    coefficient space (label-free; the directions in which the ensemble is uncertain),
    eigenvalues lam;
  * the seven derived scalars s(c) are computed with the scorer's own functionals but the
    TRAIN-only envelope (WORK/ds_b/diii_d_train/env_train.npz) -- never the scorer mask;
  * Jacobian J = ds/d delta by forward finite differences along U (step h_i = fd*sqrt(lam_i));
  * one damped Gauss-Newton step  min_d  sum_j ((s0_j + J_j d - t_j) / sig_j)^2
                                      + sum_i d_i^2 / (tau^2 lam_i),  |d_i| <= cap*sqrt(lam_i)
    toward targets t (nan target = scalar unused), sig_j = scalar scale (consistency is the
    mean of per-scalar R^2, so residuals are normalised by each scalar's spread);
  * psi_out = psi_ens + V U d.
Targets come from a file (N,7) aligned with the frames (e.g. the members' scalar-space mean,
or for the diagnostic oracle the VAL GT derived scalars -- labels, diagnostic only).

  python -B fec/cons_project.py membscal --members <recipe json> --out <npz> [--every 4]
         (per-member derived scalars with env_train, label-free; VAL frames)
  python -B fec/cons_project.py project --members <recipe json> --targets <npz:key> --out <dir>
         [--every 4 --k 6 --tau 1.0 --cap 2.0 --fd 0.5 --use R_axis,...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec import score as SC  # noqa: E402
from fec.features import load_ds  # noqa: E402
from fec.features_b import no_thomson_frames  # noqa: E402
from fec.paths import WORK  # noqa: E402

LS = SC.LS
NAMES = list(LS.CONS_SCALARS)
VAL_IDX = WORK / "models" / "d3d_b_ix_C2_fp32" / "val_idx.npy"
# per-scalar scale (label-free: std of the baseline ensemble's own derived scalars on VAL,
# fec_work/cons_refine/fr_C2.npz cons_pred); used only to normalise residuals
SIG = np.array([0.0384, 0.0372, 0.0885, 0.2335, 0.2240, 1.142, 0.240])
_ENV = None


def env():
    global _ENV
    if _ENV is None:
        from fec.cons_fast import patch
        patch()                     # bit-identical vectorised helpers (fec/cons_fast.py check)
        e = np.load(WORK / "ds_b" / "diii_d_train" / "env_train.npz")
        mc = e["env"].astype(bool)
        _ENV = (e["grid_R"].astype(np.float64), e["grid_Z"].astype(np.float64), mc,
                mc.astype(np.float64))
    return _ENV


def derive_env(psi2d):
    R, Z, mc, mf = env()
    ex, _ = LS.extract_lcfs_with_sign(psi2d, R, Z, "DIII-D", mc, mf, n_iter=LS.N_ITER,
                                      axis_sign=-1.0)
    v = LS.derive_frame(psi2d, R, Z, "DIII-D", mc, mf, contour=ex, with_li=True, axis_sign=-1.0)
    return np.array([v[n] for n in NAMES], np.float64)


def load_members(recipe):
    r = json.loads(Path(recipe).read_text())
    tags = [m["tag"] for m in r["members"]]
    w = np.array([m["weight"] for m in r["members"]], np.float64); w /= w.sum()
    fb = [f["tag"] for f in r["fallback"]] if r.get("fallback") else []
    return tags, w, fb


def member_weights(tags, w, fb, nt):
    """(M,N) per-frame member weights: recipe weights; fallback members (equal) on no-TS frames."""
    W = np.broadcast_to(w[:, None], (len(tags), len(nt))).copy()
    if fb:
        W[:, nt] = 0.0
        for i, t in enumerate(tags):
            if t in fb:
                W[i, nt] = 1.0 / len(fb)
    return W


def _frames(every):
    idx = np.load(VAL_IDX)
    ds = load_ds("diii_d_train")
    shots = ds["shot"][idx]
    us = np.unique(shots)[::every]
    pos = np.flatnonzero(np.isin(shots, us))
    nt = no_thomson_frames({"ts": ds["ts"][idx[pos]]})
    return idx, pos, shots, nt


def _psi_member(tag, pos):
    return np.load(WORK / "models" / f"{tag}_fp32" / "val_psi.npy", mmap_mode="r")


# ------------------------------------------------------------------ member scalars (label-free)
def _ms_worker(args):
    P = args                                   # (M,T,65,65) float32
    out = np.full(P.shape[:2] + (7,), np.nan)
    for m in range(P.shape[0]):
        for k in range(P.shape[1]):
            out[m, k] = derive_env(P[m, k].astype(np.float64))
    return out


def membscal(recipe, out, every=4, nproc=6):
    tags, w, fb = load_members(recipe)
    idx, pos, shots, nt = _frames(every)
    mm = [_psi_member(t, pos) for t in tags]
    jobs, js = [], []
    for s in np.unique(shots[pos]):
        j = pos[shots[pos] == s]
        jobs.append(np.stack([np.asarray(m[j], np.float32) for m in mm])); js.append(j)
    t0 = time.time()
    with Pool(nproc) as pool:
        res = pool.map(_ms_worker, jobs, chunksize=1)
    S = np.concatenate(res, axis=1)            # (M,N,7)
    W = member_weights(tags, w, fb, nt)
    ok = np.isfinite(S)
    Wb = W[:, :, None] * ok
    mean = np.where(Wb.sum(0) > 0, (np.nan_to_num(S) * Wb).sum(0) / np.maximum(Wb.sum(0), 1e-12),
                    np.nan)
    np.savez(out, pos=np.concatenate(js), scal=S, mean=mean, W=W, tags=np.array(tags))
    print(f"member scalars {S.shape} {time.time() - t0:.0f}s -> {out}", flush=True)


# ------------------------------------------------------------------ projection
def _proj_worker(args):
    (C, W, psi_ens, V, tgt, use, k, tau, cap, fd) = args
    T = psi_ens.shape[0]
    out = psi_ens.copy()
    info = np.zeros((T, 4))
    for f in range(T):
        wf = W[:, f]
        sel = wf > 0
        cm = C[sel, f]; ww = wf[sel] / wf[sel].sum()
        c0 = (ww[:, None] * cm).sum(0)
        D = cm - c0
        Cov = (ww[:, None, None] * D[:, :, None] * D[:, None, :]).sum(0)
        lam, U = np.linalg.eigh(Cov)
        o = np.argsort(lam)[::-1][:k]; lam, U = np.maximum(lam[o], 1e-30), U[:, o]
        t = tgt[f]
        jj = np.flatnonzero(use & np.isfinite(t))
        if len(jj) == 0:
            continue
        base = psi_ens[f].astype(np.float64)
        s0 = derive_env(base)
        jj = jj[np.isfinite(s0[jj])]
        if len(jj) == 0:
            continue
        h = fd * np.sqrt(lam)
        J = np.zeros((len(jj), len(lam)))
        good = np.ones(len(jj), bool)
        for i in range(len(lam)):
            d = (V @ (U[:, i] * h[i])).reshape(65, 65)
            si = derive_env(base + d)
            fin = np.isfinite(si[jj])
            good &= fin
            J[:, i] = np.where(fin, (si[jj] - s0[jj]) / h[i], 0.0)
        jj = jj[good]; J = J[good]
        if len(jj) == 0:
            continue
        A = J / SIG[jj, None]; r = (t[jj] - s0[jj]) / SIG[jj]
        Hm = A.T @ A + np.diag(1.0 / (tau ** 2 * lam))
        dlt = np.linalg.solve(Hm, A.T @ r)
        dlt = np.clip(dlt, -cap * np.sqrt(lam), cap * np.sqrt(lam))
        out[f] = (base + (V @ (U @ dlt)).reshape(65, 65)).astype(np.float32)
        info[f] = [len(jj), float((r ** 2).sum()), float(((r - A @ dlt) ** 2).sum()),
                   float(np.sqrt((dlt ** 2 / lam).sum()))]
    return out, info


def project(recipe, targets, out, every=4, k=6, tau=1.0, cap=2.0, fd=0.5, use=None, nproc=6):
    tags, w, fb = load_members(recipe)
    ck = torch.load(WORK / "models" / tags[0] / "model.pt", weights_only=False, map_location="cpu")
    mu = ck["mean"].double().numpy(); V = ck["V"].double().numpy()
    idx, pos, shots, nt = _frames(every)
    W = member_weights(tags, w, fb, nt)
    tf, key = targets.rsplit(":", 1)
    tz = np.load(tf)
    tpos = tz["pos"]; T = tz[key]
    tmap = {p: i for i, p in enumerate(tpos)}
    tgt = np.stack([T[tmap[p]] for p in pos])
    usem = np.ones(7, bool) if not use else np.isin(NAMES, use.split(","))
    mm = [_psi_member(t, pos) for t in tags]
    jobs, js = [], []
    for s in np.unique(shots[pos]):
        sel = np.flatnonzero(shots[pos] == s); j = pos[sel]
        P = np.stack([np.asarray(m[j], np.float64).reshape(len(j), -1) for m in mm])  # (M,T,4225)
        C = (P - mu) @ V                                                                 # (M,T,128)
        Wj = W[:, sel]
        ens = np.einsum("mt,mtp->tp", Wj, P).reshape(len(j), 65, 65)
        jobs.append((C, Wj, ens, V, tgt[sel], usem, k, tau, cap, fd)); js.append(j)
        del P
    t0 = time.time()
    with Pool(nproc) as pool:
        res = pool.map(_proj_worker, jobs, chunksize=1)
    o = Path(out); o.mkdir(parents=True, exist_ok=True)
    allpos = np.concatenate(js)
    psi_out = np.concatenate([r[0] for r in res]); info = np.concatenate([r[1] for r in res])
    # write a VAL-shaped array (unprojected frames = C2 ensemble) for the frame evaluator
    base = np.load(WORK / "models" / "d3d_b_ix_C2_fp32" / "val_psi.npy", mmap_mode="r")
    full = np.lib.format.open_memmap(o / "val_psi.npy", mode="w+", dtype=np.float32, shape=base.shape)
    full[:] = base[:]
    full[allpos] = psi_out; full.flush()
    np.save(o / "val_scal.npy", np.load(WORK / "models" / "d3d_b_ix_C2_fp32" / "val_scal.npy"))
    np.save(o / "val_idx.npy", idx)
    np.savez(o / "proj_info.npz", pos=allpos, info=info)
    (o / "members.json").write_text(json.dumps({"recipe": str(recipe), "targets": targets,
                                                "every": every, "k": k, "tau": tau, "cap": cap,
                                                "fd": fd, "use": use}))
    print(f"projected {len(allpos)} frames in {time.time() - t0:.0f}s; mean chi2 before "
          f"{info[:, 1].mean():.3f} after(lin) {info[:, 2].mean():.3f} step {info[:, 3].mean():.3f}"
          f" -> {o}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["membscal", "project"])
    ap.add_argument("--members", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--targets"); ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--k", type=int, default=6); ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--cap", type=float, default=2.0); ap.add_argument("--fd", type=float, default=0.5)
    ap.add_argument("--use"); ap.add_argument("--nproc", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "membscal":
        membscal(a.members, a.out, a.every, a.nproc)
    else:
        project(a.members, a.targets, a.out, a.every, a.k, a.tau, a.cap, a.fd, a.use, a.nproc)
