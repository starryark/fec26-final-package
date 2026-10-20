"""Off-span psi decoder (H1dec): the Final package stage of `final_A22li_dec` (agent decpkg, 2026-10-14).

Adopted on Dev (EVIDENCE B61: 987055 DIII-D 0.996748 vs T2 0.996671); pre-declarations in the family doc
docs/research/FEC_COMPETITION_20261007.md amendment 26 ("Dev candidate H1dec", the ordering amendment, "H1dec read");
review research/reviews/REVIEW_H1DEC_20261014.md section 8.  This module reproduces the Dev candidate's three scripts
(fec_work/h1dec/h1dec_data.py, h1dec_train.py, h1dec_apply2.py dec) step for step; only the paths are parameters.

The DIII-D members predict psi as mean + V c (c: 128 PCA coefficients; (mean, V) stored in every member's model.pt).
A decoder maps c to the off-span residual r = psi - mean - V c (the part of psi no member can represent):

  basis     (mean, V) of the members' model.pt (all members of the recipe, primary AND fallback, must agree to
            max |delta| <= 1e-6, else the stage refuses); the first primary member's tensors are the basis.
  envelope  d3d_env_m4: polar envelope (about R = 1.7 m, Z = 0) of the label LCFS points of 1,500 TRAIN shots (rng 37,
            every 2nd frame) + 4 cm.  Rebuilt here from TRAIN labels (not shipped) and checked against its SHA-256;
            used only as a loss weight.  The official scorer masks are never opened.
  data      all labelled DIII-D shots of the frozen split (train + val + test = the members' --alldata rows), every 2nd
            frame (sorted by fidx), float32 labels from <IDX_WORK>/shots/diii_d_train/<stem>.npz:
            c = V^T (psi - mean), r = psi - mean - V c (stored fp16 as 1e4 r), per-frame RMS(r), and six LCFS
            weight points; frames with RMS(r) >= 1e-4 (label artefacts) are dropped in training.
  train     one decoder per seed: MLP 128 -> 1536 x 3 -> 4225 (GELU), output projected off-span; loss = weighted pixel
            MSE + 64 x weighted gradient MSE (weights peak on the label LCFS inside the envelope and at the lowest
            and highest label-LCFS points, not located X-points);
            AdamW lr 3e-4 wd 1e-5, OneCycle (pct 0.05), 40 epochs, batch 2048, fp32.  Seeds 40, 41, 42.
  apply     psi_out = float32(psi + mean of the decoders' r_hat(c(psi))) on every frame of a preds dir (the q95-hybrid
            output); every other file of the dir is copied unchanged, so q95 / betaN stay byte-identical.  The li-range
            repair (lirep4) runs AFTER this step (ordering amendment).

    python -B fec/offdec.py basis    --members-recipe R.json --models DIR --out CACHE/member_basis.npz
    python -B fec/offdec.py envelope --out CACHE/d3d_env_m4.npy
    python -B fec/offdec.py data     --cache CACHE [--workers 8]
    python -B fec/offdec.py train    --cache CACHE --seed 40 --out DIR/dec_s40.pt [--epochs 40] [--device cuda]
    python -B fec/offdec.py apply    --src PREDS --out PREDS_dec --config diii_d_public_test --decoders a.pt,b.pt,c.pt
                                     --members-recipe R.json --models DIR [--device cuda] [--expect-basis SHA]
Inputs at inference: the members' own psi only (no label, no scorer mask, no MAST file).
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("data", "envelope"):
    # single-threaded BLAS before numpy loads (h1dec_data.py): the float64 projections are then bit-reproducible on
    # one machine whatever the core count; the pool workers inherit these variables
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import IDX_WORK  # noqa: E402

# ---------------------------------------------------------------------------- frozen recipe (h1dec_*.py)
GRID_R = (0.84, 2.54, 65)
GRID_Z = (-1.6, 1.6, 65)
NPIX = 4225
K = 128
RSCALE = 1e4                  # r stored as fp16(1e4 r)
ARTEFACT_RMS = 1e-4           # frames with off-span label RMS >= this are dropped in training
SPLITS = ("train", "val", "test")
FRAME_STRIDE = 2
HIDDEN = 1536
EPOCHS = 40
BATCH = 2048
LR = 3e-4
WEIGHT_DECAY = 1e-5
PCT_START = 0.05
GRAD_WEIGHT = 64.0
SEEDS = (40, 41, 42)
APPLY_CHUNK = 8192
BASIS_TOL = 1e-6
# envelope d3d_env_m4 (fec_ralph s37_mask.py): 1,500 TRAIN shots, rng 37, 720 polar bins, 9-bin max filter, + 0.04 m
ENV_SEED, ENV_SHOTS, ENV_BINS, ENV_FILTER, ENV_MARGIN, ENV_R0 = 37, 1500, 720, 9, 0.04, 1.7
ENV_SHA256 = "d89f985c4677674e2fdd88159ac8e3766d3586d94dd17f8e8d26459609e60a31"
# SHA-256 of the envelope's 65 x 65 grid mask (env_mask: uint8 bytes, rows Z, columns R, float64 np.linspace grid;
# 2,287 pixels inside), the property the training uses.  Accepted when the polygon file differs only by last-ulp
# floating point (another CPU / libm); checked to belong to the reference polygon in tests/test_fec_offdec.py.
ENV_MASK_SHA256 = "0f1ca77c57f299bfe8e5213ec7fb23a88ed442e54f3c62336bc020f145c4b59a"
# basis_sha256 of the shipped final_A22li members (HF weights e8531871; d3d_li2ftA22_shape_s0 = the Dev-read decoders'
# member_basis.npz); the Dev-read decoders do not record their basis, so the apply step checks the members against this
SHIPPED_BASIS_SHA256 = "3883fa7101b76907eba39cc856d1b3691fc3d925d0330b0354a0108b4a588469"


def sha256(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, p)


# ============================================================================ basis
def recipe_member_tags(recipe_path):
    """(primary member tags, fallback tags) of a fec/predict_d3d_b.py recipe JSON, in file order."""
    r = json.loads(Path(recipe_path).read_text())
    fb = r.get("fallback") or []
    fb = [fb] if isinstance(fb, dict) else fb
    return [m["tag"] for m in r["members"]], [m["tag"] for m in fb]


def basis_sha256(mean32, V32):
    """SHA-256 of the basis as stored in model.pt: float32 mean (4225,) bytes then float32 V (4225, 128) bytes."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mean32, np.float32).tobytes())
    h.update(np.ascontiguousarray(V32, np.float32).tobytes())
    return h.hexdigest()


def _load_basis32(model_pt):
    import torch
    ck = torch.load(str(model_pt), map_location="cpu", weights_only=False, mmap=True)
    m = ck["mean"].detach().cpu().numpy() if hasattr(ck["mean"], "detach") else np.asarray(ck["mean"])
    V = ck["V"].detach().cpu().numpy() if hasattr(ck["V"], "detach") else np.asarray(ck["V"])
    m, V = np.array(m, np.float32).reshape(-1), np.array(V, np.float32)
    del ck
    if m.shape != (NPIX,) or V.shape != (NPIX, K):
        raise ValueError(f"{model_pt}: basis shapes mean {m.shape} V {V.shape}, want ({NPIX},) and ({NPIX}, {K})")
    return m, V


def member_basis(models, tags, tol=BASIS_TOL, log=print):
    """(mean float64, V float64, info): the basis of tags[0], after checking that EVERY tag's model.pt holds the same
    (mean, V) to max |delta| <= tol (ValueError otherwise, listing the offenders).  float64 = exact copies of the
    float32 tensors (the Dev candidate's member_basis.npz)."""
    models = Path(models)
    miss = [t for t in tags if not (models / t / "model.pt").is_file()]
    if miss:
        raise ValueError(f"missing model.pt in {models} for {len(miss)} member(s): {miss[:5]}")
    m0, V0 = _load_basis32(models / tags[0] / "model.pt")
    m64, V64 = m0.astype(np.float64), V0.astype(np.float64)
    dev, bad = {}, []
    for t in tags[1:]:
        m, V = _load_basis32(models / t / "model.pt")
        d = max(float(np.abs(m.astype(np.float64) - m64).max()), float(np.abs(V.astype(np.float64) - V64).max()))
        dev[t] = d
        if not d <= tol:
            bad.append((t, d))
    if bad:
        raise ValueError(f"members do not share one PCA basis (tolerance {tol}): {bad[:5]} "
                         f"({len(bad)} of {len(tags) - 1} differ from {tags[0]})")
    info = dict(source=tags[0], n_members=len(tags), tol=tol, max_delta=max(dev.values()) if dev else 0.0,
                n_bit_identical=int(sum(v == 0.0 for v in dev.values())) + 1,
                basis_sha256=basis_sha256(m0, V0))
    log(f"basis: {len(tags)} members share the basis of {tags[0]} (max |delta| {info['max_delta']:.3g} <= {tol}; "
        f"{info['n_bit_identical']} bit-identical); sha256 {info['basis_sha256'][:12]}")
    return m64, V64, info


def cmd_basis(a):
    prim, fb = recipe_member_tags(a.members_recipe)
    m, V, info = member_basis(a.models, prim + fb, a.tol)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, m=m, V=V)
    write_json(out.with_suffix(".json"), dict(info, members_recipe=str(a.members_recipe), primary=len(prim),
                                              fallback=len(fb)))
    print(f"BASIS_OK {out}", flush=True)


def load_basis_npz(path):
    z = np.load(path)
    m, V = z["m"], z["V"]
    if m.shape != (NPIX,) or V.shape != (NPIX, K) or m.dtype != np.float64 or V.dtype != np.float64:
        raise ValueError(f"{path}: bad basis {m.shape} {m.dtype} {V.shape} {V.dtype}")
    return m, V


# ============================================================================ envelope (s37_mask.py, margin 0.04)
def split_stems(info, splits):
    return [s for s, sp in zip(info["order"], info["splits"]) if sp in splits]


def build_envelope(shots_dir, info):
    """d3d_env_m4 polygon (720, 2) [R, Z] from TRAIN label LCFS points (fec_ralph s37_mask.py, margin 0.04)."""
    from scipy import ndimage
    tr = split_stems(info, ("train",))
    rng = np.random.default_rng(ENV_SEED)
    rmax = np.zeros(ENV_BINS)
    for s in rng.choice(tr, ENV_SHOTS, replace=False):
        z = np.load(Path(shots_dir) / f"{s}.npz", allow_pickle=True)
        n = np.nan_to_num(z["lcfs_n"]).astype(int)
        for k in range(0, len(n), 2):
            lr, lz = z["lcfs_r"][k, :n[k]], z["lcfs_z"][k, :n[k]]
            th = np.arctan2(lz, lr - ENV_R0); r = np.hypot(lr - ENV_R0, lz)
            b = ((th + np.pi) / (2 * np.pi) * ENV_BINS).astype(int) % ENV_BINS
            np.maximum.at(rmax, b, r)
    rm = ndimage.maximum_filter1d(rmax, ENV_FILTER, mode="wrap") + ENV_MARGIN
    th = (np.arange(ENV_BINS) + 0.5) / ENV_BINS * 2 * np.pi - np.pi
    lr, lz = ENV_R0 + rm * np.cos(th), rm * np.sin(th)
    return np.c_[lr, lz]


def env_mask(env):
    """65 x 65 bool grid mask of the envelope polygon on the float64 linspace grid (rows Z, columns R)."""
    from matplotlib.path import Path as MPath
    R, Z = np.linspace(*GRID_R), np.linspace(*GRID_Z)
    ZZ, RR = np.meshgrid(Z, R, indexing="ij")
    return MPath(np.asarray(env)).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(65, 65)


def env_mask_sha(env):
    return hashlib.sha256(env_mask(env).astype(np.uint8).tobytes()).hexdigest()


def check_envelope(path, log=print):
    """'exact' when the file's SHA-256 is the Dev envelope's; 'mask' when only the 65 x 65 grid mask matches
    (ENV_MASK_SHA256; floating-point noise of another CPU); ValueError otherwise."""
    got = sha256(path)
    if got == ENV_SHA256:
        return "exact"
    env = np.load(path)
    msha = env_mask_sha(env)
    if ENV_MASK_SHA256 and msha == ENV_MASK_SHA256:
        log(f"WARNING envelope {path}: sha256 {got[:12]} != {ENV_SHA256[:12]} but its grid mask is the Dev "
            f"envelope's ({msha[:12]}): accepted (floating-point noise of this machine)")
        return "mask"
    raise ValueError(f"envelope {path}: sha256 {got} != {ENV_SHA256} and grid mask sha {msha} != {ENV_MASK_SHA256}")


def cmd_envelope(a):
    t0 = time.time()
    ds = Path(a.ds)
    info = json.loads((ds / "info.json").read_text())
    env = build_envelope(a.shots, info)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, env)
    how = check_envelope(out)
    write_json(out.with_suffix(".json"), dict(sha256=sha256(out), check=how, mask_sha256=env_mask_sha(env),
                                              seconds=round(time.time() - t0, 1)))
    print(f"ENVELOPE_OK {out} ({how}) {time.time() - t0:.0f}s", flush=True)


# ============================================================================ data (h1dec_data.py)
_W = {}


def _init_worker(basis_path, shots_dir):
    m, V = load_basis_npz(basis_path)
    _W.update(M0=m, Q=V, P=Path(shots_dir), R=np.linspace(*GRID_R), Z=np.linspace(*GRID_Z))


def _load_shot(args):
    from scipy.interpolate import RegularGridInterpolator
    stem, fidx = args
    M0, Q, R, Z = _W["M0"], _W["Q"], _W["R"], _W["Z"]
    z = np.load(_W["P"] / f"{stem}.npz")
    psi = z["psi"][fidx].astype(np.float32)
    x = psi.reshape(len(fidx), -1).astype(np.float64) - M0
    c = x @ Q
    r = x - c @ Q.T
    wp = np.full((len(fidx), 6), np.nan, np.float32)
    for j, k in enumerate(fidx):
        n = int(z["lcfs_n"][k])
        if n < 5 or not np.isfinite(z["raxis"][k]):
            continue
        f = RegularGridInterpolator((Z, R), psi[j].astype(np.float64), bounds_error=False, fill_value=None)
        lr = z["lcfs_r"][k, :n]; lz = z["lcfs_z"][k, :n]
        pa = f([[z["zaxis"][k], z["raxis"][k]]])[0]
        pb = f(np.c_[lz, lr]).mean()
        i1, i2 = np.argmax(lz), np.argmin(lz)
        wp[j] = [pa, pb, lr[i1], lz[i1], lr[i2], lz[i2]]
    return c.astype(np.float32), (r * RSCALE).astype(np.float16), np.sqrt((r ** 2).mean(1)), wp


def data_jobs(ds_dir, splits=SPLITS, max_shots=None):
    """[(stem, fidx)] in the cache's row order: shots of `splits` in info.json order, each shot's frames sorted by fidx,
    every FRAME_STRIDE-th."""
    ds = Path(ds_dir)
    info = json.loads((ds / "info.json").read_text())
    sh = np.load(ds / "shot.npy"); fi = np.load(ds / "fidx.npy")
    sel = np.array([i for i, k in enumerate(info["splits"]) if k in splits])
    if max_shots:
        sel = sel[:max_shots]
    order = np.argsort(sh, kind="stable")
    starts = np.searchsorted(sh[order], sel); ends = np.searchsorted(sh[order], sel, side="right")
    jobs = []
    for s, a, b in zip(sel, starts, ends):
        rows = order[a:b]
        rows = rows[np.argsort(fi[rows])][::FRAME_STRIDE]
        jobs.append((info["order"][s], fi[rows]))
    return jobs


def build_data(cache, ds_dir, shots_dir, workers=8, max_shots=None, log=print):
    """tr_c.npy (n,128) f32, tr_r.npy (n,4225) f16 = 1e4 r, tr_rms.npy (n,) f64, tr_wp.npy (n,6) f32 in `cache`
    (member_basis.npz must be there); data_meta.json is written last."""
    from multiprocessing import Pool
    t0 = time.time()
    cache = Path(cache)
    basis = cache / "member_basis.npz"
    load_basis_npz(basis)
    jobs = data_jobs(ds_dir, max_shots=max_shots)
    n = sum(len(j[1]) for j in jobs)
    log(f"shots {len(jobs)} frames {n}")
    (cache / "data_meta.json").unlink(missing_ok=True)
    rm = np.lib.format.open_memmap(cache / "tr_r.npy", mode="w+", dtype=np.float16, shape=(n, NPIX))
    c = np.zeros((n, K), np.float32); rms = np.zeros(n); wp = np.zeros((n, 6), np.float32)
    a = 0
    with Pool(int(workers), initializer=_init_worker, initargs=(str(basis), str(shots_dir))) as pool:
        for k, (ci, ri, si, wi) in enumerate(pool.imap(_load_shot, jobs, chunksize=4)):
            b = a + len(ci); c[a:b] = ci; rm[a:b] = ri; rms[a:b] = si; wp[a:b] = wi; a = b
            if k % 1000 == 0:
                log(f"{k} {a} {time.time() - t0:.0f}s")
    rm.flush(); del rm
    np.save(cache / "tr_c.npy", c); np.save(cache / "tr_rms.npy", rms); np.save(cache / "tr_wp.npy", wp)
    kept = int((rms < ARTEFACT_RMS).sum())
    meta = dict(shots=len(jobs), frames=n, kept=kept, wp_nan_frac=float(np.isnan(wp[:, 0]).mean()),
                median_offspan_rms=float(np.median(rms)), splits=list(SPLITS), stride=FRAME_STRIDE,
                basis_sha256=json.loads((cache / "member_basis.json").read_text()).get("basis_sha256")
                if (cache / "member_basis.json").exists() else None,
                sha256={f: sha256(cache / f) for f in ("tr_c.npy", "tr_rms.npy", "tr_wp.npy")},
                workers=int(workers), max_shots=max_shots, seconds=round(time.time() - t0, 1))
    write_json(cache / "data_meta.json", meta)
    log(f"done {n} kept(rms<{ARTEFACT_RMS:g}) {kept} wp nan {meta['wp_nan_frac']} median off-span rms "
        f"{meta['median_offspan_rms']:.3e} {time.time() - t0:.0f}s")
    return meta


def cmd_data(a):
    build_data(a.cache, a.ds, a.shots, a.workers, a.max_shots, log=lambda s: print(s, flush=True))
    print(f"DATA_OK {a.cache}", flush=True)


# ============================================================================ train (h1dec_train.py)
def make_decoder(Qt, hidden=HIDDEN):
    import torch.nn as nn

    class Dec(nn.Module):
        def __init__(s, h=hidden):
            super().__init__()
            s.f = nn.Sequential(nn.Linear(K, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                nn.Linear(h, NPIX))

        def forward(s, x):
            y = s.f(x)
            return y - (y @ Qt) @ Qt.T
    return Dec()


def train_decoder(cache, seed, out, epochs=EPOCHS, device="cuda", max_frames=None, log=print):
    """One decoder seed, h1dec_train.py step for step (same tensors, same order of random draws).  max_frames (tests):
    only the first max_frames kept frames.  Writes `out` (torch.save; atomic rename) and returns its dict."""
    import torch
    from matplotlib.path import Path as MPath
    cache, out = Path(cache), Path(out)
    torch.manual_seed(seed)
    dev = device
    t0 = time.time()
    M0, Q = load_basis_npz(cache / "member_basis.npz")
    keep = np.load(cache / "tr_rms.npy") < ARTEFACT_RMS
    if max_frames:
        keep &= np.cumsum(keep) <= max_frames
    kidx = np.flatnonzero(keep)
    c = np.load(cache / "tr_c.npy")[keep]
    mu, sd = c.mean(0), c.std(0)
    rmm = np.load(cache / "tr_r.npy", mmap_mode="r")
    n = len(kidx)
    log(f"kept {n} of {len(keep)}")
    Rt = torch.empty((n, NPIX), dtype=torch.float16, device=dev)
    s1 = s2 = 0.0
    for a in range(0, n, 20000):
        blk = np.asarray(rmm[kidx[a:a + 20000]])
        b64 = blk.astype(np.float64); s1 += b64.sum(); s2 += (b64 ** 2).sum()
        Rt[a:a + len(blk)] = torch.from_numpy(blk).to(dev)
    m1 = s1 / (n * NPIX); rs_st = float(np.sqrt(s2 / (n * NPIX) - m1 ** 2))
    rs = rs_st / RSCALE
    log(f"r std {rs:.3e} Wb/rad, upload {time.time() - t0:.0f}s")
    Ct = torch.tensor((c - mu) / sd, device=dev)
    del c
    Qt = torch.tensor(Q, dtype=torch.float32, device=dev)
    M0t = torch.tensor(M0, dtype=torch.float32, device=dev)
    Rg = torch.linspace(*GRID_R, device=dev); Zg = torch.linspace(*GRID_Z, device=dev)
    ZZ, RR = torch.meshgrid(Zg, Rg, indexing="ij"); RR = RR.reshape(-1); ZZ = ZZ.reshape(-1)
    env = MPath(np.load(cache / "d3d_env_m4.npy")).contains_points(np.c_[RR.cpu().numpy(), ZZ.cpu().numpy()])
    env = torch.tensor(env, dtype=torch.float32, device=dev)
    wp = torch.tensor(np.load(cache / "tr_wp.npy")[keep], device=dev)
    sdt = torch.tensor(sd, dtype=torch.float32, device=dev); mut = torch.tensor(mu, dtype=torch.float32, device=dev)

    def rbatch(idx):
        return Rt[idx].float() / rs_st

    def weights(idx):
        pt = M0t + (Ct[idx] * sdt + mut) @ Qt.T + rbatch(idx) * rs
        w6 = wp[idx]
        pn = (pt - w6[:, :1]) / (w6[:, 1:2] - w6[:, :1])
        w = 1 + 5 * torch.exp(-0.5 * ((pn - 1) / 0.05) ** 2) * env
        for j in (2, 4):
            w = w + 3 * torch.exp(-((RR - w6[:, j:j + 1]) ** 2 + (ZZ - w6[:, j + 1:j + 2]) ** 2) / (2 * 0.08 ** 2))
        w = torch.where(torch.isfinite(w6[:, :1]), w, torch.ones_like(w))
        return w / w.mean()

    net = make_decoder(Qt).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bs = BATCH
    spe = n // bs
    if spe < 1:
        raise ValueError(f"{n} kept frames < one batch of {bs}")
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * spe, pct_start=PCT_START)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    last = None
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev, generator=gen)
        tot = 0.0
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            e = net(Ct[idx]) - rbatch(idx)
            w = weights(idx).view(-1, 65, 65); e = e.view(-1, 65, 65)
            dr = e[:, :, 1:] - e[:, :, :-1]; dz = e[:, 1:, :] - e[:, :-1, :]
            wr = 0.5 * (w[:, :, 1:] + w[:, :, :-1]); wz = 0.5 * (w[:, 1:, :] + w[:, :-1, :])
            loss = (w * e ** 2).mean() + GRAD_WEIGHT * ((wr * dr ** 2).mean() + (wz * dz ** 2).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sch.step()
            tot += loss.item()
        last = tot / spe
        if ep % 10 == 9 or ep == epochs - 1:
            log(f"ep {ep + 1} train {last:.4f} {time.time() - t0:.0f}s")
    bjs = cache / "member_basis.json"
    ck = {"state": net.state_dict(), "mu": mu, "sd": sd, "rs": rs, "basis": "member_basis.npz", "seed": seed,
          "epochs": epochs,
          # decpkg additions (h1dec_apply2 reads only state/mu/sd/rs)
          "basis_sha256": json.loads(bjs.read_text())["basis_sha256"] if bjs.exists() else None,
          "recipe": "h1dec", "n_kept": int(n), "max_frames": max_frames, "final_loss": last,
          "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if str(dev).startswith("cuda")
          else None, "seconds": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, out)
    log(f"s{seed} saved {out} {time.time() - t0:.0f}s")
    return ck


def cmd_train(a):
    train_decoder(a.cache, a.seed, a.out, a.epochs, a.device, a.max_frames, log=lambda s: print(s, flush=True))
    print(f"TRAIN_OK {a.out}", flush=True)


# ============================================================================ apply (h1dec_apply2.py dec)
def load_decoders(paths, Qt, device):
    import torch
    decs, info = [], []
    for p in paths:
        k = torch.load(str(p), map_location=device, weights_only=False)
        net = make_decoder(Qt).to(device); net.load_state_dict(k["state"]); net.eval()
        decs.append((net, torch.tensor(k["mu"], device=device), torch.tensor(k["sd"], device=device), float(k["rs"])))
        info.append(dict(path=str(p), sha256=sha256(p), seed=k.get("seed"), epochs=k.get("epochs"),
                         basis_sha256=k.get("basis_sha256")))
    return decs, info


def dec_correction(x64, M0, Q, decs, device):
    """Mean decoder correction (float64, (n, 4225)) for float64 psi rows x64 (n, 4225): c = (x - mean) @ V in
    float64, cast to float32 on `device`; each decoder's r_hat * rs accumulated in float64."""
    import torch
    c = torch.tensor((x64 - M0) @ Q, dtype=torch.float32, device=device)
    acc = np.zeros_like(x64)
    with torch.no_grad():
        for net, mu, sd, rs in decs:
            acc += (net((c - mu) / sd) * rs).double().cpu().numpy()
    acc /= len(decs)
    return acc


def apply_decoders(src, out, config, dec_paths, M0, Q, device="cuda", chunk=APPLY_CHUNK, expect_basis=None,
                   basis_sha=None, log=print):
    """<out>/<config>_psi.npy = float32(psi + mean decoder correction) of <src>/<config>_psi.npy; every other file of
    src copied unchanged.  Writes <out>/<config>_dec_meta.json (checks, statistics) and returns it.  expect_basis:
    the basis SHA-256 a decoder without its own basis_sha256 (the Dev-read decoders) must have been trained in;
    basis_sha: the members' basis SHA-256 (ValueError when a decoder's basis differs)."""
    import torch
    t0 = time.time()
    src, out = Path(src), Path(out)
    Qt = torch.tensor(Q, dtype=torch.float32, device=device)
    decs, dinfo = load_decoders(dec_paths, Qt, device)
    if basis_sha is not None:
        bad = [d["path"] for d in dinfo if (d["basis_sha256"] or expect_basis) != basis_sha]
        if bad:
            raise ValueError(f"decoders {bad} were trained in another basis than the members' ({basis_sha[:12]}; "
                             f"decoder basis_sha256 or the recipe's shipped_basis_sha256 {str(expect_basis)[:12]})")
    sp = src / f"{config}_psi.npy"
    if (out / f"{config}_psi.npy").exists():
        raise SystemExit(f"{out}/{config}_psi.npy exists (never overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.name != f"{config}_psi.npy":
            shutil.copyfile(f, out / f.name)
    psi = np.load(sp, mmap_mode="r")
    tmp = out / f"{config}_psi.part.npy"
    o = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=psi.shape)
    rms, insp_max, insp_ss, nan = [], 0.0, 0.0, 0
    for a in range(0, len(psi), chunk):
        x = np.asarray(psi[a:a + chunk], np.float64).reshape(-1, NPIX)
        acc = dec_correction(x, M0, Q, decs, device)
        y = (x + acc).astype(np.float32)
        nan += int((~np.isfinite(y)).any(1).sum())
        o[a:a + chunk] = y.reshape(-1, 65, 65)
        rms.append(np.sqrt((acc ** 2).mean(1)))
        ip = acc @ Q
        insp_max = max(insp_max, float(np.abs(ip).max())); insp_ss += float((ip ** 2).sum())
    o.flush(); del o
    os.replace(tmp, out / f"{config}_psi.npy")
    rms = np.concatenate(rms)
    n = int(len(rms))
    meta = dict(src=str(src), out=str(out), config=config, frames=n, device=str(device), chunk=int(chunk),
                decoders=dinfo, basis_sha256=basis_sha, nan_frames=nan, nan=bool(nan),
                inspan_max_abs=insp_max, inspan_rms=float(np.sqrt(insp_ss / max(1, n * K))),
                corr_rms_median=float(np.median(rms)), corr_rms_p99=float(np.quantile(rms, 0.99)),
                corr_rms_max=float(rms.max()), copied_unchanged=sorted(f.name for f in src.iterdir()
                                                                        if f.is_file() and f.name != sp.name),
                seconds=round(time.time() - t0, 1))
    write_json(out / f"{config}_dec_meta.json", meta)
    log(f"DEC_APPLY_DONE {out}: {n} frames, corr rms median {meta['corr_rms_median']:.3e} p99 "
        f"{meta['corr_rms_p99']:.3e}, in-span max {insp_max:.2e} rms {meta['inspan_rms']:.2e}, nan {nan}, "
        f"{meta['seconds']:.0f}s")
    return meta


def cmd_apply(a):
    prim, fb = recipe_member_tags(a.members_recipe)
    M0, Q, binfo = member_basis(a.models, prim + fb, a.tol)
    dev = a.device
    meta = apply_decoders(a.src, a.out, a.config, [p for p in a.decoders.split(",") if p], M0, Q, dev, a.chunk,
                          a.expect_basis, binfo["basis_sha256"])
    meta["basis"] = binfo
    write_json(Path(a.out) / f"{a.config}_dec_meta.json", meta)


# ============================================================================ CLI
def get_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("basis")
    b.add_argument("--members-recipe", required=True)
    b.add_argument("--models", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--tol", type=float, default=BASIS_TOL)
    e = sp.add_parser("envelope")
    e.add_argument("--out", required=True)
    e.add_argument("--ds", default=str(IDX_WORK / "ds" / "diii_d_train"))
    e.add_argument("--shots", default=str(IDX_WORK / "shots" / "diii_d_train"))
    d = sp.add_parser("data")
    d.add_argument("--cache", required=True)
    d.add_argument("--ds", default=str(IDX_WORK / "ds" / "diii_d_train"))
    d.add_argument("--shots", default=str(IDX_WORK / "shots" / "diii_d_train"))
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--max-shots", dest="max_shots", type=int, default=None, help="tests: first N shots only")
    t = sp.add_parser("train")
    t.add_argument("--cache", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=EPOCHS)
    t.add_argument("--device", default="cuda")
    t.add_argument("--max-frames", dest="max_frames", type=int, default=None, help="tests: first N kept frames")
    p = sp.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--decoders", required=True, help="comma list of dec_s<seed>.pt")
    p.add_argument("--members-recipe", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=APPLY_CHUNK)
    p.add_argument("--tol", type=float, default=BASIS_TOL)
    p.add_argument("--expect-basis", dest="expect_basis", default=None,
                   help="basis SHA-256 for decoders that do not record theirs (the shipped Dev-read decoders)")
    return ap


def main(argv=None):
    a = get_parser().parse_args(argv)
    {"basis": cmd_basis, "envelope": cmd_envelope, "data": cmd_data, "train": cmd_train,
     "apply": cmd_apply}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
