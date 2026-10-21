"""DIII-D temporal models: per-frame inputs of neighbouring EFIT frames -> PCA(psi) + scalars.

Per-frame inputs are fec.features.d3d_inputs (inputs only) plus EFIT-time spacing features
(dt to previous/next frame of the same shot). Two architectures:

* ``win`` (default): every training sample is one centre frame (random frames from many shots
  per batch, as in fec.train_d3d). A context branch embeds the inputs of frames of the SAME
  shot at fixed index offsets (dense +-1..8, sparse to +-128) together with their time offset
  to the centre (seconds; EFIT spacing varies), either flattened (``--ctx flat``) or mixed by a
  small transformer (``--ctx attn``); the centre inputs plus context feed a residual MLP that
  predicts the train-only PCA coefficients of psi and five scalars (q95, betaN, li, raxis,
  zaxis). Offsets outside the shot are masked (bidirectional, inputs only).
* ``shot``: whole-shot sequences (per-frame MLP encoder -> dilated conv / attention over all
  frames of a shot -> per-frame head). Tried first; with ~30 shots per step it trained ~10x
  slower than per-frame batching (kept for the record).

Whole-shot frozen split from fec.dataset (info.json "splits"); preprocessing (input stats,
PCA basis, target scales) fitted on TRAIN shots only; VAL used only for monitoring/selection;
TEST-shot labels are never read (PCA/target code reads TRAIN and VAL rows only).

Subcommands
  train    --tag d3d_seq_x ...            -> WORK/models/<tag>/{model.pt,val_*.npy,summary.json}
  predict  --tags a[,b] --out <tag> [--v2 d3d_v2 --wv2 0.5] [--config diii_d_public_test]
           -> WORK/preds/<out>/<config>_psi.npy (N,65,65) f32, <config>_scal.npy (N,5)
  blendval --tags a,b --weights 0.5,0.5 --out <tag>
           -> WORK/models/<out>/val_{psi,scal,idx}.npy (weighted average of saved VAL preds)
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
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.features import d3d_inputs, load_ds  # noqa: E402
from fec.paths import WORK  # noqa: E402

DEV = "cuda"
CACHE = WORK / "cache" / "d3d_seq"
NSCAL = 5


# ----------------------------------------------------------------------------- data helpers

def shot_bounds(shot):
    """Contiguous runs of equal shot id -> (starts, lengths)."""
    shot = np.asarray(shot)
    starts = np.flatnonzero(np.r_[True, shot[1:] != shot[:-1]])
    ends = np.r_[starts[1:], len(shot)]
    return starts.astype(np.int64), (ends - starts).astype(np.int64)


def frame_bounds(shot):
    """Per-frame [start, end) of its shot (shots are contiguous runs)."""
    st, ln = shot_bounds(shot)
    rep = np.repeat(np.arange(len(st)), ln)
    return st[rep], (st + ln)[rep]


def time_features(t_ms, shot):
    """EFIT-time spacing features (inputs/scaffolding only): log dt to prev/next frame."""
    t = np.asarray(t_ms, np.float64)
    same = np.asarray(shot)[1:] == np.asarray(shot)[:-1]
    dtp = np.zeros(len(t)); dtn = np.zeros(len(t))
    d = np.diff(t)
    dtp[1:] = np.where(same, d, 0.0)
    dtn[:-1] = np.where(same, d, 0.0)
    hp = (dtp > 0).astype(np.float32); hn = (dtn > 0).astype(np.float32)
    lp = np.where(dtp > 0, np.log(np.maximum(dtp, 1e-3) / 20.0), 0.0)
    ln = np.where(dtn > 0, np.log(np.maximum(dtn, 1e-3) / 20.0), 0.0)
    return np.stack([np.clip(lp, -3, 6), np.clip(ln, -3, 6), hp, hn], 1).astype(np.float32)


def seq_inputs(ds, stats=None):
    X, s, stats = d3d_inputs(ds, stats)
    return np.concatenate([X, time_features(ds["t_ms"], ds["shot"])], 1), s, stats


def pca_basis(psi_mm, idx, K, nsample=300000, seed=0):
    """Same estimator as fec.train_d3d.pca_basis (TRAIN rows only)."""
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(idx, size=min(nsample, len(idx)), replace=False))
    mean = torch.zeros(4225, device=DEV, dtype=torch.float64)
    C = torch.zeros(4225, 4225, device=DEV, dtype=torch.float64)
    n = 0
    for i in range(0, len(sel), 20000):
        x = torch.as_tensor(np.asarray(psi_mm[sel[i:i + 20000]], dtype=np.float32).reshape(-1, 4225),
                            device=DEV, dtype=torch.float64)
        mean += x.sum(0); C += x.T @ x; n += len(x)
    mean /= n
    C = C / n - torch.outer(mean, mean)
    ev, V = torch.linalg.eigh(C)
    del C
    order = torch.argsort(ev, descending=True)
    ev, V = ev[order], V[:, order]
    return mean.float(), V[:, :K].float().contiguous(), ev


def targets(ds, tr, va, K):
    """Train-only PCA basis and projections of TRAIN and VAL rows only (TEST rows stay 0).

    Cached under WORK/cache/d3d_seq keyed by K and a hash of the TRAIN index set.
    Returns mean (4225,), V (4225,K), ev, Y (N,K) float32 numpy (raw coefficients).
    """
    h = hashlib.sha256(np.asarray(tr, np.int64).tobytes()).hexdigest()[:16]
    f = CACHE / f"pca{K}_{h}.pt"
    fy = CACHE / f"Y{K}_{h}.npy"
    if f.exists() and fy.exists():
        ck = torch.load(f, weights_only=False)
        return ck["mean"], ck["V"], ck["ev"], np.load(fy, mmap_mode="r")
    CACHE.mkdir(parents=True, exist_ok=True)
    mean, V, ev = pca_basis(ds["psi"], tr, K)
    N = len(ds["shot"])
    Y = np.zeros((N, K), np.float32)
    rows = np.sort(np.concatenate([tr, va]))
    for i in range(0, len(rows), 50000):
        r = rows[i:i + 50000]
        x = torch.as_tensor(np.asarray(ds["psi"][r], dtype=np.float32).reshape(-1, 4225), device=DEV)
        Y[r] = ((x - mean) @ V).cpu().numpy()
    torch.save({"mean": mean.cpu(), "V": V.cpu(), "ev": ev.cpu(), "ntr": len(tr)}, f)
    tmp = fy.with_suffix(".tmp.npy")
    np.save(tmp, Y)
    tmp.replace(fy)
    return mean.cpu(), V.cpu(), ev.cpu(), Y


# ----------------------------------------------------------------------------- shared blocks

class Block(nn.Module):
    def __init__(self, w, drop=0.0):
        super().__init__()
        self.n = nn.LayerNorm(w)
        self.f = nn.Sequential(nn.Linear(w, w * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(w * 2, w))

    def forward(self, x):
        return x + self.f(self.n(x))


# ----------------------------------------------------------------------------- window model

def make_offsets(dense, sparse):
    o = sorted(set(list(range(1, dense + 1)) + [int(x) for x in str(sparse).split(",") if x]))
    return [-x for x in reversed(o)] + [0] + o


class EncLayer(nn.Module):
    """Pre-norm transformer layer over context tokens with key padding mask."""

    def __init__(self, d, heads=4, drop=0.0):
        super().__init__()
        self.h = heads
        self.n = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(2 * d, d))

    def forward(self, z, valid):
        B, M, d = z.shape
        q, k, v = self.qkv(self.n(z)).view(B, M, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :] > 0.5)
        z = z + self.o(y.transpose(1, 2).reshape(B, M, d))
        return z + self.mlp(z)


def ctx_columns(fin, which):
    """Input columns given to context tokens. Layout of seq_inputs: cn0 (21), diffs (210),
    Thomson values (64), Thomson masks (64), sign Ip, sign bcoil, t, 4 dt features."""
    if which == "all":
        return list(range(fin))
    assert fin == 366, fin
    if which == "lean":        # offset-0 currents, Thomson values+masks, t
        return list(range(21)) + list(range(231, 359)) + [361]
    if which == "cur":         # offset-0 currents and t only
        return list(range(21)) + [361]
    if which == "ts":          # Thomson values+masks and t only
        return list(range(231, 359)) + [361]
    raise ValueError(which)


class WinNet(nn.Module):
    def __init__(self, fin, K, M, w=1536, depth=8, ctx="flat", e=64, dc=256, clayers=2,
                 drop=0.0, nscal=NSCAL, cols="all"):
        super().__init__()
        self.M, self.ci, self.ctx = M, M // 2, ctx
        cc = ctx_columns(fin, cols)
        self.register_buffer("cols", torch.as_tensor(cc, dtype=torch.int64), persistent=False)
        ft = len(cc)
        if ctx == "none":
            cdim = 0
        elif ctx == "flat":
            self.tok = nn.Linear(ft + 2, e)
            self.pos = nn.Parameter(torch.zeros(M, e))
            self.tok2 = nn.Sequential(nn.GELU(), nn.Linear(e, e))
            cdim = M * e
        else:
            self.tok = nn.Linear(ft + 2, dc)
            self.pos = nn.Parameter(torch.zeros(M, dc))
            self.layers = nn.ModuleList([EncLayer(dc, 4, drop) for _ in range(clayers)])
            self.cn = nn.LayerNorm(dc)
            cdim = 2 * dc
        self.inp = nn.Linear(fin + cdim, w)
        self.blocks = nn.Sequential(*[Block(w, drop) for _ in range(depth)])
        self.n = nn.LayerNorm(w)
        self.psi = nn.Linear(w, K)
        self.scal = nn.Linear(w, nscal)

    def forward(self, xc, tf, valid):
        """xc (B,M,F) inputs of window frames, tf (B,M,2) time-offset feats, valid (B,M) float."""
        x0 = xc[:, self.ci].float()
        if self.ctx == "none":
            c = None
        else:
            tok = self.tok(torch.cat([xc[..., self.cols], tf.to(xc.dtype)], -1)) + self.pos
            if self.ctx == "flat":
                c = ((tok + self.tok2(tok)) * valid[..., None]).flatten(1)
            else:
                z = tok
                for layer in self.layers:
                    z = layer(z, valid)
                z = self.cn(z)
                pooled = (z * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
                c = torch.cat([z[:, self.ci], pooled], -1)
        with torch.autocast("cuda", enabled=False):
            h = self.inp(x0 if c is None else torch.cat([x0, c.float()], -1))
        h = self.n(self.blocks(h))
        return self.psi(h), self.scal(h)


class Windows:
    """GPU-side window gathering for centre frames (same-shot neighbours only)."""

    def __init__(self, Xn, T, shot, offsets):
        fs, fe = frame_bounds(shot)
        self.Xn, self.T = Xn, T
        self.fs = torch.as_tensor(fs, device=DEV); self.fe = torch.as_tensor(fe, device=DEV)
        self.off = torch.as_tensor(offsets, device=DEV, dtype=torch.int64)

    def __call__(self, c, tokdrop=0.0):
        j = c[:, None] + self.off[None, :]
        valid = (j >= self.fs[c][:, None]) & (j < self.fe[c][:, None])
        if tokdrop > 0:
            keep = torch.rand(valid.shape, device=DEV) >= tokdrop
            keep[:, len(self.off) // 2] = True
            valid = valid & keep
        j = torch.where(valid, j, c[:, None])
        xc = self.Xn[j]
        dt = (self.T[j] - self.T[c][:, None]) * valid
        tf = torch.stack([dt, torch.sign(dt) * torch.log1p(dt.abs() / 0.02) / 5.0], -1)
        return xc, tf, valid.to(torch.float32)


@torch.no_grad()
def infer_win(net, win, idx, chunk=8192):
    net.eval()
    P, Q = [], []
    for i in range(0, len(idx), chunk):
        xc, tf, valid = win(idx[i:i + chunk])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p, q = net(xc, tf, valid)
        P.append(p.float()); Q.append(q.float())
    net.train()
    return torch.cat(P), torch.cat(Q)


# ----------------------------------------------------------------------------- whole-shot model

class TConv(nn.Module):
    """Pre-norm residual dilated (non-causal) conv over frames + pointwise MLP."""

    def __init__(self, d, dil, k=3, drop=0.0):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, k, padding=dil * (k // 2), dilation=dil)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(2 * d, d))

    def forward(self, z, m, t):
        y = self.conv((self.n(z) * m).transpose(1, 2)).transpose(1, 2)
        z = (z + y) * m
        return (z + self.mlp(z)) * m


class TAttn(nn.Module):
    """Pre-norm self-attention over all frames of a shot with ALiBi-style |dt| bias."""

    def __init__(self, d, heads=8, drop=0.0):
        super().__init__()
        self.h = heads
        self.n = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        sl = [0.0] + [0.25 * 2 ** i for i in range(heads - 1)]
        self.register_buffer("slope", torch.tensor(sl, dtype=torch.float32), persistent=False)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(2 * d, d))

    def forward(self, z, m, t):
        B, L, d = z.shape
        q, k, v = self.qkv(self.n(z)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        dt = (t[:, :, None] - t[:, None, :]).abs()
        bias = -self.slope[None, :, None, None] * dt[:, None]
        bias = bias.masked_fill(m[:, None, None, :, 0] < 0.5, float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        z = (z + self.o(y.transpose(1, 2).reshape(B, L, d))) * m
        return (z + self.mlp(z)) * m


class SeqNet(nn.Module):
    def __init__(self, fin, K, nscal=NSCAL, w=1024, enc_depth=4, d=512, dils=(1, 2, 4, 8, 16, 32),
                 attn=0, head_depth=2, drop=0.0, tdrop=0.0, ksize=3):
        super().__init__()
        self.inp = nn.Linear(fin, w)
        self.enc = nn.Sequential(*[Block(w, drop) for _ in range(enc_depth)])
        self.down = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, d))
        tb = [TConv(d, int(x), ksize, tdrop) for x in dils]
        tb += [TAttn(d, 8, tdrop) for _ in range(attn)]
        self.tblocks = nn.ModuleList(tb)
        self.tn = nn.LayerNorm(d)
        self.up = nn.Linear(d, w)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)   # start as per-frame model
        self.head = nn.Sequential(*[Block(w, drop) for _ in range(head_depth)])
        self.n = nn.LayerNorm(w)
        self.psi = nn.Linear(w, K)
        self.scal = nn.Linear(w, nscal)

    def forward(self, x, m, t):
        with torch.autocast("cuda", enabled=False):
            h = self.inp(x.float())
        h = self.enc(h)
        z = self.down(h) * m
        for b in self.tblocks:
            z = b(z, m, t)
        h = h + self.up(self.tn(z))
        h = self.n(self.head(h))
        return self.psi(h), self.scal(h)


def plan_epoch(starts, lens, budget, rng, crop_p, min_crop, crop_len=0):
    """Shuffle shots, optionally random-crop, bucket by length -> list of (seg_starts, seg_lens).

    crop_len > 0: every shot is cut into consecutive chunks of crop_len frames (random phase),
    so each frame appears once per epoch and a batch holds budget//crop_len different chunks.
    """
    st = starts.copy(); ln = lens.copy()
    if crop_len > 0:
        ss, ll = [], []
        for s0, n in zip(st, ln):
            ph = int(rng.integers(0, crop_len))
            cuts = np.unique(np.clip(np.r_[0, np.arange(crop_len - ph, n, crop_len), n], 0, n))
            ss.append(s0 + cuts[:-1]); ll.append(np.diff(cuts))
        st = np.concatenate(ss); ln = np.concatenate(ll)
    elif crop_p > 0:
        c = (rng.random(len(st)) < crop_p) & (ln > min_crop)
        L = np.array([rng.integers(min_crop, n + 1) for n in ln[c]], np.int64)
        o = np.array([rng.integers(0, n - l + 1) for n, l in zip(ln[c], L)], np.int64)
        st[c] = st[c] + o; ln[c] = L
    p = rng.permutation(len(st))
    st, ln = st[p], ln[p]
    out = []
    for g in range(0, len(st), 512):
        gs, gl = st[g:g + 512], ln[g:g + 512]
        o = np.argsort(-gl, kind="stable")
        gs, gl = gs[o], gl[o]
        i = 0
        while i < len(gs):
            nb = max(1, budget // max(int(gl[i]), 1))
            out.append((gs[i:i + nb], gl[i:i + nb]))
            i += nb
    order = rng.permutation(len(out))
    return [out[k] for k in order]


def gather(seg_st, seg_ln, Xn, T):
    """Batch of contiguous segments -> x (B,L,F), m (B,L,1), t (B,L), idx, mask."""
    st = torch.as_tensor(seg_st, device=DEV); ln = torch.as_tensor(seg_ln, device=DEV)
    L = int(seg_ln.max())
    ar = torch.arange(L, device=DEV)
    mask = ar[None, :] < ln[:, None]
    idx = torch.where(mask, st[:, None] + ar[None, :], st[:, None])
    t = T[idx]
    t = (t - t[:, :1]) * mask
    return Xn[idx], mask[..., None].to(torch.float32), t, idx, mask


@torch.no_grad()
def infer_shot(net, Xn, T, starts, lens, budget=16384):
    net.eval()
    o = np.argsort(-lens, kind="stable")
    st, ln = starts[o], lens[o]
    Ps, Qs, Is = [], [], []
    i = 0
    while i < len(st):
        nb = max(1, budget // int(ln[i]))
        x, m, t, idx, mask = gather(st[i:i + nb], ln[i:i + nb], Xn, T)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p, q = net(x, m, t)
        Ps.append(p.float()[mask]); Qs.append(q.float()[mask]); Is.append(idx[mask])
        i += nb
    net.train()
    I = torch.cat(Is); P = torch.cat(Ps); Q = torch.cat(Qs)
    o = torch.argsort(I)
    return I[o], P[o], Q[o]


# ----------------------------------------------------------------------------- construction

def build_net(fin, K, a):
    if a.get("arch", "shot") == "win":
        M = len(make_offsets(a["dense"], a["sparse"]))
        return WinNet(fin, K, M, a["width"], a["depth"], a["ctx"], a["e"], a["dc"], a["clayers"],
                      a["drop"], NSCAL, a.get("cols", "all"))
    dils = tuple(int(x) for x in str(a["dils"]).split(",") if x != "")
    return SeqNet(fin, K, NSCAL, a["width"], a["enc_depth"], a["d"], dils, a["attn"],
                  a["head_depth"], a["drop"], a["tdrop"], a["ksize"])


class Predictor:
    """Uniform VAL/test inference for either architecture on all frames of `idx` (whole shots)."""

    def __init__(self, net, a, Xn, T, shot):
        self.net, self.a, self.Xn, self.T, self.shot = net, a, Xn, T, shot
        if a.get("arch", "shot") == "win":
            self.win = Windows(Xn, T, shot, make_offsets(a["dense"], a["sparse"]))

    def __call__(self, idx_np, model=None):
        net = model if model is not None else self.net
        idx = torch.as_tensor(idx_np, device=DEV)
        if self.a.get("arch", "shot") == "win":
            return infer_win(net, self.win, idx)
        st, ln = shot_bounds(self.shot[idx_np])
        I, P, Q = infer_shot(net, self.Xn, self.T, idx_np[st], ln)
        assert torch.equal(I, idx)
        return P, Q


# ----------------------------------------------------------------------------- train

def train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = WORK / "models" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])[ds["shot"]]
    tr = np.flatnonzero(splits == "train")
    va = np.flatnonzero(splits == "val")
    X, s, stats = seq_inputs(ds)
    xmu = X[tr].mean(0); xsd = X[tr].std(0) + 1e-4
    Xn = torch.as_tensor(((X - xmu) / xsd).astype(np.float16), device=DEV)
    del X
    T = torch.as_tensor(ds["t_ms"] / 1000.0, device=DEV, dtype=torch.float32)
    print("inputs", tuple(Xn.shape), f"{time.time() - t0:.0f}s", flush=True)
    mean, V, ev, Ynp = targets(ds, tr, va, a.K)
    tot = ev.clamp_min(0).sum().item()
    ysd = math.sqrt(tot / 4225)
    Y = torch.empty((len(Ynp), a.K), device=DEV, dtype=torch.float16 if a.yhalf else torch.float32)
    for i in range(0, len(Ynp), 200000):
        Y[i:i + 200000] = (torch.as_tensor(np.array(Ynp[i:i + 200000])) / ysd).to(Y.dtype)
    del Ynp
    scal_all = np.asarray(ds["scal"], np.float32)
    smu = torch.as_tensor(scal_all[tr].mean(0), device=DEV)
    ssd = torch.as_tensor(scal_all[tr].std(0), device=DEV)
    sc = np.zeros_like(scal_all)
    sc[tr] = scal_all[tr]; sc[va] = scal_all[va]           # TEST rows never copied
    S = (torch.as_tensor(sc, device=DEV) - smu) / ssd
    del scal_all, sc
    print("targets ready", f"{time.time() - t0:.0f}s", flush=True)

    av = vars(a)
    net = build_net(Xn.shape[1], a.K, av).to(DEV)
    print("params", round(sum(p.numel() for p in net.parameters()) / 1e6, 2), "M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    pred = Predictor(net, av, Xn, T, ds["shot"])
    if a.arch == "win":
        nb = len(tr) // a.bs
        plans = None
        steps = a.epochs * nb
        tr_t = torch.as_tensor(tr, device=DEV)
    else:
        st_all, ln_all = shot_bounds(ds["shot"])
        sp_sh = np.array(ds["info"]["splits"])[ds["shot"][st_all]]
        tr_st, tr_ln = st_all[sp_sh == "train"], ln_all[sp_sh == "train"]
        plans = [plan_epoch(tr_st, tr_ln, a.budget, rng, a.crop_p, a.min_crop, a.crop_len)
                 for _ in range(a.epochs)]
        steps = sum(len(p) for p in plans)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=a.warm, final_div_factor=1e3)
    wsc = torch.tensor([1.0, 1.0, 0.3, 0.3, 0.3], device=DEV) * a.wscal
    ema = None
    if a.ema > 0:
        ema = torch.optim.swa_utils.AveragedModel(
            net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema))
    va_t = torch.as_tensor(va, device=DEV)

    def evaluate(model):
        P, Q = pred(va, model)
        e = ((P - Y[va_t].float()) ** 2).sum(1).mean().item() / 4225
        r2 = 1 - ((Q - S[va_t]) ** 2).mean(0) / S[va_t].var(0)
        return e, r2.cpu().numpy(), P, Q

    def batches(ep):
        if plans is not None:
            for seg_st, seg_ln in plans[ep]:
                x, m, t, idx, mask = gather(seg_st, seg_ln, Xn, T)
                if a.noise > 0:
                    x = x + a.noise * torch.randn_like(x)
                yield (x, m, t), idx, mask.float()
        else:
            perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
            for i in range(0, nb * a.bs, a.bs):
                b = perm[i:i + a.bs]
                xc, tf, valid = pred.win(b, a.tokdrop)
                if a.noise > 0:
                    xc = xc + a.noise * torch.randn_like(xc)
                yield (xc, tf, valid), b, None

    for ep in range(a.epochs):
        tl = 0.0; nf = 0.0
        for inp, idx, mf in batches(ep):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p, q = net(*inp)
            ep_ = ((p.float() - Y[idx].float()) ** 2).sum(-1)
            eq_ = (q.float() - S[idx]) ** 2
            if mf is None:
                n = float(len(idx))
                lp = ep_.mean() / 4225 * 1e3
                lq = (eq_.mean(0) * wsc).sum()
            else:
                n = mf.sum()
                lp = (ep_ * mf).sum() / n / 4225 * 1e3
                lq = (((eq_ * mf[..., None]).sum((0, 1)) / n) * wsc).sum()
                n = n.item()
            loss = lp + lq
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            if ema is not None:
                ema.update_parameters(net)
            tl += loss.item() * n; nf += n
        if ep % a.eval_every == a.eval_every - 1 or ep == a.epochs - 1:
            e, r2s, _, _ = evaluate(net)
            msg = (f"ep {ep} loss {tl / nf:.4f} val psiMSEn {e:.3e} r2 q95 {r2s[0]:.4f} "
                   f"bN {r2s[1]:.4f} li {r2s[2]:.4f} R {r2s[3]:.4f} Z {r2s[4]:.4f}")
            if ema is not None and ep >= a.epochs // 2:
                e2, r2e, _, _ = evaluate(ema.module)
                msg += f" | ema {e2:.3e} q95 {r2e[0]:.4f} bN {r2e[1]:.4f} li {r2e[2]:.4f}"
            mem = torch.cuda.max_memory_allocated() / 2 ** 30
            print(msg + f"  {time.time() - t0:.0f}s  peak {mem:.2f}GiB", flush=True)
    gva = None

    def save(model, odir):
        nonlocal gva
        odir.mkdir(parents=True, exist_ok=True)
        ck = {"net": model.state_dict(), "xmu": xmu, "xsd": xsd, "stats": stats, "mean": mean.cpu(),
              "V": V.cpu(), "ysd": ysd, "smu": smu.cpu(), "ssd": ssd.cpu(), "args": av,
              "fin": Xn.shape[1], "K": a.K}
        torch.save(ck, odir / "model.pt")
        e, r2s, P, Q = evaluate(model)
        Vd, md = V.to(DEV), mean.to(DEV)
        psi_va = np.empty((len(va), 4225), np.float32)
        for i in range(0, len(va), 8192):
            psi_va[i:i + 8192] = ((P[i:i + 8192] * ysd) @ Vd.T + md).cpu().numpy()
        sc_va = (Q * ssd + smu).cpu().numpy()
        np.save(odir / "val_psi.npy", psi_va.reshape(-1, 65, 65))
        np.save(odir / "val_scal.npy", sc_va)
        np.save(odir / "val_idx.npy", va)
        if gva is None:   # VAL labels (VAL rows only) for the pooled R2 printout
            gva = np.asarray(ds["psi"][va], dtype=np.float32).reshape(-1, 4225)
        ss_res = 0.0; s1 = 0.0; s2 = 0.0
        for i in range(0, len(va), 8192):
            g = gva[i:i + 8192].astype(np.float64)
            ss_res += ((g - psi_va[i:i + 8192]) ** 2).sum(); s1 += g.sum(); s2 += (g ** 2).sum()
        n = len(va) * 4225
        r2 = 1 - ss_res / (s2 - s1 ** 2 / n)
        print(odir.name, "VAL pooled R2 psi", r2, "psiMSEn", e, "r2 scal", np.round(r2s, 5),
              f"total {time.time() - t0:.0f}s", flush=True)
        (odir / "summary.json").write_text(json.dumps({"val_r2_psi": float(r2), "val_psiMSEn": e,
                                                       "val_r2_scal": [float(v) for v in r2s],
                                                       "train_seconds": time.time() - t0,
                                                       "args": av}))

    save(net, out)
    if ema is not None:   # EMA weights as a sibling model dir <tag>_ema
        save(ema.module, WORK / "models" / f"{a.tag}_ema")


# ----------------------------------------------------------------------------- predict / blend

@torch.no_grad()
def predict_seq(ds, tag):
    """Model `tag` on every frame of ds (inputs only) -> psi (N,4225) f32, scal (N,5)."""
    ck = torch.load(WORK / "models" / tag / "model.pt", weights_only=False)
    a = ck["args"]
    net = build_net(ck["fin"], ck["K"], a).to(DEV)
    net.load_state_dict(ck["net"]); net.eval()
    X, _, _ = seq_inputs(ds, ck["stats"])
    Xn = torch.as_tensor(((X - ck["xmu"]) / ck["xsd"]).astype(np.float16), device=DEV)
    T = torch.as_tensor(ds["t_ms"] / 1000.0, device=DEV, dtype=torch.float32)
    P, Q = Predictor(net, a, Xn, T, ds["shot"])(np.arange(len(Xn)))
    V, mean = ck["V"].to(DEV), ck["mean"].to(DEV)
    psi = np.empty((len(Xn), 4225), np.float32)
    for i in range(0, len(Xn), 8192):
        psi[i:i + 8192] = ((P[i:i + 8192] * ck["ysd"]) @ V.T + mean).cpu().numpy()
    sc = (Q * ck["ssd"].to(DEV) + ck["smu"].to(DEV)).cpu().numpy()
    del Xn, net, P, Q
    torch.cuda.empty_cache()
    return psi, sc


@torch.no_grad()
def predict_v2(ds, tag):
    """Per-frame fec.train_d3d model (e.g. d3d_v2) on every frame, same math as predict_d3d."""
    from fec.train_d3d import Net
    ck = torch.load(WORK / "models" / tag / "model.pt", weights_only=False)
    a = ck["args"]
    X, _, _ = d3d_inputs(ds, ck["stats"])
    net = Net(ck["fin"], a["K"], 5, a["width"], a["depth"], a.get("drop", 0.0)).to(DEV)
    net.load_state_dict(ck["net"]); net.eval()
    mean, V = ck["mean"].to(DEV), ck["V"].to(DEV)
    psi = np.empty((len(X), 4225), np.float32); sc = np.empty((len(X), 5), np.float32)
    for i in range(0, len(X), 16384):
        x = torch.as_tensor((X[i:i + 16384] - ck["xmu"]) / ck["xsd"], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p, q = net(x)
        psi[i:i + 16384] = ((p.float() * ck["ysd"]) @ V.T + mean).cpu().numpy()
        sc[i:i + 16384] = (q.float() * ck["ssd"].to(DEV) + ck["smu"].to(DEV)).cpu().numpy()
    return psi, sc


def predict(a):
    ds = load_ds(a.config)
    tags = [x for x in a.tags.split(",") if x]
    psi = None; sc = None
    for tag in tags:
        p, q = predict_seq(ds, tag)
        psi = p if psi is None else psi + p
        sc = q if sc is None else sc + q
    psi /= len(tags); sc /= len(tags)
    if a.v2:
        p2, q2 = predict_v2(ds, a.v2)
        psi = (1 - a.wv2) * psi + a.wv2 * p2
        sc = (1 - a.wv2) * sc + a.wv2 * q2
    out = WORK / "preds" / a.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{a.config}_psi.npy", psi.reshape(-1, 65, 65).astype(np.float32))
    np.save(out / f"{a.config}_scal.npy", sc.astype(np.float32))
    (out / f"{a.config}_meta.json").write_text(json.dumps(
        {"tags": tags, "v2": a.v2, "wv2": a.wv2 if a.v2 else 0.0, "N": int(len(psi)),
         "finite": bool(np.isfinite(psi).all() and np.isfinite(sc).all())}))
    print("wrote", out, psi.shape, sc.shape, "finite", np.isfinite(psi).all(), flush=True)


def blendval(a):
    tags = a.tags.split(",")
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0 / len(tags)] * len(tags)
    assert len(w) == len(tags) and abs(sum(w) - 1) < 1e-6
    idx0 = None; psi = None; sc = None
    for tag, wi in zip(tags, w):
        m = WORK / "models" / tag
        idx = np.load(m / "val_idx.npy")
        if idx0 is None:
            idx0 = idx
        assert np.array_equal(idx, idx0), tag
        p = np.load(m / "val_psi.npy", mmap_mode="r"); q = np.load(m / "val_scal.npy")
        if psi is None:
            psi = np.zeros(p.shape, np.float32); sc = np.zeros(q.shape, np.float64)
        for i in range(0, len(p), 8192):
            psi[i:i + 8192] += wi * np.asarray(p[i:i + 8192], np.float32)
        sc += wi * q
    out = WORK / "models" / a.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "val_psi.npy", psi)
    np.save(out / "val_scal.npy", sc.astype(np.float32))
    np.save(out / "val_idx.npy", idx0)
    (out / "summary.json").write_text(json.dumps({"blend": tags, "weights": w}))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--tag", default="d3d_seq_v1")
    t.add_argument("--arch", default="win", choices=["win", "shot"])
    t.add_argument("--K", type=int, default=128)
    # win
    t.add_argument("--width", type=int, default=1536)
    t.add_argument("--depth", type=int, default=8)
    t.add_argument("--ctx", default="flat", choices=["flat", "attn", "none"])
    t.add_argument("--dense", type=int, default=8)
    t.add_argument("--sparse", default="10,12,16,20,24,32,40,48,64,96,128")
    t.add_argument("--e", type=int, default=64)
    t.add_argument("--dc", type=int, default=256)
    t.add_argument("--clayers", type=int, default=2)
    t.add_argument("--tokdrop", type=float, default=0.0)
    t.add_argument("--cols", default="all", choices=["all", "lean", "cur", "ts"])
    t.add_argument("--bs", type=int, default=4096)
    # shot
    t.add_argument("--enc_depth", type=int, default=4)
    t.add_argument("--d", type=int, default=512)
    t.add_argument("--dils", default="1,2,4,8,16,32,64")
    t.add_argument("--ksize", type=int, default=3)
    t.add_argument("--attn", type=int, default=0)
    t.add_argument("--head_depth", type=int, default=2)
    t.add_argument("--tdrop", type=float, default=0.0)
    t.add_argument("--budget", type=int, default=8192, help="padded frames per batch")
    t.add_argument("--crop_p", type=float, default=0.5)
    t.add_argument("--min_crop", type=int, default=16)
    t.add_argument("--crop_len", type=int, default=0)
    # common
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--lr", type=float, default=7e-4)
    t.add_argument("--warm", type=float, default=0.05)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--wscal", type=float, default=0.3)
    t.add_argument("--drop", type=float, default=0.0)
    t.add_argument("--noise", type=float, default=0.0)
    t.add_argument("--ema", type=float, default=0.0)
    t.add_argument("--yhalf", action="store_true", help="store PCA targets fp16 on GPU (VRAM)")
    t.add_argument("--eval_every", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("predict")
    p.add_argument("--tags", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="diii_d_public_test")
    p.add_argument("--v2", default="")
    p.add_argument("--wv2", type=float, default=0.5)
    b = sub.add_parser("blendval")
    b.add_argument("--tags", required=True)
    b.add_argument("--weights", default="")
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        assert a.tag.startswith("d3d_seq"), "tags must start with d3d_seq"
        train(a)
    elif a.cmd == "predict":
        assert a.config.startswith("diii_d"), "DIII-D model"
        predict(a)
    else:
        assert a.out.startswith("d3d_seq")
        blendval(a)


if __name__ == "__main__":
    main()
