"""Batched-member trainer for the d3d_b family (compute-light badge, gates C0/C1; agent clight 2026-10-10).

Trains several fec/train_d3d_b.py members in one process with their weights stacked along a leading
member axis (torch.baddbmm), sharing the input/target preparation. Per-member semantics follow
train_d3d_b.py (doc: docs/research/COMPUTE_LIGHT_20261010.md):

  * Members are train_d3d_b argv strings (--member "...", repeatable) or a pod job list (--jobs
    fec/pod/jobs_*.json, shell lines calling fec/train_d3d_b.py); each is parsed by
    train_d3d_b.get_args, so defaults and the saved `args` dict are identical. Members may differ in
    tag, seed, feat (base | nots), wgrad, win, wband, sig, wx, rx, wli and linskip; the schedule
    (epochs, bs, lr, wd, wscal, K, width, depth, drop, ema, noise, trainval/alldata, eval_every)
    must be shared.
  * Init: torch.manual_seed(seed) then build_net() on CPU, exactly as train_d3d_b, copied into the
    stacks. Data order: torch.randperm per epoch on a per-member CPU generator that continues the
    seeded stream after init, i.e. the permutations train_d3d_b draws for that seed. Input noise
    (if any) from a per-member CUDA generator seeded with the member seed.
  * Preprocessing: fits as in train_d3d_b on the run's training rows tr ("TRAIN rows" below: internal
    TRAIN in gate runs, train+val with --trainval, every labelled row with --alldata) (current-normalisation
    stats, input standardisation per feature set, seed-0 PCA basis/projection, target scales), computed once
    and shared (each train_d3d_b run recomputes the same quantities).
  * Loss per member: shape-aware pixel loss (band, inside, blobs at the lowest and highest label-LCFS points
    (the code's "X-point" blobs), gradient term, optional
    wli sample weights) + scalar heads; fused AdamW, OneCycleLR, gradient-norm clipping at 1.0 PER
    MEMBER, optional EMA; bf16 autocast forward with fp32 LayerNorm, as in train_d3d_b.
  * Outputs per member (drop-in for predict_d3d_b.py / eval tooling): WORK/models/<tag>/model.pt
    (NetB state_dict, same checkpoint keys), and unless --trainval: val_psi.npy, val_scal.npy,
    val_idx.npy, summary.json, as train_d3d_b writes them.
Numerics differ only at kernel level (stacked bmm vs per-member GEMMs, reduction orders), so results
are not bit-identical to train_d3d_b. Refused: --tsdrop > 0, --feat ext/hist, mixed pixel/non-pixel
losses in one run, existing model dirs (unless --overwrite).

Sobolev / B_p supervision (agent mscale 2026-10-10, docs/research/D3D_MEMBER_SCALE_20261010.md section 3;
member flags --wbp RHO [--pcgrad] [--bp_band 1] [--bp_sig 0.05], shared by all members of a run, pixel
members only; implemented here only): a per-frame normalised poloidal-field loss on the error map
(scorer finite differences, 1/R, interior of the true LCFS + a band around it) as a third objective whose
gradient is weighted to RHO x |g_psi| and, with --pcgrad, projected onto {g . g_psi >= 0, g . g_scal >= 0}
per member (three VJPs per step, train_step_sb). Without --wbp the training path is unchanged.
With --ft_from (li2ft fine-tune) the same flags select arm ASB (family doc amendment 18 a): train_step_ft_sb, whose
g_psi is the li-augmented psi gradient of train_step_ft (docs/archive/fec/D3D_ASB_20261013.md).

Auxiliary training-only heads (Ch1 family mtask, family doc amendment 22; agent mtask 2026-10-13; detail doc
docs/archive/fec/D3D_MTASK_20261013.md; run-level flags --aux_cache DIR --aux_heads shape|shape_jphi [--aux_w 1.0],
default OFF): a separate stacked linear head (aux_w (M,w,n), aux_b (M,n); StackedNetBAux) on the final LayerNorm output
predicts TRAIN-standardised targets of fec/mtask_aux.py (kappa, tri_top, tri_bot, volume [+ 16 <J_phi>(psiN) points])
with a masked squared-error loss (train_step_aux): weight aux_w x wscal per scalar head and aux_w x wscal / 16 per J_phi
point (the profile is one head). Init: nn.Linear's default U(+-1/sqrt(w)) from a dedicated generator seeded
AUX_SEED + seed, drawn after the member's init/data-order RNG capture, so the base init and data order equal the
flags-off run. The heads are dropped at export: model.pt is the unchanged NetB (member_state_dict), so predict_d3d_b /
inference are untouched; aux.json + aux_head.pt are written next to it. Without --aux_heads every code path is the
pre-mtask one (train_stack / train_step; tests/test_fec_mtask_bx.py pins it).

Usage (FEC_WORK=<idx root>; wrap in fec/gpu_slot.py):
  python -B fec/train_d3d_bx.py --member "--tag T1 --wband 5 --sig 0.05 --wx 3 --rx 0.08 --wgrad 1 --seed 100" \
      --member "..." [--group_size 2] [--pix_chunk 1] [--x_bf16 1] [--tf32] [--timing_json PATH]
  python -B fec/train_d3d_bx.py --jobs fec/pod/jobs_d3d_axi.json --tag_replace d3d_b60axi=d3d_bxaxi
  --bench_steps N: time N steps of the first stack (after 5 warm-up steps), save nothing, exit.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.train_d3d_b import build_net, get_args as _member_get_args  # noqa: E402


def member_get_args(argv):
    """train_d3d_b.get_args for a member argv, with the Sobolev/B_p flags allowed (implemented here)."""
    return _member_get_args(argv, sb_ok=True)

SHARED = ("epochs", "bs", "lr", "wd", "wscal", "K", "width", "depth", "drop", "ema", "noise",
          "trainval", "alldata", "eval_every", "tsdrop")
NSCAL = 5


# ----------------------------------------------------------------------------- stacked network
class _SBlock(nn.Module):
    def __init__(self, M, w):
        super().__init__()
        z = lambda *s: nn.Parameter(torch.zeros(*s))  # noqa: E731
        self.ln_w, self.ln_b = z(M, w), z(M, w)
        self.w1, self.b1 = z(M, w, 2 * w), z(M, 2 * w)
        self.w2, self.b2 = z(M, 2 * w, w), z(M, w)


def _ln_affine(h, w, b, out_dtype=None):
    """fp32 LayerNorm (eps 1e-5, as nn.LayerNorm under autocast) with a per-member affine:
    h (M,B,n), w/b (M,n). out_dtype: cast of the result (what autocast does before the next matmul)."""
    hf = h.float() if h.dtype in (torch.bfloat16, torch.float16) else h
    z = F.layer_norm(hf, (h.shape[-1],)) * w.unsqueeze(1) + b.unsqueeze(1)
    return z if out_dtype is None else z.to(out_dtype)


_LN_COMPILED = []


def _ln_compiled():
    if not _LN_COMPILED:
        _LN_COMPILED.append(torch.compile(_ln_affine, dynamic=False))
    return _LN_COMPILED[0]


def _amp_dtype(x):
    """Active autocast dtype for x's device, or None."""
    dev = x.device.type
    try:
        on = torch.is_autocast_enabled(dev)
    except TypeError:                            # older torch: CUDA-only signature
        on = torch.is_autocast_enabled() if dev == "cuda" else torch.is_autocast_cpu_enabled()
    if not on:
        return None
    try:
        return torch.get_autocast_dtype(dev)
    except AttributeError:
        return torch.get_autocast_gpu_dtype() if dev == "cuda" else torch.get_autocast_cpu_dtype()


class StackedNetB(nn.Module):
    """M fec.train_d3d_b.NetB members with weights stacked on a leading member axis.

    Linear weights are stored transposed, (M, in, out), for torch.baddbmm (on autocast's bf16 list,
    like nn.Linear); LayerNorm runs in fp32 with a per-member affine (autocast runs layer_norm in
    fp32); psi|scal heads are one merged (M, w, K+5) matmul; the zero-initialised linear skip exists
    for every member when any member has linskip and is multiplied by a per-member 0/1 mask (masked
    members keep exactly zero skip weights: zero grad, zero Adam state, decoupled decay of 0)."""

    def __init__(self, M, fin, K, nscal=NSCAL, w=1536, depth=8, drop=0.0, skip_mask=None):
        super().__init__()
        self.M, self.fin, self.K, self.nscal, self.w, self.depth, self.drop = \
            M, fin, K, nscal, w, depth, drop
        z = lambda *s: nn.Parameter(torch.zeros(*s))  # noqa: E731
        self.inp_w, self.inp_b = z(M, fin, w), z(M, w)
        self.blocks = nn.ModuleList([_SBlock(M, w) for _ in range(depth)])
        self.n_w, self.n_b = z(M, w), z(M, w)
        self.head_w, self.head_b = z(M, w, K + nscal), z(M, K + nscal)
        self.has_skip = skip_mask is not None and any(bool(s) for s in skip_mask)
        if self.has_skip:
            self.skip_w, self.skip_b = z(M, fin, K + nscal), z(M, K + nscal)
            self.register_buffer("skip_mask", torch.tensor([1.0 if s else 0.0 for s in skip_mask])
                                 .view(M, 1, 1))

    @classmethod
    def from_nets(cls, nets):
        n0 = nets[0]
        skip = [n.skip is not None for n in nets]
        sn = cls(len(nets), n0.inp.in_features, n0.K, n0.scal.out_features, n0.inp.out_features,
                 len(n0.blocks), n0.blocks[0].f[2].p, skip).to(n0.inp.weight.dtype)
        with torch.no_grad():
            for m, net in enumerate(nets):
                sn.inp_w[m].copy_(net.inp.weight.T); sn.inp_b[m].copy_(net.inp.bias)
                for blk, nb in zip(sn.blocks, net.blocks):
                    blk.ln_w[m].copy_(nb.n.weight); blk.ln_b[m].copy_(nb.n.bias)
                    blk.w1[m].copy_(nb.f[0].weight.T); blk.b1[m].copy_(nb.f[0].bias)
                    blk.w2[m].copy_(nb.f[3].weight.T); blk.b2[m].copy_(nb.f[3].bias)
                sn.n_w[m].copy_(net.n.weight); sn.n_b[m].copy_(net.n.bias)
                sn.head_w[m].copy_(torch.cat([net.psi.weight, net.scal.weight], 0).T)
                sn.head_b[m].copy_(torch.cat([net.psi.bias, net.scal.bias]))
                if net.skip is not None:
                    sn.skip_w[m].copy_(net.skip.weight.T); sn.skip_b[m].copy_(net.skip.bias)
        return sn

    ln_impl = "eager"

    def forward(self, x):
        """x (M, B, fin) -> p (M, B, K), q (M, B, nscal); dtypes follow autocast like NetB."""
        h = torch.baddbmm(self.inp_b.unsqueeze(1), x, self.inp_w)
        if self.ln_impl == "eager":
            ln, od = _ln_affine, None
        else:                                    # compiled LN + affine (+ the autocast cast)
            ln, od = _ln_compiled(), _amp_dtype(h)
        for blk in self.blocks:
            z = ln(h, blk.ln_w, blk.ln_b, od)
            z = F.gelu(torch.baddbmm(blk.b1.unsqueeze(1), z, blk.w1))
            if self.drop > 0:
                z = F.dropout(z, self.drop, self.training)
            h = h + torch.baddbmm(blk.b2.unsqueeze(1), z, blk.w2)
        z = ln(h, self.n_w, self.n_b, od)
        o = torch.baddbmm(self.head_b.unsqueeze(1), z, self.head_w)
        if self.has_skip:
            s = torch.baddbmm(self.skip_b.unsqueeze(1), x, self.skip_w)
            o = o + s * self.skip_mask.to(s.dtype)
        return o[..., :self.K], o[..., self.K:]

    def member_state_dict(self, m, linskip):
        """NetB state_dict (same keys as train_d3d_b checkpoints) of member m, CPU float32."""
        g = lambda t: t.detach()[m].float().cpu().contiguous()  # noqa: E731
        sd = {"inp.weight": g(self.inp_w).T.contiguous(), "inp.bias": g(self.inp_b)}
        for i, blk in enumerate(self.blocks):
            sd[f"blocks.{i}.n.weight"] = g(blk.ln_w); sd[f"blocks.{i}.n.bias"] = g(blk.ln_b)
            sd[f"blocks.{i}.f.0.weight"] = g(blk.w1).T.contiguous(); sd[f"blocks.{i}.f.0.bias"] = g(blk.b1)
            sd[f"blocks.{i}.f.3.weight"] = g(blk.w2).T.contiguous(); sd[f"blocks.{i}.f.3.bias"] = g(blk.b2)
        sd["n.weight"] = g(self.n_w); sd["n.bias"] = g(self.n_b)
        hw, hb = g(self.head_w), g(self.head_b)
        sd["psi.weight"] = hw[:, :self.K].T.contiguous(); sd["psi.bias"] = hb[:self.K].contiguous()
        sd["scal.weight"] = hw[:, self.K:].T.contiguous(); sd["scal.bias"] = hb[self.K:].contiguous()
        if linskip:
            sd["skip.weight"] = g(self.skip_w).T.contiguous(); sd["skip.bias"] = g(self.skip_b)
        return sd


@torch.no_grad()
def clip_grad_norm_per_member(params, M, max_norm=1.0):
    """torch.nn.utils.clip_grad_norm_(max_norm) applied to each member's slice separately."""
    norms = torch.stack([torch.linalg.vector_norm(p.grad.reshape(M, -1), 2, dim=1)
                         for p in params if p.grad is not None])          # (P, M)
    total = torch.linalg.vector_norm(norms, 2, dim=0)                     # (M,)
    coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(coef.view((M,) + (1,) * (p.grad.ndim - 1)))
    return total


# ----------------------------------------------------------------------------- losses
def _pm(vals, dev):
    """Per-member constant: a Python float when shared (as in train_d3d_b), else (m,1,1) tensor."""
    if all(v == vals[0] for v in vals):
        return float(vals[0])
    return torch.tensor([float(v) for v in vals], dtype=torch.float32, device=dev).view(-1, 1, 1)


@torch.no_grad()
def pixel_weights(yb, b, cfg, ctx):
    """train_d3d_b.pixel_loss weights for m members: yb (m,B,K) targets, b (m,B) row positions.
    cfg: dict of per-member lists (wband, sig, win, wx, rx, wli). Returns wn (m,B,4225)."""
    m, B, K = yb.shape
    dev = yb.device
    wn = torch.addmm(ctx["mean"], yb.reshape(-1, K), ctx["VTs"]).view(m, B, -1)
    bb = ctx["bnd"][b]
    wn.sub_(bb[..., 0:1]).div_(bb[..., 1:2] - bb[..., 0:1])
    inside = (wn < 1.0) if any(v > 0 for v in cfg["win"]) else None
    wn.sub_(1.0).div_(_pm(cfg["sig"], dev)).square_().mul_(-0.5).exp_() \
        .mul_(ctx["maskc"] * _pm(cfg["wband"], dev)).add_(1.0)
    if inside is not None:
        wn.add_(inside.float().mul_(ctx["maskc"] * _pm(cfg["win"], dev)))
    if any(v > 0 for v in cfg["wx"]):
        c = _pm([-0.5 / r ** 2 for r in cfg["rx"]], dev)
        wx = _pm(cfg["wx"], dev)
        bxb = ctx["bx"][b]
        for j in (0, 2):
            d = (ctx["RRg"] - bxb[..., j:j + 1]).square_()
            d.add_((ctx["ZZg"] - bxb[..., j + 1:j + 2]).square_())
            wn.add_(d.mul_(c).exp_().mul_(wx))
    if ctx.get("frame_norm", False):                   # li2ft arm N: each frame's weights average to 1
        wn.div_(wn.mean(dim=2, keepdim=True))
    else:
        wn.div_(wn.mean(dim=(1, 2), keepdim=True))
    if any(v > 0 for v in cfg["wli"]):
        wl = torch.tensor([float(v) for v in cfg["wli"]], device=dev).view(-1, 1)
        sb = 1.0 + wl * (ctx["li"][b] - 1.3).clamp(0.0, 3.0)
        wn.mul_((sb / sb.mean(1, keepdim=True))[..., None])
    return wn


def pixel_loss_members(p, yb, wn, wgrad, VT):
    """Per-member shape-aware pixel loss (m,) of train_d3d_b.pixel_loss; p, yb (m,B,K) fp32."""
    m, B, K = p.shape
    err = ((p - yb).reshape(-1, K) @ VT).view(m, B, -1)
    lp = (wn * err.square()).mean(dim=(1, 2)) * 1e3
    if any(v > 0 for v in wgrad):
        e2 = err.view(m, B, 65, 65)
        w2 = wn.view(m, B, 65, 65)
        dr = (e2[..., :, 1:] - e2[..., :, :-1]) ** 2 * 0.5 * (w2[..., :, 1:] + w2[..., :, :-1])
        dz = (e2[..., 1:, :] - e2[..., :-1, :]) ** 2 * 0.5 * (w2[..., 1:, :] + w2[..., :-1, :])
        wg = torch.tensor([float(v) for v in wgrad], device=p.device)
        lp = lp + wg * (dr.mean(dim=(1, 2, 3)) + dz.mean(dim=(1, 2, 3))) * 1e3 * 64.0
    return lp


def pixel_loss_fused(pc, yb, bb, bxb, sb, mean, VT, VTs, maskc, RRg, ZZg, sig, wband, win, cx, wx,
                     wgrad, has_win: bool, has_wx: bool, has_wgrad: bool, frame_norm: bool = False):
    """Out-of-place form of pixel_weights + pixel_loss_members (identical math, one function, so
    torch.compile can fuse the ~30 elementwise passes over the (m,B,4225) maps and their backward).

    pc, yb (m,B,K) fp32; bb (m,B,2) bnd rows; bxb (m,B,4) X-point rows or None; sb (m,B) wli sample
    weights or None; per-member constants sig, wband, win, cx (= -0.5/rx^2), wx as (m,1,1) tensors,
    wgrad (m,). The weights depend on targets only (no gradient path), as in train_d3d_b."""
    m, B, K = pc.shape
    psin = (torch.addmm(mean, yb.reshape(-1, K), VTs).view(m, B, -1) - bb[..., 0:1])         / (bb[..., 1:2] - bb[..., 0:1])
    wn = torch.exp(((psin - 1.0) / sig).square() * -0.5) * (maskc * wband) + 1.0
    if has_win:
        wn = wn + (psin < 1.0).float() * (maskc * win)
    if has_wx:
        for j in (0, 2):
            d = (RRg - bxb[..., j:j + 1]).square() + (ZZg - bxb[..., j + 1:j + 2]).square()
            wn = wn + torch.exp(d * cx) * wx
    if frame_norm:                                     # li2ft arm N: each frame's weights average to 1
        wn = wn / wn.mean(dim=2, keepdim=True)
    else:
        wn = wn / wn.mean(dim=(1, 2), keepdim=True)
    if sb is not None:
        wn = wn * (sb / sb.mean(1, keepdim=True))[..., None]
    err = ((pc - yb).reshape(-1, K) @ VT).view(m, B, -1)
    lp = (wn * err.square()).mean(dim=(1, 2)) * 1e3
    if has_wgrad:
        e2 = err.view(m, B, 65, 65)
        w2 = wn.view(m, B, 65, 65)
        dr = (e2[..., :, 1:] - e2[..., :, :-1]).square() * 0.5 * (w2[..., :, 1:] + w2[..., :, :-1])
        dz = (e2[..., 1:, :] - e2[..., :-1, :]).square() * 0.5 * (w2[..., 1:, :] + w2[..., :-1, :])
        lp = lp + wgrad * (dr.mean(dim=(1, 2, 3)) + dz.mean(dim=(1, 2, 3))) * 1e3 * 64.0
    return lp


_PIX_COMPILED = {}


def pix_fn(impl):
    """'fused' -> pixel_loss_fused (eager), 'compile' -> torch.compile(pixel_loss_fused)."""
    if impl == "fused":
        return pixel_loss_fused
    if impl not in _PIX_COMPILED:
        _PIX_COMPILED[impl] = torch.compile(pixel_loss_fused, dynamic=False)
    return _PIX_COMPILED[impl]


def _chunk_consts(cfg, c0, c1, ctx, dev):
    """Per-member constant tensors of members c0:c1 for pixel_loss_fused, cached in ctx under a key
    that includes the constants themselves (ctx is shared by all stacks of a run)."""
    sub = {k: v[c0:c1] for k, v in cfg.items()}
    key = ("_pc", c0, c1) + tuple((k, tuple(float(x) for x in sub[k])) for k in sorted(sub))
    if key not in ctx:
        t = lambda v: torch.tensor([float(x) for x in v], dtype=torch.float32, device=dev)  # noqa: E731
        ctx[key] = dict(sig=t(sub["sig"]).view(-1, 1, 1), wband=t(sub["wband"]).view(-1, 1, 1),
                        win=t(sub["win"]).view(-1, 1, 1), wx=t(sub["wx"]).view(-1, 1, 1),
                        cx=t([-0.5 / r ** 2 for r in sub["rx"]]).view(-1, 1, 1), wgrad=t(sub["wgrad"]),
                        wli=t(sub["wli"]).view(-1, 1),
                        has_win=any(v > 0 for v in sub["win"]), has_wx=any(v > 0 for v in sub["wx"]),
                        has_wgrad=any(v > 0 for v in sub["wgrad"]), has_wli=any(v > 0 for v in sub["wli"]))
    return ctx[key]


def psi_loss_and_grad(p32, yb, b, cfg, ctx, pix, chunk, impl="eager"):
    """Per-member psi loss (M,) and its gradient w.r.t. p32 (M,B,K), computed chunk by chunk of
    members so that only one chunk's (B,4225) maps are alive at a time.
    impl: 'eager' (in-place train_d3d_b form), 'fused' or 'compile' (pixel_loss_fused)."""
    M = p32.shape[0]
    lp_all = torch.empty(M, device=p32.device)
    gp = torch.empty_like(p32)
    for c0 in range(0, M, chunk):
        c1 = min(M, c0 + chunk)
        pc = p32[c0:c1].detach().requires_grad_(True)
        yc, bc = yb[c0:c1], b[c0:c1]
        with torch.enable_grad():
            if pix and impl == "eager":
                sub = {k: v[c0:c1] for k, v in cfg.items()}
                wn = pixel_weights(yc, bc, sub, ctx)
                lp = pixel_loss_members(pc, yc, wn, sub["wgrad"], ctx["VT"])
            elif pix:
                k = _chunk_consts(cfg, c0, c1, ctx, p32.device)
                sb = (1.0 + k["wli"] * (ctx["li"][bc] - 1.3).clamp(0.0, 3.0)) if k["has_wli"] else None
                bxb = ctx["bx"][bc] if k["has_wx"] else None
                lp = pix_fn(impl)(pc, yc, ctx["bnd"][bc], bxb, sb, ctx["mean"], ctx["VT"], ctx["VTs"],
                                  ctx["maskc"], ctx["RRg"], ctx["ZZg"], k["sig"], k["wband"], k["win"],
                                  k["cx"], k["wx"], k["wgrad"], k["has_win"], k["has_wx"], k["has_wgrad"],
                                  **({"frame_norm": True} if ctx.get("frame_norm", False) else {}))
            else:
                lp = ((pc - yc) ** 2).sum(-1).mean(-1) / 4225 * 1e3
            g, = torch.autograd.grad(lp.sum(), pc)
        gp[c0:c1] = g
        lp_all[c0:c1] = lp.detach()
    return lp_all, gp


# ----------------------------------------------------------------------------- host batches
class HostBatcher:
    """Per-step (M,B) gathers from pinned host X/Y into double-buffered pinned tensors + async
    H2D copies (--data_gpu 0). The training loop never syncs per step, so the CPU gather of step
    i+1 overlaps the GPU work of step i; a buffer is reused only after its copy event completed."""

    def __init__(self, Xh, Yh, M, B, dev):
        self.Xh, self.Yh, self.dev = Xh, Yh, dev
        self.xbuf = [torch.empty((M, B, Xh.shape[1]), dtype=Xh.dtype).pin_memory() for _ in range(2)]
        self.ybuf = [torch.empty((M, B, Yh.shape[1]), dtype=Yh.dtype).pin_memory() for _ in range(2)]
        self.ev = [None, None]
        self.k = 0

    def get(self, bh):
        k = self.k
        self.k ^= 1
        if self.ev[k] is not None:
            self.ev[k].synchronize()
        idx = bh.reshape(-1)
        torch.index_select(self.Xh, 0, idx, out=self.xbuf[k].view(-1, self.Xh.shape[1]))
        torch.index_select(self.Yh, 0, idx, out=self.ybuf[k].view(-1, self.Yh.shape[1]))
        xb = self.xbuf[k].to(self.dev, non_blocking=True)
        yb = self.ybuf[k].to(self.dev, non_blocking=True)
        self.ev[k] = torch.cuda.Event()
        self.ev[k].record()
        return xb, yb


# ----------------------------------------------------------------------------- one step / EMA
class StackedEMA:
    """torch.optim.swa_utils.AveragedModel + get_ema_multi_avg_fn(decay) (train_d3d_b --ema) for a
    stacked module: the first update copies the weights, later ones lerp by (1 - decay)."""

    def __init__(self, module, decay):
        self.module = copy.deepcopy(module)
        self.decay = decay
        self.n_averaged = 0
        self.params = list(self.module.parameters())

    @torch.no_grad()
    def update(self, params):
        cur = [p.detach() for p in params]
        if self.n_averaged == 0:
            torch._foreach_copy_(self.params, cur)
        else:
            torch._foreach_lerp_(self.params, cur, 1 - self.decay)
        self.n_averaged += 1


def train_step(snet, params, opt, sched, xb, yb, Sb, b, cfg, ctx, wsc, pix, chunk,
               amp_dtype=torch.bfloat16, ema=None, impl="eager"):
    """One train_d3d_b optimisation step for all M stacked members.

    xb (M,B,fin) inputs, yb (M,B,K) PCA targets, Sb (M,B,5) normalised scalars, b (M,B) row
    positions (pixel-loss weights). Loss_m = pixel/PCA loss + scalar loss; gradients are
    per-member (members share no parameters), clipped per member at 1.0; one AdamW step over the
    stacked tensors is the per-member AdamW step (elementwise; per-member state = its slice);
    optional EMA. amp_dtype=None disables autocast (fp32 tests). snet may be a torch.compile'd
    StackedNetB; impl selects the pixel-loss implementation. Returns per-member loss (M,)."""
    M = xb.shape[0]
    with torch.autocast(xb.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        p, q = snet(xb)
    p32 = p.float()
    lp, gp = psi_loss_and_grad(p32, yb.float(), b, cfg, ctx, pix, chunk, impl)
    lq = (((q.float() - Sb) ** 2).mean(1) * wsc).sum(1)
    opt.zero_grad(set_to_none=True)
    torch.autograd.backward([p32, lq.sum()], [gp, None])
    clip_grad_norm_per_member(params, M, 1.0)
    opt.step()
    if sched is not None:
        sched.step()
    if ema is not None:
        ema.update(params)
    return (lp + lq).detach()


# ----------------------------------------------------------------------------- auxiliary heads (mtask, amendment 22)
AUX_SEED = 1_000_003                              # aux-head init generator seed = AUX_SEED + member seed
AUX_JPHI = 16                                     # fec/mtask_aux.NJ


class StackedNetBAux(StackedNetB):
    """StackedNetB + a training-only auxiliary linear head on the final LayerNorm output (aux_w (M,w,n), aux_b (M,n),
    registered after every base parameter). forward returns (p, q, a); the base (p, q) is computed exactly as
    StackedNetB.forward (same ops, same order). member_state_dict (inherited) exports the plain NetB."""

    n_aux = 0

    def init_aux(self, n_aux, seeds):
        dt, dv = self.head_w.dtype, self.head_w.device
        self.n_aux = int(n_aux)
        self.aux_w = nn.Parameter(torch.zeros(self.M, self.w, self.n_aux, dtype=dt, device=dv))
        self.aux_b = nn.Parameter(torch.zeros(self.M, self.n_aux, dtype=dt, device=dv))
        bound = 1.0 / math.sqrt(self.w)            # nn.Linear default (kaiming_uniform a=sqrt(5)) and bias bound
        with torch.no_grad():
            for m, s in enumerate(seeds):
                g = torch.Generator().manual_seed(AUX_SEED + int(s))
                self.aux_w[m].copy_(((torch.rand(self.n_aux, self.w, generator=g) * 2 - 1) * bound).T)
                self.aux_b[m].copy_((torch.rand(self.n_aux, generator=g) * 2 - 1) * bound)
        return self

    def forward(self, x):
        """x (M, B, fin) -> p (M, B, K), q (M, B, nscal), a (M, B, n_aux)."""
        h = torch.baddbmm(self.inp_b.unsqueeze(1), x, self.inp_w)
        if self.ln_impl == "eager":
            ln, od = _ln_affine, None
        else:
            ln, od = _ln_compiled(), _amp_dtype(h)
        for blk in self.blocks:
            z = ln(h, blk.ln_w, blk.ln_b, od)
            z = F.gelu(torch.baddbmm(blk.b1.unsqueeze(1), z, blk.w1))
            if self.drop > 0:
                z = F.dropout(z, self.drop, self.training)
            h = h + torch.baddbmm(blk.b2.unsqueeze(1), z, blk.w2)
        z = ln(h, self.n_w, self.n_b, od)
        o = torch.baddbmm(self.head_b.unsqueeze(1), z, self.head_w)
        if self.has_skip:
            s = torch.baddbmm(self.skip_b.unsqueeze(1), x, self.skip_w)
            o = o + s * self.skip_mask.to(s.dtype)
        a = torch.baddbmm(self.aux_b.unsqueeze(1), z, self.aux_w)
        return o[..., :self.K], o[..., self.K:], a

    def aux_state(self, m):
        return {"aux.weight": self.aux_w.detach()[m].float().cpu().T.contiguous(),
                "aux.bias": self.aux_b.detach()[m].float().cpu().contiguous()}


def aux_weights(heads, aux_w, wscal):
    """Per-column loss weights: aux_w x wscal per scalar head (kappa, tri_top, tri_bot, volume); the J_phi profile is
    ONE head of weight aux_w x wscal, i.e. aux_w x wscal / 16 per point (pre-declared reading, D3D_MTASK doc)."""
    w = [aux_w * wscal] * 4
    if heads == "shape_jphi":
        w += [aux_w * wscal / AUX_JPHI] * AUX_JPHI
    return w


def aux_context(cache_dir, heads, rows, tr, va, dev):
    """Aux targets at the run's `rows` (global ds frame indices): cache columns of `heads`, standardised with
    TRAIN-row (tr) nan-aware mean/sd; NaN -> mask 0. Every TRAIN row must be in the cache (invalid ones are NaN)."""
    d = Path(cache_dir)
    meta = json.loads((d / "meta.json").read_text())
    n = {"shape": 4, "shape_jphi": 4 + AUX_JPHI}[heads]
    G = np.load(d / "gidx.npy")
    Araw = np.load(d / "aux.npy", mmap_mode="r")
    assert Araw.shape[1] >= n and np.all(np.diff(G) > 0), (Araw.shape, n)
    pos = np.clip(np.searchsorted(G, rows), 0, len(G) - 1)
    have = G[pos] == rows
    if not have[tr].all():
        raise SystemExit(f"--aux_cache {d}: {int((~have[tr]).sum())} TRAIN rows missing")
    A = np.full((len(rows), n), np.nan, np.float64)
    A[have] = np.asarray(Araw[pos[have], :n], np.float64)
    mu = np.nanmean(A[tr], 0)
    sd = np.nanstd(A[tr], 0)
    assert np.isfinite(mu).all() and (sd > 0).all(), (mu, sd)
    Z = (A - mu) / sd
    Mk = np.isfinite(Z)
    info = {"cache": str(d), "heads": heads, "names": meta["names"][:n], "train_mu": mu.tolist(),
            "train_sd": sd.tolist(), "n_rows": int(len(rows)), "train_masked_frac": (~Mk[tr]).mean(0).tolist(),
            "val_present": int(have[va].sum()), "val_masked_frac": (~Mk[va]).mean(0).tolist() if len(va) else None,
            "cache_aux_md5": meta.get("aux_md5"), "cache_gidx_md5": meta.get("gidx_md5")}
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    return {"A": t(np.where(Mk, Z, 0.0)), "M": t(Mk.astype(np.float32)), "info": info, "n": n}


def train_step_aux(snet, params, opt, sched, xb, yb, Sb, Ab, Mb, b, cfg, ctx, wsc, waux, pix, chunk,
                   amp_dtype=torch.bfloat16, ema=None, impl="eager"):
    """train_step + the masked auxiliary loss: la_m = sum_j waux_j sum_b M (a - A)^2 / max(sum_b M, 1) on the
    standardised targets Ab (M,B,n) with mask Mb; backward of [p32 with the pixel cotangent, sum(lq + la)].
    Returns (per-member lp + lq, per-member la)."""
    M = xb.shape[0]
    with torch.autocast(xb.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        p, q, a = snet(xb)
    p32 = p.float()
    lp, gp = psi_loss_and_grad(p32, yb.float(), b, cfg, ctx, pix, chunk, impl)
    lq = (((q.float() - Sb) ** 2).mean(1) * wsc).sum(1)
    la = ((((a.float() - Ab) ** 2) * Mb).sum(1) / Mb.sum(1).clamp_min(1.0) * waux).sum(1)
    opt.zero_grad(set_to_none=True)
    torch.autograd.backward([p32, (lq + la).sum()], [gp, None])
    clip_grad_norm_per_member(params, M, 1.0)
    opt.step()
    if sched is not None:
        sched.step()
    if ema is not None:
        ema.update(params)
    return (lp + lq).detach(), la.detach()


# ----------------------------------------------------------------------------- Sobolev / B_p (mscale)
SB_DEFAULTS = {"wbp": 0.0, "bp_sig": 0.05, "bp_band": 1.0, "pcgrad": False}


def sb_of(m):
    """Sobolev/B_p settings of a parsed member (absent flags -> defaults)."""
    return {k: getattr(m, k, v) for k, v in SB_DEFAULTS.items()}


def fd_matrix(x):
    """(n, n) matrix G with G @ f == np.gradient(f, x) along axis 0 (numpy default edge_order=1):
    second-order central differences inside (non-uniform form), first-order one-sided at both ends.
    The scorer's li uses np.gradient(psi, Z, R) (fusion_scoring/derive.py::_poloidal_field)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    h = np.diff(x)
    G = np.zeros((n, n))
    G[0, 0], G[0, 1] = -1.0 / h[0], 1.0 / h[0]
    G[n - 1, n - 2], G[n - 1, n - 1] = -1.0 / h[-1], 1.0 / h[-1]
    for i in range(1, n - 1):
        hs, hd = h[i - 1], h[i]
        den = hs * hd * (hs + hd)
        G[i, i - 1] = -hd * hd / den
        G[i, i] = (hd * hd - hs * hs) / den
        G[i, i + 1] = hs * hs / den
    return G


def bp_context(grid_R, grid_Z, dev, dtype=torch.float32):
    """Constants of the B_p loss on the (Z rows, R columns) grid: GZ (nZ,nZ), GRt = GR^T (nR,nR),
    invR2 (nR,) = 1/R^2, ring (nZ,nR) = 0 on the outer grid ring, 1 inside."""
    ring = torch.zeros(len(grid_Z), len(grid_R), dtype=dtype, device=dev)
    ring[1:-1, 1:-1] = 1.0
    return {"GZ": torch.as_tensor(fd_matrix(grid_Z), dtype=dtype, device=dev),
            "GRt": torch.as_tensor(fd_matrix(grid_R).T.copy(), dtype=dtype, device=dev),
            "invR2": torch.as_tensor(1.0 / np.asarray(grid_R, np.float64) ** 2, dtype=dtype, device=dev),
            "ring": ring}


def bp_loss(pc, yc, bb, mean, VTs, maskc, GZ, GRt, invR2, ring, sig, band, eps_rel=1e-3):
    """Per-member normalised poloidal-field loss (m,) and its parts (m, 2) = (mean L_int, mean L_band).

    pc, yc (m,B,K): predicted / target PCA coefficients (ysd units); bb (m,B,2): psi at the axis and at the
    LCFS (bnd2 rows). Error map E = (pc - yc) @ VTs (stored psi units), so B_hat - B = (G_R E, G_Z E) / R.
    Per frame L_int = sum I |dB|^2 / (sum I |B|^2 + eps), L_band likewise with W; I = [psiN < 1]*env*ring,
    W = exp(-((psiN - 1)/sig)^2 / 2)*env*ring from the TRUE psi (no gradient path); eps = eps_rel x the
    member's batch-mean denominator. Loss = mean over frames of L_int + band * L_band."""
    m, B, K = pc.shape
    nz, nr = GZ.shape[0], GRt.shape[0]
    with torch.no_grad():
        T = torch.addmm(mean, yc.reshape(-1, K), VTs).view(m, B, nz, nr)
        p0 = bb[..., 0, None, None]
        psin = (T - p0) / (bb[..., 1, None, None] - p0)
        env = maskc.view(nz, nr) * ring
        I = (psin < 1.0).to(T.dtype) * env
        W = torch.exp(((psin - 1.0) / sig).square() * -0.5) * env
        del psin
        B2 = (torch.matmul(GZ, T).square() + torch.matmul(T, GRt).square()) * invR2
        del T
        di = (I * B2).sum((-2, -1))
        dw = (W * B2).sum((-2, -1))
        del B2
        di = di + eps_rel * di.mean(1, keepdim=True)
        dw = dw + eps_rel * dw.mean(1, keepdim=True)
    E = ((pc - yc).reshape(-1, K) @ VTs).view(m, B, nz, nr)
    D2 = (torch.matmul(GZ, E).square() + torch.matmul(E, GRt).square()) * invR2
    l_int = (I * D2).sum((-2, -1)) / di
    l_band = (W * D2).sum((-2, -1)) / dw
    return (l_int + band * l_band).mean(1), torch.stack([l_int.detach().mean(1), l_band.detach().mean(1)], 1)


def bp_loss_and_grad(p32, yb, b, ctx, sb, chunk):
    """B_p loss (M,), its gradient w.r.t. p32 (M,B,K) and parts (M,2), chunk by chunk of members."""
    M = p32.shape[0]
    lb_all = torch.empty(M, device=p32.device, dtype=p32.dtype)
    parts = torch.empty(M, 2, device=p32.device, dtype=p32.dtype)
    gb = torch.empty_like(p32)
    for c0 in range(0, M, chunk):
        c1 = min(M, c0 + chunk)
        pc = p32[c0:c1].detach().requires_grad_(True)
        with torch.enable_grad():
            lb, pt = bp_loss(pc, yb[c0:c1], ctx["bnd"][b[c0:c1]], ctx["mean"], ctx["VTs"], ctx["maskc"],
                             ctx["GZ"], ctx["GRt"], ctx["invR2"], ctx["ring"], float(sb["bp_sig"]),
                             float(sb["bp_band"]))
            g, = torch.autograd.grad(lb.sum(), pc)
        gb[c0:c1] = g
        lb_all[c0:c1] = lb.detach()
        parts[c0:c1] = pt
    return lb_all, gb, parts


@torch.no_grad()
def member_gram(ga, gq, gb, M):
    """(M,3,3) float64 Gram matrix of the per-member gradient vectors (g_psi, g_scal, g_B); lists of
    stacked gradients (leading member axis). Full fp32 products (TF32 off), float64 accumulation over
    parameters. Each tensor is split into rows of its last dimension, (M, r, 3, c) @ (M, r, c, 3) summed
    over r: a single (3 x n) @ (n x 3) GEMM with n up to 4.7M was ~20x slower (2026-10-10 bench)."""
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        G = None
        for a, q, g in zip(ga, gq, gb):
            c = a.shape[-1] if a.ndim > 1 else 1
            X = torch.stack([a.reshape(M, -1, c), q.reshape(M, -1, c), g.reshape(M, -1, c)], 2).float()
            part = torch.matmul(X, X.transpose(-1, -2)).sum(1).double()
            G = part if G is None else G + part
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return G


@torch.no_grad()
def sb_coeffs(gram, rho, pcgrad, tiny=1e-30):
    """Per-member coefficients of the SB update g = ca*g_psi + cq*g_scal + cb*g_B.

    gram (M,3,3) Gram matrix of (g_psi, g_scal, g_B); rho (M,). Weight w = rho*|g_psi|/|g_B|; with
    pcgrad, g_B -> g_B + la*g_psi + lq*g_scal = the Euclidean projection of g_B onto the cone
    {g . g_psi >= 0, g . g_scal >= 0} (KKT cases: none / psi / scal / both active; sequential PCGrad as a
    fallback for a degenerate Gram). Returns ca, cq, cb (M,) float64 and diagnostics (M,) tensors."""
    aa, qq, gg = gram[:, 0, 0], gram[:, 1, 1], gram[:, 2, 2]
    aq, ga, gq = gram[:, 0, 1], gram[:, 2, 0], gram[:, 2, 1]
    zero = torch.zeros_like(aa)
    w = torch.where(gg > tiny, rho.to(aa.dtype) * torch.sqrt(aa) / torch.sqrt(gg.clamp_min(tiny)), zero)
    la, lq = zero, zero
    if pcgrad:
        aac, qqc = aa.clamp_min(tiny), qq.clamp_min(tiny)
        ok0 = (ga >= 0) & (gq >= 0)
        la_a = (-ga / aac).clamp_min(0)                      # only the psi constraint active
        ok_a = (ga < 0) & (gq + la_a * aq >= 0)
        lq_q = (-gq / qqc).clamp_min(0)                      # only the scalar constraint active
        ok_q = (gq < 0) & (ga + lq_q * aq >= 0)
        det = aa * qq - aq * aq                              # both active: g' . g_psi = g' . g_scal = 0
        detc = torch.where(det.abs() > tiny, det, torch.ones_like(det))
        la_b = (-ga * qq + gq * aq) / detc
        lq_b = (-gq * aa + ga * aq) / detc
        ok_b = (det > 1e-12 * aa * qq) & (la_b >= 0) & (lq_b >= 0)
        la_s = la_a                                          # fallback: sequential PCGrad (psi, then scal)
        lq_s = (-(gq + la_s * aq) / qqc).clamp_min(0)
        la = torch.where(ok0, zero, torch.where(ok_a, la_a, torch.where(ok_q, zero, torch.where(ok_b, la_b, la_s))))
        lq = torch.where(ok0, zero, torch.where(ok_a, zero, torch.where(ok_q, lq_q, torch.where(ok_b, lq_b, lq_s))))
    ca, cq, cb = 1.0 + w * la, 1.0 + w * lq, w
    den = torch.sqrt(gg.clamp_min(tiny))
    diag = {"cos_psi": ga / (den * torch.sqrt(aa.clamp_min(tiny))),
            "cos_scal": gq / (den * torch.sqrt(qq.clamp_min(tiny))),
            "act_psi": (la > 0).to(aa.dtype), "act_scal": (lq > 0).to(aa.dtype),
            "ratio": torch.sqrt(gg / aa.clamp_min(tiny)), "_aa": aa, "_qq": qq, "_aq": aq}
    return ca, cq, cb, diag


def train_step_sb(snet, params, opt, sched, xb, yb, Sb, b, cfg, ctx, wsc, pix, chunk, sb, rho,
                  amp_dtype=torch.bfloat16, ema=None, impl="eager", acc=None):
    """train_step plus the Sobolev/B_p objective (section 3 of D3D_MEMBER_SCALE_20261010.md).

    Three VJPs through the same forward: g_psi (pixel/PCA loss, unchanged), g_scal (scalar heads,
    unchanged), g_B (bp_loss); combined per member by sb_coeffs, then the unchanged per-member clip at 1.0,
    AdamW, OneCycle and EMA. rho (M,) = --wbp per member. acc: optional dict of (M,) float64 accumulators
    for the diagnostics. Returns the per-member pixel + scalar loss (M,), as train_step."""
    M = xb.shape[0]
    with torch.autocast(xb.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        p, q = snet(xb)
    p32 = p.float()
    y32 = yb.float()
    lp, gp = psi_loss_and_grad(p32, y32, b, cfg, ctx, pix, chunk, impl)
    lb, gbp, parts = bp_loss_and_grad(p32, y32, b, ctx, sb, chunk)
    lq = (((q.float() - Sb) ** 2).mean(1) * wsc).sum(1)
    opt.zero_grad(set_to_none=True)
    g_a = torch.autograd.grad(p32, params, gp, retain_graph=True, allow_unused=True)
    g_q = torch.autograd.grad(lq.sum(), params, retain_graph=True, allow_unused=True)
    g_b = torch.autograd.grad(p32, params, gbp, allow_unused=True)
    fill = lambda gs: [torch.zeros_like(w) if g is None else g for g, w in zip(gs, params)]  # noqa: E731
    g_a, g_q, g_b = fill(g_a), fill(g_q), fill(g_b)
    ca, cq, cb, diag = sb_coeffs(member_gram(g_a, g_q, g_b, M), rho, bool(sb["pcgrad"]))
    for w, a, qg, g in zip(params, g_a, g_q, g_b):
        sh = (M,) + (1,) * (a.ndim - 1)
        a.mul_(ca.view(sh).to(a.dtype)).addcmul_(qg, cq.view(sh).to(a.dtype)).addcmul_(g, cb.view(sh).to(a.dtype))
        w.grad = a
    del g_q, g_b
    gnorm = clip_grad_norm_per_member(params, M, 1.0)
    opt.step()
    if sched is not None:
        sched.step()
    if ema is not None:
        ema.update(params)
    if acc is not None:
        with torch.no_grad():
            gram_plain = (diag.pop("_aa") + diag.pop("_qq") + 2 * diag.pop("_aq")).clamp_min(0).sqrt()
            for k, v in (("l_int", parts[:, 0]), ("l_band", parts[:, 1]), ("w", cb), ("gnorm", gnorm),
                         ("clip", (gnorm > 1.0).double()), ("gnorm_plain", gram_plain), *diag.items()):
                acc[k] = acc.get(k, 0.0) + v.double()
            acc["n"] = acc.get("n", 0) + 1
    return (lp + lq).detach()


# ----------------------------------------------------------------------------- li2ft fine-tune (agent li2ft)
# Opt-in mode (members with --ft_from; docs/archive/fec/D3D_LI2FT_20261012.md; pre-declaration: family doc block
# "Pre-declaration, DIII-D li(2)-loss fine-tune li2ft" + Amendment 1). Without --ft_from nothing below runs.
#   * start from WORK/models/<ft_from>/model.pt; keep its PCA basis (mean, V, ysd), input normalisation (stats, xmu,
#     xsd) and scalar normalisation (smu, ssd); targets Y = the ds psi projected on that basis (the base loss
#     unchanged: pixel/PCA + scalar heads, the member's own flags, which must equal the checkpoint's);
#   * fresh fused AdamW (same wd), LR cosine from ft_lr_frac x lr to 0 over epochs x (pool // bs) steps, per-member
#     clip 1.0, bf16 autocast as the base trainer;
#   * data order: per member torch.Generator seeded with --seed, one randperm of the pool per epoch; the pool is the
#     TRAIN rows, plus (gap_os - 1) extra copies of the frames within 600 ms after a > 100 ms EFIT-time gap;
#   * li(2) term (li2_rho > 0): lambda_m * sum_i w_i ((li2(psi_hat_i; C_true,i) - y_i) / sigma)^2 / B, li2 at the TRUE
#     map's cached scorer contour (fec/li2_torch.py), w = li2_w where y > li2_thr else 1, only on cached frames;
#     lambda_m = rho_m x median|g_base| / median|g_li| over the psi output layer (head columns :K + bias) in a
#     measurement-only pass over the first ft_cal_batches batches of the epoch-0 order (no optimizer step), then
#     frozen; training starts at batch 0 of that same order;
#   * feat base_gap / nots_gap: the checkpoint's base / nots inputs + fec/features_b.gap_columns, standardised with
#     TRAIN statistics; the new first-layer (and linear-skip) input columns start at 0.
FT_DEFAULTS = {"ft_from": None, "ft_lr_frac": 0.1, "li2_rho": 0.0, "li2_w": 4.0, "li2_thr": 1.7, "gap_os": 1,
               "shape_rho": 0.0, "wnorm": "batch"}
FT_SHARED = ("ft_lr_frac", "li2_w", "li2_thr", "gap_os", "shape_rho", "wnorm")
FT_SAME_AS_BASE = ("K", "width", "depth", "drop", "linskip", "bs", "lr", "wd", "wscal", "wband", "sig", "win",
                   "wgrad", "wli", "wx", "rx", "noise", "ema", "tsdrop", "trainval", "alldata")
GAP_WINDOW_MS = 600.0
LI2_EPS = 1e-12                    # Bp^2 floor under the square root (gradient stays finite at a Bp = 0 node)


def ft_of(m):
    """li2ft settings of a parsed member (absent flags -> defaults)."""
    return {k: getattr(m, k, v) for k, v in FT_DEFAULTS.items()}


def is_ft(m):
    return getattr(m, "ft_from", None) is not None


def check_ft_members(ms):
    """parse_members' extra checks for a fine-tune run (all members fine-tune; shared schedule flags)."""
    from fec.train_d3d_b import FT_KEYS
    fts = {is_ft(m) for m in ms}
    if len(fts) > 1:
        raise SystemExit("fine-tune (--ft_from) and from-scratch members cannot share a run")
    if fts == {False}:
        if any(hasattr(m, k) for m in ms for k in FT_KEYS):
            raise SystemExit("--ft_lr_frac/--li2_*/--gap_os need --ft_from")
        return
    for k in FT_SHARED:
        vals = {json.dumps(ft_of(m)[k]) for m in ms}
        if len(vals) > 1:
            raise SystemExit(f"fine-tune members must share --{k}: {sorted(vals)}")
    for m in ms:
        f = ft_of(m)
        if f["li2_rho"] < 0 or f["gap_os"] < 1 or f["ft_lr_frac"] <= 0 or f["li2_w"] < 0 or f["shape_rho"] < 0:
            raise SystemExit(f"{m.tag}: need li2_rho >= 0, gap_os >= 1, ft_lr_frac > 0, li2_w >= 0")
        if sb_of(m)["wbp"] > 0 and (f["shape_rho"] > 0 or f["wnorm"] != "batch"):
            raise SystemExit("--wbp in a fine-tune (arm ASB) combines only with the li(2) term, not --shape_rho / "
                             "--wnorm frame")


def ft_net(m, ck, fin):
    """NetB of member m holding checkpoint ck's weights; input columns beyond ck['fin'] get zero weights (first layer
    and linear skip)."""
    torch.manual_seed(m.seed)
    net = build_net(vars(m), fin, NSCAL)
    sd = {k: v.detach().clone() for k, v in ck["net"].items()}
    f0 = int(ck["fin"])
    if fin != f0:
        for k in ("inp.weight", "skip.weight"):
            if k in sd:
                w = sd[k]
                sd[k] = torch.cat([w, torch.zeros(w.shape[0], fin - f0, dtype=w.dtype)], 1)
    net.load_state_dict(sd, strict=True)
    return net


def ft_pool(tr, after_gap_tr, gap_os):
    """Row positions of one fine-tune epoch before shuffling: TRAIN rows + (gap_os - 1) copies of the after-gap ones."""
    tr = np.asarray(tr)
    extra = [tr[np.asarray(after_gap_tr, bool)]] * (int(gap_os) - 1)
    return np.concatenate([tr] + extra) if extra else tr.copy()


def cosine_factor(steps):
    """LambdaLR factor: cosine from 1 (first step) to 0 over `steps` optimizer steps."""
    return lambda s: 0.5 * (1.0 + math.cos(math.pi * min(s, steps) / steps))


def li2_context(cache_dir, rows, tr, grid_R, grid_Z, dev, store_dev, w_hi, thr):
    """Device tensors of the li term: per-row compact cache index ci (-1 = no valid target), y per row, the cached
    contours / packed inside masks of the valid cached rows, the Li2Grid and sigma_TRAIN."""
    from fec.li2_torch import Li2Cache, Li2Grid
    c = Li2Cache(cache_dir)
    if not (np.allclose(c.meta["grid_R"], grid_R) and np.allclose(c.meta["grid_Z"], grid_Z)):
        raise SystemExit("li2 cache grid differs from env_train's")
    pos = c.positions(rows)
    ok = pos >= 0
    ok[ok] = c.valid[pos[ok]]
    sel = np.flatnonzero(ok)                                   # row positions with a valid target
    ci = np.full(len(rows), -1, np.int64)
    ci[sel] = np.arange(len(sel))
    y = np.full(len(rows), np.nan, np.float32)
    y[sel] = c.y[pos[sel]]
    trm = np.zeros(len(rows), bool)
    trm[tr] = True
    yt = c.y[pos[sel[trm[sel]]]]
    sigma = float(np.std(yt))
    order = pos[sel]
    if store_dev == "mmap":                                    # rows read from the memory-mapped cache per step
        C, I = c.C, c.inside  # noqa: E741
    else:
        C = torch.from_numpy(np.ascontiguousarray(c.C[order]))
        I = torch.from_numpy(np.ascontiguousarray(c.inside[order]))  # noqa: E741
        if store_dev == "cpu":
            C, I = C.pin_memory(), I.pin_memory()  # noqa: E741
        else:
            C, I = C.to(dev), I.to(dev)  # noqa: E741
    rep = {"cache": str(cache_dir), "train_rows": int(len(tr)), "train_rows_with_target": int(trm[sel].sum()),
           "train_coverage": float(trm[sel].sum() / max(len(tr), 1)), "sigma_train": sigma,
           "train_y_gt_thr": int((yt > thr).sum()), "store": store_dev}
    return {"ci": torch.as_tensor(ci, device=dev), "y": torch.as_tensor(y, device=dev), "C": C, "I": I,
            "order": order, "grid": Li2Grid(grid_R, grid_Z, dev, torch.float32), "sigma": sigma,
            "w_hi": float(w_hi), "thr": float(thr), "report": rep}


def li2_loss_and_grad(p32, b, ctx, chunk):
    """Per-member li term (M,) and its gradient w.r.t. p32 (M,B,K): mean over the B frames of
    w * ((li2(psi_hat; C_true) - y) / sigma)^2 on frames with a valid cached target (0 elsewhere)."""
    from fec.li2_torch import contour_weights, li2, unpack_inside
    L = ctx["li2"]
    g = L["grid"]
    M, B, K = p32.shape
    ll = torch.zeros(M, device=p32.device, dtype=p32.dtype)
    gl = torch.zeros_like(p32)
    for c0 in range(0, M, chunk):
        c1 = min(M, c0 + chunk)
        bc = b[c0:c1]
        ci = L["ci"][bc]
        ok = ci >= 0
        cc = ci.clamp_min(0)
        if isinstance(L["C"], np.ndarray):                     # memory-mapped cache: gather this batch's rows
            cx = L["order"][cc.cpu().numpy().ravel()]
            C = torch.from_numpy(np.asarray(L["C"][cx])).view(tuple(cc.shape) + (-1, 2)).to(p32.device)
            Ip = torch.from_numpy(np.asarray(L["I"][cx])).view(tuple(cc.shape) + (-1,)).to(p32.device)
        elif L["C"].device != p32.device:
            cx = cc.cpu()
            C = L["C"][cx].to(p32.device, non_blocking=True)
            Ip = L["I"][cx].to(p32.device, non_blocking=True)
        else:
            C = L["C"][cc]
            Ip = L["I"][cc]
        y = torch.nan_to_num(L["y"][bc], nan=0.0)
        w = torch.where(y > L["thr"], L["w_hi"], 1.0) * ok
        with torch.no_grad():
            cw = contour_weights(C, g)
            ins = unpack_inside(Ip)
        pc = p32[c0:c1].detach().requires_grad_(True)
        with torch.enable_grad():
            psi = (pc.reshape(-1, K) @ ctx["VTs"] + ctx["mean"]).view(c1 - c0, B, 65, 65)
            li = li2(psi, ins, cw, g, eps=LI2_EPS)
            fin = ok & torch.isfinite(li)
            r = torch.where(fin, (li - y) / L["sigma"], torch.zeros_like(li))
            lv = (w * r.square()).sum(1) / B
            gg, = torch.autograd.grad(lv.sum(), pc)
        gl[c0:c1] = gg
        ll[c0:c1] = lv.detach()
    return ll, gl


def train_step_ft(snet, params, opt, sched, xb, yb, Sb, b, cfg, ctx, wsc, pix, chunk, lam,
                  amp_dtype=torch.bfloat16, ema=None, impl="eager"):
    """train_step with the li term added to the psi-output gradient: g_p = g_base + lam_m * g_li (lam None = off,
    exactly train_step). Returns (base loss (M,), li term (M,) or None)."""
    M = xb.shape[0]
    with torch.autocast(xb.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        p, q = snet(xb)
    p32 = p.float()
    lp, gp = psi_loss_and_grad(p32, yb.float(), b, cfg, ctx, pix, chunk, impl)
    ll = None
    if lam is not None:
        ll, gl = li2_loss_and_grad(p32, b, ctx, chunk)
        gp = gp + lam.view(M, 1, 1).to(gp.dtype) * gl
    lq = (((q.float() - Sb) ** 2).mean(1) * wsc).sum(1)
    opt.zero_grad(set_to_none=True)
    torch.autograd.backward([p32, lq.sum()], [gp, None])
    clip_grad_norm_per_member(params, M, 1.0)
    opt.step()
    if sched is not None:
        sched.step()
    if ema is not None:
        ema.update(params)
    return (lp + lq).detach(), ll


def head_grad_norms(snet, p32, gp, K):
    """Per-member norm (M,) of the psi-output-layer gradient (head columns :K and bias :K) for dL/dp = gp."""
    gw, gb = torch.autograd.grad(p32, [snet.head_w, snet.head_b], gp, retain_graph=True)
    return torch.sqrt(gw[..., :K].double().square().sum((1, 2)) + gb[..., :K].double().square().sum(1))


def li2_calibrate(snet, get_batch, nb, cfg, ctx, pix, chunk, rho, K, impl="eager", amp_dtype=torch.bfloat16,
                  head=None):
    """lambda (M,) = rho * median_n |g_base| / median_n |g_li| on the psi output layer over nb batches (no step).
    head: the StackedNetB holding head_w/head_b (default snet; pass the module when snet is torch.compile'd)."""
    head = head if head is not None else snet
    nbase, nli = [], []
    for i in range(nb):
        xb, yb, b = get_batch(i)
        with torch.autocast(xb.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
            p, _ = snet(xb)
        p32 = p.float()
        _, gp = psi_loss_and_grad(p32, yb.float(), b, cfg, ctx, pix, chunk, impl)
        _, gl = li2_loss_and_grad(p32, b, ctx, chunk)
        nbase.append(head_grad_norms(head, p32, gp, K))
        nli.append(head_grad_norms(head, p32, gl, K))
        del p, p32, gp, gl
    nbase = torch.stack(nbase)
    nli = torch.stack(nli)                                                 # (nb, M)
    mb, ml = nbase.median(0).values, nli.median(0).values
    lam = torch.where(ml > 0, rho.double() * mb / ml.clamp_min(1e-300), torch.zeros_like(mb))
    return lam, {"median_g_base": mb.tolist(), "median_g_li": ml.tolist(), "rho": rho.tolist(),
                 "lambda": lam.tolist(), "batches": nb}


def shape_context(shape_dir, li2, rows, tr, dev, store):
    """li2ft arm S: the true-geometry cache (fec/li2_shape.py) on the li2 context's compact index (same rows / order),
    plus Var_TRAIN of R_axis, kappa, V over TRAIN rows with a regular true branch."""
    from fec.li2_shape import ShapeCache
    sc = ShapeCache(shape_dir)
    order = li2["order"]                                       # cache positions of the compact index
    a = sc.a
    ci = li2["ci"].cpu().numpy()
    trm = np.zeros(len(rows), bool)
    trm[tr] = True
    tpos = order[ci[trm & (ci >= 0)]]
    var = {"r_ax": float(np.var(a["r_ax"][tpos][a["ax_ok"][tpos]])),
           "kap": float(np.var(a["kap"][tpos][a["kv_ok"][tpos] & a["kap_ok"][tpos]])),
           "vol": float(np.var(a["vol"][tpos][a["kv_ok"][tpos]]))}
    t = lambda x, dt=None: torch.as_tensor(np.ascontiguousarray(x), dtype=dt)  # noqa: E731
    if store == "mmap":                                        # rows read from the memory maps per step
        big = {k: a[k] for k in ("phiC", "g", "theta", "pt_ok")}
    else:
        big = {"phiC": t(a["phiC"][order]), "g": t(a["g"][order]), "theta": t(a["theta"][order]),
               "pt_ok": t(a["pt_ok"][order])}
    small = {"act_rz": t(a["act_rz"][order]), "b_act": t(a["b_act"][order]), "ext": t(a["ext"][order].astype(np.int64)),
             "msign": t(a["msign"][order]), "ax_cell": t(a["ax_cell"][order].astype(np.int64)),
             "ax_ok": t(a["ax_ok"][order]), "kv_ok": t(a["kv_ok"][order]), "kap_ok": t(a["kap_ok"][order]),
             "r_ax": t(a["r_ax"][order]), "kap": t(a["kap"][order]), "vol": t(a["vol"][order])}
    if store == "cpu":
        big = {k: v.pin_memory() for k, v in big.items()}
    elif store != "mmap":
        big = {k: v.to(dev) for k, v in big.items()}
    small = {k: v.to(dev) for k, v in small.items()}
    rep_ = {"shape_cache": str(shape_dir), "var_train": var, "store": store,
            "train_rows_ax_ok": int(a["ax_ok"][tpos].sum()), "train_rows_kv_ok": int(a["kv_ok"][tpos].sum()),
            "train_rows_kap_ok": int((a["kv_ok"][tpos] & a["kap_ok"][tpos]).sum()), "train_rows": int(trm.sum())}
    return {"big": big, "order": order, "small": small, "var": var, "report": rep_}


def shape_loss_and_grad(p32, b, ctx, chunk=1):
    """li2ft arm S: per-member shape loss (M,) (fec/li2_shape.shape_loss on R_axis, kappa, V at the frozen true geometry)
    and its gradient w.r.t. p32 (M,B,K); frames without a valid cached geometry contribute 0. One member at a time (each
    member has its own batch rows)."""
    from fec.li2_shape import NPTS, SIGN, shape_loss, shape_terms
    L, Sx = ctx["li2"], ctx["shape"]
    g = L["grid"]
    grid = {"R": g.R, "Z": g.Z, "R_np": g.R_np, "Z_np": g.Z_np}
    M, B, K = p32.shape
    ls = torch.zeros(M, device=p32.device, dtype=p32.dtype)
    gs = torch.zeros_like(p32)
    sh = torch.arange(7, -1, -1, device=p32.device, dtype=torch.uint8)
    for k in range(M):
        ci = L["ci"][b[k]]
        ok = ci >= 0
        cc = ci.clamp_min(0)
        if isinstance(L["C"], np.ndarray):
            C = torch.from_numpy(np.asarray(L["C"][L["order"][cc.cpu().numpy()]])).to(p32.device)
        else:
            C = L["C"][cc.to(L["C"].device)].to(p32.device)
        if isinstance(Sx["big"]["phiC"], np.ndarray):          # memory-mapped cache
            pos = Sx["order"][cc.cpu().numpy()]
            big = {kk: torch.from_numpy(np.asarray(v[pos])).to(p32.device) for kk, v in Sx["big"].items()}
        else:
            ccb = cc.to(Sx["big"]["phiC"].device)
            big = {kk: v[ccb].to(p32.device, non_blocking=True) for kk, v in Sx["big"].items()}
        geo = {kk: v[cc] for kk, v in Sx["small"].items()}
        th = big["theta"].float()
        geo.update(C=C.float(), phiC=big["phiC"].float(), g=big["g"].float(),
                   n=torch.stack([torch.cos(th), torch.sin(th)], -1),
                   pt_ok=((big["pt_ok"].unsqueeze(-1) >> sh) & 1).reshape(B, -1)[:, :NPTS].bool())
        geo["ax_ok"] = geo["ax_ok"] & ok
        geo["kv_ok"] = geo["kv_ok"] & ok
        pc = p32[k:k + 1].detach().requires_grad_(True)
        with torch.enable_grad():
            phi = SIGN * (pc.reshape(-1, K) @ ctx["VTs"] + ctx["mean"]).view(1, B, 65, 65)
            tm = shape_terms(phi, geo, grid)
            lv = shape_loss(tm, {f: geo[f] for f in ("r_ax", "kap", "vol")}, Sx["var"], B)
            gg, = torch.autograd.grad(lv.sum(), pc)
        gs[k:k + 1] = gg
        ls[k] = lv.detach()[0]
    return ls, gs


def shape_calibrate(snet, get_batch, nb, cfg, ctx, pix, chunk, rho, K, impl="eager", amp_dtype=torch.bfloat16,
                    head=None):
    """li2ft arm S: lambda_shape (M,) = rho / median_n (|g_shape,n| / |g_base,n|) on the psi output layer over nb batches
    (no optimizer step): the median per-batch ratio |lambda g_shape| / |g_base| equals rho (pre-declared form)."""
    head = head if head is not None else snet
    ratios = []
    for i in range(nb):
        xb, yb, b = get_batch(i)
        with torch.autocast(xb.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
            p, _ = snet(xb)
        p32 = p.float()
        _, gp = psi_loss_and_grad(p32, yb.float(), b, cfg, ctx, pix, chunk, impl)
        _, gs = shape_loss_and_grad(p32, b, ctx)
        ratios.append(head_grad_norms(head, p32, gs, K) / head_grad_norms(head, p32, gp, K).clamp_min(1e-300))
        del p, p32, gp, gs
    r = torch.stack(ratios)                                                   # (nb, M)
    med = r.median(0).values
    lam = torch.where(med > 0, rho.double() / med.clamp_min(1e-300), torch.zeros_like(med))
    return lam, {"median_ratio_g_shape_over_g_base": med.tolist(), "rho": rho.tolist(), "lambda_shape": lam.tolist(),
                 "batches": nb, "contract": "lambda_shape = rho / median_b(|g_shape,b| / |g_base,b|) (median of ratios)"}


def train_step_ft_shape(snet, params, opt, sched, xb, yb, Sb, b, cfg, ctx, wsc, pix, chunk, lam, lam_s, head,
                        amp_dtype=torch.bfloat16, ema=None, impl="eager"):
    """train_step_ft plus the arm-S shape term, whose gradient reaches the psi output layer only (trunk stop-grad):
    head_w / head_b gradients get + d(lam_s * L_shape)/d(head) computed with the trunk features held fixed."""
    M = xb.shape[0]
    with torch.autocast(xb.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        p, q = snet(xb)
    p32 = p.float()
    lp, gp = psi_loss_and_grad(p32, yb.float(), b, cfg, ctx, pix, chunk, impl)
    ll = None
    if lam is not None:
        ll, gl = li2_loss_and_grad(p32, b, ctx, chunk)
        gp = gp + lam.view(M, 1, 1).to(gp.dtype) * gl
    lsh, gsh = shape_loss_and_grad(p32, b, ctx)
    gw, gb_ = torch.autograd.grad(p32, [head.head_w, head.head_b], lam_s.view(M, 1, 1).to(gsh.dtype) * gsh,
                                  retain_graph=True)
    lq = (((q.float() - Sb) ** 2).mean(1) * wsc).sum(1)
    opt.zero_grad(set_to_none=True)
    torch.autograd.backward([p32, lq.sum()], [gp, None])
    head.head_w.grad.add_(gw)
    head.head_b.grad.add_(gb_)
    clip_grad_norm_per_member(params, M, 1.0)
    opt.step()
    if sched is not None:
        sched.step()
    if ema is not None:
        ema.update(params)
    return (lp + lq).detach(), ll, lsh


def train_step_ft_sb(snet, params, opt, sched, xb, yb, Sb, b, cfg, ctx, wsc, pix, chunk, lam, sb, rho,
                     amp_dtype=torch.bfloat16, ema=None, impl="eager", acc=None):
    """Arm ASB (family doc amendment 18 a): the li2ft step plus the Sobolev/B_p objective of train_step_sb.

    g_psi = the VJP of A's psi cotangent gp + lam_m * g_li (pixel/PCA loss plus the li(2) term, as train_step_ft);
    g_scal and g_B as in train_step_sb; per member w = rho * |g_psi| / |g_B| and, with pcgrad, g_B projected onto
    {g . g_psi >= 0, g . g_scal >= 0} (sb_coeffs); then the per-member clip at 1.0, AdamW, the schedule and EMA.
    lam None = train_step_sb exactly. Returns (base loss (M,), li term (M,) or None), as train_step_ft."""
    M = xb.shape[0]
    with torch.autocast(xb.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        p, q = snet(xb)
    p32 = p.float()
    y32 = yb.float()
    lp, gp = psi_loss_and_grad(p32, y32, b, cfg, ctx, pix, chunk, impl)
    ll = None
    if lam is not None:
        ll, gl = li2_loss_and_grad(p32, b, ctx, chunk)
        gp = gp + lam.view(M, 1, 1).to(gp.dtype) * gl
    lb, gbp, parts = bp_loss_and_grad(p32, y32, b, ctx, sb, chunk)
    lq = (((q.float() - Sb) ** 2).mean(1) * wsc).sum(1)
    opt.zero_grad(set_to_none=True)
    g_a = torch.autograd.grad(p32, params, gp, retain_graph=True, allow_unused=True)
    g_q = torch.autograd.grad(lq.sum(), params, retain_graph=True, allow_unused=True)
    g_b = torch.autograd.grad(p32, params, gbp, allow_unused=True)
    fill = lambda gs: [torch.zeros_like(w) if g is None else g for g, w in zip(gs, params)]  # noqa: E731
    g_a, g_q, g_b = fill(g_a), fill(g_q), fill(g_b)
    ca, cq, cb, diag = sb_coeffs(member_gram(g_a, g_q, g_b, M), rho, bool(sb["pcgrad"]))
    for w, a, qg, g in zip(params, g_a, g_q, g_b):
        sh = (M,) + (1,) * (a.ndim - 1)
        a.mul_(ca.view(sh).to(a.dtype)).addcmul_(qg, cq.view(sh).to(a.dtype)).addcmul_(g, cb.view(sh).to(a.dtype))
        w.grad = a
    del g_q, g_b
    gnorm = clip_grad_norm_per_member(params, M, 1.0)
    opt.step()
    if sched is not None:
        sched.step()
    if ema is not None:
        ema.update(params)
    if acc is not None:
        with torch.no_grad():
            gram_plain = (diag.pop("_aa") + diag.pop("_qq") + 2 * diag.pop("_aq")).clamp_min(0).sqrt()
            for k, v in (("l_int", parts[:, 0]), ("l_band", parts[:, 1]), ("w", cb), ("gnorm", gnorm),
                         ("clip", (gnorm > 1.0).double()), ("gnorm_plain", gram_plain), *diag.items()):
                acc[k] = acc.get(k, 0.0) + v.double()
            acc["n"] = acc.get("n", 0) + 1
    return (lp + lq).detach(), ll


def finetune_main(a, ms, pix):
    """--ft_from members: the li2ft fine-tune (see the block comment above)."""
    from fec.features import load_ds
    from fec.features_b import after_gap, d3d_inputs_b
    from fec.paths import WORK
    from fec.train_d3d_b import project_rows
    groups = {}
    for m in ms:
        groups.setdefault(m.feat, []).append(m)
    stacks = []
    for f, fm in groups.items():
        G = a.group_size or len(fm)
        stacks += [fm[i:i + G] for i in range(0, len(fm), G)]
    print("finetune stacks:", [[f"{m.tag}<-{m.ft_from}" for m in s] for s in stacks], "pix", pix,
          "li2_rho", [ft_of(m)["li2_rho"] for m in ms], flush=True)
    for m in ms:
        f = WORK / "models" / m.ft_from / "model.pt"
        if not f.exists():
            raise SystemExit(f"--ft_from {m.ft_from}: {f} missing")
    use_li = any(ft_of(m)["li2_rho"] > 0 for m in ms)
    use_shape = any(ft_of(m)["shape_rho"] > 0 for m in ms)
    if use_shape and not (getattr(a, "shape_cache", None) and a.li2_cache):
        raise SystemExit("--shape_rho > 0 needs --shape_cache DIR (python -B fec/li2_shape.py build) and --li2_cache")
    if use_li and not a.li2_cache:
        raise SystemExit("--li2_rho > 0 needs --li2_cache DIR (python -B fec/li2_torch.py build)")
    if use_li and not (Path(a.li2_cache) / "meta.json").exists():
        raise SystemExit(f"li2 cache {a.li2_cache} incomplete (meta.json missing)")
    use_aux = getattr(a, "aux_heads", None) is not None   # agent mtprod (amendment 34): the MT li2ft fine-tune
    base_aux = ft_aux_bases(a, ms, use_shape) if use_aux else {}
    if a.dry:
        return
    a0 = ms[0]
    f0 = ft_of(a0)
    dev = "cuda"
    for m in ms:
        out = WORK / "models" / m.tag
        if out.exists() and any(out.iterdir()) and not a.overwrite and not a.bench_steps:
            raise SystemExit(f"refusing to overwrite existing model dir {out} (use --overwrite)")
    if a0.alldata and not (WORK / "test_eval_b" / "DONE").exists():
        raise SystemExit("--alldata refused: WORK/test_eval_b/DONE missing")
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    tot_mem = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / tot_mem))
    timing = {"members": [m.tag for m in ms], "ft_from": [m.ft_from for m in ms], "stacks": [],
              "argv": sys.argv[1:], "mode": "li2ft finetune"}
    t0 = time.time()
    cks = {m.tag: torch.load(WORK / "models" / m.ft_from / "model.pt", map_location="cpu", weights_only=False,
                             mmap=True) for m in ms}
    c0 = cks[a0.tag]
    for m in ms:
        ck = cks[m.tag]
        for k in ("mean", "V", "smu", "ssd"):
            if not torch.equal(ck[k], c0[k]):
                raise SystemExit(f"{m.ft_from}: checkpoint {k} differs from {a0.ft_from}'s (one basis per run)")
        if ck["ysd"] != c0["ysd"] or json.dumps(ck["stats"]) != json.dumps(c0["stats"]):
            raise SystemExit(f"{m.ft_from}: ysd/stats differ from {a0.ft_from}'s")
        want = m.feat[:-4] if m.feat.endswith("_gap") else m.feat
        if ck["feat"] != want:
            raise SystemExit(f"{m.tag}: feat {m.feat} does not extend the checkpoint's {ck['feat']}")
        bad = [k for k in FT_SAME_AS_BASE if ck["args"].get(k) != getattr(m, k)]
        if bad:
            raise SystemExit(f"{m.tag}: flags differ from {m.ft_from}'s: {bad} (the base loss/net must be unchanged)")

    # ---- shared preparation (fits on tr, all rows if --alldata, unless the checkpoint fixes them)
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])[ds["shot"]]
    rows = np.arange(len(splits)) if a0.alldata else np.flatnonzero((splits == "train") | (splits == "val"))
    is_tr = (splits[rows] == "train") | a0.trainval
    tr = np.flatnonzero(is_tr)
    va = np.flatnonzero(splits[rows] == "val")
    mean = c0["mean"].to(dev)
    V = c0["V"].to(dev)
    ysd = float(c0["ysd"])
    Y = project_rows(ds["psi"], rows, mean, V)
    ds["psi"] = None
    Y = Y / ysd
    va_d = torch.as_tensor(va, device=dev)
    Yva = Y[va_d].clone()
    if not a.data_gpu:
        Y = Y.cpu().pin_memory()
        torch.cuda.empty_cache()
    scal = torch.as_tensor(ds["scal"][rows], device=dev)
    smu = c0["smu"].to(dev)
    ssd = c0["ssd"].to(dev)
    S = (scal - smu) / ssd
    ctx = {"li": torch.nan_to_num(scal[:, 2].float(), nan=1.0)}
    sfx = "_all" if a0.alldata else ""
    mk = np.load(WORK / "ds_b" / "diii_d_train" / "env_train.npz")
    if pix:
        ctx["bnd"] = torch.as_tensor(np.load(WORK / "ds_b" / "diii_d_train" / f"bnd2{sfx}.npy")[rows], device=dev)
        assert torch.isfinite(ctx["bnd"][torch.as_tensor(tr, device=dev)]).all()
        ctx["maskc"] = torch.as_tensor(mk["env"].astype(np.float32).reshape(-1), device=dev)
        RRg, ZZg = np.meshgrid(mk["grid_R"], mk["grid_Z"])
        ctx["RRg"] = torch.as_tensor(RRg.reshape(-1), device=dev, dtype=torch.float32)
        ctx["ZZg"] = torch.as_tensor(ZZg.reshape(-1), device=dev, dtype=torch.float32)
        if any(m.wx > 0 for m in ms):
            ctx["bx"] = torch.as_tensor(np.load(WORK / "ds_b" / "diii_d_train" / f"bnd_x{sfx}.npy")[rows], device=dev)
            assert torch.isfinite(ctx["bx"][torch.as_tensor(tr, device=dev)]).all()
    ctx["mean"] = mean
    ctx["VT"] = V.T.contiguous()
    ctx["VTs"] = ctx["VT"] * ysd
    if f0["wnorm"] == "frame":                                  # li2ft arm N
        ctx["frame_norm"] = True
        print("pixel weights normalised per frame (li2ft arm N)", flush=True)
    if sb_of(a0)["wbp"] > 0:                                     # arm ASB: Sobolev/B_p constants (pix members only)
        ctx.update(bp_context(mk["grid_R"], mk["grid_Z"], dev))
        timing["sb"] = sb_of(a0)
        print("B_p loss (arm ASB):", json.dumps(sb_of(a0)), flush=True)
    if use_li or use_shape:
        store = a.li2_cache_dev if a.li2_cache_dev != "auto" else ("cuda" if a.data_gpu else "mmap")
        ctx["li2"] = li2_context(a.li2_cache, rows, tr, mk["grid_R"], mk["grid_Z"], dev, store, f0["li2_w"],
                                 f0["li2_thr"])
        timing["li2"] = ctx["li2"]["report"]
        print("li2 term:", json.dumps(ctx["li2"]["report"]), flush=True)
        if use_shape:                                           # li2ft arm S
            ctx["shape"] = shape_context(a.shape_cache, ctx["li2"], rows, tr, dev, store)
            timing["shape"] = ctx["shape"]["report"]
            print("shape term:", json.dumps(ctx["shape"]["report"]), flush=True)
    ag = after_gap(ds, GAP_WINDOW_MS)[rows]
    timing["after_gap_frac"] = {"train": float(ag[tr].mean()), "val": float(ag[va].mean()) if len(va) else None}
    pool = ft_pool(tr, ag[tr], f0["gap_os"])
    timing["pool"] = {"rows": int(len(pool)), "train_rows": int(len(tr)), "gap_os": f0["gap_os"]}
    steps = a0.epochs * (len(pool) // a0.bs)
    print(f"targets ready {time.time() - t0:.0f}s; after-gap (<= {GAP_WINDOW_MS:.0f} ms) frac TRAIN "
          f"{timing['after_gap_frac']['train']:.4f} VAL {timing['after_gap_frac']['val']}; pool {len(pool)} rows "
          f"(gap_os {f0['gap_os']}); steps {steps}", flush=True)
    timing["prep_targets_s"] = time.time() - t0
    wsc = torch.tensor([1.0, 1.0, 0.3, 0.3, 0.3], device=dev) * a0.wscal
    x_bf16 = bool(a.x_bf16) and a0.noise == 0
    gt_va = None
    aux = ft_aux_context(a, ms, base_aux, rows, tr, va, dev, a0, timing) if use_aux else None   # agent mtprod
    cur_feat = None
    for stack in stacks:
        feat = stack[0].feat
        ck0 = cks[stack[0].tag]
        if feat != cur_feat:
            tf = time.time()
            Xg = None
            torch.cuda.empty_cache()
            X, _, _ = d3d_inputs_b(ds, "diii_d_train", ck0["stats"], feat=feat)
            X = X[rows]
            f_ck = int(ck0["fin"])
            xmu = np.asarray(ck0["xmu"], np.float32)
            xsd = np.asarray(ck0["xsd"], np.float32)
            if X.shape[1] > f_ck:                     # new columns: TRAIN statistics (train_d3d_b's convention)
                xmu = np.concatenate([xmu, X[tr, f_ck:].mean(0)]).astype(np.float32)
                xsd = np.concatenate([xsd, X[tr, f_ck:].std(0) + 1e-4]).astype(np.float32)
            assert X.shape[1] == len(xmu) == len(xsd), (X.shape, len(xmu))
            Xn = torch.as_tensor((X - xmu) / xsd)
            del X
            Xg = Xn.to(dtype=torch.bfloat16 if x_bf16 else torch.float32)
            Xg = Xg.to(dev) if a.data_gpu else Xg.pin_memory()
            del Xn
            cur_feat = feat
            timing.setdefault("prep_inputs_s", {})[feat] = time.time() - tf
            print("inputs", feat, tuple(Xg.shape), Xg.dtype, f"{time.time() - t0:.0f}s", flush=True)
        for m in stack:
            ck = cks[m.tag]
            assert np.array_equal(np.asarray(ck["xmu"], np.float32), xmu[:int(ck["fin"])]), m.tag
            assert np.array_equal(np.asarray(ck["xsd"], np.float32), xsd[:int(ck["fin"])]), m.tag
        if aux is None:
            st = train_stack_ft(stack, cks, a, Xg, Y, Yva, S, va, pool, ctx, wsc, steps, pix, dev, t0, use_li)
        else:                                                    # agent mtprod (amendment 34): + the MT aux heads
            st = train_stack_ft_aux(stack, cks, a, Xg, Y, Yva, S, va, pool, ctx, wsc, steps, pix, dev, t0, use_li,
                                    aux)
        if a.bench_steps:
            st["projected_train_s"] = st["ms_per_step"] * steps / 1e3
            timing.setdefault("bench", []).append(st)
            torch.cuda.empty_cache()
            continue
        ts = time.time()
        if not a0.trainval and gt_va is None:
            psi_mm = np.load(WORK / "ds" / "diii_d_train" / "psi.npy", mmap_mode="r")
            gt_va = np.asarray(psi_mm[rows[va]], dtype=np.float32).reshape(len(va), -1)
            del psi_mm
        for k, m in enumerate(stack):
            save_member(m, st["models"][k], WORK, xmu, xsd, ck0["stats"], mean, V, ysd, smu, ssd, Xg.shape[1], rows,
                        va, st["P"][k] if st["P"] is not None else None, st["Q"][k] if st["Q"] is not None else None,
                        gt_va, t0)
            info = {"ft_from": m.ft_from, **ft_of(m), "lambda": st["lambda"][k], "cal": st.get("cal"),
                    "lambda_shape": (st.get("lambda_shape") or [None] * len(stack))[k], "cal_shape": st.get("cal_shape"),
                    "steps": st["steps"], "pool_rows": int(len(pool)), "fin_base": int(cks[m.tag]["fin"]),
                    "fin": int(Xg.shape[1]), "li2": timing.get("li2"), "li_term_last_epoch": st["li_last"][k]}
            if "sb" in st:                                       # arm ASB only
                info["sb"] = {**{kk: v for kk, v in st["sb"].items() if kk != "last_epoch"},
                              "last_epoch": {kk: v[k] for kk, v in (st["sb"]["last_epoch"] or {}).items()}}
            out = WORK / "models" / m.tag
            sj = json.loads((out / "summary.json").read_text())
            sj["finetune"] = info
            (out / "summary.json").write_text(json.dumps(sj, default=float))
            (out / "finetune.json").write_text(json.dumps(info, indent=1, default=float))
            if aux is not None:                                  # agent mtprod: the fine-tuned aux head + aux.json
                save_aux_ft(m, k, st, aux, WORK)
        st["save_s"] = time.time() - ts
        if aux is not None:
            st.pop("aux_states")
        for k in ("models", "P", "Q"):
            st.pop(k)
        timing["stacks"].append(st)
        torch.cuda.empty_cache()
    timing["total_s"] = time.time() - t0
    if a.bench_steps:
        print("BENCH_PROJECTION", json.dumps({
            "stacks": [[b_["M"], round(b_["ms_per_step"], 1)] for b_ in timing["bench"]], "steps": steps,
            "projected_train_s": round(sum(b_["projected_train_s"] for b_ in timing["bench"])),
            "cal_s": [b_.get("cal_s") for b_ in timing["bench"]], "prep_targets_s": round(timing["prep_targets_s"])}),
            flush=True)
    print("BX_DONE", f"total {timing['total_s']:.0f}s", flush=True)
    if a.timing_json:
        Path(a.timing_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.timing_json).write_text(json.dumps(timing, indent=1, default=float))


def train_stack_ft(stack, cks, a, Xg, Y, Yva, S, va, pool, ctx, wsc, steps, pix, dev, t0, use_li):
    a0 = stack[0]
    f0 = ft_of(a0)
    M = len(stack)
    fin = Xg.shape[1]
    nets = [ft_net(m, cks[m.tag], fin) for m in stack]
    gens = [torch.Generator().manual_seed(int(m.seed)) for m in stack]
    ngens = []
    if a0.noise > 0:
        for m in stack:
            ng = torch.Generator(device=dev)
            ng.manual_seed(int(m.seed))
            ngens.append(ng)
    snet = StackedNetB.from_nets(nets).to(dev)
    del nets
    snet.ln_impl = a.ln_impl
    fwd = torch.compile(snet) if a.compile_net else snet
    params = list(snet.parameters())
    ema = StackedEMA(snet, a0.ema) if a0.ema > 0 else None
    ema_net = ema.module if ema is not None else None
    opt = torch.optim.AdamW(params, lr=f0["ft_lr_frac"] * a0.lr, weight_decay=a0.wd, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_factor(steps))
    cfg = cfg_of(stack)
    chunk = a.pix_chunk or M
    rho = torch.tensor([float(ft_of(m)["li2_rho"]) for m in stack], dtype=torch.float64, device=dev)
    stack_li = use_li and bool((rho > 0).any())
    pool_t = torch.as_tensor(pool)
    va_t = torch.as_tensor(va, device=dev)
    va_x = va_t if a.data_gpu else torch.as_tensor(va)
    batcher = None if a.data_gpu else HostBatcher(Xg, Y, M, a0.bs, dev)
    Sva = S[va_t]
    tags = [m.tag for m in stack]
    torch.cuda.reset_peak_memory_stats()

    def evaluate():
        mdl = ema_net if ema_net is not None else snet
        mdl.eval()
        P, Q = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(va_t), 16384):
                xb = Xg[va_x[i:i + 16384]].to(dev).unsqueeze(0).expand(M, -1, -1)
                p, q = mdl(xb)
                P.append(p.float())
                Q.append(q.float())
        mdl.train()
        P = torch.cat(P, 1)
        Q = torch.cat(Q, 1)
        e_psi = ((P - Yva) ** 2).sum(-1).mean(-1) / 4225
        r2s = 1 - ((Q - Sva) ** 2).mean(1) / Sva.var(0)
        return e_psi.cpu().numpy(), r2s.cpu().numpy(), P, Q

    def batch(ph, perm, i):
        bh, b = ph[:, i * a0.bs:(i + 1) * a0.bs], perm[:, i * a0.bs:(i + 1) * a0.bs]
        if batcher is None:
            xb, yb = Xg[b], Y[b]
        else:
            xb, yb = batcher.get(bh)
        if a0.noise > 0:
            xb = xb + a0.noise * torch.stack([torch.randn(xb.shape[1:], generator=g, device=dev) for g in ngens])
        return xb, yb, b

    lam = None
    cal = None
    ph = torch.stack([pool_t[torch.randperm(len(pool_t), generator=g)] for g in gens])     # epoch-0 order
    perm = ph.to(dev)
    tc = time.time()
    if stack_li:
        lam64, cal = li2_calibrate(fwd, lambda i: batch(ph, perm, i), a.ft_cal_batches, cfg, ctx, pix, chunk, rho,
                                   snet.K, impl=a.pix_impl, head=snet)
        lam = lam64.float()
        print(f"[li2 calibration] {json.dumps(cal)} ({time.time() - tc:.0f}s)", flush=True)
    lam_s, cal_shape = None, None
    srho = torch.tensor([float(ft_of(m)["shape_rho"]) for m in stack], dtype=torch.float64, device=dev)
    if "shape" in ctx and bool((srho > 0).any()):              # li2ft arm S: second measurement pass, same batches
        ts_ = time.time()
        lam_s64, cal_shape = shape_calibrate(fwd, lambda i: batch(ph, perm, i), a.ft_cal_batches, cfg, ctx, pix, chunk,
                                             srho, snet.K, impl=a.pix_impl, head=snet)
        lam_s = lam_s64.float()
        print(f"[shape calibration] {json.dumps(cal_shape)} ({time.time() - ts_:.0f}s)", flush=True)
    cal_s = time.time() - tc
    shape_acc = []
    step_fn = train_step_ft
    if lam_s is not None:                                      # the arm-S step (shape gradient into the map head only)
        def step_fn(fwd_, params_, opt_, sched_, xb, yb, Sb, b, cfg_, ctx_, wsc_, pix_, chunk_, lam_, ema=None,
                    impl="eager"):
            loss_, ll_, lsh_ = train_step_ft_shape(fwd_, params_, opt_, sched_, xb, yb, Sb, b, cfg_, ctx_, wsc_, pix_,
                                                   chunk_, lam_, lam_s, snet, ema=ema, impl=impl)
            shape_acc.append(lsh_)
            return loss_, ll_
    sb = sb_of(a0)
    sb_acc, sb_last = {}, None
    if sb["wbp"] > 0:                                          # arm ASB: + the Sobolev/B_p objective (train_step_ft_sb)
        sb_rho = torch.tensor([float(sb_of(m)["wbp"]) for m in stack], dtype=torch.float64, device=dev)

        def step_fn(fwd_, params_, opt_, sched_, xb, yb, Sb, b, cfg_, ctx_, wsc_, pix_, chunk_, lam_, ema=None,
                    impl="eager"):
            return train_step_ft_sb(fwd_, params_, opt_, sched_, xb, yb, Sb, b, cfg_, ctx_, wsc_, pix_, chunk_, lam_,
                                    sb, sb_rho, ema=ema, impl=impl, acc=sb_acc)
    if a.bench_steps:
        for i in range(5):
            xb, yb, b = batch(ph, perm, i)
            step_fn(fwd, params, opt, sched, xb, yb, S[b], b, cfg, ctx, wsc, pix, chunk, lam, ema=ema,
                    impl=a.pix_impl)
        torch.cuda.synchronize()
        tb = time.time()
        for i in range(5, 5 + a.bench_steps):
            xb, yb, b = batch(ph, perm, i)
            step_fn(fwd, params, opt, sched, xb, yb, S[b], b, cfg, ctx, wsc, pix, chunk, lam, ema=ema,
                    impl=a.pix_impl)
        torch.cuda.synchronize()
        ms_step = 1e3 * (time.time() - tb) / a.bench_steps
        res = {"tags": tags, "M": M, "bench_steps": a.bench_steps, "ms_per_step": ms_step, "li": bool(stack_li),
               "cal_s": cal_s if stack_li else 0.0, "cal_batches": a.ft_cal_batches if stack_li else 0,
               "ms_per_member_step": ms_step / M, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
               "lambda": lam.tolist() if lam is not None else None, "cal": cal,
               "lambda_shape": lam_s.tolist() if lam_s is not None else None, "cal_shape": cal_shape}
        if sb["wbp"] > 0:
            res["sb"] = sb
        print("BENCH", json.dumps(res), flush=True)
        return res
    print(f"stack {tags}: pool {len(pool)} val frames {len(va)} fin {fin} steps {steps} "
          f"lr {f0['ft_lr_frac'] * a0.lr:.3e} cosine->0 lambda {lam.tolist() if lam is not None else 0}", flush=True)
    nstep = 0
    ts = time.time()
    li_last = [None] * M
    for ep in range(a0.epochs):
        te = time.time()
        if ep > 0:
            ph = torch.stack([pool_t[torch.randperm(len(pool_t), generator=g)] for g in gens])
            perm = ph.to(dev)
        tl = torch.zeros(M, device=dev)
        tli = torch.zeros(M, device=dev)
        nb = 0
        for i in range(len(pool_t) // a0.bs):
            xb, yb, b = batch(ph, perm, i)
            loss, ll = step_fn(fwd, params, opt, sched, xb, yb, S[b], b, cfg, ctx, wsc, pix, chunk, lam,
                               ema=ema, impl=a.pix_impl)
            tl += loss
            if ll is not None:
                tli += ll
            nb += 1
            nstep += 1
        tlc = (tl / nb).cpu().numpy()
        tlic = (tli / nb).cpu().numpy()
        if shape_acc:                                          # li2ft arm S: mean shape term of the epoch
            print(f"shape-term ep {ep} {np.round(torch.stack(shape_acc).mean(0).cpu().numpy(), 6).tolist()}", flush=True)
            shape_acc.clear()
        if sb_acc:                                             # arm ASB: mean SB diagnostics of the epoch
            n = max(sb_acc.pop("n", 1), 1)
            d = {k: (v / n).cpu().numpy() for k, v in sb_acc.items()}
            sb_acc.clear()
            sb_last = {k: [float(x) for x in v] for k, v in d.items()}
            for k, t in enumerate(tags):
                print(f"[{t}] sb ep {ep} Lint {d['l_int'][k]:.4e} Lband {d['l_band'][k]:.4e} "
                      f"cos_psi {d['cos_psi'][k]:+.3f} cos_scal {d['cos_scal'][k]:+.3f} "
                      f"act_psi {d['act_psi'][k]:.3f} act_scal {d['act_scal'][k]:.3f} "
                      f"|gB|/|gpsi| {d['ratio'][k]:.3f} w {d['w'][k]:.3e} | pre-clip |g| {d['gnorm'][k]:.3f} "
                      f"(|gpsi+gscal| {d['gnorm_plain'][k]:.3f}) clipped {d['clip'][k]:.3f}", flush=True)
        if stack_li:
            li_last = [float(x) for x in tlic]
        last = ep == a0.epochs - 1
        if ep % a0.eval_every == a0.eval_every - 1 or last:
            mem = torch.cuda.max_memory_allocated() / 1e9
            if a0.trainval:
                for k, t in enumerate(tags):
                    print(f"[{t}] ep {ep} loss {tlc[k]:.4f} li-term {tlic[k]:.4e} (trainval: no held-out eval)  "
                          f"{time.time() - t0:.0f}s maxmem {mem:.2f}GB", flush=True)
            else:
                e, r2s, _, _ = evaluate()
                for k, t in enumerate(tags):
                    r = r2s[k]
                    print(f"[{t}] ep {ep} loss {tlc[k]:.4f} li-term {tlic[k]:.4e} val psiMSEn {e[k]:.3e} "
                          f"r2 q95 {r[0]:.4f} bN {r[1]:.4f} li {r[2]:.4f} R {r[3]:.4f} Z {r[4]:.4f}  "
                          f"{time.time() - t0:.0f}s maxmem {mem:.2f}GB", flush=True)
        print(f"epoch {ep} {time.time() - te:.1f}s ({1e3 * (time.time() - te) / nb:.1f} ms/step, M={M}, "
              f"lr {sched.get_last_lr()[0]:.3e})", flush=True)
    train_s = time.time() - ts
    mdl = ema_net if ema_net is not None else snet
    P = Q = None
    if not a0.trainval:
        _, _, P, Q = evaluate()
        P, Q = P.cpu(), Q.cpu()
    models = [mdl.member_state_dict(k, bool(m.linskip)) for k, m in enumerate(stack)]
    res = {"tags": tags, "M": M, "train_s": train_s, "steps": nstep, "cal_s": cal_s, "cal": cal,
           "lambda": lam.tolist() if lam is not None else [0.0] * M, "li_last": li_last,
           "lambda_shape": lam_s.tolist() if lam_s is not None else None, "cal_shape": cal_shape,
           "ms_per_step": 1e3 * train_s / max(nstep, 1), "ms_per_member_step": 1e3 * train_s / max(nstep, 1) / M,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9, "models": models, "P": P, "Q": Q}
    if sb["wbp"] > 0:
        res["sb"] = {**sb, "rho": [float(x) for x in sb_rho.tolist()], "last_epoch": sb_last}
    del snet, ema_net, ema, opt
    return res


# ----------------------------------------------------------------------------- MT: aux heads in the li2ft fine-tune
# Agent mtprod (family doc amendment 34, declared deviation; detail doc docs/archive/fec/D3D_MTPROD_20261014.md).
# The mtask trainer (36c45c1) refuses --aux_heads with --ft_from. The "MT" production fine-tunes the aux-head bases
# with the li2ft arm-A recipe and the head losses kept on:
#   * the base's trained aux head (aux_head.pt) is loaded, not re-initialised (a fresh head would push large aux
#     gradients through the trunk in the first steps); its Adam state starts at zero like every fine-tuned parameter;
#   * aux targets, standardisation (TRAIN rows of the run) and weights are the base run's (aux_context / aux_weights);
#     the recomputed train_mu / train_sd / weights / cache md5s must equal the base's aux.json exactly;
#   * train_step_ft_aux = train_step_ft + the masked aux loss of train_step_aux; with lam = None it is train_step_aux
#     (same ops, same order; tested bit for bit), with lam it is AdamW on base + aux + lam * li (tested);
#   * the li(2) lambda calibration sees only (p, q) (the aux head does not enter the psi-output-layer gradients);
#   * only the arm-A fine-tune is built: no --shape_rho, --wbp, --wnorm frame with --aux_heads;
#   * export: model.pt is the plain NetB (inference unchanged), aux_head.pt + aux.json are written next to it.
# With --aux_heads absent every fine-tune line runs the unchanged train_stack_ft.
def ft_aux_bases(a, ms, use_shape):
    """Checks (before --dry returns) that every --ft_from base carries the same aux heads; returns {tag: aux.json}."""
    from fec.paths import WORK
    if use_shape or sb_of(ms[0])["wbp"] > 0 or any(ft_of(m)["wnorm"] != "batch" for m in ms):
        raise SystemExit("--aux_heads with --ft_from is implemented for the li2ft (arm A) fine-tune only "
                         "(no --shape_rho, --wbp, --wnorm frame)")
    if not a.aux_cache or not (Path(a.aux_cache) / "meta.json").exists():
        raise SystemExit(f"--aux_cache {a.aux_cache}: meta.json missing")
    meta = json.loads((Path(a.aux_cache) / "meta.json").read_text())
    out = {}
    for m in ms:
        d = WORK / "models" / m.ft_from
        if not ((d / "aux.json").exists() and (d / "aux_head.pt").exists()):
            raise SystemExit(f"--aux_heads fine-tune: {m.ft_from} has no aux head (aux.json / aux_head.pt)")
        info = json.loads((d / "aux.json").read_text())
        if info.get("heads") != a.aux_heads or float(info.get("aux_w", -1)) != float(a.aux_w):
            raise SystemExit(f"{m.tag}: --aux_heads/--aux_w {a.aux_heads}/{a.aux_w} differ from {m.ft_from}'s "
                             f"{info.get('heads')}/{info.get('aux_w')}")
        if info.get("cache_aux_md5") != meta.get("aux_md5") or info.get("cache_gidx_md5") != meta.get("gidx_md5"):
            raise SystemExit(f"{m.tag}: --aux_cache md5s differ from the cache {m.ft_from} was trained with")
        out[m.tag] = info
    print("aux heads (MT fine-tune):", a.aux_heads, "cache", a.aux_cache, "aux_w", a.aux_w, "bases",
          sorted({m.ft_from for m in ms}), flush=True)
    return out


def ft_aux_context(a, ms, base_aux, rows, tr, va, dev, a0, timing):
    """aux_context of the fine-tune run (the base run's rows/standardisation), checked against each base's aux.json."""
    aux = aux_context(a.aux_cache, a.aux_heads, rows, tr, va, dev)
    w = aux_weights(a.aux_heads, a.aux_w, a0.wscal)
    aux["waux"] = torch.tensor(w, device=dev)
    aux["info"]["weights"] = w
    aux["info"]["aux_w"] = a.aux_w
    for m in ms:
        bi = base_aux[m.tag]
        bad = [k for k in ("train_mu", "train_sd", "weights", "names", "n_rows")
               if json.loads(json.dumps(aux["info"][k])) != bi.get(k)]
        if bad:
            raise SystemExit(f"{m.tag}: aux {bad} differ from {m.ft_from}'s aux.json (the base's targets must be kept)")
    timing["aux"] = aux["info"]
    print("aux targets (MT fine-tune):", json.dumps({k: aux["info"][k] for k in ("heads", "n_rows", "val_present",
                                                                                  "weights")}), flush=True)
    return aux


def load_aux_states(snet, states):
    """Copy saved aux heads ({'aux.weight' (n, w), 'aux.bias' (n,)} per member, aux_state's layout) into snet."""
    with torch.no_grad():
        for m, s in enumerate(states):
            W, b = s["aux.weight"], s["aux.bias"]
            assert tuple(W.shape) == (snet.n_aux, snet.w) and tuple(b.shape) == (snet.n_aux,), (W.shape, b.shape)
            snet.aux_w[m].copy_(W.T)
            snet.aux_b[m].copy_(b)
    return snet


def train_step_ft_aux(snet, params, opt, sched, xb, yb, Sb, Ab, Mb, b, cfg, ctx, wsc, waux, pix, chunk, lam,
                      amp_dtype=torch.bfloat16, ema=None, impl="eager"):
    """train_step_ft with the mtask aux loss: g_p = g_base + lam_m g_li (lam None = off) and the masked aux loss of
    train_step_aux in the scalar backward. lam None: exactly train_step_aux. Returns (base loss (M,), li term (M,) or
    None, aux loss (M,))."""
    M = xb.shape[0]
    with torch.autocast(xb.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        p, q, a = snet(xb)
    p32 = p.float()
    lp, gp = psi_loss_and_grad(p32, yb.float(), b, cfg, ctx, pix, chunk, impl)
    ll = None
    if lam is not None:
        ll, gl = li2_loss_and_grad(p32, b, ctx, chunk)
        gp = gp + lam.view(M, 1, 1).to(gp.dtype) * gl
    lq = (((q.float() - Sb) ** 2).mean(1) * wsc).sum(1)
    la = ((((a.float() - Ab) ** 2) * Mb).sum(1) / Mb.sum(1).clamp_min(1.0) * waux).sum(1)
    opt.zero_grad(set_to_none=True)
    torch.autograd.backward([p32, (lq + la).sum()], [gp, None])
    clip_grad_norm_per_member(params, M, 1.0)
    opt.step()
    if sched is not None:
        sched.step()
    if ema is not None:
        ema.update(params)
    return (lp + lq).detach(), ll, la.detach()


def train_stack_ft_aux(stack, cks, a, Xg, Y, Yva, S, va, pool, ctx, wsc, steps, pix, dev, t0, use_li, aux):
    """train_stack_ft (arm A path: no shape / SB terms) on StackedNetBAux with each base's trained aux head loaded;
    same nets, RNG streams, schedule, calibration batches and export as train_stack_ft."""
    from fec.paths import WORK
    a0 = stack[0]
    f0 = ft_of(a0)
    M = len(stack)
    fin = Xg.shape[1]
    nets = [ft_net(m, cks[m.tag], fin) for m in stack]
    gens = [torch.Generator().manual_seed(int(m.seed)) for m in stack]
    ngens = []
    if a0.noise > 0:
        for m in stack:
            ng = torch.Generator(device=dev)
            ng.manual_seed(int(m.seed))
            ngens.append(ng)
    states = [torch.load(WORK / "models" / m.ft_from / "aux_head.pt", map_location="cpu", weights_only=False)
              for m in stack]
    snet = StackedNetBAux.from_nets(nets).init_aux(aux["n"], [m.seed for m in stack])
    snet = load_aux_states(snet, states).to(dev)
    del nets
    snet.ln_impl = a.ln_impl
    fwd = snet
    fwd_pq = lambda x: fwd(x)[:2]  # noqa: E731   (calibration / evaluation see the plain (p, q))
    params = list(snet.parameters())
    ema = StackedEMA(snet, a0.ema) if a0.ema > 0 else None
    ema_net = ema.module if ema is not None else None
    opt = torch.optim.AdamW(params, lr=f0["ft_lr_frac"] * a0.lr, weight_decay=a0.wd, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_factor(steps))
    cfg = cfg_of(stack)
    chunk = a.pix_chunk or M
    rho = torch.tensor([float(ft_of(m)["li2_rho"]) for m in stack], dtype=torch.float64, device=dev)
    stack_li = use_li and bool((rho > 0).any())
    A, Mk, waux = aux["A"], aux["M"], aux["waux"]
    pool_t = torch.as_tensor(pool)
    va_t = torch.as_tensor(va, device=dev)
    va_x = va_t if a.data_gpu else torch.as_tensor(va)
    batcher = None if a.data_gpu else HostBatcher(Xg, Y, M, a0.bs, dev)
    Sva = S[va_t]
    tags = [m.tag for m in stack]
    torch.cuda.reset_peak_memory_stats()

    def evaluate():
        mdl = ema_net if ema_net is not None else snet
        mdl.eval()
        P, Q = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(va_t), 16384):
                xb = Xg[va_x[i:i + 16384]].to(dev).unsqueeze(0).expand(M, -1, -1)
                p, q = mdl(xb)[:2]
                P.append(p.float())
                Q.append(q.float())
        mdl.train()
        P = torch.cat(P, 1)
        Q = torch.cat(Q, 1)
        e_psi = ((P - Yva) ** 2).sum(-1).mean(-1) / 4225
        r2s = 1 - ((Q - Sva) ** 2).mean(1) / Sva.var(0)
        return e_psi.cpu().numpy(), r2s.cpu().numpy(), P, Q

    def batch(ph, perm, i):
        bh, b = ph[:, i * a0.bs:(i + 1) * a0.bs], perm[:, i * a0.bs:(i + 1) * a0.bs]
        if batcher is None:
            xb, yb = Xg[b], Y[b]
        else:
            xb, yb = batcher.get(bh)
        if a0.noise > 0:
            xb = xb + a0.noise * torch.stack([torch.randn(xb.shape[1:], generator=g, device=dev) for g in ngens])
        return xb, yb, b

    def step_fn(xb, yb, b, lam_):
        return train_step_ft_aux(fwd, params, opt, sched, xb, yb, S[b], A[b], Mk[b], b, cfg, ctx, wsc, waux, pix,
                                 chunk, lam_, ema=ema, impl=a.pix_impl)

    lam = None
    cal = None
    ph = torch.stack([pool_t[torch.randperm(len(pool_t), generator=g)] for g in gens])     # epoch-0 order
    perm = ph.to(dev)
    tc = time.time()
    if stack_li:
        lam64, cal = li2_calibrate(fwd_pq, lambda i: batch(ph, perm, i), a.ft_cal_batches, cfg, ctx, pix, chunk, rho,
                                   snet.K, impl=a.pix_impl, head=snet)
        lam = lam64.float()
        print(f"[li2 calibration] {json.dumps(cal)} ({time.time() - tc:.0f}s)", flush=True)
    cal_s = time.time() - tc
    if a.bench_steps:
        for i in range(5):
            xb, yb, b = batch(ph, perm, i)
            step_fn(xb, yb, b, lam)
        torch.cuda.synchronize()
        tb = time.time()
        for i in range(5, 5 + a.bench_steps):
            xb, yb, b = batch(ph, perm, i)
            step_fn(xb, yb, b, lam)
        torch.cuda.synchronize()
        ms_step = 1e3 * (time.time() - tb) / a.bench_steps
        res = {"tags": tags, "M": M, "bench_steps": a.bench_steps, "ms_per_step": ms_step, "li": bool(stack_li),
               "cal_s": cal_s if stack_li else 0.0, "cal_batches": a.ft_cal_batches if stack_li else 0,
               "ms_per_member_step": ms_step / M, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
               "lambda": lam.tolist() if lam is not None else None, "cal": cal, "lambda_shape": None,
               "cal_shape": None, "aux": aux["info"]["heads"]}
        print("BENCH", json.dumps(res), flush=True)
        return res
    print(f"stack {tags}: pool {len(pool)} val frames {len(va)} fin {fin} steps {steps} "
          f"lr {f0['ft_lr_frac'] * a0.lr:.3e} cosine->0 lambda {lam.tolist() if lam is not None else 0} aux "
          f"{aux['info']['heads']} ({aux['n']} columns, base heads loaded)", flush=True)
    nstep = 0
    ts = time.time()
    li_last = [None] * M
    tlac = np.zeros(M)
    for ep in range(a0.epochs):
        te = time.time()
        if ep > 0:
            ph = torch.stack([pool_t[torch.randperm(len(pool_t), generator=g)] for g in gens])
            perm = ph.to(dev)
        tl = torch.zeros(M, device=dev)
        tli = torch.zeros(M, device=dev)
        tla = torch.zeros(M, device=dev)
        nb = 0
        for i in range(len(pool_t) // a0.bs):
            xb, yb, b = batch(ph, perm, i)
            loss, ll, la = step_fn(xb, yb, b, lam)
            tl += loss
            tla += la
            if ll is not None:
                tli += ll
            nb += 1
            nstep += 1
        tlc = (tl / nb).cpu().numpy()
        tlic = (tli / nb).cpu().numpy()
        tlac = (tla / nb).cpu().numpy()
        if stack_li:
            li_last = [float(x) for x in tlic]
        last = ep == a0.epochs - 1
        if ep % a0.eval_every == a0.eval_every - 1 or last:
            mem = torch.cuda.max_memory_allocated() / 1e9
            if a0.trainval:
                for k, t in enumerate(tags):
                    print(f"[{t}] ep {ep} loss {tlc[k]:.4f} li-term {tlic[k]:.4e} aux {tlac[k]:.4f} (trainval: no "
                          f"held-out eval)  {time.time() - t0:.0f}s maxmem {mem:.2f}GB", flush=True)
            else:
                e, r2s, _, _ = evaluate()
                for k, t in enumerate(tags):
                    r = r2s[k]
                    print(f"[{t}] ep {ep} loss {tlc[k]:.4f} li-term {tlic[k]:.4e} aux {tlac[k]:.4f} val psiMSEn "
                          f"{e[k]:.3e} r2 q95 {r[0]:.4f} bN {r[1]:.4f} li {r[2]:.4f} R {r[3]:.4f} Z {r[4]:.4f}  "
                          f"{time.time() - t0:.0f}s maxmem {mem:.2f}GB", flush=True)
        print(f"epoch {ep} {time.time() - te:.1f}s ({1e3 * (time.time() - te) / nb:.1f} ms/step, M={M}, "
              f"lr {sched.get_last_lr()[0]:.3e})", flush=True)
    train_s = time.time() - ts
    mdl = ema_net if ema_net is not None else snet
    P = Q = None
    if not a0.trainval:
        _, _, P, Q = evaluate()
        P, Q = P.cpu(), Q.cpu()
    models = [mdl.member_state_dict(k, bool(m.linskip)) for k, m in enumerate(stack)]
    res = {"tags": tags, "M": M, "train_s": train_s, "steps": nstep, "cal_s": cal_s, "cal": cal,
           "lambda": lam.tolist() if lam is not None else [0.0] * M, "li_last": li_last,
           "lambda_shape": None, "cal_shape": None,
           "ms_per_step": 1e3 * train_s / max(nstep, 1), "ms_per_member_step": 1e3 * train_s / max(nstep, 1) / M,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9, "models": models, "P": P, "Q": Q,
           "aux_states": [mdl.aux_state(k) for k in range(M)],
           "aux_last_train_loss": {t: float(tlac[k]) for k, t in enumerate(tags)}}
    del snet, ema_net, ema, opt
    return res


def save_aux_ft(m, k, st, aux, WORK):
    """aux.json + aux_head.pt of a fine-tuned MT member (as save_aux; the head continues the base's)."""
    out = WORK / "models" / m.tag
    torch.save(st["aux_states"][k], out / "aux_head.pt")
    info = dict(aux["info"])
    info.update({"ft_from": m.ft_from, "aux_head_init": f"models/{m.ft_from}/aux_head.pt (loaded, not re-initialised)",
                 "trainer": "train_d3d_bx (mtask aux heads + li2ft fine-tune; agent mtprod, amendment 34)",
                 "val_r2": None, "last_train_aux_loss": st["aux_last_train_loss"][m.tag],
                 "note": "training-only head; model.pt is the plain NetB (inference unchanged)"})
    (out / "aux.json").write_text(json.dumps(info, indent=1, default=float))


# ----------------------------------------------------------------------------- members / data
def member_argvs(a):
    argvs = [shlex.split(s) for s in (a.member or [])]
    if a.jobs:
        for line in json.loads(Path(a.jobs).read_text()):
            toks = shlex.split(line)
            i = next(k for k, t in enumerate(toks) if t.replace("\\", "/").endswith("fec/train_d3d_b.py"))
            rest = []
            for t in toks[i + 1:]:
                if t.startswith(">") or t in ("2>&1", "|", "&&", ";"):
                    break
                rest.append(t)
            argvs.append(rest)
    for rep in a.tag_replace or []:
        old, new = rep.split("=", 1)
        for v in argvs:
            k = v.index("--tag")
            v[k + 1] = v[k + 1].replace(old, new)
    return argvs


def parse_members(argvs):
    ms = [member_get_args(v) for v in argvs]
    if not ms:
        raise SystemExit("no members (--member / --jobs)")
    tags = [m.tag for m in ms]
    if len(set(tags)) != len(tags):
        raise SystemExit(f"duplicate member tags: {tags}")
    for k in SHARED:
        vals = {json.dumps(getattr(m, k)) for m in ms}
        if len(vals) > 1:
            raise SystemExit(f"members must share --{k}: {sorted(vals)}")
    for m in ms:
        if m.tsdrop > 0:
            raise SystemExit("--tsdrop is not supported by train_d3d_bx")
        if m.feat not in ("base", "nots") and not (is_ft(m) and m.feat in ("base_gap", "nots_gap")):
            raise SystemExit(f"--feat {m.feat} is not supported by train_d3d_bx (base, nots; base_gap/nots_gap "
                             "with --ft_from)")
        if m.alldata:
            m.trainval = True                   # as train_d3d_b
    pix = {m.wband > 0 or m.win > 0 or m.wgrad > 0 or m.wx > 0 for m in ms}
    if len(pix) > 1:
        raise SystemExit("members mix pixel and plain PCA losses; run them separately")
    sbs = {json.dumps(sb_of(m), sort_keys=True) for m in ms}
    if len(sbs) > 1:
        raise SystemExit(f"members must share the Sobolev/B_p flags (--wbp --bp_sig --bp_band --pcgrad): {sorted(sbs)}")
    sb = sb_of(ms[0])
    if sb["wbp"] < 0 or (sb["pcgrad"] and sb["wbp"] <= 0):
        raise SystemExit("--pcgrad needs --wbp > 0; --wbp must be >= 0")
    check_ft_members(ms)                      # li2ft fine-tune (no-op for runs without --ft_from)
    pix = pix.pop()
    if sb["wbp"] > 0 and not pix:
        raise SystemExit("--wbp needs pixel-loss members (bnd2 / TRAIN envelope)")
    return ms, pix


def cfg_of(ms):
    return {k: [getattr(m, k) for m in ms] for k in ("wband", "sig", "win", "wx", "rx", "wli", "wgrad")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", action="append", help="train_d3d_b argv string (repeatable)")
    ap.add_argument("--jobs", help="pod job list JSON (shell lines calling fec/train_d3d_b.py)")
    ap.add_argument("--tag_replace", action="append", help="OLD=NEW substring replacement in tags")
    ap.add_argument("--group_size", type=int, default=0, help="members per stack (0 = all of a feat)")
    ap.add_argument("--pix_chunk", type=int, default=0, help="members per pixel-loss chunk (0 = stack)")
    ap.add_argument("--x_bf16", type=int, default=1, help="keep inputs on GPU as bf16 (exact under "
                    "autocast; ignored when --noise > 0)")
    ap.add_argument("--data_gpu", type=int, default=1, help="inputs/targets resident on the GPU "
                    "(0: pinned host memory, gathered per step into pinned buffers + async copy)")
    ap.add_argument("--pix_impl", default="eager", choices=["eager", "fused", "compile"],
                    help="pixel loss: in-place train_d3d_b form, out-of-place fused form, or "
                    "torch.compile of the fused form (Linux/Triton)")
    ap.add_argument("--compile_net", action="store_true", help="torch.compile the whole stacked "
                    "network (slower than --ln_impl compile on the H100, 2026-10-10)")
    ap.add_argument("--ln_impl", default="eager", choices=["eager", "compile"],
                    help="LayerNorm+affine(+bf16 cast): eager, or one torch.compile'd fused kernel")
    ap.add_argument("--y_dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="storage dtype of the PCA targets (loss always in fp32)")
    ap.add_argument("--tf32", action="store_true", help="TF32 for the fp32 pixel-loss matmuls")
    ap.add_argument("--vram_gb", type=float, default=8.0, help="caching-allocator cap (GB)")
    ap.add_argument("--bench_steps", type=int, default=0)
    ap.add_argument("--timing_json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry", action="store_true", help="parse members, print stacks, exit")
    ap.add_argument("--li2_cache", help="li2ft fine-tune: li(2) target/contour cache dir (fec/li2_torch.py build)")
    ap.add_argument("--li2_cache_dev", default="auto", choices=["auto", "cuda", "cpu", "mmap"],
                    help="where the li2 cache lives: cuda, pinned cpu RAM, or the memory map read per step "
                    "(auto: cuda when --data_gpu 1, else mmap)")
    ap.add_argument("--shape_cache", help="li2ft arm S: true-geometry shape cache dir (fec/li2_shape.py build)")
    ap.add_argument("--ft_cal_batches", type=int, default=100, help="li2ft lambda calibration batches "
                    "(pre-declared 100)")
    ap.add_argument("--aux_heads", default=None, choices=["shape", "shape_jphi"],
                    help="mtask (amendment 22): training-only auxiliary heads; default OFF")
    ap.add_argument("--aux_cache", help="mtask aux target cache dir (fec/mtask_aux.py build)")
    ap.add_argument("--aux_w", type=float, default=1.0, help="aux head weight in units of wscal (pre-declared 1.0)")
    a = ap.parse_args(argv)
    ms, pix = parse_members(member_argvs(a))
    if a.aux_heads is not None:               # mtask: refuse the combinations it was not built for
        if not a.aux_cache:
            raise SystemExit("--aux_heads needs --aux_cache")
        if sb_of(ms[0])["wbp"] > 0 or a.compile_net:   # agent mtprod (amendment 34): --ft_from allowed (li2ft)
            raise SystemExit("--aux_heads is implemented for the plain train_step and the li2ft fine-tune paths "
                             "only (no --wbp, --compile_net)")
        if a.aux_w <= 0:
            raise SystemExit("--aux_w must be > 0")
    if is_ft(ms[0]):                          # li2ft fine-tune mode (opt-in; members with --ft_from)
        return finetune_main(a, ms, pix)
    a0 = ms[0]
    feats = [f for f in ("base", "nots") if any(m.feat == f for m in ms)]
    stacks = []
    for f in feats:
        fm = [m for m in ms if m.feat == f]
        G = a.group_size or len(fm)
        stacks += [fm[i:i + G] for i in range(0, len(fm), G)]
    print("stacks:", [[m.tag for m in s] for s in stacks], "pix", pix, flush=True)
    if a.aux_heads is not None:
        print("aux heads:", a.aux_heads, "cache", a.aux_cache, "aux_w", a.aux_w, flush=True)
    if a.dry:
        return

    from fec.features import load_ds
    from fec.features_b import d3d_inputs_b
    from fec.paths import WORK
    from fec.train_d3d_b import pca_basis, project_rows
    dev = "cuda"
    for m in ms:
        out = WORK / "models" / m.tag
        if out.exists() and any(out.iterdir()) and not a.overwrite and not a.bench_steps:
            raise SystemExit(f"refusing to overwrite existing model dir {out} (use --overwrite)")
    if a0.alldata and not (WORK / "test_eval_b" / "DONE").exists():
        raise SystemExit("--alldata refused: WORK/test_eval_b/DONE missing (run the one-time "
                         "internal-TEST evaluation first, or copy its DONE marker)")
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    tot_mem = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / tot_mem))
    timing = {"members": [m.tag for m in ms], "stacks": [], "argv": sys.argv[1:] if argv is None else argv}
    t0 = time.time()

    # ---- shared preparation (as train_d3d_b.main: fits on the training rows tr, all rows with --alldata) --
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])[ds["shot"]]
    if a0.alldata:
        rows = np.arange(len(splits))
    else:
        rows = np.flatnonzero((splits == "train") | (splits == "val"))   # never 'test' rows
    is_tr = (splits[rows] == "train") | a0.trainval
    tr = np.flatnonzero(is_tr)
    va = np.flatnonzero(splits[rows] == "val")
    fit = (splits == "train") | ((splits == "val") & a0.trainval) | a0.alldata
    mean, V, ev = pca_basis(ds["psi"], rows[tr], a0.K)
    tot = ev.clamp_min(0).sum().item()
    print("PCA unexplained var frac", {k: 1 - ev[:k].sum().item() / tot for k in (64, 128, 256)},
          flush=True)
    Y = project_rows(ds["psi"], rows, mean, V)
    ds["psi"] = None
    ysd = math.sqrt(tot / 4225)
    Y = Y / ysd                                           # targets (N, K) fp32
    va_d = torch.as_tensor(va, device=dev)
    Yva = Y[va_d].clone()
    if a.y_dtype != "fp32":
        ydt = {"fp16": torch.float16, "bf16": torch.bfloat16}[a.y_dtype]
        ytr = Y[torch.as_tensor(tr[::10], device=dev)]
        qe = ((ytr - ytr.to(ydt).float()) ** 2).sum(1).mean() / 4225
        del ytr
        timing["y_quant_psiMSEn"] = qe.item()
        print(f"y_dtype {a.y_dtype}: target rounding psiMSEn (normalised, TRAIN) {qe.item():.3e}", flush=True)
        Y = Y.to(ydt)
    if not a.data_gpu:                                    # host-resident targets (VRAM budget)
        Y = Y.cpu().pin_memory()
        torch.cuda.empty_cache()
    scal = torch.as_tensor(ds["scal"][rows], device=dev)
    tr_d = torch.as_tensor(tr, device=dev)
    smu = scal[tr_d].mean(0); ssd = scal[tr_d].std(0)
    S = (scal - smu) / ssd
    ctx = {"li": torch.nan_to_num(scal[:, 2].float(), nan=1.0)}
    if pix:
        sfx = "_all" if a0.alldata else ""
        ctx["bnd"] = torch.as_tensor(np.load(WORK / "ds_b" / "diii_d_train" / f"bnd2{sfx}.npy")[rows],
                                     device=dev)
        assert torch.isfinite(ctx["bnd"][tr_d]).all()
        mk = np.load(WORK / "ds_b" / "diii_d_train" / "env_train.npz")       # TRAIN-only envelope
        ctx["maskc"] = torch.as_tensor(mk["env"].astype(np.float32).reshape(-1), device=dev)
        ctx["mean"] = mean
        ctx["VT"] = V.T.contiguous()
        ctx["VTs"] = ctx["VT"] * ysd
        RRg, ZZg = np.meshgrid(mk["grid_R"], mk["grid_Z"])
        ctx["RRg"] = torch.as_tensor(RRg.reshape(-1), device=dev, dtype=torch.float32)
        ctx["ZZg"] = torch.as_tensor(ZZg.reshape(-1), device=dev, dtype=torch.float32)
        if any(m.wx > 0 for m in ms):
            ctx["bx"] = torch.as_tensor(np.load(WORK / "ds_b" / "diii_d_train" / f"bnd_x{sfx}.npy")[rows],
                                        device=dev)
            assert torch.isfinite(ctx["bx"][tr_d]).all()
        if sb_of(a0)["wbp"] > 0:                                              # Sobolev/B_p constants
            ctx.update(bp_context(mk["grid_R"], mk["grid_Z"], dev))
            print("B_p loss:", json.dumps(sb_of(a0)), flush=True)
    timing["prep_targets_s"] = time.time() - t0
    print("targets ready", f"{time.time() - t0:.0f}s", flush=True)
    wsc = torch.tensor([1.0, 1.0, 0.3, 0.3, 0.3], device=dev) * a0.wscal
    steps = a0.epochs * (len(tr) // a0.bs)
    aux = None
    if a.aux_heads is not None:                       # mtask: standardised aux targets at `rows` (TRAIN-fit)
        aux = aux_context(a.aux_cache, a.aux_heads, rows, tr, va, dev)
        aux["waux"] = torch.tensor(aux_weights(a.aux_heads, a.aux_w, a0.wscal), device=dev)
        aux["info"]["weights"] = aux_weights(a.aux_heads, a.aux_w, a0.wscal)
        aux["info"]["aux_w"] = a.aux_w
        timing["aux"] = aux["info"]
        print("aux targets:", json.dumps({k: aux["info"][k] for k in ("heads", "n_rows", "val_present", "weights")}),
              flush=True)
    gt_va = None
    x_bf16 = bool(a.x_bf16) and a0.noise == 0

    cur_feat = None
    for stack in stacks:
        feat = stack[0].feat
        if feat != cur_feat:                          # this feature set's inputs (fits on tr: all rows if --alldata)
            tf = time.time()
            Xg = None; torch.cuda.empty_cache()
            X, _, stats = d3d_inputs_b(ds, "diii_d_train", feat=feat, fit_mask=fit)
            X = X[rows]
            xmu = X[tr].mean(0); xsd = X[tr].std(0) + 1e-4
            Xn = torch.as_tensor((X - xmu) / xsd)
            del X
            Xg = Xn.to(dtype=torch.bfloat16 if x_bf16 else torch.float32)
            Xg = Xg.to(dev) if a.data_gpu else Xg.pin_memory()
            del Xn
            cur_feat = feat
            timing.setdefault("prep_inputs_s", {})[feat] = time.time() - tf
            print("inputs", feat, tuple(Xg.shape), Xg.dtype, f"{time.time() - t0:.0f}s", flush=True)
        if aux is None:
            st = train_stack(stack, a, Xg, Y, Yva, S, va, tr, ctx, wsc, steps, pix, dev, t0)
        else:
            st = train_stack_aux(stack, a, Xg, Y, Yva, S, va, tr, ctx, wsc, steps, pix, dev, t0, aux)
        if a.bench_steps:
            st["projected_train_s"] = st["ms_per_step"] * steps / 1e3
            timing.setdefault("bench", []).append(st)
            torch.cuda.empty_cache()
            continue
        ts = time.time()
        if not a0.trainval and gt_va is None:
            psi_mm = np.load(WORK / "ds" / "diii_d_train" / "psi.npy", mmap_mode="r")
            gt_va = np.asarray(psi_mm[rows[va]], dtype=np.float32).reshape(len(va), -1)
            del psi_mm
        for k, m in enumerate(stack):
            save_member(m, st["models"][k], WORK, xmu, xsd, stats, mean, V, ysd, smu, ssd,
                        Xg.shape[1], rows, va, st["P"][k] if st["P"] is not None else None,
                        st["Q"][k] if st["Q"] is not None else None, gt_va, t0)
            if aux is not None:
                save_aux(m, k, st, aux, WORK)
        st["save_s"] = time.time() - ts
        for k in ("models", "P", "Q", "aux_states"):
            st.pop(k, None)
        timing["stacks"].append(st)
        torch.cuda.empty_cache()
    timing["total_s"] = time.time() - t0
    if a.bench_steps:
        pt = sum(b["projected_train_s"] for b in timing["bench"])
        timing["projected_train_s_all_stacks"] = pt
        timing["steps_per_member"] = steps
        print("BENCH_PROJECTION", json.dumps({"stacks": [[b["M"], round(b["ms_per_step"], 1)] for b in timing["bench"]],
                                              "steps": steps, "projected_train_s": round(pt),
                                              "prep_targets_s": round(timing["prep_targets_s"]),
                                              "prep_inputs_s": timing.get("prep_inputs_s")}), flush=True)
    print("BX_DONE", f"total {timing['total_s']:.0f}s", flush=True)
    if a.timing_json:
        Path(a.timing_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.timing_json).write_text(json.dumps(timing, indent=1, default=float))


def train_stack(stack, a, Xg, Y, Yva, S, va, tr, ctx, wsc, steps, pix, dev, t0):
    a0 = stack[0]
    M = len(stack)
    fin = Xg.shape[1]
    nets, gens, ngens = [], [], []
    for m in stack:                                    # exactly train_d3d_b's RNG sequence
        torch.manual_seed(m.seed)
        nets.append(build_net(vars(m), fin, NSCAL))
        g = torch.Generator(); g.set_state(torch.get_rng_state())
        gens.append(g)
        if a0.noise > 0:
            ng = torch.Generator(device=dev); ng.manual_seed(m.seed)
            ngens.append(ng)
    snet = StackedNetB.from_nets(nets).to(dev)
    del nets
    snet.ln_impl = a.ln_impl
    fwd = torch.compile(snet) if a.compile_net else snet
    params = list(snet.parameters())
    ema = StackedEMA(snet, a0.ema) if a0.ema > 0 else None
    ema_net = ema.module if ema is not None else None
    opt = torch.optim.AdamW(params, lr=a0.lr, weight_decay=a0.wd, fused=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a0.lr, total_steps=steps,
                                                pct_start=0.05, final_div_factor=1e3)
    cfg = cfg_of(stack)
    chunk = a.pix_chunk or M
    sb = sb_of(a0)                                     # shared by all members (parse_members)
    use_sb = sb["wbp"] > 0
    rho = (torch.tensor([float(sb_of(m)["wbp"]) for m in stack], dtype=torch.float64, device=dev)
           if use_sb else None)
    sb_acc = {}
    tr_t = torch.as_tensor(tr)
    va_t = torch.as_tensor(va, device=dev)
    va_x = va_t if a.data_gpu else torch.as_tensor(va)
    batcher = None if a.data_gpu else HostBatcher(Xg, Y, M, a0.bs, dev)
    Sva = S[va_t]
    tags = [m.tag for m in stack]
    torch.cuda.reset_peak_memory_stats()
    ts = time.time()

    def evaluate():
        mdl = ema_net if ema_net is not None else snet
        mdl.eval()
        P, Q = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(va_t), 16384):
                xb = Xg[va_x[i:i + 16384]].to(dev).unsqueeze(0).expand(M, -1, -1)
                p, q = mdl(xb)
                P.append(p.float()); Q.append(q.float())
        mdl.train()
        P = torch.cat(P, 1); Q = torch.cat(Q, 1)
        e_psi = ((P - Yva) ** 2).sum(-1).mean(-1) / 4225
        r2s = 1 - ((Q - Sva) ** 2).mean(1) / Sva.var(0)
        return e_psi.cpu().numpy(), r2s.cpu().numpy(), P, Q

    def one_step(bh, b):
        if batcher is None:
            xb, yb = Xg[b], Y[b]
        else:
            xb, yb = batcher.get(bh)
        if a0.noise > 0:
            xb = xb + a0.noise * torch.stack([torch.randn(xb.shape[1:], generator=g, device=dev)
                                              for g in ngens])
        if use_sb:
            return train_step_sb(fwd, params, opt, sched, xb, yb, S[b], b, cfg, ctx, wsc, pix, chunk, sb, rho,
                                 ema=ema, impl=a.pix_impl, acc=sb_acc)
        return train_step(fwd, params, opt, sched, xb, yb, S[b], b, cfg, ctx, wsc, pix, chunk,
                          ema=ema, impl=a.pix_impl)

    if a.bench_steps:
        ph = torch.stack([tr_t[torch.randperm(len(tr_t), generator=g)] for g in gens])
        perm = ph.to(dev)
        tw = time.time()
        for i in range(5):
            one_step(ph[:, i * a0.bs:(i + 1) * a0.bs], perm[:, i * a0.bs:(i + 1) * a0.bs])
        torch.cuda.synchronize(); tb = time.time()
        for i in range(5, 5 + a.bench_steps):
            one_step(ph[:, i * a0.bs:(i + 1) * a0.bs], perm[:, i * a0.bs:(i + 1) * a0.bs])
        torch.cuda.synchronize()
        ms_step = 1e3 * (time.time() - tb) / a.bench_steps
        res = {"tags": tags, "M": M, "bench_steps": a.bench_steps, "ms_per_step": ms_step,
               "warmup_s": tb - tw,
               "ms_per_member_step": ms_step / M,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
        print("BENCH", json.dumps(res), flush=True)
        return res

    print(f"stack {tags}: train frames {len(tr)} val frames {len(va)} fin {fin} steps {steps}",
          flush=True)
    nstep = 0
    for ep in range(a0.epochs):
        te = time.time()
        ph = torch.stack([tr_t[torch.randperm(len(tr_t), generator=g)] for g in gens])
        perm = ph.to(dev)
        tl = torch.zeros(M, device=dev); nb = 0
        for i in range(0, len(tr_t) - a0.bs + 1, a0.bs):
            tl += one_step(ph[:, i:i + a0.bs], perm[:, i:i + a0.bs]); nb += 1; nstep += 1
        tlc = (tl / nb).cpu().numpy()
        last = ep == a0.epochs - 1
        if use_sb and (ep == 0 or ep % a0.eval_every == a0.eval_every - 1 or last):
            n = max(sb_acc.pop("n", 1), 1)
            d = {k: (v / n).cpu().numpy() for k, v in sb_acc.items()}
            for k, t in enumerate(tags):
                print(f"[{t}] sb ep {ep} Lint {d['l_int'][k]:.4e} Lband {d['l_band'][k]:.4e} "
                      f"cos_psi {d['cos_psi'][k]:+.3f} cos_scal {d['cos_scal'][k]:+.3f} "
                      f"act_psi {d['act_psi'][k]:.3f} act_scal {d['act_scal'][k]:.3f} "
                      f"|gB|/|gpsi| {d['ratio'][k]:.3f} w {d['w'][k]:.3e} | pre-clip |g| {d['gnorm'][k]:.3f} "
                      f"(|gpsi+gscal| {d['gnorm_plain'][k]:.3f}) clipped {d['clip'][k]:.3f}", flush=True)
        if use_sb:
            sb_acc.clear()
        if ep % a0.eval_every == a0.eval_every - 1 or last:
            mem = torch.cuda.max_memory_allocated() / 1e9
            if a0.trainval:
                for k, t in enumerate(tags):
                    print(f"[{t}] ep {ep} loss {tlc[k]:.4f} (trainval: no held-out eval)  "
                          f"{time.time() - t0:.0f}s maxmem {mem:.2f}GB", flush=True)
            else:
                e, r2s, _, _ = evaluate()
                for k, t in enumerate(tags):
                    r = r2s[k]
                    print(f"[{t}] ep {ep} loss {tlc[k]:.4f} val psiMSEn {e[k]:.3e} "
                          f"r2 q95 {r[0]:.4f} bN {r[1]:.4f} li {r[2]:.4f} R {r[3]:.4f} "
                          f"Z {r[4]:.4f}  {time.time() - t0:.0f}s maxmem {mem:.2f}GB", flush=True)
        print(f"epoch {ep} {time.time() - te:.1f}s ({1e3 * (time.time() - te) / nb:.1f} ms/step, "
              f"M={M})", flush=True)
    train_s = time.time() - ts
    mdl = ema_net if ema_net is not None else snet
    P = Q = None
    if not a0.trainval:
        _, _, P, Q = evaluate()
        P, Q = P.cpu(), Q.cpu()
    models = [mdl.member_state_dict(k, bool(m.linskip)) for k, m in enumerate(stack)]
    res = {"tags": tags, "M": M, "train_s": train_s, "steps": nstep,
           "ms_per_step": 1e3 * train_s / max(nstep, 1), "ms_per_member_step": 1e3 * train_s / max(nstep, 1) / M,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9, "models": models, "P": P, "Q": Q}
    del snet, ema_net, ema, opt
    return res


def train_stack_aux(stack, a, Xg, Y, Yva, S, va, tr, ctx, wsc, steps, pix, dev, t0, aux):
    """train_stack with the mtask auxiliary head (StackedNetBAux + train_step_aux); same RNG sequence, schedule,
    evaluation and export (the plain NetB). The VAL evaluation also reports the aux heads' R^2 (masked rows)."""
    a0 = stack[0]
    M = len(stack)
    fin = Xg.shape[1]
    nets, gens, ngens = [], [], []
    for m in stack:                                    # exactly train_d3d_b's RNG sequence (as train_stack)
        torch.manual_seed(m.seed)
        nets.append(build_net(vars(m), fin, NSCAL))
        g = torch.Generator(); g.set_state(torch.get_rng_state())
        gens.append(g)
        if a0.noise > 0:
            ng = torch.Generator(device=dev); ng.manual_seed(m.seed)
            ngens.append(ng)
    snet = StackedNetBAux.from_nets(nets).init_aux(aux["n"], [m.seed for m in stack]).to(dev)
    del nets
    snet.ln_impl = a.ln_impl
    fwd = snet
    params = list(snet.parameters())
    ema = StackedEMA(snet, a0.ema) if a0.ema > 0 else None
    ema_net = ema.module if ema is not None else None
    opt = torch.optim.AdamW(params, lr=a0.lr, weight_decay=a0.wd, fused=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a0.lr, total_steps=steps,
                                                pct_start=0.05, final_div_factor=1e3)
    cfg = cfg_of(stack)
    chunk = a.pix_chunk or M
    A, Mk, waux = aux["A"], aux["M"], aux["waux"]
    names = aux["info"]["names"]
    tr_t = torch.as_tensor(tr)
    va_t = torch.as_tensor(va, device=dev)
    va_x = va_t if a.data_gpu else torch.as_tensor(va)
    batcher = None if a.data_gpu else HostBatcher(Xg, Y, M, a0.bs, dev)
    Sva = S[va_t]
    Ava, Mva = A[va_t], Mk[va_t]
    tags = [m.tag for m in stack]
    torch.cuda.reset_peak_memory_stats()
    ts = time.time()

    def evaluate():
        mdl = ema_net if ema_net is not None else snet
        mdl.eval()
        P, Q, Aa = [], [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(va_t), 16384):
                xb = Xg[va_x[i:i + 16384]].to(dev).unsqueeze(0).expand(M, -1, -1)
                p, q, aa = mdl(xb)
                P.append(p.float()); Q.append(q.float()); Aa.append(aa.float())
        mdl.train()
        P = torch.cat(P, 1); Q = torch.cat(Q, 1); Aa = torch.cat(Aa, 1)
        e_psi = ((P - Yva) ** 2).sum(-1).mean(-1) / 4225
        r2s = 1 - ((Q - Sva) ** 2).mean(1) / Sva.var(0)
        n = Mva.sum(0).clamp_min(1.0)
        mu = (Ava * Mva).sum(0) / n
        sst = (((Ava - mu) ** 2) * Mva).sum(0)
        r2a = 1 - (((Aa - Ava) ** 2) * Mva).sum(1) / sst.clamp_min(1e-12)
        return e_psi.cpu().numpy(), r2s.cpu().numpy(), r2a.cpu().numpy(), P, Q

    def one_step(bh, b):
        if batcher is None:
            xb, yb = Xg[b], Y[b]
        else:
            xb, yb = batcher.get(bh)
        if a0.noise > 0:
            xb = xb + a0.noise * torch.stack([torch.randn(xb.shape[1:], generator=g, device=dev)
                                              for g in ngens])
        return train_step_aux(fwd, params, opt, sched, xb, yb, S[b], A[b], Mk[b], b, cfg, ctx, wsc, waux, pix,
                              chunk, ema=ema, impl=a.pix_impl)

    if a.bench_steps:
        ph = torch.stack([tr_t[torch.randperm(len(tr_t), generator=g)] for g in gens])
        perm = ph.to(dev)
        tw = time.time()
        for i in range(5):
            one_step(ph[:, i * a0.bs:(i + 1) * a0.bs], perm[:, i * a0.bs:(i + 1) * a0.bs])
        torch.cuda.synchronize(); tb = time.time()
        for i in range(5, 5 + a.bench_steps):
            one_step(ph[:, i * a0.bs:(i + 1) * a0.bs], perm[:, i * a0.bs:(i + 1) * a0.bs])
        torch.cuda.synchronize()
        ms_step = 1e3 * (time.time() - tb) / a.bench_steps
        res = {"tags": tags, "M": M, "bench_steps": a.bench_steps, "ms_per_step": ms_step,
               "warmup_s": tb - tw, "ms_per_member_step": ms_step / M, "aux": aux["info"]["heads"],
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
        print("BENCH", json.dumps(res), flush=True)
        return res

    print(f"stack {tags}: train frames {len(tr)} val frames {len(va)} fin {fin} steps {steps} aux "
          f"{aux['info']['heads']} ({aux['n']} columns)", flush=True)
    nstep = 0
    r2a = None
    tlac = np.zeros(M)
    for ep in range(a0.epochs):
        te = time.time()
        ph = torch.stack([tr_t[torch.randperm(len(tr_t), generator=g)] for g in gens])
        perm = ph.to(dev)
        tl = torch.zeros(M, device=dev); tla = torch.zeros(M, device=dev); nb = 0
        for i in range(0, len(tr_t) - a0.bs + 1, a0.bs):
            l_, la_ = one_step(ph[:, i:i + a0.bs], perm[:, i:i + a0.bs])
            tl += l_; tla += la_; nb += 1; nstep += 1
        tlc = (tl / nb).cpu().numpy(); tlac = (tla / nb).cpu().numpy()
        last = ep == a0.epochs - 1
        if ep % a0.eval_every == a0.eval_every - 1 or last:
            mem = torch.cuda.max_memory_allocated() / 1e9
            if a0.trainval:
                for k, t in enumerate(tags):
                    print(f"[{t}] ep {ep} loss {tlc[k]:.4f} aux {tlac[k]:.4f} (trainval: no held-out eval)  "
                          f"{time.time() - t0:.0f}s maxmem {mem:.2f}GB", flush=True)
            else:
                e, r2s, r2a, _, _ = evaluate()
                for k, t in enumerate(tags):
                    r = r2s[k]; ra = r2a[k]
                    jtxt = f" jphi mean {np.mean(ra[4:]):.4f}" if len(ra) > 4 else ""
                    print(f"[{t}] ep {ep} loss {tlc[k]:.4f} aux {tlac[k]:.4f} val psiMSEn {e[k]:.3e} "
                          f"r2 q95 {r[0]:.4f} bN {r[1]:.4f} li {r[2]:.4f} R {r[3]:.4f} "
                          f"Z {r[4]:.4f} | aux r2 kap {ra[0]:.4f} tt {ra[1]:.4f} tb {ra[2]:.4f} V {ra[3]:.4f}"
                          f"{jtxt}  {time.time() - t0:.0f}s maxmem {mem:.2f}GB", flush=True)
        print(f"epoch {ep} {time.time() - te:.1f}s ({1e3 * (time.time() - te) / nb:.1f} ms/step, "
              f"M={M})", flush=True)
    train_s = time.time() - ts
    mdl = ema_net if ema_net is not None else snet
    P = Q = None
    if not a0.trainval:
        _, _, r2a, P, Q = evaluate()
        P, Q = P.cpu(), Q.cpu()
    models = [mdl.member_state_dict(k, bool(m.linskip)) for k, m in enumerate(stack)]
    res = {"tags": tags, "M": M, "train_s": train_s, "steps": nstep,
           "ms_per_step": 1e3 * train_s / max(nstep, 1), "ms_per_member_step": 1e3 * train_s / max(nstep, 1) / M,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9, "models": models, "P": P, "Q": Q,
           "aux_states": [mdl.aux_state(k) for k in range(M)],
           "aux_val_r2": ({t: dict(zip(names, map(float, r2a[k]))) for k, t in enumerate(tags)}
                          if r2a is not None else None),
           "aux_last_train_loss": {t: float(tlac[k]) for k, t in enumerate(tags)}}
    del snet, ema_net, ema, opt
    return res


def save_aux(m, k, st, aux, WORK):
    """aux.json (settings, TRAIN standardisation, final VAL aux R^2) + aux_head.pt (the training-only head)."""
    out = WORK / "models" / m.tag
    torch.save(st["aux_states"][k], out / "aux_head.pt")
    info = dict(aux["info"])
    info.update({"aux_seed": AUX_SEED + int(m.seed), "trainer": "train_d3d_bx (mtask aux heads)",
                 "val_r2": (st["aux_val_r2"] or {}).get(m.tag),
                 "last_train_aux_loss": st["aux_last_train_loss"][m.tag],
                 "note": "training-only head; model.pt is the plain NetB (inference unchanged)"})
    (out / "aux.json").write_text(json.dumps(info, indent=1, default=float))


def save_member(m, sd, WORK, xmu, xsd, stats, mean, V, ysd, smu, ssd, fin, rows, va, P, Q, gt_va, t0):
    net = build_net(vars(m), fin, NSCAL)
    net.load_state_dict(sd, strict=True)               # validates the NetB layout
    out = WORK / "models" / m.tag
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict(), "xmu": xmu, "xsd": xsd, "stats": stats,
                "mean": mean.cpu(), "V": V.cpu(), "ysd": ysd, "smu": smu.cpu(), "ssd": ssd.cpu(),
                "args": vars(m), "fin": fin, "feat": m.feat, "trainer": "train_d3d_bx"},
               out / "model.pt")
    if m.trainval:
        (out / "summary.json").write_text(json.dumps({"trainval": True, "alldata": m.alldata,
                                                      "args": vars(m), "trainer": "train_d3d_bx",
                                                      "seconds": time.time() - t0}))
        print(f"[{m.tag}] TRAINVAL saved", flush=True)
        return
    psi_va = ((P * ysd) @ V.cpu().T + mean.cpu()).numpy().astype(np.float32)
    sc_va = (Q * ssd.cpu() + smu.cpu()).numpy()
    np.save(out / "val_psi.npy", psi_va.reshape(-1, 65, 65))
    np.save(out / "val_scal.npy", sc_va)
    np.save(out / "val_idx.npy", rows[va])
    r2 = 1 - ((gt_va - psi_va) ** 2).sum() / ((gt_va - gt_va.mean()) ** 2).sum()
    print(f"[{m.tag}] VAL pooled R2 psi", r2, f"total {time.time() - t0:.0f}s", flush=True)
    (out / "summary.json").write_text(json.dumps({"val_r2_psi": float(r2), "args": vars(m),
                                                  "trainer": "train_d3d_bx",
                                                  "seconds": time.time() - t0}))


if __name__ == "__main__":
    main()
