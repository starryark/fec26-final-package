"""DIII-D supervised model, family d3d_b: inputs -> PCA coefficients of psi + scalars.

Derived from fec/train_d3d.py (same Net, PCA target, output conventions) plus:
  * shape-aware pixel loss: pixel errors (reconstructed from PCA coefficients) weighted by
    1 + wband * exp(-((psiN - 1) / sig)^2 / 2) [inside the scorer envelope] + win * [psiN < 1],
    psiN from TRAIN labels (WORK/ds_b/diii_d_train/bnd2.npy: psi at EFIT axis, EFIT LCFS
    level), envelope = TRAIN-only LCFS union (env_train.npz, fec/bnd_b.py; no scorer asset);
    optional Gaussian blobs at the lowest and highest EFIT-LCFS points (the code's "X-point" blobs, --wx)
    and weighted finite-difference (gradient) loss --wgrad
  * optional extra input features (--feat ext, fec/features_b.py)
  * EMA of weights (--ema), Gaussian input noise (--noise), dropout, weight decay
--trainval (final refit): fit on 'train'+'val' shots with identical hyperparameters/epochs; all
preprocessing is then fitted on those rows, no VAL evaluation is done and no val_*.npy written.
--alldata (Final-phase models, ONLY after the one-time internal-TEST evaluation): as --trainval
but on train+val+test shots; refuses to run unless WORK/test_eval_b/DONE exists; uses
WORK/ds_b/diii_d_train/bnd2_all.npy / bnd_x_all.npy (python -B fec/bnd_b.py --alldata).
Whole-shot frozen split; preprocessing (current stats via fit_mask, input standardisation,
PCA, target scales) fitted on the run's training rows: TRAIN shots only in the default (gate)
runs, train+val with --trainval, train+val+test with --alldata; labels are read for 'train'
and 'val' rows only, plus the 'test' rows with --alldata (otherwise never read).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.features import load_ds  # noqa: E402
from fec.features_b import d3d_inputs_b, thomson_groups  # noqa: E402
from fec.paths import WORK  # noqa: E402
from fec.train_d3d import Net  # noqa: E402

DEV = "cuda"


class NetB(Net):
    """Net plus an optional zero-initialised linear skip from inputs to all outputs."""

    def __init__(self, fin, K, nscal, w=1024, depth=6, drop=0.0, linskip=False):
        super().__init__(fin, K, nscal, w, depth, drop)
        self.K = K
        self.skip = None
        if linskip:
            self.skip = nn.Linear(fin, K + nscal)
            nn.init.zeros_(self.skip.weight); nn.init.zeros_(self.skip.bias)

    def forward(self, x):
        p, q = super().forward(x)
        if self.skip is None:
            return p, q
        s = self.skip(x)
        return p + s[:, :self.K], q + s[:, self.K:]


def build_net(args, fin, nscal=5):
    return NetB(fin, args["K"], nscal, args["width"], args["depth"], args.get("drop", 0.0),
                args.get("linskip", False))


def pca_basis(psi_mm, idx, K, nsample=300000, seed=0):
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
    order = torch.argsort(ev, descending=True)
    ev, V = ev[order], V[:, order]
    del C
    return mean.float(), V[:, :K].float().contiguous(), ev


def project_rows(psi_mm, rows, mean, V, chunk=50000):
    """PCA coefficients of psi_mm[rows] (rows sorted); only those rows are read."""
    out = torch.empty(len(rows), V.shape[1], device=DEV)
    for i in range(0, len(rows), chunk):
        x = torch.as_tensor(np.asarray(psi_mm[rows[i:i + chunk]], dtype=np.float32).reshape(-1, 4225),
                            device=DEV)
        out[i:i + chunk] = (x - mean) @ V
    return out


SB_KEYS = ("wbp", "bp_sig", "bp_band", "pcgrad")
FT_KEYS = ("ft_from", "ft_lr_frac", "li2_rho", "li2_w", "li2_thr", "gap_os", "shape_rho", "wnorm")


def get_args(argv=None, sb_ok=False):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feat", default="base", choices=["base", "ext", "hist", "nots", "base_gap", "nots_gap"],
                    help="base_gap/nots_gap: + 2 EFIT-time gap columns (fec/features_b.gap_columns; li2ft arm D)")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--wscal", type=float, default=0.3)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0, help="input noise std (normalized units)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay per step (0 = off)")
    ap.add_argument("--wband", type=float, default=0.0)
    ap.add_argument("--sig", type=float, default=0.05)
    ap.add_argument("--win", type=float, default=0.0)
    ap.add_argument("--wgrad", type=float, default=0.0)
    ap.add_argument("--wli", type=float, default=0.0, help="per-frame weight 1 + wli*clip(li-1.3,0,3) "
                    "on the pixel/gradient loss (TRAIN EFIT li label; targets the high-li ramp-down "
                    "frames that carry most li-consistency loss; root 2026-10-09)")
    ap.add_argument("--wx", type=float, default=0.0, help="weight of Gaussian blobs at the "
                    "lowest/highest EFIT-LCFS points (X-points)")
    ap.add_argument("--rx", type=float, default=0.08, help="blob radius [m]")
    ap.add_argument("--linskip", action="store_true")
    ap.add_argument("--tsdrop", type=float, default=0.0, help="per-sample prob. of simulating "
                    "fully missing Thomson (core-only and edge-only each at half this rate)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trainval", action="store_true", help="final refit on train+val shots")
    ap.add_argument("--alldata", action="store_true", help="Final-phase refit on train+val+test "
                    "shots; only after the one-time TEST evaluation (needs WORK/test_eval_b/DONE)")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--vram_gb", type=float, default=4.3, help="caching-allocator cap (GB)")
    # Sobolev / B_p supervision (agent mscale 2026-10-10, docs/research/D3D_MEMBER_SCALE_20261010.md).
    # Implemented only by fec/train_d3d_bx.py (callers pass sb_ok=True). SUPPRESS defaults: absent from the
    # namespace unless given, so argument dicts (saved args, summaries) of existing runs are unchanged.
    ap.add_argument("--wbp", type=float, default=argparse.SUPPRESS, help="B_p objective weight as a "
                    "gradient-norm ratio rho: w = rho*|g_psi|/|g_B| per member and step (train_d3d_bx only)")
    ap.add_argument("--bp_sig", type=float, default=argparse.SUPPRESS, help="LCFS band width in psiN "
                    "of the B_p band term (default 0.05)")
    ap.add_argument("--bp_band", type=float, default=argparse.SUPPRESS, help="weight of the B_p band "
                    "term relative to the interior term (default 1.0)")
    ap.add_argument("--pcgrad", action="store_true", default=argparse.SUPPRESS, help="project the B_p "
                    "gradient away from conflicts with the psi and scalar-head gradients")
    # li2ft fine-tune (agent li2ft 2026-10-12, docs/archive/fec/D3D_LI2FT_20261012.md): implemented only by
    # fec/train_d3d_bx.py; SUPPRESS defaults, so the argument dicts of every other run are unchanged.
    ap.add_argument("--ft_from", default=argparse.SUPPRESS, help="fine-tune WORK/models/<TAG>/model.pt (keeps its "
                    "PCA basis and input/target normalisation; --epochs = fine-tune epochs; train_d3d_bx only)")
    ap.add_argument("--ft_lr_frac", type=float, default=argparse.SUPPRESS, help="fine-tune LR: cosine from "
                    "frac x --lr to 0, fresh AdamW (default 0.1)")
    ap.add_argument("--li2_rho", type=float, default=argparse.SUPPRESS, help="scorer li(2) term at the true contour: "
                    "lambda = rho x median|g_base|/median|g_li| on the psi output layer, first 100 batches, then frozen "
                    "(0 = off)")
    ap.add_argument("--li2_w", type=float, default=argparse.SUPPRESS, help="li term frame weight where the target "
                    "li(2) > --li2_thr (default 4; 1 elsewhere)")
    ap.add_argument("--li2_thr", type=float, default=argparse.SUPPRESS, help="default 1.7")
    ap.add_argument("--gap_os", type=int, default=argparse.SUPPRESS, help="fine-tune sampling: frames within 600 ms "
                    "after a > 100 ms EFIT-time gap appear N times per epoch (default 1)")
    ap.add_argument("--shape_rho", type=float, default=argparse.SUPPRESS, help="li2ft arm S: scorer-shape loss on "
                    "R_axis, kappa, V at the frozen true geometry (fec/li2_shape.py), map head only; lambda_shape = rho / "
                    "median over the first 100 batches of |g_shape|/|g_base| on the psi output layer (default 0 = off)")
    ap.add_argument("--wnorm", choices=["batch", "frame"], default=argparse.SUPPRESS, help="li2ft arm N: normalise the "
                    "pixel-weight map per frame (mean over pixels) instead of over batch x pixels (default batch)")
    a = ap.parse_args(argv)
    if not sb_ok and any(hasattr(a, k) for k in FT_KEYS):
        raise SystemExit("--ft_from/--ft_lr_frac/--li2_*/--gap_os (li2ft fine-tune) are implemented only in "
                         "fec/train_d3d_bx.py")
    if not sb_ok and any(hasattr(a, k) for k in SB_KEYS):
        raise SystemExit("--wbp/--bp_sig/--bp_band/--pcgrad (Sobolev/B_p loss) are implemented only in "
                         "fec/train_d3d_bx.py")
    return a


def main():
    a = get_args()
    if a.alldata:
        if not (WORK / "test_eval_b" / "DONE").exists():
            raise SystemExit("--alldata refused: WORK/test_eval_b/DONE missing (run the one-time "
                             "internal-TEST evaluation first, or copy its DONE marker)")
        a.trainval = True                 # same no-held-out code path
    tot_mem = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / tot_mem))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = WORK / "models" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])[ds["shot"]]
    if a.alldata:
        rows = np.arange(len(splits))                                 # train+val+test shots
    else:
        rows = np.flatnonzero((splits == "train") | (splits == "val"))   # never 'test' rows
    is_tr = (splits[rows] == "train") | a.trainval
    tr = np.flatnonzero(is_tr)            # positions within rows
    va = np.flatnonzero(splits[rows] == "val")   # in-sample (unused) when --trainval
    fit = (splits == "train") | ((splits == "val") & a.trainval) | a.alldata
    X, s, stats = d3d_inputs_b(ds, "diii_d_train", feat=a.feat, fit_mask=fit)
    X = X[rows]
    xmu = X[tr].mean(0); xsd = X[tr].std(0) + 1e-4
    Xc = torch.as_tensor((X - xmu) / xsd).pin_memory()
    del X
    print("inputs", tuple(Xc.shape), f"{time.time() - t0:.0f}s", flush=True)
    tsg = {}
    for k, (cols, vals) in (thomson_groups(a.feat).items() if a.tsdrop > 0 else ()):
        cols = np.array(cols)
        tsg[k] = (torch.as_tensor(cols, device=DEV),
                  torch.as_tensor((np.array(vals, np.float32) - xmu[cols]) / xsd[cols], device=DEV))
    mean, V, ev = pca_basis(ds["psi"], rows[tr], a.K)
    tot = ev.clamp_min(0).sum().item()
    print("PCA unexplained var frac", {k: 1 - ev[:k].sum().item() / tot for k in (64, 128, 256)},
          flush=True)
    Y = project_rows(ds["psi"], rows, mean, V)
    ds["psi"] = None                      # release the 13 GB memmap (host working set)
    ysd = math.sqrt(tot / 4225)
    Y = Y / ysd
    va_d = torch.as_tensor(va, device=DEV)
    Yva = Y[va_d].clone()
    Ycpu = Y.cpu().pin_memory()          # targets live in pinned host memory (VRAM budget)
    del Y
    scal = torch.as_tensor(ds["scal"][rows], device=DEV)
    smu = scal[tr].mean(0); ssd = scal[tr].std(0)
    S = (scal - smu) / ssd
    sw = None
    if getattr(a, "wli", 0.0) > 0:          # training-time sample weights from the li label
        li = torch.nan_to_num(scal[:, 2].float(), nan=1.0)
        sw = 1.0 + a.wli * (li - 1.3).clamp(0.0, 3.0)
        print("wli sample weights: mean %.3f, frac>1 %.4f" % (sw[torch.as_tensor(tr, device=DEV)].mean().item(),
              (sw[torch.as_tensor(tr, device=DEV)] > 1).float().mean().item()), flush=True)
    pix = a.wband > 0 or a.win > 0 or a.wgrad > 0 or a.wx > 0
    if pix:
        sfx = "_all" if a.alldata else ""
        bnd = torch.as_tensor(np.load(WORK / "ds_b" / "diii_d_train" / f"bnd2{sfx}.npy")[rows],
                              device=DEV)
        assert torch.isfinite(bnd[torch.as_tensor(tr, device=DEV)]).all()
        mk = np.load(WORK / "ds_b" / "diii_d_train" / "env_train.npz")   # TRAIN-only envelope
        maskc = torch.as_tensor(mk["env"].astype(np.float32).reshape(-1), device=DEV)
        VT = V.T.contiguous()                 # (K, 4225)
        VTs = VT * ysd
        RRg, ZZg = np.meshgrid(mk["grid_R"], mk["grid_Z"])
        RRg = torch.as_tensor(RRg.reshape(-1), device=DEV, dtype=torch.float32)
        ZZg = torch.as_tensor(ZZg.reshape(-1), device=DEV, dtype=torch.float32)
        if a.wx > 0:
            bx = torch.as_tensor(np.load(WORK / "ds_b" / "diii_d_train" / f"bnd_x{sfx}.npy")[rows],
                                 device=DEV)
            assert torch.isfinite(bx[torch.as_tensor(tr, device=DEV)]).all()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print("targets ready", f"{time.time() - t0:.0f}s", flush=True)

    net = build_net(vars(a), Xc.shape[1], S.shape[1]).to(DEV)
    ema = None
    if a.ema > 0:
        ema = torch.optim.swa_utils.AveragedModel(
            net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd, fused=True)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.05, final_div_factor=1e3)
    wsc = torch.tensor([1.0, 1.0, 0.3, 0.3, 0.3], device=DEV) * a.wscal
    tr_t = torch.as_tensor(tr)

    def model_for_eval():
        return ema.module if ema is not None else net

    def evaluate():
        idx = torch.as_tensor(va)
        m = model_for_eval()
        m.eval()
        P = []; Q = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(idx), 16384):
                p, q = m(Xc[idx[i:i + 16384]].to(DEV))
                P.append(p.float()); Q.append(q.float())
        P = torch.cat(P); Q = torch.cat(Q)
        net.train()
        e_psi = ((P - Yva) ** 2).sum(1).mean().item() / 4225
        r2s = 1 - ((Q - S[va_d]) ** 2).mean(0) / S[va_d].var(0)
        return e_psi, r2s.cpu().numpy(), P, Q

    def pixel_loss(p, b, yb):
        err = (p - yb) @ VT                                        # (B,4225), units of ysd
        with torch.no_grad():                                      # weights, in place
            wn = torch.addmm(mean, yb, VTs)                        # true psi (stored units)
            wn.sub_(bnd[b, 0:1]).div_(bnd[b, 1:2] - bnd[b, 0:1])   # psiN
            inside = (wn < 1.0) if a.win > 0 else None
            wn.sub_(1.0).div_(a.sig).square_().mul_(-0.5).exp_().mul_(maskc * a.wband).add_(1.0)
            if inside is not None:
                wn.add_(inside.float().mul_(maskc * a.win))
            if a.wx > 0:
                c = -0.5 / a.rx ** 2
                for j in (0, 2):
                    d = (RRg - bx[b, j:j + 1]).square_()
                    d.add_((ZZg - bx[b, j + 1:j + 2]).square_())
                    wn.add_(d.mul_(c).exp_().mul_(a.wx))
            wn.div_(wn.mean())
            if sw is not None:
                sb = sw[b]
                wn.mul_((sb / sb.mean())[:, None])
        lp = (wn * err.square()).mean() * 1e3
        if a.wgrad > 0:
            e2 = err.view(-1, 65, 65)
            w2 = wn.view(-1, 65, 65)
            dr = (e2[:, :, 1:] - e2[:, :, :-1]) ** 2 * 0.5 * (w2[:, :, 1:] + w2[:, :, :-1])
            dz = (e2[:, 1:, :] - e2[:, :-1, :]) ** 2 * 0.5 * (w2[:, 1:, :] + w2[:, :-1, :])
            lp = lp + a.wgrad * (dr.mean() + dz.mean()) * 1e3 * 64.0
        return lp

    print(f"train frames {len(tr)} val frames {len(va)} fin {Xc.shape[1]} steps {steps}",
          flush=True)
    step = 0
    for ep in range(a.epochs):
        perm = tr_t[torch.randperm(len(tr_t))]
        tl = 0.0; nb = 0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            bc = perm[i:i + a.bs]
            xb = Xc[bc].to(DEV, non_blocking=True)
            yb = Ycpu[bc].to(DEV, non_blocking=True)
            b = bc.to(DEV, non_blocking=True)
            if a.noise > 0:
                xb = xb + a.noise * torch.randn_like(xb)
            if a.tsdrop > 0:
                u = torch.rand(len(xb), device=DEV)
                for k, lo, hi in (("core", 0.0, 1.5 * a.tsdrop), ("edge", 0.0, a.tsdrop),
                                  ("edge", 1.5 * a.tsdrop, 2.0 * a.tsdrop)):
                    sel = torch.nonzero((u >= lo) & (u < hi)).squeeze(1)
                    if len(sel):
                        cols, vals = tsg[k]
                        xb[sel[:, None], cols[None, :]] = vals[None, :].expand(len(sel), -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p, q = net(xb)
            p = p.float()
            if pix:
                lp = pixel_loss(p, b, yb)
            else:
                lp = ((p - yb) ** 2).sum(1).mean() / 4225 * 1e3
            lq = (((q.float() - S[b]) ** 2).mean(0) * wsc).sum()
            loss = lp + lq
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if ema is not None:
                ema.update_parameters(net)
            tl += loss.item(); nb += 1
        if a.trainval and (ep % a.eval_every == a.eval_every - 1 or ep == a.epochs - 1):
            print(f"ep {ep} loss {tl / nb:.4f} (trainval: no held-out eval)  "
                  f"{time.time() - t0:.0f}s maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
                  flush=True)
        elif ep % a.eval_every == a.eval_every - 1 or ep == a.epochs - 1:
            e, r2s, _, _ = evaluate()
            print(f"ep {ep} loss {tl / nb:.4f} val psiMSEn {e:.3e} "
                  f"r2 q95 {r2s[0]:.4f} bN {r2s[1]:.4f} li {r2s[2]:.4f} R {r2s[3]:.4f} "
                  f"Z {r2s[4]:.4f}  {time.time() - t0:.0f}s "
                  f"maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
    m = model_for_eval()
    torch.save({"net": m.state_dict(), "xmu": xmu, "xsd": xsd, "stats": stats,
                "mean": mean.cpu(), "V": V.cpu(), "ysd": ysd, "smu": smu.cpu(), "ssd": ssd.cpu(),
                "args": vars(a), "fin": Xc.shape[1], "feat": a.feat}, out / "model.pt")
    if a.trainval:
        print("TRAINVAL done", f"total {time.time() - t0:.0f}s", flush=True)
        (out / "summary.json").write_text(json.dumps({"trainval": True, "alldata": a.alldata,
                                                      "args": vars(a),
                                                      "n_train_frames": int(len(tr)),
                                                      "seconds": time.time() - t0}))
        return
    _, _, P, Q = evaluate()
    P, Q = P.cpu(), Q.cpu()                     # reconstruct on host (VRAM cap)
    del Yva; torch.cuda.empty_cache()
    psi_va = ((P * ysd) @ V.cpu().T + mean.cpu()).numpy().astype(np.float32)
    sc_va = (Q * ssd.cpu() + smu.cpu()).numpy()
    np.save(out / "val_psi.npy", psi_va.reshape(-1, 65, 65))
    np.save(out / "val_scal.npy", sc_va)
    np.save(out / "val_idx.npy", rows[va])
    psi_mm = np.load(WORK / "ds" / "diii_d_train" / "psi.npy", mmap_mode="r")
    gt = np.asarray(psi_mm[rows[va]], dtype=np.float32).reshape(len(va), -1)
    r2 = 1 - ((gt - psi_va) ** 2).sum() / ((gt - gt.mean()) ** 2).sum()
    print("VAL pooled R2 psi", r2, f"total {time.time() - t0:.0f}s", flush=True)
    (out / "summary.json").write_text(json.dumps({"val_r2_psi": float(r2), "args": vars(a),
                                                  "seconds": time.time() - t0}))


if __name__ == "__main__":
    main()
