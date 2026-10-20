"""DIII-D supervised model: inputs -> PCA coefficients of psi + scalars (q95, betaN, li, R, Z).

Whole-shot frozen split from fec.dataset; preprocessing (input stats, PCA basis, target
scales) fitted on TRAIN shots only.
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
from fec.features import d3d_inputs, load_ds  # noqa: E402
from fec.paths import WORK  # noqa: E402

DEV = "cuda"


class Block(nn.Module):
    def __init__(self, w, drop=0.0):
        super().__init__()
        self.n = nn.LayerNorm(w)
        self.f = nn.Sequential(nn.Linear(w, w * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(w * 2, w))

    def forward(self, x):
        return x + self.f(self.n(x))


class Net(nn.Module):
    def __init__(self, fin, K, nscal, w=1024, depth=6, drop=0.0):
        super().__init__()
        self.inp = nn.Linear(fin, w)
        self.blocks = nn.Sequential(*[Block(w, drop) for _ in range(depth)])
        self.n = nn.LayerNorm(w)
        self.psi = nn.Linear(w, K)
        self.scal = nn.Linear(w, nscal)

    def forward(self, x):
        h = self.n(self.blocks(self.inp(x)))
        return self.psi(h), self.scal(h)


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
    return mean.float(), V[:, :K].float().contiguous(), ev


def project(psi_mm, mean, V, chunk=50000):
    N = psi_mm.shape[0]
    out = torch.empty(N, V.shape[1], device=DEV)
    for i in range(0, N, chunk):
        x = torch.as_tensor(np.asarray(psi_mm[i:i + chunk], dtype=np.float32).reshape(-1, 4225),
                            device=DEV)
        out[i:i + chunk] = (x - mean) @ V
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d3d_v1")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--wscal", type=float, default=0.3)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trainval", action="store_true", help="also train on VAL (final refit)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = WORK / "models" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])[ds["shot"]]
    tr = np.flatnonzero((splits == "train") | ((splits == "val") & a.trainval))
    va = np.flatnonzero(splits == "val")
    fitm = np.zeros(len(splits), bool); fitm[tr] = True
    X, s, stats = d3d_inputs(ds, fit_mask=fitm)
    # input normalisation (train only)
    xmu = X[tr].mean(0); xsd = X[tr].std(0) + 1e-4
    Xn = torch.as_tensor((X - xmu) / xsd, device=DEV)
    print("inputs", X.shape, f"{time.time() - t0:.0f}s", flush=True)
    mean, V, ev = pca_basis(ds["psi"], tr, a.K)
    tot = ev.clamp_min(0).sum().item()
    print("PCA captured var frac", {k: 1 - ev[:k].sum().item() / tot for k in (64, 128, 256, 512)},
          flush=True)
    Y = project(ds["psi"], mean, V)  # (N,K)
    ysd = math.sqrt(tot / 4225)       # global pixel std
    Y = Y / ysd
    scal = torch.as_tensor(ds["scal"], device=DEV)
    smu = scal[tr].mean(0); ssd = scal[tr].std(0)
    S = (scal - smu) / ssd
    print("targets ready", f"{time.time() - t0:.0f}s", flush=True)
    net = Net(Xn.shape[1], a.K, S.shape[1], a.width, a.depth, a.drop).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.05, final_div_factor=1e3)
    tr_t = torch.as_tensor(tr, device=DEV)
    va_t = torch.as_tensor(va, device=DEV)
    wsc = torch.tensor([1.0, 1.0, 0.3, 0.3, 0.3], device=DEV) * a.wscal

    def evaluate(idx):
        net.eval()
        se = 0.0; ss_s = torch.zeros(5, device=DEV, dtype=torch.float64)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = []; Q = []
            for i in range(0, len(idx), 16384):
                p, q = net(Xn[idx[i:i + 16384]])
                P.append(p.float()); Q.append(q.float())
        P = torch.cat(P); Q = torch.cat(Q)
        net.train()
        e_psi = ((P - Y[idx]) ** 2).sum(1).mean().item() / 4225  # normalized pixel MSE
        r2s = 1 - ((Q - S[idx]) ** 2).mean(0) / S[idx].var(0)
        return e_psi, r2s.cpu().numpy(), P, Q

    step = 0
    for ep in range(a.epochs):
        perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
        tl = 0.0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            b = perm[i:i + a.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p, q = net(Xn[b])
            lp = ((p.float() - Y[b]) ** 2).sum(1).mean() / 4225 * 1e3
            lq = (((q.float() - S[b]) ** 2).mean(0) * wsc).sum()
            loss = lp + lq
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            tl += loss.item()
        if ep % 5 == 4 or ep == a.epochs - 1:
            e, r2s, _, _ = evaluate(va_t)
            # pooled psi R2 on VAL (relative to VAL total variance of normalized psi)
            print(f"ep {ep} loss {tl / (len(perm) // a.bs):.4f} val psiMSEn {e:.3e} "
                  f"r2 q95 {r2s[0]:.4f} bN {r2s[1]:.4f} li {r2s[2]:.4f} R {r2s[3]:.4f} "
                  f"Z {r2s[4]:.4f}  {time.time() - t0:.0f}s", flush=True)
    torch.save({"net": net.state_dict(), "xmu": xmu, "xsd": xsd, "stats": stats,
                "mean": mean.cpu(), "V": V.cpu(), "ysd": ysd, "smu": smu.cpu(), "ssd": ssd.cpu(),
                "args": vars(a), "fin": Xn.shape[1]}, out / "model.pt")
    # save VAL predictions (psi in stored units) for the scorer
    _, _, P, Q = evaluate(va_t)
    psi_va = ((P * ysd) @ V.T + mean).cpu().numpy().astype(np.float32)
    sc_va = (Q * ssd + smu).cpu().numpy()
    np.save(out / "val_psi.npy", psi_va.reshape(-1, 65, 65))
    np.save(out / "val_scal.npy", sc_va)
    np.save(out / "val_idx.npy", va)
    gt = np.asarray(ds["psi"][va], dtype=np.float32).reshape(len(va), -1)
    pv = psi_va
    r2 = 1 - ((gt - pv) ** 2).sum() / ((gt - gt.mean()) ** 2).sum()
    print("VAL pooled R2 psi", r2, flush=True)
    (out / "summary.json").write_text(json.dumps({"val_r2_psi": float(r2), "args": vars(a)}))


if __name__ == "__main__":
    main()
