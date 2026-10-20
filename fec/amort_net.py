"""Amortized MAST psi network (family AMORTIZED_MAST_20261010, G0): inputs -> psi PCA coefficients.

Trained ONLY on our own solver's synthetic equilibria (fec/amort_gen.py).  Inputs (exactly what
the per-frame physics pipeline sees): the 11 coil currents (incl. solenoid), Ip, the Thomson Te/ne
channel vectors (log-scaled, per-channel standardised, missing -> 0) with their missing masks and
the channel-radius offsets.  Target: y = (psi - psi_coil(I)) / Ip[MA], where psi_coil is the exact
vacuum flux of the measured coil currents (fec.machines.mast_setup matrix, solenoid I/2 per turn);
y is expanded in a PCA basis fitted on the synthetic TRAIN maps.  Loss: Ip^2-weighted coefficient
MSE (= pooled pixel MSE of psi up to the fixed truncation).  All standardisation, the PCA basis
and the model selection (epoch / width) use synthetic data only (a 5% shot-held-out synthetic
VAL split); the twin evaluation frames are never used for fitting or selection.

  python -B fec/gpu_slot.py run --owner amort -- python -B fec/amort_net.py train --gen p2k --tag n_p2k
  python -B fec/gpu_slot.py run --owner amort -- python -B fec/amort_net.py twin --tag n_p2k --scen B,C,A
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

OUT = WORK / "amort"
NG = 65


# ============================================================================ features / targets
def sol_f():
    """Solenoid turn fraction for the amortized family: env FEC_AMORT_SOLF (default 0.5 =
    fec.machines.MAST_SOL_TURN_FRACTION).  The coil matrix is linear in it (solenoid column)."""
    import os
    return float(os.environ.get("FEC_AMORT_SOLF", "0.5"))


def coil_matrix(f=None):
    """(spec, M) with the solenoid column scaled from the cached f = 0.5 matrix to f (no cache
    writes outside FEC_WORK/amort)."""
    from fec.machines import MAST_COILS, mast_setup
    spec, M = mast_setup()
    f = sol_f() if f is None else f
    if abs(f - 0.5) > 1e-12:
        M = M.copy()
        M[:, MAST_COILS.index("magnetics_sol_current")] *= f / 0.5
    return spec, M


def psi_coil(I, M):
    """Exact vacuum flux of the measured coil currents (B,65,65) float64."""
    return (np.asarray(I, float) @ M.T).reshape(-1, NG, NG)


def shape_norm(x, q=95.0):
    """Per-frame amplitude-free Thomson profile: x / robust-max(x) (95th percentile of the finite
    channels), NaN kept.  (post-G0, sub39b: Thomson amplitudes carry no psi information in the
    synthetic world, so the net must not see them.)"""
    with np.errstate(invalid="ignore"):
        m = np.nanpercentile(np.where(np.isfinite(x), x, np.nan), q, axis=-1, keepdims=True)
    return x / np.where(np.isfinite(m) & (m > 0), m, np.nan)


def _thomson_logs(te, ne, norm):
    if norm:
        return (np.log(np.clip(shape_norm(te), 1e-3, None)),
                np.log(np.clip(shape_norm(ne), 1e-3, None)))
    return np.log(np.clip(te, 1.0, None)), np.log(np.clip(ne, 1e17, None) / 1e19)


def fit_stats(I, Ip, R, te, ne, norm=False):
    """Standardisation constants from synthetic TRAIN frames only (te/ne: (n_real,N,C)).
    norm=True: Thomson enters only as per-frame shape-normalised profiles (no amplitudes)."""
    lte, lne = _thomson_logs(te, ne, norm)
    with np.errstate(invalid="ignore"):
        st = dict(I_mu=I.mean(0), I_sd=I.std(0) + 1.0, Ip_mu=float(Ip.mean()),
                  Ip_sd=float(Ip.std() + 1.0),
                  te_mu=np.nan_to_num(np.nanmean(lte, (0, 1))),
                  te_sd=np.nan_to_num(np.nanstd(lte, (0, 1)), nan=1.0) + 1e-3,
                  ne_mu=np.nan_to_num(np.nanmean(lne, (0, 1))),
                  ne_sd=np.nan_to_num(np.nanstd(lne, (0, 1)), nan=1.0) + 1e-3,
                  R_ref=np.nan_to_num(np.nanmedian(R, 0)), feat_norm=float(norm))
    return {k: np.asarray(v, np.float64) for k, v in st.items()}


def features(I, Ip, R, te, ne, st):
    """(N, 12 + 5C) float32 network inputs for one Thomson realisation (te/ne (N,C))."""
    xI = (I - st["I_mu"]) / st["I_sd"]
    xIp = ((Ip - st["Ip_mu"]) / st["Ip_sd"])[:, None]
    norm = bool(float(st.get("feat_norm", 0.0)))
    with np.errstate(invalid="ignore", divide="ignore"):
        lt, ln = _thomson_logs(te, ne, norm)
        mte = np.isfinite(lt); mne = np.isfinite(ln)
        lte = (lt - st["te_mu"]) / st["te_sd"]
        lne = (ln - st["ne_mu"]) / st["ne_sd"]
    lte = np.where(mte, np.clip(lte, -6, 6), 0.0)
    lne = np.where(mne, np.clip(lne, -6, 6), 0.0)
    dR = np.where(np.isfinite(R), (R - st["R_ref"]) / 0.01, 0.0)
    X = np.concatenate([xI, xIp, lte, lne, mte, mne, np.clip(dR, -5, 5)], 1)
    return X.astype(np.float32)


def fit_pca(Y, K, w=None, device="cuda"):
    """Mean + top-K principal directions of Y (N,P) (optionally sample-weighted)."""
    import torch
    Yt = torch.as_tensor(Y, device=device, dtype=torch.float64)
    wt = torch.ones(len(Y), device=device, dtype=torch.float64) if w is None else \
        torch.as_tensor(w, device=device, dtype=torch.float64)
    wt = wt / wt.sum()
    mu = (wt[:, None] * Yt).sum(0)
    C = torch.zeros(Yt.shape[1], Yt.shape[1], device=device, dtype=torch.float64)
    for i in range(0, len(Yt), 4096):
        Z = (Yt[i:i + 4096] - mu) * wt[i:i + 4096, None].sqrt()
        C += Z.T @ Z
    ev, U = torch.linalg.eigh(C)
    ev, U = ev.flip(0), U.flip(1)
    return mu.cpu().numpy(), U[:, :K].T.contiguous().cpu().numpy(), ev.cpu().numpy()


def build_mlp(n_in, K, width=1024, depth=4, drop=0.0):
    import torch.nn as nn

    class Res(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.f = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.SiLU(),
                                   nn.Dropout(drop), nn.Linear(w, w))

        def forward(self, x):
            return x + self.f(x)

    return nn.Sequential(nn.Linear(n_in, width), *[Res(width) for _ in range(depth)],
                         nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, K))


# ============================================================================ training
def targets(psi, I, ipm, M, bs=4096):
    """y = (psi - psi_coil(I)) / Ip[MA], flattened (N,P) float32 (chunked, low memory)."""
    N = len(ipm)
    Y = np.empty((N, NG * NG), np.float32)
    for i in range(0, N, bs):
        Y[i:i + bs] = ((np.asarray(psi[i:i + bs], np.float64) - psi_coil(I[i:i + bs], M)) /
                       ipm[i:i + bs, None, None]).reshape(-1, NG * NG)
    return Y


def sst_pooled(psi, bs=4096):
    """Pooled total sum of squares of psi about its scalar mean (as the R2psi metric)."""
    s1 = s2 = 0.0
    n = 0
    for i in range(0, len(psi), bs):
        p = np.asarray(psi[i:i + bs], np.float64)
        s1 += p.sum(); s2 += (p ** 2).sum(); n += p.size
    return s2 - s1 ** 2 / n


def pca_floor(tag, scens=("B", "C", "A")):
    """Ceiling check: R2psi of the twin truths' best K-term reconstruction in the model's basis
    (psi_coil exact + Ip (mu + proj)), on the realistic and all-valid subsets."""
    from fec.twin_mast import load_truth, realistic, truth_valid
    b = np.load(OUT / "models" / tag / "basis.npz")
    mu, U = b["mu"].astype(np.float64), b["U"].astype(np.float64)
    _, M = coil_matrix()
    res = {}
    for scen in scens:
        fr, tr, ts, prof, dv = load_truth(scen)
        ipm = fr["Ip"] / 1e6
        Y = targets(tr["psi"], fr["I"], ipm, M).astype(np.float64) - mu
        sse = (((Y - (Y @ U.T) @ U) * ipm[:, None]) ** 2).sum(1)
        for sub, m in (("real", realistic(scen, tr)), ("all", truth_valid(tr))):
            pt = tr["psi"][m].astype(np.float64)
            r = 1 - sse[m].sum() / np.sum((pt - pt.mean()) ** 2)
            res[f"{scen}|{sub}"] = float(r)
            print(f"PCA floor {tag} {scen} {sub}: R2psi {r:.6f} (n {m.sum()})", flush=True)
    return res


def prepare(gen_tag, K=96, val_frac=0.05, seed=0, pca_max=20000, max_chunks=0, norm=False):
    """Synthetic arrays -> features for all realisations, PCA targets, shot-held-out VAL split."""
    from fec.amort_gen import load_gen
    t0 = time.time()
    z = load_gen(gen_tag, max_chunks=max_chunks)
    spec, M = coil_matrix()
    N = len(z["Ip"])
    rng = np.random.default_rng(seed)
    shots = np.unique(z["shot"])
    vshots = rng.choice(shots, max(1, int(round(val_frac * len(shots)))), replace=False)
    is_val = np.isin(z["shot"], vshots)
    tr = ~is_val
    st = fit_stats(z["I"][tr], z["Ip"][tr], z["R"][tr], z["te"][:, tr], z["ne"][:, tr], norm)
    ipm = z["Ip"] / 1e6
    Y = targets(z["psi"], z["I"], ipm, M)                         # (N,P) float32
    itr = np.flatnonzero(tr)
    if len(itr) > pca_max:
        itr = np.sort(rng.choice(itr, pca_max, replace=False))
    mu, U, ev = fit_pca(Y[itr], K)
    C = np.empty((N, K), np.float32)
    sse = np.zeros(N)
    for i in range(0, N, 4096):
        Yc = Y[i:i + 4096].astype(np.float64) - mu
        c = Yc @ U.T
        C[i:i + 4096] = c
        sse[i:i + 4096] = (((Yc - c @ U) * ipm[i:i + 4096, None]) ** 2).sum(1)
    del Y
    X = np.stack([features(z["I"], z["Ip"], z["R"], z["te"][r], z["ne"][r], st)
                  for r in range(z["te"].shape[0])])              # (n_real, N, F)
    out = dict(X=X, C=C, ipm=ipm.astype(np.float32), is_val=is_val,
               st=st, mu=mu.astype(np.float32), U=U.astype(np.float32), ev=ev,
               real=z["realistic"], shot=z["shot"])
    for nm, m in (("train", tr), ("val", is_val)):
        print(f"PCA K={K} truncation R2psi floor ({nm}, {m.sum()} frames): "
              f"{1 - sse[m].sum() / sst_pooled(z['psi'][m]):.7f}", flush=True)
    print(f"prepared {gen_tag}: N {N} (val {is_val.sum()}, realistic {z['realistic'].mean():.3f})"
          f" F {X.shape[-1]} n_real {X.shape[0]} in {time.time() - t0:.0f}s", flush=True)
    return out


def train(D, width=1024, depth=4, drop=0.0, epochs=200, bs=512, lr=2e-3, wd=1e-4, seed=0,
          log_every=10, device="cuda"):
    """Train the MLP; returns (model state of the best synthetic-VAL epoch, history)."""
    import torch
    torch.manual_seed(seed)
    tr = np.flatnonzero(~D["is_val"]); va = np.flatnonzero(D["is_val"])
    X = torch.as_tensor(D["X"], device=device)                     # (R,N,F)
    C = torch.as_tensor(D["C"], device=device)
    w = torch.as_tensor(D["ipm"], device=device) ** 2
    w = w / w[tr].mean()
    sd = C[tr].std(0) + 1e-12
    nreal, N, F = X.shape
    K = C.shape[1]
    net = build_mlp(F, K, width, depth, drop).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * max(1, len(tr) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    trt = torch.as_tensor(tr, device=device)
    best, best_state, hist = np.inf, None, []

    def val_loss():
        net.eval()
        with torch.no_grad():
            tot = 0.0
            for r in range(nreal):
                p = net(X[r, va]) * sd
                tot += float((w[va] * ((p - C[va]) ** 2).sum(1)).mean())
        net.train()
        return tot / nreal

    t0 = time.time()
    for ep in range(epochs):
        perm = trt[torch.randperm(len(tr), device=device)]
        rr = torch.randint(0, nreal, (len(tr),), device=device)
        tl, nb = 0.0, 0
        for i in range(0, len(tr) - bs + 1, bs):
            b = perm[i:i + bs]
            x = X[rr[i:i + bs], b]
            p = net(x) * sd
            loss = (w[b] * ((p - C[b]) ** 2).sum(1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sch.step()
            tl = tl + loss.detach(); nb += 1
        tl = float(tl)
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            vl = val_loss()
            hist.append((ep + 1, tl / max(nb, 1), vl))
            if vl < best:
                best = vl
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"ep {ep + 1} train {tl / max(nb, 1):.4e} val {vl:.4e} best {best:.4e} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    net.load_state_dict(best_state)
    return net, sd, hist, best


def save_model(tag, net, sd, D, cfg, hist):
    import torch
    d = OUT / "models" / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=net.state_dict(), sd=sd.cpu(), cfg=cfg), d / "model.pt")
    np.savez(d / "basis.npz", mu=D["mu"], U=D["U"], ev=D["ev"],
             **{f"st_{k}": v for k, v in D["st"].items()})
    (d / "cfg.json").write_text(json.dumps(dict(cfg, hist=hist), indent=1))


def load_model(tag, device="cuda"):
    import torch
    d = OUT / "models" / tag
    ck = torch.load(d / "model.pt", map_location=device, weights_only=False)
    b = np.load(d / "basis.npz")
    st = {k[3:]: b[k] for k in b.files if k.startswith("st_")}
    cfg = ck["cfg"]
    net = build_mlp(cfg["n_in"], cfg["K"], cfg["width"], cfg["depth"], cfg["drop"]).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["sd"].to(device), b["mu"], b["U"], st, cfg


# ============================================================================ inference
def predict_psi(model, I, Ip, R, te, ne, M, device="cuda", bs=4096):
    """psi (N,65,65) float32 = psi_coil(I) + Ip[MA] (mu + c U)."""
    import torch
    net, sd, mu, U, st, cfg = model
    X = features(I, Ip, R, te, ne, st)
    out = np.empty((len(Ip), NG, NG), np.float32)
    Ut = torch.as_tensor(U, device=device); mut = torch.as_tensor(mu, device=device)
    with torch.no_grad():
        for i in range(0, len(Ip), bs):
            x = torch.as_tensor(X[i:i + bs], device=device)
            y = (net(x) * sd) @ Ut + mut
            out[i:i + bs] = (y.cpu().numpy().astype(np.float64) *
                             (np.asarray(Ip[i:i + bs]) / 1e6)[:, None]).reshape(-1, NG, NG) + \
                psi_coil(I[i:i + bs], M)
    return out


def axis_boundary(S, psi, bs=2048):
    """psi_ax, Rax, Zax, psi_b (+ X-point found flag) of arbitrary maps with the solver's own
    O-point/X-point finders (same conventions as the truth and the kfit2 states)."""
    import torch
    keys = ("psi_ax", "Rax", "Zax", "psi_b", "xfound")
    out = {k: np.empty(len(psi)) for k in keys}
    for i in range(0, len(psi), bs):
        p = torch.as_tensor(psi[i:i + bs], device=S.dev, dtype=S.dt)
        pa, Ra, Za = S.find_axis(p)
        (fu, vu, _, _), (fl, vl, _, _) = S.find_xpoints(p, Ra, Za, pa)
        neg = torch.full_like(vu, -1e30)
        pb = torch.maximum(torch.where(fu, vu, neg), torch.where(fl, vl, neg))
        nox = ~(fu | fl)
        pb = torch.where(nox, pa - 0.05 * pa.abs().clamp_min(1e-3), pb)
        for k, v in zip(keys, (pa, Ra, Za, pb, ~nox)):
            out[k][i:i + bs] = v.detach().cpu().numpy()
    out["xfound"] = out["xfound"].astype(bool)
    return out


def predict_frames(model, S, M, I, Ip, R, te, ne):
    psi = predict_psi(model, I, Ip, R, te, ne, M, device=S.dev)
    ab = axis_boundary(S, psi)
    return dict(psi=psi, **ab)


# ============================================================================ CLI
def cmd_train(a):
    import torch
    D = prepare(a.gen, K=a.K, seed=a.seed, max_chunks=a.max_chunks)
    widths = [int(x) for x in str(a.width).split(",")]
    best = None
    for wdt in widths:
        net, sd, hist, vl = train(D, width=wdt, depth=a.depth, drop=a.drop, epochs=a.epochs,
                                  bs=a.bs, lr=a.lr, wd=a.wd, seed=a.seed,
                                  log_every=a.log_every)
        print(f"width {wdt}: best synthetic-VAL loss {vl:.4e}", flush=True)
        if best is None or vl < best[0]:
            best = (vl, wdt, net, sd, hist)
    vl, wdt, net, sd, hist = best
    cfg = dict(gen=a.gen, max_chunks=a.max_chunks, K=a.K, width=wdt, depth=a.depth, drop=a.drop, epochs=a.epochs,
               bs=a.bs, lr=a.lr, wd=a.wd, seed=a.seed, n_in=int(D["X"].shape[-1]),
               val_loss=vl, widths_tried=widths, n_train=int((~D["is_val"]).sum()),
               n_val=int(D["is_val"].sum()))
    save_model(a.tag, net, sd, D, cfg, hist)
    print(f"TRAIN_DONE {a.tag} width {wdt} val {vl:.4e}", flush=True)
    del D
    torch.cuda.empty_cache()


def cmd_seeds(a):
    """Extra ensemble members of a FROZEN recipe: the base model's cfg (width, K, epochs, bs,
    lr, wd) and the identical synthetic split / PCA basis (prepare seed = base seed); only the
    training seed changes.  Saves <tag>_s<k>."""
    import torch
    base = json.loads((OUT / "models" / a.tag / "cfg.json").read_text())
    D = prepare(base["gen"], K=base["K"], seed=base["seed"],
                max_chunks=base.get("max_chunks", 0))
    for s in [int(x) for x in a.seeds.split(",")]:
        net, sd, hist, vl = train(D, width=base["width"], depth=base["depth"], drop=base["drop"],
                                  epochs=base["epochs"], bs=base["bs"], lr=base["lr"],
                                  wd=base["wd"], seed=s, log_every=a.log_every)
        cfg = {k: v for k, v in base.items() if k != "hist"}
        cfg.update(seed_train=s, val_loss=vl, member_of=a.tag)
        save_model(f"{a.tag}_s{s}", net, sd, D, cfg, hist)
        print(f"SEED_DONE {a.tag}_s{s} val {vl:.4e}", flush=True)
    torch.cuda.empty_cache()


def cmd_trainn(a):
    """post-G0, sub39b: the frozen G0 recipe (cfg of --base: width, K, epochs, bs, lr, wd, same
    synthetic split/PCA seed) with shape-normalised Thomson inputs; seeds --seeds -> <tag>_s<k>."""
    import torch
    base = json.loads((OUT / "models" / a.base / "cfg.json").read_text())
    D = prepare(base["gen"], K=base["K"], seed=base["seed"], max_chunks=base.get("max_chunks", 0),
                norm=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        net, sd, hist, vl = train(D, width=base["width"], depth=base["depth"], drop=base["drop"],
                                  epochs=base["epochs"], bs=base["bs"], lr=base["lr"],
                                  wd=base["wd"], seed=s, log_every=a.log_every)
        cfg = {k: v for k, v in base.items() if k != "hist"}
        cfg.update(seed_train=s, val_loss=vl, feat="thomson shape-normalised (no amplitudes)",
                   recipe_of=a.base, label="post-G0, sub39b")
        save_model(f"{a.tag}_s{s}", net, sd, D, cfg, hist)
        print(f"SEED_DONE {a.tag}_s{s} val {vl:.4e}", flush=True)
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "floor", "seeds", "trainn"])
    ap.add_argument("--base", default="n_main")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--gen", default="p2k")
    ap.add_argument("--tag", default="n_p2k")
    ap.add_argument("--K", type=int, default=96)
    ap.add_argument("--width", default="1024")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-chunks", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "seeds":
        cmd_seeds(a)
    elif a.cmd == "trainn":
        cmd_trainn(a)
    else:
        pca_floor(a.tag)


if __name__ == "__main__":
    main()
