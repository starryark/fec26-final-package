"""Amortized MAST psi network v2 (post-G0): fec/amort_net.py + short causal state, known
nominal passive flux, posterior-mean (multi-latent) training data and a seed ensemble.

Inputs: amort_net.features (11 coils + Ip, Thomson log Te/ne, masks, radius offsets) plus the
standardised causal state of fec.amort_gen2.causal_state (5/20 ms backward derivatives of the
12 MAST_COILS columns and P6U-P6L with its derivatives).  Known part of psi (inputs only):
psi_coil(I) + psi_passive(C1, r = 1) from the cached per-frame C1 currents
(FEC_WORK/passive/frames/mast_public_test/c1_r1/I.npy, the production v3 passive input).
Target: y = (psi - known) / Ip[MA] in a PCA basis fitted on synthetic TRAIN; Ip^2-weighted
MSE; width/epoch selection on the shot-held-out synthetic VAL only; n_seeds models averaged.

  python -B fec/gpu_slot.py run --owner amort -- python -B fec/amort_net2.py train --gen v2main --tag n2_main --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.amort_net import (NG, OUT, build_mlp, coil_matrix, features, fit_pca, fit_stats,  # noqa
                           axis_boundary, train)
from fec.paths import WORK  # noqa: E402

PASS1 = WORK / "passive" / "frames" / "mast_public_test" / "c1_r1" / "I.npy"


def pass1_path():
    """C1 r=1 per-frame passive currents for the active solenoid fraction (amort_net.sol_f):
    the cached production file at f = 0.5, else FEC_WORK/amort/passive_f<f>/c1_r1 (amort_gen6)."""
    from fec.amort_net import sol_f
    f = sol_f()
    if abs(f - 0.5) < 1e-12:
        return PASS1
    return WORK / "amort" / f"passive_f{f:g}" / "c1_r1" / "I.npy"
KEYS = ["g", "shot", "I", "Ip", "R", "te", "ne", "psi", "realistic", "lat"]


def passive_model():
    from fec.passive import PassiveModel
    return PassiveModel()


def known_psi(I, g, M, Gg, bs=4096):
    """psi_coil(I) + psi_passive(C1, r=1) for global frames g, (N,65,65) float64."""
    Ip1 = np.load(pass1_path(), mmap_mode="r")
    out = np.empty((len(g), NG, NG))
    for i in range(0, len(g), bs):
        gi = np.asarray(g[i:i + bs])
        out[i:i + bs] = ((np.asarray(I[i:i + bs], float) @ M.T) +
                         np.asarray(Ip1[gi], float) @ Gg.T).reshape(-1, NG, NG)
    return out


def fit_cstats(Cs):
    return dict(cs_mu=Cs.mean(0), cs_sd=Cs.std(0) + 1e-3)


def features2(I, Ip, R, te, ne, Cs, st):
    xc = np.clip((Cs - st["cs_mu"]) / st["cs_sd"], -8, 8).astype(np.float32)
    return np.concatenate([features(I, Ip, R, te, ne, st), xc], 1)


def load_thomson(src, tag, max_chunks=0):
    """te/ne (n_real, N_valid, C) of a re-synthesised Thomson set gen_<tag> (one file per chunk
    of gen_<src>, all frames), masked with the source chunks' validity (load_gen order)."""
    from fec.amort_gen import OUT as GOUT
    fs = sorted((GOUT / f"gen_{src}").glob("chunk_*.npz"))
    if max_chunks:
        fs = fs[:max_chunks]
    te, ne = [], []
    for f in fs:
        v = np.load(f)["valid"]
        t = np.load(GOUT / f"gen_{tag}" / f.name)
        te.append(t["te"][:, v]); ne.append(t["ne"][:, v])
    return np.concatenate(te, 1), np.concatenate(ne, 1)


def prepare2(gen_tag, K=96, val_frac=0.05, seed=0, pca_max=20000, max_chunks=0, norm=False,
             thomson=None):
    """norm: shape-normalised Thomson (amort_net.shape_norm); thomson: re-synthesised Thomson
    set replacing the generator's (fec/amort_gen4.py resynth)."""
    from fec.amort_gen import load_gen
    from fec.amort_gen2 import causal_state
    t0 = time.time()
    keys = [k for k in KEYS if not (thomson and k in ("te", "ne"))]
    z = load_gen(gen_tag, keys=keys, max_chunks=max_chunks)
    if thomson:
        z["te"], z["ne"] = load_thomson(gen_tag, thomson, max_chunks)
        assert z["te"].shape[1] == len(z["Ip"])
    spec, M = coil_matrix()
    Gg = passive_model().Gg
    N = len(z["Ip"])
    rng = np.random.default_rng(seed)
    shots = np.unique(z["shot"])
    vshots = rng.choice(shots, max(1, int(round(val_frac * len(shots)))), replace=False)
    is_val = np.isin(z["shot"], vshots)
    tr = ~is_val
    Cs = causal_state(z["g"])
    st = fit_stats(z["I"][tr], z["Ip"][tr], z["R"][tr], z["te"][:, tr], z["ne"][:, tr], norm)
    st.update(fit_cstats(Cs[tr]))
    ipm = z["Ip"] / 1e6
    Y = np.empty((N, NG * NG), np.float32)
    for i in range(0, N, 4096):
        sl = slice(i, i + 4096)
        Y[sl] = ((z["psi"][sl].astype(np.float64) - known_psi(z["I"][sl], z["g"][sl], M, Gg)) /
                 ipm[sl, None, None]).reshape(-1, NG * NG)
    itr = np.flatnonzero(tr)
    if len(itr) > pca_max:
        itr = np.sort(rng.choice(itr, pca_max, replace=False))
    mu, U, ev = fit_pca(Y[itr], K)
    C = np.empty((N, K), np.float32)
    for i in range(0, N, 4096):
        C[i:i + 4096] = (Y[i:i + 4096].astype(np.float64) - mu) @ U.T
    del Y
    X = np.stack([features2(z["I"], z["Ip"], z["R"], z["te"][r], z["ne"][r], Cs, st)
                  for r in range(z["te"].shape[0])])
    print(f"prepared2 {gen_tag}: N {N} (val {is_val.sum()}, realistic {z['realistic'].mean():.3f},"
          f" latents {len(np.unique(z['lat']))}) F {X.shape[-1]} n_real {X.shape[0]} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return dict(X=X, C=C, ipm=ipm.astype(np.float32), is_val=is_val, st=st,
                mu=mu.astype(np.float32), U=U.astype(np.float32), ev=ev)


def save2(tag, nets, D, cfg, hists):
    import torch
    d = OUT / "models" / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save(dict(states=[n.state_dict() for n, _ in nets], sds=[s.cpu() for _, s in nets],
                    cfg=cfg), d / "model.pt")
    np.savez(d / "basis.npz", mu=D["mu"], U=D["U"], ev=D["ev"],
             **{f"st_{k}": v for k, v in D["st"].items()})
    (d / "cfg.json").write_text(json.dumps(dict(cfg, hist=hists), indent=1))


def load2(tag, device="cuda"):
    import torch
    d = OUT / "models" / tag
    ck = torch.load(d / "model.pt", map_location=device, weights_only=False)
    b = np.load(d / "basis.npz")
    st = {k[3:]: b[k] for k in b.files if k.startswith("st_")}
    cfg = ck["cfg"]
    nets = []
    for s, sd in zip(ck["states"], ck["sds"]):
        n = build_mlp(cfg["n_in"], cfg["K"], cfg["width"], cfg["depth"], cfg["drop"]).to(device)
        n.load_state_dict(s); n.eval()
        nets.append((n, sd.to(device)))
    return nets, b["mu"], b["U"], st, cfg


def predict2(model, S, M, Gg, g, I, Ip, R, te, ne, bs=4096):
    """Ensemble-mean psi + axis/boundary for frames g (inputs only)."""
    import torch
    from fec.amort_gen2 import causal_state
    nets, mu, U, st, cfg = model
    X = features2(I, Ip, R, te, ne, causal_state(g), st)
    kn = known_psi(I, g, M, Gg)
    psi = np.empty((len(Ip), NG, NG), np.float32)
    Ut = torch.as_tensor(U, device=S.dev); mut = torch.as_tensor(mu, device=S.dev)
    with torch.no_grad():
        for i in range(0, len(Ip), bs):
            x = torch.as_tensor(X[i:i + bs], device=S.dev)
            c = torch.stack([n(x) * sd for n, sd in nets]).mean(0)
            y = (c @ Ut + mut).cpu().numpy().astype(np.float64)
            psi[i:i + bs] = y.reshape(-1, NG, NG) * (np.asarray(Ip[i:i + bs]) / 1e6)[:, None, None] \
                + kn[i:i + bs]
    return dict(psi=psi, **axis_boundary(S, psi))


def cmd_train(a):
    D = prepare2(a.gen, K=a.K, seed=0, max_chunks=a.max_chunks, norm=a.norm,
                 thomson=a.thomson or None)
    widths = [int(x) for x in str(a.width).split(",")]
    best = None
    for wdt in widths:                           # width chosen on synthetic VAL (seed 0)
        net, sd, hist, vl = train(D, width=wdt, depth=a.depth, epochs=a.epochs, bs=a.bs,
                                  lr=a.lr, wd=a.wd, seed=0, log_every=a.log_every)
        print(f"width {wdt}: best synthetic-VAL loss {vl:.4e}", flush=True)
        if best is None or vl < best[0]:
            best = (vl, wdt, [(net, sd)], [hist])
    vl, wdt, nets, hists = best
    for s in range(1, a.seeds):
        net, sd, hist, v = train(D, width=wdt, depth=a.depth, epochs=a.epochs, bs=a.bs,
                                 lr=a.lr, wd=a.wd, seed=s, log_every=a.log_every)
        print(f"seed {s}: synthetic-VAL loss {v:.4e}", flush=True)
        nets.append((net, sd)); hists.append(hist)
    cfg = dict(gen=a.gen, thomson=a.thomson, norm=a.norm, K=a.K, width=wdt, depth=a.depth, drop=0.0, epochs=a.epochs, bs=a.bs,
               lr=a.lr, wd=a.wd, seeds=a.seeds, n_in=int(D["X"].shape[-1]), val_loss=vl,
               widths_tried=widths, n_train=int((~D["is_val"]).sum()),
               n_val=int(D["is_val"].sum()))
    save2(a.tag, nets, D, cfg, hists)
    print(f"TRAIN2_DONE {a.tag} width {wdt} seeds {a.seeds}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--gen", default="v2main")
    ap.add_argument("--tag", default="n2_main")
    ap.add_argument("--K", type=int, default=96)
    ap.add_argument("--width", default="512,1024")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--norm", action="store_true", help="shape-normalised Thomson inputs")
    ap.add_argument("--thomson", default="", help="re-synthesised Thomson set (gen_<tag>)")
    a = ap.parse_args()
    cmd_train(a)


if __name__ == "__main__":
    main()
