"""DIII-D ramp-down residual specialist (agent rdspec, 2026-10-09).

Plan, frozen gates and results: docs/research/D3D_RAMPDOWN_SPECIALIST_20261009.md.

An input-only blend weight g in [0, 1] selects ramp-down frames (after the in-shot Ip peak,
|Ip| / max_shot |Ip| <= 0.6 -> g = 1, linear to 0 at 0.75, pre-peak / no-Thomson -> 0).  The
output on a frame is base + g * (specialist ensemble - parent ensemble); every g = 0 frame is
the base array itself (byte-identical, `blend`).  Specialists are short fine-tunes of existing
d3d_b members on TRAIN ramp-down frames (oversampled) plus a replay sample of other TRAIN
frames, with the parent's own loss; the checkpoint format is the parent's, so
fec/predict_d3d_b.py runs them unchanged.

Stage 1 (TRAIN-internal, no VAL labels): a whole-shot holdout H of TRAIN shots
(`holdout`); fold bases are the parent recipes re-trained on TRAIN minus H (`foldbase`,
a wrapper around fec/train_d3d_b.main that relabels H as its held-out split and the real
VAL/TEST shots as never-read); specialists are fine-tuned from the fold bases and compared
on H (`evalho`).  Stage 2 (VAL, once): specialists from the full-TRAIN members, `valbuild`.

  python -B fec/d3d_rdspec.py gate --config diii_d_public_test
  python -B fec/d3d_rdspec.py holdout
  python -B fec/d3d_rdspec.py foldbase --parent d3d_b60ix_shape_s0 --tag rdspec_fold_shape_s0
  python -B fec/d3d_rdspec.py finetune --parent TAG --tag rdspec_ft_... [--holdout] [--lr ..]
  python -B fec/d3d_rdspec.py evalho --base TAG[,TAG] --spec TAG[,TAG] --out NAME
Run with FEC_WORK=<FEC_IDX_WORK> (idx inputs/models); labels are read only for
TRAIN rows (stage 1) via fec.paths.TRAIN_WORK.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import TRAIN_WORK, WORK  # noqa: E402

G_LO, G_HI = 0.6, 0.75            # g = 1 for f <= G_LO (post-peak), linear to 0 at G_HI
H_FRAC, H_SEED = 0.15, 20261009   # TRAIN-internal whole-shot holdout (stage 1)
IP_CH = "magnetics_plasma_current"
IP_OFF = 6                         # offsets_ms index of 0 ms


# ----------------------------------------------------------------------------- gate
def ip_frac(ip, shot, t_ms):
    """f = |Ip| / max over the shot's frames and post = frame strictly after the (first) time-
    ordered argmax of |Ip| in its shot. Inputs only (the whole shot's Ip trace is an input).
    Non-finite Ip counts as 0 (and is flagged separately by `gate_weight`)."""
    x = np.nan_to_num(np.abs(np.asarray(ip, np.float64)), nan=0.0, posinf=0.0, neginf=0.0)
    shot = np.asarray(shot); t_ms = np.asarray(t_ms)
    f = np.zeros(len(x)); post = np.zeros(len(x), bool)
    if len(x) == 0:
        return f, post
    order = np.lexsort((t_ms, shot))
    bounds = np.flatnonzero(np.diff(shot[order])) + 1
    for seg in np.split(order, bounds):
        v = x[seg]
        k = int(np.argmax(v)); mx = v[k]
        f[seg] = v / mx if mx > 0 else 0.0
        p = np.zeros(len(seg), bool); p[k + 1:] = True
        post[seg] = p
    return f, post


def gate_weight(f, post, ip_finite=None, lo=G_LO, hi=G_HI):
    """Smooth blend weight: post-peak frames g = clip((hi - f) / (hi - lo), 0, 1); else 0.
    Frames with non-finite Ip (ip_finite False) get 0."""
    g = np.clip((hi - np.asarray(f, np.float64)) / (hi - lo), 0.0, 1.0)
    g = np.where(np.asarray(post, bool), g, 0.0)
    if ip_finite is not None:
        g = np.where(np.asarray(ip_finite, bool), g, 0.0)
    return g


def hard_gate(f, post, lo=G_LO):
    """The analysis gate gate_rd.npy: post-peak and f < lo."""
    return np.asarray(post, bool) & (np.asarray(f) < lo)


def ds_gate(ds, no_ts_zero=True):
    """g (N,) for a loaded ds (fec.features.load_ds); inputs only. g = 0 on frames without any
    finite Thomson value (there the base uses its 'nots' fallback and is left untouched)."""
    ip = ds["info"]["channels"].index(IP_CH)
    x = np.asarray(ds["cur"][:, ip, IP_OFF], np.float64)
    f, post = ip_frac(x, ds["shot"], ds["t_ms"])
    g = gate_weight(f, post, np.isfinite(x))
    if no_ts_zero:
        from fec.features_b import no_thomson_frames
        g = np.where(no_thomson_frames(ds), 0.0, g)
    return g, f, post


def blend(base, spec, parent, g):
    """base + g * (spec - parent) on g > 0 frames; g = 0 frames are copied from base exactly.
    base/spec/parent: (N, ...) arrays; g: (N,). Returns float32."""
    base = np.asarray(base)
    out = np.array(base, dtype=np.float32, copy=True)
    sel = np.flatnonzero(np.asarray(g) > 0)
    if len(sel):
        gs = np.asarray(g, np.float64)[sel].reshape((-1,) + (1,) * (base.ndim - 1))
        d = np.asarray(spec, np.float64)[sel] - np.asarray(parent, np.float64)[sel]
        out[sel] = (np.asarray(base, np.float64)[sel] + gs * d).astype(np.float32)
    return out


# -------------------------------------------------------------------------- holdout
def holdout_shots(splits, has_gate, frac=H_FRAC, seed=H_SEED, exclude=None):
    """Whole-shot TRAIN holdout H (shot indices), stratified by 'shot has gated frames';
    `exclude` (shot indices) are removed from the pool first (H2 = 15% of TRAIN minus H)."""
    splits = np.asarray(splits); has_gate = np.asarray(has_gate, bool)
    pool = np.ones(len(splits), bool)
    if exclude is not None:
        pool[np.asarray(exclude)] = False
    rng = np.random.default_rng(seed)
    out = []
    for stratum in (True, False):
        c = np.flatnonzero((splits == "train") & (has_gate == stratum) & pool)
        out.append(rng.choice(c, int(round(frac * len(c))), replace=False))
    return np.sort(np.concatenate(out))


def relabel_splits(splits, H):
    """Fold-base split: H -> 'val' (held out, scored), other TRAIN -> 'train', real VAL/TEST
    -> 'test' (train_d3d_b never reads 'test' rows)."""
    splits = np.asarray(splits)
    assert (splits[np.asarray(H)] == "train").all(), "holdout must be TRAIN shots"
    new = np.where(splits == "train", "train", "test").astype(object)
    new[np.asarray(H)] = "val"
    return [str(s) for s in new]


H2_SEED = 20261010                 # second holdout (doc §6.3)


def h_path():
    return WORK / "models" / "rdspec_H" / "h_shots.npy"


def h2_path():
    return WORK / "models" / "rdspec_H2" / "h_shots.npy"


def make_h2(splits, has_gate, H):
    """H2: 15% of TRAIN minus H (stratified, seed H2_SEED); disjoint from H and inside TRAIN."""
    H2 = holdout_shots(splits, has_gate, H_FRAC, H2_SEED, exclude=H)
    assert len(np.intersect1d(H2, H)) == 0, "H2 must be disjoint from H"
    assert (np.asarray(splits)[H2] == "train").all(), "H2 must be TRAIN shots"
    return H2


def load_h():
    return np.load(h_path())


def cmd_holdout(a):
    from fec.features import load_ds
    ds = load_ds("diii_d_train")
    g, _, _ = ds_gate(ds, no_ts_zero=False)
    splits = np.array(ds["info"]["splits"])
    has = np.zeros(len(splits), bool)
    has[np.unique(ds["shot"][g >= 1.0])] = True
    if getattr(a, "second", False):
        H = make_h2(splits, has, load_h()); p = h2_path()
    else:
        H = holdout_shots(splits, has); p = h_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not np.array_equal(np.load(p), H):
        raise SystemExit(f"{p} exists with different content")
    np.save(p, H)
    fr = np.isin(ds["shot"], H)
    print(f"H: {len(H)} shots ({has[H].sum()} with gate), {fr.sum()} frames, "
          f"{int((fr & (g >= 1)).sum())} g=1 frames, {int((fr & (g > 0)).sum())} g>0 frames -> {p}")


# ------------------------------------------------------------------------- foldbase
def _labels():
    """TRAIN/VAL labels via TRAIN_WORK (identical bytes to the idx root's copies)."""
    d = TRAIN_WORK / "ds" / "diii_d_train"
    return np.load(d / "psi.npy", mmap_mode="r"), np.load(d / "scal.npy")


def cmd_foldbase(a):
    """Re-train a parent recipe on TRAIN minus H (H = held-out 'val'); outputs ho_*.npy."""
    import fec.train_d3d_b as T
    from fec.features import load_ds as _load
    H = np.load(a.hfile) if getattr(a, "hfile", None) else load_h()
    summ = json.loads((WORK / "models" / a.parent / "summary.json").read_text())
    pa = dict(summ["args"])
    for k in ("trainval", "alldata"):
        assert not pa.get(k), k

    def load_ds_fold(config):
        ds = _load(config)
        assert config == "diii_d_train"
        info = dict(ds["info"]); info["splits"] = relabel_splits(info["splits"], H)
        ds["info"] = info
        ds["psi"], ds["scal"] = _labels()
        return ds

    T.load_ds = load_ds_fold
    argv = []
    for k, v in pa.items():
        if k in ("tag", "trainval", "alldata", "vram_gb"):
            continue
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
        else:
            argv += [f"--{k}", str(v)]
    argv += ["--tag", a.tag, "--vram_gb", str(a.vram_gb)]
    print("foldbase argv", " ".join(argv), flush=True)
    sys.argv = ["train_d3d_b.py"] + argv
    T.main()
    out = WORK / "models" / a.tag
    for n in ("psi", "scal", "idx"):
        (out / f"val_{n}.npy").replace(out / f"ho_{n}.npy")
    (out / "rdspec_meta.json").write_text(json.dumps(
        {"kind": "foldbase", "parent": a.parent, "H": str(a.hfile or h_path()), "n_H": int(len(H)),
         "note": "ho_* = predictions on TRAIN holdout H (not VAL)"}, indent=1))
    print("FOLDBASE_DONE", a.tag, flush=True)


# ------------------------------------------------------------------------- finetune
def sub_ds(ds, rows):
    """Row subset of a ds dict (all d3d_b base/nots features are per-frame)."""
    return {"info": ds["info"], "cur": np.asarray(ds["cur"][rows]), "ts": np.asarray(ds["ts"][rows]),
            "t_ms": ds["t_ms"][rows], "shot": ds["shot"][rows], "fidx": ds["fidx"][rows]}


def finetune_rows(train_mask, g, n_replay, seed):
    """Focus rows (TRAIN, g > 0) and a random replay sample of the other TRAIN rows."""
    rng = np.random.default_rng(seed)
    focus = np.flatnonzero(train_mask & (g > 0))
    other = np.flatnonzero(train_mask & (g <= 0))
    rep = np.sort(rng.choice(other, min(n_replay, len(other)), replace=False))
    return focus, rep


def cmd_finetune(a):
    import torch
    from fec.features import load_ds
    from fec.features_b import d3d_inputs_b
    from fec.train_d3d_b import build_net
    dev = "cuda"
    tot = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb * 1e9 / tot))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    ck = torch.load(WORK / "models" / a.parent / "model.pt", weights_only=False, map_location="cpu")
    pa = ck["args"]
    assert not pa.get("trainval") and not pa.get("alldata"), "parent must be a TRAIN-only model"
    feat = ck.get("feat", pa.get("feat", "base"))
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])[ds["shot"]]
    trm = splits == "train"
    if a.holdout:
        trm &= ~np.isin(ds["shot"], load_h())
    g, _, _ = ds_gate(ds, no_ts_zero=(feat != "nots"))
    focus, rep = finetune_rows(trm, g, a.replay, a.seed)
    rows = np.sort(np.r_[focus, rep])
    assert (splits[rows] == "train").all()
    is_focus = np.isin(rows, focus)
    print(f"finetune {a.tag} <- {a.parent} feat {feat}: focus {len(focus)} replay {len(rep)}",
          flush=True)
    sub = sub_ds(ds, rows)
    X = d3d_inputs_b(sub, "diii_d_train", ck["stats"], feat=feat)[0]
    assert X.shape[1] == ck["fin"]
    Xc = torch.as_tensor((X - ck["xmu"]) / ck["xsd"]).float().to(dev)
    del X
    psi_mm, scal_all = _labels()
    mean, V = ck["mean"].to(dev), ck["V"].to(dev)
    ysd = float(ck["ysd"])
    Y = torch.empty(len(rows), V.shape[1], device=dev)
    for i in range(0, len(rows), 20000):
        x = torch.as_tensor(np.asarray(psi_mm[rows[i:i + 20000]], np.float32).reshape(-1, 4225),
                            device=dev)
        Y[i:i + 20000] = (x - mean) @ V / ysd
    smu, ssd = ck["smu"].to(dev), ck["ssd"].to(dev)
    S = (torch.as_tensor(scal_all[rows], device=dev) - smu) / ssd
    assert torch.isfinite(S).all() and torch.isfinite(Y).all()
    pix = pa["wband"] > 0 or pa["win"] > 0 or pa["wgrad"] > 0 or pa["wx"] > 0
    if pix:
        bnd = torch.as_tensor(np.load(WORK / "ds_b" / "diii_d_train" / "bnd2.npy")[rows], device=dev)
        assert torch.isfinite(bnd).all()
        mk = np.load(WORK / "ds_b" / "diii_d_train" / "env_train.npz")
        maskc = torch.as_tensor(mk["env"].astype(np.float32).reshape(-1), device=dev)
        VT = V.T.contiguous(); VTs = VT * ysd
        RRg, ZZg = np.meshgrid(mk["grid_R"], mk["grid_Z"])
        RRg = torch.as_tensor(RRg.reshape(-1), device=dev, dtype=torch.float32)
        ZZg = torch.as_tensor(ZZg.reshape(-1), device=dev, dtype=torch.float32)
        if pa["wx"] > 0:
            bx = torch.as_tensor(np.load(WORK / "ds_b" / "diii_d_train" / "bnd_x.npy")[rows],
                                 device=dev)
            assert torch.isfinite(bx).all()

    def pixel_loss(p, b, yb):          # fec/train_d3d_b.py pixel_loss with the parent's args
        err = (p - yb) @ VT
        with torch.no_grad():
            wn = torch.addmm(mean, yb, VTs)
            wn.sub_(bnd[b, 0:1]).div_(bnd[b, 1:2] - bnd[b, 0:1])
            inside = (wn < 1.0) if pa["win"] > 0 else None
            wn.sub_(1.0).div_(pa["sig"]).square_().mul_(-0.5).exp_().mul_(maskc * pa["wband"]).add_(1.0)
            if inside is not None:
                wn.add_(inside.float().mul_(maskc * pa["win"]))
            if pa["wx"] > 0:
                c = -0.5 / pa["rx"] ** 2
                for j in (0, 2):
                    d = (RRg - bx[b, j:j + 1]).square_()
                    d.add_((ZZg - bx[b, j + 1:j + 2]).square_())
                    wn.add_(d.mul_(c).exp_().mul_(pa["wx"]))
            wn.div_(wn.mean())
        lp = (wn * err.square()).mean() * 1e3
        if pa["wgrad"] > 0:
            e2 = err.view(-1, 65, 65); w2 = wn.view(-1, 65, 65)
            dr = (e2[:, :, 1:] - e2[:, :, :-1]) ** 2 * 0.5 * (w2[:, :, 1:] + w2[:, :, :-1])
            dz = (e2[:, 1:, :] - e2[:, :-1, :]) ** 2 * 0.5 * (w2[:, 1:, :] + w2[:, :-1, :])
            lp = lp + pa["wgrad"] * (dr.mean() + dz.mean()) * 1e3 * 64.0
        return lp

    net = build_net(pa, ck["fin"]).to(dev)
    net.load_state_dict(ck["net"]); net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=pa["wd"])
    warm = max(1, int(0.05 * a.steps))

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        u = (s - warm) / max(1, a.steps - warm)
        return a.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * u)))

    wsc = torch.tensor([1.0, 1.0, 0.3, 0.3, 0.3], device=dev) * pa["wscal"]
    fpos = torch.as_tensor(np.flatnonzero(is_focus), device=dev)
    rpos = torch.as_tensor(np.flatnonzero(~is_focus), device=dev)
    nf = int(round(a.bs * a.focus_frac)); nr = a.bs - nf
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    tl = 0.0
    for step in range(a.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        b = torch.cat([fpos[torch.randint(len(fpos), (nf,), device=dev, generator=gen)],
                       rpos[torch.randint(len(rpos), (nr,), device=dev, generator=gen)]])
        xb, yb = Xc[b], Y[b]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p, q = net(xb)
        p = p.float()
        lp = pixel_loss(p, b, yb) if pix else ((p - yb) ** 2).sum(1).mean() / 4225 * 1e3
        lq = (((q.float() - S[b]) ** 2).mean(0) * wsc).sum()
        loss = lp + lq
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tl += loss.item()
        if (step + 1) % max(1, a.steps // 5) == 0:
            print(f"step {step + 1} loss {tl / max(1, a.steps // 5):.4f} {time.time() - t0:.0f}s "
                  f"maxmem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
            tl = 0.0
    out = WORK / "models" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    args = dict(pa, tag=a.tag)
    args["rdspec"] = {"parent": a.parent, "lr": a.lr, "steps": a.steps, "bs": a.bs,
                      "focus_frac": a.focus_frac, "replay": a.replay, "seed": a.seed,
                      "holdout": bool(a.holdout), "n_focus": int(len(focus)),
                      "G_LO": G_LO, "G_HI": G_HI}
    torch.save({"net": net.state_dict(), "xmu": ck["xmu"], "xsd": ck["xsd"], "stats": ck["stats"],
                "mean": ck["mean"], "V": ck["V"], "ysd": ck["ysd"], "smu": ck["smu"], "ssd": ck["ssd"],
                "args": args, "fin": ck["fin"], "feat": feat}, out / "model.pt")
    (out / "summary.json").write_text(json.dumps({"args": args, "seconds": time.time() - t0}))
    print("FINETUNE_DONE", a.tag, f"{time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- predict
def predict_rows(tags, weights, rows, config="diii_d_train"):
    """Weighted-mean fp32 prediction of d3d_b checkpoints on ds rows: psi (n,4225), scal (n,5)."""
    from fec.features import load_ds
    from fec.predict_d3d_b import predict_tag
    ds = load_ds(config)
    sub = sub_ds(ds, rows)
    w = np.asarray(weights, np.float64); w = w / w.sum()
    cache = {}
    P = Q = None
    for t, wi in zip(tags, w):
        p, q = predict_tag(sub, config, t, cache)
        P = wi * p.astype(np.float64) if P is None else P + wi * p
        Q = wi * q.astype(np.float64) if Q is None else Q + wi * q
    return P, Q


# ------------------------------------------------------------- stage-1 evaluation on H
def _gt_worker(psi):
    """GT side exactly as local_score.build_reference: contour + seven derived scalars."""
    from fec import score as SC
    LS = SC.LS
    R, Z, mc, mf = SC._mask()
    out = []
    for p in psi:
        c = LS.extract_lcfs(p, R, Z, LS.MACHINE, mc, mf, n_points=LS.N_POINTS)
        v = LS.derive_frame(p, R, Z, LS.MACHINE, mc, mf, contour=c)
        out.append((c, np.array([v[n] for n in LS.CONS_SCALARS], np.float64),
                    LS.major_radius(c) if c is not None else np.nan))
    return out


def _pred_worker(args):
    """Prediction side as fec/cons_frames.frame_scalars (bit-identical to score_shot)."""
    from fec import score as SC
    from fec.cons_frames import frame_scalars
    psi, refc, rgeo = args
    R, Z, mc, mf = SC._mask()
    d = np.full(len(psi), np.nan); cons = np.full((len(psi), 7), np.nan)
    for k, p in enumerate(psi):
        d[k], cons[k], _ = frame_scalars(p, R, Z, mc, mf, ref_c=refc[k], rgeo=rgeo[k])
        if refc[k] is None:
            d[k] = np.nan
    return d, cons


def _chunks(n, size=60):
    return [slice(i, min(n, i + size)) for i in range(0, n, size)]


def pooled_delta(parts_b, parts_c, sst, shots_all, frames_per_shot):
    """S-proxy difference (candidate - base) on a fold from per-frame parts of the changed
    frames. parts: dict(psi_sse (n,), q (n,2) sq err, cons (n,7) sq err (nan = invalid frame),
    d (n,)). sst: dict(psi, q (2,), cons (7,)). The D_LCFS term is the mean over ALL fold shots
    of the per-shot frame-mean, so unchanged frames contribute 0 to the difference."""
    w = {}
    dpsi = parts_c["psi_sse"].sum() - parts_b["psi_sse"].sum()
    dq = np.nansum(parts_c["q"], 0) - np.nansum(parts_b["q"], 0)
    dc = np.nansum(parts_c["cons"], 0) - np.nansum(parts_b["cons"], 0)
    w["psi"] = -0.55 * dpsi / sst["psi"]
    w["qb"] = -0.15 * np.mean(dq / sst["q"])
    w["cons"] = -0.20 * np.mean(dc / sst["cons"])
    dd = np.nan_to_num(parts_c["d"] - parts_b["d"])
    per_shot = {}
    for s, x in zip(parts_b["shot"], dd):
        per_shot[s] = per_shot.get(s, 0.0) + x
    w["dlcfs"] = -0.10 * sum(v / frames_per_shot[s] for s, v in per_shot.items()) / len(shots_all)
    w["S"] = w["psi"] + w["qb"] + w["cons"] + w["dlcfs"]
    w["d_cons_each"] = (-dc / sst["cons"]).tolist()
    return w


def cmd_evalho(a):
    """Stage 1: compare blends of fold specialists vs the fold bases on H (TRAIN labels only)."""
    from multiprocessing import Pool
    from fec.features import load_ds
    from fec import score as SC
    H = load_h()
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])
    assert (splits[H] == "train").all()
    g, _, _ = ds_gate(ds)
    inH = np.isin(ds["shot"], H)
    rows = np.flatnonzero(inH & (g > 0))
    rng = np.random.default_rng(1)
    rnd = np.sort(rng.choice(np.flatnonzero(inH), a.nsst, replace=False))
    psi_mm, scal_all = _labels()
    parents = a.parents.split(",")
    variants = {}
    for item in a.specs.split(";"):
        name, tags = item.split(":")
        variants[name] = tags.split(",")
        assert len(variants[name]) == len(parents)
    lams = [float(x) for x in a.lams.split(",")]
    t0 = time.time()
    G = np.asarray(psi_mm[rows], np.float64)
    Grnd = np.asarray(psi_mm[rnd], np.float64)
    with Pool(a.nproc) as pool:
        gt = sum(pool.map(_gt_worker, [G[s] for s in _chunks(len(rows))]), [])
        gtr = sum(pool.map(_gt_worker, [Grnd[s] for s in _chunks(len(rnd))]), [])
    refc = [x[0] for x in gt]; cg = np.array([x[1] for x in gt])
    shot = ds["shot"][rows]
    rg = np.array([x[2] for x in gt])
    rgeo = np.array([np.nanmean(rg[shot == s]) if np.isfinite(rg[shot == s]).any() else np.nan
                     for s in shot])
    cr = np.array([x[1] for x in gtr])
    nH = int(inH.sum())
    psi_tot_mean = Grnd.mean()
    sst = {"psi": nH * 4225 * Grnd.var(), "q": nH * np.nanvar(scal_all[rnd][:, :2], 0),
           "cons": nH * np.nanvar(cr, 0) * np.isfinite(cr).mean(0)}
    print(f"GT done: {len(rows)} g>0 H frames, {len(rnd)} SST frames, {time.time() - t0:.0f}s, "
          f"psi mean {psi_tot_mean:.4f}", flush=True)
    fps = {s: int(n) for s, n in zip(*np.unique(ds["shot"][inH], return_counts=True))}
    gq = scal_all[rows][:, :2].astype(np.float64)

    def parts(Pp, Qp):
        with Pool(a.nproc) as pool:
            res = pool.map(_pred_worker, [(Pp[s].reshape(-1, 65, 65), refc[s], rgeo[s])
                                          for s in _chunks(len(rows))])
        d = np.concatenate([r[0] for r in res]); cp = np.concatenate([r[1] for r in res])
        cpp = np.where(np.isfinite(cp), cp, np.nan)
        err = (cg - cpp) ** 2
        # official: invalid GT frames are excluded; a nan prediction is replaced by the GT mean
        mean_c = np.nanmean(cr, 0)
        err = np.where(np.isfinite(cg) & ~np.isfinite(cp), (cg - mean_c) ** 2, err)
        err = np.where(np.isfinite(cg), err, np.nan)
        return {"psi_sse": ((G.reshape(len(rows), -1) - Pp) ** 2).sum(1), "q": (gq - Qp[:, :2]) ** 2,
                "cons": err, "d": d, "shot": shot, "cons_pred": cp}

    gw = g[rows]
    Pb, Qb = predict_rows(parents, [1.0] * len(parents), rows)
    pb = parts(Pb, Qb)
    print(f"base parts {time.time() - t0:.0f}s", flush=True)
    res = {"rows": int(len(rows)), "H_shots": int(len(H)), "parents": parents, "variants": {}}
    store = {"rows": rows, "g": gw, "cons_gt": cg, "base_cons": pb["cons_pred"], "base_d": pb["d"]}
    ush = np.unique(shot)
    for name, tags in variants.items():
        Ps, Qs = predict_rows(tags, [1.0] * len(tags), rows)
        for lam in lams:
            Pc = Pb + (lam * gw)[:, None] * (Ps - Pb)
            Qc = Qb + (lam * gw)[:, None] * (Qs - Qb)
            pc = parts(Pc, Qc)
            w = pooled_delta(pb, pc, sst, H, fps)
            # paired whole-shot bootstrap over H (shots without g>0 frames contribute 0)
            brng = np.random.default_rng(0)
            idx_by = {s: np.flatnonzero(shot == s) for s in ush}
            boots = []
            Hl = list(H)
            for _ in range(a.boot):
                pick = brng.choice(len(Hl), len(Hl), replace=True)
                sel = np.concatenate([idx_by[Hl[i]] for i in pick if Hl[i] in idx_by] or [np.zeros(0, int)])
                sub_b = {k: v[sel] for k, v in pb.items()}; sub_c = {k: v[sel] for k, v in pc.items()}
                boots.append(pooled_delta(sub_b, sub_c, sst, H, fps)["S"])
            boots = np.array(boots)
            li_b = np.nansum(pb["cons"][:, 6]); li_c = np.nansum(pc["cons"][:, 6])
            key = f"{name}_lam{lam:g}"
            res["variants"][key] = {
                "dS_proxy": w["S"], "dS_terms": {k: w[k] for k in ("psi", "qb", "cons", "dlcfs")},
                "boot_mean": float(boots.mean()), "boot_sd": float(boots.std()),
                "frac_pos": float((boots > 0).mean()),
                "li_sse_base": float(li_b), "li_sse_cand": float(li_c), "li_rel": float(li_c / li_b - 1),
                "d_cons_each": dict(zip(SC.LS.CONS_SCALARS, w["d_cons_each"])),
                "psi_sse_rel": float(pc["psi_sse"].sum() / pb["psi_sse"].sum() - 1),
                "q_sse_rel": (np.nansum(pc["q"], 0) / np.nansum(pb["q"], 0) - 1).tolist()}
            store[f"{key}_cons"] = pc["cons_pred"]
            v = res["variants"][key]
            print(f"{key:14s} dS_proxy {v['dS_proxy']:+.6f} (boot SD {v['boot_sd']:.6f}, frac>0 "
                  f"{v['frac_pos']:.2f}) li SSE {li_b:.2f} -> {li_c:.2f} ({v['li_rel']:+.1%}) psi SSE "
                  f"{v['psi_sse_rel']:+.2%} terms " + " ".join(f"{k} {x:+.6f}" for k, x in v["dS_terms"].items()),
                  flush=True)
    out = WORK / "models" / "rdspec_H" / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage1.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez(out / "stage1_frames.npz", **store)
    print("EVALHO_DONE", out, f"{time.time() - t0:.0f}s", flush=True)


def ridge_cv(X, y, groups, alphas=(1.0, 10.0, 100.0, 1000.0), k=5, seed=0):
    """Shot-grouped k-fold ridge CV R^2 (TRAIN-fitted standardisation per fold) per alpha."""
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % k))
    fid = np.array([fold_of[s] for s in groups])
    out = {}
    for al in alphas:
        pred = np.zeros_like(y)
        for f in range(k):
            tr, te = fid != f, fid == f
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            A = (X[tr] - mu) / sd; ym = y[tr].mean()
            w = np.linalg.solve(A.T @ A + al * np.eye(A.shape[1]), A.T @ (y[tr] - ym))
            pred[te] = ((X[te] - mu) / sd) @ w + ym
        out[al] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return out


def cmd_diag(a):
    """Stage-1 diagnostics on H g > 0 frames from an evalho frames file (TRAIN labels only)."""
    from fec.features import load_ds, d3d_inputs
    z = np.load(WORK / "models" / "rdspec_H" / a.src / "stage1_frames.npz")
    rows, cg, cb, g = z["rows"], z["cons_gt"][:, 6], z["base_cons"][:, 6], z["g"]
    ds = load_ds("diii_d_train")
    ok = np.isfinite(cg) & np.isfinite(cb)
    e = (cb - cg) ** 2
    print(f"OOF li MSE/frame on H g>0: {np.nanmean(np.where(np.isfinite(cg), np.where(np.isfinite(cb), e, np.nan), np.nan)):.4f} "
          f"(g=1 frames {np.nanmean(e[ok & (g >= 1)]):.4f}; ref in-sample 0.012, VAL ~0.12); "
          f"top-5% share {np.sort(e[ok])[::-1][:max(1, ok.sum() // 20)].sum() / e[ok].sum():.2f}")
    sl = np.polyfit(cg[ok], cb[ok], 1)[0]
    print(f"slope li_pred on li_gt: {sl:.3f}; mean li_gt {cg[ok].mean():.3f} mean li_pred {cb[ok].mean():.3f}")
    sub = sub_ds(ds, rows)
    X, _, _ = d3d_inputs(sub, None)            # stats fitted on these H rows: diagnostic only
    _, f, _ = ds_gate(ds)
    feats = np.c_[X[:, :231], f[rows], g]
    r = (cg - cb)[ok]
    cv = ridge_cv(feats[ok], r, ds["shot"][rows][ok])
    print("ridge CV R2 of li residual:", {k: round(v, 3) for k, v in cv.items()})
    cv2 = ridge_cv(np.c_[f[rows], g][ok], r, ds["shot"][rows][ok])
    print("ridge CV R2 (f, g only):", {k: round(v, 3) for k, v in cv2.items()})


# --------------------------------------------------------------------- stage 2 (VAL)
def spec_tag(prefix, member):
    """Specialist tag for a C2 member tag (with or without the _fp32 suffix)."""
    return prefix + member.replace("_fp32", "").replace("d3d_b60ix_", "")


def cmd_valbuild(a):
    """Blend specialists onto the stored base VAL arrays (inputs only; no labels read)."""
    from fec.features import load_ds
    bd = WORK / "models" / a.base
    mem = json.loads((bd / "members.json").read_text())
    B = np.load(bd / "val_psi.npy"); Bs = np.load(bd / "val_scal.npy"); idx = np.load(bd / "val_idx.npy")
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])[ds["shot"]]
    assert (splits[idx] == "val").all()
    g_all, _, _ = ds_gate(ds)
    gv = g_all[idx] * a.lam
    sel = np.flatnonzero(gv > 0)
    parents = [m.replace("_fp32", "") for m in mem["members"]]
    specs = [spec_tag(a.prefix, m) for m in mem["members"]]
    t0 = time.time()
    P, PQ = predict_rows(parents, mem["weights"], idx[sel])
    Sp, SQ = predict_rows(specs, mem["weights"], idx[sel])
    out_psi = np.array(B, dtype=np.float32, copy=True)
    out_sc = np.array(Bs, dtype=Bs.dtype, copy=True)
    out_psi[sel] = blend(B[sel].reshape(len(sel), -1), Sp, P, gv[sel]).reshape(-1, 65, 65)
    out_sc[sel] = blend(Bs[sel], SQ, PQ, gv[sel]).astype(Bs.dtype)
    keep = np.setdiff1d(np.arange(len(idx)), sel)
    assert out_psi[keep].tobytes() == np.asarray(B[keep], np.float32).tobytes()
    assert out_sc[keep].tobytes() == Bs[keep].tobytes()
    o = WORK / "models" / a.tag
    o.mkdir(parents=True, exist_ok=True)
    np.save(o / "val_psi.npy", out_psi); np.save(o / "val_scal.npy", out_sc); np.save(o / "val_idx.npy", idx)
    np.save(o / "val_g.npy", gv)
    (o / "rdspec_meta.json").write_text(json.dumps(
        {"kind": "valblend", "base": a.base, "prefix": a.prefix, "lam": a.lam, "specs": specs,
         "parents": parents, "weights": mem["weights"], "n_changed": int(len(sel)),
         "parent_vs_base_maxabs_psi": float(np.abs(P - B[sel].reshape(len(sel), -1)).max()),
         "seconds": time.time() - t0}, indent=1))
    print(f"VALBUILD_DONE {a.tag}: {len(sel)} of {len(idx)} VAL frames changed; "
          f"max |parent - base| psi {np.abs(P - B[sel].reshape(len(sel), -1)).max():.3e}", flush=True)


def _val_parts(tag, ds, fp16=False):
    """Per-shot official parts of saved VAL predictions (root_var/boot_score.py logic)."""
    m = WORK / "models" / tag
    psi = np.load(m / "val_psi.npy", mmap_mode="r"); sc = np.load(m / "val_scal.npy")
    idx = np.load(m / "val_idx.npy")
    shots = ds["shot"][idx]; order = ds["info"]["order"]
    psi_lab, scal_lab = _labels()
    keys, gts, preds, pos = [], [], [], []
    for s in np.unique(shots):
        j = np.flatnonzero(shots == s); gi = idx[j]
        keys.append(order[s]); pos.append(j)
        gts.append({"psi": np.asarray(psi_lab[gi], dtype=np.float32),
                    "q95": scal_lab[gi, 0].astype(np.float64), "betaN": scal_lab[gi, 1].astype(np.float64)})
        p = np.asarray(psi[j], dtype=np.float16 if fp16 else np.float32)
        preds.append({"psirz": p.astype(np.float64), "q95": sc[j, 0].astype(np.float64),
                      "betaN": sc[j, 1].astype(np.float64)})
    return keys, gts, preds, pos


def _score_parts(keys, gts, preds, nproc=6):
    from multiprocessing import Pool
    from fec import score as SC
    with Pool(nproc) as pool:
        list(pool.imap_unordered(SC._ref_worker, [(k, g["psi"]) for k, g in zip(keys, gts)], chunksize=2))
    refs = [SC._load_ref(k) for k in keys]
    ref_parts = []
    for g, ref in zip(gts, refs):
        p = g["psi"].astype(np.float64)
        r = dict(psi_sum=float(p.sum()), psi_sumsq=float((p ** 2).sum()), psi_n=p.size,
                 scal_sum=np.zeros(SC.N_SCALARS), scal_sumsq=np.zeros(SC.N_SCALARS),
                 scal_n=np.zeros(SC.N_SCALARS), cons_sum=np.zeros(SC.N_CONS),
                 cons_sumsq=np.zeros(SC.N_CONS), cons_n=np.zeros(SC.N_CONS))
        for jj, name in enumerate(SC.SCALARS):
            v = np.asarray(g[name], dtype=np.float64); v = v[np.isfinite(v)]
            r["scal_sum"][jj] = v.sum(); r["scal_sumsq"][jj] = (v ** 2).sum(); r["scal_n"][jj] = v.size
        cgt, cm = ref[1], ref[2]
        for jj in range(SC.N_CONS):
            v = cgt[cm[:, jj], jj]
            r["cons_sum"][jj] = v.sum(); r["cons_sumsq"][jj] = (v ** 2).sum(); r["cons_n"][jj] = v.size
        ref_parts.append(r)
    tot = {k: sum(r[k] for r in ref_parts) for k in ref_parts[0]}
    means = (tot["psi_sum"] / tot["psi_n"],
             np.where(tot["scal_n"] > 0, tot["scal_sum"] / np.maximum(tot["scal_n"], 1), 0.0),
             np.where(tot["cons_n"] > 0, tot["cons_sum"] / np.maximum(tot["cons_n"], 1), 0.0))
    jobs = [(k, g, p, 1.0, means) for k, g, p in zip(keys, gts, preds)]
    with Pool(nproc) as pool:
        parts = list(pool.imap(SC._score_worker, jobs, chunksize=2))
    return parts, ref_parts, refs


def _finalize(parts, ref_parts, sel):
    from fec import score as SC
    acc = SC.Accum(); acc.psi_sign = 1.0
    for i in sel:
        acc.add(parts[i])
    rs = {k: sum(ref_parts[i][k] for i in sel) for k in ref_parts[0]}
    return SC.finalize_machine(acc, rs)


def cmd_valgate(a):
    """Stage-2 gate: official S (psi fp32), paired shot bootstrap, gated li SSE, identity."""
    from multiprocessing import Pool
    from fec.features import load_ds
    from fec import score as SC
    ds = load_ds("diii_d_train")
    bd, cd = WORK / "models" / a.base, WORK / "models" / a.cand
    idx = np.load(bd / "val_idx.npy"); assert np.array_equal(idx, np.load(cd / "val_idx.npy"))
    gv = np.load(cd / "val_g.npy")
    Bp = np.load(bd / "val_psi.npy", mmap_mode="r"); Cp = np.load(cd / "val_psi.npy", mmap_mode="r")
    Bs = np.load(bd / "val_scal.npy"); Cs = np.load(cd / "val_scal.npy")
    keep = np.flatnonzero(gv <= 0)
    ident = (np.asarray(Bp[keep], np.float32).tobytes() == np.asarray(Cp[keep], np.float32).tobytes()
             and Bs[keep].tobytes() == Cs[keep].tobytes())
    t0 = time.time()
    kB, gB, pB, pos = _val_parts(a.base, ds)
    kC, gC, pC, _ = _val_parts(a.cand, ds)
    assert kB == kC
    parts_b, ref_b, refs = _score_parts(kB, gB, pB, a.nproc)
    parts_c, ref_c, _ = _score_parts(kC, gC, pC, a.nproc)
    n = len(kB); full = np.arange(n)
    rb, rc = _finalize(parts_b, ref_b, full), _finalize(parts_c, ref_c, full)
    rng = np.random.default_rng(0)
    d = []
    for _ in range(a.boot):
        s = rng.integers(0, n, n)
        d.append(_finalize(parts_c, ref_c, s)["S"] - _finalize(parts_b, ref_b, s)["S"])
    d = np.array(d)
    # per-frame li on changed frames (official per-frame functionals)
    jobs, meta = [], []
    for si, j in enumerate(pos):
        ch = np.flatnonzero(gv[j] > 0)
        if len(ch) == 0:
            continue
        contours, cons_gt, cmask, rgeo = refs[si]
        for arr in (Bp, Cp):
            jobs.append((np.asarray(arr[j[ch]], np.float32).astype(np.float64),
                         [contours[k] for k in ch], [rgeo] * len(ch)))
        meta.append((si, j[ch], cons_gt[ch, 6], cmask[ch, 6]))
    with Pool(a.nproc) as pool:
        res = pool.map(_pred_worker, jobs)
    gate_rd = np.load(WORK / "ds_b" / "diii_d_train" / "gate_rd.npy")[idx]
    li = {"g>0": [0.0, 0.0, 0], "gate_rd": [0.0, 0.0, 0]}
    mean_li = ref_b and (sum(r["cons_sum"][6] for r in ref_b) / sum(r["cons_n"][6] for r in ref_b))
    for m_i, (si, rows_v, lgt, lm) in enumerate(meta):
        lb, lc = res[2 * m_i][1][:, 6], res[2 * m_i + 1][1][:, 6]
        lb = np.where(np.isfinite(lb), lb, mean_li); lc = np.where(np.isfinite(lc), lc, mean_li)
        for key, msk in (("g>0", lm), ("gate_rd", lm & gate_rd[rows_v])):
            li[key][0] += float(((lb - lgt) ** 2)[msk].sum()); li[key][1] += float(((lc - lgt) ** 2)[msk].sum())
            li[key][2] += int(msk.sum())
    li_r2_b, li_r2_c = rb["r2_cons_each"]["li"], rc["r2_cons_each"]["li"]
    passed = bool(d.mean() > 2 * d.std() and li_r2_c > li_r2_b and ident)
    out = {"base": a.base, "cand": a.cand, "n_shots": n, "n_changed": int((gv > 0).sum()),
           "identity_g0": bool(ident), "base_res": rb, "cand_res": rc,
           "dS": rc["S"] - rb["S"], "boot": {"B": a.boot, "mean": float(d.mean()), "sd": float(d.std()),
                                              "q025": float(np.quantile(d, .025)),
                                              "q975": float(np.quantile(d, .975)),
                                              "frac_pos": float((d > 0).mean())},
           "li_sse": {k: {"base": v[0], "cand": v[1], "n": v[2]} for k, v in li.items()},
           "PASS": passed, "seconds": time.time() - t0}
    (cd / "valgate.json").write_text(json.dumps(out, indent=1, default=float))
    for lab, r in (("base", rb), ("cand", rc)):
        print(f"{lab}: S {r['S']:.6f} R2psi {r['r2_psi']:.6f} qb {r['r2_qb']:.6f} D {r['dlcfs']:.6f} "
              f"cons {r['consistency']:.6f} | " + " ".join(f"{k} {v:.5f}" for k, v in r["r2_cons_each"].items()))
    print(f"dS {out['dS']:+.6f}; bootstrap mean {d.mean():+.6f} SD {d.std():.6f} "
          f"[{out['boot']['q025']:+.6f}, {out['boot']['q975']:+.6f}] frac>0 {out['boot']['frac_pos']:.3f}")
    print("li SSE", {k: (round(v[0], 3), round(v[1], 3), v[2]) for k, v in li.items()},
          "identity g=0:", ident, "PASS:", passed, flush=True)


# ------------------------------------------------- li-range repair (doc §6.1, root-approved)
LI_LO, LI_HI, LI_TARGET, LI_ITERS = 0.1, 5.0, 4.9, 10


def env_train_path():
    return WORK / "ds_b" / "diii_d_train" / "env_train.npz"


def load_env(path=None):
    """(R, Z, mask bool, mask float) of the TRAIN-only envelope (fec/bnd_b.py env_train.npz).
    Never the scorer's d3d_envelope.npz (root ruling (b), 2026-10-09)."""
    mk = np.load(path or env_train_path())
    env = mk["env"].astype(bool)
    return (np.asarray(mk["grid_R"], np.float64), np.asarray(mk["grid_Z"], np.float64),
            env, env.astype(np.float64))


def frame_li(psi, env):
    """(valid LCFS, raw li(2)) of one 65x65 map: the starter's LCFS extraction (axis_sign -1,
    n_iter 22) inside `env` and derive.internal_inductance (unclamped)."""
    from fec import score as SC          # puts the starter / fusion_scoring on sys.path
    import derive as DV
    R, Z, mc, mf = env
    p = np.asarray(psi, np.float64).reshape(65, 65)
    ex, _ = SC.LS.extract_lcfs_with_sign(p, R, Z, "DIII-D", mc, mf, n_iter=SC.LS.N_ITER,
                                         axis_sign=-1.0)
    if ex is None:
        return False, float("nan")
    return True, float(DV.internal_inductance(p, R, Z, "DIII-D", ex))


def _li_chunk(args):
    src, i, j, env_path = args
    psi = np.load(src, mmap_mode="r") if isinstance(src, str) else src
    env = load_env(env_path)
    blk = np.asarray(psi[i:j]) if isinstance(src, str) else psi
    v = np.zeros(len(blk), bool); li = np.full(len(blk), np.nan)
    for k in range(len(blk)):
        v[k], li[k] = frame_li(blk[k], env)
    return v, li


def derived_li(psi, env_path=None, nproc=6, chunk=500):
    """Per-frame (valid, raw li) of a (N,65,65) array or .npy path (workers memory-map a path)."""
    from multiprocessing import Pool
    ep = str(env_path or env_train_path())
    n = len(np.load(psi, mmap_mode="r")) if isinstance(psi, (str, Path)) else len(psi)
    if isinstance(psi, (str, Path)):
        jobs = [(str(psi), i, min(n, i + chunk), ep) for i in range(0, n, chunk)]
    else:
        jobs = [(np.asarray(psi[i:i + chunk]), 0, None, ep) for i in range(0, n, chunk)]
    with Pool(nproc) as pool:
        res = pool.map(_li_chunk, jobs, chunksize=1)
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


def li_trigger(valid, li, lo=LI_LO, hi=LI_HI):
    """Frames whose (valid-LCFS) derived li is outside [lo, hi] (the scorer would NaN-clamp)."""
    li = np.asarray(li, np.float64)
    with np.errstate(invalid="ignore"):
        return np.asarray(valid, bool) & np.isfinite(li) & ((li < lo) | (li > hi))


def find_donors(rows, shot, t_ms, donor_ok):
    """For each row: the donor_ok frame of the same shot with the largest time < its time
    (-1 if none)."""
    shot = np.asarray(shot); t_ms = np.asarray(t_ms); donor_ok = np.asarray(donor_ok, bool)
    out = np.full(len(rows), -1, np.int64)
    for n, r in enumerate(rows):
        c = np.flatnonzero((shot == shot[r]) & donor_ok & (t_ms < t_ms[r]))
        if len(c):
            out[n] = c[np.argmax(t_ms[c])]
    return out


def bisect_repair(psi_t, psi_d, li_fn, lo=LI_LO, target=LI_TARGET, iters=LI_ITERS):
    """Smallest alpha (to 2**-iters) with psi_t + alpha (psi_d - psi_t) feasible (valid LCFS, li in
    [lo, target]); psi_d is assumed feasible (alpha = 1). li_fn(map) -> (valid, li).
    Returns (map float32, alpha, li)."""
    pt = np.asarray(psi_t, np.float64); pd = np.asarray(psi_d, np.float64)
    a_lo, a_hi = 0.0, 1.0
    l_hi = li_fn(pd)[1]
    for _ in range(iters):
        m = 0.5 * (a_lo + a_hi)
        v, li = li_fn(pt + m * (pd - pt))
        if v and np.isfinite(li) and lo <= li <= target:
            a_hi, l_hi = m, li
        else:
            a_lo = m
    return (pt + a_hi * (pd - pt)).astype(np.float32), a_hi, l_hi


def li_repair(psi, shot, t_ms, valid, li, li_fn, lo=LI_LO, hi=LI_HI, target=LI_TARGET,
              iters=LI_ITERS):
    """Repair (i): pure function of the psi array (N,65,65), shot/time index and an envelope
    (inside li_fn). valid/li = derived_li(psi). Returns (rows, maps (n,65,65) f32, info list)."""
    trig = np.flatnonzero(li_trigger(valid, li, lo, hi))
    with np.errstate(invalid="ignore"):
        donor_ok = np.asarray(valid, bool) & (np.asarray(li) >= lo) & (np.asarray(li) <= target)
    donors = find_donors(trig, shot, t_ms, donor_ok)
    rows, maps, info = [], [], []
    for r, d in zip(trig, donors):
        rec = {"row": int(r), "li_before": float(li[r]), "donor": int(d)}
        if d < 0:
            rec["status"] = "no_donor"
        else:
            m, a, l2 = bisect_repair(np.asarray(psi[r]), np.asarray(psi[d]), li_fn, lo, target, iters)
            rows.append(int(r)); maps.append(m.reshape(65, 65))
            rec.update(status="repaired", alpha=float(a), li_after=float(l2),
                       dt_ms=float(np.asarray(t_ms)[r] - np.asarray(t_ms)[d]))
        info.append(rec)
    return np.array(rows, np.int64), (np.stack(maps) if maps else np.zeros((0, 65, 65), np.float32)), info


def member_repair(member_maps, weights, li_fn, lo=LI_LO, hi=LI_HI):
    """Repair (ii) of one frame: weighted mean of the member maps whose own li is in [lo, hi]
    (weights renormalised); None if no member is in range."""
    w = np.asarray(weights, np.float64)
    ok = []
    for m in member_maps:
        v, li = li_fn(m)
        ok.append(bool(v and np.isfinite(li) and lo <= li <= hi))
    ok = np.array(ok)
    if not ok.any():
        return None
    ww = w * ok
    return np.tensordot(ww / ww.sum(), np.asarray(member_maps, np.float64), axes=1).astype(np.float32)


def apply_rows(psi, rows, maps):
    """float32 copy of psi with rows replaced (all other rows byte-identical)."""
    out = np.array(psi, dtype=np.float32, copy=True)
    if len(rows):
        out[np.asarray(rows)] = np.asarray(maps, np.float32).reshape((len(rows),) + out.shape[1:])
    return out


def _env_li_fn(env_path=None):
    env = load_env(env_path)
    return lambda m: frame_li(m, env)


def _ref_rows(rows, shot_of_rows, psi_lab, nproc):
    """GT reference (scorer code; evaluation only) for frames `rows` of TRAIN shots, with the
    per-shot rgeo over ALL frames of each shot (as build_reference)."""
    from multiprocessing import Pool
    from fec.features import load_ds
    ds = load_ds("diii_d_train")
    G = np.asarray(psi_lab[rows], np.float64)
    with Pool(nproc) as pool:
        gt = sum(pool.map(_gt_worker, [G[s] for s in _chunks(len(rows))]), [])
        rge = {}
        for s in np.unique(shot_of_rows):
            fr = np.flatnonzero(ds["shot"] == s)
            Gs = np.asarray(psi_lab[fr], np.float64)
            res = sum(pool.map(_gt_worker, [Gs[c] for c in _chunks(len(fr))]), [])
            r = np.array([x[2] for x in res])
            rge[s] = float(np.nanmean(r)) if np.isfinite(r).any() else np.nan
    return ([x[0] for x in gt], np.array([x[1] for x in gt]),
            np.array([rge[s] for s in shot_of_rows]), G)


def _frame_parts(P, G, gq, Q, refc, cg, rgeo, shot, mean_c, nproc):
    from multiprocessing import Pool
    with Pool(nproc) as pool:
        res = pool.map(_pred_worker, [(P[s].reshape(-1, 65, 65).astype(np.float64), refc[s], rgeo[s])
                                      for s in _chunks(len(P))])
    d = np.concatenate([r[0] for r in res]); cp = np.concatenate([r[1] for r in res])
    err = (cg - cp) ** 2
    err = np.where(np.isfinite(cg) & ~np.isfinite(cp), (cg - mean_c) ** 2, err)
    err = np.where(np.isfinite(cg), err, np.nan)
    return {"psi_sse": ((G.reshape(len(P), -1) - P.reshape(len(P), -1).astype(np.float64)) ** 2).sum(1),
            "q": (gq - Q[:, :2]) ** 2, "cons": err, "d": d, "shot": shot, "cons_pred": cp}


def cmd_lirep_h(a):
    """§6.1 H stage: trigger/repair on fold-base psi of ALL H frames (TRAIN labels for scoring)."""
    from multiprocessing import Pool
    from fec.features import load_ds
    from fec import score as SC
    t0 = time.time()
    H = load_h()
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])
    assert (splits[H] == "train").all()
    rowsH = np.flatnonzero(np.isin(ds["shot"], H))
    od = WORK / "models" / "rdspec_H" / "lirep"
    od.mkdir(parents=True, exist_ok=True)
    parents = ["rdspec_fold_shape_s0", "rdspec_fold_notsL_s5"]
    pp, qp = od / "base_psi_H.npy", od / "base_scal_H.npy"
    if not (od / "base_DONE").exists():
        P = np.lib.format.open_memmap(pp, mode="w+", dtype=np.float32, shape=(len(rowsH), 65, 65))
        Qa = np.zeros((len(rowsH), 5), np.float32)
        for i in range(0, len(rowsH), 20000):
            p, q = predict_rows(parents, [1.0, 1.0], rowsH[i:i + 20000])
            P[i:i + 20000] = p.reshape(-1, 65, 65).astype(np.float32); Qa[i:i + 20000] = q
        P.flush(); del P
        np.save(qp, Qa); (od / "base_DONE").write_text("ok")
        print(f"base psi for {len(rowsH)} H frames {time.time() - t0:.0f}s", flush=True)
    if not (od / "li_H.npz").exists():
        v, li = derived_li(str(pp), nproc=a.nproc)
        np.savez(od / "li_H.npz", valid=v, li=li)
        print(f"derived li (env_train) {time.time() - t0:.0f}s", flush=True)
    z = np.load(od / "li_H.npz"); valid, li = z["valid"], z["li"]
    P = np.load(pp, mmap_mode="r"); Qb = np.load(qp)
    shot_l, t_l = ds["shot"][rowsH], ds["t_ms"][rowsH]
    li_fn = _env_li_fn()
    trig = np.flatnonzero(li_trigger(valid, li))
    g, _, _ = ds_gate(ds, no_ts_zero=False)
    print(f"H: {len(rowsH)} frames, LCFS invalid (env_train) {int((~valid).sum())}, triggered "
          f"{len(trig)} (li>5 {int((li[trig] > LI_HI).sum())}, <0.1 {int((li[trig] < LI_LO).sum())}; "
          f"g=1 {int((g[rowsH[trig]] >= 1).sum())}, g>0 {int((g[rowsH[trig]] > 0).sum())}) in "
          f"{len(np.unique(shot_l[trig]))} shots", flush=True)
    rows_i, maps_i, info_i = li_repair(P, shot_l, t_l, valid, li, li_fn)
    # (ii): member maps on the triggered frames
    mm = np.stack([predict_rows([t], [1.0], rowsH[trig])[0].reshape(-1, 65, 65) for t in parents])
    rows_ii, maps_ii, info_ii = [], [], []
    for n, r in enumerate(trig):
        m = member_repair(mm[:, n], [1.0, 1.0], li_fn)
        info_ii.append({"row": int(r), "status": "repaired" if m is not None else "none_in_range"})
        if m is not None:
            rows_ii.append(int(r)); maps_ii.append(m.reshape(65, 65))
            info_ii[-1]["li_after"] = li_fn(m)[1]
    rows_ii = np.array(rows_ii, np.int64)
    maps_ii = np.stack(maps_ii) if maps_ii else np.zeros((0, 65, 65), np.float32)
    print(f"repair (i): {len(rows_i)} repaired, {sum(r['status'] == 'no_donor' for r in info_i)} no donor; "
          f"repair (ii): {len(rows_ii)} repaired  {time.time() - t0:.0f}s", flush=True)
    # ---- evaluation (labels of H = TRAIN; scorer functionals, evaluation only)
    psi_lab, scal_lab = _labels()
    ev = np.unique(np.r_[rows_i, rows_ii]).astype(np.int64)
    gr = rowsH[ev]
    refc, cg, rgeo, G = _ref_rows(gr, ds["shot"][gr], psi_lab, a.nproc)
    rng = np.random.default_rng(1)
    rnd = np.sort(rng.choice(rowsH, a.nsst, replace=False))
    Grnd = np.asarray(psi_lab[rnd], np.float64)
    with Pool(a.nproc) as pool:
        gtr = sum(pool.map(_gt_worker, [Grnd[s] for s in _chunks(len(rnd))]), [])
    cr = np.array([x[1] for x in gtr]); mean_c = np.nanmean(cr, 0)
    nH = len(rowsH)
    sst = {"psi": nH * 4225 * Grnd.var(), "q": nH * np.nanvar(scal_lab[rnd][:, :2], 0),
           "cons": nH * np.nanvar(cr, 0) * np.isfinite(cr).mean(0)}
    fps = {s: int(n) for s, n in zip(*np.unique(ds["shot"][rowsH], return_counts=True))}
    gq = scal_lab[gr][:, :2].astype(np.float64)
    shot_ev = ds["shot"][gr]
    Pb = np.asarray(P[ev], np.float32)
    pb = _frame_parts(Pb, G, gq, Qb[ev], refc, cg, rgeo, shot_ev, mean_c, a.nproc)
    res = {"H_frames": int(nH), "triggered": int(len(trig)), "evaluated_rows": int(len(ev)),
           "invalid_lcfs_env_train": int((~valid).sum()), "cands": {}}
    store = {"rowsH_ev": gr, "cons_gt": cg, "base_cons": pb["cons_pred"], "base_d": pb["d"],
             "base_psi_sse": pb["psi_sse"], "trig_rowsH": rowsH[trig], "trig_li": li[trig]}
    names = SC.LS.CONS_SCALARS
    for cname, rws, mps, info in (("i", rows_i, maps_i, info_i), ("ii", rows_ii, maps_ii, info_ii)):
        Pc = Pb.copy()
        pos = np.searchsorted(ev, rws)
        Pc[pos] = mps
        pc = _frame_parts(Pc, G, gq, Qb[ev], refc, cg, rgeo, shot_ev, mean_c, a.nproc)
        w = pooled_delta(pb, pc, sst, H, fps)
        brng = np.random.default_rng(0)
        idx_by = {s: np.flatnonzero(shot_ev == s) for s in np.unique(shot_ev)}
        Hl = list(H); boots = []
        for _ in range(a.boot):
            pick = brng.choice(len(Hl), len(Hl), replace=True)
            sel = [idx_by[Hl[i]] for i in pick if Hl[i] in idx_by]
            sel = np.concatenate(sel) if sel else np.zeros(0, int)
            boots.append(pooled_delta({k: v[sel] for k, v in pb.items()},
                                      {k: v[sel] for k, v in pc.items()}, sst, H, fps)["S"])
        boots = np.array(boots)
        m = np.isin(ev, rws)
        li_b = float(np.nansum(pb["cons"][m, 6])); li_c = float(np.nansum(pc["cons"][m, 6]))
        still_nan = int((np.isfinite(cg[m, 6]) & ~np.isfinite(pc["cons_pred"][m, 6])).sum())
        per_frame = []
        for k in np.flatnonzero(m):
            per_frame.append({"rowH": int(gr[k]), "shot": int(shot_ev[k]),
                              "li_gt": float(cg[k, 6]), "li_base": float(pb["cons_pred"][k, 6]),
                              "li_cand": float(pc["cons_pred"][k, 6]),
                              "d_psi_sse": float(pc["psi_sse"][k] - pb["psi_sse"][k]),
                              "d_dlcfs": float(pc["d"][k] - pb["d"][k]),
                              "d_sqerr": {nm: float(pc["cons"][k, j] - pb["cons"][k, j])
                                          for j, nm in enumerate(names)}})
        passed = bool(w["S"] > 2 * boots.std() and li_c < li_b)
        res["cands"][cname] = {
            "n_repaired": int(len(rws)), "dS_H": w["S"],
            "dS_terms": {k: w[k] for k in ("psi", "qb", "cons", "dlcfs")},
            "d_cons_each_R2": dict(zip(names, w["d_cons_each"])),
            "boot": {"B": a.boot, "mean": float(boots.mean()), "sd": float(boots.std()),
                     "frac_pos": float((boots > 0).mean())},
            "li_sse_trig": {"base": li_b, "cand": li_c},
            "still_nan_scorer": still_nan, "PASS": passed, "info": info, "per_frame": per_frame}
        store[f"{cname}_cons"] = pc["cons_pred"]; store[f"{cname}_d"] = pc["d"]
        print(f"({cname}) repaired {len(rws)}: dS_H {w['S']:+.6f} boot SD {boots.std():.6f} frac>0 "
              f"{(boots > 0).mean():.2f}; li SSE trig {li_b:.2f} -> {li_c:.2f}; still NaN (scorer) "
              f"{still_nan}; terms " + " ".join(f"{k} {w[k]:+.6f}" for k in ("psi", "qb", "cons", "dlcfs"))
              + " | dR2 " + " ".join(f"{nm} {x * 1e6:+.0f}e-6" for nm, x in zip(names, w["d_cons_each"]))
              + f"  PASS={passed}", flush=True)
    (od / "lirep_h.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez(od / "lirep_h_frames.npz", **store)
    print("LIREP_H_DONE", f"{time.time() - t0:.0f}s", flush=True)


def li_repair_iii(psi, shot, t_ms, valid, li, env, li_fn, lo=LI_LO, hi=LI_HI, target=LI_TARGET,
                  iters=LI_ITERS):
    """Repair (iii), doc §6.3: lishift's amendment-A1 boundary-preserving interior relabelling
    (fec/d3d_lishift.py, imported read-only: G, LCFS contour and shape scalars unchanged) solved
    for li = target on each triggered frame; frames the transform cannot reach fall back to (i).
    Pure function of (psi, shot/time index, envelope). Returns (rows, maps, info)."""
    from fec import d3d_lishift as LSH
    trig = np.flatnonzero(li_trigger(valid, li, lo, hi))
    with np.errstate(invalid="ignore"):
        donor_ok = np.asarray(valid, bool) & (np.asarray(li) >= lo) & (np.asarray(li) <= target)
    rows, maps, info, fb = [], [], [], []
    for r in trig:
        base = np.asarray(psi[r], np.float32).reshape(65, 65)
        geo = LSH.lcfs_geometry(base, env)
        st, eps = "no_lcfs", 0.0
        if geo is not None:
            eps, _, st = LSH.solve_eps(base, target, float(li[r]), geo, li_fn)
        if geo is not None and st == "ok":
            m = LSH.relabel(base, eps, geo)
            v2, l2 = li_fn(m)
            if v2 and np.isfinite(l2) and lo <= l2 <= hi:
                rows.append(int(r)); maps.append(m)
                info.append({"row": int(r), "li_before": float(li[r]), "status": "transform",
                             "eps": float(eps), "li_after": float(l2)})
                continue
            st = "verify_failed"
        fb.append((int(r), st))
    donors = find_donors(np.array([r for r, _ in fb], np.int64), shot, t_ms, donor_ok)
    for (r, st), d in zip(fb, donors):
        rec = {"row": r, "li_before": float(li[r]), "transform_status": st, "donor": int(d)}
        if d < 0:
            rec["status"] = "no_donor"
        else:
            m, a, l2 = bisect_repair(np.asarray(psi[r]), np.asarray(psi[d]), li_fn, lo, target, iters)
            rows.append(r); maps.append(m.reshape(65, 65))
            rec.update(status="fallback_i", alpha=float(a), li_after=float(l2),
                       dt_ms=float(np.asarray(t_ms)[r] - np.asarray(t_ms)[d]))
        info.append(rec)
    order = np.argsort(rows) if rows else np.zeros(0, int)
    rows = np.array(rows, np.int64)[order]
    maps = np.stack(maps)[order] if len(maps) else np.zeros((0, 65, 65), np.float32)
    return rows, maps, info


SHAPE_KEYS = ("kappa", "tri_top", "tri_bot", "volume", "R_axis", "Z_axis")


def frame_invariants(psi, env):
    """(contour, Ampere G, shape scalars dict) of one map under `env` with the starter's
    functionals (LCFS extraction axis_sign -1, derive_frame without li). None if no LCFS."""
    from fec import score as SC
    from fec import d3d_lishift as LSH
    R, Z, mc, mf = env
    p = np.asarray(psi, np.float64).reshape(65, 65)
    ex, _ = SC.LS.extract_lcfs_with_sign(p, R, Z, "DIII-D", mc, mf, n_iter=SC.LS.N_ITER, axis_sign=-1.0)
    if ex is None:
        return None
    v = SC.LS.derive_frame(p, R, Z, "DIII-D", mc, mf, contour=ex, with_li=False, axis_sign=-1.0)
    return ex, LSH.ampere_G(p, env, ex), {k: float(v[k]) for k in SHAPE_KEYS}


EXACT_KEYS = ("kappa", "tri_top", "tri_bot", "volume")
AXIS_KEYS = ("R_axis", "Z_axis")
AXIS_BOUND_M = 0.023                # one grid cell: sanity bound on the axis sub-pixel shift (§6.5a)


def invariants_equal(a, b, keys=SHAPE_KEYS):
    """Exact equality of (contour, G, scalars in `keys`) tuples (NaN == NaN)."""
    if a is None or b is None:
        return False
    if a[0].shape != b[0].shape or not np.array_equal(a[0], b[0]) or a[1] != b[1]:
        return False
    return all((a[2][k] == b[2][k]) or (np.isnan(a[2][k]) and np.isnan(b[2][k])) for k in keys)


def axis_shift(a, b):
    """max |delta| of R_axis, Z_axis between two invariant tuples."""
    return max(abs(a[2][k] - b[2][k]) for k in AXIS_KEYS)


def li_repair_iv(psi, valid, li, env, li_fn, lo=LI_LO, hi=LI_HI, target=LI_TARGET, check=True):
    """Repair (iv), doc §6.5: lishift amendment-A1 relabelling (fec/d3d_lishift.py, read-only) to
    li = target on triggered frames where it reaches it (solver status 'ok', verified li in
    [lo, target + lishift LI_TOL]); out-of-reach frames are LEFT UNCHANGED (no fallback). With
    check=True each repaired frame must keep its env_train LCFS contour, Ampere G and shape scalars
    exactly (AssertionError otherwise). Pure function of (psi, envelope). Returns (rows, maps, info)."""
    from fec import d3d_lishift as LSH
    trig = np.flatnonzero(li_trigger(valid, li, lo, hi))
    rows, maps, info = [], [], []
    for r in trig:
        base = np.asarray(psi[r], np.float32).reshape(65, 65)
        rec = {"row": int(r), "li_before": float(li[r])}
        geo = LSH.lcfs_geometry(base, env)
        if geo is None:
            rec["status"] = "left_out_no_lcfs"; info.append(rec); continue
        eps, _, st = LSH.solve_eps(base, target, float(li[r]), geo, li_fn)
        if st != "ok":
            rec.update(status="left_out_unreachable", solver=st); info.append(rec); continue
        m = LSH.relabel(base, eps, geo)
        v2, l2 = li_fn(m)
        if not (v2 and np.isfinite(l2) and lo <= l2 <= target + LSH.LI_TOL):
            rec.update(status="left_out_verify", li_try=float(l2)); info.append(rec); continue
        if check:
            # §6.5a: contour, G, kappa, delta_top/bot, volume exactly unchanged; the axis is refined
            # sub-pixel from neighbours of the peak node (inside the relabelled set) and may move.
            ia, ib = frame_invariants(base, env), frame_invariants(m, env)
            assert invariants_equal(ia, ib, EXACT_KEYS), \
                f"row {r}: contour/G/kappa/delta/volume changed by the relabelling"
            rec["axis_shift_m"] = axis_shift(ia, ib)
            assert rec["axis_shift_m"] < AXIS_BOUND_M, f"row {r}: axis moved {rec['axis_shift_m']}"
        rows.append(int(r)); maps.append(m)
        rec.update(status="repaired", eps=float(eps), li_after=float(l2))
        info.append(rec)
    return (np.array(rows, np.int64),
            np.stack(maps) if maps else np.zeros((0, 65, 65), np.float32), info)


def _same_bytes(path_a, path_b, chunk=20000):
    a = np.load(path_a, mmap_mode="r"); b = np.load(path_b, mmap_mode="r")
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return all(np.asarray(a[i:i + chunk]).tobytes() == np.asarray(b[i:i + chunk]).tobytes()
               for i in range(0, len(a), chunk))


def cmd_lirep_build4(a):
    """§6.5 (iv) label-free build: mirror a q95-hybrid preds dir, replace only <config>_psi.npy."""
    import shutil
    from fec import score as SC
    from fec.features import load_ds
    t0 = time.time()
    src, dst = WORK / "preds" / a.src, WORK / "preds" / a.out
    sp = src / f"{a.config}_psi.npy"
    if (dst / f"{a.config}_psi.npy").exists():
        raise SystemExit(f"{dst} already has {a.config}_psi.npy (never overwrite)")
    same = None
    if a.base:
        same = _same_bytes(sp, WORK / "preds" / a.base / f"{a.config}_psi.npy")
        print(f"{a.src} psi byte-identical to {a.base} psi: {same}", flush=True)
    if same and a.li_cache:
        z = np.load(a.li_cache); valid, li = z["valid"], z["li"]
    else:                                   # repair the q95h psi itself (recompute li on it)
        valid, li = derived_li(str(sp), nproc=a.nproc)
    ds = load_ds(a.config)
    assert len(valid) == ds["info"]["N"]
    psi = np.load(sp, mmap_mode="r")
    env = load_env()
    rows, maps, info = li_repair_iv(psi, valid, li, env, lambda m: frame_li(m, env), check=True)
    out = apply_rows(psi, rows, maps)
    keep = np.setdiff1d(np.arange(len(psi)), rows)
    dmax = 0.0
    for i in range(0, len(keep), 20000):
        k = keep[i:i + 20000]
        assert out[k].tobytes() == np.asarray(psi[k], np.float32).tobytes()
        dmax = max(dmax, float(np.abs(out[k] - np.asarray(psi[k], np.float32)).max()))
    assert np.isfinite(out).all()
    # evaluation-style checks with the scorer mask (no labels, no refs)
    senv = SC._mask()
    per = []
    for rec in info:
        r = rec["row"]
        e = {"row": r, "status": rec["status"], "li_env_train_raw": rec["li_before"],
             "li_env_train_after": rec.get("li_after"),
             "li_scorer_raw": frame_li(psi[r], senv)[1]}
        if rec["status"] == "repaired":
            j = int(np.flatnonzero(rows == r)[0])
            e["li_scorer_after"] = frame_li(maps[j], senv)[1]
            ia, ib = frame_invariants(psi[r], senv), frame_invariants(maps[j], senv)
            e["scorer_contour_equal"] = bool(np.array_equal(ia[0], ib[0]))
            e["scorer_G_equal"] = bool(ia[1] == ib[1])
            e["scorer_max_abs_delta"] = {k: abs(ia[2][k] - ib[2][k]) for k in SHAPE_KEYS}
            e["env_train_axis_shift_m"] = rec.get("axis_shift_m")
            e["eps"] = rec["eps"]
        e["g"] = None
        per.append(e)
    g, _, _ = ds_gate(ds, no_ts_zero=False)
    for e in per:
        e["g"] = float(g[e["row"]])
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != f"{a.config}_psi.npy":
            shutil.copyfile(f, dst / f.name)
    np.save(dst / f"{a.config}_psi.npy", out)
    meta = {"kind": "li-range repair (iv) transform-only, doc D3D_RAMPDOWN_SPECIALIST_20261009 §6.5/§6.5a",
            "src": str(src), "src_psi_identical_to_base": same, "base": a.base, "config": a.config,
            "env": str(env_train_path()), "target": LI_TARGET, "triggered": len(info),
            "repaired": int(len(rows)), "left_out": int(len(info) - len(rows)),
            "max_abs_dpsi_outside_repaired": dmax, "frames": per,
            "copied_unchanged": sorted(f.name for f in src.iterdir() if f.name != f"{a.config}_psi.npy"),
            "seconds": time.time() - t0}
    (dst / f"{a.config}_lirep4_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    for e in per:
        print(json.dumps(e, default=float), flush=True)
    print(f"LIREP_BUILD4_DONE {dst}: triggered {len(info)}, repaired {len(rows)}, left out "
          f"{len(info) - len(rows)}, max|dpsi| outside {dmax}, {time.time() - t0:.0f}s", flush=True)


def run_repair(cand, psi, shot, t_ms, valid, li, env_path=None):
    """Dispatch: cand 'i' (§6.1), 'iii' (§6.3) or 'iv' (§6.5) on an indexable psi array."""
    env = load_env(env_path)
    li_fn = lambda m: frame_li(m, env)  # noqa: E731
    if cand == "i":
        return li_repair(psi, shot, t_ms, valid, li, li_fn)
    if cand == "iii":
        return li_repair_iii(psi, shot, t_ms, valid, li, env, li_fn)
    if cand == "iv":
        return li_repair_iv(psi, valid, li, env, li_fn)
    raise ValueError(cand)


def shot_deltas(pb, pc, fps):
    """Per-shot paired contributions (cand - base): dict shot -> vector
    [dpsi_sse, dq95, dbN, dcons x7, d_dlcfs_mean] (D_LCFS already divided by the shot's frames)."""
    out = {}
    for k, s in enumerate(pb["shot"]):
        v = np.zeros(11)
        v[0] = pc["psi_sse"][k] - pb["psi_sse"][k]
        v[1:3] = np.nan_to_num(pc["q"][k] - pb["q"][k])
        v[3:10] = np.nan_to_num(pc["cons"][k] - pb["cons"][k])
        v[10] = np.nan_to_num(pc["d"][k] - pb["d"][k]) / fps[int(s)]
        out[int(s)] = out.get(int(s), 0.0) + v
    return out


def terms_from(D, sst, n_shots):
    """S-proxy terms from summed per-shot deltas D (11,) (same formulas as pooled_delta)."""
    t = {"r2_psi": -D[0] / sst["psi"], "qb": float(np.mean(-D[1:3] / sst["q"])),
         "dlcfs": -D[10] / n_shots}                       # sign: + = better
    for j, nm in enumerate(["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]):
        t[f"cons_{nm}"] = -D[3 + j] / sst["cons"][j]
    cons = np.mean([t[f"cons_{nm}"] for nm in ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot",
                                                "volume", "li"]])
    t["S"] = 0.55 * t["r2_psi"] + 0.15 * t["qb"] + 0.20 * cons + 0.10 * t["dlcfs"]
    return t


def boot_gate(per_shot, shots_all, sst, B=1000, seed=0):
    """Point terms + paired whole-shot bootstrap SDs (resampling all fold shots; shots without
    changed frames contribute 0)."""
    shots_all = np.asarray(shots_all); n = len(shots_all)
    pos = {int(s): i for i, s in enumerate(shots_all)}
    V = np.zeros((n, 11))
    for s, v in per_shot.items():
        V[pos[s]] = v
    point = terms_from(V.sum(0), sst, n)
    rng = np.random.default_rng(seed)
    bs = {k: [] for k in point}
    for _ in range(B):
        c = np.bincount(rng.integers(0, n, n), minlength=n)
        t = terms_from(c @ V, sst, n)
        for k in point:
            bs[k].append(t[k])
    return {k: {"delta": float(point[k]), "sd": float(np.std(bs[k])),
                "frac_pos": float(np.mean(np.array(bs[k]) > 0))} for k in point}


def gate_verdict(bt, li_b, li_c):
    worse = [t for t in ("r2_psi", "dlcfs", "cons_kappa", "cons_tri_top", "cons_tri_bot", "cons_volume")
             if bt[t]["delta"] < 0 and abs(bt[t]["delta"]) > 2 * bt[t]["sd"]]
    ok = bool(bt["S"]["delta"] > 2 * bt["S"]["sd"] and li_c < li_b and not worse)
    return ok, worse


def holdout_eval(hshots, parents, od, nproc=6, nsst=5000, cands=("i", "iii")):
    """§6.3 per-holdout stage: fold-base psi on all holdout frames (cached), env_train li
    (cached), triggers, repairs per candidate, official per-frame functionals (labels = TRAIN
    holdout; scorer mask evaluation only) on the repaired frames, SSTs from a seed-1 sample."""
    from multiprocessing import Pool
    from fec.features import load_ds
    t0 = time.time()
    ds = load_ds("diii_d_train")
    splits = np.array(ds["info"]["splits"])
    assert (splits[hshots] == "train").all()
    rowsH = np.flatnonzero(np.isin(ds["shot"], hshots))
    od = Path(od); od.mkdir(parents=True, exist_ok=True)
    pp, qp = od / "base_psi_H.npy", od / "base_scal_H.npy"
    if not (od / "base_DONE").exists():
        P = np.lib.format.open_memmap(pp, mode="w+", dtype=np.float32, shape=(len(rowsH), 65, 65))
        Qa = np.zeros((len(rowsH), 5), np.float32)
        for i in range(0, len(rowsH), 20000):
            p, q = predict_rows(parents, [1.0] * len(parents), rowsH[i:i + 20000])
            P[i:i + 20000] = p.reshape(-1, 65, 65).astype(np.float32); Qa[i:i + 20000] = q
        P.flush(); del P
        np.save(qp, Qa); (od / "base_DONE").write_text("ok")
        print(f"[{od.parent.name}] base psi {len(rowsH)} frames {time.time() - t0:.0f}s", flush=True)
    if not (od / "li_H.npz").exists():
        v, li = derived_li(str(pp), nproc=nproc)
        np.savez(od / "li_H.npz", valid=v, li=li)
        print(f"[{od.parent.name}] derived li {time.time() - t0:.0f}s", flush=True)
    z = np.load(od / "li_H.npz"); valid, li = z["valid"], z["li"]
    P = np.load(pp, mmap_mode="r"); Qb = np.load(qp)
    shot_l, t_l = ds["shot"][rowsH], ds["t_ms"][rowsH]
    trig = np.flatnonzero(li_trigger(valid, li))
    reps = {c: run_repair(c, P, shot_l, t_l, valid, li) for c in cands}
    ev = np.unique(np.concatenate([reps[c][0] for c in cands] + [np.zeros(0, np.int64)])).astype(np.int64)
    psi_lab, scal_lab = _labels()
    gr = rowsH[ev]
    refc, cg, rgeo, G = _ref_rows(gr, ds["shot"][gr], psi_lab, nproc)
    rng = np.random.default_rng(1)
    rnd = np.sort(rng.choice(rowsH, nsst, replace=False))
    Grnd = np.asarray(psi_lab[rnd], np.float64)
    with Pool(nproc) as pool:
        gtr = sum(pool.map(_gt_worker, [Grnd[s] for s in _chunks(len(rnd))]), [])
    cr = np.array([x[1] for x in gtr]); mean_c = np.nanmean(cr, 0)
    nH = len(rowsH)
    sst = {"psi": nH * 4225 * Grnd.var(), "q": nH * np.nanvar(scal_lab[rnd][:, :2], 0),
           "cons": nH * np.nanvar(cr, 0) * np.isfinite(cr).mean(0)}
    fps = {int(s): int(n) for s, n in zip(*np.unique(ds["shot"][rowsH], return_counts=True))}
    gq = scal_lab[gr][:, :2].astype(np.float64)
    shot_ev = ds["shot"][gr]
    Pb = np.asarray(P[ev], np.float32)
    parts = {"base": _frame_parts(Pb, G, gq, Qb[ev], refc, cg, rgeo, shot_ev, mean_c, nproc)}
    for c in cands:
        rws, mps, _ = reps[c]
        Pc = Pb.copy(); Pc[np.searchsorted(ev, rws)] = mps
        parts[c] = _frame_parts(Pc, G, gq, Qb[ev], refc, cg, rgeo, shot_ev, mean_c, nproc)
    g, _, _ = ds_gate(ds, no_ts_zero=False)
    print(f"[{od.parent.name}] {nH} frames, triggered {len(trig)} (g=1 {int((g[rowsH[trig]] >= 1).sum())}) "
          f"in {len(np.unique(shot_l[trig]))} shots; repaired " +
          ", ".join(f"({c}) {len(reps[c][0])}" for c in cands) + f"  {time.time() - t0:.0f}s", flush=True)
    return {"H": np.asarray(hshots), "rowsH_ev": gr, "ev": ev, "cons_gt": cg, "parts": parts,
            "sst": sst, "fps": fps, "reps": {c: (reps[c][0], reps[c][2]) for c in cands},
            "n_frames": nH, "triggered": int(len(trig))}


def _cand_report(res, c, label):
    """Gate terms + per-frame deltas on the repaired frames for one candidate of one result set."""
    pb, pc = res["parts"]["base"], res["parts"][c]
    rws = res["reps"][c][0]
    m = np.isin(res["ev"], rws)
    bt = boot_gate(shot_deltas(pb, pc, res["fps"]), res["H"], res["sst"], B=res.get("B", 1000))
    li_b = float(np.nansum(pb["cons"][m, 6])); li_c = float(np.nansum(pc["cons"][m, 6]))
    ok, worse = gate_verdict(bt, li_b, li_c)
    names = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
    dsq = (pc["cons"] - pb["cons"])[m]
    pf = {"d_kappa_sq": dsq[:, 2], "d_tri_top_sq": dsq[:, 3], "d_tri_bot_sq": dsq[:, 4],
          "d_volume_sq": dsq[:, 5], "d_dlcfs": (pc["d"] - pb["d"])[m], "d_li_sq": dsq[:, 6]}
    still = int((np.isfinite(res["cons_gt"][m, 6]) & ~np.isfinite(pc["cons_pred"][m, 6])).sum())
    st = [r.get("status") for r in res["reps"][c][1]]
    out = {"label": label, "n_repaired": int(m.sum()), "status_counts": {s: st.count(s) for s in set(st)},
           "boot": bt, "li_sse_repaired": {"base": li_b, "cand": li_c}, "worse_beyond_noise": worse,
           "PASS": ok, "still_nan_scorer": still,
           "per_frame_summary": {k: {"sum": float(np.nansum(v)), "max": float(np.nanmax(v)) if len(v) else 0.0,
                                     "n_worse": int(np.nansum(v > 1e-12))} for k, v in pf.items()},
           "per_frame": {k: [float(x) for x in v] for k, v in pf.items()}}
    print(f"{label:22s} repaired {out['n_repaired']:3d} {out['status_counts']}: dS {bt['S']['delta']:+.7f} "
          f"SD {bt['S']['sd']:.7f} ({bt['S']['delta'] / max(bt['S']['sd'], 1e-30):.2f} SD, frac>0 "
          f"{bt['S']['frac_pos']:.2f}); li SSE {li_b:.2f} -> {li_c:.2f}; still NaN {still}; worse {worse}; "
          + " ".join(f"{k} {bt[k]['delta'] * 1e6:+.0f}e-6" for k in
                     ("r2_psi", "dlcfs", "cons_kappa", "cons_tri_top", "cons_tri_bot", "cons_volume",
                      "cons_R_axis", "cons_Z_axis")) + f"  PASS={ok}", flush=True)
    return out


def pool_results(r1, r2):
    """H u H2 as one fold: concatenated per-frame parts, summed SSTs, union of shots."""
    out = {"H": np.concatenate([r1["H"], r2["H"]]), "ev": None, "fps": {**r1["fps"], **r2["fps"]},
           "sst": {k: r1["sst"][k] + r2["sst"][k] for k in r1["sst"]},
           "cons_gt": np.concatenate([r1["cons_gt"], r2["cons_gt"]]), "parts": {}, "reps": {}}
    assert len(np.intersect1d(r1["H"], r2["H"])) == 0
    off = len(r1["rowsH_ev"]) + 1
    out["ev"] = np.concatenate([np.arange(len(r1["ev"])), off + np.arange(len(r2["ev"]))])
    for k in r1["parts"]:
        out["parts"][k] = {f: np.concatenate([r1["parts"][k][f], r2["parts"][k][f]])
                           for f in r1["parts"][k]}
    for c in r1["reps"]:
        m1 = np.searchsorted(r1["ev"], r1["reps"][c][0])
        m2 = off + np.searchsorted(r2["ev"], r2["reps"][c][0])
        out["reps"][c] = (np.concatenate([m1, m2]).astype(np.int64), r1["reps"][c][1] + r2["reps"][c][1])
    return out


def _local_ev(res):
    """Re-index a holdout result so reps rows are positions in res['ev'] (as pool_results)."""
    r = dict(res)
    r["reps"] = {c: (np.searchsorted(res["ev"], res["reps"][c][0]).astype(np.int64), res["reps"][c][1])
                 for c in res["reps"]}
    r["ev"] = np.arange(len(res["ev"]))
    return r


def cmd_lirep_pool(a):
    """§6.3: H (cached) and H2 holdout stages, pooled H u H2 gate, preference rule."""
    t0 = time.time()
    root = WORK / "models"
    r1 = holdout_eval(load_h(), ["rdspec_fold_shape_s0", "rdspec_fold_notsL_s5"],
                      root / "rdspec_H" / "lirep", a.nproc, a.nsst)
    rep = {"H": {}, "H2": {}, "pooled": {}}
    l1 = _local_ev(r1); l1["B"] = a.boot
    rep["H"]["i"] = _cand_report(l1, "i", "H (i)")
    rep["H"]["iii"] = _cand_report(l1, "iii", "H (iii) POST HOC")
    if a.h_only:
        (root / "rdspec_H" / "lirep" / "lirep_pool_H.json").write_text(json.dumps(rep, indent=1, default=float))
        print("LIREP_POOL_H_ONLY_DONE", f"{time.time() - t0:.0f}s", flush=True)
        return
    r2 = holdout_eval(np.load(h2_path()), ["rdspec_fold2_shape_s0", "rdspec_fold2_notsL_s5"],
                      root / "rdspec_H2" / "lirep", a.nproc, a.nsst)
    l2 = _local_ev(r2); l2["B"] = a.boot
    rep["H2"]["i"] = _cand_report(l2, "i", "H2 (i)")
    rep["H2"]["iii"] = _cand_report(l2, "iii", "H2 (iii)")
    pr = pool_results(r1, r2); pr["B"] = a.boot
    rep["pooled"]["i"] = _cand_report(pr, "i", "POOLED (i)")
    rep["pooled"]["iii"] = _cand_report(pr, "iii", "POOLED (iii)")
    pi, piii = rep["pooled"]["i"], rep["pooled"]["iii"]
    if piii["PASS"] and piii["boot"]["S"]["delta"] >= pi["boot"]["S"]["delta"]:
        choice = "iii"
    elif pi["PASS"]:
        choice = "i"
    else:
        choice = None
    rep["choice"] = choice
    (root / "rdspec_H2" / "lirep" / "lirep_pool.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"LIREP_POOL_DONE choice={choice}  {time.time() - t0:.0f}s", flush=True)


def repair_array(psi_path, shot, t_ms, nproc=6, li_cache=None, cand="i"):
    """Repair (cand 'i' or 'iii') of a whole prediction file (inputs-only shot/time index;
    env_train envelope). Returns (repaired float32 array, rows, info, valid, li)."""
    if li_cache is not None and Path(li_cache).exists():
        z = np.load(li_cache); valid, li = z["valid"], z["li"]
    else:
        valid, li = derived_li(str(psi_path), nproc=nproc)
        if li_cache is not None:
            np.savez(li_cache, valid=valid, li=li)
    psi = np.load(psi_path, mmap_mode="r")
    rows, maps, info = run_repair(cand, psi, shot, t_ms, valid, li)
    out = apply_rows(psi, rows, maps)
    keep = np.setdiff1d(np.arange(len(psi)), rows)
    for i in range(0, len(keep), 20000):          # byte identity of every non-repaired row
        k = keep[i:i + 20000]
        assert out[k].tobytes() == np.asarray(psi[k], np.float32).tobytes()
    return out, rows, info, valid, li


def cmd_lirep_val(a):
    """§6.1 VAL confirmation (once): repair (i) on the stored C2 VAL psi, official S both ways."""
    from multiprocessing import Pool
    from fec.features import load_ds
    t0 = time.time()
    ds = load_ds("diii_d_train")
    bd = WORK / "models" / a.base
    idx = np.load(bd / "val_idx.npy")
    assert (np.array(ds["info"]["splits"])[ds["shot"][idx]] == a.split).all()
    assert a.split in ("val", "train"), "never TEST"   # 'train' only for the code dry run on H rows
    cd = WORK / "models" / a.tag
    cd.mkdir(parents=True, exist_ok=True)
    out, rows, info, valid, li = repair_array(bd / "val_psi.npy", ds["shot"][idx], ds["t_ms"][idx],
                                              a.nproc, cd / "li_env_train.npz", cand=a.cand)
    Bs = np.load(bd / "val_scal.npy")
    np.save(cd / "val_psi.npy", out); np.save(cd / "val_scal.npy", Bs); np.save(cd / "val_idx.npy", idx)
    gmask = np.zeros(len(idx)); gmask[rows] = 1.0
    np.save(cd / "val_g.npy", gmask)
    trig = np.flatnonzero(li_trigger(valid, li))
    sc_old = WORK / "models" / "rdspec_H" / "nanli_val_C2.npz"
    env_cmp = None
    if sc_old.exists() and a.base == "d3d_b_ix_C2_fp32":
        zz = np.load(sc_old)
        s_trig = np.flatnonzero(zz["ok"] & ~np.isfinite(zz["cons"][:, 6]))
        env_cmp = {"scorer_mask": s_trig.tolist(), "env_train": trig.tolist(),
                   "only_scorer": np.setdiff1d(s_trig, trig).tolist(),
                   "only_env_train": np.setdiff1d(trig, s_trig).tolist()}
    print(f"VAL: triggered {len(trig)} (env_train), repaired {len(rows)}; envelope compare {env_cmp}; "
          f"{time.time() - t0:.0f}s", flush=True)
    # ---- official evaluation (labels of VAL; scorer functionals inside the gate only)
    kB, gB, pB, pos = _val_parts(a.base, ds)
    kC, gC, pC, _ = _val_parts(a.tag, ds)
    assert kB == kC
    parts_b, ref_b, refs = _score_parts(kB, gB, pB, a.nproc)
    parts_c, ref_c, _ = _score_parts(kC, gC, pC, a.nproc)
    n = len(kB); full = np.arange(n)
    rb, rc = _finalize(parts_b, ref_b, full), _finalize(parts_c, ref_c, full)
    terms = ["S", "r2_psi", "dlcfs"] + [f"cons_{k}" for k in rb["r2_cons_each"]]

    def tv(r):
        out_ = {"S": r["S"], "r2_psi": r["r2_psi"], "dlcfs": -r["dlcfs"]}   # sign: larger = better
        out_.update({f"cons_{k}": v for k, v in r["r2_cons_each"].items()})
        return out_
    rng = np.random.default_rng(0)
    D = {t: [] for t in terms}
    for _ in range(a.boot):
        s = rng.integers(0, n, n)
        tb, tc = tv(_finalize(parts_b, ref_b, s)), tv(_finalize(parts_c, ref_c, s))
        for t in terms:
            D[t].append(tc[t] - tb[t])
    tb, tc = tv(rb), tv(rc)
    boot = {t: {"delta": tc[t] - tb[t], "mean": float(np.mean(D[t])), "sd": float(np.std(D[t]))}
            for t in terms}
    # per-frame deltas on the repaired frames (official per-frame functionals, refs = labels)
    psi_lab, _ = _labels()
    Bp = np.load(bd / "val_psi.npy", mmap_mode="r")
    shots = ds["shot"][idx]
    per_frame = []
    jobs, meta = [], []
    for r in rows:
        si = int(np.flatnonzero(np.unique(shots) == shots[r])[0])
        j = pos[si]; k = int(np.flatnonzero(j == r)[0])
        contours, cons_gt, cmask, rgeo = refs[si]
        for arr in (Bp, out):
            jobs.append((np.asarray(arr[r:r + 1], np.float64), [contours[k]], [rgeo]))
        meta.append((r, cons_gt[k], cmask[k]))
    with Pool(a.nproc) as pool:
        res = pool.map(_pred_worker, jobs) if jobs else []
    mean_cons = (np.array([sum(rr["cons_sum"][j] for rr in ref_b) for j in range(7)])
                 / np.array([sum(rr["cons_n"][j] for rr in ref_b) for j in range(7)]))
    names = list(rb["r2_cons_each"].keys())
    li_b = li_c = 0.0
    for m_i, (r, cgt, cm) in enumerate(meta):
        db, cb = res[2 * m_i][0][0], res[2 * m_i][1][0]
        dc, cc = res[2 * m_i + 1][0][0], res[2 * m_i + 1][1][0]
        cb = np.where(np.isfinite(cb), cb, mean_cons); cc = np.where(np.isfinite(cc), cc, mean_cons)
        eb = np.where(cm, (cb - cgt) ** 2, np.nan); ec = np.where(cm, (cc - cgt) ** 2, np.nan)
        g_ = np.asarray(psi_lab[idx[r]], np.float64)
        li_b += float(np.nan_to_num(eb[6])); li_c += float(np.nan_to_num(ec[6]))
        per_frame.append({"val_pos": int(r), "shot": int(shots[r]), "li_gt_valid": bool(cm[6]),
                          "li_base_scored": float(res[2 * m_i][1][0][6]),
                          "li_cand_scored": float(res[2 * m_i + 1][1][0][6]),
                          "li_se_base": float(eb[6]), "li_se_cand": float(ec[6]),
                          "li_gt": float(cgt[6]),
                          "d_R_axis": float(res[2 * m_i + 1][1][0][0] - res[2 * m_i][1][0][0]),
                          "d_Z_axis": float(res[2 * m_i + 1][1][0][1] - res[2 * m_i][1][0][1]),
                          "d_psi_sse": float(((g_ - out[r]) ** 2).sum() - ((g_ - Bp[r]) ** 2).sum()),
                          "d_dlcfs": float(dc - db),
                          "d_sqerr": {nm: float(ec[j] - eb[j]) for j, nm in enumerate(names)}})
    ident = bool(np.load(cd / "val_scal.npy").tobytes() == Bs.tobytes())
    worse = [t for t in ("r2_psi", "dlcfs", "cons_kappa", "cons_tri_top", "cons_tri_bot", "cons_volume")
             if boot[t]["delta"] < 0 and abs(boot[t]["delta"]) > 2 * boot[t]["sd"]]
    passed = bool(li_c < li_b and boot["S"]["delta"] >= 0 and not worse and ident)
    outj = {"base": a.base, "cand": a.tag, "n_shots": n, "triggered": int(len(trig)),
            "repaired": int(len(rows)), "info": info, "envelope_compare": env_cmp,
            "base_res": rb, "cand_res": rc, "boot": boot, "li_sse_repaired": {"base": li_b, "cand": li_c},
            "per_frame": per_frame, "worse_beyond_noise": worse, "identity_nonrepaired": True,
            "PASS": passed, "seconds": time.time() - t0}
    (cd / "lirep_val.json").write_text(json.dumps(outj, indent=1, default=float))
    for lab, r in (("base", rb), ("cand", rc)):
        print(f"{lab}: S {r['S']:.6f} R2psi {r['r2_psi']:.6f} qb {r['r2_qb']:.6f} D {r['dlcfs']:.6f} "
              f"cons {r['consistency']:.6f} | " + " ".join(f"{k} {v:.5f}" for k, v in r["r2_cons_each"].items()))
    print("paired deltas (cand - base; dlcfs sign flipped so + = better): " +
          "; ".join(f"{t} {v['delta']:+.6f} (SD {v['sd']:.6f})" for t, v in boot.items()))
    print(f"li SSE on repaired frames {li_b:.3f} -> {li_c:.3f}; worse beyond noise: {worse}; PASS={passed}",
          flush=True)


def cmd_lirep_build(a):
    """Repaired public/private-test psi (inputs only) into a NEW preds dir; never overwrites."""
    from fec.features import load_ds
    import shutil
    src = WORK / "preds" / a.src
    dst = WORK / "preds" / a.out
    if (dst / f"{a.config}_psi.npy").exists():
        raise SystemExit(f"{dst} already has {a.config}_psi.npy (never overwrite)")
    dst.mkdir(parents=True, exist_ok=True)
    ds = load_ds(a.config)
    t0 = time.time()
    lic = dst / f"{a.config}_li_env_train.npz"
    if a.li_cache and not lic.exists():
        import shutil as _sh
        _sh.copyfile(a.li_cache, lic)          # label-free env_train li computed earlier on src
    out, rows, info, valid, li = repair_array(src / f"{a.config}_psi.npy", ds["shot"], ds["t_ms"],
                                              a.nproc, lic, cand=a.cand)
    assert out.shape[0] == ds["info"]["N"] and np.isfinite(out).all()
    np.save(dst / f"{a.config}_psi.npy", out)
    shutil.copyfile(src / f"{a.config}_scal.npy", dst / f"{a.config}_scal.npy")
    g, _, _ = ds_gate(ds, no_ts_zero=False)
    trig = np.flatnonzero(li_trigger(valid, li))
    meta = {"kind": f"li-range repair ({a.cand}), doc D3D_RAMPDOWN_SPECIALIST_20261009 §6.1/§6.3",
            "src": str(src), "config": a.config, "env": str(env_train_path()),
            "lo": LI_LO, "hi": LI_HI, "target": LI_TARGET, "iters": LI_ITERS,
            "triggered": int(len(trig)), "repaired": int(len(rows)),
            "triggered_g1": int((g[trig] >= 1).sum()), "info": info,
            "scal": "copied unchanged from src (re-apply the q95 hybrid if wanted)",
            "seconds": time.time() - t0}
    (dst / f"{a.config}_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"LIREP_BUILD_DONE {dst}: triggered {len(trig)}, repaired {len(rows)}, "
          f"li after {[round(r.get('li_after', np.nan), 3) for r in info]}", flush=True)


# ---------------------------------------------------------------------------- CLI
def get_parser():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gate"); g.add_argument("--config", default="diii_d_public_test")
    ho = sp.add_parser("holdout")
    ho.add_argument("--second", action="store_true", help="build H2 (doc §6.3) instead of H")
    f = sp.add_parser("foldbase")
    f.add_argument("--parent", required=True); f.add_argument("--tag", required=True)
    f.add_argument("--hfile", default=None, help="holdout shot list (default: H)")
    f.add_argument("--vram_gb", type=float, default=4.0)
    t = sp.add_parser("finetune")
    t.add_argument("--parent", required=True); t.add_argument("--tag", required=True)
    t.add_argument("--holdout", action="store_true", help="exclude H shots (stage 1)")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--steps", type=int, default=600)
    t.add_argument("--bs", type=int, default=2048)
    t.add_argument("--focus_frac", type=float, default=0.5)
    t.add_argument("--replay", type=int, default=200000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--vram_gb", type=float, default=4.0)
    e = sp.add_parser("evalho")
    e.add_argument("--parents", required=True, help="comma list of fold-base tags")
    e.add_argument("--specs", required=True, help="name:tag1,tag2;name2:... (same order as parents)")
    e.add_argument("--lams", default="1,0.5", help="blend scale(s) on g")
    e.add_argument("--out", required=True)
    e.add_argument("--nsst", type=int, default=5000)
    e.add_argument("--boot", type=int, default=300)
    e.add_argument("--nproc", type=int, default=6)
    d = sp.add_parser("diag"); d.add_argument("--src", required=True)
    v = sp.add_parser("valbuild")
    v.add_argument("--base", default="d3d_b_ix_C2_fp32")
    v.add_argument("--prefix", required=True, help="specialist tag prefix, e.g. rdspec_ftA1_")
    v.add_argument("--lam", type=float, default=1.0)
    v.add_argument("--tag", required=True)
    w = sp.add_parser("valgate")
    w.add_argument("--base", default="d3d_b_ix_C2_fp32")
    w.add_argument("--cand", required=True)
    w.add_argument("--boot", type=int, default=1000)
    w.add_argument("--nproc", type=int, default=6)
    lh = sp.add_parser("lirep_h")
    lh.add_argument("--nproc", type=int, default=6); lh.add_argument("--nsst", type=int, default=5000)
    lh.add_argument("--boot", type=int, default=1000)
    lv = sp.add_parser("lirep_val")
    lv.add_argument("--base", default="d3d_b_ix_C2_fp32"); lv.add_argument("--tag", default="rdspec_C2_lirep_fp32")
    lv.add_argument("--boot", type=int, default=1000); lv.add_argument("--nproc", type=int, default=6)
    lv.add_argument("--cand", default="i", choices=["i", "iii", "iv"])
    lv.add_argument("--split", default="val", choices=["val", "train"],
                    help="'train' = code dry run on TRAIN-holdout rows only")
    lb = sp.add_parser("lirep_build")
    lb.add_argument("--src", default="d3d_ip_C2_tv"); lb.add_argument("--out", default="d3d_ip_C2_tv_lirep")
    lb.add_argument("--config", default="diii_d_public_test"); lb.add_argument("--nproc", type=int, default=6)
    lb.add_argument("--cand", default="i", choices=["i", "iii"])
    lb.add_argument("--li_cache", default=None, help="precomputed env_train li npz of the src psi")
    l4 = sp.add_parser("lirep_build4")
    l4.add_argument("--src", required=True, help="q95-hybrid preds dir to mirror")
    l4.add_argument("--base", default=None, help="base preds dir whose psi src should equal")
    l4.add_argument("--out", required=True); l4.add_argument("--config", default="diii_d_public_test")
    l4.add_argument("--li_cache", default=None); l4.add_argument("--nproc", type=int, default=6)
    lp = sp.add_parser("lirep_pool")
    lp.add_argument("--nproc", type=int, default=6); lp.add_argument("--nsst", type=int, default=5000)
    lp.add_argument("--boot", type=int, default=1000)
    lp.add_argument("--h_only", action="store_true", help="H stage only (H2 not ready)")
    return ap


def cmd_gate(a):
    from fec.features import load_ds
    ds = load_ds(a.config)
    g, f, post = ds_gate(ds)
    hg = hard_gate(f, post)
    n = len(g)
    print(f"{a.config}: N {n}, shots {len(np.unique(ds['shot']))}, g>0 {np.mean(g > 0):.4%}, "
          f"g=1 {np.mean(g >= 1):.4%}, hard gate {np.mean(hg):.4%}, "
          f"shots with g>0 {len(np.unique(ds['shot'][g > 0]))}, mean g {g.mean():.5f}")
    if "splits" in ds["info"] and a.config == "diii_d_train":
        sp = np.array(ds["info"]["splits"])[ds["shot"]]
        for s in ("train", "val"):
            m = sp == s
            print(f"  {s}: g>0 {np.mean(g[m] > 0):.4%}, g=1 {np.mean(g[m] >= 1):.4%}, "
                  f"hard {np.mean(hg[m]):.4%}")
    return g


def main(argv=None):
    a = get_parser().parse_args(argv)
    return {"gate": cmd_gate, "holdout": cmd_holdout, "foldbase": cmd_foldbase,
            "finetune": cmd_finetune, "evalho": cmd_evalho,
            "diag": cmd_diag, "valbuild": cmd_valbuild, "valgate": cmd_valgate,
            "lirep_h": cmd_lirep_h, "lirep_val": cmd_lirep_val, "lirep_build": cmd_lirep_build,
            "lirep_pool": cmd_lirep_pool, "lirep_build4": cmd_lirep_build4}[a.cmd](a)


if __name__ == "__main__":
    main()
