"""Apply d3d_b model(s) (fec/train_d3d_b.py) to a config, inputs only -> psi (N,65,65), scal (N,5).

Stable CLI (all outputs WORK/preds/<out>/<config>_psi.npy float32 (N,65,65), <config>_scal.npy
float32 (N,5: q95, betaN, li, R_axis, Z_axis), <config>_meta.json, in WORK/ds/<config> order):

  run      --config C --out NAME --tags t1,t2,... [--weights w1,w2,...] [--fallback TAG]
  recipe   --config C --out NAME --recipe RECIPE.json
           JSON: {"members": [{"kind": "b"|"seq", "tag": T, "weight": w}, ...],
                  "fallback": null | {"kind": "b", "tag": T} | [{"kind","tag","weight"}, ...]}
           (a fallback list = weighted mean of those models, used on no-Thomson frames)
           ("seq" members are fec/train_d3d_seq.py models, run via fec/predict_seq_fp32_b.py)
  combine  --config C --out NAME --preds p1,p2,... [--weights ...] [--fallback-pred P]
           (weighted average of existing WORK/preds/<p>/ files)
  val      --tag TAG        (fp32 re-prediction of the model's VAL frames -> models/<TAG>_fp32)
Legacy forms kept: `<config> <out> <tags> [fallback]`, `--val <tag>`,
`--combine <config> <out> <preds> [fallback_pred]`.

Weights are normalised to sum 1 (default equal).  The optional fallback (a feat="nots" model)
replaces the ensemble on frames without any finite Thomson value (inputs-only criterion,
features_b.no_thomson_frames).  Inference is full fp32 (no autocast; bf16 autocast inference
raised VAL psi SSE by ~6%) on the device in env FEC_B_DEV (default "cpu").  Works for any
config with WORK/ds/<config> arrays (diii_d_public_test, diii_d_private_test, diii_d_train);
feat="ext"/"hist" models additionally need WORK/ds_b/<config>/xb.npy (fec/extract_b.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.features import load_ds  # noqa: E402
from fec.features_b import d3d_inputs_b, no_thomson_frames  # noqa: E402
from fec.paths import WORK  # noqa: E402
from fec.train_d3d_b import build_net  # noqa: E402

DEV = os.environ.get("FEC_B_DEV", "cpu")
SCAL_COLS = ["q95", "betaN", "li", "R_axis", "Z_axis"]


def predict_tag(ds, config, tag, cache, rows=None):
    """fp32 predictions of one model on ds frames (all, or `rows`): psi (n,4225), scal (n,5)."""
    ck = torch.load(WORK / "models" / tag / "model.pt", weights_only=False, map_location="cpu")
    a = ck["args"]
    feat = ck.get("feat", a.get("feat", "base"))
    key = (feat, json.dumps(ck["stats"]))
    if key not in cache:
        cache[key] = d3d_inputs_b(ds, config, ck["stats"], feat=feat)[0]
    X = cache[key] if rows is None else cache[key][rows]
    Xn = torch.as_tensor((X - ck["xmu"]) / ck["xsd"])
    net = build_net(a, ck["fin"]).to(DEV)
    net.load_state_dict(ck["net"]); net.eval()
    mean, V = ck["mean"].to(DEV), ck["V"].to(DEV)
    smu, ssd = ck["smu"].to(DEV), ck["ssd"].to(DEV)
    P, Q = [], []
    with torch.no_grad():
        for i in range(0, len(Xn), 4096):
            p, q = net(Xn[i:i + 4096].to(DEV))
            P.append(((p * ck["ysd"]) @ V.T + mean).cpu())
            Q.append((q * ssd + smu).cpu())
    return torch.cat(P).numpy(), torch.cat(Q).numpy()


def _member(ds, config, kind, tag, cache, rows=None):
    if kind == "b":
        return predict_tag(ds, config, tag, cache, rows=rows)
    if kind == "seq":
        from fec.predict_seq_fp32_b import predict_seq_fp32
        return predict_seq_fp32(ds, tag, rows)
    raise ValueError(kind)


def _weights(w, n):
    w = np.ones(n) if not w else np.array([float(x) for x in w], np.float64)
    assert len(w) == n and (w >= 0).all() and w.sum() > 0
    return w / w.sum()


def _write(config, out_name, ds, psi, sc, meta):
    psi = psi.reshape(-1, 65, 65).astype(np.float32)
    sc = sc.astype(np.float32)
    assert psi.shape[0] == ds["info"]["N"] and np.isfinite(psi).all() and np.isfinite(sc).all()
    out = WORK / "preds" / out_name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{config}_psi.npy", psi)
    np.save(out / f"{config}_scal.npy", sc)
    meta = dict(meta, config=config, N=int(psi.shape[0]), scal_cols=SCAL_COLS,
                inference=f"{DEV} fp32")
    (out / f"{config}_meta.json").write_text(json.dumps(meta, indent=1))
    print(out, psi.shape, sc.shape, "fallback frames", meta.get("fallback_frames", 0), flush=True)
    return psi, sc


def predict_members(config, out_name, members, fallback=None):
    """members: list of (kind, tag, weight); fallback: None, (kind, tag) or a list of
    (kind, tag, weight) whose weighted mean replaces the ensemble on no-Thomson frames."""
    ds = load_ds(config)
    cache = {}
    w = _weights([m[2] for m in members], len(members))
    psi = None; sc = None
    for (kind, tag, _), wi in zip(members, w):
        P, Q = _member(ds, config, kind, tag, cache)
        psi = wi * P.astype(np.float64) if psi is None else psi + wi * P
        sc = wi * Q.astype(np.float64) if sc is None else sc + wi * Q
    nfb = 0
    fbl = None
    if fallback:
        fbl = [tuple(fallback) + (1.0,)] if isinstance(fallback[0], str) else list(fallback)
        sel = np.flatnonzero(no_thomson_frames(ds))
        if len(sel):
            fw = _weights([f[2] for f in fbl], len(fbl))
            fp = 0.0; fq = 0.0
            for (kind, tag, _), wi in zip(fbl, fw):
                P, Q = _member(ds, config, kind, tag, cache, rows=sel)
                fp = fp + wi * P.astype(np.float64); fq = fq + wi * Q.astype(np.float64)
            psi[sel] = fp; sc[sel] = fq
        nfb = int(len(sel))
    return _write(config, out_name, ds, psi, sc, {
        "members": [{"kind": k, "tag": t, "weight": float(x)} for (k, t, _), x in zip(members, w)],
        "fallback": None if not fbl else [{"kind": k, "tag": t, "weight": float(x)}
                                          for k, t, x in fbl],
        "fallback_frames": nfb})


def predict(config, tags, out_name, fallback=None, weights=None):
    w = weights or [1.0] * len(tags)
    return predict_members(config, out_name, [("b", t, x) for t, x in zip(tags, w)],
                           ("b", fallback) if fallback else None)


def predict_recipe(config, out_name, recipe_path):
    r = json.loads(Path(recipe_path).read_text())
    members = [(m.get("kind", "b"), m["tag"], float(m.get("weight", 1.0))) for m in r["members"]]
    fb = r.get("fallback")
    if isinstance(fb, dict):
        fb = [fb]
    fbl = [(f.get("kind", "b"), f["tag"], float(f.get("weight", 1.0))) for f in fb] if fb else None
    return predict_members(config, out_name, members, fbl)


def predict_val(tag):
    """Re-predict a model's saved VAL frames in fp32 -> WORK/models/<tag>_fp32 (for eval_val)."""
    m = WORK / "models" / tag
    ds = load_ds("diii_d_train")
    if (m / "val_idx.npy").exists():
        idx = np.load(m / "val_idx.npy")
    else:                                   # all frames of 'val' shots (trainer convention)
        idx = np.flatnonzero(np.array(ds["info"]["splits"])[ds["shot"]] == "val")
    P, Q = predict_tag(ds, "diii_d_train", tag, {}, rows=idx)
    o = WORK / "models" / (tag + "_fp32")
    o.mkdir(parents=True, exist_ok=True)
    np.save(o / "val_psi.npy", P.reshape(-1, 65, 65).astype(np.float32))
    np.save(o / "val_scal.npy", Q.astype(np.float32))
    np.save(o / "val_idx.npy", idx)
    (o / "members.json").write_text(json.dumps({"source": tag, "inference": f"{DEV} fp32"}))
    print(o, P.shape, flush=True)


def combine(config, out_name, preds, fallback=None, weights=None):
    ds = load_ds(config)
    w = _weights(weights, len(preds))
    psi = None; sc = None
    for d, wi in zip(preds, w):
        P = np.load(WORK / "preds" / d / f"{config}_psi.npy").astype(np.float64)
        Q = np.load(WORK / "preds" / d / f"{config}_scal.npy").astype(np.float64)
        psi = wi * P if psi is None else psi + wi * P
        sc = wi * Q if sc is None else sc + wi * Q
    nfb = 0
    if fallback:
        sel = no_thomson_frames(ds)
        psi[sel] = np.load(WORK / "preds" / fallback / f"{config}_psi.npy", mmap_mode="r")[sel]
        sc[sel] = np.load(WORK / "preds" / fallback / f"{config}_scal.npy")[sel]
        nfb = int(sel.sum())
    return _write(config, out_name, ds, psi, sc, {
        "combined_preds": preds, "weights": [float(x) for x in w], "fallback_pred": fallback,
        "fallback_frames": nfb})


def _csv(s):
    return [x for x in s.split(",") if x] if s else None


def main(argv):
    if argv and argv[0] == "--val":                       # legacy
        return predict_val(argv[1])
    if argv and argv[0] == "--combine":                   # legacy
        return combine(argv[1], argv[2], _csv(argv[3]), argv[4] if len(argv) > 4 else None)
    if argv and argv[0] not in ("run", "recipe", "combine", "val", "-h", "--help"):   # legacy
        return predict(argv[0], _csv(argv[2]), argv[1], argv[3] if len(argv) > 3 else None)
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--config", required=True)
    r.add_argument("--out", required=True); r.add_argument("--tags", required=True)
    r.add_argument("--weights"); r.add_argument("--fallback")
    c = sp.add_parser("recipe"); c.add_argument("--config", required=True)
    c.add_argument("--out", required=True); c.add_argument("--recipe", required=True)
    m = sp.add_parser("combine"); m.add_argument("--config", required=True)
    m.add_argument("--out", required=True); m.add_argument("--preds", required=True)
    m.add_argument("--weights"); m.add_argument("--fallback-pred", dest="fallback_pred")
    v = sp.add_parser("val"); v.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return predict(a.config, _csv(a.tags), a.out, a.fallback, _csv(a.weights))
    if a.cmd == "recipe":
        return predict_recipe(a.config, a.out, a.recipe)
    if a.cmd == "combine":
        return combine(a.config, a.out, _csv(a.preds), a.fallback_pred, _csv(a.weights))
    return predict_val(a.tag)


if __name__ == "__main__":
    main(sys.argv[1:])
