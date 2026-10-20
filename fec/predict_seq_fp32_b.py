"""fp32 (no autocast, fp32 inputs) CPU inference for fec/train_d3d_seq.py models, inputs only.

The seq trainer's own predict path casts inputs to float16 and runs under bf16 autocast; this
wrapper reuses its model/feature code unchanged (imports fec.train_d3d_seq, sets its DEV to
"cpu" so the bf16 CUDA autocast contexts are inert) and feeds float32 inputs.

    python -B fec/predict_seq_fp32_b.py --val <tag>                      # -> models/<tag>_fp32
    python -B fec/predict_seq_fp32_b.py --config <config> --tag <tag> [--out <name>]
                                                                         # -> preds/<name>/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fec.train_d3d_seq as SEQ  # noqa: E402
from fec.features import load_ds  # noqa: E402
from fec.paths import WORK  # noqa: E402

SEQ.DEV = "cpu"


@torch.no_grad()
def predict_seq_fp32(ds, tag, idx=None):
    """psi (n,4225) f32 and scal (n,5) for frames idx (default all) of ds; whole-shot context
    windows are taken from all frames of ds (inputs only)."""
    ck = torch.load(WORK / "models" / tag / "model.pt", weights_only=False, map_location="cpu")
    a = ck["args"]
    net = SEQ.build_net(ck["fin"], ck["K"], a)
    net.load_state_dict(ck["net"]); net.eval()
    X, _, _ = SEQ.seq_inputs(ds, ck["stats"])
    Xn = torch.as_tensor(((X - ck["xmu"]) / ck["xsd"]).astype(np.float32))
    del X
    T = torch.as_tensor(ds["t_ms"] / 1000.0, dtype=torch.float32)
    idx = np.arange(len(Xn)) if idx is None else np.asarray(idx)
    P, Q = SEQ.Predictor(net, a, Xn, T, ds["shot"])(idx)
    psi = ((P.float() * ck["ysd"]) @ ck["V"].float().T + ck["mean"].float()).numpy().astype(np.float32)
    sc = (Q.float() * ck["ssd"].float() + ck["smu"].float()).numpy().astype(np.float32)
    return psi, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=None, help="tag: re-predict its saved VAL frames")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.val:
        m = WORK / "models" / a.val
        idx = np.load(m / "val_idx.npy")
        psi, sc = predict_seq_fp32(load_ds("diii_d_train"), a.val, idx)
        o = WORK / "models" / (a.val + "_fp32")
        o.mkdir(parents=True, exist_ok=True)
        np.save(o / "val_psi.npy", psi.reshape(-1, 65, 65)); np.save(o / "val_scal.npy", sc)
        np.save(o / "val_idx.npy", idx)
        (o / "members.json").write_text(json.dumps({"source": a.val, "inference": "cpu fp32 seq"}))
        print(o, psi.shape, flush=True)
        return
    ds = load_ds(a.config)
    psi, sc = predict_seq_fp32(ds, a.tag)
    assert np.isfinite(psi).all() and np.isfinite(sc).all()
    o = WORK / "preds" / (a.out or a.tag)
    o.mkdir(parents=True, exist_ok=True)
    np.save(o / f"{a.config}_psi.npy", psi.reshape(-1, 65, 65)); np.save(o / f"{a.config}_scal.npy", sc)
    (o / f"{a.config}_meta.json").write_text(json.dumps(
        {"config": a.config, "tags": [a.tag], "inference": "cpu fp32 seq", "N": int(len(psi)),
         "scal_cols": ["q95", "betaN", "li", "R_axis", "Z_axis"]}, indent=1))
    print(o, psi.shape, flush=True)


if __name__ == "__main__":
    main()
