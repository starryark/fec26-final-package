"""Frame-level model inputs built from consolidated arrays (inputs only)."""
from __future__ import annotations

import json

import numpy as np

from fec.paths import WORK


def load_ds(config):
    d = WORK / "ds" / config
    info = json.loads((d / "info.json").read_text())
    out = {"info": info,
           "cur": np.load(d / "cur.npy", mmap_mode="r"),
           "ts": np.load(d / "ts.npy", mmap_mode="r"),
           "t_ms": np.load(d / "t_ms.npy"),
           "shot": np.load(d / "shot.npy"),
           "fidx": np.load(d / "fidx.npy")}
    if (d / "psi.npy").exists():
        out["psi"] = np.load(d / "psi.npy", mmap_mode="r")
        out["scal"] = np.load(d / "scal.npy")
    return out


def d3d_inputs(ds, stats=None, fit_mask=None):
    """X (N,F) float32 and the sign s=sign(Ip) used to canonicalize currents.

    Currents are multiplied by s so that Ip > 0 on every frame; EFIT DIII-D psi keeps
    the axis at a minimum regardless, so psi_stored = F(s*currents) is one function.
    """
    ch = ds["info"]["channels"]
    ip = ch.index("magnetics_plasma_current")
    cur = np.asarray(ds["cur"], dtype=np.float32)          # (N,C,K)
    s = np.sign(cur[:, ip, 6]).astype(np.float32)
    s[s == 0] = 1.0
    c = cur * s[:, None, None]
    if stats is None:
        # fit current-normalisation constants on the rows the caller passes (fit_mask; every row if None): the
        # run's training rows, i.e. every labelled diii_d_train row under --alldata (train_d3d_b / train_d3d_bx)
        c0 = c[:, :, 6] if fit_mask is None else c[fit_mask][:, :, 6]
        stats = {"cmu": np.nanmean(c0, 0).tolist(), "csd": (np.nanstd(c0, 0) + 1e-3).tolist()}
    cmu = np.array(stats["cmu"], np.float32)[None, :, None]
    csd = np.array(stats["csd"], np.float32)[None, :, None]
    cn = (c - cmu) / csd
    K = cn.shape[2]
    # local differences emphasise dynamics (eddy currents): value(t+o) - value(t)
    diff = cn - cn[:, :, 6:7]
    feats = [cn[:, :, 6], diff[:, :, [k for k in range(K) if k != 6]].reshape(len(cn), -1) * 5.0]
    ts = np.asarray(ds["ts"], dtype=np.float32)
    m = np.isfinite(ts)
    tsv = np.where(m, ts, 0.0)
    tsv = np.sign(tsv) * np.log1p(np.abs(tsv))
    feats += [tsv, m.astype(np.float32)]
    bc = ch.index("magnetics_bcoil") if "magnetics_bcoil" in ch else ch.index("magnetics_tf_current")
    feats += [s[:, None], np.sign(c[:, bc, 6])[:, None], (ds["t_ms"] / 1000.0).astype(np.float32)[:, None]]
    X = np.concatenate(feats, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, s, stats
