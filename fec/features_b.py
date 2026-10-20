"""Model inputs for the d3d_b family: base d3d_inputs + optional history/context features.

feat="base": exactly fec.features.d3d_inputs.
feat="ext":  base + extra columns from WORK/ds_b/<config>/xb.npy (fec/extract_b.py), all
             computed from input columns only (coil/plasma currents, Thomson, efit_times).
feat="nots": base minus all Thomson columns (fallback model for frames without Thomson).
feat="hist": base + only the Ip/ECOILA history columns of "ext" (long offsets, EWMAs) and
             their missing flag; no shot-level constants (t_end, Ipmax) and no Thomson EWMA.
feat="base_gap" / "nots_gap" (opt-in, li2ft arm D, agent li2ft 2026-10-12): base / nots plus two columns
             from the EFIT time stamps only (WORK/ds/<config>/t_ms.npy + shot.npy, efit_times scaffolding):
             the time since the first frame after the latest within-shot gap > 100 ms between consecutive
             frames (capped at 2 s; 2 s if no gap so far) and that gap's length (capped at 2 s; 0 if none),
             both in seconds (gap_columns). The trainer standardises them with TRAIN statistics.
Ip/ECOILA-derived columns are canonicalised by the same s = sign(Ip(t)) as the base currents
and expressed relative to their value at t.  Missing values -> 0 plus a missing flag.
"""
from __future__ import annotations

import numpy as np

from fec.features import d3d_inputs
from fec.paths import WORK


def load_xb(config):
    return np.load(WORK / "ds_b" / config / "xb.npy", mmap_mode="r")


def ext_columns(ds, xb, s):
    ch = ds["info"]["channels"]
    ip = ch.index("magnetics_plasma_current")
    ec = ch.index("magnetics_ECOILA")
    cur = np.asarray(ds["cur"][:, :, 6], dtype=np.float32)
    ipnow = (cur[:, ip] * s)[:, None]
    ecnow = (cur[:, ec] * s)[:, None]
    x = np.asarray(xb, dtype=np.float32)
    sc = s[:, None]
    f = [(x[:, 0:8] * sc - ipnow) / 1000.0,      # Ip history/future rel. to now [MA]
         (x[:, 8:16] * sc - ecnow) / 10.0,       # ECOILA history/future rel. to now
         (x[:, 16:21] * sc - ipnow) / 1000.0,    # Ip EWMA - Ip(t)
         (x[:, 21:26] * sc - ecnow) / 10.0,      # ECOILA EWMA - ECOILA(t)
         x[:, 16:21] * sc / 1000.0,              # Ip EWMA absolute [MA]
         x[:, 26:29]]                            # t_end - t, |Ip|/Ipmax, Ipmax
    th = x[:, 29:37]
    f.append(np.sign(th) * np.log1p(np.abs(th)))
    X = np.concatenate(f, axis=1)
    miss = ~np.isfinite(X)
    # columns: 0:8 Ip offs, 8:16 EC offs, 16:21/21:26 EWMA rel, 26:31 Ip EWMA abs,
    # 31:34 timing, 34:40 Thomson EWMA, 40:42 Thomson now
    flags = np.stack([miss[:, 0:31].any(1), miss[:, 31:34].any(1), miss[:, 34:40].any(1),
                      miss[:, 40:42].any(1)], 1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([X, flags], axis=1).astype(np.float32)


def d3d_inputs_b(ds, config, stats=None, feat="base", fit_mask=None):
    """fit_mask: boolean (N,) TRAIN-frame mask used to fit the current-normalisation stats
    when stats is None (passed through to fec.features.d3d_inputs)."""
    X, s, stats = d3d_inputs(ds, stats, fit_mask=fit_mask)
    if feat == "base":
        return X, s, stats
    if feat == "ext":
        E = ext_columns(ds, load_xb(config), s)
        return np.concatenate([X, E], axis=1), s, stats
    if feat == "nots":
        keep = np.r_[0:231, 359:X.shape[1]]
        return X[:, keep], s, stats
    if feat == "hist":
        E = ext_columns(ds, load_xb(config), s)
        return np.concatenate([X, E[:, 0:31], E[:, 42:43]], axis=1), s, stats
    if feat in ("base_gap", "nots_gap"):                  # opt-in (li2ft arm D): base / nots + gap_columns
        Xb, s, stats = d3d_inputs_b(ds, config, stats, feat=feat[:-4], fit_mask=fit_mask)
        return np.concatenate([Xb, gap_columns(ds)], axis=1), s, stats
    raise ValueError(feat)


GAP_MS = 100.0          # a gap: > 100 ms between consecutive EFIT times of a shot
GAP_CAP_MS = 2000.0     # both columns capped at 2 s


def gap_clock(t_ms, shot, min_gap=GAP_MS):
    """Per frame (any order): time since the first frame after the latest within-shot gap of > min_gap ms
    between consecutive EFIT times (inf if none so far) and that gap's length (0 if none), in ms. Equals
    fec.xli.gap_clock (vectorised; frames are ordered by (shot, t) internally)."""
    t = np.asarray(t_ms, np.float64); sh = np.asarray(shot)
    o = np.lexsort((t, sh))
    ts, ss = t[o], sh[o]
    n = len(ts)
    start = np.zeros(n, bool)
    if n > 1:
        start[1:] = (ss[1:] == ss[:-1]) & (np.diff(ts) > min_gap)
    last = np.maximum.accumulate(np.where(start, np.arange(n), -1)) if n else np.zeros(0, np.int64)
    ok = last >= 0
    ok[ok] = ss[last[ok]] == ss[ok]
    tsg = np.full(n, np.inf); glen = np.zeros(n)
    lo = last[ok]
    tsg[ok] = ts[ok] - ts[lo]
    glen[ok] = ts[lo] - ts[lo - 1]
    out_t = np.empty(n); out_g = np.empty(n)
    out_t[o] = tsg; out_g[o] = glen
    return out_t, out_g


def gap_columns(ds):
    """(N, 2) float32: [min(time since the latest > 100 ms gap, 2 s), min(gap length, 2 s)] in seconds."""
    tsg, glen = gap_clock(ds["t_ms"], ds["shot"])
    return np.stack([np.minimum(tsg, GAP_CAP_MS) / 1000.0, np.minimum(glen, GAP_CAP_MS) / 1000.0],
                    1).astype(np.float32)


def after_gap(ds, window_ms=600.0):
    """Frames within window_ms after a > 100 ms gap (li2ft arm D oversampling set; label-free)."""
    return gap_clock(ds["t_ms"], ds["shot"])[0] <= window_ms


def thomson_groups(feat="base", nbase=362):
    """Column indices of Thomson-derived inputs and their raw values when data are missing.

    Returns {"core": (cols, vals), "edge": (cols, vals)} for the d3d_inputs layout
    [cur(21), diff(210), tsv(64: core Te 20, core ne 20, edge Te 12, edge ne 12), mask(64),
    s, sign(bcoil), t] (+ ext columns, whose central-Thomson part derives from the core chord).
    """
    if feat not in ("base", "ext"):
        raise ValueError(f"tsdrop not defined for feat={feat}")
    t0, m0 = 231, 295
    core = list(range(t0, t0 + 40)) + list(range(m0, m0 + 40))
    edge = list(range(t0 + 40, t0 + 64)) + list(range(m0 + 40, m0 + 64))
    cv = [0.0] * len(core)
    ev = [0.0] * len(edge)
    if feat == "ext":
        e0 = nbase
        cols = list(range(e0 + 34, e0 + 42))          # Thomson EWMA + now (value 0 if missing)
        core += cols + [e0 + 44, e0 + 45]              # their missing flags -> 1
        cv += [0.0] * len(cols) + [1.0, 1.0]
    return {"core": (core, cv), "edge": (edge, ev)}


def no_thomson_frames(ds):
    """Frames with no finite Thomson value at all (where a 'nots' fallback model applies)."""
    return ~np.isfinite(np.asarray(ds["ts"], dtype=np.float32)).any(1)
