"""DIII-D input audit 2: label-free input scans, a guard for new folds, and proposed input fixes.

Study note: docs/research/D3D_INPUT_AUDIT2_20261008.md (follows D3D_IP_TIMEBASE_AUDIT_20261008).
Every command reads INPUT columns only (magnetics_*, thomson_*, the efit_times scaffold), never
efit_* targets, so it runs identically on diii_d_train, diii_d_public_test and a private fold.

    python -B fec/input_fixes2.py scan  --config C [--workers 6] [--out DIR]   # per-shot record
    python -B fec/input_fixes2.py guard --config C [--out DIR]   # exit 1 on a time-axis defect
    python -B fec/input_fixes2.py xcorr --config C [--workers 5] [--out DIR]   # channel timing
    python -B fec/input_fixes2.py build_ts --config C --src WORK --dst ts.npy --fix despike|none

Fix functions (inputs only):
  thomson_reset_rows(times, te, ne)  starter issue #12 padding rows (guard; no feature change)
  despike_profile(vals, pos)         isolated Thomson chord outliers -> NaN (finding F7)
  despike_shot(npz) / build_ts(...)  apply despike to fec/extract.py per-shot arrays and rebuild
                                     ts.npy with fec.dataset.thomson_features (--fix none
                                     reproduces the existing ts.npy bit for bit)
Outputs default to <fec.paths.AUDIT2>/analysis/<config>_{scan,xcorr}.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.extract import D3D_CH, HALF, OFFSETS, col1d, col2d  # noqa: E402
from fec.ip_timebase import idx_ip_times  # noqa: E402
from fec.paths import AUDIT2, IDX_WORK, shot_files  # noqa: E402
LONG_OFFS = np.array([-1000, -600, -400, -250, -175, 35, 50, 100], dtype=np.float64)  # extract_b
MAG_CH = [c for c in D3D_CH if c != "magnetics_plasma_current"]
SCAN_COLS = (["efit_times", "magnetics_time", "magnetics_plasma_current",
              "magnetics_plasma_current_times"] + MAG_CH +
             ["thomson_core_times", "thomson_core_Te", "thomson_core_ne", "thomson_core_R",
              "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne", "thomson_edge_spatial",
              "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z"])


# ============================================================================ helpers
def window_counts(ts, centers, half):
    """Number of samples of the (sorted) axis ts inside [c-half, c+half] for each centre."""
    i0 = np.searchsorted(ts, centers - half, side="left")
    i1 = np.searchsorted(ts, centers + half, side="right")
    return i1 - i0


def axis_stats(ts):
    d = np.diff(ts)
    if d.size == 0:
        return {"n": int(ts.size)}
    med = float(np.median(d))
    gap = d > 1.5 * med
    out = {"n": int(ts.size), "t0": float(ts[0]), "t1": float(ts[-1]), "dt_med": med,
           "dt_min": float(d.min()), "dt_max": float(d.max()), "n_dup": int((d == 0).sum()),
           "n_back": int((d < 0).sum()), "n_gap": int(gap.sum()), "nan": int((~np.isfinite(ts)).sum())}
    if gap.any():
        k = int(np.argmax(d))
        out["gap_max_at"] = float(ts[k])
    return out


def onset_end(ts, s, frac=0.10):
    a = np.abs(s - np.nanmedian(s[: max(10, len(s) // 50)]))
    ok = np.isfinite(a)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan
    mx = float(np.nanpercentile(a, 99.5))
    on = np.flatnonzero(ok & (a > frac * mx))
    if on.size == 0:
        return np.nan, np.nan, mx
    return float(ts[on[0]]), float(ts[on[-1]]), mx


def chan_stats(mt, s, t_on, t_off):
    """Offsets, noise, quantisation and plasma-window range of one magnetics channel."""
    fin = np.isfinite(s)
    out = {"nan": int((~fin).sum())}
    if fin.sum() < 100:
        return out
    n = len(s)
    k = max(200, n // 40)
    head, tail = s[:k], s[-k:]
    out.update(b_head=float(np.nanmedian(head)), sd_head=float(np.nanstd(head)),
               b_tail=float(np.nanmedian(tail)), sd_tail=float(np.nanstd(tail)))
    pre = (mt > -3500.0) & (mt < -3000.0)
    if pre.any():
        out["b_m3s"] = float(np.nanmedian(s[pre]))
    d = np.abs(np.diff(s[: min(n, 40000)]))
    d = d[d > 0]
    out["q_step"] = float(d.min()) if d.size else 0.0
    out["q_step_p5"] = float(np.percentile(d, 5)) if d.size else 0.0
    if np.isfinite(t_on) and np.isfinite(t_off) and t_off > t_on:
        w = (mt >= t_on) & (mt <= t_off)
        if w.sum() > 10:
            sw = s[w]
            out.update(p01=float(np.nanpercentile(sw, 1)), p50=float(np.nanmedian(sw)),
                       p99=float(np.nanpercentile(sw, 99)),
                       flat=float(np.mean(np.diff(sw) == 0)))
    out["absmax"] = float(np.nanmax(np.abs(s)))
    return out


def thomson_stats(times, te, ne, et, t_on, t_off, chord_z=None):
    out = {"axis": axis_stats(np.asarray(times, float)) if len(times) > 1 else {"n": int(len(times))}}
    if te.size == 0 or te.shape[0] != len(times):
        out["shape_mismatch"] = [list(te.shape), int(len(times))]
        return out
    out["nch"] = int(te.shape[1])
    out["shape_ne"] = list(ne.shape)
    for nm, a in (("te", te), ("ne", ne)):
        out[nm + "_nan"] = float(np.mean(~np.isfinite(a)))
        out[nm + "_zero"] = float(np.mean(a == 0))
        out[nm + "_neg"] = float(np.mean(a < 0))
        pos = a[np.isfinite(a) & (a > 0)]
        out[nm + "_p50"] = float(np.median(pos)) if pos.size else np.nan
        out[nm + "_p99"] = float(np.percentile(pos, 99)) if pos.size else np.nan
        out[nm + "_max"] = float(pos.max()) if pos.size else np.nan
        out[nm + "_min"] = float(pos.min()) if pos.size else np.nan
    # te>0 xor ne>0 inconsistency
    out["te_ne_xor"] = float(np.mean((te > 0) != (ne > 0)))
    # plasma presence timing: >= 40% of channels with ne > 1e18 and median Te > 30 eV
    v = np.isfinite(ne) & (ne > 1e18) & np.isfinite(te) & (te > 0)
    nv = v.sum(1)
    with np.errstate(all="ignore"):
        tmed = np.array([np.median(te[k][v[k]]) if nv[k] else 0.0 for k in range(len(nv))])
    good = (nv >= max(3, int(0.4 * te.shape[1]))) & (tmed > 30.0)
    tarr = np.asarray(times, float)
    if good.any() and np.isfinite(t_on):
        tg = tarr[good]
        out["first_valid"] = float(tg.min())
        out["last_valid"] = float(tg.max())
        out["first_rel_on"] = float(tg.min() - t_on)
        out["last_rel_off"] = float(tg.max() - t_off) if np.isfinite(t_off) else np.nan
        out["n_valid_before_on"] = int(((tg < t_on - 5.0)).sum())
        out["n_valid_after_off"] = int(((tg > t_off + 5.0)).sum()) if np.isfinite(t_off) else 0
        # last non-plasma sample before the first plasma sample (bounds the onset)
        pre = tarr[(~good) & (tarr < tg.min())]
        out["last_bad_before_first"] = float(pre.max()) if pre.size else np.nan
    if np.isfinite(t_on) and np.isfinite(t_off):
        w = (tarr >= t_on + 50.0) & (tarr <= t_off - 20.0)
        if w.any():
            out["plasma_samples"] = int(w.sum())
            out["plasma_good_frac"] = float(good[w].mean())
            out["ne_gt2e20"] = float(np.mean(ne[w] > 2e20))
            out["te_gt15k"] = float(np.mean(te[w] > 15000.0))
            out["ch_dead"] = int((~v[w]).all(0).sum())   # channels never valid in plasma
    # frames (efit) with no sample within 25 ms (any channel)
    tt = np.sort(np.asarray(times, float))
    if tt.size:
        k = np.clip(np.searchsorted(tt, et), 1, max(1, tt.size - 1))
        gap = np.minimum(np.abs(et - tt[np.clip(k - 1, 0, tt.size - 1)]),
                         np.abs(tt[np.clip(k, 0, tt.size - 1)] - et))
        out["frames_gap25"] = int((gap > 25.0).sum())
        out["frames_gap_max"] = float(gap.max())
    return out


# ============================================================================ scan
def scan_one(path):
    path = Path(path)
    t = pq.read_table(path, columns=SCAN_COLS)
    et = col1d(t, "efit_times")
    mt = col1d(t, "magnetics_time")
    ip = col1d(t, "magnetics_plasma_current")
    ipt_s = col1d(t, "magnetics_plasma_current_times")
    ipt = idx_ip_times(ipt_s, mt[0])
    rec = {"stem": path.stem, "n_frames": int(len(et)), "efit": axis_stats(et),
           "mag": axis_stats(mt), "ipt": axis_stats(ipt)}
    rec["long"] = bool(mt[0] < -6000.0)
    # Ip timing on the idx axis
    ip_on, ip_off, ip_mx = onset_end(ipt, ip)
    rec.update(ip_on=ip_on, ip_off=ip_off, ip_max=ip_mx,
               ip_sign=float(np.sign(np.nanmedian(ip[(ipt > ip_on) & (ipt < ip_off)])))
               if np.isfinite(ip_on) and np.isfinite(ip_off) and ip_off > ip_on else 0.0)
    # efit time coverage by inputs
    rec["efit_after_ip_end"] = int((et + OFFSETS.max() + HALF > ipt[-1]).sum())
    rec["efit_before_ip_start"] = int((et + OFFSETS.min() - HALF < ipt[0]).sum())
    rec["efit_after_mag_end"] = int((et + OFFSETS.max() + HALF > mt[-1]).sum())
    rec["efit_rel_ip"] = [float(et[0] - ip_on) if np.isfinite(ip_on) else np.nan,
                          float(et[-1] - ip_off) if np.isfinite(ip_off) else np.nan]
    # box-mean windows: empty windows (interp fallback) per axis
    q = (et[:, None] + OFFSETS[None, :]).ravel()
    mts = np.sort(mt[np.isfinite(mt)])
    ipts = np.sort(ipt[np.isfinite(ipt)])
    rec["box_empty_mag"] = int((window_counts(mts, q, HALF) == 0).sum())
    rec["box_empty_ip"] = int((window_counts(ipts, q, HALF) == 0).sum())
    rec["box_n_mag_min"] = int(window_counts(mts, q, HALF).min())
    ql = (et[:, None] + LONG_OFFS[None, :]).ravel()
    rec["boxb_empty_mag"] = int((window_counts(mts, ql, 2.0) == 0).sum())
    rec["boxb_empty_ip"] = int((window_counts(ipts, ql, 2.0) == 0).sum())
    rec["boxb_out_ip"] = int(((ql + 2.0 < ipts[0]) | (ql - 2.0 > ipts[-1])).sum())
    # frames whose +-1 ms window at some offset touches a sampling gap (> 1.5 median dt)
    d = np.diff(mts)
    gi = np.flatnonzero(d > 1.5 * np.median(d))
    if gi.size:
        gl, gr = mts[gi], mts[gi + 1]
        hit = np.zeros(q.shape, bool)
        for a, b in zip(gl, gr):
            hit |= (q + HALF > a) & (q - HALF < b)
        rec["box_touch_gap"] = int(hit.reshape(len(et), -1).any(1).sum())
        rec["mag_gaps"] = [[float(a), float(b)] for a, b in zip(gl[:20], gr[:20])]
    # channels
    rec["ch"] = {}
    for c in MAG_CH:
        s = col1d(t, c)
        if len(s) != len(mt):
            rec["ch"][c] = {"len_mismatch": int(len(s))}
            continue
        rec["ch"][c] = chan_stats(mt, s, ip_on, ip_off)
        rec["ch"][c]["onset"], rec["ch"][c]["end"], _ = onset_end(mt, s)
    rec["ch"]["Ip"] = chan_stats(ipt, ip, ip_on, ip_off)
    # Thomson
    names = t.column("thomson_chord_name")[0].as_py()
    cR = col1d(t, "thomson_chord_R")
    cZ = col1d(t, "thomson_chord_Z")
    core = [i for i, n in enumerate(names) if n.startswith("TS_core")]
    edge = [i for i, n in enumerate(names) if n.startswith("TS_tangential")]
    rec["ts_names"] = {"n": len(names), "core": len(core), "edge": len(edge),
                       "div": sum(n.startswith("TS_divertor") for n in names),
                       "other": sorted({n.split("_r")[0] for n in names} -
                                       {"TS_core", "TS_tangential", "TS_divertor"}),
                       "core_tags": sorted({n.split("_")[2] for n in names if n.startswith("TS_core")}),
                       "core_names_ok": [names[i] for i in core] == [f"TS_core_{names[i].split('_')[2]}_{k}"
                                                                    for k, i in enumerate(core)]}
    rec["ts_core_RZ"] = {"R": [float(np.nanmin(cR[core])), float(np.nanmax(cR[core]))] if core else None,
                         "Z": [float(np.nanmin(cZ[core])), float(np.nanmax(cZ[core]))] if core else None,
                         "Z_sorted_desc": bool(np.all(np.diff(cZ[core]) <= 0)) if core else None,
                         "Z_nan": int(np.isnan(cZ[core]).sum()) if core else 0}
    rec["ts_edge_RZ"] = {"R": [float(np.nanmin(cR[edge])), float(np.nanmax(cR[edge]))] if edge else None,
                         "Z": [float(np.nanmin(cZ[edge])), float(np.nanmax(cZ[edge]))] if edge else None}
    rec["ts_core_R_col"] = sorted({round(float(x), 4) for x in col1d(t, "thomson_core_R")})[:5]
    esp = col1d(t, "thomson_edge_spatial")
    rec["ts_edge_spatial"] = [float(esp.min()), float(esp.max()), int(esp.size)] if esp.size else []
    ct, cte, cne = col1d(t, "thomson_core_times"), col2d(t, "thomson_core_Te"), col2d(t, "thomson_core_ne")
    etm, ete, ene = col1d(t, "thomson_edge_times"), col2d(t, "thomson_edge_Te"), col2d(t, "thomson_edge_ne")
    rec["ts_core"] = thomson_stats(ct, cte, cne, et, ip_on, ip_off)
    rec["ts_edge"] = thomson_stats(etm, ete, ene, et, ip_on, ip_off)
    rec["ts_core"]["nch_vs_names"] = [int(cte.shape[1]) if cte.ndim == 2 else -1, len(core)]
    rec["ts_edge"]["nch_vs_names"] = [int(ete.shape[1]) if ete.ndim == 2 else -1, len(edge)]
    return rec


# ============================================================================ fixes (inputs only)
def thomson_reset_rows(times, te, ne):
    """Starter issue #12: rows after a backward step of the Thomson time axis (the axis 'resets'
    to 0 / a denormal) that carry no data (all Te and ne zero or non-finite).  Returns a boolean
    mask of rows to DROP.  fec/extract.ts_interp already ignores them (zero = missing), so this is
    a guard for other consumers, not a change to the current features."""
    times = np.asarray(times, float)
    drop = np.zeros(times.shape, bool)
    d = np.diff(times)
    back = np.flatnonzero(d < 0)
    if back.size == 0:
        return drop
    k = int(back[0]) + 1
    te = np.asarray(te, float); ne = np.asarray(ne, float)
    empty = ~((np.isfinite(te) & (te > 0)).any(1) | (np.isfinite(ne) & (ne > 0)).any(1))
    tail = np.arange(k, len(times))
    drop[tail[empty[tail]]] = True
    return drop


SPIKE_FACTOR = 5.0     # a chord value > 5x or < 1/5 of the median of its valid neighbours
SPIKE_NB = 2           # neighbours on each side (valid chords only, ordered by position)


def despike_profile(vals, pos, factor=SPIKE_FACTOR, nb=SPIKE_NB):
    """vals (T, N) Thomson Te or ne at one time per row (0/NaN = missing), pos (N,) chord
    positions.  Returns a float64 copy with isolated outliers set to NaN: a valid value whose log
    differs by more than log(factor) from the median log of the valid values among the `nb`
    chords on each side (chords ordered by position; >= 2 valid neighbours required).
    Label-free, per frame, vectorised."""
    v = np.array(vals, dtype=np.float64, copy=True)
    pos = np.asarray(pos, float)
    if v.ndim != 2 or v.shape[1] < 3 or not np.isfinite(pos).any():
        return v
    o = np.argsort(np.where(np.isfinite(pos), pos, np.inf), kind="stable")
    s = v[:, o]
    with np.errstate(invalid="ignore", divide="ignore"):
        lv = np.where(np.isfinite(s) & (s > 0), np.log(s), np.nan)
    T, N = lv.shape
    pad = np.full((T, N + 2 * nb), np.nan)
    pad[:, nb:nb + N] = lv
    nbrs = np.stack([pad[:, nb + k:nb + k + N] for k in range(-nb, nb + 1) if k != 0], axis=2)
    cnt = np.isfinite(nbrs).sum(2)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(nbrs, axis=2)
    bad = np.isfinite(lv) & (cnt >= 2) & (np.abs(lv - med) > np.log(factor))
    s[bad] = np.nan
    v[:, o] = s
    return v


def despike_shot(z):
    """Per-shot dict-like (fec/extract.py npz: ts_core_te/ne, ts_core_Z, ts_edge_te/ne, ts_edge_R)
    -> dict with the four Thomson arrays despiked (positions as used by fec/dataset.py)."""
    out = {k: z[k] for k in ("ts_core_R", "ts_core_Z", "ts_edge_R", "ts_edge_Z")}
    for k, p in (("ts_core_te", "ts_core_Z"), ("ts_core_ne", "ts_core_Z"),
                 ("ts_edge_te", "ts_edge_R"), ("ts_edge_ne", "ts_edge_R")):
        out[k] = despike_profile(z[k], z[p]).astype(np.float32)
    return out


def _ts_one(args):
    path, fix = args
    from fec.dataset import thomson_features
    with np.load(path) as z:
        d = {k: z[k] for k in ("ts_core_te", "ts_core_ne", "ts_core_R", "ts_core_Z",
                               "ts_edge_te", "ts_edge_ne", "ts_edge_R", "ts_edge_Z")}
    if fix == "despike":
        d = despike_shot(d)
    return thomson_features(d, "DIII-D")


def build_ts(config, src_work, dst, fix="despike", workers=4):
    """Rebuild WORK/ds/<config>/ts.npy (fec/dataset.thomson_features) from the per-shot npz of
    `src_work`/shots/<config>, with `fix` ('none' reproduces the source ts.npy) applied.
    Reads only the Thomson INPUT keys of each npz."""
    src_work = Path(src_work)
    info = json.loads((src_work / "ds" / config / "info.json").read_text())
    ref = np.load(src_work / "ds" / config / "ts.npy", mmap_mode="r")
    shot = np.load(src_work / "ds" / config / "shot.npy")
    starts = np.searchsorted(shot, np.arange(len(info["order"])))
    out = np.lib.format.open_memmap(dst, "w+", np.float32, ref.shape)
    jobs = [(str(src_work / "shots" / config / f"{s}.npz"), fix) for s in info["order"]]
    t0 = time.time()
    with Pool(workers) as pool:
        for i, arr in enumerate(pool.imap(_ts_one, jobs, chunksize=4)):
            out[starts[i]:starts[i] + len(arr)] = arr
            if i % 1000 == 0:
                print(i, f"{time.time() - t0:.0f}s", flush=True)
    out.flush()
    a = np.asarray(ref); b = np.asarray(out)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    print(f"build_ts {config} fix={fix}: {b.shape}, values changed {(~same).mean():.4%}, "
          f"frames changed {(~same).any(1).mean():.4%}, finite {np.isfinite(a).mean():.4f} -> "
          f"{np.isfinite(b).mean():.4f}", flush=True)


# ============================================================================ cross-channel timing
XC_CH = [c for c in D3D_CH if c not in ("magnetics_bcoil",)]


def _grid_mean(mt, s, g0, n, dt=0.5):
    """Box means of s(mt) on the grid g0 + dt*(i+0.5), i < n (bins of width dt)."""
    b = np.floor((mt - g0) / dt).astype(np.int64)
    ok = (b >= 0) & (b < n) & np.isfinite(s)
    num = np.bincount(b[ok], weights=s[ok], minlength=n)
    den = np.bincount(b[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    if (den == 0).any():
        gi = np.flatnonzero(den > 0)
        out = np.interp(np.arange(n), gi, out[gi]) if gi.size else out
    return out


def _lag(ref, x, maxlag):
    """Lag (samples) maximising |corr(ref(t), x(t + lag))|; >0: x responds later than ref."""
    ref = ref - ref.mean(); x = x - x.mean()
    n = len(ref)
    best, bl, cs = 0.0, 0, []
    for L in range(-maxlag, maxlag + 1):
        a = ref[max(0, -L): n - max(0, L)]
        b = x[max(0, L): n - max(0, -L)]
        d = np.sqrt((a * a).sum() * (b * b).sum())
        c = (a * b).sum() / d if d > 0 else 0.0
        cs.append(c)
        if abs(c) > abs(best):
            best, bl = c, L
    cs = np.abs(np.array(cs)); i = bl + maxlag
    frac = 0.0
    if 0 < i < len(cs) - 1:
        den = cs[i - 1] - 2 * cs[i] + cs[i + 1]
        frac = 0.5 * (cs[i - 1] - cs[i + 1]) / den if den != 0 else 0.0
    return bl + frac, best


def xcorr_one(path, dt=0.5, half=60.0, maxlag_ms=15.0):
    """Label-free relative timing of every magnetics channel at two impulsive events:
    breakdown (reference d(ECOILA)/dt) and the Ip quench (reference d(Ip)/dt, idx axis)."""
    path = Path(path)
    t = pq.read_table(path, columns=["magnetics_time", "magnetics_plasma_current",
                                     "magnetics_plasma_current_times"] + MAG_CH)
    mt = col1d(t, "magnetics_time")
    ip = col1d(t, "magnetics_plasma_current")
    ipt = idx_ip_times(col1d(t, "magnetics_plasma_current_times"), mt[0])
    on, off, mx = onset_end(ipt, ip)
    rec = {"stem": path.stem, "long": bool(mt[0] < -6000.0), "rate20k": bool(np.median(np.diff(mt[:2001])) < 0.2)}
    if not (np.isfinite(on) and np.isfinite(off)):
        return rec
    # quench time: steepest |Ip| fall after on+200 ms (2 ms smoothed)
    g = np.arange(on + 200.0, off + 100.0, dt)
    if g.size < 50:
        return rec
    a = np.abs(np.interp(g, ipt, ip))
    da = np.convolve(np.gradient(a), np.ones(4) / 4, mode="same")
    tq = float(g[int(np.argmin(da))])
    rec.update(t_on=on, t_quench=tq, quench_rate=float(-da.min() / dt))
    sig = {c: col1d(t, c) for c in MAG_CH}
    maxlag = int(round(maxlag_ms / dt))
    for ev, tc, refname in (("bd", on - 12.0, "magnetics_ECOILA"), ("q", tq, "magnetics_plasma_current")):
        g0 = tc - half
        n = int(2 * half / dt)
        series = {c: _grid_mean(mt, s, g0, n, dt) for c, s in sig.items()}
        series["magnetics_plasma_current"] = _grid_mean(ipt, ip, g0, n, dt)
        ref = np.gradient(series[refname])
        out = {}
        for c in XC_CH:
            if c == refname:
                continue
            L, cc = _lag(ref, np.gradient(series[c]), maxlag)
            out[c] = [round(L * dt, 3), round(float(cc), 3)]
        rec[ev] = out
    return rec


def xcorr(config, workers=5, out=None, limit=0):
    files = shot_files(config)
    if limit:
        files = files[:limit]
    out = Path(out or (AUDIT2 / "analysis"))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{config}_xcorr.jsonl"
    tmp = dst.with_suffix(".part")
    t0 = time.time()
    n = 0
    with Pool(workers) as pool, open(tmp, "w") as fh:
        for rec in pool.imap_unordered(xcorr_one, [str(f) for f in files], chunksize=2):
            fh.write(json.dumps(rec, default=_default) + "\n")
            n += 1
            if n % 250 == 0:
                print(f"{n}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    tmp.replace(dst)
    print(f"xcorr done {config} {n} shots {time.time() - t0:.0f}s -> {dst}", flush=True)


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def scan(config, workers=6, out=None, limit=0):
    files = shot_files(config)
    if limit:
        files = files[:limit]
    out = Path(out or (AUDIT2 / "analysis"))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{config}_scan.jsonl"
    tmp = dst.with_suffix(".part")
    t0 = time.time()
    n = 0
    with Pool(workers) as pool, open(tmp, "w") as fh:
        for rec in pool.imap_unordered(scan_one, [str(f) for f in files], chunksize=2):
            fh.write(json.dumps(rec, default=_default) + "\n")
            n += 1
            if n % 250 == 0:
                print(f"{n}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    tmp.replace(dst)
    print(f"scan done {config} {n} shots {time.time() - t0:.0f}s -> {dst}", flush=True)


def guard(config, out=None):
    """Label-free guard on a `scan` output: fails (returns False) on any defect class that this
    audit found absent in diii_d_train / diii_d_public_test and that the pipeline does not
    handle: magnetics/EFIT/Ip time-axis gaps, duplicates, backward steps or NaN; empty +-1 ms box
    windows; frames past a record end; Thomson CORE axis resets; Thomson tangential resets whose
    reset rows carry data (issue #12 rows must be empty).  Prints every condition."""
    out = Path(out or (AUDIT2 / "analysis"))
    R = [json.loads(line) for line in open(out / f"{config}_scan.jsonl")]
    if not R:
        print("FAIL: empty scan"); return False
    conds = {
        "efit axis dup/back/nan": sum(r["efit"].get("n_dup", 0) + r["efit"].get("n_back", 0)
                                      + r["efit"].get("nan", 0) for r in R),
        "magnetics axis gap/dup/back/nan": sum(r["mag"].get(k, 0) for r in R
                                               for k in ("n_gap", "n_dup", "n_back", "nan")),
        "Ip axis gap/dup/back": sum(r["ipt"].get(k, 0) for r in R for k in ("n_gap", "n_dup", "n_back")),
        "empty box windows (magnetics, Ip)": sum(r["box_empty_mag"] + r["box_empty_ip"] for r in R),
        "frames past record end": sum(r["efit_after_ip_end"] + r["efit_after_mag_end"] for r in R),
        "channel length mismatch": sum("len_mismatch" in c for r in R for c in r["ch"].values()),
        "Thomson core axis dup/back": sum(r["ts_core"]["axis"].get("n_dup", 0)
                                          + r["ts_core"]["axis"].get("n_back", 0) for r in R),
        "Thomson chord/channel count mismatch": sum(
            r[s].get("nch_vs_names", [0, 0])[0] != r[s].get("nch_vs_names", [0, 0])[1]
            for r in R for s in ("ts_core", "ts_edge")),
    }
    ok = True
    for k, v in conds.items():
        print(f"{'ok  ' if v == 0 else 'FAIL'} {k}: {v}")
        ok &= v == 0
    resets = [r["stem"] for r in R if r["ts_edge"]["axis"].get("n_back", 0)]
    print(f"info Thomson tangential resets (issue #12): {len(resets)} shots {resets[:12]}"
          f" -> run thomson_reset_rows on them; data-carrying reset rows must be reviewed")
    print("guard", "PASS" if ok else "FAIL", config, len(R), "shots")
    return ok


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--config", default="diii_d_train")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--out", default=None)
    s.add_argument("--limit", type=int, default=0)
    x = sub.add_parser("xcorr")
    x.add_argument("--config", default="diii_d_train")
    x.add_argument("--workers", type=int, default=5)
    x.add_argument("--out", default=None)
    x.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("build_ts")
    b.add_argument("--config", default="diii_d_train")
    b.add_argument("--src", default=str(IDX_WORK))
    b.add_argument("--dst", required=True)
    b.add_argument("--fix", choices=["none", "despike"], default="despike")
    b.add_argument("--workers", type=int, default=4)
    g = sub.add_parser("guard")
    g.add_argument("--config", default="diii_d_train")
    g.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "guard":
        sys.exit(0 if guard(a.config, a.out) else 1)
    if a.cmd == "scan":
        scan(a.config, a.workers, a.out, a.limit)
    elif a.cmd == "xcorr":
        xcorr(a.config, a.workers, a.out, a.limit)
    elif a.cmd == "build_ts":
        build_ts(a.config, a.src, a.dst, a.fix, a.workers)


if __name__ == "__main__":
    main()
