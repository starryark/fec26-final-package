"""DIII-D plasma-current (Ip) time-base audit and an input-only correction.

Study note: docs/research/D3D_IP_TIMEBASE_AUDIT_20261008.md.

Every DIII-D file ships the same `magnetics_plasma_current_times` axis (30719 samples, 0.5 ms
from -858.19 ms).  The starter's `fix_d3d_ip_times` (== `fec.extract.d3d_ip_times`) keeps it
when the first |Ip| crossing of 10% of p99.5 lands in [-500, 1000] ms and otherwise re-origins
it to magnetics_time[0].  Every axis considered here is `t0 + (shipped - shipped[0])`, so a rule
is one number per shot: the time t0 of Ip sample 0.

Label use (MEASUREMENT only, TRAIN shots only): the EFIT-implied plasma current at efit_times,
sum of -Delta*psi/(mu0 R) dR dZ inside the EFIT LCFS (`fec.d3d_physics.label_node_currents`),
gives per shot the label-aligned start t0_label (least squares over a shift).  Candidate rules
read INPUT columns only (magnetics_*, efit_times scaffold); they are applied to every shot,
including VAL/TEST shots, whose labels are never read here.

Magnetics record classes (inputs only): "short" records start near -4.06 s (all 20 kHz);
"long" records start near -9.23 s (2 kHz, plus two 20 kHz TRAIN shots).  The starter re-origins
every short record and keeps the shipped axis on every long record.

Rules (t0 of Ip sample 0; inputs only):
  shipped  the shipped axis
  starter  fec.extract.d3d_ip_times (current pipeline)
  mag0     magnetics_time[0] on every shot (the blanket shift the starter warns against)
  idx      record-clock rule: t0 = magnetics_time[0] + (8192 ms if long record else 0)
  idxe     idx minus EFIT_LEAD_MS (TRAIN-median lead of EFIT's current over measured Ip)
  ecoil    Ip breakdown onset aligned to the ECOILA breakdown-swing onset on magnetics_time:
           t0 = ecoil_onset - (Ip onset - t0 on the starter axis) + ECOIL_BIAS_MS

    python -B fec/ip_timebase.py scan [--config C] [--workers 6]   # per-shot inputs (+TRAIN labels)
    python -B fec/ip_timebase.py check [--config C]   # label-free guard; exit 1 on any failure
    python -B fec/ip_timebase.py report                           # tables -> report.json
    python -B fec/ip_timebase.py features --rule idx [--workers 6] # corrected Ip box means
`scan` works before any extraction (no WORK/ds/<C>/info.json -> parquet list, no labels), so a
new fold can be guarded first; `check` fails on the conditions of
research/reviews/fec_ip_timebase_review_20261008.md sec. 6 (see `guard`).
Outputs: WORK/ip_timebase/ ([<C>_]shots.json, frames_train.npz, report.json, ipcur_<rule>.npy).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import DATA, WORK  # noqa: E402

CONFIG = "diii_d_train"
OUT = WORK / "ip_timebase"
IP_COL = "magnetics_plasma_current"
IPT_COL = "magnetics_plasma_current_times"
F_COLS = [f"magnetics_F{i}{s}" for s in "AB" for i in range(1, 10)]
IN_COLS = ["efit_times", "magnetics_time", IP_COL, IPT_COL, "magnetics_ECOILA"]
OFFSETS = np.array([-100, -50, -20, -10, -5, -2, 0, 2, 5, 10, 20], dtype=np.float64)  # extract.py
HALF = 1.0                                                                             # extract.py
RATE_SPLIT_MS = 0.2          # median magnetics_time step < 0.2 ms -> "20k" (0.05 ms) else "2k"
# rule constants, frozen from the TRAIN scan (report: t0_label - magnetics_time[0] per rate)
LONG_RECORD_T0_MS = -6000.0   # magnetics_time[0] < this -> long record (gap -8998 .. -3927 ms)
IDX_LONG_OFFSET_MS = 8192.0   # 16384 samples x 0.5 ms; TRAIN: 8190.24 - (-1.65) = 8191.9 ms
EFIT_LEAD_MS = 1.7            # TRAIN median of (idx t0 - label t0), both classes (1.65 / 1.76)
ECOIL_BIAS_MS = 13.4          # TRAIN median of (label t0 - ecoil t0 at bias 0)
LAG_SEARCH_MS = (-1200.0, 1200.0)


# ============================================================================ small helpers
def _col(t, name):
    from fec.extract import col1d
    return col1d(t, name)


def rate_class(mt):
    return "20k" if float(np.median(np.diff(mt[: min(len(mt), 20001)]))) < RATE_SPLIT_MS else "2k"


def starter_t0(ipt, ip, mt0):
    """t0 chosen by the starter's fix_d3d_ip_times (fec.extract.d3d_ip_times)."""
    from fec.extract import d3d_ip_times
    return float(d3d_ip_times(ipt, ip, mt0)[0])


def axis(ipt, t0):
    return float(t0) + (ipt - ipt[0])


def idx_ip_times(ipt, mag_t0):
    """Drop-in replacement for fec.extract.d3d_ip_times(ipt, ip, mag_t0) (inputs only):
    Ip sample 0 sits at magnetics_time[0] on short records and 8192 ms later on long records
    (magnetics_time[0] < -6000 ms).  On every diii_d_train / diii_d_public_test shot this equals
    the starter axis on short records and moves long records 56-330 ms earlier."""
    mag_t0 = float(mag_t0)
    return axis(ipt, mag_t0 + (IDX_LONG_OFFSET_MS if mag_t0 < LONG_RECORD_T0_MS else 0.0))


def onset_time(ts, s, frac_lo=0.02, frac_hi=0.10, pre_ms=300.0, smooth_ms=2.0):
    """Start of the first large excursion of s(ts) from its pre-excursion level.

    Level crossing at frac_hi x (p99.5 |s - base|) locates the excursion, then the onset is the
    last time before it where |s - base| <= frac_lo x amplitude (base = median over pre_ms
    before the crossing).  Returns NaN if no excursion."""
    ok = np.isfinite(ts) & np.isfinite(s)
    ts, s = ts[ok], s[ok]
    if ts.size < 50:
        return float("nan")
    dt = float(np.median(np.diff(ts)))
    k = max(1, int(round(smooth_ms / max(dt, 1e-6))))
    if k > 1:
        s = np.convolve(s, np.ones(k) / k, mode="same")
    base0 = float(np.median(s[: max(10, len(s) // 50)]))
    amp = float(np.nanpercentile(np.abs(s - base0), 99.5))
    if not amp > 0:
        return float("nan")
    hit = np.flatnonzero(np.abs(s - base0) > frac_hi * amp)
    if hit.size == 0:
        return float("nan")
    i1 = int(hit[0])
    pre = (ts >= ts[i1] - pre_ms) & (ts < ts[i1] - 20.0)
    base = float(np.median(s[pre])) if pre.sum() >= 5 else base0
    amp = float(np.nanpercentile(np.abs(s - base), 99.5))
    j = i1
    while j > 0 and abs(s[j] - base) > frac_lo * amp:
        j -= 1
    return float(ts[j])


def ecoil_onset(mt, ec, window=(-300.0, 600.0)):
    """E-coil breakdown swing onset on magnetics_time (inputs only): the fastest ECOILA change
    (+-5 ms rate of a 2 ms box mean) inside `window` (DIII-D t=0 convention), walked back to
    where the rate first fell below 10% of that maximum.  (The slower pre-charge ramp ends
    near -400 ms and is excluded by the window.)"""
    m = (mt >= window[0]) & (mt <= window[1]) & np.isfinite(ec)
    if m.sum() < 100:
        return float("nan")
    ts, s = mt[m], ec[m]
    dt = float(np.median(np.diff(ts)))
    k = max(1, int(round(2.0 / dt)))
    if k > 1:
        s = np.convolve(s, np.ones(k) / k, mode="same")[k:-k]
        ts = ts[k:-k]
    # rate over +-5 ms
    w = max(1, int(round(5.0 / dt)))
    r = np.zeros_like(s)
    r[w:-w] = (s[2 * w:] - s[:-2 * w]) / (ts[2 * w:] - ts[:-2 * w])
    a = np.abs(r)
    amax = float(a.max())
    if not amax > 0:
        return float("nan")
    i1 = int(np.argmax(a))
    j = i1
    while j > 0 and a[j] > 0.1 * amax:
        j -= 1
    return float(ts[j])


def ip_onset(ax, ip):
    """Ip breakdown onset on a given Ip axis (inputs only)."""
    return onset_time(ax, np.abs(ip), frac_lo=0.02, frac_hi=0.10)


def box_means(ts, s, centers, half=HALF):
    from fec.extract import box_means as bm
    return bm(ts, s, centers, half)


# ============================================================================ labels (TRAIN only)
def efit_ip_ka(stem):
    """|EFIT-implied Ip| [kA] per frame of a TRAIN shot from WORK/shots (labels).  Caller must
    ensure the shot is a TRAIN shot."""
    from fec.d3d_physics import label_node_currents
    z = np.load(WORK / "shots" / CONFIG / f"{stem}.npz")
    n = len(z["t"])
    lc = np.full((n, z["lcfs_r"].shape[1], 2), np.nan, np.float32)
    lc[..., 0] = z["lcfs_r"]
    lc[..., 1] = z["lcfs_z"]
    In = label_node_currents(-np.asarray(z["psi"]), lc, z["lcfs_n"], z["grid_R"], z["grid_Z"])
    return np.asarray(z["t"], np.float64), np.abs(In.reshape(n, -1).sum(1).astype(np.float64)) / 1e3


def label_lag(et, ie, ax, ip, search=LAG_SEARCH_MS, coarse=1.0, use_mask=None):
    """Least-squares shift tau [ms] with |Ip|(ax)(t + tau) ~ g * ie(t) on frames with
    ie > max(50 kA, 5% max).  tau > 0: our axis is LATE by tau (true t0 = ax[0] - tau).
    Coarse scan with g = 1, then 0.05 ms refinement with free gain.  Returns dict."""
    ok = np.isfinite(ip) & np.isfinite(ax)
    a, tx = np.abs(ip[ok]), ax[ok]
    use = np.isfinite(ie) & (ie > max(50.0, 0.05 * np.nanmax(ie)))
    if use_mask is not None:
        use &= use_mask
    out = {"n_use": int(use.sum())}
    if use.sum() < 5:
        return out
    e, tk = ie[use], et[use]
    taus = np.arange(search[0], search[1] + coarse / 2, coarse)
    M = np.interp((tk[None, :] + taus[:, None]).ravel(), tx, a).reshape(len(taus), -1)
    sse = ((M - e[None]) ** 2).sum(1)
    k = int(np.argmin(sse))
    # second-best separated minimum (>= 30 ms away): identifiability
    far = np.abs(taus - taus[k]) >= 30.0
    sse2 = float(sse[far].min()) if far.any() else float("nan")
    fine = taus[k] + np.arange(-2 * coarse, 2 * coarse + 1e-9, 0.05)
    Mf = np.interp((tk[None, :] + fine[:, None]).ravel(), tx, a).reshape(len(fine), -1)
    g = (Mf * e[None]).sum(1) / np.maximum((Mf * Mf).sum(1), 1e-30)
    ssef = ((g[:, None] * Mf - e[None]) ** 2).sum(1)
    kf = int(np.argmin(ssef))
    tau = float(fine[kf])
    m0 = np.interp(tk, tx, a)
    # curvature-based 1-sigma (g=1 coarse profile, +-5 ms)
    sel = np.abs(taus - taus[k]) <= 5.0
    sig = float("nan")
    if sel.sum() >= 5:
        c = np.polyfit(taus[sel] - taus[k], sse[sel], 2)[0]
        if c > 0:
            sig = float(np.sqrt(2 * ssef[kf] / max(use.sum() - 2, 1) / (2 * c)))
    out.update(tau=tau, gain=float(g[kf]), rms0=float(np.sqrt(((m0 - e) ** 2).mean())),
               rms_best=float(np.sqrt(ssef[kf] / use.sum())), sse_min=float(sse[k]),
               sse_2nd=sse2, sigma=sig, ie_range=float(e.max() - e.min()))
    return out


# ============================================================================ per-shot scan
def _pq_path(config, stem):
    """Parquet of a shot: read-only E: root first, then the writable HF mirror (fec.paths)."""
    from fec.paths import HF_DATA
    f = DATA / config / f"{stem}.parquet"
    return f if f.exists() else HF_DATA / config / f"{stem}.parquet"


def _out_name(config, name):
    return name if config == CONFIG else f"{config}_{name}"


def scan_one(args):
    stem, split, config = args
    import pyarrow.parquet as pq
    t = pq.read_table(_pq_path(config, stem), columns=IN_COLS)
    et, mt, ip, ipt, ec = (_col(t, c) for c in IN_COLS)
    rec = {"stem": stem, "split": split, "n_frames": int(len(et)), "et0": float(et[0]),
           "et1": float(et[-1]), "n_mag": int(len(mt)), "mag_t0": float(mt[0]),
           "mag_t1": float(mt[-1]), "mag_dt": float(np.median(np.diff(mt))),
           "rate": rate_class(mt), "n_ip": int(len(ip)), "ipt0": float(ipt[0]),
           "ipt1": float(ipt[-1]), "ipt_dt": float(np.median(np.diff(ipt))),
           "ipt_dt_min": float(np.min(np.diff(ipt))), "ipt_dt_max": float(np.max(np.diff(ipt))),
           "ip_finite": float(np.isfinite(ip).mean()),
           "ip_p995": float(np.nanpercentile(np.abs(ip), 99.5)),
           "ip_sign": float(np.sign(np.nansum(ip)))}
    t0s = starter_t0(ipt, ip, mt[0])
    rec["t0_starter"] = t0s
    rec["starter_fixed"] = bool(t0s != float(ipt[0]))
    ax = axis(ipt, t0s)
    rec["ip_onset_starter"] = ip_onset(ax, ip)                  # on the starter axis
    rec["ip_onset_rel"] = rec["ip_onset_starter"] - t0s          # onset relative to sample 0
    rec["ecoil_onset"] = ecoil_onset(mt, ec)
    rec["ec_pre"] = float(np.nanmedian(ec[(mt > -400) & (mt < -100)])) if ((mt > -400) & (mt < -100)).any() else float("nan")
    frames = None
    if config == CONFIG and split == "train":                     # labels: TRAIN shots only
        etl, ie = efit_ip_ka(stem)
        assert np.array_equal(etl, et), stem
        lag = label_lag(et, ie, ax, ip)
        rec.update({f"lab_{k}": v for k, v in lag.items()})
        if "tau" in lag:
            rec["t0_label"] = t0s - lag["tau"]
            # ramp-up / ramp-down only (constant offset vs. clock drift), around the global fit
            imax = np.nanmax(ie)
            hi = np.flatnonzero(ie >= 0.9 * imax)
            up = np.zeros(len(ie), bool); dn = np.zeros(len(ie), bool)
            if hi.size:
                up[: hi[0]] = True
                dn[hi[-1] + 1:] = True
            for nm, msk in (("up", up), ("dn", dn)):
                if msk.sum() >= 3 and np.ptp(ie[msk]) > 100.0:
                    r = label_lag(et, ie, ax, ip, search=(lag["tau"] - 100, lag["tau"] + 100),
                                  coarse=0.5, use_mask=msk)
                    if "tau" in r:
                        rec[f"lab_tau_{nm}"] = r["tau"]
                        rec[f"lab_n_{nm}"] = r["n_use"]
        ipm = box_means(ax, ip, et)
        frames = {"t": et, "ie": ie, "ip_starter": np.abs(ipm)}
    return rec, frames


def scan(workers=6, limit=0, config=CONFIG):
    """Per-shot input metadata for every shot of `config` (+ TRAIN-label lags on diii_d_train
    'train' shots only; other configs/splits are inputs-only)."""
    f_info = WORK / "ds" / config / "info.json"
    if f_info.exists():
        info = json.loads(f_info.read_text())
        order, splits = info["order"], info["splits"]
    else:                       # before extraction (e.g. a new fold): no split -> no labels
        from fec.paths import shot_files
        order = [f.stem for f in shot_files(config)]
        if not order:
            raise SystemExit(f"no parquet files for {config}")
        splits = ["unsplit"] * len(order)
    jobs = [(s, sp, config) for s, sp in zip(order, splits)]
    if limit:
        jobs = jobs[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    recs, fr = {}, {}
    t0 = time.time()
    with Pool(workers) as pool:
        for n, (rec, frames) in enumerate(pool.imap_unordered(scan_one, jobs, chunksize=4)):
            recs[rec["stem"]] = rec
            if frames is not None:
                fr[rec["stem"]] = frames
            if n % 500 == 0:
                print(n, f"{time.time() - t0:.0f}s", flush=True)
    rows = [recs[s] for s, _, _ in jobs]
    (OUT / _out_name(config, "shots.json")).write_text(json.dumps(rows))
    if not fr:
        print(f"scan done {len(rows)} shots (inputs only) {time.time() - t0:.0f}s", flush=True)
        return
    st = [s for s, _, _ in jobs if s in fr]
    pos = {s: i for i, s in enumerate(order)}
    shot_idx = np.concatenate([np.full(len(fr[s]["t"]), pos[s], np.int32) for s in st])
    np.savez(OUT / "frames_train.npz", shot=shot_idx,
             **{k: np.concatenate([fr[s][k] for s in st]) for k in ("t", "ie", "ip_starter")})
    print(f"scan done {len(rows)} shots ({len(st)} TRAIN with labels) {time.time() - t0:.0f}s",
          flush=True)


# ============================================================================ rules (inputs only)
def rule_t0(rec, rule):
    """t0 of Ip sample 0 under `rule`, from the per-shot INPUT metadata of `scan`."""
    if rule == "starter":
        return rec["t0_starter"]
    if rule == "shipped":
        return rec["ipt0"]
    if rule == "mag0":
        return rec["mag_t0"]
    if rule == "idx":
        return rec["mag_t0"] + (IDX_LONG_OFFSET_MS if record_class(rec) == "long" else 0.0)
    if rule == "idxe":
        return rule_t0(rec, "idx") - EFIT_LEAD_MS
    if rule == "ecoil":
        if not (np.isfinite(rec["ecoil_onset"]) and np.isfinite(rec["ip_onset_rel"])):
            return rec["t0_starter"]
        return rec["ecoil_onset"] - rec["ip_onset_rel"] + ECOIL_BIAS_MS
    raise ValueError(rule)


RULES = ("shipped", "starter", "mag0", "idx", "idxe", "ecoil")


def record_class(rec):
    """'long' (magnetics record starts before LONG_RECORD_T0_MS) or 'short'; inputs only."""
    return "long" if rec["mag_t0"] < LONG_RECORD_T0_MS else "short"


# ============================================================================ report
def _q(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    d = {f"p{q}": round(float(np.percentile(x, q)), 3) for q in qs}
    d.update(n=int(x.size), mean=round(float(x.mean()), 3), min=round(float(x.min()), 3),
             max=round(float(x.max()), 3))
    return d


def report():
    rows = json.loads((OUT / "shots.json").read_text())
    tr = [r for r in rows if r["split"] == "train" and "t0_label" in r]
    rep = {"n_shots": len(rows), "n_train_with_lag": len(tr),
           "shipped_axis_identical": len({(r["ipt0"], r["n_ip"], round(r["ipt_dt"], 6)) for r in rows}) == 1,
           "ipt0_values": sorted({r["ipt0"] for r in rows})[:5],
           "n_ip_values": sorted({r["n_ip"] for r in rows})[:5]}
    # cross-tab rate x starter decision (all splits; inputs only)
    xt = {}
    for r in rows:
        k = f"{record_class(r)}|{r['rate']}|fixed={r['starter_fixed']}"
        xt.setdefault(k, {"train": 0, "val": 0, "test": 0})
        xt[k][r["split"]] += 1
    rep["rate_x_starter"] = xt
    # lag of the starter axis (tau > 0 = late) by group; identifiable subset
    def ident(r):
        return (r.get("lab_n_use", 0) >= 5 and np.isfinite(r.get("lab_sigma", np.nan))
                and r["lab_sigma"] < 5 and r.get("lab_sse_2nd", 0) > 2 * r.get("lab_sse_min", 1)
                and r.get("lab_ie_range", 0) > 200)
    rep["n_identifiable"] = int(sum(ident(r) for r in tr))
    groups = {}
    for r in tr:
        for g in ("all", record_class(r), r["rate"], f"{record_class(r)}|{r['rate']}",
                  f"{r['rate']}|fixed={r['starter_fixed']}"):
            groups.setdefault(g, []).append(r)
    rep["starter_lag_ms"] = {g: {"all": _q([r["lab_tau"] for r in rs]),
                                 "identifiable": _q([r["lab_tau"] for r in rs if ident(r)]),
                                 "frac_abs_gt20": round(float(np.mean([abs(r["lab_tau"]) > 20 for r in rs])), 4),
                                 "frac_abs_gt5": round(float(np.mean([abs(r["lab_tau"]) > 5 for r in rs])), 4)}
                             for g, rs in groups.items()}
    rep["label_minus_mag0_ms"] = {g: _q([r["t0_label"] - r["mag_t0"] for r in rs if ident(r)])
                                  for g, rs in groups.items()}
    rep["gain"] = _q([r["lab_gain"] for r in tr if ident(r)])
    rep["rms_kA"] = {"starter": _q([r["lab_rms0"] for r in tr]),
                     "label_aligned": _q([r["lab_rms_best"] for r in tr])}
    du = [r["lab_tau_up"] - r["lab_tau_dn"] for r in tr if "lab_tau_up" in r and "lab_tau_dn" in r and ident(r)]
    rep["tau_up_minus_dn_ms"] = _q(du)
    # residual misalignment of each rule (t0_rule - t0_label; > 0 = rule axis late)
    rep["rule_residual_ms"] = {}
    for rule in RULES:
        res = {}
        for g, rs in groups.items():
            d = [rule_t0(r, rule) - r["t0_label"] for r in rs if ident(r)]
            res[g] = _q(d)
            res[g]["frac_abs_gt5"] = round(float(np.mean(np.abs(d) > 5)), 4) if d else None
            res[g]["frac_abs_gt20"] = round(float(np.mean(np.abs(d) > 20)), 4) if d else None
        rep["rule_residual_ms"][rule] = res
    # campaign proxies: magnetics start time / length vs residual
    rep["mag_t0_by_rate"] = {k: _q([r["mag_t0"] for r in rows if r["rate"] == k]) for k in ("20k", "2k")}
    rep["n_mag_by_rate"] = {k: sorted({r["n_mag"] for r in rows if r["rate"] == k})[:10] for k in ("20k", "2k")}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


# ============================================================================ corrected features
def features_one(args):
    stem, t0, config = args
    import pyarrow.parquet as pq
    t = pq.read_table(_pq_path(config, stem), columns=["efit_times", IP_COL, IPT_COL])
    et, ip, ipt = (_col(t, c) for c in ("efit_times", IP_COL, IPT_COL))
    ax = axis(ipt, t0)
    q = (et[:, None] + OFFSETS[None, :]).ravel()
    return stem, box_means(ax, ip, q).reshape(len(et), len(OFFSETS)).astype(np.float32)


def features(rule, workers=6, config=CONFIG):
    """Ip box means at efit_times + OFFSETS under `rule` for every frame of `config` (inputs
    only), in WORK/ds/<config> frame order: OUT/[<config>_]ipcur_<rule>.npy (N, 11) kA, signed
    (same units/sign/box means as ds cur[:, Ip, :]; rule 'starter' reproduces it exactly)."""
    d = WORK / "ds" / config
    info = json.loads((d / "info.json").read_text())
    order = info["order"]
    shot = np.load(d / "shot.npy")
    starts = np.searchsorted(shot, np.arange(len(order)))
    ends = np.searchsorted(shot, np.arange(len(order)), side="right")
    rows = {r["stem"]: r for r in json.loads((OUT / _out_name(config, "shots.json")).read_text())}
    jobs = [(s, rule_t0(rows[s], rule), config) for s in order]
    out = np.full((len(shot), len(OFFSETS)), np.nan, np.float32)
    pos = {s: i for i, s in enumerate(order)}
    t0 = time.time()
    with Pool(workers) as pool:
        for n, (stem, arr) in enumerate(pool.imap_unordered(features_one, jobs, chunksize=8)):
            i = pos[stem]
            assert ends[i] - starts[i] == len(arr), stem
            out[starts[i]:ends[i]] = arr
            if n % 1000 == 0:
                print(n, f"{time.time() - t0:.0f}s", flush=True)
    f = OUT / _out_name(config, f"ipcur_{rule}.npy")
    np.save(f, out)
    meta = {"config": config, "rule": rule, "long_record_t0_ms": LONG_RECORD_T0_MS,
            "idx_long_offset_ms": IDX_LONG_OFFSET_MS, "efit_lead_ms": EFIT_LEAD_MS,
            "ecoil_bias_ms": ECOIL_BIAS_MS,
            "offsets_ms": OFFSETS.tolist(), "half_ms": HALF, "n": int(len(out)),
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            "src_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / _out_name(config, f"ipcur_{rule}.json")).write_text(json.dumps(meta, indent=1))
    print("features done", meta, f"{time.time() - t0:.0f}s", flush=True)


# ============================================================================ label-free guard
# Fail conditions of research/reviews/fec_ip_timebase_review_20261008.md sec. 6.  Constants are
# the values observed on diii_d_train (7041 shots) and diii_d_public_test (874), inputs only.
KNOWN_IP_AXIS = {"ipt0": -858.1870727539062, "ipt1": 14500.8125, "n_ip": 30719, "dt": 0.5}
AXIS_TOL_MS = 1e-3
SHORT_START_MS = (-4301.22 - 100.0, -3926.54 + 100.0)   # observed short-record starts +-100 ms
LONG_START_MS = (-9371.81 - 100.0, -8998.45 + 100.0)    # observed long-record starts +-100 ms
ECOIL_BAND_MS = (-20.0, -5.0)   # E-coil breakdown-swing onset on 7003/7041 + 868/874 shots
ECOIL_MIN_COVERAGE = 0.99       # observed 0.9946 / 0.9931
GUARD_MS = 25.0                 # per shot |t0_idx - t0_ecoil|; max seen in band 19.8 / 18.3 ms
CLASS_DELAY_TOL_MS = 2.0        # |median delay long - median delay short| (= |8192 - implied|)
REF_DELAY_MS = 14.9             # median (Ip onset on idx axis - E-coil onset): 14.80 / 15.00 ms
REF_DELAY_TOL_MS = 3.0
MIN_CLASS_N = 10                # in-band shots needed to judge a present class (else fail closed)


def breakdown_delay(rec):
    """Ip onset on the idx axis minus the E-coil breakdown-swing onset [ms]; inputs only."""
    return rec["ip_onset_rel"] + rule_t0(rec, "idx") - rec["ecoil_onset"]


def _axis_ok(r):
    ok = (r["n_ip"] == KNOWN_IP_AXIS["n_ip"]
          and abs(r["ipt0"] - KNOWN_IP_AXIS["ipt0"]) <= AXIS_TOL_MS
          and abs(r["ipt1"] - KNOWN_IP_AXIS["ipt1"]) <= AXIS_TOL_MS
          and abs(r["ipt_dt"] - KNOWN_IP_AXIS["dt"]) <= AXIS_TOL_MS)
    for k in ("ipt_dt_min", "ipt_dt_max"):          # recorded by scans after 2026-10-08 03:00
        if k in r:
            ok = ok and abs(r[k] - KNOWN_IP_AXIS["dt"]) <= 1e-3
    return bool(ok)


def _in_cluster(m0):
    return SHORT_START_MS[0] <= m0 <= SHORT_START_MS[1] or LONG_START_MS[0] <= m0 <= LONG_START_MS[1]


def guard(rows):
    """The six fail conditions of review sec. 6, plus the per-shot E-coil tolerance, on scanned
    per-shot INPUT metadata.  Returns {"n", "classes", "conditions": {name: {"pass", ...}},
    "n_failed"}; any failed condition means the fold needs a reviewed decision."""
    cond = {}
    n = len(rows)
    cls = [record_class(r) for r in rows]
    # 1. every shipped Ip axis is the single known axis
    bad = [r["stem"] for r in rows if not _axis_ok(r)]
    cond["1_shipped_ip_axis_known"] = {
        "pass": n > 0 and not bad, "n_bad": len(bad), "bad": bad[:50], "known": KNOWN_IP_AXIS,
        "seen": sorted({(r["ipt0"], r["ipt1"], r["n_ip"], r["ipt_dt"]) for r in rows})[:10]}
    # 2. starter decision equals record class on every shot
    bad = [r["stem"] for r, c in zip(rows, cls) if (c == "short") != bool(r["starter_fixed"])]
    cond["2_starter_decision_eq_class"] = {"pass": n > 0 and not bad, "n_bad": len(bad),
                                           "bad": bad[:50]}
    # 3. every magnetics_time[0] inside the observed clusters +-100 ms (none in the gap)
    bad = [(r["stem"], r["mag_t0"]) for r in rows if not _in_cluster(r["mag_t0"])]
    cond["3_record_start_in_clusters"] = {"pass": n > 0 and not bad, "n_bad": len(bad),
                                          "bad": bad[:50], "short_ms": SHORT_START_MS,
                                          "long_ms": LONG_START_MS}
    # 4. E-coil coverage >= 99% in band (NaN counts as out of band); every miss listed
    inb = [bool(np.isfinite(r["ecoil_onset"]) and np.isfinite(r["ip_onset_rel"])
                and ECOIL_BAND_MS[0] <= r["ecoil_onset"] <= ECOIL_BAND_MS[1]) for r in rows]
    cov = float(np.mean(inb)) if n else 0.0
    cond["4_ecoil_coverage"] = {"pass": n > 0 and cov >= ECOIL_MIN_COVERAGE,
                                "coverage": round(cov, 5), "min": ECOIL_MIN_COVERAGE,
                                "out_of_band": [(r["stem"], r["ecoil_onset"])
                                                for r, b in zip(rows, inb) if not b]}
    band = [(r, c) for r, c, b in zip(rows, cls, inb) if b]
    # per-shot tolerance on in-band shots (the first guard's test)
    d = np.array([rule_t0(r, "idx") - rule_t0(r, "ecoil") for r, _ in band])
    bad = [r["stem"] for (r, _), x in zip(band, d) if abs(x) > GUARD_MS]
    cond["0_per_shot_idx_vs_ecoil"] = {"pass": not bad, "tol_ms": GUARD_MS, "n_bad": len(bad),
                                       "bad": bad[:50],
                                       "idx_minus_ecoil_ms": _q(d, (1, 50, 99)) if len(d) else {}}
    # 5./6. per-class median breakdown delay on the idx axis
    med, cnt = {}, {}
    for c in ("short", "long"):
        dl = [breakdown_delay(r) for r, cc in band if cc == c]
        cnt[c] = {"shots": cls.count(c), "in_band": len(dl)}
        med[c] = float(np.median(dl)) if len(dl) >= MIN_CLASS_N else None
    present = [c for c in ("short", "long") if cnt[c]["shots"] > 0]
    judged = bool(present) and all(med[c] is not None for c in present)
    diff = implied = None
    if len(present) == 2 and judged:
        diff = med["long"] - med["short"]
        ok5 = abs(diff) <= CLASS_DELAY_TOL_MS
        implied = IDX_LONG_OFFSET_MS - diff        # input-only estimate of the 8192 ms offset
    else:
        ok5 = len(present) == 1 and judged         # single-class fold: judged by 6 alone
    cond["5_class_delay_agreement"] = {"pass": bool(ok5), "median_delay_ms": med,
                                       "n_class": cnt, "long_minus_short_ms": diff,
                                       "implied_long_offset_ms": implied,
                                       "tol_ms": CLASS_DELAY_TOL_MS}
    ok6 = judged and all(abs(med[c] - REF_DELAY_MS) <= REF_DELAY_TOL_MS for c in present)
    cond["6_delay_matches_reference"] = {"pass": bool(ok6), "median_delay_ms": med,
                                         "ref_ms": REF_DELAY_MS, "tol_ms": REF_DELAY_TOL_MS,
                                         "min_class_n": MIN_CLASS_N}
    return {"n": n, "classes": {k: cls.count(k) for k in ("short", "long")},
            "conditions": dict(sorted(cond.items())),
            "n_failed": int(sum(not v["pass"] for v in cond.values()))}


def check(config=CONFIG):
    """Inputs-only guard of the idx rule on a scanned config (e.g. a new test fold).  Prints
    every condition and returns the number of failed conditions (CLI exit code 1 if any).  A
    failing fold needs a reviewed decision, not a silent fallback to either rule."""
    rows = json.loads((OUT / _out_name(config, "shots.json")).read_text())
    out = dict(config=config, **guard(rows))
    print(json.dumps(out, indent=1, default=float))
    failed = [k for k, v in out["conditions"].items() if not v["pass"]]
    print(f"check {config}: {'PASS' if not failed else 'FAIL'} {failed}", flush=True)
    return out["n_failed"]


# ============================================================================ TRAIN frame audit
def frame_audit(rules=("starter", "idx", "ecoil")):
    """TRAIN frames only: |Ip_rule(t)| (box mean, offset 0) vs EFIT-implied Ip per phase."""
    fr = dict(np.load(OUT / "frames_train.npz"))
    ds_shot = np.load(WORK / "ds" / CONFIG / "shot.npy")
    t_ms = np.load(WORK / "ds" / CONFIG / "t_ms.npy")
    info = json.loads((WORK / "ds" / CONFIG / "info.json").read_text())
    rows = json.loads((OUT / "shots.json").read_text())
    rate = np.array([record_class(r) for r in rows])
    # map frames_train rows -> ds rows (shot index + time)
    key_ds = ds_shot.astype(np.int64) * 10 ** 7 + np.round(t_ms).astype(np.int64)
    key_fr = fr["shot"].astype(np.int64) * 10 ** 7 + np.round(fr["t"]).astype(np.int64)
    srt = np.argsort(key_ds)
    pos = srt[np.searchsorted(key_ds[srt], key_fr)]
    assert np.array_equal(key_ds[pos], key_fr)
    assert set(np.array(info["splits"])[fr["shot"]]) == {"train"}
    ie = fr["ie"]
    # phase from the EFIT current (TRAIN labels; audit only)
    ph = np.full(len(ie), "flat", dtype=object)
    for s in np.unique(fr["shot"]):
        ix = np.flatnonzero(fr["shot"] == s)
        top = np.flatnonzero(ie[ix] >= 0.9 * ie[ix].max())
        ph[ix[:top[0]]] = "ramp_up"
        ph[ix[top[-1] + 1:]] = "ramp_down"
    rep = {}
    for rule in rules:
        ipr = np.abs(np.load(OUT / f"ipcur_{rule}.npy", mmap_mode="r")[pos, 6].astype(np.float64))
        d = ipr - ie
        rel = np.abs(d) / np.maximum(ie, 50.0)
        out = {}
        for cls in ("all", "short", "long"):
            cm = np.ones(len(ie), bool) if cls == "all" else (rate[fr["shot"]] == cls)
            for p in ("all", "ramp_up", "flat", "ramp_down"):
                m = cm & ((ph == p) if p != "all" else True) & np.isfinite(d)
                if not m.any():
                    continue
                out[f"{cls}|{p}"] = {"n": int(m.sum()),
                                     "rms_kA": round(float(np.sqrt((d[m] ** 2).mean())), 2),
                                     "medabs_kA": round(float(np.median(np.abs(d[m]))), 2),
                                     "frac_rel_gt5pct": round(float((rel[m] > 0.05).mean()), 4),
                                     "ratio_p95": round(float(np.percentile(ie[m] / np.maximum(ipr[m], 1.0), 95)), 4)}
        rep[rule] = out
    (OUT / "frame_audit.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ============================================================================ VAL comparison
def valcmp(tags, rule_phase="idx", nproc=6, official=True):
    """Final comparison only (VAL labels): models' saved VAL predictions, overall and by shot
    class (magnetics rate) and phase (phase from the corrected INPUT Ip, rule `rule_phase`)."""
    from fec.features import load_ds
    ds = load_ds(CONFIG)
    info = ds["info"]
    rows = json.loads((OUT / "shots.json").read_text())
    rate = np.array([record_class(r) for r in rows])
    res = {}
    base_idx = None
    for tag in tags:
        m = WORK / "models" / tag
        idx = np.load(m / "val_idx.npy")
        if base_idx is None:
            base_idx = idx
            sp = np.array(info["splits"])[ds["shot"][idx]]
            assert set(sp) == {"val"}
            shots = ds["shot"][idx]
            gt = np.asarray(ds["psi"][idx], dtype=np.float32).reshape(len(idx), -1)
            gsc = ds["scal"][idx].astype(np.float64)
            ipc = np.abs(np.load(OUT / f"ipcur_{rule_phase}.npy", mmap_mode="r")[idx, 6])
            ph = np.full(len(idx), "flat", dtype=object)
            for s in np.unique(shots):
                ix = np.flatnonzero(shots == s)
                top = np.flatnonzero(ipc[ix] >= 0.9 * np.nanmax(ipc[ix]))
                ph[ix[:top[0]]] = "ramp_up"
                ph[ix[top[-1] + 1:]] = "ramp_down"
        assert np.array_equal(idx, base_idx), tag
        pv = np.load(m / "val_psi.npy", mmap_mode="r")
        P = np.asarray(pv, dtype=np.float32).reshape(len(idx), -1)
        sc = np.load(m / "val_scal.npy").astype(np.float64)
        e2 = ((P - gt) ** 2).mean(1)
        out = {}
        for cls in ("all", "short", "long"):
            cm = np.ones(len(idx), bool) if cls == "all" else (rate[shots] == cls)
            for p in ("all", "ramp_up", "flat", "ramp_down"):
                mm = cm & ((ph == p) if p != "all" else True)
                if not mm.any():
                    continue
                g = gt[mm]
                r2 = 1 - e2[mm].sum() * g.shape[1] / ((g - g.mean()) ** 2).sum()
                q = {}
                for j, nm in ((0, "q95"), (1, "betan")):
                    y, yh = gsc[mm, j], sc[mm, j]
                    ok = np.isfinite(y)
                    q[f"r2_{nm}"] = round(float(1 - ((yh[ok] - y[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()), 6)
                    q[f"mae_{nm}"] = round(float(np.abs(yh[ok] - y[ok]).mean()), 5)
                out[f"{cls}|{p}"] = dict(n=int(mm.sum()), n_shots=int(len(np.unique(shots[mm]))),
                                         psi_rmse_mWb=round(float(np.sqrt(e2[mm].mean()) * 1e3), 4),
                                         r2_psi=round(float(r2), 7), **q)
        if official:
            from fec import score as SC
            order = info["order"]
            for cls in ("all", "short", "long"):
                us = [s for s in np.unique(shots) if cls == "all" or rate[s] == cls]
                keys, gts, preds = [], [], []
                for s in us:
                    j = np.flatnonzero(shots == s)
                    g = idx[j]
                    keys.append(order[s])
                    gts.append({"psi": np.asarray(ds["psi"][g], dtype=np.float32),
                                "q95": ds["scal"][g, 0].astype(np.float64),
                                "betaN": ds["scal"][g, 1].astype(np.float64)})
                    preds.append({"psirz": np.asarray(pv[j], dtype=np.float16).astype(np.float64),
                                  "q95": sc[j, 0], "betaN": sc[j, 1]})
                r = SC.score(keys, gts, preds, nproc=nproc, verbose=False)
                out[f"official|{cls}"] = {k: r[k] for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")}
                out[f"official|{cls}"]["n_shots"] = len(us)
        res[tag] = out
        print(tag, json.dumps({k: v for k, v in out.items() if k.startswith("official") or k.endswith("|all")}), flush=True)
    f = OUT / ("valcmp_" + "__".join(tags) + ".json")
    f.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", f)


def pershot(pairs):
    """Final comparison only (VAL labels): per-shot paired psi MSE of model pairs 'ctrl:treat'
    by record class -> fraction of shots improved, median and pooled MSE ratio."""
    from fec.features import load_ds
    ds = load_ds(CONFIG)
    rows = json.loads((OUT / "shots.json").read_text())
    cls = np.array([record_class(r) for r in rows])
    res = {}
    gt = None
    for pair in pairs:
        a, b = pair.split(":")
        idx = np.load(WORK / "models" / a / "val_idx.npy")
        assert np.array_equal(idx, np.load(WORK / "models" / b / "val_idx.npy"))
        if gt is None:
            assert set(np.array(ds["info"]["splits"])[ds["shot"][idx]]) == {"val"}
            gt = np.asarray(ds["psi"][idx], dtype=np.float32).reshape(len(idx), -1)
            shots = ds["shot"][idx]
            us = np.unique(shots)
        e = {}
        for t in (a, b):
            P = np.asarray(np.load(WORK / "models" / t / "val_psi.npy", mmap_mode="r"),
                           dtype=np.float32).reshape(len(idx), -1)
            fe = ((P - gt) ** 2).mean(1)
            e[t] = np.array([fe[shots == s].mean() for s in us])
        out = {}
        for c in ("all", "short", "long"):
            m = np.ones(len(us), bool) if c == "all" else (cls[us] == c)
            r = e[b][m] / e[a][m]
            out[c] = {"n_shots": int(m.sum()), "frac_improved": round(float((r < 1).mean()), 4),
                      "median_ratio": round(float(np.median(r)), 4),
                      "pooled_ratio": round(float(e[b][m].sum() / e[a][m].sum()), 4)}
        res[pair] = out
        print(pair, json.dumps(out), flush=True)
    (OUT / "pershot.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--workers", type=int, default=6)
    s.add_argument("--limit", type=int, default=0); s.add_argument("--config", default=CONFIG)
    sub.add_parser("report")
    f = sub.add_parser("features"); f.add_argument("--rule", default="idx", choices=RULES)
    f.add_argument("--workers", type=int, default=6); f.add_argument("--config", default=CONFIG)
    c = sub.add_parser("check"); c.add_argument("--config", default=CONFIG)
    fa = sub.add_parser("audit"); fa.add_argument("--rules", default="starter,idx,ecoil")
    v = sub.add_parser("valcmp"); v.add_argument("--tags", required=True)
    v.add_argument("--phase_rule", default="idx"); v.add_argument("--nproc", type=int, default=6)
    v.add_argument("--no_official", action="store_true")
    ps = sub.add_parser("pershot"); ps.add_argument("--pairs", required=True)
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(a.workers, a.limit, a.config)
    elif a.cmd == "report":
        report()
    elif a.cmd == "check":
        sys.exit(1 if check(a.config) else 0)
    elif a.cmd == "audit":
        frame_audit(tuple(a.rules.split(",")))
    elif a.cmd == "pershot":
        pershot(a.pairs.split(","))
    elif a.cmd == "valcmp":
        valcmp(a.tags.split(","), a.phase_rule, a.nproc, not a.no_official)
    elif a.cmd == "features":
        features(a.rule, a.workers, a.config)


if __name__ == "__main__":
    main()
