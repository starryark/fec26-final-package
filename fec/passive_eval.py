"""Label-free evaluation of MAST passive-current models (fec/passive.py) on real MAST inputs.

Re-runs the fec/mast_v1 stage-1 selection logic (copied, not imported: force-free vertical
secant, 8-shape COMBOS family, inboard/outboard Thomson Te flux-function mismatch from
fec/kinetic) on a fixed random subset of mast_public_test frames, with and without the
passive-structure flux added as extra external 'coil' columns.  Inputs only: coil currents,
Ip, Thomson Te on the Z=0 chord.  No MAST labels, demo parquet or scorer masks are read.

CLI:
    python -B fec/passive_eval.py run --variants none,c1_r1,c5_r1,c1_r0.5,c1_r2,c1_r1_filament
    python -B fec/passive_eval.py report
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

# copied from fec/mast_v1.py (2026-10-07) so that this study does not depend on later edits
COMBOS = [(0.05, 2, 1), (0.05, 1, 2), (0.05, 2, 2), (0.05, 3, 1), (0.3, 2, 1), (0.05, 1, 3),
          (0.3, 1, 2), (0.3, 1, 3)]
DEFAULT = COMBOS.index((0.05, 2, 2))
PEAKED = [COMBOS.index((0.05, 1, 2)), COMBOS.index((0.05, 1, 3))]
CONFIG = "mast_public_test"
OUT = WORK / "passive" / "eval"
SEED = 20261007
N_FRAMES = 1000


def thomson_frame(z, k):
    R = np.concatenate([z["ts_core_R"], z["ts_edge_R"]])
    te = np.concatenate([z["ts_core_te"][k], z["ts_edge_te"][k]]).astype(float)
    ne = np.concatenate([z["ts_core_ne"][k], z["ts_edge_ne"][k]]).astype(float)
    return R, te, ne


def te_peak_R(R, te):
    """Robust Te-peak radius: Te-weighted centroid of the top-20% (median-filtered) points."""
    from scipy.ndimage import median_filter
    ok = np.isfinite(R) & np.isfinite(te) & (te > 0)
    if ok.sum() < 20:
        return np.nan
    o = np.argsort(R[ok])
    r, v = R[ok][o], median_filter(te[ok][o], size=5, mode="nearest")
    top = v >= 0.8 * v.max()
    return float(np.sum(r[top] * v[top]) / np.sum(v[top]))


def frame_subset(n=N_FRAMES, seed=SEED):
    from fec.features import load_ds
    ds = load_ds(CONFIG)
    N = len(ds["t_ms"])
    idx = np.sort(np.random.default_rng(seed).choice(N, size=n, replace=False))
    return idx, ds


def passive_currents(variant, idx):
    if variant == "none":
        return None
    I = np.load(WORK / "passive" / "frames" / CONFIG / variant / "I.npy", mmap_mode="r")
    return np.asarray(I[idx], dtype=np.float64)


def run_variant(variant, idx, ds, chunk=500, ts=None, keep_psi=False, save_as=None):
    """mast_v1 stage-1 selection on frames idx with passive variant columns.

    ts: optional list of (R, te, ne) per frame (synthetic Thomson for twins); keep_psi stores
    the selected psi maps; save_as overrides the output name (OUT/<name>.npz).
    """
    import torch
    from fec.gs import Solver
    from fec.kinetic import chord_psiN, symmetry_mismatch
    from fec.machines import MAST_CH, MAST_COILS, mast_setup
    from fec.passive import PassiveModel
    info = ds["info"]
    cur = np.asarray(ds["cur"], dtype=np.float64)[idx][:, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    spec, M = mast_setup()
    Ipas = passive_currents(variant, idx)
    if Ipas is not None:
        pm = PassiveModel(plasma="filament" if variant.endswith("_filament") else "ellipse")
        M = pm.augment(M)
    S = Solver(spec, M)
    B_all = len(idx)
    res = {k: np.full(B_all, np.nan) for k in ("psi_ax", "psi_b", "Rax", "Zax", "Zt", "c",
                                              "combo", "mm", "mm_default", "dI", "te_peak_R")}
    res["mm_all"] = np.full((B_all, len(COMBOS)), np.nan)
    res["Rax_all"] = np.full((B_all, len(COMBOS)), np.nan)
    cache = {}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for s0 in range(0, B_all, chunk):
        sl = slice(s0, min(B_all, s0 + chunk))
        B = sl.stop - sl.start
        I = cur[sl][:, ci] * 1e3
        if Ipas is not None:
            I = np.concatenate([I, Ipas[sl]], 1)
        Ip = np.maximum(cur[sl, MAST_CH.index("magnetics_plasma_current")] * 1e3, 5e4)
        prof0 = np.tile(COMBOS[DEFAULT], (B, 1))
        o0 = S.solve(I, Ip, prof0, np.zeros(B), n_iter=150, Rinit=np.full(B, 0.9))
        o1 = S.solve(I, Ip, prof0, np.full(B, 0.05), n_iter=60, I0=o0["I"], c0=o0["c"])
        c0, c1 = o0["c"].cpu().numpy(), o1["c"].cpu().numpy()
        den = np.where(np.abs(c1 - c0) > 1e-12, c1 - c0, np.nan)
        Zs = np.nan_to_num(np.clip(-c0 * 0.05 / den, -0.3, 0.3))
        TS = []
        if ts is not None:
            TS = list(ts[sl.start:sl.stop])
        for g in ([] if ts is not None else idx[sl]):
            stem = info["order"][ds["shot"][g]]
            if stem not in cache:
                cache.clear()
                cache[stem] = dict(np.load(WORK / "shots" / CONFIG / f"{stem}.npz"))
            TS.append(thomson_frame(cache[stem], ds["fidx"][g]))
        best_mm = np.full(B, np.inf); best = np.full(B, DEFAULT)
        sel = {k: np.full(B, np.nan) for k in ("psi_ax", "psi_b", "Rax", "Zax", "c", "dI")}
        dflt = None
        best_psi = dflt_psi = None
        for cidx, combo in enumerate(COMBOS):
            o = S.solve(I, Ip, np.tile(combo, (B, 1)), Zs, n_iter=100, I0=o0["I"], c0=o0["c"])
            psi = o["psi"].cpu().numpy(); pa = o["psi_ax"].cpu().numpy()
            pb = o["psi_b"].cpu().numpy(); ra = o["Rax"].cpu().numpy()
            za = o["Zax"].cpu().numpy(); dI = o["dI"].cpu().numpy(); cc = o["c"].cpu().numpy()
            mm = np.full(B, np.nan)
            for i in range(B):
                R, te, _ = TS[i]
                pn = chord_psiN(psi[i], pa[i], pb[i], spec.gR, spec.gZ, R, np.zeros_like(R))
                mm[i] = symmetry_mismatch(pn, R, te, ra[i])[0]
            mm = np.where(dI < 1e-2, mm, np.nan)   # unconverged frames cannot win
            res["mm_all"][sl, cidx] = mm
            res["Rax_all"][sl, cidx] = ra
            vals = dict(psi_ax=pa, psi_b=pb, Rax=ra, Zax=za, c=cc, dI=dI)
            if cidx == DEFAULT:
                dflt = vals
            better = np.isfinite(mm) & (mm < best_mm)
            if cidx == 0:
                better = np.ones(B, bool)
                best_mm = np.where(np.isfinite(mm), mm, np.inf)
            else:
                best_mm = np.where(better, mm, best_mm)
            best = np.where(better, cidx, best)
            for k in sel:
                sel[k] = np.where(better, vals[k], sel[k])
            if keep_psi:
                best_psi = psi if best_psi is None else np.where(better[:, None, None], psi,
                                                                 best_psi)
                if cidx == DEFAULT:
                    dflt_psi = psi
        nofit = ~np.isfinite(best_mm)
        best[nofit] = DEFAULT
        for k in sel:
            sel[k] = np.where(nofit, dflt[k], sel[k])
        if keep_psi:
            best_psi = np.where(nofit[:, None, None], dflt_psi, best_psi)
            res.setdefault("psi", np.full((B_all,) + best_psi.shape[1:], np.nan,
                                          dtype=np.float32))[sl] = best_psi
        for k, v in sel.items():
            res[k][sl] = v
        res["Zt"][sl] = Zs
        res["combo"][sl] = best
        res["mm"][sl] = np.where(nofit, np.nan, best_mm)
        res["mm_default"][sl] = res["mm_all"][sl, DEFAULT]
        res["te_peak_R"][sl] = [te_peak_R(R, te) for (R, te, _) in TS]
        print(f"[{variant}] {sl.stop}/{B_all} {time.time() - t0:.0f}s combos "
              f"{np.bincount(best, minlength=len(COMBOS)).tolist()} mm med "
              f"{np.nanmedian(res['mm'][:sl.stop]):.4f} peakMB "
              f"{torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)
        torch.cuda.empty_cache()
    res["idx"] = idx
    res["seconds"] = np.array(time.time() - t0)
    f = OUT / f"{save_as or variant}.npz"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **res)
    return res


# ----------------------------------------------------------------------------- twin
def make_truth(variant, idx, ds, combo_idx):
    """Synthetic truth equilibria: real currents (+ passive variant), random in-family shape."""
    import torch
    from fec.gs import Solver
    from fec.machines import MAST_CH, MAST_COILS, mast_setup
    from fec.passive import PassiveModel
    cur = np.asarray(ds["cur"], dtype=np.float64)[idx][:, :, 6]
    I = cur[:, [MAST_CH.index(c) for c in MAST_COILS]] * 1e3
    Ipas = passive_currents(variant, idx)
    spec, M = mast_setup()
    if Ipas is not None:
        M = PassiveModel().augment(M)
        I = np.concatenate([I, Ipas], 1)
    S = Solver(spec, M)
    B = len(idx)
    Ip = np.maximum(cur[:, MAST_CH.index("magnetics_plasma_current")] * 1e3, 5e4)
    prof0 = np.tile(COMBOS[DEFAULT], (B, 1))
    o0 = S.solve(I, Ip, prof0, np.zeros(B), n_iter=150, Rinit=np.full(B, 0.9))
    o1 = S.solve(I, Ip, prof0, np.full(B, 0.05), n_iter=60, I0=o0["I"], c0=o0["c"])
    c0, c1 = o0["c"].cpu().numpy(), o1["c"].cpu().numpy()
    den = np.where(np.abs(c1 - c0) > 1e-12, c1 - c0, np.nan)
    Zs = np.nan_to_num(np.clip(-c0 * 0.05 / den, -0.3, 0.3))
    prof = np.array([COMBOS[k] for k in combo_idx], dtype=float)
    o = S.solve(I, Ip, prof, Zs, n_iter=200, I0=o0["I"], c0=o0["c"])
    out = {k: o[k].cpu().numpy() for k in ("psi", "psi_ax", "psi_b", "Rax", "Zax", "dI")}
    out["valid"] = (out["dI"] < 1e-2) & np.isfinite(out["psi_b"]) & (out["psi_ax"] > out["psi_b"])
    out["combo"] = np.asarray(combo_idx)
    torch.cuda.empty_cache()
    return out, spec


def synth_thomson(truth, spec, real_ts, seed):
    """Te = 1 keV (1 - psiN^1.3)^1.6 (10 eV outside) of the truth on the real chord radii,
    with the real missing-channel pattern and 3% noise; ne mirrors Te (unused)."""
    from fec.kinetic import chord_psiN
    rng = np.random.default_rng(seed)
    out = []
    for i, (R, te, _) in enumerate(real_ts):
        pn = chord_psiN(truth["psi"][i], truth["psi_ax"][i], truth["psi_b"][i], spec.gR,
                        spec.gZ, R, np.zeros_like(R))
        v = np.where(pn < 1, 1e3 * np.clip(1 - np.clip(pn, 0, 1) ** 1.3, 0, 1) ** 1.6, 10.0)
        v = v * (1 + 0.03 * rng.standard_normal(v.size))
        v[~(np.isfinite(te) & (te > 0))] = np.nan
        out.append((R, v, v.copy()))
    return out


def twin_metrics(truth, rec, spec):
    RR, ZZ = np.meshgrid(spec.gR, spec.gZ)
    box = (RR > 0.25) & (RR < 1.45) & (np.abs(ZZ) < 1.2)
    ok = truth["valid"]
    span = truth["psi_ax"] - truth["psi_b"]
    pnt = (truth["psi_ax"][:, None, None] - truth["psi"]) / span[:, None, None]
    pnr = (rec["psi_ax"][:, None, None] - rec["psi"]) / (rec["psi_ax"] - rec["psi_b"])[:, None, None]
    inside = (pnt < 1) & box[None]
    e_pn = np.array([np.sqrt(np.mean((pnr[i] - pnt[i])[inside[i]] ** 2)) if inside[i].any()
                     else np.nan for i in range(len(ok))])
    e_psi = np.sqrt(np.mean(((rec["psi"] - truth["psi"])[:, box]) ** 2, 1)) / span
    return dict(n=int(ok.sum()), combo_acc=float(np.mean(rec["combo"][ok] == truth["combo"][ok])),
                psiN_rms=float(np.nanmedian(e_pn[ok])), psi_rms_rel=float(np.nanmedian(e_psi[ok])),
                dRax=float(np.nanmedian(np.abs(rec["Rax"] - truth["Rax"])[ok])),
                peaked=float(np.mean(np.isin(rec["combo"][ok].astype(int), PEAKED))))


def twin(n=400, seed=SEED + 1, plans=(("c1_r1", ("none", "c1_r1", "c1_r2", "c5_r1")),
                                      ("none", ("none", "c1_r1")))):
    idx_all, ds = frame_subset()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(idx_all, n, replace=False))
    info = ds["info"]
    real_ts, cache = [], {}
    for g in idx:
        stem = info["order"][ds["shot"][g]]
        if stem not in cache:
            cache.clear()
            cache[stem] = dict(np.load(WORK / "shots" / CONFIG / f"{stem}.npz"))
        real_ts.append(thomson_frame(cache[stem], ds["fidx"][g]))
    combo_idx = rng.integers(0, len(COMBOS), n)
    rows = []
    (OUT / "twin").mkdir(parents=True, exist_ok=True)
    for tv, recs in plans:
        t0 = time.time()
        truth, spec = make_truth(tv, idx, ds, combo_idx)
        np.savez(OUT / "twin" / f"truth_{tv}.npz", idx=idx, **truth)
        print(f"[twin truth {tv}] valid {truth['valid'].sum()}/{n} {time.time() - t0:.0f}s",
              flush=True)
        ts = synth_thomson(truth, spec, real_ts, seed)
        for v in recs:
            rec = run_variant(v, idx, ds, chunk=n, ts=ts, keep_psi=True,
                              save_as=f"twin/truth_{tv}__rec_{v}")
            m = twin_metrics(truth, rec, spec)
            m.update(truth=tv, recon=v)
            rows.append(m)
            print("[twin]", json.dumps(m), flush=True)
    (OUT / "twin" / "twin_metrics.json").write_text(json.dumps(rows, indent=1))
    return rows


def xpoint_check():
    """CPU: X-points of the two twin truths (same frames/shapes, with vs without C1 flux)."""
    import torch
    from fec.gs import Solver
    from fec.machines import mast_setup
    spec, M = mast_setup()
    S = Solver(spec, M, device="cpu")
    out = {}
    for tv in ("none", "c1_r1"):
        z = dict(np.load(OUT / "twin" / f"truth_{tv}.npz"))
        psi = torch.as_tensor(z["psi"], dtype=torch.float32)
        pa, Ra, Za = S.find_axis(psi)
        (fu, vu, Ru, Zu), (fl, vl, Rl, Zl) = S.find_xpoints(psi, Ra, Za, pa)
        lim_up = (vu >= vl) & fu
        Rx = torch.where(lim_up, Ru, Rl).numpy(); Zx = torch.where(lim_up, Zu, Zl).numpy()
        out[tv] = dict(Rx=Rx, Zx=Zx, valid=z["valid"], found=(fu | fl).numpy())
    ok = out["none"]["valid"] & out["c1_r1"]["valid"]
    d = np.hypot(out["c1_r1"]["Rx"] - out["none"]["Rx"], out["c1_r1"]["Zx"] - out["none"]["Zx"])
    near_arm = lambda o: np.mean((np.abs(o["Zx"][ok]) > 1.5) & (o["Rx"][ok] < 0.75))
    res = dict(n=int(ok.sum()), median_shift_m=float(np.median(d[ok])),
               frac_shift_gt_5cm=float(np.mean(d[ok] > 0.05)),
               absZx_median_none=float(np.median(np.abs(out["none"]["Zx"][ok]))),
               absZx_median_c1=float(np.median(np.abs(out["c1_r1"]["Zx"][ok]))),
               frac_near_P2_structures_none=float(near_arm(out["none"])),
               frac_near_P2_structures_c1=float(near_arm(out["c1_r1"])))
    (OUT / "twin" / "xpoint_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return res


# ----------------------------------------------------------------------------- report
def _phase(ds, idx=None):
    from fec.machines import MAST_CH
    cur = np.asarray(ds["cur"], dtype=np.float64)
    if idx is not None:
        cur = cur[idx]
    offs = ds["info"]["offsets_ms"]
    ip = MAST_CH.index("magnetics_plasma_current")
    Ip = cur[:, ip, offs.index(0.0)]
    dIp = (cur[:, ip, offs.index(5.0)] - cur[:, ip, offs.index(-5.0)]) / 10.0  # kA/ms = MA/s
    ph = np.full(len(Ip), "flat", dtype=object)
    ph[dIp > 1.5] = "ramp_up"
    ph[dIp < -1.5] = "ramp_down"
    return ph, Ip, dIp


def report(variants=None):
    from fec.features import load_ds
    from fec.green import greens_BrBz
    from fec.passive import PassiveModel
    ds = load_ds(CONFIG)
    variants = variants or [p.stem for p in sorted(OUT.glob("*.npz"))]
    base = dict(np.load(OUT / "none.npz")) if (OUT / "none.npz").exists() else None
    lines = []
    P = lambda s: (print(s), lines.append(s))
    P("## Selection on the evaluation subset (paired with 'none')")
    P("| variant | n finite | median best mm | paired median (none -> var) | frac improved | "
      "median mm default | peaked (0.05,1,2)+(0.05,1,3) | combo counts | Rax-RTe med / |med| [m] |")
    P("|---|---|---|---|---|---|---|---|---|")
    for v in variants:
        r = dict(np.load(OUT / f"{v}.npz", allow_pickle=True))
        mm = r["mm"]; fin = np.isfinite(mm)
        combo = r["combo"].astype(int)
        cnt = np.bincount(combo[fin], minlength=len(COMBOS))
        peaked = np.isin(combo[fin], PEAKED).mean()
        d = r["Rax"] - r["te_peak_R"]
        ok = np.isfinite(d) & fin
        if base is not None:
            both = fin & np.isfinite(base["mm"])
            pm = f"{np.median(base['mm'][both]):.4f} -> {np.median(mm[both]):.4f} (n={both.sum()})"
            imp = f"{np.mean(mm[both] < base['mm'][both]):.2f}"
        else:
            pm, imp = "-", "-"
        P(f"| {v} | {fin.sum()} | {np.median(mm[fin]):.4f} | {pm} | {imp} | "
          f"{np.nanmedian(r['mm_default']):.4f} | {peaked:.2f} | {cnt.tolist()} | "
          f"{np.median(d[ok]):+.3f} / {np.median(np.abs(d[ok])):.3f} |")
    P("")
    P("## Passive totals vs Ip (all 69,075 frames; kA, median [p10, p90])")
    ph, Ip, dIp = _phase(ds)
    P("| variant | phase | n | total passive | total/Ip | casing | outer wall | end plates | "
      "coil cases | Bz_pas(0.9,0) [mT] |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    fdir = WORK / "passive" / "frames" / CONFIG
    for v in variants:
        if v == "none" or not (fdir / v / "totals.npz").exists():
            continue
        t = dict(np.load(fdir / v / "totals.npz"))
        pm = PassiveModel(plasma="filament" if v.endswith("_filament") else "ellipse")
        Br, Bz = greens_BrBz(0.9, 0.0, pm.Rc, pm.Zc)
        I = np.load(fdir / v / "I.npy", mmap_mode="r")
        bz = np.concatenate([np.asarray(I[s:s + 8192], dtype=np.float64) @ Bz
                             for s in range(0, I.shape[0], 8192)]) * 1e3
        tot = t["total"] / 1e3
        cas = sum(t[k] for k in t if k.startswith("cc_casing")) / 1e3
        ow = sum(t[k] for k in t if k.startswith("outer_wall")) / 1e3
        ep = sum(t[k] for k in t if k.startswith("end_plate")) / 1e3
        cs = sum(t[k] for k in t if k.startswith("case_")) / 1e3
        q = lambda x: f"{np.median(x):.1f} [{np.percentile(x, 10):.1f}, {np.percentile(x, 90):.1f}]"
        for p in ("ramp_up", "flat", "ramp_down"):
            m = (ph == p) & np.isfinite(Ip) & (Ip > 50)
            if not m.any():
                continue
            P(f"| {v} | {p} | {m.sum()} | {q(tot[m])} | {np.median(tot[m] / Ip[m]):.3f} | "
              f"{q(cas[m])} | {q(ow[m])} | {q(ep[m])} | {q(cs[m])} | {q(bz[m])} |")
    txt = "\n".join(lines)
    (OUT / "report.md").write_text(txt + "\n")
    return txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "twin", "xcheck"])
    ap.add_argument("--variants", default="none,c1_r1,c5_r1,c1_r0.5,c1_r2,c1_r1_filament")
    ap.add_argument("--n", type=int, default=N_FRAMES)
    ap.add_argument("--chunk", type=int, default=500)
    a = ap.parse_args()
    if a.cmd == "run":
        idx, ds = frame_subset(a.n)
        for v in a.variants.split(","):
            run_variant(v, idx, ds, chunk=a.chunk)
    elif a.cmd == "twin":
        twin(n=a.n)
    elif a.cmd == "xcheck":
        xpoint_check()
    else:
        report(a.variants.split(",") if a.variants else None)
