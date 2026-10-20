"""Final MAST composite (agent fpkg 2026-10-12, docs/archive/fec/FINAL_PACKAGE_20261012.md).

The Final MAST half (Dev MAST 0.796343, sub46w / C25) is a composite of three label-free parts,
each run here with the MAST config as a parameter (mast_public_test | mast_private_test):

  maps     amortized v7h network (fec/amort_net2.py, trained on our own synthetic GS equilibria
           at solenoid turn fraction f) on every real frame, then within-shot Gaussian smoothing
           sigma 30 ms.  Exact ports of fec/amort_gen6.passive_frames (C1 passive currents at f),
           fec/amort_real.run(v2=True) and fec/mast_tsm.smooth_maps (psi part), whose originals
           hard-code mast_public_test;
  scalars  q95 (and the stage-2 keys) of the physics average avg_f05 = equal-weight mean, in this
           order, of zsc C1, zsc no-passive, the deep-crossing merge of zsc C1 with its
           centre-column-limiter twin, zfix 0 and zsc C1 tsm30 (fec_work/mastz/cand_avg_f05),
           built with fec/mast_pipeline.py's Plan (its JSON recipes cannot average a post node);
  betaN    the frozen lam_fgw model on the avg_f05 records, shot-filled with the fill3 rule (a
           NaN frame takes its shot's median only if the shot has >= 3 finite frames, else the
           fold median; then the 5-frame running median and clip [0, 8]).  Port of
           fec_work/scratch/bnmap_fillfix.py (POST-HOC label-free hygiene, C14).

    python -B fec/final_mast.py passive_f --config C --f 0.55 --out DIR [--rscales 1]
    python -B fec/final_mast.py amort     --config C --net DIR --f 0.55 --passive DIR --out DIR
                                          [--device cuda] [--n N]
    python -B fec/final_mast.py tsm       --config C --src DIR --out DIR --sigma-ms 30 [--device cuda]
    python -B fec/final_mast.py physics   --config C --out ROOT [--limit N --limit-align chunk]
                                          [--owner O --gpu-priority P --workers W --jobs J]
    python -B fec/final_mast.py composite --config C --maps DIR --scalars DIR --out DIR
                                          [--min-finite 3]
  Oct 14 map variants (agent fpkgm; docs/archive/fec/FINAL_PKG_MAST_PRESETS_20261013.md):
    python -B fec/final_mast.py amort ... --rscale 0.5        (r05: passive DIR/c1_r0.5)
    python -B fec/final_mast.py blend --a RAW_V7H --b RAW_TX1S --out DIR    (VB raw; then tsm)
    python -B fec/final_mast.py combo --config C --vb DIR --r05 DIR --s4 DIR --out DIR [--device]
  The TX1s maps themselves come from fec/final_tx1s.py infer (kept out of this module's imports).

Inputs only: coil currents, Ip, Thomson, C1 passive currents (from the inputs).  No MAST label,
demo parquet or scorer mask is read.  The solenoid fraction f = 0.55 is the user-approved Dev
exception (AGENTS Rule 3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

CONFIGS = ("mast_public_test", "mast_private_test")
NG = 65
AVG_F05_MEMBERS = ("zsc_c1", "zsc_nopas", "dc", "zfix0_c1", "zsc_c1_tsm30")
MIN_FINITE = 3


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist()
                                     if hasattr(o, "tolist") else str(o)), encoding="utf-8")


def set_f(f):
    """Solenoid turn fraction of the amortized family (read by fec.amort_net.coil_matrix)."""
    os.environ["FEC_AMORT_SOLF"] = f"{float(f):g}"
    from fec.amort_net import sol_f
    if abs(sol_f() - float(f)) > 1e-12:
        raise RuntimeError(f"FEC_AMORT_SOLF={os.environ['FEC_AMORT_SOLF']} != f {f}")
    return float(f)


# ============================================================================ passive at f
def passive_frames(config, f, out, rscales=("1",), log=print):
    """C1 per-frame passive currents at solenoid fraction f for one config: exactly
    fec.amort_gen6.passive_frames (solenoid drive column of the f = 0.5 model scaled by f/0.5,
    fec.passive.c1_shot integration, linear interpolation to the frame times) with the config
    as a parameter.  Writes out/c1_r<rs>/I.npy ((N, Np) float32, ds frame order)."""
    from fec.machines import MAST_COILS
    from fec.passive import PassiveModel, c1_shot, frame_index
    model = PassiveModel()
    isol = MAST_COILS.index("magnetics_sol_current")
    model.Mpc = model.Mpc.copy()
    model.Mpc[:, isol] *= float(f) / 0.5
    per, N, _ = frame_index(config)
    sd = WORK / "passive" / "sig" / config
    t0 = time.time()
    done = {}
    for rs in rscales:
        od = Path(out) / f"c1_r{rs}"
        if (od / "I.npy").exists():
            done[rs] = "exists"
            continue
        od.mkdir(parents=True, exist_ok=True)
        I_out = np.lib.format.open_memmap(od / "I.part.npy", "w+", np.float32,
                                          (N, len(model.piece)))
        modes = model.modes(float(rs))
        for stem, frl in per.items():
            z = np.load(sd / f"{stem}.npz")
            t, I = c1_shot(model, z, float(rs), modes)
            gg = np.array([a for a, _ in frl]); tf = np.array([b for _, b in frl])
            for j in range(I.shape[1]):
                I_out[gg, j] = np.interp(tf, t, I[:, j])
        I_out.flush(); del I_out
        (od / "I.part.npy").replace(od / "I.npy")
        done[rs] = "made"
        log(f"passive f={f:g} r={rs} {config} ({time.time() - t0:.0f}s)")
    info = dict(kind="C1 passive currents at solenoid fraction f", config=config, f=float(f),
                N=int(N), rscales=list(rscales), status=done,
                files={rs: sha256(Path(out) / f"c1_r{rs}" / "I.npy") for rs in rscales},
                code="fec/final_mast.py passive_f (port of fec/amort_gen6.passive_frames)")
    _write_json(Path(out) / "config.json", info)
    return info


# ============================================================================ amortized maps
def frame_inputs(g, ds, config):
    """fec.amort_gen.frame_inputs with the shots cache of `config` (the original reads
    shots/mast_public_test)."""
    from fec.machines import MAST_CH, MAST_COILS
    from fec.mast_v1 import thomson_frame
    info, shot = ds["info"], ds["shot"]
    cur = np.asarray(ds["cur"][g], dtype=np.float64)[:, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    Rl, tel, nel = [], [], []
    cache = {}
    for k in g:
        stem = info["order"][shot[k]]
        if stem not in cache:
            cache.clear()
            cache[stem] = dict(np.load(WORK / "shots" / config / f"{stem}.npz"))
        R, te, ne = thomson_frame(cache[stem], ds["fidx"][k])
        Rl.append(R); tel.append(te); nel.append(ne)
    nch = max(len(r) for r in Rl)
    pad = lambda L: np.stack([np.pad(np.asarray(x, float), (0, nch - len(x)),  # noqa: E731
                                     constant_values=np.nan) for x in L])
    return dict(g=np.asarray(g), I=cur[:, ci] * 1e3,
                Ip=np.maximum(cur[:, MAST_CH.index("magnetics_plasma_current")] * 1e3, 5e4),
                R=pad(Rl), te_real=pad(tel), ne_real=pad(nel))


def causal_state(g, ds):
    """fec.amort_gen2.causal_state on a given ds (the original loads mast_public_test)."""
    from fec.machines import MAST_CH, MAST_COILS
    offs = ds["info"]["offsets_ms"]
    k0, k5, k20 = offs.index(0.0), offs.index(-5.0), offs.index(-20.0)
    cur = np.asarray(ds["cur"][np.asarray(g)], np.float64)
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    c0, c5, c20 = (np.nan_to_num(cur[:, ci, k]) for k in (k0, k5, k20))
    u, l = MAST_COILS.index("magnetics_p6u_current"), MAST_COILS.index("magnetics_p6l_current")
    p6 = lambda c: c[:, u] - c[:, l]  # noqa: E731
    return np.concatenate([(c0 - c5) / 5.0, (c0 - c20) / 20.0,
                           np.stack([p6(c0), (p6(c0) - p6(c5)) / 5.0,
                                     (p6(c0) - p6(c20)) / 20.0], 1)], 1)


def load_net(net_dir, device="cuda"):
    """fec.amort_net2.load2 from an explicit model dir (model.pt + basis.npz)."""
    import torch
    from fec.amort_net import build_mlp
    d = Path(net_dir)
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


def known_psi(I, g, M, Gg, I_pass, bs=4096):
    """psi_coil(I) + psi_passive(C1 at f, r = 1) (fec.amort_net2.known_psi, explicit passive)."""
    out = np.empty((len(g), NG, NG))
    for i in range(0, len(g), bs):
        gi = np.asarray(g[i:i + bs])
        out[i:i + bs] = ((np.asarray(I[i:i + bs], float) @ M.T) +
                         np.asarray(I_pass[gi], float) @ Gg.T).reshape(-1, NG, NG)
    return out


def predict_psi(model, dev, M, Gg, I_pass, g, I, Ip, R, te, ne, ds, bs=4096):
    """fec.amort_net2.predict2's psi (the axis/boundary part is not needed here)."""
    import torch
    from fec.amort_net2 import features2
    nets, mu, U, st, cfg = model
    X = features2(I, Ip, R, te, ne, causal_state(g, ds), st)
    kn = known_psi(I, g, M, Gg, I_pass)
    psi = np.empty((len(Ip), NG, NG), np.float32)
    Ut = torch.as_tensor(U, device=dev); mut = torch.as_tensor(mu, device=dev)
    with torch.no_grad():
        for i in range(0, len(Ip), bs):
            x = torch.as_tensor(X[i:i + bs], device=dev)
            c = torch.stack([n(x) * sd for n, sd in nets]).mean(0)
            y = (c @ Ut + mut).cpu().numpy().astype(np.float64)
            psi[i:i + bs] = y.reshape(-1, NG, NG) * (np.asarray(Ip[i:i + bs]) / 1e6)[:, None, None] \
                + kn[i:i + bs]
    return psi


def rscale_tag(rscale):
    """Passive resistance-scale tag of the c1_r<tag> dirs: '1', '0.5', '2' (fpkgm, Oct 14 r = 0.5 map)."""
    return f"{float(rscale):g}"


def amort_maps(config, net_dir, f, passive_dir, out, device="cuda", bs=8192, n=None, log=print,
               rscale="1"):
    """Ensemble-mean psi of one amortized net on the first n frames (default all) of `config`:
    the arithmetic of fec.amort_real.run([tag], v2=True) (8192-frame blocks, float64
    accumulation, one float32 rounding).  Writes out/psi.npy + config.json (no meta: the tsm node
    recomputes it on the smoothed maps).

    rscale (default '1' = production S4): the C1 passive currents passive_dir/c1_r<rscale>/I.npy
    enter the known vacuum part only (the network inputs do not depend on them), so rscale 0.5 is
    the Oct 14 T1 knob map = S4 network + the r = 0.5 passive-flux increment, exactly as
    fec/p2pass_knobs.py real --knob r0.5 (amort_real.run with pass1_path swapped)."""
    import torch
    from fec.amort_net import coil_matrix
    from fec.features import load_ds
    from fec.passive import PassiveModel
    set_f(f)
    ds = load_ds(config)
    N_all = len(ds["t_ms"])
    N = N_all if n is None else int(n)
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    _, M = coil_matrix()
    model = load_net(net_dir, device=device)
    net_cfg = model[4]
    if net_cfg.get("gen") and not str(net_cfg["gen"]).endswith(f"f{float(f):g}"):
        raise ValueError(f"net {net_dir} was trained on generator {net_cfg['gen']}, not at f {f}")
    Gg = PassiveModel().Gg
    rs = rscale_tag(rscale)
    pf = Path(passive_dir) / f"c1_r{rs}" / "I.npy"
    I_pass = np.load(pf, mmap_mode="r")
    if I_pass.shape[0] != N_all:
        raise ValueError(f"{pf} has {I_pass.shape[0]} frames, {config} has {N_all}")
    dev = torch.device(device)
    tmp = d / "psi.part.npy"
    P = np.lib.format.open_memmap(tmp, "w+", np.float32, (N, NG, NG))
    t0 = time.time()
    for a in range(0, N, bs):
        g = np.arange(a, min(N, a + bs))
        fr = frame_inputs(g, ds, config)
        acc = np.zeros((len(g), NG, NG))
        acc += predict_psi(model, dev, M, Gg, I_pass, g, fr["I"], fr["Ip"], fr["R"],
                           fr["te_real"], fr["ne_real"], ds)
        P[g] = (acc / 1).astype(np.float32)
        log(f"{g[-1] + 1}/{N} frames {time.time() - t0:.0f}s")
    P.flush(); del P
    os.replace(tmp, d / "psi.npy")
    info = dict(kind="amortized psi network (synthetic-trained, inputs only)", config=config,
                N=int(N), n_full=int(N_all), net=str(net_dir), sol_frac=float(f),
                net_cfg={k: net_cfg.get(k) for k in ("gen", "K", "width", "depth", "seeds",
                                                     "n_in", "norm")},
                net_sha256=sha256(Path(net_dir) / "model.pt"),
                basis_sha256=sha256(Path(net_dir) / "basis.npz"),
                passive=str(pf), passive_sha256=sha256(pf), device=device,
                psi_sha256=sha256(d / "psi.npy"), seconds=round(time.time() - t0, 1),
                code="fec/final_mast.py amort (port of fec/amort_real.run v2)")
    if rs != "1":
        info["rscale"] = rs
    _write_json(d / "config.json", info)
    log("AMORT_DONE")
    return info


# ============================================================================ smoothing
def smooth_psi(src, out, shot, t_ms, sigma_ms, device="cuda", batch=2048, log=print,
               meta_fn=None):
    """fec.mast_tsm.smooth_maps without the scalar copies: within-shot Gaussian smoothing of
    src/psi.npy (float64 tensordot per shot, one float32 rounding) + production meta recomputed
    on the smoothed maps in batches of whole shots of >= batch frames."""
    from fec.mast_tsm import META_KEYS, gaussian_weights, shot_segments, solver_meta_fn
    src, out = Path(src), Path(out)
    if src.resolve() == out.resolve():
        raise ValueError("out must differ from src")
    P = np.load(src / "psi.npy", mmap_mode="r")
    N = P.shape[0]
    shot = np.asarray(shot)[:N]
    t = np.asarray(t_ms)[:N].astype(np.float64)
    if not np.isfinite(t).all():
        raise ValueError("t_ms has non-finite entries")
    meta_fn = meta_fn or solver_meta_fn(device)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "psi.part.npy"
    psi_out = np.lib.format.open_memmap(tmp, "w+", np.float32, P.shape)
    meta = {k: np.full(N, np.nan) for k in META_KEYS}
    meta["method"] = np.ones(N)
    starts, ends = shot_segments(shot)
    buf_idx, buf = [], []
    t0 = time.time()

    def flush():
        if not buf_idx:
            return
        idx = np.concatenate(buf_idx)
        r = meta_fn(np.concatenate(buf).astype(np.float32))
        for k in META_KEYS:
            meta[k][idx] = r[k]
        buf_idx.clear(); buf.clear()

    n_acc = 0
    for a, e in zip(starts, ends):
        w = gaussian_weights(t[a:e], float(sigma_ms))
        x = np.tensordot(w, np.asarray(P[a:e], dtype=np.float64), axes=1).astype(np.float32)
        psi_out[a:e] = x
        buf_idx.append(np.arange(a, e)); buf.append(x)
        n_acc += e - a
        if n_acc >= batch:
            flush()
            n_acc = 0
            log(f"{e}/{N} {time.time() - t0:.0f}s")
    flush()
    psi_out.flush(); del psi_out
    os.replace(tmp, out / "psi.npy")
    np.savez(out / "meta.npz", **meta)
    info = dict(kind="temporal smoothing of psi", src=str(src), N=int(N), sigma_ms=float(sigma_ms),
                shots=int(len(starts)), batch=int(batch), device=device,
                psi_sha256=sha256(out / "psi.npy"), seconds=round(time.time() - t0, 1),
                code="fec/final_mast.py tsm (= fec/mast_tsm.smooth_maps psi + meta)")
    _write_json(out / "config.json", info)
    log("TSM_DONE")
    return info


# ============================================================================ physics avg_f05
def plan_avg_f05(plan):
    """avg_f05 on a fec.mast_pipeline.Plan: members, merge, the finish of zsc_c1 (its tsm node
    copies zsc_c1's scalars), tsm30 of zsc_c1, the equal-weight average in the cand_avg_f05
    member order, and the finish (stage 2 + frozen betaN) of the average."""
    from fec import mast_pipeline as MP
    for name in ("zsc_c1", "zsc_nopas", "zsc_c1_lcc", "zfix0_c1"):
        plan.add_member(dict(MP.NAMED_SPECS[name], name=name))
    plan.add_merge("dc", "zsc_c1", "zsc_c1_lcc", 0.18)
    plan.add_finish("zsc_c1")
    plan.add_tsm("zsc_c1_tsm30", "zsc_c1", 30.0)
    plan.add_average("avg_f05", list(AVG_F05_MEMBERS), [0.2] * len(AVG_F05_MEMBERS))
    plan.add_finish("avg_f05")
    return plan


def run_physics(a):
    from fec import mast_pipeline as MP
    plan = MP._make_plan(a, blim=True, betan="betan_v2.npy")
    plan_avg_f05(plan)
    return MP._execute_plan(a, plan, "avg_f05", extra=dict(plan="avg_f05",
                                                          members=list(AVG_F05_MEMBERS)),
                            pipeline_json=True, final="avg_f05")


# ============================================================================ betaN fill3
def postprocess_fill3(x, shot, k=5, clip=(0.0, 8.0), min_finite=MIN_FINITE):
    """fill3 shot fill (fec_work/scratch/bnmap_fillfix.postprocess_minfill): NaN frames take the
    shot median only if the shot has >= min_finite finite frames, else the fold median of the
    finite frames; then the per-shot 5-frame running median and the clip."""
    from scipy.ndimage import median_filter
    x = np.asarray(x, dtype=np.float64)
    shot = np.asarray(shot)
    y = x.copy()
    glob = np.nanmedian(x)
    for s in np.unique(shot[~np.isfinite(x)]):
        m = shot == s
        nf = int(np.isfinite(x[m]).sum())
        y[m & ~np.isfinite(x)] = np.nanmedian(x[m]) if nf >= min_finite else glob
    z = y.copy()
    for s in np.unique(shot):
        m = np.flatnonzero(shot == s)
        if len(m) >= 3:
            z[m] = median_filter(y[m], size=min(k, len(m) | 1), mode="nearest")
    return np.clip(z, *clip)


def betan_fill3(records, shot_all, min_finite=MIN_FINITE):
    """Raw frozen lam_fgw betaN of the records (fec.betan_transfer.predict, the selection.json
    model) placed on the fold's frames, then postprocess_fill3.  Returns (betaN, raw)."""
    import fec.betan_transfer as BT
    from fec.run_final import betan_frozen_files
    fz = betan_frozen_files()
    if not all(v["reviewed"] for v in fz.values()):
        raise RuntimeError(f"betaN frozen files changed: {fz}")
    sel = json.loads(Path(fz["selection.json"]["path"]).read_text())
    model = sel["candidates"]["lam_fgw"]["model"]
    r = dict(records)
    if "Bgeo" not in r:
        r["Bgeo"] = r["B0"] * r["Rax"] / r["Rgeo"]
    raw = np.full(len(shot_all), np.nan)
    raw[r["gidx"].astype(int)] = BT.predict(model, r)[0]
    return postprocess_fill3(raw, shot_all, min_finite=min_finite), raw


# ============================================================================ composite
def composite(config, maps, scalars, out, min_finite=MIN_FINITE, betan="betan_v2.npy",
              log=print):
    """Submission MAST dir: psi.npy + meta.npz of the smoothed amortized maps, scalars.npz and
    betan_records.npz of the physics average, betan_v2.npy = fill3 betaN of those records."""
    from fec.features import load_ds
    maps, scalars, out = Path(maps), Path(scalars), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shot_all = np.asarray(load_ds(config)["shot"])
    N = np.load(maps / "psi.npy", mmap_mode="r").shape[0]
    with np.load(scalars / "scalars.npz") as z:
        ns = {k: len(z[k]) for k in z.files if z[k].ndim == 1}
    if N != len(shot_all) or any(v != N for v in ns.values()):
        raise ValueError(f"maps {N} / scalars {ns} / fold {len(shot_all)} frame counts differ")
    for f in ("psi.npy", "meta.npz"):
        dst = out / f
        if dst.exists():
            dst.unlink()
        try:
            os.link(maps / f, dst)          # hard link: no 1.2 GB copy
        except OSError:
            shutil.copyfile(maps / f, dst)
    for f in ("scalars.npz", "betan_records.npz"):
        shutil.copyfile(scalars / f, out / f)
    with np.load(scalars / "betan_records.npz") as z:
        rec = {k: z[k] for k in z.files}
    bn, raw = betan_fill3(rec, shot_all, min_finite=min_finite)
    np.save(out / betan, bn.astype(np.float64))
    info = dict(kind="composite: amortized maps + physics-average scalars + fill3 betaN",
                config=config, N=int(N), maps=str(maps), scalars=str(scalars),
                psi_sha256=sha256(out / "psi.npy"), scalars_sha256=sha256(out / "scalars.npz"),
                records_sha256=sha256(out / "betan_records.npz"),
                betan=betan, betan_sha256=sha256(out / betan), min_finite=int(min_finite),
                raw_finite_frac=float(np.isfinite(raw).mean()),
                betan_median=float(np.median(bn)),
                code="fec/final_mast.py composite")
    _write_json(out / "config.json", info)
    log("COMPOSITE_DONE " + json.dumps({k: info[k] for k in ("N", "raw_finite_frac",
                                                             "betan_median")}))
    return info


# ============================================================================ Oct 14 map variants
# Agent fpkgm (2026-10-13; docs/archive/fec/FINAL_PKG_MAST_PRESETS_20261013.md): the stages the
# Final package needs if a 2026-10-14 Dev read adopts one of
#   r05    amort maps with rscale 0.5 (above), then tsm                 (T1, family doc amendment 5)
#   VB     0.5 (raw v7h + raw TX1s maps), then tsm                      (T2; TX1s maps: fec/final_tx1s.py)
#   combo  VB tsm + (r05 tsm - S4 tsm), meta recomputed                 (amendment 11, C59)
# Each is the arithmetic of the Dev-read build, config-parametrised (parity: the detail doc).  The betaN
# 0.85 m reference-field twin (C49) was removed after amendment 12: every preset keeps the fill3 betaN.
BLEND_CHUNK = 4096
COMBO_CHUNK = 4096


def blend_maps(src_a, src_b, out, n=None, chunk=BLEND_CHUNK, log=print):
    """VB raw node: psi = 0.5 * (A + B) in float32 arithmetic over A = src_a/psi.npy (raw v7h maps)
    and B = src_b/psi.npy (raw TX1s maps); the expression of the Dev-read build
    (fec_work/scratch/root_a3/vb_blend_build.py node(), the 'plasma-part blend on one vacuum')."""
    A = np.load(Path(src_a) / "psi.npy", mmap_mode="r")
    B = np.load(Path(src_b) / "psi.npy", mmap_mode="r")
    if A.shape != B.shape or A.dtype != np.float32 or B.dtype != np.float32:
        raise ValueError(f"blend inputs differ: {A.shape} {A.dtype} / {B.shape} {B.dtype}")
    N = A.shape[0] if n is None else int(n)
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "psi.part.npy"
    P = np.lib.format.open_memmap(tmp, "w+", np.float32, (N,) + A.shape[1:])
    t0 = time.time()
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        P[a:b] = 0.5 * (np.asarray(A[a:b], np.float32) + np.asarray(B[a:b], np.float32))
    P.flush(); del P
    os.replace(tmp, d / "psi.npy")
    info = dict(kind="VB blend: 0.5 * (A + B), float32", N=int(N), a=str(src_a), b=str(src_b),
                a_psi_sha256=sha256(Path(src_a) / "psi.npy"),
                b_psi_sha256=sha256(Path(src_b) / "psi.npy"),
                psi_sha256=sha256(d / "psi.npy"), seconds=round(time.time() - t0, 1),
                code="fec/final_mast.py blend (= scratch/root_a3/vb_blend_build.py node)")
    _write_json(d / "config.json", info)
    log("BLEND_DONE")
    return info


def meta_batches(shot, batch=2048):
    """Frame-index batches exactly as smooth_psi / fec.mast_tsm.smooth_maps flush their meta
    buffer: whole shots queued in order, flushed once >= batch frames; the remainder last
    (= fec/combo_build.py meta_batches)."""
    from fec.mast_tsm import shot_segments
    starts, ends = shot_segments(np.asarray(shot))
    out, buf, n_acc = [], [], 0
    for a, e in zip(starts, ends):
        buf.append(np.arange(a, e))
        n_acc += e - a
        if n_acc >= batch:
            out.append(np.concatenate(buf))
            buf, n_acc = [], 0
    if buf:
        out.append(np.concatenate(buf))
    return out


def maps_meta(P, shot, meta_fn, batch=2048, log=print):
    """Production meta (psi_ax, psi_b, Rax, Zax, limited; method = 1) of stored float32 maps with
    the smooth_psi batching (= fec/combo_build.py meta_on_maps).  Returns (meta, n_batches)."""
    from fec.mast_tsm import META_KEYS
    N = P.shape[0]
    meta = {k: np.full(N, np.nan) for k in META_KEYS}
    meta["method"] = np.ones(N)
    bl = meta_batches(np.asarray(shot)[:N], batch)
    t0 = time.time()
    for k, idx in enumerate(bl):
        r = meta_fn(np.asarray(P[idx[0]:idx[-1] + 1], np.float32))   # batches are contiguous
        for key in META_KEYS:
            meta[key][idx] = r[key]
        if k % 5 == 0:
            log(f"meta batch {k + 1}/{len(bl)} {idx[-1] + 1}/{N} {time.time() - t0:.0f}s")
    return meta, len(bl)


def combine_chunk(vb, r05, s4):
    """combo = VB + (r05 - S4) in float64, rounded once to float32 (fec/combo_build.py)."""
    vb, r05, s4 = (np.asarray(x) for x in (vb, r05, s4))
    if not (vb.shape == r05.shape == s4.shape):
        raise ValueError(f"shape mismatch {vb.shape} {r05.shape} {s4.shape}")
    return (vb.astype(np.float64) + (r05.astype(np.float64) - s4.astype(np.float64))).astype(np.float32)


def combo_maps(config, vb, r05, s4, out, device="cuda", chunk=COMBO_CHUNK, batch=2048, meta_fn=None,
               log=print):
    """Amendment 11 combo node: psi = VB + (r05 - S4), all three after tsm30 (dirs with psi.npy),
    streamed in 4,096-frame chunks; meta.npz recomputed on the combo maps with the production
    V3Solver meta and the smooth_psi batching (composite hard-links psi.npy + meta.npz)."""
    from fec.features import load_ds
    from fec.mast_tsm import solver_meta_fn
    Pv, Pr, Ps = (np.load(Path(x) / "psi.npy", mmap_mode="r") for x in (vb, r05, s4))
    if not (Pv.shape == Pr.shape == Ps.shape) or {Pv.dtype, Pr.dtype, Ps.dtype} != {np.dtype(np.float32)}:
        raise ValueError(f"combo inputs differ: {Pv.shape} {Pr.shape} {Ps.shape}")
    N = Pv.shape[0]
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "psi.part.npy"
    P = np.lib.format.open_memmap(tmp, "w+", np.float32, Pv.shape)
    t0 = time.time()
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        P[a:b] = combine_chunk(Pv[a:b], Pr[a:b], Ps[a:b])
    P.flush(); del P
    os.replace(tmp, d / "psi.npy")
    t1 = time.time()
    shot = np.asarray(load_ds(config)["shot"])
    meta, nb = maps_meta(np.load(d / "psi.npy", mmap_mode="r"), shot,
                         meta_fn or solver_meta_fn(device), batch=batch, log=log)
    np.savez(d / "meta.npz", **meta)
    info = dict(kind="combo: VB + (r05 - S4), all after tsm; float64, one float32 rounding",
                config=config, N=int(N), vb=str(vb), r05=str(r05), s4=str(s4),
                inputs_psi_sha256={k: sha256(Path(x) / "psi.npy") for k, x in
                                   (("vb", vb), ("r05", r05), ("s4", s4))},
                psi_sha256=sha256(d / "psi.npy"), meta_sha256=sha256(d / "meta.npz"),
                meta_batches=int(nb), device=device, seconds_maps=round(t1 - t0, 1),
                seconds_meta=round(time.time() - t1, 1),
                code="fec/final_mast.py combo (= fec/combo_build.py build + meta)")
    _write_json(d / "config.json", info)
    log("COMBO_DONE")
    return info


# ============================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("passive_f")
    p.add_argument("--config", required=True, choices=CONFIGS)
    p.add_argument("--f", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rscales", default="1")
    p = sub.add_parser("amort")
    p.add_argument("--config", required=True, choices=CONFIGS)
    p.add_argument("--net", required=True)
    p.add_argument("--f", type=float, required=True)
    p.add_argument("--passive", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--rscale", default="1", help="C1 passive file passive/c1_r<rscale> (Oct 14 r05: 0.5)")
    p = sub.add_parser("tsm")
    p.add_argument("--config", required=True, choices=CONFIGS)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sigma-ms", type=float, required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("physics")
    from fec import mast_pipeline as MP
    MP._common(p)
    p = sub.add_parser("composite")
    p.add_argument("--config", required=True, choices=CONFIGS)
    p.add_argument("--maps", required=True)
    p.add_argument("--scalars", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-finite", type=int, default=MIN_FINITE)
    p = sub.add_parser("blend")
    p.add_argument("--a", required=True, help="raw v7h maps dir (psi.npy)")
    p.add_argument("--b", required=True, help="raw TX1s maps dir (psi.npy, fec/final_tx1s.py infer)")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=None)
    p = sub.add_parser("combo")
    p.add_argument("--config", required=True, choices=CONFIGS)
    p.add_argument("--vb", required=True)
    p.add_argument("--r05", required=True)
    p.add_argument("--s4", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    log = lambda m: print(m, flush=True)  # noqa: E731
    if a.cmd == "passive_f":
        passive_frames(a.config, a.f, a.out, tuple(x for x in a.rscales.split(",") if x), log=log)
    elif a.cmd == "amort":
        amort_maps(a.config, a.net, a.f, a.passive, a.out, device=a.device, n=a.n, log=log,
                   rscale=a.rscale)
    elif a.cmd == "tsm":
        from fec.features import load_ds
        ds = load_ds(a.config)
        smooth_psi(a.src, a.out, ds["shot"], ds["t_ms"], a.sigma_ms, device=a.device, log=log)
    elif a.cmd == "physics":
        return run_physics(a)
    elif a.cmd == "composite":
        composite(a.config, a.maps, a.scalars, a.out, min_finite=a.min_finite, log=log)
    elif a.cmd == "blend":
        blend_maps(a.a, a.b, a.out, n=a.n, log=log)
    elif a.cmd == "combo":
        combo_maps(a.config, a.vb, a.r05, a.s4, a.out, device=a.device, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
