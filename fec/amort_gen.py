"""Synthetic MAST training set for the amortized psi network (family AMORTIZED_MAST_20261010, G0).

Every "truth" is our own free-boundary GS solution (fec.twin_mast.TwinSolver + solve_truth),
driven by REAL mast_public_test INPUTS only: coil currents and Ip at offset index 6, the Thomson
channel radii and the real missing-channel pattern.  No MAST label, demo parquet or scorer
envelope mask is opened.

Frames: every frame of every mast_public_test shot EXCEPT the 40 twin evaluation shots
(fec.twin_mast.load_frames(seed=0)), cut into 25-frame segments (the twin's window length).
Prior P_train = twin scenario B: in-family J shape (beta0 in [0,0.7], a1=a2 in [1,3],
n1=n2 in [0.5,3]) as smooth tracks per segment (twin_mast.truth_params, knots every 8 frames),
dv in +-10 mT, force-free height (twin_mast.solve_truth, 3 secant steps).
Te-peak following (cheap form of twin_mast.build_truth): K candidate tracks per segment are
screened with one Z=0 solve; each segment keeps the track whose axis follows the real Thomson
Te-peak radius best (build_truth's cost), and only that track is solved force-free.
Synthetic Thomson: twin_mast.synth_thomson (noise, outliers, real missing pattern), n_real
independent realisations per truth (used as training-time augmentation).

CLI (GPU through the scheduler; outputs under FEC_WORK/amort/gen_<tag>/chunk_*.npz):
  python -B fec/gpu_slot.py run --owner amort -- python -B fec/amort_gen.py gen --tag p2k --max-frames 2000
  python -B fec/gpu_slot.py run --owner amort -- python -B fec/amort_gen.py gen --tag main
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

OUT = WORK / "amort"
TWIN = WORK / "twin"
SEG = 25            # segment (track) length in frames, = twin window
N_EVAL_SHOTS = 40   # twin evaluation shots, load_frames(n_shots=40, win=25, seed=0)


# ============================================================================ frames
def twin_eval_shots():
    """Shot indices of the twin evaluation frames, recomputed with load_frames' first RNG call
    (no Thomson loading) and cross-checked against the cached twin truth when present."""
    from fec.features import load_ds
    ds = load_ds("mast_public_test")
    shot = ds["shot"]
    rng = np.random.default_rng(0)
    nfr = np.bincount(shot, minlength=len(ds["info"]["order"]))
    elig = np.flatnonzero(nfr >= SEG)
    shots = np.sort(rng.choice(elig, N_EVAL_SHOTS, replace=False))
    f = TWIN / "truth_B.npz"
    if f.exists():
        cached = np.unique(np.load(f)["fr_shot"])
        if not np.array_equal(cached, shots):
            raise RuntimeError("twin evaluation shots differ from the cached truth_B")
    return shots


def segments(shot, fidx, seg=SEG):
    """Global segment id per frame: consecutive `seg`-frame blocks within each shot (a tail
    shorter than seg/2 joins the previous block).  Frames must be sorted by (shot, fidx)."""
    out = np.empty(len(shot), np.int64)
    nxt = 0
    for s in np.unique(shot):
        ix = np.flatnonzero(shot == s)
        n = len(ix)
        nb = max(1, int(round(n / seg)))
        loc = np.minimum(np.arange(n) // seg, nb - 1)
        out[ix] = nxt + loc
        nxt += nb
    return out


def frame_inputs(g, ds=None):
    """Inputs for global frames g exactly as twin_mast.load_frames builds them."""
    from fec.features import load_ds
    from fec.machines import MAST_CH, MAST_COILS
    from fec.mast_v1 import thomson_frame
    ds = load_ds("mast_public_test") if ds is None else ds
    info, shot = ds["info"], ds["shot"]
    cur = np.asarray(ds["cur"][g], dtype=np.float64)[:, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    Rl, tel, nel = [], [], []
    cache = {}
    for k in g:
        stem = info["order"][shot[k]]
        if stem not in cache:
            cache.clear()
            cache[stem] = dict(np.load(WORK / "shots" / "mast_public_test" / f"{stem}.npz"))
        R, te, ne = thomson_frame(cache[stem], ds["fidx"][k])
        Rl.append(R); tel.append(te); nel.append(ne)
    nch = max(len(r) for r in Rl)
    pad = lambda L: np.stack([np.pad(np.asarray(x, float), (0, nch - len(x)),
                                     constant_values=np.nan) for x in L])
    return dict(g=np.asarray(g), shot=shot[g], fidx=ds["fidx"][g], t_ms=ds["t_ms"][g],
                I=cur[:, ci] * 1e3,
                Ip=np.maximum(cur[:, MAST_CH.index("magnetics_plasma_current")] * 1e3, 5e4),
                Itf=cur[:, MAST_CH.index("magnetics_tf_current")],
                R=pad(Rl), te_real=pad(tel), ne_real=pad(nel))


def train_frames():
    """All frames of the non-twin shots, sorted by (shot, fidx), with segment ids (cached)."""
    f = OUT / "frames_train.npz"
    if f.exists():
        return dict(np.load(f))
    from fec.features import load_ds
    ds = load_ds("mast_public_test")
    ex = twin_eval_shots()
    shot, fidx = ds["shot"], ds["fidx"]
    g = np.flatnonzero(~np.isin(shot, ex))
    g = g[np.lexsort((fidx[g], shot[g]))]
    fr = frame_inputs(g, ds)
    fr["seg"] = segments(fr["shot"], fr["fidx"])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(f, **fr)
    return fr


def take(fr, ix):
    return {k: v[ix] for k, v in fr.items()}


def midplane_rin_rout(psi, psi_ax, psi_b, gR, gZ):
    """Midplane LCFS radii from the Z=0 row (as twin_mast._scal_one), vectorised over frames."""
    iz0 = int(np.argmin(np.abs(gZ)))
    Rf = np.linspace(gR[0], gR[-1], 2001)
    B = len(psi)
    Rin, Rout = np.full(B, np.nan), np.full(B, np.nan)
    for i in range(B):
        pn = (psi_ax[i] - psi[i, iz0]) / (psi_ax[i] - psi_b[i])
        pf = np.interp(Rf, gR, pn)
        j0 = int(np.argmin(np.where((Rf > 0.3) & (Rf < 1.4), pf, np.inf)))
        if pf[j0] < 1:
            jo = np.flatnonzero(pf[j0:] >= 1)
            ji = np.flatnonzero(pf[:j0 + 1][::-1] >= 1)
            Rout[i] = Rf[j0 + jo[0]] if len(jo) else np.nan
            Rin[i] = Rf[j0 - ji[0]] if len(ji) else np.nan
    return Rin, Rout


# ============================================================================ generation
def screen_select(S, fr, K, n_screen, seed):
    """K scenario-B tracks per segment; keep per segment the track whose Z=0 screening axis
    follows the real Te-peak radius best (cost as twin_mast.build_truth)."""
    from fec.twin_mast import real_te_peak, truth_params
    B = len(fr["Ip"])
    pk = real_te_peak(fr)
    cands = []
    for k in range(K):
        prof, dv = truth_params(fr, "B", seed=seed + k)
        Ix = np.concatenate([fr["I"], dv[:, None]], 1)
        o = S.solve_gen(Ix, fr["Ip"], prof, np.zeros(B), n_iter=n_screen,
                        Rinit=np.full(B, 0.9))
        Rax = o["Rax"].cpu().numpy()
        ok = (o["dI"].cpu().numpy() < 1e-2) & (o["xu"][0] | o["xl"][0]).cpu().numpy() & \
            (Rax > 0.47) & (Rax < 1.28)
        cands.append((prof, dv, Rax, ok))
    sh = fr["shot"]                      # segment ids (tracks are per segment)
    sel = np.zeros(B, int)
    for s_ in np.unique(sh):
        ix = np.flatnonzero(sh == s_)
        cost = []
        for prof, dv, Rax, ok in cands:
            d = np.abs(Rax[ix] - pk[ix])
            d = np.where(np.isfinite(d), d, 0.05)
            cost.append(np.mean(np.where(ok[ix], d, 0.5)))
        sel[ix] = int(np.argmin(cost))
    r = np.arange(B)
    prof = np.stack([c[0] for c in cands])[sel, r]
    dv = np.stack([c[1] for c in cands])[sel, r]
    return prof, dv, sel, pk


def gen(tag, max_frames=0, chunk=2000, K=4, n_screen=100, n_real=4, seed=0, limit_chunks=0):
    import torch
    from fec.twin_mast import make_solver, solve_truth, synth_thomson, truth_valid, sol_fraction
    fr_all = train_frames()
    d = OUT / f"gen_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    segs = np.unique(fr_all["seg"])
    rng.shuffle(segs)                    # chunks = random mixes of segments (shots)
    if max_frames:
        cnt = np.bincount(fr_all["seg"])[segs]
        segs = segs[:int(np.searchsorted(np.cumsum(cnt), max_frames)) + 1]
    cnt = np.bincount(fr_all["seg"], minlength=segs.max() + 1)
    groups, cur, n = [], [], 0
    for s in segs:
        if n + cnt[s] > chunk and cur:
            groups.append(cur); cur, n = [], 0
        cur.append(s); n += cnt[s]
    if cur:
        groups.append(cur)
    if limit_chunks:
        groups = groups[:limit_chunks]
    print(f"gen {tag}: {sum(len(x) for x in groups)} segments, "
          f"{int(sum(cnt[x].sum() for x in groups))} frames, {len(groups)} chunks, K={K}, "
          f"n_screen={n_screen}, n_real={n_real}", flush=True)
    S, spec, M = make_solver()
    t_all = time.time()
    n_done = 0
    for ci, grp in enumerate(groups):
        f = d / f"chunk_{ci:04d}.npz"
        if f.exists():
            continue
        ix = np.flatnonzero(np.isin(fr_all["seg"], grp))
        fr = take(fr_all, ix)
        fr["shot_real"] = fr["shot"]
        fr["shot"] = fr["seg"]           # twin helpers draw smooth tracks per "shot"
        B = len(ix)
        torch.cuda.synchronize(); t0 = time.time()
        prof, dv, sel, pk = screen_select(S, fr, K, n_screen, seed=1000 * (seed + 1) + 17 * ci)
        torch.cuda.synchronize(); t1 = time.time()
        tr = solve_truth(S, fr, prof, dv)
        torch.cuda.synchronize(); t2 = time.time()
        valid = truth_valid(tr)
        tes, nes = [], []
        for r in range(n_real):
            ts = synth_thomson(fr, tr, spec.gR, spec.gZ, seed=10_000 * (seed + 1) + 97 * ci + r)
            tes.append(ts["te"].astype(np.float32)); nes.append(ts["ne"].astype(np.float32))
        Rin, Rout = midplane_rin_rout(tr["psi"], tr["psi_ax"], tr["psi_b"], spec.gR, spec.gZ)
        t3 = time.time()
        with np.errstate(invalid="ignore"):
            real = valid & np.isfinite(Rout) & (Rout <= 1.55) & (Rin >= 0.20)
        np.savez(f, g=fr["g"], shot=fr["shot_real"], seg=fr["seg"], fidx=fr["fidx"],
                 I=fr["I"], Ip=fr["Ip"], Itf=fr["Itf"], R=fr["R"].astype(np.float32),
                 te=np.stack(tes), ne=np.stack(nes), prof=prof, dv=dv, track=sel,
                 te_peak_real=pk, psi=tr["psi"], psi_ax=tr["psi_ax"], psi_b=tr["psi_b"],
                 Rax=tr["Rax"], Zax=tr["Zax"], c=tr["c"], dI=tr["dI"], Zt=tr["Zt"],
                 xfound=tr["xfound"], valid=valid, Rin=Rin, Rout=Rout, realistic=real,
                 sol_fraction=sol_fraction())
        n_done += B
        dd = np.abs(tr["Rax"] - pk)
        print(f"chunk {ci}: B {B} screen {t1 - t0:.0f}s truth {t2 - t1:.0f}s ts {t3 - t2:.0f}s "
              f"| {B / (t3 - t0):.1f} fr/s | valid {valid.mean():.3f} realistic {real.mean():.3f}"
              f" | Rax pct {np.percentile(tr['Rax'], [5, 50, 95]).round(3).tolist()} "
              f"|Rax-pk| med {np.nanmedian(dd):.3f} | total {n_done} frames "
              f"{time.time() - t_all:.0f}s", flush=True)
    print(f"GEN_DONE {tag} frames {n_done} wall {time.time() - t_all:.0f}s", flush=True)


def load_gen(tag, valid_only=True, keys=None, max_chunks=0):
    """Concatenate the chunks of a generated set (valid truths only by default)."""
    d = OUT / f"gen_{tag}"
    fs = sorted(d.glob("chunk_*.npz"))
    if max_chunks:
        fs = fs[:max_chunks]
    if not fs:
        raise FileNotFoundError(d)
    parts = []
    for f in fs:
        z = np.load(f)
        ks = [k for k in z.files if k != "sol_fraction"] if keys is None else keys
        m = z["valid"] if valid_only else np.ones(len(z["Ip"]), bool)
        p = {}
        for k in ks:
            a = z[k]
            p[k] = a[:, m] if k in ("te", "ne") else a[m]
        parts.append(p)
    return {k: np.concatenate([p[k] for p in parts], 1 if k in ("te", "ne") else 0)
            for k in parts[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frames", "gen"])
    ap.add_argument("--tag", default="p2k")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n-screen", type=int, default=100)
    ap.add_argument("--n-real", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-chunks", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "frames":
        t0 = time.time()
        fr = train_frames()
        print(f"train frames {len(fr['g'])} shots {len(np.unique(fr['shot']))} segments "
              f"{len(np.unique(fr['seg']))} ({time.time() - t0:.0f}s)", flush=True)
    else:
        gen(a.tag, a.max_frames, a.chunk, a.K, a.n_screen, a.n_real, a.seed, a.limit_chunks)


if __name__ == "__main__":
    main()
