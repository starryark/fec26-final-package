"""NRS-L cascade stage of the MAST half (configs final_*_nrsl; family doc amendment 68; agent nrsl_pkg 2026-10-20).

What it does, per frame of the MAST fold, on the maps the composite would otherwise ship (the 30 ms-smoothed amortized
maps, `composite/maps`):
  1. re-solve the plasma with the frozen operator (fec/nrsl_op.py) in the MAST physical field: coil currents at solenoid
     fraction 0.55, the maps' own C1 passive currents, a uniform-Bz column; measured Ip (floor 100 kA); profile = an
     8-term fit to the base map's own node currents; height held at the base map's current centroid;
  2. stage L: the profile's peaking follows the label-free li target of the frozen current-diffusion estimator
     (fec/nrsl_li.py: fec/current_diffusion.py + its frozen DIII-D/synthetic TRAIN calibration, run on the base maps and
     the shot's Ip / Thomson inputs), radius held by a pressure-like current term;
     stage P (frames where stage L fails its rule): radius held by the uniform-Bz column, no li target;
     a frame where both fail keeps the base map in the cascade (fec/nrsl_cascade.py), i.e. contributes no increment of
     its own;
  3. only the increment is smoothed: psi = base + W30 (cascade - base) per shot (the shipped 30 ms Gaussian weights; the
     increment is computed on the already-smoothed base maps, so psi = tsm30(raw + increment)); a frame without an
     increment of its own still carries its neighbours' smoothed increments;
  4. label-free fold assertions (amendment-64 vetoes, amendment-67(a) axis bounds, li-target coverage / range / the
     share of estimates clipped by the frozen calibration (its li_cd range in cdiff/frozen.json), finiteness).  A fold
     whose assertions fail keeps the parent config's maps unchanged (logged, exit 0).  Any other error (code check,
     prep, li step, stage L / P, expansion, post, a child's non-zero exit, memory / file errors) writes no node maps
     and exits non-zero, so fec/run_final.py stops the build with the cause.
Only the MAST maps change; q95 / betaN come from the composite's scalars node as before.  CPU float64, 8 threads per
process, MKL_CBWR=AVX2, OPENBLAS_CORETYPE=Haswell, fixed 256-frame batches in fold order.  `--procs K` splits the batch
list into K contiguous runs of whole batches (each batch is the one-process run's batch), merged in order.

    python -B fec/mast_nrsl.py run --config mast_private_test --recipe RECIPE.json --maps MAPS_DIR \
        --passive PASSIVE_I.npy --out NODE_DIR [--workers 6] [--procs 1]
        [--base-npz SUB.npz]   (base maps from a submission file instead of MAPS_DIR/psi.npy; checks / tests)
        [--li LI.npz] [--limit-shots K] [--skip-eval]
-> NODE_DIR/psi.npy, meta.npz (the composite's maps), nrsl.json (report), config.json; work files in NODE_DIR/work/.
No MAST label, scorer mask / envelope, demo shot or MAST score is read anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

FEC = Path(__file__).resolve().parent
PKG = FEC.parent
PY = sys.executable
# the stage's subprocess scripts (frozen NRS-L files; their SHA-256 are pinned in the recipe block, code_sha256)
SCRIPTS = {"run": FEC / "nrsl_run.py", "li": FEC / "nrsl_li.py", "post": FEC / "nrsl_post.py",
           "expand": FEC / "nrsl_expand.py", "cascade": FEC / "nrsl_cascade.py"}
FROZEN_FILES = {"frozen_json": ("cdiff", "frozen.json"), "sigma_json": ("licond", "sigma.json")}
META_KEYS = ("psi_ax", "psi_b", "Rax", "Zax", "limited")


def sha256_file(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha256_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stage_env(q):
    """Child environment: q['threads'] BLAS / OpenMP threads per process and the pinned BLAS code paths."""
    n = str(int(q["threads"]))
    e = dict(OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n, PYTHONUNBUFFERED="1")
    e.update({k: str(v) for k, v in q["env"].items()})
    return e


def check_code(q):
    """G1 at run time: every stage file equals its pinned SHA-256, the stage settings equal the recipe's, the li
    estimator's frozen files are the calibrated ones.  Returns a list of problems (empty = ok)."""
    bad = []
    for rel, want in q["code_sha256"].items():
        p = PKG / rel
        got = sha256_file(p) if p.exists() else None
        if got != want:
            bad.append(f"{rel} sha256 {got} != {want}")
    from fec import current_diffusion as CD
    from fec import nrsl_run
    if sha256_file(CD.__file__) not in q["li"]["current_diffusion_sha256"]:
        bad.append("fec/current_diffusion.py is not the frozen estimator")
    if json.loads(json.dumps(nrsl_run.STAGES)) != q["stages"]:
        bad.append(f"nrsl_run.STAGES {nrsl_run.STAGES} != recipe stages {q['stages']}")
    for k, (d, f) in FROZEN_FILES.items():
        p = WORK / d / f
        got = sha256_file(p) if p.exists() else None
        if got != q["li"][f"{k}_sha256"]:
            bad.append(f"{p.name} sha256 {got} != {q['li'][f'{k}_sha256']}")
    return bad


def frame_chunks(frames, procs, bs):
    """Split a frame list into <= procs contiguous runs of whole bs-frame batches (positions in the list), so every
    batch any process forms equals the one-process run's batch."""
    nb = -(-len(frames) // bs)
    per = -(-nb // max(1, int(procs)))
    return [frames[i * per * bs:(i + 1) * per * bs] for i in range(-(-nb // per))] if nb else []


def run_stage(W, st, frames, q, procs, log, env):
    """nrsl_run.py for stage st over `frames` (None = all frames), one process or `procs` processes on contiguous
    whole-batch runs; returns the stage npz path (keys as nrsl_run writes them)."""
    out = W / f"stage_{st}{'' if frames is None else 'sub'}_cpu.npz"
    base = [PY, "-B", str(SCRIPTS["run"]), "--fold-dir", str(W), "--stage", st, "--device", q["device"],
            "--threads", str(q["threads"]), "--bs", str(q["batch"]), "--n_iter", str(q["n_iter"])]
    if procs <= 1:                         # the canonical one-process command (mast_solver2's ms2_cpu_canon.sh)
        if frames is not None:
            np.save(W / f"fr_{st}sub_cpu.npy", np.asarray(frames))
            base += ["--frames", str(W / f"fr_{st}sub_cpu.npy")]
        sh(base + ["--out", str(out)], log, env)
        return out
    allf = np.arange(np.load(W / "base.npy", mmap_mode="r").shape[0]) if frames is None else np.asarray(frames)
    parts = frame_chunks(allf, procs, int(q["batch"]))
    pd = W / "parts"
    pd.mkdir(exist_ok=True)
    procs_, files = [], []
    for i, fr in enumerate(parts):
        fp, op = pd / f"fr_{st}_{i:02d}.npy", pd / f"{st}_{i:02d}.npz"
        np.save(fp, fr)
        files.append(op)
        lf = open(pd / f"{st}_{i:02d}.log", "w", encoding="utf-8")
        cmd = base + ["--frames", str(fp), "--out", str(op)]
        log(f"$ {' '.join(cmd)}")
        procs_.append((subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(PKG),
                                        env=dict(os.environ, **env)), lf))
    rc = [p.wait() for p, _ in procs_]
    for _, lf in procs_:
        lf.close()
    if any(rc):
        raise RuntimeError(f"stage {st} processes exit {rc} (logs {pd})")
    zs = []
    for f in files:
        with np.load(f) as z:
            zs.append({k: z[k] for k in z.files})
    fr = np.concatenate([z["frames"] for z in zs])
    want = np.asarray(frames) if frames is not None else np.arange(len(fr))
    if not np.array_equal(fr, want):
        raise RuntimeError(f"stage {st}: merged frames differ from the requested frames")
    np.savez(out, frames=fr, **{k: np.concatenate([z[k] for z in zs]) for k in zs[0] if k != "frames"})
    return out


def sh(cmd, log, env):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([str(c) for c in cmd], cwd=str(PKG), env=dict(os.environ, **env))
    if r.returncode:
        raise RuntimeError(f"exit {r.returncode}: {' '.join(str(c) for c in cmd)}")


def prep(W, config, maps, passive, base_npz=None, limit_shots=0):
    """The fold dir the frozen tools read: base.npy (N,65,65 float32), inputs.npz, grid.npz (as mast_solver2's
    ms2_mast_prep: coils = ds cur (MAST_COILS, A), passive = the maps' f 0.55 C1 currents, Ip, P2U + P2L, TF)."""
    from fec.features import load_ds
    from fec.geometry import machine_geometry
    from fec.machines import MAST_CH, MAST_COILS
    ds = load_ds(config)
    shot, fidx, t_ms = np.asarray(ds["shot"]), np.asarray(ds["fidx"]), np.asarray(ds["t_ms"], float)
    nsh = len(ds["info"]["order"])
    bounds = np.searchsorted(shot, np.arange(nsh + 1))
    if base_npz:
        zf = np.load(base_npz)
        N = int(bounds[min(limit_shots, nsh)] if limit_shots else len(shot))
        B = np.lib.format.open_memmap(W / "base.npy", "w+", np.float32, (N, 65, 65))
        for i in range(nsh):
            if bounds[i] >= N:
                break
            p = zf[f"shot_{i:04d}_psirz"]
            if p.shape[0] != bounds[i + 1] - bounds[i] or p.dtype != np.float32:
                raise ValueError(f"{base_npz} shot {i}: {p.shape} {p.dtype}")
            B[bounds[i]:bounds[i + 1]] = p
        src = dict(base_npz=str(base_npz), base_npz_sha256=sha256_file(base_npz))
    else:
        P = np.load(Path(maps) / "psi.npy", mmap_mode="r")
        if P.dtype != np.float32 or P.shape[1:] != (65, 65) or P.shape[0] > len(shot):
            raise ValueError(f"maps {P.shape} {P.dtype} vs fold {len(shot)} frames")
        N = int(bounds[min(limit_shots, nsh)] if limit_shots else P.shape[0])
        B = np.lib.format.open_memmap(W / "base.npy", "w+", np.float32, (N, 65, 65))
        for a in range(0, N, 4096):
            B[a:min(N, a + 4096)] = P[a:min(N, a + 4096)]
        src = dict(maps=str(maps), maps_psi_sha256=sha256_file(Path(maps) / "psi.npy"))
    B.flush()
    del B
    cur = np.asarray(ds["cur"], np.float64)[:N, :, 6]
    ci = [MAST_CH.index(c) for c in MAST_COILS]
    Ipas = np.load(passive, mmap_mode="r")
    if Ipas.shape[0] < N:
        raise ValueError(f"passive {Ipas.shape} shorter than {N} frames")
    Iext = np.concatenate([cur[:, ci] * 1e3, np.asarray(Ipas[:N], np.float64)], 1)
    Ip = cur[:, MAST_CH.index("magnetics_plasma_current")] * 1e3
    p2 = cur[:, MAST_CH.index("magnetics_p2u_current")] + cur[:, MAST_CH.index("magnetics_p2l_current")]
    tf = cur[:, MAST_CH.index("magnetics_tf_current")]
    np.savez(W / "inputs.npz", shot=shot[:N], fidx=fidx[:N], t_ms=t_ms[:N], Iext=Iext, Ip=Ip, p2=p2, tf=tf,
             coils=np.array(MAST_COILS), n_coils=len(MAST_COILS), stems=np.array(ds["info"]["order"]))
    g = machine_geometry(config)
    np.savez(W / "grid.npz", gR=g["grid_R"], gZ=g["grid_Z"])
    base_arr = np.load(W / "base.npy", mmap_mode="r")
    return dict(N=N, base_sha256=sha256_arr(base_arr), passive=str(passive), passive_sha256=sha256_file(passive),
                **src)


def assertions(q, post, li):
    """Spec §7 (label-free, per fold): problems list (empty = pass)."""
    A, bad = q["asserts"], []
    v = post.get("veto_primary") or {}
    if v.get("lcfs_found", 0) < A["lcfs_found_min"]:
        bad.append(f"LCFS found {v.get('lcfs_found')} < {A['lcfs_found_min']}")
    if v.get("inside_wall", 0) < A["inside_wall_min"]:
        bad.append(f"inside-wall {v.get('inside_wall')} < {A['inside_wall_min']}")
    if not v.get("ip_err_median", np.inf) <= A["ip_err_median_max"]:
        bad.append(f"map-Ip error median {v.get('ip_err_median')} > {A['ip_err_median_max']}")
    if not v.get("absZ_gt_0p3", np.inf) <= A["absZ_gt_0p3_max"]:
        bad.append(f"|Z_ax| > 0.3 m share {v.get('absZ_gt_0p3')} > {A['absZ_gt_0p3_max']}")
    if v.get("nonfinite_frames", 1) > A["nonfinite_max"]:
        bad.append(f"non-finite frames {v.get('nonfinite_frames')}")
    ax = (post.get("axis_veto_a67") or {}).get("primary") or {}
    for k in ("R_axis", "Z_axis"):
        r = ax.get(k) or {}
        med, big = r.get("median_cm", np.nan), r.get("frac_gt3cm", np.nan)
        if not (abs(med) <= A["axis_median_cm_max"] and big <= A["axis_frac_gt3cm_max"]):
            bad.append(f"a67 {k}: median {med} cm, |d| > 3 cm share {big}")
    if not li.get("coverage", 0) >= A["li_coverage_min"]:
        bad.append(f"li coverage {li.get('coverage')} < {A['li_coverage_min']}")
    if not li.get("frac_inrange_syn", 0) >= A["li_inrange_min"]:
        bad.append(f"li in-range {li.get('frac_inrange_syn')} < {A['li_inrange_min']}")
    clip = (li.get("clip") or {}).get("share_of_finite", 1)
    if not clip <= A["li_clamp_max"]:
        bad.append(f"li clip share {clip} > {A['li_clamp_max']}")
    return bad


def li_clip(W):
    """Share of the finite li estimates clipped by the frozen calibration (fec.betap_li.predict_reg clips li_cd to the
    model's [lo, hi]; cdiff/frozen.json models.alone), with the clip's input range and output bounds.  The frozen li
    tool's own frac_clamp counts frames at 0.625 / 1.665, which are not the calibration's bounds (EVIDENCE C58), and
    does not gate."""
    from fec import betap_li as BL
    m = json.loads((WORK / "cdiff" / "frozen.json").read_text())["models"]["alone"]
    with np.load(W / "li.npz") as lz:
        cyl = np.asarray(lz["li_cyl"], float)
    fin = np.isfinite(cyl)
    _, inr = BL.predict_reg(m, {"li_cd": cyl})
    lo, hi = float(m["lo"][0]), float(m["hi"][0])
    out = BL.predict_reg(m, {"li_cd": np.array([lo, hi])})[0]
    n = max(int(fin.sum()), 1)
    return dict(share_of_finite=float((fin & ~inr).sum() / n), below=float((cyl[fin] < lo).sum() / n),
                above=float((cyl[fin] > hi).sum() / n), li_cd_range=[lo, hi],
                output_bounds=[float(out[0]), float(out[1])])


def write_meta(psi_path, out, bs=2048):
    """meta.npz of the written maps (fec.mast_tsm.solver_meta_fn on the CPU, as the tsm node's; validation only)."""
    from fec.mast_tsm import solver_meta_fn
    P = np.load(psi_path, mmap_mode="r")
    N = P.shape[0]
    fn = solver_meta_fn("cpu")
    meta = {k: np.full(N, np.nan) for k in META_KEYS}
    meta["method"] = np.ones(N)
    for a in range(0, N, bs):
        r = fn(np.asarray(P[a:min(N, a + bs)], np.float32))
        for k in META_KEYS:
            meta[k][a:min(N, a + bs)] = r[k]
    np.savez(out, **meta)


def copy_atomic(src, dst):
    tmp = Path(str(dst) + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def run(a):
    """The stage.  Returns the report dict.  A fold whose label-free assertions fail keeps the parent maps (reported,
    fallback True, status ok).  Any other exception -- the code / settings check, prep, the li step, stage L / P, the
    expansion, the post step and its evaluation, a child process's non-zero exit, a memory or file error, writing the
    node maps -- is recorded in nrsl.json (status error), leaves no node maps and is re-raised, so the process exits
    non-zero and fec/run_final.py stops the build."""
    t00 = time.time()
    out = Path(a.out)
    W = out / "work"
    if W.exists():                         # a re-run starts clean (stale stage / base files must never be reused)
        shutil.rmtree(W)
    for f in ("psi.npy", "meta.npz", "nrsl.json", "config.json"):
        (out / f).unlink(missing_ok=True)
    W.mkdir(parents=True, exist_ok=True)
    logf = open(out / "nrsl.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    rep = dict(stage="mast_nrsl", status="running", config=a.config, recipe=None, out=str(out), fallback=None,
               reasons=[], host=platform.node(), python=platform.python_version(), numpy=np.__version__,
               procs=int(a.procs), workers=int(a.workers), seconds={}, li_override=a.li or None,
               limit_shots=int(a.limit_shots or 0))
    t = time.time()

    def lap(k):
        nonlocal t
        rep["seconds"][k] = round(time.time() - t, 1)
        t = time.time()

    def drop_node():
        for f in ("psi.npy", "meta.npz", "psi.npy.part", "meta.npz.part"):
            (out / f).unlink(missing_ok=True)
    try:
        recipe = json.loads(Path(a.recipe).read_text())
        q = recipe["mast"]["composite"]["nrsl"]
        rep["recipe"] = recipe.get("name")
        env = stage_env(q)
        os.environ.update(env)
        rep["env"] = {k: os.environ.get(k) for k in env}
        import torch
        rep["torch"] = torch.__version__
        try:
            rep["cpu"] = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo")
                               if ln.startswith("model name")), None) if os.path.exists("/proc/cpuinfo") else \
                platform.processor()
        except OSError:
            rep["cpu"] = None
        bad = check_code(q)
        rep["code_check"] = bad or "ok"
        if bad:
            raise RuntimeError(f"code / settings check failed: {bad}")
        rep["prep"] = prep(W, a.config, a.maps, a.passive, a.base_npz, a.limit_shots)
        log(f"prep {rep['prep']['N']} frames base sha256 {rep['prep']['base_sha256'][:16]}")
        lap("prep")
        if a.li:
            shutil.copyfile(a.li, W / "li.npz")
            lz = dict(np.load(W / "li.npz"))
            if len(lz["li"]) != rep["prep"]["N"]:
                raise ValueError(f"--li {a.li}: {len(lz['li'])} frames, fold {rep['prep']['N']}")
            from fec.nrsl_li import SYN_RANGE as SY
            fin = np.isfinite(lz["li_est_raw"])
            lij = dict(override=str(a.li), sha256=sha256_file(a.li), coverage=float(fin.mean()),
                       filled=float(lz["filled"].mean()),
                       frac_inrange_syn=float(np.mean((lz["li"] >= SY[0]) & (lz["li"] <= SY[1]))))
            (W / "li.json").write_text(json.dumps(lij, indent=1))
        else:
            sh([PY, "-B", SCRIPTS["li"], "--fold-dir", W, "--config", a.config, "--nproc", a.workers,
                "--device", q["device"]], log, env)
        rep["li"] = json.loads((W / "li.json").read_text())
        rep["li"]["clip"] = li_clip(W)
        with np.load(W / "li.npz") as lz:
            rep["li_sha256"] = sha256_arr(lz["li"])
        lap("li")
        fL = run_stage(W, "L", None, q, int(a.procs), log, env)
        with np.load(fL) as zL:
            rep["sha256_stage_L"] = sha256_arr(zL["psi"])
            useL = zL["use"]
        lfail = np.flatnonzero(~useL)
        np.save(W / "fr_Lfail_cpu.npy", lfail)
        rep["stage_L_use"] = float(np.mean(useL))
        log(f"stage L use {rep['stage_L_use']:.4f}; L fallback frames {lfail.size}")
        lap("stage_L")
        if lfail.size:
            fP = run_stage(W, "P", lfail, q, int(a.procs), log, env)
            with np.load(fP) as zP:
                rep["sha256_stage_Psub"] = sha256_arr(zP["psi"])
        lap("stage_P")
        sh([PY, "-B", SCRIPTS["expand"], W / "stage_Psub_cpu.npz", W / "base.npy", W / "stage_P_cpu.npz"], log, env)
        post_cmd = [PY, "-B", SCRIPTS["post"], "--fold-dir", W, "--tag", "_cpu", "--nproc", a.workers]
        sh(post_cmd + (["--skip-eval"] if a.skip_eval else []), log, env)
        post = json.loads((W / "post_cpu.json").read_text())
        with np.load(W / "casc_cpu.npz") as C:
            rep["sha256_cascade"] = sha256_arr(C["psi"])
            rep["stage_counts"] = {"L": int((C["src"] == 0).sum()), "P": int((C["src"] == 1).sum()),
                                   "base": int((C["src"] == -1).sum())}
        prim = np.load(W / "primary_cpu.npy", mmap_mode="r")
        rep["sha256_primary"] = sha256_arr(prim)
        rep["primary_finite"] = bool(np.isfinite(prim).all())
        del prim
        rep["post"] = {k: post.get(k) for k in ("stage_counts", "sha256", "jitter_rms", "veto_base", "veto_casc",
                                                 "veto_primary", "axis_veto_a67", "classes")}
        log(f"stage counts L/P/base {rep['stage_counts']}; primary sha256 {rep['sha256_primary'][:16]}")
        lap("post")
        if a.skip_eval:
            rep.update(status="ok", fallback=None, reasons=["--skip-eval: no assertions, no node maps written"])
        else:
            bad = assertions(q, post, rep["li"])
            if not rep["primary_finite"]:
                bad.append("primary has non-finite values")
            rep["assertions"] = bad or "pass"
            rep["fallback"] = bool(bad)       # the ONLY fallback: the fold's label-free assertions failed
            rep["reasons"] = bad
            # the node's maps: the cascade (primary) if the fold passed, else the parent config's maps unchanged
            t = time.time()
            if not rep["fallback"]:
                copy_atomic(W / "primary_cpu.npy", out / "psi.npy")
                write_meta(out / "psi.npy", out / "meta.npz")
                log("NRSL_PASS: node maps = cascade (primary)")
            else:
                if a.base_npz or a.limit_shots:
                    log("fallback without the parent maps (checks run): no node maps written")
                else:
                    copy_atomic(Path(a.maps) / "psi.npy", out / "psi.npy")
                    copy_atomic(Path(a.maps) / "meta.npz", out / "meta.npz")
                log(f"NRSL_FALLBACK: the fold's label-free assertions failed; node maps = parent maps; reasons "
                    f"{rep['reasons']}")
            rep["seconds"]["node"] = round(time.time() - t, 1)
            if (out / "psi.npy").exists():
                rep["psi_sha256"] = sha256_file(out / "psi.npy")
            rep["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - recorded, no node maps, re-raised: the build stops with the cause
        drop_node()
        rep.update(status="error", fallback=None, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-3000:])
        rep.pop("psi_sha256", None)
        rep["seconds"]["total"] = round(time.time() - t00, 1)
        (out / "nrsl.json").write_text(json.dumps(rep, indent=1, default=float))
        log(f"NRSL_ERROR {type(e).__name__}: {e} (no node maps written; the build must stop)")
        logf.close()
        raise
    rep["seconds"]["total"] = round(time.time() - t00, 1)
    (out / "nrsl.json").write_text(json.dumps(rep, indent=1, default=float))
    if not a.skip_eval:
        (out / "config.json").write_text(json.dumps(dict(
            kind="NRS-L cascade of the composite maps (fec/mast_nrsl.py)", config=a.config,
            N=rep.get("prep", {}).get("N"), maps=str(a.maps), fallback=rep["fallback"], psi_sha256=rep.get("psi_sha256"),
            base_file_sha256=rep.get("prep", {}).get("maps_psi_sha256"), seconds=rep["seconds"]["total"],
            code="fec/mast_nrsl.py"), indent=1))
    log(f"NRSL_DONE status ok fallback={rep['fallback']} total {rep['seconds']['total']:.0f}s")
    logf.close()
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--recipe", required=True, help="recipe JSON with mast.composite.nrsl")
    r.add_argument("--maps", default=None, help="dir of the composite maps (psi.npy, meta.npz)")
    r.add_argument("--base-npz", default=None, help="base maps from a submission file's shot_*_psirz (checks)")
    r.add_argument("--passive", required=True, help="the maps' C1 passive currents I.npy (N, 386)")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=6, help="pool size of the li estimator and the vetoes")
    r.add_argument("--procs", type=int, default=1, help="operator processes (contiguous whole-batch runs)")
    r.add_argument("--li", default=None, help="use this li.npz as the li target (checks)")
    r.add_argument("--limit-shots", type=int, default=0, help="first K shots only (smoke tests)")
    r.add_argument("--skip-eval", action="store_true", help="stage files + cascade + primary only (no vetoes, "
                                                            "no node maps; determinism checks)")
    a = ap.parse_args(argv)
    if bool(a.maps) == bool(a.base_npz):
        ap.error("give exactly one of --maps / --base-npz")
    try:
        run(a)
    except Exception as e:  # noqa: BLE001 - exit non-zero with the cause (the traceback is in nrsl.json and below)
        traceback.print_exc()
        print(f"NRSL_FAILED {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1
    return 0      # cascade written, or the fold's assertions failed and the parent maps were kept (logged)


if __name__ == "__main__":
    sys.exit(main())
