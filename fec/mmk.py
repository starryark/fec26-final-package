"""Member-mean kappa finishing ("mmk") of the DIII-D ensemble psi map: a label-free post-process on the TRAIN envelope.

Recipes with a d3d.mmk block (final_MTfull_mmk) run it in two places of fec/run_final.py:
  d3d_b    while the ensemble members are predicted (run_final.predict_b_chunked), MemberKappa computes, frame by frame,
           the elongation kappa of every primary member's own psi map (the no-Thomson fallback models are not members)
           and the stage writes <preds>/<b name>/<config>_mmk_kappa.npy: kappa (N, M) float64 in recipe member order,
           with <config>_mmk_kappa.json (tags, envelope / operator SHA-256, the stage's run id, the SHA-256 of its inputs
           and of the ensemble it wrote).  No member map is stored.  Any error in this part (setup, a worker that dies or
           stalls, close / save) leaves the ensemble files as they are and writes no kappa file.
  d3d_mmk  after the off-span decoder and before lirep4 (CPU): `python -B fec/mmk.py apply` edits the post-decoder
           ensemble map frame by frame with the kappa step of a frozen operator file (fec/mmk_op.py) toward the member
           mean T = nanmean over the members of kappa (equal weights).  The scalar file (q95, betaN, ...) is copied byte
           for byte; every other file of the source directory is copied unchanged.

Functional.  kappa is the challenge scorer's own LCFS extraction and shape read-out (fusion_scoring lcfs / derive,
through the operator's EnvFunctional, with the bit-identical helpers of fec/cons_fast.py) evaluated with the TRAIN-only
envelope ds_b/diii_d_train/env_train.npz (frozen/d3d/env_train.npz in the package; SHA-256 ENV_TRAIN_SHA256), never the
scorer's mask.  Inputs only: the predicted maps.  No label, no scorer mask and no other frame is read: each frame is
edited from its own map and its own target alone, the same way on every split.

Operator: a frozen file registered in OPERATORS (module name -> SHA-256); fec/mmk_op.py (alpha 0.5) is the default.
Each registered file is fec/mmk_op.py with at most the ALPHA line changed (`make-operator` writes one; tests check it).
Only its kappa step `_step(psi32, T, func, "k")` is called (its delta step is never used).  Radial compression /
expansion about the magnetic axis, weight 1 inside the separatrix tapering smoothly to 0 at psi_N 1.4, localised in Z
by Gaussians (sigma 0.25 m) at the heights of the inboard and outboard extremes; the factor c is found by bisection on
[0.9, 1.1] so that kappa moves the fraction alpha of the way to T, and kappa of the written float32 map is re-checked.
Skip rule (the frame stays bit-identical): no admissible root in the bracket, a realised move of the wrong sign, an
envelope extraction failure on the original or the edited map, or a target that is not finite.  Before the operator:
fallback frames (status 10 "fallback_frame": the ensemble's no-Thomson fallback replaced the members' mean there;
predict_b_chunked writes those rows to <config>_mmk_fallback_rows.npy, the same input-derived rule the ensemble uses)
and frames with fewer than --min-members finite member kappa (status 11 "too_few_members") are not edited and nothing
is evaluated on them.

Worker pools are spawn ProcessPoolExecutors with bounded look-ahead: a worker that dies (BrokenProcessPool), raises, or
lets no task finish for --stall-timeout seconds (PoolStall) ends the mmk part with an exception; the caller falls back.

CLI (the output does not depend on --nproc):
  python -B fec/mmk.py kappa --maps a.npy[,b.npy ...] --tags t1[,t2 ...] --out <file>.npy --env env_train.npz [--nproc N]
        member kappa of stored (N, 65, 65) float32 member maps (the d3d_b hook's functional on files)
  python -B fec/mmk.py apply --src DIR --config CFG --kappa <file>.npy --out DIR --env env_train.npz
        [--skip-rows <cfg>_mmk_fallback_rows.npy] [--min-members K] [--operator mmk_op] [--stall-timeout S] [--nproc N]
        <out>/<cfg>_psi.npy (edited), the other files of src copied, <cfg>_mmk_flags.npz, <cfg>_mmk_meta.json
  python -B fec/mmk.py check [--operator mmk_op]      the operator file and constants are the frozen ones
  python -B fec/mmk.py make-operator --alpha A --out fec/mmk_op_aNNN.py
        a new operator file = fec/mmk_op.py with only ALPHA changed; prints its SHA-256 and the registration steps
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec import mmk_op as OP  # noqa: E402  (the default frozen operator; variants are imported by name)
from fec.paths import SCORING  # noqa: E402

FEC_DIR = Path(__file__).resolve().parent
OP_DEFAULT = "mmk_op"
# Frozen operator files (module name in fec/ -> SHA-256 of its bytes).  One file per alpha; every file is fec/mmk_op.py
# with only its ALPHA line changed (`make-operator`).  A new entry also needs a static `from fec import <name>` line
# below (the package builder ships the import closure) and a .gitattributes `text eol=lf` rule (the hash must hold in
# every checkout).
OPERATORS = {"mmk_op": "5592926f8909e0dfaafefca6fb03f2a9569a95f97ef46f1f9d1ec7a496b9cdad"}
OP_FILE = Path(OP.__file__).resolve()
OP_SHA256 = OPERATORS[OP_DEFAULT]
ENV_TRAIN_SHA256 = "fc4ce12f0fc3e36c4ce8bcacaac4bbe62d3410b851477ad9ee1ee0f55e3d8505"
# The operator constants a recipe's d3d.mmk block states; check_frozen() compares them with the module's own values.
# Every constant but alpha is the same for every registered operator.
FROZEN_BASE = {"sigma_z_m": 0.25, "psin_taper": [1.0, 1.4], "c_bracket": [0.9, 1.1], "maxit": 30, "rtol": 1e-4,
               "atol": 1e-6, "lcfs_n_iter": 22}
FROZEN = dict(FROZEN_BASE, alpha=0.5)                   # the default operator's
FLAG_KEYS = ("k_status", "k_c", "k_f0", "k_T", "k_target", "k_solved", "k_real", "k_nmid", "R_axis0", "R_axis1",
             "n_calls")
NPIX = 65 * 65
FALLBACK_FRAME = 10         # a frame the ensemble's no-Thomson fallback wrote: never edited
TOO_FEW_MEMBERS = 11        # a frame with fewer than min_members finite member kappa: never edited
STATUS = {**OP.STATUS, FALLBACK_FRAME: "fallback_frame", TOO_FEW_MEMBERS: "too_few_members"}
SKIP_CLASS = {**OP.SKIP_CLASS, FALLBACK_FRAME: "fallback_frame", TOO_FEW_MEMBERS: "too_few_members"}
KAPPA_BLOCK = 256           # member rows per kappa task
APPLY_CHUNK = 64            # ensemble frames per apply task
STALL_TIMEOUT_S = 600.0     # no worker task finished for this long -> PoolStall (a typical task takes 1-15 s)
_W = {}


class PoolStall(RuntimeError):
    """No worker task finished within the stall timeout."""


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=_jsonable), encoding="utf-8")


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))


# ------------------------------------------------------------------------------------------------ frozen operator
def operator_module(name=OP_DEFAULT):
    """The registered frozen operator module fec.<name> (ValueError for an unregistered name)."""
    if name not in OPERATORS:
        raise ValueError(f"operator {name!r} is not a registered frozen operator {sorted(OPERATORS)}")
    return OP if name == OP_DEFAULT else importlib.import_module(f"fec.{name}")


def op_constants(op=None):
    """The kappa step's constants as an operator module defines them (FROZEN's keys)."""
    op = op or OP
    return {"alpha": op.ALPHA, "sigma_z_m": op.SIGMA_Z, "psin_taper": list(op.PSIN_TAPER),
            "c_bracket": list(op.C_BRACKET), "maxit": op.MAXIT, "rtol": op.RTOL, "atol": op.ATOL,
            "lcfs_n_iter": op.N_ITER}


def alpha_only_variant(name):
    """True iff fec/<name>.py is fec/mmk_op.py with at most its ALPHA line changed (line for line)."""
    a = (FEC_DIR / f"{OP_DEFAULT}.py").read_bytes().split(b"\n")
    b = (FEC_DIR / f"{name}.py").read_bytes().split(b"\n")
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return len(a) == len(b) and all(a[i].startswith(b"ALPHA = ") and b[i].startswith(b"ALPHA = ") for i in diff)


def check_frozen(block=None, op_name=None):
    """Problems of an operator against the registry (file SHA-256, alpha-only variant of fec/mmk_op.py, constants) and,
    if given, a recipe d3d.mmk block (its operator, operator_sha256, alpha, env_train_sha256, step and the other
    constants).  The operator is op_name, else the block's `operator` file stem, else the default.  [] = frozen."""
    bad = []
    name = op_name or (Path(block["operator"]).stem if block is not None and block.get("operator") else OP_DEFAULT)
    if name not in OPERATORS:
        return [f"operator {name!r} is not registered {sorted(OPERATORS)}"]
    f = FEC_DIR / f"{name}.py"
    h = sha256(f) if f.is_file() else None
    if h != OPERATORS[name]:
        bad.append(f"fec/{name}.py sha256 {h} != {OPERATORS[name]}")
        return bad
    if name != OP_DEFAULT and not alpha_only_variant(name):
        bad.append(f"fec/{name}.py differs from fec/{OP_DEFAULT}.py beyond its ALPHA line")
    c = op_constants(operator_module(name))
    if {k: c[k] for k in FROZEN_BASE} != FROZEN_BASE:
        bad.append(f"fec/{name}.py constants {c} != {FROZEN_BASE} (+ alpha)")
    if block is not None:
        if block.get("operator_sha256") != OPERATORS[name]:
            bad.append(f"recipe operator_sha256 {block.get('operator_sha256')} != {OPERATORS[name]}")
        if block.get("alpha") != c["alpha"]:
            bad.append(f"recipe alpha {block.get('alpha')!r} != fec/{name}.py ALPHA {c['alpha']!r}")
        if block.get("env_train_sha256") != ENV_TRAIN_SHA256:
            bad.append(f"recipe env_train_sha256 {block.get('env_train_sha256')} != {ENV_TRAIN_SHA256}")
        if block.get("step") != "kappa":
            bad.append(f"recipe step {block.get('step')!r} != 'kappa'")
        for k, v in FROZEN_BASE.items():
            if block.get(k) != v:
                bad.append(f"recipe {k} {block.get(k)!r} != {v!r}")
    return bad


def make_operator(alpha, out):
    """Write `out` = fec/mmk_op.py with only `ALPHA = 0.5` replaced by `ALPHA = <alpha>`; returns its SHA-256.  The new
    file is not registered: add it to OPERATORS (+ a static import line, + .gitattributes) and to the recipe."""
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0) or alpha == FROZEN["alpha"]:
        raise ValueError(f"alpha must be in (0, 1] and differ from {FROZEN['alpha']}: {alpha}")
    src = (FEC_DIR / f"{OP_DEFAULT}.py").read_bytes()
    old = b"ALPHA = 0.5\n"
    if src.count(old) != 1:
        raise ValueError("fec/mmk_op.py has no single 'ALPHA = 0.5' line")
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists (never overwrite)")
    out.write_bytes(src.replace(old, f"ALPHA = {alpha!r}\n".encode()))
    return sha256(out)


# ------------------------------------------------------------------------------------------------ workers
def _init_worker(env_path, scoring_dir, op_name=OP_DEFAULT):
    """Worker state: the scorer on sys.path, the bit-identical fast helpers, the operator, the TRAIN-envelope
    functional."""
    sd = str(scoring_dir)
    if sd not in sys.path:
        sys.path.insert(0, sd)
    from fec.cons_fast import patch
    patch()
    op = operator_module(op_name)
    R, Z, mask = op.load_env(env_path)
    _W["op"] = op
    _W["func"] = op.EnvFunctional(R, Z, mask, scoring_dir)


def frame_kappa(func, row):
    """kappa (TRAIN envelope) of one float32 map row ((4225,) or (65, 65)); NaN on extraction failure."""
    return func(np.asarray(row, np.float64).reshape(func.Z.size, func.R.size))["kappa"]


def _kappa_task(args):
    key, rows = args
    func = _W["func"]
    k = np.empty(len(rows))
    for i in range(len(rows)):
        k[i] = frame_kappa(func, rows[i])
    return key, k


def edit_frame(func, row32, T, op=None):
    """The frozen kappa step on one float32 map: (float32 (65, 65) map, flags dict of FLAG_KEYS)."""
    op = op or _W.get("op") or OP
    c0 = func.ncalls
    psi32 = np.asarray(row32, np.float32).reshape(func.Z.size, func.R.size)
    a, f = op._step(psi32, T, func, "k")
    fl = {"k_status": f["status"], "k_c": f["x"], "k_f0": f["f0"], "k_T": f["T"], "k_target": f["target"],
          "k_solved": f["solved"], "k_real": f["real"], "k_nmid": f["nmid"],
          "R_axis0": f["info0"]["R_axis"] if f.get("info0") is not None else np.nan,
          "R_axis1": f["info1"]["R_axis"] if (f["status"] == 0 and f.get("info1") is not None) else np.nan,
          "n_calls": func.ncalls - c0}
    return a, fl


def _apply_task(args):
    """(key, rows, T, code): code 0 = edit; FALLBACK_FRAME / TOO_FEW_MEMBERS = copy the row, nothing evaluated."""
    key, rows, T, code = args
    func = _W["func"]
    out = np.empty((len(rows), func.Z.size, func.R.size), np.float32)
    fl = {k: np.full(len(rows), np.nan) for k in FLAG_KEYS}
    for i in range(len(rows)):
        if code[i]:
            out[i] = np.asarray(rows[i], np.float32).reshape(func.Z.size, func.R.size)
            fl["k_status"][i], fl["n_calls"][i] = code[i], 0
            continue
        out[i], f = edit_frame(func, rows[i], T[i])
        for k in FLAG_KEYS:
            fl[k][i] = f[k]
    return key, out, fl


def make_pool(nproc, env_path, scoring_dir=SCORING, op_name=OP_DEFAULT):
    """A spawn ProcessPoolExecutor (safe in a process that already holds CUDA) of nproc workers with the
    TRAIN-envelope functional.  A worker that dies makes every pending result raise BrokenProcessPool."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=int(nproc), mp_context=mp.get_context("spawn"), initializer=_init_worker,
                               initargs=(str(env_path), str(scoring_dir), op_name))


def kill_pool(ex):
    """End a pool now, hung workers included (best effort, never raises)."""
    for p in list((getattr(ex, "_processes", None) or {}).values()):
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001
        pass


def bounded_map(ex, fn, tasks, depth, stall_timeout=STALL_TIMEOUT_S):
    """Results of fn over tasks in completion order, at most `depth` tasks submitted ahead.  Raises PoolStall when no
    task finishes for stall_timeout seconds, BrokenProcessPool when a worker died, and a worker's own exception."""
    from concurrent.futures import FIRST_COMPLETED, wait
    it = iter(tasks)
    pending = set()

    def fill():
        while len(pending) < depth:
            try:
                t = next(it)
            except StopIteration:
                return
            pending.add(ex.submit(fn, t))
    fill()
    while pending:
        done, rest = wait(pending, timeout=stall_timeout, return_when=FIRST_COMPLETED)
        if not done:
            raise PoolStall(f"no worker task finished within {stall_timeout:g} s")
        pending.clear()
        pending.update(rest)
        for f in done:
            yield f.result()
        fill()


# ------------------------------------------------------------------------------------------------ member kappa
class MemberKappa:
    """kappa (TRAIN envelope) of every frame of every primary member's own psi map, filled block by block while the
    members are predicted (add() returns when its rows are done).  close() ends the pool; save() writes the array.
    terminate() ends it at once (after an error)."""

    task = staticmethod(_kappa_task)

    def __init__(self, n_frames, tags, env_path, nproc, scoring_dir=SCORING, block=KAPPA_BLOCK, log=None,
                 stall_timeout=None):
        self.tags = [str(t) for t in tags]
        self.kappa = np.full((int(n_frames), len(self.tags)), np.nan)
        self.done = np.zeros(self.kappa.shape, bool)
        self.env_path, self.nproc, self.block = Path(env_path), int(nproc), int(block)
        self.stall_timeout = float(stall_timeout or STALL_TIMEOUT_S)
        self.env_sha256 = sha256(self.env_path)
        self.log = log or (lambda m: None)
        self.t0 = time.time()
        self.pool = make_pool(self.nproc, self.env_path, scoring_dir)

    def add(self, m, s0, P):
        """P: rows s0 .. s0 + len(P) of member m ((n, 4225) or (n, 65, 65) float32)."""
        P = np.asarray(P, np.float32).reshape(len(P), NPIX)
        tasks = (((int(m), int(s0) + a), P[a:a + self.block]) for a in range(0, len(P), self.block))
        for (mm, r0), k in bounded_map(self.pool, self.task, tasks, 4 * self.nproc, self.stall_timeout):
            self.kappa[r0:r0 + len(k), mm] = k
            self.done[r0:r0 + len(k), mm] = True

    def close(self):
        self.pool.shutdown(wait=True)

    def terminate(self):
        kill_pool(self.pool)

    def meta(self):
        return {"tags": self.tags, "shape": list(self.kappa.shape), "dtype": "float64",
                "complete": bool(self.done.all()), "nan_per_member": [int(x) for x in np.isnan(self.kappa).sum(0)],
                "finite_members_min": int(np.isfinite(self.kappa).sum(1).min()) if len(self.kappa) else None,
                "env_train": str(self.env_path), "env_train_sha256": self.env_sha256,
                "mmk_op_sha256": sha256(OP_FILE), "functional": "fec/mmk_op.EnvFunctional kappa (TRAIN envelope)",
                "nproc": self.nproc, "stall_timeout_s": self.stall_timeout, "seconds": round(time.time() - self.t0, 1)}

    def save(self, path, extra=None):
        """<path> (.npy, kappa (N, M) float64) + <path>.json (meta, + extra); refuses an incomplete array.  The json is
        written last: a kappa file without its json is incomplete."""
        if not self.done.all():
            raise ValueError(f"member kappa incomplete: {int((~self.done).sum())} cells missing")
        path = Path(path)
        tmp = path.with_name(path.stem + ".part.npy")
        np.save(tmp, self.kappa)
        os.replace(tmp, path)
        meta = dict(self.meta(), sha256=sha256(path), **(extra or {}))
        write_json(kappa_meta_path(path), meta)
        return meta


def kappa_meta_path(path):
    path = Path(path)
    return path.with_name(path.stem + ".json")


def member_targets(kappa):
    """T = equal NaN-ignoring mean over members (axis 1) of kappa (N, M); NaN where every member is NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(kappa, np.float64), 1)


def kappa_of_maps(maps, tags, out, env_path, nproc=6, rows_per_add=32768, log=print, stall_timeout=None):
    """Member kappa of stored (N, 65, 65) / (N, 4225) float32 member maps, through MemberKappa."""
    arrs = [np.load(m, mmap_mode="r") for m in maps]
    N = arrs[0].shape[0]
    if any(a.shape[0] != N for a in arrs) or len(tags) != len(arrs):
        raise SystemExit(f"maps / tags disagree: {[a.shape for a in arrs]} {tags}")
    mk = MemberKappa(N, tags, env_path, nproc, log=log, stall_timeout=stall_timeout)
    try:
        for m, A in enumerate(arrs):
            for s0 in range(0, N, rows_per_add):
                mk.add(m, s0, np.asarray(A[s0:s0 + rows_per_add], np.float32))
            log(f"member {m + 1}/{len(arrs)} {tags[m]} {time.time() - mk.t0:.0f}s")
        mk.close()
    except BaseException:
        mk.terminate()
        raise
    return mk.save(out)


# ------------------------------------------------------------------------------------------------ apply
def apply(src, out, config, kappa, env_path, nproc=6, chunk=APPLY_CHUNK, skip_rows=None, scoring_dir=SCORING,
          log=print, min_members=0, op_name=OP_DEFAULT, stall_timeout=STALL_TIMEOUT_S, task=_apply_task):
    """<out>/<config>_psi.npy = the frozen kappa step of operator op_name on every frame of <src>/<config>_psi.npy toward
    T = member_targets (kappa file), except the rows listed in skip_rows (.npy of int rows: the ensemble's fallback
    frames) and the frames with fewer than min_members finite member kappa, which are copied bit for bit; every other
    file of src copied unchanged; flags + meta written.  A worker failure or stall raises (the output psi is then not
    written).  Returns the meta dict."""
    t0 = time.time()
    bad = check_frozen(op_name=op_name)
    if bad:
        raise SystemExit(f"mmk operator is not a frozen one: {bad}")
    op = operator_module(op_name)
    src, out = Path(src), Path(out)
    sp = src / f"{config}_psi.npy"
    if (out / f"{config}_psi.npy").exists():
        raise SystemExit(f"{out}/{config}_psi.npy exists (never overwrite)")
    psi = np.load(sp, mmap_mode="r")
    N = psi.shape[0]
    if psi.dtype != np.float32 or int(np.prod(psi.shape[1:])) != NPIX:
        raise SystemExit(f"{sp}: expected float32 (N, 65, 65), got {psi.dtype} {psi.shape}")
    K = np.load(kappa)
    if K.ndim != 2 or K.shape[0] != N:
        raise SystemExit(f"kappa {K.shape} does not match {N} frames")
    if not 0 <= int(min_members) <= K.shape[1]:
        raise SystemExit(f"min_members {min_members} outside [0, {K.shape[1]}]")
    T = member_targets(K)
    n_fin = np.isfinite(K).sum(1)
    skip = np.zeros(N, bool)
    if skip_rows is not None:
        rows_ = np.asarray(np.load(skip_rows), np.int64).ravel()
        if rows_.size and (rows_.min() < 0 or rows_.max() >= N or len(np.unique(rows_)) != rows_.size):
            raise SystemExit(f"{skip_rows}: rows must be unique and within [0, {N})")
        skip[rows_] = True
    few = (n_fin < int(min_members)) & ~skip
    code = np.where(skip, FALLBACK_FRAME, np.where(few, TOO_FEW_MEMBERS, 0)).astype(np.int64)
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in sorted(src.iterdir()):
        if f.is_file() and f.name != sp.name:
            shutil.copyfile(f, out / f.name)
            copied.append(f.name)
    tmp = out / f"{config}_psi.part.npy"
    o = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=psi.shape)
    flags = {k: np.full(N, np.nan) for k in FLAG_KEYS}
    # copies (np.array): no task may hold a view of the source memmap (a broken pool keeps its work items)
    tasks = ((s, np.array(psi[s:s + chunk], np.float32).reshape(-1, 65, 65), T[s:s + chunk], code[s:s + chunk])
             for s in range(0, N, chunk))
    pool = make_pool(nproc, env_path, scoring_dir, op_name)
    n_done, last = 0, time.time()
    try:
        for s, rows, fl in bounded_map(pool, task, tasks, 4 * int(nproc), stall_timeout):
            o[s:s + len(rows)] = rows.reshape((len(rows),) + psi.shape[1:])
            for k in FLAG_KEYS:
                flags[k][s:s + len(rows)] = fl[k]
            n_done += len(rows)
            if time.time() - last > 60:
                last = time.time()
                log(f"mmk apply {n_done}/{N} frames {time.time() - t0:.0f}s")
        pool.shutdown(wait=True)
    except BaseException:
        kill_pool(pool)
        del o
        psi = None              # drop the source memmap (the task generator shares this variable) before raising
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    o.flush()
    st = flags["k_status"].astype(int)
    changed = np.zeros(N, bool)
    finite = True
    for a in range(0, N, 8192):
        x = np.array(o[a:a + 8192]).reshape(-1, NPIX)          # copies: no view may keep the memmap open
        y = np.array(psi[a:a + 8192]).reshape(-1, NPIX)
        changed[a:a + len(x)] = (x.view(np.uint32) != y.view(np.uint32)).any(1)
        finite = finite and bool(np.isfinite(x).all())
    del o
    os.replace(tmp, out / f"{config}_psi.npy")
    flags["k_status"] = st
    np.savez(out / f"{config}_mmk_flags.npz", T_kappa=T, n_finite_members=n_fin, **flags)
    ok = st == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratio = (flags["k_real"] - flags["k_f0"]) / (flags["k_target"] - flags["k_f0"])
    qs = (0.01, 0.5, 0.99)
    meta = dict(
        src=str(src), out=str(out), config=config, frames=int(N), members=int(K.shape[1]),
        kappa=str(kappa), kappa_sha256=sha256(kappa), kappa_nan_per_member=[int(x) for x in np.isnan(K).sum(0)],
        T_nan=int(np.isnan(T).sum()), env_train=str(env_path), env_train_sha256=sha256(env_path),
        skip_rows=str(skip_rows) if skip_rows is not None else None,
        skip_rows_sha256=sha256(skip_rows) if skip_rows is not None else None,
        fallback_frames_not_edited=int(skip.sum()), min_members=int(min_members),
        too_few_members_frames=int(few.sum()),
        finite_members_hist={str(int(v)): int((n_fin == v).sum()) for v in np.unique(n_fin)},
        operator=op_name, mmk_op_sha256=sha256(FEC_DIR / f"{op_name}.py"), mmk_py_sha256=sha256(Path(__file__).resolve()),
        frozen=op_constants(op), stall_timeout_s=float(stall_timeout),
        status_counts={STATUS[s]: int((st == s).sum()) for s in sorted(STATUS)},
        skip_counts={c: int(np.isin(st, [s for s, v in SKIP_CLASS.items() if v == c]).sum())
                     for c in sorted(set(SKIP_CLASS.values()))},
        edited=int(ok.sum()), changed=int(changed.sum()), edited_rows_equal_ok_rows=bool(np.array_equal(changed, ok)),
        finite=finite, nan=not finite,
        realised_over_intended=({str(q): float(np.quantile(ratio[ok], q)) for q in qs} if ok.any() else {}),
        c_quantiles=({str(q): float(np.quantile(flags["k_c"][ok], q)) for q in qs} if ok.any() else {}),
        copied_unchanged=copied, nproc=int(nproc), chunk=int(chunk), seconds=round(time.time() - t0, 1))
    write_json(out / f"{config}_mmk_meta.json", meta)
    log(f"MMK_APPLY_DONE {out}: {N} frames, edited {meta['edited']}, status {meta['status_counts']}, "
        f"T_nan {meta['T_nan']}, too_few_members {meta['too_few_members_frames']}, {meta['seconds']:.0f}s")
    return meta


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kappa", help="member kappa of stored member maps")
    k.add_argument("--maps", required=True, help="comma list of (N, 65, 65) float32 .npy member maps")
    k.add_argument("--tags", required=True, help="comma list of member tags (same order)")
    k.add_argument("--out", required=True, help="output .npy (kappa (N, M) float64) + .json")
    k.add_argument("--env", required=True, help="TRAIN envelope env_train.npz")
    k.add_argument("--nproc", type=int, default=6, help="worker processes (default 6)")
    k.add_argument("--stall-timeout", type=float, default=STALL_TIMEOUT_S)
    a_ = sub.add_parser("apply", help="the frozen kappa step on an ensemble preds dir")
    a_.add_argument("--src", required=True)
    a_.add_argument("--out", required=True)
    a_.add_argument("--config", required=True)
    a_.add_argument("--kappa", required=True, help="member kappa .npy (d3d_b's <config>_mmk_kappa.npy)")
    a_.add_argument("--env", required=True, help="TRAIN envelope env_train.npz")
    a_.add_argument("--nproc", type=int, default=6, help="worker processes (default 6)")
    a_.add_argument("--chunk", type=int, default=APPLY_CHUNK)
    a_.add_argument("--skip-rows", default=None,
                    help=".npy of the rows the ensemble's no-Thomson fallback wrote (never edited)")
    a_.add_argument("--min-members", type=int, default=0,
                    help="frames with fewer finite member kappa are never edited (default 0: no minimum)")
    a_.add_argument("--operator", default=OP_DEFAULT, help=f"registered frozen operator {sorted(OPERATORS)}")
    a_.add_argument("--stall-timeout", type=float, default=STALL_TIMEOUT_S,
                    help="seconds without a finished worker task before the run stops (default %(default)s)")
    c_ = sub.add_parser("check", help="operator file and constants are the frozen ones")
    c_.add_argument("--operator", default=OP_DEFAULT)
    m_ = sub.add_parser("make-operator", help="write fec/mmk_op.py with only ALPHA changed (not registered)")
    m_.add_argument("--alpha", type=float, required=True)
    m_.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    if a.cmd == "check":
        bad = check_frozen(op_name=a.operator)
        print(json.dumps({"frozen": not bad, "problems": bad, "operator": a.operator,
                          "sha256": OPERATORS.get(a.operator)}))
        return 1 if bad else 0
    if a.cmd == "make-operator":
        h = make_operator(a.alpha, a.out)
        print(json.dumps({"out": a.out, "alpha": a.alpha, "sha256": h,
                          "next": ["add {name: sha256} to fec/mmk.py OPERATORS and a static `from fec import <name>`",
                                   "add `/fec/<name>.py text eol=lf` to .gitattributes",
                                   "set run_final MMK_KAPPA operator / operator_sha256 / alpha",
                                   "update the alpha in README / PROVENANCE (tests compare them with the recipe)"]}))
        return 0
    if a.cmd == "kappa":
        meta = kappa_of_maps(a.maps.split(","), a.tags.split(","), a.out, a.env, a.nproc, log=log,
                             stall_timeout=a.stall_timeout)
        log(f"MMK_KAPPA_DONE {a.out} {meta['shape']} nan {meta['nan_per_member']} {meta['seconds']:.0f}s")
        return 0
    apply(a.src, a.out, a.config, a.kappa, a.env, a.nproc, a.chunk, skip_rows=a.skip_rows, log=log,
          min_members=a.min_members, op_name=a.operator, stall_timeout=a.stall_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
