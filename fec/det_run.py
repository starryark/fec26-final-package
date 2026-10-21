"""Reproducibility wrapper and comparison tools (agent fpkg 2026-10-12, FINAL_PACKAGE_20261012 G3).

    python -B fec/det_run.py run [--strict] [--seed S] -- <script.py> [args ...]
    python -B fec/det_run.py cmp-models A/model.pt B/model.pt
    python -B fec/det_run.py cmp-arrays A.npy B.npy
    python -B fec/det_run.py env

`run` executes a training/inference script in this interpreter after fixing every global source
of run-to-run variation that the scripts themselves do not fix:
  * PYTHONHASHSEED is checked (set it in the parent; the hash seed of a running interpreter cannot
    change) and exported to child processes;
  * Python `random`, NumPy's legacy global RNG, torch CPU and every CUDA RNG are seeded with --seed
    (the scripts then reseed with their own member seeds: torch.manual_seed(seed) in
    fec/train_d3d_b.py / train_d3d_bx.py / amort_net.train);
  * cuDNN: deterministic, benchmark off;
  * TF32 OFF for matmul and cuDNN, float32 matmul precision 'highest' (our production runs never
    enabled TF32: train_d3d_bx enables it only with --tf32, which no Final job passes);
  * --strict: torch.use_deterministic_algorithms(True) (raises on any nondeterministic kernel) and
    CUBLAS_WORKSPACE_CONFIG=:4096:8 (required by cuBLAS for deterministic GEMMs on CUDA >= 10.2).
Data order inside the scripts is already seeded (per-member torch.Generator / torch.randperm under
the member seed; no DataLoader workers, no filesystem-order dependence: shot lists come from
info.json and sorted globs).  bf16 autocast stays as in production (it is part of the recipe).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def set_determinism(seed=0, strict=False):
    """Process-wide settings (see module doc).  Returns the applied settings."""
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if strict:
        torch.use_deterministic_algorithms(True)
    return env_report(strict=strict, seed=seed)


def env_report(strict=None, seed=None):
    import platform
    import torch
    r = dict(python=sys.version.split()[0], platform=platform.platform(),
             numpy=np.__version__, torch=torch.__version__,
             cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
             tf32_cudnn=torch.backends.cudnn.allow_tf32,
             cudnn_deterministic=torch.backends.cudnn.deterministic,
             deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
             cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
             pythonhashseed=os.environ.get("PYTHONHASHSEED"))
    if strict is not None:
        r.update(strict=strict, seed=seed)
    return r


def state_dict_digest(path):
    """{key: sha256 of the tensor bytes} of a torch checkpoint (nested dicts/lists flattened)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {}

    def walk(o, pre):
        if isinstance(o, torch.Tensor):
            t = o.detach().contiguous().cpu()
            out[pre] = hashlib.sha256(t.view(torch.uint8).numpy().tobytes()
                                      if t.numel() else b"").hexdigest() + f"|{t.dtype}|{tuple(t.shape)}"
        elif isinstance(o, dict):
            for k in sorted(o, key=str):
                walk(o[k], f"{pre}.{k}" if pre else str(k))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{pre}[{i}]")
    walk(ck, "")
    return out


def cmp_models(a, b):
    da, db = state_dict_digest(a), state_dict_digest(b)
    keys = sorted(set(da) | set(db))
    diff = [k for k in keys if da.get(k) != db.get(k)]
    res = dict(tensors=len(keys), identical=not diff, n_diff=len(diff), diff=diff[:10])
    if diff:
        import torch
        ca = torch.load(a, map_location="cpu", weights_only=False)
        cb = torch.load(b, map_location="cpu", weights_only=False)

        def get(o, k):
            import re
            for part in re.findall(r"[^.\[\]]+|\[\d+\]", k):
                o = o[int(part[1:-1])] if part.startswith("[") else o[part]
            return o
        mx = 0.0
        for k in diff:
            try:
                x, y = get(ca, k), get(cb, k)
                if hasattr(x, "float") and x.shape == y.shape:
                    mx = max(mx, float((x.float() - y.float()).abs().max()))
            except (KeyError, IndexError, TypeError):
                pass
        res["max_abs_diff"] = mx
    return res


def cmp_arrays(a, b, chunk=4096):
    A, B = np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")
    if A.shape != B.shape:
        return dict(identical=False, shape=(A.shape, B.shape))
    mx, nd = 0.0, 0
    for i in range(0, A.shape[0], chunk):
        x = np.asarray(A[i:i + chunk], np.float64); y = np.asarray(B[i:i + chunk], np.float64)
        d = np.abs(x - y)
        d[np.isnan(x) & np.isnan(y)] = 0
        mx = max(mx, float(np.nanmax(d)) if d.size else 0.0)
        nd += int((d > 0).sum())
    return dict(identical=nd == 0, max_abs_diff=mx, n_diff=nd, shape=list(A.shape))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        if "--" not in argv:
            raise SystemExit("usage: det_run.py run [--strict] [--seed S] -- script.py [args]")
        i = argv.index("--")
        ap = argparse.ArgumentParser()
        ap.add_argument("--strict", action="store_true")
        ap.add_argument("--seed", type=int, default=0)
        a = ap.parse_args(argv[1:i])
        script, sargs = argv[i + 1], argv[i + 2:]
        rep = set_determinism(a.seed, a.strict)
        print("DET_ENV " + json.dumps(rep), flush=True)
        sys.argv = [script] + sargs
        sys.path.insert(0, str(Path(script).resolve().parent))
        runpy.run_path(script, run_name="__main__")
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cmp-models"); p.add_argument("a"); p.add_argument("b")
    p = sub.add_parser("cmp-arrays"); p.add_argument("a"); p.add_argument("b")
    sub.add_parser("env")
    a = ap.parse_args(argv)
    if a.cmd == "cmp-models":
        print(json.dumps(cmp_models(a.a, a.b)))
    elif a.cmd == "cmp-arrays":
        print(json.dumps(cmp_arrays(a.a, a.b)))
    else:
        print(json.dumps(env_report()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
