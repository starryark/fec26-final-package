"""mast_solver2 (DESIGN §8): the frozen R1 operator (tools/ms_op.py, unchanged) on MAST, one stage per call.
Only the machine description (MAST V3Solver, f 0.55 coils + C1 r2 passive + uniform-Bz column) and I/O are new.

    python -B ms2_mast_run.py --fold-dir outm/public --stage L|P --out stage_L.npz [--device cuda] [--bs 256]
fold-dir holds base.npy (N,65,65 float32, axis-maximum), inputs.npz (Iext (N, nc+386) A, Ip A), li.npz (stage L).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
STAGES = dict(L=dict(rmode="pinb", kp=5.0, k_li=2.0, dv_clamp=0.010, kr=0.02, li=True),
              P=dict(rmode="pin", kp=3.0, k_li=2.0, dv_clamp=0.010, kr=0.02, li=False))


def make_solver_mast(device="cuda"):
    """V3Solver (limiter None, float64) on [coils f 0.55 | C1 passive pieces | RR^2/2]."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from fec.machines import mast_setup
    from fec.mast_select_v3 import V3Solver
    from fec.passive import PassiveModel
    spec, M = mast_setup(sol_fraction=0.55)
    Mp = PassiveModel().augment(M)
    RR, _ = np.meshgrid(spec.gR, spec.gZ)
    Mx = np.concatenate([Mp, (RR ** 2 / 2).reshape(-1, 1)], 1)
    return V3Solver(spec, Mx, device=device, dtype=torch.float64, limiter=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", required=True)
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--n_iter", type=int, default=150)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frames", default="", help="optional npy of frame indices (smoke tests)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from fec import nrsl_op as OP
    st = STAGES[a.stage]
    d = Path(a.fold_dir)
    base_all = np.load(d / "base.npy", mmap_mode="r")
    z = np.load(d / "inputs.npz")
    N = base_all.shape[0]
    fr = np.arange(N)
    if a.frames:
        fr = np.load(a.frames)
    if a.limit:
        fr = fr[:a.limit]
    Iext_all, Ip_all = z["Iext"], np.maximum(z["Ip"], 1e5)
    li_t = None
    if st["li"]:
        lz = np.load(d / "li.npz")
        assert len(lz["li"]) == N
        li_t = lz["li"].astype(np.float64)
    S = make_solver_mast(a.device)
    keys = ("gam", "li_fin", "lsp", "sp", "bet", "beta0", "last_dR", "last_dZ", "use", "conv", "xok", "inbox",
            "miss_ok", "good_base", "dI", "dv", "c", "Rt", "Zt", "Rc", "Zc", "fit_rel", "Ib_sum", "Rax", "Zax",
            "psi_ax", "psi_b", "coef")
    acc = {k: [] for k in keys}
    psi32 = np.zeros((len(fr), 65, 65), np.float32)
    h64 = hashlib.sha256()
    t0 = time.time()
    for s in range(0, len(fr), a.bs):
        ii = fr[s:s + a.bs]
        base = np.asarray(base_all[ii], np.float64)
        o = OP.operator(S, Iext_all[ii], Ip_all[ii], base, rmode=st["rmode"], n_iter=a.n_iter, kr=st["kr"],
                        li_target=None if li_t is None else li_t[ii], k_li=st["k_li"], dv_clamp=st["dv_clamp"],
                        kp=st["kp"])
        for k in keys:
            acc[k].append(np.asarray(o[k]))
        psi32[s:s + len(ii)] = o["psi_out"].astype(np.float32)
        h64.update(np.ascontiguousarray(o["psi_out"]).tobytes())
        if (s // a.bs) % 20 == 0:
            print(f"op {a.stage} {s + len(ii)}/{len(fr)} {time.time() - t0:.0f}s use {np.mean(o['use']):.3f}", flush=True)
    r = {k: np.concatenate(v) for k, v in acc.items()}
    np.savez(a.out, frames=fr, psi=psi32, **r)
    info = dict(fold_dir=str(d), stage=a.stage, **{k: v for k, v in st.items()}, device=a.device, n=int(len(fr)),
                seconds=time.time() - t0, use=float(np.mean(r["use"])), torch=torch.__version__, host=platform.node(),
                gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu",
                sha_psi64=h64.hexdigest(), sha_psi32=hashlib.sha256(psi32.tobytes()).hexdigest(),
                **{k: float(np.mean(r[k])) for k in ("conv", "xok", "inbox", "miss_ok", "good_base")})
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print("RUN_DONE " + json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
