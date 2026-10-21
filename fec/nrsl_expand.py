"""mast_solver2: expand a stage output computed on a frame SUBSET to the full fold (psi = base, use = False elsewhere),
so the frozen tools/ms_cascade.py can consume it.  python -B ms2_expand.py SUB.npz BASE.npy OUT.npz"""
import hashlib, json, sys
from pathlib import Path
import numpy as np


def main():
    sub, base_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
    B = np.load(base_p, mmap_mode="r")
    N = B.shape[0]
    psi = np.asarray(B, np.float32).copy()
    use = np.zeros(N, bool)
    if Path(sub).exists():
        z = np.load(sub)
        fr = z["frames"]
        psi[fr] = z["psi"]
        use[fr] = z["use"]
    np.savez(out, frames=np.arange(N), psi=psi, use=use)
    info = dict(sub=sub, n=int(N), n_sub=int(use.size and (np.load(sub)["frames"].size if Path(sub).exists() else 0)),
                use=float(use.mean()), sha_psi32=hashlib.sha256(psi.tobytes()).hexdigest())
    Path(out).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print("EXPAND_DONE " + json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
