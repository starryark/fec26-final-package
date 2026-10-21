"""Per-frame cascade of operator outputs: first file whose 'use' is True, else the base. python -B tools/ms_cascade.py OUT.npz BASE.npz F1.npz F2.npz ..."""
import sys
import numpy as np


def main():
    out, base, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    b = np.load(base)
    psi = b["psi"].astype(np.float32).copy()
    fr = b["frames"]
    src = np.full(len(fr), -1)
    for k, f in enumerate(files):
        z = np.load(f)
        assert np.array_equal(z["frames"], fr)
        take = (src < 0) & z["use"]
        psi[take] = z["psi"][take]
        src[take] = k
    np.savez(out, frames=fr, psi=psi, src=src, use=src >= 0)
    print("CASCADE", out, {k: int((src == k).sum()) for k in range(-1, len(files))})


if __name__ == "__main__":
    main()
