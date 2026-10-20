"""Validate the two submission files of a split (keys, shapes, dtypes, finiteness against the
local ds scaffold; fec/submit.py validate) and print their SHA-256.

    python validate_submission.py --split private_test [--dir <work>/fec_work/final_<split>/submission]
        [--work WORK_DIR]
Exit 0 only if both files validate.  Final file names: diii_d_private_test.npz and
mast_private_test.npz (Codabench Final phase; psirz float32).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
from infer import env_for, sha256  # noqa: E402


def data_roots(env, split):
    """FEC_WORK roots whose ds/<config> the predictions of `split` were made from, as fec/run_final.py's submission stage
    validates them (submission_sources: the DIII-D predictor's root + the canonical MAST root).  Every shipped config
    is an idx recipe, so run_final reads the DIII-D caches its own data stages wrote: on public_test the shadow root
    <FEC_WORK>/final_public_test/shadow (run_final's public rehearsal layout), on private_test the idx root
    (FEC_IDX_WORK).  MAST: the canonical <FEC_WORK> on both splits (the MAST stages read it; infer.py builds it on
    public_test, R2-1).  Rebuild defect R2-4 (agent pkgfix2, 2026-10-13): this used to read the idx root on public_test
    too, which a fresh public_test run never builds.  Nothing is built or required beyond these two ds dirs."""
    W = Path(env["FEC_WORK"])
    d3d = W / f"final_{split}" / "shadow" if split == "public_test" else Path(env["FEC_IDX_WORK"])
    return {f"diii_d_{split}": d3d, f"mast_{split}": W}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="private_test", choices=["public_test", "private_test"])
    ap.add_argument("--work", default=str(PKG / "work"))
    ap.add_argument("--dir", default=None)
    a = ap.parse_args(argv)
    env = env_for(a.work, None, "cuda")
    d = Path(a.dir) if a.dir else Path(env["FEC_WORK"]) / f"final_{a.split}" / "submission"
    rc = 0
    roots = data_roots(env, a.split)
    for cfg in (f"diii_d_{a.split}", f"mast_{a.split}"):
        f = d / f"{cfg}.npz"
        if not f.exists():
            print(f"MISSING {f}")
            rc = 1
            continue
        work = roots[cfg]
        r = subprocess.run([sys.executable, "-B", str(PKG / "fec" / "submit.py"), "validate",
                            str(f), "--config", cfg, "--psi-dtype", "float32", "--work", str(work)],
                           cwd=str(PKG), env=env).returncode
        print(f"{f.name}: {'OK' if r == 0 else 'FAILED'} sha256 {sha256(f)} (ds of {work})")
        rc |= r
    return rc


if __name__ == "__main__":
    sys.exit(main())
