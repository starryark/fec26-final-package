"""CPU smoke run of infer.py on a 3-shot fold per machine (clean checkout).

    python tests/smoke_cpu.py --source-data-root SRC --weights WEIGHTS [--config final_B]
        [--shots 3] [--work DIR]

Builds a tiny data root whose diii_d_private_test / mast_private_test configs hold the first
--shots public-test INPUT files (the public-test parquet files carry no labels), then runs
`infer.py --device cpu --skip ipguard,order` (the Ip record-class guard needs a full fold; the
HF stream-order check needs the network) and `validate_submission.py`.  Passes when both
submission files exist, validate, and every psi / q95 / betaN value is finite.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-data-root", required=True,
                    help="data root holding fusion-equilibrium-challenge/data/<public configs>")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default="final_B")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--work", default=str(PKG / "work_smoke"))
    ap.add_argument("--pick", default="first", choices=["first", "shortest"])
    a = ap.parse_args(argv)
    work = Path(a.work).resolve()
    droot = work / "data"
    src = Path(a.source_data_root) / "fusion-equilibrium-challenge" / "data"
    for pub, priv in (("diii_d_public_test", "diii_d_private_test"),
                      ("mast_public_test", "mast_private_test")):
        files = sorted((src / pub).glob("*.parquet"))
        if a.pick == "shortest":
            files = sorted(files[:200], key=lambda f: f.stat().st_size)
        out = droot / "fusion-equilibrium-challenge" / "data" / priv
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(files[:a.shots]):
            if not (out / f.name).exists():
                shutil.copyfile(f, out / f.name)
    cmd = [sys.executable, "-B", str(PKG / "infer.py"), "--config", a.config, "--split",
           "private_test", "--weights", a.weights, "--data-root", str(droot), "--work",
           str(work), "--device", "cpu", "--skip", "ipguard,order", "--workers", "2",
           "--no-weight-hash"]
    print("$ " + subprocess.list2cmdline(cmd), flush=True)
    if subprocess.run(cmd, cwd=str(PKG)).returncode:
        print("SMOKE_FAILED infer")
        return 1
    if subprocess.run([sys.executable, "-B", str(PKG / "validate_submission.py"), "--split",
                       "private_test", "--work", str(work)], cwd=str(PKG)).returncode:
        print("SMOKE_FAILED validate")
        return 1
    man = work / "fec_work" / "final_private_test" / "manifest"
    for st in ("d3d_b", "d3d_q95h", "d3d_lirep4", "mast_amort", "mast_physics", "mast_composite",
               "submission"):
        m = json.loads((man / f"{st}.json").read_text())
        fb = (m.get("info") or {}).get("fallback")
        if m.get("status") != "ok" or fb:
            print(f"SMOKE_FAILED stage {st}: status {m.get('status')} fallback {fb} "
                  f"{(m.get('info') or {}).get('error', '')[:300]}")
            return 1
    sub = work / "fec_work" / "final_private_test" / "submission"
    for cfg in ("diii_d_private_test", "mast_private_test"):
        with np.load(sub / f"{cfg}.npz") as z:
            bad = [k for k in z.files if not np.isfinite(z[k]).all()]
            if bad:
                print(f"SMOKE_FAILED non-finite {cfg} {bad[:5]}")
                return 1
            print(f"{cfg}: {len(z.files)} members, all finite")
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
