"""Inference entrypoint of the Final reproduction package.

    python infer.py --config final_B --split private_test --weights WEIGHTS_DIR \
        [--data-root DATA_ROOT] [--work WORK_DIR] [--device cuda|cpu] [--skip ipguard,order]
        [--owner repro] [--workers 6] [--dry-run]

Writes <work>/fec_work/final_<split>/submission/{diii_d_<split>.npz, mast_<split>.npz} (the two
files of a Codabench submission) and prints their SHA-256.  On public_test, missing canonical
mast_public_test input caches are first built with train.py's data-stage commands (ensure_mast_caches,
rebuild defect R2-1).  What runs is fec/run_final.py `all` with the candidate's recipe
(configs/<config>.json):

  download  configs missing under --data-root are fetched from Hugging Face
            (private_test: ganatma/fusion-equilibrium-challenge-final at the data_manifest.json pin; fec/final_data.py)
  ipguard   DIII-D Ip time-axis record-class guard + input time-axis guard (inputs only; a failure
            STOPS the run: see PROVENANCE.md, idx time axis)
  extract / order / dataset / passive        inputs -> per-frame arrays (CPU)
  mast_amort    amortized v7h network at solenoid fraction 0.55 + 30 ms smoothing (GPU or CPU)
  mast_physics  physics average avg_f05 -> q95 / betaN records (GPU; CPU with --device cpu)
  d3d_b         DIII-D C2 ensemble (44 members) ; d3d_q95h G1 q95 hybrid ; d3d_lirep4 li repair
                (final_A22li_dec: d3d_dec, the off-span decoders, between d3d_q95h and d3d_lirep4)
  mast_composite + submission (fec/final_build.py with the frozen TRAIN clip ranges) + validate

--weights is the directory of weights_manifest.json's files (download them from the weights
repository named in weights_manifest.json; every file is checked against its SHA-256).
Retrained weights (train.py --stage export -> <work>/weights_out, with its own sha256.json) cannot
match those hashes: run them with --no-weight-hash, as `reproduce.sh <config> full` does.
--device cpu runs everything on the CPU (slow; portability/smoke runs) through cpu_shim/.
--workers N (default 6): CPU worker processes of fec/run_final.py's pooled CPU stages, the MAST physics stage and its
stage-2 pool included (reproduce.sh: WORKERS=N); there is no CPU, memory or wall-clock limit at inference (README.md,
"Inference: worker processes, hardware and wall clock").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent


def sha256(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def env_for(work, data_root, device):
    work = Path(work).resolve()
    e = dict(os.environ)
    e.update(FEC_PKG_WORK=str(work), FEC_WORK=str(work / "fec_work"),
             FEC_IDX_WORK=str(work / "fec_work_idx"), FEC_TRAIN_WORK=str(work / "fec_work"),
             FEC_AUDIT2_WORK=str(work / "fec_work_audit2"),
             FEC_GPU_SLOT_DIR=str(work / "fec_work" / "gpu_slots"), PYTHONUNBUFFERED="1",
             PYTHONHASHSEED=os.environ.get("PYTHONHASHSEED", "0"))
    if data_root:
        e["FUSION_EXTERNAL_DATA_ROOT"] = str(Path(data_root).resolve())
    if device == "cpu":
        e["FEC_DEVICE"] = "cpu"
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["PYTHONPATH"] = os.pathsep.join([str(PKG / "cpu_shim")] +
                                          ([e["PYTHONPATH"]] if e.get("PYTHONPATH") else []))
    return e


def install_frozen(work, manifest):
    """Copy the frozen artefacts (data_manifest.json 'frozen') into the work roots, checked."""
    work = Path(work).resolve()
    roots = {"fec_work": work / "fec_work", "fec_work_idx": work / "fec_work_idx"}
    n = 0
    for rel, rec in manifest["frozen"].items():
        src = PKG / rec["file"]
        root, sub = rec["dest"].split("/", 1)
        dst = roots[root] / sub
        if sha256(src) != rec["sha256"]:
            raise SystemExit(f"frozen artefact {src} does not match its SHA-256")
        if rec.get("template"):           # {FEC_WORK} -> this work root (path-free package)
            txt = src.read_text(encoding="utf-8").replace(
                "{FEC_WORK}", roots["fec_work"].as_posix())
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(txt, encoding="utf-8")
            n += 1
            continue
        if not dst.exists() or sha256(dst) != rec["sha256"]:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            n += 1
    return n


def uses_tx1s(cfg):
    """True for configs whose MAST maps use the TX1s network (composite maps kind vb / combo)."""
    return ((cfg.get("mast") or {}).get("composite") or {}).get("maps", {}).get("kind") in ("vb", "combo")


def mast_tags(cfg):
    """weights_manifest `mast` tags the config's outputs read: the amortized net its composite maps name
    (maps.net; kinds amort / vb / combo) and TX1s for vb / combo.  Empty for a physics-map config (final_C: maps,
    q95 and betaN from the physics average), which reads no MAST network (rebuild defect R2-5)."""
    maps = ((cfg.get("mast") or {}).get("composite") or {}).get("maps") or {}
    return ({maps["net"]} if maps.get("net") else set()) | ({"tx1s"} if uses_tx1s(cfg) else set())


def install_weights(work, weights, wman, tags=None, check=True, tx1s=True, mast_tags=None):
    """Place the model files under the work roots (hard link when possible), SHA-256 checked.

    Rows absent from `weights` that the run does not need are skipped with a message instead of stopping it,
    because a retrained tree (train.py --stage export -> weights_out) holds only what the training made:
    `mast_cache` rows (fec/passive.py geometry caches, rebuilt on demand by PassiveModel(cache=True); e.g.
    the plasma="filament" cache is never made by a Final stage), the TX1s rows when tx1s=False (config
    without vb / combo maps), and any `mast` row whose tag is not in `mast_tags` (mast_tags(cfg); a physics-map
    config such as final_C trains and reads no MAST network, R2-5).  Every file that is present is still checked
    against its SHA-256 (check=True)."""
    work = Path(work).resolve()
    roots = {"fec_work": work / "fec_work", "fec_work_idx": work / "fec_work_idx"}
    n = 0
    for rec in wman["files"]:
        if tags is not None and rec.get("tag") not in tags and                 rec.get("group") not in ("mast", "mast_cache"):
            continue
        src = Path(weights) / rec["path"]
        root, sub = rec["dest"].split("/", 1)
        dst = roots[root] / sub
        if not src.exists():
            if rec.get("group") == "mast_cache":
                print(f"mast_cache {rec['path']} not in {weights}: rebuilt on demand", flush=True)
                continue
            if rec.get("group") == "mast" and rec.get("tag") == "tx1s" and not tx1s:
                print(f"{rec['path']} not in {weights}: TX1s is not used by this config", flush=True)
                continue
            if rec.get("group") == "mast" and mast_tags is not None and rec.get("tag") not in mast_tags:
                print(f"{rec['path']} not in {weights}: the config reads no MAST network {rec.get('tag')}", flush=True)
                continue
            raise SystemExit(f"missing weight file {src} (download the weights repository first)")
        if dst.exists() and dst.stat().st_size == rec["bytes"] and not check:
            continue
        if check and sha256(src) != rec["sha256"]:
            raise SystemExit(f"weight file {src} does not match its SHA-256")
        if dst.exists():
            if dst.stat().st_size == rec["bytes"] and (not check or sha256(dst) == rec["sha256"]):
                continue
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        n += 1
    return n


def mast_cache_markers(fec_work, cfg):
    """Completion markers of the canonical MAST input caches that fec/final_mast.py (mast_amort) and the physics
    average (mast_physics) read: shots/<cfg>/_meta.json (extract), ds/<cfg>/info.json (dataset) and the C1 passive
    frames passive/frames/<cfg>/c1_r1/{meta.json, I.npy} (written last; they need every passive/sig/<cfg> file)."""
    W = Path(fec_work)
    fr = W / "passive" / "frames" / cfg / "c1_r1"
    return [W / "shots" / cfg / "_meta.json", W / "ds" / cfg / "info.json", fr / "meta.json", fr / "I.npy"]


def local_parquet(env, cfg):
    """True when fec.paths.shot_files(cfg) would find parquet files: the data root, then <FEC_WORK>/hf_data/data."""
    base = Path(env.get("FEC_PKG_WORK") or PKG / "work")
    data = Path(env["FUSION_EXTERNAL_DATA_ROOT"]) if env.get("FUSION_EXTERNAL_DATA_ROOT") else base / "data"
    roots = (data / "fusion-equilibrium-challenge" / "data", Path(env["FEC_WORK"]) / "hf_data" / "data")
    return any(any((r / cfg).glob("*.parquet")) for r in roots)


def mast_cache_cmds(cfg, workers):
    """The MAST data commands of train.py --stage data (same CLIs, arguments and outputs)."""
    py = [sys.executable, "-B"]
    return [py + ["fec/extract.py", cfg, "--workers", str(workers)],
            py + ["fec/dataset.py", cfg],
            py + ["fec/passive.py", "signals", "--config", cfg, "--workers", str(workers)],
            py + ["fec/passive.py", "frames", "--model", "c1", "--rscale", "1", "--config", cfg]]


def ensure_mast_caches(env, split, workers, dry=False, run=subprocess.run):
    """Rebuild defect R2-1 (agent rebuild2; fix agent pkgfix2, 2026-10-13).  On public_test, fec/run_final.py writes its
    data stages to a shadow root (final_public_test/shadow, for its `verify` comparison) while every MAST stage reads
    the CANONICAL caches, so on an empty work dir mast_amort stopped.  Here, before run_final, missing canonical
    mast_public_test caches are built with train.py's data-stage commands (downloading the config first when no
    parquet is local), then checked; a failure stops the run before any GPU stage.  private_test needs nothing: its
    data stages (extract with the organisers' Thomson rule, dataset, passive) write the canonical caches themselves,
    and building them here with train.py's extract rule would conflict with that rule.  Returns the commands."""
    if split != "public_test":
        return []
    cfg = f"mast_{split}"
    miss = [p for p in mast_cache_markers(env["FEC_WORK"], cfg) if not p.exists()]
    if not miss:
        return []
    cmds = ([] if local_parquet(env, cfg) else [[sys.executable, "-B", "fec/final_data.py", "download", cfg]]) + \
        mast_cache_cmds(cfg, workers)
    print(f"{cfg}: canonical MAST input caches missing ({len(miss)} of 4 markers, e.g. {miss[0]}); building them with "
          f"train.py's data-stage commands before run_final (R2-1)", flush=True)
    for cmd in cmds:
        cmd = [str(c) for c in cmd]
        print("$ " + subprocess.list2cmdline(cmd), flush=True)
        if dry:
            continue
        rc = run(cmd, cwd=str(PKG), env=env).returncode
        if rc:
            raise SystemExit(f"exit {rc}: {cmd} (building the canonical {cfg} caches before run_final's mast_amort)")
    if not dry:
        still = [str(p) for p in mast_cache_markers(env["FEC_WORK"], cfg) if not p.exists()]
        if still:
            raise SystemExit(f"{cfg}: canonical MAST input caches still missing after the build: {still}")
    return cmds


def recipe_tags(cfg):
    """DIII-D member + fallback tags of a candidate (its d3d.b recipe file)."""
    r = json.loads((PKG / cfg["d3d"]["b"]["recipe"]).read_text())
    fb = r.get("fallback") or []
    fb = [fb] if isinstance(fb, dict) else fb
    return {m["tag"] for m in r["members"] + fb}


def dec_tags(cfg):
    """weights_manifest `offdec` tag of a config with the off-span decoder stage (d3d.dec: final_A22li_dec; its three
    dec_s<seed>.pt), else empty: install_weights then skips the decoder rows, as for every other config."""
    q = (cfg.get("d3d") or {}).get("dec") or {}
    return {q["tag"]} if q.get("tag") else set()


def weight_paths(wman, cfg):
    """weights_manifest paths that `reproduce.sh` inference mode downloads (and completeness-checks) for a config: every
    row as before, except the off-span decoder rows (group offdec) of a config without that stage (d3d.dec; its stage
    list has no d3d_dec).  So the decoder rows can stay in the manifest while every other config's inference works at a
    weights revision without them (review rev_decpkg D1)."""
    dt = dec_tags(cfg)
    return sorted({r["path"] for r in wman["files"] if r.get("group") != "offdec" or r.get("tag") in dt})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    cands = sorted(p.stem for p in (PKG / "configs").glob("final_*.json"))
    ap.add_argument("--config", required=True,
                    help=" | ".join(cands) + " (configs/<config>.json)")
    ap.add_argument("--split", default="private_test", choices=["public_test", "private_test"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work", default=str(PKG / "work"))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--skip", default="", help="run_final stages to skip (e.g. order)")
    ap.add_argument("--with-order", action="store_true",
                    help="also run run_final's HF stream-order check (needs <FEC_WORK>/venv_hf "
                         "with the `datasets` package; Windows layout)")
    ap.add_argument("--owner", default="repro")
    ap.add_argument("--workers", type=int, default=6,
                    help="CPU worker processes of the pooled CPU stages (default 6; reproduce.sh: WORKERS)")
    ap.add_argument("--no-weight-hash", action="store_true",
                    help="skip the SHA-256 check against weights_manifest.json (faster re-runs; required "
                         "for retrained weights, which differ from the shipped ones)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfgf = PKG / "configs" / f"{a.config}.json"
    cfg = json.loads(cfgf.read_text())
    dman = json.loads((PKG / "data_manifest.json").read_text())
    wman = json.loads((PKG / "weights_manifest.json").read_text())
    env = env_for(a.work, a.data_root, a.device)
    print(f"infer {a.config} split {a.split} device {a.device} work {env['FEC_PKG_WORK']}",
          flush=True)
    print(f"frozen artefacts installed: {install_frozen(a.work, dman)}", flush=True)
    tags = recipe_tags(cfg)
    dtags = dec_tags(cfg)       # final_A22li_dec: + the 3 off-span decoders (empty for every other config)
    print(f"weights installed: {install_weights(a.work, a.weights, wman, tags | dtags, not a.no_weight_hash, uses_tx1s(cfg), mast_tags(cfg))}"
          f" ({len(tags)} DIII-D members + MAST net{' + decoders' if dtags else ''})", flush=True)
    ensure_mast_caches(env, a.split, a.workers, a.dry_run)       # R2-1: public_test on an empty work dir
    cmd = [sys.executable, "-B", str(PKG / "fec" / "run_final.py"), "all", "--split", a.split,
           "--recipe", str(cfgf), "--owner", a.owner, "--workers", str(a.workers),
           "--no-runlog", "--gpu-priority", "high"]
    skip = [s for s in a.skip.split(",") if s]
    if not a.with_order and "order" not in skip:
        skip.append("order")           # HF stream-order check: needs a Windows venv_hf (opt-in)
    if a.split == "public_test" and "verify" not in skip:
        skip.append("verify")          # verify compares with research-machine references
    if skip:
        cmd += ["--skip", ",".join(skip)]
    if a.dry_run:
        cmd.append("--dry-run")
    print("$ " + subprocess.list2cmdline(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(PKG), env=env).returncode
    if rc or a.dry_run:
        return rc
    sub = Path(env["FEC_WORK"]) / f"final_{a.split}" / "submission"
    for f in sorted(sub.glob("*.npz")):
        print(f"{f.name} sha256 {sha256(f)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
