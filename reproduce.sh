#!/usr/bin/env bash
# End-to-end reproduction of one Final candidate.  Official full-rebuild target: Linux x86-64 with one 80 GB
# A100-class GPU; Windows is supported for inference and smoke runs only.
#   bash reproduce.sh final_A22li [inference|full]
#   SPLIT=public_test bash reproduce.sh final_A22li full     (rehearsal on the Development fold)
#   WORKERS=N bash reproduce.sh final_A22li                 (CPU worker processes of the inference: infer.py --workers;
#                                                             unset = infer.py's default 6, the setting of our runs)
# In full mode a failed early inputs download only warns: training continues, and missing or incomplete inputs are
#   downloaded again before the inference with HF_HUB_OFFLINE lifted for that one step; the run stops only if that
#   also fails.
# inference (default): download the pinned weights and the $SPLIT inputs, then (offline) infer.py + validate.
# full: download the training data (pinned dataset revision) and the $SPLIT inputs, then (offline) train.py
#       data, d3d, mast, export (README.md: measured hours), then infer.py with the retrained weights.
# Environment: environment/install_env.sh (Linux: torch 2.13.0+cu126 from the PyTorch index, NVIDIA driver
#   >= 525.60.13, then environment/requirements.txt); FEC_SKIP_INSTALL=1 keeps the current environment.
# Hugging Face: HF_TOKEN is optional (a read token raises the anonymous rate limit, HTTP 429); this script never
#   starts an interactive login.  HF_HUB_DISABLE_XET=1 (plain HTTP transfers) unless you set it.  Every step after
#   the downloads runs with HF_HUB_OFFLINE=1 (except that one inputs re-download), so a hidden network call fails
#   instead of passing unnoticed.
# Caches (Hugging Face, torch inductor, triton, temporary files) live under $WORK/cache.  $WORK/run_fingerprint.json
#   records the environment, commits, config hashes and data/weights revisions of the run.
set -euo pipefail
CFG=${1:-final_A22li}
MODE=${2:-inference}
SPLIT=${SPLIT:-private_test}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HERE/work}
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
WEIGHTS=${WEIGHTS:-$HERE/weights}
# the token of an earlier `hf auth login` stays readable after HF_HOME moves under $WORK
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-${HF_HOME:-${HOME:-$WORK}/.cache/huggingface}/token}
export HF_HOME="$WORK/cache/hf"
export HF_HUB_CACHE="$WORK/cache/hf/hub"
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torchinductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export TMPDIR="$WORK/cache/tmp"
mkdir -p "$HF_HUB_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
cd "$HERE"
if [ "${FEC_SKIP_INSTALL:-0}" != 1 ]; then
  bash environment/install_env.sh
fi
NOHASH=""
if [ "$MODE" = full ]; then
  python -B train.py --config "$CFG" --stage preflight --work "$WORK"
  # the $SPLIT inputs first: a missing dataset shows at once, but only the final inference needs them
  if ! python -B train.py --config "$CFG" --stage inputs --split "$SPLIT" --work "$WORK"; then
    echo "WARNING: the $SPLIT inputs could not be downloaded (Final inputs: ganatma/fusion-equilibrium-challenge-final at the data_manifest.json pin; rehearse with SPLIT=public_test); training continues, the offline $SPLIT inference at the end will stop" >&2
  fi
  python -B train.py --config "$CFG" --stage download --work "$WORK"
  export HF_HUB_OFFLINE=1   # before the fingerprint, so it records the offline run (R2-6)
  python -B train.py --config "$CFG" --stage fingerprint --mode full --split "$SPLIT" --work "$WORK"
  python -B train.py --config "$CFG" --stage data --work "$WORK"
  python -B train.py --config "$CFG" --stage d3d --work "$WORK"
  python -B train.py --config "$CFG" --stage mast --work "$WORK"
  python -B train.py --config "$CFG" --stage export --work "$WORK"
  # the $SPLIT inputs before the offline inference: if the early download failed or is incomplete, download them
  # again (pinned / current commit, the normal retry policy) with HF_HUB_OFFLINE lifted for this one step only
  if ! python -B train.py --config "$CFG" --stage inputs --split "$SPLIT" --work "$WORK" --local-only; then
    echo "the $SPLIT inputs are missing or incomplete: downloading them again (online for this step only)" >&2
    if ! HF_HUB_OFFLINE=0 python -B train.py --config "$CFG" --stage inputs --split "$SPLIT" --work "$WORK"; then
      echo "ERROR: the $SPLIT inputs could not be downloaded; the retrained weights are in $WORK/weights_out. When the dataset is reachable run: python -B train.py --config $CFG --stage inputs --split $SPLIT --work $WORK && python -B infer.py --config $CFG --split $SPLIT --weights $WORK/weights_out --work $WORK --no-weight-hash" >&2
      exit 1
    fi
  fi
  WEIGHTS="$WORK/weights_out"
  # retrained files cannot match weights_manifest.json's SHA-256 of the shipped weights (seeded, not
  # bit-identical across GPUs / library versions); their own list is $WORK/weights_out/sha256.json
  NOHASH="--no-weight-hash"
else
  python -B train.py --config "$CFG" --stage inputs --split "$SPLIT" --work "$WORK"
  python - "$WEIGHTS" "$CFG" <<'PY'
import json
import sys
from fec.final_data import snapshot_complete, xet_off
from infer import weight_paths
xet_off()
w = json.load(open("weights_manifest.json"))
paths = weight_paths(w, json.load(open(f"configs/{sys.argv[2]}.json")))   # decoder rows only for d3d.dec configs
snapshot_complete(w["repository"], paths, sys.argv[1], what="download weights",
                  repo_type=w.get("repo_type", "model"), revision=w["revision"], allow_patterns=paths)
PY
  export HF_HUB_OFFLINE=1   # before the fingerprint, so it records the offline run (R2-6)
  python -B train.py --config "$CFG" --stage fingerprint --mode inference --split "$SPLIT" --work "$WORK"
fi
python -B infer.py --config "$CFG" --split "$SPLIT" --weights "$WEIGHTS" --work "$WORK" ${WORKERS:+--workers "$WORKERS"} $NOHASH
python -B validate_submission.py --split "$SPLIT" --work "$WORK"
