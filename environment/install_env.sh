#!/usr/bin/env bash
# Python environment of the Final reproduction package (reproduce.sh runs it; run it yourself in a FRESH venv,
# Python 3.13, when you call train.py / infer.py directly):
#   bash environment/install_env.sh
# Linux x86-64 (the official rebuild target): torch 2.13.0 built for CUDA 12.6 from the PyTorch index first
# (NVIDIA driver >= 525.60.13; pip's default PyPI wheel is +cu130 and needs driver >= 580), then the pinned
# requirements, whose torch==2.13.0 that build already satisfies.  FEC_TORCH_INDEX overrides the index
# (e.g. https://download.pytorch.org/whl/cu130).  Other systems: the pinned requirements only.
# In an environment that already holds another torch 2.13.0 build, pip keeps it: check torch.version.cuda in
# $WORK/run_fingerprint.json.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "$(uname -s)" = Linux ]; then
  python -m pip install "torch==2.13.0" --index-url "${FEC_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
fi
python -m pip install -r "$HERE/requirements.txt"
