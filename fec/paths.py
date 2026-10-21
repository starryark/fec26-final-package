"""Paths of the Final reproduction package (package build of fec/paths.py).

Every location is an environment variable; the defaults live under <package>/work so a clean
checkout runs without any machine-specific path:
  FEC_PKG_WORK              base of the defaults below            (default <package>/work)
  FUSION_EXTERNAL_DATA_ROOT challenge data root holding fusion-equilibrium-challenge/data/<config>/
                            (default <base>/data; missing configs are downloaded into
                            <FEC_WORK>/hf_data by fec/final_data.py)
  FEC_WORK                  main work root (caches, models, preds)  (default <base>/fec_work)
  FEC_IDX_WORK              DIII-D root on the idx Ip time axis      (default <FEC_WORK>/../fec_work_idx)
  FEC_TRAIN_WORK            root holding the DIII-D TRAIN caches     (default FEC_WORK)
  FEC_AUDIT2_WORK           input-audit scratch                      (default <FEC_WORK>/../fec_work_audit2)
  FEC_GPU_SLOT_DIR          GPU slot registry                        (default <FEC_WORK>/gpu_slots)
  FEC_HF_TOKEN_FILE         (optional) HF token file, read per command only; never shipped
No secret, user name or absolute path is stored here.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_BASE = Path(os.environ.get("FEC_PKG_WORK", str(REPO / "work")))
DATA = Path(os.environ.get("FUSION_EXTERNAL_DATA_ROOT", str(_BASE / "data"))) \
    / "fusion-equilibrium-challenge" / "data"
WORK = Path(os.environ.get("FEC_WORK", str(_BASE / "fec_work")))
GPU_SLOT_DIR = Path(os.environ.get("FEC_GPU_SLOT_DIR", str(WORK / "gpu_slots")))
IDX_WORK = Path(os.environ.get("FEC_IDX_WORK", str(WORK.parent / "fec_work_idx")))
AUDIT2 = Path(os.environ.get("FEC_AUDIT2_WORK", str(WORK.parent / "fec_work_audit2")))
TRAIN_WORK = Path(os.environ.get("FEC_TRAIN_WORK", str(WORK)))
RUNPOD_KEY_FILE = None
HF_TOKEN_FILE = Path(os.environ.get("FEC_HF_TOKEN_FILE", "hf_token.txt"))
STARTER = REPO / "fusion-equilibrium-challenge-starter"
SCORING = STARTER / "fusion_scoring"

CONFIGS = ("diii_d_train", "diii_d_public_test", "mast_public_test")


def registered_work_roots():
    """The research machine's work-root registry is not part of the package."""
    return []


def warn_unregistered_work():
    return None


HF_DATA = WORK / "hf_data" / "data"   # writable mirror for configs not on the data root


def shot_files(config):
    """Lexically sorted parquet files == HF stream order for a `data/<config>/*.parquet` glob.

    Looks in the data root first, then the writable HF mirror (fec/final_data.py download)."""
    for root in (DATA, HF_DATA):
        files = sorted((root / config).glob("*.parquet"))
        if files:
            return files
    return []
