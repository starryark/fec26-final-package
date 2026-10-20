"""Final-phase prediction driver: every stage of the submission pipeline for one split.

    python -B fec/run_final.py <stage> --split public_test|private_test [options]
    python -B fec/run_final.py all --split private_test --recipe sub07 --gpu-priority high --owner root
    python -B fec/run_final.py status --split private_test --recipe sub07

--recipe sub04 = Codabench 969279: DIII-D idx-Ip C2' TRAIN+VAL ensemble
(fec/recipes_b/final_ip_C2_tv.json, CUDA fp32, chunked), float32 psi, MAST v3 limiter-off +
boundary-limiter scalars + betan_v2.  --recipe sub07 (= current, root 2026-10-08) adds the frozen
DIII-D q95 G1 hybrid (stage d3d_q95h).  Without --recipe the legacy sub02 recipe (starter Ip axis)
runs, kept for its pinned unit tests.

Stages (`all` runs them in this order; each one is skipped when its outputs are complete and
its manifest says ok, unless --force; a failed stage can simply be re-run):

  download     private: fetch configs missing locally into fec_work/hf_data (fec/download_hf.py);
               --remote also compares local file counts with the Hugging Face listing
  ipguard      idx recipes only: fec/ip_timebase.py scan + check on the DIII-D parquet inputs
               (inputs only, before extraction); any failed condition STOPS the run (review
               fec_ip_timebase_review_20261008 sec. 6: a failing fold needs a reviewed decision);
               then fec/input_fixes2.py scan + guard (time-axis defects; failure STOPS)
  extract      fec/extract.py <cfg> [--ip-timebase idx|starter for DIII-D] per-shot aligned
               inputs (both configs); refuses a non-fresh DIII-D dir extracted on another Ip axis
  order        HF stream order == sorted names + first rows' efit_times (venv_hf `datasets`)
  dataset      fec/dataset.py <cfg>            frame arrays; coil/grid geometry check
  [d3d_train   opt-in (`all --with-train`, recipes with d3d.train only): train the DIII-D members
               from the recipe's job list with FEC_WORK = the DIII-D work root: fec/train_d3d_bx.py
               (trainer bx, production layout) or one fec/train_d3d_b.py job per member (serial,
               the default and fallback; --d3d-trainer overrides)       -> <d3d work>/models]
  passive      fec/passive.py signals + frames C1 (MAST passive currents, inputs only)
  mast_loff    fec/mast_select_v3.py --zmode root_t --kfit_warm zs --passive c1_r1
               --limiter off (GPU slot)                         -> mast/loff (submitted psi maps)
  [mast_lon    fec/mast_select_v3.py ... --limiter on            -> mast/lon     ] only when
  [mast_merge  merge_deep_crossing(loff, lon, R_in < 0.18 m)     -> mast/merged  ] mast.final=merged
  [mast_pipeline  recipes with mast.pipeline only (e.g. --recipe sub07_avg6): fec/mast_pipeline.py
               recipe = members + merge/average + stage 2 + betaN; replaces mast_loff, mast_lon,
               mast_merge, mast_stage2, mast_betan          -> mast/pipeline/<final node>]
  d3d_seq      fec/train_d3d_seq.py predict    (GPU slot)        -> preds/<seq name>
  d3d_b        fec/predict_d3d_b.py            (fp32, FEC_B_DEV) -> preds/<b name>
  d3d_mix      blend_preds (weighted pixel/scalar mean)          -> preds/<mix name>
  d3d_q95h     recipes with d3d.q95h only: fec/q95_hybrid.py apply --mode G1 --recompute (CPU;
               frozen constants checked; psi + other scalars byte-identical; failure -> head)
                                                                -> preds/<final name>_q95h
  d3d_dec      recipes with d3d.dec only (final_A22li_dec): fec/offdec.py apply, the off-span
               decoders on the q95h psi (GPU; scalars byte-identical; failure -> q95h preds)
                                                                -> preds/<final name>_q95h_dec
  d3d_mmk      recipes with d3d.mmk only (final_MTfull_mmk): fec/mmk.py apply, the frozen kappa step
               (fec/mmk_op.py) toward the members' mean TRAIN-envelope kappa, on the d3d_dec output
               (CPU; scalars byte-identical; failure -> d3d_dec preds); the members' kappa is
               computed in d3d_b while they are predicted   -> <that dir>_mmk
  d3d_lirep4   recipes with d3d.lirep4: fec/d3d_rdspec.py lirep_build4 on the d3d_mmk output (else
               the d3d_dec output, else the q95h output)        -> <that dir>_lirep4
  mast_stage2  fec/mast_v1.py 2 --boundary-limiter (q95, q90, betaN_e, R_geo, a; psi_b =
               max(psi_X, psi at R = 0.20 m, |Z| <= 1.0 m) for the scalars only)
                                                                -> <final>/scalars.npz
  mast_betan   frozen fec/betan_transfer.py lam_fgw model, geometry with the same boundary
               limiter (limited_geometry)                       -> <final>/betan_v2.npy
  submission   fec/build_submission.py --split <split> --mast-betan betan_v2.npy --psi-dtype
               <recipe psi_dtype> + fec/submit.py validate of both files  -> submission/
<final> = mast/loff by default (review fec_mast_v3_config_review_20261008 R1); mast_lon and
mast_merge stay callable on their own.
Rehearsal only (public_test): mast_v2eq (v3 driver, all switches off == select_v2), verify.
Opt-in, never in `all` unless --with-upload (root only):
  upload       fec/submit.py upload of the two validated files to the PRIVATE HF dataset repo;
               a dry run unless --confirm-upload; records the commit SHA
  pointer      fec/submit.py pointer --revision <upload SHA> --phase dev|final; the scorer read
               token is read by submit.py from env FEC_HF_READ_TOKEN only (never on argv, never in
               the driver's logs/manifests, which are scanned for it afterwards).  The pointer
               zip itself carries the token by design of the challenge pointer format.
  Codabench submission stays manual (fec/submit.py codabench submit ... --confirm).

Layout.  Run outputs live in fec_work/final_<split>/ (logs/, manifest/, preds/, mast/,
submission/, verify/).  Three caches are read by frozen modules at fixed paths and therefore
stay canonical: fec_work/shots/<cfg> + fec_work/ds/<cfg> (fec.features.load_ds) and
fec_work/passive/frames/<cfg>/c1_r1 (fec.mast_select_v3.stage1_v3).  For private_test the
data stages (extract, dataset, passive) write those canonical caches (new configs, nothing to
overwrite).  For public_test they already exist, so the same commands run with
FEC_WORK=fec_work/final_public_test/shadow and `verify` compares the shadow copies with the
canonical ones; every later MAST stage reads the canonical caches.

idx (DIII-D Ip time axis) routing.  A recipe with d3d.ip_timebase "idx" names a DIII-D work root
(d3d.work, default <FEC_WORK>_idx, i.e. fec_work_idx; --d3d-work overrides).  The DIII-D data
stages write <d3d work>/shots|ds/diii_d_<split> (private) or the shadow (public rehearsal) with
--ip-timebase idx; the predictor runs with FEC_WORK=<that root>, whose models/ and hf_data/ are
links (junctions on Windows) to fec_work/models and fec_work/hf_data.  build_submission and
validate run with FEC_WORK=<root>/view, whose ds/ entries link to exactly the caches the
predictions were made from (diii_d_train: canonical, for the TRAIN label clip ranges).

No existing fec/*.py is modified: each stage calls the module's own CLI (absolute output dirs
work because pathlib replaces WORK when joined with an absolute path) or, where a module has
no config parameter (betaN) or needs bounded memory (chunked DIII-D ensemble), reuses its
functions unchanged in a thin wrapper.
Inputs only: no labels of any test fold, no MAST label, no scorer mask.
Runbooks: docs/research/FINAL_RUNBOOK_20261009.md (current), FINAL_RUNBOOK_20261008.md.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK, shot_files  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
FEC = REPO / "fec"
SPLITS = ("public_test", "private_test")
HF_REPO = "Sophelio/fusion-equilibrium-challenge"       # public configs (Dev phase)


def hf_dataset_repo(split):
    """HF dataset of a split's inputs: the Final configs live in a separate dataset,
    ganatma/fusion-equilibrium-challenge-final (fec/final_data.py; env FEC_HF_DATASET)."""
    from fec.final_data import dataset_repo
    return dataset_repo(split)
MAST_CHUNK = 1024                      # fec.mast_select_v3.stage1_v3 max_frames (shot chunks)

# DIII-D: preds/d3d_mix_seq3_bens3 (sub02 as built 2026-10-07 21:39; preset "b_ens5w2fb" is
# sub02 as rebuilt 22:46 -- root picks the Final DIII-D recipe).  MAST: v3 production
# limiter OFF (psi maps submitted as-is) + boundary-only centre-column limiter for the stage-2
# scalars and the betaN geometry (research/reviews/fec_mast_v3_config_review_20261008.md, R1).
# "final": "merged" switches to the optional deep-crossing merge of limiter-off/on solutions.
DEFAULT_RECIPE = {
    "name": "sub02_d3d_mix_seq3_bens3__mast_v3_loff_blim",
    "d3d": {
        "seq": {"name": "d3d_seq_t12_bv2", "tags": ["d3d_seq_t1", "d3d_seq_t2"],
                "v2": "d3d_v2", "wv2": 1.0 / 3.0},
        "b": {"name": "d3d_b_ens3_fp32",
              "tags": ["d3d_b60_shape_s0", "d3d_b60_shapeA_s2", "d3d_b60_shapeL_s3"],
              "fallback": None, "device": "cpu"},
        "mix": {"name": "d3d_mix_seq3_bens3",
                "members": {"d3d_seq_t12_bv2": 0.5, "d3d_b_ens3_fp32": 0.5}},
    },
    "mast": {
        "stage1": {"zmode": "root_t", "kfit_warm": "zs", "passive": "c1_r1"},
        "final": "loff",
        "boundary_limiter": True,
        "merge": {"rule": "deep_crossing", "r_in_threshold": 0.18},
        "betan": "betan_v2.npy",
    },
}
# Named recipes for --recipe (a JSON file path also works).  d3d.seq / d3d.b / d3d.mix may be
# null; the submitted DIII-D predictions are preds/<mix name>, else <b name>, else <seq name>.
RECIPES = {
    "sub02_mix_seq3_bens3": DEFAULT_RECIPE,
    # Codabench sub02 as rebuilt 2026-10-07 22:46 (submissions/sub02/build.json d3d
    # d3d_b_ens5w2fb_fp32 = predict_d3d_b.py --combine of the five single-model preds with
    # notsL_s5 twice and the notsL_s5 no-Thomson fallback; listing tags is the same arithmetic)
    "b_ens5w2fb": dict(DEFAULT_RECIPE, name="d3d_b_ens5w2fb_fp32__mast_v3_loff_blim", d3d={
        "seq": None,
        "b": {"name": "d3d_b_ens5w2fb_fp32",
              "tags": ["d3d_b60_shape_s0", "d3d_b60_shape_s1", "d3d_b60_shapeA_s2",
                       "d3d_b60_shapeL_s3", "d3d_b60_notsL_s5", "d3d_b60_notsL_s5"],
              "fallback": "d3d_b60_notsL_s5", "device": "cpu"},
        "mix": None}),
}
# CURRENT submission recipe: Codabench sub04 (969279, DIII-D S 0.99624), built 2026-10-08 from
# fec_work_idx/preds/d3d_ip_C2_tv (predict_d3d_b.py recipe final_ip_C2_tv.json, FEC_B_DEV=cuda,
# FEC_WORK=fec_work_idx) + final_public_test/mast/loff (scalars --boundary-limiter, betan_v2.npy)
# with build_submission --psi-dtype float32.  d3d.b.chunk_rows: the same arithmetic as
# predict_d3d_b.predict_members in 4096-aligned row blocks (peak RAM ~5 GB instead of ~16 GB).
SUB04 = {
    "name": "sub04_d3d_ip_C2_tv__mast_v3_loff_blim__psi_f32",
    "psi_dtype": "float32",
    "d3d": {
        "ip_timebase": "idx",
        "work": "fec_work_idx",          # relative: sibling of FEC_WORK (fec.paths.IDX_WORK)
        "seq": None,
        "b": {"name": "d3d_ip_C2_tv", "recipe": "fec/recipes_b/final_ip_C2_tv.json",
              "device": "cuda", "chunk_rows": 32768},
        "mix": None,
    },
    "mast": json.loads(json.dumps(DEFAULT_RECIPE["mast"])),
    "upload": {"repo": "starrydark/fec-preds"},
    "reference": {"d3d": "fec_work_idx/preds/d3d_ip_C2_tv",
                  "mast": "fec_work/final_public_test/mast/loff",
                  "submission": "fec_work/final_public_test/submission_sub03_fp32"},
}
RECIPES["sub04"] = SUB04
RECIPES["current"] = SUB04
# Final variant: the same ensemble refit on train+val+test (--alldata, tags d3d_b60axi_*); the
# d3d_b stage refuses until every member's model.pt exists.
RECIPES["final_ip_C2_all"] = dict(json.loads(json.dumps(SUB04)),
                                  name="final_d3d_ip_C2_all__mast_v3_loff_blim__psi_f32",
                                  reference={})
RECIPES["final_ip_C2_all"]["d3d"]["b"].update(name="d3d_ip_C2_all",
                                               recipe="fec/recipes_b/final_ip_C2_all.json")
# sub07 = sub04 + DIII-D q95 G1 hybrid (root 2026-10-08; research/reviews/
# fec_q95_g1_review_20261008.md ACCEPT-with-conditions): q95 = physics contour q95 P of the
# predicted psi where P/H - 1 <= tau, else the NN head H; psi and the other 4 scalars unchanged.
# Frozen: mode G1, tau 0.03, w 1, envelope fec/q95_hybrid_envelope.json (sha256 below), AXIS_BOX,
# TF_TURNS 144.  Never retuned.  The physics parts are always recomputed (--recompute); on failure
# the submission falls back to the head predictions (recorded in the d3d_q95h manifest).
Q95H_FROZEN = {"mode": "G1", "tau": 0.03, "tf_turns": 144,
               "envelope_sha256": "aae80002ecede60a7a0980962ca9b822fd688a4d7307e3307b035bc3ad984617"}
SUB07 = json.loads(json.dumps(SUB04))
SUB07.update(name="sub07_d3d_ip_C2_tv_q95hG1__mast_v3_loff_blim__psi_f32")
SUB07["d3d"]["q95h"] = dict(Q95H_FROZEN, suffix="_q95h", fallback="head")
SUB07["reference"] = dict(SUB04["reference"], d3d_q95h="fec_work_idx/preds/d3d_ip_C2_tv_q95h",
                          submission="fec_work/final_public_test/submission_sub07_q95h")
RECIPES["sub07"] = SUB07
RECIPES["current"] = SUB07             # root 2026-10-08: the G1 q95 hybrid is the current recipe


# MAST candidate pipelines (fec/mast_pipeline.py, docs/research/MAST_PIPELINE_20261009.md): a recipe
# whose mast block names "pipeline": "<fec/recipes_mast/*.json>" runs ONE stage, mast_pipeline, in
# place of mast_loff/mast_lon/mast_merge/mast_stage2/mast_betan; the pipeline's final node is the
# MAST dir of the submission stage.  Recipes without the key are unchanged.  Presets: sub07 DIII-D +
# the sub09 (zsc) or sub23 (mean of 6 maps) MAST half; references = the public candidate dirs.
def _pipeline_preset(name, pipeline, ref_mast, ref_submission):
    r = json.loads(json.dumps(SUB07))
    r["name"] = name
    r["mast"]["pipeline"] = pipeline
    r["reference"] = dict(SUB07["reference"], mast=ref_mast, submission=ref_submission)
    return r


RECIPES["sub07_zsc"] = _pipeline_preset(
    "sub07_d3d_ip_C2_tv_q95hG1__mast_pipeline_sub09_zsc__psi_f32", "fec/recipes_mast/sub09_zsc.json",
    "fec_work/mastz/cand_zsc_z0", "fec_work/final_public_test/submission_sub09_q95h_zsc")
RECIPES["sub07_avg6"] = _pipeline_preset(
    "sub07_d3d_ip_C2_tv_q95hG1__mast_pipeline_sub23_avg6__psi_f32", "fec/recipes_mast/sub23_avg6.json",
    "fec_work/mastz/cand_avg6", "fec_work/final_public_test/submission_sub23_q95h_avg6")


# Final candidates (mastpost 2026-10-10): DIII-D = the 22-member all-data C2' ensemble
# (fec/recipes_b/final_ip_C2_all22.json; Dev 0.996429 >= 0.996221 -> Final DIII-D recipe, root
# 2026-10-10) + the frozen q95 G1 hybrid, exactly as preds/d3d_ip_C2_all22(_q95h) were made (run
# root_d3d_all22_pred); MAST = a pipeline recipe with the tsm post node (fec/mast_tsm.py).
# Reference submissions: the local sub29/sub30 build dirs hold the tv DIII-D file (the Dev pointer
# carried the all22 file, NOTE_d3d_revision.txt), so verify's DIII-D comparison against them is
# expected to differ; their MAST files are the meaningful comparison.
def _all22_preset(name, pipeline, ref_mast, ref_submission):
    r = _pipeline_preset(name, pipeline, ref_mast, ref_submission)
    r["d3d"]["b"].update(name="d3d_ip_C2_all22", recipe="fec/recipes_b/final_ip_C2_all22.json")
    r["reference"].update(d3d="fec_work_idx/preds/d3d_ip_C2_all22",
                          d3d_q95h="fec_work_idx/preds/d3d_ip_C2_all22_q95h")
    return r


RECIPES["all22_tsm30"] = _all22_preset(
    "final_d3d_ip_C2_all22_q95hG1__mast_pipeline_sub29_tsm30__psi_f32",
    "fec/recipes_mast/sub29_tsm30.json", "fec_work/mastz/cand_zsc_tsm30",
    "fec_work/final_public_test/submission_sub29_q95h_zsc_tsm30")
RECIPES["all22_avg6_tsm30"] = _all22_preset(
    "final_d3d_ip_C2_all22_q95hG1__mast_pipeline_sub30_avg6_tsm30__psi_f32",
    "fec/recipes_mast/sub30_avg6_tsm30.json", "fec_work/mastz/cand_avg6_tsm30",
    "fec_work/final_public_test/submission_sub30_q95h_avg6_tsm30")


# Compute-light badge rebuild (agent clight 2026-10-10, docs/research/COMPUTE_LIGHT_20261010.md):
# the same 22 all-data members retrained by fec/train_d3d_bx.py under tags d3d_bxax* (stacks 8 + 14,
# compiled pixel loss + LayerNorm, GPU-resident data; one H100: ~75 min of training), predicted
# with fec/recipes_b/final_ip_C2_bxall22.json.  d3d.train names the job list (train_d3d_b argv per
# member) that BOTH trainers read: "bx" = one train_d3d_bx.py process, "serial" = the existing
# path, one train_d3d_b.py job per member (default when d3d.train.trainer is absent, and the
# fallback via --d3d-trainer serial; same tags either way).  Training is the opt-in stage
# d3d_train (`all --with-train`); recipes without d3d.train are unchanged.
TRAINERS = ("serial", "bx")
D3D_TRAIN_BX = {"group_size": 22, "pix_impl": "compile", "ln_impl": "compile", "vram_gb": 80,
                "data_gpu": 1}
D3D_TRAIN_ALL22_BX = {"jobs": "fec/pod/jobs_d3d_axi.json", "tag_replace": ["d3d_b60ax=d3d_bxax"],
                      "trainer": "bx", "bx": dict(D3D_TRAIN_BX)}


def _bx_all22_preset(name, pipeline, ref_mast, ref_submission):
    r = _all22_preset(name, pipeline, ref_mast, ref_submission)
    r["d3d"]["b"].update(name="d3d_ip_C2_bxall22", recipe="fec/recipes_b/final_ip_C2_bxall22.json")
    r["d3d"]["train"] = json.loads(json.dumps(D3D_TRAIN_ALL22_BX))
    r["reference"].update(d3d="fec_work_idx/preds/d3d_ip_C2_bxall22",
                          d3d_q95h="fec_work_idx/preds/d3d_ip_C2_bxall22_q95h")
    return r


RECIPES["all22bx_tsm30"] = _bx_all22_preset(
    "final_d3d_ip_C2_bxall22_q95hG1__mast_pipeline_sub29_tsm30__psi_f32",
    "fec/recipes_mast/sub29_tsm30.json", "fec_work/mastz/cand_zsc_tsm30",
    "fec_work/final_public_test/submission_sub29_q95h_zsc_tsm30")
RECIPES["all22bx_avg6_tsm30"] = _bx_all22_preset(
    "final_d3d_ip_C2_bxall22_q95hG1__mast_pipeline_sub30_avg6_tsm30__psi_f32",
    "fec/recipes_mast/sub30_avg6_tsm30.json", "fec_work/mastz/cand_avg6_tsm30",
    "fec_work/final_public_test/submission_sub30_q95h_avg6_tsm30")
PIPELINE_REPLACES = ("mast_loff", "mast_lon", "mast_merge", "mast_stage2", "mast_betan")


# ---------------------------------------------------------------------------- Final candidates
# Agent fpkg 2026-10-12 (docs/archive/fec/FINAL_PACKAGE_20261012.md).  DIII-D = a C2 ensemble
# recipe (fec/predict_d3d_b.py recipe JSON, CUDA fp32, chunked) -> frozen q95 G1 hybrid ->
# lirep4 (fec/d3d_rdspec.py lirep_build4: label-free li-range repair of the frames whose li(2) lies outside
# [0.1, 5], any phase (in practice ramp-down frames), Dev
# +0.000049, EVIDENCE B5; stage d3d_lirep4).  MAST = the composite of
#   maps    amortized v7h network at solenoid fraction f (fec/final_mast.py amort; FEC_AMORT_SOLF)
#           + within-shot Gaussian smoothing sigma 30 ms (fec/final_mast.py tsm),
#   scalars q95 from stage 2 of the physics average avg_f05 (fec/final_mast.py physics: zsc C1,
#           zsc no-passive, deep-crossing merge of zsc C1 with its centre-column-limiter twin,
#           zfix 0, zsc C1 tsm30; equal weights in that order = fec_work/mastz/cand_avg_f05),
#   betaN   frozen lam_fgw records of avg_f05 with the fill3 shot-fill rule (min 3 finite frames).
# Stages mast_amort -> mast_physics -> mast_composite replace the legacy MAST stages.
LIREP4_FROZEN = {"suffix": "_lirep4", "fallback": "src", "target": 4.9}
COMPOSITE_V7H055 = {
    "maps": {"kind": "amort", "net": "n7h_f0.55", "sol_frac": 0.55, "tsm_sigma_ms": 30.0},
    "scalars": {"plan": "avg_f05"},
    "betan_fill": {"rule": "fill3", "min_finite": 3},
}


def _final_preset(name, b_name, b_recipe, ref_submission, composite=None):
    """A Final candidate: SUB07's DIII-D chain (+ lirep4) with recipe b_recipe, MAST composite."""
    r = json.loads(json.dumps(SUB07))
    r["name"] = name
    r["d3d"]["b"].update(name=b_name, recipe=b_recipe)
    r["d3d"]["lirep4"] = dict(LIREP4_FROZEN)
    r["train_ranges"] = "fec/final_train_ranges.json"     # frozen TRAIN clip ranges (final_build)
    r["mast"] = {"stage1": {"passive": "c1_r1"}, "boundary_limiter": True,
                 "betan": "betan_v2.npy",
                 "composite": json.loads(json.dumps(composite or COMPOSITE_V7H055))}
    r["reference"] = {"d3d": f"fec_work_idx/preds/{b_name}",
                      "d3d_q95h": f"fec_work_idx/preds/{b_name}_q95h",
                      "d3d_lirep4": f"fec_work_idx/preds/{b_name}_q95h_lirep4",
                      "mast": "fec_work/mastz/cand_v7h055tsm_sc38",
                      "mast_maps": "fec_work/amort/real_v7h_f0.55_tsm30",
                      "mast_scalars": "fec_work/mastz/cand_avg_f05",
                      "submission": ref_submission}
    return r


# Final A (W3k44: 16 W3kE120 F1 + 28 W3kE120 notsL F2, OLD 28 notsL no-Thomson fallback; Oct 13
# candidate, reviewed USE-WITH-NOTES, B16) and Final B (W3e44: 16 W3kE120 F1 + 28 old notsL; Dev
# 0.996580, B17), both with the v7h f0.55 tsm30 composite (Dev MAST 0.796343, C25).
RECIPES["w3k44_v7h055"] = _final_preset(
    "final_w3k44_q95hG1_lirep4__mast_v7h055tsm30_avgf05sc_fill3__psi_f32", "d3d_ip_C2_w3k44",
    "fec/recipes_final/final_ip_C2_w3k44.json",
    "fec_work/final_public_test/submission_sub52_w3k44l4_v7h055tsm_sc38")
RECIPES["w3e44_v7h055"] = _final_preset(
    "final_w3e44_q95hG1_lirep4__mast_v7h055tsm30_avgf05sc_fill3__psi_f32", "d3d_ip_C2_w3e44",
    "fec/recipes_b/final_ip_C2_w3e44.json",
    "fec_work/final_public_test/submission_sub46w_w3e44l4_v7h055tsm_sc38_f3")
RECIPES["final_A"] = RECIPES["w3k44_v7h055"]
RECIPES["final_B"] = RECIPES["w3e44_v7h055"]
# Final C (repair reserve, root decides): Final B's DIII-D half + a MAST half without the
# amortized network or synthetic data -- maps, q95 and fill3 betaN all from the physics average
# avg_f05 (the same physics run A/B take their scalars from; cand_avg_f05 maps, cf. sub38).
RECIPES["final_C"] = _final_preset(
    "final_w3e44_q95hG1_lirep4__mast_avgf05maps_avgf05sc_fill3__psi_f32", "d3d_ip_C2_w3e44",
    "fec/recipes_b/final_ip_C2_w3e44.json",
    "fec_work/final_public_test/submission_sub46w_w3e44l4_v7h055tsm_sc38_f3",
    composite={"maps": {"kind": "physics"}, "scalars": {"plan": "avg_f05"},
               "betan_fill": {"rule": "fill3", "min_finite": 3}})
RECIPES["final_C"]["reference"].update(mast="", mast_maps="fec_work/mastz/cand_avg_f05")
# Final A24 (proposed, not built or scored; FINAL_PACKAGE_20261012 G2): a 12 + 12 subset of W3k44's
# members (old-28 fallback) that fits the 24 A100-h rebuild budget at the measured PCIe factor.
RECIPES["final_A24"] = _final_preset(
    "final_w3k24_q95hG1_lirep4__mast_v7h055tsm30_avgf05sc_fill3__psi_f32", "d3d_ip_C2_w3k24",
    "fec/recipes_final/final_ip_C2_w3k24.json", "")
RECIPES["final_A24"]["reference"].update(d3d="", d3d_q95h="", d3d_lirep4="")
# Final A22li (li2ft Amendment 3, EVIDENCE B31/B32; docs/archive/fec/D3D_LI2FT22_20261013.md section 7): final_A24's
# chain with W3k22 + arm A = W3k24 minus shapeL_s83 / notsL_s127, each fine-tuned on all labelled shots with the scorer
# li(2) loss (fec/pod/jobs_d3d_li2ft22.json), equal weights; the old-28 notsL fallback (not fine-tuned).
RECIPES["final_A22li"] = _final_preset(
    "final_w3k22liA_q95hG1_lirep4__mast_v7h055tsm30_avgf05sc_fill3__psi_f32", "d3d_ip_C2_w3k22liA",
    "fec/recipes_final/final_ip_C2_w3k22liA.json", "")
RECIPES["final_A22li"]["reference"].update(d3d="", d3d_q95h="", d3d_lirep4="")
# Oct 14 MAST map variants (agent fpkgm; docs/archive/fec/FINAL_PKG_MAST_PRESETS_20261013.md): final_A22li's DIII-D
# half + the MAST map a Dev read may adopt (family doc amendments 5, 11, 12).  Same physics scalars (avg_f05) and
# fill3 betaN as S4; only the maps differ:
#   r05    amortized v7h maps with the C1 passive currents at resistance scale 0.5, tsm30   (S3; C50; npz e59abc5e)
#   vb     0.5 (raw v7h + raw TX1s maps), tsm30 (TX1s: fec/final_tx1s.py, xfer/models1/tx1s) (T2; C45; npz 33309be6)
#   combo  vb tsm30 + (r05 tsm30 - S4 tsm30), meta recomputed                               (C59; npz d3d2ada3)
MAPS_KINDS = ("physics", "amort", "vb", "combo")
TX1S_TAG = "tx1s"


def _composite_variant(kind, rscale=None, tx1s=None):
    c = json.loads(json.dumps(COMPOSITE_V7H055))
    c["maps"]["kind"] = kind
    if rscale is not None:
        c["maps"]["rscale"] = rscale
    if tx1s:
        c["maps"]["tx1s"] = tx1s
    return c


COMPOSITE_R05 = _composite_variant("amort", rscale=0.5)
# r = 2 (agent decpkg 2026-10-14, root scope addition: knob adopted on Dev, EVIDENCE C66, 987175 MAST 0.802845 vs S4
# 0.796343): the S4 chain with the C1 passive resistances x 2 (docs/archive/fec/MAST_KNOBS_20261013.md sections 2-3;
# Dev file xfer/subs/sub_knob_r2, npz 12e264e5)
COMPOSITE_R2 = _composite_variant("amort", rscale=2)
COMPOSITE_VB = _composite_variant("vb", tx1s=TX1S_TAG)
COMPOSITE_COMBO = _composite_variant("combo", rscale=0.5, tx1s=TX1S_TAG)
for _name, _tag, _comp, _ref in (
        ("final_A22li_r05", "v7h055r05tsm30", COMPOSITE_R05, "fec_work/mastz/cand_knob_r05_v7h055tsm_sc38"),
        ("final_A22li_r2", "v7h055r2tsm30", COMPOSITE_R2, "fec_work/mastz/cand_knob_r2_v7h055tsm_sc38"),
        ("final_A22li_vb", "vb_v7h055tx1s_tsm30", COMPOSITE_VB, "fec_work/mastz/cand_blendp_tx1stsm_v7h055tsm_sc38"),
        ("final_A22li_combo", "combo_vb_r05", COMPOSITE_COMBO, "fec_work/mastz/cand_combo_vb_r05_sc38")):
    RECIPES[_name] = _final_preset(
        f"final_w3k22liA_q95hG1_lirep4__mast_{_tag}_avgf05sc_fill3__psi_f32", "d3d_ip_C2_w3k22liA",
        "fec/recipes_final/final_ip_C2_w3k22liA.json", "", composite=_comp)
    RECIPES[_name]["reference"].update(d3d="", d3d_q95h="", d3d_lirep4="", mast=_ref, mast_maps=_ref)
# Final A22li + dec (agent decpkg 2026-10-14; H1dec adopted on Dev, EVIDENCE B61; family doc amendment 26): final_A22li's
# chain with the off-span decoder stage d3d_dec between d3d_q95h and d3d_lirep4 (ordering amendment: q95h -> dec ->
# lirep4).  fec/offdec.py apply: psi + mean of the 3 decoders (seeds 40/41/42, trained by train.py --stage offdec on all
# labelled shots in the members' basis) on every frame; q95 / betaN byte-identical; lirep4 then runs on the decoded psi
# (full derived_li pass, no li cache).  On a failed check the stage records fallback=src and lirep4 runs on the q95h
# preds (= final_A22li).  MAST half = final_A22li_r2's (C1 passive resistances x 2; root, 2026-10-14 10:13 UTC).
DEC_H1DEC = {"suffix": "_dec", "fallback": "src", "dir": "offdec/d3d_ip_C2_w3k22liA", "tag": "offdec_d3d_ip_C2_w3k22liA",
             "seeds": [40, 41, 42], "chunk": 8192, "inspan_rms_max": 1e-8,
             # per-frame correction RMS of the loop's TRAIN-only VAL decoders (iter41; Dev sanity band 0.5-2x)
             "ref_corr_rms": {"median": 8.179523392453171e-06, "p99": 2.191760097187346e-05}, "ratio_band": [0.5, 2.0],
             # basis of the shipped members (HF weights e8531871), for the Dev-read decoders that do not record theirs
             "shipped_basis_sha256": "3883fa7101b76907eba39cc856d1b3691fc3d925d0330b0354a0108b4a588469"}
RECIPES["final_A22li_dec"] = _final_preset(
    "final_w3k22liA_q95hG1_dec_lirep4__mast_v7h055r2tsm30_avgf05sc_fill3__psi_f32", "d3d_ip_C2_w3k22liA",
    "fec/recipes_final/final_ip_C2_w3k22liA.json", "", composite=COMPOSITE_R2)
RECIPES["final_A22li_dec"]["d3d"]["dec"] = json.loads(json.dumps(DEC_H1DEC))
RECIPES["final_A22li_dec"]["reference"].update(d3d="", d3d_q95h="", d3d_lirep4="",
                                               mast="fec_work/mastz/cand_knob_r2_v7h055tsm_sc38",
                                               mast_maps="fec_work/mastz/cand_knob_r2_v7h055tsm_sc38")
# Final MT-full (agent mtpkg 2026-10-15; family doc amendment 39, EVIDENCE B75 Dev 0.996775, npz 68f168a6): final_A22li_dec
# with the 22 li2ft members replaced by their "MT" twins (bases d3d_bxw3emt_* with the mtask M2 aux heads, all-data aux
# targets; li2ft A with the heads on; docs/archive/fec/D3D_MTPROD_20261014.md, D3D_MTF2_20261014.md).  Old-28 fallback,
# q95h, the off-span decoder stage (the same DEC_H1DEC block: the MT members' PCA basis is bit-identical to A22li's, so
# the Dev-read decoders apply and full mode retrains them in that basis) and lirep4 unchanged; MAST = the r = 2 maps.
RECIPES["final_MTfull"] = _final_preset(
    "final_w3k22liAmtfull_q95hG1_dec_lirep4__mast_v7h055r2tsm30_avgf05sc_fill3__psi_f32", "d3d_ip_C2_w3k22liAmtfull",
    "fec/recipes_final/final_ip_C2_w3k22liAmtfull.json", "", composite=COMPOSITE_R2)
RECIPES["final_MTfull"]["d3d"]["dec"] = json.loads(json.dumps(DEC_H1DEC))
RECIPES["final_MTfull"]["reference"].update(d3d="", d3d_q95h="", d3d_lirep4="",
                                            mast="fec_work/mastz/cand_knob_r2_v7h055tsm_sc38",
                                            mast_maps="fec_work/mastz/cand_knob_r2_v7h055tsm_sc38")
# Final A24 + r = 2 MAST (agent mtpkg 2026-10-15; family doc amendment 46 ruling after rev_div: S2 carries r = 2 unless a
# new MAST config is adopted; final_A24 itself carries the r = 1 maps and stays unchanged): final_A24's DIII-D half
# exactly, MAST half = the r = 2 composite of final_A22li_dec / final_MTfull (COMPOSITE_R2; Dev file 12e264e5, C66).
RECIPES["final_A24r2"] = _final_preset(
    "final_w3k24_q95hG1_lirep4__mast_v7h055r2tsm30_avgf05sc_fill3__psi_f32", "d3d_ip_C2_w3k24",
    "fec/recipes_final/final_ip_C2_w3k24.json", "", composite=COMPOSITE_R2)
RECIPES["final_A24r2"]["reference"].update(d3d="", d3d_q95h="", d3d_lirep4="",
                                           mast="fec_work/mastz/cand_knob_r2_v7h055tsm_sc38",
                                           mast_maps="fec_work/mastz/cand_knob_r2_v7h055tsm_sc38")
# Final MTli (agent mtlipkg 2026-10-16; family doc amendment 50 (A), EVIDENCE B71 file 728f71e1): final_A22li_dec with only
# the 11 F1 li2ft members replaced by their MT twins (= final_MTfull's F1); the 11 F2 li2ft members and the old-28
# fallback are final_A22li_dec's.  Members recipe = byte copy of fec_work/mtprod/final_ip_C2_w3k22liAmt.json (sha
# 568dc572); q95h -> the same DEC_H1DEC decoder block (728f71e1 used h1dec/cache/dec_s{40,41,42}.pt; the MT F1 bases'
# PCA basis is bit-identical to A22li's, D3D_MTPROD section 4.3) -> lirep4; MAST = the r = 2 maps.
RECIPES["final_MTli"] = _final_preset(
    "final_w3k22liAmt_q95hG1_dec_lirep4__mast_v7h055r2tsm30_avgf05sc_fill3__psi_f32", "d3d_ip_C2_w3k22liAmt",
    "fec/recipes_final/final_ip_C2_w3k22liAmt.json", "", composite=COMPOSITE_R2)
RECIPES["final_MTli"]["d3d"]["dec"] = json.loads(json.dumps(DEC_H1DEC))
RECIPES["final_MTli"]["reference"].update(d3d="", d3d_q95h="", d3d_lirep4="",
                                          mast="fec_work/mastz/cand_knob_r2_v7h055tsm_sc38",
                                          mast_maps="fec_work/mastz/cand_knob_r2_v7h055tsm_sc38")
# Final MT-full + mmk (agent mmks2 2026-10-19): final_MTfull exactly (the same 22 members, ensemble, q95h, off-span
# decoder, lirep4 and r = 2 MAST half) with a label-free kappa finishing stage d3d_mmk between d3d_dec and d3d_lirep4
# (q95h -> dec -> mmk -> lirep4).  fec/mmk.py: in d3d_b each of the 22 members' own psi gives its TRAIN-envelope kappa
# per frame (no member map stored); d3d_mmk moves the post-decoder map's kappa half way (alpha 0.5) to the equal member
# mean with the frozen kappa step of fec/mmk_op.py (SHA-256 below; constants checked against this block before the
# stage runs); q95 / betaN byte-identical; on a failed step or check the stage records fallback=src and lirep4 runs on
# the d3d_dec output (= final_MTfull).
MMK_KAPPA = {"suffix": "_mmk", "fallback": "src", "step": "kappa", "operator": "fec/mmk_op.py",
             "operator_sha256": "5592926f8909e0dfaafefca6fb03f2a9569a95f97ef46f1f9d1ec7a496b9cdad",
             "alpha": 0.5, "sigma_z_m": 0.25, "psin_taper": [1.0, 1.4], "c_bracket": [0.9, 1.1], "maxit": 30,
             "rtol": 1e-4, "atol": 1e-6, "lcfs_n_iter": 22,
             "targets": "equal mean over the primary members of their own maps' TRAIN-envelope kappa",
             "env_train_sha256": "fc4ce12f0fc3e36c4ce8bcacaac4bbe62d3410b851477ad9ee1ee0f55e3d8505",
             # 59e: a frame with fewer finite member kappa is not edited; no worker task finished for this long -> the
             # mmk part stops (d3d_b: no kappa file, ensemble untouched; d3d_mmk: fallback=src)
             "min_members": 18, "stall_timeout_s": 600.0}
RECIPES["final_MTfull_mmk"] = json.loads(json.dumps(RECIPES["final_MTfull"]))
RECIPES["final_MTfull_mmk"]["name"] = \
    "final_w3k22liAmtfull_q95hG1_dec_mmk_lirep4__mast_v7h055r2tsm30_avgf05sc_fill3__psi_f32"
RECIPES["final_MTfull_mmk"]["d3d"]["mmk"] = json.loads(json.dumps(MMK_KAPPA))
# Final A22li_dec + mmk (agent mmk2pkg 2026-10-20): final_A22li_dec exactly (the same 22 li2ft members, ensemble, q95h,
# off-span decoder, lirep4 and r = 2 MAST half) with the same d3d_mmk stage as final_MTfull_mmk (the MMK_KAPPA block:
# frozen operator, alpha, fallback rows, minimum members, stall timeout), between d3d_dec and d3d_lirep4.
RECIPES["final_A22li_dec_mmk"] = json.loads(json.dumps(RECIPES["final_A22li_dec"]))
RECIPES["final_A22li_dec_mmk"]["name"] = \
    "final_w3k22liA_q95hG1_dec_mmk_lirep4__mast_v7h055r2tsm30_avgf05sc_fill3__psi_f32"
RECIPES["final_A22li_dec_mmk"]["d3d"]["mmk"] = json.loads(json.dumps(MMK_KAPPA))
COMPOSITE_STAGES = ("mast_amort", "mast_physics", "mast_composite")


def run_device(default):
    """Torch device of a stage: env FEC_DEVICE=cpu (reproduction package --device cpu, with
    cpu_shim/ on PYTHONPATH) forces the CPU; otherwise the recipe/default device."""
    return "cpu" if os.environ.get("FEC_DEVICE", "") == "cpu" else default
IP_TIMEBASES = ("idx", "starter")
CHUNK_ALIGN = 4096                     # fec.predict_d3d_b.predict_tag inference batch
# Boundary-only limiter (fec.mast_v1._scal_worker --boundary-limiter, fec.mast_select_v3.LIMITER_CC):
# psi_b = max(psi_X, max psi on R = 0.20 m, |Z| <= 1.0 m below the axis flux).
LIMITER_R, LIMITER_ZMAX = 0.20, 1.0
# Frozen betaN correction (research/reviews/fec_betan_transfer_review_20261008.md).
BETAN_REVIEWED = {
    "fec/betan_transfer.py": "0cbe115c202975272a023bd1c4058297c7e42168c94082feafac9856ec0b21a5",
    "selection.json": "ea36ea773298627fb0455a6e02b31816f930a3413964d4c0e6ac422197f150f5",
    "val_report.json": "81d271ea1b563099db293bf4c4b41358af9f76d5da0ee2ccb3fb92a6a20ccc42",
}
# Public-test artefacts the rehearsal is compared with (read-only).
REFERENCE = {
    "preds": WORK / "preds",
    "mast_v2eq": WORK / "mast_v2",
    "mast_loff": WORK / "mast_v3_loff",
    "mast_lon": WORK / "mast_v3_lon",
    "passive": WORK / "passive",
    "betan_records": WORK / "betan_transfer" / "records_mast.npz",
    "betan": WORK / "mast_v2" / "betan_v2.npy",
    "submission": WORK / "submissions" / "sub02",
}
TOL = 1e-6


class StageError(RuntimeError):
    pass


# ============================================================================ small helpers
def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable))
    os.replace(tmp, path)


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def cmdline(cmd):
    return subprocess.list2cmdline([str(c) for c in cmd])


def is_under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def resolve_path(p, base=None):
    """Absolute paths as given; relative ones are siblings of FEC_WORK (WORK.parent / p)."""
    p = Path(p)
    return p if p.is_absolute() else Path(base or WORK.parent) / p


def recipe_ip_timebase(recipe):
    """DIII-D Ip time axis the recipe's models were trained on (legacy recipes: starter)."""
    tb = recipe["d3d"].get("ip_timebase", "starter")
    if tb not in IP_TIMEBASES:
        raise ValueError(f"recipe d3d.ip_timebase must be one of {IP_TIMEBASES}, not {tb!r}")
    return tb


def recipe_d3d_work(recipe, override=None):
    """DIII-D data/prediction root: --d3d-work, else recipe d3d.work, else WORK."""
    w = override or recipe["d3d"].get("work")
    return resolve_path(w) if w else WORK


def is_dir_link(p):
    p = Path(p)
    return p.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(p))


def same_dir(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def ensure_dir_link(link, target, replace=False):
    """Make `link` a directory link to `target` (junction on Windows, symlink elsewhere).

    An existing link/dir that already resolves to target is kept.  replace=True swaps a link
    pointing elsewhere (only links are ever removed, never real directories).  Returns
    "exists" | "created" | "replaced"; raises StageError on a conflicting real directory."""
    link, target = Path(link), Path(target)
    if not target.is_dir():
        raise StageError(f"link target {target} is not a directory")
    if link.exists() or is_dir_link(link):
        if same_dir(link, target):
            return "exists"
        if not (replace and is_dir_link(link)):
            raise StageError(f"{link} exists and does not resolve to {target} "
                             f"({link.resolve() if link.exists() else 'dangling'})")
        os.rmdir(link)                      # removes the link only
        status = "replaced"
    else:
        status = "created"
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target.resolve()), str(link))
    else:
        os.symlink(target.resolve(), link, target_is_directory=True)
    if not same_dir(link, target):
        raise StageError(f"link {link} does not resolve to {target}")
    return status


def shots_ip_timebase(shots_dir):
    """ip_timebase recorded by fec/extract.py in <shots>/_meta.json (absent = legacy starter)."""
    f = Path(shots_dir) / "_meta.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("ip_timebase", "starter")


def subset_limit(shot, n_shots, max_frames=MAST_CHUNK):
    """Frame count of the smallest prefix of production shot chunks covering >= n_shots shots.

    stage1_v3 processes whole-shot chunks (fec.mast_select_v2.shot_chunks); a --limit at a chunk
    boundary reproduces the full run's batches, so the prefix can be compared bit for bit."""
    from fec.mast_select_v2 import shot_chunks
    shot = np.asarray(shot)
    for sl in shot_chunks(shot, max_frames):
        if shot[sl.stop - 1] >= n_shots - 1:
            return int(sl.stop)
    return int(len(shot))


# ============================================================================ layout
class Layout:
    """Paths of one split's run (see module doc).

    d3d_work: canonical DIII-D data root (idx recipes: the idx work root; default = work).
    d3d_fresh: DIII-D predictions read the DIII-D caches the data stages wrote (shadow on
    public_test) instead of the canonical ones (idx recipes)."""

    def __init__(self, split, subset_shots=0, work=WORK, root=None, d3d_work=None,
                 d3d_fresh=False):
        if split not in SPLITS:
            raise ValueError(split)
        self.split, self.work = split, Path(work)
        self.root = Path(root) if root else self.work / f"final_{split}"
        self.d3d_cfg, self.mast_cfg = f"diii_d_{split}", f"mast_{split}"
        self.configs = (self.d3d_cfg, self.mast_cfg)
        self.shadow = split == "public_test"
        self.subset_shots = int(subset_shots or 0)
        self.sfx = f"_sub{self.subset_shots}" if self.subset_shots else ""
        self.logs = self.root / "logs"
        self.manifest_dir = self.root / "manifest"
        self.preds = self.root / "preds"
        self.mast = self.root / "mast"
        self.verify_dir = self.root / "verify"
        self.view = self.root / "view"
        # FEC_WORK for the data stages: canonical caches (private) or a shadow root (public)
        self.data_work = self.root / "shadow" if self.shadow else self.work
        self.d3d_work = Path(d3d_work) if d3d_work else self.work
        self.d3d_data_work = self.data_work if self.shadow else self.d3d_work
        # FEC_WORK of the DIII-D predictor
        self.d3d_pred_work = self.d3d_data_work if d3d_fresh else self.d3d_work

    def cfg_work(self, cfg, canonical=False):
        """Data root of one config: DIII-D -> d3d_work (or the shadow), MAST -> work (or shadow)."""
        if cfg.startswith("diii_d"):
            return self.d3d_work if canonical else self.d3d_data_work
        return self.work if canonical else self.data_work

    def mast_dir(self, arm):
        return self.mast / f"{arm}{self.sfx}"

    def shots_dir(self, cfg, canonical=False):
        return self.cfg_work(cfg, canonical) / "shots" / cfg

    def ds_dir(self, cfg, canonical=False):
        return self.cfg_work(cfg, canonical) / "ds" / cfg

    def passive_dir(self, cfg, tag="c1_r1", canonical=False):
        return (self.work if canonical else self.data_work) / "passive" / "frames" / cfg / tag

    def writable(self, path):
        """Run outputs go under the split root; private data stages also write canonical caches."""
        if is_under(path, self.root):
            return True
        if self.shadow:
            return False
        ok = [self.work / "hf_data", self.work / "geometry"]
        for cfg in self.configs:
            ok += [self.work / "shots" / cfg, self.work / "ds" / cfg,
                   self.work / "passive" / "sig" / cfg, self.work / "passive" / "frames" / cfg]
        # DIII-D work root (idx recipes: fec_work_idx): this split's caches + the guard scan
        ok += [self.d3d_work / "shots" / self.d3d_cfg, self.d3d_work / "ds" / self.d3d_cfg,
               self.d3d_work / "ip_timebase", self.d3d_work / "input_guard"]
        return any(is_under(path, r) for r in ok)


# ============================================================================ blend / merge
def blend_preds(members, config, out_dir, src_dirs, chunk=8192):
    """Weighted mean of per-frame predictions: psi (N,65,65) and scal (N,5) float32.

    members {name: weight} (weights sum to 1); src_dirs {name: dir with <config>_psi.npy /
    <config>_scal.npy}.  float64 accumulation, one rounding to float32 (exact for 50/50)."""
    names = list(members)
    w = [float(members[n]) for n in names]
    if not names or abs(sum(w) - 1.0) > 1e-9:
        raise ValueError(f"blend weights must sum to 1: {members}")
    P = [np.load(Path(src_dirs[n]) / f"{config}_psi.npy", mmap_mode="r") for n in names]
    Q = [np.load(Path(src_dirs[n]) / f"{config}_scal.npy") for n in names]
    if len({p.shape for p in P}) != 1 or len({q.shape for q in Q}) != 1 or len(P[0]) != len(Q[0]):
        raise ValueError(f"member shapes differ: {[p.shape for p in P]} {[q.shape for q in Q]}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    N = P[0].shape[0]
    tmp = out_dir / f"{config}_psi.part.npy"
    out = np.lib.format.open_memmap(tmp, "w+", np.float32, P[0].shape)
    for s0 in range(0, N, chunk):
        sl = slice(s0, min(N, s0 + chunk))
        acc = np.zeros((sl.stop - sl.start,) + P[0].shape[1:], np.float64)
        for wi, p in zip(w, P):
            acc += wi * np.asarray(p[sl], np.float64)
        out[sl] = acc.astype(np.float32)
    out.flush()
    del out
    os.replace(tmp, out_dir / f"{config}_psi.npy")
    sc = np.zeros(Q[0].shape, np.float64)
    for wi, q in zip(w, Q):
        sc += wi * q.astype(np.float64)
    np.save(out_dir / f"{config}_scal.npy", sc.astype(np.float32))
    meta = {"config": config, "members": dict(zip(names, w)), "N": int(N),
            "sources": {n: str(src_dirs[n]) for n in names},
            "finite": bool(np.isfinite(sc).all())}
    write_json(out_dir / f"{config}_meta.json", meta)
    return meta


def _read_cfg(d):
    f = Path(d) / "config.json"
    return json.loads(f.read_text()) if f.exists() else {}


def merge_deep_crossing(loff_dir, lon_dir, out_dir, r_in_threshold=0.18, *,
                        take_no_boundary=False, grid=None, chunk=4096):
    """Merge limiter-off and limiter-on stage-1 outputs frame by frame (inputs only).

    For every limiter-off frame the inner midplane separatrix radius R_in (psiN = 1 on the
    Z = 0 row, fec.mast_v3_study.midplane_lcfs: 1941-point interpolation, walk inward from the
    row minimum in 0.3-1.4 m; grid edge when the plasma reaches it) is computed.  Frames whose
    separatrix penetrates the centre column deeply (R_in < r_in_threshold; armour at 0.20 m)
    take the limiter-on solution (psi and every meta key); all others keep limiter-off.
    Frames without a limiter-off boundary (R_in NaN) stay limiter-off unless take_no_boundary.
    Writes out_dir/psi.npy, meta.npz (+ merged_from_lon), merge.npz (take, r_in_loff,
    r_out_loff), merge.json and config.json; returns the summary dict."""
    from fec.mast_v3_study import midplane_lcfs
    loff_dir, lon_dir, out_dir = Path(loff_dir), Path(lon_dir), Path(out_dir)
    po = np.load(loff_dir / "psi.npy", mmap_mode="r")
    pl = np.load(lon_dir / "psi.npy", mmap_mode="r")
    if po.shape != pl.shape:
        raise ValueError(f"psi shapes differ: {po.shape} vs {pl.shape}")
    with np.load(loff_dir / "meta.npz") as z:
        mo = {k: z[k] for k in z.files}
    with np.load(lon_dir / "meta.npz") as z:
        ml = {k: z[k] for k in z.files}
    co, cl = _read_cfg(loff_dir), _read_cfg(lon_dir)
    if co and cl:
        diff = {k for k in set(co) | set(cl) if k not in ("limiter",) and co.get(k) != cl.get(k)}
        if diff or co.get("limiter") != "off" or cl.get("limiter") != "on":
            raise ValueError(f"stage-1 configs do not form an off/on pair: {co} vs {cl}")
    if grid is None:
        from fec.geometry import machine_geometry
        g = machine_geometry("mast_public_test")       # the solver grid (fec.machines.mast_setup)
        gR, gZ = g["grid_R"], g["grid_Z"]
    else:
        gR, gZ = (np.asarray(x, float) for x in grid)
    N = po.shape[0]
    if len(mo["psi_ax"]) != N or len(ml["psi_ax"]) != N:
        raise ValueError("meta length != psi length")
    r_in = np.full(N, np.nan)
    r_out = np.full(N, np.nan)
    for s0 in range(0, N, chunk):
        sl = slice(s0, min(N, s0 + chunk))
        r_in[sl], r_out[sl] = midplane_lcfs(np.asarray(po[sl]), mo["psi_ax"][sl], mo["psi_b"][sl],
                                            gR, gZ)
    nob = ~np.isfinite(r_in)
    take = (~nob) & (r_in < r_in_threshold)
    if take_no_boundary:
        take |= nob
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "psi.part.npy"
    out = np.lib.format.open_memmap(tmp, "w+", np.float32, po.shape)
    for s0 in range(0, N, chunk):
        sl = slice(s0, min(N, s0 + chunk))
        out[sl] = np.where(take[sl, None, None], pl[sl], po[sl])
    out.flush()
    del out
    os.replace(tmp, out_dir / "psi.npy")
    keys = sorted(set(mo) & set(ml))
    meta = {k: np.where(take, ml[k], mo[k]) for k in keys}
    meta["merged_from_lon"] = take
    np.savez(out_dir / "meta.npz", **meta)
    np.savez(out_dir / "merge.npz", take=take, r_in_loff=r_in, r_out_loff=r_out)
    lim = ml.get("limited")
    summary = dict(rule="deep_crossing", r_in_threshold=float(r_in_threshold),
                   take_no_boundary=bool(take_no_boundary), N=int(N), n_from_lon=int(take.sum()),
                   frac_from_lon=float(take.mean()) if N else 0.0,
                   n_no_boundary_loff=int(nob.sum()),
                   n_rin_below_0p20=int(np.sum(r_in < 0.20)),
                   n_rin_grid_edge=int(np.sum(r_in <= gR[0] + 1e-12)),
                   lon_limited_frac_of_taken=(float(np.mean(lim[take] > 0.5))
                                              if lim is not None and take.any() else None),
                   r_in_median_taken=float(np.median(r_in[take])) if take.any() else None,
                   loff_dir=str(loff_dir), lon_dir=str(lon_dir), loff_config=co, lon_config=cl,
                   loff_meta_sha256=sha256(loff_dir / "meta.npz"),
                   lon_meta_sha256=sha256(lon_dir / "meta.npz"), meta_keys=keys)
    cfg = dict(co or {}, limiter="merged", merge={k: summary[k] for k in
                                                   ("rule", "r_in_threshold", "take_no_boundary")})
    write_json(out_dir / "config.json", cfg)
    write_json(out_dir / "merge.json", summary)
    return summary


# ============================================================================ betaN wrapper
def betan_frozen_files():
    from fec import betan_transfer as BT
    files = {"fec/betan_transfer.py": FEC / "betan_transfer.py",
             "selection.json": BT.OUT / "selection.json",
             "val_report.json": BT.OUT / "val_report.json"}
    return {k: dict(path=str(p), sha256=sha256(p), reviewed=sha256(p) == BETAN_REVIEWED[k])
            for k, p in files.items()}


def column_flux(phi, gR, gZ, psi_ax, R=LIMITER_R, zmax=LIMITER_ZMAX):
    """Boundary-limiter flux of one map, exactly as fec.mast_v1._scal_worker --boundary-limiter:
    max of phi linearly interpolated to R on the rows |Z| <= zmax, over values below psi_ax;
    None if no such point."""
    j = int(np.searchsorted(gR, R)) - 1
    w = (R - gR[j]) / (gR[j + 1] - gR[j])
    rows = np.abs(gZ) <= zmax + 1e-9
    pl = (1 - w) * phi[rows, j] + w * phi[rows, j + 1]
    pl = pl[pl < psi_ax]
    return float(pl.max()) if pl.size else None


def limited_geometry(phi, gR, gZ, axis_box):
    """fec.betan_transfer.geometry with the boundary-only centre-column limiter.

    The frozen geometry (scorer closed-surface bisection, no masks) runs unchanged; where the
    column flux (column_flux, axis flux = the geometry's own psi_ax) exceeds its boundary flux,
    the boundary becomes psi_b = column flux and the LCFS polygon the closed axis-enclosing
    contour of phi at that level (scorer contour code, 512 points), so a, R_geo, kappa, area,
    the inside cells and chord psiN all follow the limited boundary as in stage 2.  Returns
    None if that contour cannot be found."""
    from fec import betan_transfer as BT
    geo = getattr(BT, "_geometry_unlimited", BT.geometry)(phi, gR, gZ, axis_box)
    if geo is None:
        return None
    plim = column_flux(phi, gR, gZ, geo["psi_ax"])
    if plim is None or not plim > geo["psi_b"]:
        return geo
    lcfs, op = BT._scoring()
    from contour import grid_to_physical, resample_closed  # fusion_scoring (path set by _scoring)
    am = BT.box_mask(gR, gZ, axis_box)
    _, _, iz, ir, _ = op.find_o_point(phi, gR, gZ, am)
    c = lcfs._best_contour_at(np.asarray(phi, dtype=np.float64), plim, iz, ir, None)
    if c is None:
        return None
    C = resample_closed(grid_to_physical(c, gR, gZ), 512)
    r, z = C[:, 0], C[:, 1]
    a = 0.5 * (r.max() - r.min())
    if not a > 0:
        return None
    area = 0.5 * abs(np.dot(r, np.roll(z, 1)) - np.dot(z, np.roll(r, 1)))
    return dict(geo, C=C, psi_b=plim, a=a, Rgeo=0.5 * (r.max() + r.min()),
                kappa=(z.max() - z.min()) / (2 * a), area=area, limited=True)


def _betan_pool_init(boundary_limiter):
    """Pool initializer: install limited_geometry in fec.betan_transfer (worker processes only;
    frame_record looks `geometry` up in its module at call time)."""
    if boundary_limiter:
        from fec import betan_transfer as BT
        if not hasattr(BT, "_geometry_unlimited"):
            BT._geometry_unlimited = BT.geometry
            BT.geometry = limited_geometry


def betan_records(config, psi_path, nproc=6, log=print, boundary_limiter=False):
    """fec.betan_transfer.build_records('mast') for any MAST config and psi file (same jobs,
    same _shot_job, same assembly; only the config and the flux map are parameters).  psi may
    cover a whole-shot prefix of the config (subset runs).  boundary_limiter: the workers use
    limited_geometry (same psi_b rule as stage 2 --boundary-limiter)."""
    from multiprocessing import Pool
    from fec import betan_transfer as BT
    from fec.features import load_ds
    ds = load_ds(config)
    info = ds["info"]
    BT._check_channels("MAST", info["channels"])
    order = info["order"]
    shot = np.asarray(ds["shot"])
    fidx = np.asarray(ds["fidx"])
    n = np.load(psi_path, mmap_mode="r").shape[0]
    if n > len(shot) or (n < len(shot) and shot[n - 1] == shot[n]):
        raise ValueError(f"{psi_path}: {n} frames is not a whole-shot prefix of {config}")
    src = WORK / "shots" / config
    jobs, gl = [], []
    for s in np.unique(shot[:n]):
        g = np.flatnonzero(shot[:n] == s)
        jobs.append(("MAST", str(src / f"{order[s]}.npz"), fidx[g].tolist(), str(psi_path),
                     g.tolist(), 1.0))
        gl.append(g)
    t0 = time.time()
    recs = []
    with Pool(nproc, initializer=_betan_pool_init, initargs=(bool(boundary_limiter),)) as pool:
        for i, r in enumerate(pool.imap(BT._shot_job, jobs, chunksize=2)):
            recs.extend(r)
            if i % 100 == 0:
                log(f"betaN records {config} {i}/{len(jobs)} shots {time.time() - t0:.0f}s")
    g = np.concatenate(gl)
    keys = list(recs[0].keys())
    arr = {k: np.array([r[k] for r in recs], dtype=np.float64) for k in keys}
    arr["gidx"] = g
    arr["shot"] = shot[g]
    arr["t_ms"] = np.asarray(ds["t_ms"])[g]
    return arr


def betan_apply(records, shot_all):
    """Frozen lam_fgw model + production post-processing (fec.betan_transfer.mast_betan)."""
    from fec import betan_transfer as BT
    rep = json.loads((BT.OUT / "val_report.json").read_text())
    if not rep.get("acceptance", {}).get("accepted"):
        raise StageError(f"betaN correction not accepted: {rep.get('acceptance')}")
    sel = json.loads((BT.OUT / "selection.json").read_text())
    model = sel["candidates"][sel["selected"]]["model"]
    shot_all = np.asarray(shot_all)
    if not np.array_equal(shot_all[records["gidx"]], records["shot"]):
        raise StageError("records do not match the frame order")
    out, inr = BT.mast_betan(model, records, shot_all)
    base = BT.base_betan(records, model["base"])
    old = BT.postprocess(BT.BASELINE_K * BT.base_betan(records, "v1", guard=False), records["shot"])
    info = dict(selected=sel["selected"], n=int(len(out)),
                inrange=float(np.mean(inr[np.isfinite(base)])) if np.isfinite(base).any() else None,
                median=float(np.median(out)), p5_p95=np.percentile(out, [5, 95]).tolist(),
                corr_vs_2p3xv1=float(np.corrcoef(out, old)[0, 1]) if len(out) > 2 else None)
    return out, info


# ============================================================================ chunked DIII-D ensemble
def _recipe_members(recipe_path):
    """(members, fallback list or None) exactly as fec.predict_d3d_b.predict_recipe parses them."""
    r = json.loads(Path(recipe_path).read_text())
    members = [(m.get("kind", "b"), m["tag"], float(m.get("weight", 1.0))) for m in r["members"]]
    fb = r.get("fallback")
    if isinstance(fb, dict):
        fb = [fb]
    fbl = [(f.get("kind", "b"), f["tag"], float(f.get("weight", 1.0))) for f in fb] if fb else None
    return members, fbl


def predict_b_chunked(config, out_dir, recipe_path, chunk_rows=32768, log=print, mmk=None):
    """fec.predict_d3d_b `recipe` in bounded memory, bit-identical by construction.

    predict_members accumulates every member over all N frames in float64 (~16 GB peak for
    193k frames).  Here the frames are processed in blocks of chunk_rows (a multiple of the
    predictor's 4096-row inference batch, so every network call sees the same batches, with
    the same memory order, as the one-shot run); per block the members are summed in the same
    order with the same
    expressions (first: w*P.astype(f64); then psi + w*P), the no-Thomson fallback rows (computed
    once on the same row set as predict_members) overwrite their rows, and the block is cast
    to float32 into a memmapped <config>_psi.npy.  Uses predict_d3d_b.predict_tag/_member/
    _weights unchanged; FEC_WORK / FEC_B_DEV are read by those modules from the environment.

    mmk (recipes with d3d.mmk only): dict(env=<env_train.npz>, nproc=<workers>, run_id=<d3d_b run id>,
    stall_timeout=<s>).  fec.mmk.MemberKappa then computes the TRAIN-envelope kappa of every primary member's own rows
    as they are predicted (the fallback models are not members) and <config>_mmk_kappa.npy (+ .json: run id, input
    fingerprint, SHA-256 of the ensemble files) and <config>_mmk_fallback_rows.npy are written next to the ensemble.
    psi and scal are unchanged.  Any error of that part (import, setup, a dead or stalled worker, close, save) is logged
    and leaves no kappa file (files of an earlier run are removed first); the ensemble is written regardless."""
    from fec import predict_d3d_b as PB
    from fec.features import load_ds
    from fec.features_b import no_thomson_frames
    chunk_rows = int(chunk_rows)
    if chunk_rows <= 0 or chunk_rows % CHUNK_ALIGN:
        raise ValueError(f"chunk_rows must be a positive multiple of {CHUNK_ALIGN}")
    members, fbl = _recipe_members(recipe_path)
    if any(k != "b" for k, _, _ in members + (fbl or [])):
        raise ValueError("chunked prediction supports kind 'b' members only")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_ds(config)
    N = int(ds["info"]["N"])
    mk = mk_error = None
    kfiles = [out_dir / f"{config}_mmk_kappa.json", out_dir / f"{config}_mmk_kappa.npy",
              out_dir / f"{config}_mmk_fallback_rows.npy"]
    if mmk:                 # member kappa for the d3d_mmk stage (spawn pool, before any CUDA work)
        _mmk_clear(kfiles)  # never leave an earlier run's kappa next to this run's ensemble
        try:
            from fec.mmk import MemberKappa
            mk = MemberKappa(N, [t for _, t, _ in members], mmk["env"], mmk["nproc"], log=log,
                             stall_timeout=mmk.get("stall_timeout"))
        except Exception as e:  # noqa: BLE001 - never stop the ensemble: no kappa file -> d3d_mmk fallback
            mk_error = _mmk_fail(log, config, "setup", e, None, kfiles)
    cache = {}
    t0 = time.time()
    w = PB._weights([m[2] for m in members], len(members))
    sel = np.zeros(0, np.int64)
    fp = fq = None
    if fbl:
        sel = np.flatnonzero(no_thomson_frames(ds))
        if len(sel):
            fw = PB._weights([f[2] for f in fbl], len(fbl))
            fp = 0.0
            fq = 0.0
            for (kind, tag, _), wi in zip(fbl, fw):
                P, Q = PB._member(ds, config, kind, tag, cache, rows=sel)
                fp = fp + wi * P.astype(np.float64)
                fq = fq + wi * Q.astype(np.float64)
            log(f"fallback {len(fbl)} models on {len(sel)} no-Thomson frames {time.time() - t0:.0f}s")
    tmp = out_dir / f"{config}_psi.part.npy"
    out = np.lib.format.open_memmap(tmp, "w+", np.float32, (N, 65, 65))
    scal = np.zeros((N, 5), np.float32)
    for s0 in range(0, N, chunk_rows):
        s1 = min(N, s0 + chunk_rows)
        # a slice, not an index array: X[s0:s1] keeps the input's memory order.  feat="nots"
        # inputs are Fortran-ordered (features_b: X[:, keep]); a fancy-indexed C-order copy
        # reaches the CUDA GEMM with another stride order and rounds differently (measured
        # 2026-10-08: up to 6e-8 in psi), while the slice reproduces the one-shot run exactly.
        rows = slice(s0, s1)
        psi = sc = None
        for mi, ((kind, tag, _), wi) in enumerate(zip(members, w)):
            P, Q = PB._member(ds, config, kind, tag, cache, rows=rows)
            if mk is not None:
                try:
                    mk.add(mi, s0, P)
                except Exception as e:  # noqa: BLE001 - never stop the ensemble: no kappa file -> d3d_mmk fallback
                    mk, mk_error = None, _mmk_fail(log, config, "member kappa", e, mk, kfiles)
            psi = wi * P.astype(np.float64) if psi is None else psi + wi * P
            sc = wi * Q.astype(np.float64) if sc is None else sc + wi * Q
            del P, Q
        if len(sel):
            m = (sel >= s0) & (sel < s1)
            if m.any():
                psi[sel[m] - s0] = fp[m]
                sc[sel[m] - s0] = fq[m]
        psi = psi.reshape(-1, 65, 65).astype(np.float32)
        sc = sc.astype(np.float32)
        if not (np.isfinite(psi).all() and np.isfinite(sc).all()):
            raise ValueError(f"non-finite prediction in rows {s0}:{s1}")
        out[s0:s1] = psi
        scal[s0:s1] = sc
        del psi, sc
        log(f"rows {s1}/{N} {time.time() - t0:.0f}s")
    out.flush()
    del out
    os.replace(tmp, out_dir / f"{config}_psi.npy")
    np.save(out_dir / f"{config}_scal.npy", scal)
    kmeta = None
    if mk is not None:      # + the rows the fallback wrote: d3d_mmk never edits them (fec/mmk.py)
        try:
            mk.close()
            fr = out_dir / f"{config}_mmk_fallback_rows.npy"
            np.save(fr, np.asarray(sel, np.int64))
            inputs, inputs_sha = mmk_inputs(config, recipe_path, N, [t for _, t, _ in members],
                                            WORK / "ds" / config / "info.json")
            kmeta = mk.save(out_dir / f"{config}_mmk_kappa.npy", extra={
                "fallback_rows": dict(file=fr.name, n=int(len(sel)), sha256=sha256(fr)),
                "run_id": mmk.get("run_id"), "inputs": inputs, "inputs_sha256": inputs_sha,
                "ensemble_psi_sha256": sha256(out_dir / f"{config}_psi.npy"),
                "ensemble_scal_sha256": sha256(out_dir / f"{config}_scal.npy")})
            log(f"mmk member kappa {kmeta['shape']} nan per member {kmeta['nan_per_member']} "
                f"{kmeta['seconds']:.0f}s")
        except Exception as e:  # noqa: BLE001 - the ensemble is already written; drop only the kappa part
            kmeta, mk_error = None, _mmk_fail(log, config, "close/save", e, mk, kfiles)
    meta = {"members": [{"kind": k, "tag": t, "weight": float(x)} for (k, t, _), x in zip(members, w)],
            "fallback": None if not fbl else [{"kind": k, "tag": t, "weight": float(x)}
                                              for k, t, x in fbl],
            "fallback_frames": int(len(sel)) if fbl else 0,
            "config": config, "N": N, "scal_cols": PB.SCAL_COLS, "inference": f"{PB.DEV} fp32",
            "chunked": {"chunk_rows": chunk_rows, "driver": "fec/run_final.py predict_b_chunked",
                        "recipe": str(recipe_path), "recipe_sha256": sha256(recipe_path),
                        "fec_work": str(WORK), "seconds": round(time.time() - t0, 1)}}
    if kmeta is not None or mk_error:
        meta["mmk_kappa"] = kmeta or {"error": mk_error}
    (out_dir / f"{config}_meta.json").write_text(json.dumps(meta, indent=1))
    log(f"{out_dir} ({N}, 65, 65) fallback frames {meta['fallback_frames']} "
        f"{time.time() - t0:.0f}s")
    return meta


def mmk_inputs(config, recipe_path, n_frames, tags, ds_info):
    """(fingerprint dict, its SHA-256) of the inputs that fix the members' kappa: config, frame count, members recipe
    file, ds info.json, member tags.  d3d_b records it in the kappa meta; d3d_mmk recomputes it for the current run."""
    ds_info = Path(ds_info)
    d = dict(config=str(config), N=int(n_frames), recipe_sha256=sha256(recipe_path),
             ds_info_sha256=sha256(ds_info) if ds_info.is_file() else None, tags=[str(t) for t in tags])
    return d, hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def _mmk_clear(files):
    for f in files:
        try:
            Path(f).unlink()
        except FileNotFoundError:
            pass


def _mmk_fail(log, config, where, e, mk, files):
    """Log a member-kappa failure, end its pool, remove its files; the error string for the d3d_b meta."""
    log(f"WARNING mmk member kappa FAILED in {where} ({type(e).__name__}: {e}); no {config}_mmk_kappa.npy, the "
        f"ensemble is unaffected and d3d_mmk will fall back")
    if mk is not None:
        try:
            mk.terminate()
        except Exception:  # noqa: BLE001
            pass
    try:
        _mmk_clear(files)
    except Exception:  # noqa: BLE001
        pass
    return f"{where}: {type(e).__name__}: {e}"


def _internal_exec(argv):
    """`run_final.py _exec --env=K=V ... -- cmd ...`: run cmd with K=V added to its environment.

    Used under fec/gpu_slot.py so that the slot wrapper registers in the shared
    <FEC_WORK>/gpu_slots while the GPU child sees a routed FEC_WORK (e.g. fec_work_idx)."""
    if "--" not in argv:
        raise SystemExit("_exec needs -- before the command")
    i = argv.index("--")
    env = dict(os.environ)
    for a in argv[:i]:
        if not a.startswith("--env=") or "=" not in a[6:]:
            raise SystemExit(f"_exec: bad argument {a!r} (use --env=KEY=VALUE)")
        k, v = a[6:].split("=", 1)
        env[k] = v
    return subprocess.run(argv[i + 1:], env=env).returncode


def _internal_predict_b(argv):
    """`python -B fec/run_final.py _predict_b ...` (subprocess of the d3d_b stage)."""
    ap = argparse.ArgumentParser(prog="run_final.py _predict_b")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--chunk-rows", type=int, default=32768)
    ap.add_argument("--mmk-env", default=None, help="d3d.mmk recipes: TRAIN envelope of the member kappa")
    ap.add_argument("--mmk-nproc", type=int, default=6, help="d3d.mmk recipes: member kappa worker processes")
    ap.add_argument("--mmk-run-id", default=None, help="d3d.mmk recipes: the d3d_b run id recorded in the kappa meta")
    ap.add_argument("--mmk-stall-timeout", type=float, default=None,
                    help="d3d.mmk recipes: seconds without a finished member-kappa task before that part stops")
    a = ap.parse_args(argv)
    predict_b_chunked(a.config, a.out, a.recipe, a.chunk_rows,
                      log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True),
                      mmk=dict(env=a.mmk_env, nproc=a.mmk_nproc, run_id=a.mmk_run_id,
                               stall_timeout=a.mmk_stall_timeout) if a.mmk_env else None)
    return 0


# ============================================================================ comparisons
def _eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def cmp_arrays(a, b, rows=None, chunk=4096, tol=TOL):
    """Chunked comparison of two (memmapped) arrays, optionally on the first `rows` rows."""
    n = len(a) if rows is None else int(rows)
    if a.shape[1:] != b.shape[1:] or len(a) < n or len(b) < n or (rows is None and len(a) != len(b)):
        return dict(ok=False, reason=f"shape {a.shape} vs {b.shape} rows {rows}")
    mx = 0.0
    nd = 0
    nf_mismatch = 0
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0 + chunk))
        x = np.asarray(a[sl], np.float64)
        y = np.asarray(b[sl], np.float64)
        fx, fy = np.isfinite(x), np.isfinite(y)
        both_nf = ~fx & ~fy
        with np.errstate(invalid="ignore"):
            same_nf = both_nf & ((np.isnan(x) & np.isnan(y)) | (x == y))
        nf_mismatch += int(np.sum(fx != fy)) + int(np.sum(both_nf & ~same_nf))
        both = fx & fy
        d = np.abs(x[both] - y[both])
        nd += int(np.sum(d > 0))
        if d.size:
            mx = max(mx, float(d.max()))
    exact = nd == 0 and nf_mismatch == 0
    return dict(ok=bool(nf_mismatch == 0 and mx <= tol), exact=bool(exact), max_abs=mx,
                n_diff=nd, nonfinite_mismatch=nf_mismatch, rows=n)


def cmp_npz(fa, fb, rows=None):
    """Key-by-key comparison of two npz files (optionally first `rows` rows of 1-D+ arrays)."""
    with np.load(fa) as za, np.load(fb) as zb:
        ka, kb = set(za.files), set(zb.files)
        res = {"keys_only_a": sorted(ka - kb), "keys_only_b": sorted(kb - ka), "keys": {}}
        for k in sorted(ka & kb):
            x, y = za[k], zb[k]
            if rows is not None and x.ndim >= 1 and len(x) >= rows and len(y) >= rows:
                x, y = x[:rows], y[:rows]
            if x.dtype.kind in "fcbiu" and y.dtype.kind in "fcbiu" and x.shape == y.shape and x.size:
                r = cmp_arrays(x.reshape(len(x), -1) if x.ndim else x.reshape(1, 1),
                               y.reshape(len(y), -1) if y.ndim else y.reshape(1, 1))
            else:
                r = dict(ok=_eq(x, y), exact=_eq(x, y))
            res["keys"][k] = r
    res["ok"] = all(r["ok"] for r in res["keys"].values()) and not res["keys_only_a"]
    res["exact"] = all(r.get("exact", False) for r in res["keys"].values())
    return res


def cmp_shot_dirs(da, db, stems):
    """Per-shot npz caches: identical keys and arrays (NaN-equal) for every stem."""
    bad = []
    for s in stems:
        fa, fb = Path(da) / f"{s}.npz", Path(db) / f"{s}.npz"
        if not (fa.exists() and fb.exists()):
            bad.append((s, "missing"))
            continue
        with np.load(fa) as za, np.load(fb) as zb:
            if set(za.files) != set(zb.files):
                bad.append((s, "keys"))
                continue
            for k in za.files:
                if not _eq(za[k], zb[k]):
                    bad.append((s, k))
                    break
    return dict(ok=not bad, exact=not bad, n=len(stems), mismatches=bad[:20], n_bad=len(bad))


def cmp_file_hashes(da, db, names):
    out = {}
    for n in names:
        fa, fb = Path(da) / n, Path(db) / n
        out[n] = (fa.exists() and fb.exists() and sha256(fa) == sha256(fb))
    return dict(ok=all(out.values()), exact=all(out.values()), files=out)


def cmp_pred_dirs(da, db, config):
    r = {}
    for kind in ("psi", "scal"):
        fa, fb = Path(da) / f"{config}_{kind}.npy", Path(db) / f"{config}_{kind}.npy"
        if not (fa.exists() and fb.exists()):
            return dict(ok=False, reason=f"missing {fa if not fa.exists() else fb}")
        r[kind] = cmp_arrays(np.load(fa, mmap_mode="r"), np.load(fb, mmap_mode="r"))
    return dict(ok=all(x["ok"] for x in r.values()), exact=all(x["exact"] for x in r.values()), **r)


def cmp_mast_dirs(da, db, rows):
    """Stage-1 outputs: psi rows and every common meta key on the first `rows` frames.
    db may be an unfinished production dir (meta_partial.npz with 'done')."""
    da, db = Path(da), Path(db)
    mb = db / "meta.npz"
    if not mb.exists():
        mb = db / "meta_partial.npz"
        if not mb.exists():
            return dict(ok=False, reason=f"no meta in {db}")
        with np.load(mb) as z:
            done = int(z["done"]) if "done" in z.files else 0
        if done < rows:
            return dict(ok=False, pending=True, reason=f"{db} done {done} < {rows}")
    r = {"psi": cmp_arrays(np.load(da / "psi.npy", mmap_mode="r"),
                           np.load(db / "psi.npy", mmap_mode="r"), rows=rows)}
    m = cmp_npz(da / "meta.npz", mb, rows=rows)
    m["keys"].pop("done", None)
    r["meta"] = {k: v for k, v in m["keys"].items()}
    ok = r["psi"]["ok"] and all(v["ok"] for v in r["meta"].values())
    ex = r["psi"]["exact"] and all(v.get("exact", False) for v in r["meta"].values())
    return dict(ok=ok, exact=ex, rows=rows, ref=str(db), meta_ref=mb.name, **r)


# ============================================================================ driver context
class Ctx:
    def __init__(self, args, recipe):
        self.args = args
        self.recipe = recipe
        self.ip_timebase = recipe_ip_timebase(recipe)
        idx_style = self.ip_timebase == "idx" or bool(recipe["d3d"].get("work"))
        self.L = Layout(args.split, args.subset_shots, root=args.root,
                        d3d_work=recipe_d3d_work(recipe, getattr(args, "d3d_work", None)),
                        d3d_fresh=idx_style)
        if not args.dry_run and args.stage != "status":       # status is read-only
            self.L.logs.mkdir(parents=True, exist_ok=True)
        self.cmds = []
        self._git = None

    # -------------------------------------------------------------- logging / manifest
    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if not self.L.logs.exists():           # dry run / status: print only
            return
        with open(self.L.logs / "run_final.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def key(self, name):
        """Manifest/log key: the submission stage is keyed by its --submission-name."""
        if name == "submission":
            name = self.args.submission_name
        return f"{name}{self.L.sfx}"

    def stage_log(self, name):
        return self.L.logs / f"{self.key(name)}.log"

    def manifest_file(self, name):
        return self.L.manifest_dir / f"{self.key(name)}.json"

    def manifest(self, name):
        f = self.manifest_file(name)
        return json.loads(f.read_text()) if f.exists() else {}

    def code_hashes(self):
        """sha256 of every fec/*.py (fec/ is untracked in git; hashes are its provenance)."""
        if getattr(self, "_code", None) is None:
            self._code = {p.name: sha256(p) for p in sorted(FEC.glob("*.py"))}
        return self._code

    def git(self):
        if self._git is None:
            from fec.runlog import git_state
            self._git = git_state()
        return self._git

    def runlog(self, row):
        if self.args.no_runlog or self.args.dry_run:
            return
        from fec import runlog
        try:
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                runlog.append(dict(row, owner=self.args.owner))
        except OSError as e:          # registry is best effort; the manifest is authoritative
            self.log(f"runlog append failed: {e}")

    # -------------------------------------------------------------- execution
    def check_writable(self, *paths):
        for p in paths:
            if not self.L.writable(p):
                raise StageError(f"refusing to write outside {self.L.root} (or the private-fold "
                                 f"caches): {p}")

    def sh(self, cmd, log, gpu=False, env=None):
        cmd = [str(c) for c in cmd]
        if gpu:
            if env:     # the slot wrapper keeps the shared FEC_WORK (its gpu_slots/ registry);
                cmd = [PY, "-B", str(FEC / "run_final.py"), "_exec"] + \
                    [f"--env={k}={v}" for k, v in env.items()] + ["--"] + cmd   # child env only
                env = None
            cmd = [PY, "-B", str(FEC / "gpu_slot.py"), "run", "--owner", self.args.owner,
                   "--priority", self.args.gpu_priority, "--"] + cmd
        envd = dict(os.environ, PYTHONUNBUFFERED="1")
        envd.update(env or {})
        extra = " ".join(f"{k}={v}" for k, v in (env or {}).items())
        self.cmds.append((extra + " " if extra else "") + cmdline(cmd))
        self.log(f"$ {(extra + ' ') if extra else ''}{cmdline(cmd)}")
        if self.args.dry_run:
            return
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n# {now()} {(extra + ' ') if extra else ''}{cmdline(cmd)}\n")
            fh.flush()
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO),
                                env=envd).returncode
        if rc:
            raise StageError(f"exit code {rc}: {cmdline(cmd)} (log {log})")

    def data_env(self, cfg=None):
        """FEC_WORK of a data stage: the shadow (public) or the config's canonical root."""
        return {"FEC_WORK": str(self.L.cfg_work(cfg) if cfg else self.L.data_work)}

    def link(self, link, target, replace=False):
        """ensure_dir_link, logged; dry runs only print."""
        msg = f"link {link} -> {target}"
        self.cmds.append(msg)
        if self.args.dry_run:
            self.log(msg + " (dry run)")
            return "dry_run"
        st = ensure_dir_link(link, target, replace=replace)
        self.log(f"{msg}: {st}")
        return st

    def d3d_root_links(self, root, need=("models", "hf_data")):
        """A DIII-D work root other than WORK sees the model store and the HF mirror via links."""
        root = Path(root)
        if same_dir(root, self.L.work):
            return {}
        # hf_data only when the canonical mirror exists (public data live on the E: root)
        return {n: self.link(root / n, self.L.work / n) for n in need
                if n != "hf_data" or (self.L.work / n).is_dir()}

    def stage_lock(self, name):
        """One writer per stage and root: <manifest>/<key>.lock holds the owning pid.  A live
        holder (e.g. the DIII-D branch started in a second shell) is waited for; a dead one is
        cleared.  Returns the lock path (None in dry runs)."""
        if self.args.dry_run:
            return None
        import psutil
        lock = self.L.manifest_dir / f"{self.key(name)}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        t0 = last = time.time()
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return lock
            except FileExistsError:
                try:
                    pid = int(lock.read_text().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if not pid or not psutil.pid_exists(pid):
                    lock.unlink(missing_ok=True)
                    continue
                if time.time() - last > 300 or last == t0:
                    self.log(f".. {name}: running in pid {pid}; waiting ({time.time() - t0:.0f}s)")
                    last = time.time() + 1
                time.sleep(20)

    def run(self, name):
        lock = self.stage_lock(name)
        try:
            return self._run(name)
        finally:
            if lock:
                lock.unlink(missing_ok=True)

    def _run(self, name):
        fn, done = STAGES[name]
        prev = self.manifest(name)
        try:
            complete = prev.get("status") == "ok" and bool(done(self))
        except Exception:  # noqa: BLE001 - a missing/garbled output means "not done"
            complete = False
        if not self.args.force and complete:
            self.log(f"== {name}: done ({prev.get('finished')}), skipping (--force to redo)")
            return prev
        rid = f"final_{self.L.split}{self.L.sfx}.{name}.{time.strftime('%Y%m%d%H%M%S')}"
        self.log(f"== {name}: start ({rid})")
        self.cmds = []
        rec = dict(stage=name, split=self.L.split, subset_shots=self.L.subset_shots,
                   run_id=rid, started=now(), recipe=self.recipe.get("name"),
                   argv=sys.argv, log=str(self.stage_log(name)), status="interrupted",
                   git=self.git(), code_sha256=self.code_hashes())
        self.runlog(dict(run_id=rid, event="start", status="running",
                         purpose=f"final pipeline {self.L.split}{self.L.sfx} stage {name}",
                         cmd="python -B fec/run_final.py " + " ".join(sys.argv[1:]),
                         log=str(self.stage_log(name)), outputs=str(self.L.root), **self.git()))
        t0 = time.time()
        try:
            info = fn(self) or {}
            rec.update(status="dry_run" if self.args.dry_run else "ok", info=info)
        except Exception as e:
            rec.update(status="failed", error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc()[-4000:])
            self.log(f"!! {name} FAILED: {type(e).__name__}: {e}")
            raise
        finally:
            rec.update(finished=now(), seconds=round(time.time() - t0, 1), commands=self.cmds)
            if not self.args.dry_run:
                write_json(self.manifest_file(name), rec)
            self.runlog(dict(run_id=rid, event="finish",
                             status="ok" if rec.get("status") in ("ok", "dry_run") else
                             ("stopped" if rec.get("status") == "interrupted" else "failed"),
                             metrics=dict(seconds=rec["seconds"])))
        self.log(f"== {name}: {rec['status']} in {rec['seconds']:.0f}s")
        return rec


# ============================================================================ stages
def _n_frames(cfg, canonical=True, L=None):
    d = (L.ds_dir(cfg, canonical) if L else WORK / "ds" / cfg)
    return json.loads((d / "info.json").read_text())["N"]


def st_download(c):
    L = c.L
    info = {}
    for cfg in L.configs:
        fs = shot_files(cfg)
        info[cfg] = dict(local_files=len(fs), root=str(fs[0].parent) if fs else None)
    missing = [cfg for cfg in L.configs if not info[cfg]["local_files"]]
    if missing:
        if L.shadow:
            raise StageError(f"public configs missing locally: {missing}")
        c.check_writable(L.work / "hf_data")
        c.sh([PY, "-B", FEC / "final_data.py", "download", *missing], c.stage_log("download"))
        for cfg in missing:
            fs = shot_files(cfg)
            info[cfg] = dict(local_files=len(fs), root=str(fs[0].parent) if fs else None,
                             downloaded=True)
    if (c.args.remote or missing) and not c.args.dry_run:
        from fec.final_data import list_repo_files, remote_files   # retried on HTTP 429 / 5xx (pkgfix D1)
        repo = hf_dataset_repo(L.split)
        files = list_repo_files(repo, log=c.log)
        for cfg in L.configs:
            n = len(remote_files(repo, cfg, files)[0])
            info[cfg].update(remote_files=n, repo=repo)
            if n != info[cfg]["local_files"]:
                raise StageError(f"{cfg}: local {info[cfg]['local_files']} != remote {n} files")
    if not c.args.dry_run and not all(info[cfg]["local_files"] for cfg in L.configs):
        raise StageError(f"configs still missing: {info}")
    return info


def done_download(c):
    return all(shot_files(cfg) for cfg in c.L.configs)


def st_ipguard(c):
    """Inputs-only guard of the idx Ip time axis on the DIII-D fold, before extraction:
    fec/ip_timebase.py scan --config <cfg> (parquet list when no ds exists yet; labels only ever
    for diii_d_train TRAIN shots) and check (exit 1 on any failed condition)."""
    L = c.L
    if c.ip_timebase != "idx":
        return dict(skipped=f"recipe ip_timebase {c.ip_timebase}: the idx guard does not apply")
    root = L.cfg_work(L.d3d_cfg)
    c.check_writable(root / "ip_timebase")
    links = c.d3d_root_links(root, need=("hf_data",))
    log = c.stage_log("ipguard")
    env = c.data_env(L.d3d_cfg)
    c.sh([PY, "-B", FEC / "ip_timebase.py", "scan", "--config", L.d3d_cfg,
          "--workers", c.args.workers], log, env=env)
    try:
        c.sh([PY, "-B", FEC / "ip_timebase.py", "check", "--config", L.d3d_cfg], log, env=env)
    except StageError as e:
        raise StageError(f"STOP: idx Ip time-axis guard FAILED on {L.d3d_cfg} ({e}). Do not "
                         f"predict: the fold needs a reviewed decision (research/reviews/"
                         f"fec_ip_timebase_review_20261008.md sec. 6-7); see {log}") from None
    # label-free time-axis defect guard (fec/input_fixes2.py, docs/research/
    # D3D_INPUT_AUDIT2_20261008.md): magnetics/EFIT/Ip axis gaps, duplicates, backward steps,
    # empty +-1 ms box windows, frames past a record end, Thomson axis resets, chord mismatches
    ig = root / "input_guard"
    c.check_writable(ig)
    c.sh([PY, "-B", FEC / "input_fixes2.py", "scan", "--config", L.d3d_cfg,
          "--workers", c.args.workers, "--out", ig], log, env=env)
    issue12 = None
    try:
        c.sh([PY, "-B", FEC / "input_fixes2.py", "guard", "--config", L.d3d_cfg, "--out", ig],
             log, env=env)
    except StageError as e:
        # starter issue #12: a Thomson CORE dup/back step alone (organiser-announced Final
        # cases) is tolerated when fec/final_data.py issue12 accepts it (fec.extract.ts_interp
        # sorts samples by time and ignores empty rows); anything else still STOPS
        try:
            c.sh([PY, "-B", FEC / "final_data.py", "issue12", "--config", L.d3d_cfg, "--scan",
                  ig / f"{L.d3d_cfg}_scan.jsonl", "--out", ig / "issue12.json"], log, env=env)
            issue12 = str(ig / "issue12.json")
            c.log(f"WARNING: input_fixes2 guard failed only on issue-#12 Thomson core samples "
                  f"within tolerance ({issue12}); continuing")
        except StageError:
            raise StageError(f"STOP: DIII-D input time-axis guard (fec/input_fixes2.py) FAILED "
                             f"on {L.d3d_cfg} ({e}). Do not predict before the defect is "
                             f"reviewed (docs/research/D3D_INPUT_AUDIT2_20261008.md); see "
                             f"{log}") from None
    if c.args.dry_run:
        return {}
    rows = json.loads((root / "ip_timebase" / f"{L.d3d_cfg}_shots.json").read_text())
    from fec.ip_timebase import guard
    g = guard(rows)
    if g["n_failed"]:
        raise StageError(f"STOP: guard re-evaluation failed: {g['conditions']}")
    n_scan = sum(1 for _ in open(ig / f"{L.d3d_cfg}_scan.jsonl", encoding="utf-8"))
    return dict(config=L.d3d_cfg, n=g["n"], classes=g["classes"], links=links,
                conditions={k: v["pass"] for k, v in g["conditions"].items()},
                shots_json=str(root / "ip_timebase" / f"{L.d3d_cfg}_shots.json"),
                input_guard=dict(passed=True, scan=str(ig / f"{L.d3d_cfg}_scan.jsonl"),
                                 shots=n_scan, issue12_tolerated=issue12,
                                 code_sha256=sha256(FEC / "input_fixes2.py")))


def done_ipguard(c):
    if c.ip_timebase != "idx":
        return True
    root = c.L.cfg_work(c.L.d3d_cfg)
    f = root / "ip_timebase" / f"{c.L.d3d_cfg}_shots.json"
    g = root / "input_guard" / f"{c.L.d3d_cfg}_scan.jsonl"
    info = c.manifest("ipguard").get("info", {})
    return f.exists() and g.exists() and bool(info.get("input_guard", {}).get("passed"))


def extract_timebase_args(c, cfg):
    """--ip-timebase for DIII-D configs (MAST has no Ip time-axis fix: module default)."""
    return ["--ip-timebase", c.ip_timebase] if cfg.startswith("diii_d") else []


def check_fresh_shots(d, cfg, want_tb):
    """fec/extract.py skips existing per-shot npz, so a DIII-D dir must be empty or hold an
    extraction on the same Ip axis (resume); anything else is refused."""
    d = Path(d)
    n = sum(1 for p in d.glob("*.npz") if not p.name.endswith(".part.npz")) if d.is_dir() else 0
    if not n or not cfg.startswith("diii_d"):
        return dict(existing=n)
    tb = shots_ip_timebase(d)
    if tb is None:
        raise StageError(f"{d}: {n} shot npz without _meta.json; extract into a FRESH dir")
    if tb != want_tb:
        raise StageError(f"{d}: {n} shot npz extracted with ip_timebase={tb!r}, recipe needs "
                         f"{want_tb!r}; extract.py skips existing shots, so use a FRESH dir "
                         f"(--root / d3d.work), never mix Ip axes")
    return dict(existing=n, resumed_ip_timebase=tb)


TS_RULES = ("auto", "runmax", "sort")


def ts_rule(c):
    """Thomson time rule of the extraction (starter issue #12): 'runmax' = the organisers' rule
    (fec/extract_final.py: keep a sample only if later than every earlier one), 'sort' =
    fec/extract.py (sort by time).  auto: runmax on private_test, sort on public_test (the
    public features every model was trained/validated on; the two rules give identical inputs
    on all public shots, FINAL_PACKAGE_20261012)."""
    r = getattr(c.args, "ts_rule", "auto") or "auto"
    if r == "auto":
        return "runmax" if c.L.split == "private_test" else "sort"
    return r


def shots_ts_rule(shots_dir):
    f = Path(shots_dir) / "_meta.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("thomson_rule", "sort")


def st_extract(c):
    L = c.L
    info = {}
    rule = ts_rule(c)
    script = FEC / ("extract_final.py" if rule == "runmax" else "extract.py")
    for cfg in L.configs:
        d = L.shots_dir(cfg)
        c.check_writable(d)
        fresh = check_fresh_shots(d, cfg, c.ip_timebase) if not c.args.dry_run else {}
        if fresh.get("existing") and shots_ts_rule(d) != {"runmax": "running_max"}.get(rule, rule):
            raise StageError(f"{d}: {fresh['existing']} shot npz extracted with Thomson rule "
                             f"{shots_ts_rule(d)!r}, this run needs {rule!r}; extract skips existing "
                             f"shots, so use a FRESH dir")
        if cfg.startswith("diii_d"):
            c.d3d_root_links(L.cfg_work(cfg), need=("hf_data",))
        c.sh([PY, "-B", script, cfg, "--workers", c.args.workers] +
             extract_timebase_args(c, cfg), c.stage_log("extract"), env=c.data_env(cfg))
        if not c.args.dry_run:
            n_npz = sum(1 for _ in d.glob("*.npz")) - sum(1 for _ in d.glob("*.part.npz"))
            n_src = len(shot_files(cfg))
            if n_npz != n_src:
                raise StageError(f"{cfg}: {n_npz} shot npz for {n_src} parquet files")
            tb = shots_ip_timebase(d)
            if cfg.startswith("diii_d") and tb != c.ip_timebase:
                raise StageError(f"{d}/_meta.json ip_timebase {tb!r} != {c.ip_timebase!r}")
            info[cfg] = dict(shots=n_npz, dir=str(d), ip_timebase=tb, before=fresh,
                             thomson_rule=shots_ts_rule(d))
    return info


def done_extract(c):
    for cfg in c.L.configs:
        d = c.L.shots_dir(cfg)
        if not (d / "_meta.json").exists():
            return False
        if len(json.loads((d / "_meta.json").read_text())["order"]) != \
                sum(1 for _ in d.glob("*.npz")):
            return False
        if cfg.startswith("diii_d") and shots_ip_timebase(d) != c.ip_timebase:
            return False
        if shots_ts_rule(d) != {"runmax": "running_max"}.get(ts_rule(c), ts_rule(c)):
            return False
    return True


ORDER_SCRIPT = r'''
import json, os, sys
import numpy as np
from datasets import load_dataset_builder, load_dataset
repo, split, out = sys.argv[1], sys.argv[2], sys.argv[3]
items = json.loads(sys.argv[4])
res = {}
for cfg, meta, shots in items:
    b = load_dataset_builder(repo, cfg)
    keys = list(b.config.data_files)
    if split not in keys and len(keys) == 1:      # Final dataset: one split of another name
        split = keys[0]
    names = [os.path.basename(str(f)).split("?")[0] for f in b.config.data_files[split]]
    local = [s + ".parquet" for s in json.load(open(meta))["order"]]
    r = {"remote": len(names), "local": len(local), "same_order": names == local, "rows": []}
    ds = load_dataset(repo, cfg, split=split, streaming=True).select_columns(["efit_times"])
    for i, row in enumerate(ds):
        if i >= 3:
            break
        z = np.load(os.path.join(shots, local[i][:-8] + ".npz"))
        r["rows"].append(bool(len(row["efit_times"]) == len(z["t"]) and
                              np.allclose(row["efit_times"], z["t"])))
    res[cfg] = r
    print(cfg, json.dumps(r), flush=True)
json.dump(res, open(out, "w"), indent=1)
'''


def st_order(c):
    L = c.L
    vpy = L.work / "venv_hf" / "Scripts" / "python.exe"
    if not vpy.exists():
        raise StageError(f"missing {vpy} (venv with `datasets`)")
    out = L.root / "order_check.json"
    items = [[cfg, str(L.shots_dir(cfg) / "_meta.json"), str(L.shots_dir(cfg))]
             for cfg in L.configs]
    script = L.root / "tools" / "order_check_hf.py"      # own dir: nothing else importable there
    if not c.args.dry_run:
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(ORDER_SCRIPT.lstrip(), encoding="utf-8")
    c.sh([vpy, "-I", script, hf_dataset_repo(L.split), L.split, out, json.dumps(items)],
         c.stage_log("order"))
    if c.args.dry_run:
        return {}
    res = json.loads(out.read_text())
    bad = [cfg for cfg, r in res.items() if not (r["same_order"] and all(r["rows"]) and r["rows"])]
    if bad:
        raise StageError(f"HF stream order check failed for {bad}: {res}")
    return res


def done_order(c):
    f = c.L.root / "order_check.json"
    if not f.exists():
        return False
    res = json.loads(f.read_text())
    return set(res) == set(c.L.configs) and all(r["same_order"] and r["rows"] and all(r["rows"])
                                               for r in res.values())


def geometry_check(cfg, ref_cfg):
    """Coil elements and EFIT grid of cfg (several shots) == the reference config's."""
    import pyarrow.parquet as pq
    from fec.geometry import GEOM_COLS, check_geometry_constant
    const, bad = check_geometry_constant(cfg)
    fa = shot_files(cfg)[0]
    ga = {k: pq.read_table(fa, columns=GEOM_COLS).column(k)[0].as_py() for k in GEOM_COLS}
    if shot_files(ref_cfg):
        fb = shot_files(ref_cfg)[0]
        gb = {k: pq.read_table(fb, columns=GEOM_COLS).column(k)[0].as_py() for k in GEOM_COLS}
    else:       # reproduction package without the public inputs: the frozen geometry cache
        gj = WORK / "geometry" / f"{ref_cfg}.json"   # (fec.geometry.machine_geometry's format)
        gb = {k: v for k, v in json.loads(gj.read_text()).items() if k in GEOM_COLS}
    return dict(constant_within=bool(const), first_mismatch=bad,
                equal_to=ref_cfg, equal=bool(ga == gb),
                differing_columns=[k for k in GEOM_COLS if ga[k] != gb[k]])


def st_dataset(c):
    L = c.L
    info = {}
    for cfg in L.configs:
        c.check_writable(L.ds_dir(cfg))
        c.sh([PY, "-B", FEC / "dataset.py", cfg], c.stage_log("dataset"), env=c.data_env(cfg))
        if c.args.dry_run:
            continue
        di = json.loads((L.ds_dir(cfg) / "info.json").read_text())
        sm = json.loads((L.shots_dir(cfg) / "_meta.json").read_text())
        if di["order"] != sm["order"]:
            raise StageError(f"{L.ds_dir(cfg)} order != {L.shots_dir(cfg)}/_meta.json order")
        ref = "mast_public_test" if cfg.startswith("mast") else "diii_d_public_test"
        geo = geometry_check(cfg, ref)
        info[cfg] = dict(N=di["N"], shots=len(di["order"]), n_split=di["n_split"], geometry=geo,
                         ds=str(L.ds_dir(cfg)), ip_timebase=sm.get("ip_timebase", "starter"))
        if cfg.startswith("mast") and not (geo["equal"] and geo["constant_within"]):
            raise StageError(f"{cfg} MAST coil/grid geometry differs from mast_public_test (the "
                             f"solver geometry, fec.machines.mast_setup): {geo}")
        if not geo["equal"]:
            c.log(f"WARNING {cfg} geometry differs from {ref}: {geo}")
    return info


def done_dataset(c):
    return all((c.L.ds_dir(cfg) / "info.json").exists() for cfg in c.L.configs)


def _pred_done(d, cfg, N):
    f = Path(d) / f"{cfg}_psi.npy"
    q = Path(d) / f"{cfg}_scal.npy"
    return f.exists() and q.exists() and np.load(f, mmap_mode="r").shape[0] == N


def d3d_pred_N(c):
    """Frame count of the DIII-D ds the predictor reads (FEC_WORK = L.d3d_pred_work)."""
    return json.loads((c.L.d3d_pred_work / "ds" / c.L.d3d_cfg / "info.json").read_text())["N"]


def d3d_input_check(c):
    """The predictor's DIII-D inputs carry the recipe's Ip time axis (review condition: never
    feed idx inputs to starter-trained models or the reverse) and the ds matches its shots."""
    L = c.L
    sd, dd = L.d3d_pred_work / "shots" / L.d3d_cfg, L.d3d_pred_work / "ds" / L.d3d_cfg
    if c.args.dry_run:
        return {}
    tb = shots_ip_timebase(sd)
    if tb != c.ip_timebase:
        raise StageError(f"{sd}/_meta.json ip_timebase {tb!r} != recipe d3d.ip_timebase "
                         f"{c.ip_timebase!r}: refusing to mix Ip time axes")
    di = json.loads((dd / "info.json").read_text())
    if di["order"] != json.loads((sd / "_meta.json").read_text())["order"]:
        raise StageError(f"{dd} was not built from {sd} (shot order differs)")
    return dict(fec_work=str(L.d3d_pred_work), shots=str(sd), ds=str(dd), ip_timebase=tb,
                N=int(di["N"]), ds_info_sha256=sha256(dd / "info.json"))


def d3d_env(c, extra=None):
    """Environment of DIII-D prediction subprocesses: FEC_WORK only when it is not WORK."""
    env = dict(extra or {})
    if not same_dir(c.L.d3d_pred_work, c.L.work):
        env["FEC_WORK"] = str(c.L.d3d_pred_work)
    return env


def model_inventory(c, tags, hash_models=True, store=None):
    """model.pt of every tag in the predictor's model store (or `store`): exists (else StageError
    listing every missing tag), size, sha256 (provenance of the frozen members)."""
    store = Path(store) if store is not None else c.L.d3d_pred_work / "models"
    miss = [t for t in dict.fromkeys(tags) if not (store / t / "model.pt").exists()]
    if miss:
        raise StageError(f"missing model.pt in {store} for {len(miss)} tag(s): {miss} "
                         f"(train them first; see the runbook rebuild section)")
    inv = {}
    for t in dict.fromkeys(tags):
        f = store / t / "model.pt"
        inv[t] = dict(bytes=f.stat().st_size, sha256=sha256(f) if hash_models else None)
    return dict(store=str(store), store_resolved=str(store.resolve()), models=inv)


def d3d_part(c, key):
    r = c.recipe["d3d"].get(key)
    if not r:
        raise StageError(f"recipe has no d3d.{key}")
    return r


def d3d_final_name(recipe):
    d = recipe["d3d"]
    for key in ("mix", "b", "seq"):
        if d.get(key):
            return d[key]["name"]
    raise ValueError("recipe has no DIII-D predictor")


def st_d3d_seq(c):
    L, r = c.L, d3d_part(c, "seq")
    out = L.preds / r["name"]
    c.check_writable(out)
    chk = d3d_input_check(c)
    cmd = [PY, "-B", FEC / "train_d3d_seq.py", "predict", "--tags", ",".join(r["tags"]),
           "--out", out, "--config", L.d3d_cfg, "--v2", r.get("v2") or "",
           "--wv2", repr(float(r.get("wv2", 0.0)))]
    c.sh(cmd, c.stage_log("d3d_seq"), gpu=True, env=d3d_env(c) or None)
    if c.args.dry_run:
        return {}
    meta = json.loads((out / f"{L.d3d_cfg}_meta.json").read_text())
    if not meta.get("finite") or meta["N"] != d3d_pred_N(c):
        raise StageError(f"bad seq predictions: {meta}")
    return dict(out=str(out), meta=meta, inputs=chk)


def done_d3d_seq(c):
    return _pred_done(c.L.preds / d3d_part(c, "seq")["name"], c.L.d3d_cfg, d3d_pred_N(c))


def st_d3d_b(c):
    L, r = c.L, d3d_part(c, "b")
    out = L.preds / r["name"]
    c.check_writable(out)
    pred = FEC / "predict_d3d_b.py"
    info = {}
    chunk = int(r.get("chunk_rows") or 0)
    if r.get("recipe"):     # predict_d3d_b.py recipe JSON (weighted b/seq members, fallbacks)
        rp = resolve_path(r["recipe"], REPO)
        pr = json.loads(rp.read_text())
        fbs = pr.get("fallback") or []
        fbs = [fbs] if isinstance(fbs, dict) else fbs
        tags = [m["tag"] for m in pr["members"] + fbs if m.get("kind", "b") == "b"]
        fb = None
        if chunk:           # bounded-memory wrapper, same arithmetic (predict_b_chunked)
            cmd = [PY, "-B", FEC / "run_final.py", "_predict_b", "--config", L.d3d_cfg,
                   "--out", out, "--recipe", rp, "--chunk-rows", chunk]
        else:
            cmd = [PY, "-B", pred, "recipe", "--config", L.d3d_cfg, "--out", out, "--recipe", rp]
        info["predict_recipe"] = dict(path=str(rp), sha256=sha256(rp), content=pr)
    else:
        if chunk:
            raise StageError("d3d.b.chunk_rows needs the recipe form (d3d.b.recipe)")
        tags = list(r["tags"])
        fb = r.get("fallback")
        if r.get("weights"):
            cmd = [PY, "-B", pred, "run", "--config", L.d3d_cfg, "--out", out, "--tags",
                   ",".join(tags), "--weights", ",".join(repr(float(w)) for w in r["weights"])] + \
                (["--fallback", fb] if fb else [])
        else:               # legacy positional form (equal weights)
            cmd = [PY, "-B", pred, L.d3d_cfg, out, ",".join(tags)] + ([fb] if fb else [])
    if c.recipe["d3d"].get("mmk"):  # member kappa of d3d_mmk, computed while the members are predicted
        if not (r.get("recipe") and chunk):
            raise StageError("d3d.mmk needs the chunked recipe form of d3d.b (d3d.b.recipe + chunk_rows)")
        envf, mlinks = mmk_env(c)
        q = c.recipe["d3d"]["mmk"]
        run_id = "dry-run" if c.args.dry_run else f"d3d_b.{time.strftime('%Y%m%dT%H%M%S')}.{os.getpid()}.{os.urandom(4).hex()}"
        cmd += ["--mmk-env", envf, "--mmk-nproc", c.args.workers, "--mmk-run-id", run_id,
                "--mmk-stall-timeout", repr(float(q.get("stall_timeout_s", 600.0)))]
        info["mmk"] = dict(env=str(envf), links=mlinks, workers=int(c.args.workers), run_id=run_id)
    if not same_dir(L.d3d_pred_work, L.work):
        info["links"] = c.d3d_root_links(L.d3d_pred_work, need=("models",))
    if not c.args.dry_run:          # ext/hist models need fec_work/ds_b/<cfg>/xb.npy (extract_b)
        import torch
        info["inputs"] = d3d_input_check(c)
        info["models"] = model_inventory(c, tags + ([fb] if fb else []),
                                         hash_models=not c.args.no_model_hash)
        store = L.d3d_pred_work / "models"
        feats = {}
        for t in dict.fromkeys(tags + ([fb] if fb else [])):
            ck = torch.load(store / t / "model.pt", weights_only=False, map_location="cpu",
                            mmap=True)
            feats[t] = ck.get("feat", ck["args"].get("feat", "base"))
            del ck
        info["feats"] = feats
        need = [t for t, f in feats.items() if f in ("ext", "hist")]
        if need and not (L.d3d_pred_work / "ds_b" / L.d3d_cfg / "xb.npy").exists():
            raise StageError(f"{need} need fec/extract_b.py {L.d3d_cfg} (reads E: DATA only; "
                             f"private folds live in hf_data: run extract_b with a shot_files "
                             f"fallback first)")
    dev = run_device(r.get("device", "cpu"))
    c.sh(cmd, c.stage_log("d3d_b"), gpu=dev.startswith("cuda"),
         env=d3d_env(c, {"FEC_B_DEV": dev}))
    if c.args.dry_run:
        return {}
    meta = json.loads((out / f"{L.d3d_cfg}_meta.json").read_text())
    if meta.get("N") != d3d_pred_N(c):
        raise StageError(f"d3d_b predictions N {meta.get('N')} != ds {d3d_pred_N(c)}")
    if info.get("mmk"):
        kp = mmk_kappa_path(c)
        info["mmk"].update(kappa=str(kp), kappa_sha256=sha256(kp) if kp.exists() else None)
    return dict(info, out=str(out), meta=meta, device=dev, chunk_rows=chunk or None,
                psi_sha256=sha256(out / f"{L.d3d_cfg}_psi.npy"),
                scal_sha256=sha256(out / f"{L.d3d_cfg}_scal.npy"))


def done_d3d_b(c):
    done = _pred_done(c.L.preds / d3d_part(c, "b")["name"], c.L.d3d_cfg, d3d_pred_N(c))
    if done and c.recipe["d3d"].get("mmk"):     # d3d.mmk recipes: the members' kappa as well
        done = mmk_kappa_path(c).exists()
    return done


# ---------------------------------------------------------------------------- d3d_train (opt-in)
def check_train_block(t):
    """Validate recipe d3d.train (jobs, tag_replace, trainer, bx; see D3D_TRAIN_ALL22_BX)."""
    if not isinstance(t, dict) or not isinstance(t.get("jobs"), str):
        raise ValueError("recipe d3d.train needs 'jobs': a job list of fec/train_d3d_b.py lines")
    unknown = set(t) - {"jobs", "tag_replace", "trainer", "bx"}
    if unknown:
        raise ValueError(f"unknown d3d.train keys {sorted(unknown)}")
    if t.get("trainer", "serial") not in TRAINERS:
        raise ValueError(f"recipe d3d.train.trainer must be one of {TRAINERS}, not "
                         f"{t.get('trainer')!r}")
    for rep in t.get("tag_replace") or []:
        if not isinstance(rep, str) or "=" not in rep or not rep.split("=", 1)[0]:
            raise ValueError(f"d3d.train.tag_replace entries are 'OLD=NEW' strings, not {rep!r}")
    extra = set(t.get("bx") or {}) - set(D3D_TRAIN_BX)
    if extra:
        raise ValueError(f"unknown d3d.train.bx keys {sorted(extra)} (known {sorted(D3D_TRAIN_BX)})")


def train_jobs(recipe):
    """[(tag, train_d3d_b argv)] of recipe d3d.train: the job list's fec/train_d3d_b.py arguments
    with the tag replacements applied, parsed exactly as fec/train_d3d_bx.py parses them."""
    t = recipe["d3d"].get("train")
    if not t:
        raise StageError("recipe has no d3d.train")
    from fec.train_d3d_bx import member_argvs
    ns = argparse.Namespace(member=None, jobs=str(resolve_path(t["jobs"], REPO)),
                            tag_replace=list(t.get("tag_replace") or []))
    return [(v[v.index("--tag") + 1], v) for v in member_argvs(ns)]


def d3d_b_tags(recipe):
    """b-kind model tags the d3d.b predictor reads (recipe members + fallbacks, or tags)."""
    r = recipe["d3d"].get("b") or {}
    if r.get("recipe"):
        pr = json.loads(resolve_path(r["recipe"], REPO).read_text())
        fbs = pr.get("fallback") or []
        fbs = [fbs] if isinstance(fbs, dict) else fbs
        return [m["tag"] for m in pr["members"] + fbs if m.get("kind", "b") == "b"]
    return list(r.get("tags") or []) + ([r["fallback"]] if r.get("fallback") else [])


def d3d_trainer(c):
    """--d3d-trainer, else recipe d3d.train.trainer, else 'serial' (the existing path)."""
    t = d3d_part(c, "train")
    tr = getattr(c.args, "d3d_trainer", None) or "recipe"
    tr = t.get("trainer", "serial") if tr == "recipe" else tr
    if tr not in TRAINERS:
        raise StageError(f"unknown DIII-D trainer {tr!r} (one of {TRAINERS})")
    return tr


def st_d3d_train(c):
    """Train the recipe's DIII-D members (opt-in: `d3d_train` or `all --with-train`).

    FEC_WORK = the DIII-D work root: its ds/diii_d_train + ds_b/diii_d_train TRAIN caches (labels of
    TRAIN(+VAL, +TEST for --alldata members) shots only, as train_d3d_b) and models/ (the
    predictor's store).  bx: one fec/train_d3d_bx.py process in the recipe's production layout
    (it refuses existing model dirs; --force passes --overwrite).  serial: one fec/train_d3d_b.py
    job per member, the existing path; members whose model.pt exists are kept unless --force."""
    L, t = c.L, d3d_part(c, "train")
    trainer = d3d_trainer(c)
    jobs = train_jobs(c.recipe)
    tags = [g for g, _ in jobs]
    miss = [g for g in d3d_b_tags(c.recipe) if g not in tags]
    if miss:
        raise StageError(f"d3d.b reads {len(miss)} tag(s) that d3d.train does not train: {miss}")
    root = L.d3d_work
    store = root / "models"
    jobs_path = resolve_path(t["jobs"], REPO)
    alldata = any("--alldata" in v for _, v in jobs)
    info = dict(trainer=trainer, jobs=str(jobs_path), tags=tags, store=str(store),
                fec_work=str(root), alldata=alldata)
    if not c.args.dry_run:
        sfx = "_all" if alldata else ""
        dd, bd = root / "ds" / "diii_d_train", root / "ds_b" / "diii_d_train"
        need = [dd / "info.json", dd / "psi.npy", bd / f"bnd2{sfx}.npy", bd / f"bnd_x{sfx}.npy",
                bd / "env_train.npz"] + ([root / "test_eval_b" / "DONE"] if alldata else [])
        absent = [str(f) for f in need if not f.exists()]
        if absent:
            raise StageError(f"training inputs missing: {absent} (fec/dataset.py diii_d_train, "
                             f"fec/bnd_b.py [--alldata]; --alldata members also need the one-time "
                             f"internal-TEST evaluation marker test_eval_b/DONE)")
        info["jobs_sha256"] = sha256(jobs_path)
    env = {"FEC_WORK": str(root)}
    log = c.stage_log("d3d_train")
    timing = L.manifest_dir / f"{c.key('d3d_train')}_timing.json"
    t0 = time.time()
    if trainer == "bx":
        b = dict(D3D_TRAIN_BX, **(t.get("bx") or {}))
        cmd = [PY, "-B", FEC / "train_d3d_bx.py", "--jobs", jobs_path]
        for rep in t.get("tag_replace") or []:
            cmd += ["--tag_replace", rep]
        cmd += ["--group_size", b["group_size"], "--pix_impl", b["pix_impl"], "--ln_impl",
                b["ln_impl"], "--vram_gb", b["vram_gb"], "--data_gpu", b["data_gpu"],
                "--timing_json", timing] + (["--overwrite"] if c.args.force else [])
        info["bx"] = b
        c.sh(cmd, log, gpu=True, env=env)
    else:
        for g, v in jobs:
            if not c.args.force and not c.args.dry_run and (store / g / "model.pt").exists():
                c.log(f".. d3d_train: {g} exists, kept (--force retrains it)")
                continue
            c.sh([PY, "-B", FEC / "train_d3d_b.py"] + v, log, gpu=True, env=env)
    if c.args.dry_run:
        return {}
    info["seconds"] = round(time.time() - t0, 1)
    if trainer == "bx" and timing.exists():
        info["timing"] = json.loads(timing.read_text())
    info["models"] = model_inventory(c, tags, hash_models=not c.args.no_model_hash, store=store)
    return info


def done_d3d_train(c):
    store = c.L.d3d_work / "models"
    return all((store / g / "model.pt").exists() for g, _ in train_jobs(c.recipe))


def st_d3d_mix(c):
    L, r = c.L, d3d_part(c, "mix")
    out = L.preds / r["name"]
    c.check_writable(out)
    src = {n: L.preds / n for n in r["members"]}
    c.log(f"in-process: blend_preds({r['members']}, {L.d3d_cfg}, {out})")
    c.cmds.append(f"in-process blend_preds {json.dumps(r['members'])} -> {out}")
    if c.args.dry_run:
        return {}
    meta = blend_preds(r["members"], L.d3d_cfg, out, src)
    if meta["N"] != d3d_pred_N(c):
        raise StageError(f"mix frames {meta['N']} != ds")
    return meta


def done_d3d_mix(c):
    return _pred_done(c.L.preds / d3d_part(c, "mix")["name"], c.L.d3d_cfg, d3d_pred_N(c))


def q95h_dirs(c):
    """(source head preds dir, q95-hybrid output dir) of a recipe with d3d.q95h."""
    q = c.recipe["d3d"].get("q95h")
    if not q:
        raise StageError("recipe has no d3d.q95h")
    src = Path(c.args.q95h_src) if getattr(c.args, "q95h_src", None) else \
        c.L.preds / d3d_final_name(c.recipe)
    return src, c.L.preds / (d3d_final_name(c.recipe) + q.get("suffix", "_q95h"))


def st_d3d_q95h(c):
    """fec/q95_hybrid.py apply --mode G1 on the DIII-D predictions (CPU, ~55 CPU-min / 193k
    frames).  Frozen constants are checked before and after; the physics parts are always
    recomputed (--recompute, never a cached --parts).  psi and scalar columns 1-4 must stay
    byte-identical to the head; only q95 changes.  On a failure of the apply step the stage
    records fallback=head and the submission uses the head predictions (review condition 4)."""
    L, q = c.L, c.recipe["d3d"]["q95h"]
    src, out = q95h_dirs(c)
    c.check_writable(out)
    ds = L.d3d_pred_work / "ds" / L.d3d_cfg
    bad = {k: (q.get(k), v) for k, v in Q95H_FROZEN.items() if q.get(k) != v}
    env_sha = sha256(FEC / "q95_hybrid_envelope.json")
    if env_sha != Q95H_FROZEN["envelope_sha256"]:
        bad["envelope_file"] = (env_sha, Q95H_FROZEN["envelope_sha256"])
    if bad and not c.args.allow_unreviewed:
        raise StageError(f"q95 hybrid differs from the reviewed frozen setting: {bad} "
                         f"(--allow-unreviewed to proceed)")
    cmd = [PY, "-B", FEC / "q95_hybrid.py", "apply", "--src", src, "--ds", ds, "--out", out,
           "--mode", q["mode"], "--config", L.d3d_cfg, "--nproc", c.args.workers, "--recompute"]
    if c.args.dry_run:
        c.sh(cmd, c.stage_log("d3d_q95h"))
        return {}
    for f in ("psi", "scal"):
        if not (src / f"{L.d3d_cfg}_{f}.npy").exists():
            raise StageError(f"no head predictions {src}/{L.d3d_cfg}_{f}.npy: run d3d_b first")
    info = dict(src=str(src), out=str(out), ds=str(ds), frozen=Q95H_FROZEN,
                src_scal_sha256=sha256(src / f"{L.d3d_cfg}_scal.npy"),
                src_psi_sha256=sha256(src / f"{L.d3d_cfg}_psi.npy"),
                q95_hybrid_py_sha256=sha256(FEC / "q95_hybrid.py"))
    try:
        dst = out / f"{L.d3d_cfg}_psi.npy"
        if dst.exists():          # apply copies psi only when absent: never keep a stale copy
            dst.unlink()
        c.sh(cmd, c.stage_log("d3d_q95h"))
        meta = json.loads((out / f"{L.d3d_cfg}_meta.json").read_text())["q95_hybrid"]
        a = np.load(src / f"{L.d3d_cfg}_scal.npy")
        b = np.load(out / f"{L.d3d_cfg}_scal.npy")
        chk = dict(mode=meta.get("mode") == Q95H_FROZEN["mode"],
                   tau=meta.get("tau") == Q95H_FROZEN["tau"],
                   tf_turns=meta.get("tf_turns") == Q95H_FROZEN["tf_turns"],
                   envelope=meta.get("envelope_sha256") == Q95H_FROZEN["envelope_sha256"],
                   source=meta.get("source_scal_sha256") == info["src_scal_sha256"],
                   shape=a.shape == b.shape,
                   other_scalars_identical=bool(np.array_equal(a[:, 1:], b[:, 1:])),
                   q95_finite=bool(np.isfinite(b[:, 0]).all()),
                   psi_identical=sha256(dst) == info["src_psi_sha256"])
        if not all(chk.values()):
            raise StageError(f"q95 hybrid output checks failed: {chk}")
        info.update(fallback=None, checks=chk, q95_hybrid=meta,
                    scal_sha256=sha256(out / f"{L.d3d_cfg}_scal.npy"),
                    q95_changed_frac=float(np.mean(a[:, 0] != b[:, 0])))
    except Exception as e:  # noqa: BLE001 - review condition 4: keep H if the P step fails
        if q.get("fallback") != "head":
            raise
        c.log(f"WARNING d3d_q95h FAILED ({type(e).__name__}: {e}); the submission falls back "
              f"to the head predictions {src}")
        info.update(fallback="head", error=f"{type(e).__name__}: {e}")
    return info


def done_d3d_q95h(c):
    src, out = q95h_dirs(c)
    i = c.manifest("d3d_q95h").get("info", {})
    if i.get("fallback") == "head":
        return i.get("src") == str(src)
    return (i.get("src") == str(src) and _pred_done(out, c.L.d3d_cfg, d3d_pred_N(c)) and
            i.get("src_scal_sha256") == sha256(src / f"{c.L.d3d_cfg}_scal.npy"))


def d3d_submit_dir(c):
    """DIII-D preds the submission uses: --d3d-dir, else the lirep4 dir (recipe d3d.lirep4,
    unless that stage fell back to its source), else the q95-hybrid dir (recipe d3d.q95h, unless
    that stage fell back to the head), else preds/<final name>."""
    if c.args.d3d_dir:
        return Path(c.args.d3d_dir), None
    d, fb = _d3d_submit_dir(c)
    if mmk_fell_back(c):    # d3d.mmk recipes: an mmk fallback is a submission-level fallback (submission manifest)
        c.log("WARNING: d3d_mmk fell back to its source; the DIII-D file carries no kappa finishing")
        fb = "mmk_src" if fb is None else f"{fb}+mmk_src"
    return d, fb


def _d3d_submit_dir(c):
    if not c.recipe["d3d"].get("lirep4"):
        return _d3d_pre_lirep_dir(c)
    src, out = lirep4_dirs(c)
    m = c.manifest("d3d_lirep4")
    if c.args.dry_run and not m:
        return out, None
    if m.get("status") != "ok":
        raise StageError("recipe has d3d.lirep4: run the d3d_lirep4 stage first")
    if m["info"].get("fallback") == "src":
        c.log(f"WARNING: d3d_lirep4 fell back to its source ({m['info'].get('error')}); "
              f"submitting {src}")
        return src, "lirep4_src"
    return out, None


def lirep4_dirs(c):
    """(source preds dir, lirep4 output dir) of a recipe with d3d.lirep4: the source is what the
    submission would use without lirep4 (the q95-hybrid dir, or the head on its fallback)."""
    q = c.recipe["d3d"].get("lirep4")
    if not q:
        raise StageError("recipe has no d3d.lirep4")
    src, _ = _d3d_pre_lirep_dir(c)
    return src, src.parent / (src.name + q.get("suffix", "_lirep4"))


def lirep4_work(c):
    """FEC_WORK of lirep_build4: it reads ds/<d3d cfg> (the predictor's caches) and the TRAIN-only
    envelope ds_b/diii_d_train/env_train.npz.  The predictor root when it has both, else
    <root>/view_lirep4 whose two entries link to them."""
    L = c.L
    root = L.d3d_pred_work
    if (root / "ds_b" / "diii_d_train" / "env_train.npz").exists():
        return root, {}
    v = L.root / "view_lirep4"
    links = {f"ds/{L.d3d_cfg}": c.link(v / "ds" / L.d3d_cfg, root / "ds" / L.d3d_cfg,
                                        replace=True),
             "ds_b/diii_d_train": c.link(v / "ds_b" / "diii_d_train",
                                         L.d3d_work / "ds_b" / "diii_d_train", replace=True)}
    return v, links


def st_d3d_lirep4(c):
    """fec/d3d_rdspec.py lirep_build4 on the submission's DIII-D preds (CPU, ~15 min / 193k
    frames): frames whose li(2) of the predicted psi (valid contour inside env_train) lies outside [0.1, 5],
    in any phase (in practice ramp-down frames), get a label-free li-range repair to li 4.9 where reachable;
    every other psi row and every scalar stay byte-identical (checked).  On a failure the stage
    records fallback=src and the submission uses the source (q95-hybrid) predictions."""
    L, q = c.L, c.recipe["d3d"]["lirep4"]
    src, out = lirep4_dirs(c)
    c.check_writable(out)
    log = c.stage_log("d3d_lirep4")
    cmd = [PY, "-B", FEC / "d3d_rdspec.py", "lirep_build4", "--src", src, "--out", out,
           "--config", L.d3d_cfg, "--nproc", c.args.workers]
    if c.args.dry_run:
        c.sh(cmd, log, env={"FEC_WORK": str(L.d3d_pred_work)})
        return {}
    for f in ("psi", "scal"):
        if not (src / f"{L.d3d_cfg}_{f}.npy").exists():
            raise StageError(f"no source predictions {src}/{L.d3d_cfg}_{f}.npy")
    work, links = lirep4_work(c)
    envf = work / "ds_b" / "diii_d_train" / "env_train.npz"
    info = dict(src=str(src), out=str(out), work=str(work), links=links,
                src_psi_sha256=sha256(src / f"{L.d3d_cfg}_psi.npy"),
                src_scal_sha256=sha256(src / f"{L.d3d_cfg}_scal.npy"),
                env_train_sha256=sha256(envf), d3d_rdspec_py_sha256=sha256(FEC / "d3d_rdspec.py"),
                frozen=dict(q))
    try:
        if out.exists():          # lirep_build4 never overwrites: clear this run's own output
            if not is_under(out, L.root):
                raise StageError(f"refusing to clear {out} outside {L.root}")
            for f in out.iterdir():
                f.unlink()
        c.sh(cmd, log, env={"FEC_WORK": str(work)})
        meta = json.loads((out / f"{L.d3d_cfg}_lirep4_meta.json").read_text())
        a = np.load(src / f"{L.d3d_cfg}_scal.npy")
        b = np.load(out / f"{L.d3d_cfg}_scal.npy")
        chk = dict(target=float(meta.get("target")) == float(q.get("target", 4.9)),
                   scal_identical=bool(np.array_equal(a, b)),
                   max_abs_dpsi_outside_zero=float(meta["max_abs_dpsi_outside_repaired"]) == 0.0,
                   n_frames=np.load(out / f"{L.d3d_cfg}_psi.npy", mmap_mode="r").shape[0] ==
                   d3d_pred_N(c))
        if not all(chk.values()):
            raise StageError(f"lirep4 output checks failed: {chk}")
        info.update(fallback=None, checks=chk, triggered=meta["triggered"],
                    repaired=meta["repaired"], left_out=meta["left_out"],
                    psi_sha256=sha256(out / f"{L.d3d_cfg}_psi.npy"),
                    scal_sha256=sha256(out / f"{L.d3d_cfg}_scal.npy"))
    except Exception as e:  # noqa: BLE001 - keep the source if the repair step fails
        if q.get("fallback") != "src":
            raise
        c.log(f"WARNING d3d_lirep4 FAILED ({type(e).__name__}: {e}); the submission falls back "
              f"to {src}")
        info.update(fallback="src", error=f"{type(e).__name__}: {e}")
    return info


def done_d3d_lirep4(c):
    src, out = lirep4_dirs(c)
    i = c.manifest("d3d_lirep4").get("info", {})
    if i.get("fallback") == "src":
        return i.get("src") == str(src)
    return (i.get("src") == str(src) and _pred_done(out, c.L.d3d_cfg, d3d_pred_N(c)) and
            i.get("src_psi_sha256") == sha256(src / f"{c.L.d3d_cfg}_psi.npy"))


def _d3d_pre_lirep_dir(c):
    """The pre-lirep4 submission source: the d3d_mmk output for a recipe with d3d.mmk (its source on the mmk fallback),
    else _d3d_pre_mmk_dir (unchanged for every recipe without d3d.mmk)."""
    if not c.recipe["d3d"].get("mmk"):
        return _d3d_pre_mmk_dir(c)
    src, out = mmk_dirs(c)
    m = c.manifest("d3d_mmk")
    if c.args.dry_run and not m:
        return out, None
    if m.get("status") != "ok":
        raise StageError("recipe has d3d.mmk: run the d3d_mmk stage first")
    if m["info"].get("fallback") == "src":
        c.log(f"WARNING: d3d_mmk fell back to its source ({m['info'].get('error')}); using {src}")
        return src, "mmk_src"
    return out, None


def _d3d_pre_mmk_dir(c):
    """The pre-mmk (and, without d3d.mmk, pre-lirep4) source: the d3d_dec output for a recipe with d3d.dec (its source
    on the dec fallback), else the legacy d3d_submit_dir rule (_d3d_pre_dec_dir, unchanged for every recipe without
    d3d.dec)."""
    if not c.recipe["d3d"].get("dec"):
        return _d3d_pre_dec_dir(c)
    src, out = dec_dirs(c)
    m = c.manifest("d3d_dec")
    if c.args.dry_run and not m:
        return out, None
    if m.get("status") != "ok":
        raise StageError("recipe has d3d.dec: run the d3d_dec stage first")
    if m["info"].get("fallback") == "src":
        c.log(f"WARNING: d3d_dec fell back to its source ({m['info'].get('error')}); using {src}")
        return src, "dec_src"
    return out, None


# ---------------------------------------------------------------------------- d3d_mmk (final_MTfull_mmk)
def mmk_dirs(c):
    """(source preds dir, mmk output dir) of a recipe with d3d.mmk: the source is the d3d_dec output (or its source on
    the dec fallback), the output <source>_mmk."""
    q = c.recipe["d3d"].get("mmk")
    if not q:
        raise StageError("recipe has no d3d.mmk")
    src, _ = _d3d_pre_mmk_dir(c)
    return src, src.parent / (src.name + q.get("suffix", "_mmk"))


def mmk_kappa_path(c):
    """The members' kappa (N, M) that d3d_b writes for a d3d.mmk recipe (fec/mmk.py MemberKappa)."""
    return c.L.preds / d3d_part(c, "b")["name"] / f"{c.L.d3d_cfg}_mmk_kappa.npy"


def mmk_fallback_rows_path(c):
    """The rows d3d_b's no-Thomson fallback wrote (d3d.mmk recipes): d3d_mmk never edits them."""
    return c.L.preds / d3d_part(c, "b")["name"] / f"{c.L.d3d_cfg}_mmk_fallback_rows.npy"


def mmk_env(c):
    """(path, links) of the TRAIN-only envelope env_train.npz of the mmk kappa functional: lirep4's (lirep4_work)."""
    work, links = lirep4_work(c)
    return work / "ds_b" / "diii_d_train" / "env_train.npz", links


def mmk_recipe_tags(c):
    """The primary members of the d3d.b recipe (the columns of the kappa file), in recipe order."""
    members, _ = _recipe_members(resolve_path(c.recipe["d3d"]["b"]["recipe"], REPO))
    return [t for _, t, _ in members]


def st_d3d_mmk(c):
    """fec/mmk.py apply on the post-decoder preds (CPU, --workers processes, ~0.2 CPU-s per frame): the frozen kappa step
    of fec/mmk_op.py moves each frame's TRAIN-envelope kappa half way to the equal mean of the members' own kappa
    (d3d_b's kappa file); frames the no-Thomson fallback wrote, and frames the skip rule rejects, stay bit-identical; the
    scalar file is copied (q95 / betaN byte-identical, checked).  The operator file and constants must equal the recipe's
    d3d.mmk block and the envelope its SHA-256.  On a failed step or check the stage records fallback=src and lirep4
    runs on the source (d3d_dec) preds."""
    L, q = c.L, c.recipe["d3d"]["mmk"]
    src, out = mmk_dirs(c)
    c.check_writable(out)
    log = c.stage_log("d3d_mmk")
    kap, fbr = mmk_kappa_path(c), mmk_fallback_rows_path(c)
    envf, links = mmk_env(c)
    cmd = [PY, "-B", FEC / "mmk.py", "apply", "--src", src, "--out", out, "--config", L.d3d_cfg, "--kappa", kap,
           "--skip-rows", fbr, "--env", envf, "--nproc", c.args.workers,
           "--min-members", int(q.get("min_members", 0)), "--operator", Path(q.get("operator", "fec/mmk_op.py")).stem,
           "--stall-timeout", repr(float(q.get("stall_timeout_s", 600.0)))]
    if c.args.dry_run:
        c.sh(cmd, log)
        return {}
    for f in ("psi", "scal"):
        if not (src / f"{L.d3d_cfg}_{f}.npy").exists():
            raise StageError(f"no source predictions {src}/{L.d3d_cfg}_{f}.npy")
    info = dict(src=str(src), out=str(out), kappa=str(kap), fallback_rows=str(fbr), env=str(envf), links=links,
                frozen=dict(q), workers=int(c.args.workers),
                src_psi_sha256=sha256(src / f"{L.d3d_cfg}_psi.npy"),
                src_scal_sha256=sha256(src / f"{L.d3d_cfg}_scal.npy"),
                mmk_py_sha256=sha256(FEC / "mmk.py"), mmk_op_py_sha256=sha256(FEC / "mmk_op.py"),
                kappa_sha256=sha256(kap) if kap.is_file() else None,
                fallback_rows_sha256=sha256(fbr) if fbr.is_file() else None)
    try:
        from fec import mmk as MMK
        bad = MMK.check_frozen(q)
        for f in (kap, fbr, envf):
            if not Path(f).is_file():
                bad.append(f"missing {f} (d3d_b of a d3d.mmk recipe writes the kappa and the fallback rows)")
        if not bad:
            info["env_train_sha256"] = sha256(envf)
            km = json.loads(MMK.kappa_meta_path(kap).read_text())
            want = dict(tags=mmk_recipe_tags(c), env_train_sha256=q["env_train_sha256"], sha256=info["kappa_sha256"],
                        complete=True)
            bad += [f"kappa meta {k} {km.get(k)!r} != {v!r}" for k, v in want.items() if km.get(k) != v]
            if info["env_train_sha256"] != q["env_train_sha256"]:
                bad.append(f"env_train sha256 {info['env_train_sha256']} != {q['env_train_sha256']}")
            shp = np.load(kap, mmap_mode="r").shape
            if shp != (d3d_pred_N(c), len(want["tags"])):
                bad.append(f"kappa shape {shp} != ({d3d_pred_N(c)}, {len(want['tags'])})")
            bad += mmk_stale(c, km)
        if bad:
            raise StageError(f"d3d_mmk inputs are not the frozen ones: {bad}")
        if out.exists():          # apply never overwrites: clear this run's own output
            if not is_under(out, L.root):
                raise StageError(f"refusing to clear {out} outside {L.root}")
            for f in out.iterdir():
                f.unlink()
        c.sh(cmd, log)
        meta = json.loads((out / f"{L.d3d_cfg}_mmk_meta.json").read_text())
        n_fb = int(np.load(fbr).size)
        chk = dict(scal_identical=sha256(out / f"{L.d3d_cfg}_scal.npy") == info["src_scal_sha256"],
                   n_frames=np.load(out / f"{L.d3d_cfg}_psi.npy", mmap_mode="r").shape[0] == d3d_pred_N(c),
                   finite=bool(meta["finite"]), changed_rows_are_edited_rows=bool(meta["edited_rows_equal_ok_rows"]),
                   operator=meta["mmk_op_sha256"] == q["operator_sha256"],
                   env_train=meta["env_train_sha256"] == q["env_train_sha256"],
                   fallback_rows_not_edited=(meta["status_counts"].get("fallback_frame") == n_fb ==
                                             meta["fallback_frames_not_edited"]))
        if not all(chk.values()):
            raise StageError(f"d3d_mmk output checks failed: {chk}")
        info.update(fallback=None, checks=chk, psi_sha256=sha256(out / f"{L.d3d_cfg}_psi.npy"),
                    **{k: meta[k] for k in ("frames", "members", "edited", "status_counts", "skip_counts", "T_nan",
                                            "fallback_frames_not_edited", "min_members", "too_few_members_frames",
                                            "finite_members_hist", "kappa_nan_per_member", "operator",
                                            "realised_over_intended", "c_quantiles", "seconds")})
    except Exception as e:  # noqa: BLE001 - keep the source if the mmk step fails
        if q.get("fallback") != "src":
            raise
        c.log(f"WARNING d3d_mmk FAILED ({type(e).__name__}: {e}); lirep4 and the submission fall back to {src}")
        info.update(fallback="src", error=f"{type(e).__name__}: {e}")
    return info


def mmk_stale(c, km):
    """Stale-kappa guard: problems of a kappa meta against the current run (the d3d_b manifest's run id, the input
    fingerprint recomputed now, the ensemble files d3d_b wrote).  [] = this run's kappa."""
    L, bad = c.L, []
    want_id = ((c.manifest("d3d_b").get("info") or {}).get("mmk") or {}).get("run_id")
    if not want_id or km.get("run_id") != want_id:
        bad.append(f"kappa run_id {km.get('run_id')!r} != d3d_b manifest run_id {want_id!r}")
    _, ish = mmk_inputs(L.d3d_cfg, resolve_path(c.recipe["d3d"]["b"]["recipe"], REPO), d3d_pred_N(c),
                        mmk_recipe_tags(c), L.d3d_pred_work / "ds" / L.d3d_cfg / "info.json")
    if km.get("inputs_sha256") != ish:
        bad.append(f"kappa inputs_sha256 {km.get('inputs_sha256')} != the current inputs' {ish}")
    bdir = L.preds / d3d_part(c, "b")["name"]
    for k in ("psi", "scal"):
        f = bdir / f"{L.d3d_cfg}_{k}.npy"
        h = sha256(f) if f.is_file() else None
        if km.get(f"ensemble_{k}_sha256") != h:
            bad.append(f"kappa ensemble_{k}_sha256 {km.get(f'ensemble_{k}_sha256')} != {f.name} {h}")
    return bad


def mmk_fell_back(c):
    """d3d.mmk recipes: True when the d3d_mmk stage recorded fallback=src (lirep4 then read the d3d_dec output)."""
    return bool(c.recipe["d3d"].get("mmk")) and \
        (c.manifest("d3d_mmk").get("info") or {}).get("fallback") == "src"


def done_d3d_mmk(c):
    """Done only for the same source, the same member kappa / fallback rows and the same fec/mmk*.py."""
    src, out = mmk_dirs(c)
    i = c.manifest("d3d_mmk").get("info", {})
    kap, fbr = mmk_kappa_path(c), mmk_fallback_rows_path(c)
    same = (i.get("src") == str(src) and i.get("mmk_py_sha256") == sha256(FEC / "mmk.py") and
            i.get("mmk_op_py_sha256") == sha256(FEC / "mmk_op.py") and
            i.get("kappa_sha256") == (sha256(kap) if kap.is_file() else None) and
            i.get("fallback_rows_sha256") == (sha256(fbr) if fbr.is_file() else None))
    if i.get("fallback") == "src":
        return same
    return (same and _pred_done(out, c.L.d3d_cfg, d3d_pred_N(c)) and
            i.get("src_psi_sha256") == sha256(src / f"{c.L.d3d_cfg}_psi.npy"))


# ---------------------------------------------------------------------------- d3d_dec (final_A22li_dec)
def dec_dirs(c):
    """(source preds dir, decoder output dir) of a recipe with d3d.dec: the source is the q95-hybrid dir (or the head
    on its fallback), the output <source>_dec."""
    q = c.recipe["d3d"].get("dec")
    if not q:
        raise StageError("recipe has no d3d.dec")
    src, _ = _d3d_pre_dec_dir(c)
    return src, src.parent / (src.name + q.get("suffix", "_dec"))


def dec_weights_dir(c):
    """Decoder files dec_s<seed>.pt: --dec-dir, else <FEC_WORK>/<d3d.dec.dir> (train.py --stage offdec / infer.py)."""
    if getattr(c.args, "dec_dir", None):
        return Path(c.args.dec_dir)
    return c.L.work / c.recipe["d3d"]["dec"]["dir"]


def dec_decoder_shas(c):
    """{dec_s<seed>.pt: SHA-256, or None when the file is missing} of the recipe's decoders (dec_weights_dir)."""
    w = dec_weights_dir(c)
    return {f"dec_s{s}.pt": (sha256(w / f"dec_s{s}.pt") if (w / f"dec_s{s}.pt").is_file() else None)
            for s in c.recipe["d3d"]["dec"]["seeds"]}


def dec_checks(q, meta, src_scal_sha, out_scal_sha, n_frames, want_n):
    """Pass/fail of a fec/offdec.py apply output (label-free; family doc amendment 26 sanity list)."""
    lo, hi = q.get("ratio_band", [0.5, 2.0])
    ref = q.get("ref_corr_rms") or {}
    rm = meta["corr_rms_median"] / ref["median"] if ref.get("median") else None
    rp = meta["corr_rms_p99"] / ref["p99"] if ref.get("p99") else None
    return dict(scal_identical=src_scal_sha == out_scal_sha, n_frames=n_frames == want_n,
                no_nan=not meta["nan"], inspan=meta["inspan_rms"] <= float(q.get("inspan_rms_max", 1e-8)),
                ratio_median=rm is None or lo <= rm <= hi, ratio_p99=rp is None or lo <= rp <= hi,
                n_decoders=len(meta["decoders"]) == len(q["seeds"]))


def st_d3d_dec(c):
    """fec/offdec.py apply on the q95-hybrid preds (GPU, ~1-2 min / 193k frames): psi + mean decoder correction on
    every frame, every other file copied (q95 / betaN byte-identical, checked).  The members' basis is read from their
    model.pt (all primary + fallback members must agree to 1e-6) and must be the decoders' training basis.  On a failed
    step or check the stage records fallback=src and lirep4 runs on the source (q95-hybrid) preds."""
    L, q = c.L, c.recipe["d3d"]["dec"]
    src, out = dec_dirs(c)
    c.check_writable(out)
    log = c.stage_log("d3d_dec")
    wdir = dec_weights_dir(c)
    decs = [wdir / f"dec_s{s}.pt" for s in q["seeds"]]
    rp = resolve_path(c.recipe["d3d"]["b"]["recipe"], REPO)
    store = L.d3d_pred_work / "models"
    dev = run_device(q.get("device") or c.recipe["d3d"]["b"].get("device", "cuda"))
    cmd = [PY, "-B", FEC / "offdec.py", "apply", "--src", src, "--out", out, "--config", L.d3d_cfg,
           "--decoders", ",".join(str(p) for p in decs), "--members-recipe", rp, "--models", store,
           "--device", dev, "--chunk", int(q.get("chunk", 8192))]
    if q.get("shipped_basis_sha256"):
        cmd += ["--expect-basis", q["shipped_basis_sha256"]]
    if c.args.dry_run:
        c.sh(cmd, log, gpu=dev.startswith("cuda"))
        return {}
    for f in ("psi", "scal"):
        if not (src / f"{L.d3d_cfg}_{f}.npy").exists():
            raise StageError(f"no source predictions {src}/{L.d3d_cfg}_{f}.npy")
    info = dict(src=str(src), out=str(out), decoder_dir=str(wdir), store=str(store), device=dev, frozen=dict(q),
                src_psi_sha256=sha256(src / f"{L.d3d_cfg}_psi.npy"),
                src_scal_sha256=sha256(src / f"{L.d3d_cfg}_scal.npy"),
                offdec_py_sha256=sha256(FEC / "offdec.py"), decoders=dec_decoder_shas(c))
    try:
        miss = [str(wdir / n) for n, h in info["decoders"].items() if h is None]
        if miss:
            raise StageError(f"missing decoder files {miss} (train.py --stage offdec, or the weights repository)")
        if not same_dir(L.d3d_pred_work, L.work):
            info["links"] = c.d3d_root_links(L.d3d_pred_work, need=("models",))
        if out.exists():          # apply never overwrites: clear this run's own output
            if not is_under(out, L.root):
                raise StageError(f"refusing to clear {out} outside {L.root}")
            for f in out.iterdir():
                f.unlink()
        c.sh(cmd, log, gpu=dev.startswith("cuda"))
        meta = json.loads((out / f"{L.d3d_cfg}_dec_meta.json").read_text())
        chk = dec_checks(q, meta, info["src_scal_sha256"], sha256(out / f"{L.d3d_cfg}_scal.npy"),
                         np.load(out / f"{L.d3d_cfg}_psi.npy", mmap_mode="r").shape[0], d3d_pred_N(c))
        if not all(chk.values()):
            raise StageError(f"d3d_dec output checks failed: {chk}")
        info.update(fallback=None, checks=chk, psi_sha256=sha256(out / f"{L.d3d_cfg}_psi.npy"),
                    **{k: meta[k] for k in ("frames", "inspan_max_abs", "inspan_rms", "corr_rms_median",
                                            "corr_rms_p99", "corr_rms_max", "basis_sha256", "seconds")})
    except Exception as e:  # noqa: BLE001 - keep the source if the decoder step fails
        if q.get("fallback") != "src":
            raise
        c.log(f"WARNING d3d_dec FAILED ({type(e).__name__}: {e}); lirep4 and the submission fall back to {src}")
        info.update(fallback="src", error=f"{type(e).__name__}: {e}")
    return info


def done_d3d_dec(c):
    """Done only for the same source, the same decoder files and the same fec/offdec.py (review rev_decpkg D2): new
    decoders, a fixed fallback cause or changed code re-run the stage without --force."""
    src, out = dec_dirs(c)
    i = c.manifest("d3d_dec").get("info", {})
    same = (i.get("src") == str(src) and i.get("decoders") == dec_decoder_shas(c) and
            i.get("offdec_py_sha256") == sha256(FEC / "offdec.py"))
    if i.get("fallback") == "src":
        return same
    return (same and _pred_done(out, c.L.d3d_cfg, d3d_pred_N(c)) and
            i.get("src_psi_sha256") == sha256(src / f"{c.L.d3d_cfg}_psi.npy"))


def _d3d_pre_dec_dir(c):
    """The pre-decoder (and, without d3d.dec, pre-lirep4) submission source (the legacy d3d_submit_dir rule)."""
    if not c.recipe["d3d"].get("q95h"):
        return c.L.preds / d3d_final_name(c.recipe), None
    src, out = q95h_dirs(c)
    m = c.manifest("d3d_q95h")
    if c.args.dry_run and not m:
        return out, None
    if m.get("status") != "ok":
        raise StageError("recipe has d3d.q95h: run the d3d_q95h stage first")
    if m["info"].get("fallback") == "head":
        c.log(f"WARNING: d3d_q95h fell back to the head ({m['info'].get('error')}); submitting "
              f"{src}")
        return src, "head"
    return out, None


def _passive_args(tag):
    model, r = tag.split("_r")
    if model not in ("c1", "c5"):
        raise StageError(f"unsupported passive variant {tag}")
    return model, r


def st_passive(c):
    L = c.L
    tag = c.recipe["mast"]["stage1"]["passive"]
    if tag == "none":
        return dict(skipped="recipe passive none")
    model, rs = _passive_args(tag)
    c.check_writable(L.passive_dir(L.mast_cfg, tag), L.data_work / "passive" / "sig" / L.mast_cfg)
    log = c.stage_log("passive")
    c.sh([PY, "-B", FEC / "passive.py", "signals", "--config", L.mast_cfg,
          "--workers", c.args.workers], log, env=c.data_env())
    c.sh([PY, "-B", FEC / "passive.py", "frames", "--model", model, "--rscale", rs,
          "--config", L.mast_cfg], log, env=c.data_env())
    if c.args.dry_run:
        return {}
    d = L.passive_dir(L.mast_cfg, tag)
    meta = json.loads((d / "meta.json").read_text())
    n = np.load(d / "I.npy", mmap_mode="r").shape[0]
    if n != json.loads((L.ds_dir(L.mast_cfg) / "info.json").read_text())["N"]:
        raise StageError(f"passive frames {n} != ds")
    return dict(dir=str(d), N=int(n), meta=meta)


def done_passive(c):
    tag = c.recipe["mast"]["stage1"]["passive"]
    if tag == "none":
        return True
    d = c.L.passive_dir(c.L.mast_cfg, tag)
    return (d / "meta.json").exists() and (d / "I.npy").exists()


def mast_arm_options(recipe, arm):
    if arm == "v2eq":
        return dict(zmode="v2", kfit_warm="o0", passive="none", limiter="off")
    s1 = recipe["mast"]["stage1"]
    return dict(zmode=s1["zmode"], kfit_warm=s1["kfit_warm"], passive=s1["passive"],
                limiter={"loff": "off", "lon": "on"}[arm])


def mast_limit(c):
    if not c.L.subset_shots:
        return 0
    from fec.features import load_ds
    return subset_limit(load_ds(c.L.mast_cfg)["shot"], c.L.subset_shots)


def _st_mast_arm(c, arm):
    L = c.L
    if arm == "v2eq" and not L.shadow:
        raise StageError("mast_v2eq is a public_test rehearsal check")
    out = L.mast_dir(arm)
    c.check_writable(out)
    o = mast_arm_options(c.recipe, arm)
    pas = L.passive_dir(L.mast_cfg, o["passive"], canonical=True) / "I.npy"
    if o["passive"] != "none" and not c.args.dry_run and not pas.exists():
        raise StageError(f"missing passive currents for {L.mast_cfg}: run the passive stage")
    limit = mast_limit(c)
    adopted = None
    if c.args.adopt:
        adopted = adopt_stage1(c, Path(c.args.adopt), out, o, limit)
    else:
        cmd = [PY, "-B", FEC / "mast_select_v3.py", "--out", out, "--config", L.mast_cfg,
               "--zmode", o["zmode"], "--kfit_warm", o["kfit_warm"], "--passive", o["passive"],
               "--limiter", o["limiter"]] + (["--limit", limit] if limit else [])
        c.sh(cmd, c.stage_log(f"mast_{arm}"), gpu=True)
    if c.args.dry_run:
        return dict(limit=limit)
    n = np.load(out / "psi.npy", mmap_mode="r").shape[0]
    with np.load(out / "meta.npz") as z:
        lim = float(np.mean(z["limited"])) if "limited" in z.files else None
        zf = float(np.mean(z["zflag"] > 0)) if "zflag" in z.files else None
    return dict(out=str(out), N=int(n), limit=limit, options=o, limited_frac=lim, zflag_frac=zf,
                adopted=adopted)


def adopt_stage1(c, src, out, opts, limit):
    """Copy a finished stage-1 run (psi.npy, meta.npz, config.json) made with exactly these
    options and frame count into `out` instead of recomputing it (e.g. root's production
    fec_work/mast_v3_loff).  Copies, never links: a later re-run of the stage must not be able
    to truncate the source."""
    import shutil
    L = c.L
    cfg = _read_cfg(src)
    want_n = limit or _n_frames(L.mast_cfg)
    want = dict(config=L.mast_cfg, N=want_n, **opts)
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise StageError(f"--adopt {src}: config.json differs from this stage: {bad}")
    if not (src / "meta.npz").exists() or (src / "meta_partial.npz").exists():
        raise StageError(f"--adopt {src}: stage 1 not finished (meta.npz missing or partial)")
    if np.load(src / "psi.npy", mmap_mode="r").shape[0] != want_n:
        raise StageError(f"--adopt {src}: psi.npy frame count != {want_n}")
    c.cmds.append(f"adopt (copy) {src} -> {out}")
    c.log(c.cmds[-1])
    if c.args.dry_run:
        return None
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for name in ("config.json", "meta.npz", "psi.npy"):
        tmp = out / (name + ".part")
        shutil.copyfile(src / name, tmp)
        os.replace(tmp, out / name)
        info[name] = dict(source=str(src / name), sha256=sha256(out / name),
                          source_sha256=sha256(src / name))
        if info[name]["sha256"] != info[name]["source_sha256"]:
            raise StageError(f"copy of {src / name} differs from its source")
    return info


def _done_mast_arm(c, arm):
    d = c.L.mast_dir(arm)
    return (d / "meta.npz").exists() and not (d / "meta_partial.npz").exists()


def _mast_input(c, which):
    """Stage-1 dir of an arm ('loff', 'lon', 'merged') or, for which='final', the MAST dir that
    stage 2 / betaN / submission use: --mast-dir, else the recipe's final dir (loff or merged)."""
    a = c.args
    if which == "final" and not a.mast_dir and c.recipe["mast"].get("composite"):
        return composite_dirs(c)["final"]
    if which == "final" and not a.mast_dir and mast_pipeline_file(c.recipe):
        return pipeline_final_dir(c)
    if which == "final":
        return Path(a.mast_dir) if a.mast_dir else c.L.mast_dir(c.recipe["mast"].get("final", "loff"))
    given = {"loff": a.loff_dir, "lon": a.lon_dir, "merged": None}[which]
    return Path(given) if given else c.L.mast_dir(which)


def boundary_limiter(c):
    """Stage-2 / betaN boundary-only limiter: --blim on|off overrides the recipe."""
    if c.args.blim != "recipe":
        return c.args.blim == "on"
    return bool(c.recipe["mast"].get("boundary_limiter", False))


def boundary_limiter_stats(mast_dir, grid=None, chunk=2048):
    """Frames where stage 2 --boundary-limiter replaces the stage-1 psi_b (column_flux rule,
    vectorised; stage-1 meta psi_ax/psi_b), incl. frames whose stage-1 psi_b was the no-X-point
    sentinel."""
    d = Path(mast_dir)
    if grid is None:
        from fec.geometry import machine_geometry
        g = machine_geometry("mast_public_test")
        gR, gZ = g["grid_R"], g["grid_Z"]
    else:
        gR, gZ = (np.asarray(x, float) for x in grid)
    psi = np.load(d / "psi.npy", mmap_mode="r")
    with np.load(d / "meta.npz") as z:
        pa, pb = z["psi_ax"].astype(float), z["psi_b"].astype(float)
    j = int(np.searchsorted(gR, LIMITER_R)) - 1
    w = (LIMITER_R - gR[j]) / (gR[j + 1] - gR[j])
    rows = np.flatnonzero(np.abs(gZ) <= LIMITER_ZMAX + 1e-9)
    N = psi.shape[0]
    plim = np.full(N, -np.inf)
    for s0 in range(0, N, chunk):
        sl = slice(s0, min(N, s0 + chunk))
        x = np.asarray(psi[sl][:, rows], dtype=np.float64)
        pl = (1 - w) * x[:, :, j] + w * x[:, :, j + 1]
        pl = np.where(pl < pa[sl, None], pl, -np.inf)
        plim[sl] = pl.max(1)
    ok = np.isfinite(pa) & np.isfinite(pb)
    active = ok & (plim > pb)
    return dict(N=int(N), active=int(active.sum()), frac_active=float(active.mean()) if N else 0.0,
                sentinel_frames=int(np.sum(ok & (pb <= -1e20))),
                sentinel_rescued=int(np.sum(active & (pb <= -1e20))))


def st_mast_merge(c):
    out = c.L.mast_dir("merged")
    c.check_writable(out)
    loff, lon = _mast_input(c, "loff"), _mast_input(c, "lon")
    thr = float(c.recipe["mast"]["merge"]["r_in_threshold"])
    c.cmds.append(f"in-process merge_deep_crossing({loff}, {lon}, {out}, r_in_threshold={thr})")
    c.log(c.cmds[-1])
    if c.args.dry_run:
        return {}
    for d in (loff, lon):
        if not (d / "meta.npz").exists():
            raise StageError(f"{d} has no finished stage-1 output (meta.npz)")
    s = merge_deep_crossing(loff, lon, out, r_in_threshold=thr)
    c.log(f"merge: {s['n_from_lon']}/{s['N']} frames from limiter-on "
          f"({100 * s['frac_from_lon']:.2f}%), no loff boundary {s['n_no_boundary_loff']}")
    return {k: v for k, v in s.items() if k not in ("loff_config", "lon_config")}


def done_mast_merge(c):
    return (c.L.mast_dir("merged") / "merge.json").exists()


def _same_target(c, name, d, blim):
    """A finished stage counts as done only for the same MAST dir and limiter setting."""
    info = c.manifest(name).get("info", {})
    return info.get("mast_dir") == str(d) and info.get("boundary_limiter") == blim


def mast_v1_nproc(workers):
    """fec/mast_v1.py stage 2 pool size from --workers (organisers' Final guide: the worker count is a parameter).  The
    stage's historical pool was a fixed 6, which is also the --workers default; at 6 nothing is appended, so default
    commands are unchanged (agent repin, v7)."""
    return [] if int(workers) == 6 else ["--nproc", int(workers)]


def st_mast_stage2(c):
    d = _mast_input(c, "final")
    c.check_writable(d)
    blim = boundary_limiter(c)
    if not c.args.dry_run and not (d / "meta.npz").exists():
        raise StageError(f"{d} has no finished stage-1 output (meta.npz)")
    c.sh([PY, "-B", FEC / "mast_v1.py", "2", "--config", c.L.mast_cfg, "--out", d] +
         (["--boundary-limiter"] if blim else []) + mast_v1_nproc(c.args.workers), c.stage_log("mast_stage2"))
    if c.args.dry_run:
        return {}
    info = dict(mast_dir=str(d), boundary_limiter=blim, scalars_sha256=sha256(d / "scalars.npz"))
    with np.load(d / "scalars.npz") as z:
        info["scalars"] = {k: dict(finite=float(np.isfinite(z[k]).mean()),
                                   median=float(np.nanmedian(z[k]))) for k in z.files}
        if "q95" in z.files:
            info["q95_gt20_frac"] = float(np.mean(z["q95"] > 20))
    if blim:
        info["limiter"] = boundary_limiter_stats(d)
    return info


def done_mast_stage2(c):
    d = _mast_input(c, "final")
    return (d / "scalars.npz").exists() and _same_target(c, "mast_stage2", d, boundary_limiter(c))


def st_mast_betan(c):
    from fec.features import load_ds
    d = _mast_input(c, "final")
    c.check_writable(d)
    name = c.recipe["mast"]["betan"]
    blim = boundary_limiter(c)
    c.cmds.append(f"in-process betan_records({c.L.mast_cfg}, {d / 'psi.npy'}, "
                  f"boundary_limiter={blim}) + frozen lam_fgw")
    c.log(c.cmds[-1])
    if c.args.dry_run:
        return {}
    frozen = betan_frozen_files()
    for k, v in frozen.items():
        if not v["reviewed"]:
            msg = f"{k} sha256 {v['sha256']} != reviewed {BETAN_REVIEWED[k]}"
            if not c.args.allow_unreviewed:
                raise StageError(msg + " (--allow-unreviewed to proceed)")
            c.log("WARNING " + msg)
    with open(c.stage_log("mast_betan"), "a", encoding="utf-8") as fh:
        def lg(m):
            fh.write(m + "\n")
            fh.flush()
        rec = betan_records(c.L.mast_cfg, d / "psi.npy", nproc=int(c.args.workers), log=lg,
                            boundary_limiter=blim)
    np.savez(d / "betan_records.npz", **rec)
    with np.load(d / "betan_records.npz") as z:      # reload as cmd_mast does (load_records)
        rec = {k: z[k] for k in z.files}
    n = np.load(d / "psi.npy", mmap_mode="r").shape[0]
    shot_all = np.asarray(load_ds(c.L.mast_cfg)["shot"])[:n]
    out, info = betan_apply(rec, shot_all)
    np.save(d / name, out.astype(np.float64))
    info.update(mast_dir=str(d), boundary_limiter=blim, file=str(d / name),
                sha256=sha256(d / name), frozen=frozen, geo_ok=float(np.mean(rec["ok_geo"])),
                records_sha256=sha256(d / "betan_records.npz"))
    write_json(d / (Path(name).stem + ".json"), info)
    return info


def done_mast_betan(c):
    d = _mast_input(c, "final")
    return ((d / c.recipe["mast"]["betan"]).exists() and
            _same_target(c, "mast_betan", d, boundary_limiter(c)))


# ---------------------------------------------------------------------------- MAST pipeline recipes
def mast_pipeline_file(recipe):
    """fec/recipes_mast/*.json named by recipe mast.pipeline (repo-relative), else None."""
    p = recipe["mast"].get("pipeline")
    if not p:
        return None
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def pipeline_dir(c):
    return c.L.mast / f"pipeline{c.L.sfx}"


def pipeline_final_dir(c):
    from fec import mast_pipeline as MP
    return pipeline_dir(c) / MP.load_recipe(mast_pipeline_file(c.recipe))["final"]


def st_mast_pipeline(c):
    """fec/mast_pipeline.py recipe: members (GPU steps wrapped in gpu_slot by the pipeline
    itself), merge/average, stage 2 + frozen betaN of the final node (idempotent)."""
    from fec import mast_pipeline as MP
    L = c.L
    f = mast_pipeline_file(c.recipe)
    pr = MP.load_recipe(f)
    out = pipeline_dir(c)
    c.check_writable(out)
    blim = boundary_limiter(c)
    if pr["boundary_limiter"] != blim:
        raise StageError(f"pipeline {f.name} boundary_limiter {pr['boundary_limiter']} != {blim}")
    if pr["betan"] != c.recipe["mast"]["betan"]:
        raise StageError(f"pipeline betan {pr['betan']} != recipe {c.recipe['mast']['betan']}")
    if not c.args.dry_run:
        miss = [t for t in MP.recipe_passive_tags(pr)
                if not (L.passive_dir(L.mast_cfg, t, canonical=True) / "I.npy").exists()]
        if miss:
            raise StageError(f"missing passive currents {miss} for {L.mast_cfg}: run the passive "
                             f"stage (recipe mast.stage1.passive) / fec/passive.py frames")
    limit = mast_limit(c)
    c.sh([PY, "-B", FEC / "mast_pipeline.py", "recipe", "--recipe", f, "--config", L.mast_cfg,
          "--out", out, "--owner", c.args.owner, "--gpu-priority", c.args.gpu_priority,
          "--workers", c.args.workers] + (["--limit", limit] if limit else []),
         c.stage_log("mast_pipeline"))
    if c.args.dry_run:
        return dict(pipeline=str(f), limit=limit)
    d = pipeline_final_dir(c)
    man = json.loads((out / "pipeline.json").read_text())
    if man.get("status") != "ok" or Path(man.get("final_dir", "")) != d:
        raise StageError(f"{out / 'pipeline.json'}: status {man.get('status')}, final "
                         f"{man.get('final_dir')} (want {d})")
    return dict(pipeline=str(f), pipeline_sha256=sha256(f), pipeline_name=pr["name"],
                mast_dir=str(d), boundary_limiter=blim, N=man.get("N"), limit=limit,
                scalars_sha256=sha256(d / "scalars.npz"), betan_sha256=sha256(d / pr["betan"]),
                steps=[(s.get("node"), s.get("step"), s.get("state"), s.get("seconds"))
                       for s in man.get("steps", [])])


def done_mast_pipeline(c):
    d = pipeline_final_dir(c)
    info = c.manifest("mast_pipeline").get("info", {})
    return ((d / "scalars.npz").exists() and (d / c.recipe["mast"]["betan"]).exists() and
            _same_target(c, "mast_pipeline", d, boundary_limiter(c)) and
            info.get("pipeline_sha256") == sha256(mast_pipeline_file(c.recipe)))


# ---------------------------------------------------------------------------- MAST composite
def composite_block(c):
    b = c.recipe["mast"].get("composite")
    if not b:
        raise StageError("recipe has no mast.composite")
    return b


def composite_dirs(c):
    """Node dirs of the composite under <root>/mast/composite<sfx>/."""
    m = composite_block(c)["maps"]
    base = c.L.mast / f"composite{c.L.sfx}"
    D = dict(base=base, physics=base / "physics", final=base / "final")
    if m.get("kind") == "physics":          # reserve: maps of the physics average itself
        D["maps"] = D["physics"] / "avg_f05"
        return D
    f = float(m["sol_frac"])
    D.update(passive=base / f"passive_f{f:g}", amort=base / f"amort_{m['net']}",
             tsm=base / f"amort_{m['net']}_tsm{float(m['tsm_sigma_ms']):g}")
    D["maps"] = D["tsm"]
    kind, rs, sig = m.get("kind"), maps_rscale(m), f"{float(m['tsm_sigma_ms']):g}"
    if kind == "amort" and rs != "1":       # r05: the knob node replaces the production one
        D.update(amort=base / f"amort_{m['net']}_r{rs}", tsm=base / f"amort_{m['net']}_r{rs}_tsm{sig}")
        D["maps"] = D["tsm"]
    elif kind in ("vb", "combo"):           # D["amort"] / D["tsm"]: the production (r = 1) nodes
        t = m["tx1s"]
        D.update(tx1s=base / f"tx1s_{t}", vb=base / f"vb_{m['net']}_{t}",
                 vb_tsm=base / f"vb_{m['net']}_{t}_tsm{sig}")
        D["maps"] = D["vb_tsm"]
        if kind == "combo":
            D.update(amort_r=base / f"amort_{m['net']}_r{rs}", tsm_r=base / f"amort_{m['net']}_r{rs}_tsm{sig}",
                     combo=base / f"combo_{m['net']}_{t}_r{rs}")
            D["maps"] = D["combo"]
    return D


def maps_rscale(m):
    """C1 passive resistance-scale tag of a composite maps block ('1' = production; fec.final_mast.rscale_tag)."""
    return f"{float(m.get('rscale', 1)):g}"


def composite_tx1s_dir(c):
    """TX1s model dir (vb / combo maps): --tx1s-model-dir, else <FEC_WORK>/xfer/models1/<tx1s tag>."""
    m = composite_block(c)["maps"]
    return Path(c.args.tx1s_model_dir) if getattr(c.args, "tx1s_model_dir", None) else \
        c.L.work / "xfer" / "models1" / m["tx1s"]


def composite_net_dir(c):
    """Amortized net dir: --amort-net-dir, else <FEC_WORK>/amort/models/<net>."""
    m = composite_block(c)["maps"]
    return Path(c.args.amort_net_dir) if getattr(c.args, "amort_net_dir", None) else \
        c.L.work / "amort" / "models" / m["net"]


def composite_scalars_dir(c):
    """Physics-average node whose scalars/betaN records the composite takes: --mast-scalars-dir
    (rehearsal only: a stored node), else <composite>/physics/<plan final node>."""
    if getattr(c.args, "mast_scalars_dir", None):
        return Path(c.args.mast_scalars_dir)
    plan = composite_block(c)["scalars"]["plan"]
    if plan != "avg_f05":
        raise StageError(f"unknown physics plan {plan!r}")
    return composite_dirs(c)["physics"] / "avg_f05"


def st_mast_amort(c):
    """Amortized maps: C1 passive currents at f (CPU; f = 0.5 uses the canonical passive
    frames), the net on every frame of the fold (GPU slot), within-shot smoothing + meta (GPU)."""
    L, m = c.L, composite_block(c)["maps"]
    if m.get("kind") == "physics":
        return dict(skipped="composite maps kind physics: the physics average is the map source")
    if m.get("kind") in ("vb", "combo"):
        return _st_mast_amort_tx1s(c)
    if m.get("kind") != "amort":
        raise StageError(f"composite maps kind {m.get('kind')!r} is not 'amort'")
    D = composite_dirs(c)
    c.check_writable(D["base"])
    log = c.stage_log("mast_amort")
    f = float(m["sol_frac"])
    net = composite_net_dir(c)
    fm = [PY, "-B", FEC / "final_mast.py"]
    limit = mast_limit(c)
    rs = maps_rscale(m)
    if abs(f - 0.5) < 1e-12:
        if rs != "1":
            raise StageError("rscale != 1 needs the passive_f node (sol_frac != 0.5)")
        pdir = L.passive_dir(L.mast_cfg, "c1_r1", canonical=True).parent
    else:
        pdir = D["passive"]
        c.sh(fm + ["passive_f", "--config", L.mast_cfg, "--f", repr(f), "--out", pdir,
                   "--rscales", rs], log)
    dev = run_device("cuda")
    c.sh(fm + ["amort", "--config", L.mast_cfg, "--net", net, "--f", repr(f), "--passive", pdir,
               "--out", D["amort"], "--device", dev] + (["--n", limit] if limit else []) +
         (["--rscale", rs] if rs != "1" else []),
         log, gpu=dev == "cuda")
    c.sh(fm + ["tsm", "--config", L.mast_cfg, "--src", D["amort"], "--out", D["tsm"],
               "--sigma-ms", repr(float(m["tsm_sigma_ms"])), "--device", dev], log,
         gpu=dev == "cuda")
    if c.args.dry_run:
        return {}
    ca = json.loads((D["amort"] / "config.json").read_text())
    ct = json.loads((D["tsm"] / "config.json").read_text())
    if ca.get("sol_frac") != f or ca.get("config") != L.mast_cfg:
        raise StageError(f"amort node made for {ca.get('config')} f {ca.get('sol_frac')}")
    if ca.get("rscale", "1") != rs:
        raise StageError(f"amort node made with rscale {ca.get('rscale', '1')}, recipe {rs}")
    return dict(net=str(net), sol_frac=f, passive=str(pdir), limit=limit or None,
                amort=dict(dir=str(D["amort"]), psi_sha256=ca["psi_sha256"],
                           net_sha256=ca["net_sha256"], basis_sha256=ca["basis_sha256"],
                           passive_sha256=ca["passive_sha256"], seconds=ca["seconds"]),
                tsm=dict(dir=str(D["tsm"]), psi_sha256=ct["psi_sha256"], seconds=ct["seconds"]),
                **({"rscale": rs} if rs != "1" else {}))


def _st_mast_amort_tx1s(c):
    """vb / combo maps (Oct 14 T2 / amendment 11): passive at f, the production amortized maps (r = 1), TX1s maps
    (fec/final_tx1s.py infer), VB = 0.5 (amort + TX1s) then tsm; combo also smooths the production maps (S4), runs
    the r = rscale knob maps + tsm and forms VB + (r - S4) with meta (fec/final_mast.py combo)."""
    L, m = c.L, composite_block(c)["maps"]
    kind = m["kind"]
    D = composite_dirs(c)
    c.check_writable(D["base"])
    log = c.stage_log("mast_amort")
    f = float(m["sol_frac"])
    if abs(f - 0.5) < 1e-12:
        raise StageError(f"maps kind {kind} needs the passive_f node (sol_frac != 0.5)")
    net, tx = composite_net_dir(c), composite_tx1s_dir(c)
    fm = [PY, "-B", FEC / "final_mast.py"]
    limit = mast_limit(c)
    lim = ["--n", limit] if limit else []
    rs = maps_rscale(m)
    if (kind == "combo") == (rs == "1"):
        raise StageError(f"maps kind {kind} with rscale {rs}: combo needs rscale != 1, vb rscale 1")
    sig = repr(float(m["tsm_sigma_ms"]))
    dev = run_device("cuda")
    gpu = dev == "cuda"
    pdir = D["passive"]
    c.sh(fm + ["passive_f", "--config", L.mast_cfg, "--f", repr(f), "--out", pdir,
               "--rscales", "1" if kind == "vb" else f"{rs},1"], log)
    c.sh(fm + ["amort", "--config", L.mast_cfg, "--net", net, "--f", repr(f), "--passive", pdir,
               "--out", D["amort"], "--device", dev] + lim, log, gpu=gpu)
    c.sh([PY, "-B", FEC / "final_tx1s.py", "infer", "--config", L.mast_cfg, "--model", tx, "--passive", pdir,
          "--out", D["tx1s"], "--device", dev] + lim, log, gpu=gpu)
    c.sh(fm + ["blend", "--a", D["amort"], "--b", D["tx1s"], "--out", D["vb"]], log)
    c.sh(fm + ["tsm", "--config", L.mast_cfg, "--src", D["vb"], "--out", D["vb_tsm"], "--sigma-ms", sig,
               "--device", dev], log, gpu=gpu)
    if kind == "combo":
        c.sh(fm + ["tsm", "--config", L.mast_cfg, "--src", D["amort"], "--out", D["tsm"], "--sigma-ms", sig,
                   "--device", dev], log, gpu=gpu)
        c.sh(fm + ["amort", "--config", L.mast_cfg, "--net", net, "--f", repr(f), "--passive", pdir,
                   "--out", D["amort_r"], "--device", dev] + lim + ["--rscale", rs], log, gpu=gpu)
        c.sh(fm + ["tsm", "--config", L.mast_cfg, "--src", D["amort_r"], "--out", D["tsm_r"], "--sigma-ms", sig,
                   "--device", dev], log, gpu=gpu)
        c.sh(fm + ["combo", "--config", L.mast_cfg, "--vb", D["vb_tsm"], "--r05", D["tsm_r"], "--s4", D["tsm"],
                   "--out", D["combo"], "--device", dev], log, gpu=gpu)
    if c.args.dry_run:
        return {}
    ca = json.loads((D["amort"] / "config.json").read_text())
    cx = json.loads((D["tx1s"] / "config.json").read_text())
    if ca.get("sol_frac") != f or ca.get("config") != L.mast_cfg or cx.get("config") != L.mast_cfg:
        raise StageError(f"amort / TX1s nodes made for {ca.get('config')} / {cx.get('config')}, f "
                         f"{ca.get('sol_frac')}")
    nodes = {k: json.loads((D[k] / "config.json").read_text())["psi_sha256"]
             for k in ("amort", "tx1s", "vb", "vb_tsm") + (("tsm", "amort_r", "tsm_r", "combo")
                                                            if kind == "combo" else ())}
    return dict(kind=kind, net=str(net), tx1s_model=str(tx), tx1s_model_sha256=cx["model_sha256"],
                sol_frac=f, rscale=rs, passive=str(pdir), limit=limit or None, maps_dir=str(D["maps"]),
                nodes_psi_sha256=nodes, tx1s_seconds=cx.get("seconds"))


def done_mast_amort(c):
    if composite_block(c)["maps"].get("kind") == "physics":
        return True
    D = composite_dirs(c)
    i = c.manifest("mast_amort").get("info", {})
    if composite_block(c)["maps"].get("kind") in ("vb", "combo"):
        return (D["maps"] / "psi.npy").exists() and (D["maps"] / "meta.npz").exists() and \
            i.get("maps_dir") == str(D["maps"]) and i.get("net") == str(composite_net_dir(c)) and \
            i.get("tx1s_model") == str(composite_tx1s_dir(c))
    f = D["tsm"] / "psi.npy"
    return f.exists() and (D["tsm"] / "meta.npz").exists() and \
        i.get("tsm", {}).get("dir") == str(D["tsm"]) and \
        i.get("net") == str(composite_net_dir(c))


def st_mast_physics(c):
    """Physics average avg_f05 (fec/final_mast.py physics = fec/mast_pipeline.py Plan: 4 stage-1
    members on the GPU, merge, tsm, average, stage 2 + frozen betaN; idempotent per step)."""
    L = c.L
    D = composite_dirs(c)
    if getattr(c.args, "mast_scalars_dir", None):
        return dict(skipped=f"--mast-scalars-dir {c.args.mast_scalars_dir} (rehearsal)")
    c.check_writable(D["physics"])
    if not c.args.dry_run:
        miss = [t for t in ("c1_r1",)
                if not (L.passive_dir(L.mast_cfg, t, canonical=True) / "I.npy").exists()]
        if miss:
            raise StageError(f"missing passive currents {miss} for {L.mast_cfg}: run the passive "
                             f"stage first")
    limit = mast_limit(c)
    c.sh([PY, "-B", FEC / "final_mast.py", "physics", "--config", L.mast_cfg, "--out",
          D["physics"], "--owner", c.args.owner, "--gpu-priority", c.args.gpu_priority,
          "--workers", c.args.workers, "--jobs", getattr(c.args, "mast_jobs", 1)] +
         (["--limit", limit, "--limit-align", "chunk"] if limit else []),
         c.stage_log("mast_physics"))
    if c.args.dry_run:
        return dict(limit=limit)
    d = D["physics"] / "avg_f05"
    man = json.loads((D["physics"] / "pipeline.json").read_text())
    if man.get("status") != "ok" or Path(man.get("final_dir", "")) != d:
        raise StageError(f"{D['physics'] / 'pipeline.json'}: status {man.get('status')}")
    return dict(dir=str(d), N=man.get("N"), limit=limit or None,
                scalars_sha256=sha256(d / "scalars.npz"),
                records_sha256=sha256(d / "betan_records.npz"),
                steps=[(s.get("node"), s.get("step"), s.get("state"), s.get("seconds"))
                       for s in man.get("steps", [])])


def done_mast_physics(c):
    if getattr(c.args, "mast_scalars_dir", None):
        return True
    d = composite_dirs(c)["physics"] / "avg_f05"
    i = c.manifest("mast_physics").get("info", {})
    return (d / "scalars.npz").exists() and (d / "betan_records.npz").exists() and \
        i.get("dir") == str(d)


def st_mast_composite(c):
    """Submission MAST dir: smoothed amortized maps + physics-average scalars + fill3 betaN."""
    L, b = c.L, composite_block(c)
    D = composite_dirs(c)
    c.check_writable(D["final"])
    sc = composite_scalars_dir(c)
    fill = b.get("betan_fill") or {}
    if fill.get("rule") != "fill3":
        raise StageError(f"betan_fill rule {fill.get('rule')!r} is not fill3")
    c.sh([PY, "-B", FEC / "final_mast.py", "composite", "--config", L.mast_cfg,
          "--maps", D["maps"], "--scalars", sc, "--out", D["final"],
          "--min-finite", int(fill.get("min_finite", 3))], c.stage_log("mast_composite"))
    if c.args.dry_run:
        return {}
    ci = json.loads((D["final"] / "config.json").read_text())
    return dict(mast_dir=str(D["final"]), maps=str(D["maps"]), scalars=str(sc),
                scalars_override=bool(getattr(c.args, "mast_scalars_dir", None)),
                boundary_limiter=boundary_limiter(c), **{k: ci[k] for k in (
                    "psi_sha256", "scalars_sha256", "records_sha256", "betan_sha256",
                    "raw_finite_frac", "betan_median")})


def done_mast_composite(c):
    D = composite_dirs(c)
    i = c.manifest("mast_composite").get("info", {})
    return all((D["final"] / f).exists() for f in ("psi.npy", "scalars.npz", "betan_v2.npy")) \
        and i.get("mast_dir") == str(D["final"]) and \
        i.get("scalars") == str(composite_scalars_dir(c))


def psi_dtype(c):
    """psirz storage dtype: --psi-dtype, else recipe psi_dtype, else float16 (legacy)."""
    d = getattr(c.args, "psi_dtype", None) or c.recipe.get("psi_dtype", "float16")
    if d not in ("float16", "float32"):
        raise StageError(f"psi_dtype must be float16 or float32, not {d!r}")
    return d


def submission_sources(c):
    """ds dirs build_submission / validate must see: the DIII-D ds the predictions were made
    from, the canonical MAST ds (stage 1 reads it) and the canonical diii_d_train (TRAIN label
    clip ranges)."""
    L = c.L
    src = {"diii_d_train": L.work / "ds" / "diii_d_train",
           L.d3d_cfg: L.d3d_pred_work / "ds" / L.d3d_cfg,
           L.mast_cfg: L.work / "ds" / L.mast_cfg}
    if c.recipe.get("train_ranges"):        # fec/final_build.py: frozen ranges, no TRAIN cache
        src.pop("diii_d_train")
    return src


def submission_work(c):
    """FEC_WORK for build_submission / validate: None (= WORK) when every source is canonical,
    else <root>/view whose ds/<cfg> entries are links to the sources (re-pointed every run)."""
    L = c.L
    src = submission_sources(c)
    if all(same_dir(p, L.work / "ds" / k) for k, p in src.items()):
        return None, {}
    links = {k: dict(target=str(p), status=c.link(L.view / "ds" / k, p, replace=True))
             for k, p in src.items()}
    return L.view, links


def st_submission(c):
    L = c.L
    out = L.root / c.args.submission_name
    c.check_writable(out)
    d3d, d3d_fallback = d3d_submit_dir(c)
    mast = _mast_input(c, "final")
    pdt = psi_dtype(c)
    if not c.args.dry_run and not c.args.mast_dir:
        blim = boundary_limiter(c)          # scalars / betaN must come from this dir + setting
        for st in (("mast_composite",) if c.recipe["mast"].get("composite") else
                   ("mast_pipeline",) if mast_pipeline_file(c.recipe) else
                   ("mast_stage2", "mast_betan")):
            if not _same_target(c, st, mast, blim):
                raise StageError(f"{st} outputs in {mast} were not made by this recipe "
                                 f"(boundary_limiter={blim}); re-run {st}")
    src = submission_sources(c)
    if not c.args.dry_run:
        for cfg, f in ((L.d3d_cfg, d3d / f"{L.d3d_cfg}_psi.npy"), (L.mast_cfg, mast / "psi.npy")):
            n = np.load(f, mmap_mode="r").shape[0]
            want = json.loads((src[cfg] / "info.json").read_text())["N"]
            if n != want:
                raise StageError(f"{f}: {n} frames, {cfg} has {want} (subset outputs cannot be "
                                 f"submitted)")
    vw, links = submission_work(c)
    env = {"FEC_WORK": str(vw)} if vw else None
    wk = ["--work", vw] if vw else []
    log = c.stage_log("submission")
    if c.recipe.get("train_ranges"):
        rp = resolve_path(c.recipe["train_ranges"], REPO)
        c.sh([PY, "-B", FEC / "final_build.py", "build", "--name", out, "--d3d", d3d, "--mast",
              mast, "--split", L.split, "--mast-betan", c.recipe["mast"]["betan"],
              "--psi-dtype", pdt, "--ranges", rp], log, env=env)
    else:
        c.sh([PY, "-B", FEC / "build_submission.py", "--name", out, "--d3d", d3d, "--mast", mast,
              "--split", L.split, "--mast-betan", c.recipe["mast"]["betan"], "--psi-dtype", pdt],
             log, env=env)
    info = dict(dir=str(out), d3d=str(d3d), d3d_fallback=d3d_fallback, mast=str(mast),
                psi_dtype=pdt, files={},
                work=str(vw) if vw else str(L.work), view_links=links)
    for cfg in L.configs:
        f = out / f"{cfg}.npz"
        c.sh([PY, "-B", FEC / "submit.py", "validate", f, "--config", cfg, "--json",
              "--psi-dtype", pdt] + wk, log)
        if not c.args.dry_run:
            info["files"][cfg] = dict(path=str(f), bytes=f.stat().st_size, sha256=sha256(f))
    if not c.args.dry_run:
        info["build"] = json.loads((out / "build.json").read_text())
        if info["build"].get("psi_dtype", "float16") != pdt:
            raise StageError(f"build.json psi_dtype {info['build'].get('psi_dtype')} != {pdt}")
    return info


def done_submission(c):
    out = c.L.root / c.args.submission_name
    return all((out / f"{cfg}.npz").exists() for cfg in c.L.configs) and \
        c.manifest("submission").get("info", {}).get("psi_dtype", "float16") == psi_dtype(c)


# ---------------------------------------------------------------------------- upload / pointer
READ_TOKEN_ENV = "FEC_HF_READ_TOKEN"     # == fec.submit.READ_TOKEN_ENV


def hf_repo(c):
    r = c.args.hf_repo or (c.recipe.get("upload") or {}).get("repo")
    if not r:
        raise StageError("no HF repo: give --hf-repo or recipe upload.repo")
    return r


def _validated_files(c):
    """The submission stage's two validated npz files, re-hashed (bytes unchanged since)."""
    m = c.manifest("submission")
    files = (m.get("info") or {}).get("files") or {}
    if c.args.dry_run and not files:
        sub = c.L.root / c.args.submission_name
        return {cfg: sub / f"{cfg}.npz" for cfg in c.L.configs}, {}
    if m.get("status") != "ok" or set(files) != set(c.L.configs):
        raise StageError(f"run the submission stage first ({c.manifest_file('submission')})")
    out = {}
    for cfg in c.L.configs:
        f = Path(files[cfg]["path"])
        if c.args.dry_run:
            out[cfg] = f
            continue
        h = sha256(f)
        if h != files[cfg]["sha256"]:
            raise StageError(f"{f} changed since validation (sha256 {h} != {files[cfg]['sha256']})")
        out[cfg] = f
    return out, m


def st_upload(c):
    """fec/submit.py upload of the validated files to the PRIVATE HF dataset repo (one commit).
    Dry run (checks + plan, nothing written remotely) unless --confirm-upload.  Uses the local
    HF login (write) token of huggingface_hub, never FEC_HF_READ_TOKEN."""
    files, m = _validated_files(c)
    repo = hf_repo(c)
    log = c.stage_log("upload")
    vw = (m.get("info") or {}).get("work")
    cmd = [PY, "-B", FEC / "submit.py", "upload", "--files", *[files[k] for k in c.L.configs],
           "--repo", repo, "--psi-dtype", psi_dtype(c)] + (["--work", vw] if vw else [])
    cmd += ["--confirm"] if c.args.confirm_upload else ["--dry-run"]
    c.sh(cmd, log)
    if c.args.dry_run:
        return {}
    info = dict(repo=repo, confirmed=bool(c.args.confirm_upload),
                files={k: dict(path=str(f), sha256=m["info"]["files"][k]["sha256"])
                       for k, f in files.items()})
    if c.args.confirm_upload:
        text = Path(log).read_text(encoding="utf-8", errors="replace")
        commits = [ln.split()[1] for ln in text.splitlines() if ln.startswith("COMMIT ")]
        if not commits or len(commits[-1]) != 40:
            raise StageError(f"no COMMIT <sha40> line in {log}")
        info["revision"] = commits[-1]
    return info


def done_upload(c):
    i = c.manifest("upload").get("info", {})
    return bool(i.get("confirmed") and i.get("revision"))


def st_pointer(c):
    """fec/submit.py pointer for the uploaded revision.  The read token stays in the process
    environment (FEC_HF_READ_TOKEN, inherited by submit.py): it is never put on argv, never
    passed through Ctx.sh's logged env, and the driver's logs/manifests are scanned for it."""
    repo = hf_repo(c)
    rev = c.args.hf_revision or c.manifest("upload").get("info", {}).get("revision")
    if not rev and c.args.dry_run:
        rev = "<upload-commit-sha40>"
    if not rev:
        raise StageError("no revision: run upload --confirm-upload first or give --hf-revision")
    phase = "final" if c.L.split == "private_test" else "dev"
    out = Path(c.args.pointer_out) if c.args.pointer_out else \
        c.L.root / c.args.submission_name / f"pointer_{phase}_{rev[:8]}.zip"
    c.check_writable(out)
    has_tok = bool((os.environ.get(READ_TOKEN_ENV) or "").strip())
    if not has_tok and not c.args.dry_run:
        raise StageError(f"set env {READ_TOKEN_ENV} (fine-grained read token for {repo}) in "
                         f"this shell; it is never read from argv or files")
    cmd = [PY, "-B", FEC / "submit.py", "pointer", "--repo", repo, "--revision", rev,
           "--out", out, "--phase", phase] + (["--force"] if c.args.force else []) + \
        (["--offline"] if c.args.pointer_offline else [])
    c.sh(cmd, c.stage_log("pointer"))
    if c.args.dry_run:
        return {}
    leaks = token_leaks(c)
    if leaks:
        raise StageError(f"read token found in driver files {leaks}: delete them, revoke the "
                         f"token and make a new one")
    return dict(repo=repo, revision=rev, phase=phase, zip=str(out), zip_bytes=out.stat().st_size,
                zip_sha256=sha256(out), offline=bool(c.args.pointer_offline),
                token_scan="clean (logs, manifests, recipe.json, view/verify json)",
                note="the pointer zip contains the read token by design; do not commit or share it")


def token_leaks(c, extra=()):
    """Files under the run root (except pointer zips) that contain the env read token."""
    tok = (os.environ.get(READ_TOKEN_ENV) or "").strip()
    if not tok:
        return []
    needle = tok.encode()
    bad = []
    pats = ("logs/*", "manifest/*.json", "recipe.json", "verify/*.json", "*/*.json", "*.json")
    seen = set()
    for pat in pats:
        for f in c.L.root.glob(pat):
            if f in seen or not f.is_file() or f.suffix == ".zip":
                continue
            seen.add(f)
            if needle in f.read_bytes():
                bad.append(str(f))
    for f in extra:
        if Path(f).is_file() and needle in Path(f).read_bytes():
            bad.append(str(f))
    return bad


def done_pointer(c):
    return False


# ---------------------------------------------------------------------------- verify
def reference(c, kind):
    """Rehearsal reference dir for kind d3d|mast|submission: --ref-<kind>, else the recipe's
    reference block (paths relative to the parent of FEC_WORK), else None."""
    arg = getattr(c.args, f"ref_{kind}", None)
    if arg:
        return Path(arg)
    r = (c.recipe.get("reference") or {}).get(kind)
    return resolve_path(r) if r else None


def verify_hybrid(c):
    """End-to-end check of a (subset) rehearsal against the submitted files.

    The recomputed final-MAST rows (psi, every scalars.npz key, betaN) are spliced into a copy
    of the reference final MAST dir; build_submission (recipe psi dtype, same view of the ds
    caches) then builds both files with the recomputed DIII-D predictions, and the two npz are
    compared with the reference submission (sha256 and member by member).  Identical files
    mean the recomputed rows reproduce the submitted members exactly."""
    L = c.L
    rsub = reference(c, "submission") or REFERENCE["submission"]
    rmast = reference(c, "mast") or REFERENCE["mast_loff"]
    d = _mast_input(c, "final")
    d3d, _ = d3d_submit_dir(c)
    bn = c.recipe["mast"]["betan"]
    need = [d / "psi.npy", d / "scalars.npz", d / bn, rmast / "psi.npy", rmast / "scalars.npz",
            rmast / bn, d3d / f"{L.d3d_cfg}_psi.npy", rsub / f"{L.d3d_cfg}.npz",
            rsub / f"{L.mast_cfg}.npz"]
    miss = [str(f) for f in need if not f.exists()]
    if miss:
        return dict(ok=False, pending=True, reason=f"missing {miss}")
    n = np.load(d / "psi.npy", mmap_mode="r").shape[0]
    hyb = L.verify_dir / f"mast_hybrid{L.sfx}"
    hyb.mkdir(parents=True, exist_ok=True)
    rp = np.load(rmast / "psi.npy", mmap_mode="r")
    sp = np.load(d / "psi.npy", mmap_mode="r")
    tmp = hyb / "psi.part.npy"
    out = np.lib.format.open_memmap(tmp, "w+", rp.dtype, rp.shape)
    for s0 in range(0, rp.shape[0], 4096):          # reference rows, then recomputed prefix
        s1 = min(rp.shape[0], s0 + 4096)
        out[s0:s1] = rp[s0:s1]
    for s0 in range(0, n, 4096):
        s1 = min(n, s0 + 4096)
        out[s0:s1] = sp[s0:s1]
    out.flush()
    del out
    os.replace(tmp, hyb / "psi.npy")
    with np.load(rmast / "scalars.npz") as zr, np.load(d / "scalars.npz") as zs:
        if set(zr.files) != set(zs.files):
            return dict(ok=False, reason=f"scalars keys differ {sorted(zr.files)} {sorted(zs.files)}")
        sc = {}
        for k in zr.files:
            a = np.array(zr[k])
            a[:n] = zs[k][:n]
            sc[k] = a
    np.savez(hyb / "scalars.npz", **sc)
    b = np.array(np.load(rmast / bn))
    b[:n] = np.load(d / bn)[:n]
    np.save(hyb / bn, b)
    write_json(hyb / "hybrid.json", dict(rows=n, recomputed=str(d), reference=str(rmast)))
    name = L.verify_dir / f"submission_hybrid{L.sfx}"
    vw, links = submission_work(c)
    c.sh([PY, "-B", FEC / "build_submission.py", "--name", name, "--d3d", d3d, "--mast", hyb,
          "--split", L.split, "--mast-betan", bn, "--psi-dtype", psi_dtype(c)],
         c.stage_log("verify"), env={"FEC_WORK": str(vw)} if vw else None)
    from fec.features import load_ds
    shot = np.asarray(load_ds(L.mast_cfg)["shot"])
    res = dict(rows_recomputed=int(n), mast_shots_recomputed=int(shot[n - 1]) + 1,
               mast_hybrid=str(hyb), d3d=str(d3d), reference=str(rsub), build=str(name),
               view_links=links, files={})
    for cfg in L.configs:
        fa, fb = name / f"{cfg}.npz", rsub / f"{cfg}.npz"
        r = cmp_npz_members(fa, fb)
        r.update(sha256=sha256(fa), reference_sha256=sha256(fb))
        r["same_sha256"] = r["sha256"] == r["reference_sha256"]
        res["files"][cfg] = r
    res["ok"] = all(r["ok"] for r in res["files"].values())
    res["exact"] = all(r["same_sha256"] for r in res["files"].values())
    return res


def st_verify(c):
    """public_test rehearsal: compare every rehearsal output that has a reference artefact."""
    L = c.L
    if not L.shadow:
        raise StageError("verify compares public_test rehearsal outputs with existing artefacts")
    res = {}
    if c.args.dry_run:
        return {}
    only = set(c.args.verify_only.split(",")) if c.args.verify_only else None

    def want(k):
        return only is None or k in only

    def put(k, r):
        res[k] = r
        c.log(f"verify {k}: ok={r.get('ok')} exact={r.get('exact')} "
              f"{'' if r.get('ok') else json.dumps(r, default=_jsonable)[:300]}")
    # data caches (shadow copies vs canonical)
    for cfg in L.configs:
        sd, cd = L.shots_dir(cfg), L.shots_dir(cfg, canonical=True)
        if want("extract") and (sd / "_meta.json").exists():
            stems = json.loads((cd / "_meta.json").read_text())["order"]
            r = cmp_shot_dirs(sd, cd, stems)
            ma = json.loads((sd / "_meta.json").read_text())
            mb = json.loads((cd / "_meta.json").read_text())
            if cfg.startswith("mast") and "ip_timebase" not in mb:   # legacy MAST cache: the
                r["ignored_meta_keys"] = ["ip_timebase"]             # key does not apply to MAST
                ma.pop("ip_timebase", None)
            r["meta_equal"] = ma == mb
            r["ok"] = r["ok"] and r["meta_equal"]
            r["reference"] = str(cd)
            put(f"extract/{cfg}", r)
        sdd, cdd = L.ds_dir(cfg), L.ds_dir(cfg, canonical=True)
        if want("dataset") and (sdd / "info.json").exists():
            r = cmp_file_hashes(sdd, cdd, sorted(p.name for p in cdd.iterdir() if p.is_file()))
            r["reference"] = str(cdd)
            put(f"dataset/{cfg}", r)
    tag = c.recipe["mast"]["stage1"]["passive"]
    sp, cp = L.passive_dir(L.mast_cfg, tag), L.passive_dir(L.mast_cfg, tag, canonical=True)
    if want("passive") and (sp / "I.npy").exists():
        r = {"I": cmp_arrays(np.load(sp / "I.npy", mmap_mode="r"),
                             np.load(cp / "I.npy", mmap_mode="r")),
             "totals": cmp_npz(sp / "totals.npz", cp / "totals.npz")}
        ss = L.data_work / "passive" / "sig" / L.mast_cfg
        cs = L.work / "passive" / "sig" / L.mast_cfg
        stems = sorted(p.stem for p in ss.glob("*.npz") if not p.name.endswith(".part.npz"))
        r["signals"] = cmp_shot_dirs(ss, cs, stems)
        ms = json.loads((sp / "meta.json").read_text())["model_cache"]
        mc = json.loads((cp / "meta.json").read_text())["model_cache"]
        r["model_cache"] = cmp_npz(ms, mc) if Path(ms).exists() and Path(mc).exists() else \
            dict(ok=False, reason="missing model cache")
        r["ok"] = all(v["ok"] for v in r.values() if isinstance(v, dict))
        r["exact"] = all(v.get("exact", False) for v in r.values() if isinstance(v, dict))
        put("passive", r)
    # DIII-D predictions
    if want("d3d"):
        for key in ("seq", "b", "mix"):
            if not c.recipe["d3d"].get(key):
                continue
            name = c.recipe["d3d"][key]["name"]
            da = L.preds / name
            ref = reference(c, "d3d")
            db = ref if ref and name == d3d_final_name(c.recipe) else REFERENCE["preds"] / name
            if (da / f"{L.d3d_cfg}_psi.npy").exists() and (db / f"{L.d3d_cfg}_psi.npy").exists():
                r = cmp_pred_dirs(da, db, L.d3d_cfg)
                r["reference"] = str(db)
                r["same_sha256"] = {k: sha256(da / f"{L.d3d_cfg}_{k}.npy") ==
                                    sha256(db / f"{L.d3d_cfg}_{k}.npy") for k in ("psi", "scal")}
                put(f"d3d/{name}", r)
        if c.recipe["d3d"].get("q95h"):          # q95 hybrid output vs its reference
            _, da = q95h_dirs(c)
            db = reference(c, "d3d_q95h")
            if db and (da / f"{L.d3d_cfg}_scal.npy").exists() and \
                    (db / f"{L.d3d_cfg}_scal.npy").exists():
                fa, fb = da / f"{L.d3d_cfg}_scal.npy", db / f"{L.d3d_cfg}_scal.npy"
                r = cmp_arrays(np.load(fa), np.load(fb))
                r.update(reference=str(db), scal_sha256=sha256(fa), reference_scal_sha256=sha256(fb),
                         same_sha256=sha256(fa) == sha256(fb))
                put(f"d3d/{da.name}", r)
        if c.recipe["d3d"].get("lirep4"):        # lirep4 output (psi + scal) vs its reference
            _, da = lirep4_dirs(c)
            db = reference(c, "d3d_lirep4")
            if db and (da / f"{L.d3d_cfg}_psi.npy").exists() and                     (db / f"{L.d3d_cfg}_psi.npy").exists():
                r = cmp_pred_dirs(da, db, L.d3d_cfg)
                r["reference"] = str(db)
                r["same_sha256"] = {k: sha256(da / f"{L.d3d_cfg}_{k}.npy") ==
                                    sha256(db / f"{L.d3d_cfg}_{k}.npy") for k in ("psi", "scal")}
                put(f"d3d/{da.name}", r)
    # MAST stage 1 subsets vs production artefacts
    if want("mast"):
        for arm in ("v2eq", "loff", "lon"):
            d = L.mast_dir(arm)
            if (d / "meta.npz").exists():
                rows = np.load(d / "psi.npy", mmap_mode="r").shape[0]
                ref = (reference(c, "mast") if arm == "loff" else None) or REFERENCE[f"mast_{arm}"]
                put(f"mast_{arm}{L.sfx}", cmp_mast_dirs(d, ref, rows) if ref.exists()
                    else dict(ok=False, pending=True, reason=f"no reference {ref}"))
        # final MAST dir (subset or full): stage-2 scalars, betaN records and betan_v2 rows vs
        # the reference final MAST dir (same recipe: --boundary-limiter + limited betaN)
        d, ref = _mast_input(c, "final"), reference(c, "mast") or REFERENCE["mast_loff"]
        if (d / "scalars.npz").exists() and (ref / "scalars.npz").exists():
            rows = np.load(d / "psi.npy", mmap_mode="r").shape[0]
            r = cmp_npz(d / "scalars.npz", ref / "scalars.npz", rows=rows)
            r.update(rows=rows, reference=str(ref / "scalars.npz"))
            put(f"mast_stage2_final{L.sfx}", r)
            bn = c.recipe["mast"]["betan"]
            if (d / bn).exists() and (ref / bn).exists():
                x = np.load(d / bn)
                y = np.load(ref / bn)[:len(x)]
                r = cmp_arrays(x[:, None], y[:, None])
                r.update(rows=int(len(x)), reference=str(ref / bn))
                put(f"mast_betan_final{L.sfx}", r)
            if (d / "betan_records.npz").exists() and (ref / "betan_records.npz").exists():
                with np.load(d / "betan_records.npz") as za, \
                        np.load(ref / "betan_records.npz") as zb:
                    sel = zb["gidx"] < rows
                    bad = [k for k in za.files if k not in zb.files or not _eq(za[k], zb[k][sel])]
                    put(f"mast_betan_records_final{L.sfx}",
                        dict(ok=not bad and len(za["gidx"]) == int(sel.sum()), exact=not bad,
                             rows=int(sel.sum()), mismatched_fields=bad))
        d = L.mast_dir("v2eq")
        if (d / "scalars.npz").exists():
            rows = np.load(d / "psi.npy", mmap_mode="r").shape[0]
            put(f"mast_stage2_v2eq{L.sfx}", cmp_npz(d / "scalars.npz",
                                                    REFERENCE["mast_v2eq"] / "scalars.npz",
                                                    rows=rows))
    # MAST composite: smoothed amortized maps, physics scalars and fill3 betaN vs references
    if want("composite") and c.recipe["mast"].get("composite"):
        D = composite_dirs(c)
        rm, rs, rf = reference(c, "mast_maps"), reference(c, "mast_scalars"), reference(c, "mast")
        if rm and (D["maps"] / "psi.npy").exists() and (rm / "psi.npy").exists():
            rows = np.load(D["maps"] / "psi.npy", mmap_mode="r").shape[0]
            r = cmp_arrays(np.load(D["maps"] / "psi.npy", mmap_mode="r"),
                           np.load(rm / "psi.npy", mmap_mode="r"), rows=rows)
            r.update(rows=rows, reference=str(rm / "psi.npy"))
            if rows == np.load(rm / "psi.npy", mmap_mode="r").shape[0]:
                r["same_sha256"] = sha256(D["maps"] / "psi.npy") == sha256(rm / "psi.npy")
            put(f"composite/maps{L.sfx}", r)
        sd = composite_scalars_dir(c)
        if rs and (sd / "scalars.npz").exists() and (rs / "scalars.npz").exists():
            rows = np.load(sd / "psi.npy", mmap_mode="r").shape[0]
            r = cmp_npz(sd / "scalars.npz", rs / "scalars.npz", rows=rows)
            r.update(rows=rows, reference=str(rs / "scalars.npz"), dir=str(sd))
            put(f"composite/physics_scalars{L.sfx}", r)
            if rows == np.load(rs / "psi.npy", mmap_mode="r").shape[0]:
                put(f"composite/physics_psi{L.sfx}", dict(
                    cmp_arrays(np.load(sd / "psi.npy", mmap_mode="r"),
                               np.load(rs / "psi.npy", mmap_mode="r")), reference=str(rs)))
            else:
                put(f"composite/physics_psi{L.sfx}", dict(
                    cmp_arrays(np.load(sd / "psi.npy", mmap_mode="r"),
                               np.load(rs / "psi.npy", mmap_mode="r"), rows=rows),
                    rows=rows, reference=str(rs)))
            if (sd / "betan_records.npz").exists() and (rs / "betan_records.npz").exists():
                with np.load(sd / "betan_records.npz") as za,                         np.load(rs / "betan_records.npz") as zb:
                    sel = zb["gidx"] < rows
                    bad = [k for k in za.files if k not in zb.files or not _eq(za[k], zb[k][sel])]
                    put(f"composite/physics_betan_records{L.sfx}",
                        dict(ok=not bad and len(za["gidx"]) == int(sel.sum()), exact=not bad,
                             rows=int(sel.sum()), mismatched_fields=bad))
        bn = c.recipe["mast"]["betan"]
        if rf and (D["final"] / bn).exists() and (rf / bn).exists():
            x, y = np.load(D["final"] / bn), np.load(rf / bn)
            r = cmp_arrays(x[:, None], y[:len(x), None])
            r.update(rows=int(len(x)), reference=str(rf / bn),
                     same_sha256=sha256(D["final"] / bn) == sha256(rf / bn))
            put(f"composite/betan_fill3{L.sfx}", r)
    # betaN: record wiring on the reference psi (subset prefix) + frozen model on full records
    if want("betan"):
        from fec.features import load_ds
        shot_all = np.asarray(load_ds(L.mast_cfg)["shot"])
        with np.load(REFERENCE["betan_records"]) as z:
            ref = {k: z[k] for k in z.files}
        out, _ = betan_apply(ref, shot_all)
        put("betan/apply_full_records", cmp_arrays(out[:, None],
                                                   np.load(REFERENCE["betan"])[:, None]))
        n = subset_limit(shot_all, L.subset_shots or 64)
        part = L.verify_dir / f"psi_mast_v2_prefix{n}.npy"
        if not part.exists():
            part.parent.mkdir(parents=True, exist_ok=True)
            np.save(part, np.load(REFERENCE["mast_v2eq"] / "psi.npy", mmap_mode="r")[:n])
        rec = betan_records(L.mast_cfg, part, nproc=int(c.args.workers))
        sel = ref["gidx"] < n
        r = {"rows": int(sel.sum())}
        bad = []
        for k in rec:
            x = ref[k][sel]
            if not _eq(rec[k], x):
                bad.append(k)
        r.update(ok=not bad and len(rec["gidx"]) == int(sel.sum()), exact=not bad,
                 mismatched_fields=bad)
        put(f"betan/records_prefix{n}", r)
        # limited betaN of the final MAST dir: records may differ from the frozen (unlimited)
        # geometry only on frames whose boundary flux changed
        d = _mast_input(c, "final")
        bm = c.manifest("mast_betan").get("info", {})
        if (d / "betan_records.npz").exists() and bm.get("boundary_limiter") and \
                bm.get("mast_dir") == str(d):
            with np.load(d / "betan_records.npz") as z:
                lim = {k: z[k] for k in z.files}
            unl = betan_records(L.mast_cfg, d / "psi.npy", nproc=int(c.args.workers))
            fields = [k for k in lim if k not in ("gidx", "shot", "t_ms")]
            diff = np.zeros(len(lim["gidx"]), bool)
            for k in fields:
                x, y = lim[k], unl[k]
                diff |= ~((x == y) | (np.isnan(x) & np.isnan(y)))
            span_same = (lim["flux_span"] == unl["flux_span"]) | \
                (np.isnan(lim["flux_span"]) & np.isnan(unl["flux_span"]))
            n = len(shot_all) if len(lim["gidx"]) == len(shot_all) else int(lim["gidx"].max()) + 1
            bl, _ = betan_apply(lim, shot_all[:n])
            bu, _ = betan_apply(unl, shot_all[:n])
            rel = bl[diff] / np.maximum(bu[diff], 1e-9) - 1 if diff.any() else np.zeros(0)
            put("betan/limiter_vs_frozen_geometry",
                dict(ok=bool(not np.any(diff & span_same)), exact=None, frames=int(len(diff)),
                     frames_changed=int(diff.sum()), frac_changed=float(diff.mean()),
                     changed_without_boundary_change=int(np.sum(diff & span_same)),
                     betan_rel_change_median=float(np.median(rel)) if rel.size else 0.0,
                     betan_rel_change_p5_p95=(np.percentile(rel, [5, 95]).tolist()
                                              if rel.size else [0.0, 0.0]),
                     a_ratio_median_changed=float(np.nanmedian(lim["a"][diff] / unl["a"][diff]))
                     if diff.any() else 1.0))
    # end-to-end: the reference final MAST dir with the recomputed rows spliced in, built
    # together with the recomputed DIII-D predictions, vs the submitted files (sha256 + members)
    if want("hybrid"):
        put(f"hybrid{L.sfx}", verify_hybrid(c))
    # submission (rehearsal build with the reference MAST dir) vs submitted files
    if want("submission"):
        rsub = reference(c, "submission") or REFERENCE["submission"]
        rb = rsub / "build.json"
        ref_build = json.loads(rb.read_text()) if rb.exists() else {}
        for sub in sorted(L.root.glob("submission*")):
            if not (sub / "build.json").exists():
                continue
            mine = json.loads((sub / "build.json").read_text())
            for cfg in L.configs:
                if cfg.startswith("diii"):
                    keys = ("d3d", "psi_dtype")
                elif c.recipe["mast"].get("composite"):     # composite dir name differs by design
                    keys = ("mast_betan", "k_beta", "psi_dtype")
                else:
                    keys = ("mast", "mast_betan", "k_beta", "psi_dtype")
                diff = {k: (mine.get(k), ref_build.get(k)) for k in keys
                        if Path(str(mine.get(k))).name != Path(str(ref_build.get(k))).name}
                fa, fb = sub / f"{cfg}.npz", rsub / f"{cfg}.npz"
                if diff:
                    put(f"{sub.name}/{cfg}", dict(ok=False, pending=True, reason=(
                        f"reference {rsub} was built from other inputs "
                        f"(rehearsal, reference): {diff}")))
                elif fa.exists() and fb.exists():
                    r = cmp_npz_members(fa, fb)
                    r["same_sha256"] = sha256(fa) == sha256(fb)
                    put(f"{sub.name}/{cfg}", r)
    vf = L.verify_dir / f"verify{L.sfx}.json"
    allres = json.loads(vf.read_text()) if vf.exists() else {}
    allres.update(res)                      # --verify-only runs update their own keys only
    write_json(vf, allres)
    summary = {k: dict(ok=v.get("ok"), exact=v.get("exact"), pending=v.get("pending", False))
               for k, v in res.items()}
    failed = [k for k, v in res.items() if not v.get("ok") and not v.get("pending")]
    if failed:
        raise StageError(f"verification failed: {failed}")
    return summary


def cmp_npz_members(fa, fb):
    """Submission npz: same member names and identical arrays (members read one at a time)."""
    with np.load(fa) as za, np.load(fb) as zb:
        ka, kb = set(za.files), set(zb.files)
        bad = [k for k in sorted(ka & kb) if not _eq(za[k], zb[k])]
    return dict(ok=ka == kb and not bad, exact=ka == kb and not bad, members=len(ka),
                only_a=sorted(ka - kb)[:5], only_b=sorted(kb - ka)[:5], mismatched=bad[:10],
                n_mismatched=len(bad))


def done_verify(c):
    return False


STAGES = {
    "download": (st_download, done_download),
    "ipguard": (st_ipguard, done_ipguard),
    "extract": (st_extract, done_extract),
    "order": (st_order, done_order),
    "dataset": (st_dataset, done_dataset),
    "d3d_train": (st_d3d_train, done_d3d_train),
    "d3d_seq": (st_d3d_seq, done_d3d_seq),
    "d3d_b": (st_d3d_b, done_d3d_b),
    "d3d_mix": (st_d3d_mix, done_d3d_mix),
    "d3d_q95h": (st_d3d_q95h, done_d3d_q95h),
    "d3d_dec": (st_d3d_dec, done_d3d_dec),
    "d3d_mmk": (st_d3d_mmk, done_d3d_mmk),
    "d3d_lirep4": (st_d3d_lirep4, done_d3d_lirep4),
    "passive": (st_passive, done_passive),
    "mast_loff": (lambda c: _st_mast_arm(c, "loff"), lambda c: _done_mast_arm(c, "loff")),
    "mast_lon": (lambda c: _st_mast_arm(c, "lon"), lambda c: _done_mast_arm(c, "lon")),
    "mast_merge": (st_mast_merge, done_mast_merge),
    "mast_stage2": (st_mast_stage2, done_mast_stage2),
    "mast_betan": (st_mast_betan, done_mast_betan),
    "mast_pipeline": (st_mast_pipeline, done_mast_pipeline),
    "mast_amort": (st_mast_amort, done_mast_amort),
    "mast_physics": (st_mast_physics, done_mast_physics),
    "mast_composite": (st_mast_composite, done_mast_composite),
    "submission": (st_submission, done_submission),
    "mast_v2eq": (lambda c: _st_mast_arm(c, "v2eq"), lambda c: _done_mast_arm(c, "v2eq")),
    "verify": (st_verify, done_verify),
    "upload": (st_upload, done_upload),
    "pointer": (st_pointer, done_pointer),
}
ALL = ("download", "ipguard", "extract", "order", "dataset", "passive", "mast_loff", "mast_lon",
       "mast_merge", "d3d_seq", "d3d_b", "d3d_mix", "d3d_q95h", "d3d_dec", "d3d_mmk", "d3d_lirep4", "mast_stage2",
       "mast_betan", "submission")
MERGE_ONLY = ("mast_lon", "mast_merge")
UPLOAD = ("upload", "pointer")


def all_stages(recipe):
    """`all` order for a recipe: limiter-on run and merge only when mast.final == 'merged'."""
    final = recipe["mast"].get("final", "loff")
    if final not in ("loff", "merged"):
        raise ValueError(f"recipe mast.final must be 'loff' or 'merged', not {final!r}")
    absent = {f"d3d_{k}" for k in ("seq", "b", "mix") if not recipe["d3d"].get(k)}
    if recipe_ip_timebase(recipe) != "idx":
        absent.add("ipguard")
    if not recipe["d3d"].get("q95h"):
        absent.add("d3d_q95h")
    if not recipe["d3d"].get("lirep4"):
        absent.add("d3d_lirep4")
    if not recipe["d3d"].get("dec"):
        absent.add("d3d_dec")
    if not recipe["d3d"].get("mmk"):
        absent.add("d3d_mmk")
    if recipe["mast"].get("composite"):     # amortized maps + physics scalars (Final A/B)
        out = []
        for s in ALL:
            if s in absent or (s in PIPELINE_REPLACES and s != "mast_loff") or s in MERGE_ONLY:
                continue
            if s == "mast_loff":
                out += ["mast_amort", "mast_physics"]
                continue
            if s == "submission":
                out.append("mast_composite")
            out.append(s)
        return out
    if mast_pipeline_file(recipe):          # one mast_pipeline stage replaces the MAST stages
        return [("mast_pipeline" if s == "mast_loff" else s) for s in ALL
                if s == "mast_loff" or (s not in PIPELINE_REPLACES and s not in absent)]
    return [s for s in ALL if (final == "merged" or s not in MERGE_ONLY) and s not in absent]


def load_recipe(path, root):
    if not path:
        src = DEFAULT_RECIPE
    elif path in RECIPES:
        src = RECIPES[path]
    else:
        src = json.loads(Path(path).read_text())
    rec = json.loads(json.dumps(src))
    for k in ("d3d", "mast"):
        if k not in rec:
            raise ValueError(f"recipe lacks {k!r}")
    d = rec["d3d"]
    d3d_final_name(rec)
    recipe_ip_timebase(rec)
    if rec.get("psi_dtype", "float16") not in ("float16", "float32"):
        raise ValueError(f"recipe psi_dtype must be float16 or float32: {rec.get('psi_dtype')}")
    if d.get("mix"):
        known = {d[k]["name"] for k in ("seq", "b") if d.get(k)}
        if set(d["mix"]["members"]) - known:
            raise ValueError(f"mix members {set(d['mix']['members']) - known} are not produced "
                             f"by the recipe")
        if d["mix"]["name"] in known:
            raise ValueError("mix name must differ from its members' names")
    comp = rec["mast"].get("composite")
    if comp:
        if mast_pipeline_file(rec):
            raise ValueError("mast.composite and mast.pipeline are exclusive")
        m = comp.get("maps") or {}
        if m.get("kind") == "physics":
            pass
        elif m.get("kind") in ("vb", "combo"):              # Oct 14 variants (fpkgm)
            if not m.get("net") or "sol_frac" not in m or "tsm_sigma_ms" not in m or not m.get("tx1s") or \
                    (m["kind"] == "combo") == (maps_rscale(m) == "1"):
                raise ValueError(f"mast.composite.maps kind {m['kind']} needs net, sol_frac, tsm_sigma_ms, tx1s "
                                 f"(and rscale != 1 for combo, rscale 1 for vb): {m}")
        elif m.get("kind") != "amort" or not m.get("net") or "sol_frac" not in m or                 "tsm_sigma_ms" not in m:
            raise ValueError(f"mast.composite.maps needs kind physics, or kind amort with net, "
                             f"sol_frac, tsm_sigma_ms: {m}")
        if (comp.get("scalars") or {}).get("plan") != "avg_f05":
            raise ValueError(f"mast.composite.scalars.plan must be avg_f05: {comp.get('scalars')}")
        if (comp.get("betan_fill") or {}).get("rule") != "fill3":
            raise ValueError(f"mast.composite.betan_fill.rule must be fill3")
    if mast_pipeline_file(rec):             # MAST pipeline recipe: validated up front
        from fec import mast_pipeline as MP
        MP.load_recipe(mast_pipeline_file(rec))
    if d.get("train"):
        check_train_block(d["train"])
    if d.get("dec"):                        # off-span decoder stage (final_A22li_dec)
        q = d["dec"]
        if not d.get("b") or not d["b"].get("recipe"):
            raise ValueError("recipe d3d.dec needs a d3d.b recipe file (the members whose basis the decoders use)")
        if not isinstance(q.get("dir"), str) or not q.get("seeds") or \
                not all(isinstance(s, int) for s in q["seeds"]) or q.get("fallback") not in ("src", None):
            raise ValueError(f"recipe d3d.dec needs dir (str), seeds (list of int), fallback 'src' or null: {q}")
    if d.get("mmk"):                        # member-mean kappa finishing (final_MTfull_mmk)
        q = d["mmk"]
        if not d.get("b") or not d["b"].get("recipe") or not d["b"].get("chunk_rows"):
            raise ValueError("recipe d3d.mmk needs a chunked d3d.b recipe (the members' kappa is computed in d3d_b)")
        if q.get("step") != "kappa" or not isinstance(q.get("operator_sha256"), str) or \
                not isinstance(q.get("env_train_sha256"), str) or q.get("fallback") not in ("src", None) or \
                not isinstance(q.get("min_members", 0), int) or q.get("min_members", 0) < 0 or \
                not float(q.get("stall_timeout_s", 600.0)) > 0:
            raise ValueError(f"recipe d3d.mmk needs step 'kappa', operator_sha256, env_train_sha256, fallback "
                             f"'src' or null, min_members >= 0 (int) and stall_timeout_s > 0: {q}")
    f = Path(root) / "recipe.json"
    return rec, f


def status(c):
    rows = []
    pipe = bool(mast_pipeline_file(c.recipe))
    for name in ["d3d_train"] + list(ALL) + ["mast_pipeline"] + list(COMPOSITE_STAGES) + \
            ["mast_v2eq", "verify", "upload", "pointer"]:
        no_merge = c.recipe["mast"].get("final", "loff") != "merged"
        optional = (name in MERGE_ONLY and no_merge) or \
            (name.startswith("d3d_") and not c.recipe["d3d"].get(name[4:])) or \
            (name == "ipguard" and c.ip_timebase != "idx") or name in UPLOAD or \
            (name == "d3d_q95h" and not c.recipe["d3d"].get("q95h")) or \
            (name == "d3d_lirep4" and not c.recipe["d3d"].get("lirep4")) or \
            (name in COMPOSITE_STAGES and not c.recipe["mast"].get("composite")) or \
            (name in PIPELINE_REPLACES and bool(c.recipe["mast"].get("composite"))) or \
            (name == "mast_pipeline" and not pipe) or (name in PIPELINE_REPLACES and pipe) or             (name == "d3d_train" and not c.recipe["d3d"].get("train"))
        if optional and not c.manifest(name):
            continue
        m = c.manifest(name)
        try:
            done = STAGES[name][1](c)
        except Exception as e:  # noqa: BLE001 - status is best effort
            done = f"? ({type(e).__name__})"
        rows.append((name, m.get("status", "-"), m.get("finished", "-"), m.get("seconds", "-"),
                     done))
    print(f"split {c.L.split}{c.L.sfx}  root {c.L.root}  recipe {c.recipe.get('name')}  "
          f"ip_timebase {c.ip_timebase}  d3d work {c.L.d3d_work}")
    for r in rows:
        print(f"  {r[0]:12s} {str(r[1]):8s} {str(r[2]):26s} {str(r[3]):>8s}s  outputs complete: {r[4]}")
    tot = sum(float(r[3]) for r in rows if isinstance(r[3], (int, float)))
    print(f"  total recorded stage time {tot:.0f}s ({tot / 3600:.2f} h)")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_predict_b":          # internal: chunked DIII-D ensemble subprocess
        return _internal_predict_b(argv[1:])
    if argv and argv[0] == "_exec":               # internal: env wrapper under gpu_slot
        return _internal_exec(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["all", "status"])
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--recipe", help=f"preset name {sorted(RECIPES)} or a recipe JSON file "
                                     f"(default: legacy DEFAULT_RECIPE; the current submission "
                                     f"recipe is 'sub04')")
    ap.add_argument("--subset-shots", type=int, default=0,
                    help="MAST stage 1 on the smallest production chunk prefix covering this "
                         "many shots (smoke/rehearsal; outputs get a _sub<K> suffix)")
    ap.add_argument("--workers", type=int, default=6, help="CPU pool size (extract/passive/betaN)")
    ap.add_argument("--owner", default="root", help="gpu_slot / runlog owner")
    ap.add_argument("--gpu-priority", default="normal", choices=["normal", "high"])
    ap.add_argument("--force", action="store_true", help="re-run stages that are already done")
    ap.add_argument("--skip", default="", help="comma-separated stages to skip in `all`")
    ap.add_argument("--from", dest="from_stage", default=None, help="start `all` at this stage")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    ap.add_argument("--no-runlog", action="store_true", help="do not append research/fec_runs.jsonl")
    ap.add_argument("--remote", action="store_true", help="download: compare with the HF listing")
    ap.add_argument("--loff-dir", help="mast_merge: limiter-off stage-1 dir (default mast/loff)")
    ap.add_argument("--lon-dir", help="mast_merge: limiter-on stage-1 dir (default mast/lon)")
    ap.add_argument("--mast-dir", help="stage2/betan/submission MAST dir (default: the recipe's "
                                       "mast.final dir, mast/loff or mast/merged)")
    ap.add_argument("--blim", default="recipe", choices=["recipe", "on", "off"],
                    help="stage2/betan boundary-only limiter (default: recipe mast.boundary_limiter)")
    ap.add_argument("--d3d-dir", help="submission: DIII-D preds dir used as-is (default: the "
                                      "q95-hybrid dir of a d3d.q95h recipe, else preds/<final name>)")
    ap.add_argument("--q95h-src", help="d3d_q95h: head predictions dir (default preds/<final name>)")
    ap.add_argument("--submission-name", default="submission",
                    help="output dir name under the split root")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="mast_betan: proceed if the frozen betaN files' hashes changed")
    ap.add_argument("--root", default=None,
                    help="run root (default fec_work/final_<split>)")
    ap.add_argument("--adopt", default=None,
                    help="mast_loff/mast_lon: copy this finished stage-1 dir (same options and "
                         "frame count, checked) instead of recomputing it")
    ap.add_argument("--verify-only", default="",
                    help="verify: comma list of extract,dataset,passive,d3d,mast,betan,hybrid,"
                         "submission")
    ap.add_argument("--ref-d3d", help="verify: reference DIII-D preds dir (default: recipe "
                                      "reference.d3d, else fec_work/preds/<name>)")
    ap.add_argument("--ref-mast", help="verify: reference final MAST dir (default: recipe "
                                       "reference.mast, else fec_work/mast_v3_loff)")
    ap.add_argument("--ref-submission", help="verify: reference submission dir (default: recipe "
                                             "reference.submission, else submissions/sub02)")
    ap.add_argument("--d3d-work", help="DIII-D work root (default: recipe d3d.work, relative to "
                                       "the parent of FEC_WORK; legacy recipes: FEC_WORK)")
    ap.add_argument("--psi-dtype", choices=["float16", "float32"],
                    help="submission psirz dtype (default: recipe psi_dtype, else float16)")
    ap.add_argument("--no-model-hash", action="store_true",
                    help="d3d_b: skip sha256 of every model.pt in the manifest")
    ap.add_argument("--with-train", action="store_true",
                    help="all: also run d3d_train (recipe d3d.train) before the DIII-D predictor")
    ap.add_argument("--d3d-trainer", default="recipe", choices=["recipe"] + list(TRAINERS),
                    help="d3d_train: recipe d3d.train.trainer (default serial = train_d3d_b.py per "
                    "member), or force serial / bx (fec/train_d3d_bx.py production layout)")
    ap.add_argument("--amort-net-dir", help="mast_amort: amortized net dir (default "
                                             "<FEC_WORK>/amort/models/<recipe net>)")
    ap.add_argument("--tx1s-model-dir", help="mast_amort, maps kind vb/combo: TX1s model dir (default "
                                              "<FEC_WORK>/xfer/models1/<recipe tx1s>)")
    ap.add_argument("--dec-dir", help="d3d_dec: directory of the decoders dec_s<seed>.pt (default "
                                      "<FEC_WORK>/<recipe d3d.dec.dir>)")
    ap.add_argument("--mast-scalars-dir", help="mast_physics/mast_composite (REHEARSAL ONLY): take "
                                               "scalars + betaN records from this finished "
                                               "physics node instead of running avg_f05")
    ap.add_argument("--ts-rule", default="auto", choices=list(TS_RULES),
                    help="extract: Thomson time rule (issue #12): auto = runmax (organisers' "
                         "running maximum, fec/extract_final.py) on private_test, sort "
                         "(fec/extract.py) on public_test")
    ap.add_argument("--mast-jobs", type=int, default=1,
                    help="mast_physics: independent pipeline steps run concurrently")
    ap.add_argument("--with-upload", action="store_true",
                    help="`all`: append upload + pointer (upload stays a dry run without "
                         "--confirm-upload)")
    ap.add_argument("--confirm-upload", action="store_true",
                    help="upload: really create the HF commit (root only)")
    ap.add_argument("--hf-repo", help="upload/pointer: HF dataset repo (default recipe upload.repo)")
    ap.add_argument("--hf-revision", help="pointer: 40-hex commit (default: upload manifest)")
    ap.add_argument("--pointer-out", help="pointer: zip path (default <submission>/pointer_*.zip)")
    ap.add_argument("--pointer-offline", action="store_true",
                    help="pointer: skip the online token/repo checks (submit.py --offline)")
    args = ap.parse_args(argv)
    if args.adopt and args.stage not in ("mast_loff", "mast_lon"):
        ap.error("--adopt applies to a single mast_loff or mast_lon stage")
    if args.confirm_upload and args.stage not in ("upload", "all"):
        ap.error("--confirm-upload applies to the upload stage (or all --with-upload)")
    recipe, rfile = load_recipe(args.recipe, Layout(args.split, root=args.root).root)
    c = Ctx(args, recipe)
    if args.stage == "status":
        status(c)
        return 0
    if rfile.exists():
        held = json.loads(rfile.read_text())
        if held != json.loads(json.dumps(recipe)) and not args.force:
            raise SystemExit(f"{rfile} holds a different recipe ({held.get('name')}); use the same "
                             f"--recipe or --force (outputs of the old recipe stay in place)")
    if not args.dry_run:
        write_json(rfile, recipe)
    stages = all_stages(recipe) if args.stage == "all" else [args.stage]
    if args.with_train and args.stage == "all":
        if not recipe["d3d"].get("train"):
            raise SystemExit(f"--with-train: recipe {recipe.get('name')} has no d3d.train")
        first = next(i for i, s in enumerate(stages) if s in ("d3d_seq", "d3d_b", "d3d_mix"))
        stages.insert(first, "d3d_train")
    if args.stage == "all":
        if args.from_stage:
            stages = stages[stages.index(args.from_stage):]
        skip = {s for s in args.skip.split(",") if s}
        if c.L.subset_shots:
            skip |= {"submission", *UPLOAD}
        stages = [s for s in stages if s not in skip]
        if c.L.shadow and "verify" not in skip:
            stages.append("verify")
        if args.with_upload:
            stages += [s for s in UPLOAD if s not in skip]
        if args.split == "private_test" and not args.recipe:
            c.log("WARNING: no --recipe: running the LEGACY sub02 recipe (starter Ip axis). The "
                  "current submission recipe is --recipe sub04.")
    c.log(f"run_final {args.stage} split={args.split} subset={args.subset_shots} "
          f"recipe={recipe.get('name')} stages={stages}")
    for s in stages:
        c.run(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
