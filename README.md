# Fusion Equilibrium Challenge: Final reproduction package

This package reproduces our Final submissions (Codabench 17739) for both challenges: Challenge 1,
DIII-D equilibrium reconstruction, and Challenge 2, MAST zero-shot. It contains the source, the
pinned environment, the configs per Final candidate, the training and inference entrypoints, the
per-machine data provenance (`PROVENANCE.md`) and the manifests of the exact weights and data
(SHA-256).

Built 2026-10-20T14:23:19+00:00 from source commit `ca734729c959a293cbcbe68a5ae46a6f5b5bcbda`. Licence: MIT (`LICENSE`). The vendored
`fusion-equilibrium-challenge-starter/fusion_scoring` is the organisers' MIT code (commit
`5b467988d2e5f28e20be490a926d0a71649de28d`).

## Final candidates (`configs/`)

| config | DIII-D | MAST |
|---|---|---|
| `final_A24` | W3k24: 12 width-3072 / 120-epoch shape members + 12 width-3072 / 120-epoch no-Thomson members (all-data), the old 28 no-Thomson members as fallback, then the q95 G1 hybrid and lirep4; sized to fit the 24 A100-h rebuild | amortized v7h net at f = 0.55 + 30 ms smoothing (maps); `avg_f05` physics average (q95); lam_fgw + fill3 (betaN) |
| `final_B` | W3e44: 16 W3k shape members + 28 width-1536 / 60-epoch no-Thomson members, then q95 G1 and lirep4 | as A24 |
| `final_C` | as B | reserve without the network: maps, q95 and betaN all from `avg_f05` |
| `final_A22li` | W3k22 + li2ft arm A: the W3k24 members minus `shapeL_s83` and `notsL_s127` (11 + 11), each fine-tuned on all labelled shots with the li(2) loss below; equal weights, families 1/2 each; the old 28 no-Thomson members (not fine-tuned) as fallback, then q95 G1 and lirep4 | as A24 |
| `final_A22li_r05` | as `final_A22li` | as `final_A22li`, with the MAST C1 passive currents at passive-resistance scale r = 0.5 for the amortized maps (no new training) |
| `final_A22li_r2` | as `final_A22li` | as `final_A22li`, with the MAST C1 passive currents at passive-resistance scale r = 2 for the amortized maps (no new training) |
| `final_A22li_dec` | `final_A22li`'s members, then q95 G1, then the **off-span decoder** (below), then lirep4 | as `final_A22li_r2` |
| `final_A24r2` | as `final_A24` (same members, train spec and chain) | as `final_A22li_r2` (the r = 2 maps of `final_A22li_dec` / `final_MTfull`; no new training) |
| `final_MTfull` | `final_A22li_dec` with the 22 fine-tuned members replaced by their **MT** twins (below): the same 11 + 11 W3k22 recipes and seeds, trained with the auxiliary shape / J_phi heads, then the li2ft fine-tune with the heads kept on; the same old 28 no-Thomson fallback, q95 G1, off-span decoder and lirep4 | as `final_A22li_r2` |
| `final_MTli` | `final_A22li_dec` with only its 11 fine-tuned shape (F1) members replaced by their **MT** twins (= `final_MTfull`'s F1: bases with the auxiliary heads, li2ft with the heads kept on); the 11 fine-tuned no-Thomson (F2) members, the old 28 no-Thomson fallback, q95 G1, off-span decoder and lirep4 are `final_A22li_dec`'s | as `final_A22li_r2` |
| `final_MTfull_mmk` | `final_MTfull` exactly (members, fallback, q95 G1, off-span decoder, lirep4), with the **member-mean κ finishing** (below) between the off-span decoder and lirep4 | as `final_A22li_r2` |
| `final_A22li_dec_mmk` | `final_A22li_dec` exactly (members, fallback, q95 G1, off-span decoder, lirep4), with the **member-mean κ finishing** (below) between the off-span decoder and lirep4 | as `final_A22li_r2` |

**Off-span decoder (`final_A22li_dec`, `fec/offdec.py`).** Every member predicts psi as mean + V c in one shared
128-direction PCA basis (stored in each `model.pt`). Three small networks (seeds 40, 41, 42; MLP 128 → 1536 × 3 → 4225,
output projected orthogonal to V) learn the part of psi outside that basis from c: trained on the labels of every
labelled `diii_d_train` shot (the members' all-data rows, every 2nd frame), loss = weighted pixel MSE + 64 × weighted
gradient MSE (weights peak on the label LCFS inside a TRAIN-label envelope and at the lowest and highest points of the
label LCFS polygon, not located X-points), AdamW 3e-4, 40 epochs.
At inference the mean of the three corrections is added to the q95-hybrid psi on every frame (q95 and betaN unchanged),
then lirep4 runs on the result (`fec/run_final.py` stage `d3d_dec`). `train.py --stage d3d` trains the decoders right
after the members (`--stage offdec` alone re-runs them); it stops unless every member, fallback included, holds the same
basis to 1e-6. The envelope is rebuilt from TRAIN labels and checked against its SHA-256; no scorer mask is used. If a
check of the inference step fails, the stage falls back to the q95-hybrid psi (= `final_A22li`) and says so in its
manifest.

**li2ft fine-tune (`final_A22li`).** Each W3k22 member is fine-tuned for 12 epochs from its trained
checkpoint (`--ft_from`; same flags, PCA basis and normalisation, learning rate 10% of the base
rate, configs/jobs/li2ftA22.json). The base loss is unchanged. One term is added: the organisers'
li(2) functional of the predicted psi, evaluated on the TRUE map's own LCFS (the scorer's LCFS
routine), against the li(2) of the true map. Frames whose true li(2) is above 1.7 weigh 4x. The term's
weight λ is set per member to 10% of the base gradient (ratio of median gradient norms on the psi
output layer, measured on the first batches, then frozen). Targets and contours come from a CPU
cache built once from the `diii_d_train` labels of the frozen all-data split
(`train.py --stage li2cache`, `fec/li2_torch.py`). The cache uses the TRAIN-only envelope
`frozen/d3d/env_train.npz` and no scorer mask. The li2ft stack runs last in `train.py --stage d3d`,
with `--li2_cache`.

**MT members (`final_MTfull`; the shape half of `final_MTli`; `fec/train_d3d_bx.py --aux_heads`, `fec/mtask_aux_all.py`).** Each W3k22 base is
trained exactly as in `final_A22li` (same job line, seed and flags; tags `d3d_bxw3emt_*`) with one addition: a linear
auxiliary head on the shared trunk predicts 20 training-only targets of the TRUE map, κ, δ_top, δ_bot and volume of
the scorer's LCFS plus the flux-surface-averaged J_φ at 16 normalised-flux points (`fec/mtask_aux.py`; weights 0.3 for
each shape column and 0.01875 for each J_φ column, `--aux_w 1.0` in units of the scalar weight; NaN targets are
masked). The targets come from a CPU cache built from the li(2) cache's contours and the `diii_d_train` labels of the
frozen all-data split (`train.py --stage auxcache`: `<work>/fec_work_idx/mtask/aux_all`, 1,559,340 rows; TRAIN-only
envelope, no scorer mask). The li2ft fine-tune (`d3d_li2ftA22mt_*`, same lines as `final_A22li`) loads the base's
trained head and keeps its loss on (`fec/train_d3d_bx.py` fine-tune extension; with the heads off it is the
`final_A22li` fine-tune). The heads are discarded at inference: the exported `model.pt` holds the same network as
every other member, so inference, the decoder basis and the shipped decoders are unchanged. Group size 4 in job order
gives the production stacks. The bases are only fine-tune starting points and are not in `weights_manifest.json`.

**Member-mean κ finishing (`final_MTfull_mmk`, `final_A22li_dec_mmk`; `fec/mmk.py`, frozen operator `fec/mmk_op.py`).** A label-free
post-process of the ensemble map: it trains nothing and reads no label. While the ensemble members are predicted
(`fec/run_final.py` stage `d3d_b`), the elongation κ of each of the 22 members' own predicted psi is computed per frame
with the organisers' LCFS extraction and shape read-out (`fusion_scoring`), inside the TRAIN-only envelope
`frozen/d3d/env_train.npz` (no scorer mask); no member map is stored. After the off-span decoder, stage `d3d_mmk`
moves κ of the ensemble map half way (α = 0.5) toward the equal mean of the members' κ. The edit is a radial
compression or expansion about the magnetic axis, with weight 1 inside the separatrix tapering to 0 at ψ_N = 1.4 and
localised in Z by Gaussians (σ = 0.25 m) at the heights of the inboard and outboard extremes; the factor is found per
frame by bisection on [0.9, 1.1], and κ of the written float32 map is re-checked. A frame stays bit-identical when no
factor in that bracket reaches the target, when the realised move has the wrong sign, when the envelope extraction
fails, when fewer than 18 of the 22 members' κ could be read for the frame, or when the ensemble's no-Thomson fallback
wrote the frame (the members that set the target do not form the map there). q95 and βN are copied byte for byte, and
lirep4 then runs on the result. The operator file is pinned by SHA-256, and its constants and the envelope's SHA-256
are checked against the config before the stage runs. Any error in the member-κ part of `d3d_b` (an import or setup
error, a worker that dies, no finished worker task for 600 s, a close or save error) leaves the ensemble files as they
are and writes no κ file. The κ file records the `d3d_b` run id, the SHA-256 of that stage's inputs and of the ensemble
it wrote; `d3d_mmk` checks them against the current run. If that check, a step, a worker (the same 600 s limit) or an
output check fails, `d3d_mmk` falls back to the decoder output (the chain of `final_MTfull` or `final_A22li_dec`), says so in its manifest, and
the submission manifest records `d3d_fallback` `mmk_src`. Each frame is edited from its own map and its own target
only, the same way on every split, and the output does not depend on the worker count. How it was gated:
`PROVENANCE.md` (DIII-D, post-hoc transforms).

W3k44 (all 16 + 28 W3k members) was a Dev-phase candidate. It is not shipped: it exceeds the 24 A100-h
rebuild limit (see the budget table).

## Provision of this package

Under the organisers' Final rules (Final guide, 2026-10-19) a Final submission carries no Code URL. After the Final
closes, the organisers request a reproduction package from the top five entries per challenge, which then have 7 days
to provide it. This package, at its pinned commit, is what we would provide; each of our Final submissions was made
with one config in `configs/`.

## Full rebuild (official target: Linux x86-64, one 80 GB A100-class GPU)

```bash
bash reproduce.sh final_A22li full
```

This command is the Challenge-1 anchor rebuild. `final_A24`, `final_B` and `final_C` are separate
candidates; do not substitute one of them when reproducing `final_A22li` (run
`bash reproduce.sh <config> full` with the config the submission was made with).

* The official full-rebuild target is Linux x86-64 on one 80 GB A100-class NVIDIA GPU. Windows is
  supported for inference and smoke runs only.
* `reproduce.sh <config> full` runs, in order:
  1. `environment/install_env.sh` (see Environment; `FEC_SKIP_INSTALL=1` keeps your environment);
  2. `train.py --stage preflight`: free disk space under `$WORK` (see Disk space);
  3. **downloads (network):** `train.py --stage inputs` fetches the inputs of the fold to predict
     (`SPLIT`, default `private_test`; first, so a missing dataset shows at once), then
     `--stage download` fetches the training data at the pinned dataset revision;
  4. `HF_HUB_OFFLINE=1` is set, then `train.py --stage fingerprint` writes and prints
     `$WORK/run_fingerprint.json` (Python, torch, `torch.version.cuda`, NVIDIA driver, GPU name and
     memory, package and source commits, SHA-256 of the config files and manifests, dataset and
     weights revisions, and the offline setting the run uses);
  5. **offline (`HF_HUB_OFFLINE=1`):** `train.py --stage data`, `d3d`, `mast`, `export`, then
     `train.py --stage inputs --local-only` (the `SPLIT` inputs must be complete locally), then
     `infer.py --no-weight-hash` on `$WORK/weights_out` and `validate_submission.py`. A hidden
     network call in these steps fails instead of passing unnoticed.
* A failed or incomplete inputs download never costs the training. In `full` mode a failed early
  inputs download prints a warning and training continues. Before the inference, missing or
  incomplete inputs are downloaded again (pinned or current commit, the normal retry policy) with
  `HF_HUB_OFFLINE` lifted for that one step only. Only if that also fails does the run stop, with
  the two commands that finish it from `$WORK/weights_out`. In inference mode a failed inputs
  download stops the run at once.
* The Final inputs (`private_test`) are the public dataset `ganatma/fusion-equilibrium-challenge-final`, downloaded at
  the revision pinned in `data_manifest.json` (`965040c12b814f33c8e7209f6b1799932247e793`).
  `SPLIT=public_test bash reproduce.sh <config> full` rehearses on the Development fold.
* Hugging Face: `HF_TOKEN` is optional. A read token raises the anonymous rate limit (HTTP 429;
  without one, our first timed rebuild was rate limited after 979 of 7,041 training files).
  No step starts an interactive login; a token cached by an earlier `hf auth login` is still
  used. `reproduce.sh` sets `HF_HUB_DISABLE_XET=1` (plain HTTP transfers) unless you set it.
* Caches stay inside the work tree: `HF_HOME`, `HF_HUB_CACHE`, `TORCHINDUCTOR_CACHE_DIR`,
  `TRITON_CACHE_DIR` and `TMPDIR` point to `$WORK/cache/…`.
* `reproduce.sh <config> full` runs `infer.py --no-weight-hash` on `<work>/weights_out`.
  Retrained files need not match the SHA-256 of the shipped weights in `weights_manifest.json`
  (retrained DIII-D members do not; the retrained MAST network matched it bit for bit on two A100 pods, "Expected
  rebuild variance"); their own hashes are in `weights_out/sha256.json`.

## Quick start (inference with the shipped weights, one GPU)

```bash
bash reproduce.sh final_A22li          # weights + inputs download, then offline inference and validation
```

or step by step:

```bash
bash environment/install_env.sh        # fresh venv, Python 3.13 (see Environment)
export HF_HUB_DISABLE_XET=1            # plain HTTP transfers (fec/final_data.py also sets it)
# optional: export HF_TOKEN=<a read-only token>   (raises the anonymous rate limit; no login needed)
python -B train.py --config final_A22li --stage inputs --split private_test   # the fold's inputs
# weights: the repository + revision in weights_manifest.json (every file is SHA-256 checked)
python -B infer.py --config final_A22li --split private_test --weights ./weights --work ./work
python -B validate_submission.py --split private_test --work ./work
# -> work/fec_work/final_private_test/submission/{diii_d_private_test,mast_private_test}.npz
```

`fec/final_data.py` downloads the training configs (`train.py --stage download` / `data`) and the
Final inputs (`--stage inputs`, or `infer.py` when they are missing). It sets
`HF_HUB_DISABLE_XET=1` itself unless you set it, and it uses `HF_TOKEN` or an earlier login when
one exists (it never prints the token). It retries HTTP 429 and 5xx errors with exponential
backoff: `FEC_HF_RETRIES` retries (default 10), first wait `FEC_HF_BACKOFF_S` seconds (default 30),
doubling up to 900 s. An interrupted download resumes where it stopped, and a download counts as
complete only when every listed file has arrived. `--device cpu` runs everything on the CPU
through `cpu_shim/`; it is slow and meant for smoke and portability runs
(`python tests/smoke_cpu.py ...` runs a 3-shot fold).

**Dataset revisions.** The public dataset `Sophelio/fusion-equilibrium-challenge` and the Final dataset
`ganatma/fusion-equilibrium-challenge-final` are listed and downloaded at the commits pinned in `data_manifest.json`
(`datasets`). A mirror named by the env variable `FEC_HF_DATASET` has no pin and is resolved to its current commit. Every downloaded config's commit, file list and
count are recorded in `work/fec_work/hf_data/revisions.json` and copied into the run fingerprint.
A pinned config already recorded complete is not downloaded again.

### Checks after a run

* `work/fec_work/final_<split>/manifest/*.json`: every stage must have `status` ok. No stage may
  record a `fallback`; the q95 hybrid, the off-span decoder, the κ finishing (`d3d_mmk`) and lirep4 fall back to
  their inputs on failure, so check `info.fallback` is null.  `d3d_mmk` also records its per-reason counts of edited
  and unedited frames (`status_counts`, `fallback_frames_not_edited`, `too_few_members_frames`); an mmk fallback
  also shows in the submission manifest as `d3d_fallback` `mmk_src`.
* The `ipguard` stage re-checks the DIII-D Ip record classes on the private inputs and stops on
  any failure.
* The HF stream-order check (`order`) is opt-in (`--with-order`; Windows `venv_hf` layout).
* `work/run_fingerprint.json`: the environment and revisions of the run.

### Inference: worker processes, hardware and wall clock

* Our Final submission files are produced by this package's inference entrypoint (`bash reproduce.sh <config>`,
  inference mode) at its pinned commit on RunPod GPU pods, one per slot; each build's GPU and CPU model is recorded in its
  `run_fingerprint.json` (GPU) and build log `env.txt` (GPU and CPU). The measurements below are of earlier runs of
  the same entrypoint.
* There is no CPU, memory or wall-clock limit at inference. `infer.py --workers N` (default 6) is the number of CPU
  worker processes that `fec/run_final.py` gives its pooled CPU stages: the input guards, extraction, passive currents,
  q95 hybrid, κ finishing (`final_MTfull_mmk`, `final_A22li_dec_mmk`: the member-κ pool inside `d3d_b` and the
  `d3d_mmk` stage), li repair
  and the MAST betaN and physics stages, including the physics stage's stage-2 scalar pool (a few inner pools are
  capped at 4 or 6). `WORKERS=N bash reproduce.sh <config>` passes it on. Unset, every command is
  the one of our runs.
* Measured on the Development fold (`public_test`) with the shipped weights and the default 6 workers:
  * our GPU (Windows 11, NVIDIA RTX 5060 Ti 16 GB, shared; AMD x86-64 CPU with 12 logical processors),
    `final_A22li_dec`, 2026-10-16: 4.71 h wall, including 29 min to download the inputs and 70 GB of weights. Of it
    `mast_physics` took 3.38 h, `d3d_b` 15 min, `d3d_lirep4` 15 min, `d3d_q95h` 11 min, `mast_amort` 3.3 min, and every
    other stage under 1.5 min;
  * RunPod A100 pods (A100-SXM4-80GB at 400 W or A100 80GB PCIe at 300 W; Linux; 8-CPU quota), `final_A22li_dec`,
    `final_A24r2`, `final_MTfull` and `final_MTli`, 2026-10-16: 119–121 min of inference per config, of which
    `mast_physics` 82–86 min, plus about 2 min for the inputs and 4–14 min for the weights.
* `mast_physics` dominates the wall clock, and on the pods its time was set by the CPU quota, not by the GPU.
* `final_MTfull_mmk` and `final_A22li_dec_mmk` add CPU work at inference to the runs of `final_MTfull` and
  `final_A22li_dec` (the κ finishing): one κ evaluation per member and frame in the member-κ pool inside `d3d_b` (22 per
  frame; the GPU is idle while it runs) and the κ step in the CPU stage `d3d_mmk` (about 20 evaluations per frame).
  Both pools have `--workers` processes (`infer.py --workers N`, `WORKERS=N bash reproduce.sh <config>`; default 6),
  and more workers shorten them: on a 196,301-frame fold with 6 workers on an 8-CPU pod, the member-κ pool added about
  29 min to `d3d_b` (41 min against 12 min without it, on a pod with the same CPU model) and `d3d_mmk` took 27 min; at
  16 workers the member-κ pool would take about 11 min (estimate). Measured on our shared Windows machine on 53,012
  internal-VAL frames with 8 workers: 8 members' κ in 436 s (about 8 ms per evaluation and worker) and the κ step in
  900–1,103 s (about 0.14–0.17 CPU-s per frame). On a 196,301-frame fold that is about
  19 CPU-h in all, i.e. about 3.2 h of wall clock at the default 6 workers; `WORKERS`
  shortens it on hosts with more cores.

## From-scratch training (`train.py`)

```bash
python -B train.py --config final_A22li --stage download  # training data at the pinned revision (network)
python -B train.py --config final_A22li --stage data      # extract (idx Ip axis), dataset, bnd_b (CPU)
python -B train.py --config final_A22li --stage d3d       # li2 cache, then the DIII-D members (GPU)
python -B train.py --config final_A22li --stage mast      # passive at f, synthetic v7h equilibria, MAST net
python -B train.py --config final_A22li --stage export    # weights_out/ + SHA-256 list
```

* Use a fresh `--work` (`reproduce.sh`: `WORK`) for each config. The fingerprint step warns when
  a work dir was used before with another config, package commit or config hash.
* The trainer never overwrites a model directory, and that includes shipped weights that
  `infer.py` installed in the same work dir.
* Each config runs only the stages its outputs read. `--stage mast` (MAST synthetic set and
  amortized net) runs only when the config's MAST maps use that net; `final_C` takes its maps,
  q95 and βN from the physics average, so it skips this stage, as its budget row assumes.
  `--stage li2cache` runs only for `final_A22li*`, `final_MTfull`, `final_MTfull_mmk` and `final_MTli`, and the TX1s
  training only for vb/combo configs. The off-span decoders (`offdec`) train only for `final_A22li_dec`,
  `final_A22li_dec_mmk`, `final_MTfull`, `final_MTfull_mmk` and `final_MTli`, at the end of `--stage d3d`. `--stage auxcache` (the MT aux targets) runs only
  for `final_MTfull`, `final_MTfull_mmk` and `final_MTli`, inside `--stage d3d` after the li(2) cache.
  `final_MTfull_mmk` and `final_A22li_dec_mmk` train exactly the members and decoders of `final_MTfull` and
  `final_A22li_dec`; the κ finishing trains nothing.

### Resume and completeness rules

Every stage can be re-run after an interruption without manual repair:

* **DIII-D groups (`--stage d3d`).** A group whose members all finished (`model.pt` +
  `summary.json`, plus `finetune.json` for fine-tune members) is skipped. A partial group (a crash
  inside it) is moved aside automatically: all of that group's non-empty member directories go to
  `<models>/_partial_<group>_<UTC>/` (nothing is deleted or overwritten; finished groups are never
  touched), one log line names them, and the whole group retrains. A group cannot resume
  member by member: its stacks are trained jointly. Cost of a group retrain at the conservative
  A100-PCIe rate (budget table): `w3k22_F1` (11 shape members) ≈ 7.7 h, `w3k22_F2a` (6
  no-Thomson members) ≈ 4.3 h, `w3k22_F2b` (5) ≈ 3.6 h, each old no-Thomson group (14 members) ≈
  1.5 h, the li2ft fine-tune group (22 members) ≈ 1.9 h; for `final_A24`, `w3k24_F1` (12) ≈ 8.4 h
  and `w3k24_F2a`/`F2b` (6 each) ≈ 4.3 h each; for `final_B`/`final_C`, `w3k_F1` (16) ≈ 11 h. The
  measured budget is for an uninterrupted run: a crash late in a large group can push a rebuild
  past 24 h.
* **li(2) cache (`final_A22li`).** An existing cache (`meta.json`, written last) is reused only
  after `fec/li2_torch.py check` passes on it. A cache that fails the check is moved to
  `cache_all_bad_<UTC>` and rebuilt (CPU, ≈ 25 min on 8 cores).
* **MT aux targets (`final_MTfull`, `final_MTli`).** The same rule with `fec/mtask_aux.py check` (300 random rows re-derived):
  a failing cache moves to `aux_all_bad_<UTC>` and is rebuilt from the li(2) cache (CPU, minutes on 8 cores). The
  MT groups (`mt22_F1`, `mt22_F2`: 11 bases each; `mtftA22_F1`, `mtftA22_F2`: 11 fine-tunes each) resume as above;
  a base-group retrain costs ≈ 7.7 / 7.9 h at the conservative A100-PCIe rate, a fine-tune group ≈ 0.95 h.
  `final_MTli` trains `mt22_F1` and `mtftA22_F1` with `final_A22li`'s `w3k22_F2a` / `w3k22_F2b` and `li2ftA22_F2` (the
  11 no-Thomson lines of the li2ft group, ≈ 0.95 h).
* **MAST synthetic set (`--stage mast`).** The set counts as complete only when it holds exactly
  the chunk files listed in `data_manifest.json` (`synthetic`) and every chunk opens as an npz.
  Unreadable chunks (e.g. truncated by a killed generator) are moved to
  `amort/_bad_gen_v7h_f0.55_<UTC>/`, the generator fills the gaps (it keeps the existing chunks),
  and the set is checked again; a set that is still incomplete, or that holds unlisted chunk files,
  stops the run. `--synthetic DIR` installs the SHA-256-pinned copy instead: the training set of the
  Development-phase MAST network, which in our A100 test did not give the shipped network ("Expected rebuild
  variance").
* **Off-span decoders (`final_A22li_dec`, `--stage d3d` / `offdec`).** A decoder file trained in the
  current members' basis (its recorded `basis_sha256`) is kept; any other decoder file is moved to
  `<dir>/_stale_<UTC>/` and retrained. The CPU cache (`<FEC_IDX_WORK>/offdec_cache/<name>/`) is rebuilt when
  its `data_meta.json` (written last) is missing or names another basis; its 6.6 GB `tr_r.npy` is deleted once
  all three decoders exist (`--keep-offdec-cache` keeps it). Seconds per step: `logs/timing_offdec.json`.
* **Export (`--stage export`).** It stops before copying anything when a weight file the config
  needs (its DIII-D members and fallback, the MAST net, TX1s for vb/combo, the decoders for
  `final_A22li_dec`, `final_MTfull` and `final_MTli`) is missing, and lists every missing path.
* **Links.** A broken or stale work-tree link (e.g. after the work dir was moved) is recreated.
* **MAST input caches for public_test inference.** On `--split public_test`, `infer.py` first builds
  any missing canonical `mast_public_test` caches (shots, ds, C1 passive frames) with the MAST commands
  of `train.py --stage data`, so inference also runs in an empty work dir. `private_test` needs
  nothing extra: its own data stages write those caches.

### Disk space

`train.py --stage preflight` (and `--stage all`) checks the free space under the work dir. Until
the measured peak of a full rebuild is set in `train.py` (`PEAK_DISK_GB`), it only prints a
warning. Once it is set, the run stops unless the free space plus what the work dir already holds
is at least 1.25 × that peak. The large items: the training data (≈ 98 GB of parquet), its
extracted caches, the trained models and the synthetic set (2.5 GB). `final_A22li_dec` and `final_A22li_dec_mmk` need
1 GB more
(`PEAK_DISK_EXTRA_GB`): its decoder stage peaks at +7.1 GB before the export peak and leaves 0.7 GB.
`final_MTfull`, `final_MTfull_mmk` and `final_MTli` also reserve 1 GB more: the same decoder stage plus the 0.14 GB
aux cache; their exports hold fewer files (the MT bases are not exported).

## Environment

`environment/install_env.sh` installs the environment; `reproduce.sh` runs it. In a fresh Python
3.13 venv on Linux x86-64 it runs

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r environment/requirements.txt     # torch==2.13.0 is then already satisfied
```

and on other systems only the second line. torch 2.13.0+cu126 needs an NVIDIA driver
≥ 525.60.13 (CUDA 12 minor-version compatibility). Pip's default Linux wheel of torch 2.13.0 is
+cu130, which needs a driver ≥ 580; `FEC_TORCH_INDEX=https://download.pytorch.org/whl/cu130`
selects it. `environment/requirements.txt` pins the versions of our inference machine;
`environment/environment.json` records the build machine, and `environment/requirements.lock.txt`
is the full `pip freeze` of that machine.

| use | Python | torch | GPU, driver |
|---|---|---|---|
| training of the shipped weights, except the MAST network | 3.11 (RunPod image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) | 2.4.0+cu124 | H100 80GB SXM |
| training of the shipped MAST network `n7h_f0.55` (`train.py --stage mast` in the timed clean-clone rebuild of `final_MTfull`; the same bytes again on a second pod) | 3.11.10 (venv as `environment/install_env.sh` sets it up) | 2.13.0+cu126 | A100-SXM4-80GB, driver 595.71.05; second pod: A100 80GB PCIe, driver 580.159.04 |
| inference that produced our Development-phase submitted files | 3.13.15 (Windows 11) | 2.13.0+cu132 | RTX 5060 Ti 16 GB |
| inference that produces our Final submission files (`bash reproduce.sh <config>`, inference mode, at the pinned commit) | 3.11 (RunPod image Python, venv) | 2.13.0+cu126 | RunPod GPU pod, one per slot; GPU and CPU model recorded per build (`run_fingerprint.json`, `env.txt`) |
| from-scratch rebuild test, 2026-10-13 (`pip install -r environment/requirements.txt` in a fresh venv) | 3.13.16 (Linux) | 2.13.0+cu130 | A100-SXM4-80GB, driver 580.159.04 |
| **rebuild environment of this package** (`environment/install_env.sh`) | 3.13 (Linux x86-64) | 2.13.0+cu126 | A100-class 80 GB, driver ≥ 525.60.13 |

Shipped-weight inference test (final_A22li, public_test, one A100-SXM4-80GB; run with the Development-phase MAST
network, which this package no longer ships, "Expected rebuild variance"):

* torch 2.13.0+cu126 and torch 2.13.0+cu130 write byte-identical submission files (DIII-D SHA-256 `30d39fdc…`,
  MAST `8b7e7a38…`; every array equal, the same 10 li-repair rows).
* Against our Development-phase submitted files, made on the Windows RTX 5060 Ti machine, the A100 files agree to
  float32 rounding:
  DIII-D ψ RMS 2.60e-8 (max |Δ| 1.57e-5) and MAST ψ RMS 1.62e-9. The MAST q95 and βN differ by RMS 2.7e-3 and
  8.0e-4: the contour and Grad-Shafranov steps of the physics-average scalar path (avg_f05) amplify the map-level
  rounding at a few frames. The DIII-D li-repair rows are the same 10.

## Compute: measured hours and the 24 GPU-h budget

### Measured hours

All GPU numbers were measured. The A100 factor k = A100 time / H100 time per member-step was
measured on 2026-10-12 with the production trainer (W3kE120, stacks of 4, compile, 300 steps):

| GPU | shape stack, ms/step | no-Thomson stack, ms/step | k (shape / no-Thomson) |
|---|---|---|---|
| H100 SXM 80GB (training pods) | 114.6 | 113.3 | 1 |
| A100-SXM4-80GB | 189.7 | 188.9 | 1.65 / 1.67 (nominal) |
| A100 80GB PCIe | 222.0 | 222.2 | 1.93 / 1.96 (conservative) |

Hours measured on our shared RTX 5060 Ti convert through the measured generator ratio:
5060 Ti-shared / H100 = 6,043 s / 829 s = 7.3, so 5060 Ti-shared / A100 = 7.3 / k.

| component | hardware, wall clock (measured) | A100-h, PCIe k |
|---|---|---|
| DIII-D data: download, extract (idx), dataset, `bnd_b` | CPU: 1,393–2,017 s on A100 pods (download 530–1,502 s) | CPU (declared) |
| W3kE120 shape member (F1), all data, 45,600 steps | H100: 1,307 s per member (20,911 s / 16); A100 PCIe: 10,124 s per stack of 4 | 0.70 per member |
| W3kE120 no-Thomson member (F2), all data | H100: 1,315 s per member (36,834 s / 28); A100 PCIe: 10,131 s per stack of 4 | 0.72 per member |
| width-1536 / 60-epoch no-Thomson member (old F2; A's fallback) | H100 batched: 22,800 steps x 9.0 ms = 205 s per member | 0.11 per member |
| li(2) target cache for the li2ft fine-tune (`final_A22li`; `train.py --stage li2cache`) | CPU: 1,494 s on an 8-core pod (1,559,340 frames) | CPU (declared) |
| li2ft fine-tune of the 22 W3k22 members (`final_A22li`), all data, 12 epochs | H100: 3,466 s for the whole run (6 stacks of 4/4/3 per family x 4,560 steps; 129 ms per stack-of-4 step, 97 ms per stack-of-3) | 1.89 (whole run at k 1.963) |
| PCA basis, input/target statistics | inside each run (every labelled `diii_d_train` row, train+val+test, under `--alldata`; seed 0): 27–30 s on A100 | included |
| MAST C1 passive currents at f = 0.55 (r = 0.5, 1, 2) | CPU 317 s | CPU |
| MAST synthetic equilibria v7h f0.55 (67k truths, public inputs) | H100: 829 s. Or ship the SHA-256-pinned cache, as the rules allow (it is the Development-phase network's training set and did not give the shipped network in our A100 test: "Expected rebuild variance") | 0.45 (0 if shipped) |
| MAST amortized net n7h_f0.55 | RTX 5060 Ti (shared): 376 s (the Development-phase network); A100-SXM4: 207 s (the shipped network, in the timed clean-clone rebuild of `final_MTfull`) | 0.03 (from the 5060 Ti time; 0.06 measured on the A100-SXM4) |
| Inference DIII-D, 44 members (per fold) | RTX 5060 Ti: 12 min uncontended | 0.05 |
| Inference MAST maps + 30 ms smoothing | RTX 5060 Ti: 2.5 min | 0.01 |
| TX1 -> TX1s training (`final_A22li_vb` / `_combo` only, shipped by `final_pkg.py build --with-tx1s`; `fec/final_tx1s.py train`): TX1 12 epochs on DIII-D TRAIN, then 4 epochs with the own-solver synthetic MAST curriculum | RTX 5060 Ti (shared): 2,883 s + 606 s (the Dev-read runs); data steps CPU ≈ 1.6 h (synthetic curriculum 3,604 s, DIII-D TRAIN Jhat/tokens 1,762 s) | 0.26 |
| Inference MAST map variants (`fec/final_mast.py` amort --rscale / blend / combo, `fec/final_tx1s.py infer`) | RTX 5060 Ti: r05 76 s, VB +110 s GPU (+ 50 s CPU), combo +225 s GPU in total | ≤ 0.02 |
| Off-span decoder stage (`final_A22li_dec` only; `train.py --stage d3d` runs it after the members, `--stage offdec` alone): member-basis check of 50 `model.pt` 3 s, envelope 62 s, data 96 s (CPU, 8 processes, 781,427 frames), then 3 seeds x 40 epochs on 781,326 frames, 235 s each | A100-SXM (2026-10-14, pod v3m0rwtwus7kh0): 867 s for the whole stage, of which GPU training 707 s; peak VRAM 7.2 GB allocated (7.98 GB used); RTX 5060 Ti (shared): 1,155-1,213 s per seed | 0.28 (whole stage at k 1.17; 0.30 at k 1.236) |
| Inference: off-span decoders (`d3d_dec`, 193k frames, incl. the 50-member basis check) | A100-SXM: 30 s; RTX 5060 Ti: 107 s | 0.01 |
| MT aux target cache (`final_MTfull` only; `train.py --stage auxcache`, `fec/mtask_aux_all.py`): 20 targets of 1,559,340 labelled frames from the li(2) cache | CPU: 418 s with 4 processes on our shared Windows box (2026-10-15; byte-identical to the cache the MT members were trained with, aux md5 93fe30ea); 8-core pod: measured in the timed rebuild | CPU (declared) |
| MT bases (`final_MTfull`): the W3kE120 F1 / F2 members + the aux heads (same steps) | H100, same-pod 100-step benches, aux / plain ms per step (stacks of 4): F1 1.0024, F2 1.0027 (mtprod / mtf2 pods, 2026-10-14) | 0.70 × 1.0024 / 0.72 × 1.0027 per member |
| MT li2ft fine-tune (`final_MTfull`): the 22 fine-tunes with the heads on, 12 epochs | H100, same-pod benches, aux / plain: F1 1.0035, F2 1.0029 (whole-run ms per step over the plain bench: F1 1.012, F2 1.013) | 1.89 × (½ · 1.0035 + ½ · 1.0029) |
| Inference MAST physics avg_f05 (4 stage-1 members) | RTX 5060 Ti (shared): 17,704 s | 1.32 |
| q95 hybrid, lirep4, stage 2, betaN, merge, average, composite | CPU 1–2 h, GPU < 2 min | CPU |
| κ finishing (`final_MTfull_mmk`, `final_A22li_dec_mmk`): the 22 members' κ in `d3d_b` (the GPU is idle while it runs), the κ step in `d3d_mmk` | CPU work at inference, outside the GPU-h cap (organisers' Final guide: CPU at inference is uncapped, the worker count a parameter: `infer.py --workers N`, `WORKERS=N` for `reproduce.sh`, default 6): about 56 min of wall clock at 6 workers on an 8-CPU pod for a 196,301-frame fold (up to about 19 CPU-h on our slower shared Windows machine; README, "Inference: worker processes") | CPU |

### Totals per candidate

A100-h, training + inference, at the nominal SXM k and the conservative PCIe k:

| candidate | training | + inference (SXM / PCIe) | fits 24 A100-h? | expected VAL ΔS vs W3k44 (B15) |
|---|---|---|---|---|
| W3k44 (16 + 28 W3k, old-28 fallback; Dev candidate, **not shipped**) | 29.7 / 34.9 | 30.9 / 36.3 | **no** (+7 to +12 h over) | 0 |
| `final_B` = W3e44 (16 W3k + 28 old) | 12.7 / 14.8 | 13.8 / 16.2 | yes | −0.0001 (B12: old F2 vs W3k F2 in C2, +0.000096 for W3k) |
| `final_C` (as B; MAST without net or synthetic data) | 12.3 / 14.3 | 13.5 / 15.7 | yes | as B (DIII-D) |
| 8 + 8 W3k + old-28 fallback | 12.7 / 15.0 | 13.9 / 16.4 | yes | −1.5e-5 to −3.1e-5 |
| `final_A24` = **12 + 12 W3k + old-28 fallback** (`fec/recipes_final/final_ip_C2_w3k24.json`) | 17.6 / 20.6 | 18.8 / **22.0** | **yes**, 2.0 h margin at PCIe k | −0.7e-5 to −1.4e-5 |
| `final_A22li` = **W3k22 + li2ft arm A** (W3k24 minus shapeL_s83 / notsL_s127, each fine-tuned with the li(2) loss; old-28 fallback; `fec/recipes_final/final_ip_C2_w3k22liA.json`) | — / 21.1 (W3k22 base 19.2 + fine-tune 1.89) | — / **22.49** (measured fine-tune; SXM not measured) | **yes**, 1.5 h margin at PCIe k | not measurable on VAL (all-data members); the pre-declared 4 + 4 TRAIN-only gate gave +0.000037 for the fine-tune (S 0.996914 → 0.996951) |
| `final_A22li_vb` / `final_A22li_combo` = `final_A22li` + the VB / combo MAST maps (TX1s network; Oct 14 reads; `--with-tx1s` packages only) | — / 21.4 (+ TX1 -> TX1s 0.26) | — / **22.77** | **yes**, 1.2 h margin at PCIe k (below the 23.0 trigger of the pre-declaration) | as `final_A22li` (DIII-D) |
| `final_A22li_r05` = `final_A22li` + the r = 0.5 passive-scale MAST maps (no new training) | — / 21.1 | — / 22.49 | **yes** | as `final_A22li` (DIII-D) |
| `final_A22li_r2` = `final_A22li` + the r = 2 passive-scale MAST maps (no new training; at inference the r = 2 amortized node replaces the production one, as r05: row "Inference MAST map variants", the C1 currents at r = 2 are CPU) | — / 21.1 | — / 22.49 | **yes** | as `final_A22li` (DIII-D) |
| `final_A22li_dec` = `final_A22li_r2` + the off-span decoder stage (3 decoders on all labelled shots in the members' basis; applied between q95 G1 and lirep4) | — / 21.4 (21.1 + 0.28) | — / **22.78** (22.49 + 0.28 + 0.01; 22.80 at k 1.236; the r = 2 maps as in `final_A22li_r2`) | **yes**, 1.2 h margin at PCIe k | Dev +0.000077 vs `final_A22li` (987055 vs 986001; DIII-D 0.996748); internal VAL gate (TRAIN-only decoders) +6.1e-5 |
| `final_A24r2` = `final_A24` + the r = 2 passive-scale MAST maps (no new training; the MAST stages are `final_A24`'s: the same synthetic set and amortized net, and at inference the r = 2 amortized node replaces the production one, as `final_A22li_r2`: row "Inference MAST map variants", the C1 currents at r = 2 are CPU) | 17.6 / 20.6 | 18.8 / **22.0** (= `final_A24`) | **yes**, 2.0 h margin at PCIe k | as `final_A24` (DIII-D) |
| `final_MTfull` = `final_A22li_dec` with the MT members (bases with the aux heads, li2ft with the heads on; same old-28 fallback, decoder stage, r = 2 maps) | — / 21.45 (21.4 + 0.05) | — / **22.83** (22.78 + 11 × 0.70 × 0.0024 + 11 × 0.72 × 0.0027 + 1.89 × ½ × (0.0035 + 0.0029), same-pod bench ratios); **23.02** on the conservative reading (whole-run ms per step over the plain bench: F1 1.0144 / 1.0121, F2 1.0130 / 1.0132), above our internal 23.0 bar, below 24 | **yes** at the bench reading, 1.2 h margin at PCIe k; 1.0 h on the conservative reading | Dev +0.000027 vs `final_A22li_dec` (0.996775 vs 0.996748); internal VAL gate (TRAIN-only members, both halves with heads vs a same-argv plain control) +0.000108, P 0.997 |
| `final_MTli` = `final_A22li_dec` with the MT shape (F1) members only (the 11 shape bases with the aux heads, their li2ft with the heads on, = `final_MTfull`'s F1; the no-Thomson members, old-28 fallback, decoder stage and r = 2 maps as `final_A22li_dec`; plus the CPU aux cache) | — / 21.42 (21.4 + 0.02) | — / **22.80** (22.78 + 11 × 0.70 × 0.0024 + 1.89 × ½ × 0.0035, same-pod bench ratios); **22.90** on the conservative reading (whole-run ms per step over the plain bench: F1 1.0144 / 1.0121) | **yes**, 1.2 h margin at PCIe k (1.1 h on the conservative reading) | internal VAL gate of the heads on the shape half (TRAIN-only members vs a same-argv plain control) +0.0000716, P 0.9995 |
| `final_MTfull_mmk` = `final_MTfull` + the κ finishing stage (no new training; CPU work at inference) | — / 21.45 (= `final_MTfull`) | — / **22.83** (= `final_MTfull`; the κ finishing is CPU) | **yes**, as `final_MTfull` | internal VAL gate of the κ finishing on the post-decoder map of `final_MTfull`'s VAL replica (TRAIN-only members and decoders) +0.0000443, P 1.000; +0.0000470 with the no-Thomson fallback frames left unedited, as shipped |
| `final_A22li_dec_mmk` = `final_A22li_dec` + the κ finishing stage (no new training; CPU work at inference) | — / 21.4 (= `final_A22li_dec`) | — / **22.78** (= `final_A22li_dec`; the κ finishing is CPU) | **yes**, as `final_A22li_dec` | internal VAL check of the κ finishing on the post-decoder map of `final_A22li_dec`'s VAL replica (TRAIN-only li2ft members and decoders) +0.0000462 with the no-Thomson fallback frames left unedited, as shipped (+0.0000482 with them edited), P 1.000 |
| 16 + 8 W3k + old-28 fallback | 17.5 / 20.6 | 18.7 / 22.0 | yes, 2.0 h margin | −0.9e-5 to −1.8e-5 |
| 14 + 14 W3k + old-28 fallback | 20.0 / 23.5 | 21.2 / 24.9 | **no** at PCIe k | −0.5e-5 to −0.9e-5 |
| 14 + 14 (or 16 + 12) W3k + 7 old fallback members | 18.0 / 21.1 | 19.2 / 22.5 | yes, 1.5 h margin | −0.5e-5 to −1.0e-5 (fallback shrink: < 2e-7) |

These totals are the ledger forecasts made before the rebuilds. Measured values, which supersede them for
`final_A22li_dec`, `final_MTfull` and `final_MTli`, are in "Timed clean-clone rebuilds (measured)" below.

Notes on the ΔS estimate:

* B15 gives 1 − S = a + b/N per family, b = 0.000408. For the equal-weight two-family C2 mix,
  the excess is (b/4)(1/N1 + 1/N2) if the families' seed noise is independent; the second
  number uses b/2 (conservative).
* The paired VAL SD is ≈ 0.0002 at 234 shots, so every size effect in the table is below one SD.
* A subset of W3k44 needs no new training: the members already exist.


### Timed clean-clone rebuilds (measured)

Each of these Final configs was rebuilt once from a clean clone of its package (`bash reproduce.sh <config> full`,
every member retrained) on one A100-SXM4 pod, and timed. The GPU hours (wall clock of the GPU stages) are measured;
the A100 80GB PCIe hours are conversions with the two factors k below. Each package ships only some of these
configs.

| config | package commit rebuilt | A100-SXM4 GPU h (measured) | PCIe h at k 1.170 | PCIe h at k 1.236 (pessimistic) |
|---|---|---|---|---|
| `final_A22li_dec` | 30a9be2 | 20.203 | 23.64 | 24.97 |
| `final_MTfull` | f155e97 | 19.61 | 22.949 | 24.24 |
| `final_MTli` | a85e3ac | 19.982 | 23.385 | 24.698 |

k 1.170 = 222.0 / 189.7, the PCIe / SXM step-time ratio of the production trainer's shape stack in "Measured hours"
above, applied to every GPU stage. k 1.236 is our pessimistic factor, 222.0 / 179.6: the same PCIe step time over
the SXM step time measured in our first timed rebuild, which ran a newer torch (2.13 against the bench's 2.4), for
the case that this speed-up on SXM does not carry over to PCIe. The three readings above 24 h are all at the
pessimistic k 1.236. `final_MTfull_mmk` and `final_A22li_dec_mmk` were not rebuilt separately: their training
(members, decoders, MAST network) is that of `final_MTfull` and `final_A22li_dec`. The κ finishing is CPU work at
inference, which the organisers' Final guide leaves outside the cap (24 GPU-h on one 80 GB A100, including test-input
adaptation and data-generating networks; CPU at inference uncapped, with the worker count a parameter), so the GPU work
of these two configs is that of `final_MTfull` and `final_A22li_dec`. For information only: a wall-clock ledger that
also counts the member-κ CPU time inside `d3d_b` (about 29 min at 6 workers, "Inference: worker processes") gives about
24.2 PCIe-h at k 1.170 for `final_A22li_dec_mmk` and about 23.5 for `final_MTfull_mmk`.

## Determinism

* Seeds:
  * Python, NumPy and torch CPU/CUDA are seeded per member (`torch.manual_seed(seed)`);
  * the data order is a per-member `torch.Generator` stream;
  * there are no DataLoader workers, and shot lists come from `info.json` and sorted globs.
* TF32 is never enabled. bf16 autocast is part of the DIII-D recipe.
* The rebuild uses the same production settings as the original training: the recorded member
  arguments and seeds, bf16 autocast, TF32 off, and no deterministic-algorithms mode.
  `fec/det_run.py run [--strict]` adds cuDNN-deterministic mode, `PYTHONHASHSEED`,
  `CUBLAS_WORKSPACE_CONFIG` and `torch.use_deterministic_algorithms`; strict mode is a diagnostic
  only. It can select other kernels than the production run, and neither the shipped weights nor
  `train.py` use it.
* Shipped-weights inference is deterministic on a given host: two repeated runs on our GPU wrote byte-identical
  files. Our Final submission files are produced by this package's inference entrypoint (`bash reproduce.sh
  <config>`, inference mode) at its pinned commit on RunPod GPU pods, one per slot; the GPU model of each build is
  recorded in its `run_fingerprint.json`, and its GPU and CPU model in the build log `env.txt`. Our
  Development-phase MAST files came from an earlier MAST network that this package does not ship ("Expected rebuild
  variance").
  Inference results differ across GPU and CPU models at float-rounding size: the DIII-D files and the MAST maps match
  to float32 rounding (DIII-D, RTX 5060 Ti against A100: ψ RMS 1.5–1.6e-8 of the range and at most 1e-5 of it, q95 and
  βN within 4e-5 relative; the shipped MAST network's maps, our GPU against an A100, RMS 4.2e-9 and at most 7.5e-8 of
  the ψ range). The MAST bytes also depend on the host CPU from the passive stage onward: of four A100 pods, three
  wrote one MAST file and the fourth another, and neither equalled our GPU's file. The MAST q95 and βN can differ
  more on a few frames (A100: 37 and 52 of the 69,075 public-test frames by more than 0.1 %, at
  most 5 % and 15 %), because the MAST physics stage takes discrete branches (a convergence
  fallback, a root choice, a running median) that rounding can flip.
  In `final_MTfull_mmk` and `final_A22li_dec_mmk` the κ finishing turns that rounding into larger differences on a
  few frames: its factor is found per frame by bisection to a tolerance, and a frame near a skip threshold can be
  edited on one host and not on another. Two builds of `final_MTfull_mmk`, one from an earlier commit with the same κ
  code, on an A40 and on an RTX A5000 pod (same CPU model) differed in DIII-D ψ by an RMS of 1.3e-5 of the RMS ψ, by
  more than 1e-5 of the frame's ψ range on 4,445 of 196,301 frames and by more than 1e-3 on 24 (at most 1.9e-2); q95
  and βN, which the step does not change, stayed within 6e-7 relative.
* Retraining is seeded but not bit-identical across GPU models or library versions. A rebuild is
  judged by its predictions and scores, not by the SHA-256 of the retrained checkpoints.

## Expected rebuild variance

A rebuild uses the SAME recorded member seeds as the submitted models, so seed-to-seed ensemble
variance is not the expected source of differences. Differences come from floating-point and
kernel differences between GPU models and software stacks.

A scale for comparison: removing 2 members from a 24-member DIII-D ensemble changed the
public-test psi predictions by RMS 4.1e-5. Under a model of independent, identically distributed
per-member noise,

    SD(mean24 - mean22) = sigma * sqrt((1/12)^2 * (1/2 + 1/22)) = 0.06155 * sigma,

so sigma ≈ 6.66e-4, and two independently RESEEDED 22-member ensemble means would differ by about

    sigma * sqrt(2/22) ≈ 2.01e-4

in psi RMS. This 2.01e-4 is a scale estimate for independent reseeding. It is not a prediction of
the same-seed rebuild difference, and it is not a tolerance. A rebuild tolerance published by the
organisers supersedes anything stated here.

Measured on Dev: the timed clean-clone rebuilds of `final_A22li_dec`, `final_MTfull` and `final_MTli`
(same recorded seeds, every member retrained) read within about 1e-5 S of the DIII-D files we submitted:
−0.000005 (Dev 991560), +0.000007 (Dev 991548) and +0.000003 (Dev 994963).

The MAST stage is different. Its amortized network is retrained from scratch in a full rebuild,
and that retraining is chaotic (in our tests, changing a single training frame changed the selected
network width), so a MAST rebuild on other hardware or software can give different maps. Our
Development-phase MAST files came from a network trained on our own GPU, which this package's training
entrypoint did not reproduce in our A100 tests: two measured rebuilds read 0.0067 and 0.0090 lower
in MAST S than those files on the Development folds (Dev 986833, the r = 1 map of `final_A22li`;
Dev 991560, the r = 2 map). This package therefore ships the network that the entrypoint produced on the
two A100 pods we tested (`mast/n7h_f0.55`, model.pt SHA-256 `bfbb346c…`), not the Development-phase one:
`train.py --stage mast`, with the synthetic set regenerated (the default), trained it in the timed
clean-clone rebuild of `final_MTfull` on an A100-SXM4-80GB, and the same command gave the same bytes again
on an A100 80GB PCIe with another driver. Other software stacks and hosts are untested for this retraining:
in particular the Python 3.13 environment of `environment/install_env.sh` and the Quick start (both pods used a
Python 3.11.10 venv), the default +cu130 torch wheel, other host CPUs and the organisers' own container image. The
three timed clean-clone rebuilds (`final_A22li_dec`, `final_MTfull`,
`final_MTli`) wrote the same r = 2 MAST file, the 0.0090-lower file above (0.793869 against 0.802845), and the
shipped network reproduces its maps bit for bit on an A100. The SHA-256-pinned synthetic set in the weights
repository (`--synthetic`) is the Development-phase network's training set; in our one test on an A100 it
trained a third network, so a rebuild meant to reproduce the shipped network regenerates the set. The Final
MAST files come from the shipped-weights inference path, "Quick start (inference with the shipped weights,
one GPU)": `bash reproduce.sh <config>`, run at the pinned package commit on RunPod GPU pods, one per slot. That path is
deterministic on a given host; across GPU and CPU models the MAST maps agree to float32 rounding, the MAST bytes depend
on the host CPU from the passive stage onward, and the MAST q95 and βN can change on a few frames ("Determinism").

## Layout

```text
fec/            pipeline code (import closure of the entrypoints); fec/run_final.py drives every stage
configs/        final_*.json (recipes), train_d3d_final_*.json + jobs/ (member arguments)
frozen/         small frozen artefacts (SHA-256 in data_manifest.json)
environment/    install_env.sh, requirements.txt, requirements.lock.txt, environment.json
data_manifest.json, weights_manifest.json, PROVENANCE.md, LICENSE
infer.py, train.py, validate_submission.py, reproduce.sh, cpu_shim/, tests/
```
