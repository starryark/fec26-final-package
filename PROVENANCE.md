# Per-machine data provenance statement

Package built 2026-10-21T06:56:45+00:00 from source commit `908ec3a76e57bd127d6911ad153fe9d3a2a3eebe`. This statement covers every
config in `configs/` (final_B, final_C, final_A24, final_A22li, final_A22li_r05, final_A22li_r2, final_A22li_dec, final_MTfull, final_A24r2, final_MTli, final_MTfull_mmk, final_A22li_dec_mmk, final_MTfull_mmk_nrsl, final_A22li_dec_mmk_nrsl); every Final submission made from this package uses one of them.
Where the configs differ, the bullets below name the configs they apply to. An inaccurate provenance
statement disqualifies by itself, so anything not listed here was **not** used.

## Common to both machines

* **Challenge data.** Hugging Face dataset `Sophelio/fusion-equilibrium-challenge`, revision
  `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb` (2026-07-30, the current `main`): `diii_d_train` (inputs + EFIT labels, the
  only labels we used), `diii_d_public_test` and `mast_public_test` (inputs only). Final
  predictions use `ganatma/fusion-equilibrium-challenge-final`, revision `965040c12b814f33c8e7209f6b1799932247e793`
  (`diii_d_private_test`, `mast_private_test`, inputs only). Apart from published machine geometry (listed per machine
  below), no other dataset, no external equilibrium database and no pretrained weights enter any
  config of this package; public data read only for analysis are listed under "Other public
  information read" below.
* **Inputs at prediction time:** magnetics coil currents, plasma current, Thomson Te/ne, and the
  `efit_times`, `efit_grid_R`, `efit_grid_Z` scaffolding. Never an `efit_*` output,
  `magnetics_dsep(_times)` or a scorer envelope mask.
* **Thomson time axes (starter issue #12).** For the Final inputs we apply the organisers'
  supported rule (`fec/extract_final.py`, the default of `fec/run_final.py` on `private_test`):
  * per Thomson system, a sample is kept only if its time is later than every earlier sample's;
  * the same mask applies to Te and ne;
  * zero or NaN values are missing;
  * there is no t <= 0 filter and no truncation at the first backward step.

  Training and the Dev predictions used `fec/extract.py`, which sorts samples by time. The two
  rules give **identical** inputs on every public shot. diii_d_train has 10 shots with 26 dropped
  samples, all edge-padding rows with t ≈ 0 and no data, and 0 changed frames. diii_d_public_test
  and mast_public_test have no non-monotone samples. The DIII-D input guard
  (`fec/input_fixes2.py`) tolerates the announced cases (`fec/final_data.py issue12`).
* **Post-processing (`fec/final_build.py`):** `float32` psi; DIII-D q95/betaN are clipped to the
  DIII-D TRAIN label range (`fec/final_train_ranges.json`, TRAIN split only); MAST scalars get a
  label-free per-shot repair of our own predictions (missing or implausible frames are replaced
  using the same shot's frames, or the fold median when a shot has none; then a running median),
  and MAST q95 is clipped to [1.5, 20] and betaN to [0, 8].
* **Development-phase feedback.** The public Dev board compared and selected **genuine candidates
  only**: complete, physically motivated pipelines. We made no probe submissions (constant, shifted
  or rescaled predictions).
  * *Rules and their timing.* Most reads followed a rule written before the score. `DEV_LOG.md`
    lists the pre-declared rule of each entry with the commit that first contains it and marks the
    exceptions; the three MAST-only entries (MAST section below) carry no rule text or commit. The
    marked exceptions:
    * the 2026-10-08 feedback policy and the 2026-10-09 comparison set were first committed to our
      private repository after the reads they govern (the log's own timestamps say they were
      written before);
    * three choices were made without a pre-declared bar: the production MAST q95/betaN source, the
      carry-forward of the all44 DIII-D ensemble, and the decision not to carry the denoised-height
      candidate forward after its declared comparison had been met (no reason was written down).
  * *Constants set with board feedback.* No constant was fitted to board numbers except under two
    team-approved exceptions:
    * the solenoid turn fraction f, set to **f = 0.55** (MAST maps of every config except
      `final_C`);
    * from 2026-10-13, single physical machine knobs, read one setting at a time as pre-declared
      candidates. Of these, only the passive-resistance scale r set a constant: it was read at 0.5,
      2 and 4, and **r = 2** was adopted. Two label-free calibrations of the same passive model,
      which `DEV_LOG.md` lists under this exception (one of them also retrained the network, at
      f = 0.50), were read on 2026-10-16 and not adopted (MAST section below). r scales the passive
      currents of the amortized maps only: the MAST section below
      names the configs whose maps use r = 2 and the one whose maps use the non-adopted r = 0.5. All
      other maps, and the physics average behind every config's scalars, use the nominal resistances
      (r = 1) or no passive currents.

    The organisers allow Dev scores to select among submitted candidates, including several
    settings of one knob (forum thread 2827; starter issue #9). The 30 ms width of the MAST maps'
    temporal smoothing was selected on Dev between two candidate widths, 30 ms (adopted against no
    smoothing) and 50 ms (rejected; MAST section below); it is a discrete choice between two
    candidates, not a fitted value.
  * *Readouts.* Some readouts combined our own board R²ψ with label-free distances between our
    maps (λ*, cosine, f*). They decided which pre-built candidate was submitted next, and f* set f.
  * *Detailed Results of our own entries.* Beyond the summary scores, we used the per-term and
    per-scalar R² and the "worst 20 shots" tables of our own entries to locate failure classes,
    such as frames with parked P2 coils. They also sized and motivated hypotheses, including the
    physical-knob and transfer candidates. No constant was fitted to them.
  * `DEV_LOG.md` lists every Dev comparison for both machines, with its scores, including rejected
    candidates; for the three MAST-only entries it gives the scores and the comparison only. The
    decisions that shaped these configs are summarised per machine below.
* **Other public information read (analysis only).**
  * The public Dev leaderboard, including other teams' per-term MAST Detailed Results and the board
    history, was read for analysis. No constant was fitted to it and nothing was submitted from it.
  * On 2026-10-14 one other team's public model weights (DIII-D only) and that team's public code repositories
    were read. Its code repository that contains a MAST EFIT++ neural-equilibrium campaign was read only by a
    separate clean-room agent, which could report only provenance and machine-description facts.
    Our records show nothing adopted from them (family doc amendment 24; the clean-room agent's
    file log). This item covers that team's artefacts read on that date only.
  * Before the Final phase, other teams' public code repositories, model archives, method write-ups and issue
    threads, and third-party summaries of public information about other teams, were also read for analysis. No code
    or weights from them enter any config. Some of that material, and some read during the Final phase, reported
    other teams' scores on the three labelled MAST demo shots; no copy of them was kept, and the values were not used.
  * *Public FAIR-MAST vacuum-shot input signals (analysis only; in no config).* On 2026-10-13 we
    screened MAST shots 20000–30473 by their measured plasma-current signal alone. We then
    downloaded the level-1 coil-current (`amc`) and raw-magnetics (`amb` flux loops and pickup
    coils) input signals of the 156 vacuum shots with coil activity (out of 902 with |Ip| < 10 kA),
    the level-2 `magnetics` channel geometry (channel names, positions and probe lengths) of shot
    30420, and that shot's metadata key listing (names, units and descriptions; no data).
    We used them to calibrate a passive-model variant, and four candidates for this package's MAST
    pipeline were built on it. On 2026-10-13 a frozen-network map and a retrained generator and
    network (at solenoid fractions 0.55 and 0.50) each failed one pre-declared label-free gate (the
    fraction of frames whose predicted boundary lies inside the wall, within 1 cm; bar 0.95) and
    passed the others. After that gate had twice predicted the wrong sign of a passive-scale change
    on Dev, we dropped it for passive-model candidates on 2026-10-14 (a board-informed decision). A
    frozen-network map with per-group passive scales (2026-10-14) then passed every other check; its
    inside-the-wall fraction (0.932) was also below the 0.95 bar. The retrained candidate at 0.50
    and the per-group map were read on Dev (Dev 991548 and Dev 991439, 2026-10-16) and not adopted;
    the 2026-10-13 frozen-network map and the retrained candidate at 0.55 were never submitted.
    On 2026-10-14
    the same downloaded signals and channel geometry, with no new download, were used to build and
    check magnetic-probe response tables of our solver. The calibration and these tables were also
    used by the different MAST pipeline of the three MAST-only Development entries listed under
    "Dev feedback (MAST)", which is not part of this package or of any Final submission. No
    constant, signal or model from this work enters any config of this package. For this study, no
    data of any EFIT++ group (`amm`, `efm`, `epm`), equilibrium or summary group, or EFIT pseudo-coil
    signal was requested (the shot-30420 key listing above covers names only), and no plasma shot's
    signals were downloaded beyond the screening plasma current. Two responses of the archive's shot catalogue, whose items carry EFIT summary
    statistics, were discarded without any value being printed or used.
  * *Public FAIR-MAST internal diagnostics of the matched archive shots (analysis only; in no submission and no
    config).* On 2026-10-15, for candidates of the withdrawn matched-shot line ("Other MAST-only entries" under "Dev
    feedback (MAST)"), we read level-1 signals of the same 1,206 matched archive shots: motional Stark effect
    polarisation angles with their noise, A-coefficients, channel radii and beam flags (`ams`), carbon
    charge-exchange ion temperatures and radii (`act`), neutral-beam powers (`anb`), CO2-interferometer line density
    (`ane`), and the linear Dα camera's outer plasma radius and edge density-gradient position (`ada`, `adg`). Before
    that we read the metadata listings of these groups on 3 and 5 of those shots and on one vacuum shot, and the MSE
    view-geometry and analysis-settings text of one other public shot. No EFIT, equilibrium, flux-coordinate or
    model-derived variable was requested. None of these signals entered any Development submission (Dev 988761,
    988789 and 988995 were built from the matched magnetics and the challenge inputs) or any config of this
    package; the line was withdrawn the same day.
  * *Public FAIR-MAST plasma-shot signals (analysis only; in no config).* On 2026-10-18 we downloaded the level-1
    coil and plasma currents (`amc`), flux loops and pickup coils (`amb`) and core and edge Thomson data (`ayc`,
    `aye`) of 269 archive plasma shots chosen by a shot-number rule: among shots 11695–30473 whose level-1 plasma
    current peaks above 300 kA and that have core Thomson data (a screen, for which we downloaded only each
    candidate's plasma current and core Thomson temperature), every second such shot whose number is divisible by 37
    (60 shots) and every such shot whose number is divisible by 19 but not by 37 (209). No archive shot was joined to any challenge shot (only
    population-level and fixed-hardware comparisons were made), and no EFIT, equilibrium or summary group was
    requested. We used them to test whether the residual of the measured probes against probe signals
    forward-modelled from our MAST maps could rank those maps; it could not, and nothing was adopted.
* **Other Final teams' public material.** From 2026-10-19, other Final teams' public method descriptions,
  papers and code repositories were read. The code, envelope and gates of the κ finishing step
  (`final_MTfull_mmk`, `final_A22li_dec_mmk`; DIII-D section, post-hoc transforms) were written by us, and no
  code or weights were taken from any team.
* **AI assistance.**
  * Code, experiments and analysis were largely carried out by Claude Code agents (Anthropic) under
    the team's direction. Separate Claude Code reviewer agents reviewed pre-declared candidates and
    post-hoc rulings.
  * Some candidates were first proposed by an autonomous fresh-eyes loop: headless Claude Code
    sessions (`claude -p`) that saw the task specification, the public leaderboard and, in the run
    that proposed the off-span decoder, a state-only description of our DIII-D model and that
    model's internal-VAL predictions as a black-box base, but not our research history. The
    off-span decoder (DIII-D section) is one of them.
  * Literature and public-information research used ChatGPT deep research and, at times, GLM.
  * *Signature.* This statement was prepared and signed by the team's AI agent (Claude Code) on the
    entrant's behalf. As challenge rule 4 requires, the entrant takes responsibility for the
    submission.

## DIII-D (Challenge 1)

* **Training data:** `diii_d_train` with a frozen whole-shot split (`frozen/split/`). Model
  selection and the pre-declared gates used the internal VAL shots, except the Dev reads listed
  under "Dev feedback (DIII-D)" below. The internal TEST shots were evaluated **once**
  (2026-10-08). The Final members are then refit on train+val+test (`--alldata`) with unchanged
  recipes.
* **Labels used:** EFIT psi and the five scalars of `diii_d_train` only, for (i) targets
  (PCA coefficients, scalar heads; the Final `--alldata` members fit the PCA basis and the scalar target
  scales, like their input standardisation, on every labelled `diii_d_train` row (train+val+test), not on
  internal TRAIN; only the gate members fit them on internal TRAIN), (ii) TRAIN-only loss-weight envelopes
  and the per-frame LCFS bands of the same rows
  (`fec/bnd_b.py`, `frozen/d3d/env_train.npz`), (iii) the frozen q95-hybrid envelope
  (`fec/q95_hybrid_envelope.json`, TRAIN LCFS), (iv) the clip ranges above.
* **li2ft fine-tune (`final_A22li*`, `final_MTfull`, `final_MTfull_mmk`, `final_MTfull_mmk_nrsl` and `final_MTli`):** its li target, the organisers' li(2) of each true EFIT
  psi map on that map's own LCFS, is derived from the same `diii_d_train` labels of the frozen
  all-data split (`--alldata`) exactly like the base loss, with the TRAIN-only envelope
  `frozen/d3d/env_train.npz` and no scorer mask.
* **Off-span decoder (`final_A22li_dec`, `final_A22li_dec_mmk`, `final_A22li_dec_mmk_nrsl`, `final_MTfull`, `final_MTfull_mmk`, `final_MTfull_mmk_nrsl` and `final_MTli`; `fec/offdec.py`):** trained on the EFIT psi of the
  same all-data rows (train+val+test, every 2nd frame) in the members' own PCA basis; its loss
  weights use the label LCFS, axis and the lowest and highest points of the label LCFS polygon (not
  located X-points) of those frames and an envelope rebuilt from 1,500
  TRAIN shots' label LCFS (SHA-256 checked); no scorer mask. At inference its input is the members'
  predicted psi only. It was proposed by the fresh-eyes loop (AI assistance, above), which chose its
  variant among five decoder arms and two shrink settings of the best arm on one half of the internal
  VAL shots. Our development gate (TRAIN-only decoders scored on internal VAL), pre-declared before
  that loop run (family doc amendment 26), then passed at ΔS +0.0000613 on all VAL shots and
  +0.0000367 on the half not used for the loop's choice; the all-shot decoders were then compared
  once on Dev against `final_A22li` (`DEV_LOG.md`).
* **MT members (`final_MTfull` and `final_MTfull_mmk`, and the shape half of `final_MTli`; `fec/mtask_aux_all.py`, `fec/train_d3d_bx.py --aux_heads`):** the auxiliary
  heads' targets (κ, δ_top, δ_bot and volume of the scorer's LCFS of each true EFIT psi map, and that map's
  flux-surface-averaged J_φ at 16 normalised-flux points) come from the same `diii_d_train` labels of the all-data rows
  (train+val+test, like every other loss of the all-data members; the internal TEST rows enter only as training
  rows (targets and the per-run fits, the aux-target standardisation included), after their one-time evaluation), with the li(2) cache's contours, the TRAIN-only envelope and no scorer
  mask. The heads are training-only; inference never reads them. Development record and disclosures:
  * Gates (TRAIN-only members on internal VAL, pre-declared): the heads on the shape half gave +0.0000716 (P 0.9995);
    on the no-Thomson half +0.0000363 (P 0.9025) under the declared reading, in which each object's own no-Thomson
    members also serve the 588 VAL frames without Thomson data. On the reading closer to production (the separate
    old fallback members on those frames; computed after the verdict, reported as information) the no-Thomson-half
    gain is +0.0000134 (P 0.732): the evidence that the heads help the no-Thomson half in production is weak. Both
    halves together vs a same-argv plain control: +0.000108 (P 0.997); the pre-declared absolute bar (+0.00011) was
    narrowly missed and replaced by a relative rule before any number was read. The all-shot ensemble was then
    compared once on Dev against `final_A22li_dec` (`DEV_LOG.md`).
  * The fine-tune extension (heads kept on during the li2ft fine-tune) was certified with the heads OFF: with the li
    term off too, it is bit-identical to the reference trainer on the GPU; with the li term on, the reference trainer
    is itself not bit-reproducible on the GPU, and the extension agreed within that spread (the calibrated λ differed
    by one float32 unit in the last place; accepted after the fact). The heads-on GPU path is covered by CPU unit tests
    only. Two of the three shape-half production stacks ran their fine-tunes before that check had finished (same
    trainer file); they were kept after it passed.
  * Argv: the MT production members were trained with the MT pods' run-level arguments (`--group_size 4 --pix_impl
    compile --ln_impl compile --vram_gb 80`), not the original arguments of the `final_A22li` members, so the Dev
    difference to `final_A22li_dec` mixes the heads with argv / kernel drift (up to about +0.0003 per member before
    ensemble averaging). The no-Thomson stacks ([s5 s6 s7 s45], [s46 s47 s85 s86], [s87 s125 s126]) differ from the
    original W3k22 pods' grouping; stacked members share kernels, not data or parameters. A rebuild uses this
    package's own arguments (`train.py`: `--vram_gb 76`).
  * Compute: 22.83 A100-PCIe-h on the README basis (same-pod bench ratios); 23.02 on the conservative whole-run reading,
    above our internal 23.0 bar and below the 24 h limit (`README.md` "Compute"). The timed clean-clone rebuild of
    `final_MTfull` then measured 19.61 A100-SXM hours, 22.949 A100-PCIe-h at k 1.170; at the pessimistic factor
    k 1.236 it reads 24.24, above 24 h (`README.md` "Timed clean-clone rebuilds").
* **Ip time axis (deviation from `data_fixes.py`, disclosed).** The starter's fix leaves Ip
  late on the ~1/3 of shots whose magnetics record starts near −9.2 s (against EFIT on TRAIN, by a
  median of 170 ms; 5th percentile 56 ms, 95th percentile 294 ms, maximum 327 ms). We place Ip sample
  0 at `magnetics_time[0] + 8192 ms` on those records. This is an input-only rule, confirmed
  without labels by E-coil breakdown timing and guarded by `fec/ip_timebase.py check`. The
  organisers asked (2026-10-11) that no time-axis correction other than `data_fixes.py` be
  applied. We keep ours because every DIII-D model was trained on it and, against TRAIN labels,
  its residual against the EFIT alignment has a median of +1.7 ms (95th percentile +6.9 ms, maximum
  15.8 ms), where the shipped long-record axis is late by a median of 170 ms. It uses inputs only and is applied identically to the private inputs.
  The organisers' Final guide (2026-10-19) allows label-free processing of the inputs, including time-axis alignment,
  if it uses inputs only, is applied the same way on every split and is described in the methods report; ours meets
  all three conditions. The
  record-class rule it relies on (short records near −4.06 s, long near −9.23 s; the starter's
  re-origin decision equals the record class on every shot) is re-checked on the private inputs
  by the `ipguard` stage before any prediction. A failure stops the run.
* **Post-hoc transforms (label-free at inference; q95 G1 and lirep4 in every config, frozen before the Final; the κ
  finishing in four configs only, added during the Final phase):**
  * q95 G1 hybrid: the physics contour q95 of the predicted psi replaces the network head unless it
    is missing, out of range or more than 3% above the head (a one-sided gate, τ = 0.03). G1 was
    chosen after seeing internal VAL, after the pre-declared two-sided gate had failed (post hoc;
    independently reviewed), and adopted on internal VAL. Its first Dev read was on 2026-10-09
    (0.996331 against 0.996239 for the same ensemble without it, +0.000092);
  * lirep4: li-range repair of the frames whose li(2) of the predicted psi, on a valid LCFS contour, lies
    outside [0.1, 5], whatever the phase (in practice it fires on ramp-down frames). Its diagnostics read the
    organisers' DIII-D scorer envelope for logging only; the outputs do not depend on it.
  * How lirep4 was adopted: on a pre-declared, exact, paired Development read. The repaired and
    unrepaired files differ only on the repaired frames (+0.000049, li term only). A holdout cannot
    resolve a gain confined to about 1–2% of shots, which is why VAL did not gate it.
  * κ finishing (`final_MTfull_mmk`, `final_A22li_dec_mmk` and their NRS-L variants `final_MTfull_mmk_nrsl`,
    `final_A22li_dec_mmk_nrsl` only; `fec/mmk.py`, README): κ of the post-decoder ensemble map is moved half way
    to the equal mean of the 22 members' own κ, every κ read with the organisers' LCFS routine inside the TRAIN-only
    envelope (no scorer mask, no label); q95 and βN are unchanged, and frames written by the no-Thomson fallback are
    not edited. It uses the predicted maps only and is applied the same way on every split.
  * How the κ finishing was adopted: it was added after the Development phase closed, during the Final phase (from
    2026-10-19), and gated on internal VAL only, under rules written before each gate's numbers existed, with independent reviews. On the post-decoder
    ensemble map of a VAL replica of `final_MTfull` (8 TRAIN-only members, TRAIN-only decoders) it passed at ΔS
    +0.0000443 (shot bootstrap P 1.000; ΔR²ψ −3.6e-7 and ΔD_LCFS +1.2e-5, both within their pre-declared bounds). An
    earlier declared variant with an added triangularity step failed its D_LCFS bound and is not used; the κ step alone
    had been scored as information in that run (ensemble map before the decoder, same VAL shots) before its own gate
    was declared, so that gate re-checked a known effect on the same shots. Leaving the
    fallback frames unedited was decided after that gate, as a consistency rule (post hoc; on VAL it moves ΔS to
    +0.0000470), before this stage was run on any Final input. The same step on the post-decoder ensemble map of a VAL
    replica of `final_A22li_dec` (8 TRAIN-only li2ft members, TRAIN-only decoders), with the fallback frames unedited,
    gave ΔS +0.0000462 (shot bootstrap P 1.000; ΔR²ψ −2.5e-7, ΔD_LCFS +1.1e-5). It was never read on Dev and was
    chosen without any Final score. We had made no Final submission when the step or the slots were
    chosen, and every run on the private inputs was label-free. On 2026-10-19, before the step existed, the configs then
    planned (`final_A22li_dec`, `final_MTli`, `final_MTfull`) were built from earlier package commits. From 2026-10-20
    the step ran in a rehearsal of `final_MTfull_mmk` with an earlier package commit, in acceptance builds of both κ
    configs with this package's code before its last documentation commit, and in the builds from the pinned commit
    that make the files of all three submissions (`final_MTfull` among them, without the step). These runs were read
    only for build checks (stage status counts, the range of the edit factor, realised against intended κ moves,
    lirep4 counts, file hashes and format validation, the differences from our files of the same members without the
    step) and timings. The rehearsal's timings were used to check the worker time-out margin; its counts, and the
    2026-10-19 files and lirep4 counts, set the bounds and references of the later builds' pre-declared acceptance
    checks, which decided only whether a build was accepted; the acceptance builds' files and counts are in turn the
    references for the pinned-commit builds. Nothing that changes the output was chosen from any of them.
    What the step does (`fec/mmk.py` with the frozen operator file `fec/mmk_op.py`, pinned by SHA-256): every κ is
    read with the organisers' LCFS routine inside the TRAIN-only envelope; the target is the equal mean of the 22
    members' own κ; the edit is a radial compression or expansion of the map about the magnetic axis, with weight 1
    inside the separatrix tapering to 0 by ψ_N = 1.4 and Gaussian weights in Z (σ_z = 0.25 m) at the heights of the
    inboard and outboard extremes; its factor is found per frame by bisection on [0.9, 1.1] so that κ moves half way
    (α = 0.5) to the target, and κ of the written float32 map is re-checked. Frames without an admissible factor or
    whose re-check fails, frames with fewer than 18 of the 22 members' κ readable, and frames written by the
    no-Thomson fallback are left unchanged. We had tested moving the map toward the members' mean shape scalars with
    a different operator on 2026-10-08 and had not adopted it.
* **Dev feedback (DIII-D).** Candidates were built and gated on internal VAL, except lirep4 and the
  post-hoc G1 rule (both above). The Dev board then scored the entries below, all listed in
  `DEV_LOG.md`, and from 2026-10-10 it decided or vetoed complete choices:
  * the first DIII-D entries (2026-10-08, Dev 968120, 968560 and 969279): three successive ensembles
    selected on internal VAL, the third on the `idx` Ip axis (0.996239); then that ensemble with the
    q95 G1 hybrid, adopted on internal VAL (2026-10-09, Dev 970439 to 970443, one identical file:
    0.996331). No DIII-D choice was made from these scores. Dev 969574 and 969641 (2026-10-08)
    repeated the third file for a MAST comparison and a resubmission;
  * the all22 ensemble (2026-10-10; its rule made it the Final recipe at the time; later
    superseded);
  * lirep4 (2026-10-11; in every config);
  * all44 (2026-10-11; carried forward without a pre-declared bar);
  * the GS projection gsp1 (2026-10-12, not adopted) and a post-hoc variant gsp3 (read for
    information only);
  * W3e44 against all44 (2026-10-12; 0.996580 against 0.996495; W3e44 adopted): the DIII-D half of
    `final_B` and `final_C`;
  * W3k44 against W3e44 (2026-10-13; 0.996653). Its pre-declared rule chose the W3k24 ensemble
    (`final_A24`), which was then amended to W3k22 + li2ft arm A;
  * W3k22 + li2ft arm A under a pre-declared veto (2026-10-14; 0.996671 ≥ 0.996603: kept), with a
    pre-declared anchor rule against W3k24: the DIII-D half of `final_A22li`, `final_A22li_r05` and
    `final_A22li_r2`;
  * the off-span decoder (2026-10-14; 0.996748 against 0.996671, pre-declared bar +0.00002:
    adopted): `final_A22li_dec`;
  * the MT ensemble `final_MTfull` (2026-10-15, Dev 988521, read once, together with the r = 4 MAST
    map; 0.996775 against `final_A22li_dec`'s 0.996748, +0.000027). That is below the pre-declared
    +0.00003 bar for replacing `final_A22li_dec`. Under a second pre-declared rule (at least
    `final_A24`'s 0.996646 + 0.00002) `final_MTfull` was chosen as a separate Final submission, then with
    its own package. Its VAL gain against the matched control was +0.000108, so
    the Dev gain is about a quarter of it;
  * the MT shape-half ensemble `final_MTli` (`final_MTfull`'s 11 shape members with `final_A22li_dec`'s 11 no-Thomson
    members, the same fallback and chain; its file was built on 2026-10-14). It was read once on Dev on 2026-10-16
    (Dev 991974: 0.996760) under a rule written before that read: it would replace `final_A24r2` as a Final submission
    if it scored at least 0.996666, its package path reproduced that file and its timed clean-clone rebuild took at
    most 23.0 A100-PCIe-h at k 1.170. An addendum written after that read and after the rebuild had started, before
    any of its GPU stages had been timed (forecast 22.97 A100-PCIe-h), also accepted a flagged reading: at most 24.0
    A100-PCIe-h, and at most 23.0 when normalised by step times to the 500 W pod of our first timed rebuild. Outcome:
    the package path reproduced that file byte for byte on our GPU (with the MAST network shipped then;
    this package ships another, MAST section); the timed rebuild passed only as flagged (Compute
    below); and the rebuild's own Dev read was green (0.996763, Dev 994963, 2026-10-17; pre-declared bar 0.996660). So
    `final_MTli` replaced `final_A24r2` in our Final plan; before any Final submission was made,
    `final_MTfull_mmk` took its place ("Final submissions" below). Compute: 22.80 A100-PCIe-h on the README basis
    (same-pod bench ratios); 22.90 on the conservative whole-run reading (`README.md` "Compute"). The timed
    clean-clone rebuild of `final_MTli` then measured 19.982 A100-SXM hours, 23.385 A100-PCIe-h at k 1.170: above the
    23.0 bar and below the 24 h limit; normalised, 22.581, so a flagged pass only. At the pessimistic factor k 1.236
    it reads 24.698, above 24 h (`README.md` "Timed clean-clone rebuilds").
  * `final_MTfull_mmk` and `final_A22li_dec_mmk` (`final_MTfull` and `final_A22li_dec` with the κ finishing; README)
    were not read on Dev: the κ finishing was added after the Development phase and gated on internal VAL only.
  * `final_MTfull_mmk_nrsl` and `final_A22li_dec_mmk_nrsl` have the DIII-D half of `final_MTfull_mmk` and
    `final_A22li_dec_mmk` unchanged (their NRS-L cascade changes MAST maps only) and were not read on Dev.

  Three clean-clone rebuilds were read on Dev under pre-declared green rules, as confirmation and
  not selection: `final_A22li` 0.996672 (Dev 986833, 2026-10-14), `final_C` 0.996577 (Dev 987128,
  2026-10-14) and `final_A24` 0.996646 (Dev 988498, 2026-10-15; also the DIII-D half of `final_A24r2`;
  green against its reference
  0.996636). Many Dev
  reads paired a MAST candidate with a DIII-D candidate; the two halves are scored independently.
  Reads made for a MAST candidate alone repeated an already-scored DIII-D file and are not
  Challenge 1 evidence.

  On 2026-10-16 files produced from our Final configs were read on Dev under pre-declared green rules,
  again as confirmation and not selection: the timed clean-clone rebuilds of `final_MTfull`
  (0.996782, Dev 991548, against its 0.996775) and of `final_A22li_dec` (0.996743, Dev 991560,
  against its 0.996748), and the `final_A24r2` file that its package made from the weights shipped then
  (0.996656, Dev 991860, against the `final_A24` rebuild's 0.996646). Dev 991439 (2026-10-16)
  repeated the `final_MTfull` file for a MAST read.
  On 2026-10-17 the timed clean-clone rebuild of `final_MTli` was read under a pre-declared green rule
  (at least 0.996660). Unlike the reads above, this read was also a condition for `final_MTli` replacing
  `final_A24r2`: 0.996763 (Dev 994963, against its 0.996760), green. The other conditions held (above), so
  `final_MTli` replaced `final_A24r2` in our Final plan (later replaced in turn; "Final submissions" below).
  The last Development entry (Dev 994977, 2026-10-17) repeated two already-scored files, the
  `final_MTfull` rebuild's DIII-D file (Dev 991548) and the r = 2 MAST file, so that the board's last
  entry shows them; it carries no read.

## MAST (Challenge 2, zero-shot)

* **No MAST labels.** No model was trained on MAST EFIT equilibria or on any MAST label. Our MAST
  network was trained only on our own synthetic equilibria, which are driven by the public MAST
  inputs (below). No MAST or MAST-U EFIT output, reconstruction or statistic from any source was
  used. No model, feature, loss, mask, prior or selection used the starter's three labelled MAST
  demo parquet files or the MAST scorer envelope (`mast_envelope.npz`), and no code of ours reads
  them. Our workspace records show them opened only for inventory and audit: on 2026-09-08 a
  generic file-catalog script (removed from our repository on 2026-10-11) read the demo files'
  parquet footers (schema and row counts only, no target values) and the envelope's file hash and
  array headers (no mask values), and on 2026-10-07 an independent review agent read the
  envelope's spatial extent to compare it with our solver box (audit only; the result was not
  used). One sentence of the dataset card's MAST statistics (a flux-amplitude ratio between the two
  machines) was read by us on 2026-10-14 (UTC) when a redaction script missed it; it was recorded
  and not used anywhere. A separate clean-room agent read the card's other MAST statistic lines
  only to check them for non-MAST uses; its report carries no MAST content, and nothing from those
  lines was used in any model or selection.
* **Machine description:** the conductor geometry shipped in the inputs. One physics member
  (`zsc_c1_lcc`) applies a centre-column limiter from the official MAST wall outline (FAIR-MAST
  level-2 `wall` limiter_r/z of shot 30421, limiter geometry for shots 0–40000;
  `frozen/geometry/mast_limiter_fairmast.npz`). This is machine
  geometry, not equilibrium data. Two published machine constants are also used. The TF coil
  has 24 turns in series (B_T for betaN; Voss 1998, Sykes 2002). The 656-turn solenoid is driven as two parallel
  circuits (Battaglia et al., UKAEA-CCFE-PR(19)34); this is the f = 0.5 baseline below.
* **Passive currents:** our C1 induction model of the vessel, driven by the measured coil and
  plasma currents (inputs only). Its conductor geometry is the published MAST passive-structure
  machine description: FAIR-MAST level-2 group `pf_passive` (UKAEA mast-geometry
  `pfcoils.dat_M4`, shot range 0–40000), element geometry only (r, z, width, height, shape
  angles), read once for shot 30420 and hard-coded in `fec/passive.py`; the group's current
  values (`AMM_*`), which are EFIT wall-model outputs, were not read. This is machine
  geometry, not equilibrium data. Resistances are geometric (rho·2πR/area) with handbook
  resistivities, multiplied by the scale r below in the configs that use it. The model's total
  induced current was range-checked, not fitted, against one published 3D eddy-current simulation
  (Berkery et al., PPCF 63 (2021) 055014, Fig. 1a, shot 23822).
* **Amortized network (every config except `final_C`):** trained only on our own synthetic
  free-boundary GS equilibria (`fec/amort_gen7.py`, family v7h). Those equilibria are driven by the
  public MAST INPUTS (coils, Ip, Thomson) with hypothesised current profiles: a 50/50 mix of a
  two-shape family and a profile family that takes its configuration parameters from the published
  MAST-U EFIT++ configuration (Kogan et al. 2022: polynomial p′ and FF′ with two degrees of freedom
  each); no EFIT output is used, and the family is not
  verified for MAST. The generator prerequisites (`frozen/amort/*`) are fitted to public MAST
  inputs only (Thomson profile prior, edge-crossing pool).
* **Maps per config.**
  * `final_B`, `final_A24`, `final_A22li`: the network map at solenoid fraction f = 0.55, computed
    with the C1 passive flux at its nominal resistances (r = 1) in its known part, then 30 ms
    temporal smoothing.
  * `final_A22li_r2`, `final_A22li_dec`, `final_MTfull`, `final_A24r2`, `final_MTli`, `final_MTfull_mmk`, `final_A22li_dec_mmk`: the same network map plus the vacuum-flux
    change from scaling every C1 resistance by r = 2, with the network frozen; then 30 ms smoothing.
  * `final_MTfull_mmk_nrsl`, `final_A22li_dec_mmk_nrsl`: the r = 2 maps of `final_MTfull_mmk` and
    `final_A22li_dec_mmk` (previous item), then the NRS-L cascade on them ("NRS-L cascade" below).
  * `final_A22li_r05`: as `final_A22li_r2`, with r = 0.5 (read on Dev and not adopted, below).
  * `final_C`: no network. Its maps are those of the physics average `avg_f05` (below), whose
    members use the f = 0.5 baseline and the C1 passive currents at r = 1 or no passive currents.
* **Solenoid turn fraction f = 0.55 (Dev-set constant 1 of 2; every config except `final_C`).**
  f is the effective fraction of the solenoid turns that carry the measured solenoid current in
  our coil model.
  * The physics pipeline (self-consistent heights, no passive currents) was read on Dev at f = 0.5
    (baseline) and 0.65.
  * A readout was declared after the f = 0.5 score and before the f = 0.65 score. It combined the
    board R²ψ of those two entries with a label-free curvature of the map change and gave
    f* = 0.540, rounded to 0.55 by the same rule.
  * The amortized maps were then read at f = 0.5, 0.55 (adopted, ΔS_map +0.0103) and 0.6
    (rejected, −0.0091), and the search stopped.
  * Physics maps at f = 0.65 and 0.75 also entered Dev inside map averages. No config of this
    package contains them.
  * f is a continuous constant set with board feedback (team-approved exception).
* **Passive-resistance scale r = 2 (Dev-set constant 2 of 2; `final_A22li_r2`,
  `final_A22li_dec`, `final_MTfull`, `final_A24r2`, `final_MTli`, `final_MTfull_mmk`, `final_A22li_dec_mmk`,
  `final_MTfull_mmk_nrsl`, `final_A22li_dec_mmk_nrsl`).** The scale multiplies every C1 resistance; it changes only the computed
  vacuum flux, with the network frozen.
  * *Reads.* The scale was read on Dev one setting at a time against pre-declared bars
    (ΔS_map ≥ +0.0048, or +0.003 for a further setting of an adopted knob, with ≥ 2 of 3 map terms
    agreeing): r = 0.5: ΔS_map −0.031154, not adopted (`final_A22li_r05` keeps that map); r = 2:
    +0.006502, 3/3 terms, **adopted**; r = 4 against r = 2: +0.000711, not adopted. r = 3 was
    dropped without a read.
  * *Board-informed direction.* r = 2 had failed our own label-free pre-Dev wall gate. Its read
    reversed our pre-declared "no Dev read after the label-free wall gate fails" rule, because of
    the r = 0.5 Dev number.
  * *Bar simplification.* The +0.003 bar omits the 2·SD term of our general map rule (a stated
    simplification).
  * *Outside the training range.* r = 4 lies outside the generator's training range of r, so that
    read tested the vacuum model, not the network.
* **NRS-L cascade (`final_MTfull_mmk_nrsl` and `final_A22li_dec_mmk_nrsl` only).** A label-free stage after the
  r = 2 maps (README, "NRS-L cascade"): each frame is re-solved in the MAST physical field (coil currents at f = 0.55, the
  map's own C1 passive currents, a uniform vertical field), its current profile starting from the base map's own current
  density; stage L steers the profile's peaking towards a label-free li(2) target, stage P is the fallback without that
  target, and a frame where both fail contributes no increment of its own. Only the increment is smoothed (computed on
  the already-smoothed maps, so a frame without an increment of its own still carries its neighbours' smoothed
  increments), and only the MAST `psirz` changes; q95 and βN are those of the parent config.
  * *Chosen on DIII-D, with DIII-D labels.* The stage, its order and every setting were chosen on DIII-D, not on MAST.
    Design variants were scored on DIII-D TRAIN frames against DIII-D EFIT labels (two 40-shot TRAIN draws; 15
    variants, 12 single-stage and 3 cascades; the li-target stage was added after the first pre-declared arm fell short
    of its bar on TRAIN, and the frozen cascade was the second-best variant on the TRAIN probe, by 0.0002 in S_map).
    The frozen operator was then read once on a DIII-D VAL testbed with DIII-D labels: the same method on DIII-D coil
    currents, Ip and Thomson, applied to the maps of an amortized network trained only on synthetic DIII-D equilibria,
    and compared with those maps on 40 VAL shots (4,750 frames), both scored against the DIII-D EFIT labels with the
    challenge functionals, with a paired shot bootstrap. The rule written before that read was ΔS_map ≥ +0.005 with P ≥ 0.90, no loss in the consistency term
    and no increase of the LCFS distance; the read gave ΔS_map +0.0107 (P(ΔS_map > 0) 1.000), consistency +0.046 and
    LCFS distance −0.0067. On that testbed the magnetic-axis R term got worse (R² −0.905 to −0.992, an outward shift of
    about 0.6 cm on stage-L frames), so a MAST-side bound on the axis shift was declared before any MAST number was
    computed; it is one of the stage's checks.
  * *li target: label-free on MAST, calibrated on DIII-D TRAIN only.* On MAST the target is computed from the inputs
    only (the shot's measured Ip and Thomson Te / ne, the TF current) and the base map's geometry, by the
    current-diffusion estimator `fec/current_diffusion.py` (a frozen variant). Its linear calibration was fitted on
    DIII-D TRAIN frames against DIII-D EFIT li (`frozen/cdiff/frozen.json`), frames without a value take the nearest
    value of the same shot or else the DIII-D TRAIN median (`frozen/licond/sigma.json`), and its range check uses the
    li range of synthetic training equilibria. No MAST li or other MAST label enters it.
  * *No MAST reconstruction anywhere.* No MAST or MAST-U EFIT output, reconstruction or statistic, no MAST label, no
    demo shot, no scorer envelope and no MAST score of any kind was used to build, choose, calibrate or check the
    stage. Its MAST-side checks (README, "NRS-L cascade": LCFS found, LCFS inside the public FAIR-MAST wall polygon,
    the map's enclosed current against the measured Ip, |Z_axis|, the axis shift against the base maps, li-target
    coverage and range) are label-free statistics of the stage's own maps; they can only make a fold keep the parent
    maps and were never used to choose anything. The stage was added during the Final phase
    (2026-10-20), after the Final inputs were published; before it entered this package its label-free checks were
    run on both MAST folds' inputs (public and private), which gave only those statistics. No Development-phase read
    of it exists, and no Final-phase score was used.
* **Scalars (every config):** q95 from stage 2 of the physics average `avg_f05` (our kinetic-fit
  equilibria at the f = 0.5 baseline; inputs only). betaN from a model fitted on DIII-D TRAIN
  (`lam_fgw`, `frozen/betan_transfer/`) and applied to MAST Thomson records of the same
  equilibria, with B at the magnetic axis. The fill3 shot-fill rule (≥ 3 finite frames, else the
  fold median) is a post-hoc, label-free hygiene fix.
  * *How the source was chosen.* The q95/betaN source was chosen after the 2026-10-11 Dev read
    (per-term q95 R² 0.701, against 0.597 for the amortized maps' own contour q95), without a rule
    written before that read.
  * *Later variants.* Every later scalar variant (q95_mix, q95_sau, q95_ts, betaN at the geometric
    centre, a kinetic betaN) was read against it under a pre-declared per-term rule (≥ +0.01), and
    none beat it. A betaN variant with B0 = F_vac/0.85 m was withdrawn before any read.
  * *fill3.* The rule was found from inputs (a shot with a single finite Thomson frame) and applied
    to every slot from 2026-10-12. It was not compared on Dev, though the size of the Dev betaN
    spread between entries motivated looking for it.
* **Dev feedback (MAST).** The Dev board compared genuine candidates; three further reads checked the
  reproducibility of rebuilt MAST files and were not candidates (Dev 986833, 2026-10-14, and Dev
  991560 and 994963, "Package checks" below). Each read followed a rule written before it, except where
  `DEV_LOG.md` marks otherwise (the policy timing and the post-hoc scalar source above); the three
  MAST-only entries' reads were pre-declared outside this package, in our private research log ("Other MAST-only
  entries" below).
  * *Choices that shaped these MAST halves:* self-consistent heights (sub09); C1 passive currents
    kept (sub18); boundary-limiter q95/betaN (sub05; later replaced by the `avg_f05` source, sub38);
    the deep-crossing centre-column limiter member (sub28); 30 ms smoothing (sub29;
    50 ms rejected, sub51); the `avg_f05` member set and its stage-2 q95/betaN (sub38; the scalar
    choice post hoc; `final_C`'s maps); the amortized v7h maps (sub43; v7a not adopted, sub45);
    f = 0.55 (sub21 readout, sub46w; 0.6 rejected, sub49); betaN at B(R_axis) (B(R_geo) rejected,
    sub44); r = 2 (Dev 987175, 2026-10-14; r = 0.5 Dev 987128 and r = 4 Dev 988521 not adopted;
    the DIII-D half of 988521 was `final_MTfull`).
  * *Not adopted:* f = 0.6, a map average and 50 ms smoothing (Dev 981382, 981386, 981389;
    2026-10-13); DIII-D-trained transfer maps (Dev 981390, 2026-10-13; Dev 986001, 2026-10-14); a
    15-seed network ensemble (Dev 988498, 2026-10-15; its DIII-D half was the `final_A24` rebuild);
    the passive model calibrated on public vacuum shots ("Other public information read" above),
    each read under a pre-declared bar (ΔS_map ≥ +0.0048 versus r = 2 with ≥ 2 of the 3 map terms
    agreeing; the bar of our r = 0.5 and r = 2 reads): the frozen-network map with per-group
    passive scales, ΔS_map −0.000814 (Dev
    991439, 2026-10-16), and the retrained generator and network at f = 0.50, ΔS_map −0.024664
    (Dev 991548, 2026-10-16; its DIII-D half was the `final_MTfull` rebuild).
  * *Package checks (2026-10-16 and 2026-10-17):* the MAST file of the two timed clean-clone rebuilds was read once
    as a reproducibility check against a pre-declared tolerance of ±0.010 around 0.802845 (Dev 991560;
    result under "Final submissions" below). The r = 2 file made from the Development-phase MAST network
    (model.pt SHA-256 `529ebce2…`, not shipped in this package; "Final submissions" below) was
    submitted unchanged with two DIII-D reads (Dev 991860 and 991974) and read 0.802845
    both times.
    A third timed clean-clone rebuild, of `final_MTli`, produced the same MAST file byte for byte; read
    again with that rebuild's DIII-D file (Dev 994963, 2026-10-17), it scored 0.793869 again, inside the
    same tolerance.
    The r = 2 file from the Development-phase network was submitted unchanged once more in the last Development
    entry (Dev 994977, 2026-10-17; no read) and read 0.802845 again.
  * *Other MAST-only entries:* three further MAST-only Development entries (Dev 988761, 988789 and
    988995; 2026-10-15) whose pipeline used public FAIR-MAST level-1 magnetics (flux loops, pickup coils) of archive
    shots that we matched to the public-test shots by their plasma-current and coil-current signals (the accepted
    match had identical plasma-current samples). That pipeline is not part of this package or of any Final
    submission. Their rules carry no commit in `DEV_LOG.md`, which lists them. We withdrew the line on 2026-10-15
    after re-reading the Terms; no matched data, match list or code from it enters any Final config, and the match
    list is not shipped. The last of the three (Dev 988995) was adopted by a post-hoc ruling as the Challenge 2 file
    we then intended to submit; it was withdrawn with the rest of the line and is in no Final submission.
  * *Not carried forward:* the denoised-height candidate (sub37b) met its declared comparison
    against sub09 but was not carried forward, and the reason is not written down.
  * `DEV_LOG.md` lists every MAST comparison, including the rejected ones; the three MAST-only
    entries are listed with their scores and comparison only.
* **Final submissions: MAST half, rebuild and award scoring.**
  * *MAST half.* Our Final submissions use three of `final_A22li_dec_mmk`, `final_MTfull_mmk`, `final_MTfull`,
    `final_A22li_dec_mmk_nrsl` and `final_MTfull_mmk_nrsl`; the submission record names which. Each is built from a
    pinned package commit (provided after the Final on request: the organisers ask the top five entries per
    challenge, which then have 7 days). In each, the MAST half starts from the r = 2 network map of "Maps per
    config" with the scalars of "Scalars", made with the shipped network weights; in the two `_nrsl` configs the
    NRS-L cascade ("NRS-L cascade" above) is then applied to those maps. The shipped MAST network is not the one behind our
    Development-phase MAST files (next item).
  * *The shipped MAST network and its rebuild.* A clean-clone rebuild retrains the amortized network from
    scratch, and that retraining is chaotic: in our tests, changing a single training frame changed
    the selected network width and moved the maps. Our Development anchor, the r = 2 file that read 0.802845
    (Dev 987175, 2026-10-14), came from a network trained on our own GPU, which this package's training
    entrypoint did not reproduce in our A100 test. A retrained rebuild must land within 0.01, and a
    shipped-weights rebuild within 1e-4, of the submitted S on each ranked machine (organisers' clarification,
    2026-10-18). We therefore ship the network that the entrypoint (`train.py --stage mast`, synthetic set
    regenerated) produced on the two A100 pods we tested: model.pt SHA-256 `bfbb346c…`, trained in the timed
    clean-clone rebuild of `final_MTfull` (A100-SXM4-80GB) and reproduced bit for bit by the same command on an A100
    80GB PCIe with another driver. Both pods ran Python 3.11.10 and torch 2.13.0+cu126 in the same container image.
    The Python 3.13 environment that `environment/install_env.sh` documents for this package, the default PyPI torch
    wheel (+cu130), other host CPUs and the organisers' own image have not been tested for this retraining, so an
    organiser rebuild is not guaranteed to reproduce the shipped network. The timed clean-clone rebuilds of
    `final_A22li_dec`, `final_MTfull` and `final_MTli` produced the same r = 2 MAST file, and the shipped network
    reproduces its maps bit for bit on an A100; that file read 0.793869 on Dev (Dev 991560, 2026-10-16), 0.008976
    below the Development anchor. The r = 2 configuration is unchanged. We made this choice on 2026-10-18, for the
    organisers' rebuild tolerance, before any Final input existed and without any Final score; the earlier network
    is not shipped. An earlier clean-clone rebuild of `final_A22li` read its r = 1 MAST file 0.789608 against
    0.796343 for our file of the same recipe (Dev 986833, 2026-10-14). Our
    Final MAST files are made from the shipped weights by this package's inference entrypoint at its pinned commit
    on RunPod GPU pods, one per slot (README, "Quick start (inference with the
    shipped weights, one GPU)": `bash reproduce.sh <config>`).
  * *Award scoring.* Under the Final rules our best scores count: our highest DIII-D S across our scored Final
    submissions for Award #1, our highest MAST S over that highest DIII-D S for Award #2, and the submission with
    the highest MAST S for the honourable mentions; no nomination is made. All Final submissions start from the same
    r = 2 maps of the same shipped MAST network (C1 passive-resistance scale r = 2, chosen with Development-phase
    feedback, as stated). The `_nrsl` configs add the NRS-L cascade, which was added during the Final phase and has
    no Development-phase read. No Final-phase score was used to choose any MAST component. No MAST equilibrium reconstruction was used for any build step of
    these three submissions; the only MAST-derived feedback they used is the platform-returned
    Development-phase scores and Detailed Results of our own entries stated above.
