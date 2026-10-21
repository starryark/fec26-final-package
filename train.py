"""From-scratch training entrypoint of the Final reproduction package.

    python train.py --config final_A22li --stage data|li2cache|auxcache|d3d|mast|export|all [--data-root DATA_ROOT]
        [--work WORK_DIR] [--group-size 4] [--pix-impl compile|eager] [--dry-run]
    python train.py --config final_A22li --stage preflight|download|inputs|fingerprint [--split private_test]
        [--mode full|inference] [--work WORK_DIR]

Plumbing stages (reproduce.sh; agent pkgfix2 2026-10-13, R35 package hygiene):
  preflight    free disk space under <work> (PEAK_DISK_GB below: warns while it is None, else fails below
               DISK_MARGIN x the measured peak, crediting what <work> already holds).
  download     the public training configs at the dataset revision pinned in data_manifest.json (fec/final_data.py;
               the resolved commit is recorded in <work>/fec_work/hf_data/revisions.json).  Network.
  inputs       the --split fold's inputs (diii_d_<split>, mast_<split>) into the same mirror, where infer.py finds
               them.  Network.  reproduce.sh runs every later step with HF_HUB_OFFLINE=1.
  fingerprint  <work>/run_fingerprint.json (Python, torch, CUDA, driver, GPU, package commit, config SHA-256,
               dataset and weights revisions), printed.

Stages (each idempotent; outputs under <work>):
  data  DIII-D TRAIN inputs + labels: fec/final_data.py download diii_d_train (and the two public
        configs the MAST generator and geometry read: diii_d_public_test, mast_public_test),
        fec/extract.py <cfg> --ip-timebase idx (DIII-D), fec/dataset.py <cfg>, fec/bnd_b.py and
        fec/bnd_b.py --alldata (label LCFS targets of every labelled row, TRAIN+VAL+TEST, + the TRAIN-only
        envelope env_train.npz),
        fec/passive.py signals/frames (MAST C1 passive currents), and checks that the frozen clip
        ranges equal the TRAIN labels' (fec/final_build.py check).  CPU only.
  li2cache  only for configs whose train spec has a group marked "li2_cache" (final_A22li): the
        li(2) target cache of the li2ft fine-tune, fec/li2_torch.py build --splits train,val,test
        (the scorer's li(2) of every labelled diii_d_train frame at the true map's own LCFS, with the
        TRAIN-only envelope env_train.npz; no scorer mask) into <work>/fec_work_idx/li2/cache_all,
        then fec/li2_torch.py check.  Needs the data stage (test_eval_b/DONE).  CPU only.  The d3d
        stage runs it first when the cache is missing (no-op for the other configs).
  auxcache  only for configs whose train spec has a group marked "aux" (final_MTfull's MT members): the mtask M2
        auxiliary training targets (kappa, delta_top, delta_bot, volume + 16-point flux-surface-averaged J_phi of the
        LABEL psi at the scorer's LCFS) of every labelled diii_d_train frame, fec/mtask_aux_all.py build from the li2
        cache above into <work>/fec_work_idx/mtask/aux_all, then fec/mtask_aux.py check.  Training-only targets (never
        inputs).  CPU only.  The d3d stage runs it first when the cache is missing (no-op for the other configs);
        an existing cache (meta.json) is re-checked before reuse, as the li2 cache.
  d3d   the candidate's DIII-D members (configs/train_d3d_<config>.json: train_d3d_b argument
        lists; same tags, seeds and hyper-parameters as the submitted models) with the batched
        trainer fec/train_d3d_bx.py (stacks of --group-size members; --alldata = TRAIN+VAL+TEST
        shots of the frozen split).  PCA basis, input standardisation (with the current normalisation)
        and target scales, and an "aux" group's aux-target standardisation, are fitted inside each run on
        its training rows (seed-0 PCA, as trained): with --alldata, as for every member here, on all
        labelled diii_d_train rows (TRAIN+VAL+TEST); only the gate members (no --alldata) fit them on
        TRAIN frames.  The loss-weight envelope env_train.npz stays TRAIN-only.  A "li2_cache" group
        (final_A22li's li2ftA22, last) fine-tunes the members trained before it (--ft_from) and
        gets --li2_cache <work>/fec_work_idx/li2/cache_all.  An "aux" group (final_MTfull: the MT bases and their
        li2ft fine-tunes) gets the run-level heads --aux_heads shape_jphi --aux_cache <work>/fec_work_idx/mtask/aux_all
        --aux_w 1.0 (the MT production pods' flags).  GPU.  A re-run skips every group whose
        members all finished (model.pt + summary.json).  A half-finished group (a crash) is retrained
        whole: its member dirs are moved to <models>/_partial_<group>_<UTC>/ first (nothing is deleted
        or overwritten; finished groups are never touched).  An existing li2 cache (meta.json) is
        re-checked with fec/li2_torch.py check before it is reused (a failing cache is moved aside and
        rebuilt).
  offdec  only for configs with the off-span decoder stage (d3d.dec: final_A22li_dec); --stage d3d (and all) runs
        it right after the members, so reproduce.sh needs no extra step; --stage offdec alone re-runs it.  The three
        decoders (seeds 40/41/42) of fec/offdec.py from the trained members' PCA basis and the labels of all
        labelled diii_d_train shots (the members' --alldata rows): basis (read from every member's model.pt,
        primary and fallback; refuses unless they agree to 1e-6), envelope (rebuilt from TRAIN labels, SHA-256
        checked), data (CPU; <FEC_IDX_WORK>/offdec_cache/<name>/), train (GPU, one run per seed) ->
        <FEC_WORK>/<d3d.dec.dir>/dec_s<seed>.pt.  tr_r.npy (6.6 GB) is deleted when all decoders exist
        (--keep-offdec-cache keeps it).  A re-run skips decoders trained in the current members' basis.
        A no-op for every other config.
  mast  the amortized MAST network (only for configs whose MAST maps read it, maps.net; a physics-map config
        such as final_C skips this stage, R2-5): C1 passive currents at f = 0.55 (CPU), our own synthetic GS
        equilibria v7h_f0.55 driven by the PUBLIC MAST inputs (fec/amort_gen7.py gen; GPU, 0.23
        H100-h; on A100 hardware (in our Python 3.11.10 / torch 2.13.0+cu126 pods) this default route trains the
        shipped n7h_f0.55 bit for bit; or --synthetic DIR
        installs the SHA-256-pinned copy from the weights repository's synthetic/gen_v7h_f0.55/, the
        Development-phase network's training set, which gave a different network in our A100 test (README.md
        "Expected rebuild variance"), checked against data_manifest.json 'synthetic'; a set counts as
        complete only when it holds exactly the chunk files listed there and each opens as an npz,
        otherwise unreadable chunks are moved aside and the generator, which keeps existing chunks,
        fills the gaps before a re-check), then fec/amort_net2.py train --gen v7h_f0.55 --norm --tag
        n7h_f0.55 --seeds 3.  Configs whose
        MAST maps use TX1s (final_A22li_vb, final_A22li_combo) then train TX1 -> TX1s with
        fec/final_tx1s.py train (the original fec/xfer_tx0.py / xfer_tx1.py / xfer_tx1s.py CLIs: DIII-D TRAIN
        data, 12 epochs; then 4 epochs with the own-solver synthetic MAST curriculum).  No MAST
        label or EFIT output is used anywhere.
Outputs replace the downloaded weights: <work>/fec_work/models/<tag>/model.pt and
<work>/fec_work/amort/models/n7h_f0.55/.  Then run infer.py with --weights <work>/weights_out
(train.py --stage export writes that tree + its SHA-256 list; it stops, before copying anything,
when a weight file the config needs is missing).

Measured compute (hardware, wall clock) per component is in README.md; retraining is seeded but
not bit-identical across GPU models / library versions (see PROVENANCE.md, determinism).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
from infer import dec_tags, env_for, install_frozen, mast_tags, recipe_tags, sha256  # noqa: E402

# Disk preflight (train.py --stage preflight, reproduce.sh full).  PEAK_DISK_GB is the measured peak disk use of a full
# rebuild under --work, in GB; root sets it from the timed A100 rebuild.  While it is None the preflight only warns.
# 282: clean-clone final_A22li rebuild on one A100-SXM, 2026-10-14 (272 GB sampled, ~282 GB at the end of export;
# docs/archive/fec/FINAL_REBUILD_A22LI_20261013.md, rebuild defect D9), so the preflight asks for 1.25 x 282 = 353 GB.
PEAK_DISK_GB = 282
DISK_MARGIN = 1.25          # required: free space + what --work already holds >= DISK_MARGIN x PEAK_DISK_GB
# configs whose rebuild leaves more on disk at the export peak (GB, added to PEAK_DISK_GB; every other config: 0).
# final_A22li_dec: the offdec stage's tr_r.npy (6.6 GB) is deleted before mast/export; what stays is the decoders in
# <FEC_WORK> and weights_out (2 x 3 x 45.7 MB) and tr_c/tr_rms/tr_wp + basis (0.43 GB): 0.7 GB, rounded up.
PEAK_DISK_EXTRA_GB = {"final_A22li_dec": 1,
                      # final_MTfull: final_A22li_dec's decoder files + the aux cache (0.14 GB); its 22 MT bases are not
                      # weights_manifest rows, so its export holds fewer files than final_A22li_dec's (estimate, R4)
                      "final_MTfull": 1,
                      # final_MTli: the same decoder files + aux cache; its 11 MT F1 bases are not weights_manifest rows
                      # (estimate as final_MTfull's)
                      "final_MTli": 1,
                      # final_MTfull_mmk: final_MTfull's training exactly (the mmk stage trains nothing)
                      "final_MTfull_mmk": 1,
                      # final_A22li_dec_mmk: final_A22li_dec's training exactly (the mmk stage trains nothing)
                      "final_A22li_dec_mmk": 1}


def peak_disk_gb(cfg_name=None):
    """Measured peak disk of a full rebuild of the config (PEAK_DISK_GB + PEAK_DISK_EXTRA_GB), None while unset."""
    return None if PEAK_DISK_GB is None else PEAK_DISK_GB + PEAK_DISK_EXTRA_GB.get(cfg_name, 0)
TRAIN_CONFIGS = ("diii_d_train", "diii_d_public_test", "mast_public_test")   # public configs of the data stage
SYNTH_KEYS = ("g", "valid")  # arrays every fec/amort_gen7.py chunk holds (np.savez in gen)


def utc_stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sh(cmd, env, dry, log=None):
    print("$ " + subprocess.list2cmdline([str(c) for c in cmd]), flush=True)
    if dry:
        return 0
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            rc = subprocess.run([str(c) for c in cmd], cwd=str(PKG), env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
    else:
        rc = subprocess.run([str(c) for c in cmd], cwd=str(PKG), env=env).returncode
    if rc:
        raise SystemExit(f"exit {rc}: {cmd} (log {log})")
    return rc


def stage_data(a, env):
    py = [sys.executable, "-B"]
    W, I = Path(env["FEC_WORK"]), Path(env["FEC_IDX_WORK"])
    logs = W / "logs"
    sh(py + ["fec/final_data.py", "download", "diii_d_train", "diii_d_public_test",
             "mast_public_test"], env, a.dry_run, logs / "download.log")
    if not a.dry_run:      # the idx root reads the downloaded parquet through <idx>/hf_data (as run_final links it)
        link_dir(I / "hf_data", W / "hf_data")
    for cfg in ("diii_d_train", "diii_d_public_test"):          # DIII-D: idx Ip time axis
        e = dict(env, FEC_WORK=str(I))
        sh(py + ["fec/extract.py", cfg, "--workers", a.workers, "--ip-timebase", "idx"], e,
           a.dry_run, logs / f"extract_{cfg}.log")
        sh(py + ["fec/dataset.py", cfg], e, a.dry_run, logs / f"dataset_{cfg}.log")
    e = dict(env, FEC_WORK=str(I))
    sh(py + ["fec/bnd_b.py"], e, a.dry_run, logs / "bnd_b.log")
    if not a.dry_run:      # bnd_b.py --alldata (and train_d3d_bx --alldata) refuse without this marker
        write_test_eval_marker(I)
    sh(py + ["fec/bnd_b.py", "--alldata"], e, a.dry_run, logs / "bnd_b_all.log")
    for cfg in ("mast_public_test",):
        sh(py + ["fec/extract.py", cfg, "--workers", a.workers], env, a.dry_run,
           logs / f"extract_{cfg}.log")
        sh(py + ["fec/dataset.py", cfg], env, a.dry_run, logs / f"dataset_{cfg}.log")
        sh(py + ["fec/passive.py", "signals", "--config", cfg, "--workers", a.workers], env,
           a.dry_run, logs / f"passive_{cfg}.log")
        sh(py + ["fec/passive.py", "frames", "--model", "c1", "--rscale", "1", "--config", cfg],
           env, a.dry_run, logs / f"passive_{cfg}.log")
    # the DIII-D TRAIN caches used for the clip ranges live in the idx root
    sh(py + ["fec/final_build.py", "check", "--ranges", "fec/final_train_ranges.json"],
       dict(env, FEC_WORK=str(I)), a.dry_run, logs / "ranges_check.log")


def write_test_eval_marker(idx_root):
    """<idx>/test_eval_b/DONE: our internal TEST split was evaluated once (2026-10-08), so the all-data refit
    (bnd_b.py --alldata, train_d3d_bx.py --alldata) may use the TEST shots.  Written before the first --alldata
    step (it used to be written at the end of stage_data, after bnd_b.py --alldata had refused)."""
    marker = Path(idx_root) / "test_eval_b" / "DONE"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("internal TEST split of diii_d_train evaluated once (2026-10-08); "
                      "the Final members are refit on train+val+test (--alldata)\n")
    return marker


def is_link(p):
    """True for a symlink or a Windows junction (Path.is_symlink() is False for junctions), broken or not."""
    p = Path(p)
    return p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())


def remove_link(p):
    """Remove a directory symlink / junction itself, never its target."""
    try:
        os.unlink(p)
    except OSError:
        os.rmdir(p)


def link_dir(link, target):
    """`link` -> `target` directory link (junction on Windows), as fec/run_final.py makes it, so
    models trained under the idx root land in <FEC_WORK>/models where inference reads them.
    A broken link (its target was moved or deleted) or a stale one (pointing elsewhere, e.g. after the
    work tree was moved) is removed and recreated; the old target's content is never touched.  A real
    non-empty directory in the way stops the run."""
    link, target = Path(link), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if is_link(link):
        if link.exists() and link.resolve() == target.resolve():
            return
        print(f"link {link}: {'stale, pointed elsewhere' if link.exists() else 'broken'}; recreated -> {target}",
              flush=True)
        remove_link(link)
    elif link.exists():
        if link.resolve() == target.resolve():
            return
        if any(link.iterdir()):
            raise SystemExit(f"{link} exists and is not a link to {target}")
        link.rmdir()
    elif os.path.lexists(link):         # any other dangling entry
        remove_link(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target.resolve()), str(link))
    else:
        os.symlink(target.resolve(), link, target_is_directory=True)


def li2_cache_dir(env):
    """li(2) target cache of the li2ft fine-tune groups (in the DIII-D idx work root)."""
    return Path(env["FEC_IDX_WORK"]) / "li2" / "cache_all"


def li2_groups(cfg_name):
    """Names of the config's DIII-D train groups marked "li2_cache" (fine-tune groups)."""
    spec = json.loads((PKG / "configs" / f"train_d3d_{cfg_name}.json").read_text())
    return [g["name"] for g in spec["groups"] if g.get("li2_cache")]


def stage_li2cache(a, env, cfg_name):
    """fec/li2_torch.py build (all labelled diii_d_train frames: train+val+test) + check; CPU.  No-op for configs
    without a li2_cache group.  An existing cache (meta.json, written last by the build) is reused only after
    fec/li2_torch.py check passes on it; a cache that fails the check is moved to cache_all_bad_<UTC> and rebuilt.
    (The check fails on missing/corrupt arrays, a length mismatch or a mask mismatch; it prints its relative li
    errors but has no numeric pass threshold.)"""
    groups = li2_groups(cfg_name)
    if not groups:                      # silent inside d3d/all: the other configs' runs are unchanged
        if a.stage == "li2cache":
            print(f"li2cache: {cfg_name} has no li2_cache group; nothing to build", flush=True)
        return
    out = li2_cache_dir(env)
    e = dict(env, FEC_WORK=env["FEC_IDX_WORK"])          # ds/, shots/, ds_b/env_train.npz, test_eval_b/DONE
    logs = Path(env["FEC_WORK"]) / "logs"
    check = [sys.executable, "-B", "fec/li2_torch.py", "check", "--cache", out, "--n", 300]
    if (out / "meta.json").exists():
        try:
            sh(check, e, a.dry_run, logs / "li2_cache_check.log")
        except SystemExit as ex:
            bad = out.with_name(f"{out.name}_bad_{utc_stamp()}")
            out.rename(bad)
            print(f"li2cache: {out} has meta.json but FAILS fec/li2_torch.py check ({ex}); moved to {bad}; "
                  f"rebuilding", flush=True)
        else:
            print(f"li2cache: {out} is complete (meta.json) and "
                  f"{'(dry run) would be checked' if a.dry_run else 'passed fec/li2_torch.py check'}; build skipped",
                  flush=True)
            return
    print(f"li2cache: for groups {groups}", flush=True)
    sh([sys.executable, "-B", "fec/li2_torch.py", "build", "--out", out, "--nproc", a.workers,
        "--splits", "train,val,test"], e, a.dry_run, logs / "li2_cache_build.log")
    sh(check, e, a.dry_run, logs / "li2_cache_check.log")


def aux_cache_dir(env):
    """All-row mtask aux target cache of the "aux" train groups (final_MTfull; in the DIII-D idx work root)."""
    return Path(env["FEC_IDX_WORK"]) / "mtask" / "aux_all"


def aux_groups(cfg_name):
    """Names of the config's DIII-D train groups marked "aux" (trained with the mtask auxiliary heads)."""
    spec = json.loads((PKG / "configs" / f"train_d3d_{cfg_name}.json").read_text())
    return [g["name"] for g in spec["groups"] if g.get("aux")]


def stage_auxcache(a, env, cfg_name):
    """fec/mtask_aux_all.py build (every labelled diii_d_train frame, from the all-data li2 cache) + fec/mtask_aux.py
    check; CPU.  No-op for configs without an "aux" group.  An existing cache (meta.json, written last) is reused only
    after the check passes on it; a cache that fails is moved to aux_all_bad_<UTC> and rebuilt.  Needs the li2 cache
    (stage li2cache, which --stage auxcache / d3d / all run first)."""
    groups = aux_groups(cfg_name)
    if not groups:                      # silent inside d3d/all: the other configs' runs are unchanged
        if a.stage == "auxcache":
            print(f"auxcache: {cfg_name} has no aux group; nothing to build", flush=True)
        return
    out, li2 = aux_cache_dir(env), li2_cache_dir(env)
    if not a.dry_run and not (li2 / "meta.json").exists():
        raise SystemExit(f"auxcache: the li2 cache {li2} is missing (train.py --stage li2cache first)")
    e = dict(env, FEC_WORK=env["FEC_IDX_WORK"])          # ds/, shots/, ds_b/env_train.npz
    logs = Path(env["FEC_WORK"]) / "logs"
    check = [sys.executable, "-B", "fec/mtask_aux.py", "check", "--aux", out, "--li2", li2, "--n", 300]
    if (out / "meta.json").exists():
        try:
            sh(check, e, a.dry_run, logs / "aux_cache_check.log")
        except SystemExit as ex:
            bad = out.with_name(f"{out.name}_bad_{utc_stamp()}")
            out.rename(bad)
            print(f"auxcache: {out} has meta.json but FAILS fec/mtask_aux.py check ({ex}); moved to {bad}; "
                  f"rebuilding", flush=True)
        else:
            print(f"auxcache: {out} is complete (meta.json) and "
                  f"{'(dry run) would be checked' if a.dry_run else 'passed fec/mtask_aux.py check'}; build skipped",
                  flush=True)
            return
    print(f"auxcache: for groups {groups}", flush=True)
    t0 = time.time()
    sh([sys.executable, "-B", "fec/mtask_aux_all.py", "build", "--li2", li2, "--out", out, "--nproc", a.workers],
       e, a.dry_run, logs / "aux_cache_build.log")
    sh(check, e, a.dry_run, logs / "aux_cache_check.log")
    if not a.dry_run:
        tf = logs / "timing_auxcache.json"
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text(json.dumps(dict(utc=utc_stamp(), config=cfg_name, seconds=round(time.time() - t0, 1),
                                      workers=a.workers), indent=1))


def group_tags(grp):
    """(member tags, fine-tune group?) of a train group: the --tag of every configs/jobs line after the group's
    tag_replace (as fec/train_d3d_bx.py member_argvs applies it); fine-tune = the lines carry --ft_from."""
    import shlex
    tags, ft = [], False
    for line in json.loads((PKG / "configs" / grp["jobs"]).read_text()):
        toks = shlex.split(line)
        tag = toks[toks.index("--tag") + 1]
        for r in grp.get("tag_replace", []):
            old, new = r.split("=", 1)
            tag = tag.replace(old, new)
        tags.append(tag)
        ft = ft or "--ft_from" in toks
    return tags, ft


def member_state(models, tag, ft=False):
    """'done' when <models>/<tag> holds model.pt and summary.json (+ finetune.json for a fine-tune member): the
    trainer writes them last, in that order; 'partial' for any other non-empty dir; 'todo' otherwise."""
    d = Path(models) / tag
    need = ("model.pt", "summary.json") + (("finetune.json",) if ft else ())
    if all((d / f).is_file() for f in need):
        return "done"
    return "partial" if d.is_dir() and any(d.iterdir()) else "todo"


def quarantine_group(grp, models, tags, stamp=None):
    """Move every non-empty member dir of ONE group (finished or not) to <models>/_partial_<group>_<UTC>/<tag>.
    fec/train_d3d_bx.py refuses to start when any member dir of its run is non-empty, and the group must retrain
    whole (its stacks are trained jointly, so keeping the finished members would change the stack composition).
    Same filesystem (a rename): nothing is copied, deleted or overwritten.  Returns (quarantine dir, moved tags)."""
    models = Path(models)
    base = f"_partial_{grp['name']}_{stamp or utc_stamp()}"
    q, k = models / base, 1
    while q.exists():
        k += 1
        q = models / f"{base}_{k}"
    moved = []
    for t in tags:
        d = models / t
        if d.is_dir() and any(d.iterdir()):
            q.mkdir(parents=True, exist_ok=True)
            d.rename(q / t)
            moved.append(t)
    return q, moved


def group_finished(grp, models):
    """Resume support for --stage d3d (fec/train_d3d_bx.py refuses to overwrite a non-empty model dir): True
    (skip, with a log line) when every member of the group is done; False (train) when none has started.  A group
    with only some members done, or with a model dir without a finished training (a crash; also shipped weights that
    infer.py installed in the same --work), is quarantined (quarantine_group: all of THIS group's non-empty member
    dirs move to <models>/_partial_<group>_<UTC>/; other groups are never touched) and then retrains whole (False)."""
    tags, ft = group_tags(grp)
    st = {t: member_state(models, t, ft) for t in tags}
    if all(v == "done" for v in st.values()):
        print(f"d3d group {grp['name']}: all {len(tags)} members finished (model.pt + summary.json"
              f"{' + finetune.json' if ft else ''} in {models}); skipped", flush=True)
        return True
    if all(v == "todo" for v in st.values()):
        return False
    done = [t for t, v in st.items() if v == "done"]
    part = [t for t, v in st.items() if v == "partial"]
    q, moved = quarantine_group(grp, models, tags)
    print(f"d3d group {grp['name']}: PARTIAL ({len(done)} of {len(tags)} members finished, {len(part)} model dirs "
          f"without a finished training); moved {len(moved)} member dirs to {q}; retraining the whole group "
          f"({len(tags)} members, README \"Resume\")", flush=True)
    return False


def stage_d3d(a, env, cfg_name):
    jobs = PKG / "configs" / f"train_d3d_{cfg_name}.json"
    spec = json.loads(jobs.read_text())
    e = dict(env, FEC_WORK=env["FEC_IDX_WORK"])
    if not a.dry_run and any(g.get("li2_cache") for g in spec["groups"]) and \
            not (li2_cache_dir(env) / "meta.json").exists():   # fail before hours of base training
        raise SystemExit(f"li2 cache {li2_cache_dir(env)} missing: run train.py --stage li2cache first")
    if not a.dry_run and any(g.get("aux") for g in spec["groups"]) and \
            not (aux_cache_dir(env) / "meta.json").exists():    # final_MTfull: fail before hours of base training
        raise SystemExit(f"aux cache {aux_cache_dir(env)} missing: run train.py --stage auxcache first")
    if not a.dry_run:
        link_dir(Path(env["FEC_IDX_WORK"]) / "models", Path(env["FEC_WORK"]) / "models")
    for grp in spec["groups"]:
        if not a.dry_run and group_finished(grp, Path(env["FEC_IDX_WORK"]) / "models"):
            continue
        jf = PKG / "configs" / grp["jobs"]
        cmd = [sys.executable, "-B", "fec/train_d3d_bx.py", "--jobs", jf,
               "--group_size", grp.get("group_size", a.group_size), "--pix_impl", a.pix_impl,
               "--ln_impl", a.pix_impl, "--vram_gb", a.vram_gb,
               "--timing_json", Path(env["FEC_WORK"]) / "logs" / f"timing_{grp['name']}.json"]
        for r in grp.get("tag_replace", []):
            cmd += ["--tag_replace", r]
        if grp.get("li2_cache"):         # li2ft fine-tune group (members --ft_from the groups above)
            cmd += ["--li2_cache", li2_cache_dir(env)]
        if grp.get("aux"):               # mtask auxiliary heads (final_MTfull's MT bases and fine-tunes)
            cmd += ["--aux_heads", grp["aux"]["heads"], "--aux_cache", aux_cache_dir(env), "--aux_w", grp["aux"]["w"]]
        sh(cmd, e, a.dry_run, Path(env["FEC_WORK"]) / "logs" / f"train_{grp['name']}.log")


def dec_spec(cfg_name):
    """The config's off-span decoder block (configs/<cfg>.json d3d.dec; final_A22li_dec), or None."""
    cfg = json.loads((PKG / "configs" / f"{cfg_name}.json").read_text())
    return (cfg.get("d3d") or {}).get("dec") or None


def offdec_paths(env, q):
    """(decoder dir <FEC_WORK>/<d3d.dec.dir>, CPU cache dir <FEC_IDX_WORK>/offdec_cache/<name>)."""
    return Path(env["FEC_WORK"]) / q["dir"], Path(env["FEC_IDX_WORK"]) / "offdec_cache" / Path(q["dir"]).name


def decoder_basis(path):
    """basis_sha256 recorded in a decoder file (None for a decoder that records none, e.g. the shipped Dev-read ones)."""
    import torch
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False).get("basis_sha256")
    except Exception:  # noqa: BLE001 - an unreadable decoder is retrained
        return None


def stage_offdec(a, env, cfg_name):
    """fec/offdec.py basis -> envelope -> data -> train (one run per seed) for a config with d3d.dec (see the module
    doc); no-op for every other config.  Decoders already trained in the current members' basis are kept; any other
    decoder file (another basis, none recorded, unreadable) is moved to <dir>/_stale_<UTC>/ and retrained.  Wall-clock
    seconds per step: <FEC_WORK>/logs/timing_offdec.json."""
    q = dec_spec(cfg_name)
    if not q:
        if a.stage == "offdec":
            print(f"offdec: {cfg_name} has no decoder stage (d3d.dec); nothing to do", flush=True)
        return
    cfg = json.loads((PKG / "configs" / f"{cfg_name}.json").read_text())
    W, I = Path(env["FEC_WORK"]), Path(env["FEC_IDX_WORK"])
    out, cache = offdec_paths(env, q)
    logs = W / "logs"
    ds, shots = I / "ds" / "diii_d_train", I / "shots" / "diii_d_train"
    py = [sys.executable, "-B", "fec/offdec.py"]
    timing, t00 = {}, time.time()

    def step(name, cmd, log):
        t0 = time.time()
        sh(cmd, env, a.dry_run, logs / log)
        timing[name] = round(time.time() - t0, 1)
        if not a.dry_run:
            print(f"offdec {name}: {timing[name]:.0f} s", flush=True)

    step("basis", py + ["basis", "--members-recipe", PKG / cfg["d3d"]["b"]["recipe"], "--models", W / "models",
                        "--out", cache / "member_basis.npz"], "offdec_basis.log")
    bsha = None if a.dry_run else json.loads((cache / "member_basis.json").read_text())["basis_sha256"]
    seeds = list(q["seeds"])
    if not a.dry_run:
        todo, stale = [], []
        for s in seeds:
            f = out / f"dec_s{s}.pt"
            if f.is_file() and decoder_basis(f) == bsha:
                print(f"offdec: {f} was trained in the current basis {bsha[:12]}; kept", flush=True)
                continue
            todo.append(s)
            if f.exists():
                stale.append(f)
        if stale:
            q_dir = out / f"_stale_{utc_stamp()}"
            q_dir.mkdir(parents=True, exist_ok=True)
            for f in stale:
                f.rename(q_dir / f.name)
            print(f"offdec: {len(stale)} decoder files of another (or no recorded) basis moved to {q_dir}", flush=True)
        seeds = todo
    if seeds:
        env_f = cache / "d3d_env_m4.npy"
        if a.dry_run or not (env_f.is_file() and (cache / "d3d_env_m4.json").is_file()):
            step("envelope", py + ["envelope", "--out", env_f, "--ds", ds, "--shots", shots], "offdec_envelope.log")
        meta_f = cache / "data_meta.json"
        fresh = not a.dry_run and meta_f.is_file() and (cache / "tr_r.npy").is_file() and \
            json.loads(meta_f.read_text()).get("basis_sha256") == bsha
        if not fresh:
            step("data", py + ["data", "--cache", cache, "--ds", ds, "--shots", shots, "--workers", a.workers],
                 "offdec_data.log")
        for s in seeds:
            step(f"train_s{s}", py + ["train", "--cache", cache, "--seed", s, "--out", out / f"dec_s{s}.pt",
                                      "--device", "cuda"], f"offdec_train_s{s}.log")
    if not a.dry_run:
        r = cache / "tr_r.npy"
        if r.exists() and not a.keep_offdec_cache and all((out / f"dec_s{s}.pt").is_file() for s in q["seeds"]):
            r.unlink()
            print(f"offdec: all {len(q['seeds'])} decoders present; deleted {r} (--keep-offdec-cache keeps it)",
                  flush=True)
        timing["total"] = round(time.time() - t00, 1)
        tf = logs / "timing_offdec.json"
        old = json.loads(tf.read_text()) if tf.exists() else []
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text(json.dumps(old + [dict(utc=utc_stamp(), config=cfg_name, seeds_trained=seeds, **timing)],
                                 indent=1))


def install_synthetic(src, gen, dman):
    """Copy the pinned synthetic chunks from src into gen; every data_manifest 'synthetic' file must
    be present with its SHA-256 (no partial installs, no extra files)."""
    files = dman["synthetic"]["files"]
    src, gen = Path(src), Path(gen)
    bad = [f["path"] for f in files
           if not (src / Path(f["path"]).name).exists() or sha256(src / Path(f["path"]).name) != f["sha256"]]
    if bad:
        raise SystemExit(f"synthetic set: {len(bad)} of {len(files)} files missing or with a wrong SHA-256, "
                         f"e.g. {bad[:3]}")
    gen.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(src / Path(f["path"]).name, gen / Path(f["path"]).name)
    print(f"installed {len(files)} pinned synthetic chunks into {gen}", flush=True)
    return len(files)


def npz_ok(p, keys=SYNTH_KEYS):
    """The file opens as a zip (its central directory is written last, so a chunk truncated by a killed generator
    fails) and holds the arrays `keys`."""
    try:
        with zipfile.ZipFile(p) as z:
            names = set(z.namelist())
    except (OSError, ValueError, zipfile.BadZipFile):
        return False
    return all(f"{k}.npy" in names for k in keys)


def synthetic_state(gen, dman):
    """(wanted, missing, unreadable, extra) chunk names of the synthetic set in `gen` against data_manifest.json
    'synthetic' files: every listed chunk must exist and pass npz_ok, and no unlisted chunk_*.npz may be present
    (fec/amort_net2.py trains on every chunk_*.npz of the directory).  The same rule holds for an installed set
    (install_synthetic also checked its SHA-256) and a regenerated one (not bit-identical across GPUs, so not hashed)."""
    want = [Path(f["path"]).name for f in ((dman or {}).get("synthetic") or {}).get("files") or []]
    gen = Path(gen)
    have = {p.name for p in gen.glob("chunk_*.npz")} if gen.is_dir() else set()
    missing = [n for n in want if n not in have]
    bad = [n for n in want if n in have and not npz_ok(gen / n)]
    extra = sorted(have - set(want))
    return want, missing, bad, extra


def ensure_synthetic(gen, dman, run_gen, dry=False):
    """Complete synthetic set or SystemExit.  A complete set skips the generator.  Otherwise unreadable chunks are
    moved to <amort>/_bad_<gen>_<UTC>/ and the generator runs (fec/amort_gen7.py gen keeps every existing chunk and
    makes the missing ones with their own per-chunk seeds), then the set is checked again.  Unlisted chunk files
    stop the run (a different generator configuration)."""
    gen = Path(gen)
    want, missing, bad, extra = synthetic_state(gen, dman)
    if not want:
        if dry:                          # (dry run of a test skeleton without the synthetic file list)
            run_gen()
            return "dry"
        raise SystemExit("data_manifest.json 'synthetic' lists no chunk files: the synthetic set cannot be checked")
    if extra:
        raise SystemExit(f"synthetic set {gen}: {len(extra)} chunk files not listed in data_manifest.json "
                         f"(e.g. {extra[:3]}); fec/amort_net2.py would train on them: move them out or use a fresh "
                         f"--work")
    if not missing and not bad:
        if not dry:
            print(f"synthetic set {gen}: all {len(want)} chunks present and readable; generator skipped", flush=True)
        return "complete"
    if bad and not dry:
        q = gen.parent / f"_bad_{gen.name}_{utc_stamp()}"
        q.mkdir(parents=True, exist_ok=True)
        for n in bad:
            (gen / n).rename(q / n)
        print(f"synthetic set {gen}: {len(bad)} unreadable chunks moved to {q}", flush=True)
    if not dry and gen.exists():
        print(f"synthetic set {gen}: {len(missing) + len(bad)} of {len(want)} chunks missing or unreadable; "
              f"running the generator (it keeps the existing chunks)", flush=True)
    run_gen()
    if dry:
        return "dry"
    want, missing, bad, extra = synthetic_state(gen, dman)
    if missing or bad or extra:
        raise SystemExit(f"synthetic set {gen} incomplete after the generator: {len(missing)} missing (e.g. "
                         f"{missing[:3]}), {len(bad)} unreadable, {len(extra)} unlisted, of {len(want)} listed in "
                         f"data_manifest.json")
    print(f"synthetic set {gen}: all {len(want)} chunks present and readable after the generator", flush=True)
    return "generated"


MAST_NET = "n7h_f0.55"      # the amortized net stage_mast trains (weights_manifest mast tag, configs' maps.net)


def needs_mast_net(cfg_name):
    """True when the config's outputs read the amortized net stage_mast trains (mast_tags: its composite maps name
    it, kinds amort / vb / combo).  False for a physics-map config (final_C), whose maps, q95 and betaN all come from
    the physics average: then passive_f, the synthetic set, the net and TX1s are outputs nothing reads (R2-5)."""
    cfg = json.loads((PKG / "configs" / f"{cfg_name}.json").read_text())
    return MAST_NET in mast_tags(cfg)


def stage_mast(a, env, dman=None, cfg_name=None):
    if cfg_name and not needs_mast_net(cfg_name):
        print(f"mast: {cfg_name} reads no amortized MAST net (composite maps from the physics average): passive_f, the "
              f"synthetic set and the net training are not run (R2-5)", flush=True)
        return
    py = [sys.executable, "-B"]
    e = dict(env, FEC_AMORT_SOLF="0.55")
    logs = Path(env["FEC_WORK"]) / "logs"
    amort = Path(env["FEC_WORK"]) / "amort"
    sh(py + ["fec/final_mast.py", "passive_f", "--config", "mast_public_test", "--f", "0.55",
             "--out", amort / "passive_f0.55", "--rscales", "0.5,1,2"], e, a.dry_run,
       logs / "passive_f055.log")
    gen = amort / "gen_v7h_f0.55"
    if a.synthetic and not a.dry_run:
        install_synthetic(a.synthetic, gen, dman)
    ensure_synthetic(gen, dman, lambda: sh(py + ["fec/amort_gen7.py", "gen", "--f", "0.55", "--flavour", "h"], e,
                                           a.dry_run, logs / "gen7h_f055.log"), a.dry_run)
    sh(py + ["fec/amort_net2.py", "train", "--gen", "v7h_f0.55", "--norm", "--tag", "n7h_f0.55",
             "--seeds", "3"], e, a.dry_run, logs / "train_n7h_f055.log")
    if cfg_name and needs_tx1s(cfg_name):
        stage_tx1s(a, env)


def needs_tx1s(cfg_name):
    """True for configs whose MAST maps use the TX1s network (composite maps kind vb / combo)."""
    cfg = json.loads((PKG / "configs" / f"{cfg_name}.json").read_text())
    return ((cfg.get("mast") or {}).get("composite") or {}).get("maps", {}).get("kind") in ("vb", "combo")


def stage_tx1s(a, env):
    """TX1 (12 epochs on DIII-D TRAIN) -> TX1s (4 epochs + the own-solver synthetic MAST curriculum): the Dev-read
    recipe step by step through the original CLIs (fec/final_tx1s.py train; GPU ~0.3 A100-PCIe-h).  Reads the DIII-D
    shot caches of the idx root (linked as <FEC_WORK>/shots/diii_d_train), the frozen xfer artefacts and the MAST
    passive currents + synthetic chunks made above.  Model: <FEC_WORK>/xfer/models1/tx1s."""
    W, I = Path(env["FEC_WORK"]), Path(env["FEC_IDX_WORK"])
    if not a.dry_run:
        link_dir(W / "shots" / "diii_d_train", I / "shots" / "diii_d_train")
    sh([sys.executable, "-B", "fec/final_tx1s.py", "train"], env, a.dry_run, W / "logs" / "train_tx1s.log")


def export_required(rec, tags, mtags, dtags=()):
    """True for a weights_manifest row that infer.py install_weights needs for the config (it stops on such a row
    when the file is missing): the recipe's DIII-D members + fallback (`tags`), the MAST networks the config reads
    (`mtags` = infer.mast_tags: its maps net, plus TX1s for vb / combo; none for a physics-map config) and the
    off-span decoders of a d3d.dec config (`dtags` = infer.dec_tags; empty otherwise).  mast_cache rows are rebuilt on
    demand and rows of other configs' members are optional."""
    g = rec.get("group")
    if g == "d3d":
        return rec.get("tag") in tags
    if g == "mast":
        return rec.get("tag") in mtags
    if g == "offdec":
        return rec.get("tag") in dtags
    return False


def stage_export(a, env, cfg_name):
    """weights_out/ in the weights_manifest layout + SHA-256 list (compare with the manifest).  Stops BEFORE copying
    anything when a file the config needs (export_required) is missing, listing every missing path; files the config
    does not need are exported when present (as before)."""
    wman = json.loads((PKG / "weights_manifest.json").read_text())
    out = Path(a.work) / "weights_out"
    if a.dry_run:
        print(f"(dry run) would export the trained files listed in weights_manifest.json to {out}")
        return
    cfg = json.loads((PKG / "configs" / f"{cfg_name}.json").read_text())
    tags, mtags, dtags = recipe_tags(cfg), mast_tags(cfg), dec_tags(cfg)
    roots = {"fec_work": Path(env["FEC_WORK"]), "fec_work_idx": Path(env["FEC_IDX_WORK"])}
    plan, missing = [], []
    for rec in wman["files"]:
        root, sub = rec["dest"].split("/", 1)
        src = roots[root] / sub
        if src.is_file():
            plan.append((rec, src))
        elif export_required(rec, tags, mtags, dtags):
            missing.append(str(src))
    if missing:
        raise SystemExit(f"export {cfg_name}: {len(missing)} weight files the config needs are missing (training "
                         f"incomplete?); nothing exported:\n  " + "\n  ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec, src in plan:
        dst = out / rec["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        rows.append(dict(path=rec["path"], sha256=sha256(dst), manifest_sha256=rec["sha256"]))
    (out / "sha256.json").write_text(json.dumps(rows, indent=1))
    print(f"exported {len(rows)} files to {out} (retrained files differ from the manifest "
          f"hashes unless trained on the same GPU/stack)")


# ============================================================================ plumbing stages (reproduce.sh)
def dir_bytes(root):
    """Bytes of the regular files under `root`; symlinks and junctions are not followed (no double counting of
    the <idx>/models -> <FEC_WORK>/models link)."""
    tot, stack = 0, [str(root)]
    while stack:
        try:
            it = os.scandir(stack.pop())
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_symlink() or (hasattr(e, "is_junction") and e.is_junction()):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        tot += e.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    return tot


def preflight_disk(work, peak_gb=None, margin=DISK_MARGIN, log=print):
    """Free space under `work` for a full rebuild.  peak_gb None (PEAK_DISK_GB not set yet): a warning only.
    Otherwise SystemExit unless free + already used under `work` >= margin x peak_gb (a resumed run is credited with
    what it already holds)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work).free / 1e9
    if peak_gb is None:
        log(f"WARNING disk preflight: {free:.0f} GB free under {work}; the measured peak of a full rebuild is not set "
            f"yet (train.py PEAK_DISK_GB = None), so this check only warns")
        return dict(status="warn", free_gb=round(free, 1))
    used = dir_bytes(work) / 1e9
    need = margin * peak_gb - used
    if free < need:
        raise SystemExit(f"disk preflight FAILED: {free:.0f} GB free under {work}, {need:.0f} GB needed ({margin} x "
                         f"the measured peak {peak_gb} GB, minus {used:.0f} GB already under the work dir); free "
                         f"space or choose another --work / WORK")
    log(f"disk preflight: {free:.0f} GB free under {work} >= {need:.0f} GB needed ({margin} x peak {peak_gb} GB "
        f"- {used:.0f} GB already used): OK")
    return dict(status="ok", free_gb=round(free, 1), need_gb=round(need, 1), used_gb=round(used, 1))


def stage_download(a, env):
    """Public training configs at the dataset revision pinned in data_manifest.json (fec/final_data.py download
    reads the pin and records the resolved commit in <FEC_WORK>/hf_data/revisions.json).  The data stage's own
    download call then finds them complete without a network call."""
    sh([sys.executable, "-B", "fec/final_data.py", "download", *TRAIN_CONFIGS], env, a.dry_run,
       Path(env["FEC_WORK"]) / "logs" / "download.log")


def input_state(env, cfg):
    """(complete?, reason) of one input config, without any network call: complete when the data root holds its
    parquet files (as fec.paths.shot_files looks there first), or when <FEC_WORK>/hf_data/revisions.json records a
    finished download of it whose file list equals the local one (fec/final_data.py writes the record only after
    the local list matched the remote listing)."""
    base = Path(env.get("FEC_PKG_WORK") or PKG / "work")
    data = Path(env["FUSION_EXTERNAL_DATA_ROOT"]) if env.get("FUSION_EXTERNAL_DATA_ROOT") else base / "data"
    if any((data / "fusion-equilibrium-challenge" / "data" / cfg).glob("*.parquet")):
        return True, "data root"
    hf = Path(env["FEC_WORK"]) / "hf_data"
    have = sorted(p.name for p in (hf / "data" / cfg).glob("*.parquet"))
    try:
        rec = json.loads((hf / "revisions.json").read_text()).get(cfg) or {}
    except (OSError, ValueError):
        rec = {}
    if have and rec.get("files") == have:
        return True, f"{len(have)} files, recorded at {str(rec.get('revision'))[:12]}"
    if not have:
        return False, "missing"
    return False, f"incomplete ({len(have)} local files, no matching download record)"


def stage_inputs(a, env):
    """The --split fold's inputs into <FEC_WORK>/hf_data/data/<config>, where infer.py's download stage finds them
    (private_test: the Final dataset ganatma/fusion-equilibrium-challenge-final at the data_manifest.json pin; --split
    public_test rehearses on the Development fold).  Configs already complete locally (input_state) are not downloaded again.  --local-only checks
    without any network access and stops when a config is missing or incomplete (reproduce.sh full runs it before the
    offline inference, then retries the download online for that one step)."""
    cfgs = [f"diii_d_{a.split}", f"mast_{a.split}"]
    todo = []
    for c in cfgs:
        ok, why = input_state(env, c)
        print(f"inputs {c}: {'complete (' + why + ')' if ok else why}", flush=True)
        if not ok:
            todo.append(c)
    if not todo:
        return
    if getattr(a, "local_only", False):
        raise SystemExit(f"inputs {todo} missing or incomplete in {Path(env['FEC_WORK']) / 'hf_data'} (--local-only)")
    sh([sys.executable, "-B", "fec/final_data.py", "download", *todo], env, a.dry_run,
       Path(env["FEC_WORK"]) / "logs" / f"download_{a.split}.log")


def nvidia_smi():
    """[{name, driver_version, memory_total_mib}] from nvidia-smi, or None (no NVIDIA driver / tool)."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode:
        return None
    rows = []
    for ln in p.stdout.strip().splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) == 3:
            rows.append(dict(name=f[0], driver_version=f[1], memory_total_mib=f[2]))
    return rows or None


def config_files(cfg_name):
    """Package files that define the config: configs/<cfg>.json, its DIII-D recipe, configs/train_d3d_<cfg>.json and
    its job lists."""
    rel = [f"configs/{cfg_name}.json"]
    cfg = json.loads((PKG / rel[0]).read_text())
    r = ((cfg.get("d3d") or {}).get("b") or {}).get("recipe")
    if r:
        rel.append(r)
    t = PKG / "configs" / f"train_d3d_{cfg_name}.json"
    if t.exists():
        rel.append(f"configs/train_d3d_{cfg_name}.json")
        rel += sorted({f"configs/{g['jobs']}" for g in json.loads(t.read_text())["groups"]})
    return [x for x in rel if (PKG / x).is_file()]


def run_fingerprint(work, cfg_name, mode, split):
    """The run fingerprint: interpreter, torch / CUDA / driver / GPU, package and source commits, SHA-256 of the
    config files and manifests, pinned and downloaded dataset revisions, the weights revision and the cache /
    offline settings.  Never a token (only whether HF_TOKEN is set)."""
    fp = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), mode=mode, config=cfg_name, split=split,
              python=platform.python_version(), platform=platform.platform())
    try:
        import torch
        fp.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            fp["gpu_torch"] = dict(name=p.name, memory_gb=round(p.total_memory / 1e9, 1),
                                   count=torch.cuda.device_count())
    except Exception as ex:  # noqa: BLE001
        fp["torch"] = f"not importable ({type(ex).__name__})"
    fp["nvidia_smi"] = nvidia_smi()
    try:
        g = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PKG), capture_output=True, text=True, timeout=60)
        fp["package_commit"] = g.stdout.strip() if g.returncode == 0 and g.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        fp["package_commit"] = None
    man = PKG / "MANIFEST.json"
    fp["source_git_sha"] = json.loads(man.read_text()).get("source_git_sha") if man.exists() else None
    fp["config_sha256"] = {f: sha256(PKG / f) for f in config_files(cfg_name)}
    for f in ("data_manifest.json", "weights_manifest.json"):
        fp[f.replace(".", "_") + "_sha256"] = sha256(PKG / f) if (PKG / f).exists() else None
    dman = json.loads((PKG / "data_manifest.json").read_text()) if (PKG / "data_manifest.json").exists() else {}
    fp["datasets_pinned"] = {r: v.get("revision") for r, v in (dman.get("datasets") or {}).items()}
    rec = Path(work) / "fec_work" / "hf_data" / "revisions.json"
    fp["datasets_downloaded"] = {c: {k: v.get(k) for k in ("repo", "revision", "pinned", "n")}
                                 for c, v in json.loads(rec.read_text()).items()} if rec.exists() else {}
    wm = PKG / "weights_manifest.json"
    if wm.exists():
        w = json.loads(wm.read_text())
        fp["weights"] = dict(repository=w.get("repository"), revision=w.get("revision"))
    fp["env"] = {k: os.environ.get(k) for k in ("HF_HUB_DISABLE_XET", "HF_HUB_OFFLINE", "HF_HOME", "HF_HUB_CACHE",
                                                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TMPDIR",
                                                "CUDA_VISIBLE_DEVICES")}
    fp["hf_token_set"] = bool(os.environ.get("HF_TOKEN"))
    return fp


def stage_fingerprint(a):
    """Write <work>/run_fingerprint.json and print it.  An earlier fingerprint of another config / package commit /
    config hash is kept as run_fingerprint_<its UTC>.json, with a warning (use a fresh --work per config)."""
    work = Path(a.work)
    fp = run_fingerprint(work, a.config, a.mode, a.split)
    if a.dry_run:
        print(json.dumps(fp, indent=1), flush=True)
        return fp
    work.mkdir(parents=True, exist_ok=True)
    f = work / "run_fingerprint.json"
    if f.exists():
        old = json.loads(f.read_text())
        diff = [k for k in ("config", "package_commit", "config_sha256", "datasets_pinned", "weights")
                if old.get(k) != fp.get(k)]
        if diff:
            keep = work / f"run_fingerprint_{str(old.get('utc', 'old')).replace(':', '')}.json"
            shutil.copyfile(f, keep)
            print(f"WARNING run fingerprint: {work} was used before with a different {diff} (kept as {keep.name}); "
                  f"use a fresh --work per config", flush=True)
    f.write_text(json.dumps(fp, indent=1))
    print(f"run fingerprint ({f}):\n" + json.dumps(fp, indent=1), flush=True)
    return fp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["data", "li2cache", "auxcache", "d3d", "offdec", "mast", "all",
                                                       "export", "preflight", "download", "inputs", "fingerprint"])
    ap.add_argument("--split", default="private_test", choices=["private_test", "public_test"],
                    help="fold of --stage inputs / fingerprint (reproduce.sh: SPLIT)")
    ap.add_argument("--mode", default="full", choices=["full", "inference"], help="recorded by --stage fingerprint")
    ap.add_argument("--local-only", dest="local_only", action="store_true",
                    help="--stage inputs: check the local inputs only (no network); exit 1 when one is missing")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work", default=str(PKG / "work"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--pix-impl", default="compile" if os.name != "nt" else "eager",
                    choices=["compile", "eager"])
    ap.add_argument("--vram-gb", dest="vram_gb", type=float, default=76)
    ap.add_argument("--synthetic", default=None,
                    help="directory with the pinned synthetic chunks (weights repository "
                         "synthetic/gen_v7h_f0.55/); skips the generator")
    ap.add_argument("--keep-offdec-cache", dest="keep_offdec_cache", action="store_true",
                    help="--stage offdec: keep <FEC_IDX_WORK>/offdec_cache/<name>/tr_r.npy (6.6 GB) after training")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    env = env_for(a.work, a.data_root, "cuda")
    if a.stage == "preflight":
        if not a.dry_run:
            preflight_disk(a.work, peak_disk_gb(a.config))
        return 0
    if a.stage == "fingerprint":
        stage_fingerprint(a)
        return 0
    if a.stage in ("download", "inputs"):
        (stage_download if a.stage == "download" else stage_inputs)(a, env)
        return 0
    dman = json.loads((PKG / "data_manifest.json").read_text())
    if not a.dry_run:
        if a.stage == "all":
            preflight_disk(a.work, peak_disk_gb(a.config))
        install_frozen(a.work, dman)
    if a.stage in ("data", "all"):
        stage_data(a, env)
    if a.stage in ("li2cache", "auxcache", "d3d", "all"):   # d3d: builds the li2 cache first if a group needs it
        stage_li2cache(a, env, a.config)
    if a.stage in ("auxcache", "d3d", "all"):        # final_MTfull: the aux cache (from the li2 cache) before d3d
        stage_auxcache(a, env, a.config)
    if a.stage in ("d3d", "all"):
        stage_d3d(a, env, a.config)
    if a.stage in ("d3d", "offdec", "all"):          # d3d.dec configs (final_A22li_dec): the decoders right after
        stage_offdec(a, env, a.config)              # their members; a silent no-op for every other config
    if a.stage in ("mast", "all"):
        stage_mast(a, env, dman, a.config)
    if a.stage in ("export", "all"):
        stage_export(a, env, a.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
