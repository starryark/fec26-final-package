"""Submission tooling: validate npz -> upload to a private HF dataset repo -> pointer zip -> Codabench.

    python -B fec/submit.py validate <npz> --config diii_d_public_test
    python -B fec/submit.py upload --files A.npz B.npz [--repo starrydark/fec-preds] [--confirm]
    python -B fec/submit.py pointer --repo R --revision SHA40 --out submission_pointer.zip
    python -B fec/submit.py codabench list
    python -B fec/submit.py codabench submit <zip> [--phase dev] [--confirm]
    python -B fec/submit.py submissions      # print the submissions table from the registry

Every command that writes remotely (upload, codabench submit) is a dry run unless --confirm is
given. The pointer read token is taken only from env FEC_HF_READ_TOKEN, never from argv.
Each confirmed submission is recorded in fec_work/submissions/index.jsonl (Optimize.md A5);
`codabench list` fills its scores and `submissions` regenerates the family-doc table.

Codabench facts (read-only API GETs, 2026-10-07): competition 17739 (the 17456 in the starter
README returns 404; the organizer resource hub links 17739), phases Development 29580 / task 35328
and Final 29581 / task 35329, leaderboard submission_rule Force_Last. The upload sequence mirrors
the Codabench web client (codalab/codabench src/static/js/ours/client.js create_dataset and
submission_upload.tag): POST /api/datasets/ -> PUT presigned URL (Content-Type application/zip)
-> PUT /api/datasets/completed/<key>/ -> POST /api/submissions/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import WORK  # noqa: E402

GRID = (65, 65)
SCALARS = ("q95", "betaN")
KINDS = ("psirz",) + SCALARS
PUBLIC_FILES = ("diii_d_public_test.npz", "mast_public_test.npz")
PRIVATE_FILES = ("diii_d_private_test.npz", "mast_private_test.npz")
SCORER_FILES = PUBLIC_FILES + PRIVATE_FILES
DEFAULT_REPO = "starrydark/fec-preds"
READ_TOKEN_ENV = "FEC_HF_READ_TOKEN"
SHA40 = re.compile(r"[0-9a-f]{40}")
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")

CODABENCH = "https://www.codabench.org"
COMPETITION = 17739
# name -> (phase id, task id, files the scorer looks for); checked against the live API before submit.
PHASES = {"dev": ("Development", 29580, 35328, PUBLIC_FILES),
          "final": ("Final", 29581, 35329, PRIVATE_FILES)}
DEFAULT_PROFILE = Path(os.environ.get(
    "FEC_CODABENCH_PROFILE",
    str(Path(os.environ.get("APPDATA", "")) / "Mozilla/Firefox/Profiles/knfwcx7u.default-release")))
# One submission registry (Optimize.md A5): one JSON line per submission; `codabench submit
# --confirm` appends, `codabench list` fills the scores, `submissions` prints the family-doc table.
SUBMISSIONS_INDEX = WORK / "submissions" / "index.jsonl"

TOKEN_STEPS = """\
Create the scorer's read token by hand (Hugging Face has no API that mints a user access token
with a normal user token; huggingface_hub 1.28 exposes only whoami/OAuth device login):
  1. https://huggingface.co/settings/tokens -> "Create new token" -> token type "Fine-grained".
  2. Name it e.g. fec-scorer-read. Leave every User/Org permission unticked.
  3. Under "Repositories permissions" search and select {repo}
     (the repo must already exist: run `upload --create-only --confirm` or a real upload first),
     tick ONLY "Read access to contents of selected repos".
  4. Create, copy the hf_... value once, and set it in the shell that runs `pointer`:
       PowerShell:  $env:FEC_HF_READ_TOKEN = "hf_..."
  5. Revoke it after the competition. It is stored inside every submitted pointer zip."""


def sha256_file(path, chunk=1 << 23):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def _fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


# ----------------------------------------------------------------------------- validate

def expected_lengths(config, work=None):
    """Per-shot frame counts T_i for `config` from <work>/ds/<config>/{info.json,shot.npy,fidx.npy}."""
    import numpy as np
    d = Path(work or WORK) / "ds" / config
    info = json.loads((d / "info.json").read_text())
    n = len(info["order"])
    shot = np.load(d / "shot.npy")
    fidx = np.load(d / "fidx.npy")
    if shot.ndim != 1 or shot.shape != fidx.shape:
        raise ValueError(f"{d}: shot/fidx shape mismatch {shot.shape} {fidx.shape}")
    if len(shot) and (shot.min() < 0 or shot.max() >= n or np.any(np.diff(shot) < 0)):
        raise ValueError(f"{d}: shot.npy is not sorted in [0, {n})")
    if "N" in info and int(info["N"]) != len(shot):
        raise ValueError(f"{d}: info N={info['N']} != {len(shot)} frames")
    t = np.bincount(shot, minlength=n)
    start = np.concatenate([[0], np.cumsum(t)[:-1]])
    if not np.array_equal(fidx, np.arange(len(shot)) - start[shot]):
        raise ValueError(f"{d}: fidx is not 0..T-1 within each shot")
    return [int(x) for x in t]


def _npy_header(fp):
    import numpy as np
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
    else:
        raise ValueError(f"unsupported .npy version {version}")
    return shape, fortran, dtype


def _rounded_decimals(a):
    """Decimal places `a` was rounded to before storage, else None (fixed-decimal rounding guts
    MAST Consistency; float16 does not). Only values whose own representation error eps*|x| is far
    below the step are informative, so the test also works after a float16 cast."""
    import numpy as np
    a = np.asarray(a)
    eps = float(np.finfo(a.dtype).eps) if a.dtype.kind == "f" else 0.0
    x = a.astype(np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 64 or not np.any(x):
        return None
    tol = eps * np.abs(x) + 1e-12
    for d in range(1, 6):
        step = 10.0 ** (-d)
        keep = tol < step / 8
        if np.count_nonzero(keep) < 64:
            continue
        xi = x[keep]
        if np.all(np.abs(xi - np.round(xi / step) * step) <= tol[keep]):
            return d
    return None


def validate_npz(path, config, work=None, psi_dtype="float16", headers_only=False, quiet=False):
    """Check one submission npz without loading it whole. Returns a summary dict ('ok' key)."""
    import numpy as np
    path = Path(path)
    errors, warnings = [], []
    t_list = expected_lengths(config, work)
    n = len(t_list)
    want = {"psirz": np.dtype(psi_dtype), "q95": np.dtype("float32"), "betaN": np.dtype("float32")}
    stats = {k: {"min": math.inf, "max": -math.inf, "nonfinite": 0, "values": 0} for k in KINDS}
    if path.name != f"{config}.npz":
        warnings.append(f"file name {path.name!r} != {config + '.npz'!r}; the scorer finds files by name")
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]
        dup = [k for k, c in Counter(names).items() if c > 1]
        if dup:
            errors.append(f"duplicate zip members: {dup[:5]}")
        bad = [x for x in names if not x.endswith(".npy") or "/" in x or "\\" in x]
        if bad:
            errors.append(f"non-root or non-.npy members: {bad[:5]}")
        keys = {x[:-4] for x in names if x.endswith(".npy")}
        expected = {f"shot_{i:04d}_{k}" for i in range(n) for k in KINDS}
        missing, extra = sorted(expected - keys), sorted(keys - expected)
        if missing:
            errors.append(f"{len(missing)} missing keys, e.g. {missing[:5]}")
        if extra:
            errors.append(f"{len(extra)} unexpected keys, e.g. {extra[:5]}")
        compress = Counter({zipfile.ZIP_STORED: "stored", zipfile.ZIP_DEFLATED: "deflated"}.get(
            i.compress_type, str(i.compress_type)) for i in infos)
        by_name = {i.filename: i for i in infos}
        frames = 0
        for i in range(n):
            T = t_list[i]
            for kind in KINDS:
                key = f"shot_{i:04d}_{kind}"
                info = by_name.get(key + ".npy")
                if info is None:
                    continue
                shape_want = (T,) + GRID if kind == "psirz" else (T,)
                with zf.open(info) as fp:
                    try:
                        shape, _, dtype = _npy_header(fp)
                    except Exception as exc:  # malformed member
                        errors.append(f"{key}: bad .npy header ({type(exc).__name__}: {exc})")
                        continue
                    if dtype.hasobject:
                        errors.append(f"{key}: object dtype (needs pickle)")
                        continue
                    if tuple(shape) != shape_want:
                        errors.append(f"{key}: shape {tuple(shape)} != {shape_want}")
                    if dtype != want[kind]:
                        errors.append(f"{key}: dtype {dtype.str} != {want[kind].str}")
                    if kind == "psirz":
                        frames += shape[0] if shape else 0
                    if headers_only or tuple(shape) != shape_want or dtype.kind != "f":
                        continue
                    nbytes = math.prod(shape) * dtype.itemsize
                    buf = fp.read(nbytes)
                    if len(buf) != nbytes:
                        errors.append(f"{key}: truncated data ({len(buf)} of {nbytes} bytes)")
                        continue
                    if fp.read(1):
                        errors.append(f"{key}: trailing bytes after array data")
                    a = np.frombuffer(buf, dtype=dtype)
                    fin = np.isfinite(a)
                    s = stats[kind]
                    s["values"] += a.size
                    bad_n = int(a.size - np.count_nonzero(fin))
                    if bad_n:
                        s["nonfinite"] += bad_n
                        errors.append(f"{key}: {bad_n} non-finite values")
                    if a.size and bad_n < a.size:
                        good = a[fin] if bad_n else a
                        s["min"] = min(s["min"], float(good.min()))
                        s["max"] = max(s["max"], float(good.max()))
                    if kind == "psirz" and i < 3:
                        d = _rounded_decimals(a)
                        if d is not None:
                            warnings.append(f"{key}: values look rounded to {d} decimals (costs MAST "
                                            "Consistency; cast to float16 instead of np.round)")
    summary = {
        "file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "config": config, "shots": n, "frames_expected": sum(t_list), "frames_in_psirz": frames,
        "members": len(names), "compression": dict(compress), "psi_dtype": psi_dtype,
        "data_scanned": not headers_only,
        "stats": {k: (v if v["values"] else None) for k, v in stats.items()},
        "errors": len(errors), "warnings": len(warnings), "ok": not errors,
    }
    if not quiet:
        print(f"{path.name}: config={config} shots={n} frames={sum(t_list)} members={len(names)} "
              f"compression={dict(compress)} size={summary['bytes']} bytes")
        for k, v in summary["stats"].items():
            if v:
                print(f"  {k:6s} min={v['min']:.6g} max={v['max']:.6g} values={v['values']} "
                      f"nonfinite={v['nonfinite']}")
        for w in warnings[:10]:
            print(f"  WARNING: {w}")
        for e in errors[:40]:
            print(f"  ERROR: {e}")
        if len(errors) > 40:
            print(f"  ... and {len(errors) - 40} more errors")
        print(f"  sha256 {summary['sha256']}")
        print(("OK" if summary["ok"] else "FAILED") + f" ({len(errors)} errors, {len(warnings)} warnings"
              + (", headers only: finiteness NOT checked)" if headers_only else ")"))
    summary["error_list"], summary["warning_list"] = errors, warnings
    return summary


def cmd_validate(args):
    try:
        s = validate_npz(args.npz, args.config, args.work, args.psi_dtype, args.headers_only)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return _fail(f"{args.npz}: {type(exc).__name__}: {exc}")
    if args.json:
        print(json.dumps({k: v for k, v in s.items() if k not in ("error_list",)}, default=str))
    return 0 if s["ok"] else 1


# ----------------------------------------------------------------------------- registry

def read_index():
    rows = []
    if SUBMISSIONS_INDEX.exists():
        for line in SUBMISSIONS_INDEX.read_text(errors="replace").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def write_index(rows):
    SUBMISSIONS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    tmp = SUBMISSIONS_INDEX.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(tmp, SUBMISSIONS_INDEX)


def _git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=str(Path(__file__).resolve().parents[1]),
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def record_submission(zpath, kind, detail, sub, phase_name):
    """Append the submission registry line after a successful Codabench POST.
    Never fails the submit itself: a registry problem prints a loud warning."""
    try:
        m = re.search(r"sub(\d+)", Path(zpath).name)
        row = {"n": int(m.group(1)) if m else None,
               "codabench_id": sub.get("id"),
               "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "phase": phase_name, "route": kind, "zip": Path(zpath).name,
               "zip_sha256": sha256_file(zpath),
               "hf_revision": detail["manifest"]["revision"] if kind == "pointer" else None,
               "d3d_sha256": None, "mast_sha256": None,
               "git_sha": _git_sha(), "recipe": {}, "scores": {}, "decision": ""}
        SUBMISSIONS_INDEX.parent.mkdir(parents=True, exist_ok=True)
        with open(SUBMISSIONS_INDEX, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"recorded submission row in {SUBMISSIONS_INDEX} (fill recipe/decision by editing it)")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: submission made but NOT recorded in {SUBMISSIONS_INDEX}: {e!r}"
              "\n  append the row by hand (n, codabench_id, utc, route, zip_sha256, scores).")


def fill_scores(subs):
    """Fill empty `scores` of registry rows from a `codabench list` submissions payload."""
    by_id = {s.get("id"): s for s in subs}
    rows = read_index()
    changed = 0
    for r in rows:
        s = by_id.get(r.get("codabench_id"))
        if not s or r.get("scores"):
            continue
        r["scores"] = {str(sc.get("column_key", sc.get("index"))): sc.get("score")
                       for sc in (s.get("scores") or []) if sc.get("score") is not None}
        if r["scores"]:
            changed += 1
    if changed:
        write_index(rows)
        print(f"filled scores for {changed} submission row(s) in {SUBMISSIONS_INDEX}")
    return changed


def cmd_submissions(_args):
    """Print the family-doc submissions table from the registry (markdown)."""
    rows = read_index()
    if not rows:
        print(f"no rows in {SUBMISSIONS_INDEX}")
        return 0
    print("| sub | codabench | utc | route | DIII-D | MAST | recipe (d3d / mast) | decision |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r.get("n") or 0, r.get("codabench_id") or 0)):
        sc = r.get("scores") or {}
        rec = r.get("recipe") or {}
        d3d = sc.get("d3d_S", sc.get("S"))
        mast = sc.get("mast_S")
        cells = [f"sub{r.get('n', 0):02d}" if r.get("n") else "-",
                 str(r.get("codabench_id") or "-"),
                 str(r.get("utc") or "-"),
                 (f"pointer@{(r.get('hf_revision') or '')[:8]}" if r.get("hf_revision")
                  else str(r.get("route") or "-")),
                 f"{d3d:.5f}" if isinstance(d3d, (int, float)) else "-",
                 f"{mast:.4f}" if isinstance(mast, (int, float)) else "-",
                 f"{rec.get('d3d', '-')} / {rec.get('mast', '-')}",
                 str(r.get("decision") or "-")]
        print("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    return 0


# ----------------------------------------------------------------------------- upload

def _hf_role(who):
    at = (who.get("auth") or {}).get("accessToken") or {}
    return at.get("role"), at


def cmd_upload(args):
    if args.confirm and args.dry_run:
        return _fail("--confirm and --dry-run are mutually exclusive")
    dry = not args.confirm
    if not REPO_ID.fullmatch(args.repo):
        return _fail(f"bad repo id {args.repo!r}")
    files = [Path(f) for f in (args.files or [])]
    if not files and not args.create_only:
        return _fail("--files is required unless --create-only")
    names = [p.name for p in files]
    if len(set(names)) != len(names):
        return _fail(f"duplicate file names {names}")
    for p in files:
        if p.name not in SCORER_FILES:
            return _fail(f"{p.name} is not a scorer file name {SCORER_FILES}")
        if not p.is_file():
            return _fail(f"{p} not found")
    if files and not (set(names) <= set(PUBLIC_FILES) or set(names) <= set(PRIVATE_FILES)):
        return _fail("mixing public_test and private_test files in one commit")
    records = []
    for p in files:
        config = p.stem
        has_ref = (Path(args.work or WORK) / "ds" / config).is_dir()
        if not args.skip_validate and p.name in PUBLIC_FILES and not has_ref:
            return _fail(f"no local reference {Path(args.work or WORK) / 'ds' / config}; cannot validate")
        if not args.skip_validate and has_ref:
            print(f"validating {p} ...")
            s = validate_npz(p, config, args.work, args.psi_dtype)
            if not s["ok"]:
                return _fail(f"{p.name} failed validation; nothing uploaded")
            records.append({"path_in_repo": p.name, "bytes": s["bytes"], "sha256": s["sha256"]})
        else:
            if not args.skip_validate:
                print(f"NOTE: no local reference for {config}; {p.name} not validated")
            records.append({"path_in_repo": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)})

    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    api = HfApi()
    who = api.whoami()
    role, _ = _hf_role(who)
    print(f"HF login: {who.get('name')} (token role {role})")
    if args.repo.split("/")[0] not in {who.get("name"), *(o.get("name") for o in who.get("orgs", []))}:
        return _fail(f"{args.repo} is outside your namespaces")
    try:
        info = api.repo_info(args.repo, repo_type="dataset")
        exists, private = True, bool(info.private)
    except RepositoryNotFoundError:
        exists, private = False, None
    print(f"repo {args.repo}: " + (f"exists, private={private}" if exists else "does not exist"))
    if exists and not private and not args.allow_public:
        return _fail(f"{args.repo} is PUBLIC; refusing (make it private or pass --allow-public)")
    for r in records:
        print(f"  {r['path_in_repo']:28s} {r['bytes']:>14d} B  sha256 {r['sha256']}")
    if dry:
        print(f"\nDRY RUN: would {'reuse' if exists else 'create PRIVATE dataset repo'} {args.repo}"
              + ("" if args.create_only else f" and upload {len(records)} file(s) to its root in one commit")
              + ". Re-run with --confirm to do it.")
        return 0
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(args.repo, repo_type="dataset")
    if not info.private and not args.allow_public:
        return _fail(f"{args.repo} is PUBLIC after create; refusing to upload")
    if args.create_only:
        print(f"PRIVATE dataset repo {args.repo} ready; nothing uploaded.\n")
        print(TOKEN_STEPS.format(repo=args.repo))
        return 0
    ops = [CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in files]
    commit = api.create_commit(args.repo, ops, repo_type="dataset",
                               commit_message=args.message or f"fec predictions: {', '.join(names)}")
    sha = commit.oid
    if not SHA40.fullmatch(sha or ""):
        return _fail(f"expected a 40-hex commit SHA, got {sha!r}")
    at = api.repo_info(args.repo, repo_type="dataset", revision=sha, files_metadata=True)
    remote = {s.rfilename: s for s in at.siblings}
    bad = []
    for r in records:
        s = remote.get(r["path_in_repo"])
        lfs = getattr(s, "lfs", None) if s else None
        rsha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        if s is None or s.size != r["bytes"] or (rsha and rsha != r["sha256"]):
            bad.append(r["path_in_repo"])
        r["remote_sha256"] = rsha
    if bad:
        return _fail(f"commit {sha}: remote size/sha256 mismatch or missing for {bad}")
    print(f"\nCOMMIT {sha}")
    print(json.dumps({"repo_id": args.repo, "revision": sha, "files": records}))
    print(f"\nNext: python -B fec/submit.py pointer --repo {args.repo} --revision {sha} "
          "--out submission_pointer.zip")
    return 0


# ----------------------------------------------------------------------------- pointer

def check_read_token(token, repo, revision, phase="dev"):
    """Online checks of a scorer read token. Returns (errors, warnings)."""
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
    errors, warnings = [], []
    if get_token() == token:
        return ["that is your local HF login token, not a separate read token"], warnings
    api = HfApi()
    try:
        who = api.whoami(token=token)
    except Exception as exc:
        return [f"token rejected by whoami ({type(exc).__name__})"], warnings
    role, at = _hf_role(who)
    if role != "fineGrained":
        errors.append(f"token role {role!r}; a fine-grained read-only token is required")
    else:
        fg = at.get("fineGrained") or {}
        scoped = fg.get("scoped") or []
        perms = {p for s in scoped for p in (s.get("permissions") or [])} | set(fg.get("global") or [])
        broad = sorted(p for p in perms if "write" in p or "admin" in p)
        if broad:
            errors.append(f"token grants {broad}; only repo read is needed")
        ents = {(s.get("entity") or {}).get("name") for s in scoped}
        if repo in ents:
            if len(ents) > 1:
                warnings.append(f"token is scoped to {len(ents)} entities, {repo} alone would do")
        elif repo.split("/")[0] in ents:
            warnings.append(f"token is scoped to the whole {repo.split('/')[0]!r} namespace")
        else:
            warnings.append(f"token scope {sorted(e for e in ents if e)} does not name {repo}")
    try:
        info = api.repo_info(repo, repo_type="dataset", revision=revision, token=token)
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        errors.append(f"token cannot read {repo}@{revision} ({type(exc).__name__})")
        return errors, warnings
    if info.sha and info.sha != revision:
        errors.append(f"repo_info returned sha {info.sha} != {revision}")
    root = {s.rfilename for s in info.siblings if "/" not in s.rfilename}
    need = PHASES[phase][3]
    have = [f for f in need if f in root]
    if not have:
        errors.append(f"none of {need} at the repo root at {revision}; root has {sorted(root)[:10]}")
    elif len(have) < len(need):
        warnings.append(f"only {have} at the root: Challenge 2 (G_ratio) needs both machines")
    return errors, warnings


def _write_pointer_zip(out, manifest):
    zi = zipfile.ZipInfo("manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    tmp = out.with_name(out.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as z:
        z.writestr(zi, json.dumps(manifest, indent=2))
    os.replace(tmp, out)


def cmd_pointer(args):
    if not REPO_ID.fullmatch(args.repo):
        return _fail(f"bad repo id {args.repo!r}")
    if not SHA40.fullmatch(args.revision):
        return _fail("--revision must be the 40-character lowercase commit SHA (not a branch/tag)")
    token = (os.environ.get(READ_TOKEN_ENV) or "").strip()
    if not token:
        print(TOKEN_STEPS.format(repo=args.repo), file=sys.stderr)
        return _fail(f"set {READ_TOKEN_ENV} to the fine-grained read token")
    if not token.startswith("hf_") or any(c.isspace() for c in token):
        return _fail(f"{READ_TOKEN_ENV} does not look like an hf_ token")
    out = Path(args.out)
    if out.suffix != ".zip":
        return _fail("--out must end in .zip (Codabench rejects other names)")
    if out.exists() and not args.force:
        return _fail(f"{out} exists (use --force to replace)")
    if args.offline:
        print("WARNING: --offline: token scope and repo contents NOT checked")
    else:
        errors, warnings = check_read_token(token, args.repo, args.revision, args.phase)
        for w in warnings:
            print(f"WARNING: {w}")
        if errors:
            for e in errors:
                print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(f"read token can read {args.repo}@{args.revision} and sees the {args.phase} files")
    manifest = {"repo_id": args.repo, "revision": args.revision, "token": token}
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_pointer_zip(out, manifest)
    kind, back = inspect_submission_zip(out, args.phase)
    if kind != "pointer" or back["manifest"] != manifest:
        return _fail("pointer zip read-back mismatch")
    print(json.dumps({**manifest, "token": "hf_***redacted***"}))
    print(f"wrote {out.resolve()} ({out.stat().st_size} bytes, sha256 {sha256_file(out)})")
    print("It contains the read token: do not commit it; revoke the token after the competition.")
    return 0


# ----------------------------------------------------------------------------- codabench

def inspect_submission_zip(path, phase="dev"):
    """Classify a Codabench upload: ('pointer', {...}) or ('direct', {...}). Raises ValueError."""
    path = Path(path)
    if path.suffix != ".zip":
        raise ValueError("Codabench only accepts .zip uploads")
    need = PHASES[phase][3]
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate members {names}")
        if names == ["manifest.json"]:
            m = json.loads(zf.read("manifest.json").decode("utf-8"))
            if not isinstance(m, dict) or set(m) != {"repo_id", "revision", "token"}:
                raise ValueError(f"manifest keys {sorted(m) if isinstance(m, dict) else type(m)}")
            if not REPO_ID.fullmatch(str(m["repo_id"])):
                raise ValueError(f"bad repo_id {m['repo_id']!r}")
            if not SHA40.fullmatch(str(m["revision"])):
                raise ValueError("revision is not a 40-character lowercase commit SHA")
            if not str(m["token"]).startswith("hf_"):
                raise ValueError("token does not start with hf_")
            return "pointer", {"manifest": m, "members": names}
        if "manifest.json" in names:
            raise ValueError(f"manifest.json mixed with other members {names}")
        bad = [x for x in names if x not in need]
        if bad:
            raise ValueError(f"members {bad} are not {phase} scorer files {need} at the zip root")
        if not names:
            raise ValueError("empty zip")
        sizes = {i.filename: i.file_size for i in zf.infolist()}
        return "direct", {"members": names, "sizes": sizes}


def read_cookies(profile):
    """Codabench cookies from a Firefox profile (read-only; values never printed)."""
    db = Path(profile) / "cookies.sqlite"
    q = ("SELECT host,name,value,path,isSecure FROM moz_cookies "
         "WHERE host='codabench.org' OR host LIKE '%.codabench.org'")
    try:
        with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=5) as c:
            return c.execute(q).fetchall()
    except sqlite3.OperationalError:
        with tempfile.TemporaryDirectory() as td:  # Firefox holds a lock: read a private copy
            for suffix in ("", "-wal"):
                src = Path(str(db) + suffix)
                if src.exists():
                    shutil.copy2(src, Path(td) / ("cookies.sqlite" + suffix))
            with sqlite3.connect(Path(td, "cookies.sqlite").as_uri() + "?mode=ro", uri=True) as c:
                rows = c.execute(q).fetchall()
            return rows


class Codabench:
    def __init__(self, profile):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self.requests = requests
        rows = read_cookies(profile)
        if not rows:
            raise RuntimeError(f"no codabench.org cookies in {profile}; log in with that Firefox profile")
        self.s = requests.Session()
        self.s.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])))
        for host, name, value, path, secure in rows:
            self.s.cookies.set(name, value, domain=host, path=path, secure=bool(secure))

    def get(self, path, **params):
        r = self.s.get(CODABENCH + path, params=params or None, timeout=60)
        r.raise_for_status()
        return r.json()

    def get_all(self, path, **params):
        page = self.get(path, page_size="all", **params)
        if isinstance(page, list):
            return page
        out = list(page.get("results", []))
        while page.get("next"):
            r = self.s.get(page["next"], timeout=60)
            r.raise_for_status()
            page = r.json()
            out += page.get("results", [])
        return out

    def _unsafe(self, method, path, payload=None):
        csrf = self.s.cookies.get("csrftoken")
        if not csrf:
            raise RuntimeError("no csrftoken cookie; reload codabench.org in Firefox")
        headers = {"X-CSRFToken": csrf, "Referer": f"{CODABENCH}/competitions/{COMPETITION}/",
                   "Origin": CODABENCH, "X-Requested-With": "XMLHttpRequest",
                   "Content-Type": "application/json", "Accept": "application/json"}
        data = None if payload is None else json.dumps(payload)
        r = self.s.request(method, CODABENCH + path, data=data, headers=headers, timeout=120)
        if not r.ok:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:500]}")
        try:
            return r.json() if r.content else None
        except ValueError:
            return {"_text": r.text[:200]}


def _score_text(scores):
    return " ".join(f"{s.get('column_key', s.get('index'))}={s.get('score')}"
                    for s in sorted(scores or [], key=lambda s: s.get("index", 0)))


def phase_report(cb, comp):
    rows = []
    for p in comp["phases"]:
        try:
            can = cb.get(f"/api/can_make_submission/{p['id']}/")
        except Exception as exc:
            can = {"can": None, "reason": type(exc).__name__}
        rows.append({k: p.get(k) for k in ("id", "name", "status", "start", "end", "max_submissions_per_day",
                                            "max_submissions_per_person", "used_submissions_per_day",
                                            "used_submissions_per_person")}
                    | {"tasks": [t.get("id") for t in p.get("tasks", [])], "can_submit": can.get("can"),
                       "reason": can.get("reason")})
    return rows


def my_leaderboard(cb, phase_id, me):
    try:
        board = cb.get(f"/api/phases/{phase_id}/get_leaderboard/", page_size="all")
    except Exception as exc:
        return {"error": type(exc).__name__}
    for rank, e in enumerate(board.get("submissions", []), 1):
        if e.get("owner") == me:
            return {"rank": rank, "of": board.get("count"), "submission_id": e.get("id"),
                    "created_when": e.get("created_when"), "scores": _score_text(e.get("scores"))}
    return {"rank": None, "of": board.get("count")}


def cmd_codabench_list(args):
    cb = Codabench(args.profile)
    me = cb.get("/api/my_profile/").get("username")
    comp = cb.get(f"/api/competitions/{COMPETITION}/")
    rule = [lb.get("submission_rule") for lb in comp.get("leaderboards", [])]
    print(f"competition {comp['id']} {comp.get('title')!r}; user {me}; participant_status "
          f"{comp.get('participant_status')}; leaderboard rule {rule}")
    phases = phase_report(cb, comp)
    for p in phases:
        print(f"phase {p['id']} {p['name']} [{p['status']}] {p['start']} -> {p['end']} tasks {p['tasks']} "
              f"used today {p['used_submissions_per_day']}/{p['max_submissions_per_day']} total "
              f"{p['used_submissions_per_person']}/{p['max_submissions_per_person']} can_submit "
              f"{p['can_submit']} {p['reason'] or ''}".rstrip())
    subs = cb.get_all("/api/submissions/", phase__competition=COMPETITION)
    print(f"\nmy submissions: {len(subs)}")
    for s in subs:
        line = (f"  #{s.get('id')} {s.get('phase_name')} {s.get('created_when')} {s.get('status')} "
                f"file={s.get('filename')} on_leaderboard={s.get('on_leaderboard')} {_score_text(s.get('scores'))}")
        print(line.rstrip())
    try:
        fill_scores(subs)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not fill submission scores in {SUBMISSIONS_INDEX}: {e!r}")
    boards = {}
    for p in phases:
        boards[p["name"]] = my_leaderboard(cb, p["id"], me)
        print(f"leaderboard {p['name']}: {boards[p['name']]}")
    if args.json:
        keep = ("id", "phase", "phase_name", "created_when", "status", "status_details", "filename",
                "on_leaderboard", "leaderboard", "scores", "fact_sheet_answers", "parent", "has_children")
        print(json.dumps({"user": me, "competition": comp["id"], "rule": rule, "phases": phases,
                          "submissions": [{k: s.get(k) for k in keep} for s in subs],
                          "leaderboard": boards}, default=str))
    return 0


def cmd_codabench_submit(args):
    name, phase_id, task_id, _ = PHASES[args.phase]
    zpath = Path(args.zip)
    try:
        kind, detail = inspect_submission_zip(zpath, args.phase)
    except (ValueError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        return _fail(f"{zpath}: {exc}")
    size = zpath.stat().st_size
    print(f"{zpath.name}: {kind} submission, {size} bytes, sha256 {sha256_file(zpath)}, "
          f"members {detail['members']}")
    if kind == "pointer" and not args.skip_hf_check:
        m = detail["manifest"]
        errors, warnings = check_read_token(m["token"], m["repo_id"], m["revision"], args.phase)
        for w in warnings:
            print(f"WARNING: {w}")
        if errors:
            for e in errors:
                print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(f"pointer OK: {m['repo_id']}@{m['revision']} readable with the manifest token")
    if kind == "direct":
        print(f"NOTE: direct upload; Codabench storage ~0.5 MB/s -> ~{size / 5e5 / 60:.0f} min transfer, "
              "counts against the 15 GB quota")
    fact = None
    if args.method_name and not args.allow_method_name:
        return _fail("--method-name is shown publicly on the leaderboard; the user asked (2026-10-08) "
                     "to omit it. Pass --allow-method-name only if they explicitly ask for one.")
    if args.method_name or args.code_url or args.compute_light:
        fact = {"method_name": args.method_name or "", "code_url": args.code_url or "",
                "compute_light": bool(args.compute_light)}
    # Organisers' Final guide (forum 2875 §6, 2026-10-19): no Code URL at submission; packages are requested after
    # the Final. Never pass a placeholder.
    if args.phase == "final" and args.code_url:
        print("NOTE: the Final no longer takes a Code URL (organisers' Final guide, 2026-10-19); it will be sent in "
              "the fact sheet anyway")

    cb = Codabench(args.profile)
    me = cb.get("/api/my_profile/").get("username")
    comp = cb.get(f"/api/competitions/{COMPETITION}/")
    live = {p["name"]: p for p in comp["phases"]}.get(name)
    if not live or live["id"] != phase_id or [t["id"] for t in live.get("tasks", [])] != [task_id]:
        return _fail(f"live API phase/task for {name} != pinned {phase_id}/{task_id}; re-check before submitting")
    if comp.get("participant_status") != "approved":
        return _fail(f"participant_status {comp.get('participant_status')!r}")
    if live.get("status") != "Current":
        return _fail(f"phase {name} status {live.get('status')!r} (not accepting submissions)")
    can = cb.get(f"/api/can_make_submission/{phase_id}/")
    if not can.get("can"):
        return _fail(f"Codabench says no: {can.get('reason')}")
    rule = [lb.get("submission_rule") for lb in comp.get("leaderboards", [])]
    print(f"user {me}; phase {name} {phase_id} task {task_id}; used today "
          f"{live.get('used_submissions_per_day')}/{live.get('max_submissions_per_day')}, total "
          f"{live.get('used_submissions_per_person')}/{live.get('max_submissions_per_person')}; rule {rule}")
    print(f"current leaderboard entry: {my_leaderboard(cb, phase_id, me)}")
    if "Force_Last" in rule:
        print("Force_Last: once this submission finishes it REPLACES the entry above on the leaderboard.")
    payload = {"phase": phase_id, "fact_sheet_answers": fact, "tasks": [task_id],
               "organization": None, "queue": None}
    if not args.confirm:
        print(f"\nDRY RUN: would POST /api/datasets/ (type=submission, competition={COMPETITION}, "
              f"file_name={zpath.name}, file_size={size}), PUT the zip to the presigned URL, "
              f"PUT /api/datasets/completed/<key>/, then POST /api/submissions/ {json.dumps(payload)}. "
              "Re-run with --confirm to submit.")
        return 0

    r = None
    for attempt in range(1, 4):   # storage stalls >60 s used to abort the whole upload
        ds = cb._unsafe("POST", "/api/datasets/", {
            "type": "submission", "competition": COMPETITION, "request_sassy_file_name": zpath.name,
            "file_name": zpath.name, "file_size": size})
        key, url = ds.get("key"), ds.get("sassy_url")
        if not key or not url:
            return _fail(f"dataset creation returned keys {sorted(ds)}")
        print(f"dataset {key} created; uploading {size} bytes (attempt {attempt}) ...", flush=True)
        try:
            with open(zpath, "rb") as f:  # presigned storage URL: no cookies, Content-Type signed
                # (connect/write timeout, read timeout): socket writes may stall for minutes
                r = cb.requests.put(url, data=f, headers={"Content-Type": "application/zip",
                                                          "Content-Length": str(size)},
                                    timeout=(900, None))
            if r.ok:
                break
            print(f"storage PUT -> HTTP {r.status_code}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"storage PUT failed: {e!r}", flush=True)
            r = None
    if r is None or not r.ok:
        return _fail(f"storage PUT failed after retries; no submission made")
    cb._unsafe("PUT", f"/api/datasets/completed/{key}/")
    sub = cb._unsafe("POST", "/api/submissions/", {"data": key, **payload})
    print(f"SUBMITTED: submission id {sub.get('id')} status {sub.get('status')} dataset {key}")
    record_submission(zpath, kind, detail, sub, name)
    print("Check progress/scores with: python -B fec/submit.py codabench list")
    return 0


# ----------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check one npz against the local frame scaffold")
    v.add_argument("npz", type=Path)
    v.add_argument("--config", required=True, help="e.g. diii_d_public_test, mast_public_test")
    v.add_argument("--work", type=Path, default=None, help=f"FEC work root (default {WORK})")
    v.add_argument("--psi-dtype", default="float16", choices=("float16", "float32"))
    v.add_argument("--headers-only", action="store_true", help="skip the data scan (no finiteness check)")
    v.add_argument("--json", action="store_true", help="also print a one-line JSON summary")
    v.set_defaults(func=cmd_validate)

    si = sub.add_parser("submissions", help="print the submissions table from the registry")
    si.set_defaults(func=cmd_submissions)

    u = sub.add_parser("upload", help="push npz files to a PRIVATE HF dataset repo (dry run by default)")
    u.add_argument("--files", nargs="+", type=Path)
    u.add_argument("--repo", default=DEFAULT_REPO)
    u.add_argument("--dry-run", action="store_true", help="default; check and print the plan only")
    u.add_argument("--confirm", action="store_true", help="actually create the repo / upload")
    u.add_argument("--create-only", action="store_true", help="only create the private repo (token step)")
    u.add_argument("--skip-validate", action="store_true")
    u.add_argument("--allow-public", action="store_true")
    u.add_argument("--message", default=None)
    u.add_argument("--work", type=Path, default=None)
    u.add_argument("--psi-dtype", default="float16", choices=("float16", "float32"))
    u.set_defaults(func=cmd_upload)

    p = sub.add_parser("pointer", help=f"write the manifest.json pointer zip (token from ${READ_TOKEN_ENV})")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--revision", required=True, help="40-character commit SHA")
    p.add_argument("--out", type=Path, default=Path("submission_pointer.zip"))
    p.add_argument("--phase", choices=tuple(PHASES), default="dev")
    p.add_argument("--offline", action="store_true", help="skip HF token/repo checks")
    p.add_argument("--force", action="store_true", help="replace an existing --out")
    p.set_defaults(func=cmd_pointer)

    c = sub.add_parser("codabench", help=f"Codabench competition {COMPETITION} via Firefox cookies")
    csub = c.add_subparsers(dest="ccmd", required=True)
    cl = csub.add_parser("list", help="read-only: phases, limits, my submissions and scores")
    cl.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(func=cmd_codabench_list)
    cs = csub.add_parser("submit", help="upload a pointer/direct zip (dry run unless --confirm)")
    cs.add_argument("zip", type=Path)
    cs.add_argument("--phase", choices=tuple(PHASES), default="dev")
    cs.add_argument("--confirm", action="store_true")
    cs.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    cs.add_argument("--skip-hf-check", action="store_true", help="do not verify the pointer on HF")
    cs.add_argument("--method-name", default=None,
                    help="public on the leaderboard; refused unless --allow-method-name (user, 2026-10-08)")
    cs.add_argument("--allow-method-name", action="store_true")
    cs.add_argument("--code-url", default=None)
    cs.add_argument("--compute-light", action="store_true")
    cs.set_defaults(func=cmd_codabench_submit)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
