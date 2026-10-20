"""Final-phase input data: which Hugging Face dataset holds a config, download, issue-#12 guard.

Agent fpkg 2026-10-12 (docs/archive/fec/FINAL_PACKAGE_20261012.md).  The organisers published the
Final inputs (configs diii_d_private_test, 879 shots, and mast_private_test, 1,205 shots; inputs
only) in a SEPARATE public dataset, ganatma/fusion-equilibrium-challenge-final, revision
965040c12b814f33c8e7209f6b1799932247e793 (organisers' Final guide, 2026-10-19; the package pins it in
data_manifest.json 'datasets'); the public configs stay in Sophelio/fusion-equilibrium-challenge.  Env
FEC_HF_DATASET overrides both (tests, mirrors).

    python -B fec/final_data.py repo diii_d_private_test
    python -B fec/final_data.py download diii_d_private_test mast_private_test
    python -B fec/final_data.py issue12 --config diii_d_private_test --scan <input_guard>/<cfg>_scan.jsonl

Download layout: <FEC_WORK>/hf_data/data/<config>/*.parquet (fec.paths.HF_DATA, read by
fec.paths.shot_files).  Remote files may sit under data/<config>/ or <config>/; both are mapped
to that layout.  File count is verified against the listing.

Rate limits (agent pkgfix 2026-10-13, timed rebuild defect D1): anonymous downloads of the public
dataset hit HTTP 429 after ~1,000 files (also on the xet token route, raised as a plain
ConnectionError).  download() therefore (a) sets HF_HUB_DISABLE_XET=1 unless the caller set it,
(b) uses an HF login when one exists (FEC_HF_TOKEN_FILE if set, else huggingface_hub's HF_TOKEN /
`hf auth login` cache; only its presence is logged, never the value), and (c) retries the listing
and each snapshot_download on HTTP 429 / 5xx and transient transport errors.  Retry policy (rebuild defect
R2-2, agent pkgfix2): the budget is FEC_HF_RETRIES (default 10) CONSECUTIVE failures without progress (a
snapshot's progress = files complete on disk; any progress resets the count and the backoff); 429 / 5xx /
rate-limit messages wait FEC_HF_BACKOFF_S (default 30 s) doubling to 900 s, at least the Retry-After or
RateLimit reset; transport errors (ReadTimeout, resets, HTTP 408 / 499: R2-3) wait FEC_HF_SHORT_BACKOFF_S (default 10 s) doubling
to 60 s with jitter; FEC_HF_STALL_S (default 3 h) without progress ends the download.  snapshot_download
resumes, so a retry continues where the previous attempt stopped.  Because snapshot_download silently returns an existing local_dir when
its repo listing is refused, each attempt also checks that every listed file arrived (IncompleteDownload
is retried).  Other errors (401/403/404, ...) are raised at once.

Revisions (agent pkgfix2 2026-10-13, R35 package hygiene): every download is made at ONE commit.  A dataset pinned in
the package's data_manifest.json 'datasets' (40-hex `revision`; env FEC_DATA_MANIFEST overrides the file) is listed
and downloaded at that commit; an unpinned one (no 40-hex pin, e.g. a FEC_HF_DATASET mirror, or any dataset in the
research repo, which has no manifest) is resolved to its current commit first.  The commit, file list and count of every downloaded config are
recorded in <hf_data>/../revisions.json (<FEC_WORK>/hf_data/revisions.json).  A pinned config whose record matches the
pin and the local file list is not downloaded again, with no network call (reproduce.sh runs everything after its
download stage with HF_HUB_OFFLINE=1).  OfflineModeIsEnabled (HF_HUB_OFFLINE=1) is never retried.  HF_TOKEN is
optional; nothing here starts an interactive login.

issue12: starter issue #12 (organiser): Thomson timestamps may be non-monotone; the supported
rule keeps a sample only if its time exceeds all previous ones.  The Final inputs contain one
DIII-D edge-padding case and one MAST core out-of-order sample.  Our extraction
(fec/extract.ts_interp) sorts the samples by time and treats zero/NaN values as missing, so a
padding row (time 0, no data) is ignored and an out-of-order sample is used at its own time; no
feature changes.  fec/input_fixes2.py guard (DIII-D only) reports tangential (edge) resets as
info but FAILS on any Thomson CORE dup/back step, which would stop run_final's ipguard.
`tolerable_core_issue12` lets ipguard continue when that is the ONLY failing condition and every
affected shot has at most MAX_CORE_ANOMALIES non-monotone core samples or only empty rows after
its first backward step.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PUBLIC_REPO = "Sophelio/fusion-equilibrium-challenge"
FINAL_REPO = "ganatma/fusion-equilibrium-challenge-final"   # organisers' Final guide (2026-10-19); pinned in data_manifest.json
MAX_CORE_ANOMALIES = 2          # non-monotone core samples per shot tolerated (issue #12)
MAX_CORE_SHOTS = 3              # shots with such samples tolerated per fold
HF_BACKOFF_CAP_S = 900.0        # longest exponential-backoff wait between download attempts
HF_RETRY_AFTER_MAX_S = 3600.0   # longest server Retry-After honoured
HF_SHORT_CAP_S = 60.0           # longest backoff after a transport error (ReadTimeout, reset, ...)
TRANSPORT_STATUS = (408, 499)   # HTTP statuses retried like transport errors (R2-3: a 499 ended a token download
                                # at 5,045 of 9,121 files); 401 / 403 / 404 stay fatal (404 = missing config)
LOGIN_HINT = ("optional: set HF_TOKEN to a read token (or use an earlier `hf auth login`) and keep "
              "HF_HUB_DISABLE_XET=1: the dataset is public, but anonymous downloads are rate limited (HTTP 429)")
SHA_RE = re.compile(r"[0-9a-f]{40}")
REVISIONS_FILE = "revisions.json"   # <hf_data parent>/revisions.json: commit + files of every downloaded config
# transient transport errors of httpx / requests / urllib3, matched by class name (no import needed)
_TRANSPORT = {"TransportError", "NetworkError", "TimeoutException", "ConnectError", "ConnectTimeout",
              "ReadTimeout", "ReadError", "WriteError", "WriteTimeout", "PoolTimeout", "RemoteProtocolError",
              "ProtocolError", "ChunkedEncodingError", "Timeout"}
_RATE_MSG = re.compile(r"\b429\b|too many requests|rate[ -]?limit", re.I)


def _env_true(v):
    return str(v).strip().upper() in ("1", "ON", "YES", "TRUE")


def http_status(exc):
    """HTTP status code of an exception carrying a response (huggingface_hub HfHubHTTPError), else None."""
    st = getattr(getattr(exc, "response", None), "status_code", None)
    return st if isinstance(st, int) else None


def classify(exc):
    """(retry?, reason, backoff kind) for an exception of a Hugging Face call.  kind "short" also for HTTP 408 / 499
    (TRANSPORT_STATUS, R2-3).  kind "long": HTTP 429 / 5xx, a
    rate-limit message without a response (the xet layer raised `ConnectionError: Rate limited ... 429` on the
    rebuild pod) and IncompleteDownload (snapshot_download's silent return after a refused listing); kind "short":
    transient transport errors (ReadTimeout, connection resets, ChunkedEncodingError, ...).  The __cause__ /
    __context__ chain is walked (huggingface_hub wraps connection errors).  Never retried: OfflineModeIsEnabled
    (HF_HUB_OFFLINE=1), 401/403/404 and anything else."""
    e, seen = exc, set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if any(c.__name__ == "OfflineModeIsEnabled" for c in type(e).__mro__):   # a ConnectionError subclass
            return False, "OfflineModeIsEnabled (HF_HUB_OFFLINE=1: no network call is allowed here)", None
        st = http_status(e)
        if st is not None:
            if st in TRANSPORT_STATUS:        # R2-3: 408 timeout, 499 client closed (rebuild2 token download)
                return True, f"HTTP {st}", "short"
            return (st == 429 or 500 <= st <= 599), f"HTTP {st}", "long"
        if _RATE_MSG.search(str(e)):
            return True, f"{type(e).__name__} (rate limited, 429)", "long"
        if isinstance(e, IncompleteDownload):
            return True, "IncompleteDownload", "long"
        if isinstance(e, (ConnectionError, TimeoutError)) or \
                {c.__name__ for c in type(e).__mro__} & _TRANSPORT:
            return True, type(e).__name__, "short"
        e = e.__cause__ or e.__context__
    return False, type(exc).__name__, None


def retryable(exc):
    """(retry?, reason) of classify()."""
    ok, why, _ = classify(exc)
    return ok, why


class IncompleteDownload(ConnectionError):
    """snapshot_download returned without every wanted file.  huggingface_hub returns an existing, non-empty
    local_dir WITHOUT raising when its repo listing fails (HTTP 429 / 5xx / connection error; _snapshot_download
    "Returning existing local_dir ... as remote repo cannot be accessed"), so a shortfall is retried, with the long
    (rate-limit) backoff (pkgfix review, 2026-10-13)."""


def _headers(e):
    return getattr(getattr(e, "response", None), "headers", None)


def retry_after_s(exc):
    """Seconds of an HTTP Retry-After header (delta-seconds form) on the exception or its cause chain, else None."""
    e, seen = exc, set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        hdr = _headers(e)
        try:
            v = hdr.get("Retry-After") if hdr is not None else None
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            return None
        e = e.__cause__ or e.__context__
    return None


_RATELIMIT_T = re.compile(r"\"\w+\"\s*;\s*r\s*=\s*\d+\s*;\s*t\s*=\s*(\d+)")


def ratelimit_reset_s(exc):
    """Seconds until the hub's rate-limit window resets, from its `RateLimit: "api";r=0;t=55` header (IETF draft
    format, as huggingface_hub.utils.parse_ratelimit_headers reads it) on the exception or its cause chain."""
    e, seen = exc, set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        hdr = _headers(e)
        if hdr is not None:
            try:
                for k in hdr:
                    if str(k).lower() == "ratelimit":
                        m = _RATELIMIT_T.search(str(hdr[k]))
                        if m:
                            return float(m.group(1))
            except (TypeError, ValueError, AttributeError):
                return None
        e = e.__cause__ or e.__context__
    return None


def hf_retry(fn, *args, what="HF request", log=print, tries=None, base_s=None, cap_s=HF_BACKOFF_CAP_S,
             sleep=time.sleep, progress=None, short_s=None, short_cap_s=HF_SHORT_CAP_S, stall_s=None,
             clock=time.monotonic, rand=random.random, **kw):
    """fn(*args, **kw), retried on classify() errors (rebuild defect R2-2, agent pkgfix2 2026-10-13).

    * Budget: `tries` CONSECUTIVE failures without progress (env FEC_HF_RETRIES, default 10).  progress() (optional;
      snapshot_complete passes the number of wanted files already complete on disk) is read after every failure; a
      higher value than ever before resets the failure count, the backoff and the stall clock, so a download that
      keeps advancing between timeouts is never cut off by the count.
    * Backoff after the k-th consecutive failure without progress (k = 0, 1, ...): kind "long" (429, 5xx, rate-limit
      message, IncompleteDownload): base_s * 2**k (env FEC_HF_BACKOFF_S, default 30 s), capped at cap_s, and at
      least the server's Retry-After or its RateLimit reset (+1 s), up to HF_RETRY_AFTER_MAX_S; kind "short"
      (transport errors): short_s * 2**k (env FEC_HF_SHORT_BACKOFF_S, default 10 s) capped at short_cap_s (60 s),
      times a jitter in [0.75, 1.25).
    * Stall cap: SystemExit when the next wait would end more than stall_s (env FEC_HF_STALL_S, default 10,800 s =
      3 h) after the last progress (or the start), so a dead network still fails.
    Logs the error class / status only (never the message: it may carry request URLs).  SystemExit with the hint
    when the budget or the stall cap is exhausted."""
    tries = int(os.environ.get("FEC_HF_RETRIES", "10")) if tries is None else int(tries)
    base_s = float(os.environ.get("FEC_HF_BACKOFF_S", "30")) if base_s is None else float(base_s)
    short_s = float(os.environ.get("FEC_HF_SHORT_BACKOFF_S", "10")) if short_s is None else float(short_s)
    stall_s = float(os.environ.get("FEC_HF_STALL_S", "10800")) if stall_s is None else float(stall_s)
    best = progress() if progress else None
    t_prog = clock()
    fails = 0
    while True:
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            ok, why, kind = classify(e)
            if not ok:
                raise
            note = ""
            if progress:
                p = progress()
                if p > best:
                    note = f"; progress {best} -> {p}, retry budget reset"
                    best, fails, t_prog = p, 0, clock()
                else:
                    note = f"; no progress ({p})"
            if fails >= tries:
                raise SystemExit(f"{what}: still failing after {tries} consecutive retries without progress ({why}); "
                                 f"{LOGIN_HINT}") from e
            if kind == "short":
                wait = min(short_cap_s, short_s * 2 ** fails) * (0.75 + 0.5 * rand())
            else:
                wait = min(cap_s, base_s * 2 ** fails)
                hint = [x for x in (retry_after_s(e), (lambda r: None if r is None else r + 1.0)(ratelimit_reset_s(e)))
                        if x is not None]
                if hint:
                    wait = max(wait, min(max(hint), HF_RETRY_AFTER_MAX_S))
            if clock() - t_prog + wait > stall_s:
                raise SystemExit(f"{what}: no progress for {clock() - t_prog:.0f} s (stall cap {stall_s:.0f} s, "
                                 f"{why}); {LOGIN_HINT}") from e
            log(f"{what}: {why}{note}; retry {fails + 1}/{tries} in {wait:.0f} s")
            sleep(wait)
            fails += 1


def hf_token(log=print):
    """Optional HF login for downloads: the file named by FEC_HF_TOKEN_FILE (only when that variable is
    set), else huggingface_hub.get_token() (HF_TOKEN, then the `hf auth login` cache).  Returns the token
    or None.  Only whether a login was found is logged, never the token."""
    tok, src = None, None
    f = os.environ.get("FEC_HF_TOKEN_FILE")
    if f and Path(f).is_file():
        tok, src = Path(f).read_text(encoding="utf-8").strip() or None, "FEC_HF_TOKEN_FILE"
    if tok is None:
        from huggingface_hub import get_token
        tok, src = get_token(), "HF_TOKEN / hf auth login"
    log(f"HF login: yes ({src})" if tok else f"HF login: none; {LOGIN_HINT}")
    return tok


def xet_off():
    """HF_HUB_DISABLE_XET=1 unless the caller set it (the rebuild pod's xet route hit 429 / stalled even
    with a login; plain HTTP did not).  Also sets huggingface_hub.constants when the hub is already
    imported, because the constant is read at import time.  Returns the effective setting."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    off = _env_true(os.environ["HF_HUB_DISABLE_XET"])
    if "huggingface_hub.constants" in sys.modules:
        sys.modules["huggingface_hub.constants"].HF_HUB_DISABLE_XET = off
    return off


def data_manifest_file():
    """The package's data_manifest.json (next to the package's fec/), or env FEC_DATA_MANIFEST."""
    env = os.environ.get("FEC_DATA_MANIFEST")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "data_manifest.json"


def pinned_revision(repo, manifest=None):
    """The immutable revision (a 40-hex commit) of dataset `repo` in data_manifest.json 'datasets', else None (no
    manifest, an unlisted repo, or a revision that is not a 40-hex commit)."""
    p = Path(manifest) if manifest else data_manifest_file()
    try:
        ds = json.loads(p.read_text(encoding="utf-8")).get("datasets") or {}
    except (OSError, ValueError):
        return None
    rev = (ds.get(repo) or {}).get("revision")
    return rev if isinstance(rev, str) and SHA_RE.fullmatch(rev) else None


def resolve_revision(repo, revision=None, token=None, log=print, **retry_kw):
    """Commit of `repo` at `revision` (None: the default branch) from HfApi().dataset_info, with hf_retry.  A pinned
    revision must resolve to itself."""
    from huggingface_hub import HfApi
    try:
        info = hf_retry(HfApi(token=token).dataset_info, repo, revision=revision,
                        what=f"resolve {repo}@{revision or 'main'}", log=log, **retry_kw)
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ != "RepositoryNotFoundError":
            raise
        when = (" To rehearse on the Development fold use the public_test split (reproduce.sh: SPLIT=public_test)."
                if repo == FINAL_REPO else "")
        raise SystemExit(f"{repo}: dataset not found or not accessible (RepositoryNotFoundError).{when}") from e
    sha = getattr(info, "sha", None)
    if not (isinstance(sha, str) and SHA_RE.fullmatch(sha)):
        raise SystemExit(f"{repo}: could not resolve the commit of {revision or 'main'}")
    if revision and sha != revision:
        raise SystemExit(f"{repo}: pinned revision {revision} resolved to {sha}")
    return sha


def read_revisions(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def list_repo_files(repo, log=print, token=None, revision=None, **retry_kw):
    """HfApi().list_repo_files of a dataset repo with hf_retry (shared by download and fec/run_final.py), at
    `revision`, else at the data_manifest pin of the repo, else at the default branch."""
    from huggingface_hub import HfApi
    revision = revision or pinned_revision(repo)
    kw = {"revision": revision} if revision else {}
    return hf_retry(HfApi(token=token).list_repo_files, repo, repo_type="dataset",
                    what=f"list {repo}", log=log, **kw, **retry_kw)


def snapshot_complete(repo, files, local_dir, log=print, what=None, retry_kw=None, **snap_kw):
    """hf_retry of [snapshot_download(repo, local_dir=..., **snap_kw), then every path of `files` (relative to
    local_dir) must exist].  A shortfall raises IncompleteDownload, which is retried; snapshot_download resumes.
    Returns len(files).  Used for the dataset configs (download) and the weights (reproduce.sh).  Progress for
    hf_retry = the number of `files` complete on disk (snapshot_download moves a file into place when it is
    complete), so the retry budget only runs down while no new file arrives."""
    from huggingface_hub import snapshot_download
    local_dir = Path(local_dir)

    def once():
        snapshot_download(repo, local_dir=str(local_dir), **snap_kw)
        miss = [f for f in files if not (local_dir / f).is_file()]
        if miss:
            raise IncompleteDownload(f"{len(miss)} of {len(files)} files missing after snapshot_download")
        return len(files)
    kw = dict(retry_kw or {})
    kw.setdefault("progress", lambda: sum((local_dir / f).is_file() for f in files))   # R2-2: complete files
    return hf_retry(once, what=what or f"download {repo}", log=log, **kw)


def dataset_repo(config_or_split):
    """HF dataset repo of a config or split name (env FEC_HF_DATASET overrides)."""
    env = os.environ.get("FEC_HF_DATASET")
    if env:
        return env
    return FINAL_REPO if "private_test" in str(config_or_split) else PUBLIC_REPO


def remote_files(repo, config, files=None):
    """Sorted remote parquet paths of `config` (data/<cfg>/ or <cfg>/ prefix) and the prefix."""
    if files is None:
        files = list_repo_files(repo)
    for pre in (f"data/{config}/", f"{config}/"):
        want = sorted(f for f in files if f.startswith(pre) and f.endswith(".parquet"))
        if want:
            return want, pre
    return [], None


def download(configs, repo=None, log=print, hf_data=None, revision=None, **retry_kw):
    """Snapshot-download each config's parquet files into fec.paths.HF_DATA/<config>/ (hf_data overrides,
    tests).  HF_HUB_DISABLE_XET=1 by default, optional HF login, listing and snapshot retried on 429 / 5xx
    (hf_retry; retry_kw: tries, base_s, cap_s, sleep).  Used for the public configs (train.py) and for the
    Final private_test inputs (fec/run_final.py download stage).  One commit per repo: `revision`, else the
    data_manifest pin, else the resolved current commit; recorded in <HF_DATA>/../revisions.json.  A pinned config
    already recorded complete at that commit (same local file list) is skipped without any network call."""
    off = xet_off()
    log(f"HF_HUB_DISABLE_XET={os.environ['HF_HUB_DISABLE_XET']} (xet transfer {'off' if off else 'on'})")
    if hf_data is None:
        from fec.paths import HF_DATA as hf_data
    HF_DATA = Path(hf_data)
    rec_file = HF_DATA.parent / REVISIONS_FILE
    record = read_revisions(rec_file)
    out, todo = {}, []
    for cfg in configs:
        r = repo or dataset_repo(cfg)
        pin = revision or pinned_revision(r)
        have = sorted(g.name for g in (HF_DATA / cfg).glob("*.parquet"))
        rc = record.get(cfg) or {}
        if pin and have and rc.get("repo") == r and rc.get("revision") == pin and rc.get("files") == have:
            log(f"{cfg}: complete at the pinned revision {pin[:12]} ({len(have)} files, {rec_file.name}); "
                f"no download")
            out[cfg] = dict(repo=r, prefix=rc.get("prefix"), n=len(have), revision=pin, pinned=True,
                            downloaded=False)
            continue
        todo.append((cfg, r, pin))
    if not todo:
        return out
    token = hf_token(log)
    commits, listing = {}, {}
    for cfg, r, pin in todo:
        if (r, pin) not in commits:
            commits[(r, pin)] = resolve_revision(r, pin, token=token, log=log, **retry_kw)
        rev = commits[(r, pin)]
        if (r, rev) not in listing:
            listing[(r, rev)] = list_repo_files(r, log=log, token=token, revision=rev, **retry_kw)
        want, pre = remote_files(r, cfg, listing[(r, rev)])
        log(f"{cfg}: {r}@{rev[:12]} ({'pinned' if pin else 'unpinned: current commit'}) remote files {len(want)} "
            f"(prefix {pre})")
        if not want:
            raise SystemExit(f"{cfg}: no parquet files in {r}@{rev}")
        # remote paths data/<cfg>/x.parquet or <cfg>/x.parquet are relative to the matching local_dir
        snapshot_complete(r, want, HF_DATA.parent if pre.startswith("data/") else HF_DATA, log=log,
                          what=f"download {cfg}", retry_kw=retry_kw, repo_type="dataset", revision=rev,
                          allow_patterns=[f"{pre}*.parquet"], max_workers=8, token=token)
        got = sorted((HF_DATA / cfg).glob("*.parquet"))
        ok = [Path(f).name for f in want] == [g.name for g in got]
        log(f"{cfg}: local files {len(got)} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"{cfg}: local file list differs from {r}@{rev}")
        record[cfg] = dict(repo=r, revision=rev, pinned=bool(pin), prefix=pre, n=len(got),
                           files=[g.name for g in got], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        rec_file.parent.mkdir(parents=True, exist_ok=True)
        rec_file.write_text(json.dumps(record, indent=1), encoding="utf-8")
        out[cfg] = dict(repo=r, prefix=pre, n=len(got), revision=rev, pinned=bool(pin), downloaded=True)
    return out


# ============================================================================ issue #12
def guard_conditions(R):
    """The fail conditions of fec.input_fixes2.guard (same formulas) on scan rows R."""
    return {
        "efit axis dup/back/nan": sum(r["efit"].get("n_dup", 0) + r["efit"].get("n_back", 0)
                                      + r["efit"].get("nan", 0) for r in R),
        "magnetics axis gap/dup/back/nan": sum(r["mag"].get(k, 0) for r in R
                                               for k in ("n_gap", "n_dup", "n_back", "nan")),
        "Ip axis gap/dup/back": sum(r["ipt"].get(k, 0) for r in R for k in ("n_gap", "n_dup", "n_back")),
        "empty box windows (magnetics, Ip)": sum(r["box_empty_mag"] + r["box_empty_ip"] for r in R),
        "frames past record end": sum(r["efit_after_ip_end"] + r["efit_after_mag_end"] for r in R),
        "channel length mismatch": sum("len_mismatch" in c for r in R for c in r["ch"].values()),
        "Thomson core axis dup/back": sum(r["ts_core"]["axis"].get("n_dup", 0)
                                          + r["ts_core"]["axis"].get("n_back", 0) for r in R),
        "Thomson chord/channel count mismatch": sum(
            r[s].get("nch_vs_names", [0, 0])[0] != r[s].get("nch_vs_names", [0, 0])[1]
            for r in R for s in ("ts_core", "ts_edge")),
    }


def monotone_keep(times):
    """Organiser rule (issue #12): keep a sample only if its time exceeds all previous ones."""
    t = np.asarray(times, dtype=np.float64)
    if t.size == 0:
        return np.zeros(0, bool)
    prev = np.maximum.accumulate(np.r_[-np.inf, t[:-1]])
    return t > prev


def core_anomaly(times, te, ne):
    """(n non-monotone samples by the organiser rule, True if every row after the first
    backward step is empty)."""
    t = np.asarray(times, dtype=np.float64)
    n_bad = int((~monotone_keep(t)).sum())
    back = np.flatnonzero(np.diff(t) < 0)
    if back.size == 0:
        return n_bad, True
    k = int(back[0]) + 1
    te = np.asarray(te, float); ne = np.asarray(ne, float)
    data = (np.isfinite(te) & (te > 0)).any(1) | (np.isfinite(ne) & (ne > 0)).any(1)
    return n_bad, not data[k:].any()


def tolerable_core_issue12(R, read_core=None):
    """Whether a failed input_fixes2 guard may continue: the only failing condition is the
    Thomson core dup/back count, at most MAX_CORE_SHOTS shots are affected, and each has at most
    MAX_CORE_ANOMALIES non-monotone samples (organiser rule) or only empty rows after its first
    backward step.  read_core(stem) -> (times, te, ne) re-reads a shot's core Thomson (inputs).
    Returns (ok, detail)."""
    conds = guard_conditions(R)
    failing = sorted(k for k, v in conds.items() if v)
    detail = dict(failing=failing, conditions=conds, shots={})
    if failing != ["Thomson core axis dup/back"]:
        detail["reason"] = "other guard conditions fail" if failing else "guard passes"
        return (not failing), detail
    hit = [r["stem"] for r in R if r["ts_core"]["axis"].get("n_dup", 0)
           or r["ts_core"]["axis"].get("n_back", 0)]
    if len(hit) > MAX_CORE_SHOTS:
        detail["reason"] = f"{len(hit)} shots > {MAX_CORE_SHOTS}"
        return False, detail
    ok = True
    for stem in hit:
        if read_core is None:
            detail["shots"][stem] = "not re-read"
            ok = False
            continue
        n_bad, empty_tail = core_anomaly(*read_core(stem))
        good = n_bad <= MAX_CORE_ANOMALIES or empty_tail
        detail["shots"][stem] = dict(n_nonmonotone=n_bad, empty_after_back=empty_tail, ok=good)
        ok &= good
    detail["reason"] = "issue #12 core sample(s): tolerated" if ok else "beyond tolerance"
    return ok, detail


def parquet_core_reader(config):
    """read_core for tolerable_core_issue12 from the fold's parquet files (inputs only)."""
    import pyarrow.parquet as pq
    from fec.paths import shot_files
    files = {f.stem: f for f in shot_files(config)}

    def read(stem):
        t = pq.read_table(files[stem], columns=["thomson_core_times", "thomson_core_Te",
                                                "thomson_core_ne"])
        tm = np.asarray(t.column("thomson_core_times")[0].as_py(), float)
        te = np.array([np.asarray(r, float) for r in t.column("thomson_core_Te")[0].as_py()])
        ne = np.array([np.asarray(r, float) for r in t.column("thomson_core_ne")[0].as_py()])
        return tm, te.reshape(len(tm), -1), ne.reshape(len(tm), -1)
    return read


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("repo")
    p.add_argument("config")
    p = sub.add_parser("download")
    p.add_argument("configs", nargs="+")
    p.add_argument("--repo", default=None)
    p.add_argument("--revision", default=None, help="commit to download (default: the data_manifest.json pin, else the current commit; recorded in <hf_data>/../revisions.json)")
    p = sub.add_parser("issue12")
    p.add_argument("--config", required=True)
    p.add_argument("--scan", required=True)
    p.add_argument("--out", default=None, help="write the decision json here")
    a = ap.parse_args(argv)
    if a.cmd == "repo":
        print(dataset_repo(a.config))
    elif a.cmd == "download":
        print(json.dumps(download(a.configs, a.repo, revision=a.revision)))
    else:
        R = [json.loads(line) for line in open(a.scan, encoding="utf-8")]
        ok, det = tolerable_core_issue12(R, parquet_core_reader(a.config))
        print(json.dumps(dict(ok=ok, **det), default=int))
        if a.out:
            Path(a.out).write_text(json.dumps(dict(ok=ok, **det), indent=1, default=int))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
