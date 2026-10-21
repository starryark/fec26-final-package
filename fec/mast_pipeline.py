"""MAST candidate pipeline: stage-1 members -> optional deep-crossing merge -> optional map
average -> stage 2 + frozen betaN of the final node -> optional post-processing nodes (tsm:
within-shot temporal smoothing of the maps, scalars/betaN copied) -> submission build.

Owner mastpipe, 2026-10-09 (docs/research/MAST_PIPELINE_20261009.md).  Consolidates root's scratch
chain (fec_work/root_var/chain_variant.py, avg_maps.py, merge_dc.py and
fec_work/mastz/mz_finish_cand.py) with the MAST config as a parameter (mast_public_test |
mast_private_test), so the Final is one command.  Inputs only: coil currents, Ip, Thomson Te/ne,
C1 passive currents.  No MAST label, demo parquet or scorer mask is read.

    python -B fec/mast_pipeline.py member   --config C --out ROOT --spec '{"name": ...}' [--no-finish]
    python -B fec/mast_pipeline.py merge-dc --config C --out ROOT --loff A --lon B [--name N]
    python -B fec/mast_pipeline.py average  --config C --out ROOT --members a,b,c [--weights 1,1,1]
    python -B fec/mast_pipeline.py tsm      --config C --out ROOT --of NODE_OR_DIR --sigma-ms 30 [--name N]
    python -B fec/mast_pipeline.py build    --config C --mast DIR --d3d PREDS_DIR --name OUT [--subset]
    python -B fec/mast_pipeline.py recipe   --config C --out ROOT --recipe fec/recipes_mast/x.json
    python -B fec/mast_pipeline.py status   --out ROOT

Common options: --limit N (frames; rounded UP to a whole-shot boundary, or with --limit-align chunk
to the production 1024-frame shot-chunk boundary so stage-1 batches equal a full run's),
--owner / --gpu-priority (fec/gpu_slot.py), --workers (CPU pools of stage 2 / betaN), --jobs K
(independent steps run concurrently), --overwrite (recompute nodes made with another spec),
--dry-run (print the plan).

Layout: ROOT/<node>/ holds psi.npy, meta.npz, config.json (+ scalars.npz, betan_v2.npy, ...);
ROOT/<node>/_steps/<step>.json records every step (signature, command, timing, status, output
hashes); logs in ROOT/logs/<node>.<step>.log; ROOT/manifests/<time>_<command>.json per
invocation; ROOT/pipeline.json for `recipe` (read by fec/run_final.py stage mast_pipeline).
A step is skipped when its record says ok with the same signature, its outputs exist and the
records of its inputs are the ones it was made from.  Signatures chain through dependencies.

Post nodes (mastpost 2026-10-10, recipe key "post"): {"name", "kind": "tsm", "of", "sigma_ms"}
runs fec/mast_tsm.py on node `of` (any earlier node, finished with stage 2 + betaN, which the plan
adds): smoothed psi + meta recomputed, scalars.npz / betaN files copied from `of`; the post node
itself never gets stage 2 / betaN.  "final" may name a post node (default: the last post node).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
FEC = REPO / "fec"
PY = sys.executable
CONFIGS = ("mast_public_test", "mast_private_test")
MAST_CHUNK = 1024                       # fec.mast_zsc.stage1_zsc max_frames (shot chunks)
DRIVERS = ("zsc", "zsc_bpli")
FALLBACKS = ("best", "z0", "prod")
SPEC_DEFAULT = dict(driver="zsc", passive="c1_r1", sol_frac=None, zfix=None, limiter_cc=False,
                    smooth_h=0, fallback="z0", heights_from=None, targets_base=None, nbasis=4)
SPEC_KEYS = frozenset(SPEC_DEFAULT) | {"name"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
RESERVED = frozenset({"logs", "manifests", "_steps", "pipeline.json"})
PASSIVE_RE = re.compile(r"^(none|c\d+_r[0-9.]+(_[A-Za-z0-9]+)?)$")
# Named member specs (the sub23 members, the limiter-on twin and the sub24 constraint member)
NAMED_SPECS = {
    "zsc_c1": dict(driver="zsc", passive="c1_r1"),
    "zsc_nopas": dict(driver="zsc", passive="none"),
    "zsc_nopas_s065": dict(driver="zsc", passive="none", sol_frac=0.65),
    "zsc_c1_s075": dict(driver="zsc", passive="c1_r1", sol_frac=0.75),
    "zsc_c1_sm4": dict(driver="zsc", passive="c1_r1", smooth_h=4),
    "zfix0_c1": dict(driver="zsc", passive="c1_r1", zfix=0.0),
    "zsc_c1_lcc": dict(driver="zsc", passive="c1_r1", limiter_cc=True),
    "bpli_c1": dict(driver="zsc_bpli", passive="c1_r1", targets_base="zsc_c1",
                    heights_from="zsc_c1"),
}
RECIPE_KEYS = frozenset({"name", "description", "boundary_limiter", "betan", "finish_members",
                         "members", "merge", "average", "post", "final", "reference_public",
                         "notes"})
PRODUCERS = ("stage1", "merge", "average", "tsm")   # the step that writes a node's psi.npy/meta.npz
POST_KINDS = ("tsm",)
POST_KEYS = frozenset({"name", "kind", "of", "sigma_ms"})
TSM_SIGMA_RANGE = (1.0, 1000.0)                 # ms
AVG_META_KEYS = ("psi_ax", "psi_b", "Rax", "Zax", "limited")


class PipelineError(RuntimeError):
    pass


# ============================================================================ small helpers
def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def now():
    import datetime as dt
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _jsonable(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable))
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def cmdline(cmd):
    return subprocess.list2cmdline([str(c) for c in cmd])


def sig_of(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=_jsonable)
                          .encode()).hexdigest()[:20]


def is_path_ref(ref):
    """Member names never contain path separators or drive colons (NAME_RE)."""
    s = str(ref)
    return "/" in s or "\\" in s or ":" in s


def resolve_dir(p):
    """Absolute paths as given; relative ones are siblings of FEC_WORK (as fec/run_final.py)."""
    from fec.paths import WORK
    p = Path(p)
    return p if p.is_absolute() else WORK.parent / p


def code_hashes():
    return {p.name: sha256(p) for p in sorted(FEC.glob("*.py"))}


def git_state():
    try:
        from fec.runlog import git_state as gs
        return gs()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ============================================================================ specs / recipes
def auto_name(spec):
    """Readable default member name, e.g. zsc_c1_r1_sol0.65_sm4."""
    parts = [spec["driver"], spec["passive"]]
    if spec["sol_frac"] is not None:
        parts.append(f"sol{spec['sol_frac']:g}")
    if spec["zfix"] is not None:
        parts.append(f"zfix{spec['zfix']:g}")
    if spec["limiter_cc"]:
        parts.append("lcc")
    if spec["smooth_h"]:
        parts.append(f"sm{spec['smooth_h']}")
    if spec["fallback"] != "z0":
        parts.append(f"fb{spec['fallback']}")
    if spec.get("nbasis", 4) != 4:
        parts.append(f"nb{spec['nbasis']}")
    return "_".join(parts)


def _num(v, what, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{what} must be a number, not {v!r}")
    v = float(v)
    if not (lo <= v <= hi):
        raise ValueError(f"{what} {v} outside [{lo}, {hi}]")
    return v


def parse_spec(obj):
    """Normalised member spec (dict with every SPEC_DEFAULT key + name) from a dict, an inline
    JSON string, a NAMED_SPECS name or a JSON file path.  Raises ValueError on unknown keys,
    bad values or option combinations the driver does not implement."""
    if isinstance(obj, (str, Path)):
        s = str(obj).strip()
        if s.startswith("{"):
            obj = json.loads(s)
        elif s in NAMED_SPECS:
            obj = dict(NAMED_SPECS[s], name=s)
        elif Path(s).is_file():
            obj = json.loads(Path(s).read_text())
        else:
            raise ValueError(f"spec {s!r} is not JSON, a named spec {sorted(NAMED_SPECS)} or a file")
    if not isinstance(obj, dict):
        raise ValueError(f"member spec must be an object, not {type(obj).__name__}")
    unknown = set(obj) - SPEC_KEYS
    if unknown:
        raise ValueError(f"unknown member spec keys {sorted(unknown)} (allowed {sorted(SPEC_KEYS)})")
    sp = dict(SPEC_DEFAULT)
    sp.update({k: v for k, v in obj.items() if k != "name"})
    if sp["driver"] not in DRIVERS:
        raise ValueError(f"driver must be one of {DRIVERS}, not {sp['driver']!r}")
    if not isinstance(sp["passive"], str) or not PASSIVE_RE.match(sp["passive"]):
        raise ValueError(f"passive must be 'none' or a tag like c1_r1, not {sp['passive']!r}")
    if sp["sol_frac"] is not None:
        sp["sol_frac"] = _num(sp["sol_frac"], "sol_frac", 0.05, 1.5)
    if sp["zfix"] is not None:
        sp["zfix"] = _num(sp["zfix"], "zfix", -0.3, 0.3)
    if not isinstance(sp["limiter_cc"], bool):
        raise ValueError(f"limiter_cc must be true/false, not {sp['limiter_cc']!r}")
    if isinstance(sp["smooth_h"], bool) or not isinstance(sp["smooth_h"], int) or sp["smooth_h"] < 0:
        raise ValueError(f"smooth_h must be an integer >= 0, not {sp['smooth_h']!r}")
    if isinstance(sp["nbasis"], bool) or sp["nbasis"] not in (4, 6):
        raise ValueError(f"nbasis must be 4 or 6 (kfit3 current bases), not {sp['nbasis']!r}")
    sp["nbasis"] = int(sp["nbasis"])
    if sp["fallback"] not in FALLBACKS:
        raise ValueError(f"fallback must be one of {FALLBACKS}, not {sp['fallback']!r}")
    for k in ("heights_from", "targets_base"):
        if sp[k] is not None and (not isinstance(sp[k], str) or not sp[k].strip()):
            raise ValueError(f"{k} must be a member name or a directory, not {sp[k]!r}")
    if sp["zfix"] is not None and sp["smooth_h"]:
        raise ValueError("zfix holds a constant height; smooth_h would be ignored by fec/mast_zsc.py")
    if sp["driver"] == "zsc":
        if sp["heights_from"] is not None or sp["targets_base"] is not None:
            raise ValueError("heights_from / targets_base apply to driver zsc_bpli only")
    else:
        if sp["targets_base"] is None:
            raise ValueError("driver zsc_bpli needs targets_base (member whose maps feed the "
                             "beta_p / li target features)")
        bad = {k: sp[k] for k in ("zfix", "limiter_cc", "smooth_h", "fallback", "nbasis")
               if sp[k] != SPEC_DEFAULT[k]}
        if bad:
            raise ValueError(f"fec/mast_zsc_bpli.py does not implement {bad} (defaults only)")
    name = obj.get("name") or auto_name(sp)
    if not isinstance(name, str) or not NAME_RE.match(name) or name in RESERVED:
        raise ValueError(f"bad member name {name!r} (letters, digits, _ . -; max 64)")
    sp["name"] = name
    return sp


def load_recipe(src):
    """Validated MAST pipeline recipe (dict) from a JSON file path or a dict (module doc)."""
    path = None
    if isinstance(src, dict):
        raw = json.loads(json.dumps(src))
    else:
        path = Path(src)
        if not path.is_absolute() and not path.exists():
            path = REPO / path
        path = path.resolve()
        raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("recipe must be a JSON object")
    unknown = set(raw) - RECIPE_KEYS
    if unknown:
        raise ValueError(f"unknown recipe keys {sorted(unknown)} (allowed {sorted(RECIPE_KEYS)})")
    if not isinstance(raw.get("name"), str) or not NAME_RE.match(raw["name"]):
        raise ValueError(f"recipe needs a name matching {NAME_RE.pattern}")
    r = dict(name=raw["name"], description=raw.get("description", ""),
             boundary_limiter=raw.get("boundary_limiter", True),
             betan=raw.get("betan", "betan_v2.npy"),
             finish_members=raw.get("finish_members", False),
             members=[], merge=None, average=None, post=[], final=raw.get("final"),
             reference_public=raw.get("reference_public"), notes=raw.get("notes"))
    if not isinstance(r["boundary_limiter"], bool) or not isinstance(r["finish_members"], bool):
        raise ValueError("boundary_limiter / finish_members must be true/false")
    b = r["betan"]
    if not isinstance(b, str) or not b.endswith(".npy") or is_path_ref(b) or not NAME_RE.match(b):
        raise ValueError(f"betan must be a plain .npy file name, not {b!r}")
    mem = raw.get("members")
    if not isinstance(mem, list) or not mem:
        raise ValueError("recipe needs a non-empty members list")
    seen = []
    for m in mem:
        sp = parse_spec(m)
        if sp["name"] in seen:
            raise ValueError(f"duplicate member name {sp['name']!r}")
        for k in ("heights_from", "targets_base"):
            ref = sp[k]
            if ref is not None and not is_path_ref(ref) and ref not in seen:
                raise ValueError(f"member {sp['name']}: {k} {ref!r} must name an EARLIER member "
                                 f"(or be a directory)")
        seen.append(sp["name"])
        r["members"].append(sp)
    nodes = list(seen)
    if raw.get("merge"):
        mg = raw["merge"]
        if not isinstance(mg, dict) or set(mg) - {"name", "loff", "lon", "r_in_threshold"}:
            raise ValueError(f"merge must be {{name, loff, lon, r_in_threshold}}: {mg}")
        for k in ("loff", "lon"):
            if mg.get(k) not in seen:
                raise ValueError(f"merge.{k} {mg.get(k)!r} is not a member")
        if mg["loff"] == mg["lon"]:
            raise ValueError("merge.loff and merge.lon must differ")
        mg = dict(name=mg.get("name") or f"dc_{mg['loff']}", loff=mg["loff"], lon=mg["lon"],
                  r_in_threshold=_num(mg.get("r_in_threshold", 0.18), "r_in_threshold", 0.0, 1.0))
        if not NAME_RE.match(mg["name"]) or mg["name"] in nodes or mg["name"] in RESERVED:
            raise ValueError(f"bad or duplicate merge name {mg['name']!r}")
        nodes.append(mg["name"])
        r["merge"] = mg
    if raw.get("average"):
        av = raw["average"]
        if not isinstance(av, dict) or set(av) - {"name", "members", "weights"}:
            raise ValueError(f"average must be {{name, members, weights}}: {av}")
        ms = av.get("members")
        if not isinstance(ms, list) or len(ms) < 2 or len(set(ms)) != len(ms):
            raise ValueError("average.members must list >= 2 distinct nodes")
        for m in ms:
            if m not in nodes:
                raise ValueError(f"average member {m!r} is not a member / merge node")
        w = av.get("weights")
        if w is not None:
            if not isinstance(w, list) or len(w) != len(ms):
                raise ValueError("average.weights must match average.members")
            w = [_num(x, "weight", 1e-12, 1e12) for x in w]
        av = dict(name=av.get("name") or f"avg{len(ms)}", members=list(ms), weights=w)
        if not NAME_RE.match(av["name"]) or av["name"] in nodes or av["name"] in RESERVED:
            raise ValueError(f"bad or duplicate average name {av['name']!r}")
        nodes.append(av["name"])
        r["average"] = av
    post = raw.get("post")
    if post:
        if not isinstance(post, list):
            raise ValueError("post must be a list of {name, kind, of, sigma_ms}")
        for p in post:
            if not isinstance(p, dict) or set(p) - POST_KEYS:
                raise ValueError(f"post node must be {{name, kind, of, sigma_ms}}: {p}")
            if p.get("kind") not in POST_KINDS:
                raise ValueError(f"post kind must be one of {POST_KINDS}, not {p.get('kind')!r}")
            if p.get("of") not in nodes:
                raise ValueError(f"post.of {p.get('of')!r} is not an earlier node of the recipe")
            sig = _num(p.get("sigma_ms"), "sigma_ms", *TSM_SIGMA_RANGE)
            pn = dict(name=p.get("name") or f"{p['of']}_tsm{sig:g}", kind=p["kind"], of=p["of"],
                      sigma_ms=sig)
            if not NAME_RE.match(pn["name"]) or pn["name"] in nodes or pn["name"] in RESERVED:
                raise ValueError(f"bad or duplicate post node name {pn['name']!r}")
            nodes.append(pn["name"])
            r["post"].append(pn)
    if r["final"] is None:
        if r["post"]:
            r["final"] = r["post"][-1]["name"]
        elif r["average"]:
            r["final"] = r["average"]["name"]
        elif r["merge"]:
            r["final"] = r["merge"]["name"]
        elif len(seen) == 1:
            r["final"] = seen[0]
        else:
            raise ValueError("recipe with several members and no merge/average needs 'final'")
    if r["final"] not in nodes:
        raise ValueError(f"final {r['final']!r} is not a node of the recipe {nodes}")
    r["nodes"] = nodes
    r["file"] = str(path) if path else None
    return r


def recipe_passive_tags(recipe):
    """Passive-current tags the recipe's members read (fec_work/passive/frames/<cfg>/<tag>)."""
    return sorted({m["passive"] for m in recipe["members"] if m["passive"] != "none"})


def whole_shot_limit(shot, limit, align="shot", max_frames=MAST_CHUNK):
    """Frames of the run: all (limit 0 or >= N), else the smallest whole-shot prefix >= limit
    (align 'shot') or the smallest production shot-chunk prefix >= limit (align 'chunk',
    fec.mast_select_v2.shot_chunks, so stage-1 batches equal those of a full run)."""
    shot = np.asarray(shot)
    n = len(shot)
    if not limit or limit >= n:
        return int(n)
    if align == "chunk":
        from fec.mast_select_v2 import shot_chunks
        for sl in shot_chunks(shot, max_frames):
            if sl.stop >= limit:
                return int(sl.stop)
        return int(n)
    if align != "shot":
        raise ValueError(f"limit align must be 'shot' or 'chunk', not {align!r}")
    b = np.flatnonzero(np.diff(shot) != 0) + 1
    b = b[b >= limit]
    return int(b[0]) if len(b) else int(n)


# ============================================================================ plan
class Step:
    __slots__ = ("node", "name", "cmd", "outputs", "sig", "deps", "gpu", "params")

    def __init__(self, node, name, cmd, outputs, params, deps=(), gpu=False):
        self.node, self.name, self.cmd = node, name, [str(c) for c in cmd]
        self.outputs, self.params = list(outputs), params
        self.sig = sig_of(params)
        self.deps = [tuple(d) for d in deps if d]
        self.gpu = gpu

    @property
    def key(self):
        return (self.node, self.name)


def step_record_path(out, node, step):
    return Path(out) / node / "_steps" / f"{step}.json"


def read_record(out, node, step):
    return read_json(step_record_path(out, node, step))


class Plan:
    """Ordered steps of one invocation (members, merge, average, finishing)."""

    def __init__(self, out, config, N, n_full, owner="mastpipe", priority="normal", workers=6,
                 blim=True, betan="betan_v2.npy", allow_unreviewed=False):
        if config not in CONFIGS:
            raise ValueError(f"config must be one of {CONFIGS}, not {config!r}")
        self.out, self.config, self.N, self.n_full = Path(out), config, int(N), int(n_full)
        self.owner, self.priority, self.workers = owner, priority, int(workers)
        self.blim, self.betan, self.allow_unreviewed = bool(blim), betan, bool(allow_unreviewed)
        self.steps, self.by_key, self.producer, self.kind = [], {}, {}, {}

    def node_dir(self, node):
        return self.out / node

    def _add(self, st):
        if st.key in self.by_key:
            raise PipelineError(f"step {st.key} planned twice")
        self.steps.append(st)
        self.by_key[st.key] = st

    def _gpu(self, cmd):
        return [PY, "-B", FEC / "gpu_slot.py", "run", "--owner", self.owner, "--priority",
                self.priority, "--"] + list(cmd)

    def limit_args(self):
        return ["--limit", self.N] if self.N < self.n_full else []

    def ref(self, ref):
        """(dir, identity, dependency key or None) of a node of this plan, a finished node of
        this root (earlier invocation) or an external stage-1 directory."""
        if is_path_ref(ref):
            d = resolve_dir(ref)
            meta = d / "meta.npz"
            ident = dict(path=str(d), meta_sha256=sha256(meta) if meta.exists() else None)
            return d, ident, None
        if ref in self.producer:
            k = self.producer[ref]
            return self.node_dir(ref), self.by_key[k].sig, k
        for st in PRODUCERS:
            rec = read_record(self.out, ref, st)
            if rec and rec.get("status") == "ok":
                return self.node_dir(ref), rec["sig"], (ref, st)
        raise PipelineError(f"{ref!r} is neither a node of this plan, a finished node of "
                            f"{self.out} nor a directory")

    def add_member(self, spec):
        spec = parse_spec(spec)
        name = spec["name"]
        self._new_node(name)
        d = self.node_dir(name)
        s = {k: v for k, v in spec.items() if k not in ("name", "heights_from", "targets_base")}
        if s.get("nbasis") == 4:              # default: keep pre-nbasis signatures and commands
            del s["nbasis"]
        params = dict(kind="stage1", config=self.config, N=self.N, spec=s)
        deps = []
        if spec["driver"] == "zsc":
            inner = [PY, "-B", FEC / "mast_zsc.py", "--out", d, "--config", self.config,
                     "--passive", s["passive"], "--fallback", s["fallback"],
                     "--smooth-h", s["smooth_h"]]
            if s["limiter_cc"]:
                inner.append("--limiter-cc")
            if s["sol_frac"] is not None:
                inner += ["--sol-frac", repr(s["sol_frac"])]
            if s["zfix"] is not None:
                inner += ["--zfix", repr(s["zfix"])]
            if s.get("nbasis", 4) != 4:
                inner += ["--nbasis", s["nbasis"]]
            inner += self.limit_args()
        else:
            bdir, bid, bkey = self.ref(spec["targets_base"])
            tp = dict(kind="bpli_targets", config=self.config, N=self.N, base=bid)
            tcmd = [PY, "-B", FEC / "mast_pipeline.py", "_bpli_targets", "--config", self.config,
                    "--base", bdir, "--out", d, "--nproc", min(4, self.workers), "--n", self.N]
            self._add(Step(name, "targets", tcmd, ["bpli_targets.npz"], tp, [bkey]))
            params["targets"] = self.by_key[(name, "targets")].sig
            deps.append((name, "targets"))
            inner = [PY, "-B", FEC / "mast_pipeline.py", "_bpli_stage1", "--config", self.config,
                     "--out", d, "--passive", s["passive"], "--targets", d / "bpli_targets.npz"]
            if s["sol_frac"] is not None:
                inner += ["--sol-frac", repr(s["sol_frac"])]
            inner += self.limit_args()
            if spec["heights_from"] is not None:
                hdir, hid, hkey = self.ref(spec["heights_from"])
                params["heights_from"] = hid
                inner += ["--heights-from", hdir]
                deps.append(hkey)
            else:
                params["heights_from"] = None      # full constrained secant (slow, ~7 h public)
        self._add(Step(name, "stage1", self._gpu(inner), ["psi.npy", "meta.npz", "config.json"],
                       params, deps, gpu=True))
        self.producer[name] = (name, "stage1")
        self.kind[name] = "member"
        return name

    def _new_node(self, name):
        if not isinstance(name, str) or not NAME_RE.match(name) or name in RESERVED:
            raise PipelineError(f"bad node name {name!r} (letters, digits, _ . -; max 64)")
        if name in self.kind:
            raise PipelineError(f"node {name} planned twice")

    def add_merge(self, name, loff, lon, r_in_threshold=0.18):
        self._new_node(name)
        (d1, i1, k1), (d2, i2, k2) = self.ref(loff), self.ref(lon)
        p = dict(kind="merge", config=self.config, N=self.N, loff=i1, lon=i2,
                 r_in_threshold=float(r_in_threshold), rule="deep_crossing")
        cmd = [PY, "-B", FEC / "mast_pipeline.py", "_merge", "--config", self.config,
               "--loff", d1, "--lon", d2, "--out", self.node_dir(name),
               "--threshold", repr(float(r_in_threshold)), "--n", self.N]
        self._add(Step(name, "merge", cmd, ["psi.npy", "meta.npz", "merge.json", "config.json"],
                       p, [k1, k2]))
        self.producer[name] = (name, "merge")
        self.kind[name] = "merge"
        return name

    def add_average(self, name, members, weights=None):
        self._new_node(name)
        refs = [self.ref(m) for m in members]
        w = [1.0] * len(members) if weights is None else [float(x) for x in weights]
        if len(w) != len(refs) or min(w) <= 0:
            raise PipelineError(f"weights {w} do not match members {members}")
        p = dict(kind="average", config=self.config, N=self.N, members=[r[1] for r in refs],
                 weights=w)
        inner = [PY, "-B", FEC / "mast_pipeline.py", "_average", "--config", self.config,
                 "--out", self.node_dir(name), "--n", self.N]
        for r in refs:
            inner += ["--member", r[0]]
        for x in w:
            inner += ["--weight", repr(x)]
        self._add(Step(name, "average", self._gpu(inner), ["psi.npy", "meta.npz", "config.json"],
                       p, [r[2] for r in refs], gpu=True))
        self.producer[name] = (name, "average")
        self.kind[name] = "average"
        return name

    def add_finish(self, node):
        k = self.producer[node]
        src = self.by_key[k].sig
        d = self.node_dir(node)
        p2 = dict(kind="stage2", config=self.config, input=src, boundary_limiter=self.blim)
        c2 = [PY, "-B", FEC / "mast_v1.py", "2", "--config", self.config, "--out", d] + \
            (["--boundary-limiter"] if self.blim else []) + \
            ([] if self.workers == 6 else ["--nproc", self.workers])   # stage-2 pool; default 6 = the old fixed pool
        self._add(Step(node, "stage2", c2, ["scalars.npz"], p2, [k]))
        pb = dict(kind="betan", config=self.config, input=src, boundary_limiter=self.blim,
                  name=self.betan)
        cb = [PY, "-B", FEC / "mast_pipeline.py", "_betan", "--config", self.config, "--dir", d,
              "--name", self.betan, "--nproc", self.workers] + \
            (["--blim"] if self.blim else []) + \
            (["--allow-unreviewed"] if self.allow_unreviewed else [])
        self._add(Step(node, "betan", cb, [self.betan, "betan_records.npz",
                                           Path(self.betan).stem + ".json"], pb, [k]))

    def finished_ref(self, ref):
        """(dir, identity, dependency keys) of a node whose maps AND scalars / betaN a post step
        reads: a non-post node needs its stage2 + betan steps (planned here or finished in this
        root), a post node its tsm step (which copied them), a directory the files themselves."""
        from fec.mast_tsm import copy_names
        d, ident, key = self.ref(ref)
        if key is None:                         # external directory
            miss = [f for f in copy_names(self.betan) if not (d / f).exists()]
            if miss:
                raise PipelineError(f"{d} lacks {miss}: finish it first (stage 2 + betaN)")
            return d, dict(ident, scalars_sha256=sha256(d / "scalars.npz"),
                           betan_sha256=sha256(d / self.betan)), []
        if key[1] == "tsm":
            return d, ident, [key]
        fin = {}
        for st in ("stage2", "betan"):
            k = (key[0], st)
            if k in self.by_key:
                fin[st] = (k, self.by_key[k].sig)
                continue
            rec = read_record(self.out, *k)
            if not (rec and rec.get("status") == "ok"):
                raise PipelineError(f"node {key[0]} has no finished {st} step: a post node copies "
                                    f"its scalars / betaN (plan add_finish({key[0]!r}) first)")
            fin[st] = (k, rec["sig"])
        return d, dict(maps=ident, stage2=fin["stage2"][1], betan=fin["betan"][1]),             [key, fin["stage2"][0], fin["betan"][0]]

    def add_tsm(self, name, of, sigma_ms, device="cuda"):
        """Post node: within-shot Gaussian smoothing of node `of`'s maps (fec/mast_tsm.py, meta
        recomputed on the GPU), its scalars.npz / betaN files copied unchanged."""
        from fec.mast_tsm import copy_names
        self._new_node(name)
        sig = _num(sigma_ms, "sigma_ms", *TSM_SIGMA_RANGE)
        d, ident, deps = self.finished_ref(of)
        p = dict(kind="tsm", config=self.config, N=self.N, src=ident, sigma_ms=sig,
                 betan=self.betan, device=device)
        inner = [PY, "-B", FEC / "mast_tsm.py", "--config", self.config, "--src", d,
                 "--out", self.node_dir(name), "--sigma-ms", repr(sig), "--n", self.N,
                 "--betan", self.betan, "--device", device]
        outputs = ["psi.npy", "meta.npz", "config.json"] + list(copy_names(self.betan))
        self._add(Step(name, "tsm", self._gpu(inner), outputs, p, deps, gpu=True))
        self.producer[name] = (name, "tsm")
        self.kind[name] = "tsm"
        return name


def post_base(recipe, node):
    """The non-post node at the root of a post chain (whose stage 2 / betaN the chain copies)."""
    post = {p["name"]: p for p in recipe.get("post") or []}
    while node in post:
        node = post[node]["of"]
    return node


def plan_recipe(recipe, plan):
    for m in recipe["members"]:
        plan.add_member(m)
    if recipe["merge"]:
        mg = recipe["merge"]
        plan.add_merge(mg["name"], mg["loff"], mg["lon"], mg["r_in_threshold"])
    if recipe["average"]:
        av = recipe["average"]
        plan.add_average(av["name"], av["members"], av["weights"])
    post = recipe.get("post") or []
    post_names = {p["name"] for p in post}
    bases = {post_base(recipe, p["name"]) for p in post}
    for node in recipe["nodes"]:
        if node in post_names:              # post nodes copy their scalars: never finished
            continue
        if node == recipe["final"] or node in bases or                 (recipe["finish_members"] and plan.kind[node] == "member"):
            plan.add_finish(node)
    for p in post:
        plan.add_tsm(p["name"], p["of"], p["sigma_ms"])
    return plan


# ============================================================================ execution
def _default_execute(cmd, log):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n# {now()} {cmdline(cmd)}\n")
        fh.flush()
        return subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(REPO), env=dict(os.environ, PYTHONUNBUFFERED="1")).returncode


class Runner:
    """Skip/execute the steps of a Plan (DAG order; --jobs concurrent independent steps)."""

    def __init__(self, plan, overwrite=False, jobs=1, log=print, execute=None, check=True):
        self.plan, self.overwrite, self.jobs = plan, bool(overwrite), max(1, int(jobs))
        self.log, self.execute = log, execute or _default_execute
        self.check = check                     # post-checks of the outputs (array lengths)
        self.state, self.results = {}, []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ state
    def _outputs_ok(self, st):
        d = self.plan.node_dir(st.node)
        if st.name == "stage1" and (d / "meta_partial.npz").exists():
            return False
        return all((d / o).exists() for o in st.outputs)

    def _dep_run_ids(self, st):
        return {f"{n}/{s}": (read_record(self.plan.out, n, s) or {}).get("run_id")
                for n, s in st.deps}

    def prepare(self):
        """state[key] = 'done' | 'todo'; raises PipelineError when a node of this root was made
        with another spec (unless overwrite)."""
        P = self.plan
        for st in P.steps:
            rec = read_record(P.out, st.node, st.name)
            d = P.node_dir(st.node)
            if rec and rec.get("sig") != st.sig and rec.get("status") == "ok" and not self.overwrite:
                raise PipelineError(
                    f"{d} step {st.name} was made with another spec (sig {rec.get('sig')} != "
                    f"{st.sig}); use another --out, a new node name or --overwrite")
            if rec is None and st.name in PRODUCERS and not self.overwrite and \
                    any((d / o).exists() for o in ("psi.npy", "meta.npz")):
                raise PipelineError(f"{d} holds stage-1 outputs without a step record; refusing "
                                    f"to overwrite (use another --out or --overwrite)")
            upstream_todo = any(self.state.get(k) == "todo" for k in st.deps)
            done = (rec is not None and rec.get("status") == "ok" and rec.get("sig") == st.sig
                    and self._outputs_ok(st) and not upstream_todo
                    and rec.get("dep_run_ids", {}) == self._dep_run_ids(st))
            self.state[st.key] = "done" if done else "todo"
        return self.state

    # ------------------------------------------------------------------ checks
    def post_check(self, st):
        P = self.plan
        d = P.node_dir(st.node)
        info = {}
        if st.name in PRODUCERS:
            shp = np.load(d / "psi.npy", mmap_mode="r").shape
            if shp[0] != P.N or tuple(shp[1:]) != (65, 65):
                raise PipelineError(f"{d}/psi.npy shape {shp} != ({P.N}, 65, 65)")
            with np.load(d / "meta.npz") as z:
                bad = [k for k in z.files if z[k].ndim == 1 and len(z[k]) != P.N]
                if bad:
                    raise PipelineError(f"{d}/meta.npz keys {bad} have length != {P.N}")
                if "method" in z.files:
                    info["kfit_frac"] = float(np.nanmean(z["method"]))
                if "Zax" in z.files:
                    info["abs_zax_median"] = float(np.nanmedian(np.abs(z["Zax"])))
                if "zsc_conv" in z.files:
                    info["zsc_conv"] = float(np.nanmean(z["zsc_conv"]))
            info["psi_sha256"] = sha256(d / "psi.npy")
            info["meta_sha256"] = sha256(d / "meta.npz")
            cfg = read_json(d / "config.json") or {}
            if cfg.get("config") not in (None, P.config):
                raise PipelineError(f"{d}/config.json config {cfg.get('config')} != {P.config}")
            if st.name == "tsm":
                info.update(scalars_sha256=sha256(d / "scalars.npz"),
                            betan_sha256=sha256(d / P.betan), sigma_ms=cfg.get("sigma_ms"),
                            src=cfg.get("src"))
        elif st.name == "targets":
            with np.load(d / "bpli_targets.npz") as z:
                if len(z["li_t"]) < P.N:
                    raise PipelineError(f"targets cover {len(z['li_t'])} < {P.N} frames")
                info["li_finite"] = float(np.isfinite(z["li_t"][:P.N]).mean())
        elif st.name == "stage2":
            with np.load(d / "scalars.npz") as z:
                bad = [k for k in z.files if len(z[k]) != P.N]
                if bad:
                    raise PipelineError(f"{d}/scalars.npz keys {bad} have length != {P.N}")
                info["finite"] = {k: float(np.isfinite(z[k]).mean()) for k in z.files}
                info["q95_median"] = float(np.nanmedian(z["q95"]))
            info["scalars_sha256"] = sha256(d / "scalars.npz")
        elif st.name == "betan":
            b = np.load(d / P.betan)
            if len(b) != P.N:
                raise PipelineError(f"{d}/{P.betan} has {len(b)} != {P.N} rows")
            info.update(median=float(np.median(b)), finite=float(np.isfinite(b).mean()),
                        sha256=sha256(d / P.betan))
        return info

    # ------------------------------------------------------------------ execution
    def exec_step(self, st):
        P = self.plan
        d = P.node_dir(st.node)
        d.mkdir(parents=True, exist_ok=True)
        logf = P.out / "logs" / f"{st.node}.{st.name}.log"
        rec = dict(node=st.node, step=st.name, sig=st.sig, params=st.params, cmd=cmdline(st.cmd),
                   run_id=uuid.uuid4().hex, status="running", started=now(), log=str(logf),
                   dep_run_ids=self._dep_run_ids(st), host=platform.node(), pid=os.getpid(),
                   config=P.config, N=P.N)
        write_json(step_record_path(P.out, st.node, st.name), rec)
        with self._lock:
            self.log(f"[{time.strftime('%H:%M:%S')}] run  {st.node}/{st.name}: {cmdline(st.cmd)}")
        t0 = time.time()
        try:
            rc = self.execute(st.cmd, logf)
            if rc:
                raise PipelineError(f"exit code {rc} (log {logf})")
            if not self._outputs_ok(st):
                raise PipelineError(f"outputs missing after {st.node}/{st.name}: {st.outputs}")
            rec["post"] = self.post_check(st) if self.check else {}
            rec["status"] = "ok"
        except Exception as e:
            rec.update(status="failed", error=f"{type(e).__name__}: {e}")
            raise
        finally:
            rec.update(finished=now(), seconds=round(time.time() - t0, 1))
            write_json(step_record_path(P.out, st.node, st.name), rec)
            with self._lock:
                self.log(f"[{time.strftime('%H:%M:%S')}] {rec['status']:4s} {st.node}/{st.name} "
                         f"{rec['seconds']:.0f}s")
        return rec

    def run(self, on_update=None):
        P = self.plan
        if not self.state:
            self.prepare()
        order = [st.key for st in P.steps]
        done = {k for k in order if self.state[k] == "done"}
        for k in order:
            if k in done:
                rec = read_record(P.out, *k) or {}
                self.results.append(dict(node=k[0], step=k[1], sig=P.by_key[k].sig,
                                         state="skipped", status="ok", log=rec.get("log"),
                                         seconds=0.0, cmd=cmdline(P.by_key[k].cmd)))
                self.log(f"skip {k[0]}/{k[1]} (done {rec.get('finished')})")
        todo = [k for k in order if k not in done]
        running, failed = {}, None
        with ThreadPoolExecutor(max_workers=self.jobs) as ex:
            while todo or running:
                if failed is None:
                    for k in list(todo):
                        if len(running) >= self.jobs:
                            break
                        st = P.by_key[k]
                        if all(dk in done or dk not in P.by_key for dk in st.deps):
                            todo.remove(k)
                            running[ex.submit(self.exec_step, st)] = k
                if not running:
                    if todo and failed is None:
                        raise PipelineError(f"unsatisfiable dependencies: {todo}")
                    break
                fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for f in fin:
                    k = running.pop(f)
                    st = P.by_key[k]
                    try:
                        rec = f.result()
                        done.add(k)
                        self.results.append(dict(node=k[0], step=k[1], sig=st.sig, state="ran",
                                                 status="ok", log=rec["log"],
                                                 seconds=rec["seconds"], cmd=rec["cmd"],
                                                 post=rec.get("post")))
                    except Exception as e:  # noqa: BLE001
                        failed = failed or (k, e)
                        self.results.append(dict(node=k[0], step=k[1], sig=st.sig, state="ran",
                                                 status="failed", error=f"{type(e).__name__}: {e}",
                                                 cmd=cmdline(st.cmd)))
                    if on_update:
                        on_update(self.results)
                if failed:
                    todo = []
        if failed:
            raise PipelineError(f"step {failed[0][0]}/{failed[0][1]} failed: {failed[1]}")
        return self.results


class RootLock:
    """One pipeline writer per output root (pid file; a dead holder is cleared)."""

    def __init__(self, out):
        self.f = Path(out) / ".pipeline.lock"

    def __enter__(self):
        self.f.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    pid = int(self.f.read_text().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                import psutil
                if pid and psutil.pid_exists(pid):
                    raise PipelineError(f"{self.f.parent} is in use by pid {pid}")
                self.f.unlink(missing_ok=True)
        raise PipelineError(f"cannot lock {self.f}")

    def __exit__(self, *exc):
        self.f.unlink(missing_ok=True)


# ============================================================================ in-process steps
def finish_betan(d, config, name="betan_v2.npy", blim=True, nproc=6, allow_unreviewed=False,
                 log=print):
    """Frozen lam_fgw betaN of one MAST dir: exactly fec_work/mastz/mz_finish_cand.py (and
    fec/run_final.py st_mast_betan) with the config as a parameter."""
    from fec.features import load_ds
    from fec.run_final import betan_apply, betan_frozen_files, betan_records
    d = Path(d)
    frozen = betan_frozen_files()
    bad = [k for k, v in frozen.items() if not v["reviewed"]]
    if bad and not allow_unreviewed:
        raise SystemExit(f"betaN frozen files changed: {bad} (--allow-unreviewed is root's call)")
    rec = betan_records(config, d / "psi.npy", nproc=nproc, boundary_limiter=blim, log=log)
    np.savez(d / "betan_records.npz", **rec)
    with np.load(d / "betan_records.npz") as z:      # reload as mz_finish_cand / run_final do
        rec = {k: z[k] for k in z.files}
    n = np.load(d / "psi.npy", mmap_mode="r").shape[0]
    shot_all = np.asarray(load_ds(config)["shot"])[:n]
    out, info = betan_apply(rec, shot_all)
    np.save(d / name, out.astype(np.float64))
    info.update(mast_dir=str(d), config=config, boundary_limiter=bool(blim), file=str(d / name),
                sha256=sha256(d / name), frozen=frozen, geo_ok=float(np.mean(rec["ok_geo"])),
                records_sha256=sha256(d / "betan_records.npz"), unreviewed=bad)
    write_json(d / (Path(name).stem + ".json"), info)
    log("FINISHED " + json.dumps({k: info[k] for k in ("n", "median", "geo_ok")}))
    return info


def v3_meta_fn(device="cuda"):
    """Production-format meta of a map batch (fec_work/root_var/avg_maps.py): V3Solver axis,
    X-points and boundary flux (no solver limiter, final=True)."""
    import torch
    from fec.machines import mast_setup
    from fec.mast_select_v3 import V3Solver
    spec, M = mast_setup()
    S = V3Solver(spec, M, device=device, limiter=None)

    def fn(m):
        with torch.no_grad():
            ps = torch.as_tensor(m, device=S.dev, dtype=S.dt)
            pa, Ra, Za = S.find_axis(ps)
            xp = S.find_xpoints(ps, Ra, Za, pa)
            pb, lim = S.boundary(ps, pa, Ra, Za, xp, final=True)
        return {k: v.detach().cpu().numpy()
                for k, v in zip(AVG_META_KEYS, (pa, pb, Ra, Za, lim))}
    return fn


def average_maps(member_dirs, weights, out_dir, meta_fn=None, chunk=2048, n=None, config=None,
                 log=print):
    """Weighted mean of K psi maps (float64 accumulation in member order, one float32 rounding:
    the arithmetic of avg_maps.py) with production-format meta recomputed on the mean maps."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    W = np.asarray(weights, dtype=float)
    W = W / W.sum()
    P = [np.load(Path(p) / "psi.npy", mmap_mode="r") for p in member_dirs]
    if len({x.shape for x in P}) != 1:
        raise ValueError(f"member psi shapes differ: {[x.shape for x in P]}")
    N = P[0].shape[0]
    if n is not None and N != int(n):
        raise ValueError(f"members have {N} frames, the run has {n}")
    for p in member_dirs:
        c = read_json(Path(p) / "config.json") or {}
        if config and c.get("config") not in (None, config):
            raise ValueError(f"{p} was made for {c.get('config')}, not {config}")
    meta_fn = meta_fn or v3_meta_fn()
    tmp = out / "psi.part.npy"
    psi_out = np.lib.format.open_memmap(tmp, "w+", np.float32, P[0].shape)
    meta = {k: np.full(N, np.nan) for k in AVG_META_KEYS}
    meta["method"] = np.ones(N)
    t0 = time.time()
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        m = sum(w * np.asarray(x[a:b], dtype=np.float64) for x, w in zip(P, W)).astype(np.float32)
        psi_out[a:b] = m
        r = meta_fn(m)
        for k in AVG_META_KEYS:
            meta[k][a:b] = r[k]
        log(f"{b}/{N} {time.time() - t0:.0f}s")
    psi_out.flush()
    del psi_out
    os.replace(tmp, out / "psi.npy")
    np.savez(out / "meta.npz", **meta)
    write_json(out / "config.json", dict(kind="map average", config=config, N=int(N),
                                         cands=[str(p) for p in member_dirs],
                                         weights=W.tolist(), code="fec/mast_pipeline.py average"))
    log("AVG_DONE")
    return dict(N=int(N), weights=W.tolist())


def merge_dc(loff, lon, out, config, r_in_threshold=0.18, n=None, grid=None):
    """Deep-crossing merge (fec.run_final.merge_deep_crossing: limiter-off frames whose inner
    midplane separatrix R_in < threshold take the limiter-on solution).  The exact config pair
    check of run_final is replaced by: same config, same frame count, loff a limiter-off and lon
    a limiter-on stage-1 run (config.json 'limiter'; missing = off)."""
    import fec.run_final as RF
    loff, lon, out = Path(loff), Path(lon), Path(out)
    co, cl = RF._read_cfg(loff), RF._read_cfg(lon)
    for d, c in ((loff, co), (lon, cl)):
        if c.get("config") not in (None, config):
            raise ValueError(f"{d} was made for {c.get('config')}, not {config}")
    no = np.load(loff / "psi.npy", mmap_mode="r").shape
    nl = np.load(lon / "psi.npy", mmap_mode="r").shape
    if no != nl or (n is not None and no[0] != int(n)):
        raise ValueError(f"psi shapes {no} / {nl} (run N {n})")
    if (co.get("limiter") or "off") != "off" or (cl.get("limiter") or "off") != "on":
        raise ValueError(f"loff must be a limiter-off and lon a limiter-on (--limiter-cc) run: "
                         f"{co.get('limiter')} / {cl.get('limiter')}")
    orig = RF._read_cfg
    RF._read_cfg = lambda d: {}         # pair check replaced by the checks above
    try:
        s = RF.merge_deep_crossing(loff, lon, out, r_in_threshold=r_in_threshold, grid=grid)
    finally:
        RF._read_cfg = orig
    s.update(loff_config=co, lon_config=cl)
    RF.write_json(out / "merge.json", s)
    RF.write_json(out / "config.json", dict(
        config=config, N=int(s["N"]), limiter="merged",
        merge={k: s[k] for k in ("rule", "r_in_threshold", "take_no_boundary")},
        loff=str(loff), lon=str(lon), loff_config=co, lon_config=cl,
        code="fec/mast_pipeline.py merge-dc"))
    return s


# ============================================================================ build
def _subset_ds(src, dst, n_frames=None, n_shots=None):
    """Whole-shot prefix copy of a ds dir (cur, ts, t_ms, shot, fidx; per-shot info lists
    truncated) for subset smoke builds.  Returns (frames, shots)."""
    src, dst = Path(src), Path(dst)
    info = json.loads((src / "info.json").read_text())
    shot = np.load(src / "shot.npy")
    if n_shots is not None:
        n_frames = int(np.searchsorted(shot, int(n_shots)))
    n_frames = int(n_frames)
    if n_frames < len(shot) and shot[n_frames - 1] == shot[n_frames]:
        raise ValueError(f"{n_frames} frames is not a whole-shot prefix of {src}")
    k = int(shot[n_frames - 1]) + 1
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("cur.npy", "ts.npy", "t_ms.npy", "shot.npy", "fidx.npy"):
        np.save(dst / f, np.asarray(np.load(src / f, mmap_mode="r")[:n_frames]))
    n_order = len(info["order"])
    new = {key: (v[:k] if isinstance(v, list) and len(v) == n_order else v)
           for key, v in info.items()}
    new.update(N=n_frames, subset_of=str(src), subset_note="smoke scaffold, never submittable")
    (dst / "info.json").write_text(json.dumps(new))
    return n_frames, k


def build(mast_dir, d3d_dir, name, config, psi_dtype="float32", betan="betan_v2.npy",
          subset=False, d3d_ds=None, d3d_subset_shots=4, log=print, dry_run=False):
    """fec/build_submission.py + fec/submit.py validate of both files (chain_variant.py with the
    split as a parameter).  A MAST dir shorter than the fold needs subset=True: the build then
    runs against a whole-shot subset scaffold (<name>_view, FEC_WORK) with the first
    d3d_subset_shots DIII-D shots -- a code-path smoke only, never submittable."""
    from fec.paths import WORK
    from fec.run_final import ensure_dir_link, is_dir_link
    if config not in CONFIGS:
        raise ValueError(config)
    split = config[len("mast_"):]
    dcfg = f"diii_d_{split}"
    mast_dir, d3d_dir, name = Path(mast_dir), Path(d3d_dir), Path(name)
    n_mast = np.load(mast_dir / "psi.npy", mmap_mode="r").shape[0]
    n_fold = json.loads((WORK / "ds" / config / "info.json").read_text())["N"]
    d3d_src = Path(d3d_ds) if d3d_ds else WORK / "ds" / dcfg
    if not (d3d_src / "info.json").exists():
        raise PipelineError(f"no DIII-D ds {d3d_src}: pass --d3d-ds <the ds the predictions were "
                            f"made from> (idx recipes: fec_work_idx/ds/{dcfg})")
    view, d3d_use, info = None, d3d_dir, dict(subset=False)
    train_link = None
    if n_mast != n_fold:
        if not subset:
            raise PipelineError(f"{mast_dir} has {n_mast} frames, {config} has {n_fold}: pass "
                                f"--subset for a smoke build (never submittable)")
        view = Path(str(name) + "_view")
        nm, km = _subset_ds(WORK / "ds" / config, view / "ds" / config, n_frames=n_mast)
        nd, kd = _subset_ds(d3d_src, view / "ds" / dcfg, n_shots=d3d_subset_shots)
        d3d_use = view / "preds" / d3d_dir.name
        d3d_use.mkdir(parents=True, exist_ok=True)
        for k in ("psi", "scal"):
            np.save(d3d_use / f"{dcfg}_{k}.npy",
                    np.asarray(np.load(d3d_dir / f"{dcfg}_{k}.npy", mmap_mode="r")[:nd]))
        train_link = view / "ds" / "diii_d_train"     # junction; removed after the build
        ensure_dir_link(train_link, WORK / "ds" / "diii_d_train")
        info = dict(subset=True, view=str(view), mast_frames=nm, mast_shots=km, d3d_frames=nd,
                    d3d_shots=kd)
    elif d3d_ds and Path(d3d_ds).resolve() != (WORK / "ds" / dcfg).resolve():
        view = Path(str(name) + "_view")       # ds the DIII-D predictions were made from
        for k, p in (("diii_d_train", WORK / "ds" / "diii_d_train"), (dcfg, Path(d3d_ds)),
                     (config, WORK / "ds" / config)):
            ensure_dir_link(view / "ds" / k, p, replace=True)
        info = dict(subset=False, view=str(view))
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if view:
        env["FEC_WORK"] = str(view)
    cmds = [(None, [PY, "-B", FEC / "build_submission.py", "--name", name, "--d3d", d3d_use,
                    "--mast", mast_dir, "--split", split, "--mast-betan", betan,
                    "--psi-dtype", psi_dtype])]
    for cfg in (dcfg, config):
        cmds.append((cfg, [PY, "-B", FEC / "submit.py", "validate", name / f"{cfg}.npz",
                           "--config", cfg, "--psi-dtype", psi_dtype, "--json"] +
                     (["--work", view] if view else [])))
    info["commands"] = [(f"FEC_WORK={view} " if view else "") + cmdline(c) for _, c in cmds]
    if dry_run:
        for c in info["commands"]:
            log("$ " + c)
        return info
    info["validate"] = {}
    try:
        for i, (cfg, c) in enumerate(cmds):
            log("$ " + info["commands"][i])
            r = subprocess.run([str(x) for x in c], cwd=str(REPO), env=env, capture_output=True,
                               text=True)
            if r.returncode:
                log(r.stdout[-3000:] + r.stderr[-3000:])
                raise PipelineError(f"exit code {r.returncode}: {info['commands'][i]}")
            if cfg:
                js = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
                s = json.loads(js[-1]) if js else {}
                info["validate"][cfg] = {k: s.get(k) for k in ("ok", "errors", "warnings", "shots",
                                                                "frames_expected", "sha256", "bytes")}
                log(f"validate {cfg}: ok={s.get('ok')} errors={s.get('errors')} "
                    f"warnings={s.get('warnings')} shots={s.get('shots')}")
    finally:
        # a smoke view must not keep a junction into the canonical TRAIN ds (a later recursive
        # delete of the output dir could follow it); os.rmdir removes the link only
        if train_link is not None and is_dir_link(train_link):
            os.rmdir(train_link)
    info.update(ok=all(v.get("ok") for v in info["validate"].values()),
                build=read_json(name / "build.json"), mast=str(mast_dir), d3d=str(d3d_dir),
                config=config, psi_dtype=psi_dtype)
    if info["subset"]:
        (name / "SUBSET_SMOKE_NOT_SUBMITTABLE.txt").write_text(
            f"Built from the first {info['mast_frames']} MAST frames ({info['mast_shots']} shots) "
            f"and {info['d3d_shots']} DIII-D shots against the scaffold {view}. Never submit.\n")
    write_json(name / "pipeline_build.json", info)
    return info


# ============================================================================ CLI
def _fold_shot(config):
    from fec.features import load_ds
    return np.asarray(load_ds(config)["shot"])


def _common(ap, out_required=True):
    ap.add_argument("--config", required=True, choices=CONFIGS)
    ap.add_argument("--out", required=out_required, help="pipeline root (one dir per node)")
    ap.add_argument("--limit", type=int, default=0, help="frames (rounded up to whole shots)")
    ap.add_argument("--limit-align", default="shot", choices=["shot", "chunk"])
    ap.add_argument("--owner", default="mastpipe", help="gpu_slot owner")
    ap.add_argument("--gpu-priority", default="normal", choices=["normal", "high"])
    ap.add_argument("--workers", type=int, default=6, help="CPU pool size (stage 2 / betaN)")
    ap.add_argument("--jobs", type=int, default=1, help="independent steps run concurrently")
    ap.add_argument("--blim", default="on", choices=["on", "off"],
                    help="stage 2 / betaN boundary-only limiter (recipes: their own setting)")
    ap.add_argument("--betan", default="betan_v2.npy")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="betaN: proceed if the frozen betaN files' hashes changed (root's call)")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute nodes of this root that were made with another spec")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--runlog", action="store_true", help="append start/finish rows to "
                    "research/fec_runs.jsonl (fec/runlog.py)")


def _make_plan(a, blim=None, betan=None):
    shot = _fold_shot(a.config)
    N = whole_shot_limit(shot, a.limit, a.limit_align)
    return Plan(a.out, a.config, N, len(shot), owner=a.owner, priority=a.gpu_priority,
                workers=a.workers, blim=(a.blim == "on") if blim is None else blim,
                betan=betan or a.betan, allow_unreviewed=a.allow_unreviewed)


def _execute_plan(a, plan, command, extra=None, pipeline_json=False, final=None):
    runner = Runner(plan, overwrite=a.overwrite, jobs=a.jobs)

    def show():
        state = runner.prepare()
        print(f"{command}: config {plan.config} N {plan.N}/{plan.n_full} out {plan.out}")
        for st in plan.steps:
            print(f"  [{state[st.key]:4s}] {st.node}/{st.name} sig {st.sig}"
                  + (f"  $ {cmdline(st.cmd)}" if a.dry_run else ""))
    if a.dry_run:
        show()
        return 0
    stamp = time.strftime("%Y%m%d%H%M%S")
    man = dict(kind="mast_pipeline", command=command, argv=sys.argv, config=plan.config,
               N=plan.N, n_full=plan.n_full, limit_requested=a.limit, limit_align=a.limit_align,
               out=str(plan.out), boundary_limiter=plan.blim, betan=plan.betan,
               started=now(), status="running", host=platform.node(), git=git_state(),
               code_sha256=code_hashes(), steps=[], **(extra or {}))
    if final:
        man.update(final=final, final_dir=str(plan.node_dir(final)))
    mf = plan.out / "manifests" / f"{stamp}_{command}.json"
    targets = [mf] + ([plan.out / "pipeline.json"] if pipeline_json else [])

    def save(results=None):
        if results is not None:
            man["steps"] = list(results)
        for f in targets:
            write_json(f, man)
    rid = f"mastpipe.{command}.{plan.out.name}.{stamp}"
    t0 = time.time()
    with RootLock(plan.out):
        show()                                  # skip/run state read under the root lock
        if a.runlog:
            from fec import runlog
            runlog.append(dict(run_id=rid, event="start", status="running", owner=a.owner,
                               purpose=f"MAST pipeline {command} {plan.config} N={plan.N}",
                               cmd="python -B fec/mast_pipeline.py " + " ".join(sys.argv[1:]),
                               log=str(plan.out / "logs"), outputs=str(plan.out)))
        save()
        try:
            runner.run(on_update=save)
            man["status"] = "ok"
        except Exception as e:
            man.update(status="failed", error=f"{type(e).__name__}: {e}")
            raise
        finally:
            man.update(finished=now(), seconds=round(time.time() - t0, 1))
            save(runner.results)
            if a.runlog:
                from fec import runlog
                runlog.append(dict(run_id=rid, event="finish", owner=a.owner,
                                   status="ok" if man["status"] == "ok" else "failed",
                                   metrics=dict(seconds=man["seconds"],
                                                ran=sum(r["state"] == "ran" for r in runner.results))))
    print(f"{command}: {man['status']} in {man['seconds']:.0f}s; manifest {mf}"
          + (f"; final {man.get('final_dir')}" if final else ""))
    return 0


def cmd_member(a):
    sp = parse_spec(a.spec)
    plan = _make_plan(a)
    plan.add_member(sp)
    if not a.no_finish:
        plan.add_finish(sp["name"])
    return _execute_plan(a, plan, "member", extra=dict(spec=sp), final=sp["name"])


def cmd_merge(a):
    plan = _make_plan(a)
    name = a.name or f"dc_{Path(a.loff).name}"
    plan.add_merge(name, a.loff, a.lon, a.threshold)
    if not a.no_finish:
        plan.add_finish(name)
    return _execute_plan(a, plan, "merge-dc", final=name)


def cmd_average(a):
    members = [m for m in a.members.split(",") if m]
    w = [float(x) for x in a.weights.split(",")] if a.weights else None
    plan = _make_plan(a)
    name = a.name or f"avg{len(members)}"
    plan.add_average(name, members, w)
    if not a.no_finish:
        plan.add_finish(name)
    return _execute_plan(a, plan, "average", final=name)


def cmd_tsm(a):
    plan = _make_plan(a)
    name = a.name or f"{Path(a.of).name}_tsm{a.sigma_ms:g}"
    plan.add_tsm(name, a.of, a.sigma_ms)
    return _execute_plan(a, plan, "tsm", extra=dict(of=a.of, sigma_ms=a.sigma_ms), final=name)


def cmd_recipe(a):
    r = load_recipe(a.recipe)
    plan = _make_plan(a, blim=r["boundary_limiter"], betan=r["betan"])
    plan_recipe(r, plan)
    f = Path(r["file"])
    extra = dict(recipe_name=r["name"], recipe_file=str(f), recipe_sha256=sha256(f), recipe=r)
    return _execute_plan(a, plan, "recipe", extra=extra, pipeline_json=True, final=r["final"])


def cmd_build(a):
    info = build(a.mast, a.d3d, a.name, a.config, psi_dtype=a.psi_dtype, betan=a.betan,
                 subset=a.subset, d3d_ds=a.d3d_ds, d3d_subset_shots=a.d3d_subset_shots,
                 dry_run=a.dry_run)
    if a.dry_run:
        return 0
    print(("BUILD_OK" if info["ok"] else "BUILD_FAILED") + f" {a.name}"
          + (" (SUBSET smoke, never submittable)" if info["subset"] else ""))
    return 0 if info["ok"] else 1


def cmd_status(a):
    out = Path(a.out)
    man = read_json(out / "pipeline.json")
    if man:
        print(f"pipeline.json: {man.get('recipe_name')} {man.get('config')} N {man.get('N')} "
              f"status {man.get('status')} final {man.get('final_dir')}")
    for f in sorted(out.glob("*/_steps/*.json")):
        r = read_json(f) or {}
        print(f"  {r.get('node', '?'):24s} {r.get('step', '?'):8s} {str(r.get('status')):8s} "
              f"{str(r.get('finished', '-')):26s} {r.get('seconds', '-')!s:>8}s sig {r.get('sig')}")
    return 0


# ---------------------------------------------------------------------------- internal steps
def _int_bpli_targets(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args(argv)
    from fec.mast_zsc_bpli import compute_targets
    nb = np.load(Path(a.base) / "psi.npy", mmap_mode="r").shape[0]
    if nb < a.n:
        raise SystemExit(f"targets base {a.base} has {nb} < {a.n} frames")
    compute_targets(a.base, a.out, config=a.config, nproc=a.nproc)
    return 0


def _int_bpli_stage1(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passive", default="c1_r1")
    ap.add_argument("--sol-frac", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--heights-from", default=None)
    a = ap.parse_args(argv)
    from fec.mast_zsc_bpli import stage1
    if a.heights_from:
        n = a.limit or len(_fold_shot(a.config))
        with np.load(Path(a.heights_from) / "meta.npz") as z:
            nz = len(z["Zt"])
        c = read_json(Path(a.heights_from) / "config.json") or {}
        if nz < n or c.get("config") not in (None, a.config):
            raise SystemExit(f"heights_from {a.heights_from}: {nz} frames of "
                             f"{c.get('config')} for a {n}-frame {a.config} run")
    stage1(a.out, config=a.config, passive=a.passive, sol_frac=a.sol_frac, limit=a.limit,
           targets=a.targets, heights_from=a.heights_from)
    return 0


def _int_average(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--weight", action="append", type=float, required=True)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args(argv)
    average_maps(a.member, a.weight, a.out, n=a.n, config=a.config,
                 log=lambda m: print(m, flush=True))
    return 0


def _int_merge(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--loff", required=True)
    ap.add_argument("--lon", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.18)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args(argv)
    s = merge_dc(a.loff, a.lon, a.out, a.config, r_in_threshold=a.threshold, n=a.n)
    print("MERGE_DONE", json.dumps({k: s[k] for k in ("N", "n_from_lon", "frac_from_lon",
                                                       "n_no_boundary_loff")}), flush=True)
    return 0


def _int_betan(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", default="betan_v2.npy")
    ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--blim", action="store_true")
    ap.add_argument("--allow-unreviewed", action="store_true")
    a = ap.parse_args(argv)
    finish_betan(a.dir, a.config, name=a.name, blim=a.blim, nproc=a.nproc,
                 allow_unreviewed=a.allow_unreviewed, log=lambda m: print(m, flush=True))
    return 0


INTERNAL = {"_bpli_targets": _int_bpli_targets, "_bpli_stage1": _int_bpli_stage1,
            "_average": _int_average, "_merge": _int_merge, "_betan": _int_betan}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in INTERNAL:
        return INTERNAL[argv[0]](argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("member", help="one stage-1 member (+ stage 2 + betaN)")
    _common(p)
    p.add_argument("--spec", required=True, help="inline JSON, a JSON file or a named spec "
                   f"{sorted(NAMED_SPECS)}")
    p.add_argument("--no-finish", action="store_true", help="stage 1 only")
    p.set_defaults(func=cmd_member)
    p = sub.add_parser("merge-dc", help="deep-crossing merge of a limiter-off/on member pair")
    _common(p)
    p.add_argument("--loff", required=True)
    p.add_argument("--lon", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--threshold", type=float, default=0.18)
    p.add_argument("--no-finish", action="store_true")
    p.set_defaults(func=cmd_merge)
    p = sub.add_parser("average", help="mean of member maps (+ stage 2 + betaN)")
    _common(p)
    p.add_argument("--members", required=True, help="comma list of node names or directories")
    p.add_argument("--weights", default=None, help="comma list (default equal)")
    p.add_argument("--name", default=None)
    p.add_argument("--no-finish", action="store_true")
    p.set_defaults(func=cmd_average)
    p = sub.add_parser("tsm", help="within-shot temporal smoothing of a finished node's maps "
                       "(fec/mast_tsm.py; scalars / betaN copied)")
    _common(p)
    p.add_argument("--of", required=True, help="finished node name of this root or a directory")
    p.add_argument("--sigma-ms", type=float, required=True)
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_tsm)
    p = sub.add_parser("recipe", help="a whole MAST recipe (idempotent)")
    _common(p)
    p.add_argument("--recipe", required=True, help="fec/recipes_mast/<x>.json")
    p.set_defaults(func=cmd_recipe)
    p = sub.add_parser("build", help="build_submission.py + submit.py validate")
    p.add_argument("--config", required=True, choices=CONFIGS)
    p.add_argument("--mast", required=True, help="finished MAST node dir")
    p.add_argument("--d3d", required=True, help="DIII-D preds dir (<cfg>_psi.npy, _scal.npy)")
    p.add_argument("--name", required=True, help="output submission dir")
    p.add_argument("--psi-dtype", default="float32", choices=["float16", "float32"])
    p.add_argument("--betan", default="betan_v2.npy")
    p.add_argument("--d3d-ds", default=None, help="ds dir the DIII-D preds were made from "
                   "(default canonical fec_work/ds/<cfg>, as chain_variant.py)")
    p.add_argument("--subset", action="store_true", help="smoke build of a limited MAST dir "
                   "against a whole-shot subset scaffold (never submittable)")
    p.add_argument("--d3d-subset-shots", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("status", help="step records of a root")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_status)
    a = ap.parse_args(argv)
    if getattr(a, "jobs", 1) < 1:
        ap.error("--jobs must be >= 1")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
