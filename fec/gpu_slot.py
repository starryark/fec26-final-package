"""Cooperative GPU slot scheduler (one RTX 5060 Ti shared by root and subagents).

Normal jobs share N slots; one extra slot is reserved for --priority high (root's
critical path, e.g. the production MAST run or a submission build).  A job waits until a
slot is free, records itself in fec.paths.GPU_SLOT_DIR/<slot>.json (machine-wide, independent of FEC_WORK), runs, and releases on exit.
Stale slots (dead PIDs) are reclaimed automatically.

    python -B fec/gpu_slot.py run --owner seq-agent -- python -B fec/train_d3d_seq.py train ...
    python -B fec/gpu_slot.py run --priority high --owner root -- python -B fec/mast_v1.py 1
    python -B fec/gpu_slot.py status

In-process: `with gpu_slot("owner"): ...`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.paths import GPU_SLOT_DIR  # noqa: E402

SLOT_DIR = GPU_SLOT_DIR
NORMAL_SLOTS = int(os.environ.get("FEC_GPU_SLOTS", "3"))
RESERVED = 1


def _alive(pid):
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:  # never use os.kill on Windows (it terminates the process)
        return True


def _clean():
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    held = []
    for f in SLOT_DIR.glob("*.json"):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            rec = {}
        pids = [p for p in (rec.get("pid"), rec.get("child_pid")) if p]
        if pids and not any(_alive(p) for p in pids):
            f.unlink(missing_ok=True)
            continue
        held.append((f, rec))
    return held


def _try_acquire(owner, priority, cmd):
    held = _clean()
    limit = NORMAL_SLOTS + (RESERVED if priority == "high" else 0)
    if len(held) >= limit:
        return None
    names = {f.stem for f, _ in held}
    for i in range(limit):
        name = f"slot{i}"
        if name in names:
            continue
        path = SLOT_DIR / f"{name}.json"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        rec = {"slot": name, "owner": owner, "priority": priority, "pid": os.getpid(),
               "cmd": " ".join(cmd)[:300], "since": time.strftime("%Y-%m-%d %H:%M:%S")}
        os.write(fd, json.dumps(rec).encode()); os.close(fd)
        return path
    return None


@contextlib.contextmanager
def gpu_slot(owner, priority="normal", cmd=(), poll=20, verbose=True):
    t0 = time.time(); last = 0
    while True:
        path = _try_acquire(owner, priority, list(cmd))
        if path:
            break
        if verbose and time.time() - last > 300:
            print(f"[gpu_slot] {owner} waiting for a GPU slot ({time.time() - t0:.0f}s)", flush=True)
            last = time.time()
        time.sleep(poll)
    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--owner", default="anon")
    r.add_argument("--priority", choices=["normal", "high"], default="normal")
    r.add_argument("argv", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "status":
        held = _clean()
        print(f"normal slots {NORMAL_SLOTS} + reserved {RESERVED}; held {len(held)}")
        for f, rec in held:
            print(f"  {f.stem}: {rec.get('owner')} [{rec.get('priority')}] pid {rec.get('pid')}/"
                  f"{rec.get('child_pid')} since {rec.get('since')}: {rec.get('cmd', '')[:100]}")
        return
    argv = a.argv[1:] if a.argv and a.argv[0] == "--" else a.argv
    with gpu_slot(a.owner, a.priority, argv) as path:
        p = subprocess.Popen(argv)
        rec = json.loads(Path(path).read_text()); rec["child_pid"] = p.pid
        Path(path).write_text(json.dumps(rec))
        sys.exit(p.wait())


if __name__ == "__main__":
    main()
