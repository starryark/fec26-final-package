"""Append-only registry for competition runs (multi-hour GPU jobs the strict ledger cannot hold).

research/fec_runs.jsonl rows: {"run_id","event","time_utc",...}. Events: start, finish, note.
    python -B fec/runlog.py start --run d3d_v3 --purpose "..." --cmd "python -B fec/train_d3d.py ..." \
        --log train_d3d_v3.log --outputs fec_work/models/d3d_v3
    python -B fec/runlog.py finish --run d3d_v3 --status ok --metrics '{"S":0.994}'
    python -B fec/runlog.py note --run d3d_v3 --text "VAL every1 scored"
    python -B fec/runlog.py check              # validate every row against the schema
Records git HEAD + dirty flag at start. The strict research/experiments.jsonl ledger is FROZEN
for the paused routes (AGENTS.md); this file is the single active registry for competition work.
Cross-references: agents live in research/fec_agents.json (roster), submissions in
fec_work/submissions/index.jsonl (fec/submit.py maintains it).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "research" / "fec_runs.jsonl"

# Required keys per event (Optimize.md B3). Rows missing them are reported by `check`;
# `append` refuses to write a row that violates its own event schema.
SCHEMA = {"run_id": ("start", "finish", "note"),
          "event": ("start", "finish", "note"),
          "time_utc": ("start", "finish", "note"),
          "owner": ("start", "finish", "note"),
          "purpose": ("start",), "cmd": ("start",), "log": ("start",), "outputs": ("start",),
          "status": ("finish",), "metrics": ("finish",),
          "text": ("note",)}


def validate_row(row, lineno=None):
    """Return a list of schema violations for one registry row."""
    errs = []
    ev = row.get("event")
    if ev not in ("start", "finish", "note"):
        errs.append(f"unknown event {ev!r}")
        ev = None
    for key, events in SCHEMA.items():
        if ev in events and row.get(key) in (None, ""):
            errs.append(f"missing {key}")
    if not isinstance(row.get("run_id", ""), str) or not row.get("run_id"):
        errs.append("bad run_id")
    return [f"line {lineno}: {e}" if lineno else e for e in errs]


def check():
    if not REG.exists():
        print(f"no registry at {REG}")
        return 0
    errs = []
    n = 0
    for i, line in enumerate(REG.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        n += 1
        try:
            row = json.loads(line)
        except ValueError as e:
            errs.append(f"line {i}: not JSON ({e})")
            continue
        if not isinstance(row, dict):
            errs.append(f"line {i}: not an object")
            continue
        errs += validate_row(row, i)
    print(f"{n} rows checked: {'OK' if not errs else str(len(errs)) + ' violation(s)'}")
    for e in errs[:40]:
        print(" ", e)
    return 1 if errs else 0


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=20).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "fec"], cwd=REPO,
                                    capture_output=True, text=True, timeout=60).stdout.strip())
        return {"commit": head, "fec_dirty": dirty}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def append(row):
    row["time_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    errs = validate_row(row)
    if errs:
        print("ERROR: refusing to append a registry row that violates its schema:", file=sys.stderr)
        for e in errs:
            print(" ", e, file=sys.stderr)
        raise SystemExit(2)
    with open(REG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event", choices=["start", "finish", "note", "check"])
    ap.add_argument("--run", required=False)
    ap.add_argument("--purpose"); ap.add_argument("--cmd"); ap.add_argument("--log")
    ap.add_argument("--outputs"); ap.add_argument("--owner", default="root")
    ap.add_argument("--status", choices=["ok", "failed", "stopped", "superseded"])
    ap.add_argument("--metrics", help="JSON object"); ap.add_argument("--text")
    a = ap.parse_args()
    if a.event == "check":
        raise SystemExit(check())
    if not a.run:
        ap.error("--run is required for start/finish/note")
    row = {"run_id": a.run, "event": a.event, "owner": a.owner}
    if a.event == "start":
        row.update(purpose=a.purpose, cmd=a.cmd, log=a.log, outputs=a.outputs, status="running",
                   **git_state())
    elif a.event == "finish":
        row.update(status=a.status or "ok", metrics=json.loads(a.metrics) if a.metrics else {})
    else:
        row.update(text=a.text)
    append({k: v for k, v in row.items() if v is not None})


if __name__ == "__main__":
    main()
