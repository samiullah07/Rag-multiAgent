# Generated according to Phase 0 execution plan (Claude Code)
"""Utility functions for logging and retrieving experiment runs.

Each query (baseline or multi‑agent) must append a JSONL record to
`data/experiments/runs.jsonl`. The `append_run` function handles file creation
if the file does not yet exist. The `list_runs` function reads the file and
returns a list of dictionaries, returning an empty list when the file is
absent.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

RUNS_FILE = Path("data/experiments/runs.jsonl")
RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)

def append_run(record: Dict) -> None:
    """Append a single experiment record to the runs file.

    The record is expected to be JSON‑serialisable. An ISO‑8601 timestamp is
    added automatically if not present.
    """
    if "timestamp" not in record:
        record["timestamp"] = datetime.utcnow().isoformat()
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def list_runs() -> List[Dict]:
    """Return all experiment records as a list of dictionaries.

    If the runs file does not exist, an empty list is returned rather than
    raising an exception.
    """
    if not RUNS_FILE.is_file():
        return []
    with RUNS_FILE.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
