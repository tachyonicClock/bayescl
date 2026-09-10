"""Run-artifact IO: JSONL trial logs, meta files, and run discovery."""

from __future__ import annotations

import json
from pathlib import Path


def score(acc: float, ece: float) -> float:
    """Aggregate objective: balance accuracy against calibration error."""
    return 0.5 * (acc + (1.0 - ece))


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def latest_run(parent: Path) -> Path:
    """Newest timestamped run directory under ``parent``.

    Run ids are ``%Y-%m-%d_%H-%M-%S`` timestamps, so lexicographic order is
    chronological order.
    """
    runs = sorted(p for p in parent.iterdir() if p.is_dir()) if parent.exists() else []
    if not runs:
        raise SystemExit(f"No runs under {parent} - run `tune` first.")
    return runs[-1]
