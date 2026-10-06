"""Readers of a ROS run's ledger.jsonl (tatbot_session.ledger), shared by the session, the CLI and research."""
from __future__ import annotations

import json
from pathlib import Path

PROGRESS = ("sent", "done", "aborted", "skipped")
FINISHED = ("done", "skipped")


def read(path: str | Path) -> list[dict]:
    """Every parseable line; a torn last line (a crash mid-write) is dropped."""
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def last_events(rows: list[dict], arm: str) -> dict[str, dict]:
    """op id -> the last sent/done/aborted/skipped row of that op for `arm` (decisions excluded)."""
    last: dict[str, dict] = {}
    for row in rows:
        if row.get("arm") == arm and row.get("op") and row.get("event") in PROGRESS:
            last[row["op"]] = row
    return last


def op_status(rows: list[dict], arm: str, op_id: str) -> str | None:
    """The last event of one op: None (never sent), sent, done, aborted or skipped."""
    row = last_events(rows, arm).get(op_id)
    return row["event"] if row else None


def next_index(ops: list[dict], rows: list[dict], arm: str, start: int = 0) -> int | None:
    """Index of the first op at or after `start` whose last event is not done/skipped; None when finished."""
    last = last_events(rows, arm)
    for index in range(start, len(ops)):
        row = last.get(ops[index]["id"])
        if row is None or row["event"] not in FINISHED:
            return index
    return None


def resume_arc(rows: list[dict], arm: str, op_id: str) -> float:
    """Where an op resumes: the arc of its last `aborted`, else 0 (never sent, or REDRAW from the start)."""
    row = last_events(rows, arm).get(op_id)
    if row and row["event"] == "aborted":
        return float(row.get("arc_m") or 0.0)
    return 0.0


def uncertain(ops: list[dict], rows: list[dict], arm: str) -> list[str]:
    """Stroke ids whose last event is `sent`: a crash left them; Decide REDRAW or SKIP resolves each. (A tool change
    left `sent` waits for its cartridge, and a dip left `sent` is dipped again: neither asks.)"""
    last = last_events(rows, arm)
    return [op["id"] for op in ops if op.get("op", "stroke") == "stroke" and last.get(op["id"], {}).get("event") == "sent"]


def counts(rows: list[dict], arm: str | None = None) -> dict[str, int]:
    """Ops whose last event is done / skipped, over `arm` or every arm."""
    arms = {row.get("arm") for row in rows} if arm is None else {arm}
    out = {"done": 0, "skipped": 0}
    for name in arms:
        for row in last_events(rows, name).values():
            if row["event"] in out:
                out[row["event"]] += 1
    return out
