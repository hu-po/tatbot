"""The run ledger: <run_dir>/ledger.jsonl, one JSON object per line, append-only, fsync per line.

    {"t": 1790000000.123, "event": "sent",    "arm": "right", "op": "s0003", "index": 3, "arc_m": 0.0}
    {"t": ...,            "event": "done",    "arm": "right", "op": "s0003", "index": 3, "arc_m": 0.0421, "line": [...]}
    {"t": ...,            "event": "aborted", "arm": "right", "op": "s0004", "index": 4, "arc_m": 0.0133, "reason": "latched: estop"}
    {"t": ...,            "event": "skipped", "arm": "right", "op": "s0005", "index": 5}
    {"t": ...,            "event": "decision","arm": "right", "op": "s0005", "index": 5, "decision": "skip"}
    {"t": ...,            "event": "page",    "arm": "right", "page": {...}, "workspace": {...}}
    {"t": ...,            "event": "dip_phase", "arm": "right", "op": "d0002", "phase": "dwell", "status": "ok", ...}
    {"t": ...,            "event": "pen_trim", "arm": "right", "resource": "r1", "trim_m": -0.0002}

`t` is Unix seconds. `arc_m` is the stroke arc position (m): the start arc on `sent`, the reached arc
on `done`/`aborted` (the planned arc nearest the measured tip at the latch or failure); `line` the measured
pen-down samples in page metres. The next op is the first op whose last progress event is not `done`/`skipped`;
an `aborted` stroke resumes from its arc_m; a stroke whose last event is `sent` is uncertain until Decide REDRAW or
SKIP, and a tool change whose last event is `sent` waits for the operator's cartridge, which resuming the run says
is fitted. `page` keeps the run's measured page, which a resume with the same fitted tool uses again; `pen_trim`
the operator's pen trim for one resource (cartridge), the last of which a resume starts that resource at. Pure Python,
no ROS; the readers are tatbot_contracts.ros_progress, shared with the CLI.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from tatbot_contracts.ros_progress import (  # noqa: F401 -- the session's ledger API
    FINISHED,
    PROGRESS,
    counts,
    last_events,
    next_index,
    op_status,
    read,
    resume_arc,
    uncertain,
)

EVENTS = (*PROGRESS, "decision", "page", "dip_phase", "pen_trim")


class Ledger:
    """Append-only JSONL for one run; readable by every arm's executor."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mended = False

    def append(self, event: str, arm: str, op: str | None = None, index: int | None = None,
               arc_m: float | None = None, **extra) -> dict:
        if event not in EVENTS:
            raise ValueError(f"ledger event {event!r} not in {EVENTS}")
        row = {"t": round(time.time(), 3), "event": event, "arm": arm}
        if op is not None:
            row["op"] = op
        if index is not None:
            row["index"] = int(index)
        if arc_m is not None:
            row["arc_m"] = round(float(arc_m), 6)
        row.update(extra)
        with self._lock:
            self._mend()
            self._write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def _mend(self) -> None:
        """Once per instance: end a torn last line (a crash mid-write) so the next row starts its own line."""
        if self._mended:
            return
        self._mended = True
        try:
            with self.path.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                torn = stream.read(1) != b"\n"
        except OSError:  # missing or empty
            return
        if torn:
            self._write("\n")

    def _write(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())

    def rows(self) -> list[dict]:
        return read(self.path)
