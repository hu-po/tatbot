"""The run ledger: <run_dir>/ledger.jsonl, one JSON object per line, append-only, fsync per line.

    {"t": 1790000000.123, "event": "sent",    "arm": "right", "op": "s0003", "index": 3, "arc_m": 0.0}
    {"t": ...,            "event": "done",    "arm": "right", "op": "s0003", "index": 3, "arc_m": 0.0421, "line": [...]}
    {"t": ...,            "event": "aborted", "arm": "right", "op": "s0004", "index": 4, "arc_m": 0.0133, "reason": "latched: estop"}
    {"t": ...,            "event": "skipped", "arm": "right", "op": "s0005", "index": 5}
    {"t": ...,            "event": "decision","arm": "right", "op": "s0005", "index": 5, "decision": "skip"}
    {"t": ...,            "event": "page",    "arm": "right", "page": {...}, "workspace": {...}}
    {"t": ...,            "event": "dip_phase", "arm": "right", "op": "d0002", "phase": "dwell", "status": "ok", ...}
    {"t": ...,            "event": "pen_trim", "arm": "right", "resource": "r1", "trim_m": -0.0002}
    {"t": ...,            "event": "dip_scale", "arm": "right", "op": "s0123", "resource": "r1", "scale": 0.9, "key": "+"}

`t` is Unix seconds (the session clock's utc), `t_clock` the session clock's own seconds. `arc_m` is the stroke arc position (m): the start arc on `sent`, the reached arc
on `done`/`aborted` (the planned arc nearest the measured tip at the latch or failure); `line` the measured
pen-down samples in page metres. The next op is the first op whose last progress event is not `done`/`skipped`;
an `aborted` stroke resumes from its arc_m; a stroke whose last event is `sent` is uncertain until Decide REDRAW or
SKIP, and a tool change whose last event is `sent` waits for the operator's cartridge, which resuming the run says
is fitted. `page` keeps the run's measured page, which a resume with the same fitted tool uses again; `pen_trim`
the operator's pen trim for one resource (cartridge), the last of which a resume starts that resource at (one a set-up
or resume eased back carries `eased_from_m`; a needle stroke's `sent` row carries the deepest trim it takes,
`trim_cap_m`); `dip_scale`
likewise its dip scale (tatbot_session.charge), a `+` row also the numpad's dip, which the goal it was pressed in makes
or drops: a resumed goal dips before its first stroke anyway, so none is made again. A compiled dip the charge still
covered is `skipped`. Pure Python, no ROS; the readers are tatbot_contracts.ros_progress, shared with the CLI.

GoalLog writes <run_dir>/goals.jsonl (stack.yaml session.goal_log): every trajectory goal sent, its op, tag
(ports.goal_tag), the session clock's time at sending (`stamp`) and where the goal starts on it (`start`; null when
the controller starts it at its first sample), and its knots as sent (t, q, qd).
"""
from __future__ import annotations

import json
import os
import threading
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

from tatbot_session.ports import WALL

EVENTS = (*PROGRESS, "decision", "page", "dip_phase", "pen_trim", "dip_scale")


class Ledger:
    """Append-only JSONL for one run; readable by every arm's executor."""

    def __init__(self, path: str | Path, clock=None):
        self.path, self.clock = Path(path), clock or WALL
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mended = False

    def append(self, event: str, arm: str, op: str | None = None, index: int | None = None,
               arc_m: float | None = None, **extra) -> dict:
        if event not in EVENTS:
            raise ValueError(f"ledger event {event!r} not in {EVENTS}")
        row = {"t": round(self.clock.utc(), 3), "t_clock": round(self.clock.now(), 6), "event": event, "arm": arm}
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


class GoalLog:
    """<run_dir>/goals.jsonl: one row per goal sent (module docstring)."""

    def __init__(self, path: str | Path):
        self.path, self._lock = Path(path), threading.Lock()

    def append(self, op: str, tag: str, stamp: float, start: float | None, t, q, qd) -> None:
        row = {"op": op, "tag": tag, "stamp": stamp, "start": start, "t": [float(v) for v in t],
               "q": [[float(v) for v in r] for r in q], "qd": [[float(v) for v in r] for r in qd]}
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
