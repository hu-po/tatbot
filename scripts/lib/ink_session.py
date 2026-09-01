"""The ink SESSION: one tool, one charge, many runs.

An episode is 30 s of one task; a tattoo is hours of many, with the same
needle carrying the same charge across all of them, dipping when it runs
dry. The sim's per-episode charge (tatbot_sim.env) and the robot's
per-invocation ``Charge`` (scripts/il_dip.py) both forgot this: nothing on
the bench held "how much ink is on the needle right now" between one
rollout and the next, so a dip was always a session start and a stroke was
never debited. This module is that memory.

One file per node, ``<log root>/ink/session.json``, holding the OPEN
session (there is at most one per node — one tool in one gripper). Every
mutation is also a ledger event (scripts/lib/ink_spec.py), so the session
file is a cache of the ledger's tail, not a second source of truth:
``rebuild()`` recovers it from the events.

    s = start(tool, policy)                # session.start in the ledger
    apply_dip(s, slot, ink_id, uptake_ul)  # after the arm has dipped
    apply_stroke(s, contact_mm, contact_s) # after a rollout is analysed
    end(s)                                 # session.end with the totals

A launcher that says ``--no-ink`` never touches this; one that says ``--dip``
opens a session if none is open. A rollout with neither leaves the session
as it found it, and its analysis debits the open one if there is one.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ink_spec
else:
    try:  # scripts/lib on sys.path
        import ink_spec
    except ImportError as err:  # loaded by path
        import importlib.util
        import sys

        _p = Path(__file__).resolve().with_name("ink_spec.py")
        _s = importlib.util.spec_from_file_location("ink_spec", _p)
        if _s is None or _s.loader is None:
            raise ImportError(f"Cannot load module spec for {_p}") from err
        ink_spec = importlib.util.module_from_spec(_s)
        sys.modules["ink_spec"] = ink_spec
        _s.loader.exec_module(ink_spec)

SCHEMA_VERSION = 1


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def session_path() -> Path:
    override = os.environ.get("TATBOT_INK_SESSION")
    if override:
        return Path(override).expanduser()
    return ink_spec.ledger_path().with_name("session.json")


@dataclass
class Session:
    session_id: str
    node: str
    tool_id: str
    mode: str                 # real | rehearsal
    capacity_ul: float
    started_utc: str
    ended_utc: str | None = None
    charge_ul: float = 0.0
    ink_id: str | None = None
    last_dip_utc: str | None = None
    dips: int = 0
    strokes: int = 0
    used_ul: float = 0.0
    contact_mm: float = 0.0
    contact_s: float = 0.0
    runs: list[str] = field(default_factory=list)
    need_ul: float | None = None
    """What the planned program will spend, if anyone said (ink.py
    mise-en-place --program); None is "unknown", and a dip is then always
    worth taking."""
    note: str | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def open(self) -> bool:
        return self.ended_utc is None

    @property
    def charge_frac(self) -> float:
        return 0.0 if self.capacity_ul <= 0 else max(0.0, min(1.0, self.charge_ul / self.capacity_ul))

    def remaining_need_ul(self) -> float | None:
        if self.need_ul is None:
            return None
        return max(0.0, self.need_ul - self.used_ul)

    def needs_dip(self, policy: ink_spec.InkPolicy, next_ul: float | None = None) -> str | None:
        """Why the next thing to do is a dip, or None if the charge will do.
        ``next_ul`` is what the next run is expected to spend (None: unknown,
        so anything but a fresh charge asks for a dip)."""
        if self.dips == 0 and self.charge_ul <= 0:
            return "session_start"
        if next_ul is None:
            return None if self.charge_ul >= policy.uptake_ul * 0.9 else "low_charge"
        return "low_charge" if next_ul > self.charge_ul else None


def _new_id(node: str) -> str:
    # a short random tail: two sessions opened in the same second (a forced
    # restart) must not share an id, the ledger keys on it
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "-" + node + "-" + uuid.uuid4().hex[:4]


def load(path: Path | None = None) -> Session | None:
    path = path or session_path()
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    d.pop("open", None)
    return Session(**d)


def current(path: Path | None = None) -> Session | None:
    """The open session on this node, or None."""
    s = load(path)
    return s if s is not None and s.open else None


def save(s: Session, path: Path | None = None) -> Path:
    path = path or session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(s), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def start(tool, policy: ink_spec.InkPolicy, need_ul: float | None = None, note: str | None = None,
          path: Path | None = None, ledger: Path | None = None, mirror: Path | None = None,
          force: bool = False) -> Session:
    """Open a session for the fitted tool. Refuses while another is open
    unless ``force`` ends it first (the arm cannot carry two charges)."""
    if not policy.dips:
        raise ValueError(f"{getattr(tool, 'tool_id', tool)} has ink.mode {policy.mode}; no session")
    prior = current(path)
    if prior is not None:
        if not force:
            raise ValueError(
                f"session {prior.session_id} ({prior.tool_id}) is still open — "
                "`scripts/ink.py session end` it, or --force")
        end(prior, path=path, ledger=ledger, mirror=mirror, note="ended by force")
    node = socket.gethostname()
    s = Session(session_id=_new_id(node), node=node, tool_id=tool.tool_id, mode=policy.mode,
                capacity_ul=policy.charge_capacity_ul, started_utc=_utc(), need_ul=need_ul, note=note)
    ink_spec.append_event("session.start", policy.mode, path=ledger, mirror=mirror,
                          session_id=s.session_id, tool_id=s.tool_id, need_ul=need_ul, note=note)
    save(s, path)
    return s


def end(s: Session, path: Path | None = None, ledger: Path | None = None,
        mirror: Path | None = None, note: str | None = None) -> Session:
    if not s.open:
        return s
    s.ended_utc = _utc()
    ink_spec.append_event("session.end", s.mode, path=ledger, mirror=mirror,
                          session_id=s.session_id, tool_id=s.tool_id, dips=s.dips, strokes=s.strokes,
                          used_ul=s.used_ul, contact_mm=s.contact_mm, contact_s=s.contact_s,
                          charge_ul=s.charge_ul, runs=list(s.runs), note=note)
    save(s, path)
    return s


def apply_dip(s: Session, policy: ink_spec.InkPolicy, slot: str, ink_id: str | None, uptake_ul: float,
              reason: str, depth_m: float | None = None, why_slot: str | None = None,
              run_id: str | None = None, path: Path | None = None, ledger: Path | None = None,
              mirror: Path | None = None, **extra) -> dict:
    """The arm dipped: credit the charge (a colour change starts from a wiped
    needle), record why, and write the ledger event that owns the truth."""
    before = s.charge_ul
    if reason == "color_change" or (ink_id is not None and s.ink_id is not None and ink_id != s.ink_id):
        before = 0.0
        s.charge_ul = 0.0
    charge = ink_spec.Charge(s.charge_ul, policy.charge_capacity_ul, s.ink_id)
    charge.credit(uptake_ul, ink_id)
    s.charge_ul = charge.ul
    s.ink_id = charge.ink_id
    s.dips += 1
    s.last_dip_utc = _utc()
    if run_id and run_id not in s.runs:
        s.runs.append(run_id)
    ev = ink_spec.append_event(
        "dip", policy.mode, path=ledger, mirror=mirror, slot=slot, ink_id=ink_id,
        uptake_ul=uptake_ul, reason=reason, charge_before=before, charge_after=s.charge_ul,
        depth_m=depth_m, why_slot=why_slot, tool_id=s.tool_id, run_id=run_id,
        session_id=s.session_id, **extra)
    save(s, path)
    return ev


def apply_stroke(s: Session, policy: ink_spec.InkPolicy, contact_mm: float, contact_s: float,
                 run_id: str | None = None, basis: str | None = None, path: Path | None = None,
                 ledger: Path | None = None, mirror: Path | None = None) -> dict | None:
    """A run's contact, measured after the fact (scripts/il_analyze_rollout.py):
    debit the charge and record the raw quantities next to the derived µL,
    so the constants can be re-fit without re-recording. A run already in
    the session is not debited twice."""
    if run_id and run_id in s.runs and s.strokes > 0 and any(
            e.get("run_id") == run_id and e.get("kind") == "stroke"
            for e in ink_spec.read_events(ledger, include_remote=False)):
        return None
    ul = policy.stroke_ul(contact_mm, contact_s)
    charge = ink_spec.Charge(s.charge_ul, policy.charge_capacity_ul, s.ink_id)
    taken = charge.debit(ul)
    s.charge_ul = charge.ul
    s.used_ul += taken
    s.contact_mm += max(0.0, contact_mm)
    s.contact_s += max(0.0, contact_s)
    s.strokes += 1
    if run_id and run_id not in s.runs:
        s.runs.append(run_id)
    ev = ink_spec.append_event(
        "stroke", policy.mode, path=ledger, mirror=mirror, ink_id=s.ink_id,
        contact_mm=contact_mm, contact_s=contact_s, ul=ul, taken_ul=taken,
        charge_after=s.charge_ul, tool_id=s.tool_id, run_id=run_id, session_id=s.session_id,
        contact_basis=basis)
    save(s, path)
    return ev


def rebuild(session_id: str, events: list[dict], path: Path | None = None,
            capacity_ul: float | None = None) -> Session | None:
    """Recover a session's state from its ledger events — the file is a
    cache; this is the proof. The tool's capacity is not in the events (a
    `dip` only says charge_after), so pass it from the datasheet; without
    it the largest charge seen stands in, and a session with no dip yet
    reports 0."""
    evs = [e for e in events if e.get("session_id") == session_id]
    if not evs:
        return None
    start_ev = next((e for e in evs if e.get("kind") == "session.start"), None)
    if start_ev is None:
        return None
    s = Session(session_id=session_id, node=start_ev.get("node", "?"), tool_id=start_ev.get("tool_id", "?"),
                mode=start_ev.get("mode", "rehearsal"), capacity_ul=float(capacity_ul or 0.0),
                started_utc=start_ev.get("utc", ""), need_ul=start_ev.get("need_ul"), note=start_ev.get("note"))
    for e in evs:
        k = e.get("kind")
        if k == "dip":
            s.dips += 1
            s.charge_ul = float(e.get("charge_after", s.charge_ul))
            s.ink_id = e.get("ink_id", s.ink_id)
            s.last_dip_utc = e.get("utc")
            s.capacity_ul = max(s.capacity_ul, s.charge_ul)
        elif k == "stroke":
            s.strokes += 1
            s.used_ul += float(e.get("taken_ul", e.get("ul", 0.0)))
            s.contact_mm += float(e.get("contact_mm", 0.0))
            s.contact_s += float(e.get("contact_s", 0.0))
            s.charge_ul = float(e.get("charge_after", s.charge_ul))
        elif k == "session.end":
            s.ended_utc = e.get("utc")
        rid = e.get("run_id")
        if rid and rid not in s.runs:
            s.runs.append(rid)
    return s


def describe(s: Session | None, policy: ink_spec.InkPolicy | None = None) -> str:
    if s is None:
        return "no open ink session on this node"
    state = "open" if s.open else f"ended {s.ended_utc}"
    lines = [f"session {s.session_id} ({state}) — {s.tool_id} [{s.mode}] since {s.started_utc}",
             f"  charge {s.charge_ul:.2f}/{s.capacity_ul:.2f} uL ({100 * s.charge_frac:.0f}%)"
             f" of {s.ink_id or 'nothing'}; {s.dips} dip(s), {s.strokes} stroke run(s),"
             f" {s.used_ul:.2f} uL spent over {s.contact_mm:.0f} mm / {s.contact_s:.0f} s"]
    if s.need_ul is not None:
        lines.append(f"  planned need {s.need_ul:.1f} uL, {s.remaining_need_ul():.1f} uL remaining")
    if s.runs:
        lines.append(f"  runs: {', '.join(s.runs[-6:])}{' …' if len(s.runs) > 6 else ''}")
    if policy is not None and s.open:
        why = s.needs_dip(policy, s.remaining_need_ul())
        lines.append(f"  next: {'dip (' + why + ')' if why else 'charge will do'}")
    return "\n".join(lines)
