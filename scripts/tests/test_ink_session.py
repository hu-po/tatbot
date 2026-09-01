"""The ink session: one charge across many runs (scripts/lib/ink_session.py).

    uvx --with pytest pytest -q scripts/tests/test_ink_session.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import ink_session  # noqa: E402
import ink_spec  # noqa: E402
import tool_spec  # noqa: E402


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TATBOT_INK_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("TATBOT_INK_SESSION", str(tmp_path / "session.json"))
    return tmp_path


def _tool(tool_id="lutin-ballpoint-dot"):
    t = tool_spec.load_tool(tool_id, REPO)
    return t, ink_spec.policy_for(t)


def test_a_session_carries_the_charge_across_runs(paths):
    tool, pol = _tool()
    assert ink_session.current() is None
    s = ink_session.start(tool, pol, need_ul=1.0)
    assert s.open and s.needs_dip(pol, 1.0) == "session_start"
    ink_session.apply_dip(s, pol, "inkcap_right_medium_0", None, pol.uptake_ul, "session_start",
                          run_id="run-a")
    assert s.charge_ul == pytest.approx(pol.uptake_ul) and s.dips == 1
    assert s.needs_dip(pol, 0.5) is None, "a charged needle covers a half-microlitre run"
    ev = ink_session.apply_stroke(s, pol, 100.0, 5.0, run_id="run-a", basis="touchoff_tip")
    assert ev["ul"] == pytest.approx(pol.stroke_ul(100.0, 5.0))
    assert s.charge_ul == pytest.approx(pol.uptake_ul - ev["ul"]) and s.strokes == 1
    # the file is a cache of the ledger: reload and rebuild agree
    again = ink_session.current()
    assert again.charge_ul == pytest.approx(s.charge_ul) and again.runs == ["run-a"]
    rebuilt = ink_session.rebuild(s.session_id, ink_spec.read_events())
    assert rebuilt.dips == 1 and rebuilt.strokes == 1
    assert rebuilt.charge_ul == pytest.approx(s.charge_ul) and rebuilt.runs == ["run-a"]
    # the same run is not debited twice
    assert ink_session.apply_stroke(s, pol, 100.0, 5.0, run_id="run-a") is None
    ink_session.end(s)
    assert ink_session.current() is None
    kinds = [e["kind"] for e in ink_spec.read_events()]
    assert kinds == ["session.start", "dip", "stroke", "session.end"]


def test_rebuild_takes_the_capacity_from_the_datasheet(paths):
    tool, pol = _tool()
    s = ink_session.start(tool, pol)
    ink_session.end(s)
    bare = ink_session.rebuild(s.session_id, ink_spec.read_events())
    assert bare.capacity_ul == 0.0, "no dip on record, nothing in the events says the capacity"
    full = ink_session.rebuild(s.session_id, ink_spec.read_events(), capacity_ul=pol.charge_capacity_ul)
    assert full.capacity_ul == pol.charge_capacity_ul and not full.open


def test_low_charge_and_colour_change(paths):
    tool, pol = _tool()
    s = ink_session.start(tool, pol)
    ink_session.apply_dip(s, pol, "inkcap_right_medium_0", "nighthawk_black", pol.uptake_ul, "session_start")
    ink_session.apply_stroke(s, pol, 300.0, 20.0)
    assert s.needs_dip(pol, 1.0) == "low_charge"
    ink_session.apply_dip(s, pol, "inkcap_right_medium_1", "true_blue", pol.uptake_ul, "color_change")
    assert s.ink_id == "true_blue" and s.charge_ul == pytest.approx(pol.uptake_ul)
    evs = [e for e in ink_spec.read_events() if e["kind"] == "dip"]
    assert evs[-1]["charge_before"] == 0.0, "a colour change starts from a wiped needle"


def test_one_open_session_per_node(paths):
    tool, pol = _tool()
    s = ink_session.start(tool, pol)
    with pytest.raises(ValueError, match="still open"):
        ink_session.start(tool, pol)
    s2 = ink_session.start(tool, pol, force=True)
    assert s2.session_id != s.session_id and ink_session.current().session_id == s2.session_id
    laser, none = _tool("picosecond-laser-pen")
    with pytest.raises(ValueError, match="ink.mode none"):
        ink_session.start(laser, none, force=True)


def test_events_mirror_into_a_run_dir(paths):
    tool, pol = _tool()
    mirror = paths / "run-x" / "ink.jsonl"
    s = ink_session.start(tool, pol, mirror=mirror)
    ink_session.apply_dip(s, pol, "inkcap_right_medium_0", None, 1.0, "session_start",
                          run_id="run-x", mirror=mirror)
    mirrored = ink_spec.read_events(mirror, include_remote=False)
    assert [e["kind"] for e in mirrored] == ["session.start", "dip"]
    # the mirror and the ledger hold the SAME events (same ids), read once
    ids = {e["id"] for e in ink_spec.read_events()}
    assert {e["id"] for e in mirrored} <= ids
