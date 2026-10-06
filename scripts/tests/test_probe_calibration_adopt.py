"""Adopting a probe tip: the arm's tip moves across its axis by the fitted change, its length stays, the touch-off
record names the run, and every other line is untouched. Another tool's candidate is refused."""
from __future__ import annotations

from pathlib import Path

import probe_calibration_adopt as adopt
import tool_spec

WORKSPACE = (Path(__file__).resolve().parents[2] / "config" / "workspace.yaml").read_text()
FITTED = tool_spec.active_tool_id(workspace=tool_spec.parse_simple_yaml(WORKSPACE))


def candidate(**over):
    base = {"arm": "right", "tool": FITTED, "run_id": "20261003T010000Z-node-abcd",
            "utc": "2026-10-03T01:30:00Z", "contacts": 32, "tip0_m": [0.00002, -0.002674, 0.074495],
            "held_out_rms_m": {"S1": 0.0003, "S4 -60/+0": 0.0004}, "tip_moved_m": {"S1": 0.0002},
            "fit": {"tip_m": [0.00042, 0.000826, 0.074495], "rms_m": {"side": 0.0002}}}
    base.update(over)
    return base


def _value(text, arm, key):
    section = text[text.index(f"\n{arm}:"):]
    return float(section.split(f"  {key}:")[1].split()[0])


def test_the_tip_moves_across_its_axis_and_nothing_else_changes():
    edited, why = adopt.apply(candidate(), WORKSPACE)
    assert why == []
    for key, moved in (("pen_tip_offset_x", 0.0004), ("pen_tip_offset_y", 0.0035), ("mechanical_contact_x", 0.0004),
                       ("mechanical_contact_y", 0.0035), ("pen_tip_offset_z", 0.0), ("mechanical_contact_z", 0.0)):
        assert abs(_value(edited, "right", key) - _value(WORKSPACE, "right", key) - moved) < 1e-9, key
    assert "ros-calib/20261003T010000Z-node-abcd" in edited
    left = WORKSPACE.index("\nleft:")
    assert edited[edited.index("\nleft:"):] == WORKSPACE[left:]
    assert edited[:edited.index("\nright:")] == WORKSPACE[:WORKSPACE.index("\nright:")]


def test_another_tools_candidate_and_a_nonfinite_tip_are_refused():
    assert adopt.apply(candidate(tool="picosecond-laser-pen"), WORKSPACE) == (WORKSPACE, [
        f"the candidate measured picosecond-laser-pen, but {FITTED} is fitted"])
    bad = candidate()
    bad["fit"]["tip_m"][0] = float("nan")
    assert adopt.apply(bad, WORKSPACE)[1] == ["the candidate's tip is not finite"]
