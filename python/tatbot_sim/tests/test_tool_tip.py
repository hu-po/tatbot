"""The sim welds the measured tip only for the tool it was measured with.

2026-08-31: the ballpoint's touch-off (68.5 mm) was silently welded onto
every simulated tool — a laser losing 61 mm of protrusion — because the sim
read workspace.yaml's tip with no tool-identity check. The real rig refuses
that mix via require_stated_tool; the sim must make the same call.
"""

from __future__ import annotations

from tatbot_sim import urdf
from tatbot_sim.tools import active_tool, registry


def _ws(reg, tool_id):
    return {"right": {
        "tool_id": tool_id,
        "tip_frame": reg.tip_frame("right"),
        "pen_tip_offset_x": 0.001,
        "pen_tip_offset_y": 0.002,
        "pen_tip_offset_z": 0.070,
    }}


def test_the_measured_tip_belongs_to_the_calibrated_tool_only():
    reg, spec = registry(), active_tool()
    mine = urdf._tool_tip_m(reg, _ws(reg, spec.tool_id), spec)
    assert mine == (0.001, 0.002, 0.070), "own touch-off must be used"
    other = urdf._tool_tip_m(reg, _ws(reg, "someone-elses-tool"), spec)
    assert other == spec.touchoff_nominal_m, (
        "another tool's touch-off must not leak into this tool's geometry")
    assert urdf._tool_tip_m(reg, {}, spec) == spec.touchoff_nominal_m, (
        "no touch-off at all falls back to the datasheet nominal")
