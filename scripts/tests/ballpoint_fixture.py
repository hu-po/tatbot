"""The pink arm's workspace with the ballpoint fitted, for the tests about the ballpoint whatever the arm carries.

arm_kinematics.BALLPOINT_TIP_IN_LINK6, the follower the module plans with and the C++ executor compiles in, is
the ballpoint's touch-off carried to link 6. While the ballpoint is fitted config/workspace.yaml holds that
touch-off and is read as it is, so a new one still has to match the constant; with another tool fitted the
right section below (the ballpoint's 2026-09-19 touch-off) stands in for it. Copy a new ballpoint touch-off
here when the constant changes.
"""
from __future__ import annotations

import re
from pathlib import Path

import tool_spec

REPO = Path(__file__).resolve().parents[2]
TOOL = "lutin-ballpoint-dot"
RIGHT = """\
right:
  tip_frame: right/tool_mount
  pen_tip_offset_x: 0.000020
  pen_tip_offset_y: -0.002674
  pen_tip_offset_z: 0.074495
  mechanical_contact_x: 0.000020
  mechanical_contact_y: -0.002674
  mechanical_contact_z: 0.074245
  mechanical_contact_reference: seated_ball
  mechanical_contact_profile: ballpoint-precision
  mechanical_contact_status: fit_passed
  touchoff:
    utc: 2026-09-19T17:48:39Z
    method: hand-guided three-pit palette calibration
    n_plate: 9
    n_pad: 0
    cond: 13.0
    residual_mm: 2.473
    holdout_mm: 2.104
    tip_loo_max_mm: 2.104
    spread_deg: 83.0
  tool_id: lutin-ballpoint-dot
  carriage_m: 0.000000

"""


def workspace_text() -> str:
    text = (REPO / "config/workspace.yaml").read_text()
    if tool_spec.active_tool_id(workspace=tool_spec.parse_simple_yaml(text)) == TOOL:
        return text
    text, count = re.subn(r"^right:\n.*?(?=^\S)", lambda _: RIGHT, text, count=1, flags=re.M | re.S)
    assert count == 1, "config/workspace.yaml has no right section followed by another"
    return text


def workspace(repo=None) -> dict:
    """Parsed; it stands in for tool_spec.read_workspace, whichever checkout asks."""
    return tool_spec.parse_simple_yaml(workspace_text())
