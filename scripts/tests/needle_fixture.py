"""The WXAI follower's workspace with the 3RL fitted and a complete pivot receipt, for tests that need a qualified
contact tool whatever config/workspace.yaml holds today.

The live right section records the 3RL's probe-station side pairs, which leave cond, spread and the pad count
unrecorded; a pose export refuses that as incomplete. The right section below stands in for it: the 3RL's
2026-10-22 tip with a synthetic receipt that qualifies (tool_spec.contact_pose_qualification_error).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL = "lutin-3rl-bugpin"
RIGHT = """\
right:
  tip_frame: right/tool_mount
  pen_tip_offset_x: 0.000083
  pen_tip_offset_y: 0.002685
  pen_tip_offset_z: 0.062179
  mechanical_contact_x: 0.000083
  mechanical_contact_y: 0.002685
  mechanical_contact_z: 0.061929
  mechanical_contact_reference: tube_end
  mechanical_contact_profile: needle-tube
  mechanical_contact_status: fit_passed
  touchoff:
    utc: 2026-10-22T17:11:10Z
    method: synthetic receipt for tests (the 3RL's tip, its pivot diagnostics filled in)
    n_plate: 17
    n_pad: 0
    cond: 13.0
    residual_mm: 0.146
    holdout_mm: 0.325
    tip_loo_max_mm: 0.405
    spread_deg: 83.0
  tool_id: lutin-3rl-bugpin
  carriage_m: 0.000000

"""


def workspace_text() -> str:
    """config/workspace.yaml with its right section replaced by RIGHT."""
    text = (REPO / "config/workspace.yaml").read_text()
    text, count = re.subn(r"^right:\n.*?(?=^\S)", lambda _: RIGHT, text, count=1, flags=re.M | re.S)
    assert count == 1, "config/workspace.yaml has no right section followed by another"
    return text
