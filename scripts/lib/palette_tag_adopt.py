"""The palette roof tag's seat, adopted from a wrist hand-eye run (`tatbot ros calib apply --run <wristcam run>`).

The roof tag is a sticker placed by hand on the printed palette; the station fix reads the palette's yaw and place from
it through its seat in the palette's design (cad/<palette>/design.json `runtime.installed_tag`, emitted into
urdf/palette.urdf as `palette_tag_joint`). A sticker turned on its seat turns the whole palette about the probe ball by
as much: on the xArm rig the ID-0 sticker stood 1.29 deg turned and 0.9 mm aside of its CAD seat, which moved the tag
3.6 mm about the ball and the caps 1-2 mm (both wrist D405s, five runs, 2026-11-09).

`calib wristcam` solves the tag's pose in the arm's base and, against the station touch's palette, records the seat:
its turn about its normal and its shift on the palette, and the seat they are measured against, palette.urdf's on the
node that ran it (candidate.json `tag_seat`, its `against`). This sets design.json's seat to that seat turned and
shifted by them and rewrites the URDF's joint to match, as cad/<palette>/build.py emits it. The seat adopted is the
one the run measured, whatever seat the design holds now: a run applied twice, or after another, gives the same seat.
A run from before `against` was recorded holds the turn and shift alone: it is the design's seat already when the
design names it, and otherwise its seat cannot be told, so a new run is asked for.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

JOINT = re.compile(r'(<joint name="palette_tag_joint" type="fixed">\s*<parent link="palette_root"/><child '
                   r'link="palette_tag"/>\s*<origin )xyz="[^"]*" rpy="[^"]*"(/>)')


def _root_mm(design: dict) -> np.ndarray:
    """palette_root in the design's frame: on the probe axis, runtime.ball_above_root_mm under the ball."""
    probe = design["probe"]
    return np.array([*probe["xy"], probe["ball_z"] - design["runtime"]["ball_above_root_mm"]], float)


def apply(candidate: dict, design_text: str, urdf_text: str) -> tuple[str, str, list[str]]:
    """(design.json, palette.urdf, what moved) with the candidate's tag seat adopted; ValueError when it holds none or
    the URDF has no tag joint to rewrite."""
    seat = candidate.get("tag_seat")
    if not seat:
        raise ValueError("the run measured no tag seat: a calib wristcam run on the station touch does")
    design = json.loads(design_text)
    tag = design["runtime"]["installed_tag"]
    run = candidate.get("gauge", {}).get("run", "")
    against = seat.get("against")
    if not against:
        if run and f"run {run})" in tag.get("measured", ""):
            return design_text, urdf_text, [f"already adopted: the design's seat is the one run {run} measured"]
        raise ValueError(f"run {run or '?'} does not record the seat it was measured against, so its turn and shift "
                         "cannot be placed on the palette: measure again (tatbot ros calib wristcam) and apply that run")
    shift = np.array([*seat["shift_mm"], 0.0])   # the sticker lies on the palette's flat top: its height stays
    was = _root_mm(design) + np.asarray(against["xyz_m"], float) * 1000.0
    xyz = np.round([was[0] + shift[0], was[1] + shift[1], tag["xyz_mm"][2]], 2)
    yaw = round(float(against["yaw_deg"]) + float(seat["turn_deg"]), 3)
    tag.update(xyz_mm=xyz.tolist(), yaw_deg=yaw)
    tag["measured"] = (f"turned {seat['turn_deg']:+.2f} deg and shifted ({seat['shift_mm'][0]:+.2f}, "
                       f"{seat['shift_mm'][1]:+.2f}) mm on its seat by the wrist hand-eye on the arm's station touch "
                       f"(tatbot ros calib wristcam, run {run})")
    metres = " ".join(f"{v / 1000:.9f}" for v in xyz - _root_mm(design))
    origin = f'xyz="{metres}" rpy="0.000000000 0.000000000 {math.radians(yaw):.9f}"'
    edited, n = JOINT.subn(lambda m: m.group(1) + origin + m.group(2), urdf_text)
    if n != 1:
        raise ValueError("palette.urdf has no palette_tag_joint off palette_root to rewrite")
    edited_design = json.dumps(design, indent=1) + "\n"
    if (edited_design, edited) == (design_text, urdf_text):
        return design_text, urdf_text, [f"already adopted: the design's seat is the one run {run} measured"]
    lines = [f"the roof tag's seat turned {seat['turn_deg']:+.2f} deg to {yaw:+.3f} and moved "
             f"({shift[0]:+.2f}, {shift[1]:+.2f}) mm to ({xyz[0]:.2f}, {xyz[1]:.2f}) mm in the design"]
    return edited_design, edited, lines
