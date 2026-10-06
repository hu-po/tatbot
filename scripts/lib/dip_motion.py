"""The shape of one cap visit. The simulator is its caller: the arm's dip,
``scripts/il_dip.py``, was deleted with ``tatbot dip`` on 2026-09-26, and the
ROS stack's ballpoint does not dip.

A dip is the one motion that leaves the sheet, and it was the one the two
stacks built separately. ``scripts/il_dip.py`` placed the hover and the plunge
along the palette's own entry axis; ``tatbot_sim/dipping.py`` assumed world Z
and left the tool's orientation to IK. The two agreed only while the rack was
level, and nothing ever compared them: the simulator's solved reference sat 23
degrees off the axis it commanded, at every cap, and the charge was credited
anyway (measured 2026-09-09).

Frames are the caller's. Pass a rim and an axis expressed in one frame and
every point comes back in that frame (world, for the simulator; ``il_dip``
passed its arm base). The axis points INTO the cap, so the tool travels along
``+axis`` to enter and ``-axis`` to leave, and a tool held for the dip points
along ``+axis`` too.

Where the rack *is* is deliberately not here. The simulator reads a synthetic
placement and may randomize it; a hardware caller reads a measured SE(3) and
refuses without one. ``ink_spec.palette_rim_layout(measured=...)`` is that
seam, and it stays the only one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class DipPoses:
    """The ordered points of one cap visit, in the caller's frame.

    ``transit`` -> ``above`` -> ``bottom``, and the reverse to leave. Only
    ``above`` and ``bottom`` are geometry of the cap itself; ``transit`` is the
    height the tool crosses the bench at, which each stack routes its own way.
    """

    rim: np.ndarray
    above: np.ndarray
    bottom: np.ndarray
    transit: np.ndarray
    axis: np.ndarray

    @property
    def tool_axis(self) -> np.ndarray:
        """Where the tool points to enter the cap — along the entry axis."""
        return self.axis

    @property
    def outward_normal(self) -> np.ndarray:
        """The cap mouth's outward normal, the sign a surface normal carries."""
        return -self.axis


def dip_poses(rim, axis, *, hover_m: float, plunge_m: float,
              lift_m: float = 0.0, lift_axis=WORLD_UP) -> DipPoses:
    """Hover above a cap, plunge into it, along ``axis``.

    ``hover_m`` is clearance above the rim on the approach and the retract;
    ``plunge_m`` is depth below the rim, which the caller takes from
    ``ink_spec.dip_plunge_m`` so it tracks the cap's fill. ``lift_m`` raises the
    transit point along ``lift_axis`` for a stack that crosses the bench at a
    fixed height; at 0 the transit point is the hover point.
    """
    rim = np.asarray(rim, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    if rim.shape != (3,) or axis.shape != (3,):
        raise ValueError("rim and axis are single 3-vectors")
    if not (np.isfinite(rim).all() and np.isfinite(axis).all()):
        raise ValueError("non-finite dip geometry")
    norm = float(np.linalg.norm(axis))
    if not (math.isfinite(norm) and norm > 1e-9):
        raise ValueError("dip entry axis has no direction")
    axis = axis / norm
    for name, value in (("hover_m", hover_m), ("plunge_m", plunge_m), ("lift_m", lift_m)):
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be a finite, non-negative distance")
    above = rim - axis * hover_m
    lift = np.asarray(lift_axis, dtype=np.float64)
    lift_norm = float(np.linalg.norm(lift))
    if lift_m and not (math.isfinite(lift_norm) and lift_norm > 1e-9):
        raise ValueError("lift axis has no direction")
    transit = above + (lift / lift_norm) * lift_m if lift_m else above
    return DipPoses(rim=rim, above=above, bottom=rim + axis * plunge_m,
                    transit=transit, axis=axis)


def palette_entry_axis(base_from_root) -> np.ndarray:
    """The cap entry axis in the arm base frame, from the rig transform.

    Caps open along the palette root's +Z, so the tool enters along -Z carried
    into the base frame. A level rack makes this world -Z, which is what both
    stacks used to hard-code; a tilted one is exactly the case that silently
    diverged.
    """
    rotation = np.asarray(base_from_root, dtype=np.float64)[:3, :3]
    axis = rotation @ np.array([0.0, 0.0, -1.0])
    norm = float(np.linalg.norm(axis))
    if not (math.isfinite(norm) and norm > 1e-9):
        raise ValueError("palette transform has no usable entry axis")
    return axis / norm
