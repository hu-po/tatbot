"""A tool whose tip is a tube's end, on the station probe's ball: where each touch starts, and what a contact says.

A tool the station can probe ends in a halo (its datasheet's `calibration.face_kind: halo`): a ring, the tube's
end, around a hole. The laser pen's nose is a chrome ring 9.0 mm in radius outside, around a 7.7 mm hole with the
lens recessed in it. A ballpoint's moving refill can project from its narrow plastic nozzle (`ballpoint_tip`), so
its side touches meet the rigid nozzle at its side height. Contacts use the wall, never the hole:

- **side**: the tool moves across its own axis until the wall's side meets the ball. The ball's centre is then
  `side_radius` from the axis (the wall's radius + the ball's, less the trigger's bias). Opposed pairs cancel
  the bias; together the sides fix the axis.
- **search**: a side touch low enough that the stylus stands in the tool's way whatever the ball's height,
  from starts far enough out that no ball within the prior's error lies under them. It meets the ball, the
  shaft or the cone with the wall or whatever stands above it, so it centres the stylus and goes to no fit.
- **edge pass**: a side pass at a given end height either meets the stylus (the ball's top is above the end)
  or passes over it. A few passes bracket the ball's top without ever lowering the tool onto anything.

The hazard is a hole wider than the ball. A side pass whose end plane stands just under the ball's top meets it
on the ring's edge, pushing the stylus down its axis where the trigger force is highest, and may ride over it into
the hole: edge passes stop with the ball still under the ring, and a side touch is only planned where the ball's
equator is on the wall. A cartridge's bore is narrower than the ball, which cannot enter it (`Halo.hole_takes_ball`).

The rest of the tool, its envelope (the datasheet's profile, never inside the wall), is what must stay clear of the
station's other parts (`meets`). Poses are 4x4 <arm>/base_link <- tcp (the tcp is the halo end plane's centre or
the compressed working ballpoint; +z along the tool). Pure numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from tatbot_description.transforms import rpy_matrix
from tatbot_motion.timelaw import axis_rotation

BALL_RADIUS_M = 0.001   # the probe's 2.000 mm ball (urdf/palette.urdf)
ENVELOPE_CHECKED_M = 0.120   # the tool's envelope checked against the station up to here from the end: above it
                             # the tool stands higher than every part, the roof tag's 60 mm over the ball included


@dataclass(frozen=True)
class Halo:
    """Rigid nozzle sides, with axial reference on its ring or the compressed working ball. Metres."""
    wall_radius_m: float    # calibration.wall_radius_m: the ring's outside, where side touches meet the ball
    hole_radius_m: float    # calibration.face_radius_m: the hole (the laser's lens face, a cartridge's bore)
    rim_radius_m: float     # calibration.rim_touch_radius_m: an edge pass over a bore stops with the ball under it
    side_height_m: float    # calibration.side_touch_height_m: a side touch meets the ball this far up the wall
    envelope: tuple = ()    # ((h, r), ...): the tool's outside radius r, h up the axis from the end plane
    axial_tip: bool = False  # ballpoint_tip: the working ball moves; the side touches meet the rigid nozzle

    @classmethod
    def from_datasheet(cls, calibration: dict, profile=()) -> Halo:
        """The halo a datasheet's `calibration:` block declares, with its body `profile` ((z, r), ..., the tip
        last) as the envelope; ValueError when it declares none, or a dimension is missing or out of order (the
        hole inside the ring's middle, inside its outside)."""
        axial_tip = calibration.get("face_kind") == "ballpoint_tip"
        if calibration.get("face_kind") not in ("halo", "ballpoint_tip"):
            raise ValueError(f"calibration.face_kind is {calibration.get('face_kind')!r}, not halo or ballpoint_tip")
        fields = ("wall_radius_m", "face_radius_m", "rim_touch_radius_m", "side_touch_height_m")
        missing = [name for name in fields if calibration.get(name) is None]
        if missing:
            raise ValueError("the halo lacks " + ", ".join(f"calibration.{name}" for name in missing))
        end = float(profile[-1][0]) if profile else 0.0
        envelope = tuple(sorted((end - float(z), float(r)) for z, r in profile))
        halo = cls(*(float(calibration[name]) for name in fields), envelope, axial_tip)
        ring = 0.0 <= halo.hole_radius_m < halo.rim_radius_m < halo.wall_radius_m
        if not (ring and halo.side_height_m > 0.0):
            raise ValueError(f"the halo's dimensions are out of order: hole {halo.hole_radius_m}, rim "
                             f"{halo.rim_radius_m}, wall {halo.wall_radius_m}, side height {halo.side_height_m} m")
        return halo

    @property
    def hole_takes_ball(self) -> bool:
        """A hole wider than the ball lets it rise inside, clear of the ring: the laser's does, a bore does not."""
        return self.hole_radius_m > BALL_RADIUS_M

    def radius_at(self, h: float) -> float:
        """The tool's outside radius h up the axis from the end plane: its envelope, never inside the wall."""
        if not self.envelope:
            return self.wall_radius_m
        heights, radii = zip(*self.envelope, strict=True)
        return max(self.wall_radius_m, float(np.interp(h, heights, radii)))

    def widest_m(self, up_to_m: float) -> float:
        """The tool's widest outside radius from the end plane up to up_to_m."""
        return max(self.radius_at(h) for h in np.linspace(0.0, up_to_m, int(up_to_m / 0.0005) + 2))


@dataclass(frozen=True)
class TouchPlan:
    kind: str                 # side | search | edge | back
    label: str                # e.g. "+d", "rim 120"
    start: np.ndarray         # tcp start pose (4x4, base)
    direction: np.ndarray     # unit, base
    prior_m: float            # the expected contact this far along the direction


def tool_pose(position, yaw_rad: float, tilt_rad: float = 0.0, tilt_axis_rad: float = 0.0) -> np.ndarray:
    """A tcp pose pointing down (+z toward the table), turned yaw_rad about the vertical and then tilted tilt_rad
    about the horizontal axis at tilt_axis_rad."""
    out = rpy_matrix(position, [math.pi, 0., yaw_rad])
    tilt = axis_rotation([math.cos(tilt_axis_rad), math.sin(tilt_axis_rad), 0.], tilt_rad)
    out[:3, :3] = tilt @ out[:3, :3]
    return out


def _across(axis: np.ndarray, heading: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit directions across the tool axis: `heading` made perpendicular to it, and their cross."""
    d = heading - (heading @ axis) * axis
    d /= np.linalg.norm(d)
    return d, np.cross(axis, d)


def side_touches(ball: np.ndarray, rotation: np.ndarray, heading: np.ndarray, halo: Halo, side_radius_m: float,
                 standoff_m: float = 0.010, height_m: float | None = None, kind: str = "side") -> list[TouchPlan]:
    """Four side touches (+-d, +-p across the axis) with the ball `height_m` up the wall from its end: each starts
    standoff_m short of the expected contact. A "search" touch finds the stylus and goes to no fit."""
    axis = rotation[:, 2]
    h = halo.side_height_m if height_m is None else height_m
    d, p = _across(axis, np.asarray(heading, float))
    plans = []
    for label, u in (("+d", d), ("-d", -d), ("+p", p), ("-p", -p)):
        end = ball + h * axis - (side_radius_m + standoff_m) * u   # the end plane's centre at the start
        start = np.eye(4)
        start[:3, :3], start[:3, 3] = rotation, end
        plans.append(TouchPlan(kind, label, start, u, standoff_m))
    return plans


def edge_pass(ball_xy_side: TouchPlan, end_height_m: float, prior_m: float | None = None) -> TouchPlan:
    """A side pass with the end plane at a given base height: it meets the stylus when the ball's top is above
    the end, and passes over it otherwise. A shorter `prior_m` stops it where a ball whose top the ring's edge
    rides over (a contact too shallow to trip) is still under the ring, short of the hole."""
    start = ball_xy_side.start.copy()
    start[2, 3] = end_height_m
    return TouchPlan("edge", f"edge {end_height_m * 1000:.1f}", start, ball_xy_side.direction,
                     ball_xy_side.prior_m if prior_m is None else prior_m)


def route(here: np.ndarray, plan: TouchPlan, clearance_m: float, extra_m: float = 0.0025) -> list[np.ndarray]:
    """The end-plane centre's way for a goal: up to the clearance plane (motion.yaml probe.clearance_m) over the
    start, across, down to the start, then the touch's travel to its cap (the executor's station_via and
    plan_touch)."""
    start = plan.start[:3, 3]
    top = max(here[2], start[2] + clearance_m)
    return [here, np.array([here[0], here[1], top]), np.array([start[0], start[1], top]), start,
            start + (plan.prior_m + extra_m) * plan.direction]


def meets(points: list[np.ndarray], axis: np.ndarray, zones, halo: Halo, floor_m: float = 0.0,
          margin_m: float = 0.003):
    """The first station part (station.parts) the tool would meet along the polyline of end-plane centres, or
    None. The tool is its envelope from the end up ENVELOPE_CHECKED_M along -axis, never narrower than floor_m (a
    wall the side touches measured wider). A post is met by any of the tool over it lower than its top."""
    heights = np.arange(0.0, ENVELOPE_CHECKED_M + 1e-9, 0.0025)
    radii = np.array([max(halo.radius_at(h), floor_m) for h in heights])
    for a, b in zip(points[:-1], points[1:], strict=True):
        ends = a + np.linspace(0.0, 1.0, max(1, int(np.linalg.norm(b - a) / 0.001)) + 1)[:, None] * (b - a)
        tool = ends[:, None, :] - heights[None, :, None] * np.asarray(axis, float)   # (steps, heights, xyz)
        for name, centre, r, post in zones:
            if post:
                near = np.hypot(*(tool[..., :2] - centre[:2]).transpose(2, 0, 1)) < r + radii + margin_m
                if np.any(near & (tool[..., 2] < centre[2] + margin_m)):
                    return name
            elif np.any(np.linalg.norm(tool - centre, axis=2) < r + radii + margin_m):
                return name
    return None
