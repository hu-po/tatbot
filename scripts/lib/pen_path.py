"""Offline design/surface sampling for simulator references and geometry tests.

Contract: docs/surface-formats.md. This module turns design strokes on an
anchored `HeightFieldSurface` (scripts/lib/surface_model.py, imported lazily)
and supplied contact/hold poses into Cartesian samples. Simulator consumers
serialize those samples for the retained offline C++ parser/planner
(`path_plan_check`). This module opens no driver and grants no motion authority.
Physical drawing uses the ROS 2 stack; its time laws live in
`ros/tatbot_motion/tatbot_motion/timelaw.py`.

Geometry conventions:

- The surface lives in `root`; samples are written in the selected arm model's
  base frame. Root and the configured arm base differ by one translation,
  so normals and rotations are the same in both.
- Orientation rule (decision 3): R_i = align(n_c -> n_i) @ R_c. The operator's
  approach angle at contact is preserved relative to the surface; spin changes
  minimally.
- Time law: quintic ease / cruise / ease over arc length,
  sampled at t = k * period for k = 1..ceil(duration / period).
- Every refusal is a `DrawRefusal(code, detail)`; nothing is clamped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import arm_kinematics as dk
import candidate_plan
import numpy as np
import stroke_material as material
import stroke_operation as operation
from motion_constants import SHA, C

SAMPLES_SCHEMA = "tatbot.draw-samples/1"
START_TOLERANCE_M = 0.001
TIP_SPEED_CAP_M_S = 0.020          # pen-down and the approach
TIP_SPEED_CAP_PEN_UP_M_S = 0.120   # pen-up travel to and from the rack (2026-09-03)
NORMAL_SWING_CAP_DEG = 60.0
GIRTH_FRACTION = 0.8          # of pi * radius: the injective half of a cylinder chart, with margin
# Pen-up travel between the hold and the surface: hold -> the approach standoff
# above the anchor at APPROACH_SPEED_M_S (rotation slerp bounded by
# APPROACH_OMEGA_MAX), down at DESCENT_SPEED_M_S to FINAL_DESCENT_M above the
# surface, the last millimetres at FINAL_DESCENT_SPEED_M_S, then the settle. The
# first draw (2026-09-01) approached from the 80 mm orbit standoff at 10 mm/s and
# descended at 5 mm/s: 29 s the operator called "slow and from too far".
# The standoff is APPROACH_STANDOFF_M after a camera orbit (the hold is at the
# orbit's last viewpoint); a no-scan session holds NO_SCAN_LIFT_M straight
# above the touch and that hold IS the approach start (`approach_standoff_m`),
# so the 30 mm lift-and-return the operator called wasted time (2026-09-04,
# runs 32d4/ad18) is gone: lift 10 mm, touch off from there, approach from
# 10 mm. `path.approach_mm` overrides either.
APPROACH_STANDOFF_M = 0.030
NO_SCAN_LIFT_M = C.orbit.no_scan_lift_mm * 1e-3
# A native contact search's measured surface correction: how far the touch may
# move the strokes from the camera chart along the normal (the cover below it
# or the clearance above it, never past twice the widest cover).
CONTACT_OFFSET_MAX_M = 2.0 * C.touchoff.cover_max_m


def pen_offsets(config: dict) -> tuple[float, float]:
    """(pressure lift, native contact offset) in metres, each inside its band.

    The lift is the operator's pressure knob, +/-3 mm. The contact offset is
    the session's own touch: where the native search met the surface relative
    to the camera chart, carried rigidly along the local normal so a 20 mm
    camera height bias (2026-09-17) reaches the strokes without eating the
    pressure band.
    """
    path = config.get("path", {})
    lift = float(path.get("lift_mm", 0.0)) * 1e-3
    if not (math.isfinite(lift) and abs(lift) <= 0.003):
        raise DrawRefusal("design", f"path.lift_mm must be within -3..3, got {lift * 1e3}")
    offset = float(path.get("contact_offset_mm", 0.0)) * 1e-3
    if not (math.isfinite(offset) and abs(offset) <= CONTACT_OFFSET_MAX_M):
        raise DrawRefusal("design", f"path.contact_offset_mm must be within +/-{CONTACT_OFFSET_MAX_M * 1e3:.0f}, got {offset * 1e3}")
    return lift, offset


APPROACH_SPEED_M_S = 0.020
APPROACH_OMEGA_MAX_RAD_S = math.radians(8.0)
DESCENT_SPEED_M_S = 0.010
FINAL_DESCENT_M = 0.005
FINAL_DESCENT_SPEED_M_S = 0.003
LIFT_SPEED_M_S = 0.010
DESCENT_SETTLE_S = 1.0        # pen-up hold at the contact before the pen-down handover
QUINTIC_PEAK = 1.875          # max of d/du (10u^3 - 15u^4 + 6u^5)
MIN_SEGMENT_S = 0.25
FEEDFORWARD_WINDOW_TICKS = 21  # symmetric box over the central-difference velocity (52 ms at 400 Hz)
# Corner-aware retiming of session strokes and pen-up transitions (2026-09-14).
# The design polylines carry hairpins of ~0.2 mm radius and hundreds of sharp
# vertices; run through them at the cruise speed, the tip reference turns by
# 2-90 deg in one tick and the offline joint plan follows it (0.2-100 rad/s^2
# planned joint acceleration on the guided swallow, against ~0.03 rad/s^2 on
# the quintic approach). The retimer keeps every sample on the same polyline
# and only changes s(t): the tip speed at a vertex is capped so the velocity
# change the corner asks for, spread over the feedforward window the planner
# smooths it with, stays under RETIME_CORNER_ACCEL, and the speed ramps into
# and out of that cap at the same acceleration. Pen-up legs become one
# continuous accelerate/cruise/decelerate profile over the unchanged
# hold -> standoff -> near -> contact polyline instead of three quintic
# segments each starting and ending at rest (a 2.6 mm stroke-to-stroke travel
# used to run as a 0.25 s quintic: 240 mm/s^2). `path.timing: "legacy"`
# restores the previous law exactly; the values are planner budgets, not
# controller caps.
RETIME_CORNER_ACCEL_M_S2 = 0.010        # pen-down: velocity change budget across a corner / ramp acceleration
RETIME_PEN_UP_ACCEL_M_S2 = 0.010        # pen-up legs (the legacy quintic legs peaked at 3-13 mm/s^2)
RETIME_TOUCHDOWN_ACCEL_M_S2 = 0.003     # the last FINAL_DESCENT_M into / out of the surface: the legacy quintic's peak
RETIME_DEADBAND_BLEND_DEG = 3.0         # C1 blend width at the orientation deadband edge (0 keeps the kink)
# The measured height field is Catmull-Rom interpolated: its normal is continuous
# but turns at a rate that jumps at every 1 mm cell edge (up to 2.3 deg/mm
# against the 1.3 deg/mm of the fitted cylinder on the guided swallow), and the
# offline planner turns each jump into a one-tick joint velocity step. The orientation
# transport therefore follows the normal averaged over +-this arc length along
# the source stroke (triangular window); positions still use the exact model
# and the lean check still measures against the exact normal (the averaged one
# differs from it by ~0.2 deg on that surface, reported per chunk).
RETIME_ORIENTATION_SMOOTHING_M = 1e-3
RETIME_ORIENTATION_GRID_M = 50e-6
RETIME_RAMP_S = 0.2                     # S-curve time constant: a ramp's jerk budget is its acceleration / this
RETIME_ANGULAR_ACCEL_RAD_S2 = 0.1       # rotary budget: change of the tool's angular velocity (see budgeted_speed_profile)
RETIME_KNOT_STEP_M = 5e-6               # arc-length knot spacing the speed ramps are integrated on
RETIME_PEN_UP_KNOT_STEP_M = 10e-6
COLUMNS = ("t_s", "px", "py", "pz", "vx", "vy", "vz",
           "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22", "pen", "capture", "dip")
LEGACY_COLUMNS = COLUMNS[:-1]   # files written before the dip column (2026-09-03) still read


class DrawRefusal(RuntimeError):  # noqa: N818 - the contract's name
    """A preflight refusal: `code` names what the preflight refused."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass
class Samples:
    """One row per control tick, tip in base. `R` is the link-6 target rotation."""

    period_s: float
    t: np.ndarray        # (N,)
    p: np.ndarray        # (N, 3)
    v: np.ndarray        # (N, 3)
    R: np.ndarray        # (N, 3, 3)  # noqa: N815 - matches the contract's `R`
    pen: np.ndarray      # (N,) int
    capture: np.ndarray  # (N,) int
    dip: np.ndarray | None = None  # (N,) int: k > 0 on the row where dip k bottoms out (2026-09-03)

    @property
    def dips(self) -> np.ndarray:
        return np.zeros(self.n, dtype=np.int64) if self.dip is None else np.asarray(self.dip, np.int64)

    @property
    def n(self) -> int:
        return int(len(self.t))

    @property
    def duration_s(self) -> float:
        return float(self.t[-1]) if self.n else 0.0


# --- design geometry ---------------------------------------------------------


def polyline_length(poly: np.ndarray) -> float:
    poly = np.asarray(poly, float)
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()) if len(poly) > 1 else 0.0


# --- time law ----------------------------------------------------------------

def _quintic(u):
    u = np.asarray(u, float)
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    return 10.0 * u3 - 15.0 * u4 + 6.0 * u5


def time_law(path_length_m: float, duration_s: float, ease_s: float, period_s: float):
    """C++ ease-in / cruise / ease-out over arc length: (t, s, sdot), t = k*period for k = 1..ticks."""
    if not (math.isfinite(duration_s) and duration_s > 0.0 and math.isfinite(ease_s) and ease_s > 0.0
            and ease_s * 2.0 < duration_s and math.isfinite(period_s) and period_s > 0.0):
        raise ValueError("time law needs 0 < 2*ease < duration and a positive period")
    if not (math.isfinite(path_length_m) and path_length_m > 0.0):
        raise ValueError("path length must be positive")
    ticks = int(math.ceil(duration_s / period_s))
    if ticks == 0 or ticks > dk.PLAN_MAX_TICKS:
        raise ValueError(f"{ticks} ticks is outside the guarded range")
    cruise = path_length_m / (duration_s - ease_s)
    t = np.minimum(duration_s, np.arange(1, ticks + 1, dtype=float) * period_s)
    s = np.empty(ticks)
    sdot = np.empty(ticks)

    head = t < ease_s
    u = t[head] / ease_s
    s[head] = cruise * ease_s * _distance_blend(u)
    sdot[head] = cruise * _quintic(u)

    body = ~head & (t <= duration_s - ease_s)
    s[body] = 0.5 * cruise * ease_s + cruise * (t[body] - ease_s)
    sdot[body] = cruise

    tail = t > duration_s - ease_s
    u = (duration_s - t[tail]) / ease_s
    s[tail] = path_length_m - cruise * ease_s * _distance_blend(u)
    sdot[tail] = cruise * _quintic(u)

    s = np.clip(s, 0.0, path_length_m)
    return t, s, sdot


def _distance_blend(u):
    u = np.asarray(u, float)
    u4 = u ** 4
    return 2.5 * u4 - 3.0 * u4 * u + u4 * u * u


resample_polyline_by_arclength = material.resample_polyline_by_arclength


# --- surface lift ------------------------------------------------------------

def lift_to_surface(surface, uv: np.ndarray):
    """(points_root (N,3), normals (N,3)) at chart coordinates `uv` via surface.frame."""
    point, _, _, normal = surface.frame(np.asarray(uv, float))
    return np.asarray(point, float), np.asarray(normal, float)


def transported_rotations(normals: np.ndarray, n_c, r_c: np.ndarray, deadband_rad: float = 0.0,
                          blend_rad: float = 0.0) -> np.ndarray:
    """R_i = align(n_c -> n_i) @ R_c for every row of `normals` (decision 3).

    ``deadband_rad`` > 0 relaxes the rule: the rotation carrying n_c to n_i is
    shortened by that angle (about the same axis) and vanishes when the normal
    swing is inside it, so the tool leans up to the deadband off the local
    normal before the wrist follows. On a 38 mm bottle the +-19 deg swing of a
    15 mm spiral otherwise rolls joint 3 through ~100 deg per turn from a touch
    near the wrist singularity (runs 9078 and f3b0, 2026-09-02) and no speed
    passes the joint-velocity cap; a 12 deg deadband passes at 3.5 mm/s with
    joint velocity 0.19 of 0.25 rad/s. The preflight's lean budget still applies.

    ``blend_rad`` > 0 rounds the deadband edge: max(swing - deadband, 0) has a
    kink where the wrist starts following, and a stroke crossing it on a
    cylinder at 3 mm/s steps its angular rate from 0 to v/R in one tick. The
    quadratic blend over swing in [deadband - blend, deadband + blend] is C1,
    never leans the tool farther than the deadband, and starts the wrist at
    most ``blend`` earlier (its lean at the edge is deadband - blend/4).
    """
    normals = np.asarray(normals, float)
    n_c = np.asarray(n_c, float)
    n_c = n_c / np.linalg.norm(n_c)
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    v = np.cross(n_c[None, :], n)
    c = n @ n_c
    if deadband_rad > 0.0:
        angle = np.arccos(np.clip(c, -1.0, 1.0))
        excess = angle - float(deadband_rad)
        reduced = np.maximum(excess, 0.0)
        if blend_rad > 0.0:
            w = float(blend_rad)
            inside = np.abs(excess) < w
            reduced = np.where(inside, (excess + w) ** 2 / (4.0 * w), reduced)
        norm = np.linalg.norm(v, axis=1)
        axis = np.where(norm[:, None] > 1e-12, v / np.maximum(norm, 1e-300)[:, None], 0.0)
        k = np.zeros((len(n), 3, 3))
        k[:, 0, 1] = -axis[:, 2]
        k[:, 0, 2] = axis[:, 1]
        k[:, 1, 0] = axis[:, 2]
        k[:, 1, 2] = -axis[:, 0]
        k[:, 2, 0] = -axis[:, 1]
        k[:, 2, 1] = axis[:, 0]
        s_r = np.sin(reduced)[:, None, None]
        c_r = (1.0 - np.cos(reduced))[:, None, None]
        rot = np.eye(3)[None] + s_r * k + c_r * (k @ k)
        return rot @ np.asarray(r_c, float)[None]
    k = np.zeros((len(n), 3, 3))
    k[:, 0, 1] = -v[:, 2]
    k[:, 0, 2] = v[:, 1]
    k[:, 1, 0] = v[:, 2]
    k[:, 1, 2] = -v[:, 0]
    k[:, 2, 0] = -v[:, 1]
    k[:, 2, 1] = v[:, 0]
    safe = c > -1.0 + 1e-9
    factor = np.where(safe, 1.0 / (1.0 + np.where(safe, c, 0.0)), 0.0)
    align = np.eye(3)[None] + k + (k @ k) * factor[:, None, None]
    for i in np.flatnonzero(~safe):
        align[i] = dk.align_rotation(n_c, n[i])
    return align @ np.asarray(r_c, float)[None]


# --- segments ----------------------------------------------------------------

def _segment_ticks(distance: float, speed_max: float, period_s: float) -> int:
    duration = max(MIN_SEGMENT_S, QUINTIC_PEAK * distance / speed_max)
    return max(1, int(math.ceil(duration / period_s)))


def _rotation_slerp(r0: np.ndarray, r1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Rotations from r0 to r1 along the geodesic at fractions `blend`."""
    axis, angle = dk.rotation_log(np.asarray(r1, float) @ np.asarray(r0, float).T)
    out = np.empty((len(blend), 3, 3))
    for i, b in enumerate(blend):
        out[i] = dk.axis_rotation(axis, float(b) * angle) @ r0
    return out


def line_segment(p0, p1, r0, r1, speed_max: float, period_s: float, include_start: bool = False,
                 omega_max: float | None = None):
    """Quintic straight move p0 -> p1 with rotation slerp r0 -> r1; rows at u = k/K (k from 0 or 1).

    ``omega_max`` (rad/s) also bounds the rotation rate: a 48 deg rig turn over a
    100 mm move at 10 mm/s is only 0.08 rad/s in Cartesian terms, but near the
    wrist singularity joint 5 has to spin far faster than that and the offline
    joint planner may refuse at its configured joint-speed cap.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    ticks = _segment_ticks(float(np.linalg.norm(p1 - p0)), speed_max, period_s)
    if omega_max is not None and omega_max > 0.0:
        angle = dk.rotation_angle(np.asarray(r1, float) @ np.asarray(r0, float).T)
        ticks = max(ticks, _segment_ticks(float(angle), float(omega_max), period_s))
    k = np.arange(0 if include_start else 1, ticks + 1, dtype=float)
    blend = _quintic(k / ticks)
    p = p0[None, :] + blend[:, None] * (p1 - p0)[None, :]
    return p, _rotation_slerp(r0, r1, blend)


def hold_rows(p, r, duration_s: float, period_s: float):
    ticks = max(1, int(round(duration_s / period_s)))
    return np.repeat(np.asarray(p, float)[None, :], ticks, axis=0), np.repeat(np.asarray(r, float)[None], ticks, axis=0)


def feedforward(p: np.ndarray, period_s: float, window_ticks: int = FEEDFORWARD_WINDOW_TICKS) -> np.ndarray:
    """Tip velocity for offline planning: central differences of p, box-smoothed over `window_ticks`.

    A polyline path is C0, so its raw central difference is piecewise constant
    with a jump at every chord — and the offline carriage-IK planner turns a
    velocity jump into a carriage acceleration it caps at 0.02 m/s^2. A short
    symmetric window keeps the feedforward continuous (no lag: it is centred),
    and the loop's position gain absorbs the few-micron mismatch that remains.
    """
    p = np.asarray(p, float)
    if len(p) < 2:
        return np.zeros_like(p)
    v = np.gradient(p, period_s, axis=0)
    window = max(1, int(window_ticks) | 1)
    if window == 1 or len(v) < window:
        return v
    half = window // 2
    padded = np.concatenate([np.repeat(v[:1], half, axis=0), v, np.repeat(v[-1:], half, axis=0)])
    kernel = np.full(window, 1.0 / window)
    return np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(p.shape[1])], axis=1)


def assemble(period_s: float, parts: list[tuple]) -> Samples:
    """Stack (p, R, pen, capture|None[, dip|None]) parts; t = k*period; v = smoothed central differences of p."""
    p = np.concatenate([np.asarray(part[0], float) for part in parts])
    r = np.concatenate([np.asarray(part[1], float) for part in parts])
    pen = np.concatenate([np.full(len(part[0]), int(part[2]), dtype=np.int64) for part in parts])
    capture = np.concatenate([
        np.zeros(len(part[0]), dtype=np.int64) if part[3] is None else np.asarray(part[3], np.int64)
        for part in parts])
    dip = np.concatenate([
        np.zeros(len(part[0]), dtype=np.int64) if len(part) < 5 or part[4] is None else np.asarray(part[4], np.int64)
        for part in parts])
    n = len(p)
    t = np.arange(1, n + 1, dtype=float) * period_s
    v = feedforward(p, period_s)
    return Samples(period_s=float(period_s), t=t, p=p, v=v, R=r, pen=pen, capture=capture,
                   dip=dip if dip.any() else None)


def _pose(pose: dict):
    tip = np.asarray(pose["tip"], float)
    rotation = np.asarray(pose["rotation"], float)
    if tip.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("pose needs tip (3,) and rotation (3,3)")
    return tip, rotation


def _descend(p_from, r_from, target, r_target, normal, approach_m: float, period_s: float,
             include_start: bool = False):
    """Pen-up parts from p_from down to the surface point `target`: approach, descent, final descent.

    Straight to ``approach_m`` above the target along its normal at APPROACH_SPEED_M_S
    (rotation slerp to r_target, rate-bounded), down to FINAL_DESCENT_M at
    DESCENT_SPEED_M_S, then the last FINAL_DESCENT_M at FINAL_DESCENT_SPEED_M_S.
    Every quintic segment starts and ends at rest.
    """
    if not approach_m > FINAL_DESCENT_M:
        raise ValueError(f"approach standoff {approach_m * 1e3:.1f} mm must exceed the "
                         f"{FINAL_DESCENT_M * 1e3:.0f} mm final descent")
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    target = np.asarray(target, float)
    above = target + approach_m * normal
    near = target + FINAL_DESCENT_M * normal
    parts = []
    p, r = line_segment(p_from, above, r_from, r_target, APPROACH_SPEED_M_S, period_s,
                        include_start=include_start, omega_max=APPROACH_OMEGA_MAX_RAD_S)
    parts.append((p, r, 0, None))
    p, r = line_segment(above, near, r_target, r_target, DESCENT_SPEED_M_S, period_s)
    parts.append((p, r, 0, None))
    p, r = line_segment(near, target, r_target, r_target, FINAL_DESCENT_SPEED_M_S, period_s)
    parts.append((p, r, 0, None))
    return parts


def approach_standoff_m(config: dict) -> float:
    """The pen-up standoff the path approaches from and lifts back to: ``path.approach_mm``
    when given, else the no-scan lift for a ``no_scan`` session (the hold is the lift's
    top) and APPROACH_STANDOFF_M after a camera orbit."""
    override = config.get("path", {}).get("approach_mm")
    if override is not None:
        return float(override) * 1e-3
    return NO_SCAN_LIFT_M if config.get("no_scan") else APPROACH_STANDOFF_M


# --- corner-aware retiming ---------------------------------------------------

@dataclass(frozen=True)
class Retime:
    """Planner budgets for the corner-aware time law (see the RETIME_* constants)."""

    corner_accel_m_s2: float = RETIME_CORNER_ACCEL_M_S2
    pen_up_accel_m_s2: float = RETIME_PEN_UP_ACCEL_M_S2
    touchdown_accel_m_s2: float = RETIME_TOUCHDOWN_ACCEL_M_S2
    window_s: float = FEEDFORWARD_WINDOW_TICKS * C.period_s
    angular_accel_rad_s2: float = RETIME_ANGULAR_ACCEL_RAD_S2
    ramp_s: float = RETIME_RAMP_S
    deadband_blend_rad: float = math.radians(RETIME_DEADBAND_BLEND_DEG)
    orientation_smoothing_m: float = RETIME_ORIENTATION_SMOOTHING_M

    def report(self) -> dict:
        return {"mode": "corner", "corner_accel_mm_s2": self.corner_accel_m_s2 * 1e3,
                "pen_up_accel_mm_s2": self.pen_up_accel_m_s2 * 1e3,
                "touchdown_accel_mm_s2": self.touchdown_accel_m_s2 * 1e3,
                "angular_accel_rad_s2": self.angular_accel_rad_s2,
                "corner_window_s": self.window_s, "ramp_s": self.ramp_s,
                "lean_deadband_blend_deg": math.degrees(self.deadband_blend_rad),
                "orientation_smoothing_mm": self.orientation_smoothing_m * 1e3}


def retime_settings(config: dict, period_s: float) -> Retime | None:
    """The `path.timing` choice: None for ``"legacy"`` (the pre-2026-09-14 law), else the budgets.

    ``path.corner_accel_mm_s2``, ``path.pen_up_accel_mm_s2``, ``path.touchdown_accel_mm_s2``,
    ``path.angular_accel_rad_s2``, ``path.lean_deadband_blend_deg`` and
    ``path.orientation_smoothing_mm`` override the defaults; each must be a finite
    non-negative number and the linear accelerations positive (angular 0 disables
    the rotary budget)."""
    path = config.get("path") or {}
    mode = path.get("timing", "corner")
    if mode == "legacy":
        return None
    if mode != "corner":
        raise DrawRefusal("design", f"path.timing must be 'corner' or 'legacy', got {mode!r}")

    def number(key, default, positive):
        value = path.get(key)
        value = default if value is None else float(value) * 1e-3
        if not math.isfinite(value) or value < 0.0 or (positive and value <= 0.0):
            raise DrawRefusal("design", f"path.{key} must be a {'positive' if positive else 'non-negative'} number")
        return value

    blend = path.get("lean_deadband_blend_deg")
    blend = math.radians(RETIME_DEADBAND_BLEND_DEG if blend is None else float(blend))
    if not (math.isfinite(blend) and 0.0 <= blend <= math.radians(45.0)):
        raise DrawRefusal("design", "path.lean_deadband_blend_deg must be within 0..45")
    angular = path.get("angular_accel_rad_s2")
    angular = RETIME_ANGULAR_ACCEL_RAD_S2 if angular is None else float(angular)
    if not (math.isfinite(angular) and angular >= 0.0):
        raise DrawRefusal("design", "path.angular_accel_rad_s2 must be a non-negative number")
    return Retime(corner_accel_m_s2=number("corner_accel_mm_s2", RETIME_CORNER_ACCEL_M_S2, True),
                  pen_up_accel_m_s2=number("pen_up_accel_mm_s2", RETIME_PEN_UP_ACCEL_M_S2, True),
                  touchdown_accel_m_s2=number("touchdown_accel_mm_s2", RETIME_TOUCHDOWN_ACCEL_M_S2, True),
                  angular_accel_rad_s2=angular,
                  window_s=FEEDFORWARD_WINDOW_TICKS * float(period_s), deadband_blend_rad=blend,
                  orientation_smoothing_m=number("orientation_smoothing_mm", RETIME_ORIENTATION_SMOOTHING_M, False))


def smoothed_normal_field(surface, stroke_uv: np.ndarray, half_width_m: float, step_m: float = RETIME_ORIENTATION_GRID_M):
    """Exact unit normals along a whole stroke on an arc-length grid every ``step_m``, with
    the averaging half width: ``(arc, normals, half_width_m)`` for :func:`normal_field_lookup`
    (see RETIME_ORIENTATION_SMOOTHING_M)."""
    stroke_uv = np.asarray(stroke_uv, float)
    length = polyline_length(stroke_uv)
    count = max(2, int(math.ceil(length / step_m)) + 1)
    arc = np.linspace(0.0, length, count)
    uv, _ = resample_polyline_by_arclength(stroke_uv, arc)
    _, _, _, normals = surface.frame(uv)
    normals = np.asarray(normals, float)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return arc, normals, float(half_width_m)


def normal_field_lookup(arc: np.ndarray, normals: np.ndarray, half_width_m: float, offset_m: float = 0.0):
    """A callable mapping local arc lengths to unit normals: the grid field, read as
    piecewise linear, convolved exactly with a triangular window of +-``half_width_m``
    (the second difference of its double integral, so the result is C2 in the query
    and a knot of the grid never steps its rate), ``offset_m`` along the stroke (a
    chunk cut from it). Beyond the grid ends each end's own rate continues."""
    arc = np.asarray(arc, float)
    normals = np.asarray(normals, float)
    if half_width_m <= 0.0 or len(arc) < 2:
        def exact(s):
            s = np.asarray(s, float).reshape(-1) + float(offset_m)
            out = np.stack([np.interp(s, arc, normals[:, k]) for k in range(3)], axis=1)
            return out / np.linalg.norm(out, axis=1, keepdims=True)
        return exact
    step = float(arc[1] - arc[0])
    pad = int(math.ceil(half_width_m / step)) + 2
    # extend each end along its own rate (unnormalized), so the average at a
    # stroke end is unbiased where the field turns steadily, as on a cylinder
    before = normals[0] + (normals[0] - normals[1]) * np.arange(pad, 0, -1)[:, None]
    after = normals[-1] + (normals[-1] - normals[-2]) * np.arange(1, pad + 1)[:, None]
    grid = np.concatenate([before, normals, after])
    origin = float(arc[0]) - pad * step
    count = len(grid)
    # F: integral of the piecewise-linear field up to each knot; F2: integral of F
    slope = np.diff(grid, axis=0)
    segment = step * 0.5 * (grid[:-1] + grid[1:])
    integral = np.concatenate([np.zeros((1, 3)), np.cumsum(segment, axis=0)])
    double_segment = integral[:-1] * step + grid[:-1] * step * step / 2.0 + slope * step * step / 6.0
    double = np.concatenate([np.zeros((1, 3)), np.cumsum(double_segment, axis=0)])

    def double_integral(x):
        x = np.clip(x, origin, origin + (count - 1) * step)
        index = np.clip(np.floor((x - origin) / step).astype(int), 0, count - 2)
        xi = (x - origin - index * step)[:, None]
        return (double[index] + integral[index] * xi + grid[index] * xi * xi / 2.0
                + slope[index] * xi * xi * xi / (6.0 * step))

    def lookup(s):
        s = np.asarray(s, float).reshape(-1) + float(offset_m)
        w = float(half_width_m)
        out = (double_integral(s + w) - 2.0 * double_integral(s) + double_integral(s - w)) / (w * w)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    return lookup


def polyline_turn_angles(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(arc length at every vertex, turn angle in radians at every vertex; 0 at the ends).

    Zero-length chords are skipped when measuring the turn, so a repeated
    vertex neither hides a corner nor invents one."""
    poly = np.asarray(poly, float)
    seg = np.diff(poly, axis=0)
    length = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(length)])
    turn = np.zeros(len(poly))
    moving = np.flatnonzero(length > 0.0)
    if len(moving) >= 2:
        unit = seg[moving] / length[moving][:, None]
        cos = np.einsum("ij,ij->i", unit[:-1], unit[1:])
        # the turn belongs to the vertex that ends the earlier chord
        turn[moving[1:]] = np.arccos(np.clip(cos, -1.0, 1.0))
    return arc, turn


def corner_speed_limits(arc: np.ndarray, turn: np.ndarray, cruise: float, accel: float, window_s: float,
                        iterations: int = 40) -> np.ndarray:
    """Tip speed cap at every vertex so the direction change within the planner's
    feedforward window costs at most ``accel``: v * Theta(s, v * window) = accel * window,
    where Theta sums the turn angles of the vertices within +-v*window/2 of the vertex
    (always including its own). An isolated corner gets accel*window/theta; a dense
    arc of tiny turns gets sqrt(accel / curvature). The fixed point is found by
    bisection (the left side is monotone in v); vertices with no turn nearby keep the cruise."""
    arc = np.asarray(arc, float)
    turn = np.asarray(turn, float)
    cumulative = np.concatenate([[0.0], np.cumsum(turn)])
    budget = float(accel) * float(window_s)

    def excess(v):
        half = 0.5 * v * float(window_s)
        lo = np.searchsorted(arc, arc - half, side="left")
        hi = np.searchsorted(arc, arc + half, side="right")
        return v * (cumulative[hi] - cumulative[lo]) - budget

    caps = np.full(len(arc), float(cruise))
    limited = excess(caps) > 0.0
    if not limited.any():
        return caps
    lo = np.zeros(len(arc))
    hi = caps.copy()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        over = excess(mid) > 0.0
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
    caps[limited] = lo[limited]
    return caps


def rotation_rates(rotations: np.ndarray, arc: np.ndarray) -> np.ndarray:
    """Rotation-vector rate (rad/m, 3-vector) of each knot interval of a rotation sequence,
    small-angle: the intervals are a few micrometres apart."""
    r = np.asarray(rotations, float)
    step = np.einsum("nji,njk->nik", r[:-1], r[1:])   # R_i^T R_{i+1}, in the link frame
    vector = 0.5 * np.stack([step[:, 2, 1] - step[:, 1, 2], step[:, 0, 2] - step[:, 2, 0],
                             step[:, 1, 0] - step[:, 0, 1]], axis=1)
    ds = np.diff(np.asarray(arc, float))
    return vector / np.maximum(ds, 1e-300)[:, None]


def rotary_speed_limits(arc: np.ndarray, rates: np.ndarray, cruise: float, alpha: float, window_s: float,
                        iterations: int = 40) -> np.ndarray:
    """Speed cap at every knot so the change of angular velocity across the planner's
    feedforward window costs at most ``alpha`` (rad/s^2): v * |k(s + v w / 2) - k(s - v w / 2)|
    = alpha * w, with k the rotation-vector rate per metre of :func:`rotation_rates`
    (a hatch turnaround under a normal-following wrist reverses k; a smooth
    curve barely changes it). Bisection as in :func:`corner_speed_limits`."""
    arc = np.asarray(arc, float)
    mids = 0.5 * (arc[:-1] + arc[1:])
    rates = np.asarray(rates, float)
    budget = float(alpha) * float(window_s)

    def rate_at(s):
        return np.stack([np.interp(s, mids, rates[:, k]) for k in range(3)], axis=1)

    def excess(v):
        half = 0.5 * v * float(window_s)
        change = np.linalg.norm(rate_at(arc + half) - rate_at(arc - half), axis=1)
        return v * change - budget

    caps = np.full(len(arc), float(cruise))
    limited = excess(caps) > 0.0
    if not limited.any():
        return caps
    lo = np.zeros(len(arc))
    hi = caps.copy()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        over = excess(mid) > 0.0
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
    caps[limited] = lo[limited]
    return caps


def _refine_knots(arc: np.ndarray, values: np.ndarray, max_step: float, fill: float):
    """Insert knots so no arc-length gap exceeds ``max_step``; inserted knots take ``fill``."""
    arc = np.asarray(arc, float)
    values = np.asarray(values, float)
    gaps = np.diff(arc)
    counts = np.maximum(1, np.ceil(gaps / max_step - 1e-12).astype(int))
    if not (counts > 1).any():
        return arc.copy(), values.copy()
    pieces_s = [arc[:1]]
    pieces_v = [values[:1]]
    for i in np.flatnonzero(counts >= 1):
        if counts[i] > 1:
            inner = arc[i] + gaps[i] * np.arange(1, counts[i]) / counts[i]
            pieces_s.append(inner)
            pieces_v.append(np.full(len(inner), fill))
        pieces_s.append(arc[i + 1:i + 2])
        pieces_v.append(values[i + 1:i + 2])
    return np.concatenate(pieces_s), np.concatenate(pieces_v)


def _ramp_pass(arc: np.ndarray, caps: np.ndarray, accel: np.ndarray, jerk, v_start: float | None):
    """Forward pass: the fastest speed at each knot reachable from the knot before it.

    With ``jerk`` (per interval, m/s^3) the acceleration of a ramp is also held
    under sqrt(2 j (v - v_base)) (it grows from zero at the speed the ramp left,
    ``v_base``) and under sqrt(2 j (cap - v)) (it decays to zero as the next cap
    is reached), so a ramp that reaches its plateau is an S-curve; a ramp cut
    short by the opposite pass keeps a step in acceleration at the cut."""
    v = np.asarray(caps, float).copy()
    if v_start is not None:
        v[0] = min(v[0], float(v_start))
    base = v[0]
    for i in range(len(arc) - 1):
        ds = arc[i + 1] - arc[i]
        a = float(accel[i])
        reach = math.sqrt(v[i] * v[i] + 2.0 * a * ds)
        if jerk is not None and ds > 0.0:
            j = float(jerk[i])
            room = v[i] - base
            if room <= 0.0:
                reach = min(reach, base + 0.5 * j * (6.0 * ds / j) ** (2.0 / 3.0))
            else:
                reach = min(reach, math.sqrt(v[i] * v[i] + 2.0 * min(a, math.sqrt(2.0 * j * room)) * ds))
            gap = caps[i + 1] - v[i]
            if gap > 0.0:
                reach = min(reach, math.sqrt(v[i] * v[i] + 2.0 * math.sqrt(2.0 * j * gap) * ds))
        v[i + 1] = min(v[i + 1], reach)
        if v[i + 1] >= caps[i + 1] - 1e-15:
            base = v[i + 1]
    return v


def accel_limited_speeds(arc: np.ndarray, caps: np.ndarray, accel, v_start: float | None = None,
                         v_end: float | None = None, jerk=None) -> np.ndarray:
    """Largest speed profile under the per-knot caps whose tangential acceleration
    stays within ``accel`` (a scalar, or one value per knot interval): the forward /
    backward v^2 = v0^2 + 2 a ds passes, each S-curved by ``jerk`` (m/s^3, scalar or
    per interval) when given. ``v_start`` / ``v_end`` pin the end speeds."""
    arc = np.asarray(arc, float)
    caps = np.asarray(caps, float)
    ds = np.diff(arc)
    a = np.broadcast_to(np.asarray(accel, float), ds.shape)
    j = None if jerk is None else np.broadcast_to(np.asarray(jerk, float), ds.shape)
    forward = _ramp_pass(arc, caps, a, j, v_start)
    backward = _ramp_pass(arc[-1] - arc[::-1], caps[::-1], a[::-1], None if j is None else j[::-1], v_end)[::-1]
    return np.minimum(forward, backward)


def budgeted_speed_profile(knots: np.ndarray, caps: np.ndarray, accel, retime: Retime, cruise: float,
                           rotations=None, v_start: float | None = None, v_end: float | None = None):
    """Speeds at ``knots`` under the caps, the linear budget ``accel`` (scalar or per
    interval) and, with the knot ``rotations``, the rotary budget
    ``retime.angular_accel_rad_s2``: the speed is also capped where the angular
    velocity would change too fast (:func:`rotary_speed_limits`) and the ramps are
    held to alpha / k where the wrist turns k rad per metre of path, so a
    normal-following wrist reverses at a hatch turnaround no faster than a pen-up
    leg turns. Returns (speeds, per-interval acceleration budgets)."""
    knots = np.asarray(knots, float)
    caps = np.asarray(caps, float).copy()
    accel = np.broadcast_to(np.asarray(accel, float), (len(knots) - 1,)).copy()
    alpha = retime.angular_accel_rad_s2
    if rotations is not None and alpha > 0.0 and len(knots) > 2:
        rates = rotation_rates(rotations, knots)
        caps = np.minimum(caps, rotary_speed_limits(knots, rates, cruise, alpha, retime.window_s))
        k = np.linalg.norm(rates, axis=1)
        accel = np.minimum(accel, alpha / np.maximum(k, 1e-9))
    jerk = accel / retime.ramp_s
    speeds = accel_limited_speeds(knots, caps, accel, v_start=v_start, v_end=v_end, jerk=jerk)
    # A ramp cut by the opposite pass before either reaches a cap is a
    # triangle: its acceleration flips sign in one knot. Lower the caps over
    # such a hump to its peak and integrate again, so both S-curves decay
    # their acceleration to zero into a plateau there instead; each pass
    # lowers the peak a little further until the S-curves meet at rest.
    for _ in range(8):
        below = speeds < caps * (1.0 - 1e-9)
        edges = np.flatnonzero(np.diff(np.concatenate([[False], below, [False]]).astype(int)))
        lowered = False
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            region = speeds[start:stop]
            top = int(np.argmax(region))
            peak = float(region[top])
            # a ramp that reaches a cap peaks at one end of its region; a cut ramp
            # peaks strictly inside it, both ends clearly lower
            if 0 < top < len(region) - 1 and region[0] < peak * (1.0 - 1e-6) and region[-1] < peak * (1.0 - 1e-6):
                caps[start:stop] = np.minimum(caps[start:stop], peak)
                lowered = True
        if not lowered:
            break
        speeds = accel_limited_speeds(knots, caps, accel, v_start=v_start, v_end=v_end, jerk=jerk)
    return speeds, accel


def hermite_arclength(t_knots: np.ndarray, s_knots: np.ndarray, v_knots: np.ndarray, t: np.ndarray):
    """Cubic Hermite s(t) and ds/dt through (t, s, v) knots; exact for constant-acceleration
    intervals, C1 everywhere, clipped to the knot range."""
    t_knots = np.asarray(t_knots, float)
    s_knots = np.asarray(s_knots, float)
    v_knots = np.asarray(v_knots, float)
    t = np.clip(np.asarray(t, float), t_knots[0], t_knots[-1])
    index = np.clip(np.searchsorted(t_knots, t, side="right") - 1, 0, len(t_knots) - 2)
    h = t_knots[index + 1] - t_knots[index]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(h > 0.0, (t - t_knots[index]) / np.where(h > 0.0, h, 1.0), 0.0)
    u2 = u * u
    u3 = u2 * u
    s0, s1 = s_knots[index], s_knots[index + 1]
    v0, v1 = v_knots[index] * h, v_knots[index + 1] * h
    s = (2 * u3 - 3 * u2 + 1) * s0 + (u3 - 2 * u2 + u) * v0 + (-2 * u3 + 3 * u2) * s1 + (u3 - u2) * v1
    ds = (6 * u2 - 6 * u) * s0 + (3 * u2 - 4 * u + 1) * v0 + (-6 * u2 + 6 * u) * s1 + (3 * u2 - 2 * u) * v1
    sdot = np.where(h > 0.0, ds / np.where(h > 0.0, h, 1.0), 0.0)
    return s, sdot


def corner_time_law(poly: np.ndarray, cruise: float, ease_s: float, period_s: float, retime: Retime,
                    rotation_at=None):
    """The legacy stroke law (quintic ease / cruise / ease over arc length) dilated at corners.

    Returns (t, s, sdot, info) sampled at t = k*period like :func:`time_law`. Where
    the polyline turns, the tick's arc-length speed is held to the corner caps of
    :func:`corner_speed_limits` with ramps at ``retime.corner_accel_m_s2``; where it
    does not, the legacy law is reproduced (same ticks, same ease, same cruise).
    ``rotation_at`` (arc lengths -> link-6 rotations) adds the rotary budget of
    :func:`budgeted_speed_profile`. Every returned s lies on the same polyline;
    the duration grows by the time the corners cost, reported in ``info``."""
    poly = np.asarray(poly, float)
    arc, turn = polyline_turn_angles(poly)
    length = float(arc[-1])
    duration = max(length / cruise + ease_s, 2.0 * ease_s + period_s)
    t, s, sdot = time_law(length, duration, ease_s, period_s)
    info = {"nominal_duration_s": duration, "duration_s": float(t[-1]), "added_s": 0.0,
            "limited_vertices": 0, "min_speed_mm_s": cruise * 1e3, "turn_max_deg": float(np.degrees(turn.max()))}
    caps = corner_speed_limits(arc, turn, cruise, retime.corner_accel_m_s2, retime.window_s)
    knots, knot_caps = _refine_knots(arc, caps, RETIME_KNOT_STEP_M, cruise)
    rotations = None if rotation_at is None else np.asarray(rotation_at(knots), float)
    knot_speed, _ = budgeted_speed_profile(knots, knot_caps, retime.corner_accel_m_s2, retime, cruise, rotations)
    limited = knot_speed < cruise * (1.0 - 1e-9)
    if not limited.any():
        return t, s, sdot, info
    # Dilate the legacy law: dt' = (cruise / v_limit(s)) dt on the union of the
    # legacy tick positions (exact times) and the ramp knots (linear in between,
    # a few nanometres at most). Where no corner reaches, lambda = 1 and the
    # legacy ticks come back unchanged; the ease regions stay exact.
    s_legacy = np.concatenate([[0.0], s])
    t_legacy = np.concatenate([[0.0], t])
    grid = np.unique(np.concatenate([s_legacy, np.clip(knots, 0.0, length)]))
    grid = grid[np.concatenate([[True], np.diff(grid) > 1e-9])]   # a tick and a knot at one place
    t_grid = np.interp(grid, s_legacy, t_legacy)
    dilation = cruise / np.interp(grid, knots, knot_speed)
    steps = np.diff(t_grid) * 0.5 * (dilation[:-1] + dilation[1:])
    t_dilated = np.concatenate([[0.0], np.cumsum(steps)])
    secant = np.diff(grid) / np.maximum(np.diff(t_dilated), 1e-300)
    v_dilated = np.concatenate([[0.0], 0.5 * (secant[:-1] + secant[1:]), [0.0]])
    total = float(t_dilated[-1])
    ticks = int(math.ceil(total / period_s - 1e-9))
    if ticks > dk.PLAN_MAX_TICKS:
        raise DrawRefusal("design", f"retimed stroke needs {ticks} ticks, outside the guarded range")
    t_new = np.minimum(total, np.arange(1, ticks + 1, dtype=float) * period_s)
    s_new, sdot_new = hermite_arclength(t_dilated, grid, v_dilated, t_new)
    s_new = np.clip(np.maximum.accumulate(s_new), 0.0, length)
    s_new[-1] = length
    sdot_new = np.clip(sdot_new, 0.0, cruise)
    info.update({"duration_s": total, "added_s": total - float(t[-1]), "limited_vertices": int((caps < cruise).sum()),
                 "min_speed_mm_s": float(knot_speed.min()) * 1e3})
    return t_new, s_new, sdot_new, info


def _slerp_rows(r0, r1, fraction: np.ndarray) -> np.ndarray:
    return _rotation_slerp(np.asarray(r0, float), np.asarray(r1, float), np.asarray(fraction, float))


def pen_up_leg(points, r_from, r_to, speed_caps, retime: Retime, period_s: float, *,
               gentle_start: bool = False, gentle_end: bool = False, include_start: bool = False,
               omega_max: float | None = None):
    """One continuous pen-up motion along the polyline ``points`` (M >= 2 rows), from rest to rest.

    ``speed_caps`` (M-1) bound each chord; corners between chords get the
    :func:`corner_speed_limits` cap for the pen-up acceleration budget; the
    profile accelerates, cruises and decelerates at ``retime.pen_up_accel_m_s2``
    (S-curved, and under the rotary budget of :func:`budgeted_speed_profile`)
    except within FINAL_DESCENT_M of a ``gentle_start`` / ``gentle_end`` end,
    where the touchdown budget applies. The rotation slerps r_from -> r_to over
    the first chord as a quintic function of distance (its rate starts and ends
    the chord at zero) and holds r_to after it; ``omega_max`` bounds that chord's
    speed so the peak rotation rate stays under it. Returns (p, R, chord index
    per row). The geometry is exactly the polyline: a tick's position is a point of it."""
    points = np.asarray(points, float)
    caps = np.asarray(speed_caps, float)
    seg = np.diff(points, axis=0)
    length = np.linalg.norm(seg, axis=1)
    keep = length > 0.0
    if not keep.any():
        raise ValueError("pen-up leg has no length")
    r_from = np.asarray(r_from, float)
    r_to = np.asarray(r_to, float)
    # The matrix difference stays exactly zero for identical chunk endpoints;
    # acos(trace(R @ R.T)) can invent a small turn from roundoff there.
    angle = 2.0 * math.asin(min(1.0, float(np.linalg.norm(r_to - r_from)) / math.sqrt(8.0)))
    first = int(np.argmax(keep))
    if angle > 1e-9 and first != 0:
        raise ValueError("pen-up leg cannot turn in place: its first chord has no length")
    chord_index = np.flatnonzero(keep)
    vertices = np.concatenate([points[:1], points[1:][keep]])
    chord_len = length[keep]
    chord_caps = caps[keep].copy()
    if omega_max is not None and omega_max > 0.0 and angle > 1e-9:
        chord_caps[0] = min(chord_caps[0], float(omega_max) * chord_len[0] / (QUINTIC_PEAK * angle))
    arc, turn = polyline_turn_angles(vertices)
    cruise = float(chord_caps.max())
    vertex_caps = corner_speed_limits(arc, turn, cruise, retime.pen_up_accel_m_s2, retime.window_s)
    # a vertex is also bound by the slower of the chords meeting there
    vertex_caps[:-1] = np.minimum(vertex_caps[:-1], chord_caps)
    vertex_caps[1:] = np.minimum(vertex_caps[1:], chord_caps)
    knots, knot_caps = _refine_knots(arc, vertex_caps, RETIME_PEN_UP_KNOT_STEP_M, math.inf)
    chord_of_knot = np.clip(np.searchsorted(arc, knots, side="right") - 1, 0, len(chord_len) - 1)
    knot_caps = np.minimum(knot_caps, chord_caps[chord_of_knot])
    mids = 0.5 * (knots[:-1] + knots[1:])
    # the touchdown budget over the gentle FINAL_DESCENT_M, blended linearly
    # into the pen-up budget over the next FINAL_DESCENT_M so the acceleration
    # budget itself never steps
    accel = np.full(len(mids), retime.pen_up_accel_m_s2)
    span = retime.pen_up_accel_m_s2 - retime.touchdown_accel_m_s2
    if gentle_start:
        blend = np.clip((mids - FINAL_DESCENT_M) / FINAL_DESCENT_M, 0.0, 1.0)
        accel = np.minimum(accel, retime.touchdown_accel_m_s2 + span * blend)
    if gentle_end:
        blend = np.clip((arc[-1] - FINAL_DESCENT_M - mids) / FINAL_DESCENT_M, 0.0, 1.0)
        accel = np.minimum(accel, retime.touchdown_accel_m_s2 + span * blend)

    def rotation_at(s):
        if angle <= 1e-9:
            return np.repeat(r_to[None], len(s), axis=0)
        return _slerp_rows(r_from, r_to, _quintic(np.clip(s / chord_len[0], 0.0, 1.0)))

    speed, _ = budgeted_speed_profile(knots, knot_caps, accel, retime, cruise, rotation_at(knots),
                                      v_start=0.0, v_end=0.0)
    pair = speed[:-1] + speed[1:]
    if not np.all(pair > 0.0):
        raise ValueError("pen-up profile stalls")
    t_knots = np.concatenate([[0.0], np.cumsum(2.0 * np.diff(knots) / pair)])
    total = float(t_knots[-1])
    ticks = int(math.ceil(total / period_s - 1e-9))
    if ticks > dk.PLAN_MAX_TICKS:
        raise DrawRefusal("design", f"pen-up leg needs {ticks} ticks, outside the guarded range")
    t = np.arange(0 if include_start else 1, ticks + 1, dtype=float) * period_s
    t = np.minimum(t, total)
    s, _ = hermite_arclength(t_knots, knots, speed, t)
    s = np.clip(np.maximum.accumulate(s), 0.0, arc[-1])
    s[-1] = arc[-1]
    p, _ = resample_polyline_by_arclength(vertices, s)
    p[-1] = vertices[-1]
    row_chord = np.clip(np.searchsorted(arc, s, side="right") - 1, 0, len(chord_len) - 1)
    return p, rotation_at(s), chord_index[row_chord]


def _descend_continuous(p_from, r_from, target, r_target, normal, approach_m: float, period_s: float,
                        retime: Retime, include_start: bool = False):
    """The :func:`_descend` polyline (hold -> standoff -> FINAL_DESCENT_M above -> contact) as one
    continuous pen-up leg, returned as the same three labelled parts. A hold that already sits
    at the standoff with a different rotation turns there first, as the legacy travel did."""
    if not approach_m > FINAL_DESCENT_M:
        raise ValueError(f"approach standoff {approach_m * 1e3:.1f} mm must exceed the "
                         f"{FINAL_DESCENT_M * 1e3:.0f} mm final descent")
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    target = np.asarray(target, float)
    p_from = np.asarray(p_from, float)
    above = target + approach_m * normal
    near = target + FINAL_DESCENT_M * normal
    parts = []
    r_leg = np.asarray(r_from, float)
    if np.linalg.norm(above - p_from) <= 1e-12 and dk.rotation_angle(np.asarray(r_target, float) @ r_leg.T) > 1e-9:
        p, r = line_segment(p_from, above, r_from, r_target, APPROACH_SPEED_M_S, period_s,
                            include_start=include_start, omega_max=APPROACH_OMEGA_MAX_RAD_S)
        parts.append((p, r, 0, None))
        include_start = False
        r_leg = np.asarray(r_target, float)
    p, r, chord = pen_up_leg(
        np.stack([p_from, above, near, target]), r_leg, r_target,
        [APPROACH_SPEED_M_S, DESCENT_SPEED_M_S, FINAL_DESCENT_SPEED_M_S], retime, period_s,
        gentle_end=True, include_start=include_start, omega_max=APPROACH_OMEGA_MAX_RAD_S)
    for index in range(3):
        rows = chord == index
        if index == 0 and parts:
            # the in-place turn above already stands for the travel part
            parts[0] = (np.concatenate([parts[0][0], p[rows]]), np.concatenate([parts[0][1], r[rows]]), 0, None)
            continue
        parts.append((p[rows], r[rows], 0, None))
    return parts


# --- the path -----------------------------------------------------------------


def _orientation_normals_at(field, arc_lengths, exact):
    """The normals the rotation transport follows at ``arc_lengths``: ``field`` (a
    :func:`normal_field_lookup`) when given, else the exact ones; and the largest
    angle between the two (degrees)."""
    if field is None:
        return exact, 0.0
    smoothed = np.asarray(field(arc_lengths), float)
    deviation = _angle_deg(smoothed, exact)
    return smoothed, float(deviation.max()) if len(deviation) else 0.0


def _stroke_rotation_lookup(surface, stroke, field, reference_normal, r_c, deadband, blend):
    """Arc lengths along ``stroke`` -> transported link-6 rotations, for the rotary budget."""
    def rotation_at(arc_lengths):
        uv, _ = resample_polyline_by_arclength(stroke, arc_lengths)
        _, _, _, exact = surface.frame(uv)
        exact = np.asarray(exact, float)
        exact /= np.linalg.norm(exact, axis=1, keepdims=True)
        chosen, _ = _orientation_normals_at(field, arc_lengths, exact)
        return transported_rotations(chosen, reference_normal, r_c, deadband, blend)
    return rotation_at


def _stroke_timing(stroke, speed, ease_s, period_s, retime, rotation_at):
    """(arc lengths per tick, arc speed per tick, timing report) for one pen-down stroke.

    A requested cruise speed is an upper bound. Preserve both configured easing
    ramps on short strokes by slowing them, never dropping geometry or making
    their acceleration ramps shorter to fit the requested speed."""
    length = polyline_length(stroke)
    duration = max(length / speed + ease_s, 2.0 * ease_s + period_s)
    if retime is None:
        law_t, arc, speed_law = time_law(length, duration, ease_s, period_s)
        return arc, speed_law, {"nominal_duration_s": duration, "duration_s": float(law_t[-1]), "added_s": 0.0}
    _, arc, speed_law, timing = corner_time_law(stroke, speed, ease_s, period_s, retime, rotation_at)
    return arc, speed_law, timing


def _lift_leg(p_from, r_from, p_to, period_s, retime):
    """Straight pen-up lift off the surface: the legacy quintic, or the continuous leg
    with the gentle touchdown budget over its first FINAL_DESCENT_M."""
    if retime is None:
        return line_segment(p_from, p_to, r_from, r_from, LIFT_SPEED_M_S, period_s)
    p, r, _ = pen_up_leg(np.stack([p_from, p_to]), r_from, r_from, [LIFT_SPEED_M_S], retime, period_s,
                         gentle_start=True)
    return p, r


def _descent_parts(p_from, r_from, target, r_target, normal, approach, period_s, retime, include_start):
    """The three labelled pen-up parts down to a stroke start, legacy or continuous."""
    if retime is None:
        return _descend(p_from, r_from, target, r_target, normal, approach, period_s, include_start=include_start)
    return _descend_continuous(p_from, r_from, target, r_target, normal, approach, period_s, retime,
                               include_start=include_start)


def _surface_points_normals(surface, uv):
    points, _, _, normals = surface.frame(uv)
    return points, normals


def _placement_ids_per_stroke(placement_ids, stroke_count):
    if placement_ids is None:
        return None
    if (len(placement_ids) != stroke_count
            or any(not isinstance(value, str) or not value for value in placement_ids)):
        raise DrawRefusal("design", "session placement IDs must match surface strokes")
    return placement_ids


def _surface_material_orientation(projected, normal_field, arc, reference_normal, r_c, deadband, blend):
    orientation_normals, deviation = _orientation_normals_at(normal_field, arc, projected.normals)
    rotations = transported_rotations(orientation_normals, reference_normal, r_c, deadband, blend)
    return rotations, deviation


MAX_SESSION_CHUNKS = material.MAX_SESSION_CHUNKS


class _SessionNormalFields:
    """Smoothed normal fields per source stroke (built on first use), so every chunk cut
    from a stroke reads the rotation transport off one field and shares the cut rotation."""

    def __init__(self, surface, strokes, smoothing_m: float):
        self.surface = surface
        self.strokes = strokes
        self.smoothing_m = float(smoothing_m)
        self.fields: dict[int, tuple] = {}

    def lookup(self, source: int, offset_m: float):
        """The chunk's field (local arc length -> normals), or None when following the exact model."""
        if self.smoothing_m <= 0.0:
            return None
        if source not in self.fields:
            self.fields[source] = smoothed_normal_field(self.surface, self.strokes[source], self.smoothing_m)
        return normal_field_lookup(*self.fields[source], offset_m=offset_m)

    def chunk(self, source: int, offset_m: float, exact_normal):
        """(normal the chunk's contact rotation is transported to, ``orientation_normals`` for its compile)."""
        field = self.lookup(source, offset_m)
        if field is None:
            return np.asarray(exact_normal, float), None
        return field(np.zeros(1))[0], [field]


def chunk_end(first_chunk, end_chunk, source_chunk_count):
    """Validate an exclusive material address without changing the partition."""
    if first_chunk >= source_chunk_count:
        raise DrawRefusal("design", "first uncommitted chunk is outside the material plan")
    if type(end_chunk) is not int or not first_chunk < end_chunk <= source_chunk_count:
        raise DrawRefusal("design", "invalid look-ahead end chunk")
    return end_chunk


def _project_contact_candidate(surface, candidate, index, count, placement_id):
    def projector(uv):
        return _surface_points_normals(surface, uv)

    selected = material.project_candidate(
        candidate, index, count, placement_id, projector, sample_projector=projector)
    if selected.normals is None:
        raise DrawRefusal('design', 'contact material needs measured surface normals')
    return selected


def _contact_candidate_geometry(surface, selected, contact, hold, period_s, *,
                                config, reference, normal_field, cursor, model,
                                max_contact_s, cap_ticks):
    """Build one contact interval's measured geometry and interaction phases."""
    candidate = selected.candidate
    stroke, speed = candidate.uv, candidate.speed_m_s
    tip_c, _ = _pose(contact)
    tip_hold, r_hold = _pose(hold)
    reference_normal, reference_rotation = reference
    approach = approach_standoff_m(config)
    ease_s = float(config.get('ease_s', 0.05))
    deadband = math.radians(float(config.get('lean_deadband_deg') or 0.0))
    lift, contact_offset = pen_offsets(config)
    placement = lift + contact_offset
    retime = retime_settings(config, period_s)
    blend = retime.deadband_blend_rad if retime else 0.0
    base_from_root = dk.base_from_root if model is None else model.base_from_root
    normals = selected.normals
    if normals is None or selected.projector is None:
        raise DrawRefusal('design', 'contact material needs a measured surface projection')
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    vertex_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(stroke, axis=0), axis=1))]
    points_base = base_from_root(np.asarray(selected.points, float) + placement * normals)
    orientation_normals_here, _ = _orientation_normals_at(normal_field, vertex_arc, normals)
    rotations = transported_rotations(orientation_normals_here, reference_normal,
                                      reference_rotation, deadband, blend)
    start_point, start_normal, start_rotation = points_base[0], normals[0], rotations[0]
    descent = _descent_parts(tip_hold, r_hold, start_point, start_rotation, start_normal,
                             approach, period_s, retime, True)
    before = [operation.PhasePart(kind, part, 0) for kind, part in zip(
        ('travel', 'approach', 'contact_intent'), descent, strict=True)]
    settle_p, settle_r = hold_rows(start_point, start_rotation, DESCENT_SETTLE_S, period_s)
    before.append(operation.PhasePart('contact_dwell', (settle_p, settle_r, 0, None), 0))
    length = polyline_length(stroke)
    if length <= 0.0:
        raise DrawRefusal('design', 'surface stroke 0 has no length')
    rotation_at = _stroke_rotation_lookup(surface, stroke, normal_field, reference_normal,
                                          reference_rotation, deadband, blend)
    arc, speed_law, timing = _stroke_timing(stroke, speed, ease_s, period_s, retime, rotation_at)
    work = operation.WorkPolicy(
        arc, 'contact',
        orientation=lambda sampled: _surface_material_orientation(
            sampled, normal_field, arc, reference_normal, reference_rotation, deadband, blend),
        projector=selected.projector,
        placement=lambda sampled: base_from_root(np.asarray(sampled.points, float)
                                                 + placement * sampled.normals),
        normalize_normals=True, pen=1, stroke_index=0)

    def after_work(rows):
        final_point = rows.points[-1] + approach * rows.normals[-1]
        p, rotation = _lift_leg(rows.points[-1], rows.rotations[-1], final_point, period_s, retime)
        return [operation.PhasePart('retreat', (p, rotation, 0, None), 0)]

    def contact_report(planned):
        samples, phases = planned.samples, planned.phases
        start_row, end_row = planned.work_range
        up = samples.pen == 0
        step_length = np.linalg.norm(np.diff(samples.p, axis=0), axis=1)
        pen_up_edges = up[:-1] & up[1:]
        report = {
            'kind': 'execution-program-path', 'design_length_mm': length * 1e3,
            'strokes': [{'stroke_index': 0, 'length_mm': length * 1e3,
                         'duration_s': timing['duration_s'],
                         'nominal_duration_s': timing['nominal_duration_s'],
                         'cruise_speed_mm_s': float(speed_law.max()) * 1e3,
                         'samples': len(planned.rows.points), 'sample_range': [start_row, end_row],
                         'timing': timing,
                         'orientation_normal_deviation_max_deg': planned.rows.orientation_report}],
            'stroke_sample_ranges': [[start_row, end_row]], 'segments': phases.segments,
            'anchor_gap_mm': float(np.linalg.norm(points_base[0] - tip_c)) * 1e3,
            'contact_normal_base': [float(value) for value in reference_normal],
            'approach_mm': approach * 1e3, 'lean_deadband_deg': math.degrees(deadband),
            'lift_mm': lift * 1e3, 'contact_offset_mm': contact_offset * 1e3,
            'timing': {'mode': 'legacy'} if retime is None else retime.report(),
            'contact_nominal_s': timing['nominal_duration_s'],
            'contact_added_s': timing['added_s'], 'sample_count': samples.n,
            'duration_s': samples.duration_s,
            'pen_up_travel_mm': float(step_length[pen_up_edges].sum()) * 1e3,
            'dips': [], 'dip_count': 0,
        }
        # Cuts obey the nominal contact budget; corner retiming may add the
        # reported time, while the whole candidate stays under the tick cap.
        nominal_ticks = int(math.ceil(report['contact_nominal_s'] / period_s - 1e-9))
        if nominal_ticks > cap_ticks or samples.n > dk.PLAN_MAX_TICKS:
            raise DrawRefusal('design', 'generated chunk exceeds contact or planner tick budget')
        report.update({**planned.material_origin, 'max_contact_s': max_contact_s,
                       'contact_s': float(np.count_nonzero(samples.pen)) * period_s,
                       'hardware_authority': False})
        return report

    return candidate_plan.CandidateGeometry(
        cursor, work, before, after_work, period_s, assemble, contact_report)


def iter_chunk_candidates(surface, strokes_uv, speeds_m_s, contact, hold, period_s, *,
                            config, max_contact_s, preflight=None, contact_resources=None,
                            first_chunk=0, end_chunk=None,
                            placement_ids=None,
                            chunk_origin=None,
                            plan_candidate=None,
                            model: dk.ArmModel | None = None):
    """Yield `(material candidate, samples, report)` after each passes preflight.

    Only one sampled motion chunk is retained by this generator. The session
    has a separate finite chunk budget; PLAN_MAX_TICKS remains a per-plan bound.

    ``surface`` supplies the actual chart and normals; rotations are never used
    as a substitute for a surface normal. A geometry-only caller supplies
    ``preflight(samples, report)`` and must return exactly True after offline
    acceptance. Session preparation instead supplies ``contact_resources``;
    its setup, native Draw check and post-acceptance accounting run inside the
    shared planner. Neither path executes motion. Runtime execution still
    needs preflight from fresh measured state.
    The original contact orientation/reference is shared across every cut.
    ``first_chunk`` selects the remaining material without changing the cuts or
    orientation reference; only its first approach starts at the supplied hold.
    ``end_chunk`` is an exclusive look-ahead boundary. Material outside that
    interval is partitioned for identity but never sampled or preflighted.
    ``chunk_origin`` numbers a successor's candidate locally before native
    acceptance; the source offset remains lineage in its report.
    ``plan_candidate`` receives the selected material and projected geometry
    before native acceptance so a source-bound one-Draw producer can own that
    acceptance and publication without changing batch callers.
    """
    if type(first_chunk) is not int or not 0 <= first_chunk < MAX_SESSION_CHUNKS:
        raise DrawRefusal("design", "invalid first uncommitted chunk")
    if ((preflight is None) == (contact_resources is None)
            or (preflight is not None and not callable(preflight))
            or not math.isfinite(period_s) or period_s <= 0.0
            or not math.isfinite(max_contact_s) or max_contact_s <= 0.0):
        raise DrawRefusal("design", "session chunks require a budget, period and preflight")
    if period_s != C.period_s or max_contact_s > dk.PLAN_MAX_TICKS * period_s:
        raise DrawRefusal("design", "session period or contact budget exceeds the offline planner contract")
    strokes = [np.asarray(stroke, float) for stroke in strokes_uv]
    speeds = [float(speed) for speed in speeds_m_s]
    if not strokes or len(strokes) != len(speeds):
        raise DrawRefusal("design", "session strokes/speeds mismatch")
    placement_ids = _placement_ids_per_stroke(placement_ids, len(strokes))
    ease = float(config.get("ease_s", 0.05))
    cap_ticks = int(math.floor(max_contact_s / period_s + 1e-9))
    # Reserve a tick for floating-point ceil in the shared time law.
    duration_cap = (cap_ticks - 1) * period_s
    if not math.isfinite(ease) or ease <= 0.0 or duration_cap <= 2.0 * ease:
        raise DrawRefusal("design", "contact budget cannot contain the shared easing law")
    try:
        pieces = material.material_candidates(strokes, speeds,
            max_lengths_m=[speed * (duration_cap - ease) for speed in speeds],
            max_candidates=MAX_SESSION_CHUNKS)
    except ValueError as error:
        raise DrawRefusal("design", str(error)) from error
    end_chunk = chunk_end(first_chunk, len(pieces) if end_chunk is None else end_chunk, len(pieces))
    if (chunk_origin is not None
            and (type(chunk_origin) is not int or not 0 <= chunk_origin <= first_chunk)):
        raise DrawRefusal('design', 'invalid source chunk origin')
    # The taught rotation belongs to the contact address, not to the first
    # artwork vertex. Moving an unchanged design must not change this reference.
    contact_tip, reference_rotation = _pose(contact)
    root_from_base = dk.root_from_base if model is None else model.root_from_base
    base_from_root = dk.base_from_root if model is None else model.base_from_root
    contact_uv, _ = surface.project(root_from_base(contact_tip)[None, :])
    _, _, _, normals = surface.frame(contact_uv)
    reference_normal = np.asarray(normals[0], float)
    reference_normal /= np.linalg.norm(reference_normal)
    reference = (reference_normal, reference_rotation)
    deadband = math.radians(float(config.get("lean_deadband_deg") or 0.0))
    lift, contact_offset = pen_offsets(config)
    placement = lift + contact_offset
    retime = retime_settings(config, period_s)
    blend = retime.deadband_blend_rad if retime else 0.0
    fields = _SessionNormalFields(surface, strokes, retime.orientation_smoothing_m if retime else 0.0)
    current_hold = hold
    for index, candidate in enumerate(pieces[first_chunk:end_chunk], start=first_chunk):
        source, arc_range = candidate.source_stroke, candidate.arc_range_m
        local_index = index - (chunk_origin or 0)
        source_count = len(pieces) - (chunk_origin or 0)
        placement_id = None if placement_ids is None else placement_ids[source]
        selected = _project_contact_candidate(
            surface, candidate, local_index, source_count, placement_id)
        point, normal = selected.points, np.asarray(selected.normals[0], float).copy()
        normal /= np.linalg.norm(normal)
        normal, orientation_normals = fields.chunk(source, arc_range[0], normal)
        rotation = transported_rotations(normal[None], reference_normal, reference_rotation, deadband, blend)[0]
        chunk_contact = {"tip": base_from_root(np.asarray(point[0], float) + placement * normal).tolist(),
                         "rotation": rotation.tolist()}
        cursor = operation.StrokeCursor(
            candidate, local_index, source_count, placement_id, chunk_origin)
        geometry = _contact_candidate_geometry(
            surface, selected, chunk_contact, current_hold, period_s,
            config=config, reference=reference,
            normal_field=None if orientation_normals is None else orientation_normals[0],
            cursor=cursor, model=model, max_contact_s=max_contact_s, cap_ticks=cap_ticks)
        planner = candidate_plan.plan_next if plan_candidate is None else plan_candidate
        planned = planner(geometry, contact_resources=contact_resources).planned
        report = planned.report
        samples = planned.samples
        if preflight is not None and preflight(samples, report) is not True:
            raise DrawRefusal("runtime_preflight", f"session chunk {index} was not accepted")
        current_hold = {"tip": samples.p[-1].tolist(), "rotation": samples.R[-1].tolist()}
        yield candidate, samples, report


# --- preflight ------------------------------------------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def preflight(samples: Samples, surface, config: dict, tool_axis_in_link6, start_pose: dict,
              design_length_m: float | None = None, *, model: dk.ArmModel | None = None) -> dict:
    """Refuse (DrawRefusal) or report, per docs/surface-formats.md. `surface` may be None for an orbit."""
    p, v, r = samples.p, samples.v, samples.R
    if samples.n == 0:
        raise DrawRefusal("nan", "no samples")
    if not (np.isfinite(p).all() and np.isfinite(v).all() and np.isfinite(r).all()):
        raise DrawRefusal("nan", "a sample is not finite")

    tip_s, r_s = _pose(start_pose)
    start_gap = float(np.linalg.norm(p[0] - tip_s))
    start_turn = dk.rotation_angle(r[0] @ r_s.T)
    if start_gap > START_TOLERANCE_M:
        raise DrawRefusal(
            "start_tolerance",
            f"row 1 is {start_gap * 1e3:.2f} mm from the start pose")

    speeds = np.linalg.norm(v, axis=1)
    tip_speed_max = float(speeds.max())
    down_rows = samples.pen > 0
    down_max = float(speeds[down_rows].max()) if down_rows.any() else 0.0
    if down_max > TIP_SPEED_CAP_M_S:
        raise DrawRefusal("tip_speed", f"pen-down {down_max * 1e3:.2f} mm/s exceeds {TIP_SPEED_CAP_M_S * 1e3} mm/s")
    if tip_speed_max > TIP_SPEED_CAP_PEN_UP_M_S:
        raise DrawRefusal("tip_speed", f"{tip_speed_max * 1e3:.2f} mm/s exceeds {TIP_SPEED_CAP_PEN_UP_M_S * 1e3} mm/s")

    report = {
        "sample_count": samples.n,
        "duration_s": samples.duration_s,
        "tip_speed_max_mm_s": tip_speed_max * 1e3,
        "travel_length_mm": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) * 1e3,
        "start_gap_mm": start_gap * 1e3,
        "start_turn_rad": start_turn,
        "pen_down_count": int(samples.pen.sum()),
        "lean_max_deg": None,
        "normal_swing_max_deg": None,
        "path_length_mm": None,
        "arc_length_ratio": None,
        "holes": 0,
    }
    down = samples.pen > 0
    if surface is None or not down.any():
        return report

    axis = np.asarray(tool_axis_in_link6, float)
    p_down_root = (dk.root_from_base(p[down]) if model is None else model.root_from_base(p[down]))
    uv, signed = surface.project(p_down_root)
    uv = np.asarray(uv, float)
    signed = np.asarray(signed, float)
    if not (np.isfinite(uv).all() and np.isfinite(signed).all()):
        raise DrawRefusal("nan", "a pen-down sample does not project onto the surface")
    _, normals = lift_to_surface(surface, uv)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    half_w = 0.5 * float(surface.width_m)
    half_h = 0.5 * float(surface.height_m)
    outside = (np.abs(uv[:, 0]) > half_w) | (np.abs(uv[:, 1]) > half_h)
    if outside.any():
        raise DrawRefusal("girth", f"{int(outside.sum())} pen-down samples fall outside the mapped canvas")
    chart = getattr(surface, "chart", None)
    chart_kind = getattr(chart, "kind", "plane")
    if chart_kind == "cylinder":
        radius = float(getattr(chart, "radius_m", getattr(chart, "radius", math.nan)))
        extent_v = float(uv[:, 1].max() - uv[:, 1].min())
        limit = math.pi * radius * GIRTH_FRACTION
        if extent_v > limit:
            raise DrawRefusal(
                "girth", f"design spans {extent_v * 1e3:.1f} mm around a {radius * 1e3:.1f} mm cylinder "
                f"(limit {limit * 1e3:.1f} mm)")

    count = np.asarray(surface.count)
    rows, cols = count.shape
    col = np.rint((uv[:, 0] + half_w) / (2.0 * half_w) * (cols - 1)).astype(int) if cols > 1 else np.zeros(len(uv), int)
    row = np.rint((uv[:, 1] + half_h) / (2.0 * half_h) * (rows - 1)).astype(int) if rows > 1 else np.zeros(len(uv), int)
    col = np.clip(col, 0, cols - 1)
    row = np.clip(row, 0, rows - 1)
    mapped = count > 0
    hole_fill_m = float(config.get("map", {}).get("hole_fill_mm", 0.0)) * 1e-3
    interpolated = 0
    if hole_fill_m > 0.0 and rows > 1 and cols > 1:
        # Admit only cells within the stated metric radius of an observation.
        # A square dilation can exceed that radius, even for a one-cell gap.
        du, dv = 2.0 * half_w / (cols - 1), 2.0 * half_h / (rows - 1)
        grown = mapped.copy()
        row_radius = min(rows - 1, int(math.ceil(hole_fill_m / dv)))
        col_radius = min(cols - 1, int(math.ceil(hole_fill_m / du)))
        for dr in range(-row_radius, row_radius + 1):
            for dc in range(-col_radius, col_radius + 1):
                if math.hypot(dr*dv, dc*du) > hole_fill_m + 1e-12:
                    continue
                r0, r1 = max(0, dr), min(rows, rows+dr)
                c0, c1 = max(0, dc), min(cols, cols+dc)
                grown[r0:r1, c0:c1] |= mapped[r0-dr:r1-dr, c0-dc:c1-dc]
        interpolated = int((grown & ~mapped)[row, col].sum())
        mapped = grown
    holes = int((~mapped[row, col]).sum())
    report["holes"] = holes
    report["interpolated_samples"] = interpolated
    report["hole_fill_mm"] = hole_fill_m * 1e3
    if holes:
        raise DrawRefusal("holes", f"{holes} pen-down samples land on unmapped cells")

    tool_axes = np.einsum("nij,j->ni", r[down], axis)
    lean = _angle_deg(tool_axes, -normals)
    report["lean_max_deg"] = float(lean.max())
    budget = float(config.get("lean_budget_deg", C.path.lean_budget_deg))
    if report["lean_max_deg"] > budget:
        raise DrawRefusal("lean_over_budget", f"tool leans {report['lean_max_deg']:.1f} deg off the normal (budget {budget})")

    swing = _angle_deg(np.repeat(normals[:1], len(normals), axis=0), normals)
    report["normal_swing_max_deg"] = float(swing.max())
    if report["normal_swing_max_deg"] > NORMAL_SWING_CAP_DEG:
        raise DrawRefusal("normal_swing", f"surface normal swings {report['normal_swing_max_deg']:.1f} deg across the design")

    surface_length = float(np.linalg.norm(np.diff(p_down_root, axis=0), axis=1).sum())
    chart_length = float(np.linalg.norm(np.diff(uv, axis=0), axis=1).sum())
    denominator = chart_length if design_length_m is None else float(design_length_m)
    report["path_length_mm"] = surface_length * 1e3
    report["chart_length_mm"] = chart_length * 1e3
    report["arc_length_ratio"] = surface_length / denominator if denominator > 0.0 else None
    report["signed_dist_max_mm"] = float(np.abs(signed).max()) * 1e3
    report["chart_kind"] = chart_kind
    return report


# --- samples file ---------------------------------------------------------------

def _fmt(value: float) -> str:
    return format(float(value), ".12g")


def _arm_header(model: dk.ArmModel) -> list[tuple[str, str]]:
    """Serialize the physical chain for the offline C++ planner adapter.

    The stable configured arm ID remains in the session context. A renamed ID
    can bind the same physical URDF prefix without changing the follower CSV.
    The parser currently refuses other prefixes; the offline checker
    can accept one only with an explicit, model-verified ``--arm-prefix``.
    """
    frame = ("frame", model.frame)
    return [frame] if model.prefix == "right" else [("arm", model.prefix), frame]


def _samples_model(arm, model):
    if model is None:
        return dk.ArmModel("right" if arm is None else arm)
    if arm is not None and arm != model.arm:
        raise ValueError(f"sample arm {arm!r} differs from model arm {model.arm!r}")
    return model


def write_samples_csv(path, samples: Samples, kind: str, tip_in_link6, extra: dict | None = None,
                      arm: str | None = None, *, model: dk.ArmModel | None = None) -> None:
    """docs/surface-formats.md samples format: key,value header, `columns,...`, one row per tick.

    The `constants_sha` line names the config/motion_constants.json the planner ran
    with; the offline planner refuses a file compiled against another.
    A follower file is unchanged; another physical chain declares its prefix
    ahead of its base frame. The offline C++ adapter accepts right and left;
    offline checks can bind a separately verified WXAI prefix explicitly.
    """
    if kind not in ("orbit", "path"):
        raise ValueError(f"kind must be orbit or path, got {kind!r}")
    model = _samples_model(arm, model)
    tip = np.asarray(tip_in_link6, float)
    header = [
        ("schema", SAMPLES_SCHEMA), ("constants_sha", SHA), ("kind", kind),
        *_arm_header(model),
        ("period_s", _fmt(samples.period_s)),
        ("tip_x_m", _fmt(tip[0])), ("tip_y_m", _fmt(tip[1])), ("tip_z_m", _fmt(tip[2])),
        ("sample_count", str(samples.n)),
    ]
    if kind == "orbit":
        header.append(("capture_count", str(int(samples.capture.max()) if samples.n else 0)))
    header.append(("start_tolerance_m", _fmt(START_TOLERANCE_M)))
    if kind == "path":
        header.append(("dip_count", str(int(samples.dips.max()) if samples.n else 0)))
    reserved = {key for key, _ in header} | {"columns"}
    fixed = dict(header)
    for key, value in (extra or {}).items():
        if "," in key or "\n" in key:
            raise ValueError(f"extra key {key!r} is malformed")
        if key in reserved:
            if key == "columns" or str(value) != fixed[key]:
                raise ValueError(f"extra key {key!r} conflicts with the fixed header")
            continue
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (int, np.integer)):
            text = str(int(value))
        elif isinstance(value, (float, np.floating)):
            text = _fmt(value)
        elif value is None:
            text = ""
        else:
            text = str(value)
            if "," in text or "\n" in text:
                raise ValueError(f"extra value for {key!r} must not contain commas or newlines")
        header.append((key, text))
    lines = [f"{key},{value}" for key, value in header]
    lines.append("columns," + ",".join(COLUMNS))
    flat = samples.R.reshape(samples.n, 9)
    dips = samples.dips
    for i in range(samples.n):
        row = [_fmt(samples.t[i])]
        row += [_fmt(x) for x in samples.p[i]]
        row += [_fmt(x) for x in samples.v[i]]
        row += [_fmt(x) for x in flat[i]]
        row += [str(int(samples.pen[i])), str(int(samples.capture[i])), str(int(dips[i]))]
        lines.append(",".join(row))
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


# Header values that stay strings: a 12-hex SHA can look like an int or a float.
TEXT_HEADER_KEYS = frozenset({"schema", "constants_sha", "kind", "frame", "arm"})


def read_samples_csv(path) -> tuple[Samples, dict]:
    """Inverse of write_samples_csv: (Samples, header dict with scalars parsed)."""
    header: dict = {}
    rows = []
    columns = None
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            if columns is None:
                key, _, value = line.partition(",")
                if key == "columns":
                    columns = value.split(",")
                    if tuple(columns) not in (COLUMNS, LEGACY_COLUMNS):
                        raise ValueError(f"unexpected columns {columns}")
                    continue
                header[key] = value if key in TEXT_HEADER_KEYS else _parse_scalar(value)
                continue
            rows.append([float(x) for x in line.split(",")])
    if header.get("schema") != SAMPLES_SCHEMA:
        raise ValueError(f"unknown samples schema {header.get('schema')!r}")
    if header.get("constants_sha") not in (None, SHA):
        raise ValueError(f"samples file constants_sha {header['constants_sha']} is not this planner's {SHA}")
    if columns is None:
        raise ValueError("samples file has no columns line")
    data = np.asarray(rows, float).reshape(-1, len(columns))
    dip = data[:, 18].astype(np.int64) if len(columns) == len(COLUMNS) else None
    samples = Samples(
        period_s=float(header["period_s"]),
        t=data[:, 0], p=data[:, 1:4], v=data[:, 4:7], R=data[:, 7:16].reshape(-1, 3, 3),
        pen=data[:, 16].astype(np.int64), capture=data[:, 17].astype(np.int64),
        dip=dip if dip is not None and dip.any() else None)
    if int(header.get("sample_count", samples.n)) != samples.n:
        raise ValueError("sample_count header disagrees with the row count")
    return samples, header


def _parse_scalar(text: str):
    if text == "":
        return None
    if text in ("true", "false"):
        return text == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
