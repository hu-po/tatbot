"""Design strokes -> surface -> tip samples for `tatbot draw` (numpy only).

Contract: docs/draw.md. This module turns a design (v1: the spiral), an
anchored `HeightFieldSurface` (scripts/lib/draw_surface.py, imported lazily)
and the executor's contact/hold poses into the samples file the C++ executor
streams — and it writes the orbit the executor flies before the map exists.

Geometry conventions:

- The surface lives in `root`; samples are written in `right/base_link`
  ("base"). The two differ by one translation (draw_kinematics.BASE_IN_ROOT),
  so normals and rotations are the same in both.
- Orientation rule (decision 3): R_i = align(n_c -> n_i) @ R_c. The operator's
  approach angle at contact is preserved relative to the surface; spin changes
  minimally.
- Time law: the C++ spiral's quintic ease / cruise / ease over arc length,
  sampled at t = k * period for k = 1..ceil(duration / period).
- Every refusal is a `DrawRefusal(code, detail)`; nothing is clamped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import draw_kinematics as dk
import numpy as np

SAMPLES_SCHEMA = "tatbot.draw-samples/1"
SAMPLES_FRAME = "right/base_link"
START_TOLERANCE_M = 0.001
START_TOLERANCE_RAD = 0.02
TIP_SPEED_CAP_M_S = 0.020
NORMAL_SWING_CAP_DEG = 60.0
GIRTH_FRACTION = 0.8          # of pi * radius: the injective half of a cylinder chart, with margin
# Pen-up travel between the hold and the surface: hold -> APPROACH_STANDOFF_M above
# the anchor at APPROACH_SPEED_M_S (rotation slerp bounded by APPROACH_OMEGA_MAX),
# down at DESCENT_SPEED_M_S to FINAL_DESCENT_M above the surface, the last
# millimetres at FINAL_DESCENT_SPEED_M_S, then the settle. The first draw
# (2026-09-01) approached from the 80 mm orbit standoff at 10 mm/s and descended
# at 5 mm/s: 29 s the operator called "slow and from too far".
APPROACH_STANDOFF_M = 0.030
APPROACH_SPEED_M_S = 0.020
APPROACH_OMEGA_MAX_RAD_S = math.radians(8.0)
DESCENT_SPEED_M_S = 0.010
FINAL_DESCENT_M = 0.005
FINAL_DESCENT_SPEED_M_S = 0.003
LIFT_SPEED_M_S = 0.010
ORBIT_HOLD_S = 1.0            # stationary rows before a capture (the executor also waits for the measured arm)
DESCENT_SETTLE_S = 1.0        # pen-up hold at the contact before the pen-down handover
QUINTIC_PEAK = 1.875          # max of d/du (10u^3 - 15u^4 + 6u^5)
MIN_SEGMENT_S = 0.25
DESIGN_STEP_M = 1e-5          # chord length of the design polylines; the spiral's origin has a 0.16 mm radius
FEEDFORWARD_WINDOW_TICKS = 21  # symmetric box over the central-difference velocity (52 ms at 400 Hz)
COLUMNS = ("t_s", "px", "py", "pz", "vx", "vy", "vz",
           "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22", "pen", "capture")


class DrawRefusal(RuntimeError):  # noqa: N818 - the contract's name
    """A preflight refusal: `code` is one of the names in docs/draw.md."""

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

    @property
    def n(self) -> int:
        return int(len(self.t))

    @property
    def duration_s(self) -> float:
        return float(self.t[-1]) if self.n else 0.0


# --- design geometry ---------------------------------------------------------

def spiral_path_length(radius_m: float, turns: float) -> float:
    """Closed-form Archimedean arc length, the C++ formula."""
    total_angle = 2.0 * math.pi * turns
    scale = radius_m / total_angle
    return 0.5 * scale * (total_angle * math.sqrt(1.0 + total_angle * total_angle) + math.asinh(total_angle))


def spiral_polyline(radius_m: float, turns: float, step_m: float = 1e-4) -> np.ndarray:
    """Archimedean spiral r = a*theta from the origin to (radius, 0), sampled every `step_m` of arc."""
    if not (radius_m > 0.0 and turns > 0.0 and step_m > 0.0):
        raise ValueError("spiral radius, turns and step must be positive")
    total_angle = 2.0 * math.pi * turns
    scale = radius_m / total_angle
    length = spiral_path_length(radius_m, turns)
    count = max(2, int(math.ceil(length / step_m)) + 1)
    distance = np.linspace(0.0, length, count)
    angle = _spiral_angle(distance, scale, total_angle, length)
    radius = scale * angle
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


def _spiral_angle(distance, scale, total_angle, length):
    """Newton inversion of the spiral arc length, vectorised; the C++ does six iterations."""
    angle = total_angle * np.asarray(distance, float) / length
    for _ in range(6):
        root = np.sqrt(1.0 + angle * angle)
        integrated = 0.5 * scale * (angle * root + np.arcsinh(angle))
        angle = angle - (integrated - distance) / (scale * root)
        angle = np.clip(angle, 0.0, total_angle)
    return angle


def design_strokes(design: dict) -> list[np.ndarray]:
    """The design as polylines in chart metres, design x -> u, y -> v, centred on the anchor."""
    kind = design.get("kind")
    if kind != "spiral":
        raise DrawRefusal("design", f"unsupported design kind {kind!r} (v1 draws the spiral only)")
    strokes = [spiral_polyline(float(design["radius_mm"]) * 1e-3, float(design["turns"]), DESIGN_STEP_M)]
    rotation = math.radians(float(design.get("rotation_deg", 0.0)))
    if rotation:
        c, s = math.cos(rotation), math.sin(rotation)
        rot = np.array([[c, -s], [s, c]])
        strokes = [stroke @ rot.T for stroke in strokes]
    return strokes


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


def _quintic_rate(u):
    u = np.asarray(u, float)
    u2 = u * u
    return 30.0 * u2 - 60.0 * u2 * u + 30.0 * u2 * u2


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


def resample_polyline_by_arclength(poly: np.ndarray, s: np.ndarray):
    """Points and unit tangents of a polyline at arc lengths `s` (linear between vertices)."""
    poly = np.asarray(poly, float)
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s = np.clip(np.asarray(s, float), 0.0, cum[-1])
    index = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(seg_len[index] > 0.0, (s - cum[index]) / seg_len[index], 0.0)
    points = poly[index] + frac[:, None] * seg[index]
    good = seg_len > 0.0
    unit = np.zeros_like(seg)
    unit[good] = seg[good] / seg_len[good][:, None]
    tangents = unit[index]
    return points, tangents


# --- surface lift ------------------------------------------------------------

def lift_to_surface(surface, uv: np.ndarray):
    """(points_root (N,3), normals (N,3)) at chart coordinates `uv` via surface.frame."""
    point, _, _, normal = surface.frame(np.asarray(uv, float))
    return np.asarray(point, float), np.asarray(normal, float)


def transported_rotations(normals: np.ndarray, n_c, r_c: np.ndarray, deadband_rad: float = 0.0) -> np.ndarray:
    """R_i = align(n_c -> n_i) @ R_c for every row of `normals` (decision 3).

    ``deadband_rad`` > 0 relaxes the rule: the rotation carrying n_c to n_i is
    shortened by that angle (about the same axis) and vanishes when the normal
    swing is inside it, so the tool leans up to the deadband off the local
    normal before the wrist follows. On a 38 mm bottle the +-19 deg swing of a
    15 mm spiral otherwise rolls joint 3 through ~100 deg per turn from a touch
    near the wrist singularity (runs 9078 and f3b0, 2026-09-02) and no speed
    passes the joint-velocity cap; a 12 deg deadband passes at 3.5 mm/s with
    joint velocity 0.19 of 0.25 rad/s. The preflight's lean budget still applies.
    """
    normals = np.asarray(normals, float)
    n_c = np.asarray(n_c, float)
    n_c = n_c / np.linalg.norm(n_c)
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    v = np.cross(n_c[None, :], n)
    c = n @ n_c
    if deadband_rad > 0.0:
        angle = np.arccos(np.clip(c, -1.0, 1.0))
        reduced = np.maximum(angle - float(deadband_rad), 0.0)
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
    wrist singularity joint 5 has to spin far faster than that and the executor
    refuses at its 0.25 rad/s cap (e2e from the carriage-IK witness pose).
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


def arc_segment(center, standoff_m: float, n_a, n_b, n_c, r_c, speed_max: float, period_s: float):
    """Quintic move on the standoff sphere from direction n_a to n_b (rotation transported from n_c)."""
    center = np.asarray(center, float)
    n_a = np.asarray(n_a, float) / np.linalg.norm(n_a)
    n_b = np.asarray(n_b, float) / np.linalg.norm(n_b)
    axis, angle = dk.rotation_log(dk.align_rotation(n_a, n_b))
    ticks = _segment_ticks(standoff_m * angle, speed_max, period_s)
    blend = _quintic(np.arange(1, ticks + 1, dtype=float) / ticks)
    normals = np.stack([dk.axis_rotation(axis, float(b) * angle) @ n_a for b in blend])
    p = center[None, :] + standoff_m * normals
    return p, transported_rotations(normals, n_c, r_c)


def hold_rows(p, r, duration_s: float, period_s: float):
    ticks = max(1, int(round(duration_s / period_s)))
    return np.repeat(np.asarray(p, float)[None, :], ticks, axis=0), np.repeat(np.asarray(r, float)[None], ticks, axis=0)


def feedforward(p: np.ndarray, period_s: float, window_ticks: int = FEEDFORWARD_WINDOW_TICKS) -> np.ndarray:
    """Tip velocity for the executor: central differences of p, box-smoothed over `window_ticks`.

    A polyline path is C0, so its raw central difference is piecewise constant
    with a jump at every chord — and the executor's carriage-IK loop turns a
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


def assemble(period_s: float, parts: list[tuple[np.ndarray, np.ndarray, int, np.ndarray | None]]) -> Samples:
    """Stack (p, R, pen, capture|None) parts; t = k*period; v = smoothed central differences of p."""
    p = np.concatenate([np.asarray(part[0], float) for part in parts])
    r = np.concatenate([np.asarray(part[1], float) for part in parts])
    pen = np.concatenate([np.full(len(part[0]), int(part[2]), dtype=np.int64) for part in parts])
    capture = np.concatenate([
        np.zeros(len(part[0]), dtype=np.int64) if part[3] is None else np.asarray(part[3], np.int64)
        for part in parts])
    n = len(p)
    t = np.arange(1, n + 1, dtype=float) * period_s
    v = feedforward(p, period_s)
    return Samples(period_s=float(period_s), t=t, p=p, v=v, R=r, pen=pen, capture=capture)


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


# --- the path -----------------------------------------------------------------

def compile_path(surface, config: dict, contact: dict, hold: dict, period_s: float):
    """Full path in base from the hold pose: approach, descent, strokes on the surface, lift.

    Returns (Samples, report). The contact normal n_c is the anchored surface's
    normal at the contact tip; the design is laid out in chart coordinates
    around that anchor and every stroke rotation is transported from R_c.
    """
    tip_c, r_c = _pose(contact)
    tip_hold, r_hold = _pose(hold)
    approach = float(config.get("path", {}).get("approach_mm", APPROACH_STANDOFF_M * 1e3)) * 1e-3
    duration_s = float(config["duration_s"])
    ease_s = float(config["ease_s"])
    draw_speed = float(config.get("draw_speed_mm_s") or 0.0) * 1e-3
    if not (math.isfinite(draw_speed) and draw_speed >= 0.0):
        raise DrawRefusal("design", f"draw_speed_mm_s must be a positive number or absent, got {draw_speed * 1e3}")
    deadband = math.radians(float(config.get("lean_deadband_deg") or 0.0))
    if not (math.isfinite(deadband) and 0.0 <= deadband <= math.radians(90.0)):
        raise DrawRefusal("design", f"lean_deadband_deg must be within 0..90, got {math.degrees(deadband)}")

    c_root = dk.root_from_base(tip_c)
    uv_c, dist_c = surface.project(c_root[None, :])
    uv_c = np.asarray(uv_c, float)[0]
    anchor_root, n_c = lift_to_surface(surface, uv_c[None, :])
    anchor_root = anchor_root[0]
    n_c = n_c[0] / np.linalg.norm(n_c[0])
    anchor_base = dk.base_from_root(anchor_root)
    anchor_gap_m = float(np.linalg.norm(anchor_base - tip_c))

    strokes = design_strokes(config["design"])
    lengths = [polyline_length(stroke) for stroke in strokes]
    total_length = float(sum(lengths))
    if total_length <= 0.0:
        raise DrawRefusal("design", "the design has no length")

    parts = _descend(tip_hold, r_hold, anchor_base, r_c, n_c, approach, period_s, include_start=True)
    approach_ticks = len(parts[0][0])
    # Settle at the contact with the pen still "up" (carriage locked): the descent may end
    # a few tenths of a mm off under the pen-up model-error cap, and burning that off after
    # the handover drives the carriage past its 1 mm/s cap (first live bottle path). At the
    # 4/s position gain a second of hold leaves ~1 % of the residual for the pen-down regime.
    p, r = hold_rows(anchor_base, r_c, DESCENT_SETTLE_S, period_s)
    parts.append((p, r, 0, None))

    stroke_reports = []
    last_point = anchor_base
    last_normal = n_c
    last_rotation = r_c
    for index, (stroke, length) in enumerate(zip(strokes, lengths, strict=True)):
        uv_stroke = stroke + uv_c[None, :]
        if index > 0:
            first_root, first_normal = lift_to_surface(surface, uv_stroke[:1])
            first_base = dk.base_from_root(first_root[0])
            first_normal = first_normal[0]
            r_first = transported_rotations(first_normal[None], n_c, r_c, deadband)[0]
            up = last_point + approach * last_normal
            p, r = line_segment(last_point, up, last_rotation, last_rotation, LIFT_SPEED_M_S, period_s)
            parts.append((p, r, 0, None))
            parts += _descend(up, last_rotation, first_base, r_first, first_normal, approach, period_s)
            p, r = hold_rows(first_base, r_first, DESCENT_SETTLE_S, period_s)
            parts.append((p, r, 0, None))
        # A draw speed sets the cruise directly (cruise = length / (duration - ease));
        # without one the configured duration is split over the strokes by length.
        stroke_duration = length / draw_speed + ease_s if draw_speed > 0.0 else duration_s * length / total_length
        if not stroke_duration > 2.0 * ease_s:
            raise DrawRefusal("design", f"stroke {index} ({length * 1e3:.1f} mm) gets {stroke_duration:.2f} s, "
                                        f"not more than twice the {ease_s:g} s ease")
        t, s, sdot = time_law(length, stroke_duration, ease_s, period_s)
        uv, _ = resample_polyline_by_arclength(uv_stroke, s)
        points_root, normals = lift_to_surface(surface, uv)
        r = transported_rotations(normals, n_c, r_c, deadband)
        p = dk.base_from_root(points_root)
        parts.append((p, r, 1, None))
        stroke_reports.append({
            "length_mm": length * 1e3, "duration_s": stroke_duration,
            "cruise_speed_mm_s": float(sdot.max()) * 1e3, "samples": len(p)})
        last_point = p[-1]
        last_normal = normals[-1] / np.linalg.norm(normals[-1])
        last_rotation = r[-1]

    p, r = line_segment(last_point, last_point + approach * last_normal, last_rotation, last_rotation,
                        LIFT_SPEED_M_S, period_s)
    parts.append((p, r, 0, None))

    samples = assemble(period_s, parts)
    report = {
        "kind": "path",
        "design": dict(config["design"]),
        "design_length_mm": total_length * 1e3,
        "strokes": stroke_reports,
        "anchor_uv_m": [float(uv_c[0]), float(uv_c[1])],
        "anchor_signed_dist_mm": float(np.asarray(dist_c, float)[0]) * 1e3,
        "anchor_gap_mm": anchor_gap_m * 1e3,
        "contact_normal_base": [float(x) for x in n_c],
        "approach_mm": approach * 1e3,
        "draw_speed_mm_s": draw_speed * 1e3 if draw_speed > 0.0 else None,
        "lean_deadband_deg": math.degrees(deadband),
        "approach_ticks": approach_ticks,
        "sample_count": samples.n,
        "duration_s": samples.duration_s,
    }
    return samples, report


# --- the orbit ----------------------------------------------------------------

def orbit_viewpoints(n_c, poses: int, tilt_rad: float):
    """Unit view directions: straight above, then +-tilt about e_u, then about e_v."""
    n_c = np.asarray(n_c, float) / np.linalg.norm(n_c)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n_c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e_u = np.cross(n_c, helper)
    e_u /= np.linalg.norm(e_u)
    e_v = np.cross(n_c, e_u)
    order = [None, (e_u, tilt_rad), (e_u, -tilt_rad), (e_v, tilt_rad), (e_v, -tilt_rad)]
    if not 1 <= poses <= len(order):
        raise ValueError(f"orbit poses must be 1..{len(order)}, got {poses}")
    views = []
    for entry in order[:poses]:
        views.append(n_c if entry is None else dk.axis_rotation(entry[0], entry[1]) @ n_c)
    return np.stack(views)


ORBIT_LIFT_M = 0.020          # straight lift off the contact before the rig starts turning
PATCH_RING_M = 0.008          # the least design patch the cameras must see: the contact plus this ring
PATCH_MARGIN_M = 0.003        # ring = design reach about the anchor + this, when that is larger
OFF_AXIS_BACKOFF_RAD = math.radians(5.0)   # camera-orbit off-axis step when the ring does not fit the frustums
OFF_AXIS_MIN_RAD = math.radians(10.0)


def design_patch_ring_m(design: dict | None) -> float:
    """The ring about the anchor the cameras must see: the design's reach + PATCH_MARGIN_M, at least PATCH_RING_M."""
    if not design:
        return PATCH_RING_M
    try:
        strokes = design_strokes(design)
    except (DrawRefusal, KeyError, TypeError, ValueError):
        return PATCH_RING_M
    reach = max((float(np.linalg.norm(stroke, axis=1).max()) for stroke in strokes if len(stroke)), default=0.0)
    return max(PATCH_RING_M, reach + PATCH_MARGIN_M)


def _patch(tip_c, n_c, ring_m: float = PATCH_RING_M):
    helper = np.array([1.0, 0.0, 0.0]) if abs(n_c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e_u = np.cross(n_c, helper)
    e_u /= np.linalg.norm(e_u)
    e_v = np.cross(n_c, e_u)
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = [tip_c] + [tip_c + ring_m * (np.cos(a) * e_u + np.sin(a) * e_v) for a in angles]
    pts = np.stack(ring)
    return pts, np.tile(n_c, (len(pts), 1))


def _frame_from_two(x_target, y_hint, x_src, y_src):
    """Rotation R with R x_src = x_target and R y_src as close to y_hint as the right angle allows."""
    def basis(x, y):
        x = x / np.linalg.norm(x)
        y = y - (y @ x) * x
        y /= np.linalg.norm(y)
        return np.stack([x, y, np.cross(x, y)], axis=1)
    return basis(x_target, y_hint) @ basis(x_src, y_src).T


def camera_orbit_viewpoints(tip_c, r_c, n_c, poses: int, tilt_rad: float, distance_m: float,
                            off_axis_rad: float, carriage_m: float, max_incidence_deg: float = 50.0,
                            ring_m: float = PATCH_RING_M):
    """Viewpoints that put the CAMERAS ``distance_m`` from the contact, looking at it (decision: camera-centric orbit).

    The rig is rigid: the two D405s converge ~147 mm ahead at the tip and their
    wide image axis is link-6 y. For each direction n_k (the tip-orbit's tilt
    set) the mean camera sits at P + distance n_k; the mean view is rotated
    ``off_axis_rad`` away from -n_k toward the tangent direction e_k (the wide
    axis), so the patch sits off-centre on the pen-free side of the image and
    the tip lands at ``distance - 147 cos a`` above and ``147 sin a`` across,
    the pose the operator chose by hand on 2026-09-01 (~45 mm up, ~80 mm
    across, cameras 145-176 mm off the patch). Both sides of the wide axis are
    tried; the side that keeps the whole patch in both frustums with the
    lowest incidence is preferred. Returns the workable sides as a list of
    (viewpoints, side, scores) in preference order; refuses when none works.
    The caller runs each through the advisory joint planner, because a side
    that looks fine in Cartesian terms can still ask the wrist for a joint it
    does not have (the carriage-IK witness pose sits near the wrist
    singularity and one side pins joint 3 at its limit).
    """
    rig = dk.wrist_camera_rig()
    tip6 = dk.BALLPOINT_TIP_IN_LINK6 + float(carriage_m) * dk.CARRIAGE_AXIS_IN_LINK6
    offset6 = tip6 - rig["mean_position"]
    pts, pts_n = _patch(tip_c, n_c, ring_m)
    if not 1 <= poses <= 5:
        raise DrawRefusal("orbit_view", f"orbit poses must be 1..5, got {poses}")
    candidates = []
    for side in (1.0, -1.0):
        e_ref = side * (r_c @ rig["wide_axis"])
        e_ref = e_ref - (e_ref @ n_c) * n_c
        if np.linalg.norm(e_ref) < 1e-6:
            continue
        e_ref /= np.linalg.norm(e_ref)
        # Tilts aligned with the rig: about the wide axis (the cameras swing along
        # their baseline) and about the baseline (both cameras swing together), so
        # the incidence each camera sees is tilt plus the fixed baseline half-angle.
        b_ref = np.cross(n_c, e_ref)
        order = [None, (e_ref, tilt_rad), (e_ref, -tilt_rad), (b_ref, tilt_rad), (b_ref, -tilt_rad)]
        views_n = [n_c if entry is None else dk.axis_rotation(entry[0], entry[1]) @ n_c for entry in order[:poses]]
        viewpoints, scores, ok, worst = [], [], True, 0.0
        for n_k in views_n:
            e_k = e_ref - (e_ref @ n_k) * n_k
            e_k /= np.linalg.norm(e_k)
            view_k = -math.cos(off_axis_rad) * n_k + math.sin(off_axis_rad) * e_k
            r_k = _frame_from_two(view_k, e_k, rig["view"], rig["wide_axis"])
            centre_k = tip_c + distance_m * n_k
            tip_k = centre_k + r_k @ offset6
            cams = {role: dk.camera_view(pose, pts, pts_n) for role, pose in dk.rig_cameras(tip_k, r_k, carriage_m).items()}
            if any(c[0] < 1.0 or not (c[2] <= max_incidence_deg) for c in cams.values()):
                ok = False
            worst = max([worst] + [c[2] for c in cams.values() if c[2] == c[2]])
            viewpoints.append((tip_k, r_k))
            scores.append({"tip_height_mm": float((tip_k - tip_c) @ n_k) * 1e3,
                           "tip_offset_mm": float(np.linalg.norm((tip_k - tip_c) - ((tip_k - tip_c) @ n_k) * n_k)) * 1e3,
                           "cameras": {role: {"in_view": c[0], "distance_mm": c[1] * 1e3, "incidence_deg": c[2]}
                                       for role, c in cams.items()}})
        if ok:
            candidates.append((worst, viewpoints, side, scores))
    if not candidates:
        raise DrawRefusal("orbit_view", "no side of the wide image axis keeps the contact patch in both D405 frustums "
                                        f"under {max_incidence_deg:.0f} deg incidence at every viewpoint; lower "
                                        "off_axis_deg or tilt_deg, or use orbit mode tip")
    candidates.sort(key=lambda c: c[0])
    return [(viewpoints, side, scores) for _, viewpoints, side, scores in candidates]


def orbit_samples(config: dict, trigger: dict, period_s: float, tool_axis_in_link6=None):
    """Lift from contact to standoff, visit the viewpoints with a hold (capture row) at each."""
    tip_c, r_c = _pose(trigger)
    orbit = config.get("orbit", {})
    standoff = float(orbit.get("standoff_mm", 120.0)) * 1e-3
    tilt = math.radians(float(orbit.get("tilt_deg", 15.0)))
    poses = int(orbit.get("poses", 5))
    speed = float(orbit.get("speed_mm_s", 20.0)) * 1e-3
    axis = dk.tool_axis_in_link6() if tool_axis_in_link6 is None else np.asarray(tool_axis_in_link6, float)
    n_c = -(r_c @ axis)
    n_c /= np.linalg.norm(n_c)

    mode = str(orbit.get("mode", "camera"))
    if mode == "camera":
        distance = float(orbit.get("camera_distance_mm", 160.0)) * 1e-3
        off_axis_config = math.radians(float(orbit.get("off_axis_deg", 35.0)))
        ring = design_patch_ring_m(config.get("design"))
        carriage_m = float(trigger.get("carriage_m", dk.CARRIAGE_IK_BIAS_M))
        max_incidence = float(orbit.get("max_incidence_deg", 50.0))
        omega_max = math.radians(float(orbit.get("rotation_deg_s", 8.0)))
        refusals = []
        samples = None
        chosen = None
        speed_factor = 1.0

        def build(viewpoints, factor):
            parts = []
            lifted = tip_c + ORBIT_LIFT_M * n_c
            p, r = line_segment(tip_c, lifted, r_c, r_c, speed * factor, period_s, include_start=True)
            parts.append((p, r, 0, None))
            prev_p, prev_r = lifted, r_c
            for k, (tip_k, r_k) in enumerate(viewpoints):
                p, r = line_segment(prev_p, tip_k, prev_r, r_k, speed * factor, period_s,
                                    omega_max=omega_max * factor)
                parts.append((p, r, 0, None))
                p, r = hold_rows(tip_k, r_k, ORBIT_HOLD_S, period_s)
                capture = np.zeros(len(p), dtype=np.int64)
                capture[-1] = k + 1
                parts.append((p, r, 0, capture))
                prev_p, prev_r = tip_k, r_k
            return assemble(period_s, parts)

        joints = trigger.get("joints")
        off_axis = off_axis_config
        frustum_refused = []
        # Off-axis backoff, 5 deg steps down to OFF_AXIS_MIN: the 35 deg default puts an
        # 8 mm patch at the edge of both images -- on the first draw (2026-09-01) every
        # capture's depth ended 13 mm from the contact on the pen side and a 15 mm
        # design landed on unmapped cells; 30 deg from the same touch kept an 18-23 mm
        # ring in both frustums at the same orbit time. A side the frustums accept can
        # still be one the joint planner refuses (30 deg pins joint 2 from the
        # carriage-IK witness pose), so the planner's verdict also drives the backoff.
        while True:
            try:
                sides = camera_orbit_viewpoints(tip_c, r_c, n_c, poses, tilt, distance, off_axis, carriage_m,
                                                max_incidence, ring_m=ring)
            except DrawRefusal as refusal:
                if refusal.code != "orbit_view":
                    raise
                frustum_refused.append(f"{math.degrees(off_axis):.0f}")
                sides = []
            for viewpoints, side, side_scores in sides:
                if joints is None:
                    samples, plan_stats = build(viewpoints, 1.0), None
                    chosen = (side, side_scores)
                    break
                # Quick by default (operator: the first orbit felt slow); when the joint
                # planner refuses on speed from this pose, halve the tip speed and the
                # rotation rate together and try again before giving up on the side.
                for factor in (1.0, 0.5, 0.25):
                    candidate = build(viewpoints, factor)
                    try:
                        plan = dk.plan_joints(candidate, np.asarray(joints, float), carriage_m, period_s,
                                              lock_carriage_when_up=True)
                    except dk.PlanRefusal as refusal:
                        refusals.append(f"{math.degrees(off_axis):.0f} deg side {side:+.0f} x{factor:g}: {refusal}")
                        if refusal.reason not in ("joint_velocity", "carriage_velocity", "carriage_acceleration"):
                            break
                        continue
                    samples, plan_stats = candidate, plan["stats"]
                    chosen = (side, side_scores)
                    speed_factor = factor
                    break
                if samples is not None:
                    break
            if samples is not None or off_axis - OFF_AXIS_BACKOFF_RAD < OFF_AXIS_MIN_RAD - 1e-9:
                break
            off_axis -= OFF_AXIS_BACKOFF_RAD
        if samples is None:
            raise DrawRefusal(
                "orbit_view",
                f"no camera orbit holds a {ring * 1e3:.0f} mm design patch in both D405 frustums and passes the "
                f"joint planner from this contact pose (off-axis {math.degrees(off_axis_config):.0f} down to "
                f"{math.degrees(off_axis):.0f} deg; frustums refused at {', '.join(frustum_refused) or 'none'} deg; "
                "planner: " + ("; ".join(refusals) or "none") +
                ") -- lower tilt_deg, re-touch from a less wrapped wrist, or use orbit mode tip")
        side, scores = chosen
        report = {
            "kind": "orbit", "mode": "camera", "capture_count": poses,
            "camera_distance_mm": distance * 1e3, "off_axis_deg": math.degrees(off_axis),
            "off_axis_config_deg": math.degrees(off_axis_config), "patch_ring_mm": ring * 1e3,
            "tilt_deg": math.degrees(tilt), "speed_mm_s": speed * speed_factor * 1e3,
            "rotation_deg_s": math.degrees(omega_max) * speed_factor, "speed_factor": speed_factor,
            "wide_axis_side": side,
            "contact_normal_base": [float(x) for x in n_c],
            "viewpoints_base": [tip_k.tolist() for tip_k, _ in viewpoints],
            "viewpoint_scores": scores,
            "camera_distance_max_mm": max(c["distance_mm"] for s in scores for c in s["cameras"].values()),
            "camera_incidence_max_deg": max(c["incidence_deg"] for s in scores for c in s["cameras"].values()),
            "tip_height_min_mm": min(s["tip_height_mm"] for s in scores),
            "sides_refused": refusals,
            "advisory_plan": None if plan_stats is None else {
                k: float(v) for k, v in plan_stats.items() if isinstance(v, (int, float))},
            "sample_count": samples.n, "duration_s": samples.duration_s,
        }
        return samples, report
    if mode != "tip":
        raise DrawRefusal("orbit_view", f"unknown orbit mode {mode!r}; use camera or tip")

    views = orbit_viewpoints(n_c, poses, tilt)
    parts = []
    p, r = line_segment(tip_c, tip_c + standoff * n_c, r_c, r_c, speed, period_s, include_start=True)
    parts.append((p, r, 0, None))
    for k in range(poses):
        if k > 0:
            p, r = arc_segment(tip_c, standoff, views[k - 1], views[k], n_c, r_c, speed, period_s)
            parts.append((p, r, 0, None))
        position = tip_c + standoff * views[k]
        rotation = transported_rotations(views[k][None], n_c, r_c)[0]
        p, r = hold_rows(position, rotation, ORBIT_HOLD_S, period_s)
        capture = np.zeros(len(p), dtype=np.int64)
        capture[-1] = k + 1
        parts.append((p, r, 0, capture))
    samples = assemble(period_s, parts)
    report = {
        "kind": "orbit",
        "mode": "tip",
        "capture_count": poses,
        "standoff_mm": standoff * 1e3,
        "tilt_deg": math.degrees(tilt),
        "speed_mm_s": speed * 1e3,
        "contact_normal_base": [float(x) for x in n_c],
        "viewpoints_base": (tip_c[None, :] + standoff * views).tolist(),
        "sample_count": samples.n,
        "duration_s": samples.duration_s,
    }
    return samples, report


# --- preflight ------------------------------------------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def preflight(samples: Samples, surface, config: dict, tool_axis_in_link6, start_pose: dict,
              design_length_m: float | None = None) -> dict:
    """Refuse (DrawRefusal) or report, per docs/draw.md. `surface` may be None for an orbit."""
    p, v, r = samples.p, samples.v, samples.R
    if samples.n == 0:
        raise DrawRefusal("nan", "no samples")
    if not (np.isfinite(p).all() and np.isfinite(v).all() and np.isfinite(r).all()):
        raise DrawRefusal("nan", "a sample is not finite")

    tip_s, r_s = _pose(start_pose)
    start_gap = float(np.linalg.norm(p[0] - tip_s))
    start_turn = dk.rotation_angle(r[0] @ r_s.T)
    if start_gap > START_TOLERANCE_M or start_turn > START_TOLERANCE_RAD:
        raise DrawRefusal(
            "start_tolerance",
            f"row 1 is {start_gap * 1e3:.2f} mm / {start_turn:.4f} rad from the start pose")

    speeds = np.linalg.norm(v, axis=1)
    tip_speed_max = float(speeds.max())
    if tip_speed_max > TIP_SPEED_CAP_M_S:
        raise DrawRefusal("tip_speed", f"{tip_speed_max * 1e3:.2f} mm/s exceeds {TIP_SPEED_CAP_M_S * 1e3} mm/s")

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
    p_down_root = dk.root_from_base(p[down])
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
        # A hole no farther than hole_fill_mm from mapped cells is interpolated by the
        # fuse infill, and the operator has accepted that: dilate the mapped mask that
        # many cells. Default 0 keeps the strict rule (docs/draw.md). The one hole the
        # standoff orbit leaves is under the pen itself; a static capture with the pen
        # 2 mm off the surface shadows a ~7 mm patch at the contact (live 2026-09-01).
        cell_m = min(2.0 * half_w / (cols - 1), 2.0 * half_h / (rows - 1))
        grown = mapped.copy()
        for _ in range(int(math.ceil(hole_fill_m / cell_m))):
            padded = np.pad(grown, 1)
            grown = np.zeros_like(grown)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    grown |= padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
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
    budget = float(config.get("lean_budget_deg", 20.0))
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


def write_samples_csv(path, samples: Samples, kind: str, tip_in_link6, extra: dict | None = None) -> None:
    """docs/draw.md samples format: key,value header, `columns,...`, one row per tick."""
    if kind not in ("orbit", "path"):
        raise ValueError(f"kind must be orbit or path, got {kind!r}")
    tip = np.asarray(tip_in_link6, float)
    header = [
        ("schema", SAMPLES_SCHEMA), ("kind", kind), ("frame", SAMPLES_FRAME),
        ("period_s", _fmt(samples.period_s)),
        ("tip_x_m", _fmt(tip[0])), ("tip_y_m", _fmt(tip[1])), ("tip_z_m", _fmt(tip[2])),
        ("sample_count", str(samples.n)),
    ]
    if kind == "orbit":
        header.append(("capture_count", str(int(samples.capture.max()) if samples.n else 0)))
    header.append(("start_tolerance_m", _fmt(START_TOLERANCE_M)))
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
    for i in range(samples.n):
        row = [_fmt(samples.t[i])]
        row += [_fmt(x) for x in samples.p[i]]
        row += [_fmt(x) for x in samples.v[i]]
        row += [_fmt(x) for x in flat[i]]
        row += [str(int(samples.pen[i])), str(int(samples.capture[i]))]
        lines.append(",".join(row))
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


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
                    if tuple(columns) != COLUMNS:
                        raise ValueError(f"unexpected columns {columns}")
                    continue
                header[key] = _parse_scalar(value)
                continue
            rows.append([float(x) for x in line.split(",")])
    if header.get("schema") != SAMPLES_SCHEMA:
        raise ValueError(f"unknown samples schema {header.get('schema')!r}")
    if columns is None:
        raise ValueError("samples file has no columns line")
    data = np.asarray(rows, float).reshape(-1, len(COLUMNS))
    samples = Samples(
        period_s=float(header["period_s"]),
        t=data[:, 0], p=data[:, 1:4], v=data[:, 4:7], R=data[:, 7:16].reshape(-1, 3, 3),
        pen=data[:, 16].astype(np.int64), capture=data[:, 17].astype(np.int64))
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
