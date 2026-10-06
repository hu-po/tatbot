"""NumPy time laws from pen_path: one position/rotation per tick; assembled paths use feedforward."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

QUINTIC_PEAK = 1.875          # max of d/du (10u^3 - 15u^4 + 6u^5)
MIN_SEGMENT_S = 0.25
FEEDFORWARD_WINDOW_TICKS = 21  # symmetric box over the central-difference velocity (52 ms at 400 Hz)
KNOT_STEP_M = 5e-6             # arc-length knot spacing the speed ramps are integrated on (pen down)
PEN_UP_KNOT_STEP_M = 10e-6
MAX_TICKS = 250_000            # config/motion_constants.json planner.max_ticks


@dataclass(frozen=True)
class Retime:
    """Corner, pen-up, touchdown and rotary budgets; window_s averages velocity, ramp_s bounds jerk."""

    corner_accel_m_s2: float = 0.010
    pen_up_accel_m_s2: float = 0.010
    touchdown_accel_m_s2: float = 0.003
    window_s: float = FEEDFORWARD_WINDOW_TICKS * 0.0025
    ramp_s: float = 0.2
    final_m: float = 0.005
    angular_accel_rad_s2: float = 0.1


def polyline_length(poly: np.ndarray) -> float:
    poly = np.asarray(poly, float)
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()) if len(poly) > 1 else 0.0


def resample_polyline_by_arclength(poly: np.ndarray, s: np.ndarray):
    """Points and unit tangents of a polyline at arc lengths `s` (scripts/lib/stroke_material.py)."""
    poly = np.asarray(poly, float)
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s = np.clip(np.asarray(s, float), 0.0, cum[-1])
    index = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(seg_len[index] > 0.0, (s - cum[index]) / seg_len[index], 0.0)
    points = poly[index] + frac[:, None] * seg[index]
    unit = np.zeros_like(seg)
    good = seg_len > 0.0
    unit[good] = seg[good] / seg_len[good][:, None]
    return points, unit[index]


def trim_polyline(poly: np.ndarray, from_s: float) -> np.ndarray:
    """The part of a polyline from arc length `from_s` on (its first row is the point at from_s)."""
    poly = np.asarray(poly, float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    if from_s <= 0.0:
        return poly
    start, _ = resample_polyline_by_arclength(poly, np.array([from_s]))
    return np.concatenate([start, poly[cum > from_s + 1e-12]])


# --- rotations ---------------------------------------------------------------------------------

def axis_rotation(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis (the C++ closed form)."""
    x, y, z = (float(v) for v in axis)
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    return np.array([[x * x * k + c, x * y * k - z * s, x * z * k + y * s],
                     [y * x * k + z * s, y * y * k + c, y * z * k - x * s],
                     [z * x * k - y * s, z * y * k + x * s, z * z * k + c]])


def rotation_angle(rotation: np.ndarray) -> float:
    return math.acos(max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5)))


def rotation_log(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """(unit axis, angle) of a rotation; the axis is arbitrary at angle 0."""
    angle = rotation_angle(rotation)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    if math.pi - angle < 1e-6:
        sym = rotation + np.eye(3)
        col = int(np.argmax(np.linalg.norm(sym, axis=0)))
        return sym[:, col] / np.linalg.norm(sym[:, col]), angle
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]]) / (2.0 * math.sin(angle))
    return axis, angle


def align_rotation(a, b) -> np.ndarray:
    """Minimal rotation carrying unit vector a onto b (a deterministic half turn when antiparallel)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s2 = float(v @ v)
    if s2 < 1e-24:
        if c > 0.0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        return axis_rotation(axis / np.linalg.norm(axis), math.pi)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1.0 - c) / s2)


def rotation_slerp(r0: np.ndarray, r1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Rotations from r0 to r1 along the geodesic at fractions `blend`."""
    r0 = np.asarray(r0, float)
    axis, angle = rotation_log(np.asarray(r1, float) @ r0.T)
    return np.stack([axis_rotation(axis, float(b) * angle) @ r0 for b in np.atleast_1d(blend)])


# --- the stroke time law -----------------------------------------------------------------------

def _quintic(u):
    u = np.asarray(u, float)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _distance_blend(u):
    u = np.asarray(u, float)
    u4 = u ** 4
    return 2.5 * u4 - 3.0 * u4 * u + u4 * u * u


def time_law(path_length_m: float, duration_s: float, ease_s: float, period_s: float):
    """Quintic ease-in / cruise / ease-out over arc length: (t, s, sdot), t = k*period for k = 1..ticks."""
    if not (math.isfinite(duration_s) and duration_s > 0.0 and math.isfinite(ease_s) and ease_s > 0.0
            and ease_s * 2.0 < duration_s and math.isfinite(period_s) and period_s > 0.0):
        raise ValueError("time law needs 0 < 2*ease < duration and a positive period")
    if not (math.isfinite(path_length_m) and path_length_m > 0.0):
        raise ValueError("path length must be positive")
    ticks = int(math.ceil(duration_s / period_s))
    if ticks > MAX_TICKS:
        raise ValueError(f"{ticks} ticks is outside the planner's range")
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
    return t, np.clip(s, 0.0, path_length_m), sdot


# --- corner-aware retiming ----------------------------------------------------------------------

def polyline_turn_angles(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(arc length at every vertex, turn angle at every vertex; 0 at the ends). Zero-length chords are
    skipped, so a repeated vertex neither hides a corner nor invents one."""
    poly = np.asarray(poly, float)
    seg = np.diff(poly, axis=0)
    length = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(length)])
    turn = np.zeros(len(poly))
    moving = np.flatnonzero(length > 0.0)
    if len(moving) >= 2:
        unit = seg[moving] / length[moving][:, None]
        cos = np.einsum("ij,ij->i", unit[:-1], unit[1:])
        turn[moving[1:]] = np.arccos(np.clip(cos, -1.0, 1.0))
    return arc, turn


def corner_speed_limits(arc: np.ndarray, turn: np.ndarray, cruise: float, accel: float, window_s: float,
                        iterations: int = 40) -> np.ndarray:
    """Tip speed cap at every vertex so the direction change within the feedforward window costs at most
    `accel`: v * Theta(s, v * window) = accel * window, Theta summing the turns within +-v*window/2.
    An isolated corner gets accel*window/theta; a dense arc of tiny turns about sqrt(accel/curvature)."""
    arc = np.asarray(arc, float)
    cumulative = np.concatenate([[0.0], np.cumsum(np.asarray(turn, float))])
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
    lo, hi = np.zeros(len(arc)), caps.copy()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        over = excess(mid) > 0.0
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
    caps[limited] = lo[limited]
    return caps


def _refine_knots(arc: np.ndarray, values: np.ndarray, max_step: float, fill: float):
    """Insert knots so no arc-length gap exceeds `max_step`; inserted knots take `fill`."""
    gaps = np.diff(arc)
    counts = np.maximum(1, np.ceil(gaps / max_step - 1e-12).astype(int))
    if not (counts > 1).any():
        return np.asarray(arc, float).copy(), np.asarray(values, float).copy()
    pieces_s, pieces_v = [arc[:1]], [values[:1]]
    for i in range(len(gaps)):
        if counts[i] > 1:
            inner = arc[i] + gaps[i] * np.arange(1, counts[i]) / counts[i]
            pieces_s.append(inner)
            pieces_v.append(np.full(len(inner), fill))
        pieces_s.append(arc[i + 1:i + 2])
        pieces_v.append(values[i + 1:i + 2])
    return np.concatenate(pieces_s), np.concatenate(pieces_v)


def _ramp_pass(arc, caps, accel, jerk, v_start):
    """Forward pass: the fastest speed at each knot reachable from the one before, S-curved by `jerk`."""
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


def accel_limited_speeds(arc, caps, accel, v_start=None, v_end=None, jerk=None) -> np.ndarray:
    """Largest speed profile under per-knot caps with tangential acceleration within `accel` (scalar or per
    interval): forward/backward v^2 = v0^2 + 2 a ds passes, S-curved by `jerk` when given."""
    arc = np.asarray(arc, float)
    caps = np.asarray(caps, float)
    ds = np.diff(arc)
    a = np.broadcast_to(np.asarray(accel, float), ds.shape)
    j = None if jerk is None else np.broadcast_to(np.asarray(jerk, float), ds.shape)
    forward = _ramp_pass(arc, caps, a, j, v_start)
    backward = _ramp_pass(arc[-1] - arc[::-1], caps[::-1], a[::-1], None if j is None else j[::-1], v_end)[::-1]
    return np.minimum(forward, backward)


def rotation_rates(rotations: np.ndarray, arc: np.ndarray) -> np.ndarray:
    """Rotation-vector rate (rad/m, 3-vector) of each knot interval, small-angle (pen_path)."""
    r = np.asarray(rotations, float)
    step = np.einsum("nji,njk->nik", r[:-1], r[1:])   # R_i^T R_{i+1}, in the tool frame
    vector = 0.5 * np.stack([step[:, 2, 1] - step[:, 1, 2], step[:, 0, 2] - step[:, 2, 0],
                             step[:, 1, 0] - step[:, 0, 1]], axis=1)
    return vector / np.maximum(np.diff(np.asarray(arc, float)), 1e-300)[:, None]


def rotary_speed_limits(arc, rates, cruise: float, alpha: float, window_s: float, iterations: int = 40):
    """Speed cap at every knot so the change of angular velocity across the feedforward window costs at
    most `alpha`: v * |k(s + v w / 2) - k(s - v w / 2)| = alpha * w, k from `rotation_rates`."""
    arc = np.asarray(arc, float)
    mids = 0.5 * (arc[:-1] + arc[1:])
    rates = np.asarray(rates, float)
    budget = float(alpha) * float(window_s)

    def rate_at(s):
        return np.stack([np.interp(s, mids, rates[:, k]) for k in range(3)], axis=1)

    def excess(v):
        half = 0.5 * v * float(window_s)
        return v * np.linalg.norm(rate_at(arc + half) - rate_at(arc - half), axis=1) - budget

    caps = np.full(len(arc), float(cruise))
    limited = excess(caps) > 0.0
    if not limited.any():
        return caps
    lo, hi = np.zeros(len(arc)), caps.copy()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        over = excess(mid) > 0.0
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
    caps[limited] = lo[limited]
    return caps


def budgeted_speed_profile(knots, caps, accel, retime: Retime, v_start=None, v_end=None, *,
                           rotations=None, cruise=None):
    """Speeds at `knots` under the caps and the linear budget, with S-curve ramps. With the knot
    `rotations`, the rotary budget `retime.angular_accel_rad_s2` also caps the speed where the angular
    velocity would change too fast and holds the ramps to alpha / k, k the turn in rad per metre. A ramp
    cut short by the opposite pass (a triangle) is flattened to a plateau at its peak so both S-curves
    decay into it."""
    caps = np.asarray(caps, float).copy()
    accel = np.broadcast_to(np.asarray(accel, float), (len(knots) - 1,)).copy()
    alpha = retime.angular_accel_rad_s2
    if rotations is not None and alpha > 0.0 and len(knots) > 2:
        rates = rotation_rates(rotations, knots)
        caps = np.minimum(caps, rotary_speed_limits(knots, rates, cruise, alpha, retime.window_s))
        accel = np.minimum(accel, alpha / np.maximum(np.linalg.norm(rates, axis=1), 1e-9))
    jerk = accel / retime.ramp_s
    speeds = accel_limited_speeds(knots, caps, accel, v_start=v_start, v_end=v_end, jerk=jerk)
    for _ in range(8):
        below = speeds < caps * (1.0 - 1e-9)
        edges = np.flatnonzero(np.diff(np.concatenate([[False], below, [False]]).astype(int)))
        lowered = False
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            region = speeds[start:stop]
            top = int(np.argmax(region))
            peak = float(region[top])
            if 0 < top < len(region) - 1 and region[0] < peak * (1 - 1e-6) and region[-1] < peak * (1 - 1e-6):
                caps[start:stop] = np.minimum(caps[start:stop], peak)
                lowered = True
        if not lowered:
            break
        speeds = accel_limited_speeds(knots, caps, accel, v_start=v_start, v_end=v_end, jerk=jerk)
    return speeds


def hermite_arclength(t_knots, s_knots, v_knots, t):
    """Cubic Hermite s(t), ds/dt through (t, s, v) knots; C1, clipped to the knot range."""
    t_knots = np.asarray(t_knots, float)
    t = np.clip(np.asarray(t, float), t_knots[0], t_knots[-1])
    index = np.clip(np.searchsorted(t_knots, t, side="right") - 1, 0, len(t_knots) - 2)
    h = t_knots[index + 1] - t_knots[index]
    safe = np.where(h > 0.0, h, 1.0)
    u = np.where(h > 0.0, (t - t_knots[index]) / safe, 0.0)
    u2, u3 = u * u, u * u * u
    s0, s1 = np.asarray(s_knots)[index], np.asarray(s_knots)[index + 1]
    v0, v1 = np.asarray(v_knots)[index] * h, np.asarray(v_knots)[index + 1] * h
    s = (2 * u3 - 3 * u2 + 1) * s0 + (u3 - 2 * u2 + u) * v0 + (-2 * u3 + 3 * u2) * s1 + (u3 - u2) * v1
    ds = (6 * u2 - 6 * u) * s0 + (3 * u2 - 4 * u + 1) * v0 + (-6 * u2 + 6 * u) * s1 + (3 * u2 - 2 * u) * v1
    return s, np.where(h > 0.0, ds / safe, 0.0)


def corner_time_law(poly: np.ndarray, cruise: float, ease_s: float, period_s: float, retime: Retime):
    """The stroke law (quintic ease / cruise / ease over arc length) dilated at corners.

    Returns (t, s, sdot, info) at t = k*period like `time_law`. Where the polyline turns, the speed is held
    to `corner_speed_limits` with ramps at `retime.corner_accel_m_s2`; elsewhere the plain law comes back
    unchanged. Every s lies on the same polyline; `info["added_s"]` is what the corners cost."""
    poly = np.asarray(poly, float)
    arc, turn = polyline_turn_angles(poly)
    length = float(arc[-1])
    duration = max(length / cruise + ease_s, 2.0 * ease_s + period_s)
    t, s, sdot = time_law(length, duration, ease_s, period_s)
    info = {"duration_s": float(t[-1]), "added_s": 0.0, "limited_vertices": 0}
    caps = corner_speed_limits(arc, turn, cruise, retime.corner_accel_m_s2, retime.window_s)
    knots, knot_caps = _refine_knots(arc, caps, KNOT_STEP_M, cruise)
    knot_speed = budgeted_speed_profile(knots, knot_caps, retime.corner_accel_m_s2, retime)
    if not (knot_speed < cruise * (1.0 - 1e-9)).any():
        return t, s, sdot, info
    # Dilate the plain law: dt' = (cruise / v_limit(s)) dt on the union of its tick positions and the
    # ramp knots; where no corner reaches, the plain ticks come back unchanged.
    s_plain = np.concatenate([[0.0], s])
    t_plain = np.concatenate([[0.0], t])
    grid = np.unique(np.concatenate([s_plain, np.clip(knots, 0.0, length)]))
    grid = grid[np.concatenate([[True], np.diff(grid) > 1e-9])]
    t_grid = np.interp(grid, s_plain, t_plain)
    dilation = cruise / np.interp(grid, knots, knot_speed)
    t_dilated = np.concatenate([[0.0], np.cumsum(np.diff(t_grid) * 0.5 * (dilation[:-1] + dilation[1:]))])
    secant = np.diff(grid) / np.maximum(np.diff(t_dilated), 1e-300)
    v_dilated = np.concatenate([[0.0], 0.5 * (secant[:-1] + secant[1:]), [0.0]])
    total = float(t_dilated[-1])
    ticks = int(math.ceil(total / period_s - 1e-9))
    if ticks > MAX_TICKS:
        raise ValueError(f"retimed stroke needs {ticks} ticks, outside the planner's range")
    t_new = np.minimum(total, np.arange(1, ticks + 1, dtype=float) * period_s)
    s_new, sdot_new = hermite_arclength(t_dilated, grid, v_dilated, t_new)
    s_new = np.clip(np.maximum.accumulate(s_new), 0.0, length)
    s_new[-1] = length
    info.update(duration_s=total, added_s=total - float(t[-1]), limited_vertices=int((caps < cruise).sum()))
    return t_new, s_new, np.clip(sdot_new, 0.0, cruise), info


# --- segments ----------------------------------------------------------------------------------

def _segment_ticks(distance: float, speed_max: float, period_s: float) -> int:
    return max(1, int(math.ceil(max(MIN_SEGMENT_S, QUINTIC_PEAK * distance / speed_max) / period_s)))


def line_segment(p0, p1, r0, r1, speed_max: float, period_s: float, include_start: bool = False,
                 omega_max: float | None = None):
    """Quintic straight move p0 -> p1 with rotation slerp r0 -> r1, rows at u = k/K; `omega_max` (rad/s)
    also bounds the rotation rate."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ticks = _segment_ticks(float(np.linalg.norm(p1 - p0)), speed_max, period_s)
    if omega_max:
        angle = rotation_angle(np.asarray(r1, float) @ np.asarray(r0, float).T)
        ticks = max(ticks, _segment_ticks(angle, float(omega_max), period_s))
    blend = _quintic(np.arange(0 if include_start else 1, ticks + 1, dtype=float) / ticks)
    return p0[None, :] + blend[:, None] * (p1 - p0)[None, :], rotation_slerp(r0, r1, blend)


def hold_rows(p, r, duration_s: float, period_s: float):
    ticks = max(1, int(round(duration_s / period_s)))
    return np.repeat(np.asarray(p, float)[None, :], ticks, axis=0), np.repeat(np.asarray(r, float)[None], ticks, axis=0)


def feedforward(p: np.ndarray, period_s: float, window_ticks: int = FEEDFORWARD_WINDOW_TICKS) -> np.ndarray:
    """Tip velocity: central differences of p, box-smoothed over `window_ticks` (centred, so no lag). A
    polyline path is C0; the window keeps the feedforward continuous and the loop's position gain
    absorbs the few-micron mismatch that remains."""
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


def pen_up_leg(points, r_from, r_to, speed_caps, retime: Retime, period_s: float, *,
               gentle_start: bool = False, gentle_end: bool = False, include_start: bool = False,
               omega_max: float | None = None):
    """One continuous pen-up motion along the polyline `points` (M >= 2 rows), from rest to rest.

    `speed_caps` (M-1) bound each chord; corners get the `corner_speed_limits` cap for the pen-up budget;
    the profile accelerates, cruises and decelerates at `retime.pen_up_accel_m_s2` (S-curved, under the
    rotary budget of `budgeted_speed_profile`) except
    within `retime.final_m` of a gentle end, where the touchdown budget applies. The rotation slerps
    r_from -> r_to over the first chord as a quintic of distance and holds r_to after it; `omega_max`
    bounds that chord's speed. Returns (p, R, chord index per row); every row is a point of the polyline."""
    points = np.asarray(points, float)
    caps = np.asarray(speed_caps, float)
    length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = length > 0.0
    if not keep.any():
        raise ValueError("pen-up leg has no length")
    r_from, r_to = np.asarray(r_from, float), np.asarray(r_to, float)
    angle = 2.0 * math.asin(min(1.0, float(np.linalg.norm(r_to - r_from)) / math.sqrt(8.0)))
    if angle > 1e-9 and not keep[0]:
        raise ValueError("pen-up leg cannot turn in place: its first chord has no length")
    chord_index = np.flatnonzero(keep)
    vertices = np.concatenate([points[:1], points[1:][keep]])
    chord_len = length[keep]
    chord_caps = caps[keep].copy()
    if omega_max and angle > 1e-9:
        chord_caps[0] = min(chord_caps[0], float(omega_max) * chord_len[0] / (QUINTIC_PEAK * angle))
    arc, turn = polyline_turn_angles(vertices)
    cruise = float(chord_caps.max())
    vertex_caps = corner_speed_limits(arc, turn, cruise, retime.pen_up_accel_m_s2, retime.window_s)
    vertex_caps[:-1] = np.minimum(vertex_caps[:-1], chord_caps)
    vertex_caps[1:] = np.minimum(vertex_caps[1:], chord_caps)
    knots, knot_caps = _refine_knots(arc, vertex_caps, PEN_UP_KNOT_STEP_M, math.inf)
    chord_of_knot = np.clip(np.searchsorted(arc, knots, side="right") - 1, 0, len(chord_len) - 1)
    knot_caps = np.minimum(knot_caps, chord_caps[chord_of_knot])
    mids = 0.5 * (knots[:-1] + knots[1:])
    # the touchdown budget over the gentle final_m, blended into the pen-up budget over the next final_m
    accel = np.full(len(mids), retime.pen_up_accel_m_s2)
    span = retime.pen_up_accel_m_s2 - retime.touchdown_accel_m_s2
    final = retime.final_m
    if gentle_start:
        accel = np.minimum(accel, retime.touchdown_accel_m_s2 + span * np.clip((mids - final) / final, 0.0, 1.0))
    if gentle_end:
        accel = np.minimum(accel, retime.touchdown_accel_m_s2
                           + span * np.clip((arc[-1] - final - mids) / final, 0.0, 1.0))

    def rotation_at(s):
        if angle <= 1e-9:
            return np.repeat(r_to[None], len(s), axis=0)
        return rotation_slerp(r_from, r_to, _quintic(np.clip(s / chord_len[0], 0.0, 1.0)))

    speed = budgeted_speed_profile(knots, knot_caps, accel, retime, v_start=0.0, v_end=0.0,
                                   rotations=rotation_at(knots), cruise=cruise)
    pair = speed[:-1] + speed[1:]
    if not np.all(pair > 0.0):
        raise ValueError("pen-up profile stalls")
    t_knots = np.concatenate([[0.0], np.cumsum(2.0 * np.diff(knots) / pair)])
    total = float(t_knots[-1])
    ticks = int(math.ceil(total / period_s - 1e-9))
    if ticks > MAX_TICKS:
        raise ValueError(f"pen-up leg needs {ticks} ticks, outside the planner's range")
    t = np.minimum(np.arange(0 if include_start else 1, ticks + 1, dtype=float) * period_s, total)
    s, _ = hermite_arclength(t_knots, knots, speed, t)
    s = np.clip(np.maximum.accumulate(s), 0.0, arc[-1])
    s[-1] = arc[-1]
    p, _ = resample_polyline_by_arclength(vertices, s)
    p[-1] = vertices[-1]
    row_chord = np.clip(np.searchsorted(arc, s, side="right") - 1, 0, len(chord_len) - 1)
    return p, rotation_at(s), chord_index[row_chord]
