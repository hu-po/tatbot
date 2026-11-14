"""Program operations through control-rate time laws and CLIK to JTC trajectories; ros/README.md §4.4."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from tatbot_motion import timelaw as tl
from tatbot_motion.clik import Kinematics, PlanError, clik, depth_sensitivity

# Draw feedback PHASE_* (tatbot_interfaces/action/Draw.action); no ROS import here.
PHASE_TRAVEL, PHASE_DESCEND, PHASE_SETTLE, PHASE_DRAW, PHASE_LIFT, PHASE_TOUCH = 2, 3, 4, 5, 6, 7
_GROUP_TRAVEL, _GROUP_DRAW, _GROUP_LIFT = "travel", "draw", "lift"


@dataclass
class Trajectory:
    """One FollowJointTrajectory goal's worth of motion."""

    joint_names: tuple[str, ...]
    t: np.ndarray       # (M,) s from the goal start, strictly increasing, t[0] = 0 at the seed
    q: np.ndarray       # (M, n) joints in controller order: rad, the WXAI's carriage m
    qd: np.ndarray      # (M, n) rad/s, carriage m/s (the velocity feed-forward); qd[-1] = 0
    tip: np.ndarray     # (M, 3) planned TCP position (FK of q) in <arm>/base_link, m
    arc_m: np.ndarray   # (M,) stroke arc position, m from the stroke start; NaN when not drawing
    phase: np.ndarray   # (M,) uint8 Draw feedback PHASE_*
    info: dict = field(default_factory=dict)  # duration, model error, peak joint speed, slow-downs, trim
    axis: np.ndarray | None = None    # (M, 3) unit depth axis in the base: the pen's height over the surface grows along it
    dq_dh: np.ndarray | None = None   # (M, n) joints per metre along axis (clik.depth_sensitivity); trim.compose adds a trim

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])


@dataclass
class _Leg:
    p: np.ndarray
    rot: np.ndarray
    pen: bool
    phase: np.ndarray | int
    group: str
    arc: np.ndarray | None = None
    carriage_v: np.ndarray | None = None
    axis: np.ndarray | None = None   # (3,) or (n, 3): the depth axis a trim moves the pen along


def _cfg(motion: dict, *, lift: bool = False):
    """(the control period, the legs' budgets). The gentle stretch is a descent's last approach.final_m, or with
    `lift` a lift's first approach.lift_first_m."""
    dt = 1.0 / float(motion["control_rate_hz"])
    corners, approach = motion["corners"], motion["approach"]
    retime = tl.Retime(corner_accel_m_s2=corners["velocity_change_m_s2"], pen_up_accel_m_s2=corners["pen_up_m_s2"],
                       touchdown_accel_m_s2=corners["touchdown_m_s2"], window_s=tl.FEEDFORWARD_WINDOW_TICKS * dt,
                       ramp_s=corners["s_curve_s"], final_m=approach["lift_first_m" if lift else "final_m"],
                       angular_accel_rad_s2=float(approach["angular_accel_rad_s2"]))
    return dt, retime


def _rows(leg_p, n):
    return np.full(n, leg_p) if np.isscalar(leg_p) else np.asarray(leg_p)


def _solve(build, q_seed, kin: Kinematics, motion: dict, *, check=lambda rows: None) -> Trajectory:
    """Solve control-rate legs, slowing groups to their joint caps; check before knot resampling."""
    dt, _ = _cfg(motion)
    q_seed = np.asarray(q_seed, float)
    seed = kin.fk(q_seed)
    caps = motion["joint_speed"]
    clik_cfg = motion["clik"]
    scales: dict[str, float] = {}
    for _attempt in range(int(clik_cfg["slowdown_tries"]) + 1):
        legs = build(scales)
        first = legs[0]
        p = np.concatenate([seed[None, :3, 3]] + [leg.p for leg in legs])
        rot = np.concatenate([seed[None, :3, :3]] + [leg.rot for leg in legs])
        pen = np.concatenate([[False]] + [np.full(len(leg.p), leg.pen) for leg in legs])
        phase = np.concatenate([[_rows(first.phase, 1)[0]]] + [_rows(leg.phase, len(leg.p)) for leg in legs])
        arc = np.concatenate([[first.arc[0] if first.arc is not None else math.nan]]
                             + [leg.arc if leg.arc is not None else np.full(len(leg.p), math.nan) for leg in legs])
        cv = np.concatenate([[math.nan]] + [leg.carriage_v if leg.carriage_v is not None
                                            else np.full(len(leg.p), math.nan) for leg in legs])
        group = np.concatenate([[first.group]] + [np.full(len(leg.p), leg.group) for leg in legs])
        axis = None if any(leg.axis is None for leg in legs) else np.concatenate(
            [np.broadcast_to(first.axis, (len(first.p), 3))[:1]] + [np.broadcast_to(leg.axis, (len(leg.p), 3))
                                                                     for leg in legs])
        result = clik(kin, q_seed, p, tl.feedforward(p, dt), rot, pen, dt, clik_cfg, motion["carriage"],
                      carriage_v=cv, carriage_ik=bool(motion["carriage"]["ik"]))
        speed = np.abs(result.qd[:, kin.revolute]).max(axis=1)
        ratio = speed / np.where(pen, caps["pen_down_rad_s"], caps["pen_up_rad_s"])
        over = {g: float(ratio[group == g].max()) for g in set(group.tolist()) if ratio[group == g].max() > 1.0}
        if not over:
            break
        for g, r in over.items():
            scales[g] = scales.get(g, 1.0) * 0.95 / r
    else:
        raise PlanError(f"joint speed stays over its cap after {clik_cfg['slowdown_tries']} slow-downs: {over}")
    worst = float(result.model_error_m.max())
    if worst > float(clik_cfg["max_model_error_m"]):
        row = int(np.argmax(result.model_error_m))
        raise PlanError(f"the tip trails its reference by {worst * 1e3:.2f} mm at t={row * dt:.2f} s")
    worst_rot = float(result.orientation_error_rad.max())
    if worst_rot > float(clik_cfg["max_orientation_error_rad"]):
        row = int(np.argmax(result.orientation_error_rad))
        raise PlanError(f"the tool axis lags its reference by {worst_rot:.4f} rad at t={row * dt:.2f} s")
    qd = result.qd.copy()
    qd[:-1] = 0.5 * (result.qd[:-1] + result.qd[1:])  # the knot's velocity, not the tick's
    qd[-1] = 0.0
    traj = Trajectory(tuple(kin.joint_names), np.arange(len(p)) * dt, result.q, qd, result.tip, arc,
                      phase.astype(np.uint8), axis=axis)
    traj.info = {"duration_s": traj.duration_s, "samples": len(p), "max_model_error_m": worst,
                 "max_orientation_error_rad": worst_rot,
                 "max_joint_speed_rad_s": float(speed.max()), "slowdowns": dict(scales)}
    check(result.q)  # specialized envelopes check every control tick before knot resampling
    knots = to_knots(traj, float(motion["knot_rate_hz"]))
    if knots.axis is not None:
        knots.dq_dh = depth_sensitivity(kin, knots.q, knots.axis)
    return knots


# --- legs ----------------------------------------------------------------------------------------

def _carriage_at(kin: Kinematics, q) -> float | None:
    """Where the arm's carriage stands in joints q, or None for an arm without one."""
    return None if kin.carriage is None else float(np.asarray(q)[kin.carriage])


def _carriage_restore(c0: float | None, motion: dict, dt: float, rows: int):
    """(extra hold rows, carriage velocity per row) that bring the carriage from c0 to its bias over at
    least `rows` ticks at no more than carriage.travel_m_s (quintic); none for an arm without one (c0 None)."""
    if c0 is None:
        return 0, None
    carriage = motion["carriage"]
    delta = float(carriage["bias_m"]) - float(c0)
    if abs(delta) < 1e-6:
        return 0, None
    need = int(math.ceil(tl.QUINTIC_PEAK * abs(delta) / float(carriage["travel_m_s"]) / dt))
    total = max(rows, need)
    c = c0 + delta * tl._quintic(np.arange(total + 1) / total)
    return total - rows, np.diff(c) / dt


def _travel_legs(p_from, r_from, points, r_to, caps, scale, motion, c0, *, gentle_end, phases):
    """A pen-up leg along `points` (first chord scaled), turning in place first when the first chord
    has no length; the carriage returns to its bias over the first chord."""
    dt, retime = _cfg(motion)
    approach = motion["approach"]
    omega = float(approach["omega_max_rad_s"]) * scale
    legs = []
    p_from = np.asarray(p_from, float)
    points = [np.asarray(x, float) for x in points]
    caps = list(caps)
    caps[0] *= scale
    if np.linalg.norm(points[0] - p_from) <= 1e-9 and tl.rotation_angle(r_to @ np.asarray(r_from).T) > 1e-9:
        p, r = tl.line_segment(p_from, p_from, r_from, r_to, caps[0], dt, omega_max=omega)
        legs.append(_Leg(p, r, False, phases[0], _GROUP_TRAVEL))
        r_from = r_to
    polyline = np.stack([p_from, *points])
    if np.linalg.norm(np.diff(polyline, axis=0), axis=1).max() > 1e-9:
        p, r, chord = tl.pen_up_leg(polyline, r_from, r_to, caps, retime, dt, gentle_end=gentle_end, omega_max=omega)
        legs.append(_Leg(p, r, False, np.asarray(phases)[chord], _GROUP_TRAVEL))
    elif not legs:
        p, r = tl.hold_rows(p_from, r_to, 0.1, dt)
        legs.append(_Leg(p, r, False, phases[0], _GROUP_TRAVEL))
    travel_rows = sum(int(np.count_nonzero(_rows(leg.phase, len(leg.p)) == phases[0])) for leg in legs)
    pad, cv = _carriage_restore(c0, motion, dt, max(travel_rows, 1))
    if cv is not None:
        if pad:
            hp, hr = tl.hold_rows(p_from, legs[0].rot[0], pad * dt, dt)
            legs.insert(0, _Leg(hp[:pad], hr[:pad], False, phases[0], _GROUP_TRAVEL))
        flat = np.zeros(sum(len(leg.p) for leg in legs))
        flat[:len(cv)] = cv
        for leg, part in zip(legs, np.split(flat, np.cumsum([len(leg.p) for leg in legs])[:-1]), strict=True):
            leg.carriage_v = part
    return legs


def _lift_leg(p_from, rot, to, motion, scale, phase=PHASE_LIFT) -> _Leg:
    dt, retime = _cfg(motion, lift=True)
    p, r, _ = tl.pen_up_leg(np.stack([p_from, to]), rot, rot, [motion["approach"]["lift_m_s"] * scale], retime, dt,
                            gentle_start=True)
    return _Leg(p, r, False, phase, _GROUP_LIFT)


def pen_rotation(base_from_page: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
    """The drawing rotation: tcp +z along -page z (into the paper), spun as little as possible from r_ref
    (pen_path's R = align(n_c -> n) @ R_c on a flat page)."""
    return tl.align_rotation(np.asarray(r_ref)[:, 2], -np.asarray(base_from_page)[:3, 2]) @ np.asarray(r_ref)


PRESS, RIDE, HOVER = "press", "ride", "hover"


@dataclass(frozen=True)
class PenDown:
    """How the pen-down path meets the touched page (motion.yaml `pen`, ros/README.md 4.4)."""

    mode: str                       # PRESS, RIDE or HOVER
    height_m: float                 # the path over the touched page along its normal; negative presses into it
    settle_s: float                 # the dwell at touchdown before drawing
    stroke_m: float | None = None   # ride: the fitted tool's stroke
    reach_m: float | None = None    # ride, a needle cartridge: how far its needles pass its tube's end at the bottom
    fade_m: float | None = None     # the resource's tool's fade_lift_mm: deposition 0 rides this much higher

    @property
    def machine(self) -> bool:
        """The tattoo machine runs while this pen draws."""
        return self.mode == RIDE

    def describe(self) -> str:
        where = f"{abs(self.height_m) * 1e3:.2f} mm {'into' if self.height_m < 0 else 'over'} the touched page"
        if self.machine:
            needles = ("" if self.reach_m is None else
                       f", the needles {(self.reach_m - self.height_m) * 1e3:.2f} mm past the page at its bottom")
            return (f"{self.mode}: the path {where}, {self.height_m / self.stroke_m:.2f} of the "
                    f"{self.stroke_m * 1e3:.1f} mm stroke{needles}, the tattoo machine running")
        return f"{self.mode}: the path {where}, the tattoo machine off"


def pen_down(motion: dict, tool=None) -> PenDown:
    """The pen of motion.yaml pen.mode for the fitted `tool` (a tool_spec.ToolSpec; only ride reads it).

    hover: the machine stays off and the path holds pen.hover.height_m over the page, touching nothing: the
    whole drawing pipeline (page, tracking, dips, tool changes) rehearsed without a mark.

    A page touch trips only once the tip is pushed up to the top of its stroke, where it stops, so the
    touched page is where the tip meets the paper there. press: the machine stays off, and the path
    presses pen.press.lift_m into the page while the arm's and mount's give holds the stopped tip on the
    paper. ride: the machine runs, and the path rides pen.ride.fraction of the tool's stroke over the
    page, so the tip meets the paper that far down each stroke. A tip inside its tube at the top of its
    stroke would have the touch meet the tube instead: that tool cannot ride, unless the tube's end is its contact
    reference (a needle cartridge): the gauge then measures the tube's end, the path is that end's height, and the
    needles reach the datasheet's needle_reach_mm past it at the bottom of each stroke. The fraction in use is the
    WXAI follower's 3RL's (motion.yaml pen.ride): tune it on ink."""
    mode = motion["pen"]["mode"]
    cfg = motion["pen"][mode]
    if mode == PRESS:
        return PenDown(PRESS, float(cfg["lift_m"]), float(cfg["settle_s"]))
    if mode == HOVER:
        return PenDown(HOVER, float(cfg["height_m"]), float(cfg["settle_s"]))
    name = getattr(tool, "tool_id", "the fitted tool")
    raw = getattr(tool, "raw", None) or {}
    tube = (raw.get("calibration") or {}).get("contact_reference") == "tube_end"
    stroke, out = getattr(tool, "stroke_m", None), getattr(tool, "tip_out_at_top_m", None)
    if stroke is None or (out is None and not tube):
        raise ValueError(f"pen.mode ride needs {name}'s stroke_mm and tip_out_at_top_mm, and its datasheet lacks one")
    if not tube and out <= 0.0:
        raise ValueError(f"{name}'s tip sits inside its tube at the top of its stroke: a touch meets the tube, so "
                         "the ride height over the touched page is unknown")
    reach = raw.get("needle_reach_mm") if tube else None
    return PenDown(RIDE, float(cfg["fraction"]) * stroke, float(cfg["settle_s"]), stroke,
                   None if reach is None else float(reach) / 1000.0)


def deposition_lift(op: dict, fade_m: float | None) -> np.ndarray | float:
    """How much higher than the pen's path each of a stroke's points rides: (1 - its deposition) of its tool's fade lift
    (its resource's fade_lift_mm), so a whip's ramp to 0 lifts its mark out; 0 without deposition. Only ever up: a fade
    adds no depth the plan, the trim and its cap did not already allow."""
    if "deposition" not in op:
        return 0.0
    if fade_m is None:
        raise ValueError(f"stroke {op.get('id', '?')} fades, and its tool records no fade_lift_mm")
    return float(fade_m) * (1.0 - np.clip(np.asarray(op["deposition"], float), 0.0, 1.0))


def _stroke_xy(op: dict, fade_m: float | None, blend_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A stroke's print points, closed back to its start when it is closed, and each one's deposition lift; with
    `blend_m` its corners rounded (blend_corners)."""
    pts = np.asarray(op["points_m"], float).reshape(-1, 2)
    lift = np.broadcast_to(deposition_lift(op, fade_m), (len(pts),))
    if op.get("closed") and len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts, lift = np.vstack([pts, pts[:1]]), np.r_[lift, lift[:1]]
    if blend_m > 0.0 and len(pts) > 2:
        arc = tl.arc_lengths(pts)
        pts, at = blend_corners(pts, blend_m)
        lift = np.interp(at, arc, lift)
    return pts, lift


BLEND_TURN_RAD = math.radians(5.0)   # blend_corners leaves a gentler turn alone, and steps its arcs this finely


def blend_corners(points: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    """A polyline with each corner cut by an arc of `radius_m` tangent to its two chords, and the arc length along
    the original polyline each returned point stands for (its ends keep theirs). The arc leaves each chord no
    farther than a third of it from the corner, so neighbouring corners keep a straight piece between them, and
    it never passes farther than `radius_m` from the corner it cuts: on short chords and sharp turns the arc is
    smaller. A turn under BLEND_TURN_RAD is left as it is: a curve sampled finely is already round."""
    points = np.asarray(points, float)
    arc = tl.arc_lengths(points)
    _, turn = tl.polyline_turn_angles(points)
    out, at = [points[0]], [0.0]
    for k in range(1, len(points) - 1):
        a, b = points[k] - points[k - 1], points[k + 1] - points[k]
        la, lb, theta = float(np.linalg.norm(a)), float(np.linalg.norm(b)), float(turn[k])
        if theta < BLEND_TURN_RAD or la <= 0.0 or lb <= 0.0:
            out.append(points[k])
            at.append(arc[k])
            continue
        half = 0.5 * theta
        # the tangent length: the radius's, within a third of each chord, the arc's bulge within the radius
        reach = min(radius_m * math.tan(half), la / 3.0, lb / 3.0,
                    radius_m * math.sin(half) / max(1.0 - math.cos(half), 1e-12))
        radius = reach / math.tan(half)
        ua, ub = a / la, b / lb
        start = points[k] - reach * ua
        side = 1.0 if ua[0] * ub[1] - ua[1] * ub[0] > 0.0 else -1.0
        centre = start + side * radius * np.array([-ua[1], ua[0]])
        steps = max(2, int(math.ceil(theta / BLEND_TURN_RAD)))
        angle0 = math.atan2(start[1] - centre[1], start[0] - centre[0])
        for i in range(steps + 1):
            angle = angle0 + side * theta * i / steps
            out.append(centre + radius * np.array([math.cos(angle), math.sin(angle)]))
            at.append(arc[k] - reach + 2.0 * reach * i / steps)
    out.append(points[-1])
    at.append(arc[-1])
    return np.array(out), np.array(at)


def corner_blend_m(op: dict, motion: dict) -> float:
    """The radius a stroke's corners are rounded by: motion.yaml corners.blend_radius_m, no larger than half the
    width the stroke was generated at (its mark); 0, today's sharp corner, without either."""
    radius = float(motion["corners"].get("blend_radius_m", 0.0))
    width = op.get("generation_width_m")
    return min(radius, 0.5 * float(width)) if radius > 0.0 and width else 0.0


def page_points(op: dict, base_from_page: np.ndarray, height_m: float, fade_m: float | None = None,
                blend_m: float = 0.0) -> np.ndarray:
    """(N, 3) stroke polyline in the base: page (x, y) at height_m above z = 0, each point higher by its deposition
    lift (deposition_lift), its corners rounded by `blend_m` (blend_corners)."""
    pts, lift = _stroke_xy(op, fade_m, blend_m)
    local = np.column_stack([pts, float(height_m) + lift, np.ones(len(pts))])
    return (np.asarray(base_from_page, float) @ local.T).T[:, :3]


def _height(tip, base_from_page) -> float:
    return float(base_from_page[:3, 2] @ (np.asarray(tip) - base_from_page[:3, 3]))


def _liftoff(tip, rot, base_from_page, motion, scale) -> list[_Leg]:
    """A straight lift to the standoff when the tip is below it (never drag the pen across the page)."""
    standoff = float(motion["approach"]["standoff_m"])
    height = _height(tip, base_from_page)
    if height >= standoff - 1e-3:
        return []
    return [_lift_leg(tip, rot, tip + (standoff - height) * base_from_page[:3, 2], motion, scale)]


def _approach_legs(tip0, r0, start, r_pen, base_from_page, c0, motion, scales, settle_s) -> list[_Leg]:
    """Lift off if low, travel to the standoff above `start`, descend (the last final_m gently), settle."""
    dt, _ = _cfg(motion)
    approach = motion["approach"]
    normal = base_from_page[:3, 2]
    legs = _liftoff(tip0, r0, base_from_page, motion, scales.get(_GROUP_LIFT, 1.0))
    p_from = legs[-1].p[-1] if legs else tip0
    legs += _travel_legs(p_from, r0, [start + approach["standoff_m"] * normal, start + approach["final_m"] * normal,
                                      start], r_pen,
                         [motion["tip_speed"]["pen_up_max_m_s"], approach["descend_m_s"], approach["final_m_s"]],
                         scales.get(_GROUP_TRAVEL, 1.0), motion, c0, gentle_end=True,
                         phases=[PHASE_TRAVEL, PHASE_DESCEND, PHASE_DESCEND])
    p, r = tl.hold_rows(start, r_pen, settle_s, dt)
    return legs + [_Leg(p, r, False, PHASE_SETTLE, _GROUP_TRAVEL)]


def _stop_ease_s(length: float, cruise: float, motion: dict) -> float:
    """The quintic ease of a stroke end that stops on the skin: draw.ease_s, a short stroke's shrunk."""
    draw = motion["draw"]
    return max(float(draw["min_ease_s"]),
               min(float(draw["ease_s"]), float(draw["short_ease_fraction"]) * length / cruise))


def drawing_time_law(poly, cruise, motion):
    """Shared drawing profile for the planner and offline duration estimates (no IK)."""
    dt, retime = _cfg(motion)
    return tl.corner_time_law(poly, cruise, _stop_ease_s(tl.polyline_length(poly), cruise, motion), dt, retime)


# --- stroke ends (motion.yaml draw.ends) ---------------------------------------------------------
STOP, FLY = "stop", "fly"


@dataclass(frozen=True)
class Fly:
    """motion.yaml draw.fly: how a stroke end is drawn moving (draw.ends: fly)."""

    band_m: float     # the needle can mark while the pen is within this of its pen-down path
    clear_m: float    # the pen starts and stops moving sideways this far over that band
    glide_m: float    # along the stroke's line, the distance over which the pen comes down from there (and goes up)
    inset_m: float    # how far inside the stroke's end the pen reaches (and leaves) its pen-down path
    ease_s: float     # the quintic ease from rest to the draw speed, in the air

    @property
    def height_m(self) -> float:
        """How far over its pen-down path the pen starts and stops moving sideways."""
        return self.band_m + self.clear_m


def stroke_ends(motion: dict) -> Fly | None:
    """The moving stroke ends of motion.yaml draw.ends: fly, or None for stop (a stroke without the key stops)."""
    draw = motion["draw"]
    mode = draw.get("ends", STOP)
    if mode == STOP:
        return None
    if mode != FLY:
        raise ValueError(f"draw.ends must be {STOP} or {FLY}, got {mode!r}")
    return Fly(**{key: float(draw["fly"][key]) for key in ("band_m", "clear_m", "glide_m", "inset_m", "ease_s")})


def _end_tangent(poly: np.ndarray, *, first: bool) -> np.ndarray:
    """The unit direction of a polyline's first (or last) chord that has a length."""
    seg = np.diff(np.asarray(poly, float), axis=0)
    length = np.linalg.norm(seg, axis=1)
    k = np.flatnonzero(length > 0.0)[0 if first else -1]
    return seg[k] / length[k]


def fly_run_m(fly: Fly, cruise: float) -> float:
    """How far outside a flying stroke end the pen starts or stops moving sideways: the ease's distance and the
    glide, less the inset."""
    return 0.5 * cruise * fly.ease_s + fly.glide_m - fly.inset_m


def flying_law(poly, cruise: float, motion: dict, fly: Fly, start: bool, end: bool):
    """The drawing law of a stroke whose `start` and/or `end` is drawn moving: (path, s, lift, s0).

    `path` is the stroke's polyline extended straight along its end tangents by each flying end's run (fly_run_m);
    `s` the arc length along it at each control tick; `lift` the pen's height over its pen-down path there; `s0`
    where the stroke's first point lies on the path. A flying end eases from rest to `cruise` over fly.ease_s
    while the pen is fly.height_m up, then comes down over fly.glide_m (a quintic of distance, level at both
    ends), reaching the path fly.inset_m inside the stroke; the end mirrors it (the stroke is at least as long as
    its flying ends' insets: _fly_ends). An end that stops keeps today's
    ease on the skin (_stop_ease_s) and no lift. Corners slow it as they slow any stroke. No IK."""
    dt, retime = _cfg(motion)
    poly = np.asarray(poly, float)
    length = tl.polyline_length(poly)
    run, inset = fly_run_m(fly, cruise), fly.inset_m
    stop_ease = _stop_ease_s(length, cruise, motion)
    path, s0 = poly, 0.0
    if start:
        path, s0 = np.vstack([poly[0] - run * _end_tangent(poly, first=True), path]), run
    if end:
        path = np.vstack([path, poly[-1] + run * _end_tangent(poly, first=False)])
    _, s, _, _ = tl.corner_time_law(path, cruise, fly.ease_s if start else stop_ease, dt, retime,
                                    fly.ease_s if end else stop_ease)
    lift = np.zeros(len(s))
    if start:
        lift = np.maximum(lift, fly.height_m * (1.0 - tl._quintic(
            np.clip((s - (s0 + inset - fly.glide_m)) / fly.glide_m, 0.0, 1.0))))
    if end:
        lift = np.maximum(lift, fly.height_m * tl._quintic(np.clip((s - (s0 + length - inset)) / fly.glide_m, 0.0, 1.0)))
    return path, s, lift, s0


def _flying_legs(poly, normals, from_arc_m, r_start, cruise, motion, fly: Fly, start: bool, end: bool,
                 turn: bool = False) -> list[_Leg]:
    """The legs of a stroke with flying ends (flying_law): the run in (pen up, PHASE_DESCEND), the stroke's own
    extent (pen down, PHASE_DRAW, its arc) and the run out (PHASE_LIFT), one group so a slow-down slows them
    together. `normals` is the page's normal, or one per point of `poly` on a curved page: the lift is along the
    normal at the nearest point of the stroke, and with `turn` the pen is turned to it."""
    path, s, lift, s0 = flying_law(poly, cruise, motion, fly, start, end)
    length = tl.polyline_length(poly)
    p, _ = tl.resample_polyline_by_arclength(path, s)
    on = np.clip(s - s0, 0.0, length)
    normals = np.asarray(normals, float)
    if normals.ndim == 1:
        n = np.broadcast_to(normals, (len(s), 3))
    else:
        arc = tl.arc_lengths(poly)
        n = np.stack([np.interp(on, arc, normals[:, k]) for k in range(3)], axis=1)
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
    p = p + lift[:, None] * n
    rot = (np.stack([tl.align_rotation(r_start[:, 2], -nk) @ r_start for nk in n]) if turn
           else np.repeat(r_start[None], len(p), axis=0))
    before, after = s < s0 - 1e-9, s > s0 + length + 1e-9
    legs = []
    for rows, pen, phase in ((before, False, PHASE_DESCEND), (~before & ~after, True, PHASE_DRAW),
                             (after, False, PHASE_LIFT)):
        if rows.any():
            leg = _Leg(p[rows], rot[rows], pen, phase, _GROUP_DRAW, arc=float(from_arc_m) + on[rows] if pen else None)
            leg.axis = n[rows]
            legs.append(leg)
    return legs


def _fly_ends(fly: Fly | None, poly, xy, clear_m, cruise: float, *, joined: bool, from_arc_m: float,
              lift_at_end: bool) -> tuple[bool, bool, list[str]]:
    """(whether the start flies, whether the end flies, why an end that would fly stops instead).

    A start flies when the pen comes down for it at the stroke's first point: never joined pen-down to the op
    before, nor resumed part-way. An end flies when the pen lifts there: never where the next op continues the
    path. Either stops instead when its run would leave the page's clear area (`clear_m`, its size about the
    page's centre; None: not checked). Both stop on a stroke shorter than the insets of its flying ends: its pen
    would never reach its pen-down path, and would be inside the marking band for longer than the stroke is."""
    if fly is None:
        return False, False, []
    start, end, why = not joined and from_arc_m <= 0.0, bool(lift_at_end), []
    length = tl.polyline_length(poly)
    if (start or end) and length < fly.inset_m * (start + end):
        return False, False, [f"both ends stop: the stroke is {length * 1e3:.1f} mm, shorter than the "
                              f"{fly.inset_m * (start + end) * 1e3:.1f} mm its flying ends come down and leave over"]
    if clear_m is not None:
        xy = np.asarray(xy, float)
        run = fly_run_m(fly, cruise)
        half = np.asarray(clear_m, float) / 2.0
        if start and np.any(np.abs(xy[0] - run * _end_tangent(xy, first=True)) > half):
            start = False
            why.append(f"the start stops: its {run * 1e3:.1f} mm run-in would leave the page's clear area")
        if end and np.any(np.abs(xy[-1] + run * _end_tangent(xy, first=False)) > half):
            end = False
            why.append(f"the end stops: its {run * 1e3:.1f} mm run-out would leave the page's clear area")
    return start, end, why


def _draw_leg(poly, from_arc_m, r_pen, cruise, motion) -> _Leg:
    """The stroke at `cruise` with quintic end eases (short strokes shrink theirs) and corner slow-downs; a
    stroke without length is a dot, a pen-down dwell."""
    dt, _ = _cfg(motion)
    draw = motion["draw"]
    length = tl.polyline_length(poly)
    if length <= 1e-6:
        p, r = tl.hold_rows(poly[0], r_pen, 2.0 * float(draw["min_ease_s"]), dt)
        return _Leg(p, r, True, PHASE_DRAW, _GROUP_DRAW, arc=np.full(len(p), float(from_arc_m)))
    _, s, _, _ = drawing_time_law(poly, cruise, motion)
    p, _ = tl.resample_polyline_by_arclength(poly, s)
    return _Leg(p, np.repeat(r_pen[None], len(p), axis=0), True, PHASE_DRAW, _GROUP_DRAW, arc=float(from_arc_m) + s)


def plan_op(op: dict, *, base_from_page: np.ndarray, q_seed: np.ndarray, kin: Kinematics, motion: dict,
            speed_m_s: float, from_arc_m: float = 0.0, pen_down_at_start: bool = False,
            lift_at_end: bool = True, pen: PenDown | None = None, surface=None, turn_pen: bool = True,
            clear_m=None, check=None) -> Trajectory:
    """A program op from the seed joints (measured revolute joints, the last carriage command).

    stroke: travel to the standoff above the start (from `from_arc_m`), descend, settle, draw, lift; a
    caller sets lift_at_end=False only when the following op continues this path. The pen-down path
    rides `pen` (pen_down; None: motion.yaml's, which must press). The program `continues` field
    describes the incoming join, never the ending lift. With pen_down_at_start: within dispatch_drift.start_tolerance_m of the start
    (resume_reapproach_m when resuming mid-stroke) it joins the path pen-down and draws; farther, it lifts
    and re-approaches. A tip below the standoff
    lifts straight off the page before any travel.
    `base_from_page` is the touched, trimmed page pose. A curved `surface` (tatbot_motion.surface.PageSurface over
    that page) plans on it instead (_plan_op_surface; `turn_pen`: the pen turned to each point's normal, else held
    square to the page); a flat one or None plans on the plane exactly as before.

    With motion.yaml draw.ends: fly the ends where the pen comes down or lifts are drawn moving (flying_law): the
    pen eases to the draw speed in the air behind the stroke's first point, comes down through the marking band
    at that speed, and leaves the same way. An end joined pen-down to its neighbour, resumed part-way, or whose
    run would leave the page's clear area (`clear_m`, the page's clear size) stops as it does today; and when
    the flying plan cannot be made at all (PlanError: reach, a joint cap, or `check`, which is given the flying
    plan's joints at every control tick and raises PlanError for a way it refuses, the collision guard's) the
    stroke is planned with stopping ends, exactly today's plan. info["ends"] says what each end does and why."""
    if surface is not None and not surface.flat:
        return _plan_op_surface(op, surface=surface, q_seed=q_seed, kin=kin, motion=motion, speed_m_s=speed_m_s,
                                from_arc_m=from_arc_m, pen_down_at_start=pen_down_at_start, lift_at_end=lift_at_end,
                                pen=pen, turn_pen=turn_pen, clear_m=clear_m, check=check)
    base_from_page = np.asarray(base_from_page, float)
    kind = op.get("op", "stroke")
    if kind != "stroke":
        raise NotImplementedError(f"op {kind!r}: plan_op plans strokes; tatbot_motion.dip plans a dip")
    dt, _ = _cfg(motion)
    pen = pen or pen_down(motion)
    seed = kin.fk(q_seed)
    tip0, r0 = seed[:3, 3], seed[:3, :3]
    r_pen = pen_rotation(base_from_page, r0)
    blend = corner_blend_m(op, motion)
    poly = tl.trim_polyline(page_points(op, base_from_page, pen.height_m, pen.fade_m, blend), float(from_arc_m))
    start, normal = poly[0], base_from_page[:3, 2]
    joined = pen_down_at_start and _joined(tip0, start, motion, from_arc_m)
    cruise0 = min(float(speed_m_s), float(motion["tip_speed"]["pen_down_max_m_s"]))
    fly = stroke_ends(motion)
    fly_start, fly_end, why = _fly_ends(fly, poly, _stroke_xy(op, pen.fade_m, blend)[0], clear_m, cruise0,
                                        joined=joined, from_arc_m=from_arc_m, lift_at_end=lift_at_end)

    def build(scales, fly_start=fly_start, fly_end=fly_end):
        cruise = cruise0 * scales.get(_GROUP_DRAW, 1.0)
        down = start   # where the pen comes down: a flying start's is behind the stroke, over the marking band
        if fly_start:
            down = start - fly_run_m(fly, cruise) * _end_tangent(poly, first=True) + fly.height_m * normal
        if not joined:
            legs = _approach_legs(tip0, r0, down, r_pen, base_from_page, _carriage_at(kin, q_seed), motion, scales,
                                  pen.settle_s)
        elif float(np.linalg.norm(tip0 - start)) > 1e-9:
            p, r = tl.line_segment(tip0, start, r0, r_pen, cruise, dt)
            legs = [_Leg(p, r, True, PHASE_DRAW, _GROUP_DRAW, arc=np.full(len(p), float(from_arc_m)))]
        else:
            legs = []
        if fly_start or fly_end:
            legs += _flying_legs(poly, normal, from_arc_m, r_pen, cruise, motion, fly, fly_start, fly_end)
        else:
            legs.append(_draw_leg(poly, from_arc_m, r_pen, cruise, motion))
        if lift_at_end:
            end = legs[-1].p[-1]
            up = motion["approach"]["standoff_m"] - (fly.height_m if fly_end else 0.0)
            legs.append(_lift_leg(end, r_pen, end + up * normal, motion, scales.get(_GROUP_LIFT, 1.0)))
        for leg in legs:   # a flat page: every knot's depth axis is its normal (a surface would give each its own)
            leg.axis = normal
        return legs

    traj, ends = _solve_ends(build, q_seed, kin, motion, fly, fly_start, fly_end, why, check)
    traj.info.update(op=op.get("id"), from_arc_m=float(from_arc_m), stroke_m=tl.polyline_length(poly), joined=joined,
                     **ends)
    return traj


def _solve_ends(build, q_seed, kin, motion, fly, fly_start: bool, fly_end: bool, why: list[str], check):
    """_solve a stroke whose `build` takes (scales, fly_start, fly_end): with flying ends when it can be planned and
    `check` lets it by, else with stopping ones (today's plan, which `check` is not asked about), and info's `ends`
    entry ({} without draw.ends: fly, so today's plans keep their info)."""
    if fly_start or fly_end:
        try:
            traj = _solve(lambda scales: build(scales, fly_start, fly_end), q_seed, kin, motion,
                          check=check or (lambda rows: None))
            return traj, {"ends": {"start": FLY if fly_start else STOP, "end": FLY if fly_end else STOP, "why": why}}
        except PlanError as error:
            why = [*why, f"both ends stop: the flying plan was refused ({error})"]
    traj = _solve(lambda scales: build(scales, False, False), q_seed, kin, motion)
    return traj, ({} if fly is None else {"ends": {"start": STOP, "end": STOP, "why": why}})


def _joined(tip, start, motion: dict, from_arc_m: float) -> bool:
    tol = motion["resume_reapproach_m"] if from_arc_m > 0.0 else motion["dispatch_drift"]["start_tolerance_m"]
    return float(np.linalg.norm(np.asarray(tip, float) - start)) <= float(tol)


# Whether plan_op, pen down at the start, joins the stroke's path where `tip` (base) stands: its start from from_arc_m
# within dispatch_drift.start_tolerance_m (resume_reapproach_m resuming mid-stroke). Else it lifts and re-approaches: a
# descent, which a caller measuring the skin first must measure before.
def joins(op: dict, tip, *, base_from_page, motion: dict, from_arc_m: float = 0.0, pen=None, surface=None) -> bool:
    pen = pen or pen_down(motion)
    h, fade = pen.height_m, pen.fade_m
    blend = corner_blend_m(op, motion)
    poly = (surface_points(op, surface, h, fade, blend)[0] if surface is not None and not surface.flat
            else page_points(op, base_from_page, h, fade, blend))
    return _joined(tip, tl.trim_polyline(poly, float(from_arc_m))[0], motion, from_arc_m)


def plan_travel(*, q_seed: np.ndarray, base_from_tcp: np.ndarray, kin: Kinematics, motion: dict,
                via: list | None = None) -> Trajectory:
    """Pen-up travel to a TCP pose (through the `via` positions when given): one accelerate/cruise/decelerate
    profile under the tip and joint caps, the rotation slerped over the first chord at no more than
    approach.omega_max_rad_s, the carriage back to its bias."""
    target = np.asarray(base_from_tcp, float)
    seed = kin.fk(q_seed)
    route = [np.asarray(v, float) for v in (via or [])] + [target[:3, 3]]

    def build(scales):
        return _travel_legs(seed[:3, 3], seed[:3, :3], route, target[:3, :3],
                            [motion["tip_speed"]["pen_up_max_m_s"]] * len(route), scales.get(_GROUP_TRAVEL, 1.0),
                            motion, _carriage_at(kin, q_seed), gentle_end=False, phases=[PHASE_TRAVEL] * len(route))
    return _solve(build, q_seed, kin, motion)


HOP_STEP_M = 0.0005   # a hop's curve is sampled this finely: the corner budget then reads its curvature (at 1 mm the
                      # chords' kinks held a 43 mm hop 6% longer, at 2 mm 60%)


def hop_via(p_from, up_from, p_to, up_to, rise_m: float, step_m: float = HOP_STEP_M) -> list[np.ndarray]:
    """plan_travel's via points for one pen-up move from `p_from` up and over onto `p_to`: a cubic Bezier leaving
    along `up_from` and coming down `up_to` (unit, base), its inner control points `rise_m` out along them, sampled
    every `step_m` or closer, its ends left out. It stays in their hull: never under the chord between the ends, and
    0.75 rise_m over it midway when both point the same way. A lift, a stop at its top and a travel down are three
    goals and two more of the arm's starts and stops; this is one."""
    p0, p3 = np.asarray(p_from, float), np.asarray(p_to, float)
    p1 = p0 + float(rise_m) * np.asarray(up_from, float) / np.linalg.norm(up_from)
    p2 = p3 + float(rise_m) * np.asarray(up_to, float) / np.linalg.norm(up_to)
    longest = max(np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1), np.linalg.norm(p3 - p2))   # |B'| <= 3 longest
    # A short hop's longest leg is the rise itself, a whole number of steps (touch.lift_m 20 mm: 120): the unit
    # normal's last bit made it 120 or 120.00000000000004, 120 samples or 121, and the simulator's two drivers
    # planned one setup hop a tick apart on it (the parity gate's P1 to P3, 2026-11-08 to 11-12).
    u = np.linspace(0.0, 1.0, max(4, int(math.ceil(3.0 * longest / float(step_m) - 1e-9))) + 1)[1:-1, None]
    w = 1.0 - u
    return list(w ** 3 * p0 + 3.0 * u * w * w * p1 + 3.0 * u * u * w * p2 + u ** 3 * p3)


def plan_hover(op: dict, *, base_from_page: np.ndarray, q_seed: np.ndarray, kin: Kinematics, motion: dict,
               from_arc_m: float = 0.0, pen: PenDown | None = None, surface=None, turn_pen: bool = True) -> Trajectory:
    """Pen-up travel to the standoff above a stroke's start (from `from_arc_m`) with the drawing rotation, at rest:
    where plan_op's approach would turn down to the page. Stencil tracking stops there to measure, then plans the
    stroke from it. A tip below the standoff lifts straight off the page first. A curved `surface`: on it."""
    if surface is not None and not surface.flat:
        return _plan_hover_surface(op, surface=surface, q_seed=q_seed, kin=kin, motion=motion, from_arc_m=from_arc_m,
                                   pen=pen, turn_pen=turn_pen)
    base_from_page = np.asarray(base_from_page, float)
    seed = kin.fk(q_seed)
    pen = pen or pen_down(motion)
    start = tl.trim_polyline(page_points(op, base_from_page, pen.height_m, pen.fade_m, corner_blend_m(op, motion)),
                             float(from_arc_m))[0]
    hover = start + float(motion["approach"]["standoff_m"]) * base_from_page[:3, 2]
    r_pen = pen_rotation(base_from_page, seed[:3, :3])

    def build(scales):
        legs = _liftoff(seed[:3, 3], seed[:3, :3], base_from_page, motion, scales.get(_GROUP_LIFT, 1.0))
        p_from = legs[-1].p[-1] if legs else seed[:3, 3]
        return legs + _travel_legs(p_from, seed[:3, :3], [hover], r_pen, [motion["tip_speed"]["pen_up_max_m_s"]],
                                   scales.get(_GROUP_TRAVEL, 1.0), motion, _carriage_at(kin, q_seed), gentle_end=False,
                                   phases=[PHASE_TRAVEL])

    traj = _solve(build, q_seed, kin, motion)
    traj.info.update(op=op.get("id"), hover=True)
    return traj


def plan_lift(*, q_seed: np.ndarray, direction: np.ndarray, distance_m: float, kin: Kinematics,
              motion: dict) -> Trajectory:
    """Straight pen-up move of `distance_m` along `direction` (unit, base) at approach.lift_m_s, the first
    approach.lift_first_m at the touchdown budget: off the page after a touch or an interrupted stroke."""
    seed = kin.fk(q_seed)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    return _solve(lambda scales: [_lift_leg(seed[:3, 3], seed[:3, :3], seed[:3, 3] + distance_m * d, motion,
                                            scales.get(_GROUP_LIFT, 1.0))], q_seed, kin, motion)


def plan_touch(*, q_seed: np.ndarray, start_base_from_tcp: np.ndarray, direction: np.ndarray, max_travel_m: float,
               kin: Kinematics, motion: dict, prior_distance_m: float | None = None,
               speed_m_s: float | None = None, via: list | None = None) -> Trajectory:
    """Guarded descent along `direction` (unit, base) from the start pose (travelled to first when the tip
    is farther than dispatch_drift.start_tolerance_m, through the `via` positions when given): settle, then
    touch.fast_m_s to touch.slow_above_m short of the prior (phase DESCEND), then touch.slow_m_s (or
    speed_m_s) with the +-wiggle across the direction (phase TOUCH) up to max_travel_m in all; without a
    prior the slow leg covers touch.search_m. The driver's tip-lag guard ends it: arm it (guard_mode 1) at
    `guard_arm_time_s(traj)`, the first TOUCH knot, so the fast leg's own tracking lag has decayed before
    the 0.5 s arming runs out."""
    touch = motion["touch"]
    dt, retime = _cfg(motion)
    start = np.asarray(start_base_from_tcp, float)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    total = float(max_travel_m or touch["max_travel_m"])
    slow = float(speed_m_s or touch["slow_m_s"])
    fast_len = 0.0 if prior_distance_m is None else min(max(0.0, prior_distance_m - touch["slow_above_m"]), total)
    if prior_distance_m is None:
        total = min(total, float(touch["search_m"]))
    lateral = np.array([1.0, 0.0, 0.0]) - d[0] * d
    if np.linalg.norm(lateral) < 0.1:
        lateral = np.array([0.0, 1.0, 0.0]) - d[1] * d
    lateral /= np.linalg.norm(lateral)
    seed = kin.fk(q_seed)
    far = (np.linalg.norm(seed[:3, 3] - start[:3, 3]) > motion["dispatch_drift"]["start_tolerance_m"]
           or tl.rotation_angle(start[:3, :3] @ seed[:3, :3].T) > 1e-3)

    route = [np.asarray(v, float) for v in (via or [])] + [start[:3, 3]]

    def build(scales):
        legs = []
        if far:
            legs += _travel_legs(seed[:3, 3], seed[:3, :3], route, start[:3, :3],
                                 [motion["tip_speed"]["pen_up_max_m_s"]] * len(route), scales.get(_GROUP_TRAVEL, 1.0),
                                 motion, _carriage_at(kin, q_seed), gentle_end=False, phases=[PHASE_TRAVEL] * len(route))
        p, r = tl.hold_rows(start[:3, 3], start[:3, :3], touch["settle_s"], dt)
        legs.append(_Leg(p, r, False, PHASE_SETTLE, _GROUP_TRAVEL))
        points = [start[:3, 3]] + ([start[:3, 3] + fast_len * d] if fast_len > 1e-6 else []) + [start[:3, 3] + total * d]
        caps = ([touch["fast_m_s"]] if fast_len > 1e-6 else []) + [slow]
        p, r, chord = tl.pen_up_leg(np.stack(points), start[:3, :3], start[:3, :3], caps, retime, dt)
        phase = np.where(chord == len(caps) - 1, PHASE_TOUCH, PHASE_DESCEND)
        slow_rows = np.flatnonzero(phase == PHASE_TOUCH)
        t_rel = (slow_rows - slow_rows[0]) * dt
        span = t_rel[-1] if len(t_rel) else 0.0
        fade = 0.25 / float(touch["wiggle_hz"])
        envelope = tl._quintic(np.clip(np.minimum(t_rel, span - t_rel) / fade, 0.0, 1.0))
        p = p.copy()
        p[slow_rows] += (float(touch["wiggle_amplitude_m"]) * envelope
                         * np.sin(2.0 * math.pi * float(touch["wiggle_hz"]) * t_rel))[:, None] * lateral[None, :]
        legs.append(_Leg(p, r, False, phase, _GROUP_TRAVEL))
        return legs

    traj = _solve(build, q_seed, kin, motion)
    traj.info.update(fast_m=fast_len, travel_m=total, lateral=lateral.tolist())
    return traj


def probe_travel_m(direction: np.ndarray, prior_distance_m: float, motion: dict) -> float:
    """A probe touch's whole guarded travel: to the expected contact `prior_distance_m` along `direction` (unit,
    base), then past it no farther than motion.yaml probe.axial_cap_m along the stylus and probe.lateral_cap_m
    across it. The stylus stands along base z: the station sits on the arm's table, a few degrees at most off it."""
    probe = motion["probe"]
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    axial = abs(float(d[2]))
    lateral = math.sqrt(max(0.0, 1.0 - axial * axial))
    caps = [float(probe["axial_cap_m"]) / axial if axial > 1e-9 else math.inf,
            float(probe["lateral_cap_m"]) / lateral if lateral > 1e-9 else math.inf]
    return float(prior_distance_m) + min(caps)


def guard_arm_time_s(traj: Trajectory) -> float | None:
    """When to arm the touch guard: the first PHASE_TOUCH knot's time, or None."""
    rows = np.flatnonzero(traj.phase == PHASE_TOUCH)
    return float(traj.t[rows[0]]) if len(rows) else None


def to_knots(traj: Trajectory, rate_hz: float) -> Trajectory:
    """Resample to FollowJointTrajectory knots at rate_hz (50-100 Hz; JTC interpolates at 400 Hz). The last
    sample is always a knot, at rest."""
    step = 1.0 / float(rate_hz)
    t = np.arange(0.0, traj.t[-1], step)
    if traj.t[-1] - t[-1] > 1e-9:
        t = np.append(t, traj.t[-1])
    else:
        t[-1] = traj.t[-1]
    dt = float(traj.t[1] - traj.t[0]) if len(traj.t) > 1 else 1.0
    nearest = np.clip(np.rint(t / dt).astype(int), 0, len(traj.t) - 1)

    def interp(values):
        return np.stack([np.interp(t, traj.t, values[:, k]) for k in range(values.shape[1])], axis=1)

    qd = interp(traj.qd)
    qd[-1] = 0.0
    axis = None
    if traj.axis is not None:
        axis = interp(traj.axis)
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return Trajectory(traj.joint_names, t, interp(traj.q), qd, interp(traj.tip), traj.arc_m[nearest],
                      traj.phase[nearest], dict(traj.info, knot_rate_hz=float(rate_hz)), axis,
                      None if traj.dq_dh is None else interp(traj.dq_dh))


START_BLEND_S = 0.5   # blend_start fades a goal's start offset out over this (a quintic): 2 mm at 7.5 mm/s, 20 mrad at 75 mrad/s


def blend_start(traj: Trajectory, q_from: np.ndarray, kin: Kinematics, blend_s: float = START_BLEND_S) -> Trajectory:
    """`traj` begun from `q_from` rather than its first knot: the offset fades out over its first `blend_s` (no later than
    its end) as a quintic, the knots' velocities with it, and its tip follows. JTC starts a goal from the measured joints
    and closes any gap to the first knot sent, 10 ms in: pen-up goals pre-planned from a stroke's untrimmed end and
    let through by dispatch_drift (pen_up_*: 2 mm, 20 mrad) started up to 2.0 mm off the arm, the reference sweeping it
    at up to 1.6 rad/s (2026-10-20)."""
    delta = np.asarray(q_from, float) - np.asarray(traj.q[0], float)
    t = np.asarray(traj.t, float)
    span = min(float(blend_s), float(t[-1]))
    if span <= 0.0 or not np.any(delta):
        return traj
    u = np.clip(t / span, 0.0, 1.0)
    fade, rate = 1.0 - tl._quintic(u), np.where(u < 1.0, -30.0 * u * u * (1.0 - u) ** 2 / span, 0.0)
    out = copy.copy(traj)
    out.q, out.qd = traj.q + fade[:, None] * delta, traj.qd + rate[:, None] * delta
    out.tip = np.array(traj.tip, float)
    out.tip[fade > 0.0] = [kin.fk(row)[:3, 3] for row in out.q[fade > 0.0]]
    out.info = {**getattr(traj, "info", {}), "start_blend_rad": float(np.abs(delta[kin.revolute]).max())}
    return out


def dispatch_drift(traj: Trajectory, q_measured: np.ndarray, kin: Kinematics, motion: dict,
                   pen_down: bool | None = None) -> dict:
    """How far the measured pose drifted from the plan's first knot: joints (revolute, rad) and tip (m),
    against dispatch_drift pen_down_* / pen_up_* (pen down when the plan starts drawing). The carriage
    seed is the last command, so the tip uses the plan's carriage. ok False = re-plan from q_measured."""
    q = np.asarray(q_measured, float).copy()
    if kin.carriage is not None:
        q[kin.carriage] = traj.q[0, kin.carriage]
    joint = float(np.abs(q[kin.revolute] - traj.q[0, kin.revolute]).max())
    tip = float(np.linalg.norm(kin.fk(q)[:3, 3] - traj.tip[0]))
    if pen_down is None:
        pen_down = int(traj.phase[0]) == PHASE_DRAW
    limits = motion["dispatch_drift"]
    key = "pen_down" if pen_down else "pen_up"
    ok = joint <= float(limits[f"{key}_rad"]) and tip <= float(limits[f"{key}_m"])
    return {"ok": ok, "joint_rad": joint, "tip_m": tip, "pen_down": bool(pen_down)}


# --- a curved page (tatbot_motion.surface): every point its own normal ------------------------------------------
SURFACE_STEP_M = 0.0005   # stroke points are spaced this finely in print mm before they are lifted onto a curve: a
                          # chord across a 20 mm segment on a 40 mm radius sags 1.3 mm under the skin


def surface_points(op: dict, surface, height_m: float, fade_m: float | None = None,
                   blend_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """((N, 3) stroke polyline in the base, height_m (and each point's deposition lift) along the surface's normal over
    the print points, (N, 3) unit normals): page_points on a curve, the print's (u, v) through the surface's chart."""
    pts, lift = _stroke_xy(op, fade_m, blend_m)
    if len(pts) > 1:
        arc = tl.arc_lengths(pts)
        fine = np.unique(np.r_[arc, np.arange(0.0, arc[-1], SURFACE_STEP_M)])
        pts, lift = np.stack([np.interp(fine, arc, pts[:, 0]), np.interp(fine, arc, pts[:, 1])], axis=1), np.interp(
            fine, arc, lift)
    return surface.lift(pts, float(height_m) + lift)


def _trim_with(poly: np.ndarray, normals: np.ndarray, from_arc_m: float) -> tuple[np.ndarray, np.ndarray]:
    """tl.trim_polyline with each kept point's normal (the cut point's interpolated)."""
    out = tl.trim_polyline(poly, float(from_arc_m))
    if len(out) == len(poly):
        return out, normals
    arc = tl.arc_lengths(poly)
    first = np.array([np.interp(float(from_arc_m), arc, normals[:, k]) for k in range(3)])
    first /= np.linalg.norm(first)
    return out, np.vstack([first, normals[len(poly) - len(out) + 1:]])


def _surface_liftoff(tip, rot, surface, motion, scale) -> list[_Leg]:
    """_liftoff over a curve: straight along the normal under the tip to the standoff over the surface."""
    standoff = float(motion["approach"]["standoff_m"])
    gap, normal = surface.height_over(tip)
    normal = np.asarray(normal, float)
    if not (np.isfinite(gap) and np.isfinite(normal).all() and np.linalg.norm(normal) > 0.5):
        # no foot under the tip (a steep field's normal search ran off it; the pen back from a dip, 2026-10-21):
        # the whole standoff up the page's normal, away from any skin under it
        gap, normal = 0.0, np.asarray(surface.base[:3, 2], float)
    if gap >= standoff - 1e-3:
        return []
    leg = _lift_leg(tip, rot, np.asarray(tip) + (standoff - gap) * normal, motion, scale)
    leg.axis = normal
    return [leg]


def _surface_draw_leg(poly, normals, from_arc_m, r_start, cruise, motion, turn: bool = True) -> _Leg:
    """_draw_leg with the pen turned at every knot to the surface's normal there (the least turn from the start's);
    without `turn` the pen holds r_start."""
    dt, _ = _cfg(motion)
    draw = motion["draw"]
    if tl.polyline_length(poly) <= 1e-6:
        p, r = tl.hold_rows(poly[0], r_start, 2.0 * float(draw["min_ease_s"]), dt)
        leg = _Leg(p, r, True, PHASE_DRAW, _GROUP_DRAW, arc=np.full(len(p), float(from_arc_m)))
        leg.axis = normals[0]
        return leg
    _, s, _, _ = drawing_time_law(poly, cruise, motion)
    p, _ = tl.resample_polyline_by_arclength(poly, s)
    arc = tl.arc_lengths(poly)
    n = np.stack([np.interp(s, arc, normals[:, k]) for k in range(3)], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    rot = (np.stack([tl.align_rotation(r_start[:, 2], -nk) @ r_start for nk in n]) if turn
           else np.repeat(r_start[None], len(p), axis=0))
    leg = _Leg(p, rot, True, PHASE_DRAW, _GROUP_DRAW, arc=float(from_arc_m) + s)
    leg.axis = n
    return leg


def _plan_op_surface(op: dict, *, surface, q_seed: np.ndarray, kin: Kinematics, motion: dict, speed_m_s: float,
                     from_arc_m: float, pen_down_at_start: bool, lift_at_end: bool, pen: PenDown | None,
                     turn_pen: bool = True, clear_m=None, check=None) -> Trajectory:
    """plan_op on a curved page: the stroke's print points lifted pen.height_m along the surface's normal at each,
    the pen turned to that normal along the stroke (`turn_pen`; else held square to the surface's page, as the gauge
    holds were made: _pen_rotation), the approach and the lift along the normal at its two ends, and each knot's
    depth axis its own normal (a trim moves the pen along it). A flying end's run (draw.ends: fly) is straight
    along the stroke's end tangent, lifted along the end's normal: in the plane touching the skin there, so over a
    skin that curves away from it and never under fly.height_m less the sag of a run that short."""
    kind = op.get("op", "stroke")
    seed = kin.fk(q_seed)
    tip0, r0 = seed[:3, 3], seed[:3, :3]
    if kind != "stroke":
        raise NotImplementedError(f"op {kind!r}: plan_op plans strokes; tatbot_motion.dip plans a dip")
    dt, _ = _cfg(motion)
    pen = pen or pen_down(motion)
    blend = corner_blend_m(op, motion)
    poly, normals = _trim_with(*surface_points(op, surface, pen.height_m, pen.fade_m, blend), float(from_arc_m))
    start, n0, n1 = poly[0], normals[0], normals[-1]
    r_pen = _pen_rotation(surface, n0, r0, turn_pen)
    joined = pen_down_at_start and _joined(tip0, start, motion, from_arc_m)
    cruise0 = min(float(speed_m_s), float(motion["tip_speed"]["pen_down_max_m_s"]))
    approach = motion["approach"]
    fly = stroke_ends(motion)
    fly_start, fly_end, why = _fly_ends(fly, poly, _stroke_xy(op, pen.fade_m, blend)[0], clear_m, cruise0,
                                        joined=joined, from_arc_m=from_arc_m, lift_at_end=lift_at_end)

    def build(scales, fly_start=fly_start, fly_end=fly_end):
        cruise = cruise0 * scales.get(_GROUP_DRAW, 1.0)
        legs = []
        down = start   # where the pen comes down: a flying start's is behind the stroke, over the marking band
        if fly_start:
            down = start - fly_run_m(fly, cruise) * _end_tangent(poly, first=True) + fly.height_m * n0
        if not joined:
            legs = _surface_liftoff(tip0, r0, surface, motion, scales.get(_GROUP_LIFT, 1.0))
            p_from = legs[-1].p[-1] if legs else tip0
            legs += _travel_legs(p_from, r0, [down + approach["standoff_m"] * n0, down + approach["final_m"] * n0,
                                              down], r_pen,
                                 [motion["tip_speed"]["pen_up_max_m_s"], approach["descend_m_s"], approach["final_m_s"]],
                                 scales.get(_GROUP_TRAVEL, 1.0), motion, _carriage_at(kin, q_seed), gentle_end=True,
                                 phases=[PHASE_TRAVEL, PHASE_DESCEND, PHASE_DESCEND])
            p, r = tl.hold_rows(down, r_pen, pen.settle_s, dt)
            legs.append(_Leg(p, r, False, PHASE_SETTLE, _GROUP_TRAVEL))
        elif float(np.linalg.norm(tip0 - start)) > 1e-9:
            p, r = tl.line_segment(tip0, start, r0, r_pen, cruise, dt)
            legs = [_Leg(p, r, True, PHASE_DRAW, _GROUP_DRAW, arc=np.full(len(p), float(from_arc_m)))]
        for leg in legs:
            leg.axis = n0 if leg.axis is None else leg.axis
        if fly_start or fly_end:
            legs += _flying_legs(poly, normals, from_arc_m, r_pen, cruise, motion, fly, fly_start, fly_end, turn_pen)
        else:
            legs.append(_surface_draw_leg(poly, normals, from_arc_m, r_pen, cruise, motion, turn_pen))
        if lift_at_end:
            end, r_end = legs[-1].p[-1], legs[-1].rot[-1]
            up = approach["standoff_m"] - (fly.height_m if fly_end else 0.0)
            lift = _lift_leg(end, r_end, end + up * n1, motion, scales.get(_GROUP_LIFT, 1.0))
            lift.axis = n1
            legs.append(lift)
        return legs

    traj, ends = _solve_ends(build, q_seed, kin, motion, fly, fly_start, fly_end, why, check)
    traj.info.update(op=op.get("id"), from_arc_m=float(from_arc_m), stroke_m=tl.polyline_length(poly), joined=joined,
                     surface=True, **ends)
    return traj


def _pen_rotation(surface, normal, r0, turn: bool) -> np.ndarray:
    """The pen's rotation over a curved page: along `normal` (turned the least from r0), or with `turn` False square
    to the surface's page, as the flat planner holds it. Turned to each normal, the arm's own error moved with the
    pen's tilt: a tool 8.6 deg over read the forearm's skin ~1.1 mm higher, and holds turned 5-23 deg disagreed by
    1-3 mm (2026-10-07); held square, the pen is in the pose the gauge holds measured the skin in, and up to 14 deg off
    the skin's normal on a 20 mm design it reaches cos 14 = 97% as deep."""
    if turn:
        return tl.align_rotation(r0[:, 2], -np.asarray(normal, float)) @ r0
    return pen_rotation(np.asarray(surface.base, float), r0)


def _plan_hover_surface(op: dict, *, surface, q_seed: np.ndarray, kin: Kinematics, motion: dict, from_arc_m: float,
                        pen: PenDown | None, turn_pen: bool = True) -> Trajectory:
    """plan_hover on a curved page: the standoff along the normal over the stroke's start, the pen along it (or
    square to the page: _pen_rotation)."""
    seed = kin.fk(q_seed)
    pen = pen or pen_down(motion)
    poly, normals = _trim_with(*surface_points(op, surface, pen.height_m, pen.fade_m, corner_blend_m(op, motion)),
                               float(from_arc_m))
    hover = poly[0] + float(motion["approach"]["standoff_m"]) * normals[0]
    r_pen = _pen_rotation(surface, normals[0], seed[:3, :3], turn_pen)

    def build(scales):
        legs = _surface_liftoff(seed[:3, 3], seed[:3, :3], surface, motion, scales.get(_GROUP_LIFT, 1.0))
        p_from = legs[-1].p[-1] if legs else seed[:3, 3]
        travel = _travel_legs(p_from, seed[:3, :3], [hover], r_pen, [motion["tip_speed"]["pen_up_max_m_s"]],
                              scales.get(_GROUP_TRAVEL, 1.0), motion, _carriage_at(kin, q_seed), gentle_end=False,
                              phases=[PHASE_TRAVEL])
        for leg in travel:
            leg.axis = normals[0]
        return legs + travel

    traj = _solve(build, q_seed, kin, motion)
    traj.info.update(op=op.get("id"), hover=True, surface=True)
    return traj
