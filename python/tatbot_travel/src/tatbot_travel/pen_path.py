"""Surface paths to a timed joint stream for the blue arm: the one executor every decider shares.

A decider -- the classical tracer, Astra, or a person -- names points on the skin. Each becomes a
pen pose: the working point (the laser datasheet's tcp, 10 mm past the lens face) on the point and
the pen axis along the inward normal, so the lens hovers the datasheet's standoff above the skin.
Approach and retreat run along the normal from a clearance height. The path is timed at a constant
speed with smooth ends, sampled at the control rate, and solved by the episode model's own IK
seeded from (and held near) the previous sample, so the wrist roll varies continuously. Every step is checked
against a joint speed cap before anything is commanded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tatbot_travel.camera import Intrinsics, render_plan
from tatbot_travel.kinematics import ArmKinematics
from tatbot_travel.scene import SceneBuilder
from tatbot_travel.tool import load_pen

RATE_HZ = 30.0
SPEED_M_S = 0.015  # along the ink
TRAVEL_SPEED_M_S = 0.04  # approach and retreat
CLEARANCE_M = 0.04  # approach starts this far above the hover pose, along the normal
MAX_JOINT_SPEED = 0.5  # rad/s, any joint, any sample
AXIS_SIGMA_M = 0.006  # pen-axis smoothing along the path
AXIS_RATE = 0.3  # rad/s the pen axis may turn, whatever the path asks
MAX_TILT_RAD = np.radians(30)  # the pen leans at most this far from straight down, whatever the normal
# The pink arm stands 0.68 m to the blue arm's right (base frame -y; registrations of 2026-10-06), both
# facing +x. Every streamed pose keeps the blue arm's links, camera, lens and working point at least
# WALL_MARGIN_M on the blue side of a vertical wall at y = WALL_Y_M, leaving the middle to neither arm.
WALL_Y_M = -0.20
WALL_MARGIN_M = 0.05
WALL_BODIES = ("link_2", "link_3", "link_4", "link_5", "link_6", "carriage_left", "carriage_right",
               "realsense_mount_d405")


def arm_kinematics(intr: Intrinsics | None = None) -> ArmKinematics:
    """The blue arm with the fitted laser: the gap site is the datasheet's working point."""
    pen = load_pen()
    plan = render_plan(intr or Intrinsics.left_wrist())
    return ArmKinematics(SceneBuilder(plan=plan, gap_m=pen.tcp_z - pen.lens_z, pen=pen).compile())


def timed(points: np.ndarray, speed_m_s: float, rate_hz: float = RATE_HZ) -> np.ndarray:
    """Resample a polyline at the control rate for a constant speed, easing in and out (cosine)."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return points[:1]
    ease = 0.25  # fraction of the time spent accelerating, and again decelerating
    duration = s[-1] / ((1 - ease) * speed_m_s)  # so the cruise is ``speed_m_s``
    n = max(2, int(np.ceil(duration * rate_hz)) + 1)
    u = np.linspace(0.0, 1.0, n)
    # Trapezoidal-in-velocity profile: arc length as a function of time, smoothed at the corners.
    v = np.minimum(1.0, np.minimum(u, 1.0 - u) / ease)
    v = 0.5 - 0.5 * np.cos(np.pi * v)
    arc = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2)])
    arc = arc / arc[-1] * s[-1]
    return np.stack([np.interp(arc, s, points[:, i]) for i in range(points.shape[1])], axis=1)


def interpolate_axes(axes: np.ndarray, s_src: np.ndarray, s_dst: np.ndarray) -> np.ndarray:
    out = np.stack([np.interp(s_dst, s_src, axes[:, i]) for i in range(3)], axis=1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def smooth_axes(axes: np.ndarray, s: np.ndarray, sigma_m: float = AXIS_SIGMA_M) -> np.ndarray:
    """Gaussian-average unit axes over arc length: measured normals jitter by degrees from point to
    point, and a pen that followed them would twitch its wrist; the skin curves far more gently."""
    w = np.exp(-0.5 * ((s[:, None] - s[None, :]) / sigma_m) ** 2)
    out = w @ axes
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def tilt_capped(axes: np.ndarray, max_tilt: float = MAX_TILT_RAD) -> np.ndarray:
    """Pen axes leaned back toward straight down where the skin's normal tips past ``max_tilt``: on the
    forearm's flank the full normal is out of reach, and a hover only needs the working point on the ink."""
    down = np.array([0.0, 0.0, -1.0])
    out = np.array(axes, float)
    for i, a in enumerate(out):
        tilt = float(np.arccos(np.clip(a @ down, -1.0, 1.0)))
        if tilt > max_tilt:
            side = a - (a @ down) * down
            side /= np.linalg.norm(side)
            out[i] = np.cos(max_tilt) * down + np.sin(max_tilt) * side
    return out


def rate_limit_axes(start: np.ndarray, axes: np.ndarray, rate_rad_s: float = AXIS_RATE) -> np.ndarray:
    """Each sample's axis turned from the previous toward the requested one by at most the rate allows:
    where a new path's axes disagree with where the pen points now, the wrist turns there gradually."""
    step = rate_rad_s / RATE_HZ
    out = np.empty_like(axes)
    prev = start / np.linalg.norm(start)
    for i, want in enumerate(axes):
        angle = float(np.arccos(np.clip(prev @ want, -1.0, 1.0)))
        if angle > step:
            ortho = want - (want @ prev) * prev
            ortho /= np.linalg.norm(ortho)
            want = np.cos(step) * prev + np.sin(step) * ortho
        out[i] = prev = want / np.linalg.norm(want)
    return out


@dataclass
class Plan:
    """A joint stream at the control rate, and what it is meant to do."""

    q: np.ndarray  # (T, 6)
    targets: np.ndarray  # (T, 3) working-point targets
    axes: np.ndarray  # (T, 3) pen axes (toward the skin)
    phase: np.ndarray  # (T,) 0 approach, 1 trace, 2 retreat
    position_error_m: float  # worst IK residual
    axis_error_rad: float
    peak_joint_speed: float  # rad/s

    @property
    def duration_s(self) -> float:
        return len(self.q) / RATE_HZ


class PlanError(RuntimeError):
    pass


def plan_trace(kin: ArmKinematics, q_start: np.ndarray, q_rest: np.ndarray, points: np.ndarray,
               normals: np.ndarray, *, speed_m_s: float = SPEED_M_S, clearance_m: float = CLEARANCE_M) -> Plan:
    """Move (in joint space) above the first point, descend along its normal, follow the points, rise.

    ``points`` lie on the skin with their outward ``normals``.
    """
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    axes = tilt_capped(smooth_axes(-normals, s))  # the pen points into the skin
    normals = -axes
    above_first = points[0] + clearance_m * normals[0]
    above_last = points[-1] + clearance_m * normals[-1]
    reach = kin.solve(q_start, above_first, axes[0], q_rest, iters=200)
    if not reach.ok(position_tol=0.002, axis_tol=0.05):
        raise PlanError(f"cannot reach above the stroke: {reach.position_error * 1000:.1f} mm / "
                        f"{np.degrees(reach.axis_error):.1f} deg off")
    approach = plan_move(q_start, reach.q)

    pieces = []  # (positions, axes, phase)
    descend = timed(np.stack([above_first, points[0]]), TRAVEL_SPEED_M_S)[1:]
    pieces.append((descend, np.repeat(axes[:1], len(descend), 0), 0))
    trace = timed(points, speed_m_s)[1:]
    s_trace = np.array([s[np.argmin(np.linalg.norm(points - p, axis=1))] for p in trace])
    pieces.append((trace, interpolate_axes(axes, s, s_trace), 1))
    rise = timed(np.stack([points[-1], above_last]), TRAVEL_SPEED_M_S)[1:]
    pieces.append((rise, np.repeat(axes[-1:], len(rise), 0), 2))

    targets = np.concatenate([p for p, _, _ in pieces])
    target_axes = rate_limit_axes(axes[0], np.concatenate([a for _, a, _ in pieces]))
    phase = np.concatenate([np.full(len(p), ph) for p, _, ph in pieces])
    plan = _stream(kin, reach.q, targets, target_axes, phase)
    # The approach is joint space; its working points are its FK, for the record.
    fk = [kin.gap_pose(qa) for qa in approach]
    plan = Plan(q=np.vstack([approach, plan.q]), targets=np.vstack([[p for p, _ in fk], plan.targets]),
                axes=np.vstack([[x for _, x in fk], plan.axes]),
                phase=np.concatenate([np.zeros(len(approach), int), plan.phase]),
                position_error_m=plan.position_error_m, axis_error_rad=plan.axis_error_rad, peak_joint_speed=0.0)
    return _checked(plan, q_start)


def plan_follow(kin: ArmKinematics, q_now: np.ndarray, points: np.ndarray, normals: np.ndarray, *,
                speed_m_s: float = SPEED_M_S) -> Plan:
    """Already hovering: carry on from where the working point is, through ``points``, ending hovering.

    The first leg joins the working point to the first point; a decider that starts on the far side of
    the pen gets a straight hover move there (phase 0), then the points themselves (phase 1)."""
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    here, here_axis = kin.gap_pose(q_now)
    path = np.vstack([here, points])
    axes = np.vstack([here_axis, -normals])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    axes = tilt_capped(smooth_axes(axes, s))
    targets = timed(path, speed_m_s)[1:]
    s_t = np.array([s[np.argmin(np.linalg.norm(path - p, axis=1))] for p in targets])
    target_axes = rate_limit_axes(here_axis, interpolate_axes(axes, s, s_t))
    phase = np.where(s_t <= seg[0] + 1e-9, 0, 1)
    return _checked(_stream(kin, q_now, targets, target_axes, phase), q_now)


def _stream(kin: ArmKinematics, q0: np.ndarray, targets: np.ndarray, axes: np.ndarray, phase: np.ndarray) -> Plan:
    """IK for every sample, seeded from and held near the previous one: the least joint motion, so the free
    roll about the pen axis cannot flip between solutions from one sample to the next."""
    q = np.empty((len(targets), 6))
    q_prev, worst_pos, worst_axis = np.asarray(q0, float), 0.0, 0.0
    for i, (target, axis) in enumerate(zip(targets, axes, strict=True)):
        result = kin.solve(q_prev, target, axis, q_prev, iters=8)
        worst_pos, worst_axis = max(worst_pos, result.position_error), max(worst_axis, result.axis_error)
        q[i] = q_prev = result.q
    return Plan(q=q, targets=targets, axes=axes, phase=phase, position_error_m=worst_pos,
                axis_error_rad=worst_axis, peak_joint_speed=0.0)


def _checked(plan: Plan, q_start: np.ndarray) -> Plan:
    """The plan with its peak joint speed, refused if the IK missed or a joint would go too fast."""
    plan.peak_joint_speed = float((np.abs(np.diff(np.vstack([q_start, plan.q]), axis=0)).max(axis=1) * RATE_HZ).max())
    if plan.position_error_m > 0.002 or plan.axis_error_rad > 0.05:
        raise PlanError(f"unreachable: IK residual {plan.position_error_m * 1000:.1f} mm / "
                        f"{np.degrees(plan.axis_error_rad):.1f} deg")
    if plan.peak_joint_speed > MAX_JOINT_SPEED:
        raise PlanError(f"a joint would turn at {plan.peak_joint_speed:.2f} rad/s (cap {MAX_JOINT_SPEED})")
    return plan


def plan_move(q_from: np.ndarray, q_to: np.ndarray, max_speed: float = 0.25) -> np.ndarray:
    """A joint-space move at the control rate, eased, no joint faster than ``max_speed`` rad/s."""
    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    span = np.abs(q_to - q_from).max()
    duration = max(1.0, 2.0 * span / max_speed)  # cosine easing peaks at twice the mean speed
    u = np.linspace(0.0, 1.0, int(np.ceil(duration * RATE_HZ)) + 1)[1:]
    return q_from + (0.5 - 0.5 * np.cos(np.pi * u))[:, None] * (q_to - q_from)


def wall_crossing(kin: ArmKinematics, q_traj: np.ndarray, wall_y: float = WALL_Y_M) -> str | None:
    """Why ``q_traj`` would take the blue arm toward the pink arm's side of the wall, or None."""
    from tatbot_travel.scene import ARM, GAP_SITE, LENS_SITE, OPTICAL_FRAME, PEN_BODY

    model, data = kin.model, kin.data
    bodies = [model.body(f"{ARM}/{name}").id for name in WALL_BODIES] + [model.body(PEN_BODY).id,
                                                                       model.body(OPTICAL_FRAME).id]
    sites = [model.site(LENS_SITE).id, model.site(GAP_SITE).id]
    limit = wall_y + WALL_MARGIN_M
    for i in list(range(0, len(q_traj), 3)) + [len(q_traj) - 1]:
        kin.gap_pose(q_traj[i])  # sets the model's kinematics at this pose
        lowest = min(data.xpos[bodies, 1].min(), data.site_xpos[sites, 1].min())
        if lowest < limit:
            return f"pose {i} reaches y = {lowest:+.3f} m, past the wall at {wall_y:+.2f} m (+{WALL_MARGIN_M} margin)"
    return None

