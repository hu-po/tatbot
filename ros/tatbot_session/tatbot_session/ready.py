"""The joint-space move to the ready pose: the tool over the page centre at the standoff, pointing into
the paper. The arm rests at its staged/sleep pose with joint_1 and joint_2 on their lower limits, where
the Cartesian planner (CLIK with a joint-limit margin) cannot start. Pure numpy on tatbot_motion.Kinematics (fk, jacobian, limits).
"""
from __future__ import annotations

import numpy as np
from tatbot_motion.timelaw import rotation_log

MARGIN_RAD = 0.0667  # twice motion.yaml clik.joint_limit_margin_rad: the seed sits well inside the guard


def _rotation_error(r_target: np.ndarray, r_now: np.ndarray) -> np.ndarray:
    """Axis-angle vector (base frame) taking r_now to r_target."""
    axis, angle = rotation_log(r_target @ r_now.T)
    return axis * angle


def solve_ik(kin, target: np.ndarray, q_seed: np.ndarray, *, iterations: int = 400, damping: float = 0.02,
             tolerance_m: float = 1e-5, tolerance_rad: float = 1e-4) -> np.ndarray:
    """Damped-least-squares IK of the revolute joints to a base_from_tcp pose; the carriage holds its
    seed. The seed is first clipped MARGIN_RAD inside the limits. Raises ValueError when it does not
    converge."""
    lower, upper = np.asarray(kin.lower), np.asarray(kin.upper)
    q = np.asarray(q_seed, dtype=float).copy()
    q[:6] = np.clip(q[:6], lower[:6] + MARGIN_RAD, upper[:6] - MARGIN_RAD)
    for _ in range(iterations):
        now = kin.fk(q)
        err = np.concatenate([target[:3, 3] - now[:3, 3], _rotation_error(target[:3, :3], now[:3, :3])])
        if np.linalg.norm(err[:3]) < tolerance_m and np.linalg.norm(err[3:]) < tolerance_rad:
            return q
        jac = np.asarray(kin.jacobian(q))[:, :6]
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
        q[:6] = np.clip(q[:6] + step, lower[:6] + MARGIN_RAD / 2, upper[:6] - MARGIN_RAD / 2)
    raise ValueError(f"no IK solution for the ready pose (residual {np.linalg.norm(err[:3]) * 1000:.2f} mm)")


def solve_ik_seeded(kin, target: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
    """solve_ik from q_seed, else from the middle of the revolute limits (the carriage kept). From rest on the
    limits the damped solve can stall short of a pose it reaches from elsewhere (2026-09-28: 8.9 mm left over
    the station on the blue arm, which the same pose from a rest a little different reached)."""
    try:
        return solve_ik(kin, target, q_seed)
    except ValueError:
        mid = np.asarray(q_seed, dtype=float).copy()
        mid[:6] = (np.asarray(kin.lower)[:6] + np.asarray(kin.upper)[:6]) / 2.0
        return solve_ik(kin, target, mid)


HEADING_TIE_RAD = 0.05   # headings whose joint-limit margins are this close count as tied (the first is taken)
# A heading near the dips' is taken when its margin over the page's five points reaches this: every dip turns the
# wrist from the strokes' heading to the station touch's and back at 8 deg/s, 75-82 deg each way and ~36 s a dip over
# a forearm whose strokes went at -105 deg (owner, 2026-10-20: "getting rid of that null space move"). There the
# touch's -30 deg read 0.32 rad, and the whole curved design planned at -30 and -45 deg with 0.12-0.44 rad left, the
# five points reading 0.25-0.31 rad more than a design whose pen turns to the skin.
PREFER_MARGIN_RAD = 0.45


def _heading(page: np.ndarray, yaw: float) -> np.ndarray:
    heading = np.eye(4)
    heading[:3, 0] = np.cos(yaw) * page[:3, 0] + np.sin(yaw) * page[:3, 1]
    return heading


def heading_yaw(page: np.ndarray, rotation: np.ndarray) -> float:
    """A tcp rotation's heading (its x axis) as a turn in the page's plane from the page's x axis."""
    x = np.asarray(rotation)[:3, 0]
    return float(np.arctan2(x @ page[:3, 1], x @ page[:3, 0]))


def pen_heading(kin, page: np.ndarray, clear_m, standoff_m: float, q_seed: np.ndarray,
                prefer: np.ndarray | None = None) -> np.ndarray | None:
    """A tcp pose whose x axis is the pen heading (its turn about its own axis) that keeps every joint farthest
    from its limits with the tool down at the standoff over five points of the page's clear area: 24 headings
    15 deg apart, the first tied (HEADING_TIE_RAD) with the best. None when no heading reaches all five. With
    `prefer` (a tcp rotation: the dips' heading) its heading on the page comes first, then 15 deg steps either side
    of it, the better of each pair: the first whose margin reaches PREFER_MARGIN_RAD, else the search. The executor
    searches a new page's heading with it, and the simulator its sampled page's."""
    from tatbot_session.geometry import tool_down_pose

    hx, hy = (0.45 * v for v in clear_m)
    points = [(0.0, 0.0), (-hx, -hy), (hx, -hy), (-hx, hy), (hx, hy)]
    lower, upper = np.asarray(kin.lower)[:6], np.asarray(kin.upper)[:6]

    def margin(heading) -> float:
        least = np.inf
        for xy in points:
            try:
                q = solve_ik_seeded(kin, tool_down_pose(page, xy, standoff_m, heading), q_seed)
            except ValueError:
                return -np.inf
            least = min(least, float(np.min(np.minimum(q[:6] - lower, upper - q[:6]))))
        return least

    if prefer is not None:
        yaw = heading_yaw(page, prefer)
        for k in range(13):
            steps = (0.0,) if k == 0 else (180.0,) if k == 12 else (15.0 * k, -15.0 * k)
            score, heading = max(((margin(h), h) for h in (_heading(page, yaw + np.radians(d)) for d in steps)),
                                 key=lambda scored: scored[0])
            if score >= PREFER_MARGIN_RAD:
                return heading
    scored = [(margin(h), h) for h in (_heading(page, yaw) for yaw in np.radians(np.arange(0.0, 360.0, 15.0)))]
    best = max(m for m, _ in scored)
    if not np.isfinite(best):
        return None
    return next(h for m, h in scored if m >= best - HEADING_TIE_RAD)


def station_way(kin, motion: dict, q: np.ndarray, start: np.ndarray) -> tuple[np.ndarray | None, list]:
    """A goal's way to its start under the probe guard: up to the clearance plane (motion.yaml probe.clearance_m
    over the start, or where the tip already is when higher), across it and down, never a straight line past the
    ball. From rest on the joint limits, where the Cartesian planner cannot start, a joint-space move to the joints
    over the start comes first (pen_up_joint_move). Returns (those joints, else None; the via points for the
    Cartesian planner). The executor moves by it, and the station calibration plans the same way to keep its
    wrist clear of the overhead camera's post."""
    q = np.asarray(q, dtype=float)
    here, goal = kin.fk(q)[:3, 3], np.asarray(start, float)[:3, 3]
    if np.linalg.norm(here - goal) <= float(motion["dispatch_drift"]["start_tolerance_m"]):
        return None, []
    z = max(float(here[2]), float(goal[2]) + float(motion["probe"]["clearance_m"]))
    margin = float(motion["clik"]["joint_limit_margin_rad"])
    lower, upper = np.asarray(kin.lower)[:6], np.asarray(kin.upper)[:6]
    if np.min(np.minimum(q[:6] - lower, upper - q[:6])) < margin:
        over = np.array(start, float)
        over[2, 3] = z
        return solve_ik_seeded(kin, over, q), []   # straight over the start: the planner's travel goes down onto it
    via = [np.array([here[0], here[1], z]), np.array([goal[0], goal[1], z])]
    return None, [v for i, v in enumerate(via) if np.linalg.norm(v - ([here] + via)[i]) > 1e-4]


STATION_DIP_M = 0.010   # a joint move over the start may not take the tip this far under its lower end


def station_dip(kin, traj, q_from: np.ndarray, q_to: np.ndarray) -> float | None:
    """The lowest tip z of station_way's joint move `traj` when it dips more than STATION_DIP_M under the lower of
    its ends, which the executor refuses; else None. The station calibration checks its goals by the same rule."""
    low = float(np.min(np.asarray(traj.tip)[:, 2]))
    ends = min(float(kin.fk(np.asarray(q_from, float))[2, 3]), float(kin.fk(np.asarray(q_to, float))[2, 3]))
    return low if low < ends - STATION_DIP_M else None


# Every configured joint move (pen_up_joint_move, the locate's views) also stays under these. Under the pen-up joint
# speed alone the ready move from the last locate view (0.6-0.7 rad in 1.15-1.26 s) peaked at 2.4-2.7 rad/s^2, its
# jerk stepping to 20-24 rad/s^3 as it started and stopped, and the tip rang 0.6-0.7 mm for half a second after it;
# the drop from the ready move's high pose to the standoff (0.19 rad, 50 mm toward the skin) took the 0.5 s floor,
# 4.4 rad/s^2 and a 91 rad/s^3 step. One of 2.7 s at 1.1 rad/s^2 and 4.4 rad/s^3 stopped within 0.1 mm (2026-10-20/21).
JOINT_ACCEL_RAD_S2 = 1.0
JOINT_JERK_RAD_S3 = 5.0
QUINTIC_ACCEL = 10.0 / np.sqrt(3.0)   # a quintic's peak acceleration over delta / T^2...
QUINTIC_JERK = 60.0                   # ...and its jerk as it starts and stops, over delta / T^3
# One from rest (joints 1 and 2 within MARGIN_RAD of their lower stops) covers its largest joint's first LIFT_RAD at
# most LIFT_RAD_S, and the rest of the way starts at that quintic's peak. At rest the stops carry the arm, every motor
# near zero torque: joint_3's velocity integral sat at its clamp against its 2.4 N m gravity feedforward (2.2-2.6 N m
# the other way). Lifted off, the wrist fell ahead of its reference until the loop caught it, 10-13 mrad leaving rest
# at 0.05-0.1 rad/s and 17.6-18.2 mrad at 0.3-0.5 rad/s, the tip shaking 1.2-1.8 and 3.1-3.5 mm (2-20 Hz) as it left
# (2026-10-20/21: the first locate view's swing and three 1 rad/s ones from rest).
LIFT_RAD = 0.1
LIFT_RAD_S = 0.1


def view_move(kin, motion: dict, q_from: np.ndarray, target: np.ndarray, rate_hz: float, phase: int = 2):
    """A calibration view's way (Touch MODE_VIEW): one pen_up_joint_move to the joints solve_ik_seeded finds for the
    tcp pose `target` from q_from. The executor moves by it and the calibration checks the same way. Over the
    station's way (up, across, down) a view of the roof tag took 11-23 s, its whole turn pivoting the tool about
    its tip on the rise: joints 1-4 slid at 0.15-0.18 rad/s and joint 3 rang 8 mrad at 4.5 Hz (2026-10-22)."""
    return pen_up_joint_move(kin, motion, q_from, solve_ik_seeded(kin, target, q_from), rate_hz, phase)


def pen_up_joint_move(kin, motion: dict, q_from: np.ndarray, q_to: np.ndarray, rate_hz: float, phase: int = 2):
    """Configured pen-up joint move, shared by page travel, the locate's views and station calibration."""
    resting = np.all(np.asarray(q_from, float)[1:3] - np.asarray(kin.lower)[1:3] < MARGIN_RAD)
    return joint_move(kin.joint_names, q_from, q_to, max_rad_s=float(motion["joint_speed"]["pen_up_rad_s"]),
                      max_m_s=float(motion["carriage"]["max_m_s"]), rate_hz=rate_hz, kin=kin, phase=phase,
                      max_rad_s2=JOINT_ACCEL_RAD_S2, max_rad_s3=JOINT_JERK_RAD_S3, lift_rad=LIFT_RAD * resting)


def joint_move(joint_names, q_from: np.ndarray, q_to: np.ndarray, *, max_rad_s: float, max_m_s: float,
               rate_hz: float, kin=None, phase: int = 2, max_rad_s2: float | None = None,
               max_rad_s3: float | None = None, lift_rad: float = 0.0):
    """Quintic joint-space move from q_from to q_to, peak joint speed <= max_rad_s (carriage max_m_s), and with them
    its acceleration <= max_rad_s2 and the jerk it starts and stops with <= max_rad_s3, sampled at rate_hz; one over
    twice `lift_rad` first lifts off its rest (LIFT_RAD). Returns a tatbot_motion.Trajectory (arc NaN, the given phase)."""
    from tatbot_motion import Trajectory

    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    delta = q_to - q_from
    peak = 15.0 / 8.0  # a quintic's peak speed over its mean speed
    span = float(np.max(np.abs(delta[:6])))

    def seconds(part: float, rad_s: float) -> float:   # a quintic's over `part` of the way, under the caps
        return max(peak * part * span / rad_s, peak * part * abs(float(delta[6])) / max_m_s,
                   float(np.sqrt(QUINTIC_ACCEL * part * span / max_rad_s2)) if max_rad_s2 else 0.0,
                   float(np.cbrt(QUINTIC_JERK * part * span / max_rad_s3)) if max_rad_s3 else 0.0)

    lift = lift_rad / span if span > 2.0 * lift_rad else 0.0   # the part of the way the lift-off covers
    first, main = seconds(lift, LIFT_RAD_S), max(seconds(1.0 - lift, max_rad_s), 0.5)
    duration = first / 2.0 + main
    t = np.arange(0.0, duration, 1.0 / rate_hz)
    t = np.append(t, duration) if duration - t[-1] > 1e-9 else t
    shape = rate = 0.0   # the sum of the lift-off's quintic and the rest's, which starts at the lift-off's peak
    for part, begin, length in ((lift, 0.0, first or 1.0), (1.0 - lift, first / 2.0, main)):
        s = np.clip((t - begin) / length, 0.0, 1.0)
        shape, rate = shape + part * (10 * s**3 - 15 * s**4 + 6 * s**5), rate + part * 30 * s**2 * (1 - s) ** 2 / length
    q = q_from + shape[:, None] * delta
    qd = rate[:, None] * delta
    tip = np.array([kin.fk(row)[:3, 3] for row in q]) if kin is not None else np.full((len(t), 3), np.nan)
    return Trajectory(tuple(joint_names), t, q, qd, tip, np.full(len(t), np.nan), np.full(len(t), phase, np.uint8))
