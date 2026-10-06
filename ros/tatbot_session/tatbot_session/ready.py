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


def pen_up_joint_move(kin, motion: dict, q_from: np.ndarray, q_to: np.ndarray, rate_hz: float, phase: int = 2):
    """Configured pen-up joint move, shared by page travel and station calibration."""
    return joint_move(kin.joint_names, q_from, q_to, max_rad_s=float(motion["joint_speed"]["pen_up_rad_s"]),
                      max_m_s=float(motion["carriage"]["max_m_s"]), rate_hz=rate_hz, kin=kin, phase=phase)


def joint_move(joint_names, q_from: np.ndarray, q_to: np.ndarray, *, max_rad_s: float, max_m_s: float,
               rate_hz: float, kin=None, phase: int = 2):
    """Quintic joint-space move from q_from to q_to, peak joint speed <= max_rad_s (carriage max_m_s),
    sampled at rate_hz. Returns a tatbot_motion.Trajectory (arc NaN, the given phase)."""
    from tatbot_motion import Trajectory

    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    delta = q_to - q_from
    peak = 15.0 / 8.0  # a quintic's peak speed over its mean speed
    duration = max(peak * float(np.max(np.abs(delta[:6]))) / max_rad_s, peak * abs(float(delta[6])) / max_m_s, 0.5)
    t = np.arange(0.0, duration, 1.0 / rate_hz)
    t = np.append(t, duration) if duration - t[-1] > 1e-9 else t
    s = t / duration
    shape = 10 * s**3 - 15 * s**4 + 6 * s**5
    rate = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
    q = q_from + shape[:, None] * delta
    qd = rate[:, None] * delta
    tip = np.array([kin.fk(row)[:3, 3] for row in q]) if kin is not None else np.full((len(t), 3), np.nan)
    return Trajectory(tuple(joint_names), t, q, qd, tip, np.full(len(t), np.nan), np.full(len(t), phase, np.uint8))
