"""Station reachability and wrist clearance along the executor's planned path.

Tool contact envelopes are checked by halo.meets; the wrist bodies must also
clear the overhead camera's post by CLEARANCE_M. The shared measured station
model places that post using the arm registration. Joint/Cartesian paths use
tatbot_motion and tatbot_session.ready, matching executor motion.
"""
from __future__ import annotations

import math

import numpy as np
from tatbot_motion.station import POST_RADIUS_M, Post, overhead_post  # noqa: F401 -- existing calibration API

from tatbot_calib import halo

# The wrist's bodies as (frame suffix, sphere radius, m): each face of the wrist tag cube (tags of 47 mm on a cube
# about 40 mm across, off the finger carriage), the wrist D405 and its bracket, the wrist flange and the forearm.
WRIST_BODIES = (("wrist_tag", 0.035), ("realsense_color_optical_frame", 0.040), ("link_6", 0.045),
                ("link_5", 0.045), ("link_4", 0.050))
CLEARANCE_M = 0.015          # a wrist body's surface keeps this far from the post's
APPROACH_OVER_M = 0.030      # S1 checks the way to the tool upright this far over the ball before it touches
REST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5708, 0.0])   # the arm landed: where each goal's joint path starts


def wrist_bodies(kin, arm: str) -> list[tuple[str, float]]:
    """(frame, radius) for each of the wrist's bodies the arm's kinematics has."""
    out = []
    for suffix, radius in WRIST_BODIES:
        names = ([f"{arm}/{suffix}{i}" for i in range(16)] if suffix == "wrist_tag" else [f"{arm}/{suffix}"])
        for name in names:
            try:
                kin.frame(REST, name)
            except (KeyError, ValueError, IndexError, RuntimeError):
                continue
            out.append((name, radius))
    return out


def wrist_clearance(kin, q, bodies, posts) -> float:
    """The smallest gap (m) between a wrist body's sphere and a post's surface at joints q; +inf without posts. A
    body over a post's top is clear of it."""
    gap = math.inf
    for name, radius in bodies:
        centre = kin.frame(np.asarray(q, float), name)[:3, 3]
        for post in posts:
            if centre[2] - radius > post.top_m:
                continue
            gap = min(gap, float(np.hypot(centre[0] - post.xy[0], centre[1] - post.xy[1])) - radius - post.radius_m)
    return gap


def planned_path(kin, motion: dict, q, target: np.ndarray, rate_hz: float = 100.0) -> np.ndarray:
    """The joints, row by row, along the way the executor plans to `target` under the probe guard: the joint move
    over it from rest on the limits, then the Cartesian planner's travel through the clearance plane
    (tatbot_session.ready.station_way, tatbot_motion.plan_travel). A touch's own leg past its start is not in it.
    Raises ValueError (no joints over the start, or a joint move the executor refuses for dipping) or tatbot_motion's
    PlanError (the planner refuses the travel)."""
    from tatbot_motion import plan_travel
    from tatbot_session import ready

    q = np.asarray(q, float)
    q_over, via = ready.station_way(kin, motion, q, target)
    rows = [q[None, :]]
    if q_over is not None:
        move = ready.pen_up_joint_move(kin, motion, q, q_over, rate_hz)
        low = ready.station_dip(kin, move, q, q_over)
        if low is not None:   # the executor refuses it (2026-09-29: round 5's first views, round 6's first)
            raise ValueError(f"the joint move over the start would take the tip down to z {low:.3f} m")
        rows.append(np.asarray(move.q))
        q = q_over
    rows.append(np.asarray(plan_travel(q_seed=q, base_from_tcp=target, kin=kin, motion=motion, via=via).q))
    return np.vstack(rows)


def path_clearance(kin, rows, bodies, posts, every: int = 5) -> tuple[float, np.ndarray]:
    """The smallest wrist_clearance along rows of joints (every `every`th and the last), and the row it is at."""
    rows = np.asarray(rows, float)
    picked = list(rows[::every]) + [rows[-1]]
    gaps = [wrist_clearance(kin, row, bodies, posts) for row in picked]
    worst = int(np.argmin(gaps))
    return gaps[worst], picked[worst]


def _margin(kin, pose: np.ndarray, bodies, posts, guard=None, arm: str = "", station=None) -> float:
    """The joint margin (rad) the arm keeps at `pose` from rest, or -1 when it has no IK there, its wrist comes
    within CLEARANCE_M of a post, (with the stack's tatbot_motion.collision.Guard) it stands within the guard's
    margin of the other arm, which the executor will not go near, or `station` (joints -> bool) says its bodies are
    not clear of the station's parts."""
    from tatbot_session import ready

    try:
        q = ready.solve_ik_seeded(kin, pose, REST)
    except ValueError:
        return -1.0
    if wrist_clearance(kin, q, bodies, posts) < CLEARANCE_M:
        return -1.0
    if guard is not None and guard.scene.clearance({**guard.others(arm, {})[0], arm: q}).distance_m < guard.margin_m:
        return -1.0
    if station is not None and not station(q):
        return -1.0
    return float(np.min(np.minimum(q[:6] - np.asarray(kin.lower)[:6], np.asarray(kin.upper)[:6] - q[:6])))


def choose_heading(kin, arm: str, ball: np.ndarray, attitudes, margin_rad: float, posts=(),
                   guard=None, station=None) -> tuple[float, list]:
    """The base heading (degrees, every 15) whose attitudes the arm reaches over the ball from rest, keeping the
    planner's joint margin and its wrist clear of the posts: S1 upright, then each S4 (yaw, tilt) about the heading.
    The most of them, then the widest margin. Returns (the heading, the attitudes it cannot reach). The station
    stands where the scene puts it (0.33 m out and toward the blue arm on the demo table, 2026-09-29), so a fixed
    heading can leave S1 itself out of reach, or turn the wrist into the camera's post. S1's views are chosen apart
    from it (Calibration.probe_views)."""
    wanted = [(0.0, 0.0)] + [tuple(a) for a in attitudes]
    bodies = wrist_bodies(kin, arm) if posts else []
    best = None
    for heading in np.arange(-180.0, 180.0, 15.0):
        reached, margins = [], []
        for yaw, tilt in wanted:   # S1 upright first: a heading that misses it is no base
            margin = _margin(kin, halo.tool_pose(ball, math.radians(heading + yaw), math.radians(tilt)), bodies,
                             posts, guard, arm, station)
            if margin >= margin_rad:
                reached.append((yaw, tilt))
                margins.append(margin)
            elif (yaw, tilt) == (0.0, 0.0):
                break
        score = (len(reached), min(margins, default=0.0))
        if (0.0, 0.0) in reached and (best is None or score > best[0]):
            best = (score, float(heading), [a for a in wanted if a not in reached])
    if best is None:
        raise RuntimeError(f"the arm reaches the station's ball at {np.round(ball, 4).tolist()} upright at no heading")
    return best[1], best[2]
