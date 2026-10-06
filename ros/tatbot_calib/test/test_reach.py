"""The arm at the demo station, with the pink arm's own kinematics: the base heading whose attitudes it reaches, and
its wrist against the post the overhead D555 stands on. On 2026-09-29 the S1 view turned to +30 degrees put the wrist
tag cube against that post (operator); the view at -30 touched nothing. The model says both."""
from __future__ import annotations

import math

import numpy as np
import pytest
from tatbot_calib import halo, program, reach

tatbot_motion = pytest.importorskip("tatbot_motion")

# The demo station before the post was moved (2026-09-29): the D555's fix of the ball, the D555's optical centre, both
# in right/base_link, and the three S1 views (heading -30, turns 0/30/60) as the stack measured their joints.
BALL = np.array([0.0951, 0.3117, 0.0760])
CAMERA = np.array([0.0268, 0.3628, 0.7395])
VIEWS = {-30.0: [1.5215, 0.9302, 0.6773, -0.5907, 0.329, 1.919, 0.002],
         0.0: [0.9581, 0.9726, 0.701, -0.6136, -0.4187, 1.1095, 0.002],
         30.0: [0.717, 1.4223, 1.0359, -0.9947, -0.7647, 0.2707, 0.002]}
# After the post and the palette were moved and the arm was registered again (2026-09-29).
MOVED_BALL, MOVED_CAMERA = np.array([0.1333, 0.3356, 0.0706]), np.array([-0.0133, 0.3522, 0.7265])
# The arm where a free-air probe touch left it, 0.3 m out and 0.21 m up (2026-09-29).
FREE_AIR = np.array([-0.0116, 0.0814, 0.1379, -0.0731, -0.0124, 1.5703, 0.0020])


@pytest.fixture(scope="module")
def kin():
    from tatbot_description import robot_description

    return tatbot_motion.Kinematics(robot_description(None, arms=("right",)), "right")


def test_the_wrist_met_the_post_turned_to_30_degrees(kin):
    post, bodies = reach.overhead_post(CAMERA), reach.wrist_bodies(kin, "right")
    assert {name for name, _ in bodies} >= {"right/link_6", "right/realsense_color_optical_frame", "right/wrist_tag4"}
    gaps = {turn: reach.wrist_clearance(kin, q, bodies, [post]) for turn, q in VIEWS.items()}
    assert gaps[30.0] < -0.05 and gaps[30.0] < gaps[0.0] < gaps[-30.0]      # the cube on the post, deepest at +30
    assert reach.wrist_clearance(kin, VIEWS[30.0], bodies, []) == math.inf  # no post, nothing to meet


def test_the_wrist_is_checked_along_the_way_the_executor_plans(kin):
    """reach.planned_path is the executor's way to a station goal: from rest, the joint move over it and the
    planner's travel down. At the moved station the view turned to +30 degrees still takes the wrist into the post
    and the others clear it. From where a free-air touch left the arm, 0.3 m out, the planner's crossing meets
    joint_1's limit, as the executor's would (2026-09-29: a stepped IK stalled there 16.7 mm short and ended the
    run before any motion)."""
    from tatbot_motion import load_motion
    from tatbot_motion.clik import PlanError

    motion = load_motion()
    post, bodies = reach.overhead_post(MOVED_CAMERA), reach.wrist_bodies(kin, "right")
    over = MOVED_BALL + [0.0, 0.0, 0.030]
    gaps = {}
    for turn in (-30.0, 0.0):
        rows = reach.planned_path(kin, motion, reach.REST, halo.tool_pose(over, math.radians(turn)))
        assert np.allclose(rows[0], reach.REST) and np.linalg.norm(kin.fk(rows[-1])[:3, 3] - over) < 1e-3
        gaps[turn] = reach.path_clearance(kin, rows, bodies, [post])[0]
    assert min(gaps.values()) >= reach.CLEARANCE_M, gaps
    assert reach.path_clearance(kin, rows, bodies, [])[0] == math.inf
    # turned toward the post, +20 takes the wrist within 10 mm of it (+15 did through the 09-19 tag layout, whose cube
    # stood ~10 mm off its seat; the seated one clears it by 15.9 mm); +30 from rest the executor refuses, the joint
    # move over it dipping (ready.station_dip)
    rows = reach.planned_path(kin, motion, reach.REST, halo.tool_pose(over, math.radians(20.0)))
    assert reach.path_clearance(kin, rows, bodies, [post])[0] < reach.CLEARANCE_M
    with pytest.raises(ValueError, match="would take the tip down"):
        reach.planned_path(kin, motion, reach.REST, halo.tool_pose(over, math.radians(30.0)))
    with pytest.raises(PlanError, match="joint_1"):
        reach.planned_path(kin, motion, FREE_AIR, halo.tool_pose(over, 0.0))


def test_a_goal_into_the_post_or_without_a_plan_is_not_sent(kin):
    from types import SimpleNamespace

    from tatbot_description import repo_root
    from tatbot_motion import load_motion

    joints = {"q": reach.REST}
    rig = SimpleNamespace(joints=lambda: np.asarray(joints["q"], float))
    cal = program.Calibration(rig, SimpleNamespace(dir=None), repo_root(None), "right",
                              halo.Halo(0.00135, 0.0004, 0.0007, 0.002), 0.0, kin)
    cal.way_check("view 30", [halo.tool_pose(MOVED_BALL, math.radians(30.0))])   # no post, nothing to check
    cal.posts, cal.bodies, cal.motion = [reach.overhead_post(MOVED_CAMERA)], reach.wrist_bodies(kin, "right"), load_motion()
    view = {turn: halo.tool_pose(MOVED_BALL + [0.0, 0.0, 0.030], math.radians(turn)) for turn in (0.0, 20.0, 30.0)}
    cal.way_check("view 0", [view[0.0]])
    with pytest.raises(RuntimeError, match="view 20: the wrist would pass"):
        cal.way_check("view 20", [view[20.0]])
    with pytest.raises(RuntimeError, match="view 30: no plan to it .*would take the tip down"):
        cal.way_check("view 30", [view[30.0]])
    joints["q"] = FREE_AIR
    with pytest.raises(RuntimeError, match="view 0: no plan to it"):
        cal.way_check("view 0", [view[0.0]])


def test_the_base_heading_reaches_the_attitudes_and_keeps_the_wrist_off_the_post(kin):
    """Without the post the blue arm's 30 degrees would have put the third view and the +60 yaw out of reach: the
    heading is chosen from the arm's reach. With the post, the chosen attitudes keep the wrist clear of it, and in
    the moved layout S1 and at least three S4 attitudes remain."""
    from tatbot_session import ready

    attitudes = program._attitudes(program.DEFAULT_ATTITUDES)
    heading, unreached = reach.choose_heading(kin, "right", BALL, program._attitudes("-30/0,30/0,60/0,0/15,0/-15"),
                                              0.0333)
    assert unreached == [] and -75.0 <= heading <= 0.0
    post, bodies = reach.overhead_post(MOVED_CAMERA), reach.wrist_bodies(kin, "right")
    heading, unreached = reach.choose_heading(kin, "right", MOVED_BALL, attitudes, 0.0333, [post])
    kept = [a for a in attitudes if a not in unreached]
    assert (0.0, 0.0) not in unreached and len(kept) >= 3
    for yaw, tilt in kept:
        q = ready.solve_ik_seeded(kin, halo.tool_pose(MOVED_BALL, math.radians(heading + yaw), math.radians(tilt)),
                                  reach.REST)
        assert reach.wrist_clearance(kin, q, bodies, [post]) >= reach.CLEARANCE_M, (heading, yaw, tilt)
    with pytest.raises(RuntimeError, match="no heading"):
        reach.choose_heading(kin, "right", np.array([0.90, 0.50, 0.05]), attitudes, 0.0333)



def test_the_heading_keeps_the_arms_bodies_clear_of_the_station(kin):
    """A heading whose poses the station check refuses is not chosen: 2026-09-30, from heading 0 the blue gripper's
    pads cleared the roof tag's box by -5 mm, from +90 by 48."""
    attitudes = program._attitudes("0/15")
    free, _ = reach.choose_heading(kin, "right", BALL, attitudes, 0.0333)
    refused = []

    def station(q):
        heading = round(math.degrees(math.atan2(kin.fk(q)[1, 0], kin.fk(q)[0, 0])))
        refused.append(heading)
        return abs(heading - round(math.degrees(math.atan2(kin.fk(ready_q(kin, free))[1, 0],
                                                            kin.fk(ready_q(kin, free))[0, 0])))) > 20

    chosen, _ = reach.choose_heading(kin, "right", BALL, attitudes, 0.0333, station=station)
    assert refused and chosen != free


def ready_q(kin, heading):
    from tatbot_session import ready

    return ready.solve_ik_seeded(kin, halo.tool_pose(BALL, math.radians(heading)), reach.REST)
