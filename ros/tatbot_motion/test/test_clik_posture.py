"""CLIK on a chain longer than the task's six (the xArm 7): its self-motion follows its model's posture through the
task's null space, and the tip never feels it."""
import numpy as np
import pytest
import tatbot_motion as tm
from tatbot_description import arm_model, robot_description
from tatbot_motion.clik import clik

PERIOD = 0.0025


@pytest.fixture(scope="module")
def xarm():
    return tm.Kinematics(robot_description(arms=("xarm",)), "xarm")


@pytest.fixture(scope="module")
def motion():
    return tm.load_motion()


def _hold(kin, q0, seconds):
    """(positions, velocities, rotations, pen) holding the tip where q0 puts it."""
    n = int(round(seconds / PERIOD))
    pose = kin.fk(q0)
    return (np.repeat(pose[None, :3, 3], n, axis=0), np.zeros((n, 3)), np.repeat(pose[None, :3, :3], n, axis=0),
            np.zeros(n, bool))


def test_a_held_tip_lets_the_arm_roll_back_to_its_posture(xarm, motion):
    """The tip held where rest puts it, the arm started 0.2 rad along its one self-motion: the posture pulls it back to
    rest along that self-motion. A straight step along the self-motion's tangent leaves its curve, ~0.9 mm at the tip,
    which the task closes in its first second; from then the tip holds within 0.02 mm while the arm rolls back."""
    rest = np.array(arm_model.for_arm("xarm").rest)
    _, _, vt = np.linalg.svd(xarm.jacobian(rest))
    q0 = rest + 0.2 * vt[-1]                          # the Jacobian's null direction at rest
    p, v, rot, pen = _hold(xarm, rest, 4.0)
    out = clik(xarm, q0, p, v, rot, pen, PERIOD, motion["clik"], motion["carriage"])
    assert np.abs(out.q[-1] - rest).max() < 0.01 < np.abs(q0 - rest).max()
    second = int(round(1.0 / PERIOD))
    assert out.model_error_m[second:].max() < 2e-5 and out.orientation_error_rad[second:].max() < 5e-4


def test_a_line_across_the_page_keeps_the_wrist_off_its_singularity_and_inside_its_windows(xarm, motion):
    """A 150 mm stroke across the front of the arm at 10 mm/s from rest: the task tracked, joint6 bent well off 0
    and every wrist joint inside its model's range for the stack's own moves."""
    model = arm_model.for_arm("xarm")
    q0 = np.array(model.rest)
    start = xarm.fk(q0)
    n = int(round(0.150 / 0.010 / PERIOD))
    p = start[:3, 3] + np.outer(np.linspace(0.0, 0.150, n), [0.0, 1.0, 0.0])
    v = np.gradient(p, PERIOD, axis=0)
    rot = np.repeat(start[None, :3, :3], n, axis=0)
    out = clik(xarm, q0, p, v, rot, np.ones(n, bool), PERIOD, motion["clik"], motion["carriage"])
    assert out.model_error_m.max() < 2e-4
    wrist = list(model.wrist)
    low, high = np.asarray(model.wrist_range_rad).T
    assert np.all(out.q[:, wrist] >= low) and np.all(out.q[:, wrist] <= high)
    assert np.abs(out.q[:, 5]).min() > 0.5


def test_a_tip_held_far_from_the_posture_stays_put_while_the_arm_turns_toward_it(xarm, motion):
    """The joints of an xArm dip's descent into the palette's L2 (2026-11-09), j4 and j6 0.4-0.9 rad off the posture:
    the hold that is the dip's dwell keeps the tip while the pull turns the joints. Through a damped null space the pull
    leaked into the tip, which drifted 0.6 mm in the second and failed the dwell."""
    q0 = np.array([0.3612, 0.1891, 0.1383, 0.4846, -0.0759, 0.2973, 0.573])
    p, v, rot, pen = _hold(xarm, q0, 1.0)
    out = clik(xarm, q0, p, v, rot, pen, PERIOD, motion["clik"], motion["carriage"])
    assert np.abs(out.q[-1] - q0).max() > 0.005   # the pull did turn the arm
    assert out.model_error_m.max() < 2e-5 and out.orientation_error_rad.max() < 5e-4


def test_a_lift_plans_from_a_pose_a_radian_off_the_posture(xarm, motion):
    """Where a wrist-camera calibration's hold left the arm on 2026-11-13, joints 1, 3 and 7 over a radian from the
    posture: the pull alone ran joint 3 at 1.06 rad/s in the plan's first 0.14 s, past the 1 rad/s pen-up cap at
    every slow-down, so neither the 15 mm lift before landing nor the landing's own lift planned. The self-motion is
    held to its model's max_rad_s, the lift plans, and the tip stays on its line."""
    held = np.array([1.1817327737808228, 0.5679506063461304, -1.121138572692871, 0.9498639106750488,
                     0.9093840718269348, 1.0460790395736694, -1.0296635627746582])
    bound = arm_model.for_arm("xarm").posture.max_rad_s
    assert np.abs(np.array(arm_model.for_arm("xarm").posture.target) - held).max() > 1.0
    for direction, distance in (([0.0, 0.0, 1.0], 0.015), (-xarm.fk(held)[:3, 2], 0.010)):
        traj = tm.plan_lift(q_seed=held, direction=np.asarray(direction), distance_m=distance, kin=xarm, motion=motion)
        assert traj.info["slowdowns"] == {}
        assert traj.info["max_joint_speed_rad_s"] < bound + 0.05 < motion["joint_speed"]["pen_up_rad_s"]
        line = traj.tip - traj.tip[0]
        assert np.linalg.norm(line[-1]) == pytest.approx(distance, abs=2e-5)
