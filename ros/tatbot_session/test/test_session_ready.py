"""The ready move: IK from the staged pose (on the joint_1/joint_2 lower limits) to the tool over the
centre of the configured fixed page, and the quintic joint-space move under the pen-up joint speed."""
from __future__ import annotations

import numpy as np
import pytest
from tatbot_session import geometry, ready

STAGED = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0])


@pytest.mark.parametrize('angle', [0., .4, np.pi-1e-7, np.pi])
def test_ready_rotation_residual_reconstructs_half_turns_in_base_frame(angle):
    from scipy.spatial.transform import Rotation

    axis = np.array([1., -2., 3.]) / np.sqrt(14)
    now = Rotation.from_euler('xyz', [.3, -.4, .7]).as_matrix()
    target = Rotation.from_rotvec(angle*axis).as_matrix() @ now
    error = ready._rotation_error(target, now)
    np.testing.assert_allclose(Rotation.from_rotvec(error).as_matrix() @ now, target, atol=2e-7)
    # The trace-based angle loses O(sqrt(machine epsilon)) precision at pi.
    assert np.linalg.norm(error) == pytest.approx(angle, abs=3e-8)


@pytest.fixture(scope="module")
def kin():
    from tatbot_description import repo_root, robot_description
    from tatbot_motion import Kinematics

    try:
        return Kinematics(robot_description(repo_root(None), arms=("right",)), "right")
    except NotImplementedError:
        pytest.skip("tatbot_motion.Kinematics is not implemented yet")


def test_ready_ik_from_the_staged_pose(kin):
    from tatbot_description import load_stack

    fixed = load_stack()["page"]["fixed"]["right"]  # stack.yaml's page, read as page.source fixed reads it
    page = geometry.rpy_matrix(fixed["xyz"], fixed.get("rpy", [0.0, 0.0, 0.0]))
    target = geometry.tool_down_pose(page, [0.0, 0.0], 0.010, kin.fk(STAGED))
    q = ready.solve_ik(kin, target, STAGED)
    reached = kin.fk(q)
    assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 1e-5
    assert geometry.rotation_angle(reached, target) < 1e-4
    assert np.all(q[:6] >= kin.lower[:6] + ready.MARGIN_RAD / 2 - 1e-12)
    assert np.all(q[:6] <= kin.upper[:6] - ready.MARGIN_RAD / 2 + 1e-12)
    assert q[6] == STAGED[6]  # the carriage holds


def test_joint_move_is_a_rest_to_rest_quintic_under_the_cap(kin):
    q_to = STAGED + np.array([0.4, 1.2, 0.9, -0.3, -0.5, -0.7, 0.002])
    move = ready.joint_move(kin.joint_names, STAGED, q_to, max_rad_s=1.0, max_m_s=0.001, rate_hz=100, kin=kin)
    assert np.allclose(move.q[0], STAGED) and np.allclose(move.q[-1], q_to)
    assert np.allclose(move.qd[0], 0) and np.allclose(move.qd[-1], 0)
    assert np.abs(move.qd[:, :6]).max() <= 1.0 + 1e-6 and abs(move.qd[:, 6]).max() <= 0.001 + 1e-9
    assert np.all(np.diff(move.t) > 0) and move.t[0] == 0.0
    assert np.allclose(move.tip[-1], kin.fk(q_to)[:3, 3])
    with pytest.raises(ValueError):
        ready.solve_ik(kin, geometry.rpy_matrix([2.0, 0.0, 0.0], [0, 0, 0]), STAGED, iterations=50)


def test_a_seeded_solve_falls_back_to_the_middle_of_the_limits(monkeypatch):
    from types import SimpleNamespace

    kin = SimpleNamespace(lower=np.full(7, -1.0), upper=np.array([1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 0.04]))
    seeds = []

    def solve_ik(kin, target, q_seed):
        seeds.append(np.array(q_seed))
        if len(seeds) == 1:
            raise ValueError("no IK solution for the ready pose (residual 8.88 mm)")
        return np.array(q_seed)

    monkeypatch.setattr(ready, "solve_ik", solve_ik)
    got = ready.solve_ik_seeded(kin, np.eye(4), STAGED)
    assert np.allclose(seeds[0], STAGED)
    assert np.allclose(got[:6], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) and got[6] == STAGED[6]


def test_a_station_goals_way_starts_in_joint_space_only_from_rest(kin):
    """ready.station_way, the executor's way to a goal under the probe guard (and the calibration's wrist check's):
    from rest on the joint limits the joints over the start at the clearance plane; elsewhere the plane's corners
    for the Cartesian planner; nothing at the start itself."""
    from tatbot_motion import load_motion

    motion = load_motion()
    rise = float(motion["probe"]["clearance_m"])
    down = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])   # tool down

    def pose(xyz):
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = down, xyz
        return out

    start = pose([0.25, 0.10, 0.07])
    q_over, via = ready.station_way(kin, motion, STAGED, start)
    z = max(float(kin.fk(STAGED)[2, 3]), 0.07 + rise)
    assert via == [] and np.allclose(kin.fk(q_over)[:3, 3], [0.25, 0.10, z], atol=1e-4)
    q = ready.solve_ik(kin, pose([0.30, -0.05, 0.15]), STAGED)   # off the limits
    q_over, via = ready.station_way(kin, motion, q, start)
    assert q_over is None and np.allclose(via, [[0.25, 0.10, 0.15]], atol=1e-4)   # already at the plane's height
    q_over, via = ready.station_way(kin, motion, ready.solve_ik(kin, start, q), start)
    assert q_over is None and via == []
