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

    return Kinematics(robot_description(repo_root(None), arms=("right",)), "right")


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


OFF_STOPS = STAGED + np.array([0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0])   # joints 1 and 2 clear of their rest stops


def test_a_configured_joint_move_starts_and_stops_under_its_acceleration_and_jerk_caps(kin):
    """2026-10-20's ready move from the last locate view (0.67 rad) and its drop from the high ready pose to the
    standoff (0.19 rad): under the pen-up joint speed alone 1.26 s and the 0.5 s floor, at 2.4 and 4.4 rad/s^2 with
    the jerk stepping to 20 and 91 rad/s^3 as they started; under ready.JOINT_ACCEL_RAD_S2 and JOINT_JERK_RAD_S3, 2.0
    and 1.3 s. A move the caps do not bind (0.3 rad/s) keeps its time; an uncapped one is as it was."""
    motion = {"joint_speed": {"pen_up_rad_s": 1.0}, "carriage": {"max_m_s": 0.001}}
    shape = np.array([0.8, -0.5, 1.0, 0.7, -0.9, 0.6, 0.0])
    for span, alone_s, capped_s in ((0.67, 1.256, 2.004), (0.19, 0.5, 1.316)):
        q_to = OFF_STOPS + span * shape
        alone = ready.joint_move(kin.joint_names, OFF_STOPS, q_to, max_rad_s=1.0, max_m_s=0.001, rate_hz=400)
        move = ready.pen_up_joint_move(kin, motion, OFF_STOPS, q_to, 400.0)
        assert alone.duration_s == pytest.approx(alone_s, abs=1e-3) and move.duration_s == pytest.approx(capped_s, abs=1e-3)
        accel = np.gradient(move.qd[:, :6], move.t, axis=0)
        assert np.abs(accel).max() <= ready.JOINT_ACCEL_RAD_S2 * 1.001
        assert np.abs(accel[1]).max() / move.t[1] <= ready.JOINT_JERK_RAD_S3 * 1.01   # the jerk it starts with
        assert np.allclose(move.q[-1], q_to) and np.allclose(move.qd[[0, -1]], 0.0)
    slow = ready.joint_move(kin.joint_names, STAGED, STAGED + 1.25 * shape, max_rad_s=0.3, max_m_s=0.002, rate_hz=100,
                            max_rad_s2=ready.JOINT_ACCEL_RAD_S2, max_rad_s3=ready.JOINT_JERK_RAD_S3)
    assert slow.duration_s == pytest.approx(15.0 / 8.0 * 1.25 / 0.3, abs=0.01)


def test_a_configured_joint_move_from_rest_lifts_off_slowly_then_speeds_up_under_the_caps(kin):
    """Leaving rest, where the stops carry the arm and joint_3's velocity integral sits at its clamp, the wrist fell
    10-13 mrad ahead of a reference at 0.05-0.1 rad/s and 17.6-18.2 mrad ahead of one at 0.3-0.5 rad/s; the swing to
    the first locate view at 0.3 rad/s (8.1 s) then chattered at 5 Hz as it slowed, the tip 2.8-6.7 mm (2026-10-20/21).
    From rest a configured move covers its largest joint's first LIFT_RAD/2 under LIFT_RAD_S and only then speeds up,
    never slowing before its peak, under the caps: that swing in 3.6 s. Off the stops, or short, it is the plain one."""
    motion = {"joint_speed": {"pen_up_rad_s": 1.0}, "carriage": {"max_m_s": 0.001}}
    way = np.array([-0.05, 1.25, 1.30, -1.16, -0.04, -0.04, 0.0])   # that first view, from rest
    move = ready.pen_up_joint_move(kin, motion, STAGED, STAGED + way, 400.0)
    plain = ready.pen_up_joint_move(kin, motion, OFF_STOPS, OFF_STOPS + way, 400.0)
    assert plain.duration_s == pytest.approx(np.sqrt(ready.QUINTIC_ACCEL * 1.30), abs=1e-3)   # 2.74 s
    assert move.duration_s == pytest.approx(15.0 / 16.0 + np.sqrt(ready.QUINTIC_ACCEL * 1.20), abs=1e-3)   # 3.57 s
    speed = np.abs(move.qd[:, 2])   # joint_2 leads
    assert speed[np.abs(move.q[:, 2] - STAGED[2]) <= ready.LIFT_RAD / 2].max() <= ready.LIFT_RAD_S + 1e-9
    assert np.all(np.diff(speed[: int(np.argmax(speed)) + 1]) >= -1e-9)
    accel = np.gradient(move.qd[:, :6], move.t, axis=0)
    assert np.abs(move.qd[:, :6]).max() <= 1.0 and np.abs(accel).max() <= ready.JOINT_ACCEL_RAD_S2 * 1.001
    assert np.abs(np.diff(accel, axis=0) / np.diff(move.t)[:, None]).max() <= ready.JOINT_JERK_RAD_S3 * 1.01
    assert np.allclose(move.q[[0, -1]], [STAGED, STAGED + way]) and np.allclose(move.qd[[0, -1]], 0.0)
    short = ready.pen_up_joint_move(kin, motion, STAGED, STAGED + 0.15 * np.sign(way), 400.0)
    assert short.duration_s == pytest.approx(np.cbrt(ready.QUINTIC_JERK * 0.15 / ready.JOINT_JERK_RAD_S3), abs=1e-3)


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


def _base_yaw(page, heading) -> float:
    pose = geometry.tool_down_pose(page, (0.0, 0.0), 0.010, heading)
    return float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0])))


def _touch(yaw_deg: float) -> np.ndarray:
    """A station touch's tool rotation: down, its x axis at yaw_deg in the base."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def test_a_new_page_takes_the_dips_heading_where_it_keeps_its_margin(kin):
    """2026-10-20, a forearm print's set-up page: the search took -105 deg, and every dip turned the wrist 76 deg to
    the station touch's -30 and back at 8 deg/s, ~36 s a dip. The touch's heading first, then 15 deg steps either side
    of it: -30 keeps 0.32 rad, -45 the floor, and its whole curved design planned there."""
    from tatbot_session import inspect as ins

    page = geometry.rpy_matrix([0.3429, 0.0275, 0.0808], [-0.0857, 0.1174, -0.0157])
    clear = [0.045, 0.063]
    assert _base_yaw(page, ready.pen_heading(kin, page, clear, 0.010, ins.REST)) == pytest.approx(-105.3, abs=0.5)
    assert _base_yaw(page, ready.pen_heading(kin, page, clear, 0.010, ins.REST, prefer=_touch(-30.0))) == \
        pytest.approx(-45.3, abs=0.5)


def test_the_dips_heading_steps_to_the_nearest_heading_that_keeps_the_margin(kin, monkeypatch):
    """Over a page near the base line (x along the base's -y) the search's own best is -30 deg (0.56 rad); from a
    touch at -75 (0.16) the steps go -60 / -90, -45 / -105, -30 / -120: the nearest that keeps the margin. The
    margins follow the fitted tool's tip in urdf/tatbot.urdf, so each threshold sits between the two it separates."""
    from tatbot_session import inspect as ins

    page = np.eye(4)
    page[:3, :3] = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    page[:3, 3] = [0.296, 0.013, 0.003]
    clear = [0.090, 0.134]

    def chosen(**kw):
        return _base_yaw(page, ready.pen_heading(kin, page, clear, 0.010, ins.REST, **kw))

    assert chosen() == pytest.approx(-30.0, abs=0.1)
    assert chosen(prefer=_touch(-75.0)) == pytest.approx(-30.0, abs=0.1)        # -45 keeps 0.36 rad
    monkeypatch.setattr(ready, "PREFER_MARGIN_RAD", 0.30)                        # over -60's 0.21, under -45's 0.36
    assert chosen(prefer=_touch(-75.0)) == pytest.approx(-45.0, abs=0.1)
    assert chosen(prefer=_touch(-60.0)) == pytest.approx(-45.0, abs=0.1)        # -45 (0.36) over -75 (0.16)
    monkeypatch.setattr(ready, "PREFER_MARGIN_RAD", 1.0)                         # none keeps it: the search
    assert chosen(prefer=_touch(-75.0)) == pytest.approx(-30.0, abs=0.1)
