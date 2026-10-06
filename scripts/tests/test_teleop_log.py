"""Pin the .wxtl reader's effort channel and contact classification.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_teleop_log.py

The touch-off reads contacts post-hoc from the flight log, so what must hold:
the follower_eff column lands at offset 5+4J (a one-off slicing error would
silently read velocities as efforts), the gripper is excluded from the arm
contact signal (grip force must never masquerade as a touch), and the contact
threshold comes from the log's own baseline.
"""

import struct

import numpy as np
import pytest
from calib_synth import NUM_JOINTS, write_wxtl  # noqa: E402
from teleop_log import TeleopLog  # noqa: E402

FREE = np.full(NUM_JOINTS, 0.05)


def make_log(tmp_path):
    q0 = np.array([0.1, -0.4, 0.5, 0.0, 0.3, -0.2, 0.0])
    q1 = q0 + 0.4
    q2 = q0 - 0.3
    contact = FREE.copy()
    contact[2] = 2.5                     # arm joint pressing
    grip_only = FREE.copy()
    grip_only[-1] = 60.0                 # gripper squeezing, arm free
    path = tmp_path / "teleop.wxtl"
    centers = write_wxtl(path, [q0, q1, q2], [FREE, contact, grip_only])
    return path, centers, (q0, q1, q2)


def test_effort_channel_and_intervals(tmp_path):
    path, centers, (q0, q1, q2) = make_log(tmp_path)
    log = TeleopLog(path)
    assert log.num_joints == NUM_JOINTS
    intervals = log.still_intervals()
    assert len(intervals) == 3, [round(i["duration_s"], 2) for i in intervals]
    # the effort channel is the one that was written, not a neighbour column
    assert abs(intervals[1]["arm_eff_med_nm"] - 2.5) < 0.1
    assert abs(intervals[0]["arm_eff_med_nm"] - 0.05) < 0.1
    # median joints match what the log held still at
    assert np.allclose(intervals[0]["follower_pos"], q0, atol=0.001)
    # absolute time: interval brackets the known still center
    assert intervals[0]["start_unix"] < centers[0] < intervals[0]["end_unix"]


def test_contact_classification_ignores_gripper(tmp_path):
    path, _, _ = make_log(tmp_path)
    log = TeleopLog(path)
    intervals = log.still_intervals()
    info = log.classify_contacts(intervals)
    assert [i["contact"] for i in intervals] == [False, True, False], (
        "only the arm-effort interval is a touch — 60 N on the gripper is a "
        "grip, not a contact")
    assert info["threshold_nm"] < 2.5
    assert info["baseline_nm"] < 0.2


def test_still_interval_boundaries_match_full_window_reference():
    rng = np.random.default_rng(27)
    for scale in (0.0, 0.0002, 0.002, 0.02):
        positions = np.cumsum(rng.normal(0, scale, (1500, 7)), axis=0)
        log = object.__new__(TeleopLog)
        log.follower_pos = positions
        log.follower_vel = np.zeros_like(positions)
        log.arm_eff = np.zeros(len(positions))
        log.unix_seconds = np.arange(len(positions)) * 0.0025
        expected = []
        start = 0
        for index in range(1, len(positions)):
            if np.abs(positions[start:index + 1] - positions[start]).max() > 0.003:
                if index - 1 > start:
                    expected.append((start, index - 1))
                start = index
        if len(positions) - 1 > start:
            expected.append((start, len(positions) - 1))
        actual = log.still_intervals(min_duration=0)
        assert [(r["start_unix"], r["end_unix"], r["ticks"]) for r in actual] == [
            (log.unix_seconds[a], log.unix_seconds[b], b - a + 1) for a, b in expected]


def timed_log(tmp_path, rows, trailing=b""):
    path = tmp_path / "timed.wxtl"
    header = struct.pack("<8sQddddQq", b"WXTLOG1\0", NUM_JOINTS, .0025, .01, .02, 0.,
                         1, 1_700_000_000_123_456_789)
    path.write_bytes(header + np.asarray(rows, dtype="<f8").tobytes() + trailing)
    return TeleopLog(path)


def timing_rows():
    rows = np.zeros((3, 5 + 6 * NUM_JOINTS))
    # The final sample follows a recorder gap; no synthetic ticks may appear.
    rows[:, :5] = [[0., .0001, .0003, .0007, .0011],
                   [.0025, .0026, .0029, .0035, .0040],
                   [.0125, .0127, .0131, .0140, .0148]]
    rows[:, 5 + 2 * NUM_JOINTS:5 + 3 * NUM_JOINTS] = np.arange(3)[:, None] + .2
    rows[:, 5 + 5 * NUM_JOINTS:] = 99.  # Commanded target differs from measured joints.
    return rows


def test_recorded_read_timestamps_preserve_offsets_gaps_and_measured_rows(tmp_path):
    rows = timing_rows()
    log = timed_log(tmp_path, rows)
    origin = log.wall_start_ns / 1e9
    for relative, absolute, column in [
            (log.t_wake, log.unix_seconds, 1),
            (log.t_leader_read, log.leader_read_unix_seconds, 2),
            (log.t_follower_read, log.follower_read_unix_seconds, 3),
            (log.t_cmd, log.command_unix_seconds, 4)]:
        np.testing.assert_array_equal(relative, rows[:, column])
        np.testing.assert_array_equal(absolute, origin + rows[:, column])
    assert len(log) == 3
    assert log.t_follower_read[2] - log.t_follower_read[1] > .01
    np.testing.assert_array_equal(log.follower_pos[:, 0], [.2, 1.2, 2.2])
    assert not np.array_equal(log.follower_read_unix_seconds, log.unix_seconds)


def test_truncated_timing_record_is_not_exposed(tmp_path):
    rows = timing_rows()
    log = timed_log(tmp_path, rows[:2], rows[2].astype("<f8").tobytes()[:-8])
    assert len(log) == 2
    np.testing.assert_array_equal(log.t_follower_read, rows[:2, 3])
    np.testing.assert_array_equal(log.t_leader_read, rows[:2, 2])
    np.testing.assert_array_equal(log.t_cmd, rows[:2, 4])
    assert len(log.follower_read_unix_seconds) == len(log.follower_pos) == 2


def test_header_only_log_has_empty_read_timelines(tmp_path):
    log = timed_log(tmp_path, np.empty((0, 5 + 6 * NUM_JOINTS)))
    assert len(log) == 0
    assert all(len(values) == 0 for values in [log.t_follower_read, log.t_leader_read, log.t_cmd,
        log.follower_read_unix_seconds, log.leader_read_unix_seconds, log.command_unix_seconds])


@pytest.mark.parametrize('flags,joints', [(7, (0,)), (31, (0, 4, 5)), (63, (0, 4, 5))])
def test_reversed_log_requires_explicit_physical_arm_handling(tmp_path, flags, joints):
    path, _, _ = make_log(tmp_path)
    data = bytearray(path.read_bytes())
    data[:8] = b'WXTLOG2\0'
    struct.pack_into('<Q', data, 48, flags)
    path.write_bytes(data)
    with pytest.raises(ValueError, match='physical-arm-aware'):
        TeleopLog(path)
    log = TeleopLog(path, allow_reversed=True)
    assert (log.leader_arm, log.follower_arm) == ('right', 'left')
    assert log.mirrored_joints == joints
    assert log.start_pose_anchored == (flags == 63)
    windows = log.still_intervals(arm='leader')
    assert len(windows) == 1  # input column in this fixture is stationary
    assert windows[0]['sampled_role'] == 'leader'
    assert windows[0]['sampled_pos'] == [0.] * 7
    assert windows[0]['max_speed_rad_s'] == 0
    for invalid_flags in (3, 15, 127):
        struct.pack_into('<Q', data, 48, invalid_flags)
        path.write_bytes(data)
        with pytest.raises(ValueError, match='unsupported mirrored'):
            TeleopLog(path, allow_reversed=True)
