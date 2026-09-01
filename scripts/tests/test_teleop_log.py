"""Pin the .wxtl reader's effort channel and contact classification.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_teleop_log.py

The touch-off reads contacts post-hoc from the flight log, so what must hold:
the follower_eff column lands at offset 5+4J (a one-off slicing error would
silently read velocities as efforts), the gripper is excluded from the arm
contact signal (grip force must never masquerade as a touch), and the contact
threshold comes from the log's own baseline.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
