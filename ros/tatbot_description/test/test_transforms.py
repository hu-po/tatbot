"""Shared pose conversions keep URDF Euler order and message quaternion conventions."""
import numpy as np
import pytest
from tatbot_description.transforms import matrix_quat, quat_matrix, rpy_matrix


def test_mixed_urdf_angles_are_extrinsic_xyz():
    x, y, z = .3, -.4, .7
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    pose = rpy_matrix([.1, .2, .3], [x, y, z])
    np.testing.assert_allclose(pose[:3, :3], rz @ ry @ rx, atol=1e-15)
    np.testing.assert_array_equal(pose[:3, 3], [.1, .2, .3])
    np.testing.assert_array_equal(pose[3], [0, 0, 0, 1])


@pytest.mark.parametrize('rpy', [[0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi], [.3, -.4, .7]])
def test_half_turns_and_general_message_poses_round_trip(rpy):
    pose = rpy_matrix([.1, -.2, .3], rpy)
    q = matrix_quat(pose)
    assert q[3] >= 0 and all(type(v) is float for v in q)
    np.testing.assert_allclose(quat_matrix(pose[:3, 3], q), pose, atol=1e-15)
    # ROS producers can send a scaled equivalent quaternion; normalize it.
    np.testing.assert_allclose(quat_matrix(pose[:3, 3], -3*np.array(q)), pose, atol=1e-15)
    with pytest.raises(ValueError, match='zero norm'):
        quat_matrix([0, 0, 0], [0, 0, 0, 0])
