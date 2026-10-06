"""Calibration/arm-root boundaries use independently specified transforms."""

import numpy as np
import pytest
from tatbot_sim import calibration


def test_rotated_mount_and_calibration_roundtrip(monkeypatch):
    mount = np.array([[0., -1, 0, .1], [1, 0, 0, -.3], [0, 0, 1, .2], [0, 0, 0, 1]])
    world = np.array([[0., 0, 1, .5], [0, 1, 0, .4], [-1, 0, 0, .3], [0, 0, 0, 1]])
    monkeypatch.setattr(calibration, 'rig_from_follower_base', lambda: mount)
    transform = calibration.world_from_follower_base(
        {'bundle_id': 'measured'}, {'calibration_id': 'measured', 'world_from_base': world.tolist()})
    point_base = np.array([.2, .1, .3, 1])
    expected_world = world @ (mount @ point_base)
    np.testing.assert_allclose(transform @ point_base, expected_world, atol=1e-12)
    np.testing.assert_allclose(np.linalg.inv(transform) @ expected_world, point_base, atol=1e-12)


@pytest.mark.parametrize('identity', [None, 'other'])
def test_simulator_refuses_unbound_or_mismatched_robot_world(identity):
    with pytest.raises(ValueError, match='calibration IDs differ'):
        calibration.world_from_follower_base(
            {'bundle_id': 'current'}, {'calibration_id': identity, 'world_from_base': np.eye(4).tolist()})


def test_simulator_rejects_nonrigid_world_transform():
    matrix = np.eye(4)
    matrix[0, 0] = 2
    with pytest.raises(ValueError, match='rigid'):
        calibration.world_from_follower_base(
            {'bundle_id': 'current'}, {'calibration_id': 'current', 'world_from_base': matrix.tolist()})
