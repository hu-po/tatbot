"""An arm's registration holds only in the camera world it was solved in, or one carried onto."""

import numpy as np
import pytest
from robot_world import registration


def test_registration_must_be_this_arms_on_this_bundle_or_carried_without_a_world_change():
    world_from_base = np.eye(4)
    world_from_base[:3, 3] = (0.1, 0.02, 0.0)
    bundle = {'bundle_id': 'bundle-1'}
    record = {'schema': 'tatbot.arm-registration/1', 'arm': 'left', 'calibration_id': 'bundle-1',
              'world_from_arm_base': world_from_base.tolist()}
    matrix, provenance = registration('left', bundle, record, None)
    np.testing.assert_allclose(matrix, world_from_base)
    assert provenance == {'calibration_id': 'bundle-1', 'carried': None}
    with pytest.raises(ValueError, match='for the left arm'):
        registration('left', bundle, dict(record, arm='right'), None)
    stale = dict(record, calibration_id='bundle-0')
    with pytest.raises(ValueError, match='not carried'):
        registration('left', bundle, stale, None)
    golden = {'calibration_id': 'bundle-1', 'carried': {'from_calibration_id': 'bundle-0', 'world_change': np.eye(4).tolist()}}
    matrix, provenance = registration('left', bundle, stale, golden)
    np.testing.assert_allclose(matrix, world_from_base)
    assert provenance['carried']['onto'] == 'bundle-1'
    moved = np.eye(4)
    moved[0, 3] = 0.01
    with pytest.raises(ValueError, match='moved the fixed-camera world'):
        registration('left', bundle, stale, {'calibration_id': 'bundle-1',
                                             'carried': {'from_calibration_id': 'bundle-0', 'world_change': moved.tolist()}})
