"""Bridge measured rig calibration into the single-arm simulator frame."""

import numpy as np
from robot_world import root_from_world

from tatbot_sim.urdf import rig_from_follower_base


def world_from_follower_base(bundle, registration):
    """The simulator root is the follower base; the calibration root is the rig."""
    identity = bundle.get('bundle_id')
    if not identity or registration.get('calibration_id') != identity:
        raise ValueError('camera and robot-world calibration IDs differ')
    return np.linalg.inv(root_from_world(registration)) @ rig_from_follower_base()
