from dataclasses import dataclass

import numpy as np

from tatbot.data import Yaml
from tatbot.data.pose import Pose


@dataclass
class Skin(Yaml):
    """Skin is a collection of points that represent the skin of the robot."""

    description: str
    """Description of the skin."""

    # points: np.ndarray | None
    # """Points in the skin."""

    design_pose: Pose
    """Pose of the design in the global frame."""

    yaml_dir: str = "~/tatbot/config/skins"
    """Directory containing the skin configs."""