from dataclasses import dataclass

import numpy as np

from tatbot.data import Yaml
from tatbot.data.pose import Pose


@dataclass
class Skin(Yaml):
    """Skin is a collection of points that represent the skin of the robot."""

    description: str
    """Description of the skin."""

    image_width_m: float
    """Width of the image in meters."""
    image_height_m: float
    """Height of the image in meters."""
    image_width_px: int
    """Width of the image in pixels."""
    image_height_px: int
    """Height of the image in pixels."""

    design_pose: Pose
    """Pose of the design in the global frame."""

    zone_pose: Pose
    """Zone is a rectangular voxel with a 6D pose. In the zone frame, height is z, width is x, and depth is y."""

    zone_height_m: float
    """Height of the zone in meters."""
    zone_width_m: float
    """Width of the zone in meters."""
    zone_depth_m: float
    """Depth of the zone in meters."""

    yaml_dir: str = "~/tatbot/config/skins"
    """Directory containing the skin configs."""