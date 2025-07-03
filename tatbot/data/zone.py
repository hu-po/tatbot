from dataclasses import dataclass

from tatbot.data import Yaml
from tatbot.data.pose import Pose

@dataclass
class Zone(Yaml):
    """Zone is a rectangular voxel with a 6D pose. In the zone frame, height is z, width is x, and depth is y."""
    name: str
    """Name of the zone."""
    pose: Pose
    """Pose of the zone in the global frame."""
    height_m: float
    """Height of the zone in meters."""
    width_m: float
    """Maximum width of the zone in meters."""
    width_min_m: float
    """Minimum width of the zone in meters."""
    depth_max_m: float
    """Maximum depth of the zone in meters."""
    depth_min_m: float
    """Minimum depth of the zone in meters."""
    yaml_dir: str = "~/tatbot/config/zones"
    """Directory containing the config yaml files."""