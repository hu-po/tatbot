import os
from dataclasses import dataclass

from . import Yaml

@dataclass
class URDF(Yaml):
    path: str
    """Path to the URDF file for the robot."""
    ee_link_names: tuple[str, str]
    """Names of the ee (end effector) links in the URDF."""
    tag_link_names: tuple[str, ...]
    """Names of the tag (apriltag) links in the URDF."""
    cam_link_names: tuple[str, ...]
    """Names of the camera links in the URDF."""
    ink_link_names: tuple[str, ...]
    """Names of the inkcap links in the URDF."""
    palette_link_name: str
    """Name of the inkpalette link in the URDF."""
    origin_link_name: str
    """Name of the origin link in the URDF."""
    skin_link_name: str
    """Name of the skin link in the URDF."""
    yaml_dir: str = os.path.expanduser("~/tatbot/config/urdf")
    """Directory containing the urdf configs."""