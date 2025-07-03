from dataclasses import dataclass

from tatbot.data import Yaml


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
    root_link_name: str
    """Name of the origin/root link in the URDF."""

    yaml_dir: str = "~/tatbot/config/urdf"
    """Directory containing the urdf configs."""