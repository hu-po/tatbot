from dataclasses import dataclass

from tatbot.data import Yaml

@dataclass
class Scene(Yaml):
    """Scene is a collection of objects that represent the scene."""

    arms_config_name: str
    """Name of the arms config (ArmsConfig)."""
    camera_config_name: str
    """Name of the camera config (CamerasConfig)."""
    zone_config_name: str
    """Name of the zone config (Zone)."""
    urdf_config_name: str
    """Name of the urdf config (URDF)."""
    skin_config_name: str
    """Name of the skin config (Skin)."""
    ink_config_name: str
    """Name of the ink config (Ink)."""


    home_pos_l_name: str
    """Name of the left arm pose (ArmPose)."""
    home_pos_r_name: str
    """Name of the right arm pose (ArmPose)."""

    yaml_dir: str = "~/tatbot/config/scenes"
    """Directory containing the scene configs."""