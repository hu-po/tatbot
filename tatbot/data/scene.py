from dataclasses import dataclass, field

import numpy as np

from tatbot.data import Yaml
from tatbot.data.arms import Arms
from tatbot.data.pose import ArmPose
from tatbot.data.cams import Cams
from tatbot.data.zone import Zone
from tatbot.data.urdf import URDF
from tatbot.data.skin import Skin
from tatbot.data.inks import Inks
from tatbot.data.tags import Tags
from tatbot.utils.log import get_logger

log = get_logger("scene", "🌆")

@dataclass
class Scene(Yaml):
    """Scene is a collection of objects that represent the scene."""

    name: str
    """Name of the scene."""

    arms_config_name: str
    """Name of the arms config (Arms)."""
    cams_config_name: str
    """Name of the camera config (Cameras)."""
    zone_config_name: str
    """Name of the zone config (Zone)."""
    urdf_config_name: str
    """Name of the urdf config (URDF)."""
    skin_config_name: str
    """Name of the skin config (Skin)."""
    inks_config_name: str
    """Name of the ink config (Inks)."""
    tags_config_name: str
    """Name of the tag config (Tags)."""

    home_pos_l_name: str
    """Name of the left arm pose (ArmPose)."""
    home_pos_r_name: str
    """Name of the right arm pose (ArmPose)."""

    yaml_dir: str = "~/tatbot/config/scenes"
    """Directory containing the scene configs."""

    arms: Arms = field(init=False)
    cams: Cams = field(init=False)
    zone: Zone = field(init=False)
    urdf: URDF = field(init=False)
    skin: Skin = field(init=False)
    inks: Inks = field(init=False)
    tags: Tags = field(init=False)
    home_pos_l: ArmPose = field(init=False)
    home_pos_r: ArmPose = field(init=False)
    home_pos_full: np.ndarray = field(init=False)

    def __post_init__(self):
        log.info(f"📂 Loading scene config: {self.yaml_dir}/{self.name}.yaml")
        self.arms = Arms.from_name(self.arms_config_name)
        self.cams = Cams.from_name(self.cams_config_name)
        self.zone = Zone.from_name(self.zone_config_name)
        self.urdf = URDF.from_name(self.urdf_config_name)
        self.skin = Skin.from_name(self.skin_config_name)
        self.inks = Inks.from_name(self.inks_config_name)
        self.tags = Tags.from_name(self.tags_config_name)
        self.home_pos_l = ArmPose.from_name(self.home_pos_l_name)
        self.home_pos_r = ArmPose.from_name(self.home_pos_r_name)
        self.home_pos_full = ArmPose.make_bimanual_joints(self.home_pos_l, self.home_pos_r)