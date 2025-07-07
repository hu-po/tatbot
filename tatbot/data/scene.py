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

    sleep_pos_l_name: str
    """Name of the left arm sleep pose (ArmPose)."""
    sleep_pos_r_name: str
    """Name of the right arm sleep pose (ArmPose)."""

    ready_pos_l_name: str
    """Name of the left arm ready pose (ArmPose)."""
    ready_pos_r_name: str
    """Name of the right arm ready pose (ArmPose)."""

    inkready_pos_l_name: str
    """Name of the left arm inkready pose (ArmPose)."""
    inkready_pos_r_name: str
    """Name of the right arm inkready pose (ArmPose)."""

    yaml_dir: str = "~/tatbot/config/scenes"
    """Directory containing the scene configs."""

    arms: Arms = field(init=False)
    cams: Cams = field(init=False)
    zone: Zone = field(init=False)
    urdf: URDF = field(init=False)
    skin: Skin = field(init=False)
    inks: Inks = field(init=False)
    tags: Tags = field(init=False)

    # sleep pose is the arm folded up, resting on itself, facing forwards
    sleep_pos_l: ArmPose = field(init=False)
    sleep_pos_r: ArmPose = field(init=False)
    
    # ready pose is the arm raised up, facing towards skin, tilted down
    ready_pos_l: ArmPose = field(init=False)
    ready_pos_r: ArmPose = field(init=False)

    # inkready pose is the arms raised up, facing towards ink palette, tilted down
    inkready_pos_l: ArmPose = field(init=False)
    inkready_pos_r: ArmPose = field(init=False)

    def __post_init__(self):
        log.info(f"📂 Loading scene config: {self.yaml_dir}/{self.name}.yaml")
        self.arms = Arms.from_name(self.arms_config_name)
        self.cams = Cams.from_name(self.cams_config_name)
        self.zone = Zone.from_name(self.zone_config_name)
        self.urdf = URDF.from_name(self.urdf_config_name)
        self.skin = Skin.from_name(self.skin_config_name)
        self.inks = Inks.from_name(self.inks_config_name)
        self.tags = Tags.from_name(self.tags_config_name)
        
        self.sleep_pos_l = ArmPose.from_name(self.sleep_pos_l_name)
        self.sleep_pos_r = ArmPose.from_name(self.sleep_pos_r_name)
        self.sleep_pos_full = ArmPose.make_bimanual_joints(self.sleep_pos_l, self.sleep_pos_r)
        
        self.ready_pos_l = ArmPose.from_name(self.ready_pos_l_name)
        self.ready_pos_r = ArmPose.from_name(self.ready_pos_r_name)
        self.ready_pos_full = ArmPose.make_bimanual_joints(self.ready_pos_l, self.ready_pos_r)
        
        self.inkready_pos_l = ArmPose.from_name(self.inkready_pos_l_name)
        self.inkready_pos_r = ArmPose.from_name(self.inkready_pos_r_name)