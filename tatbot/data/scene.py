from dataclasses import dataclass, field
import os
import json

from tatbot.data import Yaml
from tatbot.data.arms import Arms
from tatbot.data.pose import ArmPose, Pos, Rot
from tatbot.data.cams import Cams
from tatbot.data.urdf import URDF
from tatbot.data.skin import Skin
from tatbot.data.inks import Inks, InkCap, Ink
from tatbot.data.tags import Tags
from tatbot.utils.log import get_logger
from tatbot.bot.urdf import get_link_poses

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

    pen_names_l: list[str]
    """Name of pens that will be drawn using left arm."""
    pen_names_r: list[str]
    """Name of pens that will be drawn using right arm."""
    pens_config_path: str
    """Path to the DrawingBotV3 Pens config file."""

    stroke_length: int
    """All strokes will be resampled to this length."""

    ee_rot_l: Rot
    """<w, x, y, z> quaternion of left arm end effector when performing a stroke."""
    ee_rot_r: Rot
    """<w, x, y, z> quaternion of right arm end effector when performing a stroke."""

    hover_offset: Pos
    """<x, y, z> offset for hover (first and last poses in each stroke) in meters."""

    ee_offset_l: Pos
    """<x, y, z> offset for left ee (applied to full stroke) in meters."""
    ee_offset_r: Pos
    """<x, y, z> offset for right ee (applied to full stroke) in meters."""

    design_dir_path: str | None = None
    """Path to the design directory."""
    design_img_path: str | None = None
    """Path to the design image."""

    yaml_dir: str = "~/tatbot/config/scenes"
    """Directory containing the scene configs."""

    arms: Arms = field(init=False)
    cams: Cams = field(init=False)
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
        self.urdf = URDF.from_name(self.urdf_config_name)
        self.skin = Skin.from_name(self.skin_config_name)
        self.tags = Tags.from_name(self.tags_config_name)
        self.inks = Inks.from_name(self.inks_config_name)
        
        self.sleep_pos_l = ArmPose.from_name(self.sleep_pos_l_name)
        self.sleep_pos_r = ArmPose.from_name(self.sleep_pos_r_name)
        self.sleep_pos_full = ArmPose.make_bimanual_joints(self.sleep_pos_l, self.sleep_pos_r)
        
        self.ready_pos_l = ArmPose.from_name(self.ready_pos_l_name)
        self.ready_pos_r = ArmPose.from_name(self.ready_pos_r_name)
        self.ready_pos_full = ArmPose.make_bimanual_joints(self.ready_pos_l, self.ready_pos_r)
        
        self.inkready_pos_l = ArmPose.from_name(self.inkready_pos_l_name)
        self.inkready_pos_r = ArmPose.from_name(self.inkready_pos_r_name)
        self.inkready_pos_full = ArmPose.make_bimanual_joints(self.inkready_pos_l, self.inkready_pos_r)

        # load pens from config file
        pens_config_path = os.path.expanduser(self.pens_config_path)
        assert os.path.exists(pens_config_path), f"❌ Pens config file {pens_config_path} does not exist"
        log.info(f"📂 Loading pens from config file: {pens_config_path}")
        with open(pens_config_path, 'r') as f:
            pens_config = json.load(f)
        self.pens_config = {pen["name"]: pen for pen in pens_config["data"]["pens"]}
        log.info(f"✅ Found {len(self.pens_config)} pens")
        log.debug(f"Pens in config: {self.pens_config.keys()}")
        for pen_name in self.pen_names_l:
            assert pen_name in self.pens_config, f"❌ Pen {pen_name} (left) not in pen config"
        for pen_name in self.pen_names_r:
            assert pen_name in self.pens_config, f"❌ Pen {pen_name} (right) not in pen config"

        # get the link poses for the inkcaps
        link_poses = get_link_poses(self.urdf.path, self.urdf.ink_link_names, self.inkready_pos_full)
        self.inkcaps_l: dict[str, InkCap] = {}
        self.inkcaps_r: dict[str, InkCap] = {}
        for inkcap in self.inks.inkcaps:
            assert inkcap.name in self.urdf.ink_link_names, f"❌ Inkcap {inkcap.name} not found in URDF"
            if inkcap.ink is not None:
                ink: Ink = Ink(**inkcap.ink)
                _inkcap: InkCap = InkCap(
                    name=inkcap.name,
                    diameter_m=inkcap.diameter_m,
                    depth_m=inkcap.depth_m,
                    ink=ink,
                    pose=link_poses[inkcap.name],
                )
                log.debug(f"Inkcap {inkcap.name} is filled with {ink.name}")
                if "left" in inkcap.name:
                    assert ink.name in self.pen_names_l, f"❌ Ink {ink.name} not found in left pen names"
                    self.inkcaps_l[ink.name] = _inkcap
                else:
                    assert ink.name in self.pen_names_r, f"❌ Ink {ink.name} not found in right pen names"
                    self.inkcaps_r[ink.name] = _inkcap
        log.info(f"✅ Found {len(self.inkcaps_l)} left inkcaps and {len(self.inkcaps_r)} right inkcaps")
        log.debug(f"Left inkcaps: {self.inkcaps_l}")
        log.debug(f"Right inkcaps: {self.inkcaps_r}")

        if self.design_dir_path is not None:
            log.info(f"📂 Loading design from {self.design_dir_path}")
            self.design_dir = os.path.expanduser(self.design_dir_path)
            assert os.path.exists(self.design_dir), f"❌ Design directory {self.design_dir} does not exist"
            log.debug(f"📂 Design directory: {self.design_dir}")

            # final design image 
            design_img_filename = None
            for file in os.listdir(self.design_dir):
                if file.endswith('.png') and '_F' not in file and 'plotted' in file:
                    design_img_filename = file
                    break
            assert design_img_filename is not None, f"❌ No design image found in {self.design_dir}"
            self.design_img_path = os.path.join(self.design_dir, design_img_filename)