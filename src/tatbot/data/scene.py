import functools
import json
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import model_validator

from tatbot.bot.urdf import get_link_poses
from tatbot.data.arms import Arms
from tatbot.data.base import BaseCfg
from tatbot.data.cams import Cams
from tatbot.data.inks import InkCap, Inks
from tatbot.data.pose import ArmPose, Pose
from tatbot.data.skin import Skin
from tatbot.data.tags import Tags
from tatbot.data.urdf import URDF
from tatbot.utils.constants import (
    CONF_POSES_DIR,
    resolve_design_dir,
    resolve_pens_config_path,
)


class Scene(BaseCfg):
    """Scene is a collection of objects that represent the scene."""
    
    model_config = {'arbitrary_types_allowed': True}

    name: str
    """Name of the scene."""

    arms: Arms
    cams: Cams
    urdf: URDF
    skin: Skin
    inks: Inks
    tags: Tags

    sleep_pos_l_name: str
    """Name of the left arm sleep pose (ArmPose)."""
    sleep_pos_r_name: str
    """Name of the right arm sleep pose (ArmPose)."""

    ready_pos_l_name: str
    """Name of the left arm ready pose (ArmPose)."""
    ready_pos_r_name: str
    """Name of the right arm ready pose (ArmPose)."""

    pen_names_l: List[str]
    """Name of pens that will be drawn using left arm."""
    pen_names_r: List[str]
    """Name of pens that will be drawn using right arm."""
    pens_config_name: str
    """Name of the DrawingBotV3 pens config (JSON) under the pens configs directory."""

    stroke_length: int
    """All strokes will be resampled to this length."""

    design_name: Optional[str] = None
    """Name of the design directory under NFS designs."""
    design_img_path: Optional[Path] = None
    """Path to the design image."""

    # Populated by validators
    sleep_pos_l: Optional[ArmPose] = None
    sleep_pos_r: Optional[ArmPose] = None
    ready_pos_l: Optional[ArmPose] = None
    ready_pos_r: Optional[ArmPose] = None
    sleep_pos_full: Optional[ArmPose] = None
    ready_pos_full: Optional[ArmPose] = None
    pens_config: Optional[dict] = None
    inkcaps_l: Optional[dict[str, InkCap]] = None
    inkcaps_r: Optional[dict[str, InkCap]] = None
    calibrator_pose: Optional[Pose] = None
    lasercross_pose: Optional[Pose] = None
    design_dir: Optional[Path] = None

    # No path expansion needed; pens config is resolved by name

    @model_validator(mode='after')
    def load_poses(self) -> 'Scene':
        # Load poses using cached function
        poses = self._load_poses_cached(
            self.sleep_pos_l_name, 
            self.sleep_pos_r_name,
            self.ready_pos_l_name, 
            self.ready_pos_r_name
        )
        
        # Update self directly instead of returning a copy
        for key, value in poses.items():
            setattr(self, key, value)
        
        return self
    
    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _load_poses_cached(sleep_l_name: str, sleep_r_name: str, ready_l_name: str, ready_r_name: str) -> dict:
        """Cache pose loading to avoid repeated file I/O."""
        poses_dir = CONF_POSES_DIR
        
        # Load poses
        with open(poses_dir / f"{sleep_l_name}.yaml") as f:
            sleep_pos_l = ArmPose(**yaml.safe_load(f))
        with open(poses_dir / f"{sleep_r_name}.yaml") as f:
            sleep_pos_r = ArmPose(**yaml.safe_load(f))
        with open(poses_dir / f"{ready_l_name}.yaml") as f:
            ready_pos_l = ArmPose(**yaml.safe_load(f))
        with open(poses_dir / f"{ready_r_name}.yaml") as f:
            ready_pos_r = ArmPose(**yaml.safe_load(f))

        sleep_pos_full = ArmPose.make_bimanual_joints(sleep_pos_l, sleep_pos_r)
        ready_pos_full = ArmPose.make_bimanual_joints(ready_pos_l, ready_pos_r)
        
        return {
            'sleep_pos_l': sleep_pos_l,
            'sleep_pos_r': sleep_pos_r,
            'ready_pos_l': ready_pos_l,
            'ready_pos_r': ready_pos_r,
            'sleep_pos_full': sleep_pos_full,
            'ready_pos_full': ready_pos_full,
        }

    @model_validator(mode='after')
    def load_urdf_poses(self) -> 'Scene':
        link_names = self.urdf.ink_link_names + (self.urdf.calibrator_link_name, self.urdf.lasercross_link_name)
        link_poses = get_link_poses(self.urdf.path, link_names, self.ready_pos_full.joints)

        # Update inkcap poses without mutating existing objects
        updated_inkcaps = []
        for inkcap in self.inks.inkcaps:
            if inkcap.name in link_poses:
                # Create a copy with updated pose
                updated_inkcaps.append(inkcap.model_copy(update={'pose': link_poses[inkcap.name]}))
            else:
                updated_inkcaps.append(inkcap)
        
        self.inks = self.inks.model_copy(update={'inkcaps': tuple(updated_inkcaps)})
        self.calibrator_pose = link_poses[self.urdf.calibrator_link_name]
        self.lasercross_pose = link_poses[self.urdf.lasercross_link_name]
        return self

    @model_validator(mode='after')
    def check_pens_and_inks(self) -> 'Scene':        
        if self.pens_config_name:
            pens_config_path = resolve_pens_config_path(self.pens_config_name)
            if not pens_config_path.exists():
                raise ValueError(f"Pens config does not exist: {pens_config_path}")
            with pens_config_path.open('r') as f:
                pens_config = json.load(f)
            pens_config_dict = {pen["name"]: pen for pen in pens_config["data"]["pens"]}
            self.pens_config = pens_config_dict
            
            # Validate pen names exist in config
            for pen_name in self.pen_names_l:
                if pen_name not in pens_config_dict:
                    raise ValueError(f"Pen {pen_name} (left) not in pen config")
            for pen_name in self.pen_names_r:
                if pen_name not in pens_config_dict:
                    raise ValueError(f"Pen {pen_name} (right) not in pen config")
        
        # Create inkcap dictionaries keyed by ink name
        inkcaps_l = {
            inkcap.ink.name: inkcap for inkcap in self.inks.inkcaps if "left" in inkcap.name and inkcap.ink
        }
        inkcaps_r = {
            inkcap.ink.name: inkcap for inkcap in self.inks.inkcaps if "right" in inkcap.name and inkcap.ink
        }
        self.inkcaps_l = inkcaps_l
        self.inkcaps_r = inkcaps_r

        # Validate pens are available in inkcaps
        for pen_name in self.pen_names_l:
            if pen_name not in [inkcap.ink.name for inkcap in inkcaps_l.values() if inkcap.ink]:
                raise ValueError(f"Pen {pen_name} (left) not in any inkcap")
        
        for pen_name in self.pen_names_r:
            if pen_name not in [inkcap.ink.name for inkcap in inkcaps_r.values() if inkcap.ink]:
                raise ValueError(f"Pen {pen_name} (right) not in any inkcap")
        
        return self

    @model_validator(mode='after')
    def find_design_image(self) -> 'Scene':        
        if self.design_name:
            design_dir = resolve_design_dir(self.design_name)
            self.design_dir = design_dir
            if not design_dir.exists():
                raise ValueError(f"Design directory does not exist: {design_dir}")
            
            if not self.design_img_path:
                design_img_path = None
                for file in design_dir.iterdir():
                    if file.suffix == ".png" and "_F" not in file.name and "plotted" in file.name:
                        design_img_path = file
                        break
                if design_img_path:
                    self.design_img_path = design_img_path
                else:
                    raise ValueError(f"No design image found in {design_dir}")
        
        return self

    def to_yaml(self, filepath: str):
        """Save the Scene to a YAML file."""
        # Use model_dump to get all data, excluding computed properties
        data = self.model_dump(exclude={
            'sleep_pos_l', 'sleep_pos_r', 'ready_pos_l', 'ready_pos_r',
            'sleep_pos_full', 'ready_pos_full', 'inkcaps_l', 'inkcaps_r',
            'design_img_path'
        })
        
        # Convert Path objects to strings for YAML serialization
        for key, value in data.items():
            if hasattr(value, '__fspath__'):  # Check if it's a Path-like object
                data[key] = str(value)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(data, f)
