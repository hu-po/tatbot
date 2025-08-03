from pydantic import field_validator
from tatbot.data.base import BaseCfg
from pathlib import Path

from tatbot.data.pose import Pose

class Skin(BaseCfg):
    """Skin is a collection of points that represent the skin of the robot."""

    description: str
    """Description of the skin."""

    image_width_m: float
    """Width of the image in meters."""
    image_height_m: float
    """Height of the image in meters."""

    design_pose: Pose
    """Pose of the design in the global frame."""

    """
    Zone is a rectangular voxel with a 6D pose.
    The zone is centered at the design pose and used to crop pointclouds.
    """
    zone_depth_m: float
    """Depth of the zone in meters. (x)"""
    zone_width_m: float
    """Width of the zone in meters. (y)"""
    zone_height_m: float
    """Height of the zone in meters. (z)"""
    
    plymesh_dir: Path
    """Directory containing the ply and mesh files."""

    @field_validator('plymesh_dir', mode='before')
    def expand_user_path(cls, v):
        return Path(v).expanduser()

    @field_validator('plymesh_dir')
    def path_must_exist(cls, v: Path):
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v
