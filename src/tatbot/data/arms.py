from pydantic import field_validator
from pathlib import Path
import ipaddress

from tatbot.data.base import BaseCfg
from tatbot.data.pose import Pos, Rot

class Arms(BaseCfg):
    ip_address_l: str
    """IP address of the left robot arm."""
    ip_address_r: str
    """IP address of the right robot arm."""

    arm_l_config_filepath: Path
    """YAML file containing left arm config."""
    arm_r_config_filepath: Path
    """YAML file containing right arm config."""

    goal_time_fast: float
    """Robot travel time when executing fast actions, usually small movements."""
    goal_time_slow: float
    """Robot travel time when moving slowly, usually larger movements."""
    connection_timeout: float
    """Timeout when connecting to the robot arms in seconds."""

    ee_rot_l: Rot
    """<w, x, y, z> quaternion of left arm end effector when performing a stroke."""
    ee_rot_r: Rot
    """<w, x, y, z> quaternion of right arm end effector when performing a stroke."""

    hover_offset: Pos
    """<x, y, z> offset for hover (first and last poses in each stroke) in meters."""

    offset_range: tuple[float, float]
    """Range of offset points in meters."""
    offset_num: int
    """Number of offset points."""

    ee_offset_l: Pos
    """<x, y, z> offset for left end effector to account for arm slop."""
    ee_offset_r: Pos
    """<x, y, z> offset for right end effector to account for arm slop."""

    align_x_size_m: float
    """Size of the laser X when performing align strokes."""

    @field_validator('ip_address_l', 'ip_address_r')
    def validate_ip(cls, v):
        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError(f"'{v}' is not a valid IP address")
        return v
    
    @field_validator('arm_l_config_filepath', 'arm_r_config_filepath', mode='before')
    def expand_user_path(cls, v):
        return Path(v).expanduser()

    @field_validator('arm_l_config_filepath', 'arm_r_config_filepath')
    def path_must_exist(cls, v: Path):
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v
