from dataclasses import dataclass

from tatbot.data import Yaml
from tatbot.data.pose import Pos, Rot


@dataclass
class Arms(Yaml):
    ip_address_l: str
    """IP address of the left robot arm."""
    ip_address_r: str
    """IP address of the right robot arm."""

    arm_l_config_filepath: str
    """YAML file containing left arm config."""
    arm_r_config_filepath: str
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

    align_x_size_m: float = 0.01
    """Size of the laser X when performing align strokes."""

    yaml_dir: str = "~/tatbot/config/arms"
    """Directory containing the config yaml files."""
