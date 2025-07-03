from dataclasses import dataclass

from tatbot.data import Yaml

@dataclass
class ArmsConfig(Yaml):
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

    yaml_dir: str = "~/tatbot/config/arms"
    """Directory containing the config yaml files."""