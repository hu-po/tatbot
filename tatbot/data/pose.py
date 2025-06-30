import os
from dataclasses import dataclass

from tatbot.data import Yaml


@dataclass
class Pos:
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position in meters (xyz)."""

@dataclass
class Rot:
    wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Orientation quaternion (wxyz)."""

@dataclass
class Pose(Yaml):
    pos: Pos
    """Position in meters (xyz)."""
    rot: Rot
    """Orientation quaternion (wxyz)."""

@dataclass
class ArmPose(Yaml):
    joints: tuple[float, ...]
    """Joint positions in radians."""
    yaml_dir: str = os.path.expanduser("~/tatbot/config/poses")
