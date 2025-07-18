from dataclasses import dataclass, field

import numpy as np

from tatbot.data import Yaml


@dataclass
class Pos(Yaml):
    xyz: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    """Position in meters (xyz)."""


@dataclass
class Rot(Yaml):
    wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    """Orientation quaternion (wxyz)."""


@dataclass
class Pose(Yaml):
    pos: Pos
    """Position in meters (xyz)."""
    rot: Rot
    """Orientation quaternion (wxyz)."""


@dataclass
class ArmPose(Yaml):
    joints: np.ndarray = field(default_factory=lambda: np.array([0.0] * 7, dtype=np.float32))
    """Joint positions in radians."""
    yaml_dir: str = "~/tatbot/config/poses"
    """Directory containing the config yaml files."""

    @staticmethod
    def make_bimanual_joints(pose_l: "ArmPose", pose_r: "ArmPose") -> np.ndarray:
        return np.concatenate([pose_l.joints, pose_r.joints])
