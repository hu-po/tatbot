from pydantic import Field, field_validator
import numpy as np
import yaml
from pathlib import Path
from pydantic_numpy.typing import NpNDArray

from tatbot.data.base import BaseCfg

class Pos(BaseCfg):
    """Position in meters (xyz)."""
    model_config = {'arbitrary_types_allowed': True}
    xyz: NpNDArray

    @field_validator('xyz', mode='before')
    def convert_to_array(cls, v):
        # Handle pydantic-numpy serialized format
        if isinstance(v, dict) and 'data' in v:
            return np.array(v['data'], dtype=np.float32)
        return np.array(v, dtype=np.float32)

    @field_validator('xyz')
    def check_shape(cls, v):
        if v.shape != (3,):
            raise ValueError("xyz must have shape (3,)")
        return v

class Rot(BaseCfg):
    """Orientation quaternion (wxyz)."""
    model_config = {'arbitrary_types_allowed': True}
    wxyz: NpNDArray

    @field_validator('wxyz', mode='before')
    def convert_to_array(cls, v):
        # Handle pydantic-numpy serialized format
        if isinstance(v, dict) and 'data' in v:
            return np.array(v['data'], dtype=np.float32)
        return np.array(v, dtype=np.float32)

    @field_validator('wxyz')
    def check_shape(cls, v):
        if v.shape != (4,):
            raise ValueError("wxyz must have shape (4,)")
        return v

class Pose(BaseCfg):
    """6D pose."""
    pos: Pos
    """Position in meters (xyz)."""
    rot: Rot
    """Orientation quaternion (wxyz)."""

class ArmPose(BaseCfg):
    """Robot arm pose."""
    model_config = {'arbitrary_types_allowed': True}
    joints: NpNDArray
    """Joint positions in radians."""

    @field_validator('joints', mode='before')
    def convert_to_array(cls, v):
        # Handle pydantic-numpy serialized format
        if isinstance(v, dict) and 'data' in v:
            return np.array(v['data'], dtype=np.float32)
        return np.array(v, dtype=np.float32)

    @staticmethod
    def make_bimanual_joints(pose_l: "ArmPose", pose_r: "ArmPose") -> NpNDArray:
        """Concatenate two arm poses to a single bimanual pose."""
        return np.concatenate([pose_l.joints, pose_r.joints]).astype(np.float32)

    @staticmethod
    def get_yaml_dir() -> str:
        """Get the directory containing pose YAML files."""
        return str(Path("~/tatbot/config/poses").expanduser())
