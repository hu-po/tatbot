
from typing import Any

import numpy as np
from pydantic import field_validator
from pydantic_numpy.typing import NpNDArray

from tatbot.data.base import BaseCfg
from tatbot.utils.constants import CONF_POSES_DIR


class Pos(BaseCfg):
    """Position in meters (xyz)."""
    model_config = {'arbitrary_types_allowed': True}
    xyz: NpNDArray

    @field_validator('xyz', mode='before')
    def convert_to_array(cls, v: Any) -> np.ndarray:
        # Handle pydantic-numpy serialized format
        if isinstance(v, dict) and 'data' in v:
            return np.array(v['data'], dtype=np.float32)
        # Convert lists to arrays, keep arrays as-is
        if isinstance(v, list):
            return np.asarray(v, dtype=np.float32)
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v
    
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override model_dump to convert small arrays to lists for YAML."""
        data = super().model_dump(**kwargs)
        # Convert small arrays to lists for YAML storage
        if hasattr(self, 'xyz') and isinstance(self.xyz, np.ndarray):
            data['xyz'] = self.xyz.tolist()
        return data

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
        # Convert lists to arrays, keep arrays as-is
        if isinstance(v, list):
            return np.asarray(v, dtype=np.float32)
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v
    
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override model_dump to convert small arrays to lists for YAML."""
        data = super().model_dump(**kwargs)
        # Convert small arrays to lists for YAML storage
        if hasattr(self, 'wxyz') and isinstance(self.wxyz, np.ndarray):
            data['wxyz'] = self.wxyz.tolist()
        return data

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
        # Convert lists to arrays, keep arrays as-is
        if isinstance(v, list):
            return np.asarray(v, dtype=np.float32)
        if not isinstance(v, np.ndarray):
            v = np.asarray(v, dtype=np.float32)
        return v
    
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override model_dump to convert small arrays to lists for YAML."""
        data = super().model_dump(**kwargs)
        # Convert small arrays to lists for YAML storage
        if hasattr(self, 'joints') and isinstance(self.joints, np.ndarray):
            data['joints'] = self.joints.tolist()
        return data

    @classmethod
    def make_bimanual_joints(cls, pose_l: "ArmPose", pose_r: "ArmPose") -> "ArmPose":
        """Concatenate two arm poses to a single bimanual pose."""
        bimanual_joints = np.concatenate([pose_l.joints, pose_r.joints]).astype(np.float32)
        return cls(joints=bimanual_joints)

    @staticmethod
    def get_yaml_dir() -> str:
        """Get the directory containing pose YAML files."""
        return str(CONF_POSES_DIR)
