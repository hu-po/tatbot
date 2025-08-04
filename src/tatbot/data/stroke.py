import os
from typing import Optional

import jax_dataclasses as jdc
import numpy as np
from jaxtyping import Array, Float
from pydantic import field_validator
from safetensors.flax import load_file, save_file

from tatbot.data.base import BaseCfg
from tatbot.utils.log import get_logger

log = get_logger("data.stroke", "🔳")


class Stroke(BaseCfg):
    """A stroke in the tatbot system."""
    
    model_config = {'arbitrary_types_allowed': True}
    
    description: str
    """Natural language description of the path."""
    arm: str
    """Arm that will execute the path, either left or right."""

    meter_coords: Optional[np.ndarray] = None  # (N, 3)
    """Numpy array of meter coordinates for each pose in path <x, y, z> (design frame)."""
    pixel_coords: Optional[np.ndarray] = None  # (N, 2)
    """Numpy array of pixel coordinates for each pose in path <x (0-width), y (0-height)> (design frame)."""
    
    gcode_text: Optional[str] = None
    """G-code text for the stroke."""
    frame_path: Optional[str] = None
    """Relative path to the frame image for this stroke, or None if not applicable."""

    ee_pos: Optional[np.ndarray] = None  # (N, 3)
    """End effector position in meters <x, y, z> (world frame)."""
    ee_rot: Optional[np.ndarray] = None  # (N, 4)
    """End effector orientation in quaternion <x, y, z, w> (world frame)."""
    normals: Optional[np.ndarray] = None  # (N, 3)
    """Surface normals for each pose in the stroke <x, y, z> (world frame)."""

    is_rest: bool = False
    """Whether the stroke is a rest stroke."""
    is_inkdip: bool = False
    """Whether the stroke is an inkdip stroke."""

    inkcap: Optional[str] = None
    """Name of the inkcap which provided the ink for the stroke."""
    color: Optional[str] = None
    """Color of the stroke."""
    
    @field_validator('arm')
    def validate_arm(cls, v):
        if v not in ['left', 'right']:
            raise ValueError("arm must be 'left' or 'right'")
        return v
    
    @field_validator('meter_coords', 'pixel_coords', 'ee_pos', 'ee_rot', 'normals', mode='before')
    def convert_numpy_arrays(cls, v):
        """Convert numpy arrays to lists for serialization."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v    


class StrokeList(BaseCfg):
    """List of stroke pairs."""
    
    strokes: list[tuple[Stroke, Stroke]]
    """List of stroke pairs."""




@jdc.pytree_dataclass
class StrokeBatch:
    """
    batch of strokes:
    b = batch size
    l = stroke length
    o = offset num
    """

    ee_pos_l: Float[Array, "b l o 3"]
    """End effector frame position in meters (x, y, z) for left arm."""
    ee_pos_r: Float[Array, "b l o 3"]
    """End effector frame position in meters (x, y, z) for right arm."""
    ee_rot_l: Float[Array, "b l o 4"]
    """End effector frame orientation as quaternion (w, x, y, z) for left arm."""
    ee_rot_r: Float[Array, "b l o 4"]
    """End effector frame orientation as quaternion (w, x, y, z) for right arm."""
    joints: Float[Array, "b l o 14"]
    """Joint positions in radians (URDF convention)."""

    def save(self, filepath: str) -> None:
        log.debug(f"💾 Saving StrokeBatch to {filepath}")
        save_file({k: getattr(self, k) for k in self.__dataclass_fields__}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "StrokeBatch":
        filepath = os.path.expanduser(filepath)
        assert os.path.exists(filepath), f"❌ File {filepath} does not exist"
        log.debug(f"💾 Loading StrokeBatch from {filepath}")
        data = load_file(filepath)
        return cls(**data)

    def offset_joints(
        self, stroke_idx: int, pose_idx: int, offset_idx_l: int, offset_idx_r: int
    ) -> Float[Array, "14"]:
        left_joints = self.joints[stroke_idx, pose_idx, offset_idx_l][:7]
        right_joints = self.joints[stroke_idx, pose_idx, offset_idx_r][7:]
        return np.concatenate([left_joints, right_joints])
