import os
from dataclasses import dataclass

import jax_dataclasses as jdc
import numpy as np
from jaxtyping import Array, Float
from safetensors.flax import load_file, save_file

from tatbot.data import Yaml
from tatbot.utils.log import get_logger

log = get_logger('data.stroke', '🔳')


@dataclass
class Stroke(Yaml):
    description: str
    """Natural language description of the path."""
    arm: str
    """Arm that will execute the path, either left or right."""
    ee_pos: np.ndarray | None = None # (N, 3)
    """End effector position in meters <x, y, z>."""
    ee_rot: np.ndarray | None = None # (N, 4)
    """End effector orientation in quaternion <x, y, z, w>."""
    dt: np.ndarray | None = None # (N, 1)
    """Time between poses in seconds."""
    pixel_coords: np.ndarray | None = None # (N, 2)
    """Numpy array of pixel coordinates for each pose in path <x (0-width), y (0-height)>."""
    gcode_text: str | None = None
    """G-code text for the stroke."""
    inkcap: str | None = None
    """Name of the inkcap which provided the ink for the stroke."""
    is_inkdip: bool = False
    """Whether the path is an inkdip."""
    color: str | None = None
    """Color of the stroke."""
    frame_path: str | None = None
    """Relative path to the frame image for this stroke, or None if not applicable.""" 

@dataclass
class StrokeList(Yaml):
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
    joints: Float[Array, "b l o 16"]
    """Joint positions in radians (URDF convention)."""
    dt: Float[Array, "b l o"]
    """Travel time from pose N to pose N+1 in seconds."""
    
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
    
    def offset_joints(self, stroke_idx: int, pose_idx: int, offset_idx_l: int, offset_idx_r: int) -> Float[Array, "16"]:
        left_joints = self.joints[stroke_idx, pose_idx, offset_idx_l][:8]
        right_joints = self.joints[stroke_idx, pose_idx, offset_idx_r][8:]
        return np.concatenate([left_joints, right_joints])
