import os
from dataclasses import dataclass

import jax_dataclasses as jdc
import numpy as np
from jaxtyping import Array, Float
from safetensors.flax import load_file, save_file

from tatbot.data import Yaml
from tatbot.utils.log import get_logger

log = get_logger("data.stroke", "🔳")


@dataclass
class Stroke(Yaml):
    description: str
    """Natural language description of the path."""
    arm: str
    """Arm that will execute the path, either left or right."""

    meter_coords: np.ndarray | None = None  # (N, 3)
    """Numpy array of meter coordinates for each pose in path <x, y, z> (design frame)."""
    pixel_coords: np.ndarray | None = None  # (N, 2)
    """Numpy array of pixel coordinates for each pose in path <x (0-width), y (0-height)> (design frame)."""
    
    gcode_text: str | None = None
    """G-code text for the stroke."""
    frame_path: str | None = None
    """Relative path to the frame image for this stroke, or None if not applicable."""

    ee_pos: np.ndarray | None = None  # (N, 3)
    """End effector position in meters <x, y, z> (world frame)."""
    ee_rot: np.ndarray | None = None  # (N, 4)
    """End effector orientation in quaternion <x, y, z, w> (world frame)."""
    normals: np.ndarray | None = None  # (N, 3)
    """Surface normals for each pose in the stroke <x, y, z> (world frame)."""
    dt: np.ndarray | None = None  # (N, 1)
    """Time between poses in seconds."""

    is_rest: bool = False
    """Whether the stroke is a rest stroke."""
    is_inkdip: bool = False
    """Whether the stroke is an inkdip stroke."""

    inkcap: str | None = None
    """Name of the inkcap which provided the ink for the stroke."""
    color: str | None = None
    """Color of the stroke."""    


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
    joints: Float[Array, "b l o 14"]
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

    def offset_joints(
        self, stroke_idx: int, pose_idx: int, offset_idx_l: int, offset_idx_r: int
    ) -> Float[Array, "14"]:
        left_joints = self.joints[stroke_idx, pose_idx, offset_idx_l][:7]
        right_joints = self.joints[stroke_idx, pose_idx, offset_idx_r][7:]
        return np.concatenate([left_joints, right_joints])
