from dataclasses import dataclass

import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float, Int
import numpy as np
from safetensors.flax import load_file, save_file

from _log import get_logger

log = get_logger('_path')

@dataclass
class PixelPath:
    pixels: list[tuple[int, int]]
    """List of pixel coordinates in the image."""
    color: str = "black"
    """Natural language description of the color of the path."""
    description: str = ""
    """Description of the path."""

    def __len__(self) -> int:
        return len(self.pixels)

    def to_dict(self):
        return {
            "pixels": self.pixels,
            "color": self.color,
            "description": self.description,
        }

@dataclass
class Path:
    ee_pos_l: Float[Array, "l 3"]
    """End effector frame position in meters (x, y, z) for left arm."""
    ee_pos_r: Float[Array, "l 3"]
    """End effector frame position in meters (x, y, z) for right arm."""
    ee_wxyz_l: Float[Array, "l 4"]
    """End effector frame orientation as quaternion (w, x, y, z) for left arm."""
    ee_wxyz_r: Float[Array, "l 4"]
    """End effector frame orientation as quaternion (w, x, y, z) for right arm."""
    joints: Float[Array, "l 16"]
    """Joint positions in radians (URDF convention)."""
    dt: Float[Array, "l 1"]
    """Travel time from pose N to pose N+1 in seconds."""

    @classmethod
    def padded(cls, pad_len: int) -> "Path":
        return cls(
            ee_pos_l=np.zeros((pad_len, 3), dtype=np.float32),
            ee_pos_r=np.zeros((pad_len, 3), dtype=np.float32),
            ee_wxyz_l=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (pad_len, 1)),
            ee_wxyz_r=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (pad_len, 1)),
            joints=np.zeros((pad_len, 16), dtype=np.float32),
            dt=np.zeros((pad_len, 1), dtype=np.float32),
        )

@jdc.pytree_dataclass
class PathBatch:
    ee_pos_l: Float[Array, "b l 3"]
    """End effector frame position in meters (x, y, z) for left arm."""
    ee_pos_r: Float[Array, "b l 3"]
    """End effector frame position in meters (x, y, z) for right arm."""
    ee_wxyz_l: Float[Array, "b l 4"]
    """End effector frame orientation as quaternion (w, x, y, z) for left arm."""
    ee_wxyz_r: Float[Array, "b l 4"]
    """End effector frame orientation as quaternion (w, x, y, z) for right arm."""
    joints: Float[Array, "b l 16"]
    """Joint positions in radians (URDF convention)."""
    dt: Float[Array, "b l 1"]
    """Travel time from pose N to pose N+1 in seconds."""
    mask: Int[Array, "b l"]
    """Paths are padded to same length, mask is 1 for valid poses in path."""

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "PathBatch":
        return cls(
            ee_pos_l=jnp.array([path.ee_pos_l for path in paths]),
            ee_pos_r=jnp.array([path.ee_pos_r for path in paths]),
            ee_wxyz_l=jnp.array([path.ee_wxyz_l for path in paths]),
            ee_wxyz_r=jnp.array([path.ee_wxyz_r for path in paths]),
            joints=jnp.array([path.joints for path in paths]),
            dt=jnp.array([path.dt for path in paths]),
            mask=jnp.ones((len(paths), max(len(path.ee_pos_l) for path in paths))),
        )

    def save(self, filepath: str) -> None:
        log.debug(f"💾 Saving PathBatch to {filepath}")
        save_file({k: getattr(self, k) for k in self.__dataclass_fields__}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PathBatch":
        log.debug(f"💾 Loading PathBatch from {filepath}")
        data = load_file(filepath)
        return cls(**data)