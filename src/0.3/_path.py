from dataclasses import dataclass, asdict

import dacite
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float, Int
import numpy as np
from safetensors.flax import load_file, save_file
import yaml

from _log import get_logger

log = get_logger('_path')

@dataclass
class PixelPath:
    pixels: list[list[int]]
    """List of pixel coordinates in the image."""
    color: str = "black"
    """Natural language description of the color of the path."""
    description: str = ""
    """Description of the path."""

    def __len__(self) -> int:
        return len(self.pixels)

    @classmethod
    def from_yaml(cls, filepath: str) -> list["PixelPath"]:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return [dacite.from_dict(cls, p) for p in data]

    @staticmethod
    def to_yaml(pixelpaths: list["PixelPath"], filepath: str):
        with open(filepath, "w") as f:
            yaml.safe_dump([asdict(p) for p in pixelpaths], f)

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
    dt: Float[Array, "l"]
    """Travel time from pose N to pose N+1 in seconds."""
    mask: Int[Array, "l"]
    """Mask for valid poses in path (1 for valid, 0 for padding)."""

    @classmethod
    def padded(cls, pad_len: int) -> "Path":
        log.debug(f"🔳 Creating empty padded path (pad_len={pad_len})...")
        return cls(
            ee_pos_l=np.zeros((pad_len, 3), dtype=np.float32),
            ee_pos_r=np.zeros((pad_len, 3), dtype=np.float32),
            ee_wxyz_l=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (pad_len, 1)),
            ee_wxyz_r=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (pad_len, 1)),
            joints=np.zeros((pad_len, 16), dtype=np.float32),
            dt=np.zeros((pad_len,), dtype=np.float32),
            mask=np.zeros((pad_len,), dtype=np.int32),
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
    dt: Float[Array, "b l"]
    """Travel time from pose N to pose N+1 in seconds."""
    mask: Int[Array, "b l"]
    """Paths are padded to same length, mask is 1 for valid poses in path."""

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "PathBatch":
        log.debug(f"🔳 Creating PathBatch from {len(paths)} paths...")
        return cls(
            ee_pos_l=jnp.array([path.ee_pos_l for path in paths]),
            ee_pos_r=jnp.array([path.ee_pos_r for path in paths]),
            ee_wxyz_l=jnp.array([path.ee_wxyz_l for path in paths]),
            ee_wxyz_r=jnp.array([path.ee_wxyz_r for path in paths]),
            joints=jnp.array([path.joints for path in paths]),
            dt=jnp.array([path.dt for path in paths]),
            mask=jnp.array([path.mask for path in paths]),
        )

    def save(self, filepath: str) -> None:
        log.debug(f"🔳💾 Saving PathBatch to {filepath}")
        save_file({k: getattr(self, k) for k in self.__dataclass_fields__}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PathBatch":
        log.debug(f"🔳💾 Loading PathBatch from {filepath}")
        data = load_file(filepath)
        return cls(**data)