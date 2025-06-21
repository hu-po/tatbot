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
    
@dataclass
class Stroke:
    description: str | None = None
    """Natural language description of the path."""
    arm: str | None = None
    """Arm that will execute the path, either left or right."""
    color: str = "solid black"
    """Natural language description of the color of the path."""

    pixel_coords: list[list[int]] | None = None
    """List of pixel coordinates for each pose in path."""
    meter_coords: list[list[float]] | None = None
    """List of coordinates for each pose in path in meters."""
    meters_center: tuple[float, float] | None = None
    """Center of Mass of the path in meters."""

    norm_coords: list[list[float]] | None = None
    """List of coordinates for each pose in path in normalized image coordinates (0-1)."""
    norm_center: tuple[float, float] | None = None
    """Center of Mass of the path in normalized image coordinates (0-1)."""

    is_inkdip: bool = False
    """Whether the path is an inkdip."""
    is_completed: bool = False
    """Whether the path is completed."""
    completion_time: float | None = None
    """Time taken to complete the path in seconds."""

    def __len__(self) -> int:
        return len(self.pixel_coords)

    @classmethod
    def from_yaml(cls, filepath: str) -> list["Stroke"]:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return [dacite.from_dict(cls, p) for p in data]

    @staticmethod
    def to_yaml(pathmetas: list["Stroke"], filepath: str):
        with open(filepath, "w") as f:
            yaml.safe_dump([asdict(p) for p in pathmetas], f)