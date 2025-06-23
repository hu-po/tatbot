from dataclasses import dataclass, asdict

import dacite
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float
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

    @classmethod
    def empty(cls, length: int) -> "Path":
        log.debug(f"🔳 Creating empty path of length {length}...")
        return cls(
            ee_pos_l=np.zeros((length, 3), dtype=np.float32),
            ee_pos_r=np.zeros((length, 3), dtype=np.float32),
            ee_wxyz_l=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (length, 1)),
            ee_wxyz_r=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (length, 1)),
            joints=np.zeros((length, 16), dtype=np.float32),
            dt=np.zeros((length,), dtype=np.float32),
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
    color: str | None = None
    """Natural language description of the color of the path."""

    pixel_coords: np.ndarray | None = None
    """Numpy array of pixel coordinates for each pose in path <x (0-width), y (0-height)>."""

    meter_coords: np.ndarray | None = None
    """Numpy array of coordinates for each pose in path in meters <x, y, z>."""
    meters_center: np.ndarray | None = None
    """Center of Mass of the path in meters."""

    norm_coords: np.ndarray | None = None
    """Numpy array of coordinates for each pose in path in normalized image coordinates <x (0-1), y (0-1)>."""
    norm_center: np.ndarray | None = None
    """Center of Mass of the path in normalized image coordinates."""

    is_inkdip: bool = False
    """Whether the path is an inkdip."""
    inkcap: str | None = None
    """Name of the inkcap which provided the ink for the stroke."""

    is_completed: bool = False
    """Whether the path is completed."""
    completion_time: float | None = None
    """Time taken to complete the path in seconds."""

    def __len__(self) -> int:
        if self.pixel_coords is None:
            return 0
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