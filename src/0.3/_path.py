from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from jaxtyping import Array, Float, Int
from safetensors.flax import load_file, save_file


from _log import get_logger

log = get_logger('_path')
log.info(f"🧠 JAX devices: {jax.devices()}")

@jdc.pytree_dataclass
class Pose:
    pos: Float[Array, "3"] = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
    """Position in meters (x, y, z)."""
    wxyz: Float[Array, "4"] = field(default_factory=lambda: jnp.array([1.0, 0.0, 0.0, 0.0]))
    """Orientation as quaternion (w, x, y, z)."""
    pixel_coords: Int[Array, "2"] = field(default_factory=lambda: jnp.array([0, 0]))
    """Pixel coordinates of the pose in image space (width, height), origin is top left."""
    metric_coords: Float[Array, "2"] = field(default_factory=lambda: jnp.array([0.0, 0.0]))
    """Metric (meters) coordinates of the pose in foo space, origin is center of foo (x, y)."""

@jdc.pytree_dataclass
class Path:
    positions: Float[Array, "l 3"]
    orientations: Float[Array, "l 4"]
    pixel_coords: Int[Array, "l 2"]
    metric_coords: Float[Array, "l 2"]
    goal_time: Float[Array, "l 1"]
    """Travel time from pose N to pose N+1 in seconds."""

@jdc.pytree_dataclass
class PathBatch:
    positions: Float[Array, "b l 3"]
    orientations: Float[Array, "b l 4"]
    pixel_coords: Int[Array, "b l 2"]
    metric_coords: Float[Array, "b l 2"]
    goal_time: Float[Array, "b l 1"]
    mask: Int[Array, "b l"]
    """Paths are padded to same length, mask is 1 for valid poses in path."""

    def save(self, filepath: str) -> None:
        log.debug(f"💾 Saving PathBatch to {filepath}")
        save_file(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PathBatch":
        log.debug(f"💾 Loading PathBatch from {filepath}")
        return load_file(filepath)

@jax.jit
def offset_paths(paths: PathBatch, offset: Float[Array, "3"]) -> PathBatch:
    """Batched position (xyz) offset for all paths."""
    return paths.replace(positions = paths.positions + offset[None, None, :])