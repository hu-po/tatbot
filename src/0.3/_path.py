import jax_dataclasses as jdc
from jaxtyping import Array, Float, Int
from safetensors.flax import load_file, save_file

from _log import get_logger

log = get_logger('_path')

@jdc.pytree_dataclass
class PathBatch:
    ee_pos_l: Float[Array, "b l 3"]
    """End effector frame position in meters (x, y, z) for left arm."""
    ee_pos_r: Float[Array, "b l 3"]
    """End effector frame position in meters (x, y, z) for right arm."""
    wxyz_l: Float[Array, "b l 4"]
    """End effector frame orientation as quaternion (w, x, y, z) for left arm."""
    wxyz_r: Float[Array, "b l 4"]
    """End effector frame orientation as quaternion (w, x, y, z) for right arm."""
    joints: Float[Array, "b l 16"]
    """Joint positions in radians (URDF convention)."""
    dt: Float[Array, "b l 1"]
    """Travel time from pose N to pose N+1 in seconds."""
    mask: Int[Array, "b l"]
    """Paths are padded to same length, mask is 1 for valid poses in path."""

    def save(self, filepath: str) -> None:
        log.debug(f"💾 Saving PathBatch to {filepath}")
        save_file({k: getattr(self, k) for k in self.__dataclass_fields__}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PathBatch":
        log.debug(f"💾 Loading PathBatch from {filepath}")
        return load_file(filepath)