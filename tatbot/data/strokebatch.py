import os

import jax_dataclasses as jdc
from jaxtyping import Array, Float
from safetensors.flax import load_file, save_file

from tatbot.utils.log import get_logger

log = get_logger('strokebatch', '🔳')

@jdc.pytree_dataclass
class StrokeBatch:
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