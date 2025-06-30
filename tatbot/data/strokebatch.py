import jax_dataclasses as jdc
import jax.numpy as jnp
from jaxtyping import Array, Float
from safetensors.flax import load_file, save_file

from tatbot.utils.log import get_logger

log = get_logger('strokebatch', '🔳')

@jdc.pytree_dataclass
class StrokeBatch:
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
    def empty(cls, length: int) -> "StrokeBatch":
        log.debug(f"Creating empty StrokeBatch of length {length}...")
        return cls(
            ee_pos_l=jnp.zeros((1, length, 3), dtype=jnp.float32),
            ee_pos_r=jnp.zeros((1, length, 3), dtype=jnp.float32),
            ee_wxyz_l=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32), (1, length, 1)),
            ee_wxyz_r=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32), (1, length, 1)),
            joints=jnp.zeros((1, length, 16), dtype=jnp.float32),
            dt=jnp.full((1, length,), 0.01, dtype=jnp.float32), # seconds
        )
    
    def save(self, filepath: str) -> None:
        log.debug(f"💾 Saving StrokeBatch to {filepath}")
        save_file({k: getattr(self, k) for k in self.__dataclass_fields__}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "StrokeBatch":
        log.debug(f"💾 Loading PathBatch from {filepath}")
        data = load_file(filepath)
        return cls(**data)