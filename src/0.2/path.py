import logging
from dataclasses import dataclass, field

import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float, Int

log = logging.getLogger('tatbot')

@jdc.pytree_dataclass
class Pose:
    pos: Float[Array, "3"] = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
    """Position in meters (x, y, z)."""
    wxyz: Float[Array, "4"] = field(default_factory=lambda: jnp.array([1.0, 0.0, 0.0, 0.0]))
    """Orientation in quaternion (w, x, y, z)."""
    pixel_coords: Int[Array, "2"] | None = None
    """Pixel coordinates of the pose in image space (width, height), origin is top left."""
    metric_coords: Float[Array, "2"] | None = None
    """Metric (meters) coordinates of the pose in foo space, origin is center of foo (x, y)."""

@jdc.pytree_dataclass
class Path:
    poses: list[Pose] = field(default_factory=list)
    """Ordered list of poses defining a path."""

@dataclass
class Pattern:
    paths: list[Path] = field(default_factory=list)
    """Ordered list of paths defining a pattern."""
    name: str = "pattern"
    """Name of the pattern."""
    width_m: float = 0.04
    """Width of the pattern in meters."""
    height_m: float = 0.04
    """Height of the pattern in meters."""
    width_px: int = 256
    """Width of the pattern in pixels."""
    height_px: int = 256
    """Height of the pattern in pixels."""

def make_pathviz_image(pattern: Pattern):
    # looks like image with overlayed paths, colorgradient for index of pose in path (time)
    pass

def make_pathlen_image(pattern: Pattern):
    # two part image: top is path length histogram with colorbar, color is length
    # bottom is pathviz, but colored by path length histogram above
    pass

@jdc.jit
def resample_path(path: Path, num_points: int):
    # resample path to num_points length
    pass

def resample_pattern(pattern: Pattern, num_points: int):
    # resample each path in pattern to num_points length
    pass

