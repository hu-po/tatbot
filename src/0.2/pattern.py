import logging
from dataclasses import dataclass, field

import cv2
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
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
    image_np: np.ndarray | None = field(default=None, repr=False, compare=False)
    """Optional image for visualization."""

def make_pathviz_image(pattern: Pattern) -> np.ndarray:
    """Creates an image with overlayed paths from a pattern.

    The path is visualized with a color gradient indicating the order of poses (time).

    Args:
        pattern: The pattern containing the paths and optional image to visualize.

    Returns:
        The visualization image as a numpy array (BGR).
    """
    if pattern.image_np is None:
        path_viz_np = np.full((pattern.height_px, pattern.width_px, 3), 255, dtype=np.uint8)
    else:
        path_viz_np = pattern.image_np.copy()

    for path in pattern.paths:
        # Convert JAX arrays to list of numpy arrays for processing
        pixel_coords = [np.array(pose.pixel_coords) for pose in path.poses if pose.pixel_coords is not None]

        if len(pixel_coords) < 2:
            continue

        path_indices = np.linspace(0, 255, len(pixel_coords), dtype=np.uint8)
        colormap = cv2.applyColorMap(path_indices.reshape(-1, 1), cv2.COLORMAP_JET)

        for path_idx in range(len(pixel_coords) - 1):
            p1 = tuple(pixel_coords[path_idx].astype(int))
            p2 = tuple(pixel_coords[path_idx + 1].astype(int))
            color = colormap[path_idx][0].tolist()
            cv2.line(path_viz_np, p1, p2, color, 2)

    return path_viz_np

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

