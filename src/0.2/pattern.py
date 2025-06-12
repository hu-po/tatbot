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
    pixel_coords: Int[Array, "2"] = field(default_factory=lambda: jnp.array([0, 0]))
    """Pixel coordinates of the pose in image space (width, height), origin is top left."""
    metric_coords: Float[Array, "2"] = field(default_factory=lambda: jnp.array([0.0, 0.0]))
    """Metric (meters) coordinates of the pose in foo space, origin is center of foo (x, y)."""

@jdc.pytree_dataclass
class Path:
    positions: Float[Array, "N 3"] = field(default_factory=lambda: jnp.array([[0.0, 0.0, 0.0]]))
    orientations: Float[Array, "N 4"] = field(default_factory=lambda: jnp.array([[1.0, 0.0, 0.0, 0.0]]))
    pixel_coords: Int[Array, "N 2"] = field(default_factory=lambda: jnp.array([[0, 0]]))
    metric_coords: Float[Array, "N 2"] = field(default_factory=lambda: jnp.array([[0.0, 0.0]]))

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx) -> Pose:
        return Pose(
            pos=self.positions[idx],
            wxyz=self.orientations[idx],
            pixel_coords=self.pixel_coords[idx],
            metric_coords=self.metric_coords[idx],
        )

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

    @classmethod
    def from_json(cls, data: dict) -> "Pattern":
        paths = []
        for path_data in data.get("paths", []):
            poses_data = path_data.get("poses", [])
            if not poses_data:
                continue
            paths.append(
                Path(
                    positions=jnp.array([p.get("pos", [0, 0, 0]) for p in poses_data]),
                    orientations=jnp.array([p.get("wxyz", [1, 0, 0, 0]) for p in poses_data]),
                    pixel_coords=jnp.array([p.get("pixel_coords", [0, 0]) for p in poses_data]),
                    metric_coords=jnp.array([p.get("metric_coords", [0.0, 0.0]) for p in poses_data]),
                )
            )
        return cls(
            paths=paths,
            name=data.get("name", cls.name),
            width_m=data.get("width_m", cls.width_m),
            height_m=data.get("height_m", cls.height_m),
            width_px=data.get("width_px", cls.width_px),
            height_px=data.get("height_px", cls.height_px),
        )
    
COLORS: dict[str, tuple[int, int, int]] = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "purple": (255, 0, 255),
}

@jdc.jit
def offset_path(path: Path, offset: Float[Array, "3"]) -> Path:
    """Offsets all poses in a path by a given vector. JIT-compiled."""
    return path.replace(positions=path.positions + offset)

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
        if path.pixel_coords is None:
            continue
        pixel_coords = [np.array(pc) for pc in path.pixel_coords]

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

