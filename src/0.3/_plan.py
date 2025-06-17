from dataclasses import dataclass, field, replace

import cv2
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from jaxtyping import Array, Float, Int
import json

from _log import COLORS, get_logger

log = get_logger('_plan')

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

@dataclass
class PathMetadata:
    description: str = ""
    """Description of the path in natural language."""

@jdc.pytree_dataclass
class PathBatch:
    positions: Float[Array, "b l 3"]
    orientations: Float[Array, "b l 4"]
    pixel_coords: Int[Array, "b l 2"]
    metric_coords: Float[Array, "b l 2"]
    mask: Int[Array, "b l"]
    """Paths are padded to same length, mask is 1 for valid poses in path."""

@jdc.pytree_dataclass
class Plan:
    paths: PathBatch

@dataclass
class PlanMetadata:
    name: str = "plan"
    """Name of the plan."""

    image_path: str = ""
    """Path to the image for the plan."""
    image_width_m: float = 0.04
    """Width of the image in meters."""
    image_height_m: float = 0.04
    """Height of the image in meters."""
    image_width_px: int = 256
    """Width of the image in pixels."""
    image_height_px: int = 256
    """Height of the image in pixels."""

    path_pad_len: int = 128
    """Length to pad paths to."""
