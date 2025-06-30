from dataclasses import dataclass, field
import os

from . import Yaml

@dataclass
class Plan(Yaml):
    left_arm_pen_names: list[str]
    """Name of pens that will be drawn using left arm."""
    right_arm_pen_names: list[str]
    """Name of pens that will be drawn using right arm."""

    yaml_dir: str = os.path.expanduser("~/tatbot/config/plans")

    image_width_m: float = 0.074 # A7 size
    """Width of the image in meters."""
    image_height_m: float = 0.105 # A7 size
    """Height of the image in meters."""
    image_width_px: int | None = None
    """Width of the image in pixels."""
    image_height_px: int | None = None
    """Height of the image in pixels."""    

    points_per_path: int = 108
    """Number of points to sample per SVG path."""
    ik_batch_size: int = 1024
    """Batch size for IK computation."""
    path_length: int = 108
    """All paths will be resampled to this length."""
    path_dt_fast: float = 0.1
    """Time between poses in seconds for fast movement."""
    path_dt_slow: float = 2.0
    """Time between poses in seconds for slow movement."""