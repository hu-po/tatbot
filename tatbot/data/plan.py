from dataclasses import dataclass
import os

from . import Yaml

@dataclass
class Plan(Yaml):
    left_arm_pen_names: list[str]
    """Name of pens that will be drawn using left arm."""
    right_arm_pen_names: list[str]
    """Name of pens that will be drawn using right arm."""

    image_width_m: float
    """Width of the image in meters."""
    image_height_m: float
    """Height of the image in meters."""
    image_width_px: int
    """Width of the image in pixels."""
    image_height_px: int
    """Height of the image in pixels."""

    path_length: int
    """All paths will be resampled to this length."""
    ik_batch_size: int
    """Batch size for IK computation."""
    path_dt_fast: float
    """Time between poses in seconds for fast movement."""
    path_dt_slow: float
    """Time between poses in seconds for slow movement."""

    inkdip_hover_offset: tuple[float, float, float]
    """<x, y, z> offset for inkdip hover in meters."""

    yaml_dir: str = os.path.expanduser("~/tatbot/config/plans")
    """Directory containing the plan configs."""