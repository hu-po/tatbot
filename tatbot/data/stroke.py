from dataclasses import dataclass

import numpy as np

from tatbot.data import Yaml


@dataclass
class Stroke(Yaml):
    description: str
    """Natural language description of the path."""
    arm: str
    """Arm that will execute the path, either left or right."""
    ee_pos: np.ndarray # (N, 3)
    """End effector position in meters <x, y, z>."""
    ee_rot: np.ndarray # (N, 4)
    """End effector orientation in quaternion <x, y, z, w>."""
    dt: np.ndarray # (N, 1)
    """Time between poses in seconds."""
    pixel_coords: np.ndarray | None = None # (N, 2)
    """Numpy array of pixel coordinates for each pose in path <x (0-width), y (0-height)>."""
    svg_path_obj: str | None = None
    """SVG path object in string format."""
    inkcap: str | None = None
    """Name of the inkcap which provided the ink for the stroke."""
    is_inkdip: bool = False
    """Whether the path is an inkdip."""
    is_alignment: bool = False
    """Whether the path is an alignment stroke."""
    color: str | None = None
    """Color of the stroke."""
    frame_path: str | None = None
    """Relative path to the frame image for this stroke, or None if not applicable.""" 

@dataclass
class StrokeList(Yaml):
    strokes: list[tuple[Stroke, Stroke]]
    """List of stroke pairs."""