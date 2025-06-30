from dataclasses import dataclass

import numpy as np


@dataclass
class Stroke:
    idx: int
    """Index of the stroke in the plan and strokebatch."""
    description: str
    """Natural language description of the path."""
    arm: str
    """Arm that will execute the path, either left or right."""
    pixel_coords: np.ndarray # (N, 2)
    """Numpy array of pixel coordinates for each pose in path <x (0-width), y (0-height)>."""
    ee_pos: np.ndarray # (N, 3)
    """End effector position in meters <x, y, z>."""
    ee_wxyz: np.ndarray # (N, 4)
    """End effector orientation in quaternion <x, y, z, w>."""
    dt: np.ndarray # (N, 1)
    """Time between poses in seconds."""
    inkcap: str | None = None
    """Name of the inkcap which provided the ink for the stroke."""
    is_inkdip: bool = False
    """Whether the path is an inkdip."""
    is_alignment: bool = False
    """Whether the path is an alignment stroke."""