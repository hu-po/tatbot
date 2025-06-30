from dataclasses import dataclass

import numpy as np

@dataclass
class Stroke:
    description: str
    """Natural language description of the path."""
    arm: str
    """Arm that will execute the path, either left or right."""
    color: str
    """Natural language description of the color of the path."""

    pixel_coords: np.ndarray
    """Numpy array of pixel coordinates for each pose in path <x (0-width), y (0-height)>."""

    meter_coords: np.ndarray
    """Numpy array of coordinates for each pose in path in meters <x, y, z>."""
    meters_center: np.ndarray
    """Center of Mass of the path in meters."""

    norm_coords: np.ndarray
    """Numpy array of coordinates for each pose in path in normalized image coordinates <x (0-1), y (0-1)>."""
    norm_center: np.ndarray
    """Center of Mass of the path in normalized image coordinates."""

    inkcap: str | None = None
    """Name of the inkcap which provided the ink for the stroke."""
    is_inkdip: bool = False
    """Whether the path is an inkdip."""