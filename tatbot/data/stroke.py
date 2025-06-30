@dataclass
class Stroke:
    description: str | None = None
    """Natural language description of the path."""
    arm: str | None = None
    """Arm that will execute the path, either left or right."""
    color: str | None = None
    """Natural language description of the color of the path."""

    pixel_coords: np.ndarray | None = None
    """Numpy array of pixel coordinates for each pose in path <x (0-width), y (0-height)>."""

    meter_coords: np.ndarray | None = None
    """Numpy array of coordinates for each pose in path in meters <x, y, z>."""
    meters_center: np.ndarray | None = None
    """Center of Mass of the path in meters."""

    norm_coords: np.ndarray | None = None
    """Numpy array of coordinates for each pose in path in normalized image coordinates <x (0-1), y (0-1)>."""
    norm_center: np.ndarray | None = None
    """Center of Mass of the path in normalized image coordinates."""

    is_inkdip: bool = False
    """Whether the path is an inkdip."""
    inkcap: str | None = None
    """Name of the inkcap which provided the ink for the stroke."""