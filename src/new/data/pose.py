@dataclass
class Pose:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position in meters (xyz)."""
    wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Orientation quaternion (wxyz)."""