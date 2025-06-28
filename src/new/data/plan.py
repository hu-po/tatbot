@dataclass
class Plan:
    name: str = "plan"
    """Name of the plan."""

    dirpath: str = ""
    """Path to the directory containing the plan files."""

    strokes: dict[str, Stroke] = field(default_factory=dict)
    """Dictionary of path metadata objects."""
    path_idx_to_strokes: list[list[Stroke]] = field(default_factory=list)
    """Map from pathbatch idx to list of strokes that make up that path."""

    image_width_m: float = 0.074 # A7 size
    """Width of the image in meters."""
    image_height_m: float = 0.105 # A7 size
    """Height of the image in meters."""
    image_width_px: int | None = None
    """Width of the image in pixels."""
    image_height_px: int | None = None
    """Height of the image in pixels."""

    ik_batch_size: int = 1024
    """Batch size for IK computation."""
    path_length: int = 108
    """All paths will be resampled to this length."""
    path_dt_fast: float = 0.1
    """Time between poses in seconds for fast movement."""
    path_dt_slow: float = 2.0
    """Time between poses in seconds for slow movement."""