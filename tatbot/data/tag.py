from dataclasses import dataclass


@dataclass
class TagConfig:
    family: str = "tag16h5"
    """Family of AprilTags to use."""
    size_m: float = 0.041
    """Size of AprilTags: distance between detection corners (meters)."""
    enabled_tags: dict[int, str] = field(default_factory=lambda: {
        6: "arm_l",
        7: "arm_r",
        9: "palette",
        10: "origin",
        11: "skin",
    })
    """ Dictionary of enabled AprilTag IDs."""
    urdf_link_names: dict[int, str] = field(default_factory=lambda: {
        6: "tag6",
        7: "tag7",
        9: "tag9",
        10: "tag10",
        11: "tag11",
    })
    """ Dictionary of AprilTag IDs to URDF link names."""
    decision_margin: float = 20.0
    """Minimum decision margin for AprilTag detection filtering."""
