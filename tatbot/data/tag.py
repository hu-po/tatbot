from dataclasses import dataclass

from tatbot.data import Yaml

@dataclass
class TagConfig(Yaml):
    family: str = "tag16h5"
    """Family of AprilTags to use."""
    size_m: float = 0.041
    """Size of AprilTags: distance between detection corners (meters)."""
    enabled_tags: tuple[int, ...] = (6, 7, 9, 10, 11)
    """Enabled AprilTag IDs."""
    decision_margin: float = 20.0
    """Minimum decision margin for AprilTag detection filtering."""
