from dataclasses import dataclass

from tatbot.data import Yaml

@dataclass
class Tags(Yaml):
    family: str
    """Family of AprilTags to use."""
    size_m: float
    """Size of AprilTags: distance between detection corners (meters)."""
    enabled_tags: tuple[int, ...]
    """Enabled AprilTag IDs."""
    decision_margin: float
    """Minimum decision margin for AprilTag detection filtering."""

    yaml_dir: str = "~/tatbot/config/tags"
    """Directory containing the config yaml files."""