from typing import Tuple

from tatbot.data.base import BaseCfg


class Tags(BaseCfg):
    family: str
    """Family of AprilTags to use."""
    size_m: float
    """Size of AprilTags: distance between detection corners (meters)."""
    enabled_tags: Tuple[int, ...]
    """Enabled AprilTag IDs."""
    decision_margin: float
    """Minimum decision margin for AprilTag detection filtering."""
