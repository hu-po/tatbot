from typing import Optional, Tuple

from tatbot.data.base import BaseCfg
from tatbot.data.pose import Pose


class Ink(BaseCfg):
    name: str
    """Name of this ink."""
    rgb: Tuple[int, int, int]
    """Color of the ink in rgb (0-255)."""

class InkCap(BaseCfg):
    name: str
    """Name of the inkcap."""
    diameter_m: float
    """Diameter of the inkcap (meters)."""
    depth_m: float
    """Depth of the inkcap (meters)."""
    ink: Optional[Ink] = None
    """Ink inside the inkcap, if None the inkcap is empty."""
    pose: Optional[Pose] = None
    """Pose of the inkcap."""

class Inks(BaseCfg):
    inkcaps: Tuple[InkCap, ...]
    """Ordered list of inkcaps."""
