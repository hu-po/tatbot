from dataclasses import dataclass

from tatbot.data import Yaml


@dataclass
class Ink(Yaml):
    name: str
    """Name of this ink."""
    rgb: tuple[int, int, int]
    """Color of the ink in rgb (0-255)."""

@dataclass
class InkCap(Yaml):
    name: str
    """Name of the inkcap."""
    diameter_m: float
    """Diameter of the inkcap (meters)."""
    depth_m: float
    """Depth of the inkcap (meters)."""
    ink: Ink | None = None
    """Ink inside the inkcap, if None the inkcap is empty."""

@dataclass
class Inks(Yaml):
    inkcaps: tuple[InkCap, ...]
    """Ordered list of inkcaps."""
    yaml_dir: str = "~/tatbot/config/inks"
    """Directory containing the config yaml files."""