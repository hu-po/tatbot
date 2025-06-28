from dataclasses import dataclass
from . import Yaml

INK_CONFIG_DIR = "~/tatbot/config/inks"

@dataclass
class Ink:
    name: str
    """Name of this ink."""
    rgb: tuple[int, int, int]
    """Color of the ink in rgb (0-255)."""

@dataclass
class InkCap:
    name: str
    """Name of the inkcap."""
    diameter_m: float
    """Diameter of the inkcap (meters)."""
    depth_m: float
    """Depth of the inkcap (meters)."""
    color: Ink
    """Color of the ink inside the inkcap."""

@dataclass
class InkPalette(Yaml):
    inkcaps: tuple[InkCap, ...]
    """Ordered list of inkcaps in the palette."""