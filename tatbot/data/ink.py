import os
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
    ink: Ink
    """Ink inside the inkcap."""
    name: str
    """Name of the inkcap."""
    diameter_m: float
    """Diameter of the inkcap (meters)."""
    depth_m: float
    """Depth of the inkcap (meters)."""

@dataclass
class InkPalette(Yaml):
    inkcaps: tuple[InkCap, ...]
    """Ordered list of inkcaps in the palette."""
    yaml_dir: str = os.path.expanduser("~/tatbot/config/inks")