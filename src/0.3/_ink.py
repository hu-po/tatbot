from dataclasses import dataclass

from _log import get_logger

log = get_logger('_ink')

@dataclass
class InkCap:
    """Individual cylindrical inkcap."""
    palette_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Relative position (meters) of the inkcap in the palette frame (x, y, z)."""
    diameter_m: float = 0.008
    """Diameter of the inkcap (meters)."""
    depth_m: float = 0.01
    """Depth of the inkcap (meters)."""
    color: str = "black"
    """Natural language description of the color of the ink inside the inkcap."""

@dataclass
class InkPalette:
    inkcaps: dict[str, InkCap] = {
        # outer row
        # "small_0": InkCap(
        #     palette_pos=(0.0, 0.0, 0.0),
        #     color=""
        # ),
        # "small_1": InkCap(
        #     palette_pos=(0.0, 0.0, 0.0),
        #     color=""
        # ),
        "large_0": InkCap(
            palette_pos=(0.0, 0.0, 0.0), # center of palette is center of big inkcap
            diameter_m=0.014,
            depth_m=0.014,
            color="black"
        ),
        # "small_2": InkCap(
        #     palette_pos=(0.0, 0.0, 0.0),
        #     color=""
        # ),
        # "small_3": InkCap(
        #     palette_pos=(0.0, 0.0, 0.0),
        #     color=""
        # ),
        # inner row
        "medium_0": InkCap(
            palette_pos=(0.018, -0.025, 0.0),
            diameter_m=0.012,
            color="red"
        ),
        "medium_1": InkCap(
            palette_pos=(0.0, -0.02, 0.0),
            diameter_m=0.012,
            color="green"
        ),
        "medium_2": InkCap(
            palette_pos=(-0.018, -0.025, 0.0),
            diameter_m=0.012,
            color="blue"
        ),

    }