from dataclasses import dataclass, field, asdict

import dacite
import numpy as np
import yaml

from _log import get_logger

log = get_logger('_ink')

@dataclass
class InkCap:
    """Individual cylindrical inkcap."""
    palette_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Relative position (meters) of the inkcap in the palette frame (x, y, z)."""
    diameter_m: float = 0.008
    """Diameter of the inkcap (meters)."""
    depth_m: float = 0.01
    """Depth of the inkcap (meters)."""
    color: str = "black"
    """Natural language description of the color of the ink inside the inkcap."""

@dataclass
class InkConfig:
    inkcaps: dict[str, InkCap] = field(default_factory=lambda: {
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
            palette_pos=[0.0, 0.0, 0.0], # center of palette is center of big inkcap
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
            palette_pos=[0.018, -0.025, 0.0],
            diameter_m=0.012,
            color="red"
        ),
        "medium_1": InkCap(
            palette_pos=[0.0, -0.02, 0.0],
            diameter_m=0.012,
            color="green"
        ),
        "medium_2": InkCap(
            palette_pos=[-0.018, -0.025, 0.0],
            diameter_m=0.012,
            color="blue"
        ),
    })

    inkpalette_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.04], dtype=np.float32))
    """position (xyz, meters) of the inkpalette in global frame."""
    inkpalette_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    """orientation quaternion (wxyz) of the inkpalette in global frame."""

    inkdip_hover_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.02], dtype=np.float32))
    """position offset (xyz, meters) when hovering over inkcap, relative to ee frame."""

    @classmethod
    def from_yaml(cls, filepath: str) -> "InkConfig":
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return dacite.from_dict(
            cls,
            data,
            config=dacite.Config(type_hooks={np.ndarray: np.array})
        )

    def save_yaml(self, filepath: str):
        with open(filepath, "w") as f:
            yaml.safe_dump(asdict(self), f)

    def find_best_inkcap(self, color: str) -> str:
        """Find the best inkcap for a given color."""
        for name, inkcap in self.inkcaps.items():
            # TODO: something more sophisticated here
            if inkcap.color == color:
                log.debug(f"🎨 found inkcap {name} for color {color}")
                return name
        _msg = f"🎨❌ No inkcap found for color {color}"
        log.error(_msg)
        raise ValueError(_msg)