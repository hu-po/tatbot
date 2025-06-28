from dataclasses import dataclass, field, asdict
import os

import dacite
import numpy as np
import yaml

from _log import get_logger

log = get_logger('_ink')

@dataclass
class InkCap:
    """Individual cylindrical inkcap."""
    urdf_link_name: str = "inkcap_large"
    """URDF link name of the inkcap."""
    diameter_m: float = 0.008
    """Diameter of the inkcap (meters)."""
    depth_m: float = 0.01
    """Depth of the inkcap (meters)."""
    color: str = "black"
    """Natural language description of the color of the ink inside the inkcap."""

@dataclass
class InkConfig:
    urdf_link_name: str = "inkpalette"
    """URDF link name of the inkpalette."""
    inkcaps: dict[str, InkCap] = field(default_factory=lambda: {
        "small_1": InkCap(
            urdf_link_name="inkcap_small_1",
            color="pink"
        ),
        "large": InkCap(
            urdf_link_name="inkcap_large",
            diameter_m=0.014,
            depth_m=0.014,
            color="black"
        ),
        "small_2": InkCap(
            urdf_link_name="inkcap_small_2",
            color="blue"
        ),
        "small_3": InkCap(
            urdf_link_name="inkcap_small_3",
            color="white"
        ),
        "medium_1": InkCap(
            urdf_link_name="inkcap_medium_1",
            diameter_m=0.012,
            color="red"
        ),
        "medium_2": InkCap(
            urdf_link_name="inkcap_medium_2",
            diameter_m=0.012,
            color="green"
        ),
    })

    # TODO: these can be removed once we use URDF for pose tracking
    inkpalette_pos: np.ndarray = field(default_factory=lambda: np.array([-0.03, 0.0, -0.0055], dtype=np.float32))
    """position (xyz, meters) of the inkpalette in global frame."""
    inkpalette_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    """orientation quaternion (wxyz) of the inkpalette in global frame."""

    inkdip_hover_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.008], dtype=np.float32))
    """position offset (xyz, meters) when hovering over inkcap, relative to ee frame."""

    @classmethod
    def from_yaml(cls, filepath: str) -> "InkConfig":
        with open(os.path.expanduser(filepath), "r") as f:
            data = yaml.safe_load(f)
        return dacite.from_dict(
            cls,
            data,
            config=dacite.Config(type_hooks={np.ndarray: np.array})
        )

    def save_yaml(self, filepath: str):
        with open(os.path.expanduser(filepath), "w") as f:
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