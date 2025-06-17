from dataclasses import dataclass, field
import os

import numpy as np
import yaml

from _log import get_logger
from _path import PathBatch

log = get_logger('_plan')

# plan objects stored inside folder, these are the filenames
METADATA_FILENAME: str = "meta.yaml"
IMAGE_FILENAME: str = "image.png"
PATHS_FILENAME: str = "paths.safetensors"

@dataclass
class Plan:
    name: str = "plan"
    """Name of the plan."""

    path_descriptions: list[str] = field(default_factory=list)
    """Descriptions for each path in the plan."""

    image_width_m: float = 0.04
    """Width of the image in meters."""
    image_height_m: float = 0.04
    """Height of the image in meters."""
    image_width_px: int = 256
    """Width of the image in pixels."""
    image_height_px: int = 256
    """Height of the image in pixels."""

    path_pad_len: int = 128
    """Length to pad paths to."""

    ee_design_pos: tuple[float, float, float] = (0.08, 0.0, 0.04)
    """position of the design ee transform."""
    ee_design_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the design ee transform."""

    hover_offset: tuple[float, float, float] = (0.0, 0.0, 0.006)
    """position offset when hovering over point, relative to current ee frame."""
    needle_offset: tuple[float, float, float] = (0.0, 0.0, -0.0065)
    """position offset to ensure needle touches skin, relative to current ee frame."""

    view_offset: tuple[float, float, float] = (0.0, -0.16, 0.16)
    """position offset when viewing design with right arm (relative to design ee frame)."""
    ee_view_wxyz: tuple[float, float, float, float] = (0.67360666, -0.25201478, 0.24747439, 0.64922119)
    """orientation quaternion (wxyz) of the view ee transform."""

    ee_inkcap_pos: tuple[float, float, float] = (0.16, 0.0, 0.04)
    """position of the inkcap ee transform."""
    ee_inkcap_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the inkcap ee transform."""
    dip_offset: tuple[float, float, float] = (0.0, 0.0, -0.029)
    """position offset when dipping inkcap (relative to current ee frame)."""

    ink_dip_every_n_poses: int = 64
    """Dip ink every N poses, will complete the full path before dipping again."""

    @classmethod
    def from_yaml(cls, dirpath: str) -> "Plan":
        log.info(f"⚙️ Loading plan from {dirpath}...")
        filepath = os.path.join(dirpath, METADATA_FILENAME)
        with open(filepath, "r") as f:
            return cls(**yaml.safe_load(f))

    @classmethod
    def image_filepath(cls, dirpath: str) -> str:
        filepath = os.path.join(dirpath, IMAGE_FILENAME)
        return filepath

    @classmethod
    def paths_np(cls, dirpath: str) -> np.ndarray:
        filepath = os.path.join(dirpath, PATHS_FILENAME)
        return np.array(PathBatch.load(filepath))
    
