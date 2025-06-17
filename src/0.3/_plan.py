from dataclasses import dataclass, field
import os

import numpy as np
import yaml
from PIL import Image
import jax.numpy as jnp

from _ik import batch_ik
from _log import get_logger
from _path import Path, PathBatch, PixelPath

log = get_logger('_plan')

# plan objects stored inside folder, these are the filenames
METADATA_FILENAME: str = "meta.yaml"
IMAGE_FILENAME: str = "image.png"
PATHS_FILENAME: str = "paths.safetensors"

@dataclass
class Plan:
    name: str = "plan"
    """Name of the plan."""

    dirpath: str = ""
    """Path to the directory containing the plan files."""

    path_descriptions: dict[str, str] = field(default_factory=dict)
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
    path_dt_fast: float = 0.1
    """Time between poses in seconds for fast movement."""
    path_dt_slow: float = 2.0
    """Time between poses in seconds for slow movement."""

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

    ink_dip_interval: int = 2
    """Dip ink every 2 paths."""

    @classmethod
    def from_yaml(cls, dirpath: str) -> "Plan":
        log.info(f"⚙️ Loading plan from {dirpath}...")
        filepath = os.path.join(dirpath, METADATA_FILENAME)
        with open(filepath, "r") as f:
            return cls(**yaml.safe_load(f))

    @classmethod
    def image_np(cls, dirpath: str) -> np.ndarray:
        filepath = os.path.join(dirpath, IMAGE_FILENAME)
        return np.array(Image.open(filepath).convert("RGB"))

    @classmethod
    def paths_np(cls, dirpath: str) -> np.ndarray:
        filepath = os.path.join(dirpath, PATHS_FILENAME)
        return np.array(PathBatch.load(filepath))
    
    def save(self, image: np.ndarray = None):
        log.info(f"⚙️ Saving plan to {self.dirpath}")
        os.makedirs(self.dirpath, exist_ok=True)

        meta_path = os.path.join(self.dirpath, METADATA_FILENAME)
        log.info(f"⚙️💾 Saving metadata to {meta_path}")
        with open(meta_path, "w") as f:
            yaml.safe_dump(self.__dict__, f)

        if image is not None:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image_path = os.path.join(self.dirpath, IMAGE_FILENAME)
            log.info(f"⚙️💾 Saving image to {image_path}")
            image.save(image_path)

    def add_pixel_paths(self, pixel_paths: list[PixelPath]):
        num_paths = len(pixel_paths)
        log.info(f"⚙️ Adding {num_paths} pixel paths...")

        scale_x = self.image_width_m / self.image_width_px
        scale_y = self.image_height_m / self.image_height_px

        paths = []
        for path_idx, pixel_path in enumerate(pixel_paths):
            log.debug(f"🧮 Adding path {path_idx} of {num_paths}...")
            path = Path.padded(self.path_pad_len)
            self.path_descriptions[f'path_{path_idx}'] = pixel_path.description

            if len(pixel_path) + 2 > self.path_pad_len:
                log.warning(f"path {path_idx} has more than {self.path_pad_len} poses, truncating...")
                pixel_path = pixel_path[:self.path_pad_len - 2] # -2 for hover positions

            for i, (pw, ph) in enumerate(pixel_path):
                # pixel coordinates first need to be converted to meters
                x_m, y_m = pw * scale_x, ph * scale_y
                # center in design frame, add needle offset
                _pos_left = [
                    self.ee_design_pos[0] + x_m - self.image_width_m / 2,
                    self.ee_design_pos[1] + y_m - self.image_height_m / 2,
                    self.ee_design_pos[2] + self.needle_offset[2],
                ]
                path.ee_pos_l[i + 1, :] = _pos_left
                path.ee_wxyz_l[i + 1, :] = self.ee_design_wxyz
                # right hand just stares at center of design frame
                _pos_right = [
                    self.ee_design_pos[0] + self.view_offset[0],
                    self.ee_design_pos[1] + self.view_offset[1],
                    _pos_left[2] + self.view_offset[2],
                ]
                path.ee_pos_r[i + 1, :] = _pos_right
                path.ee_wxyz_r[i + 1, :] = self.ee_view_wxyz
            # add hover positions to the beginning and end of the path
            _hover_pos_start_left = [
                path.ee_pos_l[0, 0] + self.hover_offset[0],
                path.ee_pos_l[0, 1] + self.hover_offset[1],
                path.ee_pos_l[0, 2] + self.hover_offset[2],
            ]
            path.ee_pos_l[0, :] = _hover_pos_start_left
            path.ee_wxyz_l[0, :] = self.ee_design_wxyz
            path.ee_pos_l[-1, :] = [
                path.ee_pos_l[-1, 0] + self.hover_offset[0],
                path.ee_pos_l[-1, 1] + self.hover_offset[1],
                path.ee_pos_l[-1, 2] + self.hover_offset[2],
            ]
            path.ee_wxyz_l[-1, :] = self.ee_design_wxyz
            # make sure right hand has same number of poses, but no hover
            path.ee_pos_r[0, :] = path.ee_pos_r[1, :]
            path.ee_wxyz_r[0, :] = path.ee_wxyz_r[1, :]
            path.ee_pos_r[-1, :] = path.ee_pos_r[-2, :]
            path.ee_wxyz_r[-1, :] = path.ee_wxyz_r[-2, :]
            # compute joint positions
            target_wxyz = jnp.stack([path.ee_wxyz_l, path.ee_wxyz_r], axis=1)
            target_pos = jnp.stack([path.ee_pos_l, path.ee_pos_r], axis=1)
            path.joints = batch_ik(
                target_wxyz=target_wxyz,
                target_pos=target_pos,
            )
            # slow movement at the hover positions
            path.dt[0, 0] = self.path_dt_slow
            path.dt[1:-1, 0] = self.path_dt_fast
            path.dt[-1, 0] = self.path_dt_slow

            paths.append(path)

        path_batch = PathBatch.from_paths(paths)
        path_batch.save(os.path.join(self.dirpath, PATHS_FILENAME))


def wrap(plan: Plan, mesh) -> Plan:
    """Wrap a 2D plan onto a 3D mesh. """
    pass
