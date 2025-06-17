from dataclasses import dataclass, field
import os

import numpy as np
import yaml
from PIL import Image

from _log import get_logger
from _path import Path, PathBatch

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
    
    def add_pixel_paths(self, paths: list[tuple[int, int]]):
        scale_x = self.image_width_m / self.image_width_px
        scale_y = self.image_height_m / self.image_height_px
        num_paths = len(paths)
        positions = np.zeros((num_paths, self.path_pad_len, 3), dtype=np.float32)
        orientations = np.zeros((num_paths, self.path_pad_len, 4), dtype=np.float32)
        pixel_coords = np.zeros((num_paths, self.path_pad_len, 2), dtype=np.int32)
        metric_coords = np.zeros((num_paths, self.path_pad_len, 2), dtype=np.float32)
        goal_time = np.zeros((num_paths, self.path_pad_len, 1), dtype=np.float32)
        mask = np.zeros((num_paths, self.path_pad_len), dtype=np.uint8)
        for i, path_px in enumerate(paths):
            n = min(len(path_px), self.path_pad_len)
            positions_list = [[p[0] * scale_x, p[1] * scale_y, 0.0] for p in path_px[:n]]
            orientations_list = [[1.0, 0.0, 0.0, 0.0] for _ in range(n)] # TODO
            pixel_coords_list = [list(p) for p in path_px[:n]]
            metric_coords_list = [[p[0] * scale_x, p[1] * scale_y] for p in path_px[:n]]
            positions[i, :n, :] = positions_list
            orientations[i, :n, :] = orientations_list
            pixel_coords[i, :n, :] = pixel_coords_list
            metric_coords[i, :n, :] = metric_coords_list
            goal_time[i, :n, 0] = 0.0  # TODO
            mask[i, :n] = 1

        path_batch = PathBatch(
            positions=positions,
            orientations=orientations,
            pixel_coords=pixel_coords,
            metric_coords=metric_coords,
            goal_time=goal_time,
            mask=mask,
        )
        path_batch.save(os.path.join(self.dirpath, PATHS_FILENAME))


def wrap(plan: Plan, mesh) -> Plan:
    """Wrap a 2D plan onto a 3D mesh. """
    pass

def add_ink_dips(plan: Plan) -> Plan:
    """Add ink dips to the plan."""
    pass

def add_orientations(plan: Plan) -> Plan:
    """Add orientations to the plan."""
    pass
