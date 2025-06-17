from dataclasses import dataclass, field
import os

import numpy as np
import yaml
from PIL import Image

from _ik import batch_ik
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

    def add_pixel_paths(self, paths: list[tuple[int, int]]):

        num_paths = len(paths)
        num_ink_dips = 1 + num_paths // self.ink_dip_interval
        total_num_paths = num_paths + num_ink_dips

        ee_pos_l = np.zeros((total_num_paths, self.path_pad_len, 3), dtype=np.float32)
        ee_pos_r = np.zeros((total_num_paths, self.path_pad_len, 3), dtype=np.float32)
        wxyz_l = np.zeros((total_num_paths, self.path_pad_len, 4), dtype=np.float32)
        wxyz_r = np.zeros((total_num_paths, self.path_pad_len, 4), dtype=np.float32)
        joints = np.zeros((total_num_paths, self.path_pad_len, 16), dtype=np.float32)
        dt = np.zeros((total_num_paths, self.path_pad_len, 1), dtype=np.float32)
        mask = np.zeros((total_num_paths, self.path_pad_len), dtype=np.uint8)

        scale_x = self.image_width_m / self.image_width_px
        scale_y = self.image_height_m / self.image_height_px

        path_idx = 0
        for i in range(total_num_paths):
            _ee_pos_l = []
            _ee_pos_r = []
            _wxyz_l = []
            _wxyz_r = []
            _dt = []
            if i % self.ink_dip_interval == 0:
                self.path_descriptions.append("dip ink")
                # hover over inkcap
                _ee_pos_l.append(self.ee_inkcap_pos)
                _wxyz_l.append(self.ee_inkcap_wxyz)
                _ee_pos_r.append(self.ee_inkcap_pos + self.view_offset)
                _wxyz_r.append(self.ee_view_wxyz)
                _dt.append(3.0)
                # dip into inkcap
                _ee_pos_l.append(self.ee_inkcap_pos + self.dip_offset)
                _wxyz_l.append(self.ee_inkcap_wxyz)
                _ee_pos_r.append(self.ee_inkcap_pos + self.view_offset)
                _wxyz_r.append(self.ee_view_wxyz)
                _dt.append(3.0)
                # pause inside inkcap
                _ee_pos_l.append(self.ee_inkcap_pos + self.dip_offset)
                _wxyz_l.append(self.ee_inkcap_wxyz)
                _ee_pos_r.append(self.ee_inkcap_pos + self.view_offset)
                _wxyz_r.append(self.ee_view_wxyz)
                _dt.append(3.0)
                # retract from inkcap
                _ee_pos_l.append(self.ee_inkcap_pos)
                _wxyz_l.append(self.ee_inkcap_wxyz)
                _ee_pos_r.append(self.ee_inkcap_pos + self.view_offset)
                _wxyz_r.append(self.ee_view_wxyz)
                _dt.append(3.0)
            else:
                path_px = paths[path_idx]
                self.path_descriptions.append(f"path {path_idx} of {num_paths}")
                path_idx += 1

                pad_len_no_hover = self.path_pad_len - 2
                if len(path_px) > pad_len_no_hover:
                    log.warning(f"path {path_idx} has more than {pad_len_no_hover} poses, truncating...")
                    path_px = path_px[:pad_len_no_hover]

                for pw, ph in path_px:
                    # pixel coordinates first need to be converted to meters
                    x_m, y_m = pw * scale_x, ph * scale_y
                    # center in design frame, add needle offset
                    _pos_left = [
                        self.ee_design_pos[0] + x_m - self.image_width_m / 2,
                        self.ee_design_pos[1] + y_m - self.image_height_m / 2,
                        self.ee_design_pos[2] + self.needle_offset[2],
                    ]
                    _ee_pos_l.append(_pos_left)
                    _wxyz_l.append(self.ee_design_wxyz)
                    # right hand just stares at center of design frame
                    _pos_right = [
                        self.ee_design_pos[0] + self.view_offset[0],
                        self.ee_design_pos[1] + self.view_offset[1],
                        _pos_left[2] + self.view_offset[2],
                    ]
                    _ee_pos_r.append(_pos_right)
                    _wxyz_r.append(self.ee_view_wxyz)
                # add hover positions to the beginning and end of the path
                _hover_pos_start_left = [
                    ee_pos_l[0][0] + self.hover_offset[0],
                    ee_pos_l[0][1] + self.hover_offset[1],
                    ee_pos_l[0][2] + self.hover_offset[2],
                ]
                _ee_pos_l.insert(0, _hover_pos_start_left)
                _wxyz_l.insert(0, self.ee_design_wxyz)
                _hover_pos_end_left = [
                    ee_pos_l[-1][0] + self.hover_offset[0],
                    ee_pos_l[-1][1] + self.hover_offset[1],
                    ee_pos_l[-1][2] + self.hover_offset[2],
                ]
                _ee_pos_l.append(_hover_pos_end_left)
                _wxyz_l.append(self.ee_design_wxyz)
                # make sure right hand has same number of poses, but no hover
                _ee_pos_r.insert(0, _ee_pos_r[0])
                _wxyz_r.insert(0, _wxyz_r[0])
                _ee_pos_r.append(_ee_pos_r[-1])
                _wxyz_r.append(_wxyz_r[-1])

            ee_pos_l[i, :, :] = _ee_pos_l
            ee_pos_r[i, :, :] = _ee_pos_r
            wxyz_l[i, :, :] = _wxyz_l
            wxyz_r[i, :, :] = _wxyz_r
            joints[i, :, :] = batch_ik(
                target_wxyz=np.array([wxyz_l, wxyz_r]),
                target_position=np.array([ee_pos_l, ee_pos_r]),
            )
            dt[i, :, 0] = dt
            mask[i, :] = 1

        path_batch = PathBatch(
            ee_pos_l=ee_pos_l,
            ee_pos_r=ee_pos_r,
            wxyz_l=wxyz_l,
            wxyz_r=wxyz_r,
            joints=joints,
            dt=dt,
            mask=mask,
        )
        path_batch.save(os.path.join(self.dirpath, PATHS_FILENAME))


def wrap(plan: Plan, mesh) -> Plan:
    """Wrap a 2D plan onto a 3D mesh. """
    pass
