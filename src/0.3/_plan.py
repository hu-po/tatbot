from dataclasses import dataclass, field
import os
import math

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
PIXELPATHS_FILENAME: str = "pixelpaths.yaml"
PATHSTATS_FILENAME: str = "pathstats.yaml"

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
    image_width_px: int | None = None
    """Width of the image in pixels."""
    image_height_px: int | None = None
    """Height of the image in pixels."""

    ik_batch_size: int = 256
    """Batch size for IK computation."""
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

    def load_image_np(self) -> np.ndarray:
        filepath = os.path.join(self.dirpath, IMAGE_FILENAME)
        return np.array(Image.open(filepath).convert("RGB"))
    
    def load_pathbatch(self) -> 'PathBatch':
        filepath = os.path.join(self.dirpath, PATHS_FILENAME)
        return PathBatch.load(filepath)

    def load_pixelpaths(self) -> list[PixelPath]:
        filepath = os.path.join(self.dirpath, PIXELPATHS_FILENAME)
        with open(filepath, "r") as f:
            return [PixelPath.from_dict(p) for p in yaml.safe_load(f)]

    def load_pathstats(self) -> dict:
        filepath = os.path.join(self.dirpath, PATHSTATS_FILENAME)
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    
    def save(self, image: np.ndarray = None):
        log.info(f"⚙️💾 Saving plan to {self.dirpath}")
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

    def add_pixelpaths(self, pixelpaths: list[PixelPath], image: Image):
        num_paths = len(pixelpaths)
        log.info(f"⚙️ Adding {num_paths} pixel paths...")

        log.debug(f"⚙️ Image shape: {image.size}")
        self.image_width_px = image.size[0]
        self.image_height_px = image.size[1]
        scale_x = self.image_width_m / self.image_width_px
        scale_y = self.image_height_m / self.image_height_px

        pixelpaths_path = os.path.join(self.dirpath, PIXELPATHS_FILENAME)
        log.debug(f"⚙️💾 Saving pixelpaths to {pixelpaths_path}...")
        with open(pixelpaths_path, "w") as f:
            yaml.safe_dump([p.to_dict() for p in pixelpaths], f)

        paths = []
        for path_idx, pixelpath in enumerate(pixelpaths):
            log.debug(f"⚙️ Adding pixelpath {path_idx} of {num_paths}...")
            path = Path.padded(self.path_pad_len)
            self.path_descriptions[f'path_{path_idx:03d}'] = pixelpath.description

            pixelpath_length = len(pixelpath.pixels)
            if pixelpath_length + 2 > self.path_pad_len:
                # TODO: resample to fit within pad_len
                log.warning(f"⚙️⚠️ pixelpath {path_idx} has more than {self.path_pad_len} poses, truncating...")
                pixelpath.pixels = pixelpath.pixels[:self.path_pad_len - 2] # -2 for hover positions
                pixelpath_length = len(pixelpath.pixels)

            for i, (pw, ph) in enumerate(pixelpath.pixels):
                # pixel coordinates first need to be converted to meters
                x_m, y_m = pw * scale_x, ph * scale_y
                # center in design frame, add needle offset
                path.ee_pos_l[i + 1, :] = [
                    self.ee_design_pos[0] + x_m - self.image_width_m / 2,
                    self.ee_design_pos[1] + y_m - self.image_height_m / 2,
                    self.ee_design_pos[2] + self.needle_offset[2],
                ]
                path.ee_wxyz_l[i + 1, :] = self.ee_design_wxyz
                # right hand just stares at center of design frame
                path.ee_pos_r[i + 1, :] = [
                    self.ee_design_pos[0] + self.view_offset[0],
                    self.ee_design_pos[1] + self.view_offset[1],
                    self.ee_design_pos[2] + self.view_offset[2],
                ]
                path.ee_wxyz_r[i + 1, :] = self.ee_view_wxyz

            # add hover positions to the beginning and end of the path
            path.ee_pos_l[0, :] = [
                path.ee_pos_l[1, 0] + self.hover_offset[0],
                path.ee_pos_l[1, 1] + self.hover_offset[1],
                path.ee_pos_l[1, 2] + self.hover_offset[2],
            ]
            path.ee_wxyz_l[0, :] = self.ee_design_wxyz
            path.ee_pos_l[pixelpath_length + 1, :] = [
                path.ee_pos_l[pixelpath_length, 0] + self.hover_offset[0],
                path.ee_pos_l[pixelpath_length, 1] + self.hover_offset[1],
                path.ee_pos_l[pixelpath_length, 2] + self.hover_offset[2],
            ]
            path.ee_wxyz_l[pixelpath_length + 1, :] = self.ee_design_wxyz
            # right hand has no hover offset, just use the first and last valid poses
            path.ee_pos_r[0, :] = path.ee_pos_r[1, :]
            path.ee_wxyz_r[0, :] = path.ee_wxyz_r[1, :]
            path.ee_pos_r[pixelpath_length + 1, :] = path.ee_pos_r[pixelpath_length, :]
            path.ee_wxyz_r[pixelpath_length + 1, :] = path.ee_wxyz_r[pixelpath_length, :]
            # slow movement at the hover positions
            path.dt[0] = self.path_dt_slow
            path.dt[1:pixelpath_length + 1] = self.path_dt_fast
            path.dt[pixelpath_length + 1] = self.path_dt_slow
            # set mask: 1 for all valid points (hover + path)
            path.mask[:pixelpath_length + 2] = 1

            paths.append(path)

        # ---- Batch IK ----
        flat_target_pos   : list[list[np.ndarray]] = []
        flat_target_wxyz  : list[list[np.ndarray]] = []
        index_map: list[tuple[int, int]] = [] # (path_idx, pose_idx)
        for p_idx, path in enumerate(paths):
            for pose_idx in range(path.ee_pos_l.shape[0]):
                # Skip padded entries (both arms at zero => unused slot)
                if (np.allclose(path.ee_pos_l[pose_idx], 0.0) and
                    np.allclose(path.ee_pos_r[pose_idx], 0.0)):
                    continue
                index_map.append((p_idx, pose_idx))
                flat_target_pos.append(
                    [path.ee_pos_l[pose_idx], path.ee_pos_r[pose_idx]]
                )
                flat_target_wxyz.append(
                    [path.ee_wxyz_l[pose_idx], path.ee_wxyz_r[pose_idx]]
                )
        target_pos   = jnp.array(flat_target_pos)    # (B, 2, 3)
        target_wxyz  = jnp.array(flat_target_wxyz)   # (B, 2, 4)
        for start in range(0, target_pos.shape[0], self.ik_batch_size):
            end = start + self.ik_batch_size
            batch_pos   = target_pos[start:end]       # (b, 2, 3)
            batch_wxyz  = target_wxyz[start:end]      # (b, 2, 4)
            batch_joints = batch_ik(
                target_wxyz=batch_wxyz,
                target_pos=batch_pos,
            )                                         # (b, 16)
            # write results back into the corresponding path / pose slots
            for local_idx, joints in enumerate(batch_joints):
                p_idx, pose_idx = index_map[start + local_idx]
                paths[p_idx].joints[pose_idx] = np.asarray(joints, dtype=np.float32)

        # overwrites image and metadata
        self.save(image)
        pathbatch = PathBatch.from_paths(paths)
        pathbatch.save(os.path.join(self.dirpath, PATHS_FILENAME)) 

        # compute path stats
        path_lengths_px = [
            sum(np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(path.pixels[:-1], path.pixels[1:]))
            if len(path.pixels) > 1 else 0.0
            for path in pixelpaths
        ]
        # metric lengths
        path_lengths_m = [
            float(np.sum(np.linalg.norm(np.diff(pathbatch.ee_pos_l[i][pathbatch.mask[i] == 1], axis=0), axis=1)))
            if np.sum(pathbatch.mask[i]) > 1 else 0.0
            for i in range(pathbatch.ee_pos_l.shape[0])
        ]
        stats = {
            "count": len(path_lengths_px),
            "min_px": float(np.min(path_lengths_px)) if path_lengths_px else 0.0,
            "max_px": float(np.max(path_lengths_px)) if path_lengths_px else 0.0,
            "mean_px": float(np.mean(path_lengths_px)) if path_lengths_px else 0.0,
            "sum_px": float(np.sum(path_lengths_px)) if path_lengths_px else 0.0,
            "min_m": float(np.min(path_lengths_m)) if path_lengths_m else 0.0,
            "max_m": float(np.max(path_lengths_m)) if path_lengths_m else 0.0,
            "mean_m": float(np.mean(path_lengths_m)) if path_lengths_m else 0.0,
            "sum_m": float(np.sum(path_lengths_m)) if path_lengths_m else 0.0,
        }
        pathstats_path = os.path.join(self.dirpath, PATHSTATS_FILENAME)
        log.debug(f"⚙️💾 Saving pathstats to {pathstats_path}...")
        with open(pathstats_path, "w") as f:
            yaml.safe_dump(stats, f)