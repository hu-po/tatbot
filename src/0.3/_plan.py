from dataclasses import dataclass, field, asdict
import os
import math

import dacite
import numpy as np
import yaml
from PIL import Image
import jax.numpy as jnp

from _ik import batch_ik, fk, transform_and_offset
from _ink import InkCap, InkPalette
from _log import get_logger
from _path import Path, PathBatch, Stroke

log = get_logger('_plan')

# plan objects stored inside folder, these are the filenames
METADATA_FILENAME: str = "meta.yaml"
IMAGE_FILENAME: str = "image.png"
PATHBATCH_FILENAME: str = "pathbatch.safetensors"

@dataclass
class Plan:
    name: str = "plan"
    """Name of the plan."""

    dirpath: str = ""
    """Path to the directory containing the plan files."""

    strokes: dict[str, Stroke] = field(default_factory=dict)
    """Dictionary of path metadata objects."""
    stroke_order: list[str] = field(default_factory=list)
    """Order of strokes in the plan."""
    pathstats: dict = field(default_factory=dict)
    """Path statistics."""

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

    ik_robot_config: IKRobotConfig = field(default_factory=IKRobotConfig)
    """Configuration for the IK robot."""

    ee_design_pos: list[float] = field(default_factory=lambda: [0.08, 0.0, 0.04])
    """position in meters (xyz) of end effector when centered on design."""
    
    ee_design_wxyz_l: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, -0.5])
    """orientation quaternion (wxyz) of left arm end effector when centered on design."""
    ee_design_wxyz_r: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, -0.5])
    """orientation quaternion (wxyz) of right arm end effector when centered on design."""

    hover_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.006])
    """position offset when hovering over point, relative to current ee frame."""
    needle_offset_l: list[float] = field(default_factory=lambda: [0.0, 0.0, -0.0065])
    """position offset to ensure needle touches skin, relative to current ee frame."""
    needle_offset_r: list[float] = field(default_factory=lambda: [0.0, 0.0, -0.0065])
    """position offset to ensure needle touches skin, relative to current ee frame."""

    inkpalette: InkPalette = field(default_factory=InkPalette)
    """Ink palette to use for the plan."""
    ee_inkpalette_pos: list[float] = field(default_factory=lambda: [0.16, 0.0, 0.04])
    """position of the inkpalette ee transform."""
    ee_inkpalette_wxyz: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, -0.5])
    """orientation quaternion (wxyz) of the inkpalette ee transform."""
    inkdip_hover_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.03])
    """position offset when hovering over inkcap, relative to current ee frame."""

    def save(self):
        log.info(f"⚙️💾 Saving plan to {self.dirpath}")
        os.makedirs(self.dirpath, exist_ok=True)
        meta_path = os.path.join(self.dirpath, METADATA_FILENAME)
        log.info(f"⚙️💾 Saving metadata to {meta_path}")
        with open(meta_path, "w") as f:
            yaml.safe_dump(asdict(self), f)

    @classmethod
    def from_yaml(cls, dirpath: str) -> "Plan":
        log.info(f"⚙️ Loading plan from {dirpath}...")
        filepath = os.path.join(dirpath, METADATA_FILENAME)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return dacite.from_dict(cls, data)

    def load_image_np(self) -> np.ndarray:
        filepath = os.path.join(self.dirpath, IMAGE_FILENAME)
        log.debug(f"⚙️💾 Loading plan image from {filepath}")
        return np.array(Image.open(filepath).convert("RGB"))
    
    def save_image_np(self, image: np.ndarray) -> None:
        filepath = os.path.join(self.dirpath, IMAGE_FILENAME)
        log.debug(f"⚙️💾 Saving plan image to {filepath}")
        Image.fromarray(image).save(filepath)
    
    def load_pathbatch(self) -> 'PathBatch':
        filepath = os.path.join(self.dirpath, PATHBATCH_FILENAME)
        return PathBatch.load(filepath)
    
    def save_pathbatch(self, pathbatch: PathBatch) -> None:
        filepath = os.path.join(self.dirpath, PATHBATCH_FILENAME)
        log.debug(f"⚙️💾 Saving pathbatch to {filepath}")
        pathbatch.save(filepath)

    def add_strokes(self, raw_strokes: list[Stroke], image: Image):
        log.debug(f"⚙️ Input image shape: {image.size}")
        self.save_image_np(image)
        self.image_width_px = image.size[0]
        self.image_height_px = image.size[1]
        scale_x = self.image_width_m / self.image_width_px
        scale_y = self.image_height_m / self.image_height_px

        log.info(f"⚙️ Adding {len(raw_strokes)} raw paths to plan...")
        for idx, stroke in enumerate(raw_strokes):
            stroke_length = len(stroke.pixel_coords)
            desired_length = self.path_pad_len - 4 # -4 for rest/hover positions
            if stroke_length != desired_length:
                log.warning(f"⚙️⚠️ stroke {idx} has len {stroke_length}, resampling to {desired_length}...")
                if stroke_length > 1:
                    pixel_coords_np = np.array(stroke.pixel_coords)
                    
                    # Calculate the cumulative distance along the path
                    distances = np.sqrt(np.sum(np.diff(pixel_coords_np, axis=0)**2, axis=1))
                    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
                    
                    # Create a new set of evenly spaced distances
                    new_distances = np.linspace(0, cumulative_distances[-1], desired_length)
                    
                    # Interpolate the x and y coordinates
                    x_coords_resampled = np.interp(new_distances, cumulative_distances, pixel_coords_np[:, 0])
                    y_coords_resampled = np.interp(new_distances, cumulative_distances, pixel_coords_np[:, 1])
                    
                    stroke.pixel_coords = np.stack((x_coords_resampled, y_coords_resampled), axis=-1).tolist()
                elif stroke_length == 1:
                    stroke.pixel_coords = stroke.pixel_coords * desired_length
                else: # stroke_length == 0
                    stroke.pixel_coords = [[0,0]] * desired_length
                stroke_length = desired_length
                
            # add normalized coordinates: top left is 0, 0
            stroke.norm_coords = [
                [pw / self.image_width_px, ph / self.image_height_px]
                for pw, ph in stroke.pixel_coords
            ]
            # calculate center of mass of stroke
            stroke.norm_center = (
                sum(pw for pw, _ in stroke.norm_coords) / stroke_length,
                sum(ph for _, ph in stroke.norm_coords) / stroke_length,
            )
            # calculate meters coordinates: center is 0, 0
            stroke.meter_coords = [
                [pw * scale_x - self.image_width_m / 2, ph * scale_y - self.image_height_m / 2]
                for pw, ph in stroke.pixel_coords
            ]
            # calculate center of mass of stroke
            stroke.meters_center = (
                sum(pw for pw, _ in stroke.meter_coords) / stroke_length,
                sum(ph for _, ph in stroke.meter_coords) / stroke_length,
            )
            self.strokes[f'raw_stroke_{idx:03d}'] = stroke

        self.calculate_pathbatch()
        self.calculate_pathstats()

    def make_inkdip_path(self, inkcap: str) -> tuple[np.ndarray, np.ndarray]:
        assert inkcap in self.inkpalette.inkcaps, f"⚙️❌ Inkcap {inkcap} not found in palette"
        inkcap_pos = self.inkpalette.inkcaps[inkcap].palette_pos
        inkdip_pos = np.tile(np.array(inkcap_pos, dtype=np.float32), (self.path_pad_len, 1))
        inkdip_wxyz = np.tile(np.array(self.ee_inkpalette_wxyz, dtype=np.float32), (self.path_pad_len, 1))
        # hover over the inkcap
        inkcap_hover_pos = transform_and_offset(
            inkcap_pos,
            self.ee_inkpalette_pos,
            self.ee_inkpalette_wxyz,
            self.inkdip_hover_offset,
        )
        inkcap_hover_wxyz = self.ee_inkpalette_wxyz
        # start and end of inkdip are hover positions
        inkdip_pos[0, :] = inkcap_hover_pos
        inkdip_pos[-1, :] = inkcap_hover_pos
        # middle of inkdip is a series of points traveling to depth of inkcap
        inkdip_middle = np.linspace(0, self.inkpalette.inkcaps[inkcap].depth_m, self.path_pad_len - 2)
        inkdip_pos[1:-1, :] = transform_and_offset(
            inkdip_middle,
            self.ee_inkpalette_pos,
            self.ee_inkpalette_wxyz,
        )
        return inkdip_pos, inkdip_wxyz

    def calculate_pathbatch(self) -> None:
        paths: list[Path] = []

        # get rest positions for both arms using forward kinematics
        rest_pos_l, rest_pos_r, rest_wxyz_l, rest_wxyz_r = fk(self.ik_robot_config.rest_pose)

        # sort strokes by width in norm coords
        sorted_strokes = sorted(self.strokes.values(), key=lambda x: x.norm_center[0])
        left_arm_pointer = 0 # left arm starts with leftmost stroke, moves rightwards
        half_length = len(sorted_strokes) // 2
        right_arm_pointer = half_length # right arm starts from middle moves rightwards
        while left_arm_pointer < half_length and right_arm_pointer < len(sorted_strokes):
            
            if sorted_strokes[left_arm_pointer].inkcap is None:
                # get ink color from stroke, left are will dip
                inkcap_name = self.inkpalette.find_best_inkcap(sorted_strokes[left_arm_pointer].color)
                sorted_strokes[left_arm_pointer].inkcap = inkcap_name
                inkdip_pos, inkdip_wxyz = self.make_inkdip_path(inkcap_name)
                path.ee_pos_l[1:stroke_length + 1, :] = inkdip_pos
                path.ee_wxyz_l[:, :] = inkdip_wxyz
                if right_arm_pointer == half_length:
                    # this is the first stroke of the session, keep right arm at rest for this path
                    



            assert stroke_name in self.strokes, f"⚙️❌ Stroke {stroke_name} not found in plan"
            log.debug(f"⚙️ Building path from stroke {stroke_name}...")
            stroke = self.strokes[stroke_name]
            stroke_length = len(stroke.meter_coords)
            path = Path.padded(self.path_pad_len)
            if stroke.arm == "left":
                # transform to design frame, add needle offset
                path.ee_pos_l[1:stroke_length + 1, :] = transform_and_offset(
                    stroke.meter_coords,
                    self.ee_design_pos,
                    self.ee_design_wxyz_l,
                    self.needle_offset_l,
                )
                # add rest and hover positions to beginning and end
                path.ee_pos_l[0, :] = rest_pos_l
                path.ee_pos_l[1, :] = transform_and_offset(
                    stroke.meter_coords[0],
                    self.ee_design_pos,
                    self.ee_design_wxyz_l,
                    self.hover_offset,
                )
                path.ee_pos_l[-2, :] = transform_and_offset(
                    stroke.meter_coords[-1],
                    self.ee_design_pos,
                    self.ee_design_wxyz_l,
                    self.hover_offset,
                )
                path.ee_pos_l[-1, :] = rest_pos_l
                # orientation is always in design frame
                path.ee_wxyz_l[:, :] = self.ee_design_wxyz_l
                # right arm will be ink dipping
                inkdip_pos, inkdip_wxyz = self.make_inkdip_path(stroke.inkcap)
                path.ee_pos_r[1:stroke_length + 1, :] = inkdip_pos
                path.ee_wxyz_r[:, :] = inkdip_wxyz
            else:
                # transform to design frame, add needle offset
                path.ee_pos_r[1:stroke_length + 1, :] = transform_and_offset(
                    stroke.meter_coords,
                    self.ee_design_pos,
                    self.ee_design_wxyz_r,
                    self.needle_offset_r,
                )
                # add hover positions to beginning and end
                path.ee_pos_r[0, :] = transform_and_offset(
                    stroke.meter_coords[0],
                    self.ee_design_pos,
                    self.ee_design_wxyz_r,
                    self.hover_offset,
                )
                path.ee_pos_r[stroke_length + 1, :] = transform_and_offset(
                    stroke.meter_coords[-1],
                    self.ee_design_pos,
                    self.ee_design_wxyz_r,
                    self.hover_offset,
                )
                # orientation is always in design frame
                path.ee_wxyz_r[:, :] = self.ee_design_wxyz_r
                # left arm will be ink dipping
                inkdip_pos, inkdip_wxyz = self.make_inkdip_path(stroke.inkcap)
                path.ee_pos_l[1:stroke_length + 1, :] = inkdip_pos
                path.ee_wxyz_l[:, :] = inkdip_wxyz

            # slow movement at the rest and hover positions
            path.dt[:] = self.path_dt_fast
            path.dt[:2] = self.path_dt_slow
            path.dt[-2:] = self.path_dt_slow
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
                robot_config=self.ik_robot_config,
            )                                         # (b, 16)
            # write results back into the corresponding path / pose slots
            for local_idx, joints in enumerate(batch_joints):
                p_idx, pose_idx = index_map[start + local_idx]
                paths[p_idx].joints[pose_idx] = np.asarray(joints, dtype=np.float32)

        pathbatch = PathBatch.from_paths(paths)
        self.save_pathbatch(pathbatch)

    def calculate_pathstats(self) -> None:
        path_lengths_px = [
            sum(np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(path.pixel_coords[:-1], path.pixel_coords[1:]))
            if len(path.pixel_coords) > 1 else 0.0
            for path in self.strokes
        ]
        # # metric lengths
        # path_lengths_m = [
        #     float(np.sum(np.linalg.norm(np.diff(pathbatch.ee_pos_l[i][pathbatch.mask[i] == 1], axis=0), axis=1)))
        #     if np.sum(pathbatch.mask[i]) > 1 else 0.0
        #     for i in range(pathbatch.ee_pos_l.shape[0])
        # ]
        self.pathstats = {
            "count": len(path_lengths_px),
            "min_px": float(np.min(path_lengths_px)) if path_lengths_px else 0.0,
            "max_px": float(np.max(path_lengths_px)) if path_lengths_px else 0.0,
            "mean_px": float(np.mean(path_lengths_px)) if path_lengths_px else 0.0,
            "sum_px": float(np.sum(path_lengths_px)) if path_lengths_px else 0.0,
            # "min_m": float(np.min(path_lengths_m)) if path_lengths_m else 0.0,
            # "max_m": float(np.max(path_lengths_m)) if path_lengths_m else 0.0,
            # "mean_m": float(np.mean(path_lengths_m)) if path_lengths_m else 0.0,
            # "sum_m": float(np.sum(path_lengths_m)) if path_lengths_m else 0.0,
        }
        self.save()