from dataclasses import dataclass, field, asdict
import os
import math

import dacite
import numpy as np
import yaml
from PIL import Image
import jax.numpy as jnp

from _bot import BotConfig
from _ik import batch_ik, transform_and_offset
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
    path_idx_to_strokes: list[list[Stroke]] = field(default_factory=list)
    """Map from pathbatch idx to list of strokes that make up that path."""

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
    path_length: int = 64
    """All paths will be resampled to this length."""
    path_dt_fast: float = 0.1
    """Time between poses in seconds for fast movement."""
    path_dt_slow: float = 2.0
    """Time between poses in seconds for slow movement."""

    design_pos: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.0, -0.04], dtype=np.float32))
    """position in meters (xyz) of origin of design frame."""
    design_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    """orientation quaternion (wxyz) of the design frame."""

    # TODO: these will have to be updated to be relative to the design frame
    ee_design_wxyz_l: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, -0.5], dtype=np.float32))
    """orientation quaternion (wxyz) of left arm end effector when performing a path."""
    ee_design_wxyz_r: np.ndarray = field(default_factory=lambda: np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float32))
    """orientation quaternion (wxyz) of right arm end effector when performing a path."""

    hover_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.006], dtype=np.float32))
    """position offset when hovering over point, relative to current ee frame."""
    needle_offset_l: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.0065], dtype=np.float32))
    """position offset to ensure needle touches skin, relative to current ee frame."""
    needle_offset_r: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.0065], dtype=np.float32))
    """position offset to ensure needle touches skin, relative to current ee frame."""

    inkpalette: InkPalette = field(default_factory=InkPalette)
    """Ink palette to use for the plan."""
    inkpalette_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.04], dtype=np.float32))
    """position of the inkpalette ee transform."""
    inkpalette_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    """orientation quaternion (wxyz) of the inkpalette ee transform."""
    inkdip_hover_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.03], dtype=np.float32))
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
        return dacite.from_dict(cls, data, config=dacite.Config(type_hooks={np.ndarray: np.array}))

    def load_image_np(self) -> np.ndarray:
        filepath = os.path.join(self.dirpath, IMAGE_FILENAME)
        log.debug(f"⚙️💾 Loading plan image from {filepath}")
        return np.array(Image.open(filepath).convert("RGB"))
    
    def save_image_np(self, image: np.ndarray | Image.Image) -> None:
        filepath = os.path.join(self.dirpath, IMAGE_FILENAME)
        log.debug(f"⚙️💾 Saving plan image to {filepath}")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(filepath)
    
    def load_pathbatch(self) -> 'PathBatch':
        filepath = os.path.join(self.dirpath, PATHBATCH_FILENAME)
        return PathBatch.load(filepath)
    
    def save_pathbatch(self, pathbatch: PathBatch) -> None:
        filepath = os.path.join(self.dirpath, PATHBATCH_FILENAME)
        log.debug(f"⚙️💾 Saving pathbatch to {filepath}")
        pathbatch.save(filepath)

    def add_strokes(self, strokes: list[Stroke], image: Image):
        log.debug(f"⚙️ Input image shape: {image.size}")
        self.save_image_np(image)
        self.image_width_px = image.size[0]
        self.image_height_px = image.size[1]
        scale_x = self.image_width_m / self.image_width_px
        scale_y = self.image_height_m / self.image_height_px

        log.info(f"⚙️ Adding {len(strokes)} raw paths to plan...")
        for idx, stroke in enumerate(strokes):
            if stroke.pixel_coords is not None:
                stroke.pixel_coords = np.array(stroke.pixel_coords, dtype=int)
            stroke_length = len(stroke.pixel_coords)
            desired_length = self.path_length - 2 # -2 for hover positions
            if stroke_length != desired_length:
                log.warning(f"⚙️⚠️ stroke {idx} has len {stroke_length}, resampling to {desired_length}...")
                if stroke_length > 1:
                    # Calculate the cumulative distance along the path
                    distances = np.sqrt(np.sum(np.diff(stroke.pixel_coords, axis=0)**2, axis=1))
                    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
                    
                    # Create a new set of evenly spaced distances
                    new_distances = np.linspace(0, cumulative_distances[-1], desired_length)
                    
                    # Interpolate the x and y coordinates
                    x_coords_resampled = np.interp(new_distances, cumulative_distances, stroke.pixel_coords[:, 0])
                    y_coords_resampled = np.interp(new_distances, cumulative_distances, stroke.pixel_coords[:, 1])
                    
                    stroke.pixel_coords = np.round(np.stack((x_coords_resampled, y_coords_resampled), axis=-1)).astype(int)
                elif stroke_length == 1:
                    stroke.pixel_coords = np.tile(stroke.pixel_coords, (desired_length, 1))
                else: # stroke_length == 0
                    stroke.pixel_coords = np.zeros((desired_length, 2), dtype=int)
                stroke_length = desired_length
                
            # add normalized coordinates: top left is 0, 0
            stroke.norm_coords = stroke.pixel_coords / np.array([self.image_width_px, self.image_height_px], dtype=np.float32)
            # calculate center of mass of stroke
            stroke.norm_center = np.mean(stroke.norm_coords, axis=0)
            # calculate meters coordinates: center is 0, 0
            meter_coords_2d = (
                stroke.pixel_coords * np.array([scale_x, scale_y], dtype=np.float32)
                - np.array([self.image_width_m / 2, self.image_height_m / 2], dtype=np.float32)
            )
            stroke.meter_coords = np.hstack([
                meter_coords_2d,
                np.zeros((stroke_length, 1), dtype=np.float32)
            ])
            # calculate center of mass of stroke
            stroke.meters_center = np.mean(stroke.meter_coords, axis=0)
            self.strokes[f'stroke_{idx:03d}'] = stroke

        self.save() # update metadata
        self.calculate_pathbatch()

    def make_inkdip_pos(self, inkcap_name: str) -> np.ndarray:
        assert inkcap_name in self.inkpalette.inkcaps, f"⚙️❌ Inkcap {inkcap_name} not found in palette"
        inkcap: InkCap = self.inkpalette.inkcaps[inkcap_name]
        inkcap_pos = np.array(inkcap.palette_pos, dtype=np.float32)
        # initialize the inkdip path to the inkcap position
        inkdip_pos = np.tile(inkcap_pos, (self.path_length, 1))
        # hover over the inkcap
        inkcap_hover_pos = transform_and_offset(
            np.expand_dims(inkcap_pos, axis=0),
            self.inkpalette_pos,
            self.inkpalette_wxyz,
            self.inkdip_hover_offset,
        )
        # set start and end to hover position
        inkdip_pos[0, :] = inkcap_hover_pos
        inkdip_pos[-1, :] = inkcap_hover_pos
        # middle of inkdip is a series of points: down, wait, up
        num_dip_points = self.path_length - 2
        # Split: 1/3 down, 1/3 wait, 1/3 up (adjust as needed)
        num_down = num_dip_points // 3
        num_wait = num_dip_points // 3
        num_up = num_dip_points - num_down - num_wait
        # Down: from 0 to depth
        down_z = np.linspace(0, inkcap.depth_m, num_down, endpoint=False)
        # Wait: at depth
        wait_z = np.full(num_wait, inkcap.depth_m)
        # Up: from depth to 0
        up_z = np.linspace(inkcap.depth_m, 0, num_up, endpoint=True)
        # Concatenate all z
        z_offsets = -np.concatenate([down_z, wait_z, up_z]).reshape(-1, 1)
        xy_offsets = np.zeros((num_dip_points, 2))
        offsets = np.concatenate([xy_offsets, z_offsets], axis=1)
        inkdip_pos[1:-1, :] = transform_and_offset(
            np.tile(inkcap_pos, (num_dip_points, 1)),
            self.inkpalette_pos,
            self.inkpalette_wxyz,
            offsets,
        )
        return inkdip_pos

    def calculate_pathbatch(self, filter_completed: bool = False) -> None:
        # paths will be accumulated then concatenated into a pathbatch object
        paths: list[Path] = []

        if filter_completed:
            # if pathbatch is re-calculated halfway through the session
            log.debug(f"⚙️ Filtering out completed strokes...")
            _strokes = {k: v for k, v in self.strokes.items() if v.is_completed}
        else:
            _strokes = self.strokes

        # sort strokes along the X axis (width) in normalized coords
        sorted_strokes = sorted(_strokes.items(), key=lambda x: x[1].norm_center[0])
        left_arm_pointer = 0 # left arm starts with leftmost stroke, moves rightwards
        half_length = len(sorted_strokes) // 2
        right_arm_pointer = half_length # right arm starts from middle moves rightwards
        while left_arm_pointer < half_length and right_arm_pointer < len(sorted_strokes):
            stroke_name_l, stroke_l = sorted_strokes[left_arm_pointer]
            assert stroke_name_l in self.strokes, f"⚙️❌ Stroke {stroke_name_l} not found in plan"
            stroke_name_r, stroke_r = sorted_strokes[right_arm_pointer]
            assert stroke_name_r in self.strokes, f"⚙️❌ Stroke {stroke_name_r} not found in plan"
            log.debug(f"⚙️ Building path from strokes left: {stroke_name_l} and right: {stroke_name_r}...")
            
            # create a new empty path for this stroke
            path = Path.empty(self.path_length)
            # set the default time between poses to fast movement
            path.dt[:] = self.path_dt_fast
            # slow movement to and from hover positions
            path.dt[0] = self.path_dt_slow
            path.dt[-1] = self.path_dt_slow
            # TODO: for now orientation is just design orientation (for inkdips as well)
            path.ee_wxyz_l[:, :] = np.tile(self.ee_design_wxyz_l, (self.path_length, 1))
            path.ee_wxyz_r[:, :] = np.tile(self.ee_design_wxyz_r, (self.path_length, 1))

            # left arm pointer hits a stroke with no inkcap
            if self.strokes[stroke_name_l].inkcap is None:
                # get ink color from stroke, left arm will dip for this path
                inkcap_name = self.inkpalette.find_best_inkcap(self.strokes[stroke_name_l].color)
                self.strokes[stroke_name_l].inkcap = inkcap_name
                path.ee_pos_l = self.make_inkdip_pos(inkcap_name)
                # make a new stroke object for the inkdip path
                stroke_l = Stroke(
                    description=f"left arm inkdip into {inkcap_name}",
                    arm="left",
                    is_inkdip=True,
                    inkcap=inkcap_name,
                )
                if left_arm_pointer == 0:
                    # this is the first stroke of the session, keep right arm at rest for this path
                    stroke_r = Stroke(
                        description="right arm rest",
                        arm="right",
                    )
                    self.path_idx_to_strokes.append([stroke_l, stroke_r])
                    paths.append(path)
                    continue
            else:
                # left arm pointer hits a stroke with an inkcap
                # transform to design frame, add needle offset
                path.ee_pos_l[1:-1, :] = transform_and_offset(
                    self.strokes[stroke_name_l].meter_coords,
                    self.design_pos,
                    self.design_wxyz,
                    self.needle_offset_l,
                )
                # add hover positions to start and end
                path.ee_pos_l[0, :] = transform_and_offset(
                    np.expand_dims(self.strokes[stroke_name_l].meter_coords[0], axis=0),
                    self.design_pos,
                    self.design_wxyz,
                    self.hover_offset,
                )
                path.ee_pos_l[-1, :] = transform_and_offset(
                    np.expand_dims(self.strokes[stroke_name_l].meter_coords[-1], axis=0),
                    self.design_pos,
                    self.design_wxyz,
                    self.hover_offset,
                )
                left_arm_pointer += 1

            if self.strokes[stroke_name_r].inkcap is None:
                inkcap_name = self.inkpalette.find_best_inkcap(self.strokes[stroke_name_r].color)
                self.strokes[stroke_name_r].inkcap = inkcap_name
                path.ee_pos_r = self.make_inkdip_pos(inkcap_name)
                # make a new stroke object for the inkdip path
                stroke_r = Stroke(
                    description=f"right arm inkdip into {inkcap_name}",
                    arm="right",
                    is_inkdip=True,
                    inkcap=inkcap_name,
                )
            else:
                # right arm pointer hits a stroke with an inkcap
                # transform to design frame, add needle offset
                path.ee_pos_r[1:-1, :] = transform_and_offset(
                    self.strokes[stroke_name_r].meter_coords,
                    self.design_pos,
                    self.design_wxyz,
                    self.needle_offset_r,
                )
                # add hover positions to start and end
                path.ee_pos_r[0, :] = transform_and_offset(
                    np.expand_dims(self.strokes[stroke_name_r].meter_coords[0], axis=0),
                    self.design_pos,
                    self.design_wxyz,
                    self.hover_offset,
                )
                path.ee_pos_r[-1, :] = transform_and_offset(
                    np.expand_dims(self.strokes[stroke_name_r].meter_coords[-1], axis=0),
                    self.design_pos,
                    self.design_wxyz,
                    self.hover_offset,
                )
                right_arm_pointer += 1

            self.path_idx_to_strokes.append([stroke_l, stroke_r])
            paths.append(path)

        # Perform IK in batches (batch size will be hardware specific)
        flat_target_pos   : list[list[np.ndarray]] = []
        flat_target_wxyz  : list[list[np.ndarray]] = []
        index_map: list[tuple[int, int]] = [] # (path_idx, pose_idx)
        for p_idx, path in enumerate(paths):
            for pose_idx in range(path.ee_pos_l.shape[0]):
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

        # HACK: the right arm of the very first path should be at rest while left arm is ink dipping
        paths[0].joints[:, 8:] = np.tile(BotConfig().rest_pose[8:], (self.path_length, 1))

        pathbatch = PathBatch.from_paths(paths)
        self.save_pathbatch(pathbatch)
        self.save() # update metadata
