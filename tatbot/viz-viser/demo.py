import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, NamedTuple

import numpy as np
from PIL import Image
import viser
import viser.transforms as vtf

@dataclass
class VizConfig:
    """Configuration for generating and visualizing pixel targets."""
    image_path: str = "flower.png"
    """Path to the input PNG image."""
    skin_center_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Center of the image on the skin patch in world coordinates (meters)."""
    skin_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Normal vector of the skin surface (pointing outwards from the surface)."""
    skin_width_m: float = 0.09
    """Width of the area on the skin where the image will be projected (meters)."""
    skin_height_m: float = 0.12
    """Height of the area on the skin where the image will be projected (meters)."""
    image_threshold: int = 127
    """[0, 255] threshold for B/W image. Pixels <= threshold are targets."""
    max_draw_pixels: int = 0
    """Maximum number of target pixels to process. If 0 or less, process all."""
    invert_image: bool = False
    """If True, pixels > threshold become targets (e.g., for white lines on black bg)."""
    show_skin_plane: bool = True
    """Whether to visualize the skin plane itself."""
    skin_plane_thickness: float = 0.001
    """Thickness of the visualized skin plane box (meters)."""
    skin_plane_color: Tuple[int, int, int] = (220, 180, 150)
    """RGB color for the skin plane, e.g., a skin-like tone."""
    splat_length: float = 0.000001
    """Length of the splat along its main oriented axis (meters)"""
    splat_thickness: float = 0.000001
    """Thickness of the splat for its other two axes (meters)"""
    splat_color: Tuple[int, int, int] = (0, 0, 0)
    """Color for the splats"""

@dataclass
class PixelTarget:
    position: np.ndarray
    orientation: vtf.SO3

class ProcessedImageData(NamedTuple):
    targets: List[PixelTarget]
    skin_frame_origin: np.ndarray
    skin_frame_T1: np.ndarray
    skin_frame_T2: np.ndarray
    skin_frame_N: np.ndarray

def main(config: VizConfig):
    try:
        img = Image.open(config.image_path)
    except Exception:
        return
    img = img.convert("L")
    arr = np.array(img)
    h_px, w_px = arr.shape
    if config.invert_image:
        target_mask = arr > config.image_threshold
    else:
        target_mask = arr <= config.image_threshold
    target_rows, target_cols = np.where(target_mask)
    if target_rows.size == 0:
        processed_data = ProcessedImageData([], np.array(config.skin_center_position), np.array([1,0,0]), np.array([0,1,0]), np.array(config.skin_normal))
    else:
        norm_u = (target_cols / w_px) - 0.5
        norm_v = 0.5 - (target_rows / h_px)
        target_coords_normalized = list(zip(norm_u.tolist(), norm_v.tolist()))
        if config.max_draw_pixels > 0 and len(target_coords_normalized) > config.max_draw_pixels:
            target_coords_normalized = random.sample(target_coords_normalized, config.max_draw_pixels)
        skin_N_vec = np.array(config.skin_normal, dtype=float)
        if np.linalg.norm(skin_N_vec) < 1e-9:
            skin_N_vec = np.array([0.0, 0.0, 1.0])
        else:
            skin_N_vec /= np.linalg.norm(skin_N_vec)
        world_Z = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(skin_N_vec, world_Z)) > 0.999:
            skin_T1_vec = np.cross(np.array([0.0, 1.0, 0.0]), skin_N_vec)
            if np.linalg.norm(skin_T1_vec) < 1e-5:
                skin_T1_vec = np.cross(np.array([1.0, 0.0, 0.0]), skin_N_vec)
        else:
            skin_T1_vec = np.cross(world_Z, skin_N_vec)
        skin_T1_vec /= np.linalg.norm(skin_T1_vec)
        skin_T2_vec = np.cross(skin_N_vec, skin_T1_vec)
        skin_T2_vec /= np.linalg.norm(skin_T2_vec)
        pixel_targets_list: List[PixelTarget] = []
        skin_origin_pos = np.array(config.skin_center_position)
        for u, v in target_coords_normalized:
            pos_offset = u * config.skin_width_m * skin_T1_vec + v * config.skin_height_m * skin_T2_vec
            pixel_world_pos = skin_origin_pos + pos_offset
            z_axis_tool = skin_N_vec
            x_axis_tool = skin_T1_vec
            y_axis_tool = np.cross(z_axis_tool, x_axis_tool)
            y_axis_tool /= np.linalg.norm(y_axis_tool)
            rotation_matrix = np.stack([x_axis_tool, y_axis_tool, z_axis_tool], axis=1)
            so3_orientation = vtf.SO3.from_matrix(rotation_matrix)
            pixel_targets_list.append(PixelTarget(position=pixel_world_pos, orientation=so3_orientation))
        processed_data = ProcessedImageData(
            targets=pixel_targets_list,
            skin_frame_origin=skin_origin_pos,
            skin_frame_T1=skin_T1_vec,
            skin_frame_T2=skin_T2_vec,
            skin_frame_N=skin_N_vec
        )
    server = viser.ViserServer()
    targets = processed_data.targets
    skin_origin = processed_data.skin_frame_origin
    T1, T2, N = processed_data.skin_frame_T1, processed_data.skin_frame_T2, processed_data.skin_frame_N
    with server.atomic():
        if config.show_skin_plane:
            skin_rot_matrix = np.stack([T1, T2, N], axis=1)
            skin_so3 = vtf.SO3.from_matrix(skin_rot_matrix)
            server.scene.add_box(
                name="/skin_patch",
                wxyz=skin_so3.wxyz,
                position=skin_origin,
                dimensions=(config.skin_width_m, config.skin_height_m, config.skin_plane_thickness),
                color=config.skin_plane_color
            )
            server.scene.add_frame(
                name="/skin_coordinate_frame",
                position=skin_origin,
                wxyz=skin_so3.wxyz,
                axes_length=min(config.skin_width_m, config.skin_height_m) * 0.6,
                axes_radius=config.skin_plane_thickness * 2.5
            )
    if targets:
        num_targets = len(targets)
        positions_np = np.array([target.position for target in targets])
        orientations_np = np.array([target.orientation.wxyz for target in targets])
        colors_np = np.full((num_targets, 3), config.splat_color, dtype=np.uint8)
        scales_np = np.full((num_targets, 3), (config.splat_thickness, config.splat_thickness, config.splat_length), dtype=np.float32)
        covariances_np = np.array([
            target.orientation.as_matrix() @ np.diag([config.splat_thickness, config.splat_thickness, config.splat_length]) @ target.orientation.as_matrix().T
            for target in targets
        ], dtype=np.float32)
        opacities_np = np.full((num_targets, 1), 1.0, dtype=np.float32)
        with server.atomic():
            server.scene.add_gaussian_splats(
                name="/pixel_targets/oriented_gaussians",
                centers=positions_np,
                covariances=covariances_np,
                rgbs=colors_np,
                opacities=opacities_np
            )
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main(VizConfig())