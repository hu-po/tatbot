"""
https://viser.studio/main/
"""

import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, NamedTuple, Dict, Any

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

@dataclass
class State:
    visuals: Dict[str, Any]
    processed_data: ProcessedImageData
    skin_control: Any = None

def process_skin_and_targets(config: VizConfig, target_rows, target_cols, skin_center_position, T1, T2, N):
    if target_rows.size == 0:
        return ProcessedImageData([], np.array(skin_center_position), np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]))
    h_px, w_px = target_rows.max() + 1, target_cols.max() + 1
    norm_u = (target_cols / w_px) - 0.5
    norm_v = 0.5 - (target_rows / h_px)
    target_coords_normalized = list(zip(norm_u.tolist(), norm_v.tolist()))
    if config.max_draw_pixels > 0 and len(target_coords_normalized) > config.max_draw_pixels:
        target_coords_normalized = random.sample(target_coords_normalized, config.max_draw_pixels)
    pixel_targets_list: List[PixelTarget] = []
    skin_origin_pos = np.array(skin_center_position)
    for u, v in target_coords_normalized:
        pos_offset = u * config.skin_width_m * T1 + v * config.skin_height_m * T2
        pixel_world_pos = skin_origin_pos + pos_offset
        rotation_matrix = np.stack([T1, T2, N], axis=1)
        so3_orientation = vtf.SO3.from_matrix(rotation_matrix)
        pixel_targets_list.append(PixelTarget(position=pixel_world_pos, orientation=so3_orientation))
    return ProcessedImageData(
        targets=pixel_targets_list,
        skin_frame_origin=skin_origin_pos,
        skin_frame_T1=T1,
        skin_frame_T2=T2,
        skin_frame_N=N
    )

def update_scene(server, config, state):
    for handle in state.visuals.values():
        try:
            handle.remove()
        except Exception:
            pass
    state.visuals.clear()
    processed_data = state.processed_data
    targets = processed_data.targets
    skin_origin = processed_data.skin_frame_origin
    T1, T2, N = processed_data.skin_frame_T1, processed_data.skin_frame_T2, processed_data.skin_frame_N
    skin_rot_matrix = np.stack([T1, T2, N], axis=1)
    skin_so3 = vtf.SO3.from_matrix(skin_rot_matrix)
    with server.atomic():
        if config.show_skin_plane:
            state.visuals['skin_box'] = server.scene.add_box(
                name="/skin_patch",
                wxyz=skin_so3.wxyz,
                position=skin_origin,
                dimensions=(config.skin_width_m, config.skin_height_m, config.skin_plane_thickness),
                color=config.skin_plane_color
            )
            state.visuals['skin_frame'] = server.scene.add_frame(
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
            state.visuals['splats'] = server.scene.add_gaussian_splats(
                name="/pixel_targets/oriented_gaussians",
                centers=positions_np,
                covariances=covariances_np,
                rgbs=colors_np,
                opacities=opacities_np
            )
    return skin_origin, skin_so3

def update_from_gizmo(server, config, state):
    config.skin_center_position = tuple(state.skin_control.position)
    rot_matrix = vtf.SO3(state.skin_control.wxyz).as_matrix()
    T1 = rot_matrix[:, 0]
    T2 = rot_matrix[:, 1]
    N = rot_matrix[:, 2]
    config.skin_normal = tuple(N.tolist())
    state.processed_data = process_skin_and_targets(
        config,
        state.target_rows,
        state.target_cols,
        config.skin_center_position,
        T1, T2, N
    )
    update_scene(server, config, state)

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
    # Initial axes from config
    N0 = np.array(config.skin_normal, dtype=float)
    if np.linalg.norm(N0) < 1e-9:
        N0 = np.array([0.0, 0.0, 1.0])
    else:
        N0 /= np.linalg.norm(N0)
    world_Z = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(N0, world_Z)) > 0.999:
        T1_0 = np.cross(np.array([0.0, 1.0, 0.0]), N0)
        if np.linalg.norm(T1_0) < 1e-5:
            T1_0 = np.cross(np.array([1.0, 0.0, 0.0]), N0)
    else:
        T1_0 = np.cross(world_Z, N0)
    T1_0 /= np.linalg.norm(T1_0)
    T2_0 = np.cross(N0, T1_0)
    T2_0 /= np.linalg.norm(T2_0)
    processed_data = process_skin_and_targets(config, target_rows, target_cols, config.skin_center_position, T1_0, T2_0, N0)
    server = viser.ViserServer()
    state = State(visuals={}, processed_data=processed_data)
    state.target_rows = target_rows
    state.target_cols = target_cols
    skin_origin, skin_so3 = update_scene(server, config, state)
    state.skin_control = server.scene.add_transform_controls(
        name="/skin_patch_control",
        position=skin_origin,
        wxyz=skin_so3.wxyz,
        scale=0.15,
    )
    state.skin_control.on_update(lambda _: update_from_gizmo(server, config, state))
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main(VizConfig())