import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, NamedTuple

import numpy as np
from PIL import Image, ImageOps
import viser
import viser.transforms as vtf

import logging

# Basic logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DesignConfig:
    """Configuration for generating and visualizing pixel targets."""
    image_path: str = "circle.png"
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
    pixel_marker_size: float = 0.0005
    """Radius of the spheres used to visualize pixel target positions (meters)."""

    # Viser related settings
    show_skin_plane: bool = True
    """Whether to visualize the skin plane itself."""
    skin_plane_thickness: float = 0.001 # 0.04
    """Thickness of the visualized skin plane box (meters)."""
    skin_plane_color: Tuple[int, int, int] = (220, 180, 150)
    """RGB color for the skin plane, e.g., a skin-like tone."""
    animation_delay_s: float = 0.005
    """Delay in seconds between adding each pixel to the Viser visualization."""

@dataclass
class PixelTarget:
    """Represents a 6D pose for a single pixel target."""
    position: np.ndarray  # (3,) array for x, y, z in world coordinates
    orientation: vtf.SO3  # SO3 object for orientation in world coordinates

class ProcessedImageData(NamedTuple):
    """Holds the generated targets and skin frame information."""
    targets: List[PixelTarget]
    skin_frame_origin: np.ndarray
    skin_frame_T1: np.ndarray  # X-axis on skin plane (image width direction)
    skin_frame_T2: np.ndarray  # Y-axis on skin plane (image height direction)
    skin_frame_N: np.ndarray   # Z-axis (normal) of skin plane

def generate_pixel_targets(config: DesignConfig) -> ProcessedImageData:
    """
    Loads an image, processes it, and calculates 6D poses for target pixels
    on a defined skin plane.

    Returns:
        ProcessedImageData containing the list of PixelTarget objects and
        the skin's coordinate frame vectors.
    """
    logger.info(f"🖼️ Loading image from: {config.image_path}")
    try:
        img = Image.open(config.image_path)
    except FileNotFoundError:
        logger.error(f"Image file not found: {config.image_path}")
        return ProcessedImageData([], np.array(config.skin_center_position), np.array([1,0,0]), np.array([0,1,0]), np.array(config.skin_normal))
    except Exception as e:
        logger.error(f"Error opening image {config.image_path}: {e}")
        return ProcessedImageData([], np.array(config.skin_center_position), np.array([1,0,0]), np.array([0,1,0]), np.array(config.skin_normal))


    img = img.convert("L")  # Convert to grayscale

    arr = np.array(img)
    h_px, w_px = arr.shape

    # Determine target mask based on threshold and inversion setting
    if config.invert_image:
        target_mask = arr > config.image_threshold # Target white/lighter pixels
        logger.info(f"Image inverted. Targeting pixels with intensity > {config.image_threshold}.")
    else:
        target_mask = arr <= config.image_threshold # Target black/darker pixels
        logger.info(f"Targeting pixels with intensity <= {config.image_threshold}.")

    target_rows, target_cols = np.where(target_mask)

    if target_rows.size == 0:
        logger.info("No target pixels found after thresholding.")
        return ProcessedImageData([], np.array(config.skin_center_position), np.array([1,0,0]), np.array([0,1,0]), np.array(config.skin_normal))

    # Normalize pixel coordinates: u,v range from [-0.5, 0.5] with origin at image center.
    # Image (0,0) is top-left. u corresponds to width (cols), v to height (rows).
    # Standard image Y is downwards, standard 3D Y is upwards.
    norm_u = (target_cols / w_px) - 0.5  # Image X, normalized: -0.5 (left) to 0.5 (right)
    norm_v = 0.5 - (target_rows / h_px)  # Image Y, normalized & inverted: -0.5 (bottom) to 0.5 (top)

    target_coords_normalized = list(zip(norm_u.tolist(), norm_v.tolist()))

    # Randomly sample if max_draw_pixels is set
    if config.max_draw_pixels > 0 and len(target_coords_normalized) > config.max_draw_pixels:
        logger.info(f"Sampling {config.max_draw_pixels} pixels from {len(target_coords_normalized)}.")
        target_coords_normalized = random.sample(target_coords_normalized, config.max_draw_pixels)

    num_pixels_to_process = len(target_coords_normalized)
    logger.info(f"✨ Processing {num_pixels_to_process} target pixels.")

    # --- Define Skin Plane Coordinate System (T1, T2, N) ---
    skin_N_vec = np.array(config.skin_normal, dtype=float)
    if np.linalg.norm(skin_N_vec) < 1e-9: # Avoid division by zero if normal is zero vector
        logger.warning("Skin normal vector has zero length. Defaulting to (0,0,1).")
        skin_N_vec = np.array([0.0, 0.0, 1.0])
    else:
        skin_N_vec /= np.linalg.norm(skin_N_vec)

    # Create tangent vectors T1 ("image right" on skin) and T2 ("image up" on skin)
    # T1 is chosen to be as horizontal as possible in the world frame (minimal Z component).
    world_Z = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(skin_N_vec, world_Z)) > 0.999:  # skin_N is (anti-)aligned with world_Z
        # Skin plane is horizontal. Image X-axis (T1) can align with world X-axis.
        skin_T1_vec = np.cross(np.array([0.0, 1.0, 0.0]), skin_N_vec) # Cross with world Y
        if np.linalg.norm(skin_T1_vec) < 1e-5 : # if N was along Y, cross with world X
             skin_T1_vec = np.cross(np.array([1.0, 0.0, 0.0]), skin_N_vec)
    else:  # Normal is not vertical, so skin_N can be crossed with world_Z.
        skin_T1_vec = np.cross(world_Z, skin_N_vec) # T1 will be horizontal (world_Z x N)

    skin_T1_vec /= np.linalg.norm(skin_T1_vec)
    skin_T2_vec = np.cross(skin_N_vec, skin_T1_vec) # T2 = N x T1 to form right-handed (T1, T2, N)
    skin_T2_vec /= np.linalg.norm(skin_T2_vec)

    pixel_targets_list: List[PixelTarget] = []
    skin_origin_pos = np.array(config.skin_center_position)

    for u, v in target_coords_normalized:
        # Map normalized image coords (u,v) to 3D positions on the skin plane
        # u scales with skin_width_m along skin_T1_vec
        # v scales with skin_height_m along skin_T2_vec
        pos_offset = u * config.skin_width_m * skin_T1_vec + \
                     v * config.skin_height_m * skin_T2_vec
        pixel_world_pos = skin_origin_pos + pos_offset

        # Tool Z-axis points away from the surface (along the normal)
        z_axis_tool = skin_N_vec

        # Tool's X-axis can align with skin_T1_vec (image's "right" direction on skin)
        x_axis_tool = skin_T1_vec
        # Tool's Y-axis from cross product to maintain right-handed system
        y_axis_tool = np.cross(z_axis_tool, x_axis_tool)
        y_axis_tool /= np.linalg.norm(y_axis_tool)

        # Create rotation matrix (columns are the tool's axes in world frame)
        rotation_matrix = np.stack([x_axis_tool, y_axis_tool, z_axis_tool], axis=1)
        so3_orientation = vtf.SO3.from_matrix(rotation_matrix)

        pixel_targets_list.append(PixelTarget(position=pixel_world_pos, orientation=so3_orientation))

    logger.info(f"Generated {len(pixel_targets_list)} pixel targets.")
    return ProcessedImageData(
        targets=pixel_targets_list,
        skin_frame_origin=skin_origin_pos,
        skin_frame_T1=skin_T1_vec,
        skin_frame_T2=skin_T2_vec,
        skin_frame_N=skin_N_vec
    )

def visualize_design(config: DesignConfig, processed_data: ProcessedImageData):
    """
    Visualizes the skin plane and the pixel targets sequentially using Viser.
    """
    server = viser.ViserServer() # No change here
    logger.info("Viser server started. Check your browser if it doesn't open automatically.")

    targets = processed_data.targets
    skin_origin = processed_data.skin_frame_origin
    T1, T2, N = processed_data.skin_frame_T1, processed_data.skin_frame_T2, processed_data.skin_frame_N

    # --- Add a representation for the skin patch ---
    if config.show_skin_plane:
        # Rotation matrix for the skin plane: columns are T1, T2, N
        skin_rot_matrix = np.stack([T1, T2, N], axis=1)
        skin_so3 = vtf.SO3.from_matrix(skin_rot_matrix)
        skin_quat_wxyz = skin_so3.wxyz

        # CORRECTED LINE:
        server.scene.add_box(
            name="/skin_patch",
            wxyz=skin_quat_wxyz,
            position=skin_origin,
            dimensions=(config.skin_width_m, config.skin_height_m, config.skin_plane_thickness),
            color=config.skin_plane_color
        )
        logger.info(f"Added skin patch visual at {skin_origin} with normal {config.skin_normal}")

        # Add a coordinate frame visually representing the skin's local axes
        # CORRECTED LINE:
        server.scene.add_frame(
            name="/skin_coordinate_frame",
            position=skin_origin,
            wxyz=skin_quat_wxyz,
            axes_length=min(config.skin_width_m, config.skin_height_m) * 0.6,
            axes_radius=config.skin_plane_thickness * 2.5
        )

    # --- Visualize pixel targets one by one (animated) ---
    if not targets:
        logger.info("No targets to visualize.")
    else:
        logger.info(f"Visualizing {len(targets)} pixel targets. Animation delay: {config.animation_delay_s}s")

    for i, target in enumerate(targets):
        target_name_prefix = f"/pixel_targets/pixel_{i:04d}"

        # Add a small sphere for the pixel's 3D position
        # CORRECTED LINE:
        server.scene.add_icosphere(
            name=f"{target_name_prefix}/position_marker",
            radius=config.pixel_marker_size,
            color=(30, 30, 220),
            position=target.position
        )
        # Add a coordinate frame for the pixel's 6D pose (tool orientation)
        # CORRECTED LINE:
        server.scene.add_frame(
            name=f"{target_name_prefix}/pose_frame",
            wxyz=target.orientation.wxyz,
            position=target.position,
            axes_length=config.pixel_marker_size * 15,
            axes_radius=config.pixel_marker_size * 0.3
        )

        if (i + 1) % 100 == 0:
            logger.info(f"Visualized {i + 1}/{len(targets)} pixels...")

        if config.animation_delay_s > 0:
            time.sleep(config.animation_delay_s)

    logger.info("All pixel targets visualized. Viser server is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Viser server shutting down.")

if __name__ == "__main__":
    design_config = DesignConfig()

    # --- Generate Pixel Targets ---
    # Skin normal in config is normalized inside generate_pixel_targets
    processed_image_info = generate_pixel_targets(design_config)

    if not processed_image_info.targets and design_config.image_path == "MISSING_IMAGE.png":
        logger.error("Cannot proceed without a valid image. Please set config.image_path.")
    elif not processed_image_info.targets:
        logger.warning("No pixel targets were generated. Check image and threshold settings. Visualization might be empty.")
        # Still run Viser to show skin plane if enabled
        try:
             visualize_design(design_config, processed_image_info)
        except Exception as e:
            logger.error(f"Error during visualization setup: {e}", exc_info=True)
    else:
        # --- Visualize Design ---
        try:
            visualize_design(design_config, processed_image_info)
        except Exception as e:
            logger.error(f"Unhandled error during visualization: {e}", exc_info=True)
            logger.error("Please ensure Viser is installed and dependencies are met ('pip install viser').")