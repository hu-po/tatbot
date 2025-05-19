import csv
from dataclasses import dataclass
import logging
import os
import time

import numpy as np
from PIL import Image
import trossen_arm

@dataclass
class DrawImageConfig:
    arm_model: trossen_arm.Model = trossen_arm.Model.wxai_v0
    # ip_address: str = "192.168.1.2"
    # end_effector_model: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_leader
    ip_address: str = "192.168.1.3"
    end_effector_model: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_follower
    
    # pre-defined joint positions
    joint_pos_sleep: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """Sleep pose: robot is folded up, motors can be released."""
    joint_pos_home: tuple[float, ...] = (0.0, 1.05, 0.5, -1.06, 0.0, 0.0, 0.0)
    """Home pose: robot is active, staring down at workspace"""
    cart_pos_home: tuple[float, ...] = (0.12, 0.0, 0.005)
    """Cartesian position of home pose in the robot's base frame."""

    # workspace definitions
    workspace_home_offset: tuple[float, ...] = (-0.01, 0.33, 0.1)
    """Cartesian offset of workspace frame from robot base frame."""
    skin_lower_left_corner: tuple[float, ...] = (0.1, 0.3, 0.04)
    """Lower left corner of skin in workspace frame."""
    skin_upper_right_corner: tuple[float, ...] = (0.22, 0.39, 0.04)
    """Upper right corner of skin in workspace frame."""
    
    # pen gripping parameters
    pen_holder_cart_pos_ready: tuple[float, ...] = (0.15, 0.25, 0.16)
    """Cartesian position of pen holder in workspace frame when ready to grasp pen."""
    pen_holder_cart_pos_grasp: tuple[float, ...] = (0.15, 0.25, 0.1)
    """Cartesian position of pen holder in workspace frame when grasping pen."""
    pen_holder_cart_pos_drop: tuple[float, ...] = (0.15, 0.25, 0.11) # slightly above grasp pose
    """Cartesian position of pen holder in workspace frame when dropping pen."""
    gripper_open_width_m: float = 0.024
    gripper_closed_width_m: float = 0.010
    gripper_timeout_s: float = 1.0
    gripper_external_effort_nm: float = -5.0
    gripper_pen_sleep_s: float = 2.0
    
    pen_height_delta: float = 0.1
    """Distance from pen tip to end effector tip."""

    # drawing parameters
    image_path: str = "circle.png"
    # image_path: str = "flower.png"
    progress_file_path: str = "draw_progress.csv"
    image_width_m: float = 0.06   # physical span of image in X [m]
    image_height_m: float = 0.06  # physical span of image in Y [m]
    image_threshold: int = 127   # B/W threshold


def image_pixels_to_meter_coords(
    image_path: str,
    width_meters: float,
    height_meters: float,
    threshold: int,
    log: logging.Logger,
):
    """
    Opens an image with Pillow, thresholds to B/W,
    and returns two lists of (x,y) coords in meters
    for black and for white pixels, using the image
    center as the (0,0) origin.
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    h_px, w_px = arr.shape
    black_mask = arr <= threshold
    white_mask = arr >  threshold
    black_rows, black_cols = np.where(black_mask)
    white_rows, white_cols = np.where(white_mask)
    cx = w_px / 2.0
    cy = h_px / 2.0
    scale_x = width_meters  / w_px
    scale_y = height_meters / h_px
    black_x = (black_cols - cx) * scale_x
    black_y = (black_rows - cy) * scale_y
    white_x = (white_cols - cx) * scale_x
    white_y = (white_rows - cy) * scale_y
    black_coords = list(zip(black_x.tolist(), black_y.tolist()))
    white_coords = list(zip(white_x.tolist(), white_y.tolist()))
    return black_coords, white_coords


def save_progress(index_drawn: int, log: logging.Logger):
    """Saves the index of the last successfully drawn point."""
    try:
        with open(config.progress_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([index_drawn])
        log.info(f"Progress saved: Point index {index_drawn} completed.")
    except IOError as e:
        log.error(f"Could not save progress to {config.progress_file_path}: {e}")


def load_progress(config: DrawImageConfig, log: logging.Logger) -> int:
    """Loads the index of the last drawn point. Returns starting index for the current run."""
    if os.path.exists(config.progress_file_path):
        try:
            with open(config.progress_file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                row = next(reader, None)
                if row and row[0].isdigit():
                    last_index_drawn = int(row[0])
                    log.info(f"Resuming. Last completed point index: {last_index_drawn}.")
                    return last_index_drawn + 1  # Start from the next point
                else:
                    log.warning(f"Progress file ({config.progress_file_path}) found but content is invalid. Starting from beginning.")
                    return 0 # Invalid content, start over
        except (IOError, csv.Error) as e:
            log.error(f"Error loading progress from {config.progress_file_path}: {e}. Starting from beginning.")
            return 0 # Error reading, start over
    return 0  # No progress file, start from the beginning


def goto_workspace(cart_pos: tuple[float, ...], driver: trossen_arm.TrossenArmDriver, config: DrawImageConfig, log: logging.Logger):
    """Go to a xyz position in the workspace coordinate system, copies current orientation."""
    log.info(f"🎯 Moving to workspace position {cart_pos}")
    cart_pos_current = driver.get_cartesian_positions()
    log.info(f"cart_pos_current: {cart_pos_current}")
    cart_pos_current[0] = config.cart_pos_home[0] - config.workspace_home_offset[0] + cart_pos[0]
    cart_pos_current[1] = config.cart_pos_home[1] + config.workspace_home_offset[1] - cart_pos[1] # Y axis is inverted in workspace frame
    cart_pos_current[2] = config.cart_pos_home[2] - config.workspace_home_offset[2] + cart_pos[2]
    log.info(f"cart_pos_current: {cart_pos_current}")
    # import pdb; pdb.set_trace()
    driver.set_cartesian_positions(cart_pos_current, trossen_arm.InterpolationSpace.cartesian)

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(h)
        logger.propagate = False

    try:
        config = DrawImageConfig()
        driver = trossen_arm.TrossenArmDriver()
        logger.info("🚀 Initializing driver...")
        driver.configure(
            config.arm_model,
            config.end_effector_model,
            config.ip_address,
            True # whether to clear the error state of the robot
        )

        logger.info("🦾 Starting up arm (position control)...")
        driver.set_all_modes(trossen_arm.Mode.position)
        joint_pos_startup = driver.get_all_positions()
        cart_pos_startup = driver.get_cartesian_positions()
        logger.info(f"joint_pos_startup: {joint_pos_startup}")
        logger.info(f"cart_pos_startup: {cart_pos_startup}")

        logger.info("🏠 Moving to home pose...")
        driver.set_all_positions(trossen_arm.VectorDouble(list(config.joint_pos_home)))
        joint_pos_home = driver.get_all_positions()
        cart_pos_home = driver.get_cartesian_positions()
        logger.info(f"joint_pos_home: {joint_pos_home}")
        logger.info(f"cart_pos_home: {cart_pos_home}")

        logger.info("🖋️ Going to pen holder")
        goto_workspace(config.pen_holder_cart_pos_ready, driver, config, logger)
        logger.info("🖐️ Opening gripper")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_open_width_m/2.0, config.gripper_timeout_s)
        logger.info("⬇️ Lowering to grasp pen")
        goto_workspace(config.pen_holder_cart_pos_grasp, driver, config, logger)
        logger.info("👌 Closing gripper")
        driver.set_gripper_position(config.gripper_closed_width_m/2.0, config.gripper_timeout_s)
        driver.set_gripper_mode(trossen_arm.Mode.external_effort)
        driver.set_gripper_external_effort(
            config.gripper_external_effort_nm,
            config.gripper_timeout_s,
            True
        )
        logger.info("⬆️ lifting pen")
        goto_workspace(config.pen_holder_cart_pos_ready, driver, config, logger)

        logger.info("📷 Loading & thresholding image...")
        black_pts_m, _ = image_pixels_to_meter_coords(
            config.image_path,
            config.image_width_m,
            config.image_height_m,
            config.image_threshold,
            logger,
        )
        logger.info(f"🖼️  {len(black_pts_m)} black pixels to draw.")

        logger.info("🎯 Going to origin of drawing (center of skin)")
        draw_origin = (
            (config.skin_lower_left_corner[0] + config.skin_upper_right_corner[0]) / 2.0,
            (config.skin_lower_left_corner[1] + config.skin_upper_right_corner[1]) / 2.0,
            config.skin_lower_left_corner[2] + config.pen_height_delta
        )
        goto_workspace(draw_origin, driver, config, logger)

        # logger.info("✍️  Starting image trace...")
        # start_index = load_progress(config, logger)
        # if start_index > 0 and start_index < len(black_pts_m):
        #     logger.info(f"Resuming drawing from point index {start_index} (out of {len(black_pts_m)}).")
        # num_points_drawn_this_session = 0
        # for i, (x_m, y_m) in enumerate(black_pts_m):
        #     if i < start_index:
        #         continue  # Skip points already drawn in a previous session

        #     xw = ox + x_m
        #     yw = oy + y_m

        #     # move above pixel
        #     driver.set_cartesian_positions(
        #         np.array([xw, yw, z_up]),
        #         trossen_arm.InterpolationSpace.cartesian
        #     )
        #     # pen down
        #     driver.set_cartesian_positions(
        #         np.array([xw, yw, z_down]),
        #         trossen_arm.InterpolationSpace.cartesian
        #     )
        #     # pen up
        #     driver.set_cartesian_positions(
        #         np.array([xw, yw, z_up]),
        #         trossen_arm.InterpolationSpace.cartesian
        #     )
        #     save_progress(i, logger)
        #     num_points_drawn_this_session += 1

        # if num_points_drawn_this_session > 0:
        #     logger.info(f"✍️  Completed drawing {num_points_drawn_this_session} points in this session.")
        # elif start_index > 0 and start_index >= len(black_pts_m) and len(black_pts_m) > 0:
        #      logger.info("✍️  No new points drawn. Previous drawing was already complete.")
        # elif len(black_pts_m) == 0:
        #     logger.info("🖼️ No black pixels to draw.")
        # else:
        #     logger.info("✍️  Drawing was already complete or no points to draw from start.")

        # if os.path.exists(config.progress_file_path):
        #     try:
        #         logger.info("Drawing complete or all resume points processed. Deleting progress file.")
        #         os.remove(config.progress_file_path)
        #     except OSError as e:
        #         logger.error(f"Error deleting progress file {config.progress_file_path}: {e}")


        logger.info("🎯 Returning to origin of drawing")
        goto_workspace(draw_origin, driver, config, logger)

        logger.info("🖋️ Going to pen holder")
        goto_workspace(config.pen_holder_cart_pos_ready, driver, config, logger)
        logger.info("⬇️ Lowering to drop pen")
        goto_workspace(config.pen_holder_cart_pos_drop, driver, config, logger)
        logger.info("🖐️ Opening gripper")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_open_width_m/2.0, config.gripper_timeout_s)
        logger.info("⬆️ Lifting up after dropping pen")
        goto_workspace(config.pen_holder_cart_pos_ready, driver, config, logger)
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    finally:
        driver.cleanup()
        driver.configure(
            config.arm_model,
            config.end_effector_model,
            config.ip_address,
            True # whether to clear the error state of the robot
        )
        logger.info("😴 Returning to sleep pose.")
        driver.set_all_modes(trossen_arm.Mode.position)
        driver.set_all_positions(trossen_arm.VectorDouble(list(config.joint_pos_sleep)))
        logger.info("🧹 Idling motors")
        driver.set_all_modes(trossen_arm.Mode.idle)
        logger.info("🏁 Script complete.")