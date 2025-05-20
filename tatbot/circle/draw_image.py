import csv
from dataclasses import dataclass
import logging
import os
import time
import random

import numpy as np
from PIL import Image
import trossen_arm

@dataclass
class DrawImageConfig:
    
    # Left arm
    # arm_model: trossen_arm.Model = trossen_arm.Model.wxai_v0
    # ip_address: str = "192.168.1.2"
    # end_effector_model: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_leader

    # Right arm
    arm_model: trossen_arm.Model = trossen_arm.Model.wxai_v0
    ip_address: str = "192.168.1.3"
    end_effector_model: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_follower
    
    # pre-defined joint positions
    joint_pos_sleep: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """7d joint radians: sleep pose,robot is folded up, motors can be released."""
    joint_pos_home: tuple[float, ...] = (0.0, 1.05, 0.5, -1.06, 0.0, 0.0, 0.0)
    """7d joint radians: home pose, robot is active, staring down at workspace"""
    cart_pos_home: tuple[float, ...] = (0.12, 0.0, 0.005)
    """(x, y, z) in meters: Cartesian position of end effector in home pose in the robot's base frame."""

    # workspace definitions
    workspace_home_offset: tuple[float, ...] = (0.01, 0.33, 0.1)
    """Cartesian offset of workspace frame from robot base frame."""
    skin_lower_left_corner: tuple[float, ...] = (0.1, 0.3, 0.04)
    """Lower left corner of skin in workspace frame."""
    skin_upper_right_corner: tuple[float, ...] = (0.22, 0.39, 0.04)
    """Upper right corner of skin in workspace frame."""
    
    # Tattoo Pen
    pen_holder_cart_pos_ready: tuple[float, ...] = (0.15, 0.25, 0.22)
    """(x, y, z) in meters: cartesian position of pen holder in workspace frame when ready to grasp pen."""
    pen_holder_cart_pos_grasp: tuple[float, ...] = (0.15, 0.25, 0.13)
    """(x, y, z) in meters: cartesian position of pen holder in workspace frame when grasping pen."""
    pen_holder_cart_pos_drop: tuple[float, ...] = (0.15, 0.25, 0.14) # slightly above grasp pose
    """(x, y, z) in meters: cartesian position of pen holder in workspace frame when dropping pen."""
    gripper_open_width: float = 0.04
    """meters: width of the gripper when open."""
    gripper_grip_width: float = 0.032
    """meters: width of the gripper before using effort based gripping."""
    gripper_grip_timeout: float = 1.0
    """seconds: timeout for effort based gripping."""
    gripper_grip_effort: float = -20.0
    """newtons: maximum force for effort based gripping."""
    pen_height_delta: float = 0.136
    """meters: distance from pen tip to end effector tip."""
    pen_stroke_length: float = 0.008
    """meters: length of pen stroke when drawing a pixel."""

    # Ink Cup
    ink_cup_cart_pos_ready: tuple[float, ...] = (0.11, 0.31, 0.19)
    """(x, y, z) in meters: cartesian position of ink cup in workspace frame when ready to dip ink cup."""
    ink_cup_cart_pos_dip: tuple[float, ...] = (0.11, 0.31, 0.17)
    """(x, y, z) in meters: cartesian position of ink cup in workspace frame when dipping ink cup."""

    # drawing parameters
    # image_path: str = "circle.png"
    image_path: str = "flower.png"
    progress_file_path: str = "draw_progress.csv"
    """path to file to save progress of drawing."""
    image_width_m: float = 0.04
    """meters: width of the image in the workspace frame."""
    image_height_m: float = 0.04
    """meters: height of the image in the workspace frame."""
    image_threshold: int = 127
    """[0, 255] threshold for B/W image."""
    max_draw_pixels: int = 600
    """maximum number of black pixels to draw. If 0, draw all."""
    num_pixels_per_ink_dip: int = 60
    """Number of pixels to draw before dipping the pen in ink cup again."""

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
        driver.set_gripper_position(config.gripper_open_width/2.0, config.gripper_grip_timeout)
        logger.info("⬇️ Lowering to grasp pen")
        goto_workspace(config.pen_holder_cart_pos_grasp, driver, config, logger)
        logger.info("👌 Closing gripper")
        driver.set_gripper_position(config.gripper_grip_width/2.0, config.gripper_grip_timeout)
        driver.set_gripper_mode(trossen_arm.Mode.external_effort)
        driver.set_gripper_external_effort(
            config.gripper_grip_effort,
            config.gripper_grip_timeout,
            True
        )
        logger.info("⬆️ lifting pen")
        goto_workspace(config.pen_holder_cart_pos_ready, driver, config, logger)

        logger.info("🎨 Going to ink cup")
        goto_workspace(config.ink_cup_cart_pos_ready, driver, config, logger)
        logger.info("✒️ Dipping into ink cup")
        goto_workspace(config.ink_cup_cart_pos_dip, driver, config, logger)
        logger.info("⬆️ Retracting from ink cup")
        goto_workspace(config.ink_cup_cart_pos_ready, driver, config, logger)

        logger.info("📷 Loading & thresholding image...")
        """
        Opens an image with Pillow, thresholds to B/W,
        and returns two lists of (x,y) coords in meters
        for black and for white pixels, using the image
        center as the (0,0) origin.
        """
        img = Image.open(config.image_path).convert("L")
        arr = np.array(img)
        h_px, w_px = arr.shape
        black_mask = arr <= config.image_threshold
        white_mask = arr > config.image_threshold
        black_rows, black_cols = np.where(black_mask)
        white_rows, white_cols = np.where(white_mask)
        cx = w_px / 2.0
        cy = h_px / 2.0
        scale_x = config.image_width_m  / w_px
        scale_y = config.image_height_m / h_px
        black_x = (black_cols - cx) * scale_x
        black_y = (black_rows - cy) * scale_y
        white_x = (white_cols - cx) * scale_x
        white_y = (white_rows - cy) * scale_y
        black_coords = list(zip(black_x.tolist(), black_y.tolist()))
        # Randomly sample up to max_draw_pixels if set
        if config.max_draw_pixels and config.max_draw_pixels > 0:
            if len(black_coords) > config.max_draw_pixels:
                black_coords = random.sample(black_coords, config.max_draw_pixels)
        num_points_to_draw = len(black_coords)
        logger.info(f"🖼️  {num_points_to_draw} black pixels to draw.")

        logger.info("🎯 Going to origin of drawing (center of skin)")
        draw_origin = (
            (config.skin_lower_left_corner[0] + config.skin_upper_right_corner[0]) / 2.0,
            (config.skin_lower_left_corner[1] + config.skin_upper_right_corner[1]) / 2.0,
            config.skin_lower_left_corner[2] + config.pen_height_delta
        )
        goto_workspace(draw_origin, driver, config, logger)

        logger.info("✍️  Starting image trace...")
        if os.path.exists(config.progress_file_path):
            try:
                with open(config.progress_file_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    row = next(reader, None)
                    if row and row[0].isdigit():
                        last_index_drawn = int(row[0])
                        logger.info(f"Resuming. Last completed point index: {last_index_drawn}.")
                        start_index = last_index_drawn + 1  # Start from the next point
                    else:
                        logger.warning(f"Progress file ({config.progress_file_path}) found but content is invalid. Starting from beginning.")
                        start_index = 0 # Invalid content, start over
            except (IOError, csv.Error) as e:
                logger.error(f"Error loading progress from {config.progress_file_path}: {e}. Starting from beginning.")
                start_index = 0 # Error reading, start over
        else:
            start_index = 0  # No progress file, start from the beginning
        if start_index > 0 and start_index < num_points_to_draw:
            logger.info(f"Resuming drawing from point index {start_index} (out of {num_points_to_draw}).")
        num_points_drawn_this_session = 0
        for i, (x_m, y_m) in enumerate(black_coords):
            if i < start_index:
                continue  # Skip points already drawn in a previous session
            # Dip pen in ink after every num_pixels_per_ink_dip pixels
            if config.num_pixels_per_ink_dip > 0 and num_points_drawn_this_session > 0 and num_points_drawn_this_session % config.num_pixels_per_ink_dip == 0:
                logger.info("🎨 Going to ink cup for re-dip")
                goto_workspace(config.ink_cup_cart_pos_ready, driver, config, logger)
                logger.info("✒️ Dipping into ink cup")
                goto_workspace(config.ink_cup_cart_pos_dip, driver, config, logger)
                logger.info("⬆️ Retracting from ink cup")
                goto_workspace(config.ink_cup_cart_pos_ready, driver, config, logger)
            xw = draw_origin[0] + x_m
            yw = draw_origin[1] + y_m
            logger.info(f"🎯 Moving to pixel {i} of {num_points_to_draw} at {xw}, {yw}")
            goto_workspace((xw, yw, draw_origin[2]), driver, config, logger)
            logger.info("🖌️  Drawing pixel")
            goto_workspace((xw, yw, draw_origin[2] - config.pen_stroke_length), driver, config, logger)
            logger.info(" lifting pen")
            goto_workspace((xw, yw, draw_origin[2]), driver, config, logger)
            try:
                with open(config.progress_file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i])
                logger.info(f"Progress saved: Point index {i} completed.")
            except IOError as e:
                logger.error(f"Could not save progress to {config.progress_file_path}: {e}")
            num_points_drawn_this_session += 1

        if num_points_drawn_this_session > 0:
            logger.info(f"✍️  Completed drawing {num_points_drawn_this_session} points in this session.")
        elif start_index > 0 and start_index >= num_points_to_draw:
             logger.info("✍️  No new points drawn. Previous drawing was already complete.")
        elif num_points_to_draw == 0:
            logger.info("🖼️ No black pixels to draw.")
        else:
            logger.info("✍️  Drawing was already complete or no points to draw from start.")

        if os.path.exists(config.progress_file_path):
            try:
                logger.info("Drawing complete or all resume points processed. Deleting progress file.")
                os.remove(config.progress_file_path)
            except OSError as e:
                logger.error(f"Error deleting progress file {config.progress_file_path}: {e}")

        logger.info("🎯 Returning to origin of drawing")
        goto_workspace(draw_origin, driver, config, logger)

        logger.info("🖋️ Going to pen holder")
        goto_workspace(config.pen_holder_cart_pos_ready, driver, config, logger)
        logger.info("⬇️ Lowering to drop pen")
        goto_workspace(config.pen_holder_cart_pos_drop, driver, config, logger)
        logger.info("🖐️ Opening gripper")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_open_width/2.0, config.gripper_grip_timeout)
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
        logger.info("🖐️ Opening gripper")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_open_width/2.0, config.gripper_grip_timeout)
        logger.info("😴 Returning to sleep pose.")
        driver.set_all_modes(trossen_arm.Mode.position)
        driver.set_all_positions(trossen_arm.VectorDouble(list(config.joint_pos_sleep)))
        logger.info("🧹 Idling motors")
        driver.set_all_modes(trossen_arm.Mode.idle)
        logger.info("🏁 Script complete.")