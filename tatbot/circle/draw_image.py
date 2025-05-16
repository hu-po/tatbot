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
    t_flange_tool_x_offset: float = 0.0
    arm_up_delta: float = 0.06
    arm_forward_delta: float = 0.1
    pen_height_delta: float = 0.01
    # image_path: str = "flower.png"
    image_path: str = "circle.png"
    image_width_m: float = 0.08   # physical span of image in X [m]
    image_height_m: float = 0.08  # physical span of image in Y [m]
    image_threshold: int = 127   # B/W threshold
    gripper_open_width_m: float = 0.024
    gripper_closed_width_m: float = 0.010
    gripper_timeout_s: float = 1.0
    gripper_external_effort_nm: float = -5.0
    gripper_pen_sleep_s: float = 2.0
    progress_file_path: str = "draw_progress.csv"


def image_pixels_to_meter_coords(
    image_path: str,
    width_meters: float,
    height_meters: float,
    threshold: int = 127
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


def save_progress(index_drawn: int, logger_instance):
    """Saves the index of the last successfully drawn point."""
    try:
        with open(config.progress_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([index_drawn])
        logger_instance.info(f"Progress saved: Point index {index_drawn} completed.")
    except IOError as e:
        logger_instance.error(f"Could not save progress to {config.progress_file_path}: {e}")


def load_progress(logger_instance) -> int:
    """Loads the index of the last drawn point. Returns starting index for the current run."""
    if os.path.exists(config.progress_file_path):
        try:
            with open(config.progress_file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                row = next(reader, None)
                if row and row[0].isdigit():
                    last_index_drawn = int(row[0])
                    logger_instance.info(f"Resuming. Last completed point index: {last_index_drawn}.")
                    return last_index_drawn + 1  # Start from the next point
                else:
                    logger_instance.warning(f"Progress file ({config.progress_file_path}) found but content is invalid. Starting from beginning.")
                    return 0 # Invalid content, start over
        except (IOError, csv.Error) as e:
            logger_instance.error(f"Error loading progress from {config.progress_file_path}: {e}. Starting from beginning.")
            return 0 # Error reading, start over
    return 0  # No progress file, start from the beginning

if __name__ == "__main__":
    # 1) logging setup
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

    # 2) config & driver
    config = DrawImageConfig()
    driver = trossen_arm.TrossenArmDriver()
    sleep_joint_positions = None

    try:
        logger.info("🚀 Initializing driver...")
        driver.configure(
            config.arm_model,
            config.end_effector_model,
            config.ip_address,
            True
        )

        ee = driver.get_end_effector()
        ee.t_flange_tool[0] = config.t_flange_tool_x_offset
        driver.set_end_effector(ee)

        logger.info("🦾 Setting arm to position mode...")
        driver.set_all_modes(trossen_arm.Mode.position)

        logger.info("🛌 Capturing sleep joint positions...")
        sleep_joint_positions = np.array(driver.get_all_positions())

        # initial lift & forward
        origin = driver.get_cartesian_positions()
        logger.info(f"🔼 Lifting by {config.arm_up_delta}m")
        origin[2] += config.arm_up_delta
        driver.set_cartesian_positions(origin, trossen_arm.InterpolationSpace.cartesian)

        logger.info(f"▶️ Moving forward by {config.arm_forward_delta}m")
        origin[0] += config.arm_forward_delta
        driver.set_cartesian_positions(origin, trossen_arm.InterpolationSpace.cartesian)

        # open+close gripper to grasp pen
        logger.info("🖐️ Opening gripper")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_open_width_m/2.0,
                                    config.gripper_timeout_s)

        logger.info("🖋️ Closing gripper onto pen")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_closed_width_m/2.0,
                                    config.gripper_timeout_s)
        logger.info(f"sleep for {config.gripper_pen_sleep_s}s")
        time.sleep(config.gripper_pen_sleep_s)
        driver.set_gripper_mode(trossen_arm.Mode.external_effort)
        driver.set_gripper_external_effort(
            config.gripper_external_effort_nm,
            config.gripper_timeout_s,
            True
        )

        # lift pen to drawing height
        logger.info(f"🔼 Lifting pen by {config.pen_height_delta}m")
        origin[2] += config.pen_height_delta
        driver.set_cartesian_positions(origin, trossen_arm.InterpolationSpace.cartesian)

        logger.info("📷 Loading & thresholding image...")
        black_pts_m, _ = image_pixels_to_meter_coords(
            config.image_path,
            config.image_width_m,
            config.image_height_m,
            config.image_threshold
        )
        logger.info(f"🖼️  {len(black_pts_m)} black pixels to draw.")

        # origin of drawing = current XY
        draw_origin = driver.get_cartesian_positions()
        ox, oy, oz = draw_origin[0], draw_origin[1], draw_origin[2]
        z_up   = oz
        z_down = oz - config.pen_height_delta

        # --- Load progress before starting the drawing loop ---
        start_index = load_progress(logger)

        logger.info("✍️  Starting image trace...")
        if start_index > 0 and start_index < len(black_pts_m):
            logger.info(f"Resuming drawing from point index {start_index} (out of {len(black_pts_m)}).")
        elif start_index >= len(black_pts_m) and len(black_pts_m) > 0:
            # This case means the previous run completed all points but might have failed before cleanup.
            logger.info("Previous drawing seems completed. All points will be skipped if it was a full completion.")
            # The loop will handle this correctly by skipping all points.

        num_points_drawn_this_session = 0
        for i, (x_m, y_m) in enumerate(black_pts_m):
            if i < start_index:
                continue  # Skip points already drawn in a previous session

            xw = ox + x_m
            yw = oy + y_m

            # move above pixel
            driver.set_cartesian_positions(
                np.array([xw, yw, z_up]),
                trossen_arm.InterpolationSpace.cartesian
            )
            # pen down
            driver.set_cartesian_positions(
                np.array([xw, yw, z_down]),
                trossen_arm.InterpolationSpace.cartesian
            )
            # pen up
            driver.set_cartesian_positions(
                np.array([xw, yw, z_up]),
                trossen_arm.InterpolationSpace.cartesian
            )

            save_progress(i, logger)
            num_points_drawn_this_session += 1

        if num_points_drawn_this_session > 0:
            logger.info(f"✍️  Completed drawing {num_points_drawn_this_session} points in this session.")
        elif start_index > 0 and start_index >= len(black_pts_m) and len(black_pts_m) > 0:
             logger.info("✍️  No new points drawn. Previous drawing was already complete.")
        elif len(black_pts_m) == 0:
            logger.info("🖼️ No black pixels to draw.")
        else:
            logger.info("✍️  Drawing was already complete or no points to draw from start.")

        if os.path.exists(config.progress_file_path):
            try:
                logger.info("Drawing complete or all resume points processed. Deleting progress file.")
                os.remove(config.progress_file_path)
            except OSError as e:
                logger.error(f"Error deleting progress file {config.progress_file_path}: {e}")


        logger.info("🎯 Returning home...")
        driver.set_cartesian_positions(draw_origin,
                                      trossen_arm.InterpolationSpace.cartesian)

        logger.info("🖐️ Releasing pen")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_open_width_m/2.0,
                                    config.gripper_timeout_s)

        logger.info(f"▶️ Moving back by {config.arm_forward_delta}m")
        draw_origin[0] -= config.arm_forward_delta
        driver.set_cartesian_positions(draw_origin,
                                      trossen_arm.InterpolationSpace.cartesian)

        logger.info(f"🔼 Lowering by {config.arm_up_delta}m")
        draw_origin[2] -= config.arm_up_delta
        driver.set_cartesian_positions(draw_origin,
                                      trossen_arm.InterpolationSpace.cartesian)

        logger.info("✅ Done! Returning to sleep pose.")
        if sleep_joint_positions is not None:
            driver.set_all_positions(sleep_joint_positions)
        else:
            logger.warning("🟡 No sleep pose recorded; manual reset may be needed.")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.info("💀 Attempting error recovery...")
        try:
            driver.cleanup()
            logger.info("Re-configuring driver for recovery...")
            driver.configure(
                config.arm_model,
                config.end_effector_model,
                config.ip_address,
                True
            )
            logger.info("Setting arm to position mode post-recovery...")
            driver.set_all_modes(trossen_arm.Mode.position)
            if sleep_joint_positions is not None:
                logger.info("🛌 Returning to sleep pose post-recovery...")
                driver.set_all_positions(sleep_joint_positions)
            else:
                logger.warning("🟡 No sleep pose recorded; manual reset may be needed post-recovery.")
        except Exception as recovery_e:
            logger.error(f"❌❌ Error during recovery: {recovery_e}")
            logger.error("Manual intervention likely required.")

    finally:
        logger.info("🧹 Cleaning up and idling motors")
        driver.set_all_modes(trossen_arm.Mode.idle)
        driver.cleanup()
        logger.info("🏁 Script complete.")