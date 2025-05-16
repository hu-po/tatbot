#!/usr/bin/env python3
import logging
import math
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image
import trossen_arm

# -------------------------------------------------------
# Utility: suppress unwanted logger output
# -------------------------------------------------------
def fully_suppress_logger(logger_name, level=logging.ERROR):
    """Attempts to suppress a logger by setting its level, removing handlers,
    adding a NullHandler, and disabling propagation."""
    target_logger = logging.getLogger(logger_name)
    target_logger.setLevel(level)
    for h in target_logger.handlers[:]:
        target_logger.removeHandler(h)
        if hasattr(h, 'close'):
            h.close()
    target_logger.addHandler(logging.NullHandler())
    target_logger.propagate = False

# -------------------------------------------------------
# Image → black/white pixel lists → meter‐space coords
# -------------------------------------------------------
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
    # load & convert to grayscale
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    h_px, w_px = arr.shape

    # masks
    black_mask = arr <= threshold
    white_mask = arr >  threshold

    # pixel indices
    black_rows, black_cols = np.where(black_mask)
    white_rows, white_cols = np.where(white_mask)

    # center in px
    cx = w_px / 2.0
    cy = h_px / 2.0

    # scaling factors
    scale_x = width_meters  / w_px
    scale_y = height_meters / h_px

    # convert to meters
    black_x = (black_cols - cx) * scale_x
    black_y = (black_rows - cy) * scale_y
    white_x = (white_cols - cx) * scale_x
    white_y = (white_rows - cy) * scale_y

    black_coords = list(zip(black_x.tolist(), black_y.tolist()))
    white_coords = list(zip(white_x.tolist(), white_y.tolist()))
    return black_coords, white_coords

# -------------------------------------------------------
# Robot‐drawing configuration
# -------------------------------------------------------
@dataclass
class DrawImageConfig:
    ip_address: str = "192.168.1.3"
    t_flange_tool_x_offset: float = 0.0
    arm_up_delta: float = 0.06
    arm_forward_delta: float = 0.1
    pen_height_delta: float = 0.01

    # **Image‐drawing parameters**
    image_path: str = "flower.png"
    image_width_m: float = 0.06   # physical span of image in X [m]
    image_height_m: float = 0.06  # physical span of image in Y [m]
    image_threshold: int = 127   # B/W threshold

    # gripper settings
    gripper_open_width_m: float = 0.024
    gripper_closed_width_m: float = 0.010
    gripper_external_effort_nm: float = -5.0
    gripper_timeout_s: float = 1.0

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
    fully_suppress_logger("trossen_arm", logging.ERROR)

    # 2) config & driver
    config = DrawImageConfig()
    driver = trossen_arm.TrossenArmDriver()
    sleep_joint_positions = None

    try:
        logger.info("🚀 Initializing driver...")
        driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_follower,
            config.ip_address,
            False
        )
        fully_suppress_logger("trossen_arm", logging.ERROR)

        ee = driver.get_end_effector()
        ee.t_flange_tool[0] = config.t_flange_tool_x_offset
        driver.set_end_effector(ee)

        logger.info("🦾 Setting arm to position mode...")
        driver.set_all_modes(trossen_arm.Mode.position)

        logger.info("🛌 Capturing sleep joint positions...")
        sleep_joint_positions = np.array(driver.get_all_positions())

        # initial lift & forward
        cart = driver.get_cartesian_positions()
        logger.info(f"🔼 Lifting by {config.arm_up_delta}m")
        cart[2] += config.arm_up_delta
        driver.set_cartesian_positions(cart, trossen_arm.InterpolationSpace.cartesian)

        logger.info(f"▶️ Moving forward by {config.arm_forward_delta}m")
        cart[0] += config.arm_forward_delta
        driver.set_cartesian_positions(cart, trossen_arm.InterpolationSpace.cartesian)

        # open+close gripper to grasp pen
        logger.info("🖐️ Opening gripper")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_open_width_m/2.0,
                                    config.gripper_timeout_s)

        logger.info("🖋️ Closing gripper onto pen")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_closed_width_m/2.0,
                                    config.gripper_timeout_s)
        driver.set_gripper_mode(trossen_arm.Mode.external_effort)
        driver.set_gripper_external_effort(
            config.gripper_external_effort_nm,
            config.gripper_timeout_s,
            True
        )

        # lift pen to drawing height
        logger.info(f"🔼 Lifting pen by {config.pen_height_delta}m")
        cart[2] += config.pen_height_delta
        driver.set_cartesian_positions(cart, trossen_arm.InterpolationSpace.cartesian)

        # ---------------------------------------------------
        #   IMAGE DRAWING
        # ---------------------------------------------------
        logger.info("📷 Loading & thresholding image...")
        black_pts_m, _ = image_pixels_to_meter_coords(
            config.image_path,
            config.image_width_m,
            config.image_height_m,
            config.image_threshold
        )
        logger.info(f"🖼️  {len(black_pts_m)} black pixels detected")

        # origin of drawing = current XY
        draw_origin = driver.get_cartesian_positions()
        logger.info(f"DEBUG: draw_origin value: {draw_origin}, type: {type(draw_origin)}")
        ox, oy, oz = draw_origin[0], draw_origin[1], draw_origin[2]
        z_up   = oz
        z_down = oz - config.pen_height_delta

        logger.info("✍️  Starting image trace...")
        for x_m, y_m in black_pts_m:
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

        logger.info("🎯 Returning home...")
        driver.set_cartesian_positions(draw_origin,
                                      trossen_arm.InterpolationSpace.cartesian)

        # release pen & retreat
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
        # ... existing recovery logic unchanged ...
        # (omitted here for brevity—just copy your original except/recovery/finally)
        raise

    finally:
        logger.info("🧹 Cleaning up and idling motors")
        driver.set_all_modes(trossen_arm.Mode.idle)
        driver.cleanup()
        logger.info("🏁 Script complete.")
