import trossen_arm
import math
import time
import numpy as np
import logging
from dataclasses import dataclass

# Define the suppression function
def fully_suppress_logger(logger_name, level=logging.ERROR):
    """Attempts to suppress a logger by setting its level, removing handlers,
    adding a NullHandler, and disabling propagation."""
    target_logger = logging.getLogger(logger_name)
    target_logger.setLevel(level)
    # Remove existing handlers
    for h in target_logger.handlers[:]: # Iterate over a copy
        target_logger.removeHandler(h)
        if hasattr(h, 'close'): # Check if handler has close method
            h.close() # Close handler to release resources if any
    # Add a NullHandler to prevent "No handlers could be found" warnings
    target_logger.addHandler(logging.NullHandler())
    target_logger.propagate = False

# 1. Configure the root logger first.
# This will make ERROR the default for all loggers, including trossen_arm's,
# unless they are specifically configured otherwise and don't propagate.
logging.basicConfig(level=logging.ERROR)

# 2. Now, get and configure your script's specific logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Your script's logger to INFO

# If your script's logger needs specific formatting, and basicConfig wasn't used for it:
if not logger.handlers: # Add handler only if not already configured (e.g. by basicConfig)
    handler = logging.StreamHandler()
    # Using your original format for the script's logger
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False # Prevent messages from going to the root logger's basicConfig handler

# 3. Explicitly suppress the trossen_arm logger.
fully_suppress_logger("trossen_arm", logging.ERROR)

@dataclass
class CartesianCircleConfig:
    # IP address of the Trossen arm control box.
    ip_address: str = "192.168.1.3"
    # Offset in meters along the x-axis of the tool (e.g., pen tip) from the arm's flange.
    t_flange_tool_x_offset: float = 0.0
    # Vertical distance in meters to move the arm upwards as an initial setup motion.
    arm_up_delta: float = 0.06
    # Horizontal distance in meters to move the arm forwards as an initial setup motion.
    arm_forward_delta: float = 0.1
    # Radius in meters of the circle to be traced by the end effector.
    circle_radius: float = 0.02
    # Number of discrete points used to approximate the circle's circumference.
    circle_num_points: int = 12
    # Initial opening of the gripper in meters (e.g., to make space for a pen).
    gripper_open_width_m: float = 0.024
    # opening of the gripper in meters when holding a pen (e.g., to make space for a pen).
    gripper_closed_width_m: float = 0.015
    # Target external effort in Newton-meters for the gripper to apply when closing (e.g., to gently grasp a pen).
    gripper_external_effort_nm: float = -5.0
    # Goal time in seconds when the goal external effort should be reached, default is 2.0s.
    gripper_timeout_s: float = 1.0
    # Vertical distance in meters to move the pen up and down when drawing points on the circle
    pen_height_delta: float = 0.01

if __name__=='__main__':
    config = CartesianCircleConfig()
    driver = trossen_arm.TrossenArmDriver()
    recovery_already_configured_flag = True
    sleep_joint_positions = None

    try:
        logger.info("🚀 Initializing and configuring the driver...")
        driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_follower,
            config.ip_address,
            False
        )
        # Re-affirm trossen_arm logger suppression after potential (re)configuration by the driver.
        fully_suppress_logger("trossen_arm", logging.ERROR)

        end_effector_obj = driver.get_end_effector()
        end_effector_obj.t_flange_tool[0] = config.t_flange_tool_x_offset
        driver.set_end_effector(end_effector_obj)

        logger.info("🦾 Setting arm modes to position...")
        driver.set_all_modes(trossen_arm.Mode.position)

        logger.info("🛌 Getting initial joint positions for sleep state...")
        sleep_joint_positions = np.array(driver.get_all_positions())

        cartesian_positions = driver.get_cartesian_positions()

        logger.info(f"🔼 Moving arm up by {config.arm_up_delta}m...")
        cartesian_positions[2] += config.arm_up_delta
        driver.set_cartesian_positions(
            cartesian_positions,
            trossen_arm.InterpolationSpace.cartesian
        )

        logger.info(f"▶️ Moving arm forward by {config.arm_forward_delta}m...")
        cartesian_positions[0] += config.arm_forward_delta
        driver.set_cartesian_positions(
            cartesian_positions,
            trossen_arm.InterpolationSpace.cartesian
        )

        logger.info("🖐️ Opening gripper...")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_open_width_m/2.0, config.gripper_timeout_s)

        logger.info(f"🖋️ Closing gripper to grasp pen with {config.gripper_external_effort_nm}N*m effort...")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_closed_width_m/2.0, config.gripper_timeout_s)
        driver.set_gripper_mode(trossen_arm.Mode.external_effort)
        driver.set_gripper_external_effort(
            config.gripper_external_effort_nm,
            config.gripper_timeout_s,
            True
        )

        logger.info(f"🔼 Moving arm up by {config.pen_height_delta}m...")
        cartesian_positions[2] += config.pen_height_delta
        driver.set_cartesian_positions(
            cartesian_positions,
            trossen_arm.InterpolationSpace.cartesian
        )

        logger.info(f"🔄 Tracing a circle with radius {config.circle_radius}m and {config.circle_num_points} points...")

        circle_center_x = cartesian_positions[0]
        circle_center_y = cartesian_positions[1]

        for i in range(config.circle_num_points + 1):
            angle = 2 * math.pi * i / config.circle_num_points
            cartesian_positions[0] = circle_center_x + config.circle_radius * math.cos(angle)
            cartesian_positions[1] = circle_center_y + config.circle_radius * math.sin(angle)

            # Move the pen to the circle point
            driver.set_cartesian_positions(
                cartesian_positions,
                trossen_arm.InterpolationSpace.cartesian
            )
            
            # Move the pen down
            cartesian_positions[2] -= config.pen_height_delta
            driver.set_cartesian_positions(
                cartesian_positions,
                trossen_arm.InterpolationSpace.cartesian
            )

            # Move the pen up
            cartesian_positions[2] += config.pen_height_delta
            driver.set_cartesian_positions(
                cartesian_positions,
                trossen_arm.InterpolationSpace.cartesian
            )
            
        logger.info("🎯 Returning to the center of the circle...")
        cartesian_positions[0] = circle_center_x
        cartesian_positions[1] = circle_center_y
        driver.set_cartesian_positions(
            cartesian_positions,
            trossen_arm.InterpolationSpace.cartesian
        )

        logger.info("🖐️ Releasing pen...")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_open_width_m/2.0, config.gripper_timeout_s)

        logger.info(f"▶️ Moving arm backwards by {config.arm_forward_delta}m...")
        cartesian_positions[0] -= config.arm_forward_delta
        driver.set_cartesian_positions(
            cartesian_positions,
            trossen_arm.InterpolationSpace.cartesian
        )

        logger.info(f"🔼 Moving arm down by {config.arm_up_delta}m...")
        cartesian_positions[2] -= config.arm_up_delta
        driver.set_cartesian_positions(
            cartesian_positions,
            trossen_arm.InterpolationSpace.cartesian
        )

        logger.info("✅ Operations complete. Returning to sleep position...")
        if sleep_joint_positions is not None:
            driver.set_all_positions(sleep_joint_positions)
        else:
            logger.warning("🟡 Sleep positions not defined, cannot return to sleep position automatically.")

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")
        logger.info("🛠️ Attempting to recover and return to sleep position...")
        try:
            driver.cleanup()
            logger.info("🔧 Re-configuring the driver after error...")
            driver.configure(
                trossen_arm.Model.wxai_v0,
                trossen_arm.StandardEndEffector.wxai_v0_follower,
                config.ip_address,
                True
            )
            # Re-affirm trossen_arm logger suppression after potential (re)configuration by the driver in error recovery.
            fully_suppress_logger("trossen_arm", logging.ERROR)
            logger.info("🦾 Setting arm modes to position after error recovery...")
            driver.set_all_modes(trossen_arm.Mode.position)
            if sleep_joint_positions is not None:
                logger.info("🛌 Returning to sleep position after error recovery...")
                driver.set_all_positions(sleep_joint_positions)
            else:
                logger.warning("🟡 Sleep positions were not captured; cannot automatically return to sleep position during recovery.")
                logger.warning(" MANUAL INTERVENTION REQUIRED: Please manually ensure the arm is in a safe state. 🔴")
            try:
                logger.info("🖐️ Attempting to open gripper during recovery...")
                driver.set_gripper_mode(trossen_arm.Mode.position)
                driver.set_gripper_position(config.gripper_open_width_m/2.0)
                time.sleep(config.gripper_open_delay_s)
            except Exception as gripper_recovery_e:
                logger.error(f"❌ Failed to open gripper during recovery: {gripper_recovery_e}")
        except Exception as recovery_e:
            logger.error(f"❌ An error occurred during recovery: {recovery_e}")
            logger.error("🆘 Unable to automatically recover. Please check the arm manually. 🔴")

    finally:
        logger.info("🧹 Performing final cleanup...")
        logger.info("💨 Setting all motors to idle mode...")
        driver.set_all_modes(trossen_arm.Mode.idle)
        driver.cleanup()
        logger.info("🏁 Script finished.")
