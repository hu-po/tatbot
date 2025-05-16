import trossen_arm
import math
import time
import numpy as np
import logging
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.getLogger("trossen_arm").setLevel(logging.ERROR)

@dataclass
class CartesianCircleConfig:
    # IP address of the Trossen arm control box.
    ip_address: str = "192.168.1.3"
    # Offset in meters along the x-axis of the tool (e.g., pen tip) from the arm's flange.
    t_flange_tool_x_offset: float = 0.0
    # Vertical distance in meters to move the arm upwards as an initial setup motion.
    arm_up_delta: float = 0.05
    # Horizontal distance in meters to move the arm forwards as an initial setup motion.
    arm_forward_delta: float = 0.1
    # Radius in meters of the circle to be traced by the end effector.
    circle_radius: float = 0.01
    # Number of discrete points used to approximate the circle's circumference.
    circle_num_points: int = 6
    # Target external effort in Newton-meters for the gripper to apply when closing (e.g., to gently grasp a pen).
    gripper_external_effort_nm: float = -1.0
    # Maximum current in Amperes to be supplied to the gripper motor, as a safety limit.
    gripper_current_limit_a: float = 1.0
    # Delay in milliseconds to wait after issuing a command for the gripper to open, allowing time for the action.
    gripper_open_delay_ms: int = 500
    # Initial opening of the gripper in meters (e.g., to make space for a pen).
    gripper_open_width_m: float = 0.009
    # Width of the pen in meters.
    pen_width_m: float = 0.009

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
        logging.getLogger("trossen_arm").setLevel(logging.ERROR)

        end_effector_obj = driver.get_end_effector()
        end_effector_obj.t_flange_tool[0] = config.t_flange_tool_x_offset
        driver.set_end_effector(end_effector_obj)

        logger.info("🦾 Setting arm modes to position...")
        driver.set_all_modes(trossen_arm.Mode.position)

        logger.info("🖐️ Opening gripper...")
        driver.set_gripper_mode(trossen_arm.Mode.position)
        driver.set_gripper_position(config.gripper_open_width_m/2.0)
        time.sleep(config.gripper_open_delay_ms / 1000.0)

        logger.info(f"🖋️ Closing gripper slowly to grasp pen with {config.gripper_external_effort_nm}N*m effort...")
        driver.set_gripper_mode(trossen_arm.Mode.external_effort)
        driver.set_gripper_external_effort(
            config.gripper_external_effort_nm,
            config.gripper_current_limit_a,
            True
        )
        driver.set_gripper_position(config.pen_width_m/2.0)

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

        logger.info(f"🔄 Tracing a circle with radius {config.circle_radius}m and {config.circle_num_points} points...")

        circle_center_x = cartesian_positions[0]
        circle_center_y = cartesian_positions[1]

        for i in range(config.circle_num_points + 1):
            angle = 2 * math.pi * i / config.circle_num_points
            cartesian_positions[0] = circle_center_x + config.circle_radius * math.cos(angle)
            cartesian_positions[1] = circle_center_y + config.circle_radius * math.sin(angle)

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
        driver.set_gripper_position(config.gripper_open_width_m/2.0)
        time.sleep(config.gripper_open_delay_ms / 1000.0)

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
                time.sleep(config.gripper_open_delay_ms / 1000.0)
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
