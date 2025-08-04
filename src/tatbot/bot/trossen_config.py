"""
Configure the robot via YAML, originally from:

https://github.com/TrossenRobotics/trossen_arm/blob/main/demos/python/configuration_in_yaml.py
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trossen_arm
import yaml

from tatbot.data.arms import Arms
from tatbot.data.pose import ArmPose
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger("bot.trossen", "🎛️")


def load_pose_from_yaml(pose_name: str) -> ArmPose:
    """Load an ArmPose from a YAML file in the poses directory."""
    poses_dir = Path("~/tatbot/src/conf/poses").expanduser()
    pose_path = poses_dir / f"{pose_name}.yaml"
    
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    
    with open(pose_path) as f:
        pose_data = yaml.safe_load(f)
    
    return ArmPose(**pose_data)


@dataclass
class TrossenConfig:
    debug: bool = False
    """Enable debug logging."""
    arm_l_ip: str = "192.168.1.3"
    """IP address of the left arm."""
    arm_l_config_filepath: str = os.path.expanduser("~/tatbot/config/trossen/arm_l.yaml")
    """YAML file containing left arm config."""
    test_pose_name_l: str = "left/rest"
    """Test pose for the left arm (ArmPose)."""
    arm_r_ip: str = "192.168.1.2"
    """IP address of the right arm."""
    arm_r_config_filepath: str = os.path.expanduser("~/tatbot/config/trossen/arm_r.yaml")
    """YAML file containing right arm config."""
    test_pose_name_r: str = "right/rest"
    """Test pose for the right arm (ArmPose)."""
    test_tolerance: float = 0.1
    """Tolerance for the test pose."""


def print_configurations(driver: trossen_arm.TrossenArmDriver):
    log.debug(f"EEPROM factory reset flag: {driver.get_factory_reset_flag()}")
    log.debug(f"EEPROM IP method: {driver.get_ip_method()}")
    log.debug(f"EEPROM manual IP: {driver.get_manual_ip()}")
    log.debug(f"EEPROM DNS: {driver.get_dns()}")
    log.debug(f"EEPROM gateway: {driver.get_gateway()}")
    log.debug(f"EEPROM subnet: {driver.get_subnet()}")
    log.debug(f"EEPROM effort corrections: {driver.get_effort_corrections()}")
    log.debug(f"EEPROM friction transition velocities: {driver.get_friction_transition_velocities()}")
    log.debug(f"EEPROM friction constant terms: {driver.get_friction_constant_terms()}")
    log.debug(f"EEPROM friction coulomb coefs: {driver.get_friction_coulomb_coefs()}")
    log.debug(f"EEPROM friction viscous coefs: {driver.get_friction_viscous_coefs()}")
    log.debug(f"Modes: {[mode.value for mode in driver.get_modes()]}")

    end_effector = driver.get_end_effector()
    log.debug("End effector:")
    log.debug("  palm:")
    log.debug(f"    mass: {end_effector.palm.mass}")
    log.debug(f"    inertia: {end_effector.palm.inertia}")
    log.debug(f"    origin xyz: {end_effector.palm.origin_xyz}")
    log.debug(f"    origin rpy: {end_effector.palm.origin_rpy}")
    log.debug("  finger left:")
    log.debug(f"    mass: {end_effector.finger_left.mass}")
    log.debug(f"    inertia: {end_effector.finger_left.inertia}")
    log.debug(f"    origin xyz: {end_effector.finger_left.origin_xyz}")
    log.debug(f"    origin rpy: {end_effector.finger_left.origin_rpy}")
    log.debug("  finger right:")
    log.debug(f"    mass: {end_effector.finger_right.mass}")
    log.debug(f"    inertia: {end_effector.finger_right.inertia}")
    log.debug(f"    origin xyz: {end_effector.finger_right.origin_xyz}")
    log.debug(f"    origin rpy: {end_effector.finger_right.origin_rpy}")
    log.debug(f"  offset finger left: {end_effector.offset_finger_left}")
    log.debug(f"  offset finger right: {end_effector.offset_finger_right}")
    log.debug(f"  pitch circle radius: {end_effector.pitch_circle_radius}")
    log.debug(f"  t flange tool: {end_effector.t_flange_tool}")

    joint_limits = driver.get_joint_limits()
    log.debug("Joint limits:")
    for i, joint_limit in enumerate(joint_limits):
        log.debug(f"  Joint {i}:")
        log.debug(f"    position min: {joint_limit.position_min}")
        log.debug(f"    position max: {joint_limit.position_max}")
        log.debug(f"    position tolerance: {joint_limit.position_tolerance}")
        log.debug(f"    velocity max: {joint_limit.velocity_max}")
        log.debug(f"    velocity tolerance: {joint_limit.velocity_tolerance}")
        log.debug(f"    effort max: {joint_limit.effort_max}")
        log.debug(f"    effort tolerance: {joint_limit.effort_tolerance}")

    motor_parameters = driver.get_motor_parameters()
    log.debug("Motor parameters:")
    for i, motor_param in enumerate(motor_parameters):
        log.debug(f"  Joint {i}:")
        for mode, param in motor_param.items():
            log.debug(f"    Mode {mode.value}:")
            log.debug("      Position loop:")
            log.debug(
                f"        kp: {param.position.kp}, ki: {param.position.ki}, "
                f"kd: {param.position.kd}, imax: {param.position.imax}"
            )
            log.debug("      Velocity loop:")
            log.debug(
                f"        kp: {param.velocity.kp}, ki: {param.velocity.ki}, "
                f"kd: {param.velocity.kd}, imax: {param.velocity.imax}"
            )

    algorithm_parameter = driver.get_algorithm_parameter()
    log.debug("Algorithm parameter:")
    log.debug(f"  singularity threshold: {algorithm_parameter.singularity_threshold}")


def driver_from_arms(config: Arms) -> tuple[trossen_arm.TrossenArmDriver, trossen_arm.TrossenArmDriver]:
    log.info("Setting up left arm driver...")
    arm_l = trossen_arm.TrossenArmDriver()
    arm_l.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_base,
        config.ip_address_l,
        True,  # clear_error
        timeout=config.connection_timeout,
    )
    config_filepath = os.path.expanduser(config.arm_l_config_filepath)
    log.debug(f"Loading left arm config from {config_filepath}")
    arm_l.load_configs_from_file(config_filepath)
    arm_l.set_all_modes(trossen_arm.Mode.position)

    log.info("Setting up right arm driver...")
    arm_r = trossen_arm.TrossenArmDriver()
    arm_r.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_base,
        config.ip_address_r,
        True,  # clear_error
        timeout=config.connection_timeout,
    )
    config_filepath = os.path.expanduser(config.arm_r_config_filepath)
    log.debug(f"Loading right arm config from {config_filepath}")
    arm_r.load_configs_from_file(config_filepath)
    arm_r.set_all_modes(trossen_arm.Mode.position)
    return arm_l, arm_r


def configure_arm(filepath: str, ip: str, test_pose_name: str, test_tolerance: float):
    assert os.path.exists(filepath), f"❌📄 yaml file does not exist: {filepath}"
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,  # model
        trossen_arm.StandardEndEffector.wxai_v0_base,  # end_effector
        ip,  # serv_ip
        True,  # clear_error
    )
    assert driver is not None, f"❌🦾 failed to connect to arm {ip}"
    # print_configurations(driver)
    # driver.save_configs_to_file(filepath)
    # log.info(f"✅📄 saved config to {filepath}")
    driver.load_configs_from_file(filepath)
    log.info(f"✅📄 loaded config from {filepath}")
    print_configurations(driver)
    log.info(f"✅🦾 arm {ip} configured successfully")
    driver.set_all_modes(trossen_arm.Mode.position)
    sleep_pose = driver.get_all_positions()[:7]
    log.info(f"🦾 sleep pose: {sleep_pose}")
    test_pose = load_pose_from_yaml(test_pose_name).joints
    log.info(f"🦾 Testing arm {ip} with pose {test_pose}")
    driver.set_all_positions(trossen_arm.VectorDouble(test_pose), blocking=True)
    current_pose = driver.get_all_positions()[:7]
    assert np.allclose(current_pose, test_pose, atol=test_tolerance), (
        f"❌🦾 current pose {current_pose} does not match test pose {test_pose}"
    )
    driver.set_all_positions(trossen_arm.VectorDouble(sleep_pose), blocking=True)
    current_pose = driver.get_all_positions()[:7]
    assert np.allclose(current_pose, sleep_pose, atol=test_tolerance), (
        f"❌🦾 current pose {current_pose} does not match sleep pose {sleep_pose}"
    )
    driver.set_all_modes(trossen_arm.Mode.idle)


if __name__ == "__main__":
    args = setup_log_with_config(TrossenConfig)
    # TODO: waiting on https://github.com/TrossenRobotics/trossen_arm/issues/86#issue-3144375498
    logging.getLogger("trossen_arm").setLevel(logging.ERROR)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args, log)
    log.info("🦾 Configuring left arm")
    configure_arm(args.arm_l_config_filepath, args.arm_l_ip, args.test_pose_name_l, args.test_tolerance)
    log.info("🦾 Configuring right arm")
    configure_arm(args.arm_r_config_filepath, args.arm_r_ip, args.test_pose_name_r, args.test_tolerance)
