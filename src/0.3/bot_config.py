"""
Configure the robot via YAML, originally from:

https://github.com/TrossenRobotics/trossen_arm/blob/main/demos/python/configuration_in_yaml.py

First update the firmware to the latest version:

https://docs.trossenrobotics.com/trossen_arm/main/getting_started/software_setup.html#software-upgrade

Download latest firmware:

https://docs.trossenrobotics.com/trossen_arm/main/downloads.html

> unzip firmware-wxai_v0.zip
> teensy_loader_cli --mcu=TEENSY41 -s firmware-wxai_v0.hex

TODO: Set the velocity_tolerance to 0.2 times the velocity max
https://docs.trossenrobotics.com/trossen_arm/main/getting_started/configuration.html#joint-limits

TODO: Edit end effector parameters for left arm (tattoo needle arm):
https://docs.trossenrobotics.com/trossen_arm/main/api/structtrossen__arm_1_1EndEffector.html#struct-documentation
TODO: Create trossen_arm.StandardEndEffector.wxai_v0_tatbot_l and trossen_arm.StandardEndEffector.wxai_v0_tatbot_r
"""

from dataclasses import dataclass
import logging
import os

import trossen_arm

from _log import setup_log_with_config, get_logger, print_config

log = get_logger('bot_config')

@dataclass
class TrossenConfig:
    debug: bool = False
    """Enable debug logging."""
    arm_l_ip: str = "192.168.1.3"
    """IP address of the left arm."""
    arm_l_config_filepath: str = os.path.expanduser("~/tatbot/config/trossen_arm_l.yaml")
    """YAML file containing left arm config."""
    arm_r_ip: str = "192.168.1.2"
    """IP address of the right arm."""
    arm_r_config_filepath: str = os.path.expanduser("~/tatbot/config/trossen_arm_r.yaml")
    """YAML file containing right arm config."""

def print_configurations(driver: trossen_arm.TrossenArmDriver):
    log.debug("EEPROM factory reset flag:", driver.get_factory_reset_flag())
    log.debug("EEPROM IP method:", driver.get_ip_method())
    log.debug("EEPROM manual IP:", driver.get_manual_ip())
    log.debug("EEPROM DNS:", driver.get_dns())
    log.debug("EEPROM gateway:", driver.get_gateway())
    log.debug("EEPROM subnet:", driver.get_subnet())
    log.debug("EEPROM effort corrections:", driver.get_effort_corrections())
    log.debug(
        "EEPROM friction transition velocities:",
        driver.get_friction_transition_velocities()
    )
    log.debug(
        "EEPROM friction constant terms:",
        driver.get_friction_constant_terms()
    )
    log.debug("EEPROM friction coulomb coefs:", driver.get_friction_coulomb_coefs())
    log.debug("EEPROM friction viscous coefs:", driver.get_friction_viscous_coefs())
    log.debug("Modes:", [mode.value for mode in driver.get_modes()])

    end_effector = driver.get_end_effector()
    log.debug("End effector:")
    log.debug("  palm:")
    log.debug("    mass:", end_effector.palm.mass)
    log.debug("    inertia:", end_effector.palm.inertia)
    log.debug("    origin xyz:", end_effector.palm.origin_xyz)
    log.debug("    origin rpy:", end_effector.palm.origin_rpy)
    log.debug("  finger left:")
    log.debug("    mass:", end_effector.finger_left.mass)
    log.debug("    inertia:", end_effector.finger_left.inertia)
    log.debug("    origin xyz:", end_effector.finger_left.origin_xyz)
    log.debug("    origin rpy:", end_effector.finger_left.origin_rpy)
    log.debug("  finger right:")
    log.debug("    mass:", end_effector.finger_right.mass)
    log.debug("    inertia:", end_effector.finger_right.inertia)
    log.debug("    origin xyz:", end_effector.finger_right.origin_xyz)
    log.debug("    origin rpy:", end_effector.finger_right.origin_rpy)
    log.debug("  offset finger left:", end_effector.offset_finger_left)
    log.debug("  offset finger right:", end_effector.offset_finger_right)
    log.debug("  pitch circle radius:", end_effector.pitch_circle_radius)
    log.debug("  t flange tool:", end_effector.t_flange_tool)

    joint_limits = driver.get_joint_limits()
    log.debug("Joint limits:")
    for i, joint_limit in enumerate(joint_limits):
        log.debug(f"  Joint {i}:")
        log.debug("    position min:", joint_limit.position_min)
        log.debug("    position max:", joint_limit.position_max)
        log.debug("    position tolerance:", joint_limit.position_tolerance)
        log.debug("    velocity max:", joint_limit.velocity_max)
        log.debug("    velocity tolerance:", joint_limit.velocity_tolerance)
        log.debug("    effort max:", joint_limit.effort_max)
        log.debug("    effort tolerance:", joint_limit.effort_tolerance)

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
    log.debug("  singularity threshold:", algorithm_parameter.singularity_threshold)

def configure_arm(filepath: str, ip: str, ee: trossen_arm.StandardEndEffector):
    driver = trossen_arm.TrossenArmDriver(config=filepath)
    assert os.path.exists(filepath), f"❌📄 yaml file does not exist: {filepath}"
    driver.configure(
        trossen_arm.Model.wxai_v0, # model
        trossen_arm.StandardEndEffector.wxai_v0_base, # end_effector
        ip, # serv_ip
        False # clear_error
    )
    assert driver is not None, f"❌🦾 failed to connect to arm {ip}"
    print_configurations(driver)
    driver.save_configs_to_file(filepath)
    log.debug(f"✅🎛️📄 saved config to {filepath}")
    driver.load_configs_from_file(filepath)
    log.debug(f"✅🎛️📄 loaded config from {filepath}")
    print_configurations(driver)
    log.debug(f"✅🎛️🦾 arm {ip} configured successfully")


if __name__=='__main__':
    args = setup_log_with_config(TrossenConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    log.debug("🎛️🦾 Configuring left arm")
    configure_arm(args.arm_l_config_filepath, args.arm_l_ip, args.arm_l_ee)
    log.debug("🎛️🦾 Configuring right arm")
    configure_arm(args.arm_r_config_filepath, args.arm_r_ip, args.arm_r_ee)