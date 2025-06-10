# https://github.com/TrossenRobotics/trossen_arm/blob/main/demos/python/configuration_in_yaml.py

import trossen_arm
from dataclasses import dataclass
import tyro

@dataclass
class CLIArgs:
    arm: str = "l"  # 'l' for left, 'r' for right
    """Which arm to configure ('l' or 'r')."""

def print_configurations(driver: trossen_arm.TrossenArmDriver):
    print("EEPROM factory reset flag:", driver.get_factory_reset_flag())
    print("EEPROM IP method:", driver.get_ip_method())
    print("EEPROM manual IP:", driver.get_manual_ip())
    print("EEPROM DNS:", driver.get_dns())
    print("EEPROM gateway:", driver.get_gateway())
    print("EEPROM subnet:", driver.get_subnet())
    print("EEPROM effort corrections:", driver.get_effort_corrections())
    print(
        "EEPROM friction transition velocities:",
        driver.get_friction_transition_velocities()
    )
    print(
        "EEPROM friction constant terms:",
        driver.get_friction_constant_terms()
    )
    print("EEPROM friction coulomb coefs:", driver.get_friction_coulomb_coefs())
    print("EEPROM friction viscous coefs:", driver.get_friction_viscous_coefs())
    print("Modes:", [mode.value for mode in driver.get_modes()])

    end_effector = driver.get_end_effector()
    print("End effector:")
    print("  palm:")
    print("    mass:", end_effector.palm.mass)
    print("    inertia:", end_effector.palm.inertia)
    print("    origin xyz:", end_effector.palm.origin_xyz)
    print("    origin rpy:", end_effector.palm.origin_rpy)
    print("  finger left:")
    print("    mass:", end_effector.finger_left.mass)
    print("    inertia:", end_effector.finger_left.inertia)
    print("    origin xyz:", end_effector.finger_left.origin_xyz)
    print("    origin rpy:", end_effector.finger_left.origin_rpy)
    print("  finger right:")
    print("    mass:", end_effector.finger_right.mass)
    print("    inertia:", end_effector.finger_right.inertia)
    print("    origin xyz:", end_effector.finger_right.origin_xyz)
    print("    origin rpy:", end_effector.finger_right.origin_rpy)
    print("  offset finger left:", end_effector.offset_finger_left)
    print("  offset finger right:", end_effector.offset_finger_right)
    print("  pitch circle radius:", end_effector.pitch_circle_radius)
    print("  t flange tool:", end_effector.t_flange_tool)

    joint_limits = driver.get_joint_limits()
    print("Joint limits:")
    for i, joint_limit in enumerate(joint_limits):
        print(f"  Joint {i}:")
        print("    position min:", joint_limit.position_min)
        print("    position max:", joint_limit.position_max)
        print("    position tolerance:", joint_limit.position_tolerance)
        print("    velocity max:", joint_limit.velocity_max)
        print("    velocity tolerance:", joint_limit.velocity_tolerance)
        print("    effort max:", joint_limit.effort_max)
        print("    effort tolerance:", joint_limit.effort_tolerance)

    motor_parameters = driver.get_motor_parameters()
    print("Motor parameters:")
    for i, motor_param in enumerate(motor_parameters):
        print(f"  Joint {i}:")
        for mode, param in motor_param.items():
            print(f"    Mode {mode.value}:")
            print("      Position loop:")
            print(
                f"        kp: {param.position.kp}, ki: {param.position.ki}, "
                f"kd: {param.position.kd}, imax: {param.position.imax}"
            )
            print("      Velocity loop:")
            print(
                f"        kp: {param.velocity.kp}, ki: {param.velocity.ki}, "
                f"kd: {param.velocity.kd}, imax: {param.velocity.imax}"
            )

    algorithm_parameter = driver.get_algorithm_parameter()
    print("Algorithm parameter:")
    print("  singularity threshold:", algorithm_parameter.singularity_threshold)

if __name__=='__main__':
    args = tyro.cli(CLIArgs)
    driver = trossen_arm.TrossenArmDriver()

    if args.arm == "l":
        print("Configuring left arm")
        ip = "192.168.1.3"
        ee = trossen_arm.StandardEndEffector.wxai_v0_base
        yaml_file = "config/trossen_arm_r.yaml"
    elif args.arm == "r":
        print("Configuring right arm")
        ip = "192.168.1.2"
        ee = trossen_arm.StandardEndEffector.wxai_v0_follower
        yaml_file = "config/trossen_arm_l.yaml"
    else:
        raise ValueError(f"Invalid arm: {args.arm}")

    driver.configure(trossen_arm.Model.wxai_v0, ee, ip, False)
    print_configurations(driver)
    driver.save_configs_to_file(yaml_file)
    driver.load_configs_from_file(yaml_file)
    print_configurations(driver)