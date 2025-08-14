#!/usr/bin/env python3

"""
Configure and home a Trossen Arm.

This script does the following:
1.  Prompts the user to confirm that they are ready to configure the arm with the specified model
    and IP address.
2.  Prompts the user to confirm that the gripper fingers have been removed from the carriages.
3.  Prompts the user to confirm that the arm is in its calibration jigs.
4.  Prompts the user to confirm that the gripper finger carriages are closed.
5.  Sets the home position of the arm to its current position.
6.  Creates and configures the arm driver with the specified model.
7.  Checks that the arm joint positions are all close to zero to confirm calibration.
8.  Checks that the gripper position is close to zero to confirm calibration.
9.  Prompts the user to confirm that the arm controller will be rebooted.
10. Cleans up and reboots the arm driver.
11. Prompts the user to remove the calibration jigs from the waist and wrist rotate motors.
12. Prompts the user to wait for the green light to turn on.
13. Reconfigures the arm driver.
14. Puts the arm into gravity compensation mode.
15. Prompts the user to confirm that the arm holds its position when moved.
16. Prompts the user to move the arm back to its sleep position.
"""

import argparse
import socket
from enum import Enum
from typing import Dict

import trossen_arm


class ArmModel(Enum):
    WXAI_V0 = "wxai_v0"


IP_ADDRESS_DEFAULT = "192.168.1.2"
PORT_DEFAULT = 50_001
INDICATOR_SET_HOME = b"\x01\x00\x01"
ARM_MODELS = (ArmModel.WXAI_V0,)
MAP_MODEL_TO_INT: Dict[ArmModel, trossen_arm.Model] = {
    ArmModel.WXAI_V0: trossen_arm.Model.wxai_v0,
}
ARM_MODEL_DEFAULT = ArmModel.WXAI_V0
POSITION_NEAR_ZERO_ARM = 0.005  # rad (~0.3deg)
POSITION_NEAR_ZERO_GRIPPER = 1e-5  # m


def run(args):
    arm_model_str: str = args.model
    arm_model = ArmModel(arm_model_str)
    ip_address = args.ip

    # Prompt the user to confirm that they are ready to configure this model arm
    print(f"Configuring and calibrating arm with model: '{arm_model_str}'.")
    print(f"Expected current IP address of the controller: '{ip_address}'")
    print("WARNING: This will calibrate the joint positions to their current positions.")
    print(
        "WARNING: This will factory reset the arm controller. "
        "It is recommended to back up the controller configuration before proceeding."
    )
    print("Press Enter to continue or Ctrl+C to cancel")
    input()

    # Prompt the user to confirm that the gripper fingers have been removed from the carriages
    print("Make sure the gripper fingers have been removed from the gripper carriages")
    print("Press Enter to confirm that the gripper fingers have been removed")
    input()

    # Prompt user to ready the arm for calibration
    print("Put the arm into its home position calibration jigs")
    print("Press Enter to confirm that the arm is in its calibration jigs")
    input()
    print("Close the gripper finger carriages")
    print("Press Enter to confirm that the gripper finger carriages are closed")
    input()

    # Set the home position first since this is not a feature available in the driver
    # We need to send a tcp message length header 1 and the set home command
    # The arm will respond with a message length header 1 and then the message 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((ip_address, PORT_DEFAULT))
        # Set a timeout for the recv call
        sock.settimeout(1.0)
        # Send the set home command
        sock.sendall(INDICATOR_SET_HOME)
        # Wait for the arm to respond
        result = b""
        while len(result) < 2:
            data = sock.recv(2 - len(result))
            if not data:
                print("ERROR: failed to receive data from arm")
                return
            result += data
        # Check that the arm responded with a success message
        if result[1] != 0:
            print("ERROR: arm did not respond with a success message")
            print(f"Arm response: {result}")
            return
    except Exception as e:
        print(f"ERROR: An exception occurred: {e}")
        return
    finally:
        sock.close()

    # Create and configure the arm driver
    model = MAP_MODEL_TO_INT.get(arm_model).value
    if model is None:
        raise ValueError(f"Invalid arm model: '{arm_model}'")
    arm = trossen_arm.TrossenArmDriver()
    arm.configure(
        model=trossen_arm.Model.wxai_v0,
        end_effector=trossen_arm.StandardEndEffector.wxai_v0_leader,
        serv_ip=ip_address,
        clear_error=False,  # Don't clear errors since we want to check the arm health
    )

    # Check that the joint positions are all close to zero
    arm_pos = arm.get_all_positions()[:-1]
    print(f"Joint positions: {arm_pos}rad")
    if any(abs(p) > POSITION_NEAR_ZERO_ARM for p in arm_pos):
        print("ERROR: Joint positions are not close to zero")
        print(f"Joint positions: {arm_pos}")
        print("Please move the arm to its home position and try again")
        arm.cleanup()
        return

    # Check that the gripper position is close to zero
    gripper_pos = arm.get_all_positions()[-1]
    print(f"Gripper position: {gripper_pos}m")
    if gripper_pos > POSITION_NEAR_ZERO_GRIPPER:
        print("ERROR: Gripper position is not close to zero")
        print(f"Gripper position: {gripper_pos}")
        print("Please move the arm to its home position and try again")
        arm.cleanup()
        return

    print("The arm controller will be rebooted.")
    print("Press Enter to continue or Ctrl+C to cancel")
    input()

    # Clean up and reboot the arm driver
    arm.cleanup(reboot_controller=True)

    print("Remove the calibration jigs from the waist and wrist rotate motors")
    print("Press Enter when jigs are removed")
    input()

    print("Wait for the green light to turn on")
    print("Press Enter when green light is on")
    input()

    # Reconfigure the arm driver
    arm.configure(
        model=trossen_arm.Model.wxai_v0,
        end_effector=trossen_arm.StandardEndEffector.wxai_v0_leader,
        serv_ip=IP_ADDRESS_DEFAULT,
        clear_error=False,  # Don't clear errors since we want to check the arm health
    )

    # Prompt the user to move the arm around
    print("Putting the arm into gravity compensation mode")
    print("!! If the arm was homed incorrectly, it may move unexpectedly !!")
    print("!! Make sure you and others are safely out of the way of the arm !!")
    print("!! Be prepared to turn off the control box if things go wrong !!")
    print("Press Enter when ready")
    input()

    # Check that gravity compensation mode works
    arm.set_all_modes(trossen_arm.Mode.external_effort)
    arm.set_all_external_efforts(
        [0.0] * arm.get_num_joints(),
        goal_time=0.0,
        blocking=False,
    )

    print("Confirm that the arm holds its position when moved")
    print("Press Enter once confirmed")
    input()

    print("Move the arm back to its sleep position")
    print("Press Enter when done")
    input()

    print("Arm finish tests complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the home position of a Trossen Arm")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=ARM_MODEL_DEFAULT,
        choices=ARM_MODELS,
        help=f"Model of the arm. If not specified, the default is {ARM_MODEL_DEFAULT}",
    )
    parser.add_argument(
        "-i",
        "--ip",
        type=str,
        default=IP_ADDRESS_DEFAULT,
        help=f"IP address of the arm controller. Default is {IP_ADDRESS_DEFAULT}",
    )
    args = parser.parse_args()
    run(args)
