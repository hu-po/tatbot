"""`python3 -m tatbot_description [--arms ARMS] [--hardware mock|fake|real] [-o FILE]`: print the URDF.

For check_urdf and inspection; the launch builds the same text in process.
"""
import argparse
import sys

from tatbot_description import drawing_arm, robot_description


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m tatbot_description")
    parser.add_argument("--repo", default=None, help="checkout holding urdf/ and config/ (default $TATBOT_REPO)")
    parser.add_argument("--arms", default=None, help="comma list (default: the drawing arm, stack.yaml arms)")
    parser.add_argument("--hardware", choices=("mock", "fake", "real"), default=None,
                        help="add <ros2_control> (default: kinematics only)")
    parser.add_argument("--registration", action="append", default=[], metavar="ARM=PATH",
                        help="arm-registration JSON placing <arm>/base_link in world")
    parser.add_argument("-o", "--output", default="-")
    ns = parser.parse_args(argv)
    registrations = dict(item.split("=", 1) for item in ns.registration)
    text = robot_description(ns.repo, arms=tuple((ns.arms or drawing_arm(ns.repo)).split(",")), hardware=ns.hardware,
                             registrations=registrations or None)
    if ns.output == "-":
        sys.stdout.write(text + "\n")
    else:
        with open(ns.output, "w") as stream:
            stream.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
