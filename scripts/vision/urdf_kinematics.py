#!/usr/bin/env python3
"""Minimal URDF forward kinematics (stdlib + numpy only).

Enough to answer "where is link L in the robot's base frame when the joints
are at q?" — which is what ties the camera rig to the robot: the arm's
encoders and its URDF are the one metric reference in the scene that was not
guessed.

Deliberately supports only what the tatbot URDF uses: fixed and revolute
(and continuous/prismatic) joints in a tree.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def driver_joint_names(prefix, count=7):
    """WXAI driver vector order: six arm joints, then left carriage metres."""
    if not 1 <= count <= 7:
        raise ValueError(f"unsupported WXAI joint vector length {count}")
    names = [f"{prefix}/joint_{i}" for i in range(min(count, 6))]
    if count == 7:
        names.append(f"{prefix}/left_carriage_joint")
    return names


def _rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle_to_matrix(axis, angle):
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    cross = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    return (np.eye(3) + np.sin(angle) * cross
            + (1 - np.cos(angle)) * (cross @ cross))


class UrdfChain:
    """Joint tree with forward kinematics by link name."""

    def __init__(self, urdf_path):
        root = ET.parse(Path(urdf_path).expanduser()).getroot()
        self.joints = {}
        self.parent_of = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            child_elem = joint.find("child")
            parent_elem = joint.find("parent")
            child = child_elem.get("link") if child_elem is not None else None
            parent = parent_elem.get("link") if parent_elem is not None else None
            if not name or not child or not parent:
                continue
            origin = joint.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                xyz_attr = origin.get("xyz")
                if xyz_attr:
                    xyz = np.array([float(v) for v in xyz_attr.split()])
                rpy_attr = origin.get("rpy")
                if rpy_attr:
                    rpy = np.array([float(v) for v in rpy_attr.split()])
            axis_element = joint.find("axis")
            axis_attr = axis_element.get("xyz") if axis_element is not None else None
            axis = ([float(v) for v in axis_attr.split()]
                    if axis_attr else [1.0, 0.0, 0.0])
            transform = np.eye(4)
            transform[:3, :3] = _rpy_to_matrix(*rpy)
            transform[:3, 3] = xyz
            self.joints[name] = {
                "name": name, "type": joint.get("type"), "parent": parent,
                "child": child, "origin": transform, "axis": np.asarray(axis, float),
            }
            self.parent_of[child] = name

    def link_pose(self, link, joint_values=None):
        """Pose of `link` in the root frame, given a {joint_name: value} map."""
        joint_values = joint_values or {}
        pose = np.eye(4)
        chain = []
        current = link
        seen = set()
        while current in self.parent_of:
            if current in seen:
                raise ValueError(f"cycle in URDF at {current}")
            seen.add(current)
            joint = self.joints[self.parent_of[current]]
            chain.append(joint)
            current = joint["parent"]
        for joint in reversed(chain):
            pose = pose @ joint["origin"]
            if joint["type"] in ("revolute", "continuous"):
                angle = float(joint_values.get(joint["name"], 0.0))
                motion = np.eye(4)
                motion[:3, :3] = _axis_angle_to_matrix(joint["axis"], angle)
                pose = pose @ motion
            elif joint["type"] == "prismatic":
                offset = float(joint_values.get(joint["name"], 0.0))
                motion = np.eye(4)
                motion[:3, 3] = joint["axis"] * offset
                pose = pose @ motion
            # fixed joints contribute only their origin
        return pose

    def arm_joint_names(self, prefix, count=6):
        """The `prefix/joint_N` names an arm's encoder vector maps onto."""
        return [f"{prefix}/joint_{i}" for i in range(count)]

    def driver_joint_names(self, prefix, count=7):
        """All driver joints, including the prismatic gripper carriage."""
        return driver_joint_names(prefix, count)


if __name__ == "__main__":
    import sys
    chain = UrdfChain(sys.argv[1] if len(sys.argv) > 1
                      else Path(__file__).resolve().parents[2] / "urdf/tatbot.urdf")
    zeros = {}
    for link in ("right/link_6", "right/realsense_link", "palette_tag8", "palette_root"):
        try:
            pose = chain.link_pose(link, zeros)
            print(f"{link:32s} xyz={np.round(pose[:3, 3], 4)}")
        except (KeyError, ValueError) as error:
            print(f"{link:32s} unavailable: {error}")
