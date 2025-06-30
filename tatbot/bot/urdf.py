import functools
import os

import numpy as np
import pyroki as pk
import yourdfpy

from tatbot.data.pose import Pose, Pos, Rot
from tatbot.utils.log import get_logger

log = get_logger('bot.urdf', '🧱')


@functools.lru_cache(maxsize=1)
def load_robot(urdf_path: str) -> tuple[yourdfpy.URDF, pk.Robot]:
    log.info(f"loading PyRoKi robot from yourdfpy URDF at {urdf_path}...")
    urdf_path = os.path.expanduser(urdf_path)
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    return urdf, robot

@functools.lru_cache(maxsize=4)
def get_link_indices(urdf_path: str, link_names: tuple[str, ...]) -> np.ndarray:
    log.info(f"getting link indices for {link_names}")
    _, robot = load_robot(urdf_path)
    link_indices = np.array([robot.links.names.index(link_name) for link_name in link_names], dtype=np.int32)
    log.debug(f"link indices: {link_indices}")
    return link_indices

def get_link_poses(
    urdf_path: str,
    link_names: tuple[str, ...],
    joint_positions: np.ndarray,
) -> dict[str, Pose]:
    log.info(f"getting link poses for {link_names}")
    _, robot = load_robot(urdf_path)
    link_indices = get_link_indices(urdf_path, link_names)
    all_link_poses = robot.forward_kinematics(joint_positions)
    pos = all_link_poses[link_indices, :3]
    wxyz = all_link_poses[link_indices, 3:]
    link_poses = {
        link_name: Pose(pos=Pos(xyz=pos[i]), rot=Rot(wxyz=wxyz[i]))
        for i, link_name in enumerate(link_names)
    }
    log.debug(f"link poses: {link_poses}")
    return link_poses

def urdf_joints_to_action(urdf_joints: list[float]) -> dict[str, float]:
    _action = {
        "left.joint_0.pos": urdf_joints[0],
        "left.joint_1.pos": urdf_joints[1],
        "left.joint_2.pos": urdf_joints[2],
        "left.joint_3.pos": urdf_joints[3],
        "left.joint_4.pos": urdf_joints[4],
        "left.joint_5.pos": urdf_joints[5],
        "left.gripper.pos": urdf_joints[6],
        "right.joint_0.pos": urdf_joints[8],
        "right.joint_1.pos": urdf_joints[9],
        "right.joint_2.pos": urdf_joints[10],
        "right.joint_3.pos": urdf_joints[11],
        "right.joint_4.pos": urdf_joints[12],
        "right.joint_5.pos": urdf_joints[13],
        "right.gripper.pos": urdf_joints[14],
    }
    log.debug(f"Action: {_action}")
    return _action