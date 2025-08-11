import functools
import os

import numpy as np
import pyroki as pk
import yourdfpy

from tatbot.data.pose import Pos, Pose, Rot
from tatbot.utils.log import get_logger

log = get_logger("bot.urdf", "🧱")


@functools.lru_cache(maxsize=1)
def load_robot(urdf_path: str) -> tuple[yourdfpy.URDF, pk.Robot]:
    log.debug(f"loading PyRoKi robot from yourdfpy URDF at {urdf_path}...")
    urdf_path = os.path.expanduser(urdf_path)
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    return urdf, robot


@functools.lru_cache(maxsize=4)
def get_link_indices(urdf_path: str, link_names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    log.debug(f"getting link indices for {link_names}")
    _, robot = load_robot(urdf_path)
    resolved_indices: list[int] = []
    resolved_names: list[str] = []
    for link_name in link_names:
        try:
            idx = robot.links.names.index(link_name)
            resolved_indices.append(idx)
            resolved_names.append(link_name)
        except ValueError:
            log.warning(f"URDF link not found: {link_name}")
            continue
    link_indices = np.asarray(resolved_indices, dtype=np.int32)
    log.debug(f"link indices: {link_indices}")
    return link_indices, tuple(resolved_names)


def get_link_poses(
    urdf_path: str,
    link_names: tuple[str, ...],
    joint_positions: np.ndarray,
) -> dict[str, Pose]:
    log.debug(f"getting link poses for {link_names}")
    _, robot = load_robot(urdf_path)
    # Resolve only valid link indices/names
    link_indices, resolved_names = get_link_indices(urdf_path, link_names)
    if link_indices.size == 0:
        log.warning("No valid URDF links provided; returning empty poses map")
        return {}
    # Ensure correct dtype and memory layout for FK computation
    joint_positions = np.asarray(joint_positions, dtype=np.float64).copy()
    all_link_poses = np.asarray(robot.forward_kinematics(joint_positions))
    # Copy out to avoid referencing underlying C buffers
    wxyz = np.ascontiguousarray(all_link_poses[link_indices, :4]).copy()
    pos = np.ascontiguousarray(all_link_poses[link_indices, 4:]).copy()
    link_poses = {
        link_name: Pose(pos=Pos(xyz=pos[i].astype(np.float32)), rot=Rot(wxyz=wxyz[i].astype(np.float32)))
        for i, link_name in enumerate(resolved_names)
    }
    log.debug(f"link poses: {link_poses}")
    return link_poses
