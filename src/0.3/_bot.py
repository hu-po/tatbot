from dataclasses import dataclass, field
import functools
import os
from typing import Callable, Any
import time

import numpy as np
import pyroki as pk
import yourdfpy

from _log import get_logger

log = get_logger('_bot')

@dataclass
class BotConfig:
    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf")
    """Local path to the URDF file for the robot."""
    target_link_names: tuple[str, str] = ("left/tattoo_needle", "right/tattoo_needle")
    """Names of the ee links in the URDF for left and right ik solving."""
    rest_pose: list[float] = field(default_factory=lambda: [
        # left arm
        -0.8, 0.6, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0,
        # right arm
        0.8, 0.6, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0])
    """Rest pose for the robot."""

@functools.lru_cache(maxsize=2)
def load_robot(urdf_path: str, target_links_name: tuple[str, str]) -> tuple[pk.Robot, np.ndarray]:
    log.debug(f"🤖 Loading PyRoKi robot from URDF at {urdf_path}...")
    start_time = time.time()
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    ee_link_indices = np.array([
        robot.links.names.index(target_links_name[0]), # left arm
        robot.links.names.index(target_links_name[1]), # right arm
    ], dtype=np.int32)
    log.debug(f"🤖 load robot time: {time.time() - start_time:.4f}s")
    return robot, ee_link_indices

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
    log.debug(f"🤖 Action: {_action}")
    return _action

def safe_loop(loop: Callable, config: Any) -> None:
    from lerobot.common.robots import make_robot_from_config
    from lerobot.common.robots.tatbot.config_tatbot import TatbotConfig

    try:
        loop(config)
    except Exception as e:
        log.error(f"Error: {e}")
    except KeyboardInterrupt:
        log.info("🤖🛑⌨️ Keyboard interrupt detected. Disconnecting robot...")
    finally:
        log.info("🤖🛑 Disconnecting robot...")
        robot = make_robot_from_config(TatbotConfig())
        robot._connect_l(clear_error=False)
        log.error(robot._get_error_str_l())
        robot._connect_r(clear_error=False)
        log.error(robot._get_error_str_r())
        robot.disconnect()