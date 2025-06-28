from dataclasses import asdict, dataclass, field
import functools
import os
import time
from typing import Any, Callable

import dacite
import numpy as np
import pyroki as pk
import yaml
import yourdfpy

from _log import get_logger

log = get_logger('_bot')

@dataclass
class BotConfig:
    urdf_path: str = "~/tatbot/assets/urdf/tatbot.urdf"
    """Local path to the URDF file for the robot."""
    target_link_names: tuple[str, str] = ("left/tattoo_needle", "right/tattoo_needle")
    """Names of the ee links in the URDF for left and right ik solving."""
    rest_pose: np.ndarray = field(default_factory=lambda: np.array([
        -1.0, 0.1, 0.5, -1.2, 0.0, 0.0, 0.0, 0.0, # left arm
        1.0, 0.1, 0.5, -1.2, 0.0, 0.0, 0.0, 0.0, # right arm
        ], dtype=np.float32))
    """Rest pose for the robot."""

    @classmethod
    def from_yaml(cls, filepath: str) -> "BotConfig":
        with open(os.path.expanduser(filepath), "r") as f:
            data = yaml.safe_load(f)
        if "target_link_names" in data and isinstance(data["target_link_names"], list):
            data["target_link_names"] = tuple(data["target_link_names"])
        if "rest_pose" in data and isinstance(data["rest_pose"], list):
            data["rest_pose"] = np.array(data["rest_pose"], dtype=np.float32)
        return dacite.from_dict(
            cls,
            data,
            config=dacite.Config(type_hooks={tuple: tuple, np.ndarray: np.array})
        )

    def save_yaml(self, filepath: str):
        with open(os.path.expanduser(filepath), "w") as f:
            yaml.safe_dump(asdict(self), f)


@functools.lru_cache(maxsize=2)
def load_robot(urdf_path: str) -> tuple[yourdfpy.URDF, pk.Robot]:
    log.debug(f"🤖 Loading PyRoKi robot from URDF at {urdf_path}...")
    start_time = time.time()
    urdf_path = os.path.expanduser(urdf_path)
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    log.debug(f"🤖 load robot time: {time.time() - start_time:.4f}s")
    return urdf, robot

@functools.lru_cache(maxsize=4)
def get_link_indices(link_names: tuple[str, ...], urdf_path: str) -> np.ndarray:
    _, robot = load_robot(urdf_path)
    return np.array([robot.links.names.index(link_name) for link_name in link_names], dtype=np.int32)

def get_link_poses(
    link_names: list[str],
    joint_positions: np.ndarray | None = None,
    bot_config: BotConfig = BotConfig(),
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    _, robot = load_robot(bot_config.urdf_path)
    link_indices = get_link_indices(tuple(link_names), bot_config.urdf_path)
    if joint_positions is None:
        joint_positions = bot_config.rest_pose
    log.debug("🤖 performing forward kinematics...")
    start_time = time.time()
    all_link_poses = robot.forward_kinematics(joint_positions)
    pos = all_link_poses[link_indices, :3]
    wxyz = all_link_poses[link_indices, 3:]
    log.debug(f"🤖 forward kinematics time: {time.time() - start_time:.4f}s")
    return {link_name: (pos, wxyz) for link_name in link_names}

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
    from lerobot.common.robots.tatbot.config_tatbot import TatbotBotOnlyConfig

    try:
        loop(config)
    except Exception as e:
        log.error(f"❌Error:\n{e}\n")
    except KeyboardInterrupt:
        log.info("🤖🛑⌨️ Keyboard interrupt detected. Disconnecting robot...")
    finally:
        log.info("🤖🛑 Disconnecting robot...")
        robot = make_robot_from_config(TatbotBotOnlyConfig)
        robot._connect_l(clear_error=False)
        log.error(robot._get_error_str_l())
        robot._connect_r(clear_error=False)
        log.error(robot._get_error_str_r())
        robot.disconnect()