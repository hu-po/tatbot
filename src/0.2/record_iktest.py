"""
Records episodes using Vizer ik target controls.
"""

from dataclasses import asdict, dataclass
import logging
import os
from pprint import pformat
import time
from typing import Any, Dict

import jax.numpy as jnp
import lerobot.record
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.common.teleoperators.config import TeleoperatorConfig
from lerobot.common.teleoperators.teleoperator import Teleoperator
from lerobot.record import DatasetRecordConfig, RecordConfig
import numpy as np
import pyroki as pk
import viser
from viser.extras import ViserUrdf
import yourdfpy

from ik import IKConfig, ik
from log import setup_log_with_config, get_logger, print_config
from robot import robot_safe_loop

log = get_logger('record_iktest')

@dataclass
class RecordIKTestConfig:
    debug: bool = False
    """Enable debug logging."""

    hf_username: str = os.environ.get("HF_USER", "hu-po")
    """Hugging Face username."""
    dataset_name: str | None = None
    """Dataset will be saved to Hugging Face Hub repository ID, e.g. 'hf_username/dataset_name'."""
    display_data: bool = False
    """Display data on screen using Rerun."""
    output_dir: str = os.path.expanduser("~/tatbot/output/record")
    """Directory to save the dataset."""
    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub."""
    tags: tuple[str, ...] = ("tatbot", "wxai", "trossen")
    """Tags to add to the dataset on Hugging Face."""
    episode_time_s: float = 240.0
    """Time of each episode."""
    num_episodes: int = 1
    """Number of episodes to record."""

    robot_goal_time_fast: float = 2.0
    """Goal time for the robot when moving fast."""
    robot_goal_time_slow: float = 3.0
    """Goal time for the robot when moving slowly."""
    robot_block_mode: str = "left"
    """Block mode for the robot. One of: left, right, both."""


@dataclass
class IKTargetTeleopConfig(TeleoperatorConfig):
    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf")
    """Local path to the URDF file for the robot."""
    target_links_name: tuple[str, str] = ("left/tattoo_needle", "right/ee_gripper_link")
    """Names of the ee links in the URDF for left and right ik solving."""
    ik_config: IKConfig = IKConfig()
    """Configuration for the IK solver."""

    transform_control_scale: float = 0.2
    """Scale of the transform control frames for visualization."""
    transform_control_opacity: float = 0.2
    """Opacity of the transform control frames for visualization."""
    view_camera_position: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""
    env_map_hdri: str = "forest"
    """HDRI for the environment map."""
    
    ee_ik_target_l_pos_init: tuple[float, float, float] = (0.08, 0.0, 0.04)
    """Initial position of the left end effector IK target."""
    ee_ik_target_l_wxyz_init: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """Initial orientation quaternion (wxyz) of the left end effector IK target."""
    ee_ik_target_r_pos_init: tuple[float, float, float] = (0.08, -0.16, 0.16)
    """Initial position of the right end effector IK target."""
    ee_ik_target_r_wxyz_init: tuple[float, float, float, float] = (0.67360666, -0.25201478, 0.24747439, 0.64922119)
    """Initial orientation quaternion (wxyz) of the right end effector IK target."""


class IKTargetTeleop(Teleoperator):
    config_class = IKTargetTeleopConfig
    name = "iktarget"

    def __init__(self, config: IKTargetTeleopConfig):
        super().__init__(config)
        self.config = config

        log.info("🚀 Starting viser server...")
        self.server: viser.ViserServer = viser.ViserServer()
        self.server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = config.view_camera_position
            client.camera.look_at = config.view_camera_look_at

        log.info("🎯 Adding ik targets...")
        self.ik_target_l = self.server.scene.add_transform_controls(
            "/ik_target_l",
            position=config.ee_ik_target_l_pos_init,
            wxyz=config.ee_ik_target_l_wxyz_init,
            scale=config.transform_control_scale,
            opacity=config.transform_control_opacity,
        )
        self.ik_target_r = self.server.scene.add_transform_controls(
            "/ik_target_r",
            position=config.ee_ik_target_r_pos_init,
            wxyz=config.ee_ik_target_r_wxyz_init,
            scale=config.transform_control_scale,
            opacity=config.transform_control_opacity,
        )

        log.info("🦾 Adding robots...")
        urdf : yourdfpy.URDF = yourdfpy.URDF.load(self.config.urdf_path)
        self.robot: pk.Robot = pk.Robot.from_urdf(urdf)
        self.urdf_vis = ViserUrdf(self.server, urdf, root_node_name="/root")
            
    @property
    def action_features(self) -> Dict[str, Any]:
        action_features = {}
        for side in ["left", "right"]:
            for i in range(6):
                action_features[f"{side}.joint_{i}.pos"] = float
            action_features[f"{side}.gripper.pos"] = float
        return action_features

    @property
    def feedback_features(self) -> Dict[str, Any]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.server is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True):
        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self):
        log.debug(f"🎯 Left arm IK target - pos: {self.ik_target_l.position}, wxyz: {self.ik_target_l.wxyz}")
        log.debug(f"🎯 Right arm IK target - pos: {self.ik_target_r.position}, wxyz: {self.ik_target_r.wxyz}")
        target_link_indices = jnp.array([
            self.robot.links.names.index(self.config.target_links_name[0]),
            self.robot.links.names.index(self.config.target_links_name[1])
        ])
        solution = ik(
            robot=self.robot,
            target_link_indices=target_link_indices,
            target_wxyz=jnp.array([self.ik_target_l.wxyz, self.ik_target_r.wxyz]),
            target_position=jnp.array([self.ik_target_l.position, self.ik_target_r.position]),
            config=self.config.ik_config,
        )
        self.urdf_vis.update_cfg(np.array(solution))
        action = {
            "left.joint_0.pos": solution[0],
            "left.joint_1.pos": solution[1],
            "left.joint_2.pos": solution[2],
            "left.joint_3.pos": solution[3],
            "left.joint_4.pos": solution[4],
            "left.joint_5.pos": solution[5],
            "left.gripper.pos": solution[6],
            "right.joint_0.pos": solution[8],
            "right.joint_1.pos": solution[9],
            "right.joint_2.pos": solution[10],
            "right.joint_3.pos": solution[11],
            "right.joint_4.pos": solution[12],
            "right.joint_5.pos": solution[13],
            "right.gripper.pos": solution[14],
        }
        log.debug(f"🦾 Action: {action}")
        return action

    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        pass

    def disconnect(self):
        pass


# HACK: monkeypatch custom teleoperators into lerobot record types
original_make_teleoperator_from_config = lerobot.record.make_teleoperator_from_config

def make_teleoperator_from_config(config: TeleoperatorConfig):
    if isinstance(config, IKTargetTeleopConfig):
        return IKTargetTeleop(config)
    return original_make_teleoperator_from_config(config)


lerobot.record.make_teleoperator_from_config = make_teleoperator_from_config

if __name__ == "__main__":
    logging.getLogger('trossen_arm').setLevel(logging.ERROR)
    logging.getLogger('lerobot').setLevel(logging.DEBUG)
    args = setup_log_with_config(RecordIKTestConfig)
    dataset_name = args.dataset_name or f"iktest-{int(time.time())}"
    repo_id = f"{args.hf_username}/{dataset_name}"
    log.info("🎮 Using IKTargetTeleop.")
    cfg = RecordConfig(
        robot=TatbotConfig(
            goal_time_slow=args.robot_goal_time_slow,
            goal_time_fast=args.robot_goal_time_fast,
            block_mode=args.robot_block_mode,
        ),
        dataset=DatasetRecordConfig(
            repo_id=repo_id,
            # TODO: task done per episode using direction vector to natural language
            single_task="Move using cartesian control",
            root=f"{args.output_dir}/{dataset_name}",
            fps=30,
            episode_time_s=args.episode_time_s,
            num_episodes=args.num_episodes,
            video=True,
            tags=list(args.tags),
            push_to_hub=args.push_to_hub,
        ),
        teleop=IKTargetTeleopConfig(),
        display_data=True,
        play_sounds=True,
        resume=False,
    )
    print_config(cfg)
    robot_safe_loop(lambda: lerobot.record.record(cfg), cfg)