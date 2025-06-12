"""
Records episodes using Vizer ik target controls.

> cd ~/tatbot/src/0.2
> git pull
> deactivate && rm -rf .venv && rm uv.lock
> uv venv && source .venv/bin/activate && uv pip install .
> DISPLAY=:0 uv run record_iktarget.py --debug

[esc] stop recording
[left arrow] rerecord the last episode
[right arrow] exit recording loop
"""

from dataclasses import asdict, dataclass
import logging
import os
from pprint import pformat
import time
from typing import Any, Dict

import jax.numpy as jnp
import lerobot.record
from lerobot.common.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.common.teleoperators.config import TeleoperatorConfig
from lerobot.common.teleoperators.teleoperator import Teleoperator
from lerobot.record import DatasetRecordConfig, RecordConfig
import numpy as np
import pyroki as pk
import tyro
import viser
from viser.extras import ViserUrdf
import yourdfpy

from ik import IKConfig, ik

log = logging.getLogger('tatbot')

@dataclass
class CLIArgs:
    debug: bool = False
    """Enable debug logging."""
    teleop: str = "iktarget"
    """Type of custom teleoperator to use, one of: iktarget, path"""
    dataset_name: str = f"test-{int(time.time())}"
    """Name of the dataset to record."""
    output_dir: str = os.path.expanduser("~/tatbot/output/record")
    """Directory to save the dataset."""
    episode_time_s: float = 60.0
    """Time of each episode."""
    num_episodes: int = 1
    """Number of episodes to record."""
    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub."""


@dataclass
class IKTargetTeleopConfig(TeleoperatorConfig):
    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf")
    """Local path to the URDF file for the robot."""
    target_links_name: tuple[str, str] = ("left/tattoo_needle", "right/ee_gripper_link")
    """Names of the links to be controlled."""
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
    ik_target_l_pos_init: tuple[float, float, float] = (0.08, 0.0, 0.04)
    """Initial position of the left IK target."""
    ik_target_l_ori_init: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """Initial orientation of the left IK target."""
    ik_target_r_pos_init: tuple[float, float, float] = (0.2, -0.2, 0.1)
    """Initial position of the right IK target."""
    ik_target_r_ori_init: tuple[float, float, float, float] = (0.7071, 0.0, 0.7071, 0.0)
    """Initial orientation of the right IK target."""


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
            position=config.ik_target_l_pos_init,
            wxyz=config.ik_target_l_ori_init,
            scale=config.transform_control_scale,
            opacity=config.transform_control_opacity,
        )
        self.ik_target_r = self.server.scene.add_transform_controls(
            "/ik_target_r",
            position=config.ik_target_r_pos_init,
            wxyz=config.ik_target_r_ori_init,
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
    args = tyro.cli(CLIArgs)
    if args.debug:
        log.setLevel(logging.DEBUG)
        # logging.getLogger('lerobot').setLevel(logging.DEBUG)
        log.debug("🐛 Debug mode enabled.")
    os.makedirs(args.output_dir, exist_ok=True)
    log.info(f"💾 Saving output to {args.output_dir}")
    log.info("🎮 Using IKTargetTeleop.")
    cfg = RecordConfig(
        robot=TatbotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=f"hu-po/tatbot-iktarget-{args.dataset_name}",
            single_task="Move using cartesian control",
            root=f"{args.output_dir}/{args.dataset_name}",
            fps=10,
            episode_time_s=args.episode_time_s,
            num_episodes=args.num_episodes,
            video=True,
            tags=["tatbot", "wxai", "trossen"],
            push_to_hub=args.push_to_hub,
        ),
        teleop=IKTargetTeleopConfig(),
        display_data=True,
        play_sounds=True,
        resume=False,
    )
    log.info(pformat(asdict(cfg)))
    lerobot.record.record(cfg)
