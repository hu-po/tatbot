import os
from dataclasses import dataclass
import logging

import jax
import jax.numpy as jnp
from lerobot.common.teleoperators.config import TeleoperatorConfig
from lerobot.common.teleoperators.teleoperator import Teleoperator
import numpy as np
import pyroki as pk
import viser
from viser.extras import ViserUrdf
import yourdfpy
from typing import Dict, Any

from ik import IKConfig, ik

log = logging.getLogger('tatbot')

@dataclass
class ToolpathTeleopConfig(TeleoperatorConfig):
    toolpath_path: str = os.path.expanduser("~/tatbot/output/design/cat_toolpaths.json")
    """Local path to the toolpath file generated with design.py file."""

    seed: int = 42
    """Seed for random behavior."""
    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf")
    """Local path to the URDF file for the robot."""
    target_links_name: tuple[str, str] = ("left/tattoo_needle", "right/ee_gripper_link")
    """Names of the links to be controlled."""
    ik_config: IKConfig = IKConfig()
    """Configuration for the IK solver."""

    view_camera_position: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""
    env_map_hdri: str = "forest"
    """HDRI for the environment map."""

    joint_pos_design_l: tuple[float, float, float, float, float, float, float] = (0.10, 1.23, 1.01, -1.35, 0, 0, 0.02)
    """Joint positions of the left arm for robot hovering over design."""
    joint_pos_design_r: tuple[float, float, float, float, float, float, float] = (3.05, 0.49, 1.09, -1.52, 0, 0, 0.04)
    """Joint positions of the rgiht arm for robot hovering over design."""

    ee_design_pos: tuple[float, float, float] = (0.08, 0.0, 0.04)
    """position of the design ee transform."""
    ee_design_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the design ee transform."""

    ee_inkcap_pos: tuple[float, float, float] = (0.16, 0.0, 0.04)
    """position of the inkcap ee transform."""
    ee_inkcap_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the inkcap ee transform."""


class ToolpathTeleop(Teleoperator):
    config_class = ToolpathTeleopConfig
    name = "toolpath"

    def __init__(self, config: ToolpathTeleopConfig):
        super().__init__(config)
        self.config = config
        log.info(f"🌱 Setting random seed to {config.seed}...")
        self.rng = jax.random.PRNGKey(config.seed)

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
