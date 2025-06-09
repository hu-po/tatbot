from dataclasses import asdict, dataclass
import os
import logging
import time
from pprint import pformat

import jax
import jax.numpy as jnp
import numpy as np
import pyroki as pk
import viser
from viser.extras import ViserUrdf
import yourdfpy
from typing import Dict, Any
import tyro

from ik import IKConfig, ik

log = logging.getLogger('tatbot')

@dataclass
class ToolpathConfig:

    debug: bool = False
    """Enable debug logging."""
    dataset_name: str = f"toolpath-test-{int(time.time())}"
    """Name of the dataset to record."""
    output_dir: str = os.path.expanduser("~/tatbot/output/record")
    """Directory to save the dataset."""
    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub."""

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


def main(config: ToolpathConfig):
    config = config
    log.info(f"🌱 Setting random seed to {config.seed}...")
    rng = jax.random.PRNGKey(config.seed)

    log.info("🚀 Starting viser server...")
    server: viser.ViserServer = viser.ViserServer()
    server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = config.view_camera_position
        client.camera.look_at = config.view_camera_look_at

    log.info("🦾 Adding robots...")
    urdf : yourdfpy.URDF = yourdfpy.URDF.load(config.urdf_path)
    robot: pk.Robot = pk.Robot.from_urdf(urdf)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/root")

    while True:
        target_link_indices = jnp.array([
            robot.links.names.index(config.target_links_name[0]),
            robot.links.names.index(config.target_links_name[1])
        ])
        solution = ik(
            robot=robot,
            target_link_indices=target_link_indices,
            target_wxyz=jnp.array([ik_target_l.wxyz, ik_target_r.wxyz]),
            target_position=jnp.array([ik_target_l.position, ik_target_r.position]),
            config=config.ik_config,
        )
        urdf_vis.update_cfg(np.array(solution))
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


if __name__ == "__main__":
    args = tyro.cli(ToolpathConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
        # logging.getLogger('lerobot').setLevel(logging.DEBUG)
        log.debug("🐛 Debug mode enabled.")
    os.makedirs(args.output_dir, exist_ok=True)
    log.info(f"💾 Saving output to {args.output_dir}")
    log.info(pformat(asdict(args)))
    main(args)