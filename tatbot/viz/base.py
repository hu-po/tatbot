import logging
import time
from dataclasses import dataclass

import numpy as np
import viser
from viser.extras import ViserUrdf

from tatbot.bot.urdf import get_link_poses, load_robot
from tatbot.data.ink import InkPalette
from tatbot.data.pose import ArmPose
from tatbot.data.urdf import URDF
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger('viz.base', '🖥️')

@dataclass
class BaseVizConfig:
    debug: bool = False
    """Enable debug logging."""

    urdf_name: str = "default"
    """Name of the urdf (URDF)."""
    ink_palette_name: str = "default"
    """Name of the ink palette (InkPalette)."""
    left_arm_pose_name: str = "left/rest"
    """Name of the left arm pose (ArmPose)."""
    right_arm_pose_name: str = "right/rest"
    """Name of the right arm pose (ArmPose)."""

    env_map_hdri: str = "forest"
    """HDRI for the environment map."""
    view_camera_position: tuple[float, float, float] = (0.3, 0.3, 0.3)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""

    speed: float = 1.0
    """Speed multipler for visualization."""

class BaseViz:
    def __init__(self, config: BaseVizConfig):
        self.config = config

        self.urdf: URDF = URDF.from_name(config.urdf_name)
        log.info(f"✅ Loaded URDF: {self.urdf}")
        log.debug(f"URDF: {self.urdf}")
        self.left_arm_pose: ArmPose = ArmPose.from_name(config.left_arm_pose_name)
        log.info("✅ Loaded left arm pose")
        log.debug(f"Left arm pose: {self.left_arm_pose}")
        self.right_arm_pose: ArmPose = ArmPose.from_name(config.right_arm_pose_name)
        log.info("✅ Loaded right arm pose")
        log.debug(f"Right arm pose: {self.right_arm_pose}")
        self.rest_pose: np.ndarray = np.concatenate([self.left_arm_pose.joints, self.right_arm_pose.joints])
        self.ink_palette: InkPalette = InkPalette.from_name(config.ink_palette_name)
        log.info(f"✅ Loaded ink palette: {self.ink_palette}")
        log.debug(f"Ink palette: {self.ink_palette}")

        log.info("Starting viser server")
        self.server: viser.ViserServer = viser.ViserServer()
        self.server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = config.view_camera_position
            client.camera.look_at = config.view_camera_look_at

        self.step_sleep = 1.0 / 30.0 # 30 fps
        self.speed_slider = self.server.gui.add_slider(
            "speed",
            min=0.1,
            max=100.0,
            step=0.1,
            initial_value=self.config.speed,
        )

        log.debug("Adding robot to viser from URDF")
        _urdf, self.robot = load_robot(self.urdf.path)
        self.viser_urdf = ViserUrdf(self.server, _urdf, root_node_name="/root")
        self.joints = self.rest_pose.copy()
        self.robot_at_rest: bool = True
    
        log.debug("Adding inkpalette to viser")
        link_poses = get_link_poses(self.urdf.path, self.urdf.ink_link_names, self.rest_pose)
        for inkcap in self.ink_palette.inkcaps:
            self.server.scene.add_icosphere(
                name=f"/inkcaps/{inkcap.name}",
                radius=inkcap.diameter_m / 2,
                color=inkcap.ink.rgb,
                position=tuple(link_poses[inkcap.name].pos.xyz),
                subdivisions=4,
                visible=True,
            )

    def step(self):
        log.info("Empty step function, implement in subclass")
        pass

    def run(self):
        while True:
            start_time = time.time()
            if self.viser_urdf is not None:
                log.debug("Updating viser robot")
                self.viser_urdf.update_cfg(self.joints)
            self.step()
            log.debug(f"Step time: {time.time() - start_time:.4f}s")
            time.sleep(self.step_sleep / self.speed_slider.value)

if __name__ == "__main__":
    args = setup_log_with_config(BaseVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    viz = BaseViz(args)
    viz.run()