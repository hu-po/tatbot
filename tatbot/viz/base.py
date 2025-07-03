import logging
import time
from dataclasses import dataclass

import numpy as np
import viser
from viser.extras import ViserUrdf

from tatbot.bot.urdf import get_link_poses, load_robot
from tatbot.data.ink import InkPalette
from tatbot.data.pose import ArmPose, make_bimanual_joints
from tatbot.data.skin import Skin
from tatbot.data.urdf import URDF
from tatbot.data.cam import CameraConfig
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
    skin_name: str = "default"
    """Name of the skin (Skin)."""
    camera_config_name: str = "default"
    """Name of the camera config (CameraConfig)."""

    env_map_hdri: str = "forest"
    """HDRI for the environment map."""
    view_camera_position: tuple[float, float, float] = (0.3, 0.3, 0.3)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""

    realsense_frustrum_scale: float = 0.02
    """Scale of the realsense camera frustrums used for visualization."""
    realsense_frustrum_color: tuple[int, int, int] = (200, 200, 200)
    """Color of the realsense camera frustrums used for visualization."""

    camera_frustrum_scale: float = 0.04
    """Scale of the ip camera frustrum used for visualization."""
    camera_frustrum_color: tuple[int, int, int] = (200, 200, 200)
    """Color of the ip camera frustrum used for visualization."""

    speed: float = 1.0
    """Speed multipler for visualization."""

class BaseViz:
    def __init__(self, config: BaseVizConfig):
        self.config = config
        self.urdf: URDF = URDF.from_name(config.urdf_name)
        self.left_arm_pose: ArmPose = ArmPose.from_name(config.left_arm_pose_name)
        self.right_arm_pose: ArmPose = ArmPose.from_name(config.right_arm_pose_name)
        self.rest_pose: np.ndarray = make_bimanual_joints(self.left_arm_pose, self.right_arm_pose)
        self.ink_palette: InkPalette = InkPalette.from_name(config.ink_palette_name)
        self.skin: Skin = Skin.from_name(config.skin_name)
        self.camera_config: CameraConfig = CameraConfig.from_name(config.camera_config_name)

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

        log.info("Adding realsense camera frustrums ...")
        self.realsense1_frustrum = self.server.scene.add_camera_frustum(
            f"/realsense1",
            fov=self.camera_config.intrinsics["realsense1"].fov,
            aspect=self.camera_config.intrinsics["realsense1"].aspect,
            scale=config.realsense_frustrum_scale,
            color=config.realsense_frustrum_color,
        )
        self.realsense2_frustrum = self.server.scene.add_camera_frustum(
            f"/realsense2",
            fov=self.camera_config.intrinsics["realsense2"].fov,
            aspect=self.camera_config.intrinsics["realsense2"].aspect,
            scale=config.realsense_frustrum_scale,
            color=config.realsense_frustrum_color,
        )

        log.info("Adding ip camera frustrums ...")
        self.camera1_frustrum = self.server.scene.add_camera_frustum(
            f"/camera1",
            fov=self.camera_config.intrinsics["camera1"].fov,
            aspect=self.camera_config.intrinsics["camera1"].aspect,
            scale=config.camera_frustrum_scale,
            color=config.camera_frustrum_color,
        )
        self.camera2_frustrum = self.server.scene.add_camera_frustum(
            f"/camera2",
            fov=self.camera_config.intrinsics["camera2"].fov,
            aspect=self.camera_config.intrinsics["camera2"].aspect,
            scale=config.camera_frustrum_scale,
            color=config.camera_frustrum_color,
        )
        self.camera3_frustrum = self.server.scene.add_camera_frustum(
            f"/camera3",
            fov=self.camera_config.intrinsics["camera3"].fov,
            aspect=self.camera_config.intrinsics["camera3"].aspect,
            scale=config.camera_frustrum_scale,
            color=config.camera_frustrum_color,
        )
        self.camera4_frustrum = self.server.scene.add_camera_frustum(
            f"/camera4",
            fov=self.camera_config.intrinsics["camera4"].fov,
            aspect=self.camera_config.intrinsics["camera4"].aspect,
            scale=config.camera_frustrum_scale,
            color=config.camera_frustrum_color,
        )
        self.camera5_frustrum = self.server.scene.add_camera_frustum(
            f"/camera5",
            fov=self.camera_config.intrinsics["camera5"].fov,
            aspect=self.camera_config.intrinsics["camera5"].aspect,
            scale=config.camera_frustrum_scale,
            color=config.camera_frustrum_color,
        )

        log.info("Positioning camera frustrums based on URDF ...")
        link_poses = get_link_poses(self.urdf.path, self.urdf.camera_link_names, self.rest_pose)
        realsense1_pose = link_poses[self.camera_config.realsense1_urdf_link_name]
        self.realsense1_frustrum.position = realsense1_pose[:3]
        self.realsense1_frustrum.wxyz = realsense1_pose[3:]
        realsense2_pose = link_poses[self.camera_config.realsense2_urdf_link_name]
        self.realsense2_frustrum.position = realsense2_pose[:3]
        self.realsense2_frustrum.wxyz = realsense2_pose[3:]
        camera1_pose = link_poses[self.camera_config.camera1_urdf_link_name]
        self.camera1_frustrum.position = camera1_pose[:3]
        self.camera1_frustrum.wxyz = camera1_pose[3:]
        camera2_pose = link_poses[self.camera_config.camera2_urdf_link_name]
        self.camera2_frustrum.position = camera2_pose[:3]
        self.camera2_frustrum.wxyz = camera2_pose[3:]
        camera3_pose = link_poses[self.camera_config.camera3_urdf_link_name]
        self.camera3_frustrum.position = camera3_pose[:3]
        self.camera3_frustrum.wxyz = camera3_pose[3:]
        camera4_pose = link_poses[self.camera_config.camera4_urdf_link_name]
        self.camera4_frustrum.position = camera4_pose[:3]
        self.camera4_frustrum.wxyz = camera4_pose[3:]
        camera5_pose = link_poses[self.camera_config.camera5_urdf_link_name]
        self.camera5_frustrum.position = camera5_pose[:3]
        self.camera5_frustrum.wxyz = camera5_pose[3:]

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