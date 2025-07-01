import logging
import os
from dataclasses import dataclass

from tatbot.data.scan import Scan
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.viz.base import BaseViz, BaseVizConfig

log = get_logger('viz.plan', '🖥️')

@dataclass
class VizScanConfig(BaseVizConfig):
    scan_dir: str = "~/tatbot/nfs/record/scan-test"
    """Directory containing scan."""

    realsense_frustrum_scale: float = 0.02
    """Scale of the realsense camera frustrums used for visualization."""
    realsense_frustrum_color: tuple[int, int, int] = (200, 200, 200)
    """Color of the realsense camera frustrums used for visualization."""

    camera_frustrum_scale: float = 0.04
    """Scale of the ip camera frustrum used for visualization."""
    camera_frustrum_color: tuple[int, int, int] = (200, 200, 200)
    """Color of the ip camera frustrum used for visualization."""


class VizScan(BaseViz):
    def __init__(self, config: VizScanConfig):
        super().__init__(config)
        self.scan = Scan.from_yaml(config.scan_dir)
        self.bot_config: BotConfig = self.scan.bot_config
        self.ink_config: InkConfig = self.scan.ink_config

        log.info(f"🖥️ Adding realsense camera frustrums ...")
        self.realsense1_frustrum = self.server.scene.add_camera_frustum(
            f"/realsense1",
            fov=self.scan.intrinsics["realsense1"].fov,
            aspect=self.scan.intrinsics["realsense1"].aspect,
            scale=config.realsense_frustrum_scale,
            color=config.realsense_frustrum_color,
        )
        self.realsense2_frustrum = self.server.scene.add_camera_frustum(
            f"/realsense2",
            fov=self.scan.intrinsics["realsense2"].fov,
            aspect=self.scan.intrinsics["realsense2"].aspect,
            scale=config.realsense_frustrum_scale,
            color=config.realsense_frustrum_color,
        )

        log.info(f"🖥️ Adding ip camera frustrums ...")
        self.camera1_frustrum = self.server.scene.add_camera_frustum(
            f"/camera1",
            fov=self.scan.intrinsics["camera1"].fov,
            aspect=self.scan.intrinsics["camera1"].aspect,
            scale=config.camera_frustrum_scale,
            color=config.camera_frustrum_color,
        )
        self.camera2_frustrum = self.server.scene.add_camera_frustum(
            f"/camera2",
            fov=self.scan.intrinsics["camera2"].fov,
            aspect=self.scan.intrinsics["camera2"].aspect,
            scale=config.camera_frustrum_scale,
            color=config.camera_frustrum_color,
        )
        self.camera3_frustrum = self.server.scene.add_camera_frustum(
            f"/camera3",
            fov=self.scan.intrinsics["camera3"].fov,
            aspect=self.scan.intrinsics["camera3"].aspect,
            scale=config.camera_frustrum_scale,
            color=config.camera_frustrum_color,
        )
        self.camera4_frustrum = self.server.scene.add_camera_frustum(
            f"/camera4",
            fov=self.scan.intrinsics["camera4"].fov,
            aspect=self.scan.intrinsics["camera4"].aspect,
            scale=config.camera_frustrum_scale,
            color=config.camera_frustrum_color,
        )
        self.camera5_frustrum = self.server.scene.add_camera_frustum(
            f"/camera5",
            fov=self.scan.intrinsics["camera5"].fov,
            aspect=self.scan.intrinsics["camera5"].aspect,
            scale=config.camera_frustrum_scale,
            color=config.camera_frustrum_color,
        )

        log.info("🖥️ Positioning camera frustrums based on URDF ...")
        all_link_poses = fk(self.bot_config.rest_pose, self.bot_config)
        realsense1_pose = all_link_poses[self.robot.links.names.index(self.scan.realsense1_urdf_link_name)]
        self.realsense1_frustrum.position = realsense1_pose[:3]
        self.realsense1_frustrum.wxyz = realsense1_pose[3:]
        realsense2_pose = all_link_poses[self.robot.links.names.index(self.scan.realsense2_urdf_link_name)]
        self.realsense2_frustrum.position = realsense2_pose[:3]
        self.realsense2_frustrum.wxyz = realsense2_pose[3:]
        camera1_pose = all_link_poses[self.robot.links.names.index(self.scan.camera1_urdf_link_name)]
        self.camera1_frustrum.position = camera1_pose[:3]
        self.camera1_frustrum.wxyz = camera1_pose[3:]
        camera2_pose = all_link_poses[self.robot.links.names.index(self.scan.camera2_urdf_link_name)]
        self.camera2_frustrum.position = camera2_pose[:3]
        self.camera2_frustrum.wxyz = camera2_pose[3:]
        camera3_pose = all_link_poses[self.robot.links.names.index(self.scan.camera3_urdf_link_name)]
        self.camera3_frustrum.position = camera3_pose[:3]
        self.camera3_frustrum.wxyz = camera3_pose[3:]
        camera4_pose = all_link_poses[self.robot.links.names.index(self.scan.camera4_urdf_link_name)]
        self.camera4_frustrum.position = camera4_pose[:3]
        self.camera4_frustrum.wxyz = camera4_pose[3:]
        camera5_pose = all_link_poses[self.robot.links.names.index(self.scan.camera5_urdf_link_name)]
        self.camera5_frustrum.position = camera5_pose[:3]
        self.camera5_frustrum.wxyz = camera5_pose[3:]

    def step(self):
        pass

if __name__ == "__main__":
    args = setup_log_with_config(VizScanConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    viz = VizScan(args)
    viz.run()