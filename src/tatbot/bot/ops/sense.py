from dataclasses import dataclass
import os

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from PIL import Image

from tatbot.bot.ops.record import RecordOp, RecordOpConfig
from lerobot.robots import make_robot_from_config, Robot
from lerobot.robots.tatbot.config_tatbot import TatbotConfig

from tatbot.cam.depth import DepthCamera
from tatbot.bot.urdf import get_link_poses
from tatbot.utils.log import get_logger

log = get_logger("bot.ops.sense", "🔍")

@dataclass
class SenseOpConfig(RecordOpConfig):

    num_plys: int = 2
    """Number of PLY pointcloud files to capture."""


class SenseOp(RecordOp):

    op_name: str = "sense"

    def make_robot(self) -> Robot:
        """Make a robot from the config."""
        return make_robot_from_config(
            TatbotConfig(
                ip_address_l=self.scene.arms.ip_address_l,
                ip_address_r=self.scene.arms.ip_address_r,
                arm_l_config_filepath=self.scene.arms.arm_l_config_filepath,
                arm_r_config_filepath=self.scene.arms.arm_r_config_filepath,
                goal_time_fast=self.scene.arms.goal_time_fast,
                goal_time_slow=self.scene.arms.goal_time_slow,
                connection_timeout=self.scene.arms.connection_timeout,
                home_pos_l=self.scene.sleep_pos_l.joints,
                home_pos_r=self.scene.sleep_pos_r.joints,
                rs_cameras={
                    cam.name : RealSenseCameraConfig(
                        fps=cam.fps,
                        width=cam.width,
                        height=cam.height,
                        serial_number_or_name=cam.serial_number,
                    ) for cam in self.scene.cams.realsenses
                },
                ip_cameras={
                    cam.name: OpenCVCameraConfig(
                        fps=cam.fps,
                        width=cam.width,
                        height=cam.height,
                        ip=cam.ip,
                        username=cam.username,
                        password=os.environ.get(cam.password, None),
                        rtsp_port=cam.rtsp_port,
                    )
                    for cam in self.scene.cams.ipcameras
                },
            )
        )

    async def _run(self):
        _msg = f"🤖 Sending robot to ready position"
        log.info(_msg)
        yield {
            'progress': 0.2,
            'message': _msg,
        }
        ready_action = self.robot._urdf_joints_to_action(self.scene.ready_pos_full)
        self.robot.send_action(ready_action, goal_time=self.robot.config.goal_time_slow, block="both")

        _msg = f"🤖 Recording observation (png images)"
        log.info(_msg)
        yield {
            'progress': 0.3,
            'message': _msg,
        }
        observation = self.robot.get_observation_full()
        for key, data in observation.items():
            if key in self.robot.rs_cameras:
                image_path = os.path.join(self.dataset_dir, f"{key}.png")
                Image.fromarray(data).save(image_path)
                log.info(f"✅ Saved frame to {image_path}")
            elif key in self.robot.ip_cameras:
                image_path = os.path.join(self.dataset_dir, f"{key}.png")
                Image.fromarray(data).save(image_path)
                log.info(f"✅ Saved frame to {image_path}")
        _msg = f"✅ Saved image frames"
        log.info(_msg)
        yield {
            'progress': 0.31,
            'message': _msg,
        }
        
        _msg = "Disconnecting realsense cameras..."
        log.info(_msg)
        yield {
            'progress': 0.4,
            'message': _msg,
        }
        for cam in self.robot.rs_cameras.values():
            cam.disconnect()

        _msg = "Connecting to Depth cameras..."
        log.info(_msg)
        yield {
            'progress': 0.5,
            'message': _msg,
        }
        link_poses = get_link_poses(
                self.scene.urdf.path, self.scene.urdf.cam_link_names, self.scene.ready_pos_full
            )
        depth_cameras = {}
        for realsense in self.scene.cams.realsenses:
            depth_cameras[realsense.name] = DepthCamera(
                realsense.serial_number,
                link_poses[realsense.urdf_link_name],
                save_prefix=f"{realsense.name}_",
                save_dir=self.dataset_dir,
            )

        _msg = f"Capturing {self.config.num_plys} pointclouds..."
        log.info(_msg)
        yield {
            'progress': 0.6,
            'message': _msg,
        }
        for ply_idx in range(self.config.num_plys):
            log.info(f"Capturing pointcloud {ply_idx + 1}/{self.config.num_plys}...")
            for depth_cam in depth_cameras.values():
                depth_cam.get_pointcloud(save=True)
        
