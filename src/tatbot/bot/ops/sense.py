from dataclasses import dataclass
import os

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

from tatbot.bot.ops.record import RecordOp, RecordOpConfig
from lerobot.robots import make_robot_from_config, Robot
from lerobot.robots.tatbot.config_tatbot import TatbotConfig

from tatbot.utils.log import get_logger

log = get_logger("bot.ops.sense", "🔍")

@dataclass
class SenseOpConfig(RecordOpConfig):
    pass


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
                home_pos_l=self.scene.sleep_pos_l.joints[:7],
                home_pos_r=self.scene.sleep_pos_r.joints[:7],
                cameras={
                    cam.name : RealSenseCameraConfig(
                        fps=cam.fps,
                        width=cam.width,
                        height=cam.height,
                        serial_number_or_name=cam.serial_number,
                    ) for cam in self.scene.cams.realsenses
                },
                cond_cameras={
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

    async def run(self):
        async for progress_update in super().run():
            yield progress_update
        
        # Add your sense-specific functionality here
        _msg = f"🔍 Running sense operation..."
        log.info(_msg)
        yield {
            'progress': 0.95,
            'message': _msg,
        }
        
        # Your sense-specific code goes here
        # For example:
        # - Process sensor data
        # - Analyze robot state
        # - Perform sensing operations
        
        _msg = f"✅ Sense operation completed"
        log.info(_msg)
        yield {
            'progress': 1.0,
            'message': _msg,
        }