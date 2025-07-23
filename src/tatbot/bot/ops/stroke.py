from dataclasses import dataclass
import os
from io import StringIO
import logging

from lerobot.cameras.realsense import RealSenseCameraConfig

from tatbot.bot.ops.record import RecordOp, RecordOpConfig
from lerobot.robots import make_robot_from_config, Robot
from lerobot.robots.tatbot.config_tatbot import TatbotConfig

from tatbot.utils.log import get_logger, LOG_FORMAT, TIME_FORMAT

log = get_logger("bot.ops.stroke", "🖌️")


@dataclass
class StrokeOpConfig(RecordOpConfig):
    pass


class StrokeOp(RecordOp):

    op_name: str = "stroke"

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
                cond_cameras={},
            )
        )

    async def run(self):
        async for progress_update in super().run():
            yield progress_update

        logs_dir = os.path.join(self.dataset_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        episode_log_buffer = StringIO()

        class EpisodeLogHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record)
                episode_log_buffer.write(msg + "\n")

        episode_handler = EpisodeLogHandler()
        episode_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=TIME_FORMAT))
        logging.getLogger().addHandler(episode_handler)
        _msg = f"🗃️ Creating logs directory for episode logs at {logs_dir}..."
        log.info(_msg)
        yield {
            'progress': 0.03,
            'message': _msg,
        }