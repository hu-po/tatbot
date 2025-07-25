import logging

from lerobot.robots import Robot, make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig

from tatbot.ops.base import BaseOp, BaseOpConfig
from tatbot.utils.log import get_logger

log = get_logger("ops.reset", "🤖")


class ResetOp(BaseOp):

    op_name: str = "reset"

    def __init__(self, config: BaseOpConfig):
        super().__init__(config)
        if config.debug:
            logging.getLogger("lerobot").setLevel(logging.DEBUG)
        self.robot: Robot | None = None

    def make_robot(self) -> Robot:
        """Make a robot from the config."""
        return make_robot_from_config(
            TatbotConfig(
                ip_address_l=self.scene.arms.ip_address_l,
                ip_address_r=self.scene.arms.ip_address_r,
                arm_l_config_filepath=self.scene.arms.arm_l_config_filepath,
                arm_r_config_filepath=self.scene.arms.arm_r_config_filepath,
                goal_time=self.scene.arms.goal_time_slow,
                connection_timeout=self.scene.arms.connection_timeout,
                home_pos_l=self.scene.sleep_pos_l.joints,
                home_pos_r=self.scene.sleep_pos_r.joints,
                rs_cameras={},
                ip_cameras={},
            )
        )

    async def run(self):
        _msg = "🤖 Resetting robot..."
        log.info(_msg)
        yield {
            'progress': 0.5,
            'message': _msg,
        }
        self.robot = self.make_robot()
        self.robot.disconnect()
