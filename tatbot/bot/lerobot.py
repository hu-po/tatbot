from typing import Any, Callable

from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.tatbot.config_tatbot import TatbotBotOnlyConfig

from tatbot.utils.log import get_logger

log = get_logger('bot.lerobot', '🤖')

def safe_loop(loop: Callable, config: Any) -> None:
    try:
        loop(config)
    except Exception as e:
        log.error(f"❌Error:\n{e}\n")
    except KeyboardInterrupt:
        log.info("🛑⌨️ Keyboard interrupt detected. Disconnecting robot...")
    finally:
        log.info("🛑 Disconnecting robot...")
        robot = make_robot_from_config(TatbotBotOnlyConfig)
        robot._connect_l(clear_error=False)
        log.error(robot._get_error_str_l())
        robot._connect_r(clear_error=False)
        log.error(robot._get_error_str_r())
        robot.disconnect()