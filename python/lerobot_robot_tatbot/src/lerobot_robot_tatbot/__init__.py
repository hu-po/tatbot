"""tatbot LeRobot robot plugin.

Importing this package (which lerobot does automatically for any installed
package named ``lerobot_robot_*``) registers the ``tatbot_follower`` robot
type: a Trossen WidowX AI follower whose last joint is the carriage of a
mounted tool, seated at rest and retracted by the safety layer (nothing is
gripped since 2026-08-30).
"""

from lerobot_robot_tatbot.config_tatbot_follower import TatbotFollowerConfig
from lerobot_robot_tatbot.tatbot_follower import TatbotFollower

__all__ = [
    "TatbotFollower",
    "TatbotFollowerConfig",
]
