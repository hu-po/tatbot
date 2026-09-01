"""tatbot LeRobot robot plugin.

Importing this package (which lerobot does automatically for any installed
package named ``lerobot_robot_*``) registers the ``tatbot_follower`` robot
type: a Trossen WidowX AI follower whose gripper runs under the bounded-grip
force law proven in cpp/teleop/wxai_teleop.cpp.
"""

from lerobot_robot_tatbot.config_tatbot_follower import TatbotFollowerConfig
from lerobot_robot_tatbot.config_tatbot_leader import TatbotLeaderTeleopConfig
from lerobot_robot_tatbot.tatbot_follower import TatbotFollower
from lerobot_robot_tatbot.tatbot_leader import TatbotLeader, TatbotLeaderTeleop

__all__ = [
    "TatbotFollower",
    "TatbotFollowerConfig",
    "TatbotLeader",
    "TatbotLeaderTeleop",
    "TatbotLeaderTeleopConfig",
]
