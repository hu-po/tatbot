"""tatbot synthetic data factory: importing this package registers the
``tatbot_wxai`` agent and ``TatbotDraw-v0`` env with ManiSkill."""

from tatbot_sim.agent import TatbotWXAI
from tatbot_sim.env import TatbotDrawEnv

__all__ = ["TatbotWXAI", "TatbotDrawEnv"]
