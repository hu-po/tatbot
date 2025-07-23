from dataclasses import dataclass

from tatbot.bot.behaviors.base import BaseBehavior, BaseBehaviorConfig


@dataclass
class StrokeBehaviorConfig(BaseBehaviorConfig):
    pass


class StrokeBehavior(BaseBehavior):
    def __init__(self, config: StrokeBehaviorConfig):
        super().__init__(config)

    def run(self):
        super().run()