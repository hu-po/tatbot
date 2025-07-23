from dataclasses import dataclass

from tatbot.bot.behaviors.base import BaseBehavior, BaseBehaviorConfig


@dataclass
class AlignBehaviorConfig(BaseBehaviorConfig):
    pass


class AlignBehavior(BaseBehavior):
    def __init__(self, config: AlignBehaviorConfig):
        super().__init__(config)

    def run(self):
        super().run()