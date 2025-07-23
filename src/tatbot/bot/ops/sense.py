from dataclasses import dataclass
from typing import AsyncGenerator, Any

from tatbot.bot.behaviors.base import BaseBehavior, BaseBehaviorConfig


@dataclass
class SenseBehaviorConfig(BaseBehaviorConfig):
    pass


class SenseBehavior(BaseBehavior):
    def __init__(self, config: SenseBehaviorConfig):
        super().__init__(config)

    async def run(self) -> AsyncGenerator[dict[str, Any], None]:
        """Run the sense operation with progress updates."""
        yield {
            'progress': 0.5,
            'message': 'Sensing environment...',
            'step': 'sensing'
        }
        
        # Here you would implement the actual sensing logic
        # For now, just a placeholder
        
        yield {
            'progress': 1.0,
            'message': 'Sensing complete',
            'step': 'complete',
            'data': {'sensor_data': 'example_data'}
        }