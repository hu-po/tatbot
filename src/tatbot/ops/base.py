import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from tatbot.main import compose_and_validate_scene
from tatbot.utils.log import get_logger, print_config

log = get_logger("ops.base", "✳️")


@dataclass
class BaseOpConfig:
    debug: bool = False
    """Enable debug logging."""

    scene: str = "default"
    """Name of the scene config to use (Scene)."""


class BaseOp:

    op_name: str = "base"
    """Name of the operation."""

    def __init__(self, config: BaseOpConfig):
        if config.debug:
            log.setLevel(logging.DEBUG)
        log.info(f"Initializing robot operation: {self.op_name}")
        print_config(config, log)
        self.config = config
        self.scene = compose_and_validate_scene(config.scene)

    def cleanup(self):
        """Cleanup the operation."""
        pass

    async def run(self) -> AsyncGenerator[dict[str, Any], None]:
        """Run the operation and yield intermediate results.
        
        Yields:
            dict: Intermediate results with keys like:
                - 'progress': float (0.0 to 1.0)
                - 'message': str (status message)
        """
        time.sleep(1)
        yield {
            'progress': 0.5,
            'message': 'Starting dummy base operation...',
        }
        time.sleep(1)
        yield {
            'progress': 1.0,
            'message': 'Completed dummy base operation...',
        }