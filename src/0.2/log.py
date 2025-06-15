from dataclasses import asdict
import os
import logging
from pprint import pformat
from typing import Any

import tyro

TIME_FORMAT = "%Y-%mm-%dd-%Hh-%Mm-%Ss"
LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"

# used for gui log text/circles/bboxes
COLORS: dict[str, tuple[int, int, int]] = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "purple": (255, 0, 255),
}

def get_logger(name: str) -> logging.Logger:
    """Get a logger with a specific name."""
    return logging.getLogger(f"tatbot.{name}")

log = get_logger('log')

def print_config(args: Any):
    log.info(f"🛠️ Full Config of type {type(args)}:")
    log.info(pformat(asdict(args)))

def setup_log_with_config(config: Any) -> Any:
    args = tyro.cli(config)
    logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        log.debug("🐛 Debug mode enabled.")
    if hasattr(args, "output_dir"):
        os.makedirs(args.output_dir, exist_ok=True)
        log.info(f"💾 Saving output to {args.output_dir}")
    print_config(args)
    return args
