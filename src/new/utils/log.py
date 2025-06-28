from dataclasses import asdict
import logging
import os
from pprint import pformat
from typing import Any

import tyro

TIME_FORMAT: str = "%Yy-%mm-%dd-%Hh-%Mm-%Ss"
LOG_FORMAT: str = "%(asctime)s %(levelname)s: %(message)s"
SUBMODULES: list[str] = ['_bot', '_ik', '_ink', '_log', '_map', '_net', '_path', '_plan', '_scan', '_tag', '_viz']

def get_logger(name: str) -> logging.Logger:
    """Get a logger with a specific name."""
    return logging.getLogger(f"tatbot.{name}")

log = get_logger('_log')

def print_config(args: Any):
    log.info(f"🛠️ Full Config of type {type(args)}:")
    log.info(pformat(asdict(args)))

def setup_log_with_config(config: Any) -> Any:
    args = tyro.cli(config)
    logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        for submodule in SUBMODULES:
            logging.getLogger(f"tatbot.{submodule}").setLevel(logging.DEBUG)
        log.debug("🐛 Debug mode enabled.")
    if hasattr(args, "output_dir"):
        os.makedirs(args.output_dir, exist_ok=True)
        log.info(f"💾 Saving output to {args.output_dir}")
    return args
