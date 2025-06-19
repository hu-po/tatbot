from dataclasses import asdict
import logging
import os
from pprint import pformat
from typing import Any

import tyro

TIME_FORMAT: str = "%Yy-%mm-%dd-%Hh-%Mm-%Ss"
LOG_FORMAT: str = "%(asctime)s %(levelname)s: %(message)s"
SUBMODULES: list[str] = ['_cam', '_ik', '_ink', '_log', '_map', '_net', '_path', '_plan']

# used for gui log text/circles/bboxes
COLORS: dict[str, tuple[int, int, int]] = {
    "black":      (  0,   0,   0),
    "white":      (255, 255, 255),
    "blue":       ( 82, 153, 224),
    "green":      ( 82, 224, 105),
    "red":        (224,  86,  82),
    "yellow":     (224, 212,  82),
    "purple":     (189,  82, 224),
    "orange":     (224, 117,  82),
    "gold":       (189, 224,  82),
    "chartreuse": (105, 224,  82),
    "mint":       ( 82, 224, 177),
    "teal":       ( 82, 224, 224),
    "cyan":       ( 82, 189, 224),
    "indigo":     ( 82,  82, 224),
    "violet":     (129,  82, 224),
    "magenta":    (224,  82, 201),
    "pink":       (224,  82, 129),
}

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
