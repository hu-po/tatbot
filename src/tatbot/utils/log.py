import logging
import os
from dataclasses import asdict
from pprint import pformat
from typing import Any

import tyro

TIME_FORMAT: str = "%Yy-%mm-%dd-%Hh-%Mm-%Ss"
LOG_FORMAT: str = "%(asctime)s %(levelname)s: %(message)s"
SUBMODULES: list[str] = ['bot', 'data', 'gen', 'map', 'net', 'tag', 'utils', 'vla', 'viz']

def get_logger(name: str, emoji: str = "❓") -> logging.Logger:
    """Get a logger with a specific name."""
    _log = logging.getLogger(f"tatbot.{name}")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(f"{LOG_FORMAT} {emoji}"))
    if not _log.hasHandlers():
        _log.addHandler(handler)
    _log.propagate = False
    return _log

log = get_logger('utils.log', '📝')

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
    try:
        import jax
        log.info(f"🧠 JAX devices: {jax.devices()}")
    except ImportError:
        log.info("🧠 JAX not available")
    return args
