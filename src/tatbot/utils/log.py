import logging
import os
import traceback
from dataclasses import asdict
from pprint import pformat
from typing import Any

import tyro

TIME_FORMAT: str = "%Yy-%mm-%dd-%Hh-%Mm-%Ss"
LOG_FORMAT: str = "%(asctime)s %(levelname)s: %(message)s"
SUBMODULES: list[str] = ["bot", "cam", "data", "gen", "mcp", "ops", "utils", "viz"]


def get_logger(name: str, emoji: str = "❓") -> logging.Logger:
    """Get a logger with a specific name."""
    _log = logging.getLogger(f"tatbot.{name}")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(f"{emoji} {LOG_FORMAT}"))
    if not _log.hasHandlers():
        _log.addHandler(handler)
    _log.propagate = False
    return _log


log = get_logger("utils.log", "📝")


def print_config(args: Any, log: logging.Logger = log) -> None:
    log.debug(f"🛠️ Full Config of type {type(args)}:")
    log.debug(pformat(asdict(args)))


def setup_log_with_config(config: Any, submodules: list[str] = SUBMODULES) -> Any:
    args = tyro.cli(config)
    logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for submodule in submodules:
            logging.getLogger(f"tatbot.{submodule}").setLevel(logging.DEBUG)
        log.debug("🐛 Debug mode enabled.")
    if hasattr(args, "output_dir"):
        os.makedirs(args.output_dir, exist_ok=True)
        log.info(f"💾 Saving output to {args.output_dir}")
    try:
        import jax

        log.info(f"🧠 JAX devices: {jax.devices()}")
    except Exception:
        log.info(f"🧠 JAX not available: {traceback.format_exc()}")
    return args
