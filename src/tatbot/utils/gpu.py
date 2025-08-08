"""GPU-related utility helpers."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Optional

import yaml

from tatbot.utils.log import get_logger

log = get_logger("utils.gpu", "🧪")


def check_local_gpu() -> bool:
    """Return True if the current node has GPU extras configured."""
    try:
        hostname = socket.gethostname()
        node_name = hostname.lower()
        config_dir = Path(__file__).resolve().parent.parent / "conf" / "mcp"
        node_config_file = config_dir / f"{node_name}.yaml"

        if node_config_file.exists():
            with open(node_config_file, "r") as f:
                node_config = yaml.safe_load(f) or {}
            return "gpu" in node_config.get("extras", [])
        return False
    except Exception as e:  # pragma: no cover - environment dependent
        log.warning(f"Failed to check local GPU support: {e}")
        return False


