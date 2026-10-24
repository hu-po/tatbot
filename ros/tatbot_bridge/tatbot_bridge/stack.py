"""stack.yaml, the tatbot bus endpoint and the adopted registrations, for the bridge and the Rerun
bridge. No ROS import: `ament_index_python` is used only when it is importable."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from tatbot_description import repo_root


def stack_path(config: str = "", *, repo=None, fallback=True) -> Path:
    """The stack.yaml to read: the argument, else the installed share copy, else the checkout's."""
    if config:
        return Path(config).expanduser()
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("tatbot_bringup")) / "config" / "stack.yaml"
    except (ImportError, LookupError, ValueError):
        if not fallback:
            raise
        return repo_root(repo) / "ros" / "tatbot_bringup" / "config" / "stack.yaml"


def load(config: str = "", **location) -> dict:
    return yaml.safe_load(stack_path(config, **location).read_text())


def arms(value, stack: dict) -> list[str]:
    """A comma list or a list; '' or empty = stack.yaml arms."""
    if isinstance(value, str):
        value = [a.strip() for a in value.split(",") if a.strip()]
    return list(value or stack.get("arms") or ["right"])


def fleet():
    """tatbot_cli.nodes from $TATBOT_REPO/scripts/lib (config/nodes.json is the only place addresses live)."""
    lib = str(repo_root() / "scripts" / "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    from tatbot_cli import nodes
    return nodes


def bus_endpoint() -> str:
    """The tatbot bus router's rig-LAN endpoint, tcp/<lan>:7447."""
    nodes = fleet()
    return nodes.bus_endpoint(nodes.load(repo_root()), address="lan")


def string_parameters(node, keys) -> dict:
    """Declare `keys` as ROS parameters of any type (a launch or `-p k:=20` may pass a number or a list)
    and return them as strings; '' (unset) means the stack.yaml value."""
    from rcl_interfaces.msg import ParameterDescriptor

    out = {}
    for key in keys:
        value = node.declare_parameter(key, "", ParameterDescriptor(dynamic_typing=True)).value
        out[key] = ",".join(map(str, value)) if isinstance(value, (list, tuple)) else ("" if value is None else str(value))
    return out


def registration_path(stack: dict, arm: str) -> Path | None:
    value = (stack.get("registration") or {}).get(arm)
    return Path(value).expanduser() if value else None


def world_from_arm_base(stack: dict, arm: str):
    """The adopted registration's world_from_arm_base, or None when the file is absent."""
    import numpy as np
    from tatbot_description import registration

    path = registration_path(stack, arm)
    if path is None or not path.is_file():
        return None
    return np.asarray(registration(path), float)


def open_bus(endpoint: str):
    """A zenoh client session on the tatbot bus; the bridge only reads it."""
    import json

    import zenoh
    config = zenoh.Config()
    settings = [("mode", "client"), ("connect/endpoints", [endpoint]), ("scouting/multicast/enabled", False),
                ("connect/timeout_ms", 3000), ("connect/exit_on_failure", False)]
    credentials = fleet().bus_credentials()   # the router admits only it once its node holds one
    if credentials:
        settings += [("transport/auth/usrpwd/user", credentials[0]), ("transport/auth/usrpwd/password", credentials[1])]
    for key, value in settings:
        config.insert_json5(key, json.dumps(value))
    return zenoh.open(config)

