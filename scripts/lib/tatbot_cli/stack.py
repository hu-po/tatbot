"""The drawing stack's configuration as the CLI reads it: this checkout's stack.yaml, the file `tatbot ros deploy`
sends and the stack starts from. It is the one place that says which arm draws: the first of its `arms`
(tatbot_description.drawing_arm reads the same line on the ROS side). Stdlib, so every verb can read it."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STACK_YAML = "ros/tatbot_bringup/config/stack.yaml"

# stack.yaml's nested settings stack_defaults reads, by (top-level block, key): the name it gives each
STACK_KEYS = {("estop", "source"): "estop", ("estop", "relay_gpio"): "relay_gpio", ("page", "source"): "page",
              ("page", "pattern_id"): "pattern_id", ("probe", "relay_gpio"): "probe_gpio",
              ("probe", "touched"): "probe_touched", ("keypad", "port"): "keypad_port",
              ("keypad", "from_addr"): "keypad_from"}


def stack_defaults(repo: Path = REPO) -> dict:
    """hardware, arms, STACK_KEYS and machine.* from this checkout's stack.yaml (the one deploy sends). A line reader,
    so the backend stays stdlib."""
    out, block = {}, None
    for line in (repo / STACK_YAML).read_text().splitlines():
        body = line.split("#", 1)[0].rstrip()
        key, _, value = body.strip().partition(":")
        value = value.strip().strip('"')
        if not body.strip():
            continue
        if not body.startswith(" "):
            block = key
            if key in ("hardware", "arms"):
                out[key] = value.strip("[]").replace(" ", "")
        elif block == "machine":
            out[f"machine_{key}"] = value
        elif (block, key) in STACK_KEYS:
            out[STACK_KEYS[block, key]] = value
    return out


def drawing_arm(repo: Path = REPO) -> str:
    """The arm that draws, which a verb given no --arm drives: the first of stack.yaml `arms`."""
    return stack_defaults(repo)["arms"].split(",")[0]
