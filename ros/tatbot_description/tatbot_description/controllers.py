"""controller_manager's parameters for the arms a stack drives (ros/README.md "Controllers"): tatbot_bringup's
arm-neutral config/controllers.yaml with its `per_arm` controllers rendered under each arm's names, the arm's joints
from its model and its safety GPIO's interfaces from the names module. No ROS import."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from tatbot_description import names


def _arm_controller(arm: str) -> dict:
    return {"joints": list(names.joint_names(arm))}


def _safety_controller(arm: str) -> dict:
    gpio = names.safety_gpio(arm)
    return {"gpios": [gpio],
            "command_interfaces": {gpio: {"interfaces": list(names.SAFETY_COMMAND_INTERFACES)}},
            "state_interfaces": {gpio: {"interfaces": list(names.SAFETY_STATE_INTERFACES)}}}


PER_ARM = {"arm_controller": (names.arm_controller, _arm_controller),
           "safety_controller": (names.safety_controller, _safety_controller)}


def render(template: dict, arms) -> dict:
    """controllers.yaml as loaded, filled for `arms`: each `per_arm` entry becomes the arm's controller, its type
    declared to controller_manager and its parameters led by the arm's own."""
    config = copy.deepcopy(template)
    per_arm = config.pop("per_arm")
    manager = config["controller_manager"]["ros__parameters"]
    for arm in arms:
        for key, (name, own) in PER_ARM.items():
            manager[name(arm)] = {"type": per_arm[key]["type"]}
            config[name(arm)] = {"ros__parameters": {**own(arm), **copy.deepcopy(per_arm[key]["ros__parameters"])}}
    return config


def render_text(path: str | Path, arms) -> str:
    """The parameters file controller_manager and the spawner read, from the controllers.yaml at `path`."""
    rendered = render(yaml.safe_load(Path(path).read_text()), arms)
    return (f"# {Path(path).name} rendered for {','.join(arms)} by tatbot_description.controllers\n"
            + yaml.safe_dump(rendered, sort_keys=False))
