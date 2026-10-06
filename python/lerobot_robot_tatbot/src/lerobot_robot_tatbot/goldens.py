"""Golden config files: the three-file model.

  leader.yaml / follower.yaml — full arm EEPROM images (driver
      load_configs_from_file format). Loaded into the controller at every
      connect, because controller state is scratch RAM that reverts on power
      cycle. One per ROLE, not per experiment.
  tatbot.yaml — everything that is ours rather than the firmware's: the
      carriage, smoothing, slew limits, motion scale, staged pose. Single
      source of truth for the follower plugin, the C++ teleop and the travel runner.

All three are hand-edited and read-only here; git history is the changelog —
there is deliberately no profile library.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Container
from dataclasses import MISSING, fields
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ENV_CONFIG_DIR = "TATBOT_CONFIG_DIR"


def config_dir() -> Path:
    """Resolve config/trossen: $TATBOT_CONFIG_DIR, else walk up from this
    file (editable install → repo checkout), else fail closed."""
    env = os.environ.get(ENV_CONFIG_DIR)
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "trossen"
        if candidate.is_dir():
            return candidate
    # Fail closed (plan Phase 1): no guessing a local checkout path when the
    # walk-up finds nothing — say how to point at one instead.
    raise FileNotFoundError(
        "no config/trossen found above this install; set TATBOT_CONFIG_DIR "
        "to the directory holding the arm YAMLs")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# tatbot.yaml
# ---------------------------------------------------------------------------


def apply_section(config, section: dict | None, skip: Container[str] = frozenset()) -> list[str]:
    """Apply a tatbot.yaml section onto a plugin config dataclass.

    CLI overrides win: a yaml value is applied only when the config attribute
    still equals its dataclass default (i.e. the user did not set it for this
    session). Returns the list of applied field names.
    """
    if not section:
        return []
    defaults = {}
    for f in fields(type(config)):
        if f.default is not MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            defaults[f.name] = f.default_factory()  # type: ignore[misc]
    applied = []
    for key, value in section.items():
        if key in skip or not hasattr(config, key):
            continue
        current = getattr(config, key)
        if key in defaults and current != defaults[key]:
            logger.info(
                "tatbot.yaml: keeping CLI override %s=%r (yaml has %r)",
                key, current, value,
            )
            continue
        setattr(config, key, value)
        applied.append(key)
    if applied:
        logger.info("tatbot.yaml: applied %s", ", ".join(applied))
    return applied


def load_tatbot_yaml(cfg_dir: Path | None = None) -> dict:
    path = (cfg_dir or config_dir()) / "tatbot.yaml"
    if not path.exists():
        logger.warning("no tatbot.yaml at %s — using dataclass defaults", path)
        return {}
    return load_yaml(path)


# ---------------------------------------------------------------------------
# Arm golden apply (connect-time)
# ---------------------------------------------------------------------------


def apply_arm_golden(driver, trossen_arm_mod, path: Path) -> list[str]:
    """Write an arm golden YAML into the controller via field-wise setters.

    Deliberately NOT load_configs_from_file: that path also rewrites the
    network EEPROM block on every connect (a needless flash write and a
    suspect for TCP-server resets) and hard-fails on any schema drift
    between driver versions (the 1.8.8 position_offset incident). Here we
    read each parameter group, overlay the YAML's values field by field
    (unknown keys ignored, missing keys keep controller values), and write
    it back. Modes and network config are never touched; the end-effector
    preset is applied separately by the caller.
    """
    doc = load_yaml(path)
    applied: list[str] = []

    if doc.get("joint_characteristics"):
        jc = driver.get_joint_characteristics()
        for obj, entry in zip(jc, doc["joint_characteristics"], strict=True):
            for key, val in entry.items():
                if hasattr(obj, key):
                    setattr(obj, key, float(val))
        driver.set_joint_characteristics(jc)
        applied.append("joint_characteristics")

    if doc.get("joint_limits"):
        jl = driver.get_joint_limits()
        for obj, entry in zip(jl, doc["joint_limits"], strict=True):
            for key, val in entry.items():
                if hasattr(obj, key):
                    setattr(obj, key, float(val))
        driver.set_joint_limits(jl)
        applied.append("joint_limits")

    if doc.get("motor_parameters"):
        mp = driver.get_motor_parameters()
        mode_by_name = dict(trossen_arm_mod.Mode.__members__)
        for j, entry in enumerate(doc["motor_parameters"]):
            for mode_name, loops in entry.items():
                mode = mode_by_name.get(mode_name)
                if mode is None or mode not in mp[j]:
                    continue
                # pybind attribute access returns copies — pull, edit, push.
                motor = mp[j][mode]
                for loop_name in ("position", "velocity"):
                    if loop_name not in loops:
                        continue
                    pid = getattr(motor, loop_name)
                    for key, val in loops[loop_name].items():
                        if hasattr(pid, key):
                            setattr(pid, key, float(val))
                    setattr(motor, loop_name, pid)
                mp[j][mode] = motor
        driver.set_motor_parameters(mp)
        applied.append("motor_parameters")

    algo = doc.get("algorithm_parameter")
    if algo and "singularity_threshold" in algo:
        ap = driver.get_algorithm_parameter()
        ap.singularity_threshold = float(algo["singularity_threshold"])
        driver.set_algorithm_parameter(ap)
        applied.append("algorithm_parameter")

    return applied
