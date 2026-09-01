"""Golden config files: the three-file model.

  leader.yaml / follower.yaml — full arm EEPROM images (driver
      load_configs_from_file format). Loaded into the controller at every
      connect, because controller state is scratch RAM that reverts on power
      cycle. One per ROLE, not per experiment.
  tatbot.yaml — everything that is ours rather than the firmware's: grip law,
      smoothing, slew limits, motion scale, tuning-server settings. Single
      source of truth read by both plugins (and, later, the C++ cockpit).

Save-to-golden updates only the values the registry tunes, preserving the
rest of the arm YAMLs byte-for-byte where possible. Git history is the
changelog — there is deliberately no profile library.
"""

from __future__ import annotations

import contextlib
import logging
import os
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

TATBOT_YAML_HEADER = """\
# tatbot.yaml — custom (non-firmware) teleop/inference parameters.
# Single source of truth for the carriage contact cap, smoothing, slew limits and the
# tuning server. Loaded by the lerobot plugins at connect; values here apply
# unless the same field was explicitly overridden on the CLI (--robot.X=...).
# Saved by the tuning cockpit's "Save to golden" — git history is the
# changelog. The C++ teleop mirrors the carriage constants; keep
# cpp/teleop/wxai_teleop.cpp in sync when changing them here.
"""


def apply_section(config, section: dict | None, skip: set[str] = frozenset()) -> list[str]:
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


def save_tatbot_yaml(follower_config, tuning: dict, cfg_dir: Path | None = None,
                     leader_config=None) -> Path:
    """Update tatbot.yaml from the live follower config.

    Updates only the keys this function owns and preserves everything else that
    is already in the file. It used to rebuild the document from a literal
    dict, which silently deleted any section or key it did not know about the
    first time anyone touched a slider — a quiet data-loss bug for anything a
    future version of this file might carry.
    """
    path = (cfg_dir or config_dir()) / "tatbot.yaml"
    doc: dict = {}
    if path.is_file():
        with contextlib.suppress(Exception):
            doc = load_yaml(path) or {}
    doc.setdefault("tuning", {})
    doc.setdefault("follower", {})
    doc.setdefault("leader", {})

    updates = {
        "tuning": tuning,
        "follower": {
            "carriage_rest_m": float(follower_config.carriage_rest_m),
            "carriage_retract_m": float(follower_config.carriage_retract_m),
            "carriage_contact_cap_n": float(follower_config.carriage_contact_cap_n),
            "target_filter_tau": float(follower_config.target_filter_tau),
            "max_joint_velocity": float(follower_config.max_joint_velocity),
            "max_relative_target": (
                None if follower_config.max_relative_target is None
                else float(follower_config.max_relative_target)
            ),
            "min_time_to_move_multiplier": float(
                follower_config.min_time_to_move_multiplier
            ),
            "motion_scale": [float(v) for v in follower_config.motion_scale],
            "staged_positions": [float(v) for v in follower_config.staged_positions],
        },
        # Leader feel is otherwise driver-side (leader.yaml); this section
        # carries only the parameters that live in OUR code.
        "leader": (
            {"leader_damping": float(leader_config.leader_damping)}
            if leader_config is not None
            and hasattr(leader_config, "leader_damping")
            else {}
        ),
    }
    for section, values in updates.items():
        if isinstance(doc.get(section), dict) and isinstance(values, dict):
            doc[section].update(values)
        elif values or section not in doc:
            doc[section] = values

    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        f.write(TATBOT_YAML_HEADER)
        yaml.safe_dump(doc, f, sort_keys=False)
    os.replace(tmp, path)
    logger.info("saved %s", path)
    return path


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


# ---------------------------------------------------------------------------
# Arm golden updates (leader.yaml / follower.yaml)
# ---------------------------------------------------------------------------

# registry param name → (yaml list key, per-item field)
CHARACTERISTIC_FIELDS = {
    "friction_constant_term": "friction_constant_term",
    "friction_coulomb_coef": "friction_coulomb_coef",
    "friction_viscous_coef": "friction_viscous_coef",
    "friction_transition_velocity": "friction_transition_velocity",
    "effort_correction": "effort_correction",
}
LIMIT_FIELDS = (
    "velocity_max", "velocity_tolerance", "position_tolerance",
    "effort_max", "effort_tolerance",
)


def update_arm_golden(path: Path, values: dict[str, list[float]], prefix: str) -> Path:
    """Write tuned registry values back into an arm golden YAML.

    ``values`` maps registry names (e.g. "leader_friction_viscous_coef" or
    "follower_position_kp") to 7-vectors; ``prefix`` is "leader" or
    "follower". Only known tunable fields are touched.
    """
    doc = load_yaml(path)

    for reg_suffix, yaml_field in CHARACTERISTIC_FIELDS.items():
        vec = values.get(f"{prefix}_{reg_suffix}")
        if vec is None:
            continue
        for entry, v in zip(doc["joint_characteristics"], vec, strict=True):
            entry[yaml_field] = float(v)

    for field_name in LIMIT_FIELDS:
        vec = values.get(f"{prefix}_{field_name}")
        if vec is None:
            continue
        for entry, v in zip(doc["joint_limits"], vec, strict=True):
            entry[field_name] = float(v)

    kp = values.get(f"{prefix}_position_kp")
    vkp = values.get(f"{prefix}_velocity_kp")
    vki = values.get(f"{prefix}_velocity_ki")
    if any(v is not None for v in (kp, vkp, vki)):
        for j, entry in enumerate(doc["motor_parameters"]):
            pos_mode = entry["position"]  # the position-control mode block
            if kp is not None:
                pos_mode["position"]["kp"] = float(kp[j])
            if vkp is not None:
                pos_mode["velocity"]["kp"] = float(vkp[j])
            if vki is not None:
                pos_mode["velocity"]["ki"] = float(vki[j])

    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    os.replace(tmp, path)
    logger.info("saved %s", path)
    return path
