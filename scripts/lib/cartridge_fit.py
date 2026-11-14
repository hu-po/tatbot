"""The cartridge fitted on each arm and each cartridge's measured tip, as the ROS owner keeps them.

`config/workspace.yaml` in git seeds them: the tool an arm was first calibrated with, and its tip. The ROS owner keeps
its own in `calib/cartridge-<arm>.json` beside its deployed repo, in the directory `tatbot ros deploy` makes and never
syncs. `tatbot ros cartridge fit` writes the fitted tool there and `tatbot ros calib apply` a measured tip, so a swap
or a tip adoption takes no commit, land or deploy. `tool_spec.read_workspace` lays the record over the arm's section,
so every reader on the owner (the robot description and its tool block, the session, the calibrations) sees one
fitted tool and one tip. The stack reads them when it starts; both verbs restart a running one.

A record keeps a tip per cartridge type, so fitting a type again starts from the last tip measured with it. That tip
is `carried`, not measured: a fresh 3RL seated its tube's end ~2 mm across and 2.45 mm along from the last one's
(2026-11-10). A type never measured on the arm starts at its datasheet's nominal tip. Stdlib only: tool_spec imports
this.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

SCHEMA = "tatbot.cartridge-fit/1"
# A section's keys that belong to its tool's tip (config/workspace.yaml's header). A record replaces them all, so no
# other tool's tip shows through; the frame, joint offsets and payload stay the section's.
TIP_KEYS = ("pen_tip_offset_x", "pen_tip_offset_y", "pen_tip_offset_z", "mechanical_contact_x",
            "mechanical_contact_y", "mechanical_contact_z", "mechanical_contact_reference",
            "mechanical_contact_profile", "mechanical_contact_status", "mechanical_contact_session", "touchoff",
            "tool_fit")
SEED = "config/workspace.yaml"   # a tip's source when the record was seeded from the deployed section
NOMINAL = "nominal"              # ... when its tool was fitted with no tip measured on the arm


def calib_dir(repo) -> Path:
    """The ROS owner's measured state: `calib/` beside its deployed `repo/` (tatbot_ros.deploy makes both)."""
    return Path(repo).expanduser().resolve().parent / "calib"


def path(repo, arm: str) -> Path:
    return calib_dir(repo) / f"cartridge-{arm}.json"


def load(repo, arm: str) -> dict | None:
    """The arm's record, or None where none was written (every checkout but a ROS owner's deployed one)."""
    try:
        text = path(repo, arm).read_text()
    except FileNotFoundError:
        return None
    return validate(json.loads(text), arm)


def validate(record: dict, arm: str) -> dict:
    """The record, or ValueError: a broken one must not fall back to git's tip, which may be another cartridge's."""
    where = f"cartridge-{arm}.json"
    if record.get("schema") != SCHEMA or record.get("arm") != arm:
        raise ValueError(f"{where}: not a {SCHEMA} record for the {arm} arm")
    tool, tips = record.get("tool_id"), record.get("tips")
    if not isinstance(tool, str) or not tool or not isinstance(tips, dict) or tool not in tips:
        raise ValueError(f"{where}: no tip for its fitted tool {tool!r}")
    for name, tip in tips.items():
        offsets = [tip.get("section", {}).get(f"pen_tip_offset_{axis}") for axis in "xyz"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in offsets):
            raise ValueError(f"{where}: {name}'s tip is not three finite numbers")
    return record


def overlay(workspace: dict, repo) -> dict:
    """`workspace` (config/workspace.yaml, parsed) with each arm's record over its section."""
    out = dict(workspace)
    for arm, section in workspace.items():
        if isinstance(section, dict) and (record := load(repo, arm)) is not None:
            out[arm] = fitted_section(section, record)
    return out


def fitted_section(section: dict, record: dict) -> dict:
    """The section with the record's fitted tool and that tool's tip in place of its own."""
    kept = {key: value for key, value in section.items() if key not in TIP_KEYS}
    return {**kept, "tool_id": record["tool_id"], **record["tips"][record["tool_id"]]["section"]}


def seeded(section: dict, arm: str) -> dict:
    """An arm's first record, from its deployed section: that tool fitted, its tip that section's."""
    tool = section.get("tool_id")
    if not tool:
        raise ValueError(f"config/workspace.yaml names no tool_id for the {arm} arm")
    tip = {"utc": (section.get("touchoff") or {}).get("utc"), "source": SEED,
           "section": {key: section[key] for key in TIP_KEYS if key in section}}
    return {"schema": SCHEMA, "arm": arm, "tool_id": tool, "instance": None, "fitted_utc": None, "tips": {tool: tip}}


def fit(record: dict, tool: str, nominal_tip, *, instance: str | None, utc: str) -> dict:
    """The record with `tool` fitted at `utc`: the last tip measured with that type, else `nominal_tip` (its
    datasheet's, in the mount frame)."""
    tips = dict(record["tips"])
    if tool not in tips:
        x, y, z = (float(v) for v in nominal_tip)
        tips[tool] = {"utc": None, "source": NOMINAL, "section": {
            "pen_tip_offset_x": x, "pen_tip_offset_y": y, "pen_tip_offset_z": z, "tool_fit": {"status": NOMINAL}}}
    return {**record, "tool_id": tool, "instance": instance, "fitted_utc": utc, "tips": tips}


def adopt(record: dict, section: dict, *, run_id: str, utc: str) -> dict:
    """The record with `section` (probe_calibration_adopt's tip keys) as its fitted tool's measured tip."""
    tips = {**record["tips"], record["tool_id"]: {"utc": utc, "source": run_id, "section": section}}
    return {**record, "tips": tips}


def tip_state(record: dict) -> str:
    """How the fitted cartridge's tip stands: `measured` (a run's, adopted since the cartridge was fitted, or since the
    record was seeded when no fit was recorded), `carried` (git's seed, or a run's from an earlier cartridge of its
    type) or `nominal`."""
    tip = record["tips"][record["tool_id"]]
    if tip["source"] in (NOMINAL, SEED):
        return NOMINAL if tip["source"] == NOMINAL else "carried"
    fitted = record.get("fitted_utc")
    return "measured" if not fitted or str(tip.get("utc") or "") >= fitted else "carried"


def write(repo, record: dict) -> Path:
    """Install the record, the one it replaces kept beside it as .replaced-<utc> (as the wrist gauges are)."""
    target = path(repo, record["arm"])
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".new")
    staged.write_text(json.dumps(validate(record, record["arm"]), indent=1) + "\n")
    if target.exists():
        target.rename(target.with_name(target.name + time.strftime(".replaced-%Y%m%dT%H%M%SZ", time.gmtime())))
    staged.rename(target)
    return target
