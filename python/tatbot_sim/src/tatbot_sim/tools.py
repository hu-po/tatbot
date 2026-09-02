"""Access to the repo's tool registry from inside the sim package.

``scripts/lib/tool_spec.py`` is the single implementation — the sim is a
separate uv project, so it loads that file by path rather than vendoring a
second copy that would drift. The registry is stdlib-only precisely so this
works without adding a dependency.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

from tatbot_sim.repo import repo_root

REPO = repo_root()
_MODULE_NAME = "tatbot_tool_spec"
_MODULE_PATH = REPO / "scripts" / "lib" / "tool_spec.py"
CALIBRATION_DELTA_ENV = "TATBOT_SIM_TIP_DELTA_M"
SIM_WORKSPACE_RELPATH = "config/examples/workspace.yaml"
ARM_GOLDEN_RELPATH = "config/trossen/tatbot.yaml"
SIM_ARM_GOLDEN_RELPATH = "config/examples/tatbot-sim.yaml"


@lru_cache(maxsize=1)
def registry():
    """The tool_spec module itself, for ToolSpec/load_tool/dataset metadata."""
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"tool registry not importable at {_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the module's dataclasses use postponed
    # annotations, which dataclasses resolves through sys.modules.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def active_tool():
    """The tool config/workspace.yaml says is fitted to the follower.

    ``TATBOT_TOOL_ID`` overrides it for PREVIEWING a tool that is not fitted —
    rendering the 3RL or the laser without lying about what is in the gripper.
    It changes nothing on disk, and deliberately cannot: the fitted tool is a
    calibration fact and only a touch-off gets to write it.

    Cached: the URDF build and the agent class body both ask at import time,
    and one process cannot swap tools mid-run anyway.
    """
    override = os.environ.get("TATBOT_TOOL_ID")
    if override:
        return registry().load_tool(override, REPO)
    return registry().load_active_tool(REPO, workspace=workspace())


def active_substrate():
    """What the fitted tool works on: the paper pad, or the silicone skin.

    A tool and a substrate are a pair on this bench, so the scene follows the
    gripper — swapping to the laser swaps the whole working surface, its size
    and its appearance, rather than leaving a letter-size ruled pad under a
    tool that never touches one.
    """
    return registry().substrate_for(active_tool(), REPO)


def workspace_path() -> Path:
    """Use live calibration when present, else the public simulation fixture."""
    live = REPO / registry().WORKSPACE_RELPATH
    return live if live.is_file() else REPO / SIM_WORKSPACE_RELPATH


def workspace() -> dict:
    path = workspace_path()
    return registry().parse_simple_yaml(path.read_text()) if path.is_file() else {}


def calibration_delta_m() -> tuple[float, float, float]:
    """Process-scoped mount-frame tip perturbation selected by the factory.

    A physical seat persists for a session, so one simulator shard gets one
    draw rather than changing the tool between episodes.  The factory sets the
    value before re-exec/import so the derived URDF, IK and metadata all see
    the same geometry.
    """
    raw = os.environ.get(CALIBRATION_DELTA_ENV)
    if not raw:
        return (0.0, 0.0, 0.0)
    try:
        values = tuple(float(value) for value in json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{CALIBRATION_DELTA_ENV} must be a JSON 3-vector") from exc
    if len(values) != 3:
        raise ValueError(f"{CALIBRATION_DELTA_ENV} must contain three values")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{CALIBRATION_DELTA_ENV} must contain finite values")
    return values


def resolved_geometry(spec=None, ws: dict | None = None):
    """The exact per-process geometry shared by URDF, IK and metadata."""
    reg = registry()
    active = spec or active_tool()
    current = workspace() if ws is None else ws
    return reg.resolved_tool_geometry(
        active, current, "right", REPO, tip_delta_m=calibration_delta_m())


def arm_golden_path() -> Path:
    """Select the rig profile or the explicitly simulation-only public fixture."""
    live = REPO / ARM_GOLDEN_RELPATH
    return live if live.is_file() else REPO / SIM_ARM_GOLDEN_RELPATH


@lru_cache(maxsize=1)
def arm_golden() -> dict:
    """The selected follower profile for the staged pose and carriage rest.

    A private rig checkout reads ``config/trossen/tatbot.yaml``. A public
    checkout instead reads an explicitly simulation-only fixture that carries
    no controller limits or powered-operation authority.
    """
    path = arm_golden_path()
    data = registry().parse_simple_yaml(path.read_text())
    return data["follower"]


def staged_pose() -> list[float]:
    """The follower's 7-value staged/idle pose (six joints + carriage)."""
    pose = [float(v) for v in arm_golden()["staged_positions"]]
    if len(pose) != 7:
        raise ValueError(f"{arm_golden_path()}: staged_positions has {len(pose)} values, need 7")
    return pose


def carriage_rest_m() -> float:
    """Where the carriage rests: the pen's extended position (0.0 = closed hard stop)."""
    return float(arm_golden()["carriage_rest_m"])


def tool_source_paths() -> list[Path]:
    """Files whose edits invalidate a derived URDF."""
    tool = active_tool()
    paths = [workspace_path()]
    if tool.source is not None:
        paths.append(Path(tool.source))
    return [p for p in paths if p.exists()]


# --- ink: the fourth leg of (task, tool, substrate, ink) -----------------------------

_INK_MODULE_NAME = "tatbot_ink_spec"
_INK_MODULE_PATH = REPO / "scripts" / "lib" / "ink_spec.py"


@lru_cache(maxsize=1)
def ink_registry():
    """``scripts/lib/ink_spec.py``, loaded by path like the tool registry.
    Registering the tool registry first lets ink_spec find it in sys.modules."""
    registry()
    spec = importlib.util.spec_from_file_location(_INK_MODULE_NAME, _INK_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"ink registry not importable at {_INK_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_INK_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def active_ink_policy():
    """The fitted tool's ``ink:`` block: real (3RL), rehearsal (ballpoint),
    or none (laser). Cached with the tool, for the same reason."""
    return ink_registry().policy_for(active_tool())


def palette():
    return ink_registry().load_palette(REPO)


# The sim's ink SUPPLY: which palette load the planner, the validator and the
# env see. ("bench", None) reads config/palette_load.yaml — what was poured on
# the real rack this morning. A simulator is not the bench, so generate and
# the factory default to a synthetic wet rack (set_supply("wet", ink_id)) and a
# batch is never refused because nobody has run `ink.py load` today; the run's
# meta/ink.json records which supply it drew from.
_SUPPLY: tuple[str, str | None] = ("bench", None)


def set_supply(kind: str, ink_id: str | None = None) -> None:
    """Choose the palette load every later ``palette_load()`` returns:
    ``bench`` (the yaml), ``wet`` (every right-arm cap full of ``ink_id``) or
    ``dry`` (every cap empty). Process-wide, like the fitted tool."""
    global _SUPPLY
    ink = ink_registry()
    if kind not in ink.SUPPLIES:
        raise ValueError(f"supply {kind!r} not one of {ink.SUPPLIES}")
    if kind == "wet":
        if not ink_id:
            raise ValueError("--supply wet needs --supply-ink <ink_id>")
        if ink_id not in ink.load_inks(REPO):
            raise ValueError(f"unknown ink {ink_id!r}; have {', '.join(ink.load_inks(REPO))}")
    _SUPPLY = (kind, ink_id if kind == "wet" else None)


def supply() -> tuple[str, str | None]:
    return _SUPPLY


def palette_load():
    """What is in each cap for THIS process: config/palette_load.yaml as the
    bench holds it right now (read fresh, not cached — an operator fills a
    cap between runs, not between imports), or the synthetic supply chosen
    by ``set_supply``."""
    ink = ink_registry()
    kind, ink_id = _SUPPLY
    return ink.supply_load(kind, ink.load_palette(REPO), ink_id, REPO)
