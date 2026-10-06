"""Access to the repo's tool registry from inside the sim package.

``scripts/lib`` holds the single implementation of the tool, ink and dip
contracts. The sim is a separate uv project, so it depends on that directory
as ``tatbot-scriptlib`` and imports the modules normally; they are the same
files the scripts run from a bare clone, so neither side can drift.

The accessors below stay functions because callers hold them, not because the
import is expensive.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import ink_spec as _ink_spec
import tool_spec as _tool_spec

from tatbot_sim.repo import repo_root

REPO = repo_root()
CALIBRATION_DELTA_ENV = "TATBOT_SIM_TIP_DELTA_M"
SIM_WORKSPACE_RELPATH = "config/examples/workspace.yaml"
ARM_GOLDEN_RELPATH = "config/trossen/tatbot.yaml"
SIM_ARM_GOLDEN_RELPATH = "config/examples/tatbot-sim.yaml"


def registry():
    """The tool_spec module itself, for ToolSpec/load_tool/dataset metadata."""
    return _tool_spec


def active_tool():
    """The tool config/workspace.yaml says is fitted to the follower.

    ``TATBOT_TOOL_ID`` overrides it for PREVIEWING a tool that is not fitted —
    rendering the 3RL or the laser without lying about what is in the gripper.
    It changes nothing on disk, and deliberately cannot: the fitted tool is a
    calibration fact and only a touch-off gets to write it.

    This is an input resolver. Runtime constructors retain an explicit
    resolved configuration instead of caching process-wide tool selection.
    """
    override = os.environ.get("TATBOT_TOOL_ID")
    if override:
        return registry().load_tool(override, REPO)
    return registry().load_active_tool(REPO, workspace=workspace())


SUBSTRATE_ENV = "TATBOT_SUBSTRATE"


def active_substrate():
    """What the fitted tool works on: a paper fixture, or the silicone skin.

    A tool and its substrates are a pair on this bench, so the scene follows
    the gripper — swapping to the laser swaps the whole working surface, its
    size and its appearance, rather than leaving a gridded pad under a tool
    that never touches one. ``TATBOT_SUBSTRATE`` picks among the substrates
    the fitted tool's datasheet admits (the ballpoint: the paper pad or the
    paper cylinder); naming one it does not admit is refused, not substituted.
    """
    return registry().substrate_for(active_tool(), REPO, name=os.environ.get(SUBSTRATE_ENV) or None)


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
    value during configuration resolution so the derived URDF, IK and metadata all see
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
    """Resolve default-input geometry; runtimes retain ResolvedConfig.geometry."""
    reg = registry()
    active = spec or active_tool()
    current = workspace() if ws is None else ws
    return reg.resolved_tool_geometry(
        active, current, "right", REPO, tip_delta_m=calibration_delta_m())


def geometry_basis(geometry=None) -> str:
    """Truthful, stable provenance label for offline geometry selection."""
    resolved = geometry or resolved_geometry()
    if resolved.contact_status == "pivot-calibrated":
        return "measured-pivot"
    if resolved.source == "datasheet-nominal":
        return "nominal-datasheet"
    return "synthetic-development"


def geometry_warnings(spec=None, geometry=None) -> list[str]:
    """Warnings that never prevent offline simulation by themselves."""
    active = spec or active_tool()
    resolved = geometry or resolved_geometry(active)
    if not active.contact or resolved.contact_status == "pivot-calibrated":
        return []
    detail = resolved.contact_qualification_error or "no quality-gated pivot TCP"
    return [
        f"{active.tool_id} uses {geometry_basis(resolved)} geometry in simulation: "
        f"contact status is {resolved.contact_status} ({detail})",
    ]


def arm_golden_path() -> Path:
    """Select the rig profile or the explicitly simulation-only public fixture."""
    live = REPO / ARM_GOLDEN_RELPATH
    return live if live.is_file() else REPO / SIM_ARM_GOLDEN_RELPATH


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


# --- ink: the fourth leg of (task, tool, substrate, ink) -----------------------------

def ink_registry():
    """``ink_spec``: which dips happen, and the charge model behind them."""
    return _ink_spec


def dip_motion():
    """``dip_motion``: the cap entry geometry the arm uses, so the simulator
    hovers and plunges along the same axis rather than its own idea of down."""
    import dip_motion

    return dip_motion


def active_ink_policy():
    """The fitted tool's ``ink:`` block: real (3RL), rehearsal (ballpoint),
    or none (laser)."""
    return ink_registry().policy_for(active_tool())


def palette():
    from tatbot_sim.palette import load

    return load(REPO).palette


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
    return ink.supply_load(kind, palette(), ink_id, REPO)
