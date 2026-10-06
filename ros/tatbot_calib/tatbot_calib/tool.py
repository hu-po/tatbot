"""The tool a probe calibration touches the station with, refused before anything moves when it cannot be one.

The tool is the arm's fitted tool: config/workspace.yaml `<arm>.tool_id`, whose tip the stack's tcp is built from.
It is never a default. A stated tool must be that one: the stack plans every touch through the fitted tool's tip,
so calibrating another tool against it would plan with the wrong tip, tens of millimetres off between two tools.
Its datasheet must declare the contact model the touches are planned for (`calibration.face_kind`): a tool
without one is refused, never probed on a guess.
"""
from __future__ import annotations

import sys
from pathlib import Path

from tatbot_calib import halo


class ToolRefusedError(RuntimeError):
    """The calibration must not start with this tool: a gate refusal (the CLI's exit 3)."""


def fitted_tool(repo: Path, arm: str, stated: str | None = None):
    """(the fitted tool's datasheet, its halo) for `arm` from the checkout `repo`, or ToolRefusedError."""
    from tatbot_description import repo_root

    lib = str(repo_root(None) / "scripts" / "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    import tool_spec

    tool_id = tool_spec.active_tool_id(repo, arm)
    if not tool_id:
        raise ToolRefusedError(f"config/workspace.yaml names no tool_id for the {arm} arm: the fitted tool is unknown")
    if stated and stated != tool_id:
        raise ToolRefusedError(f"{stated!r} is stated, but config/workspace.yaml fits {tool_id!r} on the {arm} arm "
                               "and the stack plans every touch through that tool's tip. For a new tool, record it "
                               f"under `{arm}:` there with its datasheet's nominal tip, deploy, and calibrate again")
    tool = tool_spec.load_tool(tool_id, repo)
    try:
        return tool, halo.Halo.from_datasheet(tool.raw.get("calibration") or {}, tool.profile)
    except ValueError as error:
        raise ToolRefusedError(f"{tool_id}: {error}: its datasheet declares no contact model for the station "
                               "probe, so no touch can be planned for it") from None
