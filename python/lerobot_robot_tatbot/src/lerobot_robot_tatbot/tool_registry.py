"""The fitted tool's datasheet, from inside the arm plugin.

``scripts/lib/tool_spec.py`` is the single implementation — the plugin is a
separate uv project, so it depends on that directory as ``tatbot-scriptlib``
and imports the module normally. Same dependency as ``tatbot_sim.tools``; the
registry is stdlib-only, so it costs the venv nothing.

Until this existed the plugin was entirely tool-unaware. Since 2026-08-30 the
tool sits in a mount rather than the gripper, so what the plugin asks the
registry for is the mount (a tool without one refuses to connect) and the
cross-check against the calibration — no grip force any more.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .paths import repo_root

logger = logging.getLogger(__name__)

REPO = repo_root()


@lru_cache(maxsize=1)
def registry():
    """The tool_spec module, or None when it is not reachable.

    Returning None rather than raising: a bench session with no repo checkout
    around it should still connect, just without a tool cross-check. The
    module is a declared dependency, so absence now means a venv assembled
    without it rather than a missing file.
    """
    try:
        import tool_spec
    except ImportError:
        logger.warning("tool registry not importable; grip falls back to config")
        return None
    return tool_spec


def stated_tool(tool_id, arm: str = "right", context: str = "this run"):
    """The tool the CALLER says is in the mount, cross-checked against the
    calibration in workspace.yaml. Raises rather than guessing.

    Replaced fitted_tool() on 2026-08-26. That read the tool out of
    workspace.yaml, which sounds like the same thing and is not: the file
    records the tool the last TOUCH-OFF was measured with, so after a physical
    swap it names the previous tool, confidently, with a full set of its
    constants attached. Every behaviour that needs tool geometry — teleop,
    recording, rollouts, calibration — now states the tool it is holding, and
    a disagreement is an error instead of a silent substitution.
    """
    reg = registry()
    if reg is None:
        # No registry on this machine. Still refuse a STATED tool we cannot
        # verify, because the caller has told us geometry matters for this run.
        if tool_id:
            raise RuntimeError(
                f"{context}: tool {tool_id!r} was stated but the tool registry "
                f"is unreachable, so nothing can confirm it — refusing rather "
                f"than guessing.")
        return None
    return reg.require_stated_tool(tool_id, REPO, arm, context=context)
