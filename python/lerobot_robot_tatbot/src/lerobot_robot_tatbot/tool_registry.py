"""The fitted tool's datasheet, from inside the arm plugin.

``scripts/lib/tool_spec.py`` is the single implementation — the plugin is a
separate uv project, so it loads that file by path rather than vendoring a copy
that would drift. Same shim as ``tatbot_sim.tools``; the registry is
stdlib-only precisely so this needs no new dependency.

Until this existed the plugin was entirely tool-unaware. Since 2026-08-30 the
tool sits in a mount rather than the gripper, so what the plugin asks the
registry for is the mount (a tool without one refuses to connect) and the
cross-check against the calibration — no grip force any more.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from functools import lru_cache

from .paths import repo_root

logger = logging.getLogger(__name__)

REPO = repo_root()
_MODULE_NAME = "tatbot_tool_spec"
_MODULE_PATH = REPO / "scripts" / "lib" / "tool_spec.py"


@lru_cache(maxsize=1)
def registry():
    """The tool_spec module, or None when it is not reachable.

    Returning None rather than raising: a bench session with no repo checkout
    around it should still connect, just without a tool cross-check.
    """
    if not _MODULE_PATH.is_file():
        logger.warning("tool registry not found at %s; grip falls back to config",
                       _MODULE_PATH)
        return None
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    if spec is None or spec.loader is None:
        logger.warning("could not load tool registry spec from %s; grip falls back to config",
                       _MODULE_PATH)
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module          # dataclasses resolve annotations here
    spec.loader.exec_module(module)
    return module


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
