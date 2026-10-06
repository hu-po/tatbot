"""tatbot — the unified CLI.

A façade over the launchers and tools in this repo. It adds a noun-verb
grammar, safety tiers, node routing, ``--dry-run``/``--explain``/``--json``
and a machine-readable schema; it does **not** reimplement the bash safety
libraries (``estop_guard``, ``arm_gate``, ``ink_hook``, ``ee_tool``) — every
hardware verb ``exec``s the launcher that sources them, so they run exactly
as they do when the launcher is called by path.

Standard library only, Python >= 3.10 (the oldest system interpreter in the fleet).
"""

from __future__ import annotations

import sys
from pathlib import Path

# One bare-clone bootstrap for CLI entrypoints and embedding callers. Installed
# simulator distributions resolve tatbot-contracts through package metadata.
_CONTRACTS_SOURCE = Path(__file__).resolve().parents[3] / "python/tatbot_contracts/src"
if _CONTRACTS_SOURCE.is_dir() and str(_CONTRACTS_SOURCE) not in sys.path:
    sys.path.insert(0, str(_CONTRACTS_SOURCE))

# Exit codes — stable, documented in docs/cli.md. Agents branch on these.
EXIT_OK = 0
EXIT_TOOL_FAILED = 1
EXIT_USAGE = 2
EXIT_GATE_REFUSED = 3   # e-stop override, tool not stated, ...
EXIT_WRONG_NODE = 4     # this verb needs a role this node does not have
EXIT_HW_UNREACHABLE = 5 # arm / camera / e-stop not present
EXIT_BUSY = 6           # arm held, training lock, SWEEP_PAUSE

EXIT_NAMES = {
    EXIT_OK: "ok",
    EXIT_TOOL_FAILED: "underlying tool failed",
    EXIT_USAGE: "usage error",
    EXIT_GATE_REFUSED: "safety gate refused",
    EXIT_WRONG_NODE: "wrong node for this verb",
    EXIT_HW_UNREACHABLE: "hardware unreachable",
    EXIT_BUSY: "busy",
}

__version__ = "0.9.0"
