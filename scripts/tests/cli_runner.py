"""Run the stdlib `tatbot` CLI the way its tests do: one subprocess, one strict environment.

Ten test modules each carried their own copy of this, differing only by
accident — the default node, a 15/20/30/60 s timeout, whether a stray
TATBOT_EE_TOOL in the caller's shell was allowed to leak into a run. One
runner, one set of defaults; a module that needs another node passes it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tatbot_cli import nodes

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "scripts" / "lib" / "tatbot_cli"


def tatbot(*args: str, node: str | None = None, env: dict | None = None, timeout: float = 60,
           isolated: bool = False) -> subprocess.CompletedProcess:
    """Invoke `tatbot` as `node` (default: the operator example node) and return the process.

    TATBOT_TRAIN_ROOT points nowhere so a training tree on the host cannot make
    a verb look available, and TATBOT_EE_TOOL is dropped so a tool statement in
    the caller's shell cannot pass for one the test made. `env` wins over both.
    `isolated` runs the interpreter with -S: the CLI's own stdlib-only claim.
    """
    e = dict(os.environ, TATBOT_NODE=node or nodes.example_node("operator"), TATBOT_TRAIN_ROOT="/nonexistent")
    e.pop("TATBOT_EE_TOOL", None)
    e.update(env or {})
    argv = [sys.executable, *(["-S"] if isolated else []), str(CLI), *args]
    return subprocess.run(argv, capture_output=True, text=True, env=e, timeout=timeout)
