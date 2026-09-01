"""Interpreter selection that used to live in scripts/dataset_hub.sh (phase 3 absorb).

The sim host has no lerobot venv and the training nodes have theirs under
~/il-train, so pick whichever known environment can import huggingface_hub
and fall back to a throwaway `uv run --with` environment. Same order, same
fallback, same failure as the bash it replaces; test_cli.py pins each branch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

HUB_REQUIREMENT = "huggingface_hub[hf_xet]>=1.0"


def _imports(python: Path, module: str) -> bool:
    if not os.access(python, os.X_OK):
        return False
    try:
        return subprocess.run([str(python), "-c", f"import {module}"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=30).returncode == 0
    except Exception:
        return False


def uv_binary() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    home = Path(os.environ.get("HOME", "~")).expanduser() / ".local" / "bin" / "uv"
    return str(home) if os.access(home, os.X_OK) else None


def hub_python(repo: Path, *, env: dict | None = None) -> tuple[list[str] | None, str]:
    """(argv prefix that runs Python with huggingface_hub, note) — or (None, why not)."""
    e = env if env is not None else os.environ
    train_root = Path(e.get("TATBOT_TRAIN_ROOT") or (Path(e.get("HOME", "~")).expanduser() / "il-train"))
    for candidate in (repo / "python/lerobot_robot_tatbot/.venv/bin/python", train_root / ".venv/bin/python"):
        if _imports(candidate, "huggingface_hub"):
            return [str(candidate)], f"{candidate} imports huggingface_hub"
    uv = uv_binary()
    if uv:
        return [uv, "run", "--quiet", "--no-project", "--with", HUB_REQUIREMENT, "python"], "throwaway uv environment"
    return None, "no python with huggingface_hub and no uv on this node"
