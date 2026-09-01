"""Repo-root resolution for the simulator — TATBOT_REPO, else this checkout.

The simulator reads the URDF, tool registry, and inkmap lexicon from the
repo. TATBOT_REPO points elsewhere (tests, temporary clones); the fallback
is the editable-install location of this file (plan Phase 1).
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("TATBOT_REPO")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[4]
