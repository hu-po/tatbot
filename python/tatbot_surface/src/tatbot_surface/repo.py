"""Repo-root resolution for the surface readers: TATBOT_REPO, else this checkout.

The readers open the body assets and pose catalog in the repo. TATBOT_REPO
points them at another tree (tests, a public validation copy); the fallback is
the editable-install location of this file.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("TATBOT_REPO")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[4]
