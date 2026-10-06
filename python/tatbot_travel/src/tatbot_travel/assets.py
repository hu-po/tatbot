"""Where the rig's checked-in geometry lives.

Nothing here is copied: the arm chain, the laser's datasheet, its measured
touch-off and the body model are read from the tatbot checkout, so a
recalibration there reaches the generator without an edit here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

URDF_RELATIVE = Path("urdf/tatbot.urdf")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """The tatbot checkout: ``TATBOT_REPO`` if set, else the one containing this file."""
    override = os.environ.get("TATBOT_REPO")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / URDF_RELATIVE).is_file():
            raise FileNotFoundError(f"TATBOT_REPO={root} has no {URDF_RELATIVE}")
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / URDF_RELATIVE).is_file():
            return parent
    raise FileNotFoundError("no tatbot checkout above this package; set TATBOT_REPO")


def urdf_path() -> Path:
    return repo_root() / URDF_RELATIVE


def tool_datasheet(tool_id: str) -> Path:
    return repo_root() / "config" / "tools" / f"{tool_id}.yaml"


def body_regions() -> Path:
    return repo_root() / "web" / "inkmap" / "public" / "bodies" / "mhr-soma-v1.regions.json"
