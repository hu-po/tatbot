"""Read-only, point-in-time advisory lock observations (not execution gates)."""

from __future__ import annotations

import fcntl
import os
import stat
from pathlib import Path


def held(path: Path) -> bool:
    """Never create/truncate/remove a lock file; propagate inaccessible state."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"not a regular lock file: {path}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)
