"""The palette lease: one arm's run holds the palette, and the other arm's goals keep out of it (2026-09-30: the two
arms share one stack, the pink one drawing while the blue one calibrates at the palette; before, pink's inspection
views came near the palette unchecked).

The holder (tatbot_calib's run) takes an exclusive flock on LEASE_PATH, one file on the node whatever stack or lane
runs there, since the palette is one, and writes the zone it keeps: a cylinder in the camera's world, standing on
the table's normal about every part of the palette. The session node reads it at each goal (held_zone): a lock it
can share means no holder. The kernel drops the lock with the holder's process, so a crashed run leaves no lease.
Pure Python.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

LEASE_PATH = Path("/tmp/tatbot-palette.lease")


class PaletteLease:
    """The palette held by `arm`'s run until close(): {"arm", "run", "world_from_zone" (4x4, its z the table's
    normal, its origin on the table under the zone's axis), "radius_m", "height_m"}. RuntimeError when another run
    holds it."""

    def __init__(self, zone: dict, path: Path = LEASE_PATH):
        self.path = Path(path)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = _read(self._fd)
            os.close(self._fd)
            self._fd = None
            raise RuntimeError(f"the palette is held by the {holder.get('arm', '?')} arm's run "
                               f"{holder.get('run', '?')}; wait for it to end") from None
        os.ftruncate(self._fd, 0)
        os.pwrite(self._fd, json.dumps(zone).encode(), 0)
        os.fsync(self._fd)

    def close(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> PaletteLease:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def held_zone(path: Path = LEASE_PATH) -> dict | None:
    """The zone of the run that holds the palette, or None when none does."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return _read(fd) or None
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def _read(fd: int) -> dict:
    try:
        return json.loads(os.pread(fd, 65536, 0).decode() or "{}")
    except (OSError, ValueError):
        return {}


# A dip writes this marker before its tool goes into a cap and removes it once the tool is back over the rim. It
# outlives the session (a crash, a cancel that holds in the cap, a reboot): while it stands, the arm is not landed,
# since the driver's staged landing would sweep the tool sideways through the cap's wall, and the run's next draw
# withdraws along the cap axis first (tatbot_session.cap.withdraw_after_restart).
IN_CAP_DIR = Path.home() / ".local/state/tatbot"


def _in_cap_path(arm: str) -> Path:
    return IN_CAP_DIR / f"in-cap-{arm}.json"


def mark_in_cap(arm: str, record: dict) -> None:
    IN_CAP_DIR.mkdir(parents=True, exist_ok=True)
    path = _in_cap_path(arm)
    path.with_suffix(".tmp").write_text(json.dumps(record))
    os.replace(path.with_suffix(".tmp"), path)


def clear_in_cap(arm: str) -> None:
    _in_cap_path(arm).unlink(missing_ok=True)


def in_cap(arm: str) -> dict | None:
    """The marker of `arm`'s dip whose tool may still be in a cap, or None."""
    try:
        return json.loads(_in_cap_path(arm).read_text())
    except FileNotFoundError:
        return None


def in_cap_refusal(arm: str) -> str | None:
    marker = in_cap(arm)
    if marker is None:
        return None
    return (f"the {arm} arm's tool may be in cap {marker.get('slot')} (run {marker.get('run')}, op {marker.get('op')}): "
            f"resume that run to withdraw it along the cap axis, or lift it out by hand and remove {_in_cap_path(arm)}")
