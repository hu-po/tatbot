"""The in-cap marker: a dip's tool may still be in a cap. Pure Python."""
from __future__ import annotations

import json
import os
from pathlib import Path

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
