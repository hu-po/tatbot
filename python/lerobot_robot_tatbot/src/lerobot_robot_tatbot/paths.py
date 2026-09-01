"""Repo/log-root resolution for the plugin — one place, env-overridable.

The plugin depends on the repo checkout for the URDF, golden configs, and
tool registry. TATBOT_REPO overrides where that checkout is; the fallback is
the editable-install location of this file. Flight logs resolve
TATBOT_LOG_ROOT > the checkout's config/runlog.json log_root > the XDG state
dir — never a hardcoded home path (plan Phase 1).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("TATBOT_REPO")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def log_root() -> Path:
    env = os.environ.get("TATBOT_LOG_ROOT")
    if env:
        return Path(env).expanduser()
    try:
        cfg = json.loads((repo_root() / "config" / "runlog.json").read_text())
        configured = cfg.get("log_root")
    except (OSError, json.JSONDecodeError):
        configured = None
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".local/state"
    return base / "tatbot" / "logs"


def profile_driver() -> dict:
    """The resolved hardware profile's driver stanza, or {}.

    Same resolution as the CLI's profile gate (TATBOT_PROFILE, else the
    private tatbot profile when its file exists). Config dataclass defaults
    pull addresses and the e-stop device from here, so the rig behaves as
    before while a public clone — with no profile file — defaults to empty
    values and fails closed at connect (plan Phase 2).
    """
    name = os.environ.get("TATBOT_PROFILE", "").strip() or "tatbot"
    path = repo_root() / "config" / "profiles" / f"{name}.json"
    try:
        p = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        # An EXISTING but unreadable profile is a broken deployment, not a
        # public clone — degrade loudly (audit 2026-08-31, finding 3).
        import logging

        logging.getLogger(__name__).warning("hardware profile %s unreadable: %s", path, e)
        return {}
    driver = p.get("driver")
    return driver if isinstance(driver, dict) and p.get("hardware") is True else {}


def driver_default(key: str, env: str) -> str:
    """Default for an address/device field: env override > profile > ''."""
    v = os.environ.get(env, "").strip()
    if v:
        return v
    return str(profile_driver().get(key) or "")


def flight_dir(configured: str) -> Path | None:
    """Directory for flight CSVs: '' disables, 'auto:<workflow>' resolves to
    log_root()/<workflow>, anything else is an explicit path used as-is."""
    if not configured:
        return None
    if configured.startswith("auto:"):
        return log_root() / configured.split(":", 1)[1]
    return Path(os.path.expanduser(configured))
