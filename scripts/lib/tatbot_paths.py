"""Central filesystem-root resolution for every tatbot tool.

A fresh clone
must run with a clean environment from any location, so no tool may hardcode
the author's home directory or checkout path. Every root resolves through
here, with ONE precedence order, most explicit first:

    1. CLI flag           (the caller passes it in; never guessed here)
    2. TATBOT_* env var
    3. user config        ~/.config/tatbot/  (XDG_CONFIG_HOME honored)
    4. repo config        <repo>/config/
    5. XDG default        state/data dirs under the XDG base-dir spec

`~/tatbot-logs` is the private rig's convention, supplied by its repo config
layer (config/runlog.json) — it is deliberately NOT a default in this file.

STDLIB ONLY, ON PURPOSE — same contract as tatbot_runlog.py: four disjoint
interpreters load this by path and none can install packages into the others.
"""

from __future__ import annotations

import os
from pathlib import Path


class PathConfigError(RuntimeError):
    """A required root is missing or invalid; message says how to supply it."""


def _xdg(env: str, fallback: str) -> Path:
    base = os.environ.get(env, "").strip()
    return Path(base) if base else Path.home() / fallback


def repo_root() -> Path:
    """The checkout this file lives in, or TATBOT_REPO when set."""
    env = os.environ.get("TATBOT_REPO")
    if env:
        p = Path(env).expanduser().resolve()
        if not (p / "config").is_dir():
            raise PathConfigError(
                f"TATBOT_REPO={env} is not a tatbot checkout (no config/ inside)")
        return p
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """Repo configuration; TATBOT_CONFIG_DIR overrides for tests/profiles."""
    env = os.environ.get("TATBOT_CONFIG_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise PathConfigError(f"TATBOT_CONFIG_DIR={env} does not exist")
        return p
    return repo_root() / "config"


def user_config_dir() -> Path:
    """Per-node operator overrides (~/.config/tatbot); may not exist."""
    return _xdg("XDG_CONFIG_HOME", ".config") / "tatbot"


def state_root() -> Path:
    """Default writable state root (XDG state dir); created on demand."""
    return _xdg("XDG_STATE_HOME", ".local/state") / "tatbot"


def data_root() -> Path:
    """Default writable data root (XDG data dir); created on demand."""
    return _xdg("XDG_DATA_HOME", ".local/share") / "tatbot"


def log_root(configured: str | None = None) -> Path:
    """Run-log root: TATBOT_LOG_ROOT > configured (runlog config) > XDG state.

    `configured` is the value the caller read from its config layers
    (config/runlog.json, then ~/.config/tatbot/runlog.json) — the private rig
    sets "~/tatbot-logs" there. A clean public clone lands in XDG state.
    """
    env = os.environ.get("TATBOT_LOG_ROOT")
    if env:
        return Path(env).expanduser()
    if configured:
        return Path(configured).expanduser()
    return state_root() / "logs"


def output_dir(cli_value: str | os.PathLike | None, *, purpose: str,
               env: str | None = None, default: Path | None = None) -> Path:
    """An output root a tool writes into. Explicit beats implicit:

    CLI value > `env` var (when named) > `default` (when the tool has a safe
    one, e.g. under data_root()). With none of those, this raises with a
    message naming the flag — a tool must never silently write to the
    current directory or /tmp.
    """
    if cli_value:
        return Path(cli_value).expanduser()
    if env:
        v = os.environ.get(env)
        if v:
            return Path(v).expanduser()
    if default is not None:
        return default
    hint = f" or {env}" if env else ""
    raise PathConfigError(f"no output directory for {purpose}: pass the flag{hint}")
