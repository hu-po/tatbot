"""Hardware profiles — the single source for what arm this deployment drives.

A profile is a
JSON file under config/profiles/ naming the driver backend, arm addresses,
e-stop device, and limit provenance. Every hardware command must resolve a
profile that passes `hardware_errors()` BEFORE any driver connection; missing
or incomplete profiles fail closed with a nonzero exit and a hint.

Structural rules (decisions 2026-08-31):
- synthetic/example profiles carry ``"hardware": false`` and no driver
  stanza — they can never satisfy the gate by editing one field;
- the scrubbed public Trossen profile is hardware-capable in shape but ships
  with null addresses: a deployment must state its own arms explicitly;
- the private Tatbot profile (config/profiles/tatbot.json) carries the real
  addresses and is the implicit default ONLY where it exists — a public
  clone has no default profile and therefore no implicit hardware path.

Resolution: explicit name/path > $TATBOT_PROFILE > "tatbot" if present.
STDLIB ONLY — loaded by path from every interpreter, like tatbot_runlog.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA = 1
ENV = "TATBOT_PROFILE"
DEFAULT_NAME = "tatbot"  # only used when its file exists (private rig)


class ProfileError(RuntimeError):
    """Profile missing, unreadable, or invalid; message says how to fix it."""


def profiles_dir(repo: Path) -> Path:
    return repo / "config" / "profiles"


def available(repo: Path) -> list[str]:
    d = profiles_dir(repo)
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


def resolve_name(repo: Path, stated: str | None = None) -> str | None:
    """The profile to use, or None when nothing is stated and no default exists."""
    if stated:
        return stated
    env = os.environ.get(ENV, "").strip()
    if env:
        return env
    if (profiles_dir(repo) / f"{DEFAULT_NAME}.json").is_file():
        return DEFAULT_NAME
    return None


def load(repo: Path, stated: str | None = None) -> dict:
    """Load and schema-check a profile; raises ProfileError with guidance."""
    name = resolve_name(repo, stated)
    if name is None:
        known = ", ".join(available(repo)) or "none found"
        raise ProfileError(
            "no hardware profile stated: pass --profile <name> or set "
            f"{ENV}. Known profiles: {known}. Synthetic work can use "
            "'example'; driving an arm needs a completed hardware profile.")
    path = Path(name).expanduser()
    if path.suffix != ".json" or not path.is_file():
        path = profiles_dir(repo) / f"{name}.json"
    if not path.is_file():
        known = ", ".join(available(repo)) or "none found"
        raise ProfileError(f"profile '{name}' not found at {path}. Known: {known}")
    try:
        p = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ProfileError(f"profile {path} unreadable: {e}") from e
    if p.get("schema") != SCHEMA:
        raise ProfileError(f"profile {path}: unsupported schema {p.get('schema')!r}")
    if not p.get("name"):
        raise ProfileError(f"profile {path}: missing name")
    p["_path"] = str(path)
    return p


def hardware_errors(p: dict) -> list[str]:
    """Why this profile may NOT drive hardware; empty list means it may."""
    errs = []
    if p.get("hardware") is not True:
        errs.append("profile is not a hardware profile (hardware != true)")
    driver = p.get("driver")
    if not isinstance(driver, dict):
        errs.append("no driver stanza")
        return errs  # nothing more to check without one
    if not driver.get("backend"):
        errs.append("driver.backend missing")
    for field in ("leader_ip", "follower_ip", "estop_device"):
        if not driver.get(field):
            errs.append(f"driver.{field} not stated — complete the profile "
                        "with your deployment's value")
    arm = p.get("arm") or {}
    if not (arm.get("limits") or {}).get("source"):
        errs.append("arm.limits.source missing (e.g. 'controller')")
    if not arm.get("provenance"):
        errs.append("arm.provenance missing — state where the values came from")
    return errs


def main(argv: list[str]) -> int:
    """`python3 tatbot_profile.py --export` prints eval-able export lines for
    launchers that are run directly rather than through the tatbot CLI. A
    missing or gate-incapable profile prints nothing and exits 1 — the
    launcher's own empty-address check then fails closed."""
    if argv != ["--export"]:
        print(__doc__)
        return 2
    repo = Path(__file__).resolve().parents[2]
    try:
        p = load(repo)
    except ProfileError as e:
        print(f"# tatbot_profile: {e}")
        return 1
    if hardware_errors(p):
        print(f"# tatbot_profile: profile '{p['name']}' is not hardware-capable")
        return 1
    for k, v in env_exports(p).items():
        print(f"export {k}={v!r}")
    return 0


def env_exports(p: dict) -> dict[str, str]:
    """Environment the launchers and drivers consume, from a gated profile."""
    driver = p.get("driver") or {}
    out = {ENV: p["name"]}
    if driver.get("leader_ip"):
        out["TATBOT_LEADER_IP"] = str(driver["leader_ip"])
    if driver.get("follower_ip"):
        out["TATBOT_FOLLOWER_IP"] = str(driver["follower_ip"])
    if driver.get("estop_device"):
        out["TATBOT_ESTOP_DEVICE"] = str(driver["estop_device"])
    endpoints = p.get("endpoints") or {}
    if endpoints.get("teleop_telemetry_udp"):
        out["TATBOT_TELEMETRY_UDP"] = str(endpoints["teleop_telemetry_udp"])
    return out


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
