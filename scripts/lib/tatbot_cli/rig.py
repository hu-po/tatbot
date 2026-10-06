"""The rig's power state: `power` records in config/nodes.json and the sleep marker.

A rig node is any node whose config/nodes.json record carries a ``power``
object. `tatbot rig sleep` acts on those nodes only, in the ways each record
allows, and leaves a marker (``~/tatbot-logs/rig/state.json``) on every node it
touched and on the node it ran from. `tatbot status` shows the marker and the
CLI refuses hardware verbs while it says ``asleep`` — the fix is `tatbot rig
wake`, not a retry.

    "power": {"sleep": ["services", "suspend"], "wake": "wol", "mac": "…"}
    "power": {"sleep": ["services", "suspend"], "wake": "rtc", "units": ["extra.service"]}

- ``services``: stop the node's manifested ``services`` units (plus ``units``)
  in reverse order at sleep; start them in order at wake.
- ``suspend``: suspend the host to RAM. A ``wol`` node is woken by a magic
  packet to ``mac`` from a node on the rig LAN; an ``rtc`` node can only wake
  itself from its clock, so it is suspended only when `rig sleep --wake-at` is
  given and otherwise sleeps its services alone.

Pure: no ssh, no subprocess. The orchestration lives in scripts/lib/rig_power.py.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "tatbot.rig-state/1"
SLEEP_MODES = ("services", "suspend")
WAKE_MODES = ("wol", "rtc")
# Roles whose verbs need hardware that sleep switches off; the CLI refuses
# them while the marker says asleep (motion tiers are refused regardless).
SLEEPING_ROLES = ("arm", "estop", "realsense", "poe-cameras", "overhead-depth", "rerun-server")
_MAC = re.compile(r"^([0-9a-f]{2}:){5}[0-9a-f]{2}$")


def marker_path() -> Path:
    override = os.environ.get("TATBOT_RIG_STATE")
    return Path(override).expanduser() if override else Path("~/tatbot-logs/rig/state.json").expanduser()


def read_state(path: Path | None = None) -> dict | None:
    """The marker, or None where none was written. A malformed marker raises."""
    path = path or marker_path()
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"unsupported rig state in {path}")
    if data.get("state") not in ("asleep", "awake"):
        raise ValueError(f"invalid rig state {data.get('state')!r} in {path}")
    return data


def write_state(state: dict, path: Path | None = None) -> Path:
    path = path or marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}")
    tmp.write_text(json.dumps({"schema": SCHEMA, **state}, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def asleep(path: Path | None = None) -> dict | None:
    """The marker when it says asleep; None when awake, absent, or unreadable
    (a broken marker must not turn into a refusal of every hardware verb)."""
    try:
        state = read_state(path)
    except (OSError, ValueError):
        return None
    return state if state and state["state"] == "asleep" else None


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def power_record(rec: dict) -> dict | None:
    """A node's validated `power` record, or None for a node sleep never touches."""
    power = rec.get("power")
    if power is None:
        return None
    if not isinstance(power, dict):
        raise ValueError("power must be an object")
    modes = power.get("sleep")
    if not isinstance(modes, list) or not modes or any(m not in SLEEP_MODES for m in modes):
        raise ValueError(f"power.sleep must list some of {SLEEP_MODES}")
    wake = power.get("wake")
    if "suspend" in modes:
        if wake not in WAKE_MODES:
            raise ValueError(f"a node that suspends needs power.wake in {WAKE_MODES}")
        if wake == "wol" and not _MAC.match(str(power.get("mac", "")).lower()):
            raise ValueError("power.wake=wol needs power.mac as aa:bb:cc:dd:ee:ff")
    elif wake is not None:
        raise ValueError("power.wake is meaningful only with sleep mode suspend")
    extra = power.get("units", [])
    if not isinstance(extra, list) or any(not re.fullmatch(r"tatbot[-\w@.]+\.service", u) for u in extra):
        raise ValueError("power.units must name tatbot*.service units")
    return {"sleep": list(modes), "wake": wake, "mac": (power.get("mac") or "").lower() or None,
            "units": list(extra)}


def rig_nodes(nmap: dict) -> dict[str, dict]:
    """node -> validated power record, in config order."""
    out = {}
    for name, rec in nmap.items():
        try:
            power = power_record(rec)
        except ValueError as exc:
            raise ValueError(f"config/nodes.json {name}: {exc}") from exc
        if power:
            out[name] = power
    return out


def plan(nmap: dict, *, hosts: bool = True, wake_at: str | None = None) -> dict[str, dict]:
    """What sleep does per rig node, from the map alone (no probing).

    units: stop order at sleep (the reverse is the start order at wake).
    suspend: whether the host is suspended; `note` says why not when it could be.
    """
    out = {}
    for name, power in rig_nodes(nmap).items():
        rec = nmap[name]
        units = [s["unit"] for s in rec.get("services", []) if isinstance(s, dict) and s.get("unit")]
        units = list(dict.fromkeys([*(units if "services" in power["sleep"] else []), *power["units"]]))
        suspend, note = False, None
        if "suspend" in power["sleep"]:
            if not hosts:
                note = "--no-hosts: services only"
            elif power["wake"] == "rtc" and not wake_at:
                note = "wakes only from its own clock: suspended only with --wake-at"
            else:
                suspend = True
        out[name] = {"ssh": rec.get("ssh"), "checkout": rec.get("checkout"),
                     "units": list(reversed(units)), "suspend": suspend, "wake": power["wake"] if suspend else None, "mac": power["mac"], "note": note}
    return out


def magic_packet(mac: str) -> bytes:
    mac = mac.lower()
    if not _MAC.match(mac):
        raise ValueError(f"not a MAC address: {mac}")
    return b"\xff" * 6 + bytes.fromhex(mac.replace(":", "")) * 16


def parse_wake_at(text: str, *, now_unix: float | None = None) -> int:
    """`HH:MM` (the next occurrence, local time) or an absolute epoch second."""
    import time as _time
    if re.fullmatch(r"\d{9,10}", text):
        return int(text)
    m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", text)
    if not m:
        raise ValueError("--wake-at takes HH:MM (local, next occurrence) or a unix epoch second")
    base = now_unix if now_unix is not None else _time.time()
    local = _time.localtime(base)
    candidate = _time.mktime((local.tm_year, local.tm_mon, local.tm_mday, int(m.group(1)), int(m.group(2)), 0,
                              local.tm_wday, local.tm_yday, -1))
    if candidate <= base + 60:
        candidate += 86400
    return int(candidate)
