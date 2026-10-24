"""The rig's power state: `power` records in config/nodes.json and the sleep marker.

A rig node is any node whose config/nodes.json record carries a ``power``
object. `tatbot rig sleep` acts on those nodes only, in the modes each record
lists, and leaves a marker (``~/tatbot-logs/rig/state.json``) on every node it
touched, on the node it ran from and, where they answer, on the operator
nodes. `tatbot status` shows the marker and the CLI refuses motion verbs and
the sleeping roles' verbs while it says ``asleep`` — the fix is `tatbot rig
wake`, not a retry.

    "power": {"sleep": ["services"]}
    "power": {"sleep": ["window"]}

- ``services``: the node's manifested ``services`` units. Sleep stops the ones
  running, displays (part ``station``, the demo cockpit) first; it never stops
  the bus router or its presence reporter (the units carrying ROUTER_KEY), which
  the drawing stack it leaves running holds a session to. Wake starts the
  enabled ones in stages: the router until its key answers on the bus, then the
  rest until each is active, on the bus and not restarting, then the displays
  and the node's display ``hardware`` units (the viewer hub, the load monitor).
- ``window``: the node's demo viewer window. Sleep closes it; wake opens it
  with WINDOW_OPEN (a window that runs is left as it is), and only over the
  node's direct cable to the hub where it has one: a window on the rig port
  delays the arm controller's packets.

A display or a window is best-effort: it never fails a sleep or a wake, so it
cannot keep the gate that refuses motion. A window node's marker is another
matter: a wake that cannot clear it names the node and exits nonzero. A node that relays the e-stop never
sleeps, and nothing here moves an arm, stops the drawing stack or cuts power.
Pure: no ssh, no subprocess. The orchestration lives in scripts/lib/rig_power.py.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "tatbot.rig-state/1"
SLEEP_MODES = ("services", "window")
# Roles whose verbs need what sleep switches off (the D555 owner, the stencil and
# tool trackers); the CLI refuses them while the marker says asleep (motion tiers
# are refused regardless).
SLEEPING_ROLES = ("overhead-depth", "track")
# The bus router's own liveliness key, held by its presence reporter. Sleep never
# stops the units that carry it (the drawing stack's bridge holds a session to the
# router all night); wake starts them first, in case one died, and waits for the
# key before anything that dials the bus.
ROUTER_KEY = "zenohd"
# What a window node runs at wake; station/bin/station-launch opens the same at boot. No --restart: a
# window that runs is left as it is, so a wake on an awake rig does not replay the hub's buffer into it.
WINDOW_OPEN = ("--memory-limit", "4GB", "--fullscreen")
WINDOW_UNIT = "tatbot-viewer-window.service"  # scripts/vision/rerun_session.sh VIEWER_WINDOW_UNIT
# The ROS 2 drawing stack (tatbot_cli.tatbot_ros.UNITS): `tatbot ros up|down` own it, sleep never does.
DRAWING_STACK = ("tatbot-ros-router.service", "tatbot-ros.service")


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


def power_record(rec: dict) -> list[str] | None:
    """A node's validated sleep modes, or None for a node sleep never touches."""
    power = rec.get("power")
    if power is None:
        return None
    if not isinstance(power, dict) or set(power) != {"sleep"}:
        raise ValueError("power must be an object with `sleep` alone")
    modes = power["sleep"]
    if not isinstance(modes, list) or not modes or any(m not in SLEEP_MODES for m in modes):
        raise ValueError(f"power.sleep must list some of {SLEEP_MODES}")
    roles = rec.get("roles", [])
    if "estop-relay" in roles:
        raise ValueError("the e-stop relay never sleeps: its silence stops and latches a running arm")
    if "window" in modes and "rerun-server" in roles:
        raise ValueError("power.sleep window closes a window, and on the rerun-server node `viewer stop` stops the hub")
    return list(modes)


def rig_nodes(nmap: dict) -> dict[str, list[str]]:
    """node -> validated sleep modes, in config order."""
    out = {}
    for name, rec in nmap.items():
        try:
            modes = power_record(rec)
        except ValueError as exc:
            raise ValueError(f"config/nodes.json {name}: {exc}") from exc
        if modes:
            out[name] = modes
    return out


def _hardware_units(rec: dict) -> tuple[list[str], list[str], list[str]]:
    """(system units, user units, display system units) of a node's `hardware` list,
    a template `name@.service` as the login user's instance (as `tatbot rig check` expects it)."""
    user = (rec.get("ssh") or "").split("@")[0]
    system, users, display = [], [], []
    for h in rec.get("hardware") or []:
        unit = (h.get("unit") or h.get("user_unit") or "").replace("@.service", f"@{user}.service")
        if not unit:
            continue
        (system if "unit" in h else users).append(unit)
        if "unit" in h and h.get("part") == "station":
            display.append(unit)
    return system, users, display


def plan(nmap: dict) -> dict[str, dict]:
    """What sleep and wake do per rig node, from the map alone (no probing).

    units: what sleep stops, in order (displays first); never the bus.
    bus, start, display: wake's three stages, each in manifest order.
    ensure: display `hardware` units sleep leaves running, started at wake in case one died.
    liveliness: unit -> the `<node>/<key>` wake requires on the bus once it runs.
    kept, kept_user: what sleep leaves running there.
    """
    out = {}
    for name, modes in rig_nodes(nmap).items():
        rec = nmap[name]
        services = [s for s in rec.get("services", []) if isinstance(s, dict) and s.get("unit")] \
            if "services" in modes else []
        display = [s["unit"] for s in services if s.get("part") == "station"]
        bus = [s["unit"] for s in services if s.get("liveliness") == ROUTER_KEY and s["unit"] not in display]
        start = [s["unit"] for s in services if s["unit"] not in display and s["unit"] not in bus]
        system, users, shown = _hardware_units(rec)
        window = "window" in modes
        stack = list(DRAWING_STACK) if "ros" in rec.get("roles", []) else []
        out[name] = {"checkout": rec.get("checkout"), "units": display + list(reversed(start)),
                     "bus": bus, "start": start, "display": display, "ensure": shown if services else [],
                     "liveliness": {s["unit"]: f"{name}/{s['liveliness']}" for s in services if s.get("liveliness")},
                     "window": window, "kept": list(dict.fromkeys(bus + system + stack)),
                     "kept_user": [u for u in users if not (window and u == WINDOW_UNIT)]}
    return out


def off_order(nmap: dict, me: str | None) -> list[str]:
    """The nodes `tatbot rig off` shuts down (``"off": true`` in config/nodes.json),
    in shutdown order: the arm and camera hosts first, the e-stop relay after
    them (it is the stop path while an arm host is up), and the node running the
    command last, so it can report the others before it goes."""
    names = [n for n, rec in nmap.items() if isinstance(rec, dict) and rec.get("off") is True]
    return sorted(names, key=lambda n: (n == me, "estop-relay" in nmap[n].get("roles", [])))
