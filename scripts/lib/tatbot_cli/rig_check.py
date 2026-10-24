"""What `tatbot rig check` expects of each demo-case host, and how it judges what it finds.

Expectations come from the files that already own them: the roles and
`services` in config/nodes.json, the wrist cameras' serials and owners in
vision.toml, the controller addresses in the hardware profile, and a node's
`hardware` list for what nothing else records (a touchscreen, a relay unit).
Pure: no ssh, no subprocess. The probing lives in scripts/lib/rig_check.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tatbot_cli import nodes

SCHEMA = "tatbot.rig-check/1"

# Which part of the demo a role serves. A node's general health (clock, disk,
# links) is reported under the first part its roles or hardware name.
PARTS = ("pink drawing", "station", "blue arm")
ROLE_PART = {"ros": "pink drawing", "overhead-depth": "pink drawing", "bus-router": "pink drawing",
             "track": "pink drawing", "estop-relay": "pink drawing",
             "station": "station", "rig-clock": "station", "arm": "blue arm", "estop": "blue arm"}
ARM_PART = {"right": "pink drawing", "left": "blue arm"}

# A D405's USB serial descriptor is not the serial vision.toml names (librealsense
# reads that from the device), so a wrist camera is found by its product id.
D405_USB_ID = "8086:0b5b"
USB3_MBPS = 5000
LINK_MBPS = 1000
JUMBO_MTU = 9000
CLOCK_SKEW_S = 2.0
DISK_FREE_FRACTION = 0.10
HOT_C = 85.0


def demo_hosts(nmap: dict) -> list[str]:
    """The hosts in the demo case: every node `tatbot rig off` shuts down."""
    return [n for n, rec in nmap.items() if not n.startswith("__") and isinstance(rec, dict) and rec.get("off")]


def realsense_cameras(text: str) -> list[dict]:
    """`name`, `serial`, `arm`, `owner_role` of every USB `[[cameras.realsense]]`
    block in vision.toml (a `transport = "dds"` camera is on the network, not USB).
    Regex, not tomllib: the Thors run Python 3.10."""
    out = []
    for block in re.split(r"^\[\[cameras\.realsense\]\]\s*$", text, flags=re.MULTILINE)[1:]:
        head = re.split(r"^\[", block, maxsplit=1, flags=re.MULTILINE)[0]
        fields = dict(re.findall(r"^(\w+)\s*=\s*\"([^\"]*)\"", head, re.MULTILINE))
        if fields.get("serial") and fields.get("owner_role") and fields.get("transport") != "dds":
            out.append({k: fields.get(k) for k in ("name", "serial", "arm", "owner_role")})
    return out


def _profile_driver(repo: Path) -> dict:
    path = repo / "config/profiles/tatbot.json"
    try:
        return json.loads(path.read_text()).get("driver", {})
    except (OSError, ValueError):
        return {}


HARDWARE_KINDS = {"usb": ("usb", "id"), "dev": ("devs", "path"), "unit": ("units", "unit"),
                  "user_unit": ("user_units", "unit")}


def _hardware(exp: dict, hw: list[dict], user: str) -> None:
    """A template unit (`name@.service`) is expected as the node's login-user
    instance, the way `tatbot viewer install` enables it."""
    for h in hw:
        kind = next(k for k in HARDWARE_KINDS if k in h)
        key, field = HARDWARE_KINDS[kind]
        value = h[kind].lower() if kind == "usb" else h[kind].replace("@.service", f"@{user}.service")
        exp[key].append({"name": h["name"], "part": h.get("part", exp["part"]), field: value})


def _peers(roles: list[str], part: str, *, lan: bool, gateway, driver: dict, overhead) -> list[dict]:
    """Addresses the node itself must reach: what its roles talk to."""
    want = []
    if lan and gateway:
        want.append(("rig router", gateway, part))
    if "ros" in roles:
        want.append(("pink controller", driver.get("follower_ip"), "pink drawing"))
    if "arm" in roles:
        want += [("blue controller", driver.get("leader_ip"), "blue arm"),
                 ("pink controller", driver.get("follower_ip"), "blue arm")]
    if "overhead-depth" in roles:
        want.append(("overhead D555", overhead, "pink drawing"))
    return [{"name": n, "address": a, "part": p} for n, a, p in want if a]


def expectations(repo: Path, nmap: dict) -> dict[str, dict]:
    """node -> {"part", "usb", "devs", "units", "user_units", "peers", "lan", "jumbo"}."""
    vision = repo / "rust/visiond/config/vision.toml"
    cams = realsense_cameras(vision.read_text()) if vision.exists() else []
    ctx = {"gateway": nodes.rig(repo).get("gateway"), "driver": _profile_driver(repo),
           "overhead": nodes.cameras(repo).get("overhead_depth")}
    out = {}
    for name in demo_hosts(nmap):
        rec, roles = nmap[name], nodes.roles_of(nmap, name)
        hw = rec.get("hardware") or []
        parts = [ROLE_PART[r] for r in roles if r in ROLE_PART] + [h["part"] for h in hw if h.get("part")]
        part = next((p for p in PARTS if p in parts), "station")
        # A manifested service serves its node's part unless it names its own (a display: `station`).
        units = [(s["unit"], s.get("part", part)) for s in rec.get("services") or []]
        exp = {"part": part, "devs": [], "user_units": [], "lan": rec.get("lan"), "jumbo": "overhead-depth" in roles,
               "usb": [{"name": f"{c['arm']} wrist D405", "id": D405_USB_ID, "min_mbps": USB3_MBPS,
                        "part": ARM_PART.get(c["arm"], part)} for c in cams if c["owner_role"] in roles],
               "units": [{"name": u, "unit": u, "part": p} for u, p in units],
               "peers": _peers(roles, part, lan=bool(rec.get("lan")), **ctx)}
        if "estop-relay" in roles and (line := _estop_gpio(repo)):
            exp["estop_gpio"] = {"line": line, "part": ROLE_PART["estop-relay"]}
        _hardware(exp, hw, (rec.get("ssh") or "").split("@")[0])
        out[name] = exp
    return out


def _estop_gpio(repo: Path) -> str | None:
    """The drawing e-stop's GPIO line on the relay Pi (stack.yaml estop.relay_gpio), e.g. GPIO17."""
    path = repo / "ros/tatbot_bringup/config/stack.yaml"
    found = re.search(r"^estop:.*?^\s+relay_gpio:\s*(GPIO\d+)", path.read_text(), re.M | re.S) if path.exists() else None
    return found.group(1) if found else None


def _estop(node: str, exp: dict, facts: dict) -> list[dict]:
    """The e-stop line, sampled for two seconds: held closed (LOW) is released; HIGH is pressed or a broken wire;
    both is a loose contact (2026-10-16: a wire loosened in the move flipped it a dozen times a minute, and two
    setups failed on it while this check said READY)."""
    want = exp.get("estop_gpio")
    if not want:
        return []
    levels = facts.get("estop_line") or []
    if not levels:
        return [_r(want["part"], node, "warn", f"the e-stop line ({want['line']}) on {node} could not be read")]
    lows = levels.count("lo")
    if lows == len(levels):
        return [_r(want["part"], node, "ok", f"e-stop released, its line steady for {len(levels)} reads")]
    if lows == 0:
        return [_r(want["part"], node, "fail", "e-stop pressed, or its wire open: twist the button out, check its "
                                               "two wires on the Pi")]
    return [_r(want["part"], node, "fail", f"e-stop line flickering ({len(levels) - lows} of {len(levels)} reads "
                                           "open): a loose wire on the button or the Pi's header")]


def _r(part: str, node: str, status: str, what: str) -> dict:
    return {"part": part, "node": node, "status": status, "what": what}


def _host(node: str, part: str, facts: dict, local_unix: float | None) -> list[dict]:
    out = [_r(part, node, "ok", f"{node} up {int(facts.get('uptime_s', 0)) // 60} min")]
    warn = []
    if facts.get("ntp_synced") is False:
        warn.append(f"{node} clock not NTP-synced (boots on its last shutdown time)")
    skew = facts["unix"] - local_unix if local_unix is not None and facts.get("unix") is not None else 0.0
    if abs(skew) > CLOCK_SKEW_S:
        warn.append(f"{node} clock {skew:+.1f} s from this node's")
    disk = facts.get("disk") or {}
    if disk.get("total") and disk.get("free", 0) / disk["total"] < DISK_FREE_FRACTION:
        warn.append(f"{node} disk {disk['free'] / 1e9:.0f} GB free of {disk['total'] / 1e9:.0f} GB")
    if (facts.get("temp_c") or 0) >= HOT_C:
        warn.append(f"{node} running hot: {facts['temp_c']:.0f} C")
    return out + [_r(part, node, "warn", w) for w in warn]


def _failed_units(node: str, part: str, exp: dict, facts: dict) -> list[dict]:
    """Failed units nothing expects, as warnings; an expected unit fails under its own part (`_present`)."""
    expected = {u["unit"] for key in ("units", "user_units") for u in exp[key]}
    return [_r(part, node, "warn", f"{node} unit failed: {u}") for u in facts.get("failed_units") or []
            if u not in expected]


def _rig_link(node: str, part: str, exp: dict, ifaces: dict) -> list[dict]:
    iface = next((i for i, f in ifaces.items() if exp["lan"] in f.get("ipv4", [])), None)
    if iface is None:
        return [_r(part, node, "fail", f"{node} has no rig address {exp['lan']}")]
    f = ifaces[iface]
    speed = f.get("speed")
    if speed is not None and 0 < speed < LINK_MBPS:
        out = [_r(part, node, "warn", f"{node} rig link {iface} at {speed} Mb/s (expect {LINK_MBPS}; cable or switch port)")]
    else:
        out = [_r(part, node, "ok", f"{node} rig link {iface} {speed} Mb/s")]
    if exp.get("jumbo") and (f.get("mtu") or 0) < JUMBO_MTU:
        out.append(_r(part, node, "fail", f"{node} rig link MTU {f.get('mtu')}; the D555 needs {JUMBO_MTU}"))
    return out


def _links(node: str, part: str, exp: dict, facts: dict) -> list[dict]:
    ifaces = facts.get("ifaces") or {}
    out = _rig_link(node, part, exp, ifaces) if exp.get("lan") else []
    for prof in facts.get("nm_profiles") or []:
        f = ifaces.get(prof["iface"])
        if f is not None and not f.get("carrier"):
            out.append(_r(part, node, "warn", f"{node} {prof['name']}: {prof['iface']} has no link (other end off or unplugged?)"))
        elif f is not None and f.get("speed"):
            out.append(_r(part, node, "ok", f"{node} {prof['name']}: {prof['iface']} {f['speed']} Mb/s"))
    return out


def _usb(node: str, exp: dict, facts: dict) -> list[dict]:
    usb, out = list(facts.get("usb") or []), []
    for want in exp["usb"]:
        hit = next((d for d in usb if d.get("id") == want["id"]), None)
        if hit is None:
            out.append(_r(want["part"], node, "fail", f"{want['name']} not on {node}'s USB (unplugged, or its hub/screen unpowered)"))
            continue
        usb.remove(hit)  # two of a kind on one node need two devices
        if want.get("min_mbps") and (hit.get("mbps") or 0) < want["min_mbps"]:
            out.append(_r(want["part"], node, "warn", f"{want['name']} on {node} at {hit.get('mbps')} Mb/s, not USB 3"))
        else:
            out.append(_r(want["part"], node, "ok", f"{want['name']} on {node}"))
    return out


def _present(node: str, exp: dict, facts: dict) -> list[dict]:
    out = []
    for want in exp["devs"]:
        ok = (facts.get("devs") or {}).get(want["path"])
        out.append(_r(want["part"], node, "ok" if ok else "fail",
                      f"{want['name']} {want['path']} on {node}" + ("" if ok else " missing")))
    for key in ("units", "user_units"):
        for want in exp[key]:
            s = (facts.get(key) or {}).get(want["unit"], {})
            if s.get("active") != "active":
                out.append(_r(want["part"], node, "fail", f"{want['name']} on {node} is {s.get('active', 'not loaded')}"))
                continue
            note = f" (restarted {s['restarts']}x)" if s.get("restarts") else ""
            out.append(_r(want["part"], node, "ok", f"{want['name']} running on {node}{note}"))
    for want in exp["peers"]:
        ok = (facts.get("pings") or {}).get(want["address"])
        out.append(_r(want["part"], node, "ok" if ok else "fail",
                      f"{want['name']} {want['address']} " + ("answers" if ok else "does not answer") + f" from {node}"))
    return out


def judge(node: str, exp: dict, facts: dict | None, *, error: str | None = None,
          local_unix: float | None = None) -> list[dict]:
    """Results for one node from what its probe found. `facts` None = unreachable."""
    part = exp["part"]
    if facts is None:
        what = f"{node}'s probe failed: {error}" if error else f"{node} does not answer ssh (still booting, off, or a Thor waiting for its power button)"
        served = {part, *(i["part"] for k in ("usb", "devs", "units", "user_units", "peers") for i in exp[k])}
        return [_r(p, node, "fail", what) for p in sorted(served, key=PARTS.index)]
    return (_host(node, part, facts, local_unix) + _failed_units(node, part, exp, facts)
            + _links(node, part, exp, facts) + _usb(node, exp, facts) + _present(node, exp, facts)
            + _estop(node, exp, facts))


def verdicts(results: list[dict]) -> dict[str, dict]:
    """part -> {"ready", "fail", "warn", "ok"} for every part any result names."""
    out = {}
    for part in PARTS:
        mine = [r for r in results if r["part"] == part]
        if mine:
            counts = {s: sum(1 for r in mine if r["status"] == s) for s in ("fail", "warn", "ok")}
            out[part] = {"ready": counts["fail"] == 0, **counts}
    return out
