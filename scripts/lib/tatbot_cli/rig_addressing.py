"""Rig addressing check: every rig-LAN address agrees with config/nodes.json.

`__rig__` in config/nodes.json names the rig's own subnet and gateway. The
addresses that have to live in it are spread over files that different tools
read — the PoE camera list in `rust/visiond/config/vision.toml`, the arm
controllers in the hardware driver profile and in `config/trossen*/` (which
`wxai_teleop` writes into the controllers' EEPROM), the `ssh_lan`/`lan`
addresses the bulk paths dial. None of those files may carry an address the
map does not, so renumbering the rig is one edit to nodes.json followed by
this check naming every file that still disagrees. `scripts/check nodes` runs
it; a checkout without `__rig__` skips it.
"""

from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path

from tatbot_cli import nodes

VISION_TOML = Path("rust/visiond/config/vision.toml")
PROFILES = Path("config/profiles")
CONTROLLER_YAMLS = ("leader.yaml", "follower.yaml")


def _addr(value: str) -> ipaddress.IPv4Address | None:
    try:
        return ipaddress.IPv4Address(value.strip())
    except ValueError:
        return None


class _Subnets:
    """The rig subnet plus, during a renumbering, the one it is leaving."""

    def __init__(self, stanza: dict):
        self.current = ipaddress.IPv4Network(stanza.get("subnet", ""), strict=True)
        old = stanza.get("migrating_from")
        self.old = ipaddress.IPv4Network(old, strict=True) if old else None

    def __contains__(self, addr: ipaddress.IPv4Address) -> bool:
        return addr in self.current or (self.old is not None and addr in self.old)

    def __str__(self) -> str:
        return f"{self.current} (or {self.old}, still migrating)" if self.old else str(self.current)


def _in(subnet, value: str) -> bool:
    addr = _addr(value)
    return addr is not None and addr in subnet


class Site:
    """One address-carrying field somewhere under the repo."""

    __slots__ = ("label", "value", "role")

    def __init__(self, label: str, value: str, role: str | None = None):
        self.label, self.value, self.role = label, value, role


def _profiles(repo: Path):
    """(rel_path, profile) for every hardware profile that parses."""
    if not (repo / PROFILES).is_dir():
        return
    for path in sorted((repo / PROFILES).glob("*.json")):
        try:
            profile = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if profile.get("hardware") is True:
            yield path.relative_to(repo), profile


def _controller_yamls(repo: Path):
    """(rel_path, role, parsed) for every controller YAML in every config set —
    an alternate set (a battery-specific golden) is loaded into the same two
    controllers and must name the same addresses."""
    for ydir in sorted(p for p in repo.glob("config/trossen*") if p.is_dir()):
        for yaml_name in CONTROLLER_YAMLS:
            ypath = ydir / yaml_name
            if ypath.is_file():
                yield ypath.relative_to(repo), yaml_name.split(".")[0], controller_yaml(ypath.read_text())


def sites(repo: Path) -> list[Site]:
    """Every field that has to hold a rig-LAN address, in reporting order.
    `check` holds each to the subnet; `remaining` lists the ones still on the
    old one. Both walk this list so neither can forget a file."""
    out: list[Site] = []
    for name, addr in nodes.cameras(repo).items():
        out.append(Site(f"__cameras__.{name}", addr))
    for node, rec in nodes.load(repo).items():
        for key in ("ssh_lan", "lan"):
            if rec.get(key):
                out.append(Site(f"{node}.{key}", str(rec[key]).split("@")[-1]))
    vision = repo / VISION_TOML
    if vision.is_file():
        for name, addr in vision_cameras(vision.read_text()).items():
            out.append(Site(f"{VISION_TOML}: {name}", addr))
    for rel, profile in _profiles(repo):
        out += _profile_sites(rel, profile)
    for yrel, role, y in _controller_yamls(repo):
        out += [Site(f"{yrel}: {key}", y[key], role if key == "manual_ip" else None)
                for key in ("manual_ip", "gateway", "dns") if y.get(key)]
    return out


def _profile_sites(rel: Path, profile: dict) -> list[Site]:
    out = []
    driver = profile.get("driver") or {}
    for field, role in (("leader_ip", "leader"), ("follower_ip", "follower")):
        if driver.get(field):  # null in the scrubbed public profile: nothing to hold
            out.append(Site(f"{rel}: driver.{field}", str(driver[field]), role))
    for key, value in (profile.get("endpoints") or {}).items():
        if not key.startswith("//") and isinstance(value, str) and ":" in value:
            out.append(Site(f"{rel}: endpoints.{key}", value.rsplit(":", 1)[0]))
    return out


def remaining(repo: Path) -> list[str]:
    """What still sits on `__rig__.migrating_from`: the renumbering to-do list."""
    old = nodes.rig(repo).get("migrating_from")
    if not old:
        return []
    stale = _Subnets({"subnet": old})
    return [f"{site.label} {site.value}" for site in sites(repo) if _in(stale, site.value)]


def vision_cameras(text: str) -> dict[str, str]:
    """`name` → `address` of every `[[cameras.poe]]` block, plus `ntp_server`
    under the key `ntp_server`. Regex, not a TOML parser: the arm and camera
    nodes run Python 3.10 (no tomllib) and the CLI is stdlib only."""
    out: dict[str, str] = {}
    m = re.search(r"^ntp_server\s*=\s*\"([^\"]*)\"", text, re.MULTILINE)
    if m:
        out["ntp_server"] = m.group(1)
    for block in re.split(r"^\[\[cameras\.poe\]\]\s*$", text, flags=re.MULTILINE)[1:]:
        head = re.split(r"^\[", block, maxsplit=1, flags=re.MULTILINE)[0]
        name = re.search(r"^name\s*=\s*\"([^\"]*)\"", head, re.MULTILINE)
        addr = re.search(r"^address\s*=\s*\"([^\"]*)\"", head, re.MULTILINE)
        if name and addr:
            out[name.group(1)] = addr.group(1)
    return out


def controller_yaml(text: str) -> dict[str, str]:
    """Top-level `key: value` scalars of a Trossen controller YAML."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([a-z_]+):\s*([^\s#]+)\s*(#.*)?$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def check(repo: Path) -> tuple[list[str], str | None]:
    """(problems, skip_reason). While `__rig__.migrating_from` names an old
    subnet, an address there is not a problem — `remaining()` lists those, so
    the renumbering can land device by device with the check green."""
    stanza = nodes.rig(repo)
    if not stanza:
        return [], "no __rig__ stanza in config/nodes.json (no rig subnet described)"
    try:
        subnet = _Subnets(stanza)
    except ValueError as e:
        return [f"__rig__: {e}"], None
    gateway = str(stanza.get("gateway", ""))
    problems = [f"__rig__.gateway {gateway!r} is not in {subnet}"] if not _in(subnet, gateway) else []
    # Membership: every site, one rule. A controller's gateway is held to the
    # rig gateway below instead, and may still point at the old one mid-move.
    for site in sites(repo):
        if site.label.endswith(": gateway"):
            continue
        if not _in(subnet, site.value):
            problems.append(f"{site.label} {site.value} is not in {subnet}")
    problems += _vision_agrees(repo)
    arms, disagreements = _profile_arms(repo)
    problems += disagreements
    problems += _controllers_agree(repo, arms, gateway, subnet)
    return problems, None


def _vision_agrees(repo: Path) -> list[str]:
    """vision.toml names the same cameras at the same addresses as __cameras__."""
    vision = repo / VISION_TOML
    if not vision.is_file():
        return []
    cams = nodes.cameras(repo)
    seen = vision_cameras(vision.read_text())
    seen.pop("ntp_server", None)
    out = []
    for name, addr in seen.items():
        if name not in cams:
            out.append(f"{VISION_TOML}: {name} is not in __cameras__")
        elif cams[name] != addr:
            out.append(f"{VISION_TOML}: {name} is {addr}, __cameras__ says {cams[name]}")
    return out


def _profile_arms(repo: Path) -> tuple[dict[str, str], list[str]]:
    """role -> controller address every hardware profile agrees on, plus the
    profiles that disagree."""
    arms: dict[str, str] = {}
    out = []
    for site in sites(repo):
        if site.role and "driver." in site.label and arms.setdefault(site.role, site.value) != site.value:
            out.append(f"{site.label} {site.value} disagrees with another hardware profile ({arms[site.role]})")
    return arms, out


def _controllers_agree(repo: Path, arms: dict[str, str], gateway: str, subnet: _Subnets) -> list[str]:
    """Each controller YAML drives its arm at the profile's address through the rig gateway."""
    out = []
    for yrel, role, y in _controller_yamls(repo):
        want = arms.get(role)
        if want and y.get("manual_ip") != want:
            out.append(f"{yrel}: manual_ip {y.get('manual_ip')} but the profile drives the {role} at {want}")
        if y.get("gateway") and y["gateway"] != gateway and not (subnet.old and _in(subnet.old, y["gateway"])):
            out.append(f"{yrel}: gateway {y['gateway']} but __rig__.gateway is {gateway}")
    return out
