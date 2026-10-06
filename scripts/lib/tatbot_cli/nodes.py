"""config/nodes.json — the machine-readable node→role map, and the --on hop."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shlex
import socket
import subprocess
from pathlib import Path


def load(repo: Path) -> dict:
    """The node map, or {} where no fleet is described (a public clone has
    no config/nodes.json — single-machine use needs no map; see
    config/examples/nodes.json to describe one)."""
    path = repo / "config" / "nodes.json"
    if not path.is_file():
        return {}
    with open(path) as fh:
        data = json.load(fh)
    return {k: v for k, v in data.items() if not k.startswith("//") and not k.startswith("__")}


def cameras(repo: Path) -> dict[str, str]:
    path = repo / "config" / "nodes.json"
    if not path.is_file():
        return {}
    with open(path) as fh:
        return dict(json.load(fh).get("__cameras__", {}))


def rig(repo: Path) -> dict:
    """The `__rig__` stanza (subnet, gateway) — the rig's own address space — or {}."""
    path = repo / "config" / "nodes.json"
    if not path.is_file():
        return {}
    with open(path) as fh:
        stanza = json.load(fh).get("__rig__", {})
    return dict(stanza) if isinstance(stanza, dict) else {}


def rig_subnet(repo: Path) -> ipaddress.IPv4Network | None:
    """The rig subnet as a network, or None where the fleet map does not name one."""
    subnet = rig(repo).get("subnet")
    if not subnet:
        return None
    return ipaddress.IPv4Network(subnet, strict=False)


def repo_root() -> Path:
    from tatbot_paths import repo_root as _root
    return _root()


def remote_checkout(nodes: dict, node: str) -> str:
    """`node`'s checkout as a remote shell sees it: `~/x` becomes `$HOME/x` so
    it expands inside a quoted command sent over ssh, where a tilde would not."""
    checkout = nodes[node].get("checkout") or "~/tatbot"
    return "$HOME" + checkout[1:] if checkout.startswith("~") else checkout


def this_node(nodes: dict | None = None) -> str:
    """TATBOT_NODE, else `hostname -s` mapped through any `hostname` alias in nodes.json."""
    env = os.environ.get("TATBOT_NODE")
    if env:
        return env
    host = socket.gethostname().split(".")[0]
    for name, rec in (nodes or {}).items():
        if rec.get("hostname") == host:
            return name
    return host


def roles_of(nodes: dict, node: str) -> list[str]:
    return list(nodes.get(node, {}).get("roles", []))


def nodes_with(nodes: dict, role: str) -> list[str]:
    return [n for n, rec in nodes.items() if role in rec.get("roles", [])]


def require_role(nmap: dict, role: str) -> str:
    """Resolve a singleton fleet role; never guess when the map is ambiguous."""
    matches = nodes_with(nmap, role)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {role} node; found {len(matches)}")
    return matches[0]


def bus_endpoint(nmap: dict, *, address: str = "ssh") -> str:
    """The bus-router endpoint. Services use LAN; CLI clients use the SSH host."""
    node = require_role(nmap, "bus-router")
    if address not in ("ssh", "lan"):
        raise ValueError(f"unsupported bus address kind: {address}")
    host = host_of(nmap, node) if address == "ssh" else nmap[node].get("lan")
    if not isinstance(host, str) or not host.strip():
        raise ValueError(f"bus-router node has no {address} address")
    return f"tcp/{host}:7447"


def lan_ip(repo: Path | None = None) -> str | None:
    """This host's address on the rig subnet (`__rig__` in config/nodes.json), or
    TATBOT_RERUN_LAN_IP; None without one. A checkout that names no rig subnet
    (a public clone) gets its first RFC 1918 address instead, so a
    single-machine setup still publishes something reachable."""
    forced = os.environ.get("TATBOT_RERUN_LAN_IP")
    if forced:
        return forced
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return pick_lan_ip(out, rig_subnet(repo or repo_root()))


def pick_lan_ip(ip_addr_output: str, subnet: ipaddress.IPv4Network | None) -> str | None:
    """The first address in `subnet` from `ip -4 -o addr` output; with no subnet,
    the first private (RFC 1918) one — which excludes loopback, link-local and
    the tailnet's shared address space. Pure, so tests can feed it text."""
    for line in ip_addr_output.splitlines():
        m = re.search(r"\binet (\d+\.\d+\.\d+\.\d+)/", line)
        if not m:
            continue
        try:
            addr = ipaddress.IPv4Address(m.group(1))
        except ValueError:
            continue
        if subnet is not None:
            if addr in subnet:
                return str(addr)
        elif addr.is_private and not addr.is_loopback and not addr.is_link_local:
            return str(addr)
    return None


def rerun_server(nodes: dict) -> tuple[str, str] | None:
    """(node, LAN ip) of the fleet Rerun viewer: the one node with role `rerun-server` and a `lan` address."""
    for name, rec in nodes.items():
        if "rerun-server" in rec.get("roles", []) and rec.get("lan"):
            return name, rec["lan"]
    return None


def rerun_proxy(nodes: dict, port: int = 9876) -> str | None:
    """Where producers stream: TATBOT_RERUN_CONNECT, else the fleet viewer's proxy, else None."""
    forced = os.environ.get("TATBOT_RERUN_CONNECT")
    if forced:
        return forced
    server = rerun_server(nodes)
    return f"rerun+http://{server[1]}:{port}/proxy" if server else None


def ssh_target(nodes: dict, node: str) -> str | None:
    return nodes.get(node, {}).get("ssh")


def host_of(nodes: dict, node: str) -> str | None:
    """The address part of a node's ssh target (its Tailscale IP)."""
    t = ssh_target(nodes, node)
    return t.split("@")[-1] if t else None


def _quote_remote(arg: str) -> str:
    """shlex.quote, except a leading `~/` stays bare so the remote shell expands it.

    cli._home_relative rewrites this node's $HOME paths to `~/…` for the hop;
    shlex.quote would single-quote the tilde and the remote tool would open a
    literal '~/tatbot-logs/…' (teleop analyze did, 2026-09-03)."""
    if arg.startswith("~/"):
        return "~/" + shlex.quote(arg[2:]) if len(arg) > 2 else "~/"
    return shlex.quote(arg)


def hop_argv(nodes: dict, node: str, argv: list[str], *, tty: bool, sync: bool = False) -> list[str]:
    """The ssh command that re-runs this exact invocation on `node`.

    The remote runs the checkout's own shim so the schema, gates and run-log
    wiring are the remote node's; `--no-hop` stops a second hop even when the
    remote copy of config/nodes.json disagrees about roles.
    """
    target = ssh_target(nodes, node)
    if not target:
        raise KeyError(node)
    # Every private node states its checkout explicitly in config/nodes.json;
    # the fallback is the public repo's conventional path (plan Phase 1).
    checkout = nodes[node].get("checkout") or "~/tatbot"
    # sync: bring the remote checkout to origin/main first (fast-forward only, so a
    # diverged or conflicting tree fails loudly instead of running stale code).
    pull = "git pull -q --ff-only origin main && " if sync else ""
    # An explicitly chosen hardware profile must cross the hop: without this a
    # remote command silently resolved the REMOTE node's default profile, so
    # `TATBOT_PROFILE=example tatbot --on <node> ...` looked like it was
    # testing a synthetic profile while using the rig's (found in the
    # 2026-08-31 parity session). A profile can only ever be less capable than
    # the default, so forwarding it cannot widen what the remote may do.
    import os as _os

    profile = _os.environ.get("TATBOT_PROFILE", "").strip()
    env_prefix = f"TATBOT_PROFILE={shlex.quote(profile)} " if profile else ""
    # The fiducial tracker is configured by environment (EE_TRACKING_*, TATBOT_VISIOND_*);
    # its wrapper used to forward 31 of them through its own ssh, which the CLI hop
    # replaced on 2026-09-02. Forward that allowlist, and nothing else.
    for key, value in sorted(_os.environ.items()):
        if key.startswith(("EE_TRACKING_", "TATBOT_VISIOND_")) and value:
            env_prefix += f"{key}={shlex.quote(value)} "
    remote = (f"cd {_quote_remote(checkout)} && {pull}{env_prefix}scripts/tatbot --no-hop "
              + " ".join(_quote_remote(a) for a in argv))
    cmd = ["ssh", "-o", "BatchMode=yes"]
    if tty:
        cmd.append("-t")
    # A login shell: an `ssh host cmd` shell is neither login nor interactive,
    # so ~/.profile is never read and `uv` (~/.local/bin on every node) is not
    # on PATH — a hopped launcher then dies with exit 127 AFTER its gates ran
    # and its launch id was ledgered (observed 2026-08-29).
    cmd += [target, "bash -lc " + shlex.quote(remote)]
    return cmd


def example_node(role: str | None = None) -> str:
    """A node name for a verb's registered example.

    The first node carrying `role` in config/nodes.json, else the first node
    with any role (a retired node keeps its record but no roles), else a
    `<node>` placeholder — so fleet hostnames stay in the deployment's config
    instead of being frozen into source (the selfcheck skips dry-running an
    example that still holds a placeholder).
    """
    from tatbot_cli.registry import repo_root

    nmap = load(repo_root())
    if role:
        for name, rec in nmap.items():
            if role in rec.get("roles", []):
                return name
    return next((name for name, rec in nmap.items() if rec.get("roles")), "<node>")
