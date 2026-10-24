"""Which path a node's traffic to the fleet Rerun viewer takes (scripts/vision/rerun_session.sh).

The hub (role `rerun-server`) may share its rig port with the arm controllers,
and a viewer window pulls a heavy stream from it: each depth chunk leaves at
line rate and delays the controller's command packets queued behind it. A node
with a direct cable to the hub therefore dials the hub's address on that cable:
the hub's `links` entry in config/nodes.json, keyed by the peer's role, names
it. The cable is used only while the kernel routes it out another interface
than the rig address and the hub answers there; anything else falls back to
the rig address, so a dead or misrouted cable never costs the window.

    python3 -m tatbot_cli.viewer_path endpoint [PORT]      ADDRESS<TAB>PATH
    python3 -m tatbot_cli.viewer_path describe ADDRESS     PATH
    python3 -m tatbot_cli.viewer_path clients SS ADDRS     one line per hub client

`clients` reads `ss -H -tn state established '( sport = :PORT )'` and
`ip -4 -o addr` output, on the hub.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
from collections import Counter

from tatbot_cli import nodes
from tatbot_cli.registry import repo_root

PORT = 9876
PROBE_S = 1.0  # a TCP connect over a live cable takes well under a millisecond


def route_dev(address: str) -> str | None:
    """The interface the kernel sends to `address` from (`ip route get`), or None."""
    try:
        out = subprocess.run(["ip", "-o", "route", "get", address], capture_output=True, text=True,
                             timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"\bdev (\S+)", out)
    return m.group(1) if m else None


def answers(address: str, port: int) -> bool:
    """True when something accepts a TCP connection at address:port within PROBE_S."""
    try:
        with socket.create_connection((address, port), timeout=PROBE_S):
            return True
    except OSError:
        return False


def link(nmap: dict, node: str) -> tuple[str, str] | None:
    """(role, address) of the hub's direct cable toward `node`: the rerun-server's
    `links` entry for one of `node`'s roles. None on the hub itself or without one."""
    server = nodes.rerun_server(nmap)
    if server is None or server[0] == node:
        return None
    cables = nmap[server[0]].get("links") or {}
    for role in nodes.roles_of(nmap, node):
        if isinstance(cables.get(role), str) and cables[role]:
            return role, cables[role]
    return None


def describe(nmap: dict, node: str, address: str, *, route=route_dev) -> str:
    """What dialing the hub at `address` from `node` rides, in words."""
    dev = route(address)
    over = f" over {dev}" if dev else ""
    cable = link(nmap, node)
    if cable and address == cable[1]:
        return f"direct link to the {cable[0]}{over}"
    server = nodes.rerun_server(nmap)
    if server and address == server[1]:
        return f"rig LAN{over}"
    return f"{address}{over}"


def endpoint(nmap: dict, node: str, port: int = PORT, *, route=route_dev, probe=answers) -> tuple[str, str] | None:
    """(address, path) where `node` dials the hub, or None without a hub: the direct
    link when it routes out another interface than the rig address and the hub
    answers on it, else the rig address with the reason the link was passed over."""
    server = nodes.rerun_server(nmap)
    if server is None:
        return None
    lan = server[1]
    cable = link(nmap, node)
    if cable is None:
        return lan, describe(nmap, node, lan, route=route)
    addr = cable[1]
    dev = route(addr)
    if dev is None or dev == route(lan):
        why = f"the direct link {addr} routes over {dev or 'nothing'}, the rig address's interface"
    elif not probe(addr, port):
        why = f"the hub does not answer on the direct link {addr}:{port}"
    else:
        return addr, describe(nmap, node, addr, route=route)
    return lan, f"{describe(nmap, node, lan, route=route)}; {why}"


def _host(column: str) -> str:
    """The address of an `ss` endpoint column: `192.0.2.1:9876`, `[::ffff:192.0.2.1]:9876`."""
    return column.rsplit(":", 1)[0].strip("[]").removeprefix("::ffff:")


def clients(ss_text: str, addr_text: str, nmap: dict, node: str) -> list[str]:
    """One line per remote peer of the hub on `node` (which address it dialled and
    over what), then how many connections come from `node` itself. Pure."""
    mine = {m.group(2): m.group(1) for m in re.finditer(r"^\s*\d+:\s+(\S+)\s+inet (\d+\.\d+\.\d+\.\d+)/",
                                                         addr_text, re.MULTILINE)}
    rec = nmap.get(node, {})
    cables = {addr: role for role, addr in (rec.get("links") or {}).items()}
    seen: Counter[str] = Counter()
    own = 0
    for line in ss_text.splitlines():
        cols = line.split()
        if len(cols) < 2:
            continue
        local, peer = _host(cols[-2]), _host(cols[-1])
        dev = mine.get(local, local)
        if peer in mine or peer.startswith("127."):
            own += 1
        elif local in cables:
            seen[f"{peer} over the direct link to the {cables[local]} ({dev})"] += 1
        elif local == rec.get("lan"):
            seen[f"{peer} over the rig LAN ({dev})"] += 1
        else:
            seen[f"{peer} over {dev}"] += 1
    out = [f"{what} x{n}" for what, n in sorted(seen.items())]
    return out + ([f"{own} from this node"] if own else [])


def main(argv: list[str]) -> int:
    nmap = nodes.load(repo_root())
    node = nodes.this_node(nmap)
    what = argv[0] if argv else ""
    if what == "endpoint" and len(argv) <= 2:
        found = endpoint(nmap, node, int(argv[1]) if len(argv) == 2 else PORT)
        if found:
            print("\t".join(found))
    elif what == "describe" and len(argv) == 2:
        print(describe(nmap, node, argv[1]))
    elif what == "clients" and len(argv) == 3:
        print("\n".join(clients(argv[1], argv[2], nmap, node)))
    else:
        print("usage: python3 -m tatbot_cli.viewer_path endpoint [PORT] | describe ADDRESS | clients SS ADDRS",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
