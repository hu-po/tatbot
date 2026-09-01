"""config/nodes.json — the machine-readable node→role map, and the --on hop."""

from __future__ import annotations

import json
import os
import shlex
import socket
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


def ssh_target(nodes: dict, node: str) -> str | None:
    return nodes.get(node, {}).get("ssh")


def host_of(nodes: dict, node: str) -> str | None:
    """The address part of a node's ssh target (its Tailscale IP)."""
    t = ssh_target(nodes, node)
    return t.split("@")[-1] if t else None


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
    remote = (f"cd {checkout} && {pull}{env_prefix}scripts/tatbot --no-hop "
              + " ".join(shlex.quote(a) for a in argv))
    cmd = ["ssh", "-o", "BatchMode=yes"]
    if tty:
        cmd.append("-t")
    # A login shell: an `ssh host cmd` shell is neither login nor interactive,
    # so ~/.profile is never read and `uv` (~/.local/bin on every node) is not
    # on PATH — a hopped launcher then dies with exit 127 AFTER its gates ran
    # and its nonce was consumed (observed 2026-08-29).
    cmd += [target, "bash -lc " + shlex.quote(remote)]
    return cmd


def example_node(role: str | None = None) -> str:
    """A node name for a verb's registered example.

    The first node carrying `role` in config/nodes.json, else the first node
    at all, else a `<node>` placeholder — so fleet hostnames stay in the
    deployment's config instead of being frozen into source (the selfcheck
    skips dry-running an example that still holds a placeholder).
    """
    from tatbot_cli.registry import repo_root

    nmap = load(repo_root())
    if role:
        for name, rec in nmap.items():
            if role in rec.get("roles", []):
                return name
    return next(iter(nmap), "<node>")
