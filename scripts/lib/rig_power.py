#!/usr/bin/env python3
"""`tatbot rig sleep|wake|status` — idle the rig's services and hosts overnight.

Every node this touches is a `power` record in config/nodes.json (see
tatbot_cli.rig). Nothing here moves an arm or cuts arm power: sleep refuses
while an arm workflow runs, and the arms stay landed and idle exactly as their
last session left them. Stdlib only, so it runs from any node's bare clone.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
from tatbot_cli import (  # noqa: E402
    EXIT_BUSY,
    EXIT_HW_UNREACHABLE,
    EXIT_OK,
    EXIT_TOOL_FAILED,
    EXIT_WRONG_NODE,
    nodes,
    rig,
)

# ssh takes the FIRST value it sees for an option, so the connect timeout is
# given per call, never in this prefix.
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
WAKE_SSH_WAIT_S = 180
SETTLE_S = 15
REMOTE_MARKER = "~/tatbot-logs/rig/state.json"

# The magic-packet sender, run where a rig-LAN interface exists (locally or on a
# `cameras-lan` node): every IPv4 broadcast address the host has, plus the
# limited broadcast, three times. Directed UDP would need the target's ARP
# entry, which a suspended host no longer answers.
WOL_SENDER = r"""
import socket, subprocess, sys
mac = sys.argv[1].lower()
packet = b"\xff" * 6 + bytes.fromhex(mac.replace(":", "")) * 16
targets = {"255.255.255.255"}
out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True, check=False).stdout
for line in out.splitlines():
    words = line.split()
    if "brd" in words:
        targets.add(words[words.index("brd") + 1])
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
for _ in range(3):
    for host in sorted(targets):
        sock.sendto(packet, (host, 9))
print("sent", mac, "to", " ".join(sorted(targets)))
"""


class Remote:
    """One node's transport: a fresh, bounded ssh per call, or a local shell for
    the node this command runs on. A node holds no ssh key for itself, so
    probing it over ssh would report the very host running the command as
    asleep; it is plainly reachable, and its shell is right here."""

    def __init__(self, name: str, target: str | None, *, local: bool = False):
        self.name, self.target, self.local = name, target, local

    def run(self, script: str, *, timeout: float = 60, stdin: str | None = None) -> subprocess.CompletedProcess:
        if self.local:
            return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout,
                                  input=stdin, check=False)
        if not self.target:
            raise OSError(f"{self.name} has no ssh target")
        return subprocess.run([*SSH, "-o", "ConnectTimeout=8", self.target, "bash -c " + shlex.quote(script)],
                              capture_output=True, text=True, timeout=timeout, input=stdin, check=False)

    def reachable(self, timeout: float = 6) -> bool:
        if self.local:
            return True
        if not self.target:
            return False
        try:
            r = subprocess.run([*SSH, "-o", f"ConnectTimeout={int(timeout)}", self.target, "true"],
                               capture_output=True, text=True, timeout=timeout + 4, check=False)
        except (subprocess.SubprocessError, OSError):
            return False
        return r.returncode == 0

    def unit_states(self, units: list[str]) -> dict[str, dict]:
        if not units:
            return {}
        r = self.run("systemctl show --no-pager --property=Id,ActiveState,NRestarts " + shlex.join(units))
        if r.returncode:
            raise OSError(f"{self.name}: systemctl show failed: {r.stderr.strip()[:200]}")
        states = {}
        for block in r.stdout.strip().split("\n\n"):
            fields = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
            if fields.get("Id"):
                states[fields["Id"]] = {"active": fields.get("ActiveState", "unknown"),
                                       "restarts": int(fields["NRestarts"]) if fields.get("NRestarts", "").isdigit() else None}
        return states

    def write_marker(self, state: dict) -> None:
        text = json.dumps({"schema": rig.SCHEMA, **state}, indent=2, sort_keys=True) + "\n"
        r = self.run(f"mkdir -p ~/tatbot-logs/rig && cat > {REMOTE_MARKER}.tmp && mv -f {REMOTE_MARKER}.tmp {REMOTE_MARKER}",
                     stdin=text)
        if r.returncode:
            raise OSError(f"{self.name}: marker write failed: {r.stderr.strip()[:200]}")

    def read_marker(self) -> dict | None:
        try:
            r = self.run(f"cat {REMOTE_MARKER} 2>/dev/null || true")
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode or not r.stdout.strip():
            return None
        try:
            data = json.loads(r.stdout)
        except ValueError:
            return None
        return data if isinstance(data, dict) and data.get("schema") == rig.SCHEMA else None


def _remote(nmap: dict, name: str, me: str) -> Remote:
    """`name`'s transport from `me`, the node this command runs on."""
    return Remote(name, nodes.ssh_target(nmap, name), local=name == me)


def _emit(args, payload: dict, text: str) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else text)


def _render(verb: str, report: dict) -> str:
    lines = [f"tatbot rig {verb} — {report.get('state', '')} {report.get('at', '')}".rstrip()]
    for name, node in report["nodes"].items():
        bits = []
        if node.get("units") is not None:
            bits.append(f"units {len(node['units'])} {node.get('units_verdict', '')}".rstrip())
        if node.get("host") is not None:
            bits.append(f"host {node['host']}")
        if node.get("note"):
            bits.append(node["note"])
        if node.get("error"):
            bits.append("ERROR " + node["error"])
        lines.append(f"  {name:<12} " + "  ".join(bits))
    for note in report.get("notes", []):
        lines.append(f"  note {note}")
    return "\n".join(lines)


def _ordered(plan: dict, nmap: dict, *, router_last: bool) -> list[str]:
    """Camera and arm nodes first, the bus router last at sleep (so services can
    deregister) and first at wake (so they have a bus to register on)."""
    routers = [n for n in plan if "bus-router" in nmap.get(n, {}).get("roles", [])]
    others = [n for n in plan if n not in routers]
    return others + routers if router_last else routers + others


def _arm_busy(nmap: dict, plan: dict, me: str) -> tuple[list[str], str | None]:
    """Running arm workflows on the arm node, from its own `tatbot status`."""
    arms = nodes.nodes_with(nmap, "arm")
    if not arms:
        return [], "no arm node in config/nodes.json"
    name = arms[0]
    remote = _remote(nmap, name, me)
    if not remote.reachable():
        return [], f"the arm node {name} did not answer ssh (already asleep?)"
    checkout = nmap[name].get("checkout") or "~/tatbot"
    r = remote.run(f"cd {checkout} && scripts/tatbot --no-hop --json status --no-service-discovery --timeout-s 8", timeout=40)
    try:
        report = json.loads(r.stdout)
        running = report["observations"]["runs_running"]["value"]
    except (ValueError, KeyError, TypeError):
        raise OSError(f"could not read running workflows on {name}: {(r.stderr or r.stdout).strip()[:200]}") from None
    if running is None:
        raise OSError(f"{name} could not enumerate its running workflows; refusing to sleep blind")
    return [f"{row.get('workflow')} {row.get('run_id')}" for row in running], None


def do_plan(args, nmap: dict, plan: dict) -> int:
    report = {"schema": "tatbot.rig-plan/1", "hosts": not args.no_hosts, "wake_at": args.wake_at, "nodes": {}}
    for name, p in plan.items():
        report["nodes"][name] = {"units_stop_order": p["units"], "suspend": p["suspend"], "wake": p["wake"],
                                 "note": p["note"]}
    lines = ["tatbot rig sleep --plan (nothing executed)"]
    for name, p in plan.items():
        bits = [f"stop {len(p['units'])} units" if p["units"] else "no units",
                f"suspend (wake: {p['wake']})" if p["suspend"] else None, p["note"]]
        lines.append(f"  {name:<12} " + "  ".join(b for b in bits if b))
    _emit(args, report, "\n".join(lines))
    return EXIT_OK


def do_sleep(args) -> int:
    nmap = nodes.load(REPO)
    me = nodes.this_node(nmap)
    wake_at = rig.parse_wake_at(args.wake_at) if args.wake_at else None
    plan = rig.plan(nmap, hosts=not args.no_hosts, wake_at=args.wake_at)
    if not plan:
        print("rig sleep: no node in config/nodes.json carries a power record", file=sys.stderr)
        return EXIT_WRONG_NODE
    if args.plan:
        return do_plan(args, nmap, plan)
    if plan.get(me, {}).get("suspend"):
        print(f"rig sleep: {me} would suspend itself mid-way; run this from a node that stays awake "
              "(an operator node), or pass --no-hosts", file=sys.stderr)
        return EXIT_WRONG_NODE
    try:
        running, busy_note = _arm_busy(nmap, plan, me)
    except OSError as exc:
        print(f"rig sleep refused: {exc}", file=sys.stderr)
        return EXIT_HW_UNREACHABLE
    if running:
        print("rig sleep refused: arm workflows are running — land the arms and finish them first:\n  "
              + "\n  ".join(running), file=sys.stderr)
        return EXIT_BUSY

    import tatbot_runlog
    run = tatbot_runlog.init("rig-sleep", argv=sys.argv, meta={"plan": plan, "wake_at": wake_at})
    at = rig.now()
    report = {"schema": "tatbot.rig-sleep/1", "state": "asleep", "at": at, "run_id": run.run_id, "by": me,
              "wake_at": wake_at, "nodes": {}, "notes": [busy_note] if busy_note else []}
    marker = {"state": "asleep", "since": at, "by": me, "run_id": run.run_id, "wake_at": wake_at,
              "hosts": not args.no_hosts, "nodes": {}}
    failed = False
    for name in _ordered(plan, nmap, router_last=True):
        p = plan[name]
        remote = _remote(nmap, name, me)
        node = {"units": None, "host": None, "note": p["note"], "error": None}
        report["nodes"][name] = node
        try:
            if not remote.reachable():
                node["host"] = "unreachable"
                node["note"] = "did not answer ssh: nothing changed there"
                run.event("node.unreachable", node=name)
                continue
            if p["units"]:
                r = remote.run("sudo -n systemctl stop " + shlex.join(p["units"]), timeout=90)
                if r.returncode:
                    raise OSError(f"systemctl stop failed: {r.stderr.strip()[:200]}")
                node["units"] = remote.unit_states(p["units"])
                node["units_expected"] = ("inactive", "failed")
                still = [u for u, s in node["units"].items() if s["active"] not in node["units_expected"]]
                node["units_verdict"] = "stopped" if not still else "STILL RUNNING"
                if still:
                    raise OSError("units still running: " + ", ".join(still))
            marker["nodes"][name] = {"units_stopped": p["units"], "suspended": p["suspend"], "wake": p["wake"],
                                     "mac": p["mac"] if p["suspend"] else None}
            remote.write_marker(marker)
            if p["suspend"]:
                if wake_at:
                    r = remote.run(f"sudo -n rtcwake -m no -t {wake_at}")
                    if r.returncode:
                        raise OSError(f"rtcwake alarm failed: {r.stderr.strip()[:200]}")
                    node["note"] = f"clock alarm armed for {time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(wake_at))}"
                # The kernel's own sleep state, never overridden: a host that does
                # not wake from it is not a `suspend` node (see docs/rig.md).
                node["mem_sleep"] = remote.run("cat /sys/power/mem_sleep").stdout.strip()
                r = remote.run("sudo -n systemd-run --on-active=2 --quiet /usr/bin/systemctl suspend")
                if r.returncode:
                    raise OSError(f"could not schedule suspend: {r.stderr.strip()[:200]}")
                deadline = time.monotonic() + 45
                while remote.reachable(timeout=3) and time.monotonic() < deadline:
                    time.sleep(3)
                node["host"] = "suspended" if time.monotonic() < deadline else "STILL ANSWERING ssh"
                if node["host"] != "suspended":
                    raise OSError("host still answers ssh 45 s after the suspend request")
            else:
                node["host"] = "awake"
            run.event("node.slept", node=name, **{k: v for k, v in node.items() if k != "units"})
        except (OSError, subprocess.SubprocessError) as exc:
            failed = True
            node["error"] = str(exc)
            run.event("node.failed", node=name, error=str(exc))
    rig.write_state(marker)
    (run.dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _emit(args, report, _render("sleep", report))
    run.finalize(EXIT_TOOL_FAILED if failed else EXIT_OK)
    return EXIT_TOOL_FAILED if failed else EXIT_OK


def _wol_sender(nmap: dict, plan: dict, me: str) -> Remote | None:
    """None to send from here (this host is on the rig LAN), else a `cameras-lan`
    node that is not itself asleep."""
    if nodes.lan_ip():
        return None
    for name in nodes.nodes_with(nmap, "cameras-lan"):
        if name == me or plan.get(name, {}).get("suspend"):
            continue
        remote = _remote(nmap, name, me)
        if remote.target and remote.reachable():
            return remote
    raise OSError("no node on the rig LAN can send the wake packet: this host has no rig-LAN address "
                  "and no awake `cameras-lan` node answers ssh")


def _send_wol(sender: Remote | None, mac: str) -> str:
    argv = ["python3", "-c", WOL_SENDER, mac]
    if sender is None:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
    else:
        r = sender.run(shlex.join(argv), timeout=30)
    if r.returncode:
        raise OSError(f"wake packet failed: {(r.stderr or r.stdout).strip()[:200]}")
    return r.stdout.strip()


def do_wake(args) -> int:
    nmap = nodes.load(REPO)
    me = nodes.this_node(nmap)
    plan = rig.plan(nmap, hosts=True, wake_at="any")  # everything a sleep could have done
    if not plan:
        print("rig wake: no node in config/nodes.json carries a power record", file=sys.stderr)
        return EXIT_WRONG_NODE
    import tatbot_runlog
    run = tatbot_runlog.init("rig-wake", argv=sys.argv)
    at = rig.now()
    report = {"schema": "tatbot.rig-wake/1", "state": "awake", "at": at, "run_id": run.run_id, "by": me,
              "nodes": {}, "notes": []}
    previous = rig.asleep()
    if previous:
        report["notes"].append(f"asleep since {previous['since']} (rig sleep {previous.get('run_id')} from {previous.get('by')})")
        if previous.get("wake_failed"):
            report["notes"].append(f"last wake attempt failed: rig wake {previous['wake_failed'].get('run_id')}")
    remotes = {name: _remote(nmap, name, me) for name in plan}
    failed = False
    # 1. Wake every suspended host first, all at once; each takes a minute or more
    # to answer. The node this runs on is awake by definition and never probed.
    asleep_hosts = [n for n, p in plan.items() if "suspend" in rig.rig_nodes(nmap)[n]["sleep"] and not remotes[n].reachable()]
    sender = None
    for name in asleep_hosts:
        node = report["nodes"].setdefault(name, {"units": None, "host": None, "note": None, "error": None})
        power = rig.rig_nodes(nmap)[name]
        if power["wake"] == "wol":
            try:
                if sender is None and not nodes.lan_ip():
                    sender = _wol_sender(nmap, plan, me)
                node["note"] = _send_wol(sender, power["mac"]) + (f" via {sender.name}" if sender else " from here")
                run.event("wol.sent", node=name, via=sender.name if sender else me)
            except (OSError, subprocess.SubprocessError) as exc:
                failed, node["error"] = True, str(exc)
        else:
            node["note"] = "wakes only from its own clock alarm; nothing here can wake it"
    deadline = time.monotonic() + WAKE_SSH_WAIT_S
    pending = [n for n in asleep_hosts if not report["nodes"][n]["error"] and rig.rig_nodes(nmap)[n]["wake"] == "wol"]
    if pending:
        print(f"rig wake: wake packet sent; waiting up to {WAKE_SSH_WAIT_S} s for ssh from " + ", ".join(pending),
              file=sys.stderr, flush=True)
    while pending and time.monotonic() < deadline:
        time.sleep(5)
        pending = [n for n in pending if not remotes[n].reachable(timeout=4)]
    for name in pending:
        failed = True
        report["nodes"][name]["host"] = "unreachable"
        report["nodes"][name]["error"] = (f"no ssh answer {WAKE_SSH_WAIT_S} s after the wake packet: "
                                          "power button, or its --wake-at alarm if one was armed")
    # 2. Restore services, bus router first: every service dials the
    # bus at start and exits when it cannot, so without the router nothing is
    # started — it would only churn through restarts and be judged unhealthy.
    started: dict[str, list[str]] = {}
    slept: dict[str, dict] = {}  # rig node -> the asleep marker it holds, read before wake overwrites it
    router_down = None
    for name in _ordered(plan, nmap, router_last=False):
        p, remote = plan[name], remotes[name]
        node = report["nodes"].setdefault(name, {"units": None, "host": None, "note": None, "error": None})
        if node["error"]:
            if "bus-router" in nmap[name].get("roles", []):
                router_down = name
            continue
        try:
            if router_down and p["units"]:
                raise OSError(f"not started: the bus router {router_down} is not up")
            if not remote.reachable():
                node["host"] = "unreachable"
                if "bus-router" in nmap[name].get("roles", []):
                    router_down = name
                if rig.rig_nodes(nmap)[name]["wake"] != "rtc":
                    raise OSError("did not answer ssh")
                failed = True
                continue
            node["host"] = "awake"
            found = remote.read_marker()
            if found and found.get("state") == "asleep":
                slept[name] = found
            units = list(reversed(p["units"]))
            if units:
                r = remote.run("sudo -n systemctl start " + shlex.join(units), timeout=120)
                if r.returncode:
                    raise OSError(f"systemctl start failed: {r.stderr.strip()[:200]}")
                started[name] = units
        except (OSError, subprocess.SubprocessError) as exc:
            failed, node["error"] = True, str(exc)
            if "bus-router" in nmap[name].get("roles", []):
                router_down = name
            run.event("node.failed", node=name, error=str(exc))
            continue
        run.event("node.woken", node=name, units=units)
    if not previous and slept:
        # No marker here: the sleep ran elsewhere. Its provenance is on the rig nodes.
        previous = next(iter(slept.values()))
        report["notes"].append(f"no marker on {me}; {', '.join(slept)} asleep since {previous.get('since')} "
                               f"(rig sleep {previous.get('run_id')} from {previous.get('by')})")
    # 3. Let the services settle, then judge them: active and not restarting.
    if started:
        time.sleep(SETTLE_S)
    for name, units in started.items():
        node, remote = report["nodes"][name], remotes[name]
        try:
            first = remote.unit_states(units)
            time.sleep(5)
            second = remote.unit_states(units)
            node["units"] = second
            node["units_expected"] = ("active",)
            bad = [u for u in units if second.get(u, {}).get("active") != "active"
                   or (second[u]["restarts"] or 0) > (first.get(u, {}).get("restarts") or 0)]
            node["units_verdict"] = "running" if not bad else "NOT HEALTHY"
            if bad:
                raise OSError("units not healthy after start: " + ", ".join(bad) + f" — `tatbot logs last fleet-service` on {name}")
        except (OSError, subprocess.SubprocessError) as exc:
            failed, node["error"] = True, str(exc)
    # Keep the sleep gate until the whole rig is healthy. A resumed host alone
    # does not establish service readiness, including on this command's node.
    done = {n: {"units_started": started.get(n, [])} for n in plan}

    def still_asleep() -> dict:
        """The gate as sleep left it: when the rig slept, from where and which run
        stay the sleep's own, so a retry's note still names the sleep, not the
        failed wake; the attempt is recorded alongside."""
        kept = {k: v for k, v in (previous or {}).items() if k != "schema"} or \
            {"since": at, "by": me, "run_id": run.run_id, "wake_at": None, "nodes": {}}
        return {**kept, "state": "asleep", "wake_failed": {"run_id": run.run_id, "at": at, "by": me, "nodes": done}}

    marker = still_asleep() if failed else {"state": "awake", "since": at, "by": me, "run_id": run.run_id,
                                            "slept": (previous or {}).get("run_id"), "nodes": done}
    # 4. Publish the marker on every node that holds a gate, not only the rig
    # nodes: sleep also wrote it on the node it ran from, which need not be a rig
    # node nor the node waking now, and that copy refuses its operator's verbs
    # until it is cleared. The sleeper (the `by` of the sleep marker, here or on
    # the rig nodes) is required; every other operator node is best-effort, since
    # a gate there can only come from a sleep it ran, which the sleeper rule covers.
    # This node's own marker is written locally below.
    sleepers = {m["by"] for m in (previous, *slept.values()) if m and m.get("by")}
    targets = {n: True for n in plan if report["nodes"][n].get("host") == "awake"}
    targets.update({n: True for n in sorted(sleepers) if n not in plan and n != me})
    for name in nodes.nodes_with(nmap, "operator"):
        if name not in plan and name != me:
            targets.setdefault(name, False)
    for name, required in targets.items():
        node = report["nodes"].setdefault(name, {"units": None, "host": None, "note": None, "error": None})
        try:
            if name in plan:
                remotes[name].write_marker(marker)
                continue
            if name not in nmap:
                raise OSError("not in config/nodes.json")
            remote = remotes.setdefault(name, _remote(nmap, name, me))
            if not remote.reachable():
                node["host"] = "unreachable"
                raise OSError("did not answer ssh")
            remote.write_marker(marker)
            node["host"] = "awake"
            node["note"] = "marker published" + (" (slept the rig from here)" if required else "")
        except (OSError, subprocess.SubprocessError) as exc:
            if required:
                failed = True
                node["error"] = node["error"] or f"marker: {exc}"
            else:
                node["note"] = f"marker not published: {exc}"
            run.event("marker.failed", node=name, required=required, error=str(exc))
    if failed:
        marker = still_asleep()
        report["state"] = "wake_failed"
    rig.write_state(marker)
    (run.dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _emit(args, report, _render("wake", report))
    code = EXIT_TOOL_FAILED if failed else EXIT_OK
    run.finalize(code)
    return code


def do_status(args) -> int:
    nmap = nodes.load(REPO)
    me = nodes.this_node(nmap)
    plan = rig.plan(nmap, hosts=True, wake_at="any")
    try:
        local = rig.read_state()
    except (OSError, ValueError) as exc:
        local = {"state": "unreadable", "error": str(exc)}
    report = {"schema": "tatbot.rig-status/1", "state": (local or {}).get("state", "no marker"),
              "at": rig.now(), "marker": local, "nodes": {}}
    for name, p in plan.items():
        remote = _remote(nmap, name, me)
        node = {"units": None, "host": None, "note": None, "error": None}
        report["nodes"][name] = node
        if not remote.reachable(timeout=4):
            node["host"] = "unreachable"
            node["note"] = "suspended or off" if "suspend" in rig.rig_nodes(nmap)[name]["sleep"] else None
            continue
        node["host"] = "awake"
        try:
            if p["units"]:
                node["units"] = remote.unit_states(p["units"])
                active = sum(1 for s in node["units"].values() if s["active"] == "active")
                node["units_verdict"] = f"{active}/{len(node['units'])} active"
            marker = remote.read_marker()
            if marker:
                node["note"] = f"marker {marker.get('state')} since {marker.get('since')}"
        except (OSError, subprocess.SubprocessError) as exc:
            node["error"] = str(exc)
    _emit(args, report, _render("status", report))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", action="store_true")
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("sleep", help="stop the rig services, suspend the rig hosts")
    s.add_argument("--no-hosts", action="store_true", help="services only; no host is suspended")
    s.add_argument("--wake-at", metavar="HH:MM", help="arm every suspended host's clock alarm for this local time "
                   "(required to suspend a node that only wakes from its own clock)")
    s.add_argument("--plan", action="store_true", help="print what sleep would do per node; no ssh")
    sub.add_parser("wake", help="wake the hosts, start the services, verify")
    sub.add_parser("status", help="marker, host reachability and unit state per rig node")
    args = p.parse_args(argv)
    if args.verb == "sleep" and args.wake_at:
        try:
            rig.parse_wake_at(args.wake_at)
        except ValueError as exc:
            p.error(str(exc))
    return {"sleep": do_sleep, "wake": do_wake, "status": do_status}[args.verb](args)


if __name__ == "__main__":
    sys.exit(main())
