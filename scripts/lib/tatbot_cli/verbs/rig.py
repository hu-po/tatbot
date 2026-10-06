"""rig — sleep and wake the physical rig: its camera and bus services and its hosts."""

from __future__ import annotations

from tatbot_cli.registry import REMOTE, SENSOR, verb
from tatbot_cli.verbs._common import py

BACKEND = "scripts/lib/rig_power.py"
DOC = "docs/rig.md"


def _backend(ctx, *args):
    return py(ctx, BACKEND, *(["--json"] if ctx.json else []), *args)


def _sleep_args(p):
    p.add_argument("--no-hosts", action="store_true", help="stop services only; suspend no host")
    p.add_argument("--wake-at", metavar="HH:MM",
                   help="arm every suspended host's clock alarm for this local time; the only way a node "
                        "without Wake-on-LAN gets suspended")
    p.add_argument("--plan", action="store_true", help="print what sleep would do to each node, without ssh")


@verb(effects=("read_files", "network", "remote_exec", "stop_process", "write_files"), noun="rig", verb="sleep", tier=REMOTE,
      output="json", args=_sleep_args, wraps=(BACKEND,), example=("--plan",), doc=DOC,
      summary="stop the camera and bus services and suspend the rig hosts for the night",
      invariants=("Never moves an arm or cuts arm power: refuses (exit 6) while an arm workflow runs, and leaves "
                  "the arms landed and idle as their last session left them.",
                  "Only nodes with a `power` record in config/nodes.json are touched, only in the ways it lists.",
                  "A node that can only wake from its own clock is suspended only with --wake-at.",
                  "Leaves a marker every hardware verb refuses on (exit 5) until `tatbot rig wake`."))
def rig_sleep(ctx, ns, rest):
    args = ["sleep"]
    if ns.no_hosts:
        args.append("--no-hosts")
    if ns.wake_at:
        args += ["--wake-at", ns.wake_at]
    if ns.plan:
        args.append("--plan")
    return _backend(ctx, *args, *rest)


@verb(effects=("read_files", "network", "remote_exec", "start_process", "write_files"), noun="rig", verb="wake", tier=REMOTE,
      output="json", wraps=(BACKEND,), example=(), doc=DOC,
      summary="wake the rig hosts, start the camera and bus services, verify they stay up",
      invariants=("Wake packets are sent from a node on the rig LAN; a host that only wakes from its clock is reported, not woken.",
                  "Every manifested service is started in manifest order and judged active and not restarting before the marker clears.",
                  "Starts no arm process."))
def rig_wake(ctx, ns, rest):
    return _backend(ctx, "wake", *rest)


@verb(effects=("read_files", "network", "remote_exec"), noun="rig", verb="status", tier=SENSOR,
      output="json", wraps=(BACKEND,), example=(), doc=DOC,
      summary="the sleep marker, and each rig node's reachability and service state")
def rig_status(ctx, ns, rest):
    return _backend(ctx, "status", *rest)
