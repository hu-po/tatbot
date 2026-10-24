"""rig — sleep and wake the physical rig: its camera and tracker services and the demo window."""

from __future__ import annotations

from tatbot_cli.registry import REMOTE, SENSOR, verb
from tatbot_cli.verbs._common import py

BACKEND = "scripts/lib/rig_power.py"
DOC = "docs/rig.md"


def _backend(ctx, *args):
    return py(ctx, BACKEND, *(["--json"] if ctx.json else []), *args)


def _plan_arg(p):
    p.add_argument("--plan", action="store_true", help="print what it would do to each node, without ssh")


@verb(effects=("read_files", "network", "remote_exec", "stop_process", "write_files"), noun="rig", verb="sleep", tier=REMOTE,
      output="json", args=_plan_arg, wraps=(BACKEND,), example=("--plan",), doc=DOC,
      summary="stop the camera and tracker services and close the demo viewer window for the night",
      invariants=("Never moves an arm, cuts arm power or stops the drawing stack, its bus or the e-stop relay: "
                  "refuses (exit 3) while the stack holds an arm that has not landed, and (exit 6) while a drawing "
                  "or camera run is open on a rig node, this node or an operator node.",
                  "Only nodes with a `power` record in config/nodes.json are touched, only in the ways it lists; "
                  "a display (the demo cockpit, the window) is best-effort and never fails it.",
                  "Leaves a marker motion verbs refuse on (exit 5) until `tatbot rig wake`."))
def rig_sleep(ctx, ns, rest):
    return _backend(ctx, "sleep", *(["--plan"] if ns.plan else []), *rest)


@verb(effects=("read_files", "network", "remote_exec", "start_process", "write_files"), noun="rig", verb="wake", tier=REMOTE,
      output="json", args=_plan_arg, wraps=(BACKEND,), example=("--plan",), doc=DOC,
      summary="start the camera and tracker services in stages, verify them on the bus, reopen the demo window",
      invariants=("The bus router answers first; then every enabled manifested service must be active, hold its "
                  "liveliness key on the bus and not restart before the marker clears. A disabled unit is left off.",
                  "Displays and the window are best-effort; the window opens only over its node's direct link to the "
                  "hub. A window node whose marker it cannot clear is named, and the exit is 1.",
                  "Starts no arm process; the drawing stack's state and `tatbot rig check` are reported, never gated on."))
def rig_wake(ctx, ns, rest):
    return _backend(ctx, "wake", *(["--plan"] if ns.plan else []), *rest)


def _off_args(p):
    p.add_argument("--no-self", action="store_true", help="leave the node running this command on")
    p.add_argument("--plan", action="store_true", help="print the landing and shutdown order, without ssh")


@verb(effects=("read_files", "network", "remote_exec", "stop_process", "write_files", "autonomous_motion"),
      noun="rig", verb="off", tier=REMOTE, output="json", args=_off_args, wraps=(BACKEND,), example=("--plan",),
      doc=DOC, summary="land the arms, then shut down the demo-case hosts so the power strip can be switched off",
      invariants=("Lands through `tatbot ros down` first; a landing that fails shuts nothing down.",
                  "Refuses (exit 6) while any workflow but a service's or the drawing stack's runs on a host it "
                  "shuts down, or a drawing or camera run is open on this node or an operator node.",
                  "Shuts down only nodes marked \"off\": true in config/nodes.json: arm and camera hosts, "
                  "then the e-stop relay, then the node it runs on, unless --no-self."))
def rig_off(ctx, ns, rest):
    return _backend(ctx, "off", *(["--no-self"] if ns.no_self else []), *(["--plan"] if ns.plan else []), *rest)


@verb(effects=("read_files", "network", "remote_exec"), noun="rig", verb="status", tier=SENSOR,
      output="json", wraps=(BACKEND,), example=(), doc=DOC,
      summary="the sleep marker, and each rig node's reachability, services, window and what stays up")
def rig_status(ctx, ns, rest):
    return _backend(ctx, "status", *rest)


CHECK = "scripts/lib/rig_check.py"


@verb(effects=("read_files", "network", "remote_exec"), noun="rig", verb="check", tier=SENSOR,
      output="json", args=lambda p: p.add_argument("--all", action="store_true", help="list passing checks too"),
      wraps=(CHECK,), example=(), doc=DOC,
      summary="is every piece of demo hardware up: hosts, links, USB devices, controllers, services, clocks",
      invariants=("Read-only: pings, lists devices and reads unit state; starts, stops and opens nothing.",
                  "Exit 5 while any part (pink drawing, station, blue arm) has a failed check; warnings do not fail it."))
def rig_check(ctx, ns, rest):
    return py(ctx, CHECK, *(["--json"] if ctx.json else []), *(["--all"] if ns.all else []), *rest)


LAN = "scripts/lib/rig_lan.py"


@verb(effects=("read_files", "network", "remote_exec", "write_files", "start_process"), noun="rig", verb="lan-install",
      tier=REMOTE, output="json", args=_plan_arg, wraps=(LAN, "scripts/lib/tatbot_cli/rig_lan.py"),
      example=("--plan",), doc=DOC,
      summary="install the rig's no-uplink fallbacks on every demo-case host: ssh over rig addresses, one rig clock",
      invariants=("ssh: writes ~/.ssh/tatbot-rig.conf and makes its Include the first line of ~/.ssh/config "
                  "(the old file kept once as config.bak-tatbot-rig); a node's tailnet target then falls back to its "
                  "`lan` address while this host is on the rig subnet, the tailnet address does not answer ssh and "
                  "the rig address does, keeping the host key trusted for the tailnet address.",
                  "clock: the node with role rig-clock gets chrony (apt-installed if missing) serving the rig subnet, "
                  "its own clock with no uplink; every other demo-case host takes that node first, through chrony "
                  "or systemd-timesyncd, whichever it runs.",
                  "Idempotent: an unchanged file is not rewritten and its service not restarted. Touches no arm, "
                  "camera, network profile or drawing-stack unit."))
def rig_lan_install(ctx, ns, rest):
    return py(ctx, LAN, *(["--json"] if ctx.json else []), "install", *(["--plan"] if ns.plan else []), *rest)
