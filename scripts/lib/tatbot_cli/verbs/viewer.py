"""viewer — the one fleet Rerun viewer every workflow streams into (docs/vision.md)."""

from __future__ import annotations

import os

from tatbot_cli import nodes
from tatbot_cli.registry import OFFLINE, REMOTE, Plan, verb
from tatbot_cli.verbs._common import sh

WRAPS = ("scripts/viewer.sh", "scripts/vision/rerun_session.sh")
DOC = "docs/vision.md"
INV = ("One viewer per fleet: the node with role rerun-server (config/nodes.json) runs a headless gRPC server "
       "on :9876; every producer on every node streams to it and any node with a display attaches a capped "
       "window. No workflow starts a viewer of its own.",
       "Both memory caps are mandatory (window 1 GB, server buffer 512 MB; VIEWER_MEMORY_LIMIT / "
       "SERVER_MEMORY_LIMIT); a viewer OOM froze the operator laptop on 2026-08-20.")


@verb(effects=('read_files', 'network', 'start_process'), noun="viewer", verb="start", tier=OFFLINE, summary="start the fleet viewer server here (the rerun-server node)",
      role="rerun-server", auto_hop=True, wraps=WRAPS, example=(), doc=DOC, invariants=INV)
def viewer_start(ctx, ns, rest):
    return sh(ctx, "scripts/viewer.sh", "start", *rest)


@verb(effects=('read_files', 'stop_process', 'network'), noun="viewer", verb="stop", tier=REMOTE, summary="stop the fleet viewer server (buffered data is gone); elsewhere: close this node's window",
      wraps=WRAPS, example=(), doc=DOC, invariants=INV)
def viewer_stop(ctx, ns, rest):
    return sh(ctx, "scripts/viewer.sh", "stop", *rest)


@verb(effects=('read_files', 'network'), noun="viewer", verb="status", tier=OFFLINE,
      summary="fleet viewer reachability, the local server and window, and the server unit",
      role="rerun-server", auto_hop=True, wraps=WRAPS, example=(), doc=DOC, invariants=INV)
def viewer_status(ctx, ns, rest):
    return sh(ctx, "scripts/viewer.sh", "status", *rest)


@verb(effects=('read_files', 'network', 'start_process'), noun="viewer", verb="open", tier=OFFLINE, summary="attach a capped window on this node to the fleet viewer",
      wraps=WRAPS, example=(), doc=DOC, invariants=INV)
def viewer_open(ctx, ns, rest):
    return sh(ctx, "scripts/viewer.sh", "open", *rest)


def _view_args(p):
    p.add_argument("file", help="a recorded .rrd, a .wxtl flight log, or a teleop run dir holding one")


@verb(effects=('read_files', 'network', 'remote_exec', 'start_process'), noun="viewer", verb="view", tier=OFFLINE,
      summary="stream a recorded .rrd/.wxtl (or a teleop run dir) into the fleet viewer from the node that holds it",
      wraps=WRAPS, args=_view_args, example=("~/tatbot-logs/vision/session-last.rrd",), doc=DOC,
      invariants=INV + ("A path that does not exist here is looked for on the arm node (flight logs live there): "
                        "the verb hops over ssh and visiond replay-rerun streams from that node. --no-hop keeps it here.",
                        "Refuses with exit 5 when the fleet viewer is unreachable; never opens a local window.",))
def viewer_view(ctx, ns, rest):
    local = os.path.expanduser(ns.file)
    if not os.path.exists(local) and not ctx.no_hop:
        # `tatbot teleop replay` used to be this verb plus an unconditional hop to the
        # arm node. One verb now: the file decides — present here, stream from here;
        # absent, stream from the node that keeps the flight logs.
        nmap = nodes.load(ctx.repo)
        arm = nodes.nodes_with(nmap, "arm")
        if arm and arm[0] != ctx.node:
            from tatbot_cli.cli import _home_relative
            argv = [_home_relative(a) for a in ctx.argv]
            return Plan(argv=nodes.hop_argv(nmap, arm[0], argv, tty=False),
                        notes=[f"{ns.file} is not on this node ({ctx.node}); streaming it from {arm[0]} "
                               "(role arm, where flight logs live) into the fleet viewer"])
    return sh(ctx, "scripts/viewer.sh", "view", ns.file, *rest)


@verb(effects=('read_files', 'write_config', 'network', 'start_process'), noun="viewer", verb="install", tier=REMOTE,
      summary="install the headless server unit on the rerun-server node (sudo)",
      role="rerun-server", auto_hop=True, tty=True,
      wraps=WRAPS + ("rust/visiond/src/main.rs", "rust/visiond/src/rerun_viewer.rs",
                     "config/systemd/tatbot-viewer@.service"),
      example=(), doc=DOC, invariants=INV)
def viewer_install(ctx, ns, rest):
    return sh(ctx, "scripts/viewer.sh", "install", *rest)
