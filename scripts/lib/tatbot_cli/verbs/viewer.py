"""viewer — the one fleet Rerun viewer every workflow streams into (docs/vision.md)."""

from __future__ import annotations

import os

from tatbot_cli import nodes
from tatbot_cli.registry import OFFLINE, REMOTE, Plan, verb
from tatbot_cli.verbs._common import sh

WRAPS = ("scripts/viewer.sh", "scripts/vision/rerun_session.sh", "scripts/lib/tatbot_cli/viewer_path.py")
SYSMON = ("scripts/lib/tatbot_sysmon.py", "config/systemd/tatbot-sysmon@.service")
DOC = "docs/vision.md"
INV = ("One viewer per fleet: the node with role rerun-server (config/nodes.json) runs a headless gRPC server "
       "on :9876; every producer on every node streams to it and any node with a display attaches a capped "
       "window. No workflow starts a viewer of its own.",
       "Both memory caps are mandatory (window 1 GB, server buffer 512 MB; VIEWER_MEMORY_LIMIT / "
       "SERVER_MEMORY_LIMIT); a viewer OOM froze the operator laptop on 2026-08-20.",
       "A node with a direct cable to the server (the server's `links` in config/nodes.json, keyed by this "
       "node's role) dials the server there while the kernel routes that address off the rig interface and "
       "the server answers within 1 s, else at the server's `lan` (tatbot_cli.viewer_path): the station's "
       "window stays off the rig port the arm controller shares. status names the path.")
PIN_INV = ("viewer.sh refuses (exit 3) before any sudo unless `rerun --version` on this node reports the "
           "fleet pin (rerun_session::require_pin): a unit installed without it restarts every few seconds forever. "
           "The fix it names: uv tool install 'rerun-sdk==<pin>'.",)
SYSMON_INV = ("Installs and (re)starts tatbot-sysmon@<user>: CPU/MEM %, the hottest CPU/SoC zone in deg C "
              "(GPU % and temperature on the station) at 1 Hz under "
              "system/<role>/ in the standing recording, legended by the host's `platform` in config/nodes.json. "
              "The station's also reads each host whose `load_via` is its role over ssh (shell builtins, nothing "
              "installed there); the ros node's adds system/robot/ from the drawing stack's run files. "
              "Display only; a lost hub ends it for systemd to restart 15 s later.",)


@verb(effects=('read_files', 'network', 'start_process'), noun="viewer", verb="start", tier=OFFLINE, summary="start the fleet viewer server here (the rerun-server node)",
      role="rerun-server", auto_hop=True, wraps=WRAPS, example=(), doc=DOC, invariants=INV)
def viewer_start(ctx, ns, rest):
    return sh(ctx, "scripts/viewer.sh", "start", *rest)


@verb(effects=('read_files', 'stop_process', 'network'), noun="viewer", verb="stop", tier=REMOTE, summary="stop the fleet viewer server (buffered data is gone); elsewhere: close this node's window",
      wraps=WRAPS, example=(), doc=DOC, invariants=INV)
def viewer_stop(ctx, ns, rest):
    return sh(ctx, "scripts/viewer.sh", "stop", *rest)


@verb(effects=('read_files', 'network'), noun="viewer", verb="status", tier=OFFLINE,
      summary="fleet viewer reachability, the local server and window, the server unit and the web viewer URL",
      role="rerun-server", auto_hop=True, wraps=WRAPS, example=(), doc=DOC, invariants=INV)
def viewer_status(ctx, ns, rest):
    return sh(ctx, "scripts/viewer.sh", "status", *rest)


def _open_args(p):
    p.add_argument("--memory-limit", metavar="SIZE",
                   help="the window's row cap, an absolute size (default 1GB; the station's portrait demo page "
                        "uses 4GB); the unit's hard MemoryMax is this plus 1 GiB")
    p.add_argument("--restart", action="store_true",
                   help="close a running window first: a window never reconnects to a hub that restarted")
    p.add_argument("--fullscreen", action="store_true",
                   help="fill the display (no title or top bar) once the window maps, and give the focus back "
                        "to the window that held it: the station's demo page behind its agent terminal")


@verb(effects=('read_files', 'network', 'start_process'), noun="viewer", verb="open", tier=OFFLINE, summary="attach a capped window on this node to the fleet viewer",
      wraps=WRAPS + ("scripts/lib/x11_window.py",), args=_open_args, example=("--memory-limit", "4GB", "--fullscreen"), doc=DOC,
      invariants=INV + ("The window runs as the transient user unit tatbot-viewer-window (systemd-run --user, "
                        "MemoryMax = the cap + 1 GiB); a running window is left as it is, so a second open "
                        "is a no-op, unless --restart closes it first.",
                        "Once it maps the window is maximized, or with --fullscreen made fullscreen, by an EWMH "
                        "request to the X11 window manager (scripts/lib/x11_window.py): no key is typed and a "
                        "window already in that state is not asked again; nothing happens off X11.",))
def viewer_open(ctx, ns, rest):
    cap = ["--memory-limit", ns.memory_limit] if ns.memory_limit else []
    flags = [f for f, on in (("--restart", ns.restart), ("--fullscreen", ns.fullscreen)) if on]
    return sh(ctx, "scripts/viewer.sh", "open", *cap, *flags, *rest)


@verb(effects=('read_files', 'network'), noun="viewer", verb="reset", tier=REMOTE,
      summary="send the fixed blueprint again: every attached window drops its own edits, a hidden pane included",
      role="rerun-server", auto_hop=True, wraps=WRAPS + ("rust/visiond/src/rerun_viewer.rs",), example=(), doc=DOC,
      invariants=INV + ("A window keeps its own copy of the blueprint and a tap edits it: the eye beside a pane's "
                        "maximize button hides the pane, leaving an empty title strip, and the station's page "
                        "shows no blueprint panel to bring it back. Every window adopts each blueprint the hub "
                        "sends, so this restores the fixed layout without restarting the hub or a window.",))
def viewer_reset(ctx, ns, rest):
    return sh(ctx, "scripts/viewer.sh", "reset", *rest)


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
        # The file decides: present here, stream from here; absent, stream from
        # the node that keeps the flight logs.
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
      summary="install the headless server unit and the node's load monitor on the rerun-server node (sudo)",
      role="rerun-server", auto_hop=True, tty=True,
      wraps=WRAPS + SYSMON + ("rust/visiond/src/main.rs", "rust/visiond/src/rerun_viewer.rs",
                              "config/systemd/tatbot-viewer@.service"),
      example=(), doc=DOC, invariants=INV + PIN_INV + SYSMON_INV)
def viewer_install(ctx, ns, rest):
    return sh(ctx, "scripts/viewer.sh", "install", *rest)


@verb(effects=('read_files', 'write_config', 'network', 'start_process'), noun="viewer", verb="station-install",
      tier=REMOTE, summary="install the station's load monitor (tatbot-sysmon@) on the station (sudo)",
      role="station", auto_hop=True, tty=True, wraps=WRAPS + SYSMON,
      example=(), doc=DOC, invariants=INV + PIN_INV + SYSMON_INV)
def viewer_station_install(ctx, ns, rest):
    return sh(ctx, "scripts/viewer.sh", "station-install", *rest)
