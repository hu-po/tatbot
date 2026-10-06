"""teleop — a human on the leader arm."""

from __future__ import annotations

from tatbot_cli import EXIT_GATE_REFUSED, EXIT_OK
from tatbot_cli.registry import MOTION_HUMAN, OFFLINE, SENSOR, verb
from tatbot_cli.verbs._common import py, sh, tool_flag

EXCL_INV = "wxai_teleop and LeRobot sessions are mutually exclusive (exclusive driver connection)."


# --- teleop --------------------------------------------------------------------


def _teleop_poses_args(p):
    p.add_argument('log', help='recorded seven-axis follower flight log (.wxtl)')
    p.add_argument('--out', required=True, help='new output directory for poses and provenance')
    p.add_argument('--urdf', required=True, help='explicit recorded URDF model')
    p.add_argument('--workspace', required=True, help='explicit recorded workspace/tool calibration')
    p.add_argument('--arm', choices=('right', 'left'), default='right',
                   help='the follower (default) or the leader: whose recorded joints and tool mount to export')


@verb(effects=('read_files', 'write_files'), noun='teleop', verb='poses', tier=OFFLINE, visibility='public',
      summary='export measured follower tip poses and host read timestamps from a flight log',
      wraps=('scripts/vision/teleop_poses.py',), args=_teleop_poses_args, needs_tool=True,
      example=('flight.wxtl', '--out', '/tmp/follower-poses', '--urdf', 'urdf/tatbot.urdf',
               '--workspace', 'config/workspace.yaml'),
      invariants=('Explicit URDF/workspace and matching stated tool; output must not already exist.',))
def teleop_poses(ctx, ns, rest):
    from tatbot_cli.verbs._common import lerobot_py
    return lerobot_py(ctx, 'scripts/vision/teleop_poses.py', ns.log, '--out', ns.out,
                      '--urdf', ns.urdf, '--workspace', ns.workspace, '--arm', ns.arm, *tool_flag(ctx))


def _teleop_role_args(p):
    p.add_argument("--leader", choices=("left", "right"),
                   help="physical input arm; defaults to left, or the arm opposite --follower")
    p.add_argument("--follower", choices=("left", "right"),
                   help="physical receiving arm; defaults to right, or the arm opposite --leader")


def _teleop_roles_validate(ctx, command, ns):
    from tatbot_cli import arms
    from tatbot_cli.cli import UsageError, refuse

    try:
        if getattr(ns, "wrist_calibration", False):
            ns.leader = ns.leader or "right"
            ns.follower = ns.follower or "left"
            if ns.leader != "right" or ns.follower != "left":
                raise ValueError('--wrist-calibration requires right leader and left follower')
        roles = arms.select_roles(ns.leader, ns.follower)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    if command.verb == "start" and roles != arms.select_roles() and not ns.wrist_calibration:
        return refuse(ctx, EXIT_GATE_REFUSED, "teleop_roles", arms.REVERSE_BLOCKER)
    return None


def _teleop_start_prepare(ctx, command, ns):
    from tatbot_cli import arms

    arms.require_current_executor(ctx.repo, arms.select_roles(ns.leader, ns.follower),
                                  wrist_calibration=ns.wrist_calibration)


@verb(effects=('read_files',), noun="teleop", verb="plan", tier=OFFLINE,
      summary="resolve physical arms into teleop roles without connecting hardware",
      native=True, output="json", args=_teleop_role_args, validate=_teleop_roles_validate,
      example=("--leader", "right", "--follower", "left"), doc="docs/teleop_tuning.md",
      invariants=("Both directions can be planned; only left-leading-right execution is implemented.",
                  "Tool, controller, workspace and URDF references stay with their physical arm.",
                  "Read-only assignment, not calibration, ownership acquisition or motion authorization."))
def teleop_plan(ctx, ns, rest):
    import json

    from tatbot_cli import arms
    from tatbot_cli.cli import UsageError

    try:
        result = arms.teleop_plan(ctx.repo, arms.select_roles(ns.leader, ns.follower))
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    if ctx.json:
        print(json.dumps(result, indent=2))
    else:
        for role, arm in result["assignments"].items():
            print(f"{role}: {arm['id']} (controller {arm['controller_config']}, "
                  f"workspace {arm['workspace_section']}, URDF {arm['urdf_prefix']})")
        print(f"Receiving tool: {result['tool_binding']['workspace']} section "
              f"{result['tool_binding']['section']}")
        print("Existing executor supports assignment: "
              + ("yes" if result["execution"]["implemented"] else "no"))
        for reason in result["execution"]["blockers"]:
            print(f"  {reason}")
        print(result["execution"]["note"])
        print(result["camera_binding"])
    return EXIT_OK


def _teleop_start_args(p):
    _teleop_role_args(p)
    p.add_argument('--wrist-calibration', action='store_true',
                   help='free-space right/pink leader, left/laser follower; mirror joints 0, 4 and 5')
    p.add_argument("--touchoff", action="store_true",
                   help="this session IS the stated tool's touch-off: run it although workspace.yaml names another "
                        "tool (wxai_teleop --tool-uncalibrated; grip force from the datasheet)")


@verb(effects=('read_files', 'network', 'human_motion'), noun="teleop", verb="start", tier=MOTION_HUMAN, summary="the bare 400 Hz C++ teleop, live under your hands, telemetry to the viewer node — what other workflows attach to",
      role="arm", auto_hop=True, wraps=("scripts/teleop_start.sh",), passthrough="wxai_teleop", args=_teleop_start_args, needs_tool=True,
      validate=_teleop_roles_validate, prepare=_teleop_start_prepare,
      example=(), doc="docs/teleop_tuning.md", tty=True,
      invariants=(EXCL_INV,
                  "Physical roles default to left leader and right follower; general reverse execution refuses before routing. --wrist-calibration enables right-led mirrored free-space capture. "
                  "Use `teleop plan` to inspect either direction without hardware access.",
                  "`tatbot teleop start` routes itself to the configured arm owner; --on may name that owner "
                  "and may not appoint another. The hop is `ssh -t` and the launcher's gates run there.",
                  "Interactive and foreground on purpose: Enter before the follower aligns; after an e-stop or fault, "
                  "support the arms, then Enter to idle. Ctrl+C ends it. Refuses (exit 6) while another teleop runs.",
                  "A tool workspace.yaml was not measured with is refused unless --touchoff says this session measures it.",
                  "No cameras, no session recording beyond the .wxtl flight log; `tatbot live cockpit` on the "
                  "operator node shows every camera and the animated URDF beside it."))
def teleop_start(ctx, ns, rest):
    flags = ["--touchoff"] if ns.touchoff else []
    if ns.wrist_calibration:
        flags.append("--wrist-calibration")
    for role in ("leader", "follower"):
        if getattr(ns, role) is not None:
            flags.extend((f"--{role}", getattr(ns, role)))
    return sh(ctx, "scripts/teleop_start.sh", *tool_flag(ctx), *flags, *rest)


def _check_args(p):
    p.add_argument("--no-probe", action="store_true",
                   help="skip the arm ping: report configuration only, touch no network")
    p.add_argument("--rt-priority", type=int,
                   help="check against this SCHED_FIFO priority instead of the executable's default")


@verb(effects=('read_files', 'network', 'sensor_read'), noun="teleop", verb="check", tier=SENSOR,
      summary="report bounded teleop configuration and reachability checks",
      role="arm", auto_hop=True, native=True, output="json", args=_check_args,
      wraps=("scripts/lib/teleop_readiness.py",), example=("--no-probe",), doc="docs/teleop_tuning.md",
      invariants=("Not a repository check — `tatbot check` is that.",))
def teleop_check(ctx, ns, rest):
    import json

    import teleop_readiness

    from tatbot_cli import gates

    tool_id, source, error = gates.resolve_tool(ctx.repo, ctx.ee_tool)
    tool = {"id": tool_id, "source": source, "error": error,
            "source_label": gates.TOOL_SOURCES.get(source, source)}
    result = teleop_readiness.report(ctx.repo, tool=tool, probe_arms=not ns.no_probe,
                                     requested_priority=ns.rt_priority, node=ctx.node)
    result["node"] = ctx.node
    if ctx.json:
        print(json.dumps(result, indent=2))
        return EXIT_OK if result["ready"] else EXIT_GATE_REFUSED
    print(f"teleop check on {ctx.node} — " + ("ready to attempt a start" if result["ready"]
                                              else "NOT ready to start"))
    for row in result["observations"]:
        mark = {"ok": "ok  ", "failed": "FAIL", "unknown": "?   ",
                "not_applicable": "n/a "}[row["state"]]
        # Scalars only on the terminal line; a nested list (every login session
        # and its limit) is evidence for --json, not something to read sideways.
        value = row["value"]
        if isinstance(value, dict):
            value = ", ".join(f"{k}={v}" for k, v in value.items()
                              if not isinstance(v, (list, dict)))
        print(f"  {mark} {row['label']:<22} {value if value not in (None, '') else ''}")
        for line in ([row["reason"]] if row["reason"] else []) + row["detail"]:
            print(f"       {line}")
    print(f"  {result['limits']}.")
    return EXIT_OK if result["ready"] else EXIT_GATE_REFUSED


def _log_arg(p):
    p.add_argument("log", help="a .wxtl flight log (or run dir)")


@verb(effects=('read_files',), visibility="public", noun="teleop", verb="analyze", tier=OFFLINE, summary="loop-timing stats from a .wxtl without a GUI",
      wraps=("cpp/teleop/analyze_log.py",), args=_log_arg, example=("~/tatbot-logs/teleop/last/teleop.wxtl",))
def teleop_analyze(ctx, ns, rest):
    return py(ctx, "cpp/teleop/analyze_log.py", ns.log, *rest)


def _network_trace_args(p):
    p.add_argument('--seconds', type=int, default=90, choices=range(15, 121), metavar='15..120',
                   help='passive capture duration, including time to start manual teleop')


@verb(effects=('read_files', 'write_files', 'network', 'sensor_read'), noun='teleop', verb='trace-network',
      tier=SENSOR, summary='brief passive controller packet trace for diagnosing lost feedback; no arm commands',
      role='arm', auto_hop=True, args=_network_trace_args,
      wraps=('scripts/teleop_trace_network.sh', 'scripts/lib/teleop_network_trace.py'),
      example=('--seconds', '90'), doc='docs/teleop_tuning.md',
      invariants=('Only configured controller ARP and control-port traffic is retained, bounded by time and file size.',
                  'Needs tcpdump and noninteractive sudo on the arm owner; failure does not alter arm controls.'))
def teleop_trace_network(ctx, ns, rest):
    return sh(ctx, 'scripts/teleop_trace_network.sh', '--seconds', str(ns.seconds))
