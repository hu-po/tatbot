"""ros — the ROS 2 drawing stack (ros/README.md): deploy, run and drive it on the node owning `ros`.

Every verb execs one backend, scripts/lib/tatbot_cli/tatbot_ros.py, from this checkout. The
backend resolves the `ros` role owner in config/nodes.json and acts there over ssh inside
~/tatbot-ros (never that node's own git checkout), or locally on the owner. So these verbs declare
no role and never hop. A sim lane (`up --hardware sim`, or TATBOT_ROS_LANE for every verb) resolves
the `ros-sim` role's owner instead and runs the stack in a container there, never on the ros node.
"""

from __future__ import annotations

from tatbot_cli.registry import MOTION_AUTO, OFFLINE, REMOTE, SENSOR, verb
from tatbot_cli.verbs._common import py

BACKEND = "scripts/lib/tatbot_cli/tatbot_ros.py"
DOC = "ros/README.md"
WRAPS = (BACKEND, 'scripts/lib/drawing_python.sh', 'scripts/lib/drawing_python.py',
         'ros/tatbot_bringup/scripts/env.sh.in')


def _backend(ctx, name, *args):
    plan = py(ctx, BACKEND, name, *args)
    if ctx.dry_run:
        plan.notes.extend(_backend_plan(ctx, [name, *args]))
    return plan


def _backend_plan(ctx, argv: list[str]) -> list[str]:
    """The backend's own `--dry-run`, in process: every ssh, rsync and colcon command it would run on the
    ros node (TATBOT_ROS_ROOT / _DOMAIN / _PORT select a parallel workspace there)."""
    import contextlib
    import importlib.util
    import io

    spec = importlib.util.spec_from_file_location("tatbot_ros_backend", ctx.path(BACKEND))
    module = importlib.util.module_from_spec(spec)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            spec.loader.exec_module(module)
            code = module.main(["--dry-run", *argv])
        except SystemExit as exc:  # argparse
            code = exc.code
    lines = [f"plan   {line}" for line in out.getvalue().splitlines() if line.strip()]
    return lines + ([f"the backend's dry run exits {code}"] if code else [])


def _place(p) -> None:
    p.add_argument("--at", metavar="X,Y", help="centre the design here, mm in the page frame (x right, y toward the top "
                                               "of the print; write --at=-15,42 for a negative x)")
    p.add_argument("--width", metavar="MM", help="assert acquired canvas width in mm; regenerate to resize")
    p.add_argument("--stencil", metavar="DIR", help="the generated print it is drawn on (its directory): place the "
                                                    "design inside that print's clear centre, not the nominal page's")


def _palette_args(p):
    p.add_argument('action', choices=('status', 'load'))
    p.add_argument('caps', nargs='*', metavar='SLOT=INK', help='catalog ink ID, none (empty), absent, or unknown')
    p.add_argument('--level-mm', action='append', default=[], metavar='SLOT=HEIGHT',
                   help='measured ink-level lower bound above the inner cap floor, mm; required for each declared ink')


@verb(effects=('read_files', 'network', 'remote_exec', 'remote_write'), noun='ros', verb='palette', tier=REMOTE,
      args=_palette_args, output='json', wraps=(*WRAPS, 'scripts/lib/tatbot_cli/ros_palette.py'), doc=DOC,
      example=('load', 'M1=nighthawk_black', '--level-mm', 'M1=6'),
      summary="read or declare what each palette cap holds and its measured ink level, on the ROS owner",
      invariants=("Writes the owner's config/palette_load.yaml, which deploy leaves alone; a dip reads its level.",
                  "Moves nothing and estimates no volume."))
def ros_palette(ctx, ns, rest):
    flags = [part for value in ns.level_mm for part in ('--level-mm', value)]
    return _backend(ctx, 'palette', ns.action, *ns.caps, *flags, *rest)


def _place_flags(ns) -> list[str]:
    return [f"--{name}={value}" for name in ("at", "width", "stencil") if (value := getattr(ns, name, None))]


def _flags(ns, pairs) -> list[str]:
    out: list[str] = []
    for dest, flag in pairs:
        value = getattr(ns, dest, None)
        if value is True:
            out.append(flag)
        elif value not in (None, False):
            out += [flag, str(value)]
    return out


def _deploy_args(p):
    p.add_argument("--no-build", action="store_true", help="sync files only; skip the incremental colcon build")
    p.add_argument("--units", action="store_true", help="also (re)install the systemd units (sudo on the ros node)")
    p.add_argument("--clean", action="store_true", help="drop build/ and install/ first (a full rebuild)")


@verb(effects=("read_files", "network", "remote_exec", "remote_write", "environment_setup", "stop_process",
               "autonomous_motion"), noun="ros", verb="deploy",
      tier=REMOTE, args=_deploy_args, wraps=WRAPS, doc=DOC, example=("--no-build",),
      summary="sync this checkout's ros/, config/, urdf/ and scripts/ to the ros node's ~/tatbot-ros and colcon-build there",
      invariants=("Never touches the ros node's own git checkout.",
                  "A build of tatbot_interfaces, or a clean one, first lands and stops a running stack (as down), refused (exit 6) while a draw, touch or calibration runs there.",
                  "Copies the adopted arm registrations; writes no tracked config.",
                  "Never starts or restarts the stack (tatbot ros up does); it lands and stops one only before a "
                  "build it could no longer reach afterwards."))
def ros_deploy(ctx, ns, rest):
    return _backend(ctx, "deploy", *_flags(ns, (("no_build", "--no-build"), ("units", "--units"), ("clean", "--clean"))), *rest)


def _up_args(p):
    p.add_argument("--hardware", choices=("mock", "fake", "real", "sim"),
                   help="default: stack.yaml hardware (mock); sim: the sim lane TATBOT_ROS_LANE (required) on the "
                        "ros-sim owner, stepped by the sim server")
    p.add_argument("--config", metavar="STACK_YAML",
                   help="hardware sim: the run's stack.yaml (with the sim overlay), an absolute path on the ros-sim owner")
    p.add_argument("--estop", choices=("none", "serial", "udp"), help="e-stop source; default: stack.yaml estop.source")
    p.add_argument("--relay-addr", metavar="ADDR", help="udp: the relay's source address (default: role estop-relay `lan`)")
    p.add_argument("--page", choices=("stencil", "fixed"), help="page source; default: stack.yaml page.source")
    p.add_argument("--pattern", metavar="ID", help="the print to draw on, installed on the ros node: its pattern id or "
                                                   "the hex digits its sheet prints; default: stack.yaml page.pattern_id")
    p.add_argument("--surface", choices=("plane", "fitted"),
                   help="fitted: a curved page (a forearm) from the gauge holds' skin; default: stack.yaml "
                        "page.surface.mode (plane)")
    p.add_argument("--arms", metavar="LIST", help="comma-separated arms; default: stack.yaml arms (right)")
    p.add_argument("--no-touch", action="store_true", help="skip the page touches at run start")
    p.add_argument("--rerun", action="store_true", help="also start the Rerun bridge to the fleet viewer")
    p.add_argument("--hold", action="store_true", help="restart a running stack without landing its arms first")
    p.add_argument("--probe", action="store_true",
                   help="read the station probe's PRB1 frames (from role estop-relay `lan`), so GUARD_PROBE can arm")
    p.add_argument("--tuning", metavar="PROFILE",
                   help="connect the arms on this config/trossen/tatbot.yaml ros_tuning profile (calibration: the "
                        "velocity integral x1); default: stack.yaml tuning, the role's own")


@verb(effects=("read_files", "network", "remote_exec", "remote_write", "start_process", "stop_process"), noun="ros",
      verb="up", tier=REMOTE, args=_up_args, wraps=WRAPS, doc=DOC, example=("--hardware", "mock"),
      summary="(re)start the stack on the ros node: router, controller_manager at 400 Hz, controllers, bridge, session",
      invariants=("--hardware real connects to the configured arms only (stack.yaml arms, right by default).",
                  "--hardware sim runs on the ros-sim owner in a container, never on the ros node, and refuses "
                  "(exit 6) while CI runs there or its lane is up.",
                  "A running stack lands every arm that has not landed before the restart, unless --hold.",
                  "A running stack whose arms do not answer (no SafetyState) is refused, exit 3, while a driver may hold an arm, unless --hold.",
                  "The driver takes the arm-driver lock, so it never runs beside teleop or LeRobot.",
                  "The e-stop source is configuration: none, serial or udp."))
def ros_up(ctx, ns, rest):
    return _backend(ctx, "up", *_flags(ns, (("hardware", "--hardware"), ("config", "--config"), ("estop", "--estop"),
                                            ("relay_addr", "--relay-addr"),
                                            ("page", "--page"), ("pattern", "--pattern"), ("surface", "--surface"),
                                            ("arms", "--arms"),
                                            ("no_touch", "--no-touch"),
                                            ("rerun", "--rerun"), ("hold", "--hold"), ("probe", "--probe"),
                                            ("tuning", "--tuning"))), *rest)


def _down_args(p):
    p.add_argument("--router", action="store_true", help="also stop the ROS router")
    p.add_argument("--hold", action="store_true", help="stop without landing the arms first")


@verb(effects=("network", "remote_exec", "stop_process", "autonomous_motion"), noun="ros", verb="down", tier=REMOTE,
      args=_down_args, wraps=WRAPS, doc=DOC, example=(),
      summary="land every arm that has not landed, then stop the stack on the ros node",
      invariants=("An arm that has not landed can drop when its driver exits, so down lands it first unless --hold.",
                  "A landing that fails leaves the stack up and holding.",
                  "A running stack whose arms do not answer is refused, exit 3, while a driver may hold an arm, unless --hold."))
def ros_down(ctx, ns, rest):
    return _backend(ctx, "down", *_flags(ns, (("router", "--router"), ("hold", "--hold"))), *rest)


@verb(effects=("read_files", "network", "remote_exec", "sensor_read"), noun="ros", verb="status", tier=SENSOR,
      output="json", wraps=WRAPS, doc=DOC, example=(),
      summary="units, controllers, each arm's safety state, the page pose age and the newest run")
def ros_status(ctx, ns, rest):
    return _backend(ctx, "status", *(["--json"] if ctx.json else []), *rest)


def _page_args(p):
    p.add_argument("--arm", default="right", help="the arm whose base the pose is given in")
    p.add_argument("--seconds", type=float, default=5.0, help="how long to watch the tracker (default 5)")


@verb(effects=("network", "remote_exec", "sensor_read"), noun="ros", verb="page", tier=SENSOR, args=_page_args,
      output="json", wraps=WRAPS, doc=DOC, example=("--seconds", "10"),
      summary="the stencil page as the overhead tracker sees it now: measured or lost, its pose in the arm's base, "
              "and how far it stands from the page the newest set-up placed; moves nothing")
def ros_page(ctx, ns, rest):
    return _backend(ctx, "page", "--arm", ns.arm, "--seconds", str(ns.seconds), *(["--json"] if ctx.json else []),
                    *rest)


def _station_args(p):
    p.add_argument("--arm", default="right", help="the arm whose frame the station is placed in (its registration)")
    p.add_argument("--against", default="",
                   help="an earlier ros-station run id or station.json on the ros node: exit 1 when the station moved")


@verb(effects=("network", "remote_exec", "sensor_read", "write_files"), noun="ros", verb="station", tier=SENSOR,
      args=_station_args, wraps=WRAPS, doc=DOC, example=("--arm", "right"),
      summary="the station in an arm's frame, measured now from the palette's roof tag, never a stored pose")
def ros_station(ctx, ns, rest):
    return _backend(ctx, "station", "--arm", ns.arm, *(["--against", ns.against] if ns.against else []), *rest)


def _register_args(p):
    p.add_argument("--arm", default="right", help="the arm to register (the stack must be up with it)")
    p.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"),
                   help="the hold grid's centre in <arm>/base_link, m (default the configured page's)")
    p.add_argument("--table-z", dest="table_z", type=float, help="the table's height in <arm>/base_link, m")
    p.add_argument("--spread", type=float, help="the hold grid's half-width, m (default 0.07)")
    p.add_argument("--heights", type=float, nargs="+", help="tool heights over the table, m (default 0.12 0.17)")
    p.add_argument("--max-holds", dest="max_holds", type=int, default=0, help="stop after this many holds")
    p.add_argument("--prior", default="",
                   help="rest (the wrist tags the D555 sees at the arm's current pose; moves nothing) or an earlier "
                        "ros-register run id, whose camera plans upright holds for what they measure of the camera, "
                        "the tags' seat and joints 1-4")
    p.add_argument("--holds", type=int, default=0, help="with --prior: how many holds to plan (default 27)")
    p.add_argument("--adopt", action="store_true",
                   help="install a result that passes its gates on the ros node and restart the D555 owner")
    p.add_argument("--plan-only", dest="plan_only", action="store_true",
                   help="reach-check the holds and move nothing")


@verb(effects=("network", "remote_exec", "autonomous_motion", "sensor_read", "write_files"), noun="ros", verb="register",
      tier=MOTION_AUTO, args=_register_args, wraps=WRAPS, doc=DOC, example=("--arm", "right", "--adopt"),
      summary="register an arm to the overhead D555: wrist-tag holds the stack drives, one PnP, gated adoption",
      invariants=("The D555's colour optical frame is the world; the bundle holds its native optics at identity.",
                  "Each hold is a Touch MODE_MOVE with no guard; the arm lands at the end.",
                  "--adopt refuses a fit that fails its gates: holds, per-tag holds, corner residuals, hold-out."))
def ros_register(ctx, ns, rest):
    flags = [part for flag, value in (("--table-z", ns.table_z), ("--spread", ns.spread),
                                      ("--max-holds", ns.max_holds or None), ("--prior", ns.prior or None),
                                      ("--holds", ns.holds or None)) if value is not None
             for part in (flag, str(value))]
    if ns.center:
        flags += ["--center", *map(str, ns.center)]
    if ns.heights:
        flags += ["--heights", *map(str, ns.heights)]
    return _backend(ctx, "register", "--arm", ns.arm, *flags, *(["--adopt"] if ns.adopt else []),
                    *(["--plan-only"] if ns.plan_only else []), *rest)


def _setup_args(p):
    p.add_argument("--arm", default="right", help="the arm to set up")
    p.add_argument("--hardware", choices=("mock", "fake", "real"), help="default: stack.yaml hardware (mock)")
    p.add_argument("--estop", choices=("none", "serial", "udp"), help="e-stop source; default: stack.yaml estop.source")
    p.add_argument("--relay-addr", metavar="ADDR", help="udp: the relay's source address (default: role estop-relay `lan`)")
    p.add_argument("--pattern", metavar="ID", help="the print on the skin, installed on the ros node; default: "
                                                   "stack.yaml page.pattern_id")
    p.add_argument("--surface", choices=("plane", "fitted"), help="fitted: a curved page; default: stack.yaml")
    p.add_argument("--arms", metavar="LIST", help="comma-separated arms; default: stack.yaml arms (right)")
    p.add_argument("--rerun", action="store_true", help="also start the Rerun bridge to the fleet viewer")
    p.add_argument("--detach", action="store_true",
                   help="return at once instead of following its log (it always runs in its own session, so a "
                        "closed terminal or an agent's tool timeout cannot stop it partway); --progress follows it")
    p.add_argument("--from", dest="start", choices=("register", "overhead", "station", "height", "gauge", "page"),
                   help="resume at this step (a failed setup prints which); earlier steps keep what they measured")
    p.add_argument("--progress", action="store_true",
                   help="read only: the newest setup from this computer, running or finished, in a few lines")


@verb(effects=("network", "remote_exec", "remote_write", "autonomous_motion", "sensor_read", "write_files",
               "start_process", "stop_process"), noun="ros", verb="setup", tier=MOTION_AUTO, args=_setup_args,
      wraps=WRAPS, doc=DOC, example=("--hardware", "real", "--estop", "udp"),
      summary="measure an arm, the D555, the palette and the stencil afresh after setting the rig up: register, "
              "the D555 again through the wrist, station touch, page set-up, then land",
      invariants=("Nothing from an earlier layout is assumed: the registration's holds are planned from the tags the "
                  "D555 sees at rest.",
                  "The registration kept is the wrist camera's (calib overhead), not the tags', and is kept on this "
                  "node too, so the next deploy carries it.",
                  "A stack running an older commit of this checkout is deployed from it first (tatbot ros deploy, "
                  "its build included); one running a commit this checkout lacks is left as it is.",
                  "Each step stops the setup when it fails, and the arm lands either way."))
def ros_setup(ctx, ns, rest):
    flags = [part for flag, value in (("--ee-tool", ctx.ee_tool), ("--hardware", ns.hardware), ("--estop", ns.estop),
                                      ("--relay-addr", ns.relay_addr), ("--pattern", ns.pattern),
                                      ("--surface", ns.surface), ("--arms", ns.arms)) if value
             for part in (flag, value)]
    if ns.progress:
        return _backend(ctx, "setup", "--progress", *(["--json"] if ctx.json else []), *rest)
    return _backend(ctx, "setup", "--arm", ns.arm, *flags, *(["--rerun"] if ns.rerun else []),
                    *(["--from", ns.start] if ns.start else []), *(["--detach"] if ns.detach else []), *rest)


def _chain_args(p):
    p.add_argument("runs", nargs="+", help="ros-register run ids on the ros node, all of one arm, pooled")
    p.add_argument("--arm", default="right", help="the arm they registered")


@verb(effects=("read_files", "write_files", "network", "remote_exec"), noun="ros", verb="chain", tier=OFFLINE,
      args=_chain_args, wraps=WRAPS, doc=DOC, example=("<run-id>", "<run-id>"),
      summary="what an arm's registrations leave: the wrist tags' seat and the joints' offsets, scored held out",
      invariants=("Nothing moves and nothing is adopted: chain.json and a wrist layout candidate go to its run log.",
                  "A model is scored on runs it was not fitted to; the best is the least held-out corner error."))
def ros_chain(ctx, ns, rest):
    return _backend(ctx, "chain", "--arm", ns.arm, *ns.runs, *rest)


def _calib_args(p):
    p.add_argument("action", choices=("run", "sweep", "height", "wristcam", "gauge", "overhead", "apply"),
                   help="run: measure the arm's tool tip across its axis on the station probe; sweep: measure it "
                        "contact-free from joint-6 turns and tilts about the tip the palette camera watches, the "
                        "tilts its length; height: the station touch's "
                        "ball height from the palette camera, the tip held over it; wristcam: the wrist camera on the tool "
                        "mount from the palette's roof tag (a check: it adopts nothing); gauge: the wrist gauge's tip "
                        "from known heights over the probe body's top, adopted when they agree; overhead: the overhead "
                        "D555 registered through the wrist D405 (the roof tag, the holder's tile, both depths), not the "
                        "wrist cube; apply: adopt a run's tip")
    p.add_argument("--run", default="", help="apply: the ros-calib run id whose candidate to adopt")
    p.add_argument("--arm", default="right", help="the arm to calibrate (the stack must be up with it and --probe)")
    p.add_argument("--attitudes", default="", help="upright yaws as yaw/0 degrees about the base heading, comma "
                   "separated; none: S1 alone")
    p.add_argument("--heading-deg", dest="heading_deg", default="", help="the base attitude's tool heading")
    p.add_argument("--station-only", dest="station_only", action="store_true",
                   help="run: S1 alone, the station touch palette dips aim by; no tip is fitted")
    p.add_argument("--pattern", default="", help="overhead: the fiducial tile on the holder (default: the stack's)")
    p.add_argument("--adopt", action="store_true", help="overhead: install the registration and restart its readers")


@verb(effects=("network", "remote_exec", "autonomous_motion", "sensor_read", "write_files"), noun="ros", verb="calib",
      tier=MOTION_AUTO, args=_calib_args, wraps=WRAPS, doc=DOC, example=("run", "--arm", "right"),
      summary="measure a tool tip across its axis on the station probe (station fix, side pairs at upright yaws, fit), "
              "or contact-free, its length too, from joint-6 turns and tilts under the palette camera (sweep); apply "
              "adopts a run's tip",
      invariants=("The tool is the arm's fitted tool in config/workspace.yaml; a stated --ee-tool must be that one.",
                  "A tool whose datasheet declares no probe contact model is refused (exit 3) before anything moves.",
                  "sweep: every hold is a probe-guarded Touch MODE_MOVE; the turns and tilts go only once the park's "
                  "still shows the tip clear of the ball's top."))
def ros_calib(ctx, ns, rest):
    flags = [part for flag, value in (("--ee-tool", ctx.ee_tool), ("--attitudes", ns.attitudes),
                                      ("--heading-deg", ns.heading_deg), ("--run", ns.run), ("--pattern", ns.pattern))
             if value for part in (flag, value)] + (["--station-only"] if ns.station_only else []) + \
        (["--adopt"] if ns.adopt else [])
    return _backend(ctx, "calib", ns.action, "--arm", ns.arm, *flags, *rest)


def _ready_args(p):
    p.add_argument("--arm", default="right", help="the arm to ready before freezing a runtime")


@verb(effects=("network", "remote_exec", "autonomous_motion", "start_process"), noun="ros", verb="ready",
      tier=MOTION_AUTO, args=_ready_args, wraps=WRAPS, doc=DOC, example=(),
      summary="wake a landed arm using the ordinary draw startup, before freezing a paired runtime")
def ros_ready(ctx, ns, rest):
    return _backend(ctx, "ready", "--arm", ns.arm, *rest)


def _compile_args(p):
    p.add_argument("design", nargs="+", help="acquired DBV3 artwork.json or placed inkmap-design.json; several "
                                             "acquired artworks (one per pen) are layers, drawn in the order given")
    p.add_argument("--arm", default="right", help="whose fitted tool compiles it (default right)")
    p.add_argument("--speed", metavar="MM_S", help="draw speed in mm/s (default 3.5)")
    p.add_argument("--inks", metavar="FILE", help="tatbot-inks/3 file binding acquired pens to physical resources")
    p.add_argument("--max-segment-s", metavar="SECONDS", help="planned Cartesian duration per computational chunk (default 60)")
    p.add_argument("--max-stroke-mm", metavar="MM", help="draw a longer path as strokes of at most this length, each "
                                                        "lifted and read again (default: whole)")
    p.add_argument("--min-stroke-mm", metavar="MM", help="drop a path no wider than this that lies on its own ink's "
                                                        "line (default 1; 0 keeps every path)")
    p.add_argument("--mark-width-mm", metavar="MM", help="count the pen's passes at this mark width (default: the tool's "
                                                        "line width, else each path's generation width)")
    p.add_argument("--keep-region", metavar="X,Y;X,Y;...",
                   help="keep only the paths inside this polygon, mm in the page frame, as placed: a part of a design "
                        "drawn again over its own (a crossing path goes with most of its length, or is cut at the edge "
                        "when both sides hold over 20 mm of it); write --keep-region=... for a negative first x")
    p.add_argument("-o", "--out", metavar="DIR", help="where program.json and preview.svg go (default .)")
    _place(p)


@verb(effects=("read_files", "write_files"), noun="ros", verb="compile", tier=OFFLINE, args=_compile_args,
      wraps=WRAPS, doc=DOC, example=("inkmap-design.json",),
      summary="prepare acquired DBV3 artwork into program.json + preview.svg (tatbot_ink; no arm, no ROS)")
def ros_compile(ctx, ns, rest):
    return _backend(ctx, "compile", *ns.design, "--arm", ns.arm,
                    *(["--ee-tool", ctx.ee_tool] if ctx.ee_tool else []),
                    *_flags(ns, (("speed", "--speed"), ("inks", "--inks"), ("max_segment_s", "--max-segment-s"),
                                 ("min_stroke_mm", "--min-stroke-mm"), ("mark_width_mm", "--mark-width-mm"),
                                 ("max_stroke_mm", "--max-stroke-mm"), ("out", "--out"))), *_place_flags(ns),
                    *([f"--keep-region={ns.keep_region}"] if ns.keep_region else []), *rest)


def _draw_args(p):
    p.add_argument("program", nargs="?", help="program.json (or acquired artwork / placed design, prepared first)")
    p.add_argument("--arm", default="right", help="the arm that draws (default right)")
    p.add_argument("--from-op", metavar="ID", help="start at this op id")
    p.add_argument("--resume", metavar="RUN_ID", help="resume that run's ledger (after a cartridge swap, a landing)")
    p.add_argument("--no-inspect", action="store_true", help="skip the wrist-camera look after a complete draw")
    p.add_argument("--rehearse", action="store_true",
                   help="draw the same program in the air: page setup, gauge holds, hover stops and the stencil's "
                        "follow as with ink, the pen held over the page, the machine off, no dips, no inspection")
    p.add_argument("--standoff-mm", metavar="MM",
                   help="with --rehearse: the pen this far over the measured page (default motion.yaml pen.hover, 6)")
    p.add_argument("--hold", action="store_true",
                   help="leave the arm holding at rest after the draw (default: land it, controller idle)")
    p.add_argument("--no-wake", action="store_true", help="do not automatically restart a landed arm")
    p.add_argument("--speed", metavar="MM_S", help="draw speed in mm/s when compiling a design (default 3.5)")
    p.add_argument("--detach", action="store_true",
                   help="return at once instead of following its log (it always runs in its own session, so a "
                        "closed terminal or an agent's tool timeout cannot stop it partway); --progress follows it")
    p.add_argument("--progress", action="store_true",
                   help="read only: the newest draw from this computer, running or finished, in a few lines")
    _place(p)


@verb(effects=("read_files", "write_files", "network", "remote_exec", "autonomous_motion"), noun="ros", verb="draw",
      tier=MOTION_AUTO, args=_draw_args, wraps=WRAPS, doc=DOC, example=("program.json", "--arm", "right"),
      summary="draw a program with the running stack: page touches, then every op; waits on pause and e-stop",
      invariants=("The next op is the first op of the ledger that is not done; a crash-sent op waits for decide.",
                  "An e-stop press holds the arm; after release the arm waits for decide continue or land.",
                  "A tool change to another cartridge lands the arm; --resume says it is fitted and keeps the page."))
def ros_draw(ctx, ns, rest):
    if ns.progress:
        return _backend(ctx, "draw", "--progress", *(["--json"] if ctx.json else []), *rest)
    return _backend(ctx, "draw", *([ns.program] if ns.program else []), "--arm", ns.arm,
                    *(["--detach"] if ns.detach else []),
                    *(["--ee-tool", ctx.ee_tool] if ctx.ee_tool else []),
                    *_flags(ns, (("from_op", "--from-op"), ("resume", "--resume"), ("no_inspect", "--no-inspect"),
                                 ("hold", "--hold"), ("no_wake", "--no-wake"), ("speed", "--speed"),
                                 ("rehearse", "--rehearse"), ("standoff_mm", "--standoff-mm"))), *_place_flags(ns), *rest)


def _touch_args(p):
    p.add_argument("--arm", default="right", help="the arm that touches (default right)")
    p.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="one touch from this tcp position in <arm>/base_link (m) instead of the page touches")
    p.add_argument("--rpy", type=float, nargs=3, metavar=("R", "P", "Y"),
                   help="the single touch's tcp orientation (URDF roll-pitch-yaw in <arm>/base_link, rad; "
                        "default the measured one)")
    p.add_argument("--direction", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
                   help="the single touch's direction in <arm>/base_link (default straight down)")
    p.add_argument("--prior", type=float, help="the expected contact this far along the direction (m)")
    p.add_argument("--probe", action="store_true",
                   help="guard on the station probe (a touch needs --prior; the travel past it is capped)")
    p.add_argument("--move", action="store_true",
                   help="no touch: travel to --start under the guard and hold there (a view of the station)")
    p.add_argument("--speed", type=float, help="the slow leg (m/s; default motion.yaml)")
    p.add_argument("--max-travel", dest="max_travel", type=float, help="m (default motion.yaml)")


@verb(effects=("network", "remote_exec", "autonomous_motion"), noun="ros", verb="touch", tier=MOTION_AUTO,
      args=_touch_args, wraps=WRAPS, doc=DOC, example=("--arm", "right"),
      summary="guarded touches: three on the page (its plane beside the camera's), or one from --start")
def ros_touch(ctx, ns, rest):
    single = [part for flag, value in (("--start", ns.start), ("--rpy", ns.rpy), ("--direction", ns.direction))
              if value for part in (flag, *map(str, value))]
    single += [part for flag, value in (("--prior", ns.prior), ("--speed", ns.speed), ("--max-travel", ns.max_travel))
               if value for part in (flag, str(value))]
    flags = [flag for flag, on in (("--probe", ns.probe), ("--move", ns.move)) if on]
    return _backend(ctx, "touch", "--arm", ns.arm, *single, *flags, *rest)


def _jog_args(p):
    p.add_argument("--arm", default="right", help="the arm to jog (default right)")
    p.add_argument("--joint", type=int, required=True, help="joint index 0-5 (rad), or 6 for the carriage (m)")
    p.add_argument("--delta", type=float, required=True, help="how far: rad, or m for the carriage")
    p.add_argument("--speed", type=float, help="peak speed in rad/s (default 0.02; the carriage 0.002 m/s)")


@verb(effects=("network", "remote_exec", "autonomous_motion"), noun="ros", verb="jog", tier=MOTION_AUTO,
      args=_jog_args, wraps=WRAPS, doc=DOC, example=("--joint", "0", "--delta", "0.02"),
      summary="move one joint by a small delta from its measured position, through the trajectory controller",
      invariants=("A quintic joint move from the measured joints: no step, zero velocity at both ends.",
                  "An e-stop press holds the arm mid-jog like any other goal."))
def ros_jog(ctx, ns, rest):
    return _backend(ctx, "jog", "--arm", ns.arm, "--joint", str(ns.joint), "--delta", str(ns.delta),
                    *_flags(ns, (("speed", "--speed"),)), *rest)


def _inspect_args(p):
    p.add_argument("--depth-only", action="store_true", help="retain native RGB-D at the current held pose without moving")
    p.add_argument("--arm", default="right", help="the arm whose wrist camera looks (default right)")
    p.add_argument("--run", help="the ros-draw run to look at (default the newest)")
    p.add_argument("--frames", type=int, help="frames per pose (default 5)")
    p.add_argument("--poses", type=int, help="camera poses (default 3)")


@verb(effects=("network", "remote_exec", "autonomous_motion"), noun="ros", verb="inspect", tier=MOTION_AUTO,
      args=_inspect_args, wraps=WRAPS, doc=DOC, example=(),
      summary="hover the wrist camera over the newest drawing and write page-frame views with the plan overlaid",
      invariants=("Lifts the pen straight up first, then joint moves with the pen at least 15 mm over the page.",
                  "Writes <run>/inspect/<time>/: raw frames, page.png, overlay.png, meta.json."))
def ros_inspect(ctx, ns, rest):
    return _backend(ctx, "inspect", "--arm", ns.arm, *_flags(ns, (("run", "--run"), ("frames", "--frames"),
                                                                  ("poses", "--poses"), ("depth_only", "--depth-only"))), *rest)


def _evidence_args(p):
    p.add_argument("--page", required=True, help="physical research page ID")
    p.add_argument("--slot", required=True, help="reserved row:column")


@verb(effects=("read_files", "network", "remote_exec"), noun="ros", verb="evidence", tier=SENSOR,
      args=_evidence_args, output="json", wraps=(*WRAPS, "ros/tatbot_session/tatbot_session/research.py"), doc=DOC,
      example=("--page", "study-page-1", "--slot", "0:0"),
      summary="read a research slot claim, original run ledger, runtime and inspection evidence without motion")
def ros_evidence(ctx, ns, rest):
    return _backend(ctx, "evidence", "--page", ns.page, "--slot", ns.slot, *rest)


def _decide_args(p):
    p.add_argument("arm", help="the waiting arm (right)")
    p.add_argument("decision", choices=("continue", "land", "skip", "redraw"))


@verb(effects=("network", "remote_exec", "autonomous_motion"), noun="ros", verb="decide", tier=REMOTE,
      args=_decide_args, wraps=WRAPS, doc=DOC, example=("right", "continue"),
      summary="answer a waiting arm: continue or land after an e-stop or pause; skip or redraw a crash-sent op",
      invariants=("continue never steps the arm: the driver unlatches only onto a command within 0.05 rad of its held pose.",
                  "land runs the driver's landing sequence; idle only after a verified landing."))
def ros_decide(ctx, ns, rest):
    return _backend(ctx, "decide", ns.arm, ns.decision, *rest)


@verb(effects=("network", "remote_exec", "autonomous_motion"), noun="ros", verb="cancel", tier=REMOTE,
      wraps=WRAPS, doc=DOC,
      summary="cancel every running draw or touch on the ros node; the arm holds where it is",
      invariants=("a cancelled goal stops at its next check and the arm holds; `decide land` or a new draw follows.",
                  "a dip cancelled in a cap holds there; landing is refused until the run is resumed, which withdraws it."))
def ros_cancel(ctx, ns, rest):
    return _backend(ctx, "cancel", *rest)


def _logs_args(p):
    p.add_argument("what", nargs="?", default="last", choices=("last", "list", "show", "tail"))
    p.add_argument("run_id", nargs="?", help="for show and tail")
    p.add_argument("--workflow", default="ros-draw", choices=("ros-draw", "ros-touch", "ros-stack", "ros-cli", "ros-station",
                                                                "ros-calib", "ros-register", "ros-chain"))


@verb(effects=("read_files", "network", "remote_exec"), noun="ros", verb="logs", tier=OFFLINE, args=_logs_args,
      wraps=WRAPS, doc=DOC, example=("last",),
      summary="the ros node's ros-draw, ros-touch, ros-stack, ros-station, ros-calib, ros-register and ros-chain runs; "
              "ros-cli reads "
              "this node's deploy/up/down runs")
def ros_logs(ctx, ns, rest):
    return _backend(ctx, "logs", ns.what, *([ns.run_id] if ns.run_id else []), "--workflow", ns.workflow, *rest)


def _relay_args(p):
    p.add_argument("--dest", metavar="HOST:PORT", action="append",
                   help="default: every reader's `lan` address: the e-stop's at stack.yaml estop.udp_port, the "
                        "probe's (the ros node and the arm node) at probe.udp_port; repeatable")
    p.add_argument("--bind", metavar="ADDR", help="the relay node's IPv4 source address (when it has several interfaces)")
    p.add_argument("--probe", action="store_true",
                   help="install the station probe's relay (PRB1) as its own unit; the e-stop relay is not touched")
    p.add_argument("--machine", action="store_true",
                   help="install the tattoo machine's switch (stack.yaml machine.gpio) as its own unit; run the plain "
                        "install again after it, so the e-stop relay feeds it")
    p.add_argument("--keys", action="store_true",
                   help="install the station keyboard's relay on the station node: its Fn-layer drawing keys to the "
                        "stack's keypad node (stack.yaml keypad.port)")


@verb(effects=("read_files", "network", "remote_exec", "remote_write", "start_process"), noun="ros", verb="relay-install",
      tier=REMOTE, args=_relay_args, wraps=WRAPS + ("ros/tatbot_estop_relay/tatbot_estop_relay.py", "ros/tatbot_keys_relay/tatbot_keys_relay.py"),
      doc=DOC, example=(),
      summary="install the drawing e-stop relay (EST1 over UDP), or with --probe the station probe's relay and with "
              "--machine the tattoo machine's switch, on the estop-relay node; --keys the station keyboard's keys")
def ros_relay_install(ctx, ns, rest):
    dests = [arg for d in (ns.dest or []) for arg in ("--dest", d)]
    return _backend(ctx, "relay-install", *dests, *_flags(ns, (("bind", "--bind"), ("probe", "--probe"),
                                                                ("machine", "--machine"), ("keys", "--keys"))), *rest)
