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


def _arm(p, what: str) -> None:
    """--arm. Not given, the backend drives the arm that draws: the first of stack.yaml `arms` (tatbot_cli.stack)."""
    p.add_argument("--arm", help=f"{what} (default: the drawing arm, the first of stack.yaml arms)")


def _arm_flag(ns) -> list[str]:
    return ["--arm", ns.arm] if ns.arm else []


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
    p.add_argument("--at", metavar="X,Y", help="put the canvas centre here, mm in the page frame (x right, y toward the "
                                               "top of the print; write --at=-15,42 for a negative x)")
    p.add_argument("--width", metavar="MM", help="assert acquired canvas width in mm; regenerate to resize")
    p.add_argument("--stencil", metavar="DIR", help="the generated print it is drawn on (its directory): place the "
                                                    "artwork inside that print's clear centre, not the nominal page's")


def _cartridge_args(p):
    p.add_argument("action", choices=("status", "fit"),
                   help="status: the fitted cartridge, how its tip stands, what the station touch and wrist gauges "
                        "were measured with, and what to measure next; fit: record the cartridge now in the mount")
    p.add_argument("tool", nargs="?", default="", help="fit: its datasheet in config/tools/ (tatbot tool list)")
    _arm(p, "the arm whose mount holds it")
    p.add_argument("--instance", default="", help="fit: a label for this physical cartridge, kept with it")


@verb(effects=("read_files", "network", "remote_exec", "remote_write", "stop_process", "autonomous_motion"),
      noun="ros", verb="cartridge", tier=REMOTE, args=_cartridge_args, wraps=(*WRAPS, "scripts/lib/tatbot_cli/ros_cartridge.py"),
      output="json", doc=DOC, example=("fit", "lutin-5rl-mast", "--arm", "xarm"),
      summary="record the cartridge in an arm's mount on the ROS owner, starting at its type's last measured tip or "
              "its datasheet's nominal, or show it and what was measured with it",
      invariants=("Writes the owner's calib/cartridge-<arm>.json, which deploy never touches, over "
                  "config/workspace.yaml's tool and tip: no commit, land or deploy.",
                  "fit restarts a running stack with its own flags so it plans with the cartridge, its arms landed "
                  "first; refused (exit 6) while a draw, touch or calibration runs there.",
                  "Measures nothing: status names the calibrations a fresh cartridge still needs."))
def ros_cartridge(ctx, ns, rest):
    flags = [*_arm_flag(ns), *(["--instance", ns.instance] if ns.instance else []), *(["--json"] if ctx.json else [])]
    return _backend(ctx, "cartridge", ns.action, *([ns.tool] if ns.tool else []), *flags, *rest)


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
    p.add_argument("--arms", metavar="LIST", help="comma-separated arms; default: stack.yaml arms")
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
      invariants=("--hardware real connects to the configured arms only (stack.yaml arms, or --arms).",
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
    _arm(p, "the arm whose base the pose is given in")
    p.add_argument("--seconds", type=float, default=5.0, help="how long to watch the tracker (default 5)")


@verb(effects=("network", "remote_exec", "sensor_read"), noun="ros", verb="page", tier=SENSOR, args=_page_args,
      output="json", wraps=WRAPS, doc=DOC, example=("--seconds", "10"),
      summary="the stencil page as the overhead tracker sees it now: measured or lost, its pose in the arm's base, "
              "and how far it stands from the page the newest set-up placed; moves nothing")
def ros_page(ctx, ns, rest):
    return _backend(ctx, "page", *_arm_flag(ns), "--seconds", str(ns.seconds), *(["--json"] if ctx.json else []),
                    *rest)


def _station_args(p):
    _arm(p, "the arm whose frame the station is placed in (its registration)")
    p.add_argument("--against", default="",
                   help="an earlier ros-station run id or station.json on the ros node: exit 1 when the station moved")


@verb(effects=("network", "remote_exec", "sensor_read", "write_files"), noun="ros", verb="station", tier=SENSOR,
      args=_station_args, wraps=WRAPS, doc=DOC, example=("--arm", "xarm"),
      summary="the station in an arm's frame, measured now from the palette's roof tag, never a stored pose")
def ros_station(ctx, ns, rest):
    return _backend(ctx, "station", *_arm_flag(ns), *(["--against", ns.against] if ns.against else []), *rest)


def _register_args(p):
    _arm(p, "the arm to register (the stack must be up with it)")
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
      tier=MOTION_AUTO, args=_register_args, wraps=WRAPS, doc=DOC, example=("--arm", "xarm", "--adopt"),
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
    return _backend(ctx, "register", *_arm_flag(ns), *flags, *(["--adopt"] if ns.adopt else []),
                    *(["--plan-only"] if ns.plan_only else []), *rest)


def _setup_args(p):
    _arm(p, "the arm to set up")
    p.add_argument("--hardware", choices=("mock", "fake", "real"), help="default: stack.yaml hardware (mock)")
    p.add_argument("--estop", choices=("none", "serial", "udp"), help="e-stop source; default: stack.yaml estop.source")
    p.add_argument("--relay-addr", metavar="ADDR", help="udp: the relay's source address (default: role estop-relay `lan`)")
    p.add_argument("--pattern", metavar="ID", help="the print on the skin, installed on the ros node; default: "
                                                   "stack.yaml page.pattern_id")
    p.add_argument("--surface", choices=("plane", "fitted"), help="fitted: a curved page; default: stack.yaml")
    p.add_argument("--arms", metavar="LIST", help="comma-separated arms the stack comes up with; default: the arm "
                                                  "being set up")
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
    return _backend(ctx, "setup", *_arm_flag(ns), *flags, *(["--rerun"] if ns.rerun else []),
                    *(["--from", ns.start] if ns.start else []), *(["--detach"] if ns.detach else []), *rest)


def _chain_args(p):
    p.add_argument("runs", nargs="+", help="ros-register run ids on the ros node, all of one arm, pooled")
    _arm(p, "the arm they registered")


@verb(effects=("read_files", "write_files", "network", "remote_exec"), noun="ros", verb="chain", tier=OFFLINE,
      args=_chain_args, wraps=WRAPS, doc=DOC, example=("<run-id>", "<run-id>"),
      summary="what an arm's registrations leave: the wrist tags' seat and the joints' offsets, scored held out",
      invariants=("Nothing moves and nothing is adopted: chain.json and a wrist layout candidate go to its run log.",
                  "A model is scored on runs it was not fitted to; the best is the least held-out corner error."))
def ros_chain(ctx, ns, rest):
    return _backend(ctx, "chain", *_arm_flag(ns), *ns.runs, *rest)


def _calib_args(p):
    p.add_argument("action", choices=("run", "sweep", "height", "wristcam", "gauge", "overhead", "apply"),
                   help="run: measure the arm's tool tip across its axis on the station probe; sweep: measure it "
                        "contact-free from the last wrist joint's turns and tilts about the tip the palette camera "
                        "watches, the tilts its length; height: the station touch's ball height from the palette "
                        "camera, the tip held over it; wristcam: a wrist camera on the tool mount from the palette's "
                        "tag (--adopt installs its gauge file: a camera's first); gauge: the wrist gauge's tip "
                        "from known heights over the probe body's top, adopted when they agree; overhead: the overhead "
                        "D555 registered through the wrist D405 (the roof tag, the holder's tile, both depths), not the "
                        "wrist cube; apply: adopt a run's tip")
    p.add_argument("--run", default="", help="apply: the ros-calib run id whose candidate to adopt")
    _arm(p, "the arm to calibrate (the stack must be up with it and --probe)")
    p.add_argument("--attitudes", default="", help="upright yaws as yaw/0 degrees about the base heading, comma "
                   "separated; none: S1 alone")
    p.add_argument("--heading-deg", dest="heading_deg", default="", help="the base attitude's tool heading")
    p.add_argument("--station-only", dest="station_only", action="store_true",
                   help="run: S1 alone, the station touch palette dips aim by; no tip is fitted")
    p.add_argument("--across-only", dest="across_only", action="store_true",
                   help="apply: a sweep's tip across its axis only, the installed length kept (a fresh cartridge's "
                        "longer tube, its needles reaching that much less past it)")
    p.add_argument("--derive-gauges", dest="derive_gauges", action="store_true",
                   help="apply: move each wrist gauge's tip as the adopted tip moved, through its camera's hand-eye, "
                        "in place of measuring it again per camera; without it apply only reports what each would "
                        "become")
    p.add_argument("--pattern", default="", help="overhead: the fiducial tile on the holder (default: the stack's)")
    p.add_argument("--camera", default="", help="wristcam, gauge: the wrist D405 by its vision.toml role (wrist_1, "
                   "wrist_2; default: the session's)")
    p.add_argument("--adopt", action="store_true", help="overhead: install the registration and restart its readers; "
                   "wristcam: install the camera's gauge file")


@verb(effects=("network", "remote_exec", "remote_write", "autonomous_motion", "sensor_read", "write_files",
               "stop_process"), noun="ros", verb="calib",
      tier=MOTION_AUTO, args=_calib_args, wraps=WRAPS, doc=DOC, example=("run", "--arm", "xarm"),
      summary="measure a tool tip across its axis on the station probe (station fix, side pairs at upright yaws, fit), "
              "or contact-free, its length too, from joint-6 turns and tilts under the palette camera (sweep); apply "
              "adopts a run's tip",
      invariants=("The tool is the arm's fitted cartridge (tatbot ros cartridge); a stated --ee-tool must be that one.",
                  "apply writes the tip into the ROS owner's fitted-cartridge record, no commit or deploy, and restarts "
                  "a running stack with its own flags so it plans with it; a wristcam run's palette tag seat goes into "
                  "this checkout's palette design.",
                  "apply leaves the wrist gauge files as they are unless --derive-gauges is given; it always reports "
                  "how each would follow the tip.",
                  "A tool whose datasheet declares no probe contact model is refused (exit 3) before anything moves.",
                  "sweep: every hold is a probe-guarded Touch MODE_MOVE; the turns and tilts go only once the park's "
                  "still shows the tip clear of the ball's top."))
def ros_calib(ctx, ns, rest):
    flags = [part for flag, value in (("--ee-tool", ctx.ee_tool), ("--attitudes", ns.attitudes),
                                      ("--heading-deg", ns.heading_deg), ("--run", ns.run), ("--pattern", ns.pattern),
                                      ("--camera", ns.camera))
             if value for part in (flag, value)] + (["--station-only"] if ns.station_only else []) + \
        (["--across-only"] if ns.across_only else []) + (["--derive-gauges"] if ns.derive_gauges else []) + \
        (["--adopt"] if ns.adopt else [])
    return _backend(ctx, "calib", ns.action, *_arm_flag(ns), *flags, *rest)


def _compile_args(p):
    p.add_argument("artwork", nargs="+", help="acquired DBV3 artwork.json; several (one per pen) are layers, drawn "
                                              "in the order given")
    _arm(p, "whose fitted tool compiles it")
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
    p.add_argument("--after", metavar="PROGRAM", action="append",
                   help="a program.json already drawn on this print (repeat, oldest first): a lighter ink is refused "
                        "over a darker one's mark it drew, and its ink counts against the --stencil print's markers")
    p.add_argument("--wiped", action="store_true",
                   help="the print was wiped after the last --after program, so its ink no longer refuses this one's")
    p.add_argument("--allow-borrowed-dip", action="store_true",
                   help="draw with a cartridge whose datasheet says its dip values are another's, which is otherwise "
                        "refused until its dip card is read (tatbot lab propose --dip-card)")
    p.add_argument("--allow-marker-loss", action="store_true",
                   help="write a program whose ink leaves the --stencil print's markers under the marker budget "
                        "(stack.yaml page.markers): for a sheet fixed on its mount, drawn without tracking it")
    p.add_argument("-o", "--out", metavar="DIR", help="where program.json and preview.svg go (default .)")
    _place(p)


@verb(effects=("read_files", "write_files"), noun="ros", verb="compile", tier=OFFLINE, args=_compile_args,
      wraps=WRAPS, doc=DOC, example=("artwork.json", "--at=0,0"),
      summary="prepare acquired DBV3 artwork into program.json + preview.svg (tatbot_ink; no arm, no ROS)",
      invariants=("With --stencil naming a fiducial-tile print, a design whose ink would leave the cameras fewer of "
                  "its markers than stack.yaml's page.markers budget is refused, naming the stage and the markers "
                  "lost; --allow-marker-loss writes it anyway.",))
def ros_compile(ctx, ns, rest):
    return _backend(ctx, "compile", *ns.artwork, *_arm_flag(ns),
                    *(["--ee-tool", ctx.ee_tool] if ctx.ee_tool else []),
                    *_flags(ns, (("speed", "--speed"), ("inks", "--inks"), ("max_segment_s", "--max-segment-s"),
                                 ("min_stroke_mm", "--min-stroke-mm"), ("mark_width_mm", "--mark-width-mm"),
                                 ("max_stroke_mm", "--max-stroke-mm"), ("out", "--out"))), *_place_flags(ns),
                    *([f"--keep-region={ns.keep_region}"] if ns.keep_region else []),
                    *[flag for path in ns.after or [] for flag in ("--after", path)],
                    *(["--wiped"] if ns.wiped else []),
                    *(["--allow-borrowed-dip"] if ns.allow_borrowed_dip else []),
                    *(["--allow-marker-loss"] if ns.allow_marker_loss else []), *rest)


SHEET_AT = ("X,Y", "the page is the sheet of the print nearest this point, mm in <arm>/base_link, from now on: "
                        "`tatbot ros page` says where each sheet in view stands (write --sheet-at=-40,520)")


def _draw_args(p):
    p.add_argument("program", nargs="?", help="program.json (or acquired artwork.json, prepared first)")
    _arm(p, "the arm that draws")
    p.add_argument("--from-op", metavar="ID", help="start at this op id")
    p.add_argument("--resume", metavar="RUN_ID", help="resume that run's ledger (after a cartridge swap, a landing)")
    p.add_argument("--sheet-at", dest="sheet_at", metavar=SHEET_AT[0], help=SHEET_AT[1])
    p.add_argument("--no-inspect", action="store_true", help="skip the wrist-camera look after a complete draw")
    p.add_argument("--rehearse", action="store_true",
                   help="draw the same program in the air: page setup, gauge holds, hover stops and the stencil's "
                        "follow as with ink, the pen held over the page, the machine off, no dips, no inspection")
    p.add_argument("--standoff-mm", metavar="MM",
                   help="with --rehearse: the pen this far over the measured page (default motion.yaml pen.hover, 6)")
    p.add_argument("--dry-dips", metavar="CAP",
                   help="with --rehearse: keep the program's dips, each to this cap's hover and back (L1 ... S2), "
                        "never into it, the machine off: the dips' travel without ink")
    p.add_argument("--real-dips", action="store_true",
                   help="with --rehearse: keep the program's dips and rinses, each into its own cap as with ink, "
                        "the machine off: the dips practised over empty caps declared with their inks")
    p.add_argument("--trim-mm", metavar="MM",
                   help="start the run's cartridge at this pen trim, mm (negative is deeper; write --trim-mm=-1.75): "
                        "as if Fn+S were pressed before the first stroke; a new run only")
    p.add_argument("--ends", choices=("stop", "fly"),
                   help="draw this run's stroke ends stopping on the skin (today's) or moving (the pen eases in the "
                        "air and comes down and leaves at the draw speed), over motion.yaml draw.ends; a new run only")
    p.add_argument("--hold", action="store_true",
                   help="leave the arm holding at rest after the draw (default: land it, controller idle)")
    p.add_argument("--no-wake", action="store_true", help="do not automatically restart a landed arm")
    p.add_argument("--speed", metavar="MM_S", help="draw speed in mm/s when compiling artwork (default 3.5)")
    p.add_argument("--detach", action="store_true",
                   help="return at once instead of following its log (it always runs in its own session, so a "
                        "closed terminal or an agent's tool timeout cannot stop it partway); --progress follows it")
    p.add_argument("--progress", action="store_true",
                   help="read only: the newest draw from this computer, running or finished, in a few lines")
    _place(p)


@verb(effects=("read_files", "write_files", "network", "remote_exec", "autonomous_motion"), noun="ros", verb="draw",
      tier=MOTION_AUTO, args=_draw_args, wraps=WRAPS, doc=DOC, example=("program.json", "--arm", "xarm"),
      summary="draw a program with the running stack: page touches, then every op; waits on pause and e-stop",
      invariants=("The next op is the first op of the ledger that is not done; a crash-sent op waits for decide.",
                  "An e-stop press holds the arm; after release the arm waits for decide continue or land.",
                  "A tool change to another cartridge lands the arm; --resume says it is fitted and keeps the page."))
def ros_draw(ctx, ns, rest):
    if ns.progress:
        return _backend(ctx, "draw", "--progress", *(["--json"] if ctx.json else []), *rest)
    return _backend(ctx, "draw", *([ns.program] if ns.program else []), *_arm_flag(ns),
                    *(["--detach"] if ns.detach else []),
                    *(["--ee-tool", ctx.ee_tool] if ctx.ee_tool else []),
                    *_flags(ns, (("from_op", "--from-op"), ("resume", "--resume"), ("no_inspect", "--no-inspect"),
                                 ("hold", "--hold"), ("no_wake", "--no-wake"), ("speed", "--speed"),
                                 ("rehearse", "--rehearse"), ("standoff_mm", "--standoff-mm"),
                                 ("dry_dips", "--dry-dips"), ("real_dips", "--real-dips"))),
                    *([f"--trim-mm={ns.trim_mm}"] if ns.trim_mm else []),
                    *([f"--ends={ns.ends}"] if ns.ends else []),
                    *([f"--sheet-at={ns.sheet_at}"] if ns.sheet_at else []), *_place_flags(ns), *rest)


def _touch_args(p):
    _arm(p, "the arm that touches")
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
    p.add_argument("--sheet-at", dest="sheet_at", metavar=SHEET_AT[0], help=SHEET_AT[1])


@verb(effects=("network", "remote_exec", "autonomous_motion"), noun="ros", verb="touch", tier=MOTION_AUTO,
      args=_touch_args, wraps=WRAPS, doc=DOC, example=("--arm", "xarm"),
      summary="guarded touches: three on the page (its plane beside the camera's), or one from --start")
def ros_touch(ctx, ns, rest):
    single = [part for flag, value in (("--start", ns.start), ("--rpy", ns.rpy), ("--direction", ns.direction))
              if value for part in (flag, *map(str, value))]
    single += [part for flag, value in (("--prior", ns.prior), ("--speed", ns.speed), ("--max-travel", ns.max_travel))
               if value for part in (flag, str(value))]
    flags = [flag for flag, on in (("--probe", ns.probe), ("--move", ns.move)) if on]
    flags += [f"--sheet-at={ns.sheet_at}"] if ns.sheet_at else []
    return _backend(ctx, "touch", *_arm_flag(ns), *single, *flags, *rest)


def _jog_args(p):
    _arm(p, "the arm to jog")
    p.add_argument("--joint", type=int, required=True, help="joint index 0-5 (rad), or 6 for the carriage (m)")
    p.add_argument("--delta", type=float, required=True, help="how far: rad, or m for the carriage")
    p.add_argument("--speed", type=float, help="peak speed in rad/s (default 0.02; the carriage 0.002 m/s)")


@verb(effects=("network", "remote_exec", "autonomous_motion"), noun="ros", verb="jog", tier=MOTION_AUTO,
      args=_jog_args, wraps=WRAPS, doc=DOC, example=("--joint", "0", "--delta", "0.02"),
      summary="move one joint by a small delta from its measured position, through the trajectory controller",
      invariants=("A quintic joint move from the measured joints: no step, zero velocity at both ends.",
                  "An e-stop press holds the arm mid-jog like any other goal."))
def ros_jog(ctx, ns, rest):
    return _backend(ctx, "jog", *_arm_flag(ns), "--joint", str(ns.joint), "--delta", str(ns.delta),
                    *_flags(ns, (("speed", "--speed"),)), *rest)


def _inspect_args(p):
    p.add_argument("--depth-only", action="store_true", help="retain native RGB-D at the current held pose without moving")
    _arm(p, "the arm whose wrist camera looks")
    p.add_argument("--run", help="the ros-draw run to look at (default the newest)")
    p.add_argument("--frames", type=int, help="frames per pose (default 5)")
    p.add_argument("--poses", type=int, help="camera poses (default 3)")


@verb(effects=("network", "remote_exec", "autonomous_motion"), noun="ros", verb="inspect", tier=MOTION_AUTO,
      args=_inspect_args, wraps=WRAPS, doc=DOC, example=(),
      summary="hover the wrist camera over the newest drawing and write page-frame views with the plan overlaid",
      invariants=("Lifts the pen straight up first, then joint moves with the pen at least 15 mm over the page.",
                  "Writes <run>/inspect/<time>/: raw frames, page.png, overlay.png, meta.json."))
def ros_inspect(ctx, ns, rest):
    return _backend(ctx, "inspect", *_arm_flag(ns), *_flags(ns, (("run", "--run"), ("frames", "--frames"),
                                                                  ("poses", "--poses"), ("depth_only", "--depth-only"))), *rest)


def _views_args(p):
    _arm(p, "the arm whose wrist cameras look")
    p.add_argument("--label", required=True, help="names the capture: ~/tatbot-logs/ros-views/<label> on the ros node")
    p.add_argument("--points", help="page points x,y;x,y in mm to hold the pen over (none: the view where the arm "
                                    "stands, no motion); write --points=-40,52;... for a negative x")
    p.add_argument("--run", help="the ros-draw or ros-touch run whose page to aim by (default the newest)")
    p.add_argument("--frames", type=int, help="frames a view and camera, their median kept (default 9)")
    p.add_argument("--fetch", help="copy the frames and views.json into this directory here")
    p.add_argument("--hold", dest="land", action="store_false", help="leave the arm holding at rest after")
    p.add_argument("--no-wake", dest="wake", action="store_false", help="do not wake a landed arm first")
    p.add_argument("--overhead", action="store_true", help="the overhead D555's view too, taken first")
    p.add_argument("--no-wrist", dest="wrist", action="store_false",
                   help="no wrist views: with --overhead, a capture that never touches the arm")
    p.add_argument("--hover-mm", type=float, help="the pen's height over the page at a hold (default the close-up's, "
                                                  "page.inspect.closeup_hover_m)")
    p.add_argument("--overhead-pages", help="aim by the overhead tracker, no set-up: a hold over every sheet of this "
                                            "print it reads (pattern id or leading hex), at --points or its centre")
    p.add_argument("--sheet-at", dest="sheet_at", metavar=SHEET_AT[0],
                   help="with --overhead-pages: only the sheet nearest this point, mm in <arm>/base_link")
    p.add_argument("--seconds", type=float, help="with --overhead-pages: how long to wait for the tracker to measure a "
                                                 "sheet (default 5)")


@verb(effects=("network", "remote_exec", "autonomous_motion", "read_files", "write_files"), noun="ros", verb="views",
      tier=MOTION_AUTO, args=_views_args, wraps=(*WRAPS, "ros/tatbot_session/tatbot_session/views.py"), doc=DOC,
      example=("--label", "card-p0001-run-01-before", "--points=-40,52;0,52", "--hold"),
      summary="photograph page points with every wrist camera from a close-up hold over each, and the overhead, for the lab",
      invariants=("Every hold is planned and its wrist checked before the arm moves; a view moves the pen down to the "
                  "close-up hover height, never into the page.",
                  "Writes v<NN>-<role>.png and views.json; an after capture names the before's page run so the poses "
                  "match."))
def ros_views(ctx, ns, rest):
    args = ["views", *_arm_flag(ns), "--label", ns.label, *_flags(ns, (("run", "--run"), ("frames", "--frames"),
                                                                        ("fetch", "--fetch"),
                                                                        ("overhead_pages", "--overhead-pages")))]
    args += [f"--points={ns.points}"] if ns.points else []   # one argument: a negative x is no flag
    args += [f"--sheet-at={ns.sheet_at}"] if ns.sheet_at else []
    args += [] if ns.land else ["--hold"]
    args += [] if ns.wake else ["--no-wake"]
    args += ["--overhead"] if ns.overhead else []
    args += [] if ns.wrist else ["--no-wrist"]
    args += [f"--hover-mm={ns.hover_mm:g}"] if ns.hover_mm is not None else []
    args += [f"--seconds={ns.seconds:g}"] if ns.seconds is not None else []
    return _backend(ctx, *args, *rest)


def _decide_args(p):
    p.add_argument("arm", help="the waiting arm (xarm, right)")
    p.add_argument("decision", choices=("continue", "land", "skip", "redraw"))


@verb(effects=("network", "remote_exec", "autonomous_motion"), noun="ros", verb="decide", tier=REMOTE,
      args=_decide_args, wraps=WRAPS, doc=DOC, example=("xarm", "continue"),
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


def _timing_args(p):
    p.add_argument('run', help='local ros-draw run directory containing pipeline-timing.jsonl')
    p.add_argument('--write', action='store_true', help='save the JSON summary and Markdown report beside the log')


@verb(effects=('read_files', 'write_files'), noun='ros', verb='timing', tier=OFFLINE, args=_timing_args, output='json',
      wraps=('python/tatbot_contracts/src/tatbot_contracts/ros_pipeline.py',), doc=DOC,
      example=('/path/to/ros-draw/run',),
      summary='camera-to-controller timing and observation lineage from a local drawing run',
      invariants=('Reads retained evidence; missing timing stays unknown.',
                  'Frame ages start at host receipt; device clocks and controller clocks are never subtracted.'))
def ros_timing(ctx, ns, rest):
    return py(ctx, 'python/tatbot_contracts/src/tatbot_contracts/ros_pipeline.py', ns.run,
              *(['--json'] if ctx.json else []), *(['--write'] if ns.write else []), *rest)
