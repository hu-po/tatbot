"""teleop · record · dip — a human on the leader arm, or a scripted dip."""

from __future__ import annotations

import argparse
import math
import sys

from tatbot_cli import EXIT_GATE_REFUSED
from tatbot_cli.registry import MOTION_AUTO, MOTION_HUMAN, OFFLINE, Plan, verb
from tatbot_cli.verbs._common import ink_argv, ink_flags, nonce_arg, py, sh, tool_flag

ESTOP_INV = "A human with the E-stop is physically present before any motion; launchers fail closed without /dev/tatbot-estop."
EXCL_INV = "wxai_teleop and LeRobot sessions are mutually exclusive (exclusive driver connection)."

# --- teleop --------------------------------------------------------------------


def _teleop_run_args(p):
    p.add_argument("duration_s", nargs="?", help="session length (record_session.sh default)")


@verb(noun="teleop", verb="run", tier=MOTION_HUMAN, summary="full session: 400 Hz C++ teleop + 7 cameras + flight log",
      role="arm", wraps=("scripts/record_session.sh", "cpp/teleop/wxai_teleop.cpp"), passthrough="wxai_teleop",
      args=_teleop_run_args, needs_tool=True, example=("60",), doc="docs/teleop_tuning.md", tty=True,
      invariants=(ESTOP_INV, EXCL_INV))
def teleop_run(ctx, ns, rest):
    pos = [ns.duration_s] if ns.duration_s else []
    return sh(ctx, "scripts/record_session.sh", *pos, *tool_flag(ctx), *rest)


def _teleop_start_args(p):
    p.add_argument("--touchoff", action="store_true",
                   help="this session IS the stated tool's touch-off: run it although workspace.yaml names another "
                        "tool (wxai_teleop --tool-uncalibrated; grip force from the datasheet)")


@verb(noun="teleop", verb="start", tier=MOTION_HUMAN, summary="the bare 400 Hz C++ teleop, live under your hands, telemetry to the viewer node — what other workflows attach to",
      role="arm", wraps=("scripts/teleop_start.sh",), passthrough="wxai_teleop", args=_teleop_start_args, needs_tool=True,
      example=(), doc="docs/teleop_tuning.md", tty=True,
      invariants=(ESTOP_INV, EXCL_INV,
                  "Interactive and foreground on purpose: Enter before the follower aligns; after an e-stop or fault, "
                  "support the arms, then Enter to idle. Ctrl+C ends it. Refuses (exit 6) while another teleop runs.",
                  "A tool workspace.yaml was not measured with is refused unless --touchoff says this session measures it.",
                  "No cameras, no session recording — `tatbot teleop run` is the full session."))
def teleop_start(ctx, ns, rest):
    flags = ["--touchoff"] if ns.touchoff else []
    return sh(ctx, "scripts/teleop_start.sh", *tool_flag(ctx), *flags, *rest)


def _teleop_square_args(p):
    def bounded(label, low, high):
        def parse(value):
            number = float(value)
            if not math.isfinite(number) or not low <= number <= high:
                raise argparse.ArgumentTypeError(f"{label} must be between {low:g} and {high:g}")
            return number
        return parse

    p.add_argument("--size-mm", type=bounded("--size-mm", 1.0, 10.0), default=6.0,
                   help="square edge in millimetres (C++ gate accepts 1..10; default 6)")
    p.add_argument("--edge-s", type=bounded("--edge-s", 2.0, 30.0), default=12.0,
                   help="seconds per edge (C++ gate accepts 2..30; default 12 = 0.5 mm/s at 6 mm)")
    nonce_arg(p)


@verb(noun="teleop", verb="square", tier=MOTION_AUTO,
      summary="hand-guide to paper, then trace one preflighted joint-space square",
      role="arm", wraps=("scripts/teleop_square.sh", "scripts/teleop_start.sh",
                         "cpp/teleop/wxai_teleop.cpp"),
      passthrough="wxai_teleop", args=_teleop_square_args, needs_tool=True, nonce=True,
      example=("--size-mm", "6", "--edge-s", "12", "--nonce", "paper-square-a"),
      doc="docs/teleop_tuning.md", tty=True,
      invariants=(ESTOP_INV, EXCL_INV,
                  "Run locally on the arm node: autonomous motion and its literal single-use nonce are refused over --on.",
                  "The operator hand-guides to light contact; READY latches after 0.2 s below 0.10 rad/s, then one SPACE transfers control.",
                  "A model-preflighted joint-position trajectory keeps start Z/orientation and grows the base-X/Y square toward the arm base before closing once.",
                  "E-stop, contact cap, measured velocity or rolling arm effort retracts the pen and terminates; no auto-resume.",
                  "After Enter, follower then leader land through the shared staged-to-sleep routine; emergency release skips landing.",
                  "Reported endpoint error is encoder/FK evidence, not independent physical ink accuracy."))
def teleop_square(ctx, ns, rest):
    return sh(ctx, "scripts/teleop_square.sh", *tool_flag(ctx),
              "--size-mm", str(ns.size_mm), "--edge-s", str(ns.edge_s), *rest)


def _teleop_spiral_args(p):
    def bounded(label, low, high):
        def parse(value):
            number = float(value)
            if not math.isfinite(number) or not low <= number <= high:
                raise argparse.ArgumentTypeError(f"{label} must be between {low:g} and {high:g}")
            return number
        return parse

    p.add_argument("--radius-mm", type=bounded("--radius-mm", 2.0, 12.0), default=6.0,
                   help="final spiral radius in millimetres (default 6; clear this distance around the center)")
    p.add_argument("--turns", type=bounded("--turns", 1.0, 6.0), default=3.0,
                   help="number of expanding turns (default 3)")
    p.add_argument("--duration-s", type=bounded("--duration-s", 30.0, 600.0), default=180.0,
                   help="total draw duration (default 180 seconds)")
    p.add_argument("--ease-s", type=bounded("--ease-s", 0.5, 10.0), default=2.0,
                   help="quintic speed ease at each endpoint (default 2 seconds)")
    p.add_argument("--carriage-ik", action="store_true",
                   help="experimental ballpoint-only seven-DOF arm/carriage A/B mode")
    nonce_arg(p)


@verb(noun="teleop", verb="spiral", tier=MOTION_AUTO,
      summary="hand-guide to a center point, then trace one slow expanding spiral",
      role="arm", wraps=("scripts/teleop_spiral.sh", "scripts/teleop_start.sh",
                         "cpp/teleop/wxai_teleop.cpp"),
      passthrough="wxai_teleop", args=_teleop_spiral_args, needs_tool=True, nonce=True,
      example=("--radius-mm", "6", "--turns", "3", "--duration-s", "180", "--ease-s", "2",
               "--nonce", "paper-spiral-a"),
      doc="docs/teleop_tuning.md", tty=True,
      invariants=(ESTOP_INV, EXCL_INV,
                  "Run locally on the arm node: autonomous motion and its literal single-use nonce are refused over --on.",
                  "The trigger point is the spiral center; leave at least the selected radius clear in every base-X/Y direction.",
                  "One SPACE starts a completely preflighted constant-Z/orientation trajectory at approximately constant arc-length speed, with short endpoint eases; no scripted auto-resume.",
                  "--carriage-ik is a ballpoint-only A/B mode: keep clear through its off-paper reversal check; it then preflights and streams a guarded seven-joint tip trajectory.",
                  "E-stop, contact cap, measured velocity or rolling arm effort retracts the pen and terminates.",
                  "After Enter, follower then leader land through the shared staged-to-sleep routine; emergency release skips landing.",
                  "The 10 Hz trace is encoder/FK evidence; ink width, continuity and physical Z remain paper evidence."))
def teleop_spiral(ctx, ns, rest):
    args = [*tool_flag(ctx), "--radius-mm", str(ns.radius_mm), "--turns", str(ns.turns),
            "--duration-s", str(ns.duration_s), "--ease-s", str(ns.ease_s)]
    if ns.carriage_ik:
        args.append("--carriage-ik")
    return sh(ctx, "scripts/teleop_spiral.sh", *args, *rest)


@verb(noun="teleop", verb="lerobot", tier=MOTION_HUMAN, summary="LeRobot teleop sanity check (no recording), wrist cams shown",
      role="arm", wraps=("scripts/il_teleop.sh",), passthrough="lerobot-teleoperate", needs_tool=True, example=(),
      doc="docs/imitation_learning.md", tty=True, invariants=(ESTOP_INV, EXCL_INV))
def teleop_lerobot(ctx, ns, rest):
    return sh(ctx, "scripts/il_teleop.sh", *tool_flag(ctx), *rest)


def _tune_args(p):
    p.add_argument("--leader-only", action="store_true")


@verb(noun="teleop", verb="tune", tier=MOTION_HUMAN, summary="teleop tuning loop + web cockpit on :8899",
      role="arm", wraps=("scripts/il_tune.sh",), passthrough="lerobot_robot_tatbot.tune", args=_tune_args,
      needs_tool=True, example=(), doc="docs/teleop_tuning.md", tty=True, invariants=(ESTOP_INV,))
def teleop_tune(ctx, ns, rest):
    flags = ["--leader-only"] if ns.leader_only else []
    return sh(ctx, "scripts/il_tune.sh", *flags, *tool_flag(ctx), *rest)


def _log_arg(p):
    p.add_argument("log", help="a .wxtl flight log (or run dir)")


@verb(noun="teleop", verb="analyze", tier=OFFLINE, summary="loop-timing stats from a .wxtl without a GUI",
      wraps=("cpp/teleop/analyze_log.py",), args=_log_arg, example=("~/tatbot-logs/teleop/last/teleop.wxtl",))
def teleop_analyze(ctx, ns, rest):
    return py(ctx, "cpp/teleop/analyze_log.py", ns.log, *rest)


@verb(noun="teleop", verb="replay", tier=OFFLINE, summary="replay a flight log in Rerun",
      wraps=("cpp/teleop/rerun-importer-wxtl",), args=_log_arg, example=("~/tatbot-logs/teleop/last/teleop.wxtl",))
def teleop_replay(ctx, ns, rest):
    return Plan(argv=["rerun", ns.log, *rest])


# --- record --------------------------------------------------------------------


def _record_args(p):
    p.add_argument("dataset", help="dataset name (pushed to the hub when logged in)")
    p.add_argument("task", help="task description in quotes")
    p.add_argument("-n", "--episodes", type=int, default=5)
    p.add_argument("--push", action="store_true",
                   help="publish the dataset to the hub (default: record locally only)")
    ink_flags(p)
    nonce_arg(p)


@verb(noun="record", verb="", tier=MOTION_HUMAN, summary="record IL episodes with the LeRobot plugin (leader→follower)",
      role="arm", wraps=("scripts/il_record.sh",), passthrough="lerobot-record", args=_record_args, needs_tool=True,
      dip_hook=True, example=("squiggle-v3", '"draw a square"', "-n", "5"), doc="docs/imitation_learning.md", tty=True,
      invariants=(ESTOP_INV, EXCL_INV, "During recording: → / n next episode, ← / r re-record, ESC / q stop.",
                  "Datasets stay LOCAL unless --push is given; publish later with `tatbot data hub push`.",
                  "--dip runs a scripted dip before the human takes the leader arm: that is autonomous motion, "
                  "so it needs --nonce and is refused over --on."))
def record(ctx, ns, rest):
    push = ["--push"] if ns.push else []
    return sh(ctx, "scripts/il_record.sh", ns.dataset, ns.task, str(ns.episodes),
              *push, *tool_flag(ctx), *ink_argv(ns), *rest)


# --- dip -----------------------------------------------------------------------
#
# Three spellings, three different things. In the ink vocabulary (docs/ink.md)
# a REHEARSAL is the ballpoint physically dipping into dry caps — the same
# choreography the 3RL will make, with nothing to spill — and the ledger has
# to show one before a real needle may dip. Until 2026-08-29 `--rehearse`
# mapped to il_dip's `--dry-run`, which prints the plan and moves nothing, so
# the documented example could never satisfy the documented invariant.


def _dip_args(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true",
                   help="print the plan, the session and both rack estimates; nothing moves (il_dip --dry-run)")
    g.add_argument("--connect-only", action="store_true",
                   help="the plan, then connect: e-stop, driver, controller state, live tip vs every planned point; "
                        "the arm is never commanded")
    g.add_argument("--rehearse", action="store_true",
                   help="the rehearsal: a `rehearsal` tool (the ballpoint) dips into dry caps, ledgered")
    g.add_argument("--yes", action="store_true", help="move the arm with the stated tool; a `real` tool also needs --allow-real")
    p.add_argument("--allow-real", action="store_true", help="permit a `real` ink tool (pigment) — with --yes")
    nonce_arg(p)


def _ink_mode(ctx, tool_id: str) -> str | None:
    """The stated tool's ink.mode from its datasheet (real | rehearsal | none), or None if unreadable."""
    sys.path.insert(0, ctx.path("scripts/lib"))
    try:
        import ink_spec
        import tool_spec
        return ink_spec.policy_for(tool_spec.load_tool(tool_id, ctx.repo)).mode
    except Exception:
        return None


@verb(noun="dip", verb="", tier=MOTION_AUTO, summary="scripted, e-stop-monitored, ledgered dip into the palette caps",
      role="arm", wraps=("scripts/il_dip.sh", "scripts/il_dip.py"), passthrough="il_dip.py", args=_dip_args,
      needs_tool=True, nonce=True, nonce_exempt=("plan", "connect_only"), example=("--plan",), doc="docs/ink.md",
      invariants=(ESTOP_INV,
                  "--plan and --connect-only command nothing; --rehearse is the ballpoint into dry caps; "
                  "--yes --allow-real is pigment.",
                  "A real tool is refused until the ledger shows a ballpoint dry-cap rehearsal.",
                  "Every moving dip consumes one single-use --nonce (arm_gate), inherited when a launcher's --dip runs it."))
def dip(ctx, ns, rest):
    from tatbot_cli.cli import refuse

    if ns.plan:
        return sh(ctx, "scripts/il_dip.sh", *tool_flag(ctx), "--dry-run", *rest)
    if ns.connect_only:
        return sh(ctx, "scripts/il_dip.sh", *tool_flag(ctx), "--connect-only", *rest,
                  notes=["reads the arm and the e-stop; commands nothing"])
    if not (ns.rehearse or ns.yes):
        return refuse(ctx, EXIT_GATE_REFUSED, "dip", "say what this dip is: --plan, --connect-only, --rehearse or --yes",
                      "tatbot dip --explain")
    mode = _ink_mode(ctx, ctx.ee_tool)
    if mode == "none":
        return refuse(ctx, EXIT_GATE_REFUSED, "ink_mode", f"{ctx.ee_tool} has ink.mode none; it never dips",
                      "fit the ballpoint (lutin-ballpoint-dot) for a rehearsal, or the needle for a real dip")
    if ns.rehearse and mode != "rehearsal":
        return refuse(ctx, EXIT_GATE_REFUSED, "ink_mode",
                      f"a rehearsal is a `rehearsal` tool into dry caps; {ctx.ee_tool} is ink.mode {mode}",
                      "tatbot --ee-tool lutin-ballpoint-dot dip --rehearse --nonce <literal>")
    if ns.yes and mode == "real" and not ns.allow_real:
        return refuse(ctx, EXIT_GATE_REFUSED, "ink_mode",
                      f"{ctx.ee_tool} is a real needle (ink.mode real): pigment, real caps, real stock",
                      "add --allow-real if the ballpoint rehearsal is on record and the caps are loaded")
    flags = ["--yes"] + (["--allow-real"] if ns.allow_real else [])
    notes = [] if ns.yes else [f"rehearsal: {ctx.ee_tool} dips into dry caps; the event is ledgered as mode rehearsal"]
    return sh(ctx, "scripts/il_dip.sh", *tool_flag(ctx), *flags, *rest, notes=notes)
