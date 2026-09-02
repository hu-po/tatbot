"""draw — map a surface with the wrist D405s, then draw on it (docs/draw.md)."""

from __future__ import annotations

import argparse
import math

from tatbot_cli.registry import MOTION_AUTO, OFFLINE, verb
from tatbot_cli.verbs._common import lerobot_py, nonce_arg, sh, tool_flag
from tatbot_cli.verbs.session import ESTOP_INV, EXCL_INV

DOC = "docs/draw.md"
SESSION_WRAPS = ("scripts/draw_run.sh", "scripts/teleop_start.sh", "cpp/teleop/wxai_teleop.cpp")

SESSION_INV = (
    ESTOP_INV, EXCL_INV,
    "Runs on the arm node; `--on <arm-node>` from another node hops there over ssh -t and arms the single-use nonce there (operator decision 2026-09-02). The e-stop operator stays at the rig.",
    "Ballpoint only (--ee-tool lutin-ballpoint-dot): the carriage-IK envelope and the executor's tip constant are its.",
    "The operator hand-guides to light contact at the design centre; READY latches, then SPACE hands over.",
    "SPACE 1 starts a preflighted camera orbit for the wrist D405s (cameras ~160 mm from the contact, patch "
    "off-axis on the pen-free side, tip up and aside); SPACE 2 starts the compiled surface path; "
    "both are refused unless the preflight passed.",
    "The surface map is a session artifact under ~/tatbot-logs/draw; it is never a repo file.",
    "Hand-eye for the D405s is nominal CAD until scripts/vision/d405_handeye_plane.py publishes a correction; "
    "absolute height comes from the operator's contact.",
    "Any refusal (stage exit 3), timeout, trip or e-stop ends at the hold with the pen retracted; "
    "scripted motion never resumes in the process.",
    "After Enter, follower then leader land through the shared staged-to-sleep routine; emergency release skips landing.",
    "The 10 Hz trace is encoder/FK evidence; the pass bar is the unwrapped paper (continuous ink, arc length within 2 %).",
)


def _bounded(label, low, high):
    def parse(value):
        number = float(value)
        if not math.isfinite(number) or not low <= number <= high:
            raise argparse.ArgumentTypeError(f"{label} must be between {low:g} and {high:g}")
        return number
    return parse


def _session_args(p):
    p.add_argument("--radius-mm", type=_bounded("--radius-mm", 2.0, 30.0), default=15.0,
                   help="final spiral radius in millimetres (default 15; the surface must be clear this far around "
                        "the touch, and --extent-mm must cover it with 10 mm to spare)")
    p.add_argument("--turns", type=_bounded("--turns", 1.0, 6.0), default=3.0, help="expanding turns (default 3)")
    p.add_argument("--draw-speed-mm-s", type=_bounded("--draw-speed-mm-s", 0.2, 8.0), default=3.5,
                   help="pen-down cruise speed; the duration follows (length / speed + ease). Default 3.5: the "
                        "first bottle draw at 0.5 mm/s was invisible, and 5 mm/s reaches the executor's joint "
                        "velocity cap when the wrist is near its singularity")
    p.add_argument("--duration-s", type=_bounded("--duration-s", 10.0, 600.0), default=None,
                   help="total pen-down duration instead of a speed (the paper A/B's 120 s form)")
    p.add_argument("--ease-s", type=_bounded("--ease-s", 0.5, 10.0), default=2.0,
                   help="quintic speed ease at each endpoint (default 2 seconds)")
    p.add_argument("--standoff-mm", type=_bounded("--standoff-mm", 50.0, 250.0), default=80.0,
                   help="orbit standoff of the tip above the contact (default 80: cameras ~210 mm from the patch, "
                        "pen shadow ~35 mm off it; see `draw viewpoints`)")
    p.add_argument("--orbit-mode", choices=("camera", "tip"), default="camera",
                   help="camera: cameras camera-distance from the contact, patch off-axis on the pen-free side, "
                        "tip well up and aside (default); tip: lift the tip standoff-mm along the normal")
    p.add_argument("--camera-distance-mm", type=_bounded("--camera-distance-mm", 120.0, 300.0), default=160.0,
                   help="camera mode: mean D405 distance from the contact (default 160)")
    p.add_argument("--off-axis-deg", type=_bounded("--off-axis-deg", 0.0, 40.0), default=35.0,
                   help="camera mode: how far the patch sits off the mean optical axis along the wide image axis "
                        "(default 35; the D405 half-field is 43. 30 looked safer for image margin but pins joint 2 "
                        "from the carriage-IK witness pose)")
    p.add_argument("--tilt-deg", type=_bounded("--tilt-deg", 0.0, 30.0), default=15.0,
                   help="orbit tilt about the contact (default 15)")
    p.add_argument("--poses", type=int, choices=range(3, 9), default=5, metavar="3..8",
                   help="orbit capture poses (default 5)")
    p.add_argument("--cell-mm", type=_bounded("--cell-mm", 0.5, 5.0), default=1.0, help="map cell (default 1.0)")
    p.add_argument("--extent-mm", type=_bounded("--extent-mm", 20.0, 150.0), default=60.0,
                   help="map extent around the contact (default 60; at least 2 x radius + 20)")
    p.add_argument("--hole-fill-mm", type=_bounded("--hole-fill-mm", 0.0, 10.0), default=0.0,
                   help="accept design samples on unmapped cells within this distance of mapped ones, as "
                        "interpolated (default 0: strict; the preflight reports interpolated_samples)")
    p.add_argument("--lean-budget-deg", type=_bounded("--lean-budget-deg", 0.0, 20.0), default=20.0,
                   help="max tool-axis lean off the local normal the preflight accepts (default 20, decision 5)")
    p.add_argument("--lean-deadband-deg", type=_bounded("--lean-deadband-deg", 0.0, 20.0), default=0.0,
                   help="let the tool lean this far off the local normal before the wrist follows it (default 0: "
                        "full normal-following, decision 3). 12 lets a 15 mm spiral on a 38 mm bottle pass the "
                        "joint-velocity cap from a touch near the wrist singularity")
    p.add_argument("--no-rerun", action="store_true", help="no viewer; the shadow is still saved as shadow.rrd")
    p.add_argument("--rerun-viewer", metavar="URL",
                   help="stream the shadow to this viewer (rerun+http://HOST:9876/proxy). Default: over --on the "
                        "launching node's `tatbot draw viewer` is found through SSH_CONNECTION; on the arm node "
                        "itself a local capped viewer starts")
    nonce_arg(p)


def _session_argv(ns, scan_only: bool) -> list[str]:
    if ns.extent_mm < 2.0 * ns.radius_mm + 20.0:
        raise SystemExit(f"tatbot draw: --radius-mm {ns.radius_mm:g} needs --extent-mm >= "
                         f"{2.0 * ns.radius_mm + 20.0:g} (got {ns.extent_mm:g}); the map must hold the design "
                         "with 10 mm to spare or the preflight refuses 'girth' after the orbit has flown")
    pace = ["--duration-s", str(ns.duration_s)] if ns.duration_s is not None else ["--draw-speed-mm-s", str(ns.draw_speed_mm_s)]
    args = ["--radius-mm", str(ns.radius_mm), "--turns", str(ns.turns), *pace,
            "--ease-s", str(ns.ease_s), "--standoff-mm", str(ns.standoff_mm), "--tilt-deg", str(ns.tilt_deg),
            "--orbit-mode", ns.orbit_mode, "--camera-distance-mm", str(ns.camera_distance_mm),
            "--off-axis-deg", str(ns.off_axis_deg),
            "--poses", str(ns.poses), "--cell-mm", str(ns.cell_mm), "--extent-mm", str(ns.extent_mm),
            "--hole-fill-mm", str(ns.hole_fill_mm), "--lean-budget-deg", str(ns.lean_budget_deg),
            "--lean-deadband-deg", str(ns.lean_deadband_deg)]
    if scan_only:
        args.append("--scan-only")
    if ns.no_rerun:
        args.append("--no-rerun")
    if ns.rerun_viewer:
        args += ["--rerun-viewer", ns.rerun_viewer]
    return args


@verb(noun="draw", verb="run", tier=MOTION_AUTO,
      summary="touch, orbit-scan with the wrist D405s, map, shadow, then draw the spiral on the mapped surface",
      role="arm", wraps=SESSION_WRAPS, passthrough="wxai_teleop", args=_session_args, needs_tool=True, nonce=True,
      example=("--radius-mm", "15", "--turns", "3", "--nonce", "pipe-draw-a"), doc=DOC, tty=True,
      invariants=SESSION_INV)
def draw_run(ctx, ns, rest):
    return sh(ctx, "scripts/draw_run.sh", *tool_flag(ctx), *_session_argv(ns, scan_only=False), *rest)


@verb(noun="draw", verb="scan", tier=MOTION_AUTO,
      summary="touch, orbit-scan and map the surface; no ink",
      role="arm", wraps=SESSION_WRAPS, passthrough="wxai_teleop", args=_session_args, needs_tool=True, nonce=True,
      example=("--standoff-mm", "80", "--poses", "5", "--nonce", "pipe-scan-a"), doc=DOC, tty=True,
      invariants=SESSION_INV + ("scan_only: the map is built and shadowed at the hold; no path is streamed.",))
def draw_scan(ctx, ns, rest):
    return sh(ctx, "scripts/draw_run.sh", *tool_flag(ctx), *_session_argv(ns, scan_only=True), *rest)


def _plan_args(p):
    p.add_argument("target", help="a draw dir (surface.npz + draw.json) to compile and preflight offline")
    p.add_argument("--out", help="where path.csv / preflight.json go (default: the draw dir)")


@verb(noun="draw", verb="plan", tier=OFFLINE, summary="compile + preflight the design on an existing surface map (no arm)",
      wraps=("scripts/draw_stage.py",), passthrough="draw_stage.py", args=_plan_args,
      example=("~/tatbot-logs/draw/20260901T190000Z-robot-a1b2",), doc=DOC,
      invariants=("Files only: reads surface.npz + draw.json, writes path.csv + preflight.json; exit 3 is a refusal "
                  "(lean_over_budget, girth, holes, ...), never a clamp.",))
def draw_plan(ctx, ns, rest):
    out = ["--out", ns.out] if ns.out else []
    return lerobot_py(ctx, "scripts/draw_stage.py", "plan", ns.target, *out, *rest)


def _shadow_args(p):
    p.add_argument("draw_dir", help="a draw dir under ~/tatbot-logs/draw")
    p.add_argument("--save", action="store_true", help="write <dir>/shadow.rrd only; no viewer")


@verb(noun="draw", verb="shadow", tier=OFFLINE, summary="open the shadow of a draw dir in a capped Rerun viewer",
      wraps=("scripts/draw_shadow.sh", "scripts/draw_shadow.py"), passthrough="draw_shadow.py", args=_shadow_args,
      example=("~/tatbot-logs/draw/20260901T190000Z-robot-a1b2",), doc=DOC,
      invariants=("Nothing here touches the arm. Captures are shown in their camera optical frames; "
                  "only the mapper places them in root.",))
def draw_shadow(ctx, ns, rest):
    if ns.save:
        return lerobot_py(ctx, "scripts/draw_shadow.py", ns.draw_dir, "--save", *rest)
    return sh(ctx, "scripts/draw_shadow.sh", ns.draw_dir, *rest)


def _viewer_args(p):
    p.add_argument("--memory-limit", default="3GB", help="Rerun viewer memory cap (default 3GB)")


@verb(noun="draw", verb="viewer", tier=OFFLINE, summary="a capped Rerun viewer here, for a draw session launched with --on",
      wraps=("scripts/draw_viewer.sh",), args=_viewer_args, tty=True, doc=DOC,
      invariants=("Nothing here touches the arm. The viewer binds every interface so the arm node can stream to it; "
                  "`draw run --on <arm-node>` from this node finds it by itself (SSH_CONNECTION).",
                  "Memory is capped (--memory-limit); keep it so."))
def draw_viewer(ctx, ns, rest):
    return sh(ctx, "scripts/draw_viewer.sh", "--memory-limit", ns.memory_limit, *rest)


def _viewpoints_args(p):
    p.add_argument("log", help="teleop flight log (.wxtl) of a hand-guided rehearsal around the target")
    p.add_argument("surface", help="surface.npz from a draw dir mapped on the same target")
    p.add_argument("--every-s", type=float, default=1.0, help="sample the log every N seconds (default 1)")
    p.add_argument("--standoffs", default="60,80,120", help="orbit standoffs (mm) to score, comma separated")
    p.add_argument("--tilt", type=float, help="orbit tilt in degrees to score (default: the draw.json default)")
    p.add_argument("--json", help="also write the scores to this file")


@verb(noun="draw", verb="viewpoints", tier=OFFLINE,
      summary="score wrist-camera viewpoints from a rehearsal log against a map, beside the orbit generator's",
      wraps=("scripts/draw_viewpoints.py",), passthrough="draw_viewpoints.py", args=_viewpoints_args,
      example=("~/tatbot-logs/teleop/teleop_20260901_171002.wxtl", "~/tatbot-logs/draw/<dir>/surface.npz"),
      doc=DOC,
      invariants=("Geometry only: FK of the logged follower joints against the map's chart, no plan and no arm.",
                  "The patch is the anchor plus an 8 mm ring; a viewpoint counts when it is inside both D405 "
                  "frustums within 70-500 mm."))
def draw_viewpoints(ctx, ns, rest):
    args = [ns.log, ns.surface, "--every-s", str(ns.every_s), "--standoffs", ns.standoffs]
    if ns.tilt is not None:
        args += ["--tilt", str(ns.tilt)]
    if ns.json:
        args += ["--json", ns.json]
    return lerobot_py(ctx, "scripts/draw_viewpoints.py", *args, *rest)


# --- vision handeye d405 (the parallel track that unblocks depth-only anchoring) -----------------


def _handeye_args(p):
    p.add_argument("captures", nargs="*", help="draw dirs or capture dirs with capture-*.npz of the touched-off paper")
    p.add_argument("--out", help="where d405_handeye.json goes (default: the first capture dir)")
    p.add_argument("--self-test", action="store_true", help="recover a known synthetic perturbation")


@verb(noun="vision", verb="handeye d405", tier=OFFLINE,
      summary="plane-route hand-eye for the wrist D405s: fit depth planes against the touched-off paper",
      wraps=("scripts/vision/d405_handeye_plane.py",), passthrough="d405_handeye_plane.py", args=_handeye_args,
      example=("--self-test",), doc=DOC,
      invariants=("Files only; never edits the URDF — it prints the <origin xyz rpy> for a human to fold in.",
                  "In-plane translation is unobservable from a plane and is regularised to zero; the report says so.",))
def vision_handeye_d405(ctx, ns, rest):
    args = list(ns.captures)
    if ns.out:
        args += ["--out", ns.out]
    if ns.self_test:
        args.append("--self-test")
    return lerobot_py(ctx, "scripts/vision/d405_handeye_plane.py", *args, *rest)
