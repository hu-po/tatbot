"""vision · depth · audio · net — cameras, calibration, tracking, deploy, and the small sensors."""

from __future__ import annotations

import os

from tatbot_cli import nodes
from tatbot_cli.registry import MOTION_HUMAN, MUTATES_CONFIG, OFFLINE, REMOTE, SENSOR, Plan, verb
from tatbot_cli.verbs._common import py, sh, tool_flag

CALIB_INV = ("config/fiducials.json is the only hand-edited fiducial inventory.",
             "Golden copies live outside the repo: ~/tatbot-logs/vision/calibration-current.json, robot-world-current.json.")

# --- vision calib ----------------------------------------------------------------


def _sweep_args(p):
    p.add_argument("seconds", nargs="?", help="session budget")
    p.add_argument("--phases", help="board,wrist,tip")
    p.add_argument("--profile", choices=("debug", "full"))
    p.add_argument("--no-arm", action="store_true", help="cameras only — no teleop, no tip phase")


@verb(noun="vision", verb="calib sweep", tier=MOTION_HUMAN, summary="voice-guided field calibration: board / wrist / tip phases",
      role="calib-conductor", wraps=("scripts/vision/calib_sweep.sh", "scripts/vision/session_guide.py"),
      passthrough="calib_sweep.sh", args=_sweep_args, example=("--phases", "board", "--no-arm"), doc="docs/fiducials.md",
      tty=True, invariants=CALIB_INV + ("--ee-tool is required for any tip phase; it names what is in the gripper.",))
def calib_sweep(ctx, ns, rest):
    args = [ns.seconds] if ns.seconds else []
    if ns.phases:
        args.append(f"--phases={ns.phases}")
    if ns.profile:
        args.append(f"--profile={ns.profile}")
    if ns.no_arm:
        args.append("--no-arm")
    if ctx.ee_tool:
        args.append(f"--ee-tool={ctx.ee_tool}")
    return sh(ctx, "scripts/vision/calib_sweep.sh", *args, *rest)


def _session_arg(p):
    p.add_argument("session_dir")


def _calib_offline(name, rel, summary):
    @verb(noun="vision", verb=f"calib {name}", tier=OFFLINE, summary=summary, wraps=(rel,), passthrough=rel.split("/")[-1],
          args=_session_arg, example=("~/tatbot-logs/vision/calib-sessions/sweep-20260826_122203",), doc="docs/vision.md",
          invariants=CALIB_INV)
    def _fn(ctx, ns, rest):
        return py(ctx, rel, ns.session_dir, *rest)
    _fn.__name__ = f"calib_{name}"
    return _fn


_calib_offline("fuse", "scripts/vision/fuse_session.py", "fuse a sweep's streams into solver-ready samples")
_calib_offline("solve", "scripts/vision/solve_robot_world.py", "robot-world calibration from wrist tags")
_calib_offline("verify", "scripts/vision/verify_calibration.py", "independently verify a board-session calibration")
_calib_offline("report", "scripts/vision/calibrate_session.py", "offline pipeline + report card for a session")
_calib_offline("board", "scripts/vision/calibrate_board_session.py", "full-rig calibration from a guided board session")


def _touchoff_args(p):
    p.add_argument("target", help="session dir (touches.json / teleop.wxtl) or a .wxtl")
    p.add_argument("--write", action="store_true", help="write config/workspace.yaml (else print only)")


@verb(noun="vision", verb="touchoff", tier=MUTATES_CONFIG, summary="solve the pen-tip offset and paper plane from touch-off samples",
      wraps=("scripts/il_touchoff.py",), passthrough="il_touchoff.py", args=_touchoff_args, needs_tool=True,
      example=("~/tatbot-logs/vision/calib-sessions/sweep-20260826_122203",), doc="docs/tools.md",
      invariants=("Nothing here moves the arm.", "--write records the stated tool in config/workspace.yaml; a swap invalidates it."))
def touchoff(ctx, ns, rest):
    flag = ["--write"] if ns.write else []
    # The solver needs numpy (il_touchoff.py:54); bare python3 lacks it.
    return _vision_py(ctx, "scripts/il_touchoff.py", ns.target, *tool_flag(ctx), *flag, *rest)


# --- live views ------------------------------------------------------------------


def _live_args(p):
    p.add_argument("stream", nargs="?", choices=("sub", "main"))
    p.add_argument("duration_s", nargs="?")
    p.add_argument("fps", nargs="?")


@verb(noun="vision", verb="live", tier=SENSOR, summary="live camera view for aiming (the camera node decodes, this node views)",
      role="viewer", wraps=("scripts/vision/live_view.sh",), args=_live_args, example=("sub", "60"), doc="docs/vision.md", tty=True)
def live(ctx, ns, rest):
    pos = [x for x in (ns.stream, ns.duration_s, ns.fps) if x]
    return sh(ctx, "scripts/vision/live_view.sh", *pos, *rest)


def _surface_args(p):
    p.add_argument("xy_step_mm", nargs="?")


@verb(noun="vision", verb="surface", tier=SENSOR, summary="live multi-view surface reconstruction streamed to Rerun",
      role="viewer", wraps=("scripts/vision/live_surface.sh", "scripts/vision/live_surface.py", "scripts/vision/reconstruct_surface.py"),
      passthrough="live_surface.py", args=_surface_args, example=(), doc="docs/vision.md", tty=True)
def surface(ctx, ns, rest):
    pos = [ns.xy_step_mm] if ns.xy_step_mm else []
    return sh(ctx, "scripts/vision/live_surface.sh", *pos, *rest)


def _dur_arg(p):
    p.add_argument("duration_s", nargs="?")


@verb(noun="vision", verb="track", tier=SENSOR, summary="vision-only 5-camera EE tracking shadow — never opens an arm connection",
      role="poe-cameras", wraps=("scripts/vision/ee_tracking_shadow.sh", "scripts/vision/ee_tracker.py"), args=_dur_arg,
      example=("30",), doc="docs/ee_fiducial_tracking.md")
def track(ctx, ns, rest):
    pos = [ns.duration_s] if ns.duration_s else []
    return sh(ctx, "scripts/vision/ee_tracking_shadow.sh", *pos, *rest)


# --- tags ------------------------------------------------------------------------


def _vision_py(ctx, rel, *args, **kw):
    """Vision solvers need numpy/opencv, which the stdlib CLI interpreter lacks;
    run them under ~/.venvs/tatbot-vision (the sweep pipeline's interpreter),
    falling back to system python3."""
    from pathlib import Path
    venv = Path(os.environ.get(
        "TATBOT_VISION_PYTHON", "~/.venvs/tatbot-vision/bin/python")).expanduser()
    interp = str(venv) if venv.exists() else "python3"
    return Plan(argv=[interp, ctx.path(rel), *args], **kw)


def _palette_tip_args(p):
    p.add_argument("session", help="teleop session dir (touches.json) with the tip planted on palette_tag8")


@verb(noun="vision", verb="palette tip", tier=MUTATES_CONFIG,
      summary="authoritative palette_root from the tip planted on tag8 (pivot solve, no camera) -> palette_calibration.yaml",
      wraps=("scripts/vision/palette_cal.py",), passthrough="palette_cal.py", args=_palette_tip_args, needs_tool=True,
      example=("~/tatbot-logs/vision/calib-sessions/last",), doc="docs/ink.md",
      invariants=("Reads a captured session and solves; nothing moves. Capture it with a tip phase planted on the palette tag.",
                  "Writes only the `tip` block of config/palette_calibration.yaml; the `vision` block is left alone."))
def vision_palette_tip(ctx, ns, rest):
    return _vision_py(ctx, "scripts/vision/palette_cal.py", "tip", ns.session, *tool_flag(ctx), *rest)


def _palette_vision_args(p):
    p.add_argument("--from-scan", required=True, help="a `tatbot vision tags scan` JSON with palette_tag8")


@verb(noun="vision", verb="palette vision", tier=MUTATES_CONFIG,
      summary="quick palette_root from tag8 through the camera bundles (~7 mm) -> palette_calibration.yaml",
      wraps=("scripts/vision/palette_cal.py",), passthrough="palette_cal.py", args=_palette_vision_args,
      example=("--from-scan", "~/tatbot-logs/vision/tag-scan/scan.json"), doc="docs/ink.md",
      invariants=("Consumes a scan and the calibration bundles; nothing moves. As good as robot-world-current.json (~7 mm).",
                  "Writes only the `vision` block of config/palette_calibration.yaml; the `tip` block is left alone."))
def vision_palette_vision(ctx, ns, rest):
    return _vision_py(ctx, "scripts/vision/palette_cal.py", "vision", "--from-scan", ns.from_scan, *rest,
                      notes=["needs opencv — run on a camera node: tatbot --on <camera-node> ..."])


@verb(noun="vision", verb="tags scan", tier=SENSOR, summary="scan the cameras for AprilTag 16h5",
      role="cameras-lan", wraps=("scripts/vision/tag_scan.py",), passthrough="tag_scan.py", example=("--", "--help"), doc="docs/fiducials.md")
def tags_scan(ctx, ns, rest):
    return py(ctx, "scripts/vision/tag_scan.py", *rest)


@verb(noun="vision", verb="tags print", tier=OFFLINE, summary="render the wrist-tag print sheet",
      wraps=("scripts/vision/generate_wrist_tags.py",), passthrough="generate_wrist_tags.py", example=("--", "--help"), doc="docs/fiducials.md")
def tags_print(ctx, ns, rest):
    return py(ctx, "scripts/vision/generate_wrist_tags.py", *rest)


@verb(noun="vision", verb="tags export", tier=MUTATES_CONFIG, summary="publish the canonical wrist layout + generated URDF links",
      wraps=("scripts/vision/export_wrist_tags.py",), passthrough="export_wrist_tags.py", example=("--", "--help"), doc="docs/fiducials.md",
      invariants=CALIB_INV)
def tags_export(ctx, ns, rest):
    return py(ctx, "scripts/vision/export_wrist_tags.py", *rest)


# --- cams ------------------------------------------------------------------------


def _cams_args(p):
    p.add_argument("what", choices=("snapshot", "diff", "restore", "get", "set"))
    p.add_argument("args", nargs="*")


@verb(noun="vision", verb="cams", tier=REMOTE, summary="Amcrest CGI config: snapshot / diff / restore / get / set",
      role="cameras-lan", wraps=("scripts/camera_config.sh",), args=_cams_args, example=("snapshot", "before-ab"), doc="docs/vision.md",
      invariants=("Credentials come from ~/.config/tatbot/cameras.env (never committed).",))
def cams(ctx, ns, rest):
    return sh(ctx, "scripts/camera_config.sh", ns.what, *ns.args, *rest)


@verb(noun="vision", verb="cams-ab", tier=REMOTE, summary="reversible one-variable encoder A/B/A on one camera",
      role="cameras-lan", wraps=("scripts/vision/camera_encoder_ab.sh", "scripts/vision/summarize_camera_encoder_ab.py"),
      passthrough="camera_encoder_ab.sh", example=("--", "--help"), doc="docs/vision.md")
def cams_ab(ctx, ns, rest):
    return sh(ctx, "scripts/vision/camera_encoder_ab.sh", *rest)


# --- monitor / deploy ------------------------------------------------------------


def _verify_args(p):
    p.add_argument("commit", help="the commit the monitor must be running")
    p.add_argument("--check-only", action="store_true")


@verb(noun="vision", verb="monitor verify", tier=REMOTE, summary="restart and prove freshness of the vision monitor",
      role="vision-monitor", wraps=("scripts/vision/verify_monitor_deploy.sh", "config/systemd/tatbot-vision-monitor.service"),
      args=_verify_args, example=("HEAD", "--check-only"), doc="docs/vision.md")
def monitor_verify(ctx, ns, rest):
    flag = ["--check-only"] if ns.check_only else []
    return sh(ctx, "scripts/vision/verify_monitor_deploy.sh", ns.commit, *flag, *rest)


def _deploy_args(p):
    p.add_argument("target", nargs="?",
                   choices=(nodes.example_node("poe-cameras"), nodes.example_node("arm"), "all"),
                   default="all")


@verb(noun="vision", verb="deploy", tier=REMOTE, summary="build + deploy the Rerun/teleop stack from one pushed commit",
      wraps=("scripts/vision/deploy_visualizer.sh",), args=_deploy_args, example=(nodes.example_node("poe-cameras"),), doc="docs/vision.md",
      invariants=("Deploys what is PUSHED, not the working tree.",))
def deploy(ctx, ns, rest):
    return sh(ctx, "scripts/vision/deploy_visualizer.sh", ns.target, *rest)


@verb(noun="vision", verb="d", tier=SENSOR, summary="passthrough to the Rust visiond binary (14 clap subcommands)",
      wraps=("rust/visiond",), passthrough="tatbot-visiond", example=("--", "--help"), doc="rust/README.md")
def visiond(ctx, ns, rest):
    return Plan(argv=[ctx.path("rust/target/release/tatbot-visiond"), *rest],
                notes=["built by `cargo build --release --features rerun` (scripts/check rust builds without the release profile)"])


# --- depth / audio / net ---------------------------------------------------------


def _probe_args(p):
    p.add_argument("label", choices=("hover", "touch"))
    p.add_argument("seconds", nargs="?", default="15")


@verb(noun="depth", verb="probe", tier=SENSOR, summary="phase-0 gate: can the wrist D405s see the paper (hover / touch)",
      role="realsense", wraps=("scripts/depth_probe.py",), passthrough="depth_probe.py", args=_probe_args, example=("hover", "15"),
      doc="docs/imitation_learning.md")
def depth_probe(ctx, ns, rest):
    return py(ctx, "scripts/depth_probe.py", ns.label, ns.seconds, *rest)


def _audio_args(p):
    p.add_argument("target", nargs="*", help="run dir or wav, or analysis JSONs with --compare")


@verb(noun="audio", verb="analyze", tier=OFFLINE, summary="score a rollout's contact mic: when did the pen actually touch",
      wraps=("scripts/il_analyze_audio.py", "scripts/il_audio_record.sh"), passthrough="il_analyze_audio.py", args=_audio_args,
      example=("~/tatbot-logs/rollout_async/last",), doc="docs/vision.md")
def audio_analyze(ctx, ns, rest):
    return py(ctx, "scripts/il_analyze_audio.py", *ns.target, *rest)


@verb(noun="net", verb="status", tier=SENSOR, summary="edge/home mode, LAN and tailnet reachability",
      wraps=("scripts/network/network_status.sh",), example=(), doc="docs/development.md")
def net_status(ctx, ns, rest):
    return sh(ctx, "scripts/network/network_status.sh", *rest)
