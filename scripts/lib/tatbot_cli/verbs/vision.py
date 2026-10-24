"""vision — cameras, tracking, stencils, deploy."""

from __future__ import annotations

import os

import stencil_generate
import stencil_reference
import stencil_tile

from tatbot_cli import nodes
from tatbot_cli.registry import MUTATES_CONFIG, OFFLINE, REMOTE, SENSOR, Plan, RoleFrom, verb
from tatbot_cli.verbs._common import lerobot_py, py, sh


def _depth_compare_args(p):
    p.add_argument('--capture', required=True)
    p.add_argument('--settings', required=True, help='JSON list of filter candidates')
    p.add_argument('--ground-truth', help='independently measured metric depth NPZ')


@verb(effects=('read_files', 'write_files', 'start_process'), noun='vision', verb='depth compare', tier=OFFLINE, visibility='public',
      summary='compare filters on a retained raw scan burst',
      example=('--capture', 'capture-1.npz', '--settings', 'filters.json'),
      wraps=('scripts/vision/depth_compare.py',), args=_depth_compare_args, doc='docs/vision.md')
def depth_compare(ctx, ns, rest):
    args = ['--capture', ns.capture, '--settings', ns.settings]
    if ns.ground_truth:
        args += ['--ground-truth', ns.ground_truth]
    return Plan(argv=['uv', 'run', '--no-project', '--with', 'numpy', 'python',
                      ctx.path('scripts/vision/depth_compare.py'), *args])


# The throwaway uv environments' exact pins: the stencil renderer's, which the tests run (Pillow is
# stencil_tile.PILLOW_REQUIREMENT).
NUMPY, OPENCV = "numpy==2.5.3", "opencv-python-headless==5.0.0.93"
CALIB_INV = ("config/fiducials.json is the only hand-edited fiducial inventory.",
             "Golden copies live outside the repo: ~/tatbot-logs/vision/calibration-current.json, robot-world-current.json.")


@verb(effects=('read_files', 'write_files', 'start_process'), visibility="public",
      noun="vision", verb="stencil generate", tier=OFFLINE, output="json",
      summary="generate the skin stencil: a fiducial tile of AprilTag 16h5 markers with its 300 dpi print",
      wraps=("scripts/lib/stencil_generate.py", "scripts/lib/stencil_tile.py"), args=stencil_generate.add_arguments,
      example=("--preset", "silicone", "--seed", "0", "--output", "~/tatbot-logs/stencils/silicone-tile-0"),
      doc="docs/stencil-frames.md",
      invariants=("Writes the artwork (stencil.png, tile.json, settings.json, tracking.json) into --output and the "
                  "printable PNG into its print/ locally; first use may download the pinned numpy, Pillow and OpenCV "
                  "environment.",
                  "Every code on a sheet is distinct; the same layout and seed reproduce the print."))
def stencil_generate_frame(ctx, ns, rest):
    stencil_generate.validate(ns)
    return Plan(argv=["uv", "run", "--no-project", "--python", "3.12", "--with", NUMPY,
                      "--with", stencil_tile.PILLOW_REQUIREMENT, "--with", OPENCV,
                      "python", ctx.path("scripts/lib/stencil_generate.py"), *stencil_generate.forward(ns), *rest])


@verb(effects=('read_files', 'write_files', 'start_process'), noun="vision", verb="stencil reference", visibility="public",
      tier=OFFLINE, output="json", args=stencil_reference.add_arguments,
      summary="export a tracking manifest for a generated fiducial tile, and install it where the observer reads",
      wraps=("scripts/lib/stencil_reference.py",), doc="docs/stencil-frames.md",
      example=("--settings", "stencil/settings.json"))
def stencil_reference_export(ctx, ns, rest):
    return py(ctx, "scripts/lib/stencil_reference.py", "--settings", ns.settings,
              *(["--install"] if ns.install else []))


# --- tags ------------------------------------------------------------------------


def _vision_py(ctx, rel, *args, **kw):
    """Vision solvers need numpy/opencv, which the stdlib CLI interpreter lacks;
    run them under ~/.venvs/tatbot-vision, falling back to system python3."""
    from pathlib import Path
    venv = Path(os.environ.get(
        "TATBOT_VISION_PYTHON", "~/.venvs/tatbot-vision/bin/python")).expanduser()
    interp = str(venv) if venv.exists() else "python3"
    return Plan(argv=[interp, ctx.path(rel), *args], **kw)


@verb(effects=('read_files', 'write_files'), visibility="public", noun="vision", verb="tags print", tier=OFFLINE, summary="render the wrist-tag print sheet",
      wraps=("scripts/vision/generate_wrist_tags.py",), passthrough="generate_wrist_tags.py", example=("--", "--help"), doc="docs/fiducials.md")
def tags_print(ctx, ns, rest):
    # cv2 lives on no operator node (the vision venv has numpy only; cv2 is on the camera node), and the
    # sheet is a tracked file in THIS checkout — so a throwaway uv env, not a hop.
    return Plan(argv=["uv", "run", "--no-project", "--with", OPENCV, "--with", NUMPY,
                      "--with", stencil_tile.PILLOW_REQUIREMENT, "python", ctx.path("scripts/vision/generate_wrist_tags.py"),
                      *rest],
                notes=["renders with a throwaway uv env (opencv-python-headless, Pillow): first run downloads it"])


@verb(effects=('read_files', 'write_config'), noun="vision", verb="tags export", tier=MUTATES_CONFIG, summary="publish the canonical wrist layout + generated URDF links",
      wraps=("scripts/vision/export_wrist_tags.py",), passthrough="export_wrist_tags.py", example=("--", "--help"), doc="docs/fiducials.md",
      invariants=CALIB_INV)
def tags_export(ctx, ns, rest):
    return _vision_py(ctx, "scripts/vision/export_wrist_tags.py", *rest)


# --- deploy ----------------------------------------------------------------------


def _deploy_targets() -> tuple[str, ...]:
    """The nodes deploy_visualizer.sh knows: the arm node (teleop + realsense visiond) and the fleet
    viewer (blueprint sender + scripts), from config/nodes.json roles."""
    seen: dict[str, None] = {}
    for role in ("arm", "rerun-server"):
        seen.setdefault(nodes.example_node(role), None)
    return (*seen, "all")


def _deploy_args(p):
    p.add_argument("target", nargs="?", choices=_deploy_targets(), default="all")


@verb(effects=('read_files', 'write_files', 'write_config', 'network', 'remote_exec'), noun="vision", verb="deploy", tier=REMOTE, summary="build + deploy the Rerun/teleop stack from one pushed commit",
      wraps=("scripts/vision/deploy_visualizer.sh", "scripts/lib/tatbot_cli_install.py"), args=_deploy_args, example=(nodes.example_node("arm"),), doc="docs/vision.md",
      invariants=("Deploys what is PUSHED, not the working tree.",
                  "Every deployed node exposes the launcher as /usr/local/bin/tatbot."))
def deploy(ctx, ns, rest):
    return sh(ctx, "scripts/vision/deploy_visualizer.sh", ns.target, *rest)


@verb(effects=('read_files', 'write_files', 'sensor_read', 'network'), noun="vision", verb="d", tier=SENSOR, summary="explicit access to the vision service's own subcommands (`-- --help` lists them)",
      wraps=("rust/visiond",), passthrough="tatbot-visiond", example=("--", "--help"), doc="rust/README.md")
def visiond(ctx, ns, rest):
    return Plan(argv=[ctx.path("rust/target/release/tatbot-visiond"), *rest],
                notes=["built by `cargo build --release --features rerun` (scripts/check rust builds without the release profile)"])


def _target_track_args(p):
    p.add_argument('--target', required=True, help='rigid_target entry in the canonical fiducial inventory')
    p.add_argument('--layout', required=True, help='measured target layout; tag transforms are target-from-tag')
    p.add_argument('--calibration', required=True, help='camera calibration used by the running frame owner')
    p.add_argument('--connect', required=True, help='existing session bus endpoint')
    p.add_argument('--socket', required=True, help='existing camera-owner frame socket')
    p.add_argument('--output', required=True, help='estimate JSONL outside the repository')


@verb(effects=('read_files', 'write_files', 'sensor_read', 'network'), noun='vision', verb='track-target',
      tier=SENSOR, role='track', args=_target_track_args, wraps=('scripts/vision_track_target.sh',), doc='docs/fleet.md',
      summary='publish a measured material target from the existing camera-owner socket',
      example=('--target', 'paper', '--layout', 'paper-layout.json', '--calibration', 'calibration.json',
               '--connect', 'tcp/127.0.0.1:7447', '--socket', '/tmp/frames.sock', '--output', '/tmp/target.jsonl'))
def track_target(ctx, ns, rest):
    from pathlib import Path

    if ns.target == 'wrist':
        raise ValueError('track-target requires an independent material target')
    if Path(ns.output).resolve().is_relative_to(ctx.repo.resolve()):
        raise ValueError('target estimates must be written outside the source tree')
    return sh(ctx, 'scripts/vision_track_target.sh', '--target', ns.target, '--wrist-layout', ns.layout,
                      '--inventory', ctx.path('config/fiducials.json'), '--calibration', ns.calibration,
                      '--connect', ns.connect, '--socket', ns.socket, '--node', ctx.node,
                      '--output', ns.output, '--temporal-initializers')


def _surface_replay_args(p):
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--recording", help="one sensor's visiond frames.jsonl")
    source.add_argument("--synthetic", action="store_true", help="known image motion and loss cases")
    p.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--match-keyframes", action="store_true", help="experimental appearance bank (requires --match and --recover)")
    p.add_argument("--match", choices=("orb", "sift"), help="advisory keyframe search")
    p.add_argument("--recover", action="store_true", help="bounded reference-only reacquisition")
    p.add_argument("--rerun", action="store_true", help="save a capped replay RRD")


@verb(effects=('read_files', 'write_files'), noun="vision", verb="surface replay", tier=OFFLINE,
      summary="evaluate experimental sparse surface tracks on recorded pixels or synthetic motion",
      wraps=("scripts/vision/surface_replay.py",), args=_surface_replay_args,
      example=("--synthetic", "--max-frames", "90"), doc="docs/vision.md",
      invariants=("Default loss is latched; --recover uses bounded reference-only retries and never reseeds IDs.",))
def surface_replay(ctx, ns, rest):
    args = ["--synthetic"] if ns.synthetic else ["--recording", ns.recording]
    if ns.roi:
        args += ["--roi", *map(str, ns.roi)]
    args += ["--max-frames", str(ns.max_frames), "--width", str(ns.width), "--height", str(ns.height)]
    if ns.match_keyframes:
        args.append("--match-keyframes")
    if ns.match:
        args += ["--match", ns.match]
    if ns.recover:
        args.append("--recover")
    if ns.rerun:
        args.append("--rerun")
    return Plan(argv=["uv", "run", "--no-project", "--with", NUMPY, "--with", OPENCV,
                      *(["--with", "rerun-sdk==0.36.0"] if ns.rerun else []),
                      "python", ctx.path("scripts/vision/surface_replay.py"), *args])


def _surface_rgbd_args(p):
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="known 3D motion, dropout and ambiguity controls")
    source.add_argument("--recording", help="visiond evidence root with SENSOR_color and SENSOR_depth")
    p.add_argument("--sensor", help="recorded RealSense sensor prefix")
    p.add_argument("--method", choices=("both", "colored", "point_to_plane"), default="both")
    p.add_argument("--max-points", type=int, default=20000)
    p.add_argument("--stationary-depth", action="store_true", help="stationary ROI depth coverage and repeatability; no registration")
    p.add_argument("--image-check", action="store_true", help="advisory RGB-flow/depth consistency check")
    p.add_argument("--ground-truth", help="independently measured object-to-camera pose JSON")
    p.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    p.add_argument("--depth-range", type=float, nargs=2, metavar=("NEAR_M", "FAR_M"))
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--rerun", action="store_true", help="save a capped point-cloud comparison RRD")


@verb(effects=('read_files', 'write_files'), noun="vision", verb="surface rgbd", tier=OFFLINE,
      summary="benchmark offline RGB-D registration or stationary depth quality",
      wraps=("scripts/vision/surface_rgbd.py",), args=_surface_rgbd_args,
      example=("--synthetic", "--max-frames", "30"), doc="docs/vision.md",
      invariants=("True error requires ground truth; recorded alignment residuals are not physical accuracy.",
                  "Reuses visiond evidence decoding and the capped shared Rerun writer; numerical pools limited to one thread."))
def surface_rgbd(ctx, ns, rest):
    args = ["--synthetic"] if ns.synthetic else ["--recording", ns.recording]
    if ns.stationary_depth:
        args.append("--stationary-depth")
    if ns.image_check:
        args.append("--image-check")
    for option in ("sensor", "roi", "depth_range", "ground_truth", "method", "max_points"):
        value = getattr(ns, option)
        if value is not None:
            args += ["--" + option.replace("_", "-"), *map(str, value if isinstance(value, list) else [value])]
    args += ["--max-frames", str(ns.max_frames)]
    if ns.rerun:
        args.append("--rerun")
    open3d = [] if ns.stationary_depth else ["--with", "open3d==0.19.0"]
    realsense = [] if ns.stationary_depth else ["--with", "pyrealsense2==2.58.4.10922"]   # the nodes' librealsense
    return Plan(argv=["uv", "run", "--no-project", "--python", "3.12", *open3d,
                      "--with", OPENCV, "--with", "threadpoolctl==3.6.0", *realsense,
                      *(["--with", "rerun-sdk==0.36.0"] if ns.rerun else []),
                      "python", ctx.path("scripts/vision/surface_rgbd.py"), *args])


def _surface_attachment_benchmark_args(p):
    p.add_argument('--output', required=True, help='new capped comparison directory outside the repository')
    p.add_argument('--recording', action='append', default=[], help='retained RGB-D recording; repeat at most eight times')
    p.add_argument('--sensor', default='realsense1')
    p.add_argument('--roi', nargs=4, type=int)
    p.add_argument('--depth-range', nargs=2, type=float, default=(.03, 2.0))
    p.add_argument('--recording-max-frames', type=int, default=30)
    p.add_argument('--recording-stride', type=int, default=8)
    p.add_argument('--sustained-frames', type=int, default=300)
    p.add_argument('--sparse-only', action='store_true')
    p.add_argument('--skip-synthetic', action='store_true')


@verb(effects=('read_files', 'write_files'), noun='vision', verb='surface attachment-benchmark', tier=OFFLINE,
      summary='compare original-material attachment against sparse and dense RGB-D controls',
      args=_surface_attachment_benchmark_args, wraps=('scripts/vision/surface_attachment_benchmark.py',),
      doc='docs/vision.md', example=('--output', '/tmp/surface-comparison'),
      invariants=('Image observations and generated truth remain separate.',))
def surface_attachment_benchmark(ctx, ns, rest):
    args = ['--output', ns.output, '--sensor', ns.sensor, '--depth-range', *map(str, ns.depth_range),
            '--recording-max-frames', str(ns.recording_max_frames), '--recording-stride', str(ns.recording_stride),
            '--sustained-frames', str(ns.sustained_frames)]
    for recording in ns.recording:
        args += ['--recording', recording]
    if ns.roi:
        args += ['--roi', *map(str, ns.roi)]
    args += [f'--{flag.replace("_", "-")}' for flag in ('sparse_only', 'skip_synthetic') if getattr(ns, flag)]
    return Plan(argv=['uv', 'run', '--no-project', '--python', '3.12',
                      '--with', 'numpy==2.5.3', '--with', 'opencv-python-headless==5.0.0.93',
                      '--with', 'open3d==0.19.0', '--with', 'threadpoolctl==3.6.0',
                      '--with', 'pyrealsense2==2.58.4.10922',
                      'python', ctx.path('scripts/vision/surface_attachment_benchmark.py'), *args])


# --- the wrist D405s without an executor ----------------------------------------------------------


def _capture_args(p):
    p.add_argument("mode", choices=("once",), help="once: one capture")
    p.add_argument("dir", help="where capture-<k>.npz go")
    p.add_argument("--arm", choices=("right",),
                   help="the arm whose wrist cameras answer (required, never a default; the pink arm's alone carry any)")
    p.add_argument("--fake", action="store_true", help="synthetic tilted-plane depth; no cameras opened")
    p.add_argument("--from-owner", action="store_true", help="subscribe to the existing D405 camera owner")


def _capture_arm(ns) -> str:
    """The arm a capture invocation names: stated, never defaulted."""
    if not ns.arm:
        raise ValueError("vision capture once needs --arm right: the arm whose wrist cameras answer")
    return ns.arm


def _capture_validate(ctx, command, ns):
    from tatbot_cli.cli import UsageError
    try:
        _capture_arm(ns)
    except ValueError as error:
        raise UsageError(str(error)) from error
    return None


def _capture_owner_role(ns) -> str:
    """The fleet role owning the named arm's retained wrist cameras: the vision
    registry's `owner_role` for that arm's `wrist_cameras.CAPTURE_ROLES`
    (config/nodes.json maps the role to its node)."""
    from wrist_cameras import CAPTURE_ROLES, registry

    from tatbot_cli.registry import repo_root
    arm = _capture_arm(ns)
    cameras = [camera for camera in registry(repo_root()) if camera.get("arm") == arm]
    unretained = [camera["role"] for camera in cameras if camera["role"] not in CAPTURE_ROLES[arm]]
    owners = {camera["owner_role"] for camera in cameras}
    if not cameras or unretained or len(owners) != 1:
        raise ValueError(f"the {arm} arm's wrist cameras have no single capture owner in the vision registry "
                         f"(roles {[c['role'] for c in cameras] or 'none'}, retained {list(CAPTURE_ROLES[arm])}, "
                         f"owner roles {sorted(owners) or 'none'})")
    return owners.pop()


@verb(effects=('read_files', 'write_files', 'sensor_read'), noun="vision", verb="capture", tier=SENSOR,
      role_for=RoleFrom("the wrist camera owner of `--arm` (the vision registry's `owner_role`)", _capture_owner_role),
      validate=_capture_validate,
      summary="the wrist D405s without an executor: one capture of the named arm's wrist cameras",
      wraps=("scripts/vision_capture.py",), passthrough="vision_capture.py", args=_capture_args,
      example=("once", "/tmp/capture", "--arm", "right"),
      doc="docs/surface-formats.md",
      invariants=("Each capture carries t_wall.",
                  "Runs on the owner of the named arm's wrist cameras (vision.toml `owner_role`); --arm is stated."))
def vision_capture(ctx, ns, rest):
    flags = ["--arm", _capture_arm(ns)]
    if ns.fake:
        flags.append("--fake")
    if ns.from_owner:
        flags.append("--from-owner")
    return lerobot_py(ctx, "scripts/vision_capture.py", ns.mode, ns.dir, *flags, *rest)


def _handeye_args(p):
    p.add_argument("captures", nargs="*", help="capture dirs with capture-*.npz of the touched-off paper")
    p.add_argument("--out", help="where d405_handeye.json goes (default: the first capture dir)")
    p.add_argument("--self-test", action="store_true", help="recover a known synthetic perturbation")


@verb(effects=('read_files', 'write_files'), visibility="public", noun="vision", verb="handeye d405", tier=OFFLINE,
      summary="plane-route hand-eye for the wrist D405s: fit depth planes against the touched-off paper",
      wraps=("scripts/vision/d405_handeye_plane.py",), passthrough="d405_handeye_plane.py", args=_handeye_args,
      example=("--self-test",), doc="docs/surface-formats.md",
      invariants=("Files only; never edits the URDF — it prints the <origin xyz rpy> for a human to fold in.",
                  "In-plane translation is unobservable from a plane and is regularised to zero; the report says so.",))
def vision_handeye_d405(ctx, ns, rest):
    args = list(ns.captures)
    if ns.out:
        args += ["--out", ns.out]
    if ns.self_test:
        args.append("--self-test")
    return lerobot_py(ctx, "scripts/vision/d405_handeye_plane.py", *args, *rest)
