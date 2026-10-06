"""DrawingBotV3 acquisition and replay; no robot motion."""
from __future__ import annotations

from tatbot_cli.registry import OFFLINE, verb

from ._common import sh

SCRIPT = "scripts/drawingbot.sh"
WRAPS = (SCRIPT, "scripts/drawingbot.py", "scripts/lib/drawing_python.sh", "scripts/lib/drawing_python.py",
         "scripts/lib/drawingbot/artifacts.py", "scripts/lib/drawingbot/bridge.py",
         "scripts/lib/drawingbot/replay.py", "scripts/lib/drawingbot/recipe.py",
         "scripts/lib/drawingbot/container.py")


def _generate_args(p):
    p.add_argument("job", help="tatbot.dbv3-job/3: source identity and explicit pens/physical/PFM/export settings")
    p.add_argument("--out", required=True, help="new acquisition directory outside the repository")
    p.add_argument("--app", required=True, help="activated DrawingBotV3 Premium 1.6.22 installation")


@verb(noun="drawingbot", verb="generate", tier=OFFLINE, visibility="public",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="generate one physical DBV3 drawing with verified settings and a saved native recipe",
      args=_generate_args, wraps=(*WRAPS, "scripts/lib/drawingbot/job.py", "scripts/lib/drawingbot/pens.py"), doc="docs/drawingbot.md",
      example=("/tmp/job.json", "--out", "/tmp/acquisition", "--app", "/opt/drawingbotv3"),
      invariants=("Verifies source bytes and explicit dimensions, width, seed and effective settings.",
                  "Native acquisition only; no ROS, simulator, Node or robot access."))
def drawingbot_generate(ctx, ns, rest):
    return sh(ctx, SCRIPT, "generate", ns.job, "--out", ns.out, "--app", ns.app, *rest)


def _replay_args(p):
    p.add_argument("recipe", help="saved variant recipe directory")
    p.add_argument("--out", required=True, help="new replay output directory")
    p.add_argument("--app", required=True, help="activated Premium 1.6.22 installation")


def _export_args(p):
    _replay_args(p)
    p.add_argument("--migrate-runtime", action="store_true", help="create a new native baseline for a different Java runtime")


@verb(noun="drawingbot", verb="export", tier=OFFLINE, visibility="public",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="verify and replay native SVG acquisition only, without ROS compilation",
      args=_export_args, wraps=WRAPS, doc="docs/drawingbot.md",
      example=("/tmp/job/recipe", "--out", "/tmp/replay", "--app", "/opt/drawingbotv3"))
def drawingbot_export(ctx, ns, rest):
    migration = ["--migrate-runtime"] if ns.migrate_runtime else []
    return sh(ctx, SCRIPT, "export", ns.recipe, "--out", ns.out, "--app", ns.app, *migration, *rest)


def _container_args(p):
    p.add_argument("action", choices=("build", "smoke", "export"))
    p.add_argument("--engine", choices=("podman", "docker"), default="podman")
    p.add_argument("--image", default="localhost/tatbot-dbv3:1.6.22")
    p.add_argument("--out", required=True, help="new output directory for evidence")
    p.add_argument("--app", help="read-only Premium installation mount")
    p.add_argument("--recipe", help="saved recipe for native export")
    p.add_argument("--state", help="dedicated container home with normal activation; never copied host activation")
    p.add_argument("--migrate-runtime", action="store_true", help="create a new native baseline for the container runtime")


@verb(noun="drawingbot", verb="container", tier=OFFLINE, visibility="public",
      effects=("read_files", "write_files", "start_process", "network"),
      summary="build, smoke-test or run the private mounted-app acquisition container",
      args=_container_args, wraps=WRAPS, doc="docs/drawingbot.md",
      example=("build", "--out", "/tmp/dbv3-image"))
def drawingbot_container(ctx, ns, rest):
    args = ["--engine", ns.engine, "--image", ns.image, "--out", ns.out]
    for name in ("app", "recipe", "state"):
        if value := getattr(ns, name):
            args += [f"--{name}", value]
    if ns.migrate_runtime:
        args += ["--migrate-runtime"]
    return sh(ctx, SCRIPT, "container", ns.action, *args, *rest)
