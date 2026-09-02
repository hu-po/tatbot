"""inkmap — the tattoo mapping / preview web app (web/inkmap, Vite + three.js).

`dev`, `build` and `check` are offline: they touch nothing but the web root
and, for `dev`, a local port. `deploy` is the one remote verb: it uploads the
built bundle to the Hugging Face Space (static SDK) named in the plan.
"""

from __future__ import annotations

from tatbot_cli import nodes
from tatbot_cli.registry import OFFLINE, REMOTE, SENSOR, Plan, verb

ROOT = "web/inkmap"
DOC = "docs/inkmap.md"
WRAPS = ("web/inkmap/package.json",)


def _npm(ctx, *args: str, notes=()):
    return Plan(argv=["npm", "run", *args], cwd=ctx.repo / ROOT, notes=list(notes))


INKGEN_PORT = 8600
INKGEN_SPACE = "hu-po/inkgen"
INKGEN_SPACE_URL = "https://hu-po-inkgen.hf.space"


def inkgen_node(ctx) -> str | None:
    """The node that carries the inkgen role in config/nodes.json."""
    cands = nodes.nodes_with(nodes.load(ctx.repo), "inkgen")
    return cands[0] if cands else None


def inkgen_url(ctx) -> str:
    """Where a fleet generator answers: this node if it has the role, else the role's node."""
    nmap = nodes.load(ctx.repo)
    if "inkgen" in nodes.roles_of(nmap, ctx.node):
        return f"http://127.0.0.1:{INKGEN_PORT}"
    node = inkgen_node(ctx)
    host = nodes.host_of(nmap, node) if node else None
    return f"http://{host}:{INKGEN_PORT}" if host else INKGEN_SPACE_URL


def _dev_args(p):
    p.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to reach it from another node)")
    p.add_argument("--port", type=int, default=4180)
    p.add_argument("--api", default=None, help="generator URL; default: the fleet generator (`tatbot inkgen start`), --space for the hosted one")
    p.add_argument("--space", action="store_true", help=f"use the hosted generator {INKGEN_SPACE_URL}")


@verb(noun="inkmap", verb="dev", tier=OFFLINE, summary="the tattoo preview app on this machine, pointed at the fleet generator",
      wraps=("scripts/inkmap_dev.sh", *WRAPS), passthrough="inkmap_dev.sh", args=_dev_args, example=(), doc=DOC,
      invariants=("Installs node_modules on first run.",
                  "The generator address comes from config/nodes.json (role inkgen); --api / --space override it; ?api= on the page overrides everything."))
def inkmap_dev(ctx, ns, rest):
    api = INKGEN_SPACE_URL if ns.space else (ns.api or inkgen_url(ctx))
    argv = [ctx.path("scripts/inkmap_dev.sh"), "--host", ns.host, "--port", str(ns.port), "--api", api]
    return Plan(argv=[*argv, *rest], notes=[f"serves http://{ns.host}:{ns.port}/", f"generator: {api}"])


@verb(noun="inkmap", verb="build", tier=OFFLINE, summary="production bundle into web/inkmap/dist (what the Space serves)",
      wraps=WRAPS, example=(), doc=DOC)
def inkmap_build(ctx, ns, rest):
    return _npm(ctx, "build", *rest)


@verb(noun="inkmap", verb="check", tier=OFFLINE, summary="typecheck + core tests (anchor/frame/decal/schema) + build",
      wraps=WRAPS, passthrough="scripts/check web", example=(), doc=DOC)
def inkmap_check(ctx, ns, rest):
    return Plan(argv=[ctx.path("scripts/check"), "web"], notes=["runs npm ci first if node_modules is missing"])


@verb(noun="inkmap", verb="rig", tier=OFFLINE, summary="regenerate the checked-in HBM rigs and named poses with Blender",
      wraps=("scripts/inkmap_rig.sh", "web/inkmap/tools/rig-hbm.py", "config/inkmap/body-rig.json"),
      passthrough="inkmap_rig.sh", example=(), doc=DOC,
      invariants=("CPU-only; does not connect to a robot or camera.",
                  "Fails unless canonical face order and the 0.1 mm cross-runtime LBS gate pass."))
def inkmap_rig(ctx, ns, rest):
    return Plan(argv=[ctx.path("scripts/inkmap_rig.sh"), *rest])


def _deploy_args(p):
    p.add_argument("--space", default="tatbot/inkmap", help="Hugging Face Space id (static SDK)")
    p.add_argument("--no-build", action="store_true", help="upload the existing web/inkmap/dist as is")


@verb(noun="inkmap", verb="deploy", tier=REMOTE, summary="build and upload web/inkmap/dist to the Hugging Face Space",
      wraps=("scripts/inkmap_deploy.sh",), passthrough="inkmap_deploy.sh", args=_deploy_args, example=(), doc=DOC,
      invariants=("Never changes the Space's visibility; private stays private until a human flips it on the Hub.",
                  "Needs HF_TOKEN (env or the git-ignored .env) with repo.write on the Space's namespace.",
                  "Mirrors dist/: files that are no longer built are deleted from the Space in the same commit."))
def inkmap_deploy(ctx, ns, rest):
    argv = [ctx.path("scripts/inkmap_deploy.sh"), "--space", ns.space]
    if ns.no_build:
        argv.append("--no-build")
    if ctx.dry_run:
        argv.append("--dry-run")
    return Plan(argv=[*argv, *rest], notes=[f"target: Hugging Face Space {ns.space} (https://huggingface.co/spaces/{ns.space})"])




# ---- inkgen: the design generator behind Inkmap (web/inkgen) -----------------
# Role "inkgen" in config/nodes.json says which node runs it; every verb that
# needs the node hops there by itself and fast-forwards its checkout first.
GEN_DOC = "docs/inkmap.md"
GEN_INV = ("Runs on the node with role inkgen — the CLI hops there and fast-forwards its checkout first.",
           "First run there creates ~/.cache/tatbot/inkgen/venv with uv and downloads the model (~31 GB); needs a 16 GB+ GPU.")


def _port_arg(p):
    p.add_argument("--port", type=int, default=INKGEN_PORT)


@verb(noun="inkgen", verb="start", tier=OFFLINE, summary="start the design generator in the background on the inkgen node and wait until it answers",
      role="inkgen", auto_hop=True, sync=True, wraps=("scripts/inkgen_ctl.sh", "scripts/inkgen_serve.sh", "web/inkgen/app.py"),
      args=_port_arg, example=(), doc=GEN_DOC, invariants=GEN_INV + ("Idempotent: a running generator is reported, not restarted.",))
def inkgen_start(ctx, ns, rest):
    return Plan(argv=[ctx.path("scripts/inkgen_ctl.sh"), "start", "--port", str(ns.port), *rest],
                notes=[f"then: tatbot inkmap dev (already points at http://{nodes.host_of(nodes.load(ctx.repo), ctx.node) or ctx.node}:{ns.port})"])


@verb(noun="inkgen", verb="stop", tier=OFFLINE, summary="stop the background generator on the inkgen node", role="inkgen", auto_hop=True,
      wraps=("scripts/inkgen_ctl.sh",), example=(), doc=GEN_DOC, invariants=("Only the pid in ~/.cache/tatbot/inkgen/inkgen.pid; never a broad pkill.",))
def inkgen_stop(ctx, ns, rest):
    return Plan(argv=[ctx.path("scripts/inkgen_ctl.sh"), "stop", *rest])


@verb(noun="inkgen", verb="logs", tier=OFFLINE, summary="tail the background generator's log on the inkgen node", role="inkgen", auto_hop=True,
      wraps=("scripts/inkgen_ctl.sh",), args=lambda p: p.add_argument("-n", type=int, default=40), example=(), doc=GEN_DOC)
def inkgen_logs(ctx, ns, rest):
    return Plan(argv=[ctx.path("scripts/inkgen_ctl.sh"), "logs", "-n", str(ns.n), *rest])


@verb(noun="inkgen", verb="serve", tier=OFFLINE, summary="run the generator in the foreground on the inkgen node (Ctrl-C stops it)",
      role="inkgen", auto_hop=True, sync=True, tty=True, wraps=("scripts/inkgen_serve.sh", "web/inkgen/app.py"), passthrough="inkgen_serve.sh",
      args=lambda p: (_port_arg(p), p.add_argument("--model", default=None, help="Hugging Face model id (default Tongyi-MAI/Z-Image-Turbo)")),
      example=(), doc=GEN_DOC, invariants=GEN_INV)
def inkgen_serve(ctx, ns, rest):
    argv = [ctx.path("scripts/inkgen_serve.sh"), "--port", str(ns.port), "--host", "0.0.0.0"]
    if ns.model:
        argv += ["--model", ns.model]
    return Plan(argv=[*argv, *rest], notes=["foreground; prefer `tatbot inkgen start` for a generator that outlives the terminal"])


def _status_args(p):
    p.add_argument("--url", default=None, help="generator base URL (default: the fleet generator; --space for the hosted one)")
    p.add_argument("--space", action="store_true", help=f"check the hosted generator {INKGEN_SPACE_URL}")


@verb(noun="inkgen", verb="status", tier=SENSOR, summary="is a generator answering? (fleet node by default, --space for the Hub)",
      args=_status_args, example=(), doc=GEN_DOC)
def inkgen_status(ctx, ns, rest):
    url = INKGEN_SPACE_URL if ns.space else (ns.url or inkgen_url(ctx))
    return Plan(argv=["curl", "-fsS", "--max-time", "30", f"{url.rstrip('/')}/api/health", *rest], notes=[f"read-only GET {url}/api/health"])


@verb(noun="inkgen", verb="deploy", tier=REMOTE, summary=f"upload web/inkgen to the ZeroGPU Space {INKGEN_SPACE} and wait for /api/health",
      wraps=("scripts/inkgen_deploy.sh",), passthrough="inkgen_deploy.sh", args=lambda p: p.add_argument("--space", default=INKGEN_SPACE),
      example=(), doc=GEN_DOC, invariants=("Needs HF_TOKEN (env or .env) with repo.write on the namespace; never changes hardware or visibility.",))
def inkgen_deploy(ctx, ns, rest):
    argv = [ctx.path("scripts/inkgen_deploy.sh"), "--space", ns.space]
    if ctx.dry_run:
        argv.append("--dry-run")
    return Plan(argv=[*argv, *rest], notes=[f"target: Hugging Face Space {ns.space}"])
