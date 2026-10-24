"""inkmap — the tattoo mapping / preview web app (web/inkmap, Vite + three.js).

`dev` is offline and touches only the web root and a local port. `deploy` is
the one remote verb: it uploads the built
bundle to the Hugging Face Space (static SDK) named in the plan. The bundle
itself is built by `inkmap_deploy.sh`, and the typecheck + tests + build gate
is `tatbot check web` — the `build` and `check` verbs that duplicated those
were dropped on 2026-09-02.
"""

from __future__ import annotations

import sys

from tatbot_cli import EXIT_USAGE, nodes
from tatbot_cli.registry import OFFLINE, REMOTE, SENSOR, Plan, verb
from tatbot_cli.verbs._common import py

DOC = "docs/inkmap.md"
WRAPS = ("web/inkmap/package.json",)


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


@verb(effects=('read_files', 'write_files', 'network', 'start_process'), visibility="public", noun="inkmap", verb="dev", tier=OFFLINE, summary="the tattoo preview app on this machine",
      wraps=("scripts/inkmap_dev.sh", *WRAPS), passthrough="inkmap_dev.sh", args=_dev_args, example=(), doc=DOC,
      invariants=("Installs node_modules on first run.",))
def inkmap_dev(ctx, ns, rest):
    argv = [ctx.path("scripts/inkmap_dev.sh"), "--host", ns.host, "--port", str(ns.port)]
    return Plan(argv=[*argv, *rest], notes=[f"serves http://{ns.host}:{ns.port}/"])


def _resolve_args(p):
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="placement description or compatible full tattoo sentence")
    source.add_argument("--input", help="batch JSON path, or - for stdin")
    p.add_argument("--policy", choices=("interactive", "seeded-v1"), default="interactive")
    p.add_argument("--seed", type=int, help="required by seeded-v1; recorded in the resolution")


@verb(effects=('read_files',), visibility="public", output="json", noun="inkmap", verb="resolve", tier=OFFLINE,
      summary="resolve InkLang placement text to a canonical rest-surface anchor as JSON",
      wraps=("web/inkmap/tools/resolve.ts", "config/inkmap/inklang-resolution.schema.json", *WRAPS),
      args=_resolve_args,
      example=("--prompt", "left upper inner forearm"),
      doc="docs/inklang.md",
      invariants=("No network, generator, robot, pose, reach, or motion is involved.",
                  "Interactive policy returns choices instead of silently guessing a side or zone member.",
                  "seeded-v1 is deterministic and records its seed; output is canonical JSON."))
def inkmap_resolve(ctx, ns, rest):
    argv = ["node", "--experimental-strip-types", ctx.path("web/inkmap/tools/resolve.ts")]
    if ns.input:
        argv += ["--input", ns.input]
    else:
        argv += ["--prompt", ns.prompt]
        if ns.policy:
            argv += ["--policy", ns.policy]
        if ns.seed is not None:
            argv += ["--seed", str(ns.seed)]
    return Plan(argv=[*argv, *rest], cwd=ctx.repo)


def _release_prepare_args(p):
    p.add_argument("--output-dir", required=True, help="new packet directory outside the repository")
    p.add_argument("--space", default="tatbot/inkmap", help="intended static Space id")


@verb(effects=('read_files', 'write_files', 'network', 'start_process'), visibility="public",
      noun="inkmap", verb="release-prepare", tier=OFFLINE,
      summary="build, test, hash, and audit a gated Inkmap release/rollback packet",
      wraps=("scripts/inkmap_release.py", "scripts/inkmap_deploy.sh", *WRAPS),
      args=_release_prepare_args, example=("--output-dir", "/tmp/inkmap-release"), doc=DOC,
      invariants=("Runs the complete Inkmap web check before writing a packet.",
                  "Requires clean source and a new output outside the repository.",
                  "Records pending approvals and never deploys or changes a release gate."))
def inkmap_release_prepare(ctx, ns, rest):
    return py(
        ctx,
        "scripts/inkmap_release.py",
        "prepare",
        "--output-dir",
        ns.output_dir,
        "--space",
        ns.space,
        *rest,
    )


def _release_audit_args(p):
    p.add_argument("manifest", help="release-manifest.json to verify")
    p.add_argument("--require-current-source", action="store_true")


@verb(effects=('read_files',), visibility="public", noun="inkmap", verb="release-audit",
      tier=OFFLINE, output="json", summary="verify an Inkmap release packet and its hashes",
      wraps=("scripts/inkmap_release.py",), args=_release_audit_args,
      example=("/tmp/inkmap-release/release-manifest.json",), doc=DOC,
      invariants=("Read-only and offline; it never deploys or changes approvals.",
                  "Use --require-current-source to verify bound inputs against this checkout."))
def inkmap_release_audit(ctx, ns, rest):
    flags = ["--require-current-source"] if ns.require_current_source else []
    return py(ctx, "scripts/inkmap_release.py", "audit", ns.manifest, *flags, *rest)


def _deploy_args(p):
    p.add_argument("--space", default="tatbot/inkmap", help="Hugging Face Space id (static SDK)")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--no-build", action="store_true", help="upload the existing web/inkmap/dist as is")
    source.add_argument("--artifact-dir", help="upload a previously audited release packet")


@verb(effects=('read_files', 'write_files', 'network', 'remote_write'), noun="inkmap", verb="deploy", tier=REMOTE, summary="build and upload web/inkmap/dist to the Hugging Face Space",
      wraps=("scripts/inkmap_deploy.sh",), passthrough="inkmap_deploy.sh", args=_deploy_args, example=(), doc=DOC,
      invariants=("Never changes the Space's visibility; private stays private until a human flips it on the Hub.",
                  "Needs HF_TOKEN (env or the git-ignored .env) with repo.write on the Space's namespace.",
                  "Mirrors dist/: files that are no longer built are deleted from the Space in the same commit."))
def inkmap_deploy(ctx, ns, rest):
    argv = [ctx.path("scripts/inkmap_deploy.sh"), "--space", ns.space]
    if ns.no_build:
        argv.append("--no-build")
    if ns.artifact_dir:
        argv += ["--artifact-dir", ns.artifact_dir]
    if ctx.dry_run:
        argv.append("--dry-run")
    return Plan(argv=[*argv, *rest], notes=[f"target: Hugging Face Space {ns.space} (https://huggingface.co/spaces/{ns.space})"])




# ---- inkgen: the design generator (web/inkgen) -------------------------------
# Role "inkgen" in config/nodes.json says which node runs it; every verb that
# needs the node hops there by itself. `serve` fast-forwards that checkout
# first; `ctl` does not, so `stop` works on a diverged tree.
GEN_DOC = "docs/inkmap.md"
GEN_INV = ("Runs on the node with role inkgen — the CLI hops there; `serve` fast-forwards its checkout first.",
           "First run there creates ~/.cache/tatbot/inkgen/venv with uv and downloads the model (~31 GB); needs a 16 GB+ GPU.")


def _port_arg(p):
    p.add_argument("--port", type=int, default=INKGEN_PORT)


CTL_SUBS = ("start", "stop", "status", "logs")


def _ctl_effects(effects, ns, rest):
    if rest and rest[0] in ("status", "logs"):
        return effects - {"start_process", "stop_process", "gpu", "write_files"}
    return effects


@verb(refine_effects=_ctl_effects, effects=('read_files', 'write_files', 'network', 'start_process', 'stop_process', 'gpu'), noun="inkgen", verb="ctl", tier=OFFLINE,
      summary="-- start [--port N] [--idle-minutes N] | stop | status | logs [-n N]: the background generator on the inkgen node",
      role="inkgen", auto_hop=True, wraps=("scripts/inkgen_ctl.sh", "scripts/inkgen_serve.sh", "web/inkgen/app.py"),
      passthrough="inkgen_ctl.sh start|stop|status|logs", example=("--", "status"), doc=GEN_DOC,
      invariants=GEN_INV + ("start is idempotent: a running generator is reported, not restarted; it waits until /api/health answers.",
                            "stop signals only the pid in ~/.cache/tatbot/inkgen/inkgen.pid; never a broad pkill.",
                            "status here is the node's own pidfile + health; `tatbot inkgen status` is the health probe from any node.",
                            "the generator stops itself after --idle-minutes (default 15) with no generation; 0 keeps it up."))
def inkgen_ctl(ctx, ns, rest):
    if not rest or rest[0] not in CTL_SUBS:
        print(f"tatbot inkgen ctl: give one of {'|'.join(CTL_SUBS)} after -- "
              f"(got {rest[0] if rest else 'nothing'}); `tatbot inkgen serve` runs one in the foreground", file=sys.stderr)
        return EXIT_USAGE
    notes = []
    if rest[0] == "start":
        notes.append("stops itself after 15 idle minutes; `-- start --idle-minutes 0` keeps it up")
    return Plan(argv=[ctx.path("scripts/inkgen_ctl.sh"), *rest], notes=notes)


@verb(effects=('read_files', 'network', 'start_process', 'gpu'), noun="inkgen", verb="serve", tier=OFFLINE, summary="run the generator in the foreground on the inkgen node (Ctrl-C stops it)",
      role="inkgen", auto_hop=True, sync=True, tty=True, wraps=("scripts/inkgen_serve.sh", "web/inkgen/app.py"), passthrough="inkgen_serve.sh",
      args=lambda p: (_port_arg(p), p.add_argument("--model", default=None, help="Hugging Face model id (default Tongyi-MAI/Z-Image-Turbo)")),
      example=(), doc=GEN_DOC, invariants=GEN_INV)
def inkgen_serve(ctx, ns, rest):
    argv = [ctx.path("scripts/inkgen_serve.sh"), "--port", str(ns.port), "--host", "127.0.0.1"]
    if ns.model:
        argv += ["--model", ns.model]
    return Plan(argv=[*argv, *rest], notes=["foreground; prefer `tatbot inkgen ctl -- start` for a generator that outlives the terminal"])


def _status_args(p):
    p.add_argument("--url", default=None, help="generator base URL (default: the fleet generator; --space for the hosted one)")
    p.add_argument("--space", action="store_true", help=f"check the hosted generator {INKGEN_SPACE_URL}")


@verb(effects=('network',), output="json", noun="inkgen", verb="status", tier=SENSOR, summary="is a generator answering? (fleet node by default, --space for the Hub)",
      args=_status_args, example=(), doc=GEN_DOC)
def inkgen_status(ctx, ns, rest):
    url = INKGEN_SPACE_URL if ns.space else (ns.url or inkgen_url(ctx))
    return Plan(argv=["curl", "-fsS", "--max-time", "30", f"{url.rstrip('/')}/api/health", *rest], notes=[f"read-only GET {url}/api/health"])


def _inkgen_deploy_args(p):
    p.add_argument("--space", default=INKGEN_SPACE)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--prepare-dir", help="stage, stamp, smoke and hash a candidate outside the repository; upload nothing")
    source.add_argument("--artifact-dir", help="upload a previously prepared candidate")


def _inkgen_deploy_effects(effects, ns, rest):
    # Preparing a candidate copies files and imports them; it contacts nothing.
    return effects - {"network", "remote_write"} if ns.prepare_dir else effects


@verb(refine_effects=_inkgen_deploy_effects,
      effects=('read_files', 'write_files', 'network', 'remote_write'), noun="inkgen", verb="deploy", tier=REMOTE,
      summary=f"prepare, upload and verify the ZeroGPU Space {INKGEN_SPACE}",
      wraps=("scripts/inkgen_deploy.sh", "web/inkgen/app.py"), passthrough="inkgen_deploy.sh",
      args=_inkgen_deploy_args,
      example=(), doc=GEN_DOC,
      invariants=("Needs HF_TOKEN (env or .env) with repo.write on the namespace; never changes hardware or visibility.",
                  "--prepare-dir uploads nothing: it stages exactly the Space payload, stamps its build, imports it with this repository off sys.path, and hashes it into a release-candidate manifest.",
                  "A failed upload fails the deploy; its exit status is not piped through anything.",
                  "Verification waits for /api/health to report the build stamp that was just uploaded — a healthy previous revision is a failure, not a pass."))
def inkgen_deploy(ctx, ns, rest):
    argv = [ctx.path("scripts/inkgen_deploy.sh"), "--space", ns.space]
    for flag, value in (("--prepare-dir", ns.prepare_dir), ("--artifact-dir", ns.artifact_dir)):
        if value:
            argv += [flag, value]
    if ctx.dry_run:
        argv.append("--dry-run")
    return Plan(argv=[*argv, *rest],
                notes=[f"target: Hugging Face Space {ns.space}"] if not ns.prepare_dir
                else ["prepares a candidate for review; nothing is uploaded"])
