#!/usr/bin/env python3
"""DrawingBotV3 native acquisition and replay."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python/tatbot_contracts/src"))

from drawingbot.replay import replay


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="one physical DBV3 job; no compiler or robot")
    generate.add_argument("job", type=Path)
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--app", type=Path, required=True)
    job = commands.add_parser("export", help="replay a native acquisition without robot preparation")
    job.add_argument("recipe", type=Path)
    job.add_argument("--out", type=Path, required=True)
    job.add_argument("--app", type=Path, required=True)
    job.add_argument("--migrate-runtime", action="store_true")
    runtime = commands.add_parser("runtime-probe")
    runtime.add_argument("--app", type=Path, required=True)
    runtime.add_argument("--out", type=Path, required=True)
    container = commands.add_parser("container")
    container.add_argument("action", choices=("build", "smoke", "export"))
    container.add_argument("--engine", choices=("podman", "docker"), default="podman")
    container.add_argument("--image", default="localhost/tatbot-dbv3:1.6.22")
    container.add_argument("--out", type=Path, required=True)
    container.add_argument("--app", type=Path)
    container.add_argument("--recipe", type=Path)
    container.add_argument("--state", type=Path)
    container.add_argument("--migrate-runtime", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "generate":
        from drawingbot.job import generate
        print(generate(args.job.resolve(), args.out.resolve(), args.app.expanduser().resolve()))
    elif args.command == "export":
        replay(args.recipe.resolve(), args.out.resolve(), args.app.expanduser().resolve(),
               migrate_runtime=args.migrate_runtime)
    elif args.command == "runtime-probe":
        from drawingbot.container import runtime_probe
        runtime_probe(args.app.resolve(), args.out.resolve())
    elif args.command == "container":
        from drawingbot.container import build as container_build
        from drawingbot.container import run as container_run
        if args.action == "build":
            container_build(args.engine, args.image, args.out.resolve())
        else:
            if args.app is None or (args.action == "export" and args.recipe is None):
                parser.error("container smoke/export requires --app; export also requires --recipe and --state")
            container_run(args.engine, args.image, args.app.resolve(), args.out.resolve(),
                          recipe=args.recipe.resolve() if args.action == "export" else None,
                          state=args.state.resolve() if args.state else None, migrate_runtime=args.migrate_runtime)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"drawingbot: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
