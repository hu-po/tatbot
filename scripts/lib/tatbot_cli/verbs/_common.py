"""Plan builders shared by the verb modules. Each returns the argv a verb execs."""

from __future__ import annotations

import os
from pathlib import Path

from tatbot_cli.registry import Ctx, Plan

LEROBOT_PROJECT = "python/lerobot_robot_tatbot"
SIM_PROJECT = "python/tatbot_sim"


def sh(ctx: Ctx, rel: str, *args: str, **kw) -> Plan:
    """Exec a repo script by path — exactly what the docs tell a human to type."""
    return Plan(argv=[ctx.path(rel), *args], **kw)


def py(ctx: Ctx, rel: str, *args: str, **kw) -> Plan:
    """System python3 on a repo script (stdlib / system-numpy tools)."""
    return Plan(argv=["python3", ctx.path(rel), *args], **kw)


def uvpy(ctx: Ctx, project: str, rel: str, *args: str, **kw) -> Plan:
    """A repo script inside one of the uv projects' environments."""
    return Plan(argv=["uv", "run", "--project", ctx.path(project), "python", ctx.path(rel), *args], **kw)


def lerobot_py(ctx: Ctx, rel: str, *args: str, **kw) -> Plan:
    """Run a LeRobot tool in the environment owned by this node's role."""

    home = Path(os.environ.get("HOME", "~")).expanduser()
    candidates = (
        Path(ctx.path(LEROBOT_PROJECT)) / ".venv/bin/python",
        Path(os.environ.get("TATBOT_SERVE_ROOT", home / "il-serve")) / ".venv/bin/python",
        Path(os.environ.get("TATBOT_TRAIN_ROOT", home / "il-train")) / ".venv/bin/python",
    )
    for python in candidates:
        if os.access(python, os.X_OK):
            notes = list(kw.pop("notes", []))
            notes.append(f"interpreter: pinned LeRobot environment {python}")
            return Plan(argv=[str(python), ctx.path(rel), *args], notes=notes, **kw)
    return uvpy(ctx, LEROBOT_PROJECT, rel, *args, **kw)


def uvmod(ctx: Ctx, project: str, module: str, *args: str, **kw) -> Plan:
    return Plan(argv=["uv", "run", "--project", ctx.path(project), "python", "-m", module, *args], **kw)


def nonce_arg(p) -> None:
    p.add_argument("--nonce", metavar="LITERAL", help="unique literal you type now; single-use (arm_gate)")


def ink_flags(p) -> None:
    """The ink hook every launcher that puts a tool on the skin takes (scripts/lib/dip_hook.sh)."""
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dip", action="store_true",
                   help="scripted dip at the palette before the session (autonomous motion: needs --nonce)")
    g.add_argument("--no-ink", action="store_true", help="no dip, no session, no debit; the run is stamped tracking=false")


def ink_argv(ns) -> list[str]:
    return ["--dip"] if ns.dip else (["--no-ink"] if ns.no_ink else [])


def tool_flag(ctx: Ctx, flag: str = "--ee-tool") -> list[str]:
    """`--ee-tool X` when a tool was stated (every tool accepts it; `--tool-id` is the legacy alias)."""
    return [flag, ctx.ee_tool] if ctx.ee_tool else []
