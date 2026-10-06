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


def uvpy(ctx: Ctx, project: str, rel: str, *args: str, extra: str | None = None, with_: tuple[str, ...] = (), **kw) -> Plan:
    """A repo script inside one of the uv projects' environments; `with_` adds
    pinned requirements the project itself does not carry."""
    return Plan(argv=["uv", "run", "--project", ctx.path(project), *(["--extra", extra] if extra else []),
                      *(item for spec in with_ for item in ("--with", spec)), "python", ctx.path(rel), *args], **kw)


def lerobot_py(ctx: Ctx, rel: str, *args: str, **kw) -> Plan:
    """Run a LeRobot tool in the environment owned by this node's role.

    The source's own plugin venv first. The serve and train roots are the
    environments of the nodes holding those roles (or an explicitly named
    TATBOT_SERVE_ROOT / TATBOT_TRAIN_ROOT); anywhere else a missing plugin
    venv means `uv run`, which syncs the pinned arm SDK, never a neighbour
    role's environment that happens to exist on this host: the arm node's
    leftover training venv carries no trossen_arm, and picking it made every
    retained-source `session capture-paper` fail on import (2026-09-16).
    """
    from tatbot_cli import nodes as fleet_nodes

    home = Path(os.environ.get("HOME", "~")).expanduser()
    roles = fleet_nodes.roles_of(fleet_nodes.load(ctx.repo), ctx.node)
    candidates = [Path(ctx.path(LEROBOT_PROJECT)) / ".venv/bin/python"]
    for role, variable, default in (("serve", "TATBOT_SERVE_ROOT", "il-serve"),
                                    ("train", "TATBOT_TRAIN_ROOT", "il-train")):
        explicit = os.environ.get(variable)
        if explicit or role in roles:
            candidates.append(Path(explicit or home / default) / ".venv/bin/python")
    for python in candidates:
        if os.access(python, os.X_OK):
            notes = list(kw.pop("notes", []))
            notes.append(f"interpreter: pinned LeRobot environment {python}")
            return Plan(argv=[str(python), ctx.path(rel), *args], notes=notes, **kw)
    return uvpy(ctx, LEROBOT_PROJECT, rel, *args, **kw)


def uvmod(ctx: Ctx, project: str, module: str, *args: str, extra: str | None = None, **kw) -> Plan:
    return Plan(argv=["uv", "run", "--project", ctx.path(project), *(["--extra", extra] if extra else []), "python", "-m", module, *args], **kw)


def tag_arg(p) -> None:
    p.add_argument("--tag", metavar="LABEL",
                   help="optional tag folded into this launch's id (the id is automatic: ledgered + audited by arm_gate)")


def ink_flags(p) -> None:
    """The ink hook every launcher that puts a tool on the skin takes (scripts/lib/ink_hook.sh)."""
    p.add_argument("--no-ink", action="store_true", help="no ink session, no debit; the run is stamped tracking=false")


def ink_argv(ns) -> list[str]:
    return ["--no-ink"] if ns.no_ink else []


def tool_flag(ctx: Ctx, flag: str = "--ee-tool") -> list[str]:
    """`--ee-tool X` when a tool was stated (every tool accepts it; `--tool-id` is the legacy alias)."""
    return [flag, ctx.ee_tool] if ctx.ee_tool else []
