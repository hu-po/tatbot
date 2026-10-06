"""work — one session, one worktree, one branch, pushed always (docs/work.md)."""

from __future__ import annotations

from tatbot_cli.registry import OFFLINE, verb
from tatbot_cli.verbs._common import py

BACKEND = "scripts/lib/tatbot_work.py"
DOC = "docs/work.md"


def _backend(ctx, *args):
    return py(ctx, BACKEND, *(["--json"] if ctx.json else []), *args)


def _start_args(p):
    p.add_argument("--task", metavar="NAME", help="what the worktree is for; names the session and the branch")
    p.add_argument("--session", metavar="ID", help="explicit session id (the Claude Code hook passes its own)")
    p.add_argument("--adopt", action="store_true",
                   help="move the mirror checkout's uncommitted changes into the new worktree, leaving it clean")


@verb(effects=("read_files", "write_files", "network", "start_process"), noun="work", verb="start", tier=OFFLINE,
      output="json", args=_start_args, wraps=(BACKEND,), example=("--task", "example"), doc=DOC,
      summary="a worktree of your own under ~/tatbot-work on an agent branch cut from origin/main",
      invariants=("The shared checkout is a mirror: its pre-commit hook refuses commits, so work cannot land there by accident.",
                  "The branch is pushed as soon as it exists; every later commit is pushed by the post-commit hook."))
def work_start(ctx, ns, rest):
    args = ["start"]
    if ns.task:
        args += ["--task", ns.task]
    if ns.session:
        args += ["--session", ns.session]
    if ns.adopt:
        args.append("--adopt")
    return _backend(ctx, *args, *rest)


@verb(effects=("read_files",), noun="work", verb="status", tier=OFFLINE, output="json", wraps=(BACKEND,), example=(), doc=DOC,
      summary="every worktree on this node: dirty, ahead of main, unpushed, parked; stale agent branches on origin")
def work_status(ctx, ns, rest):
    return _backend(ctx, "status", *rest)


def _land_args(p):
    p.add_argument("--done", action="store_true", help="remove the worktree after landing")
    p.add_argument("--no-check", action="store_true", help="skip the land gate (CI still runs everything on main)")
    p.add_argument("--full-check", action="store_true", help="run the whole fast tier, not only the land gate")


@verb(effects=("read_files", "write_files", "network", "start_process"), noun="work", verb="land", tier=OFFLINE,
      output="text", args=_land_args, wraps=(BACKEND,), example=("--no-check",), doc=DOC,
      summary="rebase onto origin/main, run the land gate, push to main (retrying while main moves), drop the agent branch",
      invariants=("No cherry-picks: what lands on main is the commit you made, so SHAs and `git status` stay truthful.",
                  "The branch is pushed before any gate runs; a failed check never strands work.",
                  "The gate is what the author answers for: lint, disclosure, the export gate, complexity of the changed "
                  "files. Someone else's red test cannot refuse your change; CI runs everything on main.",
                  "Refuses (exit 6) with uncommitted changes: commit them or `tatbot work park`.",
                  "After the push, the repository's post-land hook (internal/ci/post-land, if present) gets the "
                  "landed range and answers for what must follow it; its failure fails the land, never the push."))
def work_land(ctx, ns, rest):
    args = ["land"]
    if ns.done:
        args.append("--done")
    if ns.no_check:
        args.append("--no-check")
    if ns.full_check:
        args.append("--full-check")
    return _backend(ctx, *args, *rest)


def _park_args(p):
    p.add_argument("reason", help="why this is left unlanded; shown by `tatbot work status` and the sweeper")


@verb(effects=("read_files", "write_files", "network"), noun="work", verb="park", tier=OFFLINE, output="json",
      args=_park_args, wraps=(BACKEND,), example=("waiting on the arm node",), doc=DOC,
      summary="leave the work unlanded on purpose: a wip commit if needed, pushed, with the reason recorded",
      invariants=("A parked worktree satisfies the session stop hook; an unparked, unlanded one does not.",))
def work_park(ctx, ns, rest):
    return _backend(ctx, "park", ns.reason, *rest)


def _sweep_args(p):
    p.add_argument("--autosave-after", type=float, metavar="MIN", default=30.0,
                   help="autosave a dirty worktree idle this long as a wip commit (default 30)")
    p.add_argument("--prune-after", type=float, metavar="HOURS", default=24.0,
                   help="remove a clean, landed, unparked worktree idle this long (default 24)")
    p.add_argument("--rescue-after", type=float, metavar="HOURS", default=24.0,
                   help="move edits left in the mirror this long into a pushed mirror-rescue worktree (default 24)")


@verb(effects=("read_files", "write_files", "network", "start_process"), noun="work", verb="sweep", tier=OFFLINE,
      output="json", args=_sweep_args, wraps=(BACKEND,), example=(), doc=DOC,
      summary="what the hourly timer runs: autosave idle worktrees, push, fast-forward the mirror, prune merged branches",
      invariants=("Never deletes a commit: a wip autosave is a commit on a pushed branch, a pruned branch is contained in main.",))
def work_sweep(ctx, ns, rest):
    return _backend(ctx, "sweep", "--autosave-after", str(ns.autosave_after), "--prune-after", str(ns.prune_after),
                    "--rescue-after", str(ns.rescue_after), *rest)


def _init_args(p):
    p.add_argument("--no-timer", action="store_true", help="do not install the user systemd sweeper timer")


@verb(effects=("read_files", "write_files", "start_process"), noun="work", verb="init", tier=OFFLINE, output="json",
      args=_init_args, wraps=(BACKEND,), example=("--no-timer",), doc=DOC,
      summary="make this checkout the mirror: refuse commits here, arm the hooks, install the hourly sweeper timer")
def work_init(ctx, ns, rest):
    return _backend(ctx, "init", *(["--no-timer"] if ns.no_timer else []), *rest)


def _hook_args(p):
    p.add_argument("event", choices=("stop", "session-start", "post-commit", "pre-commit-guard"))


@verb(effects=("read_files", "write_files", "network"), noun="work", verb="hook", tier=OFFLINE, output="text",
      args=_hook_args, wraps=(BACKEND,), example=("post-commit",), doc=DOC,
      summary="the Claude Code SessionStart and Stop hooks and the git hooks (.claude/settings.json, scripts/githooks)")
def work_hook(ctx, ns, rest):
    return _backend(ctx, "hook", ns.event, *rest)
