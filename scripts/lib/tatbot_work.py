#!/usr/bin/env python3
"""tatbot work — one session, one worktree, one branch, pushed always.

Why this exists (2026-09-16). Two or three agents shared one checkout and were
told to "pull --rebase and push often". A shared tree is never clean, so the
rebase always refused; agents improvised with cherry-picks into throwaway
worktrees, which put every landed commit on origin under a new SHA while the
original stayed on local main. After two weeks local main was ahead 25 and
behind 89, eleven "dirty" files were stale copies of landed commits that nobody
dared touch, and a real fix sat unpushed for a day because a commit-time gate
had refused it. Work became durable only when someone remembered to land it.

Here it is durable the moment it exists:

- `start`  gives a session its own worktree under ~/tatbot-work on a branch
           agent/<node>/<session> cut from origin/main. The main checkout is a
           mirror: a timer fast-forwards it and its pre-commit hook refuses commits.
- every commit on an agent branch is pushed by the post-commit hook, no gates.
- `land`   rebases onto origin/main, runs the land gate, pushes to main with a
           retry loop, and deletes the agent branch. No cherry-picks: SHAs match.
- `park`   pushes whatever exists (a wip commit if needed) with a stated reason.
- `sweep`  (hourly timer) autosaves idle dirty worktrees as wip commits, pushes
           anything unpushed, fast-forwards the mirror, prunes merged branches
           and reports what is stale.
- `hook`   the Claude Code SessionStart and Stop hooks, and the git hooks: a
           session cannot end with unlanded, unparked work.

Everything is stdlib and runs from a bare clone. Exit codes follow the CLI:
0 ok, 1 failed, 2 usage, 6 busy/refused.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tatbot_cli import nodes as tatbot_nodes  # noqa: E402

SCHEMA = "tatbot.work/1"
BRANCH_PREFIX = "agent"
AUTOSAVE_AFTER_MIN = 30
STALE_AFTER_H = 24
IDLE_PRUNE_AFTER_H = 24
MIRROR_RESCUE_AFTER_H = 24
STOP_BLOCK_LIMIT = 3
LAND_RETRIES = 5
EXIT_FAILED, EXIT_USAGE, EXIT_BUSY = 1, 2, 6
TIMER_UNITS = ("tatbot-work-sweep.service", "tatbot-work-sweep.timer")


# --- git plumbing -----------------------------------------------------------

def git(cwd: Path | str, *args: str, check: bool = True, timeout: int = 120, env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=timeout, env=e)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r


def out(cwd, *args, **kw) -> str:
    return git(cwd, *args, **kw).stdout.strip()


def script_repo() -> Path:
    """The checkout this script runs from (mirror or worktree)."""
    return Path(__file__).resolve().parents[2]


def mirror_root() -> Path:
    """The mirror this tool serves: the checkout the script runs from, or TATBOT_WORK_MIRROR (tests)."""
    override = os.environ.get("TATBOT_WORK_MIRROR")
    return Path(override).expanduser().resolve() if override else mirror_of(script_repo())


def mirror_of(path: Path) -> Path:
    """The main checkout that owns `path`'s repository (itself for a plain checkout)."""
    common = Path(out(path, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    return common.parent if common.name == ".git" else common


def is_main_worktree(path: Path) -> bool:
    git_dir = out(path, "rev-parse", "--path-format=absolute", "--git-dir")
    common = out(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(git_dir).resolve() == Path(common).resolve()


def toplevel(path: Path) -> Path | None:
    r = git(path, "rev-parse", "--show-toplevel", check=False)
    return Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None


def work_root() -> Path:
    return Path(os.environ.get("TATBOT_WORK_ROOT") or "~/tatbot-work").expanduser()


def this_node() -> str:
    try:
        # With the map, a `hostname` alias resolves (a node's hostname need not be its name).
        return tatbot_nodes.this_node(tatbot_nodes.load(Path(__file__).resolve().parents[2]))
    except Exception:  # noqa: BLE001 - a bare clone has no node map; the hostname still names the branch
        return os.uname().nodename.split(".")[0]


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip()).strip("-").lower()
    return s[:48] or "work"


def branch_for(node: str, session: str) -> str:
    return f"{BRANCH_PREFIX}/{node}/{session}"


def net_timeout(default: int = 60) -> int:
    """Network calls are bounded harder inside the SessionStart hook (30 s budget)."""
    return int(os.environ.get("TATBOT_WORK_NET_TIMEOUT") or default)


def fetch(repo: Path, timeout: int | None = None) -> bool:
    r = git(repo, "fetch", "--quiet", "--prune", "origin", check=False, timeout=timeout or net_timeout())
    return r.returncode == 0


def ref_exists(repo: Path, ref: str) -> bool:
    return git(repo, "rev-parse", "--verify", "--quiet", ref, check=False).returncode == 0


def count(repo: Path, spec: str) -> int:
    r = git(repo, "rev-list", "--count", spec, check=False)
    return int(r.stdout.strip() or 0) if r.returncode == 0 else 0


def porcelain(repo: Path) -> list[str]:
    return [line for line in git(repo, "status", "--porcelain", "--untracked-files=normal").stdout.splitlines() if line]


def current_branch(repo: Path) -> str | None:
    b = out(repo, "rev-parse", "--abbrev-ref", "HEAD", check=False)
    return None if b in ("", "HEAD") else b


# --- session metadata ---------------------------------------------------------

def meta_path(session: str) -> Path:
    return work_root() / f"{session}.json"


def read_meta(session: str) -> dict:
    try:
        return json.loads(meta_path(session).read_text())
    except (OSError, ValueError):
        return {}


def write_meta(session: str, meta: dict) -> None:
    work_root().mkdir(parents=True, exist_ok=True)
    tmp = meta_path(session).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    tmp.replace(meta_path(session))


def session_of(path: Path) -> str | None:
    """The session a worktree path belongs to, or None when it is not ours."""
    try:
        rel = path.resolve().relative_to(work_root().resolve())
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


# --- worktree inventory ---------------------------------------------------------

def worktrees(mirror: Path) -> list[dict]:
    rows, cur = [], {}
    for line in git(mirror, "worktree", "list", "--porcelain").stdout.splitlines():
        if not line:
            if cur:
                rows.append(cur)
            cur = {}
        elif line.startswith("worktree "):
            cur["path"] = Path(line[9:])
        elif line.startswith("branch "):
            cur["branch"] = line[7:].replace("refs/heads/", "")
        elif line == "detached":
            cur["branch"] = None
    if cur:
        rows.append(cur)
    return rows


def newest_mtime(repo: Path, entries: list[str]) -> float | None:
    times = []
    for entry in entries:
        rel = entry[3:].split(" -> ")[-1].strip('"')
        try:
            times.append((repo / rel).stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else None


def inspect(mirror: Path, wt: dict) -> dict:
    path, branch = wt["path"], wt.get("branch")
    session = session_of(path)
    meta = read_meta(session) if session else {}
    dirty = porcelain(path)
    remote = f"origin/{branch}" if branch else None
    ahead_main = count(path, "origin/main..HEAD") if ref_exists(path, "origin/main") else count(path, "HEAD")
    unpushed = count(path, f"{remote}..HEAD") if remote and ref_exists(path, remote) else ahead_main
    last_commit = int(out(path, "log", "-1", "--format=%ct", check=False) or 0)
    changed = newest_mtime(path, dirty) or last_commit
    now = time.time()
    unlanded = bool(dirty) or ahead_main > 0
    return {
        "session": session, "path": str(path), "branch": branch, "task": meta.get("task"),
        "dirty_files": len(dirty), "ahead_of_main": ahead_main, "unpushed": unpushed,
        "parked": meta.get("parked"), "unlanded": unlanded,
        "idle_minutes": round((now - changed) / 60, 1) if changed else None,
        "age_hours": round((now - dt.datetime.fromisoformat(meta["created"].replace("Z", "+00:00")).timestamp()) / 3600, 1)
        if meta.get("created") else None,
    }


def remote_agent_branches(mirror: Path) -> list[dict]:
    rows = []
    fmt = "%(refname:short)|%(committerdate:unix)|%(objectname:short)"
    for line in git(mirror, "for-each-ref", f"--format={fmt}", f"refs/remotes/origin/{BRANCH_PREFIX}/", check=False).stdout.splitlines():
        ref, when, sha = line.split("|")
        merged = git(mirror, "merge-base", "--is-ancestor", ref, "origin/main", check=False).returncode == 0
        rows.append({"branch": ref.replace("origin/", "", 1), "sha": sha, "merged": merged,
                     "age_hours": round((time.time() - int(when)) / 3600, 1)})
    return rows


def summary(mirror: Path) -> dict:
    mirror_dirty = porcelain(mirror)
    mirror_ahead = count(mirror, "origin/main..HEAD") if ref_exists(mirror, "origin/main") else 0
    trees = [inspect(mirror, wt) for wt in worktrees(mirror) if session_of(wt["path"])]
    local_branches = {t["branch"] for t in trees}
    remote = [r for r in remote_agent_branches(mirror) if r["branch"] not in local_branches or not r["merged"]]
    unlanded = [t for t in trees if t["unlanded"] and not t["parked"]]
    parked = [t for t in trees if t["parked"]]
    stale = [r for r in remote if not r["merged"] and r["age_hours"] > STALE_AFTER_H and r["branch"] not in local_branches]
    return {
        "schema": SCHEMA, "node": this_node(), "checked_at": utc(), "work_root": str(work_root()),
        "mirror": {"path": str(mirror), "dirty_files": len(mirror_dirty), "ahead_of_main": mirror_ahead,
                   "clean": not mirror_dirty and mirror_ahead == 0},
        "worktrees": trees, "remote_branches": remote,
        "unlanded": len(unlanded), "parked": len(parked), "stale_remote": len(stale),
        "oldest_unlanded_hours": max((t["age_hours"] or 0 for t in unlanded), default=0),
    }


def render(s: dict) -> str:
    lines = [f"tatbot work — {s['node']} {s['checked_at']}",
             f"  mirror {s['mirror']['path']}: " + ("clean" if s["mirror"]["clean"] else
                                                  f"{s['mirror']['dirty_files']} dirty file(s), {s['mirror']['ahead_of_main']} commit(s) ahead of origin/main")]
    if not s["worktrees"]:
        lines.append(f"  no worktrees under {s['work_root']}")
    for t in s["worktrees"]:
        state = "parked: " + t["parked"]["reason"] if t["parked"] else ("UNLANDED" if t["unlanded"] else "landed")
        lines.append(f"  {t['session']:<40} {state}")
        lines.append(f"    {t['branch']}  dirty={t['dirty_files']} ahead_of_main={t['ahead_of_main']} unpushed={t['unpushed']}"
                     f" idle={t['idle_minutes']}m task={t['task'] or '-'}")
    for r in s["remote_branches"]:
        if not r["merged"]:
            lines.append(f"  remote {r['branch']} ({r['sha']}) unmerged, {r['age_hours']} h old, no worktree here")
    lines.append(f"  unlanded={s['unlanded']} parked={s['parked']} stale_remote={s['stale_remote']}")
    return "\n".join(lines)


# --- commands -------------------------------------------------------------------

def emit(a, payload: dict, text: str) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) if a.json else text)


def cmd_status(a) -> int:
    s = summary(mirror_root())
    emit(a, s, render(s))
    return 0


def adopt_into(mirror: Path, path: Path) -> list[str]:
    """Move the mirror's uncommitted changes into the new worktree, leaving the mirror clean."""
    entries = porcelain(mirror)
    if not entries:
        return []
    patch = git(mirror, "diff", "HEAD", "--binary").stdout
    if patch:
        applied = subprocess.run(["git", "-C", str(path), "apply", "--3way"], input=patch, text=True, capture_output=True)
        if applied.returncode != 0:
            # Nothing half-applied: the mirror keeps its edits, the worktree
            # returns to its revision, and the caller hears why.
            git(path, "reset", "--hard", "--quiet", check=False)
            detail = applied.stderr.strip().splitlines()
            raise RuntimeError("the mirror's edits do not apply onto "
                               f"{out(path, 'rev-parse', '--short', 'HEAD')}: {detail[-1] if detail else 'git apply refused'}")
    moved = []
    for entry in entries:
        rel = entry[3:].split(" -> ")[-1].strip('"')
        if entry.startswith("??"):
            dest = path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(mirror / rel), str(dest))
        moved.append(rel)
    tracked = [m for m, e in zip(moved, entries, strict=True) if not e.startswith("??")]
    if tracked:
        git(mirror, "checkout", "--", *tracked)
    return moved


def cmd_start(a) -> int:
    mirror = mirror_root()
    node = this_node()
    session = a.session or f"{slug(a.task or 'work')}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    path = work_root() / session
    branch = branch_for(node, session)
    notes = []
    if path.exists():
        meta = read_meta(session)
        emit(a, {"schema": SCHEMA, "session": session, "path": str(path), "branch": meta.get("branch", branch),
                 "created": False, "notes": ["worktree already exists"]},
             f"worktree exists: {path} ({meta.get('branch', branch)})")
        return 0
    if not fetch(mirror):
        notes.append("origin unreachable: branched from the local origin/main")
    base = "origin/main" if ref_exists(mirror, "origin/main") else "HEAD"
    if a.adopt and porcelain(mirror):
        # The mirror's edits were written against the mirror's own revision:
        # a worktree cut there takes them as the patch they are, and `land`
        # rebases them onto main with the author present to resolve. Cut at a
        # newer main they become a merge, and a merge with conflicts left the
        # mirror dirty and a worktree half-made once an hour (2026-09-17/18).
        base = out(mirror, "rev-parse", "HEAD")
        notes.append(f"cut at the mirror's own revision {base[:8]}; `tatbot work land` rebases onto main")
    work_root().mkdir(parents=True, exist_ok=True)
    if ref_exists(mirror, f"refs/heads/{branch}"):
        git(mirror, "worktree", "add", str(path), branch)
        notes.append("resumed the existing local branch")
    elif ref_exists(mirror, f"refs/remotes/origin/{branch}"):
        git(mirror, "worktree", "add", "--track", "-b", branch, str(path), f"origin/{branch}")
        notes.append("resumed the branch from origin")
    else:
        git(mirror, "worktree", "add", "-b", branch, str(path), base)
    moved = adopt_into(mirror, path) if a.adopt else []
    if moved:
        notes.append(f"adopted {len(moved)} uncommitted file(s) from the mirror")
    write_meta(session, {"schema": SCHEMA, "session": session, "branch": branch, "task": a.task, "node": node,
                         "path": str(path), "created": utc(), "base": out(path, "rev-parse", "--short", "HEAD"),
                         "parked": None, "stop_blocks": {}})
    if git(path, "push", "--quiet", "-u", "origin", branch, check=False, timeout=net_timeout()).returncode != 0:
        notes.append("branch not pushed yet (origin unreachable); the sweeper and the next commit will")
    payload = {"schema": SCHEMA, "session": session, "path": str(path), "branch": branch, "base": base,
               "created": True, "adopted": moved, "notes": notes}
    emit(a, payload, f"worktree {path}\nbranch   {branch} (from {base})\n" + "".join(f"note     {n}\n" for n in notes)
         + "next     work there; `tatbot work land` when done, `tatbot work park \"why\"` to leave it")
    return 0


def current_worktree(mirror: Path) -> tuple[Path, str, str] | None:
    top = toplevel(Path.cwd())
    if not top:
        return None
    session = session_of(top)
    if not session or mirror_of(top) != mirror:
        return None
    return top, session, current_branch(top) or ""


def require_worktree(a, mirror: Path, verb: str):
    wt = current_worktree(mirror)
    if not wt or not wt[2].startswith(f"{BRANCH_PREFIX}/"):
        print(f"work {verb}: run this inside an agent worktree (tatbot work start; they live under {work_root()})",
              file=sys.stderr)
        return None
    return wt


def wip_commit(path: Path, message: str) -> str | None:
    if not porcelain(path):
        return None
    git(path, "add", "-A")
    git(path, "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "--no-verify", "-m", message)
    return out(path, "rev-parse", "--short", "HEAD")


def push_branch(path: Path, branch: str) -> bool:
    # An agent branch belongs to one session, and land rebases it, so the push
    # is force-with-lease: it replaces only what this checkout last saw there.
    return git(path, "push", "--quiet", "--force-with-lease", "-u", "origin", branch, check=False, timeout=120).returncode == 0


def cmd_park(a) -> int:
    mirror = mirror_root()
    wt = require_worktree(a, mirror, "park")
    if not wt:
        return EXIT_USAGE
    path, session, branch = wt
    wip = wip_commit(path, f"wip: parked — {a.reason}")
    pushed = push_branch(path, branch)
    meta = read_meta(session)
    meta["parked"] = {"reason": a.reason, "at": utc(), "by": this_node(), "pushed": pushed}
    write_meta(session, meta)
    emit(a, {"schema": SCHEMA, "session": session, "branch": branch, "wip_commit": wip, "pushed": pushed, "reason": a.reason},
         f"parked {branch}" + (f" (wip {wip})" if wip else "") + ("" if pushed else " — NOT pushed: origin unreachable") + f": {a.reason}")
    return 0 if pushed else EXIT_FAILED


LAND_JOBS = ("lint-py", "lint-sh", "disclosure", "export")


def run_checks(path: Path, full: bool = False) -> int:
    """The land gate: what the author is answerable for, in under a minute.

    lint, the disclosure manifest, the public export gate, and the complexity
    ratchet scoped to the files this branch changed. Not the whole fast tier:
    that can be red from someone else's breakage, and a gate that refuses your
    change for their finding strands your work (2026-09-16: main's tests job
    was red on four upstream tests the day this flow landed). --full-check
    runs the whole tier; CI runs everything on main regardless.
    """
    check = path / "scripts" / "check"
    if not check.is_file():
        return 0
    if full:
        print("work land: scripts/check (fast tier)", flush=True)
        return subprocess.run([str(check), "--no-hooks"], cwd=path).returncode
    print(f"work land: scripts/check {' '.join(LAND_JOBS)} + complexity of changed files", flush=True)
    rc = subprocess.run([str(check), "--no-hooks", *LAND_JOBS], cwd=path).returncode
    changed = [f for f in out(path, "diff", "--name-only", "origin/main...HEAD", check=False).splitlines()
               if f.endswith(".py") and (path / f).is_file()]
    budget = path / "scripts" / "complexity_budget.py"
    if changed and budget.is_file():
        rc = subprocess.run(["python3", str(budget), *changed], cwd=path).returncode or rc
    return rc


RETRY = "retry"

# --- the repository's post-land hook --------------------------------------------

POST_LAND = Path("internal/ci/post-land")


def post_land(path: Path, base: str, run) -> int:
    """Run the repository's post-land hook, if it has one, with the landed range.

    A private tree may keep infrastructure that has to react to what just
    landed -- this one's isolated CI worker refuses main once an input its
    images are built from changes, until a person rebuilds them, and landing
    is already that person's answerable act. The hook decides; the land only
    hands it the range and reports what it did. A failing hook fails the
    land's exit code after the landing bookkeeping: the push is never undone
    and the refusal is never silent.
    """
    hook = path / POST_LAND
    if not (hook.is_file() and os.access(hook, os.X_OK)):
        return 0
    head = out(path, "rev-parse", "HEAD")
    print(f"work land: {POST_LAND} {base[:8]} {head[:8]}", flush=True)
    rc = subprocess.run([str(hook), base, head], cwd=path).returncode
    run.event("land.post_land", base=base, head=head, exit_code=rc)
    if rc != 0:
        print(f"work land: landed at {head[:8]}, but {POST_LAND} exited {rc}; see its output above", file=sys.stderr)
        return EXIT_FAILED
    return 0


def _land_attempt(path: Path, branch: str, attempt: int, run, gate, landed: dict | None = None) -> int | str:
    """One rebase-gate-push attempt: an exit code, or RETRY when main moved underneath.

    `landed`, when given, receives the base the push went on top of, so the
    caller can still name the landed diff after the push has moved origin/main.
    """
    if not fetch(path):
        print("work land: origin unreachable", file=sys.stderr)
        run.event("land.fetch_failed", attempt=attempt)
        return EXIT_FAILED
    if count(path, "origin/main..HEAD") == 0:
        print("work land: nothing to land; the branch is already contained in origin/main")
        return 0
    r = git(path, "rebase", "--quiet", "origin/main", check=False, timeout=300)
    if r.returncode != 0:
        conflicted = out(path, "diff", "--name-only", "--diff-filter=U", check=False)
        git(path, "rebase", "--abort", check=False)
        print(f"work land: rebase onto origin/main conflicts in: {conflicted or '(unknown)'}\n"
              "resolve with `git rebase origin/main` in the worktree, then `tatbot work land` again "
              "(the branch stays pushed meanwhile)", file=sys.stderr)
        run.event("land.rebase_conflict", files=conflicted.split())
        return EXIT_FAILED
    push_branch(path, branch)  # durable before any gate runs
    if attempt == 1 and gate is not None:
        rc = gate()
        run.event("land.check", exit_code=rc)
        if rc != 0:
            print("work land: the land gate failed; fix and land again (the branch is pushed)", file=sys.stderr)
            return EXIT_FAILED
    base = out(path, "rev-parse", "origin/main", check=False)
    r = git(path, "push", "origin", "HEAD:main", check=False, timeout=180)
    if r.returncode == 0:
        sha = out(path, "rev-parse", "--short", "HEAD")
        run.event("land.pushed", sha=sha, attempt=attempt)
        print(f"work land: {branch} landed on main at {sha}")
        if landed is not None:
            landed["base"] = base
        return 0
    if any(word in r.stderr for word in ("rejected", "fetch first", "non-fast-forward")):
        run.event("land.retry", attempt=attempt)
        print(f"work land: main moved; rebasing again ({attempt}/{LAND_RETRIES})")
        return RETRY
    print(f"work land: push refused:\n{r.stderr.strip()}", file=sys.stderr)
    run.event("land.push_refused", stderr=r.stderr[-2000:])
    return EXIT_FAILED


def _after_landing(path: Path, session: str, branch: str) -> None:
    git(path, "push", "--quiet", "origin", "--delete", branch, check=False, timeout=60)
    meta = read_meta(session)
    meta["parked"] = None
    meta["landed"] = {"at": utc(), "sha": out(path, "rev-parse", "--short", "HEAD")}
    write_meta(session, meta)


def _remove_landed(mirror: Path, path: Path, session: str, branch: str) -> None:
    git(mirror, "worktree", "remove", "--force", str(path))
    git(mirror, "branch", "-D", branch, check=False)
    meta_path(session).unlink(missing_ok=True)
    print(f"work land: removed {path}")


def cmd_land(a) -> int:
    from tatbot_runlog import init as runlog_init
    mirror = mirror_root()
    wt = require_worktree(a, mirror, "land")
    if not wt:
        return EXIT_USAGE
    path, session, branch = wt
    if porcelain(path):
        print("work land: the worktree has uncommitted changes; commit them, or `tatbot work park \"why\"`", file=sys.stderr)
        return EXIT_BUSY
    run = runlog_init("work-land", argv=sys.argv, meta={"session": session, "branch": branch})
    gate = None if a.no_check else (lambda: run_checks(path, full=a.full_check))
    code = EXIT_BUSY
    landed: dict = {}
    try:
        for attempt in range(1, LAND_RETRIES + 1):
            result = _land_attempt(path, branch, attempt, run, gate, landed)
            if result != RETRY:
                code = result
                break
        else:
            print(f"work land: main kept moving for {LAND_RETRIES} attempts; try again", file=sys.stderr)
        if code == 0:
            _after_landing(path, session, branch)
            if landed.get("base"):
                code = post_land(path, landed["base"], run)
            if a.done:  # after the hook, which runs from this worktree
                _remove_landed(mirror, path, session, branch)
        return code
    finally:
        run.finalize(code)


def _sweep_worktree(mirror: Path, info: dict, a, actions: list) -> None:
    path, branch = Path(info["path"]), info["branch"]
    if info["dirty_files"] and info["idle_minutes"] is not None and info["idle_minutes"] >= a.autosave_after:
        sha = wip_commit(path, f"wip: autosave by tatbot work sweep on {this_node()} "
                               f"({info['dirty_files']} file(s) idle {info['idle_minutes']:.0f} min)")
        actions.append({"action": "autosave", "session": info["session"], "sha": sha})
        info["unpushed"] = max(info["unpushed"], 1)
    if info["unpushed"]:
        actions.append({"action": "push", "session": info["session"], "ok": push_branch(path, branch)})
    idle_h = (info["idle_minutes"] or 0) / 60
    if not info["dirty_files"] and info["ahead_of_main"] == 0 and not info["parked"] and idle_h >= a.prune_after:
        git(mirror, "worktree", "remove", "--force", str(path))
        git(mirror, "branch", "-D", branch, check=False)
        git(mirror, "push", "--quiet", "origin", "--delete", branch, check=False, timeout=60)
        meta_path(info["session"]).unlink(missing_ok=True)
        actions.append({"action": "prune_idle_worktree", "session": info["session"]})


def _rescue_mirror(mirror: Path, a, actions: list) -> None:
    """Edits left in the mirror are the rot this flow exists to end: nobody owns
    them and nothing pushes them. After a day idle they move into a worktree of
    their own on a pushed branch named so the owner can find them."""
    dirty = porcelain(mirror)
    idle = newest_mtime(mirror, dirty)
    if not dirty or not idle or (time.time() - idle) / 3600 < a.rescue_after:
        return
    import contextlib
    import io
    ns = argparse.Namespace(session=f"mirror-rescue-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}",
                            task="mirror-rescue", adopt=True, json=True)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = cmd_start(ns)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        actions.append({"action": "rescue_mirror_edits", "session": ns.session, "ok": False, "error": str(error)})
        return
    if rc != 0:
        return
    rescued = json.loads(buf.getvalue())
    rescue_path = Path(rescued["path"])
    sha = wip_commit(rescue_path, f"wip: rescued {len(rescued['adopted'])} file(s) left in the mirror on {this_node()}")
    actions.append({"action": "rescue_mirror_edits", "session": ns.session, "files": rescued["adopted"],
                    "sha": sha, "pushed": push_branch(rescue_path, rescued["branch"])})


def cmd_sweep(a) -> int:
    from tatbot_runlog import init as runlog_init
    mirror = mirror_root()
    run = runlog_init("work-sweep", argv=sys.argv, emit_banner=not a.quiet)
    actions: list[dict] = []
    try:
        fetch(mirror)
        for wt in worktrees(mirror):
            info = inspect(mirror, wt) if session_of(wt["path"]) else None
            if info and info["branch"] and info["branch"].startswith(f"{BRANCH_PREFIX}/"):
                _sweep_worktree(mirror, info, a, actions)
        live = {t.get("branch") for t in worktrees(mirror)}
        for r in remote_agent_branches(mirror):
            if r["merged"] and r["age_hours"] >= 1 and r["branch"] not in live:
                ok = git(mirror, "push", "--quiet", "origin", "--delete", r["branch"], check=False, timeout=60).returncode == 0
                actions.append({"action": "prune_merged_remote", "branch": r["branch"], "ok": ok})
        _rescue_mirror(mirror, a, actions)
        if not porcelain(mirror) and is_main_worktree(mirror):
            r = git(mirror, "pull", "--quiet", "--ff-only", "origin", "main", check=False, timeout=120)
            actions.append({"action": "fast_forward_mirror", "ok": r.returncode == 0,
                            "head": out(mirror, "rev-parse", "--short", "HEAD")})
        s = summary(mirror)
        s["actions"] = actions
        run.event("sweep.done", actions=actions, unlanded=s["unlanded"], stale_remote=s["stale_remote"])
        emit(a, s, render(s) + "".join(f"\n  did {json.dumps(x, sort_keys=True)}" for x in actions))
        return 0
    finally:
        run.finalize(0)


def unit_text(name: str, mirror: Path) -> str:
    # The template comes from the copy that is running (a bootstrap worktree
    # can initialise a mirror that has not pulled this code yet); the unit
    # itself points at the mirror, which is where the sweeper must run.
    template = (script_repo() / "config" / "systemd" / name).read_text()
    return template.replace("@CHECKOUT@", str(mirror)).replace("@USER@", getpass.getuser())


def cmd_init(a) -> int:
    mirror = mirror_root()
    done = []
    git(mirror, "config", "tatbot.mirror", "true")
    done.append("mirror marked (tatbot.mirror=true): commits there are refused by the pre-commit hook")
    if out(mirror, "config", "core.hooksPath", check=False) != "scripts/githooks":
        git(mirror, "config", "core.hooksPath", "scripts/githooks")
        done.append("hooks armed (core.hooksPath=scripts/githooks)")
    work_root().mkdir(parents=True, exist_ok=True)
    if not a.no_timer:
        probe = subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, text=True)
        if probe.returncode in (0, 1) and probe.stdout.strip() not in ("", "offline", "unknown"):
            unit_dir = Path("~/.config/systemd/user").expanduser()
            unit_dir.mkdir(parents=True, exist_ok=True)
            for name in TIMER_UNITS:
                (unit_dir / name).write_text(unit_text(name, mirror))
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            r = subprocess.run(["systemctl", "--user", "enable", "--now", "--quiet", TIMER_UNITS[1]], capture_output=True, text=True)
            done.append("sweeper timer enabled (systemctl --user status tatbot-work-sweep.timer)" if r.returncode == 0
                        else f"sweeper timer NOT enabled: {r.stderr.strip()[:200]}")
        else:
            done.append("no user systemd here: run `tatbot work sweep` from cron or by hand")
    emit(a, {"schema": SCHEMA, "mirror": str(mirror), "work_root": str(work_root()), "done": done},
         "\n".join(f"init     {d}" for d in done))
    return 0


# --- hooks ---------------------------------------------------------------------------

def read_payload() -> dict:
    try:
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        return json.loads(raw) if raw.strip() else {}
    except ValueError:
        return {}


def state_hash(info: dict) -> str:
    key = json.dumps({k: info.get(k) for k in ("dirty_files", "ahead_of_main", "unpushed")}, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def hook_stop(a) -> int:
    payload = read_payload()
    cwd = Path(payload.get("cwd") or os.getcwd())
    mirror = mirror_root()
    top = toplevel(cwd)
    if not top or mirror_of(top) != mirror:
        return 0
    session = session_of(top)
    if session:
        info = inspect(mirror, {"path": top, "branch": current_branch(top)})
        if not info["unlanded"] or info["parked"]:
            return 0
        where = f"your worktree {top}"
        meta_key = session
    else:
        entries = porcelain(top)
        ahead = count(top, "origin/main..HEAD") if ref_exists(top, "origin/main") else 0
        if not entries and not ahead:
            return 0
        info = {"dirty_files": len(entries), "ahead_of_main": ahead, "unpushed": ahead}
        where = f"the shared checkout {top} (this session never moved into a worktree)"
        meta_key = "mirror"
    meta = read_meta(meta_key)
    blocks = meta.setdefault("stop_blocks", {})
    key = state_hash(info)
    blocks[key] = blocks.get(key, 0) + 1
    write_meta(meta_key, meta)
    msg = (f"tatbot work: unlanded work in {where}: {info['dirty_files']} uncommitted file(s), "
           f"{info['ahead_of_main']} commit(s) not on origin/main.\n"
           "Before stopping, either land it:   tatbot work land        (rebase, land gate, push to main)\n"
           "or park it with a reason:          tatbot work park \"why\"  (pushes a wip commit to your agent branch)\n")
    if not session:
        msg += "You are in the mirror; move the changes first:  tatbot work start --adopt --task <name>\n"
    if blocks[key] > STOP_BLOCK_LIMIT:
        print(msg + f"(stop allowed after {STOP_BLOCK_LIMIT} refusals with no change; the sweeper will autosave it)", file=sys.stderr)
        return 0
    print(msg, file=sys.stderr)
    return 2


def hook_session_start(a) -> int:
    payload = read_payload()
    cwd = Path(payload.get("cwd") or os.getcwd())
    session = payload.get("session_id") or f"session-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    mirror = mirror_root()
    os.environ.setdefault("TATBOT_WORK_NET_TIMEOUT", "12")
    top = toplevel(cwd)
    lines = []
    if top and session_of(top):
        info = inspect(mirror, {"path": top, "branch": current_branch(top)})
        lines.append(f"tatbot work: this session runs in its own worktree {top} on {info['branch']}"
                     f" (dirty={info['dirty_files']}, ahead_of_main={info['ahead_of_main']}).")
    else:
        ns = argparse.Namespace(session=session, task=None, adopt=False, json=True)
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_start(ns)
        try:
            created = json.loads(buf.getvalue())
        except ValueError:
            created = {}
        if rc == 0 and created:
            lines.append(f"tatbot work: your worktree for this session is {created['path']} on branch {created['branch']}.")
            lines.append("Do all edits, commits and checks THERE (cd into it, or change_directory to it), not in this "
                         "shared checkout, whose pre-commit hook refuses commits.")
        else:
            lines.append("tatbot work: could not create a worktree for this session; run `tatbot work start` by hand.")
    lines.append("Every commit on your agent branch is pushed automatically. Finish with `tatbot work land` "
                 "(rebase, land gate, push to main) or `tatbot work park \"why\"`; a session cannot stop with "
                 "unlanded, unparked work.")
    s = summary(mirror)
    if s["unlanded"] or s["stale_remote"] or not s["mirror"]["clean"]:
        lines.append(f"tatbot work on {s['node']}: unlanded worktrees={s['unlanded']} stale remote branches={s['stale_remote']}"
                     f" mirror {'clean' if s['mirror']['clean'] else 'DIRTY'} — `tatbot work status` lists them.")
    print("\n".join(lines))
    return 0


def hook_post_commit(a) -> int:
    top = toplevel(Path.cwd())
    branch = current_branch(top) if top else None
    if not top or not branch or not branch.startswith(f"{BRANCH_PREFIX}/"):
        return 0
    if not push_branch(top, branch):
        print(f"tatbot work: {branch} not pushed (origin unreachable?); the sweeper will retry", file=sys.stderr)
    return 0


def hook_pre_commit_guard(a) -> int:
    top = toplevel(Path.cwd())
    if not top:
        return 0
    if out(top, "config", "--bool", "tatbot.mirror", check=False) == "true" and is_main_worktree(top):
        print(f"pre-commit: {top} is the mirror checkout; commits here are refused.\n"
              "  tatbot work start --adopt --task <name>   moves these changes into your own worktree", file=sys.stderr)
        return 1
    return 0


# --- main ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tatbot_work", description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--json", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--task", help="what this worktree is for; names the session")
    s.add_argument("--session", help="explicit session id (the Claude Code hook passes its own)")
    s.add_argument("--adopt", action="store_true", help="move the mirror's uncommitted changes into the new worktree")
    s.set_defaults(fn=cmd_start)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("land")
    s.add_argument("--done", action="store_true", help="remove the worktree after landing")
    s.add_argument("--no-check", action="store_true", help="skip the land gate (CI still runs everything on main)")
    s.add_argument("--full-check", action="store_true", help="run the whole fast tier, not only the land gate")
    s.set_defaults(fn=cmd_land)
    s = sub.add_parser("park")
    s.add_argument("reason")
    s.set_defaults(fn=cmd_park)
    s = sub.add_parser("sweep")
    s.add_argument("--autosave-after", type=float, default=AUTOSAVE_AFTER_MIN, metavar="MIN")
    s.add_argument("--prune-after", type=float, default=IDLE_PRUNE_AFTER_H, metavar="HOURS")
    s.add_argument("--rescue-after", type=float, default=MIRROR_RESCUE_AFTER_H, metavar="HOURS")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_sweep)
    s = sub.add_parser("init")
    s.add_argument("--no-timer", action="store_true")
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("hook")
    s.add_argument("event", choices=("stop", "session-start", "post-commit", "pre-commit-guard"))
    s.set_defaults(fn=lambda a: {"stop": hook_stop, "session-start": hook_session_start,
                                 "post-commit": hook_post_commit, "pre-commit-guard": hook_pre_commit_guard}[a.event](a))
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except RuntimeError as exc:
        print(f"work {a.cmd}: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
