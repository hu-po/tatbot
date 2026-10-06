"""tatbot work: one session, one worktree, one branch, pushed always (docs/work.md).

A throwaway fleet: a bare origin, a mirror clone and a work root, all under
tmp_path. The backend is driven exactly as the CLI and the hooks drive it.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "scripts/lib/tatbot_work.py"
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


def git(cwd, *args, check=True):
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, env={**os.environ, **GIT_ENV})
    if check:
        assert r.returncode == 0, r.stderr
    return r


def commit_file(repo, name, text="x\n", message=None):
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", message or f"add {name}")
    return git(repo, "rev-parse", "--short", "HEAD").stdout.strip()


class Fleet:
    def __init__(self, tmp_path):
        self.origin = tmp_path / "origin.git"
        self.mirror = tmp_path / "mirror"
        self.work = tmp_path / "work"
        git(tmp_path, "init", "-q", "--bare", "-b", "main", str(self.origin))
        seed = tmp_path / "seed"
        git(tmp_path, "init", "-q", "-b", "main", str(seed))
        commit_file(seed, "README.md", "seed\n")
        git(seed, "remote", "add", "origin", str(self.origin))
        git(seed, "push", "-q", "origin", "main")
        git(tmp_path, "clone", "-q", str(self.origin), str(self.mirror))
        self.env = {**os.environ, **GIT_ENV, "TATBOT_WORK_ROOT": str(self.work), "TATBOT_WORK_MIRROR": str(self.mirror),
                    "TATBOT_LOG_ROOT": str(tmp_path / "logs"), "TATBOT_WORK_NET_TIMEOUT": "30"}

    def run(self, *args, cwd=None, stdin=None, json_out=True):
        argv = ["python3", str(BACKEND), *(["--json"] if json_out else []), *args]
        r = subprocess.run(argv, cwd=str(cwd or self.mirror), capture_output=True, text=True, input=stdin, env=self.env, timeout=120)
        payload = None
        if json_out and r.stdout.strip().startswith("{"):
            payload = json.loads(r.stdout)
        return r, payload

    def start(self, task="fix", **kw):
        r, payload = self.run("start", "--task", task, **kw)
        assert r.returncode == 0, r.stderr
        return Path(payload["path"]), payload["branch"]

    def other_clone(self, tmp_path, name="other"):
        clone = tmp_path / name
        git(tmp_path, "clone", "-q", str(self.origin), str(clone))
        return clone

    def origin_has(self, ref):
        return git(self.mirror, "ls-remote", "--exit-code", "--heads", "origin", ref, check=False).returncode == 0


@pytest.fixture
def fleet(tmp_path):
    return Fleet(tmp_path)


def test_start_creates_a_worktree_on_a_pushed_agent_branch(fleet):
    path, branch = fleet.start("fix the thing")
    assert path.is_dir() and path.parent == fleet.work
    assert branch.startswith("agent/") and branch.endswith("/fix-the-thing-" + path.name.rsplit("-", 1)[1])
    assert git(path, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == branch
    assert fleet.origin_has(branch)
    r, payload = fleet.run("start", "--session", path.name)  # idempotent
    assert r.returncode == 0 and payload["created"] is False
    r, status = fleet.run("status")
    assert status["unlanded"] == 0 and status["worktrees"][0]["session"] == path.name


def test_land_rebases_over_a_moving_main_and_drops_the_branch(fleet, tmp_path):
    path, branch = fleet.start()
    mine = commit_file(path, "mine.txt")
    other = fleet.other_clone(tmp_path)
    theirs = commit_file(other, "theirs.txt")
    git(other, "push", "-q", "origin", "main")
    r, _ = fleet.run("land", "--no-check", cwd=path, json_out=False)
    assert r.returncode == 0, r.stderr + r.stdout
    log = git(fleet.mirror, "fetch", "-q", "origin") and git(fleet.mirror, "log", "--format=%s", "origin/main").stdout.split()
    assert log[:3] == ["add", "mine.txt", "add"][:3] and "theirs.txt" in git(fleet.mirror, "log", "--format=%s", "origin/main").stdout
    assert not fleet.origin_has(branch), "the agent branch is deleted once it is contained in main"
    assert git(path, "rev-list", "--count", "origin/main..HEAD").stdout.strip() == "0"
    assert mine != theirs
    r, status = fleet.run("status")
    assert status["unlanded"] == 0 and status["worktrees"][0]["ahead_of_main"] == 0


def test_land_hands_the_landed_range_to_the_repository_post_land_hook(fleet, tmp_path):
    """A private tree may keep infrastructure that must react to what landed
    (this one's isolated CI worker refuses main after an image input changes
    until a person rebuilds it). The hook decides; the land hands it the range
    and answers for its exit code, after the push, never instead of it."""
    seed = fleet.other_clone(tmp_path, "seed2")
    hook = seed / "internal" / "ci" / "post-land"
    hook.parent.mkdir(parents=True)
    calls = tmp_path / "post-land-calls"
    hook.write_text(f"#!/usr/bin/env bash\necho \"$1 $2\" >> {calls}\n[ -f {tmp_path}/hook-fails ] && exit 3\nexit 0\n")
    hook.chmod(0o755)
    git(seed, "add", "-A")
    git(seed, "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "a post-land hook")
    git(seed, "push", "-q", "origin", "main")

    path, _ = fleet.start("hooked")
    base = git(fleet.mirror, "fetch", "-q", "origin") and git(fleet.mirror, "rev-parse", "origin/main").stdout.strip()
    commit_file(path, "mine.txt")
    r, _ = fleet.run("land", "--no-check", cwd=path, json_out=False)
    assert r.returncode == 0, r.stderr + r.stdout
    head = git(path, "rev-parse", "HEAD").stdout.strip()
    assert calls.read_text().splitlines() == [f"{base} {head}"], "the range that landed, base first"

    (tmp_path / "hook-fails").write_text("")
    path, branch = fleet.start("hook-fails")
    commit_file(path, "more.txt")
    r, _ = fleet.run("land", "--no-check", cwd=path, json_out=False)
    assert r.returncode == 1 and "post-land exited 3" in r.stderr
    assert git(path, "rev-list", "--count", "origin/main..HEAD").stdout.strip() == "0", "landed all the same"
    assert not fleet.origin_has(branch)
    assert len(calls.read_text().splitlines()) == 2

    (tmp_path / "hook-fails").unlink()
    path, _ = fleet.start("hooked-done")
    commit_file(path, "last.txt")
    r, _ = fleet.run("land", "--no-check", "--done", cwd=path, json_out=False)
    assert r.returncode == 0, r.stderr + r.stdout
    assert len(calls.read_text().splitlines()) == 3, "--done runs the hook before removing the worktree"
    assert not path.exists()


def test_land_refuses_a_dirty_tree_and_park_pushes_a_wip_commit(fleet):
    path, branch = fleet.start()
    (path / "draft.txt").write_text("half\n")
    r, _ = fleet.run("land", "--no-check", cwd=path, json_out=False)
    assert r.returncode == 6 and "uncommitted" in r.stderr
    r, parked = fleet.run("park", "waiting on the arm node", cwd=path)
    assert r.returncode == 0 and parked["wip_commit"] and parked["pushed"]
    assert git(fleet.mirror, "fetch", "-q", "origin") or True
    assert "wip: parked" in git(fleet.mirror, "log", "-1", "--format=%s", f"origin/{branch}").stdout
    r, status = fleet.run("status")
    assert status["parked"] == 1 and status["worktrees"][0]["parked"]["reason"] == "waiting on the arm node"


def test_stop_hook_blocks_unlanded_work_until_landed_or_parked(fleet):
    path, branch = fleet.start()
    payload = json.dumps({"session_id": path.name, "cwd": str(path), "hook_event_name": "Stop"})
    r, _ = fleet.run("hook", "stop", cwd=path, stdin=payload, json_out=False)
    assert r.returncode == 0, "a clean worktree at origin/main may stop"
    (path / "draft.txt").write_text("half\n")
    codes = []
    for _ in range(5):
        r, _ = fleet.run("hook", "stop", cwd=path, stdin=payload, json_out=False)
        codes.append(r.returncode)
    assert codes == [2, 2, 2, 0, 0] and "tatbot work land" in r.stderr and "tatbot work park" in r.stderr
    commit_file(path, "draft.txt", "done\n")
    r, _ = fleet.run("hook", "stop", cwd=path, stdin=payload, json_out=False)
    assert r.returncode == 2, "committed but not on main is still unlanded (a new state, so it blocks again)"
    fleet.run("park", "handing over", cwd=path)
    r, _ = fleet.run("hook", "stop", cwd=path, stdin=payload, json_out=False)
    assert r.returncode == 0


def test_stop_hook_refuses_uncommitted_edits_in_the_mirror(fleet):
    (fleet.mirror / "oops.txt").write_text("edited the mirror\n")
    payload = json.dumps({"session_id": "s1", "cwd": str(fleet.mirror)})
    r, _ = fleet.run("hook", "stop", stdin=payload, json_out=False)
    assert r.returncode == 2 and "work start --adopt" in r.stderr


def test_start_adopt_moves_mirror_edits_into_the_worktree(fleet):
    commit_file(fleet.mirror, "tracked.txt", "v1\n")
    git(fleet.mirror, "push", "-q", "origin", "main")
    (fleet.mirror / "tracked.txt").write_text("v2\n")
    (fleet.mirror / "new.txt").write_text("new\n")
    r, payload = fleet.run("start", "--task", "rescue", "--adopt")
    assert r.returncode == 0 and sorted(payload["adopted"]) == ["new.txt", "tracked.txt"]
    path = Path(payload["path"])
    assert (path / "tracked.txt").read_text() == "v2\n" and (path / "new.txt").read_text() == "new\n"
    assert git(fleet.mirror, "status", "--porcelain").stdout == "", "the mirror is clean again"


def test_sweep_autosaves_idle_edits_pushes_and_fast_forwards_the_mirror(fleet, tmp_path):
    path, branch = fleet.start()
    draft = path / "draft.txt"
    draft.write_text("idle\n")
    old = time.time() - 3600
    os.utime(draft, (old, old))
    other = fleet.other_clone(tmp_path)
    commit_file(other, "upstream.txt")
    git(other, "push", "-q", "origin", "main")
    r, report = fleet.run("sweep", "--autosave-after", "30")
    assert r.returncode == 0, r.stderr
    actions = {a["action"] for a in report["actions"]}
    assert {"autosave", "push", "fast_forward_mirror"} <= actions
    assert "wip: autosave" in git(fleet.mirror, "log", "-1", "--format=%s", f"origin/{branch}").stdout
    assert (fleet.mirror / "upstream.txt").exists(), "the mirror followed origin/main"
    assert git(path, "status", "--porcelain").stdout == ""


def test_sweep_prunes_only_landed_idle_worktrees_and_merged_remote_branches(fleet, tmp_path):
    path, branch = fleet.start("done")
    for p in (path,):
        os.utime(p / "README.md", (time.time() - 90000,) * 2)
    meta = json.loads((fleet.work / f"{path.name}.json").read_text())
    meta["created"] = "2026-01-01T00:00:00Z"
    (fleet.work / f"{path.name}.json").write_text(json.dumps(meta))
    # an idle worktree with nothing ahead of main and nothing dirty is pruned...
    r, report = fleet.run("sweep", "--prune-after", "0")
    assert any(a["action"] == "prune_idle_worktree" for a in report["actions"]), report["actions"]
    assert not path.exists() and not fleet.origin_has(branch)
    # ...but one with unlanded commits is never touched
    path, branch = fleet.start("busy")
    commit_file(path, "keep.txt")
    r, report = fleet.run("sweep", "--prune-after", "0")
    assert path.exists() and fleet.origin_has(branch)


def test_init_marks_the_mirror_and_the_guard_refuses_commits_there(fleet):
    r, payload = fleet.run("init", "--no-timer")
    assert r.returncode == 0 and any("mirror marked" in d for d in payload["done"])
    assert git(fleet.mirror, "config", "--bool", "tatbot.mirror").stdout.strip() == "true"
    r, _ = fleet.run("hook", "pre-commit-guard", json_out=False)
    assert r.returncode == 1 and "work start --adopt" in r.stderr
    path, _ = fleet.start()
    r, _ = fleet.run("hook", "pre-commit-guard", cwd=path, json_out=False)
    assert r.returncode == 0, "worktrees of the mirror are where commits belong"


def test_post_commit_hook_pushes_agent_branches_only(fleet):
    path, branch = fleet.start()
    commit_file(path, "a.txt")
    r, _ = fleet.run("hook", "post-commit", cwd=path, json_out=False)
    assert r.returncode == 0
    assert git(fleet.mirror, "ls-remote", "--heads", "origin", branch).stdout.strip().startswith(
        git(path, "rev-parse", "HEAD").stdout.strip())
    commit_file(fleet.mirror, "b.txt")  # on main, not an agent branch: nothing pushed
    r, _ = fleet.run("hook", "post-commit", json_out=False)
    assert r.returncode == 0
    assert git(fleet.mirror, "rev-list", "--count", "origin/main..HEAD").stdout.strip() == "1"


def test_session_start_hook_creates_the_worktree_and_says_where_to_work(fleet):
    payload = json.dumps({"session_id": "abc123", "cwd": str(fleet.mirror), "source": "startup"})
    r, _ = fleet.run("hook", "session-start", stdin=payload, json_out=False)
    assert r.returncode == 0
    assert str(fleet.work / "abc123") in r.stdout and "tatbot work land" in r.stdout
    assert (fleet.work / "abc123").is_dir()
    payload = json.dumps({"session_id": "abc123", "cwd": str(fleet.work / "abc123"), "source": "resume"})
    r, _ = fleet.run("hook", "session-start", stdin=payload, json_out=False)
    assert "runs in its own worktree" in r.stdout


def test_pre_push_hook_gates_only_pushes_to_main(tmp_path):
    hook = REPO / "scripts/githooks/pre-push"
    refs = "refs/heads/agent/n/s 0000000000000000000000000000000000000001 refs/heads/agent/n/s 0000000000000000000000000000000000000002\n"
    r = subprocess.run(["bash", str(hook), "origin", "x"], input=refs, capture_output=True, text=True, cwd=str(REPO), timeout=30)
    assert r.returncode == 0 and "export" not in r.stdout + r.stderr


def test_sweep_rescues_edits_left_idle_in_the_mirror(fleet, tmp_path):
    (fleet.mirror / "README.md").write_text("edited in the mirror and forgotten\n")
    (fleet.mirror / "forgotten.txt").write_text("also\n")
    for name in ("README.md", "forgotten.txt"):
        os.utime(fleet.mirror / name, (time.time() - 100000,) * 2)
    # Main moved on and rewrote the same line: the rescue takes the edits as
    # the patch they are, cut at the mirror's own revision, never as a merge.
    stale = git(fleet.mirror, "rev-parse", "HEAD").stdout.strip()
    other = fleet.other_clone(tmp_path)
    commit_file(other, "README.md", "rewritten on main meanwhile\n")
    git(other, "push", "-q", "origin", "main")
    r, report = fleet.run("sweep", "--rescue-after", "0")
    assert r.returncode == 0, r.stderr
    rescue = [a for a in report["actions"] if a["action"] == "rescue_mirror_edits"]
    assert rescue and sorted(rescue[0]["files"]) == ["README.md", "forgotten.txt"] and rescue[0]["pushed"]
    assert git(fleet.mirror, "status", "--porcelain").stdout == ""
    rescued = next(t for t in report["worktrees"] if t["session"] == rescue[0]["session"])
    branch = rescued["branch"]
    assert "wip: rescued 2 file(s)" in git(fleet.mirror, "log", "-1", "--format=%s", f"origin/{branch}").stdout
    assert git(Path(rescued["path"]), "rev-parse", "HEAD~1").stdout.strip() == stale, "cut at the mirror's revision"
    assert (Path(rescued["path"]) / "README.md").read_text() == "edited in the mirror and forgotten\n"
    assert git(fleet.mirror, "rev-parse", "HEAD").stdout.strip() != stale, "the clean mirror fast-forwarded to the new main"
    r, report = fleet.run("sweep", "--rescue-after", "0")
    assert not any(a["action"] == "rescue_mirror_edits" for a in report["actions"]), "nothing left to rescue"


def test_branch_push_survives_an_amend_and_a_rebase(fleet, tmp_path):
    path, branch = fleet.start()
    commit_file(path, "a.txt")
    fleet.run("hook", "post-commit", cwd=path, json_out=False)
    git(path, "-c", "core.hooksPath=/dev/null", "commit", "-q", "--amend", "-m", "add a.txt (amended)")
    r, _ = fleet.run("hook", "post-commit", cwd=path, json_out=False)
    assert r.returncode == 0 and "not pushed" not in r.stderr
    assert git(fleet.mirror, "ls-remote", "--heads", "origin", branch).stdout.startswith(git(path, "rev-parse", "HEAD").stdout.strip())


def test_this_node_maps_a_hostname_alias(monkeypatch):
    """Agent branches are named for the node, not for a hostname alias of it."""
    import socket

    import tatbot_work
    monkeypatch.delenv("TATBOT_NODE", raising=False)
    monkeypatch.setattr(socket, "gethostname", lambda: "lab-trainer")
    monkeypatch.setattr(tatbot_work.tatbot_nodes, "load", lambda repo: {"trainer": {"hostname": "lab-trainer"}})
    assert tatbot_work.this_node() == "trainer"
