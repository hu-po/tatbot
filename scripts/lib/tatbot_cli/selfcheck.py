"""What `scripts/check cli` and `scripts/check nodes` run. Stdlib only, no pytest.

- every verb answers --help and its registered example answers --dry-run (exit 0)
  with the node and tool it needs supplied through the environment;
- no orphans: every executable under scripts/ and every sim entry module is
  wrapped by a verb or listed with a reason in config/cli-orphans.txt, and every
  listed path still exists;
- docs/cli.md is what `schema --md` generates now;
- (nodes) config/nodes.json agrees with the deployment network document
  (tatbot_cli.nodes_parity, present only where a fleet is described).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tatbot_cli import nodes
from tatbot_cli.registry import all_verbs, repo_root

SIM_PKG = "python/tatbot_sim/src/tatbot_sim"
EXTRA_ENTRY_POINTS = ("cpp/teleop/analyze_log.py",)


def _shim(repo: Path) -> list[str]:
    return [sys.executable, str(repo / "scripts" / "lib" / "tatbot_cli")]


def _run(repo: Path, args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(_shim(repo) + args, capture_output=True, text=True, env=e, timeout=60)


def check_help_and_dry_run(repo: Path) -> list[str]:
    problems = []
    nmap = nodes.load(repo)
    public_checkout = not (repo / "config" / "profiles" / "tatbot.json").is_file()
    for v in all_verbs():
        argv = [v.noun, *v.verb.split()]
        r = _run(repo, argv + ["--help"])
        if r.returncode != 0:
            problems.append(f"{v.name}: --help exit {r.returncode}: {r.stderr.strip()[:200]}")
        role_nodes = nodes.nodes_with(nmap, v.role) if v.role else []
        node = role_nodes[0] if role_nodes else (next(iter(nmap), "") or "localnode")
        env = {"TATBOT_NODE": node, "TATBOT_EE_TOOL": "picosecond-laser-pen", "TATBOT_TRAIN_ROOT": "/nonexistent"}
        if public_checkout:
            env["TATBOT_PROFILE"] = "example"
        if any(a.startswith("<") for a in v.example):
            continue  # placeholder example: no fleet described here to run it against
        routing = ["--no-hop"] if public_checkout and not nmap else []
        r = _run(repo, ["--dry-run", *routing, *argv, *v.example], env)
        expected = {0, 3} if public_checkout and v.name == "profile check" else {0}
        if r.returncode not in expected:
            problems.append(f"{v.name}: --dry-run {' '.join(v.example)} exit {r.returncode}: {(r.stderr or r.stdout).strip()[:300]}")
    return problems


def _tracked(repo: Path, *paths: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", *paths], cwd=repo, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]


def entry_points(repo: Path) -> list[str]:
    """Things a human could run: executables and *.sh/*.py under scripts/, sim entry modules."""
    found = []
    for rel in _tracked(repo, "scripts"):
        if rel.startswith(("scripts/lib/", "scripts/tests/")) or "__pycache__" in rel or rel == "scripts/tatbot":
            continue
        p = repo / rel
        if rel.endswith((".sh", ".bash")) or (not rel.endswith(".py") and os.access(p, os.X_OK)):
            found.append(rel)
        elif rel.endswith(".py"):
            text = p.read_text(errors="replace")
            if "__main__" in text or "tyro.cli" in text:
                found.append(rel)
    for rel in _tracked(repo, SIM_PKG):
        if rel.endswith(".py") and not rel.endswith("__init__.py"):
            text = (repo / rel).read_text(errors="replace")
            if "__main__" in text or "tyro.cli" in text:
                found.append(rel)
    found.extend(EXTRA_ENTRY_POINTS)
    return sorted(set(found))


def orphans_file(repo: Path) -> dict[str, str]:
    out = {}
    for line in (repo / "config" / "cli-orphans.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path, _, reason = line.partition(" ")
        out[path] = reason.strip()
    return out


def check_orphans(repo: Path) -> list[str]:
    problems = []
    wrapped = {w for v in all_verbs() for w in v.wraps}
    listed = orphans_file(repo)
    for path in entry_points(repo):
        if path in wrapped or path in listed:
            continue
        problems.append(f"orphan: {path} — give it a `tatbot` verb or a line in config/cli-orphans.txt")
    for path, reason in listed.items():
        if not (repo / path).exists():
            problems.append(f"config/cli-orphans.txt lists {path}, which no longer exists")
        if not reason:
            problems.append(f"config/cli-orphans.txt: {path} has no reason")
        if path in wrapped and "listed for the scanner" not in reason:
            problems.append(f"config/cli-orphans.txt: {path} is wrapped by a verb; drop the line")
    return problems


def check_docs(repo: Path) -> list[str]:
    from tatbot_cli import schema
    want = schema.as_markdown(repo)
    doc = repo / "docs" / "cli.md"
    have = doc.read_text() if doc.is_file() else ""
    if have.rstrip("\n") != want.rstrip("\n"):
        return ["docs/cli.md is stale — regenerate: scripts/tatbot schema --md > docs/cli.md"]
    return []


def main(argv: list[str]) -> int:
    repo = repo_root()
    what = argv[0] if argv else "cli"
    if what == "nodes":
        try:
            from tatbot_cli import nodes_parity
        except ImportError:
            print("SKIP: no nodes_parity module (no fleet document for this checkout)")
            return 0
        problems, skip = nodes_parity.check(repo)
        if skip:
            print(f"SKIP: {skip}")
            return 0
    else:
        problems = check_help_and_dry_run(repo) + check_orphans(repo) + check_docs(repo)
    for p in problems:
        print(p, file=sys.stderr)
    if not problems:
        n = len(all_verbs())
        print(f"cli: {n} verbs answer --help and --dry-run; no orphans; docs/cli.md current" if what != "nodes"
              else "nodes: config/nodes.json agrees with the network document")
    return 1 if problems else 0
