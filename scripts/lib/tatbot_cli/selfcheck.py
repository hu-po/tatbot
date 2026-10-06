"""What `scripts/check cli` and `scripts/check nodes` run. Stdlib only, no pytest.

- every verb answers --help and its registered example answers --dry-run with
  the node and tool it needs; an unconfigured public fleet must refuse ownership;
- root help lists every noun; selected effects agree;
- no orphans: every executable under scripts/ and every sim entry module is
  wrapped by a verb or listed with a reason in config/cli-orphans.txt, and every
  listed path still exists;
- docs/cli.md is what `schema --md` generates now;
- (nodes) config/nodes.json agrees with the deployment network document
  (tatbot_cli.nodes_parity, present only where a fleet is described), and
  every rig-LAN address in vision.toml, the driver profile and config/trossen
  lives in the `__rig__` subnet it names (tatbot_cli.rig_addressing).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tatbot_cli import EXIT_WRONG_NODE, nodes
from tatbot_cli.registry import all_verbs, repo_root

SIM_PKG = "python/tatbot_sim/src/tatbot_sim"
EXTRA_ENTRY_POINTS = ("cpp/teleop/analyze_log.py",)


def _shim(repo: Path) -> list[str]:
    return [sys.executable, str(repo / "scripts" / "lib" / "tatbot_cli")]


def _run(repo: Path, args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(_shim(repo) + args, capture_output=True, text=True, env=e, timeout=60)


def check_root_help(repo: Path) -> list[str]:
    """Compare rendered rows with the registry, independently of grouping tables."""
    result = _run(repo, ["--help"])
    if result.returncode:
        return [f"root help: exit {result.returncode}: {result.stderr.strip()[:200]}"]
    rows = re.findall(r"^  ([a-z][a-z0-9-]*)\s{2,}.*\[[^\n]+\]$", result.stdout, re.MULTILINE)
    expected = {v.noun for v in all_verbs()}
    problems = [f"root help: missing noun {noun}" for noun in sorted(expected - set(rows))]
    problems += [f"root help: unexpected noun {noun}" for noun in sorted(set(rows) - expected)]
    problems += [f"root help: duplicate noun {noun}" for noun in sorted(set(rows)) if rows.count(noun) != 1]
    return problems


def check_plan_effects(repo: Path) -> list[str]:
    """A few behavioral expectations, deliberately independent of effect callbacks."""
    cases = [
        ("sim sample", ["--count", "2"], {"write_files", "network", "environment_setup"}, {"autonomous_motion"}),
        ("logs", ["list"], {"read_files"}, {"write_files", "delete_files"}),
        ("logs", ["prune", "--yes"], {"delete_files"}, set()),
        ("status", [], {"read_files"}, {"remote_exec"}),
        ("status", ["--fleet"], {"remote_exec", "network"}, {"write_files", "delete_files"}),
    ]
    available = {v.name for v in all_verbs()}
    problems = []
    for name, args, required, forbidden in cases:
        if name not in available:
            continue  # Filtered exports need only qualify commands they provide.
        result = _run(repo, ["--json", "--dry-run", "--no-hop", *name.split(), *args],
                      {"TATBOT_NODE": "localnode"})
        label = " ".join([name, *args])
        try:
            if result.returncode:
                raise ValueError(f"exit {result.returncode}: {result.stderr.strip()[:200]}")
            plan = json.loads(result.stdout)
            effects = set(plan["effects"])
            missing, extra = required - effects, forbidden & effects
            if missing or extra:
                raise ValueError(f"missing effects {sorted(missing)}; forbidden effects {sorted(extra)}")
            network = bool(effects & {"network", "remote_exec", "remote_write"})
            if plan["requirements"]["network"] is not network:
                raise ValueError("network requirement disagrees with effects")
        except (ValueError, KeyError, TypeError) as error:
            problems.append(f"plan effects: {label}: {error}")
    return problems


def check_help_and_dry_run(repo: Path) -> list[str]:
    """Every verb answers --help and its example dry-runs; 229 verbs is ~460
    isolated CLI processes, so they run concurrently -- the probes share
    nothing but the read-only checkout."""
    nmap = nodes.load(repo)
    public_checkout = not (repo / "config" / "profiles" / "tatbot.json").is_file()
    verbs = list(all_verbs())
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
        results = pool.map(lambda v: _verb_problems(repo, v, nmap, public_checkout), verbs)
    return [problem for problems in results for problem in problems]


def _verb_problems(repo: Path, v, nmap, public_checkout: bool) -> list[str]:
    problems = []
    argv = [v.noun, *v.verb.split()]
    r = _run(repo, argv + ["--help"])
    if r.returncode != 0:
        problems.append(f"{v.name}: --help exit {r.returncode}: {r.stderr.strip()[:200]}")
    from tatbot_cli.cli import example_role
    role = example_role(v)
    role_nodes = nodes.nodes_with(nmap, role) if role else []
    node = role_nodes[0] if role_nodes else (next(iter(nmap), "") or "localnode")
    env = {"TATBOT_NODE": node, "TATBOT_EE_TOOL": "lutin-ballpoint-dot", "TATBOT_TRAIN_ROOT": "/nonexistent"}
    if public_checkout:
        env["TATBOT_PROFILE"] = "example"
    if any(a.startswith("<") for a in v.example):
        return problems  # placeholder example: no fleet described here to run it against
    if role and nmap and not role_nodes:
        return problems  # no node in this fleet holds the role (a retired owner): nothing to run it on
    routing = ["--no-hop"] if public_checkout and not nmap else []
    r = _run(repo, ["--json", "--dry-run", *routing, *argv, *v.example], env)
    missing_owner = role if public_checkout and not nmap and v.auto_hop else None
    expected = {0, 3} if public_checkout and v.name == "profile check" else {0}
    if missing_owner:
        expected = {EXIT_WRONG_NODE}
    return problems + _example_result_problems(v, r, expected, missing_owner)


def _example_result_problems(v, r, expected: set[int], missing_owner: str | None) -> list[str]:
    problems = []
    if r.returncode not in expected:
        problems.append(f"{v.name}: --dry-run {' '.join(v.example)} exit {r.returncode}: {(r.stderr or r.stdout).strip()[:300]}")
    elif missing_owner:
        try:
            error = json.loads(r.stderr)
            assert error["schema"] == "tatbot.cli-error/1"
            assert error["code"] == EXIT_WRONG_NODE and error["gate"] == "node"
            assert missing_owner in re.findall(r"[a-z][a-z0-9-]*", error["reason"])
        except (ValueError, KeyError, TypeError, AssertionError):
            problems.append(f"{v.name}: invalid missing-owner refusal contract")
    elif r.returncode == 0:
        try:
            plan = json.loads(r.stdout)
            assert plan["schema"] == "tatbot.cli-plan/2"
            assert plan["kind"] in ("native", "exec") and plan["effects"]
            assert isinstance(plan["requirements"]["network"], bool)
            assert plan["argv"] if plan["kind"] == "exec" else not plan["argv"]
        except (ValueError, KeyError, AssertionError, TypeError):
            problems.append(f"{v.name}: invalid JSON plan contract")
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


def _orphan_rows(repo: Path) -> list[tuple[str, str]]:
    path = repo / "config" / "cli-orphans.txt"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entry, _, reason = line.partition(" ")
        rows.append((entry, reason.strip()))
    return rows


def orphans_file(repo: Path) -> dict[str, str]:
    return dict(_orphan_rows(repo))


def duplicate_orphans(repo: Path) -> list[str]:
    seen = set()
    problems = []
    for path, _reason in _orphan_rows(repo):
        if path in seen:
            problems.append(f"config/cli-orphans.txt lists {path} more than once")
        seen.add(path)
    return problems


def check_orphans(repo: Path) -> list[str]:
    problems = []
    # A helper keeps its declared owner when a filtered checkout lacks another
    # launcher required by that verb. Help still lists only available commands.
    wrapped = {w for v in all_verbs(include_unavailable=True) for w in v.wraps}
    listed = orphans_file(repo)
    problems.extend(duplicate_orphans(repo))
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
        for reference in re.findall(r"\b(?:scripts|config|python|cpp|rust|docs|internal)/[\w./-]+\.(?:md|py|sh|json|yaml|timer|service)\b", reason):
            if not (repo / reference).is_file():
                problems.append(f"config/cli-orphans.txt: {path} cites missing owner/reference {reference}")
    return problems


def check_docs(repo: Path) -> list[str]:
    from tatbot_cli import doclinks, schema
    want = schema.as_markdown(repo)
    doc = repo / "docs" / "cli.md"
    have = doc.read_text() if doc.is_file() else ""
    problems = []
    if have.rstrip("\n") != want.rstrip("\n"):
        problems.append("docs/cli.md is stale — regenerate: scripts/tatbot schema --md > docs/cli.md")
    # The complete private table lives only on the private tree (the export drops internal/).
    full_doc = repo / "internal" / "reference" / "cli-full.md"
    if full_doc.is_file() and full_doc.read_text().rstrip("\n") != schema.as_markdown(repo, full=True).rstrip("\n"):
        problems.append("docs/cli-full.md is stale — regenerate: "
                        "scripts/tatbot schema --md --full > docs/cli-full.md")
    # The command audit is generated from the registry and the audience table,
    # so it describes what this checkout provides rather than a past review.
    audit = repo / "internal" / "reference" / "cli-disposition.md"
    if audit.is_file() and audit.read_text().rstrip("\n") != schema.as_disposition(repo).rstrip("\n"):
        # Name the file by the path already resolved above: the exported tree
        # has no such directory, so a literal would be a dead link in it.
        problems.append(f"{audit.relative_to(repo)} is stale — regenerate: "
                        f"scripts/tatbot schema --disposition > {audit.relative_to(repo)}")
    paths = {repo / v.doc.split("#")[0] for v in all_verbs() if v.doc}
    paths.add(doc)
    if full_doc.is_file():
        paths.update((full_doc, repo / "docs/cli.md", audit))
    problems.extend(doclinks.check(repo, paths))
    for v in all_verbs():
        if v.doc:
            problem = doclinks.target_problem(repo, repo / "registry.md", v.doc)
            if problem:
                problems.append(problem)
    return problems


def check_registry(repo: Path) -> list[str]:
    """Require an explicit, serializable interface contract for every command."""
    from tatbot_cli import schema
    from tatbot_cli.aliases import ALIASES
    from tatbot_cli.audience import AUDIENCES, DISPOSITION

    problems = []
    names = {v.name for v in all_verbs(include_unavailable=True)}
    for listed in sorted(set(DISPOSITION) - names):
        problems.append(f"tatbot_cli/audience.py lists {listed!r}, which is not a command")
    verbs = all_verbs()
    from tatbot_cli.registry import NOUNS
    names = [v.name for v in verbs]
    for alias in ALIASES:
        # Exported checkouts hide commands whose wrapped programs are absent.
        if alias.canonical_name not in names and (repo / "internal").is_dir():
            problems.append(f"alias has no canonical target: {alias.source}")
    if len({a.source for a in ALIASES}) != len(ALIASES):
        problems.append("duplicate alias source")
    for v in verbs:
        if names.count(v.name) != 1:
            problems.append(f"duplicate command: {v.name}")
        if v.noun not in NOUNS:
            problems.append(f"{v.name}: missing noun declaration")
        if v.group not in ("operator", "development", "administration"):
            problems.append(f"{v.name}: unknown help group {v.group}")
        if not v.effects or not v.group:
            problems.append(f"{v.name}: effects and group must be declared")
        if v.output not in ("text", "json", "json-lines"):
            problems.append(f"{v.name}: invalid output capability {v.output}")
        if v.visibility not in ("private", "public"):
            problems.append(f"{v.name}: invalid visibility {v.visibility}")
        if v.audience not in AUDIENCES:
            problems.append(f"{v.name}: unknown help audience {v.audience!r}")
        if not v.disposition:
            problems.append(f"{v.name}: no entry in tatbot_cli/audience.py — every command needs a "
                            "stated audience and the reason it sits there")
        if v.canonical_name not in names:
            problems.append(f"{v.name}: missing canonical command {v.canonical_name}")
        flags = [flag for arg in v.argument_spec.arguments for flag in arg.names if flag.startswith("-")]
        if len(flags) != len(set(flags)):
            problems.append(f"{v.name}: duplicate owned option")
        destinations = [a.schema()["name"] for a in v.argument_spec.arguments]
        if len(destinations) != len(set(destinations)) or {"_verb", "_sub"}.intersection(destinations):
            problems.append(f"{v.name}: argument destination collision")
        for arg in v.argument_spec.arguments:
            for name in arg.names:
                if name in ("--on", "--ee-tool", "--dry-run", "--explain", "--no-hop"):
                    problems.append(f"{v.name}: command option collides with global {name}")
                if name == "--json" and "--output" not in arg.names:
                    problems.append(f"{v.name}: ambiguous command --json (only the deprecated report alias is allowed)")
    try:
        json.dumps(schema.as_dict(repo))
    except (TypeError, ValueError) as exc:
        problems.append(f"schema is not JSON serializable: {exc}")
    return problems


def check_alternate_plans(repo: Path) -> list[str]:
    """Exercise selectors and compatibility through the real parser, plan only."""
    problems = []
    names = {v.name for v in all_verbs()}
    pairs = [
        ("sim compile --sample 2", "sim sample --count 2"),
        ("sim preview --reach", "sim reach"),
        ("sim cinematic --viewer /tmp/renders", "sim viewer /tmp/renders"),
        ("sim audit --samples /tmp/dataset", "sim samples /tmp/dataset"),
        ("sim eval /tmp/dataset", "sim eval dataset /tmp/dataset"),
        ("sim eval --policy-rollout -- --client-mode hold-control", "sim eval policy -- --client-mode hold-control"),
    ]
    for old, new in pairs:
        if not any(new == name or new.startswith(name + " ") for name in names):
            continue
        reports = []
        for form in (old, new):
            r = _run(repo, ["--json", "--dry-run", "--no-hop", *form.split()], {"TATBOT_NODE": "localnode"})
            try:
                assert r.returncode == 0
                report = json.loads(r.stdout)
                report.pop("deprecations")
                reports.append(report)
            except (ValueError, KeyError, AssertionError):
                problems.append(f"alternate plan failed: {form}: {r.stderr.strip()[:200]}")
        if len(reports) == 2 and reports[0] != reports[1]:
            problems.append(f"alias plan differs from canonical: {old}")
    return problems


def main(argv: list[str]) -> int:
    repo = repo_root()
    what = argv[0] if argv else "cli"
    if what == "nodes":
        from tatbot_cli import rig_addressing
        problems, rig_skip = rig_addressing.check(repo)
        for line in rig_addressing.remaining(repo):
            print(f"nodes: still on the old rig subnet: {line}")
        try:
            from tatbot_cli import nodes_parity
        except ImportError:
            nodes_parity = None
        if nodes_parity is None:
            parity_skip = "no nodes_parity module (no fleet document for this checkout)"
        else:
            parity_problems, parity_skip = nodes_parity.check(repo)
            problems += parity_problems
        if rig_skip and parity_skip:
            print(f"SKIP: {rig_skip}; {parity_skip}")
            return 0
    else:
        problems = (check_root_help(repo) + check_plan_effects(repo)
                    + check_help_and_dry_run(repo) + check_orphans(repo) + check_docs(repo)
                    + check_registry(repo) + check_alternate_plans(repo))
    for p in problems:
        print(p, file=sys.stderr)
    if not problems:
        n = len(all_verbs())
        print(f"cli: {n} command contracts, JSON plans, aliases, ownership and local documentation links pass" if what != "nodes"
              else "nodes: config/nodes.json agrees with the network document and every rig-LAN address is in __rig__")
    return 1 if problems else 0
