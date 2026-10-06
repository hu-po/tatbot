#!/usr/bin/env python3
"""Hold the line on Python cyclomatic complexity against a checked-in baseline.

`scripts/check lint-py` selects E4,E7,E9,F,I,N,B,C4,SIM -- deliberately no
complexity rule, because turning C901 on at ruff's default threshold would have
failed on 204 functions and been switched straight back off. So nothing pushed
back on complexity, and it grew: `main` in tatbot_sim/generate.py scored 95
and 52 functions were over 20.

This is the pushback, in the shape scripts/export_scan.py already uses for
replacement rules. config/complexity-debt.tsv records what each over-threshold
function scores today. A function that gets worse fails. A function that appears
over the threshold for the first time fails. A function that improves or goes
away is reported but does not fail -- that is the outcome this gate wants, and in
a repository landing ~77 commits a day from several agents, failing on it would
keep the gate red for work nobody did wrong. Entries are debt: they gate nothing
on their own and are expected to be deleted by simplification rather than
defended.

Keyed on path + function name, never line number, so the baseline survives edits
elsewhere in the file.

`--update` is the tightening step, not an escape hatch. It lowers and removes
entries freely and refuses to raise or add one -- a bulk "refresh against
main" once raised twenty functions (prepare 62 -> 73, dispatch 49 -> 61) and
added sixteen in a single commit, which is the growth this file exists to
stop. Accepting a specific function as irreducible is a per-function decision:
`--update --accept path.py::name`, with the reason in the commit message.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "config" / "complexity-debt.tsv"
THRESHOLD = 10  # ruff's C901 default; new code is held to it
MESSAGE = re.compile(r"`(.+?)` is too complex \((\d+) > (\d+)\)")

HEADER = (
    "# Functions over the C901 complexity threshold, with the score each has today.\n"
    "# Regenerate with `scripts/complexity_budget.py --update`; see that file's\n"
    "# docstring. An entry is debt to delete by simplifying, not a permanent\n"
    "# exemption -- the number may only go down.\n"
    "#\n"
    "# path\tfunction\tcomplexity\n"
)


def _ruff_cmd() -> list[str]:
    """Prefer a ruff on PATH; fall back to uvx, as scripts/check does."""
    if shutil.which("ruff"):
        return ["ruff"]
    if shutil.which("uvx"):
        return ["uvx", "ruff"]
    return []


def measure(paths: list[str]) -> dict[tuple[str, str], int]:
    ruff = _ruff_cmd()
    if not ruff:
        raise SystemExit("neither ruff nor uvx is installed")
    proc = subprocess.run(
        [*ruff, "check", "--no-fix", "--select", "C901",
         "--config", f"lint.mccabe.max-complexity={THRESHOLD}",
         "--output-format", "json", *paths],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode not in (0, 1):
        raise SystemExit(f"ruff failed ({proc.returncode}):\n{proc.stderr.strip()}")
    found: dict[tuple[str, str], int] = {}
    for item in json.loads(proc.stdout or "[]"):
        match = MESSAGE.match(item["message"])
        if not match:  # message shape changed; fail loudly rather than silently pass
            raise SystemExit(f"unparsed C901 message: {item['message']!r}")
        rel = os.path.relpath(item["filename"], REPO)
        found[(rel, match.group(1))] = int(match.group(2))
    return found


def _shown(path: Path) -> str:
    """Repo-relative when it can be, the plain path otherwise."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def load_baseline() -> dict[tuple[str, str], int]:
    if not BASELINE.is_file():
        return {}
    out: dict[tuple[str, str], int] = {}
    for line in BASELINE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        path, name, score = line.split("\t")
        out[(path, name)] = int(score)
    return out


def write_baseline(found: dict[tuple[str, str], int]) -> None:
    rows = "".join(f"{p}\t{n}\t{c}\n" for (p, n), c in sorted(found.items()))
    BASELINE.write_text(HEADER + rows)


# internal/** is private-only in internal/export/manifest.json and never ships.
# Two of its functions are over the threshold, and naming their paths in a public
# baseline would publish a node name (internal/ci/<node>/...) -- the export gate
# refuses exactly that. Skipping the tree keeps config/complexity-debt.tsv
# public-safe by construction, and keeps it identical in the private repo and the
# export, so the ratchet still holds there instead of reporting every function as
# new against an absent baseline.
EXCLUDED_PREFIX = "internal/"


def tracked_python() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO,
                         capture_output=True, text=True, check=True)
    return [path for path in out.stdout.split() if not path.startswith(EXCLUDED_PREFIX)]


def compare(found: dict[tuple[str, str], int],
            base: dict[tuple[str, str], int]) -> tuple[list, list, list, list]:
    """Split the measurement against the baseline into the four transitions."""

    shared = found.keys() & base.keys()
    return (
        sorted((k, base[k], found[k]) for k in shared if found[k] > base[k]),   # worse
        sorted(found.keys() - base.keys()),                                     # fresh
        sorted((k, base[k], found[k]) for k in shared if found[k] < base[k]),   # better
        sorted(base.keys() - found.keys()),                                     # settled
    )


def report(found, worse, fresh, better, settled) -> None:
    for (path, name), was, now in worse:
        print(f"WORSE  {path}::{name} {was} -> {now}")
    for path, name in fresh:
        print(f"NEW    {path}::{name} {found[(path, name)]} > {THRESHOLD}")
    for (path, name), was, now in better:
        print(f"BETTER {path}::{name} {was} -> {now}")
    for (path, name), was in ((k, v) for k, v in settled):
        print(f"GONE   {path}::{name} (was {was})")


def update(found: dict[tuple[str, str], int], base: dict[tuple[str, str], int],
           accepted: list[str], parser) -> int:
    """Rewrite the baseline, refusing any entry that would go up or appear."""
    keys = {f"{path}::{name}": (path, name) for path, name in found}
    unknown = [a for a in accepted if a not in keys]
    if unknown:
        parser.error("--accept names a function that is not over the threshold: " + ", ".join(unknown))
    allowed = {keys[a] for a in accepted}
    worse, fresh, _better, _settled = compare(found, base)
    worse = [row for row in worse if row[0] not in allowed]
    fresh = [key for key in fresh if key not in allowed]
    if (worse or fresh) and BASELINE.is_file():
        report(found, worse, fresh, [], [])
        flags = "".join(f" \\\n    --accept {p}::{n}" for p, n in [k for k, _w, _n in worse] + fresh)
        print(f"\n--update only lowers or removes entries; {len(worse) + len(fresh)} function(s) "
              "would be recorded higher than the baseline or added to it. Simplify them, or "
              "accept each one by name and say why in the commit message:\n"
              f"  scripts/complexity_budget.py --update{flags}")
        return 1
    write_baseline(found)
    print(f"{_shown(BASELINE)}: {len(found)} entries over {THRESHOLD}"
          + (f", {len(allowed)} accepted by name" if allowed else ""))
    return 0


def resolve_paths(args, parser) -> tuple[list[str], bool]:
    scoped = bool(args.paths)
    if scoped and args.update:
        parser.error("--update rewrites the whole baseline; do not scope it to paths")
    if not scoped:
        return tracked_python(), False
    return [p for p in args.paths if not p.startswith(EXCLUDED_PREFIX)], True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--update", action="store_true",
                        help="rewrite the baseline from the current tree; only lowers or "
                             "removes entries unless each raise is named with --accept")
    parser.add_argument("--accept", action="append", default=[], metavar="PATH::FUNCTION",
                        help="with --update: record this function at its current score even "
                             "though that is higher than the baseline (or new); repeatable")
    parser.add_argument("paths", nargs="*",
                        help="only these files (pre-commit passes the staged ones); "
                             "default is every tracked Python file")
    args = parser.parse_args(argv)

    paths, scoped = resolve_paths(args, parser)
    if not paths:
        print("no tracked Python in scope")
        return 0
    found = measure(paths)

    base = load_baseline()
    if args.update:
        return update(found, base, args.accept, parser)

    if scoped:
        # Only these files were measured, so a baseline entry for any other file
        # is unseen, not gone. This is the pre-commit path: in a shared checkout
        # the tree carries other agents' in-flight edits, and a whole-tree ratchet
        # would fail their commit over a file the committer never touched.
        seen = set(paths)
        base = {key: value for key, value in base.items() if key[0] in seen}

    worse, fresh, better, settled = compare(found, base)
    report(found, worse, fresh, better, [(k, base[k]) for k in settled])

    if worse or fresh:
        print("\nComplexity grew. Split the function, or -- if the shape is genuinely "
              "irreducible -- run `scripts/complexity_budget.py --update --accept "
              "path.py::function` in the same commit and say why in the message.")
        return 1
    if better or settled:
        # Improvement is the outcome this gate exists to produce, so it does not
        # fail. This repository lands ~77 commits a day from several agents: a
        # gate that went red every time a function got simpler or a file was
        # deleted would be red almost always, and a gate people expect to be red
        # stops being read. A stale-high baseline still refuses a new function
        # over the threshold and still refuses a recorded one getting worse,
        # which is the whole ratchet. Tighten it when convenient.
        print(f"\n{len(better) + len(settled)} function(s) improved or went away. "
              f"Run `scripts/complexity_budget.py --update` to tighten the baseline.")
        return 0
    scope = f"across {len(paths)} file(s) in scope" if scoped else \
            f"{len(found)} function(s) over {THRESHOLD}, none worse"
    print(f"complexity held: {scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
