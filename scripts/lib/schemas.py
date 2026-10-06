"""The artifact schema table both languages read.

`config/schemas.json` is the source and this module is its reader. Registered families carry
each artifact as `{"schema": <family>, "kind": <kind>, ...}`. The `legacy`
table maps each name written before the fold to its `[family, kind]`, so a run
directory recorded under an old name still opens: `is_schema` accepts a legacy
row exactly as it accepts the pair. Nothing here compares a date; the table is
emptied by a commit when the operator says old run directories need not open.

`--check` is the `scripts/check schemas` ratchet: every `tatbot.<x>/<n>`
literal in the scanned sources (either quote, test modules excluded) is a
family, a bus name, an external contract, a program version, or a `pending`
row naming the file and the lane whose file it is; a pending row nothing
writes any more is refused, so the table only shrinks.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "config" / "schemas.json"

JOB = "tatbot.session-job/4"
PROGRAM = "tatbot.session-program/2"
SURFACE = "tatbot.session-surface/1"
EVENT = "tatbot.session-event/1"
RECEIPT = "tatbot.receipt/1"

# The scan set: the fleet, stencil and stroke sources, never their test modules.
SCAN_DIRS = ("rust/fleetctl/src", "rust/tatbot-arm/src")
SCAN_GLOBS = ("scripts/lib/fleet_*.py", "scripts/lib/stroke_*.py",
              *(f"scripts/lib/{name}.py" for name in (
                  "stencil_origin", "stencil_outline", "stencil_scan", "stencil_state", "candidate_plan",
                  "live_inputs", "schemas", "stencils", "surface_rgb",
                  "tracking_rgb", "view_assets")))
LITERAL = re.compile(r"""["'](tatbot\.[a-z0-9-]+/[0-9]+)["']""")


def load(path: Path = PATH) -> dict:
    return json.loads(path.read_text())


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        _TABLE = load()
    return _TABLE


_TABLE = None


def resolve(schema, kind=None):
    """The `(family, kind)` a document's names resolve to, or None."""
    t = table()
    family = t["families"].get(schema)
    if family is not None:
        kind = kind if kind is not None else family.get("default_kind")
        return (schema, kind) if kind in t["kinds"].get(schema, []) else None
    row = t["legacy"].get(schema)
    return tuple(row) if row else None


def is_schema(document, family, kind):
    """The document is `{schema: family, kind}` or a legacy name mapping to it."""
    if not isinstance(document, dict) or not isinstance(document.get("schema"), str):
        return False
    kind_field = document.get("kind")
    return resolve(document["schema"], kind_field if isinstance(kind_field, str) else None) == (family, kind)


def stamp(family, kind):
    """`{"schema": family, "kind": kind}` to extend with the document's fields."""
    return {"schema": family, "kind": kind}


def _rust_test_lines(lines):
    """Line indexes inside `#[cfg(test)] mod name { ... }` blocks."""
    inside, i = set(), 0
    while i < len(lines):
        if lines[i].strip() == "#[cfg(test)]":
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("#["):
                j += 1
            if j < len(lines) and re.match(r"\s*(pub(\(crate\))?\s+)?mod\s+\w+\s*\{", lines[j]):
                depth, k = 0, j
                while k < len(lines):
                    depth += lines[k].count("{") - lines[k].count("}")
                    inside.add(k)
                    if depth <= 0:
                        break
                    k += 1
                i = k + 1
                continue
        i += 1
    return inside


def scan_files(root: Path):
    files = []
    for directory in SCAN_DIRS:
        files += [p for p in sorted((root / directory).rglob("*.rs"))
                  if not p.name.endswith("_tests.rs") and "tests" not in p.relative_to(root / directory).parts]
    for pattern in SCAN_GLOBS:
        files += sorted(root.glob(pattern))
    return files


def census(root: Path = ROOT):
    """`{name: {relative file, ...}}` for every literal outside a test module."""
    found = {}
    for path in scan_files(root):
        lines = path.read_text().splitlines()
        skip = _rust_test_lines(lines) if path.suffix == ".rs" else set()
        relative = path.relative_to(root).as_posix()
        for index, line in enumerate(lines):
            if index in skip:
                continue
            for match in LITERAL.finditer(line):
                found.setdefault(match.group(1), set()).add(relative)
    return found


def _unlisted(found, t):
    """Literals outside the table, each naming the file and the fix."""
    known = set(t["families"]) | set(t["bus"]) | set(t["external"]) | set(t["program_versions"])
    for name in sorted(found):
        if name in known:
            continue
        sites = t["pending"].get(name, {}).get("sites", {})
        for file in sorted(found[name] - set(sites)):
            yield (f"{file} writes {name}, which is not a family, bus name, external contract or pending site"
                   f" there: stamp it {{schema, kind}} through schemas,"
                   f" or list the file under pending[{name!r}].sites with its lane")


def _stale_rows(found, t):
    """Pending rows that no longer describe the tree: the ratchet's downward step."""
    for name, row in sorted(t["pending"].items()):
        if row.get("disposition") not in ("rename", "delete"):
            yield f"pending[{name!r}] needs a disposition of rename or delete"
        if row.get("disposition") == "rename" and name not in t["legacy"]:
            yield f"pending[{name!r}] is a rename without a legacy row"
        for file, lane in sorted(row["sites"].items()):
            if not lane:
                yield f"pending[{name!r}].sites[{file!r}] names no lane"
            if file not in found.get(name, ()):
                yield (f"pending[{name!r}].sites[{file!r}] is stale: nothing there writes it any more;"
                       f" remove the row so the table only shrinks")


def check(root: Path = ROOT, path: Path | None = None):
    """Problems the ratchet found; empty when the tree matches the table."""
    t = load(path or root / "config" / "schemas.json")
    found = census(root)
    problems = list(_unlisted(found, t)) + list(_stale_rows(found, t))
    for name, (family, kind) in sorted(t["legacy"].items()):
        if kind not in t["kinds"].get(family, []):
            problems.append(f"legacy[{name!r}] maps to {family} kind {kind}, which the kinds table lacks")
    return problems


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["--check"]:
        sys.exit("usage: schemas.py --check [ROOT]")
    root = Path(argv[1]).resolve() if len(argv) > 1 else ROOT
    problems = check(root)
    if problems:
        print("schemas: " + "\n  ".join(problems), file=sys.stderr)
        return 1
    t = load(root / "config" / "schemas.json")
    sites = sum(len(row["sites"]) for row in t["pending"].values())
    print(f"schemas: every literal is in the table; {len(t['pending'])} names at {sites} sites still pending")
    return 0


if __name__ == "__main__":
    sys.exit(main())
