#!/usr/bin/env python3
"""Known vulnerabilities in the locked and pinned dependencies, from OSV.

    scripts/deps_audit.py [--json]

Reads every lockfile and exact pin this checkout has (a public clone audits what it carries), asks
api.osv.dev about each version, and fails on any advisory that config/deps-audit-allow.tsv does not
accept. An accepted advisory names its package, a review-by date and the reason it is accepted; past that
date it fails again. RustSec's "unmaintained" notices are listed, never failed on.

Exit 0 clean, 1 findings, 2 OSV unreachable (scripts/check reports that as SKIP). Stdlib only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[1]
ALLOW = REPO / "config" / "deps-audit-allow.tsv"
OSV = "https://api.osv.dev/v1"
TOML_LOCKS = (("crates.io", "rust/Cargo.lock"), ("PyPI", "python/lerobot_robot_tatbot/uv.lock"),
              ("PyPI", "python/tatbot_sim2/uv.lock"))
NPM_LOCKS = ("web/inkmap/package-lock.json",)
PIN_GLOBS = ("docs/requirements.txt", "config/body-models/*.requirements.txt", "web/inkgen/requirements.txt*",
             "scripts/tests/requirements*.txt", "scripts/vision/requirements-*.txt")
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s;\\]+)")


def norm(ecosystem: str, name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower() if ecosystem == "PyPI" else name


def _registry_toml(text: str):
    """(name, version) of each registry package in a Cargo.lock or uv.lock; path, git and editable sources skip."""
    for pkg in tomllib.loads(text).get("package", []):
        source = pkg.get("source")
        if source and ("registry" in source if isinstance(source, dict) else source.startswith("registry+")):
            yield pkg["name"], pkg["version"]


def _npm(text: str):
    for key, pkg in json.loads(text).get("packages", {}).items():
        if key and "node_modules/" in key and "version" in pkg and not pkg.get("link"):
            yield key.rsplit("node_modules/", 1)[1], pkg["version"]


def _pins(text: str):
    for line in text.splitlines():
        if match := PIN.match(line.strip()):
            yield match[1], match[2]


def packages(repo: Path = REPO) -> dict[tuple[str, str, str], set[str]]:
    """(ecosystem, name, version) -> the files that pin it."""
    sources = [(ecosystem, repo / rel, _registry_toml) for ecosystem, rel in TOML_LOCKS]
    sources += [("npm", repo / rel, _npm) for rel in NPM_LOCKS]
    sources += [("PyPI", path, _pins) for pattern in PIN_GLOBS for path in sorted(repo.glob(pattern))]
    found: dict[tuple[str, str, str], set[str]] = {}
    for ecosystem, path, read in sources:
        if path.is_file():
            for name, version in read(path.read_text()):
                found.setdefault((ecosystem, norm(ecosystem, name), version), set()).add(str(path.relative_to(repo)))
    return found


def _post(url: str, body: dict) -> dict:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as reply:
        return json.load(reply)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as reply:
        return json.load(reply)


def advisories(found: dict) -> dict[tuple[str, str, str], list[dict]]:
    """Each package's OSV records, in full."""
    keys = sorted(found)
    ids: dict[tuple[str, str, str], list[str]] = {}
    for start in range(0, len(keys), 1000):
        chunk = keys[start:start + 1000]
        queries = [{"package": {"ecosystem": e, "name": n}, "version": v} for e, n, v in chunk]
        for key, result in zip(chunk, _post(f"{OSV}/querybatch", {"queries": queries})["results"], strict=True):
            if result.get("vulns"):
                ids[key] = [vuln["id"] for vuln in result["vulns"]]
    unique = sorted({i for found_ids in ids.values() for i in found_ids})
    with ThreadPoolExecutor(8) as pool:
        records = dict(zip(unique, pool.map(lambda i: _get(f"{OSV}/vulns/{i}"), unique), strict=True))
    return {key: [records[i] for i in found_ids] for key, found_ids in ids.items()}


def allowances(path: Path = ALLOW) -> list[dict]:
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                advisory, package, review_by, reason = line.split("\t", 3)
                rows.append({"id": advisory, "package": package, "review_by": datetime.date.fromisoformat(review_by),
                             "reason": reason})
    return rows


def severity(record: dict) -> str:
    return str((record.get("database_specific") or {}).get("severity") or "").upper() or "UNRATED"


def informational(record: dict) -> str | None:
    return next((a.get("database_specific", {}).get("informational") for a in record.get("affected", [])
                 if (a.get("database_specific") or {}).get("informational")), None)


def judge(found: dict, records: dict, allowed: list[dict], today: datetime.date) -> list[dict]:
    """One row per (package, advisory): its verdict is finding, accepted, expired or notice."""
    rows = []
    for (ecosystem, name, version), entries in sorted(records.items()):
        for record in entries:
            names = {record["id"], *record.get("aliases", [])}
            match = next((a for a in allowed if a["id"] in names and a["package"] in (name, "*")), None)
            notice = informational(record)
            verdict = ("notice" if notice == "unmaintained" else
                       "finding" if match is None else "accepted" if today <= match["review_by"] else "expired")
            rows.append({"verdict": verdict, "ecosystem": ecosystem, "package": name, "version": version,
                         "id": record["id"], "severity": notice or severity(record),
                         "summary": (record.get("summary") or record.get("details") or "").splitlines()[0][:100],
                         "files": sorted(found[(ecosystem, name, version)]),
                         "reason": match["reason"] if match else "", "review_by": str(match["review_by"]) if match else ""})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="every row as JSON")
    args = parser.parse_args(argv)
    found = packages()
    try:
        records = advisories(found)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"deps-audit: OSV unreachable ({error})", file=sys.stderr)
        return 2
    rows = judge(found, records, allowances(), datetime.date.today())
    if args.json:
        print(json.dumps(rows, indent=1))
    else:
        for row in rows:
            if row["verdict"] in ("finding", "expired"):
                note = f" (accepted until {row['review_by']})" if row["verdict"] == "expired" else ""
                print(f"{row['verdict'].upper()}: {row['ecosystem']} {row['package']} {row['version']} {row['id']} "
                      f"[{row['severity']}] {row['summary']}{note} -- {', '.join(row['files'])}")
        counts = {v: sum(row["verdict"] == v for row in rows) for v in ("finding", "expired", "accepted", "notice")}
        print(f"deps-audit: {len(found)} pinned versions; {counts['finding']} findings, {counts['expired']} expired "
              f"allowances, {counts['accepted']} accepted, {counts['notice']} unmaintained notices")
    return 1 if any(row["verdict"] in ("finding", "expired") for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
