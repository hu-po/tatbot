#!/usr/bin/env python3
"""The ROS 2 stack's size budget (ros/README.md section 1): <=20k lines of code, <=8 custom interfaces,
<=3 launch files. `scripts/check ros-budget` runs it; stdlib only.

Lines of code: non-blank lines that are not only a comment, in tracked (and new, unignored) ros/
C++, Python, xacro and CMake sources, outside test/ directories. Configuration, IDL and fixtures
are not code.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_LINES = 20_000
MAX_INTERFACES = 8
MAX_LAUNCH_FILES = 3
CODE_SUFFIXES = {".py": "#", ".cpp": "//", ".hpp": "//", ".h": "//", ".cc": "//", ".xacro": "<!--", ".cmake": "#"}


def files(repo: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "ros"],
                         cwd=repo, capture_output=True, text=True, check=True).stdout
    return [repo / rel for rel in out.splitlines() if (repo / rel).is_file()]


def comment_prefix(path: Path) -> str | None:
    if path.name == "CMakeLists.txt":
        return "#"
    return CODE_SUFFIXES.get(path.suffix)


def count(repo: Path) -> dict:
    lines, interfaces, launches = 0, [], []
    for path in files(repo):
        rel = path.relative_to(repo)
        if rel.parts[1:2] == ("tatbot_interfaces",) and rel.parent.name in ("msg", "srv", "action"):
            interfaces.append(str(rel))
        if "launch" in rel.parts[:-1] or ".launch." in path.name:
            launches.append(str(rel))
        prefix = comment_prefix(path)
        if prefix is None or "test" in rel.parts:
            continue
        for line in path.read_text(errors="replace").splitlines():
            text = line.strip()
            if text and not text.startswith(prefix) and not text.startswith(("*", "/*", "\"\"\"")):
                lines += 1
    return {"lines": lines, "interfaces": sorted(interfaces), "launch_files": sorted(launches)}


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    result = count(repo)
    problems = []
    if result["lines"] > MAX_LINES:
        problems.append(f"ros/: {result['lines']} lines of code, over the {MAX_LINES} budget")
    if len(result["interfaces"]) > MAX_INTERFACES:
        problems.append(f"ros/: {len(result['interfaces'])} custom interfaces, over {MAX_INTERFACES}: {result['interfaces']}")
    if len(result["launch_files"]) > MAX_LAUNCH_FILES:
        problems.append(f"ros/: {len(result['launch_files'])} launch files, over {MAX_LAUNCH_FILES}: {result['launch_files']}")
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"ros-budget: {result['lines']}/{MAX_LINES} lines, {len(result['interfaces'])}/{MAX_INTERFACES} interfaces, "
          f"{len(result['launch_files'])}/{MAX_LAUNCH_FILES} launch files")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
