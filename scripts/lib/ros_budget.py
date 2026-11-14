#!/usr/bin/env python3
"""The ROS 2 stack's structural limits (ros/README.md section 1): MAX_INTERFACES custom interfaces and
MAX_LAUNCH_FILES launch files, and no joint named by its number outside the tests. Lines of code are not capped.
`scripts/check ros-budget` runs it; stdlib only.

A change to a limit edits it here (and ros/README.md), never scripts/check: that file is private CI's installed
policy, so a change to it stops CI validating main until `tatbot ci install` runs again.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MAX_INTERFACES = 8
MAX_LAUNCH_FILES = 3
# Which joint is which comes from the arm's model (scripts/lib/arm_model.py), never from its number. These idioms of
# the WXAI's layout (the revolute chain as [:6], the carriage at 6, joint names by number) would run on the xArm's
# seven revolute joints without an error and misread them. A line indexing anything else says so with NOT_JOINTS.
JOINT_INDEX = re.compile(r"\[:6\]|\[:, *:6\]|\[6\]|\bkCarriage\b|\bkWristRoll\b|\bfor [a-z]\w* in range\([67]\)|joint_\{|"
                         r"/joint_\d|left_carriage_joint")
NOT_JOINTS = "arm-model: not joints"


def files(repo: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "ros"],
                         cwd=repo, capture_output=True, text=True, check=True).stdout
    return [repo / rel for rel in out.splitlines() if (repo / rel).is_file()]


def joint_indices(text: str) -> list[int]:
    """The lines (1-based) of a source that index a joint by its number."""
    return [n for n, line in enumerate(text.splitlines(), 1) if JOINT_INDEX.search(line) and NOT_JOINTS not in line]


def count(repo: Path) -> dict:
    interfaces, launches, joints = [], [], []
    for path in files(repo):
        rel = path.relative_to(repo)
        if path.suffix in (".py", ".cpp", ".hpp") and "test" not in rel.parts:
            joints += [f"{rel}:{n}" for n in joint_indices(path.read_text())]
        if rel.parts[1:2] == ("tatbot_interfaces",) and rel.parent.name in ("msg", "srv", "action"):
            interfaces.append(str(rel))
        if "launch" in rel.parts[:-1] or ".launch." in path.name:
            launches.append(str(rel))
    return {"interfaces": sorted(interfaces), "launch_files": sorted(launches), "joint_indices": joints}


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    result = count(repo)
    problems = []
    if len(result["interfaces"]) > MAX_INTERFACES:
        problems.append(f"ros/: {len(result['interfaces'])} custom interfaces, over {MAX_INTERFACES}: {result['interfaces']}")
    if len(result["launch_files"]) > MAX_LAUNCH_FILES:
        problems.append(f"ros/: {len(result['launch_files'])} launch files, over {MAX_LAUNCH_FILES}: {result['launch_files']}")
    if result["joint_indices"]:
        problems.append(f"ros/ names joints by number; take them from the arm model, or mark a line that is no joint "
                        f"with '{NOT_JOINTS}': {', '.join(result['joint_indices'])}")
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"ros-budget: {len(result['interfaces'])}/{MAX_INTERFACES} interfaces, "
          f"{len(result['launch_files'])}/{MAX_LAUNCH_FILES} launch files")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
