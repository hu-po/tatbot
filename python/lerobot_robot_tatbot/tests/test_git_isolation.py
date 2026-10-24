"""Git's repository variables never reach a test here (the conftest drops them).

A hook or a `git bisect run` step exports GIT_DIR and the like; with them set,
a test that builds a scratch repository would act on the repository they name.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LOCATION = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES")


def test_this_session_carries_none_of_gits_repository_variables():
    assert [name for name in LOCATION if name in os.environ] == []


def test_a_session_started_by_git_drops_its_repository_variables(tmp_path):
    decoy = tmp_path / "decoy"
    subprocess.run(["git", "init", "-q", str(decoy)], check=True)
    env = {**os.environ, "GIT_DIR": str(decoy / ".git"), "GIT_WORK_TREE": str(decoy),
           "GIT_INDEX_FILE": str(decoy / ".git/index"), "GIT_COMMON_DIR": str(decoy / ".git")}
    probe = f"{Path(__file__).resolve()}::test_this_session_carries_none_of_gits_repository_variables"
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", probe],
                         cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0 and "1 passed" in out.stdout, out.stdout + out.stderr
