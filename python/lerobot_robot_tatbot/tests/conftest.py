"""Tests that need a configured deployment skip where there is none.

The golden checks read the deployment's own arm goldens
(``config/trossen/leader.yaml`` etc.) — the measured EEPROM images the native
drivers apply at connect — which a public checkout does not carry. A test
that cannot run is a SKIP with a reason, never a failure; the same contract
scripts/check keeps for a node missing a toolchain.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Git exports its repository variables to hooks and `git bisect run` steps.
# With them set, a test that builds a scratch repository acts on the
# repository they name instead. No test here needs them.
GIT_LOCATION = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_PREFIX", "GIT_NAMESPACE")
for _name in GIT_LOCATION:
    os.environ.pop(_name, None)

REPO = Path(__file__).resolve().parents[3]
GOLDENS = REPO / "config" / "trossen"

# Named explicitly rather than pattern-matched: a reader can see exactly what
# is not being checked here, and a new golden-dependent test has to say so.
NEEDS_GOLDENS = {
    "test_follower_yaml_has_follower_end_effector",
    "test_goldens_match_pinned_sdk_schema",
    "test_goldens_carry_the_named_gains_the_tools_set",
}


def pytest_collection_modifyitems(config, items):  # noqa: ANN001 - pytest hook
    if (GOLDENS / "tatbot.yaml").is_file():
        return
    skip = pytest.mark.skip(reason=f"needs the deployment's arm goldens ({GOLDENS.relative_to(REPO)}/)")
    for item in items:
        if item.name.split("[")[0] in NEEDS_GOLDENS:
            item.add_marker(skip)


def pytest_report_header(config) -> str | None:  # noqa: ANN001 - pytest hook
    if (GOLDENS / "tatbot.yaml").is_file():
        return None
    return (f"tatbot: skipping {len(NEEDS_GOLDENS)} test(s) that need the "
            "deployment's arm goldens (config/trossen/) — see config/examples/")
