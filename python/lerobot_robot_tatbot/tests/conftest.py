"""Tests that need a configured deployment skip where there is none.

Most of this suite runs anywhere: the e-stop protocol, motion safety, the
tool registry, the mock driver. A subset reads the deployment's own arm
goldens (``config/trossen/leader.yaml`` etc.) — the measured EEPROM images
this rig applies at connect — which a public checkout does not carry. A test
that cannot run is a SKIP with a reason, never a failure; the same contract
scripts/check keeps for a node missing a toolchain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GOLDENS = REPO / "config" / "trossen"

# Named explicitly rather than pattern-matched: a reader can see exactly what
# is not being checked here, and a new golden-dependent test has to say so.
NEEDS_GOLDENS = {
    "test_arm_golden_roundtrip",
    "test_leader_golden_roundtrip",
    "test_repo_tatbot_yaml_matches_defaults",
    "test_carriage_constants_match_cpp_teleop",
    "test_follower_yaml_has_follower_end_effector",
    "test_apply_arm_golden_via_setters",
    "test_register_survives_dead_link",
    "test_goldens_match_pinned_sdk_schema",
    "test_coordinated_arms_on_by_default",
    "test_http_registry",
    "test_http_set_save_revert",
    "test_http_per_joint_and_errors",
    "test_http_capture_pose",
    "test_http_cockpit_page",
}


def pytest_collection_modifyitems(config, items):  # noqa: ANN001 - pytest hook
    if (GOLDENS / "tatbot.yaml").is_file():
        return
    skip = pytest.mark.skip(
        reason=f"needs the deployment's arm goldens ({GOLDENS.relative_to(REPO)}/)")
    for item in items:
        if item.name.split("[")[0] in NEEDS_GOLDENS:
            item.add_marker(skip)


def pytest_report_header(config) -> str | None:  # noqa: ANN001 - pytest hook
    if (GOLDENS / "tatbot.yaml").is_file():
        return None
    return (f"tatbot: skipping {len(NEEDS_GOLDENS)} test(s) that need the "
            "deployment's arm goldens (config/trossen/) — see config/examples/")
