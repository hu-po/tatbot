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
from tatbot_paths import bootstrap

REPO = Path(__file__).resolve().parents[3]
bootstrap(REPO)
GOLDENS = REPO / "config" / "trossen"

# Named explicitly rather than pattern-matched: a reader can see exactly what
# is not being checked here, and a new golden-dependent test has to say so.
NEEDS_GOLDENS = {
    "test_repo_tatbot_yaml_matches_defaults",
    "test_carriage_constants_match_cpp_teleop",
    "test_follower_yaml_has_follower_end_effector",
    "test_apply_arm_golden_via_setters",
    "test_goldens_match_pinned_sdk_schema",
    "test_coordinated_arms_on_by_default",
}

# These read the fitted tool and touched floor from the deployment's measured
# config/workspace.yaml; the example workspace is not substituted for it.
WORKSPACE = REPO / "config" / "workspace.yaml"
NEEDS_WORKSPACE = {
    "test_workspace_floor_measures_the_tool_tip_not_the_gripper",
    "test_workspace_floor_follows_the_measured_surface",
    "test_workspace_floor_refuses_a_urdf_built_for_another_tool",
    "test_unknown_tip_link_is_caught_by_membership_not_by_a_zero",
    "test_left_floor_uses_its_own_tool_block_joints_and_receipt",
    "test_the_stated_tool_has_a_mount",
}


@pytest.fixture(autouse=True)
def private_driver_lease(tmp_path, monkeypatch):
    """Every test takes its own driver lease, never the fleet's.

    ``driver_lease.HARDWARE_LEASE`` is the flock C++ teleop and the ROS 2
    hardware plugin share, and ``acquire`` binds it as its default. A test
    that held it (the e-stop monitor and every landing take it) refused
    another xdist worker's landing tests, or a live stack on the node running
    the suite. Patching the default reaches every caller: estop, recovery's
    ``owned`` and the arm session all call ``acquire()`` bare.
    """
    from lerobot_robot_tatbot import driver_lease

    monkeypatch.setattr(driver_lease.acquire, "__defaults__", (tmp_path / "driver.lock",))


def pytest_collection_modifyitems(config, items):  # noqa: ANN001 - pytest hook
    skips = {}
    if not (GOLDENS / "tatbot.yaml").is_file():
        reason = f"needs the deployment's arm goldens ({GOLDENS.relative_to(REPO)}/)"
        skips.update(dict.fromkeys(NEEDS_GOLDENS, pytest.mark.skip(reason=reason)))
    if not WORKSPACE.is_file():
        reason = f"needs a measured touch-off ({WORKSPACE.relative_to(REPO)})"
        skips.update(dict.fromkeys(NEEDS_WORKSPACE, pytest.mark.skip(reason=reason)))
    for item in items:
        if (skip := skips.get(item.name.split("[")[0])) is not None:
            item.add_marker(skip)


def pytest_report_header(config) -> str | None:  # noqa: ANN001 - pytest hook
    if (GOLDENS / "tatbot.yaml").is_file():
        return None
    return (f"tatbot: skipping {len(NEEDS_GOLDENS)} test(s) that need the "
            "deployment's arm goldens (config/trossen/) — see config/examples/")
