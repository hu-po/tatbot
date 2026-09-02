from __future__ import annotations

import copy

import pytest
from tatbot_sim.inkmap.contracts import (
    ContractError,
    document_sha256,
    load_placement,
    load_scenario,
    validate_placement,
    validate_scenario,
)
from tatbot_sim.repo import repo_root

EXAMPLES = repo_root() / "config" / "inkmap" / "examples"


def test_shared_v4_placement_and_v1_scenario_load():
    placement = load_placement(EXAMPLES / "forearm-placement-v4.json")
    scenario = load_scenario(EXAMPLES / "forearm-scenario-v1.json")
    assert placement["body"]["surface_sha256"] == scenario["body"]["surface_sha256"]
    assert scenario["placement"]["id"] == placement["placements"][0]["id"]


def test_old_whole_asset_placement_remains_readable():
    current = load_placement(EXAMPLES / "forearm-placement-v4.json")
    old = copy.deepcopy(current)
    old["schema_version"] = 3
    old["body"] = {
        "id": current["body"]["id"],
        "path": current["body"]["path"],
        "sha256": current["body"]["asset_sha256"],
    }
    assert validate_placement(old) is old


def test_contracts_fail_closed_on_geometry_and_frame_ambiguity():
    placement = load_placement(EXAMPLES / "forearm-placement-v4.json")
    broken = copy.deepcopy(placement)
    broken["body"].pop("surface_sha256")
    with pytest.raises(ContractError, match="surface_sha256"):
        validate_placement(broken)

    scenario = load_scenario(EXAMPLES / "forearm-scenario-v1.json")
    broken_scenario = copy.deepcopy(scenario)
    broken_scenario["pose"]["world_from_body"] = [[1, 0], [0, 1]]
    with pytest.raises(ContractError, match="4x4"):
        validate_scenario(broken_scenario)


def test_document_digest_is_key_order_independent():
    left = {"b": [2, 3], "a": 1}
    right = {"a": 1, "b": [2, 3]}
    assert document_sha256(left) == document_sha256(right)
