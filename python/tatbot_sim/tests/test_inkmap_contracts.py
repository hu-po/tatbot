from __future__ import annotations

import copy
import json

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


def test_shared_v6_placement_and_v2_scenario_load():
    placement = load_placement(EXAMPLES / "forearm-placement-v6.json")
    scenario = load_scenario(EXAMPLES / "forearm-scenario-v2.json")
    assert placement["body"]["rest_surface_sha256"] == scenario["body"]["rest_surface_sha256"]
    assert placement["body"]["topology_sha256"] == scenario["body"]["topology_sha256"]
    assert scenario["placement"]["id"] == placement["placements"][0]["id"]


def test_old_placement_and_scenario_schemas_are_generically_unsupported():
    placement = load_placement(EXAMPLES / "forearm-placement-v6.json")
    for version in (1, 2, 3, 4, 5):
        old = copy.deepcopy(placement)
        old["schema_version"] = version
        with pytest.raises(ContractError, match="unsupported schema/model"):
            validate_placement(old)
    scenario = load_scenario(EXAMPLES / "forearm-scenario-v2.json")
    scenario["schema_version"] = 1
    with pytest.raises(ContractError, match="schema_version must be 2"):
        validate_scenario(scenario)


def test_contracts_fail_closed_on_body_geometry_and_frame_ambiguity():
    placement = load_placement(EXAMPLES / "forearm-placement-v6.json")
    broken = copy.deepcopy(placement)
    broken["body"]["topology_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="unsupported schema/model"):
        validate_placement(broken)

    scenario = load_scenario(EXAMPLES / "forearm-scenario-v2.json")
    broken_scenario = copy.deepcopy(scenario)
    broken_scenario["pose"]["world_from_body"] = [[1, 0], [0, 1]]
    with pytest.raises(ContractError, match="4x4"):
        validate_scenario(broken_scenario)

    malformed_search = copy.deepcopy(scenario)
    malformed_search["placement_optimization"] = {
        "schema": "tatbot.placement-search/1", "selected": {}, "candidates": [{}],
    }
    with pytest.raises(ContractError, match="world_from_body"):
        validate_scenario(malformed_search)


def test_document_digest_is_key_order_independent():
    left = {"b": [2, 3], "a": 1}
    right = {"a": 1, "b": [2, 3]}
    assert document_sha256(left) == document_sha256(right)


def test_placement_resolution_provenance_must_match_anchor_and_body():
    placement = load_placement(EXAMPLES / "forearm-placement-v6.json")
    resolved = copy.deepcopy(placement)
    item = resolved["placements"][0]
    corpus = json.loads((EXAMPLES / "inklang" / "corpus-v1.json").read_text())
    resolution = next(
        case["expected"] for case in corpus["cases"]
        if case["id"] == "mhr-soma-v1:leaf:left:forearm"
    )
    item["anchor"] = copy.deepcopy(resolution["anchor"])
    item["language"] = {
        "sentence": "a line on the left forearm",
        "program": {},
        "resolution": copy.deepcopy(resolution),
    }
    assert validate_placement(resolved) is resolved
    item["language"]["resolution"]["anchor"]["face"] += 1
    with pytest.raises(ContractError, match="resolution is inconsistent"):
        validate_placement(resolved)
