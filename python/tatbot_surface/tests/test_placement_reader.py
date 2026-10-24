"""The v6 placement reader and the artwork records it embeds, read with no Node bridge."""

from __future__ import annotations

import copy
import json

import pytest
from tatbot_contracts.canonical import canonical_digest
from tatbot_surface.artwork import validate_artwork_record
from tatbot_surface.contracts import (
    ContractError,
    document_sha256,
    load_placement,
    validate_placement,
)
from tatbot_surface.program_contract import validate_tattoo_program
from tatbot_surface.repo import repo_root

EXAMPLES = repo_root() / "config" / "inkmap" / "examples"
DESIGNS = repo_root() / "web" / "inkmap" / "public" / "designs"
DBV3 = ("dbv3-orbit", "dbv3-ridges", "dbv3-sprout")


def _design(name: str) -> dict:
    return json.loads((DESIGNS / name / "artwork.json").read_text())


def test_v6_placement_loads_and_old_schemas_are_generically_unsupported():
    placement = load_placement(EXAMPLES / "forearm-placement-v6.json")
    assert placement["placements"][0]["anchor"]["face"] == 8729
    for version in (1, 2, 3, 4, 5):
        old = copy.deepcopy(placement)
        old["schema_version"] = version
        with pytest.raises(ContractError, match="unsupported schema/model"):
            validate_placement(old)
    broken = copy.deepcopy(placement)
    broken["body"]["topology_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="unsupported schema/model"):
        validate_placement(broken)
    off_body = copy.deepcopy(placement)
    off_body["placements"][0]["anchor"]["face"] = 36_108
    with pytest.raises(ContractError, match=r"anchor.face must be in \[0,36108\)"):
        validate_placement(off_body)


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


def test_embedded_generated_designs_must_be_present_and_valid():
    placement = load_placement(EXAMPLES / "forearm-placement-v6.json")
    placement["placements"][0]["design_id"] = "gen-orbit"
    with pytest.raises(ContractError, match="gen-orbit is not embedded"):
        validate_placement(placement)
    placement["designs"]["gen-orbit"] = _design("dbv3-orbit")
    assert validate_placement(placement) is placement
    placement["designs"]["gen-orbit"]["program"]["layers"][0]["elements"][0]["width_m"] *= 2
    with pytest.raises(ValueError, match="wrong_hash"):
        validate_placement(placement)


@pytest.mark.parametrize("name", DBV3)
def test_acquired_designs_validate_as_records_and_programs(name):
    record = _design(name)
    copied = validate_artwork_record(record)
    assert copied == record and copied is not record
    program = record["program"]
    assert validate_tattoo_program(program) is program


def test_a_rehashed_envelope_cannot_hide_a_stale_program():
    record = _design("dbv3-orbit")
    record["program"]["layers"][0]["elements"][0]["width_m"] *= 2
    record["content_sha256"] = canonical_digest(record)
    with pytest.raises(ValueError, match="wrong_hash"):
        validate_artwork_record(record)


def test_program_and_envelope_refusals_name_their_cause():
    record = _design("dbv3-ridges")
    wrong_schema = copy.deepcopy(record)
    wrong_schema["program"]["schema"] = "tatbot.ink-program/1"
    with pytest.raises(ValueError, match="wrong_schema"):
        validate_artwork_record(wrong_schema)
    # A schema no contract reader knows is a different refusal (the TypeScript reader's code too).
    unknown_schema = copy.deepcopy(record)
    unknown_schema["program"]["schema"] = "tatbot.no-such-contract/1"
    with pytest.raises(ValueError, match="unknown_schema"):
        validate_artwork_record(unknown_schema)
    off_canvas = copy.deepcopy(record)
    off_canvas["program"]["layers"][0]["elements"][0]["points_m"][0] = [1.0, 0.0]
    off_canvas["program"]["content_sha256"] = canonical_digest(off_canvas["program"])
    with pytest.raises(ValueError, match="outside canvas_m"):
        validate_artwork_record(off_canvas)
    unnamed = copy.deepcopy(record)
    unnamed["name"] = ""
    with pytest.raises(ValueError, match="artwork_record_invalid"):
        validate_artwork_record(unnamed)
    with pytest.raises(ValueError, match="expected object"):
        validate_artwork_record(None)
