from __future__ import annotations

from copy import deepcopy

import pytest
from tatbot_sim.inkmap.inklang_client import (
    InkLangConsumerError,
    parse_tattoo_request,
    resolve_placement,
    validate_resolution,
)


def test_legacy_tattoo_request_adapter_returns_the_canonical_placement_intent():
    parsed = parse_tattoo_request("a fine line spiral on the left upper inner forearm")
    assert parsed["program"]["motif"] == "spiral"
    assert parsed["program"]["style"] == "fine-line"
    assert parsed["placement_intent"]["canonical_phrase"] == "on the left upper inner forearm"
    assert parsed["placement_intent"]["site"] == {
        "id": "forearm", "laterality": "left", "aspect": "inner", "level": "upper",
    }


def test_python_consumes_the_exact_typescript_resolution_json():
    resolution = resolve_placement({
        "prompt": "left forearm",
        "policy": "seeded-v1",
        "seed": 41,
    })
    assert resolution["status"] == "resolved"
    assert resolution["resolver"]["name"] == "inklang-typescript"
    assert resolution["resolver"]["policy"] == {"id": "seeded-v1", "seed": 41}
    assert resolution["actual"]["site_id"] == "forearm"


def test_consumer_fails_closed_on_surface_and_anchor_tampering():
    resolution = resolve_placement({
        "prompt": "right outer forearm",
        "policy": "seeded-v1",
        "seed": 9,
    })
    broken = deepcopy(resolution)
    broken["body"]["rest_surface_sha256"] = "0" * 64
    with pytest.raises(InkLangConsumerError, match="INKLANG_SURFACE_MISMATCH"):
        validate_resolution(broken)
    broken = deepcopy(resolution)
    broken["anchor"]["face"] = 10**9
    with pytest.raises(InkLangConsumerError, match="INKLANG_ANCHOR_INVALID"):
        validate_resolution(broken)


def test_old_body_selection_is_not_a_supported_request_field():
    with pytest.raises(InkLangConsumerError, match="unsupported schema/model"):
        resolve_placement({"body": "legacy-body", "prompt": "left forearm"})


def test_missing_node_is_a_named_blocker_not_a_second_parser(monkeypatch):
    monkeypatch.setattr("tatbot_sim.inkmap.inklang_client.shutil.which", lambda _name: None)
    with pytest.raises(InkLangConsumerError) as failure:
        parse_tattoo_request("a spiral on the left forearm")
    assert failure.value.code == "INKLANG_RUNTIME_UNAVAILABLE"
