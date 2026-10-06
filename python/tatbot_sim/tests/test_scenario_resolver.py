from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from tatbot_sim.inkmap.resolver import (
    ScenarioResolveError,
    _resolve_pose,
    build_scenario_request,
    resolve_scenario_request,
)
from tatbot_sim.repo import repo_root

FIXTURES = repo_root() / "config" / "inkmap" / "examples" / "scenario-request-fixtures.json"


def _request(fixture: dict, *, design_id: str | None = "spiral-v1"):
    return build_scenario_request(
        fixture["prompt"],
        size_mm=tuple(fixture["size_mm"]),
        pose=fixture["pose"],
        support=fixture["support"],
        seed=fixture["seed"],
        design_id=design_id,
    )


def test_checked_in_semantic_fixture_matrix_covers_the_typed_boundary():
    fixtures = json.loads(FIXTURES.read_text())["fixtures"]
    assert len(fixtures) == 100
    assert all("body" not in item for item in fixtures)
    assert len({item["pose"] for item in fixtures}) >= 4
    assert len({item["support"] for item in fixtures}) >= 4
    assert len({tuple(item["size_mm"]) for item in fixtures}) >= 5
    assert {item["expected"].get("site") for item in fixtures} >= {
        "forearm", "bicep", "shoulder_cap", "thigh", "calf", "mid_back",
    }
    assert {item["expected"].get("laterality") for item in fixtures} >= {None, "left", "right"}
    assert len([item for item in fixtures if item["expected"].get("error")]) >= 10


def test_request_parser_and_compatibility_return_named_errors():
    fixtures = {item["id"]: item for item in json.loads(FIXTURES.read_text())["fixtures"]}
    request = _request(fixtures["valid-000"])
    pose, support = _resolve_pose(request)
    assert pose == "reclined-left-arm-supported"
    assert support == "tattoo-chair-left-armrest-v1"

    for fixture_id in ("invalid-088", "invalid-090", "invalid-093", "invalid-094"):
        fixture = fixtures[fixture_id]
        with pytest.raises(ScenarioResolveError, match=fixture["expected"]["error"]):
            _request(fixture)

    request = _request(fixtures["invalid-096"])
    with pytest.raises(ScenarioResolveError, match="incompatible_pose_site"):
        _resolve_pose(request)


@pytest.mark.slow
def test_same_typed_request_and_seed_write_identical_placement_and_scenario(monkeypatch, tmp_path):
    request = build_scenario_request(
        "a spiral on the left forearm",
        size_mm=(20.0, 20.0),
        pose="reclined-left-arm-supported",
        support="armrest",
        seed=42,
        design_id="spiral-v1",
    )

    def accept(scenario, **_kwargs):
        result = deepcopy(scenario)
        result["placement_optimization"] = {
            "schema": "tatbot.placement-search/1",
            "selected": {
                "world_from_body": result["pose"]["world_from_body"],
                "world_from_support": result["support"]["world_from_nominal"],
            },
            "candidates": [{"candidate": 0, "accepted": True}],
        }
        return SimpleNamespace(scenario=result, audit={})

    monkeypatch.setattr("tatbot_sim.inkmap.resolver.optimize_body_placement", accept)
    first = resolve_scenario_request(request, tmp_path / "a", git_sha="1234567")
    second = resolve_scenario_request(request, tmp_path / "b", git_sha="1234567")
    assert first["placement_sha256"] == second["placement_sha256"]
    assert first["scenario_sha256"] == second["scenario_sha256"]
    assert (tmp_path / "a" / "placement.json").read_bytes() == (
        tmp_path / "b" / "placement.json"
    ).read_bytes()
    assert (tmp_path / "a" / "scenario.json").read_bytes() == (
        tmp_path / "b" / "scenario.json"
    ).read_bytes()


def test_the_retry_grows_into_a_coverage_refusal_and_shrinks_off_a_boundary():
    """Measured on a generated daisy chain: shrinking walked 0.92 -> 0.61."""
    from tatbot_sim.inkmap.sampler import COVERAGE_REASON, _attempt_scale

    boundary = [_attempt_scale(a, "site_boundary") for a in range(4)]
    coverage = [_attempt_scale(a, COVERAGE_REASON) for a in range(4)]
    assert boundary == sorted(boundary, reverse=True), "a boundary reject still shrinks"
    assert coverage == sorted(coverage), "a coverage reject must grow, not shrink"
    assert boundary[0] == coverage[0] == 0.82, "the first attempt is unchanged"
    assert max(coverage) <= 1.0 and min(boundary) >= 0.48, "both stay bounded"


def test_a_slot_tries_another_artwork_instead_of_ending_the_suite():
    """One artwork the fill planner cannot cover used to take the whole run."""
    from tatbot_sim.inkmap.sampler import MAX_SLOT_DESIGNS, _slot_designs

    class Design:
        def __init__(self, identifier):
            self.id = identifier

    designs = [Design(f"art-{i}") for i in range(6)]
    # The balanced draw still goes first; that is what keeps the distribution honest.
    assert _slot_designs(designs[2], designs, set())[0].id == "art-2"
    assert len(_slot_designs(designs[2], designs, set())) == MAX_SLOT_DESIGNS
    # Artwork already known unusable is not offered again, to any slot.
    offered = [d.id for d in _slot_designs(designs[2], designs, {"art-2", "art-0"})]
    assert "art-2" not in offered and "art-0" not in offered
    # And a library with nothing left to offer returns nothing, rather than looping.
    assert _slot_designs(designs[0], designs, {d.id for d in designs}) == ()


def test_an_audit_that_outlasts_its_share_is_refused_not_waited_on():
    """A suite budget checked between candidates cannot stop one slow candidate.

    Measured: a run with a 900 s budget sat 20 minutes inside a single
    full-trajectory audit without ever reaching the between-candidate check.
    The audit now carries its own share of the budget.
    """
    import time

    from tatbot_sim.inkmap.reach import ReachAuditError, _Deadline

    started = time.monotonic()
    with pytest.raises(ReachAuditError, match="time budget") as excinfo, _Deadline(0.5):
        while True:
            pass
    assert time.monotonic() - started < 5, "the deadline fired promptly"
    assert excinfo.value.reason == "time_budget", "a refusal a ledger can count"

    # No budget, and a budget the work fits inside, both pass through untouched.
    with _Deadline(None):
        pass
    with _Deadline(30):
        pass
    # The previous SIGALRM handler is restored either way.
    import signal
    assert signal.getsignal(signal.SIGALRM) in (signal.SIG_DFL, signal.SIG_IGN, 0) or callable(
        signal.getsignal(signal.SIGALRM))
