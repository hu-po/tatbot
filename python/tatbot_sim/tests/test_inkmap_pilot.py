from __future__ import annotations

import copy
import hashlib
import json

import pytest
from tatbot_sim.inkmap.pilot import (
    audit_pilot_file,
    audit_pilot_plan,
    build_pilot_plan,
    load_pilot_spec,
)


def test_pilot_plan_accounts_for_matrix_views_identity_gates_and_episodes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tatbot_sim.inkmap.pilot.source_state",
        lambda: {"repository": "test/repository", "revision": "1" * 40, "dirty": False},
    )
    plan = build_pilot_plan(tmp_path / "pilot", seed=42)
    assert plan["counts"] == {
        "reference_scene_templates": 36,
        "all_identity_scenes": 108,
        "planned_scenes": 36,
        "identity_blocked_scenes": 72,
        "planned_perception_images": 108,
        "full_perception_images_after_identity_review": 324,
        "drawing_episodes": 12,
    }
    assert {item["cell_id"] for item in plan["drawing_episodes"]} == {item["id"] for item in load_pilot_spec()["cells"]}
    assert {item["artwork_id"] for item in plan["drawing_episodes"]} == {item["id"] for item in load_pilot_spec()["artworks"]}
    assert plan["gates"]["stencil_renderer_contract"] == "implemented_ungenerated"
    assert all(not item["status"].startswith("pending_") for item in plan["episode_variants"])
    assert all(item["success_label"] is None for item in plan["drawing_episodes"])
    assert audit_pilot_plan(plan) == []
    assert audit_pilot_file(tmp_path / "pilot/pilot-plan.json")["status"] == "pass"
    assert json.loads((tmp_path / "pilot/audit.json").read_text())["status"] == "pass"


def test_pilot_auditor_rejects_identity_admission_success_claims_and_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tatbot_sim.inkmap.pilot.source_state",
        lambda: {"repository": "test/repository", "revision": "1" * 40, "dirty": False},
    )
    plan = build_pilot_plan(tmp_path / "pilot", seed=7)
    bad = copy.deepcopy(plan)
    candidate = next(item for item in bad["scenes"] if item["identity_id"] != "reference")
    candidate["status"] = "planned"
    bad["drawing_episodes"][0]["success_label"] = True
    raw = dict(bad)
    raw["content_sha256"] = ""
    bad["content_sha256"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    problems = audit_pilot_plan(bad)
    assert any("unreviewed identity" in problem for problem in problems)
    assert any("success label" in problem for problem in problems)
    bad["content_sha256"] = "0" * 64
    assert any("digest mismatch" in problem for problem in audit_pilot_plan(bad))
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad))
    assert audit_pilot_file(bad_path)["status"] == "fail"


def test_pilot_size_is_a_config_change_and_occlusion_follows_the_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tatbot_sim.inkmap.pilot.source_state",
        lambda: {"repository": "test/repository", "revision": "1" * 40, "dirty": False},
    )
    import tatbot_sim.inkmap.pilot as pilot
    spec = copy.deepcopy(load_pilot_spec())
    spec["cells"] = spec["cells"][:2]
    spec["artworks"] = spec["artworks"][:3]
    spec["drawing_episode_count"] = 7
    small = tmp_path / "small-spec.json"
    small.write_text(json.dumps(spec))
    monkeypatch.setattr(pilot, "SPEC_PATH", small)
    plan = build_pilot_plan(tmp_path / "pilot", seed=3)
    assert plan["counts"]["reference_scene_templates"] == 2 * 3 * 2
    assert plan["counts"]["drawing_episodes"] == 7
    assert audit_pilot_plan(plan) == []
    for episode in plan["drawing_episodes"]:
        occluded = episode["variant"]["id"] == "occluded"
        assert (episode["occlusion_fraction"] > 0) == occluded
        assert ("--occlusion-fraction" in episode["generate_flags"]) == occluded
        assert episode["generate_flags"][:2] == ["--episode-variant", episode["variant"]["id"]]
    spec["drawing_episode_count"] = 2
    small.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="span every cell and artwork"):
        load_pilot_spec(small)


def test_pilot_output_must_be_new_and_outside_repository(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep").write_text("owned")
    with pytest.raises(FileExistsError):
        build_pilot_plan(occupied)
    with pytest.raises(ValueError, match="outside the repository"):
        build_pilot_plan(__import__("tatbot_sim.repo", fromlist=["repo_root"]).repo_root() / "pilot-output")
