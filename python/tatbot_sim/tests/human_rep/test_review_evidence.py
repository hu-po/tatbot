from __future__ import annotations

import argparse
import json

import pytest
from tatbot_sim.human_rep import review_evidence
from tatbot_sim.human_rep.review import make_pending_release_gates
from tatbot_sim.human_rep.review_evidence import _capability_inventory, run_evidence


def test_review_evidence_emits_four_independent_disabled_packets(tmp_path):
    phase_root = tmp_path / "run"
    for phase in range(9):
        directory = phase_root / f"phase-{phase}"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps({"phase": phase, "status": "synthetic-pass", "accepted": {}, "known_gaps": []})
        )
    gates_path = tmp_path / "gates.json"
    gates_path.write_text(
        json.dumps(make_pending_release_gates(created_utc="2026-09-04T17:00:00Z", source_sha256="a" * 64))
    )
    output = phase_root / "phase-9"
    result = run_evidence(
        argparse.Namespace(
            output=output,
            phase_root=phase_root,
            gates=gates_path,
            created_utc="2026-09-04T17:01:00Z",
        )
    )
    inventory = json.loads((output / "capability-inventory.json").read_text())
    assert all(item["status"] == "historical" for item in inventory.values())
    assert all(item["current_revision"] is False for item in inventory.values())
    assert inventory["proposal_training"]["lifecycle"] == "frozen"
    assert "proposal_training (frozen)" in (output / "capabilities.md").read_text()
    assert "P4" not in (output / "offline-full-pipeline.md").read_text()
    assert result["accepted"]["review_packets"] is True
    assert result["accepted"]["external_actions_fail_closed"] is True
    packets = sorted((output / "approvals").glob("*.json"))
    assert len(packets) == 4
    for path in packets:
        packet = json.loads(path.read_text())
        assert "required_capabilities" in packet and "required_phases" not in packet
        assert packet["approved"] is False
        assert packet["external_action_enabled"] is False


def test_review_evidence_refuses_overwrite(tmp_path):
    output = tmp_path / "phase-9"
    output.mkdir()
    (output / "keep.txt").write_text("evidence")
    with pytest.raises(FileExistsError):
        run_evidence(
            argparse.Namespace(
                output=output,
                phase_root=tmp_path,
                gates=tmp_path / "missing.json",
                created_utc="2026-09-04T17:01:00Z",
            )
        )


@pytest.mark.parametrize("revision,dirty,status", [("a" * 40, "", "reported"),
                                                   ("b" * 40, "", "stale"),
                                                   ("a" * 40, " M source.py", "stale")])
def test_named_evidence_is_revision_scoped_without_implying_acceptance(tmp_path, monkeypatch, revision, dirty, status):
    monkeypatch.setattr(review_evidence, "git_output",
                        lambda *args: "a" * 40 if args[0] == "rev-parse" else "")
    directory = tmp_path / "stroke_lowering"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({
        "schema": "tatbot.capability-evidence/1", "capability": "stroke_lowering",
        "evidence_type": "synthetic", "status": "pass", "base_git_sha": revision,
        "dirty_state": dirty,
    }))
    inventory = _capability_inventory(tmp_path)
    assert inventory["stroke_lowering"]["status"] == status
    assert inventory["stroke_lowering"]["evidence_type"] == "synthetic"
    assert inventory["stroke_lowering"]["hardware_authority"] is False
    assert inventory["nominal_body"]["status"] == "missing"


def test_named_evidence_rejects_mislabeled_capability(tmp_path):
    directory = tmp_path / "stroke_lowering"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({
        "schema": "tatbot.capability-evidence/1", "capability": "nominal_body",
        "evidence_type": "software", "status": "pass",
    }))
    with pytest.raises(ValueError, match="named capability evidence"):
        _capability_inventory(tmp_path)
