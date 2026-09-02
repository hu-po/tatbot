"""Optional independent body-axis/reseat evidence is digest-bound."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import il_touchoff  # noqa: E402
import tool_body_qualify  # noqa: E402
import tool_spec  # noqa: E402


def _fixture(tmp_path: Path):
    (tmp_path / "config" / "tools").mkdir(parents=True)
    (tmp_path / "config" / "tools" / "fixture-pen.yaml").write_text(
        "schema_version: 2\n"
        "tool_id: fixture-pen\n"
        "kind: rotary_pen\n"
        'display_name: "Fixture"\n'
        'prompt_phrase: "using fixture tip"\n'
        "profile: [[-0.04, 0.010], [0.01, 0.010], [0.05, 0.001]]\n"
        "contact: true\n"
        "tip_tolerance_m: 0.005\n"
        "seat_tolerance_deg: 5.0\n"
        "seat_residual_m: 0.001\n")
    report = {
        "schema_version": 1,
        "tool_id": "fixture-pen",
        "arm": "right",
        "frame": "right/tool_mount",
        "method": "fiducial-body-axis-v1",
        "measurement_source": "calibrated-stereo-collar-v1",
        "independent_of_tip_fit": True,
        "selected_cycle": 5,
        "samples": [],
    }
    axis = [0.05, 0.0, 0.998749217771909]
    for cycle in range(1, 6):
        x = (cycle - 1) * 0.00002
        report["samples"].append({
            "cycle": cycle,
            "utc": f"2026-09-01T15:0{cycle}:00Z",
            "touchoff_session": f"reseat-{cycle}",
            "body_origin_m": [x, 0.0, 0.0],
            "body_axis_unit": axis,
            "tip_offset_m": [x + axis[0] * 0.05, 0.0, axis[2] * 0.05],
        })
    final_tip = report["samples"][-1]["tip_offset_m"]
    right = {
        "tool_id": "fixture-pen",
        "tip_frame": "right/tool_mount",
        "pen_tip_offset_x": final_tip[0],
        "pen_tip_offset_y": final_tip[1],
        "pen_tip_offset_z": final_tip[2],
        "carriage_m": 0.0,
        "paper_plane_z": 0.006,
        "paper_band_mm": None,
        "ee_contact_z": None,
        "touchoff": {
            "utc": "2026-09-01T15:05:00Z",
            "session": "reseat-5",
            "n_plate": 8,
            "n_pad": 0,
            "cond": 5.0,
            "residual_mm": 0.2,
            "holdout_mm": 0.2,
            "spread_deg": 45.0,
            "note": "",
        },
    }
    workspace_path = tmp_path / "config" / "workspace.yaml"
    workspace_path.write_text(il_touchoff.render_workspace(right))
    report_path = tmp_path / "study.json"
    report_path.write_text(json.dumps(report))
    return report, report_path, workspace_path


def test_five_reseats_bind_canonical_evidence_and_qualify_geometry(tmp_path):
    report, report_path, workspace_path = _fixture(tmp_path)
    qualification, target = tool_body_qualify.qualify(
        report_path, "fixture-pen", repo=tmp_path,
        workspace_path=workspace_path, write=False)
    assert qualification.sample_count == 5
    assert qualification.tip_repeatability_max_m == pytest.approx(0.00008)
    assert not target.exists()

    qualification, target = tool_body_qualify.qualify(
        report_path, "fixture-pen", repo=tmp_path,
        workspace_path=workspace_path, write=True)
    assert target.is_file()
    canonical = tool_body_qualify.canonical_report_bytes(report)
    assert target.read_bytes() == canonical
    workspace = tool_spec.read_workspace(tmp_path)
    side = workspace["right"]
    assert side["tool_body_status"] == "qualified"
    assert side["tool_body_samples"] == 5
    assert side["tool_body_selected_cycle"] == 5
    assert side["tool_body_report_sha256"] == hashlib.sha256(canonical).hexdigest()
    spec = tool_spec.load_tool("fixture-pen", tmp_path)
    geometry = tool_spec.resolved_tool_geometry(spec, workspace, repo=tmp_path)
    assert geometry.status == "qualified"
    assert geometry.source == "workspace-body-pose"
    assert geometry.qualification_error is None
    assert geometry.body_tip_offset_m == pytest.approx(qualification.tip_offset_m)


@pytest.mark.parametrize("mutate,message", [
    (lambda report: report["samples"].pop(), "need at least 5"),
    (lambda report: report.update(independent_of_tip_fit=False),
     "independent_of_tip_fit"),
    (lambda report: report["samples"][0].update(
        tip_offset_m=[0.0035, 0.0, 0.04993746088859545]),
     "endpoint is 1.000 mm"),
    (lambda report: report["samples"][0].update(
        body_origin_m=[0.002, 0.0, 0.0],
        tip_offset_m=[0.0045, 0.0, 0.04993746088859545]),
     "reseat tip spread"),
    (lambda report: report["samples"][-1].update(touchoff_session="not-current"),
     "selected touch-off session"),
])
def test_report_failures_write_nothing(tmp_path, mutate, message):
    report, report_path, workspace_path = _fixture(tmp_path)
    before = workspace_path.read_bytes()
    mutate(report)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        tool_body_qualify.qualify(
            report_path, "fixture-pen", repo=tmp_path,
            workspace_path=workspace_path, write=True)
    assert workspace_path.read_bytes() == before
    report_dir = tmp_path / tool_spec.BODY_POSE_REPORT_DIR
    assert not report_dir.exists()


def test_tampered_body_report_keeps_only_pivot_contact_qualification(tmp_path):
    _, report_path, workspace_path = _fixture(tmp_path)
    _, target = tool_body_qualify.qualify(
        report_path, "fixture-pen", repo=tmp_path,
        workspace_path=workspace_path, write=True)
    target.write_text(target.read_text() + " ")
    workspace = tool_spec.read_workspace(tmp_path)
    spec = tool_spec.load_tool("fixture-pen", tmp_path)
    geometry = tool_spec.resolved_tool_geometry(spec, workspace, repo=tmp_path)
    assert geometry.status == "contact-qualified"
    assert geometry.source == "touch-axis-inferred"
    assert geometry.contact_status == "pivot-calibrated"
    assert geometry.body_pose_status == "axis-inferred"
    assert "SHA-256" in geometry.qualification_error


def test_selected_cycle_must_be_the_current_last_seat(tmp_path):
    report, report_path, workspace_path = _fixture(tmp_path)
    report["selected_cycle"] = 4
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="final reseat"):
        tool_body_qualify.qualify(
            report_path, "fixture-pen", repo=tmp_path,
            workspace_path=workspace_path)


def test_report_copy_is_never_replaced_by_different_evidence(tmp_path):
    report, report_path, workspace_path = _fixture(tmp_path)
    _, target = tool_body_qualify.qualify(
        report_path, "fixture-pen", repo=tmp_path,
        workspace_path=workspace_path, write=True)
    original = target.read_bytes()
    changed = copy.deepcopy(report)
    changed["samples"][0]["utc"] = "2026-09-01T15:00:30Z"
    report_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="refusing to replace different evidence"):
        tool_body_qualify.qualify(
            report_path, "fixture-pen", repo=tmp_path,
            workspace_path=workspace_path, write=True)
    assert target.read_bytes() == original
