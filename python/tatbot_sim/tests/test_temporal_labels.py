from __future__ import annotations

import hashlib
import json

import numpy as np
from tatbot_sim.temporal_labels import TemporalRecorder, audit_timeline


def _step(batch: int, step: int) -> dict[str, np.ndarray]:
    return {
        "tool_pose_world": np.tile([0, 0, 0.1, 1, 0, 0, 0], (batch, 1)),
        "contact_distance_m": np.zeros(batch),
        "contact_incidence": np.ones(batch),
        "pen_down": np.ones(batch, dtype=bool),
        "target_world": np.tile([0, 0, 0.1], (batch, 1)),
        "target_valid": np.ones(batch, dtype=bool),
        "surface_point_world": np.tile([0, 0, 0.1], (batch, 1)),
        "surface_normal_world": np.tile([0, 0, 1], (batch, 1)),
        "primitive_index": np.zeros(batch, dtype=np.int32),
        "layer_index": np.zeros(batch, dtype=np.int32),
        "progress": np.full(batch, (step + 1) / 3),
        "deposited_coverage": np.full(batch, step / 10),
        "remaining_target_fraction": np.full(batch, 1 - step / 10),
        "texture_synchronized": np.ones(batch, dtype=bool),
        "stencil_visible_fraction": np.zeros(batch),
        "observation_occlusion_fraction": np.zeros(batch),
    }


def test_temporal_sidecars_are_length_aligned_hashed_and_privileged(tmp_path):
    recorder = TemporalRecorder(2)
    for step in range(3):
        recorder.append(**_step(2, step))
    records = recorder.write(
        tmp_path / "meta" / "privileged",
        kept=[0, None],
        lengths=np.asarray([3, 2]),
        stroke_metadata=[[{"event_index": 0, "layer_index": 0}], []],
        scenario_sha256="a" * 64,
    )
    assert records[1] is None
    record = records[0]
    assert record is not None
    path = tmp_path / record["path"]
    manifest = json.loads((tmp_path / record["manifest"]).read_text())
    assert audit_timeline(path, manifest) == []
    assert manifest["conventions"]["policy_boundary"].startswith("privileged")
    with np.load(path, allow_pickle=False) as arrays:
        assert len(arrays["tool_pose_world"]) == 3
        assert np.all(arrays["texture_synchronized"])


def test_temporal_audit_refuses_stale_rgb_or_unexplained_contact(tmp_path):
    recorder = TemporalRecorder(1)
    values = _step(1, 0)
    values["texture_synchronized"][:] = False
    values["target_valid"][:] = False
    recorder.append(**values)
    record = recorder.write(
        tmp_path / "meta" / "privileged",
        kept=[0],
        lengths=np.asarray([1]),
        stroke_metadata=[[{"event_index": 0, "layer_index": 0}]],
        scenario_sha256="a" * 64,
    )[0]
    assert record is not None
    path = tmp_path / record["path"]
    manifest = json.loads((tmp_path / record["manifest"]).read_text())
    problems = audit_timeline(path, manifest)
    assert any("no intended target" in problem for problem in problems)
    assert any("synchronization" in problem for problem in problems)


def test_temporal_audit_refuses_a_stencil_that_reappears(tmp_path):
    recorder = TemporalRecorder(1)
    for step, visible in enumerate((0.3, 0.2, 0.25)):
        values = _step(1, step)
        values["stencil_visible_fraction"] = np.asarray([visible])
        recorder.append(**values)
    record = recorder.write(
        tmp_path, kept=[0], lengths=np.asarray([3]), stroke_metadata=None,
        scenario_sha256=None, variant_ids=["stencil-start"], expected_outcomes=["nominal"],
    )[0]
    assert record is not None
    manifest = json.loads((tmp_path / record["manifest"].split("/")[-1]).read_text())
    problems = audit_timeline(tmp_path / record["path"].split("/")[-1], manifest)
    assert any("stencil visibility grows" in problem for problem in problems)


def test_temporal_audit_requires_variant_specific_visibility_labels(tmp_path):
    recorder = TemporalRecorder(2)
    values = _step(2, 0)
    values["stencil_visible_fraction"][0] = 0.25
    values["observation_occlusion_fraction"][1] = 0.15
    recorder.append(**values)
    records = recorder.write(
        tmp_path / "meta" / "privileged",
        kept=[0, 1],
        lengths=np.asarray([1, 1]),
        stroke_metadata=[None, None],
        scenario_sha256="a" * 64,
        variant_ids=["stencil-start", "occluded"],
        expected_outcomes=["nominal", "occluded"],
    )
    for record in records:
        assert record is not None
        path = tmp_path / record["path"]
        manifest = json.loads((tmp_path / record["manifest"]).read_text())
        assert audit_timeline(path, manifest) == []

    for index, field in enumerate(
        ("stencil_visible_fraction", "observation_occlusion_fraction")
    ):
        record = records[index]
        assert record is not None
        path = tmp_path / record["path"]
        manifest = json.loads((tmp_path / record["manifest"]).read_text())
        with np.load(path, allow_pickle=False) as arrays:
            changed = {name: arrays[name] for name in arrays.files}
        changed[field] = np.zeros_like(changed[field])
        np.savez_compressed(path, **changed)
        manifest["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert any("has no" in problem for problem in audit_timeline(path, manifest))
