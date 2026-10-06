"""Known transforms, unobservable motion, and calibrated evidence boundary tests."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("open3d")
pytest.importorskip("threadpoolctl")
from surface_rgbd import Settings, evaluate, pose_error, recording_frames, rigid_matrix, synthetic_frames


def test_known_motion_and_failure_cases(tmp_path):
    for case in ("motion", "dropout", "gap", "no_overlap"):
        report = evaluate(synthetic_frames(case, 8), Settings(), tmp_path / (case + ".jsonl"))
        for method, counts in report["failure_detection"].items():
            assert counts["false_accept"] == counts["false_reject"] == 0
            assert counts["true_accept"] > 0
            if case == "motion":
                assert report["material_rms_mm"][method]["max"] < .1
            else:
                assert counts["true_reject"] > 0
        assert report["cpu_seconds"] > 0
        assert report["stage_wall_ms"]["prepare"]["p50"] > 0


def test_uniform_symmetry_exposes_false_acceptance(tmp_path):
    report = evaluate(synthetic_frames("symmetric_alias", 4), Settings(), tmp_path / "symmetry.jsonl")
    for method in report["failure_detection"]:
        assert report["failure_detection"][method]["false_accept"] == 3
        assert report["material_rms_mm"][method]["p50"] > 2


def test_error_convention_uses_physical_points():
    estimate, truth = np.eye(4), np.eye(4)
    truth[0, 3] = .004
    errors = pose_error(estimate, truth, np.array([[0, 0, .2], [.1, 0, .3]]))
    assert errors == pytest.approx({"translation_mm": 4., "rotation_deg": 0., "material_rms_mm": 4.})
    with pytest.raises(ValueError, match="rigid"):
        rigid_matrix(np.ones((4, 4)))


def evidence(tmp_path, *, skew=0, units="0.0001", distortion=None, aligned="wrist_color"):
    for kind in ("color", "depth"):
        root = tmp_path / ("wrist_" + kind)
        root.mkdir(exist_ok=True)
        rows = []
        for i in range(3):
            pixels = np.full((16, 20) if kind == "depth" else (16, 20, 3),
                             2000 if kind == "depth" else 100, dtype="<u2" if kind == "depth" else np.uint8)
            payload = pixels.tobytes()
            (root / str(i)).write_bytes(payload)
            attributes = {"intrinsics": json.dumps({"width": 20, "height": 16, "fx": 100, "fy": 100,
                           "ppx": 10, "ppy": 8, "distortion_model": "None" if distortion is None else "Unsupported", "distortion_coefficients": distortion or [0]*5})}
            if kind == "depth":
                attributes.update(depth_units_m=units, aligned_to=aligned)
            rows.append({"payload_file": str(i), "payload_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                         "metadata": {"sensor_name": "wrist_" + kind, "sequence": i,
                                      "timestamps": {"normalized_unix_ns": 1_000_000_000 + i * 50_000_000 + (skew if kind == "color" else 0)},
                                      "profile": {"width": 20, "height": 16, "format": "z16" if kind == "depth" else "bgr8"},
                                      "attributes": attributes}})
        (root / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return lambda: recording_frames(tmp_path, "wrist", (0, 0, 20, 16), (.1, .4))


def test_rgbd_reuses_verified_evidence_and_measured_units(tmp_path):
    frames = evidence(tmp_path)
    first = next(frames())
    assert np.allclose(first["points"][:, 2], .2)
    assert np.allclose(first["points"][0], [-.02, -.016, .2])
    assert len(first["points"]) == 320
    (tmp_path / "wrist_depth" / "0").write_bytes(b"x" * 640)
    with pytest.raises(ValueError, match="checksum"):
        next(frames())


@pytest.mark.parametrize("kwargs,message", [({"units": "nan"}, "units"),
    ({"distortion": [1, 0, 0, 0, 0]}, "distortion"), ({"aligned": "other"}, "aligned")])
def test_invalid_calibration_refused(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        next(evidence(tmp_path, **kwargs)())


def test_unpaired_frames_are_reported_not_silently_dropped(tmp_path):
    frames = list(evidence(tmp_path, skew=20_000_000)())
    assert len(frames) == 3
    assert all(f["invalid"] == "unpaired_rgbd" for f in frames)


def test_real_recording_does_not_claim_ground_truth(tmp_path):
    report = evaluate(evidence(tmp_path)(), Settings(), tmp_path / "out.jsonl")
    for method in report["failure_detection"]:
        assert report["failure_detection"][method]["unlabelled"] == 2
        assert report["material_rms_mm"][method]["max"] is None


def test_no_frame_pair_refused(tmp_path):
    with pytest.raises(ValueError, match="two frames"):
        evaluate(iter(()), Settings(), tmp_path / "out.jsonl")


def test_calibration_change_and_timestamp_reversal_refused(tmp_path):
    frames = evidence(tmp_path)
    path = tmp_path / "wrist_depth" / "frames.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["metadata"]["attributes"]["depth_units_m"] = "0.001"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="changed"):
        list(frames())
    rows[1]["metadata"]["timestamps"]["normalized_unix_ns"] = 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="increasing"):
        list(frames())


def test_supplied_ground_truth_requires_provenance_and_complete_timestamps(tmp_path):
    from surface_rgbd import with_ground_truth
    frames = list(evidence(tmp_path)())
    path = tmp_path / "truth.json"
    truth = {"schema": "tatbot.surface-rgbd-ground-truth/1", "provenance": "independent test fixture",
             "frames": [{"timestamp_ns": f["stamp"], "object_to_camera": np.eye(4).tolist()} for f in frames]}
    path.write_text(json.dumps(truth))
    assert all("pose" in f for f in with_ground_truth(iter(frames), path))
    truth["frames"].pop()
    path.write_text(json.dumps(truth))
    with pytest.raises(ValueError, match="missing"):
        list(with_ground_truth(iter(frames), path))
    del truth["provenance"]
    path.write_text(json.dumps(truth))
    with pytest.raises(ValueError, match="provenance"):
        list(with_ground_truth(iter(frames), path))


def test_textured_plane_recovers_motion_that_geometry_cannot(tmp_path):
    report = evaluate(synthetic_frames("flat_textured", 5), Settings(), tmp_path / "flat.jsonl")
    assert report["failure_detection"]["point_to_plane"]["false_accept"] == 4
    assert report["failure_detection"]["colored"]["true_accept"] == 4
    assert report["material_rms_mm"]["colored"]["max"] < .1


def test_distorted_depth_uses_the_existing_mapper_rays():
    from surface_rgbd import rgbd_points
    table = json.loads((Path(__file__).parent / "fixtures" / "deprojection-sdk-table.json").read_text())
    tolerance = 2e-7  # one float32 ulp of the SDK's own output, see test_rgbd_geometry
    record = table["models"]["BrownConradyInverse"]
    active = record["intrinsics"]
    meta = {"profile": {"width": 640, "height": 480},
            "attributes": {"intrinsics": active, "depth_units_m": ".0001"}}
    depth = np.zeros((480, 640), dtype=np.uint16)
    assert table["pixels"][0] == [0, 0]
    depth[0, 0] = 2000
    points, _ = rgbd_points(depth, np.zeros((480, 640, 3), dtype=np.uint8), meta, (0, 0, 640, 480), (.1, .4))
    np.testing.assert_allclose(points[0], np.asarray(record["rays"][0])*.2, atol=.2*tolerance, rtol=0)


def test_owner_informational_flags_and_cumulative_drops(tmp_path):
    frames = evidence(tmp_path)
    for kind in ('color', 'depth'):
        path = tmp_path / ('wrist_' + kind) / 'frames.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row['metadata']['flags'] = ['timestamp_domain=Global Time']
            row['metadata']['timestamps']['source_domain'] = 'real_sense_global'
            row['metadata']['dropped_before'] = 44
        rows[2]['metadata']['dropped_before'] = 45
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    output = list(frames())
    assert 'points' in output[0] and 'points' in output[1]
    assert output[2]['invalid'] == 'capture_drop_counter_changed'
    path = tmp_path / 'wrist_depth' / 'frames.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['metadata']['flags'].append('depth_units_unknown')
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert next(frames())['invalid'] == 'flagged_capture'


def test_image_consistency_detects_wrong_transform_and_refuses_missing_depth():
    import cv2
    from surface_consistency import consistency, correspondences
    rng = np.random.default_rng(10)
    gray = cv2.GaussianBlur(rng.integers(0, 256, (120, 160), dtype=np.uint8), (3, 3), 0)
    moved = cv2.warpAffine(gray, np.float32([[1, 0, 4], [0, 1, 0]]), (160, 120))
    y, x = np.mgrid[:120, :160]
    rays = np.stack(((x-80)/200, (y-60)/200, np.ones_like(x)), axis=-1)
    a = {'gray': gray, 'rays': rays, 'depth_m': np.full((120, 160), .2), 'depth_range': (.1, .3)}
    b = dict(a, gray=moved)
    pairs = correspondences(a, b, (15, 15, 130, 90))
    correct = np.eye(4)
    correct[0, 3] = .004
    assert pairs['count'] >= 12
    assert consistency(pairs, correct)['residual_mm']['p95'] < .1
    assert consistency(pairs, np.eye(4))['residual_mm']['p50'] > 3.5
    assert correspondences(a, dict(b, depth_m=np.zeros((120, 160))), (15, 15, 130, 90))['status'] == 'unavailable'
    assert correspondences(dict(a, gray=np.zeros_like(gray)), b, (15, 15, 130, 90))['reason'] == 'no_image_features'


def test_image_check_data_is_opt_in_and_does_not_become_ground_truth(tmp_path):
    evidence(tmp_path)
    frames = list(recording_frames(tmp_path, 'wrist', (0, 0, 20, 16), (.1, .4), include_images=True))
    assert frames[0]['rgbd']['gray'].shape == (16, 20)
    assert np.allclose(frames[0]['rgbd']['depth_m'], .2)
    assert 'pose' not in frames[0]
    report = evaluate(iter(frames), Settings(), tmp_path/'checked.jsonl')
    assert report['image_depth_consistency']['colored']['unavailable_pairs'] == 2
    assert report['material_rms_mm']['colored']['p50'] is None


def test_single_method_budget_runs_only_requested_solver(tmp_path):
    report = evaluate(synthetic_frames('motion', 5), Settings(max_points=5000),
                      tmp_path/'budget.jsonl', methods=('colored',))
    assert set(report['failure_detection']) == {'colored'}
    assert report['stage_wall_ms']['point_to_plane']['p50'] is None
    assert report['material_rms_mm']['colored']['max'] < .1


def test_stationary_depth_excludes_sparse_pixels_and_counts_invalid_frames(tmp_path):
    from surface_rgbd import evaluate_stationary_depth
    frames = []
    for i in range(10):
        z = np.array([[.2 + (i % 2) * .002, np.nan]], dtype=np.float32)
        if i == 0:
            z[0, 1] = .2
        frames.append({'stamp': 1_000_000_000 + i * 30_000_000, 'depth_roi_m': z})
    result = evaluate_stationary_depth(frames, tmp_path / 'quality.jsonl')
    assert result['valid_fraction_including_invalid_frames'] == pytest.approx(.55)
    assert result['pixels_with_90_percent_support_fraction'] == .5
    assert result['temporal_robust_sigma_mm']['p50'] == pytest.approx(1.4826, abs=.001)
    assert result['per_pair_depth_delta_p95_mm']['p50'] == pytest.approx(2, abs=.001)
    frames[4] = {'stamp': frames[4]['stamp'], 'invalid': 'unpaired_rgbd'}
    result = evaluate_stationary_depth(frames, tmp_path / 'invalid.jsonl')
    assert result['invalid_frames'] == 1
    assert result['consecutive_pairs'] == 7
    assert result['valid_fraction_including_invalid_frames'] == .5


def test_stationary_depth_holes_are_not_zero_noise(tmp_path):
    from surface_rgbd import evaluate_stationary_depth
    frames = [{'stamp': i * 1_000_000_000, 'depth_roi_m': np.full((2, 2), np.nan)} for i in range(1, 4)]
    result = evaluate_stationary_depth(frames, tmp_path / 'holes.jsonl')
    assert result['temporal_robust_sigma_mm']['p50'] is None
    assert result['valid_fraction_including_invalid_frames'] == 0
    assert result['consecutive_pairs'] == 0


def test_stationary_depth_uses_verified_evidence(tmp_path):
    from surface_rgbd import evaluate_stationary_depth
    evidence(tmp_path)
    frames = recording_frames(tmp_path, 'wrist', (0, 0, 20, 16), (.1, .3), include_depth_roi=True)
    result = evaluate_stationary_depth(frames, tmp_path / 'evidence-quality.jsonl')
    assert result['valid_fraction_including_invalid_frames'] == 1
    assert result['temporal_robust_sigma_mm']['max'] == 0
