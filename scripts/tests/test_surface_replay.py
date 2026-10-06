"""Material identity, failure detection and native evidence compatibility."""
import hashlib
import json

import cv2
import numpy as np
import pytest
from surface_replay import Settings, SparsePatchTracker, evaluate, recording_frames, synthetic_frames
from visiond_wire import read_evidence_frame


@pytest.mark.parametrize("case", ["motion", "brightness"])
def test_known_motion_preserves_identity_with_subpixel_error(tmp_path, case):
    report = evaluate(synthetic_frames(case, count=60), (160, 120, 320, 240), Settings(), tmp_path / "tracks.jsonl")
    assert report["states"] == {"initialized": 1, "tracked": 59}
    assert report["ground_truth_point_error_px"]["p95"] < 1
    rows = [json.loads(line) for line in (tmp_path / "tracks.jsonl").read_text().splitlines()]
    for before, after in zip(rows, rows[1:], strict=False):
        assert set(after["ids"]) <= set(before["ids"])


@pytest.mark.parametrize("case,reason", [("blank", "insufficient_texture"), ("occlusion", "insufficient_tracks"), ("gap", "timestamp_gap")])
def test_loss_stays_latched_after_texture_returns(tmp_path, case, reason):
    report = evaluate(synthetic_frames(case, count=45), (160, 120, 320, 240), Settings(), tmp_path / "tracks.jsonl")
    assert report["final_reason"] == reason
    rows = [json.loads(line) for line in (tmp_path / "tracks.jsonl").read_text().splitlines()]
    first_lost = next(i for i, row in enumerate(rows) if row["status"] == "lost")
    assert all(row["status"] == "lost" and not row["ids"] and not row["points_px"] for row in rows[first_lost:])


def test_geometry_change_and_invalid_roi():
    image, stamp, _, _ = next(synthetic_frames("motion"))
    tracker = SparsePatchTracker((160, 120, 320, 240))
    tracker.step(image, stamp)
    assert tracker.step(image[:300], stamp + 30_000_000)["reason"] == "geometry_changed"
    with pytest.raises(ValueError, match="ROI"):
        SparsePatchTracker((-1, 0, 100, 100)).step(image, stamp)


@pytest.mark.parametrize("format", ["y8", "jpeg"])
def test_native_recording_decode_and_checksum(tmp_path, format):
    image = np.full((24, 32), 90, np.uint8)
    payload = image.tobytes() if format == "y8" else cv2.imencode(".jpg", image)[1].tobytes()
    entry = {"payload_file": "frame", "payload_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
             "metadata": {"sensor_name": "camera1", "sequence": 1,
                          "profile": {"width": 32, "height": 24, "format": format},
                          "timestamps": {"normalized_unix_ns": 1_000_000_000}}}
    (tmp_path / "frame").write_bytes(payload)
    index = tmp_path / "frames.jsonl"
    index.write_text(json.dumps(entry) + "\n")
    decoded, stamp, _, _ = next(recording_frames(index))
    assert decoded.shape == ((24, 32) if format == "y8" else (24, 32, 3))
    assert stamp == 1_000_000_000
    assert np.max(abs(decoded.astype(int) - 90)) <= 1
    (tmp_path / "frame").write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="checksum"):
        read_evidence_frame(tmp_path, entry)


def test_empty_recording_is_not_success(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        evaluate(iter(()), (0, 0, 10, 10), Settings(), tmp_path / "out.jsonl")


def test_repeated_texture_documents_false_tracking_limit(tmp_path):
    report = evaluate(synthetic_frames("alias"), (160, 120, 320, 240), Settings(), tmp_path / "alias.jsonl")
    # Matching consistency alone cannot disambiguate an integer texture period.
    assert report["states"].get("tracked", 0) > 0
    assert report["ground_truth_point_error_px"]["p95"] > 10


def test_crop_at_image_edge_preserves_pixel_coordinates(tmp_path):
    report = evaluate(synthetic_frames("motion", count=60), (0, 0, 128, 128), Settings(), tmp_path / "edge.jsonl")
    assert report["states"]["tracked"] == 59
    assert report["ground_truth_point_error_px"]["p95"] < 1


def test_recording_without_normalized_time_is_rejected(tmp_path):
    index = tmp_path / "frames.jsonl"
    index.write_text(json.dumps({"metadata": {"sensor_name": "camera1", "timestamps": {"source_ns": 123}}}) + "\n")
    with pytest.raises(ValueError, match="normalized_unix_ns"):
        next(recording_frames(index))


def test_evidence_cannot_escape_its_recording(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        read_evidence_frame(tmp_path, {"payload_file": "../outside"})


def test_short_blur_reacquires_original_ids_with_bounded_error(tmp_path):
    source = list(synthetic_frames("motion", count=45))
    altered = [(cv2.GaussianBlur(im, (41, 41), 0) if 8 <= i < 14 else im, stamp, meta, matrix)
               for i, (im, stamp, meta, matrix) in enumerate(source)]
    report = evaluate(iter(altered), (160, 120, 320, 240), Settings(), tmp_path / "recovery.jsonl", recover=True)
    rows = [json.loads(s) for s in (tmp_path / "recovery.jsonl").read_text().splitlines()]
    assert report["states"].get("reacquired", 0) >= 1
    assert report["ground_truth_point_error_px"]["p95"] < 1
    assert report["final_reason"] is None
    assert all(not r["points_px"] and not r["ids"] for r in rows if r["status"] == "paused")
    original = set(rows[0]["ids"])
    assert all(set(r["ids"]) <= original for r in rows)
    recovered = next(i for i, r in enumerate(rows) if r["status"] == "reacquired")
    assert any(r["reason"] == "recovery_confirmation" for r in rows[:recovered])


def test_recovery_timeout_never_reseeds_returning_texture(tmp_path):
    source = list(synthetic_frames("motion", count=60))
    altered = [(np.zeros_like(im) if 5 <= i < 40 else im, stamp, meta, matrix)
               for i, (im, stamp, meta, matrix) in enumerate(source)]
    report = evaluate(iter(altered), (160, 120, 320, 240), Settings(), tmp_path / "timeout.jsonl", recover=True)
    assert report["final_reason"] == "recovery_timeout"
    rows = [json.loads(s) for s in (tmp_path / "timeout.jsonl").read_text().splitlines()]
    assert all(r["status"] == "lost" and not r["ids"] for r in rows[40:])


def test_recovery_does_not_relax_missing_frame_gate(tmp_path):
    report = evaluate(synthetic_frames("gap", count=45), (160, 120, 320, 240), Settings(), tmp_path / "gap.jsonl", recover=True)
    assert report["final_reason"] == "timestamp_gap"


def test_recovery_handles_blank_initialization_and_geometry_change():
    from surface_replay import RecoveringPatchTracker
    tracker = RecoveringPatchTracker((0, 0, 128, 128))
    blank = np.zeros((240, 320), np.uint8)
    assert tracker.step(blank, 1_000_000_000)["status"] == "lost"
    assert tracker.step(blank, 1_030_000_000)["status"] == "lost"
    image, stamp, _, _ = next(synthetic_frames("motion"))
    tracker = RecoveringPatchTracker((160, 120, 320, 240))
    tracker.step(image, stamp)
    assert tracker.step(image[:300], stamp + 30_000_000)["reason"] == "geometry_changed"


@pytest.mark.parametrize("case", ["blur", "short_occlusion"])
def test_recovery_suite_restores_tracks_after_short_interruption(tmp_path, case):
    report = evaluate(synthetic_frames(case, count=90), (160, 120, 320, 240), Settings(), tmp_path / "tracks.jsonl", recover=True)
    assert report["states"].get("reacquired", 0) == 1
    assert report["states"].get("lost", 0) == 0
    assert report["ground_truth_point_error_px"]["p95"] < 1
    assert report["state_processing"]["paused"]["samples"] > 0


def test_recovery_retains_repeating_texture_limitation(tmp_path):
    report = evaluate(synthetic_frames("alias"), (160, 120, 320, 240), Settings(), tmp_path / "alias.jsonl", recover=True)
    assert report["ground_truth_point_error_px"]["p95"] > 10


def test_recovery_never_reports_cached_reference_points_as_observations():
    from surface_replay import RecoveringPatchTracker
    image, stamp, _, _ = next(synthetic_frames("motion"))
    tracker = RecoveringPatchTracker((160, 120, 320, 240))
    tracker.step(image, stamp)
    result = tracker.step(np.zeros_like(image), stamp + 30_000_000)
    assert result["reference_points"] > 0
    assert result["surviving_points"] == 0 and result["ids"] == []


@pytest.mark.parametrize("angle", [90, 150])
def test_keyframe_search_finds_large_rotations_without_restoring_material_ids(angle):
    from surface_match import MatchSettings, PatchMatcher
    rng = np.random.default_rng(172)
    image = np.zeros((480, 640), np.uint8)
    image[120:360,160:480] = cv2.GaussianBlur(rng.integers(0,256,(240,320),dtype=np.uint8),(3,3),0)
    matcher = PatchMatcher((160,120,320,240), MatchSettings(method="sift"))
    assert matcher.seed(image) >= 12
    transform = cv2.getRotationMatrix2D((320,240),angle,1)
    rotated = cv2.warpAffine(image,transform,(640,480))
    result = matcher.match(rotated,1_000_000_000)
    assert result["status"] == "candidate", result
    expected = np.array([[160,120],[480,120],[480,360],[160,360]]) @ transform[:,:2].T + transform[:,2]
    assert np.max(np.linalg.norm(np.array(result["candidate_polygon_px"])-expected,axis=1)) < 4
    assert result["material_identity_verified"] is False and result["motion_authority"] is False
    assert "ids" not in result


def test_keyframe_search_rejects_duplicate_texture_and_rate_limits():
    from surface_match import MatchSettings, PatchMatcher
    rng = np.random.default_rng(19)
    texture=cv2.GaussianBlur(rng.integers(0,256,(200,200),dtype=np.uint8),(3,3),0)
    image=np.zeros((400,800),np.uint8)
    image[100:300,100:300]=texture
    matcher=PatchMatcher((100,100,200,200), MatchSettings(method="sift"))
    matcher.seed(image)
    ambiguous=image.copy()
    ambiguous[100:300,500:700]=texture
    matcher.search_level = 2  # Include both copies in the ambiguity check.
    assert matcher.match(ambiguous,1_000_000_000)["status"] != "candidate"
    assert matcher.match(ambiguous,1_050_000_000)["reason"] == "rate_limited"
    assert matcher.match(ambiguous,900_000_000)["reason"] == "timestamp_regression"


def test_advisory_match_does_not_reactivate_lost_tracker(tmp_path):
    report=evaluate(synthetic_frames("occlusion",count=90),(160,120,320,240),Settings(),tmp_path/'out.jsonl',match=True)
    rows=[json.loads(s) for s in (tmp_path/'out.jsonl').read_text().splitlines()]
    assert report['final_reason']=='insufficient_tracks'
    assert any(r.get('keyframe_match',{}).get('status')=='candidate' for r in rows[60:])
    assert all(r['status']=='lost' and not r['ids'] for r in rows[60:])


def test_expensive_search_defers_next_attempt_to_cpu_budget(monkeypatch):
    import surface_match
    rng = np.random.default_rng(4)
    im = rng.integers(0, 256, (240, 320), dtype=np.uint8)
    matcher = surface_match.PatchMatcher((40, 40, 200, 160))
    matcher.seed(im)
    clocks = iter([10.0, 10.2])
    monkeypatch.setattr(surface_match.time, "process_time", lambda: next(clocks))
    result = matcher.match(im, 1_000_000_000)
    assert result["next_attempt_after_ms"] == pytest.approx(2000)
    assert matcher.match(im, 1_600_000_000)["status"] == "skipped"
    assert matcher.match(im, 1_500_000_000)["reason"] == "timestamp_regression"


def test_matcher_rejects_invalid_workload_settings():
    from surface_match import MatchSettings, PatchMatcher
    with pytest.raises(ValueError, match="bounds"):
        PatchMatcher((0,0,100,100), MatchSettings(max_edge=10000))


@pytest.mark.parametrize("method", ["orb", "sift"])
def test_turn_suite_reports_candidate_geometry_error_without_restoring_tracks(tmp_path, method):
    report = evaluate(synthetic_frames("turn", count=90), (160,120,320,240), Settings(), tmp_path/'turn.jsonl', match=method)
    assert report['keyframe_match_states'].get('candidate', 0) > 0
    assert report['keyframe_corner_error_px']['p95'] < 4
    # The keyframe result remains independent of local optical-flow state.
    assert report['ground_truth_point_error_px']['p95'] > 10


def test_keyframe_identity_accepts_roi_on_image_boundary():
    from surface_match import MatchSettings, PatchMatcher
    image = next(synthetic_frames('motion'))[0]
    matcher = PatchMatcher((0, 0, 640, 480), MatchSettings(method='sift'))
    matcher.seed(image)
    result = matcher.match(image, 1_000_000_000)
    assert result['status'] == 'candidate', result
    assert np.max(np.abs(np.array(result['candidate_polygon_px']) -
                         [[0, 0], [640, 0], [640, 480], [0, 480]])) < 0.5


@pytest.mark.parametrize('method', ['orb', 'sift'])
def test_scene_features_obey_spatial_capacity(method):
    from surface_match import MatchSettings, PatchMatcher
    image = next(synthetic_frames('motion'))[0]
    matcher = PatchMatcher((160, 120, 320, 240), MatchSettings(method=method))
    keys, descriptors = matcher._scene_features(image)
    counts = np.zeros((3, 4), dtype=int)
    for key in keys:
        counts[min(2, int(key.pt[1] * 3 / 480)), min(3, int(key.pt[0] * 4 / 640))] += 1
    assert np.all(counts > 0)
    assert np.all(counts <= 100)
    assert len(descriptors) == len(keys) <= 1200


def test_search_expands_one_stage_per_budgeted_attempt_and_wraps():
    from surface_match import PatchMatcher
    image = next(synthetic_frames('motion'))[0]
    matcher = PatchMatcher((250, 180, 100, 100))
    matcher.seed(image)
    blank = np.zeros_like(image)
    results = [matcher.match(blank, (i + 1) * 10_000_000_000) for i in range(4)]
    assert [r['search_stage'] for r in results] == ['local', 'expanded', 'global', 'local']
    assert [r['search_window_px'][2:] for r in results[:3]] == [[150, 150], [300, 300], [640, 480]]
    level = matcher.search_level
    assert matcher.match(blank, 40_000_000_001)['status'] == 'skipped'
    assert matcher.search_level == level


def test_local_search_keeps_native_detail_and_source_coordinates():
    from surface_match import MatchSettings, PatchMatcher
    rng = np.random.default_rng(20)
    image = np.zeros((1600, 2400), dtype=np.uint8)
    image[900:1100, 1700:1900] = cv2.GaussianBlur(rng.integers(0, 256, (200, 200), dtype=np.uint8), (3, 3), 0)
    matcher = PatchMatcher((1700, 900, 200, 200), MatchSettings(method='sift'))
    matcher.seed(image)
    transform = cv2.getRotationMatrix2D((1800, 1000), 90, 1)
    transform[:, 2] += (50, 20)
    moved = cv2.warpAffine(image, transform, (2400, 1600))
    result = matcher.match(moved, 1_000_000_000)
    assert result['status'] == 'candidate', result
    assert result['search_scale'] == 1 and result['search_stage'] == 'local'
    expected = np.array([[1700,900], [1900,900], [1900,1100], [1700,1100]]) @ transform[:,:2].T + transform[:,2]
    assert np.max(np.linalg.norm(np.array(result['candidate_polygon_px']) - expected, axis=1)) < 2
    assert matcher.search_level == 0
    assert np.allclose(matcher.search_polygon, result['candidate_polygon_px'])
    assert not result['material_identity_verified'] and not result['motion_authority']


def test_local_window_clips_at_image_edge():
    from surface_match import PatchMatcher
    image = next(synthetic_frames('motion'))[0]
    matcher = PatchMatcher((0, 0, 100, 100))
    matcher.seed(image)
    assert matcher._search_window(image.shape) == (0, 0, 125, 125)


def test_global_fallback_finds_distant_patch_and_recenters_local_search():
    from surface_match import MatchSettings, PatchMatcher
    rng = np.random.default_rng(11)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (100, 100), dtype=np.uint8), (3, 3), 0)
    image = np.zeros((640, 960), dtype=np.uint8)
    image[200:300, 100:200] = texture
    matcher = PatchMatcher((100, 200, 100, 100), MatchSettings(method='sift'))
    matcher.seed(image)
    moved = np.zeros_like(image)
    moved[200:300, 700:800] = texture
    results = [matcher.match(moved, (i + 1) * 10_000_000_000) for i in range(3)]
    assert [r['search_stage'] for r in results] == ['local', 'expanded', 'global']
    assert all(r['status'] != 'candidate' for r in results[:2])
    assert results[2]['status'] == 'candidate', results[2]
    assert not results[2]['material_identity_verified']
    next_result = matcher.match(moved, 40_000_000_000)
    assert next_result['search_stage'] == 'local'
    assert next_result['search_window_px'][0] > 600
    assert next_result['status'] == 'candidate'


def bank_view(angle=0, shift=(0, 0)):
    image = next(synthetic_frames('motion'))[0]
    transform = cv2.getRotationMatrix2D((320, 240), angle, 1)
    transform[:, 2] += shift
    anchors = np.array([(x, y) for y in range(140, 341, 20) for x in range(180, 461, 20)], np.float32)
    points = anchors @ transform[:, :2].T + transform[:, 2]
    tracking = {'status': 'tracked', 'ids': list(range(len(anchors))), 'points_px': points.tolist(), 'sharpness_ratio': 1}
    return cv2.warpAffine(image, transform, (640, 480)), tracking, anchors, transform


def test_bank_admission_is_bounded_and_keeps_original_coordinates():
    from surface_match import KeyframeBank, MatchSettings
    bank = KeyframeBank((160, 120, 320, 240), MatchSettings(method='sift'))
    bank.seed(bank_view()[0])
    first = bank.references[0]
    for i, angle in enumerate((8, 16, 24)):
        image, tracking, anchors, _ = bank_view(angle)
        result = bank.match(image, (i+1)*10_000_000_000, tracking, anchors)
        assert result['status'] == 'admitted', result
        assert not result['material_identity_verified'] and not result['motion_authority']
    assert len(bank.references) == 3 and bank.references[0] is first
    image, _, _, matrix = bank_view(35)
    result = bank.match(image, 40_000_000_000)
    assert result['status'] == 'candidate', result
    assert result['reference_index'] == 2
    expected = np.array([[160,120],[480,120],[480,360],[160,360]]) @ matrix[:,:2].T + matrix[:,2]
    assert np.max(np.linalg.norm(np.array(result['candidate_polygon_px'])-expected,axis=1)) < 2


def test_bank_rejects_flow_that_disagrees_with_fixed_reference():
    from surface_match import KeyframeBank, MatchSettings
    bank = KeyframeBank((160, 120, 320, 240), MatchSettings(method='sift'))
    image, tracking, anchors, _ = bank_view(8)
    bank.seed(bank_view()[0])
    tracking['points_px'] = (np.array(tracking['points_px']) + (20, 0)).tolist()
    result = bank.match(image, 1_000_000_000, tracking, anchors)
    assert result['reason'] == 'flow_descriptor_disagreement', result
    assert len(bank.references) == 1


@pytest.mark.parametrize('status,sharpness', [('paused', 1), ('lost', 1), ('tracked', 0.2)])
def test_bank_never_admits_unavailable_or_blurred_flow(status, sharpness):
    from surface_match import KeyframeBank, MatchSettings
    bank = KeyframeBank((160, 120, 320, 240), MatchSettings(method='sift'))
    bank.seed(bank_view()[0])
    image, tracking, anchors, _ = bank_view(8)
    tracking.update(status=status, sharpness_ratio=sharpness)
    result = bank.match(image, 1_000_000_000, tracking, anchors)
    assert result['status'] != 'admitted'
    assert len(bank.references) == 1


def test_bank_admission_cost_uses_shared_budget(monkeypatch):
    import surface_match
    bank = surface_match.KeyframeBank((160,120,320,240), surface_match.MatchSettings(method='sift'))
    bank.seed(bank_view()[0])
    image, tracking, anchors, _ = bank_view(8)
    clocks=iter((1.0,1.2))
    monkeypatch.setattr(surface_match.time, 'process_time', lambda: next(clocks))
    result=bank.match(image, 1_000_000_000, tracking, anchors)
    assert result['status']=='admitted'
    assert result['next_attempt_after_ms']==pytest.approx(2000)
    assert bank.match(image, 1_500_000_000, tracking, anchors)['status']=='skipped'
    assert bank.match(image, 1_400_000_000, tracking, anchors)['reason']=='timestamp_regression'


def test_bank_replay_cannot_restore_lost_material_tracks(tmp_path, monkeypatch):
    import itertools

    import surface_match
    clocks = itertools.count(0, 0.001)
    monkeypatch.setattr(surface_match.time, "process_time", lambda: next(clocks))
    report = evaluate(synthetic_frames('occlusion',count=120), (160,120,320,240), Settings(),
                      tmp_path/'bank.jsonl', recover=True, match='sift', keyframes=True)
    rows=[json.loads(line) for line in (tmp_path/'bank.jsonl').read_text().splitlines()]
    assert report['keyframe_match_states'].get('admitted', 0)>0
    assert any(r['keyframe_match']['status']=='candidate' for r in rows[80:])
    assert all(r['status']=='lost' and r['ids']==[] for r in rows[80:])
    assert all(r['keyframe_match'].get('keyframes', 1)<=3 for r in rows)


def test_intermediate_view_recovers_scale_and_rotation_missed_by_initial_orb():
    from surface_match import KeyframeBank, MatchSettings, PatchMatcher
    base, tracking, anchors, _ = bank_view()
    settings = MatchSettings(method='orb')
    bank = KeyframeBank((160,120,320,240), settings)
    bank.seed(base)
    intermediate = cv2.getRotationMatrix2D((320,240), 30, 0.8)
    image = cv2.warpAffine(base, intermediate, (640,480))
    tracking['points_px'] = (anchors @ intermediate[:,:2].T + intermediate[:,2]).tolist()
    assert bank.match(image, 1_000_000_000, tracking, anchors)['status']=='admitted'
    final = cv2.getRotationMatrix2D((320,240), 60, 0.55)
    image = cv2.warpAffine(base, final, (640,480))
    baseline = PatchMatcher((160,120,320,240), settings)
    baseline.seed(base)
    assert baseline.match(image, 10_000_000_000)['status']!='candidate'
    result = bank.match(image, 10_000_000_000)
    assert result['status']=='candidate' and result['reference_index']==1, result
    expected = np.array([[160,120],[480,120],[480,360],[160,360]]) @ final[:,:2].T + final[:,2]
    assert np.max(np.linalg.norm(np.array(result['candidate_polygon_px'])-expected,axis=1)) < 4


def test_scale_turn_bank_uses_observed_flow_with_bounded_candidate_error(tmp_path, monkeypatch):
    import itertools

    import surface_match
    clocks = itertools.count(0, 0.001)
    monkeypatch.setattr(surface_match.time, 'process_time', lambda: next(clocks))
    report = evaluate(synthetic_frames('scale_turn', count=120), (160,120,320,240), Settings(),
                      tmp_path/'scale.jsonl', recover=True, match='orb', keyframes=True)
    assert report['peak_keyframes'] == 3
    assert report['keyframe_match_states'].get('candidate', 0) > 0
    assert report['keyframe_corner_error_px']['max'] < 4
    assert report['ground_truth_point_error_px']['p95'] < 2
    assert report['first_tracking_unavailable'] is not None
