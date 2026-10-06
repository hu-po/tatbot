"""Comparison truth is withheld and dense geometry cannot certify material identity."""
from pathlib import Path

import numpy as np
import pytest
from surface_attachment_benchmark import compare_frames, material_error, summarize
from surface_attachment_inputs import pose, render_fixture


def test_metric_score_uses_original_material_and_frame_direction():
    _, a = render_fixture(world_from_camera=pose(xyz=(.001, .002, 0)))
    _, b = render_fixture(world_from_material=pose(xyz=(.004, 0, .32)),
                          world_from_camera=pose(xyz=(-.002, .003, 0)))
    correct = b["camera_from_material"] @ np.linalg.inv(a["camera_from_material"])
    assert material_error(correct, a, b)["rms_mm"] < 1e-10
    assert material_error(np.linalg.inv(correct), a, b)["rms_mm"] > 10


def test_truth_changes_score_but_not_observer_output():
    first, a = render_fixture()
    second, b = render_fixture(stamp=1_033_333_333, sequence=1,
                               world_from_material=pose(xyz=(.002, 0, .32)))
    ordinary = compare_frames([(first, a, True), (second, b, True)], dense=False)
    changed = dict(b)
    changed["camera_from_material"] = b["camera_from_material"].copy()
    changed["camera_from_material"][0, 3] += .2
    poisoned = compare_frames([(first, a, True), (second, changed, True)], dense=False)
    left, right = ordinary[1]["methods"]["sparse"], poisoned[1]["methods"]["sparse"]
    assert left["accepted"] and right["accepted"]
    assert np.allclose(left["transform_reference_to_camera"], right["transform_reference_to_camera"])
    assert left["original_material_error"]["rms_mm"] < 1
    assert right["original_material_error"]["rms_mm"] > 190
    assert not left["excessive_error_accept"]
    assert right["excessive_error_accept"]
    assert left["material_support"] == right["material_support"]
    assert left["material_support"]["point_count"] > 0
    assert left["material_support"]["motion_authority"] is False


def test_dense_plane_alignment_false_accept_is_reported():
    pytest.importorskip("open3d")
    pytest.importorskip("threadpoolctl")
    first, a = render_fixture(appearance="smooth")
    second, b = render_fixture(appearance="smooth", stamp=1_033_333_333, sequence=1,
                               world_from_material=pose(xyz=(.005, 0, .32)))
    rows = compare_frames([(first, a, False), (second, b, False)])
    dense = rows[1]["methods"]["point_to_plane"]
    assert dense["accepted"] and dense["false_accept"]
    assert dense["original_material_error"]["rms_mm"] > 3
    assert not rows[1]["methods"]["sparse"]["accepted"]
    assert summarize(rows)["point_to_plane"]["false_accepts"] == 2


def test_recording_does_not_receive_ground_truth_accuracy():
    first, _ = render_fixture()
    first["evidence_kind"] = "recorded-rgbd"
    rows = compare_frames([(first, None, None)], dense=False)
    result = rows[0]["methods"]["sparse"]
    assert result["accepted"]
    assert result["original_material_error"] is None
    assert result["sensor_consistency"]["status"] == "available"


def test_dense_comparison_counts_geometry_preprocessing(monkeypatch):
    pytest.importorskip("open3d")
    import surface_attachment_benchmark as benchmark
    import surface_rgbd
    clock = [0.0]
    def prepare(*args):
        clock[0] += .020
        return object()
    def register(*args):
        clock[0] += .010
        return {"status": "candidate", "transform_previous_to_current": np.eye(4).tolist()}
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(benchmark, "_dense_cloud", prepare)
    monkeypatch.setattr(surface_rgbd, "register", register)
    results = benchmark._dense_results(object(), {}, object(), 30.0)
    for result in results.values():
        assert result["processing_ms"] == pytest.approx(60.)
        assert result["registration_ms"] == pytest.approx(10.)
        assert result["current_preprocessing_ms"] == pytest.approx(20.)
        assert result["reference_preprocessing_ms"] == pytest.approx(30.)


@pytest.mark.parametrize("inside_repository", [False, True])
def test_comparison_output_refuses_overwrite_and_repository(tmp_path, inside_repository):
    from surface_attachment_benchmark import main
    path = Path(__file__).resolve().parents[2]/"forbidden-comparison-artifact" if inside_repository else tmp_path
    with pytest.raises(SystemExit) as error:
        main(["--output", str(path)])
    assert error.value.code == 2
    if not inside_repository:
        assert not list(tmp_path.iterdir())


def test_colored_competitor_keeps_original_rgb_channels(monkeypatch):
    pytest.importorskip("open3d")
    import surface_attachment_benchmark as benchmark
    import surface_rgbd
    frame, _ = render_fixture(width=40, height=30, focal_px=60)
    frame["bgr"][:] = [10, 20, 30]
    monkeypatch.setattr(surface_rgbd, "cloud", lambda points, colors, settings: colors)
    colors = benchmark._dense_cloud(frame, object())
    assert np.allclose(colors, np.array([30, 20, 10])/255)
    without_color = {k: v for k, v in frame.items() if k != "bgr"}
    fallback = benchmark._dense_cloud(without_color, object())
    assert np.allclose(fallback[:, 0], fallback[:, 1])
    assert np.allclose(fallback[:, 1], fallback[:, 2])
