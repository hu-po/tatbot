"""Pure synthetic tests for the vision-only EE estimator.

Run with:
  uvx --with pytest --with numpy --with scipy --with opencv-python \
    pytest -q scripts/tests/test_ee_fiducial.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/vision"))

from ee_fiducial import (  # noqa: E402
    WRIST_IDS,
    CameraModel,
    Detection,
    DetectorConfig,
    EstimatorConfig,
    MultiCameraEstimator,
    WristLayout,
    WristTagDetector,
    invert,
    matrix_from_pose,
    rotation_distance_deg,
    transform_from_vector,
    transform_points,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_synth import look_at_rotation  # noqa: E402


def _camera(name, position, target=(0.0, 0.0, 1.0)):
    rotation = look_at_rotation(position, target)
    return CameraModel(
        name=name,
        width=1280,
        height=720,
        intrinsic=np.array([[900.0, 0.0, 640.0], [0.0, 895.0, 360.0], [0.0, 0.0, 1.0]]),
        distortion=np.array([0.03, -0.01, 0.001, -0.001, 0.002]),
        world_from_camera=matrix_from_pose(rotation, position),
    )


def _layout():
    front = np.eye(4)
    front[:3, 3] = [-0.04, 0.0, 0.0]
    side = np.eye(4)
    side[:3, :3] = transform_from_vector(np.r_[[0.0, np.pi / 2, 0.0], np.zeros(3)])[:3, :3]
    side[:3, 3] = [0.04, 0.0, 0.0]
    return WristLayout(0.056, {0: front, 3: side}, "synthetic-layout")


def _detections(cameras, layout, world_from_ee, *, timestamp_ns=1_000_000_000):
    out = []
    for camera in cameras.values():
        for tag_id in layout.ee_from_tag:
            world = transform_points(world_from_ee, layout.corners_ee(tag_id))
            pixels, depths = camera.project(world)
            if np.all(depths > 0):
                out.append(Detection(camera.name, tag_id, pixels, timestamp_ns, 60.0))
    return out


def _pose_error(estimate, truth):
    delta = invert(truth) @ estimate
    return np.linalg.norm(delta[:3, 3]), rotation_distance_deg(estimate, truth)


def test_camera_scaling_preserves_projection_geometry():
    camera = _camera("camera1", [-0.3, 0.0, 0.1])
    scaled = camera.scaled(640, 360)
    assert np.allclose(scaled.intrinsic[:2], camera.intrinsic[:2] * 0.5)
    try:
        camera.scaled(640, 400)
    except ValueError as error:
        assert "not a uniform scale" in str(error)
    else:
        raise AssertionError("cropped dimensions must not silently scale a calibration")


def test_detector_roi_returns_full_frame_corner_coordinates():
    frame = np.full((400, 600, 3), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 120)
    frame[130:250, 320:440] = marker[:, :, None]
    detector = WristTagDetector(_layout(), DetectorConfig(scale=1.0, min_side_px=20.0))
    full = detector.detect("camera1", frame, 1)
    cropped = detector.detect("camera1", frame, 1, roi_xyxy=(260, 70, 500, 310))
    assert [item.tag_id for item in full] == [0]
    assert [item.tag_id for item in cropped] == [0]
    assert np.allclose(cropped[0].corners_px, full[0].corners_px, atol=0.1)


def test_multicamera_estimate_recovers_rigid_wrist_pose():
    cameras = {
        "camera1": _camera("camera1", [-0.35, -0.1, 0.15]),
        "camera2": _camera("camera2", [0.35, -0.05, 0.2]),
        "camera3": _camera("camera3", [0.0, 0.4, 0.25]),
    }
    layout = _layout()
    truth = transform_from_vector(np.array([0.08, -0.04, 0.12, 0.01, -0.02, 1.0]))
    estimate = MultiCameraEstimator(cameras, layout).estimate(
        _detections(cameras, layout, truth), 1_000_000_000
    )
    assert estimate.status == "measured", estimate.reason
    translation, rotation = _pose_error(estimate.world_from_ee, truth)
    assert translation < 1e-6
    assert rotation < 1e-4


def test_bad_camera_tag_source_is_rejected_without_poisoning_pose():
    cameras = {
        "camera1": _camera("camera1", [-0.35, -0.1, 0.15]),
        "camera2": _camera("camera2", [0.35, -0.05, 0.2]),
        "camera3": _camera("camera3", [0.0, 0.4, 0.25]),
    }
    layout = _layout()
    truth = transform_from_vector(np.array([0.03, 0.01, -0.05, 0.0, 0.01, 1.0]))
    detections = _detections(cameras, layout, truth)
    detections = [
        Detection(item.camera, item.tag_id, item.corners_px + 35.0, item.timestamp_ns, item.side_px)
        if item.camera == "camera3" and item.tag_id == 3
        else item
        for item in detections
    ]
    estimate = MultiCameraEstimator(cameras, layout).estimate(detections, 1_000_000_000)
    assert estimate.status == "measured", estimate.reason
    assert "camera3:tag3" in estimate.rejected_sources
    translation, rotation = _pose_error(estimate.world_from_ee, truth)
    assert translation < 0.002
    assert rotation < 0.2


def test_duplicate_wrist_id_in_one_camera_fails_closed():
    cameras = {
        "camera1": _camera("camera1", [-0.35, -0.1, 0.15]),
        "camera2": _camera("camera2", [0.35, -0.05, 0.2]),
    }
    layout = _layout()
    truth = transform_from_vector(np.array([0.01, 0.02, -0.03, 0.0, 0.0, 1.0]))
    detections = _detections(cameras, layout, truth)
    original = next(item for item in detections if item.camera == "camera1" and item.tag_id == 0)
    detections.append(
        Detection(
            original.camera,
            original.tag_id,
            original.corners_px + 100.0,
            original.timestamp_ns,
            original.side_px,
        )
    )
    estimate = MultiCameraEstimator(cameras, layout).estimate(detections, 1_000_000_000)
    assert estimate.status == "rejected"
    assert "ambiguous duplicate wrist IDs" in estimate.reason


def test_lone_planar_tag_is_never_accepted_as_measurement():
    cameras = {"camera1": _camera("camera1", [0.0, 0.0, 0.0])}
    layout = _layout()
    truth = transform_from_vector(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    detections = [item for item in _detections(cameras, layout, truth) if item.tag_id == 0]
    estimate = MultiCameraEstimator(cameras, layout).estimate(detections, 1_000_000_000)
    assert estimate.status == "rejected"
    assert estimate.world_from_ee is None


def test_single_tag_multiview_requires_an_existing_vision_track():
    cameras = {
        "camera1": _camera("camera1", [-0.35, -0.1, 0.15]),
        "camera2": _camera("camera2", [0.35, -0.05, 0.2]),
    }
    layout = _layout()
    truth = transform_from_vector(np.array([0.01, -0.02, 0.03, 0.0, 0.0, 1.0]))
    single_tag = [item for item in _detections(cameras, layout, truth) if item.tag_id == 0]
    estimator = MultiCameraEstimator(cameras, layout)
    acquisition = estimator.estimate(single_tag, 1_000_000_000)
    tracked = estimator.estimate(single_tag, 1_000_000_000, initial_world_from_ee=truth)
    assert acquisition.status == "rejected"
    assert "multi-tag consensus" in acquisition.reason
    assert tracked.status == "measured", tracked.reason


def test_one_camera_accepts_two_non_coplanar_tag_faces():
    cameras = {"camera1": _camera("camera1", [0.0, -0.3, 0.1])}
    layout = _layout()
    truth = transform_from_vector(np.array([0.02, -0.03, 0.04, 0.0, 0.0, 1.0]))
    estimate = MultiCameraEstimator(cameras, layout).estimate(
        _detections(cameras, layout, truth), 1_000_000_000
    )
    assert estimate.status == "measured", estimate.reason
    translation, rotation = _pose_error(estimate.world_from_ee, truth)
    assert translation < 1e-5
    assert rotation < 1e-3


def test_two_cameras_can_bootstrap_when_each_sees_a_different_tag():
    cameras = {
        "camera1": _camera("camera1", [-0.35, -0.1, 0.15]),
        "camera2": _camera("camera2", [0.35, -0.05, 0.2]),
    }
    layout = _layout()
    truth = transform_from_vector(np.array([0.05, 0.02, -0.04, 0.0, 0.0, 1.0]))
    all_detections = _detections(cameras, layout, truth)
    sparse = [
        item for item in all_detections if (item.camera, item.tag_id) in {("camera1", 0), ("camera2", 3)}
    ]
    estimate = MultiCameraEstimator(cameras, layout).estimate(sparse, 1_000_000_000)
    assert estimate.status == "measured", estimate.reason
    translation, rotation = _pose_error(estimate.world_from_ee, truth)
    assert translation < 1e-5
    assert rotation < 1e-3


def test_timestamp_aware_motion_fit_beats_static_projection():
    cameras = {
        "camera1": _camera("camera1", [-0.35, -0.1, 0.15]),
        "camera2": _camera("camera2", [0.35, -0.05, 0.2]),
        "camera3": _camera("camera3", [0.0, 0.4, 0.25]),
    }
    layout = _layout()
    reference_ns = 2_000_000_000
    truth = transform_from_vector(np.array([0.04, -0.02, 0.06, 0.0, 0.0, 1.0]))
    velocity = np.array([0.0, 0.0, 0.8, 0.25, -0.1, 0.0])
    detections = []
    for camera, dt_ms in zip(cameras.values(), [-18, 0, 18], strict=True):
        dt = dt_ms / 1000.0
        at_time = transform_from_vector(np.r_[velocity[:3] * dt, velocity[3:] * dt]) @ truth
        for item in _detections(
            {camera.name: camera}, layout, at_time, timestamp_ns=reference_ns + dt_ms * 1_000_000
        ):
            detections.append(item)
    dynamic = MultiCameraEstimator(cameras, layout).estimate(
        detections, reference_ns, initial_world_from_ee=truth
    )
    static = MultiCameraEstimator(
        cameras, layout, EstimatorConfig(motion_compensation=False, max_total_rmse_px=20.0)
    ).estimate(detections, reference_ns, initial_world_from_ee=truth)
    assert dynamic.status == "measured", dynamic.reason
    assert dynamic.reprojection_rmse_px < static.reprojection_rmse_px * 0.35
    assert np.linalg.norm(dynamic.twist[3:] - velocity[3:]) < 0.05


def test_repository_wrist_inventory_is_four_tags_and_calibrated():
    assert {3, 6, 7, 8} == WRIST_IDS
    layout = WristLayout.load(ROOT / "config/wrist_tags_measured.json")
    assert layout.parent_frame == "right/gripper_left"
    assert set(layout.ee_from_tag) == WRIST_IDS


def test_calibrated_four_tag_layout_loads(tmp_path):
    source = ROOT / "config/wrist_tags_measured.json"
    data = json.loads(source.read_text())
    data["calibration_status"] = "calibrated"
    layout_path = tmp_path / "four-tags.json"
    layout_path.write_text(json.dumps(data))
    layout = WristLayout.load(layout_path)
    assert set(layout.ee_from_tag) == {3, 6, 7, 8}
