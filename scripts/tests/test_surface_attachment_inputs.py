"""Physical projection and truth-separation checks for offline attachment inputs."""

import numpy as np
import pytest
from surface_attachment_inputs import material_height, pose, render_fixture


@pytest.mark.parametrize("shape", ["plane", "curved"])
def test_sparse_border_keeps_material_center_clear_and_recovers_from_pixels(shape):
    from surface_attachment import SurfaceAttachmentObserver

    observer = SurfaceAttachmentObserver("paper", "original", (0, 0, 320, 240))
    reference_digest = None
    reference_points = None
    for index, options in enumerate(({}, {"world_from_material": pose(xyz=(.0005, 0, .32))},
                                     {"dark": True}, {"world_from_material": pose(xyz=(.0005, 0, .32))})):
        frame, truth = render_fixture(shape=shape, appearance="sparse-border", sequence=index,
                                      stamp=1_000_000_000+index*100_000_000, **options)
        result = observer.observe(frame, now_ns=frame["capture_timestamp_ns"])
        assert result["motion_authority"] is False
        if index == 2:
            assert not result["accepted"]
            continue
        uv = truth["pixel_material_points"]
        clear = (np.abs(uv[..., 0]) < .06) & (np.abs(uv[..., 1]) < .04)
        assert np.all(frame["rgbd"]["gray"][clear] == 215)
        assert .005 < np.mean(frame["rgbd"]["gray"] != 215) < .15
        assert result["accepted"]
        assert result["material_support"]["supported_count"] >= 12
        if reference_digest is None:
            reference_digest = result["reference_digest"]
            reference_points = observer.reference_points()
            anchors = np.asarray(list(reference_points.values()))
            assert np.all((np.abs(anchors[:, 0]) >= .06) | (np.abs(anchors[:, 1]) >= .04))
        assert result["reference_digest"] == reference_digest
        assert observer.reference_points() == reference_points
        matrix = np.asarray(result["transform_reference_to_camera"])
        expected = truth["camera_from_material"] @ np.linalg.inv(pose(xyz=(0, 0, .32)))
        withheld = truth["material_points"] + np.array([0, 0, .32])
        error = np.linalg.norm(withheld @ matrix[:3, :3].T + matrix[:3, 3]
                               - (withheld @ expected[:3, :3].T + expected[:3, 3]), axis=1)
        assert error.max() < .0015
    assert result["state"] == "recovered"


@pytest.mark.parametrize("shape", ["plane", "curved"])
@pytest.mark.parametrize("motion", ["surface", "camera", "both"])
def test_metric_ray_intersections_and_frame_direction(shape, motion):
    wm = pose(xyz=(0.004, -0.003, 0.32), angles_deg=(3, -2, 1)) if motion != "camera" else pose(xyz=(0, 0, .32))
    wc = pose(xyz=(-0.006, .002, -.008), angles_deg=(-1, 2, 3)) if motion != "surface" else np.eye(4)
    frame, truth = render_fixture(shape=shape, world_from_material=wm, world_from_camera=wc)
    rgbd = frame["rgbd"]
    points_camera = rgbd["rays"] * rgbd["depth_m"][..., None]
    cm = truth["camera_from_material"]
    predicted = truth["pixel_material_points"] @ cm[:3, :3].T + cm[:3, 3]
    valid = truth["surface_visible"]
    # Z16 quantization, not a physical sensor accuracy claim.
    assert np.linalg.norm(predicted[valid] - points_camera[valid], axis=1).max() < 0.00006
    assert np.allclose(wc @ cm, wm)
    assert valid.mean() > .8
    p = truth["pixel_material_points"][valid]
    assert np.max(np.abs(p[:, 2] - material_height(p[:, 0], p[:, 1], shape)[0])) < 1e-8


def test_true_surface_motion_is_not_a_two_dimensional_depth_warp():
    before, _ = render_fixture(shape="curved")
    after, truth = render_fixture(shape="curved", world_from_material=pose(xyz=(0, 0, .34), angles_deg=(4, 0, 0)))
    assert np.median(after["rgbd"]["depth_m"] - before["rgbd"]["depth_m"]) > .018
    assert np.allclose(truth["world_from_material"][:3, 3], [0, 0, .34])


def test_curved_patch_has_two_curvatures_and_deformation_preserves_material_coordinates():
    _, reference = render_fixture(shape="curved")
    _, deformed = render_fixture(shape="curved", deformation_m=.012)
    p, q = reference["material_points"], deformed["material_points"]
    assert np.array_equal(p[:, :2], q[:, :2])
    assert (q[:, 2] - p[:, 2]).max() > .01
    h = .001
    def f(x, y):
        return material_height(np.array(x), np.array(y), "curved")[0]
    assert (f(h, 0)-2*f(0, 0)+f(-h, 0))/h**2 == pytest.approx(1.8)
    assert (f(0, h)-2*f(0, 0)+f(0, -h))/h**2 == pytest.approx(-1.0)


def test_estimator_input_contains_no_truth_or_moving_segmentation():
    frame, truth = render_fixture()
    forbidden = {"camera_from_material", "world_from_material", "world_from_camera",
                 "material_points", "pixel_material_points", "surface_visible", "mask"}
    assert not forbidden.intersection(frame)
    assert not forbidden.intersection(frame["rgbd"])
    assert truth["ground_truth_kind"] == "generated_metric_ray_intersection"
    assert frame["evidence_kind"] == "synthetic-render"
    assert frame["capture_timestamp_ns"] == frame["stamp"]


def test_faults_change_sensor_data_without_replacing_reference_truth():
    original, truth = render_fixture()
    dark, _ = render_fixture(dark=True)
    occluded, _ = render_fixture(occlusion_fraction=1)
    holes, _ = render_fixture(depth_hole_fraction=.2)
    alternate, alternate_truth = render_fixture(appearance="alternate")
    assert dark["rgbd"]["gray"].max() == 0
    assert np.array_equal(dark["depth"], original["depth"])
    assert np.all(occluded["depth"] == 2000)
    assert .18 < np.mean(holes["depth"] == 0) < .22
    assert not np.array_equal(alternate["bgr"], original["bgr"])
    assert np.array_equal(alternate_truth["material_points"], truth["material_points"])
    repeated, _ = render_fixture()
    assert np.array_equal(original["bgr"], repeated["bgr"])


@pytest.mark.parametrize("kwargs", [{"deformation_m": float("nan")}, {"width": 5000},
                                   {"occlusion_fraction": 1.1}, {"blur_sigma": -1},
                                   {"world_from_camera": np.zeros((4, 4))}, {"shape": "cylinder"}])
def test_invalid_rendering_settings_refuse(kwargs):
    with pytest.raises(ValueError):
        render_fixture(**kwargs)


def recording_fixture(root, *, color_skew=0, bad_hash=False, missing_units=False,
                      calibration_change=False, distortion="None"):
    import hashlib
    import json
    frame, _ = render_fixture(width=40, height=30, focal_px=60)
    for kind in ("color", "depth"):
        directory = root/("camera_"+kind)
        directory.mkdir()
        rows = []
        for i in range(3):
            raw = frame["bgr"] if kind == "color" else frame["depth"]
            payload = raw.tobytes()
            (directory/str(i)).write_bytes(payload)
            intr = frame["metadata"]["attributes"]["intrinsics"] | {"distortion_model": distortion}
            attributes = {"intrinsics": json.dumps(intr)}
            if kind == "depth":
                attributes["aligned_to"] = "camera_color"
                if not missing_units:
                    attributes["depth_units_m"] = "0.0001"
            metadata = {"sensor_name": "camera_"+kind, "sequence": 40+i,
                        "timestamps": {"normalized_unix_ns": 1_000_000_000+i*33_333_333+(color_skew if kind == "color" else 0),
                                       "source_domain": "host_unix"}, "flags": [],
                        "calibration_id": "changed" if calibration_change and i > 0 else None,
                        "attributes": attributes, "profile": {"width": 40, "height": 30,
                                                              "format": "bgr8" if kind == "color" else "z16"}}
            rows.append({"metadata": metadata, "payload_file": str(i), "payload_bytes": len(payload),
                         "sha256": "bad" if bad_hash and i == 1 else hashlib.sha256(payload).hexdigest()})
        (directory/"frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))


def test_recorded_adapter_keeps_unknowns_and_capture_time(tmp_path, monkeypatch):
    import builtins
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "open3d":
            raise AssertionError("recording adapter must not import Open3D")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    from surface_attachment_inputs import recorded_frames
    recording_fixture(tmp_path)
    frames = list(recorded_frames(tmp_path, "camera", max_frames=2))
    assert len(frames) == 2
    assert all("invalid" not in f for f in frames)
    assert frames[0]["calibration_id"] is None
    assert frames[0]["producer_id"].startswith("recording-replay:")
    assert frames[0]["provenance"]["producer_build_sha"] is None
    assert frames[0]["provenance"]["robot_camera_calibration"] is None
    assert frames[0]["capture_timestamp_ns"] == 1_000_000_000
    assert frames[0]["sequence"] == 40
    assert frames[0]["evidence_kind"] == "recorded-rgbd"
    assert frames[0]["rgbd"]["depth_m"].shape == (30, 40)


def test_corrupt_payload_is_visible_without_new_reference(tmp_path):
    from surface_attachment_inputs import recorded_frames
    recording_fixture(tmp_path, bad_hash=True)
    frames = list(recorded_frames(tmp_path, "camera"))
    assert "invalid" not in frames[0]
    assert "checksum" in frames[1]["invalid"]
    assert "invalid" not in frames[2]
    assert frames[0]["producer_id"] == frames[2]["producer_id"]


@pytest.mark.parametrize("option,reason", [({"missing_units": True}, "depth_units_m"),
                                          ({"color_skew": 20_000_000}, "unpaired_rgbd"),
                                          ({"distortion": "Unknown"}, "unsupported deprojection")])
def test_recorded_metadata_faults_visible(tmp_path, option, reason):
    from surface_attachment_inputs import recorded_frames
    recording_fixture(tmp_path, **option)
    frames = list(recorded_frames(tmp_path, "camera"))
    assert reason in frames[0]["invalid"]


def test_changed_calibration_is_forwarded_for_observer_invalidation(tmp_path):
    from surface_attachment_inputs import recorded_frames
    recording_fixture(tmp_path, calibration_change=True)
    frames = list(recorded_frames(tmp_path, "camera"))
    assert frames[0]["calibration_id"] is None
    assert frames[1]["calibration_id"] == "changed"
    assert frames[0]["profile_id"] != frames[1]["profile_id"]


def test_missing_recording_manifests_are_visible(tmp_path):
    from surface_attachment_inputs import recorded_frames
    frames = list(recorded_frames(tmp_path, "camera"))
    assert len(frames) == 1
    assert frames[0]["invalid"].startswith("recording_manifest_invalid")
    assert frames[0]["capture_timestamp_ns"] is None


def test_recorded_overhead_native_depth_recovers_a_known_distorted_color_plane(monkeypatch):
    import cv2
    import surface_attachment_inputs as inputs
    from test_rgbd_geometry import alignment

    intr = {"schema": "tatbot.camera-intrinsics/1", "width": 640, "height": 360,
            "fx": 323., "fy": 322., "ppx": 323.4, "ppy": 181.3,
            "distortion_model": "BrownConrady",
            "distortion_coefficients": [-.052, .056, -.0008, -.00009, -.017]}
    y, x = np.indices((360, 640))
    pixels = np.stack((x, y), axis=-1).astype(float).reshape(-1, 1, 2)
    matrix = np.array([[323., 0., 323.4], [0., 322., 181.3], [0., 0., 1.]])
    # OpenCV inverse projection is an independent witness for the SDK ray path.
    coefficients = np.asarray(intr["distortion_coefficients"])
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
    if hasattr(cv2, "undistortPointsIter"):
        xy = cv2.undistortPointsIter(pixels, matrix, coefficients, None, None, criteria)
    else:
        xy = cv2.undistortPoints(pixels, matrix, coefficients, criteria=criteria)
    rays = np.concatenate((xy.reshape(360, 640, 2), np.ones((360, 640, 1))), axis=-1)
    truth = rays*(.35/(1-.12*rays[:, :, 0]+.08*rays[:, :, 1]))[:, :, None]
    rotation = cv2.Rodrigues(np.array([.07, -.09, .02]))[0]
    translation = np.array([-.06, .001, .004])
    native = (truth-translation)@rotation
    raw = np.rint(native[:, :, 2]/.0001).astype(np.uint16)
    model = alignment(intr, rotation, translation)
    metadata = {"attributes": {"alignment_calibration": model}}
    depth, color = {"metadata": metadata, "depth": raw}, {
        "metadata": metadata, "image": np.zeros((360, 640, 3), np.uint8)}
    monkeypatch.setattr(inputs, "_bounded_payload", lambda _, entry: entry)
    result = inputs._recorded_pixels(None, None, depth, color, .0001, intr, None, (.03, 2.))["rgbd"]
    measured = result["rays"]*result["depth_m"][:, :, None]
    assert np.max(np.linalg.norm(measured-truth, axis=2)) < .0001
    assert np.max(np.linalg.norm(rays*raw[:, :, None]*.0001-truth, axis=2)) > .02
