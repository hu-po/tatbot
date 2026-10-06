"""Known UV motion, wrong-pattern rejection, loss, recovery and depth separation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
import stencil_frame  # noqa: E402
import stencil_inputs  # noqa: E402
from stencil_evaluate import score  # noqa: E402
from stencil_features import Settings, fit  # noqa: E402
from stencil_inputs import image_frames  # noqa: E402
from stencil_tracking import StencilTracker  # noqa: E402

cv2.setNumThreads(1)


def reference(seed="tatbot-43"):
    return REPO/"docs/assets/stencil-frames"/seed/"tracking.json"


def render(seed="tatbot-43", shift=(0, 0), mirror=False):
    image = cv2.imread(str(reference(seed).with_name("stencil.png")), 0)
    height, width = image.shape
    corners = np.array([[-.5, -.5], [width-.5, -.5], [width-.5, height-.5], [-.5, height-.5]], np.float32)
    page = np.array([[170, 50], [390, 65], [415, 420], [145, 390]], np.float32)+shift
    if mirror:
        page = page[[1, 0, 3, 2]]
    homography = cv2.getPerspectiveTransform(corners, page.astype(np.float32))
    gray = cv2.warpPerspective(image, homography, (640, 480), borderValue=210)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    return gray, cv2.getPerspectiveTransform(uv, page.astype(np.float32))


def tracker(*seeds):
    return StencilTracker([reference(s) for s in seeds or ("tatbot-43",)], "skin-a")


@pytest.mark.parametrize("mirror", [False, True])
def test_automatic_identity_and_uv_mapping_without_roi(mirror):
    observer = tracker("tatbot-42", "tatbot-43", "tatbot-44")
    image, homography = render(mirror=mirror)
    observation = observer.observe(image, 1_000_000_000)
    expected = json.loads(reference().read_text())["pattern_id"]
    assert observation["image_tracking_valid"], observation["reason"]
    assert observation["pattern_id"] == expected
    assert observation["mirrored"] is mirror
    metrics = score([observation], [{"visible": True, "pattern_id": expected, "homography_uv_to_image": homography.tolist()}])
    assert metrics["image_error_rmse_px_p95"] < 3
    assert not observation["geometry_valid"] and not observation["motion_authority"]


def test_motion_depth_dropout_and_reference_recovery():
    observer = tracker()
    image, _ = render()
    first = observer.observe(image, 1_000_000_000)
    assert first["status"] == "detected"
    shifted, homography = render(shift=(7, 3))
    tracked = observer.observe(shifted, 1_033_333_333, depth_m=np.zeros_like(image, float))
    assert tracked["status"] == "tracked"
    assert tracked["image_tracking_valid"] and not tracked["geometry_valid"]
    assert tracked["depth_quality"]["valid_fraction"] == 0
    blank = observer.observe(np.full_like(image, 128), 1_066_666_666)
    assert not blank["image_tracking_valid"] and blank["homography_uv_to_image"] is None
    assert blank["landmarks"] == []
    recovered = observer.observe(shifted, 1_400_000_000)
    assert recovered["status"] == "reacquired"
    assert recovered["pattern_id"] == first["pattern_id"]
    metrics = score([tracked], [{"visible": True, "pattern_id": first["pattern_id"], "homography_uv_to_image": homography.tolist()}])
    assert metrics["image_error_rmse_px_p95"] < 3


def test_lost_flow_does_not_search_when_reference_search_is_deferred(monkeypatch):
    observer = tracker()
    image, _ = render()
    assert observer.observe(image, 1_000_000_000)["image_tracking_valid"]
    blank = np.full_like(image, 128)
    monkeypatch.setattr(observer, "_search", lambda *_: pytest.fail("deferred reference search ran"))
    lost = observer.observe(blank, 1_100_000_000, defer_search=True)
    assert lost["status"] == "lost"
    assert lost["reason"] == "reference_search_deferred"
    assert not lost["image_tracking_valid"]


def test_wrong_pattern_and_blank_do_not_inherit_identity():
    observer = tracker()
    wrong, _ = render("tatbot-44")
    assert not observer.observe(wrong, 1_000_000_000)["image_tracking_valid"]
    assert not observer.observe(np.full((480, 640), 128, np.uint8), 1_300_000_000)["image_tracking_valid"]
    assert observer.observe(render()[0], 1_600_000_000)["image_tracking_valid"]
    assert not observer.observe(wrong, 1_900_000_000)["image_tracking_valid"]


def test_reference_bank_never_silently_switches_physical_instance():
    observer = tracker("tatbot-43", "tatbot-44")
    first = observer.observe(render()[0], 1_000_000_000)
    assert first["image_tracking_valid"]
    observer.observe(np.zeros((480, 640), np.uint8), 1_100_000_000)
    wrong = observer.observe(render("tatbot-44")[0], 1_600_000_000)
    assert not wrong["image_tracking_valid"]
    assert wrong["reason"] == "different_pattern_visible"
    assert wrong["pattern_id"] == first["pattern_id"]


def test_same_pattern_swap_requires_current_decoded_print_mark(tmp_path):
    import argparse

    def marked(name, instance_id):
        parser = argparse.ArgumentParser()
        stencil_frame.add_arguments(parser)
        args = parser.parse_args(['--output', str(tmp_path/name), '--seed', 'tatbot-43',
                                  '--instance-id', instance_id])
        stencil_frame.validate(args)
        args.output = Path(args.output)
        stencil_frame.render(args, stencil_frame.build(args))
        return args.output/'tracking.json'

    first_ref = marked('first', '0123456789abcdef01234567')
    second_ref = marked('second', 'fedcba9876543210fedcba98')
    assert json.loads(first_ref.read_text())['pattern_id'] == json.loads(second_ref.read_text())['pattern_id']

    def camera_image(path):
        image = cv2.imread(str(path.with_name('stencil.png')), 0)
        height, width = image.shape
        source = np.array([[-.5, -.5], [width-.5, -.5],
                           [width-.5, height-.5], [-.5, height-.5]], np.float32)
        target = np.array([[170, 50], [390, 65], [415, 420], [145, 390]], np.float32)
        return cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target),
                                   (640, 480), borderValue=210)

    observer = StencilTracker([first_ref], 'skin-a')
    first_image = camera_image(first_ref)
    first = observer.observe(first_image, 1_000_000_000)
    assert first['image_tracking_valid'], first['reason']
    assert first['physical_instance_identity_verified']
    assert first['physical_instance_id'] == '0123456789abcdef01234567'
    swapped = observer.observe(camera_image(second_ref), 1_033_333_333)
    assert not swapped['image_tracking_valid']
    assert swapped['physical_instance_id'] is None
    assert swapped['reason'] == 'physical_instance_mismatch'
    recovered = observer.observe(first_image, 1_400_000_000)
    assert recovered['image_tracking_valid'] and recovered['physical_instance_identity_verified']

    covered = first_image.copy()
    covered[62:105, 170:390] = 210
    missing = observer.observe(covered, 1_433_333_333)
    assert not missing['physical_instance_identity_verified']
    assert not missing['image_tracking_valid']


def test_legacy_reference_tracks_artwork_without_claiming_physical_identity():
    observed = tracker().observe(render()[0], 1_000_000_000)
    assert observed['image_tracking_valid']
    assert observed['physical_instance_identity_verified'] is False
    assert observed['physical_instance_id'] is None
    assert observed['reference_physical_instance_id'] is None


def test_timestamp_regression_withholds_previous_coordinates():
    observer = tracker()
    image, _ = render()
    assert observer.observe(image, 1_000_000_000)["image_tracking_valid"]
    result = observer.observe(image, 900_000_000)
    assert result["reason"] == "timestamp_regression"
    assert not result["image_tracking_valid"] and result["landmarks"] == []


def test_collapsed_repeated_matches_are_not_page_evidence():
    uv = np.random.default_rng(4).random((40, 2))
    assert fit(uv, uv*.2 + [100, 100], (480, 640), Settings()) is None


def test_blur_withholds_and_recovers_reference_coordinates():
    observer = tracker()
    image, _ = render()
    first = observer.observe(image, 1_000_000_000)
    blurred = observer.observe(cv2.GaussianBlur(image, (41, 41), 0), 1_033_333_333)
    assert not blurred["image_tracking_valid"] and blurred["landmarks"] == []
    result = observer.observe(image, 1_300_000_000)
    assert result["status"] == "reacquired"
    assert result["reference_id"] == first["reference_id"]


@pytest.mark.parametrize("source,stamp,reason", [("new-owner", 1_033_333_333, "source_or_profile_changed"),
                                                ("recording", 2_000_000_000, "capture_gap")])
def test_discontinuities_require_reference_acquisition(source, stamp, reason):
    observer = tracker()
    image, _ = render()
    observer.observe(image, 1_000_000_000)
    result = observer.observe(image, stamp, source_id=source)
    assert result["status"] == "reacquired"
    assert result["input_discontinuity"] == reason


def test_owner_depth_requires_same_capture_and_physical_units(monkeypatch):
    metadata = {"sequence": 20, "profile": {"width": 640, "height": 480, "fps_num": 30, "fps_den": 1},
                "timestamps": {"normalized_unix_ns": 1_000_000_000, "source_ns": 1_000_000_000,
                               "source_domain": "real_sense_hardware"},
                "attributes": {"capture_epoch": "epoch-a", "frame_number": "10", "device_serial": "fixture"}}
    color = {"metadata": metadata, "image": render()[0]}
    depth = {"metadata": dict(metadata, attributes=dict(metadata["attributes"], aligned_to="wrist_color", frame_number="24")),
             "depth": np.full((480, 640), 1000, np.uint16), "depth_units_m": .0001}
    def sets(*args, **kwargs):
        yield {"frames": {"wrist_color": color, "wrist_depth": depth}}
        depth["metadata"] = dict(depth["metadata"], timestamps=dict(metadata["timestamps"], source_ns=1_033_333_333))
        yield {"frames": {"wrist_color": color, "wrist_depth": depth}}
    monkeypatch.setattr(stencil_inputs, "latest_socket_sets", sets)
    frames = list(stencil_inputs.owner_frames("/tmp/unused.sock", "wrist_color", 1))
    assert frames[0]["depth_m"].mean() == pytest.approx(.1)
    assert frames[1]["depth_m"] is None


@pytest.mark.parametrize('changed', ['sequence', 'capture_epoch', 'device_serial', 'source_domain'])
def test_live_depth_rejects_changed_capture_identity(changed):
    from copy import deepcopy

    from test_stencil_surface import geometry_fixture
    _, frame = geometry_fixture()
    dm = deepcopy(frame['depth_metadata'])
    if changed == 'sequence':
        dm['sequence'] += 1
    elif changed == 'source_domain':
        dm['timestamps'][changed] = 'host_unix'
    else:
        dm['attributes'][changed] = 'different'
    depth = {'metadata': dm, 'depth': np.ones((240, 320), np.uint16), 'depth_units_m': .0001}
    assert stencil_inputs.aligned_depth(depth, {'metadata': frame['color_metadata']}, 'fixture_color') is None


def test_equally_supported_patterns_are_ambiguous(monkeypatch):
    observer = tracker("tatbot-43", "tatbot-44")
    monkeypatch.setattr(observer.bank, "_candidate", lambda variant, *args:
                        {"pattern_id": variant["pattern_id"], "score": 10})
    result = observer.observe(render()[0], 1_000_000_000)
    assert result["reason"] == "ambiguous_reference"
    assert not result["image_tracking_valid"]


def test_repeated_search_reproduces_correspondences():
    bank = tracker().bank
    image = render()[0]
    first, _ = bank.detect(image)
    bank.detect(render(mirror=True)[0])
    repeated, _ = bank.detect(image)
    assert first["pattern_id"] == repeated["pattern_id"]
    np.testing.assert_array_equal(first["homography"], repeated["homography"])
    np.testing.assert_array_equal(first["uv"], repeated["uv"])


def test_empty_replay_exit_and_run_log_agree(tmp_path):
    index = tmp_path/"empty.jsonl"
    index.write_text("")
    output = tmp_path/"result"
    logs = tmp_path/"logs"
    result = subprocess.run([sys.executable, str(REPO/"scripts/vision/stencil_observe.py"), "replay",
                             "--reference", str(reference()), "--instance", "skin-a",
                             "--frames", str(index), "--output", str(output)],
                            env=dict(os.environ, TATBOT_LOG_ROOT=str(logs)),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 5, result.stderr
    assert json.loads((output/"report.json").read_text())["frames"] == 0
    metadata = json.loads(next(logs.glob("stencil-replay/*/meta.json")).read_text())
    assert metadata["exit_code"] == 5 and metadata["status"] == "fail"


def test_manifest_annotations_are_not_observer_inputs(tmp_path):
    cv2.imwrite(str(tmp_path/"frame.png"), render()[0])
    row = {"image": "frame.png", "timestamp_ns": 1_000_000_000,
           "expected_pattern_id": "poisoned", "homography_uv_to_image": [[999]]}
    index = tmp_path/"frames.jsonl"
    index.write_text(json.dumps(row)+"\n")
    frame = next(image_frames(index))
    assert "expected_pattern_id" not in frame and "homography_uv_to_image" not in frame
    observation = tracker().observe(frame["image"], frame["timestamp_ns"])
    assert observation["image_tracking_valid"]
    assert score([observation], [{"visible": True, "pattern_id": "poisoned"}])["false_accept_frames"] == [0]
    assert observation["pattern_id"] != "poisoned"


@pytest.mark.parametrize("mode,source", [("replay", ["--frames", "frames.jsonl"]),
                                         ("observe", ["--socket", "/tmp/frames.sock", "--sensor", "rgb"])])
def test_cli_plans_are_stdlib_and_do_not_open_hardware(tmp_path, mode, source):
    output = tmp_path/"not-created"
    command = [sys.executable, "-S", str(REPO/"scripts/lib/tatbot_cli"), "--json", "--dry-run",
               "vision", "stencil", mode, "--reference", str(reference()), "--instance", "skin-a",
               "--output", str(output), *source]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert not {"autonomous_motion", "read_arm", "write_config"} & set(plan["effects"])
    assert ("sensor_read" in plan["effects"]) == (mode == "observe")
    assert not output.exists()
