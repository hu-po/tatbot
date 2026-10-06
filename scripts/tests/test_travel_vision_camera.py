"""Wrist subscriber rejects mixed geometry/identity and retains capture age without USB ownership."""

from __future__ import annotations

import copy
import ctypes
import json
import multiprocessing
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/tatbot_travel/src"))
from tatbot_travel import vision_camera  # noqa: E402

CONFIG = {"name": "wrist", "serial": "synthetic", "owner_role": "realsense"}


def mock_capture(monkeypatch, sets):
    def captures(path, config, stop_event):
        for data in sets(path, stop_event=stop_event):
            yield vision_camera.decode_wrist(data, config, unix_ns=time.time_ns(), monotonic_s=time.monotonic())

    monkeypatch.setattr(vision_camera, "_capture_frames", captures)


def frame_set(sequence=1, *, stamp=2_000_000_000):
    intrinsics = {"schema": "tatbot.camera-intrinsics/1", "width": 640, "height": 480,
                  "fx": 392., "fy": 391., "ppx": 315., "ppy": 238.,
                  "distortion_model": "InverseBrownConrady", "distortion_coefficients": [0.] * 5}
    metadata = {"attributes": {"intrinsics": json.dumps(intrinsics), "device_serial": "synthetic",
                               "physical_arm": "left", "capture_owner_role": "realsense", "capture_epoch": "epoch"},
                "timestamps": {"normalized_unix_ns": stamp}, "sequence": sequence}
    depth = copy.deepcopy(metadata)
    depth["attributes"]["aligned_to"] = "wrist_color"
    bgr = np.empty((480, 640, 3), np.uint8)
    bgr[:] = [1, 2, 3]
    return {"frames": {"wrist_color": {"image": bgr, "image_metadata": metadata},
                       "wrist_depth": {"depth": np.full((480, 640), 1500, np.uint16),
                                       "depth_metadata": depth, "depth_units_m": 0.0001}}}


def test_depth_scale_rgb_order_and_capture_age_are_preserved():
    data = frame_set()
    data["frames"]["wrist_depth"]["depth_metadata"]["timestamps"]["normalized_unix_ns"] -= 10_000_000
    (rgb, depth, capture), source = vision_camera.decode_wrist(data, CONFIG, unix_ns=2_250_000_000, monotonic_s=40)
    assert rgb[0, 0].tolist() == [3, 2, 1]
    assert float(depth[0, 0]) == pytest.approx(0.15)
    assert capture == pytest.approx(39.74) and source["capture_unix_ns"] == 1_990_000_000
    assert not rgb.flags.writeable and not depth.flags.writeable
    # Decoding a still-older capture must not stamp it with the receiver's now.
    old, _ = vision_camera.decode_wrist(data, CONFIG, unix_ns=4_250_000_000, monotonic_s=42)
    assert old[2] == pytest.approx(capture)
    assert source["color_metadata"] == data["frames"]["wrist_color"]["image_metadata"]
    assert source["depth_metadata"] == data["frames"]["wrist_depth"]["depth_metadata"]
    data["frames"]["wrist_color"]["image_metadata"]["sequence"] = 100
    assert source["color_metadata"]["sequence"] == 1


def test_owner_canonical_inverse_brown_name_is_accepted_with_original_metadata():
    data = frame_set()
    for name, key in (("wrist_color", "image_metadata"), ("wrist_depth", "depth_metadata")):
        attributes = data["frames"][name][key]["attributes"]
        intr = json.loads(attributes["intrinsics"])
        intr["distortion_model"] = "BrownConradyInverse"
        intr["distortion_coefficients"] = [-.05, .058, -.00002, .0004, -.019]
        attributes["intrinsics"] = json.dumps(intr)
    _, source = vision_camera.decode_wrist(data, CONFIG, unix_ns=2_250_000_000, monotonic_s=40)
    assert source["intrinsics"]["model"] == "inverse_brown_conrady"
    assert source["intrinsics"]["distortion"] == tuple(intr["distortion_coefficients"])
    assert source["color_metadata"] == data["frames"]["wrist_color"]["image_metadata"]


def test_atomic_sample_keeps_original_identity_after_a_new_frame_arrives():
    camera = vision_camera.VisiondWristCamera.__new__(vision_camera.VisiondWristCamera)
    camera.lock, camera.path, camera.error, camera.source = threading.Lock(), Path("synthetic.sock"), None, None
    first, source = vision_camera.decode_wrist(frame_set(), CONFIG, unix_ns=2_250_000_000, monotonic_s=40)
    camera._accept(first, source)
    retained, metadata = camera.newest_with_metadata()
    second, newer = vision_camera.decode_wrist(frame_set(2), CONFIG, unix_ns=2_250_000_000, monotonic_s=40)
    camera._accept(second, newer)
    assert retained is first and camera.newest() is second
    assert metadata["source_sequence"] == metadata["color_metadata"]["sequence"] == 1
    assert camera.metadata()["source_sequence"] == 2
    metadata["color_metadata"]["sequence"] = 100
    assert source["color_metadata"]["sequence"] == 1


@pytest.mark.parametrize("field,value", [("aligned_to", "another_color"), ("device_serial", "another_device"),
                                          ("capture_epoch", "restarted"), ("physical_arm", "right"),
                                          ("capture_owner_role", "another_owner")])
def test_depth_cannot_be_combined_with_another_identity_or_alignment(field, value):
    data = frame_set()
    data["frames"]["wrist_depth"]["depth_metadata"]["attributes"][field] = value
    with pytest.raises(ValueError, match="aligned|identity"):
        vision_camera.decode_wrist(data, CONFIG, unix_ns=2_250_000_000, monotonic_s=40)


@pytest.mark.parametrize("stamp", [None, 0, 2_250_000_001, 1_900_000_000])
def test_bad_or_unsynchronized_capture_timestamps_are_refused(stamp):
    data = frame_set()
    data["frames"]["wrist_depth"]["depth_metadata"]["timestamps"]["normalized_unix_ns"] = stamp
    with pytest.raises(ValueError, match="timestamps"):
        vision_camera.decode_wrist(data, CONFIG, unix_ns=2_250_000_000, monotonic_s=40)


@pytest.mark.parametrize("units", [None, 0, -0.001, float("nan"), 1])
def test_missing_or_invalid_depth_scale_is_never_assumed(units):
    data = frame_set()
    data["frames"]["wrist_depth"]["depth_units_m"] = units
    with pytest.raises(ValueError, match="depth units"):
        vision_camera.decode_wrist(data, CONFIG, unix_ns=2_250_000_000, monotonic_s=40)


def test_live_adapter_uses_shared_decoder_and_cancels_without_stopping_owner(monkeypatch):
    stopped = threading.Event()

    def sets(path, *, stop_event):
        assert path == Path("synthetic.sock")
        yield frame_set(stamp=time.time_ns() - 10_000_000)
        stop_event.wait(2)
        stopped.set()

    mock_capture(monkeypatch, sets)
    camera = vision_camera.VisiondWristCamera(Path("synthetic.sock"), config=CONFIG)
    try:
        metadata = camera.wait_ready(timeout_s=1)
        assert metadata["camera"] == "wrist_color" and camera.intrinsics.width == 640
    finally:
        camera.close()
    assert stopped.is_set() and not camera.worker.is_alive()


@pytest.mark.parametrize("change", ["duplicate", "epoch"])
def test_capture_restart_or_nonadvancing_sequence_latches_failure(monkeypatch, change):
    def sets(path, *, stop_event):
        yield frame_set(stamp=time.time_ns() - 10_000_000)
        newer = frame_set(sequence=1 if change == "duplicate" else 2, stamp=time.time_ns() - 10_000_000)
        if change == "epoch":
            for name, key in (("wrist_color", "image_metadata"), ("wrist_depth", "depth_metadata")):
                newer["frames"][name][key]["attributes"]["capture_epoch"] = "new"
        yield newer

    mock_capture(monkeypatch, sets)
    camera = vision_camera.VisiondWristCamera(config=CONFIG)
    try:
        camera.worker.join(1)
        with pytest.raises(RuntimeError, match="restarted.*advance"):
            camera.newest()
    finally:
        camera.close()


def test_reader_failure_is_not_mistaken_for_operator_cancellation(monkeypatch):
    def sets(path, *, stop_event):
        yield frame_set(stamp=time.time_ns() - 10_000_000)
        stop_event.set()  # the shared reader signals its own cleanup on failure too
        raise ValueError("malformed wire payload")

    mock_capture(monkeypatch, sets)
    camera = vision_camera.VisiondWristCamera(config=CONFIG)
    try:
        camera.worker.join(1)
        with pytest.raises(RuntimeError, match="malformed wire payload"):
            camera.newest()
    finally:
        camera.close()


def _wrist_packet(frames, scenario):
    data = frame_set(frames, stamp=time.time_ns() - 10_000_000)
    if (scenario == "restart" and frames >= 3) or (scenario == "transient_restart" and frames == 3):
        for name, key in (("wrist_color", "image_metadata"), ("wrist_depth", "depth_metadata")):
            data["frames"][name][key]["attributes"]["capture_epoch"] = "new"
    if scenario == "malformed" and frames >= 3:
        data["frames"]["wrist_depth"]["depth_metadata"]["attributes"]["aligned_to"] = "other_color"
    data["frames"]["wrist_color"]["image"][1, 0, 0] = frames % 256
    data["frames"]["wrist_depth"]["depth"][1, 0] = frames % 60000
    wire_frames, payloads = [], []
    for name, kind, key, variant, format_ in (("wrist_color", "image", "image_metadata", "Video", "Bgr8"),
                                             ("wrist_depth", "depth", "depth_metadata", "Depth", "Z16")):
        metadata = data["frames"][name][key]
        metadata["sensor_name"] = name
        metadata["profile"] = {"width": 640, "height": 480, "format": format_.lower()}
        if variant == "Depth":
            metadata["attributes"]["depth_units_m"] = "0.0001"
        payload = data["frames"][name][kind].tobytes()
        payloads.append(payload)
        descriptor = {"width": 640, "height": 480, "bytes": len(payload)}
        if variant == "Video":
            descriptor["format"] = format_
        wire_frames.append({"metadata": metadata, "payload": {variant: descriptor}})
    header = json.dumps({"magic": "tatbot-vision-frame-set", "version": 1, "sequence": frames,
                         "timestamp_ns": time.time_ns(), "maximum_skew_ns": 0, "frames": wire_frames}).encode()
    return [struct.pack(">I", len(header)) + header, *payloads]


def _publish_wrist(path, stopped, report, scenario):
    """Independent owner with the real wire framing and 20-ms write deadline."""
    frames, error = 0, None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as owner:
            owner.bind(str(path))
            owner.listen(1)
            owner.settimeout(.1)
            while not stopped.is_set():
                try:
                    client, _ = owner.accept()
                    break
                except TimeoutError:
                    continue
            else:
                return
            with client:
                client.settimeout(.02)
                while not stopped.is_set():
                    frames += 1
                    for payload in _wrist_packet(frames, scenario):
                        client.sendall(payload)
                    if scenario == "eof":
                        break
                    stopped.wait(1 / 30)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        report.send({"frames": frames, "error": error})
        report.close()


@pytest.fixture
def wire_owner(tmp_path):
    context = multiprocessing.get_context("spawn")
    owners = []

    def start(scenario="live"):
        path = tmp_path / f"wrist-{len(owners)}.sock"
        stopped = context.Event()
        receiver, sender = context.Pipe(duplex=False)
        worker = context.Process(target=_publish_wrist, args=(path, stopped, sender, scenario))
        worker.start()
        sender.close()
        owners.append((worker, stopped, receiver))
        return path, worker, stopped, receiver

    yield start
    for worker, stopped, receiver in owners:
        stopped.set()
        worker.join(2)
        if worker.is_alive():
            worker.terminate()
            worker.join(1)
        receiver.close()
        worker.close()


@pytest.mark.parametrize("retain_original", [False, True])
def test_spawned_subscription_survives_parent_gil_stall_without_restamping(wire_owner, retain_original):
    path, owner, stopped, report = wire_owner()
    camera = vision_camera.VisiondWristCamera(path, config=CONFIG, retain_original=retain_original)
    try:
        first = camera.wait_ready(timeout_s=10)
        # PyDLL deliberately retains the interpreter lock: a socket reader
        # thread in this process cannot run during this 200-ms call.
        ctypes.PyDLL(None).usleep(200_000)
        deadline = time.monotonic() + 2
        while camera.metadata()["source_sequence"] <= first["source_sequence"] + 2 and time.monotonic() < deadline:
            time.sleep(.01)
        frame, source = camera.newest_with_metadata()
        assert source["source_sequence"] > first["source_sequence"] + 2
        assert source["capture_epoch"] == first["capture_epoch"]
        assert source["color_metadata"]["sequence"] == source["depth_metadata"]["sequence"] == source["source_sequence"]
        assert source["intrinsics"]["distortion"] == (0.,) * 5
        assert frame[0][0, 0].tolist() == [3, 2, 1] and float(frame[1][0, 0]) == pytest.approx(.15)
        assert not frame[0].flags.writeable and not frame[1].flags.writeable
        assert 0 <= time.monotonic() - frame[2] < .3
        if retain_original:
            _, retained_source, original = camera.newest_with_original()
            assert original["color"][640 * 3] == retained_source["source_sequence"] % 256
            raw_depth = np.frombuffer(original["depth"], dtype="<u2").reshape(480, 640)
            assert raw_depth[1, 0] == retained_source["source_sequence"] % 60000
            original["color"] = b"changed caller dictionary"
            assert len(camera.newest_with_original()[2]["color"]) == 480 * 640 * 3
            assert "_original_payloads" not in camera.metadata()
        assert owner.is_alive() and not report.poll()
        stopped.set()
        owner.join(2)
        assert report.recv()["error"] is None
    finally:
        camera.close()
    assert not camera.worker.is_alive()


@pytest.mark.parametrize("scenario,diagnostic", [("eof", "subscription ended"),
                                                ("restart", "restarted.*advance"),
                                                ("transient_restart", "restarted.*advance"),
                                                ("malformed", "not aligned")])
def test_spawned_subscriber_latches_owner_eof_and_invalid_capture(wire_owner, scenario, diagnostic):
    path, _, _, _ = wire_owner(scenario)
    camera = vision_camera.VisiondWristCamera(path, config=CONFIG)
    try:
        camera.worker.join(10)
        assert not camera.worker.is_alive()
        with pytest.raises(RuntimeError, match=diagnostic):
            camera.newest()
    finally:
        camera.close()


def test_spawned_subscriber_cancels_missing_owner_without_leaking_a_child(tmp_path):
    before = {child.pid for child in multiprocessing.active_children()}
    camera = vision_camera.VisiondWristCamera(tmp_path / "missing.sock", config=CONFIG)
    started = time.monotonic()
    camera.close()
    assert time.monotonic() - started < 2
    assert not camera.worker.is_alive()
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_latest_shared_slot_preserves_old_exposure_and_atomic_original_metadata():
    slot = vision_camera._CaptureSlot(multiprocessing.get_context("spawn"))
    first, source = vision_camera.decode_wrist(frame_set(), CONFIG, unix_ns=2_250_000_000, monotonic_s=40)
    slot.publish(first, source)
    newer, new_source = vision_camera.decode_wrist(frame_set(2, stamp=2_010_000_000), CONFIG,
                                                  unix_ns=3_250_000_000, monotonic_s=41)
    slot.publish(newer, new_source)
    serial, frame, retained = slot.take(0)
    assert serial == 2 and retained == new_source
    assert frame[2] == newer[2] == pytest.approx(39.76)
    assert slot.take(serial) is None
    # Further writes cannot mutate a previously returned pair or its metadata.
    slot.publish(first, source)
    assert retained["source_sequence"] == 2 and frame[2] == pytest.approx(39.76)


def test_subscriber_process_death_latches_failure_and_cleanup_stays_bounded(wire_owner):
    path, _, stopped, _ = wire_owner()
    camera = vision_camera.VisiondWristCamera(path, config=CONFIG)
    try:
        camera.wait_ready(timeout_s=10)
        children = [child for child in multiprocessing.active_children() if child.name == "travel-wrist-subscriber"]
        assert len(children) == 1
        children[0].terminate()
        camera.worker.join(2)
        assert not camera.worker.is_alive()
        with pytest.raises(RuntimeError, match="exited"):
            camera.newest()
    finally:
        stopped.set()
        camera.close()
