"""Evidence and Unix-wire contract tests for the shadow runner."""

from __future__ import annotations

import hashlib
import json
import socket
import struct
import threading

import numpy as np
import pytest
from ee_fiducial import Detection  # noqa: E402
from ee_tracker import (  # noqa: E402
    _capture_to_processing_age_ms,
    _expanded_roi,
    detection_sets,
    evidence_sets,
)
from visiond_wire import UnixWireReader  # noqa: E402


def test_tracker_serializes_distinct_root_and_arm_base_frames():
    from pathlib import Path

    from ee_fiducial import PoseEstimate
    from ee_tracker import registered_output_frames
    from urdf_kinematics import UrdfChain

    urdf = Path(__file__).resolve().parents[2] / 'urdf/tatbot.urdf'
    chain = UrdfChain(urdf)
    world = np.array([[0., -1, 0, .4], [1, 0, 0, .2], [0, 0, 1, .1], [0, 0, 0, 1]])
    root, base = registered_output_frames(
        {'bundle_id': 'measured'}, {'calibration_id': 'measured', 'world_from_base': world.tolist()},
        urdf, 'right/base_link')
    root_tool = chain.link_pose('right/gripper_left', {'right/joint_0': .2, 'right/joint_1': 1.2})
    estimate = PoseEstimate('measured', 1, world @ root_tool, 0., [], [], [], 4, 1., 0., 0.)
    record = estimate.as_dict(calibration_id='measured', layout_hash='layout',
        root_from_world=root, base_from_world=base, base_frame='right/base_link')
    np.testing.assert_allclose(record['root_from_ee'], root_tool, atol=1e-12)
    np.testing.assert_allclose(record['base_from_ee'],
        np.linalg.inv(chain.link_pose('right/base_link')) @ root_tool, atol=1e-12)
    assert record['base_frame'] == 'right/base_link'
    assert not np.allclose(record['base_from_ee'], record['root_from_ee'])


def test_tracker_rejects_calibration_identity_mismatch():
    from ee_tracker import registered_output_frames
    with pytest.raises(ValueError, match='calibration IDs differ'):
        registered_output_frames({'bundle_id': 'new'}, {'calibration_id': 'old'}, None, 'base')


def _metadata(camera, sequence):
    return {
        "sensor_name": camera,
        "sensor_kind": "po_e",
        "sequence": sequence,
        "profile": {
            "stream": "main",
            "width": 2,
            "height": 1,
            "fps_num": 20,
            "fps_den": 1,
            "format": "bgr8",
        },
        "timestamps": {
            "source_ns": 1000 + sequence,
            "source_domain": "camera_ntp",
            "rtp_timestamp": None,
            "pipeline_pts_ns": None,
            "pipeline_dts_ns": None,
            "host_monotonic_ns": 10,
            "host_unix_ns": 2_000_000_000,
            "normalized_unix_ns": 2_000_000_000 + sequence,
        },
        "dropped_before": 0,
        "calibration_id": None,
        "flags": [],
        "attributes": {},
    }


def _write_camera(capture, camera, sequence, payload):
    directory = capture / camera
    directory.mkdir()
    filename = f"{sequence:012d}.bgr8"
    (directory / filename).write_bytes(payload)
    entry = {
        "metadata": _metadata(camera, sequence),
        "payload_file": filename,
        "payload_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (directory / "frames.jsonl").write_text(json.dumps(entry) + "\n")


def test_evidence_reader_verifies_and_reassembles_synchronized_frames(tmp_path):
    payload1 = bytes([1, 2, 3, 4, 5, 6])
    payload2 = bytes([7, 8, 9, 10, 11, 12])
    _write_camera(tmp_path, "camera1", 3, payload1)
    _write_camera(tmp_path, "camera2", 7, payload2)
    sync = {
        "sequence": 11,
        "timestamp_basis": "normalized_unix_ns",
        "timestamp_ns": 2_000_000_005,
        "maximum_skew_ns": 2,
        "frame_sequences": {"camera1": 3, "camera2": 7},
    }
    (tmp_path / "synchronized_frames.jsonl").write_text(json.dumps(sync) + "\n")
    frame_set = next(evidence_sets(tmp_path))
    assert frame_set["sequence"] == 11
    assert frame_set["maximum_skew_ns"] == 2
    assert np.array_equal(
        frame_set["frames"]["camera1"]["image"].reshape(-1), np.frombuffer(payload1, dtype=np.uint8)
    )


def test_unix_reader_matches_rust_length_delimited_video_contract(tmp_path):
    path = tmp_path / "frames.sock"
    ready = threading.Event()
    payload = bytes([1, 2, 3, 4, 5, 6])
    header = {
        "magic": "tatbot-vision-frame-set",
        "version": 1,
        "sequence": 9,
        "timestamp_basis": "normalized_unix_ns",
        "timestamp_ns": 2_000_000_000,
        "maximum_skew_ns": 10,
        "frames": [
            {
                "metadata": _metadata("camera1", 4),
                "payload": {"Video": {"format": "bgr8", "width": 2, "height": 1, "bytes": len(payload)}},
            }
        ],
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()

    def server():
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen(1)
        ready.set()
        client, _ = listener.accept()
        client.sendall(struct.pack(">I", len(encoded)) + encoded + payload)
        client.close()
        listener.close()

    worker = threading.Thread(target=server)
    worker.start()
    ready.wait(timeout=2)
    received = UnixWireReader(path).receive()
    worker.join(timeout=2)
    assert received["sequence"] == 9
    assert received["timestamp_basis"] == "normalized_unix_ns"
    assert received["frames"]["camera1"]["image"].shape == (1, 2, 3)
    assert received["frames"]["camera1"]["image"].tobytes() == payload


def test_evidence_checksum_failure_is_not_silently_accepted(tmp_path):
    _write_camera(tmp_path, "camera1", 1, bytes([1, 2, 3, 4, 5, 6]))
    entry_path = tmp_path / "camera1/frames.jsonl"
    entry = json.loads(entry_path.read_text())
    entry["sha256"] = "0" * 64
    entry_path.write_text(json.dumps(entry) + "\n")
    sync = {
        "sequence": 0,
        "timestamp_ns": 2_000_000_000,
        "maximum_skew_ns": 0,
        "frame_sequences": {"camera1": 1},
    }
    (tmp_path / "synchronized_frames.jsonl").write_text(json.dumps(sync) + "\n")
    try:
        next(evidence_sets(tmp_path))
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("tampered evidence must be rejected")


def test_detection_roi_is_clamped_and_json_serializable():
    detection = Detection(
        "camera1",
        0,
        np.array([[5.2, 7.1], [40.0, 7.0], [40.0, 35.0], [5.0, 35.0]]),
        1,
        30.0,
    )
    roi = _expanded_roi([detection], 100, 80, 20)
    assert roi == (0, 0, 61, 56)
    assert json.loads(json.dumps({"roi": roi}))["roi"] == [0, 0, 61, 56]


def test_capture_age_uses_normalized_unix_time_and_clamps_clock_noise():
    assert _capture_to_processing_age_ms(1_000_000_000, now_ns=1_135_500_000) == 135.5
    assert _capture_to_processing_age_ms(1_000_000_001, now_ns=1_000_000_000) == 0.0


def test_detection_sets_preserve_per_camera_capture_times(tmp_path):
    source = tmp_path / "detections.jsonl"
    source.write_text(
        json.dumps(
            {
                "sequence": 7,
                "timestamp_ns": 1_000,
                "maximum_skew_ns": 20,
                "queue_latency_ms": 3.5,
                "detection_latency_ms": 4.5,
                "detections": {
                    "camera2": [
                        {
                            "camera": "camera2",
                            "tag_id": 6,
                            "corners_px": [[1, 2], [3, 2], [3, 4], [1, 4]],
                            "timestamp_ns": 1_020,
                            "side_px": 2.0,
                        }
                    ]
                },
            }
        )
        + "\n"
    )

    row = next(detection_sets(source))
    detection = row["detections"]["camera2"][0]
    assert row["sequence"] == 7
    assert row["queue_latency_ms"] == 3.5
    assert detection.camera == "camera2"
    assert detection.tag_id == 6
    assert detection.timestamp_ns == 1_020


def test_bounded_subscription_closes_client_but_preserves_owner(tmp_path):
    import time

    from visiond_wire import latest_socket_sets

    path = tmp_path / "owner.sock"
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(path))
    server.listen(1)
    disconnected = threading.Event()

    def owner():
        client, _ = server.accept()
        client.settimeout(2)
        with client:
            assert client.recv(1) == b""
        disconnected.set()

    worker = threading.Thread(target=owner)
    worker.start()
    started = time.monotonic()
    assert list(latest_socket_sets(path, duration_s=0.1)) == []
    assert time.monotonic() - started < 1.5
    assert disconnected.wait(1)
    assert path.exists()
    worker.join(1)
    server.close()


def test_external_cancellation_interrupts_missing_socket_connection(tmp_path):
    import time

    from visiond_wire import latest_socket_sets

    stop = threading.Event()
    finished = threading.Event()

    def consume():
        assert list(latest_socket_sets(tmp_path / "missing.sock", connect_timeout_s=10, stop_event=stop)) == []
        finished.set()

    worker = threading.Thread(target=consume)
    worker.start()
    time.sleep(0.05)
    stop.set()
    assert finished.wait(1)
    worker.join(1)


def test_external_cancellation_closes_a_stalled_subscription(tmp_path):
    from visiond_wire import latest_socket_sets

    path = tmp_path / "owner.sock"
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(path))
    server.listen(1)
    stop = threading.Event()
    finished = threading.Event()

    def consume():
        assert list(latest_socket_sets(path, stop_event=stop)) == []
        finished.set()

    worker = threading.Thread(target=consume)
    worker.start()
    client, _ = server.accept()
    stop.set()
    assert finished.wait(1)
    client.settimeout(1)
    assert client.recv(1) == b""
    assert path.exists()  # capture owner still owns its endpoint
    worker.join(1)
    client.close()
    server.close()


def _wire_set(sequence, payload):
    header = {
        "magic": "tatbot-vision-frame-set",
        "version": 1,
        "sequence": sequence,
        "timestamp_basis": "normalized_unix_ns",
        "timestamp_ns": 2_000_000_000 + sequence,
        "maximum_skew_ns": 10,
        "frames": [
            {
                "metadata": _metadata("camera1", sequence),
                "payload": {"Video": {"format": "bgr8", "width": 2, "height": 1, "bytes": len(payload)}},
            }
        ],
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack(">I", len(encoded)) + encoded + payload


def test_optional_original_wire_payload_is_exact_and_profile_bound():
    from visiond_wire import decode_frame_set

    payload = bytes([1, 128, 3, 128])
    metadata = {"sensor_name": "synthetic_color", "profile": {"format": "yuyv", "width": 2, "height": 1}}
    header = {"sequence": 1, "timestamp_ns": 1, "maximum_skew_ns": 0,
              "frames": [{"metadata": metadata,
                          "payload": {"Video": {"format": "Yuyv", "width": 2, "height": 1, "bytes": 4}}}]}
    plain = decode_frame_set(header, [payload])
    assert "raw_payload" not in plain["frames"]["synthetic_color"]
    retained = decode_frame_set(header, [payload], retain_payloads=True)
    assert retained["frames"]["synthetic_color"]["raw_payload"] is payload
    metadata["profile"]["format"] = "bgr8"
    with pytest.raises(ValueError, match="descriptor differs"):
        decode_frame_set(header, [payload], retain_payloads=True)


def test_original_depth_uses_the_rust_wire_z16_variant_without_a_format_field():
    from visiond_wire import decode_frame_set

    payload = bytes([0, 0, 255, 255])
    metadata = {"sensor_name": "synthetic_depth", "profile": {"format": "z16", "width": 2, "height": 1}}
    header = {"sequence": 1, "timestamp_ns": 1, "maximum_skew_ns": 0,
              "frames": [{"metadata": metadata,
                          "payload": {"Depth": {"width": 2, "height": 1, "bytes": 4}}}]}
    retained = decode_frame_set(header, [payload], retain_payloads=True)["frames"]["synthetic_depth"]
    assert retained["raw_payload"] is payload
    assert retained["depth"].tolist() == [[0, 65535]]
    metadata["profile"]["format"] = "yuyv"
    with pytest.raises(ValueError, match="descriptor differs"):
        decode_frame_set(header, [payload], retain_payloads=True)


def test_unix_reader_drains_a_whole_set_before_decoding_any_frame(tmp_path, monkeypatch):
    """The owner drops a client that stalls its writes; a decode between payload reads is such a stall."""
    import visiond_wire
    from visiond_wire import decode_frame_set

    path = tmp_path / "frames.sock"
    ready = threading.Event()
    payload = bytes([1, 2, 3, 4, 5, 6])

    def server():
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen(1)
        ready.set()
        client, _ = listener.accept()
        client.sendall(_wire_set(9, payload))
        client.close()
        listener.close()

    decoded_on = []
    real = visiond_wire.decode_video

    def spy(data, descriptor, **kwargs):
        decoded_on.append(threading.current_thread().name)
        return real(data, descriptor, **kwargs)

    monkeypatch.setattr(visiond_wire, "decode_video", spy)
    worker = threading.Thread(target=server)
    worker.start()
    ready.wait(timeout=2)
    header, payloads = UnixWireReader(path).receive_raw()
    worker.join(timeout=2)
    assert payloads == [payload] and decoded_on == []
    frame_set = decode_frame_set(header, payloads)
    assert len(decoded_on) == 1
    assert frame_set["sequence"] == 9
    assert frame_set["frames"]["camera1"]["image"].tobytes() == payload


def test_latest_socket_sets_decodes_on_the_consumer_thread_only(tmp_path, monkeypatch):
    import visiond_wire
    from visiond_wire import latest_socket_sets

    path = tmp_path / "frames.sock"
    ready = threading.Event()
    payload = bytes([1, 2, 3, 4, 5, 6])

    def owner():
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen(1)
        ready.set()
        client, _ = listener.accept()
        client.settimeout(5)
        for sequence in (9, 10, 11):
            client.sendall(_wire_set(sequence, payload))
        with client:  # stay up until the subscriber's deadline closes it
            client.recv(1)
        listener.close()

    decoded_on = []
    real = visiond_wire.decode_video

    def spy(data, descriptor, **kwargs):
        decoded_on.append(threading.current_thread().name)
        return real(data, descriptor, **kwargs)

    monkeypatch.setattr(visiond_wire, "decode_video", spy)
    worker = threading.Thread(target=owner)
    worker.start()
    ready.wait(timeout=2)
    sets = list(latest_socket_sets(path, duration_s=0.5))
    worker.join(timeout=5)
    assert sets and sets[-1]["sequence"] in (9, 10, 11)
    assert all(frame_set["frames"]["camera1"]["image"].tobytes() == payload for frame_set in sets)
    # Only taken sets are decoded, and never on the socket thread.
    assert len(decoded_on) == len(sets)
    assert "visiond-socket-reader" not in decoded_on


def test_partial_camera_membership_preserves_models_but_resize_refuses():
    from types import SimpleNamespace

    import pytest
    from ee_tracker import _merge_camera_profiles

    first = SimpleNamespace(width=2960, height=1668)
    second = SimpleNamespace(width=2960, height=1668)
    active = {}
    _merge_camera_profiles(active, {"camera1": first})
    _merge_camera_profiles(active, {"camera2": second})
    _merge_camera_profiles(active, {"camera1": first})
    assert active == {"camera1": first, "camera2": second}
    with pytest.raises(ValueError, match="profile changed for camera1"):
        _merge_camera_profiles(active, {"camera1": SimpleNamespace(width=1480, height=834)})
    assert active["camera1"] is first


def test_luma_wire_preserves_pixels_without_changing_geometry():
    from visiond_wire import decode_video

    image = decode_video(bytes([2, 5]), {"format": "y8", "width": 2, "height": 1})
    assert image.shape == (1, 2, 3)
    assert image.tolist() == [[[2, 2, 2], [5, 5, 5]]]


def test_yuyv_owner_color_decodes_and_rejects_incomplete_pairs():
    import pytest
    from visiond_wire import decode_video
    image = decode_video(bytes([128, 128, 128, 128]), {'format': 'yuyv', 'width': 2, 'height': 1})
    assert image.shape == (1, 2, 3)
    assert np.all(image[..., 0] == image[..., 1])
    assert np.all(image[..., 1] == image[..., 2])
    with pytest.raises(ValueError, match='YUYV'):
        decode_video(bytes([128, 128]), {'format': 'yuyv', 'width': 1, 'height': 1})
