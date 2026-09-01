"""Evidence and Unix-wire contract tests for the shadow runner."""

from __future__ import annotations

import hashlib
import json
import socket
import struct
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/vision"))

from ee_fiducial import Detection  # noqa: E402
from ee_tracker import (  # noqa: E402
    _capture_to_processing_age_ms,
    _expanded_roi,
    detection_sets,
    evidence_sets,
)
from visiond_wire import UnixWireReader  # noqa: E402


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
