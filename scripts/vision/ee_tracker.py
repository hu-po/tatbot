#!/usr/bin/env python3
"""Run the Amcrest wrist-fiducial tracker on evidence or a visiond socket.

Examples:
  python3 scripts/vision/ee_tracker.py evidence /tmp/poe-capture \
    --output ~/tatbot-logs/vision/ee-tracking/replay.jsonl

  python3 scripts/vision/ee_tracker.py socket /tmp/tatbot-vision.sock \
    --output ~/tatbot-logs/vision/ee-tracking/shadow.jsonl --max-fps 10

The process is shadow-only: its only side effects are the requested log and
optional annotated frames.  It has no robot, e-stop, or controller imports.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "lib"))
from ee_fiducial import (  # noqa: E402
    DetectorConfig,
    EstimatorConfig,
    MultiCameraEstimator,
    VisionOnlyTracker,
    WristLayout,
    WristTagDetector,
    invert,
    load_calibration,
)
from fiducials import load_inventory  # noqa: E402
from fiducials.detector import Detection  # noqa: E402
from tatbot_runlog import log_root  # noqa: E402
from visiond_wire import decode_video, latest_socket_sets  # noqa: E402


def _git_identity(repo: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"source_commit": commit, "source_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"source_commit": None, "source_dirty": None}


def _timestamp_ns(metadata: dict) -> int:
    stamps = metadata["timestamps"]
    for key in ("normalized_unix_ns", "source_ns", "host_unix_ns"):
        value = stamps.get(key)
        if value is not None and int(value) > 0:
            return int(value)
    raise ValueError(f"{metadata.get('sensor_name')}: frame has no usable timestamp")


def _capture_to_processing_age_ms(timestamp_ns: int, *, now_ns: int | None = None) -> float:
    """Return live capture age using visiond's normalized Unix timeline."""
    current_ns = time.time_ns() if now_ns is None else int(now_ns)
    return max(0.0, (current_ns - int(timestamp_ns)) / 1e6)


def _load_recording_index(capture: Path) -> dict[str, dict[int, dict]]:
    output = {}
    for camera_dir in sorted(capture.glob("camera*/")):
        entries = {}
        metadata_path = camera_dir / "frames.jsonl"
        if not metadata_path.is_file():
            continue
        for line in metadata_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            entries[int(entry["metadata"]["sequence"])] = entry
        output[camera_dir.name] = entries
    return output


def _read_evidence_frame(camera_dir: Path, entry: dict) -> dict:
    path = camera_dir / entry["payload_file"]
    payload = path.read_bytes()
    if len(payload) != int(entry["payload_bytes"]):
        raise ValueError(f"payload length mismatch for {path}")
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise ValueError(f"payload checksum mismatch for {path}")
    metadata = entry["metadata"]
    profile = metadata["profile"]
    descriptor = {
        "format": profile["format"],
        "width": profile["width"],
        "height": profile["height"],
    }
    return {"metadata": metadata, "image": decode_video(payload, descriptor)}


def evidence_sets(capture: Path):
    sync_path = capture / "synchronized_frames.jsonl"
    if not sync_path.is_file():
        raise FileNotFoundError(f"no synchronized frame index at {sync_path}")
    recordings = _load_recording_index(capture)
    if not recordings:
        raise FileNotFoundError(f"no decoded camera evidence under {capture}")
    for line in sync_path.read_text().splitlines():
        if not line.strip():
            continue
        sync = json.loads(line)
        frames = {}
        for camera, sequence in sync["frame_sequences"].items():
            entry = recordings.get(camera, {}).get(int(sequence))
            if entry is None:
                raise ValueError(f"synchronized set references missing {camera} frame {sequence}")
            frames[camera] = _read_evidence_frame(capture / camera, entry)
        yield {
            "sequence": int(sync["sequence"]),
            "timestamp_ns": int(sync["timestamp_ns"]),
            "maximum_skew_ns": int(sync["maximum_skew_ns"]),
            "frames": frames,
        }


def detection_sets(path: Path):
    """Load retained Rust/Python detections without decoding camera images."""
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        timestamp_ns = int(row["timestamp_ns"])
        by_camera = {}
        for camera, items in row.get("detections", {}).items():
            by_camera[camera] = [
                Detection(
                    camera=str(item.get("camera", camera)),
                    tag_id=int(item["tag_id"]),
                    corners_px=np.asarray(item["corners_px"], dtype=np.float64),
                    timestamp_ns=int(item.get("timestamp_ns", timestamp_ns)),
                    side_px=float(item["side_px"]),
                )
                for item in items
            ]
        yield {
            "sequence": int(row["sequence"]),
            "timestamp_ns": timestamp_ns,
            "maximum_skew_ns": int(row.get("maximum_skew_ns", 0)),
            "queue_latency_ms": row.get("queue_latency_ms"),
            "detection_latency_ms": float(row.get("detection_latency_ms", 0.0)),
            "detections": by_camera,
            "source_line": line_number,
        }


def _scaled_cameras(calibrated, frame_set: dict):
    scaled = {}
    for name, frame in frame_set["frames"].items():
        if name not in calibrated:
            continue
        height, width = frame["image"].shape[:2]
        scaled[name] = calibrated[name].scaled(width, height)
    return scaled


def _check_calibration_ids(frame_set: dict, expected: str):
    mismatches = []
    for name, frame in frame_set["frames"].items():
        got = frame["metadata"].get("calibration_id")
        if got is not None and got != expected:
            mismatches.append(f"{name}={got}")
    if mismatches:
        raise ValueError(f"frame calibration id does not match {expected}: " + ", ".join(mismatches))


def _annotate(frame: np.ndarray, detections, estimate_status: str) -> np.ndarray:
    output = frame.copy()
    for item in detections:
        corners = item.corners_px.astype(np.int32).reshape(1, 4, 2)
        cv2.polylines(output, corners, True, (0, 255, 0), 3)
        center = tuple(item.corners_px.mean(axis=0).astype(int))
        cv2.putText(output, f"id{item.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(output, estimate_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)
    return output


def _expanded_roi(detections, width: int, height: int, margin_px: int):
    if not detections or margin_px <= 0:
        return None
    corners = np.concatenate([item.corners_px for item in detections])
    x0, y0 = np.floor(corners.min(axis=0) - margin_px).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0) + margin_px).astype(int)
    return (
        int(max(0, x0)),
        int(max(0, y0)),
        int(min(width, x1 + 1)),
        int(min(height, y1 + 1)),
    )


def main():
    repo = Path(__file__).resolve().parents[2]
    default_logs = log_root() / "vision"
    inventory = load_inventory()
    live_profile = inventory.detector_profiles["live"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source_kind", choices=("evidence", "socket", "detections"))
    ap.add_argument("source", type=Path)
    ap.add_argument("--calibration", type=Path, default=default_logs / "calibration-current.json")
    ap.add_argument("--robot-world", type=Path, default=default_logs / "robot-world-current.json")
    ap.add_argument("--wrist-layout", type=Path, default=repo / "config/wrist_tags_measured.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--detector-scale", type=float, default=live_profile.scale)
    ap.add_argument("--adaptive-window-max", type=int, default=live_profile.adaptive_window_max)
    ap.add_argument("--min-side-px", type=float, default=live_profile.min_side_px)
    ap.add_argument(
        "--corner-refinement",
        action=argparse.BooleanOptionalAction,
        default=live_profile.corner_refinement,
    )
    ap.add_argument("--huber-px", type=float, default=2.0)
    ap.add_argument("--max-source-rmse-px", type=float, default=6.0)
    ap.add_argument("--max-total-rmse-px", type=float, default=4.5)
    ap.add_argument("--max-condition", type=float, default=2e4)
    ap.add_argument("--max-translation-sigma-mm", type=float, default=3.0)
    ap.add_argument("--max-rotation-sigma-deg", type=float, default=1.5)
    ap.add_argument("--motion-compensation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-motion-window-ms", type=float, default=50.0)
    ap.add_argument(
        "--roi-margin-px",
        type=int,
        default=100,
        help="track each camera in the last tag bounding box plus this margin; 0 disables",
    )
    ap.add_argument("--roi-hold-frames", type=int, default=3)
    ap.add_argument(
        "--full-scan-period",
        type=int,
        default=50,
        help="staggered full-frame reacquisition period per camera; 0 disables",
    )
    ap.add_argument(
        "--exclude-cameras", default="", help="comma-separated ablation list, e.g. camera1,camera5"
    )
    ap.add_argument("--max-fps", type=float, default=10.0)
    ap.add_argument("--max-sets", type=int, default=0, help="0 runs until EOF/Ctrl+C")
    ap.add_argument("--annotated-dir", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="validate inputs without consuming frames")
    args = ap.parse_args()

    cameras, bundle = load_calibration(args.calibration)
    layout = WristLayout.load(args.wrist_layout, inventory_path=inventory.source)
    robot_world = json.loads(args.robot_world.expanduser().read_text())
    base_from_world = invert(np.asarray(robot_world["world_from_base"], dtype=np.float64))
    detector_config = DetectorConfig(
        scale=args.detector_scale,
        adaptive_window_max=args.adaptive_window_max,
        min_side_px=args.min_side_px,
        corner_refinement=args.corner_refinement,
    )
    estimator_config = EstimatorConfig(
        huber_px=args.huber_px,
        max_source_rmse_px=args.max_source_rmse_px,
        max_total_rmse_px=args.max_total_rmse_px,
        max_condition=args.max_condition,
        max_translation_sigma_mm=args.max_translation_sigma_mm,
        max_rotation_sigma_deg=args.max_rotation_sigma_deg,
        min_initial_tags=inventory.target("wrist").minimum_acquisition_ids or 1,
        motion_compensation=args.motion_compensation,
        max_motion_window_ms=args.max_motion_window_ms,
    )
    if not 0 < detector_config.scale <= 1:
        raise ValueError("--detector-scale must be in (0, 1]")
    if (
        detector_config.adaptive_window_max < 3
        or not np.isfinite(detector_config.min_side_px)
        or detector_config.min_side_px <= 0
    ):
        raise ValueError("detector window must be >=3 and minimum side must be positive")
    if any(
        not np.isfinite(value) or value <= 0
        for value in (
            estimator_config.huber_px,
            estimator_config.max_source_rmse_px,
            estimator_config.max_total_rmse_px,
            estimator_config.max_condition,
            estimator_config.max_translation_sigma_mm,
            estimator_config.max_rotation_sigma_deg,
            estimator_config.max_motion_window_ms,
        )
    ):
        raise ValueError("estimator thresholds and motion window must be positive")
    if (
        not np.isfinite(args.max_fps)
        or args.max_fps < 0
        or args.max_sets < 0
        or args.roi_margin_px < 0
        or args.roi_hold_frames < 0
        or args.full_scan_period < 0
    ):
        raise ValueError("rate, count, and ROI arguments must be finite and non-negative")
    run_manifest = {
        "schema_version": 1,
        "source_kind": args.source_kind,
        "source": str(args.source.expanduser()),
        "calibration": str(args.calibration.expanduser()),
        "calibration_id": bundle["bundle_id"],
        "robot_world": str(args.robot_world.expanduser()),
        "robot_world_sha256": hashlib.sha256(args.robot_world.expanduser().read_bytes()).hexdigest(),
        "wrist_layout": str(args.wrist_layout.expanduser()),
        "wrist_layout_hash": layout.layout_hash,
        "fiducial_inventory": str(inventory.source),
        "inventory_hash": inventory.inventory_hash,
        "detector_config": dataclasses.asdict(detector_config),
        "estimator_config": dataclasses.asdict(estimator_config),
        "excluded_cameras": sorted(item.strip() for item in args.exclude_cameras.split(",") if item.strip()),
        "max_fps": args.max_fps,
        "latency_basis": {
            "socket": "capture_to_estimate",
            "evidence": "offline_processing_only",
            "detections": "retained_capture_detection_plus_replay_solver",
        }[args.source_kind],
        "roi_config": {
            "margin_px": args.roi_margin_px,
            "hold_frames": args.roi_hold_frames,
            "full_scan_period": args.full_scan_period,
        },
        **_git_identity(repo),
    }
    if args.dry_run:
        run_manifest["cameras"] = sorted(cameras)
        run_manifest["output"] = str(args.output.expanduser())
        print(json.dumps(run_manifest, indent=2))
        return

    if args.source_kind == "evidence":
        source = evidence_sets(args.source.expanduser())
    elif args.source_kind == "socket":
        source = latest_socket_sets(args.source.expanduser())
    else:
        source = detection_sets(args.source.expanduser())
    detector = None if args.source_kind == "detections" else WristTagDetector(layout, detector_config)
    excluded_cameras = {item.strip() for item in args.exclude_cameras.split(",") if item.strip()}
    unknown_exclusions = excluded_cameras - cameras.keys()
    if unknown_exclusions:
        raise ValueError(f"unknown excluded cameras: {sorted(unknown_exclusions)}")
    tracker = None
    active_shape = None
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    if args.annotated_dir:
        args.annotated_dir = args.annotated_dir.expanduser()
        args.annotated_dir.mkdir(parents=True, exist_ok=True)
    min_interval_ns = 0 if args.max_fps <= 0 else int(1e9 / args.max_fps)
    last_processed_ns = -(1 << 120)
    processed = 0
    camera_order = sorted(cameras)
    roi_by_camera = {}
    roi_misses = dict.fromkeys(cameras, 0)
    roi_enabled = args.roi_margin_px > 0
    with output_path.open("x") as output:
        for frame_set in source:
            if frame_set["timestamp_ns"] - last_processed_ns < min_interval_ns:
                continue
            last_processed_ns = frame_set["timestamp_ns"]
            started = time.perf_counter()
            queue_latency_ms = frame_set.get("queue_latency_ms")
            if args.source_kind == "socket":
                # Socket timestamps are visiond-normalized Unix capture time.
                # Measure the same capture-to-processing age as the in-process
                # Rust path; the former Python `latency_ms` began here and
                # silently omitted camera, decoder, synchronizer, and socket age.
                queue_latency_ms = _capture_to_processing_age_ms(frame_set["timestamp_ns"])
            if args.source_kind == "detections":
                scaled = cameras
            else:
                _check_calibration_ids(frame_set, bundle["bundle_id"])
                scaled = _scaled_cameras(cameras, frame_set)
            shape = tuple((name, cam.width, cam.height) for name, cam in sorted(scaled.items()))
            if tracker is None:
                tracker = VisionOnlyTracker(MultiCameraEstimator(scaled, layout, estimator_config))
                active_shape = shape
            elif shape != active_shape:
                raise ValueError(f"active camera profile changed: {active_shape} -> {shape}")
            detections = []
            by_camera = {}
            detection_rois = {}
            detection_started = time.perf_counter()
            if args.source_kind == "detections":
                by_camera = {
                    name: found
                    for name, found in frame_set["detections"].items()
                    if name in scaled and name not in excluded_cameras
                }
                detections = [item for found in by_camera.values() for item in found]
                detection_latency_ms = frame_set["detection_latency_ms"]
            else:
                for name, frame in frame_set["frames"].items():
                    if name not in scaled or name in excluded_cameras:
                        continue
                    camera_index = camera_order.index(name)
                    force_full_scan = (
                        args.full_scan_period > 0
                        and processed > 0
                        and (processed + camera_index * max(1, args.full_scan_period // len(camera_order)))
                        % args.full_scan_period
                        == 0
                    )
                    roi = None if force_full_scan or not roi_enabled else roi_by_camera.get(name)
                    found = detector.detect(
                        name,
                        frame["image"],
                        _timestamp_ns(frame["metadata"]),
                        roi_xyxy=roi,
                    )
                    by_camera[name] = found
                    detections.extend(found)
                    detection_rois[name] = list(roi) if roi is not None else None
                    if not roi_enabled:
                        continue
                    if found:
                        height, width = frame["image"].shape[:2]
                        roi_by_camera[name] = _expanded_roi(found, width, height, args.roi_margin_px)
                        roi_misses[name] = 0
                    else:
                        roi_misses[name] += 1
                        if roi_misses[name] >= args.roi_hold_frames:
                            roi_by_camera.pop(name, None)
                detection_latency_ms = (time.perf_counter() - detection_started) * 1000.0
            solver_started = time.perf_counter()
            estimate = tracker.update(detections, frame_set["timestamp_ns"])
            solver_latency_ms = (time.perf_counter() - solver_started) * 1000.0
            processing_latency_ms = (time.perf_counter() - started) * 1000.0
            if args.source_kind == "detections":
                processing_latency_ms += detection_latency_ms
            latency_ms = processing_latency_ms + (queue_latency_ms or 0.0)
            record = estimate.as_dict(
                calibration_id=bundle["bundle_id"],
                layout_hash=layout.layout_hash,
                inventory_hash=layout.inventory_hash,
                tracking_frame=layout.parent_frame,
                base_from_world=base_from_world,
                sequence=frame_set["sequence"],
                maximum_skew_ns=frame_set["maximum_skew_ns"],
                latency_ms=latency_ms,
            )
            record["detections"] = {
                name: [
                    {
                        "camera": item.camera,
                        "tag_id": item.tag_id,
                        "side_px": item.side_px,
                        "corners_px": item.corners_px.tolist(),
                        "timestamp_ns": item.timestamp_ns,
                    }
                    for item in found
                ]
                for name, found in by_camera.items()
            }
            record["detection_rois"] = detection_rois
            record["detection_latency_ms"] = detection_latency_ms
            record["solver_latency_ms"] = solver_latency_ms
            record["queue_latency_ms"] = queue_latency_ms
            record["processing_latency_ms"] = processing_latency_ms
            record["latency_basis"] = run_manifest["latency_basis"]
            record["excluded_cameras"] = sorted(excluded_cameras)
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            if args.annotated_dir and args.source_kind != "detections":
                for name, frame in frame_set["frames"].items():
                    annotated = _annotate(frame["image"], by_camera.get(name, []), estimate.status)
                    cv2.imwrite(
                        str(args.annotated_dir / f"{frame_set['sequence']:08d}_{name}.jpg"), annotated
                    )
            processed += 1
            detected_tags = sorted({item.tag_id for item in detections})
            detected_cameras = sorted({item.camera for item in detections})
            print(
                f"set={frame_set['sequence']} status={estimate.status} "
                f"tags={detected_tags} cams={detected_cameras} "
                f"rmse={estimate.reprojection_rmse_px} latency_ms={latency_ms:.1f} "
                f"detect_ms={detection_latency_ms:.1f} solve_ms={solver_latency_ms:.1f}",
                flush=True,
            )
            if args.max_sets and processed >= args.max_sets:
                break
    print(f"wrote {processed} estimates to {output_path}; manifest {manifest_path}")


if __name__ == "__main__":
    main()
