#!/usr/bin/env python3
"""Compare shadow EE estimates with timestamp-aligned follower FK.

FK is consumed only here, after vision estimation.  The report calls the
difference a disagreement rather than absolute vision error: encoder/URDF,
hand-eye, camera extrinsic, and timestamp errors all contribute.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "lib"))
from ee_fiducial import invert, rotation_distance_deg  # noqa: E402
from fiducials import load_inventory  # noqa: E402
from tatbot_runlog import log_root  # noqa: E402
from teleop_log import TeleopLog  # noqa: E402
from urdf_kinematics import UrdfChain  # noqa: E402


def _percentiles(values):
    return {
        "median": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "max": max(values) if values else None,
    }


def _interpolated_joints(log: TeleopLog, timestamp_s: float) -> np.ndarray | None:
    if not len(log) or timestamp_s < log.unix_seconds[0] or timestamp_s > log.unix_seconds[-1]:
        return None
    return np.array(
        [
            np.interp(timestamp_s, log.unix_seconds, log.follower_pos[:, index])
            for index in range(min(7, log.num_joints))
        ]
    )


def _mean_pose(poses: list[np.ndarray]) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = np.median([pose[:3, 3] for pose in poses], axis=0)
    out[:3, :3] = Rotation.from_matrix([pose[:3, :3] for pose in poses]).mean().as_matrix()
    return out


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("estimates", type=Path)
    ap.add_argument("teleop_log", type=Path)
    ap.add_argument(
        "--robot-world", type=Path, default=log_root() / "vision/robot-world-current.json"
    )
    ap.add_argument("--urdf", type=Path, default=repo / "urdf/tatbot.urdf")
    ap.add_argument(
        "--tracking-link", "--ee-link", dest="tracking_link", default=None,
        help="URDF frame represented by world_from_ee (default: fiducial inventory parent)",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--enforce-gates", action="store_true")
    args = ap.parse_args()

    records = [
        json.loads(line) for line in args.estimates.expanduser().read_text().splitlines() if line.strip()
    ]
    log = TeleopLog(args.teleop_log.expanduser())
    chain = UrdfChain(args.urdf)
    tracking_link = args.tracking_link or load_inventory().target("wrist").parent_frame
    if not tracking_link:
        raise ValueError("wrist target has no parent_frame")
    robot_world = json.loads(args.robot_world.expanduser().read_text())
    world_from_base = np.asarray(robot_world["world_from_base"], dtype=np.float64)
    measured = [record for record in records if record.get("status") == "measured"]
    translation_disagreement, rotation_disagreement = [], []
    compared = []
    for record in measured:
        timestamp_s = int(record["timestamp_ns"]) / 1e9
        joints = _interpolated_joints(log, timestamp_s)
        if joints is None:
            continue
        record_frame = record.get("tracking_frame")
        if record_frame is not None and record_frame != tracking_link:
            raise ValueError(
                f"estimate tracking_frame {record_frame!r} differs from {tracking_link!r}"
            )
        values = dict(
            zip(chain.driver_joint_names("right", len(joints)), joints, strict=True)
        )
        world_from_ee_fk = world_from_base @ chain.link_pose(tracking_link, values)
        world_from_ee_vision = np.asarray(record["world_from_ee"], dtype=np.float64)
        delta = invert(world_from_ee_fk) @ world_from_ee_vision
        translation_mm = float(1000 * np.linalg.norm(delta[:3, 3]))
        rotation_deg = rotation_distance_deg(world_from_ee_fk, world_from_ee_vision)
        translation_disagreement.append(translation_mm)
        rotation_disagreement.append(rotation_deg)
        compared.append((timestamp_s, world_from_ee_vision))

    stationary_translation, stationary_rotation = [], []
    intervals = log.still_intervals(tolerance_rad=0.003, min_duration=0.5)
    for interval in intervals:
        poses = [
            pose
            for timestamp, pose in compared
            if interval["start_unix"] <= timestamp <= interval["end_unix"]
        ]
        if len(poses) < 3:
            continue
        center = _mean_pose(poses)
        for pose in poses:
            stationary_translation.append(float(1000 * np.linalg.norm(pose[:3, 3] - center[:3, 3])))
            stationary_rotation.append(rotation_distance_deg(center, pose))

    statuses = Counter(record.get("status", "missing") for record in records)
    detected_by_camera, detected_by_tag = Counter(), Counter()
    for record in records:
        for camera, detections in record.get("detections", {}).items():
            if detections:
                detected_by_camera[camera] += 1
            for detection in detections:
                detected_by_tag[str(detection["tag_id"])] += 1
    latencies = [float(record["latency_ms"]) for record in records if record.get("latency_ms") is not None]
    detector_latencies = [
        float(record["detection_latency_ms"])
        for record in records
        if record.get("detection_latency_ms") is not None
    ]
    solver_latencies = [
        float(record["solver_latency_ms"])
        for record in records
        if record.get("solver_latency_ms") is not None
    ]
    total = max(1, len(records))
    report = {
        "schema_version": 1,
        "tracking_frame": tracking_link,
        "estimates": len(records),
        "statuses": dict(statuses),
        "measured_rate": statuses["measured"] / total,
        "fk_compared": len(compared),
        "vision_fk_translation_disagreement_mm": _percentiles(translation_disagreement),
        "vision_fk_rotation_disagreement_deg": _percentiles(rotation_disagreement),
        "stationary_translation_jitter_mm": _percentiles(stationary_translation),
        "stationary_rotation_jitter_deg": _percentiles(stationary_rotation),
        "latency_ms": _percentiles(latencies),
        "detection_latency_ms": _percentiles(detector_latencies),
        "solver_latency_ms": _percentiles(solver_latencies),
        "rejection_reasons": dict(
            Counter(record.get("reason") for record in records if record.get("reason"))
        ),
        "detection_frames_by_camera": dict(sorted(detected_by_camera.items())),
        "detections_by_tag": dict(sorted(detected_by_tag.items())),
        "calibration_ids": sorted(
            {record.get("calibration_id") for record in records if record.get("calibration_id")}
        ),
        "layout_hashes": sorted(
            {record.get("wrist_layout_hash") for record in records if record.get("wrist_layout_hash")}
        ),
    }
    failures = []
    if report["measured_rate"] < 0.80:
        failures.append("measured rate below 0.80")
    if report["latency_ms"]["p95"] is None or report["latency_ms"]["p95"] > 150:
        failures.append("latency p95 above 150 ms")
    if (
        report["vision_fk_translation_disagreement_mm"]["median"] is None
        or report["vision_fk_translation_disagreement_mm"]["median"] > 10
    ):
        failures.append("vision/FK translation disagreement median above 10 mm")
    if (
        report["vision_fk_translation_disagreement_mm"]["p95"] is None
        or report["vision_fk_translation_disagreement_mm"]["p95"] > 25
    ):
        failures.append("vision/FK translation disagreement p95 above 25 mm")
    if stationary_translation and np.percentile(stationary_translation, 95) > 3:
        failures.append("stationary translation jitter p95 above 3 mm")
    if stationary_rotation and np.percentile(stationary_rotation, 95) > 1:
        failures.append("stationary rotation jitter p95 above 1 deg")
    report["passed"] = not failures
    report["gate_failures"] = failures
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.enforce_gates and failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
