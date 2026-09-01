#!/usr/bin/env python3
"""Replay a field-calibration wrist phase through the continuous EE solver.

Field sessions store undistorted normalized tag corners in ``pose_*.json``.
This converts them back to calibrated pixels, runs the exact shadow estimator,
and reports timestamp-aligned FK disagreement.  It provides a real baseline
without requiring another powered-arm session or re-running the old static
tag-pose fuser.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ee_fiducial import (  # noqa: E402
    Detection,
    EstimatorConfig,
    MultiCameraEstimator,
    WristLayout,
    invert,
    load_calibration,
    rotation_distance_deg,
)
from fiducials import load_inventory  # noqa: E402
from urdf_kinematics import UrdfChain  # noqa: E402


def _percentiles(values):
    return {
        "median": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "max": max(values) if values else None,
    }


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", type=Path)
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--robot-world", type=Path)
    ap.add_argument("--wrist-layout", type=Path, default=repo / "config/wrist_tags_measured.json")
    ap.add_argument("--urdf", type=Path, default=repo / "urdf/tatbot.urdf")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    session = args.session.expanduser()
    calibration_path = args.calibration or session / "calibration.json"
    robot_world_path = args.robot_world or session / "robot_world.json"
    cameras, bundle = load_calibration(calibration_path)
    inventory = load_inventory()
    layout = WristLayout.load(args.wrist_layout, inventory_path=inventory.source)
    robot_world = json.loads(robot_world_path.expanduser().read_text())
    world_from_base = np.asarray(robot_world["world_from_base"], dtype=np.float64)
    base_from_world = invert(world_from_base)
    chain = UrdfChain(args.urdf)
    estimator = MultiCameraEstimator(
        cameras,
        layout,
        EstimatorConfig(min_initial_tags=inventory.target("wrist").minimum_acquisition_ids or 1),
    )
    records, translations, rotations = [], [], []
    camera_visibility, tag_visibility = Counter(), Counter()
    for index, pose_path in enumerate(sorted(session.glob("pose_*.json"))):
        pose = json.loads(pose_path.read_text())
        timestamp_ns = int((pose["meta"]["start_unix"] + pose["meta"]["end_unix"]) * 0.5e9)
        detections = []
        for tag_text, observations in pose.get("corner_obs", {}).items():
            tag_id = int(tag_text)
            if tag_id not in layout.ee_from_tag:
                continue
            for observation in observations:
                camera = cameras[observation["camera"]]
                normalized = np.asarray(observation["normalized"], dtype=np.float64)
                object_points = np.c_[normalized, np.ones(4)]
                pixels, _ = cv2.projectPoints(
                    object_points,
                    np.zeros(3),
                    np.zeros(3),
                    camera.intrinsic,
                    camera.distortion,
                )
                pixels = pixels.reshape(4, 2)
                side_px = float(
                    np.mean(
                        [np.linalg.norm(pixels[corner] - pixels[(corner + 1) % 4]) for corner in range(4)]
                    )
                )
                detections.append(Detection(camera.name, tag_id, pixels, timestamp_ns, side_px))
                camera_visibility[camera.name] += 1
                tag_visibility[str(tag_id)] += 1
        started = time.perf_counter()
        estimate = estimator.estimate(detections, timestamp_ns)
        record = estimate.as_dict(
            calibration_id=bundle["bundle_id"],
            layout_hash=layout.layout_hash,
            inventory_hash=layout.inventory_hash,
            tracking_frame=layout.parent_frame,
            base_from_world=base_from_world,
            sequence=index,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        record["source_pose"] = pose_path.name
        record["vision_fk_translation_disagreement_mm"] = None
        record["vision_fk_rotation_disagreement_deg"] = None
        if estimate.status == "measured":
            names = pose.get("joint_names") or chain.driver_joint_names(
                "right", len(pose["joints"])
            )
            joint_values = dict(zip(names, pose["joints"], strict=True))
            world_from_ee_fk = world_from_base @ chain.link_pose(
                layout.parent_frame, joint_values
            )
            delta = invert(world_from_ee_fk) @ estimate.world_from_ee
            translation = float(1000 * np.linalg.norm(delta[:3, 3]))
            rotation = rotation_distance_deg(world_from_ee_fk, estimate.world_from_ee)
            record["vision_fk_translation_disagreement_mm"] = translation
            record["vision_fk_rotation_disagreement_deg"] = rotation
            translations.append(translation)
            rotations.append(rotation)
        records.append(record)
    statuses = Counter(record["status"] for record in records)
    report = {
        "schema_version": 1,
        "session": str(session),
        "calibration_id": bundle["bundle_id"],
        "wrist_layout_hash": layout.layout_hash,
        "inventory_hash": layout.inventory_hash,
        "poses": len(records),
        "statuses": dict(statuses),
        "measured_rate": statuses["measured"] / max(1, len(records)),
        "vision_fk_translation_disagreement_mm": _percentiles(translations),
        "vision_fk_rotation_disagreement_deg": _percentiles(rotations),
        "observations_by_camera": dict(sorted(camera_visibility.items())),
        "observations_by_tag": dict(sorted(tag_visibility.items())),
        "records": records,
    }
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
