"""Synthetic sweep-session artifacts for the field-calibration tests.

Builds byte-accurate .wxtl flight logs (format from cpp/teleop/wxai_teleop.cpp)
and pinhole+Brown-Conrady cameras whose FORWARD projection mirrors the model
fuse_session.Camera inverts — so the tests exercise the real parsing and the
real geometry, not mocks of them.
"""

from __future__ import annotations

import json
import struct

import numpy as np

NUM_JOINTS = 7  # six arm joints + gripper, like the real follower
PERIOD_S = 0.0025


def write_wxtl(path, poses, efforts, wall_start_ns=1_755_000_000_000_000_000,
               still_s=1.0, move_s=0.3, period_s=PERIOD_S):
    """One still interval per pose, ramps between them. Returns still centers
    (absolute unix seconds) in pose order."""
    header = struct.pack("<8sQddddQq", b"WXTLOG1\0", NUM_JOINTS, period_s,
                         0.0, 0.0, 0.0, 1, wall_start_ns)
    rng = np.random.default_rng(3)
    records = []
    centers = []
    t = 0.0

    def emit(joints, effort, seconds):
        nonlocal t
        for _ in range(int(round(seconds / period_s))):
            row = np.zeros(5 + 6 * NUM_JOINTS)
            row[1] = t
            row[5 + 2 * NUM_JOINTS:5 + 3 * NUM_JOINTS] = (
                joints + rng.normal(0, 0.0003, NUM_JOINTS))
            row[5 + 4 * NUM_JOINTS:5 + 5 * NUM_JOINTS] = (
                effort + rng.normal(0, 0.02, NUM_JOINTS))
            records.append(row)
            t += period_s

    for index, (pose, effort) in enumerate(zip(poses, efforts, strict=True)):
        start = t
        emit(np.asarray(pose, float), np.asarray(effort, float), still_s)
        centers.append(wall_start_ns / 1e9 + (start + t) / 2.0)
        if index + 1 < len(poses):
            here, there = np.asarray(pose, float), np.asarray(poses[index + 1], float)
            steps = int(round(move_s / period_s))
            for k in range(steps):
                blend = (k + 1) / steps
                row = np.zeros(5 + 6 * NUM_JOINTS)
                row[1] = t
                row[5 + 2 * NUM_JOINTS:5 + 3 * NUM_JOINTS] = here * (1 - blend) + there * blend
                row[5 + 4 * NUM_JOINTS:5 + 5 * NUM_JOINTS] = rng.normal(0, 0.02, NUM_JOINTS)
                records.append(row)
                t += period_s
    payload = np.asarray(records).astype("<f8").tobytes()
    path.write_bytes(header + payload)
    return centers


def look_at_rotation(position, target):
    """world_from_camera rotation: camera +z looks from position toward target."""
    forward = np.asarray(target, float) - np.asarray(position, float)
    forward = forward / np.linalg.norm(forward)
    up_hint = np.array([0.0, 1.0, 0.0])
    if abs(forward @ up_hint) > 0.95:
        up_hint = np.array([1.0, 0.0, 0.0])
    right = np.cross(up_hint, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward])


def make_camera(position, target, fx=900.0, fy=895.0, cx=640.0, cy=360.0,
                dist=(0.05, -0.02, 0.001, -0.001, 0.005)):
    return {
        "position": np.asarray(position, float),
        "rotation": look_at_rotation(position, target),
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "dist": np.asarray(dist, float),
    }


def project(camera, world_point):
    """Forward pinhole + Brown-Conrady, the model calibrateCamera fits."""
    cam = camera["rotation"].T @ (np.asarray(world_point, float) - camera["position"])
    x, y = cam[0] / cam[2], cam[1] / cam[2]
    k1, k2, p1, p2, k3 = camera["dist"]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.array([camera["fx"] * xd + camera["cx"],
                     camera["fy"] * yd + camera["cy"]])


def bundle_json(cameras):
    return {
        "schema_version": 1, "bundle_id": "f" * 64, "world_frame": "synthetic",
        "cameras": {
            name: {
                "sensor_name": name,
                "intrinsics": {"width": 1280, "height": 720,
                               "fx": cam["fx"], "fy": cam["fy"],
                               "cx": cam["cx"], "cy": cam["cy"]},
                "distortion": {"model": "brown_conrady",
                               "coefficients": cam["dist"].tolist()},
                "world_from_camera": {
                    "rotation": cam["rotation"].reshape(-1).tolist(),
                    "translation_m": cam["position"].tolist()},
                "metadata": {"reproj_rmse_px": "0.1"},
            } for name, cam in cameras.items()
        },
    }


def tag_corners_world(world_from_tag, edge_m):
    half = edge_m / 2.0
    model = np.array([[-half, half, 0, 1], [half, half, 0, 1],
                      [half, -half, 0, 1], [-half, -half, 0, 1]])
    return (world_from_tag @ model.T).T[:, :3]


def write_shot(session, index, unix_seconds, detections):
    shot = session / f"shot_{index:04d}_sweep"
    shot.mkdir(parents=True, exist_ok=True)
    (shot / "detections.json").write_text(json.dumps(detections))
    (shot / "timing.json").write_text(json.dumps(
        {"unix_seconds": unix_seconds, "cameras": sorted(detections)}))
    return shot
