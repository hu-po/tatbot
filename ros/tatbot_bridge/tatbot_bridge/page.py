"""The page frame from stencild's target frame (ros/README.md section 3), and the sample it comes in.

stencild's `world_from_target` (tatbot.target-pose/1 on tatbot/tracking/target/<pattern_id>) is the
print's centre with x along reference u (right) and y along reference v (down the image), so z points
into the paper (scripts/vision/stencil_pose.py object points: ((uv - 0.5) * page_mm, 0)). The page
frame keeps x, flips y to the top of the print (-v) and z out of the paper: a half turn about x.

Pure numpy, no ROS: the node, the page watch and the tests share it.
"""
from __future__ import annotations

import json
import math

import numpy as np
from tatbot_description.transforms import matrix_quat as quaternion  # noqa: F401 -- existing page API
from tatbot_description.transforms import rpy_matrix as from_xyz_rpy  # noqa: F401

SCHEMA = "tatbot.target-pose/1"
TARGET_FROM_PAGE = np.diag([1.0, -1.0, -1.0, 1.0])


def topic(pattern_id: str) -> str:
    return f"tatbot/tracking/target/{pattern_id}"


def world_from_page(world_from_target) -> np.ndarray:
    return np.asarray(world_from_target, float) @ TARGET_FROM_PAGE


def base_from_page(world_from_arm_base, world_from_target) -> np.ndarray:
    """The page in an arm's base: inv(world_from_arm_base) @ world_from_page. This is the crossing the
    frozen session used (target_pose.to_arm_base), with the page frame's axes."""
    return np.linalg.inv(np.asarray(world_from_arm_base, float)) @ world_from_page(world_from_target)


def rigid(value) -> np.ndarray:
    """A finite rigid 4x4, or ValueError."""
    m = np.asarray(value, float)
    if (m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(m[:3, :3]), 1.0, atol=1e-6)):
        raise ValueError("pose is not a finite rigid 4x4 matrix")
    return m


def parse(data: bytes | str, pattern_id: str | None = None) -> dict | None:
    """One stencild sample as a flat dict, or None when it is not this print's target-pose.

    A `measured` sample carries `world_from_target`; a `lost` one carries None.
    """
    try:
        envelope = json.loads(data)
    except (ValueError, TypeError):
        return None
    if not isinstance(envelope, dict) or envelope.get("schema") != SCHEMA:
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or (pattern_id and payload.get("target_id") != pattern_id):
        return None
    support = payload.get("support") if isinstance(payload.get("support"), dict) else {}
    source = payload.get("source")
    pose = None
    if source == "measured":
        try:
            pose = rigid(payload.get("world_from_target"))
        except (ValueError, TypeError):
            return None
    elif source != "lost":
        return None

    def number(key):
        value = payload.get(key)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else math.nan

    return {
        "pattern_id": payload.get("target_id", ""),
        "source": source,
        "world_from_target": pose,
        "translation_sigma_m": number("translation_sigma_m"),
        "rotation_sigma_rad": number("rotation_sigma_rad"),
        "stamp_ns": int((envelope.get("stamp") or {}).get("wall_ns") or 0),
        "seq": envelope.get("seq"),
        "calibration_id": payload.get("calibration_id") or "",
        "print_id": str(support.get("physical_instance_id") or ""),
        "identity_verified": support.get("physical_instance_identity_verified") is True,
        "support": support,
    }


def covariance(translation_sigma_m: float, rotation_sigma_rad: float) -> list[float]:
    """Row-major 6x6 (x y z rx ry rz): the sigmas squared on the diagonal; -1 on the first entry when
    unknown (the ROS convention for an unknown covariance)."""
    cov = [0.0] * 36
    if not (math.isfinite(translation_sigma_m) and math.isfinite(rotation_sigma_rad)):
        cov[0] = -1.0
        return cov
    for i in range(3):
        cov[7 * i] = translation_sigma_m ** 2
        cov[7 * (i + 3)] = rotation_sigma_rad ** 2
    return cov


def rpy_deg(rotation) -> tuple[float, float, float]:
    """Roll, pitch, yaw in degrees (for logs)."""
    r = np.asarray(rotation, float)
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    return (math.degrees(math.atan2(r[2, 1], r[2, 2])), math.degrees(pitch), math.degrees(math.atan2(r[1, 0], r[0, 0])))
