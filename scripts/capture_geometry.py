"""Wrist and overhead capture deprojection, and the overhead camera's registration into the URDF root.

`tatbot vision surface rgbd` imports the owner-profile rays. Capture and surface file formats:
docs/surface-formats.md.

numpy only; `rgbd_geometry` and `robot_world` are imported where they are used.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()

DEPTH_MIN_M, DEPTH_MAX_M = 0.07, 0.5   # the D405 depth the map trusts
OVERHEAD_DEPTH_RANGE_M = (0.3, 1.5)


class StageError(RuntimeError):
    pass


class OverheadUnboundError(StageError):
    """The overhead view's live optics, profile or alignment do not bind to the
    retained calibration. Retained evidence is intact; the view
    simply cannot be trusted into this surface, and the wrist views carry it."""


def _deproject_overhead(depth, units, camera, color_metadata, depth_metadata):
    from rgbd_geometry import (
        deproject_aligned,
        document,
        paired_alignment,
        require_calibrated_model,
        verify_pair,
    )
    try:
        verify_pair(color_metadata, depth_metadata)
        ca, da = color_metadata['attributes'], depth_metadata['attributes']
        intr = document(da['intrinsics'])
        if da.get('aligned_to') != color_metadata['sensor_name'] or document(ca['intrinsics']) != intr:
            raise ValueError('overhead RGB-D alignment differs')
        require_calibrated_model(intr, camera, da.get('device_serial'))
        return deproject_aligned(depth, units, intr, paired_alignment(ca, da, intr), OVERHEAD_DEPTH_RANGE_M)
    except (ValueError, KeyError, TypeError) as error:
        raise OverheadUnboundError(str(error)) from error


def _owner_intrinsics(profile_json: str):
    """Validate the active deprojection model for full images or corners."""
    profile = json.loads(profile_json)
    intr = profile['intrinsics']
    if intr.get('schema') != 'tatbot.camera-intrinsics/1':
        raise StageError('unknown owner intrinsics schema')
    # Modified Brown-Conrady is projection-only; never silently approximate it.
    if intr['distortion_model'] not in ('None', 'BrownConradyInverse'):
        raise StageError('unsupported owner deprojection distortion model')
    values = [intr[key] for key in ('fx', 'fy', 'ppx', 'ppy')]
    if (intr['width'] != 640 or intr['height'] != 480 or min(values[:2]) <= 0
            or not np.isfinite([*values, *intr['distortion_coefficients']]).all()):
        raise StageError('invalid owner deprojection profile')
    return intr


@lru_cache(maxsize=4)
def _owner_rays(profile_json: str) -> np.ndarray:
    """Use the camera SDK's distortion convention; cache only four active profiles."""
    from rgbd_geometry import camera_rays
    intr = _owner_intrinsics(profile_json)
    try:
        return camera_rays(json.dumps(intr, sort_keys=True))
    except ValueError as error:
        raise StageError(str(error)) from error


def _deproject_owner(depth, units, profile_json):
    profile = json.loads(profile_json)
    if not np.isfinite(units) or units <= 0 or units != profile['units_m']:
        raise StageError('owner depth units disagree with capture profile')
    rays = _owner_rays(profile_json)
    depth = np.asarray(depth)
    if depth.shape != rays.shape[:2] or depth.dtype != np.uint16:
        raise StageError('owner depth dimensions or encoding changed')
    z = depth.astype(np.float64) * units
    valid = (depth > 0) & (depth < 65535) & (z >= DEPTH_MIN_M) & (z <= DEPTH_MAX_M)
    if 'alignment_calibration' in profile:
        from rgbd_geometry import color_depth_m
        try:
            z = color_depth_m(z, rays, profile['alignment_calibration'], profile['intrinsics'])
        except ValueError as error:
            raise StageError(str(error)) from error
    valid &= z > 0
    return rays[valid] * z[valid, None]


def _verified_overhead_payload(capture_dir: Path, entry: dict) -> bytes:
    path = (capture_dir / entry["payload_file"]).resolve()
    if not path.is_relative_to(capture_dir.resolve()):
        raise StageError("overhead payload escapes capture directory")
    data = path.read_bytes()
    if len(data) != int(entry["payload_bytes"]):
        raise StageError(f"overhead payload length changed: {path}")
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise StageError(f"overhead payload digest changed: {path}")
    return data


def _pose_matrix(value: dict, label: str) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(value["rotation"], float).reshape(3, 3)
    matrix[:3, 3] = np.asarray(value["translation_m"], float).reshape(3)
    if not np.isfinite(matrix).all() or not np.allclose(
            matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5):
        raise StageError(f"{label} is not a finite rigid transform")
    return matrix


def _overhead_registration(calibration, robot):
    from robot_world import root_from_world
    bundle_id = calibration.get('bundle_id')
    if not bundle_id or robot.get('calibration_id') != bundle_id:
        raise OverheadUnboundError('overhead camera and robot-world calibration IDs differ')
    depth = calibration.get('cameras', {}).get('overhead_depth_depth')
    color = calibration.get('cameras', {}).get('overhead_depth_color')
    if depth is None or color is None:
        raise StageError('retained bundle lacks aligned D555 color/depth entries')
    if (depth['world_from_camera'] != color['world_from_camera']
            or depth['intrinsics'] != color['intrinsics'] or depth.get('distortion') != color.get('distortion')):
        raise StageError('retained D555 aligned color/depth geometry differs')
    world_from_camera = _pose_matrix(depth['world_from_camera'], 'D555 world_from_camera')
    try:
        return bundle_id, depth, root_from_world(robot) @ world_from_camera
    except ValueError as error:
        raise StageError(f'robot world_from_base is not a rigid URDF-root transform: {error}') from error
