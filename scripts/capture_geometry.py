"""Wrist and overhead capture geometry: deprojection, capture views, registration.

The helpers the stencil observer,
`tatbot vision surface rgbd` and the tracking views import. Capture and surface file formats: docs/surface-formats.md.

numpy only; `rgbd_geometry`, `wrist_cameras`, `depth_quality` and
`robot_world` are imported where they are used.
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

import arm_kinematics as dk  # noqa: E402

DEPTH_MIN_M, DEPTH_MAX_M = dk.D405_DEPTH_RANGE_M   # the D405 depth the map trusts
OVERHEAD_DEPTH_RANGE_M = (0.3, 1.5)
CAMERA_LINKS = dk.CAMERA_LINKS


class StageError(RuntimeError):
    pass


def _deproject(depth: np.ndarray, units_m: float, intrinsics,
               depth_range=(DEPTH_MIN_M, DEPTH_MAX_M)) -> np.ndarray:
    """Pinhole deprojection of a uint16 depth image into the camera optical frame (N,3)."""
    fx, fy, ppx, ppy = (float(v) for v in np.asarray(intrinsics, float)[:4])
    depth = np.asarray(depth)
    valid = (depth > 0) & (depth < 65535)
    z = depth.astype(np.float64) * float(units_m)
    valid &= (z >= depth_range[0]) & (z <= depth_range[1])
    rows, cols = np.nonzero(valid)
    z = z[rows, cols]
    x = (cols.astype(np.float64) - ppx) / fx * z
    y = (rows.astype(np.float64) - ppy) / fy * z
    return np.stack([x, y, z], axis=1)


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


def declared_roles(npz) -> list[str]:
    """The views a capture declares (`camera_roles`): one arm's retained roles,
    and every owner-frame field it holds is one of them."""
    from wrist_cameras import capture_arm

    if 'camera_roles' not in npz:
        raise ValueError('capture lacks declared camera roles')
    roles = json.loads(str(npz['camera_roles']))
    if not isinstance(roles, list):
        raise ValueError('invalid capture camera roles')
    capture_arm(roles)
    observed = {key.removeprefix('owner_frames_') for key in npz if key.startswith('owner_frames_')}
    if observed and observed != set(roles):
        raise ValueError('capture owner frames differ from declared camera roles')
    return roles


def capture_views(npz) -> tuple[str | None, dict[str, str]]:
    """The arm a capture belongs to and that arm's role -> depth optical frame
    mapping (`CAMERA_LINKS[arm]`): the declared `camera_roles` (every capture
    the owner writes), else the depth views present (legacy captures), one
    arm's retained roles with every present view mapped on that arm's own
    URDF chain. A view the mapping lacks is refused by name, never registered
    through another arm's transform. A capture holding no depth view belongs
    to no arm and maps nothing: `(None, {})`."""
    from wrist_cameras import capture_arm

    present = {key.removeprefix('depth_') for key in npz if key.startswith('depth_')}
    if 'camera_roles' in npz:
        try:
            declared = set(declared_roles(npz))
        except ValueError as error:
            raise StageError(f'capture camera roles are invalid: {error}') from error
        if declared != present:
            raise StageError('capture depth fields differ from declared camera roles')
    if not present:
        return None, {}
    try:
        arm = capture_arm(sorted(present))
    except ValueError as error:
        raise StageError(f'capture camera roles {sorted(present)}: {error}') from error
    links = CAMERA_LINKS[arm]
    if present - links.keys():
        raise StageError(f'capture camera roles {sorted(present - links.keys())} are absent from the '
                         f'{arm} arm URDF mapping {sorted(links)}; use its original model')
    return arm, links


def _pinhole_capture_points(npz, role, depth, units):
    from capture_projection import validate as validate_projection
    from wrist_cameras import describe

    if f'owner_profile_{role}' in npz.files:
        raise StageError('capture has conflicting device and pinhole projection profiles')
    camera = next((camera for camera in describe(REPO) if camera.role == role), None)
    if camera is None:
        raise StageError('capture projection has no configured camera')
    try:
        link = validate_projection(json.loads(str(npz[f'projection_profile_{role}'])), camera,
            shape=npz[f'depth_{role}'].shape, intrinsics=npz[f'intrinsics_{role}'], units_m=units)
    except (ValueError, TypeError, KeyError) as error:
        raise StageError(str(error)) from error
    return _deproject(depth, units, npz[f'intrinsics_{role}']), link


def _capture_points(npz, role, depth, units, link):
    if f'projection_profile_{role}' in npz.files:
        return _pinhole_capture_points(npz, role, depth, units)
    if f'owner_profile_{role}' in npz.files:
        # The device owner explicitly aligns depth into the color optical frame.
        return (_deproject_owner(depth, units, str(npz[f'owner_profile_{role}'])),
                link.replace('_depth_optical_frame', '_color_optical_frame'))
    return _deproject(depth, units, npz[f'intrinsics_{role}']), link


def _capture_cloud(npz, chain, selected_role=None, quality=None, quality_reports=None) -> tuple[np.ndarray, list[str]]:
    """All valid depth pixels of one capture in root, plus the roles that contributed."""
    joints = np.asarray(npz["joints"], float)
    carriage = float(np.asarray(npz["carriage_m"]).reshape(-1)[0])
    if joints.shape != (6,) or not np.isfinite(joints).all() or not np.isfinite(carriage):
        raise StageError("capture has no finite measured arm/carriage pose; registration refused")
    arm, links = capture_views(npz)
    values = dk.ArmModel(arm, chain=chain).joint_map(joints, carriage) if links else {}
    clouds = []
    roles = []
    for role, link in links.items():
        if selected_role is not None and role != selected_role:
            continue
        key = f"depth_{role}"
        if key not in npz.files:
            continue
        units = float(np.asarray(npz[f"units_m_{role}"]).reshape(-1)[0])
        from depth_quality import filter_depth
        frame_count = int(npz[f'frames_{role}']) if f'frames_{role}' in npz.files else 8
        depth, quality_report = filter_depth(npz[key], units,
                               npz[f'valid_{role}'] if f'valid_{role}' in npz.files else None,
                               frame_count,
                               npz[f'temporal_mad_m_{role}'] if f'temporal_mad_m_{role}' in npz.files else None,
                               options=quality)
        points, link = _capture_points(npz, role, depth, units, link)
        if quality_reports is not None:
            quality_reports[role] = {**quality_report, 'optical_frame': link}
            if f'projection_profile_{role}' in npz.files:
                quality_reports[role]['projection_profile'] = json.loads(str(npz[f'projection_profile_{role}']))
        if len(points) == 0:
            continue
        pose = chain.link_pose(link, values)
        clouds.append(points @ pose[:3, :3].T + pose[:3, 3])
        roles.append(role)
    if not clouds:
        return np.zeros((0, 3)), roles
    return np.concatenate(clouds), roles


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
