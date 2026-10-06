#!/usr/bin/env python3
"""Retained fiducial size/depth evidence through the shared RGB-D readers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def measurement(path):
    import numpy as np

    data = path.read_bytes()
    record = json.loads(data)
    edge, uncertainty = record['edge_m'], record.get('uncertainty_m')
    if (record.get('target') != 'board' or type(edge) not in (int, float) or not np.isfinite(edge) or not 0 < edge < 1
            or not isinstance(record.get('source'), str) or not record['source'].strip()):
        raise ValueError('measurement needs a positive metre edge and explicit independent source')
    if uncertainty is not None and (type(uncertainty) not in (int, float)
                                   or not np.isfinite(uncertainty) or not 0 <= uncertainty < edge):
        raise ValueError('measurement uncertainty must be finite metres or unavailable')
    return {**record, 'uncertainty_m': uncertainty, 'sha256': hashlib.sha256(data).hexdigest()}


def interior_pixels(corners, shape):
    import cv2
    import numpy as np

    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    transform = cv2.getPerspectiveTransform(unit, np.asarray(corners, np.float32))
    x, y = np.meshgrid(np.linspace(.1, .9, 9), np.linspace(.1, .9, 9))
    pixels = np.rint(cv2.perspectiveTransform(np.c_[x.ravel(), y.ravel()][None], transform)[0]).astype(int)
    if np.any(pixels < 0) or np.any(pixels >= [shape[1], shape[0]]):
        raise ValueError('tag interior falls outside its original image')
    return pixels


def plane_comparison(frame, pixels, rotation, translation):
    import numpy as np
    from rgbd_geometry import color_depth_m, deproject_pixels, paired_alignment

    intr = frame['camera_model']
    ca, da = (frame[key]['attributes'] for key in ('color_metadata', 'depth_metadata'))
    rays = deproject_pixels(intr, pixels)
    native_z = frame['depth_m'][pixels[:, 1], pixels[:, 0]]
    units = float(da['depth_units_m'])
    if not np.isfinite(units) or units <= 0:
        raise ValueError('invalid original depth units')
    z = color_depth_m(native_z, rays, paired_alignment(ca, da, intr), intr)
    normal = rotation[:, 2]
    denominator = rays @ normal
    predicted = np.divide(normal @ translation, denominator,
                          out=np.full(len(rays), np.nan), where=np.abs(denominator) > 1e-12)
    valid = (np.isfinite(native_z) & (native_z > 0) & (native_z < 65535*units)
             & (z > 0) & np.isfinite(predicted) & (predicted > 0))
    errors = z[valid] - predicted[valid]
    return {'sample_pixels': pixels.tolist(), 'depth_supported': valid.tolist(),
            'supported_samples': int(valid.sum()), 'total_samples': len(pixels),
            'median_depth_minus_rgb_plane_m': float(np.median(errors)) if len(errors) else None,
            'max_absolute_depth_minus_rgb_plane_m': float(np.max(np.abs(errors))) if len(errors) else None}


def tag_evidence(frame, detection, edge):
    import cv2
    import numpy as np
    from fiducials.geometry import tag_model_corners
    from rgbd_geometry import deproject_pixels

    corners = np.asarray(detection.corners_px, float)
    rays = deproject_pixels(frame['camera_model'], corners)
    target = tag_model_corners(edge)
    result = cv2.solvePnPGeneric(target, np.ascontiguousarray(rays[:, :2]), np.eye(3), None,
                               flags=cv2.SOLVEPNP_IPPE_SQUARE)
    pixels = interior_pixels(corners, frame['image'].shape)
    solutions = []
    for vector, offset in zip(result[1], result[2], strict=True):
        rotation, translation = cv2.Rodrigues(vector)[0], offset.reshape(3)
        points = target @ rotation.T + translation
        if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
            continue
        normalized = points / points[:, 2, None]
        residual = np.sqrt(np.mean(np.sum((normalized[:, :2]-rays[:, :2])**2, axis=1)))
        solutions.append({'camera_from_tag_rotation': rotation.tolist(), 'camera_from_tag_translation_m': translation.tolist(),
                          'corner_normalized_ray_rmse': float(residual),
                          **plane_comparison(frame, pixels, rotation, translation)})
    return {'family': detection.family, 'id': detection.tag_id, 'corners_px': corners.tolist(),
            'measured_edge_m': edge, 'pose_branches': solutions,
            'branch_selection': 'unselected; retain every positive-depth planar solution'}


def implementation():
    import cv2
    import numpy as np

    paths = {'board_witness': Path(__file__)}
    for name in ('live_inputs', 'rgbd_geometry', 'fiducials.detector', 'stencil_surface'):
        paths[name] = Path(sys.modules[name].__file__)
    return {'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
            'versions': {'python': sys.version.split()[0], 'opencv': cv2.__version__, 'numpy': np.__version__}}


def prepare(captures, color_sensor, inventory_path, measurement_path, output, *, pose_inputs=None, mount_fit_indices=None):
    import schemas
    from fiducials import load_inventory
    from fiducials.detector import DetectorConfig, FiducialDetector
    from live_inputs import load_frames, rgbd_pair

    if not 1 <= len(captures) <= 64:
        raise ValueError('board witness needs 1–64 retained captures')
    if mount_fit_indices is not None and pose_inputs is None:
        raise ValueError('mount fit requires retained pose inputs')
    measured = measurement(measurement_path)
    inventory = load_inventory(inventory_path)
    target = inventory.target('board')
    detector = FiducialDetector.from_inventory(inventory,
        DetectorConfig.from_profile(inventory.detector_profiles['calibration']), target='board')
    manifests = [load_frames(path)[0] for path in captures]
    posed = None
    if pose_inputs is not None:
        from board_mount_witness import WristInputs
        posed = WristInputs(pose_inputs, manifests)
    records = []
    for path in captures:
        manifest, frames = load_frames(path)
        frame = rgbd_pair(frames, color_sensor)
        found = detector.detect(color_sensor, frame['image'], frame['timestamp_ns'])
        record = {'capture_file': str(path.resolve()), 'capture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'payloads_sha256': {name: entry['sha256'] for name, (entry, _) in frames.items()},
                        'source_metadata': {name: entry['metadata'] for name, (entry, _) in frames.items()},
                        'joint_read': manifest.get('joint_read'), 'wrist': manifest.get('wrist'),
                  'tags': [tag_evidence(frame, item, measured['edge_m']) for item in found]}
        if posed is not None:
            record['wrist_pose'] = posed.pose(path, manifest, color_sensor)
        records.append(record)
    report = {**schemas.stamp(schemas.RECEIPT, 'fiducial-board-witness'), 'execution': 'none',
              'implementation': implementation(),
              'measurement': measured, 'inventory_sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
              'configured_edge_m': target.edge_m, 'captures': records,
              'physical_error_bound_m': None, 'mount_error_bound_m': None,
              'scope': 'observed metric discrepancies only; planar ambiguity, optics, board flatness and measurement uncertainty remain',
              'calibration_adopted': False}
    if posed is not None:
        from board_mount_witness import consistency
        report['wrist_inputs_sha256'] = posed.hashes
        report['mount_consistency'] = consistency(records)
        if mount_fit_indices is not None:
            from board_mount_witness import fit_mount
            report['mount_fit'] = fit_mount(records, mount_fit_indices)
        for name in ('board_mount_witness', 'stencil_observer', 'urdf_kinematics', 'robot_world', 'wrist_cameras'):
            report['implementation']['source_sha256'][name] = hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output/'board-witness.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, nargs='+', required=True)
    for flag in ('inventory', 'measurement', 'out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--color-sensor', required=True)
    parser.add_argument('--pose-inputs', type=Path, help='retained wrist configuration root; optional static-tag consistency only')
    parser.add_argument('--mount-fit-indices', type=int, nargs='+', help='3–8 capture indices for unadopted hand-eye candidates; exclude validation views')
    args = parser.parse_args()
    import tatbot_runlog

    run = tatbot_runlog.init('vision-board-witness', prune_first=False)
    status = 1
    try:
        report = prepare(args.capture, args.color_sensor, args.inventory, args.measurement, args.out,
                         pose_inputs=args.pose_inputs, mount_fit_indices=args.mount_fit_indices)
        run.artifact(args.out, name='board-witness')
        print(json.dumps({'output': str(args.out), 'tags_per_capture': [len(item['tags']) for item in report['captures']],
                          'calibration_adopted': False}))
        status = 0
    finally:
        run.finalize(status, status='ok' if status == 0 else 'fail')
        print(tatbot_runlog.banner('end', run.run_id, 'exit='+str(status)))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
