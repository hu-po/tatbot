"""Planar print-size pose candidates when modeled RGB has no usable depth."""

import cv2
import numpy as np


def camera_model(value, shape):
    if (value['width'], value['height']) != (shape[1], shape[0]):
        raise ValueError('camera_model_image_dimensions_mismatch')
    if value['distortion_model'] not in ('None', 'BrownConrady'):
        raise ValueError('unsupported_rgb_pose_distortion_model')
    coeff = np.asarray(value['distortion_coefficients'], float)
    values = [value[key] for key in ('fx', 'fy', 'ppx', 'ppy')]
    if len(coeff) != 5 or not np.isfinite([*values, *coeff]).all() or min(values[:2]) <= 0:
        raise ValueError('invalid_rgb_camera_model')
    if value['distortion_model'] == 'None' and np.any(coeff):
        raise ValueError('nonzero_distortion_with_none_model')
    matrix = np.array([[value['fx'], 0, value['ppx']], [0, value['fy'], value['ppy']], [0, 0, 1.]])
    return matrix, coeff


def fit_print(points, pixels, matrix, distortion):
    if np.linalg.matrix_rank(points-points.mean(axis=0)) < 2:
        return []
    ok, rotations, translations, _ = cv2.solvePnPGeneric(points, pixels, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return []
    candidates = []
    for rotation, translation in zip(rotations, translations, strict=True):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(rotation)[0]
        pose[:3, 3] = translation.ravel()
        if not np.isfinite(pose).all() or np.min((points@pose[:3, :3].T+pose[:3, 3])[:, 2]) <= 0:
            continue
        predicted = cv2.projectPoints(points, rotation, translation, matrix, distortion)[0].reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((predicted-pixels)**2, axis=1))))
        candidates.append({'pose': pose, 'rmse_px': error})
    return sorted(candidates, key=lambda row: row['rmse_px'])


def print_uncertainty(points, pixels, matrix, distortion, pose):
    rng = np.random.default_rng(19)
    deltas = []
    for _ in range(64):
        indices = rng.integers(0, len(points), len(points))
        fits = fit_print(points[indices], pixels[indices], matrix, distortion)
        if not fits:
            continue
        candidate = fits[0]['pose']
        rotation = cv2.Rodrigues(pose[:3, :3].T@candidate[:3, :3])[0].ravel()
        deltas.append(np.r_[candidate[:3, 3]-pose[:3, 3], rotation])
    if len(deltas) < 24:
        return {'available': False, 'reason': 'degenerate_bootstrap_samples'}
    covariance = np.cov(np.asarray(deltas).T)
    return {'available': True, 'calibrated': False, 'samples': len(deltas),
            'covariance_6x6': covariance.tolist(),
            'translation_std_m': np.sqrt(np.maximum(np.diag(covariance)[:3], 0)).tolist(),
            'rotation_std_deg': np.degrees(np.sqrt(np.maximum(np.diag(covariance)[3:], 0))).tolist(),
            'convention': 'translation in camera XYZ meters; local rotation vector in stencil XYZ radians',
            'method': 'correspondence bootstrap conditional on print size and camera model',
            'excludes': ['print-scale error', 'intrinsic error', 'systematic correspondence error', 'surface curvature']}


def estimate_print_pose(observation, frame, reference):
    matrix, distortion = camera_model(frame['camera_model'], frame['image'].shape)
    uv = np.asarray([row['reference_uv'] for row in observation['landmarks']], float)
    pixels = np.asarray([row['image_px'] for row in observation['landmarks']], float)
    points = np.column_stack(((uv-.5)*np.asarray(reference['page_mm'])/1000, np.zeros(len(uv))))
    candidates = fit_print(points, pixels, matrix, distortion)
    if not candidates or candidates[0]['rmse_px'] > 3:
        return {'status': 'unavailable', 'reason': 'nominal_print_pose_inconsistent', 'candidate_valid': False}
    pose = candidates[0]['pose']
    ambiguous = len(candidates) > 1 and candidates[1]['rmse_px']-candidates[0]['rmse_px'] < .5
    return {'status': 'candidate', 'candidate_valid': True, 'reason': 'nominal_print_planar_assumption',
            'model': 'planar_pose_from_nominal_print_size', 'page_mm': reference['page_mm'],
            'print_scale_measured': reference['dimensions_measured'], 'surface_shape_measured': False,
            'camera_from_stencil_center': pose.tolist(), 'reprojection_rmse_px': candidates[0]['rmse_px'],
            'pose_branch_ambiguous': ambiguous,
            'branches': [{'camera_from_stencil_center': row['pose'].tolist(), 'rmse_px': row['rmse_px']}
                         for row in candidates],
            'uncertainty': print_uncertainty(points, pixels, matrix, distortion, pose),
            'calibration_warnings': ['nominal_print_scale', 'planarity_assumed', 'intrinsics_not_independently_validated']}
