"""Measured RGB-D interiors selected by a contemporaneous independent RGB view.

Depth and color stay on the original sensor samples. The other camera supplies
image correspondence only; its homography never creates or flattens geometry.
This is display evidence, with conditional visibility and uncalibrated error.
"""

import cv2
import numpy as np
from stencil_features import area
from stencil_interior import interior_mask
from stencil_pose import camera_model
from stencil_surface import camera_rays, rgbd_frame
from stencil_surface_fit import choose_model, design
from surface_attachment import project_measured_rays


def project_into_view(target, view):
    intrinsics = view['camera_model']
    if intrinsics['distortion_model'] == 'BrownConradyInverse':
        import json
        rays = camera_rays(json.dumps(intrinsics, sort_keys=True))
        if rays.shape[:2] != view['image'].shape[:2]:
            raise ValueError('camera_model_image_dimensions_mismatch')
        pixels, valid = project_measured_rays(target, rays)
        pixels[~valid] = np.nan
        return pixels
    matrix, distortion = camera_model(intrinsics, view['image'].shape)
    if intrinsics['distortion_model'] == 'BrownConrady' and len(distortion) == 5:
        # These fixed views have ordinary five-coefficient Brown distortion.
        # Vectorized projection avoids constructing a Rodrigues Jacobian for
        # every measured overhead pixel while retaining the exact model.
        points = np.asarray(target, dtype=np.float64)
        x, y = points[:, 0]/points[:, 2], points[:, 1]/points[:, 2]
        r2 = x*x+y*y
        k1, k2, p1, p2, k3 = distortion.ravel()
        radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        xd = x*radial+2*p1*x*y+p2*(r2+2*x*x)
        yd = y*radial+p1*(r2+2*y*y)+2*p2*x*y
        return np.column_stack((matrix[0, 0]*xd+matrix[0, 2],
                                matrix[1, 1]*yd+matrix[1, 2]))
    return cv2.projectPoints(target, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)


def rigid_transform(value):
    matrix = np.asarray(value, float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)):
        raise ValueError('cross_camera_transform_not_rigid')
    return matrix


def project_samples(frame, view, view_from_depth, depth_roi=None):
    rgbd, warnings = rgbd_frame(frame)
    if warnings:
        raise ValueError('; '.join(warnings))
    transform = rigid_transform(view_from_depth)
    depth = rgbd['depth_m']
    x0, y0, x1, y1 = depth_roi if depth_roi is not None else (0, 0, depth.shape[1], depth.shape[0])
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(depth.shape[1], int(x1)), min(depth.shape[0], int(y1))
    if x0 >= x1 or y0 >= y1:
        raise ValueError('invalid depth projection region')
    patch = depth[y0:y1, x0:x1]
    y, x = np.nonzero(np.isfinite(patch) & (patch > .03) & (patch < 2.))
    x, y = x+x0, y+y0
    xyz = rgbd['rays'][y, x]*depth[y, x, None]
    target = xyz @ transform[:3, :3].T + transform[:3, 3]
    keep = target[:, 2] > .01
    xyz, target, y, x = xyz[keep], target[keep], y[keep], x[keep]
    if not len(xyz):
        return xyz, np.empty((0, 3), np.uint8), np.empty((0, 2)), np.empty((0, 2), int)
    pixels = project_into_view(target, view)
    h, w = view['image'].shape[:2]
    if h*w > 16_000_000:
        raise ValueError('cross_camera_image_exceeds_budget')
    keep = np.isfinite(pixels).all(axis=1) & (pixels >= [0, 0]).all(axis=1) & (pixels < [w-1, h-1]).all(axis=1)
    xyz, target, pixels, y, x = xyz[keep], target[keep], pixels[keep], y[keep], x[keep]
    if not len(xyz):
        return xyz, np.empty((0, 3), np.uint8), np.empty((0, 2)), np.empty((0, 2), int)
    ij = np.rint(pixels).astype(int)
    cell = ij[:, 1]*w+ij[:, 0]
    # The depth footprint is sparse in a full-resolution fixed image. Index
    # only cells actually hit instead of filling a multi-megapixel z-buffer.
    _, inverse = np.unique(cell, return_inverse=True)
    nearest = np.full(int(inverse.max())+1, np.inf)
    np.minimum.at(nearest, inverse, target[:, 2])
    # Reject surfaces behind another projected measurement. Geometry absent
    # from this depth camera can still occlude the RGB view; this is not proof
    # of visibility or material identity from either camera.
    keep = target[:, 2] <= nearest[inverse]+.001
    return (xyz[keep], frame['image'][y[keep], x[keep], ::-1], pixels[keep],
            np.column_stack((x[keep], y[keep])))


def anchor_samples(xyz, pixels, observation, radius_px):
    anchors = np.asarray([row['image_px'] for row in observation['landmarks']], float)
    uv = np.asarray([row['reference_uv'] for row in observation['landmarks']], float)
    distances, indices = cv2.batchDistance(anchors.astype(np.float32), pixels.astype(np.float32),
                                          cv2.CV_32F, normType=cv2.NORM_L2, K=min(4, len(xyz)))
    available = (distances <= radius_px) & (indices >= 0)
    safe = np.clip(indices, 0, len(xyz)-1)
    neighbors = xyz[safe]
    # The nearest real sample may support an anchor; discontinuous neighboring
    # geometry must not be averaged into it. Require distinct depth samples.
    span = np.linalg.norm(neighbors-neighbors[:, :1], axis=2)
    valid = available[:, 0] & (np.max(np.where(available, span, 0), axis=1) <= .006)
    rows = np.flatnonzero(valid)
    _, unique = np.unique(indices[rows, 0], return_index=True)
    rows = rows[unique]
    return uv[rows], xyz[indices[rows, 0]], distances[rows, 0]


def projected_samples(frame, view, transform, cache, depth_roi):
    if cache is not None and 'samples' in cache:
        return cache['samples']
    samples = project_samples(frame, view, transform, depth_roi)
    if cache is not None:
        cache['samples'] = samples
    return samples


def cross_camera_points(frame, view, observation, reference, view_from_depth, *, max_skew_ns,
                        anchor_radius_px=3., limit=30000, projection_cache=None, depth_roi=None):
    """Return original optical XYZ/RGB, with explicit observed timestamp skew.

    The caller binds camera identities and calibration to its session inputs.
    Normalized timestamp agreement is not a claim of verified clock accuracy.
    """
    report = {'schema': 'tatbot.stencil-cross-camera/1', 'candidate_valid': False,
              'motion_authority': False, 'geometry_valid': False, 'frame': 'camera_optical',
              'dense_material_identity_verified': False, 'clock_accuracy_verified': False}
    empty = (np.empty((0, 3)), np.empty((0, 3), np.uint8))
    if not observation['image_tracking_valid']:
        return *empty, dict(report, reason='image_tracking_unavailable')
    if not 0 < max_skew_ns <= 100_000_000 or not 0 < anchor_radius_px <= 8 or not 0 < limit <= 30000:
        raise ValueError('invalid cross-camera display bounds')
    if observation['capture_timestamp_ns'] != view['timestamp_ns']:
        raise ValueError('cross_camera_observation_timestamp_mismatch')
    stamps = [frame['timestamp_ns'], view['timestamp_ns']]
    if any(type(v) is not int or v <= 0 for v in stamps):
        raise ValueError('cross_camera_timestamp_unavailable')
    skew = abs(stamps[0]-stamps[1])
    report.update(observed_skew_ns=skew, max_skew_ns=max_skew_ns,
                  depth_timestamp_ns=stamps[0], tracking_timestamp_ns=stamps[1])
    if skew > max_skew_ns:
        return *empty, dict(report, reason='cross_camera_exposure_skew')
    # Projection depends on the camera pair, not on the printed pattern. The
    # observer can reuse the same measured pixels for each reference in a turn.
    xyz, colors, pixels, depth_pixels = projected_samples(frame, view, view_from_depth,
                                                          projection_cache, depth_roi)
    if not len(xyz):
        return *empty, dict(report, reason='no_projected_depth_support')
    uv, anchors, distance = anchor_samples(xyz, pixels, observation, anchor_radius_px)
    report.update(measured_anchor_count=len(uv), reference_coverage=area(uv))
    if len(uv) < 12 or area(uv) < .1:
        return *empty, dict(report, reason='insufficient_measured_anchor_support')
    model = choose_model(uv, anchors)
    if model is None or model['loo_p95_m'] > .005:
        return *empty, dict(report, reason='cross_camera_surface_fit_inconsistent')
    mask = interior_mask(observation, reference, view['image'].shape[:2])
    ij = np.rint(pixels).astype(int)
    keep = mask[ij[:, 1], ij[:, 0]] != 0
    xyz, colors, pixels, depth_pixels = xyz[keep], colors[keep], pixels[keep], depth_pixels[keep]
    hint_uv = cv2.perspectiveTransform(pixels[None], np.linalg.inv(observation['homography_uv_to_image']))[0] if len(pixels) else np.empty((0, 2))
    residual = np.linalg.norm(xyz-design(hint_uv, model['curved']) @ model['coeff'], axis=1)
    keep = residual <= .005
    xyz, colors, depth_pixels = xyz[keep], colors[keep], depth_pixels[keep]
    stride = max(1, int(np.ceil(len(xyz)/limit)))
    if len(depth_pixels):
        low, high = depth_pixels.min(axis=0), depth_pixels.max(axis=0)+1
        report['depth_pixel_bbox'] = [int(low[0]), int(low[1]), int(high[0]), int(high[1])]
    report.update(candidate_valid=True, reason='supported_cross_camera_interior',
                  anchors=[{'reference_uv': a.tolist(), 'point_camera_m': b.tolist()}
                           for a, b in zip(uv, anchors, strict=True)],
                  model='quadratic_uv_surface' if model['curved'] else 'affine_uv_plane',
                  loo_residual_p95_m=model['loo_p95_m'], anchor_pixel_distance_p95=float(np.percentile(distance, 95)),
                  interior_points=len(xyz[::stride]), excluded_depth_points=int((~keep).sum()),
                  uncertainty='conditional local fit; camera calibration, timing and depth biases remain uncalibrated',
                  visibility='nearest projected depth sample; occluders unseen by the depth camera remain unknown')
    return xyz[::stride], colors[::stride], report
