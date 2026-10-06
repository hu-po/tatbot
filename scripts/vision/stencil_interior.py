"""Observed RGB-D samples in a tracked stencil's clear center, without filling holes.

The image warp selects pixels; it does not flatten their measured XYZ. Local
depth support rejects foreground occluders, but cannot certify blank material.
"""

import cv2
import numpy as np
from stencil_surface import estimate_surface, rgbd_frame
from stencil_surface_fit import design


def interior_mask(observation, reference, shape, margin_px=3):
    """Reference clear center clipped to observed feature support in an image."""
    height, width = shape
    u0, v0, u1, v1 = reference['clear_center_uv']
    if not (0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1):
        raise ValueError('invalid stencil clear center')
    corners = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], float)
    homography = np.asarray(observation['homography_uv_to_image'])
    polygon = cv2.perspectiveTransform(corners[None], homography)[0]
    if not np.isfinite(polygon).all() or np.max(np.abs(polygon)) > 8*max(height, width):
        raise ValueError('invalid stencil interior projection')
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    support = np.zeros_like(mask)
    anchors = np.array([row['image_px'] for row in observation['landmarks']], np.float32)
    cv2.fillConvexPoly(support, cv2.convexHull(anchors).astype(np.int32), 1)
    # Keep the raster boundary away from uncertain feature correspondences.
    return cv2.erode(mask & support, np.ones((2*margin_px+1, 2*margin_px+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)


def interior_points(frame, observation, reference, limit=30000):
    report, _ = estimate_surface(observation, frame, reference)
    empty = (np.empty((0, 3)), np.empty((0, 3), np.uint8))
    if not report['candidate_valid'] or report.get('calibration_warnings'):
        return *empty, report
    rgbd, _ = rgbd_frame(frame)
    depth = rgbd['depth_m']
    homography = np.asarray(observation['homography_uv_to_image'])
    mask = interior_mask(observation, reference, depth.shape)
    valid = (mask != 0) & np.isfinite(depth) & (depth > .03) & (depth < 2.)
    y, x = np.nonzero(valid)
    pixels = np.column_stack((x, y)).astype(float)
    uv = cv2.perspectiveTransform(pixels[None], np.linalg.inv(homography))[0] if len(pixels) else np.empty((0, 2))
    xyz = rgbd['rays'][y, x] * depth[y, x, None]
    fitted = design(uv, report['model'] == 'quadratic_uv_surface') @ np.asarray(report['coefficients_camera_m'])
    # This is a display support test, not a new motion gate or accuracy claim.
    keep = np.linalg.norm(xyz-fitted, axis=1) <= .005
    xyz, pixels = xyz[keep], pixels[keep].astype(int)
    colors = frame['image'][pixels[:, 1], pixels[:, 0], ::-1]
    stride = max(1, int(np.ceil(len(xyz)/limit)))
    report.update(interior_points=len(xyz[::stride]), excluded_depth_points=int((~keep).sum()),
                  dense_material_identity_verified=False, interior_warp='reference homography within observed anchor hull')
    return xyz[::stride], colors[::stride], report
