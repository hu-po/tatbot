"""Advisory RGB-flow/depth cross-check, sharing images with offline registration.

These are correlated sensor measurements, not independent physical ground truth.
Correspondences are measured once per pair and reused across registration methods.
"""
import cv2
import numpy as np


def correspondences(previous, current, roi, max_points=120):
    x, y, w, h = roi
    mask = np.zeros_like(previous['gray'])
    mask[y:y+h, x:x+w] = 255
    points = cv2.goodFeaturesToTrack(previous['gray'], maxCorners=max_points,
                                    qualityLevel=.02, minDistance=8, mask=mask)
    if points is None:
        return {'status': 'unavailable', 'reason': 'no_image_features', 'count': 0}
    target, valid, error = cv2.calcOpticalFlowPyrLK(previous['gray'], current['gray'], points, None,
                                                 winSize=(21, 21), maxLevel=3)
    if target is None:
        return {'status': 'unavailable', 'reason': 'image_flow_failed', 'count': 0}
    back, back_valid, _ = cv2.calcOpticalFlowPyrLK(current['gray'], previous['gray'], target, None,
                                                winSize=(21, 21), maxLevel=3)
    if back is None:
        return {'status': 'unavailable', 'reason': 'reverse_flow_failed', 'count': 0}
    a, b = points.reshape(-1, 2), target.reshape(-1, 2)
    fb = np.linalg.norm(back.reshape(-1, 2) - a, axis=1)
    keep = valid.ravel().astype(bool) & back_valid.ravel().astype(bool)
    keep &= np.isfinite(b).all(axis=1) & (fb <= 1.) & (error.ravel() <= 20.)
    keep &= (b[:, 0] >= x) & (b[:, 0] < x+w-1) & (b[:, 1] >= y) & (b[:, 1] < y+h-1)
    a, b, fb = a[keep], b[keep], fb[keep]
    source_xyz, source_valid = sample_xyz(previous, a)
    target_xyz, target_valid = sample_xyz(current, b)
    keep = source_valid & target_valid
    if int(keep.sum()) < 12:
        return {'status': 'unavailable', 'reason': 'insufficient_image_depth_matches', 'count': int(keep.sum())}
    return {'status': 'available', 'count': int(keep.sum()), 'source_xyz': source_xyz[keep],
            'target_xyz': target_xyz[keep], 'image_motion_p50_px': float(np.median(np.linalg.norm(b[keep]-a[keep], axis=1))),
            'forward_backward_p50_px': float(np.median(fb[keep]))}


def sample_xyz(frame, pixels):
    # Nearest depth pixel: the residual includes pixel quantization and depth noise.
    ij = np.rint(pixels).astype(int)
    depth = frame['depth_m']
    if not len(ij):
        return np.empty((0, 3)), np.empty(0, dtype=bool)
    valid = (ij[:, 0] >= 0) & (ij[:, 0] < depth.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < depth.shape[0])
    ij[:, 0] = np.clip(ij[:, 0], 0, depth.shape[1]-1)
    ij[:, 1] = np.clip(ij[:, 1], 0, depth.shape[0]-1)
    z = depth[ij[:, 1], ij[:, 0]]
    near, far = frame['depth_range']
    valid &= np.isfinite(z) & (z > near) & (z < far)
    rays = frame['rays'][ij[:, 1], ij[:, 0]]
    return rays * z[:, None], valid


def consistency(matches, matrix):
    public = {k: v for k, v in matches.items() if k not in ('source_xyz', 'target_xyz')}
    if matches['status'] != 'available':
        return public
    matrix = np.asarray(matrix)
    predicted = matches['source_xyz'] @ matrix[:3, :3].T + matrix[:3, 3]
    residual = np.linalg.norm(predicted - matches['target_xyz'], axis=1) * 1000
    public['residual_mm'] = {k: float(np.percentile(residual, q)) for k, q in [('p50', 50), ('p95', 95), ('max', 100)]}
    return public
