"""Shared optical rays and aligned-depth geometry for scans and stencil tracking.

RealSense alignment changes the pixel grid but retains native depth Z. A color
ray therefore needs the captured depth-to-color transform before it has metric
color-frame coordinates. This module neither estimates nor adopts calibration.
"""

import json
from functools import lru_cache

import numpy as np


def document(value):
    return json.loads(value) if isinstance(value, str) else value


# librealsense's `rs2_deproject_pixel_to_point` (rsutil.h, SDK 2.58.4) undistorts
# a normalized pixel under either Brown-Conrady model with the same ten
# fixed-point iterations; neither model is evaluated in closed form. The two
# differ only in where the tangential term is taken: at the current estimate
# (`BrownConrady`) or at the estimate with the radial scale removed
# (`BrownConradyInverse`). Ported so that deprojection needs no SDK wheel; the
# checked-in `tests/fixtures/deprojection-sdk-table.json` pins the SDK's own
# answers for both models.
DEPROJECTION_MODELS = ('None', 'BrownConrady', 'BrownConradyInverse')
DEPROJECTION_ITERATIONS = 10


def validate_intrinsics(intr):
    height, width = intr['height'], intr['width']
    coeff = intr['distortion_coefficients']
    values = [intr[key] for key in ('fx', 'fy', 'ppx', 'ppy')]
    if (intr.get('schema') != 'tatbot.camera-intrinsics/1'
            or type(height) is not int or type(width) is not int
            or not 1 <= height <= 1080 or not 1 <= width <= 1920 or len(coeff) != 5
            or not np.isfinite([*values, *coeff]).all() or min(values[:2]) <= 0):
        raise ValueError('invalid or oversized active intrinsics')
    model = intr['distortion_model']
    if model not in DEPROJECTION_MODELS or (model == 'None' and np.any(coeff)):
        raise ValueError('unsupported deprojection distortion model')
    return model


def deproject_pixels(intr, pixels):
    """Unit-depth rays for `pixels` (..., 2) under the SDK's deprojection convention."""
    model = validate_intrinsics(intr)
    pixels = np.asarray(pixels, float)
    xo = (pixels[..., 0]-intr['ppx'])/intr['fx']
    yo = (pixels[..., 1]-intr['ppy'])/intr['fy']
    x, y = xo, yo
    k1, k2, p1, p2, k3 = intr['distortion_coefficients']
    if np.any(intr['distortion_coefficients']):
        for _ in range(DEPROJECTION_ITERATIONS):
            r2 = x*x + y*y
            icdist = 1/(1 + ((k3*r2 + k2)*r2 + k1)*r2)
            xq, yq = (x/icdist, y/icdist) if model == 'BrownConradyInverse' else (x, y)
            delta_x = 2*p1*xq*yq + p2*(r2 + 2*xq*xq)
            delta_y = 2*p2*xq*yq + p1*(r2 + 2*yq*yq)
            x, y = (xo-delta_x)*icdist, (yo-delta_y)*icdist
    rays = np.stack((x, y, np.ones_like(x)), axis=-1)
    if not np.isfinite(rays).all():
        raise ValueError('nonfinite deprojection rays')
    return rays


@lru_cache(maxsize=4)
def camera_rays(intrinsics_json):
    intr = document(intrinsics_json)
    validate_intrinsics(intr)
    y, x = np.mgrid[:intr['height'], :intr['width']]
    rays = deproject_pixels(intr, np.stack((x, y), axis=-1))
    rays.setflags(write=False)
    return rays


def alignment_transform(value, intrinsics):
    model = document(value)
    if (not isinstance(model, dict) or model.get('schema') != 'tatbot.realsense-native-alignment/1'
            or model.get('transform') != 'color_from_native_depth'
            or model.get('rotation_layout') != 'column_major'
            or model.get('translation_units') != 'metres'
            or model.get('aligned_depth_value_axis') != 'native_depth_z'
            or model.get('color') != intrinsics):
        raise ValueError('aligned depth has no matching native alignment model')
    rotation = np.asarray(model['rotation'], float).reshape(3, 3, order='F')
    translation = np.asarray(model['translation'], float)
    if (translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5)):
        raise ValueError('native alignment is not a finite rigid transform')
    return rotation, translation


def paired_alignment(color_attributes, depth_attributes, intrinsics):
    color = document(color_attributes.get('alignment_calibration'))
    depth = document(depth_attributes.get('alignment_calibration'))
    if color != depth:
        raise ValueError('paired native alignment models differ')
    alignment_transform(depth, intrinsics)
    return depth


def color_depth_m(depth_m, rays, model, intrinsics):
    """Color-frame Z for each aligned pixel, preserving absent measurements.

    Let Pc = s*r and Pc = R*Pd+t. The retained Z16 measures Pd.z, so
    s = (Pd.z + R[:,2] dot t) / (R[:,2] dot r). Multiplying by Pd.z
    directly is correct only when the two optical Z axes coincide.
    """
    rotation, translation = alignment_transform(model, intrinsics)
    depth = np.asarray(depth_m, float)
    rays = np.asarray(rays)
    if rays.shape != (*depth.shape, 3) or not np.isfinite(rays).all():
        raise ValueError('aligned depth and color ray shapes differ')
    axis = rotation[:, 2]
    denominator = rays @ axis
    if np.any(denominator <= 1e-6):
        raise ValueError('native depth axis does not intersect the color rays')
    result = (depth + axis @ translation)/denominator
    return np.where(np.isfinite(depth) & (depth > 0) & np.isfinite(result) & (result > 0), result, 0.)


# The installed camera updates its intrinsics as it warms. Report the ray
# change for diagnosis; a small thermal change is not a reason to hold a run.


def optics_drift(intrinsics, saved):
    """Largest ray-direction change from bound to live optics, in radians:
    the focal scale for fx/fy and the principal-point offset over the focal length."""
    focal = np.array([saved['fx'], saved['fy']], float)
    live = np.array([intrinsics[k] for k in ('fx', 'fy', 'ppx', 'ppy')], float)
    bound = np.array([saved[k] for k in ('fx', 'fy', 'cx', 'cy')], float)
    if not np.isfinite([*live, *bound]).all() or min(focal) <= 0:
        return np.inf
    return float(np.max(np.abs(live-bound)/np.tile(focal, 2)))


def require_calibrated_model(intrinsics, camera, device_serial=None):
    """Check the camera model identity; thermal focal drift is diagnostic."""
    saved, distortion = camera['intrinsics'], camera['distortion']
    coeff = distortion['coefficients'] or [0.]*5
    live_coeff = intrinsics['distortion_coefficients']
    if not np.isfinite([intrinsics[k] for k in ('fx', 'fy', 'ppx', 'ppy')]).all():
        raise ValueError('active RGB-D optics differ from bound calibration: invalid intrinsics')
    if any(intrinsics[k] != saved[k] for k in ('width', 'height')):
        raise ValueError('active RGB-D optics differ from bound calibration: image dimensions')
    if len(coeff) != 5 or len(live_coeff) != 5 or not np.allclose(live_coeff, coeff, rtol=0, atol=1e-6):
        raise ValueError('active RGB-D optics differ from bound calibration: distortion coefficients')
    models = {'none': 'None', 'brown_conrady': 'BrownConrady'}
    if distortion['model'] not in models or (np.any(coeff) and intrinsics['distortion_model'] != models[distortion['model']]):
        raise ValueError('active RGB-D distortion model differs')
    bound = camera.get('metadata', {}).get('device_serial')
    if bound and device_serial != bound:
        raise ValueError('active RGB-D device differs from bound calibration')


def deproject_aligned(depth, units_m, intrinsics, model, depth_range):
    raw = np.asarray(depth)
    rays = camera_rays(json.dumps(intrinsics, sort_keys=True))
    if (raw.dtype != np.uint16 or raw.shape != rays.shape[:2]
            or not np.isfinite(units_m) or units_m <= 0):
        raise ValueError('invalid aligned depth encoding, dimensions or units')
    measured = raw.astype(float)*units_m
    measured[raw == 65535] = 0
    z = color_depth_m(measured, rays, model, intrinsics)
    valid = (raw > 0) & (raw < 65535) & (measured >= depth_range[0]) & (measured <= depth_range[1]) & (z > 0)
    return rays[valid]*z[valid, None]


def verify_pair(color, depth):
    """Mirror the owner's sequence/epoch and source-time exposure contract.

    Device stream counters can have different origins; equal counter values
    neither establish a pair nor replace the source timestamp comparison.
    """
    ca, da = color.get('attributes', {}), depth.get('attributes', {})
    if any(not da.get(key) or da[key] != ca.get(key) for key in ('capture_epoch', 'device_serial')):
        raise ValueError('rgbd_capture_pair_unverified')
    sequence = color.get('sequence')
    if type(sequence) is not int or sequence < 0 or type(depth.get('sequence')) is not int or depth['sequence'] != sequence:
        raise ValueError('rgbd_capture_pair_unverified')
    ct, dt = color['timestamps'], depth['timestamps']
    if not ct.get('source_domain') or ct['source_domain'] != dt.get('source_domain'):
        raise ValueError('rgbd_source_clock_mismatch')
    stamps = [ct.get('source_ns'), dt.get('source_ns')]
    if any(type(stamp) is not int or stamp < 0 for stamp in stamps):
        raise ValueError('rgbd_source_timestamps_unavailable')
    fps = []
    for metadata in (color, depth):
        n, d = metadata['profile'].get('fps_num'), metadata['profile'].get('fps_den')
        if type(n) is not int or type(d) is not int or n <= 0 or d <= 0:
            raise ValueError('rgbd_frame_rate_unavailable')
        fps.append(n/d)
    if abs(stamps[0]-stamps[1]) >= int(1_000_000_000/(2*max(fps))):
        raise ValueError('rgbd_adjacent_exposures')


# Native wrist page-plane fitting, also used by the ROS owner and capture replay.
def depth_points(depth_m: np.ndarray, k: np.ndarray, *, stride: int = 4, near_m: float = 0.07,
                 far_m: float = 0.5, intrinsics: dict | None = None) -> np.ndarray:
    """Sampled in-range pixels as native depth optical points (N, 3)."""
    d = depth_m[::stride, ::stride]
    v, u = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    ok = (d > near_m) & (d < far_m)
    z = d[ok]
    if intrinsics is not None:
        models = {'distortion.none': 'None', 'distortion.brown_conrady': 'BrownConrady',
                  'distortion.inverse_brown_conrady': 'BrownConradyInverse'}
        intr = {'schema': 'tatbot.camera-intrinsics/1', 'width': depth_m.shape[1], 'height': depth_m.shape[0],
                **{key: intrinsics[key] for key in ('fx', 'fy', 'ppx', 'ppy')},
                'distortion_coefficients': intrinsics['coeffs'], 'distortion_model': models.get(intrinsics['model'])}
        return deproject_pixels(intr, np.c_[u[ok], v[ok]]) * z[:, None]
    return np.c_[(u[ok] - k[0, 2]) * z / k[0, 0], (v[ok] - k[1, 2]) * z / k[1, 1], z]


def page_height_from_depth(points_base: np.ndarray, base_from_page: np.ndarray, *, half_m=(0.050, 0.075),
                           band_m: float = 0.030, inlier_m: float = 0.001, iterations: int = 200,
                           seed: int = 0, diagnostics: dict | None = None) -> dict | None:
    """Fit the in-band page plane by RANSAC and least squares; optionally record rejection evidence.
    Return centre-normal offset, base-frame normal, RMS and inliers, or None. The band is wide because the
    page it is centred on is a camera's guess: on 2026-10-02 the wrist depth plane sat +18.6 mm over the
    overhead page (0.5 mm RMS, two views within 0.5 mm) and a 15 mm band threw it away.
    """
    page = np.asarray(base_from_page, dtype=float)
    local = (np.linalg.inv(page) @ np.c_[points_base, np.ones(len(points_base))].T)[:3].T
    xy = np.isfinite(local).all(axis=1) & (np.abs(local[:, 0]) < half_m[0]) & (np.abs(local[:, 1]) < half_m[1])
    sel = xy & (np.abs(local[:, 2]) < band_m)
    observation = {} if diagnostics is None else diagnostics
    observation.update(points=int(len(local)), page_xy_points=int(xy.sum()), prior_band_points=int(sel.sum()),
                       prior_band_m=band_m, minimum_points=200, minimum_inliers=100, best_inliers=0,
                       page_xy_height_quantiles_m=(np.percentile(local[xy, 2], [5, 50, 95]).tolist()
                                                   if xy.any() else None))
    pts = local[sel]
    if len(pts) < 200:
        observation["reason"] = "insufficient_page_xy_depth" if xy.sum() < 200 else "depth_outside_prior_band"
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < 0.9:   # within ~25 deg of the prior normal
            continue
        inl = np.abs((pts - a) @ n) < inlier_m
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 100:
        observation.update(reason="no_supported_plane", best_inliers=0 if best is None else int(best.sum()))
        return None
    p = pts[best]
    c = p.mean(axis=0)
    n = np.linalg.svd(p - c)[2][-1]
    n = n if n[2] > 0 else -n
    offset = float(c @ n / n[2])   # z of the plane at the page centre (x = y = 0), page frame
    res = (p - c) @ n
    observation.update(reason="accepted", best_inliers=int(best.sum()))
    return {"offset_m": offset, "normal": (page[:3, :3] @ n).tolist(), "rms_m": float(np.sqrt(np.mean(res ** 2))),
            "inliers": int(best.sum()), "support_xy_m": np.percentile(p[:, :2], [5, 95], axis=0).tolist()}


def wrist_clearance(fit, base_from_page, base_from_depth, base_from_tcp, *, frame_age_s, joint_age_s,
                    systematic_sigma_m=0.008):
    """One-capture tip-to-plane comparison. Common rigid upstream FK error cancels in relative distance.

    This does not cancel camera/TCP mount error, tool flex, carriage state or cartridge phase. The
    unmeasured mount floor is kept separate from plane residual; no inverse-variance posterior is claimed.
    The fit comes from a prior-bounded ROI, so even relative clearance still depends on ROI identity.
    """
    out = {'used_for_drawing': False, 'adoption_supported': False, 'reason': 'camera_to_tool_unqualified',
           'systematic_sigma_m': float(systematic_sigma_m), 'uncertainty_scope':
           'nominal camera-to-tool floor; not calibrated tip accuracy; shared arm FK is not independent'}
    if not np.isfinite(frame_age_s) or not 0 <= frame_age_s <= 0.5:
        return {**out, 'reason': 'stale_depth'}
    if not np.isfinite(joint_age_s) or not 0 <= joint_age_s <= 0.5:
        return {**out, 'reason': 'stale_joints'}
    if fit is None:
        return {**out, 'reason': 'rejected_page_plane'}
    page, depth, tcp = (np.asarray(t, float) for t in (base_from_page, base_from_depth, base_from_tcp))
    n = np.asarray(fit['normal'], float)
    n /= np.linalg.norm(n)
    plane_base = np.r_[n, -n @ (page[:3, 3] + page[:3, 2]*fit['offset_m'])]
    plane_depth = depth.T @ plane_base
    tip_depth = (np.linalg.inv(depth) @ tcp)[:, 3]
    point_page = (np.linalg.inv(page) @ tcp)[:3, 3]
    normal_page = page[:3, :3].T @ n
    local_height = fit['offset_m'] - normal_page[:2] @ point_page[:2] / normal_page[2]
    support = np.asarray(fit['support_xy_m'])
    supported = bool(np.all(point_page[:2] >= support[0]) and np.all(point_page[:2] <= support[1]))
    out.update(relative_clearance_m=float(plane_depth @ tip_depth), fk_clearance_m=float(point_page[2]),
               local_paper_height_m=float(local_height), tip_xy_m=point_page[:2].tolist(),
               local_supported=supported, plane_depth=plane_depth.tolist(), camera_from_tcp=(np.linalg.inv(depth) @ tcp).tolist(),
               plane_rms_m=fit['rms_m'], normal_page=normal_page.tolist())
    if not supported:
        out['reason'] = 'tip_outside_plane_support'
    return out


def wrist_frame_age(timestamps, evaluated_monotonic_s):
    """Expose host receipt and sensor exposure age; an unmapped hardware clock is unavailable evidence."""
    if not timestamps:
        return None, 'missing_depth_timestamps'
    ages = []
    for stamp in timestamps:
        if not all(key in stamp for key in ('depth_domain', 'received_at_ns', 'depth_ms', 'received_monotonic_s')):
            return None, 'missing_depth_timestamps'
        if stamp.get('depth_domain') not in ('timestamp_domain.global_time', 'timestamp_domain.system_time'):
            return None, 'unmapped_depth_clock'
        lag = stamp['received_at_ns']*1e-9 - stamp['depth_ms']*.001
        elapsed = evaluated_monotonic_s-stamp['received_monotonic_s']
        if not np.isfinite([lag, elapsed]).all() or lag < -.05 or elapsed < 0:
            return None, 'invalid_depth_timestamp'
        ages.append(max(lag, 0.)+elapsed)
    return max(ages), None
