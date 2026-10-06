"""Stencil page and clear-centre boundaries, measured anchors and centre axes in
the frame the caller names, derived from the same local UV surface fit that
selected a stencil's interior points.

The fit is supported inside the observed anchor hull. The features live on
the printed border, so the page boundary lies at most the white margin past
that hull and the clear-centre boundary lies inside it; both are a small
extrapolation of the fit, drawn as display geometry only.
"""

import cv2
import numpy as np

EDGE_SAMPLES = 8       # points per rectangle edge, so a curved fit bends the loop
MAX_ANCHORS = 256      # displayed anchors; the fit itself keeps every measured one
AXIS_LENGTH_M = .025


def uv_loop(bounds, samples=EDGE_SAMPLES):
    """Closed rectangle in page UV, sampled along each edge, first point not repeated."""
    u0, v0, u1, v1 = (float(value) for value in bounds)
    if not (0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1) or not 2 <= samples <= 64:
        raise ValueError('invalid stencil UV rectangle')
    t = np.linspace(0, 1, samples, endpoint=False)
    edges = [np.column_stack([u0+(u1-u0)*t, np.full_like(t, v0)]),
             np.column_stack([np.full_like(t, u1), v0+(v1-v0)*t]),
             np.column_stack([u1-(u1-u0)*t, np.full_like(t, v1)]),
             np.column_stack([np.full_like(t, u0), v1-(v1-v0)*t])]
    return np.concatenate(edges)


def _anchors(uv, xyz):
    uv, xyz = np.asarray(uv, float), np.asarray(xyz, float)
    if uv.ndim != 2 or uv.shape[1] != 2 or xyz.shape != (len(uv), 3) or len(uv) < 12:
        raise ValueError('stencil outline needs at least twelve measured anchors')
    if not np.isfinite(uv).all() or not np.isfinite(xyz).all() or np.any((uv < 0) | (uv > 1)):
        raise ValueError('invalid stencil anchor geometry')
    return uv, xyz


def _fit(uv, xyz, coeff, curved):
    """The retained fit, or the ordinary model choice over the anchors."""
    from stencil_surface_fit import choose_model
    if coeff is None:
        model = choose_model(uv, xyz)
        if model is None:
            raise ValueError('stencil anchors do not support a surface fit')
        coeff, curved = model['coeff'], model['curved']
    coeff = np.asarray(coeff, float)
    if coeff.shape != (6 if curved else 3, 3) or not np.isfinite(coeff).all():
        raise ValueError('invalid stencil surface coefficients')
    return coeff, bool(curved)


def _center(uv, coeff, center):
    """A retained fit-frame centre pose, else one fitted only when the anchors
    enclose UV (0.5, 0.5); None leaves the axes undrawn."""
    from stencil_surface_fit import center_pose
    if center is None:
        hull = cv2.convexHull(uv.astype(np.float32))
        if cv2.pointPolygonTest(hull, (.5, .5), False) < 0:
            return None
        center = center_pose(coeff)
    if center is None:
        return None
    pose = np.asarray(center, float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('invalid stencil centre pose')
    return pose


def stencil_outline(uv, xyz, transform, reference=None, *, coeff=None, curved=None, center=None):
    """Display geometry for one tracked stencil.

    `uv`/`xyz` are the measured anchors in the fit's own frame and `transform`
    carries that frame into the display frame. `coeff`/`curved` reuse the
    retained fit; without them the anchors are fitted again with the ordinary
    model choice. `center` is a retained fit-frame centre pose; without it the
    centre axes are drawn only when the anchors enclose UV (0.5, 0.5).
    """
    from stencil_surface_fit import design
    uv, xyz = _anchors(uv, xyz)
    coeff, curved = _fit(uv, xyz, coeff, curved)
    matrix = np.asarray(transform, float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('invalid stencil display transform')

    def placed(points):
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    result = {'outline': placed(design(uv_loop((0, 0, 1, 1)), curved) @ coeff).tolist()}
    if reference is not None and 'clear_center_uv' in reference:
        result['clear_center'] = placed(design(uv_loop(reference['clear_center_uv']), curved) @ coeff).tolist()
    selected = np.linspace(0, len(xyz)-1, min(len(xyz), MAX_ANCHORS)).astype(int)
    result['anchors'] = placed(xyz[selected]).tolist()
    pose = _center(uv, coeff, center)
    if pose is not None:
        pose = matrix @ pose
        result['center'] = [pose[:3, 3].tolist(), *(pose[:3, :3].T*AXIS_LENGTH_M).tolist()]
    return result


def report_outline(report, reference, root_from_camera):
    """Outline from a camera report (`tatbot.stencil-surface/1` or the
    cross-camera shape); None when the report supports no fit."""
    if not report.get('candidate_valid') or report.get('calibration_warnings'):
        return None
    anchors = report.get('anchors') or []
    if len(anchors) < 12:
        return None
    curved = report.get('model') == 'quadratic_uv_surface'
    coeff = report.get('coefficients_camera_m')
    return stencil_outline([a['reference_uv'] for a in anchors], [a['point_camera_m'] for a in anchors],
                           root_from_camera, reference, coeff=coeff, curved=curved if coeff is not None else None,
                           center=report.get('camera_from_stencil_center'))
