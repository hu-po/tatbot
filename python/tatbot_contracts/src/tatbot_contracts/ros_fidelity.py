"""Slot-isolated, two-sided ink fidelity after independent printed-border registration."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SCHEMA = 'tatbot.ink-fidelity/1'
# Measured on the wrist views of the first physical DBV3 pair (2026-09-28, 10 px/mm page images):
PAPER_CLOSE_M = .005    # local paper estimate: wider than the 0.5 mm ballpoint line, dense ECS hatching and a
                        # 3.7 mm smear; metrics on the real views held within 0.01 for 3-6 mm
INK_RATIO = .85         # ink is darker than this share of the local paper: paper 2 mm clear of every line
                        # read >= 0.90 at its 1st percentile, ink centrelines 0.54-0.79 at their median
SHADE_RATIO = .65       # a local paper estimate darker than this share of the slot's white is shadow or a
                        # smear wider than PAPER_CLOSE_M; the two cannot be told apart, so the view is refused
TOLERANCE_M = .0005     # ink within this of the plan is on it: one native wrist pixel at the aim distance
                        # (0.18 m / fx 651 = 0.28 mm) plus the per-view border registration sigma (<= 0.24 mm)
ALIGN_RADIUS_M = .008   # the right arm's ink sat -7.3..+0.6 mm from the plan on the print (2026-09-27/28);
                        # a 2 mm search found a false peak inside it and reported 61% of a complete drawing missing


def identity():
    from tatbot_session import inspect

    return {'schema': SCHEMA, 'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(inspect.__file__))}}


def _registered(image, border, centre, extent, ppm):
    import cv2
    from tatbot_session.inspect import page_grid, to_image_px

    scale = np.asarray(border['scale'], float)
    angle = border['theta_rad']
    transform = np.array([[scale[0], -angle], [angle, scale[1]]])
    shift = np.array([border['tx_m'], border['ty_m']]) + np.asarray(centre)*(1-scale)
    gx, gy = page_grid(extent, ppm)
    raw = (np.c_[gx.ravel(), gy.ravel()]-shift) @ np.linalg.inv(transform).T
    uv = to_image_px(raw, extent, ppm) - .5
    return cv2.remap(image, uv[:, 0].reshape(gx.shape).astype(np.float32),
                     uv[:, 1].reshape(gy.shape).astype(np.float32), cv2.INTER_LINEAR, borderValue=0)


def _observed(image, roi, ppm):
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Rectification's black outside-frame region is connected to an image edge.
    _, labels = cv2.connectedComponents((gray <= 8).astype(np.uint8), connectivity=8)
    boundary = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    invalid = np.isin(labels, boundary[boundary != 0])
    valid = roi & ~invalid
    if not roi.any():
        return None, 'slot is outside the observed image'
    if valid.sum() < .95*roi.sum():
        return None, 'slot not fully observed'
    # Ink is judged against the paper around it, so the arm's shadow across a slot is not ink.
    size = int(round(PAPER_CLOSE_M*1000*ppm)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    paper = cv2.GaussianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel).astype(np.float32), (0, 0), size/4)
    white = float(np.percentile(paper[valid], 95))
    if white < 64:
        return None, 'paper brightness unresolved'
    if float(paper[valid].min()) < SHADE_RATIO*white:
        return None, 'shadow or a wide dark smear in the slot: ink cannot be separated from shading'
    return valid & (gray < INK_RATIO*paper), None


def _expected(program, shape, extent, ppm):
    import cv2
    from tatbot_session.inspect import to_image_px

    center = np.zeros(shape, np.uint8)
    mask, count = np.zeros(shape, np.uint8), np.zeros(shape, np.uint32)
    current = None
    resources = {resource['id']: resource for resource in program['resources']}
    for op in program['ops']:
        if op['op'] != 'stroke':
            continue
        width = resources[op['resource_id']]['tool']['line_width_m']
        source = op['src']
        key = (op['resource_id'], *(source.get(k) for k in ('placement', 'artwork_sha256', 'program_sha256', 'layer', 'path', 'pen')))
        if key != current:
            count += mask
            mask.fill(0)
            current = key
        points = np.rint(to_image_px(op['points_m'], extent, ppm)-.5).astype(np.int32)
        thick = max(1, round((width or op['generation_width_m'])*1000*ppm))
        cv2.polylines(mask, [points], op['closed'], 1, thick, lineType=cv2.LINE_8)
        cv2.polylines(center, [points], op['closed'], 1, 1, lineType=cv2.LINE_8)
    count += mask
    return count > 0, center.astype(bool), float((count > 1).sum()/max(1, (count > 0).sum()))


def _align(expected, ink, radius, tolerance):
    """The shift of the plan onto the slot's ink by normalised correlation, so a wide dark region does not
    outscore the drawing's shape. Ambiguous when near-equal peaks lie farther apart than the metrics'
    tolerance (a straight line along itself); a spread within it cannot change them."""
    import cv2

    if not ink.any():
        return 0, 0, True
    padded = np.pad(ink.astype(np.float32), radius)
    response = np.nan_to_num(cv2.matchTemplate(padded, expected.astype(np.float32), cv2.TM_CCORR_NORMED))
    peak = response.max()
    ys, xs = np.where(response >= peak - 1e-3)
    best = np.argmin((xs-radius)**2+(ys-radius)**2)
    dx, dy = int(xs[best]-radius), int(ys[best]-radius)
    ambiguous = float(np.hypot(xs.max()-xs.min(), ys.max()-ys.min())) > max(2., tolerance)
    return dx, dy, ambiguous


def _metrics(expected, center, ink, ppm, overlap, *, alignment=None):
    import cv2

    radius = max(1, round(ALIGN_RADIUS_M*1000*ppm))  # only this slot's ink enters the search
    tolerance = TOLERANCE_M*1000*ppm
    dx, dy, ambiguous = _align(expected, ink, radius, tolerance) if alignment is None else alignment
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    size = (ink.shape[1], ink.shape[0])
    aligned = cv2.warpAffine(expected.astype(np.uint8), transform, size).astype(bool)
    samples = cv2.warpAffine(center.astype(np.uint8), transform, size).astype(bool)
    to_ink = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5)
    to_plan = cv2.distanceTransform((~aligned).astype(np.uint8), cv2.DIST_L2, 5)
    union, intersection = (aligned | ink).sum(), (aligned & ink).sum()
    spill = ink & (to_plan > tolerance)
    widths = 2*cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)[samples & ink]/ppm
    # Planned pixels the shift carries out of the slot count as missing: ink outside a slot is not scored.
    missing = max(0, int(expected.sum()) - int((aligned & (to_ink <= tolerance)).sum())) / max(1, expected.sum())
    found = bool(ink.any() and not ambiguous and max(abs(dx), abs(dy)) < radius and intersection/max(1, union) >= .2)
    return {'alignment_found': found, 'alignment_ambiguous': ambiguous,
            'placement_error_m': [dx/(1000*ppm), -dy/(1000*ppm)],
            'missing_fraction': float(missing), 'spill_fraction': float(spill.sum()/max(1, ink.sum())),
            'spill_area_mm2': float(spill.sum()/ppm**2), 'ink_area_mm2': float(ink.sum()/ppm**2),
            'iou': float(intersection/max(1, union)), 'planned_overlap_fraction': overlap,
            'observed_width_proxy_mm': float(np.median(widths)) if len(widths) else None,
            'width_proxy_resolution_mm': 1/ppm, 'width_resolved': bool(len(widths) and np.median(widths)*ppm >= 3),
            'tolerance_mm': TOLERANCE_M*1000}


def _resources(program, shape, extent, ppm, bounds, ink, aggregate):
    """One aggregate alignment; equidistant/overlapping regions cannot identify a resource.

    This partitions geometry by proximity, never by pigment. Separate alignment
    for each resource could hide registration errors between cartridges.
    """
    import cv2

    dx, dy = aggregate['placement_error_m']
    shift = (round(dx*1000*ppm), round(-dy*1000*ppm), aggregate['alignment_ambiguous'])
    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    size = (ink.shape[1], ink.shape[0])
    masks, centers, distances, overlaps, identifiers = [], [], [], [], []
    for resource in program['resources']:
        ops = [op for op in program['ops'] if op.get('resource_id') == resource['id']]
        expected, center, overlap = _expected({**program, 'ops': ops}, shape, extent, ppm)
        if not expected.any():
            continue
        mask = expected[bounds]
        aligned = cv2.warpAffine(mask.astype(np.uint8), matrix, size).astype(bool)
        masks.append(mask)
        centers.append(center[bounds])
        distances.append(cv2.distanceTransform((~aligned).astype(np.uint8), cv2.DIST_L2, 5))
        overlaps.append(overlap)
        identifiers.append(resource['id'])
    if not distances:
        return {}
    distances = np.asarray(distances)
    nearest = distances.min(axis=0)
    unique = (np.abs(distances-nearest) <= .01).sum(axis=0) == 1
    output = {}
    for index, resource_id in enumerate(identifiers):
        region = unique & (np.abs(distances[index]-nearest) <= .01)
        aligned = cv2.warpAffine(masks[index].astype(np.uint8), matrix, size).astype(bool)
        coverage = float((aligned & region).sum()/max(1, aligned.sum()))
        output[resource_id] = {'attribution': 'geometric proximity; pigment unresolved',
            'unique_footprint_fraction': coverage, 'color_accuracy': None,
            'metrics': _metrics(masks[index], centers[index], ink & region, ppm, overlaps[index], alignment=shift) if coverage == 1 else None,
            'reason': None if coverage == 1 else 'overlapping footprints cannot be assigned without pigment evidence'}
    return output


def measure(image, program, border, centre, extent, ppm):
    from tatbot_session.inspect import page_grid

    if border is None or 'scale' not in border:
        return {'valid': False, 'reason': 'independent printed-border registration unavailable'}
    registration = [*border['scale'], border['theta_rad'], border['tx_m'], border['ty_m'], *border['sigma'][:2]]
    if not np.all(np.isfinite(registration)) or min(border['scale']) <= 0:
        return {'valid': False, 'reason': 'printed-border registration is degenerate'}
    image = _registered(image, border, centre, extent, ppm)
    at = np.asarray(program['design']['at_m']) if program.get('research') else np.zeros(2)
    half = np.asarray(program['research']['slot_m'] if program.get('research') else program['page']['clear_m'])/2
    gx, gy = page_grid(extent, ppm)
    roi = (np.abs(gx-at[0]) <= half[0]) & (np.abs(gy-at[1]) <= half[1])
    ink, reason = _observed(image, roi, ppm)
    if reason:
        return {'valid': False, 'reason': reason}
    expected, center, overlap = _expected(program, roi.shape, extent, ppm)
    ys, xs = np.where(roi)
    bounds = np.s_[ys.min():ys.max()+1, xs.min():xs.max()+1]
    result = _metrics(expected[bounds], center[bounds], ink[bounds], ppm, overlap)
    return {'valid': result['alignment_found'], 'pixel_size_mm': 1/ppm,
            'registration_sigma_mm': [float(v)*1000 for v in border['sigma'][:2]],
            'physical_width_status': {row['id']: row['tool']['line_width_status'] for row in program['resources']},
            'existing_ink': 'recorded page/slot ownership; no before-image subtraction',
            'per_resource': _resources(program, roi.shape, extent, ppm, bounds, ink[bounds], result), **result}


def analyse(views, program, borders, centre, extent, ppm):
    records = [measure(view, program, border, centre, extent, ppm) for view, border in zip(views, borders, strict=True)]
    good = [v for v in records if v['valid']]
    metrics = ('missing_fraction', 'spill_fraction', 'spill_area_mm2', 'iou', 'planned_overlap_fraction')
    per_resource = {}
    for resource in program['resources']:
        values = [v['per_resource'].get(resource['id']) for v in good]
        values = [v for v in values if v is not None and v['metrics'] is not None]
        per_resource[resource['id']] = {'valid_views': len(values), 'attribution': 'geometric proximity; pigment unresolved',
            'metrics': {key: float(np.median([v['metrics'][key] for v in values])) for key in metrics} if values else None,
            'color_accuracy': None}
    return {'scorer': identity(), 'valid_views': len(good), 'per_view': records, 'per_resource': per_resource,
            'metrics': {key: float(np.median([v[key] for v in good])) for key in metrics} if good else None}
