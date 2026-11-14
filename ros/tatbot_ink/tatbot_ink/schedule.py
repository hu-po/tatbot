"""Ordered resource transitions, physical replenishment cuts and computational chunks.

This produces preparation operations; only tatbot_session executes them.
Contact budgets estimate usable ink, not a sensor measurement of wetness.
"""
from __future__ import annotations

import math

import numpy as np

from tatbot_ink.errors import CompileError
from tatbot_ink.segment import arc_lengths, split, split_timed


def physical_sections(points, *, capacity_m, remaining_m):
    """Return (points, source arc, dip_before) plus final charge estimate.

    None capacity means a self-contained/none supply: one uninterrupted section.
    None remaining means an uncharged dipped resource, which requires an initial
    dip. Consecutive sections share a cut point; they are separate contacts.
    """
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2 or not np.isfinite(points).all():
        raise CompileError('material scheduling requires a finite 2D polyline')
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0]]
    arc = arc_lengths(points)
    length = float(arc[-1])
    if length <= 0:
        raise CompileError('material scheduling requires positive contact length')
    if capacity_m is None:
        return [(points, (0., length), False)], None
    if not math.isfinite(capacity_m) or capacity_m <= 0:
        raise CompileError('contact-length capacity must be positive')
    if remaining_m is not None and (not math.isfinite(remaining_m) or not 0 <= remaining_m <= capacity_m):
        raise CompileError('remaining charge estimate is outside its capacity')
    # Replenish at the preceding path end instead of spending a partial charge
    # on an avoidable extra contact. A full charge can start a long path as-is.
    dip = (remaining_m is None or remaining_m <= 1e-12 or
           (remaining_m < capacity_m - 1e-12 and length > remaining_m + 1e-12))
    if dip:
        remaining_m = capacity_m
    pieces = split(points, remaining_m)
    sections = [(piece, arc, dip or index > 0) for index, (piece, arc) in enumerate(pieces)]
    last_lo, last_hi = pieces[-1][1]
    remaining_m = max(0., (capacity_m if len(pieces) > 1 else remaining_m) - (last_hi - last_lo))
    return sections, remaining_m


def _wipe(resource, previous) -> dict:
    """{'wipe': True} for a resource that asks the print wiped before it draws. The wipe is the operator's, with the
    arm landed for the exchange: the first resource has nothing before it, and the same tool and ink does not land."""
    if not resource.get('wipe_before'):
        return {}
    if previous is None:
        raise CompileError(f"{resource['id']}: wipe_before on the first resource wipes nothing this program drew; a "
                           "wipe after an earlier run is stated with --after PROGRAM --wiped")
    if (previous['tool']['id'], previous['ink_id']) == (resource['tool']['id'], resource['ink_id']):
        raise CompileError(f"{resource['id']}: wipe_before needs the arm landed, and a change to the same tool and "
                           "ink does not land")
    return {'wipe': True}


def schedule_ops(strokes, bindings, resources, *, speed, max_seconds, motion, max_stroke_m=None):
    """Schedule every path in source order; cartridge changes invalidate dip credit. With `max_stroke_m` a path
    longer than it is drawn as strokes of at most that length, each lifted and read again at its own hover (a stroke
    on a person a few seconds long, 2026-10-21), not as chunks that continue pen-down."""
    table = {row['id']: row for row in resources}
    ops, current, remaining = [], None, None
    for stroke in strokes:
        identity = bindings[stroke.src['artwork_sha256'], stroke.src['pen']]
        resource = table[identity]
        if identity != current:
            ops.append({'op': 'tool_change', 'id': f't{len(ops):04d}', 'resource_id': identity,
                        'activation': resource['activation'], 'initial': current is None,
                        **_wipe(resource, None if current is None else table[current])})
            current, remaining = identity, None
        profile = resource['dip']
        capacity = None if profile is None else profile['mm_per_dip']/1000
        if stroke.deposition is not None and resource['tool'].get('fade_lift_mm') is None:
            raise CompileError(f"path {stroke.src['path']} fades, and {resource['tool']['id']}'s datasheet records no "
                               "fade_lift_mm: measure its fade with a ride ladder first")
        initial_dip, arc0 = remaining is None, getattr(stroke, 'arc0_m', 0.)   # a path cut at a keep region's edge
        sections, remaining = physical_sections(stroke.points_m, capacity_m=capacity, remaining_m=remaining)
        for points, (lo, hi), dip in sections:
            if dip:
                ops.append({'op': 'dip', 'id': f'd{len(ops):04d}', 'resource_id': identity, 'slot': resource['slot'],
                            'reason': 'initial' if initial_dip and lo == 0 else 'replenish'})
            parts = [(points, (0., hi - lo))] if max_stroke_m is None else split(points, max_stroke_m)
            for part, (pa, _) in parts:
                pieces = split_timed(part, max_seconds, speed, motion)
                for index, (piece, (a, b), seconds) in enumerate(pieces):
                    shares = stroke.deposition_at(lo + pa + a + arc_lengths(piece))   # the piece's own, cut and all
                    ops.append({'op': 'stroke', 'id': f's{len(ops):04d}', 'resource_id': identity,
                                'ink': resource['ink_id'],
                                'closed': stroke.closed and len(sections) == len(parts) == len(pieces) == 1,
                                'continues': index > 0, 'src': {**stroke.src, 'arc_m': [arc0+lo+pa+a, arc0+min(hi, lo+pa+b)]},
                                'generation_width_m': stroke.generation_width_m, 'planned_drawing_s': seconds,
                                'points_m': (piece + 0.).tolist(),
                                **({} if shares is None else {'deposition': shares.round(6).tolist()})})
    return ops


def line_per_dip(ops, resources) -> dict:
    """How much line each dip draws against what its resource says one carries: {resource id: {mm_per_dip, dips,
    line_mm (min, median, max, total), fill (the median dip's share of mm_per_dip), under_half (dips that draw
    less than half of it), first_stroke_mm (min and median of the stroke that takes the fresh load), strokes_per_dip
    and stroke_mm (medians)}} for every resource that dips.

    A dip takes the same load whatever follows it, so one that draws a short line lays the rest of its load at the
    start of that line: 2026-11-10's red drew 64 dips over 2.95 m of strokes 6.8 mm long at the median. A report,
    not a schedule: the dwell is not varied by it until the lab has a curve of load against dwell."""
    table = {row['id']: row for row in resources}
    cycles: dict[str, list[list[float]]] = {}
    current = None
    for op in ops:
        if op['op'] == 'dip':
            current = cycles.setdefault(op['resource_id'], [])
            current.append([])
        elif op['op'] == 'tool_change':
            current = None
        elif current is not None:
            points = np.asarray(op['points_m'], float)
            closing = float(np.linalg.norm(points[0] - points[-1])) if op.get('closed') else 0.0
            current[-1].append((float(arc_lengths(points)[-1]) + closing) * 1000)
    out = {}
    for identity, dips in cycles.items():
        per_dip = float(table[identity]['dip']['mm_per_dip'])
        line = np.array([sum(strokes) for strokes in dips])
        strokes = np.array([length for dip in dips for length in dip] or [0.0])
        first = np.array([dip[0] for dip in dips if dip] or [0.0])
        out[identity] = {'mm_per_dip': per_dip, 'dips': len(dips),
                         'line_mm': {'min': round(float(line.min()), 1), 'median': round(float(np.median(line)), 1),
                                     'max': round(float(line.max()), 1), 'total': round(float(line.sum()), 1)},
                         'fill': round(float(np.median(line)) / per_dip, 3),
                         'under_half': int((line < per_dip / 2).sum()),
                         'first_stroke_mm': {'min': round(float(first.min()), 1),
                                             'median': round(float(np.median(first)), 1)},
                         'strokes_per_dip': round(float(np.median([len(dip) for dip in dips])), 1),
                         'stroke_mm': round(float(np.median(strokes)), 1)}
    return out
