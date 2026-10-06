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


def schedule_ops(strokes, bindings, resources, *, speed, max_seconds, motion):
    """Schedule every path in source order; cartridge changes invalidate dip credit."""
    table = {row['id']: row for row in resources}
    ops, current, remaining = [], None, None
    for stroke in strokes:
        identity = bindings[stroke.src['artwork_sha256'], stroke.src['pen']]
        resource = table[identity]
        if identity != current:
            ops.append({'op': 'tool_change', 'id': f't{len(ops):04d}', 'resource_id': identity,
                        'activation': resource['activation'], 'initial': current is None})
            current, remaining = identity, None
        profile = resource['dip']
        capacity = None if profile is None else profile['mm_per_dip']/1000
        initial_dip = remaining is None
        sections, remaining = physical_sections(stroke.points_m, capacity_m=capacity, remaining_m=remaining)
        for points, (lo, hi), dip in sections:
            if dip:
                ops.append({'op': 'dip', 'id': f'd{len(ops):04d}', 'resource_id': identity, 'slot': resource['slot'],
                            'reason': 'initial' if initial_dip and lo == 0 else 'replenish'})
            pieces = split_timed(points, max_seconds, speed, motion)
            for index, (piece, (a, b), seconds) in enumerate(pieces):
                ops.append({'op': 'stroke', 'id': f's{len(ops):04d}', 'resource_id': identity, 'ink': resource['ink_id'],
                            'closed': stroke.closed and len(sections) == len(pieces) == 1, 'continues': index > 0,
                            'src': {**stroke.src, 'arc_m': [lo+a, min(hi, lo+b)]},
                            'generation_width_m': stroke.generation_width_m, 'planned_drawing_s': seconds,
                            'points_m': (piece + 0.).tolist()})
    return ops
