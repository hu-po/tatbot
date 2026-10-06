"""Pure material-coordinate candidates for Session stroke planners.

A candidate names a source stroke and its exact arc interval in the design's
UV chart. Contact and standoff planners may project that same candidate onto
different measured geometry; neither projection nor motion authority lives here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

MAX_SESSION_CHUNKS = 10_000
SESSION_CUT_WINDOW_FRACTION = 0.2
SESSION_CORNER_MIN_RAD = math.radians(5.0)


@dataclass(frozen=True)
class MaterialCandidate:
    source_stroke: int
    uv: np.ndarray
    arc_range_m: tuple[float, float]
    speed_m_s: float | None = None


@dataclass(frozen=True)
class ProjectedMaterial:
    uv: np.ndarray
    points: np.ndarray
    normals: np.ndarray | None


@dataclass(frozen=True)
class ProjectedCandidate:
    """One addressed material interval placed on the selected surface."""

    candidate: MaterialCandidate
    address: dict
    points: np.ndarray
    normals: np.ndarray | None
    projector: Callable | None
    placement_id: str | None


@dataclass(frozen=True)
class MaterialRows:
    """One candidate's sampled material rows, before any transit or preflight."""

    candidate: MaterialCandidate
    uv: np.ndarray
    points: np.ndarray
    normals: np.ndarray | None
    rotations: np.ndarray
    orientation_report: object
    pen: int

    def part(self):
        return self.points, self.rotations, self.pen, None


def material_address(candidate: MaterialCandidate, chunk_index: int,
                     source_chunk_count: int, placement_id: str | None = None) -> dict:
    """Bind a selected chunk to its original design interval and placement.

    The placement may be unavailable to a geometry-only caller. A session
    producer must supply the actual design placement before native admission.
    """
    if (not isinstance(candidate, MaterialCandidate)
            or type(chunk_index) is not int or type(source_chunk_count) is not int
            or not 0 <= chunk_index < source_chunk_count
            or type(candidate.source_stroke) is not int or candidate.source_stroke < 0):
        raise ValueError('session chunk needs a retained material address')
    try:
        uv = np.asarray(candidate.uv, float)
        arc = np.asarray(candidate.arc_range_m, float)
    except (TypeError, ValueError) as error:
        raise ValueError('session chunk needs a retained material address') from error
    if (uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 2
            or not np.isfinite(uv).all() or arc.shape != (2,)
            or not np.isfinite(arc).all()
            or not 0 <= arc[0] < arc[1]
            or (placement_id is not None and (not isinstance(placement_id, str) or not placement_id))):
        raise ValueError('session chunk needs a retained material address')
    address = {'chunk_index': chunk_index, 'source_chunk_count': source_chunk_count,
               'source_stroke': candidate.source_stroke,
               'source_arc_range_m': [float(arc[0]), float(arc[1])],
               'source_uv_start': uv[0].tolist(), 'source_uv_end': uv[-1].tolist()}
    if placement_id is not None:
        address['placement_id'] = placement_id
    return address


def _resampled_segments(poly: np.ndarray, s: np.ndarray):
    """Interpolate one path and retain the segment fractions for its UV chart."""
    poly = np.asarray(poly, float)
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s = np.clip(np.asarray(s, float), 0.0, cum[-1])
    index = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(seg_len[index] > 0.0, (s - cum[index]) / seg_len[index], 0.0)
    points = poly[index] + frac[:, None] * seg[index]
    good = seg_len > 0.0
    unit = np.zeros_like(seg)
    unit[good] = seg[good] / seg_len[good][:, None]
    tangents = unit[index]
    return points, tangents, index, frac


def resample_polyline_by_arclength(poly: np.ndarray, s: np.ndarray):
    """Points and unit tangents of a polyline at arc lengths `s`."""
    points, tangents, _, _ = _resampled_segments(poly, s)
    return points, tangents


def sample_candidate_motion(candidate: MaterialCandidate, distances, *, metric_vertices=None, projector=None):
    """Sample a bounded material candidate into motion positions.

    The metric is UV arc for contact, whose measured surface must be evaluated
    at every sampled UV, or the projected chart chords for standoff, whose
    existing path linearly interpolates XYZ. Both routes use the same segment
    fractions and retain a material coordinate for each motion row. The caller
    still owns its time law, orientation, touch/clearance and native preflight.
    """
    uv = np.asarray(candidate.uv, float)
    metric = uv if metric_vertices is None else np.asarray(metric_vertices, float)
    if (uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < 2 or metric.ndim != 2
            or metric.shape[0] != len(uv) or metric.shape[1] not in (2, 3)
            or not np.isfinite(uv).all() or not np.isfinite(metric).all()
            or (projector is None and metric.shape[1] != 3)):
        raise ValueError('material motion needs matching finite UV and projected vertices')
    sampled_metric, _, index, fraction = _resampled_segments(metric, distances)
    sampled_uv = uv[index] + fraction[:, None] * np.diff(uv, axis=0)[index]
    if projector is None:
        return ProjectedMaterial(sampled_uv, sampled_metric, None)
    points, normals = projector(sampled_uv)
    return ProjectedMaterial(sampled_uv, np.asarray(points, float),
                             None if normals is None else np.asarray(normals, float))


def plan_material_rows(candidate: MaterialCandidate, distances, *, orientation,
                       metric_vertices=None, projector=None, placement=None,
                       normalize_normals=False, pen=0):
    """Build the material phase shared by contact and standoff candidates.

    Sampling and row alignment live here. The caller chooses its metric,
    geometry, tool placement, orientation and pen state. The orientation
    policy returns `(row rotations, policy report)` so contact can carry its
    normal-deviation evidence without a second projection. The caller retains all
    approach/retract motion, interaction guards and native preflight. This is
    deliberately smaller than a full next-stroke or runtime planner.
    """
    projected = sample_candidate_motion(candidate, distances,
                                        metric_vertices=metric_vertices, projector=projector)
    if normalize_normals:
        if projected.normals is None:
            raise ValueError('material rows require surface normals')
        normals = projected.normals
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        projected = ProjectedMaterial(projected.uv, projected.points, normals)
    points = projected.points if placement is None else np.asarray(placement(projected), float)
    rotations, orientation_report = orientation(projected)
    rotations = np.asarray(rotations, float)
    if (points.shape != (len(projected.uv), 3) or rotations.shape != (len(projected.uv), 3, 3)
            or not np.isfinite(points).all() or not np.isfinite(rotations).all()
            or type(pen) is not int or pen not in (0, 1)):
        raise ValueError('material rows need finite aligned positions, rotations and pen policy')
    return MaterialRows(candidate, projected.uv, points, projected.normals, rotations, orientation_report, pen)


def project_uv(uv, projector):
    """Project material vertices onto the caller's geometry.

    The projector supplies measured contact-surface normals or an elevated
    standoff chart. Sampled motion rows use :func:`sample_candidate_motion`;
    timing, touch, clearance, rotation and preflight remain with each planner.
    """
    selected = np.asarray(uv, float)
    points, normals = projector(selected)
    return ProjectedMaterial(selected, np.asarray(points, float),
                             None if normals is None else np.asarray(normals, float))


def project_candidate(candidate: MaterialCandidate, chunk_index: int,
                      source_chunk_count: int, placement_id: str | None,
                      vertex_projector: Callable, *, sample_projector: Callable | None = None):
    """Bind original work to one finite surface projection before tool phases.

    The vertex projection fixes approach endpoints and the standoff metric.
    A measured surface may also project every sampled material row; a nominal
    standoff chart instead interpolates the projected vertices exactly.
    """
    address = material_address(candidate, chunk_index, source_chunk_count, placement_id)
    projected = project_uv(candidate.uv, vertex_projector)
    count = len(candidate.uv)
    if (projected.points.shape != (count, 3) or not np.isfinite(projected.points).all()
            or (projected.normals is not None
                and (projected.normals.shape != (count, 3)
                     or not np.isfinite(projected.normals).all()
                     or np.any(np.linalg.norm(projected.normals, axis=1) <= 0)))):
        raise ValueError('material candidate projection needs finite aligned surface vertices')
    return ProjectedCandidate(candidate, address, projected.points, projected.normals,
                              sample_projector, placement_id)


def cut_points(stroke, arc, count, max_length):
    """Reproducible contact-budget cuts, moved to nearby sharp corners.

    The uniform partition fixes the count; each interior cut may move at most
    a fifth of the budget to a corner, and every resulting part still fits.
    """
    stroke = np.asarray(stroke, float)
    arc = np.asarray(arc, float)
    length = float(arc[-1])
    uniform = np.linspace(0.0, length, count + 1)
    if count < 2 or len(stroke) < 3:
        return uniform
    tangents = np.diff(stroke, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-300)
    turning = np.arccos(np.clip(np.einsum("ij,ij->i", tangents[:-1], tangents[1:]), -1.0, 1.0))
    corners = np.flatnonzero(turning >= SESSION_CORNER_MIN_RAD) + 1
    window = SESSION_CUT_WINDOW_FRACTION * max_length
    cuts = uniform.copy()
    for index in range(1, count):
        previous = float(cuts[index - 1])
        lo = max(uniform[index] - window, length - (count - index) * max_length, previous)
        hi = min(uniform[index] + window, previous + max_length)
        cuts[index] = min(max(float(uniform[index]), lo), hi)
        candidates = corners[(arc[corners] > lo) & (arc[corners] < hi)]
        if len(candidates):
            sharpest = candidates[np.argmax(turning[candidates - 1])]
            cuts[index] = float(arc[sharpest])
    return cuts


def _validated_stroke(raw, speed, source):
    if (raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 2
            or not np.isfinite(raw).all() or (speed is not None and (not math.isfinite(speed) or speed <= 0))):
        raise ValueError(f'invalid session stroke {source}')
    keep = np.r_[True, np.linalg.norm(np.diff(raw, axis=0), axis=1) > 0.0]
    stroke = raw[keep]
    if len(stroke) < 2:
        raise ValueError(f'session stroke {source} has no length')
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(stroke, axis=0), axis=1))]
    return stroke, arc, float(arc[-1])


def material_candidates(strokes_uv, speeds_m_s=None, *, max_lengths_m=None,
                        max_candidates=MAX_SESSION_CHUNKS, preserve_whole=False):
    """Select validated UV candidates before either geometry planner runs.

    Without `max_lengths_m`, each design stroke is one candidate. Contact
    compilation supplies its per-stroke material length budget and receives
    reproducibly cut candidates. Both routes retain source and arc identities.
    """
    strokes = [np.asarray(stroke, float) for stroke in strokes_uv]
    speeds = None if speeds_m_s is None else [float(speed) for speed in speeds_m_s]
    lengths = None if max_lengths_m is None else [float(length) for length in max_lengths_m]
    if (not strokes or (lengths is not None and speeds is None)
            or (speeds is not None and len(speeds) != len(strokes))
            or (lengths is not None and len(lengths) != len(strokes))):
        raise ValueError('session strokes/speeds mismatch')
    result = []
    for source, raw in enumerate(strokes):
        speed = None if speeds is None else speeds[source]
        stroke, arc, length = _validated_stroke(raw, speed, source)
        if lengths is None:
            if not math.isfinite(length) or len(result) + 1 > max_candidates:
                raise ValueError('session exceeds finite chunk budget')
            # An uncut standoff stroke keeps its exported tessellation exactly;
            # contact cuts below retain the historical duplicate removal.
            result.append(MaterialCandidate(source, raw, (0.0, length), speed))
            continue
        max_length = lengths[source]
        ratio = length / max_length if max_length > 0.0 else math.inf
        if not math.isfinite(length) or not math.isfinite(ratio) or ratio > max_candidates:
            raise ValueError('session stroke exceeds the finite planner budget')
        count = max(1, int(math.ceil(ratio)))
        if len(result) + count > max_candidates:
            raise ValueError('session exceeds finite chunk budget')
        if count == 1 and preserve_whole:
            result.append(MaterialCandidate(source, raw, (0.0, length), speed))
            continue
        cuts = cut_points(stroke, arc, count, max_length)
        endpoints, _ = resample_polyline_by_arclength(stroke, cuts)
        for index in range(count):
            lo, hi = float(cuts[index]), float(cuts[index + 1])
            interior = stroke[(arc > lo) & (arc < hi)]
            piece = np.vstack([endpoints[index], interior, endpoints[index + 1]])
            result.append(MaterialCandidate(source, piece, (lo, hi), speed))
    return result
