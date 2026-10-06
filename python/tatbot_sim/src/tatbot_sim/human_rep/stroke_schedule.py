"""Rank, dedup and order material strokes before capacity splitting.

Three stages over the strokes `material_strokes()` produces, run inside each
consecutive `(source layer, ink)` group so colour changes, dips and barriers
keep their sequence:

1. **Tier.** Explicit elements (path, cubic_bezier, dots, stipple) and the
   fill planner's boundary rings (inset depth 0) are tier 0; insets 1-2 are
   tier 1; deeper insets are tier 2. A stable sort by tier, original index as
   the tie-break, draws the silhouette first and the detail last.
2. **Dedup by footprint coverage.** Walking in tier order with a union of the
   footprints kept so far, a fill ring whose own footprint adds less than
   `MIN_NEW_AREA_FRACTION` of its area is dropped, unless the drop would take
   its paint component below the coverage floor or cost it more than
   `MAX_COVERAGE_LOSS_FRACTION` of its area in total. The footprint is the tool's
   line width when the datasheet records one, else the planning width, so an
   over-fine planning width stops doubling ink. Explicit elements are never
   dropped. Every drop is reported.
3. **Order within a tier.** Greedy nearest neighbour from the previous
   stroke's end, ties to the lower original index. A closed ring is rotated
   to start at the vertex nearest the previous end; an open stroke is
   reversed when its far end is nearer. The point arrays are the truth
   (nothing executes on `start_choice` or `direction`).

The schedule is derived and lives in the compiled program; source artwork and
its canonical TattooProgram are never rewritten. This module has no body,
robot, service or ink-capacity authority.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Sequence

import numpy as np
from shapely import LineString, STRtree, union_all

from tatbot_sim.human_rep.contracts import ContractError
from tatbot_sim.human_rep.fill_geometry import (
    ARC_SEGMENTS,
    MAX_FILL_STROKES,
    PAINT_COVERAGE_FLOOR,
    deposited_coverage,
)

SCHEMA = "tatbot.stroke-schedule/1"
# A fill ring that adds less than this fraction of its own footprint area to
# what is already covered is redundant at the footprint width.
MIN_NEW_AREA_FRACTION = 0.25
# What dedup may cost a paint component in total, as a fraction of its area,
# on top of the coverage floor: a drop is a deliberate loss, not a planner
# limitation, so it may not spend the qualification budget. At 1 % the
# collection's 0.98 footprint-IoU gate is safe for every plan above 0.99.
MAX_COVERAGE_LOSS_FRACTION = 0.01
# Closed rings offer every vertex as a start; the nearest-neighbour search
# samples at most this many per ring, and the chosen ring is then rotated to
# its exact nearest vertex.
RING_SAMPLES = 64


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _closed(points: np.ndarray) -> bool:
    return len(points) > 3 and bool(np.array_equal(points[0], points[-1]))


def pen_up_travel_m(strokes: Sequence[Any]) -> float:
    """Straight-line pen-up distance between consecutive strokes."""
    return float(sum(np.linalg.norm(b.points_m[0] - a.points_m[-1])
                     for a, b in zip(strokes[:-1], strokes[1:], strict=True)))


def _groups(strokes: Sequence[Any]) -> list[list[int]]:
    """Original indices, split into consecutive runs of one (layer, ink)."""
    groups: list[list[int]] = []
    key = None
    for index, stroke in enumerate(strokes):
        current = (stroke.layer_index, stroke.ink_id)
        if current != key:
            groups.append([])
            key = current
        groups[-1].append(index)
    return groups


def _footprint(stroke: Any, width_m: float):
    return LineString(stroke.points_m).buffer(width_m / 2, quad_segs=ARC_SEGMENTS)


def _dedup(strokes: Sequence[Any], order: list[int], *, footprint_width_m: float | None,
           min_new_area_fraction: float, floor: float, max_loss_fraction: float, dropped: list[dict]) -> list[int]:
    """Stage two: the indices of `order` that survive at the footprint width."""
    def width(stroke: Any) -> float:
        return float(footprint_width_m) if footprint_width_m is not None else float(stroke.width_m)

    # Coverage of each paint component with every one of its rings kept, at
    # the footprint width. A drop may only spend what lies above the floor, and
    # it is charged the whole of the ring's new area: later rings may cover
    # some of it again, so the bound is conservative.
    components: dict[int, dict[str, Any]] = {}
    for index in order:
        stroke = strokes[index]
        if stroke.component is None:
            continue
        entry = components.setdefault(id(stroke.component), {"polygon": stroke.component, "paths": [],
                                                              "width": width(stroke), "lost": 0.0})
        entry["paths"].append(stroke.points_m)
    for entry in components.values():
        polygon = entry["polygon"]
        entry["area"] = float(polygon.area)
        entry["full"] = float(deposited_coverage(entry["paths"], entry["width"]).intersection(polygon).area)
    kept: list[int] = []
    footprints: list = []
    tree = None
    for index in order:
        stroke = strokes[index]
        footprint = _footprint(stroke, width(stroke))
        if stroke.inset_depth is not None and stroke.component is not None:
            if tree is None:
                new_area = footprint.area
            else:
                nearby = tree.query(footprint, predicate="intersects")
                covered = union_all([footprints[i] for i in nearby]) if len(nearby) else None
                new_area = footprint.area if covered is None else footprint.difference(covered).area
            fraction = new_area / footprint.area if footprint.area > 0 else 1.0
            if fraction < min_new_area_fraction:
                entry = components[id(stroke.component)]
                remaining = (entry["full"] - entry["lost"] - new_area) / entry["area"]
                if remaining >= floor and (entry["lost"] + new_area) / entry["area"] <= max_loss_fraction:
                    entry["lost"] += new_area
                    dropped.append({"index": index, "source_primitive_sha256": stroke.source_primitive_sha256,
                                    "length_m": _length(stroke.points_m), "new_area_fraction": float(fraction),
                                    "footprint_width_m": width(stroke), "tier": stroke.tier,
                                    "inset_depth": stroke.inset_depth,
                                    "reason": "redundant_at_footprint"})
                    continue
        kept.append(index)
        footprints.append(footprint)
        tree = STRtree(footprints)
    return kept


def _candidates(points: np.ndarray) -> np.ndarray:
    """Where a stroke may start: every sampled vertex of a ring, else its ends."""
    if _closed(points):
        ring = points[:-1]
        stride = max(1, math.ceil(len(ring) / RING_SAMPLES))
        return ring[::stride]
    return points[[0, -1]]


def _align(stroke: Any, cursor: np.ndarray) -> Any:
    """Rotate a ring / reverse an open stroke to start nearest the cursor."""
    points = stroke.points_m
    if _closed(points):
        ring = points[:-1]
        vertex = int(np.argmin(np.linalg.norm(ring - cursor, axis=1)))
        if vertex == 0:
            return stroke
        rotated = np.concatenate([ring[vertex:], ring[:vertex], ring[vertex:vertex + 1]])
        return replace(stroke, points_m=rotated,
                       ordering_rationale=f"{stroke.ordering_rationale}; rotated to vertex {vertex}")
    if np.linalg.norm(points[-1] - cursor) < np.linalg.norm(points[0] - cursor):
        return replace(stroke, points_m=points[::-1].copy(),
                       ordering_rationale=f"{stroke.ordering_rationale}; reversed")
    return stroke


def _order(strokes: Sequence[Any], indices: list[int], cursor: np.ndarray | None) -> tuple[list[Any], np.ndarray | None]:
    """Stage three over one tier: greedy nearest neighbour, ties to the lower index."""
    remaining = list(indices)
    output: list[Any] = []
    while remaining:
        if cursor is None:
            choice = 0
            stroke = strokes[remaining[choice]]
        else:
            distances = [float(np.linalg.norm(_candidates(strokes[i].points_m) - cursor, axis=1).min())
                         for i in remaining]
            choice = int(np.argmin(distances))
            stroke = _align(strokes[remaining[choice]], cursor)
        remaining.pop(choice)
        output.append(stroke)
        cursor = stroke.points_m[-1]
    return output, cursor


def schedule(strokes: Sequence[Any], *, footprint_width_m: float | None = None,
             min_new_area_fraction: float = MIN_NEW_AREA_FRACTION,
             max_loss_fraction: float = MAX_COVERAGE_LOSS_FRACTION,
             coverage_floor: float | None = None, report: dict | None = None, preserve_order: bool = False) -> list[Any]:
    """Tier, dedup and order material strokes; fill `report` with the audit trail.

    `footprint_width_m` is the tool's measured line width (None: each stroke's
    planning width). `report`, when given, receives the `tatbot.stroke-schedule/1`
    document: per-tier totals, every dropped stroke and the pen-up travel
    before and after.
    """
    if not (0.0 <= max_loss_fraction <= 1.0):
        raise ContractError("out_of_range", "$.max_loss_fraction", "expected [0, 1]")
    strokes = list(strokes)
    if len(strokes) > MAX_FILL_STROKES:
        raise ContractError("tattoo_program_over_budget", "$.layers", "schedule exceeds 100000 strokes")
    if footprint_width_m is not None and not (math.isfinite(footprint_width_m) and footprint_width_m > 0):
        raise ContractError("out_of_range", "$.footprint_width_m", "expected a positive finite footprint width")
    if not (0.0 <= min_new_area_fraction <= 1.0):
        raise ContractError("out_of_range", "$.min_new_area_fraction", "expected [0, 1]")
    floor = PAINT_COVERAGE_FLOOR if coverage_floor is None else float(coverage_floor)
    dropped: list[dict] = []
    output: list[Any] = []
    cursor: np.ndarray | None = None
    if preserve_order:
        output = list(strokes)
    for group in ([] if preserve_order else _groups(strokes)):
        ordered = sorted(group, key=lambda index: (strokes[index].tier, index))
        kept = _dedup(strokes, ordered, footprint_width_m=footprint_width_m,
                      min_new_area_fraction=min_new_area_fraction, floor=floor,
                      max_loss_fraction=max_loss_fraction, dropped=dropped)
        for tier in sorted({strokes[index].tier for index in kept}):
            scheduled, cursor = _order(strokes, [index for index in kept if strokes[index].tier == tier], cursor)
            output.extend(scheduled)
    output = [replace(stroke, schedule_index=position) for position, stroke in enumerate(output)]
    if report is not None:
        tiers = sorted({stroke.tier for stroke in strokes})
        report.clear()
        report.update({
            "schema": SCHEMA,
            "footprint_width_m": footprint_width_m,
            "min_new_area_fraction": float(min_new_area_fraction),
            "max_coverage_loss_fraction": float(max_loss_fraction),
            "coverage_floor": floor,
            "input_strokes": len(strokes),
            "output_strokes": len(output),
            "tiers": [{"tier": tier,
                       "strokes": sum(stroke.tier == tier for stroke in output),
                       "length_m": sum(_length(stroke.points_m) for stroke in output if stroke.tier == tier),
                       "dropped_strokes": sum(item["tier"] == tier for item in dropped),
                       "dropped_m": sum(item["length_m"] for item in dropped if item["tier"] == tier)}
                      for tier in tiers],
            "dedup_dropped_strokes": len(dropped),
            "dedup_dropped_m": sum(item["length_m"] for item in dropped),
            "dropped": sorted(dropped, key=lambda item: item["index"]),
            "pen_up_travel_m": {"generation_order": pen_up_travel_m(strokes), "scheduled": pen_up_travel_m(output)},
            "contact_length_m": {"generation_order": sum(_length(s.points_m) for s in strokes),
                                 "scheduled": sum(_length(s.points_m) for s in output)},
        })
    return output
