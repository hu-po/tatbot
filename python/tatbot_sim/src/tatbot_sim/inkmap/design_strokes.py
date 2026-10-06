"""A portable design's material centerlines as chart polylines.

`design_strokes` returns a `tatbot.draw-strokes/1` document, which the design
scene reads. The polylines are the scheduled material strokes
(`scheduled_material_strokes` + `target_stroke_points`), in chart millimetres
about chart zero and in schedule order (tier 0 silhouette first, redundant fill
rings dropped).

This is geometry only: no tool, no pose, no pressure, no authority to move.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tatbot_sim.inkmap.design import validate_design

SCHEMA = "tatbot.draw-strokes/1"
# A material stroke shorter than this is a fragment of the fill planner (a sub-millimetre
# link or sliver), not a mark the pen can pace: it is dropped and counted, never drawn.
MIN_STROKE_MM = 1.0


def design_strokes(value: dict[str, Any], *, min_stroke_mm: float = MIN_STROKE_MM,
                   fill_style: str = "concentric") -> dict[str, Any]:
    """Material centerlines of a validated plane/cylinder design, in chart millimetres."""
    from tatbot_sim.human_rep.ink_program import scheduled_material_strokes, target_stroke_points

    if not (min_stroke_mm >= 0.0 and np.isfinite(min_stroke_mm)):
        raise ValueError("min_stroke_mm must be a finite non-negative length")
    design = validate_design(value)
    geometry = None
    strokes: list[list[list[float]]] = []
    stroke_placement_ids: list[str] = []
    widths: list[float] = []
    tiers: list[int] = []
    depths: list[int | None] = []
    schedules: list[dict[str, Any]] = []
    length_mm = 0.0
    dropped = 0
    dropped_mm = 0.0
    for item in design["placements"]:
        placement = item["placement"]
        target = placement["target"]
        if target["kind"] not in {"plane", "cylinder"}:
            raise ValueError("body placements need a measured body-to-surface address mapping")
        current = {key: target[key] for key in ("kind", "canvas_m")}
        if target["kind"] == "cylinder":
            current["radius_m"] = target["radius_m"]
        if geometry is not None and current != geometry:
            raise ValueError("stroke export needs one target chart geometry across placements")
        geometry = current
        report: dict[str, Any] = {}
        scheduled = scheduled_material_strokes(design["artworks"][item["artwork_id"]]["program"], placement,
                                               report=report, fill_style=fill_style)
        schedules.append({"placement_id": item["id"], **report})
        for stroke in scheduled:
            points = np.asarray(target_stroke_points(stroke, target), dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError("material stroke is not a finite planar polyline")
            keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.0]
            points = points[keep]
            if len(points) < 2:
                continue
            stroke_mm = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) * 1000.0
            if stroke_mm < min_stroke_mm:
                dropped += 1
                dropped_mm += stroke_mm
                continue
            length_mm += stroke_mm
            strokes.append((points * 1000.0).tolist())
            stroke_placement_ids.append(item["id"])
            widths.append(float(stroke.width_m) * 1000.0)
            tiers.append(int(stroke.tier))
            depths.append(stroke.inset_depth)
    if not strokes:
        raise ValueError("design has no material strokes")
    everything = np.concatenate([np.asarray(s, dtype=float) for s in strokes])
    return {
        "schema": SCHEMA,
        "name": design["name"],
        "design_sha256": design["content_sha256"],
        "target": geometry,
        "stroke_count": len(strokes),
        "stroke_placement_ids": stroke_placement_ids,
        "stroke_width_mm": max(widths),
        "length_mm": length_mm,
        "min_stroke_mm": float(min_stroke_mm),
        "dropped_short_strokes": dropped,
        "dropped_short_mm": dropped_mm,
        "fill_style": fill_style,
        "dedup_dropped_strokes": sum(int(item["dedup_dropped_strokes"]) for item in schedules),
        "dedup_dropped_mm": sum(float(item["dedup_dropped_m"]) for item in schedules) * 1000.0,
        "stroke_tiers": tiers,
        "stroke_depths": depths,
        "schedule": schedules,
        "bounds_mm": [everything.min(axis=0).tolist(), everything.max(axis=0).tolist()],
        "radius_mm": float(np.linalg.norm(everything, axis=1).max()),
        "strokes_mm": strokes,
    }
