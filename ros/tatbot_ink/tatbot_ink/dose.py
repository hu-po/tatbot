"""A program's dose: the needle pokes each stroke lays along its line and each spot of the page takes.

tatbot_ink.passes counts how often the pen comes over a spot; this counts what the pen does while it is there.
Each stroke's pokes come from the planner's time law and the machine's rate (tatbot_motion.dose), so an ease, a
corner and the dwell where the pen comes down and lifts all show. A cell's figure is the pokes of every needle
group that covered it: a mark's width of line at the steady dose in one pass, more where the pen was slow or
came back.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from tatbot_motion import dose as stroke_dose
from tatbot_motion.plan import STOP, blend_corners, corner_blend_m
from tatbot_motion.timelaw import arc_lengths

from tatbot_ink import passes

AT_LEAST = (2, 3, 5)      # the area reported at or over these many steady passes' pokes
HOTSPOTS = 5              # the densest cells, at least HOTSPOT_APART_M apart
HOTSPOT_APART_M = 0.002
_recent: dict[str, dict] = {}   # strokes()' last results: a compile's preview asks again for what its stats just ran


def _line(op: dict, motion: dict) -> np.ndarray:
    """A stroke's page polyline as the planner draws it: closed back to its start, its corners blended."""
    points = np.asarray(op["points_m"], float).reshape(-1, 2)
    if op.get("closed") and len(points) > 1 and np.linalg.norm(points[0] - points[-1]) > 1e-9:
        points = np.vstack([points, points[:1]])
    blend = corner_blend_m(op, motion)
    return blend_corners(points, blend)[0] if blend > 0.0 and len(points) > 2 else points


def strokes(program: dict, motion: dict) -> dict:
    """{stroke op id: tatbot_motion.dose.StrokeDose} as the session will plan each: at the program's speed under
    the pen-down cap, joined pen-down where the op or the one after it continues, its corners blended and its ends
    stopping or flying by `motion`, a flying end stopping where its run would leave the page's clear area."""
    ops = program["ops"]
    key = hashlib.sha256(json.dumps([ops, program["draw_speed_m_s"], program["page"]["clear_m"],
                                     [(row["id"], row.get("pen_mode")) for row in program["resources"]], motion],
                                    sort_keys=True).encode()).hexdigest()
    if key in _recent:
        return _recent[key]
    cruise = min(float(program["draw_speed_m_s"]), float(motion["tip_speed"]["pen_down_max_m_s"]))
    settle = {row["id"]: float(motion["pen"][row.get("pen_mode") or motion["pen"]["mode"]]["settle_s"])
              for row in program["resources"]}
    out = {}
    for index, op in enumerate(ops):
        if op["op"] != "stroke":
            continue
        following = ops[index + 1] if index + 1 < len(ops) else None
        lift = not (following is not None and following["op"] == "stroke" and following.get("continues"))
        out[op["id"]] = stroke_dose.stroke(
            _line(op, motion), cruise, motion, joined=bool(op.get("continues")),
            lift_at_end=lift, clear_m=program["page"]["clear_m"], settle_s=settle[op["resource_id"]])
    while len(_recent) >= 2:
        _recent.pop(next(iter(_recent)))
    _recent[key] = out
    return out


def _spread(values) -> dict:
    values = np.asarray([v for v in values if v is not None], float)
    if not len(values):
        return {"median": None, "p95": None, "max": None}
    return {"median": round(float(np.median(values)), 1), "p95": round(float(np.percentile(values, 95)), 1),
            "max": round(float(values.max()), 1)}


def grid(program: dict, motion: dict, doses: dict | None = None):
    """(pokes [row y, column x], steady passes [row y, column x], the grid's lower-left corner (x, y), its cell), or
    None without a stroke. Each stroke is walked at its counted width (passes.marks) on cells a sixth of the
    narrowest. A cell takes the pokes laid while a disc that wide about the pen covered its centre, and the dwells
    at a stroke's ends with its first and last position. The second grid is the same in steady passes: the times
    the stroke passed the cell (tatbot_ink.passes' rule: again once it comes back after going more than its width
    along the line) times how its pokes there compare with the steady dose over the same line, and each dwell over
    the pokes a steady pass lays across the mark."""
    doses = strokes(program, motion) if doses is None else doses
    width = passes._widths(program)
    lines = [(_line(op, motion), float(width[op["id"]]), doses[op["id"]]) for op in program["ops"] if op["op"] == "stroke"]
    if not lines:
        return None
    cell = min(width for _, width, _ in lines) / 6
    pad = max(width for _, width, _ in lines) / 2 + cell
    every = np.vstack([points for points, _, _ in lines])
    origin = every.min(axis=0) - pad
    nx, ny = np.ceil((every.max(axis=0) + pad - origin) / cell).astype(int) + 1
    pokes, steady = np.zeros(nx * ny), np.zeros(nx * ny)
    for points, width, d in lines:
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0]]
        arc = arc_lengths(points)
        edges = np.linspace(0.0, arc[-1], int(np.ceil(arc[-1] / cell)) + 2)
        s = 0.5 * (edges[:-1] + edges[1:])
        laid, even = np.diff(d.laid(edges)), np.maximum(np.diff(edges), 1e-12) * 1000.0 * d.steady_per_mm
        dwell = np.zeros(len(s))
        dwell[0] += d.touchdown
        dwell[-1] += d.liftoff
        xy = np.c_[np.interp(s, arc, points[:, 0]), np.interp(s, arc, points[:, 1])] - origin
        offsets = np.arange(-int(np.ceil(width / 2 / cell)), int(np.ceil(width / 2 / cell)) + 1)
        col = np.floor(xy[:, 0] / cell).astype(int)[:, None, None] + offsets[None, None, :]
        row = np.floor(xy[:, 1] / cell).astype(int)[:, None, None] + offsets[None, :, None]
        near = ((col + 0.5) * cell - xy[:, 0, None, None]) ** 2 + ((row + 0.5) * cell - xy[:, 1, None, None]) ** 2
        sample, r, c = np.nonzero(near <= (width / 2) ** 2)
        flat = row[sample, r, 0] * nx + col[sample, 0, c]
        order = np.lexsort((sample, flat))
        flat, sample = flat[order], sample[order]
        again = np.r_[True, (np.diff(flat) != 0) | (np.diff(sample) > round(width / cell))]
        times = np.bincount(flat[again], minlength=nx * ny)
        moving, stood = np.bincount(flat, laid[sample], minlength=nx * ny), np.bincount(flat, dwell[sample], minlength=nx * ny)
        pokes += moving + stood
        steady += times * moving / np.maximum(np.bincount(flat, even[sample], minlength=nx * ny), 1e-12) \
            + stood / (d.steady_per_mm * width * 1000.0)
    return pokes.reshape(ny, nx), np.round(steady, 2).reshape(ny, nx), origin, cell


def cells(raster) -> dict:
    """The most and the 95th percentile of pokes an inked cell takes and of the steady passes that is, the mm2 at or
    over AT_LEAST steady passes, and the hotspots."""
    if raster is None:
        return {"cell_mm": None, "inked_mm2": 0.0, "max_pokes": 0.0, "p95_pokes": 0.0, "max_passes": 0.0,
                "p95_passes": 0.0, "mm2_at_least": {str(n): 0.0 for n in AT_LEAST}, "hotspots": []}
    pokes, steady, origin, cell = raster
    inked, area = pokes > 0, cell * cell * 1e6
    rows, cols = np.nonzero(steady > 1.0)
    order = np.argsort(-steady[rows, cols], kind="stable")
    at, value, count = (np.c_[cols[order], rows[order]] + 0.5) * cell + origin, steady[rows[order], cols[order]], \
        pokes[rows[order], cols[order]]
    hotspots = []
    while len(value) and len(hotspots) < HOTSPOTS:
        # + 0.0: a spot just under the axis rounds to -0.0, which no program may hold (canonical numbers)
        hotspots.append({"at_mm": [round(float(v) * 1000, 1) + 0.0 for v in at[0]], "pokes": round(float(count[0]), 1),
                         "passes": round(float(value[0]), 1)})
        keep = np.linalg.norm(at - at[0], axis=1) >= HOTSPOT_APART_M
        at, value, count = at[keep], value[keep], count[keep]
    return {"cell_mm": round(cell * 1000, 4), "inked_mm2": round(float(inked.sum() * area), 2),
            "max_pokes": round(float(pokes.max()), 1), "p95_pokes": round(float(np.percentile(pokes[inked], 95)), 1),
            "max_passes": round(float(steady.max()), 1), "p95_passes": round(float(np.percentile(steady[inked], 95)), 1),
            "mm2_at_least": {str(n): round(float((steady >= n).sum() * area), 2) for n in AT_LEAST}, "hotspots": hotspots}


def summary(program: dict, motion: dict, doses: dict | None = None) -> dict:
    """stats.dose: the machine's rate and the ends it was counted with, the steady dose, every stroke's start,
    middle, end and corner peak (pokes per mm), its share of line over twice the steady dose and the pokes on the
    spot where it comes down and lifts, their spread over the program, and the cells (cells)."""
    doses = strokes(program, motion) if doses is None else doses
    rows = {op_id: d.summary() for op_id, d in doses.items()}
    line = sum(d.length_m for d in doses.values())
    cruise = min(float(program["draw_speed_m_s"]), float(motion["tip_speed"]["pen_down_max_m_s"]))
    steady = stroke_dose.rate_hz(motion) / (cruise * 1000.0)
    total = sum(d.total for d in doses.values())
    doubled = [d for d in doses.values() if d.length_m > 1e-6 and d.total > 2.0 * steady * d.length_m * 1000.0]
    return {
        "model": stroke_dose.MODEL, "rate_hz": stroke_dose.rate_hz(motion), "ends": motion["draw"].get("ends", STOP),
        "ends_drawn": {kind: sum((d.start == kind) + (d.end == kind) for d in doses.values())
                       for kind in sorted({d.start for d in doses.values()} | {d.end for d in doses.values()})},
        "steady_per_mm": round(steady, 2), "pokes": round(total), "mean_per_mm": round(total / (line * 1000.0), 1) if line else None,
        "over_2x": {"mm": round(sum(row["over_mm"] for row in rows.values()), 1),
                    "share": round(sum(row["over_mm"] for row in rows.values()) / (line * 1000.0), 3) if line else None,
                    "strokes_end_to_end": len(doubled)},
        "start_per_mm": _spread(row["start_per_mm"] for row in rows.values()),
        "middle_per_mm": _spread(row["middle_per_mm"] for row in rows.values()),
        "end_per_mm": _spread(row["end_per_mm"] for row in rows.values()),
        "corner_peak_per_mm": {**_spread(row["corner_peak_per_mm"] for row in rows.values()),
                               "strokes": sum(row["corner_peak_per_mm"] is not None for row in rows.values())},
        "touchdown_pokes": round(max((d.touchdown for d in doses.values()), default=0.0), 1),
        "liftoff_pokes": round(max((d.liftoff for d in doses.values()), default=0.0), 1),
        "cells": cells(grid(program, motion, doses)),
        # per stroke: start, middle, end, corner peak (pokes per mm; null: a dot, no slowed corner), share over 2x,
        # and the pokes on the spot where it comes down and where it lifts (0: that end flies or is joined)
        "strokes": {op_id: [row["start_per_mm"], row["middle_per_mm"], row["end_per_mm"], row["corner_peak_per_mm"],
                            row["over_share"], row["touchdown_pokes"], row["liftoff_pokes"]]
                    for op_id, row in rows.items()},
    }


def headline(dose: dict) -> str:
    """stats.dose in a line, for a preview's title and a log."""
    def times(key):
        value = dose[key]["median"]
        return "-" if value is None else f"{value / dose['steady_per_mm']:.1f}x"

    corner = dose["corner_peak_per_mm"]
    return (f"dose at {dose['rate_hz']:g} Hz, ends {dose['ends']}: {dose['steady_per_mm']:.1f} pokes/mm steady, starts "
            f"{times('start_per_mm')}, ends {times('end_per_mm')}"
            + (f", corners to {corner['max'] / dose['steady_per_mm']:.1f}x" if corner["max"] is not None else "")
            + f", {dose['touchdown_pokes']:.0f} + {dose['liftoff_pokes']:.0f} pokes on the spot at each stroke's ends, "
            f"{(dose['over_2x']['share'] or 0.0) * 100:.0f}% of the line over 2x; a spot takes up to "
            f"{dose['cells']['max_pokes']:.0f} pokes ({dose['cells']['max_passes']:.1f} steady passes)")
