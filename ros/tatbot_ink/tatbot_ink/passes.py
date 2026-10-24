"""How often the pen passes over each spot of the page: a program's strokes, or a run's drawn lines. Tracking only."""
from __future__ import annotations

import numpy as np
from tatbot_motion.timelaw import arc_lengths

AT_LEAST = (2, 3, 5)      # the area reported at or over these many passes
HOTSPOTS = 5              # the densest HOTSPOT_M squares by mean passes, at least HOTSPOT_APART_M apart
HOTSPOT_M, HOTSPOT_APART_M = 0.001, 0.002
SMOOTH = 5                # measured samples each drawn line is averaged over (drawn_lines)


def grid(lines):
    """(passes [row y, column x], the grid's lower-left corner (x, y), its cell) of (points m, pen width m) lines, or
    None without one."""
    # Each line is walked at its width on cells a sixth of the narrowest. A cell is passed again each time a pen's
    # disc comes back over its centre after going more than a pen width along the line: a retrace, another stroke.
    lines = [(np.asarray(points, float).reshape(-1, 2), float(width)) for points, width in lines if len(points)]
    if not lines:
        return None
    cell = min(width for _, width in lines) / 6
    pad = max(width for _, width in lines) / 2 + cell
    every = np.vstack([points for points, _ in lines])
    origin = every.min(axis=0) - pad
    nx, ny = np.ceil((every.max(axis=0) + pad - origin) / cell).astype(int) + 1
    counts = np.zeros(nx * ny, np.int32)
    for points, width in lines:
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0]]
        arc = arc_lengths(points)
        s = np.linspace(0.0, arc[-1], int(np.ceil(arc[-1] / cell)) + 2)
        xy = np.c_[np.interp(s, arc, points[:, 0]), np.interp(s, arc, points[:, 1])] - origin
        offsets = np.arange(-int(np.ceil(width / 2 / cell)), int(np.ceil(width / 2 / cell)) + 1)
        col = np.floor(xy[:, 0] / cell).astype(int)[:, None, None] + offsets[None, None, :]
        row = np.floor(xy[:, 1] / cell).astype(int)[:, None, None] + offsets[None, :, None]
        near = ((col + 0.5) * cell - xy[:, 0, None, None]) ** 2 + ((row + 0.5) * cell - xy[:, 1, None, None]) ** 2
        sample, r, c = np.nonzero(near <= (width / 2) ** 2)
        flat = row[sample, r, 0] * nx + col[sample, 0, c]
        order = np.lexsort((sample, flat))
        flat, sample = flat[order], sample[order]
        np.add.at(counts, flat[np.r_[True, (np.diff(flat) != 0) | (np.diff(sample) > round(width / cell))]], 1)
    return counts.reshape(ny, nx), origin, cell


def summary(lines) -> dict:
    """The most and the 95th percentile of passes per inked cell, the mm2 at AT_LEAST passes and the hotspots."""
    raster = grid(lines)
    if raster is None:
        return {"cell_mm": None, "inked_mm2": 0.0, "max": 0, "p95": 0, "mm2_at_least": {str(n): 0.0 for n in AT_LEAST},
                "hotspots": []}
    counts, origin, cell = raster
    inked, area = counts[counts > 0], cell * cell * 1e6
    k = min(max(1, round(HOTSPOT_M / cell)), *counts.shape)
    total = np.pad(counts, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    mean = (total[k:, k:] - total[:-k, k:] - total[k:, :-k] + total[:-k, :-k]) / (k * k)
    rows, cols = np.nonzero(mean > 1.0)
    order = np.argsort(-mean[rows, cols], kind="stable")
    at, value = (np.c_[cols[order], rows[order]] + k / 2) * cell + origin, mean[rows[order], cols[order]]
    hotspots = []
    while len(value) and len(hotspots) < HOTSPOTS:
        # + 0.0: a spot just under the axis rounds to -0.0, which no program may hold (canonical numbers)
        hotspots.append({"at_mm": [round(float(v) * 1000, 1) + 0.0 for v in at[0]], "mean_passes": round(float(value[0]), 2)})
        keep = np.linalg.norm(at - at[0], axis=1) >= HOTSPOT_APART_M
        at, value = at[keep], value[keep]
    return {"cell_mm": round(cell * 1000, 4), "inked_mm2": round(inked.size * area, 2), "max": int(inked.max()),
            "p95": int(np.percentile(inked, 95)),
            "mm2_at_least": {str(n): round(float((inked >= n).sum() * area), 2) for n in AT_LEAST}, "hotspots": hotspots}


def mark(tool: dict, given_m: float | None = None) -> dict:
    """{"width_mm", "from"} a resource's lines are counted at: `given_m`, else its tool's line width, else (width_mm
    None) each path's generation width (0.3 mm for the 3RL, whose mark on skin is not measured)."""
    if given_m:
        return {"width_mm": round(given_m * 1000, 6), "from": "given"}
    if tool["line_width_m"]:
        return {"width_mm": round(tool["line_width_m"] * 1000, 6), "from": f"{tool['id']} line.width_mm"}
    return {"width_mm": None, "from": "generation width"}


def marks(program: dict) -> dict:
    """{resource id: mark} each resource's lines are counted at, chosen at preparation (stats.passes.marks); a
    program prepared before is counted at its tool's line width, else the generation width."""
    return (program.get("stats") or {}).get("passes", {}).get("marks") or {row["id"]: mark(row["tool"])
                                                                           for row in program["resources"]}


def at_marks(program: dict) -> str:
    """What the passes are counted at, for a line: '0.5 mm', several '/'-joined, 'path-width' for the generation width."""
    return "/".join(sorted({f"{m['width_mm']:g} mm" if m["width_mm"] else "path-width" for m in marks(program).values()}))


def _widths(program: dict) -> dict:
    """Each stroke op's counted width: its resource's mark (marks), else the acquired generation width."""
    mark = {key: row["width_mm"] and row["width_mm"] / 1000 for key, row in marks(program).items()}
    return {op["id"]: mark[op["resource_id"]] or op["generation_width_m"] for op in program["ops"] if op["op"] == "stroke"}


def program_lines(program: dict):
    width = _widths(program)
    return [(op["points_m"] + ([op["points_m"][0]] if op["closed"] else []), width[op["id"]])
            for op in program["ops"] if op["op"] == "stroke"]


def drawn_lines(program: dict, drawn: dict):
    """The measured pen-down lines (op id -> lines, tatbot_contracts.ros_drawn.from_rows), each a running mean."""
    # FK of the encoders at 20 Hz steps the tip 0.1-0.4 mm back and forth, which would count as passes: a run's raw
    # lines were 1.5-2.4 times the plan's length, 1.05-1.15 times over SMOOTH samples.
    width, kernel, out = _widths(program), np.ones(SMOOTH) / SMOOTH, []
    for op, lines in drawn.items():
        for points in (np.asarray(line, float).reshape(-1, 2) for line in (lines if op in width else ())):
            if len(points) > SMOOTH:
                padded = np.pad(points, ((SMOOTH // 2, SMOOTH // 2), (0, 0)), mode="edge")
                points = np.c_[np.convolve(padded[:, 0], kernel, "valid"), np.convolve(padded[:, 1], kernel, "valid")]
            out.append((points, width[op]))
    return out
