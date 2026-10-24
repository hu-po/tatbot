"""Rigid placement of acquired metric paths on the page; no scaling or reordering."""
from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tatbot_contracts.artwork import canvas_m
from tatbot_motion.timelaw import arc_lengths

from tatbot_ink.errors import CompileError

STENCIL_SIZE_M = (0.128, 0.172)
STENCIL_CLEAR_M = (0.090, 0.134)
STENCIL_MARGIN_M = 0.005
# A path no wider than this is a dot: a DBV3 contour's sub-mm fragments lie on its other lines (a forearm
# self-portrait: 38 of its 103 paths, 24 mm).
MIN_STROKE_M = 0.001
# A path crossing a keep region's edge goes with the side holding most of its length, unless the other side holds more
# than this of it: a line of its own there (a DBV3 contour outline can trace a whole design), so it is cut at the edge.
KEEP_CUT_M = 0.020


@dataclass(frozen=True)
class Stroke:
    points_m: np.ndarray
    closed: bool
    generation_width_m: float
    src: dict
    arc0_m: float = 0.0     # where points_m starts along its source path: a path cut at a keep region's edge


def place_points(points, canvas, placement):
    if not np.allclose(placement["physical_scale_m"], canvas, rtol=0, atol=1e-10):
        raise CompileError("physical size changed; regenerate DBV3 artwork at the requested size")
    value = np.asarray(points, dtype=float) - np.asarray(canvas) / 2
    if placement["mirrored"]:
        value[:, 0] *= -1
    angle = placement["rotation_rad"]
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return value @ rotation.T + np.asarray(placement["target"]["anchor_uv_m"])


def relocate(design, *, at_m=None, width_m=None):
    if at_m is None and width_m is None:
        return design
    result = copy.deepcopy(design)
    anchors = {tuple(item["placement"]["target"]["anchor_uv_m"]) for item in result["placements"]}
    if len(anchors) != 1:   # layers on one origin move together; a composed design keeps its own layout
        raise CompileError("--at/--width require one placement, or layers that share one origin")
    for item in result["placements"]:
        size = canvas_m(result["artworks"][item["artwork_id"]])
        if width_m is not None and (not math.isfinite(width_m) or abs(width_m - size[0]) > 1e-10):
            raise CompileError("--width differs from the acquired canvas; regenerate DBV3 artwork at that size")
        if at_m is not None:
            if len(at_m) != 2 or not np.isfinite(at_m).all():
                raise CompileError("--at requires two finite page coordinates")
            item["placement"]["target"]["anchor_uv_m"] = list(at_m)
    return result


def page_of(target):
    size = list(target["canvas_m"])
    margin = target["margin_m"]
    if np.allclose(size, STENCIL_SIZE_M, rtol=0, atol=1e-9):
        return {"kind": "stencil", "size_m": list(STENCIL_SIZE_M), "clear_m": list(STENCIL_CLEAR_M),
                "margin_m": max(margin, STENCIL_MARGIN_M)}
    return {"kind": "plane", "size_m": size, "clear_m": [v - 2 * margin for v in size], "margin_m": margin}


def print_page(stencil, repo) -> dict:
    """The page of a generated stencil print (its directory, or its tracking.json or settings.json): its size and
    its clear centre, so a program placed on it passes the session's check against the installed print
    (tatbot_session.geometry.off_print). The margin from the page's edge is STENCIL_MARGIN_M, or the band about
    that clear centre where that is shallower: a small print with a thin band keeps all of its clear centre."""
    lib = str(Path(repo) / "scripts" / "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    import stencil_reference

    directory = Path(stencil).expanduser()
    directory = directory if directory.is_dir() else directory.parent
    try:
        manifest = json.loads((directory / "tracking.json").read_text())
        page = stencil_reference.page_geometry(manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CompileError(f"--stencil {stencil}: not a generated stencil print ({error})") from error
    clear = page["clear_m"]
    band = min((size - inner) / 2 for size, inner in zip(page["size_m"], clear, strict=True))
    return {"kind": "stencil", "size_m": page["size_m"], "clear_m": [round(v, 9) for v in clear],
            "margin_m": round(min(STENCIL_MARGIN_M, band), 9)}


def _pen_width(tool_width, tool_widths, key):
    if tool_widths is None:
        return tool_width
    if key not in tool_widths:
        raise CompileError('every acquired pen requires its tool width for placement')
    return tool_widths[key]


def _covered(points, raster):
    counts, origin, cell = raster or (np.zeros((0, 0)), 0, 1)
    col, row = np.floor((points - origin) / cell).astype(int).T
    inside = (col >= 0) & (col < counts.shape[1]) & (row >= 0) & (row < counts.shape[0])
    return inside.all() and (counts[row[inside], col[inside]] > 0).all()


def drop_dots(strokes, width, ink, min_size_m=MIN_STROKE_M):
    """(kept, dropped, dots kept): a path whose points all lie within `min_size_m` of each other, and within a pen
    width (`width(stroke)`) of a larger path's line of its own ink (`ink(stroke)`: its resource), is a dot over ink
    the program lays anyway."""
    # Each is a descent and a lift (~10 s on the arm) and a repass for nothing. A dot on bare skin is the design's,
    # and so is one on another ink's line: dropping it would lose that colour there.
    from tatbot_ink import passes

    small = {i for i, s in enumerate(strokes) if np.ptp(s.points_m, axis=0).max() < min_size_m
             and np.linalg.norm(s.points_m[:, None] - s.points_m[None], axis=-1).max() < min_size_m}
    rasters = {key: passes.grid([(s.points_m, 2 * width(s)) for i, s in enumerate(strokes)
                                 if i not in small and ink(s) == key]) for key in {ink(strokes[i]) for i in small}}
    dropped = {i for i in small if _covered(strokes[i].points_m, rasters[ink(strokes[i])])}
    return [s for i, s in enumerate(strokes) if i not in dropped], [strokes[i] for i in sorted(dropped)], len(small) - len(dropped)


def _inside(points, polygon):
    """Even-odd: which of `points` lie inside `polygon` (page metres)."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    x, y = points[:, :1], points[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        at = a[:, 0] + (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / (b[:, 1] - a[:, 1])
    return (((a[:, 1] > y) != (b[:, 1] > y)) & (x < at)).sum(axis=1) % 2 == 1


def _runs(points, polygon):
    """[(points, inside, start arc)] of a polyline cut where it crosses the polygon's edge; every cut lies on the
    segment it cuts, so no point leaves the line."""
    if len(points) < 2:
        return [(points, bool(_inside(points, polygon)[0]), 0.0)]
    p, q = points[:-1, None], points[1:, None]
    a, b = polygon[None], np.roll(polygon, -1, axis=0)[None]
    d, e = q - p, b - a
    cross = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((a - p)[..., 0] * e[..., 1] - (a - p)[..., 1] * e[..., 0]) / cross
        u = ((a - p)[..., 0] * d[..., 1] - (a - p)[..., 1] * d[..., 0]) / cross
    hit = (cross != 0) & (t > 1e-12) & (t < 1 - 1e-12) & (u >= 0) & (u <= 1)
    pieces = [[points[i]] + [points[i] + s * (points[i + 1] - points[i]) for s in np.unique(t[i][hit[i]])]
              for i in range(len(points) - 1)]
    dense = np.array([v for piece in pieces for v in piece] + [points[-1]])
    side = _inside((dense[:-1] + dense[1:]) / 2, polygon)
    arc = arc_lengths(dense)
    edges = np.flatnonzero(np.diff(side.astype(int))) + 1
    return [(dense[lo:hi + 1], bool(side[lo]), float(arc[lo])) for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(side)], strict=True)]


def keep_region(strokes, polygon_m, *, cut_m=KEEP_CUT_M, min_size_m=MIN_STROKE_M):
    """(the strokes kept, their record): the paths inside `polygon_m` (page metres). A path crossing its edge is kept
    whole or dropped by where most of its length lies, unless the other side holds more than `cut_m` of it: then it
    is cut at the edge and its runs inside kept, those no wider than `min_size_m` dropped. Nothing is moved."""
    polygon = np.asarray(polygon_m, float)
    if polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise CompileError("a keep region is a polygon of three or more finite page points")
    kept, crossing, counts = [], [], {"inside": 0, "outside": 0}
    for stroke in strokes:
        runs = _runs(stroke.points_m, polygon)
        if len(runs) == 1:
            counts["inside" if runs[0][1] else "outside"] += 1
            kept += [stroke] if runs[0][1] else []
            continue
        lengths = [(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), inside) for pts, inside, _ in runs]
        inside_m, length_m = sum(n for n, inside in lengths if inside), sum(n for n, _ in lengths)
        decision = "cut" if min(inside_m, length_m - inside_m) > cut_m else "kept" if inside_m >= length_m / 2 else "dropped"
        pieces = [stroke] if decision == "kept" else [] if decision == "dropped" else [
            Stroke(pts, False, stroke.generation_width_m, stroke.src, stroke.arc0_m + arc0) for pts, inside, arc0 in runs
            if inside and np.ptp(pts, axis=0).max() > min_size_m]
        kept += pieces
        crossing.append({**{k: stroke.src[k] for k in ("placement", "layer", "path")}, "inside_m": round(inside_m, 9),
                         "length_m": round(length_m, 9), "decision": decision, "pieces": len(pieces)})
    if not kept:
        raise CompileError("the keep region holds none of the design's paths")
    return kept, {"polygon_m": polygon.round(9).tolist(), "cut_m": cut_m, "paths": counts, "crossing": crossing}


def placed_strokes(design, tool_width=None, *, tool_widths=None, page=None):
    """Check each pen's own physical width, then place paths without editing them, inside `page` (print_page)
    when given, else the page each placement targets."""
    if tool_width is not None and tool_widths is not None:
        raise CompileError('use either one tool width or exact per-pen widths')
    strokes, chosen = [], page
    page = None
    for item in design["placements"]:
        placement = item["placement"]
        this_page = chosen or page_of(placement["target"])
        if page is not None and page != this_page:
            raise CompileError("all placements must share one page")
        page = this_page
        artwork = design["artworks"][item["artwork_id"]]
        size = canvas_m(artwork)
        corners = place_points([[0, 0], [size[0], 0], size, [0, size[1]]], size, placement)
        half = np.minimum(np.asarray(page["clear_m"]) / 2, np.asarray(page["size_m"]) / 2 - page["margin_m"])
        if np.any(np.abs(corners) > half + 1e-10):
            raise CompileError(f"placement {item['id']} canvas leaves the page clear area")
        for layer in artwork["program"]["layers"]:
            width = _pen_width(tool_width, tool_widths, (artwork['content_sha256'], layer['ink_id']))
            for path in layer["elements"]:
                points = place_points(path["points_m"], size, placement)
                radius = max(width or 0, path["width_m"]) / 2
                if np.any(np.abs(points) + radius > half + 1e-10):
                    raise CompileError(f"path {path['id']} footprint leaves the page clear area")
                if path["closed"] and not np.array_equal(points[0], points[-1]):
                    points = np.vstack([points, points[0]])
                strokes.append(Stroke(points, path["closed"], path["width_m"], {
                    "placement": item["id"], "artwork_sha256": artwork["content_sha256"],
                    "program_sha256": artwork["program"]["content_sha256"], "layer": layer["id"],
                    "path": path["id"], "pen": layer["ink_id"], "closed": path["closed"]}))
    return strokes, page
