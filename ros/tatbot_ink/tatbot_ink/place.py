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

from tatbot_ink.errors import CompileError

STENCIL_SIZE_M = (0.100, 0.150)
STENCIL_CLEAR_M = (0.062, 0.112)
STENCIL_MARGIN_M = 0.005


@dataclass(frozen=True)
class Stroke:
    points_m: np.ndarray
    closed: bool
    generation_width_m: float
    src: dict


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
    if len(design["placements"]) != 1:
        raise CompileError("--at/--width require exactly one placement")
    result = copy.deepcopy(design)
    item = result["placements"][0]
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
    the largest clear centre about the page centre inside its border's innermost ink, so a program placed on it
    passes the session's check against the installed print (tatbot_session.geometry.off_print)."""
    lib = str(Path(repo) / "scripts" / "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    import stencil_reference

    directory = Path(stencil).expanduser()
    directory = directory if directory.is_dir() else directory.parent
    try:
        manifest = json.loads((directory / "tracking.json").read_text())
        settings_path = directory / stencil_reference.SETTINGS_FILE
        page = stencil_reference.page_geometry(manifest, json.loads(settings_path.read_text())
                                               if settings_path.is_file() else None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CompileError(f"--stencil {stencil}: not a generated stencil print ({error})") from error
    clear = page["clear_m"]
    if "inner_edges_m" in page:
        edges = page["inner_edges_m"]
        clear = [2 * min(-edges["left"], edges["right"]), 2 * min(-edges["bottom"], edges["top"])]
    return {"kind": "stencil", "size_m": page["size_m"], "clear_m": [round(v, 9) for v in clear],
            "margin_m": STENCIL_MARGIN_M}


def _pen_width(tool_width, tool_widths, key):
    if tool_widths is None:
        return tool_width
    if key not in tool_widths:
        raise CompileError('every acquired pen requires its tool width for placement')
    return tool_widths[key]


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
