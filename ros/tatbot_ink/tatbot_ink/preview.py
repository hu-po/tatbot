"""preview.svg: the page and its clear centre, strokes numbered in order and coloured by ink, pen-up travel dashed,
and where the pen passes three or more times (tatbot_ink.passes) shaded under them."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from tatbot_ink import passes

HEAT_BLOCK_M = 0.0005     # the shading's squares, under the strokes: the most passes of any cell in each
HEAT = ("#f59f00", "#e8590c", "#c92a2a")   # 3, 4, and 5 or more passes


def _xy(point) -> str:
    # Page y points to the top of the print; SVG y points down.
    return f"{point[0] * 1000:.3f},{-point[1] * 1000:.3f}"


def _heat(program: dict) -> list[str]:
    counts, origin, cell = passes.grid(passes.program_lines(program)) or (np.zeros((0, 0)), 0, 1)
    b = max(1, round(HEAT_BLOCK_M / cell))
    counts = np.pad(counts, ((0, -counts.shape[0] % b), (0, -counts.shape[1] % b)))
    blocks = counts.reshape(counts.shape[0] // b, b, counts.shape[1] // b, b).max(axis=(1, 3))
    return ['<g id="passes" opacity="0.6">', *(
        f'<rect x="{(origin[0] + col * b * cell) * 1000:.3f}" y="{-(origin[1] + (row + 1) * b * cell) * 1000:.3f}" '
        f'width="{b * cell * 1000:g}" height="{b * cell * 1000:g}" fill="{HEAT[min(blocks[row, col], 5) - 3]}"/>'
        for row, col in np.argwhere(blocks >= 3)), "</g>"]


def preview_svg(program: dict) -> str:
    width, height = (v * 1000 for v in program["page"]["size_m"])
    clear_w, clear_h = (v * 1000 for v in program["page"]["clear_m"])
    resources = {resource['id']: resource for resource in program['resources']}
    stats = program["stats"]
    title = (f"{program['design']['name']}: {stats['strokes']} strokes, {stats['contact_m'] * 1000:.0f} mm contact, "
             f"{stats['travel_m'] * 1000:.0f} mm travel, {stats['tool_changes']} tool changes, {stats['dips']} dips, "
             f"~{stats['est_s']:.0f} s")
    if any(resource['tool']['line_width_m'] is None for resource in resources.values()):
        title += "; preview uses generation width (physical width unknown)"
    if "passes" in stats:
        title += (f"; passes at {passes.at_marks(program)} marks: max {stats['passes']['max']}, p95 {stats['passes']['p95']}, "
                  f"{stats['passes']['mm2_at_least']['3']:.0f} mm2 at 3+ (shaded orange 3, red 5+)")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-width / 2:g} {-height / 2 - 8:g} {width:g} {height + 8:g}" '
           f'width="{width:g}mm" height="{height + 8:g}mm" font-family="sans-serif">',
           f"<title>{escape(title)}</title>",
           f'<text x="{-width / 2 + 1:g}" y="{-height / 2 - 3:g}" font-size="2.6">{escape(title)}</text>',
           f'<rect x="{-width / 2:g}" y="{-height / 2:g}" width="{width:g}" height="{height:g}" fill="#fff" stroke="#999" stroke-width="0.3"/>',
           f'<rect x="{-clear_w / 2:g}" y="{-clear_h / 2:g}" width="{clear_w:g}" height="{clear_h:g}" fill="none" '
           f'stroke="#9ebde6" stroke-width="0.25" stroke-dasharray="1.5 1"/>', *_heat(program)]
    region = (program.get("preparation") or {}).get("keep_region")
    if region:   # the polygon the design was kept inside (tatbot_ink.place.keep_region)
        out.append(f'<polygon id="keep-region" points="{" ".join(_xy(p) for p in region["polygon_m"])}" fill="none" '
                   f'stroke="#2f9e44" stroke-width="0.3" stroke-dasharray="1 0.8"/>')
    previous = None
    number = 0
    for op in program["ops"]:
        if op["op"] != "stroke":
            continue
        points = op["points_m"]
        resource = resources[op['resource_id']]
        pen = (resource['tool']['line_width_m'] or op['generation_width_m']) * 1000
        if op["closed"] and points[0] != points[-1]:
            points = [*points, points[0]]
        if previous is not None and not op["continues"]:
            out.append(f'<polyline points="{_xy(previous)} {_xy(points[0])}" fill="none" stroke="#e08a2c" '
                       f'stroke-width="0.2" stroke-dasharray="0.8 0.6"/>')
        colour = '#{:02x}{:02x}{:02x}'.format(*resource['rgb']) if resource['rgb'] is not None else '#111'
        out.append(f'<polyline points="{" ".join(_xy(p) for p in points)}" fill="none" stroke="{colour}" '
                   f'stroke-width="{pen:g}" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>')
        if not op["continues"]:
            number += 1
            out.append(f'<text x="{points[0][0] * 1000:.3f}" y="{-points[0][1] * 1000 - 0.6:.3f}" font-size="1.8" '
                       f'fill="#c0392b">{number}</text>')
        previous = points[-1]
    out.append("</svg>")
    return "\n".join(out) + "\n"


def write_preview(program: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(preview_svg(program))
    return path
