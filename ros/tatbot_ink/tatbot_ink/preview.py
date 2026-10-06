"""preview.svg: the page and its clear centre, strokes numbered in order and coloured by ink, pen-up travel dashed."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


def _xy(point) -> str:
    # Page y points to the top of the print; SVG y points down.
    return f"{point[0] * 1000:.3f},{-point[1] * 1000:.3f}"


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
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-width / 2:g} {-height / 2 - 8:g} {width:g} {height + 8:g}" '
           f'width="{width:g}mm" height="{height + 8:g}mm" font-family="sans-serif">',
           f"<title>{escape(title)}</title>",
           f'<text x="{-width / 2 + 1:g}" y="{-height / 2 - 3:g}" font-size="2.6">{escape(title)}</text>',
           f'<rect x="{-width / 2:g}" y="{-height / 2:g}" width="{width:g}" height="{height:g}" fill="#fff" stroke="#999" stroke-width="0.3"/>',
           f'<rect x="{-clear_w / 2:g}" y="{-clear_h / 2:g}" width="{clear_w:g}" height="{clear_h:g}" fill="none" '
           f'stroke="#9ebde6" stroke-width="0.25" stroke-dasharray="1.5 1"/>']
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
