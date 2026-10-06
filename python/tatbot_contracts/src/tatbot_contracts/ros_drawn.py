"""drawn.svg: the executed tip path (FK of the measured joints, pen-down samples) over the planned strokes.

Page frame metres in, SVG millimetres out: x right, the print's top at the top. Pure Python.
"""
from __future__ import annotations

from pathlib import Path


def _xy(point, size_m) -> tuple[float, float]:
    return (point[0] + size_m[0] / 2.0) * 1000.0, (size_m[1] / 2.0 - point[1]) * 1000.0


def _path(points, size_m) -> str:
    pts = [_xy(p, size_m) for p in points]
    return "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts)


def from_rows(rows, arm):
    """op id -> its measured pen-down lines, from every goal of the run (the ledger's done/aborted `line`)."""
    result = {}
    for row in rows:
        if row.get('arm') == arm and row.get('event') in ('done', 'aborted') and 'line' in row:
            result.setdefault(row['op'], []).append(row['line'])
    return result


def render(program: dict, drawn: dict[str, list[list]], *, size_m=(0.100, 0.150), clear_m=(0.062, 0.112),
           line_width_m: float | None = None) -> str:
    """SVG text. `drawn`: op id -> list of measured pen-down polylines, each [[x, y], ...] in page metres."""
    w, h = size_m[0] * 1000.0, size_m[1] * 1000.0
    cw, ch = clear_m[0] * 1000.0, clear_m[1] * 1000.0
    resources = {resource['id']: resource for resource in program.get('resources', [])}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.1f}mm" height="{h:.1f}mm" '
           f'viewBox="0 0 {w:.3f} {h:.3f}">',
           f'<rect x="0" y="0" width="{w:.3f}" height="{h:.3f}" fill="white" stroke="#999" stroke-width="0.2"/>',
           f'<rect x="{(w - cw) / 2:.3f}" y="{(h - ch) / 2:.3f}" width="{cw:.3f}" height="{ch:.3f}" '
           'fill="none" stroke="#ccc" stroke-width="0.2" stroke-dasharray="1 1"/>',
           '<g id="planned" fill="none" stroke="#9ab" stroke-linecap="round" stroke-linejoin="round" '
           'opacity="0.6">']
    for op in program.get("ops", []):
        if op.get("op") == "stroke" and len(op.get("points_m", [])) > 1:
            closed = " Z" if op.get("closed") else ""
            physical = resources[op['resource_id']]['tool']['line_width_m'] if resources else line_width_m
            width = (physical or op.get("generation_width_m", .0002)) * 1000
            out.append(f'<path stroke-width="{width:.3f}" id="plan-{op["id"]}" d="{_path(op["points_m"], size_m)}{closed}"/>')
    out.append("</g>")
    if any(row['tool']['line_width_m'] is None for row in resources.values()) if resources else line_width_m is None:
        out.append("<desc>Physical width unknown; planned paths use generation width when recorded, otherwise a 0.2 mm display line.</desc>")
    out.append('<g id="drawn" fill="none" stroke="#c22" stroke-linecap="round" stroke-linejoin="round" '
               'stroke-width="0.2">')
    for op_id, lines in drawn.items():
        for k, line in enumerate(lines):
            if len(line) > 1:
                out.append(f'<path id="drawn-{op_id}-{k}" d="{_path(line, size_m)}"/>')
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


def write(path: str | Path, program: dict, drawn: dict[str, list[list]], **kwargs) -> Path:
    path = Path(path)
    path.write_text(render(program, drawn, **kwargs), encoding="utf-8")
    return path
