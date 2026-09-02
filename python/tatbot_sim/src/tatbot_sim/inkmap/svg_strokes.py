"""Compile a bounded SVG subset into metric tattoo centreline strokes."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

_TOKEN = re.compile(r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


class SvgCompileError(ValueError):
    """A stable, user-facing reason an SVG cannot become robot strokes."""


@dataclass(frozen=True)
class MetricSvg:
    strokes: tuple[np.ndarray, ...]
    bounds_m: tuple[float, float, float, float]
    view_box: tuple[float, float, float, float]


def _line_distance(point, start, end) -> float:
    delta = end - start
    if np.dot(delta, delta) < 1e-20:
        return float(np.linalg.norm(point - start))
    offset = point - start
    return float(abs(delta[0] * offset[1] - delta[1] * offset[0]) / np.linalg.norm(delta))


def _flatten_cubic(p0, p1, p2, p3, tolerance, out, depth=0):
    if depth >= 20 or max(_line_distance(p1, p0, p3), _line_distance(p2, p0, p3)) <= tolerance:
        out.append(p3)
        return
    p01, p12, p23 = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p3) / 2
    p012, p123 = (p01 + p12) / 2, (p12 + p23) / 2
    mid = (p012 + p123) / 2
    _flatten_cubic(p0, p01, p012, mid, tolerance, out, depth + 1)
    _flatten_cubic(mid, p123, p23, p3, tolerance, out, depth + 1)


def _flatten_quadratic(p0, p1, p2, tolerance, out, depth=0):
    if depth >= 20 or _line_distance(p1, p0, p2) <= tolerance:
        out.append(p2)
        return
    p01, p12 = (p0 + p1) / 2, (p1 + p2) / 2
    mid = (p01 + p12) / 2
    _flatten_quadratic(p0, p01, mid, tolerance, out, depth + 1)
    _flatten_quadratic(mid, p12, p2, tolerance, out, depth + 1)


def _arc_points(start, end, rx, ry, rotation_deg, large, sweep, tolerance):
    """SVG endpoint arc conversion, following the SVG 1.1 implementation note."""
    rx, ry = abs(float(rx)), abs(float(ry))
    if rx < 1e-12 or ry < 1e-12 or np.allclose(start, end):
        return [end]
    phi = math.radians(rotation_deg % 360)
    c, s = math.cos(phi), math.sin(phi)
    delta = (start - end) / 2
    xp = c * delta[0] + s * delta[1]
    yp = -s * delta[0] + c * delta[1]
    lam = (xp / rx) ** 2 + (yp / ry) ** 2
    if lam > 1:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale
    numerator = max(0.0, rx * rx * ry * ry - rx * rx * yp * yp - ry * ry * xp * xp)
    denominator = max(rx * rx * yp * yp + ry * ry * xp * xp, 1e-30)
    sign = -1.0 if bool(large) == bool(sweep) else 1.0
    factor = sign * math.sqrt(numerator / denominator)
    cxp = factor * rx * yp / ry
    cyp = factor * -ry * xp / rx
    center = np.array([
        c * cxp - s * cyp + (start[0] + end[0]) / 2,
        s * cxp + c * cyp + (start[1] + end[1]) / 2,
    ])

    def angle(u, v):
        return math.atan2(u[0] * v[1] - u[1] * v[0], np.dot(u, v))

    u = np.array([(xp - cxp) / rx, (yp - cyp) / ry])
    v = np.array([(-xp - cxp) / rx, (-yp - cyp) / ry])
    theta = angle(np.array([1.0, 0.0]), u)
    extent = angle(u, v)
    if not sweep and extent > 0:
        extent -= 2 * math.pi
    elif sweep and extent < 0:
        extent += 2 * math.pi
    max_radius = max(rx, ry)
    max_step = 2 * math.acos(max(-1.0, min(1.0, 1 - tolerance / max_radius)))
    segments = max(1, int(math.ceil(abs(extent) / max(max_step, 1e-3))))
    points = []
    for i in range(1, segments + 1):
        t = theta + extent * i / segments
        ct, st = math.cos(t), math.sin(t)
        points.append(center + np.array([c * rx * ct - s * ry * st, s * rx * ct + c * ry * st]))
    points[-1] = end
    return points


def _path_strokes(data: str, tolerance: float) -> list[np.ndarray]:
    tokens = _TOKEN.findall(data.replace(",", " "))
    if not tokens:
        return []
    i = 0
    cmd = None
    current = np.zeros(2)
    start = np.zeros(2)
    previous_control = None
    strokes: list[list[np.ndarray]] = []
    stroke: list[np.ndarray] = []

    def number():
        nonlocal i
        if i >= len(tokens) or tokens[i].isalpha():
            raise SvgCompileError(f"path command {cmd}: missing numeric argument")
        value = float(tokens[i])
        i += 1
        return value

    def point(relative: bool):
        value = np.array([number(), number()], dtype=np.float64)
        return current + value if relative else value

    def begin(value):
        nonlocal stroke
        if len(stroke) >= 2:
            strokes.append(stroke)
        stroke = [value.copy()]

    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
        elif cmd is None:
            raise SvgCompileError("path data must begin with a command")
        assert cmd is not None
        relative = cmd.islower()
        op = cmd.upper()
        if op == "M":
            current = point(relative)
            start = current.copy()
            begin(current)
            cmd = "l" if relative else "L"
            previous_control = None
        elif op == "L":
            current = point(relative)
            stroke.append(current.copy())
            previous_control = None
        elif op == "H":
            current = current + [number(), 0] if relative else np.array([number(), current[1]])
            stroke.append(current.copy())
            previous_control = None
        elif op == "V":
            current = current + [0, number()] if relative else np.array([current[0], number()])
            stroke.append(current.copy())
            previous_control = None
        elif op == "C":
            p1, p2, end = point(relative), point(relative), point(relative)
            _flatten_cubic(current, p1, p2, end, tolerance, stroke)
            current, previous_control = end, p2
        elif op == "S":
            p1 = 2 * current - previous_control if previous_control is not None else current.copy()
            p2, end = point(relative), point(relative)
            _flatten_cubic(current, p1, p2, end, tolerance, stroke)
            current, previous_control = end, p2
        elif op == "Q":
            control, end = point(relative), point(relative)
            _flatten_quadratic(current, control, end, tolerance, stroke)
            current, previous_control = end, control
        elif op == "T":
            control = 2 * current - previous_control if previous_control is not None else current.copy()
            end = point(relative)
            _flatten_quadratic(current, control, end, tolerance, stroke)
            current, previous_control = end, control
        elif op == "A":
            rx, ry, rotation, large, sweep = number(), number(), number(), number(), number()
            end = point(relative)
            if large not in (0, 1) or sweep not in (0, 1):
                raise SvgCompileError("arc flags must be zero or one")
            stroke.extend(_arc_points(current, end, rx, ry, rotation, bool(large), bool(sweep), tolerance))
            current, previous_control = end, None
        elif op == "Z":
            if not np.allclose(current, start):
                stroke.append(start.copy())
            current = start.copy()
            previous_control = None
            cmd = None
        else:
            raise SvgCompileError(f"unsupported SVG path command {cmd!r}")
    if len(stroke) >= 2:
        strokes.append(stroke)
    return [np.asarray(values, dtype=np.float64) for values in strokes]


def _circle(cx, cy, rx, ry, tolerance):
    radius = max(rx, ry)
    max_step = 2 * math.acos(max(-1.0, min(1.0, 1 - tolerance / max(radius, 1e-12))))
    count = max(12, int(math.ceil(2 * math.pi / max(max_step, 1e-3))))
    theta = np.linspace(0, 2 * math.pi, count + 1)
    return np.stack([cx + rx * np.cos(theta), cy + ry * np.sin(theta)], axis=1)


def _shape_strokes(element: ET.Element, tolerance: float) -> list[np.ndarray]:
    tag = element.tag.rsplit("}", 1)[-1]
    if element.get("transform"):
        raise SvgCompileError(f"unsupported SVG transform on <{tag}>")

    def get(name: str, default: str = "") -> str:
        value = element.get(name)
        return default if value is None else value

    if tag in ("svg", "g", "defs", "title", "desc"):
        return []
    if tag == "path":
        return _path_strokes(get("d", ""), tolerance)
    if tag == "line":
        return [
            np.array(
                [[float(get("x1", "0")), float(get("y1", "0"))], [float(get("x2", "0")), float(get("y2", "0"))]]
            )
        ]
    if tag in ("polyline", "polygon"):
        values = [
            float(value)
            for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", get("points", ""))
        ]
        if len(values) < 4 or len(values) % 2:
            raise SvgCompileError(f"<{tag}> points must contain at least two xy pairs")
        points = np.asarray(values).reshape(-1, 2)
        if tag == "polygon" and not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        return [points]
    if tag in ("circle", "ellipse"):
        rx = float(get("r", get("rx", "0")))
        ry = float(get("r", get("ry", "0")))
        if rx <= 0 or ry <= 0:
            raise SvgCompileError(f"<{tag}> radii must be positive")
        return [_circle(float(get("cx", "0")), float(get("cy", "0")), rx, ry, tolerance)]
    if tag == "rect":
        x, y, width, height = (float(get(name, "0")) for name in ("x", "y", "width", "height"))
        if width <= 0 or height <= 0:
            raise SvgCompileError("<rect> dimensions must be positive")
        rx = min(float(get("rx", get("ry", "0"))), width / 2, height / 2)
        ry = min(float(get("ry", get("rx", "0"))), width / 2, height / 2)
        if rx <= 0 or ry <= 0:
            return [np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height], [x, y]])]
        pieces = [np.array([x + rx, y]), np.array([x + width - rx, y])]
        corners = [
            (x + width - rx, y + ry, -math.pi / 2, 0),
            (x + width - rx, y + height - ry, 0, math.pi / 2),
            (x + rx, y + height - ry, math.pi / 2, math.pi),
            (x + rx, y + ry, math.pi, 3 * math.pi / 2),
        ]
        points = [pieces[0], pieces[1]]
        count = max(3, int(math.ceil((math.pi / 2) / max(2 * math.acos(max(-1.0, 1 - tolerance / max(rx, ry))), 1e-3))))
        for index, (cx, cy, a0, a1) in enumerate(corners):
            for angle in np.linspace(a0, a1, count + 1)[1:]:
                points.append(np.array([cx + rx * math.cos(angle), cy + ry * math.sin(angle)]))
            if index < 3:
                next_corner = corners[index + 1]
                points.append(np.array([next_corner[0] + rx * math.cos(next_corner[2]), next_corner[1] + ry * math.sin(next_corner[2])]))
        points.append(points[0])
        return [np.asarray(points)]
    raise SvgCompileError(f"unsupported SVG element <{tag}>")


def compile_svg_strokes(
    svg: str,
    size_mm: tuple[float, float] | list[float],
    *,
    mirror: bool = False,
    rotation_rad: float = 0.0,
    chord_error_m: float = 1e-4,
) -> MetricSvg:
    """Return metric, centered strokes with SVG +Y mapped to body-frame +v."""
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise SvgCompileError(f"invalid SVG XML: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise SvgCompileError("document root must be <svg>")
    raw_view_box = root.get("viewBox")
    if not raw_view_box:
        raise SvgCompileError("SVG must declare a viewBox")
    values = [float(value) for value in re.split(r"[\s,]+", raw_view_box.strip())]
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise SvgCompileError("viewBox must be min-x min-y positive-width positive-height")
    min_x, min_y, view_w, view_h = values
    size = np.asarray(size_mm, dtype=np.float64)
    if size.shape != (2,) or np.any(size <= 0) or not np.isfinite(size).all():
        raise SvgCompileError("size_mm must contain two positive finite values")
    scale = size / 1000 / np.array([view_w, view_h])
    tolerance_units = chord_error_m / max(scale)
    raw_strokes = []
    for element in root.iter():
        raw_strokes.extend(_shape_strokes(element, tolerance_units))
    if not raw_strokes:
        raise SvgCompileError("SVG contains no supported drawable geometry")
    c, s = math.cos(rotation_rad), math.sin(rotation_rad)
    rotation = np.array([[c, -s], [s, c]])
    strokes = []
    for raw in raw_strokes:
        points = np.empty_like(raw)
        points[:, 0] = (raw[:, 0] - min_x) * scale[0] - size[0] / 2000
        points[:, 1] = size[1] / 2000 - (raw[:, 1] - min_y) * scale[1]
        if mirror:
            points[:, 0] *= -1
        points = points @ rotation.T
        keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12])
        points = points[keep]
        if len(points) >= 2:
            strokes.append(points.astype(np.float32))
    if not strokes:
        raise SvgCompileError("SVG drawable geometry collapsed to zero-length strokes")
    all_points = np.concatenate(strokes)
    bounds = (float(all_points[:, 0].min()), float(all_points[:, 1].min()),
              float(all_points[:, 0].max()), float(all_points[:, 1].max()))
    return MetricSvg(strokes=tuple(strokes), bounds_m=bounds, view_box=(min_x, min_y, view_w, view_h))
