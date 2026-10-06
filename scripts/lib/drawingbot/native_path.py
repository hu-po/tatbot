"""Decode DBV3/Batik absolute M/L/Q/C/Z traversal with a metric error bound.

This is not an arbitrary SVG importer. Unsupported commands fail acquisition.
Subdivision bounds distance to the finite chord, so collinear backtracking is
not lost. De Casteljau preserves direction; no sorting or deduplication occurs.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
TOKEN = re.compile(r"[MLQCZ]|" + NUMBER)
COUNTS = {"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0}
Point = tuple[float, float]


def numbers(text: str, count: int) -> list[float]:
    if re.sub(NUMBER, "", text).strip(" \t\r\n,"):
        raise ValueError("unsupported native numeric expression")
    result = [float(value) for value in re.findall(NUMBER, text)]
    if len(result) != count or not all(math.isfinite(value) for value in result):
        raise ValueError(f"expected {count} finite numbers")
    return result


def commands(text: str):
    if TOKEN.sub("", text).strip(" \t\r\n,"):
        raise ValueError("unsupported native path command or number")
    tokens = TOKEN.findall(text)
    if not tokens or tokens[0] != "M":
        raise ValueError("native path must begin with M")
    index, command = 0, None
    while index < len(tokens):
        if tokens[index] in COUNTS:
            command = tokens[index]
            index += 1
        elif command is None:
            raise ValueError("missing native path command")
        count = COUNTS[command]
        args = tokens[index:index + count]
        if len(args) != count or any(value in COUNTS for value in args):
            raise ValueError(f"incomplete native {command} command")
        values = [float(value) for value in args]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite native path coordinate")
        yield command, list(zip(values[::2], values[1::2], strict=True))
        index += count
        command = "L" if command == "M" else None if command == "Z" else command


def distance_to_segment(point: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0 if length2 == 0 else max(0, min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2))
    return math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy)


def _split(points: list[Point]) -> tuple[list[Point], list[Point]]:
    levels = [points]
    while len(levels[-1]) > 1:
        level = levels[-1]
        levels.append([((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(level, level[1:], strict=False)])
    return [level[0] for level in levels], [level[-1] for level in reversed(levels)]


@dataclass
class Budget:
    """One point budget across the entire export, including all subpaths."""
    remaining: int = 1_000_000

    def consume(self, count=1):
        self.remaining -= count
        if self.remaining < 0:
            raise ValueError("native export exceeds the decoded point budget")


def _flat(controls: list[Point], tolerance: float) -> bool:
    # Distance alone loses an internal collinear reversal. The control polygon
    # also bounds arc length; subdivide until its excess over the chord is small.
    distance = max(distance_to_segment(p, controls[0], controls[-1]) for p in controls[1:-1])
    polygon = sum(math.dist(a, b) for a, b in zip(controls, controls[1:], strict=False))
    return distance <= tolerance and polygon - math.dist(controls[0], controls[-1]) <= tolerance


def flatten(points: list[Point], tolerance: float, budget: Budget) -> list[Point]:
    """Return curve vertices after its start; fail rather than relax the bound."""
    result, stack = [], [(points, 0)]
    while stack:
        controls, depth = stack.pop()
        if _flat(controls, tolerance):
            budget.consume()
            result.append(controls[-1])
            continue
        if depth == 30:
            raise ValueError("native curve exceeds subdivision depth at the requested error bound")
        left, right = _split(controls)
        stack.extend(((right, depth + 1), (left, depth + 1)))
    return result


def decode_path(text: str, metric, tolerance: float, budget: Budget) -> list[dict]:
    paths, points, closed = [], [], False
    for command, args in commands(text):
        transformed = [metric(point) for point in args]
        if command == "M":
            _finish(paths, points, closed)
            points, closed = transformed, False
            budget.consume()
        elif command == "Z":
            if closed or len(points) < 2:
                raise ValueError("native closure requires an open drawable subpath")
            closed = True
        else:
            if closed:
                raise ValueError("native command after Z requires a new M")
            _extend(points, transformed, command, tolerance, budget)
    _finish(paths, points, closed)
    return paths


def _extend(points: list, transformed: list, command: str, tolerance: float, budget: Budget) -> None:
    if command == "L":
        budget.consume()
        points.extend(transformed)
    else:
        points.extend(flatten([points[-1], *transformed], tolerance, budget))


def _finish(paths: list, points: list, closed: bool) -> None:
    if not points:
        return
    if len(points) < 2 or not any(point != points[0] for point in points[1:]):
        raise ValueError("native path is a point; point deposition requires an explicit operation")
    paths.append({"points_m": [list(point) for point in points], "closed": closed})
