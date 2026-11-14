"""How far a binary STL reaches: its triangles, points moved into another frame, their bounds, and the mesh cut into
slabs along one axis with each slab's own bounds, so a part that is not a box is enclosed by a few. Stdlib only: the
URDF generators run where numpy may not be."""
from __future__ import annotations

import struct
from pathlib import Path

Point = tuple[float, float, float]
Bounds = tuple[Point, Point]


def triangles(path: Path, scale: float = 1.0) -> list[tuple[Point, Point, Point]]:
    """A binary STL's triangles, each vertex times `scale` (0.001 for a mesh in mm)."""
    data = Path(path).read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"{path} is not a binary STL: {len(data)} bytes for {count} triangles")
    return [tuple(tuple(v * scale for v in facet[i:i + 3]) for i in (3, 6, 9))
            for facet in struct.iter_unpack("<12f2x", data[84:])]


def moved(frame, points) -> list[Point]:
    """`points` through a 4x4 `frame` (nested rows)."""
    return [tuple(sum(frame[i][k] * p[k] for k in range(3)) + frame[i][3] for i in range(3)) for p in points]


def bounds(points) -> Bounds:
    """The least and greatest corner of `points`."""
    return tuple(min(p[i] for p in points) for i in range(3)), tuple(max(p[i] for p in points) for i in range(3))


def _cut(polygon: list[Point], axis: int, at: float, above: bool) -> list[Point]:
    """The part of a convex polygon on one side of the plane `axis` = `at` (Sutherland-Hodgman)."""
    out = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        a_in, b_in = (a[axis] >= at) == above or a[axis] == at, (b[axis] >= at) == above or b[axis] == at
        if a_in:
            out.append(a)
        if a_in != b_in:
            s = (at - a[axis]) / (b[axis] - a[axis])
            out.append(tuple(a[i] + s * (b[i] - a[i]) for i in range(3)))
    return out


def slabs(mesh, axis: int, count: int) -> list[Bounds]:
    """The bounds of `mesh` (triangles) within each of `count` equal slabs along `axis`, the empty ones left out: each
    triangle is cut at the slab's faces, so a slab's bounds hold all of the surface that lies in it."""
    low, high = (bound[axis] for bound in bounds([p for triangle in mesh for p in triangle]))
    out = []
    for k in range(count):
        lo, hi = low + (high - low) * k / count, low + (high - low) * (k + 1) / count
        inside = [p for triangle in mesh for p in _cut(_cut(list(triangle), axis, lo, True), axis, hi, False)]
        if inside:
            out.append(bounds(inside))
    return out
