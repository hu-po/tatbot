"""Ordered metric path subset of TattooProgram/1 used by drawing preparation.

No renderer, SVG parser, numerical package, or robot runtime is needed to read
frozen paths. Broader visual paint programs must be converted before reaching
this boundary. Order, IDs, direction and repeated passes are authoritative.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from datetime import datetime
from xml.sax.saxutils import quoteattr

from tatbot_contracts.canonical import canonical_digest

SCHEMA = "tatbot.tattoo-program/1"
MAX_POINTS = 1_000_000


def _keys(value: dict, required: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"expected object with fields {sorted(required)}")


def _text(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 100_000:
        raise ValueError("expected bounded nonempty text")


def _number(value: float, low: float, high: float, *, positive=False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("coordinate or physical parameter outside finite bounds")
    if positive and value <= 0:
        raise ValueError("physical parameter must be positive")


def _sha(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError("expected lowercase SHA-256")


def _unique(value: str, identifiers: set[str]) -> None:
    _text(value)
    if value in identifiers:
        raise ValueError(f"duplicate path/pen/layer identifier: {value}")
    identifiers.add(value)


def validate_geometry(geometry: dict) -> None:
    _keys(geometry, {"canvas_m", "inks", "layers", "negative_space_masks"})
    _keys(geometry["canvas_m"], {"width", "height"})
    size = [geometry["canvas_m"][key] for key in ("width", "height")]
    for dimension in size:
        _number(dimension, 0, 2, positive=True)
    if geometry["negative_space_masks"] != []:
        raise ValueError("frozen drawing paths cannot contain unresolved masks")
    inks = _inks(geometry["inks"])
    layers = geometry["layers"]
    if not isinstance(layers, list) or not 1 <= len(layers) <= 50_000:
        raise ValueError("expected 1..50000 ordered path layers")
    layer_ids, element_ids, point_count = set(), set(), 0
    for layer in layers:
        _keys(layer, {"id", "ink_id", "elements"})
        _unique(layer["id"], layer_ids)
        if layer["ink_id"] not in inks:
            raise ValueError("path layer references an unknown pen")
        point_count += _elements(layer["elements"], element_ids, size)
        if point_count > MAX_POINTS:
            raise ValueError("frozen paths exceed the point budget")


def _inks(inks: list) -> set[str]:
    if not isinstance(inks, list) or not 1 <= len(inks) <= 10_000:
        raise ValueError("expected 1..10000 pens")
    result = set()
    for ink in inks:
        _keys(ink, {"id", "color_srgb"})
        _unique(ink["id"], result)
        rgb = ink["color_srgb"]
        if not isinstance(rgb, list) or len(rgb) != 3:
            raise ValueError("expected three sRGB components")
        for component in rgb:
            _number(component, 0, 1)
    return result


def _elements(elements: list, identifiers: set[str], size: list[float]) -> int:
    if not isinstance(elements, list) or not 1 <= len(elements) <= 50_000:
        raise ValueError("expected 1..50000 ordered paths")
    count = 0
    for element in elements:
        _keys(element, {"id", "kind", "closed", "fill", "width_m", "deposition", "points_m"})
        _unique(element["id"], identifiers)
        if element["kind"] != "path" or element["fill"] is not False or type(element["closed"]) is not bool:
            raise ValueError("drawing requires frozen, unfilled paths with explicit closedness")
        _number(element["width_m"], 0, .02, positive=True)
        _number(element["deposition"], 0, 1, positive=True)
        count += _points(element["points_m"], size)
    return count


def _points(points: list, size: list[float]) -> int:
    if not isinstance(points, list) or not 2 <= len(points) <= MAX_POINTS:
        raise ValueError("a drawing path requires 2..1000000 points")
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("expected metric xy coordinates")
        for value, maximum in zip(point, size, strict=True):
            _number(value, -1e-10, maximum + 1e-10)
    if not any(p != points[0] for p in points[1:]):
        raise ValueError("point deposition is not a drawable centerline")
    return len(points)


def freeze_program(geometry: dict, *, source_sha256: str, name: str, adapter: str) -> tuple[dict, str]:
    """Acquire once; identity binds the saved path bytes, not another conversion."""
    _sha(source_sha256)
    _text(name)
    _text(adapter)
    preview = render_paths(geometry)
    program = {"schema": SCHEMA, **copy.deepcopy(geometry), "semantic_intent": name,
               "preview_sha256": hashlib.sha256(preview.encode()).hexdigest(),
               "provenance": {"producer": adapter, "version": "1", "created_utc": "1970-01-01T00:00:00Z",
                              "source_sha256": source_sha256}}
    program["content_sha256"] = canonical_digest(program)
    return program, preview


def validate_path_program(value: dict) -> dict:
    _keys(value, {"schema", "content_sha256", "canvas_m", "inks", "layers", "negative_space_masks",
                  "semantic_intent", "preview_sha256", "provenance"})
    if value["schema"] != SCHEMA:
        raise ValueError("unsupported frozen path program")
    geometry = {key: value[key] for key in ("canvas_m", "inks", "layers", "negative_space_masks")}
    validate_geometry(geometry)
    _text(value["semantic_intent"])
    _sha(value["preview_sha256"])
    _keys(value["provenance"], {"producer", "version", "created_utc", "source_sha256"})
    _sha(value["provenance"]["source_sha256"])
    for key in ("producer", "version", "created_utc"):
        _text(value["provenance"][key])
    timestamp = value["provenance"]["created_utc"]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", timestamp):
        raise ValueError("path provenance requires an RFC 3339 UTC timestamp")
    datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    if value["content_sha256"] != canonical_digest(value):
        raise ValueError("frozen path digest mismatch")
    return copy.deepcopy(value)


def render_paths(geometry: dict) -> str:
    """Derived metric review image; changes no physical width when zoomed."""
    validate_geometry(geometry)
    width, height = [geometry["canvas_m"][key] for key in ("width", "height")]
    inks = {ink["id"]: ink["color_srgb"] for ink in geometry["inks"]}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.14g} {height:.14g}" '
             f'width="{width*1000:.14g}mm" height="{height*1000:.14g}mm">']
    for layer in geometry["layers"]:
        rgb = ','.join(f'{v*255:.14g}' for v in inks[layer["ink_id"]])
        for path in layer["elements"]:
            points = " ".join(f"{x:.14g},{height-y:.14g}" for x, y in path["points_m"])
            tag = "polygon" if path["closed"] else "polyline"
            parts.append(f'<{tag} id={quoteattr(path["id"])} points="{points}" '
                         f'fill="none" stroke="rgb({rgb})" stroke-width="{path["width_m"]:.14g}" '
                         'stroke-linecap="round" stroke-linejoin="round"/>')
    return ''.join([*parts, '</svg>'])
