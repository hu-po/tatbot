"""Version-bounded DBV3 1.6.22 export decoder, independent of browser/ROS/sim.

SVG is a private transport, not an artwork authority. Accept only admitted
Batik constructs. Return ordered TattooProgram layer/element structures for
freezing in the shared contract; refuse unsupported paint and geometry.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from drawingbot.artifacts import MAX_SVG_BYTES, SVG_NS
from drawingbot.native_path import Budget, decode_path, numbers

ADAPTER = "dbv3-batik-paths/1"
IDENTITY = (1., 0., 0., 1., 0., 0.)
ALLOWED = {"svg": {"width", "height", "viewBox"}, "g": {"id", "style", "transform"},
           "path": {"id", "style", "transform", "d"}}
STYLE = {"fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "stroke-miterlimit"}


def _matrix(text: str | None) -> tuple:
    if text is None:
        return IDENTITY
    match = re.fullmatch(r"matrix\(([^()]+)\)", text)
    if match is None:
        raise ValueError("native export supports only a single matrix transform")
    value = tuple(numbers(match[1], 6))
    if abs(value[0] * value[3] - value[1] * value[2]) < 1e-20:
        raise ValueError("singular native transform")
    return value


def _compose(a: tuple, b: tuple) -> tuple:
    return (a[0]*b[0] + a[2]*b[1], a[1]*b[0] + a[3]*b[1],
            a[0]*b[2] + a[2]*b[3], a[1]*b[2] + a[3]*b[3],
            a[0]*b[4] + a[2]*b[5] + a[4], a[1]*b[4] + a[3]*b[5] + a[5])


def _style(parent: dict, node: ET.Element) -> dict:
    result = dict(parent)
    for declaration in node.get("style", "").split(";"):
        if not declaration.strip():
            continue
        pair = [part.strip() for part in declaration.split(":")]
        if len(pair) != 2 or pair[0] not in STYLE or not pair[1]:
            raise ValueError(f"unsupported native paint: {declaration}")
        result[pair[0]] = pair[1]
    return result


def _rgb(value: str) -> list[float]:
    if value == "black":
        return [0., 0., 0.]
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return [int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    match = re.fullmatch(r"rgb\(([^()]+)\)", value)
    if match:
        channels = numbers(match[1], 3)
        if all(0 <= value <= 255 for value in channels):
            return [value / 255 for value in channels]
    raise ValueError("unsupported native stroke color")


def _check_node(node: ET.Element, depth: int) -> str:
    tag = node.tag.removeprefix(f"{{{SVG_NS}}}")
    if node.tag != f"{{{SVG_NS}}}{tag}" or tag not in ALLOWED or depth > 32:
        raise ValueError("unsupported native element or excessive nesting")
    if set(node.attrib) - ALLOWED[tag] or (tag == "svg" and depth != 0):
        raise ValueError("unsupported native attribute or nested canvas")
    if (node.text or "").strip() or (node.tail or "").strip():
        raise ValueError("native export contains text")
    if tag == "path" and len(node):
        raise ValueError("native path contains children")
    return tag


def _canvas(root: ET.Element) -> tuple[list[float], list[float]]:
    from drawingbot.artifacts import _millimetres

    size = [_millimetres(root.get(key, "")) / 1000 for key in ("width", "height")]
    box = numbers(root.get("viewBox", ""), 4)
    expected = [0., 0., size[0] * 1000 * 96 / 25.4, size[1] * 1000 * 96 / 25.4]
    if any(abs(a - b) > 1e-8 * max(1., abs(b)) for a, b in zip(box, expected, strict=True)):
        raise ValueError("native canvas must retain its physical CSS-pixel viewBox")
    return size, box


def _paint(style: dict, matrix: tuple) -> tuple[list[float], float]:
    if style.get("fill") != "none" or style.get("stroke-linecap") != "round" or style.get("stroke-linejoin") != "round":
        raise ValueError("native path requires unfilled centerlines with round caps and joins")
    if "stroke-miterlimit" in style:
        numbers(style["stroke-miterlimit"], 1)
    width = numbers(style.get("stroke-width", ""), 1)[0]
    sx, sy = math.hypot(*matrix[:2]), math.hypot(*matrix[2:4])
    if not math.isclose(sx, sy, rel_tol=1e-8) or abs(matrix[0]*matrix[2] + matrix[1]*matrix[3]) > sx*sy*1e-8:
        raise ValueError("native pen transform is not uniform; physical width is ambiguous")
    width_m = width * sx * .0254 / 96
    if not 0 < width_m <= .02:
        raise ValueError("native display width must be in (0, 20 mm]")
    return _rgb(style.get("stroke", "")), width_m


@dataclass
class Decoder:
    size: list[float]
    tolerance: float
    pen_width_m: float
    budget: Budget = field(default_factory=Budget)
    pens: dict[str, dict] | None = None
    layers: list[dict] = field(default_factory=list)
    inks: list[dict] = field(default_factory=list)
    widths: list[dict] = field(default_factory=list)
    path_count: int = 0
    node_count: int = 0
    pen_groups: int = 0
    seen_groups: set[str] = field(default_factory=set)

    def walk(self, node: ET.Element, *, depth=0, matrix=IDENTITY, style=None, pen=None) -> None:
        self.node_count += 1
        if self.node_count > 50_000:
            raise ValueError("native export exceeds node budget")
        tag = _check_node(node, depth)
        transform = _compose(matrix, _matrix(node.get("transform")))
        paint = _style(style or {}, node)
        if tag == "g" and depth == 1:
            self.pen_groups += 1
            group = node.get("id", "")
            if not group:
                raise ValueError("native top-level pen group requires its name")
            if group in self.seen_groups:
                raise ValueError("native export repeats an ambiguous pen group")
            self.seen_groups.add(group)
            if self.pens is not None:
                if group not in self.pens or not self.pens[group]["enabled"]:
                    raise ValueError("native export group has no enabled readback pen")
                pen = (self.pens[group]["id"], group)
            else:
                pen = (f"pen-{self.pen_groups}", group)
        if tag == "path":
            self.path(node, transform, paint, pen)
        for child in node:
            self.walk(child, depth=depth + 1, matrix=transform, style=paint, pen=pen)

    def path(self, node: ET.Element, matrix: tuple, style: dict, pen: tuple | None) -> None:
        if pen is None:
            raise ValueError("native path has no pen group")
        rgb, display_m = _paint(style, matrix)
        generation_width_m = self._pen(pen, rgb, display_m)
        self.widths[-1]["native_path_id"] = node.get("id")
        self.path_count += 1
        paths = decode_path(node.get("d", ""), lambda p: self.metric(p, matrix), self.tolerance, self.budget)
        if not self.layers or self.layers[-1]["ink_id"] != pen[0]:
            self.layers.append({"id": f"layer-{len(self.layers) + 1}", "ink_id": pen[0], "elements": []})
        for index, path in enumerate(paths):
            self._bounds(path["points_m"])
            self.layers[-1]["elements"].append({"id": f"path-{self.path_count}-{index + 1}",
                "kind": "path", "fill": False, "width_m": generation_width_m, "deposition": 1., **path})

    def _pen(self, pen: tuple, rgb: list[float], width: float) -> float:
        ink = {"id": pen[0], "color_srgb": rgb}
        held = next((item for item in self.inks if item["id"] == pen[0]), None)
        if held is not None and held != ink:
            raise ValueError("native pen group changes stroke color")
        if held is None:
            self.inks.append(ink)
        generation_width_m = self.pen_width_m
        if self.pens is not None:
            expected = self.pens[pen[1]]
            generation_width_m = expected["generation_width_m"]
            if any(abs(a - b/255) > 1e-8 for a, b in zip(rgb, expected["rgba"][:3], strict=True)):
                raise ValueError("native stroke color differs from pen readback")
        if not math.isclose(width, generation_width_m, rel_tol=.01, abs_tol=1e-7):
            raise ValueError("native pen display width differs from recorded generation width")
        self.widths.append({"path": self.path_count + 1, "pen_id": pen[0], "pen_name": pen[1],
                            "export_width_m": width, "generation_width_m": generation_width_m})
        return generation_width_m

    def metric(self, point: tuple, matrix: tuple) -> tuple:
        x = matrix[0]*point[0] + matrix[2]*point[1] + matrix[4]
        y = matrix[1]*point[0] + matrix[3]*point[1] + matrix[5]
        result = (x * .0254 / 96, self.size[1] - y * .0254 / 96)
        if not all(math.isfinite(v) and abs(v) <= 20 for v in result):
            raise ValueError("native coordinates exceed numeric bounds")
        return tuple(0. if v == 0 else v for v in result)

    def _bounds(self, points: list) -> None:
        if any(v < -1e-10 or v > self.size[i] + 1e-10 for p in points for i, v in enumerate(p)):
            raise ValueError("native path leaves its physical canvas; regenerate with clipping")


def decode_export(svg: str, *, chord_error_m: float, pen_width_m: float, max_points: int = 1_000_000,
                  pens: list[dict] | None = None) -> tuple[dict, dict]:
    """Return shared program geometry plus decoder audit data, in export order."""
    if not math.isfinite(chord_error_m) or not 0 < chord_error_m <= .0001:
        raise ValueError("native chord error must be in (0, 0.1 mm]")
    if not math.isfinite(pen_width_m) or not 0 < pen_width_m <= .02:
        raise ValueError("generation width must be in (0, 20 mm]")
    if len(svg.encode()) > MAX_SVG_BYTES or re.search(r"<!|<\?", svg):
        raise ValueError("native decoder expects bounded normalized XML without declarations")
    root = ET.fromstring(svg)
    if root.tag != f"{{{SVG_NS}}}svg":
        raise ValueError("native export requires an SVG root")
    size, _ = _canvas(root)
    table = _pen_table(pens) if pens is not None else None
    decoder = Decoder(size, chord_error_m, pen_width_m, Budget(max_points), table)
    decoder.walk(root)
    if not decoder.layers:
        raise ValueError("native export has no drawable paths")
    geometry = {"canvas_m": {"width": size[0], "height": size[1]}, "inks": decoder.inks,
                "layers": decoder.layers, "negative_space_masks": []}
    audit = {"adapter": ADAPTER, "chord_error_m": chord_error_m, "raw_path_count": decoder.path_count,
             "decoded_points": max_points - decoder.budget.remaining, "pen_widths": decoder.widths,
             "export_width_relative_tolerance": .01}
    if table is not None:
        audit["pens"] = [{**row, "exported": group in decoder.seen_groups,
                          "raw_path_count": sum(w["pen_id"] == row["id"] for w in decoder.widths)}
                         for group, row in table.items()]
    return geometry, audit


def _pen_table(pens: list[dict]) -> dict[str, dict]:
    table, ids = {}, set()
    if not isinstance(pens, list) or not pens:
        raise ValueError("native decoder requires a nonempty pen readback table")
    for row in pens:
        group, identity, width = row.get("export_group"), row.get("id"), row.get("generation_width_m")
        if not isinstance(group, str) or not group or group in table or not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("native pen readback identity is ambiguous")
        if type(width) not in (int, float) or not math.isfinite(width) or not 0 < width <= .02:
            raise ValueError("native pen readback width is invalid")
        rgba = row.get("rgba")
        if not isinstance(rgba, list) or len(rgba) != 4 or any(type(v) is not int or not 0 <= v <= 255 for v in rgba) or rgba[3] != 255:
            raise ValueError("native pen readback requires opaque 8-bit RGBA")
        if type(row.get("enabled")) is not bool:
            raise ValueError("native pen readback enabled state is invalid")
        table[group] = row
        ids.add(identity)
    return table
