"""Explicit logical pens and the pinned native drawing-set transport.

Native names carry logical IDs so the ordinary SVG exporter cannot collapse
identity into a display name or color. Human names stay in the requested table.
"""
from __future__ import annotations

import copy
import math
import re


def validate_drawing_set(value: dict, base_width_mm: float) -> None:
    fields = {"name", "type", "distribution_order", "distribution_type", "color_separation", "pens"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"drawing_set requires {sorted(fields)}")
    for key in fields - {"pens"}:
        if not isinstance(value[key], str) or not 1 <= len(value[key]) <= 200:
            raise ValueError(f"drawing_set.{key} requires bounded text")
    if value["color_separation"] != "Default":
        raise ValueError("only fixed-color pens with Default color separation are supported")
    pens = value["pens"]
    if not isinstance(pens, list) or not 1 <= len(pens) <= 100:
        raise ValueError("drawing_set requires 1..100 pens")
    identifiers = set()
    for pen in pens:
        _validate_pen(pen, base_width_mm)
        if pen["id"] in identifiers:
            raise ValueError("logical pen IDs must be unique")
        identifiers.add(pen["id"])
    if not any(pen["enabled"] and pen["weight"] > 0 for pen in pens):
        raise ValueError("drawing_set needs an enabled pen with positive weight")


def _validate_pen(pen, base_width_mm):
    fields = {"id", "name", "type", "enabled", "rgba", "weight", "stroke_factor"}
    if not isinstance(pen, dict) or set(pen) != fields:
        raise ValueError(f"pen requires {sorted(fields)}")
    if not isinstance(pen["id"], str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,79}", pen["id"]):
        raise ValueError("pen id must be a bounded lowercase identifier")
    for key in ("name", "type"):
        if not isinstance(pen[key], str) or not 1 <= len(pen[key]) <= 200:
            raise ValueError(f"pen {key} requires bounded text")
    if type(pen["enabled"]) is not bool:
        raise ValueError("pen enabled must be boolean")
    rgba = pen["rgba"]
    if not isinstance(rgba, list) or len(rgba) != 4 or any(type(v) is not int or not 0 <= v <= 255 for v in rgba):
        raise ValueError("pen rgba requires four 8-bit channels")
    if rgba[3] != 255:
        raise ValueError("transparent pens are unsupported")
    if type(pen["weight"]) is not int or not 0 <= pen["weight"] <= 1_000_000:
        raise ValueError("pen weight requires a nonnegative bounded integer")
    factor = pen["stroke_factor"]
    if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0 or base_width_mm * factor > 20:
        raise ValueError("effective pen width must be in (0, 20 mm]")


def native_drawing_sets(drawing_set: dict) -> dict:
    """One fixed-color native set; no catalog/sampled source pens are inherited."""
    pens = []
    for row, pen in enumerate(drawing_set["pens"]):
        red, green, blue, alpha = pen["rgba"]
        argb = (alpha << 24) | (red << 16) | (green << 8) | blue
        if argb >= 2**31:
            argb -= 2**32
        # A plain source pen prevents plugin-defined variable color behavior.
        source = {"type": pen["type"], "name": pen["id"], "argb": argb,
                  "distributionWeight": pen["weight"], "strokeSize": pen["stroke_factor"],
                  "isEnabled": pen["enabled"]}
        pens.append({"source": source, "penNumber": str(row), "isEnabled": str(pen["enabled"]).lower(),
                     "type": pen["type"], "name": pen["id"], "argb": str(argb),
                     "distributionWeight": str(pen["weight"]), "strokeSize": str(pen["stroke_factor"]),
                     "colorSplitMultiplier": "1.0", "colorSplitOpacity": "1.0",
                     "colorSplitOffsetX": "0.0", "colorSplitOffsetY": "0.0"})
    return {"drawingSets": [{"name": drawing_set["name"], "type": drawing_set["type"], "pens": pens,
                              "distributionOrder": drawing_set["distribution_order"],
                              "distributionType": drawing_set["distribution_type"], "colorHandler": "Default"}],
            "activeSet": 0}


def verify_pen_readback(requested: dict, actual: dict, base_width_mm: float) -> dict:
    """Bind native rows to logical IDs; refuse drift before a generation starts."""
    if any(actual.get(key) != requested[key] for key in
           ("name", "type", "distribution_order", "distribution_type", "color_separation")):
        raise RuntimeError("native drawing-set settings differ from the request")
    rows = actual.get("pens", [])
    if len(rows) != len(requested["pens"]) or actual.get("native_set_id") != 0:
        raise RuntimeError("native drawing-set identity differs from the request")
    result, groups = copy.deepcopy(actual), set()
    for index, (expected, row) in enumerate(zip(requested["pens"], result["pens"], strict=True)):
        exact = {"native_row_id": index, "name": expected["id"], "type": expected["type"],
                 "enabled": expected["enabled"], "rgba": expected["rgba"], "weight": expected["weight"]}
        if any(row.get(key) != value for key, value in exact.items()):
            raise RuntimeError(f"native pen readback differs for {expected['id']}")
        factor = row.get("stroke_factor")
        if type(factor) not in (int, float) or not math.isfinite(factor) or not math.isclose(
                factor, expected["stroke_factor"], rel_tol=1e-6, abs_tol=1e-8):
            raise RuntimeError(f"native stroke factor differs for {expected['id']}")
        group = row.get("export_group")
        if not isinstance(group, str) or not group or group in groups:
            raise RuntimeError("native export group identity is ambiguous")
        groups.add(group)
        row.update(id=expected["id"], requested_name=expected["name"],
                   generation_width_m=base_width_mm * factor / 1000)
    return result
