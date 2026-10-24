"""The TattooProgram/1 reader and the strict field vocabulary contract readers share.

A tattoo program is the metric artwork a design record carries: inks, layers of
paths, regions, dots and curves on a canvas in metres, negative-space masks and
provenance. ``validate_tattoo_program`` checks one completely, including its
declared canonical digest, and needs neither a renderer nor Node.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from tatbot_contracts.canonical import ContractError, canonical_digest

SCHEMA = "tatbot.tattoo-program/1"
# Every schema the human-representation contract readers know. A program whose
# schema is none of them is unknown_schema; a known schema other than SCHEMA is
# wrong_schema: the codes the TypeScript reader (web/inkmap) gives.
KNOWN_SCHEMAS = {
    "tatbot.body-model-spec/1",
    "tatbot.body-identity/1",
    "tatbot.body-state/1",
    "tatbot.tattoo-program/1",
    "tatbot.surface-coordinate/1",
    "tatbot.surface-placement/1",
    "tatbot.surface-placement/2",
    "tatbot.surface-curve/1",
    "tatbot.surface-curve/2",
    "tatbot.ink-program/1",
    "tatbot.ink-program/2",
    "tatbot.surface-registration/1",
    "tatbot.execution-program/1",
    "tatbot.draw-samples-manifest/1",
    "tatbot.tissue-patch/1",
    "tatbot.anatomy-registration/1",
    "tatbot.force-displacement-calibration/1",
    "tatbot.anatomy-source-manifest/1",
}
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _obj(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("wrong_type", path, "expected object")
    return value


def _list(value: Any, path: str, *, length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError("wrong_type", path, "expected array")
    if length is not None and len(value) != length:
        raise ContractError("wrong_length", path, f"expected {length}, got {len(value)}")
    return value


def _str(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ContractError("wrong_type", path, "expected non-empty string")
    return value


def _utc_timestamp(value: Any, path: str) -> str:
    text = _str(value, path)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text):
        raise ContractError("wrong_time", path, "expected an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError("wrong_time", path, str(exc)) from exc
    if parsed.tzinfo != timezone.utc:
        raise ContractError("wrong_time", path, "expected UTC")
    return text


def _num(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("wrong_type", path, "expected number")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError("non_finite", path, repr(value))
    if number == 0.0 and math.copysign(1.0, number) < 0:
        raise ContractError("negative_zero", path, "-0 is not canonical")
    return number


def _int(
    value: Any,
    path: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError("wrong_type", path, "expected integer")
    if value < minimum:
        raise ContractError("out_of_range", path, f"must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ContractError("out_of_range", path, f"must be at most {maximum}")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError("wrong_type", path, "expected boolean")
    return value


def _digest(value: Any, path: str) -> str:
    text = _str(value, path)
    if not DIGEST_RE.fullmatch(text):
        raise ContractError("wrong_hash", path, "expected lowercase SHA-256")
    return text


def _keys(
    value: dict[str, Any],
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise ContractError("missing_field", path, ", ".join(missing))
    if unknown:
        raise ContractError("unknown_field", path, ", ".join(unknown))


def _numbers(value: Any, path: str, *, length: int | None = None) -> list[float]:
    items = _list(value, path, length=length)
    return [_num(item, f"{path}[{index}]") for index, item in enumerate(items)]


def _canvas_point(value: Any, path: str, *, width: float, height: float) -> None:
    point = _numbers(value, path, length=2)
    if not (0.0 <= point[0] <= width and 0.0 <= point[1] <= height):
        raise ContractError("out_of_range", path, "point lies outside canvas_m")


def _provenance(value: Any, path: str) -> None:
    item = _obj(value, path)
    _keys(
        item,
        path,
        required={"producer", "version", "created_utc", "source_sha256"},
        optional={"seed", "checkpoint_sha256", "prompt"},
    )
    _str(item["producer"], f"{path}.producer")
    _str(item["version"], f"{path}.version")
    _utc_timestamp(item["created_utc"], f"{path}.created_utc")
    _digest(item["source_sha256"], f"{path}.source_sha256")
    if "checkpoint_sha256" in item:
        _digest(item["checkpoint_sha256"], f"{path}.checkpoint_sha256")
    if "seed" in item:
        _int(item["seed"], f"{path}.seed")
    if "prompt" in item:
        _str(item["prompt"], f"{path}.prompt")


def _program_inks(item: dict[str, Any], path: str) -> set[str]:
    inks = _list(item["inks"], f"{path}.inks")
    if not inks:
        raise ContractError("wrong_length", f"{path}.inks", "at least one ink is required")
    ink_ids: set[str] = set()
    for index, raw in enumerate(inks):
        ink = _obj(raw, f"{path}.inks[{index}]")
        _keys(ink, f"{path}.inks[{index}]", required={"id", "color_srgb"})
        ink_id = _str(ink["id"], f"{path}.inks[{index}].id")
        if ink_id in ink_ids:
            raise ContractError("duplicate_id", f"{path}.inks[{index}].id", ink_id)
        ink_ids.add(ink_id)
        rgb = _numbers(ink["color_srgb"], f"{path}.inks[{index}].color_srgb", length=3)
        if any(value < 0.0 or value > 1.0 for value in rgb):
            raise ContractError("out_of_range", f"{path}.inks[{index}].color_srgb", "expected [0,1]")
    return ink_ids


def _program_element(raw_element: Any, element_path: str, element_ids: set[str], width: float, height: float) -> None:
    element = _obj(raw_element, element_path)
    common = {"id", "kind", "closed", "fill", "width_m", "deposition"}
    _keys(
        element,
        element_path,
        required=common,
        optional={"points_m", "control_points_m"},
    )
    element_id = _str(element["id"], f"{element_path}.id")
    if element_id in element_ids:
        raise ContractError("duplicate_id", f"{element_path}.id", element_id)
    element_ids.add(element_id)
    kind = _str(element["kind"], f"{element_path}.kind")
    if kind in {"path", "region", "dots", "stipple"}:
        _keys(element, element_path, required=common | {"points_m"})
        points_path = f"{element_path}.points_m"
        points = _list(element["points_m"], points_path)
        minimum = 3 if kind == "region" else 2 if kind == "path" else 1
        if len(points) < minimum:
            raise ContractError(
                "wrong_length",
                points_path,
                f"{kind} requires at least {minimum} point(s)",
            )
    elif kind == "cubic_bezier":
        _keys(element, element_path, required=common | {"control_points_m"})
        points_path = f"{element_path}.control_points_m"
        points = _list(element["control_points_m"], points_path, length=4)
    else:
        raise ContractError("unsupported_element", f"{element_path}.kind", kind)
    for point_index, point in enumerate(points):
        _canvas_point(point, f"{points_path}[{point_index}]", width=width, height=height)
    _bool(element["closed"], f"{element_path}.closed")
    _bool(element["fill"], f"{element_path}.fill")
    if _num(element["width_m"], f"{element_path}.width_m") <= 0:
        raise ContractError("out_of_range", f"{element_path}.width_m", "must be positive")
    deposition = _num(element["deposition"], f"{element_path}.deposition")
    if not 0.0 <= deposition <= 1.0:
        raise ContractError("out_of_range", f"{element_path}.deposition", "expected [0,1]")


def _program_layers(item: dict[str, Any], path: str, ink_ids: set[str], width: float, height: float) -> None:
    layers = _list(item["layers"], f"{path}.layers")
    if not layers:
        raise ContractError("wrong_length", f"{path}.layers", "at least one layer is required")
    layer_ids: set[str] = set()
    element_ids: set[str] = set()
    for index, raw in enumerate(layers):
        layer = _obj(raw, f"{path}.layers[{index}]")
        _keys(layer, f"{path}.layers[{index}]", required={"id", "ink_id", "elements"})
        layer_id = _str(layer["id"], f"{path}.layers[{index}].id")
        if layer_id in layer_ids:
            raise ContractError("duplicate_id", f"{path}.layers[{index}].id", layer_id)
        layer_ids.add(layer_id)
        if layer["ink_id"] not in ink_ids:
            raise ContractError("unknown_reference", f"{path}.layers[{index}].ink_id", str(layer["ink_id"]))
        elements = _list(layer["elements"], f"{path}.layers[{index}].elements")
        if not elements:
            raise ContractError(
                "wrong_length",
                f"{path}.layers[{index}].elements",
                "at least one element is required",
            )
        for element_index, raw_element in enumerate(elements):
            _program_element(raw_element, f"{path}.layers[{index}].elements[{element_index}]", element_ids,
                             width, height)


def _program_masks(item: dict[str, Any], path: str, width: float, height: float) -> None:
    mask_ids: set[str] = set()
    for index, mask in enumerate(_list(item["negative_space_masks"], f"{path}.negative_space_masks")):
        entry = _obj(mask, f"{path}.negative_space_masks[{index}]")
        _keys(entry, f"{path}.negative_space_masks[{index}]", required={"id", "points_m"})
        mask_id = _str(entry["id"], f"{path}.negative_space_masks[{index}].id")
        if mask_id in mask_ids:
            raise ContractError("duplicate_id", f"{path}.negative_space_masks[{index}].id", mask_id)
        mask_ids.add(mask_id)
        mask_points = _list(entry["points_m"], f"{path}.negative_space_masks[{index}].points_m")
        if len(mask_points) < 3:
            raise ContractError("wrong_length", f"{path}.negative_space_masks[{index}].points_m", "at least three points required")
        for point_index, point in enumerate(mask_points):
            _canvas_point(
                point,
                f"{path}.negative_space_masks[{index}].points_m[{point_index}]",
                width=width,
                height=height,
            )


def _tattoo_program(item: dict[str, Any], path: str) -> None:
    _keys(
        item,
        path,
        required={
            "schema",
            "content_sha256",
            "canvas_m",
            "inks",
            "layers",
            "negative_space_masks",
            "semantic_intent",
            "preview_sha256",
            "provenance",
        },
    )
    canvas = _obj(item["canvas_m"], f"{path}.canvas_m")
    _keys(canvas, f"{path}.canvas_m", required={"width", "height"})
    width = _num(canvas["width"], f"{path}.canvas_m.width")
    height = _num(canvas["height"], f"{path}.canvas_m.height")
    if width <= 0 or height <= 0:
        raise ContractError("out_of_range", f"{path}.canvas_m", "dimensions must be positive")
    ink_ids = _program_inks(item, path)
    _program_layers(item, path, ink_ids, width, height)
    _program_masks(item, path, width, height)
    _str(item["semantic_intent"], f"{path}.semantic_intent")
    _digest(item["preview_sha256"], f"{path}.preview_sha256")
    _provenance(item["provenance"], f"{path}.provenance")


def validate_tattoo_program(value: Any) -> dict[str, Any]:
    """Validate one complete TattooProgram/1 and return it unchanged."""

    item = _obj(value, "$")
    schema = _str(item.get("schema"), "$.schema")
    if schema not in KNOWN_SCHEMAS:
        raise ContractError("unknown_schema", "$.schema", schema)
    if schema != SCHEMA:
        raise ContractError("wrong_schema", "$.schema", f"expected {SCHEMA}, got {schema}")
    _digest(item.get("content_sha256"), "$.content_sha256")
    _tattoo_program(item, "$")
    actual = canonical_digest(item)
    if item["content_sha256"] != actual:
        raise ContractError(
            "wrong_hash",
            "$.content_sha256",
            f"declared {item['content_sha256']}, computed {actual}",
        )
    return item
