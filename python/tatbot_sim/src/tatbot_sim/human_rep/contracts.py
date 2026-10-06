"""Strict readers and canonical digests for human-representation contracts."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tatbot_contracts.body import REVIEWED_SPEC_SHA256 as REVIEWED_MODEL_SPEC_SHA256
from tatbot_contracts.body import BodyCacheError, validate_spec
from tatbot_contracts.canonical import (
    ContractError,
    canonical_digest,
    parse_json,
)
from tatbot_contracts.canonical import canonical_bytes as canonical_bytes

SCHEMA_PREFIX = "tatbot."
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
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
MID_FACE_COUNT = 36_108


def _verify_declared_digest(value: dict[str, Any], path: str) -> str:
    declared = _digest(value.get("content_sha256"), f"{path}.content_sha256")
    actual = canonical_digest(value)
    if declared != actual:
        raise ContractError("wrong_hash", f"{path}.content_sha256", f"declared {declared}, computed {actual}")
    return actual


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


def _ordered_pair(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    pair = _numbers(value, path, length=2)
    if pair[0] > pair[1]:
        raise ContractError("out_of_range", path, "lower bound exceeds upper bound")
    if minimum is not None and pair[0] < minimum:
        raise ContractError("out_of_range", path, f"lower bound must be at least {minimum}")
    if maximum is not None and pair[1] > maximum:
        raise ContractError("out_of_range", path, f"upper bound must be at most {maximum}")
    return pair


def _matrix4(value: Any, path: str) -> None:
    rows = _list(value, path, length=4)
    for index, row in enumerate(rows):
        _numbers(row, f"{path}[{index}]", length=4)
    if _numbers(rows[3], f"{path}[3]", length=4) != [0.0, 0.0, 0.0, 1.0]:
        raise ContractError("wrong_transform", path, "last row must be [0,0,0,1]")


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


def _surface_coordinate(value: Any, path: str, *, top_level: bool = False) -> None:
    item = _obj(value, path)
    required = {"topology_sha256", "face_index", "barycentric"}
    if top_level:
        required |= {"schema", "content_sha256"}
    _keys(item, path, required=required)
    if top_level and item["schema"] != "tatbot.surface-coordinate/1":
        raise ContractError("wrong_schema", f"{path}.schema", str(item["schema"]))
    _digest(item["topology_sha256"], f"{path}.topology_sha256")
    _int(item["face_index"], f"{path}.face_index", maximum=MID_FACE_COUNT - 1)
    bary = _numbers(item["barycentric"], f"{path}.barycentric", length=3)
    total = sum(bary)
    if abs(total - 1.0) > 1e-6:
        raise ContractError("bad_barycentric", f"{path}.barycentric", f"sum is {total}")
    if any(number < -1e-6 or number > 1.0 + 1e-6 for number in bary):
        raise ContractError("bad_barycentric", f"{path}.barycentric", "component out of range")


def _body_identity(item: dict[str, Any], path: str) -> None:
    _keys(
        item,
        path,
        required={
            "schema",
            "content_sha256",
            "model_spec_sha256",
            "coefficients",
            "scales",
            "bounds_prior",
            "rest_surface_sha256",
            "provenance",
        },
    )
    _digest(item["model_spec_sha256"], f"{path}.model_spec_sha256")
    _numbers(item["coefficients"], f"{path}.coefficients", length=45)
    _numbers(item["scales"], f"{path}.scales", length=68)
    prior = _obj(item["bounds_prior"], f"{path}.bounds_prior")
    _keys(
        prior,
        f"{path}.bounds_prior",
        required={"name", "version", "max_abs_coefficient", "max_abs_scale"},
    )
    _str(prior["name"], f"{path}.bounds_prior.name")
    _str(prior["version"], f"{path}.bounds_prior.version")
    limit = _num(prior["max_abs_coefficient"], f"{path}.bounds_prior.max_abs_coefficient")
    if limit <= 0:
        raise ContractError("out_of_range", f"{path}.bounds_prior.max_abs_coefficient", "must be positive")
    if any(abs(number) > limit for number in item["coefficients"]):
        raise ContractError("out_of_range", f"{path}.coefficients", "coefficient exceeds prior")
    scale_limit = _num(prior["max_abs_scale"], f"{path}.bounds_prior.max_abs_scale")
    if scale_limit <= 0:
        raise ContractError("out_of_range", f"{path}.bounds_prior.max_abs_scale", "must be positive")
    if any(abs(number) > scale_limit for number in item["scales"]):
        raise ContractError("out_of_range", f"{path}.scales", "scale exceeds prior")
    _digest(item["rest_surface_sha256"], f"{path}.rest_surface_sha256")
    _provenance(item["provenance"], f"{path}.provenance")


def _body_state(item: dict[str, Any], path: str) -> None:
    _keys(
        item,
        path,
        required={
            "schema",
            "content_sha256",
            "model_spec_sha256",
            "body_identity_sha256",
            "joint_rotations_xyzw",
            "named_pose",
            "tracked_source",
            "tatbot_from_body",
            "confidence",
            "correctives_enabled",
            "posed_surface_sha256",
            "provenance",
        },
    )
    _digest(item["model_spec_sha256"], f"{path}.model_spec_sha256")
    _digest(item["body_identity_sha256"], f"{path}.body_identity_sha256")
    rotations = _list(item["joint_rotations_xyzw"], f"{path}.joint_rotations_xyzw", length=77)
    for index, rotation in enumerate(rotations):
        quat = _numbers(rotation, f"{path}.joint_rotations_xyzw[{index}]", length=4)
        if abs(sum(number * number for number in quat) - 1.0) > 1e-5:
            raise ContractError("bad_rotation", f"{path}.joint_rotations_xyzw[{index}]", "quaternion is not unit")
    named_pose = item["named_pose"]
    tracked_source = item["tracked_source"]
    if (named_pose is None) == (tracked_source is None):
        raise ContractError(
            "wrong_pose_source",
            path,
            "exactly one of named_pose and tracked_source must be populated",
        )
    if named_pose is not None:
        _str(named_pose, f"{path}.named_pose")
    else:
        tracked = _obj(tracked_source, f"{path}.tracked_source")
        _keys(
            tracked,
            f"{path}.tracked_source",
            required={"tracker", "sample_time_utc", "capture_sha256", "source_frame"},
        )
        _str(tracked["tracker"], f"{path}.tracked_source.tracker")
        _utc_timestamp(tracked["sample_time_utc"], f"{path}.tracked_source.sample_time_utc")
        _digest(tracked["capture_sha256"], f"{path}.tracked_source.capture_sha256")
        source_frame = _str(tracked["source_frame"], f"{path}.tracked_source.source_frame")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", source_frame):
            raise ContractError("wrong_frame", f"{path}.tracked_source.source_frame", source_frame)
    _matrix4(item["tatbot_from_body"], f"{path}.tatbot_from_body")
    confidence = _num(item["confidence"], f"{path}.confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ContractError("out_of_range", f"{path}.confidence", "expected [0,1]")
    if _bool(item["correctives_enabled"], f"{path}.correctives_enabled") is not True:
        raise ContractError(
            "wrong_value",
            f"{path}.correctives_enabled",
            "canonical MHR/SOMA mode requires correctives",
        )
    _digest(item["posed_surface_sha256"], f"{path}.posed_surface_sha256")
    _provenance(item["provenance"], f"{path}.provenance")


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
            element_path = f"{path}.layers[{index}].elements[{element_index}]"
            element = _obj(raw_element, f"{path}.layers[{index}].elements[{element_index}]")
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
                for point_index, point in enumerate(points):
                    _canvas_point(
                        point,
                        f"{points_path}[{point_index}]",
                        width=width,
                        height=height,
                    )
            elif kind == "cubic_bezier":
                _keys(element, element_path, required=common | {"control_points_m"})
                controls_path = f"{element_path}.control_points_m"
                controls = _list(element["control_points_m"], controls_path, length=4)
                for point_index, point in enumerate(controls):
                    _canvas_point(
                        point,
                        f"{controls_path}[{point_index}]",
                        width=width,
                        height=height,
                    )
            else:
                raise ContractError("unsupported_element", f"{element_path}.kind", kind)
            _bool(element["closed"], f"{element_path}.closed")
            _bool(element["fill"], f"{element_path}.fill")
            if _num(element["width_m"], f"{element_path}.width_m") <= 0:
                raise ContractError("out_of_range", f"{element_path}.width_m", "must be positive")
            deposition = _num(element["deposition"], f"{element_path}.deposition")
            if not 0.0 <= deposition <= 1.0:
                raise ContractError("out_of_range", f"{element_path}.deposition", "expected [0,1]")
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
    _str(item["semantic_intent"], f"{path}.semantic_intent")
    _digest(item["preview_sha256"], f"{path}.preview_sha256")
    _provenance(item["provenance"], f"{path}.provenance")


def _surface_curve(value: Any, path: str, *, top_level: bool = False, chart_allowed: bool = False) -> None:
    item = _obj(value, path)
    required = {
        "coordinates",
        "rest_surface_arc_length_m",
        "width_m",
        "deposition",
        "direction",
        "source_primitive_sha256",
        "compiler_sha256",
    }
    if top_level:
        required |= {"schema", "content_sha256"}
    _keys(item, path, required=required)
    if top_level and item["schema"] != ("tatbot.surface-curve/2" if chart_allowed else "tatbot.surface-curve/1"):
        raise ContractError("wrong_schema", f"{path}.schema", str(item["schema"]))
    coordinates = _list(item["coordinates"], f"{path}.coordinates")
    if len(coordinates) < 2:
        raise ContractError("wrong_length", f"{path}.coordinates", "at least two coordinates required")
    topology: str | None = None
    chart_points = []
    chart = chart_allowed and isinstance(coordinates[0], dict) and "chart_uv_m" in coordinates[0]
    for index, coordinate in enumerate(coordinates):
        coordinate_path = f"{path}.coordinates[{index}]"
        coordinate_item = _obj(coordinate, coordinate_path)
        if chart:
            _keys(coordinate_item, coordinate_path, required={"target_sha256", "chart_uv_m"})
            binding = _digest(coordinate_item["target_sha256"], f"{coordinate_path}.target_sha256")
            chart_points.append(_numbers(coordinate_item["chart_uv_m"], f"{coordinate_path}.chart_uv_m", length=2))
            if topology is not None and binding != topology:
                raise ContractError("wrong_target", coordinate_path, "curve mixes target bindings")
            topology = binding
            continue
        _surface_coordinate(coordinate, coordinate_path)
        if topology is None:
            topology = coordinate_item["topology_sha256"]
        elif coordinate_item["topology_sha256"] != topology:
            raise ContractError(
                "wrong_topology",
                f"{coordinate_path}.topology_sha256",
                "surface curve mixes topology digests",
            )
    if _num(item["rest_surface_arc_length_m"], f"{path}.rest_surface_arc_length_m") <= 0:
        raise ContractError("out_of_range", f"{path}.rest_surface_arc_length_m", "must be positive")
    if chart:
        length = sum(math.dist(a, b) for a, b in zip(chart_points[:-1], chart_points[1:], strict=True))
        if not math.isfinite(length) or abs(length - item["rest_surface_arc_length_m"]) > 1e-9:
            raise ContractError("wrong_length", f"{path}.rest_surface_arc_length_m", "differs from metric chart curve")
    if _num(item["width_m"], f"{path}.width_m") <= 0:
        raise ContractError("out_of_range", f"{path}.width_m", "must be positive")
    deposition = _num(item["deposition"], f"{path}.deposition")
    if not 0.0 <= deposition <= 1.0:
        raise ContractError("out_of_range", f"{path}.deposition", "expected [0,1]")
    if item["direction"] not in {"forward", "reverse"}:
        raise ContractError("wrong_enum", f"{path}.direction", str(item["direction"]))
    _digest(item["source_primitive_sha256"], f"{path}.source_primitive_sha256")
    _digest(item["compiler_sha256"], f"{path}.compiler_sha256")


def _surface_placement(item: dict[str, Any], path: str) -> None:
    _keys(
        item,
        path,
        required={
            "schema",
            "content_sha256",
            "tattoo_program_sha256",
            "body_identity_sha256",
            "rest_surface_sha256",
            "semantic_site",
            "laterality",
            "anchor",
            "tangent_frame_rule",
            "physical_scale_m",
            "rotation_rad",
            "mirrored",
            "warp",
            "supported_domain",
            "review",
            "provenance",
        },
    )
    for field in ("tattoo_program_sha256", "body_identity_sha256", "rest_surface_sha256"):
        _digest(item[field], f"{path}.{field}")
    _str(item["semantic_site"], f"{path}.semantic_site")
    if item["laterality"] not in {"left", "right", "midline", "not_applicable"}:
        raise ContractError("wrong_enum", f"{path}.laterality", str(item["laterality"]))
    _surface_coordinate(item["anchor"], f"{path}.anchor")
    anchor = _obj(item["anchor"], f"{path}.anchor")
    _str(item["tangent_frame_rule"], f"{path}.tangent_frame_rule")
    scale = _numbers(item["physical_scale_m"], f"{path}.physical_scale_m", length=2)
    if any(number <= 0 for number in scale):
        raise ContractError("out_of_range", f"{path}.physical_scale_m", "must be positive")
    _num(item["rotation_rad"], f"{path}.rotation_rad")
    _bool(item["mirrored"], f"{path}.mirrored")
    if item["warp"] is not None:
        warp = _obj(item["warp"], f"{path}.warp")
        _keys(warp, f"{path}.warp", required={"kind", "max_displacement_m", "parameters"})
        _str(warp["kind"], f"{path}.warp.kind")
        if _num(warp["max_displacement_m"], f"{path}.warp.max_displacement_m") < 0:
            raise ContractError("out_of_range", f"{path}.warp.max_displacement_m", "must be nonnegative")
        _numbers(warp["parameters"], f"{path}.warp.parameters")
    domain = _obj(item["supported_domain"], f"{path}.supported_domain")
    _keys(domain, f"{path}.supported_domain", required={"face_indices", "margin_m"})
    faces = _list(domain["face_indices"], f"{path}.supported_domain.face_indices")
    if not faces:
        raise ContractError("wrong_length", f"{path}.supported_domain.face_indices", "at least one face required")
    for index, face in enumerate(faces):
        _int(
            face,
            f"{path}.supported_domain.face_indices[{index}]",
            maximum=MID_FACE_COUNT - 1,
        )
    if anchor["face_index"] not in faces:
        raise ContractError(
            "anchor_outside_domain",
            f"{path}.anchor.face_index",
            "anchor face is absent from supported_domain.face_indices",
        )
    if _num(domain["margin_m"], f"{path}.supported_domain.margin_m") < 0:
        raise ContractError("out_of_range", f"{path}.supported_domain.margin_m", "must be nonnegative")
    review = _obj(item["review"], f"{path}.review")
    _keys(review, f"{path}.review", required={"status", "reviewer", "evidence_sha256"})
    if review["status"] not in {"pending", "accepted", "rejected"}:
        raise ContractError("wrong_enum", f"{path}.review.status", str(review["status"]))
    _str(review["reviewer"], f"{path}.review.reviewer")
    _digest(review["evidence_sha256"], f"{path}.review.evidence_sha256")
    _provenance(item["provenance"], f"{path}.provenance")


def _target_placement(item: dict[str, Any], path: str) -> None:
    """Placement intent; analytic charts carry no invented body or rig pose."""
    common = {"schema", "content_sha256", "tattoo_program_sha256", "physical_scale_m",
              "rotation_rad", "mirrored", "warp", "review", "provenance"}
    _keys(item, path, required=common | {"target"})
    target = _obj(item["target"], f"{path}.target")
    kind = _str(target.get("kind"), f"{path}.target.kind")
    if kind == "body":
        fields = {"body_identity_sha256", "rest_surface_sha256", "semantic_site", "laterality",
                  "anchor", "tangent_frame_rule", "supported_domain"}
        _keys(target, f"{path}.target", required=fields | {"kind"})
        # Same body checks, with the real binding; no synthetic topology.
        _surface_placement({**{key: item[key] for key in common},
                            **{key: target[key] for key in fields}}, path)
        return
    if kind not in {"plane", "cylinder"}:
        raise ContractError("wrong_enum", f"{path}.target.kind", kind)
    fields = {"kind", "canvas_m", "anchor_uv_m", "margin_m"}
    _keys(target, f"{path}.target", required=fields | ({"radius_m"} if kind == "cylinder" else set()))
    canvas = _numbers(target["canvas_m"], f"{path}.target.canvas_m", length=2)
    anchor = _numbers(target["anchor_uv_m"], f"{path}.target.anchor_uv_m", length=2)
    scale = _numbers(item["physical_scale_m"], f"{path}.physical_scale_m", length=2)
    margin = _num(target["margin_m"], f"{path}.target.margin_m")
    if any(value <= 0 for value in canvas + scale) or margin < 0:
        raise ContractError("out_of_range", f"{path}.target", "positive dimensions and nonnegative margin required")
    if kind == "cylinder":
        radius = _num(target["radius_m"], f"{path}.target.radius_m")
        # v is circumferential arc length, matching surface_model.CylinderChart.
        if radius <= 0 or canvas[1] >= 2 * math.pi * radius:
            raise ContractError("girth", f"{path}.target.canvas_m", "chart must be smaller than one circumference")
    angle = _num(item["rotation_rad"], f"{path}.rotation_rad")
    _bool(item["mirrored"], f"{path}.mirrored")
    if item["warp"] is not None:
        raise ContractError("unsupported_warp", f"{path}.warp", "analytic placements require an unwarped chart")
    c, s = abs(math.cos(angle)), abs(math.sin(angle))
    extent = [(c * scale[0] + s * scale[1]) / 2, (s * scale[0] + c * scale[1]) / 2]
    if any(abs(a) + e + margin > size / 2 + 1e-12 for a, e, size in zip(anchor, extent, canvas, strict=True)):
        raise ContractError("placement_outside_domain", f"{path}.target", "rotated artwork canvas exceeds target margin")
    _digest(item["tattoo_program_sha256"], f"{path}.tattoo_program_sha256")
    review = _obj(item["review"], f"{path}.review")
    _keys(review, f"{path}.review", required={"status", "reviewer", "evidence_sha256"})
    if review["status"] not in {"pending", "accepted", "rejected"}:
        raise ContractError("wrong_enum", f"{path}.review.status", str(review["status"]))
    _str(review["reviewer"], f"{path}.review.reviewer")
    _digest(review["evidence_sha256"], f"{path}.review.evidence_sha256")
    _provenance(item["provenance"], f"{path}.provenance")


def _ink_state(value: Any, path: str) -> None:
    state = _obj(value, path)
    _keys(state, path, required={"ink_id", "load_fraction"})
    _str(state["ink_id"], f"{path}.ink_id")
    load = _num(state["load_fraction"], f"{path}.load_fraction")
    if not 0.0 <= load <= 1.0:
        raise ContractError("out_of_range", f"{path}.load_fraction", "expected [0,1]")


# The paint planners a compiled InkProgram may name in `fill_style`; absent
# means `concentric`, the contract's default.
FILL_STYLES = ("concentric", "hatch")


def _operating_budget(value: Any, path: str) -> None:
    """Validate legacy budget metadata when an old program carries it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ContractError("out_of_range", path, "expected a positive finite number of seconds")


def _ink_program_options(item: dict[str, Any], path: str) -> None:
    """The optional compile options a program carries when they are not the default."""
    if "operating_budget_s" in item:
        _operating_budget(item["operating_budget_s"], f"{path}.operating_budget_s")
    if "fill_style" in item and item["fill_style"] not in FILL_STYLES:
        raise ContractError("wrong_enum", f"{path}.fill_style", str(item["fill_style"]))


def _ink_program(item: dict[str, Any], path: str) -> None:
    _keys(
        item,
        path,
        required={
            "schema",
            "content_sha256",
            "tattoo_program_sha256",
            "surface_placement_sha256",
            "compiler_sha256",
            "initial_ink_state",
            "predicted_ink_state",
            "events",
            "total_material_path_length_m",
            "uncertainty",
            "provenance",
        },
        optional={"operating_budget_s", "fill_style"},
    )
    _ink_program_options(item, path)
    for field in ("tattoo_program_sha256", "surface_placement_sha256", "compiler_sha256"):
        _digest(item[field], f"{path}.{field}")
    for field in ("initial_ink_state", "predicted_ink_state"):
        _ink_state(item[field], f"{path}.{field}")
    events = _list(item["events"], f"{path}.events")
    if not events:
        raise ContractError("wrong_length", f"{path}.events", "at least one event required")
    allowed = {"stroke", "pen_transition", "dip", "tool_change", "barrier"}
    material_length = 0.0
    for index, raw in enumerate(events):
        event = _obj(raw, f"{path}.events[{index}]")
        kind = event.get("kind")
        if kind not in allowed:
            raise ContractError("wrong_enum", f"{path}.events[{index}].kind", str(kind))
        if kind == "stroke":
            _keys(
                event,
                f"{path}.events[{index}]",
                required={
                    "kind",
                    "curve",
                    "start_choice",
                    "allowed_tool_class",
                    "speed_m_s",
                    "orientation_tolerance_rad",
                    "contact_envelope_m",
                    "research_depth_band_m",
                    "ordering_rationale",
                },
            )
            _surface_curve(event["curve"], f"{path}.events[{index}].curve",
                           chart_allowed=item["schema"] == "tatbot.ink-program/2")
            for coordinate in event["curve"]["coordinates"]:
                if "target_sha256" in coordinate and coordinate["target_sha256"] != item["surface_placement_sha256"]:
                    raise ContractError("wrong_target", f"{path}.events[{index}].curve", "curve differs from program placement")
            material_length += _num(
                _obj(event["curve"], f"{path}.events[{index}].curve")["rest_surface_arc_length_m"],
                f"{path}.events[{index}].curve.rest_surface_arc_length_m",
            )
            if event["start_choice"] not in {"start", "end"}:
                raise ContractError("wrong_enum", f"{path}.events[{index}].start_choice", str(event["start_choice"]))
            _str(event["allowed_tool_class"], f"{path}.events[{index}].allowed_tool_class")
            if _num(event["speed_m_s"], f"{path}.events[{index}].speed_m_s") <= 0:
                raise ContractError("out_of_range", f"{path}.events[{index}].speed_m_s", "must be positive")
            if _num(event["orientation_tolerance_rad"], f"{path}.events[{index}].orientation_tolerance_rad") < 0:
                raise ContractError("out_of_range", f"{path}.events[{index}].orientation_tolerance_rad", "must be nonnegative")
            _ordered_pair(event["contact_envelope_m"], f"{path}.events[{index}].contact_envelope_m")
            if event["research_depth_band_m"] is not None:
                _ordered_pair(
                    event["research_depth_band_m"],
                    f"{path}.events[{index}].research_depth_band_m",
                )
            _str(event["ordering_rationale"], f"{path}.events[{index}].ordering_rationale")
        elif kind == "pen_transition":
            _keys(event, f"{path}.events[{index}]", required={"kind", "state"})
            if event["state"] not in {"lift", "approach", "contact_intent", "retract"}:
                raise ContractError("wrong_enum", f"{path}.events[{index}].state", str(event["state"]))
        elif kind == "dip":
            _keys(
                event,
                f"{path}.events[{index}]",
                required={"kind", "ink_id", "target_load", "trigger_reason", "dependency", "expected_load_after"},
            )
            _str(event["ink_id"], f"{path}.events[{index}].ink_id")
            _ordered_pair(
                event["target_load"],
                f"{path}.events[{index}].target_load",
                minimum=0.0,
                maximum=1.0,
            )
            _str(event["trigger_reason"], f"{path}.events[{index}].trigger_reason")
            _str(event["dependency"], f"{path}.events[{index}].dependency")
            expected_load = _num(event["expected_load_after"], f"{path}.events[{index}].expected_load_after")
            if not 0 <= expected_load <= 1:
                raise ContractError("out_of_range", f"{path}.events[{index}].expected_load_after", "expected [0,1]")
        elif kind == "tool_change":
            _keys(
                event,
                f"{path}.events[{index}]",
                required={"kind", "required_tool_class", "transition_intent"},
            )
            _str(event["required_tool_class"], f"{path}.events[{index}].required_tool_class")
            _str(event["transition_intent"], f"{path}.events[{index}].transition_intent")
        else:
            _keys(event, f"{path}.events[{index}]", required={"kind", "intent"})
            _str(event["intent"], f"{path}.events[{index}].intent")
    declared_length = _num(item["total_material_path_length_m"], f"{path}.total_material_path_length_m")
    if declared_length < 0:
        raise ContractError("out_of_range", f"{path}.total_material_path_length_m", "must be nonnegative")
    if not math.isclose(declared_length, material_length, rel_tol=1e-12, abs_tol=1e-12):
        raise ContractError(
            "wrong_value",
            f"{path}.total_material_path_length_m",
            f"declared {declared_length}; stroke sum is {material_length}",
        )
    uncertainty = _obj(item["uncertainty"], f"{path}.uncertainty")
    _keys(uncertainty, f"{path}.uncertainty", required={"length_sigma_m", "load_sigma"})
    if _num(uncertainty["length_sigma_m"], f"{path}.uncertainty.length_sigma_m") < 0:
        raise ContractError("out_of_range", f"{path}.uncertainty.length_sigma_m", "must be nonnegative")
    if _num(uncertainty["load_sigma"], f"{path}.uncertainty.load_sigma") < 0:
        raise ContractError("out_of_range", f"{path}.uncertainty.load_sigma", "must be nonnegative")
    _provenance(item["provenance"], f"{path}.provenance")


def _surface_registration(item: dict[str, Any], path: str) -> None:
    _keys(
        item,
        path,
        required={
            "schema",
            "content_sha256",
            "source_frame",
            "target_frame",
            "body_state_sha256",
            "measured_surface_sha256",
            "method",
            "correspondences",
            "observed_patch_from_body",
            "supported_cells",
            "covariance",
            "confidence",
            "capture_sha256",
            "calibration_sha256",
            "provenance",
        },
    )
    if item["source_frame"] != "body":
        raise ContractError("wrong_frame", f"{path}.source_frame", str(item["source_frame"]))
    if item["target_frame"] != "observed_patch":
        raise ContractError("wrong_frame", f"{path}.target_frame", str(item["target_frame"]))
    for field in ("body_state_sha256", "measured_surface_sha256", "capture_sha256", "calibration_sha256"):
        _digest(item[field], f"{path}.{field}")
    _str(item["method"], f"{path}.method")
    correspondences = _list(item["correspondences"], f"{path}.correspondences")
    if not correspondences:
        raise ContractError("wrong_length", f"{path}.correspondences", "at least one correspondence required")
    for index, raw in enumerate(correspondences):
        correspondence = _obj(raw, f"{path}.correspondences[{index}]")
        _keys(correspondence, f"{path}.correspondences[{index}]", required={"canonical", "observed_xyz_m", "error_m"})
        _surface_coordinate(correspondence["canonical"], f"{path}.correspondences[{index}].canonical")
        _numbers(correspondence["observed_xyz_m"], f"{path}.correspondences[{index}].observed_xyz_m", length=3)
        if _num(correspondence["error_m"], f"{path}.correspondences[{index}].error_m") < 0:
            raise ContractError("out_of_range", f"{path}.correspondences[{index}].error_m", "must be nonnegative")
    _matrix4(item["observed_patch_from_body"], f"{path}.observed_patch_from_body")
    supported_cells = _list(item["supported_cells"], f"{path}.supported_cells")
    if not supported_cells:
        raise ContractError("wrong_length", f"{path}.supported_cells", "at least one cell required")
    for index, cell in enumerate(supported_cells):
        _int(cell, f"{path}.supported_cells[{index}]")
    _numbers(item["covariance"], f"{path}.covariance", length=36)
    confidence = _num(item["confidence"], f"{path}.confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ContractError("out_of_range", f"{path}.confidence", "expected [0,1]")
    _provenance(item["provenance"], f"{path}.provenance")


def _execution_program(item: dict[str, Any], path: str) -> None:
    _keys(
        item,
        path,
        required={
            "schema",
            "content_sha256",
            "ink_program",
            "body_state",
            "surface_registration",
            "measured_surface",
            "tool",
            "robot",
            "support",
            "palette",
            "calibration",
            "exact_events",
            "uncertainty",
            "preflight",
            "samples_manifest",
            "provenance",
        },
    )
    ink_program = _obj(item["ink_program"], f"{path}.ink_program")
    body_state = _obj(item["body_state"], f"{path}.body_state")
    registration = _obj(item["surface_registration"], f"{path}.surface_registration")
    _ink_program(ink_program, f"{path}.ink_program")
    _body_state(body_state, f"{path}.body_state")
    _surface_registration(registration, f"{path}.surface_registration")
    _verify_declared_digest(ink_program, f"{path}.ink_program")
    body_state_digest = _verify_declared_digest(body_state, f"{path}.body_state")
    _verify_declared_digest(registration, f"{path}.surface_registration")
    if registration["body_state_sha256"] != body_state_digest:
        raise ContractError(
            "wrong_hash",
            f"{path}.surface_registration.body_state_sha256",
            "does not bind the embedded body state",
        )
    measured = _obj(item["measured_surface"], f"{path}.measured_surface")
    _keys(measured, f"{path}.measured_surface", required={"schema", "path", "sha256"})
    if measured["schema"] != "tatbot.surface/1":
        raise ContractError("wrong_schema", f"{path}.measured_surface.schema", str(measured["schema"]))
    _str(measured["path"], f"{path}.measured_surface.path")
    measured_digest = _digest(measured["sha256"], f"{path}.measured_surface.sha256")
    if registration["measured_surface_sha256"] != measured_digest:
        raise ContractError(
            "wrong_hash",
            f"{path}.surface_registration.measured_surface_sha256",
            "does not bind the measured surface",
        )
    tool = _obj(item["tool"], f"{path}.tool")
    _keys(tool, f"{path}.tool", required={"id", "datasheet_sha256", "class"})
    _str(tool["id"], f"{path}.tool.id")
    _digest(tool["datasheet_sha256"], f"{path}.tool.datasheet_sha256")
    tool_class = _str(tool["class"], f"{path}.tool.class")
    ink_events = _list(ink_program["events"], f"{path}.ink_program.events")
    for index, ink_event in enumerate(ink_events):
        required_class = None
        if ink_event["kind"] == "stroke":
            required_class = ink_event["allowed_tool_class"]
        elif ink_event["kind"] == "tool_change":
            required_class = ink_event["required_tool_class"]
        if required_class is not None and required_class != tool_class:
            raise ContractError(
                "execution_binding_mismatch",
                f"{path}.ink_program.events[{index}]",
                f"requires tool class {required_class!r}; execution binds {tool_class!r}",
            )
    robot = _obj(item["robot"], f"{path}.robot")
    _keys(robot, f"{path}.robot", required={"id", "urdf_sha256"})
    _str(robot["id"], f"{path}.robot.id")
    _digest(robot["urdf_sha256"], f"{path}.robot.urdf_sha256")
    support = _obj(item["support"], f"{path}.support")
    _keys(support, f"{path}.support", required={"kind", "configuration_sha256"})
    _str(support["kind"], f"{path}.support.kind")
    _digest(support["configuration_sha256"], f"{path}.support.configuration_sha256")
    palette = _obj(item["palette"], f"{path}.palette")
    _keys(palette, f"{path}.palette", required={"snapshot_sha256", "load_state_sha256", "resolved_caps"})
    _digest(palette["snapshot_sha256"], f"{path}.palette.snapshot_sha256")
    _digest(palette["load_state_sha256"], f"{path}.palette.load_state_sha256")
    caps = _list(palette["resolved_caps"], f"{path}.palette.resolved_caps")
    if not caps:
        raise ContractError("wrong_length", f"{path}.palette.resolved_caps", "at least one cap required")
    cap_inks: set[str] = set()
    cap_slots: set[int] = set()
    for index, raw in enumerate(caps):
        cap = _obj(raw, f"{path}.palette.resolved_caps[{index}]")
        _keys(cap, f"{path}.palette.resolved_caps[{index}]", required={"ink_id", "slot", "robot_base_from_cap"})
        ink_id = _str(cap["ink_id"], f"{path}.palette.resolved_caps[{index}].ink_id")
        slot = _int(cap["slot"], f"{path}.palette.resolved_caps[{index}].slot")
        if ink_id in cap_inks or slot in cap_slots:
            raise ContractError(
                "execution_binding_mismatch",
                f"{path}.palette.resolved_caps[{index}]",
                "ink IDs and cap slots must be unique",
            )
        cap_inks.add(ink_id)
        cap_slots.add(slot)
        _matrix4(cap["robot_base_from_cap"], f"{path}.palette.resolved_caps[{index}].robot_base_from_cap")
    required_inks = {
        _obj(ink_program[name], f"{path}.ink_program.{name}")["ink_id"]
        for name in ("initial_ink_state", "predicted_ink_state")
    }
    required_inks.update(
        event["ink_id"] for event in ink_events if event["kind"] == "dip"
    )
    missing_inks = sorted(required_inks - cap_inks)
    if missing_inks:
        raise ContractError(
            "execution_binding_mismatch",
            f"{path}.palette.resolved_caps",
            f"missing ink IDs: {', '.join(missing_inks)}",
        )
    calibration = _obj(item["calibration"], f"{path}.calibration")
    _keys(calibration, f"{path}.calibration", required={"sha256", "captured_utc"})
    calibration_digest = _digest(calibration["sha256"], f"{path}.calibration.sha256")
    _utc_timestamp(calibration["captured_utc"], f"{path}.calibration.captured_utc")
    if registration["calibration_sha256"] != calibration_digest:
        raise ContractError(
            "execution_binding_mismatch",
            f"{path}.surface_registration.calibration_sha256",
            "does not bind the execution calibration",
        )
    exact_events = _list(item["exact_events"], f"{path}.exact_events")
    if not exact_events:
        raise ContractError("wrong_length", f"{path}.exact_events", "at least one event required")
    actual_event_indices: list[int] = []
    previous_stop = 0
    for index, raw in enumerate(exact_events):
        event = _obj(raw, f"{path}.exact_events[{index}]")
        _keys(event, f"{path}.exact_events[{index}]", required={"ink_event_index", "kind", "sample_range"})
        ink_event_index = _int(event["ink_event_index"], f"{path}.exact_events[{index}].ink_event_index")
        kind = _str(event["kind"], f"{path}.exact_events[{index}].kind")
        if ink_event_index >= len(ink_events) or ink_events[ink_event_index]["kind"] != kind:
            raise ContractError(
                "execution_binding_mismatch",
                f"{path}.exact_events[{index}]",
                "does not identify the same embedded ink event",
            )
        actual_event_indices.append(ink_event_index)
        sample_range = _list(event["sample_range"], f"{path}.exact_events[{index}].sample_range", length=2)
        start, stop = (
            _int(value, f"{path}.exact_events[{index}].sample_range[{range_index}]")
            for range_index, value in enumerate(sample_range)
        )
        if stop <= start:
            raise ContractError("out_of_range", f"{path}.exact_events[{index}].sample_range", "range must be nonempty")
        if index and start < previous_stop:
            raise ContractError(
                "trajectory_discontinuous",
                f"{path}.exact_events[{index}].sample_range",
                "sample ranges overlap or run backward",
            )
        previous_stop = stop
    expected_event_indices = [
        index for index, event in enumerate(ink_events) if event["kind"] == "stroke"
    ]
    if actual_event_indices != expected_event_indices:
        raise ContractError(
            "execution_binding_mismatch",
            f"{path}.exact_events",
            "must map every stroke exactly once in ink-program order",
        )
    uncertainty = _obj(item["uncertainty"], f"{path}.uncertainty")
    _keys(uncertainty, f"{path}.uncertainty", required={"registration_sigma_m", "surface_sigma_m"})
    if _num(uncertainty["registration_sigma_m"], f"{path}.uncertainty.registration_sigma_m") < 0:
        raise ContractError("out_of_range", f"{path}.uncertainty.registration_sigma_m", "must be nonnegative")
    if _num(uncertainty["surface_sigma_m"], f"{path}.uncertainty.surface_sigma_m") < 0:
        raise ContractError("out_of_range", f"{path}.uncertainty.surface_sigma_m", "must be nonnegative")
    preflight = _obj(item["preflight"], f"{path}.preflight")
    _keys(preflight, f"{path}.preflight", required={"observed_cells_only", "max_surface_age_s", "policy_sha256"})
    if _bool(preflight["observed_cells_only"], f"{path}.preflight.observed_cells_only") is not True:
        raise ContractError("wrong_value", f"{path}.preflight.observed_cells_only", "must be true")
    if _num(preflight["max_surface_age_s"], f"{path}.preflight.max_surface_age_s") <= 0:
        raise ContractError("out_of_range", f"{path}.preflight.max_surface_age_s", "must be positive")
    _digest(preflight["policy_sha256"], f"{path}.preflight.policy_sha256")
    manifest = _obj(item["samples_manifest"], f"{path}.samples_manifest")
    _keys(manifest, f"{path}.samples_manifest", required={"schema", "path", "sha256", "sidecar_path", "sidecar_sha256"})
    if manifest["schema"] != "tatbot.draw-samples/1":
        raise ContractError("wrong_schema", f"{path}.samples_manifest.schema", str(manifest["schema"]))
    _str(manifest["path"], f"{path}.samples_manifest.path")
    _digest(manifest["sha256"], f"{path}.samples_manifest.sha256")
    _str(manifest["sidecar_path"], f"{path}.samples_manifest.sidecar_path")
    _digest(manifest["sidecar_sha256"], f"{path}.samples_manifest.sidecar_sha256")
    _provenance(item["provenance"], f"{path}.provenance")


def _draw_samples_manifest(item: dict[str, Any], path: str) -> None:
    _keys(
        item,
        path,
        required={
            "schema",
            "content_sha256",
            "samples",
            "compiler",
            "inputs",
            "preflight",
            "sample_ranges",
            "provenance",
        },
    )
    samples = _obj(item["samples"], f"{path}.samples")
    _keys(samples, f"{path}.samples", required={"schema", "path", "sha256"})
    if samples["schema"] != "tatbot.draw-samples/1":
        raise ContractError("wrong_schema", f"{path}.samples.schema", str(samples["schema"]))
    _str(samples["path"], f"{path}.samples.path")
    _digest(samples["sha256"], f"{path}.samples.sha256")

    compiler = _obj(item["compiler"], f"{path}.compiler")
    _keys(compiler, f"{path}.compiler", required={"name", "version", "sha256"})
    _str(compiler["name"], f"{path}.compiler.name")
    _str(compiler["version"], f"{path}.compiler.version")
    _digest(compiler["sha256"], f"{path}.compiler.sha256")

    input_names = {
        "ink_program_sha256",
        "body_state_sha256",
        "surface_registration_sha256",
        "measured_surface_sha256",
        "tool_datasheet_sha256",
        "robot_urdf_sha256",
        "support_configuration_sha256",
        "palette_snapshot_sha256",
        "palette_load_state_sha256",
        "calibration_sha256",
        "policy_sha256",
    }
    inputs = _obj(item["inputs"], f"{path}.inputs")
    _keys(inputs, f"{path}.inputs", required=input_names)
    for name in input_names:
        _digest(inputs[name], f"{path}.inputs.{name}")

    preflight = _obj(item["preflight"], f"{path}.preflight")
    _keys(
        preflight,
        f"{path}.preflight",
        required={
            "mode",
            "motion_authorized",
            "observed_cells_only",
            "registration_sigma_m",
            "surface_sigma_m",
            "max_surface_age_s",
        },
    )
    _str(preflight["mode"], f"{path}.preflight.mode")
    if _bool(preflight["motion_authorized"], f"{path}.preflight.motion_authorized") is not False:
        raise ContractError("wrong_value", f"{path}.preflight.motion_authorized", "must be false")
    if _bool(preflight["observed_cells_only"], f"{path}.preflight.observed_cells_only") is not True:
        raise ContractError("wrong_value", f"{path}.preflight.observed_cells_only", "must be true")
    for name in ("registration_sigma_m", "surface_sigma_m"):
        if _num(preflight[name], f"{path}.preflight.{name}") < 0:
            raise ContractError("out_of_range", f"{path}.preflight.{name}", "must be nonnegative")
    if _num(preflight["max_surface_age_s"], f"{path}.preflight.max_surface_age_s") <= 0:
        raise ContractError("out_of_range", f"{path}.preflight.max_surface_age_s", "must be positive")

    ranges = _list(item["sample_ranges"], f"{path}.sample_ranges")
    if not ranges:
        raise ContractError("wrong_length", f"{path}.sample_ranges", "at least one range is required")
    previous_event_index = -1
    previous_stop = 0
    for index, raw in enumerate(ranges):
        entry = _obj(raw, f"{path}.sample_ranges[{index}]")
        _keys(
            entry,
            f"{path}.sample_ranges[{index}]",
            required={"ink_event_index", "kind", "start", "stop"},
        )
        ink_event_index = _int(
            entry["ink_event_index"],
            f"{path}.sample_ranges[{index}].ink_event_index",
        )
        _str(entry["kind"], f"{path}.sample_ranges[{index}].kind")
        start = _int(entry["start"], f"{path}.sample_ranges[{index}].start")
        stop = _int(entry["stop"], f"{path}.sample_ranges[{index}].stop")
        if stop <= start:
            raise ContractError("out_of_range", f"{path}.sample_ranges[{index}]", "range must be nonempty")
        if ink_event_index <= previous_event_index or (index and start < previous_stop):
            raise ContractError(
                "trajectory_discontinuous",
                f"{path}.sample_ranges[{index}]",
                "event indices and sample ranges must be ordered and non-overlapping",
            )
        previous_event_index = ink_event_index
        previous_stop = stop
    _provenance(item["provenance"], f"{path}.provenance")


def _body_model_spec(item: dict[str, Any], path: str) -> None:
    try:
        validate_spec(item)
    except BodyCacheError as exc:
        # Retain established generic reader codes; cache entrypoints map to body_*.
        code = {
            "spec_value": "wrong_value", "spec_type": "wrong_type",
            "noncanonical_path": "path_escape", "wrong_spec_hash": "wrong_hash",
        }.get(exc.code, exc.code)
        raise ContractError(code, path, exc.detail) from exc


def _parameter_estimate(value: Any, path: str) -> None:
    estimate = _obj(value, path)
    _keys(estimate, path, required={"value", "sigma", "unit", "identified"})
    _num(estimate["value"], f"{path}.value")
    if _num(estimate["sigma"], f"{path}.sigma") < 0:
        raise ContractError("out_of_range", f"{path}.sigma", "must be nonnegative")
    _str(estimate["unit"], f"{path}.unit")
    _bool(estimate["identified"], f"{path}.identified")


def _trace_bindings(value: Any, path: str) -> None:
    bindings = _obj(value, path)
    _keys(
        bindings,
        path,
        required={
            "unit",
            "commanded_trace_sha256",
            "estimated_trace_sha256",
            "measured_trace_sha256",
        },
    )
    _str(bindings["unit"], f"{path}.unit")
    present = 0
    for field in (
        "commanded_trace_sha256",
        "estimated_trace_sha256",
        "measured_trace_sha256",
    ):
        if bindings[field] is not None:
            _digest(bindings[field], f"{path}.{field}")
            present += 1
    if present == 0:
        raise ContractError("missing_measurement", path, "at least one trace binding is required")


def _simple_research_contract(item: dict[str, Any], path: str, kind: str) -> None:
    if kind == "tatbot.tissue-patch/1":
        required = {
            "schema",
            "content_sha256",
            "material_chart_sha256",
            "rest_surface_sha256",
            "deformed_surface_sha256",
            "layer_prior_fields",
            "constitutive_model",
            "parameters",
            "uncertainty",
            "damping",
            "relaxation",
            "friction",
            "boundary_conditions",
            "support_sha256",
            "tool_geometry_sha256",
            "indentation",
            "force",
            "calibration_sha256",
            "batch_sha256",
            "provenance",
        }
    else:
        required = {
            "schema",
            "content_sha256",
            "source",
            "version",
            "license",
            "structure_ids",
            "atlas_from_body",
            "warp",
            "error_m",
            "confidence",
            "population_limitations",
            "validation_set_sha256",
            "provenance",
        }
    _keys(item, path, required=required)
    if kind == "tatbot.tissue-patch/1":
        for field in (
            "material_chart_sha256",
            "rest_surface_sha256",
            "deformed_surface_sha256",
            "support_sha256",
            "tool_geometry_sha256",
            "calibration_sha256",
            "batch_sha256",
        ):
            _digest(item[field], f"{path}.{field}")
        fields = _list(item["layer_prior_fields"], f"{path}.layer_prior_fields")
        if not fields:
            raise ContractError("wrong_length", f"{path}.layer_prior_fields", "at least one field is required")
        for index, raw in enumerate(fields):
            field = _obj(raw, f"{path}.layer_prior_fields[{index}]")
            _keys(
                field,
                f"{path}.layer_prior_fields[{index}]",
                required={"id", "role", "unit", "values_sha256", "uncertainty_sha256"},
            )
            _str(field["id"], f"{path}.layer_prior_fields[{index}].id")
            if field["role"] not in {"measured-layer", "generic-prior"}:
                raise ContractError("wrong_enum", f"{path}.layer_prior_fields[{index}].role", str(field["role"]))
            _str(field["unit"], f"{path}.layer_prior_fields[{index}].unit")
            _digest(field["values_sha256"], f"{path}.layer_prior_fields[{index}].values_sha256")
            _digest(field["uncertainty_sha256"], f"{path}.layer_prior_fields[{index}].uncertainty_sha256")
        if item["constitutive_model"] not in {
            "rigid-contact-v1",
            "compliant-heightfield-v1",
            "compliant-shell-v1",
            "volumetric-tissue-v1",
        }:
            raise ContractError("wrong_enum", f"{path}.constitutive_model", str(item["constitutive_model"]))
        parameters = _obj(item["parameters"], f"{path}.parameters")
        if not parameters:
            raise ContractError("wrong_length", f"{path}.parameters", "at least one parameter is required")
        for name, value in parameters.items():
            _str(name, f"{path}.parameters key")
            _parameter_estimate(value, f"{path}.parameters.{name}")
        uncertainty = _obj(item["uncertainty"], f"{path}.uncertainty")
        _keys(
            uncertainty,
            f"{path}.uncertainty",
            required={"method", "posterior_sha256", "out_of_distribution_policy_sha256", "confidence"},
        )
        _str(uncertainty["method"], f"{path}.uncertainty.method")
        _digest(uncertainty["posterior_sha256"], f"{path}.uncertainty.posterior_sha256")
        _digest(
            uncertainty["out_of_distribution_policy_sha256"],
            f"{path}.uncertainty.out_of_distribution_policy_sha256",
        )
        confidence = _num(uncertainty["confidence"], f"{path}.uncertainty.confidence")
        if not 0 <= confidence <= 1:
            raise ContractError("out_of_range", f"{path}.uncertainty.confidence", "expected [0,1]")
        for field in ("damping", "relaxation", "friction"):
            _parameter_estimate(item[field], f"{path}.{field}")
        boundary = _obj(item["boundary_conditions"], f"{path}.boundary_conditions")
        _keys(boundary, f"{path}.boundary_conditions", required={"kind", "definition_sha256"})
        _str(boundary["kind"], f"{path}.boundary_conditions.kind")
        _digest(boundary["definition_sha256"], f"{path}.boundary_conditions.definition_sha256")
        _trace_bindings(item["indentation"], f"{path}.indentation")
        _trace_bindings(item["force"], f"{path}.force")
    else:
        for field in ("source", "version", "license", "population_limitations"):
            _str(item[field], f"{path}.{field}")
        structures = _list(item["structure_ids"], f"{path}.structure_ids")
        if not structures:
            raise ContractError("wrong_length", f"{path}.structure_ids", "at least one structure is required")
        for index, structure in enumerate(structures):
            _str(structure, f"{path}.structure_ids[{index}]")
        _matrix4(item["atlas_from_body"], f"{path}.atlas_from_body")
        if item["warp"] is not None:
            warp = _obj(item["warp"], f"{path}.warp")
            _keys(warp, f"{path}.warp", required={"kind", "definition_sha256", "max_error_m"})
            _str(warp["kind"], f"{path}.warp.kind")
            _digest(warp["definition_sha256"], f"{path}.warp.definition_sha256")
            if _num(warp["max_error_m"], f"{path}.warp.max_error_m") < 0:
                raise ContractError("out_of_range", f"{path}.warp.max_error_m", "must be nonnegative")
        if _num(item["error_m"], f"{path}.error_m") < 0:
            raise ContractError("out_of_range", f"{path}.error_m", "must be nonnegative")
        confidence = _num(item["confidence"], f"{path}.confidence")
        if not 0 <= confidence <= 1:
            raise ContractError("out_of_range", f"{path}.confidence", "expected [0,1]")
        _digest(item["validation_set_sha256"], f"{path}.validation_set_sha256")
    _provenance(item["provenance"], f"{path}.provenance")


def _force_displacement_calibration(item: dict[str, Any], path: str) -> None:
    _keys(
        item,
        path,
        required={
            "schema",
            "content_sha256",
            "instrument_id",
            "calibration_trace_sha256",
            "resolution",
            "bias",
            "drift_per_hour",
            "synchronization_uncertainty_s",
            "temperature_reference_c",
            "temperature_sensitivity",
            "repeatability",
            "uncertainty",
            "qualification_basis",
            "qualified",
            "captured_utc",
            "provenance",
        },
    )
    _str(item["instrument_id"], f"{path}.instrument_id")
    _digest(item["calibration_trace_sha256"], f"{path}.calibration_trace_sha256")
    for field in ("resolution", "drift_per_hour", "repeatability", "uncertainty"):
        values = _obj(item[field], f"{path}.{field}")
        _keys(values, f"{path}.{field}", required={"force_n", "displacement_m"})
        for name in ("force_n", "displacement_m"):
            if _num(values[name], f"{path}.{field}.{name}") < 0:
                raise ContractError("out_of_range", f"{path}.{field}.{name}", "must be nonnegative")
    bias = _obj(item["bias"], f"{path}.bias")
    _keys(bias, f"{path}.bias", required={"force_n", "displacement_m"})
    _num(bias["force_n"], f"{path}.bias.force_n")
    _num(bias["displacement_m"], f"{path}.bias.displacement_m")
    if _num(item["synchronization_uncertainty_s"], f"{path}.synchronization_uncertainty_s") < 0:
        raise ContractError("out_of_range", f"{path}.synchronization_uncertainty_s", "must be nonnegative")
    _num(item["temperature_reference_c"], f"{path}.temperature_reference_c")
    sensitivity = _obj(item["temperature_sensitivity"], f"{path}.temperature_sensitivity")
    _keys(sensitivity, f"{path}.temperature_sensitivity", required={"force_n_per_c", "displacement_m_per_c"})
    _num(sensitivity["force_n_per_c"], f"{path}.temperature_sensitivity.force_n_per_c")
    _num(sensitivity["displacement_m_per_c"], f"{path}.temperature_sensitivity.displacement_m_per_c")
    if item["qualification_basis"] not in {"physical-reference", "synthetic-fixture-only", "unqualified"}:
        raise ContractError("wrong_enum", f"{path}.qualification_basis", str(item["qualification_basis"]))
    if not isinstance(item["qualified"], bool):
        raise ContractError("wrong_type", f"{path}.qualified", "expected boolean")
    if item["qualified"] and item["qualification_basis"] != "physical-reference":
        raise ContractError(
            "force_sensor_unqualified",
            f"{path}.qualification_basis",
            "only a physical reference calibration can qualify the instrument",
        )
    _utc_timestamp(item["captured_utc"], f"{path}.captured_utc")
    _provenance(item["provenance"], f"{path}.provenance")


def _anatomy_source_manifest(item: dict[str, Any], path: str) -> None:
    _keys(
        item,
        path,
        required={
            "schema",
            "content_sha256",
            "use_case",
            "source",
            "version",
            "pack_license_spdx",
            "objects",
            "separate_asset_root",
            "code_license_spdx",
            "provenance",
        },
    )
    for field in ("use_case", "source", "version", "pack_license_spdx", "separate_asset_root"):
        _str(item[field], f"{path}.{field}")
    if item["code_license_spdx"] != "Apache-2.0":
        raise ContractError("wrong_enum", f"{path}.code_license_spdx", str(item["code_license_spdx"]))
    objects = _list(item["objects"], f"{path}.objects")
    if not objects:
        raise ContractError("wrong_length", f"{path}.objects", "at least one source object required")
    seen = set()
    for index, raw in enumerate(objects):
        obj = _obj(raw, f"{path}.objects[{index}]")
        _keys(
            obj,
            f"{path}.objects[{index}]",
            required={
                "source_object_id",
                "source_uri",
                "sha256",
                "license_spdx",
                "attribution",
                "derivatives_permitted",
                "object_license_verified",
                "downloaded",
            },
        )
        identifier = _str(obj["source_object_id"], f"{path}.objects[{index}].source_object_id")
        if identifier in seen:
            raise ContractError("duplicate_value", f"{path}.objects[{index}].source_object_id", identifier)
        seen.add(identifier)
        for field in ("source_uri", "license_spdx", "attribution"):
            _str(obj[field], f"{path}.objects[{index}].{field}")
        _digest(obj["sha256"], f"{path}.objects[{index}].sha256")
        for field in ("derivatives_permitted", "object_license_verified", "downloaded"):
            if not isinstance(obj[field], bool):
                raise ContractError("wrong_type", f"{path}.objects[{index}].{field}", "expected boolean")
    _provenance(item["provenance"], f"{path}.provenance")


def _check_expected_topology(value: Any, expected: str, path: str = "$") -> None:
    if isinstance(value, dict):
        if {"topology_sha256", "face_index", "barycentric"} <= value.keys():
            actual = value["topology_sha256"]
            if actual != expected:
                raise ContractError("wrong_topology", f"{path}.topology_sha256", str(actual))
        for key, item in value.items():
            _check_expected_topology(item, expected, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_expected_topology(item, expected, f"{path}[{index}]")


def validate_contract(
    value: Any,
    *,
    expected_schema: str | None = None,
    expected_topology_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a complete contract and return it unchanged."""

    item = _obj(value, "$")
    schema = _str(item.get("schema"), "$.schema")
    if schema not in KNOWN_SCHEMAS:
        raise ContractError("unknown_schema", "$.schema", schema)
    if expected_schema is not None and schema != expected_schema:
        raise ContractError("wrong_schema", "$.schema", f"expected {expected_schema}, got {schema}")
    if expected_topology_sha256 is not None:
        _digest(expected_topology_sha256, "expected_topology_sha256")
        _check_expected_topology(item, expected_topology_sha256)
    _digest(item.get("content_sha256"), "$.content_sha256")
    handlers = {
        "tatbot.body-identity/1": _body_identity,
        "tatbot.body-state/1": _body_state,
        "tatbot.tattoo-program/1": _tattoo_program,
        "tatbot.surface-placement/1": _surface_placement,
        "tatbot.surface-placement/2": _target_placement,
        "tatbot.ink-program/1": _ink_program,
        "tatbot.ink-program/2": _ink_program,
        "tatbot.surface-registration/1": _surface_registration,
        "tatbot.execution-program/1": _execution_program,
    }
    if schema == "tatbot.surface-coordinate/1":
        _surface_coordinate(item, "$", top_level=True)
    elif schema in {"tatbot.surface-curve/1", "tatbot.surface-curve/2"}:
        _surface_curve(item, "$", top_level=True, chart_allowed=schema.endswith("/2"))
    elif schema in {"tatbot.tissue-patch/1", "tatbot.anatomy-registration/1"}:
        _simple_research_contract(item, "$", schema)
    elif schema == "tatbot.force-displacement-calibration/1":
        _force_displacement_calibration(item, "$")
    elif schema == "tatbot.anatomy-source-manifest/1":
        _anatomy_source_manifest(item, "$")
    elif schema == "tatbot.draw-samples-manifest/1":
        _draw_samples_manifest(item, "$")
    elif schema == "tatbot.body-model-spec/1":
        _body_model_spec(item, "$")
    else:
        handlers[schema](item, "$")
    actual = canonical_digest(item)
    if item["content_sha256"] != actual:
        raise ContractError(
            "wrong_hash",
            "$.content_sha256",
            f"declared {item['content_sha256']}, computed {actual}",
        )
    if schema == "tatbot.body-model-spec/1" and actual != REVIEWED_MODEL_SPEC_SHA256:
        raise ContractError(
            "body_model_unpinned",
            "$.content_sha256",
            f"expected reviewed spec {REVIEWED_MODEL_SPEC_SHA256}; got {actual}",
        )
    return item


def load_contract(
    path: str | Path,
    *,
    expected_schema: str | None = None,
    expected_topology_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and validate one contract from a UTF-8 JSON file."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ContractError("read_failed", str(source), str(exc)) from exc
    value = parse_json(raw)
    return validate_contract(
        value,
        expected_schema=expected_schema,
        expected_topology_sha256=expected_topology_sha256,
    )
