"""Strict single-path readers for SOMA-native placements and scenarios."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, NoReturn

from tatbot_sim.inkmap.gltf_surface import (
    MODEL_ID,
    MODEL_SPEC_SHA256,
    REST_SURFACE_SHA256,
    TOPOLOGY_SHA256,
)

PLACEMENT_CURRENT = 6
SCENARIO_CURRENT = 2
IDENTITY_SHA256 = "800babcf48d09f4ff1da9d2e8a0ebe266869a19849557450c7210c9a9fb9a1b6"
REST_ASSET_PATH = "bodies/mhr-soma-v1.glb"
REST_ASSET_SHA256 = "c1d0ec25c6f4708bb3e8d48811711da0d309598a6b85cf256975d49d1d46e841"
POSE_ASSET_SHA256 = "8270809809ca874efed77257c96eb6b36be3afe50d126c177b8fade44e7329d3"
FACE_COUNT = 36_108
_HEX = frozenset("0123456789abcdef")


class ContractError(ValueError):
    """A body-relative record is structurally unsafe or unsupported."""


def _fail(kind: str, message: str) -> NoReturn:
    raise ContractError(f"{kind}: {message}")


def _keys(value: dict, required: set[str], optional: set[str], where: str, kind: str) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        _fail(kind, f"{where} missing {', '.join(missing)}")
    if unknown:
        _fail(kind, f"{where} has unknown fields: {', '.join(unknown)}")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _digest(value: object, where: str, kind: str) -> str:
    if not _is_sha256(value):
        _fail(kind, f"{where} must be a lowercase sha256 hex digest")
    return str(value)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _anchor(value: object, where: str, kind: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(kind, f"{where} must be an object")
    _keys(value, {"face", "barycentric"}, set(), where, kind)
    face = value["face"]
    bary = value["barycentric"]
    if not isinstance(face, int) or isinstance(face, bool) or not 0 <= face < FACE_COUNT:
        _fail(kind, f"{where}.face must be in [0,{FACE_COUNT})")
    if not isinstance(bary, list) or len(bary) != 3 or not all(_finite(v) and -1e-6 <= v <= 1 + 1e-6 for v in bary):
        _fail(kind, f"{where}.barycentric must be three finite normalized weights")
    if abs(sum(bary) - 1.0) > 1e-6:
        _fail(kind, f"{where}.barycentric must sum to 1")
    return value


def _matrix4(value: object, where: str, kind: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        _fail(kind, f"{where} must be a row-major 4x4 matrix")
    if any(not isinstance(row, list) or len(row) != 4 or not all(_finite(v) for v in row) for row in value):
        _fail(kind, f"{where} must be a finite row-major 4x4 matrix")
    matrix = value
    if any(abs(float(a) - b) > 1e-9 for a, b in zip(matrix[3], [0, 0, 0, 1], strict=True)):
        _fail(kind, f"{where} must be an affine transform")
    return matrix


def canonical_json_bytes(document: object) -> bytes:
    """The canonical JSON byte representation used at this boundary."""

    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def document_sha256(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _body_binding(body: object, *, scenario: bool, kind: str) -> dict[str, Any]:
    if not isinstance(body, dict):
        _fail(kind, "body binding is required")
    required = {
        "model_spec_id", "model_spec_sha256", "identity_sha256", "topology_sha256",
        "rest_surface_sha256", "asset_path", "asset_sha256",
    }
    if scenario:
        required.add("pose_asset_sha256")
    _keys(body, required, set(), "body", kind)
    expected = {
        "model_spec_id": MODEL_ID,
        "model_spec_sha256": MODEL_SPEC_SHA256,
        "identity_sha256": IDENTITY_SHA256,
        "topology_sha256": TOPOLOGY_SHA256,
        "rest_surface_sha256": REST_SURFACE_SHA256,
        "asset_path": REST_ASSET_PATH,
        "asset_sha256": REST_ASSET_SHA256,
    }
    if scenario:
        expected["pose_asset_sha256"] = POSE_ASSET_SHA256
    if body != expected:
        _fail(kind, "unsupported schema/model")
    return body


def _placement_record(value: object, where: str, kind: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(kind, f"{where} must be an object")
    _keys(
        value,
        {"id", "design_id", "anchor", "rotation_rad", "size_mm", "mirror"},
        {"source_sha256", "site", "language"},
        where,
        kind,
    )
    if not isinstance(value["id"], str) or not value["id"] or not isinstance(value["design_id"], str) or not value["design_id"]:
        _fail(kind, f"{where} needs non-empty string id and design_id")
    _anchor(value["anchor"], f"{where}.anchor", kind)
    if not _finite(value["rotation_rad"]):
        _fail(kind, f"{where}.rotation_rad must be finite")
    size = value["size_mm"]
    if not isinstance(size, list) or len(size) != 2 or not all(_finite(v) and v > 0 for v in size):
        _fail(kind, f"{where}.size_mm must be positive [width,height]")
    if not isinstance(value["mirror"], bool):
        _fail(kind, f"{where}.mirror must be boolean")
    if "source_sha256" in value:
        _digest(value["source_sha256"], f"{where}.source_sha256", kind)
    return value


def _designs(value: object, kind: str) -> dict[str, Any]:
    from tatbot_sim.inkmap.artwork import validate_artwork_record

    if value is None:
        return {}
    if not isinstance(value, dict):
        _fail(kind, "designs must be an object keyed by design id")
    for design_id, artwork in value.items():
        if not isinstance(design_id, str) or not design_id:
            _fail(kind, "design entries need nonempty IDs")
        validate_artwork_record(artwork)
    return value


def validate_placement(document: object) -> dict[str, Any]:
    kind = "placement file"
    if not isinstance(document, dict):
        _fail(kind, "not an object")
    if document.get("schema_version") != PLACEMENT_CURRENT:
        _fail(kind, "unsupported schema/model")
    _keys(document, {"schema_version", "units", "body", "placements"}, {"designs"}, "$", kind)
    if document["units"] != {"length": "m", "tattoo_size": "mm", "up": "+z"}:
        _fail(kind, "units must be {length:m, tattoo_size:mm, up:+z}")
    _body_binding(document["body"], scenario=False, kind=kind)
    placements = document["placements"]
    if not isinstance(placements, list):
        _fail(kind, "placements must be an array")
    for index, placement in enumerate(placements):
        _placement_record(placement, f"placements[{index}]", kind)
    designs = _designs(document.get("designs"), kind)
    for placement in placements:
        if placement["design_id"].startswith("gen-") and placement["design_id"] not in designs:
            _fail(kind, f"generated design {placement['design_id']} is not embedded")
        language = placement.get("language")
        if language is None:
            continue
        if not isinstance(language, dict) or not isinstance(language.get("sentence"), str) or not isinstance(language.get("program"), dict):
            _fail(kind, "placement language needs sentence and program")
        resolution = language.get("resolution")
        if resolution is not None:
            if not isinstance(resolution, dict) or resolution.get("status") != "resolved" or resolution.get("anchor") != placement["anchor"]:
                _fail(kind, "placement language resolution is inconsistent")
            resolved_body = resolution.get("body")
            comparable = {
                key: document["body"][key]
                for key in (
                    "model_spec_id", "model_spec_sha256", "identity_sha256",
                    "topology_sha256", "rest_surface_sha256", "asset_sha256",
                )
            }
            if resolved_body != comparable:
                _fail(kind, "placement language resolution body differs from placement body")
    return document


def validate_scenario(document: object) -> dict[str, Any]:
    kind = "tattoo scenario"
    if not isinstance(document, dict):
        _fail(kind, "not an object")
    if document.get("schema_version") == 3:
        from tatbot_sim.inkmap.program_scenario import validate_program_scenario
        return validate_program_scenario(document)
    if document.get("schema_version") != SCENARIO_CURRENT:
        _fail(kind, f"schema_version must be {SCENARIO_CURRENT}")
    _keys(
        document,
        {"schema_version", "units", "seed", "body", "pose", "placement", "design", "trace", "robot", "support", "provenance"},
        {"placement_optimization"},
        "$",
        kind,
    )
    if document["units"] != {"length": "m", "tattoo_size": "mm", "angle": "rad", "up": "+z", "matrix_order": "row-major"}:
        _fail(kind, "units/frame contract mismatch")
    seed = document["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        _fail(kind, "seed must be a non-negative integer")
    _body_binding(document["body"], scenario=True, kind=kind)

    pose = document["pose"]
    if not isinstance(pose, dict):
        _fail(kind, "pose is required")
    _keys(pose, {"id", "catalog_sha256", "source", "posed_surface_sha256", "world_from_body"}, set(), "pose", kind)
    if not isinstance(pose["id"], str) or not pose["id"] or pose["source"] != "named":
        _fail(kind, "pose identity/source is invalid")
    _digest(pose["catalog_sha256"], "pose.catalog_sha256", kind)
    _digest(pose["posed_surface_sha256"], "pose.posed_surface_sha256", kind)
    _matrix4(pose["world_from_body"], "pose.world_from_body", kind)

    placement = _placement_record(document["placement"], "placement", kind)
    if "source_sha256" not in placement:
        _fail(kind, "placement.source_sha256 is required")
    design = document["design"]
    if not isinstance(design, dict):
        _fail(kind, "design is incomplete")
    _keys(design, {"id", "name", "svg", "sha256", "source"}, set(), "design", kind)
    if design["id"] != placement["design_id"] or not isinstance(design["name"], str) or not isinstance(design["svg"], str) or "<svg" not in design["svg"] or not isinstance(design["source"], dict):
        _fail(kind, "design is incomplete or differs from placement")
    _digest(design["sha256"], "design.sha256", kind)

    trace = document["trace"]
    if not isinstance(trace, dict):
        _fail(kind, "trace is incomplete")
    _keys(trace, {"compiler", "compiler_version", "sha256", "strokes"}, set(), "trace", kind)
    if trace["compiler"] != "tatbot_sim.surface_trace" or trace["compiler_version"] != 2:
        _fail(kind, "trace compiler must be tatbot_sim.surface_trace version 2")
    _digest(trace["sha256"], "trace.sha256", kind)
    strokes = trace["strokes"]
    if not isinstance(strokes, list) or not strokes:
        _fail(kind, "trace.strokes must be non-empty")
    for index, stroke in enumerate(strokes):
        if not isinstance(stroke, list) or len(stroke) < 2:
            _fail(kind, f"trace.strokes[{index}] needs at least two anchors")
        for point, anchor in enumerate(stroke):
            _anchor(anchor, f"trace.strokes[{index}][{point}]", kind)

    robot = document["robot"]
    if not isinstance(robot, dict):
        _fail(kind, "robot is incomplete")
    _keys(robot, {"urdf_sha256", "tool_id", "world_from_robot"}, set(), "robot", kind)
    _digest(robot["urdf_sha256"], "robot.urdf_sha256", kind)
    if not isinstance(robot["tool_id"], str) or not robot["tool_id"]:
        _fail(kind, "robot.tool_id is required")
    _matrix4(robot["world_from_robot"], "robot.world_from_robot", kind)

    support = document["support"]
    if not isinstance(support, dict):
        _fail(kind, "support is incomplete")
    _keys(support, {"id"}, {"world_from_nominal"}, "support", kind)
    if not isinstance(support["id"], str) or not support["id"]:
        _fail(kind, "support.id is required")
    if "world_from_nominal" in support:
        _matrix4(support["world_from_nominal"], "support.world_from_nominal", kind)

    optimization = document.get("placement_optimization")
    if optimization is not None:
        if not isinstance(optimization, dict) or optimization.get("schema") != "tatbot.placement-search/1":
            _fail(kind, "placement_optimization is incomplete")
        if not isinstance(optimization.get("candidates"), list) or not optimization["candidates"]:
            _fail(kind, "placement_optimization.candidates must be non-empty")
        selected = optimization.get("selected")
        if not isinstance(selected, dict):
            _fail(kind, "placement_optimization.selected is required")
        _matrix4(selected.get("world_from_body"), "placement_optimization.selected.world_from_body", kind)
        _matrix4(selected.get("world_from_support"), "placement_optimization.selected.world_from_support", kind)

    provenance = document["provenance"]
    if not isinstance(provenance, dict):
        _fail(kind, "provenance is incomplete")
    _keys(provenance, {"created_at", "git_sha", "generator"}, set(), "provenance", kind)
    if not all(isinstance(provenance[key], str) and provenance[key] for key in provenance):
        _fail(kind, "provenance values must be non-empty strings")
    return document


def _load(path: str | Path) -> Any:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        return json.load(stream)


def load_placement(path: str | Path) -> dict[str, Any]:
    return validate_placement(_load(path))


def load_scenario(path: str | Path) -> dict[str, Any]:
    return validate_scenario(_load(path))
