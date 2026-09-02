"""Strict readers for the Inkmap placement and posed-scenario boundaries.

The JSON Schemas are the public authority.  These dependency-free readers are
the runtime gate used before NumPy, ManiSkill, or a robot model sees a record.
Older placement versions remain readable, but all newly written files are v4.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, NoReturn

PLACEMENT_VERSIONS = (1, 2, 3, 4)
PLACEMENT_CURRENT = 4
SCENARIO_CURRENT = 1
_HEX = frozenset("0123456789abcdef")


class ContractError(ValueError):
    """A record is structurally unsafe or ambiguous."""


def _fail(kind: str, message: str) -> NoReturn:
    raise ContractError(f"{kind}: {message}")


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
    face = value.get("face")
    bary = value.get("barycentric")
    if not isinstance(face, int) or isinstance(face, bool) or face < 0:
        _fail(kind, f"{where}.face must be a non-negative integer")
    if not isinstance(bary, list) or len(bary) != 3 or not all(_finite(v) and 0 <= v <= 1 for v in bary):
        _fail(kind, f"{where}.barycentric must be three finite weights in [0,1]")
    if abs(sum(bary) - 1.0) > 1e-6:
        _fail(kind, f"{where}.barycentric must sum to 1")
    return value


def _matrix4(value: object, where: str, kind: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        _fail(kind, f"{where} must be a row-major 4x4 matrix")
    if any(not isinstance(row, list) or len(row) != 4 or not all(_finite(v) for v in row) for row in value):
        _fail(kind, f"{where} must be a finite row-major 4x4 matrix")
    return value


def canonical_json_bytes(document: object) -> bytes:
    """The only JSON digest representation used by scenario provenance."""

    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def document_sha256(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def validate_placement(document: object) -> dict[str, Any]:
    kind = "placement file"
    if not isinstance(document, dict):
        _fail(kind, "not an object")
    version = document.get("schema_version")
    if version not in PLACEMENT_VERSIONS:
        _fail(kind, f"schema_version {version!r} not in {PLACEMENT_VERSIONS}")
    units = document.get("units")
    if units != {"length": "m", "tattoo_size": "mm", "up": "+z"}:
        _fail(kind, "units must be {length:m, tattoo_size:mm, up:+z}")
    body = document.get("body")
    if not isinstance(body, dict) or not isinstance(body.get("id"), str) or not body["id"] or not isinstance(body.get("path"), str) or not body["path"]:
        _fail(kind, "body needs non-empty id and path")
    if version >= 4:
        _digest(body.get("asset_sha256"), "body.asset_sha256", kind)
        _digest(body.get("surface_sha256"), "body.surface_sha256", kind)
    else:
        _digest(body.get("sha256"), "body.sha256", kind)
    placements = document.get("placements")
    if not isinstance(placements, list):
        _fail(kind, "placements must be an array")
    for i, placement in enumerate(placements):
        where = f"placements[{i}]"
        if not isinstance(placement, dict) or not isinstance(placement.get("id"), str) or not isinstance(placement.get("design_id"), str):
            _fail(kind, f"{where} needs string id and design_id")
        _anchor(placement.get("anchor"), f"{where}.anchor", kind)
        if not _finite(placement.get("rotation_rad")):
            _fail(kind, f"{where}.rotation_rad must be finite")
        size = placement.get("size_mm")
        if not isinstance(size, list) or len(size) != 2 or not all(_finite(v) and v > 0 for v in size):
            _fail(kind, f"{where}.size_mm must be positive [width,height]")
        if not isinstance(placement.get("mirror"), bool):
            _fail(kind, f"{where}.mirror must be boolean")
    designs = document.get("designs", {})
    if not isinstance(designs, dict):
        _fail(kind, "designs must be an object keyed by design id")
    for design_id, design in designs.items():
        if not isinstance(design, dict) or not isinstance(design.get("name"), str) or not isinstance(design.get("svg"), str) or "<svg" not in design["svg"]:
            _fail(kind, f"designs[{design_id}] needs name and svg text")
    for placement in placements:
        design_id = placement["design_id"]
        if design_id.startswith("gen-") and design_id not in designs:
            _fail(kind, f"placement {placement['id']} references generated design {design_id} that is not embedded")
    return document


def validate_scenario(document: object) -> dict[str, Any]:
    kind = "tattoo scenario"
    if not isinstance(document, dict):
        _fail(kind, "not an object")
    if document.get("schema_version") != SCENARIO_CURRENT:
        _fail(kind, f"schema_version must be {SCENARIO_CURRENT}")
    if document.get("units") != {"length": "m", "tattoo_size": "mm", "angle": "rad", "up": "+z", "matrix_order": "row-major"}:
        _fail(kind, "units/frame contract mismatch")
    seed = document.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        _fail(kind, "seed must be a non-negative integer")
    body = document.get("body")
    if not isinstance(body, dict) or not all(isinstance(body.get(k), str) and body[k] for k in ("id", "path", "rig_id")):
        _fail(kind, "body identity is incomplete")
    for field in ("asset_sha256", "surface_sha256", "rig_sha256"):
        _digest(body.get(field), f"body.{field}", kind)
    pose = document.get("pose")
    if not isinstance(pose, dict) or not isinstance(pose.get("id"), str) or pose.get("source") not in ("named", "tracked"):
        _fail(kind, "pose identity/source is invalid")
    _digest(pose.get("catalog_sha256"), "pose.catalog_sha256", kind)
    _matrix4(pose.get("world_from_body"), "pose.world_from_body", kind)
    joints = pose.get("joint_rotations")
    if not isinstance(joints, dict):
        _fail(kind, "pose.joint_rotations must be an object")
    for bone, quat in joints.items():
        if not isinstance(quat, list) or len(quat) != 4 or not all(_finite(v) for v in quat):
            _fail(kind, f"pose.joint_rotations.{bone} must be finite [x,y,z,w]")
    placement = document.get("placement")
    if not isinstance(placement, dict) or not all(isinstance(placement.get(k), str) and placement[k] for k in ("id", "design_id")):
        _fail(kind, "placement is incomplete")
    _digest(placement.get("source_sha256"), "placement.source_sha256", kind)
    _anchor(placement.get("anchor"), "placement.anchor", kind)
    if not _finite(placement.get("rotation_rad")) or not isinstance(placement.get("mirror"), bool):
        _fail(kind, "placement transform is incomplete")
    size = placement.get("size_mm")
    if not isinstance(size, list) or len(size) != 2 or not all(_finite(v) and v > 0 for v in size):
        _fail(kind, "placement.size_mm must be positive [width,height]")
    design = document.get("design")
    if not isinstance(design, dict) or not all(isinstance(design.get(k), str) and design[k] for k in ("id", "name", "svg")) or "<svg" not in design["svg"] or not isinstance(design.get("source"), dict):
        _fail(kind, "design is incomplete")
    _digest(design.get("sha256"), "design.sha256", kind)
    trace = document.get("trace")
    if not isinstance(trace, dict) or trace.get("compiler") != "tatbot_sim.surface_trace" or not isinstance(trace.get("compiler_version"), int):
        _fail(kind, "trace compiler is incomplete")
    _digest(trace.get("sha256"), "trace.sha256", kind)
    strokes = trace.get("strokes")
    if not isinstance(strokes, list) or not strokes:
        _fail(kind, "trace.strokes must be a non-empty array")
    for i, stroke in enumerate(strokes):
        if not isinstance(stroke, list) or len(stroke) < 2:
            _fail(kind, f"trace.strokes[{i}] needs at least two anchors")
        for j, value in enumerate(stroke):
            _anchor(value, f"trace.strokes[{i}][{j}]", kind)
    robot = document.get("robot")
    if not isinstance(robot, dict) or not isinstance(robot.get("tool_id"), str) or not robot["tool_id"]:
        _fail(kind, "robot is incomplete")
    _digest(robot.get("urdf_sha256"), "robot.urdf_sha256", kind)
    _matrix4(robot.get("world_from_robot"), "robot.world_from_robot", kind)
    support = document.get("support")
    if not isinstance(support, dict) or not isinstance(support.get("id"), str) or not support["id"]:
        _fail(kind, "support.id is required")
    provenance = document.get("provenance")
    if not isinstance(provenance, dict) or not all(isinstance(provenance.get(k), str) and provenance[k] for k in ("created_at", "git_sha", "generator")):
        _fail(kind, "provenance is incomplete")
    return document


def _load(path: str | Path) -> Any:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        return json.load(stream)


def load_placement(path: str | Path) -> dict[str, Any]:
    return validate_placement(_load(path))


def load_scenario(path: str | Path) -> dict[str, Any]:
    return validate_scenario(_load(path))
