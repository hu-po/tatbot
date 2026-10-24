"""Strict reader for v6 body placements bound to the reference SOMA surface.

A placement addresses the body by canonical face and barycentric point. The
shared helpers here (body binding, anchors, affine matrices) are the one
implementation every body-relative record reader uses.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, NoReturn

from tatbot_surface.artwork import validate_artwork_record
from tatbot_surface.gltf_surface import (
    MODEL_ID,
    MODEL_SPEC_SHA256,
    REST_SURFACE_SHA256,
    TOPOLOGY_SHA256,
)

PLACEMENT_CURRENT = 6
IDENTITY_SHA256 = "ed4b000260e5e5598ea00848b6f664a828a241e1791afa292e12b8db54d1a993"
REST_ASSET_PATH = "bodies/mhr-soma-v1.glb"
REST_ASSET_SHA256 = "e0431426524852e6e9822b9fef8401537da8887796aed5521f1abfccb4a9a716"
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


def _load(path: str | Path) -> Any:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        return json.load(stream)


def load_placement(path: str | Path) -> dict[str, Any]:
    return validate_placement(_load(path))
