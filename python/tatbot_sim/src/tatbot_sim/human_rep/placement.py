"""Hard surface-address materialization for research and exact consumers."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from tatbot_sim.human_rep.contracts import (
    ContractError,
    canonical_digest,
    validate_contract,
)


@dataclass(frozen=True)
class HardenedPlacement:
    face_index: int
    barycentric: tuple[float, float, float]
    probability: float


def make_target_placement(*, target: dict[str, Any], tattoo_program_sha256: str,
                          physical_scale_m: Sequence[float], rotation_rad: float,
                          mirrored: bool, review: dict[str, Any], provenance: dict[str, Any],
                          warp: dict[str, Any] | None = None) -> dict[str, Any]:
    """One placement constructor for body and analytic target intent.

    Nominal chart dimensions and anchor coordinates are not robot calibration.
    Physical preparation must bind them to a fresh measured surface separately.
    """
    document = deepcopy({
        "schema": "tatbot.surface-placement/2", "content_sha256": "0" * 64,
        "tattoo_program_sha256": tattoo_program_sha256, "target": target,
        "physical_scale_m": list(physical_scale_m), "rotation_rad": rotation_rad,
        "mirrored": mirrored, "warp": warp, "review": review, "provenance": provenance,
    })
    document["content_sha256"] = canonical_digest(document)
    return validate_contract(document, expected_schema="tatbot.surface-placement/2")


def upgrade_body_placement(value: dict[str, Any]) -> dict[str, Any]:
    """Explicitly migrate a valid v1 binding without altering its source."""
    original = validate_contract(value, expected_schema="tatbot.surface-placement/1")
    fields = {"body_identity_sha256", "rest_surface_sha256", "semantic_site", "laterality",
              "anchor", "tangent_frame_rule", "supported_domain"}
    return make_target_placement(
        target={"kind": "body", **{key: original[key] for key in fields}},
        **{key: original[key] for key in original if key not in fields | {"schema", "content_sha256"}},
    )


def _normalized_barycentric(value: Sequence[float], path: str) -> tuple[float, float, float]:
    bary = np.asarray(value, dtype=np.float64)
    if bary.shape != (3,) or not np.isfinite(bary).all():
        raise ContractError("surface_coordinate_invalid", path, "expected three finite values")
    total = float(bary.sum())
    if abs(total - 1.0) > 1e-6 or np.any(bary < -1e-6) or np.any(bary > 1 + 1e-6):
        raise ContractError("surface_coordinate_invalid", path, "barycentric values are not normalized")
    bary = np.clip(bary, 0, 1)
    bary /= bary.sum()
    return (float(bary[0]), float(bary[1]), float(bary[2]))


def harden_face_distribution(
    face_indices: Sequence[int],
    probabilities: Sequence[float],
    barycentric: Sequence[Sequence[float]],
    *,
    supported_faces: Iterable[int],
) -> HardenedPlacement:
    """Choose exactly one supported face; ties resolve to the smallest face ID."""

    faces = np.asarray(face_indices)
    scores = np.asarray(probabilities, dtype=np.float64)
    bary = list(barycentric)
    if faces.ndim != 1 or len(faces) == 0 or scores.shape != faces.shape or len(bary) != len(faces):
        raise ContractError("surface_coordinate_invalid", "$.distribution", "face, probability, and barycentric lengths differ")
    if not np.issubdtype(faces.dtype, np.integer) or np.any(faces < 0):
        raise ContractError("surface_coordinate_invalid", "$.distribution.face_indices", "faces must be nonnegative integers")
    if not np.isfinite(scores).all() or np.any(scores < 0) or float(scores.sum()) <= 0:
        raise ContractError("surface_coordinate_invalid", "$.distribution.probabilities", "scores must be finite and nonnegative")
    allowed = {int(item) for item in supported_faces}
    candidates = [index for index, face in enumerate(faces) if int(face) in allowed]
    if not candidates:
        raise ContractError("anchor_outside_domain", "$.distribution", "no proposed face belongs to the supported domain")
    best = min(candidates, key=lambda index: (-float(scores[index]), int(faces[index]), index))
    return HardenedPlacement(
        face_index=int(faces[best]),
        barycentric=_normalized_barycentric(bary[best], f"$.distribution.barycentric[{best}]"),
        probability=float(scores[best] / scores.sum()),
    )


def make_surface_coordinate(
    *,
    topology_sha256: str,
    face_index: int,
    barycentric: Sequence[float],
) -> dict[str, Any]:
    document = {
        "schema": "tatbot.surface-coordinate/1",
        "content_sha256": "0" * 64,
        "topology_sha256": topology_sha256,
        "face_index": int(face_index),
        "barycentric": list(_normalized_barycentric(barycentric, "$.barycentric")),
    }
    document["content_sha256"] = canonical_digest(document)
    return validate_contract(document, expected_schema="tatbot.surface-coordinate/1")


def make_surface_placement(
    *,
    tattoo_program_sha256: str,
    body_identity_sha256: str,
    rest_surface_sha256: str,
    topology_sha256: str,
    semantic_site: str,
    laterality: str,
    anchor: HardenedPlacement | tuple[int, Sequence[float]],
    physical_scale_m: tuple[float, float],
    rotation_rad: float,
    mirrored: bool,
    supported_faces: Sequence[int],
    margin_m: float,
    review: dict[str, Any],
    provenance: dict[str, Any],
    tangent_frame_rule: str = "projected-body-up-then-oriented-normal",
    warp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(anchor, HardenedPlacement):
        face_index, barycentric = anchor.face_index, anchor.barycentric
    else:
        face_index, barycentric = anchor
    if int(face_index) not in {int(value) for value in supported_faces}:
        raise ContractError("anchor_outside_domain", "$.anchor.face_index", "not present in supported_faces")
    document = {
        "schema": "tatbot.surface-placement/1",
        "content_sha256": "0" * 64,
        "tattoo_program_sha256": tattoo_program_sha256,
        "body_identity_sha256": body_identity_sha256,
        "rest_surface_sha256": rest_surface_sha256,
        "semantic_site": semantic_site,
        "laterality": laterality,
        "anchor": {
            "topology_sha256": topology_sha256,
            "face_index": int(face_index),
            "barycentric": list(_normalized_barycentric(barycentric, "$.anchor.barycentric")),
        },
        "tangent_frame_rule": tangent_frame_rule,
        "physical_scale_m": [float(physical_scale_m[0]), float(physical_scale_m[1])],
        "rotation_rad": float(rotation_rad),
        "mirrored": bool(mirrored),
        "warp": deepcopy(warp),
        "supported_domain": {
            "face_indices": sorted({int(value) for value in supported_faces}),
            "margin_m": float(margin_m),
        },
        "review": deepcopy(review),
        "provenance": deepcopy(provenance),
    }
    if not math.isfinite(document["rotation_rad"]):
        raise ContractError("surface_coordinate_invalid", "$.rotation_rad", "must be finite")
    document["content_sha256"] = canonical_digest(document)
    return validate_contract(document, expected_schema="tatbot.surface-placement/1")


def make_surface_curve(
    *,
    topology_sha256: str,
    addresses: Sequence[tuple[int, Sequence[float]]],
    rest_points_m: Sequence[Sequence[float]],
    width_m: float,
    deposition: float,
    direction: str,
    source_primitive_sha256: str,
    compiler_sha256: str,
) -> dict[str, Any]:
    points = np.asarray(rest_points_m, dtype=np.float64)
    if points.shape != (len(addresses), 3) or len(points) < 2 or not np.isfinite(points).all():
        raise ContractError("surface_coordinate_invalid", "$.rest_points_m", "must be one finite xyz per address")
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    document = {
        "schema": "tatbot.surface-curve/1",
        "content_sha256": "0" * 64,
        "coordinates": [
            {
                "topology_sha256": topology_sha256,
                "face_index": int(face),
                "barycentric": list(_normalized_barycentric(bary, f"$.coordinates[{index}].barycentric")),
            }
            for index, (face, bary) in enumerate(addresses)
        ],
        "rest_surface_arc_length_m": length,
        "width_m": float(width_m),
        "deposition": float(deposition),
        "direction": direction,
        "source_primitive_sha256": source_primitive_sha256,
        "compiler_sha256": compiler_sha256,
    }
    document["content_sha256"] = canonical_digest(document)
    return validate_contract(document, expected_schema="tatbot.surface-curve/1")
