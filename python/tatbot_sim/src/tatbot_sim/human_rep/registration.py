"""Observed-patch registration for the deterministic human-representation path.

The production baseline in this module is deliberately local and explicit: a
set of named phantom landmarks/fiducials supplies canonical SOMA surface
coordinates and observed 3-D points, and a bounded rigid fit maps the bound
``BodyState`` surface into the measured patch frame.  Missing observations are
never filled with nominal body vertices.

Nominal-patch ICP is provided only as an offline research comparator.  It
returns every seeded hypothesis and requires an explicit score gap before a
caller can select one; the exact execution bridge continues to consume the
direct-landmark result.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from tatbot_contracts.digest import sha256_file

from tatbot_sim.human_rep.contracts import ContractError, canonical_digest, validate_contract
from tatbot_sim.inkmap.rig import BodyRig, load_body_rig

SCHEMA = "tatbot.surface-registration/1"
DIRECT_METHOD = "direct-phantom-landmark-rigid-v1"
ASSISTED_METHOD = "nominal-patch-multihypothesis-icp-v1"
WEAK_METHOD = "centroid-translation-v1"


@dataclass(frozen=True)
class RigidFit:
    """One bounded rigid fit and its uncertainty diagnostics."""

    method: str
    observed_patch_from_body: np.ndarray
    covariance: np.ndarray
    errors_m: np.ndarray
    rms_error_m: float
    max_error_m: float
    confidence: float
    observability: float
    condition_number: float

    def metrics(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "rms_error_m": self.rms_error_m,
            "max_error_m": self.max_error_m,
            "confidence": self.confidence,
            "observability": self.observability,
            "condition_number": self.condition_number,
            "translation_sigma_m": float(
                math.sqrt(max(0.0, float(np.max(np.diag(self.covariance)[:3]))))
            ),
            "rotation_sigma_rad": float(
                math.sqrt(max(0.0, float(np.max(np.diag(self.covariance)[3:]))))
            ),
        }


@dataclass(frozen=True)
class RegistrationHypothesis:
    """One preserved nominal-patch hypothesis; never execution authority."""

    seed_id: str
    fit: RigidFit
    chamfer_rms_m: float
    iterations: int

    def metrics(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "chamfer_rms_m": self.chamfer_rms_m,
            "iterations": self.iterations,
            **self.fit.metrics(),
        }


def _rigid_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def _rotation_angle(rotation: np.ndarray) -> float:
    return math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = value
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _point_arrays(source: Any, target: Any, path: str = "$.correspondences") -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(source, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.ndim != 2 or left.shape[1:] != (3,) or right.shape != left.shape:
        raise ContractError("registration_missing", path, "source and observed points need matching (N,3) arrays")
    if len(left) < 3:
        raise ContractError("registration_missing", path, "at least three landmarks are required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ContractError("registration_missing", path, "landmarks contain non-finite values")
    return left, right


def fit_rigid_landmarks(
    canonical_points_m: Any,
    observed_points_m: Any,
    *,
    weights: Sequence[float] | None = None,
    sensor_sigma_m: float = 0.0001,
    max_rms_error_m: float = 0.002,
    max_translation_m: float = 3.0,
    max_rotation_rad: float = math.pi,
    minimum_observability: float = 0.02,
    method: str = DIRECT_METHOD,
) -> RigidFit:
    """Weighted Kabsch fit with geometry, residual, and covariance checks."""

    source, target = _point_arrays(canonical_points_m, observed_points_m)
    if not math.isfinite(sensor_sigma_m) or sensor_sigma_m < 0:
        raise ContractError("registration_uncertain", "$.sensor_sigma_m", "must be finite and nonnegative")
    if not math.isfinite(max_rms_error_m) or max_rms_error_m <= 0:
        raise ContractError("registration_uncertain", "$.max_rms_error_m", "must be finite and positive")
    if weights is None:
        weight = np.ones(len(source), dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.shape != (len(source),) or not np.isfinite(weight).all() or np.any(weight <= 0):
            raise ContractError("registration_missing", "$.correspondences.weights", "weights must be positive and finite")
    weight /= weight.sum()
    source_center = np.sum(weight[:, None] * source, axis=0)
    target_center = np.sum(weight[:, None] * target, axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    singular_geometry = np.linalg.svd(np.sqrt(weight[:, None]) * centered_source, compute_uv=False)
    if singular_geometry[0] <= 1e-9 or singular_geometry[1] <= 1e-9:
        raise ContractError("registration_ambiguous", "$.correspondences", "landmarks are coincident or collinear")
    observability = float(min(1.0, singular_geometry[1] / singular_geometry[0]))
    if observability < minimum_observability:
        raise ContractError(
            "registration_ambiguous",
            "$.correspondences",
            f"landmark observability {observability:.6g} is below {minimum_observability:.6g}",
        )

    cross_covariance = (weight[:, None] * centered_target).T @ centered_source
    left, _, right_t = np.linalg.svd(cross_covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    translation = target_center - rotation @ source_center
    if float(np.linalg.norm(translation)) > max_translation_m:
        raise ContractError("registration_uncertain", "$.observed_patch_from_body", "translation exceeds the bounded fit")
    if _rotation_angle(rotation) > max_rotation_rad + 1e-12:
        raise ContractError("registration_uncertain", "$.observed_patch_from_body", "rotation exceeds the bounded fit")

    matrix = _rigid_matrix(rotation, translation)
    predicted = _transform(matrix, source)
    errors = np.linalg.norm(predicted - target, axis=1)
    rms = math.sqrt(float(np.sum(weight * errors * errors)))
    maximum = float(errors.max())

    # Linearized point residual with local translation/rotation coordinates.
    rows = []
    for point, scalar_weight in zip(predicted - translation, weight, strict=True):
        rows.append(math.sqrt(float(scalar_weight)) * np.concatenate([np.eye(3), -_skew(point)], axis=1))
    jacobian = np.concatenate(rows, axis=0)
    information = jacobian.T @ jacobian
    condition = float(np.linalg.cond(information))
    residual_variance = max(sensor_sigma_m**2, float(np.sum(weight * errors * errors)) / max(1, 3 * len(source) - 6))
    covariance = residual_variance * np.linalg.pinv(information, rcond=1e-12)
    covariance = (covariance + covariance.T) / 2.0
    if not np.isfinite(covariance).all():
        raise ContractError("registration_uncertain", "$.covariance", "linearized covariance is not finite")

    residual_score = max(0.0, 1.0 - (rms / max_rms_error_m) ** 2)
    geometry_score = min(1.0, observability / max(minimum_observability * 5.0, 1e-12))
    confidence = float(np.clip(residual_score * geometry_score, 0.0, 1.0))
    fit = RigidFit(method, matrix, covariance, errors, rms, maximum, confidence, observability, condition)
    if rms > max_rms_error_m:
        raise ContractError(
            "registration_uncertain",
            "$.correspondences",
            f"landmark RMS {rms:.6g} m exceeds {max_rms_error_m:.6g} m",
        )
    return fit


def weak_centroid_fit(canonical_points_m: Any, observed_points_m: Any) -> RigidFit:
    """Intentionally weak translation-only comparator used by the synthetic registration suite."""

    source, target = _point_arrays(canonical_points_m, observed_points_m)
    translation = target.mean(axis=0) - source.mean(axis=0)
    matrix = _rigid_matrix(np.eye(3), translation)
    errors = np.linalg.norm(_transform(matrix, source) - target, axis=1)
    rms = math.sqrt(float(np.mean(errors * errors)))
    variance = max(rms * rms, 1e-12)
    covariance = np.eye(6) * variance
    return RigidFit(
        WEAK_METHOD,
        matrix,
        covariance,
        errors,
        rms,
        float(errors.max()),
        0.0,
        0.0,
        math.inf,
    )


def state_surface_vertices(body_state: dict[str, Any], rig: BodyRig | None = None) -> np.ndarray:
    """Return expanded face vertices for the exact named surface in BodyState."""

    body = rig or load_body_rig()
    state = validate_contract(body_state, expected_schema="tatbot.body-state/1")
    if state["body_identity_sha256"] != body.identity_sha256:
        raise ContractError("execution_binding_mismatch", "$.body_state.body_identity_sha256", "body identity differs")
    pose_id = state["named_pose"]
    if pose_id in {"canonical-t", "rest", None}:
        if pose_id is None:
            raise ContractError(
                "registration_missing",
                "$.body_state.tracked_source",
                "tracked joint state requires an explicitly materialized posed surface",
            )
        vertices = np.asarray(body.rest_vertices, dtype=np.float64)
        expected_surface = body.surface_sha256
    else:
        try:
            index = body.pose_ids.index(str(pose_id))
        except ValueError as exc:
            raise ContractError("pose_unsupported", "$.body_state.named_pose", str(pose_id)) from exc
        vertices = np.asarray(body.pose_vertices[index], dtype=np.float64)
        expected_surface = body.catalog_record["poses"][pose_id]["surface_sha256"]
    if state["posed_surface_sha256"] != expected_surface:
        raise ContractError(
            "execution_binding_mismatch",
            "$.body_state.posed_surface_sha256",
            f"named surface digest is {expected_surface}",
        )
    return vertices


def coordinate_points(
    coordinates: Iterable[dict[str, Any]],
    body_state: dict[str, Any],
    *,
    rig: BodyRig | None = None,
) -> np.ndarray:
    """Materialize persistent coordinates on the BodyState's posed surface."""

    body = rig or load_body_rig()
    vertices = state_surface_vertices(body_state, body)
    points = []
    for index, coordinate in enumerate(coordinates):
        if coordinate.get("topology_sha256") != body.topology_sha256:
            raise ContractError("wrong_topology", f"$.correspondences[{index}].canonical.topology_sha256", "not the sole body topology")
        face = coordinate.get("face_index")
        bary = np.asarray(coordinate.get("barycentric"), dtype=np.float64)
        if not isinstance(face, int) or isinstance(face, bool) or not 0 <= face < len(vertices):
            raise ContractError("surface_coordinate_invalid", f"$.correspondences[{index}].canonical.face_index", str(face))
        if bary.shape != (3,) or not np.isfinite(bary).all() or np.any(bary < -1e-8) or not np.isclose(bary.sum(), 1.0, atol=1e-6):
            raise ContractError("surface_coordinate_invalid", f"$.correspondences[{index}].canonical.barycentric", "not normalized")
        points.append(bary @ vertices[face])
    if not points:
        raise ContractError("registration_missing", "$.correspondences", "no canonical landmarks")
    return np.asarray(points, dtype=np.float64)


def create_surface_registration(
    body_state: dict[str, Any],
    measured_surface_path: str | Path,
    correspondences: Sequence[dict[str, Any]],
    *,
    capture_sha256: str,
    calibration_sha256: str,
    provenance: dict[str, Any],
    supported_cells: Iterable[int] | None = None,
    sensor_sigma_m: float = 0.0001,
    max_rms_error_m: float = 0.002,
    minimum_confidence: float = 0.95,
    rig: BodyRig | None = None,
) -> tuple[dict[str, Any], RigidFit]:
    """Fit and construct a strict direct-landmark ``SurfaceRegistration/1``."""

    body = rig or load_body_rig()
    state = validate_contract(body_state, expected_schema="tatbot.body-state/1")
    if not correspondences:
        raise ContractError("registration_missing", "$.correspondences", "no landmarks supplied")
    canonical = [deepcopy(item["canonical"]) for item in correspondences]
    observed = np.asarray([item["observed_xyz_m"] for item in correspondences], dtype=np.float64)
    weights = [float(item.get("weight", 1.0)) for item in correspondences]
    source = coordinate_points(canonical, state, rig=body)
    fit = fit_rigid_landmarks(
        source,
        observed,
        weights=weights,
        sensor_sigma_m=sensor_sigma_m,
        max_rms_error_m=max_rms_error_m,
    )
    if fit.confidence < minimum_confidence:
        raise ContractError(
            "registration_uncertain",
            "$.confidence",
            f"fit confidence {fit.confidence:.6g} is below {minimum_confidence:.6g}",
        )

    surface_path = Path(measured_surface_path)
    if supported_cells is None:
        try:
            with np.load(surface_path, allow_pickle=False) as payload:
                if str(payload["schema"]) != "tatbot.surface/1":
                    raise ContractError("measured_surface_stale", "$.measured_surface", "wrong surface schema")
                cells = np.flatnonzero(np.asarray(payload["count"]).reshape(-1) > 0)
        except (OSError, KeyError, ValueError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError("measured_surface_stale", "$.measured_surface", str(exc)) from exc
    else:
        cells = np.asarray(sorted({int(value) for value in supported_cells}), dtype=np.int64)
    if not len(cells):
        raise ContractError("measured_surface_incomplete", "$.supported_cells", "surface has no observed cells")

    normalized_correspondences = [
        {
            "canonical": coordinate,
            "observed_xyz_m": observed[index].astype(float).tolist(),
            "error_m": float(fit.errors_m[index]),
        }
        for index, coordinate in enumerate(canonical)
    ]
    document = {
        "schema": SCHEMA,
        "content_sha256": "0" * 64,
        "source_frame": "body",
        "target_frame": "observed_patch",
        "body_state_sha256": state["content_sha256"],
        "measured_surface_sha256": sha256_file(surface_path),
        "method": DIRECT_METHOD,
        "correspondences": normalized_correspondences,
        "observed_patch_from_body": fit.observed_patch_from_body.astype(float).tolist(),
        "supported_cells": cells.astype(int).tolist(),
        "covariance": fit.covariance.reshape(-1).astype(float).tolist(),
        "confidence": fit.confidence,
        "capture_sha256": capture_sha256,
        "calibration_sha256": calibration_sha256,
        "provenance": deepcopy(provenance),
    }
    document["content_sha256"] = canonical_digest(document)
    return validate_contract(document, expected_schema=SCHEMA), fit


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    squared = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    indices = np.argmin(squared, axis=1)
    return indices, squared[np.arange(len(source)), indices]


def nominal_patch_hypotheses(
    canonical_patch_m: Any,
    observed_patch_m: Any,
    initial_hypotheses: Sequence[tuple[str, Any]],
    *,
    max_iterations: int = 30,
    trim_fraction: float = 0.2,
) -> tuple[RegistrationHypothesis, ...]:
    """Return, never silently collapse, seeded nominal-body ICP hypotheses."""

    source, target = _point_arrays(canonical_patch_m, observed_patch_m, "$.nominal_patch")
    if not initial_hypotheses:
        raise ContractError("registration_missing", "$.initial_hypotheses", "at least one seed is required")
    if not 0 <= trim_fraction < 0.8:
        raise ContractError("registration_uncertain", "$.trim_fraction", "expected [0,0.8)")
    results = []
    keep = max(3, int(math.ceil(len(source) * (1.0 - trim_fraction))))
    for seed_id, raw_matrix in initial_hypotheses:
        matrix = np.asarray(raw_matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ContractError("registration_ambiguous", "$.initial_hypotheses", f"{seed_id} is not a finite matrix4")
        previous = math.inf
        iteration_count = 0
        for _iteration in range(1, max_iterations + 1):
            iteration_count = _iteration
            transformed = _transform(matrix, source)
            nearest, squared = _nearest_indices(transformed, target)
            selected = np.argsort(squared, kind="stable")[:keep]
            delta = fit_rigid_landmarks(
                transformed[selected],
                target[nearest[selected]],
                max_rms_error_m=max(1.0, math.sqrt(float(squared[selected].mean())) * 2.0),
                minimum_observability=1e-4,
                method=ASSISTED_METHOD,
            )
            matrix = delta.observed_patch_from_body @ matrix
            score = math.sqrt(float(squared[selected].mean()))
            if abs(previous - score) <= 1e-10:
                break
            previous = score
        transformed = _transform(matrix, source)
        nearest, squared = _nearest_indices(transformed, target)
        chamfer = math.sqrt(float(np.mean(squared)))
        final = fit_rigid_landmarks(
            source,
            target[nearest],
            max_rms_error_m=max(1.0, chamfer * 2.0),
            minimum_observability=1e-4,
            method=ASSISTED_METHOD,
        )
        # Keep the iterated transform; final provides covariance/error scaling.
        errors = np.sqrt(squared)
        final = RigidFit(
            ASSISTED_METHOD,
            matrix,
            final.covariance,
            errors,
            chamfer,
            float(errors.max()),
            final.confidence,
            final.observability,
            final.condition_number,
        )
        results.append(RegistrationHypothesis(str(seed_id), final, chamfer, iteration_count))
    return tuple(sorted(results, key=lambda item: (item.chamfer_rms_m, item.seed_id)))


def select_separated_hypothesis(
    hypotheses: Sequence[RegistrationHypothesis],
    *,
    minimum_score_gap_m: float = 0.00025,
) -> RegistrationHypothesis:
    """Select only when the best nominal hypothesis is observably separated."""

    if not hypotheses:
        raise ContractError("registration_missing", "$.hypotheses", "no hypotheses")
    ranked = sorted(hypotheses, key=lambda item: (item.chamfer_rms_m, item.seed_id))
    if len(ranked) > 1:
        gap = ranked[1].chamfer_rms_m - ranked[0].chamfer_rms_m
        if gap < minimum_score_gap_m:
            raise ContractError(
                "registration_ambiguous",
                "$.hypotheses",
                f"best score gap {gap:.6g} m is below {minimum_score_gap_m:.6g} m",
            )
    return ranked[0]


def rotation_error_deg(estimated: np.ndarray, truth: np.ndarray) -> float:
    relative = np.asarray(estimated)[:3, :3] @ np.asarray(truth)[:3, :3].T
    return math.degrees(_rotation_angle(relative))


def transform_error(estimated: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "translation_error_m": float(np.linalg.norm(np.asarray(estimated)[:3, 3] - np.asarray(truth)[:3, 3])),
        "rotation_error_deg": rotation_error_deg(np.asarray(estimated), np.asarray(truth)),
    }
