"""Deterministic synthetic registration observability and offline replay evidence.

The suite sweeps the pinned reference identity, selected real MHR coefficient
cases, every named pose, every currently supported site, camera aspect,
occlusion, and sensor noise.  Synthetic results are diagnostic only: they do
not manufacture the separately required physical instrument repeatability or
non-human phantom baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tatbot_contracts.canonical import write_json
from tatbot_contracts.digest import sha256_file

from tatbot_sim.human_rep.contracts import load_contract
from tatbot_sim.human_rep.registration import (
    ContractError,
    fit_rigid_landmarks,
    nominal_patch_hypotheses,
    select_separated_hypothesis,
    transform_error,
    weak_centroid_fit,
)
from tatbot_sim.inkmap.rig import BodyRig, load_body_rig
from tatbot_sim.repo import git_output, repo_root

SEED = 9_042_026
SUPPORTED_SITES = ("bicep", "calf", "forearm", "shin", "thigh", "tricep")
VIEWS = ("normal", "oblique", "grazing")
OCCLUSIONS = (0.0, 0.4)
NOISE_SIGMAS_M = (0.0, 0.0001, 0.0003)
SYNTHETIC_TARGETS = {
    # A local patch cannot tightly observe the remote whole-body origin.  Gate
    # target-registration error at the patch centroid; retain the body-origin
    # translation below as a REPORT metric.  The correspondence bound is just
    # above sqrt(3)*0.3 mm, the largest injected per-axis sensor sigma.
    "local_translation_error_m": 0.0005,
    "rotation_error_deg": 1.0,
    "surface_point_rms_m": 0.0005,
    "surface_normal_error_deg": 1.0,
    "correspondence_rms_m": 0.00065,
    "path_sensitivity_rms_m": 0.00075,
}


@dataclass(frozen=True)
class SurfaceVariant:
    identifier: str
    axis: str
    triangles_m: np.ndarray
    source_sha256: str


def _rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle_rad) * skew + (1 - math.cos(angle_rad)) * (skew @ skew)


def _truth_transform(case_index: int, view: str) -> np.ndarray:
    angles = {"normal": 8.0, "oblique": 19.0, "grazing": 31.0}
    axis = np.asarray([0.25 + (case_index % 3) * 0.1, -0.55, 0.72], dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = _rotation(axis, math.radians(angles[view]))
    matrix[:3, 3] = [0.36 + (case_index % 5) * 0.004, -0.03, 0.16 + (case_index % 7) * 0.003]
    return matrix


def _transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def _face_normals(triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
    return normals


def _site_face_ids(atlas: dict[str, Any], site: str, laterality: str = "right") -> np.ndarray:
    site_index = atlas["sites"].index(site)
    laterality_code = {"left": 1, "right": 2}[laterality]
    result = np.flatnonzero(np.asarray(atlas["faces"], dtype=np.int64) == site_index * 4 + laterality_code)
    if len(result) < 12:
        raise ValueError(f"{site}:{laterality} has only {len(result)} mapped faces")
    return result


def _site_patch_faces(atlas: dict[str, Any], triangles: np.ndarray, site: str, count: int = 72) -> np.ndarray:
    candidates = _site_face_ids(atlas, site)
    anchor = atlas["regions"][f"{site}:right"]["default_anchor"]
    anchor_face = int(anchor["face"])
    anchor_point = np.asarray(anchor["barycentric"], dtype=np.float64) @ triangles[anchor_face]
    centers = triangles[candidates].mean(axis=1)
    order = np.lexsort((candidates, np.linalg.norm(centers - anchor_point, axis=1)))
    nearest = candidates[order[: min(len(order), max(count * 3, count))]]
    if len(nearest) <= count:
        return np.sort(nearest)
    positions = np.linspace(0, len(nearest) - 1, count, dtype=np.int64)
    return np.sort(nearest[positions])


def _farthest_points(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return np.arange(len(points), dtype=np.int64)
    center = points.mean(axis=0)
    chosen = [int(np.argmax(np.linalg.norm(points - center, axis=1)))]
    distance = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < count:
        candidate = int(np.argmax(distance))
        chosen.append(candidate)
        distance = np.minimum(distance, np.linalg.norm(points - points[candidate], axis=1))
    return np.asarray(chosen, dtype=np.int64)


def _visible_indices(points: np.ndarray, normals: np.ndarray, view: str, occlusion: float) -> np.ndarray:
    normal = normals.mean(axis=0)
    normal /= np.linalg.norm(normal)
    centered = points - points.mean(axis=0)
    _, _, right_t = np.linalg.svd(centered, full_matrices=False)
    tangent = right_t[0] - np.dot(right_t[0], normal) * normal
    tangent /= max(np.linalg.norm(tangent), 1e-20)
    direction = {
        "normal": normal,
        "oblique": normal + 0.8 * tangent,
        "grazing": 0.2 * normal + tangent,
    }[view]
    direction /= np.linalg.norm(direction)
    facing = normals @ direction
    order = np.lexsort((np.arange(len(points)), -facing))
    front = order[: max(12, int(math.ceil(len(points) * 0.8)))]
    # A contiguous image-side crop models occlusion without leaking random row order.
    projection = centered[front] @ tangent
    front = front[np.argsort(projection, kind="stable")]
    remove = int(math.floor(len(front) * occlusion))
    remaining = front[remove:]
    if len(remaining) < 8:
        raise ContractError("registration_missing", "$.synthetic_view", "fewer than eight visible landmarks")
    return remaining


def _surface_variant_from_triangles(identifier: str, axis: str, triangles: np.ndarray, source: str) -> SurfaceVariant:
    value = np.asarray(triangles, dtype=np.float64)
    if value.shape != (36_108, 3, 3) or not np.isfinite(value).all():
        raise ValueError(f"{identifier}: expected finite (36108,3,3), got {value.shape}")
    return SurfaceVariant(identifier, axis, value, source)


def cached_reference_variants(rig: BodyRig | None = None) -> list[SurfaceVariant]:
    """Pinned reference identity plus every checked-in exact named-pose surface."""

    body = rig or load_body_rig()
    variants = [
        _surface_variant_from_triangles("reference", "identity", body.rest_vertices, body.surface_sha256)
    ]
    for index, pose_id in enumerate(body.pose_ids):
        variants.append(
            _surface_variant_from_triangles(
                pose_id,
                "pose",
                body.pose_vertices[index],
                body.catalog_record["poses"][pose_id]["surface_sha256"],
            )
        )
    return variants


def mhr_identity_variants(
    cache_dir: str | Path,
    *,
    spec_path: str | Path,
    identity_path: str | Path,
) -> list[SurfaceVariant]:
    """Generate selected genuine bounded MHR identities through pinned SOMA-X."""

    from tatbot_sim.body_models.mhr_soma import SOMAPosedBody  # noqa: PLC0415

    body = SOMAPosedBody(spec_path=spec_path, cache_dir=cache_dir, device="cpu")
    reference = load_contract(identity_path, expected_schema="tatbot.body-identity/1")
    cases = {
        "mhr-coefficient-00-plus-1sigma": ([1.0] + [0.0] * 44, [0.0] * 68),
        "mhr-mixed-alternating-2sigma": (
            [2.0 if index % 2 == 0 else -2.0 for index in range(45)],
            [0.0] * 68,
        ),
        "mhr-scale-groups-mixed": (
            [0.0] * 45,
            [0.1 if (index // 8) % 2 == 0 else -0.1 for index in range(68)],
        ),
    }
    rotations = np.zeros((77, 4), dtype=np.float64)
    rotations[:, 3] = 1.0
    result = []
    for identifier, (coefficients, scales) in cases.items():
        identity = copy.deepcopy(reference)
        identity["coefficients"] = coefficients
        identity["scales"] = scales
        surface = body.generate(identity, rotations)
        result.append(
            _surface_variant_from_triangles(
                identifier,
                "identity",
                surface.face_vertices_m,
                surface.surface_sha256,
            )
        )
    return result


def _normal_error_deg(estimated: np.ndarray, truth: np.ndarray, normals: np.ndarray) -> float:
    left = normals @ estimated[:3, :3].T
    right = normals @ truth[:3, :3].T
    cosine = np.clip(np.einsum("ij,ij->i", left, right), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).max())


def _fit_metrics(
    fit,
    truth: np.ndarray,
    surface_points: np.ndarray,
    normals: np.ndarray,
) -> dict[str, Any]:
    error = transform_error(fit.observed_patch_from_body, truth)
    predicted = _transform(fit.observed_patch_from_body, surface_points)
    expected = _transform(truth, surface_points)
    point_error = np.linalg.norm(predicted - expected, axis=1)
    center = np.asarray(surface_points, dtype=np.float64).mean(axis=0, keepdims=True)
    local_translation_error = float(
        np.linalg.norm(_transform(fit.observed_patch_from_body, center) - _transform(truth, center))
    )
    translation_sigma = math.sqrt(max(0.0, float(np.max(np.diag(fit.covariance)[:3]))))
    metrics = {
        **fit.metrics(),
        **error,
        "local_translation_error_m": local_translation_error,
        "surface_point_rms_m": math.sqrt(float(np.mean(point_error * point_error))),
        "surface_point_max_m": float(point_error.max()),
        "surface_normal_error_deg": _normal_error_deg(fit.observed_patch_from_body, truth, normals),
        "correspondence_rms_m": fit.rms_error_m,
        "path_sensitivity_rms_m": math.sqrt(float(np.mean(point_error * point_error))),
        "translation_three_sigma_covered": bool(error["translation_error_m"] <= 3 * translation_sigma + 1e-12),
    }
    metrics["synthetic_target_pass"] = all(
        metrics[name] <= limit for name, limit in SYNTHETIC_TARGETS.items()
    )
    return metrics


def run_synthetic_observability(
    variants: Sequence[SurfaceVariant],
    *,
    atlas: Mapping[str, Any],
    sites: Sequence[str] = SUPPORTED_SITES,
    views: Sequence[str] = VIEWS,
    occlusions: Sequence[float] = OCCLUSIONS,
    noise_sigmas_m: Sequence[float] = NOISE_SIGMAS_M,
    seed: int = SEED,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate direct, nominal-assisted, and weak registration by all axes."""

    if not variants:
        raise ValueError("at least one surface variant is required")
    atlas_dict = dict(atlas)
    reference = next((item for item in variants if item.identifier == "reference"), variants[0])
    cases = []
    refusals: dict[str, int] = {}
    representative: dict[str, np.ndarray] = {}
    case_index = 0
    for variant in variants:
        for site in sites:
            face_ids = _site_patch_faces(atlas_dict, variant.triangles_m, site)
            source = variant.triangles_m[face_ids].mean(axis=1)
            normals = _face_normals(variant.triangles_m[face_ids])
            reference_points = reference.triangles_m[face_ids].mean(axis=1)
            for view in views:
                for occlusion in occlusions:
                    visible = _visible_indices(source, normals, view, float(occlusion))
                    landmark_local = visible[_farthest_points(source[visible], min(48, len(visible)))]
                    for noise_sigma in noise_sigmas_m:
                        case_seed = seed + case_index
                        rng = np.random.default_rng(case_seed)
                        truth = _truth_transform(case_index, view)
                        noiseless = _transform(truth, source)
                        observed = noiseless + rng.normal(0.0, float(noise_sigma), noiseless.shape)
                        record: dict[str, Any] = {
                            "id": f"case-{case_index:05d}",
                            "seed": case_seed,
                            "identity_or_pose": variant.identifier,
                            "variant_axis": variant.axis,
                            "variant_surface_sha256": variant.source_sha256,
                            "site": site,
                            "laterality": "right",
                            "view": view,
                            "occlusion_fraction": float(occlusion),
                            "noise_sigma_m": float(noise_sigma),
                            "visible_points": int(len(visible)),
                            "landmarks": int(len(landmark_local)),
                            "methods": {},
                        }
                        try:
                            direct = fit_rigid_landmarks(
                                source[landmark_local],
                                observed[landmark_local],
                                sensor_sigma_m=float(noise_sigma),
                                minimum_observability=0.005,
                            )
                            record["methods"]["direct_local"] = {
                                "status": "accepted",
                                **_fit_metrics(direct, truth, source[visible], normals[visible]),
                            }
                            if not representative:
                                representative = {
                                    "body_triangles_m": variant.triangles_m,
                                    "face_ids": face_ids,
                                    "observed_points_m": observed[visible],
                                    "landmarks_m": observed[landmark_local],
                                    "truth": truth,
                                    "estimate": direct.observed_patch_from_body,
                                    "covariance": direct.covariance,
                                }
                        except ContractError as exc:
                            refusals[exc.code] = refusals.get(exc.code, 0) + 1
                            record["methods"]["direct_local"] = {"status": "refused", "code": exc.code, "detail": exc.detail}

                        weak = weak_centroid_fit(source[landmark_local], observed[landmark_local])
                        record["methods"]["weak"] = {
                            "status": "diagnostic",
                            **_fit_metrics(weak, truth, source[visible], normals[visible]),
                        }

                        centroid_seed = np.eye(4)
                        centroid_seed[:3, 3] = observed[visible].mean(axis=0) - reference_points[visible].mean(axis=0)
                        alternate_seed = centroid_seed.copy()
                        local_normal = normals[visible].mean(axis=0)
                        alternate_seed[:3, :3] = _rotation(local_normal, math.pi)
                        try:
                            hypotheses = nominal_patch_hypotheses(
                                reference_points[visible],
                                observed[visible],
                                [("centroid", centroid_seed), ("half-turn", alternate_seed)],
                                trim_fraction=0.2,
                            )
                            hypothesis_records = [item.metrics() for item in hypotheses]
                            try:
                                selected = select_separated_hypothesis(hypotheses)
                                selection = {"status": "separated", "seed_id": selected.seed_id}
                            except ContractError as exc:
                                selection = {"status": "ambiguous", "code": exc.code, "detail": exc.detail}
                            best = hypotheses[0].fit
                            record["methods"]["nominal_assisted"] = {
                                "status": "diagnostic_only",
                                "selection": selection,
                                "hypotheses": hypothesis_records,
                                **_fit_metrics(best, truth, source[visible], normals[visible]),
                            }
                        except ContractError as exc:
                            refusals[exc.code] = refusals.get(exc.code, 0) + 1
                            record["methods"]["nominal_assisted"] = {
                                "status": "refused",
                                "code": exc.code,
                                "detail": exc.detail,
                                "hypotheses": [],
                            }
                        cases.append(record)
                        case_index += 1

    direct = [
        item["methods"]["direct_local"]
        for item in cases
        if item["methods"]["direct_local"]["status"] == "accepted"
    ]
    coverage = [bool(item["translation_three_sigma_covered"]) for item in direct]
    by_site = {}
    for site in sites:
        selected = [
            item["methods"]["direct_local"]
            for item in cases
            if item["site"] == site and item["methods"]["direct_local"]["status"] == "accepted"
        ]
        by_site[site] = {
            "cases": len(selected),
            "translation_error_m_max": max((item["translation_error_m"] for item in selected), default=None),
            "local_translation_error_m_max": max(
                (item["local_translation_error_m"] for item in selected), default=None
            ),
            "rotation_error_deg_max": max((item["rotation_error_deg"] for item in selected), default=None),
            "surface_point_rms_m_max": max((item["surface_point_rms_m"] for item in selected), default=None),
            "surface_normal_error_deg_max": max((item["surface_normal_error_deg"] for item in selected), default=None),
            "correspondence_rms_m_max": max((item["correspondence_rms_m"] for item in selected), default=None),
            "path_sensitivity_rms_m_max": max((item["path_sensitivity_rms_m"] for item in selected), default=None),
            "synthetic_target_pass": bool(selected) and all(item["synthetic_target_pass"] for item in selected),
        }
    result = {
        "schema": "tatbot.registration-synthetic-suite/1",
        "seed": seed,
        "axes": {
            "identity_or_pose": [item.identifier for item in variants],
            "variant_axis": sorted({item.axis for item in variants}),
            "site": list(sites),
            "laterality": ["right"],
            "view": list(views),
            "occlusion_fraction": [float(item) for item in occlusions],
            "noise_sigma_m": [float(item) for item in noise_sigmas_m],
        },
        "targets": SYNTHETIC_TARGETS,
        "target_authority": "preregistered synthetic diagnostics only; physical release bounds remain pending instrument baseline",
        "cases": cases,
        "summary": {
            "total_cases": len(cases),
            "direct_accepted": len(direct),
            "direct_refused": len(cases) - len(direct),
            "direct_target_pass": bool(direct) and all(item["synthetic_target_pass"] for item in direct),
            "translation_three_sigma_coverage": float(np.mean(coverage)) if coverage else None,
            "refusals": refusals,
            "by_site": by_site,
        },
    }
    return result, representative


def _synthetic_baseline(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    truth = np.asarray([0.0, 0.0, 0.0])
    samples = truth + rng.normal(0.00002, 0.0001, (100, 3))
    repeats = np.linalg.norm(samples - samples.mean(axis=0), axis=1)
    return {
        "schema": "tatbot.registration-repeatability-baseline/1",
        "synthetic_sensor": {
            "seed": seed,
            "samples": len(samples),
            "injected_resolution_m": 0.00001,
            "injected_bias_m": 0.00002,
            "injected_noise_sigma_m": 0.0001,
            "repeatability_rms_m": math.sqrt(float(np.mean(repeats * repeats))),
            "repeatability_p95_m": float(np.quantile(repeats, 0.95)),
        },
        "physical_instrument": {
            "status": "pending_no_collection_authority",
            "instrument_id": None,
            "resolution_m": None,
            "bias_m": None,
            "drift_m": None,
            "synchronization_s": None,
            "temperature_sensitivity": None,
            "repeatability_m": None,
            "calibration_uncertainty_m": None,
        },
        "release_threshold_status": "not_selected_pending_physical_instrument_baseline",
    }


def _surface_mesh(surface_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import surface_model  # noqa: PLC0415

    surface = surface_model.HeightFieldSurface.from_npz(surface_path)
    rows, cols = np.asarray(surface.count).shape
    u = np.linspace(-surface.width_m / 2, surface.width_m / 2, cols)
    v = np.linspace(-surface.height_m / 2, surface.height_m / 2, rows)
    vv, uu = np.meshgrid(v, u, indexing="ij")
    points, _, _, normals = surface.frame(np.stack([uu.ravel(), vv.ravel()], axis=1))
    index = np.arange(rows * cols).reshape(rows, cols)
    faces = np.concatenate(
        [
            np.stack([index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel()], axis=1),
            np.stack([index[:-1, :-1].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()], axis=1),
        ]
    )
    return points, faces, normals


def _replay_bundle(
    output: Path,
    representative: Mapping[str, np.ndarray],
    *,
    surface_path: Path,
    execution_program_path: Path | None,
    samples_path: Path | None,
    created_utc: str,
) -> dict[str, Any]:
    triangles = np.asarray(representative["body_triangles_m"], dtype=np.float64)
    matrix = np.asarray(representative["estimate"], dtype=np.float64)
    body_vertices = _transform(matrix, triangles.reshape(-1, 3))
    body_faces = np.arange(len(body_vertices), dtype=np.uint32).reshape(-1, 3)
    surface_points, surface_faces, surface_normals = _surface_mesh(surface_path)
    hashes: dict[str, str] = {"measured_surface_sha256": sha256_file(surface_path)}
    exact_path = np.empty((0, 3), dtype=np.float64)
    placed_art = np.empty((0, 3), dtype=np.float64)
    if execution_program_path is not None:
        execution = load_contract(execution_program_path, expected_schema="tatbot.execution-program/1")
        hashes["execution_program_sha256"] = execution["content_sha256"]
        hashes["surface_registration_sha256"] = execution["surface_registration"]["content_sha256"]
    if samples_path is not None:
        import arm_kinematics  # noqa: PLC0415
        import pen_path  # noqa: PLC0415

        samples, _ = pen_path.read_samples_csv(samples_path)
        exact_path = arm_kinematics.root_from_base(np.asarray(samples.p, dtype=np.float64))
        placed_art = exact_path[np.asarray(samples.pen) > 0]
        hashes["samples_sha256"] = sha256_file(samples_path)
    bundle_path = output / "replay-input.npz"
    created = datetime.strptime(created_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    np.savez(
        bundle_path,
        body_vertices_m=body_vertices.astype(np.float32),
        body_faces=body_faces,
        observed_points_m=np.asarray(representative["observed_points_m"], dtype=np.float32),
        landmarks_m=np.asarray(representative["landmarks_m"], dtype=np.float32),
        measured_vertices_m=surface_points.astype(np.float32),
        measured_faces=surface_faces.astype(np.uint32),
        measured_normals=surface_normals.astype(np.float32),
        placed_art_m=placed_art.astype(np.float32),
        exact_path_m=exact_path.astype(np.float32),
        observed_patch_from_body=matrix,
        translation_sigma_m=np.float64(
            math.sqrt(max(0.0, float(np.max(np.diag(np.asarray(representative["covariance"]))[:3]))))
        ),
        capture_time_ns=np.int64(round(created.timestamp() * 1e9)),
        hashes_json=np.str_(json.dumps(hashes, sort_keys=True, separators=(",", ":"))),
    )
    return {
        "path": str(bundle_path),
        "sha256": sha256_file(bundle_path),
        "body_points": int(len(body_vertices)),
        "observed_points": int(len(representative["observed_points_m"])),
        "exact_path_points": int(len(exact_path)),
        "hashes": hashes,
    }


def run_evidence(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evidence directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "registrations").mkdir()
    atlas = json.loads(args.atlas.read_text())
    rig = load_body_rig()
    variants = cached_reference_variants(rig)
    variants.extend(
        mhr_identity_variants(
            args.cache_dir,
            spec_path=args.spec,
            identity_path=args.identity,
        )
    )
    suite, representative = run_synthetic_observability(variants, atlas=atlas, seed=args.seed)
    write_json(output / "synthetic-suite.json", suite)
    baseline = _synthetic_baseline(args.seed)
    write_json(output / "baseline.json", baseline)

    direct_cases = [
        case for case in suite["cases"] if case["methods"]["direct_local"]["status"] == "accepted"
    ]
    write_json(output / "registrations" / "direct-local-representative.json", direct_cases[0])
    assisted_cases = [
        case for case in suite["cases"] if case["methods"]["nominal_assisted"]["status"] == "diagnostic_only"
    ]
    write_json(output / "registrations" / "nominal-hypotheses-representative.json", assisted_cases[0])
    uncertainty = {
        "schema": "tatbot.registration-uncertainty-evaluation/1",
        "cases": suite["summary"]["direct_accepted"],
        "translation_three_sigma_coverage": suite["summary"]["translation_three_sigma_coverage"],
        "covariance_is_execution_bound": True,
        "physical_calibration_status": "pending_no_collection_authority",
    }
    write_json(output / "uncertainty.json", uncertainty)
    path_sensitivity = {
        "schema": "tatbot.registration-path-sensitivity/1",
        "by_site": {
            site: values["path_sensitivity_rms_m_max"]
            for site, values in suite["summary"]["by_site"].items()
        },
        "target_m": SYNTHETIC_TARGETS["path_sensitivity_rms_m"],
        "status": "pass_synthetic" if suite["summary"]["direct_target_pass"] else "fail_synthetic",
    }
    write_json(output / "path-sensitivity.json", path_sensitivity)
    preflight = {
        "schema": "tatbot.registration-preflight/1",
        "mode": "offline-no-arm",
        "motion_authorized": False,
        "emission_authorized": False,
        "human_contact_authorized": False,
        "observed_cells_only": True,
        "hole_behavior": "structured_refusal",
        "stale_capture_behavior": "structured_refusal",
        "stale_calibration_behavior": "structured_refusal",
        "partial_fit_execution_authority": False,
        "production_method": "direct-phantom-landmark-rigid-v1",
    }
    write_json(output / "preflight.json", preflight)
    bundle = _replay_bundle(
        output,
        representative,
        surface_path=args.surface,
        execution_program_path=args.execution_program,
        samples_path=args.samples,
        created_utc=args.created_utc,
    )
    if args.write_replay:
        from tatbot_sim.human_rep.registration_replay import write_registration_replay  # noqa: PLC0415

        replay = write_registration_replay(bundle["path"], output / "replay.rrd")
        write_json(output / "replay.json", replay)

    output_names = [
        "synthetic-suite.json",
        "baseline.json",
        "uncertainty.json",
        "path-sensitivity.json",
        "preflight.json",
        "replay-input.npz",
        *( ["replay.rrd", "replay.json"] if args.write_replay else [] ),
    ]
    manifest = {
        "schema": "tatbot.capability-evidence/1",
        "capability": "registration",
        "evidence_type": "synthetic",
        "consumer": "offline registration diagnostics",
        "unresolved_dependency": "qualified instrument and held-out non-human phantom landmarks",
        "next_experiment": "measure stationary phantom registration error and uncertainty",
        "hardware_authority": False,
        "status": "pending_instrument_and_nonhuman_phantom_baseline",
        "created_utc": args.created_utc,
        "plan": None,
        "base_git_sha": git_output("rev-parse", "HEAD"),
        "dirty_state": git_output("status", "--short"),
        "remote_comparison": git_output("rev-list", "--left-right", "--count", "origin/main...HEAD"),
        "command": sys.argv,
        "runtime": {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu": platform.processor(),
            "gpu": "not_used_not_assigned",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "inputs": {
            "model_spec": {"path": str(args.spec), "sha256": sha256_file(args.spec)},
            "identity": {"path": str(args.identity), "sha256": sha256_file(args.identity)},
            "atlas": {"path": str(args.atlas), "sha256": sha256_file(args.atlas)},
            "surface": {"path": str(args.surface), "sha256": sha256_file(args.surface)},
            "cache": {"path": str(args.cache_dir), "offline_cache_verification_required": True},
        },
        "outputs": {
            name: {"sha256": sha256_file(output / name), "size": (output / name).stat().st_size}
            for name in output_names
        },
        "accepted": {
            "synthetic_observability": suite["summary"]["direct_target_pass"],
            "direct_local_production_method": True,
            "nominal_assisted_execution_authority": False,
            "offline_rerun_replay": bool(args.write_replay),
            "physical_instrument_baseline": False,
            "approved_nonhuman_phantom_transfer": False,
            "physical_registration_accepted": False,
            "motion": False,
        },
        "counts": {
            "accepted": suite["summary"]["direct_accepted"],
            "refused": suite["summary"]["direct_refused"],
            "refusals": suite["summary"]["refusals"],
        },
        "known_gaps": [
            "A qualified physical instrument repeatability baseline was not authorized or collected.",
            "A separately approved non-human phantom landmark transfer was not available.",
            "Synthetic diagnostic thresholds are not release thresholds and do not substitute for either baseline.",
        ],
        "next_dependency": "operator-authorized qualified-instrument and non-human phantom protocol",
        "replay_bundle": bundle,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=root / "config" / "body-models" / "mhr-soma-v1.json")
    parser.add_argument("--identity", type=Path, default=root / "config" / "body-models" / "mhr-soma-v1" / "identity.json")
    parser.add_argument("--atlas", type=Path, default=root / "web" / "inkmap" / "public" / "bodies" / "mhr-soma-v1.regions.json")
    parser.add_argument("--surface", type=Path, default=root / "config" / "human-representation" / "examples" / "current-surface.npz")
    parser.add_argument("--execution-program", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--created-utc", default=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    parser.add_argument("--write-replay", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_evidence(args)
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest["accepted"]["synthetic_observability"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
