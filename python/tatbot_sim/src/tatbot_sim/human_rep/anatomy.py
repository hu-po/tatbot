"""Frozen generic anatomy-prior provenance and SOMA registration interfaces.

An anatomy pack is a separately licensed research visualization layer.  It
cannot establish an individual's structure locations, site safety, diagnosis,
pain, compliance, or execution acceptance.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Sequence

import numpy as np

from tatbot_sim.human_rep.contracts import ContractError, canonical_digest, validate_contract
from tatbot_sim.human_rep.registration import fit_rigid_landmarks

SOURCE_SCHEMA = "tatbot.anatomy-source-manifest/1"
REGISTRATION_SCHEMA = "tatbot.anatomy-registration/1"


def make_source_manifest(
    *,
    use_case: str,
    source: str,
    version: str,
    pack_license_spdx: str,
    objects: Sequence[dict[str, Any]],
    separate_asset_root: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    document = {
        "schema": SOURCE_SCHEMA,
        "content_sha256": "0" * 64,
        "use_case": use_case,
        "source": source,
        "version": version,
        "pack_license_spdx": pack_license_spdx,
        "objects": [deepcopy(item) for item in objects],
        "separate_asset_root": separate_asset_root,
        "code_license_spdx": "Apache-2.0",
        "provenance": deepcopy(provenance),
    }
    document["content_sha256"] = canonical_digest(document)
    return validate_contract(document, expected_schema=SOURCE_SCHEMA)


def preflight_source_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Resolve object-level license readiness without downloading anything."""

    source = validate_contract(manifest, expected_schema=SOURCE_SCHEMA)
    gaps = []
    downloaded = []
    for obj in source["objects"]:
        if not obj["object_license_verified"]:
            gaps.append(f"{obj['source_object_id']}: object license not verified")
        if not obj["derivatives_permitted"]:
            gaps.append(f"{obj['source_object_id']}: derivative permission absent")
        if obj["license_spdx"] in {"UNKNOWN", "NOASSERTION"}:
            gaps.append(f"{obj['source_object_id']}: unresolved SPDX license")
        if obj["downloaded"]:
            downloaded.append(obj["source_object_id"])
    if downloaded:
        gaps.append(f"objects marked downloaded before preflight: {','.join(downloaded)}")
    return {
        "schema": "tatbot.anatomy-source-preflight/1",
        "manifest_sha256": source["content_sha256"],
        "objects": len(source["objects"]),
        "object_license_ready": not gaps,
        "download_ready": not gaps,
        "download_authorized": False,
        "downloaded_objects": downloaded,
        "gaps": gaps,
        "separate_from_apache_code_and_body_assets": bool(source["separate_asset_root"]),
        "use_case": source["use_case"],
    }


def fit_anatomy_registration(
    manifest: dict[str, Any],
    *,
    structure_ids: Sequence[str],
    body_landmarks_m: Any,
    atlas_landmarks_m: Any,
    validation_body_landmarks_m: Any,
    validation_atlas_landmarks_m: Any,
    population_limitations: str,
    provenance: dict[str, Any],
    max_error_m: float = 0.005,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rigid generic-prior alignment with an independent held-out landmark set."""

    source = validate_contract(manifest, expected_schema=SOURCE_SCHEMA)
    preflight = preflight_source_manifest(source)
    if not preflight["object_license_ready"]:
        raise ContractError(
            "anatomy_prior_out_of_domain",
            "$.source_manifest",
            "; ".join(preflight["gaps"]),
        )
    if not structure_ids:
        raise ContractError("anatomy_prior_out_of_domain", "$.structure_ids", "no structures selected")
    # atlas_from_body maps SOMA body landmarks to the generic atlas.
    fit = fit_rigid_landmarks(
        body_landmarks_m,
        atlas_landmarks_m,
        max_rms_error_m=max_error_m,
        minimum_observability=0.005,
        method="generic-anatomy-landmark-rigid-v1",
    )
    body_validation = np.asarray(validation_body_landmarks_m, dtype=np.float64)
    atlas_validation = np.asarray(validation_atlas_landmarks_m, dtype=np.float64)
    if body_validation.shape != atlas_validation.shape or body_validation.ndim != 2 or body_validation.shape[1] != 3:
        raise ContractError("anatomy_prior_out_of_domain", "$.validation", "held-out landmarks need matching (N,3)")
    predicted = body_validation @ fit.observed_patch_from_body[:3, :3].T + fit.observed_patch_from_body[:3, 3]
    errors = np.linalg.norm(predicted - atlas_validation, axis=1)
    error_m = float(errors.max())
    confidence = float(np.clip(1.0 - (error_m / max_error_m) ** 2, 0.0, 1.0))
    validation_set_sha256 = hashlib.sha256(
        np.ascontiguousarray(np.concatenate([body_validation, atlas_validation], axis=1), dtype="<f8").tobytes()
    ).hexdigest()
    document = {
        "schema": REGISTRATION_SCHEMA,
        "content_sha256": "0" * 64,
        "source": source["source"],
        "version": source["version"],
        "license": source["pack_license_spdx"],
        "structure_ids": list(structure_ids),
        "atlas_from_body": fit.observed_patch_from_body.astype(float).tolist(),
        "warp": None,
        "error_m": error_m,
        "confidence": confidence,
        "population_limitations": population_limitations,
        "validation_set_sha256": validation_set_sha256,
        "provenance": deepcopy(provenance),
    }
    document["content_sha256"] = canonical_digest(document)
    document = validate_contract(document, expected_schema=REGISTRATION_SCHEMA)
    metrics = {
        "fit_rms_m": fit.rms_error_m,
        "fit_max_m": fit.max_error_m,
        "heldout_rms_m": float(np.sqrt(np.mean(errors * errors))),
        "heldout_max_m": error_m,
        "heldout_errors_m": errors.astype(float).tolist(),
        "confidence": confidence,
    }
    return document, metrics


def anatomy_availability(
    registration: dict[str, Any],
    *,
    minimum_confidence: float = 0.8,
    maximum_error_m: float = 0.005,
) -> dict[str, Any]:
    document = validate_contract(registration, expected_schema=REGISTRATION_SCHEMA)
    available = document["confidence"] >= minimum_confidence and document["error_m"] <= maximum_error_m
    return {
        "status": "available_for_declared_research_visualization" if available else "unavailable",
        "confidence": document["confidence"],
        "error_m": document["error_m"],
        "structure_ids": document["structure_ids"] if available else [],
        "automatic_site_safety": False,
        "individual_structure_location": False,
        "diagnosis": False,
        "pain_inference": False,
        "execution_authority": False,
    }
