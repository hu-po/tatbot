from __future__ import annotations

import numpy as np
import pytest
from tatbot_sim.human_rep.anatomy import (
    anatomy_availability,
    fit_anatomy_registration,
    make_source_manifest,
    preflight_source_manifest,
)
from tatbot_sim.human_rep.contracts import ContractError, canonical_digest, validate_contract
from tatbot_sim.human_rep.coupled_mechanics import (
    LinearShellFixture,
    ShellMesh,
    coupled_admission,
    grid_shell,
    verify_shell_fixture,
)

WHEN = "2026-09-04T17:00:00Z"


def _manifest(*, verified=True):
    return make_source_manifest(
        use_case="generic forearm bone visualization research only",
        source="synthetic-anatomy-fixture",
        version="1",
        pack_license_spdx="CC-BY-4.0",
        objects=[
            {
                "source_object_id": "radius-ulna-synthetic",
                "source_uri": "fixture://radius-ulna",
                "sha256": "a" * 64,
                "license_spdx": "CC-BY-4.0" if verified else "UNKNOWN",
                "attribution": "Synthetic Tatbot test fixture",
                "derivatives_permitted": verified,
                "object_license_verified": verified,
                "downloaded": False,
            }
        ],
        separate_asset_root="anatomy-packs/synthetic-v1",
        provenance={
            "producer": "p8-test",
            "version": "1",
            "created_utc": WHEN,
            "source_sha256": "b" * 64,
            "seed": 17,
        },
    )


def test_shell_fixture_forward_gradient_energy_and_determinism():
    solver = LinearShellFixture(grid_shell(), local_stiffness_n_m=800.0, coupling_n_m=120.0)
    report = verify_shell_fixture(solver, seed=17)
    assert report["status"] == "pass"
    assert all(report["checks"].values())
    assert report["energy_j"] >= 0
    assert report["gradient_max_error_n_m"] < 1e-8


def test_shell_mesh_rejects_bad_quality_and_boundary():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]])
    with pytest.raises(ContractError) as caught:
        ShellMesh(vertices, faces, np.asarray([True, True, False, False]))
    assert caught.value.code == "contact_solver_unstable"


def test_shell_admission_requires_p7_residual_and_twenty_percent_gain():
    pending = coupled_admission(
        p7_spatial_coupling_residual_available=False,
        relative_primary_improvement=None,
        contact_regressions=None,
    )
    assert pending["status"] == "not_admitted_pending_evidence"
    assert pending["admitted_model"] == "rigid-contact-v1"
    rejected = coupled_admission(
        p7_spatial_coupling_residual_available=True,
        relative_primary_improvement=0.19,
        contact_regressions={},
    )
    assert rejected["status"] == "archived_not_admitted"
    admitted = coupled_admission(
        p7_spatial_coupling_residual_available=True,
        relative_primary_improvement=0.21,
        contact_regressions={"force_bias": False},
    )
    assert admitted["status"] == "admitted"


def test_anatomy_preflight_is_object_scoped_and_never_authorizes_download():
    ready = preflight_source_manifest(_manifest())
    assert ready["object_license_ready"] is True
    assert ready["download_ready"] is True
    assert ready["download_authorized"] is False
    unresolved = preflight_source_manifest(_manifest(verified=False))
    assert unresolved["object_license_ready"] is False
    assert len(unresolved["gaps"]) >= 2


def test_synthetic_anatomy_registration_and_out_of_domain_rendering():
    rng = np.random.default_rng(17)
    body = rng.normal(0, 0.05, (12, 3))
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    translation = np.asarray([0.1, -0.2, 0.3])
    atlas = body @ rotation.T + translation
    document, metrics = fit_anatomy_registration(
        _manifest(),
        structure_ids=["radius", "ulna"],
        body_landmarks_m=body[:8],
        atlas_landmarks_m=atlas[:8],
        validation_body_landmarks_m=body[8:],
        validation_atlas_landmarks_m=atlas[8:],
        population_limitations="Generic synthetic geometry; not an individual anatomy estimate.",
        provenance={
            "producer": "p8-test",
            "version": "1",
            "created_utc": WHEN,
            "source_sha256": "c" * 64,
            "seed": 17,
        },
    )
    validate_contract(document, expected_schema="tatbot.anatomy-registration/1")
    assert metrics["heldout_max_m"] < 1e-12
    available = anatomy_availability(document)
    assert available["status"] == "available_for_declared_research_visualization"
    assert available["automatic_site_safety"] is False
    assert available["execution_authority"] is False

    unavailable = dict(document)
    unavailable["confidence"] = 0.1
    unavailable["content_sha256"] = canonical_digest(unavailable)
    assert anatomy_availability(unavailable)["status"] == "unavailable"


def test_anatomy_fit_refuses_unresolved_object_license():
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(ContractError) as caught:
        fit_anatomy_registration(
            _manifest(verified=False),
            structure_ids=["radius"],
            body_landmarks_m=points,
            atlas_landmarks_m=points,
            validation_body_landmarks_m=points,
            validation_atlas_landmarks_m=points,
            population_limitations="synthetic",
            provenance={"producer": "p8-test", "version": "1", "created_utc": WHEN, "source_sha256": "c" * 64},
        )
    assert caught.value.code == "anatomy_prior_out_of_domain"
