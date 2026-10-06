from __future__ import annotations

import json
import math

import numpy as np
import pytest
from tatbot_sim.human_rep.contracts import ContractError, canonical_digest, load_contract, validate_contract
from tatbot_sim.human_rep.registration import (
    ASSISTED_METHOD,
    coordinate_points,
    create_surface_registration,
    fit_rigid_landmarks,
    nominal_patch_hypotheses,
    select_separated_hypothesis,
    state_surface_vertices,
    transform_error,
    weak_centroid_fit,
)
from tatbot_sim.human_rep.registration_suite import (
    cached_reference_variants,
    run_synthetic_observability,
)
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.repo import repo_root

REPO = repo_root()
EXAMPLES = REPO / "config" / "human-representation" / "examples"
CAPTURE = "a" * 64
CALIBRATION = "b" * 64
WHEN = "2026-09-04T15:00:00Z"


def _matrix() -> np.ndarray:
    axis = np.asarray([0.3, -0.5, 0.8], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = math.radians(23.0)
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = [0.31, -0.08, 0.17]
    return result


def _transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


@pytest.fixture(scope="module")
def landmark_context():
    rig = load_body_rig()
    state = load_contract(EXAMPLES / "body-state.json")
    atlas = json.loads((REPO / "web" / "inkmap" / "public" / "bodies" / "mhr-soma-v1.regions.json").read_text())
    site_index = atlas["sites"].index("forearm")
    candidates = np.flatnonzero(np.asarray(atlas["faces"], dtype=np.int64) == site_index * 4 + 2)
    # Pick a spatially broad, deterministic set from the right-forearm region.
    centers = rig.rest_vertices[candidates].mean(axis=1)
    chosen = [int(np.argmin(centers[:, 2])), int(np.argmax(centers[:, 2]))]
    chosen.append(int(np.argmax(centers[:, 0])))
    chosen.append(int(np.argmin(centers[:, 1])))
    chosen.extend(int(index) for index in np.linspace(0, len(candidates) - 1, 8, dtype=int))
    faces = []
    for local in chosen:
        face = int(candidates[local])
        if face not in faces:
            faces.append(face)
    coordinates = [
        {
            "topology_sha256": rig.topology_sha256,
            "face_index": face,
            "barycentric": [1 / 3, 1 / 3, 1 / 3],
        }
        for face in faces
    ]
    source = coordinate_points(coordinates, state, rig=rig)
    assert np.linalg.matrix_rank(source - source.mean(axis=0)) >= 2
    return rig, state, coordinates, source


def test_direct_landmark_fit_recovers_exact_transform(landmark_context):
    _, _, _, source = landmark_context
    truth = _matrix()
    fit = fit_rigid_landmarks(source, _transform(truth, source), sensor_sigma_m=0.0)
    error = transform_error(fit.observed_patch_from_body, truth)
    assert error["translation_error_m"] < 1e-12
    assert error["rotation_error_deg"] < 1e-6
    assert fit.rms_error_m < 1e-12
    assert fit.confidence == pytest.approx(1.0)
    assert fit.covariance.shape == (6, 6)


def test_noisy_fit_reports_covariance_and_beats_weak_baseline(landmark_context):
    _, _, _, source = landmark_context
    truth = _matrix()
    rng = np.random.default_rng(9042026)
    observed = _transform(truth, source) + rng.normal(0.0, 0.00008, source.shape)
    direct = fit_rigid_landmarks(source, observed, sensor_sigma_m=0.00008)
    weak = weak_centroid_fit(source, observed)
    assert direct.rms_error_m < 0.0002
    assert direct.rms_error_m < weak.rms_error_m
    assert direct.confidence >= 0.95
    assert np.all(np.linalg.eigvalsh(direct.covariance) >= -1e-14)


def test_direct_registration_contract_binds_surface_and_state(landmark_context):
    rig, state, coordinates, source = landmark_context
    truth = _matrix()
    observed = _transform(truth, source)
    document, fit = create_surface_registration(
        state,
        EXAMPLES / "current-surface.npz",
        [
            {"canonical": coordinate, "observed_xyz_m": point.tolist(), "weight": 1.0}
            for coordinate, point in zip(coordinates, observed, strict=True)
        ],
        capture_sha256=CAPTURE,
        calibration_sha256=CALIBRATION,
        provenance={
            "producer": "p5-registration-test",
            "version": "1",
            "created_utc": WHEN,
            "source_sha256": CAPTURE,
            "seed": 9042026,
        },
        sensor_sigma_m=0.0,
        rig=rig,
    )
    assert document["body_state_sha256"] == state["content_sha256"]
    assert document["supported_cells"] == [0, 1, 2, 3]
    assert np.allclose(document["observed_patch_from_body"], truth, atol=1e-12)
    assert all(item["error_m"] < 1e-12 for item in document["correspondences"])
    assert fit.confidence == 1.0
    validate_contract(document, expected_schema="tatbot.surface-registration/1")


def test_collinear_landmarks_refuse_as_ambiguous():
    source = np.stack([np.arange(4), np.zeros(4), np.zeros(4)], axis=1)
    with pytest.raises(ContractError) as caught:
        fit_rigid_landmarks(source, source + 1)
    assert caught.value.code == "registration_ambiguous"


def test_body_state_selects_and_hash_checks_the_posed_surface(landmark_context):
    rig, original, coordinates, _ = landmark_context
    state = json.loads(json.dumps(original))
    state["named_pose"] = "supine"
    state["posed_surface_sha256"] = rig.catalog_record["poses"]["supine"]["surface_sha256"]
    state["content_sha256"] = canonical_digest(state)
    posed = state_surface_vertices(state, rig)
    assert np.array_equal(posed, rig.pose_vertices[rig.pose_ids.index("supine")].astype(np.float64))
    assert not np.array_equal(coordinate_points(coordinates, state, rig=rig), coordinate_points(coordinates, original, rig=rig))

    state["posed_surface_sha256"] = "f" * 64
    state["content_sha256"] = canonical_digest(state)
    with pytest.raises(ContractError) as caught:
        state_surface_vertices(state, rig)
    assert caught.value.code == "execution_binding_mismatch"


def test_nominal_assisted_path_preserves_hypotheses_and_refuses_ties():
    rng = np.random.default_rng(17)
    source = rng.normal(0, 0.03, (32, 3))
    truth = _matrix()
    observed = _transform(truth, source)
    hypotheses = nominal_patch_hypotheses(
        source,
        observed,
        [("truth", truth), ("identity", np.eye(4))],
        trim_fraction=0.0,
    )
    assert len(hypotheses) == 2
    assert all(item.fit.method == ASSISTED_METHOD for item in hypotheses)
    assert hypotheses[0].chamfer_rms_m <= hypotheses[1].chamfer_rms_m

    tied = nominal_patch_hypotheses(
        source,
        observed,
        [("a", truth), ("b", truth)],
        trim_fraction=0.0,
    )
    with pytest.raises(ContractError) as caught:
        select_separated_hypothesis(tied)
    assert caught.value.code == "registration_ambiguous"


def test_synthetic_suite_keeps_all_axes_and_denominators(landmark_context):
    rig, _, _, _ = landmark_context
    atlas = json.loads((REPO / "web" / "inkmap" / "public" / "bodies" / "mhr-soma-v1.regions.json").read_text())
    suite, representative = run_synthetic_observability(
        cached_reference_variants(rig)[:1],
        atlas=atlas,
        sites=["forearm"],
        views=["normal", "grazing"],
        occlusions=[0.0, 0.4],
        noise_sigmas_m=[0.0, 0.0001],
        seed=17,
    )
    assert suite["summary"]["total_cases"] == 8
    assert suite["summary"]["direct_accepted"] + suite["summary"]["direct_refused"] == 8
    assert set(suite["cases"][0]["methods"]) == {"direct_local", "nominal_assisted", "weak"}
    assert suite["cases"][0]["landmarks"] == 48
    assert suite["summary"]["by_site"]["forearm"]["cases"] > 0
    assert representative["estimate"].shape == (4, 4)
