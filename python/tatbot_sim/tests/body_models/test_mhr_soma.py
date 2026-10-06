from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from tatbot_sim.body_models import SOMAPosedBody, canonical_topology_digest
from tatbot_sim.human_rep.contracts import load_contract

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "config/body-models/mhr-soma-v1.json"
IDENTITY = ROOT / "config/human-representation/examples/body-identity.json"
STATE = ROOT / "config/human-representation/examples/body-state.json"


def _runtime() -> tuple[str, str]:
    cache = os.environ.get("TATBOT_BODY_CACHE_DIR")
    if cache is None or importlib.util.find_spec("soma") is None:
        pytest.skip("locked SOMA environment and TATBOT_BODY_CACHE_DIR are required")
    return cache, os.environ.get("TATBOT_BODY_DEVICE", "cpu")


@pytest.fixture(scope="module")
def body() -> SOMAPosedBody:
    cache, device = _runtime()
    return SOMAPosedBody(spec_path=SPEC, cache_dir=cache, device=device)


def test_reference_contract_materializes_exact_locked_surface(body: SOMAPosedBody) -> None:
    surface = body.from_contracts(IDENTITY, STATE)
    spec = load_contract(SPEC, expected_schema="tatbot.body-model-spec/1")
    assert surface.vertices_m.shape == (18_056, 3)
    assert surface.faces.shape == (36_108, 3)
    assert surface.face_vertices_m.shape == (36_108, 3, 3)
    assert surface.joints_m.shape == (77, 3)
    assert surface.transforms.shape == (78, 4, 4)
    assert surface.surface_sha256 == spec["coordinates"]["reference_rest_surface_sha256"]
    assert canonical_topology_digest(surface.faces) == spec["geometry"]["topology_sha256"]


def test_reference_generation_is_byte_identical(body: SOMAPosedBody) -> None:
    identity = load_contract(IDENTITY, expected_schema="tatbot.body-identity/1")
    first = body.rest(identity)
    second = body.rest(identity)
    assert first.vertices_m.tobytes() == second.vertices_m.tobytes()
    assert first.joints_m.tobytes() == second.joints_m.tobytes()
    assert first.surface_sha256 == second.surface_sha256


def test_expanded_faces_equal_indexed_surface_on_ten_thousand_addresses(body: SOMAPosedBody) -> None:
    identity = load_contract(IDENTITY, expected_schema="tatbot.body-identity/1")
    surface = body.rest(identity)
    rng = np.random.default_rng(9042026)
    face_ids = rng.integers(0, len(surface.faces), size=10_000)
    barycentric = rng.dirichlet(np.ones(3), size=10_000)
    indexed = np.einsum(
        "ni,nij->nj",
        barycentric,
        surface.vertices_m[surface.faces[face_ids]].astype(np.float64),
    )
    expanded = np.einsum(
        "ni,nij->nj",
        barycentric,
        surface.face_vertices_m[face_ids].astype(np.float64),
    )
    assert np.linalg.norm(indexed - expanded, axis=1).max() <= 0.00001


def test_invalid_pose_quaternion_refuses(body: SOMAPosedBody) -> None:
    identity = load_contract(IDENTITY, expected_schema="tatbot.body-identity/1")
    rotations = np.zeros((77, 4), dtype=np.float64)
    rotations[:, 3] = 1.0
    rotations[12] = 0
    with pytest.raises(ValueError, match="pose_unsupported"):
        body.generate(identity, rotations)


def test_out_of_prior_identity_refuses_before_soma_forward(body: SOMAPosedBody) -> None:
    identity = load_contract(IDENTITY, expected_schema="tatbot.body-identity/1") | {
        "coefficients": [4.0, *([0.0] * 44)],
    }
    rotations = np.zeros((77, 4), dtype=np.float64)
    rotations[:, 3] = 1.0
    with pytest.raises(ValueError, match="identity_out_of_prior"):
        body.generate(identity, rotations)

    identity = load_contract(IDENTITY, expected_schema="tatbot.body-identity/1") | {
        "scales": [4.0, *([0.0] * 67)],
    }
    with pytest.raises(ValueError, match="identity_out_of_prior"):
        body.generate(identity, rotations)
