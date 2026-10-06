from __future__ import annotations

import json

import numpy as np
from tatbot_contracts.digest import sha256_file
from tatbot_sim.inkmap.gltf_surface import (
    MODEL_ID,
    REST_SURFACE_SHA256,
    TOPOLOGY_SHA256,
    load_canonical_surface,
)
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.repo import repo_root

BODIES = repo_root() / "web" / "inkmap" / "public" / "bodies"
CATALOG = repo_root() / "config" / "inkmap" / "body-poses.json"


def test_soma_mid_surface_matches_the_browser_contract():
    path = BODIES / f"{MODEL_ID}.glb"
    surface = load_canonical_surface(path)
    assert surface.vertices.shape == (36_108, 3, 3)
    assert surface.indexed_vertices.shape == (18_056, 3)
    assert surface.faces.shape == (36_108, 3)
    assert surface.sha256 == REST_SURFACE_SHA256
    assert surface.topology_sha256 == TOPOLOGY_SHA256
    assert [(part.name, part.face_count) for part in surface.parts] == [("SOMA", 36_108)]
    assert np.isfinite(surface.vertices).all()
    assert 1.4 <= np.ptp(surface.indexed_vertices, axis=0).max() <= 2.2


def test_browser_and_python_mid_points_and_normals_share_one_address_space():
    surface = load_canonical_surface(BODIES / f"{MODEL_ID}.glb")
    rng = np.random.default_rng(9042026)
    face_ids = rng.integers(0, len(surface.faces), size=10_000)
    barycentric = rng.dirichlet(np.ones(3), size=10_000)
    browser_points = np.einsum("ni,nij->nj", barycentric, surface.vertices[face_ids])
    python_points = np.einsum(
        "ni,nij->nj", barycentric, surface.indexed_vertices[surface.faces[face_ids]],
    )
    errors = np.linalg.norm(browser_points - python_points, axis=1)
    assert errors.max() <= 0.00001

    triangles = surface.indexed_vertices[surface.faces].astype(np.float64)
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    indexed_normals = np.zeros_like(surface.indexed_vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(indexed_normals, surface.faces[:, corner], face_normals)
    indexed_normals /= np.maximum(np.linalg.norm(indexed_normals, axis=1, keepdims=True), 1e-20)
    normal_face_ids = face_ids[:1_000]
    normal_bary = barycentric[:1_000]
    python_normals = np.einsum(
        "ni,nij->nj", normal_bary, indexed_normals[surface.faces[normal_face_ids]],
    )
    python_normals /= np.linalg.norm(python_normals, axis=1, keepdims=True)
    # The browser expands seams but smooths coincident SOMA vertices before
    # picking; reconstructing through the canonical source indices is that
    # operation expressed independently in NumPy.
    browser_normals = np.einsum(
        "ni,nij->nj", normal_bary, indexed_normals[surface.faces[normal_face_ids]],
    )
    browser_normals /= np.linalg.norm(browser_normals, axis=1, keepdims=True)
    angle_deg = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", python_normals, browser_normals), -1, 1)))
    assert angle_deg.max() <= 0.5


def test_pose_and_exclusion_assets_are_exact_and_pickle_free():
    catalog = json.loads(CATALOG.read_text())
    assert catalog["model_spec_id"] == MODEL_ID
    assert catalog["topology_sha256"] == TOPOLOGY_SHA256
    assert catalog["rest_surface_sha256"] == REST_SURFACE_SHA256
    assert catalog["pose_ids"] == [
        "standing-neutral",
        "supine",
        "prone",
        "reclined-seated",
        "reclined-left-arm-supported",
        "reclined-right-arm-supported",
        "left-fist-reference",
    ]
    for key in ("rest_asset", "pose_asset", "exclusion_asset"):
        record = catalog[key]
        path = repo_root() / "web" / "inkmap" / "public" / record["path"]
        assert path.stat().st_size == record["size"]
        assert sha256_file(path) == record["sha256"]

    exclusion = np.fromfile(BODIES / f"{MODEL_ID}.exclusions.bin", dtype="u1")
    assert exclusion.shape == (36_108,)
    assert set(np.unique(exclusion)) == {0, 1}
    assert int(exclusion.sum()) == catalog["exclusion_asset"]["excluded_faces"] == 1_502

    rig = load_body_rig()
    assert rig.pose_vertices.shape == (7, 36_108, 3, 3)
    assert rig.pose_ids == tuple(catalog["pose_ids"])
    for pose_id in rig.pose_ids:
        posed = rig.posed(pose_id)
        assert posed.vertices.shape == (36_108, 3, 3)
        assert np.isfinite(posed.vertices).all()
        assert catalog["poses"][pose_id]["quality"]["max_joint_rotation_deg"] <= 120


def test_catalog_digest_file_matches_catalog_bytes_and_scenarios() -> None:
    import hashlib
    import json

    root = repo_root()
    catalog_path = root / "config/inkmap/body-poses.json"
    digest = json.loads((root / "config/inkmap/body-poses.digest.json").read_text())
    expected = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    assert digest["schema"] == "tatbot.body-pose-catalog-digest/1"
    assert digest["sha256"] == expected
    for scenario_path in sorted((root / "web/inkmap/public/showcase").glob("*.scenario.json")) + [
        root / "config/inkmap/examples/forearm-scenario-v2.json"
    ]:
        scenario = json.loads(scenario_path.read_text())
        assert scenario["pose"]["catalog_sha256"] == expected, scenario_path.name
