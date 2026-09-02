from __future__ import annotations

import hashlib
import json

import numpy as np
from tatbot_sim.inkmap.gltf_surface import load_canonical_surface
from tatbot_sim.repo import repo_root

BODIES = repo_root() / "web" / "inkmap" / "public" / "bodies"
CATALOG = repo_root() / "config" / "inkmap" / "body-poses.json"
RIG_CONFIG = repo_root() / "config" / "inkmap" / "body-rig.json"


def test_hbm_surfaces_match_the_browser_contract():
    expected = {
        "hbm-male-stylized": "dec5cc557e39dbc45f3c485c9986a74005f08e491604a001cb698b6a60a26f48",
        "hbm-female-stylized": "00a8d9ae1813c850bb12d0e7092aadfcd6412085320df8f62b76843abaf5fffc",
    }
    for body_id, digest in expected.items():
        surface = load_canonical_surface(BODIES / f"{body_id}.glb")
        assert surface.vertices.shape == (28200, 3, 3)
        assert surface.sha256 == digest
        assert [(p.name, p.face_count) for p in surface.parts] == [
            ("Body", 25000), ("EyeL", 1600), ("EyeR", 1600),
        ]
        assert np.isfinite(surface.vertices).all()
        assert np.isclose(surface.vertices[..., 2].min(), 0.0)


def test_rigged_assets_preserve_faces_and_have_normalized_sidecars():
    catalog = json.loads(CATALOG.read_text())
    rig_config = json.loads(RIG_CONFIG.read_text())
    gates = rig_config["quality_gates"]
    assert catalog["pose_ids"] == [
        "standing-neutral",
        "supine",
        "prone",
        "reclined-seated",
        "reclined-left-arm-supported",
        "reclined-right-arm-supported",
    ]
    for body_id, record in catalog["bodies"].items():
        source = load_canonical_surface(BODIES / f"{body_id}.glb")
        rigged_path = repo_root() / "web" / "inkmap" / "public" / record["rigged_path"]
        rigged = load_canonical_surface(rigged_path)
        assert rigged.sha256 == source.sha256 == record["surface_sha256"]
        assert hashlib.sha256(rigged_path.read_bytes()).hexdigest() == record["rigged_asset_sha256"]

        sidecar_path = repo_root() / "web" / "inkmap" / "public" / record["sidecar_path"]
        assert hashlib.sha256(sidecar_path.read_bytes()).hexdigest() == record["sidecar_sha256"]
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            assert sidecar["rest_vertices"].shape == (28200, 3, 3)
            assert sidecar["pose_vertices"].shape == (6, 28200, 3, 3)
            assert sidecar["joint_indices"].shape == (28200 * 3, 4)
            assert sidecar["joint_weights"].shape == (28200 * 3, 4)
            np.testing.assert_allclose(sidecar["joint_weights"].sum(axis=1), 1.0, atol=2e-7)
            assert sidecar["pose_ids"].tolist() == catalog["pose_ids"]
        assert record["automatic_weight_fallback_vertices"] == 0
        assert record["max_lbs_error_m"] <= 1e-4
        for pose_id, pose in record["poses"].items():
            quality = pose["quality"]
            assert quality["max_joint_rotation_deg"] <= gates["max_joint_rotation_deg"]
            assert quality["edge_length_ratio_p001"] >= gates["edge_length_ratio_p001_min"]
            assert quality["edge_length_ratio_p99"] <= gates["edge_length_ratio_p99_max"]
            assert quality["triangle_area_ratio_p01"] >= gates["triangle_area_ratio_p01_min"]
            assert quality["triangle_area_ratio_p99"] <= gates["triangle_area_ratio_p99_max"]
            expected_anatomy_gates = rig_config["poses"][pose_id].get("anatomy_gates", [])
            assert bool(pose["anatomy"]) == bool(expected_anatomy_gates)
            if pose_id.startswith("reclined"):
                assert pose["anatomy"]["bend_offset_m:hip.L-knee.L-ankle.L"] >= 0.045
                assert pose["anatomy"]["bend_offset_m:hip.R-knee.R-ankle.R"] >= 0.045
                assert pose["anatomy"]["bend_off_axis_m:hip.L-knee.L-ankle.L"] <= 0.01
                assert pose["anatomy"]["bend_off_axis_m:hip.R-knee.R-ankle.R"] <= 0.01
            if pose_id.endswith("left-arm-supported"):
                assert 110 <= pose["anatomy"]["angle_deg:shoulder.L-elbow.L-wrist.L"] <= 145
                assert pose["anatomy"]["angle_deg:elbow.L-wrist.L-hand_tip.L"] >= 170
            if pose_id.endswith("right-arm-supported"):
                assert 110 <= pose["anatomy"]["angle_deg:shoulder.R-elbow.R-wrist.R"] <= 145
                assert pose["anatomy"]["angle_deg:elbow.R-wrist.R-hand_tip.R"] >= 170
