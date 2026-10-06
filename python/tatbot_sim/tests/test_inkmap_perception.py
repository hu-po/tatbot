from __future__ import annotations

import json

import numpy as np
import pytest
from tatbot_sim.inkmap.bundle import make_simulation_bundle
from tatbot_sim.inkmap.identities import admitted_identity_ids, identity_contract
from tatbot_sim.inkmap.mesh_patch_surface import MeshPatchSurface
from tatbot_sim.inkmap.perception import (
    PinholeCamera,
    render_perception_labels,
    write_perception_sidecar,
)
from tatbot_sim.inkmap.perception_audit import audit_frame
from tatbot_sim.inkmap.perception_dataset import Args, render_dataset
from tatbot_sim.inkmap.perception_variation import (
    PerceptionVariation,
    audit_split_records,
    split_for_groups,
)
from tatbot_sim.inkmap.program_scenario import compile_simulation_bundle
from tatbot_sim.inkmap.program_target import ProgramTarget
from tatbot_sim.inkmap.rig import CATALOG_PATH, load_synthetic_identity_rig
from tatbot_sim.inkmap.surface_trace import UnfoldedPatch
from tatbot_sim.repo import repo_root


def _scene():
    vertices = np.asarray([[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.5, 2.0]])
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    triangle = vertices[faces]
    uv = np.asarray([[[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]]])
    patch = UnfoldedPatch(
        seed_face=0,
        body_first_face=0,
        seed_triangle_uv=uv[0],
        mesh_vertices=triangle,
        mesh_keys=[[(0, 0, 0), (1, 0, 0), (0, 1, 0)]],
        face_indices=np.asarray([0]),
        triangles_uv=uv,
        adjacent={0: set()},
    )
    surface = MeshPatchSurface([patch], [triangle], width_m=1, height_m=1, cols=16, rows=16)
    coverage = np.ones((16, 16), dtype=np.float32)
    target = ProgramTarget(
        coverage=coverage,
        color_srgb=np.broadcast_to(np.asarray([0.05, 0.1, 0.2], np.float32), (16, 16, 3)).copy(),
        layer_id=np.ones((16, 16), dtype=np.int32),
        placement_id=np.ones((16, 16), dtype=np.int32),
        width_m=1,
        height_m=1,
        pixels_per_m=4000,
        supersample=4,
        tattoo_program_sha256="a" * 64,
        surface_placement_sha256="b" * 64,
    )
    camera = PinholeCamera(48, 48, np.asarray([[40, 0, 23.5], [0, 40, 23.5], [0, 0, 1]]), np.eye(4))
    return vertices, faces, surface, target, camera


def test_reference_labels_are_reproducible_and_project_back_to_visible_surface():
    vertices, faces, surface, target, camera = _scene()
    first = render_perception_labels(vertices, faces, camera, surface=surface, target=target, seed=9)
    again = render_perception_labels(vertices, faces, camera, surface=surface, target=target, seed=9)
    assert first.canonical_digest() == again.canonical_digest()
    assert first.body_visible.any()
    assert np.all(first.face_index[first.body_visible] == 0)
    assert np.all(first.tattoo_placement_id[first.body_visible] == 1)
    ys, xs = np.where(first.body_visible)
    for y, x in zip(ys[::17], xs[::17], strict=False):
        bary = first.barycentric[y, x]
        point = bary @ vertices[faces[0]]
        assert first.depth_m[y, x] == pytest.approx(point[2], abs=2e-6)
        assert x == pytest.approx(camera.intrinsic[0, 0] * point[0] / point[2] + camera.intrinsic[0, 2], abs=1e-5)
        assert y == pytest.approx(camera.intrinsic[1, 1] * point[1] / point[2] + camera.intrinsic[1, 2], abs=1e-5)


def test_occlusion_preserves_scene_depth_and_clears_body_tattoo_semantics():
    vertices, faces, surface, target, camera = _scene()
    occluder = np.full((48, 48), np.nan)
    occluder[20:28, 20:28] = 1.0
    labels = render_perception_labels(
        vertices,
        faces,
        camera,
        surface=surface,
        target=target,
        occluder_depth_m=occluder,
    )
    area = np.s_[20:28, 20:28]
    assert labels.depth_valid[area].all()
    assert np.all(labels.depth_m[area] == 1)
    assert not labels.body_visible[area].any()
    assert np.all(labels.face_index[area] == -1)
    assert np.all(labels.tattoo_coverage[area] == 0)
    assert np.all(labels.tattoo_placement_id[area] == 0)


def test_sidecar_manifest_and_auditor_reject_tampering(tmp_path):
    vertices, faces, surface, target, camera = _scene()
    labels = render_perception_labels(vertices, faces, camera, surface=surface, target=target)
    bindings = {name: format(index + 1, "x") * 64 for index, name in enumerate((
        "identity_sha256", "rest_surface_sha256", "posed_surface_sha256", "design_sha256",
        "placement_sha256", "scenario_sha256", "artwork_family_sha256",
    ))}
    bindings.update(
        identity_sha256="800babcf48d09f4ff1da9d2e8a0ebe266869a19849557450c7210c9a9fb9a1b6",
        split="train",
        scene_id="scene-1",
        source_repository="test/repository",
        source_revision="1" * 40,
        source_dirty=False,
        asset_provenance={"body": "fixture"},
        dependency_versions={"numpy": np.__version__},
    )
    write_perception_sidecar(
        tmp_path / "frame",
        labels,
        camera,
        bindings=bindings,
        seed_streams={"appearance": 1},
        sampled_axes={"blur_sigma_px": 0.0},
    )
    assert audit_frame(tmp_path / "frame")[0] == []
    manifest_path = tmp_path / "frame" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bindings"]["scenario_sha256"] = "bad"
    manifest_path.write_text(json.dumps(manifest))
    assert any("scenario_sha256" in problem for problem in audit_frame(tmp_path / "frame")[0])


def test_variation_streams_are_independent_and_split_groups_do_not_follow_batch_order():
    variation = PerceptionVariation()
    values, streams = variation.sample(41, "scene-a")
    assert (values, streams) == variation.sample(41, "scene-a")
    assert len(set(streams.values())) == len(streams)
    assert variation.sample(41, "scene-b") != (values, streams)
    first = split_for_groups(artwork_family_sha256="a" * 64, identity_sha256="b" * 64, seed=7)
    assert first == split_for_groups(artwork_family_sha256="a" * 64, identity_sha256="b" * 64, seed=7)
    clean = [
        {"artwork_family_sha256": "a", "identity_sha256": "i", "split": "train"},
        {"artwork_family_sha256": "a", "identity_sha256": "j", "split": "identity-held-out"},
    ]
    assert audit_split_records(clean) == []
    dirty = clean + [{"artwork_family_sha256": "a", "identity_sha256": "i", "split": "design-held-out"}]
    assert len(audit_split_records(dirty)) == 1


def test_unreviewed_identities_are_available_for_evidence_but_fail_closed_for_data():
    assert admitted_identity_ids() == ("reference",)
    assert identity_contract("reference")["content_sha256"] == "800babcf48d09f4ff1da9d2e8a0ebe266869a19849557450c7210c9a9fb9a1b6"
    with pytest.raises(ValueError, match="identity_not_admitted"):
        identity_contract("mixed-positive-2sigma")
    with pytest.raises(ValueError, match="identity_not_admitted"):
        load_synthetic_identity_rig("mixed-positive-2sigma", cache_dir="/does/not/matter")
    candidate = identity_contract("mixed-positive-2sigma", require_admitted=False)
    assert candidate["content_sha256"] == "890deb3962368a2153b70db32cb854a961172b93f50f9ae6dd00d5c99d9d36d2"


@pytest.mark.slow
def test_typed_scenario_renders_a_complete_audited_perception_corpus(tmp_path, monkeypatch):
    root = repo_root()
    placement = json.loads((root / "config/inkmap/examples/forearm-placement-v6.json").read_text())
    catalog = json.loads(CATALOG_PATH.read_text())
    catalog_sha256 = __import__("hashlib").sha256(CATALOG_PATH.read_bytes()).hexdigest()
    bundle = make_simulation_bundle(
        placement,
        {
            "pose_id": "supine",
            "pose_catalog_sha256": catalog_sha256,
            "support_id": catalog["poses"]["supine"]["support_id"],
            "tool_id": "lutin-3rl-bugpin",
            "seed": 42,
            "skin_tone": "#c07f57",
            "camera": None,
            "target_world_m": [0.29, 0, 0.04],
            "align_patch_up": True,
            "patch_yaw_rad": np.pi,
        },
    )
    scenario = compile_simulation_bundle(
        bundle,
        created_at="2026-09-05T00:00:00Z",
        git_sha="1234567",
    )
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(scenario))
    monkeypatch.setattr(
        "tatbot_sim.inkmap.perception_dataset.source_state",
        lambda: {
            "repository": "test/repository",
            "revision": "1" * 40,
            "dirty": False,
        },
    )
    report = render_dataset(
        Args(
            scenario=scenario_path,
            output_dir=tmp_path / "perception",
            views=1,
            width=48,
            height=48,
            focal_px=55,
        )
    )
    assert report["status"] == "pass"
    assert report["requested"] == report["accepted"] == 1
    assert report["bytes_per_accepted_frame"] > 0


def test_a_render_that_accepted_nothing_reports_instead_of_crashing(tmp_path, monkeypatch):
    """The reasons are the point; a TypeError in the projection buried them."""
    import json as _json

    from tatbot_sim.inkmap import perception_dataset

    output = tmp_path / "corpus"
    output.mkdir()
    (output / "requests.json").write_text(_json.dumps(
        {"schema": "tatbot.inkmap-perception-requests/1", "requests": []}) + "\n")

    monkeypatch.setattr(perception_dataset, "audit_corpus", lambda root: {
        "schema": "tatbot.inkmap-perception-audit/1", "status": "fail", "requested": 2,
        "accepted": 0, "rejected": 0, "bytes": 1234, "bytes_per_accepted_frame": None,
        "problems": ["source checkout was dirty or unknown"]})

    class Fake:
        views = 2

    with pytest.raises(RuntimeError, match="dirty or unknown"):
        perception_dataset._finalize_report(Fake(), output, elapsed=1.0)
    written = _json.loads((output / "audit.json").read_text())
    assert written["accepted"] == 0
    assert written["performance"]["projected_540_frame_bytes"] is None
    assert written["problems"] == ["source checkout was dirty or unknown"]
