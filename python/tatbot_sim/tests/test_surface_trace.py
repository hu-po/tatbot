from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from tatbot_sim import interaction
from tatbot_sim.config import DRConfig
from tatbot_sim.inkmap.compiler import compile_scenario
from tatbot_sim.inkmap.mesh_patch_surface import mesh_patch_from_scenario
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.inkmap.sampler import materialize_scenario_suite
from tatbot_sim.inkmap.scenario_scene import materialize_scenario_geometry
from tatbot_sim.inkmap.surface_trace import _resample, anchors_to_points, compile_surface_trace
from tatbot_sim.inkmap.svg_strokes import SvgCompileError, compile_svg_strokes
from tatbot_sim.planning import plan_tattoo_scenario
from tatbot_sim.repo import repo_root

PUBLIC = repo_root() / "web" / "inkmap" / "public"
EXAMPLE = repo_root() / "config" / "inkmap" / "examples" / "forearm-placement-v4.json"


def test_every_builtin_svg_compiles_to_finite_metric_strokes():
    manifest = json.loads((PUBLIC / "designs" / "manifest.json").read_text())
    for design in manifest["designs"]:
        compiled = compile_svg_strokes(
            (PUBLIC / design["path"]).read_text(), design["default_size_mm"],
        )
        assert compiled.strokes
        assert all(len(stroke) >= 2 and np.isfinite(stroke).all() for stroke in compiled.strokes)
        x0, y0, x1, y1 = compiled.bounds_m
        assert x1 - x0 <= design["default_size_mm"][0] / 1000 + 1e-7
        assert y1 - y0 <= design["default_size_mm"][1] / 1000 + 1e-7


def test_svg_metric_transform_is_exact_and_unknown_transform_fails_closed():
    rectangle = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 20"><rect width="10" height="20"/></svg>'
    compiled = compile_svg_strokes(rectangle, [30, 40], mirror=True)
    np.testing.assert_allclose(compiled.bounds_m, [-0.015, -0.020, 0.015, 0.020], atol=1e-8)
    with pytest.raises(SvgCompileError, match="unsupported SVG transform"):
        compile_svg_strokes(rectangle.replace("<rect", '<rect transform="scale(2)"'), [30, 40])


def test_forearm_trace_is_deterministic_continuous_and_follows_pose():
    placement_file = json.loads(EXAMPLE.read_text())
    placement = placement_file["placements"][0]
    metric = compile_svg_strokes(
        (PUBLIC / "designs" / "heart.svg").read_text(), placement["size_mm"], mirror=placement["mirror"],
    )
    rig = load_body_rig(placement_file["body"]["id"])
    trace = compile_surface_trace(rig, placement, metric.strokes)
    again = compile_surface_trace(rig, placement, metric.strokes)
    assert trace.sha256 == again.sha256 == "f8a16d28b1d5b70e5afaccdf0a8f0048924ff024d662986fb454efd63d1e9325"
    assert sum(map(len, trace.strokes)) == 361
    for pose_id in ("standing-neutral", "reclined-left-arm-supported"):
        points = anchors_to_points(rig.posed(pose_id).vertices, trace)
        assert max(np.linalg.norm(np.diff(stroke, axis=0), axis=1).max() for stroke in points) <= 0.000501
    rest_points = anchors_to_points(rig.posed("standing-neutral").vertices, trace)
    supported_points = anchors_to_points(rig.posed("reclined-left-arm-supported").vertices, trace)
    assert np.linalg.norm(rest_points[0] - supported_points[0], axis=1).mean() > 0.2


def test_scenario_compiler_materializes_real_checksums_and_is_reproducible():
    placement = json.loads(EXAMPLE.read_text())
    kwargs = {
        "pose_id": "reclined-left-arm-supported",
        "seed": 42,
        "created_at": "2026-09-01T13:00:00Z",
        "git_sha": "1ff80f9",
    }
    first = compile_scenario(placement, **kwargs)
    second = compile_scenario(placement, **kwargs)
    assert first == second
    assert first["trace"]["sha256"] == "f8a16d28b1d5b70e5afaccdf0a8f0048924ff024d662986fb454efd63d1e9325"
    assert first["body"]["rig_sha256"] != "c" * 64
    assert first["pose"]["catalog_sha256"] != "b" * 64
    assert first["placement"]["source_sha256"] != "d" * 64
    assert first["support"]["id"] == "tattoo-chair-left-armrest-v1"

    metric = compile_svg_strokes(
        first["design"]["svg"], first["placement"]["size_mm"], mirror=first["placement"]["mirror"],
    )
    uv = _resample(metric.strokes[0], 5e-4)
    surface = mesh_patch_from_scenario(first).env_view(0, len(uv))
    assert surface.width_m == pytest.approx(first["placement"]["size_mm"][0] / 1000)
    assert surface.height_m == pytest.approx(first["placement"]["size_mm"][1] / 1000)
    points, du, dv, normals = surface.frame(torch.as_tensor(uv, dtype=torch.float32))
    posed = load_body_rig(first["body"]["id"]).posed(
        first["pose"]["id"], np.asarray(first["pose"]["world_from_body"]),
    )
    expected = np.stack([
        np.asarray(anchor["barycentric"]) @ posed.vertices[anchor["face"]]
        for anchor in first["trace"]["strokes"][0]
    ])
    np.testing.assert_allclose(points.numpy(), expected, atol=2e-6)
    np.testing.assert_allclose(torch.linalg.norm(normals, dim=1).numpy(), 1.0, atol=1e-6)
    assert torch.linalg.eigvalsh(surface.first_fundamental_form(torch.as_tensor(uv))).min() > 0
    projected_uv, signed_distance, _ = surface.project(points)
    error = np.linalg.norm(projected_uv.numpy() - uv, axis=1)
    assert np.quantile(error, 0.95) <= 5e-4
    assert signed_distance.abs().max() <= 1e-6
    assert torch.isfinite(du).all() and torch.isfinite(dv).all()


def test_body_scenario_geometry_and_plan_are_offline_replayable(tmp_path):
    placement = json.loads(EXAMPLE.read_text())
    scenario = compile_scenario(
        placement, pose_id="reclined-left-arm-supported", seed=42,
        created_at="2026-09-01T13:00:00Z", git_sha="1ff80f9",
    )
    geometry = materialize_scenario_geometry(scenario, tmp_path)
    assert geometry.body_obj.stat().st_size > geometry.patch_obj.stat().st_size > 10_000
    assert len(geometry.collision_capsules) >= 16
    assert geometry.root.name.startswith("v3-")
    assert len(geometry.support_boxes) == 4
    assert all(box.quaternion_wxyz is not None for box in geometry.support_boxes)
    plan = plan_tattoo_scenario(
        np.random.default_rng(0), scenario, geometry.surface,
        horizon=1800, num_envs=1, dr=DRConfig(), draw_clearance=interaction.WORKING_OFFSET_M,
    )
    assert plan.targets.shape == (1, 1800, 3)
    assert plan.kinds == ["body-tattoo"]
    np.testing.assert_allclose(np.linalg.norm(plan.pen_normals, axis=2), 1.0, atol=1e-6)
    clearance = np.sum((plan.targets - plan.surface_points) * plan.surface_normals, axis=2)
    assert clearance.min() >= interaction.WORKING_OFFSET_M - 1e-6
    assert clearance.max() <= interaction.WORKING_OFFSET_M + 0.020001


def test_procedural_suite_is_bounded_balanced_and_site_exact(tmp_path):
    manifest = materialize_scenario_suite(
        tmp_path / "suite",
        count=12,
        seed=20260901,
        created_at="2026-09-01T13:00:00Z",
        git_sha="0cd952f",
    )
    assert manifest["complete"]
    assert manifest["accepted"] == 12
    assert manifest["rejection_rate"] < 0.25
    assert len(manifest["coverage"]["bodies"]) == 2
    assert len(manifest["coverage"]["poses"]) == 5
    assert len(manifest["coverage"]["sites"]) == 6
    assert len(manifest["coverage"]["designs"]) == 10
    ledger = [json.loads(line) for line in (tmp_path / "suite" / "attempts.jsonl").read_text().splitlines()]
    assert sum(item["accepted"] for item in ledger) == 12
    assert all(item.get("reason") for item in ledger if not item["accepted"])
    for record in manifest["scenarios"]:
        scenario = json.loads((tmp_path / "suite" / record["scenario"]).read_text())
        atlas = json.loads((PUBLIC / "bodies" / f"{scenario['body']['id']}.regions.json").read_text())
        site = scenario["placement"]["site"]
        code = atlas["sites"].index(site["id"]) * 4 + {None: 0, "left": 1, "right": 2}[site["laterality"]]
        assert all(
            atlas["faces"][anchor["face"]] == code
            for stroke in scenario["trace"]["strokes"]
            for anchor in stroke
        )
        if scenario["pose"]["id"].endswith("left-arm-supported"):
            assert site["id"] in ("forearm", "bicep") and site["laterality"] == "left"
        if scenario["pose"]["id"].endswith("right-arm-supported"):
            assert site["id"] in ("forearm", "bicep") and site["laterality"] == "right"


def test_generated_svg_is_materialized_before_sampling(tmp_path):
    designs = tmp_path / "generated"
    designs.mkdir()
    (designs / "prompted-wave.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
        '<path d="M1 10 Q5 2 10 10 T19 10"/></svg>',
    )
    manifest = materialize_scenario_suite(
        tmp_path / "generated-suite",
        count=1,
        seed=9,
        bodies=("hbm-male-stylized",),
        poses=("standing-neutral",),
        sites=("forearm",),
        generated_design_dir=designs,
        include_builtin_designs=False,
        created_at="2026-09-01T13:00:00Z",
        git_sha="0cd952f",
    )
    scenario = json.loads((tmp_path / "generated-suite" / manifest["scenarios"][0]["scenario"]).read_text())
    assert scenario["design"]["id"].startswith("gen-prompted-wave-")
    assert scenario["design"]["source"] == {"kind": "generated-local"}
    assert "Q5 2 10 10" in scenario["design"]["svg"]
