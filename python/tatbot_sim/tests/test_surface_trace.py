from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tatbot_sim import interaction, tools
from tatbot_sim.config import DRConfig
from tatbot_sim.human_rep.ink_program import scheduled_material_strokes
from tatbot_sim.inkmap.collection import collection_entries
from tatbot_sim.inkmap.compiler import compile_scenario
from tatbot_sim.inkmap.designs import procedural_artifact
from tatbot_sim.inkmap.mesh_patch_surface import mesh_patch_from_scenario
from tatbot_sim.inkmap.reach import optimize_body_placement
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.inkmap.robot_clearance import collision_geometries, tool_terminal_exclusion_m
from tatbot_sim.inkmap.sampler import materialize_scenario_suite
from tatbot_sim.inkmap.scenario_scene import SCENE_GEOMETRY_VERSION, materialize_scenario_geometry
from tatbot_sim.inkmap.surface_trace import (
    TRACE_CHART_STEP_M,
    SurfaceAnchor,
    SurfaceTraceError,
    _resample,
    anchors_to_points,
    compile_surface_trace,
    unfold_body_patch,
    validate_patch_footprint,
)
from tatbot_sim.inkmap.svg_strokes import SvgCompileError, compile_svg_strokes
from tatbot_sim.planning import plan_tattoo_scenario
from tatbot_sim.repo import repo_root
from tatbot_sim.tools import active_tool

PUBLIC = repo_root() / "web" / "inkmap" / "public"
EXAMPLE = repo_root() / "config" / "inkmap" / "examples" / "forearm-placement-v6.json"


def test_generated_svg_corpus_compiles_to_finite_metric_strokes():
    artifacts = [procedural_artifact(seed) for seed in range(32)]
    assert len({artifact.sha256 for artifact in artifacts}) == len(artifacts)
    for artifact in artifacts:
        compiled = compile_svg_strokes(artifact.svg, artifact.size_mm)
        assert compiled.strokes
        assert all(len(stroke) >= 2 and np.isfinite(stroke).all() for stroke in compiled.strokes)
        x0, y0, x1, y1 = compiled.bounds_m
        assert x1 - x0 <= artifact.size_mm[0] / 1000 + 1e-7
        assert y1 - y0 <= artifact.size_mm[1] / 1000 + 1e-7


def test_svg_metric_transform_is_exact_and_unknown_transform_fails_closed():
    rectangle = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 20"><rect width="10" height="20"/></svg>'
    compiled = compile_svg_strokes(rectangle, [30, 40], mirror=True)
    np.testing.assert_allclose(compiled.bounds_m, [-0.015, -0.020, 0.015, 0.020], atol=1e-8)
    transformed = rectangle.replace(
        '<rect width="10" height="20"/>',
        '<g transform="translate(2 3) rotate(90 5 10) scale(0.5)"><rect width="10" height="20"/></g>',
    )
    result = compile_svg_strokes(transformed, [30, 40])
    assert result.strokes and np.isfinite(result.strokes[0]).all()
    with pytest.raises(SvgCompileError, match="unsupported SVG transform"):
        compile_svg_strokes(rectangle.replace("<rect", '<rect transform="warp(2)"'), [30, 40])


def test_forearm_trace_is_deterministic_continuous_and_follows_pose():
    placement_file = json.loads(EXAMPLE.read_text())
    placement = placement_file["placements"][0]
    design = placement_file["designs"][placement["design_id"]]
    from tatbot_sim.inkmap.artwork import artwork_preview
    metric = compile_svg_strokes(artwork_preview(design), placement["size_mm"], mirror=placement["mirror"])
    rig = load_body_rig()
    trace = compile_surface_trace(rig, placement, metric.strokes)
    again = compile_surface_trace(rig, placement, metric.strokes)
    assert trace.sha256 == again.sha256
    assert sum(map(len, trace.strokes)) > 30
    for pose_id in ("standing-neutral", "reclined-left-arm-supported"):
        points = anchors_to_points(rig.posed(pose_id).vertices, trace)
        assert max(np.linalg.norm(np.diff(stroke, axis=0), axis=1).max() for stroke in points) <= 0.000501
    rest_points = anchors_to_points(rig.posed("standing-neutral").vertices, trace)
    supported_points = anchors_to_points(rig.posed("reclined-left-arm-supported").vertices, trace)
    assert np.linalg.norm(rest_points[0] - supported_points[0], axis=1).mean() > 0.2


def test_typescript_and_python_walk_ten_thousand_soma_points_identically(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the cross-language surface-walk parity gate")
    major = int(subprocess.check_output([node, "-p", "process.versions.node.split('.')[0]"], text=True))
    if major < 22:
        pytest.skip("Node.js 22 is required for TypeScript strip-types parity")
    placement = json.loads(EXAMPLE.read_text())["placements"][0]
    source = placement["anchor"]
    anchor = SurfaceAnchor(source["face"], tuple(source["barycentric"]))
    radius_m = 0.018
    # A continuous, seeded raster trajectory exercises crossings without
    # permitting discontinuous nearest-face choices.
    rng = np.random.default_rng(9042026)
    rows = np.linspace(-0.009, 0.009, 100)
    points = []
    for row_index, row in enumerate(rows):
        columns = np.linspace(-0.009, 0.009, 100)
        if row_index % 2:
            columns = columns[::-1]
        points.extend((float(column + rng.uniform(-1e-7, 1e-7)), float(row)) for column in columns)
    rig = load_body_rig()
    patch = unfold_body_patch(rig, anchor, float(placement["rotation_rad"]), radius_m)
    python_anchors = patch.anchors(np.asarray(points))
    request = tmp_path / "surface-parity-input.json"
    response = tmp_path / "surface-parity-output.json"
    request.write_text(json.dumps({
        "anchor": source,
        "rotation_rad": placement["rotation_rad"],
        "radius_m": radius_m,
        "points_m": points,
        "pose_ids": list(rig.pose_ids),
    }))
    subprocess.run(
        [node, "--experimental-strip-types", "web/inkmap/tools/surface_parity.ts", request, response],
        cwd=repo_root(),
        check=True,
    )
    typescript = json.loads(response.read_text())
    assert typescript["count"] == 10_000
    assert [item["face"] for item in typescript["anchors"]] == [item.face for item in python_anchors]
    error = np.max(np.abs(
        np.asarray([item["barycentric"] for item in typescript["anchors"]])
        - np.asarray([item.barycentric for item in python_anchors])
    ))
    assert error <= 1e-9
    position_errors = []
    for pose_id in rig.pose_ids:
        posed = rig.posed(pose_id).vertices
        expected = np.stack([
            np.asarray(anchor.barycentric) @ posed[anchor.face]
            for anchor in python_anchors
        ])
        actual = np.asarray(typescript["posed_positions_m"][pose_id])
        position_errors.extend(np.linalg.norm(actual - expected, axis=1))
    position_errors = np.asarray(position_errors)
    assert np.quantile(position_errors, .95) <= 0.0001
    assert position_errors.max() <= 0.0005


@pytest.mark.slow
def test_surface_parity_suite_covers_artwork_sizes_mirrors_curved_sites_and_poses(tmp_path):
    node = shutil.which("node")
    if node is None or int(subprocess.check_output(
        [node, "-p", "process.versions.node.split('.')[0]"], text=True,
    )) < 22:
        pytest.skip("Node.js 22 is required for the cross-language surface parity suite")
    specs = [
        ("linework-small-left-forearm", 8729, .015, .009375, 0.0, False),
        ("blackwork-right-forearm", 26998, .040, .040, .4, True),
        ("negative-space-large-left-shin", 11212, .060, .060, -.5, False),
        ("stipple-right-shin", 29266, .030, .020, .75, True),
        ("color-layers-large-left-thigh", 8225, .080, .050, 1.0, False),
        ("linework-mirrored-right-thigh", 26278, .020, .0125, -1.2, True),
        ("blackwork-curved-left-shoulder", 6361, .030, .030, .25, False),
    ]
    cases = []
    points_by_id = {}
    for id, face, width, height, rotation, mirrored in specs:
        rows = np.linspace(-height * .35, height * .35, 11)
        points = []
        for row_index, row in enumerate(rows):
            columns = np.linspace(-width * .35, width * .35, 11)
            if row_index % 2:
                columns = columns[::-1]
            if mirrored:
                columns = -columns
            points.extend((float(column), float(row)) for column in columns)
        points_by_id[id] = np.asarray(points)
        cases.append({
            "id": id,
            "anchor": {"face": face, "barycentric": [1 / 3, 1 / 3, 1 / 3]},
            "rotation_rad": rotation,
            "radius_m": float(np.hypot(width / 2, height / 2) + .003),
            "points_m": points,
            "footprint_m": [width, height],
        })
    rig = load_body_rig()
    request = tmp_path / "surface-parity-suite-input.json"
    response = tmp_path / "surface-parity-suite-output.json"
    request.write_text(json.dumps({"cases": cases, "pose_ids": list(rig.pose_ids)}))
    subprocess.run(
        [node, "--experimental-strip-types", "web/inkmap/tools/surface_parity.ts", request, response],
        cwd=repo_root(), check=True,
    )
    suite = json.loads(response.read_text())
    assert suite["requested"] == len(cases)
    assert suite["accepted"] == len(cases)
    assert suite["rejected"] == 0
    position_errors = []
    for case, result in zip(cases, suite["results"], strict=True):
        assert result["id"] == case["id"] and result["status"] == "accepted"
        source = case["anchor"]
        patch = unfold_body_patch(
            rig, SurfaceAnchor(source["face"], tuple(source["barycentric"])),
            case["rotation_rad"], case["radius_m"],
        )
        validate_patch_footprint(patch, *case["footprint_m"])
        expected_anchors = patch.anchors(points_by_id[result["id"]])
        assert [item["face"] for item in result["anchors"]] == [item.face for item in expected_anchors]
        np.testing.assert_allclose(
            [item["barycentric"] for item in result["anchors"]],
            [item.barycentric for item in expected_anchors], atol=1e-9, rtol=0,
        )
        for pose_id in rig.pose_ids:
            posed = rig.posed(pose_id).vertices
            expected = np.stack([
                np.asarray(anchor.barycentric) @ posed[anchor.face]
                for anchor in expected_anchors
            ])
            actual = np.asarray(result["posed_positions_m"][pose_id])
            position_errors.extend(np.linalg.norm(actual - expected, axis=1))
    position_errors = np.asarray(position_errors)
    assert np.quantile(position_errors, .95) <= 0.0001
    assert position_errors.max() <= 0.0005


def test_typescript_and_python_refuse_a_complete_wrap_chart(tmp_path):
    node = shutil.which("node")
    if node is None or int(subprocess.check_output(
        [node, "-p", "process.versions.node.split('.')[0]"], text=True,
    )) < 22:
        pytest.skip("Node.js 22 is required for the cross-language surface parity suite")
    case = {
        "id": "complete-wrap-left-forearm",
        "anchor": {"face": 8729, "barycentric": [1 / 3, 1 / 3, 1 / 3]},
        "rotation_rad": 0.0,
        "radius_m": float(np.hypot(0.5, 0.5) / 2 + 0.002),
        "points_m": [[0.0, 0.0]],
        "footprint_m": [0.5, 0.5],
    }
    request = tmp_path / "surface-wrap-input.json"
    response = tmp_path / "surface-wrap-output.json"
    request.write_text(json.dumps({"cases": [case]}))
    subprocess.run(
        [node, "--experimental-strip-types", "web/inkmap/tools/surface_parity.ts", request, response],
        cwd=repo_root(), check=True,
    )
    result = json.loads(response.read_text())["results"][0]
    assert result["status"] == "rejected"
    assert "surface_chart_overlap" in result["reason"]
    rig = load_body_rig()
    patch = unfold_body_patch(
            rig, SurfaceAnchor(8729, (1 / 3, 1 / 3, 1 / 3)), 0.0, 0.5,
    )
    with pytest.raises(SurfaceTraceError, match="surface_chart_overlap"):
        validate_patch_footprint(patch, 0.5, 0.5)


@pytest.mark.slow
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
    # The caller's generator survives the placement-file -> bundle -> scenario hop.
    assert first["provenance"]["generator"] == "tatbot sim compile"
    labelled = compile_scenario(placement, generator="tatbot sim resolve calibration", **kwargs)
    assert labelled["provenance"]["generator"] == "tatbot sim resolve calibration"
    assert first["trace"]["sha256"]
    assert first["design"]["id"] == "line-v1"
    assert first["body"]["topology_sha256"] == load_body_rig().topology_sha256
    assert first["body"]["pose_asset_sha256"] == load_body_rig().catalog_record["pose_asset"]["sha256"]
    assert first["pose"]["catalog_sha256"] != "b" * 64
    assert first["placement"]["source_sha256"] != "d" * 64
    assert first["support"]["id"] == "tattoo-chair-left-armrest-v1"

    # Use the frozen program's material path, not a recompiled preview SVG.
    binding = first["program_binding"]
    bundle = binding["bundle"]
    program = bundle["artworks"][first["design"]["id"]]["program"]
    placed = bundle["surface_placements"][0]["placement"]
    tool = tools.registry().load_tool(bundle["request"]["tool_id"], repo_root())
    stroke = scheduled_material_strokes(program, placed, footprint_width_m=tool.line_width_m)[0]
    # The mesh patch already carries rotation; the material path has it baked in.
    c, s = np.cos(placed["rotation_rad"]), np.sin(placed["rotation_rad"])
    uv = _resample(stroke.points_m @ np.asarray([[c, -s], [s, c]]), TRACE_CHART_STEP_M)
    surface = mesh_patch_from_scenario(first).env_view(0, len(uv))
    assert surface.width_m == pytest.approx(first["placement"]["size_mm"][0] / 1000)
    assert surface.height_m == pytest.approx(first["placement"]["size_mm"][1] / 1000)
    points, du, dv, normals = surface.frame(torch.as_tensor(uv, dtype=torch.float32))
    posed = load_body_rig().posed(
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

    # Projection is a world-space query and must not depend on whichever
    # chart triangle frame() happened to resolve most recently.  In the live
    # one-env replay, initialization asks for the origin frame immediately
    # before the TCP starts moving over the rest of the design.
    one = mesh_patch_from_scenario(first)
    # A point inside the 30 mm design: the patch is developed only as far as
    # the design plus 3 mm, no longer padded by the mesh-wide longest edge.
    target_uv = torch.as_tensor([[0.010, 0.012]], dtype=torch.float32)
    target_point = one.frame(target_uv)[0]
    one.frame(torch.zeros((1, 2), dtype=torch.float32))
    projected_uv, signed_distance, _ = one.project(target_point)
    assert np.isfinite(projected_uv.numpy()).all()
    assert signed_distance.abs().max() <= 1e-6


@pytest.mark.slow
def test_body_scenario_geometry_and_plan_are_offline_replayable(tmp_path):
    placement = json.loads(EXAMPLE.read_text())
    scenario = compile_scenario(
        placement, pose_id="reclined-left-arm-supported", seed=42,
        created_at="2026-09-01T13:00:00Z", git_sha="1ff80f9",
    )
    geometry = materialize_scenario_geometry(scenario, tmp_path)
    assert geometry.body_obj.stat().st_size > geometry.patch_obj.stat().st_size > 10_000
    assert len(geometry.collision_capsules) >= 16
    assert geometry.root.name.startswith(f"v{SCENE_GEOMETRY_VERSION}-")
    texture_uv = np.asarray([
        [float(value) for value in line.split()[1:]]
        for line in geometry.patch_obj.read_text().splitlines()
        if line.startswith("vt ")
    ]).reshape(geometry.surface.patches[0].triangles_uv.shape)
    chart_uv = geometry.surface.patches[0].triangles_uv
    assert np.allclose(texture_uv[..., 0], chart_uv[..., 0] / geometry.surface.width_m + .5)
    assert np.allclose(texture_uv[..., 1], .5 - chart_uv[..., 1] / geometry.surface.height_m)
    assert geometry.support_boxes == ()
    plan = plan_tattoo_scenario(
        np.random.default_rng(0), scenario, geometry.surface,
        horizon=3000, num_envs=1, dr=DRConfig(), draw_clearance=interaction.WORKING_OFFSET_M,
    )
    assert plan.targets.shape == (1, 3000, 3)
    assert plan.kinds == ["body-tattoo"]
    np.testing.assert_allclose(np.linalg.norm(plan.pen_normals, axis=2), 1.0, atol=1e-6)
    clearance = np.sum((plan.targets - plan.surface_points) * plan.surface_normals, axis=2)
    assert clearance.min() >= interaction.WORKING_OFFSET_M - 1e-6
    assert clearance.max() <= interaction.WORKING_OFFSET_M + 0.020001


@pytest.mark.slow
def test_placement_optimizer_is_bounded_deterministic_and_uses_urdf_geometry(monkeypatch):
    placement = json.loads(EXAMPLE.read_text())
    scenario = compile_scenario(
        placement, pose_id="reclined-left-arm-supported", seed=42,
        created_at="2026-09-01T13:00:00Z", git_sha="1ff80f9", tool_id=active_tool().tool_id,
    )
    center = mesh_patch_from_scenario(scenario).origin_world_np()[0]
    targets = np.linspace(center - [0.005, 0.0, 0.0], center + [0.005, 0.0, 0.0], 32)
    monkeypatch.setattr(
        "tatbot_sim.inkmap.reach.plan_tattoo_scenario",
        lambda *_args, **_kwargs: SimpleNamespace(
            targets=targets[None],
            pen_normals=np.broadcast_to([0.0, 0.0, 1.0], (1, len(targets), 3)),
            surface_points=targets[None],
            surface_normals=np.broadcast_to([0.0, 0.0, 1.0], (1, len(targets), 3)),
        ),
    )

    class FakeIK:
        chain = SimpleNamespace(get_joint_parameter_names=lambda: [f"joint_{i}" for i in range(6)])

        def step(self, _seed, target, _rotation, iters):
            del iters
            return torch.cat([target, torch.zeros((len(target), 3))], dim=1)

        def fk(self, q):
            matrix = torch.eye(4).repeat(len(q), 1, 1)
            matrix[:, :3, 3] = q[:, :3]
            return matrix

    class FakeExpert:
        def __init__(self, *_args, **_kwargs):
            self.ik = FakeIK()
            self.q_ref = None

        def solve_pose(self, _target, q0, **_kwargs):
            return q0

        def reset(self, targets_world, _q0, **_kwargs):
            # the full-trajectory gate reads the solved joint reference back
            # through fk; here q doubles as the tool position
            targets = torch.as_tensor(np.asarray(targets_world), dtype=torch.float32)
            self.q_ref = torch.cat([targets, torch.zeros((*targets.shape[:2], 3))], dim=2)

        def target_rotations(self, _normal, count):
            return torch.eye(3).repeat(count, 1, 1)

    monkeypatch.setattr("tatbot_sim.inkmap.reach.StrokeExpert", FakeExpert)
    monkeypatch.setattr(
        "tatbot_sim.inkmap.reach.tool_shaft_clearance",
        lambda *_: {"tool_shaft_m": 0.012, "tool_shaft_pair": "stub"},
    )
    monkeypatch.setattr(
        "tatbot_sim.inkmap.reach.non_tool_clearance",
        lambda *_: {"non_tool_robot_m": 0.018, "non_tool_pair": "stub"},
    )
    offsets = (((0.0, 0.0), (0.0, 0.0, 0.0)), ((0.01, 0.0), (0.0, 0.0, 0.0)))
    kwargs = {
        "trajectory_seed": 123,
        "yaw_candidates": (1.5 * np.pi,),
        "offset_candidates": offsets,
        "probe_points": 16,
    }
    first = optimize_body_placement(scenario, **kwargs)
    second = optimize_body_placement(scenario, **kwargs)
    assert first.scenario == second.scenario
    assert len(first.candidates) == 2
    # the probe passes both; the full-trajectory gate is only paid for the
    # lexicographic winner, and its verdict is retained in the separate audit
    selected = first.audit["selected"]
    assert selected["full_trajectory"]["targets"] == len(targets)
    assert selected["full_trajectory"]["max_residual_m"] <= 1e-6
    assert first.audit["search"]["full_trajectory_gate"] is True
    assert sum("full_trajectory" in item for item in first.candidates) == 1
    assert all(item["accepted"] for item in first.candidates)
    assert first.audit["search"]["configuration_sha256"]
    geometries = collision_geometries()
    assert {f"link_{index}" for index in range(1, 7)} <= {item.link for item in geometries}
    assert all(len(item.surface_points) >= 8 for item in geometries)


def test_the_shaft_check_leaves_out_exactly_the_tools_final_taper():
    """Derived from the datasheet, never re-hardcoded: a 32 mm literal outlived
    the V17 seat that made the 3RL's taper 35 mm, and kept checking cone as
    full-radius body."""
    tool = active_tool()
    (taper_z, body_r), (tip_z, tip_r) = tool.profile[-2:]
    assert tip_r < body_r and tip_z <= tool.protrusion_m
    assert tool.protrusion_m - tool_terminal_exclusion_m(tool) == pytest.approx(taper_z)


@pytest.mark.slow
@pytest.mark.timeout(600)  # Twelve complete typed artwork/body compilations, not primitive fixtures.
def test_artwork_suite_is_bounded_balanced_and_site_exact(tmp_path):
    manifest = materialize_scenario_suite(
        tmp_path / "suite",
        count=12,
        seed=20260901,
        created_at="2026-09-01T13:00:00Z",
        git_sha="0cd952f",
    )
    assert manifest["complete"]
    assert manifest["accepted"] == 12
    assert manifest["rejection_rate"] < 0.40
    assert manifest["coverage"]["bodies"] == ["mhr-soma-v1"]
    assert len(manifest["coverage"]["poses"]) == 5
    assert len(manifest["coverage"]["sites"]) == 6
    assert manifest["artwork_split"] == "train"
    assert manifest["design_source"] == "artwork"
    ledger = [json.loads(line) for line in (tmp_path / "suite" / "attempts.jsonl").read_text().splitlines()]
    assert sum(item["accepted"] for item in ledger) == 12
    assert all(item.get("reason") for item in ledger if not item["accepted"])
    requested = {entry["id"] for entry in collection_entries("train")}
    assert {item["design"] for item in ledger} == requested
    # A thin painted motif can be unrepresentable at the fitted tool's width.
    # The bounded sampler may substitute it, but its failed attempts must remain
    # visible rather than claiming every source was physically drawable.
    accepted = {item["design"] for item in ledger if item["accepted"]}
    assert set(manifest["coverage"]["designs"]) == accepted
    for missing in requested - accepted:
        refusals = [item for item in ledger if item["design"] == missing]
        assert len(refusals) == 4
        assert all(not item["accepted"] and item["reason"] and item["detail"] for item in refusals)
    for record in manifest["scenarios"]:
        scenario = json.loads((tmp_path / "suite" / record["scenario"]).read_text())
        atlas = json.loads((PUBLIC / "bodies" / f"{scenario['body']['model_spec_id']}.regions.json").read_text())
        site = scenario["placement"]["site"]
        code = atlas["sites"].index(site["id"]) * 4 + {None: 0, "left": 1, "right": 2}[site["laterality"]]
        assert all(
            atlas["faces"][anchor["face"]] == code
            for stroke in scenario["trace"]["strokes"]
            for anchor in stroke
        )
        if scenario["pose"]["id"].endswith("left-arm-supported"):
            assert site["id"] in ("forearm", "bicep", "tricep") and site["laterality"] == "left"
        if scenario["pose"]["id"].endswith("right-arm-supported"):
            assert site["id"] in ("forearm", "bicep", "tricep") and site["laterality"] == "right"


@pytest.mark.slow
def test_sampler_uses_only_canonical_resolutions_accepted_by_its_exposure_policy():
    from tatbot_sim.inkmap.rig import load_body_rig
    from tatbot_sim.inkmap.sampler import (
        DEFAULT_POSES,
        DEFAULT_SITES,
        _pose_supports_site,
        _resolved_anchor_candidates,
        _site_choices,
    )
    choices = _site_choices(DEFAULT_SITES)
    rig = load_body_rig()
    for pose_id in DEFAULT_POSES:
        posed = rig.posed(pose_id).vertices.astype(np.float64)
        for site in choices:
            if not _pose_supports_site(pose_id, site):
                continue
            resolutions = _resolved_anchor_candidates(
                pose_id, site, description=f"test {site.id}", seed=7,
            )
            assert all(value["resolver"]["name"] == "inklang-typescript" for value in resolutions)
            faces = np.asarray([value["anchor"]["face"] for value in resolutions])
            triangles = posed[faces]
            normals = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            assert np.all(normals[:, 2] >= 0.5), (pose_id, site)


@pytest.mark.slow
def test_audited_sampler_retries_a_different_physical_site(tmp_path, monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace

    from tatbot_sim.inkmap.reach import ReachAuditError

    observed_sites = []

    def select(scenario, *, trajectory_seed, audit_deadline_s=None):
        # mirrors optimize_body_placement: a caller may hand the audit a
        # slice of the suite's remaining wall clock
        observed_sites.append(scenario["placement"]["site"]["id"])
        if len(observed_sites) == 1:
            raise ReachAuditError("deliberate first pairing reject", "placement_clearance")
        accepted = deepcopy(scenario)
        accepted["placement_optimization"] = {
            "selected": {"candidate": 0},
        }
        return SimpleNamespace(
            scenario=accepted,
            audit=accepted["placement_optimization"],
            patch_yaw_rad=0.0,
            probe_max_residual_m=0.0,
            candidates=({"candidate": 0, "accepted": True},),
        )

    monkeypatch.setattr("tatbot_sim.inkmap.reach.optimize_body_placement", select)
    manifest = materialize_scenario_suite(
        tmp_path / "retry-suite",
        count=1,
        seed=4,
        poses=("supine",),
        sites=("shin", "thigh"),
        design_source="spiral",
        audit_reach=True,
        created_at="2026-09-03T13:00:00Z",
        git_sha="82f5804",
    )
    assert manifest["complete"]
    assert manifest["rejected_attempts"] == 1
    assert len(observed_sites) == 2
    assert observed_sites[0] != observed_sites[1]


def test_retry_sites_prefers_missing_coverage_and_remembers_clearance_rejects():
    from tatbot_sim.inkmap.sampler import SiteChoice, _retry_sites

    tricep = SiteChoice("tricep", "left")
    thigh = SiteChoice("thigh", "left")
    calf = SiteChoice("calf", "left")
    sites = (tricep, thigh, calf)
    draws = _retry_sites(
        "prone",
        tricep,
        sites,
        3,
        np.random.default_rng(8),
        deprioritized=frozenset({tricep}),
        priority_ids=frozenset({"calf"}),
    )
    assert draws[0] == calf
    assert draws[-1] == tricep


@pytest.mark.slow
def test_acquired_artwork_is_materialized_before_sampling(tmp_path):
    designs = tmp_path / "acquired"
    shutil.copytree(PUBLIC / "designs/dbv3-orbit", designs)
    artwork = json.loads((designs / "artwork.json").read_text())
    manifest = materialize_scenario_suite(
        tmp_path / "generated-suite",
        count=1,
        seed=9,
        poses=("supine",),
        sites=("forearm",),
        generated_design_dir=designs,
        design_source="directory",
        created_at="2026-09-01T13:00:00Z",
        git_sha="0cd952f",
    )
    scenario = json.loads((tmp_path / "generated-suite" / manifest["scenarios"][0]["scenario"]).read_text())
    assert scenario["design"]["id"] == artwork["name"]
    assert scenario["design"]["source"] == artwork["source"]
    assert scenario["schema_version"] == 3
    assert scenario["program_binding"]["bundle"]["artworks"][artwork["name"]] == artwork
