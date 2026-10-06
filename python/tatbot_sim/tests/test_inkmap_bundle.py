from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from tatbot_sim import interaction
from tatbot_sim.config import DRConfig
from tatbot_sim.generate import _primitive_schedule
from tatbot_sim.human_rep.contracts import ContractError, canonical_digest
from tatbot_sim.inkmap.bundle import make_simulation_bundle, validate_simulation_bundle
from tatbot_sim.inkmap.cli import _compile, build_parser
from tatbot_sim.inkmap.contracts import load_scenario, validate_scenario
from tatbot_sim.inkmap.program_scenario import compile_simulation_bundle
from tatbot_sim.inkmap.program_target import DEFAULT_PIXELS_PER_M, render_program_target
from tatbot_sim.planning import plan_tattoo_scenario
from tatbot_sim.repo import repo_root

ROOT = repo_root()


@pytest.fixture(scope="module")
def bundle():
    file = json.loads((ROOT / "config/inkmap/examples/forearm-placement-v6.json").read_text())
    catalog = json.loads((ROOT / "config/inkmap/body-poses.json").read_text())
    digest = json.loads((ROOT / "config/inkmap/body-poses.digest.json").read_text())["sha256"]
    return make_simulation_bundle(file, {
        "pose_id": "supine", "pose_catalog_sha256": digest, "support_id": catalog["poses"]["supine"]["support_id"],
        "tool_id": "lutin-3rl-bugpin", "seed": 42, "skin_tone": "#c07f57", "camera": None,
        "target_world_m": [.29, 0, .04], "align_patch_up": True, "patch_yaw_rad": 3.141592653589793,
    })


def test_bundle_cross_language_reader_and_rehashed_tampering(bundle):
    assert validate_simulation_bundle(bundle) == bundle
    bad = copy.deepcopy(bundle)
    bad["surface_placements"][0]["placement"]["rotation_rad"] = .5
    bad["surface_placements"][0]["placement"]["content_sha256"] = canonical_digest(bad["surface_placements"][0]["placement"])
    bad["content_sha256"] = canonical_digest(bad)
    with pytest.raises(ContractError, match="derivation/order"):
        validate_simulation_bundle(bad)


def test_bundle_validation_reuse_checks_actual_content_not_claimed_hash(bundle, monkeypatch):
    import tatbot_sim.inkmap.bundle as reader
    reader._VERIFIED_BUNDLES.clear()
    request = reader.artwork_request
    calls = []

    def observe(document):
        calls.append(document["operation"])
        return request(document)

    monkeypatch.setattr(reader, "artwork_request", observe)
    loaded = reader.validate_simulation_bundle(copy.deepcopy(bundle))
    assert reader.validate_simulation_bundle(bundle) == loaded
    assert calls == ["validate_bundle"]
    loaded["request"]["seed"] += 1  # Caller mutation cannot poison a saved validation.
    with pytest.raises(ContractError, match="bundle content differs"):
        reader.validate_simulation_bundle(loaded)
    assert reader.validate_simulation_bundle(bundle) == bundle


@pytest.mark.slow
def test_typed_placement_search_rebinds_support_request_without_mutating_source(bundle):
    from tatbot_sim.inkmap.reach import _typed_variant
    scenario = compile_simulation_bundle(bundle, created_at="2026-09-06T00:00:00Z", git_sha="1234567")
    before = copy.deepcopy(scenario)
    variant = _typed_variant(scenario, .2, (.01, 0), (0, -.01, 0))
    assert scenario == before
    assert variant["program_binding"]["bundle"]["content_sha256"] != bundle["content_sha256"]
    assert np.asarray(variant["support"]["world_from_nominal"])[:3, 3] == pytest.approx([0, -.01, 0])
    assert variant["design"] == scenario["design"]
    validate_scenario(variant)
    variant["support"]["world_from_nominal"][1][3] = 0
    with pytest.raises(ContractError, match="support|scenario differs"):
        validate_scenario(variant)


@pytest.mark.slow
def test_cli_bundle_compiles_typed_scenario_without_boundary_fallback(bundle, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy SVG boundary compiler was called")
    monkeypatch.setattr("tatbot_sim.inkmap.svg_strokes.compile_svg_strokes", forbidden)
    path = tmp_path / "input.json"
    output = tmp_path / "scenario.json"
    path.write_text(json.dumps(bundle))
    args = build_parser().parse_args(["compile", str(path), "--output", str(output),
                                    "--created-at", "2026-09-05T00:00:00Z", "--git-sha", "1234567"])
    assert _compile(args) == 0
    result = load_scenario(output)
    assert result["schema_version"] == 3
    assert result["program_binding"]["bundle"] == bundle
    assert result["trace"]["strokes"]
    assert all(e["curve"]["width_m"] == .0003 for e in result["program_binding"]["ink_program"]["events"] if e["kind"] == "stroke")
    assert result["seed"] == 42
    assert validate_scenario(result) is result
    bad = copy.deepcopy(result)
    bad["pose"]["world_from_body"][0][3] += .01
    with pytest.raises(ContractError, match="scenario differs"):
        validate_scenario(bad)
    bad = copy.deepcopy(result)
    event = next(e for e in bad["program_binding"]["ink_program"]["events"] if e["kind"] == "stroke")
    event["curve"]["width_m"] *= 2
    bad["program_binding"]["ink_program"]["content_sha256"] = canonical_digest(bad["program_binding"]["ink_program"])
    with pytest.raises(ContractError, match="compiled derivation differs"):
        validate_scenario(bad)
    bad = copy.deepcopy(result)
    bad["program_binding"]["tool_profile_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="tool profile differs"):
        validate_scenario(bad)
    from tatbot_sim.inkmap.scenario_scene import materialize_scenario_geometry
    geometry = materialize_scenario_geometry(result, tmp_path / "typed-target-cache")
    assert geometry.target is not None
    assert geometry.target.cols >= 100  # 25 mm at the required 4 px/mm reference raster.
    assert 0 < geometry.target.coverage.mean() < 1
    assert set(np.unique(geometry.target.placement_id)) <= {0, 1}
    assert np.all(geometry.blank_skin_rgba[..., :3] == [192, 127, 87])
    assert np.all(geometry.blank_skin_rgba[..., 3] == 255)
    assert not np.array_equal(
        geometry.blank_skin_rgba[0, 0, :3],
        np.rint(geometry.target.color_srgb.max(axis=(0, 1)) * 255).astype(np.uint8),
    )
    assert (geometry.root / "target-labels.npz").is_file()
    assert (geometry.root / "target-reference.png").is_file()
    assert (geometry.root / "target-coverage.png").is_file()
    manifest = json.loads((geometry.root / "target-manifest.json").read_text())
    assert manifest["pixels_per_m"] == DEFAULT_PIXELS_PER_M
    assert manifest["runtime_ink"] == "separate and blank at episode reset"
    # This immutable artwork exceeds sixty seconds at the seeded draw speed.
    # Keep the budget refusal explicit before checking full-stroke mapping.
    with pytest.raises(ValueError, match="horizon is 1800"):
        plan_tattoo_scenario(
            np.random.default_rng(7),
            result,
            geometry.surface,
            horizon=1800,
            num_envs=1,
            dr=DRConfig(),
            draw_clearance=interaction.WORKING_OFFSET_M,
        )
    plan = plan_tattoo_scenario(
        np.random.default_rng(7),
        result,
        geometry.surface,
        horizon=2400,
        num_envs=1,
        dr=DRConfig(),
        draw_clearance=interaction.WORKING_OFFSET_M,
    )
    typed_strokes = [
        event
        for event in result["program_binding"]["ink_program"]["events"]
        if event["kind"] == "stroke"
    ]
    assert len(plan.paths[0]) == len(typed_strokes)
    assert len(plan.stroke_metadata[0]) == len(typed_strokes)
    assert all(item["source_primitive_sha256"] for item in plan.stroke_metadata[0])
    primitive, layer = _primitive_schedule(plan, geometry.surface, 0)
    intended_distance = np.sum(
        (plan.targets[0] - plan.surface_points[0]) * plan.surface_normals[0],
        axis=1,
    )
    pen_down = intended_distance <= interaction.CONTACT_ABOVE_TOLERANCE_M
    assert np.all(primitive[pen_down] >= 0)
    assert np.all(layer[pen_down] >= 0)
    accepted_bytes = output.read_bytes()
    for option in [["--seed", "0"], ["--pose", "supine"], ["--preserve-pose-world"]]:
        args = build_parser().parse_args(["compile", str(path), "--output", str(output), *option])
        with pytest.raises(ValueError, match="immutable"):
            _compile(args)
        assert output.read_bytes() == accepted_bytes


@pytest.mark.slow
def test_multiple_placements_require_explicit_drawing_selection(bundle):
    file = copy.deepcopy(bundle["placement_file"])
    file["placements"].append({**copy.deepcopy(file["placements"][0]), "id": "second"})
    multiple = make_simulation_bundle(file, bundle["request"])
    with pytest.raises(ContractError, match="placement-id"):
        compile_simulation_bundle(multiple)
    selected = compile_simulation_bundle(multiple, placement_id="second", created_at="2026-09-05T00:00:00Z", git_sha="1234567")
    assert selected["placement"]["id"] == "second"
    assert len(selected["program_binding"]["bundle"]["surface_placements"]) == 2


def test_program_target_uses_the_same_rotation_and_mirror_split_as_browser(bundle):
    program = bundle["artworks"]["line-v1"]["program"]
    placement = bundle["surface_placements"][0]["placement"]
    base = render_program_target(program, placement)
    rotated = copy.deepcopy(placement)
    rotated["rotation_rad"] = 1.1
    rotated["content_sha256"] = canonical_digest(rotated)
    turned = render_program_target(program, rotated)
    np.testing.assert_array_equal(turned.coverage, base.coverage)
    np.testing.assert_array_equal(turned.color_srgb, base.color_srgb)
    mirrored = copy.deepcopy(placement)
    mirrored["mirrored"] = not placement["mirrored"]
    mirrored["content_sha256"] = canonical_digest(mirrored)
    flipped = render_program_target(program, mirrored)
    expected = base.coverage[:, ::-1] >= 0.5
    actual = flipped.coverage >= 0.5
    intersection = np.count_nonzero(expected & actual)
    union = np.count_nonzero(expected | actual)
    assert intersection / union >= 0.98


@pytest.mark.parametrize("name,size", [
    ("linework", [80, 50]), ("blackwork", [60, 60]), ("negative-space", [60, 60]),
    ("stipple", [60, 40]), ("color-layers", [80, 50]),
])
@pytest.mark.slow
def test_five_owned_artwork_classes_compile_with_typed_material_semantics(bundle, name, size):
    file = copy.deepcopy(bundle["placement_file"])
    id = file["placements"][0]["design_id"]
    file["placements"][0]["size_mm"] = [25, 25 * size[1] / size[0]]
    source = {"kind": "fixture", "identifier": name, "license": "CC0-1.0", "attribution": "Tatbot owned fixture", "generation": None}
    from tatbot_sim.inkmap.artwork import make_artwork_record
    file["designs"][id] = make_artwork_record(name=name,
        original_svg=(ROOT / f"web/inkmap/tests/fixtures/artwork/{name}.svg").read_text(), source=source,
        conversion={"canvas_m": [v / 1000 for v in size], "semantic_intent": name,
                    "width_m": .0003, "deposition": 1, "chord_error_m": .000005})
    sample = make_simulation_bundle(file, bundle["request"], sources={id: source})
    result = compile_simulation_bundle(sample, created_at="2026-09-05T00:00:00Z", git_sha="1234567")
    assert result["schema_version"] == 3
    program = result["program_binding"]["ink_program"]
    assert len(result["trace"]["strokes"]) == sum(event["kind"] == "stroke" for event in program["events"])
    assert all("union of" in event["ordering_rationale"] for event in program["events"] if event["kind"] == "stroke")
    target = render_program_target(
        sample["artworks"][id]["program"], sample["surface_placements"][0]["placement"]
    )
    assert target.digest() == render_program_target(
        sample["artworks"][id]["program"], sample["surface_placements"][0]["placement"]
    ).digest()
    assert target.cols >= 100
    assert target.coverage.dtype == np.float32
    assert target.color_srgb.dtype == np.float32
    assert target.layer_id.dtype == np.int32
    assert target.placement_id.dtype == np.int32
    assert 0 < float(target.coverage.max()) <= 1
    assert np.all(target.color_srgb[target.coverage == 0] == 0)
    if name == "negative-space":
        assert target.coverage[target.rows // 2, target.cols // 2] == 0
        assert target.coverage[target.rows // 2, target.cols * 3 // 4] > .9
    if name == "color-layers":
        assert any(event.get("trigger_reason") == "color_change" for event in program["events"])
        visible_colors = np.unique(target.color_srgb[target.coverage > .99].round(4), axis=0)
        assert len(visible_colors) >= 2
