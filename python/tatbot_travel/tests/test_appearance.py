"""Appearance varies without changing the articulated skin, ink or kinematics."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tatbot_travel.appearance import LightingConfig, ScalarVariation, SkinConfig, SurfaceConfig, sample_skin
from tatbot_travel.camera import Intrinsics, render_plan
from tatbot_travel.motion import MotionConfig, PlacementConfig, rest_height, sample_rest_pose, sample_script
from tatbot_travel.phantom import build_phantom
from tatbot_travel.scene import SceneBuilder
from tatbot_travel.shell import chart_skin
from tatbot_travel.tabletop import (
    PaperPlacementConfig,
    TabletopConfig,
    TabletopItem,
    add_tabletop,
    support_height,
)
from tatbot_travel.textures import skin_chart_texture, skin_texture
from tatbot_travel.world import WorldConfig, build_world, captured_floor_parts


def test_skin_distribution_stays_near_the_reference_with_broad_tails():
    cfg = SkinConfig()
    rng = np.random.default_rng(17)
    looks = [sample_skin(rng, cfg) for _ in range(5000)]
    rgb = np.array([look["rgb"] for look in looks])
    assert np.mean([look["branch"] == "near" for look in looks]) == pytest.approx(0.8, abs=0.02)
    np.testing.assert_allclose(np.median(rgb, axis=0), cfg.center_rgb, atol=15)
    assert np.diff(np.percentile(rgb[:, 0], [5, 95]))[0] > 95
    for name in ("specular", "shininess", "mottling", "fine_texture", "pore_density"):
        values = np.array([look[name] for look in looks])
        lo, hi = getattr(cfg, name).bounds
        assert values.min() >= lo and values.max() <= hi
        assert values.std() > getattr(cfg, name).std / 2
    assert sample_skin(np.random.default_rng(17), cfg) == looks[0]


def test_reference_skin_uses_the_existing_tone_and_material_centers():
    cfg = SkinConfig(randomize=False)
    look = sample_skin(np.random.default_rng(17), cfg)
    assert look["branch"] == "reference" and look["rgb"] == list(cfg.center_rgb)
    assert look["specular"] == 0.3 and look["shininess"] == 0.4
    assert look["mottling"] == 0.03 and look["fine_texture"] == 0.015
    assert look["pore_density"] == 0.002


@pytest.mark.parametrize("values", [{"tail_prob": 1.1}, {"center_rgb": [240, 225]},
                                    {"tail_gain": [0.9, 0.4]}, {"pore_shade": [-0.1, 1]},
                                    {"tone": ScalarVariation(0, 1, (-1, 1))}])
def test_invalid_skin_distributions_are_refused(values):
    with pytest.raises(ValueError):
        SkinConfig(**values)


def test_skin_only_seeds_preserve_the_scene_ink_and_geometry_rng(monkeypatch):
    monkeypatch.delenv("TATBOT_TRAVEL_REAL_ASSETS", raising=False)
    cfg = WorldConfig(scene_view=False, max_people=0, max_objects=0, handless_prob=0,
                      phantom_hand_pose="closed", appearance_seed=11, skin_seed=20)
    rng_a, rng_b = np.random.default_rng(40), np.random.default_rng(40)
    first = build_world(rng_a, cfg)
    second = build_world(rng_b, replace(cfg, skin_seed=21))
    appearance_a, appearance_b = (dict(w.meta["appearance"]) for w in (first, second))
    skin_a, skin_b = appearance_a.pop("skin"), appearance_b.pop("skin")
    assert appearance_a == appearance_b and skin_a != skin_b
    assert rng_a.bit_generator.state == rng_b.bit_generator.state
    np.testing.assert_array_equal(first.phantom.vertices, second.phantom.vertices)
    for name, values in first.ink.surface_arrays().items():
        np.testing.assert_array_equal(values, second.ink.surface_arrays()[name])
    for name in ("body_pos", "geom_pos", "jnt_range", "light_pos", "light_diffuse"):
        np.testing.assert_array_equal(getattr(first.model, name), getattr(second.model, name))
    for world, look in ((first, skin_a), (second, skin_b)):
        for name in ("skin", "forearm"):
            material = world.model.mat(name).id
            assert world.model.mat_specular[material] == pytest.approx(look["specular"], abs=5e-5)
            assert world.model.mat_shininess[material] == pytest.approx(look["shininess"], abs=5e-5)


def test_forearm_chart_samples_the_shared_periodic_skin_atlas(monkeypatch):
    monkeypatch.delenv("TATBOT_TRAVEL_REAL_ASSETS", raising=False)
    atlas = skin_texture(np.random.default_rng(21), shape=(129, 257), base=np.array([236, 225, 198]))
    np.testing.assert_array_equal(atlas[0], atlas[-1])
    shell = chart_skin(build_phantom())
    extent = (shell.x[0], shell.x[-1])
    chart = skin_chart_texture(atlas, extent, shell, 0, atlas.shape[:2])
    np.testing.assert_array_equal(chart, atlas)
    rotated = skin_chart_texture(atlas, extent, shell, np.pi, atlas.shape[:2])
    np.testing.assert_array_equal(rotated[0], chart[64])
    np.testing.assert_array_equal(rotated[64], chart[0])
    np.testing.assert_array_equal(rotated[0], rotated[-1])


def test_appearance_seed_changes_materials_and_lights_but_preserves_skin_and_ink(monkeypatch):
    monkeypatch.delenv("TATBOT_TRAVEL_REAL_ASSETS", raising=False)
    cfg = WorldConfig(scene_view=False, max_people=0, max_objects=0, handless_prob=0,
                      phantom_hand_pose="closed", appearance_seed=11,
                      surface=SurfaceConfig(styles={"plain": 1}, randomize_captured_floor=True))
    first = build_world(np.random.default_rng(40), cfg)
    second = build_world(np.random.default_rng(40), replace(cfg, appearance_seed=12))
    assert first.meta["appearance"] != second.meta["appearance"]
    np.testing.assert_array_equal(first.phantom.vertices, second.phantom.vertices)
    assert first.phantom.provenance == second.phantom.provenance
    for name, values in first.ink.surface_arrays().items():
        np.testing.assert_array_equal(values, second.ink.surface_arrays()[name])
    np.testing.assert_array_equal(first.model.jnt_range, second.model.jnt_range)
    np.testing.assert_array_equal(first.model.geom_pos, second.model.geom_pos)
    np.testing.assert_array_equal(first.model.body_pos, second.model.body_pos)
    assert not np.array_equal(first.model.mat_rgba[first.model.mat("table").id],
                              second.model.mat_rgba[second.model.mat("table").id])
    assert not np.array_equal(first.model.light_pos, second.model.light_pos)
    repeated = build_world(np.random.default_rng(40), cfg)
    assert first.meta["appearance"] == repeated.meta["appearance"]


def test_captured_floor_partition_preserves_coordinates_and_separates_fixtures():
    header = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 0.3\nv 1 0 0.3\nv 0 1 0.3\nvt 0 0\n"
    faces = ["f 1/1 2/1 3/1", "f 1/1 2/1 4/1", "f 4/1 5/1 6/1"]
    floor, fixtures, count = captured_floor_parts(header + "\n".join(faces) + "\n", 0.02)
    assert count == 1 and floor.startswith(header) and fixtures.startswith(header)
    assert faces[0] in floor and faces[0] not in fixtures
    assert all(face in fixtures and face not in floor for face in faces[1:])


@pytest.mark.parametrize("values", [{"styles": {"grid": -1}}, {"value": [0.9, 0.1]},
                                    {"neutral_prob": 1.1}, {"saturation": [0, float("nan")]}])
def test_invalid_surface_distributions_are_refused(values):
    with pytest.raises(ValueError):
        SurfaceConfig(**values)


def test_invalid_lighting_distributions_are_refused():
    with pytest.raises(ValueError, match="lighting probabilities"):
        LightingConfig(shadow_prob=-1)


def test_mats_and_textured_paper_exist_in_wrist_only_episodes(monkeypatch):
    monkeypatch.delenv("TATBOT_TRAVEL_REAL_ASSETS", raising=False)
    cfg = WorldConfig(scene_view=False, max_people=0, max_objects=0, max_sheets=0,
                      tabletop=TabletopConfig(mat_count=(2, 2), paper_count=(3, 3),
                                              paper_styles={"printed": 1}))
    world = build_world(np.random.default_rng(40), cfg)
    assert world.scene_plan is None
    assert len(world.tabletop) == 5
    assert [item.kind for item in world.tabletop] == ["mat", "mat", "paper", "paper", "paper"]
    for item in world.tabletop:
        geom = world.model.geom(item.name)
        material = world.model.mat(item.name)
        assert geom.matid[0] == material.id
        assert world.model.mat_texid[material.id].max() >= 0
        mesh_id = int(geom.dataid[0])
        start = world.model.mesh_vertadr[mesh_id]
        stop = start + world.model.mesh_vertnum[mesh_id]
        compiled = Rotation.from_quat(np.roll(geom.quat, -1)).apply(world.model.mesh_vert[start:stop]) + geom.pos
        assert compiled[:, 2].max() == pytest.approx(item.top, abs=1e-7)
    assert all(i["style"] == "printed" for i in world.meta["appearance"]["tabletop"] if i["kind"] == "paper")
    for index, item in enumerate(world.tabletop):
        for lower in world.tabletop[:index]:
            if item.overlaps(lower):
                assert item.bottom >= lower.top - 1e-9


def test_skin_triangle_spanning_a_mat_is_supported_even_when_no_vertex_is_inside():
    vertices = np.array([[-2, -2, 0], [2, -2, 0], [0, 2, 0]], dtype=float)
    mat = TabletopItem("mat", "mat", (0, 0), (0.1, 0.1), 0.3, 0, 0.003)
    assert not mat.contains(vertices[:, :2]).any()
    assert support_height(vertices, np.array([[0, 1, 2]]), np.zeros(2), [mat]) == pytest.approx(0.003)


def test_randomized_palm_up_placements_and_motion_use_the_tabletop_support(monkeypatch):
    monkeypatch.delenv("TATBOT_TRAVEL_REAL_ASSETS", raising=False)
    world = build_world(np.random.default_rng(40), WorldConfig(scene_view=False, handless_prob=0,
                        phantom_hand_pose="closed", max_people=0, max_objects=0))
    cfg = PlacementConfig(radius=(0.24, 0.36), azimuth_deg=(-20, 25), yaw_deg=(55, 135),
                          back_up_prob=0, palm_up_prob=1, roll_jitter_deg=10, tilt_jitter_deg=3)
    mat = TabletopItem("mat", "mat", (0.35, 0), (0.5, 0.5), 0, 0, 0.003)
    rng = np.random.default_rng(22)
    poses = [sample_rest_pose(rng, world.phantom, cfg, tabletop=[mat]) for _ in range(16)]
    positions = np.array([p.pos for p in poses])
    headings = [np.degrees(np.arctan2(*p.rot.apply([1, 0, 0])[:2][::-1])) for p in poses]
    assert np.ptp(positions[:, 0]) > 0.06 and np.ptp(positions[:, 1]) > 0.12
    assert np.ptp(headings) > 50
    assert all(p.rot.apply([0, 0, -1])[2] > 0.8 for p in poses)
    for pose in poses:
        assert pose.apply(world.phantom.vertices)[:, 2].min() == pytest.approx(mat.top, abs=1e-9)
    script = sample_script(rng, world.phantom, 3, MotionConfig(hold_s=(0.5, 0.5),
                           weights={"nudge": 1}, placement=cfg), tabletop=[mat])
    for frame in script.keyframes:
        assert frame.pose.pos[2] == pytest.approx(rest_height(world.phantom, frame.pose.rot,
                                                            frame.pose.pos[:2], [mat]))


@pytest.mark.parametrize("layout", ["scatter", "cluster", "stack"])
def test_paper_layouts_use_independent_bounds_and_stacking(layout):
    paper = PaperPlacementConfig(x_m=(0.10, 0.75), y_m=(-0.45, 0.45),
                                 layouts={layout: 1}, scatter_separation_m=0.18)
    cfg = TabletopConfig(mat_count=(0, 0), paper_count=(6, 6), x_m=(2, 3), paper_placement=paper)
    plan = render_plan(Intrinsics.left_wrist())
    items, meta = add_tabletop(SceneBuilder(plan=plan), np.random.default_rng(20), np.random.default_rng(21), cfg, 0)
    xy = np.asarray([item.xy for item in items])
    assert np.all((xy >= [0.10, -0.45]) & (xy <= [0.75, 0.45]))
    assert all(item["layout"] == layout for item in meta)
    if layout == "scatter":
        distances = np.linalg.norm(xy[:, None] - xy, axis=-1) + np.eye(len(xy)) * 10
        assert distances.min() >= paper.scatter_separation_m
        assert np.ptp(xy[:, 0]) > 0.4 and np.ptp(xy[:, 1]) > 0.6
    if layout == "stack":
        assert np.ptp(xy, axis=0).max() < 0.1
        assert np.ptp([item.yaw for item in items]) < np.radians(40)
        assert all(items[i].bottom >= items[i - 1].top for i in range(1, len(items)))
    for index, item in enumerate(items):
        for lower in items[:index]:
            if item.overlaps(lower):
                assert item.bottom >= lower.top - 1e-9
    repeated, _ = add_tabletop(SceneBuilder(plan=plan), np.random.default_rng(20), np.random.default_rng(21), cfg, 0)
    assert items == repeated


@pytest.mark.parametrize("values", [{"layouts": {"stack": -1}}, {"layouts": {"unknown": 1}},
                                    {"width_m": [0, 0.2]}, {"x_m": [0.8, 0.1]},
                                    {"cluster_spread_m": -1}])
def test_invalid_paper_placement_distributions_are_refused(values):
    with pytest.raises(ValueError):
        PaperPlacementConfig(**values)


def test_full_heading_distribution_retains_palm_up_and_tabletop_support():
    phantom = build_phantom(hand_pose="closed").scaled(0.82, 0.76)
    cfg = PlacementConfig(radius=(0.22, 0.46), azimuth_deg=(-50, 50), yaw_deg=(-180, 180),
                          back_up_prob=0, palm_up_prob=1, roll_jitter_deg=15, tilt_jitter_deg=5)
    rng = np.random.default_rng(26)
    mat = TabletopItem("mat", "mat", (0.35, 0), (0.5, 0.5), 0, 0, 0.003)
    poses = [sample_rest_pose(rng, phantom, cfg, tabletop=[mat]) for _ in range(80)]
    xy = np.array([pose.pos[:2] for pose in poses])
    heading = np.array([np.degrees(np.arctan2(*pose.rot.apply([1, 0, 0])[:2][::-1])) for pose in poses])
    assert np.ptp(heading) > 320
    assert np.ptp(xy[:, 0]) > 0.2 and np.ptp(xy[:, 1]) > 0.45
    for pose in poses:
        assert pose.rot.apply([0, 0, -1])[2] > 0.6
        assert np.linalg.norm(pose.apply(phantom.vertices)[:, :2], axis=1).min() >= 0.12
        assert pose.pos[2] == pytest.approx(rest_height(phantom, pose.rot, pose.pos[:2], [mat]))


def test_impossible_forearm_placement_is_refused():
    with pytest.raises(ValueError, match="no forearm placement"):
        sample_rest_pose(np.random.default_rng(0), build_phantom(), PlacementConfig(), base_clearance=10, tries=2)
