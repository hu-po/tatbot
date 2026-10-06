"""The simulator's palette is the one config/palette_geometry.json names.

Calibration palette v11: six caps L-M-S-S-M-L on a crescent around the probe,
every rim level at 32 mm above the rail tops, and the sticker on its rotated deck seat, turned 135 degrees as measured. Engine-free: the loader reads JSON, YAML and URDF only.
"""

from __future__ import annotations

import json
import math
import shutil

import ink_spec
import numpy as np
import pytest
from tatbot_sim import palette as sim_palette
from tatbot_sim import tools

PALETTE_SLOTS = ('inkcap_large_1', 'inkcap_medium_1', 'inkcap_small_1',
             'inkcap_small_2', 'inkcap_medium_2', 'inkcap_large_2')


@pytest.fixture(scope='module')
def scene():
    return sim_palette.load(tools.REPO)


@pytest.fixture
def repo(tmp_path):
    """A copy of just the inputs the loader reads, to change one at a time."""
    scene = sim_palette.load(tools.REPO)
    for name in dict.fromkeys((sim_palette.GEOMETRY_RELPATH, ink_spec.PALETTE_RELPATH,
                               'config/arms.json', scene.urdf, scene.mesh,
                               scene.collision_mesh, scene.tag_mesh, scene.tag_inventory)):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tools.REPO / name, tmp_path / name)
    return tmp_path


def _edit_geometry(repo, **changes):
    path = repo / sim_palette.GEOMETRY_RELPATH
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


def test_the_scene_is_the_palette_the_geometry_names(scene):
    geometry = json.loads((tools.REPO / sim_palette.GEOMETRY_RELPATH).read_text())
    assert (scene.revision, scene.urdf) == (geometry['revision'], geometry['urdf'])
    assert scene.revision == 'calibration_palette_v11'
    assert scene.urdf == 'urdf/palette.urdf'
    assert scene.mesh == scene.collision_mesh == 'urdf/meshes/palette/calibration_palette_v11.stl'
    assert scene.mesh_scale == (0.001, 0.001, 0.001)


def test_six_caps_run_large_medium_small_small_medium_large(scene):
    assert tuple(scene.palette) == PALETTE_SLOTS == tuple(scene.rims)
    assert [slot.size.size_id for slot in scene.palette.values()] == [
        'large', 'medium', 'small', 'small', 'medium', 'large']
    y = [scene.rims[name][1] for name in PALETTE_SLOTS]
    assert y == sorted(y, reverse=True), 'the crescent runs from +Y to -Y'


def test_every_rim_is_its_floor_plus_the_caps_outside_height(scene):
    """The URDF's cap frames are support floors, not rims."""
    floors = ink_spec.palette_layout_from_urdf(tools.REPO)
    for name, slot in scene.palette.items():
        x, y, rim = scene.rims[name]
        assert (x, y) == floors[name][:2]
        assert math.isclose(rim, floors[name][2] + slot.size.height_m, abs_tol=1e-12)
        assert math.isclose(rim, 0.032, abs_tol=1e-9), f'{name} rim is not level with the rest'
    assert scene.rim_basis == 'cad-estimate-unmeasured-rims'


def test_the_simulator_dips_at_the_rims_hardware_dips_at(scene):
    hardware = ink_spec.palette_rim_layout(tools.REPO)
    assert set(hardware) == set(scene.rims)
    for name, rim in scene.rims.items():
        assert rim == pytest.approx(hardware[name], abs=1e-12)


def test_the_tag_sits_on_the_urdf_palette_tag_frame_turn_included(scene):
    assert scene.tag_xyz_m == pytest.approx((0.068, -0.026, 0.0082))
    assert scene.tag_rpy == pytest.approx((0.0, 0.0, 3 * math.pi / 4))
    assert scene.tag_xyz_m == pytest.approx(ink_spec.tag_in_palette_root(tools.REPO))


def test_the_tag_mesh_is_named_by_the_inventory_palette_target(scene):
    assert scene.tag_mesh == 'urdf/meshes/tags/36h11_000_46mm/tag.glb'
    inventory = json.loads((tools.REPO / scene.tag_inventory).read_text())
    target = inventory['targets']['palette']
    family = target.get('family', inventory.get('family')).removeprefix('apriltag_')
    assert scene.tag_mesh.split('/')[-2] == (
        f"{family}_{target['ids'][0]:03d}_{round(target['edge_m'] * 1000)}mm")


def test_the_scene_pose_is_synthetic_and_rigid(scene):
    source, transform = sim_palette.base_transform(tools.REPO, scene)
    assert source == 'synthetic-installed-cad'
    geometry = json.loads((tools.REPO / sim_palette.GEOMETRY_RELPATH).read_text())
    root = np.array([*geometry['simulation_root_xyz_m'], 1.0])
    assert np.allclose(transform[:3, 3], (ink_spec.base_from_root_matrix(tools.REPO) @ root)[:3])
    assert np.allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3))


def test_a_slot_the_urdf_has_no_cap_for_is_refused(repo):
    with (repo / ink_spec.PALETTE_RELPATH).open('a') as palette:
        palette.write('  inkcap_small_3:\n    size: small\n    arm: right\n')
    with pytest.raises(ValueError, match='differ from palette.yaml'):
        sim_palette.load(repo)


def test_an_inventory_tag_without_a_rendered_mesh_is_refused(repo):
    (repo / sim_palette.FIDUCIAL_INVENTORY_RELPATH).write_text(json.dumps({
        'schema_version': 2, 'family': 'apriltag_36h11',
        'targets': {'palette': {'family': 'apriltag_36h11', 'ids': [7], 'edge_m': 0.046}}}))
    with pytest.raises(ValueError, match='36h11_007_46mm/tag.glb is missing'):
        sim_palette.load(repo)


def test_a_geometry_of_another_schema_is_refused(repo):
    _edit_geometry(repo, schema_version=1)
    with pytest.raises(ValueError, match='schema 1 is not 2'):
        sim_palette.load(repo)


def test_a_measured_rim_replaces_the_cad_estimate_and_says_so(repo):
    _edit_geometry(repo, rim_z_m={'inkcap_large_1': 0.0325})
    scene = sim_palette.load(repo)
    assert scene.rims['inkcap_large_1'][2] == 0.0325
    assert scene.rim_basis == 'partly-measured-rims'


def test_a_scene_is_not_placed_by_another_palettes_geometry(repo):
    scene = sim_palette.load(repo)
    _edit_geometry(repo, revision='another_palette')
    with pytest.raises(ValueError, match='changed revision'):
        sim_palette.base_transform(repo, scene)
