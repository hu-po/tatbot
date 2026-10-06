"""The installed palette's geometry: CAD manifest, tag frame, rims and the arm mount.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_palette_geometry.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]

import ink_spec  # noqa: E402


def test_runtime_tag_frame_matches_the_installed_cad_manifest():
    design = json.loads((REPO / ink_spec.palette_geometry(REPO)['cad_design']).read_text())['installed_tag']
    transform = ink_spec.palette_from_tag(REPO)
    assert transform[:3, 3] == pytest.approx(np.array(design['xyz_mm']) / 1000)
    yaw = np.deg2rad(design['yaw_deg'])
    assert transform[:3, 1] == pytest.approx([-np.sin(yaw), np.cos(yaw), 0])
    assert ink_spec.tag_in_palette_root(REPO) == pytest.approx(transform[:3, 3])


def test_tip_pivot_recovers_a_known_tag_centre():
    """The authoritative solve: rolling the planted tip about one point recovers
    that point in the base frame, whatever the tip offset — the FK guarantee the
    tip source rests on."""
    import il_touchoff as touchoff
    rng = np.random.default_rng(0)
    tag = np.array([0.30, -0.02, 0.16])       # the planted point (base)
    tip = np.array([0.0597, -0.0032, -0.0005])  # tip in the EE frame
    poses = []
    for _ in range(12):
        ax = rng.normal(size=3)
        ax = ax / np.linalg.norm(ax)
        rmat = _rot(ax, rng.uniform(-0.5, 0.5))
        poses.append(_homog(rmat, tag - rmat @ tip))
    fit = touchoff.solve_pivot_trimmed(poses)
    assert np.linalg.norm(fit["pivot"] - tag) < 1e-6
    assert np.linalg.norm(fit["p"] - tip) < 1e-6


def _rot(axis, angle):
    axis = np.array(axis, float)
    x, y, z = axis / np.linalg.norm(axis)
    c, sn = np.cos(angle), np.sin(angle)
    k = 1 - c
    return np.array([
        [c + x * x * k, x * y * k - z * sn, x * z * k + y * sn],
        [y * x * k + z * sn, c + y * y * k, y * z * k - x * sn],
        [z * x * k - y * sn, z * y * k + x * sn, c + z * z * k]])


def test_base_from_root_is_the_inverse_of_the_urdf_mount():
    """base_from_root maps the arm's base origin (in root) back to the origin —
    the transform cal_tip needs because touchoff solves in the root frame."""
    import xml.etree.ElementTree as ET
    bfr = ink_spec.base_from_root_matrix(REPO, "right")
    # The base origin in root is the URDF mount; base_from_root takes it home.
    mount = ET.parse(REPO / "urdf" / "tatbot.urdf").getroot().find('joint[@name="right/mount_joint"]/origin')
    base_in_root = np.fromstring(mount.get("xyz"), sep=" ")
    np.testing.assert_allclose((bfr @ np.append(base_in_root, 1))[:3], 0.0, atol=1e-12)
    # a pure rotation+translation (rigid): top-left 3x3 orthonormal
    assert np.allclose(bfr[:3, :3] @ bfr[:3, :3].T, np.eye(3), atol=1e-9)


import json  # noqa: E402


def _homog(rmat, t):
    m = np.eye(4)
    m[:3, :3] = rmat
    m[:3, 3] = t
    return m


def test_installed_geometry_matches_cad_and_robot_has_no_palette():
    import xml.etree.ElementTree as ET
    design = json.loads((REPO / ink_spec.palette_geometry(REPO)['cad_design']).read_text())['caps']
    layout = ink_spec.palette_layout_from_urdf(REPO)
    # the crescent from +Y to -Y, every cap centre on the arc about the probe axis
    assert list(layout) == ['inkcap_large_1', 'inkcap_medium_1', 'inkcap_small_1',
                            'inkcap_small_2', 'inkcap_medium_2', 'inkcap_large_2']
    assert [name.split('_')[1] for name in layout] == design['order_positive_y_to_negative_y']
    assert all(np.hypot(x, y) == pytest.approx(design['arc_radius'] / 1000) for x, y, _ in layout.values())
    assert [y for _, y, _ in layout.values()] == sorted((y for _, y, _ in layout.values()), reverse=True)
    robot = ET.parse(REPO / 'urdf/tatbot.urdf').getroot()
    assert not any(link.get('name', '').startswith(('palette_', 'inkcap_')) for link in robot.findall('link'))
    camera = next(j for j in robot.findall('joint') if j.get('name') == 'camera1_mount_joint')
    assert camera.find('parent').get('link') == 'rig_center'
    assert [float(v) for v in camera.find('origin').get('xyz').split()] == pytest.approx([-.074, 0, .085])


def test_rim_heights_come_from_the_pocket_floor_and_the_cap_outside(tmp_path):
    import shutil
    (tmp_path / 'config').mkdir()
    (tmp_path / 'urdf').mkdir()
    for path in ('config/arms.json', 'config/palette.yaml', 'config/palette_geometry.json',
                 'urdf/palette.urdf'):
        shutil.copy(REPO / path, tmp_path / path)
    rims = ink_spec.palette_rim_layout(tmp_path)
    # pocket floor plus the cap's OUTER height, read from the datasheet rather
    # than pinned here: the outside is what rests in the pocket, and the cap
    # dimensions are externally sourced and have been restated once.
    sizes = ink_spec.load_palette(tmp_path)
    floors = ink_spec.palette_layout_from_urdf(tmp_path)
    for slot in rims:
        assert rims[slot][2] == pytest.approx(floors[slot][2] + sizes[slot].size.height_m)
    # v11 seats every cap flush with its holder: all six rims 32 mm above the rail tops
    design = json.loads((REPO / ink_spec.palette_geometry(REPO)['cad_design']).read_text())
    assert {round(z, 6) for _, _, z in rims.values()} == {design['caps']['rim_z'] / 1000}
    # an explicit rim_z_m entry wins for that cap
    p = tmp_path / 'config/palette_geometry.json'
    geometry = json.loads(p.read_text())
    geometry['rim_z_m'] = {'inkcap_small_1': .033}
    p.write_text(json.dumps(geometry))
    assert ink_spec.palette_rim_layout(tmp_path)['inkcap_small_1'][2] == pytest.approx(.033)
    # but one that puts the rim below its own pocket floor is a typo, not a cap
    geometry['rim_z_m']['inkcap_small_1'] = .020
    p.write_text(json.dumps(geometry))
    with pytest.raises(ValueError, match='invalid rim height'):
        ink_spec.palette_rim_layout(tmp_path)


