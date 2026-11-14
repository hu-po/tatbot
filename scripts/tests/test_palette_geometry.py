"""The installed palette's geometry: CAD manifest, tag frame, rims and the arm mount.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_palette_geometry.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]

import ink_spec  # noqa: E402


def _design() -> dict:
    return json.loads((REPO / ink_spec.palette_geometry(REPO)['cad_design']).read_text())


def _root_mm(design: dict) -> np.ndarray:
    """The design frame's palette_root: on the probe axis, the design's runtime ball_above_root_mm under the ball."""
    probe = design['probe']
    return np.array([*probe['xy'], probe['ball_z'] - design['runtime']['ball_above_root_mm']])


def test_runtime_tag_frame_matches_the_installed_cad_manifest():
    design = _design()
    tag = design['runtime']['installed_tag']
    transform = ink_spec.palette_from_tag(REPO)
    assert transform[:3, 3] == pytest.approx((np.array(tag['xyz_mm']) - _root_mm(design)) / 1000)
    yaw = np.deg2rad(tag['yaw_deg'])
    assert transform[:3, 1] == pytest.approx([-np.sin(yaw), np.cos(yaw), 0])
    assert ink_spec.tag_in_palette_root(REPO) == pytest.approx(transform[:3, 3])


def test_the_probe_and_camera_frames_are_v11s_so_their_calibration_carries_over():
    """v9 keeps the probe and the Pi camera 122 mm apart at one height; its root keeps the ball 55.6 mm over it."""
    import xml.etree.ElementTree as ET
    joints = {j.find('child').get('link'): j.find('origin') for j in ET.parse(REPO / 'urdf/palette.urdf').getroot()
              .iter('joint')}
    xyz = {name: [float(v) for v in joints[name].get('xyz').split()] for name in ('palette_probe_ball', 'palette_camera')}
    assert xyz['palette_probe_ball'] == pytest.approx([0, 0, 0.0556])
    assert xyz['palette_camera'] == pytest.approx([0.122, 0, 0.0556])
    assert [float(v) for v in joints['palette_camera'].get('rpy').split()] == pytest.approx([0, 0, np.pi])


def test_base_from_root_is_the_inverse_of_the_urdf_mount():
    """base_from_root maps the arm's base origin (in root) back to the origin."""
    import xml.etree.ElementTree as ET
    bfr = ink_spec.base_from_root_matrix(REPO, "right")
    # The base origin in root is the URDF mount; base_from_root takes it home.
    mount = ET.parse(REPO / "urdf" / "tatbot.urdf").getroot().find('joint[@name="right/mount_joint"]/origin')
    base_in_root = np.fromstring(mount.get("xyz"), sep=" ")
    np.testing.assert_allclose((bfr @ np.append(base_in_root, 1))[:3], 0.0, atol=1e-12)
    # a pure rotation+translation (rigid): top-left 3x3 orthonormal
    assert np.allclose(bfr[:3, :3] @ bfr[:3, :3].T, np.eye(3), atol=1e-9)


import json  # noqa: E402


def test_installed_geometry_matches_cad_and_robot_has_no_palette():
    import xml.etree.ElementTree as ET
    design = _design()
    root = _root_mm(design)
    layout = ink_spec.palette_layout_from_urdf(REPO)
    # four front-to-back groups, each cap at its group's column and row, on its pocket floor
    expected = {}
    for group in design['caps']['groups']:
        for n, y in enumerate(group['row_y'], group['id_start']):
            floor = design['caps']['top_z'] - group['depth']
            expected[f"inkcap_{group['size']}_{n}"] = (np.array([group['x'], y, floor]) - root) / 1000
    assert list(layout) == list(expected)
    assert [n for n in layout if n.startswith('inkcap_small')] == [f'inkcap_small_{i}' for i in range(1, 11)]
    for name, xyz in layout.items():
        assert np.array(xyz) == pytest.approx(expected[name], abs=1e-9)
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
    # v9's pockets: the large caps' rims 7 mm over the root, the medium and small caps' 6 mm
    assert {round(z, 6) for _, _, z in rims.values()} == {0.007, 0.006}
    # an explicit rim_z_m entry wins for that cap
    p = tmp_path / 'config/palette_geometry.json'
    geometry = json.loads(p.read_text())
    geometry['rim_z_m'] = {'inkcap_small_1': .033}
    p.write_text(json.dumps(geometry))
    assert ink_spec.palette_rim_layout(tmp_path)['inkcap_small_1'][2] == pytest.approx(.033)
    # but one that puts the rim below its own pocket floor (-4 mm on v9) is a typo, not a cap
    geometry['rim_z_m']['inkcap_small_1'] = -.010
    p.write_text(json.dumps(geometry))
    with pytest.raises(ValueError, match='invalid rim height'):
        ink_spec.palette_rim_layout(tmp_path)


