"""The leader arm's laser mount: frames, meshes and datum agree with the CAD record."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/lib'), str(REPO / 'scripts'), str(REPO / 'scripts/vision')]
import tool_spec  # noqa: E402

URDF = REPO / 'urdf/tatbot.urdf'
MESHES = REPO / 'urdf/meshes/ee/leader-laser-v5'


def _rotation(rpy):
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _joint(tree, name):
    joint = tree.find(f'joint[@name="{name}"]')
    assert joint is not None, name
    origin = joint.find('origin')
    return (joint.find('parent').get('link'), joint.find('child').get('link'),
            np.fromstring(origin.get('xyz'), sep=' '), np.fromstring(origin.get('rpy'), sep=' '))


def _chain():
    from urdf_kinematics import UrdfChain
    return UrdfChain(URDF)


def _in_link6(chain, link, carriage_m=0.0):
    values = {'left/left_carriage_joint': carriage_m}
    return np.linalg.inv(chain.link_pose('left/link_6', values)) @ chain.link_pose(link, values)


def test_leader_mount_rides_the_left_carriage_rolled_half_a_turn():
    """The V5 print is bolted to left/carriage_left, rolled 180 degrees about
    the carriage x axis relative to its CAD frame (the 2x3 M3 pattern allows
    it); the URDF carries the CAD datum through that placement."""
    tree = ET.parse(URDF).getroot()
    links = {link.get('name') for link in tree.findall('link')}
    thumb = tree.find('link[@name="left/leader_finger_left_thumb"]')
    assert thumb is not None and thumb.find('visual') is None, 'the leader fingers keep frames only'
    assert {'left/ee_mount', 'left/tool_mount'} <= links
    assert 'left/laser_pen_visual' not in links, 'one pen: the datasheet-generated block'
    parent, child, xyz, rpy = _joint(tree, 'left/ee_mount_joint')
    assert (parent, child) == ('left/carriage_left', 'left/ee_mount')
    np.testing.assert_allclose(xyz, 0, atol=1e-12)
    roll = _rotation(rpy)
    np.testing.assert_allclose(roll, np.diag([1, -1, -1]), atol=1e-12)
    provenance = json.loads((MESHES / 'provenance.json').read_text())
    installed = provenance['installed_placement']
    assert (installed['parent'], installed['child']) == ('left/carriage_left', 'left/ee_mount')
    np.testing.assert_allclose(_rotation(installed['rpy']), roll, atol=1e-12)
    parent, child, xyz, rpy = _joint(tree, 'left/tool_mount_joint')
    assert (parent, child) == ('left/carriage_left', 'left/tool_mount')
    datum = provenance['tool_datum']
    np.testing.assert_allclose(xyz, roll @ (np.array(datum['clamp_fat_center_carriage_mm']) / 1000), atol=1e-9)
    rotation = _rotation(rpy)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-9)
    np.testing.assert_allclose(rotation[:, 2], roll @ datum['tool_axis_carriage'], atol=1e-9)
    np.testing.assert_allclose(rotation[:, 0], [0, 0, -1], atol=1e-9)
    # In link 6 the installed laser datum points the same way as the
    # follower's ballpoint datum: the sessions' shared tool axis is right for
    # both arms, and the CAD's mirrored (+x+y) axis was not.
    chain = _chain()
    left = _in_link6(chain, 'left/tool_mount')
    values = {'right/left_carriage_joint': 0.0}
    right = np.linalg.inv(chain.link_pose('right/link_6', values)) @ chain.link_pose('right/tool_mount', values)
    np.testing.assert_allclose(left[:3, 2], right[:3, 2], atol=1e-9)
    np.testing.assert_allclose(left[:3, 2], [np.sqrt(0.5), -np.sqrt(0.5), 0], atol=1e-9)
    # The mount rides the driven carriage: +y of link 6 per metre of travel.
    np.testing.assert_allclose(_in_link6(chain, 'left/ee_mount', 0.01)[:3, 3] - _in_link6(chain, 'left/ee_mount')[:3, 3],
                               [0, 0.01, 0], atol=1e-12)


def test_replaced_carriers_use_the_current_measured_tag_frames():
    tree = ET.parse(URDF).getroot()
    for arm, filename, ids in [('left', 'wrist_tags_measured_left.json', (1, 5, 30)),
                               ('right', 'wrist_tags_measured.json', (2, 3, 4))]:
        layout = json.loads((REPO / 'config' / filename).read_text())
        assert layout['calibration_status'] == 'calibrated'
        assert {int(tag) for tag in layout['tags']} == set(ids)
        assert {link.get('name') for link in tree.findall('link')
                if link.get('name').startswith(arm + '/wrist_tag')} == {
                    f'{arm}/wrist_tag{tag}' for tag in ids}


def test_leader_mount_meshes_are_the_recorded_build():
    tree = ET.parse(URDF).getroot()
    provenance = json.loads((MESHES / 'provenance.json').read_text())
    assert provenance['frame'] == 'left/carriage_right', 'the assembly meshes stay in their CAD frame'
    mount = tree.find('link[@name="left/ee_mount"]')
    referenced = [m.get('filename') for m in mount.findall('visual/geometry/mesh')]
    carrier = json.loads((REPO / provenance['installed_carrier']).read_text())
    meshes = {name: digest for name, digest in provenance['meshes'].items() if 'carrier' not in name}
    meshes['left_fiducial_cube_47mm.stl'] = carrier['meshes']['left_fiducial_cube_47mm.stl']
    assert sorted(Path(f).name for f in referenced) == sorted(meshes)
    for visual in mount.findall('visual'):
        assert visual.find('origin').get('rpy') == '0 0 0', 'placement lives in the joint, not the visuals'
        mesh = visual.find('geometry/mesh')
        assert mesh.get('scale') == '0.001 0.001 0.001'
        path = REPO / 'urdf' / mesh.get('filename')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == meshes[path.name], path.name
    assert provenance['installed_tags']['inventory_target'] == 'wrist_left'


def test_leader_laser_is_modelled_as_a_standoff_tool_at_its_measured_face():
    tree = ET.parse(URDF).getroot()
    workspace = tool_spec.read_workspace(REPO)
    assert tool_spec.active_tool_id(REPO, 'left', workspace) == 'picosecond-laser-pen'
    tip = tool_spec.tip_offset_m(workspace, 'left')
    assert tip is not None, 'the 2026-09-19 lens-face touch-off in left/tool_mount'
    laser = tool_spec.load_tool('picosecond-laser-pen', REPO)
    assert laser.mount_frame('left') == 'left/tool_mount'
    assert not laser.contact and laser.standoff_m > 0 and laser.contact_radius_m is None
    assert _joint(tree, 'left/tattoo_pen_joint')[:2] == ('left/tool_mount', 'left/tattoo_pen')
    _, _, tcp, _ = _joint(tree, 'left/tattoo_needle_joint')
    np.testing.assert_allclose(tcp, [0, 0, laser.protrusion_m], atol=1e-9)
    assert tree.find('link[@name="left/tattoo_needle"]/collision') is None, 'a standoff tool has no contact sphere'
    # The generated body ends at the measured lens face and the working point
    # floats the standoff past it along the same axis.
    chain = _chain()
    mount = _in_link6(chain, 'left/tool_mount')
    face = mount[:3, :3] @ np.asarray(tip) + mount[:3, 3]
    needle = _in_link6(chain, 'left/tattoo_needle')[:3, 3]
    np.testing.assert_allclose(np.linalg.norm(needle - face), laser.standoff_m, atol=1e-4)
    axis = face - mount[:3, 3]
    np.testing.assert_allclose((needle - face) / laser.standoff_m, axis / np.linalg.norm(axis), atol=1e-6)
    # The current mechanical contact is expressed through the installed V5 mount.
    np.testing.assert_allclose(face * 1000, [215.32978, -6.98780, -3.086], atol=0.05)
