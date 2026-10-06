"""Missing or misspelled frames must not silently become identity poses."""

from pathlib import Path

import numpy as np
import pytest
from urdf_kinematics import UrdfChain


def test_unknown_link_cannot_masquerade_as_the_root(tmp_path):
    urdf = tmp_path / 'rig.urdf'
    urdf.write_text('''<robot name="rig">
      <link name="root"/><link name="base"/><link name="tool"/>
      <joint name="mount" type="fixed"><parent link="root"/><child link="base"/>
        <origin xyz="0.1 -0.3 0.2" rpy="0 0 1.5707963267948966"/></joint>
      <joint name="offset" type="fixed"><parent link="base"/><child link="tool"/>
        <origin xyz="0.2 0 0"/></joint></robot>''')
    chain = UrdfChain(urdf)
    np.testing.assert_allclose(chain.link_pose('root'), np.eye(4))
    np.testing.assert_allclose(chain.link_pose('tool')[:3, 3], [.1, -.1, .2])
    for name in ('missing', 'palette_tag', 'right/base_link'):
        with pytest.raises(ValueError, match='unknown URDF link'):
            chain.link_pose(name)


def test_planner_derives_translation_and_refuses_unhandled_mount_rotation(tmp_path):
    import shutil

    import arm_kinematics as dk
    directory = tmp_path / 'urdf'
    directory.mkdir()
    config = tmp_path / 'config'
    config.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[2] / 'config/arms.json', config / 'arms.json')
    path = directory / 'tatbot.urdf'
    template = '''<robot name="rig"><link name="root"/><link name="right/base_link"/>
      <joint name="mount" type="fixed"><parent link="root"/><child link="right/base_link"/>
      <origin xyz="0.1 -0.3 0.2" rpy="0 0 {yaw}"/></joint></robot>'''
    path.write_text(template.format(yaw=0))
    np.testing.assert_allclose(dk._base_translation_in_root(tmp_path), [.1, -.3, .2])
    path.write_text(template.format(yaw=.4))
    with pytest.raises(ValueError, match='parallel URDF-root and arm-base axes'):
        dk._base_translation_in_root(tmp_path)


def test_physical_rig_uses_robot_centric_left_and_right():
    from pathlib import Path

    from ink_spec import base_from_root_matrix
    repo = Path(__file__).resolve().parents[2]
    chain = UrdfChain(repo/'urdf/tatbot.urdf')
    follower = chain.link_pose('right/base_link')
    leader = chain.link_pose('left/base_link')
    center = chain.link_pose('rig_center')
    # +X is forward, +Y is the robot's own left, +Z is up.
    np.testing.assert_allclose(follower[:3, 3], [0, -.2675, 0])
    np.testing.assert_allclose(center, np.eye(4))
    np.testing.assert_allclose((follower[:3, 3]+leader[:3, 3])/2, center[:3, 3])
    np.testing.assert_allclose(leader[:3, 3]-follower[:3, 3], [0, .535, 0])
    np.testing.assert_allclose((np.linalg.inv(center) @ chain.link_pose('600mm_vertical_top'))[:3, 3],
                               [-.0275, 0, .61])
    np.testing.assert_allclose(base_from_root_matrix(repo, 'left') @ leader, np.eye(4))


def test_renamed_arm_id_uses_its_registered_urdf_prefix(tmp_path):
    import json

    from ink_spec import base_from_root_matrix

    repo = Path(__file__).resolve().parents[2]
    (tmp_path/'config').mkdir()
    (tmp_path/'urdf').mkdir()
    (tmp_path/'urdf/tatbot.urdf').symlink_to(repo/'urdf/tatbot.urdf')
    arms = json.loads((repo/'config/arms.json').read_text())
    arms['arms']['starboard'] = arms['arms'].pop('right')
    (tmp_path/'config/arms.json').write_text(json.dumps(arms))
    np.testing.assert_allclose(base_from_root_matrix(tmp_path, 'starboard'),
                               base_from_root_matrix(repo, 'right'))


def test_both_installed_attachments_follow_their_own_carriage_and_camera():
    import hashlib
    import json
    import xml.etree.ElementTree as ET
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    path = repo/'urdf/tatbot.urdf'
    robot = ET.parse(path).getroot()
    links = {link.get('name'): link for link in robot.findall('link')}
    for old in ('left/handle', 'left/leader_finger_left_index', 'left/leader_finger_left_thumb'):
        assert links[old].find('visual') is None
        assert links[old].find('collision') is None
    assert links['right/tattoo_pen'].find('visual') is not None
    assert links['left/ee_mount'].find('visual') is not None
    assert links['left/tattoo_pen'].find('visual') is not None
    # The laser TCP hangs off the measured lens-face touch-off in left/tool_mount.
    assert 'left/tattoo_needle' in links
    text = path.read_text()
    laser_block = text[text.index('picosecond-laser-pen), datasheet'):]
    assert 'Body profile and TCP resolve from the same measured touch-off' in laser_block
    chain = UrdfChain(path)
    pose = {'left/joint_1': .6, 'left/joint_4': -.3,
            'left/left_carriage_joint': .031,
            'right/joint_1': -.2, 'right/left_carriage_joint': .008}
    # The V5 print is bolted to the driven left carriage, rolled half a turn
    # about the carriage x axis relative to its CAD frame (which is the
    # mirrored right-carriage placement the CAD assumed).
    roll = np.diag([1., -1., -1., 1.])
    np.testing.assert_allclose(chain.link_pose('left/ee_mount', pose),
                               chain.link_pose('left/carriage_left', pose) @ roll, atol=1e-12)
    assert not np.allclose(chain.link_pose('left/ee_mount', pose),
                           chain.link_pose('left/carriage_right', pose))
    wrist_from_mount = (np.linalg.inv(chain.link_pose('left/link_6', pose))
                        @ chain.link_pose('left/ee_mount', pose))
    np.testing.assert_allclose(wrist_from_mount[:3, 3], [.0865, .023+.031, 0], atol=1e-12)
    np.testing.assert_allclose(wrist_from_mount[:3, :3], roll[:3, :3], atol=1e-12)
    for arm in ('left', 'right'):
        mesh = links[f'{arm}/realsense_link'].find('visual/geometry/mesh')
        assert mesh is not None and mesh.get('filename') == 'meshes/cameras/d405.stl'
        moved = {**pose, f'{arm}/joint_5': .7}
        assert not np.allclose(chain.link_pose(f'{arm}/realsense_link', pose),
                               chain.link_pose(f'{arm}/realsense_link', moved))
    directory = repo/'urdf/meshes/ee/leader-laser-v5'
    provenance = json.loads((directory/'provenance.json').read_text())
    assert provenance['frame'] == 'left/carriage_right'
    for name, digest in provenance['meshes'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    for name, digest in provenance['source_hashes'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == digest
    for name in ('left/ee_mount',):
        for mesh in links[name].iter('mesh'):
            assert mesh.get('scale') == '0.001 0.001 0.001'
            assert (repo/'urdf'/mesh.get('filename')).is_file()


def test_mimic_uses_multiplier_offset_and_rejects_broken_sources(tmp_path):
    template = '''<robot name="rig"><link name="root"/><link name="drive"/><link name="copy"/>
      <joint name="drive" type="prismatic"><parent link="root"/><child link="drive"/>
        <axis xyz="1 0 0"/>{drive_mimic}</joint>
      <joint name="copy" type="prismatic"><parent link="root"/><child link="copy"/>
        <axis xyz="0 1 0"/><mimic joint="{source}" multiplier="-2" offset="0.01"/>
      </joint></robot>'''
    path = tmp_path/'mimic.urdf'
    path.write_text(template.format(source='drive', drive_mimic=''))
    chain = UrdfChain(path)
    np.testing.assert_allclose(chain.link_pose('copy', {'drive': .03})[:3, 3], [0, -.05, 0])
    np.testing.assert_allclose(chain.link_pose('copy')[:3, 3], [0, .01, 0])
    for source, drive_mimic, message in (
        ('missing', '', 'unknown URDF mimic source'),
        ('drive', '<mimic joint="copy"/>', 'cycle in URDF mimic'),
    ):
        path.write_text(template.format(source=source, drive_mimic=drive_mimic))
        with pytest.raises(ValueError, match=message):
            UrdfChain(path).link_pose('copy')
