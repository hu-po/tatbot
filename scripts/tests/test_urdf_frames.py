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


def test_physical_rig_uses_robot_centric_axes_and_models_the_pink_arm_alone():
    from pathlib import Path

    from ink_spec import base_from_root_matrix
    repo = Path(__file__).resolve().parents[2]
    chain = UrdfChain(repo/'urdf/tatbot.urdf')
    follower = chain.link_pose('right/base_link')
    center = chain.link_pose('rig_center')
    # +X is forward, +Y is the robot's own left, +Z up; the frame centred midway between the two arm mounts.
    np.testing.assert_allclose(follower[:3, 3], [0, -.2675, 0])
    np.testing.assert_allclose(center, np.eye(4))
    np.testing.assert_allclose((np.linalg.inv(center) @ chain.link_pose('600mm_vertical_top'))[:3, 3],
                               [-.0275, 0, .61])
    np.testing.assert_allclose(base_from_root_matrix(repo, 'right') @ follower, np.eye(4))
    # the blue (left) arm is the teleop leader alone, not modelled (owner, 2026-10-23)
    assert not any(name.startswith('left/') for name in chain.links)


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


def test_the_pink_arms_attachments_follow_its_carriage_and_camera():
    import xml.etree.ElementTree as ET
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    path = repo/'urdf/tatbot.urdf'
    links = {link.get('name'): link for link in ET.parse(path).getroot().findall('link')}
    assert links['right/tattoo_pen'].find('visual') is not None and 'right/tattoo_needle' in links
    chain = UrdfChain(path)
    pose = {'right/joint_1': -.2, 'right/left_carriage_joint': .008}
    for camera in ('right/realsense_link', 'right/realsense_lower_link'):
        mesh = links[camera].find('visual/geometry/mesh')
        assert mesh is not None and mesh.get('filename') == 'meshes/cameras/d405.stl'
        moved = {**pose, 'right/joint_5': .7}
        assert not np.allclose(chain.link_pose(camera, pose), chain.link_pose(camera, moved))


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
