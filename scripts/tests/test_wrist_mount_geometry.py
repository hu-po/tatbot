"""The pink wrist's camera mounts follow the stock bracket and ride rigidly on link_6."""
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts/lib'), str(REPO/'scripts/vision'), str(REPO/'scripts')]
from urdf_kinematics import UrdfChain  # noqa: E402
from wrist_cameras import optical_frames  # noqa: E402

# Vendor origins, from TrossenRobotics/ManiSkill-WidowX_AI/wxai_follower.urdf.
STOCK = {
    'mount_joint': ((.012, 0, 0), (0, 0, 0)),
    'joint': ((.02927207801, 0, .03824951197), (0, .3490658503988659, 0)),
    'link_joint': ((.01085, .009, .021), (0, 0, 0)),
    'color_joint': ((0, 0, 0), (0, 0, 0)),
    'depth_joint': ((0, 0, 0), (0, 0, 0)),
    'color_optical_joint': ((0, 0, 0), (-np.pi/2, 0, -np.pi/2)),
    'depth_optical_joint': ((0, 0, 0), (-np.pi/2, 0, -np.pi/2)),
}
# Each stock mount's chain prefix, and its mount's roll about link_6's x axis: the pink wrist's second D405
# (2026-10-23) is the same bracket turned 180 degrees under it, its mount joint measured (within MEASURED of stock).
MOUNTS = {'right/realsense_': 0.0, 'right/realsense_lower_': np.pi}
MEASURED = {'right/realsense_lower_': (0.003, np.radians(3.0))}   # m, rad


def test_every_wrist_d405_is_a_stock_trossen_mount_and_the_pink_wrist_has_two():
    tree = ET.parse(REPO/'urdf/tatbot.urdf').getroot()
    meshes = [m for m in tree.findall('.//visual/geometry/mesh')
              if m.get('filename') == 'meshes/cameras/d405.stl']
    assert len(meshes) == len(MOUNTS)
    for prefix, roll in MOUNTS.items():
        for suffix, (xyz, rpy) in STOCK.items():
            joint = tree.find(f'joint[@name="{prefix}{suffix}"]')
            assert joint.get('type') == 'fixed'
            origin = joint.find('origin')
            rpy = (roll, *rpy[1:]) if suffix == 'mount_joint' else rpy
            metres, radians = MEASURED[prefix] if suffix == 'mount_joint' and prefix in MEASURED else (1e-12, 1e-12)
            np.testing.assert_allclose(np.fromstring(origin.get('xyz'), sep=' '), xyz, atol=metres)
            got = np.fromstring(origin.get('rpy'), sep=' ')
            np.testing.assert_allclose(np.angle(np.exp(1j * (got - np.array(rpy)))), 0.0, atol=radians)
    for stream in ('color', 'depth'):
        bound = optical_frames(REPO, arm='right', stream=stream)
        assert bound['wrist_upper'] == f'right/realsense_{stream}_optical_frame'
        assert bound.get('wrist_lower', f'right/realsense_lower_{stream}_optical_frame') == \
            f'right/realsense_lower_{stream}_optical_frame'


def test_the_lower_wrist_camera_mirrors_the_upper_through_link_6():
    """Rolled about link_6's x axis: x the same, y and z negated, to within its measured mount's 2 deg and 3 mm."""
    chain = UrdfChain(REPO/'urdf/tatbot.urdf')
    in_link6 = {name: np.linalg.inv(chain.link_pose('right/link_6', {})) @ chain.link_pose(name, {})
                for name in ('right/realsense_color_optical_frame', 'right/realsense_lower_color_optical_frame')}
    upper, lower = in_link6.values()
    flip = np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(lower[:3, 3], flip @ upper[:3, 3], atol=0.003)
    assert np.degrees(np.arccos(np.clip(lower[:3, 2] @ (flip @ upper[:3, 2]), -1, 1))) < 2.0


def test_wrist_poses_move_with_the_arm_and_keep_their_mount():
    chain = UrdfChain(REPO/'urdf/tatbot.urdf')
    joints = {'right/joint_1': .3, 'right/joint_5': -.4}
    for frame in sorted(optical_frames(REPO, arm='right', stream='color').values()):
        initial = chain.link_pose(frame, {})
        assert np.linalg.norm(chain.link_pose(frame, joints)-initial) > .1
        mount_before = np.linalg.inv(chain.link_pose('right/link_6', {})) @ initial
        mount_after = np.linalg.inv(chain.link_pose('right/link_6', joints)) @ chain.link_pose(frame, joints)
        np.testing.assert_allclose(mount_after, mount_before, atol=1e-12)


def test_wrong_arm_mount_cannot_be_used_as_an_optical_binding(tmp_path):
    for name in ['config/arms.json', 'rust/visiond/config/vision.toml', 'urdf/tatbot.urdf']:
        out = tmp_path/name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/name, out)
    path = tmp_path/'urdf/tatbot.urdf'
    tree = ET.parse(path)
    tree.find('.//joint[@name="right/realsense_mount_joint"]/parent').set('link', 'left/link_6')
    tree.write(path)
    with pytest.raises(ValueError, match='own arm'):
        optical_frames(tmp_path, arm='right', stream='color')
