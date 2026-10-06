"""Physical camera mounts follow the stock bracket and independent arm motion."""
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


def test_each_arm_has_one_stock_trossen_mount_and_no_old_lower_camera():
    tree = ET.parse(REPO/'urdf/tatbot.urdf').getroot()
    links = {link.get('name') for link in tree.findall('link')}
    assert not any('realsense_lower' in name for name in links)
    meshes = [m for m in tree.findall('.//visual/geometry/mesh')
              if m.get('filename') == 'meshes/cameras/d405.stl']
    assert len(meshes) == 2
    # Vendor origins, from TrossenRobotics/ManiSkill-WidowX_AI/wxai_follower.urdf.
    stock = {
        'mount_joint': ((.012, 0, 0), (0, 0, 0)),
        'joint': ((.02927207801, 0, .03824951197), (0, .3490658503988659, 0)),
        'link_joint': ((.01085, .009, .021), (0, 0, 0)),
        'color_joint': ((0, 0, 0), (0, 0, 0)),
        'depth_joint': ((0, 0, 0), (0, 0, 0)),
        'color_optical_joint': ((0, 0, 0), (-np.pi/2, 0, -np.pi/2)),
        'depth_optical_joint': ((0, 0, 0), (-np.pi/2, 0, -np.pi/2)),
    }
    for arm, role in [('left', 'wrist_left'), ('right', 'wrist_upper')]:
        for stream in ('color', 'depth'):
            assert optical_frames(REPO, arm=arm, stream=stream) == {
                role: f'{arm}/realsense_{stream}_optical_frame'}
        for suffix, (xyz, rpy) in stock.items():
            joint = tree.find(f'joint[@name="{arm}/realsense_{suffix}"]')
            assert joint.get('type') == 'fixed'
            origin = joint.find('origin')
            np.testing.assert_allclose(np.fromstring(origin.get('xyz'), sep=' '), xyz, atol=1e-12)
            np.testing.assert_allclose(np.fromstring(origin.get('rpy'), sep=' '), rpy, atol=1e-12)


def test_wrist_pose_responds_only_to_its_own_arm():
    chain = UrdfChain(REPO/'urdf/tatbot.urdf')
    frames = {arm: next(iter(optical_frames(REPO, arm=arm, stream='color').values()))
              for arm in ('left', 'right')}
    initial = {arm: chain.link_pose(frame, {}) for arm, frame in frames.items()}
    for moved, parked in [('left', 'right'), ('right', 'left')]:
        joints = {f'{moved}/joint_1': .3, f'{moved}/joint_5': -.4}
        np.testing.assert_allclose(chain.link_pose(frames[parked], joints), initial[parked], atol=1e-12)
        assert np.linalg.norm(chain.link_pose(frames[moved], joints)-initial[moved]) > .1
        mount_before = np.linalg.inv(chain.link_pose(f'{moved}/link_6', {})) @ initial[moved]
        mount_after = np.linalg.inv(chain.link_pose(f'{moved}/link_6', joints)) @ chain.link_pose(frames[moved], joints)
        np.testing.assert_allclose(mount_after, mount_before, atol=1e-12)


def test_wrong_arm_mount_cannot_be_used_as_an_optical_binding(tmp_path):
    for name in ['config/arms.json', 'rust/visiond/config/vision.toml', 'urdf/tatbot.urdf']:
        out = tmp_path/name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/name, out)
    path = tmp_path/'urdf/tatbot.urdf'
    tree = ET.parse(path)
    tree.find('.//joint[@name="left/realsense_mount_joint"]/parent').set('link', 'right/link_6')
    tree.write(path)
    with pytest.raises(ValueError, match='own arm'):
        optical_frames(tmp_path, arm='left', stream='color')


def test_drawing_geometry_cannot_reuse_a_camera_on_the_other_arm(tmp_path):
    """Each arm's capture registers through its own arm's chain
    (`CAMERA_LINKS[arm]`, keyed on the capture's roles): a `wrist_left`
    capture is placed by `left/...`, never by the right wrist's transform; the
    retired `wrist_lower` maps nowhere; one capture never mixes the arms."""
    import capture_geometry
    assert capture_geometry.CAMERA_LINKS['right'] == {'wrist_upper': 'right/realsense_depth_optical_frame'}
    assert capture_geometry.CAMERA_LINKS['left'] in ({}, {'wrist_left': 'left/realsense_depth_optical_frame'})
    links = []

    class Chain:
        def link_pose(self, link, values):
            links.append((link, sorted(values)))
            return np.eye(4)

    def capture(*roles):
        path = tmp_path/f'{"-".join(roles)}.npz'
        np.savez(path, joints=np.zeros(6), carriage_m=.002,
                 **{f'depth_{role}': np.full((2, 2), 1000, dtype=np.uint16) for role in roles},
                 **{f'units_m_{role}': .0001 for role in roles},
                 **{f'intrinsics_{role}': [100., 100., 0., 0., 2., 2.] for role in roles})
        return path

    with np.load(capture('wrist_lower')) as data, pytest.raises(capture_geometry.StageError, match='wrist_lower.*invalid capture camera roles'):
        capture_geometry._capture_cloud(data, Chain())
    with np.load(capture('wrist_left', 'wrist_upper')) as data, pytest.raises(capture_geometry.StageError, match="one arm's"):
        capture_geometry._capture_cloud(data, Chain())
    with np.load(capture('wrist_left')) as data:
        if capture_geometry.CAMERA_LINKS['left']:
            _, roles = capture_geometry._capture_cloud(data, Chain())
            assert roles == ['wrist_left']
            assert links == [('left/realsense_depth_optical_frame',
                              sorted([f'left/joint_{i}' for i in range(6)] + ['left/left_carriage_joint']))]
        else:
            with pytest.raises(capture_geometry.StageError, match='wrist_left.*absent'):
                capture_geometry._capture_cloud(data, Chain())
