"""Configured arms resolve without deriving geometry from their names."""
from __future__ import annotations

import copy
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import arm_kinematics as kin
import pytest
import wrist_cameras
from tatbot_cli import arms

REPO = Path(__file__).resolve().parents[2]


def _configured_arm_config(config, count):
    registry = json.loads((REPO / 'config/arms.json').read_text())
    right = registry['arms']['right']
    left = registry['arms']['left']
    registry['arms'] = {'pink': right}
    if count >= 2:
        registry['arms']['blue'] = left
    if count >= 3:
        third = copy.deepcopy(right)
        third.update(profile_ip_field='third_ip', controller_config='config/trossen/third.yaml',
                     sdk_end_effector='wxai_v0_third', workspace_section='third', urdf_prefix='third')
        registry['arms']['green'] = third
    (config / 'arms.json').write_text(json.dumps(registry))
    profile = json.loads((REPO / 'config/profiles/trossen-wxai.json').read_text())
    profile['driver']['third_ip'] = None
    profiles = config / 'profiles'
    profiles.mkdir()
    profile_path = profiles / 'trossen-wxai.json'
    profile_path.write_text(json.dumps(profile))
    trossen = config / 'trossen'
    trossen.mkdir()
    for name in ('leader', 'follower', 'third'):
        source = 'follower' if name == 'third' else name
        shutil.copyfile(REPO / f'config/trossen/{source}.yaml', trossen / f'{name}.yaml')
    workspace = (REPO / 'config/workspace.yaml').read_text()
    if count >= 3:
        workspace += '\nthird:\n  tip_frame: third/tool_mount\n  tool_id: lutin-ballpoint-dot\n'
    (config / 'workspace.yaml').write_text(workspace)
    shutil.copytree(REPO / 'config/tools', config / 'tools')
    return profile_path


def _configured_urdf(tmp_path, count):
    urdf = ET.parse(REPO / 'urdf/tatbot.urdf')
    root = urdf.getroot()
    if count >= 3:
        for source in list(root):
            if source.tag not in ('link', 'joint') or not source.get('name', '').startswith('right/'):
                continue
            copied = copy.deepcopy(source)
            for element in copied.iter():
                for key, value in element.attrib.items():
                    if value.startswith('right/'):
                        element.set(key, 'third/' + value.removeprefix('right/'))
            if copied.get('name') == 'third/mount_joint':
                copied.find('origin').set('xyz', '0.8 -0.3 0.0')
            root.append(copied)
        for role, x in (('probe_a', '0.01'), ('probe_b', '-0.01')):
            for stream in ('color', 'depth'):
                frame = f'third/{role}_{stream}_optical_frame'
                ET.SubElement(root, 'link', name=frame)
                joint = ET.SubElement(root, 'joint', name=f'{frame}_joint', type='fixed')
                ET.SubElement(joint, 'origin', xyz=f'{x} 0 0', rpy='0 0 0')
                ET.SubElement(joint, 'parent', link='third/link_6')
                ET.SubElement(joint, 'child', link=frame)
    urdf_dir = tmp_path / 'urdf'
    urdf_dir.mkdir()
    urdf.write(urdf_dir / 'tatbot.urdf')


def _configured_vision(tmp_path, count):
    cameras = [{'name': 'right-wrist', 'serial': 'right-serial', 'role': 'wrist_upper',
                'group': 'd405', 'arm': 'pink', 'owner_role': 'capture-pink'}]
    if count >= 3:
        cameras += [
            {'name': role, 'serial': role, 'role': role, 'group': 'd405',
             'arm': 'green', 'owner_role': 'capture-green',
             'optical_frame': f'third/{role}_color_optical_frame',
             'depth_optical_frame': f'third/{role}_depth_optical_frame'}
            for role in ('probe_a', 'probe_b')
        ]
    vision = tmp_path / 'rust/visiond/config/vision.toml'
    vision.parent.mkdir(parents=True)
    blocks = []
    for camera in cameras:
        fields = '\n'.join(f'{key} = {json.dumps(value)}' for key, value in camera.items())
        blocks.append('[[cameras.realsense]]\n' + fields + '\n'
                      '[cameras.realsense.color]\nwidth = 640\nheight = 480\nfps_num = 30\nfps_den = 1\n'
                      '[cameras.realsense.depth]\nwidth = 640\nheight = 480\nfps_num = 30\nfps_den = 1\n')
    vision.write_text('\n'.join(blocks))


def configured_repo(tmp_path, count):
    config = tmp_path / 'config'
    config.mkdir()
    profile_path = _configured_arm_config(config, count)
    _configured_urdf(tmp_path, count)
    _configured_vision(tmp_path, count)
    return tmp_path, profile_path


@pytest.mark.parametrize('count', [1, 2, 3])
def test_collection_bindings_and_wrist_variations(tmp_path, count):
    repo, _ = configured_repo(tmp_path, count)
    expected = ('pink', 'blue', 'green')[:count]
    assert tuple(arms.load(repo)) == expected
    roles = wrist_cameras.capture_roles(repo)
    assert tuple(roles) == expected
    assert roles['pink'] == ('wrist_upper',)
    for arm in expected:
        model = kin.ArmModel(arm, repo=repo)
        prefix = arms.load(repo)[arm].urdf_prefix
        assert model.frame == f'{prefix}/base_link' and model.carriage_joint == f'{prefix}/left_carriage_joint'
    pink = kin.ArmModel('pink', repo=repo)
    assert pink.prefix == 'right' and pink.controller_role == 'follower'
    assert wrist_cameras.capture_arm(['wrist_upper'], repo=repo) == 'pink'
    if count >= 2:
        assert roles['blue'] == ()
        assert wrist_cameras.optical_frames(repo, arm='blue', stream='depth') == {}
    if count >= 3:
        frames = wrist_cameras.optical_frames(repo, arm='green', stream='depth')
        assert set(frames) == {'probe_a', 'probe_b'}
        assert wrist_cameras.capture_arm(['probe_a', 'probe_b'], repo=repo) == 'green'


def test_cross_arm_camera_frame_is_refused(tmp_path):
    repo, _ = configured_repo(tmp_path, 3)
    path = repo / 'rust/visiond/config/vision.toml'
    path.write_text(path.read_text().replace('third/probe_a_depth_optical_frame',
                                             'right/realsense_depth_optical_frame', 1))
    with pytest.raises(ValueError, match='another arm'):
        wrist_cameras.optical_frames(repo, arm='green', stream='depth')
