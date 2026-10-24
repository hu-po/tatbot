"""One USB owner per camera; an arm's recording and policy views are its own cameras', never another's."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

import wrist_cameras as cameras  # noqa: E402


def write_config(root, entries):
    path = root / 'rust/visiond/config/vision.toml'
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for entry in entries:
        lines = ['[[cameras.realsense]]']
        lines += [f'{key} = {json.dumps(value)}' for key, value in entry.items()]
        for stream in ('color', 'depth'):
            lines += [f'[cameras.realsense.{stream}]', 'width = 640', 'height = 480',
                      'fps_num = 30', 'fps_den = 1']
        blocks.append('\n'.join(lines))
    path.write_text('\n'.join(blocks))
    return path


@pytest.fixture
def rig(tmp_path):
    # The pink wrist's two D405s (2026-10-23), here on two USB owners.
    entries = [
        {'name': 'depth-a', 'serial': 'device-a', 'role': 'wrist_lower', 'group': 'd405', 'arm': 'right',
         'owner_role': 'lower-cameras', 'optical_frame': 'right/realsense_lower_color_optical_frame',
         'depth_optical_frame': 'right/realsense_lower_depth_optical_frame'},
        {'name': 'depth-b', 'serial': 'device-b', 'role': 'wrist_upper', 'group': 'd405', 'arm': 'right',
         'owner_role': 'realsense', 'optical_frame': 'right/realsense_color_optical_frame',
         'depth_optical_frame': 'right/realsense_depth_optical_frame'},
    ]
    write_config(tmp_path, entries)
    (tmp_path / 'config').mkdir()
    shutil.copyfile(REPO / 'config/arms.json', tmp_path / 'config/arms.json')
    (tmp_path / 'urdf').mkdir()
    shutil.copyfile(REPO / 'urdf/tatbot.urdf', tmp_path / 'urdf/tatbot.urdf')
    (tmp_path / 'config/nodes.json').write_text(json.dumps({
        'capture-a': {'roles': ['lower-cameras'], 'ssh': 'user@lower.invalid'},
        'capture-b': {'roles': ['realsense', 'arm'], 'ssh': 'user@right.invalid'},
    }))
    return tmp_path, entries


def test_each_capture_owner_selects_only_its_device(rig):
    root, _ = rig
    assert cameras.owned_names(root, 'capture-a') == ['depth-a']
    assert cameras.owned_names(root, 'capture-b') == ['depth-b']
    with pytest.raises(ValueError, match='owns no'):
        cameras.owned_names(root, 'operator')


def test_descriptions_resolve_without_device_ownership_and_keep_physical_arms(rig):
    root, _ = rig
    (root / 'config/nodes.json').unlink()
    lower, upper = cameras.describe(root)
    assert (upper.role, upper.arm, lower.role, lower.arm) == ('wrist_upper', 'right', 'wrist_lower', 'right')
    assert upper.optical_frame == 'right/realsense_color_optical_frame'
    assert lower.optical_frame == 'right/realsense_lower_color_optical_frame'
    assert (upper.width, upper.height, upper.fps) == (640, 480, 30)
    assert upper.intrinsic_basis == 'nominal-fov'
    assert 'serial' not in json.dumps(upper.as_dict())
    # Each optical frame has a rigid mount on its arm's link_6; the teleop leader carries no camera.
    for description in (lower, upper):
        cameras.fixed_chain(REPO, 'right/link_6', description.optical_frame)
        with pytest.raises(ValueError, match='fixed camera chain'):
            cameras.fixed_chain(REPO, 'right/link_5_unknown', description.optical_frame)
    assert cameras.describe(root, arms=('left',)) == ()


def test_description_rejects_ambiguous_or_cross_arm_camera_geometry(rig):
    root, entries = rig
    del entries[0]['optical_frame']
    write_config(root, entries)
    with pytest.raises(ValueError, match='explicit optical frames'):
        cameras.describe(root)
    entries[0]['optical_frame'] = 'left/realsense_color_optical_frame'
    write_config(root, entries)
    with pytest.raises(ValueError, match='another arm'):
        cameras.describe(root)


def test_remote_checkpoint_needs_its_config_before_any_action_query(tmp_path):
    with pytest.raises(ValueError, match='--checkpoint-config'):
        cameras.read_checkpoint_config('/server-only/model')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'input_features': {
        'observation.images.wrist_upper': {'shape': [3, 480, 640]},
    }}))
    path, value = cameras.read_checkpoint_config('/server-only/model', config)
    assert path == config
    cameras.validate_checkpoint_views(value, ('wrist_upper',), use_depth=False)
    with pytest.raises(ValueError, match='camera shape'):
        cameras.validate_checkpoint_views(value, ('wrist_upper',), use_depth=False,
                                          image_shapes={'wrist_upper': (3, 240, 320)})
    with pytest.raises(ValueError, match='no view substitution'):
        cameras.validate_checkpoint_views(value, ('wrist_upper', 'wrist_lower'), use_depth=False)


def test_an_arms_recording_needs_every_one_of_its_cameras_on_this_host(rig):
    root, _ = rig
    with pytest.raises(ValueError, match='another capture host'):
        cameras.for_arm(root, 'right', 'capture-b')
    with pytest.raises(ValueError, match='no D405 camera is assigned to the left arm'):
        cameras.for_arm(root, 'left', 'capture-a')


def test_two_cameras_on_the_same_physical_arm_remain_supported(rig):
    root, entries = rig
    entries[0].update(owner_role='realsense')
    write_config(root, entries)
    assert len(cameras.for_arm(root, 'right', 'capture-b')) == 2
    assert cameras.owned_names(root, 'capture-b') == ['depth-a', 'depth-b']


def test_unknown_mount_allows_raw_capture_but_refuses_arm_observations(rig):
    root, entries = rig
    del entries[0]['arm']
    write_config(root, entries)
    assert cameras.owned_names(root, 'capture-a') == ['depth-a']
    with pytest.raises(ValueError, match='assignment is incomplete'):
        cameras.for_arm(root, 'right', 'capture-b')


@pytest.mark.parametrize('field', ['serial', 'name', 'role'])
def test_duplicate_camera_identity_is_rejected(rig, field):
    root, entries = rig
    entries[0][field] = entries[1][field]
    write_config(root, entries)
    with pytest.raises(ValueError, match='distinct'):
        cameras.owned_names(root, 'capture-a')


def test_ambiguous_capture_owner_never_chooses_first_node(rig):
    root, _ = rig
    path = root / 'config/nodes.json'
    mapping = json.loads(path.read_text())
    mapping['other'] = {'roles': ['realsense']}
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match='exactly one owner'):
        cameras.owned_names(root, 'capture-b')


def checkpoint(*roles):
    return {'input_features': {f'observation.images.{r}': {'shape': [3, 480, 640]} for r in roles}}


def test_checkpoint_must_match_the_actual_local_view_set(rig):
    root, _ = rig
    roles = tuple(c['role'] for c in cameras.mounted_for_arm(root, 'right'))
    cameras.validate_checkpoint_views(checkpoint('wrist_lower', 'wrist_upper'), roles, use_depth=False)
    cameras.validate_checkpoint_views(checkpoint('wrist_lower', 'wrist_upper', 'wrist_lower_depth', 'wrist_upper_depth'),
                                      roles, use_depth=True)
    for policy in [checkpoint('wrist_upper'), checkpoint('wrist_left'), checkpoint(),
                   checkpoint('wrist_lower', 'wrist_upper', 'wrist_upper_depth')]:
        with pytest.raises(ValueError, match='no view substitution'):
            cameras.validate_checkpoint_views(policy, roles, use_depth=False)


def _capture_module(root, monkeypatch, name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts/vision_capture.py')
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    monkeypatch.setattr(capture, 'REPO', root)
    return capture


def test_session_capture_refuses_an_arm_whose_cameras_another_host_owns(rig, monkeypatch):
    root, entries = rig
    capture = _capture_module(root, monkeypatch, 'vision_capture_wrist_test')
    monkeypatch.setenv('TATBOT_NODE', 'capture-b')
    with pytest.raises(SystemExit, match='another capture host'):
        capture.registry_cameras('right')
    entries[0]['owner_role'] = 'realsense'
    write_config(root, entries)
    assert capture.registry_cameras('right') == {'wrist_lower': 'device-a', 'wrist_upper': 'device-b'}


def test_session_capture_names_the_arm_whose_wrist_cameras_answer(rig, monkeypatch):
    import numpy as np

    root, entries = rig
    capture = _capture_module(root, monkeypatch, 'vision_capture_arm_test')
    # The arm's owner answers for its wrist cameras under their retained roles; a registry role this
    # checkout keeps no geometry for is refused naming the arm, never a default.
    entries[0].update(owner_role='realsense', role='wrist_probe')
    write_config(root, entries)
    monkeypatch.setenv('TATBOT_NODE', 'capture-b')
    assert capture.registry_cameras('right') == {'wrist_probe': 'device-a', 'wrist_upper': 'device-b'}
    assert capture.ROLES == ('wrist_lower', 'wrist_upper')   # the live registry's, both on the pink wrist
    with pytest.raises(SystemExit, match='not configured'):
        capture.registry_cameras('centre')
    with pytest.raises(ValueError, match='not configured'):
        capture.open_cameras(True, False, 'centre')
    # `once` and `serve` require --arm; the synthetic cameras answer with that arm's roles only.
    out = root / 'fake-right'
    assert capture.main(['once', str(out), '--fake', '--arm', 'right']) == 0
    assert (out / 'capture-1.done').is_file()
    with np.load(out / 'capture-1.npz') as arrays:
        assert json.loads(str(arrays['camera_roles'])) == ['wrist_probe', 'wrist_upper']
        assert 'depth_wrist_probe' in arrays.files and 'depth_wrist_lower' not in arrays.files
    with pytest.raises(SystemExit):
        capture.main(['once', str(root / 'fake-none'), '--fake'])


@pytest.mark.parametrize('node,expected', [('capture-a', 'depth-a'), ('capture-b', 'depth-b')])
def test_service_passes_an_exact_sensor_selection_before_running_capture(rig, node, expected):
    root, entries = rig
    stage = root / 'release'
    source = stage / 'source'
    scripts = source / 'scripts'
    lib = scripts / 'lib'
    lib.mkdir(parents=True)
    (scripts / 'fleet_service.sh').write_text((REPO / 'scripts/fleet_service.sh').read_text())
    (lib / 'wrist_cameras.py').symlink_to(REPO / 'scripts/lib/wrist_cameras.py')
    (lib / 'tatbot_cli').symlink_to(REPO / 'scripts/lib/tatbot_cli', target_is_directory=True)
    (lib / 'cli_hint.sh').write_text('cli_hint::note() { :; }\n')
    (lib / 'runlog.sh').write_text('runlog::init() { RUN_DIR=/tmp/test-run; }\n'
                                  'runlog::run() { printf "%s\\n" "$@"; }\n')
    (source / 'config').mkdir()
    mapping = json.loads((root / 'config/nodes.json').read_text())
    mapping['router'] = {'roles': ['bus-router'], 'lan': '192.0.2.10'}
    (source / 'config/nodes.json').write_text(json.dumps(mapping))
    shutil.copyfile(REPO / 'config/arms.json', source / 'config/arms.json')
    home = root / 'home'
    home.mkdir()
    write_config(source, entries)
    result = subprocess.run(['bash', str(scripts / 'fleet_service.sh'), 'visiond-d405'],
                            env={**os.environ, 'HOME': str(home), 'TATBOT_NODE': node},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()
    assert argv.count('--sensor') == 1
    assert argv[argv.index('--sensor') + 1] == expected
    assert argv[argv.index('--group') + 1] == 'd405'


def test_live_manifest_puts_both_wrist_cameras_on_the_pink_arm_under_one_owner():
    """Both D405s are on the pink wrist since 2026-10-23, upper and lower, on one node's USB; the blue arm has none."""
    entries = cameras.registry(REPO)
    mapping = cameras.nodes.load(REPO)
    assert {c['arm'] for c in entries} == {'right'}
    assert len({cameras.owner(c, mapping) for c in entries}) == 1
    assert sorted(cameras.CAPTURE_ROLES['right']) == ['wrist_lower', 'wrist_upper']
    assert cameras.CAPTURE_ROLES['left'] == ()
    assert cameras.capture_arm(['wrist_upper']) == 'right'
    assert cameras.capture_arm(['wrist_lower', 'wrist_upper']) == 'right'
    for roles in ([], ['wrist_left'], ['wrist_upper', 'wrist_upper'], ['wrist_probe']):
        with pytest.raises(ValueError, match='camera roles'):
            cameras.capture_arm(roles)
