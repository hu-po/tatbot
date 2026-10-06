"""Inspection uses the existing owner without selecting a wrist/approach pose."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import wrist_pose
from cli_runner import tatbot
from tatbot_cli import nodes


@pytest.fixture
def owned_inspection(monkeypatch, tmp_path):
    build = {'schema': 'tatbot.arm-guide-build/1', 'native_backend_compiled': True,
             'sources': {}, 'target': 'x86_64-unknown-linux-gnu', 'profile': 'release',
             'sdk': {'version': '1.8.5', 'revision': 'fdfd9f68f57b3bd05c4e85011fa5c11296525b2f',
                     'library_sha256': 'a' * 64}}
    owner = SimpleNamespace(commands=[], closed=False, fault=None, close_code=0)

    def wait(event, **kwargs):
        if event == 'ready':
            return {'build': build, 'tool_datasheet_sha256': 'b' * 64}
        return {'event': event, 'worker': {'fault': owner.fault},
                'measured': {'positions': [.1] * 6}}

    def close():
        owner.closed = True
        return owner.close_code

    owner.send = owner.commands.append
    owner.wait = wait
    owner.close = close
    monkeypatch.setattr(wrist_pose, 'Owner', lambda *args: owner)
    monkeypatch.setattr(wrist_pose.recipe, 'selected_arm', lambda *args: SimpleNamespace(arm_id='left', controller_role='leader'))
    monkeypatch.setattr(wrist_pose.recipe, 'native_source_digests', lambda repo: {})
    monkeypatch.setattr(wrist_pose, 'sha256_file', lambda path: 'b' * 64)
    monkeypatch.setattr(wrist_pose, 'read_workspace', lambda repo: {'left': {'tool_id': 'example-tool'}})
    monkeypatch.setattr(wrist_pose.subprocess, 'check_output', lambda *args, **kwargs: json.dumps(build))
    monkeypatch.setattr(wrist_pose.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setenv('TATBOT_ESTOP_DEVICE', 'mock-only')
    args = SimpleNamespace(arm='blue', ee_tool='example-tool', inspect=True, operator_go=True,
                           wrist_deg=None, base_deg=None, hold_seconds=0)
    return args, SimpleNamespace(dir=tmp_path, run_id='synthetic-inspection'), owner


def test_inspection_retains_measurement_and_releases_without_pose_commands(owned_inspection):
    args, log, owner = owned_inspection
    wrist_pose.run(args, log)
    assert owner.commands == [{'cmd': 'connect'}, {'cmd': 'status'}, {'cmd': 'release'}]
    assert owner.closed
    assert all((log.dir / name).is_file() for name in ('native-build.json', 'connected.json', 'status.json', 'released.json'))
    assert not (log.dir / 'wrist.json').exists()


def test_inspection_requires_authorization_before_owner_is_opened(owned_inspection):
    args, log, owner = owned_inspection
    args.operator_go = False
    with pytest.raises(wrist_pose.CaptureError, match='operator-go'):
        wrist_pose.run(args, log)
    assert not owner.commands and not owner.closed


def test_inspection_fault_closes_the_existing_owner_without_motion(owned_inspection):
    args, log, owner = owned_inspection
    owner.fault = 'controller fault'
    with pytest.raises(wrist_pose.CaptureError, match='arm fault'):
        wrist_pose.run(args, log)
    assert owner.commands == [{'cmd': 'connect'}, {'cmd': 'status'}]
    assert owner.closed
    assert not (log.dir / 'released.json').exists()


def test_inspection_does_not_hide_failed_idle_cleanup(owned_inspection):
    args, log, owner = owned_inspection
    owner.close_code = 1
    with pytest.raises(wrist_pose.CaptureError, match='failed to close'):
        wrist_pose.run(args, log)
    assert owner.closed


def test_existing_pose_still_requests_the_reviewed_target(owned_inspection):
    args, log, owner = owned_inspection
    args.inspect, args.wrist_deg = False, 90
    wrist_pose.run(args, log)
    assert [command['cmd'] for command in owner.commands] == ['connect', 'wrist', 'release']
    assert owner.closed and (log.dir / 'wrist.json').is_file()


def test_cli_inspection_records_go_and_has_no_pose_target():
    result = tatbot('--dry-run', '--json', '--ee-tool', 'picosecond-laser-pen',
                    'calib', 'inspect', '--arm', 'blue', '--operator-go', node=nodes.example_node('arm'))
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['tier'] == 'motion-auto'
    assert '--inspect' in plan['argv'] and '--operator-go' in plan['argv']
    assert plan['argv'][-4:] == ['--hold-seconds', '5.0', '--ee-tool', 'picosecond-laser-pen']
    assert '--wrist-deg' not in plan['argv'] and '--base-deg' not in plan['argv']


def test_cli_inspection_refuses_missing_go():
    result = tatbot('--dry-run', '--json', 'calib', 'inspect', '--arm', 'blue', node=nodes.example_node('arm'))
    assert result.returncode == 2


@pytest.mark.parametrize('target', [('--wrist-deg', '30'), ('--base-deg', '20')])
def test_cli_inspection_refuses_pose_targets(target):
    result = tatbot('--dry-run', '--json', 'calib', 'inspect', '--arm', 'blue', '--operator-go',
                    *target, node=nodes.example_node('arm'))
    assert result.returncode == 2


@pytest.mark.parametrize('device, expected', [
    ('/dev/example-estop', '/dev/example-estop'),
    ('udp://:7640?from=192.0.2.9', 'udp://0.0.0.0:7640?from=192.0.2.9'),
    ('udp://127.0.0.1:7640?from=192.0.2.9', 'udp://127.0.0.1:7640?from=192.0.2.9'),
])
def test_native_estop_normalizes_literal_ipv4_and_preserves_serial(tmp_path, device, expected):
    assert wrist_pose.recipe.native_estop_device(tmp_path, device) == expected


def test_native_estop_resolves_relay_role_from_execution_owner(tmp_path):
    config = tmp_path / 'config'
    config.mkdir()
    (config / 'nodes.json').write_text(json.dumps({
        'example-relay': {'roles': ['estop-relay'], 'lan': '192.0.2.9'},
        'example-arm': {'roles': ['arm'], 'lan': '192.0.2.10'},
    }))
    assert wrist_pose.recipe.native_estop_device(tmp_path, 'udp://:7640?from=estop-relay') == (
        'udp://0.0.0.0:7640?from=192.0.2.9')


@pytest.mark.parametrize('relay_records', [
    {},
    {'first': {'roles': ['estop-relay'], 'lan': '192.0.2.9'},
     'second': {'roles': ['estop-relay'], 'lan': '192.0.2.10'}},
    {'first': {'roles': ['estop-relay']}},
    {'first': {'roles': ['estop-relay'], 'lan': 'hostname'}},
])
def test_native_estop_refuses_missing_ambiguous_or_invalid_relay(tmp_path, relay_records):
    config = tmp_path / 'config'
    config.mkdir()
    (config / 'nodes.json').write_text(json.dumps(relay_records))
    with pytest.raises(ValueError):
        wrist_pose.recipe.native_estop_device(tmp_path, 'udp://:7640?from=estop-relay')


@pytest.mark.parametrize('device', [
    'udp://:7640', 'udp://:7640?from=', 'udp://:0?from=192.0.2.9',
    'udp://:99999?from=192.0.2.9', 'udp://host:7640?from=192.0.2.9',
    'udp://:7640?from=192.0.2.9&from=192.0.2.10', 'udp://:7640?from=192.0.2.9&extra=1',
    'udp://:7640/path?from=192.0.2.9', 'udp://:7640?from=192.0.2.9#extra',
    'udp://user@127.0.0.1:7640?from=192.0.2.9',
])
def test_native_estop_refuses_ambiguous_or_malformed_udp_uri(tmp_path, device):
    with pytest.raises(ValueError):
        wrist_pose.recipe.native_estop_device(tmp_path, device)


def test_inspection_passes_only_owner_resolved_relay_to_native(owned_inspection, monkeypatch):
    args, log, owner = owned_inspection
    monkeypatch.setenv('TATBOT_ESTOP_DEVICE', 'udp://:7640?from=estop-relay')
    monkeypatch.setattr(nodes, 'load', lambda repo: {'example': {'roles': ['estop-relay'], 'lan': '192.0.2.9'}})
    argv = []
    monkeypatch.setattr(wrist_pose, 'Owner', lambda command, path: (argv.extend(command), owner)[1])
    wrist_pose.run(args, log)
    assert argv[argv.index('--estop-device') + 1] == 'udp://0.0.0.0:7640?from=192.0.2.9'
    assert owner.closed


def test_inspection_holds_the_current_pose_before_final_status(owned_inspection, monkeypatch):
    args, log, owner = owned_inspection
    args.hold_seconds = .3
    clock = [0.]
    monkeypatch.setattr(wrist_pose.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(wrist_pose.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    wrist_pose.run(args, log)
    assert [c['cmd'] for c in owner.commands] == ['connect', 'status', 'status', 'status', 'release']
    assert json.loads((log.dir / 'hold-status.json').read_text())['seconds'] == .3


def test_inspection_capture_subscribes_to_owner_and_closes_before_release(owned_inspection, monkeypatch):
    import sys

    args, log, owner = owned_inspection
    args.capture_wrist = True
    calls = []
    cameras = SimpleNamespace(close=lambda: calls.append('closed'))
    def capture(camera, out, k, joints, carriage, stamp, frames):
        import math
        assert camera is cameras and all(math.isnan(q) for q in joints) and math.isnan(carriage)
        assert owner.commands[-1]['cmd'] == 'status'
        calls.append('capture')
    module = SimpleNamespace(open_cameras=lambda fake, from_owner, arm: (
        calls.append((fake, from_owner, arm)), cameras)[1], capture=capture, MIN_FRAMES=8)
    monkeypatch.setitem(sys.modules, 'vision_capture', module)
    monkeypatch.setattr(wrist_pose.recipe, 'bind_wrist_capture', lambda *args: {'motion_authority': False})
    wrist_pose.run(args, log)
    assert calls == [(False, True, 'left'), 'capture', 'closed']
    assert owner.commands[-1]['cmd'] == 'release' and owner.closed



def test_owner_capture_failure_still_closes_the_arm_owner(owned_inspection, monkeypatch):
    import sys

    args, log, owner = owned_inspection
    args.capture_wrist = True
    calls = []
    def failure(*args):
        raise RuntimeError('camera unavailable')
    cameras = SimpleNamespace(close=lambda: calls.append('camera closed'))
    monkeypatch.setitem(sys.modules, 'vision_capture', SimpleNamespace(
        open_cameras=lambda *args: cameras, capture=failure, MIN_FRAMES=8))
    with pytest.raises(RuntimeError, match='camera unavailable'):
        wrist_pose.run(args, log)
    assert calls == ['camera closed'] and owner.closed
    assert not (log.dir / 'wrist-pose-binding.json').exists()


@pytest.fixture
def overhead_camera(owned_inspection, monkeypatch):
    import numpy as np

    pytest.importorskip('cv2', reason='original overhead preview retention requires OpenCV; run the full dependency profile')
    args, log, owner = owned_inspection
    args.capture_overhead = True
    camera = SimpleNamespace(closed=False, fail=False)
    def capture(after, *, retain_original):
        assert not owner.closed and owner.commands[-1]['cmd'] == 'status'
        assert retain_original
        if camera.fail:
            raise RuntimeError('overhead unavailable')
        return {'original_packet': b'synthetic exact original packet', 'image': np.zeros((3, 4, 3), np.uint8),
                'stamp_ns': after+1, 'metadata': {'timestamps': {'normalized_unix_ns': after+1}},
                'depth_metadata': None, 'depth_m': None}
    camera.capture = capture
    camera.close = lambda: setattr(camera, 'closed', True)
    monkeypatch.setattr(wrist_pose, '_overhead_camera', lambda: camera)
    def bind(packet, telemetry):
        assert owner.closed and camera.closed, 'binding must use the finalized native flight'
        assert packet.read_bytes() == b'synthetic exact original packet'
        return {'motion_authority': False, 'world_registration': 'unmeasured'}
    monkeypatch.setattr(wrist_pose.recipe, 'bind_owner_packet', bind)
    return args, log, owner, camera


def test_overhead_capture_keeps_original_bytes_during_hold_and_binds_after_release(overhead_camera):
    args, log, owner, camera = overhead_camera
    wrist_pose.run(args, log)
    assert owner.closed and camera.closed and owner.commands[-1]['cmd'] == 'release'
    assert (log.dir/'overhead-pose-binding.json').is_file()
    manifest = json.loads((log.dir/'overhead/capture.json').read_text())
    assert manifest['world_registration'] == 'unmeasured' and manifest['physical_accuracy_bound_m'] is None


def test_overhead_failure_preserves_native_terminal_cleanup(overhead_camera):
    args, log, owner, camera = overhead_camera
    camera.fail = True
    with pytest.raises(RuntimeError, match='overhead unavailable'):
        wrist_pose.run(args, log)
    assert owner.closed and camera.closed
    assert not (log.dir/'overhead-pose-binding.json').exists()


def test_failed_overhead_binding_does_not_claim_pose_or_hide_idle_release(overhead_camera, monkeypatch):
    args, log, owner, camera = overhead_camera
    def refuse(*args):
        raise wrist_pose.recipe.RecipeError('owner exposure is not bracketed')
    monkeypatch.setattr(wrist_pose.recipe, 'bind_owner_packet', refuse)
    with pytest.raises(wrist_pose.recipe.RecipeError, match='not bracketed'):
        wrist_pose.run(args, log)
    assert owner.closed and camera.closed and (log.dir/'released.json').is_file()
    assert not (log.dir/'overhead-pose-binding.json').exists()


def test_cli_overhead_capture_is_an_explicit_authorized_native_hold():
    for command, target in [('inspect', ()), ('pose', ('--wrist-deg', '90'))]:
        result = tatbot('--dry-run', '--json', '--ee-tool', 'picosecond-laser-pen', 'calib', command,
                        '--arm', 'blue', '--operator-go', '--capture-overhead', *target,
                        node=nodes.example_node('arm'))
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert '--capture-overhead' in plan['argv'] and '--operator-go' in plan['argv']
    result = tatbot('--dry-run', '--json', '--ee-tool', 'picosecond-laser-pen', 'calib', 'pose',
                    '--arm', 'blue', '--wrist-deg', '90', '--capture-overhead', node=nodes.example_node('arm'))
    assert result.returncode != 0 and 'operator-go' in result.stderr
