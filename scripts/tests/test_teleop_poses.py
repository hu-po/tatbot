"""Offline explicit-model FK and recorded follower timestamp contract."""
import hashlib
import json
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/vision'), str(REPO / 'scripts/lib')]
import arm_kinematics as kin  # noqa: E402
import ballpoint_fixture  # noqa: E402
import kinematic_calibration  # noqa: E402
import tool_spec  # noqa: E402
from teleop_poses import export  # noqa: E402
from urdf_kinematics import UrdfChain  # noqa: E402

# 2026-10-04's adopted pink offsets: the test's own, whatever config/workspace.yaml holds today.
OFFSETS = [0., .0098, .0003, -.0195, -.0023, 0., 0.]


def inputs(tmp_path):
    model = tmp_path / 'model.urdf'
    model.write_bytes((REPO / 'urdf/tatbot.urdf').read_bytes())
    workspace = tmp_path / 'workspace.yaml'
    workspace.write_text(ballpoint_fixture.workspace_text())
    rows = np.zeros((3, 47))
    rows[:, :5] = [[0., .0001, .0002, .0003, .0004],
                   [.0025, .0026, .0027, .0028, .0029],
                   [.0100, .0101, .0102, .0103, .0104]]
    rows[:, 5:12] = 9.  # Leader is not the measured follower.
    rows[:, 40:47] = 8.  # Command is not the measured follower.
    rows[:, 19:26] = [.1, -.4, .6, -.2, .3, -.1, 0.]
    rows[1:, 25] = .002
    rows[2, 19] += .4
    log = tmp_path / 'flight.wxtl'
    header = struct.pack('<8sQddddQq', b'WXTLOG1\0', 7, .0025, .01, .02, 0., 1,
                         1_700_000_000_123_456_789)
    log.write_bytes(header + rows.astype('<f8').tobytes())
    return log, model, workspace, rows


def calibrated_chain(tmp_path, model, offsets, arm='right'):
    """The model as the drawing stack carries it: joint origins turned by the encoder-zero offsets."""
    tree = ET.parse(model)
    kinematic_calibration.apply_origins(tree.getroot().findall('joint'), arm, offsets)
    path = tmp_path / f'calibrated-{arm}.urdf'
    tree.write(path)
    return UrdfChain(path)


def workspace_offsets(workspace, arm='right'):
    return tool_spec.parse_simple_yaml(workspace.read_text())[arm].get(kinematic_calibration.KEY)


def with_offsets(text: str, offsets) -> str:
    """The right section carrying `offsets` as its joint_offsets_rad, in place of any it has."""
    right, sep, left = text.partition('\nleft:')
    right = re.sub(r'^  joint_offsets_rad:.*\n', '', right, flags=re.M)
    right = right.replace('right:\n', f'right:\n  joint_offsets_rad: {json.dumps(offsets)}\n', 1)
    return right + sep + left


def fault_touchoff(text: str, field: str, value: str) -> str:
    """Set the right arm's touch-off `field` to `value`, whatever today's calibration measured.

    The workspace copy is the repository's live one while the ballpoint is
    fitted, so a literal number here lasts exactly until the next touch-off.
    """
    right, sep, left = text.partition('\nleft:')
    changed = re.sub(rf'^(\s+{field}: ).*$', rf'\g<1>{value}', right, count=1, flags=re.M)
    assert changed != right, f'{field} is not in the right arm touch-off'
    return changed + sep + left


def test_actual_follower_carriage_tip_axes_and_integer_read_time(tmp_path):
    log, model, workspace, rows = inputs(tmp_path)
    out = tmp_path / 'out'
    report = export(log, out, model, workspace, 'lutin-ballpoint-dot')
    data = np.load(out / 'poses.npz', allow_pickle=False)
    np.testing.assert_array_equal(data['follower_positions'], rows[:, 19:26])
    np.testing.assert_array_equal(data['follower_read_unix_ns'],
        np.array([1_700_000_000_123_756_789, 1_700_000_000_126_256_789, 1_700_000_000_133_756_789]))
    tcp, link6 = data['root_from_tcp'], data['root_from_link6']
    np.testing.assert_allclose(link6[0], link6[1], atol=1e-12)
    np.testing.assert_allclose(tcp[1, :3, 3]-tcp[0, :3, 3], link6[0, :3, :3] @ [0, .002, 0], atol=1e-12)
    chain = calibrated_chain(tmp_path, model, workspace_offsets(workspace))
    mount = chain.link_pose('right/tool_mount', kin.ArmModel('right').joint_map(rows[0, 19:25], 0))
    tip = tool_spec.tip_offset_m(tool_spec.parse_simple_yaml(workspace.read_text()))
    np.testing.assert_allclose(tcp[0, :3, 3], mount[:3, 3] + mount[:3, :3] @ tip)
    np.testing.assert_allclose(tcp[0, :3, :3], mount[:3, :3])
    assert not np.allclose(tcp[0, :3, :3], link6[0, :3, :3])
    assert not report['hardware_authority'] and 'not hardware sample time' in report['timestamp_semantics']
    assert report['inputs']['urdf'] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert json.loads((out / 'manifest.json').read_text()) == report


def test_poses_carry_the_workspace_joint_offsets(tmp_path):
    """The follower's poses are the drawing stack's arm model at the recorded joints: the joint
    offsets turn the chain (kinematic_calibration.apply_origins), millimetres from the nominal
    URDF; the npz keeps the joints as read and the manifest names the offsets."""
    log, model, workspace, rows = inputs(tmp_path)
    workspace.write_text(with_offsets(workspace.read_text(), OFFSETS))
    report = export(log, tmp_path / 'out', model, workspace, 'lutin-ballpoint-dot')
    assert report['joint_offsets_rad'] == OFFSETS
    data = np.load(tmp_path / 'out/poses.npz', allow_pickle=False)
    np.testing.assert_array_equal(data['follower_positions'], rows[:, 19:26])
    calibrated, nominal = calibrated_chain(tmp_path, model, OFFSETS), UrdfChain(model)
    tip = tool_spec.tip_offset_m(tool_spec.parse_simple_yaml(workspace.read_text()))
    for i in range(len(rows)):
        values = kin.ArmModel('right').joint_map(rows[i, 19:25], rows[i, 25])
        mount = calibrated.link_pose('right/tool_mount', values)
        np.testing.assert_allclose(data['root_from_link6'][i], calibrated.link_pose('right/link_6', values), atol=1e-12)
        np.testing.assert_allclose(data['root_from_tcp'][i, :3, 3], mount[:3, :3] @ tip + mount[:3, 3], atol=1e-12)
        drift = data['root_from_link6'][i, :3, 3] - nominal.link_pose('right/link_6', values)[:3, 3]
        assert np.linalg.norm(drift) > 1e-3


@pytest.mark.parametrize('fault', ['truncated', 'nonfinite', 'unordered', 'missing_tip', 'nonfinite_calibration',
                                   'joint_offsets', 'tool', 'model', 'output'])
def test_invalid_or_incomplete_inputs_refuse_before_output(tmp_path, fault):
    log, model, workspace, rows = inputs(tmp_path)
    out = tmp_path / 'out'
    tool = 'lutin-ballpoint-dot'
    if fault == 'truncated':
        log.write_bytes(log.read_bytes()[:-8])
    elif fault in ('nonfinite', 'unordered'):
        rows[1, 19 if fault == 'nonfinite' else 3] = float('nan') if fault == 'nonfinite' else 0.
        log.write_bytes(log.read_bytes()[:64] + rows.astype('<f8').tobytes())
    elif fault == 'missing_tip':
        workspace.write_text('right:\n  tool_id: lutin-ballpoint-dot\n')
    elif fault == 'nonfinite_calibration':
        workspace.write_text(fault_touchoff(workspace.read_text(), 'residual_mm', 'nan'))
    elif fault == 'joint_offsets':  # the controller's yaw joint has no model correction
        workspace.write_text(with_offsets(workspace.read_text(), [.01, *OFFSETS[1:]]))
    elif fault == 'tool':
        tool = 'another-tool'
    elif fault == 'model':
        model.write_text('<robot name="incomplete"><link name="root"/></robot>')
    else:
        out.mkdir()
        (out / 'sentinel').write_text('keep')
    with pytest.raises(ValueError):
        export(log, out, model, workspace, tool)
    if fault == 'output':
        assert (out / 'sentinel').read_text() == 'keep'
    else:
        assert not out.exists()


def test_cli_requires_explicit_inputs_and_registers_offline_verb(tmp_path):
    command = [str(REPO / 'scripts/tatbot'), '--ee-tool', 'lutin-ballpoint-dot',
               'teleop', 'poses']
    result = subprocess.run(command + ['--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert all(flag in result.stdout for flag in ('--out', '--urdf', '--workspace'))
    result = subprocess.run(command + ['recorded.wxtl'], capture_output=True, text=True)
    assert result.returncode != 0 and '--urdf' in result.stderr
    result = subprocess.run([str(REPO / 'scripts/tatbot'), 'schema'],
                            capture_output=True, text=True, check=True)
    schema = json.loads(result.stdout)
    entry = next(item for item in schema['verbs'] if item['name'] == 'teleop poses')
    assert entry['tier'] == 'offline' and entry['needs_tool']
    assert entry['effects'] == ['read_files', 'write_files']
    assert not entry['requirements']['auto_hop'] and not entry['launch_id']
    result = subprocess.run(command + ['recorded.wxtl', '--out', str(tmp_path / 'out'),
        '--urdf', 'model.urdf', '--workspace', 'workspace.yaml', '--dry-run'],
        capture_output=True, text=True)
    assert result.returncode == 0 and 'teleop_poses.py' in result.stdout
    assert '--workspace workspace.yaml' in result.stdout


def test_explicit_urdf_geometry_is_used_instead_of_default(tmp_path):
    log, model, workspace, _ = inputs(tmp_path)
    export(log, tmp_path / 'first', model, workspace, 'lutin-ballpoint-dot')
    text = model.read_text()
    assert 'xyz="0 -0.2675 0"' in text
    model.write_text(text.replace('xyz="0 -0.2675 0"', 'xyz="0 -0.1675 0"', 1))
    export(log, tmp_path / 'second', model, workspace, 'lutin-ballpoint-dot')
    a = np.load(tmp_path / 'first/poses.npz', allow_pickle=False)['root_from_tcp']
    b = np.load(tmp_path / 'second/poses.npz', allow_pickle=False)['root_from_tcp']
    np.testing.assert_allclose(b[:, :3, 3] - a[:, :3, 3], np.tile([0., .1, 0.], (len(a), 1)), atol=1e-12)
    np.testing.assert_allclose(b[:, :3, :3], a[:, :3, :3])


@pytest.mark.parametrize('fault', ['arm_fixed', 'zero_axis', 'scaled_axis', 'unknown_movable', 'detached_link6'])
def test_malformed_kinematic_ancestry_refuses(tmp_path, fault):
    log, model, workspace, _ = inputs(tmp_path)
    tree = ET.parse(model)
    root = tree.getroot()
    joints = {joint.get('name'): joint for joint in root.findall('joint')}
    if fault == 'arm_fixed':
        joints['right/joint_0'].set('type', 'fixed')
    elif fault == 'zero_axis':
        joints['right/left_carriage_joint'].find('axis').set('xyz', '0 0 0')
    elif fault == 'scaled_axis':
        joints['right/left_carriage_joint'].find('axis').set('xyz', '0 2 0')
    elif fault == 'unknown_movable':
        joints['right/tool_mount_joint'].set('type', 'revolute')
    else:
        for joint in root.findall('joint'):
            for tag in ('parent', 'child'):
                element = joint.find(tag)
                if element is not None and element.get('link') == 'right/link_6':
                    element.set('link', 'right/alternate_link6')
        extra = ET.SubElement(root, 'joint', {'name': 'detached6', 'type': 'fixed'})
        ET.SubElement(extra, 'parent', {'link': 'root'})
        ET.SubElement(extra, 'child', {'link': 'right/link_6'})
    tree.write(model)
    with pytest.raises(ValueError):
        export(log, tmp_path / 'out', model, workspace, 'lutin-ballpoint-dot')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('changed', ['source', 'tool'])
def test_source_or_tool_change_during_computation_refuses(tmp_path, monkeypatch, changed):
    import teleop_poses
    log, model, workspace, _ = inputs(tmp_path)
    read = teleop_poses._read
    target = (Path(teleop_poses.__file__) if changed == 'source'
              else REPO / 'config/tools/lutin-ballpoint-dot.yaml')
    seen = 0
    def changed_read(path, limit):
        nonlocal seen
        data = read(path, limit)
        if Path(path) == target:
            seen += 1
            if seen == 2:
                return data + b'changed'
        return data
    monkeypatch.setattr(teleop_poses, '_read', changed_read)
    with pytest.raises(ValueError, match='changed during export'):
        export(log, tmp_path / 'out', model, workspace, 'lutin-ballpoint-dot')
    assert not (tmp_path / 'out').exists()


def test_malformed_xml_cli_refuses_without_partial_output(tmp_path):
    log, model, workspace, _ = inputs(tmp_path)
    model.write_text('<robot>')
    out = tmp_path / 'out'
    result = subprocess.run([sys.executable, str(REPO / 'scripts/vision/teleop_poses.py'),
        str(log), '--out', str(out), '--urdf', str(model), '--workspace', str(workspace),
        '--ee-tool', 'lutin-ballpoint-dot'], capture_output=True, text=True)
    assert result.returncode == 3 and 'pose export refused:' in result.stderr
    assert not out.exists()


def test_output_in_repository_refuses(tmp_path):
    log, model, workspace, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match='outside the repository'):
        export(log, REPO / 'never-write-poses-here', model, workspace, 'lutin-ballpoint-dot')


@pytest.mark.parametrize('fault', ['right_scalar', 'touch_scalar', 'infinite_count', 'negative_count',
                                  'negative_condition', 'negative_residual', 'negative_holdout'])
def test_malformed_calibration_cli_refuses_before_output(tmp_path, fault):
    log, model, workspace, _ = inputs(tmp_path)
    text = workspace.read_text()
    changes = {'infinite_count': ('n_plate', 'inf'), 'negative_count': ('n_pad', '-9'),
               'negative_condition': ('cond', '-7.4'), 'negative_residual': ('residual_mm', '-1.828'),
               'negative_holdout': ('holdout_mm', '-2.969')}
    if fault == 'right_scalar':
        text = 'right: wrong\n'
    elif fault == 'touch_scalar':
        text = 'right:\n  tool_id: lutin-ballpoint-dot\n  touchoff: wrong\n'
    else:
        text = fault_touchoff(text, *changes[fault])
    workspace.write_text(text)
    out = tmp_path / 'out'
    result = subprocess.run([sys.executable, str(REPO / 'scripts/vision/teleop_poses.py'),
        str(log), '--out', str(out), '--urdf', str(model), '--workspace', str(workspace),
        '--ee-tool', 'lutin-ballpoint-dot'], capture_output=True, text=True)
    assert result.returncode == 3 and 'pose export refused:' in result.stderr
    assert 'Traceback' not in result.stderr and not out.exists()
