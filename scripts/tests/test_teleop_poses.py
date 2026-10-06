"""Offline explicit-model FK and recorded follower timestamp contract."""
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/vision'), str(REPO / 'scripts/lib')]
import arm_kinematics as kin  # noqa: E402
import ballpoint_fixture  # noqa: E402
import tool_spec  # noqa: E402
from teleop_poses import export  # noqa: E402


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


def fault_touchoff(text: str, field: str, value: str) -> str:
    """Set the right arm's touch-off `field` to `value`, whatever today's calibration measured.

    The workspace copy is the repository's live one while the ballpoint is
    fitted, so a literal number here lasts exactly until the next touch-off.
    """
    right, sep, left = text.partition('\nleft:')
    changed = re.sub(rf'^(\s+{field}: ).*$', rf'\g<1>{value}', right, count=1, flags=re.M)
    assert changed != right, f'{field} is not in the right arm touch-off'
    return changed + sep + left


def without_touchoff(text: str, arm: str) -> str:
    """The `arm` section as it reads before any touch-off: no measured tip."""
    head, sep, rest = text.partition(f'\n{arm}:\n')
    assert sep, arm
    body, sep2, tail = rest.partition('\n\n') if '\n\n' in rest else (rest, '', '')
    body = re.sub(r'^(  pen_tip_offset_[xyz]: ).*$', r'\1null', body, flags=re.M)
    return head + sep + body + sep2 + tail


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
    np.testing.assert_allclose(tcp[1, :3, 3]-tcp[0, :3, 3],
                               link6[0, :3, :3] @ kin.CARRIAGE_AXIS_IN_LINK6 * .002, atol=1e-12)
    chain = kin.urdf_chain(model)
    mount = chain.link_pose(kin.TOOL_MOUNT_NAME, kin.joint_map(rows[0, 19:25], 0))
    tip = tool_spec.tip_offset_m(tool_spec.parse_simple_yaml(workspace.read_text()))
    np.testing.assert_allclose(tcp[0, :3, 3], mount[:3, 3] + mount[:3, :3] @ tip)
    np.testing.assert_allclose(tcp[0, :3, :3], mount[:3, :3])
    assert not np.allclose(tcp[0, :3, :3], link6[0, :3, :3])
    assert not report['hardware_authority'] and 'not hardware sample time' in report['timestamp_semantics']
    assert report['inputs']['urdf'] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert json.loads((out / 'manifest.json').read_text()) == report


def test_leader_export_uses_its_recorded_joints_and_nominal_tool(tmp_path):
    """`--arm left` exports the leader's recorded joints through the left chain
    to its own tool mount; a standoff tool without a touch-off sits at its
    datasheet nominal and the manifest says so."""
    log, model, workspace, rows = inputs(tmp_path)
    leader = np.array([.2, -.3, .5, -.1, .2, .1, 0.])
    rows[:, 5:12] = leader
    rows[1:, 11] = .002
    rows[2, 5] += .3
    header = (log.read_bytes())[:64]
    log.write_bytes(header + rows.astype('<f8').tobytes())
    # The repository's left arm has carried a touch-off since 2026-09-16; the
    # premise here is the leader before one.
    workspace.write_text(without_touchoff(workspace.read_text(), 'left'))
    out = tmp_path / 'left'
    report = export(log, out, model, workspace, 'picosecond-laser-pen', arm='left')
    assert report['arm'] == 'left' and report['controller_role'] == 'leader'
    assert report['tip_source'] == 'datasheet nominal' and 'leader_pos' in report['joint_source']
    data = np.load(out / 'poses.npz', allow_pickle=False)
    np.testing.assert_array_equal(data['follower_positions'], rows[:, 5:12])
    np.testing.assert_array_equal(data['follower_read_offset_s'], rows[:, 2])
    tcp, link6 = data['root_from_tcp'], data['root_from_link6']
    chain = kin.urdf_chain(model)
    arm = kin.ArmModel('left', chain=chain, workspace=tool_spec.parse_simple_yaml(workspace.read_text()))
    # The URDF's needle link follows the measured touch-off the repository
    # holds; the nominal working point is the datasheet's protrusion along
    # the mount's +z, which is what a leader without a touch-off exports.
    nominal = np.array(tool_spec.load_tool('picosecond-laser-pen', REPO).nominal_tip_offset_m)
    for i in range(3):
        values = arm.joint_map(rows[i, 5:11], rows[i, 11])
        mount = chain.link_pose('left/tool_mount', values)
        np.testing.assert_allclose(tcp[i, :3, 3], mount[:3, :3] @ nominal + mount[:3, 3], atol=1e-12)
        np.testing.assert_allclose(link6[i], chain.link_pose('left/link_6', values), atol=1e-12)
    # the leader's tool rides left/carriage_left, which the carriage joint
    # moves along +y of link 6 (the follower's rides the other carriage, -y)
    np.testing.assert_allclose(tcp[1, :3, 3]-tcp[0, :3, 3], link6[0, :3, :3] @ [0, .002, 0], atol=1e-12)
    with pytest.raises(ValueError, match='stated tool differs'):
        export(log, tmp_path / 'wrong', model, workspace, 'lutin-ballpoint-dot', arm='left')
    with pytest.raises(ValueError, match='unknown arm'):
        export(log, tmp_path / 'middle', model, workspace, 'picosecond-laser-pen', arm='middle')


@pytest.mark.parametrize('fault', ['truncated', 'nonfinite', 'unordered', 'missing_tip', 'nonfinite_calibration', 'tool', 'model', 'output'])
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
    import xml.etree.ElementTree as ET
    log, model, workspace, _ = inputs(tmp_path)
    tree = ET.parse(model)
    root = tree.getroot()
    joints = {joint.get('name'): joint for joint in root.findall('joint')}
    if fault == 'arm_fixed':
        joints['right/joint_0'].set('type', 'fixed')
    elif fault == 'zero_axis':
        joints[kin.CARRIAGE_JOINT_NAME].find('axis').set('xyz', '0 0 0')
    elif fault == 'scaled_axis':
        joints[kin.CARRIAGE_JOINT_NAME].find('axis').set('xyz', '0 2 0')
    elif fault == 'unknown_movable':
        joints['right/tool_mount_joint'].set('type', 'revolute')
    else:
        for joint in root.findall('joint'):
            for tag in ('parent', 'child'):
                element = joint.find(tag)
                if element is not None and element.get('link') == kin.LINK6_NAME:
                    element.set('link', 'right/alternate_link6')
        extra = ET.SubElement(root, 'joint', {'name': 'detached6', 'type': 'fixed'})
        ET.SubElement(extra, 'parent', {'link': 'root'})
        ET.SubElement(extra, 'child', {'link': kin.LINK6_NAME})
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
