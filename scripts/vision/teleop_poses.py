#!/usr/bin/env python3
"""Export one arm's measured tool poses from explicit recorded inputs; no motion authority.

The follower (right arm) is the default and its export is unchanged. `--arm left`
exports the leader's recorded joints through the left chain to its own tool
mount; a leader tool without a touch-off is placed at its datasheet nominal
and the manifest says so (`tip_source`). The npz keys keep their historical
`follower_*` names for either arm; `arm` in the manifest is authoritative.
"""
import argparse
import hashlib
import json
import struct
import sys
import tempfile
from pathlib import Path
from xml.etree.ElementTree import ParseError

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
import arm_kinematics as kin  # noqa: E402
import tool_spec  # noqa: E402
from teleop_log import HEADER_LEN, MAGIC, TeleopLog  # noqa: E402


def _read(path, limit):
    with Path(path).expanduser().open('rb') as stream:
        data = stream.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError('missing or oversized pose input')
    return data


def _log_bytes(data):
    if len(data) < HEADER_LEN or data[:8] != MAGIC or struct.unpack_from('<Q', data, 8)[0] != 7:
        raise ValueError('pose export requires a seven-axis WXTLOG1 flight log')
    size = (5 + 6 * 7) * 8
    count, trailing = divmod(len(data) - HEADER_LEN, size)
    if trailing or not 1 <= count <= 250000:
        raise ValueError('empty, incomplete or oversized flight log')
    values = np.frombuffer(data, dtype='<f8', offset=HEADER_LEN).reshape(count, -1)
    timing = values[:, :5]
    if (not np.isfinite(values).all() or np.any(timing < 0)
            or np.any(np.diff(timing, axis=1) < 0) or np.any(np.diff(timing[:, 3]) <= 0)):
        raise ValueError('nonfinite or unordered recorded samples')
    period = struct.unpack_from('<d', data, 16)[0]
    origin = struct.unpack_from('<q', data, 56)[0]
    if not np.isfinite(period) or period <= 0 or not 0 < origin < 2**63:
        raise ValueError('invalid flight log clock')
    if float(timing[:, 3].max()) * 1e9 >= 2**63 - origin:
        raise ValueError('flight log timestamp overflow')


def _model(chain, model):
    """Walk the tool mount back to the root; only this arm's driven joints may move."""
    required = set(model.joint_names) | {model.carriage_joint}
    # The leader's mount rides the thumb-side carriage, a URDF mimic of its driven
    # carriage joint; it is movable but not one of the driver's seven axes.
    mimics = {name for name, joint in chain.joints.items()
              if joint.get('mimic') and joint['mimic'].get('joint') == model.carriage_joint}
    seen = set()
    links = set()
    link = model.tool_mount
    while link in chain.parent_of:
        name = chain.parent_of[link]
        if name in seen:
            raise ValueError('cyclic pose model')
        seen.add(name)
        links.add(link)
        joint = chain.joints[name]
        _joint(name, joint, required, mimics, model)
        link = joint['parent']
    if not set(model.joint_names) <= seen or not (seen & ({model.carriage_joint} | mimics)) or model.link6 not in links:
        raise ValueError(f'incomplete {model.arm} arm/tool-mount chain')
    return link


def _joint(name, joint, required, mimics, model):
    kind = joint['type']
    if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
        raise ValueError('unsupported pose-model joint')
    if (not np.isfinite(joint['origin']).all() or joint['axis'].shape != (3,)
            or not np.isfinite(joint['axis']).all()):
        raise ValueError('nonfinite pose model')
    if kind != 'fixed' and (name not in required | mimics or np.linalg.norm(joint['axis']) <= 0):
        raise ValueError('unknown movable joint or invalid axis')
    if name in model.joint_names and kind not in ('revolute', 'continuous'):
        raise ValueError('arm joints must be revolute')
    if name in {model.carriage_joint} | mimics and kind != 'prismatic':
        raise ValueError('carriage must be prismatic')
    if kind == 'prismatic' and not np.isclose(np.linalg.norm(joint['axis']), 1., rtol=0, atol=1e-9):
        raise ValueError('prismatic axis must have unit length for measured metres')


def _source_paths():
    return [Path(__file__), REPO / 'scripts/vision/teleop_log.py',
            REPO / 'scripts/vision/urdf_kinematics.py', REPO / 'scripts/lib/arm_kinematics.py',
            REPO / 'scripts/lib/tool_spec.py', REPO / 'scripts/lib/motion_constants.py',
            REPO / 'config/motion_constants.json']


def _poses(positions, chain, tip, model):
    tcp, link6 = [], []
    for row in positions:
        joints = model.joint_map(row[:6], row[6])
        pose = chain.link_pose(model.tool_mount, joints)
        pose[:3, 3] += pose[:3, :3] @ tip
        tcp.append(pose)
        link6.append(chain.link_pose(model.link6, joints))
    tcp, link6 = np.asarray(tcp), np.asarray(link6)
    for matrices in (tcp, link6):
        rotations = matrices[:, :3, :3]
        if (not np.isfinite(matrices).all() or not np.allclose(np.linalg.det(rotations), 1., atol=1e-8)
                or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-8)):
            raise ValueError('pose model produced invalid rigid transforms')
    return tcp, link6


def _workspace(data, arm='right'):
    workspace = tool_spec.parse_simple_yaml(data.decode())
    right = workspace.get(arm) if isinstance(workspace, dict) else None
    receipt = right.get('touchoff') if isinstance(right, dict) else None
    if not isinstance(right, dict) or not isinstance(receipt, dict):
        raise ValueError(f'workspace {arm} and touchoff must be mappings')
    for key in ('n_plate', 'n_pad'):
        value = receipt.get(key)
        if type(value) is not int or value < 0:
            raise ValueError('touch-off counts must be nonnegative integers')
    if receipt['n_plate'] == 0 and receipt['n_pad'] == 0 and tool_spec.tip_offset_m(workspace, arm) is None:
        return workspace  # no touch-off recorded for this arm: nothing to validate
    for key in ('cond', 'residual_mm', 'spread_deg', 'holdout_mm', 'tip_loo_max_mm'):
        value = receipt.get(key)
        if value is None and key in ('holdout_mm', 'tip_loo_max_mm'):
            continue
        number = float(value)
        if not np.isfinite(number) or number < 0 or (key == 'cond' and number == 0):
            raise ValueError('invalid finite nonnegative tip calibration evidence')
    return workspace


def _tip(spec, workspace, arm):
    """The working point in the mount frame and where it came from.

    A contact tool needs its touch-off (the tip is the working point). A
    standoff tool takes its touch-off plus the datasheet standoff when one
    exists, else the datasheet nominal: hover poses carry no contact authority,
    and the manifest names the source so nothing downstream mistakes one for
    the other."""
    measured = tool_spec.tip_offset_m(workspace, arm)
    if spec.contact:
        reason = tool_spec.contact_pose_qualification_error(spec, workspace, arm)
        if reason:
            raise ValueError(f'incomplete measured tip calibration: {reason}')
        return np.asarray(measured, dtype=float), 'touch-off'
    if measured is not None:
        return np.asarray(tool_spec.tcp_from_touchoff_m(spec, measured), dtype=float), 'touch-off plus datasheet standoff'
    return np.asarray(spec.nominal_tip_offset_m, dtype=float), 'datasheet nominal'


def export(log_path, out, urdf_path, workspace_path, ee_tool, arm='right'):
    if arm not in kin.ARM_IDS:
        raise ValueError(f'unknown arm {arm!r}')
    out = Path(out).expanduser()
    if out.exists():
        raise ValueError('pose output already exists')
    if out.resolve().is_relative_to(REPO.resolve()):
        raise ValueError('pose outputs must be outside the repository')
    inputs = {'log': _read(log_path, 128 * 1024**2), 'urdf': _read(urdf_path, 8 * 1024**2),
              'workspace': _read(workspace_path, 1024**2)}
    _log_bytes(inputs['log'])
    workspace = _workspace(inputs['workspace'], arm)
    if tool_spec.active_tool_id(workspace=workspace, arm=arm) != ee_tool:
        raise ValueError('stated tool differs from recorded workspace')
    tool_path = REPO / 'config/tools' / (ee_tool + '.yaml')
    source_paths = _source_paths()
    frozen = {path: _read(path, 8 * 1024**2) for path in [*source_paths, tool_path]}
    spec = tool_spec.load_tool(ee_tool, repo=REPO)
    spec.mount_frame(arm)  # ToolMountError for a tool with no mount
    tip, tip_source = _tip(spec, workspace, arm)
    if tip.shape != (3,) or not np.isfinite(tip).all():
        raise ValueError('nonfinite measured tip calibration')
    # Parse the exact bytes hashed below, even if a caller changes its input path.
    with tempfile.TemporaryDirectory(prefix='tatbot-poses-') as temporary:
        temporary = Path(temporary)
        for key in ('log', 'urdf'):
            (temporary / key).write_bytes(inputs[key])
        log = TeleopLog(temporary / 'log')
        chain = kin.urdf_chain(temporary / 'urdf')
        model = kin.ArmModel(arm, chain=chain, workspace=workspace)
        root = _model(chain, model)
        positions = log.follower_pos if arm == 'right' else log.leader_pos
        read_offset_s = log.t_follower_read if arm == 'right' else log.t_leader_read
        tcp, link6 = _poses(positions, chain, tip, model)
    stamps = log.wall_start_ns + np.rint(read_offset_s * 1e9).astype(np.int64)
    if np.any(np.diff(stamps) <= 0):
        raise ValueError('integer read timestamps are not strictly increasing')
    if any(_read(path, 8 * 1024**2) != data for path, data in frozen.items()):
        raise ValueError('pose implementation or tool changed during export')
    role = model.controller_role
    report = {'schema': 'tatbot.measured-follower-poses/1', 'hardware_authority': False,
              'arm': arm, 'controller_role': role, 'tip_source': tip_source,
              'samples': len(log), 'root_frame': root, 'tool_id': ee_tool,
              'orientation': f'tool_mount axes at the {tip_source} working point; root_from_link6 retained separately',
              'timestamp_semantics': f'host {role} read completion, not hardware sample time; no camera synchronization claim',
              'joint_source': f'recorded measured {role}_pos; six arm radians then carriage metres '
                              '(npz keys keep their historical follower_* names for either arm)',
              'wall_start_ns': log.wall_start_ns, 'period_s': log.period_s,
              'inputs': {key: hashlib.sha256(data).hexdigest() for key, data in inputs.items()},
              'tool_sha256': hashlib.sha256(frozen[tool_path]).hexdigest(),
              'source_sha256': {str(path.relative_to(REPO)): hashlib.sha256(frozen[path]).hexdigest()
                                for path in source_paths}}
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'poses.npz', original_index=np.arange(len(log)),
                        follower_positions=positions, follower_read_offset_s=read_offset_s,
                        follower_read_unix_ns=stamps, root_from_tcp=tcp, root_from_link6=link6)
    report['poses_sha256'] = hashlib.sha256((out / 'poses.npz').read_bytes()).hexdigest()
    (out / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log')
    parser.add_argument('--out', required=True)
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--ee-tool', required=True)
    parser.add_argument('--arm', choices=kin.ARM_IDS, default='right',
                        help='which arm the poses belong to: the follower (default) or the leader')
    args = parser.parse_args(argv)
    try:
        report = export(args.log, args.out, args.urdf, args.workspace, args.ee_tool, args.arm)
    except (ValueError, OSError, KeyError, TypeError, OverflowError, ParseError) as error:
        parser.exit(3, f'pose export refused: {error}\n')
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
