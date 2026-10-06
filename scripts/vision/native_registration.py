#!/usr/bin/env python3
"""Fit overhead registration from original native holds; never connect or adopt."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/lib'))
from tatbot_paths import bootstrap  # noqa: E402

REPO = bootstrap()
import arm_calibration as recipe  # noqa: E402
import tatbot_runlog  # noqa: E402
from tatbot_digest import sha256_file  # noqa: E402


def libraries(repo):
    """Load the shared pure geometry/fit code, without a ROS node or arm driver."""
    for name in ('tatbot_bridge', 'tatbot_description', 'tatbot_motion', 'tatbot_calib'):
        sys.path.insert(0, str(repo / 'ros' / name))
    from tatbot_calib import register
    return register


class NativeKinematics:
    """Adapter from native seven-axis measurements to the existing URDF reader."""

    def __init__(self, repo, arm, output):
        import numpy as np
        from tatbot_description import robot_description
        from urdf_kinematics import UrdfChain

        self.arm = arm
        path = output / 'model.urdf'
        path.write_text(robot_description(repo, arms=(arm,)))
        self.chain = UrdfChain(path)
        self.root_from_base = self.chain.link_pose(f'{arm}/base_link')
        self.base_from_root = np.linalg.inv(self.root_from_base)

    def frame(self, q, link):
        import numpy as np

        q = np.asarray(q, float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('native registration needs seven finite measured axes')
        values = dict(zip(self.chain.driver_joint_names(self.arm), q, strict=True))
        return self.base_from_root @ self.chain.link_pose(link, values)

    def fk(self, q):
        return self.frame(q, f'{self.arm}/tcp')


def model_inputs(repo, arm):
    from tool_spec import read_workspace

    tool = read_workspace(repo)[arm.arm_id]['tool_id']
    names = ('urdf/tatbot.urdf', 'config/workspace.yaml', 'config/arms.json', 'config/arm-labels.json',
             'config/fiducials.json', arm.wrist_layout, arm.controller_config, f'config/tools/{tool}.yaml')
    return {name: sha256_file(repo / name) for name in names}


def source_pin(repo, root, arm, hashes):
    """Refuse mixing a capture with a different model/configuration or arm."""
    meta = json.loads((root / 'meta.json').read_text())
    source = meta['git']
    if meta.get('arm') != arm.label or meta.get('status') != 'ok' or source.get('dirty') is not False:
        raise ValueError('registration requires a successful clean capture of the selected arm')
    revision = source['sha']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('capture lacks a full source revision')
    for name, expected in hashes.items():
        data = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=repo)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'capture source differs from the selected model: {name}')
    return revision


def original_capture(root, shared):
    from board_rgbd_evidence import unpack

    manifest = json.loads((root / 'overhead/capture.json').read_text())
    packet = root / 'overhead/owner-packet.bin'
    if (manifest.get('schema') != 'tatbot.overhead-owner-capture/1'
            or manifest.get('original_packet') != packet.name
            or manifest.get('original_packet_sha256') != sha256_file(packet)):
        raise ValueError('original overhead packet hash/schema mismatch')
    _, frames = unpack(packet.read_bytes(), max_bytes=64 * 2**20)
    color, depth = frames[shared.capture.COLOR], frames[shared.capture.DEPTH]
    if manifest['color_metadata'] != color[0] or manifest['depth_metadata'] != depth[0]:
        raise ValueError('retained metadata differs from original overhead packet')
    stamp = color[0]['timestamps']['normalized_unix_ns']
    if stamp != manifest['stamp_ns'] or stamp < manifest['requested_after_ns']:
        raise ValueError('overhead exposure predates its retained request')
    return packet, color, depth


def measured_hold(root, packet, shared):
    binding = recipe.bind_owner_packet(packet, root / 'arm/telemetry.bin')
    if binding != json.loads((root / 'overhead-pose-binding.json').read_text()):
        raise ValueError('retained exposure binding differs from original flight')
    if any(p['mode'] != 1 or p['estop'] != 0 for p in binding['pairs']):
        raise ValueError('overhead capture is not bound to a healthy Position hold')
    if max(binding['capture_joint_span_rad']) > shared.STILL_RAD:
        raise ValueError('measured arm moved during the overhead capture')
    pair, = [p for p in binding['pairs'] if p['camera_role'] == shared.capture.COLOR and p['stream'] == 'color']
    return [*pair['joints_rad'], pair['carriage_m']], binding


def optics_match(bundle, color, depth, shared):
    draft = shared.draft_bundle(color[0], depth[0]['profile'])
    if not shared.reusable(bundle, draft):
        raise ValueError('original overhead optics do not match the supplied bundle')
    if any(frame[0].get('calibration_id') != bundle['bundle_id'] for frame in (color, depth)):
        raise ValueError('original overhead calibration ID differs from the supplied bundle')


def capture_row(index, root, context):
    arm, hashes, bundle, shared, detector, target, kin, repo = context
    revision = source_pin(repo, root, arm, hashes)
    packet, color, depth = original_capture(root, shared)
    optics_match(bundle, color, depth, shared)
    q, binding = measured_hold(root, packet, shared)
    image = shared.capture._decode(color)
    stamp = color[0]['timestamps']['normalized_unix_ns']
    detections = detector.detect(shared.capture.COLOR, image, stamp)
    logging.info('hold %02d: tags %s; original %s; host skew %.6f ms', index,
                 sorted(d.tag_id for d in detections), binding['source_capture_sha256'], binding['maximum_skew_ms'])
    row = {'hold': index, 'reached': True, 'q': q, 'stamp_ns': stamp,
           'still_rad': max(binding['capture_joint_span_rad']),
           'tags': {str(d.tag_id): d.corners_px.tolist() for d in detections},
           'capture': str(root), 'capture_source_revision': revision, 'binding': binding,
           'source_sha256': {name: sha256_file(root / name) for name in (
               'meta.json', 'overhead/capture.json', 'overhead/owner-packet.bin',
               'arm/telemetry.bin', 'overhead-pose-binding.json')}}
    return row, shared.hold_sightings(row, kin, arm.arm_id, target)


def observations(roots, context):
    rows, sightings, seen = [], [], set()
    for index, root in enumerate(roots):
        row, detected = capture_row(index, root, context)
        digest = row['binding']['source_capture_sha256']
        if digest in seen:
            raise ValueError('duplicate original overhead capture')
        seen.add(digest)
        rows.append(row)
        sightings.extend(detected)
    return rows, sightings


def prepare(repo, label, roots, bundle_path, output, setup_id, *, seat_report=False):
    import cv2
    import numpy as np
    from fiducials import load_inventory
    from fiducials.detector import DetectorConfig, FiducialDetector

    shared = libraries(repo)
    if not setup_id.strip():
        raise ValueError('setup ID must name the selected physical setup window')
    arm = recipe.selected_arm(repo, label)
    inventory = load_inventory(repo / 'config/fiducials.json')
    target = inventory.target(arm.wrist_target)
    detector = FiducialDetector.from_inventory(inventory, DetectorConfig(min_side_px=8., refinement='edges'),
                                               target=arm.wrist_target)
    output.mkdir(parents=True, exist_ok=False)
    bundle_bytes = bundle_path.read_bytes()
    bundle = json.loads(bundle_bytes)
    (output / 'calibration.json').write_bytes(bundle_bytes)
    kin = NativeKinematics(repo, arm.arm_id, output)
    hashes = model_inputs(repo, arm)
    context = arm, hashes, bundle, shared, detector, target, kin, repo
    rows, sightings = observations(roots, context)
    recipe.write_json(output / 'observations.json', rows)
    report = _fit(shared, arm, bundle, sightings, target)
    report.update({'schema': 'tatbot.native-registration-fit/1', 'setup_id': setup_id,
                   'captures': len(rows), 'model_inputs_sha256': hashes,
                   'model_urdf_sha256': sha256_file(output / 'model.urdf'),
                   'camera_bundle_sha256': sha256_file(output / 'calibration.json'),
                   'fit_source_sha256': implementation_inputs(repo),
                   'runtime': {'numpy': np.__version__, 'opencv': cv2.__version__, 'python': sys.version},
                   'calibration_adopted': False, 'motion_authority': False,
                   'physical_accuracy_bound_m': None, 'clock_error_bound_ms': None,
                   'scope': 'explicitly grouped native holds; normalized host-time correspondence; unadopted fit'})
    if seat_report and 'camera_from_base' in report:
        report['seat_diagnostics'] = _seat_report(shared, kin, arm.arm_id, target, bundle, rows, report, output)
    if 'camera_from_base' in report:
        candidate = shared.registration_record(arm.arm_id, bundle, output / 'calibration.json',
            np.asarray(report['camera_from_base']), kin.root_from_base, report, output.name)
        candidate['rig'] = {'capture': output.name, 'method': 'native_original_owner_holds',
                            'setup_id': setup_id, 'adopted': False}
        candidate.update({'motion_authority': False, 'physical_accuracy_bound_m': None})
        recipe.write_json(output / f'arm-registration-{arm.arm_id}-candidate.json', candidate)
    if model_inputs(repo, arm) != hashes:
        raise ValueError('model inputs changed during registration fitting')
    recipe.write_json(output / 'report.json', report)
    return report


def _seat_report(shared, kin, arm, target, bundle, rows, report, output):
    """Shared camera/seat diagnostics with whole distinct views withheld; never install geometry."""
    import numpy as np
    from tatbot_calib import chain

    sightings = chain.sightings_of(rows)
    views = shared.pose_views([kin.fk(s['q']) for s in sightings])
    by_view = {str(v): sorted({s['hold'] for s, view in zip(sightings, views, strict=True) if view == v})
               for v in sorted(set(views))}
    scope = {'original_holds_by_distinct_view': by_view, 'original_sightings': len(sightings),
             'calibration_adopted': False, 'motion_authority': False, 'physical_accuracy_bound_m': None,
             'scope': 'all original sightings; shared distinct-view groups; camera and seat refitted per holdout; '
                      'joint offsets not fitted; rigid registration gates unchanged'}
    if len(by_view) < 3:
        return {**scope, 'refused': ['tag-seat diagnostics need at least three distinct tagged views']}
    optics = bundle['cameras'][shared.capture.COLOR]
    k = np.array([[optics['intrinsics']['fx'], 0, optics['intrinsics']['cx']],
                  [0, optics['intrinsics']['fy'], optics['intrinsics']['cy']], [0, 0, 1]], float)
    grouped = [dict(s, original_hold=s['hold'], hold=v) for s, v in zip(sightings, views, strict=True)]
    run = chain.Run(output.name, k, np.asarray(optics['distortion']['coefficients']), grouped,
                    np.asarray(report['camera_from_base']))
    diagnostic = chain.report_for(chain.Chain(kin, arm, target), [run], output,
                                  models={'rigid': ((), False), 'seat': ((), True)})
    for model in diagnostic['models'].values():
        held = model['held_out']
        held['by'], held['views'] = 'distinct_view', held.pop('holds')
    diagnostic.update(scope)
    diagnostic['layout_sha256'] = sha256_file(output / diagnostic['layout'])
    recipe.write_json(output / 'seat-diagnostics.json', diagnostic)
    return diagnostic


def implementation_inputs(repo):
    names = ('scripts/vision/native_registration.py', 'scripts/lib/arm_calibration.py',
             'scripts/lib/board_rgbd_evidence.py', 'scripts/vision/urdf_kinematics.py',
             'scripts/vision/fiducials/detector.py', 'scripts/vision/fiducials/geometry.py',
             'ros/tatbot_calib/tatbot_calib/register.py', 'ros/tatbot_bridge/tatbot_bridge/capture.py',
             'ros/tatbot_description/tatbot_description/__init__.py', 'scripts/lib/kinematic_calibration.py',
             'ros/tatbot_calib/tatbot_calib/chain.py')
    return {name: sha256_file(repo / name) for name in names}


def _fit(shared, arm, bundle, sightings, target):
    if len(sightings) < 3:
        return {'arm': arm.arm_id, 'refused': ['fewer than three original tag sightings'],
                'sightings': len(sightings)}
    try:
        return shared.fit(arm.arm_id, bundle, True, sightings, target)
    except (ValueError, RuntimeError) as error:
        return {'arm': arm.arm_id, 'refused': [f'shared registration fit: {error}'], 'sightings': len(sightings)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=('blue', 'pink'))
    parser.add_argument('--capture', required=True, nargs='+', type=Path)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--setup-id', required=True, help='label for this explicitly selected physical setup window')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seat-report', action='store_true',
                        help='also compare shared rigid/seat models by distinct view; SciPy required; never adopt')
    args = parser.parse_args(argv)
    run = tatbot_runlog.init('calib-register-fit', prune_first=False)
    status = 1
    try:
        report = prepare(REPO, args.arm, [p.expanduser().resolve() for p in args.capture],
                         args.bundle.expanduser().resolve(), args.out.expanduser().resolve(), args.setup_id,
                         seat_report=args.seat_report)
        run.artifact(args.out, name='native-registration-fit')
        logging.info(json.dumps({'output': str(args.out), 'refused': report['refused'], 'calibration_adopted': False}))
        status = 3 if report['refused'] else 0
    except BaseException:
        logging.exception('native registration fitting failed')
        raise
    finally:
        run.finalize(status, status='ok' if status == 0 else 'fail')
        print(tatbot_runlog.banner('end', run.run_id, 'exit=' + str(status)))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
