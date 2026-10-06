"""Original native hold evidence feeds the existing overhead geometry/gates."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import arm_calibration as recipe
import native_registration as native
import numpy as np
import pytest
from cli_runner import tatbot

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def shared():
    pytest.importorskip('cv2')
    return native.libraries(REPO)


@pytest.fixture
def capture_factory(tmp_path, shared):
    def make(index=0, q=None, moving=False):
        root = tmp_path / f'hold-{index}'
        (root / 'overhead').mkdir(parents=True)
        (root / 'arm').mkdir()
        stamp = 115_000_000 + index * 1_000_000_000
        intr = {'schema': 'tatbot.camera-intrinsics/1', 'width': 64, 'height': 48,
                'fx': 60., 'fy': 60., 'ppx': 32., 'ppy': 24.,
                'distortion_model': 'None', 'distortion_coefficients': [0.] * 5}
        frames, payloads = [], []
        for kind, name, data, encoding in (
            ('color', shared.capture.COLOR, bytes(64*48*3), 'rgb8'),
            ('depth', shared.capture.DEPTH, bytes(64*48*2), 'z16')):
            meta = {'sensor_name': name, 'sensor_kind': 'real_sense', 'sequence': index+1,
                    'profile': {'stream': kind, 'width': 64, 'height': 48, 'fps_num': 30,
                                'fps_den': 1, 'format': encoding},
                    'timestamps': {'normalized_unix_ns': stamp}, 'calibration_id': 'f'*64,
                    'attributes': {'intrinsics': json.dumps(intr), 'device_serial': 'synthetic',
                                   'depth_units_m': '.001', 'bus_source_format': encoding}}
            frames.append({'metadata': meta, 'payload': {kind: {'bytes': len(data), 'format': encoding}}})
            payloads.append(data)
        header = json.dumps({'magic': 'tatbot-vision-frame-set', 'version': 1,
                             'envelope': {'producer': {'node': 'synthetic'}}, 'frames': frames}).encode()
        packet = root / 'overhead/owner-packet.bin'
        packet.write_bytes(len(header).to_bytes(4, 'big') + header + b''.join(payloads))
        q = list(q if q is not None else [.1, .2, .3, .4, .5, .6, .002])
        flight = recipe.TELEMETRY_MAGIC + (recipe.TELEMETRY_RECORD.size).to_bytes(8, 'little')
        for i, offset in enumerate((-15_000_000, -5_000_000, 5_000_000, 15_000_000)):
            axes = [v + (.01*i if moving else 0.) for v in q]
            flight += recipe.TELEMETRY_RECORD.pack(i+1, stamp+offset, stamp+offset,
                                                   *axes, *([0.]*14), 1, 0, 1)
        (root / 'arm/telemetry.bin').write_bytes(flight)
        recipe.write_json(root / 'overhead/capture.json', {
            'schema': 'tatbot.overhead-owner-capture/1', 'original_packet': packet.name,
            'original_packet_sha256': native.sha256_file(packet), 'requested_after_ns': stamp-1,
            'stamp_ns': stamp, 'color_metadata': frames[0]['metadata'], 'depth_metadata': frames[1]['metadata']})
        recipe.write_json(root / 'overhead-pose-binding.json', recipe.bind_owner_packet(packet, root / 'arm/telemetry.bin'))
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        recipe.write_json(root / 'meta.json', {'arm': 'blue', 'status': 'ok', 'git': {'sha': revision, 'dirty': False}})
        bundle = shared.draft_bundle(frames[0]['metadata'], frames[1]['metadata']['profile'])
        bundle['bundle_id'] = 'f'*64
        return root, bundle
    return make


def test_cli_fit_is_local_offline_and_has_no_arm_or_ros_launcher():
    result = tatbot('--dry-run', '--json', 'calib', 'register-fit', '--python', '/existing/python',
                    '--arm', 'blue', '--capture', '/a', '/b', '--bundle', '/bundle',
                    '--setup-id', 'rearranged', '--out', '/fit', '--seat-report')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['tier'] == 'offline'
    assert plan['argv'][0] == '/existing/python'
    assert 'native_registration.py' in plan['argv'][1]
    assert '--seat-report' in plan['argv']
    assert not set(plan['argv']) & {'ros2', 'ssh', '--operator-go', '--ee-tool'}


def test_native_adapter_uses_arm_base_once_and_moves_the_tag_with_carriage(tmp_path, shared):
    kin = native.NativeKinematics(REPO, 'left', tmp_path)
    q = np.zeros(7)
    assert np.allclose(kin.frame(q, 'left/base_link'), np.eye(4))
    assert kin.root_from_base[1, 3] == pytest.approx(.2675)
    q[6] = .012
    delta = kin.frame(q, 'left/wrist_tag1')[:3, 3] - kin.frame(np.zeros(7), 'left/wrist_tag1')[:3, 3]
    assert np.linalg.norm(delta) == pytest.approx(.012)
    q[0], q[5] = .3, 1.2
    values = dict(zip(kin.chain.driver_joint_names('left'), q, strict=True))
    expected = np.linalg.inv(kin.chain.link_pose('left/base_link')) @ kin.chain.link_pose('left/wrist_tag1', values)
    assert np.allclose(kin.frame(q, 'left/wrist_tag1'), expected)
    assert not np.allclose(kin.frame(q, 'left/wrist_tag1'), kin.chain.link_pose('left/wrist_tag1', values))
    with pytest.raises(ValueError, match='seven finite'):
        kin.frame(q[:6], 'left/wrist_tag1')


@pytest.mark.parametrize('change,why', [
    ('packet', 'hash/schema'), ('metadata', 'metadata differs'), ('request', 'predates'),
    ('binding', 'binding differs'), ('flight', 'binding differs'), ('motion', 'moved')])
def test_tampered_or_moving_original_evidence_refuses(capture_factory, shared, change, why):
    root, _ = capture_factory(moving=change == 'motion')
    packet = root / 'overhead/owner-packet.bin'
    manifest_path = root / 'overhead/capture.json'
    manifest = json.loads(manifest_path.read_text())
    if change == 'packet':
        packet.write_bytes(packet.read_bytes() + b'x')
    if change == 'metadata':
        manifest['color_metadata']['sequence'] += 1
    if change == 'request':
        manifest['requested_after_ns'] = manifest['stamp_ns'] + 1
    if change == 'binding':
        binding_path = root / 'overhead-pose-binding.json'
        binding = json.loads(binding_path.read_text())
        binding['pairs'][0]['joints_rad'][0] += .1
        recipe.write_json(binding_path, binding)
    if change == 'flight':
        telemetry = root / 'arm/telemetry.bin'
        telemetry.write_bytes(telemetry.read_bytes()[:-recipe.TELEMETRY_RECORD.size])
    recipe.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match=why):
        native.original_capture(root, shared)
        native.measured_hold(root, packet, shared)


def test_wrong_optics_or_camera_identity_refuses(capture_factory, shared):
    root, bundle = capture_factory()
    _, color, depth = native.original_capture(root, shared)
    native.optics_match(bundle, color, depth, shared)
    bundle['cameras'][shared.capture.COLOR]['metadata']['device_serial'] = 'another-camera'
    with pytest.raises(ValueError, match='optics'):
        native.optics_match(bundle, color, depth, shared)
    bundle['cameras'][shared.capture.COLOR]['metadata']['device_serial'] = 'synthetic'
    bundle['bundle_id'] = 'e'*64
    with pytest.raises(ValueError, match='calibration ID'):
        native.optics_match(bundle, color, depth, shared)


@pytest.mark.parametrize('field,value,why', [('arm', 'pink', 'selected arm'), ('status', 'fail', 'successful'),
                                         ('dirty', True, 'clean'), ('model', None, 'model')])
def test_capture_source_identity_and_model_pin_are_enforced(capture_factory, field, value, why):
    root, _ = capture_factory()
    meta = json.loads((root / 'meta.json').read_text())
    if field == 'dirty':
        meta['git']['dirty'] = value
    elif field != 'model':
        meta[field] = value
    recipe.write_json(root / 'meta.json', meta)
    arm = recipe.selected_arm(REPO, 'blue')
    hashes = native.model_inputs(REPO, arm)
    if field == 'model':
        hashes['config/workspace.yaml'] = '0'*64
    with pytest.raises(ValueError, match=why):
        native.source_pin(REPO, root, arm, hashes)


def test_shared_fit_recovers_known_transform_and_repeated_views_do_not_qualify(tmp_path, shared):
    kin = native.NativeKinematics(REPO, 'left', tmp_path)
    from fiducials import load_inventory

    target = load_inventory(REPO / 'config/fiducials.json').target('wrist_left')
    # A known camera in a synthetic scene; each view sees all configured wrist tags.
    camera = np.eye(4)
    camera[:3, 3] = [.1, -.2, 1.5]
    k = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
    dist = np.zeros(5)
    bundle = {'bundle_id': 'synthetic', 'cameras': {shared.capture.COLOR: {
        'intrinsics': {'fx': 500., 'fy': 500., 'cx': 320., 'cy': 240.},
        'distortion': {'coefficients': dist.tolist()}}}}
    sightings = []
    for i in range(12):
        q = np.array([.1*i, .2*np.sin(i), .2*np.cos(i), .1*np.sin(i*2), .1, .4*i, .002])
        for tag in target.ids:
            points = shared.corner_points(kin, 'left', q, tag, target.edge_m)
            sightings.append({'hold': i, 'tag': tag, 'points': points,
                              'pixels': shared._project(camera, points, k, dist), 'tcp': kin.fk(q)})
    report = shared.fit('left', bundle, True, sightings, target)
    assert not report['refused'], report['refused']
    assert np.allclose(report['camera_from_base'], camera, atol=1e-6)
    assert report['hold_out']['views'] == 12
    repeated = [dict(sightings[i % 3], hold=i) for i in range(36)]
    thin = shared.fit('left', bundle, True, repeated, target)
    assert thin['fit']['holds'] == 36 and thin['fit']['poses'] == 1
    assert any('distinct poses' in reason for reason in thin['refused'])
    assert any('one view' in reason for reason in thin['refused'])


def test_duplicate_originals_refuse_without_creating_false_view_coverage(capture_factory, shared, monkeypatch):
    root, _ = capture_factory()
    binding = recipe.bind_owner_packet(root / 'overhead/owner-packet.bin', root / 'arm/telemetry.bin')
    monkeypatch.setattr(native, 'capture_row', lambda *args: ({'binding': binding}, []))
    with pytest.raises(ValueError, match='duplicate original'):
        native.observations([root, root], ())


def test_pipeline_retains_originals_and_candidate_without_adopting(capture_factory, shared, tmp_path, monkeypatch):
    pytest.importorskip('scipy')
    from fiducials import load_inventory
    from fiducials.detector import Detection, FiducialDetector

    model_dir = tmp_path / 'reference-model'
    model_dir.mkdir()
    kin = native.NativeKinematics(REPO, 'left', model_dir)
    target = load_inventory(REPO / 'config/fiducials.json').target('wrist_left')
    camera = np.eye(4)
    camera[:3, 3] = [.1, -.2, 1.5]
    q_by_stamp, roots = {}, []
    for i in range(12):
        q = np.array([.1*i, .2*np.sin(i), .2*np.cos(i), .1*np.sin(i*2), .1, .4*i, .002])
        root, bundle = capture_factory(i, q)
        roots.append(root)
        q_by_stamp[115_000_000+i*1_000_000_000] = q
    k = np.array([[60., 0, 32], [0, 60., 24], [0, 0, 1]])
    def detections(sensor, image, stamp):
        return [Detection(sensor, tag, shared._project(camera,
                    shared.corner_points(kin, 'left', q_by_stamp[stamp], tag, target.edge_m), k, np.zeros(5)),
                          stamp, 10., target.family) for tag in target.ids]
    monkeypatch.setattr(FiducialDetector, 'from_inventory', lambda *args, **kw: SimpleNamespace(detect=detections))
    def forbidden_adopt(*args):
        pytest.fail('offline native fit must never adopt')
    monkeypatch.setattr(shared, 'adopt', forbidden_adopt)
    bundle_path = tmp_path / 'bundle.json'
    recipe.write_json(bundle_path, bundle)
    originals = {root: native.sha256_file(root / 'overhead/owner-packet.bin') for root in roots}
    output = tmp_path / 'fit'
    report = native.prepare(REPO, 'blue', roots, bundle_path, output, 'synthetic-window', seat_report=True)
    assert not report['refused'], report['refused']
    assert report['captures'] == 12 and report['fit']['poses'] == 12
    assert not report['calibration_adopted'] and not report['motion_authority']
    assert report['physical_accuracy_bound_m'] is None and report['clock_error_bound_ms'] is None
    candidate = json.loads((output / 'arm-registration-left-candidate.json').read_text())
    assert np.allclose(candidate['world_from_arm_base'], camera, atol=1e-6)
    assert np.allclose(np.asarray(candidate['world_from_root']) @ kin.root_from_base, camera, atol=1e-6)
    assert not candidate['motion_authority'] and not candidate['rig']['adopted']
    assert (output / 'calibration.json').read_bytes() == bundle_path.read_bytes()
    assert originals == {root: native.sha256_file(root / 'overhead/owner-packet.bin') for root in roots}
    assert len(json.loads((output / 'observations.json').read_text())) == 12
    diagnostics = json.loads((output / 'seat-diagnostics.json').read_text())
    assert diagnostics == report['seat_diagnostics']
    assert not diagnostics['motion_authority'] and not diagnostics['calibration_adopted']
    assert diagnostics['models']['seat']['held_out']['by'] == 'distinct_view'
    assert diagnostics['models']['seat']['held_out']['views'] == 12
    assert diagnostics['layout_sha256'] == native.sha256_file(output / 'wrist-layout.json')
    with pytest.raises(FileExistsError):
        native.prepare(REPO, 'blue', roots, bundle_path, output, 'synthetic-window')


def test_seat_diagnostic_recovers_known_mount_without_counting_repeat_captures_as_new_views(tmp_path, shared):
    pytest.importorskip('scipy')
    from export_wrist_tags import record_from_solve
    from fiducials import load_inventory
    from tatbot_calib import chain

    kin = native.NativeKinematics(REPO, 'left', tmp_path)
    inventory = load_inventory(REPO / 'config/fiducials.json')
    target = inventory.target('wrist_left')
    c = chain.Chain(kin, 'left', target)
    camera = np.eye(4)
    camera[:3, 3] = [.1, -.2, 1.5]
    k = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    seat = chain._pose([.01, -.015, .005, .003, .006, -.002])
    bundle = {'cameras': {shared.capture.COLOR: {'intrinsics': {
        'fx': 500., 'fy': 500., 'cx': 320., 'cy': 240.}, 'distortion': {'coefficients': [0.] * 5}}}}
    rows = []
    for hold in range(15):
        i = hold if hold < 12 else 0
        q = np.array([.1*i, .2*np.sin(i), .2*np.cos(i), .1*np.sin(i*2), .1, .4*i, .002])
        tags = {str(tag): shared._project(camera, c.corners({'q': q, 'tag': tag}, np.zeros(7), seat),
                                         k, np.zeros(5)).tolist() for tag in target.ids}
        rows.append({'hold': hold, 'q': q.tolist(), 'still_rad': 0., 'tags': tags})
    diagnostic = native._seat_report(shared, kin, 'left', target, bundle, rows,
                                     {'camera_from_base': camera.tolist()}, tmp_path)
    assert diagnostic['original_sightings'] == 45
    assert diagnostic['original_holds_by_distinct_view']['0'] == [0, 12, 13, 14]
    assert len(diagnostic['original_holds_by_distinct_view']) == 12
    fit = diagnostic['models']['seat']['fit']
    assert np.allclose(fit['seat']['parent_from_measured'], seat, atol=1e-5)
    assert diagnostic['models']['seat']['held_out']['median_px'] < .001
    assert diagnostic['models']['rigid']['held_out']['median_px'] > .1
    layout = json.loads((tmp_path / 'wrist-layout.json').read_text())
    assert layout['observations'] == 45
    assert layout['pose_observations_by_tag'] == {str(tag): 12 for tag in target.ids}
    record = record_from_solve(layout, tmp_path / 'wrist-layout.json', inventory, target='wrist_left')
    for tag in target.ids:
        assert np.allclose(record['tags'][str(tag)]['ee_from_tag'], seat @ c.parent_from_tag[tag], atol=1e-5)
    assert not diagnostic['calibration_adopted'] and not diagnostic['motion_authority']

    thin = tmp_path / 'repeated-only'
    thin.mkdir()
    repeated = [dict(rows[0], hold=i) for i in range(15)]
    refused = native._seat_report(shared, kin, 'left', target, bundle, repeated,
                                  {'camera_from_base': camera.tolist()}, thin)
    assert refused['refused'] and len(refused['original_holds_by_distinct_view']) == 1
    assert not (thin / 'wrist-layout.json').exists()
