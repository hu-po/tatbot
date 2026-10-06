"""Independent size/depth witnesses preserve native geometry and ambiguity."""
import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from board_mount_witness import WristInputs, consistency, fit_mount
from board_witness import measurement, prepare, tag_evidence
from fiducials.geometry import tag_model_corners
from live_inputs import load_frames, rgbd_pair
from test_rgbd_geometry import alignment

REPO = Path(__file__).resolve().parents[2]


def metric_frame(native_transform=False):
    intr = {'schema': 'tatbot.camera-intrinsics/1', 'width': 640, 'height': 480,
            'fx': 400., 'fy': 400., 'ppx': 320., 'ppy': 240.,
            'distortion_model': 'None', 'distortion_coefficients': [0.]*5}
    rotation = cv2.Rodrigues(np.array([np.pi-.2, .08, .03]))[0]
    translation = np.array([.01, -.01, .25])
    normal = rotation[:, 2]
    y, x = np.indices((480, 640))
    rays = np.stack(((x-320)/400, (y-240)/400, np.ones_like(x)), axis=-1)
    truth = rays*((normal@translation)/(rays@normal))[..., None]
    rot = cv2.Rodrigues(np.array([.07, -.09, .02]))[0] if native_transform else np.eye(3)
    offset = np.array([-.012, .001, .004]) if native_transform else np.zeros(3)
    model = alignment(intr, rot, offset)
    native = np.rint(((truth-offset)@rot)[..., 2]/.0001).astype(np.uint16)
    attrs = {'capture_epoch': 'epoch', 'device_serial': 'device', 'intrinsics': json.dumps(intr),
             'alignment_calibration': json.dumps(model)}
    cm = {'sensor_name': 'wrist_color', 'sequence': 1, 'attributes': attrs,
          'profile': {'width': 640, 'height': 480, 'format': 'rgb8', 'fps_num': 30, 'fps_den': 1},
          'timestamps': {'source_domain': 'test', 'source_ns': 10000000000, 'normalized_unix_ns': 10000000000}}
    dm = copy.deepcopy(cm)
    dm['sensor_name'], dm['profile']['format'] = 'wrist_depth', 'z16'
    dm['attributes'].update(aligned_to='wrist_color', depth_units_m='0.0001')
    points = tag_model_corners(.04)@rotation.T+translation
    corners = points[:, :2]/points[:, 2, None]*400+[320, 240]
    detection = SimpleNamespace(corners_px=corners, family='apriltag_16h5', tag_id=7)
    return {'image': np.zeros((480, 640, 3), np.uint8), 'depth_m': native*.0001,
            'camera_model': intr, 'color_metadata': cm, 'depth_metadata': dm}, detection


@pytest.mark.parametrize('native_transform', [False, True])
def test_measured_square_agrees_with_independently_constructed_color_plane(native_transform):
    frame, tag = metric_frame(native_transform)
    report = tag_evidence(frame, tag, .04)
    assert len(report['pose_branches']) == 2
    best = min(report['pose_branches'], key=lambda row: row['corner_normalized_ray_rmse'])
    assert best['corner_normalized_ray_rmse'] < 1e-12
    assert best['supported_samples'] == best['total_samples'] == 81
    assert best['max_absolute_depth_minus_rgb_plane_m'] < .00007
    assert report['branch_selection'].startswith('unselected')


def test_independent_metric_scale_changes_the_depth_witness():
    frame, tag = metric_frame()
    result = tag_evidence(frame, tag, .05)
    best = min(result['pose_branches'], key=lambda row: row['corner_normalized_ray_rmse'])
    assert -.065 < best['median_depth_minus_rgb_plane_m'] < -.055


@pytest.mark.parametrize('value', [0, 65535*.0001, np.nan])
def test_missing_and_saturated_depth_stay_unavailable(value):
    frame, tag = metric_frame()
    frame['depth_m'][:] = value
    result = tag_evidence(frame, tag, .04)
    assert len(result['pose_branches']) == 2
    for branch in result['pose_branches']:
        assert branch['supported_samples'] == 0
        assert branch['median_depth_minus_rgb_plane_m'] is None
        assert branch['max_absolute_depth_minus_rgb_plane_m'] is None


def retained_capture(root):
    frame, _ = metric_frame(True)
    frames = {}
    for metadata, payload in [(frame['color_metadata'], frame['image'].tobytes()),
                              (frame['depth_metadata'], np.rint(frame['depth_m']*10000).astype('<u2').tobytes())]:
        name = metadata['sensor_name']
        filename = name+'.pixels'
        (root/filename).write_bytes(payload)
        frames[name] = {'metadata': metadata, 'payload_file': filename,
                        'payload_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    path = root/'capture.json'
    path.write_text(json.dumps({'schema': 'tatbot.session-surface/1', 'kind': 'live-capture', 'frames': frames}))
    return path


def test_original_pair_reader_refuses_corrupt_payload_and_epoch(tmp_path):
    path = retained_capture(tmp_path)
    _, frames = load_frames(path)
    frame = rgbd_pair(frames, 'wrist_color')
    assert frame['depth_m'].shape == (480, 640)
    changed = copy.deepcopy(frames)
    changed['wrist_depth'][0]['metadata']['attributes']['capture_epoch'] = 'other'
    with pytest.raises(ValueError, match='pair_unverified'):
        rgbd_pair(changed, 'wrist_color')
    (tmp_path/'wrist_color.pixels').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        load_frames(path)


def test_original_pair_reader_refuses_an_ambiguous_aligned_depth(tmp_path):
    _, frames = load_frames(retained_capture(tmp_path))
    frames['other_depth'] = copy.deepcopy(frames['wrist_depth'])
    with pytest.raises(ValueError, match='exactly one depth'):
        rgbd_pair(frames, 'wrist_color')


def test_no_tags_is_retained_without_a_calibration_claim(tmp_path):
    path = retained_capture(tmp_path)
    measured = tmp_path/'measurement.json'
    measured.write_text(json.dumps({'target': 'board', 'edge_m': .04, 'source': 'independent physical measurement'}))
    result = prepare([path], 'wrist_color', REPO/'config/fiducials.json', measured, tmp_path/'out')
    assert result['captures'][0]['tags'] == []
    assert result['measurement']['uncertainty_m'] is None
    assert result['physical_error_bound_m'] is None and result['mount_error_bound_m'] is None
    assert result['calibration_adopted'] is False
    assert len(result['implementation']['source_sha256']) == 5
    assert result['implementation']['versions']['opencv'] == cv2.__version__
    assert (tmp_path/'out/board-witness.json').is_file()
    with pytest.raises(FileExistsError):
        prepare([path], 'wrist_color', REPO/'config/fiducials.json', measured, tmp_path/'out')


@pytest.mark.parametrize('edge', [True, -.04, 0, float('nan')])
def test_bad_measurement_cannot_supply_metric_scale(tmp_path, edge):
    p = tmp_path/'measurement.json'
    p.write_text(json.dumps({'target': 'board', 'edge_m': edge, 'source': 'operator measurement'}))
    with pytest.raises(ValueError, match='measurement needs'):
        measurement(p)


def test_static_tag_comparison_preserves_planar_ambiguity_and_frame_order():
    pytest.importorskip('scipy')
    # Independent world poses: camera translations and rotations are both nonzero.
    world_tag = np.eye(4)
    world_tag[:3, :3] = cv2.Rodrigues(np.array([.2, -.1, .4]))[0]
    world_tag[:3, 3] = [.3, -.2, .5]
    records = []
    for vector, translation in [([.1, .2, -.3], [.01, -.02, .04]),
                                ([-.4, .15, .2], [.12, .05, -.03])]:
        camera = np.eye(4)
        camera[:3, :3] = cv2.Rodrigues(np.array(vector))[0]
        camera[:3, 3] = translation
        tag = np.linalg.inv(camera) @ world_tag
        alternate = tag.copy()
        alternate[:3, 3] += [.02, -.03, .04]
        branches = [{'camera_from_tag_rotation': m[:3, :3].tolist(),
                     'camera_from_tag_translation_m': m[:3, 3].tolist()} for m in (tag, alternate)]
        records.append({'wrist_pose': {'reference_root_from_camera': camera.tolist()},
                        'tags': [{'family': 'apriltag_16h5', 'id': 7, 'pose_branches': branches}]})
    result = consistency(records)
    assert len(result['comparisons']) == 4
    truth = next(row for row in result['comparisons'] if row['branch_indices'] == [0, 0])
    assert truth['translation_difference_m'] < 1e-12
    assert truth['rotation_difference_deg'] < 1e-5
    wrong = next(row for row in result['comparisons'] if row['branch_indices'] == [0, 1])
    assert wrong['translation_difference_m'] == pytest.approx(np.linalg.norm([.02, -.03, .04]))
    assert result['mount_error_bound_m'] is None and result['branch_selection'] == 'none'
    records[1]['tags'][0]['family'] = 'apriltag_36h11'
    assert consistency(records)['comparisons'] == []


def wrist_inputs(tmp_path):
    from test_stencil_observer import registration_record, wrist_capture
    from wrist_cameras import registry

    root = tmp_path/'inputs'
    root.mkdir()
    for name in ('config/arms.json', 'config/workspace.yaml', 'urdf/tatbot.urdf'):
        (root/name).parent.mkdir(exist_ok=True)
        shutil.copyfile(REPO/name, root/name)
    shutil.copyfile(REPO/'rust/visiond/config/vision.toml', root/'vision.toml')
    (root/'bundle.json').write_text(json.dumps({'bundle_id': 'test', 'cameras': {}}))
    (root/'golden.json').write_text(json.dumps({'calibration_id': 'test', 'world_from_base': np.eye(4).tolist()}))
    registration, _ = registration_record(tmp_path, arm='left')
    shutil.copyfile(registration, root/'registration.json')
    camera = next(c for c in registry(REPO) if c.get('arm') == 'left')
    path, _ = wrist_capture(tmp_path, arm='left', camera=camera['name'])
    manifest = json.loads(path.read_text())
    for stream in ('color', 'depth'):
        meta = manifest['frames'][camera['name']+'_'+stream]['metadata']
        meta['profile'] = dict(camera[stream])
        meta['attributes'].update(device_serial=str(camera['serial']), capture_owner_role=camera['owner_role'])
        if stream == 'color':
            meta['attributes']['bus_source_format'] = meta['profile']['format']
            meta['profile']['format'] = 'rgb8'
    path.write_text(json.dumps(manifest))
    return root, path, manifest, camera['name']+'_color'


@pytest.mark.parametrize('corruption', ['serial', 'profile', 'bundle', 'physical_arm', 'configuration'])
def test_posed_witness_refuses_unbound_camera_and_configuration(tmp_path, corruption):
    root, path, manifest, sensor = wrist_inputs(tmp_path)
    inputs = WristInputs(root, [manifest])
    original = inputs.pose(path, manifest, sensor)
    assert np.asarray(original['reference_root_from_camera']).shape == (4, 4)
    assert original['provenance']['motion_authority'] is False
    assert len(inputs.hashes) == 7
    if corruption == 'configuration':
        (root/'vision.toml').write_text((root/'vision.toml').read_text()+'\n# changed\n')
    else:
        meta = manifest['frames'][sensor]['metadata']
        if corruption == 'serial':
            meta['attributes']['device_serial'] = 'other-device'
        elif corruption == 'profile':
            meta['profile']['fps_num'] += 1
        elif corruption == 'bundle':
            manifest['geometry_calibration_id'] = 'other-bundle'
        elif corruption == 'physical_arm':
            meta['attributes']['physical_arm'] = 'right'
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        inputs.pose(path, manifest, sensor)


@pytest.mark.skipif(not hasattr(cv2, 'calibrateHandEye'), reason='OpenCV build lacks hand-eye Python binding')
def test_hand_eye_recovers_an_independent_mount_and_checks_excluded_motion():
    pytest.importorskip('scipy')
    camera_mount = np.eye(4)
    camera_mount[:3, :3] = cv2.Rodrigues(np.array([.3, -.2, .1]))[0]
    camera_mount[:3, 3] = [.04, -.03, .06]
    world_tag = np.eye(4)
    world_tag[:3, :3] = cv2.Rodrigues(np.array([-.15, .3, .25]))[0]
    world_tag[:3, 3] = [.25, .12, .8]
    records = []
    for n, vector in enumerate(([.1, .2, -.3], [-.4, .15, .2], [.3, -.2, .4],
                                [-.2, -.4, .1], [.4, .1, -.2])):
        flange = np.eye(4)
        flange[:3, :3] = cv2.Rodrigues(np.array(vector))[0]
        flange[:3, 3] = [.02*n, -.01*n, .15+.03*n]
        target = np.linalg.inv(flange @ camera_mount) @ world_tag
        records.append({'wrist_pose': {'reference_root_from_flange': flange.tolist(),
                                       'nominal_flange_from_camera': np.eye(4).tolist()},
                        'tags': [{'family': 'apriltag_16h5', 'id': 7, 'pose_branches':
                                  [{'camera_from_tag_rotation': target[:3, :3].tolist(),
                                    'camera_from_tag_translation_m': target[:3, 3].tolist()}]}]})
    result = fit_mount(records, [0, 1, 2, 3])
    candidate = result['candidates'][0]
    np.testing.assert_allclose(candidate['flange_from_camera'], camera_mount, atol=1e-9)
    excluded = next(row for row in candidate['validation'] if row['capture_index'] == 4)
    assert excluded['used_for_fit'] is False and excluded['translation_difference_m'] < 1e-9
    assert excluded['rotation_difference_deg'] < 1e-7
    assert result['calibration_adopted'] is False and result['mount_error_bound_m'] is None
    for indices in ([0, 0, 1], [0, 1, 2, 3, 4], [0, 1, 9]):
        with pytest.raises(ValueError):
            fit_mount(records, indices)
    for record in records:
        record['wrist_pose']['reference_root_from_flange'] = np.eye(4).tolist()
    degenerate = fit_mount(records, [0, 1, 2])
    assert all(row['status'] == 'unavailable' for row in degenerate['candidates'])
