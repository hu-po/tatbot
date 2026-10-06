"""A Session scan stays in original material coordinates across pose and scan updates."""

import copy
import hashlib
import io
import json

import cv2
import numpy as np
import pytest
from stencil_scan import ScanBinding
from stencil_state import StencilSurfaceState, digest, load_scan, scan_bindings
from stencil_surface import estimate_surface
from test_stencil_cross_camera import two_views
from test_stencil_surface import geometry_fixture

PATTERN = 'stencil-'+'a'*64
REFERENCE = {'pattern_id': PATTERN, 'reference_id': 'b'*64, 'seed': 'fixture'}


def camera(stamp=1_000_000_000, transform=None, curvature=4.):
    observation, frame = geometry_fixture(curvature)
    report, _ = estimate_surface(observation, frame)
    transform = np.eye(4) if transform is None else transform
    return {'capture_timestamp_ns': stamp, 'root_from_camera': transform.tolist(),
            'source_identity': {'producer': 'fixture', 'calibration_id': 'synthetic'},
            'stencils': [{'pattern_id': PATTERN, 'surface': report}]}


def state():
    original = camera()
    binding = scan_bindings({'depth': original}, {PATTERN: REFERENCE})[PATTERN]
    points = np.asarray(binding['points_material_m'])
    colors = np.tile([12, 34, 56], (len(points), 1)).astype(np.uint8)
    return StencilSurfaceState(binding, 'c'*64, points, colors), original


def motion():
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.array([.08, -.13, .21]))[0]
    transform[:3, 3] = [.021, -.014, .009]
    return transform


def archive(value):
    data = io.BytesIO()
    np.savez(data, stencil_scan_bindings=np.str_(json.dumps({PATTERN: value.binding})),
             stencil_rgb_points=value.points, stencil_rgb_colors=value.colors,
             stencil_rgb_patterns=np.array([PATTERN]*len(value.points)),
             stencil_rgb_sources=np.array([value.binding['source']]*len(value.points)))
    payload = data.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def retain(run, sequence, value, expected_sha=None):
    directory = run/f'surface-{sequence}'
    surface = directory/'stroke-check-0/surface.npz'
    surface.parent.mkdir(parents=True)
    payload, sha = archive(value)
    surface.write_bytes(payload)
    (directory/'registration-preview.json').write_text(json.dumps(
        {'first_chunk': 0, 'surface_sha256': expected_sha or sha}))
    return sha


def accept(run, sequence):
    directory = run/f'surface-{sequence}'
    path = directory/'registration-preview.json'
    preview = json.loads(path.read_bytes())
    preview.update(schema='tatbot.session-surface/1', kind='registration-preview')
    path.write_text(json.dumps(preview))
    receipt = {'schema': 'tatbot.session-surface/1', 'kind': 'registration-confirmation',
               'stationary_confirmed': True, 'preview_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory/'registration-accepted.json').write_text(json.dumps(receipt))
    return {'schema': 'tatbot.session-surface/1', 'kind': 'stencil-selection', 'scan_sequence': sequence,
            **{key: {'path': f'surface-{sequence}/{name}',
                     'sha256': hashlib.sha256((directory/name).read_bytes()).hexdigest()}
               for key, name in [('preview', 'registration-preview.json'), ('acceptance', 'registration-accepted.json')]}}


def test_session_selects_accepted_scan_and_ignores_newer_unaccepted_preview(tmp_path):
    original, _ = state()
    first_sha = retain(tmp_path, 10, original)
    selection = accept(tmp_path, 10)
    retain(tmp_path, 20, original)
    reader = ScanBinding(tmp_path, from_session=True)
    reader.refresh()
    assert not reader.states
    reader.select(selection)
    result = reader.observe({'depth': camera(1_100_000_000)}, 1_150_000_000)
    assert result['scan_sequence'] == 10 and result['scan_binding'] == 'session-accepted'
    assert result['selection'] == selection
    assert result['surfaces'][0]['surface_sha256'] == first_sha
    before = reader.states[PATTERN]
    reader.select(copy.deepcopy(selection))
    assert reader.states[PATTERN] is before
    assert before.observe({'depth': camera(1_100_000_000)}, 1_150_000_000)['status'] == 'lost'


@pytest.mark.parametrize('fault', ['unconfirmed', 'wrong_preview', 'changed_receipt', 'other_scan'])
def test_session_selection_needs_exact_acceptance_and_clears_old_results_on_failure(tmp_path, fault):
    original, _ = state()
    retain(tmp_path, 10, original)
    reader = ScanBinding(tmp_path, from_session=True)
    reader.select(accept(tmp_path, 10))
    retain(tmp_path, 20, original)
    selection = accept(tmp_path, 20)
    path = tmp_path/selection['acceptance']['path']
    receipt = json.loads(path.read_bytes())
    if fault == 'unconfirmed':
        receipt['stationary_confirmed'] = False
    if fault == 'wrong_preview':
        receipt['preview_sha256'] = '0'*64
    if fault == 'other_scan':
        selection['acceptance']['path'] = 'surface-10/registration-accepted.json'
    path.write_text(json.dumps(receipt)+'\n')
    if fault != 'changed_receipt':
        selection['acceptance']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    reader.select(selection)
    result = reader.observe({'depth': camera(1_100_000_000)}, 1_150_000_000)
    assert result['surfaces'] == [] and result['error']
    assert reader.display(result) == []


def test_original_curvature_and_colors_follow_measured_rigid_motion_without_renewing_scan():
    model, original = state()
    transform = motion()
    moved = camera(1_100_000_000, transform)
    result = model.observe({'depth': moved}, 1_150_000_000)
    assert result['status'] == 'tracked', result
    np.testing.assert_allclose(result['root_from_material'], transform, atol=1e-12)
    assert result['geometry_capture_ns'] == 1_000_000_000
    assert result['pose_capture_ns'] == 1_100_000_000
    points, colors = model.displayed_points(result)
    np.testing.assert_allclose(points, model.points @ transform[:3, :3].T+transform[:3, 3])
    np.testing.assert_array_equal(colors, model.colors)
    assert not result['motion_authority']
    assert model.observe({}, 1_200_000_000)['status'] == 'lost'
    original['capture_timestamp_ns'] = 1_300_000_000
    recovered = model.observe({'depth': original}, 1_350_000_000)
    np.testing.assert_allclose(recovered['root_from_material'], np.eye(4), atol=1e-12)
    assert recovered['reference_sha256'] == result['reference_sha256']


@pytest.mark.parametrize('stamp,now', [(999_999_999, 1_010_000_000), (2_000_000_000, 1_500_000_000),
                                    (1_100_000_000, 5_000_000_000)])
def test_old_future_and_pre_scan_frames_withhold_pose(stamp, now):
    model, view = state()
    view['capture_timestamp_ns'] = stamp
    result = model.observe({'depth': view}, now)
    assert result['status'] == 'lost' and result['root_from_material'] is None


def test_duplicate_frames_and_changed_camera_epoch_cannot_renew_pose():
    model, view = state()
    assert model.observe({'depth': view}, 1_100_000_000)['status'] == 'tracked'
    assert model.observe({'depth': view}, 1_110_000_000)['status'] == 'lost'
    view['capture_timestamp_ns'] += 200_000_000
    view['source_identity']['producer'] = 'restarted'
    assert model.observe({'depth': view}, 1_300_000_000)['status'] == 'invalidated'
    view['source_identity']['producer'] = 'fixture'
    view['capture_timestamp_ns'] += 200_000_000
    assert model.observe({'depth': view}, 1_500_000_000)['status'] == 'invalidated'


def test_two_visible_copies_of_one_pattern_cannot_bind_or_retarget_a_session():
    view = camera()
    view['stencils'].append(copy.deepcopy(view['stencils'][0]))
    assert PATTERN not in scan_bindings({'depth': view}, {PATTERN: REFERENCE})

    model, _ = state()
    result = model.observe({'depth': view}, 1_100_000_000)
    assert result['status'] == 'invalidated'
    assert 'ambiguous duplicate print' in result['reasons']['identity']
    view['stencils'].pop()
    assert model.observe({'depth': view}, 1_200_000_000)['status'] == 'invalidated'


def test_concurrent_cameras_must_agree_on_one_physical_print_at_scan_binding():
    original = camera()
    same = camera(stamp=1_000_000_010)
    assert PATTERN in scan_bindings({'a': original, 'b': same}, {PATTERN: REFERENCE})
    second_copy = camera(stamp=1_000_000_010, transform=motion())
    assert PATTERN not in scan_bindings({'a': original, 'b': second_copy}, {PATTERN: REFERENCE})


def test_missing_original_coordinates_conflicting_views_and_deformation_withhold_registration():
    model, original = state()
    wrong = copy.deepcopy(original)
    for row in wrong['stencils'][0]['surface']['anchors']:
        row['reference_uv'][0] += .001
    assert model.observe({'depth': wrong}, 1_100_000_000)['status'] == 'lost'
    model, _ = state()
    moved = camera(transform=motion())
    result = model.observe({'a': original, 'b': moved}, 1_100_000_000)
    assert result['status'] == 'invalidated' and 'conflicting' in result['reasons']['identity']
    assert model.observe({'a': original}, 1_200_000_000)['status'] == 'invalidated'
    model, _ = state()
    distorted = copy.deepcopy(original)
    for index, anchor in enumerate(distorted['stencils'][0]['surface']['anchors']):
        anchor['point_camera_m'][2] += .015 * (index % 3 - 1)
    result = model.observe({'depth': distorted}, 1_100_000_000)
    assert result['status'] == 'lost'


@pytest.mark.parametrize('from_session', [False, True])
def test_new_scan_keeps_original_identity_and_registers_new_geometry_to_material_frame(tmp_path, from_session):
    original, _ = state()
    reader = ScanBinding(tmp_path, from_session=from_session)
    retain(tmp_path, 10, original)
    if from_session:
        reader.select(accept(tmp_path, 10))
    first = reader.observe({'depth': camera(1_100_000_000)}, 1_150_000_000)['surfaces'][0]
    transform = motion()
    next_camera = camera(1_200_000_000, transform)
    binding = scan_bindings({'depth': next_camera}, {PATTERN: REFERENCE})[PATTERN]
    moved = StencilSurfaceState(binding, 'd'*64,
        original.points @ transform[:3, :3].T+transform[:3, 3], original.colors)
    sha = retain(tmp_path, 11, moved)
    if from_session:
        reader.select(accept(tmp_path, 11))
    result = reader.observe({'depth': camera(1_300_000_000, transform)}, 1_350_000_000)
    second = result['surfaces'][0]
    assert second['reference_sha256'] == first['reference_sha256']
    assert second['surface_sha256'] == sha
    assert second['geometry_capture_ns'] == 1_200_000_000
    np.testing.assert_allclose(second['root_from_material'], transform, atol=1e-12)
    np.testing.assert_allclose(reader.states[PATTERN].points, original.points, atol=1e-12)
    displayed = reader.display(result, {PATTERN: dict(REFERENCE, clear_center_uv=[.2, .15, .8, .85])})
    np.testing.assert_allclose(displayed[0]['points'], moved.points, atol=1e-12)
    assert displayed[0]['geometry_capture_ns'] != displayed[0]['capture_ns']
    # The original anchors and page boundary ride the same tracked pose.
    np.testing.assert_allclose(displayed[0]['anchors'],
                               original.xyz @ transform[:3, :3].T+transform[:3, 3], atol=1e-9)
    outline = np.asarray(displayed[0]['outline'])
    assert outline.shape == (32, 3) and np.asarray(displayed[0]['clear_center']).shape == (32, 3)
    _, x, y, z = np.asarray(displayed[0]['center'])
    np.testing.assert_allclose([np.linalg.norm(v) for v in (x, y, z)], .025, atol=1e-9)
    np.testing.assert_allclose(np.cross(x, y), z*.025, atol=1e-9)
    assert reader.display({'surfaces': [dict(second, status='lost')]}) == []
    geometry = second['geometry']
    assert geometry['original_surface_sha256'] == first['surface_sha256']
    assert geometry['original_geometry_capture_ns'] == first['geometry_capture_ns']
    assert geometry['surface_sha256'] == sha
    assert geometry['reference_sha256'] == first['reference_sha256']
    assert second['geometry_revision_sha256'] == digest(geometry)
    assert displayed[0]['geometry_revision_sha256'] == second['geometry_revision_sha256']
    np.testing.assert_allclose(geometry['scan_from_material'], transform, atol=1e-12)
    assert geometry['registration_evidence']['inliers'] >= 12
    # A later pose moves the same geometry without creating another scan or
    # changing its original material frame, even after all views are lost.
    another = reader.observe({'depth': camera(1_400_000_000)}, 1_450_000_000)['surfaces'][0]
    assert another['geometry'] == geometry
    assert another['geometry_revision_sha256'] == second['geometry_revision_sha256']
    lost = reader.observe({}, 1_500_000_000)['surfaces'][0]
    assert lost['status'] == 'lost' and lost['geometry'] == geometry
    # A second rescan still names the first surface and registers directly to
    # its anchors. It must not compound motion relative to the previous scan.
    next_transform = transform @ transform
    binding = scan_bindings({'depth': camera(1_600_000_000, next_transform)}, {PATTERN: REFERENCE})[PATTERN]
    newer = StencilSurfaceState(binding, 'e'*64,
        original.points @ next_transform[:3, :3].T+next_transform[:3, 3], original.colors)
    new_sha = retain(tmp_path, 12, newer)
    if from_session:
        reader.select(accept(tmp_path, 12))
    third = reader.observe({}, 1_700_000_000)['surfaces'][0]
    assert third['geometry']['original_surface_sha256'] == first['surface_sha256']
    assert third['reference_sha256'] == first['reference_sha256']
    assert third['surface_sha256'] == new_sha
    np.testing.assert_allclose(third['geometry']['scan_from_material'], next_transform, atol=1e-12)


def test_new_scan_camera_identity_change_cannot_replace_the_geometry():
    original, _ = state()
    before = original.geometry()
    view = camera(1_200_000_000, motion())
    view['source_identity']['producer'] = 'restarted'
    binding = scan_bindings({'depth': view}, {PATTERN: REFERENCE})[PATTERN]
    incoming = StencilSurfaceState(binding, 'd'*64,
        original.points @ motion()[:3, :3].T+motion()[:3, 3], original.colors)
    with pytest.raises(ValueError, match='camera identity'):
        original.incorporate_scan(incoming)
    assert original.geometry() == before


def test_repeated_small_shape_changes_cannot_drift_from_original_geometry():
    original, _ = state()
    initial = original.points.copy()
    for step in range(1, 4):
        incoming = copy.deepcopy(original)
        incoming.geometry_capture_ns = 1_000_000_000 + step*100_000_000
        incoming.points = initial + [0., 0., step*.002]
        if step < 3:
            original.incorporate_scan(incoming)
        else:
            with pytest.raises(ValueError, match='original rigid patch'):
                original.incorporate_scan(incoming)


def test_new_scan_binds_a_new_camera_epoch_for_later_observations():
    original, _ = state()
    view = camera(1_200_000_000, motion())
    view['sensor_name'] = 'new-depth'
    binding = scan_bindings({'new-depth': view}, {PATTERN: REFERENCE})[PATTERN]
    incoming = StencilSurfaceState(binding, 'd'*64,
        original.points @ motion()[:3, :3].T+motion()[:3, 3], original.colors)
    original.incorporate_scan(incoming)
    view['capture_timestamp_ns'] += 100_000_000
    view['source_identity']['producer'] = 'changed-after-scan'
    result = original.observe({'new-depth': view}, 1_400_000_000)
    assert result['status'] == 'invalidated'


def test_tampered_new_scan_clears_old_display_and_can_retry_atomic_artifact(tmp_path):
    original, _ = state()
    retain(tmp_path, 10, original)
    reader = ScanBinding(tmp_path)
    reader.refresh()
    assert reader.states
    retain(tmp_path, 11, original, expected_sha='0'*64)
    result = reader.observe({'depth': camera()}, 1_100_000_000)
    assert result['surfaces'] == [] and 'digest mismatch' in result['error']
    assert reader.display(result) == []


def test_camera_motion_does_not_move_material_and_tracking_camera_switch_is_allowed():
    model, view = state()
    moved_camera = motion()
    view['root_from_camera'] = moved_camera.tolist()
    for anchor in view['stencils'][0]['surface']['anchors']:
        point = np.asarray(anchor['point_camera_m'])
        anchor['point_camera_m'] = ((point-moved_camera[:3, 3]) @ moved_camera[:3, :3]).tolist()
    report = view['stencils'][0]['surface']
    report.update(tracking_camera='fixed_a', tracking_source_identity={'producer': 'a'})
    first = model.observe({'depth': view}, 1_100_000_000)
    np.testing.assert_allclose(first['root_from_material'], np.eye(4), atol=1e-12)
    view['capture_timestamp_ns'] += 200_000_000
    report.update(tracking_camera='fixed_b', tracking_source_identity={'producer': 'b'})
    second = model.observe({'depth': view}, 1_300_000_000)
    np.testing.assert_allclose(second['root_from_material'], np.eye(4), atol=1e-12)


def test_scan_and_live_source_aliases_share_camera_epoch_and_exposure_guards():
    view = camera()
    view['sensor_name'] = 'depth_color'
    report = view['stencils'][0]['surface']
    report.update(tracking_camera='fixed:rgb', tracking_camera_id='rgb',
                  tracking_source_identity={'producer': 'original'})
    binding = scan_bindings({'scan_depth': view}, {PATTERN: REFERENCE})[PATTERN]
    points = np.asarray(binding['points_material_m'])
    model = StencilSurfaceState(binding, 'c'*64, points, np.zeros(points.shape, np.uint8))
    assert model.observe({'live_depth': view}, 1_100_000_000)['status'] == 'tracked'
    assert model.observe({'another_alias': view}, 1_100_000_000)['status'] == 'lost'
    report.update(tracking_camera='rgb', tracking_source_identity={'producer': 'restarted'})
    view['capture_timestamp_ns'] += 100_000_000
    result = model.observe({'live_depth': view}, 1_200_000_000)
    assert result['status'] == 'invalidated'


def test_loader_rejects_changed_hash_and_reference_labels():
    original, _ = state()
    payload, sha = archive(original)
    loaded = load_scan(payload, sha)[PATTERN]
    assert loaded.binding == original.binding
    with pytest.raises(ValueError, match='digest mismatch'):
        load_scan(payload+b'corrupt', sha)
    wrong = copy.deepcopy(original)
    wrong.binding['pattern_id'] = 'stencil-'+'e'*64
    payload, sha = archive(wrong)
    with pytest.raises(ValueError, match='pattern identity'):
        load_scan(payload, sha)


def test_cross_camera_anchor_inventory_keeps_real_depth_points_and_tracking_exposure():
    from stencil_cross_camera import cross_camera_points
    frame, view, row, reference, transform = two_views(4.)
    _, _, report = cross_camera_points(frame, view, row, reference, transform, max_skew_ns=40_000_000)
    assert report['candidate_valid'] and len(report['anchors']) >= 12
    xyz = np.asarray([a['point_camera_m'] for a in report['anchors']])
    pixel = np.rint(xyz[:, :2]/xyz[:, 2, None]*450+[159.5, 119.5]).astype(int)
    np.testing.assert_allclose(xyz[:, 2], frame['depth_m'][pixel[:, 1], pixel[:, 0]])
    assert report['tracking_timestamp_ns'] == row['capture_timestamp_ns']
