"""Continuous measured appearance, bounded fixed-view searches, one anchor
camera per print, independent loss and calibration refusal."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_rgbd_geometry import alignment

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts'), str(REPO/'scripts/lib'), str(REPO/'scripts/vision')]
import stencil_frame  # noqa: E402
from live_inputs import capture  # noqa: E402
from stencil_observer import MAX_AGE_NS, LiveSurface  # noqa: E402
from stencils import bundle  # noqa: E402

OBSERVER = REPO/'scripts/vision/stencil_observer.py'


def scene(curved=False, refs=None):
    if refs is None:
        refs = [REPO/f'docs/assets/stencil-frames/tatbot-{seed}/tracking.json' for seed in (43, 42)]
    image = np.full((480, 640, 3), 210, np.uint8)
    for x, ref in zip((50, 390), refs, strict=True):
        image[90:390, x:x+200] = cv2.resize(cv2.imread(str(ref.with_name('stencil.png'))), (200, 300))
    intr = {'schema': 'tatbot.camera-intrinsics/1', 'width': 640, 'height': 480,
            'fx': 600., 'fy': 600., 'ppx': 319.5, 'ppy': 239.5,
            'distortion_model': 'None', 'distortion_coefficients': [0.]*5}
    attr = {'capture_epoch': 'test', 'device_serial': 'test', 'intrinsics': json.dumps(intr), 'depth_units_m': '.0001',
            'alignment_calibration': json.dumps(alignment(intr))}
    profile = {'stream': 'color', 'format': 'bgr8', 'width': 640, 'height': 480, 'fps_num': 30, 'fps_den': 1}
    cm = {'sequence': 1, 'sensor_name': 'color', 'profile': profile, 'calibration_id': 'test',
          'attributes': attr, 'timestamps': {'normalized_unix_ns': 1_000_000_000,
              'source_ns': 1_000_000_000, 'source_domain': 'real_sense_hardware'}}
    dm = dict(cm, sensor_name='depth', profile=dict(profile, stream='depth', format='z16'),
              attributes=dict(attr, aligned_to='color'))
    depth = np.full((480, 640), .4)
    if curved:
        depth += .2*((np.arange(640)[None, :]-319.5)/600)**2
    rgbd = {'image': np.full_like(image, [12, 34, 56]), 'depth_m': depth,
            'color_metadata': cm, 'depth_metadata': dm, 'timestamp_ns': 1_000_000_000}
    fixed = {'image': image, 'color_metadata': dict(cm, sensor_name='fixed'), 'timestamp_ns': 1_000_000_000,
             'camera_model': intr, 'source_identity': {'producer': 'fixed-test'}}
    return refs, rgbd, fixed


def advance(views, stamp):
    for frame, _ in views.values():
        frame['timestamp_ns'] = stamp
        for key in ('color_metadata', 'depth_metadata'):
            if key in frame:
                frame[key]['timestamps'] = dict(frame[key]['timestamps'], normalized_unix_ns=stamp, source_ns=stamp)
                frame[key]['sequence'] += 1


def observer(refs, tmp_path, **binding):
    return LiveSurface({'stencils': bundle(refs), 'calibration': {'bundle_id': 'test'}, 'robot_world': {},
                        **binding}, tmp_path)


def test_measured_target_binds_decoded_mark_and_unseen_swap_is_lost(tmp_path):
    def marked(name, instance_id):
        parser = argparse.ArgumentParser()
        stencil_frame.add_arguments(parser)
        args = parser.parse_args(['--output', str(tmp_path/name), '--seed', 'tatbot-43',
                                  '--instance-id', instance_id])
        stencil_frame.validate(args)
        args.output = Path(args.output)
        stencil_frame.render(args, stencil_frame.build(args))
        return args.output/'tracking.json'

    first_ref = marked('first', '0123456789abcdef01234567')
    other_ref = marked('other', 'fedcba9876543210fedcba98')
    legacy = REPO/'docs/assets/stencil-frames/tatbot-42/tracking.json'
    refs, depth, fixed = scene(refs=[first_ref, legacy])
    views = {'depth_camera': (depth, np.eye(4)), 'fixed': (fixed, np.eye(4))}
    worker = observer(refs, tmp_path/'observer', publish_targets_only=True)
    first = worker.observe(views, {}, 1_100_000_000)
    by_seed = {target['seed']: target for target in first['targets']}
    marked_target = by_seed['tatbot-43']
    assert marked_target['source'] == 'measured'
    assert marked_target['support']['physical_instance_identity_verified']
    assert marked_target['support']['physical_instance_id'] == '0123456789abcdef01234567'
    assert marked_target['support']['reference_physical_instance_id'] == '0123456789abcdef01234567'
    assert by_seed['tatbot-42']['source'] == 'measured'
    assert by_seed['tatbot-42']['support']['physical_instance_identity_verified'] is False

    fixed['image'][90:390, 50:250] = cv2.resize(cv2.imread(str(other_ref.with_name('stencil.png'))), (200, 300))
    advance(views, 2_000_000_000)
    swapped = worker.observe(views, {}, 2_100_000_000)
    by_seed = {target['seed']: target for target in swapped['targets']}
    assert by_seed['tatbot-43']['source'] == 'lost'
    assert by_seed['tatbot-43']['physical_instance_id'] is None
    assert by_seed['tatbot-43']['support']['physical_instance_identity_verified'] is False
    assert 'physical_instance_mismatch' in by_seed['tatbot-43']['support']['reason']
    assert by_seed['tatbot-42']['source'] == 'measured'


@pytest.mark.parametrize('curved', [False, True])
def test_continuous_two_surfaces_keep_rgb_and_clear_independent_loss_and_outage(tmp_path, curved):
    refs, depth, fixed = scene(curved)
    matrix = np.diag([1., -1., -1., 1.])
    matrix[:3, 3] = [.2, .3, .5]
    views = {'depth_camera': (depth, matrix), 'fixed': (fixed, matrix)}
    worker = observer(refs, tmp_path)
    first = worker.observe(views, {}, 1_100_000_000)
    assert first['timings_ms']['observe_total'] >= first['timings_ms']['fixed_views']
    assert len(first['surfaces']) == 2, first
    assert first['cameras']['depth_camera']['local_image_search'] is False
    for surface in first['surfaces']:
        points = np.asarray(surface['points'])
        assert len(points) > 3000
        np.testing.assert_array_equal(surface['colors'], np.tile([56, 34, 12], (len(points), 1)))
        if curved:
            assert np.ptp(points[:, 2]) > .002
        else:
            np.testing.assert_allclose(points[:, 2], .1)
    original = fixed['image'].copy()
    fixed['image'][:, :320] = 128
    advance(views, 1_300_000_000)
    partial = worker.observe(views, {}, 1_400_000_000)
    assert [row['seed'] for row in partial['surfaces']] == ['tatbot-42']
    assert first['cameras']['depth_camera']['capture_timestamp_ns'] == 1_000_000_000
    assert first['cameras']['fixed']['capture_timestamp_ns'] == 1_000_000_000
    # Same worker reacquires the uncovered pattern with its original identity.
    fixed['image'][:] = original
    advance(views, 1_700_000_000)
    assert len(worker.observe(views, {}, 1_800_000_000)['surfaces']) == 2
    assert worker.observe({'depth_camera': views['depth_camera']}, {'fixed': 'offline'}, 1_900_000_000)['surfaces'] == []
    assert worker.observe(views, {}, 1_700_000_000+MAX_AGE_NS+1)['surfaces'] == []


def test_depth_region_loss_retries_all_current_cameras_before_declaring_loss(tmp_path):
    refs, depth, fixed = scene()
    worker = observer(refs, tmp_path)
    views = {'depth_camera': (depth, np.eye(4)), 'fixed': (fixed, np.eye(4))}
    first = worker.observe(views, {}, 1_100_000_000)
    assert {target['source'] for target in first['targets']} == {'measured'}
    assert worker.depth_roi is not None
    worker.depth_roi = (0, 0, 4, 4)
    advance(views, 2_000_000_000)
    next_turn = worker.observe(views, {}, 2_100_000_000)
    assert {target['source'] for target in next_turn['targets']} == {'measured'}
    assert next_turn['tracking_views_used'] == ['fixed']


def test_service_tracks_all_current_views_with_one_search_and_reacquires(tmp_path):
    refs, depth, fixed = scene()
    views = {'depth_camera': (depth, np.eye(4))}
    for name in ('fixed_a', 'fixed_b', 'fixed_c'):
        views[name] = (copy.deepcopy(fixed), np.eye(4))
    worker = observer(refs, tmp_path, publish_targets_only=True)
    first = worker.observe(views, {}, 1_100_000_000)
    assert {target['source'] for target in first['targets']} == {'measured'}
    advance(views, 2_000_000_000)
    second = worker.observe(views, {}, 2_100_000_000)
    selected = {name for name in ('fixed_a', 'fixed_b', 'fixed_c')
                if worker.appearance.reports[name]['local_image_search']}
    assert selected == {'fixed_a', 'fixed_b', 'fixed_c'}
    assert sum(worker.appearance.reports[name]['reference_search_allowed'] for name in selected) == 0
    assert {target['source'] for target in second['targets']} == {'measured'}
    # A tracked view disappears. The remaining flow tracks retain both
    # prints without triggering a full search of every fixed camera.
    failed = sorted(selected)[0]
    views[failed][0]['image'][:] = 128
    advance(views, 3_000_000_000)
    third = worker.observe(views, {}, 3_100_000_000)
    assert sum(worker.appearance.reports[name]['reference_search_allowed']
               for name in ('fixed_a', 'fixed_b', 'fixed_c')) == 0
    assert {target['source'] for target in third['targets']} == {'measured'}
    assert all(len(target['support']['cameras']) >= 2 for target in third['targets'])


def test_due_reference_searches_are_spread_across_current_turns(tmp_path, monkeypatch):
    refs, depth, fixed = scene()
    views = {'depth_camera': (depth, np.eye(4))}
    for name in ('fixed_a', 'fixed_b', 'fixed_c'):
        views[name] = (copy.deepcopy(fixed), np.eye(4))
    worker = observer(refs, tmp_path, publish_targets_only=True)
    assert all(row['source'] == 'measured' for row in worker.observe(views, {}, 1_100_000_000)['targets'])
    for name, scene_view in worker.appearance.scenes.items():
        if name == 'depth_camera':
            continue
        scene_view.search_stamp -= 180_000_000_000
        for tracker in scene_view.trackers.values():
            tracker.last_verified -= 180_000_000_000
    bank = worker.appearance.bank.sift
    searches = []
    original = bank.detect_many

    def detect_many(image):
        searches.append(image.shape)
        return original(image)

    monkeypatch.setattr(bank, 'detect_many', detect_many)
    advance(views, 2_000_000_000)
    result = worker.observe(views, {}, 2_100_000_000)
    assert all(row['source'] == 'measured' for row in result['targets'])
    assert len(searches) == 0
    advance(views, 3_000_000_000)
    worker.observe(views, {}, 3_100_000_000)
    assert len(searches) == 0
    advance(views, 4_000_000_000)
    worker.observe(views, {}, 4_100_000_000)
    assert len(searches) == 1
    assert sum(worker.appearance.reports[name]['reference_search_allowed']
               for name in ('fixed_a', 'fixed_b', 'fixed_c')) == 1
    selected = [name for name in ('fixed_a', 'fixed_b', 'fixed_c')
                if worker.appearance.reports[name]['reference_search_allowed']]
    for turn in range(5, 11):
        advance(views, turn * 1_000_000_000)
        worker.observe(views, {}, turn * 1_000_000_000 + 100_000_000)
        selected.extend(name for name in ('fixed_a', 'fixed_b', 'fixed_c')
                        if worker.appearance.reports[name]['reference_search_allowed'])
    assert len(selected) == 3 and set(selected) == {'fixed_a', 'fixed_b', 'fixed_c'}


def test_simultaneous_fixed_view_loss_runs_one_actual_reference_search(tmp_path, monkeypatch):
    refs, _, fixed = scene()
    views = {name: (copy.deepcopy(fixed), np.eye(4)) for name in ('fixed_a', 'fixed_b', 'fixed_c')}
    worker = observer(refs, tmp_path, publish_targets_only=True)
    worker.observe(views, {}, 1_100_000_000)
    searches = []
    original = worker.appearance.bank.sift.detect_many

    def detect_many(image):
        searches.append(image.shape)
        return original(image)

    monkeypatch.setattr(worker.appearance.bank.sift, 'detect_many', detect_many)
    for frame, _ in views.values():
        frame['image'][:] = 128
    advance(views, 2_000_000_000)
    worker.observe(views, {}, 2_100_000_000)
    assert len(searches) == 0
    advance(views, 3_000_000_000)
    worker.observe(views, {}, 3_100_000_000)
    assert len(searches) == 0
    advance(views, 4_000_000_000)
    worker.observe(views, {}, 4_100_000_000)
    assert len(searches) == 1


def test_service_uses_other_tracked_view_when_depth_fit_fails(tmp_path, monkeypatch):
    import surface_rgb
    refs, depth, fixed = scene()
    views = {'depth_camera': (depth, np.eye(4))}
    for name in ('fixed_a', 'fixed_b', 'fixed_c'):
        frame = copy.deepcopy(fixed)
        frame['color_metadata']['sensor_name'] = name
        views[name] = (frame, np.eye(4))
    worker = observer(refs, tmp_path, publish_targets_only=True)
    first = worker.observe(views, {}, 1_100_000_000)
    assert {target['source'] for target in first['targets']} == {'measured'}
    preferred = set(first['tracking_views_used'][:2])
    assert len(preferred) == 2
    original = surface_rgb.cross_camera_points

    def failed_selected(frame, view, observation, reference, *args, **kwargs):
        if view['color_metadata']['sensor_name'] in preferred:
            return np.empty((0, 3)), np.empty((0, 3), np.uint8), {'reason': 'depth_fit_failed'}
        return original(frame, view, observation, reference, *args, **kwargs)

    monkeypatch.setattr(surface_rgb, 'cross_camera_points', failed_selected)
    advance(views, 2_000_000_000)
    second = worker.observe(views, {}, 2_100_000_000)
    assert {target['source'] for target in second['targets']} == {'measured'}
    assert 'fixed_c' in second['tracking_views_used']


def test_empty_view_skips_unchanged_image_but_searches_new_print_immediately(tmp_path):
    refs, _, fixed = scene()
    image = fixed['image'].copy()
    fixed['image'][:] = 210
    views = {'fixed': (fixed, np.eye(4))}
    worker = observer(refs, tmp_path)
    worker.observe(views, {'depth_camera': 'offline'}, 1_100_000_000)
    advance(views, 2_000_000_000)
    worker.observe(views, {'depth_camera': 'offline'}, 2_100_000_000)
    assert {row['reason'] for row in worker.appearance.views['fixed'][1]['stencils']} == {
        'unchanged_scene_since_reference_search'}
    advance(views, 182_000_000_000)
    worker.observe(views, {'depth_camera': 'offline'}, 182_100_000_000)
    assert {row['reason'] for row in worker.appearance.views['fixed'][1]['stencils']} == {
        'insufficient_image_features'}
    fixed['image'][:] = image
    advance(views, 183_000_000_000)
    worker.observe(views, {'depth_camera': 'offline'}, 183_100_000_000)
    assert {row['seed'] for row in worker.appearance.views['fixed'][1]['stencils']
            if row['image_tracking_valid']} == {'tatbot-42', 'tatbot-43'}


@pytest.mark.parametrize('failure', ['missing_track', 'missing_depth_support'])
def test_live_depth_search_recovers_a_pattern_without_fixed_view_support(tmp_path, monkeypatch, failure):
    import surface_rgb
    refs, depth, fixed = scene()
    depth['image'] = fixed['image'].copy()
    if failure == 'missing_track':
        fixed['image'][:, :320] = 128
    else:
        original = surface_rgb.cross_camera_points

        def partial_support(frame, view, observation, reference, *args, **kwargs):
            if reference['seed'] == 'tatbot-43':
                return np.empty((0, 3)), np.empty((0, 3), np.uint8), {'reason': 'unobserved_anchor_depth'}
            return original(frame, view, observation, reference, *args, **kwargs)

        monkeypatch.setattr(surface_rgb, 'cross_camera_points', partial_support)
    worker = observer(refs, tmp_path)
    result = worker.observe({'depth_camera': (depth, np.eye(4)), 'fixed': (fixed, np.eye(4))}, {}, 1_100_000_000)
    assert {row['seed'] for row in result['surfaces']} == {'tatbot-42', 'tatbot-43'}
    assert result['cameras']['depth_camera']['local_image_search'] is True


@pytest.mark.parametrize('wrist_visible', [True, False])
def test_wrist_search_defers_the_overhead_search_until_needed(tmp_path, wrist_visible):
    refs, overhead, fixed = scene()
    overhead['image'] = fixed['image'].copy()
    wrist = copy.deepcopy(overhead)
    wrist['color_metadata']['sensor_name'] = 'wrist_left_color'
    wrist['depth_metadata']['sensor_name'] = 'wrist_left_depth'
    wrist['depth_metadata']['attributes']['aligned_to'] = 'wrist_left_color'
    if not wrist_visible:
        wrist['image'][:] = 128
    fixed['image'][:] = 128
    result = observer(refs, tmp_path).observe({
        'depth_camera': (overhead, np.eye(4)),
        'fixed': (fixed, np.eye(4)),
        'wrist_left': (wrist, np.eye(4)),
    }, {}, 1_100_000_000)
    assert {target['source'] for target in result['targets']} == {'measured'}
    assert result['cameras']['wrist_left']['local_image_search'] is True
    # A successful wrist search supplies the overhead depth by projection.
    # A blind wrist makes the ordinary same-turn fallback search overhead.
    assert result['cameras']['depth_camera']['local_image_search'] is (not wrist_visible)


def retained(tmp_path, *, visible=False):
    """An overhead owner's RGB-D capture as the observer reads it; `visible`
    paints the two prints into its colour image, the default keeps it blank."""
    _, depth, fixed = scene()
    if visible:
        depth['image'] = fixed['image']
    intr = json.loads(depth['color_metadata']['attributes']['intrinsics'])
    camera = {'intrinsics': {'width': 640, 'height': 480, 'fx': 600., 'fy': 600., 'cx': 319.5, 'cy': 239.5},
              'distortion': {'model': 'none', 'coefficients': intr['distortion_coefficients']},
              'world_from_camera': {'rotation': np.eye(3).ravel().tolist(), 'translation_m': [.1, .2, .3]}}
    frames, cameras = {}, {}
    for name, meta, data in [('color', depth['color_metadata'], depth['image'].tobytes()),
                             ('depth', depth['depth_metadata'], (depth['depth_m']*10000).astype('<u2').tobytes())]:
        (tmp_path/f'{name}.pixels').write_bytes(data)
        frames[name] = {'metadata': meta, 'payload_file': f'{name}.pixels', 'payload_bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()}
        cameras[name] = dict(camera, profile=meta['profile'])
    manifest = {'schema': 'tatbot.session-surface/1', 'kind': 'live-capture', 'frames': frames, 'producer': {'node': 'owner'},
                'geometry_calibration_id': 'test', 'wrist_capture_window': {'after_ns': 999_999_999, 'before_ns': 1_000_000_001}}
    path = tmp_path/'capture.json'
    path.write_text(json.dumps(manifest))
    return path, {'bundle_id': 'test', 'cameras': cameras}, {'calibration_id': 'test', 'world_from_base': np.eye(4).tolist()}


def test_retained_capture_preserves_camera_rgb_and_robot_transform(tmp_path):
    path, calibration, robot = retained(tmp_path)
    robot['world_from_base'][0][3] = .4
    frames = capture(path, calibration, robot)
    frame, matrix = frames['color']
    np.testing.assert_array_equal(frame['image'][0, 0], [12, 34, 56])
    np.testing.assert_allclose(frame['depth_m'], .4)
    np.testing.assert_allclose(matrix[:3, 3], [-.3, .2, .3])


@pytest.mark.parametrize('change', ['registration', 'depth_profile', 'color_bytes', 'depth_window'])
def test_live_input_withholds_changed_registration_optics_profile_bytes_or_exposure(tmp_path, change):
    path, calibration, robot = retained(tmp_path)
    manifest = json.loads(path.read_text())
    if change == 'registration':
        robot['calibration_id'] = 'old'
    if change == 'depth_profile':
        manifest['frames']['depth']['metadata']['profile']['fps_num'] = 60
    if change == 'color_bytes':
        (tmp_path/'color.pixels').write_bytes(b'changed')
    if change == 'depth_window':
        manifest['frames']['depth']['metadata']['timestamps']['normalized_unix_ns'] += 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        capture(path, calibration, robot)


@pytest.mark.parametrize('drift', [3e-4, 9e-4, 1.5e-3])
def test_live_depth_admits_thermal_focal_drift_and_binds_the_device(tmp_path, drift):
    path, calibration, robot = retained(tmp_path)
    for camera in calibration['cameras'].values():
        camera['intrinsics'] = dict(camera['intrinsics'], fx=600.*(1+drift), fy=600.*(1-drift))
    frame, _ = capture(path, calibration, robot)['color']
    np.testing.assert_allclose(frame['depth_m'], .4)
    for camera in calibration['cameras'].values():
        camera['metadata'] = {'device_serial': 'another-unit'}
    with pytest.raises(ValueError, match='device differs'):
        capture(path, calibration, robot)


def test_depth_outage_keeps_visible_stencil_without_accepting_a_surface(tmp_path):
    refs, _, visible = scene()
    blank = copy.deepcopy(visible)
    blank['image'][:] = 128
    worker = observer(refs, tmp_path)
    views = {'a': (blank, np.eye(4)), 'b': (visible, np.eye(4))}
    error = {'depth_camera': 'active RGB-D optics differ from bound calibration'}
    for index in range(4):
        stamp = 1_000_000_000+index*100_000_000
        advance(views, stamp)
        result = worker.observe(views, error, stamp+100_000_000)
        # Both fixed views are searched every turn; the one with tracks is
        # the display label, and without depth nothing is measured.
        assert result['cameras']['a']['local_image_search'] is True
        assert result['cameras']['b']['local_image_search'] is True
        assert result['tracking_camera'] == 'b'
        assert result['surfaces'] == []
        assert result['motion_authority'] is False
        assert result['cameras']['depth_camera']['reason'] == error['depth_camera']
        assert {t['source'] for t in result['targets']} == {'lost'}
        assert all(t['support']['reason'].startswith('no overhead depth') for t in result['targets'])
        assert all(t['world_from_target'] is None for t in result['targets'])
    # A stale image is not a track: the next turn has no tracking view at all.
    stale = worker.observe({'b': views['b']}, error, stamp+MAX_AGE_NS+1)
    assert stale['tracking_camera'] is None


def test_live_surfaces_carry_measured_boundaries_anchors_and_centre_axes(tmp_path):
    from stencil_outline import stencil_outline, uv_loop
    refs, depth, fixed = scene()
    matrix = np.diag([1., -1., -1., 1.])
    matrix[:3, 3] = [.2, .3, .5]
    views = {'depth_camera': (depth, matrix), 'fixed': (fixed, matrix)}
    worker = observer(refs, tmp_path)
    result = worker.observe(views, {}, 1_100_000_000)
    assert len(result['surfaces']) == 2
    for surface in result['surfaces']:
        outline, center = np.asarray(surface['outline']), np.asarray(surface['clear_center'])
        assert outline.shape == (32, 3) and center.shape == (32, 3)
        # The synthetic pages lie on the measured plane, 200 x 300 px at 600 px
        # focal length and 0.4 m: 133 x 200 mm measured, not the 100 x 150 mm
        # nominal print, with the clear centre strictly inside the page.
        np.testing.assert_allclose(outline[:, 2], .1, atol=1e-3)
        np.testing.assert_allclose(np.ptp(outline[:, :2], axis=0), [.1333, .2], atol=.002)
        assert (center[:, :2].min(0) > outline[:, :2].min(0)).all()
        assert (center[:, :2].max(0) < outline[:, :2].max(0)).all()
        anchors = np.asarray(surface['anchors'])
        assert 12 <= len(anchors) <= 256 and np.allclose(anchors[:, 2], .1, atol=2e-3)
        origin, x, y, z = np.asarray(surface['center'])
        assert np.linalg.norm(origin[:2]-(outline[:, :2].min(0)+outline[:, :2].max(0))/2) < .003
        np.testing.assert_allclose(np.cross(x, y), z*.025, atol=1e-6)
    # A refused boundary is reported beside the points, never drawn.
    from stencil_observer import outline
    report = result['cameras']['depth_camera']
    pattern = result['surfaces'][0]['pattern_id']
    unsupported = copy.deepcopy(report)
    next(row for row in unsupported['stencils'] if row['pattern_id'] == pattern)['surface']['candidate_valid'] = False
    assert outline(unsupported, pattern, {}) == {'outline_reason': 'no supported surface fit'}
    assert 'outline' not in outline(report, 'stencil-'+'0'*64, {})
    with pytest.raises(ValueError):
        stencil_outline(np.zeros((11, 2)), np.zeros((11, 3)), np.eye(4))
    with pytest.raises(ValueError):
        uv_loop((0, 0, 1, 1.5))


def test_every_fixed_view_is_searched_each_turn_and_each_print_names_its_anchor(tmp_path):
    refs, depth, fixed = scene()
    half = copy.deepcopy(fixed)
    half['image'][:, :320] = 128          # this camera sees only tatbot-42
    half['color_metadata'] = dict(half['color_metadata'], sensor_name='half')
    half['source_identity'] = {'producer': 'half-test'}
    views = {'depth_camera': (depth, np.eye(4)), 'fixed': (fixed, np.eye(4)), 'half': (half, np.eye(4))}
    worker = observer(refs, tmp_path)
    result = worker.observe(views, {}, 1_100_000_000)
    assert result['cameras']['fixed']['local_image_search'] is True
    assert result['cameras']['half']['local_image_search'] is True
    # Every eligible fixed view tracks every print, so the RGB-D view's own
    # search stays deferred and each print is anchored through a fixed view.
    assert result['cameras']['depth_camera']['local_image_search'] is False
    targets = {t['seed']: t for t in result['targets']}
    assert set(targets) == {'tatbot-42', 'tatbot-43'}
    assert all(t['source'] == 'measured' for t in targets.values())
    assert targets['tatbot-43']['support']['anchor_camera'] == 'fixed'
    assert set(targets['tatbot-43']['support']['cameras']) == {'fixed'}
    # The print both cameras see carries both fits and their disagreement.
    both = targets['tatbot-42']['support']
    assert set(both['cameras']) == {'fixed', 'half'}
    assert both['anchor_camera'] in {'fixed', 'half'}
    other = ({'fixed', 'half'} - {both['anchor_camera']}).pop()
    assert both['cameras'][other]['centre_delta_mm'] < 3
    assert both['cameras'][other]['normal_delta_deg'] < 1
    assert both['cameras'][both['anchor_camera']]['centre_delta_mm'] == 0
    for target in targets.values():
        assert target['support']['anchors'] >= 12
        assert target['support']['motion_authority'] is False


def test_anchor_camera_does_not_switch_for_marginal_fit_changes(tmp_path, monkeypatch):
    refs, depth, fixed = scene()
    views = {'depth_camera': (depth, np.eye(4)), 'fixed': (fixed, np.eye(4)),
             'second': (copy.deepcopy(fixed), np.eye(4))}
    worker = observer(refs, tmp_path)
    result = worker.observe(views, {}, 1_100_000_000)
    target = next(t for t in result['targets'] if t['source'] == 'measured')
    pattern = target['pattern_id']
    current = target['support']['anchor_camera']
    source = {key: copy.deepcopy(worker._candidates(key)) for key in worker.references}
    challenger = next(c for c in source[pattern] if c['camera'] != current and c['anchor_eligible'])
    challenger_camera = challenger['camera']
    gain = 2
    omit = False

    def candidates(key):
        rows = copy.deepcopy(source[key])
        if key == pattern:
            if omit:
                return [c for c in rows if c['camera'] != challenger_camera]
            previous = next(c for c in rows if c['camera'] == current)
            challenger = next(c for c in rows if c['camera'] == challenger_camera)
            previous['anchors'] = 20
            challenger['anchors'] = 20 + gain
            challenger['loo_p95_m'] = previous['loo_p95_m']
        return rows

    monkeypatch.setattr(worker, '_candidates', candidates)
    assert next(t for t in worker.targets({}) if t['pattern_id'] == pattern)['support']['anchor_camera'] == current
    gain = 5
    assert next(t for t in worker.targets({}) if t['pattern_id'] == pattern)['support']['anchor_camera'] == challenger_camera
    omit = True
    assert next(t for t in worker.targets({}) if t['pattern_id'] == pattern)['support']['anchor_camera'] == current


def test_an_excluded_camera_is_reported_but_never_an_anchor(tmp_path):
    refs, depth, fixed = scene()
    depth['image'] = fixed['image'].copy()
    views = {'depth_camera': (depth, np.eye(4)), 'fixed': (fixed, np.eye(4))}
    result = observer(refs, tmp_path, excluded_anchors=['fixed']).observe(views, {}, 1_100_000_000)
    # No eligible fixed view tracks the prints, so the RGB-D view is searched
    # itself and anchors both; the excluded camera's fit is still reported.
    assert result['cameras']['depth_camera']['local_image_search'] is True
    for target in result['targets']:
        support = target['support']
        assert target['source'] == 'measured'
        assert support['anchor_camera'] == 'depth_camera'
        assert support['cameras']['fixed']['anchor_eligible'] is False
        assert support['cameras']['fixed']['centre_delta_mm'] < 3
        assert support['excluded_anchors'] == ['fixed']
    # With the RGB-D view blind, the excluded camera alone cannot measure.
    depth['image'][:] = 128
    advance(views, 1_200_000_000)
    lost = observer(refs, tmp_path/'blind', excluded_anchors=['fixed']).observe(views, {}, 1_300_000_000)
    assert {t['source'] for t in lost['targets']} == {'lost'}
    assert {t['support']['reason'] for t in lost['targets']} == {'only excluded cameras measured the print'}
    assert all(set(t['support']['cameras']) == {'fixed'} for t in lost['targets'])


def test_a_target_pose_carries_corners_plane_and_sigmas_and_never_motion_authority(tmp_path):
    refs, depth, fixed = scene()
    matrix = np.diag([1., -1., -1., 1.])
    matrix[:3, 3] = [.2, .3, .5]
    views = {'depth_camera': (depth, matrix), 'fixed': (fixed, matrix)}
    result = observer(refs, tmp_path).observe(views, {}, 1_100_000_000)
    assert result['inventory_sha256'] == hashlib.sha256(json.dumps(sorted(
        json.loads(ref.read_text())['reference_id'] for ref in refs)).encode()).hexdigest()
    for target, surface in zip(sorted(result['targets'], key=lambda t: t['pattern_id']),
                               sorted(result['surfaces'], key=lambda s: s['pattern_id']), strict=True):
        assert target['pattern_id'] == surface['pattern_id']
        assert target['source'] == 'measured' and target['target_frame'] == 'world'
        assert len(target['reference_id']) == 64
        pose = np.asarray(target['world_from_target'])
        assert pose.shape == (4, 4) and np.isfinite(pose).all()
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-9)
        # The pose is the page's centre frame: on the measured plane, at the
        # centre of the outline the display draws, its axes the outline's.
        outline = np.asarray(surface['outline'])
        assert abs(pose[2, 3]-.1) < 1e-3
        assert np.linalg.norm(pose[:2, 3]-(outline[:, :2].min(0)+outline[:, :2].max(0))/2) < .003
        origin, x, y, z = np.asarray(surface['center'])
        np.testing.assert_allclose(pose[:3, 3], origin, atol=1e-6)
        np.testing.assert_allclose(pose[:3, 2]*.025, z, atol=1e-6)
        support = target['support']
        corners = np.asarray(support['corners_m'])
        assert corners.shape == (4, 3) and np.allclose(corners[:, 2], .1, atol=1e-3)
        np.testing.assert_allclose(np.ptp(corners[:, :2], axis=0), [.1333, .2], atol=.002)
        clear = np.asarray(support['clear_corners_m'])
        assert (clear[:, :2].min(0) > corners[:, :2].min(0)).all()
        assert (clear[:, :2].max(0) < corners[:, :2].max(0)).all()
        np.testing.assert_allclose(support['plane']['normal'], pose[:3, 2], atol=1e-9)
        assert 0 <= target['translation_sigma_m'] < .005
        assert 0 <= target['rotation_sigma_rad'] < .05
        assert support['loo_p95_m'] < .005 and support['anchors'] >= 12
        assert support['sigma_method']
        assert support['bundle_id'] == 'test'
        assert support['motion_authority'] is False
        assert target['capture_ns'] == 1_000_000_000


def test_zero_references_and_stroke_support_are_refused(tmp_path):
    refs, depth, fixed = scene()
    with pytest.raises(ValueError, match='one to eight'):
        LiveSurface({'references': [], 'calibration': {}, 'robot_world': {}}, tmp_path)
    worker = observer(refs, tmp_path)
    worker.observe({'depth_camera': (depth, np.eye(4)), 'fixed': (fixed, np.eye(4))}, {}, 1_100_000_000)
    with pytest.raises(ValueError, match='not an observer product'):
        worker.request({'kind': 'support', 'pattern_id': next(iter(worker.references))})


def test_the_observer_protocol_answers_a_capture_with_targets_from_installed_references(tmp_path):
    """The service's own path: references by path, an identity world, one
    capture request on stdin, one line of targets on stdout."""
    refs, _, _ = scene()
    capture_dir = tmp_path/'capture'
    capture_dir.mkdir()
    path, calibration, _ = retained(capture_dir, visible=True)
    # The service hands over exposures a few hundred milliseconds old.
    stamp = time.time_ns()-200_000_000
    manifest = json.loads(path.read_text())
    for entry in manifest['frames'].values():
        entry['metadata']['timestamps'].update(normalized_unix_ns=stamp, source_ns=stamp)
    manifest['wrist_capture_window'] = {'after_ns': stamp-1, 'before_ns': stamp+1}
    path.write_text(json.dumps(manifest))
    binding = {'references': [str(ref) for ref in refs], 'calibration': calibration,
               'robot_world': {'calibration_id': 'test', 'world_from_base': np.eye(4).tolist()},
               'excluded_anchors': ['camera3'], 'observer_epoch': 'test', 'evidence_kind': 'live-rgbd',
               'publish_targets_only': True}
    (tmp_path/'binding.json').write_text(json.dumps(binding))
    request = json.dumps({'captures': {'overhead-depth': str(path)}, 'errors': {'poe-cameras': 'no fixed owner'},
                          'accepted_scan': None})+'\n'
    proc = subprocess.run([sys.executable, str(OBSERVER), str(tmp_path/'binding.json'), str(tmp_path)],
                          input=request, capture_output=True, text=True, timeout=120, check=True)
    ready, reply = proc.stdout.splitlines()
    assert json.loads(ready) == {'ready': True}
    result = json.loads(reply)
    assert 'error' not in result
    assert result['calibration_id'] == 'test'
    assert result['excluded_anchors'] == ['camera3']
    assert len(result['inventory_sha256']) == 64
    assert 'surfaces' not in result and 'cameras' not in result
    assert result['timings_ms']['per_camera']['color']['localize'] >= 0
    targets = {t['seed']: t for t in result['targets']}
    assert set(targets) == {'tatbot-42', 'tatbot-43'}
    for target in targets.values():
        assert target['source'] == 'measured', target['support']['reason']
        assert target['support']['anchor_camera'] == 'color'
        assert target['capture_ns'] == stamp
        assert target['support']['motion_authority'] is False
        assert target['support']['excluded_anchors'] == ['camera3']


def wrist_capture(directory, *, arm='right', camera='realsense2', stamp=1_000_000_000, joints=None,
                  bundle='test', physical_arm=None):
    """The service's `wrist-<arm>` capture: the arm's D405 colour/depth pair
    with the owner's active intrinsics and the joints stencild paired with it."""
    _, depth, fixed = scene()
    intr = json.loads(depth['color_metadata']['attributes']['intrinsics'])
    attributes = dict(depth['color_metadata']['attributes'], physical_arm=physical_arm or arm,
                      capture_owner_role='realsense')
    cm = dict(depth['color_metadata'], sensor_name=f'{camera}_color', attributes=attributes,
              timestamps={'normalized_unix_ns': stamp, 'source_ns': stamp, 'source_domain': 'real_sense_hardware'})
    dm = dict(cm, sensor_name=f'{camera}_depth', profile=dict(cm['profile'], stream='depth', format='z16'),
              attributes=dict(attributes, aligned_to=f'{camera}_color', depth_units_m='.0001'))
    frames = {}
    for meta, data in [(cm, fixed['image'].tobytes()), (dm, (depth['depth_m']*10000).astype('<u2').tobytes())]:
        name = meta['sensor_name']
        (directory/f'{name}.pixels').write_bytes(data)
        frames[name] = {'metadata': meta, 'payload_file': f'{name}.pixels', 'payload_bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()}
    manifest = {'schema': 'tatbot.session-surface/1', 'kind': 'live-capture', 'frames': frames, 'producer': {'node': 'arm-node'},
                'geometry_calibration_id': bundle, 'wrist_capture_window': {'after_ns': stamp-1, 'before_ns': stamp+1},
                'wrist': {'arm': arm, 'camera': camera, 'joints': [.1, -.2, .3, -.4, .5, -.6] if joints is None else joints,
                          'carriage_m': .002, 'measured_wall_ns': stamp-12_500_000, 'joints_skew_ms': 12.5,
                          'joints_calibration_id': None}}
    path = directory/f'wrist-{arm}.json'
    path.write_text(json.dumps(manifest))
    return path, intr


def subdir(root, name):
    path = root/name
    path.mkdir()
    return path


def registration_record(directory, arm='right', calibration='test'):
    world_from_root = np.diag([1., -1., -1., 1.])
    world_from_root[:3, 3] = [.2, .3, .5]
    root_from_arm_base = np.eye(4)
    root_from_arm_base[:3, 3] = [-.1, .05, 0.]
    record = {'schema': 'tatbot.arm-registration/1', 'arm': arm, 'calibration_id': calibration,
              'world_from_root': world_from_root.tolist(),
              'world_from_arm_base': (world_from_root @ root_from_arm_base).tolist()}
    path = directory/f'arm-registration-{arm}-current.json'
    path.write_text(json.dumps(record))
    return path, world_from_root


@pytest.fixture
def wrist_config(tmp_path, monkeypatch):
    """Synthetic camera inventory and zero offsets, independent of deployment."""
    import tool_spec

    monkeypatch.setattr(tool_spec, 'read_workspace', lambda _: {})
    config = tmp_path/'vision.toml'
    cameras = [('left', 'realsense1', 'wrist_left'), ('right', 'realsense2', 'wrist_upper')]
    config.write_text('schema_version = 1\n'+''.join(f'''
[[cameras.realsense]]
name = "{name}"
serial = "test-{arm}"
role = "{role}"
group = "d405"
arm = "{arm}"
owner_role = "realsense"

[cameras.realsense.color]
stream = "color"
width = 640
height = 480
fps_num = 30
fps_den = 1
format = "yuyv"

[cameras.realsense.depth]
stream = "depth"
width = 640
height = 480
fps_num = 30
fps_den = 1
format = "z16"
''' for arm, name, role in cameras))
    return config


def test_wrist_pose_applies_joint_calibration_to_a_copy_of_measured_joints(tmp_path, monkeypatch, wrist_config):
    import tool_spec
    from stencil_observer import WristPoser
    from urdf_kinematics import UrdfChain, driver_joint_names

    offsets = [0., .012, -.018, .009, -.007, 0., 0.]
    monkeypatch.setattr(tool_spec, 'read_workspace', lambda _: {'right': {'joint_offsets_rad': offsets}})
    path, world_from_root = registration_record(tmp_path)
    poser = WristPoser({'wrist': {'registrations': {'right': str(path)}, 'vision_config': str(wrist_config)}},
                       {'bundle_id': 'test'})
    raw = np.array([.1, -.2, .3, -.4, .5, -.6])
    saved = raw.copy()
    matrix, provenance = poser.pose('right', 'realsense2', raw, .002)
    values = dict(zip(driver_joint_names('right'), np.r_[raw, .002]+offsets, strict=True))
    expected = world_from_root @ UrdfChain(REPO/'urdf/tatbot.urdf').link_pose(
        'right/realsense_color_optical_frame', values)
    np.testing.assert_allclose(matrix, expected, atol=1e-12)
    np.testing.assert_array_equal(raw, saved)
    assert provenance['joint_offsets_rad'] == offsets


@pytest.mark.parametrize('offsets', [
    [0.] * 7,
    [0., .012, -.018, .009, -.007, 0., 0.],
], ids=['uncalibrated', 'calibrated'])
def test_a_wrist_view_enters_the_fusion_with_its_own_intrinsics_and_the_arms_registration(
        tmp_path, monkeypatch, offsets, wrist_config):
    """One arm's D405 pair, posed by its own registration times the URDF at
    the paired joints, carrying the owner's active intrinsics rather than a
    bundle entry: it is searched every turn, fits the prints it sees and is
    named with its joint skew; a view the service could not pose is a named
    refusal beside it, never a lost print."""
    import tool_spec
    from stencil_observer import WristPoser, wrist_view
    from urdf_kinematics import UrdfChain, driver_joint_names
    monkeypatch.setattr(tool_spec, 'read_workspace', lambda _: {'right': {'joint_offsets_rad': offsets}})
    refs, _, _ = scene()
    path, intr = wrist_capture(tmp_path)
    registration, world_from_root = registration_record(tmp_path)
    calibration = {'bundle_id': 'test', 'cameras': {}}
    robot = {'calibration_id': 'test', 'world_from_base': np.eye(4).tolist()}
    binding = {'wrist': {'registrations': {'right': str(registration)}, 'urdf': str(REPO/'urdf/tatbot.urdf'),
                         'vision_config': str(wrist_config)}}
    name, (frame, matrix), provenance = wrist_view(path, calibration, robot, WristPoser(binding, calibration))
    assert name == 'wrist_right'
    assert frame['camera_model'] == intr, 'the owner\'s active intrinsics, no bundle entry'
    assert frame['image'].shape == (480, 640, 3) and frame['depth_m'].shape == (480, 640)
    chain = UrdfChain(REPO/'urdf/tatbot.urdf')
    measured = np.array([.1, -.2, .3, -.4, .5, -.6, .002])
    values = dict(zip(driver_joint_names('right'), measured + offsets, strict=True))
    expected = world_from_root @ chain.link_pose('right/realsense_color_optical_frame', values)
    np.testing.assert_allclose(matrix, expected, atol=1e-12)
    assert provenance['joints_skew_ms'] == 12.5 and provenance['registration']['carried'] is None
    assert provenance['motion_authority'] is False
    assert provenance['registration']['joint_offsets_rad'] == offsets
    # Through the observer's own turn: the wrist view is searched (never
    # deferred like the fixed overhead), the prints it sees are fitted at the
    # pose FK gives, and the service's refusal of the other wrist is named.
    stamp = time.time_ns()-200_000_000
    live, _ = wrist_capture(subdir(tmp_path, 'live'), stamp=stamp)
    worker = LiveSurface({'stencils': bundle(refs), 'calibration': calibration, 'robot_world': robot,
                          'publish_targets_only': True, **binding}, tmp_path)
    result = worker.request({'captures': {'wrist-right': str(live)},
                             'errors': {'poe-cameras': 'no fixed owner', 'wrist-left': 'no joints inside 1000 ms of the exposure'},
                             'accepted_scan': None})
    assert 'error' not in result
    assert result['wrist_views']['wrist_left'] == {'refused': 'no joints inside 1000 ms of the exposure'}
    assert result['wrist_views']['wrist_right']['joints_skew_ms'] == 12.5
    assert result['wrist_views']['wrist_right']['capture_ns'] == stamp
    targets = {t['seed']: t for t in result['targets']}
    assert set(targets) == {'tatbot-42', 'tatbot-43'}
    for target in targets.values():
        assert target['source'] == 'measured', target['support']['reason']
        assert target['support']['anchor_camera'] == 'wrist_right'
        assert target['support']['motion_authority'] is False
    # The print painted at image x 50..250, depth .4 m: its centre in the camera
    # is deprojected through the frame's own intrinsics, then posed by FK.
    centre_camera = np.array([(150-319.5)/600*.4, (240-239.5)/600*.4, .4, 1.])
    np.testing.assert_allclose(np.asarray(targets['tatbot-43']['world_from_target'])[:3, 3],
                               (expected @ centre_camera)[:3], atol=.004)
    # Refusals name their cause: the other arm's camera, a missing registration,
    # a registration solved against another world.
    other, _ = wrist_capture(subdir(tmp_path, 'other'), physical_arm='left')
    with pytest.raises(ValueError, match="capture is of the 'left' arm, not 'right'"):
        wrist_view(other, calibration, robot, WristPoser(binding, calibration))
    left, _ = wrist_capture(subdir(tmp_path, 'left'), arm='left', camera='realsense1')
    with pytest.raises(ValueError, match='the left arm has no registration installed'):
        wrist_view(left, calibration, robot, WristPoser(binding, calibration))
    stale, _ = registration_record(subdir(tmp_path, 'stale'), calibration='e'*64)
    with pytest.raises(ValueError, match='solved against another camera bundle'):
        wrist_view(path, calibration, robot, WristPoser(
            {'wrist': {**binding['wrist'], 'registrations': {'right': str(stale)}}}, calibration))


def test_two_wrist_cameras_on_one_arm_use_distinct_roles_and_fixed_extrinsics(tmp_path, wrist_config):
    from stencil_observer import WristPoser, wrist_view

    config = wrist_config
    config.write_text(config.read_text().replace(
        'name = "realsense2"\n',
        'name = "realsense2"\noptical_frame = "right/realsense_color_optical_frame"\n'
        'depth_optical_frame = "right/realsense_depth_optical_frame"\n', 1) + '''
[[cameras.realsense]]
name = "extra"
serial = "test-extra"
role = "wrist_lower"
group = "d405"
arm = "right"
owner_role = "realsense"
optical_frame = "right/extra_color_optical_frame"
depth_optical_frame = "right/extra_depth_optical_frame"

[cameras.realsense.color]
stream = "color"
width = 640
height = 480
fps_num = 30
fps_den = 1
format = "yuyv"

[cameras.realsense.depth]
stream = "depth"
width = 640
height = 480
fps_num = 30
fps_den = 1
format = "z16"
''')
    urdf = tmp_path/'tatbot.urdf'
    extra = '''
  <link name="right/extra_color_optical_frame"/>
  <joint name="right/extra_color_joint" type="fixed">
    <origin xyz="0.06 0 0" rpy="0 0 0"/>
    <parent link="right/link_6"/><child link="right/extra_color_optical_frame"/>
  </joint>
  <link name="right/extra_depth_optical_frame"/>
  <joint name="right/extra_depth_joint" type="fixed">
    <origin xyz="0.06 0 0" rpy="0 0 0"/>
    <parent link="right/link_6"/><child link="right/extra_depth_optical_frame"/>
  </joint>
'''
    urdf.write_text((REPO/'urdf/tatbot.urdf').read_text().replace('</robot>', extra+'</robot>'))
    registration, _ = registration_record(tmp_path)
    binding = {'wrist': {'registrations': {'right': str(registration)}, 'urdf': str(urdf),
                         'vision_config': str(config)}}
    calibration = {'bundle_id': 'test', 'cameras': {}}
    robot = {'calibration_id': 'test', 'world_from_base': np.eye(4).tolist()}
    poser = WristPoser(binding, calibration)
    upper, _ = wrist_capture(subdir(tmp_path, 'upper'))
    lower, _ = wrist_capture(subdir(tmp_path, 'lower'), camera='extra')
    upper_name, (_, upper_pose), _ = wrist_view(upper, calibration, robot, poser,
                                                'wrist-right-realsense2')
    lower_name, (_, lower_pose), _ = wrist_view(lower, calibration, robot, poser,
                                                'wrist-right-extra')
    assert (upper_name, lower_name) == ('wrist_right_realsense2', 'wrist_right_extra')
    assert not np.allclose(upper_pose, lower_pose), 'second view must use its own fixed extrinsic'
    with pytest.raises(ValueError, match='capture belongs to wrist-right-extra'):
        wrist_view(lower, calibration, robot, poser, 'wrist-right-realsense2')
    missing = dict(binding, wrist=dict(binding['wrist'], vision_config=str(config)))
    config.write_text(config.read_text().replace('optical_frame = "right/extra_color_optical_frame"\n', ''))
    with pytest.raises(ValueError, match='multiple cameras require explicit optical frames'):
        wrist_view(lower, calibration, robot, WristPoser(missing, calibration))


def test_conflicting_current_views_latch_same_pattern_identity_until_new_binding(tmp_path):
    refs, depth, fixed = scene()
    depth['image'] = fixed['image']
    worker = observer(refs, tmp_path, publish_targets_only=True)
    displaced = np.eye(4)
    displaced[:3, 3] = [.1, 0., 0.]
    both = worker.observe({'a': (copy.deepcopy(depth), np.eye(4)),
                           'b': (copy.deepcopy(depth), displaced)}, {}, 1_100_000_000)
    assert all(target['source'] == 'lost' and 'ambiguous' in target['support']['reason']
               for target in both['targets'])
    later = copy.deepcopy(depth)
    advance({'a': (later, np.eye(4))}, 1_200_000_000)
    one = worker.observe({'a': (later, np.eye(4))}, {}, 1_300_000_000)
    assert all(target['source'] == 'lost' and 'new physical target binding' in target['support']['reason']
               for target in one['targets'])
