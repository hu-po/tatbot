"""Coded prints in live observation and replay: decoded as the anchor, flow-tracked with
their bits re-read every frame, lost on any refusal, and never matched by SIFT."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts'), str(REPO/'scripts/lib'), str(REPO/'scripts/vision')]
import stencil_coded  # noqa: E402
import stencil_reference  # noqa: E402
from stencil_coded_live import drawing_pads  # noqa: E402
from stencil_coded_tracker import STEPS, verify_bits  # noqa: E402
from stencil_features import ReferenceBank, Settings  # noqa: E402
from stencil_scene import SceneBank, StencilScene  # noqa: E402
from test_stencil_observer import advance, observer, scene  # noqa: E402

pytest.importorskip('PIL')
LEGACY = REPO/'docs/assets/stencil-frames/tatbot-42/tracking.json'


@pytest.fixture(scope='module')
def refs(tmp_path_factory):
    root = tmp_path_factory.mktemp('coded-live')
    return {seed: stencil_coded.generate(seed, root/seed)/'tracking.json' for seed in ('live-a', 'live-b')}


def page_view(ref, shape=(480, 640), at=(220, 90), size=(200, 300)):
    """A page at 2 px/mm on a grey table, the resolution an overhead camera sees the pad at."""
    image = np.full((*shape, 3), 205, np.uint8)
    page = cv2.resize(cv2.imread(str(Path(ref).with_name('stencil.png'))), size, interpolation=cv2.INTER_AREA)
    image[at[1]:at[1]+size[1], at[0]:at[0]+size[0]] = page
    return image


def moved(image, dx, dy):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), image.shape[1::-1],
                          borderMode=cv2.BORDER_REPLICATE)


def live_scene(paths, **options):
    options = {'decodes_per_turn': 1, 'whole_image_px': 1_400_000, **options}
    bank = SceneBank.from_paths(paths, Settings(max_gap_ms=20_000, verify_interval_ms=180_000), **options)
    return bank, StencilScene(paths, 'session', bank=bank, stable_scene_skip=True)


def test_a_decode_anchors_a_flow_track_that_rereads_the_bits(refs):
    bank, view = live_scene([refs['live-a']])
    image = page_view(refs['live-a'])
    first = view.observe(image, 1_000_000_000, source_id='fixed', regions=None)['stencils'][0]
    assert first['status'] == 'detected' and first['reason'] == 'coded_decoded'
    assert first['physical_instance_identity_verified']
    assert first['physical_instance_id'] == stencil_coded.print_id_for('live-a')
    assert first['coded']['decoded_this_frame'] and len(first['landmarks']) >= 100
    assert first['coded']['bits_disagree'] == 0 and first['coded']['bits_agree'] > 100
    bank.begin_turn()
    second = view.observe(moved(image, 3, -2), 2_000_000_000, source_id='fixed', regions=None)['stencils'][0]
    assert second['status'] == 'tracked' and second['reason'] == 'flow_verified'
    assert second['physical_instance_identity_verified'] and not second['coded']['decoded_this_frame']
    assert second['coded']['bits_agree'] > 100 and second['coded']['bits_disagree'] <= 2
    shift = np.median(np.subtract([p['image_px'] for p in second['landmarks']][:50],
                                  [p['image_px'] for p in first['landmarks']][:50]), 0)
    np.testing.assert_allclose(shift, [3, -2], atol=.3)


def test_a_swapped_print_or_a_slipped_lattice_reads_lost(refs):
    bank, view = live_scene([refs['live-a']])
    image = page_view(refs['live-a'])
    view.observe(image, 1_000_000_000, source_id='fixed')
    tracker = view.trackers[next(iter(view.trackers))]
    active, book = tracker.active, bank.coded.books[tracker.pattern_id]
    mode = bank.coded.modes[tracker.pattern_id]
    keys = [book.key_by_uv[tuple(np.round(uv, 9))] for uv in active['uv']]
    where = dict(zip(keys, active['pixels'], strict=True))
    for step in STEPS[:3]:
        slipped = np.array([where.get((q+step[0], r+step[1]), (np.nan, np.nan)) for q, r in keys])
        keep = np.isfinite(slipped).all(1)
        homography, _ = cv2.findHomography(active['uv'][keep], slipped[keep])
        agree, disagree = verify_bits(image, active['uv'][keep], slipped[keep], book, mode, homography)
        assert agree < 1.5*disagree
    # Another print with the same layout under the live track: flow follows its knots, the bits do not.
    bank.begin_turn()
    swapped = view.observe(page_view(refs['live-b']), 2_000_000_000, source_id='fixed')['stencils'][0]
    assert swapped['status'] == 'lost' and swapped['reason'] == 'coded_bits_disagree'
    assert not swapped['image_tracking_valid'] and not swapped['landmarks']


def test_only_a_registered_code_is_accepted(refs):
    _, view = live_scene([refs['live-b']])
    row = view.observe(page_view(refs['live-a']), 1_000_000_000, source_id='fixed')['stencils'][0]
    assert row['status'] == 'lost' and not row['image_tracking_valid']
    assert row['reason'].startswith('coded_')
    _, view = live_scene([refs['live-a'], refs['live-b']])
    rows = {row['seed']: row for row in view.observe(page_view(refs['live-a']), 1_000_000_000,
                                                      source_id='fixed')['stencils']}
    assert rows['live-a']['status'] == 'detected' and rows['live-b']['status'] == 'lost'


def test_a_mirrored_transfer_decodes_and_verifies(refs):
    _, view = live_scene([refs['live-a']])
    row = view.observe(page_view(refs['live-a'])[:, ::-1].copy(), 1_000_000_000, source_id='fixed')['stencils'][0]
    assert row['status'] == 'detected' and row['mirrored'] and row['physical_instance_identity_verified']


def test_a_large_view_decodes_only_in_its_regions_and_within_the_turn_budget(refs):
    bank, first = live_scene([refs['live-a']], whole_image_px=100_000)
    second = StencilScene([refs['live-a']], 'session', bank=bank, stable_scene_skip=True)
    image = page_view(refs['live-a'])
    row = first.observe(image, 1_000_000_000, source_id='first', regions=[])['stencils'][0]
    assert row['reason'] == 'coded_no_search_region'
    # A view with nothing to search spends none of the turn's decode.
    assert bank.coded.affordable()
    bank.begin_turn()
    row = first.observe(moved(image, 1, 0), 2_000_000_000, source_id='first', regions=None)['stencils'][0]
    assert row['reason'] == 'coded_no_search_region'
    bank.begin_turn()
    box = [(200, 70, 440, 410)]
    row = first.observe(moved(image, 2, 0), 3_000_000_000, source_id='first', regions=box)['stencils'][0]
    assert row['status'] == 'detected'
    # The turn's one decode is spent: another view defers its search, and decodes next turn.
    row = second.observe(image, 3_000_000_000, source_id='second', regions=box)['stencils'][0]
    assert row['status'] == 'lost' and row['reason'] == 'coded_decode_deferred'
    bank.begin_turn()
    row = second.observe(moved(image, 0, 1), 4_000_000_000, source_id='second', regions=box)['stencils'][0]
    assert row['status'] == 'detected'
    # A tracked view keeps its flow track while the budget is spent.
    row = first.observe(moved(image, 3, 0), 4_000_000_000, source_id='first', regions=[])['stencils'][0]
    assert row['status'] == 'tracked' and row['physical_instance_identity_verified']


def test_sift_never_matches_a_coded_print(refs):
    with pytest.raises(ValueError, match='tracked by its code'):
        ReferenceBank([refs['live-a']], scene=True)
    bank = SceneBank.from_paths([refs['live-a'], LEGACY])
    assert bank.sift.references.keys() != bank.coded.references.keys()
    assert set(bank.references) == set(bank.sift.references) | set(bank.coded.references)


def turns_until(worker, views, seed, start_s=1, turns=90):
    """Observer turns (one a second of capture time) until `seed` is measured: the observer's
    coded decode runs in the background and lands a few turns after it starts."""
    for turn in range(turns):
        stamp = (start_s+turn)*1_000_000_000
        advance(views, stamp)
        result = worker.observe(views, {}, stamp+100_000_000)
        by_seed = {target['seed']: target for target in result['targets']}
        if by_seed[seed]['source'] == 'measured':
            return result, turn
        time.sleep(.25)
    raise AssertionError(f'{seed} never measured: {by_seed[seed]["support"]["reason"]}')


def test_the_observer_publishes_a_coded_print_with_its_decoded_id(refs, tmp_path):
    paths, depth, fixed = scene(refs=[refs['live-a'], LEGACY])
    views = {'depth_camera': (depth, np.eye(4)), 'fixed': (fixed, np.eye(4))}
    worker = observer(paths, tmp_path/'observer', publish_targets_only=True)
    pending = worker.observe(views, {}, 1_100_000_000)
    assert {target['seed']: target['source'] for target in pending['targets']}['live-a'] == 'lost'
    first, _ = turns_until(worker, views, 'live-a', start_s=2)
    by_seed = {target['seed']: target for target in first['targets']}
    coded = by_seed['live-a']
    assert coded['source'] == 'measured', coded['support']['reason']
    assert coded['support']['physical_instance_identity_verified']
    assert coded['support']['physical_instance_id'] == stencil_coded.print_id_for('live-a')
    assert coded['support']['reference_physical_instance_id'] == stencil_coded.print_id_for('live-a')
    assert coded['support']['physical_instance_capture_ns'] == coded['capture_ns']
    assert by_seed['tatbot-42']['source'] == 'measured'
    assert first['coded_search']['pad_areas'] == 0      # no arm registration beside this observer
    json.dumps(first, allow_nan=False)
    # The other coded print put where this one was: the track's bits refuse it by name.
    fixed['image'][90:390, 50:250] = cv2.resize(cv2.imread(str(refs['live-b'].with_name('stencil.png'))), (200, 300))
    stamp = (first['targets'][0]['capture_ns']//1_000_000_000+1)*1_000_000_000
    advance(views, stamp)
    swapped = {target['seed']: target for target in worker.observe(views, {}, stamp+100_000_000)['targets']}
    assert swapped['live-a']['source'] == 'lost'
    assert 'coded_bits_disagree' in swapped['live-a']['support']['reason']
    assert swapped['tatbot-42']['source'] == 'measured'


def test_drawing_pads_read_the_workspace(tmp_path):
    config = tmp_path/'config'
    config.mkdir()
    (config/'workspace.yaml').write_text('''schema_version: 1
right:
  pivot_point_x: 0.3
  pivot_point_y: -0.23
  paper_plane_z: 6e-3
left:
  pivot_point_x: null
  pivot_point_y: null
  paper_plane_z: null
partial:
  pivot_point_x: 0.4
  paper_plane_z: 0.006
''')
    pads = drawing_pads(tmp_path)
    assert set(pads) == {'right'}
    np.testing.assert_array_equal(pads['right'], [.3, -.23, .006])


def test_missing_workspace_has_no_drawing_pads(tmp_path):
    assert drawing_pads(tmp_path) == {}


def test_replay_tracks_a_coded_print_in_a_region(refs, tmp_path):
    frames = tmp_path/'frames.jsonl'
    with frames.open('w') as stream:
        for index in range(3):
            path = tmp_path/f'{index}.png'
            cv2.imwrite(str(path), moved(page_view(refs['live-a'], shape=(1100, 1400), at=(600, 400)), index, 0))
            stream.write(json.dumps({'image': str(path), 'timestamp_ns': (index+1)*100_000_000,
                                     'source_id': 'replay'})+'\n')
    output = tmp_path/'out'
    result = subprocess.run([sys.executable, str(REPO/'scripts/vision/stencil_observe.py'), 'replay',
                             '--reference', str(refs['live-a']), '--instance', 'skin-a', '--frames', str(frames),
                             '--output', str(output), '--region', '550,350,850,750'],
                            capture_output=True, text=True, timeout=300,
                            env=dict(os.environ, TATBOT_LOG_ROOT=str(tmp_path/'logs')))
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads((output/'report.json').read_text())
    assert report['image_valid_frames'] == 3
    assert (output/'references'/report['references'][0]['pattern_id']/'coded.json').is_file()


def test_two_copies_in_separate_regions_are_refused(refs):
    image = page_view(refs['live-a'], shape=(480, 1000), at=(60, 90))
    image[90:390, 700:900] = image[90:390, 60:260]
    _, view = live_scene([refs['live-a']], whole_image_px=100_000)
    row = view.observe(image, 1_000_000_000, source_id='fixed',
                       regions=[(20, 50, 300, 430), (660, 50, 940, 430)])['stencils'][0]
    assert row['status'] == 'lost' and row['reason'] == 'coded_duplicate_decode'
    _, view = live_scene([refs['live-a']], whole_image_px=100_000)
    row = view.observe(image, 1_000_000_000, source_id='fixed', regions=[(20, 50, 300, 430)])['stencils'][0]
    assert row['status'] == 'detected'


def test_a_close_wrist_view_searches_the_part_of_the_pad_it_sees():
    from stencil_coded_live import project_areas
    from stencil_cross_camera import project_into_view
    # A D405 colour model: points outside its ray field project to NaN.
    intrinsics = {'schema': 'tatbot.camera-intrinsics/1', 'width': 640, 'height': 480, 'fx': 391.7, 'fy': 391.2,
                  'ppx': 326.3, 'ppy': 238.8, 'distortion_model': 'BrownConradyInverse',
                  'distortion_coefficients': [-.0527, .0588, -.00003, .00042, -.0192]}
    view = {'image': np.zeros((480, 640, 3), np.uint8), 'camera_model': intrinsics}
    pad = np.array([[-.15, -.15, 0], [.15, -.15, 0], [.15, .15, 0], [-.15, .15, 0]])
    for height, found in ((.12, True), (.2, True), (.28, True), (.45, False)):
        root_from_camera = np.diag([1., -1., -1., 1.])
        root_from_camera[:3, 3] = [.08, 0, height]
        boxes = project_areas([pad], np.linalg.inv(root_from_camera), lambda points: project_into_view(points, view),
                              (480, 640), 1.2)
        assert bool(boxes) == found, (height, boxes)


def test_an_unusable_reference_is_published_lost_and_others_still_track(refs, tmp_path):
    from stencil_observer import LiveSurface
    old = tmp_path/'old'
    shutil.copytree(refs['live-b'].parent, old)
    manifest = json.loads((old/'tracking.json').read_text())
    for key in ('coded', 'physical_instance_id'):
        manifest.pop(key)
    manifest['physical_instance_encoded'] = False
    manifest['reference_id'] = stencil_reference.reference_digest(manifest)
    (old/'tracking.json').write_text(json.dumps(manifest))
    paths, depth, fixed = scene(refs=[refs['live-a'], LEGACY])
    views = {'depth_camera': (depth, np.eye(4)), 'fixed': (fixed, np.eye(4))}
    worker = LiveSurface({'references': [str(path) for path in (*paths, old/'tracking.json')],
                          'calibration': {'bundle_id': 'test'}, 'robot_world': {}, 'publish_targets_only': True},
                         tmp_path/'observer')
    result, _ = turns_until(worker, views, 'live-a')
    by_seed = {target['seed']: target for target in result['targets']}
    assert by_seed['live-a']['source'] == 'measured' and by_seed['tatbot-42']['source'] == 'measured'
    assert by_seed['live-b']['source'] == 'lost' and by_seed['live-b']['pattern_id'] == manifest['pattern_id']
    assert 're-export' in by_seed['live-b']['support']['reason']
    assert list(result['refused_references']) == [str(old/'tracking.json')]


def test_registered_prints_side_by_side_each_decode(refs, tmp_path):
    third = stencil_coded.generate('live-c', tmp_path/'live-c')/'tracking.json'
    paths = [refs['live-a'], refs['live-b'], third]
    image = np.full((480, 900, 3), 205, np.uint8)
    for x, path in zip((40, 350, 660), paths, strict=True):
        image[90:390, x:x+200] = cv2.resize(cv2.imread(str(path.with_name('stencil.png'))), (200, 300),
                                            interpolation=cv2.INTER_AREA)
    _, view = live_scene(paths)
    rows = view.observe(image, 1_000_000_000, source_id='fixed')['stencils']
    assert {row['seed']: row['status'] for row in rows} == dict.fromkeys(('live-a', 'live-b', 'live-c'), 'detected')
    assert all(row['physical_instance_identity_verified'] for row in rows)
    centres = {row['seed']: np.mean([p['image_px'][0] for p in row['landmarks']]) for row in rows}
    assert centres['live-a'] < 250 < centres['live-b'] < 560 < centres['live-c']


def test_a_decode_stops_searching_at_its_deadline(refs):
    _, view = live_scene([refs['live-a']], decode_s=0.)
    row = view.observe(page_view(refs['live-a']), 1_000_000_000, source_id='fixed')['stencils'][0]
    assert row['status'] == 'lost' and row['reason'] == 'coded_decode_time_budget'


def test_a_grey_frame_decodes_as_the_luma_stream_carries_it(refs):
    _, view = live_scene([refs['live-a']])
    grey = cv2.cvtColor(page_view(refs['live-a']), cv2.COLOR_BGR2GRAY)
    row = view.observe(grey, 1_000_000_000, source_id='fixed')['stencils'][0]
    assert row['status'] == 'detected' and row['physical_instance_identity_verified']


def test_a_background_decode_lands_on_the_newest_frame(refs):
    """The decode runs beside the turns; its junctions follow the page to the frame the result
    lands on, and that frame's bits verify it."""
    bank, view = live_scene([refs['live-a']], background=True, decode_s=30)
    image = page_view(refs['live-a'])
    rows = []
    try:
        for turn in range(120):
            frame = moved(image, .5*turn, .25*turn)
            started = time.perf_counter()
            rows.append(view.observe(frame, (turn+1)*1_000_000_000, source_id='fixed')['stencils'][0])
            assert time.perf_counter()-started < 1.     # a turn never waits for the decode
            if rows[-1]['status'] == 'detected':
                break
            time.sleep(.25)
    finally:
        bank.coded.close()
    assert rows[0]['reason'] == 'coded_decode_pending'
    row = rows[-1]
    assert row['status'] == 'detected' and row['physical_instance_identity_verified'] and len(rows) > 1
    assert row['coded']['bits_disagree'] <= 2
    truth = np.array([.5, .25])*(len(rows)-1)
    decoded = np.array([p['image_px'] for p in row['landmarks']])
    reference = page_view(refs['live-a'])
    _, still = live_scene([refs['live-a']])
    placed = still.observe(reference, 1_000_000_000, source_id='still')['stencils'][0]
    at_rest = {tuple(np.round(p['reference_uv'], 9)): p['image_px'] for p in placed['landmarks']}
    shifts = [np.subtract(p['image_px'], at_rest[tuple(np.round(p['reference_uv'], 9))])
              for p in row['landmarks'] if tuple(np.round(p['reference_uv'], 9)) in at_rest]
    np.testing.assert_allclose(np.median(shifts, 0), truth, atol=.5)
    assert len(decoded) >= 100


def test_a_print_lost_under_an_arm_is_looked_for_where_it_was(refs):
    """Tracked, then hidden for a few searches (an arm over it): its own box stays the next
    search, so it is found again with no pad region at all; every HINT_EVERY-th search is the
    view's full regions."""
    from stencil_coded_live import HINT_EVERY
    bank = SceneBank.from_paths([refs['live-a']], Settings(max_gap_ms=20_000, verify_interval_ms=180_000),
                                decodes_per_turn=1, whole_image_px=1_400_000)
    view = StencilScene([refs['live-a']], 'session', bank=bank, stable_scene_skip=False)
    image = page_view(refs['live-a'])
    row = view.observe(image, 1_000_000_000, source_id='fixed', regions=[(200, 70, 440, 410)])['stencils'][0]
    assert row['status'] == 'detected'
    hidden = np.full_like(image, 205)
    tracker = next(iter(view.trackers.values()))
    for turn in range(2, 2+HINT_EVERY+1):
        bank.begin_turn()
        row = view.observe(hidden, turn*1_000_000_000, source_id='fixed', regions=[])['stencils'][0]
        assert row['status'] == 'lost'
    assert tracker.hint_polygon is not None and tracker.hint_searches >= HINT_EVERY
    assert view.coded_hint((2+HINT_EVERY+1)*1_000_000_000)   # the observer lets this view search every turn
    for turn in range(10, 10+HINT_EVERY):
        bank.begin_turn()
        row = view.observe(moved(image, 1, 0), turn*1_000_000_000, source_id='fixed', regions=[])['stencils'][0]
        if row['status'] != 'lost':
            break
    assert row['status'] == 'reacquired' and row['physical_instance_identity_verified']
    assert not view.coded_hint(20*1_000_000_000)


def test_an_urgent_search_goes_before_another_views_full_one(refs):
    from stencil_coded_live import CODED_DEFERRED, CodedBank
    bank = CodedBank([refs['live-a']], background=True, decode_s=5)
    image = page_view(refs['live-a'])
    try:
        first = bank.request(image, [(200, 70, 440, 410)], 'camera4', urgent=True)
        assert {v[1] for v in first.values()} == {'coded_decode_pending'}
        for _ in range(200):
            if bank.jobs['camera4']['future'].done():
                break
            time.sleep(.05)
        bank.request(image, [(200, 70, 440, 410)], 'camera4', urgent=True)   # its result; asked again
        other = bank.request(image, [(0, 0, 640, 480)], 'camera1')
        assert {v[1] for v in other.values()} == {CODED_DEFERRED}
    finally:
        bank.close()
