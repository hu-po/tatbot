"""Actual RGB samples survive stencil masking, root transforms and receipt replay."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts'), str(REPO/'scripts/lib'), str(REPO/'scripts/vision')]
from stencil_interior import interior_points  # noqa: E402
from stencils import bundle, materialize, prepare  # noqa: E402
from surface_rgb import ScanAppearance, wrist_frame  # noqa: E402
from test_stencil_surface import geometry_fixture  # noqa: E402

REFERENCE = REPO/'docs/assets/stencil-frames/tatbot-43/tracking.json'


def test_portable_references_bind_artwork_and_reject_duplicate_and_changed_bytes(tmp_path):
    value = bundle([REFERENCE])
    paths = materialize(value, tmp_path/'retained')
    assert json.loads(paths[0].read_bytes()) == json.loads(REFERENCE.read_bytes())
    with pytest.raises(ValueError, match='distinct'):
        bundle([REFERENCE, REFERENCE])
    value['references'][0]['png_base64'] = 'YQ=='
    with pytest.raises(ValueError, match='hash'):
        materialize(value, tmp_path/'tampered')


def test_preparation_embeds_artwork_in_hashed_draw_input(tmp_path):
    root, inputs = tmp_path/'job', tmp_path/'scan'
    root.mkdir()
    inputs.mkdir()
    value = bundle([REFERENCE])
    (root/'stencil-view.json').write_text(json.dumps(value))
    (inputs/'draw.json').write_text('{"scan_only": true}')
    prepare(root, inputs)
    assert json.loads((inputs/'draw.json').read_bytes()) == {'scan_only': True, 'stencil_view': value}


@pytest.mark.parametrize('curvature', [0., 4.])
def test_interior_is_measured_rgb_with_holes_and_occluders_excluded(curvature):
    observation, frame = geometry_fixture(curvature)
    y, x = np.indices(frame['depth_m'].shape)
    frame['image'][:] = np.stack((x % 256, y, np.full_like(x, 173)), axis=-1)
    reference = {'clear_center_uv': [.2, .2, .8, .8]}
    # Keep the measured anchor grid intact; these gaps are between its rows.
    frame['depth_m'][121:124, 163:167] = 0
    frame['depth_m'][121:124, 153:157] -= .04
    xyz, colors, report = interior_points(frame, observation, reference)
    assert len(xyz) > 2000, report
    pixels = np.rint(xyz[:, :2]/xyz[:, 2, None]*450+[159.5, 119.5]).astype(int)
    px, py = pixels.T
    np.testing.assert_array_equal(colors, frame['image'][py, px, ::-1])
    np.testing.assert_allclose(xyz[:, 2], frame['depth_m'][py, px])
    missing = ((px >= 153) & (px < 157)) | ((px >= 163) & (px < 167))
    assert not np.any((py >= 121) & (py < 124) & missing)
    assert np.all((px >= 133) & (px <= 187) & (py >= 77) & (py <= 162))
    if curvature:
        assert np.ptp(xyz[:, 2]) > .002  # Curvature is not replaced by a plane.
    assert not report['dense_material_identity_verified']


def test_wrong_pair_or_alignment_withholds_colors():
    observation, frame = geometry_fixture()
    frame['depth_metadata']['sequence'] = 2
    points, _, report = interior_points(frame, observation, {'clear_center_uv': [.2, .2, .8, .8]})
    assert not len(points) and report['reason'] == 'rgbd_capture_pair_unverified'
    frame['depth_metadata']['sequence'] = 1
    intr = json.loads(frame['color_metadata']['attributes']['intrinsics'])
    intr['fx'] += 1
    frame['color_metadata']['attributes']['intrinsics'] = json.dumps(intr)
    assert not len(interior_points(frame, observation, {'clear_center_uv': [.2, .2, .8, .8]})[0])


def test_wrist_color_uses_last_raw_exposure_never_median():
    _, frame = geometry_fixture()
    frame['color_metadata']['sequence'] = frame['depth_metadata']['sequence'] = 10
    capture = {'owner_frames_wrist_upper': json.dumps([{'metadata': frame['depth_metadata']}]*2),
               'owner_color_metadata_wrist_upper': json.dumps(frame['color_metadata']),
               'raw_depth_wrist_upper': np.array([np.full((240, 320), 4000), np.full((240, 320), 4500)], np.uint16),
               'depth_wrist_upper': np.full((240, 320), 4250, np.uint16),
               'units_m_wrist_upper': .0001, 'color_wrist_upper': frame['image'][..., ::-1]}
    np.testing.assert_allclose(wrist_frame(capture, 'wrist_upper')['depth_m'], .45)
    frame['color_metadata']['sequence'] = 11
    capture['owner_color_metadata_wrist_upper'] = json.dumps(frame['color_metadata'])
    with pytest.raises(ValueError, match='last raw depth'):
        wrist_frame(capture, 'wrist_upper')


def test_scan_matches_only_latest_camera_exposure_even_with_out_of_order_files(tmp_path, monkeypatch):
    collector = ScanAppearance(bundle([REFERENCE]), tmp_path)
    observed = []
    monkeypatch.setattr(collector, 'observe', lambda frame, *_: observed.append(frame['timestamp_ns']))
    for timestamp in (1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9):
        collector.defer({'timestamp_ns': timestamp}, 'wrist_upper', np.eye(4), None)
    path = tmp_path/'surface.npz'
    np.savez(path, fixture=np.array([1]))
    collector.write(path)
    assert observed == [12]


def test_multicamera_points_keep_rgb_identity_root_frame_and_clear_on_loss(tmp_path, monkeypatch):
    import surface_rgb as appearance
    observation, frame = geometry_fixture()
    frame['image'][:] = [12, 34, 56]
    value = bundle([REFERENCE])
    ref = value['references'][0]['reference']
    row = dict(observation, pattern_id=ref['pattern_id'], status='acquired')
    class Scene:
        def __init__(self, *args, **kwargs):
            self.bank = type('Bank', (), {'references': {ref['pattern_id']: ref}})()
        def observe(self, *args, **kwargs):
            return {'stencils': [row]}
    monkeypatch.setattr(appearance, 'StencilScene', Scene)
    collector = ScanAppearance(value, tmp_path)
    root_from_camera = np.array([[1, 0, 0, .1], [0, -1, 0, .2], [0, 0, -1, .5], [0, 0, 0, 1.]])
    collector.observe(frame, 'wrist_upper', root_from_camera)
    collector.observe(frame, 'surface_overhead', root_from_camera)
    path = tmp_path/'surface.npz'
    np.savez(path, original_motion_geometry=np.array([1, 2, 3]))
    collector.write(path)
    with np.load(path) as arrays:
        points = arrays['stencil_rgb_points']
        assert len(points) > 4000 and json.loads(str(arrays['stencil_rgb_meta']))['points'] == len(points)
        assert set(arrays['stencil_rgb_sources']) == {'wrist_upper', 'surface_overhead'}
        np.testing.assert_allclose(points[:, 2], .05)
        np.testing.assert_array_equal(arrays['stencil_rgb_colors'], np.tile([56, 34, 12], (len(points), 1)))
        np.testing.assert_array_equal(arrays['original_motion_geometry'], [1, 2, 3])
    collector.unavailable('wrist_upper', 'camera offline')
    row['image_tracking_valid'] = False
    collector.observe(frame, 'surface_overhead', root_from_camera)
    collector.write(path)
    with np.load(path) as arrays:
        assert len(arrays['stencil_rgb_points']) == 0
    # Improper reflection is not a way to make a camera view appear upright.
    root_from_camera[1, 1] = 1
    with pytest.raises(ValueError, match='rigid'):
        collector.observe(frame, 'wrist_upper', root_from_camera)


@pytest.mark.parametrize('tracking_only', [False, True])
def test_scan_borrows_other_camera_track_but_keeps_depth_camera_rgb_and_clears_loss(tmp_path, monkeypatch, tracking_only):
    import copy

    import surface_rgb as appearance
    observation, frame = geometry_fixture()
    frame['image'][:] = [12, 34, 56]
    value = bundle([REFERENCE])
    ref = value['references'][0]['reference']
    class Scene:
        def __init__(self, *args, **kwargs):
            self.bank = type('Bank', (), {'references': {ref['pattern_id']: ref}})()
        def observe(self, image, timestamp_ns, **kwargs):
            valid = image[0, 0, 0] == 99
            return {'stencils': [dict(observation, pattern_id=ref['pattern_id'],
                image_tracking_valid=bool(valid), capture_timestamp_ns=timestamp_ns,
                status='acquired' if valid else 'lost')]}
    monkeypatch.setattr(appearance, 'StencilScene', Scene)
    collector = ScanAppearance(value, tmp_path)
    view = copy.deepcopy(frame)
    view['image'][:] = [99, 0, 0]
    if tracking_only:
        view.pop('depth_m')
        view.pop('depth_metadata')
        view['source_identity'] = {'producer': 'fixed_camera', 'calibration_id': 'test'}
        view['camera_model'] = json.loads(view['color_metadata']['attributes']['intrinsics'])
    root_from_camera = np.array([[1, 0, 0, .1], [0, -1, 0, .2], [0, 0, -1, .5], [0, 0, 0, 1.]])
    collector.observe(frame, 'depth_camera', root_from_camera)
    collector.observe(view, 'tracking_camera', root_from_camera)
    assert not len(collector.batches['depth_camera'][0][0])
    path = tmp_path/'surface.npz'
    np.savez(path, original_motion_geometry=np.array([1, 2, 3]))
    collector.write(path)
    with np.load(path) as arrays:
        selected = arrays['stencil_rgb_sources'] == 'depth_camera'
        assert selected.sum() > 2000
        np.testing.assert_allclose(arrays['stencil_rgb_points'][selected, 2], .05)
        np.testing.assert_array_equal(arrays['stencil_rgb_colors'][selected], np.tile([56, 34, 12], (selected.sum(), 1)))
        report = json.loads(str(arrays['stencil_rgb_meta']))['cameras']['depth_camera']['stencils'][0]['surface']
        assert report['tracking_camera'] == 'tracking_camera'
        assert report['reason'] == 'supported_cross_camera_interior'
        if tracking_only:
            assert not np.any(arrays['stencil_rgb_sources'] == 'tracking_camera')
        np.testing.assert_array_equal(arrays['original_motion_geometry'], [1, 2, 3])
    collector.unavailable('tracking_camera', 'camera offline')
    collector.write(path)
    with np.load(path) as arrays:
        assert not len(arrays['stencil_rgb_points'])
