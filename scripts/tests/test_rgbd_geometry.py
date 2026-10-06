"""Independent metric witnesses for shared native-depth/color-frame conversion."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from rgbd_geometry import (
    camera_rays,
    color_depth_m,
    deproject_aligned,
    deproject_pixels,
    optics_drift,
    paired_alignment,
    require_calibrated_model,
)

# Sixteen pixels deprojected at unit depth by the capture owner's pinned SDK
# (`rs2_deproject_pixel_to_point`, pyrealsense2 2.58.4), one record per
# Brown-Conrady model. The SDK computes in float32, so one ulp of a ray
# component below 1 is 6e-8; the numpy port must land inside that.
SDK_TABLE = json.loads((Path(__file__).parent / 'fixtures' / 'deprojection-sdk-table.json').read_text())
SDK_TABLE_TOLERANCE = 2e-7

# The installed overhead D555 at 640x360, as its owner reported it in one
# process (unchanged capture_epoch) on 2026-09-16: the first model at 16:14,
# the second 77 minutes later. Principal point and lens coefficients did not
# move; the focal lengths did. The bundle written from the second capture binds
# exactly these values.
D555_COEFFICIENTS = [-0.052696820348501205, 0.05695461854338646, -0.000833850004710257,
                     -8.723000064492226e-05, -0.017889080569148064]
D555_WARM_EARLY = {'fx': 323.391845703125, 'fy': 322.9006652832031}
D555_WARM_LATE = {'fx': 323.2164306640625, 'fy': 322.72552490234375}
D555_COLD = {'fx': 321.98846, 'fy': 321.49942}  # the same camera, cold, on 2026-09-14


def d555_live(**focal):
    return {'schema': 'tatbot.camera-intrinsics/1', 'width': 640, 'height': 360,
            'ppx': 323.3935546875, 'ppy': 181.3507843017578, 'distortion_model': 'BrownConrady',
            'distortion_coefficients': list(D555_COEFFICIENTS), **(focal or D555_WARM_LATE)}


def d555_bound(**focal):
    return {'intrinsics': {'width': 640, 'height': 360, 'cx': 323.3935546875, 'cy': 181.3507843017578,
                           **(focal or D555_WARM_EARLY)},
            'distortion': {'model': 'brown_conrady', 'coefficients': list(D555_COEFFICIENTS)},
            'metadata': {'device_serial': 'overhead-d555'}}


def alignment(intrinsics, rotation=None, translation=None):
    rotation = np.eye(3) if rotation is None else rotation
    translation = np.zeros(3) if translation is None else translation
    return {'schema': 'tatbot.realsense-native-alignment/1',
            'transform': 'color_from_native_depth', 'rotation_layout': 'column_major',
            'translation_units': 'metres', 'aligned_depth_value_axis': 'native_depth_z',
            'color': copy.deepcopy(intrinsics), 'depth': copy.deepcopy(intrinsics),
            'color_id': 1, 'depth_id': 2, 'color_fps': 30, 'depth_fps': 30,
            'rotation': np.asarray(rotation).ravel(order='F').tolist(),
            'translation': np.asarray(translation).tolist()}


def metric_fixture(width=640, height=480):
    intr = {'schema': 'tatbot.camera-intrinsics/1', 'width': width, 'height': height,
            'fx': 450., 'fy': 450., 'ppx': (width-1)/2, 'ppy': (height-1)/2,
            'distortion_model': 'None', 'distortion_coefficients': [0.]*5}
    rotation = cv2.Rodrigues(np.array([.07, -.09, .02]))[0]
    translation = np.array([-.06, .001, .004])
    model = alignment(intr, rotation, translation)
    y, x = np.indices((height, width))
    # A known sloped plane in COLOR coordinates, independently intersected.
    rays = np.stack(((x-intr['ppx'])/450, (y-intr['ppy'])/450, np.ones_like(x)), axis=-1)
    truth = rays * (.35/(1-.12*rays[:, :, 0]+.08*rays[:, :, 1]))[:, :, None]
    native = (truth-translation) @ rotation
    raw = np.rint(native[:, :, 2]/.0001).astype(np.uint16)
    return intr, model, raw, truth


@pytest.mark.parametrize('model', ['BrownConrady', 'BrownConradyInverse'])
def test_camera_rays_match_the_sdk_table_for_both_brown_conrady_models(model):
    assert SDK_TABLE['sdk'] == 'pyrealsense2==2.58.4.10922' and len(SDK_TABLE['pixels']) == 16
    record = SDK_TABLE['models'][model]
    intr, expected = record['intrinsics'], np.asarray(record['rays'])
    assert intr['distortion_model'] == model and np.any(intr['distortion_coefficients'])
    pixels = np.asarray(SDK_TABLE['pixels'])
    np.testing.assert_allclose(deproject_pixels(intr, pixels), expected, atol=SDK_TABLE_TOLERANCE, rtol=0)
    rays = camera_rays(json.dumps(intr, sort_keys=True))
    assert rays.shape == (intr['height'], intr['width'], 3) and not rays.flags.writeable
    np.testing.assert_allclose(rays[pixels[:, 1], pixels[:, 0]], expected, atol=SDK_TABLE_TOLERANCE, rtol=0)
    # The two models are distinct conventions, not one formula under two
    # names: swapping the model on the same coefficients moves the rays.
    other = dict(intr, distortion_model=[m for m in SDK_TABLE['models'] if m != model][0])
    assert np.abs(deproject_pixels(other, pixels)-expected).max() > 1e-6
    # Undistortion is the inverse of the lens: the pinhole projection of a
    # deprojected ray does not land on its pixel until the lens is re-applied.
    pinhole = np.column_stack(((pixels[:, 0]-intr['ppx'])/intr['fx'], (pixels[:, 1]-intr['ppy'])/intr['fy']))
    assert np.abs(expected[:, :2]-pinhole).max() > 1e-3
    for unsupported in ('BrownConradyModified', 'KannalaBrandt4'):
        with pytest.raises(ValueError, match='unsupported deprojection'):
            deproject_pixels(dict(intr, distortion_model=unsupported), pixels)
    with pytest.raises(ValueError, match='unsupported deprojection'):
        deproject_pixels(dict(intr, distortion_model='None'), pixels)


def test_alignment_recovers_known_color_plane_with_native_axis_rotation_and_translation():
    intr, model, raw, truth = metric_fixture()
    result = deproject_aligned(raw, .0001, intr, model, (.03, 2.))
    np.testing.assert_allclose(result, truth.reshape(-1, 3), atol=.000065, rtol=0)
    rays = camera_rays(json.dumps(intr, sort_keys=True))
    wrong = rays*raw[:, :, None]*.0001
    assert np.max(np.linalg.norm(wrong-truth, axis=2)) > .02
    # Every reconstructed point must also satisfy the measured native Z.
    rotation = np.asarray(model['rotation']).reshape(3, 3, order='F')
    native = (result-model['translation']) @ rotation
    np.testing.assert_allclose(native[:, 2], raw.ravel()*.0001, atol=1e-12)


def test_alignment_does_not_create_measurements_from_holes_or_invalid_samples():
    intr, model, raw, _ = metric_fixture(32, 32)
    raw[0, :3] = [0, 65535, 1]
    points = deproject_aligned(raw, .0001, intr, model, (.03, 2.))
    assert len(points) == raw.size-3
    rays = camera_rays(json.dumps(intr, sort_keys=True))
    depth = raw.astype(float)*.0001
    depth[1, :3] = [np.nan, np.inf, -1]
    corrected = color_depth_m(depth, rays, model, intr)
    assert corrected[0, 0] == 0 and np.all(corrected[1, :3] == 0)


@pytest.mark.parametrize('change', ['missing', 'rotation_order', 'axis', 'reflection', 'color', 'pair'])
def test_missing_or_inconsistent_alignment_is_refused(change):
    intr, model, _, _ = metric_fixture(32, 32)
    changed = copy.deepcopy(model)
    if change == 'missing':
        changed = None
    if change == 'rotation_order':
        changed['rotation_layout'] = 'row_major'
    if change == 'axis':
        changed['aligned_depth_value_axis'] = 'color_z'
    if change == 'reflection':
        changed['rotation'] = np.diag([-1., 1., 1.]).ravel().tolist()
    if change == 'color':
        changed['color']['fx'] += 1
    first = model if change == 'pair' else changed
    if change == 'pair':
        changed['translation'][0] += .001
    with pytest.raises(ValueError):
        paired_alignment({'alignment_calibration': first}, {'alignment_calibration': changed}, intr)


def test_bound_optics_report_thermal_drift_without_refusing_the_camera():
    live, camera = d555_live(), d555_bound()
    require_calibrated_model(live, camera, 'overhead-d555')
    drift = optics_drift(live, camera['intrinsics'])
    assert 5e-4 < drift < 6e-4
    corners = np.array([[0, 0], [639, 0], [0, 359], [639, 359]], float)
    def rays(fx, fy, ppx, ppy):
        return np.column_stack(((corners[:, 0]-ppx)/fx, (corners[:, 1]-ppy)/fy, np.ones(4)))
    moved = rays(live['fx'], live['fy'], live['ppx'], live['ppy']) - rays(
        camera['intrinsics']['fx'], camera['intrinsics']['fy'], 323.3935546875, 181.3507843017578)
    assert np.max(np.linalg.norm(moved, axis=1)) < 1e-3
    # Bundles written before the device was recorded bind any serial.
    require_calibrated_model(live, dict(camera, metadata={}), 'another-unit')
    # The two warm models bind each other in either direction.
    require_calibrated_model(d555_live(**D555_WARM_EARLY), d555_bound(**D555_WARM_LATE), 'overhead-d555')
    require_calibrated_model(d555_live(**D555_COLD), camera, 'overhead-d555')


@pytest.mark.parametrize('change', ['dimensions', 'lens_model',
                                    'coefficients', 'coefficient_count', 'device', 'nonfinite'])
def test_bound_optics_refuse_genuine_mismatches(change):
    live, camera, serial = d555_live(), d555_bound(), 'overhead-d555'
    expected = 'optics differ'
    if change == 'dimensions':
        live.update(width=1280, height=720)
    if change == 'lens_model':
        live['distortion_model'], expected = 'BrownConradyInverse', 'distortion model differs'
    if change == 'coefficients':
        live['distortion_coefficients'][0] += 1e-3
    if change == 'coefficient_count':
        live['distortion_coefficients'] = live['distortion_coefficients'][:4]
    if change == 'device':
        serial, expected = 'another-unit', 'device differs'
    if change == 'nonfinite':
        live['fx'] = float('nan')
    with pytest.raises(ValueError, match=expected):
        require_calibrated_model(live, camera, serial)


def test_tracker_and_both_scan_paths_measure_the_same_color_frame_points():
    import capture_geometry
    from stencil_surface import rgbd_frame
    from test_stencil_surface import geometry_fixture
    intr, model, raw, truth = metric_fixture()
    _, frame = geometry_fixture()
    frame.update(image=np.zeros((*raw.shape, 3), np.uint8), depth_m=raw.astype(float)*.0001)
    for meta in (frame['color_metadata'], frame['depth_metadata']):
        meta['profile'].update(width=640, height=480)
        meta['attributes'].update(intrinsics=json.dumps(intr), alignment_calibration=json.dumps(model))
    rgbd, warnings = rgbd_frame(frame)
    assert not warnings
    tracked = (rgbd['rays']*rgbd['depth_m'][:, :, None]).reshape(-1, 3)
    owner = capture_geometry._deproject_owner(raw, .0001, json.dumps(
        {'intrinsics': intr, 'units_m': .0001, 'alignment_calibration': model}))
    saved = {'intrinsics': {'width': 640, 'height': 480, 'fx': 450., 'fy': 450., 'cx': 319.5, 'cy': 239.5},
             'distortion': {'model': 'none', 'coefficients': []}}
    overhead = capture_geometry._deproject_overhead(raw, .0001, saved,
        frame['color_metadata'], frame['depth_metadata'])
    np.testing.assert_array_equal(tracked, owner)
    np.testing.assert_array_equal(overhead, owner)
    np.testing.assert_allclose(owner, truth.reshape(-1, 3), atol=.000065, rtol=0)
    # Rays use the owner's active intrinsics even when the bound model drifts.
    saved['intrinsics']['fx'] += .2
    np.testing.assert_array_equal(capture_geometry._deproject_overhead(
        raw, .0001, saved, frame['color_metadata'], frame['depth_metadata']), owner)
    saved['intrinsics']['fy'] += 10
    np.testing.assert_array_equal(capture_geometry._deproject_overhead(
        raw, .0001, saved, frame['color_metadata'], frame['depth_metadata']), owner)
