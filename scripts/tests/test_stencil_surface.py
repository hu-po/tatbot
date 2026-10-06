"""Metric ray-intersection fixtures, surface rejection and independent identities."""

import json

import cv2
import numpy as np
import pytest
from stencil_scene import StencilScene  # noqa: E402
from stencil_surface import camera_rays, estimate_surface  # noqa: E402
from test_rgbd_geometry import alignment
from test_stencil_tracking import reference, render  # noqa: E402

cv2.setNumThreads(1)


def geometry_fixture(curvature=0.):
    height, width, focal, z0 = 240, 320, 450., .45
    intr = {'schema': 'tatbot.camera-intrinsics/1', 'width': width, 'height': height,
            'fx': focal, 'fy': focal, 'ppx': 159.5, 'ppy': 119.5,
            'distortion_model': 'None', 'distortion_coefficients': [0.]*5}
    rays = camera_rays(json.dumps(intr, sort_keys=True))
    depth = 2*z0/(1+np.sqrt(1-4*curvature*rays[:, :, 0]**2*z0))
    v, u = np.mgrid[.1:.91:.1, .1:.91:.1]
    uv = np.column_stack((u.ravel(), v.ravel()))
    x, y = ((uv-.5)*[.1, .15]).T
    xyz = np.column_stack((x, y, z0+curvature*x*x))
    pixels = xyz[:, :2]/xyz[:, 2, None]*focal+[159.5, 119.5]
    homography = np.array([[focal*.1/z0, 0, 159.5-focal*.05/z0],
                           [0, focal*.15/z0, 119.5-focal*.075/z0], [0, 0, 1]])
    attrs = {'frame_number': '1', 'capture_epoch': 'fixture', 'device_serial': 'fixture', 'intrinsics': json.dumps(intr),
             'alignment_calibration': json.dumps(alignment(intr))}
    color = {'sensor_name': 'fixture_color', 'sequence': 1,
             'profile': {'width': width, 'height': height, 'fps_num': 30, 'fps_den': 1},
             'attributes': attrs, 'timestamps': {'normalized_unix_ns': 1_000_000_000,
                 'source_ns': 1_000_000_000, 'source_domain': 'real_sense_hardware'}}
    dm = dict(color, sensor_name='fixture_depth', attributes=dict(attrs, aligned_to='fixture_color', depth_units_m='.0001'))
    frame = {'image': np.zeros((height, width, 3), np.uint8), 'depth_m': depth,
             'color_metadata': color, 'depth_metadata': dm, 'timestamp_ns': 1_000_000_000}
    observation = {'image_tracking_valid': True, 'homography_uv_to_image': homography,
                   'landmarks': [{'reference_uv': a.tolist(), 'image_px': b.tolist()}
                                 for a, b in zip(uv, pixels, strict=True)]}
    return observation, frame


@pytest.mark.parametrize('curvature,model', [(0., 'affine_uv_plane'), (4., 'quadratic_uv_surface')])
def test_plane_and_curve_recover_metric_center_from_measured_rays(curvature, model):
    observation, frame = geometry_fixture(curvature)
    report, mesh = estimate_surface(observation, frame)
    assert report['candidate_valid'], report
    assert report['model'] == model
    pose = np.asarray(report['camera_from_stencil_center'])
    np.testing.assert_allclose(pose[:3, 3], [0, 0, .45], atol=.0003)
    np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=.01)
    assert report['normal_toward_camera'][2] < -.99
    assert len(mesh['vertices_camera_m']) > 100 and len(mesh['triangles']) > 50
    uncertainty = report['uncertainty']
    assert uncertainty['available'] and not uncertainty['calibrated']
    assert np.linalg.eigvalsh(uncertainty['covariance_6x6']).min() > -1e-12
    assert not report['geometry_valid'] and not report['motion_authority']


def test_depth_dropout_does_not_invalidate_image_correspondence():
    observation, frame = geometry_fixture()
    frame['depth_m'][:] = 0
    report, mesh = estimate_surface(observation, frame)
    assert observation['image_tracking_valid']
    assert not report['candidate_valid'] and mesh is None
    assert report['reason'] == 'insufficient_measured_anchor_support'


def test_inconsistent_depth_is_rejected_instead_of_smoothed():
    observation, frame = geometry_fixture()
    frame['depth_m'] += np.random.default_rng(5).normal(0, .02, frame['depth_m'].shape)
    report, mesh = estimate_surface(observation, frame)
    assert not report['candidate_valid'] and mesh is None


def test_mesh_never_bridges_a_depth_hole_between_grid_vertices():
    observation, frame = geometry_fixture()
    full, full_mesh = estimate_surface(observation, frame)
    frame['depth_m'][121:123, 163:165] = 0
    partial, mesh = estimate_surface(observation, frame)
    assert full['candidate_valid'] and partial['candidate_valid']
    assert len(mesh['triangles']) < len(full_mesh['triangles'])
    for triangle in mesh['triangles']:
        pixels = mesh['image_px'][triangle].astype(np.float32)
        assert cv2.pointPolygonTest(pixels, (163.5, 121.5), False) < 0


def test_metric_center_is_withheld_outside_measured_anchor_hull():
    observation, frame = geometry_fixture()
    observation['landmarks'] = [row for row in observation['landmarks'] if row['reference_uv'][0] < .45]
    report, _ = estimate_surface(observation, frame)
    assert report['candidate_valid'] and not report['center_supported']
    assert report['camera_from_stencil_center'] is None
    assert not report['uncertainty']['available']


def test_pair_mismatch_withholds_surface_and_intrinsics_mismatch_is_explicit():
    observation, frame = geometry_fixture()
    frame['depth_metadata']['sequence'] = 2
    report, _ = estimate_surface(observation, frame)
    assert not report['candidate_valid'] and report['reason'] == 'rgbd_capture_pair_unverified'
    frame['depth_metadata']['sequence'] = 1
    altered = json.loads(frame['color_metadata']['attributes']['intrinsics'])
    altered['fx'] += 1
    frame['color_metadata']['attributes']['intrinsics'] = json.dumps(altered)
    report, _ = estimate_surface(observation, frame)
    assert 'aligned_color_depth_intrinsics_disagree' in report['calibration_warnings']
    assert not report['absolute_accuracy_verified']


def test_owner_pair_allows_different_counter_origins_but_rejects_adjacent_exposures():
    observation, frame = geometry_fixture()
    frame['depth_metadata']['attributes']['frame_number'] = '15'
    frame['depth_metadata']['timestamps'] = dict(frame['color_metadata']['timestamps'], source_ns=1_000_304_000)
    assert estimate_surface(observation, frame)[0]['candidate_valid']
    frame['depth_metadata']['timestamps']['source_ns'] = 1_033_369_000
    assert estimate_surface(observation, frame)[0]['reason'] == 'rgbd_adjacent_exposures'
    frame['depth_metadata']['timestamps']['source_ns'] = 1_000_304_000
    frame['depth_metadata']['attributes']['device_serial'] = 'other camera'
    assert estimate_surface(observation, frame)[0]['reason'] == 'rgbd_capture_pair_unverified'


def test_two_patterns_track_and_recover_independently():
    scene = StencilScene([reference('tatbot-42'), reference('tatbot-43'), reference('tatbot-44')], 'session')
    left, _ = render('tatbot-42')
    right, _ = render('tatbot-43')
    first = scene.observe(np.hstack((left, right)), 1_000_000_000)
    assert [row['image_tracking_valid'] for row in first['stencils']] == [True, True, False], first
    identities = [row['instance_id'] for row in first['stencils']]
    assert len(set(identities)) == 3
    hidden = scene.observe(np.hstack((np.full_like(left, 128), right)), 1_033_333_333)
    assert not hidden['stencils'][0]['image_tracking_valid']
    assert hidden['stencils'][1]['status'] == 'tracked'
    recovered = scene.observe(np.hstack((left, right)), 1_400_000_000)
    assert [row['instance_id'] for row in recovered['stencils']] == identities
    assert recovered['stencils'][0]['status'] == 'reacquired'
    assert recovered['stencils'][1]['status'] == 'tracked'
    assert not recovered['stencils'][2]['image_tracking_valid']
    blank = scene.observe(np.full_like(np.hstack((left, right)), 128), 1_800_000_000)
    assert not any(row['image_tracking_valid'] for row in blank['stencils'])


def test_rgb_pose_uses_explicit_nominal_scale_and_distortion():
    from stencil_pose import estimate_print_pose
    u, v = np.meshgrid(np.linspace(.1, .9, 5), np.linspace(.1, .9, 6))
    uv = np.column_stack((u.ravel(), v.ravel()))
    points = np.column_stack(((uv-.5)*[.1, .15], np.zeros(len(uv))))
    matrix = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    distortion = np.array([-.1, .01, 0, 0, 0.])
    rotation = np.array([.3, -.2, .1])
    translation = np.array([.02, -.01, .6])
    pixels = cv2.projectPoints(points, rotation, translation, matrix, distortion)[0].reshape(-1, 2)
    observation = {'landmarks': [{'reference_uv': a, 'image_px': b} for a, b in zip(uv, pixels, strict=True)]}
    model = {'width': 640, 'height': 480, 'fx': 500., 'fy': 500., 'ppx': 320., 'ppy': 240.,
             'distortion_model': 'BrownConrady', 'distortion_coefficients': distortion.tolist()}
    frame = {'image': np.zeros((480, 640, 3), np.uint8), 'camera_model': model}
    ref = {'page_mm': [100., 150.], 'dimensions_measured': False}
    report = estimate_print_pose(observation, frame, ref)
    assert report['candidate_valid'] and not report['print_scale_measured']
    pose = np.asarray(report['camera_from_stencil_center'])
    np.testing.assert_allclose(pose[:3, 3], translation, atol=1e-6)
    np.testing.assert_allclose(pose[:3, :3], cv2.Rodrigues(rotation)[0], atol=1e-6)
    larger = estimate_print_pose(observation, frame, dict(ref, page_mm=[200., 300.]))
    np.testing.assert_allclose(np.asarray(larger['camera_from_stencil_center'])[:3, 3], 2*translation, atol=1e-6)
    assert not report['uncertainty']['calibrated']
    assert 'print-scale error' in report['uncertainty']['excludes']
    with pytest.raises(ValueError, match='dimensions_mismatch'):
        estimate_print_pose(observation, dict(frame, image=frame['image'][:240]), ref)


def test_constant_depth_bias_is_not_claimed_as_calibrated_uncertainty():
    observation, frame = geometry_fixture()
    frame['depth_m'] += .01
    report, _ = estimate_surface(observation, frame)
    assert report['candidate_valid']
    assert np.asarray(report['camera_from_stencil_center'])[2, 3] == pytest.approx(.46, abs=.0001)
    assert report['uncertainty']['systematic_depth_error_m'] is None
    assert not report['absolute_accuracy_verified']


def test_rgbd_timestamp_cannot_be_borrowed_from_another_image():
    observation, frame = geometry_fixture()
    frame['timestamp_ns'] += 1
    report, mesh = estimate_surface(observation, frame)
    assert report['reason'] == 'rgbd_capture_timestamp_mismatch' and mesh is None


def test_camera_rays_follow_distorted_optical_geometry():
    _, frame = geometry_fixture()
    intr = json.loads(frame['depth_metadata']['attributes']['intrinsics'])
    intr.update(distortion_model='BrownConrady', distortion_coefficients=[-.2, .02, .001, -.002, 0.])
    rays = camera_rays(json.dumps(intr, sort_keys=True))
    pixels = np.array([[30., 40.], [290., 210.]])
    matrix = np.array([[intr['fx'], 0, intr['ppx']], [0, intr['fy'], intr['ppy']], [0, 0, 1.]])
    expected = cv2.undistortPoints(pixels[:, None], matrix, np.asarray(intr['distortion_coefficients']))[:, 0]
    np.testing.assert_allclose(rays[pixels[:, 1].astype(int), pixels[:, 0].astype(int), :2], expected, atol=1e-5)


def test_manifest_geometry_inputs_are_identified_and_camera_change_breaks_continuity(tmp_path):
    from stencil_inputs import image_frames
    observation, frame = geometry_fixture()
    cv2.imwrite(str(tmp_path/'image.png'), frame['image'])
    np.save(tmp_path/'depth.npy', frame['depth_m'])
    (tmp_path/'header.json').write_text(json.dumps({'frames': [
        {'metadata': frame['color_metadata']}, {'metadata': frame['depth_metadata']}]}))
    model = json.loads(frame['color_metadata']['attributes']['intrinsics'])
    (tmp_path/'camera.json').write_text(json.dumps(model))
    row = {'image': 'image.png', 'depth_m': 'depth.npy', 'metadata': 'header.json', 'sensor': 'fixture_color',
           'camera_model': 'camera.json', 'timestamp_ns': frame['timestamp_ns']}
    index = tmp_path/'frames.jsonl'
    index.write_text(json.dumps(row)+'\n')
    first = next(image_frames(index))
    assert all(len(first['provenance'][key]['sha256']) == 64 for key in ('depth_m', 'metadata', 'camera_model'))
    assert estimate_surface(observation, first)[0]['candidate_valid']
    model['fx'] += 1
    (tmp_path/'camera.json').write_text(json.dumps(model))
    assert next(image_frames(index))['source_id'] != first['source_id']
