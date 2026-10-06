"""Independent camera projection preserves measured shape, color and missing data."""

import copy
import json

import cv2
import numpy as np
import pytest
from stencil_cross_camera import cross_camera_points, project_into_view, project_samples  # noqa: E402
from stencil_pose import camera_model  # noqa: E402
from test_stencil_surface import geometry_fixture  # noqa: E402


def two_views(curvature=0.):
    observation, frame = geometry_fixture(curvature)
    y, x = np.indices(frame['depth_m'].shape)
    frame['image'][:] = np.stack((x % 256, y, np.full_like(x, 173)), axis=-1)
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.array([0., .06, 0.]))[0]
    transform[:3, 3] = [.015, -.008, .01]
    intr = json.loads(frame['color_metadata']['attributes']['intrinsics'])
    intr['distortion_model'] = 'BrownConrady'
    intr['distortion_coefficients'] = [.05, -.01, 0., 0., 0.]
    matrix = np.array([[450., 0., 159.5], [0., 450., 119.5], [0., 0., 1.]])
    uv = np.asarray([row['reference_uv'] for row in observation['landmarks']])
    x, y = ((uv-.5)*[.1, .15]).T
    xyz = np.column_stack((x, y, .45+curvature*x*x))
    target = xyz @ transform[:3, :3].T+transform[:3, 3]
    pixels = cv2.projectPoints(target, np.zeros(3), np.zeros(3), matrix, np.array(intr['distortion_coefficients']))[0].reshape(-1, 2)
    observation['landmarks'] = [{'reference_uv': a.tolist(), 'image_px': b.tolist()} for a, b in zip(uv, pixels, strict=True)]
    observation['homography_uv_to_image'] = cv2.findHomography(uv, pixels)[0]
    observation['capture_timestamp_ns'] = frame['timestamp_ns']
    view = {'image': np.zeros_like(frame['image']), 'timestamp_ns': frame['timestamp_ns'], 'camera_model': intr}
    return frame, view, observation, {'clear_center_uv': [.2, .2, .8, .8]}, transform


@pytest.mark.parametrize('curvature', [0., 4.])
def test_two_camera_interior_preserves_source_rgb_xyz_holes_and_curvature(curvature):
    frame, view, observation, reference, transform = two_views(curvature)
    frame['depth_m'][121:124, 163:167] = 0
    frame['depth_m'][121:124, 153:157] -= .04
    xyz, rgb, report = cross_camera_points(frame, view, observation, reference, transform, max_skew_ns=40_000_000)
    assert len(xyz) > 2000, report
    pixels = np.rint(xyz[:, :2]/xyz[:, 2, None]*450+[159.5, 119.5]).astype(int)
    x, y = pixels.T
    np.testing.assert_array_equal(rgb, frame['image'][y, x, ::-1])
    np.testing.assert_allclose(xyz[:, 2], frame['depth_m'][y, x])
    assert not np.any((y >= 121) & (y < 124) & (((x >= 153) & (x < 157)) | ((x >= 163) & (x < 167))))
    assert report['measured_anchor_count'] >= 60
    assert not report['motion_authority'] and not report['clock_accuracy_verified']
    if curvature:
        assert np.ptp(xyz[:, 2]) > .002


def test_old_observation_and_incorrect_rgbd_pair_are_not_reused():
    frame, view, row, ref, transform = two_views()
    view['timestamp_ns'] += 50_000_000
    row['capture_timestamp_ns'] = view['timestamp_ns']
    points, _, report = cross_camera_points(frame, view, row, ref, transform, max_skew_ns=40_000_000)
    assert not len(points) and report['reason'] == 'cross_camera_exposure_skew'
    row['capture_timestamp_ns'] -= 1
    with pytest.raises(ValueError, match='observation_timestamp'):
        cross_camera_points(frame, view, row, ref, transform, max_skew_ns=40_000_000)
    view['timestamp_ns'] = row['capture_timestamp_ns'] = frame['timestamp_ns']
    frame['depth_metadata']['sequence'] += 1
    with pytest.raises(ValueError, match='pair_unverified'):
        cross_camera_points(frame, view, row, ref, transform, max_skew_ns=40_000_000)


def test_projection_handles_empty_offscreen_and_rejects_reflections():
    frame, view, _, _, transform = two_views()
    reflected = copy.deepcopy(transform)
    reflected[:3, 0] *= -1
    with pytest.raises(ValueError, match='not_rigid'):
        project_samples(frame, view, reflected)
    transform[:3, 3] = [0, 0, -1]
    assert not len(project_samples(frame, view, transform)[0])
    frame['depth_m'][:] = 0
    assert not len(project_samples(frame, view, np.eye(4))[0])


def test_vectorized_brown_projection_matches_opencv_on_measured_rays():
    frame, view, _, _, transform = two_views()
    view['camera_model']['distortion_coefficients'] = [-.31, .08, .002, -.001, .003]
    rays = np.column_stack((np.linspace(-.3, .3, 1000), np.linspace(.2, -.2, 1000),
                            np.full(1000, .5)))
    measured = rays @ transform[:3, :3].T + transform[:3, 3]
    matrix, distortion = camera_model(view['camera_model'], view['image'].shape)
    expected = cv2.projectPoints(measured, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
    np.testing.assert_allclose(project_into_view(measured, view), expected, atol=1e-9)


def test_depth_projection_region_preserves_the_same_measured_support():
    frame, view, observation, reference, transform = two_views()
    xyz, _, full = cross_camera_points(frame, view, observation, reference, transform,
                                       max_skew_ns=40_000_000)
    x0, y0, x1, y1 = full['depth_pixel_bbox']
    region = (x0-20, y0-20, x1+20, y1+20)
    subset, _, cropped = cross_camera_points(frame, view, observation, reference, transform,
                                              max_skew_ns=40_000_000, depth_roi=region)
    assert cropped['candidate_valid'] and len(subset) == len(xyz)
    np.testing.assert_allclose(subset, xyz)
    assert cropped['measured_anchor_count'] == full['measured_anchor_count']
