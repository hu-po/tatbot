"""Appearance-only rigid charts use measured geometry and preserve contradictions."""

import copy

import numpy as np
import pytest
from surface_appearance_warp import MeasuredAppearanceWarp
from surface_attachment import _tracking_image
from surface_attachment_inputs import pose, render_fixture


@pytest.fixture(scope='module', params=['plane', 'curved'])
def rolled_frames(request):
    original, truth0 = render_fixture(shape=request.param)
    moved, truth1 = render_fixture(shape=request.param,
        world_from_material=pose(xyz=(.01, 0, .32), angles_deg=(0, 0, 25)))
    matrix = truth1['camera_from_material'] @ np.linalg.inv(truth0['camera_from_material'])
    return original['rgbd'], moved['rgbd'], matrix, truth0, truth1


def test_measured_projection_matches_withheld_ray_intersection_truth_after_roll(rolled_frames):
    original, moved, matrix, truth0, truth1 = rolled_frames
    yy, xx = np.mgrid[55:190:15, 65:260:15]
    pixels = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    material = truth0['pixel_material_points'][yy, xx].reshape(-1, 3)
    actual = material @ truth1['camera_from_material'][:3, :3].T + truth1['camera_from_material'][:3, 3]
    expected = actual[:, :2]/actual[:, 2, None]*300 + [(320-1)/2, (240-1)/2]
    projected, valid = MeasuredAppearanceWarp(original, moved, matrix).project(pixels)
    assert valid.all()
    # Fixture sensor quantizes original Z to 0.1 mm; truth is unquantized.
    assert np.max(np.linalg.norm(projected-expected, axis=1)) < .02


def test_warp_compensates_roll_in_actual_rendered_appearance_without_changing_rgbd(rolled_frames):
    original, moved, matrix, _, _ = rolled_frames
    original_before, moved_before = copy.deepcopy(original), copy.deepcopy(moved)
    warp = MeasuredAppearanceWarp(original, moved, matrix)
    image, valid = warp.tracking_image()
    interior = np.zeros(valid.shape, bool)
    interior[65:175, 75:245] = True
    assert valid[interior].all()
    a, b = _tracking_image(original['gray'])[interior], image[interior]
    unaligned = _tracking_image(moved['gray'])[interior]
    # Independently rendered subpixel texture is aliased at this small profile.
    # The chart is a proposal: test substantial correction, not an identity gate.
    assert np.corrcoef(a, b)[0, 1] > np.corrcoef(a, unaligned)[0, 1] + .5
    assert image.shape == original['gray'].shape and image.dtype == np.uint8
    assert valid.dtype == bool
    cached_image, cached_valid = warp.tracking_image()
    assert cached_image is image and cached_valid is valid
    assert not image.flags.writeable and not valid.flags.writeable
    for before, after in ((original_before, original), (moved_before, moved)):
        for key in ('gray', 'depth_m', 'rays'):
            np.testing.assert_array_equal(before[key], after[key])


def test_projection_inverts_current_distorted_calibrated_rays():
    from surface_attachment import _bilinear

    original, _ = render_fixture()
    original = original['rgbd']
    moved = copy.deepcopy(original)
    # Known invertible nonlinear ray field. This is not a pinhole homography.
    xy = moved['rays'][..., :2]
    xy *= 1 + .12*np.sum(xy*xy, axis=2, keepdims=True)
    pixels = np.array([[100.25, 80.4], [220.75, 150.7]])
    reference_rays, _, _, _ = _bilinear(original['rays'], pixels)
    projected, valid = MeasuredAppearanceWarp(original, moved, np.eye(4)).project(pixels)
    inverted, _, _, _ = _bilinear(moved['rays'], projected)
    assert valid.all()
    np.testing.assert_allclose(inverted, reference_rays, atol=2e-6)
    assert np.linalg.norm(projected-pixels, axis=1).min() > .1


@pytest.mark.parametrize('angles', [(0, 0, 25), (15, 25, 25)])
def test_cached_subpixel_projection_error_is_bounded_under_curvature_roll_and_distortion(rolled_frames, angles):
    from surface_attachment import measured_xyz, project_measured_rays

    original, _, _, truth0, _ = rolled_frames
    moved, truth1 = render_fixture(shape=truth0['shape'],
        world_from_material=pose(xyz=(.01, 0, .32), angles_deg=angles))
    matrix = truth1['camera_from_material'] @ np.linalg.inv(truth0['camera_from_material'])
    current = copy.deepcopy(moved['rgbd'])
    xy = current['rays'][..., :2]
    xy *= 1 + .12*np.sum(xy*xy, axis=2, keepdims=True)
    rng = np.random.default_rng(29031)
    pixels = rng.uniform([65., 55.], [250., 180.], (4096, 2))
    measured, measured_valid = measured_xyz(original, pixels, .003)
    exact, exact_valid = project_measured_rays(measured @ matrix[:3, :3].T + matrix[:3, 3], current['rays'])
    interpolated, valid = MeasuredAppearanceWarp(original, current, matrix).project(pixels)
    assert measured_valid.all() and exact_valid.all() and valid.all()
    assert np.max(np.linalg.norm(interpolated-exact, axis=1)) < .01


def test_different_valid_camera_profiles_are_projected_using_current_calibration():
    original, _ = render_fixture()
    moved, _ = render_fixture(width=352, height=256, focal_px=350)
    pixels = np.array([[90.25, 80.25], [210.75, 150.5]])
    projected, valid = MeasuredAppearanceWarp(original['rgbd'], moved['rgbd'], np.eye(4)).project(pixels)
    expected = (pixels-[159.5, 119.5])/300*350 + [175.5, 127.5]
    assert valid.all()
    np.testing.assert_allclose(projected, expected, atol=1e-8)


@pytest.mark.parametrize('damage', ['hole', 'step'])
def test_reference_holes_and_depth_steps_never_supply_a_warp(damage):
    frame, _ = render_fixture()
    original, current = copy.deepcopy(frame['rgbd']), frame['rgbd']
    if damage == 'hole':
        original['depth_m'][120, 160] = 0
    else:
        original['depth_m'][:, 160:] += .03
    pixels = np.array([[160., 120.], [160.25, 120.25], [159.9, 119.9], [175., 120.]])
    warp = MeasuredAppearanceWarp(original, current, np.eye(4))
    projected, valid = warp.project(pixels)
    np.testing.assert_array_equal(valid, [False, False, False, True])
    np.testing.assert_array_equal(projected[~valid], 0)
    _, chart_valid = warp.tracking_image()
    assert not chart_valid[120, 160]
    assert chart_valid[120, 175]


@pytest.mark.parametrize('depth_change_m', [.015, -.12])
def test_current_depth_contradiction_does_not_disappear_from_appearance_proposal(depth_change_m):
    frame, _ = render_fixture()
    original = frame['rgbd']
    current = copy.deepcopy(original)
    current['depth_m'] += depth_change_m
    baseline = MeasuredAppearanceWarp(original, original, np.eye(4))
    contradictory = MeasuredAppearanceWarp(original, current, np.eye(4))
    for baseline_array, current_array in zip(baseline.tracking_image(), contradictory.tracking_image(), strict=True):
        np.testing.assert_array_equal(current_array, baseline_array)
    projected, valid = contradictory.project(np.array([[150.25, 120.25]]))
    assert valid.all()
    np.testing.assert_allclose(projected, [[150.25, 120.25]], atol=1e-8)


def test_current_hole_invalidates_its_contributing_texture_without_fabricating_depth():
    frame, _ = render_fixture()
    original, current = frame['rgbd'], copy.deepcopy(frame['rgbd'])
    current['depth_m'][120, 160] = np.nan
    warp = MeasuredAppearanceWarp(original, current, pose(xyz=(.0004, 0, 0)))
    _, projected_valid = warp.project(np.array([[160., 120.]]))
    image, valid = warp.tracking_image()
    assert projected_valid.all()  # Geometry proposal does not consume current Z.
    assert not valid[120, 160] and image[120, 160] == 128
    assert valid[120, 175]
    assert np.isnan(current['depth_m'][120, 160])


@pytest.mark.parametrize('matrix', [pose(xyz=(10., 0, 0)), pose(xyz=(0, 0, -1.))])
def test_outside_image_and_behind_camera_do_not_become_texture(matrix):
    frame, _ = render_fixture()
    warp = MeasuredAppearanceWarp(frame['rgbd'], frame['rgbd'], matrix)
    projected, valid = warp.project(np.array([[160., 120.], [100.5, 80.5]]))
    assert not valid.any()
    np.testing.assert_array_equal(projected, 0)
    image, chart_valid = warp.tracking_image()
    assert not chart_valid.any()
    assert (image == 128).all()


@pytest.mark.parametrize('angle', [90, 180])
def test_collapsed_and_reversed_surface_projections_are_masked(angle):
    frame, truth = render_fixture()
    matrix = pose(xyz=(0, 0, .32), angles_deg=(0, angle, 0)) @ np.linalg.inv(truth['camera_from_material'])
    warp = MeasuredAppearanceWarp(frame['rgbd'], frame['rgbd'], matrix)
    _, valid = warp.project(np.array([[160., 120.], [100., 80.]]))
    assert not valid.any()
    assert not warp.tracking_image()[1].any()


def test_curved_fold_is_rejected_while_front_facing_patch_remains_available():
    from surface_attachment_inputs import material_height

    frame, truth = render_fixture(shape='curved')
    material_pose = pose(xyz=(0, 0, .32), angles_deg=(0, 80, 0))
    matrix = material_pose @ np.linalg.inv(truth['camera_from_material'])
    yy, xx = np.mgrid[50:200:5, 40:280:5]
    pixels = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    material = truth['pixel_material_points'][yy, xx].reshape(-1, 3)
    _, dx, dy = material_height(material[:, 0], material[:, 1], 'curved')
    normal = np.column_stack([-dx, -dy, np.ones(len(material))]) @ material_pose[:3, :3].T
    points = material @ material_pose[:3, :3].T + material_pose[:3, 3]
    reversed_projection = (normal*points).sum(axis=1) < 0
    assert reversed_projection.sum() > 100
    _, valid = MeasuredAppearanceWarp(frame['rgbd'], frame['rgbd'], matrix).project(pixels)
    assert valid.sum() > 500
    assert not valid[reversed_projection].any()


def test_reference_roi_has_bounded_lk_margin_and_does_not_expand_to_whole_image():
    frame, _ = render_fixture()
    warp = MeasuredAppearanceWarp(frame['rgbd'], frame['rgbd'], np.eye(4), roi=(100, 100, 60, 40))
    _, valid = warp.project(np.array([[80., 80.], [130., 120.], [45., 80.], [220., 120.]]))
    np.testing.assert_array_equal(valid, [True, True, False, False])
    image, mask = warp.tracking_image()
    assert not mask[:52].any() and not mask[:, :52].any()
    assert not mask[188:].any() and not mask[:, 208:].any()
    assert (image[~mask] == 128).all()


@pytest.mark.parametrize('matrix', [np.zeros((3, 3)), np.full((4, 4), np.nan),
    np.diag([-1., 1, 1, 1]), np.diag([2., 1, 1, 1]), np.diag([1., 1, 1, 2])])
def test_invalid_nonrigid_pose_is_refused(matrix):
    frame, _ = render_fixture()
    with pytest.raises(ValueError, match='proper rigid pose'):
        MeasuredAppearanceWarp(frame['rgbd'], frame['rgbd'], matrix)


@pytest.mark.parametrize('side', ['reference', 'current'])
@pytest.mark.parametrize('damage', ['invalid_rays', 'unnormalized_rays', 'folded_rays', 'corner_fold', 'depth_shape'])
def test_each_calibrated_context_is_validated(side, damage):
    frame, _ = render_fixture()
    changed = copy.deepcopy(frame['rgbd'])
    if damage == 'invalid_rays':
        changed['rays'][100, 100, 0] = np.nan
    elif damage == 'unnormalized_rays':
        changed['rays'][..., 2] = 0
    elif damage == 'folded_rays':
        changed['rays'][..., 0] *= -1
    elif damage == 'corner_fold':
        changed['rays'][-1, -1, :2] = changed['rays'][-2, -2, :2]
    else:
        changed['depth_m'] = changed['depth_m'][:-1]
    original, current = (changed, frame['rgbd']) if side == 'reference' else (frame['rgbd'], changed)
    with pytest.raises(ValueError):
        MeasuredAppearanceWarp(original, current, np.eye(4))


def test_nonfinite_and_extreme_queries_are_explicitly_invalid():
    frame, _ = render_fixture()
    warp = MeasuredAppearanceWarp(frame['rgbd'], frame['rgbd'], np.eye(4))
    pixels = np.array([[np.nan, 120], [np.inf, 120], [1e200, -1e200], [-1, 100]])
    with np.errstate(all='raise'):
        projected, valid = warp.project(pixels)
    assert not valid.any()
    np.testing.assert_array_equal(projected, 0)
    assert warp.project(np.empty((0, 2)))[0].shape == (0, 2)


@pytest.mark.parametrize('roi', [(0, 0, 321, 240), (0, 0, 10., 20), (-1, 0, 20, 20)])
def test_invalid_roi_is_refused(roi):
    frame, _ = render_fixture()
    with pytest.raises(ValueError, match='ROI'):
        MeasuredAppearanceWarp(frame['rgbd'], frame['rgbd'], np.eye(4), roi=roi)


@pytest.mark.parametrize('shape', [(8, 1281), (961, 8)])
def test_oversized_camera_context_is_refused(shape):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    rgbd = {'gray': np.zeros(shape, np.uint8), 'depth_m': np.full(shape, .32),
        'depth_range': (.03, 2.), 'rays': np.stack([xx/300, yy/300, np.ones(shape)], axis=-1)}
    with pytest.raises(ValueError, match='bounded camera profile'):
        MeasuredAppearanceWarp(rgbd, rgbd, np.eye(4))


@pytest.mark.parametrize('translation', [(0, 0, 0), (.01, -.005, 0)])
def test_identity_and_small_translation_keep_original_proposals_without_chart_allocation(rolled_frames, translation):
    original, _, _, _, _ = rolled_frames
    warp = MeasuredAppearanceWarp(original, original, pose(xyz=translation))
    pixels = np.array([[100.25, 80.5], [160.75, 120.25], [220.5, 160.75]])
    assert not warp.requires_warp(pixels)
    assert warp._map is None and warp._tracking is None


def test_roll_requires_resampling_for_plane_and_curved_material(rolled_frames):
    original, current, matrix, _, _ = rolled_frames
    warp = MeasuredAppearanceWarp(original, current, matrix)
    assert warp.requires_warp(np.array([[100.25, 80.5], [160.75, 120.25], [220.5, 160.75]]))
    assert warp._map is None and warp._tracking is None


@pytest.mark.parametrize('deviation_px,expected', [(.24, False), (.26, True)])
def test_translation_selector_uses_maximum_footprint_deviation(deviation_px, expected):
    frame, _ = render_fixture()
    # A known perspective scale moves a 21px footprint corner by this amount.
    scale = 1 + deviation_px/np.hypot(10, 10)
    translation_z = .32/scale - .32
    warp = MeasuredAppearanceWarp(frame['rgbd'], frame['rgbd'], pose(xyz=(0, 0, translation_z)))
    assert warp.requires_warp(np.array([[159.5, 119.5]])) is expected


@pytest.mark.parametrize('angles', [(0, 25, 0), (25, 0, 0)])
def test_out_of_plane_tilt_requires_resampling(rolled_frames, angles):
    original, current, _, truth0, _ = rolled_frames
    matrix = pose(xyz=(0, 0, .32), angles_deg=angles) @ np.linalg.inv(truth0['camera_from_material'])
    warp = MeasuredAppearanceWarp(original, current, matrix)
    assert warp.requires_warp(np.array([[100.25, 80.5], [160.75, 120.25], [220.5, 160.75]]))


@pytest.mark.parametrize('invalid', ['missing_depth', 'outside', 'behind', 'nonfinite', 'empty'])
def test_invalid_footprint_support_never_selects_translation_proposal(invalid):
    frame, _ = render_fixture()
    original, current, matrix = copy.deepcopy(frame['rgbd']), frame['rgbd'], np.eye(4)
    pixels = np.array([[160., 120.]])
    if invalid == 'missing_depth':
        original['depth_m'][110, 150] = 0
    elif invalid == 'outside':
        pixels[0] = [5, 5]
    elif invalid == 'behind':
        matrix = pose(xyz=(0, 0, -1))
    elif invalid == 'nonfinite':
        pixels[0, 0] = np.nan
    else:
        pixels = np.empty((0, 2))
    assert MeasuredAppearanceWarp(original, current, matrix).requires_warp(pixels)


def test_current_depth_contradiction_cannot_change_translation_selection():
    frame, _ = render_fixture()
    changed = copy.deepcopy(frame['rgbd'])
    changed['depth_m'] -= .12
    warp = MeasuredAppearanceWarp(frame['rgbd'], changed, np.eye(4))
    assert not warp.requires_warp(np.array([[160., 120.]]))


@pytest.mark.parametrize('window', [8, 10, 63, 21., True])
def test_translation_selector_validates_bounded_window(window):
    frame, _ = render_fixture()
    warp = MeasuredAppearanceWarp(frame['rgbd'], frame['rgbd'], np.eye(4))
    with pytest.raises(ValueError, match='odd window'):
        warp.requires_warp(np.array([[160., 120.]]), window_px=window)


@pytest.mark.parametrize('pixels', [np.zeros(2), np.zeros((2, 3)), np.broadcast_to([160., 120.], (1280*960+1, 2))])
def test_translation_selector_validates_bounded_point_request(pixels):
    frame, _ = render_fixture()
    warp = MeasuredAppearanceWarp(frame['rgbd'], frame['rgbd'], np.eye(4))
    with pytest.raises(ValueError, match='bounded'):
        warp.requires_warp(pixels)
