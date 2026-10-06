"""Drawing-local identity evidence at fixed measured points, without hardware."""

import copy

import numpy as np
import pytest
from surface_attachment_inputs import pose, render_fixture
from surface_material_support import MaterialSupport, OriginalAppearance, SupportProfile

# Fixed before observing any current frame: two outer points and two inner ones.
# Optical blur models a lower-frequency scene; no estimator threshold is changed.
PIXELS = np.array([[80, 80], [240, 160], [140, 100], [180, 145]])


def fixture(shape="plane", **kwargs):
    return render_fixture(shape=shape, blur_sigma=1.2, **kwargs)[0]


def points(frame):
    rgbd = frame["rgbd"]
    x, y = PIXELS.T
    return rgbd["rays"][y, x] * rgbd["depth_m"][y, x, None]


def shifted(shape):
    return fixture(shape, world_from_material=pose(xyz=(0.008, 0, 0.32)))


def translation():
    matrix = np.eye(4)
    matrix[0, 3] = 0.008
    return matrix


@pytest.mark.parametrize("shape", ["plane", "curved"])
def test_fixed_interior_material_request_survives_whole_rigid_motion(shape):
    reference = fixture(shape)
    request = MaterialSupport(reference["rgbd"], points(reference), "a" * 64)
    original = request.evaluate(reference["rgbd"], np.eye(4))
    current = request.evaluate(shifted(shape)["rgbd"], translation())
    assert original["accepted"] and current["accepted"], current
    assert current["supported_count"] == 4
    assert current["request_sha256"] == original["request_sha256"] == request.identity
    assert [p["id"] for p in current["points"]] == [0, 1, 2, 3]
    assert max(p["measured_residual_m"] for p in current["points"]) < 0.001
    assert current["motion_authority"] is False


@pytest.mark.parametrize("shape", ["plane", "curved"])
@pytest.mark.parametrize("angles", [(0, 0, 15), (0, 0, 25), (8, -12, 25)])
def test_measured_material_chart_preserves_identity_under_rigid_view_change(shape, angles):
    reference, before = render_fixture(shape=shape, blur_sigma=1.2)
    current, after = render_fixture(shape=shape, blur_sigma=1.2,
        world_from_material=pose(xyz=(.008, 0, .32), angles_deg=angles))
    request = MaterialSupport(reference["rgbd"], points(reference), "a"*64)
    original = request.evaluate(reference["rgbd"], np.eye(4))
    matrix = after['camera_from_material'] @ np.linalg.inv(before['camera_from_material'])
    result = request.evaluate(current['rgbd'], matrix)
    assert result['accepted'], result
    assert result['supported_count'] == 4
    assert result['request_sha256'] == original['request_sha256'] == request.identity
    assert max(point['measured_residual_m'] for point in result['points']) < .0015
    assert result['motion_authority'] is False


@pytest.mark.parametrize("shape", ["plane", "curved"])
def test_rotated_replacement_cannot_gain_original_material_from_pose(shape):
    reference, before = render_fixture(shape=shape, blur_sigma=1.2)
    replacement, after = render_fixture(shape=shape, appearance='alternate', blur_sigma=1.2,
        world_from_material=pose(xyz=(.008, 0, .32), angles_deg=(0, 0, 25)))
    request = MaterialSupport(reference['rgbd'], points(reference), 'a'*64)
    matrix = after['camera_from_material'] @ np.linalg.inv(before['camera_from_material'])
    result = request.evaluate(replacement['rgbd'], matrix)
    assert not result['accepted'] and result['supported_count'] == 0
    assert result['request_sha256'] == request.identity


@pytest.mark.parametrize("shape", ["plane", "curved"])
def test_independent_component_cannot_hide_stationary_requested_material(shape):
    reference = fixture(shape)
    request = MaterialSupport(reference["rgbd"], points(reference), "a" * 64)
    current, moving = copy.deepcopy(reference), shifted(shape)
    # Explicit two-region RGB-D compositing fault. These fixed probes avoid seams;
    # no region labels or synthetic truth enter the material support API.
    for key in ("gray", "depth_m"):
        current["rgbd"][key][60:190, 110:215] = moving["rgbd"][key][60:190, 110:215]
    result = request.evaluate(current["rgbd"], translation())
    assert not result["accepted"]
    assert result["reason"] == "original_material_appearance_contradiction"
    assert [p["status"] for p in result["points"][:2]] == ["contradictory"] * 2
    assert all(p["measured_residual_m"] > 0.007 for p in result["points"][:2])
    assert result["point_count"] == 4  # Contradictions cannot be dropped to pass.


@pytest.mark.parametrize("shape", ["plane", "curved"])
@pytest.mark.parametrize("appearance", ["smooth", "alternate"])
def test_unchanged_depth_does_not_supply_missing_original_appearance(shape, appearance):
    reference = fixture(shape)
    request = MaterialSupport(reference["rgbd"], points(reference), "a" * 64)
    replacement = fixture(shape, appearance=appearance)
    np.testing.assert_array_equal(reference["rgbd"]["depth_m"], replacement["rgbd"]["depth_m"])
    result = request.evaluate(replacement["rgbd"], np.eye(4))
    assert not result["accepted"]
    assert result["supported_count"] == 0
    assert result["contradictory_count"] == 0
    assert result["request_sha256"] == request.identity
    recovered = request.evaluate(reference["rgbd"], np.eye(4))
    assert recovered["accepted"]
    assert recovered["request_sha256"] == result["request_sha256"]


def test_request_copies_original_points_and_appearance_and_binds_context():
    reference = fixture()
    untouched = copy.deepcopy(reference)
    coordinates = points(reference)
    request = MaterialSupport(reference["rgbd"], coordinates, "a" * 64)
    identity = request.identity
    coordinates[:] += 1
    reference["rgbd"]["gray"][:] = 0
    reference["rgbd"]["depth_m"][:] = np.nan
    reference["rgbd"]["material_roi"] = (0, 0, 1, 1)
    assert request.evaluate(untouched["rgbd"], np.eye(4))["accepted"]
    assert request.identity == identity
    different_reference = MaterialSupport(untouched["rgbd"], points(untouched), "b" * 64)
    different_profile = MaterialSupport(untouched["rgbd"], points(untouched), "a" * 64,
                                        profile=SupportProfile(min_correlation=0.9))
    assert len({identity, different_reference.identity, different_profile.identity}) == 3
    untouched["rgbd"]["rays"][0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="camera context changed"):
        request.evaluate(untouched["rgbd"], np.eye(4))


@pytest.mark.parametrize("matrix", [np.eye(3), np.zeros((4, 4)), np.diag([2, 1, 1, 1]),
                                    np.full((4, 4), np.nan), np.diag([-1, 1, 1, 1])])
def test_nonrigid_or_invalid_candidate_cannot_authorize_material(matrix):
    reference = fixture()
    request = MaterialSupport(reference["rgbd"], points(reference), "a" * 64)
    with pytest.raises(ValueError, match="finite rigid pose"):
        request.evaluate(reference["rgbd"], matrix)


@pytest.mark.parametrize("coordinates", [[], [[0, 0]], [[0, 0, np.nan]], [[0, 0, 1]] * 4097])
def test_request_requires_bounded_finite_metric_points(coordinates):
    with pytest.raises(ValueError, match="bounded original-material support"):
        MaterialSupport(fixture()["rgbd"], coordinates, "a" * 64)


def test_original_unmeasured_point_is_not_a_material_reference():
    reference = fixture()
    coordinates = points(reference)
    coordinates[:, 2] += 0.02
    with pytest.raises(ValueError, match="original measured support missing"):
        MaterialSupport(reference["rgbd"], coordinates, "a" * 64)


def test_ineligible_original_context_does_not_consume_current_searches(monkeypatch):
    reference = fixture()
    reference["rgbd"]["material_roi"] = (110, 60, 105, 130)
    request = MaterialSupport(reference["rgbd"], points(reference), "a" * 64)
    calls = []
    match = OriginalAppearance._match

    def measured(instance, gray, pixels, *args):
        calls.append(len(pixels))
        return match(instance, gray, pixels, *args)

    monkeypatch.setattr(OriginalAppearance, "_match", measured)
    result = request.evaluate(reference["rgbd"], np.eye(4))
    assert calls == [2]  # Only the two fixed interior requests can gain support.
    assert result["point_count"] == 4 and not result["accepted"]
    assert [point["id"] for point in result["points"] if point["supported"]] == [2, 3]
    assert [point["status"] for point in result["points"][:2]] == ["unknown", "unknown"]


def test_originally_unobservable_material_never_starts_current_search(monkeypatch):
    reference = fixture(appearance="smooth")
    request = MaterialSupport(reference["rgbd"], points(reference), "a" * 64)

    def forbidden(*args):
        raise AssertionError("an unobservable original cannot acquire identity from a new frame")

    monkeypatch.setattr(OriginalAppearance, "_match", forbidden)
    result = request.evaluate(fixture()["rgbd"], np.eye(4))
    assert not result["accepted"] and result["supported_count"] == 0
    assert result["contradictory_count"] == 0 and result["point_count"] == 4


@pytest.mark.parametrize("background", ["distinctive", "alternate", "noise"])
def test_normalization_cannot_borrow_texture_outside_blank_material(background):
    frame, _ = render_fixture(appearance="alternate" if background == "alternate" else "distinctive")
    rgbd = frame["rgbd"]
    if background == "noise":
        rgbd["gray"][:] = np.random.default_rng(517).integers(0, 256, rgbd["gray"].shape, dtype=np.uint8)
    rgbd["gray"][35:205, 100:240] = 128
    rgbd["depth_m"][:] = 0.5
    rgbd["depth_m"][35:205, 100:240] = 0.32
    rgbd["material_roi"] = (100, 35, 140, 170)
    pixels = np.array([[111, 90], [111, 100], [111, 110], [111, 120], [112, 100]])
    for x, y in pixels:
        assert rgbd["gray"][y-10:y+11, x-10:x+11].std() == 0
    measured = rgbd["rays"][pixels[:, 1], pixels[:, 0]] * 0.32
    request = MaterialSupport(rgbd, measured, "a" * 64)
    result = request.evaluate(rgbd, np.eye(4))
    assert not result["accepted"] and result["supported_count"] == 0
    assert result["point_count"] == len(pixels)


def test_blank_foreground_cannot_borrow_texture_from_disconnected_background():
    reference = fixture()
    rgbd = reference["rgbd"]
    x, y = 140, 100
    # The queried material is a blank 5x5 foreground island. All surrounding
    # pixels have valid depth but belong to a textured surface 180 mm behind it.
    rgbd["depth_m"][:] = 0.5
    rgbd["depth_m"][y-2:y+3, x-2:x+3] = 0.32
    rgbd["gray"][y-2:y+3, x-2:x+3] = 105
    coordinates = rgbd["rays"][y, x][None, :] * 0.32
    request = MaterialSupport(rgbd, coordinates, "a" * 64)
    result = request.evaluate(rgbd, np.eye(4))
    assert not result["accepted"]
    assert result["supported_count"] == 0
    assert result["points"][0]["status"] == "unknown"


@pytest.mark.parametrize("field,value", [("reference_digest", "b" * 64),
                                          ("residual_m", 0.02), ("identity", "c" * 64)])
def test_request_binding_and_threshold_cannot_change_after_identity(field, value):
    reference = fixture()
    request = MaterialSupport(reference["rgbd"], points(reference), "a" * 64)
    original = request.evaluate(reference["rgbd"], np.eye(4))
    with pytest.raises(AttributeError):
        setattr(request, field, value)
    current = request.evaluate(reference["rgbd"], np.eye(4))
    assert current["request_sha256"] == original["request_sha256"]
    assert current["reference_digest"] == original["reference_digest"]
    assert current["accepted"]


def test_global_fit_cannot_hide_component_motion_with_detector_dropout(monkeypatch):
    from surface_attachment import SurfaceAttachmentObserver

    reference = render_fixture()[0]
    instance = SurfaceAttachmentObserver("material", "reference", (0, 0, 320, 240))
    original = instance.observe(reference, now_ns=reference["stamp"])
    assert original["accepted"]
    current = render_fixture(sequence=1, stamp=1_100_000_000)[0]
    moving = render_fixture(world_from_material=pose(xyz=(0.008, 0, 0.32)))[0]
    for key in ("gray", "depth_m"):
        current["rgbd"][key][35:220, 65:255] = moving["rgbd"][key][35:220, 65:255]
    extract = instance.matcher._scene_features

    def current_component_features(gray):
        keys, descriptors = extract(gray)
        xy = np.array([key.pt for key in keys])
        keep = (xy[:, 0] > 85) & (xy[:, 0] < 235) & (xy[:, 1] > 55) & (xy[:, 1] < 200)
        return [key for key, selected in zip(keys, keep, strict=True) if selected], descriptors[keep]

    monkeypatch.setattr(instance.matcher, "_scene_features", current_component_features)
    result = instance.observe(current, now_ns=current["stamp"])
    assert not result["accepted"]
    # Retained original points outside the detected moving component now expose
    # the contradiction during the strict fit, before drawing support is checked.
    assert result["reason"] == "deformation_or_inconsistent_correspondence"
    diagnostics = result["diagnostics"]
    assert diagnostics["recovered_original_matches"] > 0
    assert diagnostics["appearance_verified_matches"] > diagnostics["distinct_matches"]
    assert diagnostics["residual_max_m"] > 2 * instance.settings.residual_m
    assert result["reference_digest"] == original["reference_digest"]
    assert result["transform_reference_to_camera"] is None
    assert result["motion_authority"] is False


@pytest.mark.parametrize("fault", ["missing_depth", "depth_strings", "gray_float",
                                    "bad_depth_range", "list_rays", "nonfinite_rays", "nonunit_rays", "tiny_image"])
@pytest.mark.parametrize("stage", ["reference", "current"])
def test_malformed_rgbd_has_a_public_value_error(fault, stage):
    reference = fixture()
    request = MaterialSupport(reference["rgbd"], points(reference), "a" * 64)
    malformed = copy.deepcopy(reference["rgbd"])
    if fault == "missing_depth":
        del malformed["depth_m"]
    elif fault == "depth_strings":
        malformed["depth_m"] = malformed["depth_m"].astype(str)
    elif fault == "gray_float":
        malformed["gray"] = malformed["gray"].astype(float)
    elif fault == "bad_depth_range":
        malformed["depth_range"] = (0.5, 0.1)
    elif fault == "list_rays":
        malformed["rays"] = malformed["rays"].tolist()
    elif fault == "nonfinite_rays":
        malformed["rays"][0, 0, 0] = np.nan
    elif fault == "nonunit_rays":
        malformed["rays"][..., 2] = 2
    else:
        for key in ("gray", "depth_m", "rays"):
            malformed[key] = malformed[key][:1, :1]
    with pytest.raises(ValueError):
        if stage == "reference":
            MaterialSupport(malformed, points(reference), "a" * 64)
        else:
            request.evaluate(malformed, np.eye(4))


@pytest.mark.parametrize("shape", ["plane", "curved"])
def test_scoring_retains_material_with_off_center_depth_holes(shape):
    from surface_attachment import measured_xyz

    reference = render_fixture(shape=shape)[0]["rgbd"]
    pixels = np.array([[100., 80.], [160., 120.], [220., 160.]])
    coordinates, valid = measured_xyz(reference, pixels, 0.003)
    assert valid.all()
    request = MaterialSupport(reference, coordinates, "a" * 64)
    current = copy.deepcopy(reference)
    for x, y in pixels.astype(int):
        current["depth_m"][y+5:y+7, x+5:x+7] = np.nan
    result = request.evaluate(current, np.eye(4))
    assert result["accepted"] and result["supported_count"] == 3
    assert all(0.9 <= point["scoring_coverage"] < 1 for point in result["points"])


@pytest.mark.parametrize("step,fraction", [(0.08, 1/32), (0.04, 2/32), (0.08, 0.01)])
def test_subpixel_scoring_excludes_incompatible_depth_corners(step, fraction):
    from surface_material_support import _scoring_patches

    rgbd = render_fixture(appearance="alternate")[0]["rgbd"]
    rgbd["depth_m"][:] = 0.32 + step
    rgbd["depth_m"][:, 100:] = 0.32
    rgbd["gray"][:, :100] = (np.arange(rgbd["gray"].shape[0]) % 2 * 255)[:, None]
    rgbd["gray"][:, 100:] = 128
    pixels = np.array([[110-fraction, 100.]])
    # Mixed depth alone looks compatible, but the raw contributing surfaces do not.
    assert step*fraction < 0.003
    image, mask = _scoring_patches(rgbd, pixels, 21, 0.0015)
    assert not mask.reshape(21, 21)[:, 0].any()
    assert mask.mean() >= 0.9
    assert image[mask].std() == 0
    if step == 0.04:
        assert image.std() > SupportProfile().min_texture_std


def test_unscorable_competitor_cannot_create_unique_material_match(monkeypatch):
    reference = fixture()["rgbd"]
    pixel = np.array([[140., 100.]])
    coordinates = reference["rays"][100, 140][None, :] * reference["depth_m"][100, 140]
    request = MaterialSupport(reference, coordinates, "a" * 64)
    assert request.evaluate(reference, np.eye(4))["accepted"]
    current = copy.deepcopy(reference)
    current["gray"][180:221, 230:271] = 128

    def proposals(instance, gray, pixels, starts=None):
        count = 9 if starts is None else starts.shape[1]
        hypotheses = np.repeat(pixel[:, None, :], count, axis=1)
        hypotheses[:, -1] = [250., 200.]
        return hypotheses, np.ones((1, count), bool)

    monkeypatch.setattr(OriginalAppearance, "_hypotheses", proposals)
    result = request.evaluate(current, np.eye(4))
    assert not result["accepted"] and result["supported_count"] == 0
    assert result["points"][0]["unresolved_context"]
    assert result["points"][0]["status"] == "unknown"


@pytest.mark.parametrize("shape", ["plane", "curved"])
@pytest.mark.parametrize("warped", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_valid_candidate_scoring_preserves_evidence_and_skips_rejected_work(monkeypatch, shape, warped, cached):
    import surface_material_support as support
    from surface_appearance_warp import MeasuredAppearanceWarp

    reference, before = render_fixture(shape=shape, blur_sigma=1.2)
    current, after = render_fixture(shape=shape, blur_sigma=1.2,
        world_from_material=pose(xyz=(.008, 0, .32), angles_deg=(0, 0, 15) if warped else (0, 0, 0)))
    pixels = np.array([(x, y) for y in range(70, 171, 25) for x in range(85, 236, 25)], float)
    appearance = OriginalAppearance(reference['rgbd'], pixels, SupportProfile(), .0015)
    if not cached:
        appearance._original_patches = None
    rgbd = current['rgbd']
    rgbd['gray'][100:160, 170:210] = 128
    rgbd['depth_m'][80:90, 120:130] = np.nan
    matrix = after['camera_from_material'] @ np.linalg.inv(before['camera_from_material'])
    warp = MeasuredAppearanceWarp(reference['rgbd'], rgbd, matrix) if warped else None
    hypotheses = pixels[:, None, :] + np.array([[-3, 1], [0, 0], [2, -1], [8, 0], [0, 10]])
    # Cross the 32-point batch boundary, including points with no valid starts.
    valid = np.random.default_rng(23).random(hypotheses.shape[:2]) > .6
    valid[[0, 31, 34]] = False
    expected = appearance._scores(rgbd, hypotheses, warp)
    sampled = []
    original_sampler = support._scoring_positions

    def record_samples(frame, positions, *args, **kwargs):
        if frame is rgbd:
            sampled.append(len(positions))
        return original_sampler(frame, positions, *args, **kwargs)

    monkeypatch.setattr(support, '_scoring_positions', record_samples)
    hypotheses[~valid] = np.nan
    actual = appearance._scores(rgbd, hypotheses, warp, hypothesis_valid=valid)
    assert sum(sampled) == valid.sum()
    for before_scores, after_scores in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(before_scores[valid], after_scores[valid])
    assert (actual[0][~valid] == -1).all()
    assert not actual[1][~valid].any() and not actual[2][~valid].any()
    sampled.clear()
    appearance._scores(rgbd, hypotheses, warp, hypothesis_valid=np.zeros_like(valid))
    assert sampled == []
    with pytest.raises(ValueError, match='validity shape'):
        appearance._scores(rgbd, hypotheses, warp, hypothesis_valid=valid[:-1])


def test_warp_cannot_hide_duplicate_outside_original_material_chart():
    from surface_appearance_warp import MeasuredAppearanceWarp

    reference = render_fixture(blur_sigma=1.2)[0]['rgbd']
    x, y, displacement = 180, 145, 22
    reference['material_roi'] = (x-15, y-15, 31, 31)
    request = MaterialSupport(reference, reference['rays'][y, x][None, :]
                              * reference['depth_m'][y, x], 'a'*64)
    assert request._appearance.original['unique'][0]
    current = copy.deepcopy(reference)
    current.pop('material_roi')
    current['gray'][y-10:y+11, x+displacement-10:x+displacement+11] = reference['gray'][y-10:y+11, x-10:x+11]
    pixels = np.array([[x, y]], float)
    raw = request._appearance.match(current, pixels, request._original_window)
    warp = MeasuredAppearanceWarp(reference, current, np.eye(4), roi=reference['material_roi'])
    warped = request._appearance.match(current, pixels, request._original_window, warp=warp)
    assert raw['qualified'][0] and not raw['unique'][0]
    assert not warped['unique'][0]
    assert warped['alternative_correlation'][0] > .99
    assert not request.evaluate(current, np.eye(4))['accepted']


@pytest.mark.parametrize("shape", ["plane", "curved"])
def test_cached_and_uncached_original_context_have_identical_evidence(shape):
    reference = fixture(shape)
    reference["rgbd"]["material_roi"] = (110, 60, 105, 130)
    request = MaterialSupport(reference["rgbd"], points(reference), "a" * 64)
    current = shifted(shape)["rgbd"]
    cached = request.evaluate(current, translation())
    assert request._appearance._original_patches is not None
    assert all(not array.flags.writeable for array in request._appearance._original_patches)
    request._appearance._original_patches = None
    uncached = request.evaluate(current, translation())
    assert cached == uncached
    assert [point["id"] for point in cached["points"] if point["supported"]] == [2, 3]


def test_large_original_context_uses_bounded_uncached_path(monkeypatch):
    import surface_material_support

    appearance = OriginalAppearance.__new__(OriginalAppearance)
    appearance.profile = SupportProfile(window_px=61)
    appearance.pixels = np.zeros((4096, 2), np.float32)

    def forbidden(*args, **kwargs):
        raise AssertionError("large request must not allocate a full original patch bank")

    monkeypatch.setattr(surface_material_support, "_scoring_patches", forbidden)
    assert appearance._cache_original_patches() is None
