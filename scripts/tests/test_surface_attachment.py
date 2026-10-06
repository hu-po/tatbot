"""Image/depth observer tests; no truth labels are supplied to the estimator."""

import copy

import numpy as np
import pytest
from surface_attachment import AttachmentSettings, SurfaceAttachmentObserver, fit_rigid
from surface_attachment_inputs import pose, render_fixture


def observer():
    return SurfaceAttachmentObserver("material-a", "reference-a", (10, 10, 300, 220))


def sample(instance, sequence=0, **kwargs):
    frame, truth = render_fixture(sequence=sequence, stamp=1_000_000_000 + sequence * 100_000_000, **kwargs)
    return instance.observe(frame, now_ns=frame["stamp"]), frame, truth


@pytest.mark.slow  # 80 sustained frames through the real observer, ~90 s: full tier only.
def test_silhouette_features_do_not_break_sustained_material_attachment():
    from surface_attachment_benchmark import sustained_samples

    instance = SurfaceAttachmentObserver("material-a", "reference-a", (0, 0, 320, 240))
    digest = None
    for frame, _, expected in sustained_samples(80):
        result = instance.observe(frame, now_ns=frame["stamp"])
        assert result["accepted"] == expected, (frame["sequence"], result)
        digest = digest or result["reference_digest"]
        assert result["reference_digest"] == digest


@pytest.mark.parametrize("shape", ["plane", "curved"])
@pytest.mark.parametrize("motion", ["surface", "camera", "both"])
def test_original_material_metric_attachment_under_calibrated_motion(shape, motion):
    instance = observer()
    original, _, original_truth = sample(instance, shape=shape)
    assert original["accepted"]
    kwargs = {}
    if motion in ("surface", "both"):
        kwargs["world_from_material"] = pose(xyz=(0.008, 0.003, 0.32), angles_deg=(2, -2, 3))
    if motion in ("camera", "both"):
        kwargs["world_from_camera"] = pose(xyz=(0.004, -0.002, 0.001), angles_deg=(-1, 1, -2))
    result, _, truth = sample(instance, 1, shape=shape, **kwargs)
    assert result["accepted"], result
    expected = truth["camera_from_material"] @ np.linalg.inv(original_truth["camera_from_material"])
    estimated = np.array(result["transform_reference_to_camera"])
    # Withheld metric probes include points never supplied as feature labels.
    probes = np.asarray(original_truth["material_points"])
    reference_probes = (
        probes @ original_truth["camera_from_material"][:3, :3].T
        + original_truth["camera_from_material"][:3, 3]
    )
    error = np.linalg.norm(
        reference_probes @ (estimated[:3, :3] - expected[:3, :3]).T + estimated[:3, 3] - expected[:3, 3],
        axis=1,
    )
    assert error.max() < 0.0015
    assert result["reference_digest"] == original["reference_digest"]
    assert result["motion_authority"] is False
    assert len(result["geometry_support"]["ids"]) <= 128


@pytest.mark.parametrize("shape", ["plane", "curved"])
@pytest.mark.parametrize("angles", [(0, 0, 25), (5, -7, 15)])
def test_rotated_material_recovers_original_metric_attachment(shape, angles):
    instance = observer()
    first, _, original_truth = sample(instance, shape=shape, blur_sigma=1.2)
    assert first['accepted'], first
    missing, _, _ = sample(instance, 1, shape=shape, occlusion_fraction=1)
    assert not missing['accepted']
    current, _, truth = sample(instance, 2, shape=shape, blur_sigma=1.2,
        world_from_material=pose(xyz=(.008, 0, .32), angles_deg=angles))
    assert current['accepted'] and current['state'] == 'recovered', current
    assert current['reference_digest'] == first['reference_digest']
    expected = truth['camera_from_material'] @ np.linalg.inv(original_truth['camera_from_material'])
    estimated = np.asarray(current['transform_reference_to_camera'])
    original = np.asarray(original_truth['material_points'])
    points = original @ original_truth['camera_from_material'][:3, :3].T + original_truth['camera_from_material'][:3, 3]
    error = np.linalg.norm(points @ (estimated[:3, :3]-expected[:3, :3]).T + estimated[:3, 3]-expected[:3, 3], axis=1)
    assert error.max() < .0015
    assert current['motion_authority'] is False


def test_large_tilt_preserves_unrepaired_original_border_correspondence():
    instance = observer()
    first, _, _ = sample(instance, blur_sigma=1.2)
    moved, _, _ = sample(instance, 1, blur_sigma=1.2,
        world_from_material=pose(xyz=(.008, 0, .32), angles_deg=(8, -12, 25)))
    # The border anchor lacks original NCC context. Its biased legacy match
    # cannot silently vanish merely because most interior anchors now recover.
    assert not moved['accepted']
    assert moved['reason'] == 'deformation_or_inconsistent_correspondence'
    assert moved['diagnostics']['residual_max_m'] > .003
    assert moved['reference_digest'] == first['reference_digest']
    recovered, _, _ = sample(instance, 2, blur_sigma=1.2)
    assert recovered['accepted'] and recovered['state'] == 'recovered'
    assert recovered['reference_digest'] == first['reference_digest']


def test_unverified_geometry_proposal_failure_does_not_veto_original_matches(monkeypatch):
    from surface_attachment import _ObservationRefusedError

    instance = observer()
    assert sample(instance)[0]['accepted']
    candidate = instance._candidate
    calls = 0

    def unavailable_proposal(source, current):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _ObservationRefusedError('rigidity_not_supported')
        return candidate(source, current)

    monkeypatch.setattr(instance, '_candidate', unavailable_proposal)
    result, _, _ = sample(instance, 1)
    assert result['accepted'], result
    assert result['diagnostics']['material_refined_matches'] == 0


def test_folded_calibrated_ray_field_returns_advisory_refusal():
    frame, _ = render_fixture()
    frame['rgbd']['rays'][-1, -1, :2] = frame['rgbd']['rays'][-2, -2, :2]
    result = observer().observe(frame, now_ns=frame['stamp'])
    assert not result['accepted']
    assert result['reason'] == 'invalid_measured_appearance_projection'
    assert result['motion_authority'] is False


@pytest.mark.parametrize('shape', ['plane', 'curved'])
@pytest.mark.parametrize('fault', [None, 'deformation', 'replacement'])
def test_original_appearance_recovers_detector_dropout_without_hiding_change(monkeypatch, shape, fault):
    instance = observer()
    first, _, original_truth = sample(instance, shape=shape, blur_sigma=1.2)
    inventory = instance.reference_points()
    scene_features = instance.matcher._scene_features

    def dropped_detections(gray):
        keys, descriptors = scene_features(gray)
        return keys[::2], descriptors[::2]

    # Remove detections deterministically, leaving actual RGB/depth untouched.
    monkeypatch.setattr(instance.matcher, '_scene_features', dropped_detections)
    kwargs = {'shape': shape, 'blur_sigma': 1.2,
              'world_from_material': pose(xyz=(.008, 0, .32), angles_deg=(0, 0, 25))}
    if fault == 'deformation':
        kwargs['deformation_m'] = .015
    elif fault == 'replacement':
        kwargs['appearance'] = 'alternate'
    result, _, truth = sample(instance, 1, **kwargs)
    assert result['reference_digest'] == first['reference_digest']
    assert instance.reference_points() == inventory
    assert result['motion_authority'] is False
    if fault:
        assert not result['accepted'], result
        if fault == 'deformation':
            assert result['reason'] in ('deformation_or_inconsistent_correspondence',
                                       'deformation_or_inconsistent_geometry', 'original_material_appearance_contradiction')
        return
    assert result['accepted'], result
    assert result['diagnostics']['recovered_original_matches'] > 0
    ids = result['geometry_support']['ids']
    assert len(ids) == len(set(ids)) and all(str(index) in inventory for index in ids)
    expected = truth['camera_from_material'] @ np.linalg.inv(original_truth['camera_from_material'])
    estimate = np.asarray(result['transform_reference_to_camera'])
    probes = np.asarray(original_truth['material_points'])
    points = probes @ original_truth['camera_from_material'][:3, :3].T + original_truth['camera_from_material'][:3, 3]
    errors = np.linalg.norm(points @ (estimate[:3, :3]-expected[:3, :3]).T + estimate[:3, 3]-expected[:3, 3], axis=1)
    assert errors.max() < .0015


@pytest.mark.parametrize('shape', ['plane', 'curved'])
def test_partial_replacement_cannot_support_drawing_at_detector_missing_anchors(monkeypatch, shape):
    from surface_attachment import measured_xyz
    from surface_rigid_support import RigidPatchSupport

    instance = observer()
    first, reference, _ = sample(instance, shape=shape, blur_sigma=1.2)
    kwargs = {'shape': shape, 'blur_sigma': 1.2, 'sequence': 1, 'stamp': 1_100_000_000,
              'world_from_material': pose(xyz=(.008, 0, .32), angles_deg=(0, 0, 25))}
    current, _ = render_fixture(**kwargs)
    alternate, _ = render_fixture(**kwargs, appearance='alternate')
    current['rgbd']['gray'][:, 180:] = alternate['rgbd']['gray'][:, 180:]
    scene_features = instance.matcher._scene_features

    def left_detections(gray):
        keys, descriptors = scene_features(gray)
        take = [index for index, key in enumerate(keys) if key.pt[0] < 160]
        return [keys[index] for index in take], descriptors[take]

    monkeypatch.setattr(instance.matcher, '_scene_features', left_detections)
    result = instance.observe(current, now_ns=current['stamp'])
    # Enough unchanged material remains to propose and recover a global pose.
    # The actual drawing area still cannot borrow that area's recovered anchors.
    assert result['accepted'], result
    assert result['diagnostics']['recovered_original_matches'] > 0
    assert result['reference_digest'] == first['reference_digest']
    points, valid = measured_xyz(reference['rgbd'], np.array([[250., 120.]]), .003)
    assert valid.all()
    support = RigidPatchSupport(instance.reference['rgbd'], points, instance.reference['xyz'],
                                result['reference_digest'], instance.material_support.identity)
    local = support.evaluate(current['rgbd'], np.asarray(result['transform_reference_to_camera']),
                             result['material_support'])
    assert not local['accepted']
    assert local['reason'] == 'drawing_outside_supported_anchor_hull'
    assert local['motion_authority'] is False


@pytest.mark.parametrize("shape", ["plane", "curved"])
def test_occlusion_loss_recovery_retains_original_reference(shape):
    instance = observer()
    first, _, _ = sample(instance, shape=shape)
    lost, _, _ = sample(instance, 1, shape=shape, occlusion_fraction=1)
    assert not lost["accepted"]
    alternate, _, _ = sample(instance, 2, shape=shape, appearance="alternate")
    assert not alternate["accepted"]
    recovered, _, _ = sample(instance, 3, shape=shape)
    assert recovered["accepted"] and recovered["state"] == "recovered"
    assert recovered["reference_digest"] == first["reference_digest"]
    assert recovered["reference_timestamp_ns"] == first["capture_timestamp_ns"]


@pytest.mark.parametrize("shape", ["plane", "curved"])
@pytest.mark.parametrize("appearance", ["smooth", "repeating"])
def test_ambiguous_reference_is_not_identity(shape, appearance):
    result, _, _ = sample(observer(), shape=shape, appearance=appearance)
    assert not result["accepted"]


@pytest.mark.parametrize("shape", ["plane", "curved"])
def test_local_deformation_is_not_hidden_as_ransac_outliers(shape):
    instance = observer()
    assert sample(instance, shape=shape)[0]["accepted"]
    result, _, _ = sample(instance, 1, shape=shape, deformation_m=0.015)
    assert not result["accepted"]
    assert result["reason"] == "deformation_or_inconsistent_correspondence"
    assert sample(instance, 2, shape=shape)[0]["state"] == "recovered"


def test_partial_depth_holes_allow_supported_rigid_attachment():
    instance = observer()
    assert sample(instance)[0]["accepted"]
    result, _, _ = sample(instance, 1, depth_hole_fraction=0.15)
    assert result["accepted"], result
    result, _, _ = sample(instance, 2, depth_hole_fraction=1)
    assert not result["accepted"]
    assert result["reason"] == "insufficient_measured_depth_support"


@pytest.mark.parametrize("change", ["producer_id", "calibration_id", "profile_id", "rays", "evidence_kind"])
def test_context_changes_latch_reference_review(change):
    instance = observer()
    first, frame, _ = sample(instance)
    assert first["accepted"]
    frame = copy.deepcopy(frame)
    frame.update(stamp=1_100_000_000, capture_timestamp_ns=1_100_000_000, sequence=1)
    if change == "rays":
        frame["rgbd"]["rays"][0, 0, 0] += 0.01
    elif change == "evidence_kind":
        frame[change] = "recorded-rgbd"
    else:
        frame[change] = "changed"
    result = instance.observe(frame, now_ns=frame["stamp"])
    assert result["state"] == "invalidated" and not result["accepted"]
    result, _, _ = sample(instance, 2)
    assert result["reason"] == "reference_review_required"


@pytest.mark.parametrize(
    ("offset", "reason"),
    [(1, "future_capture"), (-300_000_000, "stale_capture"), (0, "duplicate_or_reordered_capture")],
)
def test_capture_age_and_reordering_are_not_refreshed(offset, reason):
    instance = observer()
    _, frame, _ = sample(instance)
    result = instance.observe(frame, now_ns=frame["stamp"] - offset)
    assert not result["accepted"] and result["reason"] == reason
    assert result["capture_timestamp_ns"] == frame["stamp"]
    assert result["diagnostics"]["dropped_frames"] == 1


def test_sequence_restart_requires_reference_review():
    instance = observer()
    sample(instance)
    _, frame, _ = sample(observer(), 1)
    frame["sequence"] = 0
    result = instance.observe(frame, now_ns=frame["stamp"])
    assert result["state"] == "invalidated"
    assert result["reason"] == "producer_sequence_regression"


def test_no_clock_no_evidence_no_authority():
    frame, _ = render_fixture()
    assert observer().observe(frame)["reason"] == "missing_capture_clock_support"
    frame["evidence_kind"] = "fabricated-physical"
    assert observer().observe(frame, now_ns=frame["stamp"])["reason"] == "missing_evidence_provenance"


def test_reference_is_bounded_and_immutable_over_long_repeated_input():
    instance = observer()
    first, frame, _ = sample(instance)
    before = instance.reference["xyz"].copy()
    descriptor_bytes = instance.matcher.descriptors.nbytes
    for i in range(1, 61):
        frame.update(
            sequence=i,
            stamp=1_000_000_000 + i * 100_000_000,
            capture_timestamp_ns=1_000_000_000 + i * 100_000_000,
        )
        result = instance.observe(frame, now_ns=frame["stamp"])
        assert result["accepted"]
        assert result["reference_digest"] == first["reference_digest"]
        assert result["diagnostics"]["reference_count"] == 1
    np.testing.assert_array_equal(instance.reference["xyz"], before)
    assert instance.matcher.descriptors.nbytes == descriptor_bytes
    assert len(instance.reference["xyz"]) <= instance.settings.max_points
    assert len(instance.reference_points()) <= 128


def test_proper_rotation_fit_does_not_reflect():
    source = np.random.default_rng(1).normal(size=(30, 3))
    transform = pose(xyz=(0.02, -0.04, 0.01), angles_deg=(5, 10, 30))
    target = source @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(fit_rigid(source, target), transform, atol=1e-12)
    assert np.linalg.det(fit_rigid(source, -source)[:3, :3]) > 0.999


@pytest.mark.parametrize(
    "kwargs",
    [{"max_points": 1000}, {"max_features": 5000}, {"residual_m": float("nan")}, {"ransac_trials": 10000}],
)
def test_workload_settings_are_bounded(kwargs):
    with pytest.raises(ValueError):
        AttachmentSettings(**kwargs)


def test_reference_digest_binds_settings_and_inventory():
    frame, _ = render_fixture()
    first = observer()
    initial = first.observe(frame, now_ns=frame["stamp"])
    smaller = SurfaceAttachmentObserver(
        "material-a", "reference-a", (10, 10, 300, 220), settings=AttachmentSettings(max_points=64)
    )
    changed = smaller.observe(frame, now_ns=frame["stamp"])
    assert initial["accepted"] and changed["accepted"]
    assert initial["reference_digest"] != changed["reference_digest"]
    assert len(smaller.reference_points()) == 64


def test_depth_support_profile_change_requires_review():
    instance = observer()
    sample(instance)
    frame, _ = render_fixture(sequence=1, stamp=1_100_000_000)
    frame["rgbd"]["depth_range"] = (0.02, 0.9)
    result = instance.observe(frame, now_ns=frame["stamp"])
    assert result["state"] == "invalidated"


def test_malformed_rays_are_not_metric_geometry():
    frame, _ = render_fixture()
    frame["rgbd"]["rays"][..., 2] = 0
    result = observer().observe(frame, now_ns=frame["stamp"])
    assert result["reason"] == "invalid_or_oversized_rgbd"
    assert not result["accepted"]


def test_slow_live_processing_does_not_manufacture_freshness(monkeypatch):
    import surface_attachment

    ticks = iter([10.0, 10.3])
    monkeypatch.setattr(surface_attachment.time, "perf_counter", lambda: next(ticks))
    frame, _ = render_fixture()
    frame["evidence_kind"] = "live-rgbd"
    result = observer().observe(frame, now_ns=frame["stamp"])
    assert not result["accepted"] and result["reason"] == "stale_after_processing"
    assert result["capture_timestamp_ns"] == frame["stamp"]
    assert result["transform_reference_to_camera"] is None
    assert result["diagnostics"]["capture_to_result_age_ms"] >= 299


@pytest.mark.parametrize("shape", ["plane", "curved"])
def test_fixed_depth_probes_detect_deformation_without_surviving_features(shape):
    instance = SurfaceAttachmentObserver("material-a", "reference-a", (0, 0, 320, 240))
    results = []
    for sequence, bend in enumerate([0, 0.012]):
        frame, _ = render_fixture(
            shape=shape, deformation_m=bend, sequence=sequence, stamp=1_000_000_000 + sequence * 100_000_000
        )
        # Fixed image rectangle in both frames suppresses local appearance.
        # It does not follow any ground-truth point or disclose motion to fitting.
        frame["rgbd"]["gray"][35:180, 120:270] = 210
        results.append(instance.observe(frame, now_ns=frame["stamp"]))
    assert results[0]["accepted"]
    bent = results[1]
    assert not bent["accepted"]
    assert bent["reason"] == "deformation_or_inconsistent_geometry"
    assert bent["diagnostics"]["residual_max_m"] < 0.0002  # Sparse rigidity alone looks good.
    assert bent["diagnostics"]["geometry_probes"]["inconsistent_count"] >= 3
    assert len(instance.reference["probes"]) <= 256


def test_subpixel_measured_depth_preserves_holes_and_depth_steps():
    from surface_attachment import measured_xyz

    y, x = np.mgrid[:8, :8]
    rays = np.stack([(x - 4) / 100, (y - 4) / 100, np.ones_like(x)], axis=-1)
    frame = {"depth_m": np.full((8, 8), 0.3), "rays": rays, "depth_range": (0.1, 1.0)}
    points = np.array([[3.2, 4.3]])
    xyz, valid = measured_xyz(frame, points, 0.003)
    assert valid.all()
    np.testing.assert_allclose(xyz, [[-0.0024, 0.0009, 0.3]], atol=1e-12)
    frame["depth_m"][4, 3] = 0
    assert not measured_xyz(frame, points, 0.003)[1].any()
    frame["depth_m"][4, 3] = 0.3
    frame["depth_m"][4, 4] = 0.32
    assert not measured_xyz(frame, points, 0.003)[1].any()


def test_projection_inverts_calibrated_nonlinear_ray_field():
    from surface_attachment import _bilinear, project_measured_rays

    y, x = np.mgrid[:120, :160]
    u, v = (x - 79.5) / 150, (y - 59.5) / 150
    # A smooth nonlinear deprojection profile; the projector sees only rays.
    r = u * u + v * v
    rays = np.stack(
        [u * (1 + 0.15 * r) + 0.02 * u * v, v * (1 + 0.15 * r) + 0.01 * u * u, np.ones_like(u)], axis=-1
    )
    pixels = np.array([[25.3, 40.4], [130.2, 90.7], [70.4, 80.2], [85.9, 22.3]])
    sampled, _, _, _ = _bilinear(rays, pixels)
    projected, valid = project_measured_rays(sampled * 0.3, rays)
    assert valid.all()
    np.testing.assert_allclose(projected, pixels, atol=1e-6)
    behind = sampled * -0.3
    assert not project_measured_rays(behind, rays)[1].any()


@pytest.mark.parametrize("shape", ["plane", "curved"])
@pytest.mark.parametrize(("lighting_gain", "expected_accept"), [(1.0, True), (0.95, True), (0.8, True)])
def test_photometric_ink_like_marks_preserve_original_material_support(shape, lighting_gain, expected_accept):
    """Static calibrated geometry with RGB-only marks, not a physical ink model.

    These image-coordinate strokes and global intensity changes assert only
    photometric robustness. They provide no motion or deformation ground truth.
    """
    instance = observer()
    initial, original, _ = sample(instance, shape=shape)
    baseline, _, _ = sample(instance, 1, shape=shape)
    assert initial["accepted"] and baseline["accepted"]
    inventory = instance.reference_points()
    descriptors = instance.matcher.descriptors.copy()
    gray = original["rgbd"]["gray"].copy()
    for y in (80, 120, 160):
        gray[y : y + 3, 90:220] = 12
    ink_gray = gray.copy()
    gray = np.clip(gray.astype(float) * lighting_gain + (8 if lighting_gain < 1 else 0), 0, 255).astype(
        np.uint8
    )
    marked = copy.deepcopy(original)
    marked["rgbd"]["gray"] = gray
    marked["bgr"] = np.repeat(gray[..., None], 3, axis=2)
    marked["photometric_control"] = "RGB-only ink-like marks with optional mild lighting change"
    for name in ("depth_m", "rays"):
        np.testing.assert_array_equal(marked["rgbd"][name], original["rgbd"][name])
    np.testing.assert_array_equal(marked["depth"], original["depth"])
    results = []
    for sequence, covered in ((2, False), (3, True), (4, False)):
        frame = copy.deepcopy(marked)
        frame.update(sequence=sequence, stamp=1_000_000_000 + sequence * 100_000_000)
        frame["capture_timestamp_ns"] = frame["stamp"]
        if sequence == 4:
            # Restore lighting while retaining every new ink-like mark.
            frame["rgbd"]["gray"] = ink_gray.copy()
            frame["bgr"] = np.repeat(ink_gray[..., None], 3, axis=2)
        if covered:
            frame["rgbd"]["gray"][:] = 12
            frame["bgr"][:] = 12
        results.append(instance.observe(frame, now_ns=frame["stamp"]))
    accepted, covered, recovered = results
    assert accepted["accepted"] is expected_accept, accepted
    if expected_accept:
        assert len(accepted["geometry_support"]["ids"]) < len(baseline["geometry_support"]["ids"])
    else:
        assert accepted["transform_reference_to_camera"] is None
    assert not covered["accepted"] and covered["transform_reference_to_camera"] is None
    assert recovered["accepted"] and recovered["state"] == "recovered"
    for result in results:
        assert result["reference_digest"] == initial["reference_digest"]
        assert result["reference_timestamp_ns"] == initial["capture_timestamp_ns"]
        assert result["diagnostics"]["reference_count"] == 1
    assert instance.reference_points() == inventory
    np.testing.assert_array_equal(instance.matcher.descriptors, descriptors)


def _lighting_control(frame, lighting):
    """Image-coordinate photometry only; measured depth/rays remain untouched."""
    changed = copy.deepcopy(frame)
    gray = frame["rgbd"]["gray"].astype(float)
    if lighting == "dim":
        gray = gray * 0.8 + 8
    elif lighting == "bright":
        gray = gray * 1.2 - 8
    elif lighting == "offset":
        gray += 20
    elif lighting == "shadow":
        y, x = np.mgrid[:gray.shape[0], :gray.shape[1]]
        shadow = np.exp(-((x - 160)**2 / 85**2 + (y - 120)**2 / 70**2))
        gray = gray * (1 - 0.4 * shadow) + 4
    else:
        raise ValueError(lighting)
    changed["rgbd"]["gray"] = np.clip(gray, 0, 255).astype(np.uint8)
    changed["bgr"] = np.repeat(changed["rgbd"]["gray"][..., None], 3, axis=2)
    return changed


@pytest.mark.parametrize("shape", ["plane", "curved"])
@pytest.mark.parametrize("lighting", ["dim", "bright", "offset", "shadow"])
def test_persistent_lighting_after_loss_preserves_original_metric_attachment(shape, lighting):
    instance = observer()
    initial, original, truth0 = sample(instance, shape=shape)
    assert initial["accepted"]
    descriptors, inventory = instance.matcher.descriptors.copy(), instance.reference_points()
    assert not sample(instance, 1, shape=shape, dark=True)[0]["accepted"]
    frame, truth = render_fixture(
        shape=shape, world_from_material=pose(xyz=(0.008, 0.003, 0.32), angles_deg=(2, -2, 3)),
        world_from_camera=pose(xyz=(0.004, -0.002, 0.001), angles_deg=(-1, 1, -2)),
    )
    changed = _lighting_control(frame, lighting)
    for key in ("depth_m", "rays"):
        np.testing.assert_array_equal(changed["rgbd"][key], frame["rgbd"][key])
    np.testing.assert_array_equal(changed["depth"], frame["depth"])
    expected = truth["camera_from_material"] @ np.linalg.inv(truth0["camera_from_material"])
    probes = truth0["material_points"] @ truth0["camera_from_material"][:3, :3].T + truth0["camera_from_material"][:3, 3]
    for sequence in (2, 3):
        # The changed illumination persists after recovery; it is never restored
        # to make a failing reidentification pass.
        changed.update(sequence=sequence, stamp=1_000_000_000 + sequence * 100_000_000)
        changed["capture_timestamp_ns"] = changed["stamp"]
        result = instance.observe(changed, now_ns=changed["stamp"])
        assert result["accepted"], result
        assert result["state"] == ("recovered" if sequence == 2 else "tracked")
        estimated = np.asarray(result["transform_reference_to_camera"])
        error = np.linalg.norm(probes @ (estimated[:3, :3] - expected[:3, :3]).T
                               + estimated[:3, 3] - expected[:3, 3], axis=1)
        assert error.max() < 0.0015  # Existing generated metric bound, not a physical tolerance.
        assert result["reference_digest"] == initial["reference_digest"]
        assert result["reference_timestamp_ns"] == initial["capture_timestamp_ns"]
        assert instance.reference_points() == inventory
        np.testing.assert_array_equal(instance.reference["gray"], original["rgbd"]["gray"])
        np.testing.assert_array_equal(instance.matcher.descriptors, descriptors)


@pytest.mark.parametrize("replacement", ["alternate", "deformation", "noise", "featureless_noise"])
def test_lighting_normalization_never_admits_changed_material_or_deformation(replacement):
    instance = observer()
    initial, original, _ = sample(instance, shape="curved")
    assert initial["accepted"]
    descriptors = instance.matcher.descriptors.copy()
    assert not sample(instance, 1, shape="curved", dark=True)[0]["accepted"]
    kwargs = {"appearance": "alternate"} if replacement == "alternate" else {}
    if replacement == "deformation":
        kwargs["deformation_m"] = 0.012
    changed, _ = render_fixture(shape="curved", **kwargs)
    if replacement in ("noise", "featureless_noise"):
        low, high = (0, 256) if replacement == "noise" else (178, 183)
        changed["rgbd"]["gray"] = np.random.default_rng(713).integers(low, high, (240, 320), dtype=np.uint8)
    else:
        for y in (80, 120, 160):
            changed["rgbd"]["gray"][y:y + 3, 90:220] = 12
    changed = _lighting_control(changed, "dim")
    changed.update(sequence=2, stamp=1_200_000_000, capture_timestamp_ns=1_200_000_000)
    result = instance.observe(changed, now_ns=changed["stamp"])
    assert not result["accepted"], result
    assert result["transform_reference_to_camera"] is None
    assert result["reference_digest"] == initial["reference_digest"]
    assert result["motion_authority"] is False
    np.testing.assert_array_equal(instance.reference["gray"], original["rgbd"]["gray"])
    np.testing.assert_array_equal(instance.matcher.descriptors, descriptors)
    recovered, _, _ = sample(instance, 3, shape="curved")
    assert recovered["accepted"] and recovered["state"] == "recovered"
    assert recovered["reference_digest"] == initial["reference_digest"]


def test_narrow_low_contrast_material_keeps_support_amid_stronger_clutter():
    """A fixed low-contrast ROI competes with stronger surrounding texture."""
    frame, _ = render_fixture()
    roi = (110, 30, 80, 180)
    x, y, width, height = roi
    patch = frame["rgbd"]["gray"][y:y + height, x:x + width]
    patch[:] = (128 + 0.4 * (patch.astype(float) - 128)).astype(np.uint8)
    limited = SurfaceAttachmentObserver("material-a", "reference-a", roi,
                                        settings=AttachmentSettings(max_features=600))
    rejected = limited.observe(frame, now_ns=frame["stamp"])
    assert not rejected["accepted"]
    assert rejected["reason"] == "insufficient_reference_depth_features"
    instance = SurfaceAttachmentObserver("material-a", "reference-a", roi)
    initial = instance.observe(frame, now_ns=frame["stamp"])
    assert initial["accepted"], initial
    assert initial["diagnostics"]["reference_coverage"] >= instance.settings.min_coverage
    assert instance.settings.min_points <= len(instance.reference_points()) <= 128
    inventory = instance.reference_points()
    frame.update(sequence=1, stamp=1_100_000_000, capture_timestamp_ns=1_100_000_000)
    repeated = instance.observe(frame, now_ns=frame["stamp"])
    assert repeated["accepted"], repeated
    assert repeated["reference_digest"] == initial["reference_digest"]
    assert instance.reference_points() == inventory


@pytest.mark.parametrize("depth_change_m", [0.0, 0.008])
def test_grid_cell_correspondence_repair_retains_depth_disagreement(monkeypatch, depth_change_m):
    """Inject a local one-cell alias, retaining real images and measured rays.

    The negative pair changes measured depth at the correct original pixel.
    Repairing an image association must not discard that metric disagreement.
    """
    from surface_attachment import measured_xyz

    instance = observer()
    initial, _, _ = sample(instance)
    assert initial["accepted"]
    inventory = instance.reference_points()
    original_matches, original_fit = instance._matches, instance._fit
    captured = {}

    def aliased_matches(rgbd, result):
        ids, source, current, pixels = original_matches(rgbd, result)
        index = int(np.argmin(np.linalg.norm(pixels - [160, 120], axis=1)))
        captured.update(ids=ids.copy(), repaired_id=int(ids[index]))
        if depth_change_m:
            x, y = np.rint(pixels[index]).astype(int)
            rgbd["depth_m"][y - 5:y + 6, x - 5:x + 6] += depth_change_m
        pixels = pixels.copy()
        pixels[index, 0] += 12  # One image-grid cell; no truth-based correspondence.
        measured, supported = measured_xyz(rgbd, pixels[index:index + 1], 0.003)
        assert supported.all()
        current = current.copy()
        current[index] = measured[0]
        return ids, source, current, pixels

    def checked_fit(ids, source, current, result):
        # Even an unrepairable match must reach the unchanged strict metric gate.
        np.testing.assert_array_equal(ids, captured["ids"])
        return original_fit(ids, source, current, result)

    monkeypatch.setattr(instance, "_matches", aliased_matches)
    monkeypatch.setattr(instance, "_fit", checked_fit)
    result, _, _ = sample(instance, 1)
    assert instance.reference_points() == inventory
    assert result["reference_digest"] == initial["reference_digest"]
    if depth_change_m:
        assert not result["accepted"]
        assert result["reason"] == "deformation_or_inconsistent_correspondence"
        assert result["diagnostics"]["residual_max_m"] > 0.003
        assert result["transform_reference_to_camera"] is None
        monkeypatch.setattr(instance, "_matches", original_matches)
        monkeypatch.setattr(instance, "_fit", original_fit)
        recovered, _, _ = sample(instance, 2)
        assert recovered["accepted"] and recovered["state"] == "recovered"
        assert recovered["reference_digest"] == initial["reference_digest"]
    else:
        assert result["accepted"], result
        assert result["diagnostics"]["reassociated_matches"] >= 1
        assert captured["repaired_id"] in result["geometry_support"]["ids"]
        np.testing.assert_allclose(result["transform_reference_to_camera"], np.eye(4), atol=1e-4)


def test_image_constraints_reduce_stationary_tilt_from_independent_range_noise(monkeypatch):
    """Sub-mm independent range noise, not a coherent depth ramp or bend."""
    instance = observer()
    assert sample(instance)[0]["accepted"]
    original_matches = instance._matches
    captured = {}

    def noisy_matches(rgbd, result):
        ids, source, current, pixels = original_matches(rgbd, result)
        noise = np.random.default_rng(4).uniform(-0.0009, 0.0009, len(current))
        noise -= noise.mean()
        assert abs(noise).max() < 0.001
        current = current * ((current[:, 2] + noise) / current[:, 2])[:, None]
        captured["depth_only"] = fit_rigid(source, current)
        return ids, source, current, pixels

    monkeypatch.setattr(instance, "_matches", noisy_matches)
    result, _, _ = sample(instance, 1)
    assert result["accepted"], result
    probes = instance.reference["probes"]  # Independent fixed points, never noisy matches.

    def stationary_error(matrix):
        matrix = np.asarray(matrix)
        return np.linalg.norm(probes @ matrix[:3, :3].T + matrix[:3, 3] - probes, axis=1).max()

    depth_only_error = stationary_error(captured["depth_only"])
    assert depth_only_error > 0.00003  # The control must actually induce measurable false tilt.
    assert stationary_error(result["transform_reference_to_camera"]) < depth_only_error / 4
    assert result["diagnostics"]["residual_max_m"] < 0.0015


def test_failed_nonfinite_alias_recheck_keeps_original_disagreement(monkeypatch):
    """A failed flow output is not a depth coordinate or permission to drop an ID."""
    instance = observer()
    initial, _, _ = sample(instance)
    assert initial["accepted"]
    original_matches, original_verify, original_fit = instance._matches, instance._verify_pixels, instance._fit
    captured = {}

    def wrong_match(rgbd, result):
        ids, source, current, pixels = original_matches(rgbd, result)
        current, pixels = current.copy(), pixels.copy()
        current[0, 0] += 0.012
        pixels[0, 0] += 12
        captured.update(ids=ids.copy(), current=current.copy())
        return ids, source, current, pixels

    def failed_recheck(gray, original_pixels, initial_pixels):
        if len(original_pixels) == 1:
            captured["failed_recheck"] = True
            return np.full((1, 2), np.nan), np.array([False])
        return original_verify(gray, original_pixels, initial_pixels)

    def checked_fit(ids, source, current, result):
        np.testing.assert_array_equal(ids, captured["ids"])
        np.testing.assert_array_equal(current, captured["current"])
        return original_fit(ids, source, current, result)

    monkeypatch.setattr(instance, "_matches", wrong_match)
    monkeypatch.setattr(instance, "_verify_pixels", failed_recheck)
    monkeypatch.setattr(instance, "_fit", checked_fit)
    result, _, _ = sample(instance, 1)
    assert captured["failed_recheck"]
    assert not result["accepted"]
    assert result["reason"] == "deformation_or_inconsistent_correspondence"
    assert result["transform_reference_to_camera"] is None
    assert result["reference_digest"] == initial["reference_digest"]
    assert result["diagnostics"]["reassociated_matches"] == 0


@pytest.mark.parametrize("failure", ["false", "error", "nonfinite"])
def test_image_pose_solver_failure_withholds_transform(monkeypatch, failure):
    import cv2

    instance = observer()
    initial, _, _ = sample(instance)
    assert initial["accepted"]

    def failed_solver(*args, **kwargs):
        if failure == "error":
            raise cv2.error("generated offline solver failure")
        if failure == "nonfinite":
            return True, np.full((3, 1), np.nan), np.zeros((3, 1))
        return False, None, None

    monkeypatch.setattr(cv2, "solvePnP", failed_solver)
    result, _, _ = sample(instance, 1)
    assert not result["accepted"]
    assert result["reason"] == "image_constrained_pose_failed"
    assert result["transform_reference_to_camera"] is None
    assert result["reference_digest"] == initial["reference_digest"]
    assert result["motion_authority"] is False


def _subpixel_scale_error(instance, monkeypatch):
    """A bounded localization fault at the measured-correspondence interface.

    Neither surface geometry nor the calibrated ray field changes. The small
    coherent pixel error models apparent image scale, not material deformation.
    """
    from surface_attachment import measured_xyz

    original_matches, original_fit = instance._matches, instance._fit
    captured = {}

    def matches(rgbd, result):
        ids, source, current, pixels = original_matches(rgbd, result)
        shifted = pixels.mean(axis=0) + (pixels - pixels.mean(axis=0)) * 1.004
        measured, supported = measured_xyz(rgbd, shifted, 0.003)
        current, pixels = current.copy(), pixels.copy()
        captured.update(ids=ids.copy(), pixel_error_max=np.linalg.norm(shifted - pixels, axis=1).max())
        current[supported], pixels[supported] = measured[supported], shifted[supported]
        return ids, source, current, pixels

    def checked_fit(ids, source, current, result):
        np.testing.assert_array_equal(ids, captured["ids"])
        return original_fit(ids, source, current, result)

    monkeypatch.setattr(instance, "_matches", matches)
    monkeypatch.setattr(instance, "_fit", checked_fit)
    return captured


@pytest.mark.parametrize("shape", ["plane", "curved"])
@pytest.mark.parametrize("moving", [False, True])
def test_measured_range_prevents_axial_bias_from_subpixel_image_scale(monkeypatch, shape, moving):
    instance = SurfaceAttachmentObserver("material-a", "reference-a", (110, 30, 80, 180))
    initial, _, truth0 = sample(instance, shape=shape)
    assert initial["accepted"]
    inventory = instance.reference_points()
    captured = _subpixel_scale_error(instance, monkeypatch)
    kwargs = {"world_from_material": pose(xyz=(0.006, -0.004, 0.33), angles_deg=(0, 0, 3))} if moving else {}
    result, _, truth = sample(instance, 1, shape=shape, **kwargs)
    assert result["accepted"], result
    assert 0 < captured["pixel_error_max"] < 0.5
    expected = truth["camera_from_material"] @ np.linalg.inv(truth0["camera_from_material"])
    estimated = np.asarray(result["transform_reference_to_camera"])
    probes = truth0["material_points"] @ truth0["camera_from_material"][:3, :3].T + truth0["camera_from_material"][:3, 3]
    error = np.linalg.norm(probes @ (estimated[:3, :3] - expected[:3, :3]).T
                           + estimated[:3, 3] - expected[:3, 3], axis=1)
    # Image-only translation produced >1.2 mm stationary axial bias for this
    # control, or refused the moved cases. This bound scores withheld geometry.
    assert error.max() < 0.0008
    assert abs(estimated[2, 3] - expected[2, 3]) < 0.0001
    assert instance.reference_points() == inventory
    assert result["reference_digest"] == initial["reference_digest"]


def test_subpixel_scale_correction_keeps_actual_local_deformation_refusal(monkeypatch):
    instance = SurfaceAttachmentObserver("material-a", "reference-a", (110, 30, 80, 180))
    initial, _, _ = sample(instance, shape="curved")
    assert initial["accepted"]
    inventory = instance.reference_points()
    _subpixel_scale_error(instance, monkeypatch)
    result, _, _ = sample(instance, 1, shape="curved", deformation_m=0.012)
    assert not result["accepted"], result
    assert result["reason"].startswith("deformation_or_inconsistent")
    assert result["transform_reference_to_camera"] is None
    assert instance.reference_points() == inventory
    assert result["reference_digest"] == initial["reference_digest"]


def test_measured_translation_exposes_coherent_range_bias_sensitivity(monkeypatch):
    """Range noise inside software gates is not guaranteed physical accuracy."""
    instance = SurfaceAttachmentObserver("material-a", "reference-a", (110, 30, 80, 180))
    initial, _, _ = sample(instance)
    assert initial["accepted"]
    original_matches = instance._matches

    def biased_range(rgbd, result):
        ids, source, current, pixels = original_matches(rgbd, result)
        current = current * ((current[:, 2] + 0.0007) / current[:, 2])[:, None]
        return ids, source, current, pixels

    monkeypatch.setattr(instance, "_matches", biased_range)
    result, _, _ = sample(instance, 1)
    assert result["accepted"], result
    matrix = np.asarray(result["transform_reference_to_camera"])
    # This deliberately biased measurement pulls translation by about 0.7 mm,
    # unlike bearing-only translation. Preserve this limitation as evidence.
    assert 0.0005 < matrix[2, 3] < 0.0009
    assert result["diagnostics"]["residual_max_m"] < instance.settings.residual_m
    assert result["diagnostics"]["geometry_probes"]["residual_max_m"] < instance.settings.residual_m
    assert result["reference_digest"] == initial["reference_digest"]
    assert result["motion_authority"] is False
