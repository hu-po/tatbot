"""The material cursor shared by contact chunks and standoff strokes."""

import numpy as np
import pytest
import stroke_material as material
import stroke_operation as operation


def test_one_source_keeps_its_material_identity_with_or_without_contact_cuts():
    stroke = np.array([[0., 0.], [.005, 0.], [.005, 0.], [.005, .005], [.01, .005]])
    next_stroke = np.array([[0., .02], [.01, .02]])
    whole = material.material_candidates([stroke, next_stroke])
    assert [part.source_stroke for part in whole] == [0, 1]
    assert [part.arc_range_m for part in whole] == [(0., .015), (0., .01)]
    np.testing.assert_array_equal(whole[0].uv, stroke)

    chunks = material.material_candidates([stroke, next_stroke], [.002, .002],
                                          max_lengths_m=[.006, .02])
    assert [part.source_stroke for part in chunks] == [0, 0, 0, 1]
    assert chunks[0].arc_range_m[0] == 0.
    assert chunks[2].arc_range_m[1] == pytest.approx(whole[0].arc_range_m[1])
    for before, after in zip(chunks, chunks[1:3], strict=False):
        np.testing.assert_array_equal(before.uv[-1], after.uv[0])
        assert before.arc_range_m[1] == after.arc_range_m[0]
    assert [part.arc_range_m for part in chunks] == [part.arc_range_m for part in
            material.material_candidates([stroke.copy(), next_stroke.copy()], [.002, .002],
                                         max_lengths_m=[.006, .02])]


def test_material_candidate_budget_and_geometry_refuse_before_projection():
    with pytest.raises(ValueError, match='no length'):
        material.material_candidates([[[0., 0.], [0., 0.]]])
    with pytest.raises(ValueError, match='finite chunk budget'):
        material.material_candidates([[[0., 0.], [.01, 0.]], [[0., 0.], [.01, 0.]]], [.002, .002],
                                     max_lengths_m=[.005, .005], max_candidates=3)
    with pytest.raises(ValueError, match='invalid session stroke'):
        material.material_candidates([[[0., 0.], [float('nan'), 0.]]])


def test_one_candidate_motion_sampler_preserves_contact_projection_and_standoff_chords():
    uv = np.array([[0., 0.], [.01, 0.], [.01, .01]])
    candidate = material.MaterialCandidate(2, uv, (.03, .05), .002)
    contact_arc = np.array([.005, .015, .02])

    def measured_surface(points):
        height = points[:, 0] ** 2 + points[:, 1]
        return np.column_stack([points, height]), np.tile([0., 0., 1.], (len(points), 1))

    contact = material.sample_candidate_motion(candidate, contact_arc, projector=measured_surface)
    expected_uv = np.array([[.005, 0.], [.01, .005], [.01, .01]])
    np.testing.assert_allclose(contact.uv, expected_uv)
    np.testing.assert_allclose(contact.points, measured_surface(expected_uv)[0])
    np.testing.assert_array_equal(contact.normals, np.tile([0., 0., 1.], (3, 1)))

    # The standoff chart uses chord distance after projection. Its old path
    # interpolated XYZ chords, so an intermediate point must remain on the
    # chord even if the underlying cylinder is curved.
    xyz = np.array([[0., 0., .1], [.008, 0., .106], [.008, .01, .106]])
    first_chord = np.linalg.norm(xyz[1] - xyz[0])
    standoff = material.sample_candidate_motion(candidate, [first_chord/2, first_chord + .005],
                                                metric_vertices=xyz)
    np.testing.assert_allclose(standoff.points, [(xyz[0] + xyz[1])/2, (xyz[1] + xyz[2])/2])
    np.testing.assert_allclose(standoff.uv, [[.005, 0.], [.01, .005]])
    assert standoff.normals is None


def test_shared_material_row_planner_preserves_interaction_policies():
    uv = np.array([[0., 0.], [.01, 0.]])
    candidate = material.MaterialCandidate(0, uv, (0., .01), .002)
    def orientation(projected):
        return np.tile(np.eye(3), (len(projected.uv), 1, 1)), None

    def measured_surface(points):
        return np.column_stack([points, np.full(len(points), .05)]), np.tile([0., 0., 2.], (len(points), 1))

    contact = material.plan_material_rows(candidate, [.005, .01], projector=measured_surface,
        placement=lambda projected: projected.points + .001 * projected.normals,
        orientation=orientation, normalize_normals=True, pen=1)
    np.testing.assert_allclose(contact.points[:, 2], .051)
    np.testing.assert_array_equal(contact.normals, np.tile([0., 0., 1.], (2, 1)))
    assert contact.part()[2] == 1 and contact.candidate is candidate

    xyz = np.array([[.3, 0., .1], [.31, 0., .1]])
    standoff = material.plan_material_rows(candidate, [.005, .01], metric_vertices=xyz,
                                           orientation=orientation, pen=0)
    np.testing.assert_allclose(standoff.points, [[.305, 0., .1], [.31, 0., .1]])
    assert standoff.normals is None and standoff.part()[2] == 0
    with pytest.raises(ValueError, match='surface normals'):
        material.plan_material_rows(candidate, [.005], metric_vertices=xyz,
                                    orientation=orientation, normalize_normals=True)
    with pytest.raises(ValueError, match='finite aligned'):
        material.plan_material_rows(candidate, [.005], metric_vertices=xyz,
                                    orientation=lambda projected: (np.zeros((1, 2, 2)), None), pen=0)


@pytest.mark.parametrize('pen', [0, 1])
def test_one_motion_phase_account_binds_material_for_both_interactions(pen):
    import pen_path

    uv = np.array([[0., 0.], [.01, 0.]])
    candidate = material.MaterialCandidate(4, uv, (.02, .03), .002)
    points = np.array([[.3, 0., .1], [.31, 0., .1]])
    phases = operation.MotionPhases()
    empty = (np.empty((0, 3)), np.empty((0, 3, 3)), 0, None)
    assert phases.add('travel', empty) == (0, 0)
    rows, span = operation.plan_work_phase(
        phases, candidate, [0., .01], 'work', stroke_index=4, metric_vertices=points,
        orientation=lambda projected: (
            np.tile(np.eye(3), (len(projected.uv), 1, 1)), None), pen=pen)
    assert span == (0, 2) and rows.candidate is candidate
    phases.add('retract', (points[-1:], np.eye(3)[None], 0, None))
    samples = phases.assemble(.0025, pen_path.assemble)
    assert samples.n == 3
    np.testing.assert_array_equal(samples.pen, [pen, pen, 0])
    assert phases.material_ranges == [(candidate, 0, 2)]
    assert [(phase['start'], phase['stop']) for phase in phases.segments] == [(0, 0), (0, 2), (2, 3)]


def test_material_address_keeps_original_design_identity_across_chunks():
    candidates = material.material_candidates(
        [np.array([[0., 0.], [.01, 0.]])], [.002], max_lengths_m=[.006])
    first = material.material_address(candidates[0], 0, len(candidates), 'design-placement')
    second = material.material_address(candidates[1], 1, len(candidates), 'design-placement')
    assert first['placement_id'] == second['placement_id'] == 'design-placement'
    assert first['source_stroke'] == second['source_stroke'] == 0
    assert first['source_arc_range_m'][1] == second['source_arc_range_m'][0]
    assert first['source_uv_end'] == second['source_uv_start']
    with pytest.raises(ValueError, match='retained material address'):
        material.material_address(candidates[0], 2, len(candidates), 'design-placement')
    with pytest.raises(ValueError, match='retained material address'):
        material.material_address(candidates[0], 0, len(candidates), '')


def test_contact_and_standoff_project_the_same_address_before_tool_phases():
    candidate = material.MaterialCandidate(2, np.array([[0., 0.], [.01, 0.]]),
                                           (.02, .03), .002)

    def contact_surface(uv):
        return np.column_stack([uv, .05 + uv[:, 0] ** 2]), np.tile([0., 0., 1.], (len(uv), 1))

    def standoff_chart(uv):
        return np.column_stack([uv, np.full(len(uv), .09)]), None

    contact = material.project_candidate(candidate, 1, 3, 'real-placement',
                                         contact_surface, sample_projector=contact_surface)
    standoff = material.project_candidate(candidate, 1, 3, 'real-placement', standoff_chart)
    assert contact.address == standoff.address == material.material_address(
        candidate, 1, 3, 'real-placement')
    assert contact.projector is contact_surface and standoff.projector is None
    np.testing.assert_allclose(contact.points[:, 2], [.05, .0501])
    np.testing.assert_allclose(standoff.points[:, 2], [.09, .09])
    assert contact.normals.shape == (2, 3) and standoff.normals is None

    with pytest.raises(ValueError, match='finite aligned surface vertices'):
        material.project_candidate(candidate, 1, 3, 'real-placement',
                                   lambda uv: (np.zeros((1, 3)), None))
    with pytest.raises(ValueError, match='finite aligned surface vertices'):
        material.project_candidate(candidate, 1, 3, 'real-placement',
                                   lambda uv: (np.full((2, 3), np.nan), None))
    with pytest.raises(ValueError, match='finite aligned surface vertices'):
        material.project_candidate(candidate, 1, 3, 'real-placement',
                                   lambda uv: (np.zeros((2, 3)), np.zeros((2, 3))))
    with pytest.raises(ValueError, match='retained material address'):
        material.project_candidate(candidate, 3, 3, 'real-placement', standoff_chart)


def test_native_solution_shape_is_checked_for_either_interaction():
    import pen_path

    samples = pen_path.assemble(.0025, [(np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)), 0, None)])
    plan = {'positions': np.zeros((2, 7)), 'sample_count': 2, 'pen': [False, False]}
    assert operation.checked_joint_positions(plan, samples, require_pen=True).shape == (2, 7)
    assert operation.checked_joint_positions({'positions': plan['positions']}, samples).shape == (2, 7)
    with pytest.raises(ValueError, match='invalid joint samples'):
        operation.checked_joint_positions({'positions': np.zeros((1, 7))}, samples)
    with pytest.raises(ValueError, match='differs from source samples'):
        operation.checked_joint_positions({**plan, 'pen': [True, False]}, samples, require_pen=True)


@pytest.mark.parametrize('contact', [False, True])
def test_shared_candidate_finalizer_binds_source_native_rows_and_policy(tmp_path, contact):
    import pen_path

    period = .0025
    seed = np.zeros(7)
    points = np.array([[.3, 0., .1], [.31, 0., .1]])
    samples = pen_path.assemble(period, [(points, np.tile(np.eye(3), (2, 1, 1)), int(contact), None)])
    path = tmp_path/'candidate.csv'
    calls = []

    def write(target, rows):
        assert rows is samples
        calls.append('write')
        target.write_text('bound material rows')

    def native(target, joints, tick):
        assert target.read_text() == 'bound material rows'
        np.testing.assert_array_equal(joints, seed)
        assert tick == period
        calls.append('native')
        plan = {'schema': 'tatbot.joint-plan/1', 'hardware_authority': False,
                'constants_sha': 'fixture-sha', 'period_s': period, 'seed': seed.tolist(),
                'sample_count': samples.n, 'positions': np.zeros((samples.n, 7)).tolist()}
        if contact:
            plan['pen'] = [True] * samples.n
        return operation.ExecutorPlan(plan, 'native stdout', '')

    def policy(rows, plan, positions):
        assert rows is samples and plan['sample_count'] == len(positions)
        calls.append('policy')
        return 'accepted policy'

    result = operation.finalize_candidate(
        samples, seed, path, write_samples=write, native_plan=native,
        constants_sha='fixture-sha', require_pen=contact, validate_policy=policy)
    assert calls == ['write', 'native', 'policy']
    assert result.native.stdout == 'native stdout' and result.policy == 'accepted policy'
    assert result.samples_path == path and result.positions.shape == (samples.n, 7)

    def changed_seed(target, joints, tick):
        altered = native(target, joints, tick)
        altered.plan['seed'][0] = 1.
        return altered

    calls.clear()
    with pytest.raises(ValueError, match='mismatched plan'):
        operation.finalize_candidate(
            samples, seed, path, write_samples=write, native_plan=changed_seed,
            constants_sha='fixture-sha', require_pen=contact, validate_policy=policy)
    assert calls == ['write', 'native']
