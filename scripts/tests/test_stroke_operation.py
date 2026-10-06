"""The two offline consumers bind their native plans to the same inputs."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import stroke_material as material
import stroke_operation as operation


def _response(source_seed, **changes):
    plan = {'schema': 'tatbot.joint-plan/1', 'hardware_authority': False,
            'constants_sha': 'constants', 'period_s': .01, 'seed': source_seed.tolist()}
    plan.update(changes)
    return SimpleNamespace(returncode=0, stdout=json.dumps(plan), stderr='planner note')


def test_executor_plan_binds_input_and_optional_arm_prefix(tmp_path):
    seed = np.arange(7, dtype=float) / 10
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return _response(seed)

    accepted = operation.executor_plan(tmp_path/'planner', tmp_path/'samples.csv', seed,
                                       .01, 'constants', arm_prefix='left', runner=runner)
    assert accepted.plan['seed'] == seed.tolist() and accepted.stderr == 'planner note'
    assert calls[0][0] == [str(tmp_path/'planner'), str(tmp_path/'samples.csv'), '0.01',
                           *map(str, seed), '--json', '--arm-prefix', 'left']
    assert calls[0][1] == {'capture_output': True, 'text': True, 'timeout': 120, 'check': False}


@pytest.mark.parametrize('change', [
    {'schema': 'other'}, {'hardware_authority': True}, {'constants_sha': 'other'},
    {'period_s': .02}, {'seed': [0.] * 7},
])
def test_executor_plan_refuses_mismatched_success(tmp_path, change):
    seed = np.arange(7, dtype=float) / 10
    with pytest.raises(ValueError, match='mismatched plan'):
        operation.executor_plan(tmp_path/'planner', tmp_path/'samples.csv', seed,
                                .01, 'constants', runner=lambda *_a, **_k: _response(seed, **change))


def test_executor_plan_distinguishes_native_refusal_from_bad_input(tmp_path):
    seed = np.arange(7, dtype=float) / 10
    rejected = SimpleNamespace(returncode=3, stdout='', stderr='joint limit')
    with pytest.raises(operation.ExecutorRefusalError, match='joint limit') as error:
        operation.executor_plan(tmp_path/'planner', tmp_path/'samples.csv', seed,
                                .01, 'constants', runner=lambda *_a, **_k: rejected)
    assert error.value.returncode == 3 and error.value.stderr == 'joint limit'
    with pytest.raises(ValueError, match='finite seven-axis seed'):
        operation.executor_plan(tmp_path/'planner', tmp_path/'samples.csv', [float('nan')]*7,
                                .01, 'constants', runner=lambda *_a, **_k: pytest.fail('called planner'))


def test_executor_plan_binds_independently_resolved_tool(tmp_path):
    seed = np.arange(7, dtype=float) / 10
    tip = [.2, .01, -.001]
    tool = {'arm_prefix': 'right', 'source': 'bound-input', 'tip_in_link6': tip,
            'carriage_axis_in_link6': [0., 1., 0.]}
    calls = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        return _response(seed, tool_model=tool)

    plan = operation.executor_plan(tmp_path/'planner', tmp_path/'samples.csv', seed,
                                   .01, 'constants', arm_prefix='right', tool_tip_in_link6=tip, runner=runner)
    assert plan.plan['tool_model'] == tool
    assert calls[0][-6:] == ['--arm-prefix', 'right', '--tool-tip-in-link6', *map(str, tip)]


@pytest.mark.parametrize('change', [
    None, {'arm_prefix': 'left'}, {'source': 'samples-header'}, {'tip_in_link6': [.201, .01, -.001]},
    {'tip_in_link6': [float('nan'), .01, -.001]}, {'tip_in_link6': [.2]},
    {'carriage_axis_in_link6': [0., -1., 0.]},
])
def test_executor_plan_refuses_unbound_or_wrong_returned_tool(tmp_path, change):
    seed = np.zeros(7)
    tip = [.2, .01, -.001]
    tool = {'arm_prefix': 'right', 'source': 'bound-input', 'tip_in_link6': tip,
            'carriage_axis_in_link6': [0., 1., 0.]}
    tool = None if change is None else dict(tool, **change)
    with pytest.raises(ValueError, match='mismatched tool model'):
        operation.executor_plan(tmp_path/'planner', tmp_path/'samples.csv', seed,
                                .01, 'constants', arm_prefix='right', tool_tip_in_link6=tip,
                                runner=lambda *_a, **_k: _response(seed, tool_model=tool))


@pytest.mark.parametrize(('prefix', 'tip'), [
    (None, [.2, .01, 0.]), ('right', [float('nan'), 0., 0.]),
    ('right', [.01, 0., 0.]), ('right', [.6, 0., 0.]), ('right', [.2]),
])
def test_executor_plan_refuses_invalid_bound_tool_before_launch(tmp_path, prefix, tip):
    with pytest.raises(ValueError, match='bound tool needs'):
        operation.executor_plan(tmp_path/'planner', tmp_path/'samples.csv', np.zeros(7),
                                .01, 'constants', arm_prefix=prefix, tool_tip_in_link6=tip,
                                runner=lambda *_a, **_k: pytest.fail('called planner'))


@pytest.mark.parametrize('pen', [0, 1])
def test_plan_next_binds_original_work_and_native_solution_for_both_policies(tmp_path, pen):
    import pen_path

    uv = np.array([[0., 0.], [.01, 0.]])
    candidate = material.MaterialCandidate(3, uv, (.02, .03), .002)
    cursor = operation.StrokeCursor(candidate, 1, 2, 'original-placement')
    positions = np.array([[.3, 0., .1], [.31, 0., .1]])
    work = operation.WorkPolicy(
        np.array([0., .01]), 'contact' if pen else 'standoff',
        orientation=lambda projected: (np.tile(np.eye(3), (len(projected.uv), 1, 1)), None),
        metric_vertices=positions, pen=pen)
    seed = np.zeros(7)
    calls = []

    def write_samples(path, samples, span):
        path.write_text(f'{samples.n} rows')
        assert span == (0, 2)
        calls.append('write')

    def native_plan(_path, joints, period):
        calls.append('native')
        plan = {'schema': 'tatbot.joint-plan/1', 'hardware_authority': False,
                'constants_sha': 'constants', 'period_s': period, 'seed': joints.tolist(),
                'positions': np.zeros((2, 7)).tolist(), 'sample_count': 2,
                'pen': [bool(pen)] * 2}
        return operation.ExecutorPlan(plan, '', '')

    planned = operation.plan_next(
        cursor, work, before=[], after_work=lambda _rows: [], period_s=.01,
        assembler=pen_path.assemble,
        native=operation.NativeRequest(
            seed, tmp_path/'candidate.csv', write_samples, native_plan, 'constants',
            require_pen=bool(pen), validate_policy=lambda *_: calls.append('policy') or True))
    assert calls == ['write', 'native', 'policy']
    assert planned.work_range == (0, 2) and planned.native.policy is True
    assert planned.material_origin == {
        'chunk_index': 1, 'source_chunk_count': 2, 'placement_id': 'original-placement',
        'source_stroke': 3, 'source_arc_range_m': [.02, .03],
        'source_uv_start': [0., 0.], 'source_uv_end': [.01, 0.]}
    np.testing.assert_array_equal(planned.samples.pen, [pen, pen])


def test_plan_next_refuses_native_work_without_original_placement(tmp_path):
    uv = np.array([[0., 0.], [.01, 0.]])
    cursor = operation.StrokeCursor(material.MaterialCandidate(0, uv, (0., .01)), 0, 1)
    work = operation.WorkPolicy(
        np.array([0., .01]), 'standoff',
        orientation=lambda projected: (np.tile(np.eye(3), (len(projected.uv), 1, 1)), None),
        metric_vertices=np.array([[0., 0., .1], [.01, 0., .1]]))
    with pytest.raises(ValueError, match='original design placement'):
        operation.plan_next(
            cursor, work, before=[], after_work=lambda _rows: [], period_s=.01,
            assembler=lambda *_: pytest.fail('assembled unbound work'),
            native=operation.NativeRequest(
                np.zeros(7), tmp_path/'unbound.csv',
                lambda *_: pytest.fail('wrote unbound work'),
                lambda *_: pytest.fail('native planner saw unbound work'), 'constants'))


@pytest.mark.parametrize('accepted', [False, True])
def test_contact_resources_commit_only_after_shared_native_acceptance(tmp_path, accepted):
    import pen_path

    candidate = material.MaterialCandidate(0, np.array([[0., 0.], [.01, 0.]]), (0., .01))
    cursor = operation.StrokeCursor(candidate, 0, 1, 'placed-artwork', 3)
    work = operation.WorkPolicy(
        np.array([0., .01]), 'contact',
        orientation=lambda projected: (np.tile(np.eye(3), (len(projected.uv), 1, 1)), None),
        metric_vertices=np.array([[0., 0., .1], [.01, 0., .1]]), pen=1)
    calls = []
    ink_credits = []

    def write_samples(path, rows, _span):
        calls.append('write')
        path.write_text(f'{rows.n} rows')

    def native(_path, seed, period):
        calls.append('native')
        if not accepted:
            raise operation.ExecutorRefusalError('joint limit', 3)
        return operation.ExecutorPlan({
            'schema': 'tatbot.joint-plan/1', 'hardware_authority': False,
            'constants_sha': 'constants', 'period_s': period, 'seed': seed.tolist(),
            'positions': np.zeros((2, 7)).tolist(), 'sample_count': 2,
            'pen': [True, True]}, '', '')

    class Resources:
        def prepare_draw(self, planned):
            calls.append('prepare')
            assert planned.report['material_origin'] == planned.material_origin
            assert planned.material_origin['chunk_index'] == 0
            assert planned.material_origin['source_chunk_count'] == 1
            assert planned.material_origin['source_chunk_origin'] == 3
            assert planned.native is None
            return operation.NativeRequest(
                np.zeros(7), tmp_path/'contact.csv', write_samples,
                native, 'constants', require_pen=True)

        def commit_draw(self, planned):
            calls.append('commit')
            assert planned.native.positions.shape == (2, 7)
            ink_credits.append('accepted')

    def plan():
        return operation.plan_next(
            cursor, work, before=[], after_work=lambda _rows: [], period_s=.01,
            assembler=pen_path.assemble,
            contact_report=lambda planned: {'material_origin': planned.material_origin},
            contact_resources=Resources())

    if accepted:
        assert plan().native is not None
        assert calls == ['prepare', 'write', 'native', 'commit']
        assert ink_credits == ['accepted']
    else:
        with pytest.raises(operation.ExecutorRefusalError, match='joint limit'):
            plan()
        assert calls == ['prepare', 'write', 'native']
        assert ink_credits == []
