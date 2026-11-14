"""Causal lineage, clock boundaries and observation-only logging failure behavior."""
import json
from types import SimpleNamespace

import pytest
from tatbot_contracts import ros_pipeline as p


class Clock:
    t = 0.

    def now(self):
        return self.t

    def utc(self):
        return 1_800_000_000 + self.t


def event(trace, clock, at, stage, **fields):
    clock.t = at
    return trace.emit(stage, **fields)


def source(trace, clock, at, camera=0, *, receipt=None):
    measurement = event(trace, clock, at, 'measurement', camera=camera, frame_receipt_t_s=receipt,
                        oldest_frame_receipt_t_s=receipt)
    event(trace, clock, at + .1, 'measurement_applied', measurement=measurement, applied=True)
    return p.snapshot(trace, 'xarm', [measurement])


def goal(trace, clock, src, *, start=3., send=4.):
    clock.t = start
    trajectory = SimpleNamespace(info={}, q=[[1, 2, 3]])
    def build():
        clock.t += .2
        return trajectory
    assert p.plan(trace, 'xarm', 's1', src, build) is trajectory
    identity = event(trace, clock, send, 'goal_sent', op='s1', **trajectory.info[p.INFO])
    return identity, trajectory


def test_reused_preplan_keeps_its_original_measurement_and_first_feedback_is_separate(tmp_path):
    clock = Clock()
    trace = p.Trace(tmp_path, clock)
    src = source(trace, clock, 2.1, receipt=2.)
    identity, trajectory = goal(trace, clock, src)
    # A newer live estimate must not silently replace the source frozen in this preplan.
    source(trace, clock, 5., receipt=4.9)
    event(trace, clock, 4.05, 'goal_accepted', goal=identity, accepted=True)
    clock.t = 4.1
    trace.feedback(identity, phase=2, elapsed_s=0.)
    clock.t = 6.
    trace.feedback(identity, phase=5, elapsed_s=1.9)
    clock.t = 6.1
    trace.feedback(identity, phase=5, elapsed_s=2.)  # rate limited, not another timing sample
    clock.t = 7.1
    trace.feedback(identity, phase=5, elapsed_s=3.)
    trace.result(identity, status=4, error_code=0)
    assert trace.close()['error'] is None
    result = p.report(tmp_path)
    assert result['recording_complete']
    assert result['coverage']['goals_with_measurement_plan_feedback_chain'] == 1
    m = result['metrics']
    assert m['send_to_first_controller_feedback_s']['n'] == 1
    assert m['send_to_first_controller_feedback_s']['p95_s'] == pytest.approx(.1)
    assert m['newest_source_receipt_age_at_draw_feedback_s']['n'] == 2
    assert m['newest_source_receipt_age_at_draw_feedback_s']['max_s'] == pytest.approx(5.1)
    assert trajectory.q == [[1, 2, 3]]
    assert 'sdk_command_application_s' in result['unmeasured']


def test_unknown_capture_clock_never_uses_submission_as_exposure_or_receipt(tmp_path):
    clock = Clock()
    trace = p.Trace(tmp_path, clock, clock_domain='simulation')
    src = source(trace, clock, 2.)
    identity, _ = goal(trace, clock, src)
    clock.t = 6.
    trace.feedback(identity, phase=5, elapsed_s=2., ros_stamp_ns=99_000_000_000)
    trace.close()
    m = p.report(tmp_path)['metrics']
    assert m['newest_source_receipt_age_at_draw_feedback_s'] == p._stats([])
    assert m['plan_age_at_draw_feedback_s']['max_s'] == pytest.approx(2.8)


def test_missing_feedback_is_reported_as_missing_coverage_not_zero_delay(tmp_path):
    clock = Clock()
    trace = p.Trace(tmp_path, clock)
    goal(trace, clock, None)
    trace.close()
    result = p.report(tmp_path)
    assert result['recording_complete']
    assert result['coverage']['goals_sent'] == 1
    assert result['coverage']['goals_with_controller_feedback'] == 0
    assert result['metrics']['send_to_first_controller_feedback_s']['p95_s'] is None


def test_full_queue_never_blocks_a_producer_and_reports_the_drop(tmp_path, monkeypatch):
    import queue

    clock = Clock()
    trace = p.Trace(tmp_path, clock)
    original = trace._queue.put_nowait
    def full(_):
        raise queue.Full
    monkeypatch.setattr(trace._queue, 'put_nowait', full)
    assert trace.emit('measurement') is None and trace.dropped == 1
    monkeypatch.setattr(trace._queue, 'put_nowait', original)
    trace.close()
    assert not p.report(tmp_path)['recording_complete']


def test_writer_failure_does_not_change_a_planned_trajectory(tmp_path):
    folder = tmp_path / 'file'
    folder.write_text('not a directory')
    clock = Clock()
    trace = p.Trace(folder, clock)
    identity, trajectory = goal(trace, clock, None)
    assert identity is not None and trajectory.q == [[1, 2, 3]]
    assert trace.close()['error']
    assert not p.report(folder)['recording_complete']


def test_failed_planning_preserves_the_original_exception(tmp_path):
    trace = p.Trace(tmp_path, Clock())
    error = RuntimeError('planner failed')
    def build():
        raise error
    with pytest.raises(RuntimeError) as caught:
        p.plan(trace, 'xarm', 's1', None, build)
    assert caught.value is error
    trace.close()
    assert any(json.loads(line)['stage'] == 'plan_failed' for line in trace.path.read_text().splitlines())


@pytest.mark.parametrize('damage', ['torn', 'duplicate', 'missing_end', 'missing_domain'])
def test_corrupt_or_incomplete_attempts_remain_incomplete(tmp_path, damage):
    trace = p.Trace(tmp_path, Clock())
    trace.close()
    rows = trace.path.read_text().splitlines()
    if damage == 'torn':
        rows.append('{')
    elif damage == 'duplicate':
        rows.append(rows[0])
    elif damage == 'missing_end':
        rows = [r for r in rows if json.loads(r)['stage'] != 'attempt_end']
    else:
        row = json.loads(rows[0])
        del row['clock_domain']
        rows[0] = json.dumps(row)
    trace.path.write_text('\n'.join(rows) + '\n')
    assert not p.report(tmp_path)['recording_complete']


def test_filter_rejections_do_not_replace_snapshot_lineage(tmp_path):
    from tatbot_session import timing

    trace = p.Trace(tmp_path, Clock())
    tracker = SimpleNamespace(ex=SimpleNamespace(pipeline_trace=trace, arm='xarm'), _timing_applied={0: 'old'})
    for verdict, accepted in [('held', True), ('added', False), (None, False)]:
        timing.applied(tracker, {'accepted': accepted, 'field': verdict, p.INFO: {'measurement': 'new'}}, 0)
        assert tracker._timing_applied == {0: 'old'}
    timing.applied(tracker, {'accepted': True, 'field': 'added', p.INFO: {'measurement': 'new'}}, 0)
    assert tracker._timing_applied == {0: 'new'}
    trace.close()


def test_late_goal_callback_cannot_join_a_new_attempt(tmp_path):
    from tatbot_session import timing

    trace = p.Trace(tmp_path, Clock(), attempt='second')
    io = SimpleNamespace(pipeline_trace=trace)
    timing.accepted(io, 'first:100', True)
    trace.close()
    assert all(json.loads(line)['stage'] != 'goal_accepted' for line in trace.path.read_text().splitlines())


def test_wrist_worker_filter_snapshot_plan_and_arm_goal_share_one_chain(tmp_path, monkeypatch):
    from concurrent.futures import Future

    import numpy as np
    from tatbot_description import names
    from tatbot_motion import Trajectory
    from tatbot_session import timing, track
    from tatbot_session.arm import ArmIO
    from tatbot_session.page_filter import PageFilter

    clock = Clock()
    clock.t = 2.
    trace = p.Trace(tmp_path, clock)
    def fit(*args):
        return {'surface': True, 'fit': 'markers', 'accepted': True, 'tip_m': [0., 0., 0.],
                'rms_m': .0001, 'inliers': 4, 'ids': [1, 2, 3, 4], 'margin': 1.}
    monkeypatch.setattr(track, '_fit', fit)
    def submit(fn, job):
        future = Future()
        future.set_result(fn(job))
        return future
    tracker = track.Tracker.__new__(track.Tracker)
    tracker.ex = SimpleNamespace(arm='xarm', pipeline_trace=trace, clock=clock, used=lambda: np.eye(4),
        stack={'page': {'size_m': [.1, .1], 'clear_m': [.1, .1]}},
        _curved=lambda: False, surface=lambda: None, skin_tilt=lambda: None,
        kin=SimpleNamespace(fk=lambda q: np.eye(4), frame=lambda *a: np.eye(4)),
        _followed=lambda: np.eye(3), _timed_moves=lambda _: [], run_dir=tmp_path, event=lambda *a, **k: None)
    tracker.pending, tracker.pool = [], SimpleNamespace(submit=submit)
    tracker.views = [{'role': 'wrist_1', 'tip_cam': np.zeros(3), 'cal': {}, 'frame': 'camera'}]
    tracker._camera, tracker._overhead = lambda _: np.eye(4), lambda _: None
    tracker.tip_cam, tracker.cal, tracker.rigid = np.zeros(3), {}, None
    tracker.field, tracker._fed, tracker._timing_applied = PageFilter(cameras=2), 0., {}
    tracker.print_held, tracker.printed, tracker.ends, tracker.wide = [], [], [], []
    for camera, received in enumerate([1.8, 1.9]):
        tracker._submit('hover' + str(camera), np.zeros(2), np.zeros((3, 3, 3), np.uint8),
                        np.zeros((3, 3), np.uint16), {}, np.zeros(7), keep=False, op_id='s1', camera=camera,
                        capture=[{'received_monotonic_s': received, 'color_ms': 123456.,
                                  'color_domain': 'hardware_clock', 'color_frame_number': 40 + camera}])
    clock.t = 2.3
    tracker.drain(wait=True)
    snap = tracker.snapshot()
    assert len(tracker._timing_applied) == 2
    trajectory = Trajectory(names.joint_names('xarm'), np.array([0., .1]), np.zeros((2, 7)), np.zeros((2, 7)),
                            np.zeros((2, 3)), np.array([0., .001]), np.array([5, 5], np.uint8))
    clock.t = 3.
    assert timing.plan(tracker.ex, {'id': 's1'}, snap, lambda: trajectory) is trajectory
    io = ArmIO.__new__(ArmIO)
    io.arm, io.op_id, io.clock, io.pipeline_trace = 'xarm', 's1', clock, trace
    io.joint_names, io.goals, io.goal_log = names.joint_names('xarm'), None, None
    accepted_future, result_future, transport = Future(), Future(), {}
    def send(goal, feedback_callback):
        transport.update(goal=goal, feedback=feedback_callback)
        return accepted_future
    io.jtc = SimpleNamespace(send_goal_async=send)
    unchanged = io.goal(trajectory)
    clock.t = 4.
    done, _ = io._send(trajectory)
    assert transport['goal'] == unchanged
    clock.t = 4.05
    accepted_future.set_result(SimpleNamespace(accepted=True, get_result_async=lambda: result_future,
                                               goal_id=SimpleNamespace(uuid=list(range(16)))))
    clock.t = 4.5
    transport['feedback'](SimpleNamespace(feedback=SimpleNamespace(
        desired=SimpleNamespace(time_from_start=SimpleNamespace(sec=0, nanosec=50_000_000)),
        header=SimpleNamespace(stamp=SimpleNamespace(sec=99, nanosec=0)))))
    result_future.set_result(SimpleNamespace(status=4, result=SimpleNamespace(error_code=0)))
    assert done.is_set()
    trace.close()
    result = p.report(tmp_path)
    assert result['recording_complete'], result
    assert result['coverage']['goals_with_measurement_plan_feedback_chain'] == 1
    assert result['coverage']['measurements_applied'] == 2
    assert result['metrics']['newest_source_receipt_age_at_draw_feedback_s']['max_s'] == pytest.approx(2.6)
    assert result['metrics']['oldest_latest_camera_receipt_age_at_draw_feedback_s']['max_s'] == pytest.approx(2.7)


def test_report_runs_with_only_standard_library_and_old_runs_stay_unknown(tmp_path):
    import os
    import subprocess
    import sys

    result = subprocess.run([sys.executable, p.__file__, str(tmp_path), '--json'],
                            env={**os.environ, 'PYTHONPATH': ''}, capture_output=True, text=True)
    assert result.returncode == 1
    summary = json.loads(result.stdout)
    assert not summary['recording_complete'] and summary['metrics']['plan_to_send_s']['p95_s'] is None


def test_resume_without_timing_cannot_hide_behind_an_earlier_complete_attempt(tmp_path):
    trace = p.Trace(tmp_path, Clock(), attempt='one')
    trace.close()
    (tmp_path / 'run.jsonl').write_text('\n'.join(json.dumps({'kind': 'draw.attempt.start', 'attempt': key})
                                                 for key in ['one', 'two']) + '\n')
    result = p.report(tmp_path)
    assert not result['recording_complete'] and any('draw attempts' in e for e in result['errors'])


def test_simulation_and_host_clock_summaries_are_never_pooled(tmp_path):
    clock = Clock()
    for domain, send in [('host_monotonic', 4.), ('simulation', 100.)]:
        trace = p.Trace(tmp_path, clock, clock_domain=domain)
        goal(trace, clock, None, send=send)
        trace.close()
    result = p.report(tmp_path)
    assert result['recording_complete']
    assert result['metrics']['plan_to_send_s']['n'] == 0
    assert result['metrics_by_clock_domain']['host_monotonic']['plan_to_send_s']['n'] == 1
    assert result['metrics_by_clock_domain']['simulation']['plan_to_send_s']['max_s'] > 90


def test_partial_camera_timestamps_do_not_claim_a_known_oldest_source_age(tmp_path):
    clock = Clock()
    trace = p.Trace(tmp_path, clock)
    first = event(trace, clock, 1., 'measurement', camera=0, frame_receipt_t_s=None)
    second = event(trace, clock, 2., 'measurement', camera=1, frame_receipt_t_s=1.9)
    src = p.snapshot(trace, 'xarm', [first, second])
    identity, _ = goal(trace, clock, src)
    clock.t = 6.
    trace.feedback(identity, phase=5, elapsed_s=2.)
    trace.close()
    metrics = p.report(tmp_path)['metrics']
    assert metrics['newest_source_receipt_age_at_draw_feedback_s']['max_s'] == pytest.approx(4.1)
    assert metrics['oldest_latest_camera_receipt_age_at_draw_feedback_s']['max_s'] is None


def test_bad_optional_goal_metadata_cannot_interrupt_acceptance_or_feedback(tmp_path):
    from tatbot_session import timing

    trace = p.Trace(tmp_path, Clock())
    io = SimpleNamespace(pipeline_trace=trace)
    timing.accepted(io, trace.attempt + ':goal', True, SimpleNamespace(uuid=None))
    timing.feedback(io, trace.attempt + ':goal', None, SimpleNamespace())
    assert trace.error
    trace.close()
    assert not p.report(tmp_path)['recording_complete']


def test_snapshot_metadata_failure_keeps_the_original_control_field(tmp_path):
    from tatbot_session import timing

    trace = p.Trace(tmp_path, Clock())
    tracker = SimpleNamespace(ex=SimpleNamespace(pipeline_trace=trace, arm='xarm'))
    field = SimpleNamespace()
    assert timing.snapshot(tracker, field) is field
    assert trace.error and not hasattr(field, 'pipeline_source')
    trace.close()


def test_concurrent_or_post_result_feedback_does_not_duplicate_first_sample(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    clock = Clock()
    trace = p.Trace(tmp_path, clock)
    identity, _ = goal(trace, clock, None)
    clock.t = 6.
    with ThreadPoolExecutor(4) as executor:
        list(executor.map(lambda _: trace.feedback(identity, phase=5, elapsed_s=2.), range(16)))
    trace.result(identity, status=4, error_code=0)
    clock.t = 8.
    trace.feedback(identity, phase=5, elapsed_s=4.)
    trace.close()
    result = p.report(tmp_path)
    assert result['recording_complete']
    assert result['metrics']['send_to_first_controller_feedback_s']['n'] == 1
    assert result['metrics']['plan_age_at_draw_feedback_s']['n'] == 1


def test_rejected_goal_response_is_not_reported_as_acceptance_or_execution(tmp_path):
    clock = Clock()
    trace = p.Trace(tmp_path, clock)
    identity, _ = goal(trace, clock, None)
    event(trace, clock, 4.05, 'goal_accepted', goal=identity, accepted=False)
    trace.close()
    result = p.report(tmp_path)
    assert result['coverage']['goals_accepted'] == 0 and result['coverage']['goals_rejected'] == 1
    assert result['metrics']['send_to_goal_response_s']['p50_s'] == pytest.approx(.05)
    assert result['metrics']['send_to_first_controller_feedback_s']['n'] == 0
