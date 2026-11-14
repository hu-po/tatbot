"""Observation-only camera -> estimator -> plan -> controller timing, stdlib only.

Session-clock durations never subtract sensor/ROS clocks. Frame age means age
since host receipt, not exposure. Controller feedback is evidence that JTC
sampled a goal, not proof of an SDK write or physical contact.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import queue
import threading
import time
import uuid
from pathlib import Path

SCHEMA = 'tatbot.pipeline-timing/1'
FILE = 'pipeline-timing.jsonl'
INFO = 'pipeline_timing'
METRICS = ('frame_receipt_to_submit_s', 'estimator_queue_s', 'estimator_worker_s', 'worker_end_to_filter_s',
           'plan_wall_s', 'plan_to_send_s', 'send_to_goal_response_s', 'send_to_first_controller_feedback_s',
           'plan_age_at_draw_feedback_s', 'newest_source_receipt_age_at_draw_feedback_s',
           'oldest_latest_camera_receipt_age_at_draw_feedback_s', 'oldest_input_frame_receipt_age_at_draw_feedback_s')


class Trace:
    """Bounded, non-blocking producers; one writer outside motion/camera callbacks."""

    def __init__(self, directory, clock, *, clock_domain='host_monotonic', attempt=None, capacity=4096, hardware='unknown'):
        self.path, self.clock = Path(directory) / FILE, clock
        self.domain, self.attempt = clock_domain, attempt or uuid.uuid4().hex
        self.dropped, self.error, self.closed = 0, None, False
        self._ids, self._queue, self._feedback = itertools.count(), queue.Queue(capacity), {}
        self._feedback_lock, self._completed = threading.Lock(), set()
        self._thread = threading.Thread(target=self._write, daemon=True, name='drawing-timing-writer')
        self._thread.start()
        self.emit('attempt_start', hardware=hardware)

    def emit(self, stage, **fields):
        if self.closed:
            return None
        identity = f'{self.attempt}:{next(self._ids)}'
        try:
            row = dict(schema=SCHEMA, id=identity, attempt=self.attempt, stage=stage,
                       clock_domain=self.domain, t_s=self.clock.now(), host_mono_s=time.monotonic(),
                       host_unix_ns=time.time_ns(), **fields)
            self._queue.put_nowait(json.dumps(row, allow_nan=False) + '\n')
        except queue.Full:
            self.dropped += 1
            return None
        except Exception as error:
            self.error = str(error)
            return None
        return identity

    def _write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', buffering=1) as stream:
                while (line := self._queue.get()) is not None:
                    stream.write(line)
        except OSError as error:
            self.error = str(error)

    def close(self):
        self.emit('attempt_end', dropped=self.dropped, error=self.error)
        self.closed = True
        try:
            self._queue.put(None, timeout=.5)
        except queue.Full:
            self.error = 'timing writer did not drain'
        self._thread.join(timeout=2.)
        if self._thread.is_alive():
            self.error = 'timing writer did not finish'
        status = {'dropped': self.dropped, 'error': self.error, 'closed': not self._thread.is_alive()}
        if status['closed']:
            self._status(status)
        return {**status, 'error': self.error}

    def _status(self, status):
        # The writer is joined and no motion producer writes here; preserve shutdown/flush failures.
        row = dict(schema=SCHEMA, id=f'{self.attempt}:status', attempt=self.attempt, stage='writer_status',
                   clock_domain=self.domain, t_s=self.clock.now(), host_mono_s=time.monotonic(), **status)
        try:
            with self.path.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        except (OSError, ValueError) as error:
            self.error = str(error)

    def feedback(self, goal, *, phase, elapsed_s, ros_stamp_ns=None, evidence='jtc_feedback'):
        now = self.clock.now()
        first = self._sample_feedback(goal, phase, now)
        if first is None:
            return
        self.emit('controller_feedback', goal=goal, phase=phase, elapsed_s=elapsed_s,
                  ros_stamp_ns=ros_stamp_ns, evidence=evidence, first=first)

    def _sample_feedback(self, goal, phase, now):
        with self._feedback_lock:
            if self.closed or goal in self._completed:
                return None
            last, drew = self._feedback.get(goal, (None, False))
            if last is not None and (phase != 5 or (drew and now - last < 1.)):
                return None
            self._feedback[goal] = (now, drew or phase == 5)
            return last is None

    def result(self, goal, **fields):
        with self._feedback_lock:
            self._completed.add(goal)
            self._feedback.pop(goal, None)
        self.emit('goal_result', goal=goal, **fields)


def snapshot(trace, arm, parents, *, kind='in_plane_filter'):
    return trace.emit('estimate_snapshot', arm=arm, measurements=sorted({p for p in parents if p is not None}), kind=kind)


def plan(trace, arm, op, source, build):
    if trace is None:
        return build()
    identity = trace.emit('plan_start', arm=arm, op=op, source=source)
    try:
        trajectory = build()
    except BaseException:
        trace.emit('plan_failed', plan=identity)
        raise
    ended = trace.emit('plan_end', plan=identity)
    try:
        trajectory.info[INFO] = {'plan': identity, 'plan_end': ended, 'source': source}
    except (AttributeError, TypeError) as error:
        trace.error = str(error)
    return trajectory


def worker_result(job, row, started):
    """No writer crosses the process boundary; only serializable timestamp evidence."""
    if context := job.get(INFO):
        row[INFO] = {**context, 'worker_start_host_mono_s': started,
                     'worker_end_host_mono_s': time.monotonic()}
    return row


def _read(path):
    rows, errors = [], []
    try:
        for number, line in enumerate(Path(path).read_text().splitlines(), 1):
            try:
                row = json.loads(line)
                if row['schema'] != SCHEMA or not isinstance(row['id'], str) or not row['id']:
                    raise ValueError('unknown schema or missing identity')
                if not all(isinstance(row[k], str) and row[k] for k in ('attempt', 'stage', 'clock_domain')):
                    raise ValueError('missing stage, attempt or clock domain')
                for key in ('t_s', 'host_mono_s'):
                    if isinstance(row[key], bool) or not math.isfinite(row[key]):
                        raise ValueError('invalid clock value')
                rows.append(row)
            except (ValueError, TypeError, KeyError) as error:
                errors.append(f'line {number}: {error}')
    except OSError as error:
        errors.append(str(error))
    return rows, errors


def _delta(a, b, *, host=False):
    if a is None or b is None or a['attempt'] != b['attempt'] or a['clock_domain'] != b['clock_domain']:
        return None
    key = 'host_mono_s' if host else 't_s'
    value = b[key] - a[key]
    return value if math.isfinite(value) and value >= 0 else None


def _stats(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v) and v >= 0)
    def percentile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {'n': len(values), 'p50_s': percentile(.5) if values else None,
            'p95_s': percentile(.95) if values else None, 'max_s': max(values) if values else None}


def _worker_durations(row):
    t = row.get('worker', {})
    start, end, submit = (t.get(k) for k in ('worker_start_host_mono_s', 'worker_end_host_mono_s', 'submitted_host_mono_s'))
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end, submit)):
        return {}
    return {'estimator_queue_s': start - submit, 'estimator_worker_s': end - start,
            'worker_end_to_filter_s': row['host_mono_s'] - end}


def _measurements(index, source):
    snapshot_row = index.get(source)
    if snapshot_row is None:
        return []
    return [index[m] for m in snapshot_row.get('measurements', [])
            if m in index and index[m]['stage'] == 'measurement' and index[m]['attempt'] == snapshot_row['attempt']]


def _receipt_age(measurement, event, key):
    received = measurement.get(key)
    if not isinstance(received, (int, float)) or isinstance(received, bool) or not math.isfinite(received):
        return None
    return _delta({**measurement, 't_s': received}, event)


def _feedback_metrics(row, index, goals):
    goal = goals.get(row.get('goal'))
    if goal is None:
        return {}, False
    plan_row, ended = index.get(goal.get('plan')), index.get(goal.get('plan_end'))
    measures = _measurements(index, goal.get('source'))
    result = {'send_to_first_controller_feedback_s': _delta(goal, row)} if row.get('first') else {}
    if row.get('phase') == 5:
        result['plan_age_at_draw_feedback_s'] = _delta(ended, row)
        all_ages = [_receipt_age(m, row, 'frame_receipt_t_s') for m in measures]
        ages = [age for age in all_ages if age is not None]
        result['newest_source_receipt_age_at_draw_feedback_s'] = min(ages) if ages else None
        result['oldest_latest_camera_receipt_age_at_draw_feedback_s'] = max(ages) if ages and len(ages) == len(all_ages) else None
        oldest_inputs = [_receipt_age(m, row, 'oldest_frame_receipt_t_s') for m in measures]
        oldest = [age for age in oldest_inputs if age is not None]
        result['oldest_input_frame_receipt_age_at_draw_feedback_s'] = max(oldest) if oldest and len(oldest) == len(oldest_inputs) else None
    return result, bool(plan_row and ended and measures and plan_row['attempt'] == row['attempt']
                        and ended['attempt'] == row['attempt'])


def _metrics(rows, index):
    samples, linked = {key: [] for key in METRICS}, set()
    goals = {r['id']: r for r in rows if r['stage'] == 'goal_sent'}
    for row in rows:
        stage, values = row['stage'], {}
        if stage == 'measurement':
            values = {'frame_receipt_to_submit_s': _receipt_age(row, row, 'frame_receipt_t_s')}
        elif stage == 'measurement_applied':
            values = _worker_durations(row)
        elif stage == 'plan_end':
            values = {'plan_wall_s': _delta(index.get(row.get('plan')), row, host=True)}
        elif stage == 'goal_sent':
            values = {'plan_to_send_s': _delta(index.get(row.get('plan_end')), row)}
        elif stage == 'goal_accepted':
            values = {'send_to_goal_response_s': _delta(goals.get(row.get('goal')), row)}
        elif stage == 'controller_feedback':
            values, has_lineage = _feedback_metrics(row, index, goals)
            if has_lineage:
                linked.add(row['goal'])
        for key, value in values.items():
            samples.setdefault(key, []).append(value)
    return {key: _stats(values) for key, values in samples.items()}, goals, linked


def _integrity(rows, errors):
    starts, ends, identities, statuses = set(), {}, set(), {}
    for row in rows:
        if row['id'] in identities:
            errors.append('duplicate event identity')
        identities.add(row['id'])
        if row['stage'] == 'attempt_start':
            if row['attempt'] in starts:
                errors.append('duplicate timing attempt start')
            starts.add(row['attempt'])
        elif row['stage'] == 'writer_status':
            statuses[row['attempt']] = row
        elif row['stage'] == 'attempt_end':
            if row['attempt'] in ends:
                errors.append('duplicate timing attempt end')
            ends[row['attempt']] = row
    _shutdown_integrity(starts, ends, statuses, errors)


def _shutdown_integrity(starts, ends, statuses, errors):
    if not starts or starts != set(ends) or starts != set(statuses):
        errors.append('one or more timing attempts are unfinished or missing')
    if any(not status.get('closed') for status in statuses.values()):
        errors.append('timing writer shutdown was incomplete')
    if any(e.get('dropped', 0) or e.get('error') for e in [*ends.values(), *statuses.values()]):
        errors.append('timing records were dropped or recording failed')


def _attempt_coverage(directory, rows, errors):
    path = Path(directory) / 'run.jsonl'
    if not path.exists():
        return
    try:
        expected = {r['attempt'] for line in path.read_text().splitlines()
                    if (r := json.loads(line)).get('kind') == 'draw.attempt.start'}
        recorded = {r['attempt'] for r in rows if r['stage'] == 'attempt_start'}
        if expected - recorded:
            errors.append('pipeline timing is missing for one or more draw attempts')
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(f'cannot establish draw-attempt coverage: {error}')


def report(directory):
    """Missing links produce coverage gaps and null metrics, never zero latency."""
    rows, errors = _read(Path(directory) / FILE)
    _integrity(rows, errors)
    _attempt_coverage(directory, rows, errors)
    index = {r['id']: r for r in rows}
    metrics, goals, linked = _metrics(rows, index)
    domains = sorted({r['clock_domain'] for r in rows})
    grouped = {d: _metrics([r for r in rows if r['clock_domain'] == d], index)[0] for d in domains}
    if len(domains) > 1:
        metrics = {key: _stats([]) for key in METRICS}
    feedback_goals = {r.get('goal') for r in rows if r['stage'] == 'controller_feedback'}
    return {'schema': SCHEMA, 'recording_complete': not errors, 'errors': errors,
            'hardware_modes': sorted({r.get('hardware', 'unknown') for r in rows if r['stage'] == 'attempt_start'}),
            'events': len(rows), 'attempts': len({r['attempt'] for r in rows}),
            'coverage': {'goals_sent': len(goals),
                         'goals_accepted': sum(r['stage'] == 'goal_accepted' and r.get('accepted') is True for r in rows),
                         'goals_rejected': sum(r['stage'] == 'goal_accepted' and r.get('accepted') is False for r in rows),
                         'drawing_goals': sum(bool(g.get('contains_draw')) for g in goals.values()),
                         'drawing_goals_with_plan': sum(bool(g.get('contains_draw')) and g.get('plan') in index for g in goals.values()),
                         'goals_with_controller_feedback': len(feedback_goals & goals.keys()),
                         'goals_with_measurement_plan_feedback_chain': len(linked),
                         'measurement_count': sum(r['stage'] == 'measurement' for r in rows),
                         'measurements_applied': sum(r['stage'] == 'measurement_applied' and bool(r.get('applied')) for r in rows),
                         'measurements_rejected_or_held': sum(r['stage'] == 'measurement_applied' and not r.get('applied') for r in rows)},
            'metrics': metrics, 'clock_domains': domains, 'metrics_by_clock_domain': grouped, 'unmeasured': ['sensor_exposure_to_host_receipt_s', 'sdk_command_application_s',
                                             'physical_contact_time_s'],
            'scope': ['frame ages start at host receipt; raw sensor clocks are retained without subtraction',
                      'newest applied measurement per known camera, not the age of every filter state',
                      'lineage covers in-plane filter snapshots; rigid follows, gauge/profile geometry and overhead are not attributed',
                      'multiple session clock domains have separate summaries; they are never pooled',
                      'controller feedback receipt is an upper bound on its publication time',
                      'worker execution includes frame retention and fitting; fit_s remains in track.jsonl',
                      'feedback is sampled at first feedback and at most 1 Hz during drawing']}


def markdown(summary):
    lines = ['# Drawing pipeline timing', '', f"Recording complete: {summary['recording_complete']}",
             'Hardware: ' + ', '.join(summary['hardware_modes']), '',
             '| Stage (seconds) | Samples | p50 | p95 | Maximum |', '| --- | ---: | ---: | ---: | ---: |']
    groups = summary['metrics_by_clock_domain'] or {'unknown': summary['metrics']}
    for domain, metrics in groups.items():
        lines.append(f'| **{domain}** | | | | |')
        lines.extend(_markdown_metrics(metrics))
    lines += ['', 'Coverage: ' + json.dumps(summary['coverage']), '', *summary['scope'], '',
              'Unmeasured: ' + ', '.join(summary['unmeasured']), *summary['errors']]
    return '\n'.join(lines) + '\n'


def _markdown_metrics(metrics):
    for name, values in metrics.items():
        cells = ['unknown' if values[k] is None else f'{values[k]:.4f}' for k in ('p50_s', 'p95_s', 'max_s')]
        yield f"| {name} | {values['n']} | {' | '.join(cells)} |"


def write_report(directory):
    directory = Path(directory)
    result = report(directory)
    (directory / 'pipeline-timing-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (directory / 'pipeline-timing-report.md').write_text(markdown(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--write', action='store_true', help='also save the JSON summary and Markdown report')
    args = parser.parse_args(argv)
    try:
        result = write_report(args.run) if args.write else report(args.run)
    except OSError as error:
        result = report(args.run)
        result['recording_complete'] = False
        result['errors'].append(f'could not write report: {error}')
    print(json.dumps(result, indent=2) if args.json else markdown(result))
    return 0 if result['recording_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
