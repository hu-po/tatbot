"""Timing remains complete across resume, and refuses partial or damaged evidence."""
import json

import pytest
from tatbot_session import timing


class Log:
    def __init__(self, path):
        self.path = path

    def event(self, kind, **fields):
        with self.path.open('a') as stream:
            stream.write(json.dumps({'kind': kind, **fields})+'\n')


def test_resumed_attempts_use_monotonic_elapsed_without_idle_gap(tmp_path, monkeypatch):
    clock = iter([10., 12.5, 90., 94.])
    monkeypatch.setattr(timing.time, 'monotonic', lambda: next(clock))
    log = Log(tmp_path/'run.jsonl')
    log.event('run.start')
    timing.finish(log, timing.begin(log))
    log.event('run.end', duration_s=2.5)
    log.event('run.resume')
    timing.finish(log, timing.begin(log))
    log.event('run.end', duration_s=4.)
    result = timing.summarize(log.path)
    assert result['complete'] and result['duration_s'] == 6.5
    assert [a['duration_s'] for a in result['attempts']] == [2.5, 4.]


def test_crashed_attempt_is_not_hidden_by_later_complete_attempt(tmp_path):
    log = Log(tmp_path/'run.jsonl')
    log.event('run.start')
    timing.begin(log)  # process dies before it writes the end
    log.event('run.resume')
    timing.finish(log, timing.begin(log))
    result = timing.summarize(log.path)
    assert not result['complete'] and result['duration_s'] is None
    assert 'unfinished' in result['reason']


@pytest.mark.parametrize('damage', ['missing_start', 'duplicate_start', 'duplicate_end', 'torn', 'negative', 'nan'])
def test_incomplete_or_corrupt_log_never_reports_partial_total(tmp_path, damage):
    log = Log(tmp_path/'run.jsonl')
    log.event('run.start')
    if damage != 'missing_start':
        log.event('draw.attempt.start', attempt='one')
    if damage == 'duplicate_start':
        log.event('draw.attempt.start', attempt='one')
    log.event('draw.attempt.end', attempt='one', duration_s={'negative': -1, 'nan': float('nan')}.get(damage, 1.))
    if damage == 'duplicate_end':
        log.event('draw.attempt.end', attempt='one', duration_s=1.)
    if damage == 'torn':
        with log.path.open('a') as stream:
            stream.write('{"kind":"draw.attempt.start"')
    result = timing.summarize(log.path)
    assert not result['complete'] and result['duration_s'] is None and result['reason']


def test_old_run_duration_is_not_substituted_for_missing_measurements(tmp_path):
    log = Log(tmp_path/'run.jsonl')
    log.event('run.start')
    log.event('run.end', duration_s=123.)
    assert timing.summarize(log.path)['duration_s'] is None
