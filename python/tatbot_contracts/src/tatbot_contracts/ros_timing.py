"""Measured draw-action attempt durations, including interrupted and resumed attempts.

The existing run event log owns the record. Monotonic durations include setup,
planning, execution, in-action waits and bag shutdown; they exclude time between
attempts, inspection and landing. No phase or physical contact time is inferred.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

SCHEMA = 'tatbot.draw-timing/1'


def begin(run) -> tuple[str, float]:
    attempt = (uuid.uuid4().hex, time.monotonic())
    run.event('draw.attempt.start', attempt=attempt[0])
    return attempt


def finish(run, attempt: tuple[str, float]) -> None:
    run.event('draw.attempt.end', attempt=attempt[0], duration_s=time.monotonic()-attempt[1])


def summarize(path: Path) -> dict:
    """A missing or damaged attempt makes the full duration unknown, never a partial sum."""
    result = {'schema': SCHEMA, 'complete': False, 'duration_s': None, 'attempts': [], 'reason': None}
    try:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        result['attempts'] = _attempts(rows)
        result.update(complete=True, duration_s=sum(a['duration_s'] for a in result['attempts']))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        result['reason'] = str(error)
    return result


def _attempts(rows: list[dict]) -> list[dict]:
    expected = sum(row.get('kind') in ('run.start', 'run.resume') for row in rows)
    attempts = {}
    for row in rows:
        kind = row.get('kind')
        if kind not in ('draw.attempt.start', 'draw.attempt.end'):
            continue
        key = row['attempt']
        if not isinstance(key, str) or not key:
            raise ValueError('invalid draw timing attempt identity')
        if kind == 'draw.attempt.start':
            if key in attempts:
                raise ValueError('duplicate draw timing start')
            attempts[key] = {'id': key, 'duration_s': None}
        else:
            if key not in attempts or attempts[key]['duration_s'] is not None:
                raise ValueError('unmatched or duplicate draw timing end')
            seconds = row['duration_s']
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError('invalid measured draw duration')
            attempts[key]['duration_s'] = seconds
    if not attempts or len(attempts) != expected:
        raise ValueError('draw timing is missing for one or more action attempts')
    if any(a['duration_s'] is None for a in attempts.values()):
        raise ValueError('an unfinished action attempt has unknown duration')
    return list(attempts.values())
