"""A program's writing_height_reference: draw at least lift_m over where an earlier run on the same print drew.

The new run still touches its page; this raises only the pen's writing buffer, never lowers it, so a new contact
estimate cannot cancel an operator's lift. The earlier run must have drawn every stroke, on the same physical print,
within 45 mm, with the tracked page moved under 3 mm, and with the same fitted tool and calibration.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
from tatbot_contracts.ros_program import program_sha256, validate_for_execution
from tatbot_contracts.ros_progress import last_events, read


def _pose(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6)):
        raise ValueError('writing reference has an invalid page pose')
    return pose


def _centre(program):
    points = np.array([p for op in program['ops'] if op['op'] == 'stroke' for p in op['points_m']])
    return (points.min(axis=0) + points.max(axis=0)) / 2


def resolve(program, arm, run_dir, page, pen, workspace):
    """Return an upward-only PenDown and evidence, or refuse a stale local reference."""
    reference = program.get('writing_height_reference')
    if reference is None:
        return pen, None
    validate_for_execution(program)
    if not pen.machine or reference['run_id'] == run_dir.name:
        raise ValueError('writing reference requires a previous powered run')
    source, old = _source(reference, arm, run_dir, workspace)
    _local(program, source, page, old)
    return _height(reference, old, page, pen)


def _source(reference, arm, run_dir, workspace):
    directory = run_dir.parent / reference['run_id']
    source = json.loads((directory / 'program.json').read_text())
    validate_for_execution(source)
    rows = read(directory / 'ledger.jsonl')
    completed = {op for op, row in last_events(rows, arm).items() if row['event'] == 'done'}
    strokes = {op['id'] for op in source['ops'] if op['op'] == 'stroke'}
    if not strokes or not strokes <= completed or source['arm'] != arm:
        raise ValueError('writing reference requires a completed drawing on the same arm')
    pages = [row for row in rows if row.get('arm') == arm and row.get('event') == 'page'
             and program_sha256(row['page']) == reference['page_sha256']]
    if len(pages) != 1:
        raise ValueError('writing reference names no page that run measured')
    if pages[0].get('workspace') != workspace:
        raise ValueError('writing reference was drawn with another fitted tool or calibration')
    return source, pages[0]['page']


def _local(program, source, page, old):
    print_id = program.get('diagnostic', {}).get('physical_print_id')
    if (not print_id or print_id != source.get('diagnostic', {}).get('physical_print_id')
            or not old.get('pattern_id') or old['pattern_id'] != page.get('pattern_id')):
        raise ValueError('writing reference belongs to a different physical print')
    if np.linalg.norm(_centre(program) - _centre(source)) > .045:
        raise ValueError('writing reference is outside its 45 mm local region')
    old_overhead = _pose(old['locate']['overhead'])
    overhead = _pose(page['locate']['overhead'])
    angle = np.arccos(np.clip((np.trace(old_overhead[:3, :3].T @ overhead[:3, :3]) - 1) / 2, -1, 1))
    if np.linalg.norm(old_overhead[:3, 3] - overhead[:3, 3]) > .003 or angle > .01:
        raise ValueError('writing reference is stale: tracked page moved over 3 mm or 0.01 rad')


def _height(reference, old, page, pen):
    old_overhead = _pose(old['locate']['overhead'])
    overhead = _pose(page['locate']['overhead'])
    old_pen = old['pen']
    if (old_pen['mode'] != 'ride' or not np.isclose(old_pen['stroke_m'], pen.stroke_m)
            or not 0 < old_pen['height_m'] < old_pen['stroke_m']):
        raise ValueError('writing reference requires an ordinary riding source height')
    old_used, used = _pose(old['used']), _pose(page['used'])
    target = float((old_used[:3, 3] - old_overhead[:3, 3]) @ old_overhead[:3, 2]) + old_pen['height_m'] + reference['lift_m']
    current = float((used[:3, 3] - overhead[:3, 3]) @ overhead[:3, 2])
    height = max(target - current, pen.height_m)
    increase = height - pen.height_m
    if not np.isfinite(height) or not 0 <= increase <= .003 or height >= .010:
        raise ValueError('writing reference requires an upward buffer adjustment within 3 mm and below 10 mm hover')
    return dataclasses.replace(pen, height_m=height), {
        **reference, 'minimum_normal_height_over_overhead_m': target,
        'normal_height_over_overhead_m': current + height,
        'ordinary_pen_height_m': pen.height_m, 'adjustment_m': increase, 'height_m': height}
