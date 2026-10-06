"""A local writing lift keeps the fresh page touches and cannot move the path down."""
import copy
import dataclasses
import json

import numpy as np
import pytest
from ros_program_fixtures import program as fixture_program
from tatbot_cli.ros_writing_height import resolve
from tatbot_contracts.ros_program import program_sha256, validate_for_execution


@dataclasses.dataclass(frozen=True)
class Pen:
    height_m: float = .003325
    stroke_m: float = .0035
    machine: bool = True


def freeze(value):
    validate_for_execution(value)
    return value


@pytest.fixture
def trial(tmp_path):
    source = fixture_program()
    r = source['resources'][0]
    r['pen_mode'] = 'ride'
    r['tool'].update(stroke_mm=3.5, tip_out_at_top_mm=2.)
    source['diagnostic'] = {'physical_print_id': 'print'}
    freeze(source)
    workspace = {'tool_id': 'lutin-ballpoint-dot', 'pen_tip_offset_z': .1}
    overhead, used = np.eye(4), np.eye(4)
    used[2, 3] = .0082
    page = {'locate': {'overhead': overhead.tolist()}, 'used': used.tolist(), 'pattern_id': 'pattern',
            'pen': {'mode': 'ride', 'height_m': .003325, 'stroke_m': .0035}}
    rows = [{'arm': 'right', 'event': 'page', 'page': copy.deepcopy(page), 'workspace': workspace},
            {'arm': 'right', 'event': 'done', 'op': source['ops'][1]['id']}]
    directory = tmp_path/'source'
    directory.mkdir()
    (directory/'program.json').write_text(json.dumps(source))
    target = copy.deepcopy(source)
    target['writing_height_reference'] = {'run_id': 'source', 'page_sha256': program_sha256(page), 'lift_m': .002}
    freeze(target)
    return {'program': target, 'arm': 'right', 'run_dir': tmp_path/'next', 'page': copy.deepcopy(page),
            'pen': Pen(), 'workspace': workspace, 'rows': rows}


def run(trial):
    rows = trial.pop('rows')
    path = trial['run_dir'].parent/'source'/'ledger.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return resolve(**trial)


def test_lower_fresh_touch_does_not_cancel_operator_lift(trial):
    trial['page']['used'][2][3] -= .0006
    pen, evidence = run(trial)
    assert pen.height_m == pytest.approx(.005925)
    assert trial['page']['used'][2][3] + pen.height_m == pytest.approx(.0082 + .003325 + .002)
    assert evidence['adjustment_m'] == pytest.approx(.0026)


def test_working_page_snapshot_cannot_replace_immutable_reference(trial):
    (trial['run_dir'].parent/'source'/'page.json').write_text('{"used": "corrupted"}')
    assert run(trial)[0].height_m == pytest.approx(.005325)


def test_higher_fresh_plane_keeps_ordinary_path_above_requested_minimum(trial):
    trial['page']['used'][2][3] += .0025
    pen, evidence = run(trial)
    assert pen.height_m == pytest.approx(.003325)
    assert evidence['adjustment_m'] == 0
    assert evidence['normal_height_over_overhead_m'] > evidence['minimum_normal_height_over_overhead_m']


@pytest.mark.parametrize('change', ['too_far_up', 'moved', 'rotated', 'distance', 'pose', 'off'])
def test_invalid_or_stale_geometry_refuses(trial, change):
    if change == 'too_far_up':
        trial['page']['used'][2][3] -= .0015
    elif change == 'moved':
        trial['page']['locate']['overhead'][0][3] = .004
    elif change == 'rotated':
        a = .02
        trial['page']['locate']['overhead'] = [[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    elif change == 'distance':
        source_path = trial['run_dir'].parent/'source'/'program.json'
        source = json.loads(source_path.read_text())
        for pt in source['ops'][1]['points_m']:
            pt[0] -= .026
        freeze(source)
        source_path.write_text(json.dumps(source))
        for pt in trial['program']['ops'][1]['points_m']:
            pt[0] += .021
        freeze(trial['program'])
    elif change == 'pose':
        trial['page']['used'][0][0] = 2
    else:
        trial['pen'] = Pen(machine=False)
    with pytest.raises(ValueError):
        run(trial)


@pytest.mark.parametrize('change', ['workspace', 'print', 'pattern', 'digest', 'sent'])
def test_changed_identity_or_incomplete_source_refuses(trial, change):
    if change == 'workspace':
        trial['workspace'] = {**trial['workspace'], 'pen_tip_offset_z': .2}
    elif change == 'print':
        trial['program']['diagnostic']['physical_print_id'] = 'other'
    elif change == 'pattern':
        trial['page']['pattern_id'] = 'other'
    elif change == 'digest':
        trial['rows'][0]['page']['used'][2][3] += .001
    else:
        trial['rows'].append({'arm': 'right', 'event': 'sent', 'op': trial['program']['ops'][1]['id']})
    with pytest.raises(ValueError):
        run(trial)


@pytest.mark.parametrize('change', ['lift', 'negative', 'path', 'field', 'digest'])
def test_reference_is_bounded(trial, change):
    p = trial['program']
    ref = p['writing_height_reference']
    if change == 'lift':
        ref['lift_m'] = .004
    elif change == 'negative':
        ref['lift_m'] = -.001
    elif change == 'path':
        ref['run_id'] = '../source'
    elif change == 'field':
        ref['override'] = True
    else:
        ref['page_sha256'] = 'invalid'
    with pytest.raises(ValueError):
        validate_for_execution(p)
