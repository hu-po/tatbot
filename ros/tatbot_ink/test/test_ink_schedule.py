"""Material boundaries and timed chunks preserve traversal and physical contact semantics."""
from __future__ import annotations

import numpy as np
import pytest
from ink_designs import element
from resource_fixtures import SPEED, drawing, fixture_repo, manifest, resource
from tatbot_ink.errors import CompileError
from tatbot_ink.place import placed_strokes
from tatbot_ink.resources import bind_resources
from tatbot_ink.schedule import physical_sections, schedule_ops
from tatbot_ink.segment import arc_lengths, resample
from tatbot_motion import load_motion
from tatbot_motion.estimate import drawing_seconds, estimate_duration


def schedule(tmp_path, layers, resources, assignments, *, substrate='paper', max_seconds=60):
    doc = drawing(layers)
    repo = fixture_repo(tmp_path / 'repo')
    path = manifest(tmp_path / 'inks.yaml', doc, resources, assignments, substrate=substrate)
    table, bindings, _ = bind_resources(path, doc, repo=repo, arm='right', speed_m_s=SPEED)
    widths = {key: next(r['tool']['line_width_m'] for r in table if r['id'] == identity)
              for key, identity in bindings.items()}
    strokes, _ = placed_strokes(doc, tool_widths=widths)
    return schedule_ops(strokes, bindings, table, speed=SPEED, max_seconds=max_seconds, motion=load_motion()), table, strokes


def test_ballpoint_sequence_preserves_repeats_and_reversals_without_dips(tmp_path):
    layers = [(pen, [element([[.002, y], [.018, y]])])
              for pen, y in [('black', .002), ('red', .01), ('black', .018)]]
    layers[-1][1][0]['points_m'].reverse()
    ops, table, strokes = schedule(tmp_path, layers, [resource('B'), resource('R', ink='red')],
                                  [('black', 'B'), ('red', 'R')])
    assert [op['op'] for op in ops] == ['tool_change', 'stroke'] * 3
    assert [op['resource_id'] for op in ops] == ['B', 'B', 'R', 'R', 'B', 'B']
    assert [op['initial'] for op in ops if op['op'] == 'tool_change'] == [True, False, False]
    actual = [op for op in ops if op['op'] == 'stroke']
    for op, source in zip(actual, strokes, strict=True):
        np.testing.assert_array_equal(op['points_m'], source.points_m)
        assert op['src']['path'] == source.src['path'] and not op['continues']
    assert actual[-1]['points_m'][0][0] > actual[-1]['points_m'][-1][0]
    timing = estimate_duration(ops, SPEED, 0, load_motion(), resources=table)
    assert timing['unknown_operations'] == {'pen_change': 3}
    assert timing['total_s'] is None  # initial identity/setup may require a physical intervention


def test_grouping_change_invalidates_same_ink_charge_and_dips_again(tmp_path):
    line = [[.002, .01], [.007, .01]]
    layers = [(pen, [element(line)]) for pen in ('three', 'five', 'three')]
    ops, table, _ = schedule(tmp_path, layers, [resource('N3', tool='round-3', mode='ride'),
                                               resource('N5', tool='round-5', mode='ride')],
                             [('three', 'N3'), ('five', 'N5')], substrate='practice')
    assert [op['op'] for op in ops] == ['tool_change', 'dip', 'stroke'] * 3
    assert [op['reason'] for op in ops if op['op'] == 'dip'] == ['initial'] * 3
    assert [op['resource_id'] for op in ops if op['op'] == 'dip'] == ['N3', 'N5', 'N3']
    assert all(op['ink'] == 'black' for op in ops if op['op'] == 'stroke')
    assert estimate_duration(ops, SPEED, 0, load_motion(), resources=table)['components_s']['settle'] == 0


def test_replenishment_favors_an_existing_path_boundary(tmp_path):
    paths = [element([[.002, y], [.008, y]]) for y in (.002, .004)]  # 6 mm each, capacity 10 mm
    ops, _, _ = schedule(tmp_path, [('pen', paths)], [resource('N', tool='round-3')],
                         [('pen', 'N')], substrate='practice')
    assert [op['op'] for op in ops] == ['tool_change', 'dip', 'stroke', 'dip', 'stroke']
    assert [op['reason'] for op in ops if op['op'] == 'dip'] == ['initial', 'replenish']
    assert all(op['src']['arc_m'] == pytest.approx([0, .006]) for op in ops if op['op'] == 'stroke')
    assert all(not op['continues'] for op in ops if op['op'] == 'stroke')


def test_long_closed_path_cuts_replenishment_before_computational_chunks(tmp_path):
    path = element([[.002, .002], [.018, .002], [.018, .018], [.002, .018]], closed=True)
    ops, table, sources = schedule(tmp_path, [('pen', [path])], [resource('N', tool='round-3')],
                                   [('pen', 'N')], substrate='practice', max_seconds=1)
    original = sources[0].points_m
    strokes = [op for op in ops if op['op'] == 'stroke']
    assert len(strokes) > 10 and all(not op['closed'] and op['src']['closed'] for op in strokes)
    assert strokes[0]['src']['arc_m'][0] == 0
    assert strokes[-1]['src']['arc_m'][1] == pytest.approx(arc_lengths(original)[-1])
    assert strokes[0]['points_m'][0] == strokes[-1]['points_m'][-1]
    contact = 0
    previous = None
    for op in ops[1:]:
        if op['op'] == 'dip':
            assert contact <= .010 + 1e-12
            contact = 0
        else:
            lo, hi = op['src']['arc_m']
            contact += hi - lo
            assert op['continues'] == (previous is not None and previous['op'] == 'stroke')
            np.testing.assert_allclose([op['points_m'][0], op['points_m'][-1]], resample(original, [lo, hi]), atol=1e-14)
            assert drawing_seconds(op, SPEED, load_motion()) <= 1 + 1e-10
        previous = op
    assert contact <= .010 + 1e-12
    for a, b in zip(strokes, strokes[1:], strict=False):
        assert a['src']['arc_m'][1] == pytest.approx(b['src']['arc_m'][0], abs=1e-14)
        np.testing.assert_allclose(a['points_m'][-1], b['points_m'][0], atol=1e-14)
    estimate = estimate_duration(ops, SPEED, 0, load_motion(), resources=table)
    touches = sum(not op['continues'] for op in strokes)
    assert estimate['components_s']['settle'] == touches * load_motion()['pen']['press']['settle_s']


def test_closed_path_below_capacity_keeps_one_contact_and_chunking_adds_no_dips(tmp_path):
    path = element([[.002, .002], [.003, .002], [.003, .003]], closed=True)
    ops, _, _ = schedule(tmp_path, [('pen', [path])], [resource('N', tool='round-3')],
                         [('pen', 'N')], substrate='practice')
    assert [op['op'] for op in ops] == ['tool_change', 'dip', 'stroke']
    assert ops[-1]['closed']
    ops, _, _ = schedule(tmp_path, [('pen', [path])], [resource('B')], [('pen', 'B')], max_seconds=.2)
    assert all(op['op'] == 'stroke' for op in ops[1:]) and len(ops) > 2
    assert [op['continues'] for op in ops[1:]] == [False] + [True] * (len(ops) - 2)


@pytest.mark.parametrize('remaining', [None, 0, .004, .010])
def test_each_physical_section_obeys_capacity_and_preserves_all_source_vertices(remaining):
    points = np.array([[0, 0], [.005, 0], [.008, .004], [.012, .007], [.017, .007], [.017, .007]])
    before = points.copy()
    sections, credit = physical_sections(points, capacity_m=.010, remaining_m=remaining)
    assert 0 <= credit <= .010
    assert sum(arc[1]-arc[0] for _, arc, _ in sections) == pytest.approx(arc_lengths(points)[-1])
    for piece, (lo, hi), dip in sections:
        assert hi - lo <= .010 + 1e-12
        if lo > 0 or remaining is None or remaining < .010:
            assert dip
        np.testing.assert_allclose([piece[0], piece[-1]], resample(points, [lo, hi]), atol=1e-14)
    np.testing.assert_array_equal(points, before)
    for vertex in points:
        assert any(np.linalg.norm(piece-vertex, axis=1).min() < 1e-14 for piece, _, _ in sections)


def test_residual_charge_carries_across_complete_paths():
    points = [[0, 0], [.003, 0]]
    first, remaining = physical_sections(points, capacity_m=.010, remaining_m=None)
    second, remaining = physical_sections(points, capacity_m=.010, remaining_m=remaining)
    assert first[0][2] and not second[0][2] and remaining == pytest.approx(.004)
    third, remaining = physical_sections(points, capacity_m=.010, remaining_m=remaining)
    assert not third[0][2] and remaining == pytest.approx(.001)


@pytest.mark.parametrize(('points', 'capacity', 'remaining'), [
    ([], .01, None), ([0, 1], .01, None), ([[0, 0], [0, 0]], .01, None),
    ([[0, 0], [float('nan'), 1]], .01, None), ([[0, 0], [1, 0]], 0, None),
    ([[0, 0], [1, 0]], float('inf'), None), ([[0, 0], [1, 0]], .01, -.01),
    ([[0, 0], [1, 0]], .01, .02), ([[0, 0], [1, 0]], .01, float('nan')),
])
def test_invalid_material_inputs_refused(points, capacity, remaining):
    with pytest.raises(CompileError):
        physical_sections(points, capacity_m=capacity, remaining_m=remaining)
