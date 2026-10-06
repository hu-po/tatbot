"""The program the ROS session executes: what validate_for_execution refuses before anything moves."""
import copy
import math

import pytest
import tatbot_cli  # noqa: F401 -- shared bare-clone contract source root
from ros_program_fixtures import program
from tatbot_contracts.ros_program import FORMAT, VERSION, validate_for_execution


def test_a_one_pen_program_with_the_fitted_tool_is_admitted():
    value = program()
    validate_for_execution(value)
    assert (FORMAT, VERSION) == ('tatbot-program', 2)
    assert value['resources'][0]['rgb'] is None


@pytest.mark.parametrize('version', [None, True, '2', 0, 1, 3])
def test_another_version_is_refused(version):
    value = program()
    value['version'] = version
    with pytest.raises(ValueError, match='unsupported.*prepare it again'):
        validate_for_execution(value)


@pytest.mark.parametrize('value', [None, [], {'format': 'another-program'}])
def test_unknown_format_refused(value):
    with pytest.raises(ValueError, match='expected a tatbot-program'):
        validate_for_execution(value)


CHANGES = {
    'missing_resource': ('names no resource', lambda value: value.update(resources=[])),
    'first_stroke': ('initial tool change', lambda value: value['ops'].pop(0)),
    'missing_initial': ('initial tool change', lambda value: value['ops'][0].pop('initial')),
    'another_resource': ('names no resource', lambda value: value['ops'][1].update(resource_id='other')),
    'duplicate_op': ('twice', lambda value: value['ops'][1].update(id=value['ops'][0]['id'])),
    'nonfinite': ('finite 2D polyline', lambda value: value['ops'][1]['points_m'][0].__setitem__(0, float('nan'))),
    'off_page': ('clear area', lambda value: value['ops'][1]['points_m'][0].__setitem__(0, 1.)),
    'dip_without_profile': ('dips at no cap', lambda value: value['ops'].insert(
        1, {'op': 'dip', 'id': 'd0001', 'resource_id': 'fitted', 'slot': 'cap'})),
    'pause': ('only tool_change, dip and stroke', lambda value: value['ops'].insert(
        1, {'op': 'pause', 'id': 'p0001', 'resource_id': 'fitted'})),
    'ride_without_stroke': ("riding tool's stroke_mm", lambda value: value['resources'][0].update(pen_mode='ride')),
    'slot_without_dipping': ('only a dipping tool', lambda value: value['resources'][0].update(slot='inkcap_medium_1')),
    'lift': ('at most 3 mm', lambda value: value.update(
        writing_height_reference={'run_id': 'run', 'page_sha256': 'a'*64, 'lift_m': .004})),
}


@pytest.mark.parametrize('change', sorted(CHANGES))
def test_a_program_the_executor_cannot_run_is_refused(change):
    value = program()
    message, edit = CHANGES[change]
    edit(value)
    with pytest.raises(ValueError, match=message):
        validate_for_execution(value)


def test_a_needle_cartridge_rides_on_its_stroke_alone():
    """Its touches and the wrist gauge meet the tube's end, so the ride needs no tip standing out at the top."""
    value = program()
    value['resources'][0]['tool'].update(stroke_mm=4.0, contact_reference='tube_end')
    value['resources'][0]['pen_mode'] = 'ride'
    validate_for_execution(value)
    value['resources'][0]['tool']['contact_reference'] = 'seated_ball'
    with pytest.raises(ValueError, match="tip_out_at_top_mm"):
        validate_for_execution(value)


def test_a_rinse_activation_is_admitted_only_whole_and_for_a_resource_that_dips():
    value = program()
    resource = value['resources'][0]
    resource['tool']['ink'] = 'dip'
    resource.update(slot='inkcap_medium_2', dip={'above_ink_m': .0005, 'dwell_s': 1.5, 'hover_m': .008,
                                                 'speed_m_s': .01, 'wall_margin_m': .0015, 'mm_per_dip': 40.})
    resource['activation'] = {'method': 'rinse', 'slot': 'inkcap_large_1', 'ink_id': 'water', 'dwell_s': 10.,
                              'above_ink_m': 0.}
    validate_for_execution(value)
    resource['activation']['dwell_s'] = 0
    with pytest.raises(ValueError, match='rinse dwell_s must be'):
        validate_for_execution(value)
    del resource['activation']['dwell_s']
    with pytest.raises(ValueError, match='exactly'):
        validate_for_execution(value)
    resource.update(slot=None, dip=None, activation={**resource['activation'], 'dwell_s': 10.})
    with pytest.raises(ValueError, match='only a resource that dips'):
        validate_for_execution(value)


def _diagnostic(*paths):
    value = program()
    template = value['ops'].pop()
    value['preparation']['adapter'] = 'native-diagnostic-to-ros/2'
    for index, points in enumerate(paths):
        stroke = copy.deepcopy(template)
        stroke.update(id=f's{index:04d}', points_m=points, closed=points[0] == points[-1])
        stroke['src']['arc_m'] = [0., sum(math.dist(a, b) for a, b in zip(points, points[1:], strict=False))]
        value['ops'].append(stroke)
    return value


@pytest.mark.parametrize('points', [
    [[0., 0.], [.004, .005], [.005, 0.]],
    [[0., 0.], [.005, 0.], [.005, .002], [0., .002], [0., 0.]],
])
def test_finished_geometry_cannot_pass_as_a_native_diagnostic(points):
    with pytest.raises(ValueError, match='analytic square, cross, line or ladder'):
        validate_for_execution(_diagnostic(points))


@pytest.mark.parametrize('paths', [
    [[[0., 0.], [.005, 0.]]],
    [[[0., 0.], [.005, 0.], [.005, .005], [0., .005], [0., 0.]]],
    [[[-.005, 0.], [.005, 0.]], [[0., -.005], [0., .005]]],
])
def test_analytic_line_square_and_cross_are_admitted(paths):
    validate_for_execution(_diagnostic(*paths))


def _ladder(fractions=(.9, 1., 1.1)):
    """Equal parallel rungs, each its own resource of the same tool and ink at its own ride fraction."""
    value = _diagnostic(*([[0., .003 * i], [.005, .003 * i]] for i in range(len(fractions))))
    initial, strokes = value['ops'][0], value['ops'][1:]
    value['resources'] = [dict(copy.deepcopy(value['resources'][0]), id=f'h{i}', ride_fraction=f)
                          for i, f in enumerate(fractions)]
    value['ops'] = [dict(initial, resource_id='h0')]
    for i, stroke in enumerate(strokes):
        if i:
            value['ops'].append(dict(initial, id=f't{i:04d}', resource_id=f'h{i}', initial=False))
        value['ops'].append(dict(stroke, resource_id=f'h{i}'))
    return value


def test_a_height_ladder_is_a_native_diagnostic_of_one_tool_and_ink():
    validate_for_execution(_ladder())
    other_ink = _ladder()
    other_ink['resources'][1]['ink_id'] = 'another_ink'
    with pytest.raises(ValueError, match='fitted tool and ink'):
        validate_for_execution(other_ink)
    with pytest.raises(ValueError, match=r'ride_fraction must be in \(0, 2\)'):
        validate_for_execution(_ladder((.9, 2.5, 1.1)))

