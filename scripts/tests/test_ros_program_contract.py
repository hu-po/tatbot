"""The program the ROS session executes: what validate_for_execution refuses before anything moves."""
import copy
import math

import pytest
import tatbot_cli  # noqa: F401 -- shared bare-clone contract source root
from ros_program_fixtures import program
from tatbot_contracts.ros_program import FORMAT, VERSION, started_at, validate_for_execution


def test_a_one_pen_program_with_the_fitted_tool_is_admitted():
    value = program()
    validate_for_execution(value)
    assert (FORMAT, VERSION) == ('tatbot-program', 2)
    assert value['resources'][0]['rgb'] is None


@pytest.mark.parametrize('arm', ['right', 'xarm'])
def test_a_program_names_an_arm_the_stack_drives(arm):
    value = program()
    value['arm'] = arm
    validate_for_execution(value)
    value['arm'] = 'left'   # the blue arm, the teleop leader alone
    with pytest.raises(ValueError, match='names no drawing arm'):
        validate_for_execution(value)


@pytest.mark.parametrize('version', [None, True, '2', 0, 1, 3])
def test_another_version_is_refused(version):
    value = program()
    value['version'] = version
    with pytest.raises(ValueError, match='unsupported.*prepare it again'):
        validate_for_execution(value)


def test_a_fading_stroke_is_drawn_by_a_tool_whose_fade_lift_is_known():
    value = program()
    stroke = next(op for op in value['ops'] if op['op'] == 'stroke')
    stroke['deposition'] = [1.0, 0.0]   # a whip: full at its start, lifted out at its end
    with pytest.raises(ValueError, match='fade_lift_mm'):
        validate_for_execution(value)
    value['resources'][0]['tool']['fade_lift_mm'] = 0.8
    validate_for_execution(value)
    for bad in ([1.0], [1.0, 1.5], [1.0, -0.1], 0.5):
        stroke['deposition'] = bad
        with pytest.raises(ValueError, match='one deposition'):
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
    'plane_page': ('stencil print', lambda value: value['page'].update(kind='plane')),
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
    value['resources'][0]['tool']['contact_reference'] = 'mechanical_nose'
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


def test_a_program_may_start_its_run_at_a_trim_that_is_a_finite_number():
    value = program()
    started = started_at(value, -0.00175)
    assert started["start_trim_m"] == -0.00175 and "start_trim_m" not in value
    validate_for_execution(started)
    with pytest.raises(ValueError, match="start_trim_m must be"):
        validate_for_execution({**value, "start_trim_m": math.nan})
    with pytest.raises(ValueError, match="start_trim_m must be"):
        validate_for_execution({**value, "start_trim_m": "deep"})


def test_a_program_may_say_how_its_runs_stroke_ends_are_drawn():
    from tatbot_contracts.ros_program import ends_drawn

    value = program()
    flown = ends_drawn(value, "fly")
    assert flown["draw_ends"] == "fly" and "draw_ends" not in value and flown["ops"] == value["ops"]
    assert ends_drawn(flown, "stop")["draw_ends"] == "stop"
    with pytest.raises(ValueError, match="draw_ends must be one of stop, fly"):
        ends_drawn(value, "whip")


def test_a_rinse_may_give_the_new_inks_first_dip_its_own_dwell():
    value = program()
    resource = value['resources'][0]
    resource['tool']['ink'] = 'dip'
    resource.update(slot='inkcap_medium_2', dip={'above_ink_m': .0005, 'dwell_s': 1.5, 'hover_m': .008,
                                                 'speed_m_s': .01, 'wall_margin_m': .0015, 'mm_per_dip': 40.})
    resource['activation'] = {'method': 'rinse', 'slot': 'inkcap_large_1', 'ink_id': 'water', 'dwell_s': 30.,
                              'above_ink_m': 0., 'first_dip_dwell_s': 3.}
    validate_for_execution(value)
    resource['activation']['first_dip_dwell_s'] = 0
    with pytest.raises(ValueError, match='first_dip_dwell_s must be'):
        validate_for_execution(value)
    resource['activation'] = {**resource['activation'], 'first_dip_dwell_s': 3., 'second_dip_dwell_s': 2.}
    with pytest.raises(ValueError, match='exactly'):
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



def test_a_rehearsal_is_the_same_strokes_in_the_air_with_no_dips(monkeypatch):
    """ros_program.rehearsal: every resource hovers (the machine off) at the given height, its cap and dips gone and
    its rinse a manual change, one tool's resources one ink (no landing between them), the source's digest kept."""
    from tatbot_contracts.ros_program import program_sha256, rehearsal

    value = program()
    dipping = copy.deepcopy(value['resources'][0])
    dipping['tool'].update(id='needles', ink='dip', stroke_mm=4.0, contact_reference='tube_end')
    dip = {'mm_per_dip': 100.0}
    value['resources'] = [{**dipping, 'id': 'black', 'ink_id': 'black', 'pen_mode': 'ride', 'slot': 'cap1',
                           'dip': dip, 'ride_fraction': 0.875},
                          {**dipping, 'id': 'grey', 'ink_id': 'grey', 'pen_mode': 'ride', 'slot': 'cap2', 'dip': dip}]
    stroke = {**value['ops'][1], 'resource_id': 'black'}
    value['ops'] = [{**value['ops'][0], 'resource_id': 'black'}, {'op': 'dip', 'id': 'd0001', 'resource_id': 'black',
                    'slot': 'cap1'}, stroke, {**value['ops'][0], 'id': 't0002', 'initial': False, 'resource_id': 'grey'},
                    {**stroke, 'id': 's0004', 'resource_id': 'grey'}]
    import tatbot_contracts.ros_program as contract
    monkeypatch.setattr(contract, 'validate_dip', lambda dip: None)   # the profile's checks are the dip contract's
    contract.validate_for_execution(value)
    out = rehearsal(value, 0.006)
    assert [op['op'] for op in out['ops']] == ['tool_change', 'stroke', 'tool_change', 'stroke']
    assert all(r['pen_mode'] == 'hover' and r['dip'] is None and r['slot'] is None and 'ride_fraction' not in r
               and r['hover_height_m'] == 0.006 and r['ink_id'] == 'black' for r in out['resources'])
    assert out['rehearsal'] == {'of': program_sha256(value), 'height_m': 0.006}
    assert [op['points_m'] for op in out['ops'] if op['op'] == 'stroke'] == [stroke['points_m']] * 2
    assert value['resources'][0]['pen_mode'] == 'ride'                   # the source is untouched
    assert rehearsal(value)['resources'][0].get('hover_height_m') is None
    with pytest.raises(ValueError, match='hover_height_m'):
        rehearsal(value, 0.001)
    dry = rehearsal(value, 0.006, 'cap9')   # dry dips (tatbot_session.cap): the dips kept, all into one empty cap
    assert [op['op'] for op in dry['ops']] == ['tool_change', 'dip', 'stroke', 'tool_change', 'stroke']
    assert [op['slot'] for op in dry['ops'] if op['op'] == 'dip'] == ['cap9']
    assert all(r['pen_mode'] == 'hover' and r['slot'] == 'cap9' and r['dip'] == dip for r in dry['resources'])
    assert dry['rehearsal'] == {'of': program_sha256(value), 'height_m': 0.006, 'dip_slot': 'cap9'}
    assert 'stats' not in value or dry['stats']['dips'] == 1
    real = rehearsal(value, 0.006, real_dips=True)   # the dips practised over empty caps: each into its own
    assert [op['op'] for op in real['ops']] == ['tool_change', 'dip', 'stroke', 'tool_change', 'stroke']
    assert [(r['slot'], r['ink_id'], r['dip_in_cap']) for r in real['resources']] == [('cap1', 'black', True),
                                                                                       ('cap2', 'grey', True)]
    assert all(r['pen_mode'] == 'hover' and 'ride_fraction' not in r for r in real['resources'])
    assert real['rehearsal'] == {'of': program_sha256(value), 'height_m': 0.006, 'real_dips': True}
    with pytest.raises(ValueError, match='not both'):
        rehearsal(value, 0.006, 'cap9', real_dips=True)
