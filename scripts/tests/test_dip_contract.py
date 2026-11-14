"""A dipping tool's settings have every field, finite and in range; a measured ink's line per dip wins."""
import math

import pytest
from tatbot_contracts.dip import DRAIN_ABOVE_RIM_M, NONNEGATIVE, POSITIVE, drain_s, resolve_dip, validate_dip


@pytest.fixture
def block():
    return dict.fromkeys(POSITIVE + NONNEGATIVE, 0.001) | {'mm_per_dip': 40.0, 'mm_per_dip_by_ink': {'red': 25.0}}


def test_every_field_is_required(block):
    settings = resolve_dip(block, 'black')
    assert settings['mm_per_dip'] == 40.0 and resolve_dip(block, 'red')['mm_per_dip'] == 25.0
    for key in POSITIVE + NONNEGATIVE:
        with pytest.raises(ValueError, match=key):
            validate_dip({name: value for name, value in settings.items() if name != key})
    with pytest.raises(ValueError, match='no dip: block'):
        resolve_dip(None, 'black')


@pytest.mark.parametrize('value', [True, None, -1, math.nan, math.inf, '0.001'])
def test_values_are_finite_numbers(block, value):
    for key in ('above_ink_m', 'hover_m', 'mm_per_dip'):
        with pytest.raises(ValueError, match='finite'):
            validate_dip(resolve_dip(block, 'black') | {key: value})


def test_margins_may_be_zero_and_lengths_may_not(block):
    validate_dip(resolve_dip(block, 'black') | {'above_ink_m': 0, 'dwell_s': 0, 'wall_margin_m': 0})
    with pytest.raises(ValueError, match='positive'):
        validate_dip(resolve_dip(block, 'black') | {'hover_m': 0})


def test_the_3rl_datasheet_resolves():
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / 'scripts' / 'lib'))
    import tool_spec

    sheet = tool_spec.load_tool('lutin-3rl-bugpin', repo).raw
    settings = resolve_dip(sheet['dip'], 'nighthawk_black')
    assert 0 < settings['above_ink_m'] < 0.002 and settings['mm_per_dip'] > 0


def test_a_drain_is_whole_or_absent_and_holds_between_the_rim_and_the_hover():
    """drain_s 0, the datasheets' default, is the dip as it was: neither drain key reaches the program."""
    block = {'above_ink_m': .001, 'dwell_s': .5, 'hover_m': .008, 'speed_m_s': .01, 'wall_margin_m': .001,
             'mm_per_dip': 40.0, 'drain_s': 0.0, 'drain_above_rim_m': .003}
    assert not {'drain_s', 'drain_above_rim_m'} & set(resolve_dip(block, 'black'))
    assert drain_s(resolve_dip(block, 'black')) == 0.0
    drained = resolve_dip({**block, 'drain_s': 1.5}, 'black')
    assert (drained['drain_s'], drained['drain_above_rim_m']) == (1.5, .003) and drain_s(drained) == 1.5
    del block['drain_above_rim_m']   # a datasheet without a height drains at the contract's
    assert resolve_dip({**block, 'drain_s': 1.5}, 'black')['drain_above_rim_m'] == DRAIN_ABOVE_RIM_M == .003
    for change, message in (({'drain_above_rim_m': None}, 'drain_above_rim_m must be'),
                            ({'drain_above_rim_m': .009}, 'at most hover_m'), ({'drain_s': -1}, 'drain_s must be')):
        with pytest.raises(ValueError, match=message):
            resolve_dip({**block, 'drain_s': 1.5, **change}, 'black')
    for partial in ({'drain_s': 1.5}, {'drain_above_rim_m': .003}, {'drain_s': 0, 'drain_above_rim_m': .003}):
        with pytest.raises(ValueError, match='a dip that drains has'):
            validate_dip({k: v for k, v in drained.items() if k not in ('drain_s', 'drain_above_rim_m')} | partial)
