"""A dipping tool's settings have every field, finite and in range; a measured ink's line per dip wins."""
import math

import pytest
from tatbot_contracts.dip import NONNEGATIVE, POSITIVE, resolve_dip, validate_dip


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
