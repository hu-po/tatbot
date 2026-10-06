"""How a dipping tool takes ink at a palette cap: its datasheet's `dip:` block, resolved for one ink.

The tool's end comes down the cap's axis to `above_ink_m` over the declared ink level and holds there `dwell_s`
(a needle cartridge's machine running, its needles pulling ink up into the tube), `hover_m` over the rim before
and after, at `speed_m_s` along the axis, keeping `wall_margin_m` from the bore. `mm_per_dip` is the line one dip
draws: the datasheet's `mm_per_dip_by_ink` for the ink when measured, else its `mm_per_dip`.

A resource whose activation is a rinse takes the cartridge from the last ink to its own without a landing: one dip
into a cap of water, its machine running `dwell_s` with the tube's end `above_ink_m` over the declared level (the
datasheet's `rinse_dwell_s` and `rinse_above_ink_m`).
"""
from __future__ import annotations

import math

POSITIVE = ('hover_m', 'speed_m_s', 'mm_per_dip')
NONNEGATIVE = ('above_ink_m', 'dwell_s', 'wall_margin_m')


def validate_dip(settings) -> None:
    """ValueError unless `settings` has every field, finite and in range."""
    missing = set(POSITIVE + NONNEGATIVE) - set(settings if isinstance(settings, dict) else ())
    if missing:
        raise ValueError(f'dip settings need {", ".join(sorted(missing))}')
    for key in POSITIVE + NONNEGATIVE:
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key in POSITIVE and value == 0):
            raise ValueError(f'dip {key} must be a finite {"positive" if key in POSITIVE else "nonnegative"} number')


def resolve_dip(block, ink_id: str) -> dict:
    """The datasheet's `dip:` block for one ink, validated; ValueError when the tool declares none."""
    if not isinstance(block, dict):
        raise ValueError('the tool datasheet declares no dip: block')
    measured = block.get('mm_per_dip_by_ink') or {}
    settings = {key: block.get(key) for key in POSITIVE + NONNEGATIVE}
    settings['mm_per_dip'] = measured.get(ink_id, block.get('mm_per_dip'))
    validate_dip(settings)
    return settings


RINSE = ('method', 'slot', 'ink_id', 'dwell_s', 'above_ink_m')


def resolve_rinse(block, slot: str, ink_id: str) -> dict:
    """A rinse activation in `slot`'s `ink_id` with the datasheet `dip:` block's rinse settings, validated."""
    block = block if isinstance(block, dict) else {}
    activation = {'method': 'rinse', 'slot': slot, 'ink_id': ink_id,
                  'dwell_s': block.get('rinse_dwell_s'), 'above_ink_m': block.get('rinse_above_ink_m')}
    validate_rinse(activation)
    return activation


def validate_rinse(activation) -> None:
    """ValueError unless a rinse names its cap and liquid and dwells a positive time at or over the liquid."""
    if not isinstance(activation, dict) or set(activation) != set(RINSE) or activation['method'] != 'rinse':
        raise ValueError(f'a rinse activation has exactly {", ".join(RINSE)}')
    for key in ('slot', 'ink_id'):
        if not isinstance(activation[key], str) or not activation[key]:
            raise ValueError(f'a rinse names its {key}')
    for key in ('dwell_s', 'above_ink_m'):
        value = activation[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == 'dwell_s' and value == 0):
            raise ValueError(f'rinse {key} must be a finite {"positive" if key == "dwell_s" else "nonnegative"} number '
                             f'(the datasheet dip: rinse_{key})')
