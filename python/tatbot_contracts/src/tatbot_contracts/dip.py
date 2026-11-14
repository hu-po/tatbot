"""How a dipping tool takes ink at a palette cap: its datasheet's `dip:` block, resolved for one ink.

The tool's end comes down the cap's axis to `above_ink_m` over the declared ink level and holds there `dwell_s`
(a needle cartridge's machine running, its needles pulling ink up into the tube), `hover_m` over the rim before
and after, at `speed_m_s` along the axis, keeping `wall_margin_m` from the bore. `mm_per_dip` is the line one dip
draws: the datasheet's `mm_per_dip_by_ink` for the ink when measured, else its `mm_per_dip`.

A drain is a hold after the dwell, the machine still running, with the tool's end `drain_above_rim_m` over the cap's
rim for `drain_s`: what the needles carry beyond the tube's hold falls back into the cap, not onto the skin. A
datasheet that states no `drain_s` does not drain, and one that states no `drain_above_rim_m` drains
DRAIN_ABOVE_RIM_M over the rim. A dip without a drain carries neither key, so it is the dip it was before drains
existed.

A resource whose activation is a rinse takes the cartridge from the last ink to its own without a landing: one dip
into a cap of water, its machine running `dwell_s` with the tube's end `above_ink_m` over the declared level (the
datasheet's `rinse_dwell_s` and `rinse_above_ink_m`). Its `first_dip_dwell_s`, when the datasheet gives
`rinse_first_dip_dwell_x`, is that many ink dwells: the new ink's first dip after the rinse holds it, so the rinsed
cartridge takes up the colour (2026-11-02).
"""
from __future__ import annotations

import math

POSITIVE = ('hover_m', 'speed_m_s', 'mm_per_dip')
NONNEGATIVE = ('above_ink_m', 'dwell_s', 'wall_margin_m')
DRAIN = ('drain_s', 'drain_above_rim_m')   # both or neither; a dip with neither does not drain
# Where a drain holds when the datasheet gives no height: with the tube's end 3 mm over the rim the running needles'
# tips, 4 mm past it on every fitted cartridge (needle_reach_mm), still stroke inside the bore, so what they shed
# lands in the cap. Unmeasured: the first drain ladder (docs/lab.md) is its test.
DRAIN_ABOVE_RIM_M = 0.003


def validate_dip(settings) -> None:
    """ValueError unless `settings` has every field, finite and in range, and a drain that is whole."""
    missing = set(POSITIVE + NONNEGATIVE) - set(settings if isinstance(settings, dict) else ())
    if missing:
        raise ValueError(f'dip settings need {", ".join(sorted(missing))}')
    drain = tuple(key for key in DRAIN if key in settings)
    for key in POSITIVE + NONNEGATIVE + drain:
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key in POSITIVE and value == 0):
            raise ValueError(f'dip {key} must be a finite {"positive" if key in POSITIVE else "nonnegative"} number')
    if not drain:
        return
    if drain != DRAIN or settings['drain_s'] == 0:
        raise ValueError('a dip that drains has a positive drain_s and its drain_above_rim_m; one that does not has '
                         'neither')
    if settings['drain_above_rim_m'] > settings['hover_m']:
        raise ValueError('dip drain_above_rim_m is at most hover_m: the drain holds between the rim and the hover')


def drain_s(settings) -> float:
    """How long a dip drains over its cap after the dwell; 0 for a dip that does not."""
    return float(settings.get('drain_s', 0.0))


def resolve_dip(block, ink_id: str) -> dict:
    """The datasheet's `dip:` block for one ink, validated; ValueError when the tool declares none."""
    if not isinstance(block, dict):
        raise ValueError('the tool datasheet declares no dip: block')
    measured = block.get('mm_per_dip_by_ink') or {}
    settings = {key: block.get(key) for key in POSITIVE + NONNEGATIVE}
    settings['mm_per_dip'] = measured.get(ink_id, block.get('mm_per_dip'))
    if block.get('drain_s'):   # 0 or absent is the dip without a drain, which carries neither key
        settings.update(drain_s=block['drain_s'], drain_above_rim_m=block.get('drain_above_rim_m', DRAIN_ABOVE_RIM_M))
    validate_dip(settings)
    return settings


RINSE = ('method', 'slot', 'ink_id', 'dwell_s', 'above_ink_m')
RINSE_OPTIONAL = ('first_dip_dwell_s',)


def resolve_rinse(block, slot: str, ink_id: str) -> dict:
    """A rinse activation in `slot`'s `ink_id` with the datasheet `dip:` block's rinse settings, validated."""
    block = block if isinstance(block, dict) else {}
    activation = {'method': 'rinse', 'slot': slot, 'ink_id': ink_id,
                  'dwell_s': block.get('rinse_dwell_s'), 'above_ink_m': block.get('rinse_above_ink_m')}
    times = block.get('rinse_first_dip_dwell_x')
    if times is not None:
        if type(times) not in (int, float) or not math.isfinite(times) or times < 1:
            raise ValueError('the datasheet dip: rinse_first_dip_dwell_x must be a finite number of ink dwells, 1 or more')
        if type(block.get('dwell_s')) not in (int, float):
            raise ValueError('the datasheet dip: rinse_first_dip_dwell_x needs its dwell_s')
        activation['first_dip_dwell_s'] = block['dwell_s'] * times
    validate_rinse(activation)
    return activation


def validate_rinse(activation) -> None:
    """ValueError unless a rinse names its cap and liquid and dwells a positive time at or over the liquid."""
    if (not isinstance(activation, dict) or not set(RINSE) <= set(activation) <= set(RINSE + RINSE_OPTIONAL)
            or activation['method'] != 'rinse'):
        raise ValueError(f'a rinse activation has exactly {", ".join(RINSE)}, and may have {", ".join(RINSE_OPTIONAL)}')
    for key in ('slot', 'ink_id'):
        if not isinstance(activation[key], str) or not activation[key]:
            raise ValueError(f'a rinse names its {key}')
    for key in ('dwell_s', 'above_ink_m'):
        value = activation[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == 'dwell_s' and value == 0):
            raise ValueError(f'rinse {key} must be a finite {"positive" if key == "dwell_s" else "nonnegative"} number '
                             f'(the datasheet dip: rinse_{key})')
    first = activation.get('first_dip_dwell_s')
    if first is not None and (type(first) not in (int, float) or not math.isfinite(first) or first <= 0):
        raise ValueError('rinse first_dip_dwell_s must be a finite positive number (the datasheet dip: '
                         'rinse_first_dip_dwell_x times its dwell_s)')
