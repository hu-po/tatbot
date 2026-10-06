"""`tatbot ros palette status|load` on the ROS owner: what each cap holds and its measured ink level, in the
owner's config/palette_load.yaml (which deploy leaves alone). A dip reads the level to set its depth; nothing here
estimates a volume or moves the arm.

    load M1=nighthawk_black --level-mm M1=6   # an ink cap, its ink this high over the cap's inner floor
    load L1=none S1=absent S2=unknown         # an empty cap, no cap, contents not known
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import ink_spec

ALIASES = {alias: 'inkcap_'+name for alias, name in (
    ('L1', 'large_1'), ('M1', 'medium_1'), ('S1', 'small_1'),
    ('S2', 'small_2'), ('M2', 'medium_2'), ('L2', 'large_2'))}


def _assignments(values, palette):
    out = {}
    for value in values:
        slot, separator, content = value.partition('=')
        slot = ALIASES.get(slot, slot)
        if not separator or not content or slot not in palette or slot in out:
            raise ValueError(f'{value!r}: use SLOT=VALUE once per slot, SLOT a cap or one of {", ".join(ALIASES)}')
        out[slot] = content
    return out


def status(repo):
    return {'ok': True, 'slots': {key: {'ink_id': value.ink_id, 'cap_present': value.cap_present,
                                        'level_lower_bound_m': value.level_lower_bound_m,
                                        'fill_ul': value.fill_ul if value.fill_known else None}
                                  for key, value in ink_spec.load_palette_load(repo).items()}}


def declare(repo, values, levels):
    palette = ink_spec.load_palette(repo)
    inks = ink_spec.load_inks(repo)
    assigned, measured = _assignments(values, palette), _assignments(levels, palette)
    if not assigned or set(measured) - set(assigned):
        raise ValueError('declare at least one cap; a --level-mm belongs to a cap declared with it')
    load = ink_spec.load_palette_load(repo, palette)
    for slot, ink in assigned.items():
        ink_id = None if ink in ('unknown', 'absent', 'none') else ink
        level = None
        if ink_id is not None:
            if ink_id not in inks or slot not in measured:
                raise ValueError(f'{slot}: an ink cap needs a catalog ink and its --level-mm {slot}=HEIGHT')
            level = float(measured[slot]) / 1000
            if not math.isfinite(level) or not 0 < level < palette[slot].size.depth_m:
                raise ValueError(f'{slot}: the level is the ink\'s height over the inner floor, under the rim')
        elif slot in measured:
            raise ValueError(f'{slot}: an unknown, absent or empty cap has no ink level')
        load[slot] = ink_spec.SlotLoad(slot, ink_id, utc=ink_spec.utc_stamp(),
                                       cap_present=None if ink == 'unknown' else ink != 'absent',
                                       level_lower_bound_m=level, fill_known=ink == 'none')
    ink_spec.write_palette_load(load, repo)
    return status(repo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('status', 'load'))
    parser.add_argument('caps', nargs='*', metavar='SLOT=INK')
    parser.add_argument('--level-mm', action='append', default=[], metavar='SLOT=HEIGHT')
    ns = parser.parse_args(argv)
    try:
        repo = Path(os.environ['TATBOT_REPO'])
        if ns.action == 'status' and (ns.caps or ns.level_mm):
            raise ValueError('cap declarations and levels belong to palette load')
        result = declare(repo, ns.caps, ns.level_mm) if ns.action == 'load' else status(repo)
    except (ValueError, KeyError) as error:
        print(json.dumps({'ok': False, 'message': str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
