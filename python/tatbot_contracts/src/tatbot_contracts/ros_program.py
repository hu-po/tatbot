"""tatbot-program version 2: what tatbot_ink prepares and the ROS session executes.

A program is one arm's drawing on one page: its resources (a tool, an ink and how the ink is supplied) and its ordered
ops. `tool_change` puts a resource on the arm (the first, `initial`, is the one the run starts with), `dip` refills a
dipping resource at its palette cap, and `stroke` is one pen-down polyline in page metres with the resource that draws
it and its source in the acquired artwork (src, arc_m), with a `deposition` per point when it fades (a build before
deposition ignores it and draws the stroke at full depth). validate_for_execution refuses what the executor cannot run:
an unknown op or resource, an op whose resource is not the one its tool change fitted, a stroke off the page's clear
area, a dip without its cap and profile, a writing lift over 3 mm. How the compiler got there (bindings, source arcs,
provenance) is tatbot_ink's to test, not the robot's to re-check.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from .dip import validate_dip, validate_rinse

FORMAT = 'tatbot-program'
VERSION = 2
OPS = ('tool_change', 'dip', 'stroke')
ARMS = ('right', 'xarm')   # the arms the ROS stack drives (tatbot_description.names.ARMS; a test keeps them equal)
DIAGNOSTIC = 'native-diagnostic-to-ros/2'   # directly authored calibration shapes, not DBV3 artwork


ENDS = ('stop', 'fly')   # how a stroke's ends are drawn (motion.yaml draw.ends; a program's draw_ends over it)


HOVER_M = (0.003, 0.020)   # a rehearsal's hover over the measured page (the session also keeps a needle's reach clear)


def rehearsal(program, height_m=None, dip_slot=None, real_dips=False):
    """The same program drawn in the air: every resource's pen held `height_m` (else motion.yaml pen.hover.height_m)
    over the measured page with the tattoo machine off (pen_mode hover), no palette cap and no dip op, so the page
    setup, the gauge holds, the hover stops, the stencil's follow and every stroke run as they will with ink, and
    nothing touches the skin. Resources of one tool become one ink (it carries none), so their tool changes go on
    without landing; a writing height reference is dropped. With `dip_slot` (a palette cap) every dip op stays and
    every dipping resource dips at that cap, dry (tatbot_session.cap: to its hover and back, the machine off). With
    `real_dips` every resource keeps its own cap, ink and rinse, and each dip and rinse goes into its cap as it will
    with ink (`dip_in_cap`), the machine off: the dips practised over empty caps. The source program's digest is
    kept as `rehearsal.of`; validated as any program."""
    import copy

    if real_dips and dip_slot is not None:
        raise ValueError('a rehearsal dips dry at one cap or into each resource\'s own, not both')
    out = copy.deepcopy(program)
    first = out['resources'][0] if out.get('resources') else None
    for resource in out.get('resources', []):
        _hover(resource, height_m)
        if real_dips:
            resource['dip_in_cap'] = resource.get('dip') is not None
        else:
            _without_ink(resource, first, dip_slot)
    if not real_dips:
        out['ops'] = [{**op, 'slot': dip_slot} if op['op'] == 'dip' else op for op in out['ops']
                      if op['op'] != 'dip' or dip_slot is not None]
    out.pop('writing_height_reference', None)   # a writing reference sets the pen's height from a run: not a hover's
    for op in out['ops']:
        if op['op'] == 'tool_change' and 'activation' in op:
            op['activation'] = next(r['activation'] for r in out['resources'] if r['id'] == op['resource_id'])
    if isinstance(out.get('stats'), dict):
        out['stats'] = {**out['stats'], 'dips': sum(op['op'] == 'dip' for op in out['ops'])}
    out['rehearsal'] = {'of': program_sha256(program), 'height_m': None if height_m is None else float(height_m),
                        **({'dip_slot': dip_slot} if dip_slot is not None else {}),
                        **({'real_dips': True} if real_dips else {})}
    validate_for_execution(out)
    return out


def started_at(program, trim_m):
    """The program with its run starting every cartridge at `trim_m`, the operator's pen trim without the race for the
    keys between the page set-up and the first stroke: the executor writes it as a pen_trim row, which a resume keeps."""
    import copy

    out = copy.deepcopy(program)
    out['start_trim_m'] = float(trim_m)
    validate_for_execution(out)
    return out


def ends_drawn(program, ends):
    """The program with its run drawing every stroke's ends `ends` (ENDS), whatever the stack's motion.yaml draw.ends
    says: `tatbot ros draw --ends`. Its stats, the dose among them, stay as it was prepared."""
    import copy

    out = copy.deepcopy(program)
    out['draw_ends'] = ends
    validate_for_execution(out)
    return out


def _hover(resource, height_m):
    """A rehearsal's resource: its pen held over the page, at `height_m` when given, the machine off."""
    resource['pen_mode'] = 'hover'
    resource.pop('ride_fraction', None)
    if height_m is not None:
        resource['hover_height_m'] = float(height_m)


def _without_ink(resource, first, dip_slot):
    """No cap but `dip_slot`'s (dry dips) and no rinse, and the ink of the first resource on the same tool, so a
    rehearsal's tool changes between them go on without landing."""
    dry = dip_slot is not None and resource.get('dip') is not None
    resource.update(slot=dip_slot if dry else None, dip=resource['dip'] if dry else None)
    if (resource.get('activation') or {}).get('method') == 'rinse':
        resource['activation'] = {'method': 'manual', **resource['tool'].get('cartridge_activation',
                                                                             {'action': 'exchange'})}
    if first is not None and resource['tool']['id'] == first['tool']['id']:
        resource['ink_id'] = first['ink_id']


def cartridge(resource) -> str:
    """The cartridge a resource draws with, which owns the operator's pen trim: its tool. Every colour a rinse takes
    one cartridge to shares it, so a depth trim carries across them (2026-11-02). A resource with no tool
    record is its own."""
    return (resource.get("tool") or {}).get("id") or resource["id"]


def program_sha256(program):
    """The run's program digest; refuses nonfinite values rather than hash them."""
    return hashlib.sha256(json.dumps(program, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_for_execution(program):
    if not isinstance(program, dict) or program.get('format') != FORMAT:
        raise ValueError('expected a tatbot-program')
    if type(program.get('version')) is not int or program['version'] != VERSION:
        raise ValueError(f'unsupported tatbot-program version {program.get("version")!r}; this stack executes '
                         f'version {VERSION}: prepare it again')
    try:
        _validate(program)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f'malformed program ({type(error).__name__}: {error})') from error


def _number(value, name, *, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'{name} must be a finite {"positive" if positive else "nonnegative"} number')
    return value


def _id(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}', value):
        raise ValueError(f'{name} must be an identifier')
    return value


def _validate(program):
    if program['arm'] not in ARMS:
        raise ValueError('the program names no drawing arm')
    _number(program['draw_speed_m_s'], 'draw_speed_m_s')
    page = program['page']
    if page['kind'] != 'stencil' or any(c > s for c, s in zip(page['clear_m'], page['size_m'], strict=True)):
        raise ValueError('the page must be a stencil print with its clear area inside it: prepare it again')
    resources = {}
    for resource in program['resources']:
        if _id(resource['id'], 'resource id') in resources:
            raise ValueError(f"resource {resource['id']} is declared twice")
        _resource(resource)
        resources[resource['id']] = resource
    _writing_reference(program)
    if 'start_trim_m' in program:
        start = program['start_trim_m']
        if type(start) not in (int, float) or not math.isfinite(start):
            raise ValueError('start_trim_m must be a finite number')
    if program.get('draw_ends', ENDS[0]) not in ENDS:
        raise ValueError(f"draw_ends must be one of {', '.join(ENDS)}")
    _ops(program['ops'], resources, page)
    if program['preparation']['adapter'] == DIAGNOSTIC:
        first = resources[program['ops'][0]['resource_id']]
        if any(op['op'] == 'dip' or (resources[op['resource_id']]['tool']['id'], resources[op['resource_id']]['ink_id'])
               != (first['tool']['id'], first['ink_id']) for op in program['ops']):
            raise ValueError('a native diagnostic is the fitted tool and ink, and strokes only')
        _diagnostic_shape([op for op in program['ops'] if op['op'] == 'stroke'])


def _ops(ops, resources, page):
    if not ops or ops[0]['op'] != 'tool_change' or not ops[0].get('initial'):
        raise ValueError('a program starts with its initial tool change')
    seen, fitted = set(), None
    for op in ops:
        if _id(op['id'], 'op id') in seen:
            raise ValueError(f"op {op['id']} appears twice")
        seen.add(op['id'])
        if op['op'] not in OPS:
            raise ValueError(f"op {op['id']}: version 2 executes only tool_change, dip and stroke operations")
        if op['resource_id'] not in resources:
            raise ValueError(f"op {op['id']} names no resource of the program")
        resource = resources[op['resource_id']]
        if op['op'] == 'tool_change':
            _wipe(op, resource, None if fitted is None else resources[fitted])
            fitted = resource['id']
        elif resource['id'] != fitted:
            raise ValueError(f"op {op['id']} uses {resource['id']}, but its tool change fitted {fitted}")
        elif op['op'] == 'dip':
            if resource['dip'] is None or op['slot'] != resource['slot']:
                raise ValueError(f"dip {op['id']}: {resource['id']} dips at no cap or not at {op['slot']}")
        else:
            _stroke(op, resource, page)


def _wipe(op, resource, previous):
    """A tool change may carry `wipe: true`, the operator wiping the print before the resource draws: only where the
    arm lands for it, a manual exchange to another tool or ink."""
    if 'wipe' not in op:
        return
    lands = (previous is not None and (resource.get('activation') or {}).get('method') == 'manual'
             and (previous['tool']['id'], previous['ink_id']) != (resource['tool']['id'], resource['ink_id']))
    if op['wipe'] is not True or not lands:
        raise ValueError(f"tool change {op['id']}: a wipe is `wipe: true` on a manual exchange to another tool or ink, "
                         "where the arm lands")


def riding_keys(tool) -> tuple[str, ...]:
    """The datasheet facts a riding tool needs: its stroke, and how far its tip stands out of its tube at the top of
    the stroke, unless the touches and the wrist gauge meet the tube's end (a needle cartridge)."""
    return ('stroke_mm',) if tool.get('contact_reference') == 'tube_end' else ('stroke_mm', 'tip_out_at_top_mm')


def _resource(resource):
    tool = resource['tool']
    _id(tool['id'], 'tool id')
    if resource['pen_mode'] not in ('press', 'ride', 'hover'):
        raise ValueError(f"{resource['id']}: pen_mode must be press, ride or hover")
    if resource['pen_mode'] == 'ride':
        for key in riding_keys(tool):
            _number(tool[key], f"{resource['id']}: a riding tool's {key}")
    if 'ride_fraction' in resource and not 0 < _number(resource['ride_fraction'], 'ride_fraction') < 2:
        raise ValueError(f"{resource['id']}: ride_fraction must be in (0, 2)")
    if 'hover_height_m' in resource and (resource['pen_mode'] != 'hover' or not
                                         HOVER_M[0] <= _number(resource['hover_height_m'], 'hover_height_m') <= HOVER_M[1]):
        raise ValueError(f"{resource['id']}: hover_height_m is a hover resource's, {HOVER_M[0] * 1e3:g}-"
                         f"{HOVER_M[1] * 1e3:g} mm")
    if tool['ink'] == 'dip' and (resource['slot'] is not None or resource['dip'] is not None):
        _id(resource['slot'], f"{resource['id']}: palette slot")
        validate_dip(resource['dip'])
    elif resource['slot'] is not None or resource['dip'] is not None:
        raise ValueError(f"{resource['id']}: only a dipping tool takes a palette slot and profile")
    if (resource.get('activation') or {}).get('method') == 'rinse':
        if resource['dip'] is None:
            raise ValueError(f"{resource['id']}: only a resource that dips can rinse into it")
        validate_rinse(resource['activation'])


def _stroke(op, resource, page):
    points = op['points_m']
    if len(points) < 2 or any(len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in p)
                              for p in points):
        raise ValueError(f"stroke {op['id']} needs a finite 2D polyline")
    if 'deposition' in op:
        _deposition(op, resource)
    radius = max(resource['tool'].get('line_width_m') or 0., op['generation_width_m']) / 2
    if any(abs(v) + radius > c / 2 + 1e-9 for p in points for v, c in zip(p, page['clear_m'], strict=True)):
        raise ValueError(f"stroke {op['id']}: its line leaves the page's clear area")


def _deposition(op, resource):
    """A fading stroke: one share of full ink per point, in [0, 1], drawn by a tool whose fade lift is known (its
    datasheet's fade_lift_mm: deposition 0 rides that much higher)."""
    shares = op['deposition']
    if not isinstance(shares, list) or len(shares) != len(op['points_m']) or any(
            type(v) not in (int, float) or not 0 <= v <= 1 for v in shares):
        raise ValueError(f"stroke {op['id']} needs one deposition in [0, 1] per point")
    _number(resource['tool'].get('fade_lift_mm'), f"{resource['id']}: the fade_lift_mm of a tool drawing deposition")


def _writing_reference(program):
    reference = program.get('writing_height_reference')
    if reference is None:
        return
    if set(reference) != {'run_id', 'page_sha256', 'lift_m'} or not re.fullmatch('[0-9a-f]{64}', reference['page_sha256']):
        raise ValueError('writing_height_reference needs run_id, page_sha256 and lift_m')
    _id(reference['run_id'], 'writing reference run')
    if _number(reference['lift_m'], 'writing reference lift') > .003:
        raise ValueError('a writing reference lifts the pen at most 3 mm')


def _diagnostic_shape(strokes):
    """Only analytic line, square, centred perpendicular cross, or ladder of equal parallel lines; never artwork."""
    points = [op['points_m'] for op in strokes]
    if len(points) == 1 and len(points[0]) == 2 and not strokes[0]['closed']:
        return
    if len(points) > 2 and all(len(line) == 2 and not op['closed'] for line, op in zip(points, strokes, strict=True)):
        edges = [[b-a for a, b in zip(*line, strict=True)] for line in points]
        length = math.hypot(*edges[0])
        if length > 0 and all(math.isclose(math.hypot(*e), length, rel_tol=0, abs_tol=1e-9)
                              and abs(e[0] * edges[0][1] - e[1] * edges[0][0]) <= 1e-9 * length ** 2 for e in edges):
            return
    if len(points) == 1 and len(points[0]) == 5 and points[0][0] == points[0][-1]:
        edges = [[b[axis]-a[axis] for axis in range(2)] for a, b in zip(points[0], points[0][1:], strict=False)]
        lengths = [math.hypot(*edge) for edge in edges]
        if (lengths[0] > 0 and all(math.isclose(length, lengths[0], rel_tol=0, abs_tol=1e-9) for length in lengths)
                and all(abs(sum(a*b for a, b in zip(edge, edges[(i+1) % 4], strict=True))) <= 1e-9 * lengths[i] * lengths[(i+1) % 4]
                        for i, edge in enumerate(edges))):
            return
    if len(points) == 2 and all(len(line) == 2 and not op['closed'] for line, op in zip(points, strokes, strict=False)):
        centres = [[(a+b)/2 for a, b in zip(*line, strict=False)] for line in points]
        edges = [[b-a for a, b in zip(*line, strict=False)] for line in points]
        if (all(math.hypot(*edge) > 0 for edge in edges)
                and all(abs(a-b) < 1e-9 for a, b in zip(*centres, strict=False))
                and abs(sum(a*b for a, b in zip(*edges, strict=True))) <= 1e-9 * math.hypot(*edges[0]) * math.hypot(*edges[1])):
            return
    raise ValueError('native diagnostic accepts only an analytic square, cross, line or ladder; acquire finished '
                     'artwork with DBV3')
