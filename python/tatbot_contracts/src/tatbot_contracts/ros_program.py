"""tatbot-program version 2: what tatbot_ink prepares and the ROS session executes.

A program is one arm's drawing on one page: its resources (a tool, an ink and how the ink is supplied) and its ordered
ops. `tool_change` puts a resource on the arm (the first, `initial`, is the one the run starts with), `dip` refills a
dipping resource at its palette cap, and `stroke` is one pen-down polyline in page metres with the resource that draws
it and its source in the acquired artwork (src, arc_m). validate_for_execution refuses what the executor cannot run:
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
DIAGNOSTIC = 'native-diagnostic-to-ros/2'   # directly authored calibration shapes, not DBV3 artwork


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
    if program['arm'] not in ('left', 'right'):
        raise ValueError('the program names no arm')
    _number(program['draw_speed_m_s'], 'draw_speed_m_s')
    page = program['page']
    if page['kind'] not in ('plane', 'stencil') or any(c > s for c, s in zip(page['clear_m'], page['size_m'], strict=True)):
        raise ValueError('the page must be flat with its clear area inside it')
    resources = {}
    for resource in program['resources']:
        if _id(resource['id'], 'resource id') in resources:
            raise ValueError(f"resource {resource['id']} is declared twice")
        _resource(resource)
        resources[resource['id']] = resource
    _writing_reference(program)
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
            fitted = resource['id']
        elif resource['id'] != fitted:
            raise ValueError(f"op {op['id']} uses {resource['id']}, but its tool change fitted {fitted}")
        elif op['op'] == 'dip':
            if resource['dip'] is None or op['slot'] != resource['slot']:
                raise ValueError(f"dip {op['id']}: {resource['id']} dips at no cap or not at {op['slot']}")
        else:
            _stroke(op, resource, page)


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
    radius = max(resource['tool'].get('line_width_m') or 0., op['generation_width_m']) / 2
    if any(abs(v) + radius > c / 2 + 1e-9 for p in points for v, c in zip(p, page['clear_m'], strict=True)):
        raise ValueError(f"stroke {op['id']}: its line leaves the page's clear area")


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
