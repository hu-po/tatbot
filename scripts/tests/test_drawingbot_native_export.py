"""Metric/traversal qualification for the bounded native DBV3 export decoder."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest
from drawingbot.artifacts import normalize_svg
from drawingbot.native_export import decode_export
from drawingbot.native_path import Budget, decode_path, distance_to_segment

SCALE = .0254 / 96


def exported(path='M10 20 L30 40', *, after='', style='', transform='matrix(1,0,0,1,0,0)'):
    return normalize_svg(f'''<svg xmlns="http://www.w3.org/2000/svg" width="50mm" height="75mm">
      <g id="Black"><g transform="{transform}" style="stroke:black;stroke-width:1.88976378;
      stroke-linecap:round;stroke-linejoin:round;{style}">
      <path d="{path}" style="fill:none;"/>{after}</g></g></svg>'''.encode())[0]


def decode(svg, **kwargs):
    return decode_export(svg, chord_error_m=5e-6, pen_width_m=.0005, **kwargs)


def elements(geometry):
    return [element for layer in geometry['layers'] for element in layer['elements']]


def test_physical_page_transform_and_width_are_separate():
    geometry, audit = decode(exported())
    assert geometry['canvas_m'] == {'width': .05, 'height': .075}
    line, = elements(geometry)
    assert line['points_m'][0] == pytest.approx([10*SCALE, .075 - 20*SCALE])
    assert line['points_m'][1] == pytest.approx([30*SCALE, .075 - 40*SCALE])
    assert line['width_m'] == .0005
    assert audit['pen_widths'][0]['export_width_m'] == pytest.approx(.0005, abs=1e-10)
    assert audit['pen_widths'][0]['pen_name'] == 'Black'


def test_nested_transforms_compose_without_refitting_ink_bounds():
    svg = exported(style='stroke-width:1.88976378;', transform='matrix(0,-1,1,0,50,80)')
    geometry, _ = decode(svg)
    assert elements(geometry)[0]['points_m'][0] == pytest.approx([70*SCALE, .075 - 70*SCALE])


def test_order_direction_subpaths_closure_and_repeat_passes_survive():
    path = 'M30 40 L10 20 M70 80 L60 50 L40 60 Z'
    geometry, audit = decode(exported(path, after=f'<path d="{path}" style="fill:none;"/>'))
    lines = elements(geometry)
    assert len(lines) == 4
    assert [p['closed'] for p in lines] == [False, True, False, True]
    assert [p['id'] for p in lines] == ['path-1-1', 'path-1-2', 'path-2-1', 'path-2-2']
    assert lines[0]['points_m'] == lines[2]['points_m']
    assert lines[0]['points_m'][0][0] > lines[0]['points_m'][1][0]
    assert audit['raw_path_count'] == 2


def test_native_pen_groups_stay_in_export_order_including_same_color():
    svg = exported()
    svg = svg.replace('</svg>', '<g id="Another_black"><g style="stroke:black;stroke-width:1.88976378;stroke-linecap:round;stroke-linejoin:round"><path d="M90 30L10 80" style="fill:none;"/></g></g></svg>')
    geometry, _ = decode(svg)
    assert [p['id'] for p in geometry['inks']] == ['pen-1', 'pen-2']
    assert [p['ink_id'] for p in geometry['layers']] == ['pen-1', 'pen-2']
    assert len(elements(geometry)) == 2


def _curve(controls, t):
    points = controls
    while len(points) > 1:
        points = [[(1-t)*a[i] + t*b[i] for i in (0, 1)] for a, b in zip(points, points[1:], strict=False)]
    return points[0]


@pytest.mark.parametrize('command,controls', [
    ('C1 1 2 -1 3 0', [(0, 0), (1, 1), (2, -1), (3, 0)]),
    ('Q1 2 3 0', [(0, 0), (1, 2), (3, 0)]),
    ('C1 0 -1 0 .5 0', [(0, 0), (1, 0), (-1, 0), (.5, 0)]),
])
def test_curves_have_bounded_error_and_preserve_collinear_reversals(command, controls):
    tolerance = .00005
    def metric(p):
        return tuple(v / 1000 for v in p)
    paths = decode_path('M0 0 ' + command, metric, tolerance, Budget())
    line = paths[0]['points_m']
    for i in range(1001):
        point = metric(_curve(controls, i/1000))
        error = min(distance_to_segment(point, a, b) for a, b in zip(line, line[1:], strict=False))
        assert error <= tolerance
    assert line[0] == [0., 0.]
    assert line[-1] == list(metric(controls[-1]))
    if command.startswith('C1 0'):
        assert any(b[0] < a[0] for a, b in zip(line, line[1:], strict=False))
        expected = sum(math.dist(_curve(controls, i/10000), _curve(controls, (i+1)/10000)) for i in range(10000))/1000
        assert sum(math.dist(a, b) for a, b in zip(line, line[1:], strict=False)) == pytest.approx(expected, abs=2*tolerance)


@pytest.mark.parametrize('path', ['M0 0 A1 1 0 0 0 20 30', 'm0 0l10 20', 'M0 0 C10 20',
                                 'M0 0 L1e999 0', 'M0 0L0 0', 'M0 0', 'M0 0L10 10ZL20 20',
                                 'M0 0 M10 10 L20 20', 'M0 0Z'])
def test_unsupported_or_empty_traversal_is_never_silently_dropped(path):
    with pytest.raises(ValueError):
        decode(exported(path))


@pytest.mark.parametrize('change', [
    lambda s: s.replace('</svg>', '<image href="external.png"/></svg>'),
    lambda s: s.replace('fill:none', 'fill:black'),
    lambda s: s.replace('fill:none', 'fill:none;opacity:0.5'),
    lambda s: s.replace('stroke:black', 'stroke:url(#gradient)'),
    lambda s: s.replace('matrix(1,0,0,1,0,0)', 'matrix(1,0,0,2,0,0)'),
    lambda s: s.replace('M10 20 L30 40', 'M-100 20 L30 40'),
    lambda s: s.replace('stroke-width:1.88976378', 'stroke-width:4'),
    lambda s: s.replace('<path ', '<path onclick="ignored()" '),
    lambda s: s.replace('<g id="Black">', '<svg>'),
])
def test_unknown_paint_geometry_and_dimensions_fail_acquisition(change):
    with pytest.raises((ValueError, ET.ParseError)):
        decode(change(exported()))


def test_subdivision_is_bounded_and_refuses_instead_of_relaxing_error():
    with pytest.raises(ValueError, match='point budget'):
        decode(exported('M10 20 C180 20 10 200 180 200'), max_points=8)


def test_points_already_on_the_canvas_are_not_scaled_to_fit():
    svg = exported('M0 0 L188.9763779527559 283.46456692913387')
    geometry, _ = decode(svg)
    assert elements(geometry)[0]['points_m'][1] == pytest.approx([.05, 0], abs=1e-14)
