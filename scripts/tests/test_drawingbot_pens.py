"""Pen identity, native readback and per-path widths cross the acquisition boundary."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import tatbot_cli  # noqa: F401 -- shared bare-clone contract source root
from drawingbot.artifacts import digest, normalize_svg
from drawingbot.job import load_job
from drawingbot.native_export import decode_export
from drawingbot.pens import native_drawing_sets, validate_drawing_set, verify_pen_readback
from drawingbot.recipe import DEFAULTS, read_json, runtime_project


def drawing_set():
    value = read_json(DEFAULTS / 'drawing-set.json')
    value['pens'] += [{**value['pens'][0], 'id': 'another-black', 'stroke_factor': 2.},
                      {**value['pens'][0], 'id': 'unused', 'weight': 0},
                      {**value['pens'][0], 'id': 'disabled', 'enabled': False}]
    return value


def readback(requested):
    return {**requested, 'native_set_id': 0,
            'pens': [{**pen, 'name': pen['id'], 'native_row_id': row,
                      'export_group': f"Ballpoint_{pen['id']}"} for row, pen in enumerate(requested['pens'])]}


def svg():
    groups = [('Ballpoint_another-black', 3.77952756), ('Ballpoint_black', 1.88976378)]
    xml = '<svg xmlns="http://www.w3.org/2000/svg" width="50mm" height="75mm">'
    for identity, width in groups:
        xml += (f'<g id="{identity}" style="stroke:black;stroke-width:{width};stroke-linecap:round;'
                'stroke-linejoin:round"><path d="M10 20L30 40" style="fill:none;"/></g>')
    return normalize_svg((xml + '</svg>').encode())[0]


def test_native_transport_preserves_requested_labels_and_plain_fixed_colors(tmp_path):
    requested = drawing_set()
    before = copy.deepcopy(requested)
    native = native_drawing_sets(requested)
    assert [p['name'] for p in native['drawingSets'][0]['pens']] == ['black', 'another-black', 'unused', 'disabled']
    assert [p['argb'] for p in native['drawingSets'][0]['pens']] == ['-16777216'] * 4
    assert [p['penNumber'] for p in native['drawingSets'][0]['pens']] == ['0', '1', '2', '3']
    path = runtime_project(DEFAULTS / 'project.json', tmp_path / 'source.png', tmp_path / 'runtime.json', drawing_set=requested)
    assert read_json(path)['data']['settings']['drawing_sets'] == native
    assert requested == before


def test_export_order_does_not_assign_identity_and_equal_colors_do_not_merge():
    requested = drawing_set()
    effective = verify_pen_readback(requested, readback(requested), .5)
    geometry, audit = decode_export(svg(), chord_error_m=5e-6, pen_width_m=.0005, pens=effective['pens'])
    assert [layer['ink_id'] for layer in geometry['layers']] == ['another-black', 'black']
    assert [layer['elements'][0]['width_m'] for layer in geometry['layers']] == [.001, .0005]
    assert [(row['id'], row['raw_path_count'], row['exported']) for row in audit['pens']] == [
        ('black', 1, True), ('another-black', 1, True), ('unused', 0, False), ('disabled', 0, False)]
    assert all(row['requested_name'] == 'Black' for row in effective['pens'])


def test_pinned_native_multicolor_export_has_verified_identity_widths_and_replay(tmp_path):
    fixture = Path(__file__).with_name('fixtures') / 'dbv3-multicolor'
    record = read_json(fixture / 'fixture.json')
    assert record['software']['version'] == '1.6.22'
    assert record['replay_exact_matches'] == {'normalized.svg': True}
    assert digest(fixture / 'normalized.svg') == record['normalized_sha256']
    # The native input is a generated RGB contour fixture, bound to exact bytes.
    from drawingbot.artifacts import write_json
    write_json(tmp_path / 'job.json', {**record['job'], 'source': {**record['job']['source'],
                        'file': str(fixture / 'source.png')}})
    job, _ = load_job(tmp_path / 'job.json')
    effective = record['effective']
    assert verify_pen_readback(job['drawing_set'], effective['drawing_set'], job['pen_width_mm']) == effective['drawing_set']
    geometry, audit = decode_export((fixture / 'normalized.svg').read_text(), chord_error_m=5e-6,
                                   pen_width_m=.0005, pens=effective['drawing_set']['pens'])
    assert [row['id'] for row in geometry['inks']] == ['red', 'black-wide', 'black']
    assert [layer['ink_id'] for layer in geometry['layers']] == ['red', 'black-wide', 'black']
    assert [layer['elements'][0]['width_m'] for layer in geometry['layers']] == [.00075, .001, .0005]
    assert {row['id']: row['raw_path_count'] for row in audit['pens']} == {
        'black': 1, 'red': 2, 'black-wide': 2, 'unused': 0, 'disabled': 0}


@pytest.mark.parametrize('field,value', [('id', 'black'), ('rgba', [0, 0, 0, 128]),
    ('weight', -1), ('enabled', 1), ('stroke_factor', float('nan')), ('stroke_factor', 100)])
def test_unsupported_or_ambiguous_requests_fail_before_native_launch(field, value):
    requested = drawing_set()
    requested['pens'][1][field] = value
    with pytest.raises(ValueError):
        validate_drawing_set(requested, .5)


@pytest.mark.parametrize('field,value', [('name', 'wrong'), ('native_row_id', 9), ('enabled', False),
    ('rgba', [255, 0, 0, 255]), ('weight', 9), ('stroke_factor', 3), ('export_group', 'Ballpoint_black')])
def test_effective_pen_drift_is_refused(field, value):
    requested = drawing_set()
    actual = readback(requested)
    actual['pens'][1][field] = value
    with pytest.raises(RuntimeError):
        verify_pen_readback(requested, actual, .5)


@pytest.mark.parametrize('change', [
    lambda s: s.replace('Ballpoint_black', 'unknown'),
    lambda s: s.replace('Ballpoint_another-black', 'Ballpoint_disabled'),
    lambda s: s.replace('Ballpoint_another-black', 'Ballpoint_black'),
    lambda s: s.replace('stroke:black', 'stroke:rgb(255,0,0)'),
    lambda s: s.replace('stroke-width:3.77952756', 'stroke-width:1.88976378'),
])
def test_export_must_match_native_color_width_and_identity(change):
    requested = drawing_set()
    effective = verify_pen_readback(requested, readback(requested), .5)
    with pytest.raises(ValueError):
        decode_export(change(svg()), chord_error_m=5e-6, pen_width_m=.0005, pens=effective['pens'])
