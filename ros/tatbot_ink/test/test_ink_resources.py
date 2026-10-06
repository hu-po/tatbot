"""Exact bindings prevent color/width inference and contradictory material schedules."""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import yaml
from ink_designs import element
from resource_fixtures import SPEED, drawing, fixture_repo, manifest, resource, save
from tatbot_ink.errors import CompileError
from tatbot_ink.place import placed_strokes
from tatbot_ink.resources import bind_resources


@pytest.fixture
def setup(tmp_path):
    repo = fixture_repo(tmp_path / 'repo')
    design = drawing([('black', [element([[.002, .01], [.018, .01]])]),
                      ('same-color', [element([[.018, .005], [.002, .005]])])])
    path = manifest(tmp_path / 'inks.yaml', design, [resource('B'), resource('R', ink='red')],
                    [('black', 'B'), ('same-color', 'R')])
    return repo, design, path


def bind(setup):
    repo, design, path = setup
    return bind_resources(path, design, repo=repo, arm='right', speed_m_s=SPEED)


def test_binding_freezes_exact_pen_ink_tool_and_input_evidence(setup):
    resources, bindings, evidence = bind(setup)
    assert list(bindings.values()) == ['B', 'R']
    assert [r['tool']['id'] for r in resources] == ['ball', 'ball']
    assert [r['ink_id'] for r in resources] == ['black', 'red']
    assert all(r['slot'] is r['dip'] is None for r in resources)
    assert evidence == {'mode': 'explicit', 'file_sha256': hashlib.sha256(setup[2].read_bytes()).hexdigest(),
                        'substrate': 'paper'}
    assert bind(setup) == (resources, bindings, evidence)
    original = copy.deepcopy(resources)
    sheet_path = setup[0] / 'config/tools/ball.yaml'
    sheet = yaml.safe_load(sheet_path.read_text())
    sheet['tip_out_at_top_mm'] += .1
    save(sheet_path, sheet)
    changed, _, _ = bind(setup)
    assert changed[0]['tool'] != original[0]['tool']
    assert resources == original


@pytest.mark.parametrize(('edit', 'message'), [
    (lambda doc: doc['bindings'].pop(), 'every acquired pen'),
    (lambda doc: doc['bindings'].append(doc['bindings'][0]), 'duplicate or unresolved'),
    (lambda doc: doc['bindings'][0].update(artwork_sha256='f'*64), 'duplicate or unresolved'),
    (lambda doc: doc['bindings'][0].update(resource_id='missing'), 'duplicate or unresolved'),
    (lambda doc: doc['bindings'][0].update(artwork_sha256=[]), 'exact artwork digest'),
    (lambda doc: doc['bindings'][0].update(pen_id={}), 'identifier'),
    (lambda doc: doc['resources'].append(doc['resources'][0]), 'IDs must be unique'),
    (lambda doc: doc['resources'][0].update(tool_id=None), 'identifier'),
    (lambda doc: doc['resources'][0].update(tool_id='../ball'), 'identifier'),
    (lambda doc: doc['resources'][0].update(ink_id='absent'), 'physical ink'),
    (lambda doc: doc['resources'][0].update(slot='inkcap_medium_1'), 'cannot declare dips'),
    (lambda doc: doc['resources'][0].update(pen_mode='automatic'), 'press, ride or hover'),
    (lambda doc: doc.update(substrate='cylinder'), 'flat substrate'),
    (lambda doc: doc.update(substrate='practice'), 'does not support substrate'),
    (lambda doc: doc.update(version=2), 'tatbot-inks/3'),
])
def test_ambiguous_or_contradictory_bindings_refused(setup, edit, message):
    doc = yaml.safe_load(setup[2].read_text())
    edit(doc)
    save(setup[2], doc)
    with pytest.raises(CompileError, match=message):
        bind(setup)


def test_selection_is_a_concrete_channel_configuration(setup):
    doc = yaml.safe_load(setup[2].read_text())
    selected = {'method': 'manual', 'action': 'select', 'channel': 'red'}
    doc['resources'][1]['activation'] = selected
    save(setup[2], doc)
    with pytest.raises(CompileError, match='own datasheet'):
        bind(setup)
    sheet = yaml.safe_load((setup[0] / 'config/tools/ball.yaml').read_text())
    sheet['cartridge_activation'] = {'action': 'select', 'channel': 'red'}
    save(setup[0] / 'config/tools/ball-red.yaml', sheet)
    doc['resources'][1]['tool_id'] = 'ball-red'
    save(setup[2], doc)
    resources, _, _ = bind(setup)
    assert resources[1]['activation'] == selected
    assert resources[0]['tool']['id'] != resources[1]['tool']['id']


def test_duplicate_yaml_identity_is_not_silently_overwritten(setup):
    setup[2].write_text(setup[2].read_text() + '\nsubstrate: practice\n')
    with pytest.raises(CompileError, match='duplicate mapping key'):
        bind(setup)


def test_two_needle_groupings_use_one_ink_without_merging_tools(setup):
    repo, design, path = setup
    manifest(path, design, [resource('three', tool='round-3', mode='ride'),
                            resource('five', tool='round-5', mode='ride')],
             [('black', 'three'), ('same-color', 'five')], substrate='practice')
    resources, bindings, _ = bind(setup)
    assert list(bindings.values()) == ['three', 'five']
    assert [r['tool']['id'] for r in resources] == ['round-3', 'round-5']
    assert all(r['tool']['ink'] == 'dip' and r['ink_id'] == 'black' for r in resources)
    assert all(r['dip']['mm_per_dip'] == 10 for r in resources)  # the sim's volume model is deliberately different
    assert resources[0]['dip'] == resources[1]['dip'] and resources[0] != resources[1]


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('mm_per_dip', None, 'finite'), ('mm_per_dip', 0, 'positive'),
    ('mm_per_dip', float('inf'), 'finite'), ('mm_per_dip', True, 'finite'),
    ('above_ink_m', -1, 'nonnegative'), ('dwell_s', -1, 'nonnegative'),
    ('wall_margin_m', -1, 'nonnegative'), ('hover_m', False, 'finite'),
    ('speed_m_s', float('inf'), 'finite'), ('speed_m_s', 0, 'positive'),
])
def test_dips_require_complete_settings(setup, field, value, message):
    repo, design, path = setup
    manifest(path, design, [resource('N', tool='round-3')], [('black', 'N'), ('same-color', 'N')], substrate='practice')
    sheet_path = repo / 'config/tools/round-3.yaml'
    sheet = yaml.safe_load(sheet_path.read_text())
    sheet['dip'][field] = value
    save(sheet_path, sheet)
    with pytest.raises(CompileError, match=message):
        bind(setup)


@pytest.mark.parametrize(('change', 'message'), [
    ({'slot': 'M1'}, 'canonical palette slot'),
    ({'slot': 'inkcap_medium_2'}, 'drawing arm'),
    ({'slot': 7}, 'identifier'),
])
def test_cap_aliases_are_refused(setup, change, message):
    _, design, path = setup
    manifest(path, design, [resource('N', tool='round-3', **change)],
             [('black', 'N'), ('same-color', 'N')], substrate='practice')
    with pytest.raises(CompileError, match=message):
        bind(setup)


def test_unknown_protrusion_prevents_ride_and_substrates_cannot_mix(setup):
    repo, design, path = setup
    manifest(path, design, [resource('N', tool='round-3', mode='ride')],
             [('black', 'N'), ('same-color', 'N')], substrate='practice')
    sheet_path = repo / 'config/tools/round-3.yaml'
    sheet = yaml.safe_load(sheet_path.read_text())
    sheet['tip_out_at_top_mm'] = None
    save(sheet_path, sheet)
    with pytest.raises(CompileError, match='tip_out_at_top_mm'):
        bind(setup)
    manifest(path, design, [resource('N', tool='round-3'), resource('B')],
             [('black', 'N'), ('same-color', 'B')], substrate='practice')
    with pytest.raises(CompileError, match='does not support substrate'):
        bind(setup)


def test_wider_replacement_uses_its_own_bounds_without_moving_source_paths():
    design = drawing([('narrow', [element([[.002, .01], [.018, .01]])]),
                      ('wide', [element([[.002, .015], [.02, .015]])])])
    artwork = next(iter(design['artworks'].values()))
    design['placements'][0]['placement']['target']['anchor_uv_m'] = [.02, 0]
    digest = artwork['content_sha256']
    widths = {(digest, 'narrow'): .0005, (digest, 'wide'): .0005}
    strokes, _ = placed_strokes(design, tool_widths=widths)
    before = strokes[1].points_m.copy()
    widths[digest, 'wide'] = .003
    with pytest.raises(CompileError, match='footprint leaves'):
        placed_strokes(design, tool_widths=widths)
    design['placements'][0]['placement']['target']['anchor_uv_m'] = [.019, 0]
    strokes, _ = placed_strokes(design, tool_widths=widths)
    np.testing.assert_allclose(strokes[1].points_m, before - [.001, 0])
    assert strokes[1].generation_width_m == .0005


RINSE = {'method': 'rinse', 'slot': 'inkcap_large_1', 'ink_id': 'water'}


def test_a_rinse_takes_the_cartridge_to_its_ink_in_a_cap_of_water(setup):
    """A resource activated by a rinse names the cap and its liquid; the datasheet gives the rinse's dwell."""
    _, design, path = setup
    manifest(path, design, [resource('N', tool='round-3', mode='ride'),
                            resource('R', tool='round-3', mode='ride', ink='red', activation=RINSE)],
             [('black', 'N'), ('same-color', 'R')], substrate='practice')
    resources, _, _ = bind(setup)
    assert resources[1]['activation'] == {**RINSE, 'dwell_s': 10.0, 'above_ink_m': 0.0}
    assert resources[0]['activation'] == {'method': 'manual', 'action': 'exchange'}


@pytest.mark.parametrize(('change', 'message'), [
    ({'slot': 'inkcap_medium_1'}, 'other than its ink'),
    ({'slot': 'inkcap_medium_2'}, 'other than its ink'),
    ({'ink_id': 'tea'}, 'absent from config/inks.yaml'),
    ({'dwell_s': 3.0}, 'slot, ink_id'),
    ({'sheet': None}, 'rinse_dwell_s'),
    ({'dry': True}, 'a resource that dips'),
])
def test_a_rinse_the_palette_or_the_datasheet_cannot_give_is_refused(setup, change, message):
    repo, design, path = setup
    change = dict(change)
    sheet, dry = change.pop('sheet', 1), change.pop('dry', False)
    extra = {'slot': None, 'mode': 'hover'} if dry else {'mode': 'ride'}
    manifest(path, design, [resource('N', tool='round-3', mode='ride'),
                            resource('R', tool='round-3', ink='red', activation={**RINSE, **change}, **extra)],
             [('black', 'N'), ('same-color', 'R')], substrate='practice')
    if sheet is None:
        sheet_path = repo / 'config/tools/round-3.yaml'
        data = yaml.safe_load(sheet_path.read_text())
        del data['dip']['rinse_dwell_s']
        save(sheet_path, data)
    with pytest.raises(CompileError, match=message):
        bind(setup)


def test_a_dipping_resource_without_a_cap_draws_dry(setup):
    _, design, path = setup
    manifest(path, design, [resource('N', tool='round-3', mode='hover', slot=None)], [('black', 'N'), ('same-color', 'N')],
             substrate='practice')
    resources, _, _ = bind(setup)
    assert resources[0]['tool']['ink'] == 'dip' and resources[0]['slot'] is resources[0]['dip'] is None
