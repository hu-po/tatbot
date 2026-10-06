"""Synthetic material records for software checks, never physical qualification."""
from __future__ import annotations

import copy

import yaml
from ink_designs import artwork, program
from tatbot_ink.input import read_input

SPEED = .0035


def dip_block():
    return {'above_ink_m': .0005, 'dwell_s': .4, 'hover_m': .003, 'speed_m_s': .002, 'wall_margin_m': .001,
            'mm_per_dip': 10, 'mm_per_dip_by_ink': {'red': 7}, 'rinse_dwell_s': 10.0, 'rinse_above_ink_m': 0.0}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value))
    return path


def fixture_repo(path):
    config = path / 'config'
    save(config / 'substrates.yaml', {'paper': {}, 'practice': {}, 'cylinder': {'shape': 'cylinder'}})
    # Equal display colors deliberately do not imply equal physical inks.
    save(config / 'inks.yaml', {key: {'rgb': rgb} for key, rgb in
                              [('black', [0, 0, 0]), ('other-black', [0, 0, 0]), ('red', [255, 0, 0]),
                               ('water', [170, 210, 235])]})
    save(config / 'palette.yaml', {'slots': {'inkcap_medium_1': {'arm': 'right'},
                                           'inkcap_medium_2': {'arm': 'left'}, 'inkcap_large_1': {'arm': 'right'}}})
    ball = {'substrate': 'paper', 'substrates': ['paper'], 'line': {'width_mm': .5, 'status': 'measured'},
            'ink': {'mode': 'cartridge'}, 'stroke_mm': 3.5, 'tip_out_at_top_mm': 2}
    needle = {'substrate': 'practice', 'line': {'width_mm': .3, 'status': 'assumed'},
              'ink': {'mode': 'real', 'mm_per_dip': 999999}, 'dip': dip_block(),
              'stroke_mm': 3.5, 'tip_out_at_top_mm': 1.5}
    for identity, sheet in [('ball', ball), ('round-3', needle), ('round-5', copy.deepcopy(needle))]:
        save(config / 'tools' / f'{identity}.yaml', sheet)
    return path


def drawing(layers):
    return read_input(artwork(program(layers)))


def resource(identity, *, tool='ball', ink='black', mode='press', **extra):
    result = {'id': identity, 'tool_id': tool, 'ink_id': ink, 'pen_mode': mode,
              'activation': {'method': 'manual', 'action': 'exchange'}, **extra}
    if tool.startswith('round-'):
        result.setdefault('slot', 'inkcap_medium_1')
    return result


def manifest(path, design, resources, assignments, *, substrate='paper'):
    digest = next(iter(design['artworks'].values()))['content_sha256']
    document = {'format': 'tatbot-inks', 'version': 3, 'substrate': substrate, 'resources': resources,
                'bindings': [{'artwork_sha256': digest, 'pen_id': pen, 'resource_id': identity}
                             for pen, identity in assignments]}
    return save(path, document)
