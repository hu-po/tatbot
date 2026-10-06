"""Frozen path identity is shared with the browser; no source reinterpretation."""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import tatbot_cli  # noqa: F401 -- bare-clone contracts source root
from tatbot_contracts.canonical import canonical_digest
from tatbot_contracts.paths import freeze_program, render_paths, validate_path_program

ROOT = Path(__file__).resolve().parents[2]


def geometry():
    return {"canvas_m": {"width": .05, "height": .075}, "inks": [{"id": "black", "color_srgb": [0, 0, 0]}],
            "layers": [{"id": "first", "ink_id": "black", "elements": [{"id": "line", "kind": "path",
                "fill": False, "closed": False, "width_m": .0005, "deposition": 1,
                "points_m": [[.001, .001], [.02, .02], [.04, .01]]}]}], "negative_space_masks": []}


def frozen():
    return freeze_program(geometry(), source_sha256='a'*64, name='frozen', adapter='dbv3-batik-paths/1')


def test_frozen_paths_validate_without_regenerating_source():
    program, svg = frozen()
    assert program == validate_path_program(json.loads(json.dumps(program)))
    assert 'stroke-width="0.0005"' in svg
    assert '0.001,0.074' in svg
    assert program['content_sha256'] == canonical_digest(program)
    before = copy.deepcopy(program)
    validate_path_program(program)['layers'][0]['elements'].reverse()
    assert program == before


@pytest.mark.parametrize('mutate', [
    lambda g: g['layers'][0]['elements'][0].update(fill=True),
    lambda g: g['layers'][0]['elements'][0].update(kind='region'),
    lambda g: g.update(negative_space_masks=[{'id': 'mask', 'points_m': [[0, 0], [.01, .01], [.01, 0]]}]),
    lambda g: g['layers'][0]['elements'][0].update(points_m=[[0, 0], [0, 0]]),
    lambda g: g['layers'][0]['elements'][0].update(points_m=[[0, 0], [.1, .1]]),
    lambda g: g['layers'][0].update(ink_id='unknown'),
    lambda g: g['layers'][0]['elements'].append(copy.deepcopy(g['layers'][0]['elements'][0])),
    lambda g: g['layers'][0]['elements'][0].update(width_m=float('nan')),
])
def test_unresolved_paint_and_invalid_geometry_cannot_enter_preparation(mutate):
    value = geometry()
    mutate(value)
    with pytest.raises(ValueError):
        freeze_program(value, source_sha256='a'*64, name='invalid', adapter='test')


def test_program_tampering_changes_identity_even_if_preview_unchanged():
    program, _ = frozen()
    program['layers'][0]['elements'][0]['points_m'].reverse()
    with pytest.raises(ValueError, match='digest mismatch'):
        validate_path_program(program)
    program['content_sha256'] = canonical_digest(program)
    assert validate_path_program(program) == program  # a new acquisition, not another SVG conversion


def test_renderer_escapes_ids_as_data():
    value = geometry()
    value['layers'][0]['elements'][0]['id'] = '"/><script>bad</script>'
    rendered = render_paths(value)
    assert '<script>' not in rendered and '&lt;script&gt;' in rendered


def test_frozen_program_has_browser_canonical_identity_without_svg_materializer():
    node = shutil.which('node')
    if node is None:
        pytest.skip('browser parity requires Node 22+')
    version = subprocess.check_output([node, '--version'], text=True, timeout=5).strip()
    if int(version.lstrip('v').split('.')[0]) < 22:
        pytest.skip('browser parity requires Node 22+')
    program, _ = frozen()
    code = '''import { validateTattooProgram } from './src/core/human-representation/tattoo-program.ts';
import { readFileSync } from 'node:fs';
const program = await validateTattooProgram(JSON.parse(readFileSync(0, 'utf8')));
process.stdout.write(program.content_sha256);'''
    process = subprocess.run([node, '--experimental-strip-types', '--input-type=module', '-e', code],
                             cwd=ROOT/'web/inkmap', input=json.dumps(program), text=True, capture_output=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert process.stdout == program['content_sha256']


def test_frozen_artwork_binds_geometry_source_and_acquisition_across_languages():
    from tatbot_contracts.artwork import freeze_artwork, validate_path_artwork
    program, _ = frozen()
    record = freeze_artwork(program, name='frozen', source={
        'kind': 'fixture', 'identifier': 'unit-test', 'license': None, 'attribution': None, 'generation': None},
        conversion={'adapter': 'dbv3-batik-paths/1', 'recipe_sha256': 'b'*64, 'chord_error_m': .000005})
    assert validate_path_artwork(record) == record
    assert 'original_svg' not in record
    node = shutil.which('node')
    if node is None:
        pytest.skip('browser parity requires Node 22+')
    code = '''import { validateArtworkRecord } from './src/core/artwork-record.ts';
import { readFileSync } from 'node:fs';
const record = await validateArtworkRecord(JSON.parse(readFileSync(0, 'utf8')));
process.stdout.write(record.content_sha256);'''
    process = subprocess.run([node, '--experimental-strip-types', '--input-type=module', '-e', code],
        cwd=ROOT/'web/inkmap', input=json.dumps(record), text=True, capture_output=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert process.stdout == record['content_sha256']
    stale = copy.deepcopy(record)
    stale['program']['layers'][0]['elements'][0]['points_m'].reverse()
    stale['content_sha256'] = canonical_digest(stale)
    with pytest.raises(ValueError, match='path digest mismatch'):
        validate_path_artwork(stale)
    stale = copy.deepcopy(record)
    stale['source_sha256'] = 'c'*64
    stale['content_sha256'] = canonical_digest(stale)
    with pytest.raises(ValueError, match='different source bytes'):
        validate_path_artwork(stale)
