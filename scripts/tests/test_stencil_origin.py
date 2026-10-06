"""The compiled scan seeds material coordinates before the first live scan."""

import copy
import hashlib
import json

import numpy as np
import pytest
from stencil_scan import ScanBinding
from test_stencil_state import accept, archive, camera, motion, retain, state


def compiled(run, original, *, tool=True):
    payload, sha = archive(original)
    index = 2 if tool else 1
    path = run/f'compiled/inputs/source-{index}'
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    program = {'schema': 'tatbot.session-program/2', 'sources': {
        'tool': {'path': 'tool.json', 'sha256': 'a'*64} if tool else None,
        'inputs': [{'path': 'surface.npz', 'sha256': sha}]}}
    program_path = run/'compiled/program.json'
    program_path.write_text(json.dumps(program))
    return {'schema': 'tatbot.session-surface/1', 'kind': 'stencil-origin',
            'program': {'path': 'compiled/program.json', 'sha256': hashlib.sha256(program_path.read_bytes()).hexdigest()},
            'surface': {'path': str(path.relative_to(run)), 'sha256': sha}}


@pytest.mark.parametrize('tool', [True, False])
def test_compiled_original_then_new_scan_preserves_first_material_frame(tmp_path, tool):
    original, _ = state()
    selection = compiled(tmp_path, original, tool=tool)
    reader = ScanBinding(tmp_path, from_session=True)
    reader.select(selection)
    assert reader.error is None
    first = reader.observe({'depth': camera(1_100_000_000)}, 1_200_000_000)
    assert first['scan_binding'] == 'session-compiled'
    assert first['surfaces'][0]['reference_sha256'] == original.reference_sha256
    moved = motion()
    incoming = copy.deepcopy(original)
    incoming.binding['geometry_capture_ns'] = 2_000_000_000
    incoming.binding['points_material_m'] = (original.xyz @ moved[:3, :3].T + moved[:3, 3]).tolist()
    incoming.points = original.points @ moved[:3, :3].T + moved[:3, 3]
    retain(tmp_path, 0, incoming)
    reader.select(accept(tmp_path, 0))
    assert reader.error is None
    current = reader.observe({'depth': camera(2_100_000_000, moved)}, 2_200_000_000)
    assert current['scan_binding'] == 'session-accepted'
    assert current['surfaces'][0]['reference_sha256'] == original.reference_sha256
    np.testing.assert_allclose(current['surfaces'][0]['root_from_material'], moved, atol=1e-7)
    reader.select(selection)
    assert 'cannot replace' in reader.error
    assert not reader.states


@pytest.mark.parametrize('fault', ['program_bytes', 'surface_bytes', 'surface_path', 'surface_digest', 'program_path'])
def test_compiled_origin_refuses_mutated_or_unbound_source(tmp_path, fault):
    original, _ = state()
    selection = compiled(tmp_path, original)
    if fault.endswith('_bytes'):
        key = fault.removesuffix('_bytes')
        (tmp_path/selection[key]['path']).write_bytes(b'changed')
    elif fault == 'surface_path':
        selection['surface']['path'] = 'compiled/inputs/source-1'
    elif fault == 'surface_digest':
        selection['surface']['sha256'] = 'f'*64
    else:
        selection['program']['path'] = 'other/program.json'
    reader = ScanBinding(tmp_path, from_session=True)
    reader.select(selection)
    assert reader.error
    assert not reader.states and not reader.originals
