"""Resolve the original surface from a verified retained program.

This is the geometry the artwork was compiled on, not a newly accepted scan.
Retained source indices follow the program's `source_artifacts` exactly.
"""

import json

import schemas
from stencil_state import load_scan
from view_assets import bound_read


def load_origin(run, selection):
    if (not schemas.is_schema(selection, schemas.SURFACE, 'stencil-origin')
            or selection['program']['path'] != 'compiled/program.json'):
        raise ValueError('invalid compiled stencil origin')
    program = json.loads(bound_read(run, selection['program']['path'], selection['program']['sha256']))
    if program.get('schema') not in ('tatbot.session-program/1', 'tatbot.session-program/2'):
        raise ValueError('compiled stencil origin needs a Session program')
    sources = program['sources']
    offset = 1 + int(sources.get('tool') is not None)
    surfaces = [(index + offset, value) for index, value in enumerate(sources['inputs'])
                if value['path'] == 'surface.npz']
    if len(surfaces) != 1:
        raise ValueError('compiled program needs exactly one original surface')
    index, source = surfaces[0]
    receipt = {'path': f'compiled/inputs/source-{index}', 'sha256': source['sha256']}
    if selection['surface'] != receipt:
        raise ValueError('stencil origin differs from the compiled surface source')
    payload = bound_read(run, receipt['path'], receipt['sha256'])
    return load_scan(payload, receipt['sha256']), receipt['sha256']
