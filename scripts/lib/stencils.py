"""Portable, hash-checked artwork retained by a named session's scan inputs."""

import base64
import json
from pathlib import Path

import schemas
import stencil_reference
import stencil_work_index

MAX_BYTES = 4 * 1024 * 1024


def bundle(paths):
    if not 1 <= len(paths) <= 8:
        raise ValueError('retain one to eight distinct stencil references')
    references = []
    for path in paths:
        manifest, image = stencil_reference.load(path)
        row = {'reference': manifest, 'png_base64': base64.b64encode(image.read_bytes()).decode('ascii')}
        if 'coded' in manifest:
            code = stencil_reference.coded_path(path).read_bytes()
            row['coded_base64'] = base64.b64encode(code).decode('ascii')
        issue_path = Path(path).with_name('issue-receipt.json')
        if issue_path.is_file():
            reference_bytes = Path(path).read_bytes()
            issue_bytes = issue_path.read_bytes()
            stencil_work_index.validate_issue_bytes(manifest, reference_bytes, issue_bytes)
            row['reference_bytes_base64'] = base64.b64encode(reference_bytes).decode('ascii')
            row['issue_receipt_base64'] = base64.b64encode(issue_bytes).decode('ascii')
        references.append(row)
    result = {**schemas.stamp(schemas.SURFACE, 'stencils'), 'references': references, 'motion_authority': False}
    if len(json.dumps(result)) > MAX_BYTES:
        raise ValueError('session stencil artwork exceeds 4 MiB')
    ids = [row['reference']['pattern_id'] for row in references]
    if len(ids) != len(set(ids)):
        raise ValueError('session stencil patterns must be distinct; identical physical copies are ambiguous')
    return result


def materialize(value, directory):
    """Revalidate embedded bytes with the ordinary reference loader."""
    if not schemas.is_schema(value, schemas.SURFACE, 'stencils') or len(json.dumps(value)) > MAX_BYTES:
        raise ValueError('invalid session stencil bundle')
    references = value['references']
    if not 1 <= len(references) <= 8:
        raise ValueError('invalid session stencil reference count')
    paths, identities = [], set()
    for index, row in enumerate(references):
        root = Path(directory) / str(index)
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'tracking.json'
        issued = 'issue_receipt_base64' in row or 'reference_bytes_base64' in row
        if issued:
            if not {'issue_receipt_base64', 'reference_bytes_base64'} <= set(row):
                raise ValueError('partial session print issue receipt')
            reference_bytes = base64.b64decode(row['reference_bytes_base64'], validate=True)
            issue_bytes = base64.b64decode(row['issue_receipt_base64'], validate=True)
            if json.loads(reference_bytes) != row['reference']:
                raise ValueError('session print issue reference differs from its view')
            stencil_work_index.validate_issue_bytes(row['reference'], reference_bytes, issue_bytes)
            path.write_bytes(reference_bytes)
            (root / 'issue-receipt.json').write_bytes(issue_bytes)
        else:
            path.write_text(json.dumps(row['reference'], allow_nan=False))
        (root / 'stencil.png').write_bytes(base64.b64decode(row['png_base64'], validate=True))
        if 'coded_base64' in row:
            (root / stencil_reference.CODED_FILE).write_bytes(base64.b64decode(row['coded_base64'], validate=True))
        reference, _ = stencil_reference.load(path)
        if reference['pattern_id'] in identities:
            raise ValueError('duplicate session stencil pattern')
        identities.add(reference['pattern_id'])
        paths.append(path)
    return paths


def prepare(root, inputs):
    path = root / 'stencil-view.json'
    if path.is_file():
        config_path = inputs / 'draw.json'
        config = json.loads(config_path.read_bytes())
        config['stencil_view'] = json.loads(path.read_bytes())
        config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')


def retain(root, job, paths):
    if paths:
        data = bundle(paths)
        path = root / 'stencil-view.json'
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        job['files']['stencil-view.json'] = stencil_reference.digest(path)
