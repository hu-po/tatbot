"""Frozen research inputs and durable state; no compiler selection or shell templates."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path

from tatbot_contracts.artwork import validate_source
from tatbot_contracts.canonical import canonical_digest, parse_json


class ResearchError(ValueError):
    """A trial cannot continue without changing a named input or resolving evidence."""


def read(path):
    return parse_json(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", value):
        raise ResearchError("identifier must contain 1..80 letters, digits, underscores or hyphens")
    return value


def frozen(path, schema):
    value = read(path)
    if value.get("schema") != schema or value.get("content_sha256") != canonical_digest(value):
        raise ResearchError(f"{path}: expected intact {schema}")
    return value


def seal(value):
    value["content_sha256"] = canonical_digest(value)
    return value


def write(path, value):
    """Atomic state publication, including directory fsync; callers hold the study lock."""
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def locked(root):
    with (Path(root) / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResearchError("another research operation owns this study") from error
        yield


def event(root, kind, **fields):
    from datetime import datetime, timezone

    row = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
    with (Path(root) / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def case_map(corpus):
    cases, families, sources = {}, {}, {}
    if corpus.get("schema") != "tatbot.dbv3-corpus/1" or not isinstance(corpus.get("cases"), list):
        raise ResearchError("expected a DBV3 source-image corpus")
    for case in corpus["cases"]:
        key, family, split = identifier(case["id"]), identifier(case["family"]), case["split"]
        if key in cases or split not in ("train", "validation", "test"):
            raise ResearchError("case IDs must be unique and splits explicit")
        if families.setdefault(family, split) != split:
            raise ResearchError("all variants of a source family must stay in one split")
        source = case["sha256"]
        if not isinstance(source, str) or not re.fullmatch('[0-9a-f]{64}', source):
            raise ResearchError("source requires SHA-256")
        if sources.setdefault(source, family) != family:
            raise ResearchError("identical source bytes cannot be relabelled as another family")
        size = case["size_mm"]
        if not isinstance(size, list) or len(size) != 2 or any(type(v) not in (int, float) or not 0 < v <= 2000 for v in size):
            raise ResearchError("case canvas dimensions must be positive millimetres within the artwork domain")
        validate_source(case['provenance'])
        cases[key] = case
    if not cases or {case['split'] for case in cases.values()} != {'train', 'validation', 'test'}:
        raise ResearchError("corpus requires separate train, validation and test source families")
    return cases


def load_study(root):
    study = frozen(Path(root) / 'study.json', 'tatbot.dbv3-study/1')
    validate_layout(study['layout'])
    case_map(study['corpus'])
    for case in study['corpus']['cases']:
        path = Path(root) / 'sources' / (case['id'] + '.png')
        if digest(path) != case['sha256']:
            raise ResearchError(f"source changed: {case['id']}")
    return study


def _layout_arrays(value, unit):
    keys = [f'{name}_{unit}' for name in ('slot', 'rows', 'columns')]
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ResearchError(f'layout requires only {", ".join(keys)}')
    arrays = [value[key] for key in keys]
    if any(not isinstance(items, list) or not items for items in arrays):
        raise ResearchError('layout dimensions and centres must be nonempty arrays')
    if len(arrays[0]) != 2 or len(arrays[2]) != 2 or len(arrays[1]) > 1000:
        raise ResearchError('layout requires [width, height], two columns and 1..1000 rows')
    try:
        valid = all(type(n) in (int, float) and math.isfinite(n) for items in arrays for n in items)
    except OverflowError:
        valid = False
    if not valid:
        raise ResearchError('layout values must be finite numbers')
    return arrays


def validate_layout(layout):
    """Validate the whole grid before any slot can be reserved, including sealed imported studies."""
    from tatbot_ink.place import STENCIL_CLEAR_M

    slot, rows, columns = _layout_arrays(layout, 'm')
    if any(size <= 0 for size in slot):
        raise ResearchError('layout slot dimensions must be positive')
    if columns[0] >= columns[1]:
        raise ResearchError('layout columns must be ordered left to right')
    for centres, size, clear in zip((columns, rows), slot, STENCIL_CLEAR_M, strict=True):
        if any(abs(centre) + size/2 > clear/2 + 1e-12 for centre in centres):
            raise ResearchError('research slot exceeds the page clear area')
        ordered = sorted(centres)
        if any(b-a <= size + 1e-12 for a, b in zip(ordered, ordered[1:], strict=False)):
            raise ResearchError('research slots need positive separation; touching or overlapping slots are refused')


def layout_from_mm(value):
    slot, rows, columns = _layout_arrays(value, 'mm')
    layout = {key: [n/1000 for n in numbers]
              for key, numbers in zip(('slot_m', 'rows_m', 'columns_m'), (slot, rows, columns), strict=True)}
    validate_layout(layout)
    return layout


def differences(a, b, prefix=''):
    """Changed leaves of a declared recipe/preparation policy, without acquisition noise."""
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result += differences(a.get(key), b.get(key), f'{prefix}.{key}' if prefix else key)
        return result
    return [] if a == b else [prefix]
