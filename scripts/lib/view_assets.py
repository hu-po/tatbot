"""Bounded, hash-bound reads of retained run artifacts for the stencil observer
and live inputs; no hardware or viewer owner."""
from __future__ import annotations

import hashlib
from pathlib import Path


def read(path, limit=64 * 1024 * 1024):
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'{Path(path).name} exceeds display size limit')
    return data


def bound_read(root, relative, digest):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError('display artifact escapes retained run')
    data = read(path)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'{relative} digest mismatch')
    return data
