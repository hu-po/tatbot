"""File digests, shared so a manifest's hashes mean one thing everywhere.

Ten evidence, audit and release generators each carried an identical private
`_sha256`. They agreed, which is the only reason nothing broke; nothing kept
them agreeing. `hashlib.file_digest` would do this in one line, but it landed in
3.11 and tatbot_contracts declares >=3.10 so the stdlib-only CLI can import it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file's bytes, read in chunks rather than all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()
