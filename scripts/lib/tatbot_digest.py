"""One sha256 for files and bytes, so a manifest's hashes mean one thing everywhere.

Twelve modules under scripts/ each carried an identical one-line file digest
and four more an identical bytes digest; which copy a module hashed with was an
accident of its import list. Stdlib only, Python >= 3.10, like the rest of
scripts/lib (`hashlib.file_digest` is 3.11). python/tatbot_sim went through the
same consolidation into tatbot_contracts.digest; scripts/ cannot import that
package from a launcher-run script, so this is its twin.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1024 * 1024


def sha256_file(path) -> str:
    """Hex sha256 of a file's bytes, read in chunks rather than all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of bytes already in memory."""
    return hashlib.sha256(data).hexdigest()
