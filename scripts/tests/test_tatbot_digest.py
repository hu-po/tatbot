"""File and byte digests agree with hashlib, including across chunk boundaries."""

from __future__ import annotations

import hashlib

from tatbot_digest import sha256_bytes, sha256_file


def test_sha256_file_matches_the_one_liner_it_replaced(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(range(256)) * 5000)  # spans a chunk boundary
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(str(path)) == sha256_file(path)


def test_sha256_bytes_matches_hashlib():
    assert sha256_bytes(b"tatbot") == hashlib.sha256(b"tatbot").hexdigest()
