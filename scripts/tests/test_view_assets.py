"""Retained artifacts are read bounded and bound to their digest."""
import hashlib

import pytest
import view_assets as assets


def test_bound_assets_refuse_tampering_and_escape(tmp_path):
    (tmp_path / 'asset').write_bytes(b'original')
    sha = hashlib.sha256(b'original').hexdigest()
    assert assets.bound_read(tmp_path, 'asset', sha) == b'original'
    (tmp_path / 'asset').write_bytes(b'changed')
    with pytest.raises(ValueError, match='digest mismatch'):
        assets.bound_read(tmp_path, 'asset', sha)
    outside = tmp_path.parent / 'outside-display-input'
    outside.write_bytes(b'original')
    with pytest.raises(ValueError, match='escapes'):
        assets.bound_read(tmp_path, '../outside-display-input', sha)
