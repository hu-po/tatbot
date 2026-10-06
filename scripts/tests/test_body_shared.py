"""Shared body boundary regressions; synthetic bytes, no model runtime required."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
import tatbot_cli  # noqa: E402, F401
from tatbot_contracts import body  # noqa: E402
from tatbot_contracts.canonical import (  # noqa: E402
    ContractError,
    canonical_bytes,
    canonical_digest,
    parse_json,
)

# Load the adapter itself so this offline suite needs no Torch/NumPy installation.
_loader = importlib.util.spec_from_file_location(
    "runtime_io", REPO / "python/tatbot_sim/src/tatbot_sim/body_models/io.py",
)
runtime = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(runtime)


@pytest.fixture
def cache_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.bin").write_bytes(b"synthetic model bytes")
    spec = json.loads((REPO / "config/body-models/mhr-soma-v1.json").read_bytes())
    spec["assets"] = [{
        "path": "data.bin", "size": 21,
        "sha256": hashlib.sha256(b"synthetic model bytes").hexdigest(),
        "license": "MIT", "reason": "test fixture",
    }]
    spec["content_sha256"] = canonical_digest(spec)
    path = tmp_path / "spec.json"
    path.write_bytes(canonical_bytes(spec))
    cache = tmp_path / "cache"
    body.bootstrap(path, source, cache)
    return spec, path, cache


@pytest.mark.parametrize("case,code", [
    ("missing", "body_asset_missing"),
    ("extra", "body_asset_path_denied"),
    ("symlink", "body_asset_path_denied"),
    ("ancestor_symlink", "body_asset_path_denied"),
    ("fifo", "body_asset_path_denied"),
    ("writable", "body_asset_path_denied"),
    ("writable_metadata", "body_asset_path_denied"),
    ("size", "body_asset_hash_mismatch"),
    ("digest", "body_asset_hash_mismatch"),
    ("metadata", "body_asset_hash_mismatch"),
    ("permissions", "body_asset_path_denied"),
])
def test_bootstrap_audit_and_runtime_refuse_same_cache_bytes(cache_fixture, tmp_path, monkeypatch, case, code):
    spec, path, cache = cache_fixture
    asset = cache / "data.bin"
    if case == "missing":
        asset.unlink()
    elif case == "extra":
        (cache / "extra").touch()
    elif case == "symlink":
        asset.unlink()
        asset.symlink_to(tmp_path / "source/data.bin")
    elif case == "ancestor_symlink":
        (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
        cache = tmp_path / "link/cache"
    elif case == "fifo":
        asset.unlink()
        os.mkfifo(asset)
    elif case.startswith("writable"):
        (cache / body.METADATA if case == "writable_metadata" else asset).chmod(0o644)
    elif case in {"size", "digest"}:
        asset.chmod(0o644)
        asset.write_bytes(b"X" if case == "size" else b"X" * 21)
        asset.chmod(0o444)
    elif case == "metadata":
        metadata = cache / body.METADATA
        metadata.chmod(0o644)
        metadata.write_bytes(b'{"schema":1,"schema":2}')
        metadata.chmod(0o444)
    else:
        original = Path.open
        def denied(file, *args, **kwargs):
            if file == asset:
                raise PermissionError("synthetic read refusal")
            return original(file, *args, **kwargs)
        monkeypatch.setattr(Path, "open", denied)
    with pytest.raises((body.BodyCacheError, OSError)):
        body.audit(path, cache)
    with pytest.raises(runtime.BodyModelError) as caught:
        runtime.verify_body_cache(spec, cache)
    assert caught.value.code == code


def test_both_boundaries_reject_cache_inside_repository(cache_fixture, tmp_path):
    spec, path, cache = cache_fixture
    repository = tmp_path / "repository"
    marker = repository / ".git"
    marker.mkdir(parents=True)
    (marker / "HEAD").write_text("ref: refs/heads/main\n")
    (marker / "objects").mkdir()
    inside = repository / "cache"
    cache.rename(inside)
    with pytest.raises(body.BodyCacheError, match="repository"):
        body.audit(path, inside)
    with pytest.raises(runtime.BodyModelError) as caught:
        runtime.verify_body_cache(spec, inside)
    assert caught.value.code == "body_asset_path_denied"


def test_runtime_rechecks_spec_before_reading_assets(cache_fixture, monkeypatch):
    spec, _, cache = cache_fixture
    spec["assets"][0]["path"] = "../outside"
    spec["content_sha256"] = canonical_digest(spec)
    def no_read(*args, **kwargs):
        pytest.fail("asset read before spec validation")
    monkeypatch.setattr(Path, "open", no_read)
    with pytest.raises(runtime.BodyModelError) as caught:
        runtime.verify_body_cache(spec, cache)
    assert caught.value.code == "body_asset_path_denied"


def test_strict_distribution_paths_and_unreviewed_specs(cache_fixture):
    spec, path, _ = cache_fixture
    with pytest.raises(runtime.BodyModelError) as caught:
        runtime.load_body_model_spec(path)
    assert caught.value.code == "body_model_unpinned"
    spec["python_package"]["wheel"]["filename"] = "../wheel"
    spec["content_sha256"] = canonical_digest(spec)
    path.write_bytes(canonical_bytes(spec))
    with pytest.raises(body.BodyCacheError) as caught:
        body.load_spec(path)
    assert caught.value.code == "path_escape"
    with pytest.raises(runtime.BodyModelError) as caught:
        runtime.load_body_model_spec(path)
    assert caught.value.code == "body_asset_path_denied"


@pytest.mark.parametrize("data,code", [
    (b'{"key":1,"key":2}', "duplicate_key"),
    (b'{"n":-0.0}', "negative_zero"),
    (b'{"n":9007199254740992}', "unsafe_integer"),
    (b'{"n":NaN}', "non_finite"),
    (b'{"s":"\\ud800"}', "invalid_json"),
])
def test_strict_json(data, code):
    with pytest.raises(ContractError) as caught:
        parse_json(data)
    assert caught.value.code == code


def test_canonical_bytes_remain_browser_compatible():
    value = {"\ue000": 1, "\U00010000": [1.0, 1e-7, 1e-6, 1e20, 1e21]}
    assert canonical_bytes(value) == '{"𐀀":[1,1e-7,0.000001,100000000000000000000,1e+21],"":1}'.encode()


def test_bare_clone_shared_imports_are_stdlib_and_read_only(tmp_path):
    program = '''
import sys
sys.path.insert(0, sys.argv[1])
import tatbot_cli
from tatbot_contracts.body import load_spec
from tatbot_contracts.canonical import canonical_digest
spec, digest = load_spec(__import__('pathlib').Path(sys.argv[2]))
assert canonical_digest(spec) == digest
assert not {'numpy', 'torch', 'soma', 'tatbot_sim'} & set(sys.modules)
'''
    result = subprocess.run([
        sys.executable, "-S", "-c", program, str(REPO / "scripts/lib"),
        str(REPO / "config/body-models/mhr-soma-v1.json"),
    ], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_spec_reader_refuses_utf16_and_special_file_without_blocking(tmp_path):
    spec = (REPO / "config/body-models/mhr-soma-v1.json").read_text()
    path = tmp_path / "spec.json"
    path.write_bytes(spec.encode("utf-16"))
    with pytest.raises(body.BodyCacheError) as caught:
        body.load_spec(path)
    assert caught.value.code == "invalid_spec"
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(body.BodyCacheError) as caught:
        body.load_spec(path)
    assert caught.value.code == "invalid_spec"
