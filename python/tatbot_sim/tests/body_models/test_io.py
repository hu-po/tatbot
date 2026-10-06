from __future__ import annotations

from pathlib import Path

import pytest
from tatbot_sim.body_models.io import BodyModelError, load_body_model_spec, verify_body_cache

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "config/body-models/mhr-soma-v1.json"


def test_reviewed_spec_loads() -> None:
    spec = load_body_model_spec(SPEC)
    assert spec["name"] == "mhr-soma-v1"
    assert spec["model"] == {
        "identity_model_type": "mhr",
        "lod": "mid",
        "mode": "dense",
        "output_unit": "m",
        "enable_procedural_transforms": True,
        "apply_correctives": True,
        "shape_components": 45,
        "scale_parameters": 68,
        "public_joints": 78,
        "control_joints": 77,
    }


def test_missing_cache_refuses_before_model_import(tmp_path: Path) -> None:
    spec = load_body_model_spec(SPEC)
    with pytest.raises(BodyModelError, match="body_asset_missing") as caught:
        verify_body_cache(spec, tmp_path / "absent")
    assert caught.value.code == "body_asset_missing"


def test_symlink_cache_refuses(tmp_path: Path) -> None:
    spec = load_body_model_spec(SPEC)
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(BodyModelError, match="body_asset_path_denied") as caught:
        verify_body_cache(spec, linked)
    assert caught.value.code == "body_asset_path_denied"


@pytest.mark.parametrize("cache_case", ["missing", "corrupt"])
def test_runtime_constructor_refuses_before_soma_import(tmp_path, monkeypatch, cache_case):
    import builtins
    import hashlib
    import json

    from tatbot_contracts.body import bootstrap
    from tatbot_contracts.canonical import canonical_bytes, canonical_digest
    from tatbot_sim.body_models import mhr_soma

    spec = json.loads(SPEC.read_bytes())
    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.bin").write_bytes(b"synthetic")
    spec["assets"] = [{
        "path": "fixture.bin", "size": 9,
        "sha256": hashlib.sha256(b"synthetic").hexdigest(),
        "license": "MIT", "reason": "constructor refusal fixture",
    }]
    spec["content_sha256"] = canonical_digest(spec)
    path = tmp_path / "spec.json"
    path.write_bytes(canonical_bytes(spec))
    cache = tmp_path / "cache"
    bootstrap(path, source, cache)
    asset = cache / "fixture.bin"
    if cache_case == "missing":
        asset.unlink()
    else:
        asset.chmod(0o644)
        asset.write_bytes(b"corrupted")
        asset.chmod(0o444)
    # Bypass only the reviewed model identity for this synthetic spec, keeping
    # the actual constructor and immediate cache verification on the runtime path.
    monkeypatch.setattr(mhr_soma, "load_body_model_spec", lambda _: spec)
    original = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "soma" or name.startswith("soma."):
            pytest.fail("SOMA imported before malformed cache refusal")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(BodyModelError) as caught:
        mhr_soma.SOMAPosedBody(spec_path=path, cache_dir=cache, device="cpu")
    assert caught.value.code == (
        "body_asset_missing" if cache_case == "missing" else "body_asset_hash_mismatch"
    )
