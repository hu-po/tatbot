from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]

import tatbot_cli  # noqa: E402, F401 -- initializes the bare-clone source path
from tatbot_cli.verbs.body import body_audit, body_bootstrap  # noqa: E402
from tatbot_contracts.body import (  # noqa: E402
    REVIEWED_LOW_VERTEX_MAP_SHA256,
    REVIEWED_MIRROR_MAP_SHA256,
    REVIEWED_REST_SURFACE_SHA256,
    REVIEWED_SPEC_SHA256,
    REVIEWED_TOPOLOGY_SHA256,
    REVIEWED_UV_SHA256,
    BodyCacheError,
)
from tatbot_contracts.body import (  # noqa: E402
    audit as _audit,
)
from tatbot_contracts.body import (  # noqa: E402
    bootstrap as _bootstrap,
)
from tatbot_contracts.body import (  # noqa: E402
    load_spec as _load_spec,
)
from tatbot_contracts.body import (  # noqa: E402
    public_refusal_code as _public_refusal_code,
)
from tatbot_contracts.body import (  # noqa: E402
    write_audit_report as _write_audit_report,
)
from tatbot_contracts.canonical import canonical_bytes as _canonical  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_spec(path: Path, asset: Path, *, revision: str = "1" * 40) -> None:
    spec = {
        "schema": "tatbot.body-model-spec/1",
        "content_sha256": "0" * 64,
        "name": "test",
        "sources": {
            "soma_x": {
                "repository": "https://example.invalid/soma",
                "tag": "v0.3.0",
                "commit": "3" * 40,
                "archive_size": 1,
                "archive_sha256": "4" * 64,
            },
            "mhr": {
                "repository": "https://example.invalid/mhr",
                "tag": "v1.0.1",
                "commit": "5" * 40,
                "archive_size": 1,
                "archive_sha256": "6" * 64,
            },
        },
        "python_package": {
            "name": "py-soma-x",
            "version": "0.3.0",
            "python": ">=3.11,<3.13",
            "wheel": {"filename": "soma.whl", "size": 1, "sha256": "7" * 64},
            "sdist": {"filename": "soma.tar.gz", "size": 1, "sha256": "8" * 64},
        },
        "asset_source": {
            "repository": "https://example.invalid/assets",
            "revision": revision,
            "manifest": {
                "path": "manifest.json",
                "size": 1,
                "sha256": "2" * 64,
            },
        },
        "assets": [
            {
                "path": "model/data.bin",
                "size": asset.stat().st_size,
                "sha256": _digest(asset),
                "license": "Apache-2.0",
                "reason": "test byte",
            }
        ],
        "denied_prefixes": ["SMPL/", "SMPLX/", "MANO/", "Anny/", "GarmentMeasurements/"],
        "model": {
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
        },
        "geometry": {
            "vertices": 18056,
            "triangles": 36108,
            "topology_sha256": REVIEWED_TOPOLOGY_SHA256,
            "low_vertex_map_sha256": REVIEWED_LOW_VERTEX_MAP_SHA256,
            "mirror_map_sha256": REVIEWED_MIRROR_MAP_SHA256,
            "uv_sets": {
                name: {
                    "interpolation": "faceVarying",
                    "values_sha256": REVIEWED_UV_SHA256[name][0],
                    "indices_sha256": REVIEWED_UV_SHA256[name][1],
                }
                for name in ("st", "st1", "st2")
            },
        },
        "coordinates": {
            "upstream": {"unit": "cm", "handedness": "right", "up": "+Y", "forward": "+Z"},
            "output": {"unit": "m", "handedness": "right", "up": "+Z", "front": "-Y"},
            "tatbot_from_soma": [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            "quantization_m": 0.00001,
            "reference_rest_surface_sha256": REVIEWED_REST_SURFACE_SHA256,
        },
        "software_lock": {
            "path": "config/body-models/test.txt",
            "sha256": "f" * 64,
            "environment": "cp312-x86_64-manylinux_2_28",
            "torch": "2.13.0",
        },
        "security": {
            "torchscript": "review-required",
            "npz_allow_pickle": False,
            "usd_parse_stage": "post-audit",
            "automatic_download": False,
            "native_binary_audit": "test inventory",
        },
    }
    material = {key: value for key, value in spec.items() if key != "content_sha256"}
    spec["content_sha256"] = hashlib.sha256(_canonical(material)).hexdigest()
    path.write_text(json.dumps(spec), encoding="utf-8")


def _source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    asset = source / "model" / "data.bin"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"reviewed model bytes")
    return source, asset


def _mark_git_repository(path: Path) -> None:
    marker = path / ".git"
    marker.mkdir(parents=True)
    (marker / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (marker / "objects").mkdir()


def _mutate_spec(path: Path, mutate) -> None:
    spec = json.loads(path.read_text(encoding="utf-8"))
    mutate(spec)
    material = {key: value for key, value in spec.items() if key != "content_sha256"}
    spec["content_sha256"] = hashlib.sha256(_canonical(material)).hexdigest()
    path.write_text(json.dumps(spec), encoding="utf-8")


def test_bootstrap_and_airgapped_hash_only_second_audit(tmp_path):
    source, asset = _source(tmp_path)
    spec = tmp_path / "spec.json"
    cache = tmp_path / "cache"
    _write_spec(spec, asset)

    bootstrapped = _bootstrap(spec, source, cache)
    before = {
        path.relative_to(cache): (path.stat().st_mode, path.stat().st_mtime_ns, _digest(path))
        for path in cache.rglob("*")
        if path.is_file()
    }
    audited = _audit(spec, cache)
    after = {
        path.relative_to(cache): (path.stat().st_mode, path.stat().st_mtime_ns, _digest(path))
        for path in cache.rglob("*")
        if path.is_file()
    }

    assert bootstrapped == audited
    assert audited["network_access"] is False
    assert audited["deserialization"] is False
    assert (cache / "model" / "data.bin").read_bytes() == b"reviewed model bytes"
    assert before == after


def test_tracked_model_spec_has_a_matching_cross_language_digest():
    spec, digest = _load_spec(REPO / "config" / "body-models" / "mhr-soma-v1.json")
    assert spec["content_sha256"] == digest
    assert digest == REVIEWED_SPEC_SHA256


def test_tracked_model_spec_binds_the_software_lock_bytes():
    spec, _ = _load_spec(REPO / "config" / "body-models" / "mhr-soma-v1.json")
    lock = REPO / spec["software_lock"]["path"]
    assert _digest(lock) == spec["software_lock"]["sha256"]


def test_bootstrap_refuses_unlisted_source_file(tmp_path):
    source, asset = _source(tmp_path)
    (source / "surprise").write_bytes(b"no")
    spec = tmp_path / "spec.json"
    _write_spec(spec, asset)
    with pytest.raises(BodyCacheError, match="surprise") as caught:
        _bootstrap(spec, source, tmp_path / "cache")
    assert caught.value.code == "unlisted_file"


def test_bootstrap_refuses_symlink_and_mutable_revision(tmp_path):
    source, asset = _source(tmp_path)
    real = asset
    real.rename(asset.with_suffix(".real"))
    asset.symlink_to(asset.with_suffix(".real").name)
    spec = tmp_path / "spec.json"
    _write_spec(spec, asset.with_suffix(".real"))
    with pytest.raises(BodyCacheError) as caught:
        _bootstrap(spec, source, tmp_path / "cache")
    assert caught.value.code == "symlink"

    source2, asset2 = _source(tmp_path / "second")
    mutable = tmp_path / "mutable.json"
    _write_spec(mutable, asset2, revision="main")
    with pytest.raises(BodyCacheError) as caught:
        _bootstrap(mutable, source2, tmp_path / "cache2")
    assert caught.value.code == "mutable_revision"


def test_bootstrap_refuses_path_escape_and_symlinked_cache_ancestor(tmp_path):
    source, asset = _source(tmp_path)
    spec = tmp_path / "spec.json"
    _write_spec(spec, asset)
    _mutate_spec(spec, lambda value: value["assets"][0].update(path="../escape"))
    with pytest.raises(BodyCacheError) as caught:
        _bootstrap(spec, source, tmp_path / "cache")
    assert caught.value.code == "path_escape"

    _write_spec(spec, asset)
    real_parent = tmp_path / "real-cache-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-cache-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(BodyCacheError) as caught:
        _bootstrap(spec, source, linked_parent / "nested" / "cache")
    assert caught.value.code == "symlink"
    assert not (real_parent / "nested").exists()


def test_bootstrap_refuses_cache_inside_repository_before_mutation(tmp_path):
    source, asset = _source(tmp_path)
    spec = tmp_path / "spec.json"
    _write_spec(spec, asset)
    repository = tmp_path / "repository"
    _mark_git_repository(repository)
    cache = repository / "body-cache"

    with pytest.raises(BodyCacheError) as caught:
        _bootstrap(spec, source, cache)
    assert caught.value.code == "cache_inside_repository"
    assert not cache.exists()


def test_missing_cache_refuses_without_downloading(tmp_path):
    source, asset = _source(tmp_path)
    spec = tmp_path / "spec.json"
    _write_spec(spec, asset)
    with pytest.raises(BodyCacheError) as caught:
        _audit(spec, tmp_path / "absent")
    assert caught.value.code == "missing_cache"


def test_audit_refuses_digest_mismatch_and_extra_file(tmp_path):
    source, asset = _source(tmp_path)
    spec = tmp_path / "spec.json"
    cache = tmp_path / "cache"
    _write_spec(spec, asset)
    _bootstrap(spec, source, cache)
    cached = cache / "model" / "data.bin"
    cached.chmod(0o644)
    cached.write_bytes(b"tampered model bytes")
    with pytest.raises(BodyCacheError) as caught:
        _audit(spec, cache)
    assert caught.value.code in {"size_mismatch", "digest_mismatch"}

    cached.write_bytes(b"reviewed model bytes")
    (cache / "extra").write_bytes(b"no")
    with pytest.raises(BodyCacheError) as caught:
        _audit(spec, cache)
    assert caught.value.code == "unlisted_file"


def test_audit_refuses_noncanonical_or_ambiguous_metadata(tmp_path):
    source, asset = _source(tmp_path)
    spec = tmp_path / "spec.json"
    cache = tmp_path / "cache"
    _write_spec(spec, asset)
    _bootstrap(spec, source, cache)
    metadata_path = cache / ".tatbot-body-cache.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata_path.chmod(0o644)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata_path.chmod(0o444)
    with pytest.raises(BodyCacheError) as caught:
        _audit(spec, cache)
    assert caught.value.code == "noncanonical_metadata"

    metadata_path.chmod(0o644)
    metadata_path.write_text(
        '{"schema":"tatbot.body-cache/1","schema":"tatbot.body-cache/1"}\n',
        encoding="utf-8",
    )
    metadata_path.chmod(0o444)
    with pytest.raises(BodyCacheError) as caught:
        _audit(spec, cache)
    assert caught.value.code == "invalid_metadata"


@pytest.mark.parametrize("relative", ["model/data.bin", ".tatbot-body-cache.json"])
def test_audit_refuses_writable_cache_files(tmp_path, relative):
    source, asset = _source(tmp_path)
    spec = tmp_path / "spec.json"
    cache = tmp_path / "cache"
    _write_spec(spec, asset)
    _bootstrap(spec, source, cache)
    (cache / relative).chmod(0o644)

    with pytest.raises(BodyCacheError) as caught:
        _audit(spec, cache)
    assert caught.value.code == "mutable_cache"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda value: value["model"].update(shape_components=True), "spec_value"),
        (lambda value: value["geometry"].update(vertices=True), "spec_type"),
        (lambda value: value["assets"][0].update(size=False), "asset_size"),
        (
            lambda value: value["coordinates"]["tatbot_from_soma"][0].__setitem__(0, True),
            "spec_type",
        ),
    ],
)
def test_model_spec_refuses_boolean_values_in_numeric_fields(tmp_path, mutate, code):
    source, asset = _source(tmp_path)
    spec = tmp_path / "spec.json"
    _write_spec(spec, asset)
    _mutate_spec(spec, mutate)

    with pytest.raises(BodyCacheError) as caught:
        _load_spec(spec)
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda value: value["geometry"].update(topology_sha256="0" * 64),
            "body_topology_mismatch",
        ),
        (
            lambda value: value["coordinates"].update(reference_rest_surface_sha256="0" * 64),
            "body_rest_surface_mismatch",
        ),
        (
            lambda value: value["coordinates"]["output"].update(up="+Y"),
            "body_units_or_axes_invalid",
        ),
    ],
)
def test_model_spec_uses_domain_refusals_for_geometry_identity(tmp_path, mutate, code):
    source, asset = _source(tmp_path)
    spec = tmp_path / "spec.json"
    _write_spec(spec, asset)
    _mutate_spec(spec, mutate)

    with pytest.raises(BodyCacheError) as caught:
        _load_spec(spec)
    assert caught.value.code == code


@pytest.mark.parametrize(
    ("cause", "public"),
    [
        ("mutable_revision", "body_model_unpinned"),
        ("invalid_json", "body_model_unpinned"),
        ("missing_source", "body_asset_missing"),
        ("missing_asset", "body_asset_missing"),
        ("digest_mismatch", "body_asset_hash_mismatch"),
        ("path_escape", "body_asset_path_denied"),
        ("mutable_cache", "body_asset_path_denied"),
        ("evidence_inside_cache", "body_asset_path_denied"),
        ("evidence_inside_repository", "body_asset_path_denied"),
        ("cache_inside_repository", "body_asset_path_denied"),
        ("evidence_exists", "body_asset_path_denied"),
        ("body_topology_mismatch", "body_topology_mismatch"),
        ("body_rest_surface_mismatch", "body_rest_surface_mismatch"),
        ("body_units_or_axes_invalid", "body_units_or_axes_invalid"),
    ],
)
def test_public_refusal_taxonomy(cause, public):
    assert _public_refusal_code(cause) == public


def test_cli_missing_cache_refusal_includes_stable_code_and_input_hash(capsys, tmp_path):
    spec = REPO / "config" / "body-models" / "mhr-soma-v1.json"
    ctx = SimpleNamespace(json=True, dry_run=False)
    ns = SimpleNamespace(spec=str(spec), cache_dir=str(tmp_path / "absent"), output=None)

    assert body_audit(ctx, ns, []) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload == {
        "action": "audit",
        "cause_code": "missing_cache",
        "code": "body_asset_missing",
        "detail": str(tmp_path / "absent"),
        "input_hashes": {"body_model_spec_file_sha256": _digest(spec)},
        "status": "refused",
    }


def test_cli_refuses_self_hashed_but_unreviewed_spec(capsys, tmp_path):
    source, asset = _source(tmp_path)
    spec = tmp_path / "spec.json"
    cache = tmp_path / "cache"
    _write_spec(spec, asset)
    ctx = SimpleNamespace(json=True, dry_run=False)
    ns = SimpleNamespace(
        spec=str(spec),
        cache_dir=str(cache),
        source_dir=str(source),
    )

    assert body_bootstrap(ctx, ns, []) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "body_model_unpinned"
    assert payload["cause_code"] == "body_model_unpinned"
    assert payload["input_hashes"] == {"body_model_spec_file_sha256": _digest(spec)}
    assert not cache.exists()


def test_audit_evidence_is_exclusive_read_only_and_outside_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    output = tmp_path / "evidence"
    payload = {
        "schema": "tatbot.body-audit/1",
        "status": "pass",
        "cache_dir": str(cache),
        "assets": [],
    }

    report = _write_audit_report(payload, output, cache)
    expected = _canonical(payload) + b"\n"
    assert report.read_bytes() == expected
    assert report.stat().st_mode & 0o222 == 0

    with pytest.raises(BodyCacheError) as caught:
        _write_audit_report({**payload, "status": "changed"}, output, cache)
    assert caught.value.code == "evidence_exists"
    assert report.read_bytes() == expected

    inside_cache = cache / "phase-0"
    with pytest.raises(BodyCacheError) as caught:
        _write_audit_report(payload, inside_cache, cache)
    assert caught.value.code == "evidence_inside_cache"
    assert not inside_cache.exists()

    repository = tmp_path / "repository"
    _mark_git_repository(repository)
    inside_repository = repository / "evidence"
    with pytest.raises(BodyCacheError) as caught:
        _write_audit_report(payload, inside_repository, cache)
    assert caught.value.code == "evidence_inside_repository"
    assert not inside_repository.exists()
