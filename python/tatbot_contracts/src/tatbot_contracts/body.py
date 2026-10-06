"""Immutable body-cache verification and bootstrap; stdlib and hash-only.

Both bootstrap and the runtime loader verify through this module. Reviewed-spec
policy belongs at each entry boundary; synthetic fixtures may exercise the same
structural verifier without claiming a reviewed model identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from .canonical import ContractError, canonical_bytes, parse_json
from .digest import sha256_file

SPEC_SCHEMA = "tatbot.body-model-spec/1"
METADATA = ".tatbot-body-cache.json"
REVIEWED_SPEC_SHA256 = "e615b8485c367509833ee68b0405cd1e0ce6015604eaa4c1b2b699f4fc8d5144"
REVIEWED_TOPOLOGY_SHA256 = "e0ca7ee25dc0b4c8d841bb2626e364bb88b7af7fae037e30854728842e320a18"
REVIEWED_LOW_VERTEX_MAP_SHA256 = "f51534883dc562a83c3d585bfada2e003a647b3642b6229e86d35e6459686d3b"
REVIEWED_MIRROR_MAP_SHA256 = "530b954b8087e4f100df8ae2ca24cfaa72f5e0046ba87332e801fb8fa033aa6b"
REVIEWED_REST_SURFACE_SHA256 = "caa66dff9b3625771c8f4c35bfe59556d30acdc3800880106f98f0ce75c49a95"
REVIEWED_UV_SHA256 = {
    "st": (
        "989b2b570c754401e89d4ada113ec5218604af84b680009febe0c701edbd5019",
        "7564d81af191a67beaa58505d0a1496a9e6f65e5be565208331038fe1bc302ca",
    ),
    "st1": (
        "5223e8f61eb4b8e4aca806eab8be7e81ff40118f8e7fbca8a0f868e0695abfa1",
        "b2c9bcdd779a5d0bd7353bc2ec5cf8ab97c9feec2abb55da5bae8b5f6affd970",
    ),
    "st2": (
        "0bdb293efbe9adcb0510de6c82004b7dfe03060aea051c1fd0d6671f028dd463",
        "9b1ddd066a8eb60c26026ac86f5002a507d9fe4204bf7093365cda15e04a7673",
    ),
}

_MISSING_CODES = {"missing_source", "missing_cache", "missing_asset"}
_HASH_MISMATCH_CODES = {
    "size_mismatch",
    "digest_mismatch",
    "copy_verification_failed",
    "metadata_mismatch",
    "invalid_metadata",
    "noncanonical_metadata",
}
_PATH_DENIED_CODES = {
    "path_escape",
    "noncanonical_path",
    "unlisted_file",
    "denied_path",
    "symlink",
    "special_file",
    "cache_not_empty",
    "mutable_cache",
    "staging_exists",
    "evidence_inside_cache",
    "evidence_inside_repository",
    "cache_inside_repository",
    "evidence_exists",
    "evidence_write_failed",
}
_UNPINNED_CODES = {
    "mutable_revision",
    "wrong_spec_hash",
    "wrong_schema",
    "invalid_spec",
    "spec_fields",
    "spec_type",
    "spec_value",
    "spec_hash",
    "asset_source_fields",
    "manifest_fields",
    "empty_allowlist",
    "asset_fields",
    "asset_size",
    "asset_hash",
    "asset_license",
    "asset_reason",
    "duplicate_asset",
    "denied_prefixes",
    "invalid_json",
    "duplicate_key",
    "non_finite",
    "negative_zero",
    "unsafe_integer",
}


class BodyCacheError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _canonical(value: Any, *, omit_digest: bool = False) -> bytes:
    try:
        return canonical_bytes(value, omit_digest=omit_digest)
    except ContractError as exc:
        code = "invalid_json" if exc.code == "wrong_type" else exc.code
        raise BodyCacheError(code, exc.detail) from exc


def _parse_json(data: str | bytes) -> Any:
    try:
        return parse_json(data)
    except ContractError as exc:
        raise BodyCacheError(exc.code, exc.detail) from exc


def _safe_relative(raw: str) -> PurePosixPath:
    if not isinstance(raw, str):
        raise BodyCacheError("noncanonical_path", repr(raw))
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise BodyCacheError("path_escape", raw)
    if path.as_posix() != raw or "\\" in raw:
        raise BodyCacheError("noncanonical_path", raw)
    return path


def _ensure_no_symlink_ancestors(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise BodyCacheError("symlink", str(candidate))


def _is_repository_root(path: Path) -> bool:
    marker = path / ".git"
    if marker.is_dir():
        return (marker / "HEAD").is_file() and (marker / "objects").is_dir()
    if marker.is_file():
        try:
            return marker.read_text(encoding="utf-8", errors="strict").startswith("gitdir: ")
        except OSError:
            return False
    return (path / "HEAD").is_file() and (path / "objects").is_dir() and (path / "refs").is_dir()


def _ensure_outside_repository(path: Path, code: str) -> None:
    resolved = path.resolve(strict=False)
    for candidate in (resolved, *resolved.parents):
        if _is_repository_root(candidate):
            raise BodyCacheError(code, str(resolved))


def _ensure_no_symlink(path: Path, root: Path) -> None:
    current = path
    while current != root.parent:
        if current.is_symlink():
            raise BodyCacheError("symlink", str(current))
        if current == root:
            break
        current = current.parent


def _fields(value: Any, path: str, required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BodyCacheError("spec_fields", f"{path} must be an object")
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        raise BodyCacheError("spec_fields", f"{path}: missing={missing}; unknown={unknown}")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise BodyCacheError("spec_type", f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BodyCacheError("spec_type", f"{path} must be a nonnegative integer")
    return value


def _number(value: Any, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BodyCacheError("spec_type", f"{path} must be a finite number")
    if value == 0 and math.copysign(1.0, value) < 0:
        raise BodyCacheError("negative_zero", path)
    return value


def _digest(value: Any, path: str) -> str:
    text = _text(value, path)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise BodyCacheError("spec_hash", path)
    return text


def _commit(value: Any, path: str) -> str:
    text = _text(value, path)
    if len(text) != 40 or any(char not in "0123456789abcdef" for char in text):
        raise BodyCacheError("mutable_revision", f"{path}={text!r}")
    return text


def _validate_source(value: Any, path: str, tag: str) -> None:
    source = _fields(value, path, {"repository", "tag", "commit", "archive_size", "archive_sha256"})
    repository = _text(source["repository"], f"{path}.repository")
    if not repository.startswith("https://"):
        raise BodyCacheError("mutable_revision", f"{path}.repository must be HTTPS")
    if source["tag"] != tag:
        raise BodyCacheError("mutable_revision", f"{path}.tag={source['tag']!r}")
    _commit(source["commit"], f"{path}.commit")
    if _integer(source["archive_size"], f"{path}.archive_size") == 0:
        raise BodyCacheError("spec_type", f"{path}.archive_size must be positive")
    _digest(source["archive_sha256"], f"{path}.archive_sha256")


def _validate_body_spec(spec: dict[str, Any]) -> None:
    sources = _fields(spec["sources"], "sources", {"soma_x", "mhr"})
    _validate_source(sources["soma_x"], "sources.soma_x", "v0.3.0")
    _validate_source(sources["mhr"], "sources.mhr", "v1.0.1")

    package = _fields(spec["python_package"], "python_package", {"name", "version", "python", "wheel", "sdist"})
    expected_package = {"name": "py-soma-x", "version": "0.3.0", "python": ">=3.11,<3.13"}
    for key, expected in expected_package.items():
        if package[key] != expected:
            raise BodyCacheError("spec_value", f"python_package.{key} must be {expected!r}")
    for distribution in ("wheel", "sdist"):
        item = _fields(package[distribution], f"python_package.{distribution}", {"filename", "size", "sha256"})
        _safe_relative(_text(item["filename"], f"python_package.{distribution}.filename"))
        if _integer(item["size"], f"python_package.{distribution}.size") == 0:
            raise BodyCacheError("spec_type", f"python_package.{distribution}.size must be positive")
        _digest(item["sha256"], f"python_package.{distribution}.sha256")

    expected_model = {
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
    model = _fields(spec["model"], "model", set(expected_model))
    for key, expected in expected_model.items():
        value = model[key]
        if type(value) is not type(expected) or value != expected:
            raise BodyCacheError(
                "spec_value",
                f"model.{key} does not match the reviewed MHR/SOMA dense contract",
            )

    geometry = _fields(
        spec["geometry"],
        "geometry",
        {"vertices", "triangles", "topology_sha256", "low_vertex_map_sha256", "mirror_map_sha256", "uv_sets"},
    )
    vertices = _integer(geometry["vertices"], "geometry.vertices")
    triangles = _integer(geometry["triangles"], "geometry.triangles")
    if vertices != 18056 or triangles != 36108:
        raise BodyCacheError("body_topology_mismatch", "geometry counts do not match SOMA mid")
    expected_geometry = {
        "topology_sha256": REVIEWED_TOPOLOGY_SHA256,
        "low_vertex_map_sha256": REVIEWED_LOW_VERTEX_MAP_SHA256,
        "mirror_map_sha256": REVIEWED_MIRROR_MAP_SHA256,
    }
    for key, expected in expected_geometry.items():
        _digest(geometry[key], f"geometry.{key}")
        if geometry[key] != expected:
            raise BodyCacheError("body_topology_mismatch", f"geometry.{key}")
    uv_sets = _fields(geometry["uv_sets"], "geometry.uv_sets", {"st", "st1", "st2"})
    for name in ("st", "st1", "st2"):
        uv = _fields(uv_sets[name], f"geometry.uv_sets.{name}", {"interpolation", "values_sha256", "indices_sha256"})
        if uv["interpolation"] != "faceVarying":
            raise BodyCacheError("body_topology_mismatch", f"geometry.uv_sets.{name}.interpolation")
        _digest(uv["values_sha256"], f"geometry.uv_sets.{name}.values_sha256")
        _digest(uv["indices_sha256"], f"geometry.uv_sets.{name}.indices_sha256")
        if (uv["values_sha256"], uv["indices_sha256"]) != REVIEWED_UV_SHA256[name]:
            raise BodyCacheError("body_topology_mismatch", f"geometry.uv_sets.{name}")

    coordinates = _fields(
        spec["coordinates"],
        "coordinates",
        {"upstream", "output", "tatbot_from_soma", "quantization_m", "reference_rest_surface_sha256"},
    )
    expected_upstream = {"unit": "cm", "handedness": "right", "up": "+Y", "forward": "+Z"}
    expected_output = {"unit": "m", "handedness": "right", "up": "+Z", "front": "-Y"}
    if coordinates["upstream"] != expected_upstream or coordinates["output"] != expected_output:
        raise BodyCacheError("body_units_or_axes_invalid", "coordinate frames do not match the reviewed transform")
    expected_transform = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    transform = coordinates["tatbot_from_soma"]
    if not isinstance(transform, list) or len(transform) != 4:
        raise BodyCacheError("body_units_or_axes_invalid", "coordinates.tatbot_from_soma must be a 4x4 matrix")
    for row_index, row in enumerate(transform):
        if not isinstance(row, list) or len(row) != 4:
            raise BodyCacheError("body_units_or_axes_invalid", "coordinates.tatbot_from_soma must be a 4x4 matrix")
        for column_index, value in enumerate(row):
            _number(value, f"coordinates.tatbot_from_soma[{row_index}][{column_index}]")
    quantization = _number(coordinates["quantization_m"], "coordinates.quantization_m")
    if transform != expected_transform or quantization != 0.00001:
        raise BodyCacheError("body_units_or_axes_invalid", "coordinate transform or quantization changed")
    _digest(coordinates["reference_rest_surface_sha256"], "coordinates.reference_rest_surface_sha256")
    if coordinates["reference_rest_surface_sha256"] != REVIEWED_REST_SURFACE_SHA256:
        raise BodyCacheError("body_rest_surface_mismatch", "coordinates.reference_rest_surface_sha256")

    software = _fields(spec["software_lock"], "software_lock", {"path", "sha256", "environment", "torch"})
    _safe_relative(_text(software["path"], "software_lock.path"))
    _digest(software["sha256"], "software_lock.sha256")
    if software["environment"] != "cp312-x86_64-manylinux_2_28" or software["torch"] != "2.13.0":
        raise BodyCacheError("spec_value", "software lock platform or Torch pin changed")

    security = _fields(
        spec["security"],
        "security",
        {"torchscript", "npz_allow_pickle", "usd_parse_stage", "automatic_download", "native_binary_audit"},
    )
    if security["torchscript"] not in {"weights-only", "review-required"}:
        raise BodyCacheError("spec_value", "security.torchscript")
    if security["npz_allow_pickle"] is not False or security["automatic_download"] is not False:
        raise BodyCacheError("spec_value", "pickle and automatic downloads must remain disabled")
    if security["usd_parse_stage"] != "post-audit":
        raise BodyCacheError("spec_value", "USD parsing must remain post-audit")
    _text(security["native_binary_audit"], "security.native_binary_audit")


def load_spec(path: Path) -> tuple[dict[str, Any], str]:
    _ensure_no_symlink_ancestors(path)
    try:
        if not path.is_file():
            raise BodyCacheError("invalid_spec", "spec must be a regular UTF-8 JSON file")
        spec = _parse_json(path.read_text(encoding="utf-8"))
    except BodyCacheError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BodyCacheError("invalid_spec", str(exc)) from exc
    return validate_spec(spec)


def validate_spec(spec: Any) -> tuple[dict[str, Any], str]:
    """Validate a self-hashed spec; reviewed identity is enforced by the caller."""
    if not isinstance(spec, dict) or spec.get("schema") != SPEC_SCHEMA:
        raise BodyCacheError("wrong_schema", str(spec.get("schema") if isinstance(spec, dict) else type(spec)))
    required = {
        "schema",
        "content_sha256",
        "name",
        "sources",
        "python_package",
        "asset_source",
        "assets",
        "denied_prefixes",
        "model",
        "geometry",
        "coordinates",
        "software_lock",
        "security",
    }
    unknown = sorted(set(spec) - required)
    missing = sorted(required - set(spec))
    if unknown or missing:
        raise BodyCacheError("spec_fields", f"missing={missing}; unknown={unknown}")
    actual = hashlib.sha256(_canonical(spec, omit_digest=True)).hexdigest()
    if spec["content_sha256"] != actual:
        raise BodyCacheError("wrong_spec_hash", f"declared {spec['content_sha256']}; computed {actual}")
    _text(spec["name"], "name")
    _validate_body_spec(spec)
    source = spec.get("asset_source")
    if not isinstance(source, dict) or set(source) != {"repository", "revision", "manifest"}:
        raise BodyCacheError("asset_source_fields", "asset_source must name repository, immutable revision, and manifest")
    repository = _text(source.get("repository"), "asset_source.repository")
    if not repository.startswith("https://"):
        raise BodyCacheError("mutable_revision", "asset_source.repository must be HTTPS")
    _commit(source.get("revision"), "asset_source.revision")
    manifest = source.get("manifest")
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"path", "size", "sha256"}
    ):
        raise BodyCacheError("manifest_fields", "manifest requires path, size, and sha256")
    _safe_relative(manifest["path"])
    if _integer(manifest["size"], "asset_source.manifest.size") == 0:
        raise BodyCacheError("manifest_fields", "manifest size must be positive")
    _digest(manifest["sha256"], "asset_source.manifest.sha256")
    if not isinstance(spec.get("assets"), list) or not spec["assets"]:
        raise BodyCacheError("empty_allowlist", "assets")
    seen: set[str] = set()
    for entry in spec["assets"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256", "license", "reason"}:
            raise BodyCacheError("asset_fields", repr(entry))
        rel = _safe_relative(entry["path"]).as_posix()
        if rel in seen:
            raise BodyCacheError("duplicate_asset", rel)
        seen.add(rel)
        if isinstance(entry["size"], bool) or not isinstance(entry["size"], int) or entry["size"] < 0:
            raise BodyCacheError("asset_size", rel)
        try:
            _digest(entry["sha256"], f"assets[{rel}].sha256")
        except BodyCacheError as exc:
            raise BodyCacheError("asset_hash", rel) from exc
        if not isinstance(entry["license"], str) or not entry["license"]:
            raise BodyCacheError("asset_license", rel)
        if not isinstance(entry["reason"], str) or not entry["reason"]:
            raise BodyCacheError("asset_reason", rel)
    denied = spec.get("denied_prefixes")
    if not isinstance(denied, list) or any(not isinstance(item, str) for item in denied):
        raise BodyCacheError("denied_prefixes", "expected string array")
    for raw in denied:
        _safe_relative(raw.rstrip("/") + "/sentinel")
    required_denials = {"SMPL/", "SMPLX/", "MANO/", "Anny/", "GarmentMeasurements/"}
    if not required_denials <= set(denied):
        raise BodyCacheError("denied_prefixes", "required model-family exclusions are missing")
    return spec, actual


def _inventory(root: Path) -> list[str]:
    _ensure_no_symlink_ancestors(root)
    if not root.is_dir():
        raise BodyCacheError("missing_cache", str(root))
    if root.is_symlink():
        raise BodyCacheError("symlink", str(root))
    result: list[str] = []
    for path in sorted(root.rglob("*")):
        _ensure_no_symlink(path, root)
        if path.is_file():
            result.append(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise BodyCacheError("special_file", str(path))
    return result


def require_reviewed_digest(spec_digest: str) -> None:
    if spec_digest != REVIEWED_SPEC_SHA256:
        raise BodyCacheError(
            "body_model_unpinned",
            f"expected reviewed spec {REVIEWED_SPEC_SHA256}; got {spec_digest}",
        )


def audit(
    spec_path: Path,
    cache_dir: Path,
    *,
    require_reviewed_spec: bool = False,
) -> dict[str, Any]:
    spec, spec_digest = load_spec(spec_path)
    if require_reviewed_spec:
        require_reviewed_digest(spec_digest)
    return verify_cache(spec, cache_dir)


def verify_cache(spec: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    """Read and hash every asset immediately before any runtime deserialization."""
    spec, spec_digest = validate_spec(spec)
    _ensure_outside_repository(cache_dir, "cache_inside_repository")
    allow = {entry["path"]: entry for entry in spec["assets"]}
    found = _inventory(cache_dir)
    expected = sorted([*allow, METADATA])
    unlisted = sorted(set(found) - set(expected))
    missing = sorted(set(expected) - set(found))
    if unlisted:
        raise BodyCacheError("unlisted_file", ", ".join(unlisted))
    if missing:
        raise BodyCacheError("missing_asset", ", ".join(missing))
    denied = tuple(str(item).rstrip("/") + "/" for item in spec["denied_prefixes"])
    for rel in found:
        if rel.startswith(denied):
            raise BodyCacheError("denied_path", rel)
    verified = []
    for rel, entry in sorted(allow.items()):
        path = cache_dir / rel
        _ensure_no_symlink(path, cache_dir)
        size = path.stat().st_size
        if size != entry["size"]:
            raise BodyCacheError("size_mismatch", f"{rel}: expected {entry['size']}, got {size}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise BodyCacheError("digest_mismatch", f"{rel}: expected {entry['sha256']}, got {digest}")
        if path.stat().st_mode & 0o222:
            raise BodyCacheError("mutable_cache", rel)
        verified.append({"path": rel, "size": size, "sha256": digest})
    metadata_path = cache_dir / METADATA
    if metadata_path.stat().st_mode & 0o222:
        raise BodyCacheError("mutable_cache", METADATA)
    try:
        metadata_bytes = metadata_path.read_bytes()
        metadata = _parse_json(metadata_bytes.decode("utf-8"))
    except BodyCacheError as exc:
        raise BodyCacheError("invalid_metadata", f"{exc.code}: {exc.detail}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BodyCacheError("invalid_metadata", str(exc)) from exc
    expected_metadata = {
        "schema": "tatbot.body-cache/1",
        "model_spec_sha256": spec_digest,
        "asset_revision": spec["asset_source"]["revision"],
        "assets": [entry["path"] for entry in spec["assets"]],
    }
    if metadata != expected_metadata:
        raise BodyCacheError("metadata_mismatch", METADATA)
    if metadata_bytes != _canonical(expected_metadata) + b"\n":
        raise BodyCacheError("noncanonical_metadata", METADATA)
    return {
        "schema": "tatbot.body-audit/1",
        "status": "pass",
        "model_spec_sha256": spec_digest,
        "asset_revision": spec["asset_source"]["revision"],
        "cache_dir": str(cache_dir.resolve()),
        "assets": verified,
        "network_access": False,
        "deserialization": False,
    }


def bootstrap(
    spec_path: Path,
    source_dir: Path,
    cache_dir: Path,
    *,
    require_reviewed_spec: bool = False,
) -> dict[str, Any]:
    spec, spec_digest = load_spec(spec_path)
    if require_reviewed_spec:
        require_reviewed_digest(spec_digest)
    _ensure_outside_repository(cache_dir, "cache_inside_repository")
    allow = {entry["path"]: entry for entry in spec["assets"]}
    found = _inventory(source_dir)
    unlisted = sorted(set(found) - set(allow))
    missing = sorted(set(allow) - set(found))
    if unlisted:
        raise BodyCacheError("unlisted_file", ", ".join(unlisted))
    if missing:
        raise BodyCacheError("missing_asset", ", ".join(missing))
    for rel, entry in sorted(allow.items()):
        path = source_dir / rel
        _ensure_no_symlink(path, source_dir)
        if path.stat().st_size != entry["size"]:
            raise BodyCacheError("size_mismatch", rel)
        if sha256_file(path) != entry["sha256"]:
            raise BodyCacheError("digest_mismatch", rel)
    if cache_dir.exists():
        if cache_dir.is_symlink():
            raise BodyCacheError("symlink", str(cache_dir))
        if not cache_dir.is_dir() or any(cache_dir.iterdir()):
            raise BodyCacheError("cache_not_empty", str(cache_dir))
        cache_dir.rmdir()
    _ensure_no_symlink_ancestors(cache_dir.parent)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    _ensure_no_symlink_ancestors(cache_dir.parent)
    staging = cache_dir.parent / f".{cache_dir.name}.bootstrap-{os.getpid()}"
    if staging.exists():
        raise BodyCacheError("staging_exists", str(staging))
    try:
        staging.mkdir(mode=0o700)
        for rel, entry in sorted(allow.items()):
            source = source_dir / rel
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if target.stat().st_size != entry["size"] or sha256_file(target) != entry["sha256"]:
                raise BodyCacheError("copy_verification_failed", rel)
            target.chmod(0o444)
        metadata = {
            "schema": "tatbot.body-cache/1",
            "model_spec_sha256": spec_digest,
            "asset_revision": spec["asset_source"]["revision"],
            "assets": [entry["path"] for entry in spec["assets"]],
        }
        (staging / METADATA).write_bytes(_canonical(metadata) + b"\n")
        (staging / METADATA).chmod(0o444)
        os.replace(staging, cache_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return audit(spec_path, cache_dir, require_reviewed_spec=require_reviewed_spec)


def public_refusal_code(code: str) -> str:
    if code in {
        "body_model_unpinned",
        "body_asset_missing",
        "body_asset_hash_mismatch",
        "body_asset_path_denied",
        "body_topology_mismatch",
        "body_rest_surface_mismatch",
        "body_units_or_axes_invalid",
    }:
        return code
    if code in _MISSING_CODES:
        return "body_asset_missing"
    if code in _HASH_MISMATCH_CODES:
        return "body_asset_hash_mismatch"
    if code in _PATH_DENIED_CODES:
        return "body_asset_path_denied"
    if code in _UNPINNED_CODES:
        return "body_model_unpinned"
    return code


def input_hashes(spec_path: Path, data_root: Path | None = None) -> dict[str, str]:
    candidates = {"body_model_spec_file_sha256": spec_path}
    if data_root is not None:
        candidates["body_cache_metadata_file_sha256"] = data_root / METADATA
    hashes: dict[str, str] = {}
    for name, path in candidates.items():
        try:
            _ensure_no_symlink_ancestors(path)
            if path.is_file():
                hashes[name] = sha256_file(path)
        except (BodyCacheError, OSError):
            continue
    return hashes


def write_audit_report(payload: dict[str, Any], output: Path, cache_dir: Path) -> Path:
    """Write one immutable report outside the audited cache without clobbering evidence."""

    try:
        _ensure_no_symlink_ancestors(output)
        cache_root = cache_dir.resolve(strict=True)
        output_root = output.resolve(strict=False)
    except BodyCacheError:
        raise
    except OSError as exc:
        raise BodyCacheError("evidence_write_failed", str(exc)) from exc
    report_target = output_root / "audit.json"
    if report_target == cache_root or cache_root in report_target.parents:
        raise BodyCacheError("evidence_inside_cache", str(report_target))
    _ensure_outside_repository(report_target, "evidence_inside_repository")

    try:
        output.mkdir(parents=True, exist_ok=True)
        _ensure_no_symlink_ancestors(output)
        report = output / "audit.json"
        _ensure_no_symlink_ancestors(report)
        with report.open("xb") as stream:
            stream.write(_canonical(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        report.chmod(0o444)
    except BodyCacheError:
        raise
    except FileExistsError as exc:
        raise BodyCacheError("evidence_exists", str(output / "audit.json")) from exc
    except OSError as exc:
        raise BodyCacheError("evidence_write_failed", str(exc)) from exc
    return report
