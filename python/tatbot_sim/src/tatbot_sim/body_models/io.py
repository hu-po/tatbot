"""Fail-closed validation for the immutable MHR/SOMA runtime inputs."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from typing import Any, NoReturn

from tatbot_contracts.body import (
    BodyCacheError,
    load_spec,
    public_refusal_code,
    require_reviewed_digest,
    verify_cache,
)

BODY_MODEL_SPEC_SCHEMA = "tatbot.body-model-spec/1"
BODY_CACHE_SCHEMA = "tatbot.body-cache/1"
BODY_CACHE_METADATA = ".tatbot-body-cache.json"


class BodyModelError(ValueError):
    """A stable refusal raised before untrusted model bytes are deserialized."""

    def __init__(self, code: str, path: str, detail: str):
        super().__init__(f"{code} at {path}: {detail}")
        self.code = code
        self.path = path
        self.detail = detail


def _fail(code: str, path: str | Path, detail: str) -> NoReturn:
    raise BodyModelError(code, str(path), detail)


def load_body_model_spec(path: str | Path) -> dict[str, Any]:
    """Refuse unreviewed specs before runtime construction."""
    try:
        spec, digest = load_spec(Path(path))
        require_reviewed_digest(digest)
        return spec
    except BodyCacheError as exc:
        raise BodyModelError(public_refusal_code(exc.code), str(path), exc.detail) from exc
    except OSError as exc:
        raise BodyModelError("body_model_unpinned", str(path), str(exc)) from exc


def verify_body_cache(spec: dict[str, Any], cache_dir: str | Path) -> Path:
    """Hash every allowlisted cache byte before SOMA or Torch sees it.

    Production callers obtain the spec from load_body_model_spec. The shared
    verifier rechecks its structure and digest here as well as every asset.
    """
    root = Path(cache_dir).expanduser()
    try:
        verify_cache(spec, root)
        return root.resolve(strict=True)
    except BodyCacheError as exc:
        raise BodyModelError(public_refusal_code(exc.code), str(root), exc.detail) from exc
    except OSError as exc:
        raise BodyModelError("body_asset_path_denied", str(root), str(exc)) from exc


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        _fail("body_model_unpinned", distribution, "required distribution is not installed")


def verify_software_lock(spec: dict[str, Any]) -> None:
    """Refuse a body runtime whose SOMA or Torch version differs from the lock."""

    package = spec["python_package"]
    actual_soma = _installed_version(package["name"])
    if actual_soma != package["version"]:
        _fail(
            "body_model_unpinned",
            package["name"],
            f"expected {package['version']}, got {actual_soma}",
        )
    expected_torch = spec["software_lock"]["torch"]
    actual_torch = _installed_version("torch").split("+", maxsplit=1)[0]
    if actual_torch != expected_torch:
        _fail("body_model_unpinned", "torch", f"expected {expected_torch}, got {actual_torch}")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        _fail("body_model_unpinned", "HF_HUB_OFFLINE", "must be 1 for model construction")
