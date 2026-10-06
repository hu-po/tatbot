"""Thin consumer for the canonical TypeScript InkLang contracts.

Python deliberately owns no placement grammar, realizer, or semantic face
picker. It invokes the checked-in batch entrypoint and validates the returned
resolution against the same body atlas before simulation consumes it.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from typing import Any

from tatbot_sim.inkmap.contracts import canonical_json_bytes
from tatbot_sim.inkmap.gltf_surface import MODEL_ID
from tatbot_sim.repo import repo_root

# Mirrors RESOLUTION_SCHEMA_VERSION in web/inkmap/src/core/inklang/types.ts.
RESOLUTION_SCHEMA_VERSION = 2

RESOLVER = repo_root() / "web" / "inkmap" / "tools" / "resolve.ts"
REQUEST_ADAPTER = repo_root() / "web" / "inkmap" / "tools" / "parse_request.ts"
CONFIG = repo_root() / "config" / "inkmap"
BODY_CATALOG = CONFIG / "body-poses.json"
ATLAS_ROOT = repo_root() / "web" / "inkmap" / "public" / "bodies"


class InkLangConsumerError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _node() -> str:
    executable = shutil.which("node")
    if executable is None:
        raise InkLangConsumerError(
            "INKLANG_RUNTIME_UNAVAILABLE",
            "canonical InkLang resolution requires Node.js 22 or newer",
        )
    return executable


def _run_json(argv: list[str], *, input_document: object | None = None) -> Any:
    completed = subprocess.run(
        [_node(), "--experimental-strip-types", *argv],
        cwd=repo_root(),
        input=(canonical_json_bytes(input_document).decode() if input_document is not None else None),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no JSON output"
        if "ERR_NO_TYPESCRIPT" in detail:
            # Debian/Ubuntu build node 22 without Amaro. The flag is accepted and
            # the strip still refuses, so this reads as a resolver fault unless
            # it is named; `npm ci` cannot fix it.
            detail = (
                "this node's Node.js is built without TypeScript support "
                "(ERR_NO_TYPESCRIPT); the InkLang resolver is a .ts module, so it "
                "needs a Node.js 22 build with type stripping -- " + detail.splitlines()[0]
            )
        elif "ERR_MODULE_NOT_FOUND" in detail:
            # The resolver imports the browser's dependencies (three, ...).
            # Name the fix instead of handing back a Node stack trace.
            detail = (
                "the InkLang resolver needs web/inkmap/node_modules; run "
                "`npm ci` in web/inkmap (scripts/check sim does this) -- " + detail.splitlines()[0]
            )
        raise InkLangConsumerError("INKLANG_RUNTIME_UNAVAILABLE", detail) from exc
    return document


def parse_tattoo_request(prompt: str) -> dict[str, Any]:
    """Parse the legacy design-plus-placement wrapper in the TS adapter."""
    result = _run_json([str(REQUEST_ADAPTER), prompt])
    if not isinstance(result, dict) or not result.get("ok"):
        code = str(result.get("code", "INKLANG_PARSE_SYNTAX")) if isinstance(result, dict) else "INKLANG_PARSE_SYNTAX"
        message = str(result.get("error", "invalid tattoo request")) if isinstance(result, dict) else "invalid tattoo request"
        raise InkLangConsumerError(code, message)
    if not isinstance(result.get("program"), dict) or not isinstance(result.get("placement_intent"), dict):
        raise InkLangConsumerError("INKLANG_RUNTIME_UNAVAILABLE", "tattoo request adapter omitted its typed intent")
    return result


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_resolution(document: object, *, expected_body: str | None = None) -> dict[str, Any]:
    """Validate structure, body/surface identity, anchor, and atlas semantics."""
    if not isinstance(document, dict):
        raise InkLangConsumerError("INKLANG_SEMANTIC_MISMATCH", "resolution is not an object")
    if document.get("resolution_schema_version") != RESOLUTION_SCHEMA_VERSION:
        raise InkLangConsumerError(
            "INKLANG_VERSION_MISMATCH",
            f"resolution schema must be {RESOLUTION_SCHEMA_VERSION}",
        )
    status = document.get("status")
    if status != "resolved":
        issues = document.get("issues")
        issue = issues[0] if isinstance(issues, list) and issues and isinstance(issues[0], dict) else {}
        raise InkLangConsumerError(str(issue.get("code", "INKLANG_SEMANTIC_MISMATCH")), str(issue.get("message", status)))
    body = document.get("body")
    if not isinstance(body, dict) or not isinstance(body.get("model_spec_id"), str):
        raise InkLangConsumerError("INKLANG_BODY_MISMATCH", "resolution body identity is incomplete")
    body_id = body["model_spec_id"]
    if expected_body is not None and body_id != expected_body:
        raise InkLangConsumerError("INKLANG_BODY_MISMATCH", f"resolved {body_id}, expected {expected_body}")
    if body_id != MODEL_ID:
        raise InkLangConsumerError("INKLANG_UNKNOWN_BODY", f"unsupported schema/model: {body_id!r}")
    catalog = json.loads(BODY_CATALOG.read_text())
    atlas_path = ATLAS_ROOT / f"{body_id}.regions.json"
    if not atlas_path.is_file():
        raise InkLangConsumerError("INKLANG_MISSING_ATLAS", str(atlas_path))
    atlas = json.loads(atlas_path.read_text())
    surface = body.get("rest_surface_sha256")
    if (
        atlas.get("atlas_schema_version") != document.get("atlas_schema_version")
        or atlas.get("inklang_version") != document.get("intent", {}).get("inklang_version")
    ):
        raise InkLangConsumerError("INKLANG_VERSION_MISMATCH", "resolution, intent, and atlas versions differ")
    if (
        body != atlas.get("body")
        or catalog.get("model_spec_id") != body_id
        or catalog.get("model_spec_sha256") != body.get("model_spec_sha256")
        or catalog.get("identity_sha256") != body.get("identity_sha256")
        or catalog.get("topology_sha256") != body.get("topology_sha256")
        or catalog.get("rest_surface_sha256") != surface
        or catalog.get("rest_asset", {}).get("sha256") != body.get("asset_sha256")
    ):
        raise InkLangConsumerError("INKLANG_SURFACE_MISMATCH", f"{body_id} surface identity differs")
    anchor = document.get("anchor")
    if not isinstance(anchor, dict) or not isinstance(anchor.get("face"), int):
        raise InkLangConsumerError("INKLANG_ANCHOR_INVALID", "anchor face is missing")
    bary = anchor.get("barycentric")
    face = anchor["face"]
    faces = atlas.get("faces")
    eligible = atlas.get("eligible_faces")
    if (
        not isinstance(faces, list)
        or not isinstance(eligible, list)
        or len(eligible) != len(faces)
        or face < 0
        or face >= len(faces)
        or not isinstance(bary, list)
        or len(bary) != 3
        or not all(_finite(value) and 0 <= value <= 1 for value in bary)
        or abs(sum(bary) - 1.0) > 1e-6
        or faces[face] < 0
        or eligible[face] != 1
    ):
        raise InkLangConsumerError("INKLANG_ANCHOR_INVALID", f"invalid anchor on face {face}")
    code = faces[face]
    atlas_site = atlas["sites"][code >> 2]
    atlas_laterality = {0: None, 1: "left", 2: "right"}.get(code & 3)
    actual = document.get("actual")
    if (
        not isinstance(actual, dict)
        or actual.get("site_id") != atlas_site
        or actual.get("laterality") != atlas_laterality
    ):
        raise InkLangConsumerError("INKLANG_SEMANTIC_MISMATCH", "actual site does not match the anchor's atlas region")
    return document


def resolve_batch(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve a batch once, preserving TS canonical JSON at the boundary."""
    result = _run_json([str(RESOLVER), "--input", "-"], input_document=requests)
    if not isinstance(result, list) or len(result) != len(requests):
        raise InkLangConsumerError("INKLANG_RUNTIME_UNAVAILABLE", "resolver returned the wrong batch shape")
    return [validate_resolution(item, expected_body=MODEL_ID) for item in result]


def resolve_placement(request: dict[str, Any]) -> dict[str, Any]:
    return resolve_batch([request])[0]
