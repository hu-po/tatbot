"""body -- argument and output adapters for the shared immutable cache contract."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from tatbot_contracts.body import (
    REVIEWED_SPEC_SHA256,
    BodyCacheError,
    audit,
    bootstrap,
    input_hashes,
    public_refusal_code,
    write_audit_report,
)

from tatbot_cli import EXIT_OK, EXIT_USAGE
from tatbot_cli.registry import OFFLINE, verb
from tatbot_cli.verbs._common import py

DOC = "docs/architecture.md#body-cache-verification"


def _export_args(parser):
    parser.add_argument("--python", required=True, help="interpreter with the pinned SOMA/Torch runtime")


@verb(effects=('read_files', 'write_files', 'start_process'), visibility="public", noun="body", verb="export", tier=OFFLINE,
      summary="bake genuine articulated SOMA poses into the shared Inkmap and simulation catalog",
      wraps=("web/inkmap/tools/export-soma.py",), args=_export_args,
      passthrough="export-soma.py", example=("--python", "/path/to/locked/python", "--", "--help"), doc=DOC,
      invariants=("Uses the pinned offline provider with pose correctives and audited immutable model assets.",
                  "--extend preserves the verified reference and existing pose bytes; records bounded platform arithmetic differences."))
def body_export(ctx, ns, rest):
    plan = py(ctx, "web/inkmap/tools/export-soma.py", *rest)
    plan.argv[0] = ns.python
    plan.env = {"HF_HUB_OFFLINE": "1"}
    return plan


def _emit(ctx, payload: dict[str, Any]) -> None:
    if ctx.json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        print(f"body {payload['status']}: {len(payload['assets'])} assets at {payload['cache_dir']}")
        print(f"  spec     {payload['model_spec_sha256']}")
        print(f"  revision {payload['asset_revision']}")
        print("  network  disabled by implementation")


def _refuse(
    ctx,
    action: str,
    exc: BodyCacheError,
    *,
    spec_path: Path | None = None,
    data_root: Path | None = None,
) -> int:
    code = public_refusal_code(exc.code)
    hashes = input_hashes(spec_path, data_root) if spec_path is not None else {}
    if ctx.json:
        print(
            json.dumps(
                {
                    "status": "refused",
                    "action": action,
                    "code": code,
                    "cause_code": exc.code,
                    "detail": exc.detail,
                    "input_hashes": hashes,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
    else:
        cause = f"; cause={exc.code}" if code != exc.code else ""
        print(f"tatbot body {action}: REFUSED [{code}{cause}]: {exc.detail}", file=sys.stderr)
        if hashes:
            print(f"  input hashes: {json.dumps(hashes, sort_keys=True, separators=(',', ':'))}", file=sys.stderr)
    return EXIT_USAGE


def _common_args(parser) -> None:
    parser.add_argument("--spec", required=True, help="tracked BodyModelSpec JSON")
    parser.add_argument("--cache-dir", required=True, help="explicit destination/cache root")


def _bootstrap_args(parser) -> None:
    _common_args(parser)
    parser.add_argument(
        "--source-dir",
        default=os.environ.get("TATBOT_BODY_SOURCE_DIR"),
        help="already-downloaded allowlisted bytes (or TATBOT_BODY_SOURCE_DIR); no downloader is included",
    )


@verb(effects=('read_files', 'write_files'), visibility="public", native=True, output="json",
    noun="body",
    verb="bootstrap",
    tier=OFFLINE,
    summary="copy a pinned, pre-downloaded MHR and SOMA allowlist into an explicit verified cache",
    args=_bootstrap_args,
    example=(
        "--spec",
        "config/body-models/mhr-soma-v1.json",
        "--cache-dir",
        "/tmp/tatbot-body-cache",
        "--source-dir",
        "/tmp/tatbot-body-source",
    ),
    doc=DOC,
    invariants=(
        "Contains no downloader; source bytes must already exist in --source-dir.",
        f"Accepts only reviewed BodyModelSpec digest {REVIEWED_SPEC_SHA256}.",
        "Refuses mutable revisions, unlisted files, symlinks, path escapes, size or digest mismatch, and non-empty destinations.",
        "Copies bytes before any TorchScript, USD, NPZ, or OBJ parser can see them.",
        "JSON refusals use the plan's stable body_* code and include available input SHA-256 values.",
    ),
)
def body_bootstrap(ctx, ns, rest):
    spec_path = Path(ns.spec)
    source_dir = Path(ns.source_dir) if ns.source_dir else None
    if rest:
        return _refuse(
            ctx,
            "bootstrap",
            BodyCacheError("unexpected_arguments", " ".join(rest)),
            spec_path=spec_path,
        )
    if not ns.source_dir:
        return _refuse(
            ctx,
            "bootstrap",
            BodyCacheError("missing_source", "pass --source-dir"),
            spec_path=spec_path,
        )
    assert source_dir is not None
    try:
        payload = bootstrap(
            spec_path,
            source_dir,
            Path(ns.cache_dir),
            require_reviewed_spec=True,
        )
    except BodyCacheError as exc:
        return _refuse(ctx, "bootstrap", exc, spec_path=spec_path)
    _emit(ctx, payload)
    return EXIT_OK


def _audit_args(parser) -> None:
    _common_args(parser)
    parser.add_argument("--output", help="optional directory for canonical audit.json evidence")


def _audit_effects(effects, ns, rest):
    return effects if ns.output else effects - {"write_files"}


@verb(refine_effects=_audit_effects, effects=('read_files', 'write_files'), visibility="public", native=True, output="json",
    noun="body",
    verb="audit",
    tier=OFFLINE,
    summary="read and hash every byte in an explicit MHR and SOMA cache without deserializing it",
    args=_audit_args,
    example=(
        "--spec",
        "config/body-models/mhr-soma-v1.json",
        "--cache-dir",
        "/tmp/tatbot-body-cache",
    ),
    doc=DOC,
    invariants=(
        "Read-only with respect to the cache and performs no network access or model deserialization.",
        f"Accepts only reviewed BodyModelSpec digest {REVIEWED_SPEC_SHA256}.",
        "Rejects writable cache files and cache or evidence paths inside a Git repository.",
        "The optional --output writes one read-only canonical audit.json outside the cache and refuses overwrite.",
        "JSON refusals use the plan's stable body_* code and include available input SHA-256 values.",
    ),
)
def body_audit(ctx, ns, rest):
    spec_path = Path(ns.spec)
    cache_dir = Path(ns.cache_dir)
    if rest:
        return _refuse(
            ctx,
            "audit",
            BodyCacheError("unexpected_arguments", " ".join(rest)),
            spec_path=spec_path,
            data_root=cache_dir,
        )
    try:
        payload = audit(spec_path, cache_dir, require_reviewed_spec=True)
        if ns.output:
            write_audit_report(payload, Path(ns.output), cache_dir)
        _emit(ctx, payload)
    except BodyCacheError as exc:
        return _refuse(ctx, "audit", exc, spec_path=spec_path, data_root=cache_dir)
    return EXIT_OK
