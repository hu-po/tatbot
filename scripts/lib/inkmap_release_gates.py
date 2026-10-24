"""Release gates: four approvals, all explicit and fail-closed.

`scripts/inkmap_deploy.sh` loads this file by path, with only
python/tatbot_contracts on the import path, and refuses a deployment or a
public export before it reads any secret unless the approval is recorded in
config/human-representation/release-gates.json.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from tatbot_contracts.canonical import ContractError, canonical_digest, parse_json

SCHEMA = "tatbot.human-representation-release-gates/1"
APPROVALS = (
    "inkmap_deployment",
    "public_release",
    "powered_evaluation",
    "human_facing_study",
)
ACTIONS = (
    "deployment_enabled",
    "public_export_enabled",
    "powered_motion_enabled",
    "human_contact_enabled",
)
CLAIMS = (
    "core_nominal_body",
    "offline_full_pipeline",
    "model_backed_training",
    "compliant_mechanics",
    "anatomy_prior",
)
APPROVAL_FOR_ACTION = dict(zip(ACTIONS, APPROVALS, strict=True))


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("wrong_type", path, "expected object")
    return value


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("wrong_type", path, "expected non-empty string")
    return value


def _sha256(value: Any, path: str) -> str:
    text = _nonempty(value, path)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ContractError("wrong_hash", path, "expected lowercase SHA-256")
    return text


def load_release_gates(path: str | Path) -> dict[str, Any]:
    """Load a gate file with the strict JSON and release-gate semantic readers."""

    try:
        value = parse_json(Path(path).read_bytes())
    except OSError as exc:
        raise ContractError("body_asset_missing", "$", str(exc)) from exc
    return validate_release_gates(_object(value, "$"))


def _validate_claims(item: dict) -> None:
    claims = _object(item["claims"], "$.claims")
    if set(claims) != set(CLAIMS):
        raise ContractError(
            "execution_binding_mismatch", "$.claims", "exact named capability claims are required"
        )
    for name in CLAIMS:
        claim = _object(claims[name], f"$.claims.{name}")
        if set(claim) != {"enabled", "reason"} or not isinstance(claim.get("enabled"), bool):
            raise ContractError(
                "execution_binding_mismatch", f"$.claims.{name}", "enabled and reason are required"
            )
        _nonempty(claim.get("reason"), f"$.claims.{name}.reason")


def _validate_approval(approvals: dict, name: str) -> None:
    approval = _object(approvals[name], f"$.approvals.{name}")
    if set(approval) != {"status", "approved", "reviewer", "reviewed_utc", "evidence_sha256"}:
        raise ContractError("execution_binding_mismatch", f"$.approvals.{name}", "approval fields differ")
    if approval["status"] not in {"pending", "approved", "rejected"} or not isinstance(
        approval["approved"], bool
    ):
        raise ContractError("execution_binding_mismatch", f"$.approvals.{name}", "invalid approval state")
    if approval["approved"]:
        if approval["status"] != "approved" or not all(
            isinstance(approval[field], str) and approval[field]
            for field in ("reviewer", "reviewed_utc", "evidence_sha256")
        ):
            raise ContractError(
                "execution_binding_mismatch", f"$.approvals.{name}", "approved gate lacks signed evidence"
            )
        _sha256(approval["evidence_sha256"], f"$.approvals.{name}.evidence_sha256")
    elif approval["status"] == "approved":
        raise ContractError(
            "execution_binding_mismatch", f"$.approvals.{name}", "status and boolean disagree"
        )
    elif approval["status"] == "pending" and any(
        approval[field] is not None for field in ("reviewer", "reviewed_utc", "evidence_sha256")
    ):
        raise ContractError(
            "execution_binding_mismatch", f"$.approvals.{name}", "pending approval must be unsigned"
        )
    elif approval["status"] == "rejected":
        _nonempty(approval["reviewer"], f"$.approvals.{name}.reviewer")
        _nonempty(approval["reviewed_utc"], f"$.approvals.{name}.reviewed_utc")
        _sha256(approval["evidence_sha256"], f"$.approvals.{name}.evidence_sha256")


def _validate_actions(item: dict, approvals: dict) -> None:
    actions = _object(item["actions"], "$.actions")
    if set(actions) != set(ACTIONS) or any(not isinstance(value, bool) for value in actions.values()):
        raise ContractError("execution_binding_mismatch", "$.actions", "exact action booleans are required")
    for action, approval in APPROVAL_FOR_ACTION.items():
        if actions[action] and not approvals[approval]["approved"]:
            raise ContractError(
                "execution_binding_mismatch", f"$.actions.{action}", f"{approval} is not approved"
            )


def _validate_rollback(item: dict) -> None:
    rollback = _object(item["rollback"], "$.rollback")
    if set(rollback) != {"strategy", "reintroduces_retired_body", "steps", "known_good_mhr_soma_revision"}:
        raise ContractError("execution_binding_mismatch", "$.rollback", "rollback fields differ")
    if rollback.get("strategy") not in {"disable-consumer", "deploy-known-good-mhr-soma-only"}:
        raise ContractError(
            "execution_binding_mismatch", "$.rollback.strategy", str(rollback.get("strategy"))
        )
    if rollback.get("reintroduces_retired_body") is not False:
        raise ContractError(
            "body_model_unsupported", "$.rollback", "rollback may not restore a second body path"
        )
    steps = rollback["steps"]
    if not isinstance(steps, list) or not steps:
        raise ContractError(
            "execution_binding_mismatch", "$.rollback.steps", "at least one rollback step is required"
        )
    for index, step in enumerate(steps):
        _nonempty(step, f"$.rollback.steps[{index}]")
    revision = rollback["known_good_mhr_soma_revision"]
    if revision is not None:
        _nonempty(revision, "$.rollback.known_good_mhr_soma_revision")


def _validate_provenance(item: dict) -> None:
    provenance = _object(item["provenance"], "$.provenance")
    if set(provenance) != {"producer", "version", "created_utc", "source_sha256"}:
        raise ContractError("execution_binding_mismatch", "$.provenance", "provenance fields differ")
    for field in ("producer", "version", "created_utc"):
        _nonempty(provenance[field], f"$.provenance.{field}")
    _sha256(provenance["source_sha256"], "$.provenance.source_sha256")


def validate_release_gates(value: Mapping[str, Any]) -> dict[str, Any]:
    item = deepcopy(dict(value))
    required = {
        "schema",
        "content_sha256",
        "repository_default",
        "claims",
        "approvals",
        "actions",
        "rollback",
        "provenance",
    }
    if set(item) != required:
        raise ContractError(
            "execution_binding_mismatch", "$", f"release-gate fields differ: {sorted(set(item) ^ required)}"
        )
    if item["schema"] != SCHEMA or item["repository_default"] != "mhr-soma-v1":
        raise ContractError(
            "body_model_unpinned", "$.repository_default", str(item.get("repository_default"))
        )
    if item["content_sha256"] != canonical_digest(item):
        raise ContractError("wrong_hash", "$.content_sha256", "release gates digest mismatch")
    _validate_claims(item)
    approvals = _object(item["approvals"], "$.approvals")
    if set(approvals) != set(APPROVALS):
        raise ContractError(
            "execution_binding_mismatch", "$.approvals", "exactly four named approvals are required"
        )
    for name in APPROVALS:
        _validate_approval(approvals, name)
    _validate_actions(item, approvals)
    _validate_rollback(item)
    _validate_provenance(item)
    return item


def assert_action_enabled(gates: Mapping[str, Any], action: str) -> None:
    document = validate_release_gates(gates)
    if action not in ACTIONS:
        raise ContractError("execution_binding_mismatch", "$.actions", f"unknown action {action}")
    if not document["actions"][action]:
        approval = APPROVAL_FOR_ACTION[action]
        raise ContractError(
            "execution_binding_mismatch", f"$.actions.{action}", f"{approval} remains pending"
        )
