"""Fail-closed admission for bounded synthetic MHR/SOMA identities."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from tatbot_sim.human_rep.contracts import canonical_digest, load_contract
from tatbot_sim.repo import repo_root

CATALOG_PATH = repo_root() / "config" / "inkmap" / "synthetic-identities.json"
REFERENCE_PATH = repo_root() / "config" / "body-models" / "mhr-soma-v1" / "identity.json"


def load_identity_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != "tatbot.inkmap-synthetic-identity-catalog/1":
        raise ValueError("unsupported synthetic identity catalog")
    identities = value.get("identities")
    if not isinstance(identities, list) or not identities:
        raise ValueError("synthetic identity catalog is empty")
    if len({item.get("id") for item in identities}) != len(identities):
        raise ValueError("duplicate synthetic identity ID")
    return value


def _parameters(pattern: str, length: int) -> list[float]:
    if pattern == "zero":
        return [0.0] * length
    if pattern == "all-plus-2":
        return [2.0] * length
    if pattern == "all-minus-2":
        return [-2.0] * length
    if pattern == "alternating-plus-minus-2":
        return [2.0 if index % 2 == 0 else -2.0 for index in range(length)]
    raise ValueError(f"unknown bounded identity pattern {pattern!r}")


def identity_contract(identity_id: str, *, require_admitted: bool = True) -> dict[str, Any]:
    """Rebuild a pinned candidate contract; refuse unreviewed data use."""

    catalog = load_identity_catalog()
    try:
        record = next(item for item in catalog["identities"] if item["id"] == identity_id)
    except StopIteration as exc:
        raise ValueError(f"unsupported synthetic identity {identity_id!r}") from exc
    review = record.get("review", {})
    if require_admitted and review.get("human_visual") not in {"accepted", "accepted_reference"}:
        raise ValueError(f"identity_not_admitted: {identity_id} human visual review is pending")
    reference = load_contract(REFERENCE_PATH, expected_schema="tatbot.body-identity/1")
    if reference["model_spec_sha256"] != catalog["model_spec_sha256"]:
        raise ValueError("synthetic identity catalog model digest differs from reference")
    if identity_id == "reference":
        if reference["content_sha256"] != record["identity_sha256"]:
            raise ValueError("reference identity digest differs from synthetic catalog")
        return reference
    contract = copy.deepcopy(reference)
    contract["coefficients"] = _parameters(record["coefficient_pattern"], 45)
    contract["scales"] = _parameters(record["scale_pattern"], 68)
    contract["rest_surface_sha256"] = record["rest_surface_sha256"]
    # The audit's evidence provenance is part of the identity digest. Preserve
    # it exactly so the concise catalogue reconstructs the reviewed contract.
    seed = {"reference": 0, "mixed-alternating-2sigma": 46, "mixed-positive-2sigma": 47, "mixed-negative-2sigma": 48}[identity_id]
    contract["provenance"] = {
        "producer": "tatbot_sim.body_models.audit",
        "version": "1",
        "created_utc": "2026-09-04T18:17:59Z",
        "source_sha256": catalog["model_spec_sha256"],
        "seed": seed,
    }
    contract["content_sha256"] = canonical_digest(contract)
    if contract["content_sha256"] != record["identity_sha256"]:
        raise ValueError(f"synthetic identity {identity_id} no longer matches reviewed digest")
    return contract


def admitted_identity_ids() -> tuple[str, ...]:
    catalog = load_identity_catalog()
    return tuple(
        item["id"]
        for item in catalog["identities"]
        if item.get("review", {}).get("human_visual") in {"accepted", "accepted_reference"}
    )
