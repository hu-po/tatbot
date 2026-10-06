"""Strict local simulation interchange, sharing the browser's derivation rule."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from tatbot_sim.human_rep.artwork_client import artwork_request
from tatbot_sim.human_rep.contracts import (
    ContractError,
    canonical_bytes,
    canonical_digest,
    validate_contract,
)
from tatbot_sim.inkmap.contracts import validate_placement
from tatbot_sim.inkmap.rig import BODY_ASSET_ROOT, CATALOG_PATH
from tatbot_sim.repo import repo_root

MAX_BUNDLE_BYTES = 20_000_000
# Compilation and surface consumers validate the same frozen bundle repeatedly.
# Cache only successful structural/derivation checks, keyed by ALL actual bytes.
# Local asset bytes are still checked independently by verify_local_assets.
_VERIFIED_BUNDLES: OrderedDict[str, None] = OrderedDict()


def _schema_check(value, schema_name="inkmap/sim-bundle"):
    registry = Registry()
    paths = ["inkmap/placement", "inkmap/artwork", "inkmap/sim-bundle", "human-representation/common",
             "human-representation/tattoo-program", "human-representation/surface-placement",
             "human-representation/ink-program", "human-representation/surface-curve"]
    for name in paths:
        schema = json.loads((repo_root() / f"config/{name}.schema.json").read_text())
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    schema = json.loads((repo_root() / f"config/{schema_name}.schema.json").read_text())
    error = next(Draft202012Validator(schema, registry=registry).iter_errors(value), None)
    if error:
        raise ContractError("sim_bundle_invalid", error.json_path, error.message)


def validate_simulation_bundle(value: Any) -> dict:
    payload = canonical_bytes(value)
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ContractError("sim_bundle_over_budget", "$", "maximum 20 MB")
    fingerprint = hashlib.sha256(payload).hexdigest()
    if fingerprint in _VERIFIED_BUNDLES:
        _VERIFIED_BUNDLES.move_to_end(fingerprint)
        return deepcopy(value)
    _schema_check(value)
    if canonical_digest(value) != value["content_sha256"]:
        raise ContractError("wrong_hash", "$.content_sha256", "bundle content differs")
    validate_placement(value["placement_file"])
    result = artwork_request({"operation": "validate_bundle", "bundle": value})
    if canonical_bytes(result) != payload:
        raise ContractError("wrong_hash", "$", "browser and Python bundle readers differ")
    for record in result["artworks"].values():
        validate_contract(record["program"], expected_schema="tatbot.tattoo-program/1")
    for record in result["surface_placements"]:
        validate_contract(record["placement"], expected_schema="tatbot.surface-placement/1")
    _VERIFIED_BUNDLES[fingerprint] = None
    if len(_VERIFIED_BUNDLES) > 32:
        _VERIFIED_BUNDLES.popitem(last=False)
    return result


def make_simulation_bundle(file: dict, request: dict, *, sources: dict | None = None) -> dict:
    return validate_simulation_bundle(artwork_request({
        "operation": "materialize_bundle", "file": file, "request": request, "sources": sources or {},
    }))


def verify_local_assets(bundle: dict) -> None:
    """Resolve only manifest keys through the installed, pinned pose catalog."""
    catalog_bytes = CATALOG_PATH.read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != bundle["request"]["pose_catalog_sha256"]:
        raise ContractError("wrong_hash", "$.request.pose_catalog_sha256", "installed catalog bytes differ")
    catalog = json.loads(catalog_bytes)
    mapping = {"body-rest": "rest_asset", "body-poses": "pose_asset", "body-exclusions": "exclusion_asset"}
    for record in bundle["assets"]:
        asset = catalog[mapping[record["key"]]]
        path = (BODY_ASSET_ROOT / asset["path"]).resolve()
        if not path.is_relative_to(BODY_ASSET_ROOT.resolve()):
            raise ContractError("sim_bundle_invalid", "$.assets", "catalog asset escapes its local root")
        try:
            size = path.stat().st_size
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as error:
            raise ContractError("sim_bundle_missing_asset", "$.assets", record["key"]) from error
        if size != record["byte_length"] or digest != record["sha256"]:
            raise ContractError("wrong_hash", "$.assets", f"local {record['key']} bytes differ")
