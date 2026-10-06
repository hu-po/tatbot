"""Read portable editor projects without invoking rendering, assets, or services."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from tatbot_contracts.artwork import require_acquired_artwork

from tatbot_sim.human_rep.contracts import ContractError, canonical_bytes, canonical_digest, parse_json
from tatbot_sim.inkmap.artwork import validate_artwork_record
from tatbot_sim.inkmap.contracts import validate_placement
from tatbot_sim.repo import repo_root

MAX_PROJECT_BYTES = 20_000_000


def _validate_chart(chart: dict | None) -> None:
    if not chart:
        return
    if len({item["id"] for item in chart["items"]}) != len(chart["items"]):
        raise ContractError("project_invalid", "$.chart.items", "duplicate chart placement ID")
    for item in chart["items"]:
        if item["artwork_id"] not in chart["artwork"]:
            raise ContractError("project_missing_artwork", "$.chart.artwork", item["artwork_id"])
    if chart["selected_id"] is not None and chart["selected_id"] not in {item["id"] for item in chart["items"]}:
        raise ContractError("project_invalid", "$.chart.selected_id", "selection names no placement")
    for held in chart["artwork"].values():
        validate_artwork_record(held)
        require_acquired_artwork(held)


def _validate_body_history(value: dict) -> None:
    file = value["placement_file"]
    selected = value["selected_id"]
    if (value["edit_before"] is None) != (selected is None) or (
        selected is not None and selected not in {p["id"] for p in file["placements"]}
    ):
        raise ContractError("project_invalid", "$.selected_id", "inconsistent pending edit")
    frames = [file["placements"], *value["history"]["past"], *value["history"]["future"]]
    if value["edit_before"] is not None:
        frames.append(value["edit_before"])
    for placements in frames:
        validate_placement({**file, "placements": placements})
        if len(placements) > 100 or len({p["id"] for p in placements}) != len(placements):
            raise ContractError("project_invalid", "$.history", "duplicate IDs or more than 100 placements")
        for placement in placements:
            if placement["design_id"] not in file.get("designs", {}):
                raise ContractError("project_missing_artwork", "$.placement_file.designs", placement["design_id"])


def validate_project(value: Any) -> dict[str, Any]:
    encoded = canonical_bytes(value)
    if len(encoded) > MAX_PROJECT_BYTES:
        raise ContractError("project_over_budget", "$", "maximum 20 MB")
    root = repo_root() / "config/inkmap"
    placement_schema = json.loads((root / "placement.schema.json").read_text())
    project_schema = json.loads((root / "project.schema.json").read_text())
    registry = Registry().with_resource(placement_schema["$id"], Resource.from_contents(placement_schema))
    for name in ("inkmap/artwork", "human-representation/tattoo-program", "human-representation/common"):
        schema = json.loads((repo_root() / f"config/{name}.schema.json").read_text())
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    validator = Draft202012Validator(project_schema, registry=registry)
    error = next(validator.iter_errors(value), None)
    if error:
        raise ContractError("project_invalid", error.json_path, error.message)
    if canonical_digest(value) != value["content_sha256"]:
        raise ContractError("project_wrong_hash", "$.content_sha256", "content digest differs")
    for key in ("chart", "chart_parked"):
        _validate_chart(value.get(key))
    if value["chart"] and value["chart_parked"] and value["chart"]["kind"] == value["chart_parked"]["kind"]:
        raise ContractError("project_invalid", "$.chart_parked", "parked draft is for the surface already showing")
    poses = json.loads((root / "body-poses.json").read_text())["pose_ids"]
    if value["editor"]["pose_id"] not in poses:
        raise ContractError("project_invalid", "$.editor.pose_id", "unknown pose")
    camera = value["editor"]["camera"]
    if camera and camera["position"] == camera["target"]:
        raise ContractError("project_invalid", "$.editor.camera", "position and target coincide")
    _validate_body_history(value)
    for record in value["placement_file"].get("designs", {}).values():
        require_acquired_artwork(record)
    return deepcopy(value)


def load_project(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        data = stream.read(MAX_PROJECT_BYTES + 1)
    if len(data) > MAX_PROJECT_BYTES:
        raise ContractError("project_over_budget", "$", "maximum 20 MB")
    return validate_project(parse_json(data))
