"""Version-3 scenario compilation from the immutable, typed editor bundle.

The v2 boundary-trace format is not reinterpreted. V3 retains the source bundle
and InkProgram and checks every derived field. Renderer qualification is a
separate consumer gate; these records never authorize physical execution.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

import numpy as np

from tatbot_sim import tools
from tatbot_sim.human_rep.contracts import ContractError, canonical_bytes, canonical_digest, validate_contract
from tatbot_sim.human_rep.ink_program import compile_ink_program
from tatbot_sim.inkmap.artwork import artwork_preview
from tatbot_sim.inkmap.bundle import _schema_check, validate_simulation_bundle, verify_local_assets
from tatbot_sim.inkmap.compiler import _default_world_from_body, _sha256_file, compile_scenario
from tatbot_sim.inkmap.contracts import document_sha256, validate_scenario
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.repo import repo_root


def _trace(ink_program):
    strokes = [[{"face": point["face_index"], "barycentric": point["barycentric"]}
                for point in event["curve"]["coordinates"]]
               for event in ink_program["events"] if event["kind"] == "stroke"]
    return {"compiler": "tatbot_sim.surface_trace", "compiler_version": 3,
            "sha256": hashlib.sha256(canonical_bytes(strokes)).hexdigest(), "strokes": strokes}


def _selected(bundle, placement_id):
    placements = bundle["placement_file"]["placements"]
    if placement_id is None:
        if len(placements) != 1:
            raise ContractError("sim_bundle_requires_choice", "$.placements", "drawing compilation requires --placement-id for multiple placements")
        placement_id = placements[0]["id"]
    for index, placement in enumerate(placements):
        if placement["id"] == placement_id:
            return placement, bundle["surface_placements"][index]["placement"]
    raise ContractError("sim_bundle_invalid", "$.placement_id", "unknown placement ID")


def _cartridge_terms(tool, operating_budget_s, initial_load_fraction):
    """Retain old scenario terms without requiring a cartridge timer."""
    if tools.ink_registry().policy_for(tool).mode != "cartridge":
        return {}
    return {**({"operating_budget_s": float(operating_budget_s)} if operating_budget_s is not None else {}),
            "initial_load_fraction": 1.0 if initial_load_fraction is None else float(initial_load_fraction)}


def compile_simulation_bundle(value: dict, *, placement_id: str | None = None,
                              created_at: str | None = None, git_sha: str | None = None,
                              generator: str = "tatbot sim compile",
                              operating_budget_s: float | None = None,
                              initial_load_fraction: float | None = None) -> dict:
    bundle = validate_simulation_bundle(value)
    verify_local_assets(bundle)
    placement, surface = _selected(bundle, placement_id)
    request = bundle["request"]
    program = bundle["artworks"][placement["design_id"]]["program"]
    tool = tools.registry().load_tool(request["tool_id"], repo_root())
    ink = compile_ink_program(program, surface, tool=tool, provenance={
        "producer": "tatbot-inkmap-sim-bundle", "version": "1", "created_utc": "2026-09-05T00:00:00Z",
        "source_sha256": bundle["content_sha256"],
    }, **_cartridge_terms(tool, operating_budget_s, initial_load_fraction))
    # Reuse the existing pose/support/world realization, but supply the typed
    # trace directly. The SVG boundary compiler is never called on this path.
    scenario = compile_scenario(bundle["placement_file"], placement_id=placement["id"],
        pose_id=request["pose_id"], seed=request["seed"], tool_id=request["tool_id"], support_id=request["support_id"],
        target_world_m=request["target_world_m"], align_patch_up=request["align_patch_up"],
        patch_yaw_rad=request["patch_yaw_rad"], created_at=created_at, git_sha=git_sha, generator=generator,
        program_binding={"bundle": bundle, "placement_id": placement["id"], "ink_program": ink,
                         "tool_profile_sha256": _sha256_file(tool.source)})
    return scenario


def validate_program_scenario(value: Any) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != 3:
        raise ContractError("wrong_schema", "$", "expected tattoo scenario version 3")
    _schema_check(value, "inkmap/tattoo-scenario-v3")
    binding = value.get("program_binding")
    if not isinstance(binding, dict) or set(binding) != {"bundle", "placement_id", "ink_program", "tool_profile_sha256"}:
        raise ContractError("sim_bundle_invalid", "$.program_binding", "missing or unknown binding fields")
    bundle = validate_simulation_bundle(binding["bundle"])
    verify_local_assets(bundle)
    placement, surface = _selected(bundle, binding["placement_id"])
    ink = validate_contract(binding["ink_program"], expected_schema="tatbot.ink-program/1")
    program = bundle["artworks"][placement["design_id"]]["program"]
    if ink["tattoo_program_sha256"] != program["content_sha256"] or ink["surface_placement_sha256"] != surface["content_sha256"]:
        raise ContractError("wrong_hash", "$.program_binding", "ink program is bound to different inputs")
    base = {key: deepcopy(item) for key, item in value.items() if key != "program_binding"}
    base["schema_version"] = 2
    # Reuse the v2 shape/body validator without weakening v3's portable trace
    # digest version. The v3 JSON Schema above already requires trace v3.
    validator_base = deepcopy(base)
    validator_base["trace"]["compiler_version"] = 2
    validate_scenario(validator_base)
    expected_placement = {**placement, "source_sha256": document_sha256(bundle["placement_file"])}
    design = bundle["artworks"][placement["design_id"]]
    request = bundle["request"]
    rig = load_body_rig()
    expected_world = _default_world_from_body(rig, request["pose_id"], placement,
        request["target_world_m"], request["align_patch_up"], request["patch_yaw_rad"])
    expected_support = np.eye(4)
    expected_support[:3, 3] = request.get("support_offset_m", [0, 0, 0])
    if (base["placement"] != expected_placement or base["design"]["svg"] != artwork_preview(design)
            or base["design"]["name"] != design["name"]
            or base["design"]["source"] != bundle["placement_file"]["designs"][placement["design_id"]].get("source", {"kind": "embedded"})
            or base["design"]["sha256"] != design["source_sha256"] or base["trace"] != _trace(ink)
            or base["seed"] != request["seed"] or base["pose"]["id"] != request["pose_id"]
            or base["pose"]["catalog_sha256"] != request["pose_catalog_sha256"]
            or base["support"]["id"] != request["support_id"] or base["robot"]["tool_id"] != request["tool_id"]
            or base["pose"]["posed_surface_sha256"] != rig.catalog_record["poses"][request["pose_id"]]["surface_sha256"]
            or not np.allclose(base["pose"]["world_from_body"], expected_world, atol=1e-12, rtol=0)
            or base["robot"]["world_from_robot"] != np.eye(4).tolist()
            or base["robot"]["urdf_sha256"] != _sha256_file(repo_root() / "urdf/tatbot.urdf")
            or base["support"].get("world_from_nominal") != expected_support.tolist()
            or "placement_optimization" in base):
        raise ContractError("wrong_hash", "$.program_binding", "scenario differs from bound bundle/program")
    # A rehashed forged InkProgram is not evidence that it was compiled from
    # this artwork. Rebuild using the exact local compiler/tool policy.
    tool = tools.registry().load_tool(request["tool_id"], repo_root())
    if binding["tool_profile_sha256"] != _sha256_file(tool.source):
        raise ContractError("wrong_hash", "$.program_binding.tool_profile_sha256", "local tool profile differs")
    # Rebuild from the bound program's own optional legacy terms.
    expected = compile_ink_program(program, surface, tool=tool, rig=rig, provenance={
        "producer": "tatbot-inkmap-sim-bundle", "version": "1", "created_utc": "2026-09-05T00:00:00Z",
        "source_sha256": canonical_digest(bundle),
    }, **_cartridge_terms(tool, ink.get("operating_budget_s"), ink["initial_ink_state"]["load_fraction"]))
    if canonical_bytes(expected) != canonical_bytes(ink):
        raise ContractError("wrong_hash", "$.program_binding.ink_program", "compiled derivation differs")
    return value
