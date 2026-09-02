"""Inkmap-to-simulation contracts and deterministic geometry compilation."""

from tatbot_sim.inkmap.compiler import compile_scenario
from tatbot_sim.inkmap.contracts import (
    canonical_json_bytes,
    document_sha256,
    load_placement,
    load_scenario,
    validate_placement,
    validate_scenario,
)
from tatbot_sim.inkmap.gltf_surface import CanonicalSurface, load_canonical_surface
from tatbot_sim.inkmap.mesh_patch_surface import MeshPatchSurface, mesh_patch_from_scenario
from tatbot_sim.inkmap.rig import BodyRig, PosedBody, load_body_rig
from tatbot_sim.inkmap.sampler import materialize_scenario_suite

__all__ = [
    "canonical_json_bytes",
    "CanonicalSurface",
    "compile_scenario",
    "BodyRig",
    "document_sha256",
    "load_placement",
    "load_canonical_surface",
    "load_body_rig",
    "load_scenario",
    "MeshPatchSurface",
    "mesh_patch_from_scenario",
    "materialize_scenario_suite",
    "validate_placement",
    "validate_scenario",
    "PosedBody",
]
