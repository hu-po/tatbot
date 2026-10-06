"""Inkmap-to-simulation contracts and deterministic geometry compilation."""

from importlib import import_module

__all__ = [
    "canonical_json_bytes",
    "CanonicalSurface",
    "compile_scenario",
    "BodyRig",
    "document_sha256",
    "load_placement",
    "load_canonical_surface",
    "load_body_rig",
    "load_synthetic_identity_rig",
    "load_scenario",
    "MeshPatchSurface",
    "mesh_patch_from_scenario",
    "materialize_scenario_suite",
    "validate_placement",
    "validate_scenario",
    "PosedBody",
    "TriangleChart",
]

_EXPORT_MODULES = {
    "canonical_json_bytes": "contracts",
    "CanonicalSurface": "gltf_surface",
    "compile_scenario": "compiler",
    "BodyRig": "rig",
    "document_sha256": "contracts",
    "load_placement": "contracts",
    "load_canonical_surface": "gltf_surface",
    "load_body_rig": "rig",
    "load_synthetic_identity_rig": "rig",
    "load_scenario": "contracts",
    "MeshPatchSurface": "mesh_patch_surface",
    "mesh_patch_from_scenario": "mesh_patch_surface",
    "materialize_scenario_suite": "sampler",
    "validate_placement": "contracts",
    "validate_scenario": "contracts",
    "PosedBody": "rig",
    "TriangleChart": "triangle_chart",
}


def __getattr__(name: str):
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
