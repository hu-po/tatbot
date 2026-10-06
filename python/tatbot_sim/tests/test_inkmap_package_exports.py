"""The inkmap package keeps its public exports."""

from __future__ import annotations


def test_inkmap_package_preserves_named_and_star_exports():
    import tatbot_sim.inkmap as inkmap

    expected = {
        "canonical_json_bytes", "CanonicalSurface", "compile_scenario", "BodyRig",
        "document_sha256", "load_placement", "load_canonical_surface", "load_body_rig",
        "load_synthetic_identity_rig", "load_scenario", "MeshPatchSurface",
        "mesh_patch_from_scenario", "materialize_scenario_suite", "validate_placement",
        "validate_scenario", "PosedBody", "TriangleChart",
    }
    assert set(inkmap.__all__) == expected
    exported = {}
    exec("from tatbot_sim.inkmap import *", exported)
    for name in expected:
        assert exported[name] is getattr(inkmap, name)
        assert name in dir(inkmap)
