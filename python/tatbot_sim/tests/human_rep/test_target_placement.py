from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
from tatbot_sim.human_rep.contracts import ContractError, canonical_digest, load_contract, validate_contract
from tatbot_sim.human_rep.ink_program import material_strokes
from tatbot_sim.human_rep.placement import upgrade_body_placement
from tatbot_sim.repo import repo_root

ROOT = repo_root()
EXAMPLES = ROOT / "config/human-representation/examples"


def test_body_migration_preserves_constraints_and_all_material_strokes():
    original = load_contract(EXAMPLES / "surface-placement.json")
    snapshot = deepcopy(original)
    migrated = upgrade_body_placement(original)
    assert migrated == load_contract(EXAMPLES / "body-placement-v2.json")
    assert original == snapshot
    artwork = load_contract(EXAMPLES / "tattoo-program.json")
    expected = material_strokes(artwork, original)
    for name in ("body-placement-v2", "plane-placement", "cylinder-placement"):
        placement = load_contract(EXAMPLES / f"{name}.json")
        strokes = material_strokes(artwork, placement)
        assert len(strokes) == len(expected)
        for actual, previous in zip(strokes, expected, strict=True):
            np.testing.assert_array_equal(actual.points_m, previous.points_m)
            assert (actual.ink_id, actual.width_m, actual.deposition, actual.source_primitive_sha256) == (
                previous.ink_id, previous.width_m, previous.deposition, previous.source_primitive_sha256)


@pytest.mark.parametrize("mutation", [
    {"anchor_uv_m": [0.1, 0]}, {"margin_m": -0.001},
    {"kind": "cylinder", "radius_m": 0.001}, {"radius_m": 0.04},
    {"kind": "body"}, {"robot_pose": [0, 0, 0]},
])
def test_invalid_geometry_is_refused_even_with_recomputed_digest(mutation):
    value = load_contract(EXAMPLES / "plane-placement.json")
    value["target"].update(mutation)
    value["content_sha256"] = canonical_digest(value)
    with pytest.raises(ContractError):
        validate_contract(value)


def test_rotation_and_body_domain_are_not_bypassed_by_new_version():
    value = load_contract(EXAMPLES / "plane-placement.json")
    value["target"]["canvas_m"] = [0.085, 0.055]
    value["rotation_rad"] = np.pi / 4
    value["content_sha256"] = canonical_digest(value)
    with pytest.raises(ContractError, match="placement_outside_domain"):
        validate_contract(value)
    value = load_contract(EXAMPLES / "body-placement-v2.json")
    value["target"]["supported_domain"]["face_indices"] = [42]
    value["content_sha256"] = canonical_digest(value)
    with pytest.raises(ContractError, match="anchor_outside_domain"):
        validate_contract(value)


def test_browser_frames_match_existing_numpy_charts():
    try:
        from surface_model import CylinderChart, PlaneChart
    finally:
        sys.path.pop(0)
    points = [[u, v] for u in [-0.02, 0, 0.02] for v in [-0.04, 0, 0.04]]
    source = """
      import { analyticFrame } from './src/core/surface-placement.ts';
      const points = JSON.parse(process.argv[1]);
      console.log(JSON.stringify(['plane','cylinder'].map(kind => points.map(uv =>
        analyticFrame({kind, radius_m:0.04, canvas_m:[0.15,0.1], anchor_uv_m:[0,0], margin_m:0},uv)))));
    """
    result = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", source,
                             json.dumps(points)], cwd=ROOT / "web/inkmap", check=True,
                            capture_output=True, text=True, timeout=30)
    rendered = json.loads(result.stdout)
    for chart, frames in zip([PlaneChart(np.zeros(3), np.eye(3)), CylinderChart(np.zeros(3), np.eye(3), 0.04)], rendered, strict=True):
        p, _, _, n = chart.frame(points)
        np.testing.assert_allclose([f["point"] for f in frames], p, atol=1e-12)
        np.testing.assert_allclose([f["normal"] for f in frames], n, atol=1e-12)
