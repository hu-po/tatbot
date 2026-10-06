"""The public fixture must remain usable without claiming measured evidence."""
import json
from pathlib import Path

import pytest
from tatbot_sim import tools
from tatbot_sim.repo import repo_root


def test_public_profile_is_nominal_and_has_an_explicit_empty_palette(sim_profile):
    if sim_profile != "public":
        pytest.skip("public-profile invariant; checkout profile retains deployment configuration")
    from tatbot_cli import arms

    root = repo_root()
    source = Path(__file__).resolve().parents[3]
    for name in ("arms.json", "arm-labels.json"):
        assert (root / "config" / name).read_bytes() == (source / "config" / name).read_bytes()
    assert set(arms.load(root)) == {"left", "right"}
    assert (root / "config/nodes.json").read_bytes() == (root / "config/examples/nodes.json").read_bytes()
    assert not (root / "config/fiducial_benchmark_poses.json").exists()
    wrist = json.loads((root / "config/wrist_tags_measured.json").read_text())
    assert wrist["source"] == "synthetic"
    assert wrist["calibration_status"] == "pending_recalibration"
    assert wrist["tags"] == {}
    geometry = tools.resolved_geometry()
    assert not geometry.measured
    assert geometry.source == "datasheet-nominal"
    assert geometry.contact_status == "unqualified"
    palette = tools.palette()
    # the six caps of the public urdf/palette.urdf, L-M-S-S-M-L
    assert list(palette) == ["inkcap_large_1", "inkcap_medium_1", "inkcap_small_1",
                             "inkcap_small_2", "inkcap_medium_2", "inkcap_large_2"]
    # Other tests can select a wet process supply. Check the fixture's saved
    # empty rack independently of that legitimate simulator state.
    assert all(slot.dry for slot in tools.ink_registry().load_palette_load(root, palette).values())
    assert set(tools.ink_registry().load_inks(root)) == {"nighthawk_black", "bright_red", "true_blue"}
