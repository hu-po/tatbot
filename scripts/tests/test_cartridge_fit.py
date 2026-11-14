"""The fitted cartridge the ROS owner keeps beside its deployed repo: laid over the workspace section there alone, a
type fitted again starting from its last tip, a type never measured from its datasheet's nominal, and no other
tool's tip showing through."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cartridge_fit
import pytest
import tool_spec

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = tool_spec.parse_simple_yaml((REPO / "config" / "workspace.yaml").read_text())
XARM = WORKSPACE["xarm"]
TOOL = XARM["tool_id"]
OTHER = next(name for name in ("lutin-5rl-mast", "lutin-3rl-bugpin") if name != TOOL)


def deployed(tmp_path: Path) -> Path:
    """A ROS owner's layout: repo/ with the workspace and datasheets, calib/ beside it."""
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    shutil.copy(REPO / "config" / "workspace.yaml", repo / "config" / "workspace.yaml")
    shutil.copytree(REPO / "config" / "tools", repo / "config" / "tools")
    return repo


def test_a_checkout_with_no_record_beside_it_reads_git_as_it_is(tmp_path):
    repo = deployed(tmp_path)
    assert not cartridge_fit.calib_dir(repo).exists()
    assert tool_spec.read_workspace(repo) == WORKSPACE
    assert cartridge_fit.calib_dir(REPO) != cartridge_fit.calib_dir(repo)


def test_the_owner_reads_its_fitted_cartridge_and_that_cartridges_tip(tmp_path):
    repo = deployed(tmp_path)
    record = cartridge_fit.fit(cartridge_fit.seeded(XARM, "xarm"), OTHER, (0.0, 0.0, 0.0755), instance="box 2",
                               utc="2026-11-10T18:00:00Z")
    assert cartridge_fit.write(repo, record) == tmp_path / "calib" / "cartridge-xarm.json"
    section = tool_spec.read_workspace(repo)["xarm"]
    assert tool_spec.active_tool_id(repo, "xarm") == OTHER
    assert section["pen_tip_offset_z"] == 0.0755 and section["tool_fit"] == {"status": "nominal"}
    for key in ("mechanical_contact_x", "mechanical_contact_session", "touchoff"):
        assert key not in section, f"{TOOL}'s {key} showed through under {OTHER}"
    assert section["payload"] == XARM["payload"] and section["tip_frame"] == XARM["tip_frame"]
    assert tool_spec.read_workspace(repo)["right"] == WORKSPACE["right"]
    assert cartridge_fit.tip_state(record) == "nominal"


def test_a_type_fitted_again_starts_from_its_last_tip_carried_until_measured():
    seed = cartridge_fit.seeded(XARM, "xarm")
    assert cartridge_fit.tip_state(seed) == "carried"
    away = cartridge_fit.fit(seed, OTHER, (0.0, 0.0, 0.0755), instance=None, utc="2026-11-10T18:00:00Z")
    back = cartridge_fit.fit(away, TOOL, (0.0, 0.0, 0.073), instance="fresh", utc="2026-11-10T19:00:00Z")
    section = cartridge_fit.fitted_section(XARM, back)
    assert section["pen_tip_offset_x"] == XARM["pen_tip_offset_x"] and section["touchoff"] == XARM["touchoff"]
    assert cartridge_fit.tip_state(back) == "carried"
    tip = {**back["tips"][TOOL]["section"], "pen_tip_offset_x": 0.001}
    adopted = cartridge_fit.adopt(back, tip, run_id="20261110T191500Z-node-abcd", utc="2026-11-10T19:15:00Z")
    assert cartridge_fit.tip_state(adopted) == "measured"
    assert cartridge_fit.fitted_section(XARM, adopted)["pen_tip_offset_x"] == 0.001
    assert adopted["tips"][OTHER] == away["tips"][OTHER]
    early = cartridge_fit.adopt(back, tip, run_id="r", utc="2026-11-10T18:59:00Z")
    assert cartridge_fit.tip_state(early) == "carried", "a run measured before this cartridge was fitted"
    unfitted = cartridge_fit.adopt(seed, seed["tips"][TOOL]["section"], run_id="r", utc="2026-11-10T19:46:46Z")
    assert cartridge_fit.tip_state(unfitted) == "measured", "a run adopted into a seeded record, no fit since"


def test_a_rewrite_keeps_the_record_it_replaces_and_a_broken_one_never_falls_back_to_git(tmp_path):
    repo = deployed(tmp_path)
    cartridge_fit.write(repo, cartridge_fit.seeded(XARM, "xarm"))
    cartridge_fit.write(repo, cartridge_fit.fit(cartridge_fit.load(repo, "xarm"), OTHER, (0, 0, 0.0755),
                                                instance=None, utc="2026-11-10T18:00:00Z"))
    assert len(list((tmp_path / "calib").glob("cartridge-xarm.json.replaced-*"))) == 1
    broken = json.loads(cartridge_fit.path(repo, "xarm").read_text())
    broken["tips"][OTHER]["section"]["pen_tip_offset_z"] = None
    cartridge_fit.path(repo, "xarm").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="not three finite numbers"):
        tool_spec.read_workspace(repo)
