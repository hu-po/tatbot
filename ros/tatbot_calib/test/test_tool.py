"""The tool gate: a calibration touches the station with the arm's fitted tool (config/workspace.yaml), never a
default; a stated tool that is not it, or a datasheet with no probe contact model, is refused before anything
moves, and `calib check` says so with exit 3."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from tatbot_calib import program
from tatbot_calib.tool import ToolRefusedError, fitted_tool
from tatbot_description import repo_root

REPO = repo_root(None)


def checkout(tmp_path: Path, tool_id: str | None, datasheets=("picosecond-laser-pen", "lutin-3rl-bugpin"),
             edit=None) -> Path:
    """A checkout's config: the named datasheets and a workspace fitting `tool_id` on the right arm."""
    tools = tmp_path / "config" / "tools"
    tools.mkdir(parents=True)
    for name in datasheets:
        text = (REPO / "config" / "tools" / f"{name}.yaml").read_text()
        (tools / f"{name}.yaml").write_text(edit(name, text) if edit else text)
    for mesh in (REPO / "config" / "tools").glob("*.stl"):
        shutil.copy(mesh, tools / mesh.name)
    line = f"  tool_id: {tool_id}\n" if tool_id else ""
    (tmp_path / "config" / "workspace.yaml").write_text(f"right:\n  tip_frame: right/tool_mount\n{line}")
    return tmp_path


def test_the_tool_is_the_fitted_one_with_its_datasheets_halo(tmp_path):
    tool, nose = fitted_tool(checkout(tmp_path, "picosecond-laser-pen"), "right")
    assert tool.tool_id == "picosecond-laser-pen"
    assert (nose.wall_radius_m, nose.hole_radius_m, nose.rim_radius_m, nose.side_height_m) == (0.0090, 0.0077, 0.00835,
                                                                                              0.006)
    assert nose.envelope[0] == (0.0, 0.0077) and nose.radius_at(0.0) == 0.0090   # never inside the measured ring
    assert nose.radius_at(0.064) == pytest.approx(0.0188, abs=1e-4)                # the pen body over the nose
    stated, _ = fitted_tool(checkout(tmp_path / "b", "picosecond-laser-pen"), "right", "picosecond-laser-pen")
    assert stated.tool_id == "picosecond-laser-pen"


def test_no_fitted_tool_and_a_stated_other_tool_are_refused(tmp_path):
    with pytest.raises(ToolRefusedError, match="names no tool_id"):
        fitted_tool(checkout(tmp_path, None), "right", "picosecond-laser-pen")   # a statement does not stand in
    with pytest.raises(ToolRefusedError, match="the stack plans every touch through that tool's tip"):
        fitted_tool(checkout(tmp_path / "b", "picosecond-laser-pen"), "right", "lutin-3rl-bugpin")


def test_the_ballpoint_side_touches_meet_its_metal_tip_under_the_body(tmp_path):
    _, nose = fitted_tool(checkout(tmp_path, "lutin-ballpoint-dot", datasheets=("lutin-ballpoint-dot",)), "right")
    # the probe's 1 mm-radius ball stays under the body's end, 2.4 mm up the tip
    assert nose.axial_tip and nose.side_height_m + 0.001 < 0.0024


def test_the_needle_cartridge_is_touched_on_its_tube_end(tmp_path):
    # the needles stay inside the tube with the machine off: the probe meets the tube's end, a ring around a hole
    # the ball cannot enter
    _, nose = fitted_tool(checkout(tmp_path, "lutin-3rl-bugpin"), "right")
    assert not nose.axial_tip and not nose.hole_takes_ball
    assert nose.hole_radius_m < nose.rim_radius_m < nose.wall_radius_m == nose.radius_at(0.0)


def drop_calibration(name, text):
    """The 3RL's datasheet without its contact model."""
    return text.split("\ncalibration:")[0] + "\n" if name == "lutin-3rl-bugpin" else text


def test_a_tool_whose_datasheet_declares_no_contact_model_is_refused(tmp_path):
    with pytest.raises(ToolRefusedError, match="face_kind is None, not halo.*no contact model"):
        fitted_tool(checkout(tmp_path, "lutin-3rl-bugpin", edit=drop_calibration), "right")

    def widen_the_hole(name, text):
        return text.replace("face_radius_m: 0.0077 ", "face_radius_m: 0.0095 ")

    with pytest.raises(ToolRefusedError, match="out of order"):
        fitted_tool(checkout(tmp_path / "b", "picosecond-laser-pen", edit=widen_the_hole), "right")


def test_calib_run_refuses_a_tool_without_a_contact_model_with_exit_3_before_anything(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(program, "fitted_tool",
                        lambda repo, arm, stated: fitted_tool(checkout(tmp_path, "lutin-3rl-bugpin",
                                                                       edit=drop_calibration), arm, stated))
    assert program.main(["run", "--arm", "right", "--station", str(tmp_path / "none.json"), "--run-started", "0"]) == 3
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] is False and "no contact model" in out["message"]
