"""`tatbot ros cartridge` and `calib apply`'s tip on the ROS owner: fitting writes the owner's record, never git's
workspace; a fresh cartridge's status names what it still needs measured; an adoption moves the fitted cartridge's tip
and makes it measured; a tool or tip the description cannot carry is refused before anything is written."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cartridge_fit
import pytest
import tool_spec
from tatbot_cli import ros_cartridge

REPO = Path(__file__).resolve().parents[2]
TOOL = tool_spec.parse_simple_yaml((REPO / "config" / "workspace.yaml").read_text())["xarm"]["tool_id"]   # the seed's
OTHER = next(name for name in ("lutin-5rl-mast", "lutin-3rl-bugpin") if name != TOOL)


@pytest.fixture
def owner(tmp_path, monkeypatch, capsys):
    """A ROS owner's layout (repo/ and calib/ side by side) and its module's JSON line per call."""
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    shutil.copy(REPO / "config" / "workspace.yaml", repo / "config" / "workspace.yaml")
    shutil.copytree(REPO / "config" / "tools", repo / "config" / "tools")
    monkeypatch.setenv("TATBOT_REPO", str(repo))
    monkeypatch.setattr(ros_cartridge, "CALIB_RUNS", tmp_path / "ros-calib")

    def call(*argv):
        assert ros_cartridge.main(list(argv)) == 0
        return json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    call.repo = repo
    return call


def gauge(repo: Path, role: str, tool: str, utc: str) -> None:
    name = f"wrist-gauge-xarm{'-' + role if role else ''}.json"
    (cartridge_fit.calib_dir(repo) / name).write_text(json.dumps({"measured_with": tool, "utc": utc}))


def test_fitting_writes_the_owners_record_and_names_what_a_fresh_cartridge_needs(owner):
    git = (owner.repo / "config" / "workspace.yaml").read_text()
    before = owner("status", "--arm", "xarm")
    assert before["tool_id"] == TOOL and before["record"] is None and before["tip"]["state"] == "carried"
    report = owner("fit", OTHER, "--arm", "xarm", "--instance", "box 2")
    assert (owner.repo / "config" / "workspace.yaml").read_text() == git
    assert report["tool_id"] == OTHER and report["instance"] == "box 2" and report["tip"]["state"] == "nominal"
    assert tool_spec.active_tool_id(owner.repo, "xarm") == OTHER
    assert any("calib sweep" in step for step in report["next"])
    gauge(owner.repo, "", OTHER, "2000-01-01T00:00:00Z")
    gauge(owner.repo, "wrist_1", TOOL, "2999-01-01T00:00:00Z")
    stale = owner("status", "--arm", "xarm")
    assert not stale["gauges"]["session"]["current"], "measured before this cartridge was fitted"
    assert not stale["gauges"]["wrist_1"]["current"], "measured with another cartridge"
    assert sum("calib wristcam" in step for step in stale["next"]) == 2
    assert any("--camera wrist_1" in step for step in stale["next"])


def test_an_adoption_moves_the_fitted_cartridges_tip_and_makes_it_measured(owner):
    owner("fit", TOOL, "--arm", "xarm", "--instance", "fresh")
    section = tool_spec.read_workspace(owner.repo)["xarm"]
    tip0 = [section[f"pen_tip_offset_{axis}"] for axis in "xyz"]
    run = owner.repo.parent / "ros-calib" / "20991231T000000Z-node-abcd"
    run.mkdir(parents=True)
    run.joinpath("candidate.json").write_text(json.dumps({
        "arm": "xarm", "tool": TOOL, "run_id": run.name, "utc": "2999-01-01T00:00:00Z", "contacts": 20,
        "tip0_m": tip0, "fit": {"tip_m": [tip0[0] + 0.001, tip0[1] - 0.0005, tip0[2] + 0.002], "rms_m": {"a": 1e-4},
                                "length_measured": True}}))
    report = owner("adopt", "--arm", "xarm", "--run", run.name, "--across-only")
    assert report["moved_mm"] == [1.0, -0.5, 0.0] and report["tip"]["state"] == "measured"
    assert tool_spec.read_workspace(owner.repo)["xarm"]["mechanical_contact_session"] == run.name
    other_arm = owner("adopt", "--arm", "right", "--run", run.name)
    assert not other_arm["ok"] and other_arm["code"] == 2


def test_adopting_the_adopted_run_again_writes_nothing_and_says_so(owner):
    """The second `calib apply` of a run finds its tip installed: the record is not rewritten, no copy is kept beside
    it, and the report says nothing changed (so the CLI leaves the running stack alone)."""
    owner("fit", TOOL, "--arm", "xarm")
    section = tool_spec.read_workspace(owner.repo)["xarm"]
    tip0 = [section[f"pen_tip_offset_{axis}"] for axis in "xyz"]
    run = owner.repo.parent / "ros-calib" / "20991231T000000Z-node-abcd"
    run.mkdir(parents=True)
    run.joinpath("candidate.json").write_text(json.dumps({
        "arm": "xarm", "tool": TOOL, "run_id": run.name, "utc": "2999-01-01T00:00:00Z", "contacts": 20,
        "tip0_m": tip0, "fit": {"tip_m": [tip0[0] + 0.001, tip0[1] - 0.0005, tip0[2] + 0.002], "rms_m": {"a": 1e-4},
                                "length_measured": True}}))
    first = owner("adopt", "--arm", "xarm", "--run", run.name)
    assert first["moved_mm"] == [1.0, -0.5, 2.0] and first["changed"]
    calib = cartridge_fit.calib_dir(owner.repo)
    files = {path.name: path.read_text() for path in calib.iterdir()}
    again = owner("adopt", "--arm", "xarm", "--run", run.name)
    assert again["moved_mm"] == [0.0, 0.0, 0.0] and not again["changed"] and again["adopted"] == run.name
    assert again["tip"] == first["tip"]
    assert {path.name: path.read_text() for path in calib.iterdir()} == files


def test_wrist_gauges_follow_an_adopted_tip_only_when_asked_and_say_why_when_they_cannot(owner):
    """`calib apply` reports how each wrist gauge would follow the tip and leaves the files; --derive-gauges moves
    them, keeping the measured one beside it. A gauge of another cartridge that records no tip is not derived."""
    owner("fit", TOOL, "--arm", "xarm")
    section = tool_spec.read_workspace(owner.repo)["xarm"]
    tip0 = [section[f"pen_tip_offset_{axis}"] for axis in "xyz"]
    calib = cartridge_fit.calib_dir(owner.repo)
    turned = [[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    session = {"tip_cam_m": [0.005, 0.0, 0.157], "cone_end_cam_m": [0.005, 0.0, 0.153], "normal_cam": [0.0, 0.0, -1.0],
               "measured_with": TOOL, "tool": TOOL, "utc": "2999-01-01T00:00:00Z", "run": "a-gauge-run",
               "mount_from_camera": turned, "tip_mount_m": tip0}
    (calib / "wrist-gauge-xarm.json").write_text(json.dumps(session))
    gauge(owner.repo, "wrist_1", OTHER, "2999-01-01T00:00:00Z")

    def sweep(name: str, start: list[float], moved: list[float]) -> str:
        run = owner.repo.parent / "ros-calib" / name
        run.mkdir(parents=True)
        run.joinpath("candidate.json").write_text(json.dumps({
            "arm": "xarm", "tool": TOOL, "run_id": name, "utc": "2999-01-02T00:00:00Z", "contacts": 20, "tip0_m": start,
            "fit": {"tip_m": [a + b for a, b in zip(start, moved, strict=True)], "rms_m": {"a": 1e-4},
                    "length_measured": True}}))
        return name

    told = owner("adopt", "--arm", "xarm", "--run", sweep("first", tip0, [0.001, 0.0, 0.002]))["gauges_followed"]
    # mount x is the camera's -y here; the 2 mm of length stands 2 mm farther from the camera, along -normal
    assert told["session"] == {"derived": False, "tip_moved_cam_mm": [0.0, -1.0, 2.0], "along_normal_mm": -2.0}
    assert not told["wrist_1"]["derived"] and OTHER in told["wrist_1"]["why"]
    assert json.loads((calib / "wrist-gauge-xarm.json").read_text()) == session, "reported, not written"

    tip1 = [tip0[0] + 0.001, tip0[1], tip0[2] + 0.002]
    report = owner("adopt", "--arm", "xarm", "--run", sweep("second", tip1, [0.0, 0.0005, 0.0]), "--derive-gauges")
    # the gauge still stands for the first tip (it was not derived then), so it follows both adoptions at once
    assert report["gauges_followed"]["session"] == {"derived": True, "tip_moved_cam_mm": [0.5, -1.0, 2.0],
                                                    "along_normal_mm": -2.0}
    derived = json.loads((calib / "wrist-gauge-xarm.json").read_text())
    assert derived["tip_cam_m"] == pytest.approx([0.0055, -0.001, 0.159]) and derived["derived"][-1]["run"] == "second"
    assert derived["tip_mount_m"] == pytest.approx([tip1[0], tip1[1] + 0.0005, tip1[2]])
    assert [json.loads(p.read_text()) for p in calib.glob("wrist-gauge-xarm.json.replaced-*")] == [session]
    assert report["gauges"]["session"]["derived"] == "second" and report["gauges"]["session"]["current"]
    assert report["gauges"]["wrist_1"]["derived"] is None


def test_a_tool_or_tip_the_description_cannot_carry_is_refused_before_anything_is_written(owner):
    unknown = owner("fit", "no-such-cartridge", "--arm", "xarm")
    assert not unknown["ok"] and unknown["code"] == 2 and OTHER in unknown["message"]
    owner("fit", TOOL, "--arm", "xarm")
    record = cartridge_fit.path(owner.repo, "xarm").read_text()
    section = tool_spec.read_workspace(owner.repo)["xarm"]
    run = owner.repo.parent / "ros-calib" / "far"
    run.mkdir(parents=True)
    tip0 = [section[f"pen_tip_offset_{axis}"] for axis in "xyz"]
    run.joinpath("candidate.json").write_text(json.dumps({
        "arm": "xarm", "tool": TOOL, "run_id": "far", "utc": "2999-01-01T00:00:00Z", "contacts": 20, "tip0_m": tip0,
        "fit": {"tip_m": [tip0[0], tip0[1], tip0[2] + 0.03], "rms_m": {"a": 1e-4}, "length_measured": True}}))
    far = owner("adopt", "--arm", "xarm", "--run", "far")
    assert not far["ok"] and far["code"] == 3 and f"from {TOOL}'s nominal tip" in far["message"]
    assert cartridge_fit.path(owner.repo, "xarm").read_text() == record
