"""The joint-6 sweep (tatbot_calib.sweep): the fit finds a tip, its length from the tilts, from stills simulated through
a pinhole at the WXAI follower's real joints and 2026-10-02's station, where the camera is not quite where the fix puts
it; the detectors find the tip and the ball's top in crops of the palette camera's own stills (the pen held 30 mm
over the ball, and no pen; the 3RL parked over it with a person moving behind the station, at night in a dim room,
and on the ball)."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from tatbot_calib import halo, reach, station, sweep
from tatbot_description import repo_root, robot_description
from tatbot_description.transforms import rpy_matrix
from tatbot_motion import Kinematics
from tatbot_session import ready
from test_program import datasheet_halo

DATA = Path(__file__).parent / "data"
PALETTE = rpy_matrix([0.17439, 0.36087, 0.03475], [0.0, 0.0, math.atan2(0.883072, -0.469238)])   # the fix
TIP0 = np.array([0.00002, -0.002674, 0.074495])   # the right pen tip in the tool mount when the stills were taken
TURNS = sweep.TURNS_DEG   # the WXAI's turns with that tip (turn_set)


@pytest.fixture(scope="module")
def kin():
    """The right arm with TIP0, the pen tip the stills were taken with (2026-10-02), whatever tool the checkout fits."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(robot_description(None, arms=("right",)))
    for joint in root.iter("joint"):
        if joint.get("name") == "right/tcp_joint":
            joint.find("origin").set("xyz", " ".join(f"{v:.9f}" for v in TIP0))
    return Kinematics(ET.tostring(root, encoding="unicode"), "right")


def test_the_turns_keep_a_pen_off_the_joints_axis_in_view_and_span_the_window_of_one_on_it(kin):
    """The WXAI's pen, 12-17 mm off joint 6's axis, turns +-30 deg; the xArm's, on joint 7's, +-60."""
    assert sweep.turn_set(kin, TIP0) == sweep.turn_set(kin, [0.000083, 0.002685, 0.062179]) == TURNS
    xarm = Kinematics(robot_description(None, arms=("xarm",)), "xarm")
    assert sweep.turn_set(xarm, [0.0, 0.0, 0.073]) == sweep.ON_AXIS_TURNS_DEG == tuple(range(-60, 61, 12))
    assert sweep.turn_set(xarm, [0.0005, 0.0, 0.073]) == sweep.ON_AXIS_TURNS_DEG   # a tip 0.5 mm off its axis


def _joints(kin, yaw_deg: float = 30.0, tilts: bool = False) -> np.ndarray:
    """The run's holds as joints: the park, its four shifts, the turns, and with `tilts` the tilts about the tip."""
    ball = (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3]
    park = halo.tool_pose(ball + [0.0, 0.0, halo.BALL_RADIUS_M + sweep.PARK_OVER_M], math.radians(yaw_deg))
    q = ready.solve_ik_seeded(kin, park, reach.rest(kin))
    rows = [q]
    for step in (PALETTE[:3, 1], -PALETTE[:3, 1], [0.0, 0.0, 1.0], -PALETTE[:3, 0]):
        pose = park.copy()
        pose[:3, 3] += sweep.SHIFT_M * np.asarray(step)
        rows.append(ready.solve_ik_seeded(kin, pose, q))
    rows += [sweep.turn_tool(kin.arm_model, q, deg) for deg in TURNS]
    if tilts:
        tilting = SimpleNamespace(fix=SimpleNamespace(base_from_palette=PALETTE))
        rows += [ready.solve_ik_seeded(kin, sweep.Sweep.tilted(tilting, park, deg), q) for deg in sweep.TILTS_DEG]
    return np.array(rows)


def _stills(kin, true_tip, seed=0, tilts=False):
    """(mounts, detected tips, the nominal camera, the true one): each hold's tip through a camera 1.1 deg and a few
    mm off the nominal one, its focal length 5 % longer and its centre 30 px off, with 0.7 px of detection noise."""
    nominal = sweep.nominal_camera(PALETTE, repo_root(None) / "urdf" / "palette.urdf")
    truth = nominal.moved([0.012, -0.008, 0.012, 0.002, -0.001, 0.004])
    truth = sweep.Camera(truth.R, truth.T, nominal.f * 1.05, nominal.c + [30.0, -20.0])
    mounts = np.array([kin.frame(q, "right/tool_mount") for q in _joints(kin, tilts=tilts)])
    uv = truth.project(mounts[:, :3, :3] @ true_tip + mounts[:, :3, 3])
    return mounts, uv + np.random.default_rng(seed).normal(0.0, 0.7, uv.shape), nominal, truth


def test_the_fit_finds_the_tip_across_its_axis_to_a_tenth_of_a_millimetre(kin):
    true_tip = TIP0 + [0.0004, -0.0007, 0.0]
    mounts, uv, nominal, _ = _stills(kin, true_tip)
    tip, camera, miss, sigma = sweep.fit(mounts, uv, TIP0, nominal)
    assert np.linalg.norm(tip[:2] - true_tip[:2]) < 1e-4
    assert tip[2] == TIP0[2]
    assert np.sqrt(np.mean(miss ** 2)) < 1.5 and np.all(sigma < 1e-4)


def test_the_tilts_measure_the_tips_length(kin):
    """A tip 12.6 mm shorter than installed and 0.7 mm aside (the 3RL as the wrist camera saw it, 2026-10-22): the
    turns alone fit it more than 10 mm off along the report's factor; with the tilts about the installed tip the fit
    finds all of it, its length too, through the camera's own error and 0.7 px of noise."""
    true_tip = TIP0 + [0.0004, -0.0007, -0.0126]
    mounts, uv, nominal, _ = _stills(kin, true_tip)
    held, *_ = sweep.fit(mounts, uv, TIP0, nominal)
    assert np.linalg.norm(held[:2] - true_tip[:2]) > 0.010
    mounts, uv, nominal, _ = _stills(kin, true_tip, tilts=True)
    tip, _, miss, sigma = sweep.fit(mounts, uv, TIP0, nominal, length=True)
    assert np.linalg.norm(tip - true_tip) < 2e-4, (tip - true_tip) * 1000
    assert len(sigma) == 3 and sigma[2] < 2e-4 and np.sqrt(np.mean(miss ** 2)) < 1.5


def test_a_runs_stills_make_a_candidate_calib_apply_adopts(kin, tmp_path):
    """The run's records (the park, four shifts, the turns) through Sweep.write: the tip, each turn left out, and the
    tip's height over the ball's top from the park's still, into a candidate the adoption takes."""
    import sys

    true_tip = TIP0 + [0.0004, -0.0007, 0.0]
    mounts, uv, _, truth = _stills(kin, true_tip)
    joints = _joints(kin)
    fix = station.StationFix("right", PALETTE, (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3], "2026-10-02T20:00:00Z",
                             "overhead_tag")
    cal = SimpleNamespace(kin=kin, repo=repo_root(None), tip0=TIP0, arm="right", run=SimpleNamespace(dir=tmp_path),
                          halo=datasheet_halo("lutin-3rl-bugpin"))
    run = sweep.Sweep(cal, fix, "lutin-3rl-bugpin", "station.json")
    kinds = ["park"] + ["shift"] * 4 + ["turn"] * len(TURNS)
    run.holds = [{"label": f"{kind} {i}", "kind": kind, "q": q.tolist(), "tip_px": p.tolist(), "tip_by": "focus",
                  "shift_px": [0.0, 0.0]} for i, (kind, q, p) in enumerate(zip(kinds, joints, uv, strict=True))]
    run.q_park = joints[0]
    run.top = truth.project(mounts[0, :3, :3] @ true_tip + mounts[0, :3, 3] - [0.0, 0.0, 0.0065])[0]
    assert run.write() == 0
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert np.linalg.norm(np.subtract(candidate["fit"]["tip_m"][:2], true_tip[:2])) < 1e-4
    assert candidate["fit"]["gap_m"] == pytest.approx(0.0065, abs=1e-4)
    assert set(candidate["held_out_rms_m"]) == {f"turn {i}" for i in range(5, 5 + len(TURNS))}
    sys.path.insert(0, str(repo_root(None) / "scripts" / "lib"))
    import probe_calibration_adopt
    import tool_spec

    section = {**tool_spec.read_workspace(repo_root(None))["right"], "tool_id": "lutin-3rl-bugpin"}  # whatever is fitted
    tip, why = probe_calibration_adopt.apply(candidate, section)
    assert why == [] and tip["touchoff"]["method"] == sweep.METHOD
    assert "length error of 1 mm reads here as" in (tmp_path / "report.md").read_text()
    assert tip["pen_tip_offset_z"] == section["pen_tip_offset_z"]


def test_a_runs_tilts_make_a_candidate_whose_length_calib_apply_adopts(kin, tmp_path):
    """With the tilts in, the candidate carries the measured length, the report says so, and the adoption sets the
    arm's pen_tip_offset_z to it."""
    import sys

    true_tip = TIP0 + [0.0004, -0.0007, -0.0126]
    _, uv, _, _ = _stills(kin, true_tip, tilts=True)
    joints = _joints(kin, tilts=True)
    fix = station.StationFix("right", PALETTE, (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3], "2026-10-02T20:00:00Z",
                             "overhead_tag")
    cal = SimpleNamespace(kin=kin, repo=repo_root(None), tip0=TIP0, arm="right", run=SimpleNamespace(dir=tmp_path),
                          halo=datasheet_halo("lutin-3rl-bugpin"))
    run = sweep.Sweep(cal, fix, "lutin-3rl-bugpin", "station.json")
    kinds = ["park"] + ["shift"] * 4 + ["turn"] * len(TURNS) + ["tilt"] * len(sweep.TILTS_DEG)
    run.holds = [{"label": f"{kind} {i}", "kind": kind, "q": q.tolist(), "tip_px": p.tolist(), "tip_by": "focus",
                  "shift_px": [0.0, 0.0]} for i, (kind, q, p) in enumerate(zip(kinds, joints, uv, strict=True))]
    run.q_park = joints[0]
    assert run.write() == 0
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert candidate["fit"]["length_measured"] and len(candidate["held_out_rms_m"]) == len(kinds) - 5
    assert np.linalg.norm(np.subtract(candidate["fit"]["tip_m"], true_tip)) < 2e-4
    assert "measured by the tilts" in (tmp_path / "report.md").read_text()
    sys.path.insert(0, str(repo_root(None) / "scripts" / "lib"))
    import probe_calibration_adopt
    import tool_spec

    section = {**tool_spec.read_workspace(repo_root(None))["right"], "tool_id": "lutin-3rl-bugpin"}
    tip, why = probe_calibration_adopt.apply(candidate, section)
    assert why == [] and tip["pen_tip_offset_z"] == pytest.approx(candidate["fit"]["tip_m"][2], abs=1e-6)


class FakeRig:
    """The arm at the joints the IK gives each goal's pose from where it stands; every goal recorded."""

    def __init__(self, kin):
        self.kin, self.q, self.goals = kin, reach.rest(kin), []
        self.safety = SimpleNamespace(estop_ok=True, latched=False, probe_triggered=False)

    def spin(self, seconds):
        pass

    def joints(self):
        return self.q.copy()

    def goal(self, start, *, move=False, **_):
        self.goals.append(start.copy())
        self.q = ready.solve_ik_seeded(self.kin, start, self.q)
        return {"tripped": False, "joints": self.q.tolist(), "run_dir": "", "message": "", "contact": []}


def _sweep(kin, tmp_path, monkeypatch, true_tip, top_dz=0.0, tip_by=None, ball_off_m=0.0):
    """A whole run on FakeRig, the stills simulated: the true tip, and the ball's top `top_dz` over the fix's and
    `ball_off_m` from it along the camera's line of sight, through the camera _stills uses. The tip is found `tip_by`,
    else in focus within DEPTH_OF_FIELD_M of the ball's plane along that line and by its change past it; the way checks
    the planner makes are the calibration's own tests'."""
    from tatbot_calib import program
    from tatbot_motion import load_motion
    from test_program import datasheet_halo

    *_, truth = _stills(kin, true_tip)
    fix = station.StationFix("right", PALETTE, (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3], "2026-10-02T20:00:00Z",
                             "overhead_tag")
    cal = program.Calibration(FakeRig(kin), SimpleNamespace(dir=tmp_path), repo_root(None), "right",
                              datasheet_halo("lutin-3rl-bugpin"), 0.0, kin)
    cal.motion, cal.bodies, cal.posts, cal.guard, cal.world_from_base, cal.tip0 = load_motion(), [], [], None, None, TIP0
    program.place_station(cal, fix)
    monkeypatch.setattr(cal, "way_check", lambda label, poses, q=None: None)
    run = sweep.Sweep(cal, fix, "lutin-3rl-bugpin", ask=lambda path: None)
    line = PALETTE[:3, 0]
    top = fix.ball + [0.0, 0.0, halo.BALL_RADIUS_M + top_dz] + ball_off_m * line

    def look(self, path, q):
        if not self.holds:
            self.top = truth.project(top)[0]
        tip = self.tip(q, true_tip)
        how = tip_by or ("focus" if abs((tip - top) @ line) < DEPTH_OF_FIELD_M else "change")
        return {"tip_px": truth.project(tip)[0].tolist(), "tip_by": how, "shift_px": [0.0, 0.0]}

    monkeypatch.setattr(sweep.Sweep, "look", look)
    return run


def _true_gap(run, true_tip, top_dz=0.0) -> float:
    """How far the true tip stood over the ball's true top at the park."""
    top = (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[2] + halo.BALL_RADIUS_M + top_dz
    return float(run.tip(sweep.parked(run.holds)["q"], true_tip)[2] - top)


SIGHTINGS = 4   # goals before the park: over the sighting, the reference still over it and back, the sighting
DEPTH_OF_FIELD_M = 0.0032   # the tube stood sharp 3 mm off the ball's plane (the "away" shift) and blurred 3.5 mm
                            # off it (2026-11-09)


def test_a_run_parks_where_the_camera_sees_best_shifts_turns_joint_6_alone_and_tilts(kin, tmp_path, monkeypatch):
    """Noise-free, the camera's own error (its focal length 5 % off the nominal, its centre 30 px) costs 0.01 mm; the
    sightings bring the tip down to PARK_OVER_M over the ball's top through that error, and the tilts hold the
    installed tip where the park put it and measure the length."""
    true_tip = TIP0 + [0.0004, -0.0007, -0.0030]
    run = _sweep(kin, tmp_path, monkeypatch, true_tip)
    assert run.run() == 0
    goals = run.cal.rig.goals
    assert len(goals) == SIGHTINGS + 1 + 4 + len(TURNS) + len(sweep.TILTS_DEG) and run.yaw in (15.0, 30.0, 45.0)
    top = (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3] + [0.0, 0.0, halo.BALL_RADIUS_M]
    assert np.allclose(goals[0][:3, 3], top + [0.0, 0.0, sweep.SIGHT_OVER_M + sweep.SIGHT_STEP_M])
    park = goals[SIGHTINGS]
    assert np.allclose(park, run.park_pose(run.yaw))
    assert _true_gap(run, true_tip) == pytest.approx(sweep.PARK_OVER_M, abs=1e-4)
    tilted = goals[-len(sweep.TILTS_DEG):]
    assert np.allclose([pose[:3, 3] for pose in tilted], park[:3, 3])
    assert np.allclose([math.degrees(math.acos((np.trace(pose[:3, :3] @ park[:3, :3].T) - 1.0) / 2.0))
                        for pose in tilted], np.abs(sweep.TILTS_DEG))
    parked = np.array([row["q"] for row in run.holds if row["kind"] == "turn"])
    assert np.max(np.abs(parked[:, :5] - run.q_park[:5])) < 0.005
    assert np.allclose(np.degrees(parked[:, 5] - run.q_park[5]), TURNS, atol=0.3)
    fitted = json.loads((tmp_path / "candidate.json").read_text())["fit"]
    assert fitted["length_measured"] and np.linalg.norm(np.subtract(fitted["tip_m"], true_tip)) < 2e-5
    assert fitted["gap_m"] == pytest.approx(_true_gap(run, true_tip), abs=2e-5)


def test_a_tool_reseated_lower_parks_over_the_ball_not_on_it_and_the_tilts_measure_it(kin, tmp_path, monkeypatch):
    """The 3RL re-seated in R7's cradle stood ~7 mm lower than installed, and a park by the installed tool put it 1 mm
    under the ball's top (2026-11-08): the sightings bring it down to PARK_OVER_M over it, and the tilts find the
    whole 7 mm."""
    true_tip = TIP0 + [0.0004, -0.0007, 0.0070]
    run = _sweep(kin, tmp_path, monkeypatch, true_tip, top_dz=0.0015)
    assert run.run() == 0
    assert _true_gap(run, true_tip, 0.0015) == pytest.approx(sweep.PARK_OVER_M, abs=1e-4)
    fitted = json.loads((tmp_path / "candidate.json").read_text())["fit"]
    assert np.linalg.norm(np.subtract(fitted["tip_m"], true_tip)) < 1e-4


def test_a_sighting_that_leaves_too_little_over_the_ball_comes_no_lower(kin, tmp_path, monkeypatch):
    """A tool 16 mm longer than installed stands 4 mm over the ball's top at the sighting: under PARK_OVER_M, so the
    arm comes no lower and the run stops."""
    run = _sweep(kin, tmp_path, monkeypatch, TIP0 + [0.0, 0.0, 0.016])
    with pytest.raises(RuntimeError, match="it comes down only"):
        run.run()
    assert len(run.cal.rig.goals) == SIGHTINGS and [row["kind"] for row in run.holds] == ["sight", "sight"]


def test_a_tip_the_sightings_do_not_find_in_focus_brings_the_arm_no_lower(kin, tmp_path, monkeypatch):
    """A tip found by its change can be the cartridge's grip 13 mm up the pen, at both sightings alike, so the step
    passes (2026-11-09, the probe tripped): the arm steps along the camera's line of sight for focus, finds none, and
    comes no lower than the sighting."""
    run = _sweep(kin, tmp_path, monkeypatch, TIP0, tip_by="change")
    with pytest.raises(RuntimeError, match="into focus"):
        run.run()
    assert len(run.cal.rig.goals) == SIGHTINGS + len(sweep.FOCUS_STEPS_M)
    assert min(pose[2, 3] for pose in run.cal.rig.goals) == pytest.approx(run.cal.rig.goals[SIGHTINGS - 1][2, 3])


def test_a_sighting_off_the_balls_plane_steps_into_focus_and_parks_on_the_ball(kin, tmp_path, monkeypatch):
    """The station placed 5 mm off the ball along the camera's line of sight (3.5 mm by the overhead, 2026-11-09):
    the lower sighting's tube is blurred, the arm steps along that line until it is sharp, the station's place moves
    with it, and the park stands in focus PARK_OVER_M over the ball's top."""
    true_tip = TIP0 + [0.0004, -0.0007, -0.0030]
    run = _sweep(kin, tmp_path, monkeypatch, true_tip, ball_off_m=0.005)
    assert run.run() == 0
    assert sweep.parked(run.holds)["tip_by"] == "focus"
    assert any(row["label"].startswith("focus") for row in run.holds)
    top = (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3] + [0.0, 0.0, halo.BALL_RADIUS_M] + 0.005 * PALETTE[:3, 0]
    tip = run.tip(sweep.parked(run.holds)["q"], true_tip)
    assert abs((tip - top) @ PALETTE[:3, 0]) < DEPTH_OF_FIELD_M
    assert tip[2] - top[2] == pytest.approx(sweep.PARK_OVER_M, abs=1e-4)


def test_an_upper_sighting_misread_stops_the_run_at_the_park(kin, tmp_path, monkeypatch):
    """The upper sighting's tip taken 1.5 mm up the pen, by its change: the step passes at 1.25 of its scale and the
    arm comes 25 % too far, still over the ball, where the park's still sees the drop at another scale and stops."""
    run = _sweep(kin, tmp_path, monkeypatch, TIP0)
    look = sweep.Sweep.look

    def misread(self, path, q):
        row = look(self, path, q)
        if not self.holds:
            row["tip_px"], row["tip_by"] = (np.array(row["tip_px"]) - [0.0, 0.0015 * 60000.0]).tolist(), "change"
        return row

    monkeypatch.setattr(sweep.Sweep, "look", misread)
    with pytest.raises(RuntimeError, match="a sighting misread"):
        run.run()
    assert len(run.cal.rig.goals) == SIGHTINGS + 1 and _true_gap(run, TIP0) > 0.0


def test_the_fit_leaves_out_a_tip_found_only_by_its_change(kin, tmp_path, monkeypatch):
    """A turn whose change against the lifted still ended 3.3 mm under the tube's end (2026-11-09): the fit takes
    the stills found in focus, which hold the other turns and every tilt, and finds the tip as if it were not there."""
    true_tip = TIP0 + [0.0004, -0.0007, -0.0030]
    run = _sweep(kin, tmp_path, monkeypatch, true_tip)
    look = sweep.Sweep.look

    def shadow(self, path, q):
        row = look(self, path, q)
        if path.stem.endswith("turn-18"):
            row["tip_px"], row["tip_by"] = (np.array(row["tip_px"]) + [0.0, 0.0033 * 60000.0]).tolist(), "change"
        return row

    monkeypatch.setattr(sweep.Sweep, "look", shadow)
    assert run.run() == 0
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert candidate["left_out"] == 1 and "turn -18" not in candidate["held_out_rms_m"]
    assert np.linalg.norm(np.subtract(candidate["fit"]["tip_m"], true_tip)) < 2e-5


def test_no_tilt_is_sent_when_the_parks_still_leaves_too_little_room_for_one(kin, tmp_path, monkeypatch):
    """A park 4.5 mm over the ball's top: the turns, which take the tip down 1.4 mm at most, go; a tilt might take a
    tip 15 mm off the installed one 3.1 mm down, so none is sent."""
    monkeypatch.setattr(sweep, "PARK_OVER_M", 0.0045)
    run = _sweep(kin, tmp_path, monkeypatch, TIP0)
    with pytest.raises(RuntimeError, match="no tilt sent"):
        run.run()
    assert len(run.cal.rig.goals) == SIGHTINGS + 5 + len(TURNS) and not any(row["kind"] == "tilt" for row in run.holds)


def test_no_turn_is_sent_when_the_parks_still_shows_the_ball_too_near(kin, tmp_path, monkeypatch):
    """The turns take the tip 1.4 mm down from a park 6 mm over the ball's top; under a clearance of 5 mm the run
    stops after the shifts (a park that comes down nearer than planned is refused so)."""
    monkeypatch.setattr(sweep, "TURN_CLEAR_M", 0.005)
    run = _sweep(kin, tmp_path, monkeypatch, TIP0)
    with pytest.raises(RuntimeError, match="no turn sent"):
        run.run()
    assert len(run.cal.rig.goals) == SIGHTINGS + 5 and not any(row["kind"] == "turn" for row in run.holds)


@pytest.mark.parametrize("fix_up_m, code", [(0.0, 0), (0.012, 3)])
def test_a_height_the_overhead_fix_refutes_is_refused_and_the_touch_kept(kin, tmp_path, monkeypatch, fix_up_m, code):
    """The ball's top 1 mm under the fix's reads so at the scale the sightings measured (the camera's focal length is
    5 % off the nominal) and is written. Under a fix 12 mm higher no reader of the touch would take it (a misread tip
    set it 12.1 mm low on 2026-10-22, and only the gauge after it refused): refused, the touch as it was."""
    from tatbot_motion import station as model

    run = _sweep(kin, tmp_path, monkeypatch, TIP0, top_dz=-0.001)
    path = tmp_path / "station-touch-right.json"
    monkeypatch.setattr(model, "touch_path", lambda arm: path)
    model.StationTouch("right", "lutin-3rl-bugpin", np.asarray(run.fix.ball, float), np.eye(3), TIP0, [0.0] * 7,
                       run.fix, "touch", "2026-10-22T20:00:00Z").write()
    before = path.read_text()
    run.overhead = dataclasses.replace(run.fix, ball=np.asarray(run.fix.ball, float) + [0.0, 0.0, fix_up_m])
    assert run.height() == code
    touch = model.load_touch("right")
    if code:
        assert path.read_text() == before
    else:
        assert touch.ball[2] - (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[2] == pytest.approx(-0.001, abs=1e-4)
        assert touch.height_source.startswith("the palette camera") and "in focus" in touch.height_source


@pytest.mark.parametrize("camera_height", [False, True])
def test_a_touch_no_camera_measured_places_the_sweeps_ball_across(kin, tmp_path, monkeypatch, camera_height):
    """S1's needle touches settle the ball's top a fixed depth under the overhead's (4.6 mm, 2026-11-09), past the 4
    mm a dip takes, so the touch is refused and the overhead's ball, 3.6 mm along the camera's line of sight, blurred
    the tube. The sweep measures the height itself: it takes the touch's place across at the overhead's height."""
    from tatbot_calib import program
    from tatbot_motion import station as model

    run = _sweep(kin, tmp_path, monkeypatch, TIP0)
    path = tmp_path / "station-touch-right.json"
    monkeypatch.setattr(model, "touch_path", lambda arm: path)
    ball = np.asarray(run.fix.ball, float) + [-0.0036, -0.0002, -0.0046]
    model.StationTouch("right", "lutin-3rl-bugpin", ball, np.eye(3), TIP0, station.joint_offsets(run.cal.repo, "right"),
                       run.fix, "touch", "2026-11-09T17:47:00Z").write()
    prior = program.touched_fix(run.cal, run.fix, "lutin-3rl-bugpin", camera_height=camera_height)
    if camera_height:
        assert np.allclose(prior.ball, [*ball[:2], run.fix.ball[2]], atol=1e-7)
        assert prior.base_from_palette[2, 3] - run.fix.base_from_palette[2, 3] == pytest.approx(0.0, abs=1e-7)
    else:
        assert prior is run.fix


def test_a_length_error_reads_across_the_axis_by_the_reported_factor(kin):
    """Joint 6 cannot see the tip along its own axis: a tip 0.5 mm longer than installed fits 0.5 mm along the
    report's factor, about (0, 1) for the WXAI mount's 45 degrees."""
    mounts, uv, nominal, _ = _stills(kin, TIP0 + [0.0, 0.0, 0.0005])
    tip, *_ = sweep.fit(mounts, uv, TIP0, nominal)
    q = _joints(kin)[0]
    a, b = kin.frame(q, "right/tool_mount"), kin.frame(sweep.turn_tool(kin.arm_model, q, 1.0), "right/tool_mount")
    n = a[:3, :3].T @ cv2.Rodrigues(b[:3, :3] @ a[:3, :3].T)[0].ravel()
    leak = -n[:2] / n[2]
    assert np.linalg.norm(tip[:2] - TIP0[:2] - 0.0005 * leak) < 1e-4
    assert abs(leak[1]) == pytest.approx(1.0, abs=0.05)


def test_the_turns_keep_joints_1_to_5_where_the_park_put_them(kin):
    """Each turn is a pose the IK reaches from joints 1 mrad off the park's by turning joint 6 alone, to its
    tolerance (10 um at the tip, which leaves a few mrad along the wrist's weakest direction)."""
    q = _joints(kin)[0]
    for deg in TURNS:
        turned = sweep.turn_tool(kin.arm_model, q, deg)
        solved = ready.solve_ik_seeded(kin, kin.fk(turned), q + 0.001)
        assert np.max(np.abs(solved[:6] - turned[:6])) < 0.005, deg


def _still(name):
    path = next(DATA.glob(f"palette-{name}-*.jpg"))
    x, y = (int(v) for v in path.stem.split("-")[-2:])
    return cv2.imread(str(path)), np.array([x, y], float)


@pytest.mark.parametrize("start", [(0, 0), (40, 0), (-60, 30), (0, -60), (60, 60)])
def test_the_tip_is_found_from_a_prediction_a_millimetre_off(start):
    """The pen held 30 mm over the ball (2026-10-02): the metal tip ends at (4669, 1606) whichever way the
    prediction is off and however it is tilted (by hand: 4667 +- 3, 1604 +- 3)."""
    image, origin = _still("pen-tip")
    for down in ((0.0, 1.0), (0.035, 1.0)):
        tip = sweep.find_tip(image, np.array([4626.0, 1617.0]) + start - origin, down, 60000.0)
        assert np.linalg.norm(tip + origin - [4669.0, 1605.8]) < 1.0


def test_the_tubes_end_is_found_in_focus_not_its_needle_nor_whoever_moves_behind():
    """The 3RL parked 8 mm over the ball (2026-10-22): a person moved behind the station between the lifted still
    and this one, and the pen's change against it ran out of its window. In focus, the pen ends at its tube's end
    (by hand the crust's foot at 4498 +- 8, 2958 +- 4), not at the needle standing 1 mm out of it."""
    image, origin = _still("3rl-park")
    for start in ((0, 0), (60, -40), (-60, 60)):
        near = np.array([4624.1, 3038.2]) + start - origin
        tip = sweep.find_tip_sharp(image, near, (0.0, 1.0), 62000.0, 0.0009)
        assert np.linalg.norm(tip + origin - [4498.0, 2958.0]) < 25.0, tip + origin
        needle = sweep.find_tip_sharp(image, near, (0.0, 1.0), 62000.0, 0.00015)
        assert needle[1] - tip[1] > 40.0


@pytest.mark.parametrize("name, near, needle, sharp_under", [
    ("dim", (4628.0, 3044.0), (4600.0, 3000.0), 2400.0),       # mean grey 45: the region in focus ends at the grip
    ("darker", (4628.0, 3040.0), (4505.0, 3050.0), 2700.0),    # 27: it ends mid-tube, where the change once did too
])
def test_a_dim_still_takes_the_pens_end_from_its_change_where_the_dark_tube_never_stands_sharp(name, near, needle,
                                                                                                 sharp_under):
    """The 3RL parked over the ball on the night of 2026-10-22: the dark tube against the dim room never stands
    sharp, and the region in focus ends 6-12 mm up; its change against the lifted still, grown through the weaker
    change, and the metal's contrast end at the needle (by hand, +- 8 px), so pick_end takes the change."""
    image, origin = _still(f"{name}-park")
    lifted, _ = _still(f"{name}-lifted")
    near, down, px = np.array(near) - origin, np.array([0.0, 1.0]), 62000.0
    sharp = sweep.find_tip_sharp(image, near, down, px, 0.0009)
    moved = sweep.find_tip_moved(image, lifted, near, down, px)
    contrast = sweep.find_park_tip(image, near, down, px)
    assert sharp[1] + origin[1] < sharp_under
    end, how = sweep.pick_end(sharp, moved, contrast, down, px)
    assert how == "change" and np.linalg.norm(end + origin - needle) < 15.0, end + origin


def test_a_prediction_far_up_the_pen_steps_down_to_the_tubes_end_in_focus():
    """The xArm rig's sighting, 2026-11-09: the overhead's 8.9 mm put the prediction on the cartridge's grip, 13 mm
    over the tube's end, where the window's top met nothing in focus; stepped down the pen the finder meets the tube
    and takes its end (by hand: 4526 +- 3, 2593 +- 3)."""
    image, origin = _still("farup-sight")
    canvas = np.zeros((sweep.SIZE[1], sweep.SIZE[0], 3), np.uint8)
    x, y = (int(v) for v in origin)
    canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
    near = np.array([4530.0, 1800.0])
    assert sweep.find_tip_sharp(canvas, near, (0.0, 1.0), 60000.0, 0.0012) is None
    stepped, found = sweep.find_tip_sharp_down(canvas, near, (0.0, 1.0), 60000.0, 0.0012)
    assert np.linalg.norm(found - [4526.3, 2592.0]) < 3.0 and stepped[1] > near[1]


@pytest.mark.parametrize("moved, contrast, how", [
    ((100.0, 1060.0), None, "focus"),                # 1 mm under: its needle under the tube's end
    ((100.0, 1800.0), (102.0, 1795.0), "change"),    # the region in focus broke 13 mm up; the contrast agrees
    ((100.0, 1800.0), None, "change"),               # nothing to decide: the lower
    ((100.0, 300.0), None, "focus"),                 # the change broke at the grip: the lower
    ((100.0, 1800.0), (100.0, 1010.0), "focus"),     # the contrast stands with the one in focus, the change ran on
])
def test_the_pens_end_is_the_finders_agreed_one_else_the_lower(moved, contrast, how):
    sharp = np.array([100.0, 1000.0])
    end, chosen = sweep.pick_end(sharp, np.array(moved), None if contrast is None else np.array(contrast),
                                 (0.0, 1.0), 60000.0)
    assert chosen == how and np.allclose(end, sharp if how == "focus" else moved)


@pytest.mark.parametrize("name, on_ball", [("onball", True), ("dim", False)])
def test_a_sighting_whose_ball_top_moved_from_the_lifted_still_stops_the_run(kin, name, on_ball):
    """Under a station prior 9 mm off (2026-10-22) the park put the 3RL on the ball, and its top read 2.2 mm
    from the lifted still's; at night with the pen clear of it the two agree to 0.02 mm."""
    canvas = {}
    for which in ("park", "lifted"):
        image, (x, y) = _still(f"{name}-{which}")
        canvas[which] = np.zeros((3900, 5300, 3), np.uint8)
        canvas[which][int(y):int(y) + image.shape[0], int(x):int(x) + image.shape[1]] = image
    fix = station.StationFix("right", PALETTE, (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3], "2026-10-22T20:00:00Z",
                             "overhead_tag")
    cal = SimpleNamespace(kin=kin, repo=repo_root(None), tip0=TIP0, arm="right", say=print,
                          halo=datasheet_halo("lutin-3rl-bugpin"))
    run = sweep.Sweep(cal, fix, "lutin-3rl-bugpin")
    run.empty = canvas["lifted"]
    if on_ball:
        with pytest.raises(RuntimeError, match="the pen is on the ball or before it"):
            run.first(canvas["park"], np.zeros(2), 62000.0)
    else:
        run.first(canvas["park"], np.zeros(2), 62000.0)
        assert abs(run.top[1] - 3339.0) < 3.0


def test_no_tip_is_found_without_the_pen():
    image, origin = _still("nopen-tip")
    assert sweep.find_tip(image, np.array([4626.0, 1617.0]) - origin, (0.0, 1.0), 60000.0) is None


@pytest.mark.parametrize("name", ["pen-ball", "nopen-ball"])
def test_the_balls_top_is_found_under_the_screens_glow(name):
    image, origin = _still(name)
    for start in ((0, 0), (40, 30), (-50, -40)):
        top = sweep.find_ball_top(image, np.array([4624.0, 3472.0]) + start - origin, (0.0, 1.0), 62000.0)
        assert abs(top[1] + origin[1] - 3349.0) < 3.0 and abs(top[0] + origin[0] - 4656.0) < 10.0


def test_the_first_still_is_searched_where_the_nominal_camera_puts_the_tip(kin, tmp_path):
    """A full still rebuilt from the crops at their places, with the arm where it held the pen for them: FK and the
    fix's camera predict the tip within a millimetre, so the run finds it there, the ball's top under it, and no
    drift."""
    canvas = np.zeros((sweep.SIZE[1], sweep.SIZE[0], 3), np.uint8)
    for name in ("pen-tip", "pen-ball"):
        image, (x, y) = _still(name)
        canvas[int(y):int(y) + image.shape[0], int(x):int(x) + image.shape[1]] = image
    path = tmp_path / "00-park.jpg"
    cv2.imwrite(str(path), canvas)
    q = ready.solve_ik_seeded(kin, rpy_matrix([0.1744, 0.3608, 0.1203], [math.pi, 0.0, -1.0472]), reach.rest(kin))
    fix = station.StationFix("right", PALETTE, (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3], "2026-10-02T20:00:00Z",
                             "overhead_tag")
    cal = SimpleNamespace(kin=kin, repo=repo_root(None), tip0=TIP0, arm="right", say=print,
                          halo=datasheet_halo("lutin-3rl-bugpin"))
    run = sweep.Sweep(cal, fix, "lutin-3rl-bugpin")
    row = run.look(path, q)
    assert np.linalg.norm(np.subtract(row["tip_px"], [4669.0, 1605.8])) < 1.0
    assert np.linalg.norm(run.offset) < 62.0 and abs(run.top[1] - 3349.0) < 3.0
    assert np.allclose(row["shift_px"], 0.0, atol=0.05)


def test_the_balls_patch_measures_the_cameras_drift():
    image, origin = _still("pen-ball")
    mark = sweep.BallMark(image, np.array([4656.0, 3349.0]) - origin, 62000.0)
    moved = cv2.warpAffine(image, np.array([[1.0, 0.0, 1.3], [0.0, 1.0, -0.7]]), image.shape[1::-1])
    assert np.allclose(mark.shift(image), 0.0, atol=0.05)
    assert np.allclose(mark.shift(moved), [1.3, -0.7], atol=0.15)
