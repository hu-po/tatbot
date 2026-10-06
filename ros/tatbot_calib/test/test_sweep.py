"""The joint-6 sweep (tatbot_calib.sweep): the fit finds a tip from stills simulated through a pinhole at the pink arm's
real joints and 2026-10-02's station, where the camera is not quite where the fix puts it; the detectors find the tip
and the ball's top in crops of the palette camera's own stills (the pen held 30 mm over the ball, and no pen)."""
from __future__ import annotations

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

DATA = Path(__file__).parent / "data"
PALETTE = rpy_matrix([0.17439, 0.36087, 0.03475], [0.0, 0.0, math.atan2(0.883072, -0.469238)])   # the fix
TIP0 = np.array([0.00002, -0.002674, 0.074495])   # config/workspace.yaml's right pen tip in the tool mount


@pytest.fixture(scope="module")
def kin():
    """The right arm with TIP0, the ballpoint's tip the stills were taken with, whatever tool the checkout fits."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(robot_description(None, arms=("right",)))
    for joint in root.iter("joint"):
        if joint.get("name") == "right/tcp_joint":
            joint.find("origin").set("xyz", " ".join(f"{v:.9f}" for v in TIP0))
    return Kinematics(ET.tostring(root, encoding="unicode"), "right")


def _joints(kin, yaw_deg: float = 30.0) -> np.ndarray:
    """The run's holds as joints: the park, its four shifts, the turns."""
    ball = (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3]
    park = halo.tool_pose(ball + [0.0, 0.0, halo.BALL_RADIUS_M + sweep.PARK_OVER_M], math.radians(yaw_deg))
    q = ready.solve_ik_seeded(kin, park, reach.REST)
    rows = [q]
    for step in (PALETTE[:3, 1], -PALETTE[:3, 1], [0.0, 0.0, 1.0], -PALETTE[:3, 0]):
        pose = park.copy()
        pose[:3, 3] += sweep.SHIFT_M * np.asarray(step)
        rows.append(ready.solve_ik_seeded(kin, pose, q))
    return np.array(rows + [sweep.Sweep.turned(q, deg) for deg in sweep.TURNS_DEG])


def _stills(kin, true_tip, seed=0):
    """(mounts, detected tips, the nominal camera, the true one): each hold's tip through a camera 1.1 deg and a few
    mm off the nominal one, its focal length 5 % longer and its centre 30 px off, with 0.7 px of detection noise."""
    nominal = sweep.nominal_camera(PALETTE, repo_root(None) / "urdf" / "palette.urdf")
    truth = nominal.moved([0.012, -0.008, 0.012, 0.002, -0.001, 0.004])
    truth = sweep.Camera(truth.R, truth.T, nominal.f * 1.05, nominal.c + [30.0, -20.0])
    mounts = np.array([kin.frame(q, "right/tool_mount") for q in _joints(kin)])
    uv = truth.project(mounts[:, :3, :3] @ true_tip + mounts[:, :3, 3])
    return mounts, uv + np.random.default_rng(seed).normal(0.0, 0.7, uv.shape), nominal, truth


def test_the_fit_finds_the_tip_across_its_axis_to_a_tenth_of_a_millimetre(kin):
    true_tip = TIP0 + [0.0004, -0.0007, 0.0]
    mounts, uv, nominal, _ = _stills(kin, true_tip)
    tip, camera, miss, sigma = sweep.fit(mounts, uv, TIP0, nominal)
    assert np.linalg.norm(tip[:2] - true_tip[:2]) < 1e-4
    assert tip[2] == TIP0[2]
    assert np.sqrt(np.mean(miss ** 2)) < 1.5 and np.all(sigma < 1e-4)


def test_a_runs_stills_make_a_candidate_calib_apply_adopts(kin, tmp_path):
    """The run's records (the park, four shifts, the turns) through Sweep.write: the tip, each turn left out, and the
    tip's height over the ball's top from the park's still, into a candidate the adoption takes."""
    import sys

    true_tip = TIP0 + [0.0004, -0.0007, 0.0]
    mounts, uv, _, truth = _stills(kin, true_tip)
    joints = _joints(kin)
    fix = station.StationFix("right", PALETTE, (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3], "2026-10-02T20:00:00Z",
                             "overhead_tag")
    cal = SimpleNamespace(kin=kin, repo=repo_root(None), tip0=TIP0, arm="right", run=SimpleNamespace(dir=tmp_path))
    run = sweep.Sweep(cal, fix, "lutin-ballpoint-dot", "station.json")
    kinds = ["park"] + ["shift"] * 4 + ["turn"] * len(sweep.TURNS_DEG)
    run.holds = [{"label": f"{kind} {i}", "kind": kind, "q": q.tolist(), "tip_px": p.tolist(), "shift_px": [0.0, 0.0]}
                 for i, (kind, q, p) in enumerate(zip(kinds, joints, uv, strict=True))]
    run.q_park = joints[0]
    run.top = truth.project(mounts[0, :3, :3] @ true_tip + mounts[0, :3, 3] - [0.0, 0.0, 0.0065])[0]
    assert run.write() == 0
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert np.linalg.norm(np.subtract(candidate["fit"]["tip_m"][:2], true_tip[:2])) < 1e-4
    assert candidate["fit"]["gap_m"] == pytest.approx(0.0065, abs=1e-4)
    assert set(candidate["held_out_rms_m"]) == {f"turn {i}" for i in range(5, 5 + len(sweep.TURNS_DEG))}
    sys.path.insert(0, str(repo_root(None) / "scripts" / "lib"))
    import probe_calibration_adopt

    workspace = (repo_root(None) / "config" / "workspace.yaml").read_text()
    workspace = workspace.replace("  tool_id: lutin-3rl-bugpin\n", "  tool_id: lutin-ballpoint-dot\n")   # whatever is fitted now
    edited, why = probe_calibration_adopt.apply(candidate, workspace)
    assert why == [] and sweep.METHOD in edited
    assert "length error of 1 mm reads here as" in (tmp_path / "report.md").read_text()


class FakeRig:
    """The arm at the joints the IK gives each goal's pose from where it stands; every goal recorded."""

    def __init__(self, kin):
        self.kin, self.q, self.goals = kin, reach.REST.copy(), []
        self.safety = SimpleNamespace(estop_ok=True, latched=False, probe_triggered=False)

    def spin(self, seconds):
        pass

    def joints(self):
        return self.q.copy()

    def goal(self, start, *, move=False, **_):
        self.goals.append(start.copy())
        self.q = ready.solve_ik_seeded(self.kin, start, self.q)
        return {"tripped": False, "joints": self.q.tolist(), "run_dir": "", "message": "", "contact": []}


def _sweep(kin, tmp_path, monkeypatch, true_tip, under_m):
    """A whole run on FakeRig, the stills simulated: the true tip and the ball's top `under_m` under the park's tip,
    through the camera _stills uses; the way checks the planner makes are the calibration's own tests'."""
    from tatbot_calib import program
    from tatbot_motion import load_motion
    from test_program import datasheet_halo

    *_, truth = _stills(kin, true_tip)
    fix = station.StationFix("right", PALETTE, (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3], "2026-10-02T20:00:00Z",
                             "overhead_tag")
    cal = program.Calibration(FakeRig(kin), SimpleNamespace(dir=tmp_path), repo_root(None), "right",
                              datasheet_halo("lutin-ballpoint-dot"), 0.0, kin)
    cal.motion, cal.bodies, cal.posts, cal.guard, cal.world_from_base, cal.tip0 = load_motion(), [], [], None, None, TIP0
    program.place_station(cal, fix)
    monkeypatch.setattr(cal, "way_check", lambda label, poses, q=None: None)
    run = sweep.Sweep(cal, fix, "lutin-ballpoint-dot", ask=lambda path: None)

    def look(self, path, q):
        if not self.holds:
            self.top = truth.project(self.tip(q, true_tip) - [0.0, 0.0, under_m])[0]
        return {"tip_px": truth.project(self.tip(q, true_tip))[0].tolist(), "shift_px": [0.0, 0.0]}

    monkeypatch.setattr(sweep.Sweep, "look", look)
    return run


def test_a_run_parks_where_the_camera_sees_best_shifts_and_turns_joint_6_alone(kin, tmp_path, monkeypatch):
    """Noise-free, the camera's own error (its focal length 5 % off the nominal, its centre 30 px) costs 0.01 mm."""
    true_tip = TIP0 + [0.0004, -0.0007, 0.0]
    run = _sweep(kin, tmp_path, monkeypatch, true_tip, 0.006)
    assert run.run() == 0
    goals = run.cal.rig.goals
    assert len(goals) == 1 + 4 + len(sweep.TURNS_DEG) and run.yaw in (15.0, 30.0, 45.0)
    assert np.allclose(goals[0], run.park_pose(run.yaw))
    parked = np.array([row["q"] for row in run.holds if row["kind"] == "turn"])
    assert np.max(np.abs(parked[:, :5] - run.q_park[:5])) < 0.005
    assert np.allclose(np.degrees(parked[:, 5] - run.q_park[5]), sweep.TURNS_DEG, atol=0.3)
    fitted = json.loads((tmp_path / "candidate.json").read_text())["fit"]
    assert np.linalg.norm(np.subtract(fitted["tip_m"][:2], true_tip[:2])) < 2e-5
    assert fitted["gap_m"] == pytest.approx(0.006, abs=2e-5)


def test_no_turn_is_sent_when_the_parks_still_shows_the_ball_too_near(kin, tmp_path, monkeypatch):
    """The fix put the ball 6 mm under the tip; the still shows it 2.5 mm under, and the turns take the tip down
    more than 0.5 mm: the run stops after the shifts."""
    run = _sweep(kin, tmp_path, monkeypatch, TIP0, 0.0025)
    with pytest.raises(RuntimeError, match="no turn sent"):
        run.run()
    assert len(run.cal.rig.goals) == 5 and not any(row["kind"] == "turn" for row in run.holds)


def test_a_length_error_reads_across_the_axis_by_the_reported_factor(kin):
    """Joint 6 cannot see the tip along its own axis: a tip 0.5 mm longer than installed fits 0.5 mm along the
    report's factor, about (0, 1) for the pink mount's 45 degrees."""
    mounts, uv, nominal, _ = _stills(kin, TIP0 + [0.0, 0.0, 0.0005])
    tip, *_ = sweep.fit(mounts, uv, TIP0, nominal)
    q = _joints(kin)[0]
    a, b = kin.frame(q, "right/tool_mount"), kin.frame(sweep.Sweep.turned(q, 1.0), "right/tool_mount")
    n = a[:3, :3].T @ cv2.Rodrigues(b[:3, :3] @ a[:3, :3].T)[0].ravel()
    leak = -n[:2] / n[2]
    assert np.linalg.norm(tip[:2] - TIP0[:2] - 0.0005 * leak) < 1e-4
    assert abs(leak[1]) == pytest.approx(1.0, abs=0.05)


def test_the_turns_keep_joints_1_to_5_where_the_park_put_them(kin):
    """Each turn is a pose the IK reaches from joints 1 mrad off the park's by turning joint 6 alone, to its
    tolerance (10 um at the tip, which leaves a few mrad along the wrist's weakest direction)."""
    q = _joints(kin)[0]
    for deg in sweep.TURNS_DEG:
        turned = sweep.Sweep.turned(q, deg)
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


def test_no_tip_is_found_without_the_pen():
    image, origin = _still("nopen-tip")
    assert sweep.find_tip(image, np.array([4626.0, 1617.0]) - origin, (0.0, 1.0), 60000.0) is None


@pytest.mark.parametrize("name", ["pen-ball", "nopen-ball"])
def test_the_balls_top_is_found_under_the_screens_glow(name):
    image, origin = _still(name)
    for start in ((0, 0), (40, 30), (-50, -40)):
        top = sweep.find_ball_top(image, np.array([4624.0, 3472.0]) + start - origin, (0.0, 1.0), 62000.0)
        assert abs(top[1] + origin[1] - 3349.0) < 3.0 and abs(top[0] + origin[0] - 4656.0) < 10.0


def test_the_park_still_is_searched_where_the_nominal_camera_puts_the_tip(kin, tmp_path):
    """A full still rebuilt from the crops at their places, with the arm where it held the pen for them: FK and the
    fix's camera predict the tip within a millimetre, so the run finds it there, the ball's top under it, and no
    drift."""
    canvas = np.zeros((sweep.SIZE[1], sweep.SIZE[0], 3), np.uint8)
    for name in ("pen-tip", "pen-ball"):
        image, (x, y) = _still(name)
        canvas[int(y):int(y) + image.shape[0], int(x):int(x) + image.shape[1]] = image
    path = tmp_path / "00-park.jpg"
    cv2.imwrite(str(path), canvas)
    q = ready.solve_ik_seeded(kin, rpy_matrix([0.1744, 0.3608, 0.1203], [math.pi, 0.0, -1.0472]), reach.REST)
    fix = station.StationFix("right", PALETTE, (PALETTE @ [0.0, 0.0, 0.0556, 1.0])[:3], "2026-10-02T20:00:00Z",
                             "overhead_tag")
    cal = SimpleNamespace(kin=kin, repo=repo_root(None), tip0=TIP0, arm="right", say=print)
    run = sweep.Sweep(cal, fix, "lutin-ballpoint-dot")
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
