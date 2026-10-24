"""A whole palette dip through tatbot_session.cap on fake arm I/O: the real right-arm kinematics and planner, the
palette where the overhead camera placed it on 2026-10-05, the station touch 7 mm from that fix, a cap declared
with ink. The arm "moves" to each goal's last knot; every tick is checked. No ROS graph, no driver."""
from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from tatbot_description.transforms import rpy_matrix
from tatbot_motion import Kinematics, load_motion, station
from tatbot_session.ledger import Ledger
from test_ports import StepClock

REPO = Path(__file__).resolve().parents[3]
# a station fix of 2026-10-05: the palette in the right arm's base, and the ball the overhead fix puts there
FIX_POSE = rpy_matrix([0.17121257, 0.35867467, 0.02895254], [0.0, 0.0, np.arctan2(0.88182395, -0.47157875)])
TOUCH_OFFSET = np.array([-0.0025, 0.0067, -0.0026])   # where S1 found the ball from such a fix (2026-10-03)
# the touch's heading there, -60 deg (the tool upright): the arm's dips at that station dwelt at it
TOUCH_ROTATION = rpy_matrix([0.0, 0.0, 0.0], [np.pi, 0.0, np.radians(-60.0)])[:3, :3]
PAGE = rpy_matrix([0.2878, 0.0014, 0.00695], [0.0, 0.0, -1.536])


class Scene:
    def base_pose(self, arm):
        return np.eye(4)

    def parts_clearance(self, arm, rows, parts, pads=None, skip=(), max_step_rad=None):
        return SimpleNamespace(distance_m=1.0, body_a="tool", body_b="")


class IO:
    """The arm follows each goal knot by knot to its end; a script of outcomes can stop one early. `settle`, per goal
    in order: (joint 1's offset, readings) the arm stands off the goal's end for after it, None for none."""

    def __init__(self, q):
        self.q, self.latched, self.goals, self.script = np.asarray(q, float), False, [], []
        self.settle, self.ringing = [], (0.0, 0)

    def measured(self):
        offset, reads = self.ringing
        self.ringing = (offset, reads - 1)
        return self.q + (np.eye(len(self.q))[0] * offset if reads > 0 else 0.0)

    def execute(self, traj, *, on_tick=None, should_stop=None, **_):
        self.goals.append(traj)
        status, stop = self.script.pop(0) if self.script else ("ok", None)
        stop = len(traj.t) - 1 if stop is None else stop(traj) if callable(stop) else stop
        for i in range(stop + 1):
            self.q = traj.q[i].copy()   # the arm where the goal has it at this knot
            if on_tick:
                on_tick(i)
            if should_stop and should_stop():
                status, stop = "cancelled", i
                break
        self.q = traj.q[stop].copy()
        settle = self.settle.pop(0) if self.settle else None
        self.ringing = settle or (0.0, 0)
        return status, stop, ""

    def unlatch(self):
        self.latched = False
        return True, "unlatched"


class Executor:
    """The parts of ArmExecutor a dip uses."""

    def __init__(self, repo, run_dir, kin, q, *, machine=True):
        self.repo, self.run_dir, self.arm, self.kin, self.motion = repo, run_dir, "right", kin, load_motion()
        self.io, self.clock = IO(q), StepClock()
        self.node = SimpleNamespace(guard=SimpleNamespace(_lock=threading.Lock(), scene=Scene(), step_rad=0.01))
        self.stack = {"registration": {"right": str(repo / "registration.json")}}
        self.machine = object() if machine else None
        self.events, self.machine_log, self.decisions, self.snag_log = [], [], [], []

    def used(self):
        return PAGE

    def knots(self, traj):
        from tatbot_motion import to_knots

        return to_knots(traj, 100.0)

    def event(self, kind, text="", op_id=""):
        self.events.append(text)

    def feedback(self, **_):
        pass

    def stop_preplan(self):
        pass

    def snag_guard(self, on):
        self.snag_log.append((on, len(self.io.goals)))

    def stopping(self):
        return False

    def cancelled(self):
        return False

    def require_off(self, activity):
        self.machine_log.append("off")

    def machine_off(self):
        self.machine_log.append("off")

    def machine_running(self, op_id, index, ledger):
        self.machine_log.append("on")

    def wait_decision(self, kind, op_id, index, ledger):
        from tatbot_session import rules

        self.decisions.append(kind)
        return rules.CONTINUE

    def land(self):
        raise AssertionError("not landing")


def _reach(sheet: Path, value: str) -> None:
    sheet.write_text(re.sub(r"(?m)^needle_reach_mm: .*$", f"needle_reach_mm: {value}", sheet.read_text()))


@pytest.fixture(scope="module")
def kin():
    return Kinematics.from_repo(arm="right")


@pytest.fixture
def rig(tmp_path, kin, monkeypatch):
    """A checkout with the needle cartridge fitted, cap L2 declared 70% full, the station touched."""
    from tatbot_session import cap, lease

    repo = tmp_path / "repo"
    shutil.copytree(REPO / "config", repo / "config")
    shutil.copytree(REPO / "urdf", repo / "urdf")
    (repo / "registration.json").write_text("{}")
    depth = 0.0135
    (repo / "config/palette_load.yaml").write_text(yaml.safe_dump({"schema_version": 1, "utc": None, "slots": {
        "inkcap_large_2": {"ink": "nighthawk_black", "cap_present": True, "level_lower_bound_m": 0.7 * depth}}}))
    sheet = repo / "config/tools/lutin-3rl-bugpin.yaml"
    _reach(sheet, "2.0")
    monkeypatch.setattr(lease, "IN_CAP_DIR", tmp_path / "state")
    fix = station.StationFix("right", FIX_POSE, (FIX_POSE @ [0, 0, 0.0556, 1])[:3], "2026-10-05T01:08:32Z",
                             "overhead_tag", {"registration_sha256": "r" * 64})
    monkeypatch.setattr("tatbot_bridge.station.observe", lambda *a, **k: fix)
    start = PAGE.copy()
    start[:3, :3] = PAGE[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    start[:3, 3] += PAGE[:3, 2] * 0.010
    q = kin.solve(start, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5708, 0.0]))   # from rest
    tcp_m = (np.linalg.inv(kin.frame(q, "right/tool_mount")) @ kin.fk(q))[:3, 3]
    touch = station.StationTouch("right", "lutin-3rl-bugpin", fix.ball + TOUCH_OFFSET, TOUCH_ROTATION,
                                 tcp_m, station.joint_offsets(repo, "right"), fix, "ros-calib-test", fix.measured_utc)
    path = touch.write(tmp_path / "station-touch-right.json")
    monkeypatch.setattr(station, "touch_path", lambda arm: path)
    resource = {"id": "black", "slot": "inkcap_large_2", "ink_id": "nighthawk_black",
                "tool": {"id": "lutin-3rl-bugpin"}, "dip": {"above_ink_m": 0.0005, "dwell_s": 0.5, "hover_m": 0.008,
                                                            "speed_m_s": 0.01, "wall_margin_m": 0.0015, "mm_per_dip": 40.0}}
    ex = Executor(repo, tmp_path / "run", kin, q)
    return SimpleNamespace(ledger=Ledger(tmp_path / "ledger.jsonl", ex.clock), ex=ex, resource=resource, touch=touch, fix=fix, cap=cap, lease=lease, q=q, start=start)


def test_a_dip_aims_by_the_touch_dwells_with_the_machine_on_and_returns_to_the_page(rig, kin):
    from tatbot_session import config

    assert config.tool_datasheet(rig.ex.repo, "lutin-3rl-bugpin").raw["needle_reach_mm"] == 2.0
    ledger = rig.ledger
    assert rig.cap.dip(rig.ex, rig.resource, "d0", 1, ledger) is True
    phases = [(row["phase"], row["status"]) for row in ledger.rows() if row["event"] == "dip_phase"]
    assert phases == [(p, "ok") for p in ("approach", "descend", "dwell", "retract", "return")]
    goals = rig.ex.io.goals
    # the dwell held the tool's end over the ink at the cap the touch places, not where the fix put it
    dwell = kin.fk(goals[2].q[-1])
    touched = station.anchored(FIX_POSE, rig.touch.ball, station.ball_in_palette(REPO / "urdf/palette.urdf"))
    from tatbot_motion.dip import read_cap

    cap = read_cap(rig.ex.repo, "inkcap_large_2", "right", touched)
    local = np.linalg.inv(cap.base_from_cap) @ np.append(dwell[:3, 3], 1.0)
    assert np.hypot(*local[:2]) < 3e-4
    assert local[2] == pytest.approx(0.0005 + 0.7 * 0.0135 + 0.0005, abs=3e-4)
    # the machine ran for the dwell alone, and the snag guard was off before the first goal: a cap is not the work
    assert rig.ex.machine_log == ["off", "off", "off", "on", "off", "off", "off"]
    assert rig.ex.snag_log == [(False, 0)]
    # back at the page standoff it left, out of the cap, the marker gone
    np.testing.assert_allclose(kin.fk(rig.ex.io.q)[:3, 3], rig.start[:3, 3], atol=1e-3)
    assert rig.lease.in_cap("right") is None
    assert any("dip inkcap_large_2 at heading -60 deg, the touch's" in text and "from the overhead fix" in text
               for text in rig.ex.events)


def test_a_dip_goes_only_at_the_touchs_heading_and_refuses_a_cap_it_does_not_reach(rig):
    """2026-10-08: the 3RL's tip read ~10 mm off across its axis, which a turn away from the touches' heading carries
    off the cap's axis (~8 mm at 45 deg). Touched at a heading from which the arm cannot reach the cap, the dip is
    refused before anything moves, and no other heading is tried."""
    import dataclasses

    dataclasses.replace(rig.touch, rotation=np.diag([-1.0, 1.0, -1.0])).write(station.touch_path("right"))   # 180
    ledger = rig.ledger
    with pytest.raises(RuntimeError, match=r"the touch's heading \+180 deg does not reach cap inkcap_large_2"):
        rig.cap.dip(rig.ex, rig.resource, "d0", 1, ledger)
    assert rig.ex.io.goals == []
    assert [(row["phase"], row["status"]) for row in ledger.rows()] == [("approach", "unplanned")]
    assert any("dip inkcap_large_2 at heading +180 deg, the touch's" in text for text in rig.ex.events)


def test_without_a_recorded_needle_reach_the_dwell_holds_with_the_machine_off(rig):
    sheet = rig.ex.repo / "config/tools/lutin-3rl-bugpin.yaml"
    _reach(sheet, "null")
    assert rig.cap.dip(rig.ex, rig.resource, "d0", 1, rig.ledger) is True
    assert "on" not in rig.ex.machine_log
    assert any("machine off (no needle reach recorded)" in text for text in rig.ex.events)


def test_an_interrupted_descent_withdraws_along_the_axis_and_dips_again(rig, kin):
    rig.ex.io.script = [("ok", None), ("latched", 30)]   # the approach ends; the descent latches part way down
    ledger = rig.ledger
    assert rig.cap.dip(rig.ex, rig.resource, "d0", 1, ledger) is False   # withdrew; the executor dips again
    phases = [(row["phase"], row["status"]) for row in ledger.rows() if row["event"] == "dip_phase"]
    assert phases == [("approach", "ok"), ("descend", "latched"), ("retract", "ok"), ("return", "ok")]
    assert rig.ex.decisions == ["latched"] and rig.lease.in_cap("right") is None
    retract = rig.ex.io.goals[2]
    np.testing.assert_allclose(kin.fk(retract.q[-1])[:2, 3], kin.fk(retract.q[0])[:2, 3], atol=2e-3)
    np.testing.assert_allclose(kin.fk(rig.ex.io.q)[:3, 3], rig.start[:3, 3], atol=1e-3)


def lagging(kin, short_m):
    def stop(traj):   # the knot the arm has reached when the goal ends short_m under its last one
        z = np.array([kin.fk(q)[2, 3] for q in traj.q])
        return int(np.argmax(z >= z[-1] - short_m))

    return stop


@pytest.mark.parametrize(("short_m", "status"), [(0.003, "ok"), (0.005, "off")])
def test_a_retract_may_end_a_little_under_its_hover_as_the_arm_catches_up(rig, kin, short_m, status):
    """2026-10-05: a retract out of cap M2 ended 0.5 mm under its hover and the dip stopped for a decision, though
    the tool was 7.5 mm over the rim. Ending within HOVER_LAG_M (4 mm since the owner doubled the stops) goes on; one
    further short withdraws and dips again once with no key."""
    rig.ex.io.script = [("ok", None)] * 3 + [("ok", lagging(kin, short_m))]
    ledger = rig.ledger
    assert rig.cap.dip(rig.ex, rig.resource, "d0", 1, ledger) is True
    phases = [(row["phase"], row["status"]) for row in ledger.rows() if row["event"] == "dip_phase"]
    assert phases[3] == ("retract", status) and rig.ex.decisions == []
    if status == "off":   # out along the axis, back to the page, then the whole dip again
        assert phases[4:] == [("retract", "ok"), ("return", "ok")] + [
            (p, "ok") for p in ("approach", "descend", "dwell", "retract", "return")]
    assert rig.lease.in_cap("right") is None


def test_a_phase_off_its_place_twice_waits_for_a_decision(rig, kin):
    rig.ex.io.script = [("ok", None)] * 3 + [("ok", lagging(kin, 0.005))] + [("ok", None)] * 5 + [
        ("ok", lagging(kin, 0.005))]
    ledger = rig.ledger
    assert rig.cap.dip(rig.ex, rig.resource, "d0", 1, ledger) is False   # withdrew; the executor dips again
    assert rig.ex.decisions == ["latched"]
    assert [row["status"] for row in ledger.rows() if row.get("phase") == "retract"] == ["off", "ok", "off", "ok"]


@pytest.mark.parametrize(("settle", "returns"), [
    ([(0.008, 3)], 1),        # ~2.3 mm off for three readings, then on its place: no second goal
    ([(0.012, 99)], 2),       # ~3.5 mm off for good: planned again once from there, and on its place
    ([(0.012, 99)] * 2, 2),   # off after the second too: the dip is made, and goes on from where the arm is
])
def test_a_return_is_judged_once_the_arm_settles_and_a_miss_never_waits(rig, kin, settle, returns):
    """2026-10-20: two returns judged as their goals ended read 1.09, then 1.66 mm off the page standoff while joint 4
    rang after the quintic stop, and the second ended a tattoo with the arm holding."""
    rig.ex.io.settle = [None] * 4 + settle
    ledger = rig.ledger
    assert rig.cap.dip(rig.ex, rig.resource, "d0", 1, ledger) is True
    statuses = [row["status"] for row in ledger.rows() if row["event"] == "dip_phase" and row["phase"] == "return"]
    assert len(statuses) == returns and statuses[-1] == ("missed" if len(settle) == 2 else "ok")
    assert rig.ex.decisions == [] and len(rig.ex.io.goals) == 4 + returns
    assert [phase for phase in (row.get("phase") for row in ledger.rows()) if phase == "approach"] == ["approach"]
    if returns == 1:
        assert rig.ex.clock.t == pytest.approx(0.15)   # three readings 50 ms apart
    else:
        assert any("planned again from where the arm stands" in text for text in rig.ex.events)


@pytest.mark.parametrize(("change", "message"), [
    ("tool", "touched with lutin-3rl-bugpin, and picosecond-laser-pen is fitted"),   # another machine
    ("moved", "the palette moved"),
    ("load", "must be declared with nighthawk_black"),
])
def test_a_dip_the_touch_or_the_cap_cannot_speak_for_is_refused_before_anything_moves(rig, monkeypatch, change, message):
    if change == "tool":
        rig.resource["tool"] = {"id": "picosecond-laser-pen"}
    elif change == "moved":
        moved = station.StationFix("right", FIX_POSE.copy(), rig.fix.ball + [0.004, 0, 0], rig.fix.measured_utc,
                                   "overhead_tag", rig.fix.detail)
        moved.base_from_palette[0, 3] += 0.004
        monkeypatch.setattr("tatbot_bridge.station.observe", lambda *a, **k: moved)
    else:
        (rig.ex.repo / "config/palette_load.yaml").write_text(yaml.safe_dump({"slots": {}}))
    with pytest.raises((station.StaleStationError, RuntimeError), match=message):
        rig.cap.dip(rig.ex, rig.resource, "d0", 1, rig.ledger)
    assert rig.ex.io.goals == []


def test_the_touch_carries_a_small_tcp_change_with_it(rig):
    delta = np.array([0.0004, -0.0003, 0.0])
    placed = station.touched_palette(rig.touch, rig.fix, tool_id="lutin-3rl-bugpin", tcp_m=rig.touch.tcp_m + delta,
                                     joint_offsets=rig.touch.joint_offsets, palette_urdf=REPO / "urdf/palette.urdf")
    ball = (placed @ np.append(station.ball_in_palette(REPO / "urdf/palette.urdf"), 1.0))[:3]
    np.testing.assert_allclose(ball, rig.touch.ball + rig.touch.rotation @ delta, atol=1e-12)
    with pytest.raises(station.StaleStationError, match="the tcp moved"):
        station.touched_palette(rig.touch, rig.fix, tool_id="lutin-3rl-bugpin", tcp_m=rig.touch.tcp_m + [0.01, 0, 0],
                                joint_offsets=rig.touch.joint_offsets, palette_urdf=REPO / "urdf/palette.urdf")
    assert json.loads(station.touch_path("right").read_text())["schema"] == station.TOUCH_SCHEMA


def test_another_cartridge_in_the_same_machine_carries_the_touch(rig):
    assert station.same_machine(REPO, "lutin-3rl-bugpin", "lutin-5rl-mast")
    assert not station.same_machine(REPO, "lutin-3rl-bugpin", "picosecond-laser-pen")
    delta = np.array([0.0, 0.0, 0.0012])      # the other cartridge 1.2 mm longer, as the wrist gauge's fits say
    placed = station.touched_palette(rig.touch, rig.fix, tool_id="lutin-5rl-mast", tcp_m=rig.touch.tcp_m + delta,
                                     joint_offsets=rig.touch.joint_offsets, palette_urdf=REPO / "urdf/palette.urdf",
                                     same_body=True)
    ball = (placed @ np.append(station.ball_in_palette(REPO / "urdf/palette.urdf"), 1.0))[:3]
    np.testing.assert_allclose(ball, rig.touch.ball + rig.touch.rotation @ delta, atol=1e-12)
    with pytest.raises(station.StaleStationError, match="the station was touched with"):
        station.touched_palette(rig.touch, rig.fix, tool_id="lutin-5rl-mast", tcp_m=rig.touch.tcp_m,
                                joint_offsets=rig.touch.joint_offsets, palette_urdf=REPO / "urdf/palette.urdf")


@pytest.mark.parametrize("shift_m", [0.0005, 0.0015])
def test_a_palette_moved_under_twice_the_fixs_repeat_is_dipped_at_the_carried_ball(rig, kin, monkeypatch, shift_m):
    """Past the fix's 1 mm repeatability the touched ball is carried across by the fixes' own shift, up to twice it
    (owner, 2026-10-20: every stop doubled); within it the ball stays where the touches found it. 4 mm refuses."""
    moved = station.StationFix("right", FIX_POSE.copy(), rig.fix.ball + [shift_m, 0, 0], rig.fix.measured_utc,
                               "overhead_tag", rig.fix.detail)
    moved.base_from_palette[0, 3] += shift_m
    placed = station.touched_palette(rig.touch, moved, tool_id="lutin-3rl-bugpin", tcp_m=rig.touch.tcp_m,
                                     joint_offsets=rig.touch.joint_offsets, palette_urdf=REPO / "urdf/palette.urdf")
    ball = (placed @ np.append(station.ball_in_palette(REPO / "urdf/palette.urdf"), 1.0))[:3]
    carried = shift_m if shift_m > station.MOVED_M["overhead_tag"] else 0.0
    np.testing.assert_allclose(ball, rig.touch.ball + [carried, 0, 0], atol=1e-9)
    monkeypatch.setattr("tatbot_bridge.station.observe", lambda *a, **k: moved)
    assert rig.cap.dip(rig.ex, rig.resource, "d0", 1, rig.ledger) is True
