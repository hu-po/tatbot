"""ArmExecutor decisions without ROS I/O: latch mid-stroke -> aborted arc -> continue resumes there;
continue refused after the arm moved; resume timeout lands; pre-plan reuse versus dispatch drift; a page
that moves stops the run unadopted; three touches fit the page plane; a Decide answers only the wait it was made for; a
latch in a stroke's final lift finishes the stroke; landing lifts a pen that is down first; a landed arm is refused
every goal. The arm I/O, kinematics and planner are small fakes;
the driver itself is only exercised on hardware and through the mock end-to-end test."""
from __future__ import annotations

import threading
import time
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import tatbot_motion
from tatbot_session import executor as ex_mod
from tatbot_session import geometry, rules
from tatbot_session.ledger import Ledger

MOTION = {"approach": {"standoff_m": 0.010}, "joint_speed": {"pen_up_rad_s": 1.0}, "carriage": {"max_m_s": 0.001},
          "pen": {"mode": "press", "press": {"lift_m": -0.002, "settle_s": 1.0},
                  "ride": {"fraction": 0.5, "settle_s": 0.0},
                  "trim": {"step_m": 0.0001, "limit_m": 0.002, "speed_m_s": 0.002, "min_s": 0.2, "lead_s": 0.1}},
          "touch": {"lift_m": 0.010, "max_travel_m": 0.040}, "dispatch_drift": {"start_tolerance_m": 0.0005},
          "resume_reapproach_m": 0.0005, "clik": {"joint_limit_margin_rad": 0.0333},
          "probe": {"axial_cap_m": 0.001, "lateral_cap_m": 0.0025, "slow_m_s": 0.001, "back_off_m": 0.003,
                    "clearance_m": 0.030}}
STACK = {"session": {"knot_rate_hz": 100, "resume_timeout_s": 300},
         "page": {"source": "fixed", "clear_m": [0.062, 0.112], "replan_translation_m": 0.001,
                  "replan_rotation_rad": 0.0087, "trim": {"right": [0.0, 0.0]}},
         "safety": {"landing": {"budget_s": 45.0}}}
LINE = {"op": "stroke", "id": "s0000", "closed": False, "points_m": [[0.0, 0.0], [0.02, 0.0]]}
LINE2 = {"op": "stroke", "id": "s0001", "closed": False, "points_m": [[0.0, 0.01], [0.02, 0.01]]}


class FakeKin:
    joint_names = tuple(f"right/j{i}" for i in range(7))
    lower, upper = np.full(7, -10.0), np.full(7, 10.0)

    def fk(self, q):
        out = np.eye(4)
        out[:3, 3] = np.asarray(q)[:3]
        return out


class FakeIO:
    def __init__(self):
        self.arm = "right"
        self.q = np.zeros(7)
        self.safety = {"estop_ok": 1.0, "latched": 0.0}
        self.script: list = []
        self.writes: list[dict] = []
        self.unlatches = 0
        self.lands = 0
        self.trip = None
        self.unlatch_delay = 0.0
        self.on_write = None
        self.executed: list = []
        self.active = None
        self.at_tick = {}         # index -> called at that tick, before a retarget (a key press mid-goal)
        self.replaced: list = []  # (elapsed, the replacement) for each retarget taken

    latched = property(lambda self: self.safety.get("latched", 0) >= 0.5)
    estop_ok = property(lambda self: self.safety.get("estop_ok", 1) >= 0.5)
    latch_reason = property(lambda self: int(self.safety.get("latch_reason", 0)))

    def flag(self, name, default=0.0):
        return self.safety.get(name, default) >= 0.5

    def latch_text(self):
        return "estop"

    def measured(self):
        return self.q.copy()

    def seed(self):
        return self.q.copy()

    def record(self, on):
        return []

    def trip_joints(self):
        return self.trip

    def write(self, **values):
        self.writes.append(values)
        if self.on_write:
            self.on_write(values)

    def wait(self, predicate, timeout):
        return predicate()

    def unlatch(self):
        time.sleep(self.unlatch_delay)
        self.unlatches += 1
        self.safety.update(latched=0.0, latch_reason=0.0)
        return True, "unlatched"

    def land(self, budget_s, on_text=None, staged=None):
        self.lands += 1
        self.safety.update(landed=1.0)   # as the driver: landed and idle until it is woken
        return True, "landed and idle"

    def execute(self, traj, *, should_stop=None, on_tick=None, watch_latch=True, retarget=None):
        """Pop the next scripted outcome: (status, index, safety updates, q at the end); none left runs
        the trajectory to its end. A retarget is offered at every tick, at that knot's time."""
        self.executed.append(traj)
        self.active = traj
        status, index, safety, q = self.script.pop(0) if self.script else ("ok", len(traj.t) - 1, {}, None)
        for i in range(index + 1):
            if on_tick:
                on_tick(i)
            if i in self.at_tick:
                self.at_tick.pop(i)()
            new = retarget(float(traj.t[i]), self.active) if retarget else None
            if new is not None:
                self.replaced.append((float(traj.t[i]), new))
                self.active = new
        self.safety.update(safety)
        self.q = np.asarray(q if q is not None else self.active.q[index], float)
        return status, index, "scripted"


class FakeNode:
    def __init__(self, camera):
        self.camera = camera
        self.page_info = {"source": "fixed", "pattern_id": "fixed_right"}
        self.later_info = None   # the page as the cameras report it after the first wait
        self.waits: list[float] = []
        self.events: list[tuple] = []

    def event(self, kind, *, arm="", op_id="", text=""):
        self.events.append((kind, op_id, text))

    def wait_camera_page(self, arm, timeout, max_lost_s=None, should_stop=None):
        self.waits.append(timeout)
        if len(self.waits) > 1 and self.later_info is not None:
            self.page_info = self.later_info
        return self.camera, dict(self.page_info)

    def camera_page(self, arm):
        return self.camera, {}


def line_traj(op, from_arc=0.0, n=11):
    """A plan whose tip runs along the stroke from from_arc (page = base, z = 0)."""
    pts = np.asarray(op["points_m"], float)
    length = geometry.stroke_length(pts)
    arc = np.linspace(from_arc, length, n)
    tip = np.column_stack([np.interp(arc, [0, length], pts[:, 0]), np.interp(arc, [0, length], pts[:, 1]),
                           np.zeros(n)])
    return stroke_plan(np.linspace(0, 1, n), tip, arc, np.full(n, 5, np.uint8))


def stroke_plan(t, tip, arc, phase):
    """A stroke plan over FakeKin (tip = q[:3]) with its depth axis, page z: a trim moves q[2]."""
    n = len(t)
    up = np.zeros((n, 7))
    up[:, 2] = 1.0
    return tatbot_motion.Trajectory(FakeKin.joint_names, np.asarray(t, float), np.column_stack([tip, np.zeros((n, 4))]),
                                    np.zeros((n, 7)), np.asarray(tip, float), np.asarray(arc, float),
                                    np.asarray(phase, np.uint8), axis=np.tile([0.0, 0.0, 1.0], (n, 1)), dq_dh=up)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    calls = []

    def plan_op(op, *, base_from_page, q_seed, kin, motion, speed_m_s, from_arc_m=0.0, pen_down_at_start=False,
                lift_at_end=True, pen=None):
        calls.append({"op": op["id"], "from_arc": from_arc_m, "pen_down": pen_down_at_start, "lift_at_end": lift_at_end,
                      "seed": q_seed.copy(), "pen": pen, "points": np.asarray(op["points_m"], float)})
        return line_traj(op, from_arc_m)

    monkeypatch.setattr(tatbot_motion, "plan_op", plan_op, raising=False)
    monkeypatch.setattr(tatbot_motion, "to_knots", lambda traj, rate: traj, raising=False)
    monkeypatch.setattr(tatbot_motion, "dispatch_drift", lambda traj, q, kin, motion, pen_down=None:
                        {"ok": True, "joint_rad": 0.0, "tip_m": 0.0}, raising=False)
    monkeypatch.setattr(ex_mod.config, "fitted_tool", lambda repo, arm: "lutin-ballpoint-dot")
    lifts = []

    def plan_lift(*, q_seed, direction, distance_m, kin, motion):
        lifts.append(distance_m)
        q = np.array([q_seed, q_seed], float)
        q[1, :3] += np.asarray(direction) * distance_m
        return SimpleNamespace(t=np.array([0.0, 1.0]), q=q, qd=np.zeros((2, 7)), tip=q[:, :3],
                               arc_m=np.full(2, np.nan), phase=np.full(2, 6, np.uint8))

    monkeypatch.setattr(tatbot_motion, "plan_lift", plan_lift, raising=False)
    hovers = []

    def plan_hover(op, *, base_from_page, q_seed, kin, motion, from_arc_m=0.0, pen=None):
        hovers.append({"op": op["id"], "start": np.asarray(op["points_m"][0], float), "seed": q_seed.copy()})
        q = np.array([q_seed, q_seed], float)
        q[1, :3] = [*op["points_m"][0], 0.01]
        return SimpleNamespace(t=np.array([0.0, 1.0]), q=q, qd=np.zeros((2, 7)), tip=q[:, :3],
                               arc_m=np.full(2, np.nan), phase=np.full(2, 1, np.uint8), info={"hover": True})

    monkeypatch.setattr(tatbot_motion, "plan_hover", plan_hover, raising=False)
    io, node = FakeIO(), FakeNode(np.eye(4))
    ex = ex_mod.ArmExecutor(node, io, FakeKin(), MOTION, STACK, tmp_path)
    ex.go_ready = lambda ledger=None: None
    ex.tool_id = "lutin-ballpoint-dot"
    return SimpleNamespace(ex=ex, io=io, node=node, calls=calls, lifts=lifts, hovers=hovers,
                           ledger=Ledger(tmp_path / "ledger.jsonl"))


def draw_strokes(ex, program, ledger, *, from_op='', touch=False):
    """The stroke loop alone, for strokes of one ballpoint resource (no tool change, no page row)."""
    import dataclasses

    ex.drawn = {}
    ex.prepare_pen(program)
    ex.setup_page(program, touch, ledger)
    ex.page_record["pen"] = dataclasses.asdict(ex.pen)
    ex.resource = {"id": "fitted", "dip": None, "tool": {}}
    program = {**program, "ops": [{"resource_id": "fitted", **op} for op in program["ops"]]}
    try:
        start = next((i for i, op in enumerate(program['ops']) if op['id'] == from_op), 0) if from_op else 0
        ex._run_ops(program, start, ledger)
    finally:
        ex.machine_off()
    return 'complete'


class Operator:
    """Sends Decides through ArmExecutor.decide, each once the executor waits (one wait's worth)."""

    def __init__(self, ex, *decisions):
        self.results: list = []
        self.thread = threading.Thread(target=self._run, args=(ex, decisions), daemon=True)
        self.thread.start()

    def _run(self, ex, decisions):
        for decision in decisions:
            deadline = time.monotonic() + 5.0
            while ex.waiting is None and time.monotonic() < deadline:
                time.sleep(0.002)
            self.results.append(ex.decide(decision, timeout=5.0))

    def replies(self):
        self.thread.join(10.0)
        return self.results


def test_latch_mid_stroke_then_continue_resumes_at_its_arc(setup):
    ex, io = setup.ex, setup.io
    latched = {"latched": 1.0, "latch_reason": 1.0, "estop_ok": 1.0}
    io.script = [("latched", 6, latched, [0.0121, 0.0, 0.0, 0, 0, 0, 0]), ("ok", 10, {}, None)]
    operator = Operator(ex, rules.CONTINUE)
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert operator.replies() == [(True, "")]
    rows = setup.ledger.rows()
    assert [r["event"] for r in rows] == ["sent", "aborted", "decision", "sent", "done"]
    assert rows[1]["arc_m"] == pytest.approx(0.012) and rows[1]["reason"] == "latched: scripted"
    assert rows[3]["arc_m"] == pytest.approx(0.012) and rows[4]["arc_m"] == pytest.approx(0.02)
    assert setup.calls[1]["from_arc"] == pytest.approx(0.012) and setup.calls[1]["pen_down"] is True
    assert io.unlatches == 1


def test_continue_is_refused_after_the_arm_moved_and_land_ends_the_draw(setup):
    ex, io = setup.ex, setup.io
    io.script = [("latched", 3, {"latched": 1.0, "latch_reason": 1.0}, [0.006, 0, 0, 0, 0, 0, 0])]
    real_measured = io.measured
    calls = {"n": 0}

    def drifting():  # the operator pushed the held arm 0.1 rad after the latch
        calls["n"] += 1
        q = real_measured()
        if calls["n"] > 2:
            q[3] += 0.1
        return q

    io.measured = drifting
    operator = Operator(ex, rules.CONTINUE, rules.LAND)
    with pytest.raises(ex_mod.Landed):
        draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False)
    (ok, why), landed = operator.replies()
    assert not ok and "moved" in why
    assert landed == (True, "")
    assert [r["event"] for r in setup.ledger.rows()] == ["sent", "aborted", "decision"]
    # the pen was down: landing unlatched with the hold goal and lifted to the standoff before the driver landed
    assert io.unlatches == 1 and setup.lifts == [pytest.approx(0.010)] and io.lands == 1


def test_continue_is_refused_while_the_estop_is_pressed(setup):
    ex, io = setup.ex, setup.io
    io.script = [("latched", 3, {"latched": 1.0, "latch_reason": 1.0, "estop_ok": 0.0}, None)]
    operator = Operator(ex, rules.CONTINUE, rules.LAND)
    with pytest.raises(ex_mod.Landed):
        draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False)
    refused, landed = operator.replies()
    assert "e-stop" in refused[1] and landed[0]
    assert io.unlatches == 0 and setup.lifts == []  # pressed: no lift, the driver lands from the hold


def test_resume_timeout_announces_and_lands(setup):
    ex, io = setup.ex, setup.io
    ex.stack = dict(STACK, session={"knot_rate_hz": 100, "resume_timeout_s": 0.2})
    io.script = [("latched", 3, {"latched": 1.0, "latch_reason": 1.0}, None)]
    with pytest.raises(ex_mod.Landed):
        draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False)
    assert any("after release: landing" in text for _, _, text in setup.node.events)


def test_preplan_is_used_unless_the_measured_pose_drifted(setup, monkeypatch):
    ex, io = setup.ex, setup.io
    io.script = [("ok", 10, {}, None), ("ok", 10, {}, None)]
    assert draw_strokes(ex, {"ops": [LINE, LINE2]}, setup.ledger, touch=False) == "complete"
    # s0001 was planned once, in the background, seeded at the end of s0000's plan
    assert [c["op"] for c in setup.calls] == ["s0000", "s0001"]
    assert np.allclose(setup.calls[1]["seed"][:3], [0.02, 0.0, 0.0])
    setup.calls.clear()
    monkeypatch.setattr(tatbot_motion, "dispatch_drift", lambda traj, q, kin, motion, pen_down=None:
                        {"ok": False, "joint_rad": 0.03, "tip_m": 0.004}, raising=False)
    io.script = [("ok", 10, {}, None), ("ok", 10, {}, None)]
    assert draw_strokes(ex, {"ops": [dict(LINE, id="s0002"), dict(LINE2, id="s0003")]}, setup.ledger, touch=False) == "complete"
    assert [c["op"] for c in setup.calls] == ["s0002", "s0003", "s0003"]  # the pre-plan drifted: planned again
    assert any(kind == "replan" and "drift" in text for kind, _, text in setup.node.events)


def test_keys_trim_the_cartridge_on_the_arm_a_step_at_a_time_within_its_limit(setup):
    """up and down move the cartridge's trim one pen.trim step within +-limit_m and reset sets 0, each change a
    ledger row; another cartridge keeps its own; without a running draw nothing is trimmed."""
    ex, events = setup.ex, setup.node.events
    ex.on_key("up")
    assert ex.trims == {} and "no draw is running" in events[-1][2]
    ex.resource, ex.ledger = {"id": "r1"}, setup.ledger
    for key in ("up", "up", "up", "down"):
        ex.on_key(key)
    assert ex.trim_target() == pytest.approx(0.0002) and events[-1][:1] == ("pen_trim",)
    for _ in range(25):
        ex.on_key("down")
    assert ex.trim_target() == pytest.approx(-0.002) and events[-1][2] == "r1 stays -2.0 mm: its limit"
    ex.resource = {"id": "r2"}
    ex.on_key("up")
    ex.resource = {"id": "r1"}
    ex.on_key("reset")
    assert ex.trims == {"r1": 0.0, "r2": pytest.approx(0.0001)}
    rows = [(r["resource"], r["trim_m"]) for r in setup.ledger.rows() if r["event"] == "pen_trim"]
    assert len(rows) == 4 + 22 + 2 and rows[-2:] == [("r2", 0.0001), ("r1", 0.0)]


def test_enter_continues_a_waiting_latch_and_says_when_nothing_waits(setup):
    ex, io, events = setup.ex, setup.io, setup.node.events
    ex.on_key("enter")
    assert events[-1][0] == "decision" and events[-1][2].startswith("enter: ")
    io.safety.update(landed=1.0)
    ex.on_key("enter")   # a cartridge swap's: client draw resumes the run
    assert len(events) == 1
    io.safety.update(landed=0.0)
    io.script = [("latched", 3, {"latched": 1.0, "latch_reason": 1.0}, None)]

    def press():
        deadline = time.monotonic() + 5.0
        while ex.waiting is None and time.monotonic() < deadline:
            time.sleep(0.002)
        ex.on_key("enter")

    thread = threading.Thread(target=press, daemon=True)
    thread.start()
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    thread.join(5.0)
    assert io.unlatches == 1 and [r["event"] for r in setup.ledger.rows()][-3:] == ["decision", "sent", "done"]


RIDE = {**MOTION, "pen": {**MOTION["pen"], "mode": "ride"}}
BALLPOINT = SimpleNamespace(tool_id="lutin-ballpoint-dot", stroke_m=0.0035, tip_out_at_top_m=0.002)
PROGRAM_TOOL = {"id": "lutin-ballpoint-dot"}


class Switch:
    """The machine's switch: `log` holds every change asked for; starts=False never reports it running."""

    def __init__(self, starts=True):
        self.on, self.log, self.starts = False, [], starts

    def set(self, on):
        if on != self.on:
            self.log.append(on)
        self.on = on

    def wait(self, on, timeout_s):
        self.set(on)
        return self.starts or not on

    def why_off(self):
        return "the e-stop is pressed"


@pytest.fixture
def riding(setup, monkeypatch):
    monkeypatch.setattr(ex_mod.config, "tool_datasheet", lambda repo, tool_id: BALLPOINT)
    setup.ex.motion, setup.ex.machine = RIDE, Switch()
    return setup


def test_a_riding_pen_runs_the_machine_for_its_strokes_and_stops_it_whenever_the_loop_waits_or_ends(riding):
    ex, io, node = riding.ex, riding.io, riding.node
    latched = {"latched": 1.0, "latch_reason": 1.0, "estop_ok": 1.0}
    io.script = [("latched", 6, latched, [0.0121, 0.0, 0.0, 0, 0, 0, 0]), ("ok", 10, {}, None), ("ok", 10, {}, None)]
    operator = Operator(ex, rules.CONTINUE)
    assert draw_strokes(ex, {"ops": [LINE, LINE2], "tool": PROGRAM_TOOL}, riding.ledger, touch=False) == "complete"
    assert operator.replies() == [(True, "")]
    # on for s0000, off while the latch waits, on again for the rest of it and s0001, off at the end
    assert ex.machine.log == [True, False, True, False]
    kinds = [(kind, text) for kind, _, text in node.events]
    assert kinds.index(("machine", "machine on")) < kinds.index(next(k for k in kinds if k[0] == "op_sent"))
    assert all(c["pen"].machine and c["pen"].height_m == pytest.approx(0.00175) for c in riding.calls)
    assert ("page", "pen down: " + ex.pen.describe()) in kinds and ex.page_record["pen"]["mode"] == "ride"
    ex.motion = MOTION   # press: the machine stays off
    assert draw_strokes(ex, {"ops": [dict(LINE, id="s0002")]}, riding.ledger, touch=False) == "complete"
    assert not riding.calls[-1]["pen"].machine and ex.machine.log == [True, False, True, False]


def test_a_draw_refuses_before_any_motion_a_program_for_another_tool_and_a_ride_it_cannot_give(riding, monkeypatch):
    ex, io = riding.ex, riding.io
    with pytest.raises(RuntimeError, match="prepared for lutin-3rl-bugpin"):
        draw_strokes(ex, {"ops": [LINE], "tool": {"id": "lutin-3rl-bugpin"}}, riding.ledger, touch=False)
    monkeypatch.setattr(ex_mod.config, "tool_datasheet", lambda repo, tool_id: SimpleNamespace(
        tool_id=tool_id, stroke_m=0.0035, tip_out_at_top_m=None))
    with pytest.raises(ValueError, match="tip_out_at_top_mm"):
        draw_strokes(ex, {"ops": [LINE], "tool": PROGRAM_TOOL}, riding.ledger, touch=False)
    monkeypatch.setattr(ex_mod.config, "tool_datasheet", lambda repo, tool_id: BALLPOINT)
    ex.machine = None
    with pytest.raises(RuntimeError, match="switches none"):
        draw_strokes(ex, {"ops": [LINE], "tool": PROGRAM_TOOL}, riding.ledger, touch=False)
    assert io.executed == []


def test_a_machine_that_does_not_start_pauses_the_run(riding):
    ex, node = riding.ex, riding.node
    ex.machine = Switch(starts=False)
    operator = Operator(ex, rules.LAND)
    with pytest.raises(ex_mod.Landed):
        draw_strokes(ex, {"ops": [LINE], "tool": PROGRAM_TOOL}, riding.ledger, touch=False)
    assert operator.replies() == [(True, "")]
    assert not ex.machine.on and not any(kind == "op_sent" for kind, _, _ in node.events)
    assert ("pause", "the tattoo machine did not start: the e-stop is pressed; continue asks again, land lands") in [
        (kind, text) for kind, _, text in node.events]


def test_page_motion_past_the_threshold_stops_the_run_without_adopting_it(setup):
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    ex.camera = np.eye(4)
    node.camera = geometry.rpy_matrix([0.0009, 0, 0], [0, 0, 0])
    ex.refresh_page()  # within the threshold: nothing happens
    assert not ex.page_record.get("replans")
    # a quarter turn and 112 mm away, as the tracker published while the arm hid the print (2026-09-30)
    node.camera = geometry.rpy_matrix([0.1123, 0, 0], [0, 0, np.pi / 2])
    with pytest.raises(ex_mod.PageMovedError, match="not adopted"):
        ex.refresh_page()
    assert np.allclose(ex.used()[:3, 3], [0, 0, 0])  # the run's touched page stands
    assert ex.page_record["replans"][0]["adopted"] is False
    assert ex.page_record["replans"][0]["shift_m"] == pytest.approx(0.1123)
    assert any(kind == "error" and "not adopted" in text for kind, _, text in node.events)


def test_a_page_that_moves_before_a_stroke_stops_the_draw_before_the_op_is_sent(setup):
    ex, io, node = setup.ex, setup.io, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    real_setup = ex.setup_page

    def setup_page(program, touch, ledger):
        real_setup(program, touch, ledger)
        node.camera = geometry.rpy_matrix([0.0, 0.1123, 0], [0, 0, 0])  # jumps once the run has its page

    ex.setup_page = setup_page
    with pytest.raises(ex_mod.PageMovedError):
        draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False)
    assert not [row for row in setup.ledger.rows() if row.get("op") == "s0000"]  # never sent: resumable
    assert all(set(np.asarray(traj.phase).tolist()) <= {6} for traj in io.executed)  # at most the lift, no stroke


def test_three_touches_fit_the_page_plane(setup, monkeypatch):
    ex, io = setup.ex, setup.io
    camera = geometry.rpy_matrix([0.37, -0.23, 0.045], [0, 0, 0])  # the camera reads the page 12 mm high
    setup.node.camera = camera
    tilt = geometry.rpy_matrix([0, 0, 0], [0.02, -0.01, 0])[:3, 2]
    anchor = np.array([0.37, -0.23, 0.033])
    seen = []

    def plan_touch(*, q_seed, start_base_from_tcp, direction, max_travel_m, kin, motion, prior_distance_m=None,
                   speed_m_s=None):
        start = start_base_from_tcp[:3, 3]
        depth = (tilt @ (start - anchor)) / (tilt @ -direction)  # where the descent meets the tilted page
        contact = start + depth * direction
        seen.append(contact)
        io.trip = np.concatenate([contact, np.zeros(4)])
        io.script.append(("latched", 1, {"latched": 1.0, "latch_reason": 6.0}, io.trip))
        return SimpleNamespace(t=np.array([0.0, 1.0]), phase=np.array([3, 7], np.uint8), q=np.zeros((2, 7)))

    monkeypatch.setattr(tatbot_motion, "plan_touch", plan_touch, raising=False)
    monkeypatch.setattr(ex, "lift", lambda what, ledger=None: None)
    ex.stack = {**ex.stack, "touch": {"fit_tilt": True}}
    ex.setup_page({"ops": [LINE]}, True)
    assert len(ex.page_record["touches"]) == 3 and io.unlatches == 7   # three trips find the paper, two each after
    assert {"guard_mode": 1} in io.writes and io.writes[-1] == {"guard_mode": 0}
    used = ex.used()
    assert np.allclose(used[:3, 2], tilt, atol=1e-9)
    for contact in seen:
        assert abs(tilt @ (contact - used[:3, 3])) < 1e-9  # every touch lies on the used page plane
    assert np.allclose(used[:2, 3], camera[:2, 3], atol=5e-4)  # x, y still the camera's


def test_a_touch_counts_the_trips_on_the_paper_and_the_next_run_starts_from_it(setup, monkeypatch):
    """2026-10-03: the guard's torque force tripped 11 of 16 trips in the air, 9-25 mm over the paper, and two in the
    air met within 1.3 mm. Unknown, the paper is three trips within confirm_tol_m of the lowest, each further descent
    3 mm over the lowest and turned; known, two, and a trip over it by more than known_band_m is not counted. The
    next run starts from the paper this one found; a trip under that drops it and searches again."""
    ex, io = setup.ex, setup.io
    setup.node.camera = geometry.rpy_matrix([0.37, -0.23, 0.045], [0, 0, 0])   # the cameras' page, 12 mm high
    page, air = {"z": 0.033}, []    # the paper; how far each descent goes before a trip in the air (None: to the paper)
    starts, turns, lifts = [], [], []

    def plan_touch(*, q_seed, start_base_from_tcp, direction, max_travel_m, kin, motion, prior_distance_m=None,
                   speed_m_s=None):
        start = start_base_from_tcp[:3, 3]
        starts.append(float(start[2]))
        turns.append(round(float(np.degrees(np.arctan2(start_base_from_tcp[1, 0], start_base_from_tcp[0, 0]))), 6))
        early = air.pop(0) if air else None
        contact = start + (start[2] - page["z"] if early is None else early) * direction
        io.trip = np.concatenate([contact, np.zeros(4)])
        io.script.append(("latched", 1, {"latched": 1.0, "latch_reason": 6.0}, io.trip))
        return SimpleNamespace(t=np.array([0.0, 1.0]), phase=np.array([3, 7], np.uint8), q=np.zeros((2, 7)))

    monkeypatch.setattr(tatbot_motion, "plan_touch", plan_touch, raising=False)
    monkeypatch.setattr(ex, "lift", lambda what, ledger=None: lifts.append(what))
    ex.stack = {**ex.stack, "touch": {"fit_tilt": True}}
    air[:] = [0.002]                             # touch 0 trips in the air 8 mm over the prior, then meets the paper
    ex.setup_page({"ops": [LINE]}, True)
    trips = [t["trips_m"] for t in ex.page_record["touches"]]
    assert trips == [pytest.approx([0.008, -0.012, -0.012, -0.012]), pytest.approx([-0.012] * 2),
                     pytest.approx([-0.012] * 2)]                     # the next touches know the paper: two each
    assert io.unlatches == 8 and len(lifts) == 8    # lifted after every trip: no descent from a loaded pose
    assert starts[:5] == pytest.approx([0.055, 0.056, 0.036, 0.036, 0.037])   # 3 mm over the lowest; 4 over the paper
    assert turns[:5] == [turns[0], turns[0] - 20.0, turns[0] + 20.0, turns[0] - 20.0, turns[0]]
    assert ex.used()[2, 3] == pytest.approx(0.033)                   # the plane through the paper, not the air

    page["z"] = 0.028                            # the next run: the paper 5 mm under where the last touches found it
    ex.setup_page({"ops": [LINE]}, True)
    assert ex.page_record["carried"]["offset_m"] == pytest.approx(-0.012)
    assert [len(t["trips_m"]) for t in ex.page_record["touches"]] == [3, 2, 2]
    assert any("under the known paper" in text for _, _, text in setup.node.events)
    assert ex.used()[2, 3] == pytest.approx(0.028)

    air[:] = [0.0005] * 6                        # every trip 3.5 mm over the known paper: in the air, refused
    with pytest.raises(RuntimeError, match="no paper in 6 trips.*tripping in the air"):
        ex.setup_page({"ops": [LINE]}, True)


def probe_plan(monkeypatch, seen):
    """plan_touch as a settle, a fast leg to 4 mm short of the prior, and the slow leg to its end."""
    def plan_touch(*, q_seed, start_base_from_tcp, direction, max_travel_m, kin, motion, prior_distance_m=None,
                   speed_m_s=None, via=None):
        seen.append({"travel": max_travel_m, "prior": prior_distance_m, "speed": speed_m_s,
                     "via": [np.round(v, 6).tolist() for v in via or []]})
        along = np.array([0.0, 0.0, prior_distance_m - 0.004, prior_distance_m, max_travel_m])
        tip = start_base_from_tcp[:3, 3] + np.outer(along, direction)
        return SimpleNamespace(t=np.arange(5.0), phase=np.array([4, 4, 3, 7, 7], np.uint8), tip=tip,
                               q=np.column_stack([tip, np.zeros((5, 4))]), qd=np.zeros((5, 7)))

    monkeypatch.setattr(tatbot_motion, "plan_touch", plan_touch, raising=False)


def test_a_probe_touch_arms_at_rest_latches_the_edge_joints_backs_out_and_sees_the_release(setup, monkeypatch):
    ex, io = setup.ex, setup.io
    seen = []
    probe_plan(monkeypatch, seen)
    start, side = np.eye(4), np.array([1.0, 0.0, 0.0])
    start[:3, 3] = [0.30, 0.0, 0.10]
    edge = np.array([0.3052, 0.0, 0.10, 0, 0, 0, 0])   # the driver's joints at the kernel edge
    io.trip = edge
    io.script = [("latched", 3, {"latched": 1.0, "latch_reason": 7.0, "guard_tripped": 1.0, "probe_triggered": 1.0},
                  edge + [0.0003, 0, 0, 0, 0, 0, 0]),
                 ("ok", 1, {"probe_triggered": 0.0}, None)]   # the back-off releases the stylus
    out = ex.touch_single(start, side, 0.0, 0.0, guard=2, prior_distance_m=0.006)
    assert io.writes[0] == {"guard_mode": 2} and io.writes[-1] == {"guard_mode": 0}  # armed before the plan
    assert seen == [{"travel": pytest.approx(0.006 + 0.0025), "prior": 0.006, "speed": 0.001,
                     "via": [[0.0, 0.0, 0.13], [0.3, 0.0, 0.13]]}]   # up, across, then down onto the start
    assert out["tripped"] and np.allclose(out["q"], edge) and np.allclose(out["contact"][:3, 3], edge[:3])
    assert out["phase"] == 7 and io.unlatches == 1
    assert setup.lifts == [0.003] and np.allclose(out["back_off"]["direction"], -side)


def test_a_probe_that_is_silent_or_already_triggered_refuses_before_any_motion(setup, monkeypatch):
    ex, io = setup.ex, setup.io
    probe_plan(monkeypatch, [])

    def refuse(values):
        if values.get("guard_mode") == 2:
            io.safety.update(latched=1.0, latch_reason=7.0, guard_tripped=0.0, probe_triggered=1.0)

    io.on_write = refuse
    with pytest.raises(RuntimeError, match="read triggered when armed"):
        ex.touch_single(np.eye(4), np.array([0.0, 0.0, -1.0]), 0.0, 0.0, guard=2, prior_distance_m=0.005)
    assert io.executed == [] and io.unlatches == 1 and not io.latched and setup.lifts == []
    with pytest.raises(ValueError, match="prior_distance_m"):
        ex.touch_single(np.eye(4), np.array([0.0, 0.0, -1.0]), 0.0, 0.0, guard=2)


def test_a_probe_touch_that_reaches_its_cap_is_a_missing_trigger_and_backs_out(setup, monkeypatch):
    ex, io = setup.ex, setup.io
    seen = []
    probe_plan(monkeypatch, seen)
    out = ex.touch_single(np.eye(4), np.array([0.0, 0.0, -1.0]), 0.0, 0.0, guard=2, prior_distance_m=0.005)
    assert not out["tripped"] and seen[0]["travel"] == pytest.approx(0.006)
    # a contact too shallow to trip must not be left pressed: the tool backs out as after a trip
    assert io.unlatches == 0 and setup.lifts == [0.003] and "back_off" in out


def test_a_probe_touch_from_rest_on_the_joint_limits_moves_over_its_start_in_joint_space_first(setup, monkeypatch):
    ex, io = setup.ex, setup.io
    seen = []
    probe_plan(monkeypatch, seen)
    ex.kin.lower = np.full(7, -0.01)   # the staged pose rests on joint limits
    start = np.eye(4)
    start[:3, 3] = [0.30, 0.0, 0.10]
    moves = []

    def joint_move(joint_names, q_from, q_to, **kw):
        moves.append(q_to)
        tip = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.13]])
        return SimpleNamespace(t=np.array([0.0, 1.0]), q=np.column_stack([tip, np.zeros((2, 4))]), qd=np.zeros((2, 7)),
                               tip=tip, phase=np.full(2, 2, np.uint8))

    monkeypatch.setattr(ex_mod.ready, "solve_ik", lambda kin, target, q: np.concatenate([target[:3, 3], np.zeros(4)]))
    monkeypatch.setattr(ex_mod.ready, "joint_move", joint_move)
    out = ex.touch_single(start, np.array([1.0, 0.0, 0.0]), 0.0, 0.0, guard=2, prior_distance_m=0.006)
    assert np.allclose(moves[0][:3], [0.30, 0.0, 0.13])   # over the start at the clearance plane
    # the joint move and the touch under the probe guard, then the untriggered touch's back-off
    assert io.writes[0] == {"guard_mode": 2} and len(io.executed) == 3 and setup.lifts == [0.003]
    assert seen[0]["via"] == [] and not out["tripped"]
    # a joint path that dips under its lower end is refused before any motion
    io.q = np.zeros(7)
    io.executed.clear()
    monkeypatch.setattr(ex_mod.ready, "joint_move", lambda *a, **kw: SimpleNamespace(
        t=np.array([0.0, 1.0, 2.0]), tip=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, -0.02], [0.3, 0.0, 0.13]])))
    with pytest.raises(RuntimeError, match="down to z"):
        ex.touch_single(start, np.array([1.0, 0.0, 0.0]), 0.0, 0.0, guard=2, prior_distance_m=0.006)
    assert io.executed == []


def test_a_station_move_takes_the_clearance_plane_under_the_probe_guard_and_a_trip_is_an_error(setup, monkeypatch):
    ex, io = setup.ex, setup.io
    seen = []

    def plan_travel(*, q_seed, base_from_tcp, kin, motion, via=None):
        seen.append([np.round(v, 6).tolist() for v in via or []])
        tip = np.array([kin.fk(q_seed)[:3, 3], base_from_tcp[:3, 3]])
        return SimpleNamespace(t=np.array([0.0, 1.0]), q=np.column_stack([tip, np.zeros((2, 4))]), qd=np.zeros((2, 7)),
                               tip=tip, phase=np.full(2, 2, np.uint8))

    monkeypatch.setattr(tatbot_motion, "plan_travel", plan_travel, raising=False)
    target = np.eye(4)
    target[:3, 3] = [0.30, 0.0, 0.10]
    out = ex.move_single(target, 2)
    assert seen == [[[0.0, 0.0, 0.13], [0.3, 0.0, 0.13]]] and io.writes[0] == {"guard_mode": 2}
    assert np.allclose(out["q"][:3], [0.30, 0.0, 0.10]) and not out["tripped"]
    io.trip = np.zeros(7)
    io.script = [("latched", 1, {"latched": 1.0, "latch_reason": 7.0, "guard_tripped": 1.0, "probe_triggered": 1.0},
                  None)]
    with pytest.raises(RuntimeError, match="move: tripped"):
        ex.move_single(target, 2)
    assert io.unlatches == 1


def test_an_unguarded_move_the_planner_refuses_goes_in_joint_space_clear_of_the_page(setup, monkeypatch):
    """From rest on the joint limits the Cartesian planner cannot start a view travel (a calibration hold,
    `touch --move`): the move goes in joint space instead, unless its planned tip would come within half the
    standoff of the page, or no page is known to check it against."""
    from tatbot_motion.clik import PlanError

    ex, io = setup.ex, setup.io

    def refuse(**kw):
        raise PlanError("joint right/joint_1 reaches its guarded limit at sample 0")

    monkeypatch.setattr(tatbot_motion, "plan_travel", refuse, raising=False)
    monkeypatch.setattr(ex_mod.ready, "solve_ik_seeded", lambda kin, target, q: np.concatenate([target[:3, 3], np.zeros(4)]))

    def joint_move(joint_names, q_from, q_to, **kw):
        tip = np.array([q_from[:3], [0.15, 0.0, 0.20], q_to[:3]], float)
        return SimpleNamespace(t=np.array([0.0, 1.0, 2.0]), q=np.column_stack([tip, np.zeros((3, 4))]),
                               qd=np.zeros((3, 7)), tip=tip, phase=np.full(3, 2, np.uint8))

    monkeypatch.setattr(ex_mod.ready, "joint_move", joint_move)
    io.q = np.array([0.0, 0.0, 0.25, 0, 0, 0, 0])
    target = np.eye(4)
    target[:3, 3] = [0.30, 0.0, 0.10]
    out = ex.move_single(target)
    assert np.allclose(out["q"][:3], [0.30, 0.0, 0.10]) and len(io.executed) == 1
    assert any("a joint move instead" in text for _, _, text in setup.node.events)
    # a hold the joint path would reach only by dipping to the page (z 0 here) is refused before any motion
    io.executed.clear()
    io.q = np.array([0.0, 0.0, 0.25, 0, 0, 0, 0])
    low = np.eye(4)
    low[:3, 3] = [0.30, 0.0, 0.004]
    with pytest.raises(RuntimeError, match="over the page"):
        ex.move_single(low)
    assert io.executed == []
    # with no page known there is nothing to keep it clear of
    setup.node.camera = None
    setup.node.camera_page = lambda arm: None
    with pytest.raises(RuntimeError, match="no page is known"):
        ex.move_single(target)
    assert io.executed == []
    # a probe move keeps the planner's refusal: its way is the clearance plane, never a joint sweep
    with pytest.raises(PlanError):
        ex.move_single(target, 2)


def test_a_probe_still_triggered_after_the_back_off_is_an_error(setup, monkeypatch):
    ex, io = setup.ex, setup.io
    probe_plan(monkeypatch, [])
    io.trip = np.zeros(7)
    io.script = [("latched", 3, {"latched": 1.0, "latch_reason": 7.0, "guard_tripped": 1.0, "probe_triggered": 1.0},
                  None)]
    with pytest.raises(RuntimeError, match="still reads triggered"):
        ex.touch_single(np.eye(4), np.array([0.0, 0.0, -1.0]), 0.0, 0.0, guard=2, prior_distance_m=0.005)


def _cancel_after(ex, seconds):
    t_end = time.monotonic() + seconds
    ex.cancelled = lambda: time.monotonic() > t_end


def test_a_decide_left_over_from_one_wait_is_never_applied_at_the_next(setup):
    ex, io = setup.ex, setup.io
    io.safety.update(latched=1.0, latch_reason=1.0)
    io.unlatch_delay = 0.3  # the hold goal and the ack take a while
    first = threading.Thread(target=lambda: setattr(ex, "got", ex.wait_decision("latched", "s0007", 7)))
    first.start()
    while ex.waiting is None:
        time.sleep(0.002)
    replies = []
    senders = [threading.Thread(target=lambda: replies.append(ex.decide(rules.CONTINUE, timeout=2.0)))
               for _ in range(2)]
    senders[0].start()
    time.sleep(0.1)  # the second `decide continue` arrives while the first is unlatching
    senders[1].start()
    for thread in (first, *senders):
        thread.join(5.0)
    assert ex.got == rules.CONTINUE and sorted(ok for ok, _ in replies) == [False, True]
    assert "nothing waits" in next(why for ok, why in replies if not ok)
    assert ex.decisions.empty() and io.unlatches == 1
    # a later carriage-contact latch waits for its own decision
    io.safety.update(latched=1.0, latch_reason=3.0)
    _cancel_after(ex, 0.3)
    with pytest.raises(ex_mod.Cancelled):
        ex.wait_decision("latched", "s0008", 8)
    assert io.unlatches == 1 and io.latched


def test_a_decide_its_caller_gave_up_on_is_never_applied(setup):
    ex, io = setup.ex, setup.io
    io.safety.update(latched=1.0, latch_reason=3.0)
    ex.waiting = "latched"  # an executor busy elsewhere: nobody reads the queue
    assert ex.decide(rules.CONTINUE, timeout=0.1) == (False, "the executor did not answer")
    ex.waiting = None
    _cancel_after(ex, 0.3)
    with pytest.raises(ex_mod.Cancelled):
        ex.wait_decision("latched", "s0007", 7)
    assert io.unlatches == 0 and ex.decisions.empty()


def test_land_accepted_while_a_goal_runs_lands_when_the_goal_ends(setup):
    ex, io = setup.ex, setup.io
    assert ex.begin_goal() and not ex.begin_goal()
    assert ex.decide(rules.LAND) == (True, "landing after the current goal stops")
    ex.end_goal()  # the goal finished before it saw the request
    assert io.lands == 1 and not ex.land_requested and not ex.busy.locked()
    assert ex.begin_goal() and not ex.land_requested
    ex.end_goal()
    assert io.lands == 1


class GoalGate:
    """SessionNode's goal acceptance and arm taking, on the executors alone: no ROS graph."""

    def __init__(self, ex):
        from tatbot_session.node import SessionNode

        self.executors, self.stack, self.events = {ex.arm: ex}, {"arms": [ex.arm]}, []
        for name in ("_goal_arms", "_goal_ok", "_acquire"):
            setattr(self, name, MethodType(getattr(SessionNode, name), self))

    def event(self, kind, *, arm="", op_id="", text=""):
        self.events.append((kind, arm, text))


def test_a_landed_arm_refuses_every_goal_until_it_is_woken(setup):
    """The driver keeps a landed arm idle until it is woken while every move sent to it still succeeded, the
    arm at rest (2026-09-29). A Draw or Touch goal for it is refused at acceptance, one accepted before the landing
    when it takes the arm, and each says how to wake it."""
    from rclpy.action import GoalResponse
    from tatbot_interfaces.action import Draw, Touch

    ex, io = setup.ex, setup.io
    gate = GoalGate(ex)
    assert gate._goal_ok(Touch.Goal(arm="right")) == GoalResponse.ACCEPT and ex.refusal() is None
    assert ex.decide(rules.LAND) == (True, "landed and idle") and io.flag("landed")
    assert gate._goal_ok(Touch.Goal(arm="right")) == gate._goal_ok(Draw.Goal()) == GoalResponse.REJECT
    assert gate.events[-1][:2] == ("error", "right") and "client wake --arm right" in gate.events[-1][2]
    with pytest.raises(RuntimeError, match=r"the right arm has landed .*client wake --arm right"):
        gate._acquire(["right"])
    assert not ex.busy.locked() and io.executed == [] and io.lands == 1


def test_a_way_too_near_the_other_arm_is_not_sent():
    """ArmIO.execute asks the node's collision guard (tatbot_motion.collision.Guard) before anything reaches JTC: a
    refusal is a failed goal carrying the guard's reason (2026-09-29: at a pink view the pink wrist's cables caught on the blue arm)."""
    from tatbot_session.arm import ArmIO

    warned, asked = [], []
    arm = _io_stub(clearance=lambda a, q: f"{a} {len(q)} rows too near", warned=warned,
                   jtc=SimpleNamespace(wait_for_server=lambda timeout_sec: asked.append(timeout_sec)))
    traj = SimpleNamespace(q=np.zeros((2, 7)), t=np.array([0.0, 1.0]))
    assert ArmIO.execute(arm, traj) == ("failed", 0, "right 2 rows too near")
    assert warned == ["right 2 rows too near"] and asked == []


def _io_stub(clearance, warned=None, jtc=None, run=None, released=None):
    from tatbot_session.arm import ArmIO

    arm = SimpleNamespace(arm="right", flag=lambda name: False, clearance=clearance, latched=False,
                          node=SimpleNamespace(get_logger=lambda: SimpleNamespace(
                              warning=(warned if warned is not None else []).append)),
                          jtc=jtc, release=(released.append if released is not None else None))
    arm._stop = lambda watch, stop: ArmIO._stop(arm, watch, stop)
    arm.terminal_refusal = lambda: ArmIO.terminal_refusal(arm)
    arm._way_clear = lambda traj, watch, stop: ArmIO._way_clear(arm, traj, watch, stop)
    arm._run = run or (lambda *a: ("ok", 1, "done"))
    return arm


def test_a_way_blocked_by_the_other_arms_goal_in_flight_waits_for_it_and_is_released(monkeypatch):
    """2026-09-30: two arms move at once in one stack. The node's guard reserves each goal's way; one blocked only
    by the other arm's way in flight waits for it to end (it ends), a standing refusal does not wait, and a way let
    through is released when its goal ends, whatever the outcome."""
    from tatbot_session import arm as arm_module
    from tatbot_session.arm import ArmIO

    monkeypatch.setattr(arm_module.time, "sleep", lambda s: None)
    answers = iter([("the left arm's goal in flight; waiting", True)] * 3 + [(None, False)])
    released = []
    io = _io_stub(clearance=lambda a, q: next(answers), released=released)
    traj = SimpleNamespace(q=np.zeros((2, 7)), t=np.array([0.0, 1.0]))
    assert ArmIO.execute(io, traj) == ("ok", 1, "done") and released == ["right"]
    asked = []
    io = _io_stub(clearance=lambda a, q: asked.append(a) or ("the left arm stands there; not sent", False),
                  released=released)
    assert ArmIO.execute(io, traj)[0] == "failed" and asked == ["right"] and released == ["right"]

    def boom(*a):
        raise RuntimeError("the goal failed")

    io = _io_stub(clearance=lambda a, q: (None, False), run=boom, released=released)
    with pytest.raises(RuntimeError):
        ArmIO.execute(io, traj)
    assert released == ["right", "right"]


def test_a_landing_is_the_drivers_joint_sweep_to_the_staged_then_the_sleep_pose():
    from tatbot_session.arm import landing_way

    q = np.array([0.5, 1.0, 0.8, -0.3, 0.2, 1.9, 0.02])
    staged = np.array([0.0, 0.3, 0.3, 0.0, 0.0, 1.5708, 0.0])
    rows = landing_way(q, staged, n=5)
    assert np.allclose(rows[0], q) and np.allclose(rows[4], [*staged[:6], q[6]])
    assert np.allclose(rows[-1], [0, 0, 0, 0, 0, 1.5708, 0.0])


def test_a_latch_in_the_final_lift_finishes_the_stroke(setup, monkeypatch):
    ex, io = setup.ex, setup.io
    calls = []

    def plan_op(op, **kw):
        calls.append(op["id"])
        traj = line_traj(op)
        lift = np.column_stack([np.full(3, 0.02), np.zeros(3), [0.001, 0.004, 0.010], np.zeros((3, 4))])
        return stroke_plan(np.linspace(0, 1.3, 14), np.vstack([traj.tip, lift[:, :3]]),
                           np.append(traj.arc_m, [np.nan] * 3), np.append(traj.phase, [6, 6, 6]))

    monkeypatch.setattr(tatbot_motion, "plan_op", plan_op, raising=False)
    io.script = [("latched", 11, {"latched": 1.0, "latch_reason": 1.0}, [0.02, 0.0, 0.001, 0, 0, 0, 0])]
    operator = Operator(ex, rules.CONTINUE)
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert operator.replies() == [(True, "")]
    rows = setup.ledger.rows()
    assert [r["event"] for r in rows] == ["sent", "done", "decision"] and rows[1]["arc_m"] == pytest.approx(0.02)
    assert calls == ["s0000"] and setup.lifts == [pytest.approx(0.010)]  # lifted, not re-planned or re-drawn


@pytest.mark.parametrize("age", [None, 3600.0])
def test_a_draw_refuses_a_page_lost_past_max_lost_s(setup, age):
    """A lost page stands for its last measured pose; an hour-old one sent a tiger ~17 mm off the print."""
    setup.node.page_info = {"source": "lost", "pattern_id": "stencil-x", "measured_age_s": age}
    with pytest.raises(RuntimeError, match="give the overhead cameras a clear view"):
        setup.ex.setup_page({"ops": [LINE]}, True)
    assert setup.io.script == [] and setup.io.unlatches == 0   # refused before any motion


def test_a_draw_waits_for_the_cameras_to_measure_a_page_the_arm_uncovered(setup):
    """The second side of a held pair found the coded print lost at once; it was measured 3.3 min later."""
    setup.ex.stack = {**setup.ex.stack, "page": {**setup.ex.stack["page"], "wait_s": 240.0}}
    setup.node.page_info = {"source": "lost", "pattern_id": "stencil-x", "measured_age_s": 618.0}
    setup.node.later_info = {"source": "measured", "pattern_id": "stencil-x", "measured_age_s": 0.0}
    setup.ex.setup_page({"ops": [LINE]}, False)
    assert setup.node.waits == [10.0, 230.0]
    assert any("waiting up to 230 s" in text for _, _, text in setup.node.events)


def test_a_lost_page_is_waited_for_from_rest_so_the_arm_does_not_hide_it(setup):
    """After a cancel the arm held over the print and the anchoring camera never saw it again (2026-09-28)."""
    from tatbot_session import inspect as ins

    setup.ex.stack = {**setup.ex.stack, "page": {**setup.ex.stack["page"], "wait_s": 240.0}}
    setup.node.page_info = {"source": "lost", "pattern_id": "stencil-x", "measured_age_s": 60.0}
    setup.node.later_info = {"source": "measured", "pattern_id": "stencil-x", "measured_age_s": 0.0}
    setup.io.q = np.array([0.02, 0.01, 0.0, 0.3, 0.1, 1.2, 0.004])      # stopped over the page
    setup.ex.setup_page({"ops": [LINE]}, False)
    np.testing.assert_allclose(setup.io.q[:6], ins.REST[:6])            # moved to rest, holding
    assert setup.io.q[6] == pytest.approx(0.004) and setup.io.lands == 0  # carriage kept; never landed
    moves = len(setup.io.writes)
    setup.ex.setup_page({"ops": [LINE]}, False)                         # already at rest: no motion
    assert len(setup.io.writes) == moves and np.allclose(setup.io.q[:6], ins.REST[:6])


def test_the_pen_lifts_first_and_a_rest_path_that_dips_is_not_taken(setup):
    setup.ex.stack = {**setup.ex.stack, "page": {**setup.ex.stack["page"], "wait_s": 240.0}}
    setup.node.page_info = {"source": "lost", "pattern_id": "stencil-x", "measured_age_s": 60.0}
    setup.ex.camera = np.eye(4)                                          # the page this goal drew on
    setup.io.q = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])         # pen on the paper
    with pytest.raises(RuntimeError, match="give the overhead cameras a clear view"):
        setup.ex.setup_page({"ops": [LINE]}, False)
    assert setup.lifts == [pytest.approx(0.010)]                         # lifted along the page normal
    assert any("not moving" in text for _, _, text in setup.node.events)  # the fake FK path meets the page


def test_a_page_still_lost_after_the_wait_is_refused_and_a_cancel_stops_the_wait(setup):
    setup.ex.stack = {**setup.ex.stack, "page": {**setup.ex.stack["page"], "wait_s": 240.0}}
    setup.node.page_info = {"source": "lost", "pattern_id": "stencil-x", "measured_age_s": 618.0}
    with pytest.raises(RuntimeError, match="give the overhead cameras a clear view"):
        setup.ex.setup_page({"ops": [LINE]}, True)
    setup.ex.cancelled = lambda: True
    with pytest.raises(ex_mod.Cancelled):
        setup.ex.setup_page({"ops": [LINE]}, True)
    assert setup.io.script == [] and setup.io.unlatches == 0   # neither moved the arm


@pytest.mark.parametrize("following", ["end", "stroke", "pause", "skipped"])
def test_compiled_continuations_through_real_cartesian_planner(setup, monkeypatch, tmp_path, following):
    """Compiler -> executor/preplanner -> real leg builder; only IK/driver are substituted."""
    import json
    from pathlib import Path

    from tatbot_ink import compile
    from tatbot_motion import plan as planner

    repo = Path(__file__).resolve().parents[3]
    from tatbot_contracts.artwork import freeze_artwork
    from tatbot_contracts.paths import freeze_program

    geometry = {"canvas_m": {"width": .02, "height": .02}, "negative_space_masks": [],
                "inks": [{"id": "pen", "color_srgb": [0, 0, 0]}],
                "layers": [{"id": "layer", "ink_id": "pen", "elements": [{"id": "path", "kind": "path",
                "closed": False, "fill": False, "width_m": .0005, "deposition": 1,
                "points_m": [[.002, .01], [.018, .01]]}]}]}
    native, _ = freeze_program(geometry, source_sha256="a" * 64, name="continuation fixture", adapter="dbv3-batik-paths/1")
    art = freeze_artwork(native, name="continuation fixture", source={"kind": "fixture", "identifier": "line",
                        "license": None, "attribution": None, "generation": None},
                        conversion={"adapter": "dbv3-batik-paths/1", "recipe_sha256": "b" * 64, "chord_error_m": .000005})
    path = tmp_path / "line.json"
    path.write_text(json.dumps(art))
    program = compile(path, repo=repo, max_segment_s=2, tool_id="lutin-ballpoint-dot")
    program["ops"] = [op for op in program["ops"] if op["op"] == "stroke"]
    last_chunk = len(program["ops"]) - 1
    assert last_chunk > 0
    assert [op["continues"] for op in program["ops"]] == [False] + [True] * last_chunk
    if following == "stroke":
        program["ops"].append(dict(LINE2, id="independent", continues=False))
    elif following == "pause":
        program["ops"].append({"op": "pause", "id": "pause"})
    elif following == "skipped":
        setup.ledger.append("skipped", "right", op=program["ops"][-1]["id"])
    observed = []

    def capture(build, q_seed, kin, motion):
        legs = build({})
        tips = np.concatenate([leg.p[[0, -1]] for leg in legs])
        phases = np.concatenate([planner._rows(leg.phase, len(leg.p))[[0, -1]] for leg in legs])
        arc = np.concatenate([leg.arc[[0, -1]] if leg.arc is not None else [np.nan, np.nan] for leg in legs])
        traj = stroke_plan(np.arange(len(tips)) * .01, tips, arc, phases)
        observed.append(traj)
        return traj

    monkeypatch.setattr(planner, "_solve", capture)
    monkeypatch.setattr(tatbot_motion, "plan_op", planner.plan_op)
    ex = setup.ex
    ex.motion = tatbot_motion.load_motion()
    ex.repo, ex.machine = repo, Switch()
    ex.prepare_resource(program['resources'][0])
    ex.camera = np.eye(4)
    # Exercise the same dispatch and speculative preplan path used by stroke().
    for index, op in enumerate(program["ops"]):
        if op["op"] != "stroke" or (following == "skipped" and index == last_chunk):
            continue
        traj, _ = ex.dispatch_plan(program, op, 0, setup.ledger.rows())
        setup.io.q = traj.q[-1]
        ex.preplan(program, index, setup.io.q, setup.ledger.rows())
        setup.ledger.append("done", "right", op=op["id"])
        expected_lift = index == (last_chunk - 1 if following == "skipped" else last_chunk) or op["id"] == "independent"
        assert (traj.phase[-1] == planner.PHASE_LIFT) == expected_lift
        assert traj.info["joined"] == (0 < index <= last_chunk)
    if following != "skipped":
        final = program["ops"][last_chunk]
        # A partial resume must still include the final lift.
        resumed = ex.plan(program, final, .001, True, setup.io.q,
                          lift_at_end=ex.lift_at_end(program, final, setup.ledger.rows()))
        assert resumed.phase[-1] == planner.PHASE_LIFT


class DipKin(FakeKin):
    """The tip is q[:3], except that a move of q[3] between 0 and 1 dips it by up to 20 mm midway: a
    joint path whose ends are both clear of the page can pass through it."""

    def fk(self, q):
        out = super().fk(q)
        out[2, 3] -= 0.08 * q[3] * (1 - q[3])
        return out


def _ready_executor(tmp_path, monkeypatch, *, high_q3):
    lifts = []

    def plan_lift(*, q_seed, direction, distance_m, kin, motion):
        lifts.append(distance_m)
        q = np.array([q_seed, q_seed], float)
        q[1, :3] += np.asarray(direction) * distance_m
        tip = np.array([kin.fk(row)[:3, 3] for row in q])
        return SimpleNamespace(t=np.array([0.0, 1.0]), q=q, qd=np.zeros((2, 7)), tip=tip,
                               arc_m=np.full(2, np.nan), phase=np.full(2, 6, np.uint8))

    def solve_ik(kin, target, q_seed):
        q = np.zeros(7)
        q[:3] = np.asarray(target)[:3, 3]
        q[3] = 1.0 if q[2] < 0.03 else high_q3   # the ready pose sits at q3 = 1, the view at q3 = 0
        return q

    monkeypatch.setattr(tatbot_motion, "plan_lift", plan_lift, raising=False)
    monkeypatch.setattr(tatbot_motion, "to_knots", lambda traj, rate: traj, raising=False)
    monkeypatch.setattr(ex_mod.ready, "solve_ik", solve_ik)
    io, node = FakeIO(), FakeNode(np.eye(4))
    ex = ex_mod.ArmExecutor(node, io, DipKin(), MOTION, STACK, tmp_path)
    ex.camera, ex.correction = np.eye(4), np.eye(4)   # the page is the base plane, z up
    ex.pen_heading = lambda: np.eye(4)
    run = []
    real = io.execute

    def execute(traj, **kw):
        run.append(np.asarray(traj.tip, float).copy())
        return real(traj, **kw)

    io.execute = execute
    io.q = np.array([0.02, 0.0, 0.015, 0.0, 0, 0, 0])   # a locate view: the pen 15 mm up, q3 = 0
    return ex, run, lifts, node


def test_the_ready_move_goes_over_the_page_when_the_direct_move_would_dip(tmp_path, monkeypatch):
    ex, run, lifts, node = _ready_executor(tmp_path, monkeypatch, high_q3=0.0)
    direct = ex_mod.ready.joint_move(ex.kin.joint_names, ex.io.q, ex_mod.ready.solve_ik(ex.kin, geometry.tool_down_pose(
        ex.used(), [0.0, 0.0], 0.010, np.eye(4)), ex.io.q), max_rad_s=1.0, max_m_s=0.001, rate_hz=100, kin=ex.kin)
    assert ex.lowest_over_page(direct) < 0.0   # the direct move would put the tip under the page
    ex.go_ready()
    assert lifts and len(run) == 3             # the lift, the tool high over the centre, then down to the standoff
    assert min(float(tip[:, 2].min()) for tip in run) >= 0.005
    assert np.allclose(run[-1][-1], [0.0, 0.0, 0.010], atol=1e-9)
    assert any("over the page centre" in text for _, _, text in node.events)


def test_the_ready_move_refuses_a_path_that_dips_anyway(tmp_path, monkeypatch):
    ex, run, lifts, _ = _ready_executor(tmp_path, monkeypatch, high_q3=-2.0)   # the high pose dips on the way too
    with pytest.raises(RuntimeError, match="not moving"):
        ex.go_ready()
    assert all(float(tip[:, 2].min()) >= 0.005 for tip in run)   # whatever ran stayed clear


# --- resources: a cartridge swap lands for the operator and the resume keeps the page; dips ----------------------
def _resource(rid, ink, *, dip=False):
    return {"id": rid, "ink_id": ink, "pen_mode": "press", "tool": {"id": "lutin-ballpoint-dot"},
            "slot": "inkcap_medium_1" if dip else None, "dip": {"mm_per_dip": 30.0} if dip else None}


def _program(*ops, resources):
    return {"arm": "right", "resources": resources, "ops": list(ops)}


@pytest.fixture
def drawing(setup, monkeypatch, tmp_path):
    from tatbot_session import lease

    monkeypatch.setattr(ex_mod.config, "workspace_arm", lambda repo, arm: {"tool_id": "lutin-ballpoint-dot"})
    monkeypatch.setattr(lease, "IN_CAP_DIR", tmp_path / "state")
    setup.ex.run_dir = tmp_path / "run-0001"
    setup.dips = []
    from tatbot_session import cap

    monkeypatch.setattr(cap, "dip", lambda ex, resource, op_id, index, ledger: setup.dips.append(op_id) or True)
    return setup


def _stroke(sid, rid):
    return {**LINE, "id": sid, "resource_id": rid}


def test_a_cartridge_swap_lands_and_the_resumed_run_keeps_its_page(drawing):
    ex, io, node = drawing.ex, drawing.io, drawing.node
    program = _program({"op": "tool_change", "id": "t0", "resource_id": "blue", "initial": True}, _stroke("s0", "blue"),
                       {"op": "tool_change", "id": "t1", "resource_id": "purple"}, _stroke("s1", "purple"),
                       resources=[_resource("blue", "sky_blue"), _resource("purple", "purple")])
    with pytest.raises(ex_mod.Landed, match="fit purple .*--resume run-0001"):
        ex.draw(program, drawing.ledger, touch=False)
    assert io.lands == 1 and [r["op"] for r in drawing.ledger.rows() if r["event"] == "done"] == ["t0", "s0"]
    io.safety.update(landed=0.0)   # woken
    waits = len(node.waits)
    ex.motion = {**MOTION, "pen": {**MOTION["pen"], "press": {"lift_m": -0.001, "settle_s": 1.0}}}   # a retuned motion.yaml
    assert ex.draw(program, drawing.ledger, touch=False) == "complete"
    assert drawing.calls[-1]["pen"].height_m == pytest.approx(-0.001)   # the pen height is motion.yaml's, not the page's
    rows = drawing.ledger.rows()
    assert [(r["event"], r.get("op")) for r in rows if r["event"] in ("page", "sent", "done")][-3:] == [
        ("done", "t1"), ("sent", "s1"), ("done", "s1")]
    assert sum(r["event"] == "page" for r in rows) == 1 and len(node.waits) == waits   # no new page setup
    assert ex.resource["id"] == "purple" and drawing.dips == []


def test_a_trim_mid_stroke_replaces_the_goal_ahead_and_a_resume_keeps_each_cartridges(drawing):
    """Keys during a stroke replace the running goal from pen.trim.lead_s ahead (unchanged before that, one change
    easing in at a time); the next stroke starts at the trim; after a swap the resume keeps each cartridge's."""
    ex, io, events = drawing.ex, drawing.io, drawing.node.events
    program = _program({"op": "tool_change", "id": "t0", "resource_id": "blue", "initial": True},
                       _stroke("s0", "blue"), _stroke("s1", "blue"),
                       {"op": "tool_change", "id": "t1", "resource_id": "purple"}, _stroke("s2", "purple"),
                       resources=[_resource("blue", "sky_blue"), _resource("purple", "purple")])
    io.at_tick = {2: lambda: ex.on_key("down"), 3: lambda: ex.on_key("down")}
    with pytest.raises(ex_mod.Landed):
        ex.draw(program, drawing.ledger, touch=False)
    first, (at, eased), (later, settled) = io.executed[0], *io.replaced
    assert (at, later) == (pytest.approx(0.2), pytest.approx(0.5))   # the second waited for the first to ease in
    ahead = first.t < at + MOTION["pen"]["trim"]["lead_s"] - 1e-9
    np.testing.assert_array_equal(eased.q[ahead], first.q[ahead])
    assert settled.q[-1, 2] - first.q[-1, 2] == pytest.approx(-0.0002) and ex.trims == {"blue": pytest.approx(-0.0002)}
    assert io.executed[1].info["trim"] == tatbot_motion.Trim(pytest.approx(-0.0002))   # s1 starts there
    io.safety.update(landed=0.0)   # the cartridge is swapped and the arm woken
    io.at_tick = {1: lambda: ex.on_key("up")}
    assert ex.draw(program, drawing.ledger, touch=False) == "complete"
    assert any(text == "trims kept from the run: blue -0.2 mm" for _, _, text in events)
    assert io.executed[-1].info["trim"].start_m == 0.0 and ex.trims == {"blue": pytest.approx(-0.0002),
                                                                       "purple": pytest.approx(0.0001)}


def test_a_height_ladder_changes_resources_of_one_ink_without_landing(drawing):
    """A tool change to another resource of the same tool and ink is another ride height, not a swap: it goes on
    from the same page, and each resource takes its own fraction of the stroke (motion.yaml's without one)."""
    ex, io = drawing.ex, drawing.io
    rungs = [dict(_resource(f"h{i}", "sky_blue"), ride_fraction=f) for i, f in enumerate((0.9, 1.1))]
    program = _program({"op": "tool_change", "id": "t0", "resource_id": "h0", "initial": True}, _stroke("s0", "h0"),
                       {"op": "tool_change", "id": "t1", "resource_id": "h1"}, _stroke("s1", "h1"),
                       resources=rungs)
    fractions, prepare_pen = [], ex.prepare_pen
    ex.prepare_pen = lambda program: fractions.append(ex.motion["pen"]["ride"]["fraction"]) or prepare_pen(program)
    assert ex.draw(program, drawing.ledger, touch=False) == "complete"
    assert io.lands == 0 and fractions == [0.9, 1.1] and ex.resource["id"] == "h1"
    ex.prepare_resource(_resource("plain", "sky_blue"))
    assert fractions[-1] == MOTION["pen"]["ride"]["fraction"]


def test_a_rinse_changes_ink_in_a_cap_of_water_without_landing_and_a_resume_rinses_again(drawing, monkeypatch):
    """A resource activated by a rinse takes the cartridge from the last ink in its cap of water: one dip there with
    the rinse's dwell and the tube's end at the water, then the run goes on with no landing. A rinse left `sent` by an
    interruption is not the operator's swap: the resumed run rinses again."""
    from tatbot_session import cap

    ex, io = drawing.ex, drawing.io
    rinse = {"method": "rinse", "slot": "inkcap_large_1", "ink_id": "water", "dwell_s": 10.0, "above_ink_m": 0.0}
    white = {**_resource("white", "snow_white", dip=True), "dip": {"mm_per_dip": 30.0, "dwell_s": 1.5}}
    red = {**white, "id": "red", "ink_id": "bright_red", "slot": "inkcap_medium_2", "activation": rinse}
    dips = []
    monkeypatch.setattr(cap, "dip", lambda ex, resource, op_id, index, ledger: dips.append(
        (op_id, resource["slot"], resource["ink_id"], resource["dip"]["dwell_s"], resource["dip"].get("above_ink_m")))
        or True)
    program = _program({"op": "tool_change", "id": "t0", "resource_id": "white", "initial": True},
                       {"op": "dip", "id": "d1", "resource_id": "white", "slot": "inkcap_medium_1"},
                       _stroke("s2", "white"),
                       {"op": "tool_change", "id": "t3", "resource_id": "red", "activation": rinse},
                       {"op": "dip", "id": "d4", "resource_id": "red", "slot": "inkcap_medium_2"},
                       _stroke("s5", "red"), resources=[white, red])
    assert ex.draw(program, drawing.ledger, touch=False) == "complete"
    assert io.lands == 0 and ex.resource["id"] == "red"
    assert dips == [("d1", "inkcap_medium_1", "snow_white", 1.5, None),
                    ("t3", "inkcap_large_1", "water", 10.0, 0.0),
                    ("d4", "inkcap_medium_2", "bright_red", 1.5, None)]
    rows = drawing.ledger.rows()
    t3 = next(r for r in rows if r.get("op") == "t3" and r["event"] == "sent")
    cut = rows[:rows.index(t3) + 1]   # the rinse was interrupted: its change is left `sent`
    assert ex._installed(program, cut, 0)["id"] == "white"


def test_a_new_tool_calibration_sets_the_page_up_again_on_resume(drawing, monkeypatch):
    ex = drawing.ex
    program = _program({"op": "tool_change", "id": "t0", "resource_id": "blue", "initial": True}, _stroke("s0", "blue"),
                       resources=[_resource("blue", "sky_blue")])
    drawing.io.script = [("failed", 4, {}, None)]
    operator = Operator(ex, rules.LAND)
    with pytest.raises(ex_mod.Landed):
        ex.draw(program, drawing.ledger, touch=False)
    assert operator.replies() == [(True, "")]
    monkeypatch.setattr(ex_mod.config, "workspace_arm", lambda repo, arm: {"tool_id": "lutin-ballpoint-dot", "tip": 2})
    drawing.io.safety.update(landed=0.0)
    assert ex.draw(program, drawing.ledger, touch=False) == "complete"
    assert sum(r["event"] == "page" for r in drawing.ledger.rows()) == 2


def test_a_dipping_resource_dips_before_its_first_stroke_in_every_goal(drawing):
    ex = drawing.ex
    program = _program({"op": "tool_change", "id": "t0", "resource_id": "red", "initial": True},
                       {"op": "dip", "id": "d0", "resource_id": "red", "slot": "inkcap_medium_1"},
                       _stroke("s0", "red"), _stroke("s1", "red"),
                       resources=[_resource("red", "red", dip=True)])
    drawing.io.script = [("ok", 10, {}, None), ("failed", 4, {}, None)]
    operator = Operator(ex, rules.LAND)
    with pytest.raises(ex_mod.Landed):
        ex.draw(program, drawing.ledger, touch=False)
    assert operator.replies() == [(True, "")] and drawing.dips == ["d0"]
    drawing.io.safety.update(landed=0.0)
    assert ex.draw(program, drawing.ledger, touch=False) == "complete"
    assert drawing.dips == ["d0", "s1"]   # the resumed goal dips again before s1


def test_a_tool_may_be_in_a_cap_refuses_landing_and_the_next_draw_withdraws_first(drawing, monkeypatch):
    from tatbot_session import cap, lease

    ex, io = drawing.ex, drawing.io
    lease.mark_in_cap("right", {"run": "run-0001", "op": "d0", "slot": "inkcap_medium_1", "resource": {}})
    ex.camera = np.eye(4)
    assert "may be in cap inkcap_medium_1" in ex._land_now()[1] and io.lands == 0
    withdrawn = []
    monkeypatch.setattr(cap, "withdraw_after_restart",
                        lambda ex, program, marker, ledger: withdrawn.append(marker["op"]) or lease.clear_in_cap("right"))
    program = _program({"op": "tool_change", "id": "t0", "resource_id": "blue", "initial": True}, _stroke("s0", "blue"),
                       resources=[_resource("blue", "sky_blue")])
    assert ex.draw(program, drawing.ledger, touch=False) == "complete"
    assert withdrawn == ["d0"] and lease.in_cap("right") is None


def test_a_from_op_past_a_tool_change_is_refused(drawing):
    program = _program({"op": "tool_change", "id": "t0", "resource_id": "blue", "initial": True}, _stroke("s0", "blue"),
                       {"op": "tool_change", "id": "t1", "resource_id": "purple"}, _stroke("s1", "purple"),
                       resources=[_resource("blue", "sky_blue"), _resource("purple", "purple")])
    with pytest.raises(RuntimeError, match="tool change was skipped"):
        drawing.ex.draw(program, drawing.ledger, from_op="s1", touch=False)


def test_the_gauge_plane_drops_a_point_off_the_rest_and_refuses_a_wild_tilt(setup):
    """2026-10-03: one gauge point read the paper 4 mm high (its frames 1.6 mm apart) and the fused height took it
    in. Of four points, one off the others' plane by more than gauge_residual_m is dropped; a plane tilting more
    than gauge_max_tilt_deg from the overhead's is not believed."""
    ex = setup.ex
    ex.page_record = {"touches": []}
    up = np.array([0.0, 0.0, 1.0])
    tilt = geometry.rpy_matrix([0, 0, 0], [0.0, np.radians(1.0), 0.0])[:3, 2]   # a 1 deg page
    pts = [np.array([x, y, 0.005 - (tilt[0] * x + tilt[1] * y) / tilt[2]]) for x, y in
           ((0.31, -0.01), (0.33, -0.01), (0.33, 0.03), (0.31, 0.03), (0.32, 0.01))]
    pts[2][2] += 0.004                                                          # the bad one, at a corner
    normal, d = ex._gauge_plane(pts, up, {})
    assert ex.page_record["height"]["used"] == [True, True, False, True, True]
    assert np.degrees(np.arccos(normal @ tilt)) < 0.01 and abs(pts[4] @ normal + d) < 1e-9
    assert ex._gauge_plane(pts[:4], up, {}) is None                            # four cannot say which one
    steep = geometry.rpy_matrix([0, 0, 0], [np.radians(10.0), 0.0, 0.0])[:3, 2]
    wild = [np.array([x, y, -(steep[0] * x + steep[1] * y) / steep[2]]) for x, y in ((0.31, -0.01), (0.33, -0.01), (0.32, 0.03))]
    assert ex._gauge_plane(wild, up, {}) is None


def test_the_gauge_measures_a_small_drawing_on_a_wide_base(setup, monkeypatch):
    ex = setup.ex
    ex.stack = {**ex.stack, "touch": {"spread_m": 0.020}}
    program = {"ops": [{"op": "stroke", "points_m": [[-0.018, 0.029], [-0.010, 0.029]]},
                       {"op": "stroke", "points_m": [[-0.018, 0.044], [-0.010, 0.044]]}]}
    assert len(ex._touch_layout(program)) == 3                                  # touches: under the drawing
    monkeypatch.setattr(ex, "_gauge_ready", lambda: True)
    layout = ex._touch_layout(program)
    assert np.allclose(layout, [[-0.026, 0.0165], [0.006, 0.0165], [0.006, 0.051], [-0.026, 0.051], [-0.014, 0.0365]])


def test_a_resumed_page_keeps_its_place_and_measures_its_height_again(setup, monkeypatch):
    """A cartridge swap takes minutes and the arm's tip height drifts (~2 mm over an hour, 2026-10-03): a resume
    with the wrist gauge in use measures the retained page's height again; no plane keeps the retained height."""
    ex = setup.ex
    ex.camera = np.eye(4)
    ex.camera[:3, 3] = [0.30, 0.0, 0.0]
    ex.correction = ex.prior_correction = np.eye(4)
    ex.page_record = {"touches": ["old"], "gauge": ["old"], "height": {"method": "retained"}}
    said = []
    monkeypatch.setattr(ex, "event", lambda kind, text, *a: said.append(text))
    monkeypatch.setattr(ex, "_gauge_ready", lambda: True)
    monkeypatch.setattr(ex, "_touch_layout", lambda program: np.zeros((5, 2)))
    risen = [np.array([0.30 + x, y, 0.0012]) for x, y in ((-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01),
                                                         (0.0, 0.0))]
    monkeypatch.setattr(ex, "page_touches", lambda points, ledger, known: risen)
    before = ex.used()
    assert ex._regauge({}, None)
    assert np.allclose(ex.used()[:3, :3], before[:3, :3]) and np.allclose(ex.used()[:2, 3], before[:2, 3])
    assert ex.used()[2, 3] - before[2, 3] == pytest.approx(0.0012)
    assert "+1.20 mm from the retained height" in said[-1]
    assert ex.page_record["retained_height"] == {"method": "retained"}
    ex.page_record = {"touches": ["old"], "gauge": ["old"], "height": {"method": "retained"}}
    monkeypatch.setattr(ex, "page_touches", lambda points, ledger, known: risen[:2])
    kept = ex.used()
    assert not ex._regauge({}, None)
    assert np.allclose(ex.used(), kept) and ex.page_record["touches"] == ["old"]


def test_the_gauge_holds_put_the_page_where_the_tool_lands_on_the_print(setup, monkeypatch):
    """2026-10-06: the locate's far views left a half-skin print 5 mm off in y, and the gauge holds' wrist frames
    saw where the tip stood on it. Their median shift moves the page onto the print; a hold a band off is dropped;
    without them a setup keeps the touched page and a resume the retained page's place."""
    ex = setup.ex
    said = []
    monkeypatch.setattr(ex, "event", lambda kind, text, *a: said.append(text))
    touched = np.eye(4)
    touched[:3, 3] = [0.30, 0.02, 0.003]
    prior = touched.copy()
    prior[2, 3] = 0.006   # the page the holds were made on: another height, the same place
    sent = [(-0.0175, -0.02), (0.0175, -0.02), (0.0175, 0.02), (0.0, 0.0), (-0.0175, 0.02)]
    off = [(-0.0016, -0.0035), (-0.0014, -0.0037), (-0.0015, -0.0036), (-0.0015, -0.0036), (-0.0015, 0.020)]
    ex.page_record = {"gauge": [{"xy": list(xy), "print": {"tip_m": [xy[0] + dx, xy[1] + dy, 0.012]}}
                                for xy, (dx, dy) in zip(sent, off, strict=True)]}
    moved = ex._hold_page(touched, prior)
    assert np.allclose(moved[:3, 3], [0.3015, 0.0236, 0.003]) and np.allclose(moved[:3, :3], np.eye(3))
    assert ex.page_record["holds"]["used"] and ex.page_record["holds"]["dropped"] == 1
    assert "-1.5 -3.6 mm off the print (4 holds" in said[-1]
    ex.page_record = {"gauge": [{"xy": [0.0, 0.0], "print": None}]}
    assert np.allclose(ex._hold_page(touched, prior), touched)
    retained = prior @ geometry.translation([0.0015, 0.0036])
    assert np.allclose(ex._hold_page(touched, retained, keep_prior=True)[:3, 3], [0.3015, 0.0236, 0.003])


def test_a_page_takes_one_ready_branch_whatever_the_arm_held_before(tmp_path):
    """2026-10-03: the flower's setup reached its page from a locate view and its resumes from rest, and the
    heading search's near-tied margins moved with the page's height: joint 6 sat at -0.10, +0.29 and +0.75 rad,
    whose tip heights differ by 2-4 mm. On the real arm model: a new page searches from rest, so where the arm
    started does not matter; then the page's ready pose (recorded, or carried to the next run) holds the heading
    and the branch while its height moves 4 mm either way, where the search itself would flip 45 to 60 deg."""
    pytest.importorskip("pinocchio")
    from tatbot_description import robot_description
    from tatbot_motion.clik import Kinematics
    from tatbot_session import inspect as ins

    kin = Kinematics(robot_description(arms=("right",)), "right")
    page = np.eye(4)
    page[:3, :3] = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]   # the print's x along the base's -y
    page[:3, 3] = [0.296, 0.013, 0.003]
    locate = np.array([-0.5652, 1.597, 0.7494, -0.7185, -0.8246, -0.0429, 0.002])   # the setup's branch

    def ready_pose(start, dz, carried=None):
        io = FakeIO()
        io.q = start.copy()
        ex = ex_mod.ArmExecutor(FakeNode(np.eye(4)), io, kin, MOTION, STACK, tmp_path)
        ex.camera, ex.correction = page.copy(), np.eye(4)
        ex.camera[2, 3] += dz
        ex.page_record = {"carried": {"ready": carried}} if carried else {}
        heading = ex.pen_heading()
        target = geometry.tool_down_pose(ex.used(), [0.0, 0.0], 0.010, heading)
        return {"q": ex_mod.ready.solve_ik_seeded(kin, target, ex._ready_seed()).tolist(), "heading": heading.tolist()}

    first = ready_pose(locate, 0.0)
    again = [ready_pose(ins.REST.copy(), 0.0)] + [ready_pose(s, dz, first) for s in (locate, ins.REST.copy())
                                                  for dz in (-0.004, 0.004)]
    for pose in again:
        np.testing.assert_allclose(pose["heading"], first["heading"], atol=1e-9)
        assert np.max(np.abs(np.subtract(pose["q"], first["q"])[:6])) < 0.02


def test_a_resumed_page_goes_back_to_its_recorded_ready_pose(tmp_path, monkeypatch):
    ex, run, _, node = _ready_executor(tmp_path, monkeypatch, high_q3=1.0)
    seeds = []
    real = ex_mod.ready.solve_ik
    monkeypatch.setattr(ex_mod.ready, "solve_ik", lambda kin, target, q: seeds.append(np.asarray(q).copy())
                        or real(kin, target, q))
    ex.go_ready()
    recorded = ex.page_record["ready"]
    assert np.allclose(recorded["q"], ex.io.q) and np.allclose(recorded["heading"], np.eye(4))
    from tatbot_session import inspect as ins
    assert np.allclose(seeds[0][:6], ins.REST[:6])          # the first ready move solves from rest, not the view
    ex.io.q = ins.REST.copy()                                 # landed for a cartridge swap
    seeds.clear()
    ex.go_ready()
    assert np.allclose(seeds[0][:6], recorded["q"][:6])      # the resume solves from the page's recorded pose
    ex.io.q = np.asarray(recorded["q"]) + [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0]
    ex._record_ready(np.eye(4))
    assert any("another IK branch" in text for _, _, text in node.events)


class StubTracker:
    """Stencil tracking with a known residual field and no camera."""
    correcting = True

    def __init__(self, residual):
        from tatbot_session import track

        self.field, self.snaps, self.lifts, self.on_lift = track.Field(), 0, [], None
        self.hovers, self.verdicts, self.residuals = [], [], []
        self.field.add(0.0, [0.0, 0.0], residual)

    def snapshot(self):
        self.snaps += 1
        return self.field.snapshot()

    def after_lift(self, op):
        self.lifts.append(op["id"])
        if self.on_lift is not None:
            self.on_lift(self)

    def at_hover(self, name, op_id, plan_xy):
        """The hover's verdict from `verdicts` (default 'added'), its residual (if any in `residuals`) set as the
        field's."""
        self.hovers.append((name, op_id, np.asarray(plan_xy, float)))
        if self.residuals:
            self.field.samples = [(time.monotonic(), np.zeros(2), np.asarray(self.residuals.pop(0), float))]
        return self.verdicts.pop(0) if self.verdicts else "added"

    def close(self):
        pass


def test_stencil_tracking_moves_each_stroke_by_the_field_and_a_chain_keeps_one_snapshot(setup):
    ex, io = setup.ex, setup.io
    ex.tracker = StubTracker([0.001, -0.002])
    cont = {"op": "stroke", "id": "s0001", "closed": False, "continues": True, "points_m": [[0.02, 0.0], [0.04, 0.0]]}
    io.script = [("ok", 10, {}, None), ("ok", 10, {}, None)]
    assert draw_strokes(ex, {"ops": [LINE, cont]}, setup.ledger, touch=False) == "complete"
    field = ex.tracker.field.snapshot()
    for call, op in zip(setup.calls, [LINE, cont], strict=True):
        moved, _ = field.predict(op["points_m"])
        assert np.allclose(call["points"], np.asarray(op["points_m"]) - moved)   # the ink lands back on the plan
    assert ex.tracker.snaps == 1                       # the continuation kept its chain's field
    assert ex.tracker.lifts == ["s0001"]               # measured after the lift, not mid-chain
    sent = [row for row in setup.ledger.rows() if row.get("event") == "sent"]
    assert all("correction_m" in row for row in sent)
    assert any(kind == "page" and "s0000 corrected" in text for kind, _, text in setup.node.events)


def test_a_preplan_whose_field_moved_is_planned_again(setup):
    ex, io = setup.ex, setup.io
    ex.tracker = StubTracker([0.001, 0.0])

    def slide(stub):
        stub.field.samples = [(time.monotonic(), np.zeros(2), np.array([0.004, 0.0]))]
    ex.tracker.on_lift = slide
    io.script = [("ok", 10, {}, None), ("ok", 10, {}, None)]
    assert draw_strokes(ex, {"ops": [LINE, LINE2]}, setup.ledger, touch=False) == "complete"
    assert [c["op"] for c in setup.calls] == ["s0000", "s0001", "s0001"]
    assert any(kind == "replan" and "correction moved" in text for kind, _, text in setup.node.events)


def _hovering(ex):
    """Stencil tracking's hover stop on, the arm up at the standoff."""
    import copy

    ex.stack = copy.deepcopy(STACK)
    ex.stack["page"]["track"] = {"hover_stop": True, "second_view_m": 0.005}
    ex.motion = {**MOTION, "approach": {**MOTION["approach"], "final_m": 0.005}}
    ex.io.q = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0])


UP = [0.02, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0]   # a stroke's end, lifted


def test_a_hover_stop_measures_before_the_descent_and_the_stroke_takes_that_field(setup):
    ex, io = setup.ex, setup.io
    ex.tracker = StubTracker([0.001, 0.0])
    _hovering(ex)
    ex.tracker.residuals = [[0.003, 0.0]]               # the first hover finds the sheet slid 2 mm more
    io.script = [("ok", 1, {}, None), ("ok", 10, {}, UP), ("ok", 1, {}, None), ("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE, LINE2]}, setup.ledger, touch=False) == "complete"
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover", "s0001-hover"]   # frames apart from the lifts'
    assert np.allclose(ex.tracker.hovers[0][2], LINE["points_m"][0])
    assert np.allclose(setup.calls[0]["points"], np.asarray(LINE["points_m"]) - [0.003, 0.0])   # not one behind
    assert [getattr(traj, "info", {}).get("hover", False) for traj in io.executed] == [True, False, True, False]
    assert [c["op"] for c in setup.calls] == ["s0000", "s0001"]          # s0001's pre-plan held: no re-plan
    assert np.allclose(setup.calls[1]["seed"], io.executed[2].q[-1])     # planned from its hover
    assert io.executed[2] is not None and len(setup.hovers) == 2         # the pre-planned travel was flown


def test_a_jump_at_the_hover_is_measured_again_from_a_second_view(setup):
    ex, io = setup.ex, setup.io
    ex.tracker = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.tracker.verdicts = ["held", "moved"]
    far = {**LINE, "points_m": [[0.01, 0.0], [0.03, 0.0]]}
    io.script = [("ok", 1, {}, None), ("ok", 1, {}, None), ("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [far]}, setup.ledger, touch=False) == "complete"
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover", "s0000-hover2"]
    assert np.allclose(ex.tracker.hovers[1][2], [0.005, 0.0])            # 5 mm toward the page centre
    assert np.allclose(setup.hovers[1]["start"], [0.005, 0.0])


def test_no_hover_stop_for_a_continuation_or_a_tip_at_the_page(setup):
    ex, io = setup.ex, setup.io
    ex.tracker = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.io.q = np.zeros(7)                                                # the tip at the page
    cont = {"op": "stroke", "id": "s0001", "closed": False, "continues": True, "points_m": [[0.02, 0.0], [0.04, 0.0]]}
    io.script = [("ok", 10, {}, None), ("ok", 10, {}, None)]
    assert draw_strokes(ex, {"ops": [LINE, cont]}, setup.ledger, touch=False) == "complete"
    assert ex.tracker.hovers == [] and setup.hovers == []



def test_a_hover_that_finds_no_print_sets_the_page_up_again_when_the_overhead_saw_the_sheet_go(setup, monkeypatch):
    """A sheet slid past the wrist fit's reach: no stroke goes down blind from the old page."""
    ex, io, node = setup.ex, setup.io, setup.node
    ex.tracker = first = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.stack["hardware"], ex.stack["page"]["track"]["mode"] = "real", "correct"
    first.verdicts = [None]
    moved = np.eye(4)
    moved[1, 3] = 0.05
    node.camera_page = lambda arm: (moved, {"source": "measured"})
    pages, again = [], StubTracker([0.0, 0.0])
    monkeypatch.setattr(ex, "_new_page", lambda program, touch, ledger: pages.append(touch))
    monkeypatch.setattr(ex_mod.track.Tracker, "start", classmethod(lambda cls, ex, since=0.0: again))
    io.script = [("ok", 1, {}, None), ("ok", 1, {}, None), ("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert [h[0] for h in first.hovers] == ["s0000-hover"] and [h[0] for h in again.hovers] == ["s0000-rehover"]
    assert pages == [False] and ex.tracker is again
    assert any("setting the page up again" in text for _, _, text in node.events)


def test_a_hover_that_finds_no_print_sets_the_page_up_again_when_the_overhead_cannot_place_the_sheet(setup, monkeypatch):
    """The arm over the page hides it from the overhead: with no fresh pose, the page is set up again rather than
    the stroke drawn blind; a fresh pose within reach draws with the field."""
    ex, io, node = setup.ex, setup.io, setup.node
    ex.tracker = first = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.stack["hardware"], ex.stack["page"]["track"]["mode"] = "real", "correct"
    first.verdicts = [None]
    node.camera_page = lambda arm: (np.eye(4), {"source": "lost", "measured_age_s": 30.0})
    pages, again = [], StubTracker([0.0, 0.0])
    monkeypatch.setattr(ex, "_new_page", lambda program, touch, ledger: pages.append(touch))
    monkeypatch.setattr(ex_mod.track.Tracker, "start", classmethod(lambda cls, ex, since=0.0: again))
    io.script = [("ok", 1, {}, None), ("ok", 1, {}, None), ("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert pages == [False]
    node.camera_page = lambda arm: (np.eye(4), {"source": "lost", "measured_age_s": 0.4})   # just hidden: in place
    ex.camera = np.eye(4)
    assert not ex._sheet_may_have_left()


def test_a_print_the_wrist_has_never_read_draws_on_the_measured_page_without_setting_it_up_again(setup, monkeypatch):
    """2026-10-05: the wrist fit read the seed-102 W transfer on silicone nowhere, and every stroke's hover set the
    page up again (~3 min each). A print never read in this goal is the fit's limit, not a sheet that left."""
    ex, io, node = setup.ex, setup.io, setup.node
    ex.tracker = StubTracker([0.0, 0.0])
    ex.tracker.field.samples = []
    _hovering(ex)
    ex.stack["hardware"], ex.stack["page"]["track"]["mode"] = "real", "correct"
    ex.tracker.verdicts = [None, None]
    node.camera_page = lambda arm: (np.eye(4), {"source": "lost", "measured_age_s": 30.0})
    pages = []
    monkeypatch.setattr(ex, "_new_page", lambda program, touch, ledger: pages.append(touch))
    io.script = [("ok", 1, {}, None), ("ok", 10, {}, UP), ("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE, LINE2]}, setup.ledger, touch=False) == "complete"
    assert pages == [] and len(ex.tracker.hovers) == 1 and not ex.tracker.correcting
    assert any("read the print nowhere in this goal" in text for _, _, text in node.events)


def test_a_resume_sets_the_page_up_again_past_the_wrist_fits_reach(setup):
    import copy

    ex, node = setup.ex, setup.node
    ex.stack = copy.deepcopy(STACK)
    ex.stack["page"].update(source="stencil", replan_translation_m=0.050, replan_rotation_rad=0.17)   # stack.yaml's stop
    page = {"camera_at_setup": np.eye(4).tolist(), "correction": np.eye(4).tolist()}
    node.camera = np.eye(4)
    node.camera[1, 3] = 0.02                                             # 20 mm: under the 50 mm stop
    assert ex.restore_page(page)                                         # without tracking it is kept
    ex.stack["hardware"], ex.stack["page"]["track"] = "real", {"mode": "correct"}
    assert not ex.restore_page(page)
    assert "past what stencil tracking follows" in node.events[-1][2]
