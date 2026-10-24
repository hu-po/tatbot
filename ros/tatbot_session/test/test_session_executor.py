"""ArmExecutor decisions without ROS I/O: latch mid-stroke -> aborted arc -> continue resumes there;
continue refused after the arm moved; resume timeout lands; pre-plan reuse versus dispatch drift; a page
that moves far is set up again; three touches fit the page plane; a Decide answers only the wait it was made for; a
latch in a stroke's final lift finishes the stroke; landing lifts a pen that is down first; a landed arm is refused
every goal. The arm I/O, kinematics and planner are small fakes;
the driver itself is only exercised on hardware and through the mock end-to-end test."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import tatbot_motion
from tatbot_session import executor as ex_mod
from tatbot_session import geometry, rules
from tatbot_session.ledger import Ledger

MOTION = {"approach": {"standoff_m": 0.010, "withdraw_m": 0.030}, "joint_speed": {"pen_up_rad_s": 1.0}, "carriage": {"max_m_s": 0.001},
          "pen": {"mode": "press", "press": {"lift_m": -0.002, "settle_s": 1.0},
                  "ride": {"fraction": 0.5, "settle_s": 0.0},
                  "trim": {"step_m": 0.0001, "limit_m": 0.004, "speed_m_s": 0.002, "min_s": 0.2, "lead_s": 0.1}},
          "touch": {"lift_m": 0.010, "max_travel_m": 0.040, "settle_s": 0.0}, "dispatch_drift": {"start_tolerance_m": 0.0005},
          "resume_reapproach_m": 0.0005, "clik": {"joint_limit_margin_rad": 0.0333},
          "probe": {"axial_cap_m": 0.001, "lateral_cap_m": 0.0025, "slow_m_s": 0.001, "back_off_m": 0.003,
                    "clearance_m": 0.030}}
STACK = {"session": {"knot_rate_hz": 100, "resume_timeout_s": 300},
         "page": {"source": "fixed", "size_m": [0.100, 0.150], "clear_m": [0.062, 0.112], "replan_translation_m": 0.050,
                  "replan_rotation_rad": 0.17, "trim": {"right": [0.0, 0.0]}},
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

    def land(self, budget_s, on_text=None):
        self.lands += 1
        self.safety.update(landed=1.0)   # as the driver: landed and idle until it is woken
        return True, "landed and idle"

    def execute(self, traj, *, should_stop=None, on_tick=None, watch_latch=True, retarget=None):
        """Pop the next scripted outcome: (status, index, safety updates, q at the end); none left runs
        the trajectory to its end. A stop request (should_stop) cancels it at that tick. A retarget is offered at
        every tick, at that knot's time."""
        self.executed.append(traj)
        self.active = traj
        status, index, safety, q = self.script.pop(0) if self.script else ("ok", len(traj.t) - 1, {}, None)
        for i in range(index + 1):
            if on_tick:
                on_tick(i)
            if i in self.at_tick:
                self.at_tick.pop(i)()
            if should_stop is not None and should_stop():
                status, index = "cancelled", i
                break
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
        self.since: list = []     # each wait's `since`: a pose measured after the arm cleared the overhead's view
        self.events: list[tuple] = []
        self.measured: list[tuple] = []   # (time.monotonic, base_from_page) the overhead measured
        self.marks: list[tuple] = []      # (time.monotonic, geometry.page_marks) of those it measured
        self.world = np.linalg.inv(TILE_WORLD)   # base_from_world: _tile_seen's page at the base

    def event(self, kind, *, arm="", op_id="", text=""):
        self.events.append((kind, op_id, text))

    def wait_camera_page(self, arm, timeout, max_lost_s=None, should_stop=None, since=None):
        self.waits.append(timeout)
        self.since.append(since)
        if len(self.waits) > 1 and self.later_info is not None:
            self.page_info = self.later_info
        return self.camera, dict(self.page_info)

    def camera_page(self, arm):
        return self.camera, {}

    def measured_pages(self, arm, since):
        return [pose for t, pose in self.measured if t > since]

    def measured_marks(self, since):
        return [(t, marks) for t, marks in self.marks if t > since]

    def base_from_world(self, arm):
        return self.world


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

    def joins(op, tip, *, base_from_page, motion, from_arc_m=0.0, pen=None, surface=None):   # plan_op's, on line_traj's
        tol = motion["resume_reapproach_m"] if from_arc_m > 0 else motion["dispatch_drift"]["start_tolerance_m"]
        start = np.asarray(base_from_page, float) @ [*line_traj(op, from_arc_m, n=2).tip[0], 1.0]
        return float(np.linalg.norm(np.asarray(tip) - start[:3])) <= tol

    monkeypatch.setattr(tatbot_motion, "joins", joins, raising=False)
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


def test_latch_mid_stroke_then_continue_lifts_and_resumes_at_its_arc(setup):
    ex, io = setup.ex, setup.io
    latched = {"latched": 1.0, "latch_reason": 1.0, "estop_ok": 1.0}
    io.script = [("latched", 6, latched, [0.0121, 0.0, 0.0, 0, 0, 0, 0]), ("ok", 1, {}, None), ("ok", 10, {}, None)]
    operator = Operator(ex, rules.CONTINUE)
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert operator.replies() == [(True, "")]
    rows = setup.ledger.rows()
    assert [r["event"] for r in rows] == ["sent", "aborted", "decision", "sent", "done"]
    assert rows[1]["arc_m"] == pytest.approx(0.012) and rows[1]["reason"] == "latched: scripted"
    assert rows[3]["arc_m"] == pytest.approx(0.012) and rows[4]["arc_m"] == pytest.approx(0.02)
    assert setup.lifts == [MOTION["touch"]["lift_m"]]                   # off the page along its normal first, so
    assert setup.calls[1]["from_arc"] == pytest.approx(0.012) and setup.calls[1]["pen_down"] is True   # it re-approaches
    assert np.allclose(setup.calls[1]["seed"][:3], [0.0121, 0.0, 0.010])
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
    assert io.unlatches == 1 and setup.lifts == [pytest.approx(ex_mod.ArmExecutor.LAND_CLEAR_M)] and io.lands == 1


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


def test_an_estop_held_holds_the_arm_and_released_lands_it_with_no_decision(setup):
    """Owner, 2026-10-24: the arm holds while the e-stop is pressed and returns to its rest once it is released
    (session.estop_release land), lifted off the skin first; no decision is waited for. Another latch (a guard's)
    still waits for one, up to resume_timeout_s."""
    ex, io = setup.ex, setup.io
    ex.stack = dict(STACK, session={"knot_rate_hz": 100, "resume_timeout_s": 300, "estop_release": "land"})
    io.script = [("latched", 6, {"latched": 1.0, "latch_reason": 1.0, "estop_ok": 0.0}, [0.0121, 0.0, 0.0, 0, 0, 0, 0])]
    released = threading.Timer(0.4, lambda: io.safety.update(estop_ok=1.0))
    released.start()
    began = time.monotonic()
    with pytest.raises(ex_mod.Landed):
        draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False)
    released.join()
    assert time.monotonic() - began >= 0.4                              # held while pressed
    assert any("e-stop released: landing" in text for _, _, text in setup.node.events)
    assert [r["event"] for r in setup.ledger.rows()] == ["sent", "aborted"] and io.lands == 1
    io.safety.update(latched=1.0, latch_reason=float(ex_mod.names.LATCH_GUARD_TIP_LAG), estop_ok=1.0)
    assert not ex._estop_released()                                      # a guard's latch: a decision waits
    io.safety.update(latch_reason=1.0)
    assert ex._estop_released()
    ex.stack["session"]["estop_release"] = "decide"
    assert not ex._estop_released()


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
    assert events[-1][2] == "r1 +0.2 mm of ±4.0"
    for _ in range(45):
        ex.on_key("down")
    assert ex.trim_target() == pytest.approx(-0.004) and events[-1][2] == "r1 stays -4.0 mm: its limit of ±4.0"
    ex.resource = {"id": "r2"}
    ex.on_key("up")
    ex.resource = {"id": "r1"}
    ex.on_key("reset")
    assert ex.trims == {"r1": 0.0, "r2": pytest.approx(0.0001)}
    rows = [(r["resource"], r["trim_m"]) for r in setup.ledger.rows() if r["event"] == "pen_trim"]
    assert len(rows) == 4 + 42 + 2 and rows[-2:] == [("r2", 0.0001), ("r1", 0.0)]


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


NEEDLES = SimpleNamespace(tool_id="lutin-3rl-bugpin", stroke_m=0.004, tip_out_at_top_m=None,
                          raw={"calibration": {"contact_reference": "tube_end"}, "needle_reach_mm": 4.0})


def _needles(setup, monkeypatch):
    """A needle cartridge on the arm (the 3RL's stroke and reach), riding, its machine's switch answering."""
    monkeypatch.setattr(ex_mod.config, "tool_datasheet", lambda repo, tool_id: NEEDLES)
    setup.ex.machine = Switch()


def _trims_sent(ledger, executed):
    """{op: (its sent row's trim_cap_m, the trim its goal went to, its ride's fold out)} of every stroke goal."""
    goals = [goal for goal in executed if "trim" in (getattr(goal, "info", None) or {})]
    rows = [row for row in ledger.rows() if row.get("event") == "sent"]
    return {row["op"]: (row.get("trim_cap_m"), goal.info["trim"].end_m - row.get("fold_m", 0.0))
            for row, goal in zip(rows, goals, strict=True)}


def test_a_riding_pen_runs_the_machine_for_its_strokes_and_stops_it_whenever_the_loop_waits_or_ends(riding):
    ex, io, node = riding.ex, riding.io, riding.node
    latched = {"latched": 1.0, "latch_reason": 1.0, "estop_ok": 1.0}
    io.script = [("latched", 6, latched, [0.0121, 0.0, 0.0, 0, 0, 0, 0]), ("ok", 1, {}, None), ("ok", 10, {}, None),
                 ("ok", 10, {}, None)]
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


def test_a_riding_draw_arms_the_snag_guard_before_its_first_stroke_and_keeps_it_through_its_lifts(riding):
    ex, io = riding.ex, riding.io
    writes = []   # (guard level, goals sent before it)
    io.on_write = lambda values: writes.append((values.get("guard_mode"), len(io.executed)))
    assert draw_strokes(ex, {"ops": [LINE, LINE2], "tool": PROGRAM_TOOL}, riding.ledger, touch=False) == "complete"
    assert writes == [(0, 0), (3, 0)] and len(io.executed) == 2   # disarmed by the setup, armed once for both strokes
    ex.motion = MOTION   # press: the setup disarms it and nothing arms it
    assert draw_strokes(ex, {"ops": [dict(LINE, id="s0002")]}, riding.ledger, touch=False) == "complete"
    assert writes[2:] == [(0, 2)]


SNAG = {"latched": 1.0, "latch_reason": float(ex_mod.names.LATCH_GUARD_SNAG), "estop_ok": 1.0}
IN_SKIN = [0.0121, 0.0, -0.0005, 0, 0, 0, 0]   # the needle stopped half a millimetre in


def _pen_down(monkeypatch):
    """FakeKin's tool pointing down the page's normal, as the real tcp's z does: along the pen out is up."""
    def fk(self, q):
        out = np.diag([1.0, -1.0, -1.0, 1.0])
        out[:3, 3] = np.asarray(q)[:3]
        return out
    monkeypatch.setattr(FakeKin, "fk", fk)


def test_a_snag_trip_withdraws_the_needle_along_the_pen_at_once_and_goes_on_measured_again(riding, monkeypatch):
    """A snag trip left the needle where it stopped, maybe in the skin with the machine off, until someone decided (up
    to resume_timeout_s). It comes out at once a standoff along the pen, the latch released as a continue releases it,
    the guard still armed, no decision asked; then the stroke goes on from the trip's arc as after any continue: the
    hover measures over where it goes on, and the needle goes down from there."""
    ex, io, node = riding.ex, riding.io, riding.node
    _pen_down(monkeypatch)
    ex.tracker = StubTracker([0.0, 0.0])
    ex.stack = {**STACK, "page": {**STACK["page"], "track": {"hover_stop": True, "second_view_m": 0.005}}}   # a copy
    ex.motion = {**RIDE, "approach": {**RIDE["approach"], "final_m": 0.005}}
    io.q = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0])
    io.script = [("ok", 1, {}, None), ("latched", 6, SNAG, IN_SKIN), ("ok", 1, {}, None), ("ok", 1, {}, None),
                 ("ok", 10, {}, UP)]                                   # hover, the trip, withdraw, hover, the rest
    assert draw_strokes(ex, {"ops": [LINE], "tool": PROGRAM_TOOL}, riding.ledger, touch=False) == "complete"
    withdraw = io.executed[2]
    assert np.allclose(withdraw.tip[-1] - withdraw.tip[0], [0.0, 0.0, MOTION["approach"]["standoff_m"]])   # along the pen
    assert riding.lifts == [MOTION["approach"]["standoff_m"]] and io.unlatches == 1 and ex.waiting is None
    assert [r["event"] for r in riding.ledger.rows()] == ["sent", "aborted", "sent", "done"]       # no decision
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover", "s0000-hover"]
    assert np.allclose(ex.tracker.hovers[1][2], [0.012, 0.0], atol=2e-4)                         # over the trip's arc
    assert riding.calls[-1]["from_arc"] == pytest.approx(0.012, abs=2e-4)
    assert ex.machine.log == [True, False, True, False] and io.writes == [{"guard_mode": 0}, {"guard_mode": 3}]   # armed
    assert any(kind == "latched" and text.startswith("snag guard:") and "withdrawn along the pen" in text
               for kind, _, text in node.events)


def test_a_second_snag_on_a_stroke_sets_the_page_up_again_and_a_third_waits_with_the_needle_out(riding, monkeypatch):
    ex, io, node = riding.ex, riding.io, riding.node
    _pen_down(monkeypatch)
    pages = []
    monkeypatch.setattr(ex, "_reacquire", lambda program, ledger, why, op_id="": pages.append(why))
    trip = ("latched", 6, SNAG, IN_SKIN)
    io.script = [trip, ("ok", 1, {}, None)] * 3 + [("ok", 10, {}, None)]
    operator = Operator(ex, rules.CONTINUE)
    assert draw_strokes(ex, {"ops": [LINE], "tool": PROGRAM_TOOL}, riding.ledger, touch=False) == "complete"
    assert operator.replies() == [(True, "")] and riding.lifts == [MOTION["approach"]["standoff_m"]] * 3
    assert pages == ["the snag guard tripped twice on s0000"]                     # each withdrawn, the second set up
    assert [r["event"] for r in riding.ledger.rows()] == ["sent", "aborted"] * 3 + ["decision", "sent", "done"]
    assert "trip" not in node.events[0][2] and "continue goes on, land lands" in [
        text for kind, _, text in node.events if kind == "latched"][-1]


@pytest.mark.parametrize("refused", ["held", "estop"])
def test_a_snag_withdraw_the_work_holds_or_the_estop_refuses_waits_and_land_withdraws_again(riding, monkeypatch, refused):
    """A needle the work still holds trips the withdraw too: the arm holds there for a person to free the needle, as
    with the e-stop pressed (no lift at all). Land then withdraws the needle a standoff along the pen under the guard,
    and lands only once that lift comes free; a needle still held trips it, and the arm holds there."""
    ex, io, node = riding.ex, riding.io, riding.node
    _pen_down(monkeypatch)
    if refused == "held":
        io.script = [("latched", 6, SNAG, IN_SKIN), ("latched", 0, SNAG, None), ("latched", 0, SNAG, None)]
    else:
        io.script = [("latched", 6, {**SNAG, "estop_ok": 0.0}, IN_SKIN)]
    operator = Operator(ex, rules.LAND)
    with pytest.raises(ex_mod.Landed, match="the snag guard held the lift" if refused == "held" else ""):
        draw_strokes(ex, {"ops": [LINE], "tool": PROGRAM_TOOL}, riding.ledger, touch=False)
    assert operator.replies() == [(True, "")]
    said = [text for kind, _, text in node.events if kind == "latched" and text.startswith("snag guard:")]
    assert "not withdrawn" in said[0] and "free the needle" in said[0] and ex.machine.log == [True, False]
    if refused == "held":
        assert riding.lifts == [0.010, 0.010] and io.lands == 0 and io.latched and io.writes[-1] == {"guard_mode": 3}
    else:
        assert "e-stop" in said[0] and riding.lifts == [] and io.lands == 1 and io.unlatches == 0


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


def test_page_motion_past_the_threshold_sets_the_page_up_again_where_the_overhead_places_it(setup):
    """Owner, 2026-10-19: "big move should set up again". The pen at the page lifts along its normal first, then the
    page is set up as at a first setup: the overhead's fresh pose places it, and the new page is the ledger's."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    ex.workspace, ex.camera = dict, np.eye(4)
    node.camera = geometry.rpy_matrix([0.049, 0, 0], [0, 0, 0.16])
    ex.refresh_page({"ops": [LINE]}, setup.ledger)  # within the threshold: nothing happens
    assert not node.waits and setup.lifts == []
    node.camera = geometry.rpy_matrix([0.1123, 0, 0], [0, 0, np.pi / 2])   # a quarter turn and 112 mm away
    ex.refresh_page({"ops": [LINE]}, setup.ledger)
    assert setup.lifts == [MOTION["touch"]["lift_m"]] and np.allclose(ex.used(), node.camera)
    assert [row["event"] for row in setup.ledger.rows()] == ["page"]
    assert any("112.3 mm, 90.0 deg (by its pose)" in text and "setting the page up again" in text
               for _, _, text in node.events)
    ex.refresh_page({"ops": [LINE]}, setup.ledger)  # the set-up page is where the overhead reads the sheet
    assert len(node.waits) == 1


def test_a_page_that_moves_before_a_stroke_is_set_up_again_and_the_stroke_goes_down_on_it(setup):
    ex, io, node = setup.ex, setup.io, setup.node
    ex.stack, ex.workspace = dict(STACK, page=dict(STACK["page"], source="stencil")), dict
    real_setup = ex.setup_page

    def setup_page(program, touch, ledger):
        real_setup(program, touch, ledger)
        node.camera = geometry.rpy_matrix([0.0, 0.1123, 0], [0, 0, 0])  # jumps once the run has its page

    ex.setup_page = setup_page
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert [row["event"] for row in setup.ledger.rows()] == ["page", "sent", "done"]
    assert np.allclose(ex.used(), node.camera)
    assert set(np.asarray(io.executed[0].phase).tolist()) == {6} and len(io.executed) == 2   # the lift, the stroke


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
        prior = prior_distance_m if prior_distance_m is not None else 0.0
        along = np.array([0.0, 0.0, prior - 0.004, prior, max_travel_m])
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


def test_a_view_goes_by_one_joint_move_under_the_probe_guard_and_one_that_dips_is_refused(setup, monkeypatch):
    """Touch MODE_VIEW, a calibration view whose way the calibration checked (Calibration.view_way): one joint move to
    the view's IK, no clearance plane, the probe guard armed; one that dips past STATION_DIP_M is not sent."""
    ex, io = setup.ex, setup.io
    monkeypatch.setattr(tatbot_motion, "plan_travel", lambda **kw: pytest.fail("a view plans no travel"), raising=False)
    monkeypatch.setattr(ex_mod.ready, "solve_ik_seeded", lambda kin, target, q: np.concatenate([target[:3, 3], np.zeros(4)]))
    moves = []

    def joint_move(joint_names, q_from, q_to, dip=0.0, **kw):
        moves.append(np.asarray(q_to, float))
        tip = np.array([q_from[:3], (np.asarray(q_from[:3]) + q_to[:3]) / 2 - [0.0, 0.0, dip], q_to[:3]], float)
        return SimpleNamespace(t=np.array([0.0, 1.0, 2.0]), q=np.column_stack([tip, np.zeros((3, 4))]),
                               qd=np.zeros((3, 7)), tip=tip, phase=np.full(3, 2, np.uint8))

    monkeypatch.setattr(ex_mod.ready, "joint_move", joint_move)
    io.q = np.array([0.2, 0.0, 0.25, 0, 0, 0, 0])
    target = np.eye(4)
    target[:3, 3] = [0.30, 0.0, 0.20]
    out = ex.move_single(target, 2, view=True)
    assert len(moves) == 1 and np.allclose(moves[0][:3], [0.30, 0.0, 0.20]) and np.allclose(out["q"][:3], moves[0][:3])
    assert io.writes[0] == {"guard_mode": 2} and len(io.executed) == 1 and not out["tripped"]
    io.executed.clear()
    io.q = np.array([0.2, 0.0, 0.25, 0, 0, 0, 0])
    monkeypatch.setattr(ex_mod.ready, "joint_move", lambda *a, **kw: joint_move(*a, dip=0.15, **kw))
    with pytest.raises(RuntimeError, match="down to z"):
        ex.move_single(target, 2, view=True)
    assert io.executed == []


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


def test_a_stack_started_under_a_pen_away_from_rest_withdraws_it_along_its_axis_before_landing(setup):
    """2026-10-06: the controller dropped 49 s into a stroke with the 3RL's needles in a half-skin; the restarted
    stack knows no page, so the pen comes out along its own axis before the driver's staged sweep, and farther than
    the standoff: the idled controller let the tip sag 9-13 mm into the skin before the restart (2026-10-08)."""
    from tatbot_session import inspect as ins

    ex, io, node = setup.ex, setup.io, setup.node
    assert ex.camera is None
    assert ex.decide(rules.LAND) == (True, "landed and idle")
    assert setup.lifts == [0.03] and len(io.executed) == 1 and io.lands == 1
    lifted = io.executed[0].q[-1][:3] - io.executed[0].q[0][:3]
    assert np.allclose(lifted, -ex.tcp()[:3, 2] * 0.03)                  # out along the pen, not the page
    assert any("lifted 30.0 mm along the pen before landing" in text for _, _, text in node.events)
    io.q, io.safety["landed"] = ins.REST.copy(), 0.0                     # at rest: nothing to withdraw
    io.q[1] = 0.15                                                       # within AT_REST_RAD on every joint
    assert ex.decide(rules.LAND)[0] and setup.lifts == [0.03] and io.lands == 2


def test_a_controller_error_ends_the_wait_at_once_and_its_landing_says_to_restart(setup):
    """No decision moves an arm whose controller dropped: the run ends rather than waiting resume_timeout_s with
    the needles in the skin, and the landing is not asked of a driver that refuses it."""
    ex, io, node = setup.ex, setup.io, setup.node
    io.safety.update(latched=1.0, controller_error=1.0)
    started = time.monotonic()
    assert ex.wait_decision("latched", "s0079") == rules.LAND and time.monotonic() - started < 1.0
    ok, why = ex._land_now()
    assert not ok and "tatbot ros up" in why and io.lands == 0 and setup.lifts == []
    assert any("the controller reports an error" in text for _, _, text in node.events)


def test_a_landed_arm_refuses_every_goal_until_it_is_woken(setup):
    """The driver keeps a landed arm idle until it is woken while every move sent to it still succeeded, the
    arm at rest (2026-09-29). A Draw or Touch goal for it is refused at acceptance, one accepted before the landing
    when it takes the arm, and each says how to wake it."""
    from rclpy.action import GoalResponse
    from tatbot_interfaces.action import Draw, Touch
    from tatbot_session import inspect as ins

    ex, io = setup.ex, setup.io
    io.q = ins.REST.copy()                                               # at rest: no pen to withdraw
    gate = GoalGate(ex)
    assert gate._goal_ok(Touch.Goal(arm="right")) == GoalResponse.ACCEPT and ex.refusal() is None
    assert ex.decide(rules.LAND) == (True, "landed and idle") and io.flag("landed")
    assert gate._goal_ok(Touch.Goal(arm="right")) == gate._goal_ok(Draw.Goal()) == GoalResponse.REJECT
    assert gate.events[-1][:2] == ("error", "right") and "client wake --arm right" in gate.events[-1][2]
    with pytest.raises(RuntimeError, match=r"the right arm has landed .*client wake --arm right"):
        gate._acquire(["right"])
    assert not ex.busy.locked() and io.executed == [] and io.lands == 1


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


def test_continuations_lift_before_a_new_path_or_a_skipped_tail(setup):
    ex = setup.ex
    ex.resource = {"id": "fitted", "dip": None}
    continuation = dict(LINE2, continues=True)
    program = {"ops": [LINE, continuation]}
    assert not ex.lift_at_end(program, LINE, [])
    assert ex.lift_at_end(program, continuation, [])
    assert ex.lift_at_end(program, LINE, [{"event": "skipped", "arm": "right", "op": continuation["id"]}])
    for following in (LINE2, {"op": "dip", "id": "d0"}, {"op": "tool_change", "id": "t0"}):
        assert ex.lift_at_end({"ops": [LINE, following]}, LINE, [])


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


def test_a_stroke_whose_continuation_the_charge_dips_before_lifts_at_its_end_and_is_not_preplanned(drawing, monkeypatch):
    """2026-10-20's tattoo: under the operator's dip scale a chain longer than a dip's line dipped between its sections
    from the first's end, pen-down; once the next one's pre-plan held the dip 30 s and ended the run, once the 20 mm lift
    off the skin could not be planned and ended it with the needle resting there. The first section now lifts at its
    end as any stroke does, and its continuation is not pre-planned across the dip."""
    ex, ledger = drawing.ex, drawing.ledger
    first = _stroke("s0", "r")
    cont = {**first, "id": "s1", "continues": True, "points_m": [[0.02, 0.0], [0.04, 0.0]]}
    program = _program(first, cont, resources=[_resource("r", "black", dip=True)])
    ledger.append("dip_scale", "right", op="s0", resource="r", scale=1.0, key="-")   # a 30 mm line a dip
    preplanned = []
    planned = ex._preplanned
    monkeypatch.setattr(ex, "_preplanned", lambda program, op, *a: preplanned.append(op["id"]) or planned(program, op, *a))
    assert ex.draw(program, ledger, touch=False) == "complete"
    assert drawing.dips == ["s0", "s1"] and preplanned == []
    assert [(c["op"], c["lift_at_end"]) for c in drawing.calls] == [("s0", True), ("s1", True)]
    ex.dip_scales, ex.inked_m, ex.resource = {"r": 1.0}, 0.0, _resource("r", "black", dip=True)   # line enough: down
    assert not ex.lift_at_end({"ops": [first, cont]}, first, [], drawn_m=0.005)


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


def test_a_resume_eases_each_needle_cartridges_trim_kept_past_skin_stop_m_back_to_it(drawing, monkeypatch):
    """A resume's page is a fresh field, no skin read sunk on it yet: each trim kept deeper than skin_stop_m goes back to
    it (a pen_trim row with eased_from_m, which the next resume starts from, and an event), and its strokes take it
    there; one within it stays (2026-10-20's resume carried -2.0 mm)."""
    _needles(drawing, monkeypatch)
    ex, ledger, events = drawing.ex, drawing.ledger, drawing.node.events
    program = _program(_stroke("s0", "black"), resources=[{**_resource("black", "black"), "pen_mode": "ride"}])
    ledger.append("pen_trim", "right", resource="black", trim_m=-0.004)
    ledger.append("pen_trim", "right", resource="red", trim_m=-0.002)
    assert ex.draw(program, ledger, touch=False) == "complete"
    assert ex.trims == {"black": pytest.approx(-0.0025), "red": -0.002}
    assert [(r["resource"], r["trim_m"], r.get("eased_from_m")) for r in ledger.rows() if r["event"] == "pen_trim"][-1] == (
        "black", -0.0025, -0.004)
    texts = [text for kind, _, text in events if kind == "pen_trim"]
    assert texts == ["trims kept from the run: black -4.0 mm, red -2.0 mm", "black -4.0 mm eased back to -2.5 mm: the run "
                     "resumed, and a deeper trim reaches only where the hovers read the skin sunk"]
    assert _trims_sent(ledger, drawing.io.executed) == {"s0": (pytest.approx(-0.0025), pytest.approx(-0.0025))}


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


def _dip(did, rid, reason):
    return {"op": "dip", "id": did, "resource_id": rid, "slot": "inkcap_medium_1", "reason": reason}


def _dip_rows(ledger, *ops):
    return [(r["op"], r["event"], round(r.get("arc_m", 0.0), 4), r.get("reason")) for r in ledger.rows()
            if r.get("op") in ops and r["event"] in ("sent", "aborted", "done", "skipped")]


def test_plus_mid_stroke_lifts_where_it_is_dips_and_the_stroke_goes_on_from_there(drawing):
    """+ while a stroke draws stops its goal at the arc reached; the pen lifts, the cap is dipped (under the stroke's
    id) and the stroke goes on from that arc. Every later dip is a step sooner: 27 of the cartridge's 30 mm, a
    dip_scale row; the compiled dip after the stroke still dips, since 14 mm drawn and the next 20 do not fit."""
    ex, io, events = drawing.ex, drawing.io, drawing.node.events
    program = _program({"op": "tool_change", "id": "t0", "resource_id": "red", "initial": True},
                       _dip("d1", "red", "initial"), _stroke("s2", "red"), _dip("d3", "red", "replenish"),
                       _stroke("s4", "red"), resources=[_resource("red", "red", dip=True)])
    io.at_tick = {3: lambda: ex.on_key("dip")}
    assert ex.draw(program, drawing.ledger, touch=False) == "complete"
    assert drawing.dips == ["d1", "s2", "d3"] and drawing.lifts == [MOTION["touch"]["lift_m"]]
    assert _dip_rows(drawing.ledger, "s2", "d3") == [
        ("s2", "sent", 0.0, None), ("s2", "aborted", 0.006, "numpad dip"), ("s2", "sent", 0.006, None),
        ("s2", "done", 0.02, None), ("d3", "sent", 0.0, None), ("d3", "done", 0.0, None)]
    assert [(r["resource"], r["scale"], r["key"], r["op"]) for r in drawing.ledger.rows()
            if r["event"] == "dip_scale"] == [("red", 0.9, "+", "s2")]
    texts = [text for _, _, text in events]
    assert "+: red dips every 27 mm (x0.9); dipping now" in texts and "the numpad's dip made" in texts


def test_dip_keys_change_nothing_without_a_draw_for_a_cartridge_that_takes_none_or_on_plus_while_a_dip_comes(
        drawing):
    ex, events = drawing.ex, drawing.node.events
    ex.on_key("dip")
    assert events[-1][2] == "+: no draw is running, so no cartridge to dip; nothing changed"
    ex.ledger, ex.resource = drawing.ledger, _resource("blue", "sky_blue")
    ex.on_key("fewer_dips")
    assert events[-1][2] == "-: blue (lutin-ballpoint-dot) takes no dips; nothing changed"
    ex.resource = _resource("red", "red", dip=True)
    ex.on_key("dip")
    ex.on_key("dip")
    assert ex.dip_asked and events[-1][2] == "+: a dip is asked already; nothing changed"
    ex.dip_asked, ex.dipping = False, True
    ex.on_key("dip")
    assert not ex.dip_asked and events[-1][2] == "+: a dip is running; nothing changed"
    ex.on_key("fewer_dips")   # - takes effect whatever runs: the next dips come a step later
    assert ex.dip_scales == {"red": 1.0} and events[-1][2] == "-: red dips every 30 mm (x1.0)"
    assert [(r["scale"], r["key"]) for r in drawing.ledger.rows() if r["event"] == "dip_scale"] == [
        (0.9, "+"), (1.0, "-")]


def test_compiled_dips_are_unchanged_when_a_kept_scale_returns_to_one(drawing):
    """Use the real compiler and executor over open/closed paths, timed continuations and replenishing dips."""
    from tatbot_ink.schedule import schedule_ops

    resource = {**_resource("red", "red", dip=True), "activation": None}
    strokes = [SimpleNamespace(points_m=np.array([[0.0, y], [length, y]]), closed=(k == 2),
                               generation_width_m=0.0003,
                               src={"artwork_sha256": "a", "pen": "red", "path": f"p{k}"})
               for k, (length, y) in enumerate(((0.020, 0.0), (0.045, 0.010), (0.010, 0.020)))]
    ops = schedule_ops(strokes, {("a", "red"): "red"}, [resource], speed=0.0035, max_seconds=2.0,
                       motion=tatbot_motion.load_motion())
    dips = [op["id"] for op in ops if op["op"] == "dip"]
    assert len(dips) > 1 and any(op.get("continues") for op in ops)
    drawing.ledger.append("dip_scale", "right", resource="red", scale=1.0, key="-")
    assert drawing.ex.draw(_program(*ops, resources=[resource]), drawing.ledger, touch=False) == "complete"
    assert drawing.dips == dips
    assert not any(row["event"] in ("aborted", "skipped") for row in drawing.ledger.rows())
    assert [row["op"] for row in drawing.ledger.rows() if row["event"] == "done"] == [op["id"] for op in ops]


def test_a_longer_line_per_dip_skips_a_compiled_dip_and_a_shorter_one_stops_a_stroke_where_its_charge_ends(drawing):
    """The red cartridge's scale comes from the run's ledger (a resume keeps it). At x2.0, 60 mm a dip, the
    replenishing dip between two 20 mm strokes is skipped; at x0.5, 15 mm, a 20 mm stroke stops at the first knot
    past its charge and SLACK_M (18 mm), dips there and goes on."""
    ex, ledger = drawing.ex, drawing.ledger
    red = _resource("red", "red", dip=True)
    program = _program({"op": "tool_change", "id": "t0", "resource_id": "red", "initial": True},
                       _dip("d1", "red", "initial"), _stroke("s2", "red"), _dip("d3", "red", "replenish"),
                       _stroke("s4", "red"), resources=[red])
    ledger.append("dip_scale", "right", resource="red", scale=2.0, key="-")
    assert ex.draw(program, ledger, touch=False) == "complete"
    assert drawing.dips == ["d1"] and [r["reason"] for r in ledger.rows() if r["event"] == "skipped"] == [
        "20 of the 60 mm a dip draws are drawn and the next 20 mm fit"]
    assert any(text == "dip scales kept from the run: red x2.0" for _, _, text in drawing.node.events)
    later = Ledger(ledger.path.with_name("later.jsonl"))
    later.append("dip_scale", "right", resource="red", scale=0.5, key="+")
    drawing.dips.clear()
    assert ex.draw(_program(*program["ops"][:3], resources=[red]), later, touch=False) == "complete"
    assert drawing.dips == ["d1", "s2"] and _dip_rows(later, "s2") == [
        ("s2", "sent", 0.0, None), ("s2", "aborted", 0.018, "charge spent"), ("s2", "sent", 0.018, None),
        ("s2", "done", 0.02, None)]


def test_a_numpad_dip_the_goal_never_made_is_dropped_and_the_resumed_goal_dips_once(drawing):
    """+ while a latched stroke waits, then land: the dip is never made, and the operator is told. The resumed goal
    dips before its first stroke, as every goal does, and makes no other dip for the old press."""
    ex, io, events = drawing.ex, drawing.io, drawing.node.events
    short = {**_stroke("s3", "red"), "points_m": [[0.0, 0.0], [0.01, 0.0]]}
    program = _program({"op": "tool_change", "id": "t0", "resource_id": "red", "initial": True},
                       _dip("d1", "red", "initial"), _stroke("s2", "red"), short,
                       resources=[_resource("red", "red", dip=True)])
    io.script = [("latched", 4, {"latched": 1.0, "latch_reason": 1.0}, None)]

    def press_then_land():
        deadline = time.monotonic() + 5.0
        while ex.waiting is None and time.monotonic() < deadline:
            time.sleep(0.002)
        ex.on_key("dip")
        ex.decide(rules.LAND, timeout=5.0)

    thread = threading.Thread(target=press_then_land, daemon=True)
    thread.start()
    with pytest.raises(ex_mod.Landed):
        ex.draw(program, drawing.ledger, touch=False)
    thread.join(5.0)
    assert drawing.dips == ["d1"] and not ex.dip_asked
    assert any("the numpad's dip was not made before the goal ended" in text for _, _, text in events)
    io.safety.update(landed=0.0)
    assert ex.draw(program, drawing.ledger, touch=False) == "complete"
    assert drawing.dips == ["d1", "s2"]


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
    from typing import Any

    ex = setup.ex
    record: dict[str, Any] = {"touches": []}
    ex.page_record = record
    up = np.array([0.0, 0.0, 1.0])
    tilt = geometry.rpy_matrix([0, 0, 0], [0.0, np.radians(1.0), 0.0])[:3, 2]   # a 1 deg page
    pts = [np.array([x, y, 0.005 - (tilt[0] * x + tilt[1] * y) / tilt[2]]) for x, y in
           ((0.31, -0.01), (0.33, -0.01), (0.33, 0.03), (0.31, 0.03), (0.32, 0.01))]
    pts[2][2] += 0.004                                                          # the bad one, at a corner
    normal, d = ex._gauge_plane(pts, up, {})
    assert ex.page_record["height"]["used"] == [True, True, False, True, True]
    assert np.degrees(np.arccos(np.clip(normal @ tilt, -1.0, 1.0))) < 0.01 and abs(pts[4] @ normal + d) < 1e-9
    assert ex._gauge_plane(pts[:4], up, {}) is None                            # four cannot say which one
    steep = geometry.rpy_matrix([0, 0, 0], [np.radians(10.0), 0.0, 0.0])[:3, 2]
    wild = [np.array([x, y, -(steep[0] * x + steep[1] * y) / steep[2]]) for x, y in ((0.31, -0.01), (0.33, -0.01), (0.32, 0.03))]
    assert ex._gauge_plane(wild, up, {}) is None


def test_a_flat_page_on_a_curved_skin_is_refused(setup):
    """2026-10-13: a flat page on the curved holder rode high at its edges and deep in its middle. A cylinder's four
    corners are coplanar, so the centre hold is the one off; it is never dropped as a bad reading."""
    from typing import Any

    ex = setup.ex
    record: dict[str, Any] = {"touches": []}
    ex.page_record = record
    radius = 0.045
    pts = [np.array([x, y, radius - (radius ** 2 - (y - 0.01) ** 2) ** 0.5]) for x, y in
           ((0.31, -0.025), (0.35, -0.025), (0.35, 0.045), (0.31, 0.045), (0.33, 0.01))]
    with pytest.raises(RuntimeError, match="surface fitted"):
        ex._gauge_plane(pts, np.array([0.0, 0.0, 1.0]), {})


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
    from typing import Any

    rec: dict[str, Any] = {"gauge": [{"xy": list(xy), "point": (prior @ [*xy, 0.0, 1.0])[:3].tolist(),
                                      "print": {"tip_m": [xy[0] + dx, xy[1] + dy, 0.012]}}
                                     for xy, (dx, dy) in zip(sent, off, strict=True)]}
    ex.page_record = rec
    moved = ex._hold_page(touched, prior)
    assert np.allclose(moved[:3, 3], [0.3015, 0.0236, 0.003]) and np.allclose(moved[:3, :3], np.eye(3))
    assert ex.page_record["holds"]["used"] and ex.page_record["holds"]["dropped"] == 1
    assert "-1.5 -3.6 mm off the print (4 holds" in said[-1]
    ex.page_record = {"gauge": [{"xy": [0.0, 0.0], "print": None}]}
    assert np.allclose(ex._hold_page(touched, prior), touched)
    retained = prior @ geometry.translation([0.0015, 0.0036])
    assert np.allclose(ex._hold_page(touched, retained, keep_prior=True)[:3, 3], [0.3015, 0.0236, 0.003])


def test_the_gauge_holds_turn_the_page_onto_a_turned_print(setup, monkeypatch):
    """A curved page's print turned 3 deg and slid 1.4 mm from the page the holds were made on: the set-up page puts
    every hold's believed point where its tip stood on the print, its height and tilt still the touched page's."""
    ex = setup.ex
    monkeypatch.setattr(ex, "event", lambda *a: None)
    monkeypatch.setattr(ex, "_curved", lambda: True)
    touched = geometry.rpy_matrix([0.32, 0.05, 0.003], [0.01, -0.02, -1.5])
    prior = geometry.rpy_matrix([0.321, 0.049, 0.006], [0.0, 0.0, -1.49])   # the locate's page: its own tilt
    sent = [(-0.0135, -0.0127), (0.0128, -0.0127), (0.0128, 0.0114), (-0.0135, 0.0114), (-0.0003, -0.0006)]
    c, s = np.cos(np.radians(3.0)), np.sin(np.radians(3.0))
    tips = [np.array([[c, -s], [s, c]]) @ (np.array(xy) - [0.004, -0.002]) + [0.0052, -0.0028] for xy in sent]
    ex.page_record = {"gauge": [{"xy": list(xy), "point": (prior @ [*xy, 0.0, 1.0])[:3].tolist(),
                                 "print": {"tip_m": [*tip, 0.010], "margin": 1.0}} for xy, tip in zip(sent, tips, strict=True)]}
    page = ex._hold_page(touched, prior)
    assert np.isclose(np.degrees(ex.page_record["holds"]["turn_rad"]), 3.0) and ex.page_record["holds"]["used"]
    assert np.allclose(page[:3, 2], touched[:3, 2]) and np.isclose((page[:3, 3] - touched[:3, 3]) @ touched[:3, 2], 0)
    for xy, tip in zip(sent, tips, strict=True):   # in the plane, prior's believed point is the print's tip point
        believed = prior @ [*xy, 0.0, 1.0]
        assert np.allclose((np.linalg.inv(page) @ believed)[:2], tip, atol=2e-5)


def test_the_gauge_holds_read_the_tip_against_the_skin_point_under_it_not_the_point_sent(setup, monkeypatch):
    """2026-10-19, a forearm print: each hold stands 10 mm over the skin square to the page it was made on, and over
    the skin's flank the skin point under the TCP (the gauge's, along the skin's normal: the TCP's foot, which the
    track's residual takes too) lay 0.4-1.2 mm from the commanded xy; read against xy, that lever became a 1 mm
    offset the track then corrected. Tips standing on the skin points leave the page where it is."""
    ex = setup.ex
    monkeypatch.setattr(ex, "event", lambda *a: None)
    page = np.eye(4)
    page[:3, 3] = [0.32, 0.05, 0.003]
    sent = [(-0.0135, -0.0127), (0.0128, -0.0127), (0.0128, 0.0114), (-0.0135, 0.0114), (-0.0003, -0.0006)]
    lever = [(-0.0006, 0.0002), (-0.0004, 0.0002), (-0.0010, 0.0010), (-0.0004, 0.0), (0.0002, -0.0007)]
    feet = [np.add(xy, d) for xy, d in zip(sent, lever, strict=True)]
    ex.page_record = {"gauge": [{"xy": list(xy), "point": (page @ [*foot, 0.0, 1.0])[:3].tolist(),
                                 "print": {"tip_m": [*foot, 0.010], "margin": 1.0}} for xy, foot in zip(sent, feet, strict=True)]}
    assert np.allclose(ex._hold_page(page, page), page, atol=1e-9)
    assert ex.page_record["holds"]["shift_m"] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_a_curved_pages_holds_far_from_the_set_up_page_leave_it(setup, monkeypatch):
    """2026-10-17, a full skin over a holder: every set-up's holds matched the print's far band, 12-13 mm across the
    curve (and 6 mm along it) from the overhead and wrist locate's page, sure of it (margins 0.09-0.15), and moved
    the page onto it; the near band put the print within 1.5 mm of the set-up's page across the curve. On a curved
    page a shift past HOLD_SHIFT_CURVED_M (12 mm since 2026-10-20: 9 before, the reviewers' 12 for the owner's 18)
    leaves the set-up's page unless its holds agree; one within it (and any on a flat page under HOLD_SHIFT_MAX_M,
    30 mm) still moves the page."""
    ex = setup.ex
    said = []
    monkeypatch.setattr(ex, "event", lambda kind, text, *a: said.append(text))
    monkeypatch.setattr(ex, "_curved", lambda: True)
    touched = np.eye(4)
    touched[:3, 3] = [0.32, 0.05, 0.003]
    sent = [(-0.022, 0.022), (-0.037, 0.004), (-0.006, 0.004), (-0.006, 0.039), (-0.037, 0.039)]

    def holds(dx, dy):
        return {"gauge": [{"xy": list(xy), "point": (touched @ [*xy, 0.0, 1.0])[:3].tolist(),
                           "print": {"tip_m": [xy[0] + dx, xy[1] + dy, 0.010], "margin": 1.0}} for xy in sent]}
    for dx in (0.019, 0.013):                                 # 20 mm, and the holder's 14 mm band
        ex.page_record = holds(dx, -0.006)
        assert np.allclose(ex._hold_page(touched, touched), touched)
        assert not ex.page_record["holds"]["used"] and "past 12 mm, not used" in said[-1]
    ex.page_record = holds(0.010, -0.006)                     # 11.7 mm: moved, within the limit
    assert np.allclose(ex._hold_page(touched, touched)[:3, 3], [0.310, 0.056, 0.003])
    monkeypatch.setattr(ex, "_curved", lambda: False)
    ex.page_record = holds(0.019, -0.006)
    assert np.allclose(ex._hold_page(touched, touched)[:3, 3], [0.301, 0.056, 0.003])


def test_a_curved_pages_holds_past_the_limit_that_agree_move_it_and_its_check_holds_confirm_it(setup, monkeypatch):
    """2026-10-20, the owner's forearm (its second set-up of the day's second run): four holds put the
    tool +5.2 -11.7 mm off the print, 1.0 / 0.1 mm apart, three by 3-5 markers, and the check holds read the same
    within 1.8 mm; refused past 9 mm, the tracker moved strokes 12 mm on one marker and the page was set up again.
    Past the shipped limit (12 mm) they move it, the check holds to confirm it; under it with nothing to confirm. Past
    HOLD_SHIFT_CURVED_M on a curved page (here at its old 9 mm, the agreement at its old 1.5 mm and the confirmation
    at 3) the holds move it when every kept reading, and every one by two markers or more, stands within HOLD_AGREE_M
    of the shift, those by two markers or more at two places or more; the surface check's holds then confirm it within
    HOLD_CONFIRM_M, or it is unconfirmed, inking or not (_new_page sets it up again once), and is placed for no later
    set-up. Holds that scatter,
    read by one marker each, outvote readings by more markers, read one place again, or stand 2.9 mm from the rest
    within the outlier cut leave the set-up's page as before (2026-10-20 review)."""
    ex = setup.ex
    said = []
    monkeypatch.setattr(ex, "event", lambda kind, text, *a: said.append(text))
    monkeypatch.setattr(ex, "_curved", lambda: True)
    touched = np.eye(4)
    touched[:3, 3] = [0.32, 0.09, 0.098]
    sent = [(0.0, 0.0), (-0.012, -0.012), (0.012, -0.012), (0.012, 0.012), (-0.012, 0.012)]
    checks = [(0.0, -0.012), (0.012, 0.0), (0.0, 0.012), (-0.012, 0.0)]

    def read(xy, dxy, markers, check=False, page=touched):
        return {"xy": list(xy), "point": (page @ [*xy, 0.0, 1.0])[:3].tolist(), "check": check,
                "print": {"tip_m": [xy[0] + dxy[0], xy[1] + dxy[1], 0.010], "margin": 1.0, "markers": markers,
                          "accepted": True}}

    def holds(dx, dy, markers=(1, 5, 4, 3, 1), scatter=0.0):
        return {"prior_correction": np.eye(4).tolist(), "gauge": [read(xy, (dx + scatter * (k - 2), dy), m)
                                                                  for k, (xy, m) in enumerate(zip(sent, markers, strict=True))]}

    ex.motion = {**MOTION, "pen": {**MOTION["pen"], "mode": "ride"}}
    ex.page_record = holds(0.0052, -0.0117)
    ex._hold_page(touched, touched)
    assert ex.page_record["holds"]["used"] and ex.page_record["holds"]["confirm"]   # 12.8 mm, past 12: they agree
    ex.page_record = holds(0.0052, -0.0097)
    ex._hold_page(touched, touched)
    assert ex.page_record["holds"]["used"] and not ex.page_record["holds"]["confirm"]   # 11.0 mm, within it
    for name, old in (("HOLD_SHIFT_CURVED_M", 0.009), ("HOLD_AGREE_M", 0.0015), ("HOLD_CONFIRM_M", 0.003)):
        monkeypatch.setattr(ex, name, old)
    ex.page_record = holds(0.0052, -0.0117)
    page = ex._hold_page(touched, touched)
    assert ex.page_record["holds"]["used"] and ex.page_record["holds"]["confirm"]
    assert np.allclose(page[:3, 3], [0.32 - 0.0052, 0.09 + 0.0117, 0.098]) and "they agree" in said[-1]
    ex.camera, ex.correction = np.eye(4), page
    gauged = list(ex.page_record["gauge"])
    for miss, mode in [(0.0012, "ride"), (0.004, "ride"), (0.004, "hover")]:
        ex.motion = {**MOTION, "pen": {**MOTION["pen"], "mode": mode}}
        ex.page_record["gauge"] = gauged + [read(xy, (miss, 0.0), 1, True, page) for xy in checks]
        ex._confirm_holds()                                    # never refused: inking or not, set up again once
        assert ex.page_record["holds"]["confirmed_m"] == pytest.approx(miss)
        assert ("it stands" if miss < ex.HOLD_CONFIRM_M else "not confirmed: set up again once") in said[-1]
        assert (ex.placed is None) == (miss > ex.HOLD_CONFIRM_M)   # placed for a later set-up once it stands
    ex.page_record["gauge"] = gauged + [read(checks[0], (0.004, 0.0), 2, True, page)]   # one check read: no confirm
    ex._confirm_holds()
    assert ex.page_record["holds"]["confirmed_m"] is None and "not confirmed" in said[-1]
    lone = [read(xy, (0.0052, -0.0117), 1) for xy in sent[:3]] + [read(sent[3], (0.0, 0.0), 3),
                                                                  read(sent[4], (0.0005, 0.0), 3)]
    again = [read(sent[2], (0.0052 + d, -0.0117), 2) for d in (0.0, 0.0001, -0.0001)] + [read(sent[0], (0.0052, -0.0117), 1)]
    apart = [read(xy, (0.0052 + d, -0.0117), 3) for xy, d in zip(sent[1:4], (0.0, 0.0002, 0.0029), strict=True)]
    for record in (holds(0.0052, -0.0117, scatter=0.0014), holds(0.0052, -0.0117, markers=(1, 1, 1, 1, 2)),
                   {"gauge": lone}, {"gauge": again}, {"gauge": apart}):
        ex.page_record = record
        assert np.allclose(ex._hold_page(touched, touched), touched)
        assert not ex.page_record["holds"]["used"] and "past 9 mm, not used" in said[-1]
    ex.page_record = holds(0.004, -0.006)   # within the limit: moved, nothing to confirm
    ex._hold_page(touched, touched)
    assert ex.page_record["holds"]["used"] and not ex.page_record["holds"]["confirm"]
    ex._confirm_holds()
    assert "confirmed_m" not in ex.page_record["holds"]


def test_a_page_no_check_holds_confirmed_is_never_carried_past_its_set_up(setup, monkeypatch):
    """2026-10-20 review: a page its holds moved past HOLD_SHIFT_CURVED_M was placed before its check holds ran, so a
    draw that inked was refused on it and the next draw, its holds reading under three prints, took it as the last
    page they placed (_placed_moved) with nothing to confirm. Unconfirmed, a draw's page (inking or a rehearsal: both
    are set up again once, then go on, since the owner's 2026-10-20 rule: adapt, never refuse) is not placed, does not
    seed the next set-up's holds, is not kept by a mid-draw set-up nor retained by a resume; a confirmed one is placed
    as before."""
    ex, node = setup.ex, setup.node
    monkeypatch.setattr(ex, "_curved", lambda: True)
    touched, seed = geometry.translation([0.32, 0.09, 0.098]), geometry.translation([0.0, 0.0, 0.002])
    sent = [(0.0, 0.0), (-0.012, -0.012), (0.012, -0.012), (0.012, 0.012), (-0.012, 0.012)]

    def read(xy, dxy, markers, check=False, page=touched):
        return {"xy": list(xy), "point": (page @ [*xy, 0.0, 1.0])[:3].tolist(), "check": check,
                "print": {"tip_m": [xy[0] + dxy[0], xy[1] + dxy[1], 0.010], "margin": 1.0, "markers": markers,
                          "accepted": True}}

    def set_up(mode, miss):   # holds 25.6 mm off that agree, then two check holds `miss` off the moved page
        ex.motion = {**MOTION, "pen": {**MOTION["pen"], "mode": mode}}
        ex.page_record = {"prior_correction": seed.tolist(), "gauge": [read(xy, (0.0104, -0.0234), 4) for xy in sent]}
        page = ex._hold_page(touched, touched)
        ex.camera, ex.correction = np.eye(4), page
        ex.prior_correction = ex.correction                    # as setup_page leaves it before the surface check
        ex.page_record["gauge"] += [read(xy, (miss, 0.0), 3, True, page) for xy in sent[1:3]]
        ex._confirm_holds()
        return page

    ex.placed, ex.overhead = None, np.eye(4)
    set_up("ride", 0.008)                                      # inking: unconfirmed (_new_page sets it up again)
    assert ex.placed is None and np.allclose(ex.prior_correction, seed)
    assert "not confirmed: set up again once" in node.events[-1][2]
    ex.page_record = {"gauge": []}                             # the next draw's holds read nothing: as touched
    assert np.allclose(ex._hold_page(touched, touched), touched) and ex.placed is None
    set_up("hover", 0.008)                                     # a rehearsal goes on, on a page carried no further
    assert ex.placed is None and np.allclose(ex.prior_correction, seed) and ex_mod._unconfirmed(ex.page_record)
    assert ex._kept_page() is None and "no check holds confirmed the page in use" in node.events[-1][2]
    rows = [{"arm": "right", "event": "page", "workspace": {}, "page": {"id": 1, "holds": {"used": True}}},
            {"arm": "right", "event": "page", "workspace": {}, "page": {"id": 2, **ex.page_record}}]
    monkeypatch.setattr(ex, "workspace", lambda: {})
    assert ex._retained_page(rows)["id"] == 1 and ex._retained_page(rows[1:]) is None
    moved = set_up("ride", 0.001)                              # confirmed: placed for a set-up whose holds read nothing
    assert np.allclose(ex.placed["page"], moved) and not ex_mod._unconfirmed(ex.page_record)
    ex.page_record = {"gauge": []}
    assert np.allclose(ex._hold_page(touched, touched), moved)


def test_a_page_its_check_holds_did_not_confirm_is_set_up_again_once_afresh_then_goes_on(setup, monkeypatch):
    """2026-10-20 review: going straight on, an unconfirmed page (_confirm_holds) inked where the newer check readings
    put the print past HOLD_CONFIRM_M. _new_page sets it up again once, afresh (as mid-draw: the overhead places it,
    or the wrist from it when the overhead reads no print, _last_page), the track's seed that set-up's holds alone
    (seed_since); still unconfirmed, it goes on there: never refused. One page row a set-up, the newest."""
    ex = setup.ex
    ex.clock, ex.workspace, calls = Ticking(500.0), dict, []

    def setup_page(program, touch, ledger=None, outcomes=(False, True)):
        calls.append((ex._again, ex.seed_since))
        ex._again, ex.page_record = False, {"n": len(calls), "holds": {"used": True, "confirm": True,
                                                                       "stands": outcomes[len(calls) - 1]}}
        ex.clock.t += 120.0

    monkeypatch.setattr(ex, "setup_page", setup_page)
    ex._new_page({"ops": [LINE]}, True, setup.ledger)
    assert calls == [(False, 0.0), (True, 620.0)] and not ex_mod._unconfirmed(ex.page_record)
    assert [r["page"]["n"] for r in setup.ledger.rows() if r["event"] == "page"] == [2]
    calls[:] = []
    monkeypatch.setattr(ex, "setup_page", lambda program, touch, ledger=None: setup_page(program, touch, ledger,
                                                                                          (False, False)))
    ex._new_page({"ops": [LINE]}, True, setup.ledger)          # unconfirmed twice: it goes on, not refused
    assert len(calls) == 2 and ex_mod._unconfirmed(ex.page_record)
    assert [r["page"]["n"] for r in setup.ledger.rows() if r["event"] == "page"] == [2, 2]
    record = dict(ex.page_record, correction=np.eye(4).tolist(), camera_at_setup=np.eye(4).tolist())
    ex.page_record, ex.stack = record, dict(STACK, page=dict(STACK["page"], locate={"holds": True}))
    monkeypatch.setattr(ex, "_gauge_ready", lambda: True)
    assert ex._last_page(True) is record                       # a mid-draw set-up the overhead cannot place starts there


def test_a_hold_read_again_as_it_steps_votes_once_by_its_last_reading(setup, monkeypatch):
    """2026-10-20 review: a curved hold reads again as it steps toward the skin, its first reading 5-10 mm higher, and
    every reading voted on the page: one hold read three times by one marker, with one more, moved a forearm's page
    inside the 9 mm limit and set its turn, and the later holds' first readings (the print's tip moving 0.14-0.28 mm
    a mm of stand there) put holds that agree at their last stand past HOLD_AGREE_M. A hold votes once, by its last
    reading, as the field and the track's seed take it; check holds are not the setup's."""
    ex = setup.ex
    said = []
    monkeypatch.setattr(ex, "event", lambda kind, text, *a: said.append(text))
    monkeypatch.setattr(ex, "_curved", lambda: True)
    touched = geometry.translation([0.32, 0.09, 0.098])
    sent = [(0.0, 0.0), (-0.012, -0.012), (0.012, -0.012), (0.012, 0.012), (-0.012, 0.012)]

    def read(i, dxy, markers=3, check=False):
        xy = sent[i % len(sent)]
        return {"touch": i, "xy": list(xy), "point": (touched @ [*xy, 0.0, 1.0])[:3].tolist(), "check": check,
                "print": {"tip_m": [xy[0] + dxy[0], xy[1] + dxy[1], 0.010], "margin": 1.0, "markers": markers,
                          "accepted": True}}

    ex.page_record = {"gauge": [read(0, (0.004, -0.003), 1)] * 3 + [read(1, (0.004, -0.003), 1)]}
    assert np.allclose(ex._hold_page(touched, touched), touched) and "holds" not in ex.page_record
    assert ex.placed is None                                  # two holds: under three votes, the page stands
    first = [(0.002, 0.0), (0.0, 0.002), (-0.002, 0.0), (0.0, -0.002), (0.0014, 0.0014)]   # 5 mm higher, 2 mm off
    ex.page_record = {"gauge": [read(0, (0.0052 + 0.004, -0.0117))] + [
        r for i, d in enumerate(first) for r in (read(i, (0.0052 + d[0], -0.0117 + d[1])),
                                                 read(i, (0.0052 + 0.0001 * (i - 2), -0.0117)))]
        + [read(100 + i, (0.003, 0.003), 4, True) for i in range(4)]}
    page = ex._hold_page(touched, touched)
    assert ex.page_record["holds"]["used"] and ex.page_record["holds"]["off_m"] < 0.0005
    assert np.allclose(page[:3, 3], [0.32 - 0.0052, 0.09 + 0.0117, 0.098], atol=2e-4) and "(5 holds" in said[-1]


def test_a_holds_print_is_read_behind_its_step_on_what_the_reading_held_and_in_place_before_the_holds_vote(
        setup, monkeypatch):
    """Reading the print in a hold's frames (inspect.hold_tip) took 1-2 s at every reading before its step was decided
    (2026-10-20). It is read in the background on what the reading held when made: on a curved page the first hold's
    skin (else the reading's own) with the camera turned by its calibration, on a flat one neither; the reading's
    record holds it once settled, which the holds' vote (_hold_page) waits for. The markers are read in every colour
    frame the hold took (gauge.WristGauge._pooled's fourth), the gauge's own when it kept none."""
    from tatbot_session import curved
    from tatbot_session import inspect as ins

    ex = setup.ex
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "locate": {"holds": True}}}   # not the shared STACK's
    gate, seen = threading.Event(), []

    def hold_tip(colors, frames, meta, tip_cam, cam_from_page, page, tile, surface=None, markers=None, every=None):
        gate.wait(5.0)
        seen.append((surface, cam_from_page, every))
        return {"tip_m": [0.001 * len(seen), 0.0, 0.010], "margin": 1.0, "markers": 3, "accepted": True}
    monkeypatch.setattr(ins, "print_artwork", lambda pattern_id: "tile")
    monkeypatch.setattr(ins, "hold_tip", hold_tip)
    monkeypatch.setattr(curved, "load_turn", lambda arm: np.array([0.0, 0.0, np.pi / 2]))
    monkeypatch.setattr(ex.kin, "frame", lambda q, name: np.eye(4), raising=False)
    raw, first = (([None], None, {}), [None], {"tip_cam_m": [0.0, 0.0, 0.1]}), SimpleNamespace()
    ex._provisional = first
    reads = [{"touch": k, "xy": [0.0, 0.0], "point": [0.0, 0.0, 0.0]} for k in range(2)]
    ex._hold_tip(reads[0], (([None], None, {}, ["every frame"]), *raw[1:]), np.zeros(7), np.eye(4), True)
    ex._provisional = SimpleNamespace()                                    # a later skin: not what it held
    ex._hold_tip(reads[1], raw, np.zeros(7), np.eye(4), False)
    assert reads[0]["print"] is None and reads[1]["print"] is None         # its step goes on meanwhile
    gate.set()
    ex.page_record = {"gauge": reads}
    ex._hold_page(np.eye(4), np.eye(4))
    assert reads[0]["print"]["tip_m"][0] == 0.001 and reads[1]["print"]["tip_m"][0] == 0.002
    assert seen[0][0] is first and not np.allclose(seen[0][1], np.eye(4))  # the skin, the camera turned
    assert seen[1][0] is None and np.allclose(seen[1][1], np.eye(4))       # flat: neither
    assert seen[0][2] == ["every frame"] and seen[1][2] is None


def test_a_holds_markers_go_to_the_wrist_pane_at_its_own_frames_stamp(setup, monkeypatch):
    """A hold's print is read behind its step, by when the next hold has grabbed: the markers it read (kept by its fit
    or not) go to the camera's tap at the stamp of the hold's own frame, which the viewer's wrist pane shows them on."""
    from tatbot_session import inspect as ins

    ex = setup.ex
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "locate": {"holds": True}}}
    gate, shown = threading.Event(), []

    def hold_tip(colors, frames, meta, tip_cam, cam_from_page, page, tile, surface=None, markers=None, every=None):
        gate.wait(5.0)
        markers.update({3: np.full((4, 2), 10.0), 8: np.full((4, 2), 20.0)})
        return {"tip_m": [0.0, 0.0, 0.010], "margin": 1.0, "markers": 1, "ids": [8], "accepted": True}
    monkeypatch.setattr(ins, "print_artwork", lambda pattern_id: "tile")
    monkeypatch.setattr(ins, "hold_tip", hold_tip)
    monkeypatch.setattr(ex.kin, "frame", lambda q, name: np.eye(4), raising=False)
    ex._cam = SimpleNamespace(stamp_ns=1_800_000_000, tap=SimpleNamespace(markers=shown.append, log=pytest.fail))
    read = {"touch": 4, "xy": [0.0, 0.0], "point": [0.0, 0.0, 0.0]}
    ex._hold_tip(read, (([np.zeros((720, 1280, 3), np.uint8)], None, {}), [None], {"tip_cam_m": [0.0, 0.0, 0.1]}),
                 np.zeros(7), np.eye(4), False)
    ex._cam.stamp_ns = 1_900_000_000                                       # the next hold's grab
    gate.set()
    ex._settle()
    (hold,) = shown
    assert (hold["frame_ns"], hold["reading"], hold["size"], hold["accepted"]) == (1_800_000_000, "hold 4", [1280, 720],
                                                                                   True)
    assert [(m["id"], m["fit"]) for m in hold["markers"]] == [(3, False), (8, True)] and read["print"]["ids"] == [8]


LOWER = {"role": "wrist_lower", "frame": "right/realsense_lower_color_optical_frame", "serial": "222",
         "cal": {"tip_cam_m": [0.0, -0.01, 0.15]}}


def test_a_holds_print_is_one_vote_of_both_cameras_each_shown_on_its_own_pane(setup, monkeypatch):
    """page.locate.cameras: a gauge hold's further camera reads the print in its own frames, on its own calibration
    and URDF pose, its markers on its own pane; the hold votes once (_hold_page): both readings averaged when both
    are sure of the print (accepted, three markers or more), the offset between them kept; else the session camera's
    when sure, else the further one's (a fallback), else the session camera's."""
    from tatbot_session import inspect as ins

    ex = setup.ex
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "locate": {"holds": True}}}
    poses, shown, accepted, read_by = [], {"upper": [], "lower": []}, {}, {}

    def hold_tip(colors, frames, meta, tip_cam, cam_from_page, page, tile, surface=None, markers=None, every=None):
        poses.append(cam_from_page)
        markers.update({7: np.full((4, 2), 1.0)})
        lower = tip_cam[1] < 0
        return {"tip_m": [0.003 if lower else 0.001, 0.0, 0.010], "turn_rad": 0.02 if lower else 0.0, "margin": 1.0,
                "markers": 5 if lower else read_by[False], "ids": [7], "rms_m": 0.0004, "accepted": accepted[lower]}
    monkeypatch.setattr(ins, "print_artwork", lambda pattern_id: "tile")
    monkeypatch.setattr(ins, "hold_tip", hold_tip)
    lower_pose = np.diag([1.0, -1.0, -1.0, 1.0])
    monkeypatch.setattr(ex.kin, "frame", lambda q, name: lower_pose if "lower" in name else np.eye(4), raising=False)
    tap = lambda key: SimpleNamespace(markers=shown[key].append, log=pytest.fail)   # noqa: E731
    ex._cam = SimpleNamespace(stamp_ns=1, tap=tap("upper"))
    shots = ([np.zeros((4, 4, 3), np.uint8)], [None], {})
    other = {"view": LOWER, "camera": SimpleNamespace(stamp_ns=2, tap=tap("lower")), "shots": shots, "frames": [None]}
    for (upper, lower), upper_markers, tip, markers in (((True, True), 4, 0.002, 5), ((True, True), 2, 0.003, 5),
                                                        ((False, True), 4, 0.003, 5), ((False, False), 4, 0.001, 4)):
        accepted.update({False: upper, True: lower})
        read_by[False] = upper_markers
        read = {"touch": 3, "xy": [0.0, 0.0], "point": [0.0, 0.0, 0.0]}
        ex._hold_tip(read, (shots, [None], {"tip_cam_m": [0.0, 0.01, 0.15]}), np.zeros(7), np.eye(4), False, (other,))
        ex._settle()
        assert read["print"]["tip_m"][0] == pytest.approx(tip) and read["print"]["markers"] == markers
        assert ("offsets_m" in read["print"]) == (tip == 0.002)       # kept where both are averaged
        assert len(read["print_cameras"]) == 2 and read.get("print_seen", [7]) == [7]   # said and dropped unplaced
    assert np.allclose(poses[1], lower_pose)                    # the lower read at its own pose (inverse of itself)
    assert [m["frame_ns"] for m in shown["upper"]] == [1] * 4 and [m["frame_ns"] for m in shown["lower"]] == [2] * 4
    assert ex._one_vote([{"accepted": True, "markers": 3, "tip_m": [0.0, 0.0]}, None])["markers"] == 3


def test_a_holds_cameras_and_the_next_holds_are_read_at_once_their_frames_written_apart(setup, monkeypatch):
    """2026-10-24: in all 25 frames of both D405s over the silicone tile a hold's print took ~13 s on the ros node,
    its holds 10 s apart; read one after another behind the frames' writes they held the setup 48 s after its holds.
    Each camera's reading is read on its own worker (READ_WORKERS), the frames written on another: a hold's two
    cameras read at once, two holds while a frame write still runs, and each hold still votes once (_settle)."""
    from tatbot_session import inspect as ins

    ex = setup.ex
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "locate": {"holds": True}}}
    together = threading.Barrier(2, timeout=5.0)   # a hold's cameras read one after the other never meet: it breaks

    def hold_tip(colors, frames, meta, tip_cam, cam_from_page, page, tile, surface=None, markers=None, every=None):
        together.wait()
        markers.update({7: np.full((4, 2), 1.0)})
        return {"tip_m": [0.002 if tip_cam[1] < 0 else 0.001, 0.0, 0.010], "turn_rad": 0.0, "margin": 1.0,
                "markers": 4, "ids": [7], "rms_m": 0.0004, "accepted": True}
    monkeypatch.setattr(ins, "print_artwork", lambda pattern_id: "tile")
    monkeypatch.setattr(ins, "hold_tip", hold_tip)
    monkeypatch.setattr(ex.kin, "frame", lambda q, name: np.eye(4), raising=False)
    ex._cam = SimpleNamespace(stamp_ns=None)
    written = threading.Event()
    ex._later.append((None, [ex._writes.submit(written.wait, 5.0)]))   # a hold's frames still being written
    shots = ([np.zeros((4, 4, 3), np.uint8)], [None], {})
    other = {"view": LOWER, "camera": SimpleNamespace(stamp_ns=None), "shots": shots, "frames": [None]}
    reads = [{"touch": t, "xy": [0.0, 0.0], "point": [0.0, 0.0, 0.0]} for t in (3, 4)]
    for read in reads:
        ex._hold_tip(read, (shots, [None], {"tip_cam_m": [0.0, 0.01, 0.15]}), np.zeros(7), np.eye(4), False, (other,))
    for _, futures in ex._later[1:]:
        assert [out["tip_m"][0] for out, _ in (f.result(timeout=5.0) for f in futures)] == [0.001, 0.002]
    assert not written.is_set()                                   # read while the write still ran
    written.set()
    ex._settle()
    assert [r["print"]["tip_m"][0] for r in reads] == pytest.approx([0.0015, 0.0015]) and ex._later == []
    assert all(len(r["print_cameras"]) == 2 and r["print_seen"] == [7] for r in reads)


def test_the_further_cameras_frames_are_taken_through_the_preview_and_kept_apart(setup, monkeypatch, tmp_path):
    """_further_frames: page.locate.cameras's cameras as the session camera's preview has them open, their frames
    written behind under touches/<role>/ (the tracker's seed reads the session camera's touches/gauge-*.npz); one not
    open yet, or one whose grab fails (said), is left out and the session camera's reading goes on."""
    from tatbot_session import gauge, track

    ex = setup.ex
    ex.run_dir = tmp_path / "run"
    monkeypatch.setattr(gauge, "CALIB", tmp_path)
    (tmp_path / "wrist-gauge-right.json").write_text(json.dumps({"tip_cam_m": [0.0, 0.0, 0.15]}))
    monkeypatch.setattr(track, "views", lambda ex_, roles, cal: [LOWER] if roles == ["wrist_lower"] else [])
    frame = np.zeros((4, 4, 3), np.uint8)
    lower = SimpleNamespace(serial="222", raw_depth=[np.zeros((4, 4), np.uint16)], depth_metadata={"m": 1},
                            grab_both=lambda n, fresh=False: ([frame] * n, []))
    ex._cam = SimpleNamespace(preview=SimpleNamespace(cameras=[lower]))
    assert ex._further_frames("gauge-2", np.zeros(7), 3) == []          # page.locate.cameras unset: none
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "locate": {"cameras": ["wrist_lower"]}}}
    ((view, cam, (colors, depths, meta)),) = ex._further_frames("gauge-2", np.zeros(7), 3)
    ex._settle()
    assert view is LOWER and cam is lower and len(colors) == 3 and meta == {"m": 1}
    assert (ex.run_dir / "touches" / "wrist_lower" / "gauge-2.npz").exists()
    assert not list((ex.run_dir / "touches").glob("gauge-*.npz"))   # not where the seed reads the session camera's
    ex._cam.preview.cameras = []                                    # the preview not open yet
    assert ex._further_frames("gauge-3", np.zeros(7), 3) == []
    said = []
    monkeypatch.setattr(ex, "event", lambda kind, text, op_id="": said.append(text))
    lower.grab_both = lambda n, fresh=False: (_ for _ in ()).throw(RuntimeError("Frame didn't arrive"))
    ex._cam.preview.cameras = [lower]
    assert ex._further_frames("gauge-4", np.zeros(7), 3) == [] and "Frame didn't arrive" in said[-1]


def test_a_further_cameras_heights_join_a_holds_only_when_they_agree_with_the_session_cameras(monkeypatch):
    """WristGauge.gauge: the further cameras' frames measured on their own calibrations; their heights join the
    hold's within page.height.gauge_spread_m of the session camera's median (a mount off moves no hold), else they
    are left out and the reading says so (GaugeRead.further)."""
    from tatbot_session import gauge

    monkeypatch.setattr(gauge, "measure", lambda d, c, meta, cal, curved=False: SimpleNamespace(h=float(d) + cal["off"]))
    monkeypatch.setattr(gauge, "height", lambda f, cal: f.h)
    upper = ([None] * 3, [0.0100, 0.0102, 0.0101], {})
    for off, used in ((0.0003, True), (0.0012, False)):
        lower = ({**LOWER, "cal": {"off": off}}, "the lower camera", ([None] * 3, [0.0100, 0.0101, 0.0102], {}))
        ex = SimpleNamespace(stack={"page": {}}, _wrist_frames=lambda *a, **k: upper, _gauge_limit=lambda: 0.0005,
                             _further_frames=lambda *a, further=lower, **k: [further], wrist_on=lambda: False)
        sensor = gauge.WristGauge(ex)
        sensor.cal = {"off": 0.0}
        read = sensor.gauge(np.zeros(7), np.eye(4), np.eye(4), curved=False, count=3, name="gauge-1")
        (other,) = read.further
        assert other["used"] is used and other["camera"] == "the lower camera" and len(other["heights_m"]) == 3
        assert len(read.heights_m) == (6 if used else 3)
        assert read.point_base[2] == pytest.approx(-np.median(read.heights_m))


def test_a_holds_cameras_are_grabbed_and_measured_at_once(monkeypatch):
    """One after the other a hold stood ~4.6 s from arrival to its reading, each camera's 25 frames ~0.83 s of it
    (2026-10-24): WristGauge.gauge opens the session camera (the further ones are its previews), then grabs and
    measures the further cameras' frames while it grabs and measures its own. Each grab here waits for the other to
    have started: one after the other, the first would wait out its timeout."""
    from tatbot_session import gauge

    monkeypatch.setattr(gauge, "measure", lambda d, c, meta, cal, curved=False: SimpleNamespace(h=float(d)))
    monkeypatch.setattr(gauge, "height", lambda f, cal: f.h)
    started, opened = {"upper": threading.Event(), "lower": threading.Event()}, []

    def grab(mine, other, out):
        started[mine].set()
        assert started[other].wait(2.0), f"{mine} grabbed alone"
        return out
    upper = ([None] * 3, [0.0100, 0.0101, 0.0102], {})
    lower = ({**LOWER, "cal": {}}, "the lower camera", ([None] * 3, [0.0100, 0.0101, 0.0102], {}))
    ex = SimpleNamespace(stack={"page": {}}, _gauge_limit=lambda: 0.0005, wrist_on=lambda: True,
                         setup_camera=lambda: opened.append(len(opened)),
                         _wrist_frames=lambda *a, **k: grab("upper", "lower", upper),
                         _further_frames=lambda *a, **k: [grab("lower", "upper", lower)])
    sensor = gauge.WristGauge(ex)
    sensor.cal = {}
    read = sensor.gauge(np.zeros(7), np.eye(4), np.eye(4), curved=False, count=3, name="gauge-1")
    assert opened == [0] and len(read.heights_m) == 6 and read.further[0]["used"]


def test_each_locate_view_is_read_by_the_further_cameras_too(setup, monkeypatch):
    """page.locate.cameras: at each locate view the further cameras read the print from the same pose, at their own
    URDF pose; their fits join the views' (combine_views), are recorded under the locate's "further", and one that
    reads the print counts as the view reading it."""
    ex = setup.ex
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "size_m": [0.045, 0.063], "pattern_id": "p"}}
    cfg = {"poses": 2, "wrist_floor_m": 0.002, "wrist_floor_rad": 0.01, "overhead_sigma_m": 0.004,
           "overhead_sigma_rad": 0.026}
    _locate_fakes(ex, monkeypatch, lambda az: 180.0)
    lower = SimpleNamespace(name="lower")
    fit = {"tx_m": 0.002, "ty_m": 0.0, "theta_rad": 0.0, "sigma": [0.001, 0.001, 0.01], "sides": ["markers"]}
    monkeypatch.setattr(ex, "_view_read", lambda cam, *a, **k: (np.zeros((4, 4, 3), np.uint8), fit if cam is lower
                                                                else None, "" if cam is lower else "no marker read"))
    asked = []
    monkeypatch.setattr(ex, "_further_frames", lambda name, q, count, every=False: asked.append(
        (name, count)) or [(LOWER, lower, ([np.zeros((4, 4, 3), np.uint8)], [], {}))])
    page, record = ex.wrist_locate(np.eye(4), cfg)
    assert [(f["view"], f["camera"], f["fit"]) for f in record["further"]] == [(0, "wrist_lower", fit),
                                                                              (1, "wrist_lower", fit)]
    assert record["wrist"]["poses"] == 2 and record["wrist"]["tx_m"] == pytest.approx(0.002)
    assert [v["fit"] for v in record["views"]] == [None, None] and asked[0] == (None, 5)   # none written: no name


def test_the_locate_plans_its_views_about_the_page_tilted_ten_degrees_at_most(setup, monkeypatch):
    """2026-10-20: the overhead's page of a forearm print tilted 19-20 deg (the chord plane of its markers in view)
    and no locate view about it was reachable, both set-ups saying "no view read the print's markers"; replayed with
    its tilt capped at 10 deg, 22 and 24 were. The views are planned about the page turned the least toward level, to
    10 deg from the base's z, its centre kept; a page tilted less is planned about as it is; the locate says how many views
    it reached."""
    import sys
    from types import SimpleNamespace

    from tatbot_session import inspect as ins

    ex = setup.ex
    ex.stack["page"] = {**ex.stack["page"], "size_m": [0.045, 0.063]}
    planned = []
    monkeypatch.setattr(ins, "aim", lambda kin, solve, page, *a, **k: planned.append(np.array(page)))
    monkeypatch.setattr(ex, "wrist_camera", lambda serial: SimpleNamespace(close=lambda: None))
    monkeypatch.setitem(sys.modules, "tatbot_cli", SimpleNamespace(ros_depth=SimpleNamespace()))
    said = []
    monkeypatch.setattr(ex, "event", lambda kind, text, *a: said.append(text))
    cfg = {"poses": 1, "wrist_floor_m": 0.002, "wrist_floor_rad": 0.01}
    for tilt in (20.0, 5.0):
        prior = ins.rpy_matrix([0.32, 0.09, 0.1], [np.radians(tilt), 0.0, np.radians(30.0)])
        page, record = ex.wrist_locate(prior, cfg)
        assert np.allclose(page, prior) and record["views"] == [] and "0 views of the print reached" in said[-1]
        n = planned[-1][:3, 2]
        assert np.allclose(planned[-1][:3, 3], prior[:3, 3])
        assert np.degrees(np.arccos(n[2])) == pytest.approx(min(tilt, 10.0), abs=1e-6)
        assert abs(n @ np.cross(prior[:3, 2], [0, 0, 1.0])) < 1e-9   # turned toward level, not aside
        assert planned[-1][:3, 0] @ prior[:3, 0] > np.cos(np.radians(10.0))


def _locate_fakes(ex, monkeypatch, answer):
    """wrist_locate with no motion and no camera: inspect.aim answers `answer(azimuths in deg, or None)` (a view's
    azimuth, or None for none reached) and records each search; every view read reads no marker."""
    import sys

    from tatbot_session import inspect as ins

    searched = []
    monkeypatch.setattr(ex, "clock", Ticking(0.0))   # simulated camera settling on already-fake motion

    def aim(kin, solve, page, xy, q, *, azimuths=None, **k):
        if "tilts_deg" not in k:   # the oblique views after the top-down ones: none reached
            return None
        searched.append(None if azimuths is None else [round(float(v)) for v in np.degrees(azimuths)])
        found = answer(searched[-1])
        return None if found is None else (np.zeros(7), {"azimuth_deg": float(found), "tilt_deg": 40.0})
    frames = ([np.zeros((4, 4, 3), np.uint8)], [np.zeros((4, 4))])
    cam = SimpleNamespace(close=lambda: None, grab_both=lambda n, observe=None, fresh=False: frames)
    monkeypatch.setattr(ins, "aim", aim)
    monkeypatch.setattr(ex, "wrist_camera", lambda serial: cam)
    monkeypatch.setitem(sys.modules, "tatbot_cli", SimpleNamespace(ros_depth=SimpleNamespace(
        observe=lambda io: None, retain=lambda *a: (None, {"reason": "none"}, None))))
    monkeypatch.setattr(ex, "lowest_over_page", lambda traj, page=None: 1.0)
    monkeypatch.setattr(ex, "move", lambda make, phase, what, ledger=None: None)
    monkeypatch.setattr(ex, "_view_read", lambda *a: (frames[0][0], None, "no marker of the print read"))
    monkeypatch.setattr(ex.kin, "frame", lambda q, name: np.eye(4), raising=False)
    return searched


def test_the_locate_searches_about_the_last_first_view_first_and_stops_once_no_view_is_reached(setup, monkeypatch,
                                                                                               tmp_path):
    """Every view was searched before the arm moved, 41-45 s on the ros node (2026-10-20), and when none was
    reachable it was searched again, alike, for each of three poses (117-122 s, 2026-10-20). The first view is searched about
    the azimuth the last setup reached its first at (the carried paper's), every view only when none is reached there;
    the views after it about the first, as before; and a search of every view that reaches none ends the locate."""
    ex = setup.ex
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "size_m": [0.045, 0.063], "pattern_id": "p"}}
    cfg = {"poses": 3, "wrist_floor_m": 0.002, "wrist_floor_rad": 0.01}
    searched = _locate_fakes(ex, monkeypatch, lambda az: None)
    page, record = ex.wrist_locate(np.eye(4), cfg)
    assert searched == [None] and record["views"] == []                  # no hint: every view, once
    ex.carried = {"arm": "right", "pattern_id": "p", "view_base_deg": 240.0}
    searched = _locate_fakes(ex, monkeypatch, lambda az: az[1] if az else 210.0)
    ex.wrist_locate(np.eye(4), cfg)
    assert searched == [[225, 240, 255], [255, 270, 285], [225, 210, 195]]   # the hint reached: no search of all
    searched = _locate_fakes(ex, monkeypatch, lambda az: 180.0 if az is None else None if 240 in az else az[0])
    _, record = ex.wrist_locate(np.eye(4), cfg)
    assert searched == [[225, 240, 255], None, [195, 210, 225], [165, 150, 135]]
    assert [v["azimuth_deg"] for v in record["views"]] == [180.0, 195.0, 165.0]
    ex.carried["pattern_id"] = "another print"                          # the reach is the arm's, whichever print
    searched = _locate_fakes(ex, monkeypatch, lambda az: None)
    ex.wrist_locate(np.eye(4), cfg)
    assert searched == [[225, 240, 255], None]


def test_the_locate_reads_from_over_the_page_centre_first_and_its_top_down_views_only_when_that_reads_too_little(
        setup, monkeypatch):
    """2026-10-24: the top-down views took ~30 s of view search and two ~5.5 rad swings of a 160 s setup, on prints the
    overhead had placed within a few mm. The locate first stands the pen square to the page over its centre at each of
    page.locate.over_m in turn (inspect.over_centre) at the ready move's heading and seed (_ready_for), until
    page.locate.over_enough camera reads place the print; the top-down views only after, numbered on. A view that
    would see none of the print's markers is not moved to, and each one's depth is kept for the first hold."""
    from tatbot_session import inspect as ins

    ex = setup.ex
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "size_m": [0.045, 0.063], "pattern_id": "p"}}
    fit = {"tx_m": 0.001, "ty_m": 0.0, "theta_rad": 0.0, "sigma": [0.001, 0.001, 0.01], "sides": ["markers"]}
    heading, seed, lower = ins.rpy_matrix([0.0, 0.0, 0.0], [np.pi, 0.0, 0.3]), np.full(7, 0.2), SimpleNamespace()
    monkeypatch.setattr(ex, "_ready_for", lambda page, pattern: (heading, seed))
    monkeypatch.setattr(ex, "_skin_seen", lambda cam, depths, q: {})
    cfg = {"over_m": [0.05, 0.10], "over_enough": 2, "poses": 2, "wrist_floor_m": 0.002, "wrist_floor_rad": 0.01,
           "overhead_sigma_m": 0.004, "overhead_sigma_rad": 0.026}

    def run(reads, further=True, seen=lambda h: 6):
        searched, overs = _locate_fakes(ex, monkeypatch, lambda az: 200.0 if az is None else az[0]), []

        def over(kin, solve, page, xy, q, h, *, height_m, seed, **k):
            overs.append((height_m, h is heading, seed is seed_))
            return np.full(7, height_m), {"over_m": height_m, "seen": seen(height_m)}
        monkeypatch.setattr(ins, "over_centre", over)
        monkeypatch.setattr(ex, "_view_read", lambda cam, *a, **k: (np.zeros((4, 4, 3), np.uint8),
                                                                    fit if reads(cam) else None, "no marker read"))
        monkeypatch.setattr(ex, "_further_frames", lambda name, q, count, every=False: [
            (LOWER, lower, ([np.zeros((4, 4, 3), np.uint8)], [], {}))] if further else [])
        ex._over_skin = []
        _, record = ex.wrist_locate(np.eye(4), cfg)
        return [h for h, *same in overs if all(same)], searched, [v.get("over_m") for v in record["views"]], record
    seed_ = seed
    overs, searched, views, record = run(lambda cam: True)          # both D405s read it from 50 mm: done
    assert overs == [0.05] and searched == [] and views == [0.05] and record["wrist"]["poses"] == 2
    assert len(ex._over_skin) == 1
    overs, searched, views, _ = run(lambda cam: True, further=False)   # the session camera alone: up to 100 mm
    assert overs == [0.05, 0.10] and searched == [] and views == [0.05, 0.10] and len(ex._over_skin) == 2
    overs, searched, views, _ = run(lambda cam: False)               # neither reads it: the top-down views after
    assert overs == [0.05, 0.10] and len(searched) == 2 and views == [0.05, 0.10, None, None]
    assert any(text.startswith("locate view 2: one reachable") for _, _, text in setup.node.events)
    overs, searched, views, _ = run(lambda cam: True, seen=lambda h: 0 if h < 0.08 else 3)   # none in view at 50 mm
    assert overs == [0.05, 0.10] and views == [0.10] and searched == []


def test_the_over_centre_view_and_the_ready_move_share_a_heading_and_branch(tmp_path, monkeypatch):
    """The over-centre view goes at the heading and IK seed the ready move will take (_ready_for): searched on a new
    page and kept for the ready move (pen_heading takes it unsearched), so the ready move under it only descends (0.15-0.22
    rad over 37 setups' pages, 2026-10-24); on a page with a carried ready pose on the same print, that pose's."""
    import functools

    pytest.importorskip("pinocchio")
    from tatbot_description import robot_description
    from tatbot_motion.clik import Kinematics
    from tatbot_session import inspect as ins

    kin = Kinematics(robot_description(arms=("right",)), "right")
    page = np.eye(4)
    page[:3, :3] = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]   # the print's x along the base's -y
    page[:3, 3] = [0.296, 0.013, 0.003]
    ex = ex_mod.ArmExecutor(FakeNode(np.eye(4)), FakeIO(), kin, MOTION, STACK, tmp_path)
    ex.camera, ex.correction, ex.page_record = page.copy(), np.eye(4), {}
    monkeypatch.setattr(ex, "_dip_heading", lambda: None)
    searched, search = [], ex_mod.ready.pen_heading
    monkeypatch.setattr(ex_mod.ready, "pen_heading", lambda *a, **k: searched.append(1) or search(*a, **k))
    heading, seed = ex._ready_for(page, "p")
    assert len(searched) == 1 and np.allclose(seed[:6], ins.REST[:6])
    view = ins.over_centre(kin, functools.partial(ex_mod.ready.solve_ik, iterations=100), page, (0.0, 0.0),
                           ex.io.measured(), heading, frame=ins.camera_frame("right"), clearance_m=0.015,
                           height_m=0.050, seed=seed)
    assert view is not None and np.allclose(ex.pen_heading(), heading) and len(searched) == 1
    q = ex_mod.ready.solve_ik_seeded(kin, geometry.tool_down_pose(ex.used(), [0.0, 0.0], 0.010, ex.pen_heading()),
                                     ex._ready_seed())
    assert np.max(np.abs(q[:6] - view[0][:6])) < 0.3
    turned = geometry.tool_down_pose(page, [0.0, 0.0], 0.010, ins.rpy_matrix([0.0, 0.0, 0.0], [np.pi, 0.0, 0.5]))
    ex.carried = {"arm": "right", "pattern_id": "p", "camera": page.tolist(), "plane": [0.0, 0.0, 1.0, -0.003],
                  "utc_s": ex.clock.utc(), "ready": {"q": [0.1] * 7, "heading": turned.tolist()}}
    ex._located_heading = None
    heading, seed = ex._ready_for(page, "p")
    assert len(searched) == 1 and np.allclose(heading, turned) and np.allclose(seed[:6], 0.1)
    assert ex._located_heading is None and ex._ready_for(page, "another print")[1][0] == pytest.approx(ins.REST[0])


def _turned(deg: float) -> np.ndarray:
    out = np.eye(4)
    out[:2, :2] = [[np.cos(np.radians(deg)), -np.sin(np.radians(deg))], [np.sin(np.radians(deg)), np.cos(np.radians(deg))]]
    return out


def test_the_first_views_hint_is_carried_in_the_arms_base_and_turned_onto_the_next_page(setup, monkeypatch, tmp_path):
    """A view's azimuth is the page's and its reach the arm's: carried as the page's, a print turned 87-125 deg since its
    last setup searched every view, 63-83 s on the ros node, four setups running (2026-10-24). The first view's azimuth
    is carried about the base's z too and turned onto the next page, on another print alone; on the same print beside
    its own (which markers the tool hides is the print's), that alone when the two lie within 15 deg; a file carried
    before, by its page."""
    ex = setup.ex
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "size_m": [0.045, 0.063], "pattern_id": "p"}}
    ex.page_record = {"touches": [{"tcp": np.eye(4).tolist()}],                # an over-centre view first
                      "locate": {"overhead": _turned(30.0).tolist(), "views": [{"over_m": 0.05}, {"azimuth_deg": 195.0}]}}
    ex.run_dir = tmp_path / "ros-draw" / "run"
    ex._carry_paper(_turned(31.0), {"pattern_id": "p"})
    carried = json.loads((tmp_path / "page-touched-right.json").read_text())
    assert carried["view_base_deg"] == pytest.approx(225.0) and carried["view_deg"] == 195.0
    ex.carried = None                                                       # a restarted stack: the carried file's
    assert ex._view_hint(np.eye(4), "q") == pytest.approx([225.0])         # another print: the base's alone
    assert ex._view_hint(_turned(-64.0), "q") == pytest.approx([289.0])    # turned: the same reach
    assert ex._view_hint(_turned(-64.0), "p") == pytest.approx([195.0, 289.0])
    assert ex._view_hint(_turned(25.0), "p") == pytest.approx([195.0])     # the print as it was: its own alone
    ex.carried, ex.run_dir = ex._carried(), None                           # the locate keeps no views here
    cfg = {"poses": 1, "wrist_floor_m": 0.002, "wrist_floor_rad": 0.01}
    searched = _locate_fakes(ex, monkeypatch, lambda az: az[1] if az else None)
    ex.wrist_locate(_turned(125.0), cfg)
    assert searched == [[180, 195, 210, 85, 100, 115]]                     # both, in one search
    ex.page_record = {"touches": [{"tcp": np.eye(4).tolist()}], "locate": {"views": []}}
    ex._carry_paper(_turned(90.0), {"pattern_id": "q"})                    # no view this setup: the last one's kept
    assert ex.carried["view_base_deg"] == pytest.approx(225.0) and ex.carried["view_deg"] == 195.0
    ex.carried = {"arm": "right", "camera": _turned(-72.0).tolist(), "view_deg": 300.0}   # carried by its page
    assert ex._view_hint(_turned(125.0), "p") == pytest.approx([103.0])
    ex.carried = {"arm": "left", "view_base_deg": 225.0}
    assert ex._view_hint(np.eye(4), "p") == []


def test_the_locate_moves_to_its_views_at_the_pen_up_joint_speed_under_the_caps(setup, monkeypatch):
    """At a third of the pen-up speed the swing from rest to the first view took 5-8 s and joint_3, lowering under
    0.3 rad/s, chattered at 5 Hz as it slowed, growing to 8.5-13.8 mrad and the tip 2.8-6.7 mm; the views between
    chattered 3 mrad for 5 s (2026-10-20/21). The views are configured pen-up joint moves (ready.pen_up_joint_move)."""
    ex = setup.ex
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "size_m": [0.045, 0.063], "pattern_id": "p"}}
    _locate_fakes(ex, monkeypatch, lambda az: 200.0 if az is None else az[0])
    moves = []
    monkeypatch.setattr(ex, "move", lambda make, phase, what, ledger=None: moves.append(make()))
    ex.io.q = np.array([0.4, -0.6, 1.0, 0.3, 0.0, 0.0, 0.0])   # every view the fakes reach is at zero
    ex.wrist_locate(np.eye(4), {"poses": 1, "wrist_floor_m": 0.002, "wrist_floor_rad": 0.01})
    (move,) = moves
    same = ex_mod.ready.pen_up_joint_move(ex.kin, ex.motion, ex.io.q, np.zeros(7), ex.knot_rate)
    assert move.duration_s == pytest.approx(same.duration_s) and np.allclose(move.q, same.q)
    assert np.abs(move.qd[:, :6]).max() > 0.6 * float(ex.motion["joint_speed"]["pen_up_rad_s"])


@pytest.mark.parametrize("top_down", [True, False])
def test_the_locate_looks_down_on_the_page_near_the_pen_down_pose_before_any_oblique_view(setup, monkeypatch, top_down):
    """2026-10-21: views 35-40 deg off from 0.18-0.22 m saw a forearm print's 12 mm markers 20-35 px and squashed and
    read none (the owner keeps the print facing up). Each view is sought first straight down (inspect.top_down: the
    page levelled, within 10 deg of the base's z from 0.19 m), and only when none is reached obliquely as before."""
    from tatbot_session import inspect as ins

    ex, grids = setup.ex, []
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "size_m": [0.045, 0.063], "pattern_id": "p"}}
    _locate_fakes(ex, monkeypatch, lambda az: None)

    def aim(kin, solve, page, xy, q, *, azimuths=None, **k):
        grids.append({key: k[key] for key in ins.TOP_DOWN if key in k})
        return (np.zeros(7), {"azimuth_deg": 200.0, "tilt_deg": 5.0}) if top_down == bool(grids[-1]) else None
    monkeypatch.setattr(ins, "aim", aim)
    tilted = geometry.rpy_matrix([0.3, 0.0, 0.1], [0.15, 0.0, 0.4])
    _, record = ex.wrist_locate(tilted, {"poses": 1, "wrist_floor_m": 0.002, "wrist_floor_rad": 0.01})
    assert len(record["views"]) == 1 and grids[0] == ins.TOP_DOWN and grids[1:] == ([] if top_down else [{}])


def test_the_locates_views_are_chosen_by_the_markers_they_see_past_the_tool(setup, monkeypatch):
    """2026-10-22: the locate's views put the silicone tile behind the tool; chosen by the markers they see
    (inspect.markers_seen, the fitted tool's footprint), straight down from as high as the print's smallest marker
    keeps its size and looking aside of it, three views over it saw 22-23 of its 23 where two 50 deg views saw 14-18."""
    from tatbot_session import inspect as ins

    ex, calls = setup.ex, []
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "size_m": [0.045, 0.063], "pattern_id": "p"}}
    _locate_fakes(ex, monkeypatch, lambda az: None)
    tile = SimpleNamespace(ids=[0], corners={0: np.zeros((4, 2))}, sides_m={0: 0.018})
    monkeypatch.setattr(ins, "print_artwork", lambda pattern: tile)
    monkeypatch.setattr(ins, "tool_footprint", lambda *a: np.zeros((720, 1280), bool))
    monkeypatch.setattr(ex_mod.config, "tool_datasheet", lambda repo, tool: SimpleNamespace(profile=[]))
    cam = ex.wrist_camera("")
    monkeypatch.setattr(ex, "wrist_camera", lambda serial: SimpleNamespace(**vars(cam), k=np.diag([651.0, 651.0, 1.0])))

    def aim(kin, solve, page, xy, q, *, azimuths=None, **k):
        calls.append(k)
        return (np.zeros(7), {"azimuth_deg": 200.0, "tilt_deg": 5.0}) if len(calls) == 1 else None
    monkeypatch.setattr(ins, "aim", aim)
    ex.wrist_locate(np.eye(4), {"poses": 2, "wrist_floor_m": 0.002, "wrist_floor_rad": 0.01})
    down, oblique = calls[0], calls[2]
    assert callable(down["see"]) and down["around"] == ins.AROUND and down["distances_m"] == (0.19, 0.27)
    assert callable(oblique["see"]) and "around" not in oblique and "distances_m" not in oblique


@pytest.mark.parametrize("case", ["read", "exhausted", "no depth", "overhead pose"])
def test_a_blind_locate_reading_no_marker_searches_the_skin_the_overhead_depth_reads(setup, monkeypatch, case):
    """2026-10-21, a forearm print inked over: the overhead read no marker, the blind locate's two views over the last
    page read none, and the set-up refused. Its views reading nothing, a locate the overhead placed no page for searches
    on over the skin the overhead's depth reads (taken at rest): a view over each search point with skin under it, its
    page raised over that skin (and its way judged on that page), until one reads the print's markers, which place the
    page as any view's do. A search that reads none, or no depth to aim it, leaves the last page standing as before (the
    set-up then refuses as before); a locate about an overhead pose searches nothing."""
    from tatbot_session import inspect as ins

    ex = setup.ex
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "size_m": [0.045, 0.063], "pattern_id": "p"}}
    _locate_fakes(ex, monkeypatch, lambda az: 200.0)
    planned, judged, read, said = [], [], [], []

    def aim(kin, solve, page, xy, q, *, azimuths=None, **k):
        planned.append((np.array(page), [float(v) for v in xy]))
        return np.zeros(7), {"azimuth_deg": 200.0, "tilt_deg": 40.0}

    sol = {"tx_m": 0.002, "ty_m": -0.041, "theta_rad": 0.03, "sigma": [0.001, 0.001, 0.005], "sides": []}

    def view_read(*a):
        read.append(a[5])
        hit = case == "read" and len(read) == 4   # the 2nd search view
        return np.zeros((4, 4, 3), np.uint8), (sol if hit else None), "" if hit else "no marker of the print read"

    x, y = np.meshgrid(np.arange(-0.06, 0.061, 0.002), np.arange(0.0, 0.081, 0.002))   # skin 4 mm up, toward +y
    skin = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.004)]
    monkeypatch.setattr(ins, "aim", aim)
    monkeypatch.setattr(ex, "_view_read", view_read)
    monkeypatch.setattr(ex, "lowest_over_page", lambda traj, page=None: judged.append(page) or 1.0)
    monkeypatch.setattr(ex, "event", lambda kind, text, *a: said.append(text))
    monkeypatch.setattr(ex, "_overhead_skin", lambda page, ledger=None: "no reply" if case == "no depth" else
                        pytest.fail("searched about an overhead pose") if case == "overhead pose" else skin)
    prior = geometry.rpy_matrix([0.32, 0.05, 0.07], [0.0, 0.0, 0.4])
    cfg = {"poses": 2, "wrist_floor_m": 0.002, "wrist_floor_rad": 0.01, "overhead_sigma_m": 0.005,
           "overhead_sigma_rad": 0.02}
    page, record = ex.wrist_locate(prior, cfg, alone=case != "overhead pose")
    if case in ("no depth", "overhead pose"):
        assert len(record["views"]) == 2 and "search" not in record and np.allclose(page, prior)
        assert ("wrist search: no overhead depth to aim it (no reply)" in said) == (case == "no depth")
        return
    aims = record["search"]
    assert len(aims) == 9 and np.allclose(aims[0]["xy"], [0.0, 0.04]) and all(a["rise_m"] == pytest.approx(0.012)
                                                                               for a in aims)   # centred on the skin
    assert "wrist search: the overhead's depth reads skin under 9 of 9 points about the last page, the views raised " \
           "up to 12.0 mm over it: 9 more locate views" in said
    for (aimed, xy), at in zip(planned[2:], aims, strict=False):   # each search view over its own skin, raised
        assert xy == at["xy"] and np.allclose(aimed, prior @ geometry.translation([0.0, 0.0, at["rise_m"]]))
    assert judged[:2] == [None, None] and all(np.allclose(j, a) for j, (a, _) in zip(judged[2:], planned[2:], strict=True))
    assert read[2][0][3] == pytest.approx(0.0315 + 0.010 + 0.04)   # the print read as far as it may have moved
    if case == "exhausted":
        assert len(record["views"]) == 11 and np.allclose(page, prior) and record["wrist"] is None
        assert said[-1].startswith("wrist locate: 11 views of the print reached, none read its markers; the last page")
        return
    assert len(record["views"]) == 4 and record["views"][3]["search"] == aims[1]
    assert abs(aims[1]["xy"][0]) == pytest.approx(0.0325) and aims[1]["xy"][1] == pytest.approx(0.04)   # beside it
    assert "wrist search: view 3 over the skin {:+.0f} +40 mm from the last page read the print's markers".format(
        aims[1]["xy"][0] * 1e3) in said
    assert np.allclose(page, prior @ np.linalg.inv(ins.correction_matrix(0.002, -0.041, 0.03)))


def test_the_search_reads_the_overheads_depth_in_the_last_pages_frame_or_says_why_not(setup, monkeypatch):
    """The overhead's capture (the D555 owner's, over the bus) deprojected, carried into the arm's base by the world
    transform and into the page's frame; without a capture or a transform, why, and no search."""
    from tatbot_bridge import capture

    ex, closed = setup.ex, []
    intrinsics = {"schema": "tatbot.camera-intrinsics/1", "fx": 50.0, "fy": 50.0, "ppx": 30.0, "ppy": 20.0}
    shot = {"depth_m": np.ones((40, 60), np.float32), "metadata": {"attributes": {"intrinsics": json.dumps(intrinsics)}}}

    class Camera:
        def __init__(self, repo):
            pass

        def capture(self, after_ns):
            return shot

        def close(self):
            closed.append(True)
    monkeypatch.setattr(capture, "Camera", Camera)
    cleared = []   # the arm at rest first, out of the overhead's view (2026-10-21: it stood over the piece)
    monkeypatch.setattr(ex, "clear_view", lambda ledger=None: cleared.append(len(closed)))
    down = geometry.rpy_matrix([0.1, 0.0, 1.2], [np.pi, 0.0, 0.0])   # looking straight down from 1.2 m
    monkeypatch.setattr(ex.node, "base_from_world", lambda arm: down, raising=False)
    points = ex._overhead_skin(geometry.translation([0.1, 0.0, 0.19]))
    assert closed and points.shape == (600, 3) and np.allclose(points[:, 2], 0.01)
    assert np.allclose(points[:, :2].mean(axis=0), [-0.02, 0.02])   # every second pixel's mean: one pixel off centre
    monkeypatch.setattr(ex.node, "base_from_world", lambda arm: None, raising=False)
    assert ex._overhead_skin(np.eye(4)) == "no transform to the arm's base"
    monkeypatch.setattr(Camera, "capture", lambda self, after_ns: (_ for _ in ()).throw(RuntimeError("no reply")))
    assert ex._overhead_skin(np.eye(4)) == "no reply" and len(closed) == 3 and cleared == [0, 1, 2]


@pytest.mark.parametrize("off_mm, deg, views, aside", [(34.9, 0.0, 2, True), (0.0, 7.7, 2, True),
                                                     (5.0, 1.0, 2, False), (34.9, 7.7, 1, False)])
def test_the_locate_sets_aside_an_overhead_pose_two_wrist_views_read_past_three_sigmas(setup, monkeypatch, off_mm, deg,
                                                                                      views, aside):
    """2026-10-20, a resume: the overhead's first pose came while its poses jumped 35-80 mm frame to frame; two wrist
    views read the print 34.9 mm, 7.7 deg from it, and the fused page stood 7 mm off the print (the holds' 7.8 mm
    shift then came within 1.2 mm of their curved limit). Past three sigmas of the two (13.4 mm or 5.4 deg here) by two
    views or more the wrist's own estimate places the page; nearer, or by one view, the two are fused as before."""
    from tatbot_session import inspect as ins

    ex = setup.ex
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "size_m": [0.045, 0.063], "pattern_id": "p"}}
    cfg = {"poses": views, "wrist_floor_m": 0.002, "wrist_floor_rad": 0.0175, "overhead_sigma_m": 0.004,
           "overhead_sigma_rad": 0.026}
    _locate_fakes(ex, monkeypatch, lambda az: 200.0 if az is None else az[0])
    read = (-off_mm / 1e3, 0.0, -np.radians(deg))
    sol = {"tx_m": read[0], "ty_m": read[1], "theta_rad": read[2], "sigma": [0.001, 0.001, 0.005], "sides": "lr"}
    monkeypatch.setattr(ex, "_view_read", lambda *a: (np.zeros((4, 4, 3), np.uint8), dict(sol), ""))
    prior = geometry.rpy_matrix([0.32, 0.04, 0.003], [0.0, 0.0, -1.55])
    page, record = ex.wrist_locate(prior, cfg)
    fused = read if aside else [ins.fuse([((0.0, 0.0, 0.0), (0.004, 0.004, 0.026)),
                                          (read, (0.002, 0.002, 0.0175))])[k] for k in ("tx_m", "ty_m", "theta_rad")]
    assert np.allclose(page, prior @ np.linalg.inv(ins.correction_matrix(*fused)))
    assert record["fused"]["overhead_set_aside"] is aside and len(record["views"]) == views


def test_a_setup_reads_by_one_wrist_camera_kept_open_and_writes_its_frames_behind(setup, monkeypatch, tmp_path):
    """Each gauge reading opened the wrist D405 (~2 s with 30 frames of auto exposure) and wrote its 50 files before
    its step was decided (1.0-1.3 s), at each of a setup's 14-23 readings (2026-10-20). The setup's camera opens at its
    first reading and stays open, each grab taking frames from now; the files are written behind it in order, as they
    were, and settled before anything reads them; a failed grab closes it and the next reading opens one afresh; the
    tracker takes it (Tracker.start), else the goal's end closes it."""
    ex = setup.ex
    opened, fail = [], []

    class Cam:
        def __init__(self):
            self.closed, self.fresh, self.depth_metadata = False, [], {"depth_units_m": 1e-4}
            opened.append(self)

        def grab_both(self, n, observe=None, fresh=False):
            if fail:
                raise RuntimeError("Frame didn't arrive within 5000")
            self.fresh.append(fresh)
            self.raw_depth = [np.full((2, 2), 7 + k, np.uint16) for k in range(n)]
            return [np.full((2, 2, 3), 3 + k, np.uint8) for k in range(n)], []

        def close(self):
            self.closed = True
    monkeypatch.setattr(ex, "wrist_camera", lambda serial: Cam())
    ex.run_dir, ex._sensing = tmp_path, SimpleNamespace(wrist="real")
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "locate": {"wrist": True}}}
    q = np.arange(7.0)
    for name in ("gauge-0", "gauge-0", "gauge-1"):
        colors, depths, meta = ex._wrist_frames(name, q, 3, every=True)
        assert len(colors) == len(depths) == 3 and meta == {"depth_units_m": 1e-4}
    assert len(opened) == 1 and opened[0].fresh == [True] * 3 and not opened[0].closed
    ex._settle()
    saved = np.load(tmp_path / "touches" / "gauge-1.npz")
    assert (saved["depth_raw"] == 7).all() and (saved["q"] == q).all() and json.loads(str(saved["meta"])) == meta
    assert (tmp_path / "touches" / "gauge-1.png").exists() and (tmp_path / "touches" / "gauge-1" / "2.jpg").exists()
    assert (np.load(tmp_path / "touches" / "gauge-1" / "2.npz")["depth_raw"] == 9).all()
    fail.append(True)
    assert ex._wrist_frames("gauge-2", q, 3) is None and opened[0].closed and ex.take_camera() is None
    fail.clear()
    ex._wrist_frames("gauge-2", q, 3)
    assert len(opened) == 2 and ex.take_camera() is opened[1]             # the tracker's to close now
    ex._wrist_frames("gauge-3", q, 1)
    ex.begin_goal()
    ex.end_goal()
    assert opened[2].closed and ex.take_camera() is None and (tmp_path / "touches" / "gauge-3.npz").exists()


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
        assert ex.camera is not None  # set on line above
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


def test_a_carried_ready_heading_turned_far_from_the_dips_is_searched_again(tmp_path, monkeypatch):
    """2026-10-20: strokes went at -105 deg and every dip turned the wrist to the station touch's -30 and back, ~36 s
    a dip. A new page prefers the dips' heading (ready.pen_heading's prefer); a ready pose carried from a run before
    that, turned past 45 deg from it, is not carried. This run's own recorded one, and a tool the touch does not
    speak for, keep today's."""
    pytest.importorskip("pinocchio")
    from tatbot_description import robot_description
    from tatbot_motion import station
    from tatbot_motion.clik import Kinematics

    kin = Kinematics(robot_description(arms=("right",)), "right")
    page = np.eye(4)
    page[:3, :3] = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]   # the print's x along the base's -y
    page[:3, 3] = [0.296, 0.013, 0.003]

    def heading(yaw_deg):
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        return np.array([[c, s, 0.0, 0.0], [s, -c, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    touch = SimpleNamespace(tool_id="lutin-3rl-bugpin", rotation=heading(-30.0)[:3, :3])
    monkeypatch.setattr(station, "load_touch", lambda arm: touch)
    monkeypatch.setattr(ex_mod.config, "fitted_tool", lambda repo, arm: "lutin-3rl-bugpin")
    monkeypatch.setattr(ex_mod.ready, "PREFER_MARGIN_RAD", 0.40)   # here -30 keeps 0.56 rad, -45 0.36, -105 0.17

    def chosen(record):
        ex = ex_mod.ArmExecutor(FakeNode(np.eye(4)), FakeIO(), kin, MOTION, STACK, tmp_path)
        ex.camera, ex.correction, ex.page_record = page.copy(), np.eye(4), record
        h = ex.pen_heading()
        return round(float(np.degrees(np.arctan2(h[1, 0], h[0, 0]))))

    carried = {"ready": {"q": [0.0] * 7, "heading": heading(-105.0).tolist()}}
    assert chosen({"carried": carried}) == -30                             # searched again, the dips' first
    assert chosen({"carried": {"ready": {**carried["ready"], "heading": heading(-60.0).tolist()}}}) == -60
    assert chosen({"ready": carried["ready"]}) == -105                     # this run's: its holds were made there
    touch.tool_id = "picosecond-laser-pen"                                 # another machine's touch
    assert chosen({"carried": carried}) == -105


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


class ConstantField:
    """A known residual, the same over the page: the newest of `samples` (t, u, r); what the tracker's filter
    (page_filter.PageFilter) gives the executor: predict, snapshot, corrected, samples."""

    def __init__(self, samples=()):
        self.samples = list(samples)

    def predict(self, points, follow=None):
        p = np.atleast_2d(np.asarray(points, float))
        r = np.asarray(self.samples[-1][2], float) if self.samples else np.zeros(2)
        return np.tile(r, (len(p), 1)), np.ones(len(p))

    def snapshot(self, follow=None):
        return ConstantField(self.samples)

    def corrected(self, op):
        pts = np.asarray(op["points_m"], float)
        corr = self.predict(pts)[0]
        return {**op, "points_m": (pts - corr).tolist()}, corr


class StubTracker:
    """Stencil tracking with a known residual field and no camera."""
    correcting = True

    def __init__(self, residual):
        self.field, self.snaps, self.lifts, self.on_lift = ConstantField([(0.0, np.zeros(2), residual)]), 0, [], None
        self.hovers, self.verdicts, self.residuals, self.off, self.restarts = [], [], [], None, 0
        self.held, self.prints = [], []

    def print_moved(self, op_id):
        """track.Tracker.print_moved over `held` (the holds' readings) and `prints` (the hover's gauge misses)."""
        from tatbot_session import track

        return [track.print_offset(self.held, fit, (0.004, -0.004)) for fit in self.prints] if self.held else []

    def restart(self):
        """The field back to its first sample, the setup's (the stub's; the tracker's filter keeps its own)."""
        self.restarts += 1
        self.field.samples = self.field.samples[:1]

    far = None

    def off_plan(self, limit_m=None, count=None):
        """`off` (`far` for ArmExecutor.OFF_SETUP's limit): (mean, [(op id, offset)]) when the last strokes ended off
        the plan, else None."""
        return self.off if limit_m is None else self.far

    def snapshot(self):
        self.snaps += 1
        return self.field.snapshot()

    def after_lift(self, op):
        self.lifts.append(op["id"])
        if self.on_lift is not None:
            self.on_lift(self)

    def while_drawing(self, op, arc_m):
        """The stroke's tick while it draws (track.Tracker.while_drawing): no frame here."""

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


def test_a_sheet_that_moves_under_a_stroke_stops_it_and_it_goes_on_from_its_arc(setup):
    """A skin pushed ~30 mm round a bottle mid-stroke (2026-10-24): the filter took the move within 5 s and the stroke
    drew on the old place 8 s more. Once the correction at the pen moves past page.track.stop_m from the one the stroke
    draws with, the stroke stops (the machine off), lifts with no decision asked, and goes on from its arc planned with
    the newest correction."""
    import copy

    ex, io = setup.ex, setup.io
    ex.stack = copy.deepcopy(ex.stack)
    ex.stack["page"].setdefault("track", {})["stop_m"] = 0.005
    ex.tracker = stub = StubTracker([0.001, 0.0])
    io.at_tick[3] = lambda: stub.field.samples.append((time.monotonic(), np.zeros(2), np.array([0.009, 0.0])))
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    rows = setup.ledger.rows()
    assert [r["event"] for r in rows] == ["sent", "aborted", "sent", "done"]
    assert rows[1]["reason"].startswith("the sheet moved: the print's correction at the pen moved 8.0 mm")
    assert 0.0 < rows[1]["arc_m"] < 0.02 and rows[2]["arc_m"] == pytest.approx(rows[1]["arc_m"])
    assert any("stopping it, lifted" in text for _, _, text in setup.node.events)
    assert setup.lifts == [MOTION["touch"]["lift_m"]]
    moved = np.asarray(setup.calls[1]["points"]) - np.asarray(LINE["points_m"])
    assert np.allclose(moved, [-0.009, 0.0])                             # the rest drawn on the moved sheet
    ex.stack["page"]["track"]["stop_m"] = 0.0
    assert ex._moved_under(LINE, 0.005) is None                         # 0: never


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


def _slid_unread(setup, verdicts, slide_m, turn_deg=0.0):
    """Hover stops correcting on the real arm, the field carrying 4 mm on its setup's 1 mm, the hover's verdicts
    scripted, and the overhead's three newest poses since setup slid slide_m in y and turned turn_deg (under the
    reach: no set-up)."""
    ex, node = setup.ex, setup.node
    ex.tracker = StubTracker([0.001, 0.0])
    ex.tracker.field.samples.append((time.monotonic(), np.zeros(2), np.array([0.004, 0.0])))
    _hovering(ex)
    ex.stack["hardware"], ex.stack["page"]["track"]["mode"] = "real", "correct"
    ex.tracker.verdicts = list(verdicts)
    slid, turn = np.eye(4), np.radians(turn_deg)
    slid[:2, :2] = [[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]
    slid[1, 3] = slide_m
    node.measured = [(time.monotonic() + 1.0, slid)] * 3
    return slid


def test_a_hover_reading_the_field_did_not_take_lets_the_overheads_slide_move_the_page(setup, monkeypatch):
    """2026-10-08: an untaped half-skin slid ~14 mm under five 1011RL strokes. The overhead measured it, the wrist
    read the inked print 4-9 mm off it and every reading was refused or held, and the field kept its 2 mm. Now the
    overhead's slide moves the page, and the field starts again from the setup's holds: its later samples carry the
    slide the page now follows."""
    ex, io, node = setup.ex, setup.io, setup.node
    monkeypatch.setattr(ex, "workspace", lambda: {"tool": "x"})
    slid = _slid_unread(setup, [None, None], 0.005)                     # the hover and its second view
    io.script = [("ok", 1, {}, None), ("ok", 1, {}, None), ("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    pages = [row for row in setup.ledger.rows() if row.get("event") == "page"]
    assert len(pages) == 1 and pages[0]["page"]["followed"]
    assert np.allclose(ex.page_overhead(), slid)
    assert ex.tracker.restarts == 1
    assert np.allclose(setup.calls[0]["points"], np.asarray(LINE["points_m"]) - [0.001, 0.0])   # the setup's 1 mm
    assert any("the page follows it; the wrist's reading at the hover was not taken" in text
               for _, _, text in node.events)


@pytest.mark.parametrize("verdicts, follows", [(["held", "held"], True), (["added"], False), (["held", "moved"], False)])
def test_a_jump_no_second_view_confirmed_lets_the_overhead_move_the_page_and_a_taken_reading_does_not(
        setup, monkeypatch, verdicts, follows):
    ex, io = setup.ex, setup.io
    monkeypatch.setattr(ex, "workspace", lambda: {"tool": "x"})
    _slid_unread(setup, verdicts, 0.005)
    io.script = [("ok", 1, {}, None)] * len(verdicts) + [("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert len(ex.tracker.hovers) == len(verdicts)
    assert sum(row.get("event") == "page" for row in setup.ledger.rows()) == int(follows) == ex.tracker.restarts


@pytest.mark.parametrize("slide_m, turn_deg", [(0.0008, 0.0), (0.0003, 2.0)])
def test_an_untaken_hover_reading_leaves_the_page_where_the_overhead_sees_the_sheet_still(setup, slide_m, turn_deg):
    """The overhead's noise: under 1.5 mm, and its yaw's 1-2 deg flicker, which follows only past the reach (2.9 deg)
    here; replayed over 32 flat tracked runs, the eight follows it caused moved still pages 0.2-1.4 mm."""
    ex, io = setup.ex, setup.io
    _slid_unread(setup, [None], slide_m, turn_deg)
    io.script = [("ok", 1, {}, None), ("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert not any(row.get("event") == "page" for row in setup.ledger.rows()) and ex.tracker.restarts == 0
    assert np.allclose(setup.calls[0]["points"][0], np.asarray(LINE["points_m"][0]) - [0.004, 0.0], atol=6e-4)


CONT = {"op": "stroke", "id": "s0001", "closed": False, "continues": True, "points_m": [[0.02, 0.0], [0.04, 0.0]]}


def test_no_hover_stop_for_a_stroke_that_joins_its_path_where_the_tip_stands_and_one_for_every_descent(setup):
    """A continuation straight on from its chain joins pen-down (plan_op's own test, tatbot_motion.joins): no hover.
    Every stroke the planner would lower the pen into is measured first: one whose tip stands on the page off its
    start, a continuation 50 mm up after a dip, or one whose page moved 2 mm off the tip."""
    ex, io = setup.ex, setup.io
    ex.tracker = StubTracker([0.0, 0.0])
    _hovering(ex)
    io.script = [("ok", 1, {}, None), ("ok", 10, {}, None), ("ok", 10, {}, None)]
    assert draw_strokes(ex, {"ops": [LINE, CONT]}, setup.ledger, touch=False) == "complete"
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover"]       # the chain's continuation joined, unmeasured
    program, ex.tracker.hovers = {"ops": [LINE, CONT]}, []
    for op, q, from_arc, page in ((CONT, [0.02, 0.0, 0.0], 0.0, np.eye(4)),                     # on its start: joins
                                  (CONT, [0.03, 0.0, 0.0], 0.01, np.eye(4)),                    # on its arc: joins
                                  (LINE, [0.0, 0.0, 0.0], 0.0, np.eye(4)),                      # lifts, goes down
                                  (CONT, [0.03, 0.0, 0.05], 0.01, np.eye(4)),                   # back from a dip
                                  (CONT, [0.02, 0.0, 0.0], 0.0, geometry.translation([0.002, 0.0, 0.0]))):   # followed
        io.q, ex.camera = np.array([*q, 0.0, 0.0, 0.0, 0.0]), page
        ex._hover_stop(program, op, from_arc, setup.ledger, 1)
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover", "s0001-hover", "s0001-hover"]
    assert np.allclose([h[2] for h in ex.tracker.hovers], [[0.0, 0.0], [0.03, 0.0], [0.02, 0.0]])


@pytest.mark.parametrize("risen", [False, True])
def test_a_continue_after_a_latch_lifts_and_measures_the_skin_again_before_the_needle_goes_back(setup, monkeypatch, risen):
    """An e-stop or a latch mid-stroke leaves the needle at the skin, and a person may move while the arm waits: continue
    rejoined the stroke pen-down where it stopped, unmeasured (the hover stop skipped a pen within final_m of the page,
    and every pen-down continuation). Now the pen lifts along the page's normal first, the overhead's markers are read
    (a skin risen 9 mm sets the page up again), the hover measures over where the stroke goes on, a continuation's too,
    and only then the needle goes down, from its arc (re-approached, never joined)."""
    ex, io, node = setup.ex, setup.io, setup.node
    ex.tracker = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.stack["page"]["source"], ex.workspace = "stencil", dict
    still, ex._printed = _tile_seen()
    node.marks = [(time.monotonic() + 10.0, still)] * 3
    order, real_setup, again = [], ex.setup_page, StubTracker([0.0, 0.0])

    def setup_page(program, touch, ledger):
        order.append("set up")
        real_setup(program, touch, ledger)
    ex.setup_page = setup_page
    monkeypatch.setattr(ex_mod.track.Tracker, "start", classmethod(lambda cls, ex, since=0.0, centre=None: again))
    monkeypatch.setattr(ex, "lift_off", lambda what, ledger=None, real=ex.lift_off: (order.append("lift"),
                                                                                     real(what, ledger)))
    cont = {"op": "stroke", "id": "s0001", "closed": False, "continues": True, "points_m": [[0.02, 0.0], [0.04, 0.0]]}
    estop = {"latched": 1.0, "latch_reason": 1.0, "estop_ok": 1.0}
    io.script = [("ok", 1, {}, None), ("ok", 10, {}, [0.02, 0, 0, 0, 0, 0, 0]),       # s0000, pen-down into s0001
                 ("latched", 6, estop, [0.0321, 0, 0, 0, 0, 0, 0]),                     # s0001 stopped at the skin
                 ("ok", 1, {}, None), ("ok", 1, {}, None), ("ok", 10, {}, UP)]         # lift, hover, the rest

    def operator():
        deadline = time.monotonic() + 5.0
        while ex.waiting is None and time.monotonic() < deadline:
            time.sleep(0.002)
        if risen:   # the person's arm 9 mm up while the arm waits (past WATCH_RISE_M, 4 mm)
            node.camera = geometry.translation([0.0, 0.0, 0.009])
            node.marks = [(time.monotonic() + 20.0, _tile_seen(lift=0.009)[0])] * 5
        replies.append(ex.decide(rules.CONTINUE, timeout=5.0))
    replies, thread = [], threading.Thread(target=operator, daemon=True)
    thread.start()
    assert draw_strokes(ex, {"ops": [LINE, cont]}, setup.ledger, touch=False) == "complete"
    thread.join(5.0)
    tracker = again if risen else ex.tracker
    assert replies == [(True, "")] and order[:2] == ["set up", "lift"]            # the draw's setup, then the lift
    assert order[2:] == (["lift", "set up", "lift"] if risen else []) and np.allclose(ex.used(), node.camera)
    moves = [("hover" if getattr(t, "info", {}).get("hover") else "stroke" if "trim" in getattr(t, "info", {})
              else "lift") for t in io.executed]
    assert moves == ["hover", "stroke", "stroke", "lift", "hover", "stroke"]   # lifted, measured, then down
    assert [h[0] for h in tracker.hovers][-1] == "s0001-hover"                    # a continuation measured too
    assert np.allclose(tracker.hovers[-1][2], [0.032, 0.0], atol=2e-4)            # over where it goes on
    sent = [row for row in setup.ledger.rows() if row["event"] == "sent"]
    assert sent[-1]["op"] == "s0001" and sent[-1]["arc_m"] == pytest.approx(0.012, abs=2e-4)
    assert setup.calls[-1]["from_arc"] == pytest.approx(0.012, abs=2e-4) and setup.calls[-1]["pen_down"]
    assert setup.lifts == [MOTION["touch"]["lift_m"]]


@pytest.mark.parametrize("stop", ["estop", "snag"])
def test_a_stop_in_a_strokes_last_half_millimetre_lifts_and_measures_its_continuation_before_it_goes_down(
        riding, monkeypatch, stop):
    """A stroke stopped within resume_reapproach_m of its end is drawn (done), and with a continuation next it neither
    lifted nor was the continuation measured: after an e-stop continue joined it pen-down where the needle stood, and
    after a snag trip's withdraw it went straight back down from 10 mm (the 2026-10-20 review; the self-portrait has six
    continuations, a quarter of its length). Now it lifts, and the continuation's hover measures the skin first; it is
    planned with the newest field, that hover's reading in it, not its chain's."""
    ex, io = riding.ex, riding.io
    _pen_down(monkeypatch)
    ex.tracker = StubTracker([0.0, 0.0])
    ex.tracker.residuals = [[0.0, 0.0], [0.001, 0.0]]                          # s0001's hover reads the print 1 mm off
    _hovering(ex)
    ex.motion = {**RIDE, "approach": ex.motion["approach"]}
    at = [0.0198, 0.0, -0.0005, 0, 0, 0, 0]                                    # 0.2 mm short of s0000's end, in
    halt = {"latched": 1.0, "latch_reason": 1.0, "estop_ok": 1.0} if stop == "estop" else SNAG
    io.script = [("ok", 1, {}, None), ("latched", 10, halt, at)] + [("ok", 1, {}, None)] * 2 + [("ok", 10, {}, UP)]
    operator = Operator(ex, *([rules.CONTINUE] if stop == "estop" else []))
    assert draw_strokes(ex, {"ops": [LINE, CONT], "tool": PROGRAM_TOOL}, riding.ledger, touch=False) == "complete"
    assert operator.replies() == ([(True, "")] if stop == "estop" else [])
    moves = [("hover" if getattr(t, "info", {}).get("hover") else "stroke" if "trim" in getattr(t, "info", {})
              else "lift") for t in io.executed]
    assert moves == ["hover", "stroke", "lift", "hover", "stroke"]             # out, measured, then down
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover", "s0001-hover"]
    assert riding.lifts == [MOTION["approach"]["standoff_m"] if stop == "snag" else MOTION["touch"]["lift_m"]]
    assert riding.calls[-1]["op"] == "s0001" and np.allclose(riding.calls[-1]["seed"][:3], [0.02, 0.0, 0.01])
    moved = ex.tracker.field.snapshot().predict(CONT["points_m"])[0]
    assert np.abs(moved).max() > 5e-4 and np.allclose(riding.calls[-1]["points"], np.asarray(CONT["points_m"]) - moved)
    assert [r["event"] for r in riding.ledger.rows()][-2:] == ["sent", "done"]


def _ended_off(ex, monkeypatch):
    """Stencil tracking correcting whose last two strokes ended ~2 mm off the plan one way (2026-10-08's s0157 and
    s0158); the page set up again gives a fresh tracker. (the first tracker, the second, the setups made)"""
    ex.tracker = first = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.stack["hardware"], ex.stack["page"]["track"]["mode"] = "real", "correct"
    ends = [("s0157", np.array([-0.00100, -0.00209])), ("s0158", np.array([-0.00093, -0.00172]))]
    first.off = (np.mean([off for _, off in ends], axis=0), ends)
    pages, again = [], StubTracker([0.0, 0.0])
    monkeypatch.setattr(ex, "_new_page", lambda program, touch, ledger: pages.append(touch))
    monkeypatch.setattr(ex_mod.track.Tracker, "start", classmethod(lambda cls, ex, since=0.0, centre=None: again))
    return first, again, pages


def test_strokes_that_ended_off_the_plan_never_pause_and_only_far_ends_set_the_page_up_again(setup, monkeypatch):
    """2026-10-20: a forearm drifting 0.5 mm/min ended two strokes 2.1 and 1.6 mm off one way, the run paused for a
    decision and the person held still 13 min. Ends past off_plan_m are said and the field follows them, no pause;
    three past OFF_SETUP's 6 mm one way (3 mm until the owner doubled every stop, 2026-10-20) set the page up again,
    the pen lifted, and the stroke is measured on it."""
    ex, io, node = setup.ex, setup.io, setup.node
    first, again, pages = _ended_off(ex, monkeypatch)
    io.script = [("ok", 1, {}, None), ("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert not [text for kind, _, text in node.events if kind == "pause"] and ex.waiting is None
    said = [text for kind, _, text in node.events if "ended off the plan" in text]
    assert said and "s0157 2.3, s0158 2.0 mm; -1.0 -1.9 mm on average): the field follows them" in said[0]
    assert pages == [] and [r["event"] for r in setup.ledger.rows()] == ["sent", "done"]
    ex.tracker = first
    ends = [(f"s{k}", np.array([-0.0062, -0.0002])) for k in range(3)]
    first.far = (np.mean([off for _, off in ends], axis=0), ends)
    io.q = np.array(UP)
    ex._off_plan({"ops": [LINE]}, LINE, 0, setup.ledger)
    assert pages == [False] and ex.tracker is again and ex.waiting is None
    assert "s0 6.2, s1 6.2, s2 6.2 mm" in node.events[-1][2] and "each past 6 mm" in node.events[-1][2]


def test_strokes_that_ended_off_the_plan_move_nothing_without_correction(setup, monkeypatch):
    ex = setup.ex
    first, _, pages = _ended_off(ex, monkeypatch)
    first.far = first.off
    ex.tracker, first.correcting = first, False                          # log only: readings, no correction
    ex._off_plan({"ops": [LINE]}, LINE, 0, setup.ledger)
    assert ex.waiting is None and pages == []


def test_a_stroke_the_offset_carries_past_the_clear_area_sets_the_page_up_again_never_pausing(setup, monkeypatch):
    """2026-10-16: a misplaced page drew 10 mm left on a faint print. A stroke that the strokes' consistent offset
    would carry onto the stencil's border past OFF_CLEAR_M sets the page up again, whatever the tracking corrects, a
    second time too: no pause (owner, 2026-10-20: no keys for decisions)."""
    ex, node = setup.ex, setup.node
    first, _, pages = _ended_off(ex, monkeypatch)
    first.correcting = False
    first.off = (np.array([0.010, 0.0]), [("s0157", np.array([0.010, 0.0])), ("s0158", np.array([0.010, 0.0]))])
    for _ in range(2):
        ex.tracker = first
        ex._off_clear({"ops": [LINE], "page": {"clear_m": [0.04, 0.04]}}, LINE, 0, setup.ledger)
    assert pages == [False, False] and not any(kind == "pause" for kind, _, _ in node.events) and ex.waiting is None
    assert "carry this stroke 10.0 mm onto the stencil's border" in node.events[-1][2]
    ex.tracker = first
    first.off = (np.array([0.0015, 0.0]), first.off[1])                  # 1.5 mm onto it: within OFF_CLEAR_M
    ex._off_clear({"ops": [LINE], "page": {"clear_m": [0.04, 0.04]}}, LINE, 0, setup.ledger)
    first.off = (np.array([-0.010, 0.0]), first.off[1])                  # carried inside: nothing
    ex._off_clear({"ops": [LINE], "page": {"clear_m": [0.04, 0.04]}}, LINE, 0, setup.ledger)
    ex._off_clear({"ops": [LINE]}, LINE, 0, setup.ledger)                 # no clear area named: no check
    assert pages == [False, False]


def test_a_hover_that_finds_no_print_sets_the_page_up_again_when_the_overhead_saw_the_sheet_go(setup, monkeypatch):
    """A sheet slid past the wrist fit's reach: no stroke goes down blind from the old page."""
    ex, io, node = setup.ex, setup.io, setup.node
    ex.tracker = first = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.stack["hardware"], ex.stack["page"]["track"]["mode"] = "real", "correct"
    first.verdicts = [None, None]                                        # the hover and its second view
    moved = np.eye(4)
    moved[1, 3] = 0.05
    node.camera_page = lambda arm: (moved, {"source": "measured"})
    pages, again = [], StubTracker([0.0, 0.0])
    monkeypatch.setattr(ex, "_new_page", lambda program, touch, ledger: pages.append(touch))
    monkeypatch.setattr(ex_mod.track.Tracker, "start", classmethod(lambda cls, ex, since=0.0, centre=None: again))
    io.script = [("ok", 1, {}, None)] * 3 + [("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert [h[0] for h in first.hovers] == ["s0000-hover", "s0000-hover2"] and [h[0] for h in again.hovers] == ["s0000-rehover"]
    assert pages == [False] and ex.tracker is again
    assert any("setting the page up again" in text for _, _, text in node.events)


def test_a_hover_that_finds_no_print_sets_the_page_up_again_when_the_overhead_cannot_place_the_sheet(setup, monkeypatch):
    """The arm over the page hides it from the overhead: with no fresh pose, the page is set up again rather than
    the stroke drawn blind; a fresh pose within reach draws with the field."""
    ex, io, node = setup.ex, setup.io, setup.node
    ex.tracker = first = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.stack["hardware"], ex.stack["page"]["track"]["mode"] = "real", "correct"
    first.verdicts = [None, None]                                        # the hover and its second view
    node.camera_page = lambda arm: (np.eye(4), {"source": "lost", "measured_age_s": 30.0})
    pages, again = [], StubTracker([0.0, 0.0])
    monkeypatch.setattr(ex, "_new_page", lambda program, touch, ledger: pages.append(touch))
    monkeypatch.setattr(ex_mod.track.Tracker, "start", classmethod(lambda cls, ex, since=0.0, centre=None: again))
    io.script = [("ok", 1, {}, None)] * 3 + [("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert pages == [False]
    node.camera_page = lambda arm: (np.eye(4), {"source": "lost", "measured_age_s": 0.4})   # just hidden: in place
    ex.camera = np.eye(4)
    assert not ex._sheet_may_have_left()


@pytest.mark.parametrize("move, order", [({}, ["clear view", "ready"]), ({"dx": 0.008}, ["clear view", "set up"]),
                                         (None, ["clear view", "set up"])])
def test_a_hover_that_finds_no_print_with_the_markers_hidden_clears_the_view_first(setup, monkeypatch, move, order):
    """2026-10-20: a forearm was set up twice (~4.5 min each) because the arm over the print hid its markers from the
    overhead since the pen left; it had stayed within 0.7 mm. The arm clears the view first: markers read where the
    page is bring it back to measure the hover again, a sheet 8 mm off (past off_m, 7 mm) or still unread is set up
    again."""
    ex, node = setup.ex, setup.node
    ex.tracker = first = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.stack["hardware"], ex.stack["page"]["track"]["mode"] = "real", "correct"
    ex.camera = ex.overhead = np.eye(4)
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at, ex.inked_at, ex.clock = still, 0.0, 100.0, Ticking(100.0)
    node.marks, done = [(float(t), still) for t in range(40, 95)], []   # none since the pen left the page at 100 s
    monkeypatch.setattr(ex, "clear_view", lambda ledger=None: (done.append("clear view"),
                                                               move is not None and _read(node, 101, 104, **move)))
    monkeypatch.setattr(ex, "go_ready", lambda ledger=None: done.append("ready"))
    monkeypatch.setattr(ex, "_reacquire", lambda program, ledger, why, op_id="": done.append("set up"))
    first.verdicts = [None, "added"]                                     # the second view, then the hover again
    assert ex._unread_hover({"ops": [LINE]}, LINE, np.zeros(2), 0.0, setup.ledger) == ("added", True)
    assert done == order and [h[0] for h in first.hovers] == ["s0000-hover2", "s0000-rehover"]
    # waited 10 s for readings that never came; an 8 mm move read 3 s and no more waited out its 15 s, then set up
    assert ex.clock.t == (110.0 if move is None else 115.0 if move else 100.0)


@pytest.mark.parametrize("prints, again", [([[-0.0063, -0.0039, 0.0]], False),   # where the holds read the print
                                           ([[-0.0205, -0.0045, 0.0]], True),    # 15 mm on: the sheet moved
                                           ([], True)])                          # no marker either: as before
def test_a_hover_whose_gauge_read_no_surface_sets_the_page_up_again_only_for_a_moved_print(setup, monkeypatch,
                                                                                             prints, again):
    """2026-10-19, hovering 6 mm over a forearm print with the arm hiding it from the overhead: the gauge read no
    surface in 43 of 61 unread readings, and ten hovers set the page up again for a sheet that never moved. The
    hover's frames still read the print by their own depth, and a print where the gauge holds read it keeps the page;
    one 15 mm on, or none read, sets it up again as before."""
    ex, io, node = setup.ex, setup.io, setup.node
    ex.tracker = first = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.stack["hardware"], ex.stack["page"]["track"]["mode"] = "real", "correct"
    first.verdicts, first.held, first.prints = [None, None], [[-0.0055, -0.0045, 0.0]] * 3, prints   # the FK pose
    node.camera_page = lambda arm: (np.eye(4), {"source": "lost", "measured_age_s": 30.0})
    pages, second = [], StubTracker([0.0, 0.0])
    monkeypatch.setattr(ex, "_new_page", lambda program, touch, ledger: pages.append(touch))
    monkeypatch.setattr(ex_mod.track.Tracker, "start", classmethod(lambda cls, ex, since=0.0, centre=None: second))
    io.script = [("ok", 1, {}, None)] * (2 + again) + [("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert pages == [False] * again and (ex.tracker is second) == again
    stays = [text for _, _, text in node.events if "the page stays" in text]
    assert len(stays) == (not again)
    assert not stays or "the print reads 1.0 mm from where the gauge holds read it" in stays[0]


def test_a_hover_that_finds_no_marker_looks_again_before_setting_the_page_up(setup, monkeypatch):
    """An empty view is a view between the print's markers, not a sheet that left: the second view
    (5 mm toward the page centre) reads it, and the page is not set up again however the overhead stands."""
    ex, io, node = setup.ex, setup.io, setup.node
    ex.tracker = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.stack["hardware"], ex.stack["page"]["track"]["mode"] = "real", "correct"
    ex.tracker.verdicts = [None, "added"]
    node.camera_page = lambda arm: (np.eye(4), {"source": "lost", "measured_age_s": 30.0})
    monkeypatch.setattr(ex, "_new_page", lambda program, touch, ledger: pytest.fail("set up again"))
    io.script = [("ok", 1, {}, None), ("ok", 1, {}, None), ("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover", "s0000-hover2"]
    assert np.isclose(np.linalg.norm(np.subtract(ex.tracker.hovers[1][2], ex.tracker.hovers[0][2])), 0.005)


def test_overhead_poses_measured_since_the_pen_left_the_page_say_whether_the_sheet_left(setup):
    """2026-10-06: a half-skin's print reads poorly at the page's foot, and the 5 s freshness rule set it up again
    three times in 25 ops with the overhead's pose from the dip half a minute old. Poses since the last stroke
    vouch for the sheet, a majority either way; with none since, it may have left."""
    import copy

    ex, node = setup.ex, setup.node
    ex.stack = copy.deepcopy(STACK)
    ex.stack["hardware"], ex.stack["page"]["track"] = "real", {"mode": "correct"}
    ex.camera = ex.overhead = np.eye(4)
    node.camera_page = lambda arm: (np.eye(4), {"source": "lost", "measured_age_s": 30.0})
    ex.inked_at = time.monotonic()
    assert ex._sheet_may_have_left()                                     # nothing measured since the stroke
    gone = np.eye(4)
    gone[1, 3] = 0.03
    node.measured = [(time.monotonic() - 60.0, gone), (time.monotonic() + 1.0, np.eye(4))]   # the dip's view
    assert not ex._sheet_may_have_left()
    node.measured += [(time.monotonic() + 1.0, gone), (time.monotonic() + 1.0, gone)]
    assert ex._sheet_may_have_left()


def test_the_sheets_slide_is_the_median_of_the_overheads_poses_since_setup(setup):
    ex, node = setup.ex, setup.node
    ex.camera = ex.overhead = np.eye(4)
    ex.overhead_at = time.monotonic()
    assert ex.sheet_slide() is None                                      # nothing measured since setup
    slid, stray = np.eye(4), np.eye(4)
    slid[:2, 3] = [-0.011, 0.004]
    stray[:2, 3] = [0.03, 0.03]
    node.measured = [(time.monotonic() + 1.0, pose) for pose in (slid, stray, slid)]
    assert np.allclose(ex.sheet_slide(), slid)


def test_without_hover_stops_the_page_follows_the_overheads_slide_before_each_stroke(setup, monkeypatch):
    """2026-10-06: a half-skin spun ~1 deg a stroke late in a draw, and the reach (2.9 deg) caught it three strokes late
    each time at ~2.5 min a set-up. The page now follows the measured slide at 1.5 mm or ~1 deg, and noise does not
    move it."""
    import copy

    ex, io, node = setup.ex, setup.io, setup.node
    ex.tracker = StubTracker([0.0, 0.0])
    ex.tracker.correcting = False
    ex.stack = copy.deepcopy(STACK)
    ex.stack["hardware"], ex.stack["page"]["track"] = "real", {"mode": "correct", "hover_stop": True}
    turn = np.radians(2.0)
    spun = np.eye(4)
    spun[:2, :2] = [[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]
    spun[:2, 3] = [0.0005, -0.0003]
    noisy = np.eye(4)
    noisy[:2, 3] = [0.0004, 0.0003]
    seen = iter([[noisy, np.eye(4), noisy], [spun, spun, spun]])
    ex.tracker.on_lift = lambda stub: node.measured.extend((time.monotonic(), p) for p in next(seen, []))
    monkeypatch.setattr(ex, "_new_page", lambda program, touch, ledger: pytest.fail("no set-up for a slide"))
    monkeypatch.setattr(ex, "workspace", lambda: {"tool": "x"})
    line3 = {**LINE2, "id": "s0002", "points_m": [[0.0, 0.02], [0.02, 0.02]]}
    io.script = [("ok", 10, {}, None)] * 3
    assert draw_strokes(ex, {"ops": [LINE, LINE2, line3]}, setup.ledger, touch=False) == "complete"
    pages = [row for row in setup.ledger.rows() if row.get("event") == "page"]
    assert len(pages) == 1 and pages[0]["page"]["followed"]               # once, before s0002 (not for the noise)
    assert np.isclose(np.degrees(np.arctan2(ex.used()[1, 0], ex.used()[0, 0])), 2.0)
    assert np.allclose(ex.page_overhead(), spun)
    assert setup.calls[-1]["op"] == "s0002" and any("the page follows it" in text for _, _, text in node.events)


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


def test_without_hover_stops_most_of_the_overheads_poses_past_reach_set_the_page_up_again(setup, monkeypatch):
    """2026-10-06, the ship on a half-skin: the wrist camera read its print nowhere, hover stops ended, and needle
    drag slid the untaped skin 15 mm and 6 deg over 20 strokes while the overhead watched and the strokes went on.
    The overhead's poses since setup now set the page up again; one stray pose does not."""
    import copy

    ex, io, node = setup.ex, setup.io, setup.node
    ex.tracker = StubTracker([0.0, 0.0])
    ex.tracker.correcting = False                                       # the print was never read
    ex.stack = copy.deepcopy(STACK)
    ex.stack["hardware"], ex.stack["page"]["track"] = "real", {"mode": "correct", "hover_stop": True}
    slid = np.eye(4)
    slid[:3, 3] = [-0.011, 0.011, 0.0]
    seen = iter([[np.eye(4), slid, np.eye(4)], [slid, slid]])            # a stray pose, then the sheet gone
    ex.tracker.on_lift = lambda stub: node.measured.extend((time.monotonic(), p) for p in next(seen, []))
    pages = []
    sent = lambda: sum(row.get("event") == "sent" for row in setup.ledger.rows())   # noqa: E731
    monkeypatch.setattr(ex, "_new_page", lambda program, touch, ledger: pages.append(sent()))
    monkeypatch.setattr(ex_mod.track.Tracker, "start", classmethod(lambda cls, ex, since=0.0, centre=None: None))
    line3 = {**LINE2, "id": "s0002", "points_m": [[0.0, 0.02], [0.02, 0.02]]}
    io.script = [("ok", 10, {}, None)] * 2   # then the lift before the set-up, and s0002
    assert draw_strokes(ex, {"ops": [LINE, LINE2, line3]}, setup.ledger, touch=False) == "complete"
    assert pages == [2]                                                  # before s0002, not s0001
    assert any("the overhead sees the sheet 15.6 mm, 0.0 deg" in text and "3 of its newest 5" in text
               for _, _, text in node.events)


def test_holds_that_read_nothing_move_the_last_placed_page_as_the_overhead_saw_the_sheet_move(setup):
    """2026-10-06: a half-skin slid 4.8 mm and turned 4 deg late in a draw, and its inked print gave the gauge holds
    nothing to read; the overhead's own pose is ~7 mm off the print there, its measure of the slide ~1-2 mm."""
    ex, node = setup.ex, setup.node
    ex.page_record = {"gauge": []}
    placed, then = np.eye(4), np.eye(4)
    placed[:3, 3] = [0.31, 0.05, 0.004]                                   # where the holds put the page
    then[:3, 3] = [0.324, 0.044, 0.008]                                   # the overhead's pose then, 7 mm away
    turn = np.radians(-4.0)
    slide = np.eye(4)
    slide[:2, :2] = [[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]
    slide[:2, 3] = [-0.0044, 0.002]
    ex.placed, ex.overhead = {"page": placed, "overhead": then}, slide @ then
    touched = np.eye(4)
    touched[:3, 3] = [0.30, 0.06, 0.0045]                                 # the new locate's place, the paper's height
    page = ex._hold_page(touched, touched)
    expect = slide @ then @ np.linalg.inv(then) @ placed
    assert np.allclose(page[:2, 3], expect[:2, 3], atol=1e-9) and np.isclose(page[2, 3], 0.0045)
    assert np.isclose(np.arctan2(page[1, 0], page[0, 0]), np.arctan2(expect[1, 0], expect[0, 0]))
    assert "the gauge holds read nothing: the page is the last one they placed" in node.events[-1][2]
    assert np.allclose(ex.placed["page"], page)                           # the next slide moves this one
    ex.placed = None
    assert np.allclose(ex._hold_page(touched, touched), touched)          # nothing placed before: as touched


def test_a_resume_keeps_its_page_only_where_the_overhead_still_sees_the_sheet(setup):
    """Compared with the overhead's own pose at setup, not the page located from it (6.8 mm apart on a half-skin,
    2026-10-06); a sheet moved 20 mm is set up again, tracking or not."""
    import copy

    ex, node = setup.ex, setup.node
    ex.stack = copy.deepcopy(STACK)
    ex.stack["page"].update(source="stencil")
    located = np.eye(4)
    located[0, 3] = 0.0068
    page = {"camera_at_setup": located.tolist(), "correction": np.eye(4).tolist(),
            "locate": {"overhead": np.eye(4).tolist()}}
    node.camera = np.eye(4)
    node.camera[1, 3] = 0.001                                            # the overhead's noise
    assert ex.restore_page(page) and np.allclose(ex.overhead, np.eye(4)) and np.allclose(ex.camera, located)
    node.camera[1, 3] = 0.02                                             # 20 mm: under page.replan_*
    assert not ex.restore_page(page)
    assert "the overhead sees the sheet 20.0 mm, 0.0 deg from where it was set up" in node.events[-1][2]


def test_a_resume_clears_the_view_for_a_sheet_the_arm_hides_and_keeps_the_holds_placed_page(setup, monkeypatch):
    """2026-10-06: the arm stopped over the print, the overhead saw no fresh pose, and the page set up again without
    readable holds landed 7.7 mm from the first half on an unmoved sheet."""
    import copy

    ex, node = setup.ex, setup.node
    ex.stack = copy.deepcopy(STACK)
    ex.stack["page"].update(source="stencil", wait_s=1.0)
    page = {"camera_at_setup": np.eye(4).tolist(), "correction": np.eye(4).tolist()}
    hidden = iter([{"source": "lost", "measured_age_s": 198.0}, {"source": "measured"}])
    node.wait_camera_page = lambda arm, timeout, max_lost_s=None, should_stop=None: (np.eye(4), next(hidden))
    cleared = []
    monkeypatch.setattr(ex, "clear_view", lambda ledger=None: cleared.append(True))
    assert ex.restore_page(page) and cleared == [True]
    workspace = {"tool": "lutin-3rl-bugpin"}
    monkeypatch.setattr(ex, "workspace", lambda: workspace)
    rows = [{"arm": "right", "event": "page", "workspace": workspace, "page": {"id": 1, "holds": {"used": True}}},
            {"arm": "right", "event": "page", "workspace": workspace, "page": {"id": 2, "holds": {"used": True}}},
            {"arm": "right", "event": "page", "workspace": workspace, "page": {"id": 3}}]
    assert ex._retained_page(rows)["id"] == 2
    assert ex._retained_page(rows[:1] + [{**rows[2], "workspace": {"tool": "other"}}]) is None


def test_stopping_the_preplan_waits_for_a_plan_still_running(setup):
    """2026-10-08, a curved page: a numpad dip between strokes waited 1 s for the planner, still lifting the next
    stroke's points along their normals, and ended the run."""
    import time

    setup.ex._planner.submit(lambda: time.sleep(1.5))
    setup.ex.stop_preplan()


def test_clearing_the_view_goes_over_the_page_first_when_the_move_to_rest_would_dip(setup, monkeypatch):
    """2026-10-08: from a hold over a skin pinned on a raised holder the joint move to rest swept the pen back over it
    4 mm up; refused, the arm hid the page from the overhead until the run gave up. Over the page centre at
    READY_HIGH_M first, then to rest; a leg that would still dip is not taken."""
    ex = setup.ex
    ex.camera = object()
    setup.io.q = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])            # a hold, away from rest
    monkeypatch.setattr(ex, "_height", lambda: 0.013)
    monkeypatch.setattr(ex_mod.ready, "solve_ik_seeded", lambda kin, target, seed: seed)
    monkeypatch.setattr(ex_mod.ready, "pen_up_joint_move", lambda *args, **kwargs: object())
    monkeypatch.setattr(ex, "used", lambda: np.eye(4))
    monkeypatch.setattr(ex, "tcp", lambda: np.eye(4))
    moves = []
    monkeypatch.setattr(ex, "move", lambda make, phase, label, ledger=None: moves.append(label))
    for lows, taken in (([0.004, 0.012, 0.030], ["over the page, high, clearing the cameras' view",
                                                  "rest, clearing the cameras' view of the page"]),
                        ([0.004, 0.002], []),
                        ([0.004, 0.012, 0.003], ["over the page, high, clearing the cameras' view"]),
                        ([0.020], ["rest, clearing the cameras' view of the page"])):
        heights, moves[:] = iter(lows), []
        monkeypatch.setattr(ex, "lowest_over_page", lambda traj, page=None, h=heights: next(h))
        ex.clear_view()
        assert moves == taken


def _curved_executor(setup, monkeypatch, check=False):
    """An executor on the real arm with page.surface.mode fitted over a page at the base's origin, and five gauge holds
    over a forearm of 40 mm radius (its crest along page y) at a 20 x 30 mm drawing's corners and centre."""
    import copy

    from tatbot_session import curved

    ex = setup.ex
    ex.run_dir = setup.ledger.path.parent if hasattr(setup.ledger, "path") else None
    ex.stack = copy.deepcopy(STACK)
    ex.stack["hardware"] = "real"
    ex.stack["page"].update(size_m=[0.083, 0.127], surface={"mode": "fitted", "check": check})
    ex.camera, ex.correction = np.eye(4), np.eye(4)
    program = {"ops": [{"op": "stroke", "id": "s0", "points_m": [[-0.01, -0.015], [0.01, 0.015]]}]}
    holds = curved.layout(program, [0.043, 0.088])
    radius = 0.040

    def skin(xy):
        return np.sqrt(radius ** 2 - np.atleast_2d(xy)[:, 0] ** 2) - radius
    rng = np.random.default_rng(3)
    ex.page_record = {"gauge": []}
    for k, xy in enumerate(holds):
        around = xy + rng.uniform(-0.02, 0.02, (1200, 2))
        ex._patches[k] = np.c_[around, skin(around) + rng.normal(0, 0.0002, len(around))]
        ex.page_record["gauge"].append({"touch": k, "xy": xy.tolist(), "point": [*xy, float(skin(xy)[0])]})
    monkeypatch.setattr(ex, "workspace", lambda: {"tool": "x"})
    monkeypatch.setattr(curved, "load_turn", lambda arm: None)          # no calibrated camera turn: fitted per setup
    return ex, program, skin


def test_a_curved_page_fits_the_holds_skin_and_plans_on_it(setup, monkeypatch):
    ex, program, skin = _curved_executor(setup, monkeypatch)
    assert ex._on_surface() == {}                                        # nothing fitted yet: the plane's calls
    ex._fit_surface(program)
    grid = np.stack(np.meshgrid(np.linspace(-0.01, 0.01, 5), np.linspace(-0.015, 0.015, 5)), -1).reshape(-1, 2)
    assert np.max(np.abs(ex.surface_field(grid) - skin(grid))) < 0.0002
    surface = ex._on_surface()["surface"]
    assert not surface.flat and np.allclose(surface.base, ex.used())
    assert ex.page_record["surface"]["report"]["holds"] == 5
    assert ex.run_dir is None or (ex.run_dir / "surface.npz").exists()
    assert any(text.startswith("surface: a degree-4 field") for _, _, text in setup.node.events)
    ex.io.q = np.array([0.0, 0.0, 0.005, 0, 0, 0, 0])                     # FakeKin: the tip at q[:3], over the crest
    assert ex._height() == pytest.approx(0.005, abs=1e-4)
    page = {"camera_at_setup": np.eye(4).tolist(), "correction": np.eye(4).tolist(),
            "surface": ex.page_record["surface"]}
    ex.surface_field = None
    ex.stack["page"]["source"] = "fixed"
    assert ex.restore_page(page) and ex.surface_field is not None        # a resume keeps the shape


def test_a_curved_page_takes_the_wrist_cameras_turn_out_of_its_patches(setup, monkeypatch):
    """A camera mounting turned 3.5 deg turns every hold's patch about its tip (on the forearm, 2026-10-07, the
    patches put the crest 3 mm off): the tilt is fitted with the field and taken out, and the field is the skin
    again. A hold whose frames disagreed at every try gives its skin, never its point (here 3 mm off)."""
    from tatbot_session import curved

    ex, program, skin = _curved_executor(setup, monkeypatch)
    turn = np.array([0.004, -0.061])
    for record in ex.page_record["gauge"]:
        tip = np.r_[record["point"][:2], record["point"][2] + 0.010]
        record["tcp"] = np.block([[np.eye(3), tip[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]).tolist()
        ex._patches[record["touch"]] = curved._turned(ex._patches[record["touch"]], tip, turn)
    ex.page_record["gauge"].append({**ex.page_record["gauge"][0], "touch": 5, "spread_m": 0.0027,
                                    "heights_m": [0.008, 0.009, 0.010, 0.0105, 0.0107],
                                    "point": [*ex.page_record["gauge"][0]["point"][:2], 0.003]})
    ex.page_record["gauge"][1].update(spread_m=0.0009, heights_m=[0.0091, 0.0100, 0.0101, 0.0103, 0.0100])
    ex._patches[5] = ex._patches[0].copy()
    ex._fit_surface(program)
    report = ex.page_record["surface"]["report"]
    assert np.allclose(report["camera_tilt_rad"], -turn, atol=0.003)
    assert report["holds"] == 6 and report["gauged"] == 5                # the disagreeing hold: its skin, no point
    assert np.isnan(report["residual_m"][-1]) and np.isnan(report["loo_m"][-1])
    grid = np.stack(np.meshgrid(np.linspace(-0.01, 0.01, 5), np.linspace(-0.015, 0.015, 5)), -1).reshape(-1, 2)
    assert np.max(np.abs(ex.surface_field(grid) - skin(grid))) < 0.0002


def test_a_calibrated_camera_turn_is_not_fitted_again(setup, monkeypatch):
    """With the camera's turn calibrated (gauge turn) the patches come turned from hold_patch: the field is fitted on
    them as they are, and the turn left in them is reported, never applied."""
    from tatbot_session import curved

    ex, program, skin = _curved_executor(setup, monkeypatch)
    monkeypatch.setattr(curved, "load_turn", lambda arm: np.zeros(3))
    for record in ex.page_record["gauge"]:
        tip = np.r_[record["point"][:2], record["point"][2] + 0.010]
        record["tcp"] = np.block([[np.eye(3), tip[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]).tolist()
    ex._fit_surface(program)
    report = ex.page_record["surface"]["report"]
    assert report["turn"] == "calibrated" and np.allclose(report["camera_tilt_rad"], 0.0, atol=0.002)
    grid = np.stack(np.meshgrid(np.linspace(-0.01, 0.01, 5), np.linspace(-0.015, 0.015, 5)), -1).reshape(-1, 2)
    assert np.max(np.abs(ex.surface_field(grid) - skin(grid))) < 0.0002


def test_the_holds_after_the_first_stand_along_its_skins_normals(setup, monkeypatch):
    """The first hold's skin (here a forearm of 40 mm radius, its crest along page y) sets the normals the others
    are held along: 12 mm round the arm the tool leans asin(12/40) = 17.5 deg toward that side, its tip 10 mm out
    along the normal from the skin."""
    from tatbot_session import config, gauge

    ex, program, skin = _curved_executor(setup, monkeypatch)
    ex.stack["page"]["height"] = {"gauge": "use"}
    ex.stack["page"]["surface"]["along_skin"] = True
    ex.prior_correction, ex.page_record["touches"], ex.page_record["gauge"] = np.eye(4), [], []
    cal = Path(setup.ex.repo) / "cal.json"
    cal.write_text(json.dumps({"measured_with": "x"}))
    monkeypatch.setattr(gauge, "calibration_path", lambda arm: cal)
    monkeypatch.setattr(config, "fitted_tool", lambda repo, arm: "x")
    patch = ex._patches[0]                                              # the skin about the centre hold
    frames = []

    def hold(i, xy, hover, prior, heading, cal, ledger=None, frame=None):
        frames.append((frame, hover))
        ex._patches[i] = patch
        point = [*xy, float(skin(xy)[0])]
        return {"touch": i, "xy": list(xy), "point": point, "height_m": 0.010, "spread_m": 0.0002,
                "tcp": np.eye(4).tolist(), "q": [0.0] * 7}
    monkeypatch.setattr(ex, "_gauge_hold", hold)
    for i, xy in enumerate(([0.0, 0.0], [0.012, 0.0])):
        assert ex._gauge_point(i, xy, {"known": False, "level": 0.0}, 0.010, np.eye(4))
    assert frames[0][0] is None and ex._provisional is not None
    frame, hover = frames[1]
    assert hover == pytest.approx(0.010)
    tilt = np.degrees(np.arccos(frame[:3, 2] @ [0.0, 0.0, 1.0]))
    assert tilt == pytest.approx(17.5, abs=1.0) and frame[0, 2] > 0     # leaning out to +x
    assert frame[2, 3] == pytest.approx(skin([[0.012, 0.0]])[0], abs=0.0003)
    ex.stack["page"]["surface"]["along_skin"] = False                   # the default: square to the page
    frames.clear()
    assert ex._gauge_point(2, [0.012, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4))
    assert frames[0][0] is None


def test_a_curved_pages_holds_stand_at_the_gauges_own_height(setup, monkeypatch):
    """The overhead's page sat 4-6 mm over the forearm's skin: a hold whose tip stands more than 2 mm off the full
    height over the skin is held again there, and the holds after it start 5 mm over there and come down a step."""
    from tatbot_session import config, gauge

    ex, program, skin = _curved_executor(setup, monkeypatch)
    ex.stack["page"]["height"] = {"gauge": "use"}
    ex.prior_correction, ex.page_record["touches"], ex._patches = np.eye(4), [], {}   # no skin: square to the page
    cal = Path(setup.ex.repo) / "cal.json"
    cal.write_text(json.dumps({"measured_with": "x"}))
    monkeypatch.setattr(gauge, "calibration_path", lambda arm: cal)
    monkeypatch.setattr(config, "fitted_tool", lambda repo, arm: "x")
    hovers = []

    def hold(i, xy, hover, prior, heading, cal, ledger=None, frame=None):
        hovers.append(round(hover, 6))
        ex.page_record["gauge"].append({"touch": i, "xy": list(xy), "point": [*xy, 0.0], "height_m": hover + 0.005,
                                        "spread_m": 0.0002, "heights_m": [hover + 0.005] * 5,
                                        "tcp": np.eye(4).tolist(), "q": [0.0] * 7})
        return ex.page_record["gauge"][-1]
    monkeypatch.setattr(ex, "_gauge_hold", hold)
    ex.page_record["gauge"] = []
    for i in range(2):
        assert ex._gauge_point(i, [0.0, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4)) is not None
    assert hovers == [0.010, 0.005, 0.010, 0.005] and ex._hold_low == pytest.approx(0.005)


def test_a_curved_pages_watch_measures_from_the_overheads_median_and_its_own_scatter(setup, monkeypatch):
    """On the forearm single overhead poses scattered +-5 deg: the watch's reference is the median of the poses during
    the setup, and a still forearm's scatter neither sets the page up again nor moves it."""
    ex, program, skin = _curved_executor(setup, monkeypatch)

    def turned(deg, dx=0.0):
        pose = geometry.translation([0.3 + dx, 0.0, 0.0])
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        pose[:2, :2] = [[c, -s], [s, c]]
        return pose
    ex.overhead, ex.overhead_at, ex.slid = turned(-5.0), 0.0, np.eye(4)
    ex.camera = ex.overhead
    poses = [turned(d, dx) for d, dx in ((1.0, 0.001), (-2.0, -0.002), (0.5, 0.0), (3.0, 0.002), (-1.0, -0.001))]
    monkeypatch.setattr(ex.node, "measured_pages", lambda arm, since: poses, raising=False)
    ex._steady_overhead()
    assert np.allclose(ex.overhead[:3, 3], [0.3, 0.0, 0.0], atol=1e-6)
    assert abs(np.degrees(np.arctan2(ex.overhead[1, 0], ex.overhead[0, 0])) - 0.5) < 1e-6
    ex.stack["page"]["track"] = {}
    poses[:] = [turned(d, dx) for d, dx in ((3.5, 0.003), (-4.0, -0.003), (3.0, 0.002), (-3.0, 0.0), (4.0, 0.003))]
    calls = []
    monkeypatch.setattr(ex, "_reacquire", lambda *a, **k: calls.append("reacquire"))
    monkeypatch.setattr(ex, "_follow_sheet", lambda *a, **k: calls.append("follow"))
    monkeypatch.setattr(ex, "_hover_on", lambda: False)
    monkeypatch.setattr(ex, "_corrects", lambda: True)
    ex._watch_sheet(program, {"id": "s1"}, 0.0, None)
    assert calls == []
    # one stray pose 12 deg off, the arm over the print: the median of the newest stands, the run goes on
    ex.stack["page"].update(source="stencil")
    monkeypatch.setattr(ex.node, "camera_page", lambda arm: (turned(12.5), {}), raising=False)
    poses.append(turned(12.5))
    ex.refresh_page(program, None)
    assert calls == []


def test_a_check_hold_reads_the_field_by_the_skin_about_its_tip(setup, monkeypatch):
    """Seen at a grazing view a check's point under the pen read the forearm's skin 1.5 mm deep while the skin about
    it sat on the field (2026-10-07): the check reads the field by that skin (curved.ring_miss), its point kept as
    point_miss_m; with too little skin about the tip, by the point."""
    from tatbot_session import gauge

    ex, program, skin = _curved_executor(setup, monkeypatch, check=True)
    ex.stack["page"]["surface"]["check"] = False
    ex._fit_surface(program)
    cal = Path(setup.ex.repo) / "cal.json"
    cal.write_text(json.dumps({"measured_with": "x"}))
    monkeypatch.setattr(gauge, "calibration_path", lambda arm: cal)
    monkeypatch.setattr(ex, "lift", lambda what, ledger=None: None)
    rng = np.random.default_rng(5)

    def hold(i, xy, hover, prior, heading, cal, ledger=None, frame=None):
        around = xy + rng.uniform(-0.02, 0.02, (1500, 2))
        ex._patches[i] = np.c_[around, skin(around)]                     # the skin as it is, its point 1.5 mm deep
        read = {"touch": i, "xy": list(xy), "point": [*xy, float(skin(xy)[0]) - 0.0015], "height_m": 0.010}
        ex.page_record["gauge"].append(read)
        return read
    monkeypatch.setattr(ex, "_gauge_hold", hold)
    ex._check_holds([[0.0, -0.012], [0.01, 0.0]], 100, "the skin between the holds")
    checks = [r for r in ex.page_record["gauge"] if r.get("check")]
    assert all(abs(r["field_miss_m"]) < 0.0002 and r["point_miss_m"] < -0.0013 for r in checks)


def test_a_curved_page_refuses_a_drawing_past_its_holds_and_measures_again_past_a_check_off_the_field(setup, monkeypatch):
    ex, program, skin = _curved_executor(setup, monkeypatch, check=True)
    wide = {"ops": program["ops"] + [{"op": "stroke", "id": "s1", "points_m": [[0.0, 0.06]]}]}
    with pytest.raises(RuntimeError, match="reaches .* from every gauge hold"):
        ex._fit_surface(wide)
    ex.correction = geometry.translation([0.0, 0.025, 0.0])   # the holds' shift moved the page: the drawing is
    with pytest.raises(RuntimeError, match="reaches .* from every gauge hold"):   # judged from where it puts their feet
        ex._fit_surface(program)
    ex.correction = np.eye(4)
    calls = []

    def hold(i, xy, hover, prior, heading, cal, ledger=None, frame=None, off=0.0):
        calls.append(xy)
        frames.append(prior)
        read = {"touch": i, "xy": list(xy), "point": [*xy, float(skin(xy)[0]) + off], "height_m": 0.010}
        ex.page_record["gauge"].append(read)
        return read
    frames = []
    ex._hold_frame = np.eye(4) @ geometry.translation([0.0, 0.0, 0.001])   # the holds' page: the checks' too
    from tatbot_session import gauge

    cal = Path(setup.ex.repo) / "cal.json"
    cal.write_text(json.dumps({"measured_with": "x"}))
    monkeypatch.setattr(gauge, "calibration_path", lambda arm: cal)
    monkeypatch.setattr(ex, "_gauge_hold", hold)
    monkeypatch.setattr(ex, "lift", lambda what, ledger=None: None)
    ex._fit_surface(program)
    assert len(calls) == 1 and max(abs(m) for m in ex.page_record["surface"]["check_m"]) < 0.0003   # holds that agree
    assert all(frame is ex._hold_frame for frame in frames)
    monkeypatch.setattr(ex, "_gauge_hold", lambda *a, **k: hold(*a, **k, off=0.001))
    ex.page_record["gauge"] = [r for r in ex.page_record["gauge"] if r["touch"] < 100]
    ex._fit_surface(program)                    # 1 mm off: all four, two more holds, the pen 1 mm higher, never refused
    assert len(calls) == 1 + 4 + 2 and ex.page_record["surface"]["rise_m"] >= 0.00095


def test_holds_that_predict_one_another_are_checked_at_one_midpoint_and_at_four_when_it_reads_far(setup, monkeypatch):
    """Four check holds took 67-71 s of a set-up (2026-10-20). Holds that each gave a point and predict one another
    within CHECK_ONE_M (their leave-one-out) are checked at the midpoint by the hold the others predict worst, and at
    the other three only when it reads past half page.surface.max_loo_m or is not counted. A hold with no point, or a
    page the checks must confirm (_confirm_holds), gets all four. Over 2026-10-13..20's 98 curved set-ups the rule
    found a check past the limit where the four did; the one among those it checks once read 0.74-0.96 mm at all four."""
    from tatbot_session import curved, gauge

    ex, program, skin = _curved_executor(setup, monkeypatch, check=True)
    ex.stack["page"]["surface"]["max_loo_m"] = 0.0009
    cal = Path(setup.ex.repo) / "cal.json"
    cal.write_text(json.dumps({"measured_with": "x"}))
    monkeypatch.setattr(gauge, "calibration_path", lambda arm: cal)
    monkeypatch.setattr(ex, "lift", lambda what, ledger=None: None)
    calls, off, held = [], {}, list(ex.page_record["gauge"])

    def hold(i, xy, hover, prior, heading, cal, ledger=None, frame=None):
        calls.append((i, np.round(np.asarray(xy, float), 6).tolist()))
        read = {"touch": i, "xy": list(xy), "point": [*xy, float(skin(xy)[0]) + off.get(i, 0.0)], "height_m": 0.010}
        ex.page_record["gauge"].append(read)
        return read
    monkeypatch.setattr(ex, "_gauge_hold", hold)

    def checked(**kw):
        calls.clear()
        off.clear()
        off.update(kw.get("off", {}))
        ex.page_record = {"gauge": [dict(r) for r in held], **kw.get("record", {})}
        for k in kw.get("apart", []):
            ex.page_record["gauge"][k]["spread_m"] = 0.002
        ex._fit_surface(program)
        return [i for i, _ in calls]
    assert checked() == [100]
    loo, holds = np.abs(ex.page_record["surface"]["report"]["loo_m"]), curved.layout(program, [0.043, 0.088])
    points = curved.check_points(holds)
    nearest = points[np.argmin(np.linalg.norm(points - holds[np.argmax(loo)], axis=1))]
    assert np.allclose(calls[0][1], nearest, atol=1e-6) and len(ex.page_record["surface"]["check_m"]) == 1
    assert checked(off={100: 0.0006}) == [100, 101, 102, 103]               # past half the limit: the other three
    assert len(ex.page_record["surface"]["check_m"]) == 4
    assert checked(off={100: 0.0006, 102: 0.0012}) == [100, 101, 102, 103, 104, 105]   # one past the limit: the
    assert ex.page_record["surface"]["rise_m"] >= 0.0011                    # measured again, the pen higher
    assert checked(apart=[2]) == [100, 101, 102, 103]                       # a hold gave no point
    assert checked(record={"holds": {"confirm": True}}) == [100, 101, 102, 103]


def test_turned_checks_hold_the_pen_as_the_strokes_will_be_and_the_strokes_ride_what_they_read(setup, monkeypatch):
    """The holds stand square to the page, the strokes turned to the skin (page.surface.turn_pen), and the arm's error
    turns with its pose: on a skin round a bottle the hovers at one end of the drawing read it 1.4-1.8 mm over the
    holds' field, the other end's 0.0 (2026-10-24). After the square check, page.surface.turned_checks holds at the
    check point where the field turns most and the one across from it, the pen turned to the field's normal; what each
    reads over the field (less the square check's there) is a lift the strokes ride along the drawing, and the skin
    watch and the ride take out (_rise_at). None with the pen square, holds along the skin, or turned_checks 0."""
    from tatbot_session import curved, gauge

    ex, program, skin = _curved_executor(setup, monkeypatch, check=True)
    cal = Path(setup.ex.repo) / "cal.json"
    cal.write_text(json.dumps({"measured_with": "x"}))
    monkeypatch.setattr(gauge, "calibration_path", lambda arm: cal)
    monkeypatch.setattr(ex, "lift", lambda what, ledger=None: None)
    calls, held, rng = [], list(ex.page_record["gauge"]), np.random.default_rng(7)

    def hold(i, xy, hover, prior, heading, ledger=None, frame=None):
        calls.append((i, np.round(np.asarray(xy, float), 6).tolist(), frame is not None))
        up = 0.0015 if i >= ex.TURNED_FIRST and xy[0] > 0 else 0.0   # turned, at the +x end: the arm's error
        around = np.asarray(xy, float) + rng.uniform(-0.02, 0.02, (1500, 2))
        ex._patches[i] = np.c_[around, skin(around) + up]
        read = {"touch": i, "xy": list(xy), "point": [*xy, float(skin(xy)[0]) + up], "height_m": 0.010}
        ex.page_record["gauge"].append(read)
        return read
    monkeypatch.setattr(ex, "_gauge_hold", hold)

    def fitted(**surface):
        calls.clear()
        ex.stack["page"]["surface"].update(surface)
        ex.page_record = {"gauge": [dict(r) for r in held]}
        ex._fit_surface(program)
        return [(i, xy) for i, xy, turned in calls if turned], ex.page_record["surface"].get("turned")
    turned_calls, turned = fitted(turn_pen=True)
    assert sorted(i for i, _ in turned_calls) == [110, 111]
    assert sorted(xy[0] for _, xy in turned_calls) == pytest.approx([-0.01, 0.01], abs=1e-6)   # most turned, across
    by_end = {np.sign(r["xy"][0]): r for r in turned}
    assert by_end[1.0]["lift_m"] == pytest.approx(0.0015, abs=2e-4) and abs(by_end[-1.0]["lift_m"]) < 2e-4
    assert by_end[1.0]["turn_deg"] == pytest.approx(np.degrees(np.arcsin(0.01 / 0.040)), abs=1.5)
    assert ex._rise_at([[0.01, 0.0], [0.0, 0.0], [-0.01, 0.0]]) == pytest.approx([0.0015, 0.00075, 0.0], abs=2e-4)
    assert isinstance(ex._on_surface()["surface"], curved.Turned)
    said = [text for _, _, text in setup.node.events if text.startswith("surface check, the pen turned")]
    assert said and "the arm's own error at their pose, which they ride" in said[-1]
    for off in ({"turn_pen": False}, {"turn_pen": True, "turned_checks": 0}, {"turned_checks": 2, "along_skin": True}):
        if off.get("along_skin"):   # holds along a first skin's normals: the checks are turned as they are
            monkeypatch.setattr(ex, "_holds_along_skin", lambda: True)
        turned_calls, turned = fitted(**off)
        assert [i for i, *_ in calls if i >= 110] == [] and not turned and not isinstance(
            ex._on_surface().get("surface"), curved.Turned)


def test_a_check_off_the_field_measures_again_and_raises_the_pen_inking_or_rehearsing_never_deeper(setup, monkeypatch):
    """Live forearms' checks read 1.01 and 1.36 mm (2026-10-18/20), and a refusal would end a tattoo at a mid-draw
    set-up. A check past page.surface.max_loo_m never refuses: inking or rehearsing alike, two more holds read the skin
    where the worst check did and halfway from there to the holds' centre (in no fit), and the pen's path rises by the
    most any reading put the skin over the field, the holds' own, kept: a check read under it never pulls the path down
    (a refitted field went 0.83 mm under it beneath a forearm's strokes, 2026-10-20's review). A skin that rose after its
    holds lifts it by the rise, one that sank (the needle shallower already) by no more than the checks read over; the
    rise outlives a resource change and a resume."""
    from tatbot_motion.plan import PenDown
    from tatbot_session import curved, gauge

    ex, program, skin = _curved_executor(setup, monkeypatch, check=True)
    ex.stack["page"]["surface"]["max_loo_m"] = 0.0009
    monkeypatch.setattr(ex, "CHECK_ONE_M", 0.0)   # holds that predict one another no better than that: four checks
    cal = Path(setup.ex.repo) / "cal.json"
    cal.write_text(json.dumps({"measured_with": "x"}))
    monkeypatch.setattr(gauge, "calibration_path", lambda arm: cal)
    monkeypatch.setattr(ex, "lift", lambda what, ledger=None: None)
    monkeypatch.setattr(ex_mod.config, "fitted_tool", lambda repo, arm: "x")
    monkeypatch.setattr(ex_mod.config, "tool_datasheet", lambda repo, tool_id: SimpleNamespace(
        tool_id="x", stroke_m=0.004, tip_out_at_top_m=0.0, raw={"calibration": {"contact_reference": "tube_end"},
                                                                 "needle_reach_mm": 4.0}))
    monkeypatch.setattr(ex, "machine", SimpleNamespace(wait=lambda on, s: True, why_off=lambda: ""), raising=False)
    calls, up, fields, fit = [], {}, [], ex._surface_fit
    monkeypatch.setattr(ex, "_surface_fit", lambda *a, **k: (fit(*a, **k), fields.append(ex.surface_field)))

    def hold(i, xy, hover, prior, heading, cal, ledger=None, frame=None):
        calls.append((i, np.asarray(xy, float)))
        read = {"touch": i, "xy": list(xy), "point": [*xy, float(skin(xy)[0]) + up.get(i, up.get("all", 0.0))],
                "height_m": 0.010}
        ex.page_record["gauge"].append(read)
        return read
    monkeypatch.setattr(ex, "_gauge_hold", hold)
    gauged, holds = list(ex.page_record["gauge"]), curved.layout(program, [0.043, 0.088])
    pens = {"ride": PenDown("ride", 0.006, 0.0, 0.004, 0.004), "hover": PenDown("hover", 0.006, 0.0)}

    def run(mode, lifted):
        ex.page_record = {"gauge": list(gauged)}
        ex.motion = {**MOTION, "pen": {**MOTION["pen"], "mode": mode, "hover": {"height_m": 0.006, "settle_s": 0.0},
                                       "ride": {"fraction": 1.5, "settle_s": 0.0}}}
        ex.pen, calls[:], fields[:] = pens[mode], [], []
        up.clear()
        up.update(lifted)
        ex._fit_surface(program)
        assert fields == [ex.surface_field]                              # the holds' field, the path only rises off it
        return ex.page_record["surface"]
    rises = {}
    for mode in ("ride", "hover"):                                      # one check 1.2 mm over: held twice more
        surface = run(mode, {102: 0.0012})
        assert [i for i, _ in calls] == [100, 101, 102, 103, 104, 105] and surface["report"]["holds"] == 5
        worst = curved.check_points(holds)[2]
        assert np.allclose(calls[-2][1], worst) and np.allclose(calls[-1][1], (np.mean(holds, axis=0) + worst) / 2)
        assert len(surface["fresh_m"]) == 2
        assert surface["rise_m"] >= max(surface["check_m"]) == pytest.approx(0.0012, abs=1e-4)
        assert ex.pen.height_m == pytest.approx(0.006 + surface["rise_m"])
        rises[mode] = surface["rise_m"]
    assert rises["ride"] == pytest.approx(rises["hover"])                # a rehearsal rises as the ink does
    said = [text for _, _, text in setup.node.events if text.startswith("surface check: the skin between the holds read")]
    assert "the pen goes on 1.2" in said[-1]
    surface = run("ride", dict.fromkeys((102, 103, 104, 105), 0.0025))   # the skin rose 2.5 mm after two checks
    assert surface["rise_m"] >= 0.0024 and max(surface["fresh_m"]) > 0.0009 and "still past" in setup.node.events[-1][2]
    assert ex.pen.height_m == pytest.approx(0.006 + surface["rise_m"])
    surface = run("ride", {103: -0.0014})                                # under the field (2026-10-18's): shallower
    assert min(surface["check_m"]) == pytest.approx(-0.0014, abs=1e-4) and surface["rise_m"] < 0.0003
    assert ex.pen.height_m >= 0.006                                      # never lower than the resource's ride
    surface = run("hover", {"all": 0.0025})                              # the skin rose 2.5 mm after the holds
    assert surface["rise_m"] >= 0.0024 and ex.pen.height_m == pytest.approx(0.006 + surface["rise_m"])
    ex.prepare_pen({})                                                   # a resource change rebuilds the pen
    assert ex.pen.height_m == pytest.approx(0.006 + surface["rise_m"])
    ex.stack["page"]["source"] = "fixed"
    page = {"camera_at_setup": np.eye(4).tolist(), "correction": np.eye(4).tolist(), "surface": surface}
    ex.page_record = {}
    ex.prepare_pen({})                                                   # a resume: the pen first, then the page
    assert ex.pen.height_m == pytest.approx(0.006)
    assert ex.restore_page(page) and ex.pen.height_m == pytest.approx(0.006 + surface["rise_m"])
    ex.motion = {**ex.motion, "pen": {**ex.motion["pen"], "mode": "ride"}}
    ex.page_record = {}
    ex.prepare_pen({})                                                   # a riding pen's ride, then risen alike
    assert ex.pen.machine and ex.pen.height_m == pytest.approx(0.006)
    assert ex.restore_page(page) and ex.pen.height_m == pytest.approx(0.006 + surface["rise_m"])


def test_a_first_curved_hold_stands_over_the_skin_the_overheads_markers_put_under_it(setup, monkeypatch):
    """A live forearm's first hold stood 7.6 mm over its skin on the silicone's page still in memory (2026-10-20). A
    first setup's first hold stands touch.start_above_m over the higher of its prior and the skin the overhead's
    marker corners put under it, less the offset the last setup's first hold read between them and the skin: never
    lower, and as before until a setup has read that offset."""
    from tatbot_session import config, gauge

    ex, program, skin = _curved_executor(setup, monkeypatch)
    ex.stack["page"]["height"] = {"gauge": "use"}
    cal = Path(setup.ex.repo) / "cal.json"
    cal.write_text(json.dumps({"measured_with": "x"}))
    monkeypatch.setattr(gauge, "calibration_path", lambda arm: cal)
    monkeypatch.setattr(config, "fitted_tool", lambda repo, arm: "x")
    monkeypatch.setattr(ex.node, "base_from_world", lambda arm: np.eye(4))
    hovers = []

    def hold(i, xy, hover, prior, heading, cal, ledger=None, frame=None):
        hovers.append(round(hover, 6))
        return {"touch": i, "xy": list(xy), "point": [*xy, 0.0], "height_m": 0.010, "spread_m": 0.0002,
                "tcp": np.eye(4).tolist(), "q": [0.0] * 7}
    monkeypatch.setattr(ex, "_gauge_hold", hold)

    def first_hold(marker_z):   # one marker 8 mm square over the page's centre, as the overhead read it
        pts = np.array([[-0.004, -0.004, marker_z], [0.004, -0.004, marker_z], [0.004, 0.004, marker_z],
                        [-0.004, 0.004, marker_z]])
        ex._marks_seen = [{7: np.c_[pts[:, :2] / pts[:, 2:], np.linalg.norm(pts, axis=1)]}] * 3
        ex.prior_correction, ex.page_record, ex._hold_low, ex._provisional = np.eye(4), {"touches": []}, 0.0, None
        hovers.clear()
        ex._gauge_point(0, [0.0, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4))
        return hovers[0]
    assert first_hold(0.012) == pytest.approx(0.010)                        # no offset read yet: as before
    assert ex.anchor_offset == pytest.approx(-0.012) and ex._marks_seen == []
    assert first_hold(0.015) == pytest.approx(0.013)                        # the markers put the skin 3 mm up
    assert first_hold(0.009) == pytest.approx(0.010)                        # ... and 3 mm down: never lower
    assert any("overhead's markers put the skin +3.0 mm" in text for _, _, text in setup.node.events)


def test_a_first_curved_hold_stands_over_the_skin_the_over_centre_views_depth_puts_under_it(setup, monkeypatch):
    """Round a bottle (2026-10-24) the first hold stood 4.2 mm over the skin and was held again 6.0 mm higher. The
    locate's over-centre views' depth (_skin_seen: the gauge's own camera, in the hold's arm configuration) puts the
    skin under the first hold before the overhead's markers do: its points within 5 mm of the hold's foot and 10 mm or
    more under the tip, raised by the tcp's height over the gauge's tip; never lower than over the prior, and with too
    few points the markers' as before."""
    from tatbot_session import config, gauge

    ex, program, skin = _curved_executor(setup, monkeypatch)
    ex.stack["page"]["height"] = {"gauge": "use"}
    cal = Path(setup.ex.repo) / "cal.json"
    cal.write_text(json.dumps({"measured_with": "x"}))
    monkeypatch.setattr(gauge, "calibration_path", lambda arm: cal)
    monkeypatch.setattr(config, "fitted_tool", lambda repo, arm: "x")
    monkeypatch.setattr(ex.node, "base_from_world", lambda arm: np.eye(4))
    hovers = []

    def hold(i, xy, hover, prior, heading, cal, ledger=None, frame=None):
        hovers.append(round(hover, 6))
        return {"touch": i, "xy": list(xy), "point": [*xy, 0.0], "height_m": 0.010, "spread_m": 0.0002,
                "tcp": np.eye(4).tolist(), "q": [0.0] * 7}
    monkeypatch.setattr(ex, "_gauge_hold", hold)
    grid = np.stack(np.meshgrid(np.linspace(-0.003, 0.003, 9), np.linspace(-0.003, 0.003, 9)), -1).reshape(-1, 2)

    def first_hold(skin_z, count=81):   # the skin under the page centre, the pen's own points over it
        pts = np.r_[np.c_[grid[:count], np.full(count, skin_z)], np.c_[grid, np.full(len(grid), 0.045)]]
        ex._over_skin = [{"points": pts.astype(np.float32), "tcp": np.array([0.0, 0.0, 0.050]),
                          "tip": np.array([0.0, 0.0, 0.0485])}]
        ex.prior_correction, ex.page_record, ex._hold_low, ex._provisional = np.eye(4), {"touches": []}, 0.0, None
        ex._marks_seen, ex.anchor_offset = [], None
        hovers.clear()
        ex._gauge_point(0, [0.0, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4))
        return hovers[0]
    assert first_hold(0.004) == pytest.approx(0.0155)          # 4 mm up, the tip 1.5 mm under the tcp
    assert any("depth from over the page centre puts the skin +5.5 mm" in text for _, _, text in setup.node.events)
    assert first_hold(-0.006) == pytest.approx(0.010)           # under the prior: never lower
    assert first_hold(0.004, count=40) == pytest.approx(0.010)  # too few points: the markers' (none here)


def _gauged(setup, monkeypatch, heights, check=False, fresh=False, patches=True):
    """A curved executor (_curved_executor) whose wrist gauge reads `heights(i, xy, hover, frame)`: a hold's pooled
    group heights (m), or None for nothing read; every hold it makes is kept as (i, xy, hover, frame) in the list it
    returns, and every lift as ("lift", what). `fresh`: no hold read yet, the next is the setup's first. `patches`
    False: the holds see no skin about them, so no first hold's skin places the holds after it (_hold_start)."""
    from tatbot_session import config, gauge

    ex, program, skin = _curved_executor(setup, monkeypatch, check=check)
    ex.stack["page"]["height"] = {"gauge": "use"}
    if not patches:
        ex._patches.clear()
    ex.prior_correction, ex.page_record["touches"], ex._hold_low = np.eye(4), [], 0.0
    if fresh:
        ex.page_record["gauge"], ex._provisional, ex._start = [], None, None
    cal = Path(setup.ex.repo) / "cal.json"
    cal.write_text(json.dumps({"measured_with": "x"}))
    monkeypatch.setattr(gauge, "calibration_path", lambda arm: cal)
    monkeypatch.setattr(config, "fitted_tool", lambda repo, arm: "x")
    made = []

    def hold(i, xy, hover, prior, heading, cal, ledger=None, frame=None):
        made.append((i, [round(float(v), 6) for v in xy], hover, frame))
        g = heights(i, xy, hover, frame)
        if g is None:
            return None
        read = {"touch": i, "xy": list(xy), "point": [*xy, float(skin(np.asarray(xy))[0])], "tcp": np.eye(4).tolist(),
                "q": [0.0] * 7, "height_m": float(np.median(g)), "heights_m": list(g), "spread_m": float(np.ptp(g))}
        ex.page_record["gauge"].append(read)
        return read
    monkeypatch.setattr(ex, "_gauge_hold", hold)
    monkeypatch.setattr(ex, "lift", lambda what, ledger=None: made.append(("lift", what)))
    return ex, program, made


def _still_skin(first_mm, at_stand=0.010):
    """A skin that stays put: each reading is `first_mm` (five group heights) shifted by how far the hold stands off
    `at_stand`, as the tip's own move shifts what the gauge reads."""
    return lambda i, xy, hover, frame: (np.asarray(first_mm) * 1e-3 + hover - at_stand).tolist()


# 2026-10-20's live forearm, its fifth hold on a steep flank: each try's readings as five pooled group heights (mm),
# with the stand over that hold's first (mm) each was read at. The deployed rule moved the tip 11.7 mm up, then 5.1
# and 8.7 mm down by the medians of frames 2.8 and 5.9 mm apart (limit 0.5); the field put it 4.8-7.1 mm over the skin.
FLANK = {-0.0135: [(0.0, [-2.37, -4.46, -1.73, 5.93, 2.30]), (11.73, [8.10, 11.22, 9.84, 17.21, 13.18])],
         -0.0095: [(11.73, [17.55, 15.06, 14.76, 13.71, 17.67]), (6.67, [14.14, 13.31, 13.62, 14.13, 13.64])],
         -0.0175: [(6.67, [7.91, 19.68, 19.60, 18.71, 13.74]), (-2.04, [13.14, 2.06, 10.67, 8.64, 6.63])]}


def test_a_curved_hold_moves_toward_the_skin_only_on_frames_that_agree(setup, monkeypatch):
    """The forearm's flank replayed (each recorded reading shifted by how far the hold stands from where it was read):
    the tip only rises. Its first reading's nearest frame put it 4.5 mm into the skin, so it goes 14.5 mm up at once,
    and the hold's next tries start there; no reading whose frames disagree moves it down, the one spread 11.8 mm
    included; the hold gives no point, and the holds after it start where they did. (A hold after the first starts
    HOLD_STEP_M over the depth before it: that depth here puts it where the first reading was made.)"""
    seen = {}

    def heights(i, xy, hover, frame):
        recorded = FLANK[round(xy[0], 4)]
        n = seen[xy[0]] = seen.get(xy[0], -1) + 1
        stood, mm = recorded[min(n, len(recorded) - 1)]
        return (np.asarray(mm) * 1e-3 + hover - (0.010 + stood * 1e-3)).tolist()
    ex, _, made = _gauged(setup, monkeypatch, heights)
    ex._hold_low = ex.HOLD_STEP_M
    assert ex._gauge_point(4, [-0.0135, 0.0114], {"known": False, "level": 0.0}, 0.010, np.eye(4)) == []
    stands = [m[2] for m in made if m[0] == 4]
    assert stands[0] == pytest.approx(0.010) and stands[1:] == pytest.approx([0.02446] * 3)
    assert all(b >= a - 1e-12 for a, b in zip(stands, stands[1:], strict=False)) and ex._hold_low == ex.HOLD_STEP_M
    texts = [text for _, _, text in setup.node.events]
    assert any("stood -1.7 mm over the skin (its frames 4.67 mm apart, the nearest -4.5); held again 14.5 mm higher"
               in t for t in texts) and not any("lower" in t for t in texts)


def test_a_curved_holds_step_is_its_nearest_frame_toward_the_skin_capped_and_away_at_once(setup, monkeypatch):
    """_step on the forearm's and the silicone's readings: away from the skin by the nearest frame on any reading
    (the 11.8 mm spread one: 2.1 mm up, where its median took the tip 8.7 mm down); toward it only on frames that agree,
    by the nearest frame and 5 mm at most, never for a check, and after a step only on a reading near its aim. One or
    two pooled groups agree with themselves, so they never take it nearer (two read silicone 20.4 mm under, 2026-10-18)."""
    ex, _, _ = _gauged(setup, monkeypatch, lambda *a: None)

    def step(mm, aim=None, up_only=False):
        g = np.asarray(mm) * 1e-3
        return ex._step({"touch": 0, "height_m": float(np.median(g)), "heights_m": g.tolist(), "spread_m": float(
            np.ptp(g))}, 0.010, None if aim is None else aim * 1e-3, up_only) * 1e3
    assert step([7.91, 19.68, 19.60, 18.71, 13.74]) == pytest.approx(-2.09)       # spread 11.8: up, never down
    assert step([17.55, 15.06, 14.76, 13.71, 17.67]) == 0.0                       # far, frames 2.8 mm apart: stays
    assert step([-2.37, -4.46, -1.73, 5.93, 2.30]) == pytest.approx(-14.46)       # near by any frame: up at once
    assert step([18.03, 17.98, 18.0, 17.99, 18.0]) == pytest.approx(5.0)          # silicone, 8 mm far: 5 mm a step
    assert step([13.01, 13.1, 12.9, 12.99, 13.09]) == pytest.approx(2.9)          # by its nearest frame
    assert step([7.18, 7.18, 7.2, 7.18, 7.19]) == pytest.approx(-2.82)            # near, agreeing: up
    assert step([9.07, 9.05, 8.66, 8.62, 9.26]) == 0.0                            # within 2 mm: held there
    assert step([9.5, 12.4, 12.5, 12.6, 13.0]) == 0.0                             # a frame under 10 mm: no nearer
    assert step([18.03, 17.98, 18.0, 17.99, 18.0], up_only=True) == 0.0           # a check never goes down
    assert step([0.95, 0.28, 1.2, 0.99, 2.36], up_only=True) == pytest.approx(-9.72)   # ... but up at once
    assert step([16.0] * 5, aim=13.0) == 0.0                                      # 3 mm off the step's aim: stays
    assert step([13.5] * 5, aim=13.0) == pytest.approx(3.5)                       # within 1.5 mm: on
    assert step([18.96]) == 0.0 and step([20.35, 20.53]) == 0.0                   # one or two groups: no nearer
    assert step([18.0, 18.1, 18.2]) == pytest.approx(5.0)                         # three that agree: on
    assert step([4.0]) == pytest.approx(-6.0)                                     # ... and away on any


def test_a_far_curved_hold_steps_down_by_5_mm_three_times_at_most_and_the_holds_after_it_start_there(setup,
                                                                                                    monkeypatch):
    """Silicone's first hold read 18 mm on agreeing frames: the tip goes down 5 mm, reads again where it aimed, and
    3 mm more; the holds after it, with no skin of its about them (_hold_start), start 5 mm over there and come down
    one step. A skin 30 mm under the tip takes three 5 mm steps and is read 15 mm under it; one that reads 3 mm off
    where the step aimed takes no second step."""
    ex, _, made = _gauged(setup, monkeypatch, _still_skin([18.03, 17.98, 18.0, 17.99, 18.0]), fresh=True, patches=False)
    assert ex._gauge_point(0, [0.0, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4))
    assert [m[2] for m in made] == pytest.approx([0.010, 0.005, 0.00202]) and ex._hold_low == pytest.approx(0.00798)
    assert ex._gauge_point(1, [0.01, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4))
    assert [m[2] for m in made[3:]] == pytest.approx([0.00702, 0.00202])
    ex, _, made = _gauged(setup, monkeypatch, _still_skin([30.0] * 5), fresh=True)
    assert ex._gauge_point(0, [0.0, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4))
    assert [m[2] for m in made] == pytest.approx([0.010, 0.005, 0.0, -0.005])
    assert ex.page_record["gauge"][-1]["height_m"] == pytest.approx(0.015)
    receding = iter([[18.0] * 5, [16.0] * 5])                                     # the skin sank 3 mm meanwhile
    ex, _, made = _gauged(setup, monkeypatch, lambda *a: (np.asarray(next(receding)) * 1e-3).tolist(), fresh=True)
    assert ex._gauge_point(0, [0.0, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4))
    assert [m[2] for m in made] == pytest.approx([0.010, 0.005])


def test_a_rise_on_frames_that_disagree_holds_for_the_holds_tries_and_the_holds_after_it_once_confirmed(setup,
                                                                                                        monkeypatch):
    """A reading whose frames disagree, one of them 3 mm under the tip, raises it 7 mm at once, and the hold's next
    try starts there; only a firm reading sets where the holds after it start (with no first hold's skin about them):
    none, and the next hold starts 5 mm over where the first began; one at the raised stand, and it starts 5 mm over
    that."""
    rise, apart = [3.0, 9.0, 12.0, 14.0, 16.0], [13.0, 8.5, 11.0, 12.0, 15.0]
    for tries, low, next_at in (([apart, apart], 0.0, 0.015), ([[11.0] * 5], -0.007, 0.022)):
        readings = iter([rise, apart, *tries, [10.0] * 5])
        ex, _, made = _gauged(setup, monkeypatch, lambda *a, r=readings: (np.asarray(next(r)) * 1e-3).tolist(),
                              fresh=True, patches=False)
        assert bool(ex._gauge_point(0, [0.0, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4))) == (low < 0)
        assert [m[2] for m in made] == pytest.approx([0.010] + [0.017] * (1 + len(tries)))
        assert ex._hold_low == pytest.approx(low)
        assert ex._gauge_point(1, [0.01, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4))
        assert made[-1][2] == pytest.approx(next_at)


def test_a_step_toward_the_skin_is_no_depth_for_the_holds_after_it_until_a_firm_reading_there_confirms_it(
        setup, monkeypatch):
    """The skin lies on the prior page. The first hold's first reading agrees but is 5 mm far: the tip goes 5 mm down,
    where a reading whose frames disagree has its nearest true, 5 mm under the tip, and raises it 5 mm again; a firm
    reading there needs no step. The next hold (no first hold's skin about it) starts 5 mm over that stand, not at the
    depth of the step taken before any reading there (5 mm over the skin, 2026-10-20's first rule). A single group's
    reading sets nothing."""
    def heights(i, xy, hover, frame):
        k = len([m for m in made if m[0] != "lift"]) - 1
        if k == 0:
            return [hover + 0.005] * 5                                             # agreeing, 5 mm far
        if k == 1:
            return [hover + v * 1e-3 for v in (0.0, 2.0, 3.0, 4.0, 7.0)]            # frames apart, the nearest true
        return [hover] * (1 if i == 2 else 5)
    ex, _, made = _gauged(setup, monkeypatch, heights, fresh=True, patches=False)
    for i in range(3):
        assert ex._gauge_point(i, [0.013 * i, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4))
    assert [(m[0], m[2]) for m in made] == [(0, pytest.approx(0.010)), (0, pytest.approx(0.005)),
                                            (0, pytest.approx(0.010)), (1, pytest.approx(0.015)),
                                            (1, pytest.approx(0.010)), (2, pytest.approx(0.015))]
    assert ex._hold_low == 0.0                                                    # the last, one group: none set


def test_a_curved_hold_after_the_first_starts_5_mm_over_the_depth_before_it(setup, monkeypatch):
    """The holds after the first fly to the depth the hold before them read, open-loop and unguarded: on 2026-10-18's
    silicone the fifth stood 1.1 mm over a skin some 9 mm higher than under the third, its frames agreeing.
    With no first hold's skin about it, each starts HOLD_STEP_M higher and comes down a firm step: there it stands
    6 mm over that skin, and rises."""
    skin = {0.0: 0.0, 0.013: 0.009}                                               # mm over the prior page, by x
    ex, _, made = _gauged(setup, monkeypatch, lambda i, xy, hover, frame: [hover - skin[round(float(xy[0]), 3)]] * 5,
                          fresh=True, patches=False)
    for i, x in enumerate((0.0, 0.013, 0.0)):
        assert ex._gauge_point(i, [x, 0.0], {"known": False, "level": 0.0}, 0.010, np.eye(4))
    over = [m[2] - skin[round(m[1][0], 3)] for m in made]
    assert over == pytest.approx([0.010, 0.006, 0.010, 0.024, 0.019, 0.014, 0.010])
    assert ex._hold_low == 0.0


def test_the_holds_after_a_firm_first_start_over_its_skin_where_its_patch_surrounds_them(setup, monkeypatch):
    """The holds after the first started HOLD_STEP_M over the stand the last firm reading needed no step at: 13.7-17.1
    mm over the skin, two readings or three each (2026-10-20). Within the patch of a first hold that read firm they
    start touch.start_above_m and HOLD_SKIN_M over its skin (a quadric through the patch) along its normal, which on a
    40 mm forearm reads within the hold's band at once; past the patch, or after a first hold read by too few frames to
    be firm, as before. Replayed over 63 set-ups (2026-10-13..20), 940 readings became 799, and no later hold started
    under 9.1 mm over its field (before: 8.1)."""
    radius = 0.040

    def over(xy):                                                               # the forearm's skin, m over the page
        return float(np.sqrt(radius ** 2 - xy[0] ** 2) - radius)
    for groups, firm in ((2, False), (5, True)):
        ex, program, made = _gauged(setup, monkeypatch, lambda i, xy, hover, frame, n=groups: [hover - over(xy)] * (
            n if i == 0 else 5), fresh=True)
        holds = curved_layout(program)
        assert ex._gauge_point(0, holds[0], {"known": False, "level": 0.0}, 0.010, np.eye(4))
        assert (ex._start is not None) == firm and ex._provisional is not None
        assert ex._gauge_point(1, holds[1], {"known": False, "level": 0.0}, 0.010, np.eye(4))
        stands = [m[2] for m in made if m[0] == 1]
        slope = abs(holds[1][0]) / np.sqrt(radius ** 2 - holds[1][0] ** 2)
        if firm:   # once, 11.5 mm over the skin along its normal
            assert len(stands) == 1 and stands[0] - over(holds[1]) == pytest.approx(
                0.0115 * np.sqrt(1 + slope ** 2), abs=3e-4)
        else:      # 5 mm over the first's stand, then a step down
            assert stands == pytest.approx([0.015, 0.010], abs=1e-6)
    low = ex._hold_low
    far = np.array([0.0, 0.05])                                                # past the firm first hold's patch
    ex._gauge_point(2, far, {"known": False, "level": 0.0}, 0.010, np.eye(4))
    assert [m[2] for m in made if m[0] == 2][0] == pytest.approx(0.010 - low + ex.HOLD_STEP_M)
    # 2026-10-24: there by the stand a hold stood 16.2 mm over the silicone and was held again; past the patch it
    # starts HOLD_SKIN_M over the skin the locate's over-centre views' depth puts there (_skin_over), when it has one
    grid = far + np.stack(np.meshgrid(*[np.linspace(-0.003, 0.003, 9)] * 2), -1).reshape(-1, 2)
    ex._over_skin = [{"points": np.c_[grid, np.full(len(grid), 0.003)], "tcp": np.array([0.0, 0.0, 0.050]),
                      "tip": np.array([0.0, 0.0, 0.0485])}]
    ex._gauge_point(3, far, {"known": False, "level": 0.0}, 0.010, np.eye(4))
    assert [m[2] for m in made if m[0] == 3][0] == pytest.approx(0.010 + 0.003 + 0.0015 + ex.HOLD_SKIN_M)


def curved_layout(program):
    from tatbot_session import curved

    return curved.layout(program, [0.043, 0.088])


def test_a_hold_along_the_skins_normal_steps_alike_and_never_moves_the_holds_after_it(setup, monkeypatch):
    """page.surface.along_skin: a hold stands 10 mm along its skin's normal and steps by the same rule, a disagreeing
    far reading not at all, and no step moves where the holds square to the page start."""
    ex, _, made = _gauged(setup, monkeypatch, _still_skin([18.0] * 5))
    frame = geometry.translation([0.0, 0.0, 0.002])
    read, _ = ex._gauge_try(2, [0.01, 0.0], 0.010, 0.010, np.eye(4), np.eye(4), frame=frame, check=True)
    assert [m[2] for m in made] == pytest.approx([0.010])                          # a check's: never down
    made.clear()
    monkeypatch.setattr(ex, "_holds_along_skin", lambda: True)
    monkeypatch.setattr(ex, "_along_skin", lambda skin, xy: frame)
    ex._gauge_try(2, [0.01, 0.0], 0.010, 0.010, np.eye(4), np.eye(4))
    assert [m[2] for m in made] == pytest.approx([0.010, 0.005, 0.002]) and all(m[3] is frame for m in made)
    assert ex._hold_low == 0.0
    made.clear()
    monkeypatch.setattr(ex, "_gauge_hold", lambda i, xy, hover, *a, **k: made.append(hover) or {
        "touch": i, "height_m": 0.015, "heights_m": [0.012, 0.014, 0.015, 0.0165, 0.018], "spread_m": 0.006})
    ex._gauge_try(2, [0.01, 0.0], 0.010, 0.010, np.eye(4), np.eye(4))
    assert made == [0.010]


def test_the_next_holds_hop_is_planned_while_a_hold_reads_and_sent_only_for_that_stand_unmoved(setup, monkeypatch):
    """Between two holds the arm stood ~1.9 s after a reading before the next hop went, about half of it planning it
    (2026-10-24). While a hold settles and reads, the hop to the stand its loop names next is planned on the planner;
    the next hold's travel sends it when it is to that same stand and the arm is within dispatch_drift (pen up) of
    where it was planned from, else plans afresh from the measured joints as before."""
    ex, program, skin = _curved_executor(setup, monkeypatch)
    ex._fit_surface(program)
    planned, sent = [], []

    def hop_plan(q, start, hop):   # FakeKin: the tip at q[:3]
        planned.append((np.asarray(q, float).copy(), hop))
        q0, q1 = np.asarray(q, float), np.r_[np.asarray(start)[:3, 3], np.zeros(4)]
        return SimpleNamespace(q=np.array([q0, q1]), tip=np.array([q0[:3], q1[:3]]), phase=np.array([1, 1]),
                               t=np.array([0.0, 1.0]))
    monkeypatch.setattr(ex, "_hop_plan", hop_plan)
    monkeypatch.setattr(ex, "move", lambda make, phase, what, ledger=None: sent.append(make()))
    monkeypatch.setattr(ex, "lowest_over_page", lambda traj, page=None: 1.0)
    from tatbot_motion import plan as motion_plan
    monkeypatch.setattr(tatbot_motion, "dispatch_drift", motion_plan.dispatch_drift)   # the fixture's always passes
    ex.motion = {**ex.motion, "dispatch_drift": {**ex.motion["dispatch_drift"], "pen_up_rad": 0.020, "pen_up_m": 0.002,
                                                 "pen_down_rad": 0.010, "pen_down_m": 0.0005}}   # motion.yaml's
    heading = ex.tcp()
    seed = np.array([0.0, 0.0, 0.010, 0.0, 0.0, 0.0, 0.0])
    nxt = geometry.translation([0.005, 0.0, 0.010])                             # FakeKin's tool turns as the stand's

    def at_hold(upcoming, moved=0.0):
        ex.io.q, ex._next_hold = seed.copy(), (lambda: upcoming)
        ex._plan_ahead()
        ex._ahead[1].result()   # planned on the planner's thread: done before the arm moves, so the count holds
        ex.io.q = seed + np.r_[moved, np.zeros(6)]
        ex._hop_from = (0, ex.kin.fk(ex.io.q)[:3, 3].copy())                 # standing at hold 0: a hop to hold 1
    at_hold(nxt)
    assert len(planned) == 1 and ex._next_hold is None and np.allclose(ex._ahead[0], nxt)
    ex._hold_travel(1, nxt)
    assert len(planned) == 1 and sent[-1] is not None and np.allclose(sent[-1].q[0], seed) and ex._ahead is None
    for upcoming, moved in ((geometry.translation([0.0, 0.005, 0.010]), 0.0), (nxt, 0.05)):
        before = len(planned)
        at_hold(upcoming, moved)                                               # another stand; the arm 50 mm off
        ex._hold_travel(1, nxt)
        assert len(planned) == before + 2 and np.allclose(planned[-1][0], ex.io.q)   # ahead, then afresh from here
    asked = []

    def gauge_point(i, xy, paper, above, heading, ledger=None):
        asked.append(None if ex._next_hold is None else ex._next_hold())
        return []
    monkeypatch.setattr(ex, "_gauge_point", gauge_point)
    ex._start, ex._hold_low, ex._hop_from = None, 0.0, None
    layout = [[0.0, 0.0], [0.005, 0.0], [0.0, 0.005]]
    ex.page_touches(layout)
    above = float(ex.motion["touch"].get("start_above_m", 0.010))
    assert asked[-1] is None and ex._next_hold is None
    assert np.allclose(asked[0], ex._next_curved_hold(layout[1], above, heading))


def test_a_check_hold_stands_over_the_field_steps_only_away_and_lifts_before_it_travels(setup, monkeypatch):
    """2026-10-20's checks stood at the holds' depth over the overhead's page, tilted 19 deg from the forearm's flank:
    5.6-7.9 mm over its skin by the field, one read 0.99 mm, and they travelled 28-44 mm between them unlifted. Each
    check stands 10 mm over the field along the holds' page normal, whatever depth the holds took; a reading nearer
    than 8 mm holds it again higher, one farther never lower; and it rises before it travels on: up and over to the
    next in the next's travel (_hold_travel), lifted off the last."""
    from tatbot_motion.surface import PageSurface
    from tatbot_session import curved

    first = {100: [0.95, 0.28, 1.2, 0.99, 2.36], 101: [13.0, 13.1, 12.9, 13.0, 13.05]}
    ex, program, made = _gauged(setup, monkeypatch, lambda i, xy, hover, frame: (
        np.asarray(first.get(i, [10.0] * 5)) * 1e-3 + hover - made_at[i]).tolist())
    ex.stack["page"]["surface"]["check"] = False
    ex._fit_surface(program)
    c, s = np.cos(np.radians(19.0)), np.sin(np.radians(19.0))
    ex._hold_frame = np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.001], [0.0, 0.0, 0.0, 1.0]])
    ex._hold_low = 0.012                                                           # the holds' depth: not the checks'
    holds = [r["xy"] for r in ex.page_record["gauge"] if r["touch"] < 100]
    points = curved.check_points(holds)
    surface = PageSurface(ex.used(), ex.surface_field)
    made_at = {100 + k: ex._over_field(surface, ex._hold_frame, xy, 0.010) for k, xy in enumerate(points)}
    made.clear()
    ex._check_holds(points, 100, "the skin between the holds")
    for k, xy in enumerate(points):
        tip = ex._hold_frame[:3, :3] @ [*xy, made_at[100 + k]] + ex._hold_frame[:3, 3]
        assert surface.height_over(tip)[0] == pytest.approx(0.010, abs=2e-5)
    assert [m[0] for m in made] == [100, 100, 101, 102, 103, "lift"]
    assert made[1][2] - made[0][2] == pytest.approx(0.00972)                        # 0.28 mm by its nearest: up
    checks = [r for r in ex.page_record["gauge"] if r.get("check")]
    assert [r["touch"] for r in checks] == [100, 101, 102, 103] and checks[0]["height_m"] == pytest.approx(0.01071)


def _hop_executor(setup, monkeypatch, refuse=False, dip_m=0.0):
    """setup's executor over the base's page with a travel planner that runs straight through its via points (the tip
    `dip_m` under them midway; refuse: PlanError for any via): the travels it plans, as their via lists."""
    from tatbot_motion import PlanError

    ex, travels = setup.ex, []
    ex.camera, ex.correction = np.eye(4), np.eye(4)

    def plan_travel(*, q_seed, base_from_tcp, kin, motion, via=None):
        travels.append(via)
        if via and refuse:
            raise PlanError("the tip trails its reference by 9.99 mm at t=1.00 s")
        tip = np.array([kin.fk(q_seed)[:3, 3], *(via or []), base_from_tcp[:3, 3]], float)
        tip[len(tip) // 2, 2] -= dip_m if via else 0.0
        return SimpleNamespace(t=np.arange(len(tip), dtype=float), q=np.column_stack([tip, np.zeros((len(tip), 4))]),
                               qd=np.zeros((len(tip), 7)), tip=tip, arc_m=np.full(len(tip), np.nan),
                               phase=np.full(len(tip), 2, np.uint8))
    monkeypatch.setattr(tatbot_motion, "plan_travel", plan_travel, raising=False)
    return ex, travels


def _stand(x, z, turn=0.0):   # a hold's stand over the base's page, the pen turned `turn` about its axis
    return geometry.rpy_matrix([x, 0.0, z], [0.0, 0.0, turn])


def test_a_hold_after_another_is_flown_to_up_and_over_in_one_move(setup, monkeypatch):
    """Between the 2026-10-20 tattoo's holds a 20 mm lift, a ~1 s stop and a travel down took 8.4-11.5 s and three of
    the arm's stick-slip starts and stops. From the stand another hold left the pen at, the next hold is one travel
    through tatbot_motion.hop_via: up along the page's normal, over, down onto its stand, with no lift. Its own steps
    travel straight; so does a hold the pen was lifted off since."""
    ex, travels = _hop_executor(setup, monkeypatch)
    ex._hold_travel(0, _stand(0.0, 0.010))
    assert travels == [None] and setup.lifts == []                       # nothing held before it: straight
    ex._hold_travel(1, _stand(0.035, 0.015))
    way = np.asarray(travels[-1])
    assert len(way) > 50 and setup.lifts == [] and len(setup.io.executed) == 2
    assert way[0, 2] > 0.010 and abs(way[0, 0]) < 1e-3                     # off the first along the normal...
    assert way[-1, 2] > 0.015 and abs(way[-1, 0] - 0.035) < 1e-3            # ...and down onto the second
    assert way[len(way) // 2, 2] == pytest.approx(0.0125 + 0.75 * MOTION["touch"]["lift_m"], abs=2e-4)   # midway
    ex._hold_travel(1, _stand(0.035, 0.010))                               # the hold's own step: straight
    assert travels[-1] is None and setup.lifts == []
    ex.lift("touch 1 lift")                                                  # lifted off since
    ex._hold_travel(2, _stand(0.0, 0.010))
    assert travels[-1] is None and setup.lifts == [MOTION["touch"]["lift_m"]]
    assert not [text for kind, _, text in setup.node.events if kind == "replan"]


def test_a_hold_the_arm_settled_an_encoder_count_off_hops_and_one_moved_off_lifts_first_and_says_so(setup, monkeypatch):
    """The arm settles a joint an encoder count after a hold's travel (up to 0.31 mm at the tip after 29 of the
    2026-10-20 tattoo's 56 holds): compared with the measured tip at 0.1 mm, half the hold changes neither hopped nor
    lifted and travelled straight at the holds' height. The stand its travel commanded is the hold's, within
    HOP_AT_M; a pen farther off it, moved since by anything but a lift, lifts off first and the event says why."""
    ex, travels = _hop_executor(setup, monkeypatch)
    ex._hold_travel(0, _stand(0.0, 0.010))
    setup.io.q[:3] += [0.00015, 0.0, -0.0001]                              # settled one count
    ex._hold_travel(1, _stand(0.035, 0.010))
    assert travels[-1] is not None and setup.lifts == []
    setup.io.q[2] += 0.004
    ex._hold_travel(2, _stand(0.0, 0.010))
    assert travels[-1] is None and setup.lifts == [MOTION["touch"]["lift_m"]]
    assert [text for kind, _, text in setup.node.events if kind == "replan"] == [
        "touch 2: no hop from touch 1 (the pen stands 4.0 mm off its stand): lifted off it first"]


def test_a_hop_a_latch_stopped_goes_on_up_and_over_from_where_it_stopped(setup, monkeypatch):
    """Continued after a latch, a hop is planned again up and over from where the arm stopped, never straight across
    at the height it had risen to."""
    ex, travels = _hop_executor(setup, monkeypatch)
    ex._hold_travel(0, _stand(0.0, 0.010))
    setup.io.script = [("latched", 2, {"latched": 1.0, "latch_reason": 1.0}, None)]
    operator = Operator(ex, rules.CONTINUE)
    ex._hold_travel(1, _stand(0.035, 0.010))
    assert operator.replies() == [(True, "")] and len(travels) == 3 and travels[-1] is not None
    assert np.allclose(setup.io.q[:3], [0.035, 0.0, 0.010]) and setup.lifts == []


@pytest.mark.parametrize("why", ["refused", "dips", "turns"])
def test_a_hop_the_arm_cannot_follow_dipping_or_turning_lifts_off_the_hold_first(setup, monkeypatch, why):
    """A hop the planner refuses, that would pass nearer the page than either end, or whose pen turns: the hold the pen
    stands at lifts along the page's normal as before, and the next hold is a straight travel from the lift's top."""
    ex, travels = _hop_executor(setup, monkeypatch, refuse=why == "refused", dip_m=0.020 if why == "dips" else 0.0)
    ex._hold_travel(0, _stand(0.0, 0.010))
    ex._hold_travel(1, _stand(0.035, 0.010, turn=0.05 if why == "turns" else 0.0))
    assert setup.lifts == [MOTION["touch"]["lift_m"]] and travels[-1] is None
    assert (len(travels) == 3) == (why != "turns")                        # the hop was planned, then not taken
    said = {"refused": "the tip trails its reference by 9.99 mm", "dips": "it passes", "turns": "its pen turns 0.050 rad"}
    assert [text for kind, _, text in setup.node.events if kind == "replan" and said[why] in text
            and text.startswith("touch 1: no hop from touch 0")]
    assert np.allclose(setup.io.q[:3], [0.035, 0.0, 0.010])


def test_a_pre_planned_hover_travel_starts_where_the_arm_stands(setup):
    """The hover travel pre-planned from the last stroke's untrimmed end, the arm 1.2 mm under it on the operator's
    trim (up to 2.0 mm, 2026-10-20, closed by the controller within 10 ms): it starts at the arm's joints and is the
    plan again after tatbot_motion.plan.START_BLEND_S."""
    from concurrent.futures import Future

    ex, io = setup.ex, setup.io
    t = np.linspace(0.0, 1.0, 101)
    q = np.zeros((101, 7))
    q[:, 0], q[:, 2] = np.linspace(0.0, 0.02, 101), 0.010
    travel = tatbot_motion.Trajectory(FakeKin.joint_names, t, q, np.zeros_like(q), q[:, :3].copy(), np.full(101, np.nan),
                                      np.full(101, 2, np.uint8))
    planned = Future()
    planned.set_result((travel, None))
    ex._preplan = ("s0000", planned, True, None)
    io.q = np.array([0.0, 0.0, 0.0088, 0.0, 0.0, 0.0, 0.0])
    out = ex._pre_travel(LINE)
    assert np.allclose(out.q[0], io.q) and np.allclose(out.tip[0], io.q[:3])
    later = t >= tatbot_motion.plan.START_BLEND_S
    assert np.allclose(out.q[later], q[later]) and np.allclose(travel.q, q)   # the pre-plan itself is left as it was


def test_gauge_holds_lift_off_the_last_only(setup, monkeypatch):
    """page.height.gauge use: the gauge holds are the touches, and each but the last leaves its lift to the next's
    travel (_hold_travel)."""
    ex = setup.ex
    ex.camera = np.eye(4)
    monkeypatch.setattr(ex, "_gauge_point", lambda i, xy, paper, above, heading, ledger=None: [np.array([*xy, 0.0])])
    contacts = ex.page_touches([[0.0, 0.0], [0.02, 0.0], [0.0, 0.02]])
    assert len(contacts) == 3 and setup.lifts == [MOTION["touch"]["lift_m"]]


# --- a curved page follows its stencil rigidly (curved.RigidWatch, ArmExecutor._rigid_follow, _skin_watch) ----------


class StubRigid(StubTracker):
    """A curved page's tracker: no stroke correction, the stencil watch fed by scripted hover readings (each the
    stencil's slide since the holds, in page metres, and the skin's height over the field there, or None)."""
    correcting = False

    def __init__(self, ex, readings):
        from tatbot_session import curved

        super().__init__([0.0, 0.0])
        self.ex, self.rigid, self.skin, self.readings = ex, curved.RigidWatch(), None, list(readings)

    def restart(self):
        super().restart()
        self.rigid.clear()

    def at_hover(self, name, op_id, plan_xy):
        self.hovers.append((name, op_id, np.asarray(plan_xy, float)))
        slide, skin_m = self.readings.pop(0) if self.readings else ((0.0, 0.0), 0.0)
        surface, at = self.ex.surface(), np.asarray(plan_xy, float)
        points, normals = surface.lift(at + np.random.default_rng(0).uniform(-0.012, 0.012, (80, 2)))
        self.skin = None if skin_m is None else points + skin_m * normals
        return self.rigid.add(at - np.asarray(slide), at, at - np.asarray(slide), at)


def _curved_hovering(setup, monkeypatch, readings, line=None):
    """The curved executor (a 40 mm forearm along page y, its field fitted) with hover stops and its stencil watch;
    the page setup keeps that page, and the planners take the surface. (ex, the stroke, plans' surfaces, set-ups)"""
    from tatbot_session import curved

    ex, program, _ = _curved_executor(setup, monkeypatch)
    ex._fit_surface(program)
    field = ex.surface_field
    ex.stack["page"]["track"] = {"hover_stop": True, "second_view_m": 0.005, "mode": "correct"}
    ex.motion = {**MOTION, "approach": {**MOTION["approach"], "final_m": 0.005}}
    ex.io.q = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0])

    def keep_page(program, touch, ledger=None):
        ex.camera, ex.correction, ex.surface_field, ex._surface_chart = np.eye(4), np.eye(4), field, None
        ex.rigid_total = (np.eye(3), np.eye(3))
    monkeypatch.setattr(ex, "setup_page", keep_page)
    surfaces = []

    def plan_op(op, *, base_from_page, q_seed, kin, motion, speed_m_s, from_arc_m=0.0, pen_down_at_start=False,
                lift_at_end=True, pen=None, surface=None, turn_pen=False):
        setup.calls.append({"op": op["id"], "points": np.asarray(op["points_m"], float), "seed": q_seed.copy()})
        surfaces.append(surface)
        return line_traj(op, from_arc_m)

    def plan_hover(op, *, base_from_page, q_seed, kin, motion, from_arc_m=0.0, pen=None, surface=None, turn_pen=False):
        setup.hovers.append({"op": op["id"], "start": np.asarray(op["points_m"][0], float)})
        q = np.array([q_seed, q_seed], float)
        q[1, :3] = [*op["points_m"][0], 0.01]
        return SimpleNamespace(t=np.array([0.0, 1.0]), q=q, qd=np.zeros((2, 7)), tip=q[:, :3],
                               arc_m=np.full(2, np.nan), phase=np.full(2, 1, np.uint8), info={"hover": True})
    monkeypatch.setattr(tatbot_motion, "plan_op", plan_op, raising=False)
    monkeypatch.setattr(tatbot_motion, "plan_hover", plan_hover, raising=False)
    ex.tracker = StubRigid(ex, readings)
    pages, again = [], None
    monkeypatch.setattr(ex, "_new_page", lambda program, touch, ledger: (pages.append(touch), keep_page(program, touch),
                                                                         ex._page_fresh()))   # setup_page's

    def restart(cls, ex_, since=0.0, centre=None):
        nonlocal again
        again = StubRigid(ex_, [])
        return again
    monkeypatch.setattr(ex_mod.track.Tracker, "start", classmethod(restart))
    stroke = line or {"op": "stroke", "id": "s0000", "closed": False, "points_m": [[0.0, 0.0], [0.0, 0.012]]}
    assert curved.config(ex.stack)["mode"] == "fitted" and ex._rigid() and ex._hover_on()
    return ex, stroke, surfaces, pages


def test_a_curved_pages_hover_stop_follows_a_moved_stencil_with_the_whole_surface(setup, monkeypatch):
    """The forearm slid 3 mm along itself between strokes: the hover reads it, a second view 5 mm off agrees, and the
    page (pose, height field and chart together) moves with it; the hover over the stroke's start on the followed page
    reads the stencil where it puts it, and the stroke goes down on that surface with its own points, uncorrected."""
    ex, stroke, surfaces, pages = _curved_hovering(setup, monkeypatch, [((0.0, 0.003), 0.0), ((0.0, 0.003), 0.0)])
    old = ex.used()
    setup.io.script = [("ok", 1, {}, None)] * 3 + [("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [stroke]}, setup.ledger, touch=False) == "complete"
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover", "s0000-hover2", "s0000-followed"]
    step = np.linalg.inv(old) @ ex.used()
    assert np.allclose(step[:3, 3], [0.0, 0.003, 0.0], atol=1e-9) and np.allclose(step[:3, :3], np.eye(3))
    assert pages == [] and np.allclose(setup.calls[-1]["points"], stroke["points_m"])   # its own points
    assert np.allclose(surfaces[-1].base, ex.used()) and surfaces[-1].height is ex.surface_field
    rows = [row for row in setup.ledger.rows() if row.get("event") == "page"]
    assert rows and rows[-1]["page"]["followed"][-1]["by"] == "wrist"
    assert any("the page and the skin's shape on it follow it" in text for _, _, text in setup.node.events)


def test_a_move_across_a_curved_arm_sets_the_page_up_again_rather_than_follow_on_a_guess(setup, monkeypatch):
    """3 mm across a 40 mm forearm: a slid arm and a print rolled round it read alike, and would put a stroke on its
    flank past ambiguity_m (1 mm since the owner doubled every stop) apart; the holds measure it again."""
    line = {"op": "stroke", "id": "s0000", "closed": False, "points_m": [[0.0, 0.0], [0.015, 0.0]]}
    ex, stroke, _, pages = _curved_hovering(setup, monkeypatch, [((0.003, 0.0), 0.0), ((0.003, 0.0), 0.0)], line)
    old = ex.used()
    setup.io.script = [("ok", 1, {}, None)] * 3 + [("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [stroke]}, setup.ledger, touch=False) == "complete"
    assert pages == [False] and np.allclose(ex.used(), old)
    assert any("across the curve" in text and "setting the page up again" in text for _, _, text in setup.node.events)


def test_a_curved_stencil_moved_past_the_rigid_limit_is_set_up_again(setup, monkeypatch):
    ex, stroke, _, pages = _curved_hovering(setup, monkeypatch, [((0.0, 0.012), 0.0), ((0.0, 0.012), 0.0)])
    setup.io.script = [("ok", 1, {}, None)] * 3 + [("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [stroke]}, setup.ledger, touch=False) == "complete"
    assert pages == [False]
    assert any("page.surface.rigid.max_m 10" in text for _, _, text in setup.node.events)


def test_a_still_curved_stencil_is_drawn_where_the_holds_put_it(setup, monkeypatch):
    """The readings' scatter under follow_m: no second view, no follow, no set-up."""
    ex, stroke, _, pages = _curved_hovering(setup, monkeypatch, [((0.0008, -0.0011), 0.0)])
    old = ex.used()
    setup.io.script = [("ok", 1, {}, None), ("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [stroke]}, setup.ledger, touch=False) == "complete"
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover"] and pages == [] and np.allclose(ex.used(), old)


@pytest.mark.parametrize("skin_m, mode, set_up", [(0.0022, "press", False), (0.003, "press", True),
                                                  (-0.003, "press", False), (0.003, "hover", False)])
def test_the_skin_under_a_curved_hover_is_read_against_the_field(setup, monkeypatch, skin_m, mode, set_up):
    """Past skin_m (2 mm) a warning (the depth keys trim it). Past skin_stop_m (2.5 mm) an inking pen's needles would
    go that much deeper over the field: the page is set up again at once, no pause; a skin under it, or a rehearsal,
    warns once (a still forearm read -1.6 mm at one hover and +0.5 at the next, 2026-10-20)."""
    ex, stroke, _, pages = _curved_hovering(setup, monkeypatch, [((0.0, 0.0), skin_m)])
    ex.motion["pen"] = {**ex.motion["pen"], "mode": mode, "hover": {"height_m": 0.008, "settle_s": 0.0}}
    setup.io.script = [("ok", 1, {}, None)] * 2 + [("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [stroke]}, setup.ledger, touch=False) == "complete"
    texts = [(kind, text) for kind, _, text in setup.node.events if "the skin about the tip" in text]
    assert texts and f"{abs(skin_m) * 1e3:.2f} mm {'over' if skin_m > 0 else 'under'} the field" in texts[0][1]
    assert not any(kind == "pause" for kind, _ in texts) and pages == ([False] if set_up else [])
    assert ("setting the page up again" in texts[0][1]) == set_up
    assert set_up or "past page.surface.rigid.skin_m" in texts[0][1]


@pytest.mark.parametrize("how", ["rise_m", "turned"])
def test_the_skin_watch_takes_out_a_surface_checks_rise(setup, monkeypatch, how):
    """The trim sweep covers keys in both directions; a surface check's carried rise is independent of them, and so is
    the turned checks' lift where the hover read (the arm's own error at the strokes' pose, _turned_checks)."""
    skin_m, trim_m, rise_m, set_up = 0.0035, 0.0, 0.0012, False
    ex, stroke, _, pages = _curved_hovering(setup, monkeypatch, [((0.0, 0.0), skin_m)])
    ex.motion["pen"] = {**ex.motion["pen"], "mode": "press"}
    ex.tracker.at_hover("s0000-hover", "s0000", np.zeros(2))
    monkeypatch.setattr(ex, "trim_target", lambda: trim_m)
    ex.page_record.setdefault("surface", {})[how] = rise_m if how == "rise_m" else [
        {"xy": [0.0, 0.03], "lift_m": rise_m}, {"xy": [0.0, -0.03], "lift_m": rise_m}]
    assert ex._skin_watch({"ops": [stroke]}, stroke, 0, setup.ledger) is set_up and pages == ([False] if set_up else [])
    said = [text for _, _, text in setup.node.events if "the skin about the tip" in text][-1]
    assert f"{(skin_m - trim_m - rise_m) * 1e3:+.2f} mm net of the pen's {(trim_m + rise_m) * 1e3:+.1f} mm lift" in said
    keys_past = not set_up and skin_m - trim_m - rise_m > 0.0025
    assert ("past skin_stop_m only with the depth keys' share, which a set-up keeps" in said) == keys_past


def _watching(setup, monkeypatch, skin_m, quick=False):
    """_curved_hovering with an inking pen and one hover's skin reading: watch(trim in key presses, + up, as on_key
    keeps it; the side the hover before read past its stop) -> (set up?, what the hover said), on that same reading.
    quick (a sweep's hundreds of readings): the frame reads skin_m itself, and a set-up is only counted."""
    ex, stroke, _, pages = _curved_hovering(setup, monkeypatch, [((0.0, 0.0), skin_m)])
    ex.motion["pen"] = {**ex.motion["pen"], "mode": "press"}
    ex.pen, tracker, trim = tatbot_motion.pen_down(ex.motion, None), ex.tracker, [0.0]   # draw()'s: the ride is on
    tracker.at_hover("s0000-hover", "s0000", np.zeros(2))
    monkeypatch.setattr(ex, "trim_target", lambda: trim[0])
    if quick:
        monkeypatch.setattr(ex_mod.curved, "skin_gap", lambda surface, skin: skin_m)
        monkeypatch.setattr(ex, "_reacquire", lambda program, ledger, why, op_id="": pages.append(False))

    def watch(steps, before=0.0):
        trim[0], ex.tracker, ex._fresh, ex._skin_off = steps * MOTION["pen"]["trim"]["step_m"], tracker, False, before
        pages.clear()
        set_up = ex._skin_watch({"ops": [stroke]}, stroke, 0, setup.ledger)
        assert pages == ([False] if set_up else [])
        return set_up, ([""] + [text for _, _, text in setup.node.events if "the skin about the tip" in text])[-1]
    return ex, watch


@pytest.mark.parametrize("skin_m", [0.0, 0.00245, 0.00265, 0.00405, -0.00495, -0.00515,
                                    -0.00805])
def test_the_skin_watch_judges_a_skin_alike_at_every_trim(setup, monkeypatch, skin_m):
    """review, 2026-10-21: judged net of the keys, a skin 0.1-0.5 mm over set the page up at once with the trim at
    -2.0..-2.5 mm, and one press past -2.5 stopped any skin under 2.5 mm over from setting it up. Across the whole
    +-4 mm trim, press by press, the skin's part is judged as at trim 0: keys pressed its way never set the page up
    nor keep it from it, keys pressed against it only take theirs out, so deeper keys never weaken the deep side."""
    _, watch = _watching(setup, monkeypatch, skin_m, quick=True)
    step, limit = MOTION["pen"]["trim"]["step_m"], MOTION["pen"]["trim"]["limit_m"]
    most = int(limit / step + 1e-6)
    deep, under = {}, {}
    for steps in range(most, -most - 1, -1):
        trim = steps * step
        deep[steps], under[steps] = watch(steps)[0], watch(steps, before=-1.0)[0]
        assert deep[steps] == (skin_m > 0.0025 + max(trim, 0.0))       # over: at once
        assert under[steps] == (deep[steps] or skin_m < -0.005 - max(-trim, 0.0))   # under: the hover before too
    assert all(deep[s] == deep[0] for s in deep if s <= 0)              # alike at every deep trim
    assert all(deep[s] <= deep[s - 1] for s in deep if s > -most)       # never weaker as the keys press deeper
    assert all(under[s] == under[0] for s in under if s >= 0 and skin_m < 0)   # keys up never set up a sunk skin


@pytest.mark.parametrize("skin_m, steps, says", [
    (0.0003, -20, "the stroke rides 0.30 mm higher by the skin's part; 2.0 mm of it is the depth keys')"),
    (-0.0003, -24, "the needles would ride that much deeper; the depth keys trim it; 2.1 mm of it is the depth keys')"),
    (0.0022, 0, "the stroke rides 2.20 mm higher by the skin's part)"),
    (0.0012, -30, "the stroke rides 1.20 mm higher by the skin's part; 3.0 mm of it is the depth keys'; past "
                  "skin_stop_m only with the depth keys' share, which a set-up keeps)")])
def test_the_skin_watch_says_what_the_ride_lifts_and_what_the_depth_keys_press(setup, monkeypatch, skin_m, steps, says):
    """review, 2026-10-21: past skin_m with the trim at -2.0 the warning said the stroke rides higher by it, but the
    ride lifts it by the skin's part alone and the keys' share stays. It says each: what the ride lifts, and how much
    of the net is the depth keys' (all of it where the skin reads under the field)."""
    ex, watch = _watching(setup, monkeypatch, skin_m)
    assert ex._rides_measured()
    set_up, said = watch(steps)
    assert not set_up and f"past page.surface.rigid.skin_m 2.0: {says}" in said


def test_the_skin_under_a_curved_hover_is_read_with_the_strokes_corrected_too(setup, monkeypatch):
    """page.surface.track_correct on (no rigid watch): the hover's skin is still read against the field. On
    2026-10-18 a forearm read 4.4 mm under its field from its first hover, and its strokes went down 2-6 mm off before
    the off-plan pause five strokes later: two hovers that read it past skin_stop_under_m (5 mm since the owner
    doubled every stop), one way, set the page up again."""
    ex, stroke, _, pages = _curved_hovering(setup, monkeypatch, [((0.0, 0.0), -0.0055), ((0.0, 0.0), -0.0055)])
    rigid = ex.tracker

    class Correcting(StubTracker):
        def at_hover(self, name, op_id, plan_xy):
            rigid.at_hover(name, op_id, plan_xy)
            self.skin = rigid.skin
            return super().at_hover(name, op_id, plan_xy)
    ex.tracker = Correcting([0.0, 0.0])
    assert not ex._rigid() and ex._hover_on()
    second = {**stroke, "id": "s0001", "points_m": [[0.0, 0.004], [0.0, 0.010]]}
    setup.io.script = [("ok", 1, {}, None), ("ok", 10, {}, UP), ("ok", 1, {}, None), ("ok", 10, {}, UP)]
    assert draw_strokes(ex, {"ops": [stroke, second]}, setup.ledger, touch=False) == "complete"
    said = [text for _, _, text in setup.node.events if "the skin about the tip" in text]
    assert len(said) == 3 and "5.50 mm under the field" in said[0] and "next hover's too" in said[0]
    assert "setting the page up again" in said[1] and "s0001's hover" in said[1] and pages == [False]
    assert "0.00 mm" in said[2] and "setting" not in said[2]   # the fresh page's rehover is read too
    assert not any(kind == "pause" for kind, _, _ in setup.node.events)


def _riding_hovers(setup, monkeypatch, readings, ops=None, mode="press"):
    """_curved_hovering with every stroke's planned pen recorded: (ex, program, pens: [(op id, height_m)], pages)."""
    ex, stroke, _, pages = _curved_hovering(setup, monkeypatch, readings)
    ex.motion["pen"] = {**ex.motion["pen"], "mode": mode, "hover": {"height_m": 0.008, "settle_s": 0.0}}
    planned, pens, first = tatbot_motion.plan_op, [], ex.tracker

    def restart(cls, ex_, since=0.0, centre=None):   # a page set up again: the next readings on the same hover list
        again = StubRigid(ex_, [])
        again.readings, again.hovers = first.readings, first.hovers
        return again
    monkeypatch.setattr(ex_mod.track.Tracker, "start", classmethod(restart))

    def plan_op(op, **kw):
        pens.append((op["id"], kw["pen"].height_m))
        return planned(op, **kw)
    monkeypatch.setattr(tatbot_motion, "plan_op", plan_op, raising=False)
    return ex, {"ops": ops or [stroke]}, pens, pages


def _rides(ledger):
    return {row["op"]: row.get("ride_m", 0.0) for row in ledger.rows() if row.get("event") == "sent"}


@pytest.mark.parametrize("skin_m, rise_m, mode, ride_m", [(0.0015, 0.0, "press", 0.0015), (0.0015, 0.0005, "press", 0.001),
                                                          (-0.002, 0.0, "press", 0.0), (0.0015, 0.0, "hover", 0.0)])
def test_a_curved_inking_stroke_rides_higher_by_the_skin_its_hover_read_over_the_field(setup, monkeypatch, skin_m,
                                                                                       rise_m, mode, ride_m):
    """2026-10-20, the forearm tattoo: 17 of its 34 hovers read the skin 1.0-2.3 mm over the field, under the 2.5 mm
    net stop, and each stroke went in that much past its plan. The stroke now rides higher by the skin its hover read
    over the field, less the surface check's rise the pen rides by already; never lower (a skin under it), and a
    rehearsal's hover rides as planned."""
    ex, program, pens, pages = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), skin_m)], mode=mode)
    ex.page_record["surface"]["rise_m"] = rise_m
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete"
    assert pages == [] and pens == [("s0000", pytest.approx(ex.pen.height_m + ride_m, abs=1e-9))]
    assert _rides(setup.ledger) == {"s0000": pytest.approx(ride_m, abs=1e-9)}
    assert any(f"rides {ride_m * 1e3:.2f} mm higher" in t for _, _, t in setup.node.events) == bool(ride_m)


@pytest.mark.parametrize("tilt_m, pages_set", [(0.0025, []), (0.0045, [False])])
def test_a_curved_stroke_rides_the_skin_its_hover_read_along_its_path(setup, monkeypatch, tilt_m, pages_set):
    """The tiny heart (2026-10-21): its skin sank on one lobe and rose on the other under one field while the hover's
    ring about the tip read it level, and one half went deep where the other missed. Each hover's skin 8-40 mm about the
    tip (track.Tracker.wide) is now read along the stroke's own path: the stroke rides it, up freely, under its plan at
    most 1 mm, one line said; a skin past 4 mm off the field along it (2 mm until 2026-10-23) sets the page up again
    before the stroke (not at a fresh page's first hover, the rehover after that set-up), which then rides the skin
    read again."""
    from tatbot_session import curved

    line = [{"op": "stroke", "id": f"s000{k}", "closed": False, "points_m": [[0.0, 0.004 * k], [0.0, 0.004 * k + 0.012]]}
            for k in range(2)]
    ex, program, pens, pages = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), 0.0)] * 4, ops=line)
    surfaces, planned = [], tatbot_motion.plan_op

    def plan_op(op, **kw):
        surfaces.append((op["id"], kw.get("surface")))
        return planned(op, **kw)
    monkeypatch.setattr(tatbot_motion, "plan_op", plan_op, raising=False)
    at_hover, rng = StubRigid.at_hover, np.random.default_rng(9)

    def tilted(self, name, op_id, plan_xy):   # the first stroke's skin level, the second's tilting along it
        out = at_hover(self, name, op_id, plan_xy)
        start = np.asarray(next(o for o in line if o["id"] == op_id)["points_m"][0], float)
        uv = start + np.c_[rng.uniform(-0.004, 0.004, 3000), rng.uniform(-0.003, 0.015, 3000)]
        points, normals = self.ex.surface().lift(uv)
        skin = (0.0 if op_id == "s0000" else tilt_m) * np.clip((uv[:, 1] - start[1] - 0.006) / 0.006, -1.0, 1.0)
        self.wide = [*getattr(self, "wide", []), (self.ex.clock.now(), op_id, points + skin[:, None] * normals)]
        return out
    monkeypatch.setattr(StubRigid, "at_hover", tilted)
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete"
    assert pages == pages_set
    said = [t for _, _, t in setup.node.events if "rides its skin" in t]
    assert len(said) == 2 and f"set up again for it: {'yes' if pages_set else 'no'}" in said[1]
    rode = [s for op_id, s in surfaces if op_id == "s0001"][-1]
    assert isinstance(rode, curved.Ridden)
    ends = rode.at(np.array(line[1]["points_m"])) + _rides(setup.ledger)["s0001"]
    assert ends[0] == pytest.approx(-0.001, abs=1e-6)            # where it sank: 1 mm under its plan at most
    assert ends[1] == pytest.approx(tilt_m, abs=0.0004)          # where it rose: followed
    assert not isinstance(next(s for op_id, s in surfaces if op_id == "s0000"), curved.Ridden)   # level: as planned


def _borrowing(setup, monkeypatch):
    """A curved riding executor on Ticking time, a 10 mm stroke whose own hover reads only its last 2 mm (the pen hides
    8 mm about its tip), and skin frames standing `off` over the field over the rest of it."""
    ex, program, _, _ = _riding_hovers(setup, monkeypatch, [])
    ex.clock = Ticking(1000.0)
    path = np.c_[np.zeros(11), np.linspace(0.0, 0.010, 11)]
    surface, rng = ex.surface(), np.random.default_rng(4)

    def frame(lo, hi, off):
        uv = np.c_[rng.uniform(-0.003, 0.003, 2000), rng.uniform(lo, hi, 2000)]
        points, normals = surface.lift(uv)
        return points + off * normals
    return ex, path, frame


def test_a_strokes_blind_start_borrows_the_skin_an_earlier_strokes_frame_read_on_its_page(setup, monkeypatch):
    """slow-10mm (2026-10-21): 7-10 mm strokes lie mostly inside the 8 mm the pen hides from its own hover. Their unread
    points take the skin the page's frames of earlier strokes read (hovers, second views, lifts), newest first, within
    profile_keep_s; their own hover's reading stands where it reads; a frame from before the page was set up again, or
    older than that, lends nothing, and those points ride their fallback, never under the plan."""
    ex, path, frame = _borrowing(setup, monkeypatch)
    own = (999.0, "s0001", frame(0.008, 0.013, -0.0004))
    ex.tracker.wide = [(990.0, "s0000", frame(-0.003, 0.013, -0.0008)), (995.0, "s0000", frame(-0.003, 0.013, -0.0006)),
                       own]
    part, ride = ex._profile("s0001", path, 0.0002, 0.0)
    assert np.isfinite(part).all()
    assert part[0] == pytest.approx(-0.0006, abs=0.0002) and part[-1] == pytest.approx(-0.0004, abs=0.0002)  # newest
    ex.tracker.wide = [(930.0, "s0000", frame(-0.003, 0.013, -0.0008)), own]          # 70 s old: past 60 s
    part, ride = ex._profile("s0001", path, 0.0002, 0.0)
    assert not np.isfinite(part[:4]).any() and np.all(ride[:4] >= 0.0002 - 1e-12)
    ex.tracker.wide = [(995.0, "s0000", frame(-0.003, 0.013, -0.0008)), own]
    ex._page_fresh()                                                                     # set up again: all gone
    assert ex._profile("s0001", path, 0.0002, 0.0) is None
    ex.tracker.wide = [(995.0, "s0000", frame(-0.003, 0.013, -0.0008))]                 # no own reading: as before
    assert ex._profile("s0001", path, 0.0002, 0.0) is None


@pytest.mark.parametrize("lent_m, used", [(-0.0006, True), (0.0008, False)])
def test_a_stroke_its_own_hover_reads_none_of_rides_a_lent_frame_only_where_it_agrees_with_its_own_skin(
        setup, monkeypatch, lent_m, used):
    """slow-10mm: the own hover reads none of a 10 mm stroke (its views reach only past the 8 mm the pen hides), only
    the skin beyond it. A lent frame builds its whole profile when its skin agrees with those views' within
    profile_agree_m (1 mm: a minute's ~0.8 mm drift and the views' 0.03-0.10 mm scatter); one 1.2 mm off it lends
    nothing, and the stroke rides its fallback as before."""
    ex, path, frame = _borrowing(setup, monkeypatch)
    ex.tracker.wide = [(995.0, "s0000", frame(-0.003, 0.018, lent_m)), (999.0, "s0001", frame(0.012, 0.018, -0.0004))]
    got = ex._profile("s0001", path, 0.0002, 0.0)
    if not used:
        assert got is None
        return
    part, ride = got
    assert np.isfinite(part).all() and np.allclose(part, lent_m, atol=0.0002) and np.allclose(ride, lent_m, atol=0.0002)


def test_a_skin_still_past_the_stop_on_fresh_holds_rides_higher_and_sets_up_again_only_past_it(setup, monkeypatch):
    """A hover past skin_stop_m sets the page up again, but on 2026-10-20 the forearm's fresh holds read its skin on the
    field where its hovers still read it 1-1.8 mm over, and the next stroke went down at the same excess unjudged. The
    rehover on the fresh page is read: past the stop it sets nothing up, the stroke rides higher by it, and a later
    hover sets the page up again only FLOOR_M past what it read."""
    line = [{"op": "stroke", "id": f"s000{k}", "closed": False, "points_m": [[0.0, 0.004 * k], [0.0, 0.004 * k + 0.012]]}
            for k in range(3)]
    ex, program, pens, pages = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), 0.003), ((0.0, 0.0), 0.0032),
                                                                   ((0.0, 0.0), 0.0035), ((0.0, 0.0), 0.0045),
                                                                   ((0.0, 0.0), 0.0)], ops=line)
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete"
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover", "s0000-rehover", "s0001-hover", "s0002-hover",
                                                 "s0002-rehover"]
    assert pages == [False, False]                        # at s0000's 3.0 and s0002's 4.5, past 3.2 + 1 mm
    assert _rides(setup.ledger) == {"s0000": pytest.approx(0.0032), "s0001": pytest.approx(0.0035), "s0002": 0.0}
    said = [t for _, _, t in setup.node.events if "the skin about the tip" in t]
    assert "read so on fresh holds" in said[1] and "within what this page's first hover read" in said[2]


def test_a_hover_that_reads_no_skin_rides_by_the_hovers_about_its_path_or_the_stop(setup, monkeypatch):
    """review, 2026-10-21: a hover whose gauge reads no skin (15 on 2026-10-19's forearm runs) put its stroke down on the
    bare field. It rides by the most the page's hovers within RIDE_AROUND_M of its path read over the field, and with
    none skin_stop_m higher. A stroke whose path, past its own hover's ring (RIDE_AROUND_M from its start), passes
    within RIDE_NEAR_M of a hover that read more rides by that: a forearm's crown read 3 mm over its flanks."""
    line = [{"op": "stroke", "id": "s0000", "closed": False, "points_m": [[0.0, 0.0], [0.0, 0.012]]},
            {"op": "stroke", "id": "s0001", "closed": False, "points_m": [[0.0, 0.010], [0.0, 0.020]]},
            {"op": "stroke", "id": "s0002", "closed": False, "points_m": [[0.004, -0.030], [0.004, 0.005]]},
            {"op": "stroke", "id": "s0003", "closed": False, "points_m": [[0.030, 0.0], [0.030, 0.012]]}]
    ex, program, pens, pages = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), 0.0012), ((0.0, 0.0), None),
                                                                   ((0.0, 0.0), 0.0004), ((0.0, 0.0), None)], ops=line)
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete"
    rides = _rides(setup.ledger)
    assert rides["s0000"] == pytest.approx(0.0012) and rides["s0001"] == pytest.approx(0.0012)   # within 20 mm
    assert rides["s0002"] == pytest.approx(0.0012)        # 30 mm on, its path passes 4 mm from s0000's hover
    assert rides["s0003"] == pytest.approx(0.0025)        # nothing read within 20 mm: skin_stop_m
    assert pages == [] and any("rides page.surface.rigid.skin_stop_m 2.5 mm higher" in t for _, _, t in setup.node.events)


def test_a_continuation_joined_pen_down_keeps_its_chains_ride_and_a_preplan_at_another_ride_is_planned_again(
        setup, monkeypatch):
    """A stroke and the continuation it joins pen-down ride at one height, the most read over both their paths, so the
    continuation joins where the tip stands (no descent unmeasured); the next stroke's pre-plan, made at that ride and
    here with no approach to ease the difference in over (_folded), is planned again at its own."""
    line = [{"op": "stroke", "id": "s0000", "closed": False, "points_m": [[0.0, 0.0], [0.0, 0.012]]},
            {"op": "stroke", "id": "s0001", "closed": False, "continues": True, "points_m": [[0.0, 0.012], [0.0, 0.024]]},
            {"op": "stroke", "id": "s0002", "closed": False, "points_m": [[0.0, 0.026], [0.0, 0.03]]}]
    ex, program, pens, pages = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), 0.001), ((0.0, 0.0), 0.0)], ops=line)
    ex._skins.append((time.monotonic(), np.array([0.0, 0.018]), 0.0016, "s9999"))   # an earlier hover by s0001's path
    ex.clock = SimpleNamespace(now=time.monotonic, utc=time.time, sleep=lambda s: None)
    setup.io.script = [("ok", 1, {}, None), ("ok", 10, {}, None), ("ok", 10, {}, None), ("ok", 1, {}, None),
                       ("ok", 10, {}, UP)]
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete"
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover", "s0002-hover"]       # the continuation joined
    rides = _rides(setup.ledger)
    assert rides["s0000"] == rides["s0001"] == pytest.approx(0.0016) and rides["s0002"] == 0.0
    heights = dict(pens)
    assert heights["s0002"] == pytest.approx(ex.pen.height_m)                        # planned again, unridden
    assert [op for op, _ in pens].count("s0002") == 2


def _approach_plan(op, descent_s=3.0, standoff_m=0.010):
    """A stroke plan from its hover: down the page's normal from the standoff over descent_s, then along its line."""
    t = np.linspace(0.0, descent_s + 1.0, 41)
    line = line_traj(op, n=41).tip
    down = t <= descent_s
    tip = np.column_stack([np.where(down, line[0, 0], line[:, 0]), np.where(down, line[0, 1], line[:, 1]),
                           np.where(down, standoff_m * (1.0 - t / descent_s), 0.0)])
    arc = np.where(down, 0.0, np.linspace(0.0, geometry.stroke_length(op["points_m"]), 41))
    return stroke_plan(t, tip, arc, np.where(down, 3, 5).astype(np.uint8))


@pytest.mark.parametrize("ride_mm, trim_mm, folds", [(0.0, 0.0, True), (1.6, 0.0, True), (-1.0, 0.0, True),
                                                     (1.6, -3.0, True), (3.0, 0.0, False), (-1.6, -3.0, False)])
def test_a_preplan_at_another_ride_is_sent_by_the_difference_when_it_eases_in_before_the_needles_near_the_skin(
        riding, monkeypatch, ride_mm, trim_mm, folds):
    """review, 2026-10-21: a stroke's ride is set at its own hover, its pre-plan at the ride before, and the ride
    planned the stroke again at dispatch on 67 of the forearm tattoo's 112 stroke pairs, 2-6 s each at the hover with
    the machine on. The pre-plan is sent by the difference along the depth axis as the trim is, eased in before the tip
    comes FOLD_CLEAR_M more than any lift over its first pen-down point, within the trim's limit; else planned again."""
    import dataclasses
    from concurrent.futures import Future

    ex = riding.ex
    ex.pen, ex.resource = tatbot_motion.pen_down(RIDE, BALLPOINT), {"id": "fitted", "dip": None, "tool": {}}
    ex.trims["fitted"], ex._ride = trim_mm * 1e-3, ("s0000", 0.0005 + ride_mm * 1e-3)
    pre, done = _approach_plan(LINE), Future()
    done.set_result((None, pre))
    ex._preplan = ("s0000", done, True, None, dataclasses.replace(ex.pen, height_m=ex.pen.height_m + 0.0005))
    monkeypatch.setattr(ex, "lift_at_end", lambda *a, **k: True)
    fresh = []
    monkeypatch.setattr(ex, "plan", lambda *a, **k: fresh.append(a) or line_traj(LINE))
    traj, sent = ex.dispatch_plan({"ops": [LINE]}, LINE, 0.0)
    assert (traj is pre and not fresh) == folds and ex._fold == pytest.approx(ride_mm * 1e-3 if folds else 0.0)
    if folds:   # the needles go down by the stroke's own ride: the first pen-down knot that much higher (+) or lower
        k = int(np.flatnonzero(pre.phase == 5)[0])
        assert sent.tip[k, 2] - pre.tip[k, 2] == pytest.approx((ride_mm + trim_mm) * 1e-3)
        assert sent.info["trim"].settled_s() <= float(pre.t[k])
    assert any("its pre-plan sent" in t for _, _, t in riding.node.events) == (ride_mm != 0 and abs(ride_mm + trim_mm) < 4)


def test_a_preplan_made_during_a_folded_stroke_is_planned_at_that_strokes_own_plan(riding, monkeypatch):
    """The next pre-plan starts at the running plan's end, which is at the ride that plan was made at: it is planned
    there and folded on, so a continuation joined pen-down keeps its chain's depth instead of easing back under it."""
    ex = riding.ex
    ex.pen, ex._ride, ex._fold = tatbot_motion.pen_down(RIDE, BALLPOINT), ("s0000", 0.0016), -0.0006
    pens = []
    monkeypatch.setattr(ex, "_preplanned", lambda program, op, seed, lift, snap, hover, pen: pens.append(pen))
    monkeypatch.setattr(ex, "lift_at_end", lambda *a, **k: True)
    ex.preplan({"ops": [LINE, dict(LINE2, continues=True)]}, 0, np.zeros(7), [])
    ex._preplan[1].result()
    assert pens[0].height_m == pytest.approx(ex.pen.height_m + 0.0022)


@pytest.mark.parametrize("skin_mm, keys_mm, used_mm", [
    (-0.15, -4.0, -2.65),   # read 0.15 mm under the field: the keys' -4.0 held 2.5 mm past it
    (-4.06, -4.0, -4.0),    # read 4.06 mm under: the keys' whole depth
    (-0.25, -2.7, -2.7),    # within the cap: as keyed
    (0.25, -4.0, -2.5),     # read over the field: the ride lifts the stroke by it, the trim no deeper than 2.5 mm
    (None, -4.0, -2.5),     # no skin read about it: as if at the field
    (-1.0, 0.7, 0.7)])      # keys pressed up: as keyed
def test_a_needle_strokes_trim_goes_past_skin_stop_m_only_as_far_as_the_skin_read_along_it_sank(setup, monkeypatch, skin_mm,
                                                                                                keys_mm, used_mm):
    """review, 2026-10-21: with the +-4 mm trim one trim per cartridge carried depth keyed for one sunk stretch of a
    forearm onto strokes it was never pressed for, up to the 3RL's needles 3.0 mm past the gauged skin. A needle
    cartridge's stroke takes the trim no deeper than skin_stop_m (2.5 mm, the operator's deepest rung) past its plan over
    the highest skin read along it, says when that holds the keys back, and the sent row keeps the cap."""
    ex, program, _, pages = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), None if skin_mm is None else skin_mm * 1e-3)],
                                          mode="ride")
    _needles(setup, monkeypatch)
    ex.trims["fitted"] = keys_mm * 1e-3
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete" and pages == []
    cap = -0.0025 + min(0.0, skin_mm or 0.0) * 1e-3
    assert _trims_sent(setup.ledger, setup.io.executed) == {"s0000": (pytest.approx(cap), pytest.approx(used_mm * 1e-3))}
    held = [text for kind, _, text in setup.node.events if kind == "pen_trim" and " held at " in text]
    assert held == ([f"fitted {keys_mm:+.1f} mm held at {used_mm:+.2f} mm on s0000: the needles go no deeper than "
                     "page.surface.rigid.skin_stop_m past its plan, " +
                     (f"the skin read along it {skin_mm:+.2f} mm over it at most" if skin_mm is not None
                      else "no skin read at its own hover")] if used_mm > keys_mm else [])
    assert ex.trims == {"fitted": pytest.approx(keys_mm * 1e-3)}   # the keys' own trim stays theirs


def test_a_needle_strokes_trim_is_held_by_the_highest_skin_read_along_its_path_and_a_ballpoints_never(setup, monkeypatch):
    """The cap takes the readings the stroke's ride lifts it by: a stroke whose own hover read the skin 4 mm under the
    field, but whose path passes a hover that read it on the field, goes no deeper than skin_stop_m past its plan there.
    A pen with no needles (the ballpoint riding) takes the keys as they are."""
    line = [{"op": "stroke", "id": "s0000", "closed": False, "points_m": [[0.0, 0.0], [0.0, 0.012]]},
            {"op": "stroke", "id": "s0001", "closed": False, "points_m": [[0.004, -0.030], [0.004, 0.005]]},
            {"op": "stroke", "id": "s0002", "closed": False, "points_m": [[0.030, 0.0], [0.030, 0.012]]}]
    ex, program, _, _ = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), 0.0), ((0.0, 0.0), -0.004),
                                                            ((0.0, 0.0), -0.004)], ops=line, mode="ride")
    _needles(setup, monkeypatch)
    ex.trims["fitted"] = -0.004
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete"
    sent = _trims_sent(setup.ledger, setup.io.executed)
    assert sent["s0001"] == (pytest.approx(-0.0025), pytest.approx(-0.0025))   # 30 mm on, 4 mm from s0000's hover
    assert sent["s0002"] == (pytest.approx(-0.0065), pytest.approx(-0.004))    # its own hover only: 4 mm sunk
    monkeypatch.setattr(ex_mod.config, "tool_datasheet", lambda repo, tool_id: BALLPOINT)
    ex.prepare_pen({})
    assert ex.pen.machine and ex._cap_for([0.0]) is None and ex._capped(ex._cap_for([0.0])) == pytest.approx(-0.004)


def test_a_needle_stroke_whose_own_hover_read_no_skin_takes_skin_stop_m_whatever_its_neighbours_read(setup, monkeypatch):
    """review, 2026-10-21: a stroke whose own hover read no skin took its cap from the hovers about its path, so keys
    pressed to -4.0 over a neighbour read 4 mm sunk reached it, 10 mm on, at their whole depth with nothing said. Its
    neighbours' readings still lift it (none here: read under), but its trim goes no deeper than skin_stop_m, and the
    hold names why."""
    line = [{"op": "stroke", "id": "s0000", "closed": False, "points_m": [[0.0, 0.0], [0.0, 0.012]]},
            {"op": "stroke", "id": "s0001", "closed": False, "points_m": [[0.0, 0.010], [0.0, 0.020]]}]
    ex, program, _, pages = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), -0.004), ((0.0, 0.0), None)], ops=line,
                                           mode="ride")
    _needles(setup, monkeypatch)
    ex.trims["fitted"] = -0.004
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete" and pages == []
    assert _rides(setup.ledger) == {"s0000": 0.0, "s0001": 0.0}
    assert _trims_sent(setup.ledger, setup.io.executed) == {"s0000": (pytest.approx(-0.0065), pytest.approx(-0.004)),
                                                            "s0001": (pytest.approx(-0.0025), pytest.approx(-0.0025))}
    held = [(op, text) for kind, op, text in setup.node.events if kind == "pen_trim" and " held at " in text]
    assert held == [("s0001", "fitted -4.0 mm held at -2.50 mm on s0001: the needles go no deeper than "
                              "page.surface.rigid.skin_stop_m past its plan, no skin read at its own hover")]
    assert ex.trims == {"fitted": pytest.approx(-0.004)}


def test_keys_pressed_past_a_strokes_cap_change_nothing_on_the_arm_and_say_where_it_holds(setup, monkeypatch):
    """A press deeper than the stroke in hand takes is the keys' (a ledger row, the next stroke may take it) but leaves
    the running goal as it is, and its event says where the stroke holds it; a press back within the cap moves it."""
    ex, program, _, _ = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), 0.0)], mode="ride")
    _needles(setup, monkeypatch)
    ex.trims["fitted"], ex.ledger = -0.0025, setup.ledger
    setup.io.at_tick = {2: lambda: ex.on_key("down"), 4: lambda: ex.on_key("up")}
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete"
    said = [text for kind, _, text in setup.node.events if kind == "pen_trim"]
    assert said[0].startswith("fitted -2.6 mm of ±4.0, held at -2.50 mm by the skin read along the stroke: the path ")
    assert said[1].startswith("fitted -2.5 mm of ±4.0: the path ") and setup.io.replaced == []
    assert [(r["trim_m"]) for r in setup.ledger.rows() if r["event"] == "pen_trim"] == [-0.0026, -0.0025]
    ex.on_key("up")
    assert ex.trim_in_use() == pytest.approx(-0.0024)


def test_a_mid_draw_set_up_eases_a_trim_kept_past_skin_stop_m_back_to_it(setup, monkeypatch):
    """A set-up's fresh field has no skin read sunk: a trim kept deeper than skin_stop_m goes back to it, with a
    pen_trim row a resume starts from and an event; a trim within it stays."""
    ex, program, _, pages = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), 0.003), ((0.0, 0.0), -0.004)], mode="ride")
    _needles(setup, monkeypatch)
    ex.trims.update(fitted=-0.0038, other=-0.002)
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete"
    assert pages == [False] and ex.trims == {"fitted": pytest.approx(-0.0025), "other": -0.002}
    assert [(r["resource"], r["trim_m"], r["eased_from_m"]) for r in setup.ledger.rows() if r["event"] == "pen_trim"] == [
        ("fitted", -0.0025, -0.0038)]
    assert ("pen_trim", "fitted -3.8 mm eased back to -2.5 mm: the page was set up again, and a deeper trim reaches "
            "only where the hovers read the skin sunk") in [(kind, text) for kind, _, text in setup.node.events]
    assert _trims_sent(setup.ledger, setup.io.executed) == {"s0000": (pytest.approx(-0.0065), pytest.approx(-0.0025))}


@pytest.mark.parametrize("mode", ["press", "hover"])
def test_a_print_the_wrist_reads_nowhere_ends_an_inking_curved_pages_hover_stops_for_the_print_only(setup, monkeypatch,
                                                                                                  mode):
    """A print the wrist read at no hold nor hover ended every later hover stop, and with them the skin watch: the
    strokes went down on the setup's field unread. An inking pen's hovers go on for the skin alone (no print reading,
    no set-up for it), each stroke riding by it; a rehearsal's end as before."""
    line = [{"op": "stroke", "id": f"s000{k}", "closed": False, "points_m": [[0.0, 0.03 * k], [0.0, 0.03 * k + 0.01]]}
            for k in range(2)]
    ex, program, pens, pages = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), 0.0008), ((0.0, 0.0), 0.0011)],
                                              ops=line, mode=mode)
    stub, rng = ex.tracker, np.random.default_rng(0)
    stub.field.samples = []

    def at_hover(name, op_id, plan_xy):   # the skin read about the tip, the print nowhere
        stub.hovers.append((name, op_id, np.asarray(plan_xy, float)))
        points, normals = ex.surface().lift(np.asarray(plan_xy, float) + rng.uniform(-0.012, 0.012, (80, 2)))
        stub.skin = points + stub.readings.pop(0)[1] * normals
    monkeypatch.setattr(stub, "at_hover", at_hover)
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete"
    inks = mode == "press"
    assert [h[0] for h in ex.tracker.hovers] == (["s0000-hover", "s0001-hover"] if inks else ["s0000-hover"])
    assert _rides(setup.ledger) == ({"s0000": pytest.approx(0.0008), "s0001": pytest.approx(0.0011)} if inks
                                    else {"s0000": 0.0, "s0001": 0.0})
    assert pages == [] and any(("the hovers read the skin alone" if inks else "no more hover stops") in t
                               for _, _, t in setup.node.events)


def test_a_tracker_that_fails_after_a_lift_is_started_again_on_an_inking_curved_page(setup, monkeypatch):
    """A failure after a stroke's lift ended tracking for the goal, and every later hover stop with it; on an inking
    curved page the wrist camera is opened again (Tracker.start), and without it each stroke rides by the readings it
    has (_ride_over)."""
    line = [{"op": "stroke", "id": f"s000{k}", "closed": False, "points_m": [[0.0, 0.03 * k], [0.0, 0.03 * k + 0.01]]}
            for k in range(2)]
    ex, program, pens, pages = _riding_hovers(setup, monkeypatch, [((0.0, 0.0), 0.0), ((0.0, 0.0), 0.0009)], ops=line)

    def fail(stub):
        raise RuntimeError("the D405 stopped")
    ex.tracker.on_lift, first = fail, ex.tracker
    assert draw_strokes(ex, program, setup.ledger, touch=False) == "complete"
    assert ex.tracker is not first and [h[0] for h in first.hovers] == ["s0000-hover", "s0001-hover"]
    assert _rides(setup.ledger)["s0001"] == pytest.approx(0.0009)
    texts = [t for _, _, t in setup.node.events]
    assert any("the D405 stopped; not tracking" in t for t in texts) and any(t == "track: started again" for t in texts)


def test_a_rehearsal_resource_hovers_at_its_own_height_clear_of_the_needles(setup, monkeypatch):
    """ros_program.rehearsal: pen_mode hover at the resource's hover_height_m, the machine off; a hover lower than a
    needle cartridge's reach past its tube plus 1 mm is refused before anything moves."""
    ex = setup.ex
    ex.motion = {**MOTION, "pen": {**MOTION["pen"], "hover": {"height_m": 0.008, "settle_s": 0.0}}}
    monkeypatch.setattr(ex_mod.config, "tool_datasheet", lambda repo, tool: SimpleNamespace(raw={"needle_reach_mm": 4.0}))
    resource = {"id": "fitted", "tool": {"id": "lutin-ballpoint-dot"}, "pen_mode": "hover", "hover_height_m": 0.006,
                "dip": None, "slot": None}
    ex.prepare_resource(resource)
    assert ex.pen.mode == "hover" and ex.pen.height_m == pytest.approx(0.006) and not ex.pen.machine
    with pytest.raises(RuntimeError, match="needles"):
        ex.prepare_resource({**resource, "hover_height_m": 0.0045})


TILE_WORLD = geometry.rpy_matrix([0, 0, 0.67], [np.radians(218.0), 0, 0])   # the overhead's frame from _tile_seen's page


def _tile_seen(dx=0.0, turn_deg=0.0, ids=(7, 8, 22), lift=0.0):
    """geometry.page_marks of a 3 x 4 grid of 12 mm markers (16.5 mm pitch) seen 38 deg off the camera's axis at
    0.67 m, slid dx along the print, turned about its centre and lifted along its normal; the grid's printed corners."""
    printed = {i: np.array([[x, y], [x + 0.012, y], [x + 0.012, y - 0.012], [x, y - 0.012]])
               for i, (x, y) in zip((8, 22, 12, 7, 15, 23), [(-0.0225 + 0.0165 * c, 0.03075 - 0.0165 * r)
                                                             for r in range(2) for c in range(3)], strict=True)}
    c, s = np.cos(np.radians(turn_deg)), np.sin(np.radians(turn_deg))
    seen = {}
    for i in ids:
        local = np.c_[printed[i] @ np.array([[c, -s], [s, c]]).T + [dx, 0.0], np.full(4, lift), np.ones(4)]
        world = (TILE_WORLD @ local.T).T[:, :3]
        seen[i] = np.c_[world[:, :2] / world[:, 2:], np.linalg.norm(world, axis=1)]
    return seen, printed


class Ticking:
    """The executor's clock in a test: sleep advances it and hands the overhead's next reading in (`then`)."""

    def __init__(self, t, then=None):
        self.t, self.then = float(t), then

    def now(self):
        return self.t

    utc = now

    def sleep(self, seconds):
        self.t += seconds
        if self.then is not None:
            self.then(self.t)


def _read(node, t0, t1, **move):
    """The overhead reading the print's markers once a second over [t0, t1), moved as _tile_seen moves them."""
    node.marks += [(float(t), _tile_seen(**move)[0]) for t in range(t0, t1)]


def test_refresh_page_judges_the_sheet_by_its_markers_never_the_overhead_pose(setup, monkeypatch):
    """2026-10-19: a still forearm's overhead pose, its tilt from one plane over the visible part of a curved skin,
    read 5.4 mm, 13.8 deg with the arm over the print and stopped a hover draw; its markers had not moved 0.2 mm. A
    sheet slid or turned past page.replan_* (a curved page's watch total, 20 mm) or risen past WATCH_RISE_M (4 mm) is
    set up again at once; a smaller one only once it has lasted (geometry.sheet_watch)."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    ex.camera = ex.overhead = np.eye(4)
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at, ex.clock = still, 0.0, Ticking(100.0)
    pages = []
    monkeypatch.setattr(ex, "_reacquire", lambda program, ledger, why, op_id="": pages.append(why))
    node.camera = geometry.rpy_matrix([0, 0, 0.0054], [np.radians(13.7), 0, 0])   # the plane's tilt, not a move

    def at_once(**move):
        node.marks = [(float(t), _tile_seen(**move)[0]) for t in range(95, 100)]
        ex.refresh_page({"ops": [LINE]}, None)
    at_once()
    at_once(ids=(8, 15))                                     # one marker in common: no reading, as a lost page
    at_once(lift=0.002, dx=0.002)                            # 2 mm up, 2 mm along: neither
    assert pages == []
    at_once(dx=0.055)                                        # the sheet slid 55 mm
    assert "54.8 mm, -0.5 deg from its setup" in pages[0]       # its markers read on the oblique view
    at_once(turn_deg=11.0)                                   # or turned 11 deg on the skin
    assert ", +11.0 deg from its setup" in pages[1]
    at_once(lift=0.035)                                      # 35 mm, under page.replan_*: risen past WATCH_RISE_M
    assert "+35.0 mm along its normal, past 4 mm" in pages[2] and len(pages) == 3
    monkeypatch.setattr(ex, "_curved", lambda: True)          # a curved page: past its watch's total, 20 mm
    at_once(dx=0.0205)
    assert "20.4 mm, -0.2 deg from its setup" in pages[3] and len(pages) == 4 and ex.clock.t == 100.0


def test_a_move_the_markers_read_is_waited_for_lifted_and_set_up_again_only_if_it_lasts(setup, monkeypatch):
    """2026-10-20, the owner: breathing in must not set the page up again for the arm to move back. An 8 mm step read
    for 6 s waits at the stroke's top, the pen lifted and the machine off: one that stays is set up again once held
    10 s; one the person moves back is not, and the stroke goes on. Nothing waits for an operator."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    ex.camera = ex.overhead = np.eye(4)
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at = still, 0.0
    pages = []
    monkeypatch.setattr(ex, "_reacquire", lambda program, ledger, why, op_id="": pages.append(why))
    for back, ended in ((None, 109.0), (103, 108.0)):           # the person moves back at 103 s, or does not
        node.marks = []
        _read(node, 40, 94)
        _read(node, 94, 100, dx=0.008)
        ex.clock = Ticking(99.0, lambda t, back=back: _read(node, int(t), int(t) + 1, dx=0.0 if back and t >= back else 0.008))
        ex.refresh_page({"ops": [LINE]}, setup.ledger)
        waited = [text for _, _, text in node.events if "waiting up to 15 s" in text]
        assert waited and "8.0 mm off the page" in waited[-1] and setup.lifts and ex.waiting is None
        assert ex.clock.t == ended and len(pages) == 1
    assert "8.0 mm off the page, held 10 s" in pages[0]


def test_a_move_waited_for_with_the_print_hidden_clears_the_overheads_view_and_comes_back(setup, monkeypatch):
    """2026-10-20: the right forearm's strokes had no overhead reading in the 15 s before them, the lifted pen over the
    print hiding it, so a move read before it waited out its 15 s unread and set the page up again. With no reading in
    CLEAR_S the arm clears the overhead's view and waits for the markers there: a person who moved back keeps the page
    (the arm goes back to ready to draw on), one who stayed has it set up again."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    ex.camera = ex.overhead = np.eye(4)
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at = still, 0.0
    pages, views = [], []
    monkeypatch.setattr(ex, "_reacquire", lambda program, ledger, why, op_id="": pages.append(why))
    monkeypatch.setattr(ex, "clear_view", lambda ledger=None: views.append("clear"))
    monkeypatch.setattr(ex, "go_ready", lambda ledger=None: views.append("ready"))
    for stays in (False, True):
        node.marks, views[:] = [], []
        _read(node, 40, 94)
        _read(node, 94, 100, dx=0.008)                      # an 8 mm step read 6 s, then the pen hides the print
        ex.clock = Ticking(99.0, lambda t, stays=stays: views and _read(node, int(t), int(t) + 1, dx=0.008 * stays))
        ex.refresh_page({"ops": [LINE]}, setup.ledger)
        assert views == (["clear"] if stays else ["clear", "ready"]) and ex.clock.t == (109.0 if stays else 106.0)
    assert len(pages) == 1 and "8.0 mm off the page, held 10 s" in pages[0]


def test_setup_keeps_where_the_overhead_read_the_markers_and_a_resume_is_judged_by_them(setup):
    import copy

    ex, node = setup.ex, setup.node
    still, ex._printed = _tile_seen()
    node.marks = [(0.5, _tile_seen(dx=0.002)[0])] + [(time.monotonic() + 10.0, still)] * 3
    ex.overhead_at = 1.0
    ex._mark_reference(ex.overhead_at)                        # the readings since setup began, not the one before
    assert sorted(ex.page_record["overhead_marks"]) == ["22", "7", "8"]
    assert np.allclose(ex.marks[22], still[22])
    ex.stack = copy.deepcopy(STACK)
    ex.stack["page"].update(source="stencil")
    page = {"camera_at_setup": np.eye(4).tolist(), "correction": np.eye(4).tolist(), "overhead": np.eye(4).tolist(),
            "overhead_marks": copy.deepcopy(ex.page_record["overhead_marks"])}
    node.camera = geometry.rpy_matrix([0, 0, 0.004], [np.radians(14.0), 0, 0])   # the pose tilts; the markers stay
    ex.marks = None
    assert ex.restore_page(page) and ex.marks is not None
    node.marks.append((time.monotonic() + 20.0, _tile_seen(dx=0.007)[0]))
    node.marks.append((time.monotonic() + 21.0, _tile_seen(dx=0.007)[0]))
    node.marks.append((time.monotonic() + 22.0, _tile_seen(dx=0.007)[0]))
    assert not ex.restore_page(page)                          # 7 mm: past RETAIN_M (6 mm), set up again
    assert "7.0 mm" in node.events[-1][2]
    node.marks += [(time.monotonic() + 30.0 + k, _tile_seen(lift=-0.0035)[0]) for k in range(3)]
    assert not ex.restore_page(page)                          # sunk 3.5 mm, past RETAIN_RISE_M: its old field is not kept
    assert "-3.5 mm along the page's normal" in node.events[-1][2]


def test_a_hover_judges_whether_the_sheet_left_by_its_markers_never_the_overhead_pose(setup):
    """2026-10-19, hovering over a forearm print: every overhead pose after setup (53 of 53) tilted past the wrist fit's
    reach while the print's markers stayed, so a hover whose wrist read no print set the page up again. Still markers
    keep the page whatever the fresh pose's tilt; a 12 mm slide that has lasted, or no reading since the pen left the
    page, may have left; one read for 4 s is waited out (_waited), and the person moving back keeps the page."""
    import copy

    ex, node = setup.ex, setup.node
    ex.clock = Ticking(15.0)   # the marker watch's whole pending interval, without a wall-clock wait
    ex.stack = copy.deepcopy(STACK)
    ex.stack["hardware"], ex.stack["page"]["track"] = "real", {"mode": "correct"}
    ex.camera = ex.overhead = np.eye(4)
    still, ex._printed = _tile_seen()
    ex.marks, ex.inked_at = still, 1.0
    node.camera = geometry.rpy_matrix([0, 0, 0.0054], [np.radians(13.7), 0, 0])   # fresh, and tilted by the plane
    node.marks = [(2.0, still)] * 5
    assert not ex._sheet_may_have_left()
    node.marks = []
    _read(node, 2, 14, dx=0.012)
    assert ex._sheet_may_have_left()
    node.marks = []
    _read(node, 2, 10)
    _read(node, 10, 16, dx=0.012)
    ex.clock = Ticking(15.0, lambda t: _read(node, int(t), int(t) + 1))   # back at once
    assert not ex._sheet_may_have_left() and ex.clock.t == 21.0
    ex.inked_at = 30.0
    assert ex._sheet_may_have_left()                                     # nothing read since the pen left the page


def test_the_overheads_watch_follows_the_markers_drift_never_its_pose_and_the_field_carries_the_rest(setup, monkeypatch):
    """Before each stroke the page follows the sheet's drift by its markers' 20 s median, in the page's own plane, once
    it moves the drawing past 1 mm; below that the filter carries it (the overhead's marker moves, track.Tracker._feed).
    A still forearm's pose tilted 13.7 deg by its plane moves nothing, with hover stops or without; a pen at the page,
    nothing."""
    import copy

    def unfollowed():   # the markers' move the page has not followed, in the page's plane (the page in use is I)
        return (ex.sheet_slide() @ np.linalg.inv(ex.slid))[np.ix_([0, 1, 3], [0, 1, 3])]

    ex, node = setup.ex, setup.node
    ex.stack = copy.deepcopy(STACK)
    ex.stack["hardware"], ex.stack["page"]["track"] = "real", {"mode": "correct", "hover_stop": True}
    ex.camera = ex.overhead = np.eye(4)
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at = still, 0.0
    node.measured = [(1.0, geometry.rpy_matrix([0, 0, 0.0054], [np.radians(13.7), 0, 0]))] * 5
    monkeypatch.setattr(ex, "workspace", lambda: {"tool": "x"})
    monkeypatch.setattr(ex, "_follow_overhead", lambda *a, **k: pytest.fail("the overhead's pose read"))
    monkeypatch.setattr(ex, "_hover_on", lambda: True)
    program = {"ops": [LINE]}
    node.marks = [(1.0, still)] * 5
    ex._watch_sheet(program, LINE, 0.0, setup.ledger)
    assert np.allclose(ex.used(), np.eye(4)) and np.allclose(unfollowed(), np.eye(3), atol=1e-9)
    node.marks += [(30.0, _tile_seen(dx=0.0008)[0])] * 5                 # 0.8 mm: the field's, not the page's
    ex._watch_sheet(program, LINE, 0.0, setup.ledger)
    assert np.allclose(ex.used(), np.eye(4)) and np.allclose(unfollowed()[:2, 2], [0.0008, 0.0], atol=1e-4)
    node.marks += [(60.0, _tile_seen(dx=0.004)[0])] * 5
    ex._watch_sheet(program, {**LINE, "continues": True}, 0.0, setup.ledger)   # a pen at the page: nothing
    ex._watch_sheet(program, LINE, 0.004, setup.ledger)
    assert np.allclose(ex.used(), np.eye(4))
    ex._watch_sheet(program, LINE, 0.0, setup.ledger)
    assert np.allclose(ex.used(), geometry.translation([0.004, 0.0]), atol=1e-3)   # in the page's plane, no tilt
    assert np.allclose(unfollowed(), np.eye(3), atol=1e-6) and setup.ledger.rows()[-1]["event"] == "page"
    assert np.allclose(ex._followed()[:2, 2], [0.004, 0.0], atol=1e-3) and ex._sheet_watch(program) == (None, "")


def test_a_curved_follow_that_a_turned_print_would_put_under_the_skin_sets_up_when_inking(setup, monkeypatch):
    """2026-10-20 review: the overhead reads a forearm slid across itself and one turned about its own axis (the print
    going round a skin whose shape stays) alike, and a follow carries the skin's shape with the page; were it a turn,
    the strokes on one flank would go under the skin. A 4.5 mm follow across a 40 mm radius would put a turned print's
    skin 1.1 mm over strokes 12 mm out, past page.track.watch roll_m (1 mm since the owner doubled every stop): an
    inking pen has the page set up again instead (the holds measure which); a rehearsal follows and says so; along
    the arm the two agree and it follows."""
    import copy

    from tatbot_motion.surface import HeightField

    ex, node = setup.ex, setup.node
    ex.stack = copy.deepcopy(STACK)
    ex.stack["hardware"], ex.stack["page"]["track"] = "real", {"mode": "correct"}
    still, ex._printed = _tile_seen()
    monkeypatch.setattr(ex, "workspace", lambda: {"tool": "x"})
    monkeypatch.setattr(ex, "_curved", lambda: True)
    pages = []
    monkeypatch.setattr(ex, "_reacquire", lambda program, ledger, why, op_id="": pages.append(why))
    program = {"ops": [{**LINE, "points_m": [[-0.012, 0.0], [0.012, 0.0]]}]}   # flank to flank across x
    across = HeightField(2, (0.0, 0.0, 0.0, -0.03125, 0.0, 0.0), 0.05, (-0.03, -0.04, 0.03, 0.04))   # h = -x^2 / 2R
    along = HeightField(2, (0.0, 0.0, 0.0, 0.0, 0.0, -0.03125), 0.05, (-0.03, -0.04, 0.03, 0.04))    # its axis along x
    for field, mode in ((across, "press"), (across, "hover"), (along, "press")):
        ex.camera = ex.overhead = np.eye(4)
        ex.correction, ex.slid, ex.surface_field, ex._surface_chart = np.eye(4), np.eye(4), field, None
        ex.marks, ex.overhead_at, ex.motion = still, 0.0, dict(MOTION, pen=dict(MOTION["pen"], mode=mode))
        node.marks, before = [(1.0, _tile_seen(dx=0.0045)[0])] * 5, len(node.events)
        ex._watch_sheet(program, LINE, 0.0, setup.ledger)
        followed = [text for _, _, text in node.events[before:] if "the page follows it" in text]
        if field is across and mode == "press":
            assert not followed and np.allclose(ex.used(), np.eye(4))
            assert "a print turned round the arm reads alike and would stand the skin 1.1 mm over" in pages[-1]
        else:
            assert len(followed) == 1 and np.allclose(ex.used()[:2, 3], [0.0045, 0.0], atol=2e-4)
            assert ("(inking, the overhead sees the sheet +4.5" in followed[0]) == (mode == "hover")
    assert len(pages) == 1


def test_a_resume_keeps_the_slide_its_page_followed_by_the_markers(setup, monkeypatch):
    """A page that followed its markers' 2.5 mm slide is restored where it went, and its reference stays setup's
    markers: restored with no slide followed, the next watch moved it 2.5 mm again (to 5 mm, the sheet at 2.5), and a
    set-up kept it there too."""
    import copy

    ex, node = setup.ex, setup.node
    ex.stack = copy.deepcopy(STACK)
    ex.stack["hardware"], ex.stack["page"]["track"] = "real", {"mode": "correct"}
    ex.stack["page"]["source"] = "stencil"
    ex.camera = ex.overhead = np.eye(4)
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at = still, 0.0
    ex.page_record = {"camera_at_setup": np.eye(4).tolist(), "correction": np.eye(4).tolist(), "overhead": np.eye(4).tolist(),
                      "overhead_marks": {str(i): c.tolist() for i, c in still.items()}}
    monkeypatch.setattr(ex, "workspace", lambda: {"tool": "x"})
    node.marks = [(time.monotonic() + 2.0, _tile_seen(dx=0.0025)[0])] * 5
    assert ex._follow_overhead({}, LINE, setup.ledger) == "followed"
    page, followed = copy.deepcopy(ex.page_record), ex.used()
    assert ex.restore_page(page) and np.allclose(ex.used(), followed)
    assert np.allclose(ex.page_overhead(), page["overhead"]) and ex._follow_overhead({}, LINE, setup.ledger) is None
    assert np.allclose(ex._kept_page()[:3, 3], followed[:3, 3], atol=2e-4)


def test_a_mid_draw_set_up_holds_on_the_last_holds_height_and_normal_whatever_the_overheads_tilt(setup):
    """2026-10-19: one of ten mid-draw re-setups took the overhead's pose with the arm over the sheet, tilted 11.5 deg
    and raised 3.8 mm, and its holds stood that far off the skin. Set up again, the page is the one in use moved in its
    own plane as the markers put the sheet: an overhead tilted 12 deg and raised 4 mm moves neither its normal nor its
    height, a 4 mm slide along the print moves it 4 mm. A first setup takes the overhead's pose."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    held = geometry.rpy_matrix([0.3, 0.05, 0.03], [0.02, -0.03, 0.4])    # the page the last holds placed
    ex.camera, ex.correction = np.eye(4), held
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at, node.world = still, 0.0, held @ np.linalg.inv(TILE_WORLD)   # it reads that page
    node.camera = held @ geometry.rpy_matrix([0, 0, 0.004], [np.radians(12.0), 0, 0])
    node.marks = [(time.monotonic() + 1.0, _tile_seen(dx=0.004)[0])] * 3
    ex._again = True
    ex.setup_page({"ops": [LINE]}, False, setup.ledger)
    d = np.linalg.inv(held) @ ex.used()
    assert np.allclose(d[:3, 2], [0, 0, 1], atol=1e-6) and np.allclose(d[:3, 3], [0.004, 0.0, 0.0], atol=2e-4)
    assert any("the overhead's pose not read" in text for _, _, text in node.events)
    ex.setup_page({"ops": [LINE]}, False, setup.ledger)                  # a first setup: the overhead places it
    assert np.allclose(ex.used(), node.camera) and node.since == [None, None]   # neither cleared the view for it


def test_a_mid_draw_set_up_of_a_risen_or_sunk_sheet_is_placed_by_the_overhead_as_at_a_first_setup(setup):
    """A limb lifted 30 mm reads ~23 mm of slide across the overhead's rays: kept on the last holds' height its holds
    would stand 20 mm in the skin and its locate views low over it. Its corners read the rise along the page's normal,
    and past RISE_M the overhead's pose, which rose with the sheet, places the page as at a first setup. 3 mm keeps
    it: its holds stand 7 mm over the skin and are held again there."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    ex.camera, ex.correction = np.eye(4), np.eye(4)
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at = still, 0.0
    node.marks = [(time.monotonic() + 1.0, _tile_seen(lift=0.003)[0])] * 3
    kept = ex._kept_page()
    assert kept is not None and np.allclose(kept[:3, 2], [0, 0, 1]) and abs(kept[2, 3]) < 1e-9
    for lift in (0.030, -0.010):
        node.marks = [(time.monotonic() + 1.0, _tile_seen(lift=lift)[0])] * 3
        assert ex._kept_page() is None
        assert f"{lift * 1e3:+.1f} mm along the page's normal" in node.events[-1][2]
    for seen in (_tile_seen(dx=0.055), _tile_seen(turn_deg=11.0)):   # a big move is never kept, risen or not
        node.marks = [(time.monotonic() + 1.0, seen[0])] * 3
        assert ex._kept_page() is None and "page.replan_*" in node.events[-1][2]
    node.camera = geometry.translation([0.0, 0.0, 0.030])
    ex._again = True
    ex.setup_page({"ops": [LINE]}, False, setup.ledger)
    assert np.allclose(ex.used(), node.camera)


@pytest.mark.parametrize("rise", [0.006, 0.012])
def test_a_risen_limb_set_up_mid_draw_starts_from_the_overheads_risen_page_never_the_paper_touched_before(setup,
                                                                                                         monkeypatch, rise):
    """Review, 2026-10-19: a limb lifted 6-14 mm mid-draw was set up again from the overhead, but the paper this run's
    first setup touched (within page.carry_move_m of the risen pose) put the ready move and the holds' prior back on
    the old skin: a curved first hold 10 mm over it stood 2 mm in a limb lifted 12 mm. A mid-draw set-up carries no
    paper's height, and a first setup none the cameras' page stands past RISE_M from along its normal; both keep the
    page's ready pose (its IK branch and pen heading)."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    ex.camera, ex.correction = np.eye(4), np.eye(4)
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at = still, 0.0
    ready = {"q": [0.1] * 7, "heading": np.eye(4).tolist()}
    ex.carried = {"arm": ex.arm, "pattern_id": node.page_info["pattern_id"], "camera": np.eye(4).tolist(),
                  "plane": [0.0, 0.0, 1.0, 0.0], "utc_s": ex.clock.utc(), "run": "first", "ready": ready}   # skin at z 0
    node.marks = [(time.monotonic() + 1.0, _tile_seen(lift=rise)[0])] * 3
    node.camera = geometry.translation([0.0, 0.0, rise])     # the overhead's pose rose with the limb
    seen = {}
    monkeypatch.setattr(ex, "go_ready", lambda ledger=None: seen.update(ready=float(ex.used()[2, 3])))

    def touches(points_xy, ledger=None, known=False):
        seen.update(prior=float((ex.camera @ ex.prior_correction)[2, 3]), known=known)
        return []

    monkeypatch.setattr(ex, "page_touches", touches)
    ex._again = True
    ex.setup_page({"ops": [LINE]}, True, setup.ledger)
    assert seen == {"ready": pytest.approx(rise), "prior": pytest.approx(rise), "known": False}
    assert ex._page_ready() == ready
    for z, carried in ((rise, False), (0.002, True)):        # a run's first setup: the paper only while it stands
        node.camera = geometry.translation([0.0, 0.0, z])
        ex.setup_page({"ops": [LINE]}, True, setup.ledger)
        assert seen["known"] is carried and seen["prior"] == pytest.approx(0.0 if carried else z, abs=1e-9)
        assert ex._page_ready() == ready
    texts = [text for _, _, text in node.events]
    assert f"the cameras' page stands {rise * 1e3:+.1f} mm along its normal from where run first touched the paper: " \
           "its height not carried" in texts and sum("the paper as run first touched it" in text for text in texts) == 1


class _LocatedError(Exception):
    pass


def test_a_kept_pages_locate_runs_on_it_and_its_field_with_no_old_correction(setup, monkeypatch):
    """A set-up that keeps the page makes its locate views on it: their pen clearance (used(), surface()) and their
    corners (the field's chart) are the kept page's with the old field on it, never the kept page carried again by the
    last setup's correction from the overhead's pose (the holds' page 8.7 mm under it and 3 deg off on run 4271), which
    put the page the clearance is judged on 8.7 mm in the skin."""
    ex, program, skin = _curved_executor(setup, monkeypatch)
    ex._fit_surface(program)
    field = ex.surface_field
    ex.stack["page"].update(source="stencil", locate={"wrist": True})
    then = geometry.rpy_matrix([0, 0, 0.0087], [np.radians(3.0), 0, 0])   # the overhead's pose at the last setup
    ex.camera, ex.correction = then, np.linalg.inv(then)                  # the holds' page: the base
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at = still, 0.0
    setup.node.marks = [(time.monotonic() + 1.0, _tile_seen(dx=0.004)[0])] * 3
    seen = {}

    def wrist_locate(prior, cfg, ledger=None, program=None, surface=None, alone=False):
        seen.update(prior=prior, used=ex.used(), surface=surface, clear=ex.surface())
        raise _LocatedError

    monkeypatch.setattr(ex, "wrist_locate", wrist_locate)
    ex._sensing = SimpleNamespace(wrist="real", ready=lambda arm: False)
    ex._again = True
    with pytest.raises(_LocatedError):
        ex.setup_page(program, False, setup.ledger)
    assert np.allclose(seen["prior"][:3, 3], [0.004, 0.0, 0.0], atol=2e-4) and np.allclose(seen["prior"][:3, 2], [0, 0, 1])
    assert np.allclose(seen["used"], seen["prior"]) and np.allclose(seen["clear"].base, seen["prior"])
    assert np.allclose(seen["surface"].base, seen["prior"]) and seen["surface"].height is field


def test_a_curved_page_set_up_afresh_starts_on_the_overheads_page_never_the_last_skins_height(setup, monkeypatch):
    """2026-10-20: a silicone's set-up left a correction 9.1 mm under the overhead's page and 5.2 deg off it, and the
    stack's next set-up, a live forearm the overhead reads 4.7-6.1 mm over its skin, would take it as its prior: its
    ready pose and first hold 6.6-9.0 mm over the skin, the ready move's floor 1.6-4 mm (the night before, so, a first
    hold 7.6 mm over a forearm). A curved page set up afresh starts on the overhead's own page, the last correction's
    place and turn in its plane kept: the locate's clearance (no old field), the ready move and the holds are judged
    there, and no carried paper's height moves it. A flat page carries the paper as before."""
    ex, program, skin = _curved_executor(setup, monkeypatch)
    ex._fit_surface(program)                                                 # the last set-up's field
    ex.stack["page"].update(source="stencil", locate={"wrist": True})
    ex._sensing = SimpleNamespace(wrist="real", ready=lambda arm: False)
    overhead = geometry.rpy_matrix([0.32, 0.05, 0.10], [0.1, 0.0, 0.4])
    n = overhead[:3, 2]
    setup.node.camera = overhead
    ex.carried = {"arm": ex.arm, "pattern_id": setup.node.page_info["pattern_id"], "camera": overhead.tolist(),
                  "plane": [*n.tolist(), 0.008 - float(n @ overhead[:3, 3])], "utc_s": ex.clock.utc(), "run": "silicone"}
    seen = {}

    def wrist_locate(prior, cfg, ledger=None, program=None, surface=None, alone=False):
        seen.update(view=ex.used(), clear=ex.surface())
        return prior, {}

    def touches(points_xy, ledger=None, known=False):
        seen.update(prior=ex.camera @ ex.prior_correction, known=known)
        raise _LocatedError

    monkeypatch.setattr(ex, "wrist_locate", wrist_locate)
    monkeypatch.setattr(ex, "go_ready", lambda ledger=None: seen.update(ready=ex.used(), field=ex.surface()))
    monkeypatch.setattr(ex, "page_touches", touches)
    ex.correction = ex.prior_correction = geometry.rpy_matrix([0.0039, 0.0051, -0.0091], np.radians([5.2, 0.0, 2.0]))
    with pytest.raises(_LocatedError):
        ex.setup_page(program, True, setup.ledger)
    level = overhead @ geometry.rpy_matrix([0.0039, 0.0051, 0.0], [0.0, 0.0, np.radians(2.0)])
    assert all(np.allclose(seen[k], level) for k in ("view", "ready", "prior")) and seen["clear"] is seen["field"] is None
    texts = [text for _, _, text in setup.node.events]
    assert not seen["known"] and not any("the paper as run silicone" in text for text in texts)
    assert any("its holds start on the overhead's page, not -9.1 mm, 5.2 deg off it as the last's" in t for t in texts)
    monkeypatch.setattr(ex, "_curved", lambda: False)                         # a flat page: the paper's height
    with pytest.raises(_LocatedError):
        ex.setup_page(program, True, setup.ledger)
    assert seen["known"] and np.allclose(seen["prior"], overhead @ geometry.translation([0.0, 0.0, -0.008]))


@pytest.mark.parametrize("move", [{"dx": 0.055}, {"lift": 0.055}, {"turn_deg": 11.0}, {"lift": 0.012}, {"lift": -0.006}])
def test_a_big_move_mid_draw_sets_the_page_up_again_where_the_overhead_places_it_and_the_draw_goes_on(setup, monkeypatch,
                                                                                                    move):
    """Owner, 2026-10-19: "big move should set up again". A sheet the markers read past page.replan_* (slid or lifted
    55 mm, turned 11 deg), or risen or sunk past RISE_M (a limb lifted 12 mm: the hover, 10 mm over the old page, stood
    in it), is set up again first in the stroke's loop, before the watch or a hover moves the arm over
    it: the pen lifts along the page's normal, the arm clears the overhead's view, the pose the overhead then measures
    places the page as at a first setup (never kept on the last holds' height, never the pose it read with the arm over
    the sheet), and the stroke goes down on it."""
    ex, node = setup.ex, setup.node
    ex.stack, ex.workspace = dict(STACK, page=dict(STACK["page"], source="stencil")), dict
    still, ex._printed = _tile_seen()
    node.marks = [(time.monotonic() + 10.0, still)] * 3
    order, real_setup, real_clear = [], ex.setup_page, ex.clear_view
    moved = geometry.rpy_matrix([0.04, 0.0, 0.01], [0.0, 0.0, 0.2])

    def setup_page(program, touch, ledger):
        order.append("set up")
        real_setup(program, touch, ledger)
        if len(order) == 1:   # the sheet moves once the run has its page; the arm over it tilts the overhead's plane
            node.camera = moved @ geometry.rpy_matrix([0, 0, 0.004], [np.radians(11.5), 0, 0])
            node.marks = [(time.monotonic() + 20.0, _tile_seen(**move)[0])] * 5

    def clear_view(ledger=None):
        order.append("clear view")
        real_clear(ledger)
        node.camera = moved   # the arm off the sheet

    ex.setup_page = setup_page
    monkeypatch.setattr(ex, "clear_view", clear_view)
    for name in ("_watch_sheet", "_hover_stop"):
        monkeypatch.setattr(ex, name, lambda *a, _name=name, **k: order.append(_name))
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger) == "complete"
    assert order == ["set up", "set up", "clear view", "_watch_sheet", "_hover_stop"]
    assert setup.lifts == [MOTION["touch"]["lift_m"]] and node.since[0] is None and node.since[1] is not None
    assert np.allclose(ex.used(), moved)                     # the overhead's pose measured after the clear placed it
    assert [row["event"] for row in setup.ledger.rows()] == ["page", "sent", "done"]
    texts = [text for _, _, text in node.events]
    assert any(text.startswith("track: the print's markers read the sheet") and "setting the page up again" in text
               for text in texts)
    assert any("the overhead places the page, as at a first setup" in text for text in texts)


def test_the_nodes_page_wait_since_a_time_takes_only_a_pose_measured_after_it():
    """A page the overhead places mid-draw is one it measured after the arm cleared its view: the newest measured pose
    from before is taken only once the wait ends without a newer one."""
    pytest.importorskip("rclpy")
    from tatbot_session.node import SessionNode

    now = {"t": 10.0, "arrives": None}

    def sleep(seconds):
        now["t"] += seconds
        if now["arrives"] is not None and now["t"] >= now["arrives"]:
            node.page_measured_at["p"] = now["t"]

    node = SimpleNamespace(stack={"page": {"pattern_id": "p"}}, page_measured_at={"p": 9.0},
                           clock=SimpleNamespace(now=lambda: now["t"], sleep=sleep),
                           camera_page=lambda arm: (np.eye(4), {"source": "measured"}))
    wait = MethodType(SessionNode.wait_camera_page, node)
    assert wait("right", 1.0, 5.0) is not None and now["t"] == 10.0          # no `since`: the newest, at once
    assert wait("right", 1.0, 5.0, since=9.5) is not None and now["t"] > 11.0  # none after it: at the deadline
    now["arrives"] = now["t"] + 0.3
    wait("right", 5.0, 5.0, since=now["t"])
    assert node.page_measured_at["p"] == now["t"] < now["arrives"] + 0.15     # the one measured after it


def test_a_sheet_moved_while_the_machine_pause_waits_is_measured_again_before_its_stroke(riding, monkeypatch):
    """The machine's pause before a riding stroke waits for the operator, as often as not with the e-stop pressed and
    hands on the limb. Continue goes back to the top of the stroke's loop: the sheet is measured again (the move check,
    the watch, the hover) before the machine is asked again, so a sheet moved 55 mm in the pause is set up again and the
    stroke goes down on it, never on the page measured before the pause."""
    ex, node = riding.ex, riding.node
    ex.stack, ex.workspace = dict(STACK, page=dict(STACK["page"], source="stencil")), dict
    still, ex._printed = _tile_seen()
    node.marks = [(time.monotonic() + 10.0, still)] * 3
    order, real_setup = [], ex.setup_page

    def setup_page(program, touch, ledger):
        order.append("set up")
        real_setup(program, touch, ledger)

    class Paused(Switch):   # its first start is refused while the operator puts the limb down 55 mm away
        def wait(self, on, timeout_s):
            if on and not self.log:
                self.set(on)
                node.camera = geometry.rpy_matrix([0.04, 0.0, 0.01], [0.0, 0.0, 0.2])
                node.marks = [(time.monotonic() + 20.0, _tile_seen(dx=0.055)[0])] * 5
                return False
            return super().wait(on, timeout_s)

    ex.setup_page, ex.machine = setup_page, Paused()
    for name in ("_watch_sheet", "_hover_stop"):
        monkeypatch.setattr(ex, name, lambda *a, _name=name, **k: order.append(_name))
    operator = Operator(ex, rules.CONTINUE)
    assert draw_strokes(ex, {"ops": [LINE], "tool": PROGRAM_TOOL}, riding.ledger) == "complete"
    assert operator.replies() == [(True, "")]
    assert order == ["set up", "_watch_sheet", "_hover_stop"] * 2 and np.allclose(ex.used(), node.camera)
    assert [row["event"] for row in riding.ledger.rows()] == ["decision", "page", "sent", "done"]
    assert ex.machine.log == [True, False, True, False]


def test_a_lost_sheet_decides_nothing_and_a_set_up_that_cannot_measure_the_sheet_refuses(setup, monkeypatch):
    """Page motion's one stop is a sheet no camera measures. With no overhead reading sharing two markers with setup
    (the arm over the print hides them) the move decides nothing, as a lost page; a set-up whose overhead has no fresh
    pose within page.wait_s refuses, as a first setup does, the pen lifted along the page's normal first."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil", wait_s=1.0))
    ex.camera = ex.overhead = np.eye(4)
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at = still, 0.0
    node.marks = [(1.0, _tile_seen(ids=(8, 15))[0])] * 5    # one marker in common
    ex.refresh_page({"ops": [LINE]}, setup.ledger)
    assert not node.waits and setup.lifts == []
    node.marks = [(2.0, _tile_seen(dx=0.055)[0])] * 5
    node.page_info = {"source": "lost", "measured_age_s": 300.0}
    with pytest.raises(RuntimeError, match="lost for 300 s"):
        ex.refresh_page({"ops": [LINE]}, setup.ledger)
    assert setup.lifts == [MOTION["touch"]["lift_m"]] and not setup.ledger.rows()


def _blind(setup, monkeypatch, reads=True):
    """A stencil page the gauge holds placed (`in_use`), its overhead now reading no print (the page lost 300 s), the
    gauge in use with page.locate.holds; each hold (page_touches) reads the skin where the page in use has it and, with
    `reads`, the tip on the print 2 mm along the page's x from where the prior it stood on puts it. Returns (the page
    in use, the record, what the holds saw: their prior)."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil", wait_s=240.0, locate={"holds": True}))
    monkeypatch.setattr(ex, "_gauge_ready", lambda: True)
    monkeypatch.setattr(ex, "clear_view", lambda ledger=None: node.events.append(("clear", "", "")))
    in_use = geometry.rpy_matrix([0.004, -0.002, 0.003], [0.01, -0.02, 0.2])
    record = {"camera_at_setup": in_use.tolist(), "correction": np.eye(4).tolist(), "overhead": np.eye(4).tolist(),
              "pattern_id": "stencil-x", "holds": {"used": True},
              "ready": {"q": [0.1] * 7, "heading": np.eye(4).tolist()}}
    ex.camera, ex.correction, ex.page_record = in_use, np.eye(4), dict(record)
    ex.marks, ex._printed = _tile_seen()
    node.page_info, node.marks = {"source": "lost", "pattern_id": "stencil-x", "measured_age_s": 300.0}, []
    seen = {}

    def touches(points_xy, ledger=None, known=False):
        seen["prior"] = prior = ex.camera @ ex.prior_correction
        points = [(in_use @ [*xy, 0.0, 1.0])[:3] for xy in points_xy]
        for i, (xy, point) in enumerate(zip(points_xy, points, strict=True)):
            at = (np.linalg.inv(prior) @ [*point, 1.0])[:2]
            ex.page_record["touches"].append({"xy": list(xy), "contact": point.tolist(), "method": "wrist gauge"})
            tip = {"tip_m": [at[0] + 0.002, at[1], 0.0], "markers": 4} if reads else None
            ex.page_record.setdefault("gauge", []).append({"touch": i, "xy": list(xy), "point": point.tolist(),
                                                           "print": tip})
        return points

    monkeypatch.setattr(ex, "page_touches", touches)
    monkeypatch.setattr(ex, "_touch_layout", lambda program: np.array(
        [[-0.01, -0.01], [0.01, -0.01], [0.01, 0.01], [-0.01, 0.01], [0.0, 0.0]]))
    return in_use, record, seen


@pytest.mark.parametrize("reads", [True, False])
def test_a_set_up_the_overhead_cannot_place_starts_from_the_last_page_and_the_holds_place_it(setup, monkeypatch, reads):
    """2026-10-20, a forearm tattoo: a mid-draw set-up waited for the overhead, which no longer read the inked print,
    and was cancelled; the wrist read 4-7 of its markers at every hold. With the gauge's holds to place it, a set-up
    whose overhead measures no page after 10 s from rest starts from the page in use raised 8 mm along its normal (its
    first hold and views as high over a limb risen since as on the overhead's page): the holds move it onto the print,
    its watch's reference waits for the overhead, and nothing is carried for the next run. Its ready pose is a new
    page's, never the last page's (the dips' heading first; the forearm's ran at -112 deg, 2026-10-20 review). Holds
    that read nothing place no page: not drawn."""
    ex, node = setup.ex, setup.node
    in_use, record, seen = _blind(setup, monkeypatch, reads)
    searched, dips, asked = geometry.rpy_matrix([0.3, 0.0, 0.1], [np.pi, 0.0, -0.8]), np.diag([1.0, -1.0, -1.0]), []
    monkeypatch.setattr(ex_mod.ready, "pen_heading", lambda *a, prefer=None: asked.append(prefer) or searched)
    ex._again = True
    if not reads:
        with pytest.raises(RuntimeError, match=r"the overhead reads no print and the gauge holds did not place the "
                           r"page on it \(0 of 5 holds read it, 0 by two markers or more; three must agree\): not drawn; "
                           r"give the cameras a print they can read"):
            ex.setup_page({"ops": [LINE]}, True, setup.ledger)
        return
    ex.setup_page({"ops": [LINE]}, True, setup.ledger)
    assert node.waits == [10.0, 10.0] and node.since[0] is not None              # from rest, 20 s, never 240
    assert np.allclose(seen["prior"], in_use @ geometry.translation([0.0, 0.0, ex.BLIND_LIFT_M]))
    assert np.allclose(np.linalg.inv(in_use) @ ex.used(), geometry.translation([-0.002, 0.0, 0.0]), atol=1e-9)
    assert ex.page_record["source"] == "wrist" and "ready" not in ex.page_record
    monkeypatch.setattr(ex, "_dip_heading", lambda: dips)
    assert np.allclose(ex.pen_heading(), searched) and asked == [dips]           # searched, the dips' heading first
    assert ex.marks == {} and ex.page_record["overhead_marks"] == {} and ex.carried is None
    assert np.allclose(ex.overhead, np.eye(4))                                  # the watch's last reference
    texts = [text for _, _, text in node.events]
    assert any("the overhead reads no print, so the wrist and the gauge holds place the page" in t for t in texts)


def test_a_resume_the_overhead_cannot_place_starts_from_its_retained_page_without_page_wait_s(setup, monkeypatch):
    """A blind resume waited page.wait_s twice (restore_page, then the set-up), about 8 min, then refused. With the
    gauge's holds it waits 10 s from where the arm stopped and 10 s from rest in each, and the holds place the page
    from the retained one, raised 15 mm (its limb may have been put back since); without them the set-up refuses as
    before. The next goal starts from no retained page until its own draw finds one."""
    ex, node = setup.ex, setup.node
    in_use, record, seen = _blind(setup, monkeypatch)
    ex.camera, ex.page_record, ex._resumed = None, {}, record
    assert not ex.restore_page(record)
    ex.setup_page({"ops": [LINE]}, True, setup.ledger)
    assert node.waits == [10.0, 10.0, 10.0, 10.0]
    assert np.allclose(seen["prior"], in_use @ geometry.translation([0.0, 0.0, ex.BLIND_RESUME_LIFT_M]))
    assert np.allclose(np.linalg.inv(in_use) @ ex.used(), geometry.translation([-0.002, 0.0, 0.0]), atol=1e-9)
    assert ex.begin_goal() and ex._resumed is None                              # a later goal's own, never this one's
    ex.end_goal()
    monkeypatch.setattr(ex, "_gauge_ready", lambda: False)
    node.waits = []
    with pytest.raises(RuntimeError, match="lost for 300 s"):
        ex.setup_page({"ops": [LINE]}, True, setup.ledger)
    assert node.waits == [10.0, 230.0]


@pytest.mark.parametrize("depths, reads, raised", [((-0.010, None, -0.012), True, 0.0), ((0.009, -0.004), True, 0.0063),
                                                    ((0.030,), False, 0.0273)])
def test_a_curved_page_placed_blind_rises_with_the_skin_its_views_read_and_its_holds_carry_the_tips_offset(
        setup, monkeypatch, depths, reads, raised):
    """Review, 2026-10-20: a blind resume started from its retained page, of any age, raised 8 mm, and nothing measured
    the skin before the ready move and the first hold; and its holds, made on the wrist's own page of the print, would
    shift 7.2 and 8.6 mm on the forearm tattoo's set-ups (the gauge tip's offset across the pen axis, which a set-up
    placed afresh carries in its correction). A goal's first blind set-up starts 15 mm over its page; a curved one is
    raised, never lowered, to stand 8 mm over the highest skin the locate views' depth planes read (less the mount's
    page.height.wrist_depth_bias_m); with the print read, its holds stand off the wrist's page as the last page's holds
    stood off the wrist's then. No print read: the page it started on, which holds the offset already."""
    ex, program, _ = _curved_executor(setup, monkeypatch)
    ex.stack["page"].update(source="stencil", wait_s=240.0, locate={"wrist": True, "holds": True},
                            height={"wrist_depth_bias_m": 0.0107})
    ex._sensing = SimpleNamespace(wrist="real", ready=lambda arm: True)
    monkeypatch.setattr(ex, "_gauge_ready", lambda: True)
    monkeypatch.setattr(ex, "clear_view", lambda ledger=None: None)
    overhead = geometry.rpy_matrix([0.32, 0.05, 0.07], [0.1, 0.0, 0.4])
    wrist = {"tx_m": 0.003, "ty_m": -0.001, "theta_rad": 0.01, "sigma": [0.001, 0.001, 0.005], "poses": 2}
    own = overhead @ np.linalg.inv(geometry.rpy_matrix([0.003, -0.001, 0.0], [0.0, 0.0, 0.01]))
    off = geometry.rpy_matrix([0.0064, -0.002, 0.0], [0.0, 0.0, np.radians(0.84)])   # ba9c's first set-up's
    held = own @ off @ geometry.rpy_matrix([0.0, 0.0, -0.0075], [0.02, -0.01, 0.0])  # its holds' page, under and tilted
    ex._resumed = {"camera_at_setup": overhead.tolist(), "correction": (np.linalg.inv(overhead) @ held).tolist(),
                   "locate": {"overhead": overhead.tolist(), "wrist": wrist}, "holds": {"used": True}}
    ex.camera, setup.node.page_info = None, {"source": "lost", "pattern_id": "stencil-x", "measured_age_s": 300.0}
    now, seen = geometry.rpy_matrix([0.004, 0.002, 0.0], [0.0, 0.0, -0.02]), {}

    def wrist_locate(prior, cfg, ledger=None, program=None, surface=None, alone=False):
        seen.update(start=prior, alone=alone)
        record = {"views": [{"depth": None if d is None else {"offset_m": d}} for d in depths]}
        return (prior @ np.linalg.inv(now), {**record, "wrist": wrist, "fused": wrist}) if reads else (prior, record)

    def touches(points_xy, ledger=None, known=False):
        seen["prior"] = ex.camera @ ex.prior_correction
        raise _LocatedError

    monkeypatch.setattr(ex, "wrist_locate", wrist_locate)
    monkeypatch.setattr(ex, "page_touches", touches)
    with pytest.raises(_LocatedError):
        ex.setup_page(program, True, setup.ledger)
    start = held @ geometry.translation([0.0, 0.0, ex.BLIND_RESUME_LIFT_M])
    assert seen["alone"] and np.allclose(seen["start"], start) and np.allclose(ex_mod._held_off(ex._resumed)[0], off)
    placed = (start @ np.linalg.inv(now) if reads else start) @ geometry.translation([0.0, 0.0, raised])
    assert np.allclose(seen["prior"], placed @ (off if reads else np.eye(4)), atol=1e-9)
    assert ex.page_record.get("locate", {}).get("raised_m", 0.0) == pytest.approx(raised)
    texts = [text for _, _, text in setup.node.events]
    assert any("the locate views' depth reads the skin" in t for t in texts) == bool(raised)


def test_on_a_page_the_wrist_placed_blind_an_unread_hover_goes_on_and_three_in_a_row_set_it_up_again(setup, monkeypatch):
    """A page the wrist placed whose print the overhead reads nowhere (marks {}): each hover that read no print cleared
    the view, waited MARKS_S for readings that could not come and set the page up again (~4 min a stroke on the
    inked forearm). UNREAD_SETUP - 1 in a row go on, the page in use kept; the UNREAD_SETUP'th sets it up again."""
    ex, node = setup.ex, setup.node
    ex.tracker = StubTracker([0.0, 0.0])
    _hovering(ex)
    ex.stack["hardware"], ex.stack["page"]["track"]["mode"] = "real", "correct"
    ex.tracker.verdicts = [None] * 6
    ex.MARKS_S = 0
    pages = []
    monkeypatch.setattr(ex, "setup_page", lambda program, touch, ledger=None: (
        setattr(ex, "camera", np.eye(4)), setattr(ex, "correction", np.eye(4)), setattr(ex, "marks", {})))
    monkeypatch.setattr(ex, "clear_view", lambda ledger=None: node.events.append(("clear", "", "")))
    monkeypatch.setattr(ex, "_reacquire", lambda program, ledger, why, op_id="": pages.append((op_id, why)))
    ops = [{**LINE, "id": f"s000{k}", "points_m": [[0.0, 0.004 * k], [0.02, 0.004 * k]]} for k in range(3)]
    assert draw_strokes(ex, {"ops": ops}, setup.ledger, touch=False) == "complete"
    assert [h[0] for h in ex.tracker.hovers] == ["s0000-hover", "s0000-hover2", "s0001-hover", "s0001-hover2",
                                                 "s0002-hover", "s0002-hover2", "s0002-rehover"]
    assert len(pages) == 1 and "no print under the hover" in pages[0][1] and ("clear", "", "") in node.events
    texts = [t for _, _, t in node.events]
    assert sum("in a row), and the overhead reads none: the stroke goes on" in t for t in texts) == 2


def _heading(page, yaw):
    """A tool-down rotation (4 x 4) whose x axis turns `yaw` from page's x in its plane."""
    out, x, z = np.eye(4), page[:3, :3] @ [np.cos(yaw), np.sin(yaw), 0.0], -page[:3, 2]
    out[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    assert ex_mod.ready.heading_yaw(page, out) == pytest.approx(yaw)
    return out


@pytest.mark.parametrize("blind", [True, False])
def test_the_holds_offset_a_set_up_carries_turns_with_the_pens_heading(setup, monkeypatch, blind):
    """The holds' offset carried from the last page (_held_off on a blind set-up, _level_prior on one the overhead
    places) is the gauge tip's ~7 mm across the pen axis: five runs' pages read it 85-103 deg from their ready heading
    at headings -15 to -106 deg (2026-10-20). Carried onto holds a new page makes at the dips' heading (-45 against the
    tattoo's -105) unturned, it stood them ~8.7 mm off; turned about the page's normal with the heading, they stand
    where the last page's stood."""
    ex, program, _ = _curved_executor(setup, monkeypatch)
    ex.stack["page"].update(source="stencil", wait_s=240.0, locate={"wrist": True, "holds": True}, height={})
    ex._sensing = SimpleNamespace(wrist="real", ready=lambda arm: True)
    monkeypatch.setattr(ex, "_gauge_ready", lambda: True)
    monkeypatch.setattr(ex, "clear_view", lambda ledger=None: None)
    overhead = geometry.rpy_matrix([0.32, 0.05, 0.07], [0.1, 0.0, 0.4])
    wrist = {"tx_m": 0.003, "ty_m": -0.001, "theta_rad": 0.01, "sigma": [0.001, 0.001, 0.005], "poses": 2}
    own = overhead @ np.linalg.inv(geometry.rpy_matrix([0.003, -0.001, 0.0], [0.0, 0.0, 0.01]))
    off = geometry.rpy_matrix([0.0064, -0.002, 0.0], [0.0, 0.0, np.radians(0.84)])
    record = {"camera_at_setup": own.tolist(), "correction": off.tolist(), "holds": {"used": True},
              "locate": {"overhead": overhead.tolist(), "wrist": wrist}, "ready": {"q": [0.0] * 7,
                                                                                    "heading": _heading(own, np.radians(-105.0)).tolist()}}
    assert ex_mod._held_off(record)[1] == pytest.approx(np.radians(-105.0))
    seen, turn = {}, np.radians(60.0)
    if blind:   # the overhead reads no print: the wrist places the page from the last one
        ex._resumed, setup.node.page_info = record, {"source": "lost", "pattern_id": "stencil-x", "measured_age_s": 300.0}
        ex.camera = None
    else:       # the overhead places it, the last page's correction carried in its plane
        ex.page_record, ex.prior_correction = dict(record), off.copy()
        setup.node.page_info, setup.node.camera = {"source": "stencil", "pattern_id": "stencil-x"}, own
        ex.camera = None

    def wrist_locate(prior, cfg, ledger=None, program=None, surface=None, alone=False):
        return prior, {"views": [], "wrist": wrist, "fused": wrist}

    def go_ready(ledger=None):   # the new page's heading, 60 deg on from the last's
        ex.page_record["ready"] = {"q": [0.0] * 7, "heading": _heading(ex.camera, np.radians(-45.0)).tolist()}

    def touches(points_xy, ledger=None, known=False):
        seen["prior"] = ex.camera @ ex.prior_correction
        raise _LocatedError

    monkeypatch.setattr(ex, "wrist_locate", wrist_locate)
    monkeypatch.setattr(ex, "go_ready", go_ready)
    monkeypatch.setattr(ex, "page_touches", touches)
    with pytest.raises(_LocatedError):
        ex.setup_page(program, True, setup.ledger)
    held = np.linalg.inv(ex.camera) @ seen["prior"]
    c, s_ = np.cos(turn), np.sin(turn)
    assert np.allclose(held[:2, 3], np.array([[c, -s_], [s_, c]]) @ off[:2, 3], atol=1e-9)
    assert abs(ex_mod._yaw(held) - ex_mod._yaw(off)) < 1e-9
    assert any("turned +60 deg with the pen's heading" in t for _, _, t in setup.node.events)


def test_a_page_the_wrist_placed_takes_its_markers_once_the_overhead_reads_them_again(setup, monkeypatch):
    """A page placed with the overhead blind keeps no marker reference (its readings from before are another sheet's
    place): no move is judged until three readings come within follow_s, then the reference is the sheet as they read
    it, the page in use standing for it, in a page row for a resume."""
    ex, node = setup.ex, setup.node
    _blind(setup, monkeypatch)
    ex._again, ex.workspace = True, dict
    ex.setup_page({"ops": [LINE]}, True, setup.ledger)
    pages = []
    monkeypatch.setattr(ex, "_reacquire", lambda program, ledger, why, op_id="": pages.append(why))
    ex.clock = Ticking(ex.marked_at + 40.0)
    _read(node, int(ex.marked_at) + 30, int(ex.marked_at) + 32, dx=0.009)    # two: no reference yet
    ex.refresh_page({"ops": [LINE]}, setup.ledger)
    assert ex.marks == {} and pages == [] and not setup.ledger.rows()
    _read(node, int(ex.marked_at) + 32, int(ex.marked_at) + 33, dx=0.009)
    ex.refresh_page({"ops": [LINE]}, setup.ledger)
    assert np.allclose(ex.marks[22], _tile_seen(dx=0.009)[0][22]) and pages == []
    assert [row["event"] for row in setup.ledger.rows()] == ["page"] and ex.page_record["overhead_marks"]
    assert ex.sheet_moved() is None                                              # judged from here on
    _read(node, int(ex.clock.t) + 1, int(ex.clock.t) + 4, dx=0.009)
    assert ex.sheet_moved()[0] < 1e-6


def test_the_overheads_readings_off_the_page_neither_enter_its_reference_nor_move_it(setup, monkeypatch):
    """2026-10-20, a silicone run: the overhead read a second copy of the print 0.5-0.64 m off the sheet seven times;
    one read markers the sheet's readings had not, which went into the watch's reference as the copy's (every later
    reading then put the sheet 154 mm off), and three of the newest five read a 128 mm rise. A reading with a marker
    past the print's reach of the page in use's centre (its farthest printed corner and MARKS_REACH_M, 0.2 m) is no
    reading of the sheet: the reference is the sheet's and nothing fires; read as before, both would. A 55 mm and a
    150 mm slide still read (review: past 50 mm, a limb's 59-69 mm slide went unread). Before a page is known every
    reading counts."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    ex.camera = ex.overhead = np.eye(4)
    every = (7, 8, 22, 12, 15, 23)
    (still, ex._printed), sheet = _tile_seen(), _tile_seen(ids=every)[0]
    other, risen = _tile_seen(dx=0.5, ids=(12, 15, 23))[0], _tile_seen(dx=0.45, lift=0.25)[0]
    pages = []
    for read_all in (False, True):
        if read_all:                                                              # as before the filter
            monkeypatch.setattr(ex, "_marks", node.measured_marks)
        monkeypatch.setattr(ex, "_reacquire", lambda program, ledger, why, op_id="": pages.append(why))
        node.marks = [(1.0, still), (2.0, other), (3.0, still)]
        ex._mark_reference(0.0)
        assert sorted(ex.marks) == sorted((7, 8, 22, 12, 15, 23) if read_all else (7, 8, 22))
        node.marks += [(float(t), sheet) for t in range(40, 95)] + [(95.0, risen), (96.0, sheet), (97.0, risen),
                                                                     (98.0, sheet), (99.0, risen)]
        ex.overhead_at = ex.marked_at = 4.0
        ex.clock = Ticking(100.0)
        ex.refresh_page({"ops": [LINE]}, setup.ledger)
        assert len(pages) == read_all
        node.marks = node.marks[:-5]
        ex.refresh_page({"ops": [LINE]}, setup.ledger)
        assert len(pages) == 2 * read_all and (not read_all or "mm, " in pages[-1])
    assert "along its normal, past 4 mm" in pages[0]                      # WATCH_RISE_M
    monkeypatch.undo()
    monkeypatch.setattr(ex, "_reacquire", lambda program, ledger, why, op_id="": pages.append(why))
    ex._mark_reference(0.0)
    for dx, n, said in ((0.055, 3, "54.8 mm"), (0.15, 4, "149.5 mm")):   # a limb's slide reads, past 50 mm too
        ex.marked_at, node.marks = 4.0, [(float(t), _tile_seen(dx=dx)[0]) for t in range(95, 100)]
        ex.refresh_page({"ops": [LINE]}, setup.ledger)
        assert len(pages) == n and said in pages[-1]
    ex.camera = None
    assert len(ex._marks(0.0)) == 5


def test_the_marker_reference_is_the_sheet_after_the_setups_last_holds(setup, monkeypatch):
    """The overhead's markers are kept once every hold has run (a curved page's check holds in _fit_surface last): the
    sheet as the holds placed the page on it, not as the setup began."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    still, ex._printed = _tile_seen()
    node.marks = [(time.monotonic() + 1.0, _tile_seen(dx=0.004)[0])]  # as the setup began
    monkeypatch.setattr(ex, "_fit_surface", lambda program, ledger=None: node.marks.extend(
        [(time.monotonic() + 2.0, still)] * 3))                          # read through the last holds
    ex.setup_page({"ops": [LINE]}, False, setup.ledger)
    assert np.allclose(ex.marks[22], still[22]) and ex.page_record["overhead_marks"]["22"] == still[22].tolist()


def test_the_marker_reference_takes_the_markers_the_arm_hid_from_the_setups_still_readings(setup):
    """A forearm (2026-10-24): the newest 20 s of its setup's readings, the arm over the print at its holds, saw
    markers 3 and 6 alone, the earlier ones 3, 6 and 16; after a 45 mm move the overhead read 6 and 16, or 9 and 16,
    and one reading in ten shared two markers with the reference. An earlier reading that reads the sheet where the
    newest do puts its other markers in it (and one it then shares two with, in turn); one 2 mm off does not."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    ex.camera = ex.overhead = np.eye(4)
    _, ex._printed = _tile_seen()
    node.marks = [(1.0, _tile_seen(ids=(7, 8, 22))[0]), (2.0, _tile_seen(ids=(8, 22, 12))[0]),
                  (3.0, _tile_seen(dx=0.002, ids=(8, 22, 15))[0])] + [(float(t), _tile_seen(ids=(7, 8))[0])
                                                                     for t in range(30, 40)]
    ex._mark_reference(0.0)
    assert sorted(ex.marks) == [7, 8, 12, 22] and sorted(ex.page_record["overhead_marks"]) == ["12", "22", "7", "8"]
    assert np.allclose(ex.marks[22], _tile_seen()[0][22]) and np.allclose(ex.marks[12], _tile_seen(ids=(12,))[0][12])
    node.marks += [(float(t), _tile_seen(dx=0.045, ids=(8, 22))[0]) for t in range(50, 53)]
    assert [round(m[0], 3) for _, m in ex._timed_moves(45.0)] == [0.045] * 3
    assert ex._timed_moves(45.0, marks={i: ex.marks[i] for i in (7, 8)}) == []   # as the reference was kept before


def test_a_long_hover_has_the_watch_judge_the_sheet_again_before_the_stroke_goes_down(setup):
    """A forearm moved 45 mm under the pen (2026-10-24): its hover cleared the overhead's view and waited 41 s, the
    readings that came in meanwhile read the move, and the stroke went down on the judgement from before the hover.
    After a hover stop of HOVER_WATCH_S the watch judges again (a move fired or not yet held, a rise past
    WATCH_RISE_M); a still sheet, or a short hover (refresh_page judges it at the next stroke), goes on."""
    ex, node = setup.ex, setup.node
    ex.stack = dict(STACK, page=dict(STACK["page"], source="stencil"))
    ex.camera = ex.overhead = np.eye(4)
    still, ex._printed = _tile_seen()
    ex.marks, ex.overhead_at, ex.marked_at, program = still, 0.0, 0.0, {"ops": [LINE]}
    _read(node, 20, 30)
    ex.clock = Ticking(40.0)
    assert not ex._moved_meanwhile(program, 20.0, LINE["id"])
    _read(node, 50, 57, dx=0.045)                       # the pen left the overhead's view at 50
    ex.clock = Ticking(57.0)
    assert not ex._moved_meanwhile(program, 50.0, LINE["id"])
    assert ex._moved_meanwhile(program, 45.0, LINE["id"])
    assert "judged again before the stroke goes down" in node.events[-1][2]


def test_a_stroke_whose_hover_saw_the_sheet_move_goes_round_its_checks_again_once(setup, monkeypatch):
    ex, calls = setup.ex, []
    monkeypatch.setattr(ex, "refresh_page", lambda program, ledger, op_id="": calls.append("refresh"))
    monkeypatch.setattr(ex, "_hover_stop", lambda program, op, from_arc, ledger, index=0: calls.append("hover"))
    monkeypatch.setattr(ex, "_moved_meanwhile", lambda program, began, op_id: calls.append("judged") or True)
    assert draw_strokes(ex, {"ops": [LINE]}, setup.ledger, touch=False) == "complete"
    assert calls == ["refresh", "hover", "judged", "refresh", "hover"]
    assert [r["event"] for r in setup.ledger.rows()] == ["sent", "done"]


def test_a_resume_sets_a_curved_page_kept_without_markers_up_again(setup, monkeypatch):
    """A curved page saved before the overhead's markers were kept has only the overhead's pose to judge it by, whose
    tilt is a plane's over the skin: it is set up again, without waiting on the overhead. One with its markers is kept
    where they still read."""
    ex, program, skin = _curved_executor(setup, monkeypatch)
    ex.stack["page"]["source"] = "stencil"
    page = {"camera_at_setup": np.eye(4).tolist(), "correction": np.eye(4).tolist(), "overhead": np.eye(4).tolist()}
    monkeypatch.setattr(ex, "clear_view", lambda ledger=None: pytest.fail("no view to clear"))
    assert not ex.restore_page(page)
    assert "a curved page kept without the overhead's markers" in setup.node.events[-1][2]
    still, ex._printed = _tile_seen()
    setup.node.marks = [(time.monotonic() + 1.0, still)] * 3
    assert ex.restore_page({**page, "overhead_marks": {str(i): c.tolist() for i, c in still.items()}})


def test_clearing_the_cameras_view_lifts_under_the_snag_guard_then_swings_to_rest_without_it(riding, monkeypatch):
    """At rest the joints lie ~1 deg off REST against their stops, the tip 4.4 mm off its command: the guard, armed
    through the swing, latched there and the run ended (silicone practice, 2026-10-21)."""
    ex, io = riding.ex, riding.io
    ex._snag, ex.camera, order = ex_mod.names.GUARD_SNAG, None, []
    io.q = np.array([0.3, 0.8, 0.6, 0.0, 0.4, 0.0, 0.0])
    monkeypatch.setattr(ex, "lift_off", lambda what, ledger=None: order.append(("lift", ex._snag)))
    monkeypatch.setattr(ex, "move", lambda make, phase, what, ledger=None: order.append(("move", ex._snag)))
    ex.clear_view()
    assert order == [("lift", ex_mod.names.GUARD_SNAG), ("move", ex_mod.names.GUARD_NONE)]
    assert io.writes[-1] == {"guard_mode": ex_mod.names.GUARD_NONE}


def test_a_snag_trip_at_rest_has_no_pen_to_withdraw(riding, monkeypatch):
    ex, io = riding.ex, riding.io
    io.q = np.array([0.0, 0.0166, 0.0128, 0.0, 0.0, 1.552, 0.0])   # where the 2026-10-21 trip left it
    monkeypatch.setattr(ex, "lift_plan", lambda *a, **k: pytest.fail("no withdraw from rest"))
    assert ex._withdraw("", False) and io.unlatches == 1


def test_a_lift_the_arm_cannot_follow_along_the_normal_goes_along_the_pen(riding, monkeypatch):
    from tatbot_motion.clik import PlanError

    ex, asked = riding.ex, []

    def plan(distance, direction=None):
        asked.append((distance, direction is None))
        if direction is None:
            raise PlanError("the tip trails its reference by 19.43 mm at t=4.01 s")
        return "along the pen"

    monkeypatch.setattr(ex, "lift_plan", plan)
    monkeypatch.setattr(ex, "move", lambda make, phase, what, ledger=None: asked.append(make()))
    ex.lift("s0114 lift before the dip")
    assert asked == [(MOTION["touch"]["lift_m"], True), (MOTION["touch"]["lift_m"], False), "along the pen"]


def _lifts_planned_up_to(ex, monkeypatch, longest_m):
    """ex.lift_plan as the planner gives it, for lifts no longer than longest_m: a longer one is a PlanError (those
    before landing on 2026-10-21 trailed their reference 19 and 31 mm). Returns the (length, along the page's normal)
    asked, in order."""
    from tatbot_motion.clik import PlanError

    asked, real = [], ex.lift_plan

    def plan(distance, direction=None):
        asked.append((round(distance, 4), direction is None))
        if distance > longest_m + 1e-9:
            raise PlanError("the tip trails its reference by 31.24 mm at t=4.40 s")
        return real(distance, direction)

    monkeypatch.setattr(ex, "lift_plan", plan)
    return asked


def test_a_landing_lift_that_plans_no_way_rises_in_steps_along_the_pen_before_the_sweep(setup, monkeypatch):
    """The lift before landing tries the skin's normal and the pen at its length and at half, then goes up in 5 mm
    steps along the pen, each from the joints the arm measures, until the tip clears; only then the driver lands.
    On 2026-10-21 one lift failed and the landing swept from the skin."""
    ex, io, node = setup.ex, setup.io, setup.node
    _pen_down(monkeypatch)
    ex.camera = np.eye(4)                                                  # the tip on the page
    asked = _lifts_planned_up_to(ex, monkeypatch, ex.LIFT_STEP_M)
    assert ex._land_now() == (True, "landed and idle") and io.lands == 1
    assert asked == [(0.04, True), (0.04, False), (0.02, True), (0.02, False)] + [(0.005, False)] * 8
    assert io.q[2] == pytest.approx(ex.LAND_CLEAR_M) and len(io.executed) == 8
    assert [text for kind, _, text in node.events if kind == "error"] == [
        "lift before landing: the tip trails its reference by 31.24 mm at t=4.40 s"]


def test_a_landing_no_lift_plans_for_holds_the_needles_off_the_sweep_until_a_person_decides(riding, monkeypatch):
    """No lift at all from the skin: the landing's wrist turn would sweep the needles over it, so the arm holds there
    with the machine off and the land says why, as an event too. Every automatic land holds again there (on a5bf49f8c
    `tatbot ros draw`'s own land after the draw was the second land and swept within seconds): a draw's, a Land
    action's. A person's first decide holds too and says why (on 538ada44c it swept from the wrapper's hold, which no
    person had read); their decide at a hold said to them lands from it, so the arm is never stuck."""
    ex, io, node = riding.ex, riding.io, riding.node
    _pen_down(monkeypatch)
    ex.camera, io.q = np.eye(4), np.array([0.01, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0])
    ex.machine.set(True)
    _lifts_planned_up_to(ex, monkeypatch, 0.0)
    with pytest.raises(ex_mod.Landed, match="not landing: no lift off the skin plans from here"):
        ex.land()                                                        # a draw's: a wait's timeout, a tool change
    assert io.lands == 0 and io.executed == [] and not ex.machine.on
    ok, why = ex.decide(rules.LAND)                                      # the Land action `tatbot ros draw` sends after
    assert not ok and why.startswith("not landing: no lift off the skin plans from here (the tip trails")
    assert "decide right land" in why and io.lands == 0 and node.events[-1] == ("error", "", why)
    assert not ex._land_now()[0] and io.lands == 0                       # ros down, a set-up's: still held
    ok, why = ex.decide(rules.LAND, asked=True)                          # the station's stop: said to the person
    assert not ok and why.endswith("decide right land` again lands from here") and io.lands == 0
    assert not ex.decide(rules.LAND)[0] and io.lands == 0                # an automatic land between: still said
    assert ex.decide(rules.LAND, asked=True) == (True, "landed and idle") and io.lands == 1
    assert [text for kind, _, text in node.events if "as a person decided" in text] == [
        "landing from the hold as a person decided: no lift planned off the skin"]


def test_the_land_action_never_lands_from_a_hold_and_the_decide_service_does(riding, monkeypatch):
    """`tatbot ros draw` ends with the Land action, as `ros down` and a set-up do: an automatic land. A person lands
    through the Decide service (`tatbot ros decide right land`, the station's land after a tap on its screen)."""
    from tatbot_interfaces.srv import Decide
    from tatbot_session.node import SessionNode

    ex, io = riding.ex, riding.io
    _pen_down(monkeypatch)
    ex.camera, io.q = np.eye(4), np.array([0.01, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0])
    _lifts_planned_up_to(ex, monkeypatch, 0.0)
    gate = GoalGate(ex)
    gate.stack["safety"] = STACK["safety"]
    handle = SimpleNamespace(request=SimpleNamespace(arms=["right"]), publish_feedback=lambda feedback: None,
                             succeed=lambda: None)
    for _ in range(2):
        assert list(MethodType(SessionNode._land, gate)(handle).landed) == [False] and io.lands == 0
    for accepted in (False, True):   # a person's first is said the hold; their next lands
        response = MethodType(SessionNode._decide, gate)(Decide.Request(arm="right", decision=Decide.Request.LAND),
                                                         Decide.Response())
        assert response.accepted == accepted and io.lands == int(accepted)


def test_a_person_s_first_land_holds_and_a_hold_the_arm_moved_from_is_said_again(riding, monkeypatch):
    """A person's decide lands from a hold only once it was said there: their first holds and says why, and an arm
    moved since (or a goal run since) holds and says it again."""
    ex, io = riding.ex, riding.io
    _pen_down(monkeypatch)
    ex.camera, io.q = np.eye(4), np.array([0.01, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0])
    _lifts_planned_up_to(ex, monkeypatch, 0.0)
    assert not ex.decide(rules.LAND, asked=True)[0] and io.lands == 0
    io.q[3] += 0.02
    assert not ex.decide(rules.LAND, asked=True)[0] and io.lands == 0
    assert ex.begin_goal()
    ex.end_goal()
    assert not ex.decide(rules.LAND, asked=True)[0] and io.lands == 0
    assert ex.decide(rules.LAND, asked=True)[0] and io.lands == 1


def test_a_landing_step_that_fails_after_it_planned_steps_again_and_holds_short_of_twice_the_near_scrape(
        setup, monkeypatch):
    """A step that planned but failed or latched (not on the snag guard) is planned again from the measured joints, as
    one that did not plan; the landing never sweeps from where it stopped. A tip left under LAND_HOLD_M holds: on
    2026-10-13 the sweep from the 10 mm standoff nearly scraped the needles across the skin."""
    ex, io = setup.ex, setup.io
    _pen_down(monkeypatch)
    ex.camera = np.eye(4)
    _lifts_planned_up_to(ex, monkeypatch, ex.LIFT_STEP_M)
    io.script = [("ok", 1, {}, None), ("failed", 0, {}, None), ("latched", 0, {"latched": 1.0, "latch_reason": 1.0}, None)]
    assert ex._land_now() == (True, "landed and idle")
    assert io.q[2] == pytest.approx(ex.LAND_CLEAR_M) and io.unlatches == 1 and len(io.executed) == 10
    io.q, io.safety["landed"] = np.zeros(7), 0.0
    io.script = [("ok", 1, {}, None), ("ok", 1, {}, None)] + [("failed", 0, {}, None)] * 9
    ok, why = ex._land_now()
    assert not ok and why.startswith("not landing") and io.q[2] == pytest.approx(0.010) and io.lands == 1


def test_a_snag_held_needle_s_landing_lift_goes_on_at_half_and_in_steps_under_the_guard_then_clear_of_the_skin(
        riding, monkeypatch):
    """The needle out a standoff along the pen, then on along the skin's normal to LAND_CLEAR_M like any landing,
    under the guard throughout: on 538ada44c the sweep began from the standoff, where it nearly scraped (2026-10-13)."""
    ex, io = riding.ex, riding.io
    _pen_down(monkeypatch)
    ex.camera, ex.pen = np.eye(4), tatbot_motion.pen_down(RIDE, BALLPOINT)
    io.safety.update(SNAG)
    asked = _lifts_planned_up_to(ex, monkeypatch, 0.005)
    assert ex._land_now() == (True, "landed and idle") and io.unlatches == 1
    assert asked == ([(0.01, False), (0.005, False), (0.005, False), (0.03, True), (0.03, False), (0.015, True),
                      (0.015, False)] + [(0.005, False)] * 6) and io.q[2] == pytest.approx(ex.LAND_CLEAR_M)
    assert io.writes == [{"guard_mode": 3}, {"guard_mode": 0}]          # armed through all, off for the landing


def test_the_land_after_a_snag_hold_withdraws_along_the_pen_under_the_guard_again(riding, monkeypatch):
    """A snag-tripped needle no lift planned for holds; the draw then disarms the guard and its latch is gone, yet the
    next land still withdraws the needle a standoff along the pen under the guard (2026-10-08), not 40 mm along the
    skin's normal unguarded."""
    ex, io = riding.ex, riding.io
    _pen_down(monkeypatch)
    ex.camera, ex.pen, io.q = np.eye(4), tatbot_motion.pen_down(RIDE, BALLPOINT), np.array(IN_SKIN, float)
    io.safety.update(SNAG)
    _lifts_planned_up_to(ex, monkeypatch, 0.0)
    assert not ex._land_now()[0] and io.writes == [{"guard_mode": 3}]
    ex.snag_guard(False)                                                 # the draw's end
    monkeypatch.delattr(ex, "lift_plan")
    asked = _lifts_planned_up_to(ex, monkeypatch, 1.0)
    assert ex._land_now() == (True, "landed and idle") and asked == [(0.01, False), (0.0305, True)]
    assert io.writes == [{"guard_mode": 3}, {"guard_mode": 0}, {"guard_mode": 3}, {"guard_mode": 0}]


def test_a_landing_no_lift_plans_for_after_gauge_holds_holds_too(setup, monkeypatch):
    """The gauge holds' hop (_hop_from) and a refused landing (_land_hold) keep their own state: one attribute held
    both, so after a set-up's holds (every tattoo's) the landing's hold check failed on the hop's tuple and the
    landing swept from the skin anyway."""
    ex, io = setup.ex, setup.io
    _pen_down(monkeypatch)
    _hop_executor(setup, monkeypatch)
    ex._hold_travel(0, _stand(0.0, 0.010))
    io.q = np.array([0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0])                 # the tip 2 mm over the skin
    _lifts_planned_up_to(ex, monkeypatch, 0.0)
    ok, why = ex._land_now()
    assert not ok and why.startswith("not landing: no lift off the skin plans from here") and io.lands == 0
    assert not [text for kind, _, text in setup.node.events if "inhomogeneous" in text]


@pytest.mark.parametrize("case", ["snag", "latch", "high"])
def test_a_latch_the_driver_keeps_holds_a_landing_that_would_sweep_from_near_the_skin(riding, monkeypatch, case):
    """The driver lands from its latch but refuses the unlatch a lift needs once the arm stands 0.05 rad off the latched
    pose, as when the work drags a snag-held needle: 538ada44c landed from there and swept the work along. It holds as
    when no lift plans; a tip LAND_HOLD_M over the skin lands, as any landing does."""
    ex, io = riding.ex, riding.io
    _pen_down(monkeypatch)
    ex.camera, ex.pen = np.eye(4), tatbot_motion.pen_down(RIDE, BALLPOINT)
    io.q = np.array(IN_SKIN, float) if case == "snag" else np.array([0.0121, 0.0, 0.002, 0, 0, 0, 0], float)
    io.q[2] += 0.028 if case == "high" else 0.0
    io.safety.update(SNAG if case == "snag" else {"latched": 1.0, "latch_reason": 1.0})
    monkeypatch.setattr(io, "unlatch", lambda: (False, "the driver kept the latch (unlatch refused: command steps from "
                                                       "the held pose)"))
    ok, why = ex._land_now()
    assert (ok, io.lands) == (True, 1) if case == "high" else (
        not ok and "the driver kept the latch" in why and io.lands == 0 and io.executed == [])
    if case != "high":
        assert not ex.decide(rules.LAND, asked=True)[0] and io.lands == 0    # said to a person first
        assert ex.decide(rules.LAND, asked=True)[0] and io.lands == 1        # then it lands as they decided


def test_a_draw_whose_landing_was_refused_ends_not_landed_and_held(riding, monkeypatch):
    """The run's outcome said "landed (not landing: ...)" for an arm left holding over the skin."""
    from tatbot_session import drawrun

    ex, io = riding.ex, riding.io
    _pen_down(monkeypatch)
    ex.camera, io.q = np.eye(4), np.array([0.01, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0])
    _lifts_planned_up_to(ex, monkeypatch, 0.0)
    monkeypatch.setattr(ex, "draw", lambda program, ledger, **kw: ex.land())
    out = drawrun.draw(ex, {}, riding.ledger, from_op="", touch=False)
    assert out.startswith("not landed, held (not landing: no lift off the skin plans from here") and io.lands == 0
    io.q[2] = 0.05
    assert drawrun.draw(ex, {}, riding.ledger, from_op="", touch=False) == "landed (landed and idle)"


def test_a_stack_that_knows_no_page_holds_a_pen_it_withdrew_short_of_withdraw_m(setup, monkeypatch):
    """With no page known the pen comes out approach.withdraw_m along its axis, the tip having sagged 9-13 mm into the
    skin (2026-10-08): 538ada44c landed once a standoff of it was out. A withdraw left short holds."""
    ex, io = setup.ex, setup.io
    assert ex.camera is None
    _lifts_planned_up_to(ex, monkeypatch, ex.LIFT_STEP_M)
    io.script = [("ok", 1, {}, None)] * 2 + [("failed", 0, {}, None)] * 8
    ok, why = ex._land_now()
    assert not ok and why.startswith("not landing") and io.lands == 0 and len(io.executed) == 10


@pytest.mark.parametrize("where", ["over", "high", "beside", "far", "unknown"])
def test_a_stack_that_knows_no_page_withdraws_and_holds_only_a_tip_the_cameras_page_puts_near_the_skin(
        setup, monkeypatch, where):
    """2026-10-21 joint_3 bench: a stack restarted for a tuning knew no page; the tip stood 0.51 m over the skin and
    0.25 m beside the print, the pen pointing up and out, and the 30 mm withdraw along it needed joint_1 past its stop:
    the landing held as if the needles were in the skin. The cameras' page, measured within page.max_lost_s, says
    where the skin is: off the print and LAND_CLEAR_M round it, or that far over it, the arm lands as it stands. Over
    the print the withdraw stands, and one left short holds under LAND_HOLD_M; with no page at all, as before."""
    ex, io, node = setup.ex, setup.io, setup.node
    _pen_down(monkeypatch)
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "size_m": [0.045, 0.063]}}
    tip = {"over": [0.01, 0.02, -0.005], "high": [0.01, 0.02, 0.025], "beside": [0.0, 0.075, 0.002],
           "far": [-0.24, -0.11, 0.51], "unknown": [-0.24, -0.11, 0.51]}[where]
    io.q = np.array([*tip, 0.0, 0.0, 0.0, 0.0])                          # away from rest
    node.camera = None if where == "unknown" else node.camera
    asked = _lifts_planned_up_to(ex, monkeypatch, 0.0)                  # no lift plans: joint_1 on its stop
    ok, why = ex._land_now()
    held = where in ("over", "unknown")
    assert ex.camera is None and (ok, io.lands, bool(asked)) == ((False, 0, True) if held else
                                                                (True, 1, where == "high"))
    assert why.startswith("not landing: no lift off the skin plans") if held else why == "landed and idle"


@pytest.mark.parametrize("y, lands", [(0.02, False), (0.075, True)])
def test_a_landing_beside_the_print_neither_lifts_along_its_normal_nor_holds(setup, monkeypatch, y, lands):
    """The page in use's plane runs on past the print: a tip beside it under LAND_CLEAR_M lifted along its normal and,
    no lift planning, held as if over the skin. Off the print and LAND_CLEAR_M round it the arm lands as it stands;
    over it the lift and the hold stand."""
    ex, io = setup.ex, setup.io
    _pen_down(monkeypatch)
    ex.camera = np.eye(4)
    ex.stack = {**ex.stack, "page": {**ex.stack["page"], "size_m": [0.045, 0.063]}}
    io.q = np.array([0.0, y, 0.005, 0.0, 0.0, 0.0, 0.0])
    asked = _lifts_planned_up_to(ex, monkeypatch, 0.0)
    ok, why = ex._land_now()
    assert (ok, io.lands, bool(asked)) == ((True, 1, False) if lands else (False, 0, True))


def test_a_fresh_run_seeds_its_blind_search_from_the_newest_earlier_run_of_its_print(setup, monkeypatch, tmp_path):
    """2026-10-21: after a stack restart a fresh run on a 2x2 forearm piece the overhead cannot read had no page to aim
    the wrist at and waited page.wait_s to refuse. The newest earlier run's retained page of the same print, under
    SEED_MAX_AGE_S, seeds it; another print's, an older one, or this run's own does not."""
    import copy
    import os

    ex = setup.ex
    ex.stack = copy.deepcopy(STACK)
    ex.stack["page"]["pattern_id"] = "stencil-a"
    workspace = {"tool": "lutin-3rl-bugpin"}
    monkeypatch.setattr(ex, "workspace", lambda: workspace)

    def run(name, pattern, age_s):
        d = tmp_path / "ros-draw" / name
        d.mkdir(parents=True)
        page = {"id": name, "pattern_id": pattern, "camera_at_setup": np.eye(4).tolist(),
                "correction": np.eye(4).tolist(), "holds": {"used": True}}
        (d / "ledger.jsonl").write_text(json.dumps({"arm": "right", "event": "page", "workspace": workspace,
                                                    "page": page}) + "\n")
        t = os.path.getmtime(d / "ledger.jsonl") - age_s
        os.utime(d / "ledger.jsonl", (t, t))

    run("old-same", "stencil-a", ex.SEED_MAX_AGE_S + 60)
    run("new-other", "stencil-b", 10)
    run("mid-same", "stencil-a", 600)
    ex.run_dir = tmp_path / "ros-draw" / "this"
    ex.run_dir.mkdir()
    assert ex._seed_page()["id"] == "mid-same"
    ex.stack["page"]["pattern_id"] = "stencil-c"
    assert ex._seed_page() is None
    ex.run_dir = None
    assert ex._seed_page() is None
