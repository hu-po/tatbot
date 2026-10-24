"""`ros2 run tatbot_session session`: the orchestrator node `tatbot_session`.

Serves /tatbot/draw, /tatbot/touch, /tatbot/land and /tatbot/decide; publishes /tatbot/events,
/tatbot/safety (one SafetyState per arm, 20 Hz and on change) and /tatbot/goals (every
FollowJointTrajectory goal it sends, for the run bag). One ArmExecutor per arm in stack.yaml `arms`.

Parameter `stack`: the effective stack.yaml that stack.launch.py writes into the ros-stack run (launch
arguments applied; '' = the installed default). The repo is $TATBOT_REPO (config/, urdf/, scripts/).
At startup it publishes its runtime identity to the stack's `runtime_record`, which the launch sets;
a stack without one, such as the installed default, publishes none.
"""
from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tatbot_description import names, repo_root
from tatbot_interfaces.action import Draw, Land, Touch
from tatbot_interfaces.msg import Event, Page, SafetyState
from tatbot_interfaces.srv import Decide
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory

from tatbot_session import config, drawrun, geometry, machine, timing
from tatbot_session.arm import ArmIO, RosClock
from tatbot_session.executor import ArmExecutor, Cancelled, Landed
from tatbot_session.ledger import GoalLog, Ledger, counts, uncertain
from tatbot_session.ports import WALL

GOALS_TOPIC = "/tatbot/goals"


class Bag:
    """`ros2 bag record` (MCAP) of the run's topics into <run_dir>/<name>, stopped with SIGINT."""

    def __init__(self, path: Path, topics: list[str]):
        self.log = open(path.parent / f"{path.name}.log", "w")  # noqa: SIM115 - lives as long as the recorder
        self.proc = subprocess.Popen(["ros2", "bag", "record", "-s", "mcap", "--include-hidden-topics",
                                      "-o", str(path), *topics], stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.log.close()


class SessionNode(Node):
    def __init__(self):
        super().__init__(names.SESSION_NODE)
        self.declare_parameter("stack", "")
        self.stack = config.load_stack(self.get_parameter("stack").value or None)
        # the session's clock: the wall clock on the rig and fake hardware; the simulator's (/clock) on hardware sim
        self.clock = (RosClock(self, (self.stack.get("sim") or {}).get("epoch_utc", 0.0))
                      if self.stack["hardware"] == "sim" else WALL)
        self.repo = repo_root(None)
        sys.path.insert(0, str(self.repo/'scripts/lib'))
        self.runlog = config.runlog(self.repo)
        from tatbot_session import runtime

        self.runtime_identity = runtime.session_identity(self.repo, self.stack.get('controller_process_file'),
                                                         self.stack.get('runtime_configuration_sha256'),
                                                         self.stack.get('runtime_workspace_sha256'))
        record = self.stack.get('runtime_record')
        if record:
            runtime.publish(record, self.runtime_identity)
        self.cb = ReentrantCallbackGroup()
        self.events = self.create_publisher(Event, names.EVENTS_TOPIC, 50)
        self.safety_pub = self.create_publisher(SafetyState, names.SAFETY_TOPIC, 20)
        self.goals_pub = self.create_publisher(JointTrajectory, GOALS_TOPIC, 10)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.pages: dict[str, Page] = {}
        self.page_measured_at: dict[str, float] = {}  # pattern_id -> self.clock time of its last measured pose
        self.page_history: dict[str, collections.deque] = {}  # pattern_id -> (self.clock time, Page, its page_marks) of the newest measured
        self.create_subscription(Page, names.PAGE_TOPIC, self._on_page, 10, callback_group=self.cb)
        self.runs: dict[str, object] = {}  # arm -> the RunLog it writes events to

        from tatbot_motion import load_motion

        motion = load_motion()
        self.io: dict[str, ArmIO] = {}
        self.executors: dict[str, ArmExecutor] = {}
        for arm in self.stack["arms"]:
            io = ArmIO(self, arm, self.cb, on_safety_change=self._on_safety_change, clock=self.clock)
            io.goals = self.goals_pub
            self.io[arm] = io
            self.executors[arm] = drawrun.build_executor(
                self.stack, arm, io=io, node=self, clock=self.clock, sensing=None, repo=self.repo, motion=motion,
                machine=machine.from_stack(self.stack, arm, clock=self.clock))
        self.guard, text = drawrun.wire_guard(self.repo, self.stack, motion, self.io)
        self.get_logger().info(text)
        self.create_subscription(JointState, names.JOINT_STATES_TOPIC, self._on_joints, 20,
                                 callback_group=MutuallyExclusiveCallbackGroup())
        for arm, ex in self.executors.items():
            self.create_subscription(PoseWithCovarianceStamped, names.ee_topic(arm), ex.io.on_ee, 20)
            self.create_subscription(String, names.keys_topic(arm), lambda msg, ex=ex: ex.on_key(msg.data), 10,
                                     callback_group=self.cb)
        self.create_timer(0.05, self._publish_safety, callback_group=self.cb)
        ActionServer(self, Draw, names.DRAW_ACTION, self._draw, goal_callback=self._goal_ok,
                     cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self.cb)
        ActionServer(self, Touch, names.TOUCH_ACTION, self._touch, goal_callback=self._goal_ok,
                     cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self.cb)
        ActionServer(self, Land, names.LAND_ACTION, self._land, callback_group=self.cb)
        self.create_service(Decide, names.DECIDE_SERVICE, self._decide, callback_group=self.cb)
        self.get_logger().info(f"arms {self.stack['arms']} hardware {self.stack['hardware']} page "
                               f"{self.stack['page']['source']} ({self.stack['page'].get('geometry', 'stack.yaml')}) "
                               f"touch {self.stack['touch']['enabled']} "
                               f"estop {self.stack['estop']['source']} machine switch "
                               f"{' '.join(f'{arm}={self._machine_switch(arm)}' for arm in self.stack['arms'])} "
                               f"repo {self.repo} "
                               f"runtime record {record or 'none'}")

    # --- inputs --------------------------------------------------------------------------------
    def _on_joints(self, msg: JointState) -> None:
        for io in self.io.values():
            io.on_joint_state(msg)

    def _on_page(self, msg: Page) -> None:
        self.pages[msg.pattern_id] = msg
        if msg.source == Page.SOURCE_MEASURED:
            self.page_measured_at[msg.pattern_id] = self.clock.now()
            # two minutes at the overhead's 1 Hz: the sheet watch's medians and a move's hold (geometry.sheet_watch)
            self.page_history.setdefault(msg.pattern_id, collections.deque(maxlen=120)).append(
                (self.clock.now(), msg, geometry.page_marks(msg.support_json)))

    def camera_page(self, arm: str):
        """(base_from_page 4x4 as the camera sees it, info) or None. page.source fixed: stack.yaml's pose."""
        page = self.stack["page"]
        if page["source"] == "fixed":
            return drawrun.fixed_page(self.stack, arm)
        msg = self.pages.get(page["pattern_id"])
        if msg is None:
            return None
        parent_from_page = self._base_from_page(msg, arm)
        if parent_from_page is None:
            return None
        return parent_from_page, {"source": {1: "measured", 2: "lost", 3: "fixed"}.get(msg.source, str(msg.source)),
                                  "pattern_id": msg.pattern_id, "print_id": msg.print_id,
                                  "identity_verified": msg.identity_verified,
                                  "stamp": msg.pose.header.stamp.sec + msg.pose.header.stamp.nanosec * 1e-9,
                                  "measured_age_s": geometry.measured_age(self.page_measured_at.get(msg.pattern_id),
                                                                 self.clock.now())}

    def measured_pages(self, arm: str, since: float) -> list[np.ndarray]:
        """base_from_page of each of the camera's newest measured poses received after `since` (self.clock),
        oldest first: a lost page's held pose is not a measurement."""
        rows = self.page_history.get(self.stack["page"].get("pattern_id"), ())
        poses = [self._base_from_page(msg, arm) for t, msg, _ in list(rows) if t > since]
        return [p for p in poses if p is not None]

    def measured_marks(self, since: float) -> list[tuple[float, dict]]:
        """(when, self.clock; geometry.page_marks) of the camera's newest measured pages received after `since`, oldest
        first; a page without marker readings gives none."""
        rows = self.page_history.get(self.stack["page"].get("pattern_id"), ())
        return [(t, marks) for t, _, marks in list(rows) if t > since and marks]

    def _base_from_page(self, msg: Page, arm: str) -> np.ndarray | None:
        """A Page's pose in the arm's base frame, or None before the transform is known."""
        pose = msg.pose.pose.pose
        parent_from_page = geometry.quat_matrix(
            (pose.position.x, pose.position.y, pose.position.z),
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
        base_from_parent = self.base_from_world(arm, msg.pose.header.frame_id or names.WORLD)
        return None if base_from_parent is None else base_from_parent @ parent_from_page

    def base_from_world(self, arm: str, frame: str = names.WORLD) -> np.ndarray | None:
        """The arm's base from `frame` (the overhead's colour optical frame, its marker readings' and poses'), or None
        before the transform is known."""
        if frame == names.base_frame(arm):
            return np.eye(4)
        try:
            tf = self.tf.lookup_transform(names.base_frame(arm), frame, rclpy.time.Time()).transform
        except Exception:  # noqa: BLE001 - no transform yet: no page yet
            return None
        return geometry.quat_matrix((tf.translation.x, tf.translation.y, tf.translation.z),
                                    (tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w))

    def wait_camera_page(self, arm: str, timeout: float, max_lost_s=None, should_stop=None, since=None):
        """The camera's page, waiting up to `timeout` for one; with `max_lost_s`, for one measured at most
        that long ago (a lost page stands for its last measured pose, which may be minutes old); with `since`
        (self.clock), for one measured after it. Returns the newest page at the deadline, or when `should_stop()`,
        whatever its age; `geometry.stale_page` says whether to refuse it."""
        deadline = self.clock.now() + timeout
        while True:
            page = self.camera_page(arm)
            fresh = (page is not None and (max_lost_s is None or geometry.stale_page(page[1], max_lost_s) is None)
                     and (since is None or self.page_measured_at.get(self.stack["page"]["pattern_id"], since) > since))
            if fresh or self.clock.now() > deadline or (should_stop is not None and should_stop()):
                return page
            self.clock.sleep(0.1)

    # --- outputs -------------------------------------------------------------------------------
    def event(self, kind: str, *, arm: str = "", op_id: str = "", text: str = "") -> None:
        run = self.runs.get(arm) or next(iter(self.runs.values()), None)
        msg = Event(stamp=self.get_clock().now().to_msg(), arm=arm, kind=kind, op_id=op_id,
                    run_id=getattr(run, "run_id", ""), text=text)
        self.events.publish(msg)
        line = f"[{arm or '-'}] {kind} {op_id} {text}".replace("  ", " ")
        # one call site per severity: rclpy refuses a call site whose severity changes, and the error that
        # raised it then ended the action with an empty message (2026-09-28)
        if kind == Event.KIND_ERROR:
            self.get_logger().error(line)
        else:
            self.get_logger().info(line)
        if run is not None:
            run.event("ros_event", arm=arm, event=kind, op=op_id, text=text)
            with open(run.dir / "console.log", "a", encoding="utf-8") as stream:
                stream.write(f"{time.strftime('%H:%M:%S')} {line}\n")

    def _safety_msg(self, arm: str) -> SafetyState:
        io = self.io[arm]
        s = io.safety

        def num(name, default=0.0):
            value = s.get(name, default)
            return default if value is None or math.isnan(value) else value

        trip = io.trip_joints()
        switch = self.executors[arm].machine
        powered, reply_age = machine.reply(switch)
        return SafetyState(
            stamp=self.get_clock().now().to_msg(), arm=arm, estop_source=int(num("estop_source")),
            estop_ok=io.estop_ok, estop_age_s=float(num("estop_age_s", -1.0)),
            probe_triggered=io.flag("probe_triggered"), latched=io.latched, latch_reason=io.latch_reason,
            guard_tripped=io.flag("guard_tripped"), guard_mode=int(num("guard_mode")),
            trip_joints=[] if trip is None else trip.tolist(), landing=io.flag("landing"), landed=io.flag("landed"),
            machine_off=switch is None or switch.is_off(), machine_powered=powered, machine_reply_age_s=reply_age,
            controller_error=io.flag("controller_error"), rt_period_max_ms=float(num("rt_period_max_ms")))

    def _publish_safety(self) -> None:
        for arm, io in self.io.items():
            if io.safety:
                self.safety_pub.publish(self._safety_msg(arm))

    def _on_safety_change(self, arm: str, before: dict, after: dict) -> None:
        self.safety_pub.publish(self._safety_msg(arm))
        was_latched, latched = before.get("latched", 0.0) >= 0.5, after.get("latched", 0.0) >= 0.5
        was_ok, ok = before.get("estop_ok", 1.0) >= 0.5, after.get("estop_ok", 1.0) >= 0.5
        if latched and not was_latched:
            self.executors[arm].machine_off()   # the arm holds; a running machine would strike one spot
            self.event(Event.KIND_LATCHED, arm=arm, text=self.io[arm].latch_text())
        if latched and ok and not was_ok:
            land = (self.stack.get("session") or {}).get("estop_release", "decide") == "land"
            self.event(Event.KIND_RELEASED, arm=arm,
                       text="e-stop released: " + ("a run waiting on it lands" if land else "decide continue or land"))

    # --- runs ----------------------------------------------------------------------------------
    def _run_meta(self, arms, program: dict | None) -> dict:
        from tatbot_session import runtime

        meta = {"revision": config.revision(self.repo), "runtime": runtime.current(self.runtime_identity), "hardware": self.stack["hardware"],
                "estop_source": self.stack["estop"]["source"],
                "page_source": self.stack["page"]["source"], "touch": self.stack["touch"]["enabled"], "arms": {}}
        if program is not None:
            from tatbot_contracts.ros_program import program_sha256

            meta["program"] = {"sha256": program_sha256(program),
                               "design": program.get("design", {}), "resources": program.get("resources", [])}
        for arm in arms:
            ws = config.workspace_arm(self.repo, arm)
            registration = str(self.stack.get("registration", {}).get(arm) or "")
            meta["arms"][arm] = {
                "tool": ws.get("tool_id"), "tcp": {"frame": ws.get("tip_frame"),
                                                   "xyz": [ws.get(f"pen_tip_offset_{a}") for a in "xyz"]},
                "trim": self.stack["page"].get("trim", {}).get(arm, [0.0, 0.0]),
                "machine_switch": self._machine_switch(arm),
                "pattern_id": (names.fixed_pattern_id(arm) if self.stack["page"]["source"] == "fixed"
                               else self.stack["page"]["pattern_id"]),
                "registration": registration, "registration_sha256": config.file_sha256(registration)}
        return meta

    def _machine_switch(self, arm: str) -> str:
        """The tattoo machine switch stack.yaml `machine` gives this arm: none, pi or sim."""
        cfg = self.stack.get("machine") or {}
        return cfg.get("switch", "none") if cfg.get("arm") == arm else "none"

    def _reload_trim(self) -> None:
        """Adopt page.trim from the stack.yaml the launch read (deployed since, perhaps) for this goal."""
        trim = config.page_trim(self.stack.get("config_path"))
        if trim is not None and trim != self.stack["page"].get("trim"):
            self.get_logger().info(f"page trim {self.stack['page'].get('trim')} -> {trim} (from {self.stack['config_path']})")
            self.stack["page"]["trim"] = trim

    def _open_run(self, workflow: str, arms, program: dict | None, run_id: str = ""):
        self._reload_trim()
        rl = self.runlog
        if run_id:
            run_dir = rl.log_root(rl.load_config()) / workflow / run_id
            if not run_dir.is_dir():
                raise FileNotFoundError(f"no run {run_id} under {run_dir.parent}")
            run = rl.RunLog(run_dir, workflow, run_id)
            run.update(status="running", resumed_at=self.clock.utc())
            run.event("run.resume", argv=[workflow, run_id])
        else:
            run = rl.init(workflow, meta=self._run_meta(arms, program), attach_logging=False,
                          argv=["tatbot_session", workflow, *arms])
        for arm in arms:
            self.runs[arm] = run
        return run

    def _start_bag(self, run, arms) -> Bag:
        topics = [*names.bag_topics(arms), GOALS_TOPIC]   # the stack-wide bag's topics and the session's goals
        k = 1
        while (run.dir / ("bag" if k == 1 else f"bag-{k}")).exists():
            k += 1
        before = self.count_subscribers(names.EVENTS_TOPIC)
        bag = Bag(run.dir / ("bag" if k == 1 else f"bag-{k}"), topics)
        deadline = self.clock.now() + 8.0
        while self.count_subscribers(names.EVENTS_TOPIC) <= before and self.clock.now() < deadline:
            self.clock.sleep(0.1)
        return bag

    def _close_run(self, run, arms, status: str, exit_code: int) -> None:
        run.update(outcome=status)
        run.finalize(exit_code, status=status)
        for arm in arms:
            self.runs.pop(arm, None)

    def _goal_arms(self, goal) -> list[str]:
        """The arms a Draw (`arms`, else every stack arm) or Touch (`arm`, else the first) goal runs on."""
        if hasattr(goal, "arms"):
            return list(goal.arms) or list(self.stack["arms"])
        return [goal.arm or self.stack["arms"][0]]

    def _goal_ok(self, goal) -> GoalResponse:
        """Accept a Draw or Touch goal only for known arms that are free and have not landed: a landed arm is
        refused here, so there is nothing to cancel. A rejection carries no text, so the reason goes out as an
        error event; the clients read it back from the arm's `landed` on /tatbot/safety."""
        for arm in self._goal_arms(goal):
            ex = self.executors.get(arm)
            if ex is None or ex.busy.locked():
                return GoalResponse.REJECT
            why = ex.refusal()
            if why:
                self.event(Event.KIND_ERROR, arm=arm, text=f"goal refused: {why}")
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _acquire(self, arms) -> list[ArmExecutor]:
        """Take every arm of an accepted goal. One that landed since the acceptance refuses it here, with the
        reason as the result's message."""
        taken = []
        try:
            for arm in arms:
                ex = self.executors.get(arm)
                if ex is None or not ex.begin_goal():
                    raise RuntimeError(f"arm {arm} is unknown or busy")
                taken.append(ex)
                why = ex.refusal()
                if why:
                    raise RuntimeError(why)
        except RuntimeError:
            for ex in taken:
                ex.end_goal()
            raise
        return taken

    # --- Draw ----------------------------------------------------------------------------------
    def _draw(self, goal_handle):
        req = goal_handle.request
        arms = self._goal_arms(req)
        result = Draw.Result()
        try:
            execs = self._acquire(arms)
        except RuntimeError as exc:
            goal_handle.abort()
            result.message = str(exc)
            return result
        outcomes: dict[str, str] = {}
        opened: dict = {}
        try:
            self._draw_run(goal_handle, execs, arms, result, outcomes, opened)
        except Exception as exc:  # noqa: BLE001 - reported in the result and the run
            outcomes.setdefault("_", f"error: {exc}")
            self.get_logger().error(traceback.format_exc())
        finally:
            for ex in execs:
                ex.end_goal()
        complete = bool(outcomes) and all(v == "complete" for v in outcomes.values())
        cancelled = goal_handle.is_cancel_requested
        result.message = "; ".join(f"{k}: {v}" for k, v in outcomes.items())
        self.event(Event.KIND_RUN_END, text=result.message)
        if "run" in opened:
            timing.finish(opened['run'], opened['timing'])
            opened['run'].update(draw_timing=timing.summarize(opened['run'].dir/'run.jsonl'))
            self._close_run(opened["run"], arms, "ok" if complete else ("interrupted" if cancelled else "fail"),
                            int(not complete))
        (goal_handle.succeed if complete else goal_handle.canceled if cancelled else goal_handle.abort)()
        return result

    def _log_stack_run(self, run) -> None:
        """The driver's flight log lives in the stack's run, and a resume may run under another stack: each attempt
        names the one it ran under."""
        stack_param = self.get_parameter("stack").value
        if stack_param:
            run.event("stack_run", stack_run=Path(stack_param).parent.name, stack_dir=str(Path(stack_param).parent))

    def _draw_run(self, goal_handle, execs, arms, result, outcomes, opened) -> None:
        """Open (or resume) the run, record the bag, draw on every arm in parallel, fill the result."""
        req = goal_handle.request
        from tatbot_contracts.canonical import parse_json
        from tatbot_contracts.ros_program import program_sha256

        if req.run_id:
            program = parse_json((self._run_dir("ros-draw", req.run_id) / "program.json").read_bytes())
        else:
            program = parse_json(req.program_json)

        drawrun.check_program(program, self.stack, arms)
        if req.run_id and req.program_json and program_sha256(program) != program_sha256(parse_json(req.program_json)):
            raise ValueError(f'run {req.run_id} drew another program: resume it with the program it started with')
        run = opened["run"] = self._open_run("ros-draw", arms, program, req.run_id)
        opened['timing'] = timing.begin(run)
        self._log_stack_run(run)
        if not (run.dir / "program.json").exists():
            (run.dir / "program.json").write_text(json.dumps(program, indent=1) + "\n")
        # motion.yaml is read at every draw, so a deployed tuning change applies without a stack restart
        # (which costs the page tracker a minute or two); the run records the file it drew with.
        motion_path, motion_text = self._reload_motion(execs)
        (run.dir / "motion.yaml").write_text(motion_text)
        from tatbot_session.research import claim_slot, validate_motion
        from tatbot_session.runtime import current, validate_research

        validate_motion(program, motion_text)
        if program.get('research'):
            validate_research(program, current(self.runtime_identity), json.loads((run.dir / 'meta.json').read_text()).get('runtime'))
        claim_slot(self.runlog.log_root(self.runlog.load_config()), program, run.run_id, resume=bool(req.run_id))
        ledger = Ledger(run.dir / "ledger.jsonl", self.clock)
        log = GoalLog(run.dir / "goals.jsonl") if self.stack["session"].get("goal_log") else None
        for ex in execs:
            ex.io.goal_log = log
        bag = self._start_bag(run, arms)
        try:
            total = len(program["ops"])
            self.event(Event.KIND_RUN_START, text=f"{run.run_id} {total} ops arms {arms} motion.yaml "
                                                  f"{hashlib.sha256(motion_text.encode()).hexdigest()[:12]}")
            threads = []
            for ex in execs:
                ex.cancelled = lambda: goal_handle.is_cancel_requested
                ex.feedback = self._feedback_fn(goal_handle, ex.arm, total)
                ex.run_dir = run.dir
                threads.append(threading.Thread(target=self._draw_arm, args=(ex, program, ledger, req.from_op,
                                                                             outcomes)))
                threads[-1].start()
            for thread in threads:
                thread.join()
        finally:
            bag.stop()
            for ex in execs:   # a later Touch goal's ways are not this run's
                ex.io.goal_log = None
        rows = ledger.rows()
        tally = counts(rows)
        result.done, result.skipped = tally["done"], tally["skipped"]
        result.uncertain = [op for arm in arms for op in uncertain(program["ops"], rows, arm)]
        result.run_id, result.run_dir = run.run_id, str(run.dir)
        for ex in execs:
            drawrun.write_page_and_drawn(run.dir, ex, program, len(execs) > 1)

    def _reload_motion(self, execs) -> tuple[Path, str]:
        """Give every executor of this goal a fresh load of motion.yaml. Returns (its path, its text)."""
        import tatbot_motion

        path = tatbot_motion.motion_path()
        motion = tatbot_motion.load_motion(path)
        for ex in execs:
            ex.motion = motion
        return path, path.read_text()

    def _draw_arm(self, ex: ArmExecutor, program, ledger, from_op, outcomes) -> None:
        outcomes[ex.arm] = drawrun.draw(ex, program, ledger, from_op=from_op, touch=bool(self.stack["touch"]["enabled"]),
                                        on_error=self.get_logger().error)

    def _run_dir(self, workflow: str, run_id: str) -> Path:
        return self.runlog.log_root(self.runlog.load_config()) / workflow / run_id

    def _feedback_fn(self, goal_handle, arm: str, total: int):
        last = {"op_id": "", "index": 0}

        def feedback(*, phase: int, op_id: str | None = None, index: int | None = None, arc_m: float = 0.0):
            if op_id is not None:
                last["op_id"] = op_id
            if index is not None:
                last["index"] = index
            goal_handle.publish_feedback(Draw.Feedback(arm=arm, op_id=last["op_id"], index=last["index"], total=total,
                                                       phase=int(phase), arc_m=float(arc_m)))

        return feedback

    # --- Touch ---------------------------------------------------------------------------------
    def _touch(self, goal_handle):
        req = goal_handle.request
        (arm,) = self._goal_arms(req)
        result = Touch.Result()
        try:
            (ex,) = self._acquire([arm])
        except RuntimeError as exc:
            goal_handle.abort()
            result.message = str(exc)
            return result
        run = bag = single = None
        ok = False
        try:
            run = self._open_run("ros-touch", [arm], None)
            bag = self._start_bag(run, [arm])
            ex.cancelled = lambda: goal_handle.is_cancel_requested
            ex.feedback = lambda **kw: None
            ex.run_dir = run.dir
            base = names.base_frame(arm)
            if req.mode == Touch.Goal.MODE_PAGE:
                ex.setup_page(None, True)
                touches = ex.page_record["touches"]
                result.contact_poses = [self._pose(base, self._contact(t)) for t in touches]
                result.joints = touches[-1]["q"] if touches else []
                result.plane = ex.page_record.get("plane", [])
                result.tripped = len(touches) == int(self.stack["touch"]["points"])
            else:
                single = self._touch_single(ex, req)
                result.tripped, result.joints = single["tripped"], single["q"]
                result.contact_poses = [self._pose(base, np.array(single["contact"]))]
            ok = True
            holds = req.mode in (Touch.Goal.MODE_MOVE, Touch.Goal.MODE_VIEW)
            result.message = "touched" if result.tripped else ("holding" if holds else "no trip")
            if single and single["tripped"] and single["phase"] != Draw.Feedback.PHASE_TOUCH:
                result.message = f"touched before the slow leg (phase {single['phase']})"
        except Landed as exc:
            result.message = f"landed ({exc})"
        except Cancelled:
            result.message = "cancelled"
        except Exception as exc:  # noqa: BLE001
            result.message = f"error: {exc}"
            self.event(Event.KIND_ERROR, arm=arm, text=str(exc))
            self.get_logger().error(traceback.format_exc())
        finally:
            if bag is not None:
                bag.stop()
            ex.end_goal()
        if run is not None:
            (run.dir / "page.json").write_text(json.dumps(ex.page_record, indent=1) + "\n")
            if single is not None:
                (run.dir / "touch.json").write_text(json.dumps(single, indent=1) + "\n")
            result.run_dir = str(run.dir)
            self._close_run(run, [arm], "ok" if ok else "fail", 0 if ok else 1)
        (goal_handle.succeed if ok else (goal_handle.canceled if goal_handle.is_cancel_requested
                                         else goal_handle.abort))()
        return result

    @staticmethod
    def _touch_single(ex, req) -> dict:
        """A MODE_SINGLE goal: its start pose (a zero quaternion keeps the tool's measured orientation), and
        the touch's outcome as JSON-ready values for the run's touch.json."""
        p, o = req.start_pose.pose.position, req.start_pose.pose.orientation
        keep = np.linalg.norm([o.x, o.y, o.z, o.w]) < 0.5
        start = geometry.quat_matrix((p.x, p.y, p.z), (0.0, 0.0, 0.0, 1.0) if keep else (o.x, o.y, o.z, o.w))
        if keep:
            start[:3, :3] = ex.kin.fk(ex.io.measured())[:3, :3]
        direction = np.array([req.direction.x, req.direction.y, req.direction.z], dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        if req.mode in (Touch.Goal.MODE_MOVE, Touch.Goal.MODE_VIEW):
            guard = req.guard
            out = ex.move_single(start, guard, view=req.mode == Touch.Goal.MODE_VIEW)
        else:
            guard = req.guard or Touch.Goal.GUARD_TIP_LAG
            out = ex.touch_single(start, direction, req.max_travel_m, req.speed_m_s, guard, req.prior_distance_m)
        return {"guard": int(guard), "start": start.tolist(), "direction": direction.tolist(),
                "prior_distance_m": req.prior_distance_m, **out, "contact": out["contact"].tolist(),
                "q": [float(v) for v in out["q"]]}

    @staticmethod
    def _contact(touch: dict) -> np.ndarray:
        """A touch's tcp pose moved to its first contact (the trip pose when no fit was made)."""
        pose = np.array(touch["tcp"], dtype=float)
        if touch.get("contact") is not None:
            pose[:3, 3] = touch["contact"]
        return pose

    def _pose(self, frame: str, matrix: np.ndarray):
        from geometry_msgs.msg import PoseStamped

        msg = PoseStamped()
        msg.header.frame_id = frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in matrix[:3, 3])
        qx, qy, qz, qw = geometry.matrix_quat(matrix)
        msg.pose.orientation.x, msg.pose.orientation.y = float(qx), float(qy)
        msg.pose.orientation.z, msg.pose.orientation.w = float(qz), float(qw)
        return msg

    # --- Land and Decide -----------------------------------------------------------------------
    def _land(self, goal_handle):
        arms = list(goal_handle.request.arms) or list(self.stack["arms"])
        result = Land.Result()
        texts = []
        for arm in arms:
            ex = self.executors.get(arm)
            if ex is None:
                result.landed.append(False)
                texts.append(f"{arm}: unknown arm")
                continue
            goal_handle.publish_feedback(Land.Feedback(arm=arm, text="landing"))
            ok, text = ex.decide(2, timeout=self.stack["safety"]["landing"]["budget_s"] + 10.0)
            if ok and ex.busy.locked():  # a running goal lands itself; wait for it
                deadline = self.clock.now() + self.stack["safety"]["landing"]["budget_s"] + 30.0
                while ex.busy.locked() and self.clock.now() < deadline:
                    self.clock.sleep(0.1)
                ok = self.io[arm].flag("landed")
                text = "landed" if ok else "not landed"
            result.landed.append(bool(ok))
            texts.append(f"{arm}: {text}")
        result.message = "; ".join(texts)
        goal_handle.succeed()
        return result

    def _decide(self, request, response):
        ex = self.executors.get(request.arm or self.stack["arms"][0])
        if ex is None:
            response.accepted, response.reason = False, f"unknown arm {request.arm}"
            return response
        response.accepted, response.reason = ex.decide(int(request.decision), asked=True)
        return response


def main(argv=None) -> int:
    rclpy.init(args=argv)
    node = SessionNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        for ex in node.executors.values():
            if ex.machine is not None:
                ex.machine.close()
        node.destroy_node()
        with contextlib.suppress(Exception):  # the SIGINT handler may have shut the context down already
            rclpy.try_shutdown()
    return 0
