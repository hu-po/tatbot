"""One arm's ROS I/O: measured joints, the <arm>_safety GPIO states and requests, and the JTC action.

The request protocols are ros/README.md "Continue and land, never a step": a hold goal at the measured
joints, then `unlatch = N` and a wait of at most 1 s for `unlatch_ack == N`; `land = N` for the
driver's landing. On mock hardware nothing latches and nothing acks: an unlatched arm needs no unlatch.
"""
from __future__ import annotations

import threading

import numpy as np
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration, Time
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import DynamicInterfaceGroupValues, InterfaceValue
from rclpy.action import ActionClient
from rclpy.duration import Duration as RosDuration
from tatbot_description import names
from trajectory_msgs.msg import JointTrajectoryPoint

from tatbot_session import ports
from tatbot_session.ports import WALL, SafetyReadings, execute, goal_points, goal_tag, stop_reason

LATCH_TEXT = {
    names.LATCH_NONE: "none", names.LATCH_ESTOP: "estop", names.LATCH_ESTOP_STALE: "estop_stale",
    names.LATCH_CARRIAGE_CONTACT: "carriage_contact", names.LATCH_STALL: "stall",
    names.LATCH_OVER_VELOCITY: "over_velocity", names.LATCH_GUARD_TIP_LAG: "guard_tip_lag",
    names.LATCH_GUARD_PROBE: "guard_probe", names.LATCH_CONTROLLER_ERROR: "controller_error",
    names.LATCH_DEACTIVATED: "deactivated", names.LATCH_STEP_REFUSED: "step_refused",
    names.LATCH_GUARD_SNAG: "guard_snag",
}

# A change in any of these publishes SafetyState at once; ages and RT periods ride the 20 Hz timer.
CHANGES = ("estop_ok", "latched", "latch_reason", "guard_mode", "guard_tripped", "unlatch_ack", "land_ack",
           "landing", "landed", "controller_error", "probe_triggered")


def _duration(seconds: float) -> Duration:
    """Rounded to the nanosecond as a whole: a time a hair under a second (14.9999999998 s, a 15 s jog's last knot)
    rounded its fraction up to 1e9 ns and wrapped to 14.0 s, and the controller rejected the goal (2026-10-12)."""
    sec, nanosec = divmod(int(round(seconds * 1e9)), 1_000_000_000)
    return Duration(sec=sec, nanosec=nanosec)


class RosClock:
    """The node's clock as the session's (ports.Clock) under use_sim_time (hardware sim): the simulator's time, which
    /clock carries; utc() puts it on the run's epoch. The rig and fake hardware keep ports.WallClock."""

    def __init__(self, node, epoch_utc: float = 0.0):
        self.clock, self.epoch_utc = node.get_clock(), float(epoch_utc)

    def now(self) -> float:
        return self.clock.now().nanoseconds * 1e-9

    def utc(self) -> float:
        return self.epoch_utc + self.now()

    def sleep(self, seconds: float) -> None:
        self.clock.sleep_until(self.clock.now() + RosDuration(seconds=seconds))

    def wait(self, cv, predicate, timeout_s: float, poll_s: float = 0.05) -> bool:
        deadline = self.now() + timeout_s
        while not predicate() and self.now() < deadline:
            cv.wait(poll_s)
        return bool(predicate())


class ArmIO(SafetyReadings):
    clock = WALL      # the session's clock (ports.Clock); the node passes RosClock on simulated hardware
    goal_log = None   # ledger.GoalLog: every goal sent, into the run's goals.jsonl (stack.yaml session.goal_log)

    def __init__(self, node, arm: str, callback_group, on_safety_change=None, clock=None):
        self.node, self.arm, self.clock = node, arm, clock or WALL
        self.joint_names = names.joint_names(arm)
        self._cv = threading.Condition()
        self.q: np.ndarray | None = None
        self.q_time = 0.0
        self._recording: list | None = None
        self._ee_recording: list | None = None
        self.ee_samples: list = []
        self.safety: dict[str, float] = {}
        self.safety_time = 0.0
        self.last_command: np.ndarray | None = None
        self.active = None   # the trajectory JTC runs or ran last: a replacement's, after a pen trim mid-goal
        self._request = int(self.clock.utc())
        self._on_change = on_safety_change
        self.goals = None  # a JointTrajectory publisher: every goal sent, for the run bag
        self.op_id = ""    # the op the executor last sent, until it is done: tags the published goals
        # (arm, joint rows) -> why the way must not be sent, or None: the arm's own clearance (the node's
        # tatbot_motion.collision.Guard); None when no registration places the arm.
        self.clearance = None
        self.jtc = ActionClient(node, FollowJointTrajectory, names.follow_joint_trajectory(arm),
                                callback_group=callback_group)
        self.gpio = node.create_publisher(DynamicInterfaceGroupValues, names.safety_commands_topic(arm), 10)
        node.create_subscription(DynamicInterfaceGroupValues, names.safety_states_topic(arm), self._on_gpio, 10,
                                 callback_group=callback_group)

    # --- state ---------------------------------------------------------------------------------
    def on_joint_state(self, msg) -> None:
        index = {name: i for i, name in enumerate(msg.name)}
        if not all(name in index for name in self.joint_names):
            return
        q = np.array([msg.position[index[name]] for name in self.joint_names])
        with self._cv:
            self.q, self.q_time = q, self.clock.now()
            if self._recording is not None and len(msg.effort) == len(msg.name):
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self._recording.append((q, np.array([msg.effort[index[name]] for name in self.joint_names]), t))
            self._cv.notify_all()

    def on_ee(self, msg) -> None:
        """The overhead-tracked EE fiducial (tatbot_bridge, in this arm's base frame), kept while recording."""
        with self._cv:
            if self._ee_recording is not None:
                p = msg.pose.pose.position
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self._ee_recording.append((t, np.array([p.x, p.y, p.z]), float(np.sqrt(max(msg.pose.covariance[0], 0.0)))))

    def record(self, on: bool) -> list:
        """Start (on) or stop keeping every (joints, external efforts, stamp) sample and every EE fiducial
        (stamp, position, sigma); stopping returns the joint samples (the fiducials: ee_samples)."""
        with self._cv:
            samples, self._recording = self._recording or [], ([] if on else None)
            self.ee_samples, self._ee_recording = (self._ee_recording or [], [] if on else None)
        return samples

    def _on_gpio(self, msg) -> None:
        group = names.safety_gpio(self.arm)
        if group not in msg.interface_groups:
            return
        values = msg.interface_values[msg.interface_groups.index(group)]
        state = dict(zip(values.interface_names, values.values, strict=False))
        with self._cv:
            before, self.safety, self.safety_time = self.safety, state, self.clock.now()
            self._cv.notify_all()
        if self._on_change is not None and any(before.get(k) != state.get(k) for k in CHANGES):
            self._on_change(self.arm, before, state)

    def latch_text(self) -> str:
        return LATCH_TEXT.get(self.latch_reason, str(self.latch_reason))

    def measured(self, timeout: float = 5.0) -> np.ndarray:
        with self._cv:
            if not self.clock.wait(self._cv, lambda: self.q is not None, timeout):
                raise RuntimeError(f"{self.arm}: no /joint_states for {self.joint_names[0]}")
            return self.q.copy()

    def observed(self, *, after=None, timeout=1.0):
        """Fresh measured joints (under 1 s old, and newer than `after`) and their receipt stamp (self.clock)."""
        with self._cv:
            if not self.clock.wait(self._cv, lambda: self.q is not None and self.clock.now() - self.q_time <= 1.0
                                   and (after is None or self.q_time > after), timeout):
                raise RuntimeError(f'{self.arm}: no fresh measured joints')
            return self.q.copy(), self.q_time

    def wait(self, predicate, timeout: float) -> bool:
        with self._cv:
            return self.clock.wait(self._cv, predicate, timeout)

    # --- GPIO requests -------------------------------------------------------------------------
    guard = float(names.GUARD_NONE)   # the guard level last written: every safety command carries it (write)

    def write(self, **values: float) -> None:
        """One safety command. The safety controller (GpioCommandController, 100 Hz) applies only the newest message
        it holds at each update, so each one carries the guard level asked for last: a disarm with a land request
        sent on its heels was never applied, and the snag guard left armed latched the next wake at rest
        (2026-10-23)."""
        if "guard_mode" in values:
            self.guard = float(values["guard_mode"])
        values = {"guard_mode": self.guard, **values}
        msg = DynamicInterfaceGroupValues()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.interface_groups = [names.safety_gpio(self.arm)]
        msg.interface_values = [InterfaceValue(interface_names=list(values),
                                               values=[float(v) for v in values.values()])]
        self.gpio.publish(msg)

    def _request_id(self) -> int:
        self._request += 1
        return self._request

    def _requested(self, command: str, ack: str, timeout: float) -> tuple[bool, int]:
        """Write command=N (repeated every 0.1 s; the same N is one request) until ack == N."""
        n = self._request_id()
        deadline = self.clock.now() + timeout
        while self.clock.now() < deadline:
            self.write(**{command: n})
            if self.wait(lambda: self.safety.get(ack) == n, timeout=0.1):
                return True, n
        return False, n

    def unlatch(self) -> tuple[bool, str]:
        """Hold goal at the measured joints, then unlatch = N and wait <= 1 s for the ack. Never a step."""
        if not self.latched:
            return True, "not latched"
        if not self.estop_ok:
            return False, "the e-stop is pressed or silent"
        q = self.measured()
        status, _, text = self.execute(self.hold(q), watch_latch=False)
        if status != "ok":
            return False, f"hold goal: {text}"
        acked, _ = self._requested("unlatch", "unlatch_ack", 1.0)
        if not acked:
            return False, "the driver did not acknowledge unlatch within 1 s"
        if self.latched:
            return False, f"the driver kept the latch ({self.latch_text()})"
        return True, "unlatched"

    def land(self, budget_s: float, on_text=None) -> tuple[bool, str]:
        """land = N, then wait for the driver's verified landing (landed = 1)."""
        # The driver commands the landing, and a woken arm holds where it rests: the last goal's carriage seeded the
        # first goal after a wake 2 mm off the landed one, a step JTC took in 10 ms and the driver latched as
        # over_velocity (fake driver, 2026-10-04).
        self.last_command = None
        acked, _ = self._requested("land", "land_ack", 2.0)
        if not acked:
            return False, "the driver did not acknowledge land (mock hardware has no landing)"
        if on_text:
            on_text("landing")
        if self.wait(lambda: self.flag("landed"), timeout=budget_s + 5.0):
            return True, "landed and idle"
        return False, f"not landed after {budget_s + 5.0:.0f} s ({self.latch_text()})"

    # --- trajectories --------------------------------------------------------------------------
    def hold(self, q: np.ndarray, seconds: float = 0.1):
        """One-point hold trajectory at q, zero velocity."""
        from tatbot_motion import Trajectory

        return Trajectory(self.joint_names, np.array([seconds]), q[None, :], np.zeros((1, 7)),
                          np.full((1, 3), np.nan), np.array([np.nan]), np.zeros(1, dtype=np.uint8))

    def goal(self, traj, first: int = 0) -> FollowJointTrajectory.Goal:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.joint_names)
        for t, q, qd in zip(*goal_points(traj, first), strict=True):   # JTC starts from the current state
            goal.trajectory.points.append(JointTrajectoryPoint(
                positions=[float(v) for v in q], velocities=[float(v) for v in qd], time_from_start=_duration(t)))
        return goal

    def _send(self, traj, stamp=None, first: int = 0, start: float | None = None) -> tuple[threading.Event, dict]:
        """Publish the goal on /tatbot/goals and send it; the box fills with handle, t0 and result. `stamp` (ROS time;
        `start` on self.clock) starts it there, not on arrival; `first` drops the knots before it."""
        done, box = threading.Event(), {"accepted": threading.Event()}

        def on_result(future):
            box["result"] = future.result()
            done.set()

        def on_accepted(future):
            box["handle"], box["t0"] = future.result(), self.clock.now() if start is None else start
            box["accepted"].set()
            if box["handle"].accepted:
                box["handle"].get_result_async().add_done_callback(on_result)
                if box.get('cancel'):
                    box['handle'].cancel_goal_async()
            else:
                done.set()

        goal = self.goal(traj, first)
        if self.goal_log is not None:
            self.goal_log.append(self.op_id, goal_tag(self.op_id, traj, first), self.clock.now(), start,
                                 *goal_points(traj, first))
        if self.goals is not None:
            goal.trajectory.header.stamp = (stamp if stamp is not None else self.node.get_clock().now()).to_msg()
            goal.trajectory.header.frame_id = goal_tag(self.op_id, traj, first)
            self.goals.publish(goal.trajectory)
            goal.trajectory.header.frame_id = ""
            goal.trajectory.header.stamp = Time()
        if stamp is not None:
            goal.trajectory.header.stamp = stamp.to_msg()
        self.jtc.send_goal_async(goal).add_done_callback(on_accepted)
        return done, box

    def _retarget(self, retarget, elapsed: float, stamp, start: float, running: tuple) -> tuple:
        """Offer the running goal to `retarget`: a replacement on its time base, from the knot before now, becomes
        the running (done, box) once JTC accepts it. JTC preempts only then, so a refused one leaves the running
        goal as it was and ends the offers (retarget None). Returns (done, box, retarget)."""
        new = retarget(elapsed, self.active)
        if new is None:
            return (*running, retarget)
        done, box = self._send(new, stamp, max(int(np.searchsorted(new.t, elapsed - ports.REPLACE_BACK_S)) - 1, 0), start)
        if box["accepted"].wait(1.0) and box["handle"].accepted:
            self.active = new
            return done, box, retarget
        self.node.get_logger().warning(f"{self.arm}: the controller refused a replacement goal; the running one goes on")
        return (*running, None)

    def execute(self, traj, *, watch_latch: bool = True, on_tick=None, should_stop=None, poll: float = 0.02,
                retarget=None):
        """Run one FollowJointTrajectory goal. Returns (status, sample index reached, text) with status
        ok | failed | latched | cancelled. A latch or a stop request cancels the goal. A landed arm is sent
        no goal (failed): its driver would ignore it, and JTC report it a success. Nor is a way that brings
        the arm onto itself (self.clearance). `retarget(elapsed_s, active)` may return a trajectory on
        the same knot times to replace the running one (a pen trim, at most pen.trim.limit_m off the way the
        clearance passed); self.active is the one JTC ran last."""
        return execute(self, traj, lambda: self._run(traj, watch_latch, on_tick, should_stop, poll, retarget),
                       self.node.get_logger().warning)

    def _run(self, traj, watch_latch: bool, on_tick, should_stop, poll: float, retarget=None):
        self.last_command = None  # unless it completes, JTC holds the measured pose: seed from measured
        if not self.jtc.wait_for_server(timeout_sec=5.0):
            return "failed", 0, f"{names.follow_joint_trajectory(self.arm)} is not available"
        stamp, start = (self.node.get_clock().now() + RosDuration(seconds=ports.STAMP_AHEAD_S),
                        self.clock.now() + ports.STAMP_AHEAD_S) if retarget else (None, None)
        done, box = self._send(traj, stamp, start=start)
        index, sent_at = 0, self.clock.now()
        while not done.wait(poll):
            if 't0' in box:
                elapsed = self.clock.now() - box["t0"]
                index = min(int(np.searchsorted(traj.t, elapsed)), len(traj.t) - 1)
                if on_tick:
                    on_tick(index)
                if retarget is not None and elapsed > 2 * ports.REPLACE_BACK_S and not box.get("cancel"):
                    done, box, retarget = self._retarget(retarget, elapsed, stamp, start, (done, box))
            deadline = box['t0'] + float(traj.t[-1]) + 30.0 if 't0' in box else sent_at + 5.0
            stop = stop_reason(self, watch_latch, should_stop)
            if stop or self.clock.now() > deadline:
                box['cancel'] = True
                if (handle := box.get('handle')) is not None and handle.accepted:
                    handle.cancel_goal_async()
                done.wait(2.0)
                text = {"latched": self.latch_text(), "cancelled": "stop requested"}.get(
                    stop, "no result 30 s past the end" if 't0' in box else "no goal acceptance within 5 s")
                return stop or "failed", index, text
        return self._outcome(box, self.active, index, watch_latch)

    def _outcome(self, box: dict, traj, index: int, watch_latch: bool):
        handle, result = box.get("handle"), box.get("result")
        if handle is None or not handle.accepted:
            return "failed", 0, "the trajectory controller rejected the goal"
        if watch_latch and self.latched:
            return "latched", index, self.latch_text()
        code = result.result.error_code
        if code != FollowJointTrajectory.Result.SUCCESSFUL or result.status != GoalStatus.STATUS_SUCCEEDED:
            return "failed", index, f"JTC error {code}: {result.result.error_string}"
        self.last_command = np.asarray(traj.q[-1]).copy()
        return "ok", len(traj.t) - 1, "done"
