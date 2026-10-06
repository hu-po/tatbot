"""One arm's ROS I/O: measured joints, the <arm>_safety GPIO states and requests, and the JTC action.

The request protocols are ros/README.md "Continue and land, never a step": a hold goal at the measured
joints, then `unlatch = N` and a wait of at most 1 s for `unlatch_ack == N`; `land = N` for the
driver's landing. On mock hardware nothing latches and nothing acks: an unlatched arm needs no unlatch.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration, Time
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import DynamicInterfaceGroupValues, InterfaceValue
from rclpy.action import ActionClient
from rclpy.duration import Duration as RosDuration
from tatbot_description import names
from trajectory_msgs.msg import JointTrajectoryPoint

from tatbot_session import rules

LATCH_TEXT = {
    names.LATCH_NONE: "none", names.LATCH_ESTOP: "estop", names.LATCH_ESTOP_STALE: "estop_stale",
    names.LATCH_CARRIAGE_CONTACT: "carriage_contact", names.LATCH_STALL: "stall",
    names.LATCH_OVER_VELOCITY: "over_velocity", names.LATCH_GUARD_TIP_LAG: "guard_tip_lag",
    names.LATCH_GUARD_PROBE: "guard_probe", names.LATCH_CONTROLLER_ERROR: "controller_error",
    names.LATCH_DEACTIVATED: "deactivated", names.LATCH_STEP_REFUSED: "step_refused",
}

# A change in any of these publishes SafetyState at once; ages and RT periods ride the 20 Hz timer.
CHANGES = ("estop_ok", "latched", "latch_reason", "guard_mode", "guard_tripped", "unlatch_ack", "land_ack",
           "landing", "landed", "controller_error", "probe_triggered")


def _duration(seconds: float) -> Duration:
    sec = int(math.floor(seconds))
    return Duration(sec=sec, nanosec=int(round((seconds - sec) * 1e9)) % 1_000_000_000)


WAY_WAIT_S = 30.0   # a way blocked by the other arm's goal in flight waits this long for it to end
# A goal that may be replaced mid-way (a pen trim) starts on an explicit stamp this far ahead and its replacements
# share it, each starting this far in the past: JTC runs one from the knot that holds now, never from the measured
# pose, which lags the command ~70 ms while tracking (README section 1).
STAMP_AHEAD_S = REPLACE_BACK_S = 0.05


def landing_way(q, staged, n: int = 20) -> np.ndarray:
    """The driver's landing in joint space (tatbot_hardware ArmCore::start_landing): from q to the staged joints
    with q's carriage, then to the sleep pose, every joint at zero but the wrist roll and the carriage at the
    staged ones; `n` rows each leg."""
    q, staged = np.asarray(q, float), np.asarray(staged, float)
    mid = staged.copy()
    mid[6] = q[6]
    sleep = np.zeros(7)
    sleep[5], sleep[6] = staged[5], staged[6]
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.vstack([q + t * (mid - q), mid + t * (sleep - mid)])


class ArmIO:
    def __init__(self, node, arm: str, callback_group, on_safety_change=None):
        self.node, self.arm = node, arm
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
        self._request = int(time.time())
        self._on_change = on_safety_change
        self.goals = None  # a JointTrajectory publisher: every goal sent, for the run bag
        # (arm, joint rows) -> why the way must not be sent, or None: the other arm's clearance (the node's
        # tatbot_motion.collision.Guard); None when nothing places the two arms in one frame. The node's also
        # reserves a way it lets through, and says whether only the other arm's way in flight blocks it:
        # (why, waiting may clear it); `release` ends the reservation and `hold_way` reserves a landing's.
        self.clearance = None
        self.release = None
        self.hold_way = None
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
            self.q, self.q_time = q, time.monotonic()
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
            before, self.safety, self.safety_time = self.safety, state, time.monotonic()
            self._cv.notify_all()
        if self._on_change is not None and any(before.get(k) != state.get(k) for k in CHANGES):
            self._on_change(self.arm, before, state)

    def flag(self, name: str, default: float = 0.0) -> bool:
        value = self.safety.get(name, default)
        return bool(value) and not math.isnan(value) and value >= 0.5

    @property
    def latched(self) -> bool:
        return self.flag("latched")

    @property
    def estop_ok(self) -> bool:
        return self.flag("estop_ok", 1.0)

    @property
    def latch_reason(self) -> int:
        value = self.safety.get("latch_reason", 0.0)
        return 0 if math.isnan(value) else int(value)

    def latch_text(self) -> str:
        return LATCH_TEXT.get(self.latch_reason, str(self.latch_reason))

    def trip_joints(self) -> np.ndarray | None:
        q = np.array([self.safety.get(f"trip_q{i}", math.nan) for i in range(7)])
        return None if np.isnan(q).any() else q

    def measured(self, timeout: float = 5.0) -> np.ndarray:
        with self._cv:
            if not self._cv.wait_for(lambda: self.q is not None, timeout=timeout):
                raise RuntimeError(f"{self.arm}: no /joint_states for {self.joint_names[0]}")
            return self.q.copy()

    def seed(self) -> np.ndarray:
        """Measured joints with the carriage from the last command (its encoder rests 10-20 um off)."""
        q = self.measured()
        if self.last_command is not None:
            q[6] = self.last_command[6]
        return q

    def observed(self, *, after=None, timeout=1.0):
        """Fresh measured joints (under 1 s old, and newer than `after`) and their monotonic receipt stamp."""
        with self._cv:
            if not self._cv.wait_for(lambda: self.q is not None and time.monotonic()-self.q_time <= 1.0
                                    and (after is None or self.q_time > after), timeout=timeout):
                raise RuntimeError(f'{self.arm}: no fresh measured joints')
            return self.q.copy(), self.q_time

    def wait(self, predicate, timeout: float) -> bool:
        with self._cv:
            return self._cv.wait_for(predicate, timeout=timeout)

    # --- GPIO requests -------------------------------------------------------------------------
    def write(self, **values: float) -> None:
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
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
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

    def land(self, budget_s: float, on_text=None, staged=None) -> tuple[bool, str]:
        """land = N, then wait for the driver's verified landing (landed = 1). With `staged` (the driver's staged
        joints) the landing's way is reserved against the other arm's next goals while it runs: the driver sweeps
        in joint space from here to the staged pose, then to the sleep pose (landing_way)."""
        # The driver commands the landing, and a woken arm holds where it rests: the last goal's carriage seeded the
        # first goal after a wake 2 mm off the landed one, a step JTC took in 10 ms and the driver latched as
        # over_velocity (fake driver, 2026-10-04).
        self.last_command = None
        if staged is not None and self.hold_way is not None and self.q is not None:
            self.hold_way(self.arm, landing_way(self.q, staged))
        try:
            acked, _ = self._requested("land", "land_ack", 2.0)
            if not acked:
                return False, "the driver did not acknowledge land (mock hardware has no landing)"
            if on_text:
                on_text("landing")
            if self.wait(lambda: self.flag("landed"), timeout=budget_s + 5.0):
                return True, "landed and idle"
            return False, f"not landed after {budget_s + 5.0:.0f} s ({self.latch_text()})"
        finally:
            if self.release is not None:
                self.release(self.arm)

    # --- trajectories --------------------------------------------------------------------------
    def hold(self, q: np.ndarray, seconds: float = 0.1):
        """One-point hold trajectory at q, zero velocity."""
        from tatbot_motion import Trajectory

        return Trajectory(self.joint_names, np.array([seconds]), q[None, :], np.zeros((1, 7)),
                          np.full((1, 3), np.nan), np.array([np.nan]), np.zeros(1, dtype=np.uint8))

    def goal(self, traj, first: int = 0) -> FollowJointTrajectory.Goal:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.joint_names)
        for t, q, qd in zip(traj.t[first:], traj.q[first:], traj.qd[first:], strict=True):
            if t <= 0.0:
                continue  # the seed knot; JTC starts from the current state
            goal.trajectory.points.append(JointTrajectoryPoint(
                positions=[float(v) for v in q], velocities=[float(v) for v in qd], time_from_start=_duration(t)))
        return goal

    def _send(self, traj, stamp=None, first: int = 0, start: float | None = None) -> tuple[threading.Event, dict]:
        """Publish the goal on /tatbot/goals and send it; the box fills with handle, t0 and result. `stamp` (ROS time;
        `start` monotonic) starts it there, not on arrival; `first` drops the knots before it."""
        done, box = threading.Event(), {"accepted": threading.Event()}

        def on_result(future):
            box["result"] = future.result()
            done.set()

        def on_accepted(future):
            box["handle"], box["t0"] = future.result(), time.monotonic() if start is None else start
            box["accepted"].set()
            if box["handle"].accepted:
                box["handle"].get_result_async().add_done_callback(on_result)
                if box.get('cancel'):
                    box['handle'].cancel_goal_async()
            else:
                done.set()

        goal = self.goal(traj, first)
        if self.goals is not None:
            goal.trajectory.header.stamp = (stamp if stamp is not None else self.node.get_clock().now()).to_msg()
            self.goals.publish(goal.trajectory)
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
        done, box = self._send(new, stamp, max(int(np.searchsorted(new.t, elapsed - REPLACE_BACK_S)) - 1, 0), start)
        if box["accepted"].wait(1.0) and box["handle"].accepted:
            self.active = new
            return done, box, retarget
        self.node.get_logger().warning(f"{self.arm}: the controller refused a replacement goal; the running one goes on")
        return (*running, None)

    def _stop(self, watch_latch: bool, should_stop) -> str | None:
        if watch_latch and self.latched:
            return "latched"
        return "cancelled" if should_stop and should_stop() else None

    def execute(self, traj, *, watch_latch: bool = True, on_tick=None, should_stop=None, poll: float = 0.02,
                retarget=None):
        """Run one FollowJointTrajectory goal. Returns (status, sample index reached, text) with status
        ok | failed | latched | cancelled. A latch or a stop request cancels the goal. A landed arm is sent
        no goal (failed): its driver would ignore it, and JTC report it a success. Nor is a way that brings
        the arm too near the other one (self.clearance). `retarget(elapsed_s, active)` may return a trajectory on
        the same knot times to replace the running one (a pen trim, at most pen.trim.limit_m off the way the
        clearance passed); self.active is the one JTC ran last."""
        self.active = traj
        if self.flag("landed"):
            return "failed", 0, rules.landed_refusal(self.arm)
        why = self._way_clear(traj, watch_latch, should_stop)
        if why is not None:
            self.node.get_logger().warning(why)
            return "failed", 0, why
        try:
            return self._run(traj, watch_latch, on_tick, should_stop, poll, retarget)
        finally:
            if self.release is not None:
                self.release(self.arm)

    def _way_clear(self, traj, watch_latch: bool, should_stop) -> str | None:
        """Why the way must not be sent (self.clearance), or None, reserved. One blocked only by the other arm's
        way in flight waits up to WAY_WAIT_S for it to end."""
        if self.clearance is None:
            return None
        deadline = time.monotonic() + WAY_WAIT_S
        while True:
            answer = self.clearance(self.arm, traj.q)
            why, waits = answer if isinstance(answer, tuple) else (answer, False)
            if why is None or not waits or time.monotonic() > deadline or self._stop(watch_latch, should_stop):
                return why
            time.sleep(0.2)

    def _run(self, traj, watch_latch: bool, on_tick, should_stop, poll: float, retarget=None):
        self.last_command = None  # unless it completes, JTC holds the measured pose: seed from measured
        if not self.jtc.wait_for_server(timeout_sec=5.0):
            return "failed", 0, f"{names.follow_joint_trajectory(self.arm)} is not available"
        stamp, start = (self.node.get_clock().now() + RosDuration(seconds=STAMP_AHEAD_S),
                        time.monotonic() + STAMP_AHEAD_S) if retarget else (None, None)
        done, box = self._send(traj, stamp, start=start)
        index, sent_at = 0, time.monotonic()
        while not done.wait(poll):
            if 't0' in box:
                elapsed = time.monotonic() - box["t0"]
                index = min(int(np.searchsorted(traj.t, elapsed)), len(traj.t) - 1)
                if on_tick:
                    on_tick(index)
                if retarget is not None and elapsed > 2 * REPLACE_BACK_S and not box.get("cancel"):
                    done, box, retarget = self._retarget(retarget, elapsed, stamp, start, (done, box))
            deadline = box['t0'] + float(traj.t[-1]) + 30.0 if 't0' in box else sent_at + 5.0
            stop = self._stop(watch_latch, should_stop)
            if stop or time.monotonic() > deadline:
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
