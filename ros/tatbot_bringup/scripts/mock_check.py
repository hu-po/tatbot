#!/usr/bin/env python3
"""Bring the stack up on mock hardware in an isolated graph and check the control layer.

    python3 mock_check.py [--domain 52] [--port 7452 | 0] [--arms right] [-- LAUNCH_ARG:=VALUE ...]

It starts its own rmw_zenohd on 127.0.0.1:<port> (0 = any free port) and `ros2 launch
tatbot_bringup stack.launch.py hardware:=mock page_source:=fixed touch:=false` against it, then:
  1. every spawned controller is active (controller_manager/list_controllers);
  2. /joint_states arrives near the 400 Hz control rate;
  3. <arm>_safety_controller/gpio_states carries the 21 safety states in order, estop_ok 1;
  4. a small FollowJointTrajectory goal (0.05 rad on joint_0 and 2 mm of carriage, and back)
     SUCCEEDS, the measured joints moved and end where it ended, and the arm is not latched
     afterwards (JTC SUCCESS alone is not execution: a latched driver holds while the goal
     "succeeds", every JTC tolerance being unchecked).
Extra NAME:=VALUE arguments override the launch's (later wins), e.g. `hardware:=fake` runs the same
checks on TatbotArm with its fake SDK. The session's runtime record always goes to the logs
directory, never beside $TATBOT_REPO, where a deployed workspace keeps its live stack's. It prints one
JSON line and stops everything it started (process groups, SIGINT then SIGKILL).
Exit 0 only when every check passed. Needs a sourced workspace holding tatbot_bringup.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RATE_WINDOW_S = 5.0
RATE_MIN_HZ = 360.0  # 400 Hz within 10 %: the ros2_control_node loop on a loaded dev host


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start(cmd: list[str], env: dict, log: Path) -> subprocess.Popen:
    stream = open(log, "w")  # noqa: SIM115 - closed with the process
    return subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def stop(proc: subprocess.Popen, sig: int = signal.SIGINT, grace_s: float = 10.0) -> None:
    """Signal the whole process group (ros2 launch wraps its nodes), then SIGKILL what is left.

    The nodes exit at once on SIGINT; rmw_zenoh 0.2.10 can then spend about 20 s timing out its own
    session close in the launch process and the router, which only costs time: the ros-stack run is
    finalized when the launch starts shutting down."""
    for signum, wait in ((sig, grace_s), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)


class Probe:
    """One rclpy node that watches /joint_states and asks the stack questions."""

    def __init__(self):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from tatbot_description import names

        self.rclpy, self.names = rclpy, names
        rclpy.init(args=[])  # the launch arguments on our argv are not ROS arguments
        self.node = rclpy.create_node("tatbot_mock_check")
        self.stamps: list[float] = []
        self.latest: dict = {}
        self.peak: dict = {}  # largest excursion of each joint from peak["from"] while a goal runs
        self.safety: dict = {}  # arm -> {state interface: value}, the latest gpio_states
        self.node.create_subscription(JointState, names.JOINT_STATES_TOPIC, self._on_joints, qos_profile_sensor_data)

    def _on_joints(self, msg):
        self.stamps.append(time.monotonic())
        self.latest.update(zip(msg.name, msg.position, strict=False))
        origin = self.peak.get("from", {})
        for name, q in zip(msg.name, msg.position, strict=False):
            if name in origin:
                self.peak[name] = max(self.peak.get(name, 0.0), abs(q - origin[name]))

    def spin_until(self, predicate, limit_s: float) -> bool:
        deadline = time.monotonic() + limit_s
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if predicate():
                return True
        return False

    def wait(self, future, limit_s: float):
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=limit_s)
        return future.result()

    def controllers(self, wanted: set, timeout_s: float) -> dict:
        from controller_manager_msgs.srv import ListControllers

        client = self.node.create_client(ListControllers, "/controller_manager/list_controllers")
        states: dict = {}

        def all_active():
            if client.service_is_ready():
                reply = self.wait(client.call_async(ListControllers.Request()), 2.0)
                states.update({c.name: c.state for c in reply.controller} if reply else {})
            return bool(states) and all(states.get(name) == "active" for name in wanted)

        return {"controllers_active": self.spin_until(all_active, timeout_s), "controllers": states}

    def joint_rate(self) -> dict:
        self.spin_until(lambda: len(self.stamps) > 10, 10.0)
        self.stamps.clear()
        self.spin_until(lambda: False, RATE_WINDOW_S)
        stamps = self.stamps
        rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        return {"joint_states_hz": round(rate, 1), "joint_states_ok": rate >= RATE_MIN_HZ}

    def gpio(self, arm: str) -> dict:
        from control_msgs.msg import DynamicInterfaceGroupValues

        names, got = self.names, {}

        def on_states(msg):
            got.update(msg=msg)
            if msg.interface_values:
                group = msg.interface_values[0]
                self.safety[arm] = dict(zip(group.interface_names, group.values, strict=False))

        self.node.create_subscription(DynamicInterfaceGroupValues, names.safety_states_topic(arm), on_states, 10)
        self.spin_until(lambda: "msg" in got, 10.0)
        groups = got["msg"].interface_values if "msg" in got else []
        ok = bool(groups) and tuple(groups[0].interface_names) == names.SAFETY_STATE_INTERFACES and \
            groups[0].values[names.SAFETY_STATE_INTERFACES.index("estop_ok")] == 1.0
        return {f"{arm}_gpio_states_ok": ok}

    def jtc(self, arm: str) -> dict:
        """0.05 rad on joint_0 and 2 mm of carriage over 1 s, then back over 1 s."""
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from trajectory_msgs.msg import JointTrajectoryPoint

        joints = self.names.joint_names(arm)
        start_q = [self.latest.get(j, float("nan")) for j in joints]
        target = [q + d for q, d in zip(start_q, (0.05, 0, 0, 0, 0, 0, 0.002), strict=True)]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joints)
        for q, t in ((target, 1), (start_q, 2)):
            point = JointTrajectoryPoint(positions=q, velocities=[0.0] * len(q))
            point.time_from_start.sec = t
            goal.trajectory.points.append(point)
        self.peak.clear()
        self.peak["from"] = dict(zip(joints, start_q, strict=True))
        action = ActionClient(self.node, FollowJointTrajectory, self.names.follow_joint_trajectory(arm))
        handle = self.wait(action.send_goal_async(goal), 5.0) if action.wait_for_server(timeout_sec=10.0) else None
        done = self.wait(handle.get_result_async(), 10.0) if handle is not None and handle.accepted else None
        code = done.result.error_code if done is not None else None
        self.spin_until(lambda: False, 0.3)
        error = max(abs(self.latest.get(j, float("nan")) - q) for j, q in zip(joints, start_q, strict=True))
        moved = (self.peak.get(joints[0], 0.0), self.peak.get(joints[6], 0.0))
        safety = self.safety.get(arm, {})
        latched = (safety.get("latched"), safety.get("latch_reason"))
        return {f"{arm}_jtc_error_code": code, f"{arm}_jtc_end_error": error,
                f"{arm}_jtc_peak": [round(v, 5) for v in moved], f"{arm}_latched": latched,
                f"{arm}_jtc_ok": (code == FollowJointTrajectory.Result.SUCCESSFUL and error < 1e-3
                                  and moved[0] > 0.045 and moved[1] > 0.0018 and latched == (0.0, 0.0))}

    def close(self):
        self.node.destroy_node()
        try:
            self.rclpy.shutdown()
        except RuntimeError as exc:  # rmw_zenoh can time out closing its session; the checks are done
            print(f"rclpy shutdown: {exc}", file=sys.stderr)


def checks(arms: list[str], timeout_s: float) -> dict:
    probe = Probe()
    names = probe.names
    wanted = {names.JOINT_STATE_BROADCASTER}
    wanted |= {n for arm in arms for n in (names.arm_controller(arm), names.safety_controller(arm))}
    try:
        result = probe.controllers(wanted, timeout_s)
        result.update(probe.joint_rate())
        for arm in arms:
            result.update(probe.gpio(arm))
            result.update(probe.jtc(arm))
    finally:
        probe.close()
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mock_check.py")
    parser.add_argument("--domain", type=int, default=52)
    parser.add_argument("--port", type=int, default=0, help="router port; 0 = a free port")
    parser.add_argument("--arms", default="right")
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds for the controllers to activate")
    parser.add_argument("--logs", default=None, help="directory for router.log and launch.log")
    parser.add_argument("launch_args", nargs="*", help="extra NAME:=VALUE launch arguments")
    ns = parser.parse_args(argv)
    port = ns.port or free_port()
    endpoint = f'["tcp/127.0.0.1:{port}"]'
    logs = Path(ns.logs or tempfile.mkdtemp(prefix="tatbot-mock-check-"))
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RMW_IMPLEMENTATION="rmw_zenoh_cpp", ROS_DOMAIN_ID=str(ns.domain))
    env["ZENOH_CONFIG_OVERRIDE"] = f"connect/endpoints={endpoint};scouting/multicast/enabled=false"
    router_env = dict(env, ZENOH_CONFIG_OVERRIDE=f"listen/endpoints={endpoint};scouting/multicast/enabled=false")
    from ament_index_python.packages import get_package_prefix

    zenohd = Path(get_package_prefix("rmw_zenoh_cpp")) / "lib" / "rmw_zenoh_cpp" / "rmw_zenohd"
    arms = [arm for arm in ns.arms.split(",") if arm]
    router = start([str(zenohd)], router_env, logs / "router.log")
    launch = None
    result: dict = {"domain": ns.domain, "port": port, "logs": str(logs)}
    try:
        time.sleep(1.0)
        # runtime_record last: no extra argument can point it at a live stack's record.
        launch = start(["ros2", "launch", "tatbot_bringup", "stack.launch.py", "hardware:=mock",
                        "estop_source:=none", "page_source:=fixed", "touch:=false", f"arms:={ns.arms}",
                        *ns.launch_args, f"runtime_record:={logs / 'runtime.json'}"],
                       env, logs / "launch.log")
        os.environ.update(env)  # rclpy reads the RMW, domain and zenoh endpoint from here
        result.update(checks(arms, ns.timeout))
    finally:
        if launch is not None:
            stop(launch)
        stop(router, signal.SIGTERM, 2.0)
    for line in (logs / "launch.log").read_text(errors="replace").splitlines():
        if "ros-stack run " in line:
            result["ros_stack_run"] = line.split("ros-stack run ", 1)[1].split(":", 1)[0]
            break
    result["ok"] = all(v for k, v in result.items() if k.endswith("_ok") or k == "controllers_active")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
