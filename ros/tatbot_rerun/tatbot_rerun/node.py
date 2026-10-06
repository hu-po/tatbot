"""`ros2 run tatbot_rerun rerun_bridge`: the stack into the fleet Rerun viewer.

It logs, in `world`, under draw/ros/: each arm's measured tip path (tf world -> <arm>/tcp, from
/joint_states through robot_state_publisher), the page outline and clear center (/tatbot/page and tf
world -> page/<pattern_id>), the program's strokes at each run start (the run's program.json), events
and safety changes as text. The stream opens only through $TATBOT_REPO/scripts/lib/tatbot_rerun.py
start() (the one rr.init; scripts/check rerun-caps), capped at stack.yaml rerun.max_hz; it never
spawns a viewer.

Parameters (strings; '' = stack.yaml / the fleet viewer): config, arms, max_hz, connect, output.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tatbot_bridge import stack
from tatbot_description import names, repo_root
from tatbot_interfaces.msg import Event, Page, SafetyState
from tf2_ros import Buffer, TransformException, TransformListener

from tatbot_rerun import shapes

ROOT = "draw/ros"
PAGE_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


def rerun_helper():
    """scripts/lib/tatbot_rerun.py from the repo the stack runs from, loaded by path: this package has
    the same name, so a plain import would find itself."""
    import importlib.util

    lib = repo_root() / "scripts" / "lib"
    if str(lib) not in sys.path:
        sys.path.insert(0, str(lib))  # the helper imports tatbot_cli
    spec = importlib.util.spec_from_file_location("tatbot_rerun_helper", lib / "tatbot_rerun.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RerunBridge(Node):
    def __init__(self):
        super().__init__(names.RERUN_NODE)
        param = stack.string_parameters(self, ("config", "arms", "max_hz", "connect", "output"))
        self.stack = stack.load(param["config"])
        self.arms = stack.arms(param["arms"], self.stack)
        hz = float(param["max_hz"] or self.stack.get("rerun", {}).get("max_hz", 30))
        page_cfg = self.stack["page"]
        self.size, self.clear = page_cfg["size_m"], page_cfg["clear_m"]
        self.log_root = Path(self._log_root())
        connect = param["connect"] or None
        if not connect and not param["output"]:
            nodes = stack.fleet()
            connect = nodes.rerun_proxy(nodes.load(repo_root()))
        self.tr = rerun_helper()
        self.rr = self.tr.start("ros", connect=connect, output=param["output"] or None,
                                recording_id=self.tr.recording_id("ros-stack"))
        self.limiter = self.tr.RateLimiter(hz)
        self.paths = {arm: shapes.TipPath() for arm in self.arms}
        self.page_frame = None
        self.safety = {}
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_subscription(JointState, names.JOINT_STATES_TOPIC, self._on_joints, 10)
        self.create_subscription(Page, names.PAGE_TOPIC, self._on_page, PAGE_QOS)
        self.create_subscription(Event, names.EVENTS_TOPIC, self._on_event, 50)
        self.create_subscription(SafetyState, names.SAFETY_TOPIC, self._on_safety, 10)
        self.get_logger().info(f"rerun: {connect or param['output']} at <= {hz:g} Hz under {ROOT}/")

    def _log_root(self) -> str:
        return os.path.expanduser(os.environ.get("TATBOT_LOG_ROOT", "~/tatbot-logs"))

    def _now(self):
        self.tr.set_capture_time(time.time_ns())

    def _lookup(self, target: str):
        try:
            t = self.buffer.lookup_transform(names.WORLD, target, Time())
        except TransformException:
            return None
        tr, q = t.transform.translation, t.transform.rotation
        return shapes.matrix([tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w])

    def _on_joints(self, _msg):
        if not self.limiter.ready():
            return
        self._now()
        for arm, path in self.paths.items():
            tcp = self._lookup(names.tcp_frame(arm))
            if tcp is not None and path.add(tcp[:3, 3]):
                self.rr.log(f"{ROOT}/{arm}/tip/{path.index:05d}", self.rr.LineStrips3D([path.array()], radii=0.0003))
                self.rr.log(f"{ROOT}/{arm}/tcp", self.rr.Points3D([tcp[:3, 3]], radii=0.002))

    def _on_page(self, msg: Page):
        self.page_frame = names.page_frame(msg.pattern_id)
        world_from_page = self._lookup(self.page_frame)
        if world_from_page is None:
            return
        self._now()
        lost = msg.source == Page.SOURCE_LOST
        self.rr.log(f"{ROOT}/page", self.rr.LineStrips3D(shapes.page_outlines(world_from_page, self.size, self.clear),
                                                          colors=[(160, 160, 160) if lost else (40, 160, 220)],
                                                          radii=0.0004))

    def _on_event(self, msg: Event):
        self._now()
        self.rr.log(f"{ROOT}/events", self.rr.TextLog(f"[{msg.arm or '-'}] {msg.kind} {msg.op_id} {msg.text}".strip()))
        if msg.kind == Event.KIND_RUN_START and msg.run_id:
            for arm, path in self.paths.items():
                path.clear()
                self.rr.log(f"{ROOT}/{arm}/tip", self.rr.Clear(recursive=True))
            self._log_program(self.log_root / names.RUN_WORKFLOW_DRAW / msg.run_id / "program.json")

    def _log_program(self, path: Path):
        world_from_page = self._lookup(self.page_frame) if self.page_frame else None
        if world_from_page is None or not path.is_file():
            return
        strokes = shapes.program_strokes(json.loads(path.read_text()), world_from_page)
        self.rr.log(f"{ROOT}/strokes", self.rr.LineStrips3D(strokes, colors=[(230, 120, 30)], radii=0.00025))

    def _on_safety(self, msg: SafetyState):
        state = (msg.estop_ok, msg.latched, msg.latch_reason, msg.guard_tripped, msg.landing, msg.landed)
        if self.safety.get(msg.arm) == state:
            return
        self.safety[msg.arm] = state
        self._now()
        self.rr.log(f"{ROOT}/{msg.arm}/safety", self.rr.TextLog(
            f"estop_ok={int(msg.estop_ok)} latched={int(msg.latched)} reason={msg.latch_reason} "
            f"guard_tripped={int(msg.guard_tripped)} landing={int(msg.landing)} landed={int(msg.landed)}",
            level="WARN" if msg.latched else "INFO"))


def main(argv=None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    try:
        node = RerunBridge()
    except Exception as error:  # noqa: BLE001  (no SDK or no sink: say so; the stack runs on without it)
        print(f"tatbot_rerun: {error}", file=sys.stderr)
        rclpy.try_shutdown()
        return 1
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0

