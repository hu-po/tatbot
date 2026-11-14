"""`ros2 run tatbot_bridge bridge`: stencild page poses from the tatbot zenoh bus into ROS.

page_source stencil: its own zenoh client session to the tatbot bus router
(tatbot_cli.nodes.bus_endpoint(address="lan")) subscribes tatbot/tracking/target/<pattern_id>
(tatbot.target-pose/1) and publishes /tatbot/page (Page, transient local, depth 1) plus the
tf world -> page/<pattern_id> at each measured sample. Of several sheets of the print in view it
publishes the one page.PageChooser says is the page. A lost sample, or one with no sheet that is the
page, republishes the last measured pose with SOURCE_LOST and no tf. It only reads the bus. While the
bus router is unreachable it retries every 2 s. Setting `page_at` to `<arm>,<x_mm>,<y_mm>` (`ros2 param
set`, or the session client's --sheet-at) makes the sheet nearest that point in <arm>/base_link the page
from the turn it is read there on; the bridge records it (page-chosen-<arm>.json in the log root) so a
restart starts there.
page_source fixed: stack.yaml page.fixed.<arm> as a static tf <arm>/base_link -> page/fixed_<arm>
and a Page per arm at 1 Hz. The arm transforms are the description's (robot_state_publisher),
never this node's.

Parameters (strings; '' = stack.yaml): config, page_source, arms, pattern_id, bus_endpoint; page_at ('' = none).
"""
from __future__ import annotations

import collections
import json
import math
import sys
import threading

import numpy as np
import rclpy
import rclpy.executors
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tatbot_description import names
from tatbot_interfaces.msg import Page
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from tatbot_bridge import page, stack

PAGE_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
LOG_BASE_EVERY_S = 10.0
BUS_RETRY_S = 2.0


def stamp(ns: int) -> Time:
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def transform(parent: str, child: str, matrix, when: Time) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp, msg.header.frame_id, msg.child_frame_id = when, parent, child
    t = msg.transform
    t.translation.x, t.translation.y, t.translation.z = (float(v) for v in matrix[:3, 3])
    t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w = page.quaternion(matrix[:3, :3])
    return msg


def page_msg(*, pattern_id, parent, matrix, when, source, sample=None) -> Page:
    msg = Page(pattern_id=pattern_id, source=source)
    msg.pose.header.stamp, msg.pose.header.frame_id = when, parent
    p = msg.pose.pose.pose
    p.position.x, p.position.y, p.position.z = (float(v) for v in matrix[:3, 3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = page.quaternion(matrix[:3, :3])
    if sample is None:
        msg.pose.pose.covariance = page.covariance(0.0, 0.0)
        return msg
    msg.print_id, msg.identity_verified = sample["print_id"], sample["identity_verified"]
    msg.translation_sigma_m, msg.rotation_sigma_rad = sample["translation_sigma_m"], sample["rotation_sigma_rad"]
    msg.pose.pose.covariance = page.covariance(sample["translation_sigma_m"], sample["rotation_sigma_rad"])
    msg.support_json = json.dumps(sample["support"], sort_keys=True)
    return msg


class Bridge(Node):
    def __init__(self):
        super().__init__(names.BRIDGE_NODE)
        param = stack.string_parameters(self, ("config", "page_source", "arms", "pattern_id", "bus_endpoint"))
        self.stack = stack.load(param["config"])
        cfg = self.stack["page"]
        self.source = param["page_source"] or cfg["source"]
        self.arms = stack.arms(param["arms"], self.stack)
        self.pattern_id = param["pattern_id"] or cfg["pattern_id"]
        self.pub = self.create_publisher(Page, names.PAGE_TOPIC, PAGE_QOS)
        self.bus = None
        self.declare_parameter("page_at", "", ParameterDescriptor(description="<arm>,<x_mm>,<y_mm>: choose the sheet "
                                                                              "nearest this point as the page"))
        self.add_on_set_parameters_callback(self._on_parameters)
        if self.source == "fixed":
            self._start_fixed(cfg)
        else:
            self._start_stencil(param["bus_endpoint"] or stack.bus_endpoint())

    # --- fixed ------------------------------------------------------------------------------
    def _start_fixed(self, cfg):
        self.static = StaticTransformBroadcaster(self)
        self.fixed = {}
        for arm in self.arms:
            nominal = cfg["fixed"][arm]
            self.fixed[arm] = page.from_xyz_rpy(nominal["xyz"], nominal.get("rpy", [0, 0, 0]))
        now = self.get_clock().now().to_msg()
        self.static.sendTransform([transform(names.base_frame(a), names.page_frame(names.fixed_pattern_id(a)), m, now)
                                   for a, m in self.fixed.items()])
        self.create_timer(1.0, self._publish_fixed)
        self._publish_fixed()
        self.get_logger().info(f"page source fixed: {', '.join(names.page_frame(names.fixed_pattern_id(a)) for a in self.fixed)}")

    def _publish_fixed(self):
        now = self.get_clock().now().to_msg()
        for arm, matrix in self.fixed.items():
            self.pub.publish(page_msg(pattern_id=names.fixed_pattern_id(arm), parent=names.base_frame(arm),
                                      matrix=matrix, when=now, source=Page.SOURCE_FIXED))

    def _on_parameters(self, params):
        for param in params:
            if param.name == "page_at" and param.value:
                try:
                    self._choose_at(str(param.value))
                except ValueError as error:
                    return SetParametersResult(successful=False, reason=str(error))
        return SetParametersResult(successful=True)

    def _choose_at(self, text: str) -> None:
        if self.source == "fixed":
            raise ValueError("page.source fixed: the page is stack.yaml's page.fixed, with no sheet to choose")
        arm, xy = page.parse_at(text)
        if self.registrations.get(arm) is None:
            known = [a for a, matrix in self.registrations.items() if matrix is not None]
            raise ValueError(f"page_at: {arm} is not an arm this bridge has a registration for "
                             f"({', '.join(known) or 'none'})")
        self.chooser.choose_at(xy, np.linalg.inv(self.registrations[arm]))
        self.chosen_arm = arm
        self.get_logger().info(f"page: the sheet nearest ({xy[0] * 1e3:+.0f}, {xy[1] * 1e3:+.0f}) mm in "
                               f"{names.base_frame(arm)}, from the turn it is read there on")

    def _record_choice(self, chosen) -> None:
        arm = self.chosen_arm
        bfp = page.base_from_page(self.registrations[arm], chosen["world_from_target"])
        self.get_logger().info(f"page chosen: the sheet at {np.round(bfp[:3, 3] * 1000, 1).tolist()} mm in "
                               f"{names.base_frame(arm)}")
        try:
            stack.chosen_path(arm).write_text(json.dumps({"arm": arm, "camera": bfp.tolist(),
                                                          "stamp_ns": chosen["stamp_ns"]}) + "\n")
        except OSError as error:
            self.get_logger().warning(f"the choice is not recorded ({error}): a restart starts where it stood before")

    # --- stencil ----------------------------------------------------------------------------
    def _start_stencil(self, endpoint):
        self.tf = TransformBroadcaster(self)
        self.frame = names.page_frame(self.pattern_id)
        self.samples = collections.deque(maxlen=64)
        self.lock = threading.Lock()
        self.last = None            # the last measured sample
        self.lost = False
        self.logged_at = self.refused_at = -math.inf
        self.registrations = {a: stack.world_from_arm_base(self.stack, a) for a in self.arms}
        self.chooser = stack.page_chooser(self.stack, self.registrations)
        self.chosen_arm = None
        self.endpoint, self.bus_error = endpoint, None
        self.ee_pubs = {a: self.create_publisher(PoseWithCovarianceStamped, names.ee_topic(a), 10) for a in self.arms}
        self.ee_samples = collections.deque(maxlen=64)
        self.ee_subs = []
        self.create_timer(0.02, self._drain)
        self.opener = self.create_timer(BUS_RETRY_S, self._open_bus)
        self._open_bus()

    def _open_bus(self):
        """Open the bus session; while its router is unreachable (rig link down at boot, router restart),
        retry every BUS_RETRY_S and log the failure once."""
        try:
            bus = stack.open_bus(self.endpoint)
        except Exception as error:  # noqa: BLE001  (zenoh raises its own ZError)
            if str(error) != self.bus_error:
                self.get_logger().warning(f"tatbot bus {self.endpoint} unreachable ({error}); retrying every "
                                          f"{BUS_RETRY_S:.0f} s")
                self.bus_error = str(error)
            return
        self.opener.cancel()
        self.bus, self.sub = bus, bus.declare_subscriber(page.topic(self.pattern_id), self._on_sample)
        self.ee_subs = [bus.declare_subscriber(f"tatbot/tracking/ee/{arm}", lambda smp, arm=arm: self._on_ee(arm, smp))
                        for arm in self.arms if self.registrations.get(arm) is not None]
        self.get_logger().info(f"page source stencil: {page.topic(self.pattern_id)} on {self.endpoint} -> "
                               f"{names.PAGE_TOPIC}, tf {names.WORLD} -> {self.frame}")

    def _on_sample(self, sample):  # zenoh's thread: only queue
        with self.lock:
            self.samples.append(sample.payload.to_bytes())

    def _on_ee(self, arm, sample):  # zenoh's thread: only queue
        with self.lock:
            self.ee_samples.append((arm, sample.payload.to_bytes()))

    def _drain_ee(self, batch):
        """EE fiducial estimates (visiond tatbot.tracking-pose/1, status measured) -> the arm's base frame,
        through the same adopted registration as the page (read only on the tatbot bus)."""
        for arm, data in batch:
            try:
                est = json.loads(data)["payload"]["estimate"]
            except (ValueError, KeyError, TypeError):
                continue
            if est.get("status") != "measured" or est.get("world_from_ee") is None:
                continue
            bfe = np.linalg.inv(self.registrations[arm]) @ np.asarray(est["world_from_ee"], dtype=float)
            msg = PoseWithCovarianceStamped()
            msg.header.stamp, msg.header.frame_id = stamp(int(est["timestamp_ns"])), names.base_frame(arm)
            p, q = bfe[:3, 3], page.quaternion(bfe[:3, :3])
            msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = (float(v) for v in p)
            (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w) = (float(v) for v in q)
            var = (float(est.get("translation_sigma_mm") or 3.0) * 1e-3) ** 2
            msg.pose.covariance = [var if i in (0, 7, 14) else 0.0 for i in range(36)]
            self.ee_pubs[arm].publish(msg)

    def _drain(self):
        with self.lock:
            batch, self.samples = list(self.samples), collections.deque(maxlen=64)
            ee, self.ee_samples = list(self.ee_samples), collections.deque(maxlen=64)
        self._drain_ee(ee)
        for data in batch:
            sample = page.parse(data, self.pattern_id)
            if sample is None:
                continue
            chosen = self._page_of(sample) if sample["source"] == "measured" else None
            if chosen is not None:
                self._measured(chosen)
            elif self.last is not None:
                if not self.lost:
                    self.get_logger().info("page lost: the last measured pose stands")
                self.lost = True
                wfp = page.world_from_page(self.last["world_from_target"])
                self.pub.publish(page_msg(pattern_id=self.pattern_id, parent=names.WORLD, matrix=wfp,
                                          when=stamp(self.last["stamp_ns"]), source=Page.SOURCE_LOST, sample=self.last))

    def _page_of(self, sample):
        """The sheet of a measured sample that is the page, how far the others stand from it in its support; None, and
        a warning every LOG_BASE_EVERY_S, when none is."""
        sheets, choosing = page.copies(sample), self.chooser.wanted is not None
        chosen = self.chooser.choose(sheets)
        if chosen is not None:
            if choosing:
                self._record_choice(chosen)
            return {**chosen, "support": {**chosen["support"], "other_copies_m": page.others_m(chosen, sheets)}}
        t = self.get_clock().now().nanoseconds / 1e9
        if t - self.refused_at >= LOG_BASE_EVERY_S:
            self.refused_at = t
            self.get_logger().warning(f"{len(sheets)} sheet(s) of the print in view, none the page: "
                                      f"{self.chooser.why_none()}. `tatbot ros page` lists them, --sheet-at X,Y "
                                      "chooses one")
        return None

    def _measured(self, sample):
        when = stamp(sample["stamp_ns"])
        wfp = page.world_from_page(sample["world_from_target"])
        self.pub.publish(page_msg(pattern_id=self.pattern_id, parent=names.WORLD, matrix=wfp, when=when,
                                  source=Page.SOURCE_MEASURED, sample=sample))
        self.tf.sendTransform(transform(names.WORLD, self.frame, wfp, when))
        if self.lost or self.last is None:
            self.get_logger().info("page measured")
        self.last, self.lost = sample, False
        t = self.get_clock().now().nanoseconds / 1e9
        if t - self.logged_at >= LOG_BASE_EVERY_S:
            self.logged_at = t
            for arm, wfb in self.registrations.items():
                if wfb is not None:
                    bfp = page.base_from_page(wfb, sample["world_from_target"])
                    self.get_logger().info(
                        f"page in {names.base_frame(arm)}: xyz_mm {np.round(bfp[:3, 3] * 1000, 1).tolist()} "
                        f"rpy_deg {np.round(page.rpy_deg(bfp[:3, :3]), 1).tolist()} "
                        f"sigma {sample['translation_sigma_m'] * 1000:.2f} mm")

    def close(self):
        if self.bus is not None:
            self.sub.undeclare()
            for sub in self.ee_subs:
                sub.undeclare()
            self.bus.close()


def main(argv=None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = Bridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
