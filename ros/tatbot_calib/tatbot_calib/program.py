"""The fitted tool's tip across its axis, from opposed side touches on the station probe (ros/README.md section 7).

S0 takes the station fix and sets the floor; S1 searches for the stylus from the fix, bisects the ball's top and
centres opposed pairs on the tool's wall at its side height; S4 does the same at further upright yaws. Turning the
tool about its own axis turns a tip error with it while the ball stands still, so the fit gives the tip across the
axis (x, y in the tool mount). The tip along the axis, the axis's tilt and the joint offsets are held: from one ball
and upright yaws they trade off with each other and the tip (2026-10-01: fitted tips 64-81 mm long, joint offsets
of either sign), and page touches cancel the tip's length. The run writes candidate.json and report.md;
`tatbot ros calib apply --run ID` adopts the tip when asked.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from tatbot_motion.collision import PALETTE_BODY_MARGIN_M, PALETTE_PADS
from tatbot_motion.station import station_world_parts  # noqa: F401 -- existing calibration API

from tatbot_calib import halo, reach, solve, station
from tatbot_calib.tool import ToolRefusedError, fitted_tool

WORKFLOW = "ros-calib"
# The run ends landed: every Cartesian park pose over the ball drove joint 3 into its limit.
RETREAT_LIFT_M = 0.06
SIDE_RETRIES = (0.0, 0.0025, 0.0050)   # a missing side trigger expects the ball one cap farther
# The arm travels this far past first contact, unseen by its encoders, before a side touch trips: the trips of
# 2026-10-02 lay on circles of 1.76-1.89 mm about the stylus where the body and ball reach 3.03 mm. Planned without
# it, every pair's second touch stopped short of the stylus by about this much.
SIDE_GIVE_M = 0.0012
HOLE_MARGIN_M = 0.0003         # an edge pass stops with the ball this far clear of the hole's edge
# An edge pass stops this far inside the side radius. Near the ring's end the wall meets the ball too shallowly to
# trip: a pass that ended 0.8 mm inside it deflected the stylus with no trigger, which read triggered only after
# the pass had stopped (2026-09-28). A ball on the wall at the side touches' height trips before this.
EDGE_SHORT_M = 0.0004
# The floor: no way the run plans takes the tool's lowest point more than FLOOR_UNDER_TOP_M under the ball's top
# (operator, 2026-09-29, after a search pass 11 mm under the fix's ball came down beside the probe onto its base).
# Until S1 has touched the top, the top is the station fix's, good to FIX_HEIGHT_M in height (the D555's depth
# through the arm's registration, and the tool's nominal tip: 0.7 mm apart on the palette camera, 2026-10-02), and
# the floor stands that much higher.
FLOOR_UNDER_TOP_M = 0.010
FIX_HEIGHT_M = 0.002
# The search's end plane under the fix's ball, over the floor: the stylus (ball, shaft, cone: 22.6 mm over the
# collar) stands in the tool's way there for a fix up to 5 mm high and FIX_HEIGHT_M low. A tool whose side height is
# lower searches SEARCH_OVER_SIDE_M over it: a ballpoint's tip at its side height passes over the ball from a fix
# 2 mm high, and 1 mm higher it meets the tip or the body's end from a fix 2 mm off either way.
SEARCH_BELOW_M = 0.004
SEARCH_STEP_DOWN_M = 0.003     # a search that met nothing goes once more this much lower (Calibration.s1)
SEARCH_OVER_SIDE_M = 0.001
# A search starts where no ball within SEARCH_ACROSS_M of the estimate lies under the tool coming down onto the
# start: the stylus reaches the search's height, a fix FIX_HEIGHT_M low and the ball's radius over its end plane, and
# the tool's envelope is widest there (Calibration.search_standoff).
SEARCH_ACROSS_M = 0.012
# When the way from where the arm stands to over the ball is not planned, a staging pose on the way in: the tool
# upright, this far along from the tool toward the station (nearest the middle first), at the tool's height and this
# much over it, turned these degrees from the base heading. 2026-09-30: at v11 the joint move from rest swept pink's
# wrist 9-16 mm (over the bubbles) from the landed blue wrist, under the guard's 20 mm, while a stop at
# (200, 180, 170) mm kept 104 mm on the way in and 29-49 mm on over the ball.
STAGING_ALONG = (0.5, 0.3, 0.7)
STAGING_OVER_M = (0.0, 0.05)
STAGING_TURN_DEG = (-30.0, 0.0, 30.0, -60.0, 60.0)
SEARCH_MARGIN_M = 0.002
# A seed from the arm's last touch (search_seed) is used only within these of it: the tip's error across the axis
# turns with the heading (pink's 3RL met the ball 7 mm apart at headings 0 and -30, 2026-10-08).
SEED_TURN_DEG = 5.0
SEED_MAX_M = 0.02
# When nothing meets the search's line, further lines run this many side radii apart: a pass meets the stylus
# within about the tool's side radius across its line (the wall and the ball, the tool widening above them), so
# the lines leave no gap. The laser's reach covers the prior's 12 mm from one line; a cartridge's tube needs more.
SEARCH_LINE_SPACING = 1.8
TOP_OVER_M = 0.012             # the top search's upper end over the fix's top
# The top the edge passes leave room for: a pass trips only with more than EDGE_TRIGGER_M of the ball over its end
# plane where the ring meets it, and it stops with the ring's edge up to its planned distance plus the executor's lag
# (EDGE_LAG_M) off the ball's centre, so over a laser's ring the passes read the ball 0.6-1 mm under its top. Round 8
# (2026-09-29) read the blue top at 73.8-74.1 mm, its sides met the wall at the ball about a millimetre higher, and
# the floor set from the reading stood 11 mm under the ball's top. The floor stands under the highest top the passes
# allow.
EDGE_TRIGGER_M = 0.0004
EDGE_LAG_M = 0.0003
TOP_RESOLUTION_M = 0.0005
# S4's upright yaws, degrees about the base heading, farthest apart first: the tip across the axis turns with the
# yaw, so the wider the fan the better it is told from the ball. The run takes the first ATTITUDES_WANTED the
# executor plans from where the last left the arm (reach.choose_heading keeps the most it can). Tilted attitudes
# measured nothing the fit can use (2026-10-01: the axis came out 5 +- 6 deg).
DEFAULT_ATTITUDES = "-60/0,60/0,-30/0,30/0,-90/0,90/0,-45/0,45/0,-15/0,15/0"
ATTITUDES_WANTED = 3


class Rig:
    """The running stack's Touch action, the arm's joints and its safety state, for one arm."""

    def __init__(self, arm: str):
        import rclpy
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        from tatbot_description import names
        from tatbot_interfaces.action import Touch
        from tatbot_interfaces.msg import SafetyState

        rclpy.init()
        self.rclpy, self.Touch, self.arm, self.names = rclpy, Touch, arm, names
        self.node = rclpy.create_node("tatbot_calib")
        self.client = ActionClient(self.node, Touch, names.TOUCH_ACTION)
        self.safety, self.q = None, None
        self.node.create_subscription(SafetyState, names.SAFETY_TOPIC, self._on_safety, 10)
        self.node.create_subscription(JointState, "/joint_states", self._on_joints, 10)

    def _on_safety(self, msg):
        if msg.arm == self.arm:
            self.safety = msg

    def _on_joints(self, msg):
        index = {name: i for i, name in enumerate(msg.name)}
        wanted = self.names.joint_names(self.arm)
        if all(name in index for name in wanted):
            self.q = np.array([msg.position[index[name]] for name in wanted])

    def spin(self, seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def joints(self) -> np.ndarray:
        self.q = None
        while self.q is None:
            self.spin(0.05)
        return self.q.copy()

    def page(self, timeout_s: float = 5.0):
        """The bridge's latest Page (stencild's or the fixed one), or None within timeout_s."""
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from tatbot_interfaces.msg import Page

        got = []
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        sub = self.node.create_subscription(Page, self.names.PAGE_TOPIC, got.append, qos)
        end = time.monotonic() + timeout_s
        while not got and time.monotonic() < end:
            self.spin(0.1)
        self.node.destroy_subscription(sub)
        return got[-1] if got else None

    def goal(self, start: np.ndarray, *, move=False, direction=(0.0, 0.0, -1.0), prior=0.0, guard=None,
             speed=0.0, view=False) -> dict:
        """One Touch goal from `start` (the tcp in the arm base). The station's probe guard unless `guard` says
        otherwise: a view far from the station (a registration hold) moves with guard 0, none. `speed` the slow
        leg's, 0 for motion.yaml probe.slow_m_s. `view`: a hold by one joint move (MODE_VIEW, Calibration.go_to)."""
        from action_msgs.msg import GoalStatus
        from tatbot_session.geometry import matrix_quat

        modes = self.Touch.Goal
        goal = modes(mode=modes.MODE_VIEW if view else modes.MODE_MOVE if move else modes.MODE_SINGLE, arm=self.arm,
                     guard=modes.GUARD_PROBE if guard is None else int(guard), prior_distance_m=float(prior),
                     speed_m_s=float(speed))
        goal.start_pose.header.frame_id = self.names.base_frame(self.arm)
        pose = goal.start_pose.pose
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in start[:3, 3])
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = matrix_quat(start)
        goal.direction.x, goal.direction.y, goal.direction.z = (float(v) for v in direction)
        if not self.client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("no /tatbot/touch server: is the stack up?")
        sent = self.client.send_goal_async(goal)
        self.rclpy.spin_until_future_complete(self.node, sent)
        handle = sent.result()
        if not handle.accepted:   # a rejection carries no reason; a landed arm's safety state gives it
            from tatbot_session.rules import landed_refusal

            self.spin(0.3)
            if self.safety is not None and self.safety.landed:
                raise RuntimeError(f"the Touch goal was rejected: {landed_refusal(self.arm)}")
            raise RuntimeError("the Touch goal was rejected (is the arm busy?)")
        done = handle.get_result_async()
        self.rclpy.spin_until_future_complete(self.node, done)
        # an aborted goal (a landed arm, a refused plan) carries an empty result, which would read as a touch
        # that met nothing: a search would then call a refusal a miss
        if done.result().status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"the Touch goal ended with status {done.result().status}, not succeeded: "
                               f"{done.result().result.message!r}")
        r = done.result().result
        return {"tripped": bool(r.tripped), "joints": list(r.joints), "run_dir": r.run_dir, "message": r.message,
                "contact": [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in r.contact_poses]}

    def land(self) -> str:
        """The driver's verified landing (the Land action): rest pose, then idle."""
        from tatbot_interfaces.action import Land

        client = self.rclpy.action.ActionClient(self.node, Land, self.names.LAND_ACTION)
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("no /tatbot/land server")
        sent = client.send_goal_async(Land.Goal(arms=[self.arm]))
        self.rclpy.spin_until_future_complete(self.node, sent)
        done = sent.result().get_result_async()
        self.rclpy.spin_until_future_complete(self.node, done)
        return done.result().result.message

    def close(self):
        self.node.destroy_node()
        try:
            self.rclpy.try_shutdown()
        except RuntimeError as error:   # zenoh's close timing out right after a stack restart: the run's work is done
            print(f"# the ROS context did not close: {error}", flush=True)


class Calibration:
    def __init__(self, rig: Rig, run, repo: Path, arm: str, halo_tip: halo.Halo, heading_deg: float, kin):
        self.rig, self.run, self.repo, self.arm, self.halo = rig, run, repo, arm, halo_tip
        self.heading = math.radians(heading_deg)
        self.kin = kin
        self.contacts: list[solve.Contact] = []
        self.ball: np.ndarray | None = None
        self.side_radius = halo_tip.wall_radius_m + halo.BALL_RADIUS_M - SIDE_GIVE_M   # where a side touch trips
        self.lateral_cap_m = 0.0025    # motion.yaml probe.lateral_cap_m (run() reads it)
        self.clearance_m = 0.030       # motion.yaml probe.clearance_m (run() reads it)
        self.search_speed_m_s = 0.0    # motion.yaml probe.search_m_s (run() reads it); 0: probe.slow_m_s
        # the station's parts to keep out of: the fix's palette pose, carried with the ball's current estimate
        # (the parts are rigid about it, and the fix itself stood 10 mm off on 2026-09-28)
        self.station_pose: np.ndarray | None = None
        self.ball_in_palette, self.inkcaps = np.zeros(3), ()
        self.proven = False            # a trip in this run has shown the probe fires (the search rule)
        self.trips = 0                 # how many touches have tripped: a search counts its own
        self.seeded = False            # the ball estimate is a search_seed, good across, not in height
        self.stuck = False             # a ring's back-out failed (back_out): the run ends without its lift
        self.world_from_base: np.ndarray | None = None    # the arm's registration: its base in the collision scene's world
        # the posts the wrist must keep clear of (reach.overhead_post), and the wrist's bodies (reach.wrist_bodies)
        self.posts: list | tuple = ()
        self.bodies: list = []
        self.motion: dict | None = None   # motion.yaml (run() loads it): the executor's way to each goal
        self.floor_z = -math.inf       # the tool's lowest point stays over this (FLOOR_UNDER_TOP_M; _stages sets it)
        self.s1_side_radius: float | None = None     # S1's side pairs' half-span: the side radius until two yaws are fitted
        self.top_bound: float | None = None          # the highest the ball's top can stand after the last edge passes (top())
        self.edges: tuple | None = None              # the last edge passes (top()): the axis's start and stop, the lowest miss, any trip
        self.guard = None              # the stack's two-arm collision guard (tatbot_motion.collision), when it has one
        self.fitted: solve.Fit | None = None             # the fit so far (solve.Fit), which places the ball for the next yaw
        self.tip0: np.ndarray | None = None               # the installed tip in the tool mount (the tcp the executor plans with)
        self.mount_in_tcp: np.ndarray | None = None

    def wall_floor(self) -> float:
        """The tool's radius at least, for keep-out checks: the wall the side touches measure (the side radius and
        the give, less the ball's) where it is wider than the datasheet's (2026-09-28: the laser's 9.0 mm against 7.7)."""
        return self.side_radius + SIDE_GIVE_M - halo.BALL_RADIUS_M

    @property
    def search_below(self) -> float:
        return min(SEARCH_BELOW_M, self.halo.side_height_m + SEARCH_OVER_SIDE_M)

    def search_standoff(self) -> float:
        """How far short of its expected contact a search touch starts (SEARCH_ACROSS_M)."""
        reach = self.search_below + FIX_HEIGHT_M + halo.BALL_RADIUS_M
        return (SEARCH_ACROSS_M + self.halo.widest_m(reach) + halo.BALL_RADIUS_M + SEARCH_MARGIN_M
                - self.side_radius)

    def station_now(self) -> np.ndarray:
        """base <- palette_root: the fix's pose, carried with the ball's estimate once there is one (2026-09-30:
        anchored to no ball, every part stood at NaN and every body read -inf mm from it)."""
        if self.ball is None:
            return np.asarray(self.station_pose, float)
        return station.anchored(self.station_pose, self.ball, self.ball_in_palette)

    def zones(self) -> list:
        if self.station_pose is None:
            return []
        pose = self.station_now()
        return station.parts(pose, self.inkcaps, self.repo / "urdf" / "palette.urdf") + [("overhead_post", np.array([*post.xy, post.top_m]), post.radius_m,
                                                     True) for post in self.posts]

    def lowest(self, points, axis) -> float:
        """The tool's lowest point with its end-plane centre at points (not the first: where it is now) and its
        axis `axis`: the end's ring hangs its radius times the axis's tilt under the centre."""
        return min(float(p[2]) for p in points[1:]) - self.halo.wall_radius_m * float(np.hypot(axis[0], axis[1]))

    def floor_check(self, label: str, points, axis, stands_at: float | None = None) -> None:
        """Refuse a way whose tool goes under the floor; with `stands_at` (the tool's lowest point now), one that
        goes no deeper than where the tool already stands is let through: a back-out retracing its own pass,
        which blue's sag had put 0.1 mm under (2026-09-30, round 17)."""
        low = self.lowest(points, axis)
        floor = self.floor_z if stands_at is None else min(self.floor_z, stands_at - STANDS_SLACK_M)
        if low < floor:
            raise RuntimeError(f"{label}: its way takes the tool to z {low * 1000:.1f} mm, under the floor at "
                               f"{self.floor_z * 1000:.1f} mm ({FLOOR_UNDER_TOP_M * 1000:.0f} mm under the ball's top); "
                               "not sent")

    def way_check(self, label: str, poses, q=None) -> np.ndarray | None:
        """Refuse (RuntimeError) a goal that has no plan, or whose way as the executor plans it from joints q (the
        arm's now by default; reach.planned_path to poses[0], then each further pose: a touch's leg to its cap)
        takes the tool under the floor or the wrist within reach.CLEARANCE_M of a post. Returns the joints the way
        ends at. Without the planner (a pose-only simulation) the poses alone are checked against the floor."""
        from tatbot_session import ready

        if self.motion is None:
            self.floor_check(label, [None, *(pose[:3, 3] for pose in poses)], poses[0][:3, 2])
            return None
        from tatbot_motion.clik import PlanError

        try:
            rows = reach.planned_path(self.kin, self.motion, self.rig.joints() if q is None else q, poses[0])
            for pose in poses[1:]:
                rows = np.vstack([rows, ready.solve_ik_seeded(self.kin, pose, rows[-1])])
        except (ValueError, PlanError) as error:   # 2026-09-29: an IK stall here ended a run before any motion
            raise RuntimeError(f"{label}: no plan to it ({error}); not sent") from None
        self.rows_check(label, rows)
        return rows[-1]

    def view_way(self, label: str, pose: np.ndarray) -> str | None:
        """Why a hold at `pose` cannot go by one joint move from where the arm stands (Touch MODE_VIEW, the
        executor's ready.view_move), or None: the move dips under its lower end (ready.station_dip), fails
        rows_check, or brings the tool's envelope onto a station part or within clearance_m of the stylus: a column
        about the ball up to that far over its top, which the station's way rises to before it crosses."""
        from tatbot_session import ready

        if self.motion is None or self.station_pose is None or self.ball is None:
            return "nothing places the station"
        q = self.rig.joints()
        try:
            move = ready.view_move(self.kin, self.motion, q, pose, 100.0)
            if ready.station_dip(self.kin, move, q, move.q[-1]) is not None:
                return "the joint move dips"
            self.rows_check(label, move.q)
        except (ValueError, RuntimeError) as error:
            return str(error)
        reach_m = halo.BALL_RADIUS_M + self.clearance_m
        zones = self.zones() + [("probe's stylus", self.ball + [0.0, 0.0, reach_m], reach_m, True)]
        tcps = [self.kin.fk(row) for row in move.q]
        for a, b in zip(tcps[:-1], tcps[1:], strict=True):
            if (zone := halo.meets([a[:3, 3], b[:3, 3]], a[:3, 2], zones, self.halo, self.wall_floor())) is not None:
                return f"the tool would meet the {zone}"
        return None

    def go_to(self, label: str, pose: np.ndarray) -> None:
        """A hold's travel: one joint move (MODE_VIEW) where view_way lets it, else the station's way (MODE_MOVE) once
        way_check passes that (RuntimeError, nothing sent, when it does not). Each hold over the station's way rose
        30 mm turning, crossed and came down, three starts and stops in 11-23 s (2026-10-22)."""
        why = self.view_way(label, pose)
        if why is None:
            self.rig.goal(pose, view=True)
            return
        self.say(f"{label}: by the station's way ({why})")
        self.way_check(label, [pose])
        self.rig.goal(pose, move=True)

    def rows_check(self, label: str, rows) -> None:
        """way_check's refusals of rows of joints from where the arm stands (rows[0]); [q, q]: the arm standing at q."""
        tips = [self.kin.fk(row) for row in rows]
        self.floor_check(label, [tip[:3, 3] for tip in tips], tips[-1][:3, 2])
        why = self.guard.refusal(self.arm, rows) if self.guard is not None else None
        if why:   # the executor refuses it too (ArmIO.execute)
            raise RuntimeError(f"{label}: {why}")
        if self.posts:
            gap, row = reach.path_clearance(self.kin, rows, self.bodies, self.posts)
            if gap < reach.CLEARANCE_M:
                raise RuntimeError(f"{label}: the wrist would pass {gap * 1000:.0f} mm from the overhead camera's "
                                   f"post (at the tcp {np.round(self.kin.fk(row)[:3, 3], 3).tolist()}); not sent")
        self.palette_check(label, rows)

    def palette_check(self, label: str, rows) -> None:
        """Refuse a way on which a body of the arm other than its tool (links, gripper carriages with their rubber
        pads, the wrist cube, the D405) comes within PALETTE_BODY_MARGIN_M of a part of the station: the roof
        tag's box, the e-stop, the palette camera, the probe's body, the inkcaps, the overhead camera's post.
        The tool's own approach is its halo's check. 2026-09-30: the blue gripper's pad hung over the tag box
        while the laser touched the probe (the operator, from photos); nothing checked it. Without a scene or a
        registration nothing can place the arm's bodies, and that is said once."""
        parts = self.world_parts()
        if parts is None or self.guard is None:
            return
        scene = self.guard.scene
        gap = scene.parts_clearance(self.arm, rows, parts, PALETTE_PADS, max_step_rad=float(self.guard.step_rad))
        if gap.distance_m < PALETTE_BODY_MARGIN_M:
            raise RuntimeError(f"{label}: the arm's {gap.body_a.split('/')[-1]} would pass "
                               f"{gap.distance_m * 1000:.0f} mm from the station's {gap.body_b} (its pads counted), "
                               f"under {PALETTE_BODY_MARGIN_M * 1000:.0f}; not sent")

    def world_parts(self) -> list | None:
        """The station's parts (zones) in the collision scene's world (station_world_parts). None without a scene
        or a registration (S0 says which)."""
        if self.guard is None or self.world_from_base is None or self.station_pose is None:
            return None
        table = float(self.station_now()[2, 3])
        return station_world_parts(self.zones(), self.world_from_base, table)

    def station_clear(self, q) -> bool:
        """Whether the arm's bodies but its tool stand PALETTE_BODY_MARGIN_M clear of the station at joints q, its
        pads counted (reach._margin's `station`); true when nothing can place them."""
        parts = self.world_parts()
        if parts is None or self.guard is None:
            return True
        return self.guard.scene.parts_clearance(self.arm, [q], parts, PALETTE_PADS).distance_m >= PALETTE_BODY_MARGIN_M

    # --- records -----------------------------------------------------------------------------
    def log(self, name: str, row: dict):
        with (self.run.dir / name).open("a") as out:
            out.write(json.dumps(row) + "\n")

    def say(self, text: str):
        print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)
        self.log("events.jsonl", {"t": time.time(), "text": text})

    def heading_vector(self, extra_deg: float = 0.0) -> np.ndarray:
        a = self.heading + math.radians(extra_deg)
        return np.array([math.cos(a), math.sin(a), 0.0])

    # --- touches -----------------------------------------------------------------------------
    def touch(self, plan: halo.TouchPlan, group: str, retries) -> dict | None:
        """One planned touch, retried expecting the contact farther on a missing trigger. The contact's
        edge joints go to the fit (side), or only to the log (search, edge passes)."""
        for extra in retries:
            start = plan.start.copy()
            here = self.kin.fk(self.rig.joints())[:3, 3]
            zone = halo.meets(halo.route(here, plan, self.clearance_m, extra + 0.0025), start[:3, 2], self.zones(),
                              self.halo, self.wall_floor())
            if zone is not None:
                raise RuntimeError(f"{plan.label}: its route would meet the {zone}; not sent")
            end = start.copy()
            end[:3, 3] += (plan.prior_m + extra + self.lateral_cap_m) * plan.direction
            self.floor_check(plan.label, [here, start[:3, 3], end[:3, 3]], start[:3, 2])
            self.way_check(plan.label, [start, end])
            result = self.rig.goal(start, direction=plan.direction, prior=plan.prior_m + extra,
                                   speed=self.search_speed_m_s if plan.kind == "search" else 0.0)
            row = {"group": group, "kind": plan.kind, "label": plan.label, "extra_m": extra,
                   "start": start.tolist(), "direction": plan.direction.tolist(), "prior_m": plan.prior_m, **result}
            self.log("touches.jsonl", row)
            if result["message"].startswith(("error", "landed", "cancelled")):
                raise RuntimeError(f"{plan.label}: {result['message']}")
            if result["tripped"]:
                self.proven = True
                self.trips += 1
                if plan.kind == "side":
                    self.contacts.append(solve.Contact("side", np.array(result["joints"]), group, plan.label[-1]))
                return result
            if plan.kind in ("side", "search") and self.halo.hole_takes_ball:
                self.back_out(plan, group)
        self.say(f"{group} {plan.label}: no trigger within its retries")
        return None

    def back_out(self, plan: halo.TouchPlan, group: str) -> None:
        """After a ring's side or search pass that met nothing, straight back along its line to where it started,
        under the probe guard, before anything lifts: a ring can end such a pass over the stylus, pressing it
        without a trigger, and the next way's lift then drags it. On 2026-09-29 (round 10) the laser's ring ended
        a pass so; the lift tripped the probe with 3 Nm on the blue arm's joints, and the hold at the measured pose
        let that windup go, 9 mm in 0.12 s, the stylus into the ring's hole. A goal starting where the arm stands
        is not routed through the clearance plane (tatbot_session.ready.station_way). A back-out that trips or is
        refused leaves the arm where it is (self.stuck): the run ends without its lift."""
        here = self.kin.fk(self.rig.joints())
        travel = float(np.linalg.norm(here[:3, 3] - plan.start[:3, 3]))
        if travel < 0.001:
            return
        back = -np.asarray(plan.direction, float)
        end = here.copy()
        end[:3, 3] += (travel + self.lateral_cap_m) * back
        label = f"{plan.label} back"
        try:
            now = self.lowest([None, here[:3, 3]], here[:3, 2])
            self.floor_check(label, [here[:3, 3], end[:3, 3]], here[:3, 2], stands_at=now)
            self.way_check(label, [here, end])
            result = self.rig.goal(here, direction=back, prior=travel)
        except RuntimeError:
            self.stuck = True
            raise
        self.log("touches.jsonl", {"group": group, "kind": "back", "label": label, "extra_m": 0.0,
                                   "start": here.tolist(), "direction": back.tolist(), "prior_m": travel, **result})
        if result["tripped"] or result["message"].startswith(("error", "landed", "cancelled", "refused")):
            self.stuck = True
            raise RuntimeError(f"{label}: {result['message'] or 'tripped'}; the ring may be on the stylus, so the arm "
                               "stays where it is for the operator")

    def pair(self, rotation, group, heading, name: str, search: bool = False) -> tuple[float, float] | None:
        """An opposed pair of side touches along the pair's axis u, re-centring on each: a contact moving along
        +u puts the ball a side radius ahead of it, so the second touch is planned from the first's contact.
        Both: (the unbiased midpoint, the half-span); otherwise None, with the estimate re-centred on whatever
        tripped. A missed second touch reads the same as a probe that did not fire, and only the other pair
        tells them apart, so it never retries; nothing retries before a trip has shown the probe fires, and a
        search, which covers its whole uncertainty in one travel, never retries."""
        if self.ball is None:
            return None
        axis = rotation[:, 2]
        d, p = halo._across(axis, heading)
        u = {"p": p, "d": d}[name]
        shape = {"standoff_m": self.search_standoff(), "height_m": self.search_below, "kind": "search"} if search else {}
        hits = []
        for k, (label, sign) in enumerate(((f"+{name}", 1.0), (f"-{name}", -1.0))):
            plans = {plan.label: plan for plan in
                     halo.side_touches(self.ball, rotation, heading, self.halo, self.side_radius, **shape)}
            try:
                result = self.touch(plans[label], group,
                                    SIDE_RETRIES if k == 0 and self.proven and not search else (0.0,))
            except RuntimeError as error:
                # a search pass the checks refuse is not sent, and its opposite may be: the wrist keeps off the
                # camera's post from one side of a line and not the other (round 13, 2026-09-30, the -4.2 mm line)
                if not search or self.stuck or _latched(self):
                    raise
                self.say(f"{group} {label}: not sent ({error})")
                continue
            if result is None:
                continue
            at = float(np.array(result["contact"][0]) @ u)
            hits.append(at)
            self.ball = self.ball + (at + sign * self.side_radius - self.ball @ u) * u
        if len(hits) < 2:
            return None
        middle = (hits[0] + hits[1]) / 2.0
        self.ball = self.ball + (middle - self.ball @ u) * u
        return middle, (hits[1] - hits[0]) / 2.0

    def sides(self, rotation: np.ndarray, group: str, heading: np.ndarray,
              search: bool = False) -> tuple[np.ndarray, float, float] | None:
        """Centre using complete opposed pairs; retry a direction only after another trip moved the estimate."""
        if self.ball is None:
            return None
        half, tried = {}, {}
        for name in ("p", "d", "p", "d", "p", "d"):
            if name in half or (name in tried and np.allclose(tried[name], self.ball, atol=1e-6)):
                continue
            tried[name] = self.ball.copy()
            got = self.pair(rotation, group, heading, name, search)
            if got is not None:
                half[name] = got[1]
            if len(half) == 2:
                return self.ball.copy(), half["d"], half["p"]
        return None

    def search(self, rotation: np.ndarray, heading: np.ndarray, group: str = "S1 search", lines: bool = True
               ) -> tuple[np.ndarray, float, float] | None:
        """S1's search, and an S4 yaw's when the fit has not placed the ball: sides() through the estimate. A pass
        meets the stylus only within the
        tool's reach across its line, a few millimetres for a cartridge's tube: when none of this search's passes
        trips, opposed pairs run along lines offset across, nearer lines first, as far as the prior's 12 mm, and the
        first that trips re-centres the estimate for sides() to centre the stylus from there."""
        if self.ball is None:
            return None
        trips = self.trips
        found = self.sides(rotation, group, heading, search=True)
        if found is not None or self.trips > trips or not lines:
            return found
        _, across = halo._across(rotation[:, 2], heading)
        origin, step = self.ball.copy(), SEARCH_LINE_SPACING * (self.side_radius + SIDE_GIVE_M)
        for k in (1, -1, 2, -2, 3, -3, 4, -4):
            if abs(k) * step > SEARCH_ACROSS_M:
                break
            self.ball = origin + k * step * across
            self.say(f"{group}: nothing met the search's line; one {k * step * 1000:+.1f} mm across it")
            self.pair(rotation, group, heading, "d", search=True)
            if self.trips > trips:
                return self.sides(rotation, group, heading, search=True)
        self.ball = origin
        return None

    # --- stages ------------------------------------------------------------------------------
    def approach(self, prior: np.ndarray) -> None:
        """Nothing when the executor plans a way from here to the tool upright over `prior`; else by way of a
        staging pose (STAGING_*) from which it does, both through way_check (the guard, posts, floor and station
        bodies as for any goal). RuntimeError when neither is planned: nothing moved."""
        over = halo.tool_pose(prior + [0.0, 0.0, reach.APPROACH_OVER_M], self.heading)
        try:
            self.way_check("over the ball", [over])
            return
        except RuntimeError as error:
            first = str(error)
        here = self.kin.fk(self.rig.joints())[:3, 3]
        refused = [first]
        for along in STAGING_ALONG:
            xy = here[:2] + along * (prior[:2] - here[:2])
            for rise in STAGING_OVER_M:   # not `over`: the way checked after the staging pose is over the ball's
                for turn in STAGING_TURN_DEG:
                    pose = halo.tool_pose(np.array([xy[0], xy[1], here[2] + rise]), self.heading + math.radians(turn))
                    label = f"staging ({xy[0]:.3f}, {xy[1]:.3f}, {here[2] + rise:.3f}) {math.degrees(self.heading) + turn:+.0f}"
                    try:
                        q = self.way_check(label, [pose])
                        self.way_check(f"{label} on over the ball", [over], q)
                    except RuntimeError as error:
                        refused.append(str(error))
                        continue
                    self.say(f"S1: no way over the ball from where the arm stands; moving by way of the {label}")
                    self.rig.goal(pose, move=True)
                    return
        raise RuntimeError("S1: no way over the ball from where the arm stands, nor by way of a staging pose: "
                           + "; ".join(refused[:3]) + (f"; and {len(refused) - 3} more" if len(refused) > 3 else ""))

    def clear(self, plans: list[halo.TouchPlan]) -> halo.TouchPlan:
        """The first plan whose way from here meets no keep-out zone."""
        here = self.kin.fk(self.rig.joints())[:3, 3]
        for plan in plans:
            if halo.meets(halo.route(here, plan, self.clearance_m), plan.start[:3, 2], self.zones(), self.halo,
                          self.wall_floor()) is None:
                return plan
        raise RuntimeError(f"every {plans[0].kind} plan's way meets a keep-out zone")

    def top(self, rotation: np.ndarray, heading: np.ndarray, met: float, over: float, group: str,
            resolution: float = TOP_RESOLUTION_M, half_spans: dict | None = None) -> float:
        """The ball's top over the centred estimate, by bisection of edge passes: at `met` the stylus stood above
        the end plane (a side or search touch met it there), and `over` is taken to pass over the top until
        tested. A pass stops with the ball under the ring, never the hole. Returns the ball centre's height.
        `half_spans` ({"d": .., "p": ..}) are this attitude's own: the side radius varies by 0.5 mm between
        directions and attitudes, and a pass stopped short of the global one read no trip down to 7 mm under
        the ball's top at the blue arm's yaw 0 (2026-09-28)."""
        if self.ball is None:
            raise RuntimeError("ball position is not initialized")
        side = self.clear(halo.side_touches(self.ball, rotation, heading, self.halo, self.side_radius))
        reach = (half_spans or {}).get(side.label[1], self.side_radius)
        # with its cap, a pass ends with the axis short of where a ball on the wall trips in its direction, and
        # never nearer than a hole's radius the ball could enter, the ball's and a margin. A bore narrower than
        # the ball keeps it out: the pass goes on until the ball is under the ring's middle.
        stop = (max(reach - EDGE_SHORT_M, self.halo.hole_radius_m + halo.BALL_RADIUS_M + HOLE_MARGIN_M)
                if self.halo.hole_takes_ball else self.halo.rim_radius_m)
        prior = min(side.prior_m, self.side_radius + side.prior_m - self.lateral_cap_m - stop)
        tested, tripped = False, False
        if not self.halo.edges_trip:
            # every pass would read a miss, and the bisection would settle on `met` (2026-10-05..12): settle there
            # without the passes' 3 minutes. The side touches that centred the stylus set its height, as they did.
            over, tested = met + TOP_RESOLUTION_M, True
        while over - met > TOP_RESOLUTION_M:
            z = (met + over) / 2.0
            if self.touch(halo.edge_pass(side, z, prior), f"{group} top", (0.0,)) is not None:
                met, tripped = z, True
            else:
                over, tested = z, True
        if not tested and self.touch(halo.edge_pass(side, over, prior), f"{group} top", (0.0,)) is not None:
            raise RuntimeError(f"the stylus still stands at z {over:.4f} m, over the searched range: the prior is "
                               "too far off to trust its height")
        start = side.start[:3, 3]   # the axis stops at the pass's cap, the executor's lag short of it
        self.edges = (start, start + (prior + self.lateral_cap_m - EDGE_LAG_M) * side.direction, over, tripped)
        self.say(f"{group}: the ball's top between {met * 1000:.1f} and {over * 1000:.1f} mm (base z)")
        self.bound_top(self.ball)
        return (met + over) / 2.0 - halo.BALL_RADIUS_M

    def bound_top(self, ball: np.ndarray) -> None:
        """The highest the ball's top can stand, its centre at `ball`, after the last edge passes (top()): over their
        lowest miss by the trigger's depth and the ball's fall from its top to the nearest the ring came on its way. A
        ring that never came within the ball's radius of its centre could not meet it, and its top is unmeasured. The
        floor stands FLOOR_UNDER_TOP_M under it."""
        start, end, over, _ = self.edges
        path = (end - start)[:2]
        along = float(np.clip((ball - start)[:2] @ path / (path @ path), 0.0, 1.0))
        near = max(0.0, float(np.linalg.norm((ball - start)[:2] - along * path)) - self.halo.wall_radius_m)
        if near >= halo.BALL_RADIUS_M:
            raise RuntimeError(f"the edge passes' ring stayed {near * 1000:.1f} mm off the ball centred at "
                               f"({ball[0]:.4f}, {ball[1]:.4f}) m: they cannot reach over it, so its top is unmeasured")
        self.top_bound = over + EDGE_TRIGGER_M + halo.BALL_RADIUS_M - math.sqrt(halo.BALL_RADIUS_M ** 2 - near ** 2)
        self.floor_z = self.top_bound - FLOOR_UNDER_TOP_M
        self.say(f"the ball's top at most {self.top_bound * 1000:.1f} mm; the floor is now z {self.floor_z * 1000:.1f} "
                 f"mm, {FLOOR_UNDER_TOP_M * 1000:.0f} mm under it")

    def search_heading(self) -> np.ndarray:
        """The search pairs' heading: the base heading turned (by under 45 degrees) until the palette camera
        stands midway between two of the four starts, which the search's wider starts would otherwise near."""
        if self.ball is None:
            return self.heading_vector()
        camera = next((centre for name, centre, *_ in self.zones() if name == "palette_camera"), None)
        if camera is None:
            return self.heading_vector()
        toward = math.atan2(camera[1] - self.ball[1], camera[0] - self.ball[0]) + math.pi / 4.0
        a = toward + round((self.heading - toward) / (math.pi / 2.0)) * math.pi / 2.0
        return np.array([math.cos(a), math.sin(a), 0.0])

    def step_down(self, across: np.ndarray) -> bool:
        """The search's estimate and the floor SEARCH_STEP_DOWN_M lower, back at `across` (where the search started:
        a pass that tripped moved the estimate). The D555 reads the station high: 3-5 mm through the 10-12
        registrations, and 6.2 mm on one, where every pass went over the ball's top. A search that centred nothing
        went over the stylus, or grazed its top with the needles, which trip only now and then (10-13: 7 of 24
        passes, never an opposed pair, at one height): either way the top stands at least that much under the fix's,
        and the floor goes down with it, still more than 10 mm under the top."""
        if self.stuck or _latched(self):
            return False
        self.ball[:2] = across[:2]
        self.ball[2] -= SEARCH_STEP_DOWN_M
        self.floor_z -= SEARCH_STEP_DOWN_M
        self.say(f"S1: nothing met; searching again {SEARCH_STEP_DOWN_M * 1000:.0f} mm lower")
        return True

    def find_stylus(self, rotation: np.ndarray):
        """S1's search from the estimate, and once more SEARCH_STEP_DOWN_M lower when nothing met it. From a seed,
        good across to a millimetre or two (search_seed), one pair at each height comes first: what a seed misses is
        the height (10-13: four lines across, 7 min, before the lower height found it). sides()'s result, or None."""
        across = self.ball.copy()
        if self.seeded:
            found = self.sides(rotation, "S1 search", self.search_heading(), search=True)
            if found is not None:
                return found
            if self.step_down(across):
                return self.sides(rotation, "S1 search", self.search_heading(), search=True) or self.search(
                    rotation, self.search_heading())
        found = self.search(rotation, self.search_heading())
        if found is None and not self.seeded and self.step_down(across):
            found = self.search(rotation, self.search_heading())
        return found

    def s1(self) -> bool:
        """The station found from a prior good to 12 mm across and 11 mm in height: search, top, sides."""
        if self.ball is None:
            return False
        rotation = halo.tool_pose(np.zeros(3), self.heading)[:3, :3]
        heading = self.heading_vector()
        prior_top = float(self.ball[2]) + halo.BALL_RADIUS_M
        found = self.find_stylus(rotation)
        if found is None:
            self.say("S1: the search found no stylus within 12 mm of the prior")
            return False
        _, rd, rp = found
        self.say(f"S1: stylus centred at ({self.ball[0]:.4f}, {self.ball[1]:.4f}) m, half-spans "
                 f"{rd * 1000:.2f}/{rp * 1000:.2f} mm (the tool on the stylus {self.search_below * 1000:.1f} mm up; "
                 f"its side radius at the ball {self.side_radius * 1000:.2f})")
        met = float(self.ball[2]) - self.search_below
        self.ball[2] = self.top(rotation, heading, met, prior_top + TOP_OVER_M, "S1")
        if not self.attitude(0.0, "S1"):
            return False
        if not self.edges[3]:
            # No edge pass met the stylus, so the bisection settled on the search's height: the top, when the prior
            # stood a search depth high (2026-10-07, the 3RL at heading 0: the side pairs centred the stylus 0.9 mm
            # from where the passes stopped); a search depth under it when the passes went by the ball, along a
            # search centre off the stylus. Where the side pairs centre it tells the two apart.
            self.bound_top(self.ball)
        return True

    def placed(self, rotation: np.ndarray) -> np.ndarray | None:
        """The ball as this yaw's tcp must see it, from the fit of two or more yaws: the fitted ball less the fitted
        tip's offset from the installed one, turned with the tool mount. None before then."""
        if self.fitted is None or self.mount_in_tcp is None or self.tip0 is None or len({c.group for c in self.contacts}) < 2:
            return None
        mount = rotation @ self.mount_in_tcp
        return self.fitted.ball - mount @ (self.fitted.tip - self.tip0)

    def attitude(self, yaw_deg: float, group: str) -> bool:
        """One upright yaw: the ball placed by the fit so far (else searched, as when the placed ball meets
        nothing), then opposed pairs on the tool's wall at its side height, along fixed base directions so each
        pair's give is alike between yaws. False when the pairs do not both trip."""
        if self.ball is None:
            return False
        rotation = halo.tool_pose(np.zeros(3), self.heading + math.radians(yaw_deg))[:3, :3]
        heading = self.heading_vector()
        placed = None if group == "S1" else self.placed(rotation)
        if placed is not None:
            self.ball = placed
        elif group != "S1":
            self.search(rotation, heading, f"{group} search")
        found = self.sides(rotation, group, heading)
        if found is None and placed is not None:
            # one search through the placed ball, not its lines across: on 2026-10-17 the 3RL's +30 and +45 placed
            # balls met nothing, and their lines out to 12 mm met nothing either (36 touches, 18 min)
            self.say(f"{group}: the placed ball met nothing; searching its line")
            self.search(rotation, heading, f"{group} search", lines=False)
            found = self.sides(rotation, group, heading)
        if found is None:
            return False
        ball, rd, rp = found
        self.say(f"{group}: ball through this tip at ({ball[0]:.4f}, {ball[1]:.4f}, {ball[2]:.4f}), "
                 f"side radius {rd * 1000:.2f}/{rp * 1000:.2f} mm")
        if group == "S1":
            self.s1_side_radius = (rd + rp) / 2.0
        return True


def _frame_at(kin, arm: str):
    return lambda q: kin.frame(q, f"{arm}/tool_mount")


def _report(run_dir: Path, fitted: solve.Fit, held: dict, moved: dict, tip0, n_touches: int) -> str:
    tip, sig = fitted.tip, fitted.sigma.get("tip_m", [math.nan] * 3)
    delta = (tip - tip0) * 1000
    worst = max(moved.values(), default=math.nan) * 1000
    lines = [f"# Probe tip calibration ({run_dir.name})", "",
             f"- contacts: {len(fitted.residuals_m)} side contacts of {n_touches} touches; rms "
             + ", ".join(f"{k} {v * 1000:.3f} mm" for k, v in fitted.rms_m.items()),
             f"- tip across the axis in the tool mount: ({tip[0] * 1000:.3f}, {tip[1] * 1000:.3f}) mm "
             f"+- ({sig[0] * 1000:.2f}, {sig[1] * 1000:.2f}); installed ({tip0[0] * 1000:.3f}, {tip0[1] * 1000:.3f}): "
             f"moved ({delta[0]:+.2f}, {delta[1]:+.2f}) mm. Along the axis it stays {tip0[2] * 1000:.3f} mm",
             f"- ball in the base: {np.round(fitted.ball * 1000, 2).tolist()} mm; side radius "
             f"{fitted.side_radius * 1000:.3f} mm",
             "- each yaw left out: its contacts miss by (rms, mm) " + ", ".join(
                 f"{g} {v * 1000:.3f}" for g, v in held.items()),
             f"- each yaw left out moves the tip by up to {worst:.2f} mm: " + ", ".join(
                 f"{g} {v * 1000:.2f}" for g, v in moved.items()),
             "", f"Adopt with `tatbot ros calib apply --arm <arm> --run {run_dir.name}`."]
    return "\n".join(lines) + "\n"



def _adopt_touch(cal, touch, live: Path):
    """The touch dips aim by: the camera's ball height of the touch it replaces carried when it speaks for this one
    (station.carry_height; the 2026-10-15 re-touch dropped it and a dip went 5.3 mm deep), the replaced file kept
    beside it."""
    previous = station.load_touch(cal.arm, live) if live.is_file() else None
    touch, note = station.carry_height(touch, previous)
    if note:
        cal.say(f"S1: {note}")
    if previous is not None:
        live.with_name(f"{live.name}.replaced-{cal.run.dir.name}").write_text(live.read_text())
    touch.write(cal.run.dir / "station-touch.json")
    cal.say(f"S1: dips aim by this touch from now on ({touch.write(live)})")
    return touch

def run(args) -> int:
    from tatbot_description import repo_root, robot_description
    from tatbot_motion import Kinematics, load_motion

    repo = repo_root(None)
    try:
        tool, halo_tip = fitted_tool(repo, args.arm, args.tool)
    except ToolRefusedError as error:
        print(json.dumps({"ok": False, "message": str(error)}))
        return 3
    args.tool = tool.tool_id   # the candidate names the tool it measured
    sys.path.insert(0, str(repo / "scripts" / "lib"))
    import tatbot_runlog

    fix = station.StationFix.from_dict(json.loads(Path(args.station).expanduser().read_text()))
    station.require_fresh(fix, args.run_started, max_age_s=args.max_age_s)
    kin = Kinematics(robot_description(None, arms=(args.arm,)), args.arm)
    mount_from_tcp = np.linalg.inv(kin.frame(np.zeros(7), f"{args.arm}/tool_mount")) @ kin.fk(np.zeros(7))
    tip0 = mount_from_tcp[:3, 3]
    run_log = tatbot_runlog.init(WORKFLOW, meta={"arm": args.arm, "tool": tool.tool_id, "station": args.station},
                                 attach_logging=False, argv=["tatbot_calib", "calib", "run", "--arm", args.arm])
    motion = load_motion()
    posts = [reach.overhead_post(fix.detail["overhead_camera_m"])] if fix.detail.get("overhead_camera_m") else []
    guard, guard_text = _guard(repo, motion)
    base = world_from_base(repo, args.arm)
    station_ok = None
    if guard is not None and base is not None:   # the arm's bodies clear of the station at each heading's poses
        inkcaps = station.inkcap_rims(repo / "urdf" / "palette.urdf", repo / "config" / "palette.yaml",
                                    base_from_palette=fix.base_from_palette)
        parts = station_world_parts(station.parts(fix.base_from_palette, inkcaps, repo / "urdf" / "palette.urdf")
                                    + [("overhead_post", np.array([*p.xy, p.top_m]), p.radius_m, True) for p in posts],
                                    base, float(fix.base_from_palette[2, 3]))
        station_ok = lambda q: (guard.scene.parts_clearance(args.arm, [q], parts, PALETTE_PADS).distance_m  # noqa: E731
                                >= PALETTE_BODY_MARGIN_M)
    heading, unreached = ((args.heading_deg, []) if args.heading_deg is not None else
                          reach.choose_heading(kin, args.arm, fix.ball, args.attitudes,
                                               float(motion["clik"]["joint_limit_margin_rad"]), posts,
                                               station_ok))
    args.attitudes = [a for a in args.attitudes if tuple(a) not in unreached]
    rig = Rig(args.arm)
    cal = Calibration(rig, run_log, repo, args.arm, halo_tip, heading, kin)
    cal.tip0, cal.mount_in_tcp = tip0, mount_from_tcp[:3, :3].T
    cal.posts, cal.bodies, cal.motion, cal.guard = posts, reach.wrist_bodies(kin, args.arm), motion, guard
    cal.world_from_base = base
    cal.say(f"S0: {guard_text}")
    cal.say("S0: the arm's bodies but its tool keep " + (
        f"{PALETTE_BODY_MARGIN_M * 1000:.0f} mm from the station's parts, the gripper's pads "
        f"{PALETTE_PADS['carriage'] * 1000:.0f} mm past its carriages" if station_ok is not None
        else "unchecked against the station (no collision scene or registration)"))
    cal.say(f"S0: the base heading {heading:.0f} deg; out of the arm's reach there (yaw/tilt about it): "
            + (", ".join(f"{yaw:+.0f}/{tilt:+.0f}" for yaw, tilt in unreached) or "none"))
    probe = motion["probe"]
    cal.lateral_cap_m, cal.clearance_m = float(probe["lateral_cap_m"]), float(probe["clearance_m"])
    cal.search_speed_m_s = float(probe.get("search_m_s", 0.0))
    code = 1
    try:
        code = _stages(cal, fix, args)
    finally:
        if not _latched(cal) and not cal.stuck:   # landed; a latched arm, or one whose ring may be on the stylus
            # (Calibration.back_out), stays where it holds
            retreat(rig, kin)
        rig.close()
        run_log.finalize(code, status="ok" if code == 0 else "fail")
    print(json.dumps({"ok": code == 0, "run_dir": str(run_log.dir)}))
    return code


def _guard(repo: Path, motion: dict):
    """The stack's two-arm collision guard, built as the stack builds it (its stack.yaml's registrations), so that the
    goals this run plans are the ones the executor sends: (the guard or None, a line saying which)."""
    try:
        from tatbot_motion.collision import Guard

        from tatbot_calib.cli import _stack
    except ImportError as error:
        return None, f"arm clearance not checked here ({error})"
    return Guard.from_stack(repo, _stack(repo), motion)


# The station's parts keep this far from every body of the arm but its tool (Calibration.palette_check), and the
# gripper's carriages are taken larger by their rubber pads, which the URDF's carriage meshes leave out: 30 mm until
# measured (2026-09-30: at the operator's photo of the pad over the tag box, the bare carriage read 25 mm from it).
STANDS_SLACK_M = 0.0005   # a back-out goes no deeper than where the tool stands, this much for the arm's own jitter


def world_from_base(repo: Path, arm: str) -> np.ndarray | None:
    """The arm's base in the overhead camera's world, from the stack's adopted registration; None without one."""
    from tatbot_calib.cli import _stack

    path = (_stack(repo).get("registration") or {}).get(arm)
    if not path or not Path(path).expanduser().is_file():
        return None
    return np.asarray(json.loads(Path(path).expanduser().read_text())["world_from_arm_base"], float)


def retreat(rig: Rig, kin) -> str:
    """Straight up RETREAT_LIFT_M under the probe guard (a landing's joint path then starts clear of the probe),
    then the verified landing."""
    pose = kin.fk(rig.joints())
    if pose[2, 2] < -0.9:   # the tool points down, as at the station: lift it first
        pose[2, 3] += RETREAT_LIFT_M
        try:
            rig.goal(pose, move=True)
        except RuntimeError as error:   # 2026-10-16: the lift met joint 3's guarded limit and the arm was left holding;
            # 2026-10-22 from roof-tag views a 15 mm lift met it as a 60 mm one did: the pose, not the height
            print(f"the lift before landing was refused ({error}); landing from where the arm stands", flush=True)
    return rig.land()


def _preflight(cal: Calibration) -> None:
    cal.rig.spin(1.0)
    s = cal.rig.safety
    if s is None:
        raise RuntimeError(f"no safety state for the {cal.arm} arm: is the stack up with it?")
    if not s.estop_ok or s.latched or s.probe_triggered:
        raise RuntimeError(f"S0: estop_ok {s.estop_ok}, latched {s.latched}, probe_triggered {s.probe_triggered}")


def _latched(cal: Calibration) -> bool:
    cal.rig.spin(0.3)
    s = cal.rig.safety
    return s is None or s.latched or not s.estop_ok


def touched_fix(cal: Calibration, fix: station.StationFix, tool_id: str) -> station.StationFix:
    """The fix carried onto the arm's station touch while the touch speaks for now (station.touched_palette: the
    tool, joint model, registration and palette unchanged), so S1's prior, the sweep's park and its camera's
    prediction, and the way checks stand where this arm's tip meets the ball. Through the registration the overhead's
    ball stands 3-10 mm off it: on 2026-10-07 the 3RL's sweep parks stood ~10 mm aside the ball (the detector took
    the stylus for the tip), and S1's lines from the fix met nothing out to 9 mm. The fix as it is when no touch
    speaks; a new touch still records the overhead fix it started from."""
    import dataclasses

    urdf = cal.repo / "urdf" / "palette.urdf"
    try:
        touch = station.load_touch(cal.arm)
        base_from_palette = station.touched_palette(touch, fix, tool_id=tool_id, tcp_m=cal.tip0,
                                                    joint_offsets=station.joint_offsets(cal.repo, cal.arm),
                                                    palette_urdf=urdf,
                                                    same_body=station.same_machine(cal.repo, touch.tool_id, tool_id))
    except station.StaleStationError as error:
        cal.say(f"S0: the station prior is the overhead fix: {error}")
        return fix
    ball = base_from_palette[:3, :3] @ station.ball_in_palette(urdf) + base_from_palette[:3, 3]
    cal.say(f"S0: the station prior is the touch {touch.run}: the ball "
            f"{np.round((ball - np.asarray(fix.ball, float)) * 1000, 1).tolist()} mm from the overhead fix")
    return dataclasses.replace(fix, base_from_palette=base_from_palette, ball=ball)


def search_seed(cal: Calibration, fix: station.StationFix, tool_id: str) -> np.ndarray | None:
    """Where S1's search starts when no touch speaks for now: the fix's ball moved by where the arm's last touch
    found the ball from its own fix, when that touch held this tool (or a cartridge of its machine) at this heading.
    The overhead puts the ball off the arm's touches by an error of the arm and its tool, not of the layout: pink's
    11RL and 3RL met it 9.0 and 9.2 mm across the fix at heading -30 a week and a re-layout apart (2026-10-05,
    2026-10-12), so a search from the fix swept 7 empty lines (8 of its 17 minutes) first. The search verifies the
    seed, and _stages falls back to the fix when it finds nothing there. None when no such touch exists, or it was
    touched under another registration."""
    try:
        touch = station.load_touch(cal.arm)
    except (station.StaleStationError, ValueError, KeyError):
        return None
    rotation = halo.tool_pose(np.zeros(3), cal.heading)[:3, :3]
    turned = math.degrees(math.acos(np.clip((np.trace(rotation.T @ touch.rotation) - 1.0) / 2.0, -1.0, 1.0)))
    same = touch.tool_id == tool_id or station.same_machine(cal.repo, touch.tool_id, tool_id)
    # through the tcp in use: the same joints put a tcp moved since by d a rotation of d further on
    # (station.touched_palette); the 1011RL's adopted tip moved it 10 mm (2026-10-12)
    moved = np.asarray(getattr(cal, "tip0", touch.tcp_m), float) - np.asarray(touch.tcp_m, float)
    offset = np.asarray(touch.ball, float) + np.asarray(touch.rotation, float) @ moved - np.asarray(touch.fix.ball, float)
    halo_tip = getattr(cal, "halo", None)
    if halo_tip is None or halo_tip.edges_trip:
        offset[2] = 0.0   # across only: a tool whose edge passes measure the top takes the fix's height
    else:
        # A needle cartridge's touch records the ball a fixed depth under the height its passes met the stylus at
        # (Calibration.top), so its own ball would walk lower each run (2026-10-12: 80.6 -> 75.8 mm). Seeded so its
        # passes run again where they met the stylus, it holds. The overhead reads the ball 6-8 mm high: every
        # search from the fix's height went over the ball before stepping down (2026-10-17, 8-14 passes a touch).
        # (Calibration.top without edge passes: the top half its resolution over the passes, the centre a radius under)
        offset[2] += halo.BALL_RADIUS_M - TOP_RESOLUTION_M / 2.0 + cal.search_below
    # The offset is the touch's from its own fix, through that fix's registration: across another one it carries the
    # old one's error (2026-10-22: the wrist's registration took out the cube's ~9 mm, and the cube's offset sent
    # the search 13.5 mm over the ball)
    registered = touch.fix.detail.get("registration_sha256") == fix.detail.get("registration_sha256")
    if (touch.arm != cal.arm or not same or not registered or turned > SEED_TURN_DEG
            or float(np.linalg.norm(offset)) > SEED_MAX_M):
        return None
    cal.say(f"S1: the search starts {np.round(offset * 1000, 1).tolist()} mm from the fix, where the touch "
            f"{touch.run} met the ball from its own")
    return np.asarray(fix.ball, float) + offset


def place_station(cal: Calibration, fix: station.StationFix) -> None:
    """The station's parts, the floor and the ball's prior from the run's fix (the joint-6 sweep's too)."""
    urdf = cal.repo / "urdf" / "palette.urdf"
    cal.station_pose, cal.ball_in_palette = fix.base_from_palette, station.ball_in_palette(urdf)
    cal.inkcaps = station.inkcap_rims(urdf, cal.repo / "config" / "palette.yaml",
                                   base_from_palette=fix.base_from_palette)
    cal.say(f"S0: station fix {fix.measured_utc}, ball prior {np.round(fix.ball, 4).tolist()} m")
    cal.floor_z = float(fix.ball[2]) + halo.BALL_RADIUS_M - FLOOR_UNDER_TOP_M + FIX_HEIGHT_M
    cal.say(f"S0: the floor is z {cal.floor_z * 1000:.1f} mm, {FLOOR_UNDER_TOP_M * 1000:.0f} mm under the fix's top "
            f"and {FIX_HEIGHT_M * 1000:.0f} mm over for its height")
    cal.ball = np.asarray(fix.ball, float).copy()


def _stages(cal: Calibration, fix: station.StationFix, args) -> int:
    _preflight(cal)
    prior = touched_fix(cal, fix, args.tool)
    if prior is not fix and not cal.halo.edges_trip:
        # No edge pass trips, so S1 settles the top at its search height, a fixed depth under the prior's: from a
        # touch's ball that walks 4.8 mm lower each run (2026-10-12: 80.6 -> 75.8 mm, and every further yaw then
        # stood under the floor). The overhead's height does not walk (80.1-80.6 mm over four registrations).
        prior = dataclasses.replace(prior, ball=np.array([*prior.ball[:2], fix.ball[2]], float))
        cal.say("S1: the prior's height is the overhead fix's: this tool's edge passes cannot measure the top")
    place_station(cal, prior)
    seed = search_seed(cal, fix, args.tool) if prior is fix else None
    floor = cal.floor_z
    if seed is not None:
        cal.ball, cal.seeded = seed.copy(), True
        # the floor 10 mm under the seed's top when that is lower than the fix's (a needle's height seed)
        cal.floor_z = min(floor, float(seed[2]) + halo.BALL_RADIUS_M - FLOOR_UNDER_TOP_M + FIX_HEIGHT_M)
    cal.approach(cal.ball)
    found = cal.s1()
    if not found and seed is not None and not cal.stuck and not _latched(cal):
        cal.say("S1: nothing at the seed; searching from the fix")
        cal.ball, cal.seeded, cal.floor_z = np.asarray(fix.ball, float).copy(), False, floor
        found = cal.s1()
    if not found:
        cal.say("S1: the side touches did not find the ball; stopping")
        return 1
    cal.say(f"S1: the ball stands {np.round((cal.ball - np.asarray(fix.ball, float)) * 1000, 1).tolist()} mm from the fix")
    measured = cal.ball.copy()          # the ball through the installed tcp: the frame the executor commands
    touch = station.StationTouch(cal.arm, args.tool, measured, halo.tool_pose(np.zeros(3), cal.heading)[:3, :3],
                                 np.asarray(cal.tip0, float), station.joint_offsets(cal.repo, cal.arm), fix,
                                 cal.run.dir.name, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if getattr(args, "touch_path", None):
        touch = _adopt_touch(cal, touch, Path(args.touch_path).expanduser())
    else:
        touch.write(cal.run.dir / "station-touch.json")
    if getattr(args, "station_only", False):
        return 0
    _fit(cal)
    s4(cal, args.attitudes, measured)
    fitted = _fit(cal)
    frame = _frame_at(cal.kin, cal.arm)
    held = solve.leave_one_group_out(cal.contacts, frame, **_fit_args(cal))
    moved = {group: float(np.linalg.norm(solve.fit([c for c in cal.contacts if c.group != group], frame,
                                                   **_fit_args(cal)).tip - fitted.tip))
             for group in sorted({c.group for c in cal.contacts}) if len({c.group for c in cal.contacts}) > 2}
    (cal.run.dir / "candidate.json").write_text(json.dumps({
        "schema": "tatbot.probe-tip-candidate/2", "arm": cal.arm, "tool": args.tool, "fit": fitted.as_dict(),
        "tip0_m": np.asarray(cal.tip0).tolist(), "held_out_rms_m": held, "tip_moved_m": moved,
        "contacts": len(cal.contacts), "station": args.station, "run_id": cal.run.dir.name,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    report = _report(cal.run.dir, fitted, held, moved, cal.tip0,
                     sum(1 for _ in (cal.run.dir / "touches.jsonl").open()))
    (cal.run.dir / "report.md").write_text(report)
    print(report, flush=True)
    return 0


def s4(cal: Calibration, attitudes, measured: np.ndarray) -> int:
    """S4: the candidate yaws until ATTITUDES_WANTED have measured the ball; one the executor will not plan, or whose
    pairs do not both trip, is skipped and its contacts dropped. The yaws kept are the list's first that measure, as
    in its own order, but those still wanted are swept from the end of their span nearer the tool's yaw: from S1's
    0, -60 -30 +60 turns the wrist 180 deg where the list's -60 +60 -30 turned it 270, and +60's ball is still placed
    by the 0..-60 fan (2026-10-22: joints 3 and 4 rang 14-15 mrad on its 20-27 s swings). Returns how many measured."""
    done, at, tried = 0, 0.0, set()
    while wanted := sorted([i for i in range(len(attitudes)) if i not in tried][:ATTITUDES_WANTED - done],
                           key=lambda i: attitudes[i][0]):
        i = min((wanted[0], wanted[-1]), key=lambda i: (abs(attitudes[i][0] - at), i))
        tried.add(i)
        yaw, tilt = attitudes[i]
        at = yaw
        group = f"S4 {yaw:+.0f}/{tilt:+.0f}"
        cal.ball = measured.copy()   # attitude() places it by the fit once two yaws are in
        kept = len(cal.contacts)
        try:
            found = cal.attitude(yaw, group)
        except RuntimeError as error:
            if cal.stuck or _latched(cal):
                raise
            found = False
            cal.say(f"{group}: skipped ({error})")
        if not found:
            del cal.contacts[kept:]
            cal.say(f"{group}: skipped; its contacts are not fitted")
            continue
        _fit(cal)
        done += 1
    return done


def _fit_args(cal: Calibration) -> dict:
    """The tip across the axis, the ball and a side radius per yaw and pair axis; the tip's length and the axis held
    (the module's note), no joint offsets."""
    return {"tip0": cal.tip0, "axis0": [0.0, 0.0, 1.0], "ball0": cal.ball, "side0": cal.side_radius,
            "sigma_tip": np.array([0.010, 0.010, 1e-7]), "sigma_axis": 1e-7, "per_axis": True}


def _fit(cal: Calibration) -> solve.Fit:
    """The fit so far. Its side radius plans the next touches only once two yaws are in: one cannot separate the
    tip from the ball, and after S1 alone (2026-09-30, pink at v11) its side radius so fattened the tool that a
    touch straight down over the ball read as meeting the palette camera 122 mm away; S1's half-span stands."""
    fitted = cal.fitted = solve.fit(cal.contacts, _frame_at(cal.kin, cal.arm), **_fit_args(cal))
    if len({contact.group for contact in cal.contacts}) >= 2:
        cal.side_radius = fitted.side_radius
    elif cal.s1_side_radius:
        cal.side_radius = cal.s1_side_radius
    cal.say(f"fit: {len(cal.contacts)} contacts, rms " + ", ".join(f"{k} {v * 1000:.3f} mm" for k, v in
                                                                   fitted.rms_m.items())
            + f"; tip across ({fitted.tip[0] * 1000:.2f}, {fitted.tip[1] * 1000:.2f}) mm")
    return fitted


def _attitudes(text: str) -> list[tuple[float, float]]:
    """"yaw/tilt,yaw/tilt" in degrees about the base heading; "none" is S1 alone. A tilt is kept for reach.choose_heading
    and the arm's attitude, though the fit measures upright yaws."""
    if text.strip().lower() == "none":
        return []
    out = []
    for item in filter(None, (part.strip() for part in text.split(","))):
        yaw, _, tilt = item.partition("/")
        out.append((float(yaw), float(tilt or 0.0)))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tatbot_calib calib")
    sub = parser.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("run", help="the fitted tool's tip across its axis, from side touches on the station probe")
    p.add_argument("--arm", required=True)
    p.add_argument("--tool", default=None, help="the tool stated to be fitted; must be config/workspace.yaml's")
    p.add_argument("--station", required=True, help="this run's station.json (tatbot ros station)")
    p.add_argument("--run-started", type=float, required=True, help="epoch seconds this calibration began")
    p.add_argument("--max-age-s", type=float, default=station.MAX_AGE_S)
    p.add_argument("--heading-deg", type=float, default=None,
                   help="the base attitude's tool heading; default the one whose attitudes the arm reaches best")
    p.add_argument("--touch-path", default=None,
                   help="where S1's station touch is adopted for dips (default ~/tatbot-ros/calib/station-touch-<arm>.json)")
    p.add_argument("--station-only", action="store_true",
                   help="S1 alone: touch the ball for the station touch dips aim by, and fit no tip")
    p.add_argument("--attitudes", type=_attitudes, default=_attitudes(DEFAULT_ATTITUDES),
                   help="further upright yaws as yaw/0 degrees about the base heading, comma separated; those the "
                        "arm cannot reach, or reach only with its wrist at the camera's post, are left out")
    args = parser.parse_args(argv)
    args.touch_path = args.touch_path or str(station.touch_path(args.arm))
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
