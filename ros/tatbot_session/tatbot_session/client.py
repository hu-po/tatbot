"""`ros2 run tatbot_session client <verb> ...`: what `tatbot ros draw|touch|decide|status` run on the arm node.

    client draw PROGRAM [--arm right] [--from-op ID] [--run-id ID]   # PROGRAM may be '' with --run-id
    client touch [--arm right]                                       # three page touches, the plane
    client jog --joint N --delta D [--arm right] [--speed RAD_S]     # one joint from its measured position
    client decide ARM continue|land|skip|redraw
    client land [--arm right]
    client wake --arm right                                          # a landed arm back, the other untouched
    client cancel                                                    # every draw/touch goal; the arm holds
    client status [--json]

Events print as they arrive; decisions go through `client decide` from another shell. Exit 0 when the
goal succeeded, 1 otherwise. With a keypad in stack.yaml, a draw that lands for a cartridge swap waits for its
Enter, wakes the arm and resumes the run; Ctrl-C there leaves the run to resume by hand. A cancelled draw, by Ctrl-C
before then or `client cancel`, does not wait: it too is resumed by hand.

Ctrl-C, SIGTERM and SIGHUP (a dropped ssh) during a draw or touch cancel its goal and wait for the arm to
hold. rclpy's own SIGINT handler is off: it shut the context down before the cancel could be sent, so the
goal kept drawing. Every wait wakes at least every WAKE_S (`_spin`), or the handler would run only when a
callback arrived: a page touch sends none between its trips, and a SIGTERM 25 s into one was still unhandled
60 s later (2026-10-04).
"""
from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import time
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from tatbot_description import names
from tatbot_interfaces.action import Draw, Land, Touch
from tatbot_interfaces.msg import Event, SafetyState
from tatbot_interfaces.srv import Decide

from tatbot_session import config, geometry, rules
from tatbot_session import ledger as ledger_mod
from tatbot_session.arm import LATCH_TEXT

PHASES = {v: k[6:].lower() for k, v in vars(Draw.Feedback).items() if k.startswith("PHASE_")}
DECISIONS = {"continue": Decide.Request.CONTINUE, "land": Decide.Request.LAND, "skip": Decide.Request.SKIP,
             "redraw": Decide.Request.REDRAW}
WAKE_S = 0.2   # the longest a signal waits for its handler


def _print_event(msg: Event) -> None:
    print(f"{time.strftime('%H:%M:%S')} [{msg.arm or '-'}] {msg.kind} {msg.op_id} {msg.text}".rstrip(), flush=True)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _spin(node: Node, future, timeout_sec: float | None = None) -> None:
    """rclpy.spin_until_future_complete, waking every WAKE_S. A signal handler runs only once rclpy's wait
    returns, and without a timeout that wait returns only for a callback."""
    end = None if timeout_sec is None else time.monotonic() + timeout_sec
    while not future.done():
        left = WAKE_S if end is None else min(WAKE_S, end - time.monotonic())
        if left <= 0:
            return
        rclpy.spin_once(node, timeout_sec=left)


def _cancel(node: Node, name: str, sent, result_future) -> None:
    """Cancel a sent goal and wait for its result: the executor stops at its next check and the arm holds.
    Further interrupts are ignored until then, so a second Ctrl-C cannot abandon a half-sent cancel."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    print(f"{name}: cancelling; the arm holds", file=sys.stderr, flush=True)
    if not sent.done():
        _spin(node, sent, 5.0)
    handle = sent.result() if sent.done() else None
    if handle is None or not handle.accepted:
        print(f"{name}: no accepted goal to cancel", file=sys.stderr)
        return
    cancel = handle.cancel_goal_async()
    _spin(node, cancel, 5.0)
    if not cancel.done() or not cancel.result().goals_canceling:
        print(f"{name}: the cancel was not accepted; run `client cancel` or `client land`", file=sys.stderr)
        return
    result_future = result_future or handle.get_result_async()
    _spin(node, result_future, 30.0)
    if not result_future.done():
        print(f"{name}: cancel accepted but the goal has not ended after 30 s", file=sys.stderr)


def _rejection(node: Node, goal) -> str:
    """Why the session rejected a Draw or Touch goal, which a rejection does not say: the landed arms of the goal
    as /tatbot/safety reports them (every arm for a Draw that names none), else a busy or unknown arm."""
    wanted = [arm for arm in (list(getattr(goal, "arms", [])) or [getattr(goal, "arm", "")]) if arm]
    states: dict[str, SafetyState] = {}
    sub = node.create_subscription(SafetyState, names.SAFETY_TOPIC, lambda m: states.__setitem__(m.arm, m), 10)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not (wanted and all(arm in states for arm in wanted)):
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    landed = [arm for arm, s in sorted(states.items()) if s.landed and (not wanted or arm in wanted)]
    return "; ".join(rules.landed_refusal(arm) for arm in landed) or "the arm is busy or unknown"


def _run_goal(node: Node, action_type, name: str, goal, on_feedback=None) -> tuple[bool, object, bool]:
    """Send a goal and wait for it: (succeeded, its result or None, cancelled). Cancelled is a goal stopped on purpose,
    by an interrupt here or a cancel from anywhere; nothing that waits for the operator may follow it, since an
    interrupt leaves every further signal ignored (`_cancel`). A draw interrupted while it landed for a cartridge swap
    used to go on to wait for the keypad, where only a kill ended it."""
    client = ActionClient(node, action_type, name)
    if not client.wait_for_server(timeout_sec=10.0):
        print(f"{name}: no server (is the stack up?)", file=sys.stderr)
        return False, None, False
    sent, result_future, interrupted = client.send_goal_async(goal, feedback_callback=on_feedback), None, False
    try:
        _spin(node, sent)
        handle = sent.result()
        if not handle.accepted:
            print(f"{name}: goal rejected: {_rejection(node, goal)}", file=sys.stderr)
            return False, None, False
        result_future = handle.get_result_async()
        _spin(node, result_future)
    except KeyboardInterrupt:
        interrupted = True
        _cancel(node, name, sent, result_future)
        if result_future is None or not result_future.done():
            return False, None, True
    response = result_future.result()
    status = GoalStatus.STATUS_UNKNOWN if response is None else response.status
    return (status == GoalStatus.STATUS_SUCCEEDED, response.result if response else None,
            interrupted or status == GoalStatus.STATUS_CANCELED)


def _cancel_all(node: Node, args) -> int:
    """Cancel every draw and touch goal on the session, whoever sent it (a zero goal id and stamp)."""
    from action_msgs.srv import CancelGoal

    canceling = 0
    for name in (names.DRAW_ACTION, names.TOUCH_ACTION):
        srv = node.create_client(CancelGoal, f"{name}/_action/cancel_goal")
        if not srv.wait_for_service(timeout_sec=5.0):
            print(json.dumps({"ok": False, "message": f"{name}: no server (is the stack up?)"}))
            return 1
        future = srv.call_async(CancelGoal.Request())
        _spin(node, future, 5.0)
        if not future.done():
            print(json.dumps({"ok": False, "message": f"{name}: no answer to the cancel"}))
            return 1
        canceling += len(future.result().goals_canceling)
    print(json.dumps({"ok": True, "canceling": canceling,
                      "message": f"{canceling} goal(s) cancelling; the arm holds" if canceling else "no goal running"}))
    return 0


def _draw(node: Node, args) -> int:
    program = Path(args.program).read_text() if args.program else ""
    goal = Draw.Goal(program_json=program, arms=[args.arm] if args.arm else [], from_op=args.from_op,
                     run_id=args.run_id)
    last = {}

    def feedback(msg):
        fb = msg.feedback
        key = (fb.arm, fb.op_id, fb.phase)
        if key != last.get("key"):
            last["key"] = key
            print(f"{time.strftime('%H:%M:%S')} [{fb.arm}] {fb.index + 1}/{fb.total} {fb.op_id} "
                  f"{PHASES.get(fb.phase, fb.phase)}", flush=True)

    while True:
        ok, result, cancelled = _run_goal(node, Draw, names.DRAW_ACTION, goal, feedback)
        if result is not None:
            print(json.dumps({"ok": ok, "done": result.done, "skipped": result.skipped,
                              "uncertain": list(result.uncertain), "run_id": result.run_id, "run_dir": result.run_dir,
                              "message": result.message}))
        swap = None if ok or cancelled or result is None else _swap(result.run_dir, args.arm)
        if swap is None:
            return 0 if ok else 1
        arm, text = swap
        print(f"{time.strftime('%H:%M:%S')} [{arm}] {text}: fit it, then press Enter on the keypad to resume "
              f"(Ctrl-C leaves it to: tatbot ros draw <program> --arm {arm} --resume {result.run_id})", flush=True)
        if not _enter(node, arm) or _wake(node, argparse.Namespace(arm=arm)) != 0:
            return 1
        goal = Draw.Goal(program_json="", arms=[arm], from_op="", run_id=result.run_id)


def _swap(run_dir: str, arm: str) -> tuple[str, str] | None:
    """(arm, what to fit) when the run stopped at a cartridge swap the keypad can answer: its next op a tool change
    left `sent` (the arm landed for it) on an arm stack.yaml gives the keypad; else None."""
    keypad = config.load_stack().get("keypad") or {}
    run = Path(run_dir or "")
    if not keypad.get("device") or not (run / "program.json").is_file() or arm not in ("", keypad.get("arm")):
        return None
    arm = keypad["arm"]
    ops, rows = json.loads((run / "program.json").read_text())["ops"], ledger_mod.read(run / "ledger.jsonl")
    index = ledger_mod.next_index(ops, rows, arm)
    if index is None or ops[index]["op"] != "tool_change" or ledger_mod.op_status(rows, arm, ops[index]["id"]) != "sent":
        return None
    return arm, f"cartridge swap {ops[index]['id']} ({ops[index]['resource_id']})"


def _enter(node: Node, arm: str) -> bool:
    """Wait for the keypad's Enter for this arm; False on Ctrl-C."""
    from std_msgs.msg import String

    keys: queue.Queue = queue.Queue()
    sub = node.create_subscription(String, names.keys_topic(arm), lambda msg: keys.put(msg.data), 10)
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.2)
            while not keys.empty():
                if keys.get() == "enter":
                    return True
    except KeyboardInterrupt:
        return False
    finally:
        node.destroy_subscription(sub)


def _touch_goal(args) -> Touch.Goal:
    """Three page touches, or with --start one touch along --direction (--probe: on the station probe)."""
    if args.start is None:
        return Touch.Goal(mode=Touch.Goal.MODE_PAGE, arm=args.arm)
    guard = Touch.Goal.GUARD_PROBE if args.probe else (0 if args.move else Touch.Goal.GUARD_TIP_LAG)
    goal = Touch.Goal(mode=Touch.Goal.MODE_MOVE if args.move else Touch.Goal.MODE_SINGLE, arm=args.arm,
                      max_travel_m=args.max_travel, speed_m_s=args.speed, prior_distance_m=args.prior, guard=guard)
    goal.start_pose.header.frame_id = f"{args.arm}/base_link"
    p = goal.start_pose.pose.position
    p.x, p.y, p.z = args.start
    o = goal.start_pose.pose.orientation   # zero: the session keeps the tool's measured orientation
    o.x, o.y, o.z, o.w = (0.0, 0.0, 0.0, 0.0) if args.rpy is None else geometry.matrix_quat(
        geometry.rpy_matrix((0.0, 0.0, 0.0), args.rpy))
    goal.direction.x, goal.direction.y, goal.direction.z = args.direction
    return goal


def _touch(node: Node, args) -> int:
    ok, result, _ = _run_goal(node, Touch, names.TOUCH_ACTION, _touch_goal(args))
    if result is not None:
        print(json.dumps({"ok": ok, "tripped": result.tripped, "plane": list(result.plane),
                          "contacts": [[p.pose.position.x, p.pose.position.y, p.pose.position.z]
                                       for p in result.contact_poses],
                          "run_dir": result.run_dir, "message": result.message}))
    return 0 if ok else 1


def _land(node: Node, args) -> int:
    ok, result, _ = _run_goal(node, Land, names.LAND_ACTION, Land.Goal(arms=[args.arm] if args.arm else []))
    if result is not None:
        print(json.dumps({"ok": ok, "landed": list(result.landed), "message": result.message}))
    return 0 if ok and all(result.landed) else 1


CONTROLLER_MANAGER = "/controller_manager"
WAKE_TIMEOUT_S = 10.0


def _safety(node: Node, arm: str, timeout_s: float = 2.0) -> SafetyState | None:
    """The arm's latest /tatbot/safety, or None within timeout_s."""
    got: list = []
    sub = node.create_subscription(SafetyState, names.SAFETY_TOPIC,
                                   lambda m: got.append(m) if m.arm == arm else None, 10)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not got:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return got[-1] if got else None


def _call(node: Node, srv_type, name: str, request, timeout_s: float = WAKE_TIMEOUT_S):
    client = node.create_client(srv_type, name)
    try:
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f"{name}: no service (is the stack up?)")
        future = client.call_async(request)
        _spin(node, future, timeout_s)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"{name}: no answer within {timeout_s:g} s")
        return future.result()
    finally:
        node.destroy_client(client)


def _switch(node: Node, *, activate=(), deactivate=()) -> None:
    from controller_manager_msgs.srv import SwitchController

    req = SwitchController.Request(activate_controllers=list(activate), deactivate_controllers=list(deactivate),
                                   strictness=SwitchController.Request.STRICT)
    if not _call(node, SwitchController, f"{CONTROLLER_MANAGER}/switch_controller", req).ok:
        raise RuntimeError(f"switch_controller refused (activate {list(activate)}, deactivate {list(deactivate)})")


def _hardware(node: Node, arm: str, state_id: int, label: str) -> None:
    from controller_manager_msgs.srv import SetHardwareComponentState
    from lifecycle_msgs.msg import State

    req = SetHardwareComponentState.Request(name=names.hardware_name(arm), target_state=State(id=state_id, label=label))
    if not _call(node, SetHardwareComponentState, f"{CONTROLLER_MANAGER}/set_hardware_component_state", req).ok:
        raise RuntimeError(f"{names.hardware_name(arm)} did not go {label}")


def _wake(node: Node, args) -> int:
    """A landed arm taken back without restarting the stack, which would take the other arm down mid-goal
    (2026-09-30, two agents on one rig): its two controllers off, its hardware alone cycled inactive and
    active (the driver re-enters control at the rest it measures and holds it, as at the stack's start), its
    controllers on again, the trajectory controller starting from the measured pose. An arm that has not
    landed is left as it is; a stack that cannot find the arm, or one whose cycle fails, exits 1."""
    from lifecycle_msgs.msg import State

    arm = args.arm
    state = _safety(node, arm)
    if state is None:
        print(json.dumps({"ok": False, "arm": arm, "message": f"no /tatbot/safety for the {arm} arm (is it in the stack?)"}))
        return 1
    if not state.landed:
        print(json.dumps({"ok": True, "arm": arm, "woken": False, "message": f"{arm}: not landed; left as it is"}))
        return 0
    controllers = [names.arm_controller(arm), names.safety_controller(arm)]
    try:
        _switch(node, deactivate=controllers)
        try:
            _hardware(node, arm, State.PRIMARY_STATE_INACTIVE, "inactive")
            _hardware(node, arm, State.PRIMARY_STATE_ACTIVE, "active")
        finally:
            _switch(node, activate=controllers)   # a failed cycle leaves the arm landed, its controllers back
    except RuntimeError as error:
        print(json.dumps({"ok": False, "arm": arm, "message": str(error)}))
        return 1
    deadline = time.monotonic() + WAKE_TIMEOUT_S
    while time.monotonic() < deadline:
        state = _safety(node, arm, 1.0)
        if state is not None and not state.landed:
            ok = not state.latched and state.estop_ok and not state.controller_error
            print(json.dumps({"ok": ok, "arm": arm, "woken": True, "latched": state.latched,
                              "message": f"{arm}: awake, holding its rest" if ok else
                              f"{arm}: awake but held ({LATCH_TEXT.get(state.latch_reason, state.latch_reason)})"}))
            return 0 if ok else 1
    print(json.dumps({"ok": False, "arm": arm, "message": f"{arm}: still reports landed after the cycle"}))
    return 1


def _goal_running(node: Node, arm: str) -> bool:
    """Whether the arm's trajectory controller reports a goal accepted or executing (a draw, a touch)."""
    from action_msgs.msg import GoalStatusArray
    from rclpy.qos import DurabilityPolicy, QoSProfile

    seen: list = []
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sub = node.create_subscription(GoalStatusArray, f"{names.follow_joint_trajectory(arm)}/_action/status",
                                   seen.append, qos)
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_subscription(sub)
    active = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING)
    return bool(seen) and any(g.status in active for g in seen[-1].status_list)


def _inspect_views(io, kin, cam, used, centre, extent, frame, out_dir, args):
    """Visit the inspect poses: aim, move, settle, grab and rectify. Returns (views, poses), or (None, [])
    when a move does not finish."""
    import cv2
    import numpy as np
    from tatbot_description import repo_root
    from tatbot_motion import load_motion
    from tatbot_motion.collision import Guard

    from tatbot_session import config
    from tatbot_session import inspect as ins
    from tatbot_session.ready import joint_move, solve_ik

    # These moves go through this process's ArmIO, not the session's guarded goals: the choice checks each view
    # and its joint way against the arm itself (inspect.path_clear).
    guard, why = Guard.from_stack(repo_root(None), config.load_stack(), load_motion())
    if guard is None:
        print(f"inspect: views chosen WITHOUT a self-collision check: {why}", file=sys.stderr)
    self_gap = None if guard is None else (lambda q: guard.self_gap(args.arm, q))
    views, poses = [], []
    best = None
    for n in range(max(1, args.poses)):
        q = io.measured()
        side = 1 if n % 2 else -1
        azimuths = None if best is None else [np.radians(best["azimuth_deg"] + side * d) for d in (15, 30, 45)]
        # Keep the arm on its own side of the design, clear of the palette mid-table (bench 2026-09-26).
        max_y = float((used @ [centre[0], centre[1], 0.0, 1.0])[1]) - args.side_margin
        aimed = ins.aim(kin, solve_ik, used, centre, q, frame=frame, clearance_m=args.clearance, azimuths=azimuths,
                        max_y=max_y, self_gap=self_gap)
        if aimed is None:
            print(f"pose {n}: no reachable camera pose with the pen {args.clearance * 1000:.0f} mm clear", file=sys.stderr)
            continue
        q_t, view = aimed
        best = best or view
        traj = joint_move(kin.joint_names, q, q_t, max_rad_s=args.speed, max_m_s=0.002, rate_hz=100.0, kin=kin)
        status, _, text = io.execute(traj)
        if status != "ok":
            print(f"pose {n}: {status} {text}", file=sys.stderr)
            return None, []
        time.sleep(args.settle)
        images = cam.grab(args.frames)
        q_m = io.measured()
        base_from_cam = kin.frame(q_m, frame)
        rectified = []
        for j, img in enumerate(images):
            cv2.imwrite(str(out_dir / f"pose{n}-{j}.png"), img)
            rectified.append(ins.rectify(img, used, base_from_cam, cam.k, cam.dist, extent, args.px_per_mm))
        # One page image per pose: the mount pose is CAD, so views from other directions do not overlay
        # to the millimetre (40 deg views disagreed by several mm, bench 2026-09-26).
        views.append(ins.median_stack(rectified))
        poses.append({"pose": n, "xy_m": centre.tolist(), **view, "q": q_m.tolist(),
                      "base_from_camera": base_from_cam.tolist(), "frames": len(images)})
        print(f"{time.strftime('%H:%M:%S')} [{args.arm}] pose {n}: camera {view['distance_m'] * 1000:.0f} mm from the "
              f"design, {view['tilt_deg']:.0f} deg off the normal at azimuth {view['azimuth_deg']:.0f} deg, "
              f"{len(images)} frames", flush=True)
    return views, poses


def _rest(io, kin, args) -> None:
    """Back to the rest (sleep) pose by a joint move, motors holding: inspect never leaves the arm up."""
    from tatbot_session import inspect as ins
    from tatbot_session.ready import joint_move

    if kin is None or io.latched or not args.rest:
        return
    q = io.measured()
    rest = ins.REST.copy()
    rest[6] = q[6]
    status, _, text = io.execute(joint_move(kin.joint_names, q, rest, max_rad_s=args.speed, max_m_s=0.002,
                                            rate_hz=100.0, kin=kin))
    print(f"{time.strftime('%H:%M:%S')} [{args.arm}] rest: {status} {text}".rstrip(), flush=True)


def _inspect_lift(io, kin, q, used, motion, args):
    from tatbot_motion import plan_lift, to_knots

    normal = used[:3, 2]
    height = float((kin.fk(q)[:3, 3] - used[:3, 3]) @ normal)
    if height < args.lift:
        status, _, text = io.execute(to_knots(plan_lift(q_seed=q, direction=normal, distance_m=args.lift - height,
                                                        kin=kin, motion=motion), 100.0))
        if status != "ok":
            print(f"lift: {status} {text}", file=sys.stderr)
            raise RuntimeError(f"lift: {status} {text}")


def _closeup_hold(io, kin, cam, used, xy, heading, cal, window, motion, cfg, args, out_dir, n):
    """One close-up hold (tatbot_session.closeup): a joint move over page point xy, straight down to the hover
    height at the drawing's heading, frames grabbed, each put on the page through the gauge-anchored camera pose
    and kept within HOLD_RADIUS_M of the tip, then straight back up. Returns the page image, or None when the
    camera does not make out the pen and the paper."""
    import cv2
    import numpy as np
    from tatbot_motion import plan_lift, to_knots

    from tatbot_session import closeup, gauge, geometry
    from tatbot_session import inspect as ins
    from tatbot_session.ready import joint_move, solve_ik_seeded

    normal = used[:3, 2]
    hover, approach = float(cfg.get("closeup_hover_m", 0.012)), float(cfg.get("closeup_approach_m", 0.040))
    above = geometry.tool_down_pose(used, xy, hover + approach, heading)
    q = io.measured()
    moves = [lambda: joint_move(kin.joint_names, q, solve_ik_seeded(kin, above, q), max_rad_s=args.speed,
                                max_m_s=0.002, rate_hz=100.0, kin=kin),
             lambda: to_knots(plan_lift(q_seed=io.measured(), direction=-normal, distance_m=approach, kin=kin,
                                        motion=motion), 100.0)]
    for make in moves:
        status, _, text = io.execute(make())
        if status != "ok":
            raise RuntimeError(f"close-up hold {n}: {status} {text}")
    try:
        time.sleep(args.settle)
        colors, _ = cam.grab_both(args.frames)
        q_m = io.measured()
        frame = gauge.measure(cam.raw_depth[0], colors[0], cam.depth_metadata, cal)
        cv2.imwrite(str(out_dir / f"closeup-hold{n}.png"), colors[0])
        if frame is None:
            print(f"close-up hold {n}: the wrist camera makes out no pen and paper", file=sys.stderr)
            return None
        pose = closeup.anchored(kin.frame(q_m, ins.camera_frame(args.arm)), kin.fk(q_m)[:3, 3], frame, cal, normal)
        here = closeup.on_paper(used, pose, frame)   # the page where this frame's depth puts the paper
        gx, gy = ins.page_grid(window, args.px_per_mm)
        far = np.hypot(gx - xy[0], gy - xy[1]) > closeup.HOLD_RADIUS_M
        pages, paper = [], gauge.on_paper(cam.raw_depth[0], cam.depth_metadata, frame, colors[0].shape)
        for img in colors:
            img = img.copy()
            img[closeup.pen_mask(img) | ~paper] = 0   # the pen by its teal, and any cartridge by its depth
            page = ins.rectify(img, here, pose, cam.k, cam.dist, window, args.px_per_mm)
            page[far] = 0
            pages.append(page)
        print(f"{time.strftime('%H:%M:%S')} [{args.arm}] close-up hold {n} over ({xy[0] * 1e3:+.1f} {xy[1] * 1e3:+.1f}) "
              f"mm: the tip {gauge.height(frame, cal) * 1e3:.1f} mm over the paper", flush=True)
        stacked = ins.median_stack(pages)
        cv2.imwrite(str(out_dir / f"closeup-page{n}.png"), stacked)
        return stacked
    finally:
        status, _, text = io.execute(to_knots(plan_lift(q_seed=io.measured(), direction=normal, distance_m=approach,
                                                        kin=kin, motion=motion), 100.0))
        if status != "ok":
            raise RuntimeError(f"close-up hold {n} lift: {status} {text}")


def _closeup(io, kin, used, page, program, clear_m, out_dir, motion, args) -> dict | None:
    """The close-up ink inspection after the far views (tatbot_session.closeup), when page.inspect.closeup and a
    wrist gauge fit: holds over the drawing at its heading, one page image of them, each stroke's ink share at
    the plan and at the shift that finds the most (closeup.json, closeup.png, closeup-overlay.png)."""
    import cv2
    import numpy as np

    from tatbot_session import closeup, config, gauge
    from tatbot_session import inspect as ins

    cfg = (config.load_stack().get("page") or {}).get("inspect") or {}
    path = gauge.calibration_path(args.arm)
    if not args.closeup or not cfg.get("closeup", False) or not path.exists():
        why = "off" if path.exists() else f"no wrist gauge fit at {path}"
        print(f"{time.strftime('%H:%M:%S')} [{args.arm}] close-up: {why}", flush=True)
        return None
    cal = json.loads(path.read_text())
    rows = page.get("gauge") or page.get("touches") or []
    heading = np.array(rows[0]["tcp"], float) if rows else kin.fk(io.measured())   # the drawing's own heading
    window = closeup.extent(program)
    _inspect_lift(io, kin, io.measured(), used, motion, args)
    cam = ins.WristCamera(serial=args.serial, depth=True)
    try:
        views = [v for n, xy in enumerate(closeup.holds(program, float(cfg.get("closeup_step_m", 0.012))))
                 if (v := _closeup_hold(io, kin, cam, used, xy, heading, cal, window, motion, cfg, args, out_dir, n))
                 is not None]
    finally:
        cam.close()
    if not views:
        return None
    image = ins.median_stack(views)
    valid = (image.max(axis=2) > 0) & closeup.in_clear(window, args.px_per_mm, clear_m)   # no printed border
    result = closeup.coverage(program, closeup.ink_score(image, valid, args.px_per_mm), valid, window, args.px_per_mm,
                              tol_m=float(cfg.get("closeup_tol_m", 0.0015)))
    cv2.imwrite(str(out_dir / "closeup.png"), image)
    cv2.imwrite(str(out_dir / "closeup-overlay.png"), closeup.overlay(image, result, program, window, args.px_per_mm))
    (out_dir / "closeup.json").write_text(json.dumps({"window_m": window, "px_per_mm": args.px_per_mm,
                                                      "holds": len(views), **result}, indent=1))
    shift = result["best_shift_m"]
    for rid, r in result["resources"].items():
        best = result["at_best_shift"]["resources"][rid]
        print(f"{time.strftime('%H:%M:%S')} [{args.arm}] close-up {rid}: ink along "
              + ("unseen" if r["ink"] is None else f"{r['ink'] * 100:.0f}%") + " of the plan; "
              + ("" if best["ink"] is None else f"{best['ink'] * 100:.0f}% {shift[0] * 1e3:+.1f} {shift[1] * 1e3:+.1f} mm off it"),
              flush=True)
    return {"resources": result["resources"], "best_shift_m": shift}


def _inspect(node: Node, args) -> int:
    """Hover the wrist D405 over a finished drawing and write page-frame views of it (tatbot_session.inspect):
    lift the pen, move to a few camera poses looking down the page normal, grab frames, rectify each onto
    the page with the plan drawn over it. The drawing is the newest ros-draw run unless --run names one."""
    import threading

    import numpy as np
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import JointState
    from tatbot_bridge import capture
    from tatbot_description import repo_root, robot_description
    from tatbot_motion import Kinematics, load_motion

    from tatbot_session import config
    from tatbot_session import inspect as ins
    from tatbot_session.arm import ArmIO

    root = Path(args.logs).expanduser()
    run = ins.newest_run(root, args.run)
    if run is None:
        print(f"no draw run with page.json under {root}", file=sys.stderr)
        return 2
    page = json.loads((run / "page.json").read_text())
    program = json.loads((run / "program.json").read_text())
    used = np.array(page["used"], dtype=float)
    stack_page = config.load_stack().get("page") or {}
    if str(page.get("pattern_id") or "").startswith("stencil-"):   # the print the run drew on (`ros up --pattern`)
        stack_page = {**stack_page, "pattern_id": page["pattern_id"]}
    stack_page = config.print_page(repo_root(None), stack_page)   # that print's page
    clear, size = stack_page["clear_m"], stack_page["size_m"]
    inner = ins.inner_edges(clear, stack_page.get("inner_edges_m"))   # a malformed key refuses before the arm moves
    if _goal_running(node, args.arm):
        print(f"{args.arm}: a trajectory goal is running (a draw or touch); inspect after it ends", file=sys.stderr)
        return 1
    group = ReentrantCallbackGroup()
    io = ArmIO(node, args.arm, group)
    node.create_subscription(JointState, names.JOINT_STATES_TOPIC, io.on_joint_state, 20, callback_group=group)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    out_dir = run / "inspect" / time.strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = None
    try:
        kin = Kinematics(robot_description(None, arms=(args.arm,)), args.arm)
        motion = load_motion()
        frame = ins.camera_frame(args.arm)
        q = io.measured()
        io.wait(lambda: bool(io.safety), timeout=2.0)
        if io.latched:
            print(f"{args.arm}: latched ({io.latch_text()}); decide first", file=sys.stderr)
            return 1
        if getattr(args, 'depth_only', False):
            from tatbot_cli.ros_depth import stationary

            stationary(node, io, kin, page, out_dir, args)
            return 0
        _inspect_lift(io, kin, q, used, motion, args)
        cam = ins.WristCamera(serial=args.serial)
        strokes = [p for op in program["ops"] if op.get("op") == "stroke" for p in op["points_m"]]
        centre = (np.min(strokes, axis=0) + np.max(strokes, axis=0)) / 2 if strokes else np.zeros(2)
        extent = ins.page_extent(size)
        views, poses = _inspect_views(io, kin, cam, used, centre, extent, frame, out_dir, args)
        if views:   # none reachable (a design on the arm's own side of the page): the close-up still looks
            gate = stack_page.get("inspect") or {}
            # the views aim at the design's centre: the border fit's scale is about it (inspect.solve_in_plane)
            analysis = ins.analyse(views, program, clear, extent, args.px_per_mm, ink_prior_m=gate.get("ink_prior_m"),
                                   ink_gate_m=float(gate.get("ink_gate_m", 0.003)), inner_m=inner, size_m=size,
                                   centre_m=centre)
            capture._lib(repo_root(None))
            from stencil_coded_live import write_inspection

            write_inspection(out_dir, poses, program, stack_page['pattern_id'], views, analysis, ins,
                             clear, extent, args.px_per_mm)
        meta = {"run": run.name, "camera": {"serial": cam.serial, "k": cam.k.tolist(), "dist": cam.dist.tolist(),
                                             "model": cam.model, "frame": frame},
                "base_from_page": used.tolist(), "extent_m": list(extent), "px_per_mm": args.px_per_mm, "poses": poses,
                "inner_edges_m": inner}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        cam.close()
        cam = None
        close = _closeup(io, kin, used, page, program, clear, out_dir, motion, args)
        print(json.dumps({"ok": bool(views) or close is not None, "dir": str(out_dir),
                          "overlays": [str(out_dir / f"overlay{n}.png") for n in range(len(views or []))],
                          "poses": len(poses), "closeup": close}))
        return 0 if views or close is not None else 1
    finally:
        if cam is not None:
            cam.close()
        if not getattr(args, "depth_only", False):
            _rest(io, kin if "kin" in locals() else None, args)
        executor.shutdown(timeout_sec=1.0)
        spinner.join(timeout=2.0)


def _jog(node: Node, args) -> int:
    """A quintic move of one joint by --delta from its measured position (ready.joint_move), through the
    arm's trajectory controller; an e-stop press mid-jog holds the arm like any goal. Bench M0."""
    import threading

    import numpy as np
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import JointState
    from tatbot_description import robot_description
    from tatbot_motion import Kinematics

    from tatbot_session.arm import ArmIO
    from tatbot_session.ready import joint_move

    if not 0 <= args.joint <= 6:
        print("--joint is 0-5 (rad) or 6 (the carriage, m)", file=sys.stderr)
        return 2
    if _goal_running(node, args.arm):
        print(f"{args.arm}: a trajectory goal is running (a draw or touch); jog after it ends", file=sys.stderr)
        return 1
    group = ReentrantCallbackGroup()
    io = ArmIO(node, args.arm, group)
    node.create_subscription(JointState, names.JOINT_STATES_TOPIC, io.on_joint_state, 20, callback_group=group)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        kin = Kinematics(robot_description(None, arms=(args.arm,)), args.arm)
        q0 = io.measured()
        io.wait(lambda: bool(io.safety), timeout=2.0)
        q1 = q0.copy()
        q1[args.joint] += args.delta
        traj = joint_move(kin.joint_names, q0, q1, max_rad_s=args.speed or 0.02, max_m_s=args.speed or 0.002,
                          rate_hz=100.0, kin=kin)
        tip_mm = float(np.linalg.norm(kin.fk(q1)[:3, 3] - kin.fk(q0)[:3, 3]) * 1000)
        print(f"{time.strftime('%H:%M:%S')} [{args.arm}] jog {kin.joint_names[args.joint]} {args.delta:+.4f} over "
              f"{traj.t[-1]:.1f} s: tip moves {tip_mm:.1f} mm", flush=True)
        status, _, text = io.execute(traj)
        q2 = io.measured()
        print(json.dumps({"ok": status == "ok", "status": status, "text": text, "joint": kin.joint_names[args.joint],
                          "delta": args.delta, "moved": float(q2[args.joint] - q0[args.joint]),
                          "tip_moved_mm": round(float(np.linalg.norm(kin.fk(q2)[:3, 3] - kin.fk(q0)[:3, 3])) * 1000, 2),
                          "latched": io.latched, "latch_reason": io.latch_text(), "q_before": q0.tolist(),
                          "q_after": q2.tolist()}))
        return 0 if status == "ok" else 1
    finally:
        executor.shutdown(timeout_sec=1.0)
        spinner.join(timeout=2.0)


def _decide(node: Node, args) -> int:
    client = node.create_client(Decide, names.DECIDE_SERVICE)
    if not client.wait_for_service(timeout_sec=10.0):
        print(f"{names.DECIDE_SERVICE}: no server (is the stack up?)", file=sys.stderr)
        return 1
    future = client.call_async(Decide.Request(arm=args.arm, decision=DECISIONS[args.decision]))
    _spin(node, future, 60.0)
    response = future.result()
    if response is None:
        print("no answer", file=sys.stderr)
        return 1
    print(json.dumps({"accepted": response.accepted, "reason": response.reason}))
    return 0 if response.accepted else 1


def _status(node: Node, args) -> int:
    states: dict[str, SafetyState] = {}
    node.create_subscription(SafetyState, names.SAFETY_TOPIC, lambda m: states.__setitem__(m.arm, m), 10)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    rows = {arm: {"estop_source": s.estop_source, "estop_ok": s.estop_ok, "estop_age_s": round(s.estop_age_s, 3),
                  "latched": s.latched, "latch_reason": LATCH_TEXT.get(s.latch_reason, s.latch_reason),
                  "guard_mode": s.guard_mode, "probe_triggered": s.probe_triggered, "landing": s.landing,
                  "landed": s.landed, "controller_error": s.controller_error, "machine_off": s.machine_off,
                  "rt_period_max_ms": round(s.rt_period_max_ms, 3)}
            for arm, s in sorted(states.items())}
    if args.json:
        print(json.dumps({"safety": rows}))
    else:
        for arm, row in rows.items():
            print(arm, " ".join(f"{k}={v}" for k, v in row.items()))
        if not rows:
            print("no /tatbot/safety (is the stack up?)")
    return 0 if rows else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tatbot_session client")
    sub = parser.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("draw")
    p.add_argument("program")
    p.add_argument("--arm", default="")
    p.add_argument("--from-op", default="")
    p.add_argument("--run-id", default="")
    p = sub.add_parser("touch")
    p.add_argument("--arm", default="right")
    p.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="one touch from this tcp position in <arm>/base_link (m), at the tool's measured orientation")
    p.add_argument("--rpy", type=float, nargs=3, metavar=("R", "P", "Y"),
                   help="the tcp's orientation at the start (URDF roll-pitch-yaw in <arm>/base_link, rad; pi 0 yaw points "
                        "the tool straight down); default the measured one")
    p.add_argument("--direction", type=float, nargs=3, metavar=("DX", "DY", "DZ"), default=(0.0, 0.0, -1.0))
    p.add_argument("--prior", type=float, default=0.0, help="the expected contact this far along the direction, m")
    p.add_argument("--probe", action="store_true", help="the station probe's guard (a touch needs --prior)")
    p.add_argument("--move", action="store_true", help="no touch: travel to --start and hold there")
    p.add_argument("--speed", type=float, default=0.0, help="the slow leg, m/s (0: motion.yaml)")
    p.add_argument("--max-travel", dest="max_travel", type=float, default=0.0, help="m (0: motion.yaml)")
    p = sub.add_parser("inspect")
    p.add_argument("--depth-only", action="store_true", help="capture native depth at the current held pose; no motion")
    p.add_argument("--arm", default="right")
    p.add_argument("--run", default="", help="a ros-draw run id (default: the newest)")
    p.add_argument("--logs", default="~/tatbot-logs/ros-draw")
    p.add_argument("--serial", default="", help="the wrist D405 (default: the one attached)")
    p.add_argument("--clearance", type=float, default=0.015, help="pen tip over the page at least, m")
    p.add_argument("--lift", type=float, default=0.040, help="lift the pen to this height first, m")
    p.add_argument("--poses", type=int, default=3)
    p.add_argument("--side-margin", dest="side_margin", type=float, default=0.0,
                   help="camera and wrist links stay this far on the arm's side of the design (base -y), m")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--settle", type=float, default=0.8)
    p.add_argument("--speed", type=float, default=0.3, help="joint speed, rad/s")
    p.add_argument("--px-per-mm", dest="px_per_mm", type=float, default=10.0)
    p.add_argument("--no-rest", dest="rest", action="store_false", help="stay at the last camera pose")
    p.add_argument("--no-closeup", dest="closeup", action="store_false",
                   help="skip the close-up ink holds (page.inspect.closeup) after the far views")
    p = sub.add_parser("jog")
    p.add_argument("--arm", default="right")
    p.add_argument("--joint", type=int, required=True, help="0-5 (rad) or 6, the carriage (m)")
    p.add_argument("--delta", type=float, required=True, help="rad, or m for the carriage")
    p.add_argument("--speed", type=float, default=None, help="peak speed: rad/s (default 0.02), or m/s for the carriage (0.002)")
    p = sub.add_parser("decide")
    p.add_argument("arm")
    p.add_argument("decision", choices=sorted(DECISIONS))
    sub.add_parser("land").add_argument("--arm", default="")
    sub.add_parser("wake").add_argument("--arm", required=True)
    sub.add_parser("cancel")
    sub.add_parser("status").add_argument("--json", action="store_true")
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv if argv is None else argv)[1:])
    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    # All three, SIGINT too: a job a script starts in the background inherits SIGINT ignored.
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)
    node = rclpy.create_node("tatbot_session_client")
    if args.verb in ("draw", "touch"):
        node.create_subscription(Event, names.EVENTS_TOPIC, _print_event, 50)
    try:
        return {"draw": _draw, "touch": _touch, "jog": _jog, "inspect": _inspect, "decide": _decide, "land": _land,
                "wake": _wake, "cancel": _cancel_all, "status": _status}[args.verb](node, args)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
