"""`ros2 run tatbot_rerun rerun_bridge`: the stack into the fleet Rerun viewer's standing recording.

Under draw/ros/, in `world` (the overhead camera's colour optical frame): the robot (each visual of the ros-stack run's
robot_description.urdf, world_from_link from tf on a timer, a link again when it moved), each arm's tip trail (tf on a
timer, never the 400 Hz /joint_states), the run's strokes (planned, done, current) in 3D and on the design map
(draw/ros/design, in the print's mm, over the print they are drawn on), the page, a nominal table and `level` (the arm
base, z up: the Demo 3D view's origin), the palette and caps, each arm's wrist D405 cloud from the tatbot bus, the
status board (draw/ros/status_board, tatbot_contracts.ros_status), its card (draw/ros/status) and events. A run is rebuilt from its files (ros_scene), so a
bridge (re)started mid-draw, or one replaying a bag on its clock (use_sim_time), draws the same scene, and nothing is
stamped before that clock runs. Display only: a failing handler logs once and is swallowed; a bridge that cannot start
idles; one whose hub went away exits 0 for the launch to respawn it (rerun never redials). The stream opens only through
$TATBOT_REPO/scripts/lib/tatbot_rerun.py start() (scripts/check rerun-caps); never a viewer.

Parameters (strings; '' = stack.yaml / the fleet viewer): config, arms, max_hz, connect, output.
"""
from __future__ import annotations

import functools
import sys
import threading
from collections import Counter

import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tatbot_bridge import stack
from tatbot_contracts import ros_scene as scene
from tatbot_contracts import ros_status as status
from tatbot_description import names, repo_root
from tatbot_description.transforms import quat_matrix
from tatbot_interfaces.action import Draw
from tatbot_interfaces.msg import Event, Page, SafetyState
from tf2_ros import Buffer, TransformException, TransformListener

ROOT = "draw/ros"
PAGE_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
PHASES = {getattr(Draw.Feedback, k): k[6:] for k in dir(Draw.Feedback) if k.startswith("PHASE_")}
LATCHES = {v: k[6:].lower() for k, v in vars(names).items() if k.startswith("LATCH_")}
LAYERS = ("plan", "design", "planned", "done", "page", "table", "palette")
KEYFRAME_S = 60.0   # every layer and link again this often: a viewer joining the hub's buffer late still gets them


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


def share(package: str):
    from ament_index_python.packages import get_package_share_directory
    try:
        return get_package_share_directory(package)
    except LookupError:
        return None


def shown(handler):
    """A display handler: its first error is logged, every error swallowed (returns None); nothing waits on it."""
    @functools.wraps(handler)
    def run(self, *args, **kwargs):
        try:
            return handler(self, *args, **kwargs)
        except Exception as error:  # noqa: BLE001
            if handler.__name__ not in self.failed:
                self.failed.add(handler.__name__)
                self.get_logger().warning(f"{handler.__name__}: {error!r} (logged once; display only)")
        return None
    return run


class RerunBridge(Node):
    def __init__(self):
        super().__init__(names.RERUN_NODE, start_parameter_services=False)   # six services fewer in each wait
        param = stack.string_parameters(self, ("config", "arms", "max_hz", "connect", "output"))
        self.stack, self.config_dir = stack.load(param["config"]), stack.stack_path(param["config"]).parent
        self.arms = stack.arms(param["arms"], self.stack)
        hz = float(param["max_hz"] or self.stack.get("rerun", {}).get("max_hz", 20))
        page_cfg = self.stack["page"]
        self.size, self.clear = page_cfg["size_m"], page_cfg["clear_m"]
        self.fitted = (page_cfg.get("surface") or {}).get("mode") == "fitted"
        self.repo = repo_root()
        connect = param["connect"] or None
        if not connect and not param["output"]:
            nodes = stack.fleet()
            connect = nodes.rerun_proxy(nodes.load(self.repo))
        self.tr = rerun_helper()
        import tatbot_runlog  # scripts/lib, on sys.path now: the session's own log root (TATBOT_LOG_ROOT, runlog.json)
        self.log_root = tatbot_runlog.log_root(tatbot_runlog.load_config())
        self.rr = self.tr.start("ros", connect=connect, output=param["output"] or None,
                                recording_id=self.tr.STANDING_RECORDING, view_coordinates=False)
        if connect:
            self.tr.watch_sink(self.get_logger().warning)
        self.paths = {arm: scene.TipPath() for arm in self.arms}
        self.failed, self.dirty, self.logged, self.poses, self.safety, self.wrist = set(), set(LAYERS), {}, {}, {}, {}
        self.page_msg = self.run = self.feedback = self.surface = self.palette = self.caps = self.highlight = None
        self.frame, self.tip_shown = None, False   # the design map's frame (design_frame); a tip dot drawn in it
        self.keyframe, self.wrist_failed, self.followed, self.board = self._now_s(), False, False, {}
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_subscription(Page, names.PAGE_TOPIC, self._on_page, PAGE_QOS)
        self.create_subscription(Event, names.EVENTS_TOPIC, self._on_event, 50)
        self.create_subscription(SafetyState, names.SAFETY_TOPIC, self._on_safety, 10)
        # Draw feedback, passively: the action's own feedback topic, never an action client
        self.create_subscription(Draw.Impl.FeedbackMessage, f"{names.DRAW_ACTION}/_action/feedback", self._on_feedback, 10)
        self.rr.log(f"{ROOT}/robot", self.rr.Clear(recursive=True), static=True)
        robot = self.config_dir / "robot_description.urdf"
        self.links = self._meshes(f"{ROOT}/robot", scene.urdf_visuals, robot, packages=share)
        self._meshes(f"{ROOT}/palette/mesh", scene.palette_visuals, self.repo, robot)   # in the arm's black, with its tag
        self.create_timer(1.0 / hz, self._on_tip)
        self.create_timer(1.0 / float(self.stack.get("rerun", {}).get("robot_hz", 12)), self._on_robot)
        self.create_timer(0.5, self._on_tick)
        threading.Thread(target=self._open_wrist, daemon=True).start()
        self.get_logger().info(f"rerun: {connect or param['output']} at <= {hz:g} Hz under {ROOT}/")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _now(self):
        self.tr.set_capture_time(self.get_clock().now().nanoseconds)

    def _lookup(self, target: str, stamp: Time | None = None):
        try:
            t = self.buffer.lookup_transform(names.WORLD, target, Time() if stamp is None else stamp)
        except TransformException:
            return None
        tr, q = t.transform.translation, t.transform.rotation
        return quat_matrix([tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w])

    def _pose(self, entity: str, pose, static: bool = False, **kwargs):
        self.rr.log(entity, self.rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3], **kwargs), static=static)

    @shown
    def _meshes(self, prefix: str, visuals, *args, **kwargs) -> list[str]:
        """visuals(*args, **kwargs)'s meshes and cylinders (scene.urdf_visuals' items), logged once (static) under
        prefix/<link>/; returns their links."""
        visuals = visuals(*args, **kwargs)
        for v in (v for v in visuals if v["kind"] == "mesh"):
            entity = f"{prefix}/{v['link']}/{v['name']}"
            self._pose(entity, v["link_from_visual"], scale=v["scale"], static=True)
            contents, media = scene.mesh_asset(v["path"])   # a GLB without its samplers; else the file
            asset = {"contents": contents, "media_type": media} if media else {"path": v["path"]}
            self.rr.log(entity, self.rr.Asset3D(**asset, albedo_factor=v["rgba"]), static=True)
        for link, batch in scene.cylinders(visuals).items():
            self.rr.log(f"{prefix}/{link}/cylinders", self.rr.Cylinders3D(**batch), static=True)
        return sorted({v["link"] for v in visuals})

    @shown
    def _on_robot(self):
        # each link when it moved, and every KEYFRAME_S (self.poses is reset then): a link fixed to world is otherwise
        # logged once, and a viewer joining after the hub dropped that row would draw it at the camera
        self._now()
        for link in self.links or ():
            pose = self._lookup(link)
            if pose is not None and scene.moved(self.poses.get(link), pose):
                self.poses[link] = pose
                self._pose(f"{ROOT}/robot/{link}", pose)

    def _page_pose(self):
        """world <- the page in use: tf world -> <arm>/base_link and the newest page row's `used`; before the
        first page row, tf world -> page/<pattern_id> from /tatbot/page."""
        if self.run is not None and (self.run.page or {}).get("used"):
            base = self._lookup(names.base_frame(self.run.arm))
            return None if base is None else base @ np.asarray(self.run.page["used"], float)
        return None if self.page_msg is None else self._lookup(names.page_frame(self.page_msg.pattern_id))

    @shown
    def _on_tip(self):
        self._now()
        for arm, path in self.paths.items():
            tcp = self._lookup(names.tcp_frame(arm))
            if tcp is not None and path.add(tcp[:3, 3]):
                self.rr.log(f"{ROOT}/{arm}/tip/{path.index:05d}", self.rr.LineStrips3D([path.array()], radii=0.0003))
                self.rr.log(f"{ROOT}/{arm}/tcp", self.rr.Points3D([tcp[:3, 3]], radii=0.002))
                page = self._page_pose() if self.run is not None and self.run.arm == arm else None
                if page is not None:
                    self._design_tip(scene.design_mm(   # the tip on the print: plus where the newest stroke moved it
                        scene.transform(np.linalg.inv(page), [tcp[:3, 3]])[:, :2] + self.run.shift, self.size))

    def _design_tip(self, tip):
        """The tip dot on the design map, only inside the design's frame: off it (travel, rest) it would widen the
        view's fit and push the design off centre."""
        if self.frame is not None and scene.within(self.frame, tip[0]):
            self.rr.log(f"{ROOT}/design/tip", self.rr.Points2D(tip, radii=0.6, colors=[scene.CURRENT_RGB]))
            self.tip_shown = True
        elif self.tip_shown:
            self.rr.log(f"{ROOT}/design/tip", self.rr.Clear(recursive=False))
            self.tip_shown = False

    @shown
    def _on_page(self, msg: Page):
        self.page_msg = msg
        if self.run is None or not self.run.page:
            self.dirty.add("page")

    @shown
    def _on_event(self, msg: Event):
        self._now()
        self.rr.log(f"{ROOT}/events", self.rr.TextLog(f"[{msg.arm or '-'}] {msg.kind} {msg.op_id} {msg.text}".strip()))
        if msg.kind == Event.KIND_RUN_END:
            self.feedback = None
        start = msg.kind == Event.KIND_RUN_START
        if msg.run_id and (start or self.run is None or msg.run_id != self.run.run_dir.name):
            self._follow(msg.run_id, clear_tips=start)

    @shown
    def _follow(self, run_id: str | None, clear_tips: bool = False):
        """Show a ros-draw run (the newest when run_id is None), rebuilt from its files. Another workflow's run (a
        touch's events) leaves the scene as it is."""
        runs, self.followed = self.log_root / names.RUN_WORKFLOW_DRAW, True
        run = scene.DrawState.load(runs / run_id if run_id else scene.newest_run(runs, self._now_s()) or runs / "-")
        if run is None:
            return
        self.run, self.surface, self.feedback = run, None, None
        for arm, path in self.paths.items() if clear_tips else ():
            path.clear()
            self.rr.log(f"{ROOT}/{arm}/tip", self.rr.Clear(recursive=True))
        for layer in ("strokes", "design", "status_board/arc"):   # the retired arc cell: the trim's now
            self.rr.log(f"{ROOT}/{layer}", self.rr.Clear(recursive=True))
        self.dirty |= set(LAYERS)

    @shown
    def _on_safety(self, msg: SafetyState):
        self.safety[msg.arm] = {
            "t": Time.from_msg(msg.stamp).nanoseconds / 1e9 or self._now_s(), "estop_source": msg.estop_source,
            "estop_ok": msg.estop_ok, "estop_age_s": msg.estop_age_s, "probe_triggered": msg.probe_triggered,
            "latched": msg.latched, "latch": LATCHES.get(msg.latch_reason, str(msg.latch_reason)),
            # SafetyState's MCS1 reply fields; None on an interface that predates them
            "machine_powered": getattr(msg, "machine_powered", None),
            "machine_reply_age_s": getattr(msg, "machine_reply_age_s", None)}
        state = (msg.estop_ok, msg.latched, msg.latch_reason, msg.guard_tripped, msg.landing, msg.landed)
        if self.logged.get(f"safety/{msg.arm}") != state:
            self.logged[f"safety/{msg.arm}"] = state
            self._now()
            self.rr.log(f"{ROOT}/{msg.arm}/safety", self.rr.TextLog(
                f"estop_ok={int(msg.estop_ok)} latched={int(msg.latched)} reason={msg.latch_reason} "
                f"guard_tripped={int(msg.guard_tripped)} landing={int(msg.landing)} landed={int(msg.landed)}",
                level="WARN" if msg.latched else "INFO"))

    @shown
    def _on_feedback(self, msg):
        fb = msg.feedback
        if self.run is None or fb.arm != self.run.arm:
            return
        self.feedback = {"t": self._now_s(), "op_id": fb.op_id, "index": fb.index, "total": fb.total,
                         "phase": PHASES.get(fb.phase, str(fb.phase)), "arc_m": fb.arc_m}
        current = self.run.current(fb.op_id, fb.arc_m) if fb.phase == Draw.Feedback.PHASE_DRAW else None
        key = None if current is None else (fb.op_id, round(fb.arc_m, 4))
        if key == self.logged.get("current"):
            return
        self.logged["current"] = key
        self._now()
        page = self._page_pose()
        world = [] if current is None or page is None else [scene.transform(page, scene.page_local(
            self.run.on_page(current), self.surface))]
        self.rr.log(f"{ROOT}/strokes/current", self.rr.LineStrips3D(world, colors=[scene.CURRENT_RGB], radii=0.0003))
        self.rr.log(f"{ROOT}/design/current", self.rr.LineStrips2D(
            [] if current is None else [scene.design_mm(current, self.size)], colors=[scene.CURRENT_RGB], radii=0.3))

    @shown
    def _on_tick(self):
        """2 Hz: the ledger's new rows, the layers they changed (each again every KEYFRAME_S), the status. The first
        tick follows the newest run: under use_sim_time a timer waits for /clock, so no Clear is stamped 1970."""
        self._now()
        now = self._now_s()
        if not self.followed:
            self._follow(None)
        if self.run is not None:
            changed = self.run.poll(now)
            if "page" in changed:
                self.surface = scene.page_surface(self.run.page, self.fitted)
            self.dirty |= changed & set(LAYERS)
        index = self.feedback["index"] if self.feedback else self.run and self.run.next_op()
        if self.run is not None and self.run.dip_slot(index) != self.highlight:
            self.highlight = self.run.dip_slot(index)
            self.dirty.add("palette")
        if now - self.keyframe > KEYFRAME_S:
            self.keyframe, self.dirty, self.poses, self.board = now, self.dirty | set(LAYERS), {}, {}
        for layer in sorted(self.dirty):
            if getattr(self, f"_log_{layer}")() is not False:
                self.dirty.discard(layer)
        self._log_status(now, index)

    def _drawn(self):
        """Every done or partial stroke's measured lines, with their ink colours."""
        drawn = [(op, line) for op, lines in self.run.lines.items() for line in lines]
        return [line for _, line in drawn], [self.run.colour(op) for op, _ in drawn]

    @shown
    def _log_plan(self):
        """The design map, page mm like drawn.svg: the print, its clear centre and the planned strokes (once a run)."""
        if self.run is not None:
            outline = [scene.design_mm(scene.rectangle(*s)[:, :2], self.size) for s in (self.size, self.clear)]
            self.rr.log(f"{ROOT}/design/page", self.rr.LineStrips2D(outline, colors=[scene.PAGE_RGB], radii=0.2))
            planned = [scene.design_mm(s, self.size) for s in self.run.planned.values()]
            self.rr.log(f"{ROOT}/design/planned", self.rr.LineStrips2D(planned, colors=[scene.PLANNED_RGBA], radii=0.15))
            self.frame = scene.design_frame(planned)
            if self.frame is not None:   # what the Demo's Design view fits to (it leaves the print outline out)
                self.rr.log(f"{ROOT}/design/frame", self.rr.LineStrips2D([self.frame], colors=[(70, 76, 84)], radii=0.1))
            self._log_stencil()

    @shown
    def _log_stencil(self):
        """The print the run draws on (its installed stencil.png) under the design map, cut to the frame; none on a
        fixed page or with no print installed."""
        import cv2
        import stencil_reference as ref

        path = ref.observer_references(self.log_root) / str(
            self.run.pattern_id or self.stack["page"].get("pattern_id")) / "tracking.json"
        manifest, image = ref.load(path) if path.is_file() else (None, None)
        under = manifest and scene.stencil_underlay(cv2.imread(str(image), cv2.IMREAD_GRAYSCALE), manifest["page_mm"],
                                                    self.size, self.frame)
        if under:
            png = cv2.imencode(".png", cv2.cvtColor(under["rgba"], cv2.COLOR_RGBA2BGRA))[1].tobytes()
            self.rr.log(f"{ROOT}/design/stencil", self.rr.Transform3D(translation=under["translation"], scale=under["scale"]),
                        self.rr.EncodedImage(contents=png, media_type="image/png", draw_order=-20.0,   # under every line
                                             magnification_filter="Linear"))

    @shown
    def _log_design(self):
        """The design map's done strokes, in their inks."""
        if self.run is not None:
            lines, colors = self._drawn()
            self.rr.log(f"{ROOT}/design/done", self.rr.LineStrips2D(
                [scene.design_mm(line, self.size) for line in lines], colors=colors, radii=0.2))

    @shown
    def _log_planned(self):
        """The program's strokes in 3D, a translucent ghost (False: no page pose yet, try again)."""
        page = self._page_pose()
        if self.run is not None and page is not None:
            self.rr.log(f"{ROOT}/strokes/planned", self.rr.LineStrips3D(
                [scene.transform(page, scene.page_local(self.run.on_page(s), self.surface)) for s in self.run.planned.values()],
                colors=[scene.PLANNED_RGBA], radii=0.00015))
        return self.run is None or page is not None

    @shown
    def _log_done(self):
        """Every done or partial stroke in 3D as the line the pen measured, in its ink, on the page in use."""
        page = self._page_pose()
        if self.run is not None and page is not None:
            lines, colors = self._drawn()
            self.rr.log(f"{ROOT}/strokes/done", self.rr.LineStrips3D(
                [scene.transform(page, scene.page_local(self.run.on_page(line), self.surface, chart=False)) for line in lines],
                colors=colors, radii=0.0002))
        return self.run is None or page is not None

    @shown
    def _log_page(self):
        """The page in use: its edge, clear centre and a 10 mm grid on its plane or fitted surface."""
        page = self._page_pose()
        if page is None:
            return self.page_msg is None and not (self.run and self.run.page)
        lost = not (self.run and self.run.page) and self.page_msg.source == Page.SOURCE_LOST
        lines = scene.page_lines(self.size, self.clear, self.surface)
        self.rr.log(f"{ROOT}/page", self.rr.LineStrips3D([scene.transform(page, line) for line in lines],
                    colors=[scene.LOST_RGB if lost else scene.PAGE_RGB], radii=[0.0004] * 2 + [0.0001] * (len(lines) - 2)))
        return True

    @shown
    def _log_table(self):
        # the nominal table; draw/ros/level (static, logged when the base moved): the base frame with z up, the Demo
        # 3D view's origin, so the table is level on screen whichever way the camera looks
        base = self._lookup(names.base_frame(self.run.arm if self.run else self.arms[0]))
        if base is None:
            return False
        if not np.array_equal(self.logged.get("level"), base):
            self.logged["level"] = base
            self._pose(f"{ROOT}/level", base, static=True)
            self.rr.log(f"{ROOT}/level", self.rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self.rr.log(f"{ROOT}/table", self.rr.LineStrips3D(
            [scene.transform(base, line) for line in scene.table_grid()], colors=[(150, 150, 150, 70)], radii=0.0002))
        self.rr.log(f"{ROOT}/table/label", self.rr.Points3D(scene.transform(base, [[0.6, -0.45, 0.0]]),
                                                            labels=["nominal table"]))
        return True

    @shown
    def _log_palette(self):
        """The palette at the newest station fix carried onto the station touch, and its caps by state, fill and label
        (its print and tag are palette/mesh, static)."""
        arm = self.run.arm if self.run else self.arms[0]
        fix = scene.palette_fix_path(self.log_root, self.run and self.run.run_dir, self._now_s())
        base = self._lookup(names.base_frame(arm))
        if fix is None or base is None:
            return fix is None
        self.palette = scene.palette_pose(self.repo, arm, fix)
        self._pose(f"{ROOT}/palette", base @ (pose := self.palette["base_from_palette"]))
        self.caps = scene.caps(self.repo, pose, self.run.dips if self.run else Counter())
        for name, (kind, args) in scene.palette_layers(self.caps, pose, self.highlight).items():   # caps, ink, marks, …
            self.rr.log(f"{ROOT}/palette/{name}", getattr(self.rr, kind)(**args))
        return True

    @shown
    def _log_status(self, now: float, index):
        """The board's items that changed (all again every KEYFRAME_S), and the card."""
        arm = self.run.arm if self.run else self.arms[0]
        items = status.items(now, run=self.run and self.run.summary(), feedback=self.feedback,
                             ink=self.run and self.run.ink(index), trim=self.run and self.run.trim(index),
                             safety=self.safety.get(arm), wrist=self.wrist.get(arm), caps=self.caps,
                             palette_s=self.palette and self.palette["measured_s"],
                             probe_enabled=bool((self.stack.get("probe") or {}).get("enabled")),
                             machine_relay=(self.stack.get("machine") or {}).get("switch", "none") != "none")
        for key in [key for key, item in items.items() if self.board.get(key) != item]:
            self.board[key] = items[key]
            self.rr.log(f"{ROOT}/status_board/{key}", self.rr.Boxes2D(**status.box(key, items[key])))
        self.rr.log(f"{ROOT}/status", self.rr.TextDocument(status.card(items, self.run and self.run.run_dir.name),
                                                           media_type=self.rr.MediaType.MARKDOWN))

    def _open_wrist(self):
        """Each arm's wrist D405s from the tatbot bus (the session's tap and its previews:
        tatbot/vision/d405/<camera>), on its own thread; any failure turns the feed off. The one page.locate.camera
        names is <arm>/wrist_d405 and the status board's wrist; another is <arm>/wrist_d405_<its role less wrist_>."""
        try:
            import wrist_cameras

            named = ((self.stack.get("page") or {}).get("locate") or {}).get("camera")
            feeds = {f"tatbot/vision/d405/{cam['name']}": (arm, "wrist_d405" + (
                "" if cam["role"] == named else "_" + cam["role"].removeprefix("wrist_")))
                for arm in self.arms for cam in wrist_cameras.mounted_for_arm(self.repo, arm)}
            self.limits = {key: self.tr.RateLimiter(1.0) for key in feeds}
            self.bus = stack.open_bus(stack.bus_endpoint())
            self.wrist_subs = [self.bus.declare_subscriber(key, functools.partial(self._on_wrist, key, *feed))
                               for key, feed in feeds.items()]
        except Exception as error:  # noqa: BLE001
            self._wrist_off(error)

    def _wrist_off(self, error):
        if not self.wrist_failed:
            self.wrist_failed = True
            self.get_logger().warning(f"wrist D405 feed off: {error!r}")

    def _on_wrist(self, key: str, arm: str, entity: str, sample):
        """A zenoh thread: at most 1 Hz per camera, the newest frame's depth placed by tf at its stamp."""
        if self.wrist_failed or not self.limits[key].ready():
            return
        try:
            from frame_set_wire import unpack

            frame = scene.wrist_cloud(unpack(sample.payload.to_bytes())[1], decode=_decode_jpeg)
            if frame is None:
                return
            pose = self._lookup(frame["ros_frame"], Time(nanoseconds=frame["stamp_ns"]))
            latest = pose is None
            pose = self._lookup(frame["ros_frame"]) if latest else pose
            if pose is None:
                return
            self.tr.set_capture_time(frame["stamp_ns"])   # this thread's own timeline cursor
            self._pose(f"{ROOT}/{arm}/{entity}", pose)
            height, width = frame["depth"].shape
            # The frustum stays at depth/ (its DepthImage cleared: an earlier release's colormapped cloud would
            # still be drawn); the RGB cloud goes in rows beside it, cloud/<i>, each within the hub's row budget.
            self.rr.log(f"{ROOT}/{arm}/{entity}/depth", self.rr.Pinhole(
                focal_length=frame["focal"], principal_point=frame["principal"], width=width, height=height,
                image_plane_distance=0.03), self.rr.DepthImage.cleared())
            for index, (points, colors, radius) in enumerate(frame["bands"]):
                self.rr.log(f"{ROOT}/{arm}/{entity}/cloud/{index}",
                            self.rr.Points3D(points, colors=colors, radii=radius))
            if entity == "wrist_d405":
                self.wrist[arm] = {"t": frame["stamp_ns"] / 1e9, "latest": latest}
        except Exception as error:  # noqa: BLE001
            self._wrist_off(error)


def _decode_jpeg(data: bytes):
    """The wrist tap's colour JPEG as RGB, or None: the stack's cv2 (apt), imported where it is used so the bridge
    runs, tint-coloured, on a node without it."""
    try:
        import cv2
    except ImportError:
        return None
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if bgr is None else bgr[:, :, ::-1]


def main(argv=None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    try:
        node = RerunBridge()
    except Exception as error:  # noqa: BLE001  (no SDK or no sink: say so once and idle, so nothing respawns it)
        node = Node(f"{names.RERUN_NODE}_idle")
        node.get_logger().error(f"tatbot_rerun not started: {type(error).__name__}: {error} (display only; the stack "
                                "runs on without it)")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0
