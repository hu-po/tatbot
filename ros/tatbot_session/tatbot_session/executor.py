"""One arm's executor: page setup and touches, the stroke loop, tool changes and palette dips, pauses, latches
and decisions, landing.

The next op is always the first op whose last ledger event is not done/skipped. Each stroke is planned by tatbot_motion
from the measured joints (the next stroke is pre-planned while the current one runs, then checked for drift at
dispatch) and sent as one FollowJointTrajectory goal. A riding pen's tattoo machine runs only inside the op loop: on
before each stroke goal, off whenever the loop waits for a decision, dips or ends, and at once when the arm latches
(the node). A tool change that needs a different cartridge lands the arm for the operator; resuming the run is the
acknowledgement. The operator's pen trim (the numpad) is added to every stroke goal as it is sent, and replaces the
running one ahead of now when it changes; a needle cartridge's goes no deeper than the skin read along the stroke
lets it (_cap_for). A dipping cartridge's charge (charge.py) adds, moves or skips dips around the program's, and the
numpad's + dips at once. ros/README.md 4.3, 4.4, 6, 8.
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import itertools
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tatbot_description import names

from tatbot_session import charge, closeup, config, curved, gauge, geometry, machine, ready, rules, track
from tatbot_session import ledger as ledger_mod
from tatbot_session.lease import in_cap, in_cap_refusal
from tatbot_session.ports import WALL, Event, Phase

P = Phase   # Draw.Feedback's PHASE_* numbers
# A flat page whose wrist gauge points stand farther than this off their plane is a curved skin: refused, never
# drawn flat (on 2026-10-13 a flat page on the curved holder rode high at its edges and deep in its middle).
CURVED_PAGE_M = 0.0015


def motion_result(status, text, activity, allowed=('ok',)):
    if status == 'cancelled':
        raise Cancelled()
    if status not in allowed:
        raise RuntimeError(f'{activity}: {status}: {text}')


def _rinses(resource: dict) -> bool:
    """The resource takes the cartridge from the last ink by a rinse in a cap of water, not a landing."""
    return (resource.get("activation") or {}).get("method") == "rinse"


class Landed(Exception):  # noqa: N818 - control flow, not an error
    """The arm landed (a decision, a Land request, or the resume timeout); the goal ends."""


class Cancelled(Exception):  # noqa: N818 - control flow, not an error
    """The goal was cancelled; the arm holds where it is."""



def _median_pose(base: np.ndarray, poses) -> np.ndarray:
    """The median of page poses about `base`: their centres' displacement and turn in base's own page frame, each
    the median (sheet_slide's way), as a pose."""
    moves = [inv @ pose for inv in [np.linalg.inv(base)] for pose in poses]
    turn, local = float(np.median([_yaw(m) for m in moves])), np.eye(4)
    local[:2, :2] = [[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]
    local[:2, 3] = np.median([m[:2, 3] for m in moves], axis=0)
    return base @ local

def _yaw(m: np.ndarray) -> float:
    """The turn of a pose (or a motion) about its z axis, rad."""
    return float(np.arctan2(m[1, 0], m[0, 0]))


def _median_move(moves) -> tuple[float, float, int, float] | None:
    """(slide m, turn rad, readings, rise m): the medians of geometry.marks_moved's moves; None without one."""
    if not moves:
        return None
    return (float(np.median([m[0] for m in moves])), float(np.median([m[1] for m in moves])), len(moves),
            float(np.median([m[3] for m in moves])))


def _unconfirmed(page: dict) -> bool:   # a page row its holds moved past the curved limit, no check holds confirming it
    return bool((holds := page.get("holds") or {}).get("confirm")) and not holds.get("stands")


# Where a page row's holds placed the page, in its plane, from the wrist's own page then (its locate's prior moved by the
# wrist's estimate): the gauge tip's offset across the pen axis among it. Eye when its wrist read no print. With the
# turn of the row's ready heading in that page (None without one): the offset turns with the pen (_turn_carried).
def _held_off(page: dict) -> tuple[np.ndarray, float | None]:
    if not (w := (locate := page.get("locate") or {}).get("wrist")) or locate.get("overhead") is None:
        return np.eye(4), None
    seen = geometry.rpy_matrix([w["tx_m"], w["ty_m"], 0.0], [0.0, 0.0, w["theta_rad"]])   # inspect.correction_matrix
    own = np.asarray(locate["overhead"], float) @ np.linalg.inv(seen)
    d = np.linalg.inv(own) @ np.asarray(page["camera_at_setup"], float) @ np.asarray(page["correction"], float)
    held = (page.get("ready") or {}).get("heading")
    return (geometry.rpy_matrix([d[0, 3], d[1, 3], 0.0], [0.0, 0.0, float(np.arctan2(d[1, 0], d[0, 0]))]),
            None if held is None else ready.heading_yaw(own, np.asarray(held, float)))


def _dense(polylines, step: float = 0.001) -> np.ndarray:
    """The polylines' points (page metres) no farther than `step` apart along them."""
    out = []
    for p in polylines:
        p = np.asarray(p, float).reshape(-1, 2)
        out += [a + (b - a) * np.linspace(0.0, 1.0, max(1, int(np.ceil(np.linalg.norm(b - a) / step))),
                                          endpoint=False)[:, None] for a, b in zip(p[:-1], p[1:], strict=True)] + [p[-1:]]
    return np.concatenate(out)


def _past(moved, translation_m: float, rotation_rad: float) -> bool:
    """Whether a sheet's move (_median_move) is past a limit: its slide and rise along the page's normal together
    (a skin lifted toward the overhead reads ~0.7 of the lift as a slide), or its turn."""
    return float(np.hypot(moved[0], moved[3])) > translation_m or abs(moved[1]) > rotation_rad


class Ask:
    """One Decide call handed to a waiting executor. It answers only the wait it was made for: an Ask
    left over from an earlier wait, or one its caller gave up on, is never applied."""

    def __init__(self, decision: int, wait_id: int):
        self.decision, self.wait_id = decision, wait_id
        self.reply: queue.Queue = queue.Queue()
        self.taken = self.abandoned = False


class ArmExecutor:
    def __init__(self, node, io, kin, motion: dict, stack: dict, repo, *, clock=None, sensing=None, planner=None):
        """node, io: ports.NodePort and ArmPort; clock: ports.Clock (the wall clock by default); sensing: ports.Sensing
        (by default stack.yaml sensing.wrist's at each use, gauge.sensing); planner: where the next stroke is pre-planned (an
        executor with submit(), one worker thread by default)."""
        self.node, self.io, self.kin, self.motion, self.stack, self.repo = node, io, kin, motion, stack, repo
        gauge.wrist(stack)   # a sensing.wrist this hardware cannot take refuses here, as the node starts, not mid-draw
        self.clock = clock or WALL
        self.ride_base = float(motion["pen"]["ride"]["fraction"])   # motion.yaml's, which draw() reloads
        self.arm = io.arm
        self.busy = threading.Lock()
        self._lock = threading.Lock()  # guards waiting/_wait_id/goal_active/land_requested and every Ask hand-off
        self.decisions: queue.Queue = queue.Queue()
        self.waiting: str | None = None
        self._wait_id = 0
        self.goal_active = False
        self.land_requested = False
        self.cancelled = lambda: False
        self.feedback = lambda **_: None
        self.run_dir = None   # set by the node per goal: where locate views are saved
        self.tracker = None   # stencil tracking (track.py) while a draw's ops run
        self._chain_snap = None   # the print correction a pen-down chain is planned with (track.Tracker.snapshot)
        self._stop_why = None   # why the stroke drawing now stops: the sheet moved under it (_moved_under)
        self._correction = None   # (mean, largest) correction of the op last dispatched, m
        self._goal_touch = False   # the draw goal's touch flag, for a page set up again mid-draw
        # the next setup_page sets the page up again mid-draw (_reacquire): _kept_page; and when _new_page set one up
        # again, its start (clock utc): the track's seed is that set-up's holds
        self._again, self.seed_since = False, 0.0
        self.tool_id = ""
        self.machine = None   # the tattoo machine's switch (tatbot_session.machine), set by the node; None: none
        self.pen = None       # the draw's tatbot_motion.PenDown; None: motion.yaml's press
        self._snag = None     # the snag guard level this executor last wrote (snag_guard); None: not yet written
        self.camera: np.ndarray | None = None
        self.overhead: np.ndarray | None = None   # the overhead's own pose of the page at setup, before any locate
        self.overhead_at = 0.0                    # when (self.clock) setup took it
        self.marks = None     # where the overhead read the print's markers at setup (_mark_reference); None: its pose
        self.marked_at = 0.0  # when (self.clock) it read them so: the sheet watch's start (_sheet_watch)
        self._moved = (None, None, {})   # _timed_moves' cache: its reference, the page's normal, reading id -> move
        self._printed = None  # the print's markers' printed corners, id -> (4, 2) page metres (_layout)
        self.inked_at = 0.0                       # when (self.clock) the last stroke's goal ended
        self.placed = None   # {"page", "overhead"}: the last page the gauge holds placed, and the overhead's pose then
        self._resumed = None  # the goal's retained page row (_retained_page): a blind resume starts there (_last_page)
        self.slid = np.eye(4)   # the sheet's slide the page has followed since setup, in the arm base (_follow_sheet)
        self.surface_field = None   # a curved page's height field over the page in use (page.surface.mode fitted)
        self._surface_chart = None  # its chart, unfolded once per field
        self._patches: dict[int, np.ndarray] = {}   # each gauge hold's surface points in the base (curved.hold_patch)
        self._hold_frame = None   # the page a curved page's gauge holds were made on: its check holds are made on it
        self._hold_low = 0.0      # how far under touch.start_above_m over that page a curved page's holds stand
        self._provisional = None  # a curved page's skin from its first hold: the others stand along its normals
        self._start = None        # (that skin, its patch's hull over the page) when firm: the others start over it
        self._hop_from = None     # (touch, the commanded tip in the base) of the gauge hold the pen was last held at
        self._marks_seen: list = []   # the overhead's marker readings as a first setup began (_marks_over)
        self._over_skin: list = []    # the setup's over-centre locate views' depth, tip and tcp (_skin_seen, _skin_over)
        self._located_heading = None  # the heading those views went at, searched for the ready move (_ready_for)
        self.anchor_offset = None     # the skin under the last setup's first curved hold less those markers' height
        self.rigid_total = (np.eye(3), np.eye(3))   # a curved page's rigid follows since its holds (plane, print mm)
        self._skin_off = 0.0        # the side the last curved hover read the skin past skin_stop_m (_skin_watch)
        self._skins: list = []      # the page in use's curved hover readings: (clock, page xy, skin over the field, op)
        self._ride = ("", 0.0)      # (op id, m): how much higher than its plan that stroke rides (_ride_over)
        self._ridden = None         # (chain op ids, print points, m over the ride): its skin's profile along it
        self._preplan_ridden = None  # the profile the pre-plan was planned on
        self._set_up_for = ""       # the stroke whose skin along its path set the page up again (_skin_watch)
        self._skin_floor: dict = {}   # side -> net: the skin a fresh page's first hover read past its stop (_skin_watch)
        self._fresh = False         # the page was set up and its first hover has not read the skin yet
        self._skin_only = False     # hover stops read the skin alone: the wrist reads the print nowhere (_hover_stop)
        self._unread = 0            # hovers in a row that read no print (_unread_hover)
        self._track_since = 0.0     # the clock utc the tracker took its gauge holds from (_track_lift starts it again)
        self.correction = np.eye(4)
        self._carried_yaw = None    # the heading (turn in its page) of the last page whose holds' offset is carried
        # The last touched correction, carried into the next setup as its starting page: the
        # fixed-camera height sat 23-27 mm above the touched paper on the pink arm (bench
        # 2026-09-26), which made every slow 0.5 mm/s touch leg ~60 s long.
        self.prior_correction = np.eye(4)
        self.carried: dict | None = None   # where the last accepted page touches found the paper (_carry_paper)
        self.page_record: dict = {}
        self.resource: dict | None = None   # the program resource installed now (tool, ink, supply)
        self.charged: set[str] = set()      # dipping resources dipped during this goal; a resume dips again
        self.drawn: dict[str, list[list]] = {}
        self.trims: dict[str, float] = {}   # resource id -> the operator's pen trim (m), kept in the run's ledger
        self.trim_end = 0.0                 # the trim the last stroke goal left the arm at
        self._trim_cap = None               # the deepest trim the stroke in hand takes (_cap_for); None: any
        self._fold = 0.0                    # the ride the stroke goal's pre-plan was sent by (_ride_fold)
        self.ledger = None                  # the running draw's, where trims are written
        self.dip_scales: dict[str, float] = {}   # resource id -> the operator's dip scale (charge.py), in the ledger
        self.inked_m = 0.0                  # the line drawn since the last dip in this goal
        self.dip_asked = False              # the numpad's + asked for a dip that has not begun
        self.dipping = False                # a dip runs (dip)
        self.op_now = ""                    # the op the draw's loop runs
        self._arc_now = 0.0                 # the planned arc the running stroke goal has reached
        self._planner = planner or ThreadPoolExecutor(max_workers=1)
        self._preplan = None
        self._next_hold = None   # the next gauge hold's stand, as its loop will ask it (_plan_ahead)
        self._ahead = None       # (that stand, the hop to it planned while this hold read) (_ahead_for)
        self._land_hold = None   # (joints, snag, told a person) of a landing refused: no lift off the skin (_lift_clear)
        self._cam = None      # the wrist D405 a goal's setup reads by (setup_camera)
        self._writes, self._later = ThreadPoolExecutor(max_workers=1), []   # a setup's frames written, prints read
        self._reads = ThreadPoolExecutor(max_workers=self.READ_WORKERS)     # (_settle), each camera's print apart
        self.knot_rate = float(stack["session"]["knot_rate_hz"])
        self._sensing, self._stack_sensing = sensing, None

    # --- plumbing ------------------------------------------------------------------------------
    def event(self, kind: str, text: str = "", op_id: str = "") -> None:
        if kind == Event.KIND_OP_SENT:
            self.io.op_id = op_id   # tags the goals it sends on /tatbot/goals, for the bag
        elif kind in (Event.KIND_OP_DONE, Event.KIND_OP_ABORTED):
            self.io.op_id = ""
        self.node.event(kind, arm=self.arm, op_id=op_id, text=text)

    @property
    def sensing(self):
        """The wrist's sensing of the page (ports.Sensing): the one given, else the stack's as it reads now."""
        if self._sensing is not None:
            return self._sensing
        if self._stack_sensing is None or self._stack_sensing.wrist != gauge.wrist(self.stack):
            self._stack_sensing = gauge.sensing(self.stack, self)
        return self._stack_sensing

    def stopping(self) -> bool:
        return self.land_requested or self.cancelled()

    def used(self) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("no page pose known")
        trim = self.stack["page"].get("trim", {}).get(self.arm, [0.0, 0.0])
        return geometry.page_used(self.camera, self.correction, trim)

    def tcp(self, q=None) -> np.ndarray:
        return self.kin.fk(self.io.measured() if q is None else q)

    def _curved(self) -> bool:
        """page.surface.mode fitted with a wrist that senses the page: fitted as a surface from the gauge holds (curved.py)."""
        return curved.config(self.stack)["mode"] == "fitted" and self.sensing.wrist != "none"

    def surface(self):
        """The curved page in use (tatbot_motion.surface.PageSurface over used()), or None: the flat page."""
        from tatbot_motion.surface import PageSurface, unfold

        if self.surface_field is None or self.camera is None:
            return None
        if self._surface_chart is None:
            self._surface_chart = unfold(self.surface_field)
        return PageSurface(self.used(), self.surface_field, self._surface_chart)

    def _on_surface(self, op: dict | None = None) -> dict:
        """The planners' surface keywords on a curved page (page.surface.turn_pen: the pen turned to each normal,
        else square to the page as the gauge holds measured it); nothing on a flat one, whose calls stay as they were.
        A stroke riding its skin along its path (_ride_over) is planned on it (curved.Ridden)."""
        surface, ridden, turned = self.surface(), self._ridden_for(op), self._turned()
        if surface is not None and turned:
            surface = curved.Turned(surface, [r["xy"] for r in turned], [r["lift_m"] for r in turned])
        if surface is not None and ridden is not None:
            surface = curved.Ridden(surface, ridden[1], ridden[2])
        return {} if surface is None else {"surface": surface, "turn_pen": bool(curved.config(self.stack)["turn_pen"])}

    def _ridden_for(self, op: dict | None):
        """(its chain's op ids, print points, offsets) when `op` rides its skin's profile (_ride_over), else None."""
        return self._ridden if op is not None and self._ridden is not None and op.get("id") in self._ridden[0] else None

    def knots(self, traj):
        from tatbot_motion import to_knots

        return to_knots(traj, self.knot_rate)

    # --- decisions -----------------------------------------------------------------------------
    def refusal(self) -> str | None:
        """Why the arm takes no Draw or Touch goal whatever it asks, or None: a landed arm idles until the stack
        restarts, and every move would report success with the arm at rest."""
        return rules.landed_refusal(self.arm) if self.io.flag("landed") else None

    def begin_goal(self) -> bool:
        """Take the arm for a Draw or Touch goal; False when it is busy."""
        with self._lock:
            if not self.busy.acquire(blocking=False):
                return False
            # a draw's own (_last_page); a landing held before it is held again once its goal moved the arm
            self.goal_active, self.land_requested, self._resumed, self._land_hold = True, False, None, None
            return True

    def end_goal(self) -> None:
        """Release the arm; a LAND accepted while the goal ran but not yet carried out lands now."""
        try:   # the setup's background work waited for and its camera closed, whatever the goal did
            for done in (self._settle, self._close_camera):
                try:
                    done()
                except Exception as error:   # a frame not written, a print not read, a camera that would not stop
                    self.event(Event.KIND_ERROR, f"setup: {error}")
            if self.land_requested and (landed := self._land_now())[0]:
                self.node.event(Event.KIND_LANDED, arm=self.arm, text=landed[1])
        finally:
            with self._lock:
                self.goal_active = self.land_requested = False
                self.busy.release()

    def decide(self, decision: int, timeout: float = 15.0, asked: bool = False) -> tuple[bool, str]:
        """Called from the Decide service thread (`asked`, a person's) and the Land action's. Hands the decision to the
        waiting executor, or acts directly on an idle arm (CONTINUE unlatches, LAND lands)."""
        with self._lock:
            ask = Ask(decision, self._wait_id) if self.waiting is not None else None
            if ask is not None:
                self.decisions.put(ask)
            elif decision == rules.LAND and self.goal_active:
                self.land_requested = True
                return True, "landing after the current goal stops"
        if ask is not None:
            try:
                return ask.reply.get(timeout=timeout)
            except queue.Empty:
                with self._lock:
                    if not ask.taken:
                        ask.abandoned = True
                        return False, "the executor did not answer"
                return ask.reply.get()  # taken: the executor answers once it has acted
        why = rules.check(None, decision)
        if why:
            return False, why
        if not self.busy.acquire(blocking=False):
            return False, "the arm is busy with another decision"
        try:
            if decision == rules.CONTINUE:
                if not self.io.latched:
                    return False, "nothing waits for continue"
                why = self.continue_refusal(self.io.measured())
                return (False, why) if why else self.io.unlatch()
            return self._land_now(asked)
        finally:
            self.busy.release()

    def continue_refusal(self, q_latched) -> str | None:
        joint, tip = geometry.pose_moved(q_latched, self.io.measured(), self.kin.fk)
        tool_changed = bool(self.tool_id) and config.fitted_tool(self.repo, self.arm) != self.tool_id
        return rules.continue_refusal(estop_ok=self.io.estop_ok, controller_error=self.io.flag("controller_error"),
                                      joint_moved_rad=joint, tip_moved_m=tip, tool_changed=tool_changed)

    def wait_decision(self, kind: str, op_id: str = "", index: int = 0, ledger=None) -> int:
        """Block until an accepted Decide. Returns the decision; LAND also comes from a Land request or
        the resume timeout. Raises Cancelled when the goal is cancelled. Only a Decide made during this
        wait answers it; one left in the queue when the wait ends is refused, never kept for the next."""
        self.machine_off()
        with self._lock:
            self._wait_id += 1
            wait_id, self.waiting = self._wait_id, kind
            self._drain("nothing waits for that decision now")
        self.feedback(phase=P.PHASE_LATCHED if kind == "latched" else P.PHASE_PAUSED, op_id=op_id, index=index)
        watch = {"q": self.io.measured() if kind == "latched" or self.io.latched else None,
                 "released": self.clock.now() if self.io.estop_ok else None}
        try:
            while True:
                if self.cancelled():
                    raise Cancelled()
                if self.land_requested or self._resume_expired(watch, op_id):
                    return rules.LAND
                if kind == "latched" and self.io.flag("controller_error"):
                    self.event(Event.KIND_DECISION, "the controller reports an error: no decision can move the arm "
                               "until the stack restarts; ending the run", op_id)
                    return rules.LAND
                try:
                    ask = self.decisions.get(timeout=0.1)
                except queue.Empty:
                    continue
                with self._lock:
                    if ask.abandoned or ask.wait_id != wait_id:
                        ask.reply.put((False, "nothing waits for that decision now"))
                        continue
                    ask.taken = True
                why = "the executor failed while acting on it"
                try:
                    why = self._refusal(kind, ask.decision, watch["q"])
                finally:
                    ask.reply.put((why is None, why or ""))
                name = rules.NAMES.get(ask.decision, str(ask.decision))
                self.event(Event.KIND_DECISION, f"{name} refused: {why}" if why else name, op_id)
                if why is None:
                    if ledger is not None:
                        ledger.append("decision", self.arm, op=op_id or None, index=index if op_id else None,
                                      decision=name)
                    return ask.decision
        finally:
            with self._lock:
                self.waiting = None
                self._drain("nothing waits for that decision now")

    def _drain(self, why: str) -> None:
        """Refuse every queued Ask (called with _lock held)."""
        while True:
            try:
                ask = self.decisions.get_nowait()
            except queue.Empty:
                return
            ask.reply.put((False, why))

    def _resume_expired(self, watch: dict, op_id: str) -> bool:
        """Track the latch and its release; True once a released latch waited past resume_timeout_s, or at once when
        it is an e-stop's and session.estop_release is land (_estop_released)."""
        if watch["q"] is None and self.io.latched:
            watch["q"] = self.io.measured()
        if watch["q"] is None:
            return False
        if not self.io.estop_ok:
            watch["released"] = None
        elif watch["released"] is None:
            watch["released"] = self.clock.now()
        timeout_s = float(self.stack["session"]["resume_timeout_s"])
        if self._estop_released():
            self.event(Event.KIND_DECISION, "e-stop released: landing (session.estop_release)", op_id)
            return True
        if rules.resume_expired(watch["released"], self.clock.now(), timeout_s):
            self.event(Event.KIND_DECISION, f"no decision {timeout_s:.0f} s after release: landing", op_id)
            return True
        return False

    def _estop_released(self) -> bool:
        """An e-stop's latch (pressed, or its heartbeat silent) with the e-stop reading released again, and
        session.estop_release land: held while pressed, the arm lands once it is released (owner, 2026-10-24)."""
        return ((self.stack.get("session") or {}).get("estop_release", "decide") == "land" and self.io.latched
                and self.io.estop_ok and self.io.latch_reason in (names.LATCH_ESTOP, names.LATCH_ESTOP_STALE))

    def _refusal(self, kind: str, decision: int, q_latched) -> str | None:
        """Why a decision is refused now, or None; an accepted CONTINUE on a latched arm has unlatched."""
        why = rules.check(kind, decision)
        if why is None and decision == rules.CONTINUE and (q_latched is not None or self.io.latched):
            why = self.continue_refusal(q_latched if q_latched is not None else self.io.measured())
            if why is None:
                ok, text = self.io.unlatch()
                why = None if ok else text
        return why

    def snag_guard(self, on: bool) -> None:
        """The driver's snag guard (ros/README.md 8) armed while a riding needle may be in the work: from a riding
        draw's first stroke goal through every lift and travel after it, until a dip, a page setup, a landing or the
        goal's end disarms it. Never for a pen that does not ride."""
        level = names.GUARD_SNAG if on and self.pen is not None and self.pen.machine else names.GUARD_NONE
        if level == self._snag:
            return
        self.io.write(guard_mode=level)
        self._snag = level
        if level:
            self.io.wait(lambda: self.io.latched or self.io.safety.get("guard_mode") == level, 1.0)

    def machine_off(self) -> None:
        """Ask for the tattoo machine off (the module docstring says when). The Pi powers it off by itself on the
        e-stop and when the session goes silent."""
        if self.machine is not None and self.machine.on:
            self.machine.set(False)
            self.event(Event.KIND_MACHINE, "machine off")

    def machine_running(self, op_id: str, index: int, ledger, again: bool = True) -> bool:
        """The tattoo machine on, as its switch reports it within machine.confirm_s, before a riding goal: True. Else the
        run pauses: land lands; continue asks again, or with again=False returns False (off): a stroke re-measures first."""
        was = self.machine.on
        while not self.machine.wait(True, self._confirm_s()):
            why = self.machine.why_off()
            self.machine.set(False)
            self.event(Event.KIND_PAUSE, f"the tattoo machine did not start: {why}; continue asks again, land lands",
                       op_id)
            if self.wait_decision("pause", op_id, index, ledger) == rules.LAND:
                self.land()
            if not again:
                return False
            was = False
        if not was:
            self.event(Event.KIND_MACHINE, "machine on", op_id)
        return True

    def _confirm_s(self) -> float:
        return float((self.stack.get("machine") or {}).get("confirm_s", 1.0))

    def require_off(self, activity):
        """The machine off, as its switch reports it, before motion that must not run it (a dip)."""
        self.machine_off()
        powered = bool(self.resource and self.resource['tool'].get('stroke_mm')) or (self.pen.machine if self.pen else False)
        machine.require_off(self.machine, powered, self._confirm_s(), activity)

    def prepare_pen(self, program: dict) -> None:
        """Before any motion: the program must be for the fitted tool, and a riding pen needs the tool's stroke
        and a switch that answers. Sets self.pen and says how the pen meets the page."""
        import tatbot_motion

        fitted, prepared = config.fitted_tool(self.repo, self.arm), (program.get("tool") or {}).get("id")
        if prepared and prepared != fitted:
            raise RuntimeError(f"the program was prepared for {prepared}, but config/workspace.yaml fits {fitted} on "
                               f"the {self.arm} arm: compile it for the fitted tool")
        riding = self.motion["pen"]["mode"] == tatbot_motion.RIDE
        self.pen = tatbot_motion.pen_down(self.motion, config.tool_datasheet(self.repo, fitted) if riding else None)
        self._pen_height()
        if riding and self.machine is None:
            raise RuntimeError(f"motion.yaml pen.mode ride runs the tattoo machine, and stack.yaml machine switches "
                               f"none on the {self.arm} arm")
        if riding and not self.machine.wait(False, self._confirm_s()):
            raise RuntimeError(f"the tattoo machine's switch does not answer: {self.machine.why_off()}")
        self.event(Event.KIND_PAGE, f"pen down: {self.pen.describe()}")

    def land(self, why: str = "") -> None:
        self.feedback(phase=P.PHASE_LANDING)
        self.event(Event.KIND_DECISION, "landing")
        ok, text = self._land_now()
        if ok:
            self.event(Event.KIND_LANDED, text)
        raise Landed(why if ok and why else text)

    def _land_now(self, asked: bool = False) -> tuple[bool, str]:
        """Lift clear of the page first when the pen may be on it (the driver's staged move would drag it
        sideways), then the driver's landing; `asked`: a person's decide (_hold_unlifted). Why not goes on
        /tatbot/events whichever way the land came: a Land action's or a decide's on an idle arm too."""
        why = in_cap_refusal(self.arm)
        if not why and self.io.flag("controller_error"):
            why = ("the controller reports an error, so the driver moves nothing: `tatbot ros up` restarts the stack, "
                   "whose configure clears it, then withdraws the pen and lands")
        why = why or self._lift_clear(asked)
        with self._lock:
            self.land_requested = False
        if not why:
            self.snag_guard(False)
            ok, why = self.io.land(self.stack["safety"]["landing"]["budget_s"])
            if ok:
                return ok, why
        self.event(Event.KIND_ERROR, why)
        return False, why

    def _height(self, q=None, page=None) -> float | None:
        """Tip height over the page in use (m), or None before a page is known."""
        return None if self.camera is None else self._over(self.tcp(q)[:3, 3], page)

    def _over(self, point, page=None) -> float:
        """How far a base point stands over the page in use (m), over its surface when curved; over `page`'s plane."""
        surface = self.surface() if page is None else None
        if surface is not None:
            return float(surface.height_over(np.asarray(point, float))[0])
        page = self.used() if page is None else page
        return float(page[:3, 2] @ (np.asarray(point, float) - page[:3, 3]))

    def lift_plan(self, distance, direction=None):
        from tatbot_motion import plan_lift

        if direction is None:
            direction = self._up(self.tcp()[:3, 3])
        return self.knots(plan_lift(q_seed=self.io.seed(), direction=direction, distance_m=distance, kin=self.kin,
                                   motion=self.motion))

    def _up(self, point) -> np.ndarray:
        """The page's normal at a base point, which a lift rises along: the curved page's there, else the page in use's."""
        surface = self.surface()
        return self.used()[:3, 2] if surface is None else surface.height_over(np.asarray(point, float))[1]

    # the landing's staged move turns the wrist: from 10 mm over the skin the needles nearly scraped across it
    # (operator, 2026-10-13), so the tip first rises this far along the skin's normal
    LAND_CLEAR_M = 0.040
    LAND_HOLD_M = 0.020   # a tip no lift took this far over the skin holds (_hold_unlifted): twice that near-scrape
    AT_REST_RAD = 0.2   # within this of the sleep pose on every arm joint the arm is at rest: no pen to withdraw

    def _lift_clear(self, asked: bool = False) -> str | None:
        """Before landing: a tip under LAND_CLEAR_M over the page lifts along the normal, unlatching with the
        hold goal when the e-stop is released. With no page known (a stack restarted after a controller drop: on
        2026-10-06 the controller closed its connection 49 s into a stroke and the 3RL's needles stayed in the
        half-skin) an arm away from rest withdraws the pen approach.withdraw_m along its own axis: the idled
        controller let the tip sag 9-13 mm into the skin before the restart (2026-10-08), so the standoff alone
        left it there; not a tip the cameras' page puts clear of the skin (_skin_height): on 2026-10-21 one 0.51 m over
        it and 0.25 m beside the print held, its withdraw needing joint_1 past its stop. A needle the snag guard stopped
        (its landing held since too: _land_hold) withdraws a standoff along the pen from wherever it stands, under the
        guard: on 2026-10-08 the work held it after the lift had reached the standoff; then, still guarded, it rises
        LAND_CLEAR_M along the skin's normal like any landing's (from the standoff the sweep nearly scraped the skin,
        2026-10-13). In steps when it must be (_lift_steps); then why the arm must not land (its landing would sweep
        the work along): the snag guard held the lift, the tip is left near the skin, the driver keeps its latch (it
        lands from the latched pose: a needle the work dragged off it, 2026-10-21 review), or anything here failed. A
        person's land (`asked`) lands from a hold they were told."""
        held, self._land_hold = self._land_hold, None
        told = held is not None and held[2] and float(np.max(np.abs(self.io.measured()[:6] - held[0][:6]))) < 0.01
        snag = (self.io.latched and self.io.latch_reason == names.LATCH_GUARD_SNAG) or (held is not None and held[1])
        said = []

        def hold(within: bool, failed: str) -> str | None:
            if not (within and asked and told):
                return self._hold_unlifted(within, failed, snag, asked or told)
            if not said:
                self.event(Event.KIND_DECISION, "landing from the hold as a person decided: no lift planned off the skin")
                said.append(True)
            return None
        try:
            if not self.io.estop_ok:
                return None
            lift = self._clearing_lift(snag)
            if lift is None:
                return None
            if self.io.latched and not (unlatched := self.io.unlatch())[0]:
                return hold(self._left_near(lift[1], True), unlatched[1])
            if snag:
                self.snag_guard(True)
            why = self._lift_steps(*lift, hold)
            if why is None and snag and self.camera is not None and self.io.estop_ok and (
                    lift := self._clearing_lift(False)) is not None:
                why = self._lift_steps(*lift, hold)
            return why
        except Exception as exc:  # noqa: BLE001 - where the tip is unknown: it holds as one near the skin
            self.event(Event.KIND_ERROR, f"lift before landing: {exc}")
            return hold(True, str(exc))

    LIFT_STEP_M, LIFT_STEPS = 0.005, 10   # _lift_steps' steps along the pen, at most this many

    def _lift_steps(self, distance, direction, where, hold) -> str | None:
        """_lift_clear's lifts: the first of _lift_plans, then LIFT_STEP_M steps along the pen, each planned from the
        measured joints (after one that failed too), until the tip clears (by its height, else all of `distance` along
        the pen). `hold` (_lift_clear's) decides."""
        from tatbot_motion.clik import PlanError

        lifted, failed = 0.0, ""
        for attempt in range(1 + self.LIFT_STEPS):
            # still to lift: along the pen by what it lifted, along the normal by the tip's height measured again
            need = distance - lifted if direction is not None else (self._clearing_lift(False) or [0.0])[0]
            if need <= 0.001:
                break
            step = min(need, self.LIFT_STEP_M)
            try:
                traj, length, how = (self._lift_plans(need, direction, where) if attempt == 0 else
                                     (self.lift_plan(step, -self.tcp()[:3, 2]), step, "along the pen"))
            except PlanError as exc:
                failed = str(exc)
                self.event(Event.KIND_ERROR, f"lift before landing: {exc}")
                if attempt:
                    break
                continue
            self.feedback(phase=P.PHASE_LIFT)
            status, _, text = self.io.execute(traj)
            if status == "latched" and self.io.latch_reason == names.LATCH_GUARD_SNAG:
                return ("not landing: the snag guard held the lift before landing, the needle held in the work. The arm "
                        "holds there: free the needle, then `tatbot ros decide land`")
            if status != "ok":
                failed = text
                self.event(Event.KIND_ERROR, f"lift before landing: {text}")
                if not self.io.estop_ok:   # pressed: it lands from where it stands, as when pressed before the lift
                    return None
                if self.io.latched and not self.io.unlatch()[0]:
                    break
                continue
            lifted += length
            self.event(Event.KIND_DECISION, f"lifted {length * 1000:.1f} mm {how} before landing")
        return hold(self._left_near(direction, lifted < distance - 0.001), failed)

    def _left_near(self, direction, short: bool) -> bool:
        """Whether a tip a lift before landing left holds (_hold_unlifted): under LAND_HOLD_M over the skin under it
        (_skin_height); one lifted along the pen (with no page in use, or a snag's needle) only `short` of the lift."""
        height = self._skin_height() if direction is None or self.camera is None else None
        return (direction is None or short) and (height is None or height < self.LAND_HOLD_M - 0.001)

    def _skin_height(self) -> float | None:
        """The tip's height over the skin a landing could sweep across (m): over the page in use (_height), else over
        the cameras' page measured within page.max_lost_s (a restarted stack knows none until a goal sets one up); inf
        past the print and LAND_CLEAR_M round it, where the page's plane runs on over no skin; None with no page."""
        tip, newest = self.tcp()[:3, 3], None if self.camera is not None else self.node.camera_page(self.arm)
        if self.camera is None and (newest is None or newest[0] is None or geometry.stale_page(
                newest[1], float(self.stack["page"].get("max_lost_s", 5.0)))):
            return None
        page = self.used() if self.camera is not None else np.asarray(newest[0], float)
        local = np.linalg.inv(page) @ [*tip, 1.0]
        if np.any(np.abs(local[:2]) > np.asarray(self.stack["page"]["size_m"], float) / 2 + self.LAND_CLEAR_M):
            return np.inf
        return self._height() if self.camera is not None else float(local[2])

    def _hold_unlifted(self, within: bool, failed: str, snag: bool, told: bool) -> str | None:
        """A tip _lift_steps left under LAND_HOLD_M over the skin (with no page or after a snag, short of its withdraw)
        holds there, the machine off, and why is returned: on 2026-10-21 the landing's wrist turn swept the needles from
        the skin after two lifts failed. Every land holds again but a person's at a hold `told` them (_lift_clear): an
        automatic land's hold (the draw wrapper's) is first said to a person whose land then holds."""
        if not within:
            return None
        self._land_hold = (self.io.measured(), snag, told)
        self.machine_off()
        return (f"not landing: no lift off the skin plans from here ({failed or 'none planned'}), and the landing's "
                "wrist turn would sweep the needles across it. The arm holds there, the machine off: " +
                (f"once the person is clear of the needles, `tatbot ros decide {self.arm} land` again lands from here"
                 if told else f"a person's `tatbot ros decide {self.arm} land` first says this to them; once they are "
                 "clear of the needles, their next lands from here"))

    def _clearing_lift(self, snag: bool):
        """The lift before landing (_lift_clear) as (distance, direction or None for the page normal, how), or None
        when the pen is clear of the skin (_skin_height) or, with no page known, at rest."""
        from tatbot_session import inspect as ins

        standoff = float(self.motion["approach"]["standoff_m"])
        height, clear = self._skin_height(), max(standoff, self.LAND_CLEAR_M)
        if self.camera is None:   # at rest, or clear of the cameras' page: nothing to withdraw
            if np.all(np.abs(self.io.measured()[:6] - ins.REST[:6]) < self.AT_REST_RAD) or (
                    height is not None and height >= clear - 0.001):
                return None
            return float(self.motion["approach"]["withdraw_m"]), -self.tcp()[:3, 2], "along the pen"
        if snag:   # the needle out the way it went in, from wherever it stands
            return standoff, -self.tcp()[:3, 2], "along the pen"
        return None if height >= clear - 0.001 else (clear - max(height, 0.0), None, "along the skin's normal")

    # --- moves ---------------------------------------------------------------------------------
    def move(self, make, phase: int, what: str, ledger=None) -> None:
        """Run a non-stroke motion (travel, lift); after a latch or failure wait for a decision and re-plan it from the
        measured joints on CONTINUE; after a snag-guard trip the needle comes out at once (_withdraw), and the first
        goes on by itself."""
        trips = 0
        while True:
            traj = make()
            self.feedback(phase=phase)
            status, _, text = self.io.execute(traj, should_stop=self.stopping)
            if status == "ok":
                return
            if status == "cancelled" and not self.land_requested:
                raise Cancelled()
            if status != "cancelled":
                self.event(Event.KIND_LATCHED if status == "latched" else Event.KIND_ERROR, f"{what}: {text}")
            trips += (snag := status == "latched" and self.io.latch_reason == names.LATCH_GUARD_SNAG)
            if not (snag and self._withdraw("", trips > 1) and trips < 2) and self.wait_decision(
                    "latched", ledger=ledger) == rules.LAND:
                self.land()

    def _lift_plans(self, distance, direction=None, where="along the skin's normal"):
        """(plan, length, how) of the first lift that plans: `distance` along `direction` (None: the page's normal)
        and along the pen, then half as far each way. PlanError when none does."""
        from tatbot_motion.clik import PlanError

        ways = [(direction, where)] + ([(-self.tcp()[:3, 2], "along the pen")] if direction is None else [])
        tries = [(length, way) for length in (distance, distance / 2) for way in ways]
        for i, (length, (way, how)) in enumerate(tries):
            try:
                return self.lift_plan(length, way), length, how
            except PlanError:
                if i == len(tries) - 1:
                    raise

    def lift(self, what: str, ledger=None) -> None:
        """Straight up along the page normal to the standoff (after a touch, before leaving the page); one the arm
        cannot follow there goes along the pen, then half as far (_lift_plans)."""
        distance = float(self.motion["touch"]["lift_m"])
        self.move(lambda: self._lift_plans(distance)[0], P.PHASE_LIFT, what, ledger)
        self._hop_from = None   # off any hold's stand (_hold_travel)

    def lift_off(self, what: str, ledger=None) -> None:
        """A tip within the standoff of the page in use lifts along its normal (lift) before it travels."""
        height = self._height()
        if height is not None and height < float(self.motion["approach"]["standoff_m"]) - 0.001:
            self.lift(what, ledger)

    BRANCH_RAD = 0.1         # a ready pose this far from the page's recorded one, on any joint, is another branch

    def _page_ready(self, record: dict | None = None) -> dict | None:
        """The page's ready pose ({"q", "heading"}): recorded by this run's first ready move, else carried with
        the paper from the last run on the same page (_carried_paper) unless its heading turns past 45 deg from the
        dips' (_dip_heading: a run before strokes went at it); of `record`, else the page record."""
        record = self.page_record if record is None else record
        own, carried = record.get("ready"), (record.get("carried") or {}).get("ready")
        if own or not carried or (dips := self._dip_heading()) is None:
            return own or carried
        turn = ready.heading_yaw(self.used(), np.asarray(carried["heading"], float)) - ready.heading_yaw(self.used(), dips)
        return carried if abs(np.angle(np.exp(1j * turn))) <= np.pi / 4 else None

    def _dip_heading(self) -> np.ndarray | None:
        """The rotation every dip of the fitted tool goes at (cap.py: the station touch's), when one speaks for it."""
        from tatbot_motion import station

        try:
            touch, fitted = station.load_touch(self.arm), config.fitted_tool(self.repo, self.arm)
        except (station.StaleStationError, OSError, ValueError, KeyError):
            return None
        same = touch.tool_id == fitted or station.same_machine(self.repo, touch.tool_id, fitted)
        return np.asarray(touch.rotation, float) if same else None

    def _ready_seed(self) -> np.ndarray:
        """Where every ready choice solves from: the page's recorded ready joints, else the landing rest pose,
        never the joints the arm happens to hold. The arm's tip height error depends on its IK branch (~5 mm per
        radian of joint 6), and on 2026-10-03 the same page reached from a locate view and from rest took
        branches with joint 6 at -0.10, +0.29 and +0.75 rad: the paper read 3.0, 4.7 and 7.0-7.9 mm."""
        from tatbot_session import inspect as ins

        recorded = (self._page_ready() or {}).get("q")
        seed = ins.REST.copy() if recorded is None else np.asarray(recorded, float).copy()
        seed[6] = self.io.seed()[6]   # the carriage holds where it is
        return seed

    def _ready_for(self, page: np.ndarray, pattern_id) -> tuple[np.ndarray, np.ndarray]:
        """The heading and IK seed the ready move will take on `page` before its record exists (pen_heading,
        _ready_seed): the carried paper's ready pose on the same print, else pen_heading's search from rest, kept for
        the ready move (_located_heading). The locate's over-centre views go at them, so the ready move only descends."""
        from tatbot_session import inspect as ins

        recorded = self._page_ready({"carried": self._carried_paper(page, {"pattern_id": pattern_id}, height=False)})
        seed = ins.REST.copy() if recorded is None else np.asarray(recorded["q"], float).copy()
        seed[6] = self.io.seed()[6]
        if recorded is not None:
            return np.asarray(recorded["heading"], float), seed
        self._located_heading = ready.pen_heading(self.kin, page, self.stack["page"]["clear_m"],
                                                  float(self.motion["approach"]["standoff_m"]), seed,
                                                  prefer=self._dip_heading())
        return (self.tcp() if self._located_heading is None else self._located_heading), seed

    def pen_heading(self) -> np.ndarray:
        """A tcp pose whose x axis is the pen heading (its turn about its own axis) that keeps every joint
        farthest from its limits over the clear centre. The ballpoint is round, so the heading is free,
        and every later plan keeps the ready pose's heading: the one the arm arrived with put joint_3
        0.04 rad from its limit over a page near the base line (2026-09-26) where
        another left 0.6 rad. The page's recorded heading when it has one (_page_ready: this run's, else the one
        carried from the last run on this page); else the search from _ready_seed, the first of the headings
        tied (ready.HEADING_TIE_RAD) with the best. The margins of several headings differ by less than a page height's
        few millimetres move them (45 and 60 deg trade places over 8 mm), and a bare best flipped between
        headings 23-49 deg apart from one resume to the next: only a new page searches, nearest the dips' heading
        first (ready.PREFER_MARGIN_RAD), so no dip turns the wrist round the pen and back. A setup's locate searches it
        first (_ready_for), and the ready move takes that one."""
        recorded = (self._page_ready() or {}).get("heading")
        if recorded is not None:
            return np.asarray(recorded, float)
        if self._located_heading is not None:
            return np.asarray(self._located_heading, float)
        heading = ready.pen_heading(self.kin, self.used(), self.stack["page"]["clear_m"],
                                    float(self.motion["approach"]["standoff_m"]), self._ready_seed(),
                                    prefer=self._dip_heading())
        return self.tcp() if heading is None else heading

    READY_HIGH_M = 0.060   # the tool over the page centre this high, when the direct ready move would dip
    REST_RAD = 0.05        # a pose this close to rest (the landing sleep pose) leaves the page to the cameras

    def clear_view(self, ledger=None) -> None:
        """Back to rest, holding, so the fixed cameras can measure a lost page again. A cancelled draw holds
        where it stopped, over the print, and the anchoring camera could not see it (2026-09-28); landing would
        restart the controller. A pen within the standoff lifts first; a path that would take it within half the
        standoff of the page is not taken, and the page wait then decides as before."""
        from tatbot_session import inspect as ins

        rest = ins.REST.copy()
        rest[6] = self.io.seed()[6]
        if float(np.max(np.abs(self.io.measured()[:6] - rest[:6]))) <= self.REST_RAD:
            return
        standoff = float(self.motion["approach"]["standoff_m"])
        self.lift_off("lift before clearing the cameras' view", ledger)
        # the lift ran under the snag guard, so a needle the work held has stopped it; the swing to rest is a travel
        # off the work, and at rest the joints lie ~1 deg off REST against their stops, the tip 4.4 mm off its
        # command: the guard latched there and the run ended (silicone practice, 2026-10-21)
        self.snag_guard(False)

        def make():
            return ready.pen_up_joint_move(self.kin, self.motion, self.io.seed(), rest, self.knot_rate, P.PHASE_TRAVEL)

        def high():
            q = self.io.seed()
            target = geometry.tool_down_pose(self.used(), [0.0, 0.0], self.READY_HIGH_M, self.tcp())
            return ready.pen_up_joint_move(self.kin, self.motion, q, ready.solve_ik_seeded(self.kin, target, q),
                                           self.knot_rate, P.PHASE_TRAVEL)

        floor = 0.5 * standoff
        if self.camera is not None and self.lowest_over_page(make()) < floor:
            # from a hold over a skin pinned on a raised holder the joint move to rest swept the pen back over it
            # 4 mm up (2026-10-08) and the arm, refusing, hid the page from the overhead until the run gave up:
            # over the page centre at READY_HIGH_M first, as the ready move goes, then to rest from there
            try:
                up = self.lowest_over_page(high())
            except Exception:   # a leg up that cannot be planned is not taken
                up = -np.inf
            if up < floor:
                self.event(Event.KIND_PAGE, "the move to rest would pass the pen within {:.0f} mm of the page; "
                           "not moving".format(floor * 1e3))
                return
            self.move(high, P.PHASE_TRAVEL, "over the page, high, clearing the cameras' view", ledger)
            if self.lowest_over_page(make()) < floor:
                self.event(Event.KIND_PAGE, "the move to rest from over the page would pass the pen within {:.0f} mm "
                           "of it; holding high".format(floor * 1e3))
                return
        self.move(make, P.PHASE_TRAVEL, "rest, clearing the cameras' view of the page", ledger)

    def lowest_over_page(self, traj, page=None) -> float:
        """The planned tip's lowest height over the page in use along its normal (m); +inf with no tip."""
        tip = np.asarray(getattr(traj, "tip", ()), float).reshape(-1, 3)
        tip = tip[np.isfinite(tip).all(axis=1)]
        if not len(tip):
            return float("inf")
        surface = self.surface() if page is None else None
        if surface is not None:
            return float(min(surface.height_over(p)[0] for p in tip[::max(1, len(tip) // 200)]))
        page = self.used() if page is None else page
        return float(np.min((tip - page[:3, 3]) @ page[:3, 2]))

    def go_ready(self, ledger=None) -> None:
        """Joint-space move to the tool over the page centre at the standoff (the arm rests on joint
        limits the Cartesian planner keeps clear of). A tip already below the standoff lifts first.

        A joint-space move keeps its ends clear of the page, not its path: from a locate view (the pen
        15 mm up, the camera tilted 50 deg) to the ready pose it swept the tip 5-12 mm through a page
        placed near the arm base (bench 2026-09-27). A planned tip that comes within half the standoff
        of the page goes up first (a straight lift, then the tool over the page centre at READY_HIGH_M)
        and down from there; a path that still dips is refused before the arm moves."""
        standoff = float(self.motion["approach"]["standoff_m"])
        floor = 0.5 * standoff
        self.lift_off("lift before the ready move", ledger)
        heading = self.pen_heading()

        seed = self._ready_seed()

        def joint_to(height_m):
            target = geometry.tool_down_pose(self.used(), [0.0, 0.0], height_m, heading)

            def make():
                q = self.io.seed()
                traj = ready.pen_up_joint_move(self.kin, self.motion, q, ready.solve_ik_seeded(self.kin, target, seed),
                                               self.knot_rate, P.PHASE_TRAVEL)
                low = self.lowest_over_page(traj)
                if low < floor:
                    raise RuntimeError("the ready move would take the pen {:+.1f} mm over the page (under {:.0f} mm); "
                                       "not moving".format(low * 1e3, floor * 1e3))
                return traj
            return make

        try:
            joint_to(standoff)()
        except RuntimeError as dip:
            self.event(Event.KIND_REPLAN, f"{dip}: over the page centre at {self.READY_HIGH_M * 1e3:.0f} mm first")
            self.lift("lift before the ready move", ledger)
            self.move(joint_to(self.READY_HIGH_M), P.PHASE_TRAVEL, "ready, high", ledger)
        self.move(joint_to(standoff), P.PHASE_TRAVEL, "ready", ledger)
        self._record_ready(heading)

    def _record_ready(self, heading) -> None:
        """The page's ready pose (joints and pen heading), which a resume and every later ready move of this page
        take again; a ready pose BRANCH_RAD off a recorded one is named: it measures and draws on another branch."""
        q = np.asarray(self.io.measured(), float)
        recorded = self._page_ready()
        self.page_record.setdefault("ready", {"q": q.tolist(), "heading": np.asarray(heading, float).tolist()})
        if recorded is None:
            return
        off = float(np.max(np.abs(q[:6] - np.asarray(recorded["q"], float)[:6])))
        if off > self.BRANCH_RAD:
            self.event(Event.KIND_PAGE, f"the ready pose is {off:.2f} rad from this page's recorded one: another IK "
                       "branch, whose tip heights differ by millimetres")

    # --- page ----------------------------------------------------------------------------------
    # The locate plans its views about the page tilted at most AIM_TILT_RAD from the base's z, its centre and turn kept;
    # each view reads the print on `prior` itself. About the overhead's page of a forearm print at 19.1 and 20.5 deg
    # (the chord plane of its markers in view, 2026-10-20) no view was reachable; about the bent layout's 14.5 and 12.6
    # deg (stencild since) 17 and 21 were, capped at 10 deg 22 and 24 (replayed without the self-collision test).
    AIM_TILT_RAD = np.radians(10.0)

    def wrist_locate(self, prior: np.ndarray, cfg: dict, ledger=None, program=None,
                     surface=None, alone: bool = False) -> tuple[np.ndarray, dict]:
        """Fuse overhead and wrist XY/yaw, recording page depth and replayable captures; `alone` (a prior no overhead
        pose placed, _blind_prior): the wrist's own, searched for over the overhead's skin when no view reads it
        (_overhead_skin).
        Each reachable view reads the print's markers on the page plane (inspect.view_fit) and samples native
        depth; see ros/README.md. On a curved page each marker corner is lifted by its own depth instead
        (track.print_read), and no depth plane gives a height: rectified onto the overhead's plane, 8.8 mm over a
        forearm's skin, a 50 deg view read no marker and the other a turn 0.5 deg off the holds' (2026-10-19). The
        corners go onto `prior` along its normal: at a first setup the overhead pose's plane, no field existing yet
        (the holds' own turn and shift then place the page); a kept page's (_kept_page) through `surface`'s chart.
        """
        from tatbot_session import inspect as ins

        # The view search solves hundreds of candidates, most unreachable: over a straight-down search's 576 (2026-10-22)
        # those that converged took 5 iterations median, 71 at most; the 274 that did not, 98% of its time at 400 each.
        solve_ik = functools.partial(ready.solve_ik, iterations=100)
        page, z, plan = self.stack["page"], prior[:3, 2], np.array(prior, float)
        tilt = min(float(np.arccos(np.clip(z[2], -1.0, 1.0))), self.AIM_TILT_RAD)   # its z turned the least onto it
        plan[:3, :3] = closeup._turn(z, [*z[:2] / max(np.hypot(*z[:2]), 1e-12) * np.sin(tilt), np.cos(tilt)]) @ prior[:3, :3]
        frame, ppm = ins.camera_frame(self.arm), 5.0
        extent = ins.page_extent(page["size_m"])
        tile = ins.print_artwork(page.get("pattern_id"))
        max_y = float(prior[1, 3]) - float(cfg.get("side_margin_m", 0.0))
        from tatbot_cli import ros_depth

        cam, clear = self.setup_camera(), float(cfg.get("clearance_m", 0.015))
        sols, views, further, best, curved_page = [], [], [], None, self._curved()
        hint = self._view_hint(plan, page.get("pattern_id"))

        def to_view(q_t):   # at a third of the pen-up speed joint_3 chattered at 5 Hz lowering, the tip 2.8-6.7 mm
            return ready.pen_up_joint_move(self.kin, self.motion, self.io.seed(), q_t, self.knot_rate, P.PHASE_TRAVEL)

        skin = self._overhead_skin(plan, ledger) if alone else None   # read at rest, the arm out of the overhead's view
        guard = getattr(self.node, "guard", None)
        self_gap = None if guard is None else (lambda q: guard.self_gap(self.arm, q))

        # The views over the print are chosen by the markers they see past the tool, straight down from as high as its
        # smallest keeps MARKER_PX and looking aside of it too; a search's (where the print may be) as before.
        see = None if tile is None else self._markers_seen(cam, plan, tile, frame)
        down = {} if see is None else {"see": see, "around": ins.AROUND,
                                       "distances_m": ins.top_down_distances(tile, cam.k[0, 0])}
        oblique = {} if see is None else {"see": see}

        def reach(n, page_n, xy):   # straight down on it first (inspect.top_down), else obliquely as before
            return self._reach_view(n, best, hint, to_view, clear, [functools.partial(
                how, self.kin, solve_ik, page_n, xy, self.io.measured(), frame=frame, clearance_m=clear, max_y=max_y,
                self_gap=self_gap, **(chosen if page_n is plan else {}))
                for how, chosen in ((ins.top_down, down), (ins.aim, oblique))], page=None if page_n is plan else page_n)

        def look(n, q_t, view, search=None):   # one view: moved to, its frames and depth kept, its print read
            nonlocal best
            best, xy = best or (view if "azimuth_deg" in view else None), (0.0, 0.0) if search is None else search["xy"]
            grid = (ins.page_extent(page["size_m"], 0.010 + max(map(abs, xy))), ppm)   # a search's print moved this far
            self.move(functools.partial(to_view, q_t), P.PHASE_TRAVEL, f"locate view {n}", ledger)
            self.clock.sleep(float(cfg.get("settle_s", 0.8)))
            frames, depths = cam.grab_both(int(cfg.get("frames", 5)), observe=lambda: ros_depth.observe(self.io),
                                           fresh=True)
            q_view = self.io.measured()
            base_from_cam = self.kin.frame(q_view, frame)
            capture = f"depth{n}-{int(self.clock.utc() * 1e9)}"
            out = None if self.run_dir is None else self.run_dir / "locate" / capture
            height, depth_observation, clearance = ros_depth.retain(cam, frames, depths, self.kin, q_view,
                                                                   prior, out, self.arm)
            self._keep_skin(view, cam, depths, q_view)
            image, sol, why = self._view_read(cam, frames, base_from_cam, prior, tile, grid, curved_page, surface)
            self._say_view(n, sol, why)
            sols.append(sol)
            views.append({**view, "fit": sol, "depth": height, "depth_observation": depth_observation, "clearance": clearance,
                          **({"search": search} if search else {})})
            self._keep_view(n, image, frames[0], views[-1], capture)
            seen = self._locate_further(n, q_view, int(cfg.get("frames", 5)), sols, further, functools.partial(
                self._view_read, prior=prior, tile=tile, grid=grid, curved_page=curved_page, surface=surface))
            return sol or seen

        taken, poses = self._over_views(prior, cfg, look, functools.partial(
            ins.over_centre, self.kin, solve_ik, prior, (0.0, 0.0), frame=frame, clearance_m=clear,
            self_gap=self_gap, see=see), to_view, sols, alone)
        for n in range(taken, taken + poses):   # each about the page centre; their search said nothing, 53 s at first
            began, aimed = self.clock.now(), reach(n, plan, (0.0, 0.0))   # (2026-10-22)
            self.event(Event.KIND_PAGE, f"locate view {n}: {'none' if aimed is None else 'one'} reachable in "
                       f"{self.clock.now() - began:.1f} s")
            if aimed is None and best is None:   # every view searched from where the arm stands: so would the
                break                            # next pose's be (2026-10-17/18: three alike; 2026-10-20: 117 s)
            if aimed is not None:
                look(n, *aimed)
        aims = self._skin_search(skin, extent, plan, poses, reach, look) if alone and not any(sols) else []
        heights = [] if curved_page else [v["depth"]["offset_m"] for v in views if v["depth"] is not None]
        wrist = ins.combine_views(sols, floor=(float(cfg["wrist_floor_m"]), float(cfg["wrist_floor_m"]),
                                               float(cfg["wrist_floor_rad"])))
        record = {"overhead": prior.tolist(), "wrist": wrist, "views": views, **({"search": aims} if aims else {}),
                  **({"further": further} if further else {})}
        if not heights and not curved_page:
            reasons = ", ".join(v["depth_observation"]["reason"] for v in views) or "no reachable view"
            self.event(Event.KIND_PAGE, f"wrist depth: no usable page plane ({reasons}); no depth height enters fusion")
        if heights:
            spread = 1.4826 * float(np.median(np.abs(np.array(heights) - np.median(heights)))) if len(heights) > 2 else \
                float(np.ptp(heights)) / 2
            cfg_h = self.stack["page"]["height"]
            bias = float(cfg_h.get("wrist_depth_bias_m", 0.0))   # the CAD mount's measured offset over the touches
            record["depth"] = {"offset_m": float(np.median(heights)) - bias, "raw_offset_m": float(np.median(heights)),
                               "bias_m": bias, "views": len(heights),
                               "sigma_m": max(spread, float(cfg_h["wrist_depth_sigma_m"]))}
        if wrist is None:
            self.event(Event.KIND_PAGE, f"wrist locate: {len(views)} views of the print reached, none read its "
                       f"markers; the {'last page, raised,' if alone else 'overhead pose'} stands")
            return prior, record
        sigma_o = (float(cfg["overhead_sigma_m"]), float(cfg["overhead_sigma_m"]), float(cfg["overhead_sigma_rad"]))
        # The wrist's own estimate stands for a prior no overhead pose placed (alone), and for an overhead pose two views
        # or more read past three sigmas of the two from it: a resume (2026-10-20) took the overhead's first pose while
        # they jumped 35-80 mm frame to frame, the wrist read the print 34.9 mm, 7.7 deg from it, the fused page stood
        # 7 mm off the print, and the holds' shift (7.8 mm) came within 1.2 mm of where they would not have moved it.
        far = wrist["poses"] > 1 and (np.hypot(wrist["tx_m"], wrist["ty_m"]) > 3 * np.hypot(sigma_o[0], max(
            wrist["sigma"][:2])) or abs(wrist["theta_rad"]) > 3 * np.hypot(sigma_o[2], wrist["sigma"][2]))
        fused = {k: wrist[k] for k in ("tx_m", "ty_m", "theta_rad", "sigma")} if alone or far else ins.fuse(
            [((0.0, 0.0, 0.0), sigma_o), ((wrist["tx_m"], wrist["ty_m"], wrist["theta_rad"]), wrist["sigma"])])
        record["fused"] = {**fused, "overhead_set_aside": bool(far and not alone)}
        pose = prior @ np.linalg.inv(ins.correction_matrix(fused["tx_m"], fused["ty_m"], fused["theta_rad"]))
        matched = sum(s is not None for s in sols)
        self.event(Event.KIND_PAGE, "wrist locate: print {:+.1f} {:+.1f} mm {:+.2f} deg from the {} "
                   "(sigma {:.1f} {:.1f} mm, {} views, {} by its markers); {} {:+.1f} {:+.1f} mm {:+.2f} deg".format(
                       -wrist["tx_m"] * 1e3, -wrist["ty_m"] * 1e3, -np.degrees(wrist["theta_rad"]),
                       "last page, raised" if alone else "overhead pose",
                       wrist["sigma"][0] * 1e3, wrist["sigma"][1] * 1e3, wrist["poses"], matched,
                       "the overhead's set aside past 3 sigma: the wrist's own" if far and not alone else
                       "the wrist's own" if alone else "fused",
                       -fused["tx_m"] * 1e3, -fused["ty_m"] * 1e3, -np.degrees(fused["theta_rad"])))
        return pose, record

    def _over_views(self, prior: np.ndarray, cfg: dict, look, over, to_view, sols: list,
                    alone: bool) -> tuple[int, int]:
        """The locate's over-centre views (inspect.over_centre: `over` with the joints and height left to give), at
        each of page.locate.over_m in turn until page.locate.over_enough of the cameras' reads place the print, at the
        ready move's heading and branch (_ready_for); each read by `look`. A view whose markers none would be seen
        (markers_seen) is not moved to, nor one whose joint path takes the pen under half the clearance over the page.
        None on a blind locate (`alone`: no overhead pose under it). (The views taken, the top-down views still to
        take: page.locate.poses, none once enough reads placed the print.)"""
        poses = int(cfg.get("poses", 3))
        heights = [] if alone else [float(h) for h in cfg.get("over_m") or ()]
        if not heights:
            return 0, poses
        clear, began, n = float(cfg.get("clearance_m", 0.015)), self.clock.now(), 0   # the first's time: the heading's
        heading, seed = self._ready_for(prior, self.stack["page"].get("pattern_id"))   # search too
        for h in heights:
            aimed = over(self.io.measured(), heading, height_m=h, seed=seed)
            seen = None if aimed is None else aimed[1]["seen"]
            self.event(Event.KIND_PAGE, f"locate view {n}: over the page centre at {h * 1e3:.0f} mm, " + (
                "not reachable" if aimed is None else "reachable" if seen is None else f"{seen} markers in view")
                + f" ({self.clock.now() - began:.1f} s)")
            low = None if aimed is None or seen == 0 else self.lowest_over_page(to_view(aimed[0]))
            if low is not None and low < 0.5 * clear:
                self.event(Event.KIND_REPLAN, f"locate view {n} skipped: the move would take the pen "
                                              f"{low * 1e3:+.1f} mm over the page")
            elif low is not None:
                look(n, *aimed)
                n += 1
                if sum(s is not None for s in sols) >= int(cfg.get("over_enough", 2)):
                    return n, 0
            began = self.clock.now()
        return n, poses

    def _locate_further(self, n: int, q, count: int, sols: list, further: list, read):
        """Locate view n read by the further cameras too (_further_frames: page.locate.cameras), from the same pose `q`
        at their own URDF pose (`read`: the view's _view_read): each fit appended to `sols` (combine_views) and its
        record to `further`, its images kept as locate/view<n>-<role>.png. The first fit read, or None."""
        found = None
        for other, cam, (shots, _, _) in self._further_frames(None, q, count):
            image, seen, why = read(cam, shots, self.kin.frame(q, other["frame"]))
            self._say_view(f"{n} by {other['role']}", seen, why)
            sols.append(seen)
            further.append({"view": n, "camera": other["role"], "fit": seen})
            self._keep_view(f"{n}-{other['role']}", image, shots[0], further[-1])
            found = found or seen
        return found

    def _markers_seen(self, cam, page: np.ndarray, tile, frame: str):
        """inspect.aim's `see` for views of `tile` laid on `page`: base_from_camera -> its markers seen past the fitted
        tool, whose datasheet footprint stands on the gauge's measured tip (inspect.tool_footprint)."""
        from tatbot_session import inspect as ins

        gauged = gauge.calibration_path(self.arm)
        tip = json.loads(gauged.read_text()).get("tip_cam_m") if gauged.exists() else None
        profile = config.tool_datasheet(self.repo, config.fitted_tool(self.repo, self.arm)).profile
        hidden = ins.tool_footprint(self.kin, self.io.measured(), cam.k, profile, frame, tip)
        return lambda base_from_camera: ins.markers_seen(page, base_from_camera, tile, cam.k, hidden)

    def _say_view(self, n: int, sol, why: str) -> None:
        """A locate view that placed nothing says why: the markers decoded and the check that dropped each."""
        if sol is None:
            self.event(Event.KIND_PAGE, f"locate view {n} read no placement: {why}")

    def _keep_view(self, n, image, frame, view: dict, capture: str | None = None) -> None:
        """A locate view's page image and first frame under the run's locate/, its depth capture named in its record."""
        if self.run_dir is None:
            return
        import cv2

        (self.run_dir / "locate").mkdir(exist_ok=True)
        cv2.imwrite(str(self.run_dir / "locate" / f"view{n}.png"), image)
        cv2.imwrite(str(self.run_dir / "locate" / f"frame{n}.png"), frame)
        if capture is not None:
            view["depth_capture"] = f"locate/{capture}.npz"

    def _reach_view(self, n: int, best, hint, to_view, clear: float, aims, page=None):
        """The view (joints, view) the locate's pose n moves to, or None: by each of `aims` in turn (the straight-down
        views, then the oblique ones), about its first view (alternating sides), else about `hint`'s (_view_hint) and
        then among every view; one whose joint path takes the pen under half `clear` over the page (`page`, a search's
        raised over its skin, else the page in use) is skipped (a view pose keeps the pen clear, the path to it need
        not: go_ready)."""
        side = 1 if n % 2 else -1
        ways = [[best["azimuth_deg"] + side * d for d in (15, 30, 45)]] if best else (
            [] if not hint else [[h + d for h in hint for d in (-15, 0, 15)]]) + [None]
        for aim, azimuths in itertools.product(aims, ways):
            aimed = aim(azimuths=None if azimuths is None else np.radians(azimuths))
            if aimed is not None and (low := self.lowest_over_page(to_view(aimed[0]), page)) < 0.5 * clear:
                self.event(Event.KIND_REPLAN, f"locate view {n} skipped: the move would take the pen "
                                              f"{low * 1e3:+.1f} mm over the page")
            elif aimed is not None:
                return aimed
        return None

    # A blind locate whose views read no marker searches the skin the overhead's depth still reads where it decodes no
    # print (2026-10-21: a forearm's markers inked over, the person's arm perhaps moved since): more views over the
    # last page's centre and the eight points the print's extent from it that have skin under them (inspect.skin_aims),
    # each over its own highest skin, BLIND_LIFT_M up, at the locate's own clearance, until one reads the print's
    # markers. A search that reads none, or no depth to aim it, leaves the refusal as it was.
    def _overhead_skin(self, page: np.ndarray, ledger=None) -> np.ndarray | str:
        """The overhead D555's depth in `page`'s frame (N, 3), every second pixel, or why there is none, read with the
        arm at rest out of its view (clear_view: no move when there already). The lens's distortion is left out: it
        aims views and measures nothing."""
        from tatbot_bridge import capture

        self.clear_view(ledger)
        try:
            camera = capture.Camera(self.repo)
            try:
                shot = camera.capture(int(self.clock.utc() * 1e9))
            finally:
                camera.close()
            k, _ = capture.k_dist(capture.intrinsics_of(shot["metadata"]))
        except Exception as error:  # noqa: BLE001 - no capture from the overhead's owner: no search
            return str(error) or type(error).__name__
        base = self.node.base_from_world(self.arm)
        if shot["depth_m"] is None or base is None:
            return "its capture carries no depth" if base is not None else "no transform to the arm's base"
        from tatbot_session import inspect as ins

        points = ins.depth_points(shot["depth_m"], k, stride=2, near_m=0.2, far_m=2.0)
        return (np.linalg.inv(page) @ base @ np.c_[points, np.ones(len(points))].T)[:3].T

    def _skin_search(self, skin, extent, plan, first: int, reach, look) -> list[dict]:
        """The search's aims, its views reached (`reach`) and read (`look`, numbered from `first`) until one reads the
        print; each planned about `plan` raised over its own skin, the print still read on the locate's prior."""
        from tatbot_session import inspect as ins

        if isinstance(skin, str):
            self.event(Event.KIND_PAGE, f"wrist search: no overhead depth to aim it ({skin})")
            return []
        aims = ins.skin_aims(skin, extent[2:], lift_m=self.BLIND_LIFT_M)
        wide = [] if aims else ins.wide_aims(skin, extent[2:], lift_m=self.BLIND_LIFT_M)
        self.event(Event.KIND_PAGE, f"wrist search: the overhead's depth reads skin under {len(aims)} of 9 points about "
                   "the last page" + ("" if aims or not wide else ", and under {} about the skin nearest it, {:+.0f} {:+.0f} "
                                      "mm from it".format(len(wide), *np.multiply(wide[0]["xy"], 1e3)))
                   + (f", the views raised up to {max(a['rise_m'] for a in aims or wide) * 1e3:.1f} mm over "
                      f"it: {len(aims or wide)} more locate views" if aims or wide else ": no search"))
        aims = aims or wide
        for n, at in enumerate(aims, first):
            aimed = reach(n, plan @ geometry.translation([0.0, 0.0, at["rise_m"]]), at["xy"])
            if aimed is None:   # said: 2026-10-21's nine aims reached none, and the log was silent
                self.event(Event.KIND_PAGE, "wrist search: view {} over the skin {:+.0f} {:+.0f} mm from the last page "
                           "not reachable".format(n, *np.multiply(at["xy"], 1e3)))
            elif look(n, *aimed, search=at) is not None:
                self.event(Event.KIND_PAGE, "wrist search: view {} over the skin {:+.0f} {:+.0f} mm from the last page "
                           "read the print's markers".format(n, *np.multiply(at["xy"], 1e3)))
                break
        return aims

    @staticmethod
    def _view_read(cam, frames, base_from_cam, prior, tile, grid, curved_page: bool, surface=None):
        """(the view's image, its fit of the print's markers or None): its frames rectified onto `prior`'s plane
        (inspect.view_fit, page `grid` (extent, px/mm)); on a curved page each marker corner in their median lifted
        by its own depth, the depth frames pooled, and carried onto `prior` and `surface`'s chart (track.print_read)."""
        from tatbot_session import inspect as ins

        if not curved_page:
            image = ins.median_stack([ins.rectify(f, prior, base_from_cam, cam.k, cam.dist, *grid) for f in frames])
            sol = ins.view_fit(image, tile, *grid)
            return image, sol, "" if sol is not None else "no marker of the print kept its shape on the page"
        image, seen = ins.median_stack(frames), {}
        read = track.print_read({"bgr": image, "frames": frames, "depth": gauge.pool(cam.raw_depth),
                                 "meta": cam.depth_metadata, "used": prior, "base_from_camera": base_from_cam,
                                 "surface": surface}, tile, seen)
        sol = read.get("print_sol") if "print_fit" in read else None
        ins.wrist_tap.show_markers(cam, getattr(cam, "stamp_ns", None), image.shape, "locate", seen, sol)
        return image, sol, read.get("print_none", "")

    PAGE_FIRST_WAIT_S = 10.0   # a page message after a stack restart; page.wait_s extends it for a lost print

    def _wait_page(self, ledger=None, clear: bool = False, last: dict | None = None) -> tuple[np.ndarray, dict] | None:
        """The overhead's fresh pose of the page and its info, waited for; refused (RuntimeError) without one. `clear`
        (a page it places mid-draw): one measured once the arm cleared its view, as at a first setup (over it: 11.5 deg).
        With `last` (_last_page: the wrist can place the page) PAGE_FIRST_WAIT_S more at most, and None without one."""
        max_lost_s = float(self.stack["page"].get("max_lost_s", 5.0))
        wait_s = min(float(self.stack["page"].get("wait_s", self.PAGE_FIRST_WAIT_S)),
                     2 * self.PAGE_FIRST_WAIT_S if last else np.inf)
        if clear:
            self.clear_view(ledger)
        since = self.clock.now() if clear else None
        camera = self.node.wait_camera_page(self.arm, timeout=min(wait_s, self.PAGE_FIRST_WAIT_S), max_lost_s=max_lost_s,
                                            since=since)
        stale = "no page pose yet" if camera is None else geometry.stale_page(camera[1], max_lost_s)
        if stale and wait_s > self.PAGE_FIRST_WAIT_S:
            # The arm itself hides a print through a draw and a held inspection; the tracker decodes it again
            # once uncovered. Waiting for that measurement keeps the freshness rule; it never accepts a stale pose.
            self.event(Event.KIND_PAGE, f"{stale}: waiting up to {wait_s - self.PAGE_FIRST_WAIT_S:.0f} s more for "
                       "the cameras to measure it again")
            self.clear_view(ledger)
            camera = self.node.wait_camera_page(self.arm, timeout=wait_s - self.PAGE_FIRST_WAIT_S,
                                                max_lost_s=max_lost_s, should_stop=self.stopping, since=since)
            if self.stopping():
                raise Cancelled()
        stale = "no page pose" if camera is None else geometry.stale_page(camera[1], max_lost_s)
        if stale and last is not None:
            self.event(Event.KIND_PAGE, f"{stale}: the overhead reads no print, so the wrist and the gauge holds "
                       "place the page from the last one")
            return None
        if camera is None:
            raise RuntimeError(f"no page pose for {self.arm} (page.source {self.stack['page']['source']})")
        if stale:
            raise RuntimeError(f"{stale}; give the overhead cameras a clear view of the page, then draw again")
        return camera

    def setup_page(self, program: dict | None, touch: bool, ledger=None) -> None:
        self.snag_guard(False)
        again, self._again = self._again, False
        self._carried_yaw, self._over_skin, self._located_heading = None, [], None
        self._page_fresh()
        kept = self._kept_page() if again else None
        fixed = self.stack["page"]["source"] == "fixed"
        last = None if kept is not None or fixed else self._last_page(again)
        found = self._wait_page(ledger, clear=again and kept is None and not fixed, last=last)
        cam, info = found or self._blind_prior(last, again)
        self.camera, self.overhead = cam, cam if found else np.asarray(last.get("overhead") or last["camera_at_setup"])
        self._marks_seen = [] if kept is not None else [m for _, m in self.node.measured_marks(
            self.clock.now() - float(self.stack["page"].get("max_lost_s", 5.0)))]
        self.overhead_at, self.slid = self.clock.now(), np.eye(4)
        if found is None:   # the last page, raised, is the prior: no correction from an overhead pose
            self.correction = self.prior_correction = np.eye(4)
        if kept is not None:   # the holds stand on it: its height and normal are the last holds', and from here on
            # it is the page in use, the old field on it: the locate's clearance and its corners are judged on it
            self.camera, self.correction, self.prior_correction = kept, np.eye(4), np.eye(4)
            cam = kept
        elif self._curved():   # placed afresh: the last setup's height, tilt and field are another skin's
            self._level_prior()
        locate = self.stack["page"].get("locate") or {}
        if locate.get("wrist") and self.sensing.wrist == "real":   # the camera rides the real arm
            cam, located = self.wrist_locate(cam, locate, ledger, program,
                                             surface=None if kept is None else self.surface(), alone=found is None)
            self.camera = cam = cam if found or not self._curved() else self._blind_placed(cam, located, last)
            info = {**info, "locate": located}
        cfg = self.stack["page"].get("height") or {}
        warn_m = float(cfg.get("warn_m", 0.002))
        sources = [("overhead", 0.0, float(cfg.get("overhead_sigma_m", 0.002)))]
        depth = (info.get("locate") or {}).get("depth")
        if depth is not None:
            sources.append(("wrist depth", float(depth["offset_m"]), float(depth["sigma_m"])))
        carried = self._page_prior(cam, info, sources, warn_m, carry=not again)
        if kept is not None:
            self.prior_correction = np.eye(4)
        self.correction = self.prior_correction
        self.page_record = {"camera_at_setup": cam.tolist(), "overhead": self.overhead.tolist(), **info, "touches": [],
                            "prior_correction": self.prior_correction.tolist(), "carried": carried}
        self.surface_field, self._surface_chart, self._patches, self._hold_frame = None, None, {}, None
        self._hold_low, self._provisional, self._start, self.rigid_total = 0.0, None, None, (np.eye(3), np.eye(3))
        self.go_ready(ledger)
        self._turn_carried()
        contacts, prior = [], cam @ self.prior_correction   # the page the gauge holds are made on
        if touch:
            contacts = self.page_touches(self._touch_layout(program), ledger,
                                         known=(carried or {}).get("offset_m") is not None or kept is not None)
        self._steady_overhead()
        n, o = cam[:3, 2], cam[:3, 3]
        normal, d = self._page_plane(contacts, cam, sources, cfg, warn_m)
        self._compare_planes(contacts, n, o)
        touched = self._hold_page(geometry.touched_page(cam, normal, d), prior, keep_prior=kept is not None)
        if found is None and not (self.page_record.get("holds") or {}).get("used"):
            raise RuntimeError(f"the overhead reads no print and the gauge holds did not place the page on it "
                               f"({self._holds_read()}): not drawn; give the cameras a print they can read (a darker "
                               "transfer, more light, the overhead's view clear), then draw again")
        self.correction = np.linalg.inv(cam) @ touched
        self.prior_correction = self.correction
        self.page_record.update(plane=[*normal.tolist(), d], touched=touched.tolist())
        self.page_record["used"] = self.used().tolist()
        self.page_record["correction"] = self.correction.tolist()
        self._fit_surface(program, ledger)
        self._confirm_holds()
        self._mark_reference(self.overhead_at)   # after every hold: the sheet as the holds placed the page on it
        if kept is None and found:   # a kept page's height is the holds', not the overhead's: the next run's start stays
            self._carry_paper(cam, info)   # the first's; a page the wrist placed has no overhead pose to carry it by
        self._reference_writing_height(program)

    # A set-up the overhead places no page for (an inked print it no longer reads: 2026-10-20, a forearm's tattoo ended
    # waiting for it while the wrist read 4-7 of its markers at every hold) starts from the last page, raised this far
    # along its normal: the overhead's page stood 4.7-9.3 mm over the gauge's skin, so its first hold and locate views
    # stand as high over its skin, unmoved, as they would on the overhead's page. The wrist's locate alone places it on
    # the print, the gauge holds then must (else not drawn), and its check holds confirm a far move as on any page. A
    # goal's first set-up starts from a page an earlier goal measured, the limb perhaps put back since: BLIND_RESUME_LIFT_M,
    # from which the first hold (25 mm over that page) still steps down to an unmoved skin in its three 5 mm steps.
    BLIND_LIFT_M, BLIND_RESUME_LIFT_M = 0.008, 0.015

    # The page record such a set-up starts from: mid-draw the page in use's, on a goal's first set-up the retained one
    # restore_page did not keep (_resumed); None without the gauge holds to place it on the print (page.locate.holds,
    # the gauge in use). _blind_prior: it raised, its info (source 'wrist'), never its ready pose: fresh holds measure
    # the page at the heading a new page takes (pen_heading, the dips' first), not the one an old page searched.
    def _last_page(self, again: bool) -> dict | None:
        record = self.page_record if again else self._resumed
        ok = record and "correction" in record and self._gauge_ready()   # unconfirmed or not: a seed for the locate
        return record if ok and (self.stack["page"].get("locate") or {}).get("holds") else None

    def _blind_prior(self, record: dict, again: bool) -> tuple[np.ndarray, dict]:
        page = np.asarray(record["camera_at_setup"], float) @ np.asarray(record["correction"], float)
        info = {k: record[k] for k in ("pattern_id", "print_id") if k in record}
        lift = self.BLIND_LIFT_M if again else self.BLIND_RESUME_LIFT_M
        return page @ geometry.translation([0.0, 0.0, lift]), {**info, "source": "wrist"}

    # A curved page so placed is raised, never lowered, to stand BLIND_LIFT_M over the highest skin the locate views'
    # depth planes read under it (less page.height.wrist_depth_bias_m; on the forearm tattoo's set-ups, 2026-10-20,
    # -0.1 to +9.1 mm over the holds' page), so its ready move and first hold rise with a limb risen since its page was
    # measured. With the print read, its holds stand off the wrist's page as the last page's did (_held_off: a set-up
    # placed afresh carries the gauge tip's ~7 mm across the pen axis in its correction, _level_prior): replayed on those
    # set-ups the holds' shift is 1.1 and 1.4 mm with it, 7.2 and 8.6 without (HOLD_SHIFT_CURVED_M, was 9 mm).
    def _blind_placed(self, cam: np.ndarray, located: dict, last: dict) -> np.ndarray:
        bias = float((self.stack["page"].get("height") or {}).get("wrist_depth_bias_m", 0.0))
        skin = max((v["depth"]["offset_m"] - bias for v in located.get("views", []) if v.get("depth")), default=-1.0)
        if (rise := skin + self.BLIND_LIFT_M) > 0.0:   # no view's plane: -1 m
            cam, located["raised_m"] = cam @ geometry.translation([0.0, 0.0, rise]), rise
            self.event(Event.KIND_PAGE, f"the locate views' depth reads the skin {skin * 1e3:+.1f} mm over the page it "
                       f"started from: raised {rise * 1e3:.1f} mm, {self.BLIND_LIFT_M * 1e3:.0f} mm over it")
        if located.get("fused"):
            self.correction, self._carried_yaw = _held_off(last)
            self.prior_correction = self.correction
        return cam

    # The holds' offset a set-up carries from the last page (_blind_placed, _level_prior) is mostly the gauge tip's ~7 mm
    # across the pen axis, fixed to the tool: the pages of five runs read it 85-103 deg from their ready heading at
    # headings -15 to -106 deg (2026-10-20). Carried unturned onto holds made at another heading (a new page takes the
    # dips', 488e27913) it stood them ~8.7 mm off. Turned here, about the page's normal, once go_ready chose the heading.
    def _turn_carried(self) -> None:
        was, self._carried_yaw = self._carried_yaw, None
        heading = (self.page_record.get("ready") or {}).get("heading")
        if was is None or heading is None:
            return
        turn = float(np.angle(np.exp(1j * (ready.heading_yaw(self.camera, np.asarray(heading, float)) - was))))
        c = self.prior_correction.copy()
        c[:2, 3] = [[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]] @ c[:2, 3]
        self.correction = self.prior_correction = c
        if abs(turn) > 1e-3:
            self.event(Event.KIND_PAGE, f"the holds' offset carried from the last page ({np.hypot(*c[:2, 3]) * 1e3:.1f} mm) "
                       f"turned {np.degrees(turn):+.0f} deg with the pen's heading")

    RISE_M = 0.004   # a sheet the markers read risen or sunk farther along the page's normal is not kept (_kept_page)
    WATCH_RISE_M = 0.004   # ... and between strokes is set up again at once (_watch_marks; under it a move waits, the
    # pen lifted, until it lasts page.track.watch hold_s). Depth, so not doubled with the stops (2026-10-20): a hover
    # whose gauge reads no skin has no other bound, the 3RL's needles reaching 4 mm from a tube 5 mm over the field

    def _kept_page(self) -> np.ndarray | None:
        """Where a page set up again mid-draw is held (_reacquire): the page in use moved in its own plane as the
        overhead's readings of the print's markers within page.max_lost_s put the sheet (_mark_moves), its height and
        normal the last gauge holds'. Never the overhead's pose: one of ten re-setups on 2026-10-19 took it with the
        arm over the sheet tilted 11.5 deg and raised 3.8 mm, and its holds stood that far off the skin. None without
        such a reading, or when they read the sheet risen or sunk past RISE_M along the page's normal (a lifted limb:
        its holds would stand that far into the skin, and the lift reads ~0.7 of itself as a slide across the rays), or
        moved past page.replan_* (a big move, the limb or sheet put down elsewhere: the last holds' height and normal
        are another place's): the overhead places the sheet, as at a first setup."""
        page = self.stack["page"]
        if _unconfirmed(self.page_record):
            self.event(Event.KIND_PAGE, "no check holds confirmed the page in use: the overhead places it, as at a first setup")
            return None
        moves = [] if self.camera is None else self._mark_moves(since=self.clock.now() - float(page.get("max_lost_s", 5.0)))
        moved = _median_move(moves)
        if moved is None:
            return None
        if abs(moved[3]) > self.RISE_M or _past(moved, page["replan_translation_m"], page["replan_rotation_rad"]):
            self.event(Event.KIND_PAGE, f"the print's markers read the sheet {moved[0] * 1e3:.1f} mm, {np.degrees(abs(moved[1])):.1f} "
                       f"deg, {moved[3] * 1e3:+.1f} mm along the page's normal (past {self.RISE_M * 1e3:.0f} mm along it "
                       "or page.replan_*): the overhead places the page, as at a first setup")
            return None
        frame = np.linalg.inv(self.slid) @ self.camera @ self.correction   # the page set up, without the trim
        kept = _median_pose(frame, [frame @ curved.in_plane(m[2]) for m in moves])
        d = np.linalg.inv(self.camera @ self.correction) @ kept
        self.event(Event.KIND_PAGE, f"the page set up again on the last holds' height and normal, moved "
                   f"{d[0, 3] * 1e3:+.1f} {d[1, 3] * 1e3:+.1f} mm {np.degrees(_yaw(d)):+.1f} deg in its plane as "
                   f"{len(moves)} readings of the print's markers put the sheet; the overhead's pose not read")
        return kept

    def _steady_overhead(self) -> None:
        """On a curved page the overhead's pose the sheet watch measures from is the median of those it measured while
        the page was set up (the page's centre and its turn, as sheet_slide takes them), not the one it started from:
        on the forearm (2026-10-07) single poses scattered +-5 deg, and one stroke after setup the watch read a still
        forearm as turned 8.8 deg."""
        if not self._curved() or self.overhead is None:
            return
        poses = self.node.measured_pages(self.arm, self.overhead_at)
        if len(poses) < 3:
            return
        first, self.overhead = self.overhead, _median_pose(self.overhead, poses)
        local = np.linalg.inv(first) @ self.overhead
        self.page_record["overhead"] = self.overhead.tolist()
        self.event(Event.KIND_PAGE, f"the overhead's page: the median of its {len(poses)} poses during the setup, "
                   f"{local[0, 3] * 1e3:+.1f} {local[1, 3] * 1e3:+.1f} mm, {np.degrees(_yaw(local)):+.1f} deg from its "
                   "first")

    def _mark_reference(self, since: float) -> None:
        """Where the overhead read the print's markers as the page was set up: each corner's median over its readings
        since `since`, the newest page.track.watch follow_s of them as the follow's median (else its newest three),
        which sheet_moved measures the sheet's moves from, with the markers the arm hid from them that the readings
        since `since` saw where it reads the sheet (geometry.marks_extended, within the watch's follow_m, follow_rad
        and rise_m); in the page record for a resume. None, and refresh_page judges the overhead's pose, when its pages
        carry no marker reading (a mock's) or the print has no marker layout. A page the wrist placed (_blind_prior)
        with no reading since `since`: {}, due once the overhead reads again."""
        cfg = self._watch()
        recent = geometry.window(self._marks(since), cfg["follow_s"], 1)
        due = not recent and self.page_record.get("source") == "wrist"   # never the readings from before it went blind
        marks = {} if due else geometry.marks_median(recent or [m for _, m in self._marks(0.0)[-3:]])
        if marks and self._layout():
            marks = geometry.marks_extended(marks, [m for _, m in self._marks(since)], self._printed,
                                            self._overhead_up(), cfg["follow_m"], cfg["follow_rad"], cfg["rise_m"])
        self.marked_at = self.clock.now()
        self.marks = marks if (marks or due) and self._layout() else None
        self.page_record["overhead_marks"] = None if self.marks is None else {
            str(i): c.tolist() for i, c in self.marks.items()}
        if self.marks is None and self.stack["page"]["source"] != "fixed":
            self.event(Event.KIND_PAGE, "the overhead's pages carry no marker readings of the print: a moved sheet is "
                       "judged by its pose")

    def _layout(self) -> dict:
        """The installed print's markers' printed corners (inspect.print_artwork), loaded once; {} without a tile."""
        if self._printed is None:
            from tatbot_session import inspect as ins

            tile = ins.print_artwork(self.stack["page"].get("pattern_id"))
            self._printed = {} if tile is None else tile.corners
        return self._printed

    def _timed_moves(self, since=None, marks=None) -> list:
        """(when, geometry.marks_moved from `marks`, setup's) of the overhead's readings since setup (`since`) that move."""
        marks = self.marks if marks is None else marks
        if not marks or not self._layout():
            return []
        up = self._overhead_up()
        key = None if up is None else np.round(up, 6).tobytes()   # each reading's move once per reference (~1 ms each)
        if not (self._moved[0] is marks and self._moved[1] == key) or len(self._moved[2]) > 500:
            self._moved = (marks, key, {})
        readings, cache = self._marks(self.overhead_at if since is None else since), self._moved[2]
        cache.update({id(r): (r, geometry.marks_moved(marks, r, self._printed, up)) for _, r in readings
                      if id(r) not in cache})   # the reading kept with its move: its id stays its own
        return [(t, m) for t, r in readings if (m := cache[id(r)][1]) is not None]

    def _overhead_up(self) -> np.ndarray | None:
        """The page in use's normal in the overhead's frame (marks_moved's rise); None before a page or the transform."""
        base = self.node.base_from_world(self.arm)
        return None if base is None or self.camera is None else base[:3, :3].T @ (self.camera @ self.correction)[:3, 2]

    # The overhead's readings of the print's markers since `since` (node.measured_marks) that read every marker within
    # its print's reach of the page in use's centre (the farthest printed corner and MARKS_REACH_M): on a silicone run
    # (2026-10-20) it read a second copy of the print 8 times in 304, a marker of each 0.33-0.79 m from the centre, one
    # went into the watch's reference, and the page was set up again twice for 154 and 271 mm "moves" the wrist never
    # saw. A limb's own move reads up to MARKS_REACH_M: the forearm tattoo's readings put a marker up to 85 mm from it,
    # and 50 mm past the print (page.replan_translation_m) left a 59-69 mm slide unread. All of them before the page or
    # the overhead's transform is known.
    MARKS_REACH_M = 0.2

    def _marks(self, since: float) -> list:
        readings, base = self.node.measured_marks(since), self.node.base_from_world(self.arm)
        if base is None or self.camera is None or not self._layout():
            return readings
        centre = (np.linalg.inv(base) @ self.used()[:, 3])[:3]
        reach = max(float(np.linalg.norm(c, axis=1).max()) for c in self._printed.values()) + self.MARKS_REACH_M
        return [(t, r) for t, r in readings if all(
            np.linalg.norm(geometry.corners_3d(c).mean(axis=0) - centre) <= reach for c in r.values())]

    def _mark_moves(self, marks=None, since=None) -> list:
        """The newest WATCH_POSES of _timed_moves' moves."""
        return [m for _, m in self._timed_moves(since, marks)][-self.WATCH_POSES:]

    def _watch(self) -> dict:
        """The overhead's watch of the sheet by its markers: geometry.WATCH, page.track.watch over it."""
        return {**geometry.WATCH, **((self.stack["page"].get("track") or {}).get("watch") or {})}

    def sheet_moved(self, marks=None, since=None) -> tuple[float, float, int, float] | None:
        """(slide m, turn rad, readings, rise m) of the sheet from where the overhead read its markers at setup (or
        `marks`): the medians of its newest readings' moves (_mark_moves); None without one, as with a lost page."""
        return _median_move(self._mark_moves(marks, since))

    def _overhead_now(self, newest: np.ndarray) -> np.ndarray:
        """The overhead's pose of the sheet now: `newest`, its newest pose; on a curved page the median of its newest
        WATCH_POSES (its match on a raised print scatters ±3-4 mm and ±5 deg pose to pose, and with the arm over the
        print it measures one now and then: a single stray pose stopped the swallow after three strokes, 2026-10-07)."""
        poses = self.node.measured_pages(self.arm, 0.0)[-self.WATCH_POSES:] if self._curved() else []
        return _median_pose(newest, poses) if len(poses) >= 3 else newest

    def _fit_surface(self, program, ledger=None) -> None:
        """page.surface.mode fitted: the height field over the page set up, from the gauge holds' points and the skin
        about them (curved.fit), checked with page.surface.check (_check_surface). Refused (RuntimeError, before any
        stroke) when the drawing reaches past page.surface.reach_m of every hold (an extrapolation there). A flat page
        fits flat and plans as the plane."""
        if not self._curved():
            return
        cfg = curved.config(self.stack)
        reads = {r["touch"]: r for r in self.page_record.get("gauge", []) if not r.get("check")}
        holds = [self._surface_hold(r) for _, r in sorted(reads.items())]
        gauged = sum(h["point"] is not None for h in holds)
        if gauged < 3:
            scattered = sum(h["point"] is None for h in holds)
            raise RuntimeError(f"a curved page needs three gauge holds at least, {gauged} read the skin: "
                               "page.surface.mode fitted cannot draw here"
                               + (f"; the wrist camera's frames scattered at {scattered} of {len(holds)} holds: glare "
                                  "on the skin or a faint stencil, so dim or turn the light away from it and set the "
                                  "page up again (2026-10-16: 1-7 mm under a bright lamp, 10 of 11 held once dimmed)"
                                  if scattered else ""))
        far = curved.coverage(program, [(np.linalg.inv(self.used()) @ [*r["point"], 1.0])[:2] for r in reads.values()],
                              float(cfg["reach_m"]))   # where the page in use, moved by the holds' shift, puts them
        if far > float(cfg["reach_m"]):
            raise RuntimeError(f"the drawing reaches {far * 1e3:.1f} mm from every gauge hold (page.surface.reach_m "
                               f"{float(cfg['reach_m']) * 1e3:.0f}): the skin's shape there is unmeasured; a smaller "
                               "drawing, or more holds")
        self._surface_fit(holds, cfg, program)
        if cfg["check"]:
            self._check_surface(holds, cfg, ledger)
        self._pen_height()

    def _surface_hold(self, read: dict) -> dict:
        """A gauge reading as a hold of curved.fit. One whose frames disagreed past gauge_spread_m at every try gives no
        point, as on the plane, but its skin: on the forearm's far side from the camera (2026-10-07) such holds read the
        skin under the tip 1-3 mm off, while their patches were the field's only long baseline across the arm."""
        return {"xy": read["xy"], "point": read["point"] if curved.disagree(read) <= self._gauge_limit() else None,
                "patch": self._patches.get(read["touch"], np.zeros((0, 3))),
                "tip": np.asarray(read["tcp"], float)[:3, 3] if "tcp" in read else None}

    def _surface_fit(self, holds: list, cfg: dict, program: dict | None = None) -> None:
        """The field through `holds` (curved.fit), in use and in the page record, its relief along `program`'s strokes."""
        size, drawing = self.stack["page"]["size_m"], self._stroke_uv(program) if program else np.zeros((0, 2))
        field, report = curved.fit(self.used(), holds, cfg, bounds=(size[0] / 2 - 0.002, size[1] / 2 - 0.002),
                                   turned=self.sensing.wrist == "real" and curved.load_turn(self.arm) is not None,
                                   drawing=drawing if len(drawing) else None)
        self.surface_field, self._surface_chart = field, None
        if self.run_dir is not None:
            np.savez_compressed(self.run_dir / "surface.npz", **{f"hold{k}": h["patch"] for k, h in enumerate(holds)})
        self.page_record["surface"] = {"field": field.as_dict(), "report": report}
        relief = (f"{report['relief_m'] * 1e3:.2f} mm of relief along the strokes, " if report["drawing_m"] else "") + (
            f"{report['relief_extent_m'] * 1e3:.2f} mm over all the skin fitted")
        self.event(Event.KIND_PAGE, "surface: a degree-{} field over {} points from {} gauge holds ({} under the tip), "
                   "{}; the wrist camera's turn {}, {:+.1f} {:+.1f} mrad {} about "
                   "their tips; each hold's skin off it {} mm (rms); without a hold the rest put its point {} mm "
                   "off".format(field.degree, report["points"], len(holds), report["gauged"], relief,
                       report["turn"], *(v * 1e3 for v in report["camera_tilt_rad"]),
                       "left in the patches" if report["turn"] == "calibrated" else "taken out of the patches",
                       ", ".join(f"{v * 1e3:.2f}" for v in report["patch_rms_m"]),
                       ", ".join(f"{v * 1e3:+.2f}" for v in report["loo_m"])))

    def skin_tilt(self):
        """The camera tilt the surface's fit took out of its holds' skin (curved.fit, its turn fitted per setup): the
        skin every later check and hover reads is turned by it alike (curved.untilt). None otherwise."""
        report = (self.page_record.get("surface") or {}).get("report") or {}
        return report.get("camera_tilt_rad") if report.get("turn") == "fitted" else None

    # Holds that each gave a point and predict one another within CHECK_ONE_M (curved.fit's leave-one-out), and a page
    # no check need confirm (_confirm_holds), are checked at one midpoint, the other three only when it is not counted
    # or reads past half max_loo_m. Over the 98 curved set-ups of 2026-10-13..20, 53 such took 59 checks for 212 and
    # found a check past the limit where the four did: the one among them read 0.74-0.96 mm at all four (a skin that
    # sank after its holds).
    CHECK_ONE_M = 0.0005

    def _check_surface(self, holds: list, cfg: dict, ledger=None) -> None:
        """Gauge holds at the midpoints of the corner holds' edges (curved.check_points, in no fit) read the skin against
        the field (_check_holds). Past page.surface.max_loo_m (the skin moved, or its shape is past the field) ink and
        a rehearsal alike go on, on the holds' field: two more holds read the skin at the worst check and halfway to the
        centre, and the pen rides or hovers higher (_pen_height) by the most any reading put the skin over it: never
        deeper for a miss, lighter lines. Refitted through the checks, one read low put a field 0.83 mm under the holds'
        beneath the strokes (2026-10-18's forearm, replayed); single readings scatter -1.6..+0.7 mm on a still arm."""
        limit, points = float(cfg["max_loo_m"]), curved.check_points([h["xy"] for h in holds])
        loo = np.abs(self.page_record["surface"]["report"]["loo_m"])   # NaN: a hold that gave no point
        one = bool(np.isfinite(loo).all() and loo.max() < self.CHECK_ONE_M) and not (
            self.page_record.get("holds") or {}).get("confirm")
        if one:   # the midpoint by the hold the others predict worst, first
            points = np.roll(points, -int(np.argmin(np.linalg.norm(points - holds[int(np.argmax(loo))]["xy"], axis=1))), 0)
        misses = self._check_holds(points[:1] if one else points, 100, "the skin between the holds", ledger)
        if one and (misses[0] is None or abs(misses[0]) > limit / 2):
            misses += self._check_holds(points[1:], 101, "the skin between the other holds", ledger)
        points = points[:len(misses)]
        counted = [m for m in misses if m is not None]
        self.page_record["surface"]["check_m"] = counted
        self._turned_checks(holds, dict(zip(map(tuple, np.round(points, 6)), misses, strict=True)), ledger)
        if not counted or max(abs(m) for m in counted) <= limit:
            return
        worst = max((k for k, m in enumerate(misses) if m is not None), key=lambda k: abs(misses[k]))
        at = [points[worst], (np.mean([h["xy"] for h in holds], axis=0) + points[worst]) / 2]
        fresh = [m for m in self._check_holds(at, 100 + len(points), "again at the worst check and halfway to the "
                                              "centre,", ledger) if m is not None]
        self.page_record["surface"].update(fresh_m=fresh, rise_m=(rise := max([0.0, *counted, *fresh])))
        held = "within" if fresh and max(abs(m) for m in fresh) <= limit else "still past, or not at"
        self.event(Event.KIND_PAGE, f"surface check: the skin between the holds read up to {max(map(abs, counted)) * 1e3:.2f} mm "
                   f"off the field (page.surface.max_loo_m {limit * 1e3:.2f}); fresh holds read it {held} the limit: the "
                   f"pen goes on {rise * 1e3:.2f} mm higher over the same field, the most any reading put the skin over it")

    TURNED_FIRST = 110   # the turned checks' touches (the square checks' from 100)

    def _turned_checks(self, holds: list, square: dict, ledger=None) -> None:
        """With page.surface.turn_pen (the strokes turned to the skin's normal) and holds made square to the page:
        page.surface.turned_checks gauge holds (2) at the check point where the field turns most and the one across
        from it, the pen turned to the field's normal there as a stroke's, each reading the skin against the field as
        a check does (curved.check_miss), less the square check's reading there where one was made (`square`: point ->
        miss). What they read is the arm's own error at the strokes' pose, which the strokes ride (curved.turned_lift,
        _on_surface) and the skin watch and the ride take out (_rise_at): on a skin round a bottle the hovers at one
        end read it 1.4-1.8 mm over the holds' field, the other end's 0.0 (2026-10-24)."""
        cfg = curved.config(self.stack)
        count = max(0, min(2, int(cfg["turned_checks"])))
        if not cfg["turn_pen"] or self._holds_along_skin() or not count:
            return
        surface, used, heading = self.surface(), self.used(), self.tcp()
        frame, above = used if self._hold_frame is None else self._hold_frame, float(self.motion["touch"].get(
            "start_above_m", 0.010))
        points = curved.check_points([h["xy"] for h in holds])
        turn = np.degrees(np.arccos(np.clip(surface.lift(points)[1] @ frame[:3, 2], -1.0, 1.0)))
        first = int(np.argmax(turn))
        picked, read_by, words, self._hop_from = [first, (first + len(points) // 2) % len(points)][:count], [], [], None
        for j, k in enumerate(picked):
            touch, xy = self.TURNED_FIRST + j, points[k]
            self._next_hold = (functools.partial(self._next_check, points[picked[j + 1]], above, frame, heading, surface,
                                                 turned=True) if j + 1 < len(picked) else None)
            read = self._gauge_try(touch, xy, above, above, frame, heading, ledger, frame=self._along_skin(surface, xy),
                                   check=True)[0]
            self._next_hold = None
            if j == len(picked) - 1:
                self.lift(f"touch {touch} lift", ledger)
            if read is None or curved.disagree(read) > self._gauge_limit():
                words.append(f"{turn[k]:.0f} deg at ({xy[0] * 1e3:+.1f} {xy[1] * 1e3:+.1f}) mm read nothing")
                continue
            read["check"] = read["turned"] = True
            tip = np.asarray(read["tcp"], float)[:3, 3] if "tcp" in read else None
            miss, read["point_miss_m"] = curved.check_miss(used, self.surface_field, self._patches.get(
                touch, np.zeros((0, 3))), read["point"], tip, self.skin_tilt())
            read["field_miss_m"], flat = miss, square.get(tuple(np.round(xy, 6)))
            read_by.append({"xy": [float(v) for v in xy], "turn_deg": float(turn[k]), "read_m": float(miss),
                            "square_m": flat, "lift_m": float(miss - (flat or 0.0))})
            words.append(f"{turn[k]:.0f} deg at ({xy[0] * 1e3:+.1f} {xy[1] * 1e3:+.1f}) mm reads {miss * 1e3:+.2f} mm"
                         + (f" ({flat * 1e3:+.2f} square)" if flat is not None else ""))
        self.page_record["surface"]["turned"] = read_by
        lifts = [r["lift_m"] for r in read_by] or [0.0]
        self.event(Event.KIND_PAGE, "surface check, the pen turned to the skin as the strokes will be: "
                   + "; ".join(words) + (f": the arm's own error at their pose, which they ride ({max(0.0, min(lifts)) * 1e3:+.2f}.."
                                         f"{max(0.0, max(lifts)) * 1e3:+.2f} mm along the drawing)" if read_by else ""))

    def _turned(self) -> list:
        """The page's turned checks' readings (_turned_checks), [] without."""
        return (self.page_record.get("surface") or {}).get("turned") or []

    def _rise_at(self, uv) -> np.ndarray:
        """(N,) how far the pen rides over the field at print points uv already: the surface check's rise_m and the
        turned checks' lift there (curved.turned_lift)."""
        turned = self._turned()
        return float((self.page_record.get("surface") or {}).get("rise_m", 0.0)) + curved.turned_lift(
            [r["xy"] for r in turned], [r["lift_m"] for r in turned], uv)

    def _pen_height(self) -> None:
        """The pen's path over the field: its resource's (tatbot_motion.pen_down) and the rise its page's surface check
        read (_check_surface), rebuilt with the pen (a resource change), the page (set up again) or a resumed page."""
        if self.pen is not None:
            cfg = self.motion["pen"][self.pen.mode]
            base = cfg["fraction"] * self.pen.stroke_m if self.pen.machine else cfg.get("height_m", cfg.get("lift_m"))
            self.pen = dataclasses.replace(self.pen, height_m=float(base) + float(
                (self.page_record.get("surface") or {}).get("rise_m", 0.0)))

    def _check_holds(self, points_xy, first: int, what: str, ledger=None) -> list:
        """Gauge holds at `points_xy` (touches first, first + 1, ...) and how far each reads the skin from the field
        (curved.check_miss: the skin about its tip, else its point; turned by the camera tilt the fit took out of the
        holds' skin); None for none read, or frames that disagree past gauge_spread_m. They are made as the holds were
        (the tool's tilt alike): a tool turned 8.6 deg further read the forearm's skin ~1 mm higher all round
        (2026-10-07), the arm's own error with its pose, which no field could take. Holds made along the skin's normals
        (_first_skin) have checks along the field's normals; holds made square to one page, checks on that page: each
        10 mm over the field (_over_field; at the holds' depth over a tilted page, 5.6-7.9 mm over a forearm's flank,
        2026-10-20), held again only away from it, and lifted after: up and over to the next (_hold_travel), off the
        last."""
        above, used, heading, misses = float(self.motion["touch"].get("start_above_m", 0.010)), self.used(), self.tcp(), []
        spread, words, surface, self._hop_from = self._gauge_limit(), [], self.surface(), None
        frame = used if self._hold_frame is None else self._hold_frame
        for k, xy in enumerate(points_xy):
            skin = self._along_skin(surface, xy) if self._holds_along_skin() else None
            self._next_hold = (functools.partial(self._next_check, points_xy[k + 1], above, frame, heading, surface)
                               if k + 1 < len(points_xy) else None)
            read = self._gauge_try(first + k, xy, above if skin is not None else self._over_field(
                surface, frame, xy, above), above, frame, heading, ledger, frame=skin, check=True)[0]
            self._next_hold = None
            if k == len(points_xy) - 1:   # the holds before it lift on the way to the next (_hold_travel)
                self.lift(f"touch {first + k} lift", ledger)
            misses.append(None)
            if read is None:
                continue
            read["check"] = True
            tip = np.asarray(read["tcp"], float)[:3, 3] if "tcp" in read else None
            miss, read["point_miss_m"] = curved.check_miss(used, self.surface_field, self._patches.get(
                first + k, np.zeros((0, 3))), read["point"], tip, self.skin_tilt())
            read["field_miss_m"] = miss
            if curved.disagree(read) <= spread:
                misses[-1] = miss
                words.append(f"{miss * 1e3:+.2f}")
            else:
                words.append(f"({miss * 1e3:+.2f}, its frames {curved.disagree(read) * 1e3:.2f} mm apart: not counted)")
        self.event(Event.KIND_PAGE, f"surface check: {what} reads " + (f"{', '.join(words)} mm from the field" if words
                                                                          else "nothing"))
        return misses

    @staticmethod
    def _over_field(surface, frame: np.ndarray, xy, above: float) -> float:
        """How far over `frame` at page point xy the tip stands `above` over `surface` along its normal there."""
        stand = above
        for _ in range(3):
            gap, normal = surface.height_over(frame[:3, :3] @ [*xy, stand] + frame[:3, 3])
            stand += (above - gap) / max(float(normal @ frame[:3, 2]), 0.5)
        return stand

    def _gauge_ready(self) -> bool:
        return self._gauge_mode() == "use" and self.sensing.ready(self.arm)

    def _touch_layout(self, program) -> np.ndarray:
        """Where the page is measured. Touches stay under a small drawing (geometry.touch_layout: a touch leaves
        a dot). The wrist gauge leaves none (geometry.gauge_layout): about a ladder 8 mm wide the touches' triangle
        was 8 mm across, too short a base for the page's tilt, and five points name a bad one (_gauge_plane)."""
        spread = float(self.stack.get("touch", {}).get("spread_m", 0.012))
        if not self._gauge_ready():
            return geometry.touch_layout(program, spread)
        if self._curved():   # a curved page: the holds' patches must cover the drawing (curved.layout)
            return curved.layout(program, self.stack["page"]["clear_m"],
                                 reach_m=float(curved.config(self.stack)["reach_m"]),
                                 inset_m=self.stack["page"].get("inset_m"))
        return geometry.gauge_layout(program, spread, self.stack["page"]["clear_m"], self.stack["page"].get("inset_m"))

    def _page_plane(self, contacts, cam, sources, cfg, warn_m) -> tuple[np.ndarray, float]:
        """The page's plane (normal, d): through the wrist gauge's points when they made it (_gauge_plane); the
        touches' plane with touch.fit_tilt; else the cameras' tilt at the fused height of every source."""
        from tatbot_session import inspect as ins

        n, o = cam[:3, 2], cam[:3, 3]
        gauged = [t for t in self.page_record["touches"] if t.get("method") == "wrist gauge"]
        plane = self._gauge_plane(contacts, n, cfg) if len(gauged) == len(contacts) >= 3 else None
        if plane is not None:
            return plane
        if contacts and self.stack.get("touch", {}).get("fit_tilt", False):
            return geometry.fit_plane(contacts, n)
        fused = ins.fuse_heights(self._height_sources(sources, contacts, n, o, cfg), warn_m=warn_m)
        self.page_record["height"] = fused
        self.event(Event.KIND_PAGE, "page height {:+.1f} mm over the overhead page (sigma {:.1f} mm) from {}".format(
            fused["offset_m"] * 1e3, fused["sigma_m"] * 1e3,
            ", ".join(f"{x['name']} {x['offset_m'] * 1e3:+.1f}" for x in fused["sources"])))
        if fused["warn"]:
            self.event(Event.KIND_PAGE, "page height sources disagree: " + ", ".join(fused["warn"]))
        return n, -(float(n @ o) + fused["offset_m"])

    def _gauge_plane(self, points, hint, cfg) -> tuple[np.ndarray, float] | None:
        """The plane through the wrist gauge's points (each good to ~0.1 mm). When one stands more than
        page.height.gauge_residual_m off it, the point the others' plane predicts worst is dropped, if the rest
        then fit; four points cannot say which is bad (their one spare residual spreads alike), five can. None,
        keeping the cameras' tilt, when they do not fit or tilt more than gauge_max_tilt_deg from the overhead's."""
        pts, tol = np.asarray(points, float), float(cfg.get("gauge_residual_m", 0.0005))
        keep = np.ones(len(pts), bool)
        normal, d = geometry.fit_plane(pts, hint)
        if self._curved():   # the residuals are the skin's curve: the height field takes them (_fit_surface)
            residual = pts @ normal + d
            self.page_record["height"] = {"method": "wrist gauge plane under a curved page",
                                          "residuals_m": residual.tolist()}
            self.event(Event.KIND_PAGE, "page plane through {} wrist gauge points under a curved page (their curve "
                       "{:.2f} mm from it at most)".format(len(pts), float(np.max(np.abs(residual))) * 1e3))
            return normal, d
        residual = pts @ normal + d
        if np.max(np.abs(residual)) > CURVED_PAGE_M:   # dropping one would hide a curve: a cylinder's corners are flat
            raise RuntimeError("the wrist gauge's points stand up to {:.1f} mm off one plane (residuals {} mm): a "
                               "curved skin (or a bad reading), and a flat page would ride high at its edges and deep "
                               "in its middle; draw it as a curved page (tatbot ros up --surface fitted)".format(
                                   float(np.max(np.abs(residual))) * 1e3,
                                   ", ".join(f"{r * 1e3:+.2f}" for r in residual)))
        if np.max(np.abs(residual)) > tol and len(pts) >= 5:
            loo = [abs(float(pts[k] @ n + c)) for k in range(len(pts))
                   for n, c in [geometry.fit_plane(np.delete(pts, k, axis=0), hint)]]
            keep[int(np.argmax(loo))] = False
            normal, d = geometry.fit_plane(pts[keep], hint)
            residual = pts @ normal + d
        tilt = float(np.degrees(np.arccos(np.clip(normal @ hint, -1.0, 1.0))))
        self.page_record["height"] = {"method": "wrist gauge plane", "residuals_m": residual.tolist(),
                                      "used": keep.tolist(), "tilt_from_overhead_deg": tilt}
        words = ", ".join(f"{r * 1e3:+.2f}" + ("" if k else " (dropped)") for r, k in zip(residual, keep, strict=True))
        if tilt > float(cfg.get("gauge_max_tilt_deg", 5.0)) or np.max(np.abs(residual[keep])) > tol:
            self.event(Event.KIND_PAGE, f"page plane: the wrist gauge's points tilt {tilt:.1f} deg from the overhead's "
                       f"(residuals {words} mm): not believed, the overhead's tilt kept")
            return None
        self.event(Event.KIND_PAGE, f"page plane through {int(keep.sum())} wrist gauge points (residuals {words} mm), "
                   f"{tilt:.2f} deg from the overhead's")
        return normal, d

    def _page_prior(self, cam, info, sources, warn_m, carry: bool = True) -> dict | None:
        """Where the touches start: the paper the last accepted touches on this page found (returned), else the cameras'
        fused height. The cameras are no witness of the paper's height: on 2026-10-03 the overhead depth plane stood
        8-10 mm and the wrist depth up to 15 mm over the paper the arm touched (1-3 mm over its base, as the 2026-09-29
        touch found it), so a search from them is long and meets the guard's trips in the air first. A mid-draw set-up
        (`carry` False) carries only its ready pose (_page_ready)."""
        from tatbot_session import inspect as ins

        # a set-up the overhead places mid-draw is a sheet that moved, rose or went unread: on a limb lifted 12 mm the
        # paper this run's first setup touched put the ready pose and the first hold in the skin (review, 2026-10-19)
        carried = self._carried_paper(cam, info, height=carry and not self._curved())   # curved: _level_prior
        if (paper := "offset_m" in (carried or {})) or len(sources) > 1 and not self._curved():
            offset = carried["offset_m"] if paper else ins.fuse_heights(sources, warn_m=warn_m)["offset_m"]
            self.prior_correction = geometry.translation([0.0, 0.0, offset])
        if paper:
            self.event(Event.KIND_PAGE, f"the paper as run {carried['run']} touched it, {carried['offset_m'] * 1e3:+.1f} mm "
                       f"over the overhead page, {carried['age_s'] / 60:.0f} min ago")
        return carried

    # A curved page placed afresh (not kept) starts on the overhead's own page, as a stack's first setup does: the last
    # correction keeps its place and turn in that page's plane (the gauge tip's offset from the TCP is ~7.9 mm of a fresh
    # set-up's shift), never its height or tilt, nor a carried paper's height, and the last field goes. The locate's
    # views, the ready move and the first hold are judged on it. The overhead reads each skin 4.7-9.3 mm over the gauge's,
    # its own way: a silicone's -7.0 mm, carried, stood a live forearm's first hold 7.6 mm over it (2026-10-20), and a
    # later one's would have left the ready move's floor 0.4-4.1 mm over the next. On the overhead's page the ready pose
    # and the first hold stand 14.5-18 mm up, and the hold steps down 5 mm at a time on frames that agree (_step).
    def _level_prior(self) -> None:
        c, yaw = self.prior_correction, float(np.arctan2(self.prior_correction[1, 0], self.prior_correction[0, 0]))
        if abs(c[2, 3]) > 1e-4 or c[2, 2] < 1.0 - 1e-6:
            self.event(Event.KIND_PAGE, "a curved page set up afresh: its holds start on the overhead's page, not "
                       f"{c[2, 3] * 1e3:+.1f} mm, {np.degrees(np.arccos(min(c[2, 2], 1.0))):.1f} deg off it as the last's")
        self.correction = self.prior_correction = geometry.rpy_matrix([c[0, 3], c[1, 3], 0.0], [0.0, 0.0, yaw])
        self.surface_field, self._surface_chart = None, None
        last, heading = self.page_record.get("camera_at_setup"), (self.page_record.get("ready") or {}).get("heading")
        if np.hypot(c[0, 3], c[1, 3]) > 0.0 and last is not None and heading is not None:   # turned with the pen's
            self._carried_yaw = ready.heading_yaw(np.asarray(last, float), np.asarray(heading, float))   # heading

    def _carry_path(self):
        from pathlib import Path

        return None if self.run_dir is None else Path(self.run_dir).parents[1] / f"page-touched-{self.arm}.json"

    def _carried(self) -> dict:   # where the last accepted touches found the paper (_carry_paper), else {}
        path = self._carry_path()
        return self.carried or (json.loads(path.read_text()) if path is not None and path.exists() else {})

    def _carried_paper(self, cam, info, height: bool = True) -> dict | None:
        """The paper the last accepted touches found, while it is the same page: the same print, the cameras' page
        within page.carry_move_m of where it was then, no older than page.carry_max_s; None otherwise. Its ready pose,
        and with `height`, within RISE_M of it along its normal, an offset along the camera page's normal (offset_m)."""
        page, record = self.stack["page"], self._carried()
        if not record or record.get("arm") != self.arm or record.get("pattern_id") != info.get("pattern_id"):
            return None
        then, age = np.asarray(record["camera"], float), self.clock.utc() - float(record["utc_s"])
        n, d = np.asarray(record["plane"][:3], float), float(record["plane"][3])
        carry_max_s = float(page.get("carry_max_s", 43200.0))   # 0: never (a simulator's runs share one clock)
        if carry_max_s <= 0.0 or age > carry_max_s or \
                np.linalg.norm(then[:3, 3] - cam[:3, 3]) > float(page.get("carry_move_m", 0.015)):
            return None
        out = {"run": record.get("run", ""), "age_s": age, "ready": record.get("ready")}   # the page's IK branch, heading
        if height and abs(rise := float((cam[:3, 3] - then[:3, 3]) @ n)) > self.RISE_M:   # a limb lifted under 15 mm
            self.event(Event.KIND_PAGE, f"the cameras' page stands {rise * 1e3:+.1f} mm along its normal from where "
                       f"run {out['run']} touched the paper: its height not carried")
        elif height:
            out["offset_m"] = -(float(n @ cam[:3, 3]) + d) / float(n @ cam[:3, 2])
        return out

    def _carry_paper(self, cam, info) -> None:
        """Keeps where this run's accepted touches tripped (the trips' own heights, not the first-contact
        estimates: the next run judges trips against it) for the next run on the same page, with the ready pose
        they were measured from: the next run takes the same IK branch, so the height it starts from is its own."""
        n = cam[:3, 2]
        trips = [float(np.asarray(t["tcp"])[:3, 3] @ n) for t in self.page_record["touches"]]
        if not trips:   # no touches this run: the paper is where it was
            return
        locate = self.page_record.get("locate") or {}
        first = next((v["azimuth_deg"] for v in locate.get("views") or () if "azimuth_deg" in v), None)
        self.carried = {"arm": self.arm, "pattern_id": info.get("pattern_id"), "camera": cam.tolist(),
                        "plane": [*n.tolist(), -float(np.mean(trips))], "utc_s": self.clock.utc(),
                        "run": self.run_dir.name if self.run_dir is not None else "",
                        "ready": self.page_record.get("ready"),
                        "view_deg": self._carried().get("view_deg") if first is None else first,
                        "view_base_deg": self._carried_view() if first is None else
                        float(first + np.degrees(_yaw(np.asarray(locate["overhead"])))) % 360.0}
        path = self._carry_path()
        if path is not None:
            path.write_text(json.dumps(self.carried, indent=1) + "\n")

    # The azimuth (deg) the last setup reached its first locate view at (the carried paper's), which the next searches
    # about first: every view (inspect.aim, 964 IK solves, 616 failing after 400 iterations) took 39-43 s on the ros
    # node before the arm moved (2026-10-20), three azimuths about 1 s. A view's azimuth is the page's (inspect.look_at)
    # and its reach the arm's, so it is carried in the arm's base too and turned onto this page: carried as the page's
    # alone, a print turned 87-125 deg since its last setup missed it and searched every view, 63-83 s, four setups
    # running (2026-10-24), where the first view stood within 182-253 deg of the base's x over every setup since
    # 2026-10-20, whichever print. Which markers the tool hides is the print's, though: on the same print turned
    # 59 deg (2026-10-23) the page's azimuth saw 10 markers and the base's 4. On the same print both are searched
    # together, the most markers seen first (inspect.aim); the page's alone when the base's lies within 15 deg of it.
    def _view_hint(self, page, pattern_id) -> list[float]:
        """The azimuths on `page` (base_from_page) the first view is searched about: the carried first view's turned
        from the base onto it, and on the same print its own; none without a carried view."""
        carried = self._carried()
        base = self._carried_view() if carried.get("arm") == self.arm else None
        if base is None:
            return []
        out = [(base - float(np.degrees(_yaw(np.asarray(page))))) % 360.0]
        if carried.get("pattern_id") == pattern_id and carried.get("view_deg") is not None:
            own = float(carried["view_deg"]) % 360.0
            out = [own] if abs((own - out[0] + 180.0) % 360.0 - 180.0) < 15.0 else [own, *out]
        return out

    def _carried_view(self) -> float | None:
        """The carried first view's azimuth about the base's z from its x; a file from before it was carried so, its
        page azimuth (view_deg) turned by the page it was carried with."""
        carried = self._carried()
        if carried.get("view_base_deg") is not None:
            return float(carried["view_base_deg"])
        if carried.get("view_deg") is None or carried.get("camera") is None:
            return None
        return float(carried["view_deg"] + np.degrees(_yaw(np.asarray(carried["camera"])))) % 360.0

    @staticmethod
    def _height_sources(sources, contacts, n, o, cfg) -> list:
        """Every sensor is a height along the page normal at the page centre: the overhead page, the wrist depth
        plane, each touch's first contact. Inverse-variance fused; one far from the rest is named in a warning,
        never dropped (bench 2026-09-26). The wrist camera sees the page, not the tip, so once the tip has
        touched, the wrist plane starts the touches and checks them but no longer votes."""
        touches = [(f"touch {i}", float((np.asarray(c) - o) @ n), float(cfg.get("touch_sigma_m", 0.002)))
                   for i, c in enumerate(contacts)]
        return [s for s in sources if not touches or s[0] != "wrist depth"] + touches

    def _compare_planes(self, contacts, n, o) -> None:
        """Every flat run measures the wrist depth plane against the touches, and the three planes' tilts against each
        other, so the wrist bias in stack.yaml and the choice of tilt rest on numbers (page_record["planes"]). Not on a
        curved page, where planes' angles are the skin's curve."""
        if self._curved():
            return

        def angle(a, b):
            return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))))

        nd = None
        views = [v["depth"] for v in (self.page_record.get("locate") or {}).get("views", []) if v.get("depth")]
        out, words = {}, []
        if views and contacts:
            nd = np.mean([v["normal"] for v in views], axis=0)
            nd /= np.linalg.norm(nd)
            p0 = o + float(np.median([v["offset_m"] for v in views])) * n
            over = [float((p0 - np.asarray(c)) @ nd / (n @ nd)) for c in contacts]
            bias = float(self.stack["page"]["height"].get("wrist_depth_bias_m", 0.0))
            out.update(wrist_over_touches_m=over, wrist_tilt_deg=angle(nd, n))
            words.append(f"wrist depth {np.mean(over) * 1e3:+.2f} mm over the touches (spread "
                         f"{np.ptp(over) * 1e3:.2f}, usually {bias * 1e3:+.1f}), {angle(nd, n):.2f} deg from the overhead")
            if abs(np.mean(over) - bias) > 0.002:
                words.append(f"the tip sits {(np.mean(over) - bias) * 1e3:+.1f} mm from where the wrist camera "
                             "expects it: a touch that tripped in the air, or the tool moved in its holder")
        if len(contacts) >= 3:
            nt, _ = geometry.fit_plane(contacts, n)
            out["touch_tilt_deg"] = angle(nt, n)
            words.append(f"touches {angle(nt, n):.2f} deg from the overhead"
                         + (f", {angle(nt, nd):.2f} deg from the wrist depth" if views and contacts and nd is not None else ""))
        if words:
            self.page_record["planes"] = out
            self.event(Event.KIND_PAGE, "planes: " + "; ".join(words))

    def _reference_writing_height(self, program):
        if not program or not program.get('writing_height_reference'):
            return
        from tatbot_cli.ros_writing_height import resolve

        self.pen, evidence = resolve(program, self.arm, self.run_dir, self.page_record, self.pen, self.workspace())
        self.page_record['writing_height_reference'] = evidence
        self.event(Event.KIND_PAGE, f"writing reference: buffer {self.pen.height_m * 1000:.2f} mm; "
                   f"at least {evidence['lift_m'] * 1000:.2f} mm over the completed source path")

    def refresh_page(self, program: dict, ledger, op_id: str = "") -> None:
        """Set the page up again (_reacquire, then _kept_page) when the sheet moved since this run set it up: first in
        every stroke's loop, before the watch or a hover moves the arm over it. With the print's markers kept, by them
        (_watch_marks), never the overhead's pose, whose tilt and height are a plane's over a curved skin: a still
        forearm's read 5.4 mm, 13.8 deg (2026-10-19). A page without them (a mock's) is judged by its pose, past
        page.replan_*."""
        page = self.stack["page"]
        if page["source"] == "fixed" or self.camera is None:
            return
        if self.marks is not None:
            return self._watch_marks(program, ledger, op_id)
        newest = self.node.camera_page(self.arm)
        if newest is None:
            return
        cam, now = self.page_overhead(), self._overhead_now(newest[0])
        shift, turn = float(np.linalg.norm(now[:3, 3] - cam[:3, 3])), geometry.rotation_angle(cam, now)
        if shift > page["replan_translation_m"] or turn > page["replan_rotation_rad"]:
            self._reacquire(program, ledger, f"the overhead reads the sheet {shift * 1e3:.1f} mm, {np.degrees(turn):.1f} "
                            "deg (by its pose) from where the page was set up, past page.replan_*", op_id)

    WAIT_S, CLEAR_S = 15, 3   # a move not yet held page.track.watch hold_s waits this long; with no reading of the
    # markers in CLEAR_S of it the arm clears the overhead's view (it hides the print)

    def _watch_marks(self, program: dict, ledger, op_id: str) -> None:
        """Set up again for a rise past WATCH_RISE_M by the newest five readings or a move _waited reads. A drift is
        _follow_marks'. A page the wrist placed blind takes its reference once three readings come in follow_s."""
        if self.marks == {}:
            if geometry.window(self._marks(self.marked_at), self._watch()["follow_s"]):
                self._rebase(ledger, "the overhead reads the print again", op_id)
            return
        moved = self.sheet_moved()
        if moved is not None and abs(moved[3]) > self.WATCH_RISE_M:
            return self._reacquire(program, ledger, f"the print's markers read the sheet {moved[3] * 1e3:+.1f} mm along "
                                   f"its normal, past {self.WATCH_RISE_M * 1e3:.0f} mm", op_id)
        verdict, why = self._waited(program, ledger, op_id)
        if verdict is not None and not self.stopping():
            self._reacquire(program, ledger, f"the print's markers read the sheet {why}", op_id)

    HOVER_WATCH_S = 10   # a hover stop this long (a view cleared for the overhead, a rehover): the watch judges again

    def _moved_meanwhile(self, program: dict, began: float, op_id: str) -> bool:
        """After a hover stop since `began` that lasted HOVER_WATCH_S, whether the overhead's readings read the sheet
        moved as _watch_marks judges it (a rise past WATCH_RISE_M, a move fired or not yet held): the stroke's loop then
        goes round again from refresh_page before it goes down, not on the judgement from before the hover. On
        2026-10-24 a forearm moved 45 mm under the pen: its hover cleared the view and waited 41 s, the readings that
        came in meanwhile read the move, and the stroke went down 18 mm off it (the mid-stroke stop lifted it)."""
        if not self.marks or self.clock.now() - began < self.HOVER_WATCH_S:
            return False
        moved = self.sheet_moved()
        if not ((moved is not None and abs(moved[3]) > self.WATCH_RISE_M) or self._sheet_watch(program)[0] is not None):
            return False
        self.event(Event.KIND_PAGE, "track: the overhead read the sheet move during the hover: judged again before the "
                   "stroke goes down", op_id)
        return True

    # _sheet_watch, a move not yet held waited out up to WAIT_S, the pen lifted and the machine off (MARKS_S more from a
    # cleared view: a lifted pen hid the right forearm's print before every stroke, 2026-10-20), back to ready from there
    # unless it lasts. Still pending at the end is a move.
    def _waited(self, program: dict | None, ledger, op_id: str = "") -> tuple[str | None, str]:
        verdict, why = self._sheet_watch(program)
        if verdict != "pending":
            return verdict, why
        self.machine_off()
        self.lift_off("lift while the overhead reads the sheet", ledger)
        self.event(Event.KIND_PAGE, f"the print's markers read the sheet {why}: waiting up to {self.WAIT_S} s for it to "
                   "stay there or come back", op_id)
        at, cleared, until = self.clock.now(), False, self.clock.now() + self.WAIT_S
        while verdict == "pending" and self.clock.now() < until and not self.stopping():
            if not cleared and self.clock.now() >= at + self.CLEAR_S and not self._timed_moves(at):
                self.clear_view(ledger)
                cleared, until = True, max(until, self.clock.now() + self.MARKS_S)
            self.clock.sleep(1.0)
            verdict, why = self._sheet_watch(program)
        if cleared and verdict is None and not self.stopping():
            self.go_ready(ledger)
        return verdict, why

    def _sheet_watch(self, program: dict | None) -> tuple[str | None, str]:
        """geometry.sheet_watch at the drawing from the page (_followed); its total page.track.watch's on a curved
        page, page.replan_* on a flat one."""
        page, cfg = self.stack["page"], self._watch()
        total = ((cfg["total_m"], cfg["total_rad"]) if self._curved() else
                 (page["replan_translation_m"], page["replan_rotation_rad"]))
        return geometry.sheet_watch(self._timed_moves(self.marked_at), self._drawing(program), self._followed(), total, cfg)

    # The move the page has followed since setup (slid), 3 x 3 in its plane: the markers' moves' frame.
    def _followed(self) -> np.ndarray:
        return (np.linalg.inv(used := self.used()) @ self.slid @ used)[np.ix_([0, 1, 3], [0, 1, 3])]

    def _drawing(self, program: dict | None) -> np.ndarray:
        """The drawing's box's corners and centre (page metres), where a move of the sheet is measured."""
        centre, half = geometry.program_box(program) or (np.zeros(2), np.zeros(2))
        return centre + half * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]])

    # The overhead's watch starts again from the sheet as it reads it now, the page in use standing for it: its markers
    # (_mark_reference) and poses over the newest page.track.watch follow_s, no slide followed since; a page row keeps
    # it.
    def _rebase(self, ledger, why: str, op_id: str = "") -> None:
        since = (now := self.clock.now()) - self._watch()["follow_s"]
        poses = self.node.measured_pages(self.arm, since)
        self.overhead = _median_pose(self.page_overhead(), poses) if poses else self.page_overhead()
        if self.tracker is not None:
            self.tracker.rebased()
        self.slid, self.overhead_at = np.eye(4), now
        self._mark_reference(since)
        self.placed = self.placed and {"page": self.camera @ self.correction, "overhead": self.overhead}
        self.page_record.update(overhead=self.overhead.tolist(), slid=self.slid.tolist())
        ledger.append("page", self.arm, page=self.page_record, workspace=self.workspace())
        self.event(Event.KIND_PAGE, f"track: {why}: its watch starts from the sheet as it reads it now", op_id)

    def guarded(self, traj, guard: int = names.GUARD_TIP_LAG) -> tuple[str, str, np.ndarray | None]:
        """Run a touch plan under a driver guard; returns (status, text, trip joints when it tripped). The
        tip-lag guard is armed at the first TOUCH knot, after the fast leg's own lag has decayed. The probe
        has no such false trips: its guard is armed at rest before the plan starts, so a probe that is silent
        or already triggered stops nothing but the start (status "refused"). The guard's own latch is
        unlatched here; self.touch_stop is the last knot reached."""
        probe = guard == names.GUARD_PROBE
        ours = names.LATCH_GUARD_PROBE if probe else names.LATCH_GUARD_TIP_LAG
        rows = np.flatnonzero(np.asarray(traj.phase) == P.PHASE_TOUCH)
        t_arm = float(traj.t[rows[0]]) if len(rows) and not probe else 0.0
        armed = {"done": probe}
        self.touch_stop = 0

        def tick(i: int) -> None:
            if not armed["done"] and traj.t[i] >= t_arm:
                armed["done"] = True
                self.io.write(guard_mode=guard)
            self.touch_stop = i
            self.feedback(phase=int(traj.phase[i]))

        try:
            if probe:
                self.io.write(guard_mode=guard)
                self.io.wait(lambda: self.io.latched or self.io.safety.get("guard_mode") == guard, 1.0)
            status, _, text = (("latched", 0, self.io.latch_text()) if self.io.latched
                               else self.io.execute(traj, should_stop=self.stopping, on_tick=tick))
        finally:
            stopped = self.io.latched and self.io.latch_reason == ours
            tripped = stopped and (not probe or self.io.flag("guard_tripped"))
            q_trip = self.io.trip_joints() if tripped else None
            refusal = self.probe_refusal() if stopped and not tripped else ""
            self.io.write(guard_mode=names.GUARD_NONE)
            self._snag = names.GUARD_NONE
        if stopped:
            ok, why = self.io.unlatch()
            if not ok:
                raise RuntimeError(f"unlatch after the touch {'trip' if tripped else 'guard'}: {why}")
        if tripped:
            return "tripped", text, self.io.measured() if q_trip is None else q_trip
        return ("refused", refusal, None) if stopped else (status, text, None)

    def probe_refusal(self) -> str:
        """Why the probe guard held without a touch (the driver's log line has it too)."""
        if self.io.flag("probe_triggered"):
            return "the probe guard held: the probe read triggered when armed (still touching, or a broken wire)"
        return "the probe guard held: no fresh probe frame (is the probe relay up, and the stack up with --probe?)"

    def refine_contact(self, samples, normal) -> dict | None:
        """First contact from a touch's recordings, three ways: the tip height along the page normal by FK
        (encoders; right until the arm and mount start to give), the tip force along the tool from the
        joint torques (its ramp extrapolated to zero, geometry.first_contact), and the overhead EE
        fiducial against FK of the same frame (the gap opens when the EE stops and the encoders do not,
        geometry.fiducial_contact_time). Returns the fused offset of first contact over the trip with
        each estimate, or None when neither found one."""
        from tatbot_description import names as dnames

        samples = samples[::4]   # 400 Hz joint states; 100 Hz is plenty for a 0.2 mm/s descent
        if len(samples) < 20:
            return None
        heights = np.array([float(self.kin.fk(q)[:3, 3] @ normal) for q, _, _ in samples])
        forces = [geometry.tip_force_along_tool(self.kin, q, tau) for q, tau, _ in samples]
        estimates = []
        force = geometry.first_contact(heights, forces, nominal_n_m=float(self.motion["touch"].get("give_n_m", 1500.0)))
        if force is not None:
            estimates.append(("force", force["offset_m"], 0.0006 if force["method"] == "fit" else 0.001))
        fid = None
        ee = getattr(self.io, "ee_samples", [])
        if len(ee) >= 16:
            ts = np.array([s[2] for s in samples])
            frame = dnames.EE_TRACKING_FRAME.format(arm=self.arm)
            et = np.array([e[0] for e in ee])
            inside = (et >= ts[0]) & (et <= ts[-1])
            if inside.sum() >= 16:
                idx = np.clip(np.searchsorted(ts, et[inside]), 0, len(ts) - 1)
                gap = [float((ee_p - self.kin.frame(samples[i][0], frame)[:3, 3]) @ normal)
                       for (_, ee_p, _), i in zip([e for e, k in zip(ee, inside, strict=True) if k], idx, strict=True)]
                fid = geometry.fiducial_contact_time(et[inside], gap)
                if fid is not None:
                    h_c = float(np.interp(fid["t_c"], ts, heights))
                    sigma = max(0.0003, float(np.std(gap)) / np.sqrt(fid["after"]))
                    estimates.append(("fiducial", h_c - heights[-1], sigma))
        top = float(self.motion["touch"].get("first_contact_max_m", 0.005))
        estimates = [e for e in estimates if -0.001 <= e[1] <= top]   # further from the trip is no contact estimate
        if not estimates:
            return None
        w = np.array([1.0 / s ** 2 for _, _, s in estimates])
        offset = float(np.sum(w * np.array([o for _, o, _ in estimates])) / w.sum())
        return {"offset_m": offset, "sigma_m": float(1.0 / np.sqrt(w.sum())),
                "estimates": [{"name": n, "offset_m": o, "sigma_m": s} for n, o, s in estimates],
                "force": force, "fiducial": fid, "method": "+".join(n for n, _, _ in estimates),
                "stiffness_n_m": (force or {}).get("stiffness_n_m", 0.0), "rise_n": (force or {}).get("rise_n", 0.0)}

    def page_touches(self, points_xy, ledger=None, known: bool = False) -> list[np.ndarray]:
        """A guarded descent at each point; a trip counts once another meets it. The guard's contact force,
        read from the joint torques, also trips with nothing under the pen (a joint's stiction winding up
        at 0.2 mm/s ramps it like the paper does: two of three touches, 13 mm over the page, 2026-10-03).
        So each trip is followed by a descent from touch.confirm_back_m over the lowest trip so far: the
        page stops it again within confirm_tol_m, while a trip in the air leaves room and the next goes lower.
        A trip the arm's own cables cause repeats at the same pose (three at one touch, 2026-10-03), so each
        confirming descent turns the tool confirm_turn_deg about its axis, alternating sides: every joint and
        the cables' pull change, the page does not. Once the paper is known (`known`: the prior is where the
        last accepted touches tripped; or this run's first accepted touch), a descent starts known_above_m over
        it and a trip more than known_band_m over it is in the air and not counted."""
        from tatbot_motion import plan_touch

        if self.camera is None:
            raise RuntimeError("no page pose known")
        cam = self.camera
        touch = self.motion["touch"]
        above = float(touch.get("start_above_m", 0.010))
        heading, contacts, self._hop_from = self.tcp(), [], None   # every touch's heading: the turns never add up
        paper = {"known": known, "level": 0.0}   # the paper's trip height over the prior page, once known
        for i, xy in enumerate(points_xy):
            trips, seen = [], []   # the trips that count, and every trip, as tip heights over the prior page (m)
            self._next_hold = (functools.partial(self._next_curved_hold, points_xy[i + 1], above, heading)
                               if self._curved() and i + 1 < len(points_xy) else None)
            gauged = self._gauge_point(i, xy, paper, above, heading, ledger)
            self._next_hold = None
            if gauged is not None:   # page.height.gauge use: the gauge is the touch ([]: no steady reading there)
                contacts += gauged
                if i == len(points_xy) - 1:   # the holds before it lift on the way to the next (_hold_travel)
                    self.lift(f"touch {i} lift", ledger)
                continue
            while True:
                prior = cam @ self.prior_correction
                start_above, start = self._touch_start(prior, xy, trips, len(seen), paper, heading)
                traj = self.knots(plan_touch(q_seed=self.io.seed(), start_base_from_tcp=start,
                                             direction=-prior[:3, 2], max_travel_m=float(touch["max_travel_m"]),
                                             kin=self.kin, motion=self.motion, prior_distance_m=start_above))
                self.io.record(True)
                try:
                    status, text, q_trip = self.guarded(traj)
                finally:
                    samples = self.io.record(False)
                if status == "tripped":
                    q_arr = self.io.measured() if q_trip is None else q_trip
                    contact = self.kin.fk(q_arr)
                    seen.append(float((contact[:3, 3] - prior[:3, 3]) @ prior[:3, 2]))
                    self._trip_snapshot(f"{i}-{len(seen)}", q_arr)
                    if not self._trip_met(i, trips, seen, paper):
                        self.lift(f"touch {i} lift", ledger)   # never plan the next descent from a loaded trip
                        continue
                    contacts.append(self._record_touch(i, xy, q_arr, contact, samples, prior[:3, 2], trips,
                                                       paper["level"] - seen[-1]))
                    self.lift(f"touch {i} lift", ledger)
                    break
                self._touch_held(i, status, text, ledger)
        return contacts

    def _touch_held(self, i: int, status: str, text: str, ledger=None) -> None:
        """A descent that ended without a trip: none within max_travel_m is an error, a cancel stops the run, a
        latch waits for the operator's decision (a LAND lands); the descent is then tried again."""
        if status == "ok":
            raise RuntimeError(f"touch {i} did not trip within {self.motion['touch']['max_travel_m'] * 1000:.0f} mm")
        if status == "cancelled" and not self.land_requested:
            raise Cancelled()
        if status != "cancelled":
            self.event(Event.KIND_LATCHED if status == "latched" else Event.KIND_ERROR, f"touch {i}: {text}")
        if self.wait_decision("latched", ledger=ledger) == rules.LAND:
            self.land()

    def _touch_start(self, prior, xy, trips: list, tries: int, paper: dict, heading) -> tuple[float, np.ndarray]:
        """Where a descent starts, over the prior page: confirm_back_m over the lowest trip that counts, else
        known_above_m over the known paper, else start_above_m; turned confirm_turn_deg about the tool's axis,
        alternating sides, from the second descent on."""
        touch = self.motion["touch"]
        if trips:
            above = min(trips) + float(touch.get("confirm_back_m", 0.003))
        elif paper["known"]:
            above = paper["level"] + float(touch.get("known_above_m", 0.004))
        else:
            above = float(touch.get("start_above_m", 0.010))
        side = 0.0 if not tries else (1.0 if tries % 2 else -1.0)
        turn = geometry.rpy_matrix([0.0, 0.0, 0.0], [0.0, 0.0, side * np.radians(float(touch.get("confirm_turn_deg", 20.0)))])
        return above, geometry.tool_down_pose(prior, xy, above, heading) @ turn

    def _trip_met(self, i: int, trips: list, seen: list, paper: dict) -> bool:
        """Judges the newest trip (seen[-1]); True once enough trips lie within touch.confirm_tol_m of the
        lowest that counts: two when the paper is known, three otherwise (two trips in the air met within
        1.3 mm, 2026-10-03). Refuses the touch after touch.confirm_trips."""
        touch = self.motion["touch"]
        tol, band = float(touch.get("confirm_tol_m", 0.0025)), float(touch.get("known_band_m", 0.003))
        over = seen[-1] - paper["level"]
        if paper["known"] and over < -band:
            paper["known"] = False
            self.event(Event.KIND_PAGE, f"touch {i}: a trip {over * 1e3:+.2f} mm under the known paper: it is not "
                       "where the last touches found it; searching for it")
        counted = not paper["known"] or over <= band
        if counted:
            trips.append(seen[-1])
        near = [h for h in trips if h - min(trips) <= tol] if trips else []
        if len(near) >= (2 if paper["known"] else 3):
            paper["known"], paper["level"] = True, float(np.mean(near))
            return True
        heights = ", ".join(f"{h * 1e3:+.2f}" for h in seen)
        if len(seen) >= int(touch.get("confirm_trips", 6)):
            raise RuntimeError(f"touch {i}: no paper in {len(seen)} trips ({heights} mm over the page prior"
                               + (f", the known paper at {paper['level'] * 1e3:+.2f}" if paper["known"] else "")
                               + "): the guard is tripping in the air")
        self.event(Event.KIND_TOUCH, f"touch {i}: trips at {heights} mm over the page prior"
                   + ("" if counted else f" (the last {over * 1e3:+.2f} over the known paper {paper['level'] * 1e3:+.2f}: in "
                      "the air, not counted)") + "; again")
        return False

    def _trip_snapshot(self, name: str, q) -> None:
        """The wrist D405's colour and raw depth at a trip, with the joints (touches/<name>.png and .npz): the
        witness of whether the pen met the paper, which the guard's force cannot tell (2026-10-03). Real
        hardware with the wrist camera only; a failed capture is reported and the touches go on."""
        self._wrist_frames(name, q, 1)

    def wrist_camera(self, serial: str):
        """The wrist D405 with depth; stack.yaml wrist_tap republishes each grab_both's last frames on the tatbot bus
        for the demo viewer, and opens its preview cameras while this one is open (scripts/lib/wrist_tap.py: display
        only, off at its first error)."""
        from tatbot_bridge import stack

        from tatbot_session import inspect as ins

        shared = {"repo": self.repo, "arm": self.arm, "open_bus": lambda: stack.open_bus(stack.bus_endpoint()),
                  "run_id": getattr(self.run_dir, "name", "")}
        tap = ins.wrist_tap.factory(self.stack.get("wrist_tap"), frames=(ins.CAMERA_FRAME, ins.DEPTH_FRAME), **shared)
        preview = ins.wrist_tap.previews(self.stack.get("wrist_tap"), **shared, open_camera=functools.partial(
            ins.WristCamera, depth=True))
        return ins.WristCamera(serial=serial, depth=True, tap=tap, preview=preview)

    # A setup's wrist D405 opens once, at its first reading, and stays open to the goal's end or the tracker
    # (track.Tracker.start). Opened at each one it cost ~2 s (the pipeline and 30 frames of auto exposure) at each of a
    # setup's 14-23 readings, and writing a reading's 50 files 1.0-1.3 s more (2026-10-20).
    def setup_camera(self):
        if self._cam is None:
            self._cam = self.wrist_camera("")   # stack.yaml's (ins.configured_serial)
        return self._cam

    def take_camera(self):
        cam, self._cam = self._cam, None
        return cam

    def _close_camera(self) -> None:
        if (cam := self.take_camera()) is not None:
            cam.close()

    # The setup's work in the background (frames written, _wrist_frames; readings' prints, _hold_tip), waited for and
    # each print put into its reading before the holds place the page, the tracker reads their frames, or the goal ends.
    def _settle(self) -> None:
        later, self._later = self._later, []
        for read, futures in later:
            results = [future.result() for future in futures]
            if read is None:
                continue
            outs = [out for out, _ in results]   # each camera's (_hold_tip), the session camera's first
            read["print_seen"] = sorted({int(i) for _, marks in results for i in marks})   # said when no placement
            if len(outs) > 1:
                read["print_cameras"] = outs
            read["print"] = result = self._one_vote(outs)
            if not (result or {}).get("accepted"):
                seen = read.pop("print_seen")
                self.event(Event.KIND_PAGE, f"touch {read['touch']}: its frames place no print: " + (
                    f"decoded {seen}" if seen else "no marker of it decoded") + ("" if result is None else
                    f", {result['markers']} kept their shape ({result['ids']}, {result['rms_m'] * 1e3:.2f} mm rms)"
                    + (", one alone farther from the page than a misread reaches" if result["markers"] == 1 else "")))

    def _wrist_frames(self, name: str, q, count: int, every: bool = False) -> tuple[list, list, dict] | None:
        """`count` colour and raw depth frames from the wrist D405 taken from now, the first written behind as
        touches/<name>.png and .npz with the joints q (with `every`, each further frame's depth too, as
        touches/<name>.<k>.npz); None without the real wrist (sensing.wrist), without the wrist camera, or when the
        capture fails (reported: the camera opens afresh at the next reading)."""
        if not self.wrist_on():
            return None
        try:
            cam = self.setup_camera()
            colors, _ = cam.grab_both(count, fresh=True)
        except RuntimeError as error:
            self._close_camera()
            self.event(Event.KIND_ERROR, f"wrist frames {name}: {error}")
            return None
        depths, meta = list(cam.raw_depth), cam.depth_metadata
        self._later.append((None, [self._writes.submit(self._write_frames, self.run_dir / "touches", name, colors,
                                                       depths, np.asarray(q, float), meta, every)]))
        return colors, depths, meta

    def wrist_on(self) -> bool:
        """Whether the setup reads the real wrist D405 (page.locate.wrist, a run directory to keep its frames)."""
        return (self.run_dir is not None and self.sensing.wrist == "real"
                and bool((self.stack["page"].get("locate") or {}).get("wrist")))

    def _further_frames(self, name: str | None, q, count: int, every: bool = False) -> list:
        """(view, its open camera, (colours, raw depths, meta)) of each further wrist camera page.locate.cameras
        names (track.views) that the session camera's preview has open (track.view_camera): `count` frames taken from
        now, with `name` written behind as touches/<role>/<name>.png and .npz (the tracker's seed reads the session
        camera's, touches/gauge-*.npz). A camera that fails a grab is said and left out; the session camera's goes on."""
        roles = (self.stack["page"].get("locate") or {}).get("cameras") or []
        path = gauge.calibration_path(self.arm)
        if self._cam is None or not roles or not path.exists():
            return []
        out = []
        for view in track.views(self, roles, json.loads(path.read_text())):
            if (other := track.view_camera(self._cam, view)) is None:
                continue
            try:
                colors, _ = other.grab_both(count, fresh=True)
            except RuntimeError as error:
                self.event(Event.KIND_PAGE, f"{view['role']} frames {name or ''}: {error}; the session camera's alone")
                continue
            depths, meta = list(other.raw_depth), other.depth_metadata
            if name is not None and self.run_dir is not None:
                self._later.append((None, [self._writes.submit(
                    self._write_frames, self.run_dir / "touches" / view["role"], name, colors, depths,
                    np.asarray(q, float), meta, every)]))
            out.append((view, other, (colors, depths, meta)))
        return out

    @staticmethod
    def _write_frames(out, name: str, colors, depths, q, meta, every: bool) -> None:
        import cv2

        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{name}.png"), colors[0])
        np.savez_compressed(out / f"{name}.npz", depth_raw=depths[0], q=q, meta=json.dumps(meta))
        if every:   # the rest of the series, to read each frame again offline: touches/<name>/<k>.npz and .jpg
            (out / name).mkdir(exist_ok=True)
            for k in range(1, len(colors)):
                cv2.imwrite(str(out / name / f"{k}.jpg"), colors[k], [cv2.IMWRITE_JPEG_QUALITY, 92])
                np.savez_compressed(out / name / f"{k}.npz", depth_raw=depths[k], q=q, meta=json.dumps(meta))

    def _gauge_mode(self) -> str:
        return str((self.stack["page"].get("height") or {}).get("gauge", "off"))

    def _gauge_limit(self) -> float:   # page.height.gauge_spread_m: a reading whose frames disagree past it is no point
        return float((self.stack["page"].get("height") or {}).get("gauge_spread_m", 0.0005))

    def _gauge_point(self, i: int, xy, paper: dict, above: float, heading, ledger=None) -> list | None:
        """page.height.gauge check or use, with the wrist gauge fitted for the fitted tool (gauge.py): the pen held
        where touch i starts, the paper's point under it by the wrist camera, recorded in page_record["gauge"]; for
        use also as the touch itself, returned in a list ([] when no reading held still). A reading whose frames
        disagree by more than page.height.gauge_spread_m (one put the paper 4 mm high, its frames 1.6 mm apart,
        2026-10-03) is taken again 4 mm to either side. None for check, or without the gauge (sensing.ready, which
        refuses a gauge measured with another tool)."""
        if self._gauge_mode() == "off" or not self.sensing.ready(self.arm):
            return None
        prior = self.camera @ self.prior_correction
        hover = paper["level"] + float(self.motion["touch"].get("known_above_m", 0.004)) if paper["known"] else above
        if self._curved():   # a known paper's plane is ~2 mm off a forearm's crest: every hold at the full height
            if not (later := bool(self.page_record.get("gauge"))):   # the setup's first hold
                self._seed_hold(prior, xy)
            hover, self._hold_frame = self._hold_start(prior, xy, above, later), prior
        for step in (0.0, 0.004, -0.004):
            at = np.asarray(xy, float) + [step, 0.0]
            read, hover = self._gauge_try(i, at, hover, above, prior, heading, ledger)
            if read is not None and curved.disagree(read) <= self._gauge_limit():
                if self._curved() and self._provisional is None:
                    self._provisional = self._first_skin(prior, read)
                    foot = np.linalg.inv(prior) @ np.r_[read["point"], 1.0]
                    over, self._marks_seen = self._marks_over(prior, foot[:2]), []
                    if over is not None:   # how the overhead reads this skin (_seed_hold)
                        self.anchor_offset = float(foot[2] - over)
                        self.event(Event.KIND_PAGE, f"touch {i}: the overhead's markers read the skin "
                                   f"{-self.anchor_offset * 1e3:.1f} mm over where the gauge put it")
                break
            if read is not None:
                self.event(Event.KIND_PAGE, f"touch {i}: the wrist gauge's frames disagree by "
                           f"{curved.disagree(read) * 1e3:.2f} mm at ({at[0] * 1e3:+.1f} {at[1] * 1e3:+.1f}) mm: not used")
        else:
            return [] if self._gauge_mode() == "use" else None
        if self._gauge_mode() != "use":
            return None
        pose = np.asarray(read["tcp"], float).copy()
        pose[:3, 3] = read["point"]
        self.page_record["touches"].append({"xy": at.tolist(), "tcp": pose.tolist(), "q": read["q"],
                                            "contact": read["point"], "method": "wrist gauge"})
        return [np.asarray(read["point"], float)]

    SEED_R_M = 0.012   # the overhead's marker corners this near a hold's foot give the skin's height there

    def _marks_over(self, prior: np.ndarray, xy) -> float | None:
        """The median height over `prior` of the overhead's marker corners within SEED_R_M of page point xy, in 3-D
        (geometry.corners_3d) as it read them when a first setup began; None with under four, or no transform."""
        base = self.node.base_from_world(self.arm)
        if base is None or not self._marks_seen:
            return None
        pts = np.concatenate([geometry.corners_3d(c) for reading in self._marks_seen for c in reading.values()])
        local = np.c_[pts, np.ones(len(pts))] @ (np.linalg.inv(prior) @ base).T
        near = local[np.hypot(*(local[:, :2] - np.asarray(xy, float)[:2]).T) < self.SEED_R_M, 2]
        return float(np.median(near)) if len(near) >= 4 else None

    def _keep_skin(self, view: dict, cam, depths, q) -> None:   # an over-centre view's, for the first curved hold
        if "over_m" in view:
            self._over_skin.append(self._skin_seen(cam, depths, q))

    def _skin_seen(self, cam, depths, q) -> dict:
        """An over-centre locate view's depth as points in the base (ros_depth.base_points), with the wrist gauge's tip
        (its calibration's, less its offset along the pen) and the tcp there in the base: _skin_over's."""
        from tatbot_cli import ros_depth

        from tatbot_session import inspect as ins

        points, _ = ros_depth.base_points(cam, depths, self.kin.frame(q, f"{self.arm}/realsense_depth_optical_frame"))
        gauged = gauge.calibration_path(self.arm)
        cal = json.loads(gauged.read_text()) if gauged.exists() else {}
        tcp = self.kin.fk(q)
        tip = None
        if cal.get("tip_cam_m") is not None:
            tip = (self.kin.frame(q, ins.camera_frame(self.arm)) @ np.r_[np.asarray(cal["tip_cam_m"], float), 1.0])[:3]
            tip = tip + float(cal.get("offset_m", 0.0)) * tcp[:3, 2]   # the gauge reads the tip over the skin less it
        return {"points": points.astype(np.float32), "tcp": tcp[:3, 3], "tip": tip}

    OVER_SEED_R_M = 0.005     # the over-centre views' depth points this near a hold's foot give the skin's height there
    OVER_SEED_POINTS = 50     # ... with this many at least (the pen's own points, 10 mm under its tip or more, left out)

    def _skin_over(self, prior: np.ndarray, xy) -> tuple[float, int] | None:
        """(the skin's height over `prior` at page point xy as the wrist gauge would put it, the points it rests on) by
        the setup's over-centre locate views (_skin_seen) in the order taken, the lowest first: the median height of
        their depth points within OVER_SEED_R_M of xy, raised by the tcp's height over the gauge's tip, from which a
        hold reads the skin. None with under OVER_SEED_POINTS."""
        inv = np.linalg.inv(prior)
        for seen in self._over_skin:
            local = seen["points"] @ inv[:3, :3].T + inv[:3, 3]
            tcp = inv[:3, :3] @ seen["tcp"] + inv[:3, 3]
            tip = tcp if seen["tip"] is None else inv[:3, :3] @ seen["tip"] + inv[:3, 3]
            near = local[(np.hypot(*(local[:, :2] - np.asarray(xy, float)[:2]).T) < self.OVER_SEED_R_M)
                         & (local[:, 2] < tip[2] - 0.010), 2]
            if len(near) >= self.OVER_SEED_POINTS:
                return float(np.median(near)) + float(tcp[2] - tip[2]), len(near)
        return None

    def _seed_hold(self, prior: np.ndarray, xy) -> None:
        """A first setup's first curved hold stands touch.start_above_m over the higher of `prior` and the skin under
        it: as the locate's over-centre views' depth puts it (_skin_over), the gauge's own camera from 40-90 mm higher
        in the hold's arm configuration; else as the overhead's markers put it (_marks_over), less the offset the last
        setup's first hold read between them and the skin. Never lower than over `prior`, and the holds measure and are
        held again as before (_step). On 2026-10-19 the markers read silicone 8.1-9.3 mm over the gauge's skin, then a
        live forearm 6.1: the silicone's page, still the prior, stood the forearm's first hold 7.6 mm over its skin, the
        markers would have 8.0; round a bottle (2026-10-24) the first hold stood 4.2 mm over the skin and was held again
        6.0 mm higher, 9 s."""
        if (wrist := self._skin_over(prior, xy)) is not None:
            self._hold_low = min(self._hold_low, -wrist[0])
            self.event(Event.KIND_PAGE, f"the wrist's depth from over the page centre puts the skin {wrist[0] * 1e3:+.1f} "
                       f"mm over the page under the first hold ({wrist[1]} points): it stands "
                       f"{-self._hold_low * 1e3:.1f} mm higher")
            return
        over = self._marks_over(prior, xy)
        if over is None or self.anchor_offset is None:
            return
        skin = over + self.anchor_offset
        self._hold_low = min(self._hold_low, -skin)
        self.event(Event.KIND_PAGE, f"the overhead's markers put the skin {skin * 1e3:+.1f} mm over the page under the "
                   f"first hold (the last setup's read {-self.anchor_offset * 1e3:.1f} mm under them): it stands "
                   f"{-self._hold_low * 1e3:.1f} mm higher")

    # The holds after a curved page's first fly there open-loop and unguarded: inside the patch of the first hold that
    # read firm, each starts touch.start_above_m and HOLD_SKIN_M over that hold's own skin (_first_skin, a quadric
    # through its patch) along its normal there, and steps as any hold (_step); elsewhere HOLD_SKIN_M over the skin the
    # locate's over-centre views' depth puts there (_skin_over), else HOLD_STEP_M over the stand the last firm reading
    # needed no step at. Replayed over the last set-up of 63 curved runs (2026-10-13..20) no later hold started under
    # 9.1 mm over its field (by the stand alone 8.1), and 940 readings became 799 (1.0 mm: 766 and 9.0 mm; 2.0 mm:
    # 853). On the silicone tile (2026-10-24) the hold outside the first's patch stood 16.2 mm by the stand and was held
    # again, where the first hold, by that depth, read 9.96 mm.
    HOLD_SKIN_M = 0.0015

    def _hold_start(self, prior: np.ndarray, xy, above: float, later: bool) -> float:
        if later and self._start is not None and self._start[1].find_simplex(np.asarray(xy, float)[:2]) >= 0:
            return self._over_field(self._start[0], prior, xy, above + self.HOLD_SKIN_M)
        if later and (wrist := self._skin_over(prior, xy)) is not None:
            return above + wrist[0] + self.HOLD_SKIN_M
        return above - self._hold_low + self.HOLD_STEP_M * later   # the next skin's unread

    @staticmethod
    def _hull(prior: np.ndarray, patch):   # the patch's convex hull over `prior`'s plane (scipy Delaunay)
        from scipy.spatial import Delaunay

        return Delaunay((np.c_[patch, np.ones(len(patch))] @ np.linalg.inv(prior).T)[:, :2])

    HOLD_HEIGHT_M = 0.002   # a curved page's hold this far from touch.start_above_m over the skin is held again there,
    HOLD_STEP_M = 0.005     # toward the skin at most this far a step, and a further step only on a reading this near
    HOLD_AIM_M = 0.0015     # where the last aimed (141 agreeing steps' readings landed -0.67 to +1.13 mm off it)

    def _gauge_try(self, i, at, hover: float, above: float, prior, heading, ledger=None, frame=None, check=False):
        """(one hold's reading at `at`, the hover its next try starts at): `hover` over `prior`, or `above` along a
        skin's normal (`frame`: a check's; a hold's by page.surface.along_skin, _along_skin); a curved hold held again
        by _step. The holds after it start HOLD_STEP_M over where a firm reading needed no step (_hold_low)."""
        if not check:
            frame = self._along_skin(self._provisional, at) if self._holds_along_skin() else None
        stand, aim, down = hover if frame is None else above, None, 0
        read = self._gauge_hold(i, at, stand, prior, heading, ledger, frame=frame)
        for _ in range(6):
            step = self._step(read, above, aim, check or down >= 3) if read is not None and self._curved() else 0.0
            if not step:
                break
            stand, aim, down = stand - step, read["height_m"] - step, down + (step > 0)
            read = self._gauge_hold(i, at, stand, prior, heading, ledger, frame=frame)
        if frame is None and not check and not step and read is not None and self._firm(read):
            self._hold_low = above - stand
        return read, stand if frame is None else hover

    def _firm(self, read: dict) -> bool:   # frames that agree, three pooled groups at least (one "agrees" with itself)
        return len(read.get("heights_m") or []) >= 3 and curved.disagree(read) <= self._gauge_limit()

    def _step(self, read: dict, above: float, aim, up_only: bool) -> float:
        """How far a curved hold's tip goes toward the skin (negative: away) to stand `above` over it by `read`: away
        by the nearest frame on any reading, toward by it and HOLD_STEP_M at most on a firm one. On 2026-10-20's
        forearm flank frames 7.9-19.7 mm apart took the tip 8.7 mm down by their median, to 4.8-7.1 mm over the skin."""
        near, apart, h = min(read.get("heights_m") or [read["height_m"]]), curved.disagree(read), read["height_m"]
        toward = (not up_only and near > above and h - above > self.HOLD_HEIGHT_M and self._firm(read)
                  and (aim is None or abs(h - aim) <= self.HOLD_AIM_M))
        step = near - above if near < above - self.HOLD_HEIGHT_M else min(near - above, self.HOLD_STEP_M) * toward
        if step:
            self.event(Event.KIND_PAGE, f"touch {read['touch']}: the tip stood {h * 1e3:.1f} mm over the skin (its "
                       f"frames {apart * 1e3:.2f} mm apart, the nearest {near * 1e3:.1f}); held again "
                       f"{abs(step) * 1e3:.1f} mm {'lower' if step > 0 else 'higher'}")
        return step

    def _first_skin(self, prior: np.ndarray, read: dict):
        """A curved page's first hold's skin as a surface over `prior` (tatbot_motion.surface.PageSurface): a quadric
        through its patch and its point, along whose normals the other holds and the check holds stand (_along_skin),
        and over which, read firm, they start (_hold_start); None when its patch is too small. Held square to one page
        the tool stood up to 13 deg off a forearm's skin at the corners, and 8.6 deg over read the skin ~1.1 mm higher
        (2026-10-07): the arm's error with its pose, which the pen, turned to each normal, meets too. Along the normals
        each hold is made in the pose the pen draws in there, the camera seeing the skin alike from every hold."""
        skin = self._skin_of(prior, self._patches.get(read["touch"]), read["point"])
        if skin is None:
            return None
        self._start = (skin, self._hull(prior, self._patches[read["touch"]])) if self._firm(read) else None
        tilt = max(np.degrees(np.arccos(np.clip(skin.normals_page([[x, y]])[0, 2], -1.0, 1.0)))
                   for x in (-0.012, 0.012) for y in (-0.012, 0.012))
        self.event(Event.KIND_PAGE, f"touch {read['touch']}: its skin turns up to {tilt:.1f} deg from the page 12 mm "
                   "from the centre" + (": the other holds stand along its normals"
                                        if self._holds_along_skin_cfg() else ""))
        return skin

    def _holds_along_skin_cfg(self) -> bool:
        return bool(curved.config(self.stack)["along_skin"])

    def _holds_along_skin(self) -> bool:
        """page.surface.along_skin, with a first skin to stand along. Off by default: along the normals the far flank
        reads (all five holds, 2026-10-07), but the arm's error with each hold's tilt turns its patch, and the field
        came out 1-3 mm off its own holds (2026-10-07)."""
        return self._provisional is not None and self._holds_along_skin_cfg()

    @staticmethod
    def _skin_of(prior: np.ndarray, patch, point):
        """One hold's skin as a quadric over `prior` (a PageSurface) through its patch and its point, or None."""
        from tatbot_motion.surface import PageSurface, fit_height

        if patch is None or len(patch) < 300:
            return None
        local = np.c_[np.r_[patch, [point]], np.ones(len(patch) + 1)] @ np.linalg.inv(prior).T
        try:
            return PageSurface(prior, fit_height(local[:, :3], np.r_[np.ones(len(patch)), 50.0], degree=2,
                                                 reject_m=0.001))
        except ValueError:
            return None

    @staticmethod
    def _along_skin(skin, xy) -> np.ndarray | None:
        """A page frame on `skin` (a PageSurface) at print point xy, z its normal there, its axes the page's turned
        the least (closeup._turn): a hold made over it with tool_down_pose(frame, (0, 0), ...) stands along the
        skin's normal. None without a skin."""
        if skin is None:
            return None
        point, normal = skin.lift([xy])
        frame = np.asarray(skin.base, float).copy()
        frame[:3, :3] = closeup._turn(frame[:3, 2], normal[0]) @ frame[:3, :3]
        frame[:3, 3] = point[0]
        return frame

    def _gauge_hold(self, i: int, xy, hover: float, prior, heading, ledger=None, frame=None) -> dict | None:
        """The pen held hover over page point xy, its wrist gauge's heights (sensing.gauge): the paper's point under the
        tip, recorded in page_record["gauge"]; None when the gauge makes out no pen and paper. On a curved page
        (page.surface.mode fitted) the gauge reads the skin under the tip (gauge.measure curved), the point stands
        along that skin's normal, and the skin about it is kept for the height field (_fit_surface)."""
        start = self._hold_pose(prior, xy, hover, heading, frame)
        self._hold_travel(i, start, ledger)
        self._plan_ahead()
        self.clock.sleep(float(self.motion["touch"].get("settle_s", 0.5)))
        q = self.io.measured()
        local = self._curved()
        count = int(curved.config(self.stack)["frames"] if local else
                    (self.stack["page"].get("height") or {}).get("gauge_frames", 5))
        tcp = self.kin.fk(q)
        sensed = self.sensing.gauge(q, tcp, prior, curved=local, count=count, name=f"gauge-{i}")
        if sensed is None:
            self.event(Event.KIND_PAGE, f"touch {i}: the wrist gauge made out no pen and paper")
            return None
        heights, point = list(sensed.heights_m), np.asarray(sensed.point_base, float)
        g, extra = float(np.median(heights)), {}
        if local:   # the skin about the tip for the field
            self._patches[i] = np.asarray(sensed.patch_base, float)
            extra = {"heights_m": heights,
                     "tool_tilt_deg": float(np.degrees(np.arccos(np.clip(-tcp[:3, 2] @ prior[:3, 2], -1.0, 1.0))))}
        read = {"touch": i, "xy": [float(v) for v in xy], "tcp": tcp.tolist(), "q": q.tolist(), "height_m": g,
                "spread_m": float(np.ptp(heights)), "frames": len(heights), "point": point.tolist(), **extra}
        for other in sensed.further:   # the further cameras' heights: in the hold's when they agree with its own
            hs, role = other["heights_m"], other["view"]["role"]
            read.setdefault("cameras", {})[role] = {"height_m": float(np.median(hs)) if hs else None, "used": other["used"]}
            if not other["used"]:
                self.event(Event.KIND_PAGE, f"touch {i}: {role} " + (f"puts the paper {np.median(hs) * 1e3:.2f} mm under "
                           "the tip, past the limit from the session camera's: its heights left out" if hs else
                           "made out no pen and paper"))
        self._hold_tip(read, sensed.raw, q, prior, local, sensed.further)
        self.page_record.setdefault("gauge", []).append(read)
        self.event(Event.KIND_PAGE, f"touch {i}: the wrist gauge puts the paper {g * 1e3:.2f} mm under the tip, at "
                   f"({point[0]:.4f} {point[1]:.4f} {point[2]:.4f}) m ({len(heights)} frames, spread "
                   f"{np.ptp(heights) * 1e3:.2f} mm)")
        return read

    HOP_TURN_RAD = 0.01   # a hold whose pen turns farther from the one the pen stands at is lifted off and travelled to
    # the pen this near the stand its hold's travel commanded stands there still: a settle moves a joint an encoder count
    # (0.38 mrad), up to 0.31 mm at the tip, after 29 of 2026-10-20's 56 holds; a lift moves it touch.lift_m
    HOP_AT_M = 0.002

    def _hold_travel(self, i: int, start: np.ndarray, ledger=None) -> None:
        """The pen to gauge hold i's stand `start`. From another hold's stand (_hop_from, which lift() clears;
        page_touches and _check_holds put that hold's lift off), in one move up and over onto it, again from where a
        latch stopped it: tatbot_motion.hop_via, rising touch.lift_m out along the page's normal at either end (_up).
        Between the 2026-10-20 tattoo's holds a lift, a ~1 s stop and a travel down took 8.4-11.5 s and three of the
        arm's stick-slip starts and stops; the hop takes 5.0-6.7 s and one. A hop the planner refuses, that would pass
        nearer the page than either end, whose pen turns past HOP_TURN_RAD, or from a pen HOP_AT_M off that stand lifts
        off the hold first and travels as before, and says why. A hold's own steps and tries aside travel straight."""
        from tatbot_motion import PlanError

        def travel(hop: bool):   # straight, or up and over from where the pen stands: the hop planned ahead if it holds
            ahead = self._ahead_for(start) if hop else None
            return ahead if ahead is not None else self._hop_plan(self.io.seed(), start, hop)

        here, held, plans, why = self.kin.fk(self.io.seed()), self._hop_from, [], ""
        if held is not None and held[0] != i:
            off, turn = float(np.linalg.norm(here[:3, 3] - held[1])), geometry.rotation_angle(here, start)
            why = (f"the pen stands {off * 1e3:.1f} mm off its stand" if off > self.HOP_AT_M else
                   f"its pen turns {turn:.3f} rad" if turn > self.HOP_TURN_RAD else "")
            try:
                plans = [] if why else [travel(True)]
                if plans and (low := self.lowest_over_page(plans[0])) < min(self._over(here[:3, 3]), self._over(start[:3, 3])) - 0.0005:
                    why, plans = f"it passes {low * 1e3:.1f} mm over the page, under its ends", []
            except (PlanError, ValueError) as error:
                why = str(error)
            if why:
                self.event(Event.KIND_REPLAN, f"touch {i}: no hop from touch {held[0]} ({why}): lifted off it first")
                self.lift(f"touch {held[0]} lift", ledger)
        hop, self._ahead = bool(plans), None
        self.move(lambda: plans.pop() if plans else travel(hop), P.PHASE_TRAVEL, f"touch {i} gauge", ledger)
        self._hop_from = (i, start[:3, 3].copy())

    @staticmethod
    def _hold_pose(prior: np.ndarray, xy, stand: float, heading, frame=None) -> np.ndarray:
        """A gauge hold's stand: the pen `stand` over `prior` at page point xy, or along a curved skin's normal
        (`frame`, _along_skin)."""
        return (geometry.tool_down_pose(prior, xy, stand, heading) if frame is None
                else geometry.tool_down_pose(frame, (0.0, 0.0), stand, heading))

    def _hop_plan(self, seed: np.ndarray, start: np.ndarray, hop: bool):
        """A hold's travel from joints `seed` to `start`: up and over (hop_via) or straight."""
        from tatbot_motion import hop_via, plan_travel

        p = self.kin.fk(seed)[:3, 3]
        way = hop_via(p, self._up(p), start[:3, 3], self._up(start[:3, 3]), self.motion["touch"]["lift_m"]) if hop else None
        return self.knots(plan_travel(q_seed=seed, base_from_tcp=start, kin=self.kin, motion=self.motion, via=way))

    # Between two holds the arm stood ~1.9 s after a reading before the next hop was sent, about half of it planning
    # the hop (2026-10-24). While a hold settles and reads, the hop to the stand its loop names next is planned on the
    # planner (idle through a setup), from where the arm stands: taken only for that same stand, the arm still there.
    def _plan_ahead(self) -> None:
        """The hop to the next hold its loop names (_next_hold: its stand as that hold will ask it, as the setup stands
        now), planned on the planner from the joints now; once a hold, so a hold read again plans none."""
        from tatbot_motion import PlanError

        upcoming, self._next_hold, self._ahead = self._next_hold, None, None
        if upcoming is None:
            return
        try:
            start = np.asarray(upcoming(), float)
        except (PlanError, ValueError):
            return
        self._ahead = (start, self._planner.submit(self._hop_plan, self.io.seed(), start, True))

    def _ahead_for(self, start: np.ndarray):
        """The hop planned ahead (_plan_ahead) when it is to `start` and the arm stands within motion.yaml
        dispatch_drift (pen up) of where it was planned from; else None, planned afresh as before."""
        from tatbot_motion import dispatch_drift

        ahead, self._ahead = self._ahead, None
        if ahead is None or not np.allclose(ahead[0], start, rtol=0.0, atol=1e-9):
            return None
        try:
            traj = ahead[1].result()
        except Exception:  # noqa: BLE001 - a failed plan ahead is planned again from the measured joints
            return None
        return traj if dispatch_drift(traj, self.io.measured(), self.kin, self.motion, pen_down=False)["ok"] else None

    def _next_curved_hold(self, xy, above: float, heading):
        """_plan_ahead's stand of the curved hold at page point xy after this one: _gauge_point's first try."""
        prior = self.camera @ self.prior_correction
        frame = self._along_skin(self._provisional, xy) if self._holds_along_skin() else None
        return self._hold_pose(prior, xy, self._hold_start(prior, xy, above, True) if frame is None else above, heading,
                               frame)

    def _next_check(self, xy, above: float, frame: np.ndarray, heading, surface, turned: bool = False):
        """_plan_ahead's stand of the check hold at page point xy after this one (_check_holds; turned: along the
        field's normal, _turned_checks)."""
        skin = self._along_skin(surface, xy) if turned or self._holds_along_skin() else None
        return self._hold_pose(frame, xy, above if skin is not None else self._over_field(surface, frame, xy, above),
                               heading, skin)

    # Read behind the hold's step, which never waited for it (1-2 s a reading, 2026-10-20), into read["print"] (_settle).
    # A hold's frames are written on one worker and each camera's print read on another (READ_WORKERS): on one worker
    # both cameras' 25 frames written (0.95 s a camera on x86) and read (1.6 s a camera there, 19 markers a frame on the
    # silicone tile) took ~13 s a hold on the ros node, its holds 10 s apart, and held the setup 48 s after its holds and
    # 12 s after its checks (2026-10-24). A read is mostly the markers' edge fits in Python, under the GIL: four reads at
    # once took as long as one after another, so one worker a camera, not more, beside the hold's own planning.
    READ_WORKERS = 2

    def _hold_tip(self, read: dict, raw, q, prior, curved_page: bool, further=()) -> None:
        """Where a gauge hold's wrist frames put the tip on the print (inspect.hold_tip), with page.locate.holds and
        the installed tile print's markers; raw: the real wrist's (shots, frames, calibration), else None; prior: the
        page the hold was made on; a curved page's read on its skin (the first hold's, else its own) in the print's mm.
        `further` (GaugeRead.further): the further cameras' frames, each read alike on its own calibration and pose
        (the URDF's, as the tracker's further views), the hold's print one vote of them all (_one_vote)."""
        from tatbot_session import inspect as ins

        page, read["print"] = self.stack["page"], None
        if raw is None or not (page.get("locate") or {}).get("holds"):
            return
        (shots, frames, cal), turn = raw, curved.load_turn(self.arm)
        camera, first, patch, point = (self.kin.frame(q, ins.camera_frame(self.arm)), self._provisional,
                                       self._patches.get(read["touch"]), read["point"])
        # each camera's markers onto its wrist pane, at the stamp its tap published its frame at: taken now, before the
        # next hold's grab
        reads = [(cam, getattr(cam, "stamp_ns", None), shots, frames, cal["tip_cam_m"], None)
                 for cam in [self._cam]] + [(f["camera"], getattr(f["camera"], "stamp_ns", None), f["shots"], f["frames"],
                                            f["view"]["cal"]["tip_cam_m"], self.kin.frame(q, f["view"]["frame"]))
                                           for f in further]

        def tip(cam, stamp, shots, got, tip_cam, base):   # one camera's, off the kinematics, on what the hold held
            tile = ins.print_artwork(page.get("pattern_id"))
            skin = (first or self._skin_of(prior, patch, point)) if curved_page else None
            colors, _, meta, *every = shots
            base = base if base is not None else camera if skin is None else curved.camera_pose(camera, tip_cam, turn)
            marks = {}
            out = None if tile is None else ins.hold_tip(colors, got, meta, tip_cam, np.linalg.inv(base) @ prior,
                                                         page, tile, markers=marks, every=next(iter(every), None),
                                                         **({"surface": skin} if skin is not None else {}))
            ins.wrist_tap.show_markers(cam, stamp, np.shape(colors[-1]), f"hold {read['touch']}", marks, out)
            return out, marks
        self._later.append((read, [self._reads.submit(tip, *each) for each in reads]))   # one vote of them (_settle)

    SURE_MARKERS = 3   # a hold's camera reading the print by this many markers or more, accepted, is sure of it

    @classmethod
    def _one_vote(cls, prints: list) -> dict | None:
        """A hold's print from its cameras' hold_tip readings, the session camera's first: a hold votes once
        (_hold_page), its cameras sharing the stop's own error. Every camera sure (accepted, SURE_MARKERS or more):
        their tip and turn averaged, its markers the most one read, and each further camera's tip offset from the
        session camera's kept (offsets_m: holds_shift has no state for a mount that moves, so it is watched); else
        the session camera's if sure, else a further one's that is, else the first accepted, else the session
        camera's."""
        sure = [p for p in prints if p and p.get("accepted") and p.get("markers", 0) >= cls.SURE_MARKERS]
        if len(sure) == len(prints) > 1:
            return {**sure[0], "tip_m": np.mean([p["tip_m"] for p in sure], axis=0).tolist(),
                    "turn_rad": float(np.mean([p["turn_rad"] for p in sure])), "markers": max(p["markers"] for p in sure),
                    "cameras": len(sure), "offsets_m": [np.subtract(sure[0]["tip_m"][:2], p["tip_m"][:2]).tolist()
                                                        for p in sure[1:]]}
        return next(iter(sure), None) or next((p for p in prints if p and p.get("accepted")), prints[0])

    HOLD_SHIFT_MAX_M = 0.030   # a larger shift from the holds is not taken: the set-up's page stands
    # A curved page's holds past HOLD_SHIFT_CURVED_M move it only when every kept reading, and every one by two markers
    # or more, stands within HOLD_AGREE_M of the shift, those by two markers or more kept at two places or more (a hold
    # made again is its place again), and the surface check's holds confirm the move (_confirm_holds). The limit stood
    # against a coded skin's band aliasing (2026-10-17); a tile print's markers name their places, and ~7.9 mm of a
    # fresh set-up's shift is the tool: the gauge's calibrated tip stands that far across the axis from the drawing
    # TCP (2026-10-20, five set-ups). On a forearm its four holds read +5.2 -11.7 mm agreeing to 1.0 / 0.1 mm and its
    # check holds the same within 1.8: refused, the tracker moved strokes 12 mm on one marker, then set up again.
    # HOLD_AGREE_M, HOLD_CONFIRM_M and HOLD_SHIFT_MAX_M are twice what they were before the owner doubled every stop
    # (2026-10-20); the curved limit 12 mm, not 18 (9 before): past it nothing stops, the check holds cross-check the
    # move, and the 2026-10-20 set-ups' shifts read 7.9 mm at most, the tool's ~7.9 mm among them.
    HOLD_SHIFT_CURVED_M, HOLD_AGREE_M, HOLD_CONFIRM_M = 0.012, 0.003, 0.006

    def _holds_read(self) -> str:
        """How a setup's gauge holds read the print, for the refusal when they place no page: a faint bicep transfer
        (2026-10-24) the overhead read nowhere and the holds by one marker at two of five, which the refusal said as
        "give the overhead cameras a clear view" over a clear view."""
        if (holds := self.page_record.get("holds")) is not None:
            return f"{holds['holds']} holds put it {np.hypot(*holds['shift_m']) * 1e3:.1f} mm off, past the limit: not taken"
        last = {r.get("touch", k): r for k, r in enumerate(self.page_record.get("gauge", [])) if not r.get("check")}
        read = [r["print"].get("markers", 0) for r in last.values() if (r.get("print") or {}).get("accepted")]
        return (f"{len(read)} of {len(last)} holds read it, {sum(m >= 2 for m in read)} by two markers or more; "
                "three must agree")

    def _hold_page(self, touched: np.ndarray, prior: np.ndarray, *, keep_prior: bool = False,
                   shift: bool = True) -> np.ndarray:
        """`touched` (the paper's height and tilt) placed on the print where the gauge holds' wrist frames put the
        tool (inspect.holds_shift): `prior`, the page the holds were made on, moved by where the tip stood on the
        print, so the plan lands on the print whatever the locate's far views made of it. Without that, the last page
        they placed moved as the overhead saw the sheet move (_placed_moved), else `touched` as it is; with
        `keep_prior` (a resume: the retained page's place) prior's place."""
        from tatbot_session import inspect as ins

        self._settle()
        # Each hold's believed point is the gauge's skin point under the TCP, as the track takes the TCP's foot: the
        # commanded xy stands 10 mm up along the prior's normal, 0.4-1.2 mm off it over a curved skin (2026-10-19).
        # A hold votes once, by its last reading (the field's, _fit_surface, and the track's seed): a curved hold reads
        # again as it steps (_gauge_try), its first reading 5-10 mm higher, where the print's tip moved 0.14-0.28 mm a
        # mm of stand on a forearm (2026-10-20); counted, one hold read three times moved a page and set its turn, and
        # those readings put agreeing holds past HOLD_AGREE_M.
        back = np.linalg.inv(prior)
        last = {r.get("touch", k): r for k, r in enumerate(self.page_record.get("gauge", [])) if not r.get("check")}
        found = ins.holds_shift([{"xy": (back @ [*r["point"], 1.0])[:2], "tip_m": r["print"]["tip_m"],
                                  "margin": r["print"].get("margin"), "turn_rad": r["print"].get("turn_rad"),
                                  "markers": r["print"].get("markers", 0)}
                                 for r in last.values() if r.get("print")], turned=self._curved())
        if found is None and not keep_prior and self.placed is not None and self.overhead is not None:
            prior, keep_prior = self._placed_moved(), True
        if found is not None:
            dx, dy = (v * 1e3 for v in found["shift_m"])
            limit = self.HOLD_SHIFT_CURVED_M if self._curved() else self.HOLD_SHIFT_MAX_M
            past = bool(np.hypot(dx, dy) > limit * 1e3)   # past it, holds that agree move the page (_confirm_holds)
            used = shift and (not past or self._curved() and found["off_m"] <= self.HOLD_AGREE_M and found["places"] > 1)
            self.page_record["holds"] = {**found, "used": used, "confirm": used and past}
            self.event(Event.KIND_PAGE, "the gauge holds put the tool {:+.1f} {:+.1f} mm{} off the print ({} holds, "
                       "spread {:.1f} {:.1f} mm, the farthest {:.1f} mm off it, {} places read by two markers or more): "
                       "{}".format(dx, dy, f", {np.degrees(found['turn_rad']):+.2f} deg" if found["turn_rad"] else "",
                                   found["holds"], *(v * 1e3 for v in found["spread_m"]), found["off_m"] * 1e3,
                                   found["places"], f"past {limit * 1e3:.0f} mm, they agree: the page moved onto it, "
                                   "the check holds to confirm it" if used and past else "the page moved onto it"
                                   if used else f"past {limit * 1e3:.0f} mm, not used" if shift else
                                   "a resume keeps its page's place, not used"))
            if used:   # the turn with the shift: about the holds' centre (inspect.holds_shift)
                prior, keep_prior = prior @ np.linalg.inv(ins.correction_matrix(*found["move"])), True
        if not keep_prior:
            return touched
        d = np.linalg.inv(touched) @ prior
        page = touched @ ins.correction_matrix(d[0, 3], d[1, 3], float(np.arctan2(d[1, 0], d[0, 0])))
        if (found is not None and self.page_record["holds"]["used"]) or self.placed is not None:   # a move past the
            # curved limit is placed once the check holds confirm it (_confirm_holds), for a later set-up's holds that
            # read nothing (_placed_moved); never before
            self.placed = None if found and _unconfirmed(self.page_record) else {"page": page, "overhead": self.overhead}
        return page

    # A page its holds moved past HOLD_SHIFT_CURVED_M (_hold_page) stands when two or more of the surface check's holds,
    # each reading the print afresh at another place, put the tip within HOLD_CONFIRM_M (their median) of the print
    # point the moved page plans the TCP at; else (the skin moved between them, or the holds misread) the page is set
    # up again once, afresh (_new_page), and a page still unconfirmed then goes on, said, the hovers' stencil watch
    # judging moves from its holds' readings; it is never placed for a later set-up, whose holds are made where this
    # one's were (its seed). Refusing ended an inking draw for good at a mid-draw set-up (owner, 2026-10-20: adapt,
    # never stop and refuse); going straight on inked a page the newer check readings put past HOLD_CONFIRM_M.
    def _confirm_holds(self) -> None:
        """A page the holds moved past the curved limit, confirmed by the check holds' own readings of the print."""
        self._settle()
        if not (holds := self.page_record.get("holds") or {}).get("confirm"):
            return
        back = np.linalg.inv(self.camera @ self.correction)
        off = [np.asarray(r["print"]["tip_m"][:2]) - (back @ [*r["point"], 1.0])[:2] for r in self.page_record["gauge"]
               if r.get("check") and (r.get("print") or {}).get("accepted")]
        miss = holds["confirmed_m"] = float(np.linalg.norm(np.median(off, 0))) if len(off) > 1 else None
        holds["stands"] = stands = miss is not None and miss <= self.HOLD_CONFIRM_M
        text = f"{len(off)} check holds read the print" + ("" if miss is None else f" {miss * 1e3:.1f} mm from the moved page")
        self.placed = {"page": self.camera @ self.correction, "overhead": self.overhead} if stands else None
        if not stands:   # nor does it seed the next set-up's holds (setup_page)
            self.prior_correction = np.asarray(self.page_record["prior_correction"], float)
        self.event(Event.KIND_PAGE, text + (": it stands" if stands else ", not confirmed: set up again once (_new_page), "
                                            "still not, it goes on there, the stencil watch judging moves from it"))


    def _placed_moved(self) -> np.ndarray:
        """The last page the gauge holds placed, moved as the overhead saw its sheet move since (its pose now against
        its pose then): where holds that read nothing leave the page. The overhead's own pose is several mm off the
        print (7.7 mm on a half-skin, 2026-10-06, when a set-up page went by it alone), but how far the sheet moved
        it measures to ~1-2 mm."""
        moved = self.overhead @ np.linalg.inv(self.placed["overhead"])
        page = moved @ self.placed["page"]
        d = np.linalg.inv(self.placed["page"]) @ page
        self.event(Event.KIND_PAGE, "the gauge holds read nothing: the page is the last one they placed, moved as the "
                   f"overhead saw the sheet move ({d[0, 3] * 1e3:+.1f} {d[1, 3] * 1e3:+.1f} mm, "
                   f"{np.degrees(np.arctan2(d[1, 0], d[0, 0])):+.1f} deg)")
        return page

    def _record_touch(self, i: int, xy, q_arr, contact, samples, normal, trips, shift: float = 0.0) -> np.ndarray:
        """The confirmed touch's contact point, recorded: the last trip moved `shift` along the normal to the mean
        of the trips that agreed (the turns' own offsets, +-1 mm, cancel there), then its first contact refined
        from the last trip's recordings."""
        point = contact[:3, 3] + shift * normal
        refined = self.refine_contact(samples, normal)
        if refined is not None:
            point = point + refined["offset_m"] * normal
        ee = len(getattr(self.io, "ee_samples", []))
        self.page_record["touches"].append({"xy": [float(v) for v in xy], "tcp": contact.tolist(), "q": q_arr.tolist(),
                                            "contact": point.tolist(), "first_contact": refined, "trips_m": trips,
                                            "ee_samples": ee})
        first = "" if refined is None else ", first contact {:+.2f} mm over the trip ({})".format(
            refined["offset_m"] * 1000, ", ".join("{} {:+.2f}".format(x["name"], x["offset_m"] * 1000)
                                                  for x in refined["estimates"]))
        self.event(Event.KIND_TOUCH, "touch {} at ({:.4f} {:.4f} {:.4f}) m, trips at {} mm over the page prior, "
                   "{} EE fiducials{}".format(i, *point, ", ".join(f"{h * 1e3:+.2f}" for h in trips), ee, first))
        return point

    # --- draw ----------------------------------------------------------------------------------
    PREPLAN_DRAIN_S = 300.0  # a curved page's stroke plans take seconds (each point lifted along its own normal): a
                             # numpad dip between strokes ended a run after 1 s (2026-10-08), and a 90 mm stroke's plan on
                             # the forearm outlasted 30 s before two dips (2026-10-21); callers hold pen-up

    def stop_preplan(self):
        if self._preplan is not None:
            self._preplan[1].cancel()
        # A barrier drains earlier untracked work on this executor's single-worker planner.
        try:
            self._planner.submit(lambda: None).result(timeout=self.PREPLAN_DRAIN_S)
        except TimeoutError as exc:
            raise RuntimeError('speculative stroke planning is still running; wait for it to end') from exc
        self._preplan = None

    def prepare_resource(self, resource: dict) -> None:
        """Pen mode, contact heading and machine for one program resource; its tool must be the fitted one."""
        fitted = config.fitted_tool(self.repo, self.arm)
        if resource['tool']['id'] != fitted:
            raise RuntimeError(f"{resource['id']} needs the {resource['tool']['id']} tool, but config/workspace.yaml "
                               f"fits {fitted} on the {self.arm} arm: fit it there, `tatbot ros deploy`, `tatbot ros up`, "
                               "`tatbot ros calib run`, then resume the run")
        self.resource = resource
        self.motion = copy.deepcopy(self.motion)
        self.motion['pen']['mode'], self.tool_id = resource['pen_mode'], fitted
        # a resource may ride at its own fraction of the stroke (a height ladder); else motion.yaml's
        self.motion['pen']['ride']['fraction'] = float(resource.get('ride_fraction', self.ride_base))
        if resource['pen_mode'] == 'hover':   # a rehearsal (ros_program.rehearsal): its own height, the needles clear
            self.motion['pen']['hover'] = {**self.motion['pen'].get('hover', {}),
                                           **({'height_m': float(resource['hover_height_m'])}
                                              if 'hover_height_m' in resource else {})}
            self._hover_clear(fitted, float(self.motion['pen']['hover']['height_m']))
        self.prepare_pen({})

    HOVER_CLEAR_M = 0.001   # a hover keeps the tube's end this far over the needles' farthest reach

    def _hover_clear(self, tool_id: str, height_m: float) -> None:
        """Refuse a hover lower than a needle cartridge's needles can reach past its tube's end (needle_reach_mm, the
        tube's end the path's reference) plus HOVER_CLEAR_M: with the machine off they rest anywhere in their stroke."""
        raw = getattr(config.tool_datasheet(self.repo, tool_id), "raw", None) or {}
        reach = raw.get("needle_reach_mm")
        if reach is not None and height_m < float(reach) / 1000.0 + self.HOVER_CLEAR_M:
            raise RuntimeError(f"a hover {height_m * 1e3:.1f} mm over the page would leave {tool_id}'s needles (they reach "
                               f"{float(reach):.1f} mm past its tube) within {self.HOVER_CLEAR_M * 1e3:.0f} mm of the skin: "
                               "rehearse higher (--standoff-mm)")

    def draw(self, program: dict, ledger, *, from_op: str = "", touch: bool) -> str:
        """Draw every op not yet done. Returns 'complete'; raises Landed or Cancelled."""
        goal_start = self.clock.utc()
        self._goal_touch = touch
        ops, rows = program["ops"], ledger.rows()
        start = next((i for i, op in enumerate(ops) if op["id"] == from_op), 0) if from_op else 0
        self.drawn, self.charged = {}, set()
        self.trims = {row["resource"]: float(row["trim_m"]) for row in rows
                      if row.get("event") == "pen_trim" and row.get("arm") == self.arm}
        self.trim_end, self._fold, self._trim_cap = 0.0, 0.0, None
        self.dip_scales, self.inked_m, self.dip_asked, self.op_now = charge.scales(rows, self.arm), 0.0, False, ""
        self._skin_off = 0.0
        self.ride_base = float(self.motion["pen"]["ride"]["fraction"])   # this goal's fresh motion.yaml
        self._snag = None
        self.snag_guard(False)
        res = self._installed(program, rows, start)
        self.prepare_resource(res)
        self.event(Event.KIND_PAGE, f"resource {res['id']}: {res['ink_id']} in "
                   f"{res['tool']['id']}")
        marker = in_cap(self.arm)
        if marker is not None:
            from tatbot_session import cap

            cap.withdraw_after_restart(self, program, marker, ledger)
        page = self._resumed = self._retained_page(rows)
        if page is None and not rows:
            self._resumed = self._seed_page()
        if page is not None and self.restore_page(page, ledger):
            regauged = self._fresh = self._regauge(program, ledger)   # its first hover's reading on fresh holds or not
            self._reference_writing_height(program)   # the pen height is motion.yaml's now, not the page's
            if self.pen is not None:
                self.page_record["pen"] = dataclasses.asdict(self.pen)
            if regauged:
                ledger.append("page", self.arm, page=self.page_record, workspace=self.workspace())
        else:
            self._new_page(program, touch, ledger)
        if self.trims:
            self.event(Event.KIND_PEN_TRIM, "trims kept from the run: " + ", ".join(
                f"{res} {trim * 1e3:+.1f} mm" for res, trim in sorted(self.trims.items())))
            self._ease_trims(ledger, "the run resumed")
        if self.dip_scales:
            self.event(Event.KIND_DIP, "dip scales kept from the run: " + ", ".join(
                f"{res} x{scale:.1f}" for res, scale in sorted(self.dip_scales.items())))
        self.ledger = ledger
        self._chain_snap = self._correction = None
        self._track_since = goal_start
        self.tracker = track.Tracker.start(self, since=goal_start, centre=self._drawing(program)[-1])
        try:
            self._run_ops(program, start, ledger)
        finally:
            self.ledger = None
            if self.dip_asked:
                self.dip_asked = False
                self.event(Event.KIND_DIP, "the numpad's dip was not made before the goal ended; a resumed goal dips "
                           "before its first stroke")
            self.machine_off()
            self.snag_guard(False)
            self._stop_tracking()
        return "complete"

    def _track_lift(self, program: dict, op: dict, ledger) -> None:
        """Stencil tracking's measurement at a stroke's lifted end (none when the next op continues pen-down)."""
        if self.tracker is None or not self.lift_at_end(program, op, ledger.rows(), drawn_m=0.0):   # inked_m has it
            return
        try:
            self.tracker.after_lift(op)
        except Exception as error:   # tracking is a log: a failure ends it, never the draw
            self.event(Event.KIND_ERROR, f"track: {error}; not tracking")
            self._stop_tracking()
            if self._rides_measured():   # its hovers are an inking curved page's depth reading: the camera opened again
                try:
                    self.tracker = track.Tracker.start(self, since=self._track_since, centre=self._drawing(program)[-1])
                except Exception as again:  # noqa: BLE001 - without it each stroke rides by the readings it has
                    self.event(Event.KIND_ERROR, f"track: not started again: {again}")
                if self.tracker is not None:
                    self.event(Event.KIND_PAGE, "track: started again")

    def _correction_row(self) -> dict:
        """The dispatched op's print correction for its `sent` row (none without stencil tracking's correct)."""
        if self._correction is None:
            return {}
        return {"correction_m": [round(float(v), 6) for v in self._correction[0]],
                "correction_max_m": round(self._correction[1], 6)}

    def _stop_tracking(self) -> None:
        tracker, self.tracker = self.tracker, None
        if tracker is not None:
            try:
                tracker.close()
            except Exception as error:   # tracking is a log: it never decides how a draw ends
                self.event(Event.KIND_ERROR, f"track: {error}")

    def _installed(self, program: dict, rows, start: int) -> dict:
        """The resource on the arm now: the last tool change done (or sent, then resumed: the operator made
        it) before the next op; the program's first resource when none was."""
        ops = program["ops"]
        resources = {r["id"]: r for r in program["resources"]}
        nxt = ledger_mod.next_index(ops, rows, self.arm, start)
        limit = len(ops) if nxt is None else nxt + 1
        last = ledger_mod.last_events(rows, self.arm)
        made = ("done", "sent")   # a sent change was the operator's to make, which the resume says they did
        changed = [op for op in ops[:limit] if op["op"] == "tool_change" and last.get(op["id"], {}).get("event")
                   in (made[:1] if _rinses(resources[op["resource_id"]]) else made)]
        return resources[(changed[-1] if changed else ops[0])["resource_id"]]

    def workspace(self) -> dict:
        """The arm's config/workspace.yaml section (fitted tool, calibrated tip, joint offsets) as the ledger has it."""
        return json.loads(json.dumps(config.workspace_arm(self.repo, self.arm), default=str))

    SEED_MAX_AGE_S = 24 * 3600.0   # a seed only aims the search (and its wide aims); 3 h missed a print set up 3.7 h before

    def _seed_page(self) -> dict | None:
        """A fresh run's place to start the wrist's search when the overhead reads no print: the newest earlier run's
        retained page on this node, its print the stack's and its ledger under SEED_MAX_AGE_S old. Only a seed
        (_last_page, _blind_prior): the page is set up afresh, its field and correction measured again; nothing of it
        is restored. A small print the overhead cannot read (a 2x2 forearm piece, 2026-10-21) otherwise leaves a fresh
        run nothing to aim the wrist at after a restart."""
        if self.run_dir is None:
            return None
        import time

        pattern = self.stack["page"].get("pattern_id")
        for path in sorted(self.run_dir.parent.glob("*/ledger.jsonl"), key=lambda q: q.stat().st_mtime, reverse=True):
            if path.parent == self.run_dir or time.time() - path.stat().st_mtime > self.SEED_MAX_AGE_S:
                continue
            try:
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            except (OSError, ValueError):
                continue
            seed = self._retained_page(rows)
            if seed is not None and seed.get("pattern_id") == pattern and "correction" in seed:
                self.event(Event.KIND_PAGE, f"no page yet: the wrist starts its search where {path.parent.name} last "
                                            "placed this print; the page is set up afresh")
                return seed
        return None

    def _retained_page(self, rows) -> dict | None:
        """The run's last measured page, when the fitted tool and its calibration are unchanged since: a resume
        (after a cartridge swap, a landing) keeps it rather than touching the paper again. The last one the gauge
        holds placed on the print beats a later one they could not: on 2026-10-06 a resume's holds read nothing on a
        half-skin's inked print and the overhead and wrist locate alone put the page 7.7 mm from where the first
        half was drawn, on a sheet the overhead saw unmoved."""
        rows = [row for row in rows if row.get("arm") == self.arm and row.get("event") == "page"]
        if not rows or rows[-1].get("workspace") != self.workspace():
            return None
        pages = [row["page"] for row in rows if row.get("workspace") == self.workspace() and not _unconfirmed(row["page"])]
        placed = [page for page in pages if (page.get("holds") or {}).get("used")]
        return (placed or pages or [None])[-1]

    # A retained page is kept while the overhead sees its sheet this near (RETAIN_M, RETAIN_RAD: 3 mm, 2.5 deg until the
    # owner doubled every stop, 2026-10-20), and risen or sunk RETAIN_RISE_M at most: its old field stays under the
    # strokes, a rise is depth, and a hover whose gauge reads no skin has no other bound.
    RETAIN_M, RETAIN_RAD, RETAIN_RISE_M = 0.006, 0.087, 0.003

    def restore_page(self, page: dict, ledger=None) -> bool:
        """Use a retained page when the overhead still reads its sheet where it was set up (its markers, sheet_moved,
        their slide and rise along the page's normal together; a flat page kept before them: its own pose then, not the
        page located from it) within RETAIN_M and RETAIN_RAD, with the slide it followed from them; False (and the
        caller sets the page up afresh) when it cannot see the sheet or the sheet moved, or a curved page has no markers
        kept. An arm that stopped over the print hides it: the view is cleared and the overhead asked again first."""
        lost = float(self.stack["page"].get("max_lost_s", 5.0))
        setup = np.asarray(page.get("overhead") or (page.get("locate") or {}).get("overhead") or page["camera_at_setup"],
                           float)
        marks = {int(i): np.asarray(c, float) for i, c in (page.get("overhead_marks") or {}).items()}
        if (page.get("holds") or {}).get("used") and page.get("used") is not None:
            self.placed = {"page": np.asarray(page["used"], float), "overhead": setup}
        if self.stack["page"]["source"] != "fixed":
            if self._curved() and not marks:   # its pose's tilt and height are a plane's over a curved skin
                self.event(Event.KIND_PAGE, "retained page: a curved page kept without the overhead's markers; setting "
                           "the page up again")
                return False
            current = self.node.wait_camera_page(self.arm, timeout=self.PAGE_FIRST_WAIT_S, max_lost_s=lost)
            if current is None or geometry.stale_page(current[1], lost):
                self.clear_view(ledger)   # not page.wait_s when the wrist can place it (setup_page, _last_page)
                wait_s = float(self.stack["page"].get("wait_s", self.PAGE_FIRST_WAIT_S))
                current = self.node.wait_camera_page(self.arm, timeout=self.PAGE_FIRST_WAIT_S if self._last_page(False)
                                                     else wait_s, max_lost_s=lost, should_stop=self.stopping)
            if current is None or geometry.stale_page(current[1], lost):
                self.event(Event.KIND_PAGE, "retained page: no fresh camera pose; setting the page up again")
                return False
            moved = self.sheet_moved(marks, 0.0) if marks else None
            if marks and (moved is None or _past(moved, self.RETAIN_M, self.RETAIN_RAD) or abs(moved[3]) > self.RETAIN_RISE_M):
                self.event(Event.KIND_PAGE, "retained page: the overhead does not read its markers where it was set up"
                           + ("" if moved is None else f" ({moved[0] * 1e3:.1f} mm, {np.degrees(abs(moved[1])):.1f} "
                              f"deg, {moved[3] * 1e3:+.1f} mm along the page's normal from them)")
                           + "; setting the page up again")
                return False
            if not marks and geometry.page_moved(setup, current[0], self.RETAIN_M, self.RETAIN_RAD):   # a flat page
                # kept before its markers were: the overhead's pose at rest, as before
                shift = float(np.linalg.norm(current[0][:3, 3] - setup[:3, 3])) * 1e3
                turn = float(np.degrees(geometry.rotation_angle(setup, current[0])))
                self.event(Event.KIND_PAGE, f"retained page: the overhead sees the sheet {shift:.1f} mm, {turn:.1f} deg "
                           "from where it was set up; setting the page up again")
                return False
        slid = np.asarray(page.get("slid") or np.eye(4), float) if marks else np.eye(4)   # else the watch follows it again
        self.overhead, self.overhead_at, self.slid = np.linalg.inv(slid) @ setup, self.clock.now(), slid
        self.marked_at = self.overhead_at
        self.rigid_total = (np.eye(3), np.eye(3))
        self.page_record = copy.deepcopy(page)
        self._pen_height()
        if marks:
            self.marks = marks
        else:   # a page kept before its markers were: the overhead's newest readings stand for the sheet it saw unmoved
            self._mark_reference(self.overhead_at)
        self.camera, self.correction = np.asarray(page["camera_at_setup"]), np.asarray(page["correction"])
        if self._curved() and (page.get("surface") or {}).get("field"):
            from tatbot_motion.surface import HeightField

            self.surface_field, self._surface_chart = HeightField.from_dict(page["surface"]["field"]), None
        self.prior_correction = self.correction.copy()
        self._page_fresh()
        self.event(Event.KIND_PAGE, "retained the run's measured page: no new touches")
        self.go_ready(ledger)
        return True

    def _regauge(self, program: dict, ledger) -> bool:
        """A resumed run keeps its page's place on the print and, with the wrist gauge in use, measures the paper's
        height again at the same points (no contact, ~70 s): the arm's tip height drifted ~2 mm over an hour on
        2026-10-03, more than the ballpoint's ~0.8 mm band of solid lines, and a cartridge swap takes minutes. The
        retained height stays when the gauge's points make no plane. True when the height was measured again."""
        if not self._gauge_ready():
            return False
        record, old = self.page_record, self.used()
        retained = {key: record.get(key) for key in ("touches", "gauge", "height")}
        record["touches"], record["gauge"] = [], []
        self._patches, self._hold_frame, self._hold_low, self._provisional, self._start, self._marks_seen = (
            {}, None, 0.0, None, None, [])
        self._over_skin = []   # the locate's depth is the skin as it lay then
        prior = self.camera @ self.prior_correction   # the retained page: its place on the print, the holds' seed
        contacts = self.page_touches(self._touch_layout(program), ledger, known=True)
        cfg = self.stack["page"].get("height") or {}
        plane = self._gauge_plane(contacts, self.camera[:3, 2], cfg) if len(contacts) >= 3 else None
        if plane is None:
            record.update(retained)
            self.event(Event.KIND_PAGE, "resumed page: the wrist gauge made no plane; the retained height kept")
            return False
        # the place stays the retained page's: on 2026-10-16 a resume's holds read the faint print 5.9 mm off the place
        # the first set-up's holds had already moved the page to, moved it again, and the drawing landed 10 mm left
        touched = self._hold_page(geometry.touched_page(self.camera, *plane), prior, keep_prior=True, shift=False)
        self.correction = self.prior_correction = np.linalg.inv(self.camera) @ touched
        record.update(plane=[*plane[0].tolist(), plane[1]], touched=touched.tolist(), used=self.used().tolist(),
                      correction=self.correction.tolist(), retained_height=retained["height"])
        moved = float((self.used()[:3, 3] - old[:3, 3]) @ old[:3, 2])
        self.event(Event.KIND_PAGE, f"resumed page: the wrist gauge puts the paper {moved * 1e3:+.2f} mm from the "
                   "retained height at the page's centre")
        self._fit_surface(program, ledger)   # a curved page's shape measured again with its height
        self._settle()
        return True

    def tool_change(self, program: dict, index: int, ledger) -> None:
        """Done at once when the resource is already on the arm (or the run starts with it, or it is the same tool and
        ink: another ride height). A resource activated by a rinse rinses the cartridge in its cap of water and goes
        on. Otherwise the arm lands for the operator to fit it; a resumed run whose next op is this sent change takes
        it as made (a sent rinse rinses again)."""
        if self.resource is None:
            raise RuntimeError("no resource installed on the arm")
        op = program["ops"][index]
        resource = next(r for r in program["resources"] if r["id"] == op["resource_id"])
        same = (resource["tool"]["id"], resource["ink_id"]) == (self.resource["tool"]["id"], self.resource["ink_id"])
        rinse = _rinses(resource)
        if (op.get("initial") or same
                or (not rinse and ledger_mod.op_status(ledger.rows(), self.arm, op["id"]) == "sent")):
            if resource["id"] != self.resource["id"]:
                self.machine_off()
                self.prepare_resource(resource)
            ledger.append("done", self.arm, op=op["id"], index=index, resource=resource["id"])
            self.event(Event.KIND_OP_DONE, f"resource {resource['id']}: {resource['ink_id']}", op["id"])
            return
        self.stop_preplan()
        self.machine_off()
        ledger.append("sent", self.arm, op=op["id"], index=index, resource=resource["id"])
        if rinse:
            self.rinse(resource, op["id"], index, ledger)
            self.prepare_resource(resource)
            ledger.append("done", self.arm, op=op["id"], index=index, resource=resource["id"])
            self.event(Event.KIND_OP_DONE, f"resource {resource['id']}: {resource['ink_id']}, rinsed", op["id"])
            return
        run_name = self.run_dir.name if self.run_dir else ""
        text = (f"fit {resource['ink_id']} ({resource['tool']['id']}, resource {resource['id']}) on the {self.arm} arm, "
                f"then resume: tatbot ros draw <program> --arm {self.arm} --resume {run_name}")
        self.event(Event.KIND_PAUSE, text, op["id"])
        self.land(text)

    def rinse(self, resource: dict, op_id: str, index: int, ledger) -> None:
        """The cartridge's last ink rinsed out before `resource` takes its own: one dip into the activation's cap of
        water with its rinse dwell, the machine running; an interrupted one withdraws and rinses again."""
        from tatbot_session import cap

        act = resource["activation"]
        water = {**resource, "slot": act["slot"], "ink_id": act["ink_id"],
                 "dip": {**resource["dip"], "dwell_s": act["dwell_s"], "above_ink_m": act["above_ink_m"]}}
        self.event(Event.KIND_PAGE, f"rinsing {self.resource['ink_id']} out in {act['slot']} ({act['ink_id']}, "
                   f"{act['dwell_s']:g} s) before {resource['ink_id']}", op_id)
        while not cap.dip(self, water, op_id, index, ledger):
            pass

    def dip(self, op_id: str, index: int, ledger) -> None:
        """A palette dip for the resource on the arm; an interrupted one withdraws and dips again (continue) or
        lands. It charges the cartridge afresh and answers a numpad dip asked before it began."""
        from tatbot_session import cap

        if self.resource is None:
            raise RuntimeError("no resource installed on the arm")
        self.dipping = True
        try:
            while not cap.dip(self, self.resource, op_id, index, ledger):
                pass
            with self._lock:
                asked, self.dip_asked = self.dip_asked, False
        finally:
            self.dipping = False
        self.charged.add(self.resource["id"])
        self.inked_m = 0.0
        if asked:
            self.event(Event.KIND_DIP, "the numpad's dip made", op_id)

    def _budget(self) -> float | None:
        """The line a dip draws now, while the operator's dip scale governs the cartridge on the arm: once + or - was
        pressed for it in the run. None before (its dips are the program's), and for a cartridge that takes none."""
        res = self.resource
        return None if res is None or res["id"] not in self.dip_scales else charge.budget_m(res, self.dip_scale())

    def _dip_first(self, ops: list[dict], index: int, from_arc: float = 0.0) -> str | None:
        """Why the stroke at `index` dips before it goes on from from_arc, or None (charge.py)."""
        res = self.resource
        if res.get("dip") is None:
            return None
        if res["id"] not in self.charged:
            return "no dip yet in this goal: dipping before the stroke"
        if self.dip_asked:
            return "the numpad's dip"
        budget, need = self._budget(), charge.need_m(ops, index, from_arc)
        if budget is None or not charge.dip_first(self.inked_m, need, budget):
            return None
        return (f"{self.inked_m * 1e3:.0f} of the {budget * 1e3:.0f} mm a dip draws are drawn and the next "
                f"{need * 1e3:.0f} mm do not fit: dipping first")

    def _dip_before(self, op: dict, index: int, ledger, why: str) -> None:
        """A dip the program does not hold, before a stroke: the pen off the page first, as a stroke's own end
        lifts it before a dip op."""
        self.event(Event.KIND_OP_SENT, why, op["id"])
        self.machine_off()
        self.lift_off(f"{op['id']} lift before the dip", ledger)
        self.dip(op["id"], index, ledger)

    def dip_op(self, ops: list[dict], index: int, ledger) -> None:
        """A program's dip op: dipped, or skipped when the charge covers it (_covered)."""
        op = ops[index]
        skip = self._covered(ops, index)
        if skip:
            ledger.append("skipped", self.arm, op=op["id"], index=index, reason=skip)
            self.event(Event.KIND_DIP, f"skipped: {skip}", op["id"])
            return
        ledger.append("sent", self.arm, op=op["id"], index=index)
        self.dip(op["id"], index, ledger)
        ledger.append("done", self.arm, op=op["id"], index=index)

    def _covered(self, ops: list[dict], index: int) -> str | None:
        """Why the compiled dip at `index` is skipped, or None: a replenishing one, in a goal that has dipped, whose
        line the charge left still covers (a numpad dip since, or a longer scale)."""
        res, nxt, budget = self.resource, index + 1, self._budget()
        if ops[index].get("reason") != "replenish" or budget is None or res["id"] not in self.charged or self.dip_asked:
            return None
        need = charge.need_m(ops, nxt) if nxt < len(ops) and ops[nxt]["op"] == "stroke" else 0.0
        if not charge.covered(self.inked_m, need, budget):
            return None
        return (f"{self.inked_m * 1e3:.0f} of the {budget * 1e3:.0f} mm a dip draws are drawn and the next "
                f"{need * 1e3:.0f} mm fit")

    def _run_ops(self, program: dict, start: int, ledger) -> None:
        ops = program["ops"]
        rows = ledger.rows()
        index = ledger_mod.next_index(ops, rows, self.arm, start)
        while index is not None:
            if self.resource is None:
                raise RuntimeError("no resource installed on the arm")
            res = self.resource
            op = ops[index]
            self.op_now = op["id"]
            self.feedback(phase=P.PHASE_PLAN, op_id=op["id"], index=index)
            if op["op"] == "tool_change":
                self.tool_change(program, index, ledger)
            elif op["resource_id"] != res["id"]:
                raise RuntimeError(f"op {op['id']} draws with {op['resource_id']}, but {res['id']} is on the "
                                   "arm: its tool change was skipped (--from-op)")
            elif op["op"] == "dip":
                self.dip_op(ops, index, ledger)
            elif ledger_mod.op_status(rows, self.arm, op["id"]) == "sent":
                self.event(Event.KIND_UNCERTAIN, "left `sent` by a crash: decide redraw or skip", op["id"])
                decision = self.wait_decision("uncertain", op["id"], index, ledger)
                if decision == rules.LAND:
                    self.land()
                if decision == rules.SKIP:
                    ledger.append("skipped", self.arm, op=op["id"], index=index)
                else:
                    ledger.append("aborted", self.arm, op=op["id"], index=index, arc_m=0.0, reason="redraw")
            else:
                why = self._dip_first(ops, index, ledger_mod.resume_arc(rows, self.arm, op["id"]))
                if why:
                    self._dip_before(op, index, ledger, why)
                self.stroke(program, index, ledger)
            rows = ledger.rows()
            index = ledger_mod.next_index(ops, rows, self.arm, start)

    def lift_at_end(self, program: dict, op: dict, rows, drawn_m: float | None = None) -> bool:
        """Stay down only for the immediate, unstarted continuation; never across a pause or skip, nor a dip the
        charge makes before it once op has drawn drawn_m more (all of it by default; _dips_before)."""
        ops = program["ops"]
        index = next(i for i, item in enumerate(ops) if item["id"] == op["id"])
        if index + 1 == len(ops):
            return True
        following = ops[index + 1]
        return not (following["op"] == "stroke" and following.get("continues")
                    and ledger_mod.op_status(rows, self.arm, following["id"]) is None) or self._dips_before(
            ops, index + 1, charge.contact_m(op) if drawn_m is None else drawn_m)

    # Whether the charge dips before the stroke at `nxt` once drawn_m more is drawn (charge.dip_first, _dip_first's).
    # A chain the operator's dip scale left longer than a dip's line dipped between its sections from the first's end
    # pen-down: on 2026-10-20's tattoo its pre-plan held the dip 30 s and ended the run, and the next one's 20 mm lift off
    # the skin could not be planned and ended it with the needle resting in the skin. Such a stroke lifts at its end as
    # any stroke does, and the next is not pre-planned (the dip moves the arm away from where it would start).
    def _dips_before(self, ops: list[dict], nxt: int, drawn_m: float) -> bool:
        res, budget = self.resource, self._budget()
        if res is None or res.get("dip") is None or budget is None or nxt >= len(ops) or ops[nxt].get("op") != "stroke":
            return False
        return charge.dip_first(self.inked_m + drawn_m, charge.need_m(ops, nxt), budget)

    def plan(self, program: dict, op: dict, from_arc: float, pen_down: bool, seed: np.ndarray,
             *, lift_at_end: bool = True, snap=None, pen=None):
        from tatbot_motion import plan_op

        if snap is not None:   # stencil tracking: the points moved by the print's residual (page_filter.PageFilter)
            op = snap.corrected(op)[0]
        return self.knots(plan_op(op, base_from_page=self.used(), q_seed=seed, kin=self.kin, motion=self.motion,
                                  speed_m_s=float(program.get("draw_speed_m_s", 0.0035)), from_arc_m=from_arc,
                                  pen_down_at_start=pen_down, lift_at_end=lift_at_end,
                                  pen=self._ride_pen() if pen is None else pen, **self._on_surface(op)))

    def preplan(self, program: dict, index: int, seed: np.ndarray, rows) -> None:
        """Plan the next stroke while the current one runs, seeded at the current plan's end."""
        ops = program["ops"]
        nxt = ledger_mod.next_index(ops, rows, self.arm, index + 1)
        drawn = max(0.0, charge.contact_m(ops[index]) - self._arc_now)   # the running stroke's line still to draw
        if (nxt is None or ops[nxt]["op"] != "stroke" or ledger_mod.op_status(rows, self.arm, ops[nxt]["id"])
                or self._dips_before(ops, nxt, drawn)):
            self._preplan = None
            return
        op = ops[nxt]
        lift = self.lift_at_end(program, op, rows, drawn_m=drawn + charge.contact_m(op))
        snap = self._snap(op)
        hover = self._hover_on() and not op.get("continues")
        pen = self._ride_pen()   # at the running plan's own ride, where its end is: dispatch_plan folds in the next's
        if pen is not None and self._fold:
            pen = dataclasses.replace(pen, height_m=pen.height_m - self._fold)
        self._preplan_ridden = self._ridden_for(op)   # a continuation riding its chain's skin is pre-planned on it
        self._preplan = (op["id"], self._planner.submit(self._preplanned, program, op, seed, lift, snap, hover, pen), lift,
                         snap, pen)

    def _preplanned(self, program: dict, op: dict, seed: np.ndarray, lift: bool, snap, hover: bool, pen=None):
        """The pre-plan's work: (stencil tracking's hover travel or None, the stroke from its end or from seed)."""
        travel = self.plan_hover(op, seed, snap) if hover else None
        start = seed if travel is None else np.asarray(travel.q[-1])
        return travel, self.plan(program, op, 0.0, bool(op.get("continues")), start, lift_at_end=lift, snap=snap, pen=pen)

    def _snap(self, op: dict):
        """The correction field an op is planned with: its chain's for a pen-down continuation, else the newest;
        None unless stencil tracking corrects."""
        if self.tracker is None or not self.tracker.correcting:
            return None
        return self._chain_snap if op.get("continues") and self._chain_snap is not None else self.tracker.snapshot()

    def _same_field(self, op: dict, planned, fresh) -> bool:
        """Whether a pre-plan's field still holds: its correction of this op within track.replan_m of the newest."""
        if planned is fresh:
            return True
        if planned is None or fresh is None:
            return False
        moved = float(np.max(np.linalg.norm(planned.corrected(op)[1] - fresh.corrected(op)[1], axis=1)))
        if moved <= float((self.stack["page"].get("track") or {}).get("replan_m", 0.001)):
            return True
        self.event(Event.KIND_REPLAN, f"the print's correction moved {moved * 1e3:.2f} mm since the pre-plan", op["id"])
        return False

    def _use_snap(self, op: dict, snap) -> None:
        self._chain_snap = snap
        if snap is None:
            self._correction = None
            return
        corr = snap.corrected(op)[1]
        mean, largest = corr.mean(axis=0), float(np.linalg.norm(corr, axis=1).max())
        self._correction = (mean, largest)
        self.event(Event.KIND_PAGE, f"track: {op['id']} corrected {mean[0] * 1e3:+.2f} {mean[1] * 1e3:+.2f} mm "
                   f"(largest {largest * 1e3:.2f} mm, {len(snap.samples)} samples)", op["id"])

    def dispatch_plan(self, program: dict, op: dict, from_arc: float, rows=()):
        """(plan, sent): the plan to send now and it with the pen trim. The pre-plan if it matches and, trimmed from
        the trim the last goal ended at (and sent by its ride's change: _ride_fold), has not drifted from the measured
        pose (motion.yaml dispatch_drift), else a fresh plan from the measured joints, trimmed from 0. plan_op joins
        the path pen-down when the tip already sits on it (a `continues` segment, a resumed stroke) and lifts and
        re-approaches otherwise."""
        from tatbot_motion import dispatch_drift

        pen_down = bool(op.get("continues")) or from_arc > 0
        lift = self.lift_at_end(program, op, rows, drawn_m=charge.contact_m(op) - from_arc)
        snap = self._snap(op)
        pre, self._preplan, self._fold = self._preplan, None, 0.0
        if (pre is not None and pre[0] == op["id"] and pre[2] == lift and from_arc <= 0
                and (fold := self._ride_fold(pre[4])) is not None and self._same_field(op, pre[3], snap)
                and self._preplan_ridden is self._ridden_for(op)):   # its skin's profile, read at its hover
            try:
                traj = pre[1].result()[1]
            except Exception as exc:  # noqa: BLE001 - a failed pre-plan is planned again below
                self.event(Event.KIND_REPLAN, f"pre-plan failed: {exc}", op["id"])
                traj = None
            if traj is not None:
                self._fold = fold
                sent = self.trimmed(traj, self.trim_end)
                drift = dispatch_drift(sent, self.io.measured(), self.kin, self.motion)
                if drift["ok"] and self._folded(traj, sent, op["id"]):
                    self._use_snap(op, pre[3])
                    return traj, sent
                if not drift["ok"]:
                    self.event(Event.KIND_REPLAN, "drift at dispatch: {:.1f} mrad, {:.2f} mm".format(
                        drift["joint_rad"] * 1000, drift["tip_m"] * 1000), op["id"])
        self._fold = 0.0
        traj = self.plan(program, op, from_arc, pen_down, self.io.seed(), lift_at_end=lift, snap=snap)
        self._use_snap(op, snap)
        return traj, self.trimmed(traj, 0.0)

    # A stroke's ride is set at its own hover, after its pre-plan was made at the ride before (preplan), and planning
    # it again at dispatch held the needles over the skin with the machine on 2-6 s (median 3.5) at 8 drift re-plans
    # on 2026-10-20; rides differed on 67 of that tattoo's 112 stroke pairs. The pre-plan is sent by the difference
    # instead, eased in along each knot's depth axis (the skin's normal, along which the ride lifts) as the trim is:
    # within the trim's limit, and eased in before the plan's tip comes FOLD_CLEAR_M more than any lift over its first
    # pen-down point (a riding tube rides past its needles' reach, and a gauge reads skin ~1 mm low), else planned again.
    FOLD_CLEAR_M = 0.003

    def _ride_fold(self, planned) -> float | None:
        """How much higher (+) than `planned` (a pre-plan's pen) the stroke's own ride sends it; None when the pens
        differ otherwise, or with the trim it would pass the trim's limit."""
        ride = self._ride_pen()
        if planned == ride:
            return 0.0
        if planned is None or ride is None or dataclasses.replace(planned, height_m=ride.height_m) != ride:
            return None
        fold = ride.height_m - planned.height_m
        return fold if abs(self.trim_in_use() + fold) <= float(self.motion["pen"]["trim"]["limit_m"]) + 1e-9 else None

    def _folded(self, traj, sent, op_id: str) -> bool:
        """Whether the ride's fold (_ride_fold) in `sent` has eased in before the needles may reach the skin; said."""
        if not self._fold:
            return True
        down = np.flatnonzero(np.asarray(traj.phase) == P.PHASE_DRAW)
        k = int(down[0]) if len(down) else len(traj.t) - 1
        over = (np.asarray(traj.tip[: k + 1]) - traj.tip[k]) @ np.asarray(traj.axis[k])
        near = float(traj.t[int(np.argmax(over <= max(self._fold, 0.0) + self.FOLD_CLEAR_M))])
        ok = (settled := sent.info["trim"].settled_s()) <= near
        self.event(Event.KIND_PAGE if ok else Event.KIND_REPLAN, f"its pre-plan sent {self._fold * 1e3:+.2f} mm by the "
                   f"ride, eased in by {settled:.1f} s of the {near:.1f} s before its needles near the skin"
                   + ("" if ok else ": planned again"), op_id)
        return ok

    # --- stencil tracking's hover stop (ros/README.md 4.4) -------------------------------------------
    def _hover_on(self) -> bool:
        return (self.tracker is not None and (self.tracker.correcting or getattr(self.tracker, "rigid", None) is not None
                                              or self._skin_only)
                and bool((self.stack["page"].get("track") or {}).get("hover_stop")))

    def _rigid(self) -> bool:
        """A curved page's stencil watch runs (track.Tracker.rigid): its follows move the whole surface."""
        return self.tracker is not None and getattr(self.tracker, "rigid", None) is not None

    def plan_hover(self, op: dict, seed: np.ndarray, snap, from_arc: float = 0.0):
        from tatbot_motion import plan_hover

        if snap is not None:
            op = snap.corrected(op)[0]
        return self.knots(plan_hover(op, base_from_page=self.used(), q_seed=seed, kin=self.kin, motion=self.motion,
                                     from_arc_m=from_arc, pen=self.pen, **self._on_surface()))

    def _hover_stop(self, program: dict, op: dict, from_arc: float, ledger, index: int = 0) -> None:
        """Stop over a stroke's start before it descends and measure the pen tip on the print there, so the stroke
        is planned with the field at its own start rather than one stroke behind. A jump is measured again from a
        second view track.second_view_m toward the page centre: one view cannot tell a lattice misfit from a moved
        sheet. A hover that finds no print while the overhead sees the sheet moved past the wrist fit's reach sets the
        page up again first, unless the overhead freshly sees the sheet within that reach or a frame whose gauge read
        no surface finds the print there by its own depth (_print_stays): no stroke goes down blind. A print the
        wrist camera has read nowhere in this goal is one its fit cannot read (the seed-102 W transfer on silicone,
        2026-10-05), not a sheet that left: hover stops end (an inking pen's read the skin alone, _skin_alone), the
        strokes go on the measured page, and the overhead watches the sheet between strokes (_watch_sheet). A hover whose reading the field did not take lets the
        overhead's slide move the page (_follow_unread). Not for a stroke that joins its path where the tip stands
        (_joins): every descent is measured, a continuation's after a latch, a trip, a pause or a dip too. On a curved
        page the watch's readings move the whole surface with the stencil (_rigid_follow), and the skin every curved
        hover sees is read against it (_skin_watch, page.track's correction on or off: replayed over 790 hover frames
        of nine still skins it never read 2 mm off the field; a forearm that sank 4-5 mm read so at its first hover,
        2026-10-18). Over a fiducial-tile print a hover that sees no whole marker looks again from the second view
        before the sheet is taken to have left: with the arm over the sheet the overhead reads it lost, and one view of
        the curved holder (2026-10-18) took no marker in, the next 5 mm on read three."""
        from tatbot_motion import timelaw

        if not self._hover_on() or self._joins(op, from_arc):
            return
        self._chain_snap = None   # a chain left goes down again planned with the newest field, this hover's reading in it
        start = timelaw.trim_polyline(np.asarray(op["points_m"], float), from_arc)[0]
        verdict = self._measure_over(op, f"{op['id']}-hover", start, self._pre_travel(op), from_arc, ledger)
        if self._skin_only or (verdict is None and not self.tracker.field.samples):
            return self._skin_alone(program, op, index, start, from_arc, ledger)
        if verdict is None:
            verdict, on = self._unread_hover(program, op, start, from_arc, ledger)
            if not on:
                return
        else:
            self._unread = 0
        if verdict == "held":
            view, at = self._second_view(op, start)
            verdict = self._measure_over(view, f"{op['id']}-hover2", at, None, from_arc, ledger)
        if self._rigid() and self.tracker.rigid.confirmed:
            return self._rigid_follow(program, op, start, from_arc, ledger, index)
        if self._curved() and self._skin_watch(program, op, index, ledger):
            return self._rehover(program, op, index, start, from_arc, ledger)
        if verdict not in ("added", "moved"):
            self._follow_unread(program, op, ledger)

    # A print the wrist camera has read nowhere in this goal (_hover_stop): an inking pen's hovers go on for the skin
    # alone, its only depth reading (_ride_over), no print read nor set-up for it; other hover stops end.
    def _skin_alone(self, program: dict, op: dict, index: int, start, from_arc: float, ledger) -> None:
        if not self._skin_only:
            self.tracker.correcting, self.tracker.rigid, self._skin_only = False, None, self._rides_measured()
            self.event(Event.KIND_PAGE, "track: the wrist camera has read the print nowhere in this goal: "
                       + ("the hovers read the skin alone" if self._skin_only else "no more hover stops")
                       + ", the strokes go on the measured page while the overhead sees the sheet stay", op["id"])
        if self._skin_only and self._skin_watch(program, op, index, ledger):
            self._rehover(program, op, index, start, from_arc, ledger)

    # Whether the stroke goes on pen-down where the tip stands (tatbot_motion.joins, plan_op's own test, on the page,
    # surface, correction and pen it is planned with): a continuation straight on from its chain, or a resume at its arc;
    # never a tip lifted after a latch, a trip or a pause, back from a dip, or one the page moved away from.
    def _joins(self, op: dict, from_arc: float) -> bool:
        from tatbot_motion import joins

        if not (op.get("continues") or from_arc > 0) or self.camera is None:
            return False
        snap = self._snap(op)
        return joins(op if snap is None else snap.corrected(op)[0], self.tcp(self.io.seed())[:3, 3], base_from_page=self.used(),
                     motion=self.motion, from_arc_m=from_arc, pen=self._ride_pen(),
                     surface=self._on_surface(op).get("surface"))

    def _unread_hover(self, program: dict, op: dict, start, from_arc: float, ledger) -> tuple[str | None, bool]:
        """A hover that read no print: the second view first (a view between the print's markers, 2026-10-18); then,
        unless a reading whose gauge read no surface finds the print in place (_print_stays), the page is set up again
        if the sheet may have left, and the hover measured anew. With the markers kept but none read since the pen left
        (the arm hides them) the arm clears the overhead's view first, and comes back to measure if the sheet stays.
        (the verdict, whether hover stops are still on)."""
        view, at = self._second_view(op, start)
        verdict = self._measure_over(view, f"{op['id']}-hover2", at, None, from_arc, ledger)
        if verdict is not None or self._print_stays(op):
            self._unread = 0
            return verdict, True
        self._unread += 1
        if self.marks == {} and self._unread < self.UNREAD_SETUP:   # a page the wrist placed, the overhead blind to it
            self.event(Event.KIND_PAGE, f"track: no print under the hover ({self._unread} in a row), and the overhead "
                       "reads none: the stroke goes on the page in use, its skin read; the page is set up again after "
                       f"{self.UNREAD_SETUP} in a row", op["id"])
            return None, True
        cleared = self.marks is not None and self._corrects() and not self._timed_moves(max(self.inked_at, self.overhead_at))
        if cleared:   # the arm hides the markers: clear the overhead's view, wait up to MARKS_S for three fresh readings
            self.machine_off()
            self.clear_view(ledger)
            at = self.clock.now()
            while len(self._timed_moves(at)) < 3 and self.clock.now() < at + self.MARKS_S and not self.stopping():
                self.clock.sleep(1.0)
        if self._sheet_may_have_left(program, ledger):
            self._reacquire(program, ledger, "no print under the hover, and the overhead cannot place the sheet "
                                             "where the page is")
            if not self._hover_on():
                return None, False
        elif not cleared:
            return None, True
        else:
            self.go_ready(ledger)
        return self._measure_over(op, f"{op['id']}-rehover", start, None, from_arc, ledger), True

    MARKS_S = 10   # the arm cleared the overhead's view: its fresh readings of the print's markers waited for this long
    # On a page the wrist placed whose print the overhead does not read (marks {}: an inked forearm, 2026-10-20) nothing
    # can show the sheet stayed or left, and each unread hover cleared the view, waited MARKS_S for readings that could not
    # come and set the page up again (~4 min): this many in a row do so, the ones before go on with their skin read.
    UNREAD_SETUP = 3

    def _print_stays(self, op: dict) -> bool:
        """Whether a hover reading whose gauge read no surface (no pen or no surface by the tip) still finds the print
        by its frame's own depth within track.stays_m of where the goal's gauge holds found it, at its markers
        (Tracker.print_moved; a turn shows there as a move): the gauge missed, the sheet did not leave. Hovering 6 mm
        over a forearm print (2026-10-19) the gauge read no surface in 43 of 61 unread readings, and ten hovers set the
        page up again for a sheet that never moved; replayed, 40 of the 43 read the print within 2.9 mm of the holds."""
        reach = float(self.stack["page"]["track"].get("stays_m", 0.010))
        near = [m for m in self.tracker.print_moved(op["id"]) if m <= reach]
        if near:
            self.event(Event.KIND_PAGE, f"track: the gauge read no surface at the hover, but the print reads "
                       f"{near[-1] * 1e3:.1f} mm from where the gauge holds read it: within the wrist fit's reach, "
                       "the page stays", op["id"])
        return bool(near)

    def _second_view(self, op: dict, start) -> tuple[dict, np.ndarray]:
        """(op moved track.second_view_m toward the page centre, its start there): a hover's second view."""
        norm = float(np.linalg.norm(start))
        away = -start / norm if norm > 1e-6 else np.array([1.0, 0.0])
        shift = away * float(self.stack["page"]["track"].get("second_view_m", 0.005))
        return {**op, "points_m": (np.asarray(op["points_m"], float) + shift).tolist()}, start + shift

    def _follow_unread(self, program: dict, op: dict, ledger) -> None:
        """A hover whose reading the filter did not take (refused by its margin or the overhead's gate, no print
        found, or a jump no second view confirmed): the overhead's slide moves the page as it does without hover stops
        (_follow_overhead), and the stroke ends read before it are cleared (track.Tracker.restart); the filter carries
        the page's follows in its model. On 2026-10-08 an untaped half-skin slid ~14 mm and 6 deg
        under five 1011RL strokes. The overhead measured it throughout, the wrist read the inked print 4-9 mm off it,
        the gate and the jump rule refused or held every reading, and with the field still at its 2 mm the strokes
        went down 4.5-9.8 mm off the design. A turn alone follows only past track.reach_rad: on still sheets the
        overhead's yaw flickers in 1-2 deg steps, and replayed over 32 flat tracked runs the eight follows it caused
        moved the page 0.2-1.4 mm for nothing and dropped the field."""
        if self.marks is not None:   # the markers move the page before each stroke (_follow_marks), and the filter
            return                   # takes their moves itself (track.Tracker._feed)
        # a curved page's overhead turns only past its own watch's limit (curved.WATCH): its raised match's yaw scatters
        reach = None if self._curved() else float(self.stack["page"]["track"].get("reach_rad", 0.05))
        if self._follow_overhead(program, op, ledger, "; the wrist's reading at the hover was not taken",
                                 turn_rad=reach) == "followed":
            self.tracker.restart()

    def _rigid_follow(self, program: dict, op: dict, start, from_arc: float, ledger, index: int) -> None:
        """A curved page whose stencil the wrist camera's readings agree has moved (curved.RigidWatch): the whole page
        surface, its pose, height field and chart, moves with it rigidly (_follow_rigid), and the hover is measured
        again over the stroke's start on it. Never a stroke's own points: on 2026-10-08 the per-stroke field carried a
        20 mm stroke 5 mm past the gauge holds. The page is set up again instead (fresh holds, a fresh field) when the
        strokes would move past page.surface.rigid.max_m or max_rad from where the holds measured them, or when the
        move is one the print cannot tell apart (curved.ambiguity): a forearm turned about its own axis moves the print
        round the curve over a skin whose shape stays, a slid one carries the shape with it, and on a strongly curved
        arm the two put the needle a millimetre apart on the flanks."""
        cfg = curved.config(self.stack)["rigid"]
        step_plane, step_print = self.tracker.rigid.step()
        total = (self.rigid_total[0] @ step_plane, self.rigid_total[1] @ step_print)
        surface, uv = self.surface(), self._stroke_uv(program)
        held = surface.moved(surface.base @ np.linalg.inv(curved.in_plane(self.rigid_total[0])))   # at the holds
        xy = held.chart.to_plane(uv)
        shift = float(np.max(np.linalg.norm(curved.moved_by(total[0], xy) - xy, axis=1)))
        turn = abs(float(np.arctan2(total[0][1, 0], total[0][0, 0])))
        apart, tilt = curved.ambiguity(held, *total, uv)
        d = step_plane[:2, 2]
        said = (f"the wrist reads the stencil moved {d[0] * 1e3:+.1f} {d[1] * 1e3:+.1f} mm, "
                f"{np.degrees(np.arctan2(step_plane[1, 0], step_plane[0, 0])):+.1f} deg")
        why = None
        if shift > float(cfg["max_m"]) or turn > float(cfg["max_rad"]):
            why = (f"{said}, the strokes {shift * 1e3:.1f} mm and {np.degrees(turn):.1f} deg from where the holds measured "
                   f"them (page.surface.rigid.max_m {float(cfg['max_m']) * 1e3:.0f}, max_rad "
                   f"{np.degrees(float(cfg['max_rad'])):.0f} deg)")
        elif apart > float(cfg["ambiguity_m"]) or tilt > float(cfg["ambiguity_rad"]):
            why = (f"{said}, across the curve: a slid skin and a print turned round the arm read alike and would put the "
                   f"strokes {apart * 1e3:.1f} mm and {np.degrees(tilt):.1f} deg apart, so the holds measure which")
        else:
            self._follow_rigid(step_plane, total, ledger, op["id"], said)
            verdict = self._measure_over(op, f"{op['id']}-followed", start, None, from_arc, ledger)
            # the stencil where the followed page puts it, or no print read on it (the two readings that agreed stand):
            # the skin read as at any hover
            if verdict in ("added", None):
                if self._skin_watch(program, op, index, ledger):
                    self._rehover(program, op, index, start, from_arc, ledger)
                return
            last = self.tracker.rigid.held[-1] if self.tracker.rigid.held else None
            off = 0.0 if last is None else float(np.linalg.norm(last[1] - last[0]))
            why = f"after the page followed the stencil it still reads {off * 1e3:.1f} mm off"
        self._reacquire(program, ledger, why, op["id"])
        self._rehover(program, op, index, start, from_arc, ledger)

    def _rehover(self, program: dict, op: dict, index: int, start, from_arc: float, ledger) -> None:
        """After the page was set up again mid-draw: the hover over the stroke's start measured on the new page (the
        watch's first reading there), as a hover that found no print does, and its skin read (_skin_watch: a fresh
        page's first reading never sets it up again; the stroke rides higher by it, _ride_over)."""
        if self._hover_on():
            self._measure_over(op, f"{op['id']}-rehover", start, None, from_arc, ledger)
            if self._curved():
                self._skin_watch(program, op, index, ledger)

    def _follow_rigid(self, step_plane: np.ndarray, total, ledger, op_id: str, said: str) -> None:
        """Move the page in use by a rigid in-plane step of its own frame (the surface rides on it: PageSurface over
        used()), recorded as a page row so a resume takes the page where it went; the overhead's expected pose of the
        sheet moves with it (slid)."""
        old, step = self.used(), curved.in_plane(step_plane)
        trim = geometry.translation(list(self.stack["page"].get("trim", {}).get(self.arm, [0.0, 0.0])))
        self.correction = self.correction @ trim @ step @ np.linalg.inv(trim)
        self.slid = old @ step @ np.linalg.inv(old) @ self.slid
        self.rigid_total = total
        self.stop_preplan()
        self._chain_snap = self._correction = None
        self.tracker.rigid.clear()
        new = self.used()
        self.page_record.update(used=new.tolist(), correction=self.correction.tolist(),
                                overhead=self.page_overhead().tolist(), slid=self.slid.tolist())
        self.page_record.setdefault("followed", []).append({"t": self.clock.utc(), "by": "wrist",
                                                             "step": step.tolist(), "total": total[0].tolist()})
        if self.placed is not None:
            self.placed = {"page": new, "overhead": self.page_overhead()}
        ledger.append("page", self.arm, page=self.page_record, workspace=self.workspace())
        self.event(Event.KIND_PAGE, f"track: {said}: the page and the skin's shape on it follow it", op_id)

    def _skin_watch(self, program: dict, op: dict, index: int, ledger) -> bool:
        """The skin the hover's frame saw about the tip (track.Tracker.skin) against the surface in use: how far the
        needle would ride off its plan if the stroke went down now, the arm's own error at this pose included, net of
        what the pen already rides higher by (the operator's trim, + up, and the surface check's rise_m): a forearm
        read 2.29 and 2.13 mm over with the trim at +0.7 and +1.0 mm, and each stopped a tattoo (2026-10-20). Every
        reading is kept for the stroke's ride (_ride_over), which lifts it by the skin over the field, never deeper.
        Past page.surface.rigid.skin_m a warning; past skin_stop_m deeper (the skin over the field) the page is set up
        again at once where an inking pen would go deeper, else when the hover before read it so too; past
        skin_stop_under_m shallower when the hover before read it so too; judged on the skin's part alike at every
        trim, keys pressed its way never added (the operator's depth, which a set-up keeps), keys pressed against it
        taken out. Not at a page's first reading after its holds (fresh holds read a forearm on the field where its
        hovers read it 1-1.8 mm over, 2026-10-20): that is the page's floor, past which (FLOOR_M) it sets it up."""
        skin, surface = getattr(self.tracker, "skin", None), self.surface()
        gap = None if skin is None or surface is None else curved.skin_gap(surface, skin)
        if gap is None:
            return False
        at = (np.linalg.inv(self.used()) @ np.c_[skin, np.ones(len(skin))].T)[:2].mean(axis=1)
        self._skins.append((self.clock.now(), at, gap, op["id"]))
        cfg = curved.config(self.stack)["rigid"]
        trim, rise = self._capped(self._cap_for([gap])), float(self._rise_at([at])[0])
        part, net, lift = gap - rise, gap - rise - trim, trim + rise   # off the plan, + deeper: the skin's, with keys
        judged = max(0.0, min(net, part)) if part > 0 else min(0.0, max(net, part))   # keys never add to the skin's
        text = (f"track: the skin about the tip at {op['id']}'s hover reads {abs(gap) * 1e3:.2f} mm "
                f"{'over' if gap > 0 else 'under'} the field" + (f", {net * 1e3:+.2f} mm net of the pen's "
                                                                 f"{lift * 1e3:+.1f} mm lift" if lift else ""))
        stop = float(cfg[key := "skin_stop_m" if net > 0 else "skin_stop_under_m"])
        side = float(np.sign(judged)) if abs(judged) > stop else 0.0
        fresh, self._fresh, floor = self._fresh, False, ""
        if fresh:
            self._skin_floor = {side: abs(judged)} if side else {}
            floor = "read so on fresh holds: the page is set up again only past it" if side else ""
        elif side and abs(judged) <= self._skin_floor.get(side, 0.0) + self.FLOOR_M:
            side, floor = 0.0, "within what this page's first hover read: not set up again"
        path = None if fresh or not self._rides_measured() else self._chain(program, index)[1]
        along = None if path is None else self._profile(op["id"], path, 0.0, self._rise_at(path))
        if along is not None and (worst := float(np.nanmax(np.abs(along[0])))) > float(cfg["profile_setup_m"]):
            self._set_up_for = op["id"]
            self._reacquire(program, ledger, f"the skin along {op['id']}'s path reads up to {worst * 1e3:.2f} mm "
                            f"off the field, past page.surface.rigid.profile_setup_m "
                            f"{float(cfg['profile_setup_m']) * 1e3:.1f}: the skin moved since the holds", op["id"])
            return True
        inks = self.motion["pen"]["mode"] != "hover"
        again = not fresh and bool(side) and (side == self._skin_off or (inks and net > 0))
        self._skin_off = side
        if not again:
            ride = part > 0 and self._rides_measured()   # the ride (_ride_over) lifts by the skin's part, not the keys'
            says = [f"the stroke rides {part * 1e3:.2f} mm higher by the skin's part" if ride else
                    f"the needles would ride that much {'deeper' if net > 0 else 'shallower'}; the depth keys trim it",
                    f"{abs(net - judged) * 1e3:.1f} mm of it is the depth keys'" if abs(net - judged) > 1e-6 else "",
                    floor or (f"past {key} the next hover's too sets the page up again" if side else f"past {key} only "
                              "with the depth keys' share, which a set-up keeps" if abs(net) > stop else "")]
            warn = "" if abs(net) <= float(cfg["skin_m"]) else (
                f" (past page.surface.rigid.skin_m {float(cfg['skin_m']) * 1e3:.1f}: {'; '.join(filter(None, says))})")
            self.event(Event.KIND_PAGE, text + warn, op["id"])
            return False
        self._reacquire(program, ledger, f"{text}, past page.surface.rigid.{key} {stop * 1e3:.1f}: the skin "
                        "rose or sank since the holds", op["id"])
        return True

    FLOOR_M = 0.001   # a hover sets the page up again only this far past what the page's first hover read past its stop

    def _page_fresh(self) -> None:
        """A page set up (or a retained one measured again): its hovers' readings start afresh (_skin_watch, _ride_over)."""
        self._skins, self._ride, self._skin_floor, self._fresh, self._skin_only, self._unread = (
            [], ("", 0.0), {}, True, False, 0)
        self._ridden = None
        if self.tracker is not None:   # skin its stroke's views read before: against the field gone
            self.tracker.wide = []

    def _rides_measured(self) -> bool:
        """An inking pen on a curved page whose hover stops read the skin (page.track correct with hover_stop, the real
        wrist): its strokes ride by the skin the hovers read (_ride_over)."""
        return (self._curved() and self.pen is not None and self.motion["pen"]["mode"] != "hover" and self._corrects()
                and bool((self.stack["page"].get("track") or {}).get("hover_stop")))

    def _ride_pen(self):
        """The pen the strokes are planned with: the page's, higher by the ride's lift (_ride_over)."""
        if self.pen is None or not self._ride[1]:
            return self.pen
        return dataclasses.replace(self.pen, height_m=self.pen.height_m + self._ride[1])

    # A hover reads the skin 8-20 mm about the stroke's start (track._skin) and a stroke runs up to ~115 mm across a
    # forearm whose hover readings run from 1.7 mm under the field on its flanks to 1.9 over on its crown (2026-10-20);
    # page.surface.rigid's stop is 2.5 mm net, and under it nothing fed the reading into the plan: each stroke went in by
    # what its hover read over, 1.0-2.3 mm past its plan on 17 of the tattoo's 34. A stroke (with the continuations it
    # joins pen-down) now rides higher by the most the skin read over the field less the surface check's rise it rides
    # by already: its own hover's readings, and past RIDE_AROUND_M from its start (beyond its ring) those of the page's
    # hovers in RIDE_KEEP_S within RIDE_NEAR_M of its path. Never lower than planned, the operator's trim on top, never
    # lowered within a stroke. A hover that read no skin takes those within RIDE_AROUND_M of its path; none: skin_stop_m.
    # Its trim cap (_cap_for) counts its own hover's readings and those past its ring, never its neighbours' alone.
    RIDE_NEAR_M, RIDE_AROUND_M, RIDE_KEEP_S = 0.010, 0.020, 300.0

    def _ride_over(self, program: dict, index: int, from_arc: float) -> None:
        op = program["ops"][index]
        if not self._rides_measured():
            self._ride, self._trim_cap = ("", 0.0), self._cap_for([])
            return self._say_held(op["id"], None)
        if self._joins(op, from_arc):   # on pen-down where the tip stands: its chain's lift and trim cap stay
            return
        ops, path = self._chain(program, index, from_arc)
        beyond = path[np.linalg.norm(path - path[0], axis=1) > self.RIDE_AROUND_M]   # past its own hover's ring
        now, rises = self.clock.now(), self._rise_at(path)   # what the pen rides by already, along its path
        rise = float(rises[0])                                # ... at its hover

        def by(xy, pts, reach):
            return len(pts) > 0 and float(np.min(np.linalg.norm(pts - xy, axis=1))) <= reach
        kept = [(xy, gap, oid) for t, xy, gap, oid in self._skins if oid == op["id"] or now - t <= self.RIDE_KEEP_S]
        own = [gap for _, gap, oid in kept if oid == op["id"]]
        read = (own + [gap for xy, gap, oid in kept if oid != op["id"] and by(xy, beyond, self.RIDE_NEAR_M)] if own
                else [gap for xy, gap, _ in kept if by(xy, path, self.RIDE_AROUND_M)])
        stop = float(curved.config(self.stack)["rigid"]["skin_stop_m"])
        lift, cap = max(0.0, max(read) - rise) if read else stop, self._cap_for(read if own else [])
        if again := self._ride[0] == op["id"]:   # measured again: never lower, nor deeper
            lift = max(lift, self._ride[1])
            cap = cap if cap is None or self._trim_cap is None else max(cap, self._trim_cap)
        before = self._ridden[2] if again and self._ridden is not None and len(self._ridden[2]) == len(path) else 0.0
        along, fallback = self._profile(op["id"], path, lift, rises), lift
        ride = None if along is None else np.maximum(along[1], self._ride[1] + before if again else -np.inf)
        if ride is not None:   # the stroke rides its skin along its path: its least as the pen's, the rest per point
            lift = float(ride.min())
        self._ridden = None if ride is None or np.ptp(ride) < 1e-5 else ({o["id"] for o in ops}, path, ride - lift)
        self._ride, self._trim_cap = (op["id"], lift), cap
        self._say_held(op["id"], max(read) - rise if own else None)
        if ride is not None:
            part, setup, self._set_up_for = along[0], self._set_up_for == op["id"], ""
            self.event(Event.KIND_PAGE, f"track: {op['id']} rides its skin: read along "
                       f"{np.isfinite(part).mean() * 100:.0f}% of its path, {np.nanmin(part) * 1e3:+.2f}.."
                       f"{np.nanmax(part) * 1e3:+.2f} mm off the field; it rides {lift * 1e3:+.2f}..{ride.max() * 1e3:+.2f}"
                       f" mm off its plan, unread {fallback * 1e3:+.2f}; set up again for it: {'yes' if setup else 'no'}",
                       op["id"])
        elif not read:
            self.event(Event.KIND_PAGE, f"track: no hover read the skin at {op['id']} nor within "
                       f"{self.RIDE_AROUND_M * 1e3:.0f} mm of its path: it rides page.surface.rigid.skin_stop_m "
                       f"{stop * 1e3:.1f} mm higher", op["id"])
        elif lift > 0.0:
            whose = ("its hover" + (f" and {len(read) - len(own)} more along its path" if len(read) > len(own) else "")
                     if own else f"no skin read at its hover; {len(read)} hovers within {self.RIDE_AROUND_M * 1e3:.0f} mm")
            self.event(Event.KIND_PAGE, f"track: {op['id']} rides {lift * 1e3:.2f} mm higher: {whose} read the skin up to "
                       f"{max(read) * 1e3:.2f} mm over the field" + (f", {rise * 1e3:.2f} of it the checks' rise"
                                                                    if rise else ""), op["id"])

    def _chain(self, program: dict, index: int, from_arc: float = 0.0) -> tuple[list, np.ndarray]:
        """(the stroke and the continuations it joins pen-down, their print points from from_arc as they are planned:
        moved by the print's correction, _dense)."""
        from tatbot_motion import timelaw

        ops = [program["ops"][index]]
        for nxt in program["ops"][index + 1:]:
            if nxt.get("op") != "stroke" or not nxt.get("continues"):
                break
            ops.append(nxt)
        snap = self._snap(ops[0])
        chain = [np.asarray((o if snap is None else snap.corrected(o)[0])["points_m"], float) for o in ops]
        return ops, _dense([timelaw.trim_polyline(chain[0], from_arc), *chain[1:]])

    def _profile(self, op_id: str, path: np.ndarray, fallback: float, rise):
        """(the skin's part over the field along `path`, NaN unread; the ride at each point) from the skin its hover's
        views read 8-40 mm about the tip (track.Tracker.wide; curved.skin_along, ride_profile); where they read none
        (the pen hides 8 mm about its tip, most of a 10 mm stroke), the page's frames of earlier strokes within
        profile_keep_s, newest first (a page set up again clears them); None with none read."""
        frames, surface = getattr(self.tracker, "wide", []), self.surface()
        own = [w for _, o, w in frames if o == op_id]
        if not own or surface is None or not len(path):
            return None
        rigid, now = curved.config(self.stack)["rigid"], self.clock.now()
        near, own = float(rigid["profile_near_m"]), np.concatenate([np.reshape(w, (-1, 3)) for w in own])
        read = curved.skin_along(surface, own, path, near)
        for _t, _, w in sorted((f for f in frames if f[1] != op_id and now - f[0] <= float(rigid["profile_keep_s"])),
                              key=lambda f: -f[0]):   # a lent frame only where it agrees with the own views' skin
            if not (unread := ~np.isfinite(read)).any():
                break
            if (off := curved.lent_off(surface, own, w, near)) is not None and abs(off) <= float(rigid["profile_agree_m"]):
                read[unread] = curved.skin_along(surface, w, path[unread], near)
        if not np.isfinite(read).any():
            return None
        return curved.ride_profile(read, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))], fallback,
                                   rise, rigid)

    OFF_SETUP = (0.006, 3)   # this many stroke ends this far off the plan one way set the page up again: the left
    # forearm's ends read 2.1 mm at most, half of it the wrist's read error there (2026-10-20); 3 mm until the owner
    # doubled every stop
    OFF_CLEAR_M = 0.002   # a stroke may land this far onto the stencil's border before the page is set up for it

    def _off_plan(self, program: dict, op: dict, index: int, ledger) -> None:
        """Before a stroke, when stencil tracking corrects and the last track.off_plan_count strokes all ended past
        track.off_plan_m from the plan one way (Tracker.off_plan): the page follows the overhead's markers at once
        (_follow_marks); when OFF_SETUP's did, the page is set up again. No pause, nor with the pen at the page."""
        if self.tracker is None or not (self.tracker.correcting or self._rigid()) or op.get("continues"):
            return
        found = self.tracker.off_plan()
        if found is None or (self._height() or 0.0) < float(self.motion["approach"]["final_m"]):
            return
        far = self.tracker.off_plan(*self.OFF_SETUP)
        mean, ends = far or found
        read = ", ".join(f"{op_id} {np.linalg.norm(off) * 1e3:.1f}" for op_id, off in ends)
        read = f"the last strokes ended off the plan ({read} mm; {mean[0] * 1e3:+.1f} {mean[1] * 1e3:+.1f} mm on average)"
        if far is not None:
            return self._reacquire(program, ledger, f"{read}, each past {self.OFF_SETUP[0] * 1e3:.0f} mm", op["id"])
        if not (self.marks is not None and self._follow_marks(program, ledger, op["id"], 0.0, f" ({read})")):
            self.event(Event.KIND_PAGE, f"track: {read}: the field follows them", op["id"])

    def _off_clear(self, program: dict, op: dict, index: int, ledger) -> None:
        """Before a stroke that the strokes' consistent offset from the plan (Tracker.off_plan) would carry past the
        print's clear area by more than OFF_CLEAR_M the page is set up again, rather than draw on the stencil's border;
        never a pause (owner, 2026-10-20: no keys for decisions). Tracking that only logs measures where strokes land."""
        clear = (program.get("page") or {}).get("clear_m")
        if self.tracker is None or op.get("continues") or not clear:
            return
        found = self.tracker.off_plan()
        if found is None:
            return
        mean = found[0]
        points = np.asarray(op["points_m"], float)[:, :2] + mean
        past = float(max(np.abs(points[:, 0]).max() - clear[0] / 2, np.abs(points[:, 1]).max() - clear[1] / 2))
        if past <= self.OFF_CLEAR_M:
            return
        self._reacquire(program, ledger, f"clear area: the strokes land {mean[0] * 1e3:+.1f} {mean[1] * 1e3:+.1f} mm off "
                        f"the plan, which would carry this stroke {past * 1e3:.1f} mm onto the stencil's border", op["id"])

    def _measure_over(self, op: dict, name: str, plan_xy, travel, from_arc: float, ledger) -> str | None:
        """Travel to the hover over op's start (the pre-planned travel first), then the tracker's verdict there."""
        snap = self.tracker.snapshot() if self.tracker.correcting else None
        planned = [travel] if travel is not None else []
        self.move(lambda: planned.pop() if planned else self.plan_hover(op, self.io.seed(), snap, from_arc),
                  P.PHASE_TRAVEL, f"{op['id']} hover", ledger)
        self.trim_end = 0.0   # a travel carries no pen trim
        return self.tracker.at_hover(name, op["id"], plan_xy)

    def _past_reach(self, setup: np.ndarray, current: np.ndarray | None) -> bool:
        """With stencil tracking correcting, whether the overhead sees the sheet moved past what the wrist fit reaches
        from the page set up at `setup` (track.reach_m, reach_rad), or has no fresh pose (None) to say it has not."""
        if not self._corrects():
            return False
        cfg = self.stack["page"]["track"]
        return current is None or geometry.page_moved(setup, current, float(cfg.get("reach_m", 0.010)),
                                                      float(cfg.get("reach_rad", 0.05)))

    def _corrects(self) -> bool:
        """Stencil tracking corrects strokes for a moved sheet (page.track.mode correct) with the real wrist camera."""
        return (self.stack["page"].get("track") or {}).get("mode") == "correct" and self.sensing.wrist == "real"

    def _sheet_may_have_left(self, program: dict | None = None, ledger=None) -> bool:
        """After a hover found no print: the overhead's pose, measured within page.max_lost_s (the arm over the page
        hides it), moved past the wrist fit's reach; else most of its poses measured since the pen last left the
        page (a sheet moves under the pen or a hand, and the overhead sees it at every dip) past it; or no such pose.
        The 5 s rule alone set a half-skin up again three times in 25 ops (2026-10-06): its print reads poorly at
        the page's foot, and every hover there found the last overhead pose half a minute old, from the dip. With
        the print's markers kept at setup: no reading of them since the pen left the page, or the watch reads the
        sheet moved once a move not yet held is waited out (_waited); the pose's tilt read 53 of 53 still forearm
        poses past reach (2026-10-19)."""
        if self.marks is not None:
            return self._corrects() and (not self._timed_moves(max(self.inked_at, self.overhead_at))
                                         or self._waited(program, ledger)[0] is not None)
        newest = self.node.camera_page(self.arm)
        lost = float(self.stack["page"].get("max_lost_s", 5.0))
        setup = self.page_overhead()
        if newest is not None and geometry.stale_page(newest[1], lost) is None:
            return self._past_reach(setup, newest[0])
        since = self.node.measured_pages(self.arm, max(self.inked_at, self.overhead_at))
        if since:
            return 2 * sum(self._past_reach(setup, pose) for pose in since) > len(since)
        return self._past_reach(setup, None)

    def page_overhead(self) -> np.ndarray | None:
        """What the overhead's later poses are compared with: its own pose of the sheet the page in use stands for, its
        pose at setup (not the page the wrist located from it: 7.5 mm apart on a half-skin, 2026-10-06, the
        overhead's bias, not a move) carried by the slides the page has followed since."""
        base = self.overhead if self.overhead is not None else self.camera
        return None if base is None else self.slid @ base

    WATCH_POSES = 5   # the overhead's newest measured poses _watch_sheet judges by; most of them must agree

    def _watch_sheet(self, program: dict, op: dict, from_arc: float, ledger) -> None:
        """Before a stroke descends, its pen lifted, the page follows the sheet: by the print's markers when kept
        (_follow_marks); else, without hover stops, by the overhead's poses: once the slide it measures since setup
        (sheet_slide) has a part the page has not followed past track.follow_m or follow_rad (1.5 mm, ~1 deg; the
        median of its five newest poses reads ~0.5 mm and turns in whole degrees), the page moves by it
        (_follow_overhead), and a step that most of those poses put past track.follow_max_m or follow_max_rad (10 mm,
        5.7 deg; a sheet picked up and put down) sets the page up again. On 2026-10-06 needle drag slid a half-skin 15
        mm and 6 deg over 20 strokes; replayed, following closes the drift between jumps (0.2-0.9 mm against 1.2-2.2)."""
        if op.get("continues") or from_arc > 0 or not self._corrects():
            return
        if self.marks is not None:
            self._follow_marks(program, ledger, op["id"])
        elif not self._hover_on():
            self._follow_overhead(program, op, ledger)

    def _follow_marks(self, program: dict, ledger, op_id: str, least: float | None = None, why: str = "") -> bool:
        """The page follows the markers' drift (sheet_slide) once its unfollowed part moves the drawing past
        page.track.watch follow_m (`least`) or turns past follow_rad (_follow_sheet; below, the filter carries it), or
        an inking pen's curved page is set up again past _rolled's limit. True when the page moved."""
        slide, cfg = self.sheet_slide(), self._watch()
        if slide is None:
            return False
        step, used, points = slide @ np.linalg.inv(self.slid), self.used(), self._drawing(program)
        at = used @ np.c_[points, np.zeros(len(points)), np.ones(len(points))].T
        moved = float(np.max(np.linalg.norm((step @ at - at)[:3], axis=0)))
        if moved <= (cfg["follow_m"] if least is None else least) and abs(_yaw(step)) < cfg["follow_rad"]:
            return False
        if (rolled := self._rolled(program, slide, op_id)) and self.motion["pen"]["mode"] != "hover":
            return self._reacquire(program, ledger, rolled, op_id) or True
        self._follow_sheet(slide, ledger, op_id, why + (f" (inking, {rolled}: set up)" if rolled else ""))
        return True

    # Why a curved page should not follow `slide` from its holds: a print turned round the arm (its skin's shape staying;
    # the overhead reads it alike) would stand the skin past page.track.watch roll_m over the strokes from op_id on, the
    # needle that much deeper. None on a flat page, or within it.
    def _rolled(self, program: dict, slide: np.ndarray, op_id: str = "") -> str | None:
        surface, uv = self.surface(), self._stroke_uv(program, op_id)
        if surface is None or not len(uv):
            return None
        frame = np.linalg.inv(self.slid) @ self.used()   # the page the holds measured
        plan, _ = surface.moved(slide @ frame).lift(uv)
        local = (np.linalg.inv(frame) @ np.c_[plan, np.ones(len(plan))].T)[:3]
        over, moved = float(np.max(surface.height(local[:2].T) - local[2])), (np.linalg.inv(frame) @ slide @ frame)[:2, 3]
        return None if over <= self._watch()["roll_m"] else (
            f"the overhead sees the sheet {moved[0] * 1e3:+.1f} {moved[1] * 1e3:+.1f} mm from its holds: a print turned round "
            f"the arm reads alike and would stand the skin {over * 1e3:.1f} mm over the followed strokes, so the holds measure which")

    # Up to 400 stroke points (page metres) of the program's ops from op_id on (all without it).
    @staticmethod
    def _stroke_uv(program: dict, op_id: str = "") -> np.ndarray:
        ops = program["ops"][next((k for k, o in enumerate(program["ops"]) if o.get("id") == op_id), 0):]
        uv = np.array([p[:2] for o in ops if o.get("op") == "stroke" for p in o["points_m"]], float).reshape(-1, 2)
        return uv[np.linspace(0, len(uv) - 1, min(len(uv), 400)).astype(int)]

    def _follow_overhead(self, program: dict, op: dict, ledger, why: str = "", turn_rad: float | None = None) -> str | None:
        """The page follows the overhead's measured slide by its poses (_watch_sheet, a page whose print's markers are
        not kept): 'set up' when most of its newest poses put the sheet past track.follow_max_m or follow_max_rad,
        'followed' when the unfollowed part passed follow_m or its turn passed turn_rad (track.follow_rad by default),
        else None (fewer than three poses since setup, or the sheet where the page is)."""
        frame, moves = self._slides() if self.camera is not None else (None, [])
        if len(moves) < 3:
            return None
        page, cfg = self.page_overhead(), self.stack["page"]["track"]
        if self._curved():   # the overhead's match on a raised surface is coarser: curved.WATCH
            cfg = {**cfg, **{k: v for k, v in curved.config(self.stack).items() if k.startswith("follow")}}
        now = [frame @ m @ np.linalg.inv(frame) @ np.linalg.inv(self.slid) @ page for m in moves]   # each reading's page
        past = [p for p in now if geometry.page_moved(page, p, float(cfg.get("follow_max_m", 0.010)),
                                                      float(cfg.get("follow_max_rad", 0.1)))]
        if 2 * len(past) > len(moves):
            far = past[-1]
            shift = float(np.linalg.norm(far[:3, 3] - page[:3, 3])) * 1000
            turn = float(np.degrees(geometry.rotation_angle(page, far)))
            self._reacquire(program, ledger, f"the overhead sees the sheet {shift:.1f} mm, {turn:.1f} deg from where the "
                            f"page was set up ({len(past)} of its newest {len(moves)} readings past)", op["id"])
            return "set up"
        slide = self.sheet_slide()
        step, used = slide @ np.linalg.inv(self.slid), self.used()
        moved = float(np.linalg.norm((step @ used)[:2, 3] - used[:2, 3]))
        turn_rad = float(cfg.get("follow_rad", 0.015)) if turn_rad is None else turn_rad
        if moved < float(cfg.get("follow_m", 0.0015)) and abs(_yaw(step)) < turn_rad:
            return None
        self._follow_sheet(slide, ledger, op["id"], why)
        return "followed"

    def _follow_sheet(self, slide: np.ndarray, ledger, op_id: str, why: str = "") -> None:
        """Move the page in use by the part of the overhead's measured slide it has not followed (no touches, no
        holds: the overhead measures a slide to ~1-2 mm though its own pose is several mm off the print), recorded as
        a page row so a resume takes the page where it went."""
        step, old = slide @ np.linalg.inv(self.slid), self.used()
        self.correction = np.linalg.inv(self.camera) @ step @ self.camera @ self.correction
        self.slid = slide
        self.stop_preplan()
        self._chain_snap = self._correction = None
        new = self.used()
        d = np.linalg.inv(old) @ new
        self.page_record.update(used=new.tolist(), correction=self.correction.tolist(),
                                overhead=self.page_overhead().tolist(), slid=self.slid.tolist())
        self.page_record.setdefault("followed", []).append({"t": self.clock.utc(), "slide": slide.tolist()})
        if self.placed is not None:
            self.placed = {"page": new, "overhead": self.page_overhead()}
        ledger.append("page", self.arm, page=self.page_record, workspace=self.workspace())
        self.event(Event.KIND_PAGE, f"track: the overhead sees the sheet move {d[0, 3] * 1e3:+.1f} {d[1, 3] * 1e3:+.1f} mm, "
                   f"{np.degrees(_yaw(d)):+.1f} deg: the page follows it{why}", op_id)

    def _recent_poses(self) -> list:
        """The overhead's newest measured poses of the page (WATCH_POSES of them) since the page was set up."""
        return self.node.measured_pages(self.arm, self.overhead_at)[-self.WATCH_POSES:]

    def _slides(self) -> tuple[np.ndarray | None, list]:
        """(a page frame, the sheet's move since setup by each of the overhead's newest readings in it: the sheet now =
        frame @ move @ frame^-1 @ the sheet then). With the print's markers kept, each reading's markers' move
        (_mark_moves) in-plane on the page they were kept on (the page in use less the slides it followed), no pose
        read: the overhead's tilt and height are a plane's over a curved skin. Else its poses against its own at setup."""
        if self.marks is not None:   # their page.track.watch follow_s median (a gap holds the newest)
            return np.linalg.inv(self.slid) @ self.used(), [curved.in_plane(m[2]) for m in geometry.window(
                self._timed_moves(), self._watch()["follow_s"])]
        base = self.overhead if self.overhead is not None else self.camera
        return base, [] if base is None else [np.linalg.inv(base) @ pose for pose in self._recent_poses()]

    def sheet_slide(self) -> np.ndarray | None:
        """The sheet's slide since setup in the arm base as the overhead measures it (a 4 x 4: the sheet now = it @ the
        sheet then), in-plane: of each reading's move (_slides) the page centre's displacement and the turn in its own
        page frame, their medians, so a turn read in whole degrees is paired with the slide at the centre, never about
        the base's origin 0.3 m away; None with no reading since setup."""
        frame, moves = self._slides()
        return _median_pose(frame, [frame @ m for m in moves]) @ np.linalg.inv(frame) if moves else None

    def _reacquire(self, program: dict, ledger, why: str, op_id: str = "") -> None:
        """Set the page up again mid-draw (the wrist camera is the page setup's while it runs) and start tracking
        again from its gauge holds. A pen within the standoff first lifts along the normal of the page it stands on,
        before any travel (a pen-down continuation, a resumed stroke, a hover under the standoff)."""
        self.event(Event.KIND_PAGE, f"track: {why}; setting the page up again", op_id)
        since, self._skin_off = self.clock.utc(), 0.0
        self.stop_preplan()
        self.machine_off()
        self._stop_tracking()
        self.lift_off("lift before setting the page up again", ledger)
        self._again = True
        self._new_page(program, self._goal_touch, ledger)
        self._ease_trims(ledger, "the page was set up again")
        self._chain_snap = self._correction = None
        self._track_since = since
        self.tracker = track.Tracker.start(self, since=since, centre=self._drawing(program)[-1])

    def _new_page(self, program: dict, touch: bool, ledger) -> None:
        self.setup_page(program, touch, ledger)
        if _unconfirmed(self.page_record):   # _confirm_holds: once more, afresh, its holds' frames the track's seed
            self.event(Event.KIND_PAGE, "the check holds did not confirm the holds' move: the page set up again once")
            self._again, self.seed_since = True, self.clock.utc()
            self.setup_page(program, touch, ledger)
        if self.pen is not None:
            self.page_record["pen"] = dataclasses.asdict(self.pen)
        ledger.append("page", self.arm, page=self.page_record, workspace=self.workspace())

    def _pre_travel(self, op: dict):
        """This op's pre-planned hover travel, when it starts where the arm is (dispatch_drift), from there: planned from
        the stroke's untrimmed end, it began up to the trim off the arm (2 mm when the trim's limit was, 2026-10-20; a
        gap past dispatch_drift's 2 mm is planned again), which the controller swept within 10 ms (blend_start)."""
        from tatbot_motion import blend_start, dispatch_drift

        pre = self._preplan
        if pre is None or pre[0] != op["id"]:
            return None
        try:
            travel = pre[1].result()[0]
        except Exception:  # noqa: BLE001 - dispatch_plan reports a failed pre-plan
            return None
        if travel is None or not dispatch_drift(travel, self.io.measured(), self.kin, self.motion, pen_down=False)["ok"]:
            return None
        return blend_start(travel, self.io.seed(), self.kin)

    # --- the operator's pen trim (motion.yaml pen.trim, ros/README.md 4.4) -------------------------
    def trim_target(self) -> float:
        return self.trims.get(self.resource["id"], 0.0) if self.resource else 0.0

    # With the +-4 mm trim, depth keyed for one sunk stretch of a forearm carried, one trim per cartridge, onto strokes,
    # set-ups' fresh fields and resumes it was never pressed for, up to the 3RL's whole reach: its tube 1.0 mm over the
    # gauged skin, its needles 3.0 mm past it, where skin_stop_m (2.5 mm past the plan) is the operator's deepest rung
    # (review, 2026-10-21). A needle cartridge's stroke takes the trim no deeper than skin_stop_m past its plan over the
    # highest skin read along it, less the rise it rides by: its own hover's readings and, past its ring, those near its
    # path (_ride_over). A deep trim reaches only as far as those hovers read the skin sunk, and a stroke that holds the
    # keys back says so. A stroke whose own hover read no skin takes skin_stop_m: the hovers about its path lift it, but
    # a sink one of them read never takes it deeper (review, 2026-10-21: keys pressed to -4.0 over skin read 4 mm sunk
    # reached, at their whole depth and unsaid, a stroke 10 mm on whose own hover read no skin). A set-up
    # or a resume eases a trim kept deeper than skin_stop_m back to it (_ease_trims). The keys never go deeper than they
    # set. Replayed over that forearm's presses at +-4 mm (2026-10-20): 41 presses on a stroke whose hover read the skin
    # 0.15 mm under the field held at -2.65 instead of -4.0, the resume's -4.0 eased to -2.5, the two strokes read
    # 3.7-4.1 mm under took -4.0, and nine earlier strokes held 0.06-0.24 mm. The eased trim is a pen_trim row
    # (eased_from_m) a resume starts from; the cap is the sent row's trim_cap_m, which the trim bar shows.
    def _cap_for(self, read: list) -> float | None:
        """The deepest trim a stroke over skin read `read` (m over the field) takes; None but for a needle cartridge."""
        if self.pen is None or not self.pen.machine or self.pen.reach_m is None:
            return None
        rise = float((self.page_record.get("surface") or {}).get("rise_m", 0.0))
        return -(float(curved.config(self.stack)["rigid"]["skin_stop_m"]) + (max(0.0, rise - max(read)) if read else 0.0))

    def _capped(self, cap: float | None) -> float:
        return self.trim_target() if cap is None else max(self.trim_target(), cap)

    def trim_in_use(self) -> float:
        """The cartridge's trim as the stroke in hand takes it: no deeper than its cap (_ride_over)."""
        return self._capped(self._trim_cap)

    def _say_held(self, op_id: str, high: float | None) -> None:
        """Say a stroke takes the trim shallower than the keys set it; high: the most skin read along it over its plan."""
        if (used := self.trim_in_use()) > (keys := self.trim_target()) + 1e-9:
            over = "no skin read at its own hover" if high is None else f"the skin read along it {high * 1e3:+.2f} mm over it at most"
            self.event(Event.KIND_PEN_TRIM, f"{self.resource['id']} {keys * 1e3:+.1f} mm held at {used * 1e3:+.2f} mm on "
                       f"{op_id}: the needles go no deeper than page.surface.rigid.skin_stop_m past its plan, {over}", op_id)

    def _ease_trims(self, ledger, why: str) -> None:
        """A fresh field's (a set-up's or a resume's): each trim kept deeper than _cap_for([]) back to it, and said."""
        most = self._cap_for([])
        for res, trim in sorted(self.trims.items()):
            if most is not None and trim < most - 1e-9:
                self.trims[res] = most
                ledger.append("pen_trim", self.arm, resource=res, trim_m=round(most, 7), eased_from_m=round(trim, 7))
                self.event(Event.KIND_PEN_TRIM, f"{res} {trim * 1e3:+.1f} mm eased back to {most * 1e3:+.1f} mm: {why}, "
                           "and a deeper trim reaches only where the hovers read the skin sunk")

    def trimmed(self, traj, start_m: float):
        """traj with the pen trim (and its pre-plan's ride fold: _ride_fold), easing from start_m (where the arm is) to
        the cartridge's."""
        from tatbot_motion import Trim, compose

        trim, target = Trim(start_m), self.trim_in_use() + self._fold
        return compose(traj, trim if abs(target - start_m) < 1e-9 else trim.to(0.0, target, self.motion["pen"]["trim"]))

    def _retarget(self, plan):
        """ArmIO.execute's retarget: the goal again, easing to a changed trim from pen.trim.lead_s ahead; None while
        a change still eases in, or when it would not settle before the goal ends (the next goal starts at it)."""
        from tatbot_motion import compose

        cfg = self.motion["pen"]["trim"]

        def retarget(elapsed: float, active):
            trim, target = active.info["trim"], self.trim_in_use() + self._fold
            if abs(trim.end_m - target) < 1e-9 or elapsed < trim.settled_s():
                return None
            new = trim.to(elapsed + float(cfg["lead_s"]), target, cfg)
            return compose(plan, new) if new.settled_s() < plan.t[-1] else None

        return retarget

    def on_key(self, key: str) -> None:
        """A numpad key: up/down move the cartridge's trim a step within its limit, reset sets 0, each a ledger
        row a resume starts from; dip (+) and fewer_dips (-) step its dip scale (dip_key); enter answers a waiting
        pause or latch CONTINUE."""
        if key in ("dip", "fewer_dips"):
            return self.dip_key(key)
        if key == "enter" and not self.io.flag("landed"):   # landed: a swap's, which `client draw` answers
            ok, why = (False, "nothing waits for continue") if self.goal_active and self.waiting is None \
                else self.decide(rules.CONTINUE)
            if not ok:
                self.event(Event.KIND_DECISION, f"enter: {why}")
        if key not in ("up", "down", "reset"):
            return
        if self.ledger is None or self.resource is None:
            return self.event(Event.KIND_PEN_TRIM, f"{key}: no draw is running, so no cartridge to trim")
        step, limit = (float(self.motion["pen"]["trim"][name]) for name in ("step_m", "limit_m"))
        res, old, most, of = self.resource["id"], self.trim_target(), int(limit / step + 1e-6), f" of ±{limit * 1e3:.1f}"
        steps = 0 if key == "reset" else round(old / step) + (1 if key == "up" else -1)
        new = max(-most, min(most, steps)) * step
        if abs(new - old) < 1e-12:
            return self.event(Event.KIND_PEN_TRIM, f"{res} stays {new * 1e3:+.1f} mm" + (f": its limit{of}" if steps else ""))
        self.trims[res] = new
        self.ledger.append("pen_trim", self.arm, resource=res, trim_m=round(new, 7))
        used = self.trim_in_use()
        held = f", held at {used * 1e3:+.2f} mm by the skin read along the stroke" if used > new + 1e-9 else ""
        over = f": the path {(self.pen.height_m + used) * 1e3:.2f} mm over the page" if self.pen else ""
        self.event(Event.KIND_PEN_TRIM, f"{res} {new * 1e3:+.1f} mm{of}{held}{over}")

    # --- the operator's dips (charge.py, ros/README.md 6) ---------------------------------------------
    def dip_scale(self) -> float:
        return self.dip_scales.get(self.resource["id"], 1.0) if self.resource else 1.0

    def dip_key(self, key: str) -> None:
        """+ (dip): a dip at the next safe point (a running stroke stops where it is, and goes on from there after
        the dip) and every later dip a step sooner; - (fewer_dips): every later dip a step later. Each a dip_scale
        ledger row a resume starts the resource at. Nothing changes without a running draw, for a resource that
        takes no dips, or on + while a dip runs or is asked."""
        res, ledger, sign = self.resource, self.ledger, "+" if key == "dip" else "-"
        with self._lock:
            why = None
            if ledger is None or res is None:
                why = "no draw is running, so no cartridge to dip"
            elif res.get("dip") is None:
                why = f"{res['id']} ({res['tool']['id']}) takes no dips"
            elif key == "dip" and (self.dipping or self.dip_asked):
                why = "a dip is running" if self.dipping else "a dip is asked already"
            elif key == "dip":
                self.dip_asked = True
        if why:
            return self.event(Event.KIND_DIP, f"{sign}: {why}; nothing changed")
        old = self.dip_scale()
        new = self.dip_scales[res["id"]] = charge.stepped(old, key)
        ledger.append("dip_scale", self.arm, op=self.op_now or None, resource=res["id"], scale=new, key=sign)
        self.event(Event.KIND_DIP, f"{sign}: {res['id']} dips every {charge.budget_m(res, new) * 1e3:.0f} mm "
                   f"(x{new:.1f}{', its limit' if new == old else ''})" + ("; dipping now" if key == "dip" else ""),
                   self.op_now)

    def _stopped(self, status: str, text: str) -> tuple[str, bool]:
        """Why a stroke goal ended short, and whether it stopped for a dip (_stroke_stop), the machine then off at once."""
        if status == "cancelled" and self._stop_why is not None:
            self.machine_off()
            return f"the sheet moved: {self._stop_why}", False
        if status == "cancelled" and not self.stopping():
            self.machine_off()
            return ("numpad dip" if self.dip_asked else "charge spent"), True
        return {"latched": f"latched: {text}", "cancelled": "cancelled"}.get(status, f"failed: {text}"), False

    def _stroke_stop(self, from_arc: float) -> bool:
        """A stroke goal's should_stop: a land or cancel, a dip due (the numpad asked for one, or the charge has drawn
        its line), or the sheet moved under it (_moved_under)."""
        budget = self._budget()
        return (self.stopping() or self.dip_asked or self._stop_why is not None
                or (budget is not None and charge.spent(self.inked_m, self._arc_now - from_arc, budget)))

    # A stroke draws with the correction its plan took; the readings since (its own while it draws, the overhead's
    # marker moves) go into the filter at once but reach only the strokes after it. On a silicone skin pushed ~30 mm
    # round a bottle mid-stroke (2026-10-24) the overhead and four wrist readings agreed on the move within 5 s, and the
    # stroke drew on the old place 8 s more until the e-stop. Past page.track.stop_m the stroke stops, lifts, and goes
    # on from its arc after the checks between strokes (the sheet followed, or the page set up again).
    def _moved_under(self, op: dict, arc_m: float) -> str | None:
        """Why the stroke drawing now must stop: the filter's correction at the pen past page.track.stop_m from the one
        the stroke is drawn with (_chain_snap), or None."""
        stop = float((self.stack["page"].get("track") or {}).get("stop_m", 0.0))
        if not stop or self._chain_snap is None or self.tracker is None or not self.tracker.correcting:
            return None
        at = {"points_m": [track.point_at(op, arc_m).tolist()]}
        moved = float(np.linalg.norm(self.tracker.snapshot().corrected(at)[1][0] - self._chain_snap.corrected(at)[1][0]))
        return None if moved <= stop else (f"the print's correction at the pen moved {moved * 1e3:.1f} mm from the one "
                                           f"{op['id']} draws with (page.track.stop_m {stop * 1e3:.0f} mm)")

    def stroke(self, program: dict, index: int, ledger) -> None:
        op = program["ops"][index]
        length = geometry.stroke_length(op["points_m"], closed=bool(op.get("closed")))
        from_arc, trips, judged = ledger_mod.resume_arc(ledger.rows(), self.arm, op["id"]), 0, False
        while True:
            self.refresh_page(program, ledger, op["id"])   # a big move: set up again before the arm goes over it
            self._off_plan(program, op, index, ledger)
            self._off_clear(program, op, index, ledger)
            self._watch_sheet(program, op, from_arc, ledger)
            began = self.clock.now()
            self._hover_stop(program, op, from_arc, ledger, index)
            if not judged and self._moved_meanwhile(program, began, op["id"]):
                judged = True   # once a stroke: refresh_page judges it, the hover measures again
                continue
            if self._dip_first(program["ops"], index, from_arc):
                return   # _run_ops dips, and the stroke goes on from from_arc
            self._ride_over(program, index, from_arc)
            self.snag_guard(True)   # a riding pen's only
            if self.pen and self.pen.machine and not self.machine_running(op["id"], index, ledger, again=False):
                # going on after a pause or a latch (below): the needle off the skin along its normal first, and the
                # skin measured again before it goes back in (a person may have moved while the arm waited)
                self.lift_off(f"{op['id']} lift to measure the skin again", ledger)
                continue
            traj, sent = self.dispatch_plan(program, op, from_arc, ledger.rows())
            ledger.append("sent", self.arm, op=op["id"], index=index, arc_m=from_arc, **self._correction_row(),
                          **({"ride_m": round(self._ride[1], 6)} if self._ride[1] else {}),
                          **({"trim_cap_m": round(self._trim_cap, 6)} if self._trim_cap is not None else {}),
                          **({"fold_m": round(self._fold, 6)} if self._fold else {}))
            self.event(Event.KIND_OP_SENT, f"{len(traj.t)} knots, {traj.t[-1]:.1f} s from arc {from_arc * 1000:.1f} mm",
                       op["id"])
            self._arc_now, self._stop_why = from_arc, None
            status, reached, text = self.io.execute(sent, should_stop=functools.partial(self._stroke_stop, from_arc),
                                                    on_tick=self._ticker(program, index, op, traj, ledger),
                                                    retarget=self._retarget(traj))
            self.inked_at = self.clock.now()
            sent = self.io.active   # a trim change mid-goal replaced it
            self.trim_end = float(sent.info["trim"].value(sent.t[reached]))
            line = self.drawn[op["id"]][-1]   # the measured pen-down samples, for drawn.svg across resumes
            if status == "ok":
                self.inked_m += length - from_arc
                ledger.append("done", self.arm, op=op["id"], index=index, arc_m=length, line=line)
                self.event(Event.KIND_OP_DONE, "", op["id"])
                self._track_lift(program, op, ledger)
                return
            self._preplan = None
            reason, for_dip = self._stopped(status, text)
            arc, finished = self._ended_short(op, index, traj, reached, from_arc, reason, ledger)
            if for_dip:
                return   # _run_ops lifts and dips, and the stroke goes on from its arc
            if status == "cancelled" and not self.land_requested and self._stop_why is None:
                raise Cancelled()
            if self._stop_why is None:   # a moved sheet waits for no decision: the checks between strokes take it
                trips = self._latched(program, op, index, ledger, status, trips)
            # off the skin along its normal, a drawn stroke's into a continuation too: the next goes down measured again
            self.lift_off(f"{op['id']} lift" + ("" if finished else " to measure the skin again"), ledger)
            if finished:
                return
            from_arc = arc

    # A stroke goal that latched or failed waits for a decision (land lands), but a snag-guard trip takes the needle out
    # at once (_withdraw) and goes on by itself, measured again; its second on the stroke sets the page up again first,
    # its third waits for a decision with the needle out. Returns the stroke's snag trips.
    def _latched(self, program: dict, op: dict, index: int, ledger, status: str, trips: int) -> int:
        trips += (snag := status == "latched" and self.io.latch_reason == names.LATCH_GUARD_SNAG)
        out = snag and self._withdraw(op["id"], trips > 2)
        if not (out and trips < 3) and self.wait_decision("latched", op["id"], index, ledger) == rules.LAND:
            self.land()
        if out and trips == 2:
            self._reacquire(program, ledger, f"the snag guard tripped twice on {op['id']}", op["id"])
        return trips

    # A snag-guard trip (reason 11) takes the needle out at once, a standoff along the pen from wherever it stopped and
    # under the guard (its level outlives the latch), the latch released as a continue releases it (refused alike), the
    # machine off: before, it stayed in the skin until someone decided (up to resume_timeout_s). False, the arm holding,
    # when the release is refused or the guard holds this lift too (the needle held in the work): a person frees it.
    def _withdraw(self, op_id: str, decide: bool) -> bool:
        from tatbot_motion.clik import PlanError

        from tatbot_session import inspect as ins

        self.machine_off()
        ok, text = (False, why) if (why := self.continue_refusal(self.io.measured())) else self.io.unlatch()
        if ok and not np.all(np.abs(self.io.measured()[:6] - ins.REST[:6]) < self.AT_REST_RAD):   # at rest: no pen in the work
            try:
                status, _, text = self.io.execute(self.lift_plan(float(self.motion["approach"]["standoff_m"]),
                                                                 -self.tcp()[:3, 2]))   # out, a land or cancel asked or not
                ok = status == "ok"
            except PlanError as exc:   # a withdraw that cannot be planned holds the arm; it never ends the run
                ok, text = False, str(exc)
        self.event(Event.KIND_LATCHED, "snag guard: the needle stood off its commanded path, held in the work or carrying "
                   "it; the machine off, " + (("withdrawn along the pen" + ("; continue goes on, land lands" if decide else ""))
                                              if ok else f"not withdrawn: {text}; it holds there: free the needle, then "
                                              "continue or land"), op_id)
        return ok

    def _ended_short(self, op: dict, index: int, traj, reached: int, from_arc: float, reason: str,
                     ledger) -> tuple[float, bool]:
        """A stroke goal that ended short of `ok`: (the arc reached, finished). Finished when it stopped in the final
        lift: the stroke is drawn, `done`, since resuming would dot its end again; else `aborted` at the arc. The
        line it drew counts against the charge."""
        sent, line = self.io.active, self.drawn[op["id"]][-1]
        length = geometry.stroke_length(op["points_m"], closed=bool(op.get("closed")))
        arc = geometry.arc_at(sent, reached, self.tcp()[:3, 3])
        drawn = np.flatnonzero(np.asarray(traj.phase) == P.PHASE_DRAW)
        finished = bool((len(drawn) and reached > drawn[-1]) or arc >= length - float(self.motion["resume_reapproach_m"]))
        self.inked_m += max(0.0, (length if finished else arc) - from_arc)
        if finished:
            ledger.append("done", self.arm, op=op["id"], index=index, arc_m=length, reason=reason, line=line)
            self.event(Event.KIND_OP_DONE, f"{reason} after the stroke was drawn", op["id"])
        else:
            ledger.append("aborted", self.arm, op=op["id"], index=index, arc_m=arc, reason=reason, line=line)
            self.event(Event.KIND_OP_ABORTED, f"{reason} at arc {arc * 1000:.1f} mm", op["id"])
        return arc, finished

    MOVED_EVERY_S = 0.5   # a stroke's correction checked against the filter's this often (its readings' stroke_s)

    def _ticker(self, program, index, op, traj, ledger):
        """on_tick for a stroke goal: the arc reached, feedback at 10 Hz, drawn samples at 20 Hz, the pre-plan once, the
        tracker's frames while it draws."""
        state = {"fb": 0.0, "drawn": 0.0, "pre": False, "line": [], "moved": -np.inf}
        self.drawn.setdefault(op["id"], []).append(state["line"])
        inv_page = np.linalg.inv(self.used())

        def tick(i: int) -> None:
            now = self.clock.now()
            if np.isfinite(traj.arc_m[i]):
                self._arc_now = max(self._arc_now, float(traj.arc_m[i]))
            if not state["pre"]:
                state["pre"] = True
                self.preplan(program, index, np.asarray(traj.q[-1]), ledger.rows())
            if now - state["fb"] >= 0.1:
                state["fb"] = now
                arc = float(traj.arc_m[i]) if np.isfinite(traj.arc_m[i]) else 0.0
                self.feedback(phase=int(traj.phase[i]), op_id=op["id"], index=index, arc_m=arc)
            if traj.phase[i] == P.PHASE_DRAW and now - state["drawn"] >= 0.05 and self.io.q is not None:
                state["drawn"] = now
                tip = self.kin.fk(self.io.q)[:3, 3]
                state["line"].append((inv_page @ np.append(tip, 1.0))[:2].tolist())
            if traj.phase[i] == P.PHASE_DRAW and self.tracker is not None and np.isfinite(traj.arc_m[i]):
                self.tracker.while_drawing(op, float(traj.arc_m[i]))   # the wrist read as it draws (stroke_s)
                if self._stop_why is None and traj.t[i] - state["moved"] >= self.MOVED_EVERY_S:
                    state["moved"] = float(traj.t[i])
                    self._stop_why = self._moved_under(op, float(traj.arc_m[i]))
                    if self._stop_why is not None:
                        self.event(Event.KIND_PAGE, f"track: {self._stop_why}: stopping it, lifted", op["id"])

        return tick

    # --- touch (the Touch action) --------------------------------------------------------------
    def touch_single(self, start: np.ndarray, direction: np.ndarray, max_travel_m: float, speed_m_s: float,
                     guard: int = names.GUARD_TIP_LAG, prior_distance_m: float = 0.0) -> dict:
        """One guarded touch from `start` along `direction` (plan_touch travels there first). Returns tripped,
        the contact tcp 4x4 (FK of the trip joints, else of the joints at the end), q, and the phase, travel
        and speed. A probe touch (GUARD_PROBE) needs the expected contact, prior_distance_m along direction:
        its travel past it is capped (tatbot_motion.probe_travel_m), so reaching the cap untriggered is a
        missing trigger, not a fault. After a trigger, or at the cap, it backs out the way it came, and the probe
        must then read released."""
        from tatbot_motion import plan_touch, probe_travel_m

        probe = guard == names.GUARD_PROBE
        travel = max_travel_m or float(self.motion["touch"]["max_travel_m"])
        if probe:
            if not prior_distance_m > 0.0:
                raise ValueError("a probe touch needs prior_distance_m: the expected contact along the direction")
            travel = min(travel, probe_travel_m(direction, prior_distance_m, self.motion))
            speed_m_s = speed_m_s or float(self.motion["probe"]["slow_m_s"])
        via = self.station_via(start) if probe else []
        traj = self.knots(plan_touch(q_seed=self.io.seed(), start_base_from_tcp=start, direction=direction,
                                     max_travel_m=travel, kin=self.kin, motion=self.motion,
                                     prior_distance_m=prior_distance_m or None, speed_m_s=speed_m_s or None,
                                     via=via))
        status, text, q_trip = self.guarded(traj, guard)
        motion_result(status, text, 'touch', ('ok', 'tripped'))
        q = self.io.measured() if q_trip is None else q_trip
        out = {"tripped": status == "tripped", "contact": self.kin.fk(q), "q": q, "travel_m": travel,
               "speed_m_s": speed_m_s, "phase": int(traj.phase[self.touch_stop])}
        if probe:   # tripped or at its cap, the tool never rests against the stylus: a contact too shallow to trip
            # there read triggered seconds after the pass ended, and every later goal was refused (2026-09-28)
            out["back_off"] = self.back_off(traj, self.touch_stop, direction)
        return out

    def move_single(self, target: np.ndarray, guard: int = names.GUARD_NONE, view: bool = False) -> dict:
        """Travel to `target` and hold there (Touch MODE_MOVE: a view of the station, a calibration hold).
        Under GUARD_PROBE it takes a probe touch's way (station_via) and the guard is armed throughout: a trip
        is an error. Unguarded, a travel the Cartesian planner refuses (from rest on the joint limits, or past
        one on the way) is a joint-space move instead (joint_travel). `view` (Touch MODE_VIEW): one joint-space
        move (ready.view_move), whose way the caller has checked clear of the station; refused, as station_via's
        joint move is, when it dips more than STATION_DIP_M under its lower end."""
        from tatbot_motion import plan_travel
        from tatbot_motion.clik import PlanError

        probe = guard == names.GUARD_PROBE
        if view:
            q = self.io.seed()
            traj = ready.view_move(self.kin, self.motion, q, target, self.knot_rate, P.PHASE_TRAVEL)
            low = ready.station_dip(self.kin, traj, q, traj.q[-1])
            if low is not None:
                raise RuntimeError(f"the joint move to the view would take the tip down to z {low:.3f} m; not moving")
        else:
            via = self.station_via(target) if probe else []
            try:
                traj = self.knots(plan_travel(q_seed=self.io.seed(), base_from_tcp=target, kin=self.kin,
                                              motion=self.motion, via=via))
            except PlanError as refused:
                if probe:
                    raise
                traj = self.joint_travel(target, refused)
        if probe:
            status, text, _ = self.guarded(traj, guard)
        else:
            status, _, text = self.io.execute(traj, should_stop=self.stopping)
        motion_result(status, text, 'move')
        q = self.io.measured()
        return {"tripped": False, "contact": self.kin.fk(q), "q": q, "travel_m": 0.0, "speed_m_s": 0.0,
                "phase": int(P.PHASE_TRAVEL)}

    def joint_travel(self, target: np.ndarray, refused: Exception):
        """A joint-space move to `target` for a travel the Cartesian planner refused, kept clear of the page like
        the ready move: its planned tip may not come within half the approach standoff of the page plane (the
        page in use, else the newest one the cameras or the configuration give); with no page known it is
        refused, as is any dip, before the arm moves."""
        q = self.io.seed()
        traj = ready.pen_up_joint_move(self.kin, self.motion, q, ready.solve_ik_seeded(self.kin, target, q), self.knot_rate, P.PHASE_TRAVEL)
        if self.camera is None:
            newest = self.node.camera_page(self.arm)
            if newest is None or newest[0] is None:
                raise RuntimeError(f"move: {refused}; no page is known to keep a joint move clear of; not moving")
            page = np.asarray(newest[0], float)
        else:
            page = self.used()
        low = self.lowest_over_page(traj, page)
        floor = 0.5 * float(self.motion["approach"]["standoff_m"])
        if low < floor:
            raise RuntimeError(f"move: {refused}; a joint move instead would take the pen {low * 1e3:+.1f} mm over "
                               f"the page (under {floor * 1e3:.0f} mm); not moving")
        self.event(Event.KIND_REPLAN, f"move: {refused}; a joint move instead")
        return traj

    def station_via(self, start: np.ndarray) -> list:
        """A probe touch's way to its start: up to the clearance plane (motion.yaml probe.clearance_m over the
        start, or where the tip already is when higher), across it, and down, never a straight line past the
        ball. From rest on the joint limits, where the Cartesian planner cannot start, a joint-space move over
        the start comes first, under the probe guard; its path may not dip 10 mm under its lower end. The way is
        ready.station_way's, which the station calibration plans with too."""
        q = self.io.seed()
        q_over, via = ready.station_way(self.kin, self.motion, q, start)
        if q_over is None:
            return via
        traj = ready.pen_up_joint_move(self.kin, self.motion, q, q_over, self.knot_rate, P.PHASE_TRAVEL)
        low = ready.station_dip(self.kin, traj, q, q_over)
        if low is not None:
            raise RuntimeError(f"the joint move over the start would take the tip down to z {low:.3f} m; not moving")
        status, text, _ = self.guarded(traj, names.GUARD_PROBE)
        motion_result(status, text, 'the joint move over the start')
        return []   # straight over the start: plan_touch's own travel goes down onto it

    def back_off(self, traj, index: int, direction: np.ndarray) -> dict:
        """After a probe trigger: back out motion.yaml probe.back_off_m the way the tip came (toward where
        the plan had it that far before the stop: -direction on the slow leg), unguarded, then wait up to
        0.5 s for the probe to read released."""
        distance = float(self.motion["probe"]["back_off_m"])
        tip = np.asarray(traj.tip)[:index + 1]
        behind = np.flatnonzero(np.linalg.norm(tip - tip[-1], axis=1) >= distance)
        back = tip[behind[-1]] - tip[-1] if len(behind) else -np.asarray(direction, float)
        back = back / np.linalg.norm(back)
        status, _, text = self.io.execute(self.lift_plan(distance, back), should_stop=self.stopping)
        motion_result(status, text, 'probe back-off')
        if not self.io.wait(lambda: not self.io.flag("probe_triggered"), 0.5):
            raise RuntimeError(f"the probe still reads triggered {distance * 1000:.1f} mm back from the touch "
                               "(a stuck stylus, or a broken wire)")
        return {"direction": back.tolist(), "distance_m": distance}
