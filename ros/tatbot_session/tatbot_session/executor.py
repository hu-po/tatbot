"""One arm's executor: page setup and touches, the stroke loop, tool changes and palette dips, pauses, latches
and decisions, landing.

The next op is always the first op whose last ledger event is not done/skipped. Each stroke is planned by
tatbot_motion from the measured joints (the next stroke is pre-planned while the current one runs, then
checked for drift at dispatch) and sent as one FollowJointTrajectory goal. A riding pen's tattoo machine
runs only inside the op loop: on before each stroke goal, off whenever the loop waits for a decision, dips
or ends, and at once when the arm latches (the node). A tool change that needs a different cartridge lands
the arm for the operator; resuming the run is the acknowledgement. The operator's pen trim (the numpad) is
added to every stroke goal as it is sent, and replaces the running one ahead of now when it changes.
ros/README.md 4.3, 4.4, 6, 8.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tatbot_description import names
from tatbot_interfaces.action import Draw
from tatbot_interfaces.msg import Event

from tatbot_session import config, geometry, machine, ready, rules, track
from tatbot_session import ledger as ledger_mod
from tatbot_session.lease import in_cap, in_cap_refusal

P = Draw.Feedback


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


class PageMovedError(RuntimeError):
    """The camera page pose left the one this run located and touched; the run stops rather than adopt it."""


class Ask:
    """One Decide call handed to a waiting executor. It answers only the wait it was made for: an Ask
    left over from an earlier wait, or one its caller gave up on, is never applied."""

    def __init__(self, decision: int, wait_id: int):
        self.decision, self.wait_id = decision, wait_id
        self.reply: queue.Queue = queue.Queue()
        self.taken = self.abandoned = False


class ArmExecutor:
    def __init__(self, node, io, kin, motion: dict, stack: dict, repo):
        self.node, self.io, self.kin, self.motion, self.stack, self.repo = node, io, kin, motion, stack, repo
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
        self._chain_snap = None   # the print-correction field a pen-down chain is planned with (track.Field)
        self._correction = None   # (mean, largest) correction of the op last dispatched, m
        self._goal_touch = False   # the draw goal's touch flag, for a page set up again mid-draw
        self.tool_id = ""
        self.machine = None   # the tattoo machine's switch (tatbot_session.machine), set by the node; None: none
        self.pen = None       # the draw's tatbot_motion.PenDown; None: motion.yaml's press
        self.camera: np.ndarray | None = None
        self.correction = np.eye(4)
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
        self.ledger = None                  # the running draw's, where trims are written
        self._planner = ThreadPoolExecutor(max_workers=1)
        self._preplan = None
        self.knot_rate = float(stack["session"]["knot_rate_hz"])

    # --- plumbing ------------------------------------------------------------------------------
    def event(self, kind: str, text: str = "", op_id: str = "") -> None:
        self.node.event(kind, arm=self.arm, op_id=op_id, text=text)

    def stopping(self) -> bool:
        return self.land_requested or self.cancelled()

    def used(self) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("no page pose known")
        trim = self.stack["page"].get("trim", {}).get(self.arm, [0.0, 0.0])
        return geometry.page_used(self.camera, self.correction, trim)

    def tcp(self, q=None) -> np.ndarray:
        return self.kin.fk(self.io.measured() if q is None else q)

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
            self.goal_active, self.land_requested = True, False
            return True

    def end_goal(self) -> None:
        """Release the arm; a LAND accepted while the goal ran but not yet carried out lands now."""
        try:
            if self.land_requested:
                ok, text = self._land_now()
                self.node.event(Event.KIND_LANDED if ok else Event.KIND_ERROR, arm=self.arm, text=text)
        finally:
            with self._lock:
                self.goal_active = self.land_requested = False
                self.busy.release()

    def decide(self, decision: int, timeout: float = 15.0) -> tuple[bool, str]:
        """Called from the Decide service thread. Hands the decision to the waiting executor, or acts
        directly on an idle arm (CONTINUE unlatches, LAND lands)."""
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
            return self._land_now()
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
                 "released": time.monotonic() if self.io.estop_ok else None}
        try:
            while True:
                if self.cancelled():
                    raise Cancelled()
                if self.land_requested or self._resume_expired(watch, op_id):
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
        """Track the latch and its release; True once a released latch waited past resume_timeout_s."""
        if watch["q"] is None and self.io.latched:
            watch["q"] = self.io.measured()
        if watch["q"] is None:
            return False
        if not self.io.estop_ok:
            watch["released"] = None
        elif watch["released"] is None:
            watch["released"] = time.monotonic()
        timeout_s = float(self.stack["session"]["resume_timeout_s"])
        if rules.resume_expired(watch["released"], time.monotonic(), timeout_s):
            self.event(Event.KIND_DECISION, f"no decision {timeout_s:.0f} s after release: landing", op_id)
            return True
        return False

    def _refusal(self, kind: str, decision: int, q_latched) -> str | None:
        """Why a decision is refused now, or None; an accepted CONTINUE on a latched arm has unlatched."""
        why = rules.check(kind, decision)
        if why is None and decision == rules.CONTINUE and (q_latched is not None or self.io.latched):
            why = self.continue_refusal(q_latched if q_latched is not None else self.io.measured())
            if why is None:
                ok, text = self.io.unlatch()
                why = None if ok else text
        return why

    def machine_off(self) -> None:
        """Ask for the tattoo machine off (the module docstring says when). The Pi powers it off by itself on the
        e-stop and when the session goes silent."""
        if self.machine is not None and self.machine.on:
            self.machine.set(False)
            self.event(Event.KIND_MACHINE, "machine off")

    def machine_running(self, op_id: str, index: int, ledger) -> None:
        """The tattoo machine on, as its switch reports it, before a riding stroke goal. One not reported running
        within machine.confirm_s pauses the run: continue asks again, land lands."""
        was = self.machine.on
        while not self.machine.wait(True, self._confirm_s()):
            why = self.machine.why_off()
            self.machine.set(False)
            self.event(Event.KIND_PAUSE, f"the tattoo machine did not start: {why}; continue asks again, land lands",
                       op_id)
            if self.wait_decision("pause", op_id, index, ledger) == rules.LAND:
                self.land()
            was = False
        if not was:
            self.event(Event.KIND_MACHINE, "machine on", op_id)

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
        self.event(Event.KIND_LANDED if ok else Event.KIND_ERROR, text)
        raise Landed(why if ok and why else text)

    def _land_now(self) -> tuple[bool, str]:
        """Lift clear of the page first when the pen may be on it (the driver's staged move would drag it
        sideways), then the driver's landing."""
        why = in_cap_refusal(self.arm)
        if why:
            with self._lock:
                self.land_requested = False
            return False, why
        self._lift_clear()
        with self._lock:
            self.land_requested = False
        return self.io.land(self.stack["safety"]["landing"]["budget_s"], staged=getattr(self.io, "staged", None))

    def _height(self, q=None, page=None) -> float | None:
        """Tip height over the page in use (m), or None before a page is known."""
        if self.camera is None:
            return None
        page = self.used() if page is None else page
        return float(page[:3, 2] @ (self.tcp(q)[:3, 3] - page[:3, 3]))

    def lift_plan(self, distance, direction=None):
        from tatbot_motion import plan_lift

        return self.knots(plan_lift(q_seed=self.io.seed(), direction=self.used()[:3, 2] if direction is None else direction,
                                   distance_m=distance, kin=self.kin, motion=self.motion))

    def _lift_clear(self) -> None:
        """Before landing: a tip within the standoff of the page lifts along the normal, unlatching with the
        hold goal when the e-stop is released. Best effort; landing follows whatever happens."""
        standoff = float(self.motion["approach"]["standoff_m"])
        try:
            height = self._height()
            if height is None or height >= standoff - 0.001 or not self.io.estop_ok:
                return
            if self.io.latched and not self.io.unlatch()[0]:
                return
            traj = self.lift_plan(standoff - max(height, 0.0))
            self.feedback(phase=P.PHASE_LIFT)
            status, _, text = self.io.execute(traj)
            if status == "ok":
                self.event(Event.KIND_DECISION, f"lifted {(standoff - max(height, 0.0)) * 1000:.1f} mm clear before landing")
            else:
                self.event(Event.KIND_ERROR, f"lift before landing: {text}")
        except Exception as exc:  # noqa: BLE001 - landing goes ahead from wherever the arm holds
            self.event(Event.KIND_ERROR, f"lift before landing: {exc}")

    # --- moves ---------------------------------------------------------------------------------
    def move(self, make, phase: int, what: str, ledger=None) -> None:
        """Run a non-stroke motion (travel, lift); after a latch or failure wait for a decision and
        re-plan it from the measured joints on CONTINUE."""
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
            if self.wait_decision("latched", ledger=ledger) == rules.LAND:
                self.land()

    def lift(self, what: str, ledger=None) -> None:
        """Straight up along the page normal to the standoff (after a touch, before leaving the page)."""
        self.move(lambda: self.lift_plan(float(self.motion["touch"]["lift_m"])), P.PHASE_LIFT, what, ledger)

    HEADING_TIE_RAD = 0.05   # headings whose joint-limit margins are this close count as tied (the first is taken)
    BRANCH_RAD = 0.1         # a ready pose this far from the page's recorded one, on any joint, is another branch

    def _page_ready(self) -> dict | None:
        """The page's ready pose ({"q", "heading"}): recorded by this run's first ready move, else carried with
        the paper from the last run on the same page (_carried_paper)."""
        return self.page_record.get("ready") or (self.page_record.get("carried") or {}).get("ready")

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

    def pen_heading(self) -> np.ndarray:
        """A tcp pose whose x axis is the pen heading (its turn about its own axis) that keeps every joint
        farthest from its limits over the clear centre. The ballpoint is round, so the heading is free,
        and every later plan keeps the ready pose's heading: the one the arm arrived with put joint_3
        0.04 rad from its limit over a page near the base line (2026-09-26) where
        another left 0.6 rad. The page's recorded heading when it has one (_page_ready: this run's, else the one
        carried from the last run on this page); else the search from _ready_seed, the first of the headings
        tied (HEADING_TIE_RAD) with the best. The margins of several headings differ by less than a page height's
        few millimetres move them (45 and 60 deg trade places over 8 mm), and a bare best flipped between
        headings 23-49 deg apart from one resume to the next: only a new page searches."""
        recorded = (self._page_ready() or {}).get("heading")
        if recorded is not None:
            return np.asarray(recorded, float)
        page = self.used()
        hx, hy = (0.45 * v for v in self.stack["page"]["clear_m"])
        points = [(0.0, 0.0), (-hx, -hy), (hx, -hy), (-hx, hy), (hx, hy)]
        standoff = float(self.motion["approach"]["standoff_m"])
        q_seed = self._ready_seed()
        lower, upper = np.asarray(self.kin.lower)[:6], np.asarray(self.kin.upper)[:6]
        scored = []
        for yaw in np.radians(np.arange(0.0, 360.0, 15.0)):
            heading = np.eye(4)
            heading[:3, 0] = np.cos(yaw) * page[:3, 0] + np.sin(yaw) * page[:3, 1]
            margin = np.inf
            for xy in points:
                try:
                    q = ready.solve_ik_seeded(self.kin, geometry.tool_down_pose(page, xy, standoff, heading), q_seed)
                except ValueError:
                    margin = -np.inf
                    break
                margin = min(margin, float(np.min(np.minimum(q[:6] - lower, upper - q[:6]))))
            scored.append((margin, heading))
        best = max(margin for margin, _ in scored)
        if not np.isfinite(best):
            return self.tcp()
        return next(heading for margin, heading in scored if margin >= best - self.HEADING_TIE_RAD)

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
        height = self._height()
        if height is not None and height < standoff - 0.001:
            self.lift("lift before clearing the cameras' view", ledger)

        def make():
            return ready.pen_up_joint_move(self.kin, self.motion, self.io.seed(), rest, self.knot_rate, P.PHASE_TRAVEL)

        if self.camera is not None and self.lowest_over_page(make()) < 0.5 * standoff:
            self.event(Event.KIND_PAGE, "the move to rest would pass the pen within {:.0f} mm of the page; "
                       "not moving".format(0.5 * standoff * 1e3))
            return
        self.move(make, P.PHASE_TRAVEL, "rest, clearing the cameras' view of the page", ledger)

    def lowest_over_page(self, traj, page=None) -> float:
        """The planned tip's lowest height over the page in use along its normal (m); +inf with no tip."""
        tip = np.asarray(getattr(traj, "tip", ()), float).reshape(-1, 3)
        tip = tip[np.isfinite(tip).all(axis=1)]
        if not len(tip):
            return float("inf")
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
        h = self._height()
        if h is not None and h < standoff - 0.001:
            self.lift("lift before the ready move", ledger)
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
    def wrist_locate(self, prior: np.ndarray, cfg: dict, ledger=None, program=None) -> tuple[np.ndarray, dict]:
        """Fuse overhead and wrist border XY/yaw, recording page depth and replayable captures.
        Each reachable clear view fits print inner edges, scanned from outside the program's drawing, and
        samples native depth; see ros/README.md.
        """
        import cv2

        from tatbot_session import inspect as ins
        from tatbot_session.ready import joint_move, solve_ik

        page = self.stack["page"]
        clear, frame, ppm = page["clear_m"], ins.camera_frame(self.arm), 5.0
        extent = ins.page_extent(page["size_m"])
        inner = ins.inner_edges(clear, page.get("inner_edges_m"))   # a malformed key refuses before the arm moves
        drawn = ins.drawing_extent(program)   # ink there, from this run or a redraw, is not the border
        artwork = ins.print_artwork(page.get("pattern_id"))   # a view whose edges do not fit matches the print
        aim_xy = (0.0, 0.0)   # the views aim at the page centre; the border fit's scale is about it
        max_y = float(prior[1, 3]) - float(cfg.get("side_margin_m", 0.0))
        speed = float(self.motion["joint_speed"]["pen_up_rad_s"]) * 0.3
        from tatbot_cli import ros_depth

        cam = ins.WristCamera(serial=str(cfg.get("serial", "")), depth=True)
        sols, views, heights, best = [], [], [], None
        try:
            guard = getattr(self.node, "guard", None)
            self_gap = None if guard is None else (lambda q: guard.self_gap(self.arm, q))
            for n in range(int(cfg.get("poses", 3))):
                q = self.io.measured()
                side = 1 if n % 2 else -1
                azimuths = None if best is None else [np.radians(best["azimuth_deg"] + side * d) for d in (15, 30, 45)]
                aimed = ins.aim(self.kin, solve_ik, prior, aim_xy, q, frame=frame,
                                clearance_m=float(cfg.get("clearance_m", 0.015)), azimuths=azimuths, max_y=max_y,
                                self_gap=self_gap)
                if aimed is None:
                    continue
                q_t, view = aimed

                def to_view(q_t=q_t):
                    return joint_move(self.kin.joint_names, self.io.seed(), q_t, max_rad_s=speed, max_m_s=0.002,
                                      rate_hz=self.knot_rate, kin=self.kin, phase=P.PHASE_TRAVEL)

                # a view pose keeps the pen clear of the page, the joint path to it need not (go_ready)
                low = self.lowest_over_page(to_view())
                if low < 0.5 * float(cfg.get("clearance_m", 0.015)):
                    self.event(Event.KIND_REPLAN, f"locate view {n} skipped: the move would take the pen "
                                                  f"{low * 1e3:+.1f} mm over the page")
                    continue
                best = best or view
                self.move(to_view, P.PHASE_TRAVEL, f"locate view {n}", ledger)
                time.sleep(float(cfg.get("settle_s", 0.8)))
                frames, depths = cam.grab_both(int(cfg.get("frames", 5)), observe=lambda: ros_depth.observe(self.io))
                q_view = self.io.measured()
                base_from_cam = self.kin.frame(q_view, frame)
                capture = f"depth{n}-{time.time_ns()}"
                out = None if self.run_dir is None else self.run_dir / "locate" / capture
                height, depth_observation, clearance = ros_depth.retain(cam, frames, depths, self.kin, q_view,
                                                                       prior, out, self.arm)
                if height is not None:
                    heights.append(height["offset_m"])
                image = ins.median_stack([ins.rectify(f, prior, base_from_cam, cam.k, cam.dist, extent, ppm)
                                          for f in frames])
                sol = ins.fit_view(image, extent, ppm, page, inner_m=inner, drawn_m=drawn, centre_m=aim_xy,
                                   free_scale=tuple(cfg.get("free_scale", (True, True))), artwork=artwork)
                sols.append(sol)
                views.append({**view, "fit": sol, "depth": height, "depth_observation": depth_observation, "clearance": clearance})
                if self.run_dir is not None:
                    (self.run_dir / "locate").mkdir(exist_ok=True)
                    cv2.imwrite(str(self.run_dir / "locate" / f"view{n}.png"), image)
                    cv2.imwrite(str(self.run_dir / "locate" / f"frame{n}.png"), frames[0])
                    views[-1]["depth_capture"] = f"locate/{capture}.npz"
        finally:
            cam.close()
        wrist = ins.combine_views(sols, floor=(float(cfg["wrist_floor_m"]), float(cfg["wrist_floor_m"]),
                                               float(cfg["wrist_floor_rad"])))
        record = {"overhead": prior.tolist(), "wrist": wrist, "views": views, "inner_edges_m": inner}
        if not heights:
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
            self.event(Event.KIND_PAGE, "wrist locate: no border fit; the overhead pose stands")
            return prior, record
        sigma_o = (float(cfg["overhead_sigma_m"]), float(cfg["overhead_sigma_m"]), float(cfg["overhead_sigma_rad"]))
        fused = ins.fuse([((0.0, 0.0, 0.0), sigma_o),
                          ((wrist["tx_m"], wrist["ty_m"], wrist["theta_rad"]), wrist["sigma"])])
        record["fused"] = fused
        pose = prior @ np.linalg.inv(ins.correction_matrix(fused["tx_m"], fused["ty_m"], fused["theta_rad"]))
        matched = sum(s is not None and s["sides"] == ["artwork"] for s in sols)
        self.event(Event.KIND_PAGE, "wrist locate: print {:+.1f} {:+.1f} mm {:+.2f} deg from the overhead pose "
                   "(sigma {:.1f} {:.1f} mm, {} views, {} by the artwork); fused {:+.1f} {:+.1f} mm {:+.2f} deg".format(
                       -wrist["tx_m"] * 1e3, -wrist["ty_m"] * 1e3, -np.degrees(wrist["theta_rad"]),
                       wrist["sigma"][0] * 1e3, wrist["sigma"][1] * 1e3, wrist["poses"], matched,
                       -fused["tx_m"] * 1e3, -fused["ty_m"] * 1e3, -np.degrees(fused["theta_rad"])))
        return pose, record

    PAGE_FIRST_WAIT_S = 10.0   # a page message after a stack restart; page.wait_s extends it for a lost print

    def setup_page(self, program: dict | None, touch: bool, ledger=None) -> None:
        max_lost_s = float(self.stack["page"].get("max_lost_s", 5.0))
        wait_s = float(self.stack["page"].get("wait_s", self.PAGE_FIRST_WAIT_S))
        camera = self.node.wait_camera_page(self.arm, timeout=min(wait_s, self.PAGE_FIRST_WAIT_S), max_lost_s=max_lost_s)
        stale = "no page pose yet" if camera is None else geometry.stale_page(camera[1], max_lost_s)
        if stale and wait_s > self.PAGE_FIRST_WAIT_S:
            # The arm itself hides a print through a draw and a held inspection; the tracker decodes it again
            # once uncovered. Waiting for that measurement keeps the freshness rule; it never accepts a stale pose.
            self.event(Event.KIND_PAGE, f"{stale}: waiting up to {wait_s - self.PAGE_FIRST_WAIT_S:.0f} s more for "
                       "the cameras to measure it again")
            self.clear_view(ledger)
            camera = self.node.wait_camera_page(self.arm, timeout=wait_s - self.PAGE_FIRST_WAIT_S,
                                                max_lost_s=max_lost_s, should_stop=self.stopping)
            if self.stopping():
                raise Cancelled()
        if camera is None:
            raise RuntimeError(f"no page pose for {self.arm} (page.source {self.stack['page']['source']})")
        stale = geometry.stale_page(camera[1], max_lost_s)
        if stale:
            raise RuntimeError(f"{stale}; give the overhead cameras a clear view of the page, then draw again")
        cam, info = camera
        self.camera = cam
        locate = self.stack["page"].get("locate") or {}
        if locate.get("wrist") and self.stack.get("hardware") == "real":   # the camera rides the real arm
            cam, located = self.wrist_locate(cam, locate, ledger, program)
            self.camera = cam
            info = {**info, "locate": located}
        cfg = self.stack["page"].get("height") or {}
        warn_m = float(cfg.get("warn_m", 0.002))
        sources = [("overhead", 0.0, float(cfg.get("overhead_sigma_m", 0.002)))]
        depth = (info.get("locate") or {}).get("depth")
        if depth is not None:
            sources.append(("wrist depth", float(depth["offset_m"]), float(depth["sigma_m"])))
        carried = self._page_prior(cam, info, sources, warn_m)
        self.correction = self.prior_correction
        self.page_record = {"camera_at_setup": cam.tolist(), **info, "touches": [], "replans": [],
                            "prior_correction": self.prior_correction.tolist(), "carried": carried}
        self.go_ready(ledger)
        contacts, prior = [], cam @ self.prior_correction   # the page the gauge holds are made on
        if touch:
            contacts = self.page_touches(self._touch_layout(program), ledger, known=carried is not None)
        n, o = cam[:3, 2], cam[:3, 3]
        normal, d = self._page_plane(contacts, cam, sources, cfg, warn_m)
        self._compare_planes(contacts, n, o)
        touched = self._hold_page(geometry.touched_page(cam, normal, d), prior)
        self.correction = np.linalg.inv(cam) @ touched
        self.prior_correction = self.correction
        self.page_record.update(plane=[*normal.tolist(), d], touched=touched.tolist())
        self.page_record["used"] = self.used().tolist()
        self.page_record["correction"] = self.correction.tolist()
        self._carry_paper(cam, info)
        self._reference_writing_height(program)

    def _gauge_ready(self) -> bool:
        from tatbot_session import gauge

        return (self._gauge_mode() == "use" and self.stack.get("hardware") == "real"
                and gauge.calibration_path(self.arm).exists())

    def _touch_layout(self, program) -> np.ndarray:
        """Where the page is measured. Touches stay under a small drawing (geometry.touch_layout: a touch leaves
        a dot). The wrist gauge leaves none, so it takes the corners of a square touch.spread_m about the drawing's
        centre, and the centre, inside the print's clear centre less 5 mm: about a ladder 8 mm wide the touches'
        triangle was 8 mm across, too short a base for the page's tilt, and five points name a bad one
        (_gauge_plane) where four cannot."""
        spread = float(self.stack.get("touch", {}).get("spread_m", 0.012))
        if not self._gauge_ready():
            return geometry.touch_layout(program, spread)
        box = geometry.program_box(program)
        centre = np.zeros(2) if box is None else box[0]
        hx, hy = (v / 2 - 0.005 for v in self.stack["page"]["clear_m"])
        corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]]) * spread + centre
        return np.clip(corners, [-hx, -hy], [hx, hy])

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
        residual = pts @ normal + d
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

    def _page_prior(self, cam, info, sources, warn_m) -> dict | None:
        """Where the touches start: the paper the last accepted touches on this page found (returned), else the
        cameras' fused height. The cameras are no witness of the paper's height: on 2026-10-03 the overhead
        depth plane stood 8-10 mm and the wrist depth up to 15 mm over the paper the arm touched (1-3 mm over
        its base, as the 2026-09-29 touch found it), so a search from them is long and meets the guard's
        trips in the air first."""
        from tatbot_session import inspect as ins

        carried = self._carried_paper(cam, info)
        if carried is not None or len(sources) > 1:
            self.prior_correction = np.eye(4)
            self.prior_correction[2, 3] = (carried["offset_m"] if carried is not None
                                           else ins.fuse_heights(sources, warn_m=warn_m)["offset_m"])
        if carried is not None:
            self.event(Event.KIND_PAGE, f"the paper as run {carried['run']} touched it, {carried['offset_m'] * 1e3:+.1f} mm "
                       f"over the overhead page, {carried['age_s'] / 60:.0f} min ago")
        return carried

    def _carry_path(self):
        from pathlib import Path

        return None if self.run_dir is None else Path(self.run_dir).parents[1] / f"page-touched-{self.arm}.json"

    def _carried_paper(self, cam, info) -> dict | None:
        """The paper the last accepted touches found, while it is the same page: the same print, the cameras'
        page within page.carry_move_m of where it was then, no older than page.carry_max_s. Returned as an
        offset along the camera page's normal; None otherwise."""
        page, path = self.stack["page"], self._carry_path()
        record = self.carried
        if record is None and path is not None and path.exists():
            record = json.loads(path.read_text())
        if not record or record.get("arm") != self.arm or record.get("pattern_id") != info.get("pattern_id"):
            return None
        then, age = np.asarray(record["camera"], float), time.time() - float(record["utc_s"])
        if age > float(page.get("carry_max_s", 43200.0)) or \
                np.linalg.norm(then[:3, 3] - cam[:3, 3]) > float(page.get("carry_move_m", 0.015)):
            return None
        n, d = np.asarray(record["plane"][:3], float), float(record["plane"][3])
        return {"run": record.get("run", ""), "age_s": age, "offset_m": -(float(n @ cam[:3, 3]) + d) / float(n @ cam[:3, 2]),
                "ready": record.get("ready")}

    def _carry_paper(self, cam, info) -> None:
        """Keeps where this run's accepted touches tripped (the trips' own heights, not the first-contact
        estimates: the next run judges trips against it) for the next run on the same page, with the ready pose
        they were measured from: the next run takes the same IK branch, so the height it starts from is its own."""
        n = cam[:3, 2]
        trips = [float(np.asarray(t["tcp"])[:3, 3] @ n) for t in self.page_record["touches"]]
        if not trips:   # no touches this run: the paper is where it was
            return
        self.carried = {"arm": self.arm, "pattern_id": info.get("pattern_id"), "camera": cam.tolist(),
                        "plane": [*n.tolist(), -float(np.mean(trips))], "utc_s": time.time(),
                        "run": self.run_dir.name if self.run_dir is not None else "",
                        "ready": self.page_record.get("ready")}
        path = self._carry_path()
        if path is not None:
            path.write_text(json.dumps(self.carried, indent=1) + "\n")

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
        """Every run measures the wrist depth plane against the touches, and the three planes' tilts against each
        other, so the wrist bias in stack.yaml and the choice of tilt rest on numbers (page_record["planes"])."""
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

    def refresh_page(self, *, retract=True) -> None:
        """Stop before dispatch when the camera page moved beyond the touched run's thresholds; never adopt it."""
        if self.stack["page"]["source"] == "fixed" or self.camera is None:
            return
        newest = self.node.camera_page(self.arm)
        if newest is None:
            return
        cam = self.camera
        page = self.stack["page"]
        if not geometry.page_moved(cam, newest[0], page["replan_translation_m"], page["replan_rotation_rad"]):
            return
        shift = float(np.linalg.norm(newest[0][:3, 3] - cam[:3, 3]))
        turn = float(np.degrees(geometry.rotation_angle(cam, newest[0])))
        self.page_record.setdefault("replans", []).append({"t": time.time(), "camera": newest[0].tolist(),
                                                           "shift_m": shift, "turn_deg": turn, "adopted": False})
        self._preplan = None
        text = (f"page moved {shift * 1000:.1f} mm, {turn:.1f} deg from the pose this run touched: not adopted, "
                "stopping; check the sheet, then draw again")
        self.event(Event.KIND_ERROR, text)
        height = self._height()
        if retract and height is not None and height < float(self.motion["approach"]["standoff_m"]) - 0.001:
            self.lift("lift: the page moved")
        raise PageMovedError(text)

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
        heading, contacts = self.tcp(), []   # every touch's heading: the turns never add up
        paper = {"known": known, "level": 0.0}   # the paper's trip height over the prior page, once known
        for i, xy in enumerate(points_xy):
            trips, seen = [], []   # the trips that count, and every trip, as tip heights over the prior page (m)
            gauged = self._gauge_point(i, xy, paper, above, heading, ledger)
            if gauged is not None:   # page.height.gauge use: the gauge is the touch ([]: no steady reading there)
                contacts += gauged
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

    def _wrist_frames(self, name: str, q, count: int) -> tuple[list, list, dict] | None:
        """`count` colour and raw depth frames from the wrist D405, the first saved as touches/<name>.png and .npz
        with the joints q; None off real hardware, without the wrist camera, or when the capture fails (reported)."""
        import cv2

        from tatbot_session import inspect as ins

        locate = self.stack["page"].get("locate") or {}
        if self.run_dir is None or self.stack.get("hardware") != "real" or not locate.get("wrist"):
            return None
        try:
            cam = ins.WristCamera(serial=str(locate.get("serial", "")), depth=True)
            try:
                colors, _ = cam.grab_both(count)
            finally:
                cam.close()
        except RuntimeError as error:
            self.event(Event.KIND_ERROR, f"wrist frames {name}: {error}")
            return None
        out = self.run_dir / "touches"
        out.mkdir(exist_ok=True)
        cv2.imwrite(str(out / f"{name}.png"), colors[0])
        np.savez_compressed(out / f"{name}.npz", depth_raw=cam.raw_depth[0], q=np.asarray(q, float),
                            meta=json.dumps(cam.depth_metadata))
        return colors, list(cam.raw_depth), cam.depth_metadata

    def _gauge_mode(self) -> str:
        return str((self.stack["page"].get("height") or {}).get("gauge", "off"))

    def _gauge_point(self, i: int, xy, paper: dict, above: float, heading, ledger=None) -> list | None:
        """page.height.gauge check or use, with the wrist gauge fitted for the fitted tool (gauge.py): the pen held
        where touch i starts, the paper's point under it by the wrist camera, recorded in page_record["gauge"]; for
        use also as the touch itself, returned in a list ([] when no reading held still). A reading whose frames
        disagree by more than page.height.gauge_spread_m (one put the paper 4 mm high, its frames 1.6 mm apart,
        2026-10-03) is taken again 4 mm to either side. None for check, or without the gauge."""
        from tatbot_session import gauge

        path = gauge.calibration_path(self.arm)
        if self._gauge_mode() == "off" or self.stack.get("hardware") != "real" or not path.exists():
            return None
        cal = json.loads(path.read_text())
        if cal.get("tool") and cal["tool"] != config.fitted_tool(self.repo, self.arm):
            self.event(Event.KIND_PAGE, f"touch {i}: the wrist gauge was fitted for {cal['tool']}: not used")
            return None
        prior = self.camera @ self.prior_correction
        hover = paper["level"] + float(self.motion["touch"].get("known_above_m", 0.004)) if paper["known"] else above
        limit = float((self.stack["page"].get("height") or {}).get("gauge_spread_m", 0.0005))
        for step in (0.0, 0.004, -0.004):
            at = np.asarray(xy, float) + [step, 0.0]
            read = self._gauge_hold(i, at, hover, prior, heading, cal, ledger)
            if read is not None and read["spread_m"] <= limit:
                break
            if read is not None:
                self.event(Event.KIND_PAGE, f"touch {i}: the wrist gauge's frames disagree by "
                           f"{read['spread_m'] * 1e3:.2f} mm at ({at[0] * 1e3:+.1f} {at[1] * 1e3:+.1f}) mm: not used")
        else:
            return [] if self._gauge_mode() == "use" else None
        if self._gauge_mode() != "use":
            return None
        pose = np.asarray(read["tcp"], float).copy()
        pose[:3, 3] = read["point"]
        self.page_record["touches"].append({"xy": at.tolist(), "tcp": pose.tolist(), "q": read["q"],
                                            "contact": read["point"], "method": "wrist gauge"})
        return [np.asarray(read["point"], float)]

    def _gauge_hold(self, i: int, xy, hover: float, prior, heading, cal: dict, ledger=None) -> dict | None:
        """The pen held hover over page point xy, its wrist frames' gauge heights: the paper's point under the tip
        (their median), recorded in page_record["gauge"]; None when the camera makes out no pen and paper."""
        from tatbot_motion import plan_travel

        from tatbot_session import gauge

        start = geometry.tool_down_pose(prior, xy, hover, heading)
        self.move(lambda: self.knots(plan_travel(q_seed=self.io.seed(), base_from_tcp=start, kin=self.kin,
                                                 motion=self.motion)), P.PHASE_TRAVEL, f"touch {i} gauge", ledger)
        time.sleep(float(self.motion["touch"].get("settle_s", 0.5)))
        q = self.io.measured()
        shots = self._wrist_frames(f"gauge-{i}", q, int((self.stack["page"].get("height") or {}).get("gauge_frames", 5)))
        frames = [] if shots is None else [gauge.measure(d, c, shots[2], cal)
                                           for c, d in zip(shots[0], shots[1], strict=True)]
        heights = [gauge.height(f, cal) for f in frames if f is not None]
        if not heights:
            self.event(Event.KIND_PAGE, f"touch {i}: the wrist gauge made out no pen and paper")
            return None
        tcp, g = self.kin.fk(q), float(np.median(heights))
        point = tcp[:3, 3] - g * prior[:3, 2]
        read = {"touch": i, "xy": [float(v) for v in xy], "tcp": tcp.tolist(), "q": q.tolist(), "height_m": g,
                "spread_m": float(np.ptp(heights)), "frames": len(heights), "point": point.tolist(),
                "print": self._hold_tip(shots, frames, q, prior, cal)}
        self.page_record.setdefault("gauge", []).append(read)
        self.event(Event.KIND_PAGE, f"touch {i}: the wrist gauge puts the paper {g * 1e3:.2f} mm under the tip, at "
                   f"({point[0]:.4f} {point[1]:.4f} {point[2]:.4f}) m ({len(heights)} frames, spread "
                   f"{np.ptp(heights) * 1e3:.2f} mm)")
        return read

    def _hold_tip(self, shots, frames, q, prior, cal: dict) -> dict | None:
        """Where a gauge hold's wrist frames put the tip on the print (inspect.hold_tip), with page.locate.holds and
        the installed print's artwork; prior: the page the hold was made on."""
        from tatbot_session import inspect as ins

        page = self.stack["page"]
        artwork = ins.print_artwork(page.get("pattern_id")) if (page.get("locate") or {}).get("holds") else None
        if shots is None or artwork is None:
            return None
        cam_from_page = np.linalg.inv(self.kin.frame(q, ins.camera_frame(self.arm))) @ prior
        return ins.hold_tip(shots[0], frames, shots[2], cal["tip_cam_m"], cam_from_page, page, artwork)

    HOLD_SHIFT_MAX_M = 0.015   # the artwork match's search: a larger shift is not the print's

    def _hold_page(self, touched: np.ndarray, prior: np.ndarray, *, keep_prior: bool = False) -> np.ndarray:
        """`touched` (the paper's height and tilt) placed on the print where the gauge holds' wrist frames put the
        tool (inspect.holds_shift): `prior`, the page the holds were made on, moved by where the tip stood on the
        print, so the plan lands on the print whatever the locate's far views made of it. Without that `touched` as
        it is, or with `keep_prior` (a resume: the retained page's place) prior's place."""
        from tatbot_session import inspect as ins

        found = ins.holds_shift([{"xy": r["xy"], "tip_m": r["print"]["tip_m"]}
                                 for r in self.page_record.get("gauge", []) if r.get("print")])
        if found is not None:
            dx, dy = (v * 1e3 for v in found["shift_m"])
            used = bool(np.hypot(dx, dy) <= self.HOLD_SHIFT_MAX_M * 1e3)
            self.page_record["holds"] = {**found, "used": used}
            self.event(Event.KIND_PAGE, "the gauge holds put the tool {:+.1f} {:+.1f} mm off the print ({} holds, "
                       "spread {:.1f} {:.1f} mm): {}".format(dx, dy, found["holds"], *(v * 1e3 for v in found["spread_m"]),
                                                            "the page moved onto it" if used else "past the artwork "
                                                            "search, not used"))
            if used:
                prior, keep_prior = prior @ geometry.translation([-v for v in found["shift_m"]]), True
        if not keep_prior:
            return touched
        d = np.linalg.inv(touched) @ prior
        return touched @ ins.correction_matrix(d[0, 3], d[1, 3], float(np.arctan2(d[1, 0], d[0, 0])))

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
    def stop_preplan(self):
        if self._preplan is not None:
            self._preplan[1].cancel()
        # A barrier drains earlier untracked work on this executor's single-worker planner.
        try:
            self._planner.submit(lambda: None).result(timeout=1.)
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
        self.prepare_pen({})

    def draw(self, program: dict, ledger, *, from_op: str = "", touch: bool) -> str:
        """Draw every op not yet done. Returns 'complete'; raises Landed or Cancelled."""
        goal_start = time.time()
        self._goal_touch = touch
        ops, rows = program["ops"], ledger.rows()
        start = next((i for i, op in enumerate(ops) if op["id"] == from_op), 0) if from_op else 0
        self.drawn, self.charged = {}, set()
        self.trims = {row["resource"]: float(row["trim_m"]) for row in rows
                      if row.get("event") == "pen_trim" and row.get("arm") == self.arm}
        self.trim_end = 0.0
        self.ride_base = float(self.motion["pen"]["ride"]["fraction"])   # this goal's fresh motion.yaml
        res = self._installed(program, rows, start)
        self.prepare_resource(res)
        self.event(Event.KIND_PAGE, f"resource {res['id']}: {res['ink_id']} in "
                   f"{res['tool']['id']}")
        marker = in_cap(self.arm)
        if marker is not None:
            from tatbot_session import cap

            cap.withdraw_after_restart(self, program, marker, ledger)
        page = self._retained_page(rows)
        if page is not None and self.restore_page(page, ledger):
            regauged = self._regauge(program, ledger)
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
        self.ledger = ledger
        self._chain_snap = self._correction = None
        self.tracker = track.Tracker.start(self, since=goal_start)
        try:
            self._run_ops(program, start, ledger)
        finally:
            self.ledger = None
            self.machine_off()
            self._stop_tracking()
        return "complete"

    def _track_lift(self, program: dict, op: dict, ledger) -> None:
        """Stencil tracking's measurement at a stroke's lifted end (none when the next op continues pen-down)."""
        if self.tracker is None or not self.lift_at_end(program, op, ledger.rows()):
            return
        try:
            self.tracker.after_lift(op)
        except Exception as error:   # tracking is a log: a failure ends it, never the draw
            self.event(Event.KIND_ERROR, f"track: {error}; not tracking")
            self._stop_tracking()

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

    def _retained_page(self, rows) -> dict | None:
        """The run's last measured page, when the fitted tool and its calibration are unchanged since: a resume
        (after a cartridge swap, a landing) keeps it rather than touching the paper again."""
        pages = [row for row in rows if row.get("arm") == self.arm and row.get("event") == "page"]
        if not pages or pages[-1].get("workspace") != self.workspace():
            return None
        return pages[-1]["page"]

    def restore_page(self, page: dict, ledger=None) -> bool:
        """Use a retained page when the cameras still see it where it was touched; False (and the caller sets the
        page up afresh) when it is stale or has moved."""
        lost = float(self.stack["page"].get("max_lost_s", 5.0))
        if self.stack["page"]["source"] != "fixed":
            current = self.node.wait_camera_page(self.arm, timeout=self.PAGE_FIRST_WAIT_S, max_lost_s=lost)
            if current is None or geometry.stale_page(current[1], lost):
                self.event(Event.KIND_PAGE, "retained page: no fresh camera pose; setting the page up again")
                return False
            replan = self.stack["page"]
            if geometry.page_moved(np.asarray(page["camera_at_setup"]), current[0], replan["replan_translation_m"],
                                   replan["replan_rotation_rad"]):
                self.event(Event.KIND_PAGE, "retained page: the sheet moved; setting the page up again")
                return False
            if self._past_reach(np.asarray(page["camera_at_setup"]), current[0]):
                self.event(Event.KIND_PAGE, "retained page: the sheet moved past what stencil tracking follows from it; "
                           "setting the page up again")
                return False
        self.page_record = copy.deepcopy(page)
        self.camera, self.correction = np.asarray(page["camera_at_setup"]), np.asarray(page["correction"])
        self.prior_correction = self.correction.copy()
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
        prior = self.camera @ self.prior_correction   # the retained page: its place on the print, the holds' seed
        contacts = self.page_touches(self._touch_layout(program), ledger, known=True)
        cfg = self.stack["page"].get("height") or {}
        plane = self._gauge_plane(contacts, self.camera[:3, 2], cfg) if len(contacts) >= 3 else None
        if plane is None:
            record.update(retained)
            self.event(Event.KIND_PAGE, "resumed page: the wrist gauge made no plane; the retained height kept")
            return False
        touched = self._hold_page(geometry.touched_page(self.camera, *plane), prior, keep_prior=True)
        self.correction = self.prior_correction = np.linalg.inv(self.camera) @ touched
        record.update(plane=[*plane[0].tolist(), plane[1]], touched=touched.tolist(), used=self.used().tolist(),
                      correction=self.correction.tolist(), retained_height=retained["height"])
        moved = float((self.used()[:3, 3] - old[:3, 3]) @ old[:3, 2])
        self.event(Event.KIND_PAGE, f"resumed page: the wrist gauge puts the paper {moved * 1e3:+.2f} mm from the "
                   "retained height at the page's centre")
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
        lands."""
        from tatbot_session import cap

        if self.resource is None:
            raise RuntimeError("no resource installed on the arm")
        while not cap.dip(self, self.resource, op_id, index, ledger):
            pass
        self.charged.add(self.resource["id"])

    def _run_ops(self, program: dict, start: int, ledger) -> None:
        ops = program["ops"]
        rows = ledger.rows()
        index = ledger_mod.next_index(ops, rows, self.arm, start)
        while index is not None:
            if self.resource is None:
                raise RuntimeError("no resource installed on the arm")
            res = self.resource
            op = ops[index]
            self.feedback(phase=P.PHASE_PLAN, op_id=op["id"], index=index)
            if op["op"] == "tool_change":
                self.tool_change(program, index, ledger)
            elif op["resource_id"] != res["id"]:
                raise RuntimeError(f"op {op['id']} draws with {op['resource_id']}, but {res['id']} is on the "
                                   "arm: its tool change was skipped (--from-op)")
            elif op["op"] == "dip":
                ledger.append("sent", self.arm, op=op["id"], index=index)
                self.dip(op["id"], index, ledger)
                ledger.append("done", self.arm, op=op["id"], index=index)
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
                if res["dip"] is not None and res["id"] not in self.charged:
                    self.event(Event.KIND_OP_SENT, "no dip yet in this goal: dipping before the stroke", op["id"])
                    self.dip(op["id"], index, ledger)
                self.stroke(program, index, ledger)
            rows = ledger.rows()
            index = ledger_mod.next_index(ops, rows, self.arm, start)

    def lift_at_end(self, program: dict, op: dict, rows) -> bool:
        """Stay down only for the immediate, unstarted continuation; never across a pause or skip."""
        ops = program["ops"]
        index = next(i for i, item in enumerate(ops) if item["id"] == op["id"])
        if index + 1 == len(ops):
            return True
        following = ops[index + 1]
        return not (following["op"] == "stroke" and following.get("continues")
                    and ledger_mod.op_status(rows, self.arm, following["id"]) is None)

    def plan(self, program: dict, op: dict, from_arc: float, pen_down: bool, seed: np.ndarray,
             *, lift_at_end: bool = True, snap=None):
        from tatbot_motion import plan_op

        if snap is not None:   # stencil tracking: the points moved by the print's residual field (track.Field)
            op = snap.corrected(op)[0]
        return self.knots(plan_op(op, base_from_page=self.used(), q_seed=seed, kin=self.kin, motion=self.motion,
                                  speed_m_s=float(program.get("draw_speed_m_s", 0.0035)), from_arc_m=from_arc,
                                  pen_down_at_start=pen_down, lift_at_end=lift_at_end, pen=self.pen))

    def preplan(self, program: dict, index: int, seed: np.ndarray, rows) -> None:
        """Plan the next stroke while the current one runs, seeded at the current plan's end."""
        ops = program["ops"]
        nxt = ledger_mod.next_index(ops, rows, self.arm, index + 1)
        if nxt is None or ops[nxt]["op"] != "stroke" or ledger_mod.op_status(rows, self.arm, ops[nxt]["id"]):
            self._preplan = None
            return
        op = ops[nxt]
        lift = self.lift_at_end(program, op, rows)
        snap = self._snap(op)
        hover = self._hover_on() and not op.get("continues")
        self._preplan = (op["id"], self._planner.submit(self._preplanned, program, op, seed, lift, snap, hover), lift,
                         snap)

    def _preplanned(self, program: dict, op: dict, seed: np.ndarray, lift: bool, snap, hover: bool):
        """The pre-plan's work: (stencil tracking's hover travel or None, the stroke from its end or from seed)."""
        travel = self.plan_hover(op, seed, snap) if hover else None
        start = seed if travel is None else np.asarray(travel.q[-1])
        return travel, self.plan(program, op, 0.0, bool(op.get("continues")), start, lift_at_end=lift, snap=snap)

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
        if moved <= float((self.stack["page"].get("track") or {}).get("replan_m", 0.0003)):
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
        the trim the last goal ended at, has not drifted from the measured pose (motion.yaml dispatch_drift), else
        a fresh plan from the measured joints, trimmed from 0. plan_op joins the path pen-down when the tip already
        sits on it (a `continues` segment, a resumed stroke) and lifts and re-approaches otherwise."""
        from tatbot_motion import dispatch_drift

        self.refresh_page()
        pen_down = bool(op.get("continues")) or from_arc > 0
        lift = self.lift_at_end(program, op, rows)
        snap = self._snap(op)
        pre, self._preplan = self._preplan, None
        if pre is not None and pre[0] == op["id"] and pre[2] == lift and from_arc <= 0 and self._same_field(op, pre[3], snap):
            try:
                traj = pre[1].result()[1]
            except Exception as exc:  # noqa: BLE001 - a failed pre-plan is planned again below
                self.event(Event.KIND_REPLAN, f"pre-plan failed: {exc}", op["id"])
                traj = None
            if traj is not None:
                sent = self.trimmed(traj, self.trim_end)
                drift = dispatch_drift(sent, self.io.measured(), self.kin, self.motion)
                if drift["ok"]:
                    self._use_snap(op, pre[3])
                    return traj, sent
                self.event(Event.KIND_REPLAN, "drift at dispatch: {:.1f} mrad, {:.2f} mm".format(
                    drift["joint_rad"] * 1000, drift["tip_m"] * 1000), op["id"])
        traj = self.plan(program, op, from_arc, pen_down, self.io.seed(), lift_at_end=lift, snap=snap)
        self._use_snap(op, snap)
        return traj, self.trimmed(traj, 0.0)

    # --- stencil tracking's hover stop (ros/README.md 4.4) -------------------------------------------
    def _hover_on(self) -> bool:
        return (self.tracker is not None and self.tracker.correcting
                and bool((self.stack["page"].get("track") or {}).get("hover_stop")))

    def plan_hover(self, op: dict, seed: np.ndarray, snap, from_arc: float = 0.0):
        from tatbot_motion import plan_hover

        if snap is not None:
            op = snap.corrected(op)[0]
        return self.knots(plan_hover(op, base_from_page=self.used(), q_seed=seed, kin=self.kin, motion=self.motion,
                                     from_arc_m=from_arc, pen=self.pen))

    def _hover_stop(self, program: dict, op: dict, from_arc: float, ledger) -> None:
        """Stop over a stroke's start before it descends and measure the pen tip on the print there, so the stroke
        is planned with the field at its own start rather than one stroke behind. A jump is measured again from a
        second view track.second_view_m toward the page centre: one view cannot tell a lattice misfit from a moved
        sheet. A hover that finds no print while the overhead sees the sheet moved past the wrist fit's reach sets the
        page up again first, unless the overhead freshly sees the sheet within that reach: no stroke goes down
        blind. A print the wrist camera has read nowhere in this goal is one its fit cannot read (the seed-102 W
        transfer on silicone, 2026-10-05), not a sheet that left: hover stops end and the strokes go on the measured
        page. Not for a pen-down continuation, nor a resume whose tip is still at the page."""
        from tatbot_motion import timelaw

        if not self._hover_on() or op.get("continues") or (self._height() or 0.0) < float(self.motion["approach"]["final_m"]):
            return
        start = timelaw.trim_polyline(np.asarray(op["points_m"], float), from_arc)[0]
        verdict = self._measure_over(op, f"{op['id']}-hover", start, self._pre_travel(op), from_arc, ledger)
        if verdict is None and not self.tracker.field.samples:
            self.event(Event.KIND_PAGE, "track: the wrist camera has read the print nowhere in this goal: no more hover "
                       "stops, the strokes go on the measured page", op["id"])
            self.tracker.correcting = False
            return
        if verdict is None and self._sheet_may_have_left():
            self._reacquire(program, ledger)
            if not self._hover_on():
                return
            verdict = self._measure_over(op, f"{op['id']}-rehover", start, None, from_arc, ledger)
        if verdict != "held":
            return
        norm = float(np.linalg.norm(start))
        away = -start / norm if norm > 1e-6 else np.array([1.0, 0.0])
        shift = away * float(self.stack["page"]["track"].get("second_view_m", 0.005))
        view = {**op, "points_m": (np.asarray(op["points_m"], float) + shift).tolist()}
        self._measure_over(view, f"{op['id']}-hover2", start + shift, None, from_arc, ledger)

    def _measure_over(self, op: dict, name: str, plan_xy, travel, from_arc: float, ledger) -> str | None:
        """Travel to the hover over op's start (the pre-planned travel first), then the tracker's verdict there."""
        snap = self.tracker.snapshot()
        planned = [travel] if travel is not None else []
        self.move(lambda: planned.pop() if planned else self.plan_hover(op, self.io.seed(), snap, from_arc),
                  P.PHASE_TRAVEL, f"{op['id']} hover", ledger)
        self.trim_end = 0.0   # a travel carries no pen trim
        return self.tracker.at_hover(name, op["id"], plan_xy)

    def _past_reach(self, setup: np.ndarray, current: np.ndarray | None) -> bool:
        """With stencil tracking correcting, whether the overhead sees the sheet moved past what the wrist fit reaches
        from the page set up at `setup` (track.reach_m, reach_rad), or has no fresh pose (None) to say it has not."""
        cfg = self.stack["page"].get("track") or {}
        if cfg.get("mode") != "correct" or self.stack.get("hardware") != "real":
            return False
        return current is None or geometry.page_moved(setup, current, float(cfg.get("reach_m", 0.010)),
                                                      float(cfg.get("reach_rad", 0.05)))

    def _sheet_may_have_left(self) -> bool:
        """After a hover found no print: the overhead's pose, measured within page.max_lost_s (the arm over the page
        hides it), moved past the wrist fit's reach, or no such pose."""
        newest = self.node.camera_page(self.arm)
        lost = float(self.stack["page"].get("max_lost_s", 5.0))
        fresh = newest is not None and geometry.stale_page(newest[1], lost) is None
        return self._past_reach(self.camera, newest[0] if fresh else None)

    def _reacquire(self, program: dict, ledger) -> None:
        """Set the page up again mid-draw (the wrist camera is the page setup's while it runs) and start tracking
        again from its gauge holds."""
        self.event(Event.KIND_PAGE, "track: no print under the hover, and the overhead cannot place the sheet within "
                   "the wrist fit's reach; setting the page up again")
        since = time.time()
        self.stop_preplan()
        self.machine_off()
        self._stop_tracking()
        self._new_page(program, self._goal_touch, ledger)
        self._chain_snap = self._correction = None
        self.tracker = track.Tracker.start(self, since=since)

    def _new_page(self, program: dict, touch: bool, ledger) -> None:
        self.setup_page(program, touch, ledger)
        if self.pen is not None:
            self.page_record["pen"] = dataclasses.asdict(self.pen)
        ledger.append("page", self.arm, page=self.page_record, workspace=self.workspace())

    def _pre_travel(self, op: dict):
        """This op's pre-planned hover travel, when it starts where the arm is."""
        from tatbot_motion import dispatch_drift

        pre = self._preplan
        if pre is None or pre[0] != op["id"]:
            return None
        try:
            travel = pre[1].result()[0]
        except Exception:  # noqa: BLE001 - dispatch_plan reports a failed pre-plan
            return None
        if travel is None or not dispatch_drift(travel, self.io.measured(), self.kin, self.motion, pen_down=False)["ok"]:
            return None
        return travel

    # --- the operator's pen trim (motion.yaml pen.trim, ros/README.md 4.4) -------------------------
    def trim_target(self) -> float:
        return self.trims.get(self.resource["id"], 0.0) if self.resource else 0.0

    def trimmed(self, traj, start_m: float):
        """traj with the pen trim, easing from start_m (where the arm is) to the cartridge's."""
        from tatbot_motion import Trim, compose

        trim = Trim(start_m)
        return compose(traj, trim if abs(self.trim_target() - start_m) < 1e-9 else
                       trim.to(0.0, self.trim_target(), self.motion["pen"]["trim"]))

    def _retarget(self, plan):
        """ArmIO.execute's retarget: the goal again, easing to a changed trim from pen.trim.lead_s ahead; None while
        a change still eases in, or when it would not settle before the goal ends (the next goal starts at it)."""
        from tatbot_motion import compose

        cfg = self.motion["pen"]["trim"]

        def retarget(elapsed: float, active):
            trim = active.info["trim"]
            if abs(trim.end_m - self.trim_target()) < 1e-9 or elapsed < trim.settled_s():
                return None
            new = trim.to(elapsed + float(cfg["lead_s"]), self.trim_target(), cfg)
            return compose(plan, new) if new.settled_s() < plan.t[-1] else None

        return retarget

    def on_key(self, key: str) -> None:
        """A numpad key: up/down move the cartridge's trim a step within its limit, reset sets 0, each a ledger
        row a resume starts from; enter answers a waiting pause or latch CONTINUE."""
        if key == "enter" and not self.io.flag("landed"):   # landed: a swap's, which `client draw` answers
            ok, why = (False, "nothing waits for continue") if self.goal_active and self.waiting is None \
                else self.decide(rules.CONTINUE)
            if not ok:
                self.event(Event.KIND_DECISION, f"enter: {why}")
        if key not in ("up", "down", "reset"):
            return
        if self.ledger is None or self.resource is None:
            return self.event(Event.KIND_PEN_TRIM, f"{key}: no draw is running, so no cartridge to trim")
        step, res, old = float(self.motion["pen"]["trim"]["step_m"]), self.resource["id"], self.trim_target()
        most = int(float(self.motion["pen"]["trim"]["limit_m"]) / step + 1e-6)
        steps = 0 if key == "reset" else round(old / step) + (1 if key == "up" else -1)
        new = max(-most, min(most, steps)) * step
        if abs(new - old) < 1e-12:
            return self.event(Event.KIND_PEN_TRIM, f"{res} stays {new * 1e3:+.1f} mm" + (": its limit" if steps else ""))
        self.trims[res] = new
        self.ledger.append("pen_trim", self.arm, resource=res, trim_m=round(new, 7))
        over = f": the path {(self.pen.height_m + new) * 1e3:.2f} mm over the page" if self.pen else ""
        self.event(Event.KIND_PEN_TRIM, f"{res} {new * 1e3:+.1f} mm{over}")

    def stroke(self, program: dict, index: int, ledger) -> None:
        op = program["ops"][index]
        length = geometry.stroke_length(op["points_m"], closed=bool(op.get("closed")))
        from_arc = ledger_mod.resume_arc(ledger.rows(), self.arm, op["id"])
        while True:
            self._hover_stop(program, op, from_arc, ledger)
            if self.pen and self.pen.machine:
                self.machine_running(op["id"], index, ledger)
            traj, sent = self.dispatch_plan(program, op, from_arc, ledger.rows())
            ledger.append("sent", self.arm, op=op["id"], index=index, arc_m=from_arc, **self._correction_row())
            self.event(Event.KIND_OP_SENT, f"{len(traj.t)} knots, {traj.t[-1]:.1f} s from arc {from_arc * 1000:.1f} mm",
                       op["id"])
            status, reached, text = self.io.execute(sent, should_stop=self.stopping,
                                                    on_tick=self._ticker(program, index, op, traj, ledger),
                                                    retarget=self._retarget(traj))
            sent = self.io.active   # a trim change mid-goal replaced it
            self.trim_end = float(sent.info["trim"].value(sent.t[reached]))
            line = self.drawn[op["id"]][-1]   # the measured pen-down samples, for drawn.svg across resumes
            if status == "ok":
                ledger.append("done", self.arm, op=op["id"], index=index, arc_m=length, line=line)
                self.event(Event.KIND_OP_DONE, "", op["id"])
                self._track_lift(program, op, ledger)
                return
            self._preplan = None
            arc = geometry.arc_at(sent, reached, self.tcp()[:3, 3])
            reason = {"latched": f"latched: {text}", "cancelled": "cancelled"}.get(status, f"failed: {text}")
            drawn = np.flatnonzero(np.asarray(traj.phase) == P.PHASE_DRAW)
            finished = (len(drawn) and reached > drawn[-1]) or arc >= length - float(self.motion["resume_reapproach_m"])
            if finished:  # stopped in the final lift: the stroke is drawn; resuming would dot its end again
                ledger.append("done", self.arm, op=op["id"], index=index, arc_m=length, reason=reason, line=line)
                self.event(Event.KIND_OP_DONE, f"{reason} after the stroke was drawn", op["id"])
            else:
                ledger.append("aborted", self.arm, op=op["id"], index=index, arc_m=arc, reason=reason, line=line)
                self.event(Event.KIND_OP_ABORTED, f"{reason} at arc {arc * 1000:.1f} mm", op["id"])
            if status == "cancelled" and not self.land_requested:
                raise Cancelled()
            if self.wait_decision("latched", op["id"], index, ledger) == rules.LAND:
                self.land()
            if finished:  # lift unless the next op continues pen-down from here (plan_op joins it)
                height = self._height()
                if self.lift_at_end(program, op, ledger.rows()) and height is not None and height < float(self.motion["approach"]["standoff_m"]) - 0.001:
                    self.lift(f"{op['id']} lift", ledger)
                return
            from_arc = arc

    def _ticker(self, program, index, op, traj, ledger):
        """on_tick for a stroke goal: feedback at 10 Hz, drawn samples at 20 Hz, the pre-plan once."""
        state = {"fb": 0.0, "drawn": 0.0, "pre": False, "line": []}
        self.drawn.setdefault(op["id"], []).append(state["line"])
        inv_page = np.linalg.inv(self.used())

        def tick(i: int) -> None:
            now = time.monotonic()
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

    def move_single(self, target: np.ndarray, guard: int = names.GUARD_NONE) -> dict:
        """Travel to `target` and hold there (Touch MODE_MOVE: a view of the station, a calibration hold).
        Under GUARD_PROBE it takes a probe touch's way (station_via) and the guard is armed throughout: a trip
        is an error. Unguarded, a travel the Cartesian planner refuses (from rest on the joint limits, or past
        one on the way) is a joint-space move instead (joint_travel)."""
        from tatbot_motion import plan_travel
        from tatbot_motion.clik import PlanError

        probe = guard == names.GUARD_PROBE
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
