"""A palette dip in the arm's executor (ros/README.md section 6).

The cap is placed by the station touch (tatbot_motion.station.touched_palette): the probe's ball where the
calibration's touches met it, the palette's yaw from a fresh overhead fix, which also says whether the palette has
moved since the touch. Then approach (page to the cap's hover), descend, dwell, retract along the cap's axis and
return to the page standoff, each one goal planned by tatbot_motion.dip from the measured joints and checked at
every tick: the arm's bodies clear of the station's parts, the tool on the cap's axis and over its dwell height
inside the cap.

The machine is off except during the dwell. It runs there when the tool's needle reach is recorded: the tube's end
holds above_ink_m over the ink and the needles' strokes pull the ink up into the tube. With no reach recorded the
dwell holds with the machine off and says so.

A phase that ended off its place (the end checks of at_end, no tick's fault) withdraws and dips again once with no
key; a return judged at the page standoff once the arm settles is planned again once, then goes on from where it is
(back). A latch, a tick's fault or a second miss waits for continue or land, withdraws (along the cap's axis when the
tool may be in the cap, then back to the page) and is dipped again, or lands. A cancel holds where it is; while the
tool may be in a cap a marker (tatbot_session.lease) refuses landing, and the run's next draw withdraws first.
"""
from __future__ import annotations

import time

import numpy as np
from tatbot_motion import dip as cap_motion
from tatbot_motion import station
from tatbot_motion.collision import PALETTE_BODY_MARGIN_M, PALETTE_PADS

from tatbot_session import rules
from tatbot_session.executor import Cancelled
from tatbot_session.lease import clear_in_cap, mark_in_cap
from tatbot_session.ports import Event

IMMERSED = ("descend", "dwell", "retract")
HOLLOW = ("hover", "cap_exit", *IMMERSED)   # stages where the tool may be over or in its cap


def dip(ex, resource: dict, op_id: str, index: int, ledger) -> bool:
    """True when the tool dwelt over the ink and is back at the page standoff; False after an interrupted dip
    withdrew (dip again). Raises Landed or Cancelled as the decision says."""
    ex.stop_preplan()
    ex.snag_guard(False)   # the needle left the work at the standoff; a cap is not the work
    ex.require_off("a dip")
    return Cycle(ex, resource, op_id, index, ledger).run()


def withdraw_after_restart(ex, program: dict, marker: dict, ledger) -> None:
    """A dip that ended with its tool possibly in a cap (a crash, a cancel, a reboot): straight out along the cap
    axis from the measured pose, then up and back to where it left the page."""
    ex.event(Event.KIND_PAGE, f"the tool may be in cap {marker['slot']} (run {marker['run']}): withdrawing first")
    ex.require_off("cap withdrawal")
    Cycle(ex, marker["resource"], marker["op"], 0, ledger, return_tcp=np.asarray(marker["return_tcp"])).withdraw()


def _load(ex, slot: str, ink_id: str):
    """The cap's declared ink level (m over its inner floor); RuntimeError unless it holds this ink."""
    import ink_spec

    load = ink_spec.load_palette_load(ex.repo).get(slot)
    if load is None or load.cap_present is not True or load.ink_id != ink_id or load.level_lower_bound_m is None:
        held = "nothing declared" if load is None else (f"{load.ink_id or 'no ink'}, level "
                                                         f"{load.level_lower_bound_m}")
        raise RuntimeError(f"cap {slot} must be declared with {ink_id} and its level (tatbot ros palette load "
                           f"{slot}={ink_id} --level-mm {slot}=MM); it holds {held}")
    return float(load.level_lower_bound_m)


class Cycle:
    def __init__(self, ex, resource: dict, op_id: str, index: int, ledger, *, return_tcp=None):
        import tool_spec
        from tatbot_bridge.station import observe

        self.ex, self.resource, self.op_id, self.index, self.ledger = ex, resource, op_id, index, ledger
        self.guard = ex.node.guard
        if self.guard is None:
            raise RuntimeError("a dip needs the collision scene (stack.yaml collision) to keep clear of the station")
        slot = resource["slot"]
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        fix = observe(ex.repo, ex.arm, 8, ex.run_dir / "station" / f"{op_id}-{stamp}", ex.stack["registration"][ex.arm])
        touch = station.load_touch(ex.arm)
        q = ex.io.measured()
        tcp_m = (np.linalg.inv(ex.kin.frame(q, f"{ex.arm}/tool_mount")) @ ex.kin.fk(q))[:3, 3]
        urdf, palette = ex.repo / "urdf/palette.urdf", ex.repo / "config/palette.yaml"
        base_from_palette = station.touched_palette(touch, fix, tool_id=resource["tool"]["id"], tcp_m=tcp_m,
                                                    joint_offsets=station.joint_offsets(ex.repo, ex.arm),
                                                    palette_urdf=urdf,
                                                    same_body=station.same_machine(ex.repo, touch.tool_id,
                                                                                   resource["tool"]["id"]))
        with self.guard._lock:
            base = self.guard.scene.base_pose(ex.arm)
        import ink_spec

        zones = station.parts(base_from_palette, station.inkcap_rims(urdf, palette, load=ink_spec.load_palette_load(ex.repo),
                                                                     base_from_palette=base_from_palette), urdf)
        self.station_top = max(float(centre[2]) + (0.0 if post else radius) for _, centre, radius, post in zones)
        post = station.overhead_post(np.linalg.inv(base)[:3, 3])
        zones.append(("overhead_post", np.array([*post.xy, post.top_m]), post.radius_m, True))
        self.parts = station.station_world_parts(zones, base, float(base_from_palette[2, 3]))
        spec = tool_spec.load_tool(resource["tool"]["id"], ex.repo)
        cap = cap_motion.read_cap(ex.repo, slot, ex.arm, base_from_palette)
        reach = spec.raw.get("needle_reach_mm")
        level = _load(ex, slot, resource["ink_id"])

        # the touches' heading only: there the tip's error across the tool's axis cancels, and at another it moves
        # the tool off the cap's axis by twice that error times the sine of half the turn (2026-10-08: the 3RL's tip
        # read ~10 mm off across its axis, ~8 mm off at a 45 deg turn against ~5 mm of a large cap's bore to spare)
        self.dip = cap_motion.Dip(cap, cap_motion.Tool.from_profile(spec.profile), resource["dip"], level,
                                  touch.rotation, reach_m=None if reach is None else float(reach) / 1000.0)
        self.heading_deg = float(np.degrees(np.arctan2(touch.rotation[1][0], touch.rotation[0][0])))
        self.machine = self.dip.reach_m is not None and ex.machine is not None
        self.return_tcp = return_tcp
        self.marker = {"run": ex.run_dir.name, "op": op_id, "slot": slot, "resource": resource}
        ex.event(Event.KIND_PAGE, f"dip {slot} at heading {self.heading_deg:+.0f} deg, the touch's: the ball "
                 f"{np.round((touch.ball - fix.ball) * 1000, 1).tolist()} mm "
                 f"from the overhead fix (touch {touch.run}); the tool's end {self.dip.settings['above_ink_m'] * 1000:.1f} "
                 f"mm over the ink, {self.dip.slack_m * 1000:.2f} mm of the bore to spare, machine "
                 + ("on for the dwell" if self.machine else "off (no needle reach recorded)"), op_id)

    # --- geometry --------------------------------------------------------------------------------
    def check(self, stage: str, rows) -> None:
        """The arm's bodies but its tool keep the station margin from every part; the tool clears every part but the
        active cap's solid proxy while it is over or in that cap (its bore is cap_motion.check_tick's)."""
        with self.guard._lock:
            gap = self.guard.scene.parts_clearance(self.ex.arm, rows, self.parts, PALETTE_PADS,
                                                   skip=("tool", "tattoo_needle"), max_step_rad=self.guard.step_rad)
            others = [part for part in self.parts if stage not in HOLLOW or part[0] != self.dip.cap.slot]
            tool = self.guard.scene.parts_clearance(self.ex.arm, rows, others, skip=(), max_step_rad=self.guard.step_rad)
        if gap.distance_m < PALETTE_BODY_MARGIN_M or tool.distance_m <= 0:
            refused = gap if gap.distance_m < PALETTE_BODY_MARGIN_M else tool
            raise RuntimeError(f"dip: {refused.body_a} cannot clear station part {refused.body_b}")

    def plan(self, phase: str, q):
        args = {"q_seed": q, "kin": self.ex.kin, "motion": self.ex.motion, "check": self.check}
        if phase == "approach":
            try:
                traj = cap_motion.plan_approach(self.dip, base_from_page=self.ex.used(), station_top_m=self.station_top,
                                                **args)
            except (RuntimeError, ValueError) as error:   # PlanError is a RuntimeError
                raise RuntimeError(f"the touch's heading {self.heading_deg:+.0f} deg does not reach cap "
                                   f"{self.dip.cap.slot} ({error}); a dip goes only at the heading the station was "
                                   "touched at: touch it at one that reaches the cap") from error
            self.return_tcp = np.asarray(traj.info["return_tcp"])
            return traj
        if phase == "return":
            return cap_motion.plan_return(self.dip, return_tcp=self.return_tcp, station_top_m=self.station_top, **args)
        return cap_motion.plan_axial(self.dip, phase, **args)

    def at_end(self, phase: str, pose) -> None:
        """Where the measured tool must be when a phase's goal ends (the return's is ended's)."""
        height, off = self.dip.local(pose)
        if phase in ("approach", "retract") and (height < self.dip.hover_height_m - cap_motion.HOVER_LAG_M
                                                 or off > self.dip.slack_m):
            raise RuntimeError(f"{phase}: the tool's end ended {height * 1000:.1f} mm up, {off * 1000:.2f} mm off the "
                               "cap's axis, not at its hover")
        elif phase in ("descend", "dwell") and not (-cap_motion.OVERSHOOT_M <= height - self.dip.dwell_height_m
                                                    <= cap_motion.DWELL_HIGH_M):
            raise RuntimeError(f"{phase}: the tool's end is {(height - self.dip.dwell_height_m) * 1000:+.2f} mm from "
                               "its dwell height")

    def ended(self, phase: str) -> tuple[str, str] | None:
        """(status, text) for a goal that ended ok off its place, else None: a return judged at the page standoff
        (cap_motion.return_miss) once the arm settles there, its stop from ~118 mm/s ringing ~0.5 s, so the first
        reading within the limits passes, else the last of RETURN_SETTLE_S (missed); another phase by at_end (off)."""
        if phase != "return":
            try:
                self.at_end(phase, self.ex.kin.fk(self.ex.io.measured()))
            except (RuntimeError, ValueError) as exc:
                return "off", str(exc)
            return None
        until = self.ex.clock.now() + cap_motion.RETURN_SETTLE_S
        while (miss := cap_motion.return_miss(self.ex.kin.fk(self.ex.io.measured()), self.return_tcp)) is not None \
                and self.ex.clock.now() < until:
            self.ex.clock.sleep(0.05)
        return None if miss is None else ("missed", miss)

    # --- motion ----------------------------------------------------------------------------------
    def phase(self, phase: str) -> tuple[str, str]:
        """One phase's goal, monitored at every tick. Returns (status, text): ok | latched | cancelled | failed (a
        tick's fault) | off (it ended off its place, at_end) | missed (a return off the page standoff, settled)."""
        if phase == "dwell" and self.machine:
            self.ex.machine_running(self.op_id, self.index, self.ledger)
        else:
            self.ex.require_off("a dip")
        try:
            traj = self.ex.knots(self.plan(phase, self.ex.io.measured()))
        except (RuntimeError, ValueError) as error:   # nothing moved; the arm holds where the last phase left it
            self.ledger.append("dip_phase", self.ex.arm, op=self.op_id, index=self.index, phase=phase, status="unplanned",
                               text=str(error), slot=self.dip.cap.slot, q=self.ex.io.measured().tolist())
            raise RuntimeError(f"dip {phase}: {error}") from error
        stages = list(traj.info.get("stages_s", {}).items())
        fault = []

        def tick(i: int) -> None:
            if fault:
                return
            stage = next((name for name, end in stages if traj.t[i] <= end), stages[-1][0] if stages else phase)
            q = self.ex.io.measured()
            try:
                self.check(stage, [q])
                cap_motion.check_tick(self.dip, stage, self.ex.kin.fk(q), self.station_top)
            except (RuntimeError, ValueError) as exc:
                fault.append(str(exc))   # stop the goal; never raise out of the controller's callback

        self.ex.feedback(op_id=self.op_id, index=self.index, phase=int(traj.phase[0]))
        try:
            status, _, text = self.ex.io.execute(traj, on_tick=tick,
                                                 should_stop=lambda: self.ex.stopping() or bool(fault))
        finally:
            if phase == "dwell":
                self.ex.machine_off()
        if fault:
            status, text = "failed", fault[0]
        elif status == "ok":
            status, text = self.ended(phase) or (status, text)
        q = self.ex.io.measured()
        self.ledger.append("dip_phase", self.ex.arm, op=self.op_id, index=self.index, phase=phase, status=status,
                           text=text, slot=self.dip.cap.slot, q=q.tolist())
        return status, text

    def run(self) -> bool:
        for again in (False, True):
            phase, status, text = self.cycle()
            if status == "ok":
                return True
            if status != "off" or again:
                return self.interrupted(phase, status, text)
            # ended off its place, no tick's fault nor a latch: out and in again once, no key (owner 2026-10-20)
            self.ex.event(Event.KIND_ERROR, f"dip {phase}: {text}: withdrawn to dip again", self.op_id)
            self.withdraw()
        return False

    def cycle(self) -> tuple[str, str, str]:
        """The dip's phases in turn: (the phase that stopped it, its status, text), or ("return", ok, "")."""
        for phase in ("approach", *IMMERSED):
            if phase == "descend":
                mark_in_cap(self.ex.arm, {**self.marker, "return_tcp": self.return_tcp.tolist()})
            status, text = self.phase(phase)
            if phase == "retract" and status == "ok":
                clear_in_cap(self.ex.arm)
            if status != "ok":
                return phase, status, text
            if phase in ("approach", "descend"):
                self.look("hover" if phase == "approach" else "dwell")
        return ("return", *self.back())

    def back(self) -> tuple[str, str]:
        """The return to the page standoff; one that ended off it once the arm settled is planned again from where the
        arm stands, once, and still off the dip goes on from there: dwelt and out of the cap, the next stroke plans
        from the measured joints and comes to its own standoff first. Two misses read before the arm settled (1.09
        and 1.66 mm) ended a tattoo with the arm holding (2026-10-20)."""
        for again in (False, True):
            status, text = self.phase("return")
            if status != "missed":
                return status, text
            self.ex.event(Event.KIND_ERROR, f"dip return: {text}: " + (
                "it goes on from where it is" if again else "planned again from where the arm stands"), self.op_id)
        return "ok", text

    def look(self, where: str) -> None:
        """A wrist D405 frame down the tool at the hover and in the dwell, saved as station/<op>-<where>.png: the
        cap's opening and the nozzle in one view, the evidence of where the dip went. The stencil tracker's open
        camera when it has one, else a capture of its own; a missing frame is reported, never a fault."""
        import cv2

        tracker = getattr(self.ex, "tracker", None)
        out = self.ex.run_dir / "station"
        try:
            if tracker is not None and getattr(tracker, "cam", None) is not None:
                while tracker.cam.pipe.poll_for_frames():   # drop queued frames: the next one is where the arm is
                    pass
                colors, _ = tracker.cam.grab_both(1)
                frame = colors[0]
            else:
                shots = self.ex._wrist_frames(f"{self.op_id}-{where}", self.ex.io.measured(), 1)
                if shots is None:
                    return
                frame = shots[0][0]
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{self.op_id}-{where}.png"), frame)
        except Exception as error:   # evidence only
            self.ex.event(Event.KIND_ERROR, f"dip {where}: no wrist frame ({error})", self.op_id)

    def interrupted(self, phase: str, status: str, text: str) -> bool:
        self.ex.event(Event.KIND_ERROR, f"dip {phase}: {status}: {text}", self.op_id)
        if status == "cancelled" and self.ex.cancelled():
            raise Cancelled()   # the arm holds; a tool that may be in the cap is withdrawn by the run's next draw
        decision = self.ex.wait_decision("latched", self.op_id, self.index, self.ledger)
        self.withdraw()
        if decision == rules.LAND:
            self.ex.land()
        return False

    def withdraw(self) -> None:
        """After an interrupted phase: out along the cap's axis when the tool may be in the cap, then the return,
        which rises at the measured attitude first when the tool is not over the cap."""
        if self.ex.io.latched:
            ok, text = self.ex.io.unlatch()
            if not ok:
                raise RuntimeError(f"cap withdrawal needs the arm unlatched: {text}; it holds where it is")
        if self.dip.in_cap(self.ex.kin.fk(self.ex.io.measured())):
            status, text = self.phase("retract")
            if status != "ok":
                raise RuntimeError(f"cap withdrawal: {status}: {text}; the tool may still be in "
                                   f"{self.dip.cap.slot}, and the arm holds")
            clear_in_cap(self.ex.arm)
        status, text = self.back()
        if status != "ok":
            raise RuntimeError(f"return from the palette: {status}: {text}; the arm holds")
