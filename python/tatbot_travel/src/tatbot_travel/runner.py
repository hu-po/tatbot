"""The travel demo on the real blue arm: wrist camera in, policy plans out. Runs on the arm node.

One process owns everything: the policy on this node's GPU, the blue arm's
wrist D405, and -- only with ``--move`` or ``--hold`` -- the arm itself, through
``hardware.BlueArm`` (driver lease, e-stop monitor before anything moves,
golden config, staging). Without ``--move`` nothing connects to the arm: the
state is pinned at the start pose and each plan is only the policy's intent,
drawn over the live image. With ``--hold`` the arm is taken and moved to the
start pose, then held: the policy sees the real view from that pose and the real
joints, and its intent is drawn, but nothing it plans is sent.

With ``--move``, the arm rises to its staged pose, moves slowly to the pose
the policy was trained to start from (the sim's park pose), and hands over.
Every 30 Hz tick the newest frame and the measured joints enter the policy's
history with the command in force; plans are computed back to back in a
background thread and blended in as they arrive (``chunking``); and the
command is gated, never the policy's alone:

- joint limits less a margin, and at most ``max_speed`` per joint;
- the workspace, by forward kinematics of every command: no link above
  ``ceiling_m`` (set per rig: the lab's overhead cameras hang about 0.55 m
  over the arm's base, and the sim's park pose already lifts the elbow to
  0.50 m), the pen tip above ``floor_m`` and within ``reach_m``; a command
  outside is held, and a policy that keeps pushing out for ``outside_s``
  ends the run;
- tracking: a joint more than ``track_rad`` from its command for
  ``track_s`` means the arm is blocked, and ends the run;
- the depth clearance along the pen's axis: closer than ``stop_mm`` the arm
  holds, and closer than ``abort_mm`` for ``abort_s`` ends the run;
- a camera frame older than ``stale_s`` holds the arm;
- a pressed e-stop freezes it and ends the run.

On exit the arm returns to its staged pose and idles; if that fails, the
recovery landing takes it under a hard budget, and the summary's ``landing``
says which (``landed``, ``recovered``) or that the arm state is ``unknown``.
Every call on the arm's driver is bounded by a watchdog process: one that
never returns (a wedged controller) ends the run where it is, and the watchdog
hands the arm to the same recovery landing and writes the summary.
Everything lands under ``~/tatbot-logs/travel_run/<run id>/``: per-tick and
per-plan logs, a video with the plan and clearance drawn in, and a summary.
"""

from __future__ import annotations

import json
import logging
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tatbot_travel.chunking import ChunkScheduler, History, Plan, condition_plan
from tatbot_travel.shadow import Overlay, axis_clearance

FPS = 30
SCENE_STALE_S = 0.5  # a scene frame this old holds the arm (the PoE sub stream runs ~16 fps over the network)
log = logging.getLogger("travel.runner")


@dataclass(frozen=True)
class Gates:
    ceiling_m: float = 0.45  # no link above this, in the arm's base frame (set per rig)
    floor_m: float = 0.0  # the pen tip (gap point) never below this
    reach_m: float = 0.55  # the pen tip within this radius of the base axis
    outside_s: float = 1.0  # a policy pushing outside the workspace this long ends the run
    track_rad: float = 0.10  # a joint this far from its command ...
    track_s: float = 0.15  # ... for this long: the arm is blocked, end the run
    max_speed: float = 0.25  # rad/s, any joint, policy phase
    approach_speed: float = 0.25  # rad/s, the move to the start pose
    margin: float = 0.05  # rad inside the controller's joint limits
    stop_mm: float = 10.0  # hold while the pen axis meets a surface closer than this
    abort_mm: float = 4.0  # end the run when it stays closer than this ...
    abort_s: float = 0.5  # ... for this long
    stale_s: float = 0.3  # hold when the newest camera frame is older than this


class Clamp:
    """A command step toward a goal: inside the joint limits, at most ``max_step`` per joint."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray, max_step: float):
        self.lower, self.upper, self.max_step = lower, upper, max_step

    def __call__(self, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        goal = np.clip(goal, self.lower, self.upper)
        return current + np.clip(goal - current, -self.max_step, self.max_step)


class Workspace:
    """Forward kinematics of a command against the rig: ceiling for every link, floor and reach for the pen."""

    def __init__(self, kin, gates: Gates):
        self.kin, self.gates = kin, gates
        self.links = [i for i in range(kin.model.nbody) if kin.model.body(i).name.startswith("left/")]

    def violation(self, q: np.ndarray) -> str | None:
        gap, _ = self.kin.gap_pose(q)
        top = float(self.kin.data.xpos[self.links, 2].max())
        if top > self.gates.ceiling_m:
            return f"a link at {top:.3f} m, over the {self.gates.ceiling_m:.2f} m ceiling"
        if gap[2] < self.gates.floor_m:
            return f"the pen tip at {gap[2]:.3f} m, under the {self.gates.floor_m:.2f} m floor"
        reach = float(np.hypot(gap[0], gap[1]))
        if reach > self.gates.reach_m:
            return f"the pen tip {reach:.3f} m out, past the {self.gates.reach_m:.2f} m reach"
        return None


class Tracking:
    """A joint that stays far from its command is blocked (it hit something): end the run."""

    def __init__(self, gates: Gates):
        self.gates, self.since = gates, None

    def blocked(self, measured: np.ndarray, commanded: np.ndarray, now: float) -> bool:
        if np.max(np.abs(measured - commanded)) <= self.gates.track_rad:
            self.since = None
            return False
        self.since = now if self.since is None else self.since
        return now - self.since >= self.gates.track_s


class Guard:
    """The depth gate: hold under ``stop_mm``, abort after ``abort_s`` under ``abort_mm``."""

    def __init__(self, gates: Gates):
        self.gates, self.close_since = gates, None

    def check(self, clearance_m: float, now: float) -> str:
        """``ok``, ``hold`` or ``abort`` for this tick's clearance."""
        if clearance_m * 1000 >= self.gates.stop_mm:
            self.close_since = None
            return "ok"
        if clearance_m * 1000 < self.gates.abort_mm:
            self.close_since = now if self.close_since is None else self.close_since
            if now - self.close_since >= self.gates.abort_s:
                return "abort"
        else:
            self.close_since = None
        return "hold"


class Recorder:
    """The run directory: meta, per-tick and per-plan logs, the annotated video, the summary."""

    def __init__(self, root: Path, meta: dict):
        import cv2

        run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{socket.gethostname()}"
        self.dir = root.expanduser() / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "meta.json").write_text(json.dumps({"run_id": run_id, "argv": sys.argv, **meta}, indent=1) + "\n")
        self.ticks = (self.dir / "run.jsonl").open("w")
        self.plans = (self.dir / "plans.jsonl").open("w")
        self.video = cv2.VideoWriter(str(self.dir / "run.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))
        self.scene_video = None  # scene.mp4, opened at the first scene frame

    def tick(self, record: dict, frame_bgr: np.ndarray | None) -> None:
        self.ticks.write(json.dumps(record) + "\n")
        if frame_bgr is not None:
            self.video.write(frame_bgr)

    def scene(self, frame_bgr: np.ndarray) -> None:
        """The scene camera's frame for this tick, as it was (raw, for later two-view data)."""
        import cv2

        if self.scene_video is None:
            h, w = frame_bgr.shape[:2]
            self.scene_video = cv2.VideoWriter(str(self.dir / "scene.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                               (w, h))
        self.scene_video.write(frame_bgr)

    def plan(self, record: dict) -> None:
        self.plans.write(json.dumps(record) + "\n")

    def close(self, summary: dict) -> None:
        self.ticks.close()
        self.plans.close()
        self.video.release()
        if self.scene_video is not None:
            self.scene_video.release()
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


def planner_window(history: History, uses_scene: bool) -> tuple:
    """What a plan is computed from: (images, states, commands), and the scene images when the policy uses them."""
    return (*history.window(), history.scene_window()) if uses_scene else history.window()


def warm_up(planner, camera, q: np.ndarray, scene=None) -> int:
    """Two plans on the first frames (lazy loading, text cache); the second's time, in ticks."""
    uses_scene = "scene" in getattr(planner, "cameras", ["wrist"])
    while camera.newest() is None or (uses_scene and scene.newest() is None):
        time.sleep(0.05)
    history = History(planner.n_obs_steps)
    history.push(camera.newest()[0], q, q, scene=scene.newest()[0] if uses_scene else None)
    planner.plan_now(planner_window(history, uses_scene))
    _, seconds = planner.plan_now(planner_window(history, uses_scene))
    log.info("plans take %.2f s here (%d ticks)", seconds, int(np.ceil(seconds * FPS)))
    return int(np.ceil(seconds * FPS))


class CloseFrames:
    """The depth stop over distinct camera frames: ``frames`` in a row closer than ``stop_mm`` (one bad frame is not)."""

    def __init__(self, gates: Gates, frames: int = 3):
        self.gates, self.frames, self.count, self.last = gates, frames, 0, None

    def close(self, clearance_m: float, frame_t: float) -> bool:
        if frame_t != self.last:
            self.last = frame_t
            self.count = self.count + 1 if clearance_m * 1000 < self.gates.stop_mm else 0
        return self.count >= self.frames


class ApproachWatch:
    """Every tick of the move to the start pose: e-stop, tracking, depth and workspace; each tick recorded."""

    def __init__(self, arm, camera, overlay: Overlay, gates: Gates, recorder: Recorder | None):
        self.arm, self.camera, self.overlay, self.recorder = arm, camera, overlay, recorder
        self.workspace, self.tracking, self.depth = Workspace(overlay.kin, gates), Tracking(gates), CloseFrames(gates)
        self.frame = None

    def problem(self, sent: np.ndarray, step: np.ndarray) -> str | None:
        """Why the arm must not take ``step`` now, or None."""
        measured, self.frame = self.arm.measured(), self.camera.newest()
        clearance = float("inf")
        if self.frame is not None:
            clearance = axis_clearance(self.frame[1], self.overlay.lens_cam, self.overlay.axis_cam, self.overlay.intr)
        self._record(measured, sent, clearance)
        if self.arm.estopped:
            return "e-stop during the move to the start pose"
        if self.tracking.blocked(measured, sent, time.monotonic()):
            return "the arm stopped following its command on the way to the start pose: blocked"
        if self.frame is not None and self.depth.close(clearance, self.frame[2]):
            return f"something at {clearance * 1000:.0f} mm on the pen axis during the approach"
        outside = self.workspace.violation(step)
        return f"the move to the start pose would put {outside}" if outside is not None else None

    def _record(self, measured: np.ndarray, sent: np.ndarray, clearance: float) -> None:
        if self.recorder is None:
            return
        frame_bgr = None
        if self.frame is not None:
            self.overlay.cam_p, self.overlay.cam_r = self.overlay.kin.camera_pose(measured)
            frame_bgr = self.overlay.draw(np.ascontiguousarray(self.frame[0][..., ::-1]), None, clearance, "APPROACH")
        self.recorder.tick({"t": time.monotonic(), "phase": "approach", "q": np.round(measured, 5).tolist(),
                            "cmd": np.round(sent, 5).tolist(), "clearance_mm": round(clearance * 1000, 1)
                            if np.isfinite(clearance) else None}, frame_bgr)

    def keep_evidence(self) -> None:
        """The depth and colour frames the move stopped on, for the run directory."""
        if self.recorder is not None and self.frame is not None:
            np.save(self.recorder.dir / "approach_stop_depth.npy", self.frame[1])
            np.save(self.recorder.dir / "approach_stop_rgb.npy", self.frame[0])


def approach(arm, camera, overlay: Overlay, clamp: Clamp, q_start: np.ndarray, gates: Gates,
             recorder: Recorder | None = None) -> None:
    """Move slowly from the staged pose to the start pose, under the e-stop, tracking, depth and workspace gates."""
    sent = arm.measured()
    watch = ApproachWatch(arm, camera, overlay, gates, recorder)
    target = np.clip(q_start, clamp.lower, clamp.upper)  # the sim's park pose sits near a wrist limit
    outside = watch.workspace.violation(target)
    if outside is not None:
        raise RuntimeError(f"the start pose puts {outside}: choose a lower --pose")
    deadline = time.monotonic() + 30.0
    while np.max(np.abs(sent - target)) > 1e-3:
        step = clamp(sent, target)
        late = "the arm did not reach the start pose in 30 s" if time.monotonic() > deadline else None
        problem = late or watch.problem(sent, step)
        if problem is not None:
            arm.freeze()
            watch.keep_evidence()
            raise RuntimeError(problem)
        sent = step
        arm.command(sent, 2.0 / FPS)
        time.sleep(1.0 / FPS)
    time.sleep(0.5)


def untrained_joints(checkpoint: Path) -> list[int]:
    """Joints whose every training delta was zero: the policy never learned to move them, so its output there is
    noise (the quantile scaler falls back to a 1 rad span for a zero range). Read from the checkpoint's own stats."""
    files = sorted(Path(checkpoint).glob("policy_preprocessor_step_*_flux3_observation_history_normalizer.safetensors"))
    if not files:
        return []
    raw = files[0].read_bytes()  # safetensors: a little-endian header length, a JSON header, the raw tensors
    size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + size])

    def tensor(name: str) -> np.ndarray:
        start, end = header[name]["data_offsets"]
        dtype = {"F32": np.float32, "F64": np.float64}[header[name]["dtype"]]
        return np.frombuffer(raw[8 + size + start:8 + size + end], dtype=dtype)

    return [int(i) for i in np.flatnonzero(tensor("action.q99") - tensor("action.q01") <= 1e-6)]


class Loop:
    """The per-tick decision: history, plans, gates, command. ``arm`` None pins the state (a shadow run)."""

    def __init__(self, planner, camera, arm, q_start: np.ndarray, latency: int, clamp: Clamp, gates: Gates,
                 recorder: Recorder, drive: bool = True, held: list[int] | None = None, smooth_ticks: int = 7,
                 timing: str = "sync", scene=None):
        self.planner, self.camera, self.arm, self.gates, self.recorder = planner, camera, arm, gates, recorder
        self.scene = scene  # a third-person camera: recorded when given, fed to policies trained on one
        self.uses_scene = "scene" in getattr(planner, "cameras", ["wrist"])
        self.scene_age: float | None = None
        self.drive = arm is not None and drive  # False with an arm: held at the start pose, plans only drawn
        self.clamp, self.guard, self.tracking = clamp, Guard(gates), Tracking(gates)
        self.history = History(planner.n_obs_steps)
        self.scheduler = ChunkScheduler(latency, back_to_back=timing == "arrival", sync=timing == "sync",
                                        coast=0 if timing == "sync" else 4)
        self.timing, self.anchors = timing, {}  # the command in force at each request: its plan's integration start
        self.overlay = Overlay(q_start)
        self.workspace, self.outside_since = Workspace(self.overlay.kin, gates), None
        self.q_start, self.held, self.smooth_ticks = q_start, list(held or []), smooth_ticks
        self.sent = arm.measured() if arm is not None else q_start.copy()
        self.tick, self.path, self.latencies, self.clearances, self.holds = 0, None, [], [], 0

    def _scene(self, now: float) -> tuple[tuple[np.ndarray, float] | None, bool]:
        """This tick's scene frame (recorded when there is one) and whether it is fresh enough for the policy."""
        scene = self.scene.newest() if self.scene is not None else None
        if scene is not None:
            self.recorder.scene(np.ascontiguousarray(scene[0][..., ::-1]))
        self.scene_age = now - scene[1] if scene is not None else None
        fresh = not self.uses_scene or (scene is not None and self.scene_age <= SCENE_STALE_S)
        return scene, fresh

    def _take_plans(self) -> None:
        """Adopt arrived plans, conditioned, and played from now (``sync``, ``arrival``) or cut to the ticks ahead."""
        for start, commands, seconds in self.planner.take():
            plan = Plan(start, condition_plan(commands, self.q_start, self.held, self.smooth_ticks))
            anchor = self.anchors.pop(start, None)
            if self.timing in ("arrival", "sync") and anchor is not None:
                plan = plan.rebased(self.tick, anchor, self.sent)
            self.scheduler.adopt(plan, self.tick)
            ahead = plan.commands[max(0, self.tick - plan.start):][:32]
            self.path = self.overlay.gaps(ahead)  # world points once per plan, reprojected per tick
            self.latencies.append(seconds)
            self.recorder.plan({"tick": start, "arrived": self.tick, "latency_s": round(seconds, 3),
                                "commands": np.round(plan.commands, 4).tolist()})

    def measured(self) -> np.ndarray:
        return self.arm.measured() if self.arm is not None else self.q_start

    def step(self, now: float) -> str:
        """One tick; returns ``run``, or why the run ends (``estop``, ``abort``)."""
        if self.arm is not None and self.arm.estopped:
            self.arm.freeze()
            return "estop"
        measured, frame = self.measured(), self.camera.newest()
        if frame is None or now - frame[2] > self.gates.stale_s:
            self._hold(measured, "stale")
            return "run"
        scene, fresh = self._scene(now)
        if not fresh:
            self._hold(measured, "stale scene")
            return "run"
        rgb, depth, _ = frame
        clearance = axis_clearance(depth, self.overlay.lens_cam, self.overlay.axis_cam, self.overlay.intr)
        self.clearances.append(clearance)
        self.history.push(rgb, measured, self.sent, scene=scene[0] if self.uses_scene else None)
        self._take_plans()  # before asking again: the planner that just delivered is free this tick
        window = planner_window(self.history, self.uses_scene)
        if self.scheduler.due(self.tick) and self.planner.request(self.tick, window):
            self.scheduler.requested()
            self.anchors[self.tick] = self.sent.copy()
        gate = self.guard.check(clearance, now)
        if gate == "abort":
            self._hold(measured, "abort")
            return "abort"
        goal = measured if gate == "hold" else self.scheduler.command(self.tick, self.sent)
        self.holds += gate == "hold"
        if self.drive and self.tracking.blocked(measured, self.sent, now):
            self.arm.freeze()
            log.error("ending the run: a joint stopped following its command (blocked)")
            return "blocked"
        gate = self._advance(goal, now, gate)
        if gate == "workspace":
            self._hold(measured, "workspace")
            return "workspace"
        if self.drive:
            self.arm.command(self.sent, 2.0 / FPS)
        self._record(now, measured, clearance, gate, rgb)  # held: ``cmd`` is what would have been sent
        if self.arm is not None and not self.drive:
            self.sent = measured.copy()  # the command in force stays where the held arm stands
        self.tick += 1
        return "run"

    def _advance(self, goal: np.ndarray, now: float, gate: str) -> str:
        """Step the command toward ``goal`` unless that leaves the workspace; ``workspace`` ends a moving run."""
        step = self.clamp(self.sent, goal)
        outside = self.workspace.violation(step)
        if outside is None:
            self.sent, self.outside_since = step, None
            return gate
        self.outside_since = now if self.outside_since is None else self.outside_since
        if self.drive and now - self.outside_since >= self.gates.outside_s:
            log.error("ending the run: the policy keeps commanding %s", outside)
            return "workspace"
        return "outside"

    def _hold(self, measured: np.ndarray, why: str) -> None:
        if self.drive:
            self.sent = self.arm.freeze()
        elif self.arm is not None:
            self.sent = measured.copy()
        self.recorder.tick({"t": time.monotonic(), "tick": self.tick, "q": np.round(measured, 5).tolist(),
                            "state": why}, None)

    def _record(self, now: float, measured: np.ndarray, clearance: float, gate: str, rgb: np.ndarray) -> None:
        if self.arm is not None:
            self.overlay.cam_p, self.overlay.cam_r = self.overlay.kin.camera_pose(measured)
        path = self.overlay.project(self.path) if self.path is not None else None
        mode = "MOVE" if self.drive else "HOLD" if self.arm is not None else "SHADOW"
        label = f"t={self.tick / FPS:5.1f}s {mode} {gate}"
        frame = self.overlay.draw(np.ascontiguousarray(rgb[..., ::-1]), path, clearance, label)
        self.recorder.tick({"t": now, "tick": self.tick, "q": np.round(measured, 5).tolist(),
                            "cmd": np.round(self.sent, 5).tolist(), "clearance_mm": round(clearance * 1000, 1)
                            if np.isfinite(clearance) else None, "state": gate,
                            "scene_age_ms": round(self.scene_age * 1000) if self.scene_age is not None else None}, frame)

    def summary(self, why: str) -> dict:
        finite = [c for c in self.clearances if np.isfinite(c)]
        return {"ended": why, "ticks": self.tick, "plans": len(self.latencies), "hold_ticks": self.holds,
                "latency_median_s": float(np.median(self.latencies)) if self.latencies else None,
                "latency_max_s": float(np.max(self.latencies)) if self.latencies else None,
                "clearance_min_mm": float(np.min(finite) * 1000) if finite else None}


def run(planner, camera, arm, q_start: np.ndarray, seconds: float, gates: Gates, root: Path,
        drive: bool = True, held: list[int] | None = None, smooth_ticks: int = 7, timing: str = "sync",
        scene=None) -> dict:
    """The whole run; ``arm`` None is a shadow run, ``drive`` False holds the arm at the start pose.

    Always lands the arm (the summary's ``landing``), closes the camera and the planner.
    """
    recorder = Recorder(root, {"gates": asdict(gates), "arm": arm is not None, "drive": arm is not None and drive,
                               "q_start": q_start.tolist(), "estop": bool(arm is None or arm.estop_required),
                               "held_joints": list(held or []), "smooth_ticks": smooth_ticks, "timing": timing})
    why = "time"
    loop = None
    try:
        latency = warm_up(planner, camera, q_start, scene)
        if arm is not None:
            arm.connect(recorder.dir / "summary.json")  # the watchdog's, if it has to end this process
            lower, upper = arm.joint_limits(gates.margin)
            approach(arm, camera, Overlay(q_start), Clamp(lower, upper, gates.approach_speed / FPS), q_start, gates,
                     recorder)
            clamp = Clamp(lower, upper, gates.max_speed / FPS)
        else:
            clamp = Clamp(np.full(6, -np.inf), np.full(6, np.inf), gates.max_speed / FPS)
        loop = Loop(planner, camera, arm, q_start, latency, clamp, gates, recorder, drive=drive, held=held,
                    smooth_ticks=smooth_ticks, timing=timing, scene=scene)
        started = next_tick = time.monotonic()
        while time.monotonic() - started < seconds:
            why = loop.step(time.monotonic())
            if why != "run":
                break
            next_tick += 1.0 / FPS
            time.sleep(max(0.0, next_tick - time.monotonic()))
        else:
            why = "time"
    except KeyboardInterrupt:
        why = "interrupted"
    except RuntimeError as error:  # a refused start: the arm was frozen where the gate caught it
        log.error("%s", error)
        why = f"refused: {error}"
    finally:
        planner.join()
        landing = arm.land() if arm is not None else None
        camera.close()
        if scene is not None:
            scene.close()
        planner.close()
        summary = loop.summary(why) if loop is not None else {"ended": why}
        summary["carriage"] = float(arm.carriage) if arm is not None else None  # the pen rides it: sim has 0
        summary["landing"] = landing
        recorder.close(summary)
    return {"out": str(recorder.dir), **summary}
