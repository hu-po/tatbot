"""The scripted demonstrator: park, approach, trace, back off.

The expert knows everything the simulator knows -- where the phantom is, how
fast it moves, where hands are -- but it only acts on what the cameras could
show: it leaves the park pose when ink is in the wrist view, and it gives up
the ink when the ink leaves that view. Parked with too little ink in view, it
turns its base to face the ink, which the scene camera (and often the edge of
the wrist view) shows. That keeps its decisions learnable from the images the
policy will get.

Tracing walks the ink's stroke graph: along a stroke, on through a junction
(usually the straightest way on), and at a dead end either back the way it
came or over bare skin to the nearest other ink.

Every command passes through a rate-limited task-space target (gap point and
pen axis) and pen-pointing IK, or through a rate-limited joint target in the
park pose, so mode changes never step the arm.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from tatbot_travel.camera import Intrinsics
from tatbot_travel.clearance import SkinDistance
from tatbot_travel.ink import InkGraph
from tatbot_travel.kinematics import ArmKinematics
from tatbot_travel.lines import DrawnLine
from tatbot_travel.motion import Pose

DOWN = np.array([0.0, 0.0, -1.0])


class Mode(enum.IntEnum):
    PARK = 0
    APPROACH = 1
    TRACE = 2
    BACKOFF = 3


@dataclass(frozen=True)
class ExpertConfig:
    gap_m: float = 0.020  # lens face to skin, along the pen
    approach_hover_m: float = 0.06
    backoff_m: float = 0.085
    wary_m: float = 0.045  # hover while hands are on the phantom: they are about to move it
    guard_m: float = 0.012  # nose-to-skin clearance that triggers an immediate escape
    closing_horizon_s: float = 0.3  # back off when the skin would close to the guard clearance this soon
    trace_clearance_m: float = 0.016  # a stroke station is traceable only with this much nose clearance
    retreat_until_m: float = 0.03  # backing off, retreat from the skin until the nose is this clear (the park
    # pose can sit 4 cm from the phantom, so more would retreat from park itself) ...
    # Waits the camera cannot show (settling, cooldowns, grace periods) are kept short: while they run the
    # expert holds still in states that look like the ones it moves in, and v7's data held still on 44 % of
    # its frames, two thirds of them with no hand on the phantom -- a policy trained on that hesitates.
    flinch_cooldown_s: float = 0.7  # no re-approach this soon after a flinch
    rise_first_m: float = 0.03  # approaching from below the hover height: rise before crossing over
    ceiling_m: float = 0.45  # no link above this over the arm's base (a rig's overhead cameras hang at 0.60 m)
    max_tilt_deg: float = 35.0  # pen axis from vertical
    max_tilt_out_deg: float = 15.0  # ... when it leans away from the arm's base, which the arm reaches badly
    trace_speed: tuple[float, float] = (0.015, 0.04)  # m/s along the ink
    end_dwell_s: tuple[float, float] = (0.1, 0.4)
    target_gain: float = 7.0  # 1/s, first-order pull of the commanded target
    speed_approach: float = 0.10
    speed_trace: float = 0.15
    speed_backoff: float = 0.35  # escaping is the one fast motion
    axis_rate_deg: float = 90.0
    # One speed for every motion, the real runner's cap: data faster than the rig executes teaches plans the
    # rig truncates (at 0.9/2.0 rad/s, 36 % of v4's ticks were over 0.25 rad/s on some joint).
    joint_speed: float = 0.25  # rad/s, any joint
    escape_joint_speed: float = 0.25  # rad/s while backing off
    park_speed: float = 0.25
    fast_linear: float = 0.18  # phantom motion that makes the expert back off
    fast_angular: float = 1.2
    slow_linear: float = 0.08  # and motion slow enough to resume or keep tracing
    slow_angular: float = 0.45
    hand_clearance: float = 0.07
    reach_step_m: float = 0.008  # spacing of the reachability check along a stroke
    straight_prob: float = 0.75  # at a junction, take the least traced, straightest way on this often
    jump_prob: float = 0.4  # at a dead end or with all ways on traced, hop to other ink this often
    jump_radius_m: float = 0.06
    hop_m: float = 0.012  # a jump this short stays on the skin; longer ones rise to the approach hover
    look_radius_m: float = 0.08  # ink this close to the attended point decides what is in view
    entry_candidates: int = 4  # strokes in view, nearest the pen first, tried as an entry
    min_interval_m: float = 0.015  # shortest reachable stretch worth approaching (or 60% of a shorter stroke)
    cooldown_s: float = 1.0  # after giving up on unreachable ink
    visible_enter: float = 0.4
    visible_lost: float = 0.15
    settle_s: float = 0.2
    visible_for_s: float = 0.1  # ink in view this long before committing to it
    lost_s: float = 0.5
    park_after_s: float = 0.5
    scan_amplitude: float = 0.6  # rad on the base joint while parked with the phantom taken away
    scan_period_s: float = 9.0
    look_after_s: float = 0.3  # parked this long with too little ink in view: turn the base to face the ink
    look_approach_s: float = 0.5  # ... then this long facing it: approach it anyway (the scene camera shows it;
    # from park the wrist camera looks out over the table, so ink lying close or low stays out of its view)


@dataclass
class Observation:
    """What the expert may use this tick (privileged, but decisions are gated on visibility)."""

    t: float
    q: np.ndarray  # measured joints
    phantom: Pose
    linear_speed: float
    angular_speed: float
    away: bool
    hands: list[np.ndarray]  # world positions of hand centres
    held: bool = False  # hands are on the phantom
    clearance: float = 1.0  # m, the pen nose to the skin at the measured arm pose
    closing_speed: float = 0.0  # m/s, how fast the phantom is pushing into the pen faster than the arm gives way


@dataclass
class ExpertState:
    mode: Mode = Mode.PARK
    s: float = 0.0
    direction: float = 1.0
    dwell: float = 0.0
    standoff: float = 0.085
    q_cmd: np.ndarray | None = None
    gap_cmd: np.ndarray | None = None
    axis_cmd: np.ndarray | None = None
    timers: dict = field(default_factory=lambda: {"slow": 0.0, "visible": 0.0, "lost": 0.0, "absent": 0.0,
                                                  "unseen": 0.0,
                                                  "cooldown": 0.0, "reach_age": 0.0})
    edge: int = 0  # the stroke being traced
    visits: dict[int, int] = field(default_factory=dict)  # strokes traced to an end, and how often
    interval: tuple[float, float] = (0.0, 0.0)  # reachable stretch of that stroke
    ik_fail: int = 0
    visible: float = 0.0
    ik_error: float = 0.0
    ceiling: bool = False  # the last command was levelled or held at the ceiling
    look_j0: float | None = None  # the base angle the park pose faces, once the expert has turned to the ink


def tilt_limited(axis: np.ndarray, max_tilt: float) -> np.ndarray:
    """Rotate a pen axis toward straight down until it is within ``max_tilt`` of it."""
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(axis @ DOWN, -1.0, 1.0))
    if angle <= max_tilt:
        return axis
    perp = axis - DOWN * (axis @ DOWN)
    norm = np.linalg.norm(perp)
    perp = perp / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    return np.cos(max_tilt) * DOWN + np.sin(max_tilt) * perp


def _toward(current: np.ndarray, goal: np.ndarray, gain: float, max_speed: float, dt: float) -> np.ndarray:
    step = gain * (goal - current) * dt
    limit = max_speed * dt
    norm = np.linalg.norm(step)
    return current + (step if norm <= limit else step * (limit / norm))


def _rotate_toward(current: np.ndarray, goal: np.ndarray, gain: float, max_rate: float, dt: float) -> np.ndarray:
    angle = float(np.arccos(np.clip(current @ goal, -1.0, 1.0)))
    if angle < 1e-6:
        return goal.copy()
    step = min(gain * angle * dt, max_rate * dt, angle)
    axis = np.cross(current, goal)
    axis /= max(np.linalg.norm(axis), 1e-12)
    rotated = current * np.cos(step) + np.cross(axis, current) * np.sin(step) + axis * (axis @ current) * (
        1 - np.cos(step))
    return rotated / np.linalg.norm(rotated)


class Expert:
    def __init__(self, cfg: ExpertConfig, kin: ArmKinematics, ink: InkGraph, intr: Intrinsics,
                 occluded: np.ndarray, rng: np.random.Generator, q_park: np.ndarray, q_rest: np.ndarray,
                 skin: SkinDistance, lens_site: int, base_z: float = 0.0):
        self.cfg, self.kin, self.ink, self.intr = cfg, kin, ink, intr
        self.skin, self.lens_site = skin, lens_site
        self.occluded = occluded  # (H, W) bool: the cradle hides these pixels
        self.rng = rng
        self.q_park, self.q_rest = q_park, q_rest
        self.park_gap, self.park_axis = kin.gap_pose(q_park)  # where a retreat folds back to
        self.speed = float(rng.uniform(*cfg.trace_speed))
        self.dt = 1.0 / 30.0
        self.state = ExpertState(standoff=cfg.backoff_m)
        self.ceiling_z = base_z + cfg.ceiling_m
        self.links = [i for i in range(kin.model.nbody) if kin.model.body(i).name.startswith("left/")]

    @property
    def line(self) -> DrawnLine:
        """The stroke being traced."""
        return self.ink.edges[self.state.edge]

    # ---- perception the expert allows itself -------------------------------------------------
    def attended(self, q: np.ndarray, phantom: Pose) -> int:
        """The ink point the expert attends to: nearest the pen, or while parked nearest the view's axis."""
        st = self.state
        if st.mode != Mode.PARK and st.gap_cmd is not None:
            local = phantom.rot.inv().apply(st.gap_cmd - phantom.pos)
            return int(np.argmin(np.linalg.norm(self.ink.points - local, axis=1)))
        cam_p, cam_r = self.kin.camera_pose(q)
        origin = phantom.rot.inv().apply(cam_p - phantom.pos)
        ray = phantom.rot.inv().apply(cam_r[:, 2])
        rel = self.ink.points - origin
        depth = rel @ ray
        off_axis = np.linalg.norm(rel - depth[:, None] * ray, axis=1)
        return int(np.argmin(np.where(depth > 0.0, off_axis, np.inf)))

    def _watched(self, q: np.ndarray, phantom: Pose) -> np.ndarray:
        """Ink samples that decide visibility: those around the attended point."""
        anchor = self.ink.points[self.attended(q, phantom)]
        near = np.flatnonzero(np.linalg.norm(self.ink.points - anchor, axis=1) < self.cfg.look_radius_m)
        return near[np.linspace(0, len(near) - 1, min(40, len(near))).astype(int)]

    def in_view(self, q: np.ndarray, phantom: Pose, indices: np.ndarray) -> np.ndarray:
        """Which of these ink points the wrist camera sees: in frame, facing it, not behind the cradle."""
        cam_p, cam_r = self.kin.camera_pose(q)
        pts = phantom.apply(self.ink.points[indices])
        nrm = phantom.rot.apply(self.ink.normals[indices])
        local = (pts - cam_p) @ cam_r
        uv = self.intr.project(local)
        h, w = self.occluded.shape
        inside = (local[:, 2] > 0.05) & (uv[:, 0] > 8) & (uv[:, 0] < w - 8) & (uv[:, 1] > 8) & (uv[:, 1] < h - 8)
        facing = ((cam_p - pts) * nrm).sum(axis=1) > 0
        ok = inside & facing
        rows = np.clip(uv[:, 1].astype(int), 0, h - 1)
        cols = np.clip(uv[:, 0].astype(int), 0, w - 1)
        return ok & ~self.occluded[rows, cols]

    def visible_fraction(self, q: np.ndarray, phantom: Pose) -> float:
        return float(self.in_view(q, phantom, self._watched(q, phantom)).mean())

    def line_target(self, phantom: Pose, s: float, standoff: float,
                    edge: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Gap-point target and pen axis over stroke ``edge`` (default: the one being traced) at ``s``."""
        line = self.line if edge is None else self.ink.edges[edge]
        point, normal, _ = line.at(s)
        p = phantom.apply(point[None])[0]
        axis = -phantom.rot.apply(normal)
        # A pen leaning away from the base (its tip further out than its tail) is hard to reach.
        outward = float(axis[:2] @ p[:2]) > 0.0
        limit = self.cfg.max_tilt_out_deg if outward else self.cfg.max_tilt_deg
        axis = tilt_limited(axis, np.radians(limit))
        return p - axis * (standoff - self.cfg.gap_m), axis

    def clearance(self, q: np.ndarray, phantom: Pose) -> float:
        _, axis = self.kin.gap_pose(q)
        return self.skin.pen(phantom, self.kin.data.site_xpos[self.lens_site].copy(), axis)

    def traceable_length(self, phantom: Pose, strokes: int = 3) -> float:
        """Longest reachable, clear stretch around the middle of the longest strokes, from the tracing posture."""
        st = self.state
        saved = (st.q_cmd, st.edge)
        best = 0.0
        for edge in np.argsort([-e.length for e in self.ink.edges])[:strokes]:
            st.q_cmd, st.edge = self.q_rest, int(edge)
            lo, hi = self.reach_interval(phantom, 0.5 * self.line.length)
            best = max(best, hi - lo)
        st.q_cmd, st.edge = saved
        return best

    def reach_interval(self, phantom: Pose, s_entry: float) -> tuple[float, float]:
        """The reachable stretch of the stroke around ``s_entry`` (empty if the entry is out of reach)."""
        stations = np.linspace(0.0, self.line.length, max(2, int(np.ceil(self.line.length / self.cfg.reach_step_m)) + 1))
        k0 = int(np.argmin(np.abs(stations - s_entry)))
        ok = np.zeros(len(stations), dtype=bool)
        for order in (range(k0, len(stations)), range(k0, -1, -1)):
            q = self.state.q_cmd
            for k in order:
                goal, axis = self.line_target(phantom, float(stations[k]), self.cfg.gap_m)
                result = self.kin.solve(q, goal, axis, self.q_rest, iters=25)
                if not result.ok(0.003, 0.08) or self.clearance(result.q, phantom) < self.cfg.trace_clearance_m:
                    break
                ok[k], q = True, result.q
        if not ok[k0]:
            return (s_entry, s_entry)
        lo, hi = k0, k0
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(stations) - 1 and ok[hi + 1]:
            hi += 1
        return float(stations[lo]), float(stations[hi])

    def entries(self, q: np.ndarray, phantom: Pose, in_view: bool = True) -> list[tuple[int, float]]:
        """Candidate (stroke, arclength) entries: the ink (in view, unless not asked) nearest the pen, one per
        stroke."""
        indices = np.arange(0, len(self.ink.points), 3)
        if in_view:
            indices = indices[self.in_view(q, phantom, indices)]
        local = phantom.rot.inv().apply(self.state.gap_cmd - phantom.pos)
        found: dict[int, float] = {}
        for k in indices[np.argsort(np.linalg.norm(self.ink.points[indices] - local, axis=1))]:
            edge, i = self.ink.index[k]
            found.setdefault(int(edge), float(self.ink.edges[edge].arclength[i]))
            if len(found) == self.cfg.entry_candidates:
                break
        return list(found.items())

    # ---- the mode machine --------------------------------------------------------------------
    def reset(self, q: np.ndarray, mode: Mode = Mode.PARK, edge: int | None = None,
              s: float | None = None) -> None:
        st = self.state
        st.mode, st.q_cmd = mode, np.asarray(q, dtype=float).copy()
        st.gap_cmd, st.axis_cmd = self.kin.gap_pose(st.q_cmd)
        st.edge = self.pick_edge() if edge is None else edge
        st.s = self.line.length * self.rng.random() if s is None else s
        st.direction = float(self.rng.choice([-1.0, 1.0]))
        st.standoff = self.cfg.gap_m if mode == Mode.TRACE else self.cfg.backoff_m
        st.interval = (0.0, self.line.length)
        st.visits = {}
        st.look_j0 = None

    def pick_edge(self) -> int:
        """A stroke drawn with probability proportional to its length."""
        lengths = np.array([e.length for e in self.ink.edges])
        return int(self.rng.choice(len(lengths), p=lengths / lengths.sum()))

    def follow(self, q: np.ndarray) -> None:
        """Adopt a command someone else issued, so the next decision starts from it (DAgger labels)."""
        st = self.state
        st.q_cmd = np.asarray(q, dtype=float).copy()
        st.gap_cmd, st.axis_cmd = self.kin.gap_pose(st.q_cmd)

    def _update_timers(self, obs: Observation) -> None:
        st, cfg, dt = self.state, self.cfg, self.dt
        slow = obs.linear_speed < cfg.slow_linear and obs.angular_speed < cfg.slow_angular and not obs.away
        st.timers["slow"] = st.timers["slow"] + dt if slow else 0.0
        st.timers["visible"] = st.timers["visible"] + dt if st.visible >= cfg.visible_enter else 0.0
        st.timers["lost"] = st.timers["lost"] + dt if st.visible < cfg.visible_lost else 0.0
        st.timers["absent"] = st.timers["absent"] + dt if st.visible == 0.0 or obs.away else 0.0
        unseen = st.visible < cfg.visible_enter and not obs.away
        st.timers["unseen"] = st.timers["unseen"] + dt if unseen else 0.0
        st.timers["cooldown"] = max(0.0, st.timers["cooldown"] - dt)
        st.timers["reach_age"] += dt

    def _hand_near(self, obs: Observation) -> bool:
        if not obs.hands or self.state.gap_cmd is None:
            return False
        return min(np.linalg.norm(h - self.state.gap_cmd) for h in obs.hands) < self.cfg.hand_clearance

    def _cornered(self, obs: Observation) -> bool:
        """The skin is under the guard clearance, or closing on the pen faster than it can wait."""
        cfg = self.cfg
        return obs.clearance < cfg.guard_m or obs.closing_speed * cfg.closing_horizon_s > obs.clearance - cfg.guard_m

    def _must_back_off(self, obs: Observation) -> bool:
        cfg, st = self.cfg, self.state
        fast = obs.linear_speed > cfg.fast_linear or obs.angular_speed > cfg.fast_angular
        # Ink lost from view ends tracing, and an approach once it comes down onto the ink; on the way over
        # the ink may not be in view yet.
        descending = st.mode == Mode.TRACE or st.standoff <= cfg.gap_m
        lost = st.timers["lost"] >= cfg.lost_s and descending
        return fast or obs.away or lost or self._cornered(obs) or self._hand_near(obs)

    def _enough(self, interval: tuple[float, float], need: float) -> bool:
        """A reachable stretch worth committing to: ``need`` long, or most of a shorter stroke."""
        return interval[1] - interval[0] >= min(need, 0.6 * self.line.length)

    def _enter(self, obs: Observation, edge: int, s: float, need: float) -> bool:
        """Take stroke ``edge`` at ``s`` if enough of it around there is reachable (state kept otherwise)."""
        st = self.state
        saved = st.edge
        st.edge = edge
        interval = self.reach_interval(obs.phantom, s)
        if not self._enough(interval, need):
            st.edge = saved
            return False
        st.interval, st.s = interval, float(np.clip(s, *interval))
        st.timers["reach_age"] = 0.0
        return True

    def begin_approach(self, phantom: Pose, s: float, need: float) -> bool:
        """Commit to the current stroke at ``s`` from wherever the arm is, as ``_try_approach`` would."""
        st = self.state
        interval = self.reach_interval(phantom, s)
        if not self._enough(interval, need):
            return False
        st.interval, st.s = interval, float(np.clip(s, *interval))
        st.mode, st.standoff = Mode.APPROACH, max(st.standoff, self.cfg.approach_hover_m)
        st.timers["reach_age"] = 0.0
        return True

    def _try_approach(self, obs: Observation, in_view: bool = True) -> None:
        """Commit to ink (in view, unless not asked) if a long enough stretch of it is reachable."""
        st, cfg = self.state, self.cfg
        for edge, s in self.entries(obs.q, obs.phantom, in_view):
            if self._enter(obs, edge, s, cfg.min_interval_m):
                st.mode = Mode.APPROACH
                st.standoff = max(st.standoff, cfg.approach_hover_m)
                return
        st.timers["cooldown"] = cfg.cooldown_s

    def _transition(self, obs: Observation) -> None:
        st, cfg = self.state, self.cfg
        ready = (st.timers["slow"] >= cfg.settle_s and not self._hand_near(obs) and st.timers["cooldown"] <= 0.0
                 and st.visible >= cfg.visible_enter)
        if st.mode in (Mode.APPROACH, Mode.TRACE) and self._must_back_off(obs):
            st.mode, st.ik_fail = Mode.BACKOFF, 0
            if obs.clearance < cfg.guard_m:
                st.timers["cooldown"] = cfg.flinch_cooldown_s
        elif st.mode in (Mode.APPROACH, Mode.TRACE) and st.ik_fail >= 6:
            st.mode, st.ik_fail, st.timers["cooldown"] = Mode.BACKOFF, 0, cfg.cooldown_s
        elif st.mode == Mode.PARK and self._cornered(obs):
            st.mode = Mode.BACKOFF  # something was brought to the parked pen: get out of its way
            st.timers["cooldown"] = cfg.flinch_cooldown_s
        elif st.mode == Mode.BACKOFF and (max(st.timers["absent"], st.timers["unseen"]) >= cfg.park_after_s
                                          or st.ik_fail >= 30):
            st.mode, st.ik_fail = Mode.PARK, 0  # too little ink in view to come back to: park, and face it
        elif st.mode in (Mode.PARK, Mode.BACKOFF) and ready and st.timers["visible"] >= cfg.visible_for_s:
            self._try_approach(obs)
        elif st.mode == Mode.PARK and self._faced_unseen(obs):
            self._try_approach(obs, in_view=False)

    def _faced_unseen(self, obs: Observation) -> bool:
        """Parked facing the ink, calm and unhindered, and still not seeing enough of it."""
        st, cfg = self.state, self.cfg
        return (st.look_j0 is not None and abs(obs.q[0] - st.look_j0) < 0.05
                and st.timers["unseen"] >= cfg.look_after_s + cfg.look_approach_s
                and st.timers["slow"] >= cfg.settle_s and not self._hand_near(obs) and st.timers["cooldown"] <= 0.0)

    def _refresh_interval(self, obs: Observation) -> None:
        """While the phantom moves under a trace, the reachable stretch moves with it."""
        st = self.state
        if st.timers["reach_age"] >= 0.5 and obs.linear_speed + obs.angular_speed > 0.005:
            st.interval = self.reach_interval(obs.phantom, st.s)
            st.timers["reach_age"] = 0.0

    def _ways_on(self) -> list[tuple[int, bool]]:
        """Strokes leaving the node reached, least traced first, then straightest; sometimes shuffled."""
        st = self.state
        options = self.ink.continuations(st.edge, at_far_end=st.direction > 0)
        _, _, tangent = self.line.at(st.s)
        heading = tangent * st.direction

        def rank(option: tuple[int, bool]) -> tuple[int, float]:
            line = self.ink.edges[option[0]]
            _, _, t = line.at(0.0 if option[1] else line.length)
            return st.visits.get(option[0], 0), -float(heading @ (t if option[1] else -t))

        options.sort(key=rank)
        if self.rng.random() >= self.cfg.straight_prob:
            self.rng.shuffle(options)
        return options

    def _next_stroke(self, obs: Observation) -> bool:
        """At a stroke's end: on through the node, or over to other ink once the way on is all traced."""
        st, cfg = self.state, self.cfg
        st.visits[st.edge] = st.visits.get(st.edge, 0) + 1
        options = self._ways_on()
        traced = all(st.visits.get(edge, 0) > 0 for edge, _ in options)
        if traced and self.rng.random() < cfg.jump_prob and self._jump(obs):
            return True
        for edge, forward in options:
            if self._enter(obs, edge, 0.0 if forward else self.ink.edges[edge].length, cfg.reach_step_m):
                st.direction = 1.0 if forward else -1.0
                return True
        return False

    def _jump(self, obs: Observation) -> bool:
        """Hop over bare skin to the nearest other ink: along the skin if close, else through the approach."""
        st, cfg = self.state, self.cfg
        here, _, _ = self.line.at(st.s)
        linked = {st.edge} | {e for end in (True, False) for e, _ in self.ink.continuations(st.edge, end)}
        edge, s, dist = self.ink.nearest_other(here, linked)
        if edge < 0 or dist > cfg.jump_radius_m or not self._enter(obs, edge, s, cfg.min_interval_m):
            return False
        st.direction = float(self.rng.choice([-1.0, 1.0]))
        if dist > cfg.hop_m:  # lift over the gap, higher for longer hops
            st.mode = Mode.APPROACH
            st.standoff = cfg.gap_m + min(cfg.approach_hover_m - cfg.gap_m, 0.01 + 0.5 * dist)
        return True

    def _advance_trace(self, obs: Observation) -> None:
        st = self.state
        if st.dwell > 0.0:
            st.dwell -= self.dt
            return
        lo, hi = st.interval
        st.s += st.direction * self.speed * self.dt
        if lo < st.s < hi and st.ik_fail < 3:
            return
        st.s = float(np.clip(st.s, lo, hi))
        stroke_end = st.s >= self.line.length - 1e-6 if st.direction > 0 else st.s <= 1e-6
        if stroke_end and st.ik_fail < 3 and self._next_stroke(obs):
            return
        st.direction *= -1.0
        st.dwell = float(self.rng.uniform(*self.cfg.end_dwell_s))
        st.ik_fail = 0

    def _task_goal(self, obs: Observation) -> tuple[np.ndarray, np.ndarray, float]:
        """Desired gap point, axis and speed limit for the task-space modes."""
        st, cfg = self.state, self.cfg
        if st.mode == Mode.APPROACH:
            goal, axis = self.line_target(obs.phantom, st.s, st.standoff)
            horizontal = np.linalg.norm((goal - st.gap_cmd)[:2])
            rise = st.gap_cmd[2] < goal[2] + cfg.rise_first_m and not st.ceiling  # no rising into the ceiling
            if st.standoff > cfg.gap_m and horizontal > 0.03 and rise:
                # Coming from below the hover height: rise first, then cross over the phantom. The pen
                # turns to its tracing axis as it rises, which keeps the wrist low.
                return st.gap_cmd + np.array([0.0, 0.0, 0.05]), axis, cfg.speed_approach
            if st.standoff > cfg.gap_m and np.linalg.norm(goal - st.gap_cmd) < 0.01:
                st.standoff = cfg.gap_m  # hovering above the entry point: come down
            elif st.standoff <= cfg.gap_m and np.linalg.norm(goal - st.gap_cmd) < 0.003:
                st.mode = Mode.TRACE
            return goal, axis, cfg.speed_approach
        if st.mode == Mode.TRACE:
            self._refresh_interval(obs)
            self._advance_trace(obs)
            if st.mode != Mode.TRACE:  # jumped to far ink: approach it from above
                return self._task_goal(obs)
            goal, axis = self.line_target(obs.phantom, st.s, cfg.wary_m if obs.held else cfg.gap_m)
            return goal, axis, cfg.speed_trace
        st.standoff = cfg.backoff_m
        calm = obs.linear_speed < cfg.slow_linear and obs.angular_speed < cfg.slow_angular
        if obs.clearance < cfg.retreat_until_m:
            # Retreat from where the pen is -- back along its axis and up -- rather than chasing
            # a phantom that is moving toward it.
            goal = st.gap_cmd - st.axis_cmd * 0.05 + np.array([0.0, 0.0, 0.03])
            return goal, st.axis_cmd, cfg.speed_backoff
        if obs.away or not calm:
            # ... then fold back to the park pose, low beside the phantom, rather than climbing on.
            return self.park_gap.copy(), self.park_axis.copy(), cfg.speed_backoff
        # Clear and calm: hold. Backing off never closes the distance; approaching does that.
        return st.gap_cmd, st.axis_cmd, cfg.speed_backoff

    def under_ceiling(self, q: np.ndarray) -> bool:
        """Every link of the arm at ``q`` under the ceiling."""
        self.kin.gap_pose(q)
        return float(self.kin.data.xpos[self.links, 2].max()) <= self.ceiling_z

    def ink_bearing(self, phantom: Pose) -> float:
        """The base angle that faces the ink (joint 0 at zero heads the arm along +x)."""
        centre = phantom.apply(self.ink.points).mean(axis=0)
        return float(np.clip(np.arctan2(centre[1], centre[0]), self.kin.lower[0], self.kin.upper[0]))

    def _park_command(self, obs: Observation) -> np.ndarray:
        st, cfg = self.state, self.cfg
        if obs.away:
            st.look_j0 = None
        elif st.timers["unseen"] > cfg.look_after_s:
            st.look_j0 = self.ink_bearing(obs.phantom)
        target = self.q_park.copy()
        if st.look_j0 is not None:
            target[0] = st.look_j0
        elif st.timers["absent"] > 1.0:
            target[0] += cfg.scan_amplitude * np.sin(2 * np.pi * obs.t / cfg.scan_period_s)
        step = st.q_cmd + np.clip(target - st.q_cmd, -cfg.park_speed * self.dt, cfg.park_speed * self.dt)
        if self.under_ceiling(step) or not self.under_ceiling(st.q_cmd):
            return step
        # The straight way back rises into the ceiling (from a raised tracing posture): move the joints that do
        # not, furthest from park first -- turning the base never rises. v6/v7's expert held still here.
        for j in np.argsort(-np.abs(step - st.q_cmd)):
            partial = st.q_cmd.copy()
            partial[j] = step[j]
            if partial[j] != st.q_cmd[j] and self.under_ceiling(partial):
                return partial
        return st.q_cmd.copy()

    def _task_command(self, obs: Observation) -> np.ndarray:
        st, cfg, dt = self.state, self.cfg, self.dt
        goal, axis, speed = self._task_goal(obs)
        gap_next = _toward(st.gap_cmd, goal, cfg.target_gain, speed, dt)
        axis_next = _rotate_toward(st.axis_cmd, axis, cfg.target_gain, np.radians(cfg.axis_rate_deg), dt)
        result = self.kin.solve(st.q_cmd, gap_next, axis_next, self.q_rest, iters=8)
        st.ik_error = result.position_error
        if not result.ok(0.005, 0.1):
            st.ik_fail += 1
            q_next = st.q_cmd.copy()  # out of reach or stuck at a stop: hold ...
            if st.mode == Mode.BACKOFF:  # ... unless retreating: head for the park pose where that opens the gap
                parkward = st.q_cmd + np.clip(self.q_park - st.q_cmd, -cfg.joint_speed * dt, cfg.joint_speed * dt)
                if (self.clearance(parkward, obs.phantom) > self.clearance(st.q_cmd, obs.phantom)
                        and self.under_ceiling(parkward)):
                    q_next = parkward
            st.gap_cmd, st.axis_cmd = self.kin.gap_pose(q_next)  # keep the target from running away
            return q_next
        limit = (cfg.escape_joint_speed if st.mode == Mode.BACKOFF else cfg.joint_speed) * dt
        q_next = st.q_cmd + np.clip(result.q - st.q_cmd, -limit, limit)
        st.ceiling = not self.under_ceiling(q_next)
        if st.ceiling:  # e.g. a retreat rising into the rig: go level instead
            level = self._level_step(gap_next, axis_next, speed, limit)
            if level is None:
                st.ik_fail += 1
                st.gap_cmd, st.axis_cmd = self.kin.gap_pose(st.q_cmd)
                return st.q_cmd.copy()
            q_next, gap_next = level
        st.ik_fail = 0
        st.gap_cmd, st.axis_cmd = gap_next, axis_next
        return q_next

    def _level_step(self, gap_next: np.ndarray, axis_next: np.ndarray, speed: float,
                    limit: float) -> tuple[np.ndarray, np.ndarray] | None:
        """The step with its rise taken out (and, backing off, drawn in toward the base): under the ceiling."""
        st = self.state
        level = gap_next.copy()
        level[2] = min(gap_next[2], st.gap_cmd[2])
        if st.mode == Mode.BACKOFF:
            inward = -np.array([st.gap_cmd[0], st.gap_cmd[1], 0.0])
            level += inward / max(np.linalg.norm(inward), 1e-9) * speed * self.dt
        result = self.kin.solve(st.q_cmd, level, axis_next, self.q_rest, iters=8)
        if not result.ok(0.005, 0.1):
            return None
        q_next = st.q_cmd + np.clip(result.q - st.q_cmd, -limit, limit)
        return (q_next, level) if self.under_ceiling(q_next) else None

    def _safe(self, q_next: np.ndarray, obs: Observation) -> np.ndarray:
        """Under the guard clearance, take the clearest of the command, a step away from the skin, or holding."""
        cfg, st = self.cfg, self.state
        after = self.clearance(q_next, obs.phantom)
        if after >= cfg.guard_m:
            return q_next
        here, axis = self.kin.gap_pose(st.q_cmd)  # step away from where the arm was commanded, not the target
        away = self.skin.escape(obs.phantom, self.kin.data.site_xpos[self.lens_site].copy(), axis)
        result = self.kin.solve(st.q_cmd, here + away * cfg.speed_backoff * self.dt, axis, self.q_rest, iters=8)
        limit = cfg.escape_joint_speed * self.dt
        escape = st.q_cmd + np.clip(result.q - st.q_cmd, -limit, limit)
        options = [q for q in (q_next, escape) if self.under_ceiling(q)] + [st.q_cmd.copy()]
        clear = [self.clearance(q, obs.phantom) for q in options]
        return options[int(np.argmax(clear))]

    def act(self, obs: Observation) -> np.ndarray:
        """The next joint command, from this tick's observation."""
        st = self.state
        st.visible = self.visible_fraction(obs.q, obs.phantom)
        self._update_timers(obs)
        self._transition(obs)
        if st.mode == Mode.PARK:
            q_next = self._park_command(obs)
            st.ik_error = 0.0
        else:
            q_next = self._task_command(obs)
        q_safe = self._safe(q_next, obs)
        st.q_cmd = q_safe
        if st.mode == Mode.PARK or q_safe is not q_next:
            # The commanded target follows the arm when it was parked or pulled clear of the skin.
            st.gap_cmd, st.axis_cmd = self.kin.gap_pose(st.q_cmd)
        return st.q_cmd.copy()
