"""How people handle the phantom during a demo, as a keyframed script.

The phantom mostly lies on the table at some heading in front of the arm.
Visitors nudge it, pick it up and put it down elsewhere, hold it up and move
it slowly to see the robot follow, or take it away and bring it back. Each
of those is a segment of keyframes; poses between keyframes follow min-jerk
timing, and hands grip the phantom while it is being moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tatbot_travel.phantom import Phantom
from tatbot_travel.tabletop import TabletopItem, support_height

TABLE_Z = 0.0
AWAY = np.array([0.15, -0.9, 0.5])  # where a phantom that was taken away waits: out of reach and view


@dataclass(frozen=True)
class Pose:
    pos: np.ndarray
    rot: Rotation

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.rot.apply(points) + self.pos

    def quat_wxyz(self) -> np.ndarray:
        x, y, z, w = self.rot.as_quat()
        return np.array([w, x, y, z])


@dataclass(frozen=True)
class Keyframe:
    t: float
    pose: Pose
    held: bool  # hands are on the phantom while it travels toward this keyframe
    away: bool = False


@dataclass(frozen=True)
class PlacementConfig:
    # The rig's practice arm lies across the table just in front of the parked pen: measured from the
    # wrist camera, it runs from 0.25 m out on the right to 0.5 m out on the left, 42 deg off the arm's
    # heading. Its heading here stays anywhere, and it may lie well off to either side (on the rig it has
    # lain alongside the pen, its ink at the edge of the parked view): the arm must turn to find it.
    radius: tuple[float, float] = (0.25, 0.45)  # forearm centre distance from the arm base
    azimuth_deg: tuple[float, float] = (-45.0, 45.0)
    back_up_prob: float = 0.82  # lying palm down, the back of the forearm up
    palm_up_prob: float = 0.10  # otherwise on its side
    roll_jitter_deg: float = 14.0
    tilt_jitter_deg: float = 3.0
    yaw_deg: tuple[float, float] = (-180.0, 180.0)


def rest_height(phantom: Phantom, rot: Rotation, xy: np.ndarray | None = None,
                tabletop: list[TabletopItem] | None = None) -> float:
    """Height of the phantom origin when its skin rests on the table, mat or paper."""
    return TABLE_Z + support_height(rot.apply(phantom.vertices), phantom.faces,
                                   np.zeros(2) if xy is None else xy, tabletop or [])


def sample_rest_pose(rng: np.random.Generator, phantom: Phantom, cfg: PlacementConfig,
                     base_clearance: float = 0.12, tries: int = 50,
                     *, tabletop: list[TabletopItem] | None = None) -> Pose:
    """Flat on the table, any heading, forearm centre within reach, clear of the arm's base."""
    for _ in range(tries):
        pose = _sample_rest_pose(rng, phantom, cfg, tabletop)
        xy = pose.apply(phantom.vertices)[:, :2]
        if np.min(np.linalg.norm(xy, axis=1)) >= base_clearance:
            return pose
    raise ValueError(f"no forearm placement satisfies the {base_clearance:g} m base clearance in {tries} attempts")


def _sample_rest_pose(rng: np.random.Generator, phantom: Phantom, cfg: PlacementConfig,
                      tabletop: list[TabletopItem] | None = None) -> Pose:
    u = rng.random()
    if u < cfg.back_up_prob:
        roll = 0.0
    elif u < cfg.back_up_prob + cfg.palm_up_prob:
        roll = np.pi
    else:
        roll = rng.choice([-1.0, 1.0]) * np.pi / 2
    roll += np.radians(rng.normal(0.0, cfg.roll_jitter_deg))
    tilt = np.radians(rng.normal(0.0, cfg.tilt_jitter_deg))
    yaw = np.radians(rng.uniform(*cfg.yaw_deg))
    rot = Rotation.from_euler("z", yaw) * Rotation.from_euler("y", tilt) * Rotation.from_euler("x", roll)
    r = rng.uniform(*cfg.radius)
    az = np.radians(rng.uniform(*cfg.azimuth_deg))
    centre = np.array([r * np.cos(az), r * np.sin(az)])
    # The forearm centre is the phantom origin, so place the origin there.
    return Pose(pos=np.array([centre[0], centre[1], rest_height(phantom, rot, centre, tabletop)]), rot=rot)


def _min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return min(max(s * s * s * (10.0 - 15.0 * s + 6.0 * s * s), 0.0), 1.0)


@dataclass
class MotionScript:
    keyframes: list[Keyframe] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.keyframes[-1].t

    def _segment(self, t: float) -> tuple[Keyframe, Keyframe, float]:
        frames = self.keyframes
        if t <= frames[0].t:
            return frames[0], frames[0], 0.0
        for a, b in zip(frames[:-1], frames[1:], strict=True):
            if t <= b.t:
                return a, b, (t - a.t) / max(b.t - a.t, 1e-9)
        return frames[-1], frames[-1], 0.0

    def pose(self, t: float) -> Pose:
        a, b, s = self._segment(t)
        w = _min_jerk(s)
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([a.pose.rot, b.pose.rot]))
        return Pose(pos=(1 - w) * a.pose.pos + w * b.pose.pos, rot=slerp([w])[0])

    def velocity(self, t: float, dt: float = 1.0 / 60.0) -> tuple[float, float]:
        """(linear m/s, angular rad/s) of the phantom origin at ``t``."""
        p0, p1 = self.pose(t - dt), self.pose(t + dt)
        linear = float(np.linalg.norm(p1.pos - p0.pos) / (2 * dt))
        angular = float((p1.rot * p0.rot.inv()).magnitude() / (2 * dt))
        return linear, angular

    def held(self, t: float, margin: float = 0.6) -> bool:
        """Hands are on the phantom while it moves, and ``margin`` s either side."""
        for a, b in zip(self.keyframes[:-1], self.keyframes[1:], strict=True):
            if b.held and a.t - margin <= t <= b.t + margin:
                return True
        return False

    def away(self, t: float) -> bool:
        a, b, _ = self._segment(t)
        return a.away and b.away


@dataclass(frozen=True)
class MotionConfig:
    hold_s: tuple[float, float] = (4.0, 14.0)
    # Mostly lying still, as on the rig; handling keeps the backing off and coming back in the data.
    weights: dict = field(default_factory=lambda: {
        "hold": 0.5, "nudge": 0.2, "relocate": 0.12, "carry": 0.12, "remove": 0.06})
    nudge_m: float = 0.06
    nudge_yaw_deg: float = 30.0
    nudge_s: tuple[float, float] = (0.8, 2.0)
    lift_m: tuple[float, float] = (0.04, 0.15)
    lift_away_m: tuple[float, float] = (0.02, 0.08)  # lifts also draw the phantom away from the arm
    lift_s: tuple[float, float] = (0.8, 1.6)
    travel_s: tuple[float, float] = (1.0, 3.0)
    carry_speed: tuple[float, float] = (0.01, 0.04)  # m/s mean; min-jerk peaks at ~1.9x, still followable
    carry_s: tuple[float, float] = (4.0, 10.0)
    carry_tilt_deg: float = 10.0
    away_s: tuple[float, float] = (2.0, 8.0)
    placement: PlacementConfig = field(default_factory=PlacementConfig)


class _Builder:
    def __init__(self, rng: np.random.Generator, phantom: Phantom, cfg: MotionConfig, start: Pose,
                 tabletop: list[TabletopItem] | None = None):
        self.rng, self.phantom, self.cfg = rng, phantom, cfg
        self.tabletop = tabletop
        self.frames = [Keyframe(0.0, start, held=False)]

    @property
    def now(self) -> Keyframe:
        return self.frames[-1]

    def add(self, dt: float, pose: Pose, held: bool, away: bool = False) -> None:
        self.frames.append(Keyframe(self.now.t + dt, pose, held, away))

    def lifted(self, pose: Pose, height: float, away: float = 0.0) -> Pose:
        """Raised by ``height`` and drawn ``away`` from the arm's base, as a person picks it up."""
        radial = np.array([pose.pos[0], pose.pos[1], 0.0])
        radial /= max(np.linalg.norm(radial), 1e-9)
        return Pose(pos=pose.pos + np.array([0.0, 0.0, height]) + radial * away, rot=pose.rot)

    def pick_up(self, height: float) -> Pose:
        return self.lifted(self.now.pose, height, self.rng.uniform(*self.cfg.lift_away_m))

    def hold(self) -> None:
        self.add(self.rng.uniform(*self.cfg.hold_s), self.now.pose, held=False)

    def nudge(self) -> None:
        cfg, rng, pose = self.cfg, self.rng, self.now.pose
        rot = Rotation.from_euler("z", np.radians(rng.uniform(-cfg.nudge_yaw_deg, cfg.nudge_yaw_deg))) * pose.rot
        shift = np.append(rng.uniform(-cfg.nudge_m, cfg.nudge_m, 2), 0.0)
        pos = pose.pos + shift
        pos[2] = rest_height(self.phantom, rot, pos[:2], self.tabletop)
        self.add(rng.uniform(*cfg.nudge_s), Pose(pos=pos, rot=rot), held=True)

    def relocate(self) -> None:
        cfg, rng = self.cfg, self.rng
        up = rng.uniform(*cfg.lift_m)
        target = sample_rest_pose(rng, self.phantom, cfg.placement, tabletop=self.tabletop)
        self.add(rng.uniform(*cfg.lift_s), self.pick_up(up), held=True)
        self.add(rng.uniform(*cfg.travel_s), self.lifted(target, up), held=True)
        self.add(rng.uniform(*cfg.lift_s), target, held=True)

    def carry(self) -> None:
        """Held up and moved slowly: the case the demo exists to show."""
        cfg, rng = self.cfg, self.rng
        up = rng.uniform(*cfg.lift_m)
        pose = self.pick_up(up)
        lying = pose.rot
        yaw = 0.0
        self.add(rng.uniform(*cfg.lift_s), pose, held=True)
        remaining = rng.uniform(*cfg.carry_s)
        while remaining > 0:
            step = min(remaining, rng.uniform(1.0, 3.0))
            speed = rng.uniform(*cfg.carry_speed)
            direction = rng.normal(size=3) * np.array([1.0, 1.0, 0.3])
            direction /= np.linalg.norm(direction)
            pos = pose.pos + direction * speed * step
            pos[2] = max(pos[2], rest_height(self.phantom, pose.rot, pos[:2], self.tabletop) + 0.02)
            # Held up, the phantom turns about the vertical and tips only a little.
            yaw += rng.normal(0.0, 0.2) * step
            tip = Rotation.from_rotvec(np.append(rng.normal(0.0, np.radians(cfg.carry_tilt_deg), 2), 0.0))
            pose = Pose(pos=pos, rot=Rotation.from_euler("z", yaw) * tip * lying)
            self.add(step, pose, held=True)
            remaining -= step
        target = Pose(pos=np.array([pose.pos[0], pose.pos[1], rest_height(
            self.phantom, pose.rot, pose.pos[:2], self.tabletop)]),
                      rot=pose.rot)
        self.add(rng.uniform(*cfg.lift_s), target, held=True)

    def remove(self) -> None:
        cfg, rng = self.cfg, self.rng
        self.add(rng.uniform(*cfg.lift_s), self.pick_up(0.12), held=True)
        away = Pose(pos=AWAY + rng.uniform(-0.2, 0.2, 3), rot=self.now.pose.rot)
        self.add(rng.uniform(*cfg.travel_s), away, held=True, away=True)
        self.add(rng.uniform(*cfg.away_s), away, held=False, away=True)
        target = sample_rest_pose(rng, self.phantom, cfg.placement, tabletop=self.tabletop)
        self.add(rng.uniform(*cfg.travel_s), self.lifted(target, 0.15), held=True)
        self.add(rng.uniform(*cfg.lift_s), target, held=True)


def sample_script(rng: np.random.Generator, phantom: Phantom, duration: float,
                  cfg: MotionConfig | None = None, *, tabletop: list[TabletopItem] | None = None) -> MotionScript:
    cfg = cfg or MotionConfig()
    start = sample_rest_pose(rng, phantom, cfg.placement, tabletop=tabletop)
    builder = _Builder(rng, phantom, cfg, start, tabletop)
    builder.hold()
    names = list(cfg.weights)
    weights = np.array([cfg.weights[n] for n in names], dtype=float)
    while builder.now.t < duration:
        getattr(builder, str(rng.choice(names, p=weights / weights.sum())))()
        if builder.now.pose.pos[2] > 0.3 and not builder.now.away:
            builder.hold()
    builder.add(1.0, builder.now.pose, held=False, away=builder.now.away)
    return MotionScript(builder.frames)
