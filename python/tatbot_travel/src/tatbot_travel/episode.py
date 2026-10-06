"""One episode: the world, the handler's script, the expert, the servo and the camera, tick by tick.

Per 30 Hz tick: pose the phantom and hands, pose the arm where it was 1-2
ticks ago (the rig's camera frames arrive that late), render the wrist view,
let the expert pick the next command from what it may know, and step the
servo. The scene camera captures at its stream's ~16 fps with the arm where
it is, and its frames reach the policy about a second later, as the rig's
do. A frame is (image, scene image, measured state, command), the LeRobot
convention that the action follows the observation it was chosen from.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tatbot_travel.camera import to_real_grid
from tatbot_travel.clearance import SkinDistance
from tatbot_travel.expert import Expert, ExpertConfig, Mode, Observation
from tatbot_travel.kinematics import LIMIT_MARGIN, ArmKinematics
from tatbot_travel.motion import MotionConfig, MotionScript, Pose, sample_script
from tatbot_travel.postfx import CameraLook, SensorState, StreamLook, StreamSensor
from tatbot_travel.scene import CAMERA, LENS_SITE, SCENE_CAMERA, SCENE_ONLY_GROUP
from tatbot_travel.selfview import Compositor, load_layers
from tatbot_travel.servo import Servo, ServoConfig
from tatbot_travel.world import HIDDEN, World, WorldConfig, build_world

TASK = "trace the ink on the arm with the laser"
Q_REST = np.array([0.0, 1.0, 1.0, -0.8, 0.0, np.pi / 2])  # tracing posture: elbow up, the tool rolled 90 deg
# Parked just off the arm's rest pose (all zero, the tool rolled 90 deg), as the rig parks it: the pen nose
# about 0.33 m out and 0.2 m up, the camera looking out over the table where the phantom lies. Joint 1 sits
# clear of the rig's limit (0 plus the runner's 0.05 margin). Tracing, the wrist roll varies about 90 deg
# as the pen follows the skin.
Q_PARK = np.array([0.0, 0.10, 0.10, 0.05, 0.0, np.pi / 2])


@dataclass(frozen=True)
class EpisodeConfig:
    fps: int = 30
    duration_s: float = 60.0
    # How an episode starts: on the ink, partway into an approach to it, or parked (the rest). Approach
    # starts add the approach's states that parked starts rarely reach (v6: 9 % of frames approached).
    start_trace_prob: float = 0.25
    start_approach_prob: float = 0.3
    park_noise_rad: float = 0.08
    scene_fps: tuple[float, float] = (14.0, 17.0)  # the scene camera's sub stream (the rig: 15-16 fps)
    scene_latency_s: tuple[float, float] = (0.7, 1.1)  # its frames' delay behind the arm (the rig: 0.9 s)
    stale_ticks: tuple[int, int] = (1, 2)  # the image shows the arm this far behind its reported state (the rig: 1-2)
    repeat_frame_prob: float = 0.02  # the camera had no new frame: the last one again (the rig: 0.7-2.8 %)
    keep_untraceable_prob: float = 0.3  # starts where no ink can be traced: keep this many, redraw the rest
    start_clearance_m: float = 0.04  # the parked pen at least this far from the skin when the episode starts
    max_draws: int = 6
    start_pose_rad: tuple[float, ...] | None = None  # measured starts may be at the physical stops
    camera_look: CameraLook | None = None
    profile_id: str | None = None
    world: WorldConfig = field(default_factory=WorldConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)


@dataclass
class Frame:
    image: np.ndarray
    state: np.ndarray
    action: np.ndarray
    labels: dict
    scene: np.ndarray | None = None  # the scene camera's frame, when the episode has one


class DelayedFrames:
    """Frames that reach the policy ``latency`` after their capture, the newest arrival held until the next."""

    def __init__(self, latency: float):
        self.latency = latency
        self.pending: deque[tuple[float, np.ndarray]] = deque()
        self.shown: np.ndarray | None = None

    def push(self, t: float, frame: np.ndarray) -> None:
        self.pending.append((t, frame))

    def view(self, t: float) -> np.ndarray:
        """The newest frame that has arrived by ``t`` (before the first arrives: the first capture)."""
        while self.pending and self.pending[0][0] <= t - self.latency:
            self.shown = self.pending.popleft()[1]
        return self.shown if self.shown is not None else self.pending[0][1]


class SceneStream:
    """The scene camera as the runner receives it: frames captured at the stream's rate, arriving late."""

    def __init__(self, model: mujoco.MjModel, plan, rng: np.random.Generator, cfg: EpisodeConfig):
        self.plan, self.rng = plan, rng
        self.renderer = mujoco.Renderer(model, plan.height, plan.width)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[SCENE_ONLY_GROUP] = 1  # the pen cradle and its tag cube, drawn for this camera
        self.look = StreamLook.sample(rng)
        self.sensor = StreamSensor(self.look, rng, plan.map_x.shape)
        self.period = 1.0 / float(rng.uniform(*cfg.scene_fps))
        self.frames = DelayedFrames(float(rng.uniform(*cfg.scene_latency_s)))
        self.next_capture = 0.0

    @property
    def latency(self) -> float:
        return self.frames.latency

    def due(self, t: float) -> bool:
        return t + 1e-9 >= self.next_capture

    def capture(self, data: mujoco.MjData, t: float) -> None:
        self.renderer.update_scene(data, camera=SCENE_CAMERA, scene_option=self.option)
        self.frames.push(t, self.sensor(to_real_grid(self.plan, self.renderer.render())))
        self.next_capture += self.period * float(self.rng.uniform(0.85, 1.15))

    def view(self, t: float) -> np.ndarray:
        return self.frames.view(t)

    def close(self) -> None:
        self.renderer.close()


def park_pose(kin: ArmKinematics, base_z: float = 0.0) -> np.ndarray:
    """The park pose: joint angles, so it rides with the base whatever its height over the table."""
    del kin, base_z
    return Q_PARK.copy()


def _pose_hands(data: mujoco.MjData, world: World, pose: Pose, held: bool) -> list[np.ndarray]:
    centres = []
    for i, offset in enumerate(world.hand_offsets):
        mid = world.mocap[f"hand{i}"]
        if not held:
            data.mocap_pos[mid] = HIDDEN
            continue
        rot = pose.rot * Rotation.from_matrix(offset[:3, :3])
        pos = pose.apply(offset[:3, 3][None])[0]
        data.mocap_pos[mid] = pos
        x, y, z, w = rot.as_quat()
        data.mocap_quat[mid] = [w, x, y, z]
        centres.append(pos)
    return centres


class EpisodeRunner:
    """Owns one world's model, renderer and state for the length of an episode."""

    def __init__(self, seed: int, cfg: EpisodeConfig | None = None):
        self.cfg = cfg or EpisodeConfig()
        for draw in range(self.cfg.max_draws):
            self._build(seed, draw)
            clear = self.expert.clearance(self.q_park, self.script.pose(0.0)) >= self.cfg.start_clearance_m
            keep = clear and (self.traceable or self.rng.random() < self.cfg.keep_untraceable_prob)
            if keep or draw == self.cfg.max_draws - 1:
                break
            self.renderer.close()
        self._start()

    def _build(self, seed: int, draw: int) -> None:
        """Draw a world, a handling script and an expert; ``draw`` > 0 redraws the same seed."""
        self.seed, self.draw = seed, draw
        self.rng = np.random.default_rng([seed, draw])
        self.world = build_world(self.rng, self.cfg.world)
        self.model = self.world.model
        self.data = mujoco.MjData(self.model)
        self.kin = ArmKinematics(self.model)
        self.renderer = mujoco.Renderer(self.model, self.world.plan.height, self.world.plan.width)
        self.script: MotionScript = sample_script(self.rng, self.world.phantom, self.cfg.duration_s, self.cfg.motion,
                                                 tabletop=self.world.tabletop)
        occluded = load_layers(self.world.selfview.root)[self.world.selfview.layer].alpha[..., 0] > 0.5
        base_z = float(self.world.meta["base_height_m"])
        self.q_park = park_pose(self.kin, base_z)
        self.skin = SkinDistance(self.world.phantom, seed=seed)
        self.lens_site = self.model.site(LENS_SITE).id
        self.expert = Expert(self.cfg.expert, self.kin, self.world.ink, self.world.intrinsics, occluded,
                             self.rng, self.q_park, Q_REST, self.skin, self.lens_site, base_z=base_z)
        self.sensor = SensorState(self.cfg.camera_look or CameraLook.sample(self.rng), self.rng, (self.world.intrinsics.height,
                                                                         self.world.intrinsics.width))
        self.stale = int(self.rng.integers(self.cfg.stale_ticks[0], self.cfg.stale_ticks[1] + 1))
        self.cradle = Compositor(self.world.selfview)
        self.traceable = self.expert.traceable_length(self.script.pose(0.0)) >= self.cfg.expert.min_interval_m
        self.scene = (SceneStream(self.model, self.world.scene_plan, self.rng, self.cfg)
                      if self.world.scene_plan is not None else None)

    def _start(self) -> None:
        q0 = self._start_pose()
        if self.expert.clearance(q0, self.script.pose(0.0)) < self.cfg.start_clearance_m:
            q0 = self.q_park.copy()
        u = self.rng.random()
        if not self.script.away(0.0):
            if u < self.cfg.start_trace_prob and self._start_tracing():
                return
            if 0.0 <= u - self.cfg.start_trace_prob < self.cfg.start_approach_prob and self._start_approaching(q0):
                return
        self.expert.reset(q0, Mode.PARK)
        self._init_servo(q0)

    def _start_pose(self) -> np.ndarray:
        """Sample observations at rest without changing the IK or hardware command limits."""
        q0 = self.q_park if self.cfg.start_pose_rad is None else np.asarray(self.cfg.start_pose_rad, dtype=float)
        margin = 0.0 if self.cfg.start_pose_rad is None else LIMIT_MARGIN
        if q0.shape != (6,) or not np.isfinite(q0).all():
            raise ValueError("start_pose_rad must contain six finite joint angles")
        return np.clip(q0 + self.rng.normal(0.0, self.cfg.park_noise_rad, 6),
                       self.kin.lower - margin, self.kin.upper + margin)

    def _ink_target(self, standoff: float) -> tuple[int, float, np.ndarray | None]:
        """A random stroke and station on it, and the arm pose that holds the pen ``standoff`` over it."""
        pose = self.script.pose(0.0)
        edge = self.expert.pick_edge()
        s = float(self.rng.uniform(0.0, self.world.ink.edges[edge].length))
        target, axis = self.expert.line_target(pose, s, standoff, edge=edge)
        result = self.kin.solve(Q_REST, target, axis, Q_REST, iters=80)
        return edge, s, result.q if result.ok() else None

    def _start_tracing(self) -> bool:
        edge, s, q0 = self._ink_target(self.cfg.expert.gap_m)
        if q0 is None or self.expert.clearance(q0, self.script.pose(0.0)) < self.cfg.expert.trace_clearance_m:
            return False
        self.expert.reset(q0, Mode.TRACE, edge=edge, s=s)
        self._init_servo(q0)
        return True

    def _start_approaching(self, q_park: np.ndarray) -> bool:
        """Partway from park toward hovering over reachable ink, already approaching it."""
        edge, s, q_hover = self._ink_target(self.cfg.expert.approach_hover_m)
        if q_hover is None:
            return False
        q0 = q_park + float(self.rng.uniform(0.2, 0.9)) * (q_hover - q_park)
        pose = self.script.pose(0.0)
        if self.expert.clearance(q0, pose) < 0.025 or not self.expert.under_ceiling(q0):
            return False
        self.expert.reset(q0, Mode.APPROACH, edge=edge, s=s)
        if not self.expert.begin_approach(pose, s, self.cfg.expert.min_interval_m):
            return False
        self._init_servo(q0)
        return True

    def _init_servo(self, q0: np.ndarray) -> None:
        dt = 1.0 / self.cfg.fps
        self.servo = Servo(q0, self.rng, self.cfg.servo, dt)
        self.history: deque[np.ndarray] = deque([q0.copy(), q0.copy()], maxlen=2)
        self.last_clearance = float("inf")
        self.last_image: np.ndarray | None = None

    def _scene_frame(self, t: float) -> np.ndarray | None:
        """The scene camera's frame as it reaches the runner at ``t``, capturing one now if one is due
        (with the arm where it is: the stream's delay comes after the capture)."""
        if self.scene is None:
            return None
        if self.scene.due(t):
            self.data.qpos[self.kin.qadr] = self.servo.q
            mujoco.mj_forward(self.model, self.data)
            self.scene.capture(self.data, t)
        return self.scene.view(t)

    def _render(self, q_view: np.ndarray, t: float) -> np.ndarray:
        self.data.qpos[self.kin.qadr] = q_view
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=CAMERA)
        return to_real_grid(self.world.plan, self.renderer.render())

    def clearance(self, q: np.ndarray, pose: Pose) -> float:
        """The pen nose's clearance to the skin with the arm at ``q``."""
        return self.expert.clearance(q, pose)

    def _image_motion(self, q_now: np.ndarray, q_before: np.ndarray) -> np.ndarray:
        """Pixel shift of the view centre (at the pen's working distance) over one exposure."""
        cam_p, cam_r = self.kin.camera_pose(q_now)
        cam_p0, cam_r0 = self.kin.camera_pose(q_before)
        point = cam_p + cam_r @ np.array([0.0, 0.0, 0.2])
        uv_now = self.world.intrinsics.project(((point - cam_p) @ cam_r)[None])[0]
        uv_before = self.world.intrinsics.project(((point - cam_p0) @ cam_r0)[None])[0]
        return (uv_now - uv_before) * (self.sensor.look.exposure_s * self.cfg.fps)

    def step(self, tick: int, execute: bool = True) -> Frame:
        """One tick. With ``execute=False`` the expert's command is only a label; call ``execute``."""
        t = tick / self.cfg.fps
        pose = self.script.pose(t)
        pid = self.world.mocap["phantom"]
        self.data.mocap_pos[pid] = pose.pos
        self.data.mocap_quat[pid] = pose.quat_wxyz()
        hands = _pose_hands(self.data, self.world, pose, self.script.held(t))
        state = self.servo.measured()
        scene = self._scene_frame(t)
        rgb = self._render(self.history[-self.stale] if self.stale else self.servo.q, t)
        image = self.sensor(rgb, self._image_motion(self.servo.q, self.history[-1]), overlay=self.cradle)
        if self.last_image is not None and self.rng.random() < self.cfg.repeat_frame_prob:
            image = self.last_image
        self.last_image = image
        linear, angular = self.script.velocity(t)
        held = self.script.held(t)
        clearance = self.clearance(self.servo.q, pose)
        # Closing on the pen: the phantom's own push, but only as far as the arm is not already giving way.
        pushed = (clearance - self.clearance(self.servo.q, self.script.pose(t + 1.0 / self.cfg.fps))) * self.cfg.fps
        closing = min(pushed, (self.last_clearance - clearance) * self.cfg.fps)
        self.last_clearance = clearance
        obs = Observation(t=t, q=state, phantom=pose, linear_speed=linear, angular_speed=angular,
                          away=self.script.away(t), hands=hands, held=held, clearance=clearance,
                          closing_speed=closing)
        action = self.expert.act(obs)
        self.history.append(self.servo.q.copy())
        if execute:
            self.servo.step(action)
        st = self.expert.state
        labels = {"mode": int(st.mode), "edge": st.edge, "s": st.s, "visible": st.visible, "ik_error": st.ik_error,
                  "clearance": clearance, "phantom_pos": pose.pos.copy(), "phantom_quat": pose.quat_wxyz(),
                  "held": held}
        return Frame(image=image, state=state.astype(np.float32), action=action.astype(np.float32), labels=labels,
                     scene=scene)

    def execute(self, command: np.ndarray) -> None:
        """Send an outside command (a policy's); the expert carries on from where it puts the arm."""
        self.servo.step(command)
        self.expert.follow(command)

    def run(self, on_frame: Callable[[Frame], None]) -> int:
        n = int(round(self.cfg.duration_s * self.cfg.fps))
        for tick in range(n):
            on_frame(self.step(tick))
        return n

    def close(self) -> None:
        self.renderer.close()
        if self.scene is not None:
            self.scene.close()
