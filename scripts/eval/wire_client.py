"""Shared no-arm client for Tatbot's deployed LeRobot policy wire.

This module intentionally lives in the LeRobot environment.  It constructs a
feature-only Tatbot follower (no cameras, driver, calibration gate, or arm
connection), maps that actual follower declaration to LeRobot features, and
speaks the same gRPC protocol as the rollout client.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import grpc
import numpy as np
import torch
from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedObservation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins
from wrist_cameras import describe, read_checkpoint_config, validate_checkpoint_views

DEFAULT_SERVER = os.environ.get("TATBOT_POLICY_SERVER", "")
DEFAULT_TASK = "draw a continuous squiggle using pen tip on the grid lines of the paper pad."


@dataclass(frozen=True)
class Scenario:
    policy_type: str
    use_depth: bool
    external_effort: bool
    depth_encoding: str
    actions_per_chunk: int
    mask_external_effort: bool = False


SCENARIOS = {
    "act_rgb": Scenario("act", False, False, "", 16),
    "act_rgbd14_masked": Scenario("act", True, True, "", 100, True),
}


def build_feature_robot(scenario: Scenario, *, sensor_profile: str = "deployment"):
    """Build the real follower's feature declaration without hardware objects.

    TatbotFollower's feature properties are explicitly callable while
    disconnected, but its third-party parent still constructs camera driver
    objects in ``__init__``.  A declaration-only instance bypasses that parent
    constructor and supplies its camera *configs* as camera-shaped namespaces.
    Tool-registry validation is disabled because this process neither connects
    nor moves a tool; the selected sim tool remains recorded by the worker.
    """

    register_third_party_plugins()
    from lerobot_robot_tatbot import TatbotFollower, TatbotFollowerConfig

    cameras = {
        camera.role: SimpleNamespace(
            serial_number_or_name=camera.role,  # Feature label; no device is opened.
            width=camera.width,
            height=camera.height,
            use_rgb=True,
            use_depth=scenario.use_depth,
        )
        for camera in describe(Path(__file__).resolve().parents[2], profile=sensor_profile)
    }
    config = TatbotFollowerConfig(
        ip_address="",
        id="tatbot_follower_right",
        cameras={},
        ee_tool=None,
        include_external_effort=scenario.external_effort,
        mask_external_effort=scenario.mask_external_effort,
        depth_policy_encoding=scenario.depth_encoding,
        use_tool_registry=False,
    )
    robot = object.__new__(TatbotFollower)
    robot.config = config
    robot.cameras = cameras
    return robot


def observation_from_worker(robot, arrays: dict[str, np.ndarray], task: str) -> dict:
    """Map exact sim arrays through the follower's current feature contract."""

    from lerobot_robot_tatbot.depth_encoding import encode_depth_mm

    qpos = np.asarray(arrays.get("qpos"), dtype=np.float32)
    ext_eff = np.asarray(arrays.get("external_effort"), dtype=np.float32)
    count = len(robot.config.joint_names)
    if qpos.shape != (count,) or ext_eff.shape != (count,):
        raise ValueError(
            f"worker state must be two {count}-vectors, got {qpos.shape} and {ext_eff.shape}"
        )
    joint_index = {name: index for index, name in enumerate(robot.config.joint_names)}
    observation: dict[str, object] = {}
    for key, feature in robot.observation_features.items():
        if not isinstance(feature, tuple):
            joint, suffix = key.rsplit(".", 1)
            index = joint_index.get(joint)
            if index is None:
                raise ValueError(f"unknown follower scalar feature {key!r}")
            if suffix == "pos":
                observation[key] = float(qpos[index])
            elif suffix == "ext_eff":
                observation[key] = 0.0 if robot.config.mask_external_effort else float(ext_eff[index])
            elif suffix in {"vel", "eff"}:
                observation[key] = 0.0
            else:
                raise ValueError(f"unsupported follower scalar feature {key!r}")
            continue
        source_key = key
        if key.endswith("_depth"):
            source_key = key
        elif key in robot.cameras:
            source_key = f"{key}_rgb"
        source = np.asarray(arrays.get(source_key))
        expected = tuple(feature)
        if key.endswith("_depth") and expected[-1] == 3:
            if source.ndim == 2:
                source = source[..., None]
            source = encode_depth_mm(source)
        elif key.endswith("_depth") and source.ndim == 2:
            source = source[..., None]
        if source.shape != expected:
            raise ValueError(
                f"worker feature {source_key!r} shape {source.shape} does not match {expected}"
            )
        observation[key] = source
    observation["task"] = task
    return observation


def wait_for_actions(stub, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(0.05, deadline - time.monotonic())
        try:
            response = stub.GetActions(services_pb2.Empty(), timeout=min(1.0, remaining))
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                continue
            raise
        if response.data:
            return pickle.loads(response.data)  # nosec B301 - trusted deployed LeRobot wire
        time.sleep(0.02)
    raise TimeoutError(f"no action chunk within {timeout_s:.1f}s")


class WirePolicyClient:
    """Synchronous adapter around the deployed async policy service."""

    def __init__(
        self,
        *,
        server: str,
        policy: str,
        scenario: Scenario,
        policy_type: str | None = None,
        actions_per_chunk: int | None = None,
        device: str = "cuda",
        timeout: float = 90.0,
        sensor_profile: str = "deployment",
        checkpoint_config: Path | None = None,
    ):
        if not server:
            raise ValueError("policy server must be stated as host:port")
        self.server = server
        self.policy = policy
        self.scenario = scenario
        self.policy_type = policy_type or scenario.policy_type
        self.actions_per_chunk = actions_per_chunk or scenario.actions_per_chunk
        self.device = device
        self.timeout = timeout
        self.robot = build_feature_robot(scenario, sensor_profile=sensor_profile)
        self.checkpoint_config_path, config = read_checkpoint_config(policy, checkpoint_config)
        image_shapes = {name: (shape[2], shape[0], shape[1])
                        for name, shape in self.robot.observation_features.items()
                        if isinstance(shape, tuple)}
        validate_checkpoint_views(config, tuple(self.robot.cameras), use_depth=scenario.use_depth,
                                  image_shapes=image_shapes)
        self.features = map_robot_keys_to_lerobot_features(self.robot)
        self.channel = grpc.insecure_channel(server, grpc_channel_options())
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.connected = False

    def connect(self) -> None:
        self.stub.Ready(services_pb2.Empty(), timeout=self.timeout)
        setup = RemotePolicyConfig(
            self.policy_type,
            self.policy,
            self.features,
            self.actions_per_chunk,
            self.device,
        )
        self.stub.SendPolicyInstructions(
            services_pb2.PolicySetup(
                data=pickle.dumps(setup, protocol=pickle.HIGHEST_PROTOCOL)
            ),
            timeout=self.timeout,
        )
        self.connected = True

    def predict(self, observation: dict, timestep: int, *, must_go: bool = True):
        if not self.connected:
            raise RuntimeError("policy client is not connected")
        timed = TimedObservation(
            timestamp=time.time(), timestep=timestep, observation=observation, must_go=must_go
        )
        self.stub.SendObservations(
            send_bytes_in_chunks(
                pickle.dumps(timed, protocol=pickle.HIGHEST_PROTOCOL),
                services_pb2.Observation,
                silent=True,
            ),
            timeout=self.timeout,
        )
        return wait_for_actions(self.stub, self.timeout)

    def close(self) -> None:
        self.channel.close()
        self.connected = False


class ActionQueue:
    """The rollout client's timestep filtering and weighted overlap semantics."""

    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim
        self.latest = -1
        self._actions: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._actions)

    def merge(self, actions) -> dict:
        accepted = skipped_old = overlapped = 0
        for timed in actions:
            timestep = int(timed.get_timestep())
            value = timed.get_action()
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (self.action_dim,) or not np.isfinite(array).all():
                raise ValueError(
                    f"policy action at timestep {timestep} must be a finite "
                    f"{self.action_dim}-vector, got {array.shape}"
                )
            if timestep <= self.latest:
                skipped_old += 1
                continue
            if timestep in self._actions:
                array = 0.3 * self._actions[timestep] + 0.7 * array
                overlapped += 1
            self._actions[timestep] = array
            accepted += 1
        return {"accepted": accepted, "skipped_old": skipped_old, "overlapped": overlapped}

    def pop(self) -> tuple[int, np.ndarray]:
        if not self._actions:
            raise IndexError("action queue is empty")
        timestep = min(self._actions)
        action = self._actions.pop(timestep)
        self.latest = timestep
        return timestep, action

    def should_replenish(self, actions_per_chunk: int, threshold: float) -> bool:
        if actions_per_chunk <= 0 or not 0 <= threshold <= 1:
            raise ValueError("invalid chunk replenishment configuration")
        return len(self) / actions_per_chunk <= threshold


class ExecutionFilter:
    """Replay TatbotFollower's 30 Hz arm EMA and target slew in simulation."""

    def __init__(
        self,
        initial_qpos: np.ndarray,
        *,
        fps: float = 30.0,
        target_filter_tau_s: float = 0.3,
        max_joint_velocity_rad_s: float = 0.25,
    ):
        initial = np.asarray(initial_qpos, dtype=np.float32)
        if initial.shape != (7,) or not np.isfinite(initial).all():
            raise ValueError("execution filter needs a finite initial 7-vector")
        if fps <= 0 or target_filter_tau_s < 0 or max_joint_velocity_rad_s <= 0:
            raise ValueError("invalid execution filter configuration")
        self.fps = float(fps)
        self.tau = float(target_filter_tau_s)
        self.velocity = float(max_joint_velocity_rad_s)
        self.filtered = initial[:6].copy()
        self.sent = initial[:6].copy()
        self.carriage = float(initial[6])
        self.saturated_steps = 0
        self.steps = 0

    def apply(self, requested: object) -> np.ndarray:
        request = np.asarray(requested, dtype=np.float32)
        if request.shape != (7,) or not np.isfinite(request).all():
            raise ValueError(f"execution request must be a finite 7-vector, got {request.shape}")
        dt = 1.0 / self.fps
        alpha = 1.0 if self.tau == 0 else dt / (self.tau + dt)
        self.filtered += alpha * (request[:6] - self.filtered)
        delta = self.filtered - self.sent
        budget = self.velocity * dt
        self.saturated_steps += int(np.any(np.abs(delta) > budget + 1e-12))
        self.sent += np.clip(delta, -budget, budget)
        self.steps += 1
        # The fitted carriage is position-held at its rest target; policy
        # output never drives it on the real follower.
        return np.concatenate([self.sent, np.asarray([self.carriage], dtype=np.float32)])

    def stats(self) -> dict[str, float | int]:
        return {
            "steps": self.steps,
            "saturated_steps": self.saturated_steps,
            "saturated_fraction": self.saturated_steps / max(self.steps, 1),
            "fps": self.fps,
            "target_filter_tau_s": self.tau,
            "max_joint_velocity_rad_s": self.velocity,
        }
