#!/usr/bin/env python3
"""Exercise Tatbot's real async policy wire with synthetic observations.

No robot is connected and no action is sent to hardware. The bench uses the
actual Tatbot follower feature declaration, RemotePolicyConfig, gRPC payload,
server preprocessing, chunk prediction, and postprocessing. It is the gate for
every new policy family or feature contract before an operator is asked to put
an arm under policy control.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import grpc
import numpy as np
from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedObservation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins


def _wrist_serial(role: str) -> str:
    """Serial for a wrist camera ROLE, from the visiond sensor registry."""
    import tomllib
    reg = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "rust/visiond/config/vision.toml").read_text())
    for cam in reg.get("cameras", {}).get("realsense", []):
        if cam.get("role") == role:
            return str(cam["serial"])
    raise SystemExit(f"sensor registry has no role={role}")


DEFAULT_SERVER = os.environ.get("TATBOT_POLICY_SERVER", "")  # host:port; state it or set the env
DEFAULT_TASK = "draw a continuous squiggle using pen tip on the grid lines of the paper pad."
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "lib"))
import tool_spec  # noqa: E402

STAGED = tool_spec.staged_positions(_REPO)  # the golden's staged pose, not a copy


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
    "groot_rgb": Scenario("groot", False, False, "", 16),
    "groot_rgbd": Scenario("groot", True, False, "depth-v1", 16),
    "legacy_rgb7": Scenario("groot", False, False, "", 48),
    "legacy_rgbd7": Scenario("groot", True, False, "", 48),
    "legacy_rgbd14": Scenario("groot", True, True, "", 48),
}


def build_robot(scenario: Scenario, ee_tool: str):
    register_third_party_plugins()
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot_robot_tatbot import TatbotFollower, TatbotFollowerConfig

    cameras = {
        "wrist_upper": RealSenseCameraConfig(
            serial_number_or_name=_wrist_serial("wrist_upper"), width=640, height=480, fps=30,
            use_depth=scenario.use_depth,
        ),
        "wrist_lower": RealSenseCameraConfig(
            serial_number_or_name=_wrist_serial("wrist_lower"), width=640, height=480, fps=30,
            use_depth=scenario.use_depth,
        ),
    }
    config = TatbotFollowerConfig(
        ip_address=os.environ.get("TATBOT_FOLLOWER_IP", ""),
        id="tatbot_follower_right",
        cameras=cameras,
        ee_tool=ee_tool,
        include_external_effort=scenario.external_effort,
        mask_external_effort=scenario.mask_external_effort,
        depth_policy_encoding=scenario.depth_encoding,
    )
    return TatbotFollower(config)


def fake_observation(robot, task: str, seed: int) -> dict:
    from lerobot_robot_tatbot.depth_encoding import encode_depth_mm

    rng = np.random.default_rng(seed)
    observation = {}
    for key, feature in robot.observation_features.items():
        if isinstance(feature, tuple):
            height, width, channels = feature
            if key.endswith("_depth"):
                depth = rng.normal(165.0, 4.0, size=(height, width, 1))
                depth[rng.random((height, width, 1)) < 0.22] = 10.0
                if channels == 3:
                    observation[key] = encode_depth_mm(depth)
                else:
                    observation[key] = np.rint(depth).astype(np.uint16)
            else:
                observation[key] = rng.integers(
                    0, 256, size=(height, width, channels), dtype=np.uint8
                )
        else:
            positions = dict(zip(
                [f"{joint}.pos" for joint in robot.config.joint_names], STAGED, strict=True
            ))
            if key.endswith(".ext_eff"):
                observation[key] = 0.0 if robot.config.mask_external_effort else 1.0
            else:
                observation[key] = positions.get(key, 0.0)
    observation["task"] = task
    return observation


def wait_for_actions(stub, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = stub.GetActions(services_pb2.Empty())
        if response.data:
            return pickle.loads(response.data)
        time.sleep(0.25)
    raise TimeoutError(f"no action chunk within {timeout_s:.1f}s")


def run(args: argparse.Namespace) -> dict:
    scenario = SCENARIOS[args.scenario]
    policy_type = args.policy_type or scenario.policy_type
    actions_per_chunk = args.actions_per_chunk or scenario.actions_per_chunk
    robot = build_robot(scenario, args.ee_tool)
    features = map_robot_keys_to_lerobot_features(robot)
    feature_shapes = {
        key: value.get("shape") if isinstance(value, dict) else value
        for key, value in features.items()
    }
    print(f"scenario={args.scenario} server={args.server} policy={args.policy}")
    print(f"policy_type={policy_type} actions_per_chunk={actions_per_chunk}")
    print(f"wire_features={feature_shapes}")

    channel = grpc.insecure_channel(args.server, grpc_channel_options())
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    stub.Ready(services_pb2.Empty(), timeout=args.timeout)
    setup = RemotePolicyConfig(
        policy_type, args.policy, features, actions_per_chunk, args.device
    )
    stub.SendPolicyInstructions(
        services_pb2.PolicySetup(data=pickle.dumps(setup, protocol=pickle.HIGHEST_PROTOCOL)),
        timeout=args.timeout,
    )

    latencies_ms = []
    first_action = None
    action_min = float("inf")
    action_max = float("-inf")
    for timestep in range(args.repetitions):
        payload = fake_observation(robot, args.task, args.seed + timestep)
        observation = TimedObservation(
            timestamp=time.time(), timestep=timestep, observation=payload, must_go=True
        )
        started = time.perf_counter()
        stub.SendObservations(
            send_bytes_in_chunks(
                pickle.dumps(observation, protocol=pickle.HIGHEST_PROTOCOL),
                services_pb2.Observation,
                silent=True,
            ),
            timeout=args.timeout,
        )
        actions = wait_for_actions(stub, args.timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0
        array = np.stack([action.get_action().numpy() for action in actions])
        if array.shape != (actions_per_chunk, args.expect_action_dim):
            raise RuntimeError(
                f"unexpected action shape {array.shape}; expected "
                f"({actions_per_chunk}, {args.expect_action_dim})"
            )
        if not np.isfinite(array).all():
            raise RuntimeError("server returned non-finite actions")
        if first_action is None:
            first_action = array[0].tolist()
        action_min = min(action_min, float(array.min()))
        action_max = max(action_max, float(array.max()))
        latencies_ms.append(latency_ms)
        print(f"chunk={timestep} latency_ms={latency_ms:.1f} shape={array.shape}")
    channel.close()

    warm = np.asarray(latencies_ms[1:] or latencies_ms, dtype=np.float64)
    result = {
        "status": "ok",
        "scenario": args.scenario,
        "server": args.server,
        "policy": args.policy,
        "policy_type": policy_type,
        "mask_external_effort": scenario.mask_external_effort,
        "actions_per_chunk": actions_per_chunk,
        "action_dim": args.expect_action_dim,
        "feature_shapes": feature_shapes,
        "cold_latency_ms": latencies_ms[0],
        "warm_p50_ms": float(np.percentile(warm, 50)),
        "warm_p95_ms": float(np.percentile(warm, 95)),
        "action_min": action_min,
        "action_max": action_max,
        "first_action": first_action,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--policy", required=True, help="checkpoint path as seen by the server")
    parser.add_argument("--policy-type", help="override the scenario's policy family")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ee-tool",
        required=True,
        help="explicit tool id from config/tools; checked against the workspace contract",
    )
    parser.add_argument("--actions-per-chunk", type=int)
    parser.add_argument("--expect-action-dim", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = run(args)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
