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
import sys
import time
from pathlib import Path

import numpy as np
from wire_client import (
    DEFAULT_SERVER,
    DEFAULT_TASK,
    SCENARIOS,
    WirePolicyClient,
)

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
import tool_spec  # noqa: E402

STAGED = tool_spec.staged_positions(_REPO)


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


def run(args: argparse.Namespace) -> dict:
    scenario = SCENARIOS[args.scenario]
    policy_type = args.policy_type or scenario.policy_type
    actions_per_chunk = args.actions_per_chunk or scenario.actions_per_chunk
    client = WirePolicyClient(
        server=args.server,
        policy=args.policy,
        scenario=scenario,
        policy_type=policy_type,
        actions_per_chunk=actions_per_chunk,
        device=args.device,
        timeout=args.timeout,
        sensor_profile=args.sensor_profile,
        checkpoint_config=args.checkpoint_config,
    )
    robot = client.robot
    features = client.features
    feature_shapes = {
        key: value.get("shape") if isinstance(value, dict) else value
        for key, value in features.items()
    }
    print(f"scenario={args.scenario} server={args.server} policy={args.policy}")
    print(f"policy_type={policy_type} actions_per_chunk={actions_per_chunk}")
    print(f"wire_features={feature_shapes}")

    client.connect()

    latencies_ms = []
    first_action = None
    action_min = float("inf")
    action_max = float("-inf")
    for timestep in range(args.repetitions):
        payload = fake_observation(robot, args.task, args.seed + timestep)
        started = time.perf_counter()
        actions = client.predict(payload, timestep, must_go=True)
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
    client.close()

    warm = np.asarray(latencies_ms[1:] or latencies_ms, dtype=np.float64)
    result = {
        "status": "ok",
        "scenario": args.scenario,
        "sensor_profile": args.sensor_profile,
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
    parser.add_argument("--checkpoint-config", type=Path, help="local config.json for a server-side checkpoint")
    parser.add_argument("--sensor-profile", choices=("deployment", "legacy-two-view"), default="deployment")
    parser.add_argument("--policy-type", help="override the scenario's policy family")
    parser.add_argument("--device", default="cuda")
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
