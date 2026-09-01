#!/usr/bin/env python3
"""No-arm trajectory-plausibility evaluation for chunked policies.

The client in this file cannot connect to a robot: it declares the policy wire
features directly and imports no Tatbot robot, camera, or arm driver.  A PASS is
only a plumbing/plausibility result.  It is never permission for powered use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

JOINT_NAMES = [
    "joint_0.pos",
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "left_carriage_joint.pos",
]
TASK = "remove the ink on the silicone skin with the removal laser"
# The carriage rests at one value for a whole recording (gripper era: the
# fingers stopped on the machine body; fixed mount: the closed hard stop).
# Encoder noise is well under a millimetre; a real trip lifts it 40 mm.
CARRIAGE_MAX_SPREAD_M = 0.002
SAFETY_WARNING = (
    "rejection-only no-arm diagnostic; PASS is not powered-use acceptance or a safety claim"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def postprocessor_artifact_hashes(postprocessor: Path) -> dict[str, str]:
    """Bind a processor definition and every external state file it names."""

    payload = json.loads(postprocessor.read_text())
    names = {postprocessor.name}
    for step in payload.get("steps", []):
        state_file = step.get("state_file")
        if not state_file:
            continue
        name = Path(state_file)
        if name.name != state_file or name.is_absolute():
            raise ValueError(f"postprocessor state file must be a local basename: {state_file!r}")
        names.add(state_file)
    result = {}
    for name in sorted(names):
        path = postprocessor.parent / name
        if not path.is_file():
            raise ValueError(f"postprocessor artifact is missing: {path}")
        result[name] = file_sha256(path)
    return result


def validate_postprocessor_binding(contract: dict[str, Any], postprocessor: Path) -> dict[str, str]:
    """Fail before scoring when a contract belongs to another processor bundle."""

    expected = contract.get("postprocessor_artifacts_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("plausibility contract lacks processor artifact hashes")
    actual = postprocessor_artifact_hashes(postprocessor)
    if actual != expected:
        names = sorted(set(actual) | set(expected))
        mismatched = [name for name in names if actual.get(name) != expected.get(name)]
        raise ValueError(
            "plausibility contract processor artifacts do not match checkpoint: "
            + ", ".join(mismatched)
        )
    if contract.get("postprocessor_sha256") != actual.get(postprocessor.name):
        raise ValueError("plausibility contract postprocessor hash is internally inconsistent")
    return actual


def _as_matrix(column: Any, width: int) -> np.ndarray:
    return np.asarray(column.to_pylist(), dtype=np.float64).reshape(-1, width)


def read_demonstrations(roots: list[Path], horizon: int) -> dict[str, np.ndarray]:
    """Read state/action rows directly, without decoding cameras or loading a policy."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    states: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    pads: list[np.ndarray] = []
    episode_ids: list[int] = []
    frame_ids: list[int] = []
    episode_offset = 0
    for root in roots:
        files = sorted((root / "data").glob("chunk-*/*.parquet"))
        if not files:
            raise ValueError(f"no demonstration parquet files under {root}")
        table = pa.concat_tables(
            [
                pq.read_table(
                    path,
                    columns=["action", "observation.state", "episode_index", "frame_index"],
                )
                for path in files
            ]
        ).combine_chunks()
        action = _as_matrix(table["action"], len(JOINT_NAMES))
        state = _as_matrix(table["observation.state"], len(JOINT_NAMES))
        episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        for row in range(len(action)):
            same_episode = np.flatnonzero(episodes[row:] == episodes[row])[:horizon] + row
            # Episodes are contiguous in LeRobot parquet. Stop at the first boundary.
            same_episode = same_episode[frames[same_episode] == frames[row] + np.arange(len(same_episode))]
            length = min(len(same_episode), horizon)
            chunk = np.repeat(action[row : row + 1], horizon, axis=0)
            chunk[:length] = action[same_episode[:length]]
            pad = np.ones(horizon, dtype=bool)
            pad[:length] = False
            states.append(state[row])
            chunks.append(chunk)
            pads.append(pad)
            episode_ids.append(int(episodes[row]) + episode_offset)
            frame_ids.append(int(frames[row]))
        episode_offset += int(episodes.max()) + 1
    stacked_states = np.stack(states)
    carriage_spread_m = float(np.ptp(stacked_states[:, JOINT_NAMES.index("left_carriage_joint.pos")]))
    if carriage_spread_m > CARRIAGE_MAX_SPREAD_M:
        raise ValueError(
            f"the carriage state moves {carriage_spread_m * 1000:.1f} mm across these "
            f"demonstrations (> {CARRIAGE_MAX_SPREAD_M * 1000:.0f} mm). Since 2026-08-30 the "
            "carriage is the pen's contact axis and rests at one value throughout a "
            "recording — only a safety trip lifts it, and a trip is not demonstration "
            "data. Something else recorded this, or the trips were not cut.")
    return {
        "state": stacked_states,
        "action": np.stack(chunks),
        "action_is_pad": np.stack(pads),
        "episode_index": np.asarray(episode_ids, dtype=np.int64),
        "frame_index": np.asarray(frame_ids, dtype=np.int64),
    }


def action_decode_contract(
    postprocessor: Path, horizon: int, joints: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalization bounds and the joints decoded relative to state.

    GR00T stores one statistics block per action modality.  A modality present
    in ``relative_action`` is decoded as a delta; excluded modalities remain
    in ``action`` and are decoded absolutely.  The corrected Tatbot contract
    deliberately uses that split: six relative arm joints plus one absolute
    carriage joint.  Treating every column as relative corrupts the normalized
    carriage evidence and can make a real checkpoint gate meaningless.
    """

    payload = json.loads(postprocessor.read_text())
    step = next(
        value
        for value in payload["steps"]
        if value.get("registry_name") == "groot_n1_7_action_decode_v1"
    )
    raw_stats = step["config"]["raw_stats"]
    action_stats = raw_stats.get("action", {})
    relative_stats = raw_stats.get("relative_action", {})
    modality_names = list(action_stats) or list(relative_stats)
    if not modality_names:
        raise ValueError("postprocessor has no action normalization statistics")

    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    relative: list[bool] = []
    modalities: list[tuple[str, int, bool]] = []
    for name in modality_names:
        is_relative = name in relative_stats
        stats = relative_stats.get(name, action_stats.get(name))
        if stats is None:
            raise ValueError(f"action modality {name!r} has no normalization statistics")
        low = np.asarray(stats["min"], dtype=np.float64)
        high = np.asarray(stats["max"], dtype=np.float64)
        if low.ndim == 1:
            low = np.repeat(low[None, :], horizon, axis=0)
            high = np.repeat(high[None, :], horizon, axis=0)
        elif low.ndim == 2:
            low = low[:horizon]
            high = high[:horizon]
        else:
            raise ValueError(f"action modality {name!r} has unsupported bounds shape {low.shape}")
        if low.shape != high.shape or low.shape[0] != horizon:
            raise ValueError(
                f"action modality {name!r} bounds have shapes {low.shape}/{high.shape}; "
                f"expected horizon {horizon}"
            )
        lows.append(low)
        highs.append(high)
        relative.extend([is_relative] * low.shape[1])
        modalities.append((name, int(low.shape[1]), is_relative))

    low = np.concatenate(lows, axis=1)
    high = np.concatenate(highs, axis=1)
    if low.shape != (horizon, joints) or high.shape != (horizon, joints):
        raise ValueError(
            f"action bounds have shapes {low.shape}/{high.shape}; expected "
            f"{(horizon, joints)} from modalities {modalities}"
        )
    if np.any(high <= low):
        raise ValueError("action normalization has a non-positive range")
    return low, high, np.asarray(relative, dtype=bool)


def postprocessor_action_mode(postprocessor: Path) -> str:
    """Identify the action normalization contract without loading policy code."""

    payload = json.loads(postprocessor.read_text())
    names = [step.get("registry_name") for step in payload.get("steps", [])]
    if "groot_n1_7_action_decode_v1" in names:
        return "groot_relative_minmax"
    if "unnormalizer_processor" in names:
        return "standard_absolute"
    raise ValueError("postprocessor has no supported action decode/unnormalize step")


def inverse_decode(
    actions: np.ndarray,
    states: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    relative_joints: np.ndarray,
) -> np.ndarray:
    native = np.array(actions, dtype=np.float64, copy=True)
    native[..., relative_joints] -= states[..., None, relative_joints]
    return 2.0 * (native - low) / (high - low) - 1.0


def distribution(values: np.ndarray) -> dict[str, list[float]]:
    flat = values.reshape(-1, values.shape[-1])
    return {
        "q50": np.quantile(flat, 0.50, axis=0).tolist(),
        "q95": np.quantile(flat, 0.95, axis=0).tolist(),
        "q99": np.quantile(flat, 0.99, axis=0).tolist(),
        "max": flat.max(axis=0).tolist(),
    }


def build_contract(
    demonstrations: dict[str, np.ndarray],
    postprocessor: Path,
    roots: list[Path],
) -> dict[str, Any]:
    states = demonstrations["state"]
    actions = demonstrations["action"]
    valid = ~demonstrations["action_is_pad"]
    if actions.ndim != 3 or states.shape != (actions.shape[0], actions.shape[2]):
        raise ValueError(f"invalid state/action shapes: {states.shape}/{actions.shape}")
    horizon, joints = actions.shape[1:]
    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    adjacent = np.abs(np.diff(actions, axis=1))[adjacent_valid]
    first_distance = np.abs(actions[:, 0] - states)
    adjacent_dist = distribution(adjacent)
    first_dist = distribution(first_distance)
    action_mode = postprocessor_action_mode(postprocessor)
    relative_joints = np.zeros(joints, dtype=bool)
    reference: dict[str, Any] = {
        "adjacent_step_abs_rad": adjacent_dist,
        "first_target_distance_abs_rad": first_dist,
    }
    thresholds: dict[str, Any] = {
        # Maxima make every genuine reference row pass by construction. This is an
        # initial rejection envelope, not a calibrated acceptance tolerance.
        "adjacent_step_abs_rad_per_joint": adjacent_dist["max"],
        "first_target_distance_abs_rad_per_joint": first_dist["max"],
        "l1_to_demo_rad_per_joint": first_dist["max"],
        "repeated_first_std_rad_per_joint": adjacent_dist["q99"],
    }
    if action_mode == "groot_relative_minmax":
        low, high, relative_joints = action_decode_contract(postprocessor, horizon, joints)
        normalized = inverse_decode(actions, states, low, high, relative_joints)
        normalized_adjacent = np.abs(np.diff(normalized, axis=1))[adjacent_valid]
        normalized_valid = normalized[valid]
        endpoint = np.abs(normalized_valid) >= 1.0 - 1e-6
        normalized_adjacent_dist = distribution(normalized_adjacent)
        reference.update(
            {
                "normalized_adjacent_step_abs": normalized_adjacent_dist,
                "normalized_endpoint_fraction_per_joint": endpoint.mean(axis=0).tolist(),
                "normalized_endpoint_fraction_overall": float(endpoint.mean()),
            }
        )
        thresholds.update(
            {
                "normalized_adjacent_step_abs_per_joint": normalized_adjacent_dist["max"],
                "normalized_endpoint_fraction_per_joint": endpoint.mean(axis=0).tolist(),
                "normalized_endpoint_fraction_overall": float(endpoint.mean()),
            }
        )
    return {
        "schema_version": 1,
        "kind": "demonstration-derived no-arm trajectory plausibility contract",
        "safety_warning": SAFETY_WARNING,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_roots": [str(path) for path in roots],
        "postprocessor": str(postprocessor),
        "postprocessor_sha256": file_sha256(postprocessor),
        "postprocessor_artifacts_sha256": postprocessor_artifact_hashes(postprocessor),
        "frames": int(len(states)),
        "episodes": int(len(np.unique(demonstrations["episode_index"]))),
        "horizon": horizon,
        "joints": joints,
        "action_semantics": {
            "normalization": action_mode,
            "relative_joint_indices": np.flatnonzero(relative_joints).tolist(),
            "absolute_joint_indices": np.flatnonzero(~relative_joints).tolist(),
        },
        "reference": reference,
        "rejection_thresholds": thresholds,
        "threshold_rationale": (
            "observed genuine-demo maxima bound decoded step, target-distance, and L1; demo "
            "q99 adjacent motion bounds repeated-input spread. GR00T relative min-max "
            "processors additionally bind normalized step and endpoint saturation. Standard "
            "absolute policies are rejected on decoded-action metrics only. Passing does not "
            "promote a checkpoint."
        ),
    }


def extract_fixture(args: argparse.Namespace) -> dict[str, Any]:
    import sys

    if args.depth_encoding:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
        from feature_views import install_depth_encoding

        install_depth_encoding(args.depth_encoding)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    indices = [int(value) for value in args.indices.split(",")]
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        delta_timestamps={"action": [step / args.fps for step in range(args.horizon)]},
        video_backend="pyav",
        return_uint8=True,
    )
    items = [dataset[index] for index in indices]
    payload: dict[str, np.ndarray] = {
        "state": np.stack([item["observation.state"].numpy() for item in items]),
        "action": np.stack([item["action"].numpy() for item in items]),
        "action_is_pad": np.stack([item["action_is_pad"].numpy() for item in items]),
        "dataset_index": np.asarray(indices, dtype=np.int64),
        "episode_index": np.asarray([int(item["episode_index"]) for item in items]),
        "frame_index": np.asarray([int(item["frame_index"]) for item in items]),
        "task": np.asarray([str(item["task"]) for item in items]),
    }
    for key in (
        "wrist_upper",
        "wrist_lower",
        "wrist_upper_depth",
        "wrist_lower_depth",
    ):
        source = f"observation.images.{key}"
        if source not in items[0]:
            continue
        images = np.stack([item[source].numpy().transpose(1, 2, 0) for item in items])
        if images.dtype != np.uint8:
            images = np.rint(np.clip(images, 0.0, 1.0) * 255.0).astype(np.uint8)
        payload[key] = images
    args.npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.npz_out, **payload)
    return {
        "kind": "genuine held-out no-arm observation fixture",
        "safety_warning": SAFETY_WARNING,
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "indices": indices,
        "depth_encoding": args.depth_encoding,
        "npz": str(args.npz_out),
        "npz_sha256": file_sha256(args.npz_out),
    }


def wire_features(scenario: str) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES},
    }
    for name in ("wrist_upper", "wrist_lower"):
        features[f"observation.images.{name}"] = {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        }
        if scenario == "groot_rgbd":
            features[f"observation.images.{name}_depth"] = {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": False},
            }
    return features


def _wait_for_actions(stub: Any, empty: Any, timeout_s: float) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = stub.GetActions(empty())
        if response.data:
            return pickle.loads(response.data)
        time.sleep(0.1)
    raise TimeoutError(f"no action chunk within {timeout_s:.1f}s")


def evaluate_predictions(
    chunks: np.ndarray,
    fixture: dict[str, np.ndarray],
    contract: dict[str, Any],
    postprocessor: Path,
    primary_joints: int = 6,
) -> tuple[dict[str, Any], list[str]]:
    validate_postprocessor_binding(contract, postprocessor)
    # chunks: fixture x repetition x horizon x joint
    states = fixture["state"]
    truth = fixture["action"]
    valid = ~fixture["action_is_pad"]
    horizon, joints = chunks.shape[2:]
    adjacent = np.abs(np.diff(chunks, axis=2))
    first_distance = np.abs(chunks[:, :, 0] - states[:, None, :])
    repeated_std = chunks[:, :, 0].std(axis=1).max(axis=0)
    error = np.abs(chunks - truth[:, None])
    keep = valid[:, None, :, None]
    denominator = keep.sum(axis=(0, 1, 2)) * chunks.shape[1]
    l1 = (error * keep).sum(axis=(0, 1, 2)) / np.maximum(denominator, 1)
    metrics = {
        "adjacent_step_abs_rad_per_joint": adjacent.max(axis=(0, 1, 2)).tolist(),
        "adjacent_step_median_rad_per_joint": np.median(adjacent, axis=(0, 1, 2)).tolist(),
        "first_target_distance_abs_rad_per_joint": first_distance.max(axis=(0, 1)).tolist(),
        "l1_to_demo_rad_per_joint": l1.tolist(),
        "repeated_first_std_rad_per_joint": repeated_std.tolist(),
    }
    thresholds = contract["rejection_thresholds"]
    if "normalized_adjacent_step_abs_per_joint" in thresholds:
        low, high, relative_joints = action_decode_contract(postprocessor, horizon, joints)
        normalized = inverse_decode(chunks, states[:, None], low, high, relative_joints)
        normalized_adjacent = np.abs(np.diff(normalized, axis=2))
        endpoint = np.abs(normalized) >= 1.0 - 1e-6
        metrics.update(
            {
                "normalized_adjacent_step_abs_per_joint": normalized_adjacent.max(
                    axis=(0, 1, 2)
                ).tolist(),
                "normalized_endpoint_fraction_per_joint": endpoint.mean(
                    axis=(0, 1, 2)
                ).tolist(),
                "normalized_endpoint_fraction_overall": float(endpoint.mean()),
            }
        )
    failures = []
    for metric, values in metrics.items():
        if metric == "adjacent_step_median_rad_per_joint":
            continue
        if metric not in thresholds:
            continue
        limit = thresholds[metric]
        if isinstance(values, list):
            for joint, (value, maximum) in enumerate(zip(values, limit, strict=True)):
                if joint < primary_joints and value > maximum + 1e-6:
                    failures.append(
                        f"{metric}[{joint}]={value:.8g} > demo_limit={maximum:.8g}"
                    )
        elif values > limit + 1e-9:
            failures.append(f"{metric}={values:.8g} > demo_limit={limit:.8g}")
    return metrics, failures


def probe(args: argparse.Namespace) -> dict[str, Any]:
    import grpc
    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

    loaded = np.load(args.fixture, allow_pickle=False)
    fixture = {key: loaded[key] for key in loaded.files}
    contract = json.loads(args.contract.read_text())
    postprocessor_artifacts = validate_postprocessor_binding(contract, args.postprocessor)
    required = ["state", "action", "action_is_pad", "wrist_upper", "wrist_lower"]
    if args.scenario == "groot_rgbd":
        required += ["wrist_upper_depth", "wrist_lower_depth"]
    missing = [key for key in required if key not in fixture]
    if missing:
        raise ValueError(f"fixture lacks scenario features: {missing}")
    if fixture["action"].shape[1:] != (contract["horizon"], contract["joints"]):
        raise ValueError("fixture action shape does not match contract")

    channel = grpc.insecure_channel(args.server, grpc_channel_options())
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    grpc.channel_ready_future(channel).result(timeout=args.timeout)
    stub.Ready(services_pb2.Empty(), timeout=args.timeout)
    policy_type = {"act_rgb": "act", "groot_rgb": "groot", "groot_rgbd": "groot"}[
        args.scenario
    ]
    setup = RemotePolicyConfig(
        policy_type, args.policy, wire_features(args.scenario), contract["horizon"], "cuda"
    )
    stub.SendPolicyInstructions(
        services_pb2.PolicySetup(data=pickle.dumps(setup, protocol=pickle.HIGHEST_PROTOCOL)),
        timeout=args.timeout,
    )
    chunks = []
    latencies = []
    timestep = 0
    try:
        for fixture_index in range(len(fixture["state"])):
            repeated = []
            payload = {
                name: float(value)
                for name, value in zip(JOINT_NAMES, fixture["state"][fixture_index], strict=True)
            }
            for key in required[3:]:
                payload[key] = fixture[key][fixture_index]
            payload["task"] = str(fixture["task"][fixture_index])
            for _ in range(args.repetitions):
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
                actions = _wait_for_actions(stub, services_pb2.Empty, args.timeout)
                array = np.stack([action.get_action().numpy() for action in actions])
                if array.shape != (contract["horizon"], contract["joints"]):
                    raise ValueError(f"server returned unexpected action shape {array.shape}")
                if not np.isfinite(array).all():
                    raise ValueError("server returned non-finite actions")
                repeated.append(array)
                latencies.append((time.perf_counter() - started) * 1000.0)
                timestep += 1
            chunks.append(np.stack(repeated))
    finally:
        channel.close()
    array = np.stack(chunks)
    metrics, failures = evaluate_predictions(array, fixture, contract, args.postprocessor)
    warm = np.asarray(latencies[1:] or latencies)
    return {
        "schema_version": 1,
        "kind": "no-arm trajectory plausibility evaluation",
        "safety_warning": SAFETY_WARNING,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "reject" if failures else "pass_no_arm_only",
        "scenario": args.scenario,
        "server": args.server,
        "policy": args.policy,
        "fixture": str(args.fixture),
        "fixture_sha256": file_sha256(args.fixture),
        "contract": str(args.contract),
        "contract_sha256": file_sha256(args.contract),
        "postprocessor_sha256": file_sha256(args.postprocessor),
        "postprocessor_artifacts_sha256": postprocessor_artifacts,
        "fixtures": int(array.shape[0]),
        "repetitions": int(array.shape[1]),
        "shape": list(array.shape),
        "cold_latency_ms": float(latencies[0]),
        "warm_p50_ms": float(np.percentile(warm, 50)),
        "warm_p95_ms": float(np.percentile(warm, 95)),
        "metrics": metrics,
        "rejection_failures": failures,
        "chunks": array.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract_parser = subparsers.add_parser("contract", help="derive a rejection envelope")
    contract_parser.add_argument("--dataset-root", action="append", type=Path, required=True)
    contract_parser.add_argument("--postprocessor", type=Path, required=True)
    contract_parser.add_argument("--horizon", type=int, default=16)
    contract_parser.add_argument("--json-out", type=Path, required=True)

    fixture_parser = subparsers.add_parser("fixture", help="freeze genuine held-out inputs")
    fixture_parser.add_argument("--dataset-root", type=Path, required=True)
    fixture_parser.add_argument("--repo-id", required=True)
    fixture_parser.add_argument("--indices", required=True, help="comma-separated dataset indices")
    fixture_parser.add_argument("--depth-encoding")
    fixture_parser.add_argument("--fps", type=int, default=30)
    fixture_parser.add_argument("--horizon", type=int, default=16)
    fixture_parser.add_argument("--npz-out", type=Path, required=True)

    probe_parser = subparsers.add_parser("probe", help="query a policy server without a robot")
    probe_parser.add_argument("scenario", choices=("act_rgb", "groot_rgb", "groot_rgbd"))
    probe_parser.add_argument("--server", default=os.environ.get("TATBOT_POLICY_SERVER", ""),
                              help="host:port of the policy server (or TATBOT_POLICY_SERVER)")
    probe_parser.add_argument("--policy", required=True)
    probe_parser.add_argument("--fixture", type=Path, required=True)
    probe_parser.add_argument("--contract", type=Path, required=True)
    probe_parser.add_argument("--postprocessor", type=Path, required=True)
    probe_parser.add_argument("--repetitions", type=int, default=8)
    probe_parser.add_argument("--timeout", type=float, default=90.0)
    probe_parser.add_argument("--json-out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "contract":
        demos = read_demonstrations(args.dataset_root, args.horizon)
        result = build_contract(demos, args.postprocessor, args.dataset_root)
    elif args.command == "fixture":
        result = extract_fixture(args)
    else:
        result = probe(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True) if hasattr(args, "json_out") else None
    output = args.json_out if hasattr(args, "json_out") else None
    if output is not None:
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.command == "probe" and result["verdict"] == "reject":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
