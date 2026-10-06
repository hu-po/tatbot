#!/usr/bin/env python3
"""No-arm trajectory-plausibility evaluation for chunked policies.

The client in this file cannot connect to a robot: it declares the policy wire
features directly and imports no Tatbot robot, camera, or arm driver.  A PASS is
only a plumbing/plausibility result.  It is never permission for powered use.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from chunk_guard import (
    compare_metrics,
    execution_metrics,
    simulate_executed_actions,
    validate_execution_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/lib"))
from tatbot_digest import sha256_file  # noqa: E402
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()

JOINT_NAMES = [
    "joint_0.pos",
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "left_carriage_joint.pos",
]
EXTERNAL_EFFORT_NAMES = [name.removesuffix(".pos") + ".ext_eff" for name in JOINT_NAMES]
TASK = "remove the ink on the silicone skin with the removal laser"
# The carriage rests at one value for a whole recording (gripper era: the
# fingers stopped on the machine body; fixed mount: the closed hard stop).
# Encoder noise is well under a millimetre; a real trip lifts it 40 mm.
CARRIAGE_MAX_SPREAD_M = 0.002
GROSS_ENVELOPE_MULTIPLIER = 2.0
RAW_HARD_ABSOLUTE_MARGIN_RAD = 0.01
NORMALIZED_HARD_ABSOLUTE_MARGIN = 0.05
SUSTAINED_SLEW_FRACTION = 0.80
SAFETY_WARNING = (
    "rejection-only no-arm diagnostic; no verdict is powered-use acceptance or a safety claim"
)


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
        result[name] = sha256_file(path)
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


def _as_matrix(column: Any, width: int | None = None) -> np.ndarray:
    matrix = np.asarray(column.to_pylist(), dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix column, got shape {matrix.shape}")
    if width is not None and matrix.shape[1] != width:
        raise ValueError(f"expected matrix width {width}, got {matrix.shape[1]}")
    return matrix


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
        state = _as_matrix(table["observation.state"])
        if state.shape[1] < len(JOINT_NAMES):
            raise ValueError(
                f"observation.state width {state.shape[1]} is smaller than the "
                f"{len(JOINT_NAMES)} position channels"
            )
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


def postprocessor_action_mode(postprocessor: Path) -> str:
    """Identify the action normalization contract without loading policy code."""

    payload = json.loads(postprocessor.read_text())
    names = [step.get("registry_name") for step in payload.get("steps", [])]
    if "unnormalizer_processor" in names:
        return "standard_absolute"
    raise ValueError("postprocessor has no supported action decode/unnormalize step")


def distribution(values: np.ndarray) -> dict[str, list[float]]:
    flat = values.reshape(-1, values.shape[-1])
    return {
        "q50": np.quantile(flat, 0.50, axis=0).tolist(),
        "q95": np.quantile(flat, 0.95, axis=0).tolist(),
        "q99": np.quantile(flat, 0.99, axis=0).tolist(),
        "max": flat.max(axis=0).tolist(),
    }


def _gross_per_joint(
    values: list[float], factor: float, absolute_margin: float
) -> list[float]:
    return [
        max(float(value) * factor, float(value) + absolute_margin)
        for value in values
    ]


def execution_distributions(
    actions: np.ndarray,
    states: np.ndarray,
    execution_model: dict[str, Any],
) -> dict[str, dict[str, list[float]]]:
    """Describe requests after the rollout EMA and target slew model."""

    sent, saturated = simulate_executed_actions(actions, states, execution_model)
    previous = np.concatenate((states[..., None, :], sent[..., :-1, :]), axis=-2)
    return {
        "executed_step_abs_rad": distribution(np.abs(sent - previous)),
        "executed_displacement_abs_rad": distribution(
            np.abs(sent - states[..., None, :])
        ),
        "execution_slew_saturation_fraction": distribution(saturated.mean(axis=-2)),
    }


def build_contract(
    demonstrations: dict[str, np.ndarray],
    postprocessor: Path,
    roots: list[Path],
    *,
    fps: float = 30.0,
    target_filter_tau_s: float = 0.3,
    max_joint_velocity_rad_s: float = 0.25,
    controller_velocity_limit_rad_s: float = 0.75,
    actions_per_chunk: int | None = None,
) -> dict[str, Any]:
    states = demonstrations["state"]
    actions = demonstrations["action"]
    valid = ~demonstrations["action_is_pad"]
    if (
        actions.ndim != 3
        or states.ndim != 2
        or states.shape[0] != actions.shape[0]
        or states.shape[1] < actions.shape[2]
    ):
        raise ValueError(f"invalid state/action shapes: {states.shape}/{actions.shape}")
    position_states = states[:, : actions.shape[2]]
    horizon, joints = actions.shape[1:]
    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    adjacent = np.abs(np.diff(actions, axis=1))[adjacent_valid]
    first_distance = np.abs(actions[:, 0] - position_states)
    adjacent_dist = distribution(adjacent)
    first_dist = distribution(first_distance)
    action_mode = postprocessor_action_mode(postprocessor)
    relative_joints = np.zeros(joints, dtype=bool)
    execution_model: dict[str, Any] = {
        "input_stage": "post_client_aggregation",
        "offline_aggregation": "identity_single_chunk",
        "fps": float(fps),
        "target_filter_tau_s": float(target_filter_tau_s),
        "max_joint_velocity_rad_s": float(max_joint_velocity_rad_s),
        "controller_velocity_limit_rad_s": float(controller_velocity_limit_rad_s),
        "actions_per_chunk": int(actions_per_chunk or horizon),
        "aggregate_fn_name": "weighted_average",
    }
    if (
        execution_model["fps"] <= 0
        or execution_model["target_filter_tau_s"] < 0
        or execution_model["max_joint_velocity_rad_s"] <= 0
        or execution_model["controller_velocity_limit_rad_s"] <= 0
        or execution_model["max_joint_velocity_rad_s"]
        > execution_model["controller_velocity_limit_rad_s"]
        or execution_model["actions_per_chunk"] <= 0
        or execution_model["actions_per_chunk"] > horizon
    ):
        raise ValueError(f"invalid execution model: {execution_model}")
    execution_reference = execution_distributions(
        actions, position_states, execution_model
    )
    reference: dict[str, Any] = {
        "adjacent_step_abs_rad": adjacent_dist,
        "first_target_distance_abs_rad": first_dist,
        **execution_reference,
    }
    thresholds: dict[str, Any] = {
        # Exact demonstration maxima are diagnostic envelopes, not calibrated
        # safety boundaries. Hard rejection begins at a gross 2x excursion.
        "adjacent_step_abs_rad_per_joint": _gross_per_joint(
            adjacent_dist["max"],
            GROSS_ENVELOPE_MULTIPLIER,
            RAW_HARD_ABSOLUTE_MARGIN_RAD,
        ),
        "first_target_distance_abs_rad_per_joint": _gross_per_joint(
            first_dist["max"],
            GROSS_ENVELOPE_MULTIPLIER,
            RAW_HARD_ABSOLUTE_MARGIN_RAD,
        ),
        "repeated_first_std_rad_per_joint": adjacent_dist["q99"],
        "executed_step_abs_rad_per_joint": [
            execution_model["max_joint_velocity_rad_s"] / execution_model["fps"]
        ]
        * joints,
        "executed_displacement_abs_rad_per_joint": _gross_per_joint(
            execution_reference["executed_displacement_abs_rad"]["max"],
            GROSS_ENVELOPE_MULTIPLIER,
            RAW_HARD_ABSOLUTE_MARGIN_RAD,
        ),
        "execution_slew_saturation_fraction_per_joint": [
            min(1.0, max(SUSTAINED_SLEW_FRACTION, value * GROSS_ENVELOPE_MULTIPLIER))
            for value in execution_reference["execution_slew_saturation_fraction"]["max"]
        ],
    }
    quality_thresholds: dict[str, Any] = {
        "adjacent_step_abs_rad_per_joint": adjacent_dist["max"],
        "first_target_distance_abs_rad_per_joint": first_dist["max"],
        "executed_displacement_abs_rad_per_joint": execution_reference[
            "executed_displacement_abs_rad"
        ]["max"],
        "execution_slew_saturation_fraction_per_joint": execution_reference[
            "execution_slew_saturation_fraction"
        ]["max"],
    }
    return {
        "schema_version": 2,
        "kind": "demonstration-derived no-arm trajectory plausibility contract",
        "safety_warning": SAFETY_WARNING,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_roots": [str(path) for path in roots],
        "postprocessor": str(postprocessor),
        "postprocessor_sha256": sha256_file(postprocessor),
        "postprocessor_artifacts_sha256": postprocessor_artifact_hashes(postprocessor),
        "frames": int(len(states)),
        "episodes": int(len(np.unique(demonstrations["episode_index"]))),
        "horizon": horizon,
        "joints": joints,
        "state_width": int(states.shape[1]),
        "action_semantics": {
            "normalization": action_mode,
            "relative_joint_indices": np.flatnonzero(relative_joints).tolist(),
            "absolute_joint_indices": np.flatnonzero(~relative_joints).tolist(),
        },
        "execution_model": execution_model,
        "reference": reference,
        "rejection_thresholds": thresholds,
        "quality_thresholds": quality_thresholds,
        "report_only_metrics": [
            "adjacent_step_median_rad_per_joint",
            "l1_to_demo_rad_per_joint",
        ],
        "threshold_calibration": {
            "gross_envelope_multiplier": GROSS_ENVELOPE_MULTIPLIER,
            "raw_absolute_margin_rad": RAW_HARD_ABSOLUTE_MARGIN_RAD,
            "normalized_absolute_margin": NORMALIZED_HARD_ABSOLUTE_MARGIN,
            "sustained_slew_fraction": SUSTAINED_SLEW_FRACTION,
            "status": "retrospective provisional",
            "evidence": (
                "2x routes the 2026-08-30 ACT 1.27x first-target miss and the "
                "hard rejection of the 2026-09-01 corrupted-sim ACT 2.43x-6.21x starts"
            ),
        },
        "threshold_rationale": (
            "Exact genuine-demo maxima produce review warnings for decoded step, target "
            "distance, and modeled execution. Gross 2x excursions are hard rejections; "
            "exact-demo L1 is report-only. Demo q99 adjacent motion still hard-bounds "
            "repeated-input spread. The "
            "execution model replays the configured EMA and target slew over a single chunk; "
            "live weighted-average requests can be fed to the same model. Passing does not "
            "promote a checkpoint."
        ),
    }


def extract_fixture(args: argparse.Namespace) -> dict[str, Any]:

    if args.depth_encoding:
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
        tolerance_s=args.tolerance_s,
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
    for source in sorted(name for name in items[0] if name.startswith("observation.images.")):
        key = source.removeprefix("observation.images.")
        images = np.stack([item[source].numpy().transpose(1, 2, 0) for item in items])
        if key.endswith("_depth") and not args.depth_encoding:
            images = images.astype(np.float32, copy=False)
        elif images.dtype != np.uint8:
            images = np.rint(np.clip(images, 0.0, 1.0) * 255.0).astype(np.uint8)
        payload[key] = images
    args.npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.npz_out, **payload)
    return {
        "kind": "genuine no-arm observation fixture",
        "safety_warning": SAFETY_WARNING,
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "indices": indices,
        "depth_encoding": args.depth_encoding,
        "npz": str(args.npz_out),
        "npz_sha256": sha256_file(args.npz_out),
    }


def wire_features(scenario: str, *, sensor_profile: str = "deployment") -> dict[str, dict[str, Any]]:
    state_names = (
        JOINT_NAMES + EXTERNAL_EFFORT_NAMES
        if scenario == "act_rgbd14_masked"
        else JOINT_NAMES
    )
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
    }
    from wrist_cameras import describe

    for camera in describe(Path(__file__).resolve().parents[2], profile=sensor_profile):
        name = camera.role
        features[f"observation.images.{name}"] = {
            "dtype": "image",
            "shape": (camera.height, camera.width, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        }
        if scenario == "act_rgbd14_masked":
            channels = 1
            features[f"observation.images.{name}_depth"] = {
                "dtype": "image",
                "shape": (camera.height, camera.width, channels),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": channels == 1},
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
) -> tuple[dict[str, Any], list[str], list[str]]:
    validate_postprocessor_binding(contract, postprocessor)
    # chunks: fixture x repetition x horizon x joint
    states = fixture["state"][..., : chunks.shape[-1]]
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
    if contract.get("schema_version") == 2:
        execution_model = validate_execution_model(contract)
        metrics.update(
            execution_metrics(chunks, states[:, None, :], execution_model)
        )
    failures = compare_metrics(
        metrics, thresholds, primary_joints=primary_joints, limit_label="hard_limit"
    )
    warnings = compare_metrics(
        metrics,
        contract.get("quality_thresholds", {}),
        primary_joints=primary_joints,
        limit_label="demo_envelope",
    )
    return metrics, failures, warnings


def probe(args: argparse.Namespace) -> dict[str, Any]:
    import grpc
    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

    loaded = np.load(args.fixture, allow_pickle=False)
    fixture = {key: loaded[key] for key in loaded.files}
    contract = json.loads(args.contract.read_text())
    postprocessor_artifacts = validate_postprocessor_binding(contract, args.postprocessor)
    features = wire_features(args.scenario, sensor_profile=args.sensor_profile)
    from wrist_cameras import read_checkpoint_config, validate_checkpoint_views

    config_path, config = read_checkpoint_config(args.policy, args.checkpoint_config)
    roles = tuple(name.removeprefix("observation.images.") for name in features
                  if name.startswith("observation.images.") and not name.endswith("_depth"))
    image_shapes = {name.removeprefix("observation.images."): (f["shape"][2], f["shape"][0], f["shape"][1])
                    for name, f in features.items() if name.startswith("observation.images.")}
    validate_checkpoint_views(config, roles, use_depth=args.scenario == "act_rgbd14_masked",
                              image_shapes=image_shapes)
    required = ["state", "action", "action_is_pad"] + [
        name.removeprefix("observation.images.") for name in features
        if name.startswith("observation.images.")
    ]
    missing = [key for key in required if key not in fixture]
    if missing:
        raise ValueError(f"fixture lacks scenario features: {missing}")
    if fixture["action"].shape[1:] != (contract["horizon"], contract["joints"]):
        raise ValueError("fixture action shape does not match contract")

    channel = grpc.insecure_channel(args.server, grpc_channel_options())
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    grpc.channel_ready_future(channel).result(timeout=args.timeout)
    stub.Ready(services_pb2.Empty(), timeout=args.timeout)
    policy_type = {
        "act_rgb": "act",
        "act_rgbd14_masked": "act",
    }[args.scenario]
    setup = RemotePolicyConfig(
        policy_type, args.policy, features, contract["horizon"], "cuda"
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
            state_values = np.array(fixture["state"][fixture_index], copy=True)
            state_names = features["observation.state"]["names"]
            if args.scenario == "act_rgbd14_masked":
                state_values[7:14] = 0.0
            payload: dict[str, Any] = {
                name: float(value)
                for name, value in zip(state_names, state_values, strict=True)
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
    metrics, failures, warnings = evaluate_predictions(
        array, fixture, contract, args.postprocessor
    )
    warm = np.asarray(latencies[1:] or latencies)
    return {
        "schema_version": 2,
        "kind": "no-arm trajectory plausibility evaluation",
        "safety_warning": SAFETY_WARNING,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": (
            "reject" if failures else "review_no_arm_only" if warnings else "pass_no_arm_only"
        ),
        "scenario": args.scenario,
        "sensor_profile": args.sensor_profile,
        "checkpoint_config_sha256": sha256_file(config_path),
        "server": args.server,
        "policy": args.policy,
        "fixture": str(args.fixture),
        "fixture_sha256": sha256_file(args.fixture),
        "contract": str(args.contract),
        "contract_sha256": sha256_file(args.contract),
        "postprocessor_sha256": sha256_file(args.postprocessor),
        "postprocessor_artifacts_sha256": postprocessor_artifacts,
        "fixtures": int(array.shape[0]),
        "repetitions": int(array.shape[1]),
        "shape": list(array.shape),
        "cold_latency_ms": float(latencies[0]),
        "warm_p50_ms": float(np.percentile(warm, 50)),
        "warm_p95_ms": float(np.percentile(warm, 95)),
        "metrics": metrics,
        "rejection_failures": failures,
        "quality_warnings": warnings,
        "chunks": array.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    contract_parser = subparsers.add_parser("contract", help="derive a rejection envelope")
    contract_parser.add_argument("--dataset-root", action="append", type=Path, required=True)
    contract_parser.add_argument("--postprocessor", type=Path, required=True)
    contract_parser.add_argument("--horizon", type=int, default=16)
    contract_parser.add_argument("--fps", type=float, default=30.0)
    contract_parser.add_argument("--target-filter-tau-s", type=float, default=0.3)
    contract_parser.add_argument("--max-joint-velocity-rad-s", type=float, default=0.25)
    contract_parser.add_argument(
        "--controller-velocity-limit-rad-s", type=float, default=0.75
    )
    contract_parser.add_argument(
        "--actions-per-chunk",
        type=int,
        required=True,
        help="actions the rollout client executes from each served chunk",
    )
    contract_parser.add_argument("--json-out", type=Path, required=True)

    fixture_parser = subparsers.add_parser("fixture", help="freeze genuine inputs")
    fixture_parser.add_argument("--dataset-root", type=Path, required=True)
    fixture_parser.add_argument("--repo-id", required=True)
    fixture_parser.add_argument("--indices", required=True, help="comma-separated dataset indices")
    fixture_parser.add_argument("--depth-encoding")
    fixture_parser.add_argument("--fps", type=int, default=30)
    fixture_parser.add_argument("--horizon", type=int, default=16)
    fixture_parser.add_argument("--tolerance-s", type=float, default=1e-4)
    fixture_parser.add_argument("--npz-out", type=Path, required=True)

    probe_parser = subparsers.add_parser("probe", help="query a policy server without a robot")
    probe_parser.add_argument("--sensor-profile", choices=("deployment", "legacy-two-view"), default="deployment")
    probe_parser.add_argument(
        "scenario", choices=("act_rgb", "act_rgbd14_masked")
    )
    probe_parser.add_argument("--server", default=os.environ.get("TATBOT_POLICY_SERVER", ""),
                              help="host:port of the policy server (or TATBOT_POLICY_SERVER)")
    probe_parser.add_argument("--policy", required=True)
    probe_parser.add_argument("--checkpoint-config", type=Path, help="local config.json for a server-side checkpoint")
    probe_parser.add_argument("--fixture", type=Path, required=True)
    probe_parser.add_argument("--contract", type=Path, required=True)
    probe_parser.add_argument("--postprocessor", type=Path, required=True)
    probe_parser.add_argument("--repetitions", type=int, default=8)
    probe_parser.add_argument("--timeout", type=float, default=90.0)
    probe_parser.add_argument("--json-out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "contract":
        demos = read_demonstrations(args.dataset_root, args.horizon)
        result = build_contract(
            demos,
            args.postprocessor,
            args.dataset_root,
            fps=args.fps,
            target_filter_tau_s=args.target_filter_tau_s,
            max_joint_velocity_rad_s=args.max_joint_velocity_rad_s,
            controller_velocity_limit_rad_s=args.controller_velocity_limit_rad_s,
            actions_per_chunk=args.actions_per_chunk,
        )
    elif args.command == "fixture":
        result = extract_fixture(args)
    else:
        result = probe(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True) if hasattr(args, "json_out") else None
    output = args.json_out if hasattr(args, "json_out") else None
    if output is not None:
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.command == "probe":
        if result["verdict"] == "reject":
            raise SystemExit(1)
        if result["verdict"] == "review_no_arm_only":
            # Keep shell automation from promoting a review result as a pass.
            raise SystemExit(3)


if __name__ == "__main__":
    main()
