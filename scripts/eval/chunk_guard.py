#!/usr/bin/env python3
"""Fail-closed per-chunk checks for demonstration-derived policy contracts.

This module has no robot or camera dependencies.  The async policy server calls
it after postprocessing and before serializing any actions. Schema 2 separates
gross hard rejections from demonstration-envelope quality warnings and models
the follower's EMA plus target slew. Repeated-input variance and exact-demo L1
remain offline trajectory-plausibility measurements.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

RAW_PER_JOINT_METRICS = (
    "adjacent_step_abs_rad_per_joint",
    "first_target_distance_abs_rad_per_joint",
)
EXECUTION_PER_JOINT_METRICS = (
    "executed_step_abs_rad_per_joint",
    "executed_displacement_abs_rad_per_joint",
    "execution_slew_saturation_fraction_per_joint",
)
NORMALIZED_PER_JOINT_METRICS = (
    "normalized_adjacent_step_abs_per_joint",
    "normalized_endpoint_fraction_per_joint",
)
NORMALIZED_OVERALL_METRICS = ("normalized_endpoint_fraction_overall",)


@lru_cache(maxsize=8)
def load_contract(path: str) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text())
    if contract.get("kind") != "demonstration-derived no-arm trajectory plausibility contract":
        raise ValueError("unsupported plausibility contract kind")
    schema_version = contract.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError("unsupported plausibility contract schema")
    thresholds = contract.get("rejection_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("plausibility contract has no rejection_thresholds")
    missing = [name for name in RAW_PER_JOINT_METRICS if name not in thresholds]
    if missing:
        raise ValueError(f"plausibility contract lacks live metrics: {missing}")
    if schema_version == 2:
        if not isinstance(contract.get("quality_thresholds"), dict):
            raise ValueError("schema-2 plausibility contract has no quality_thresholds")
        validate_execution_model(contract)
    return contract


def validate_execution_model(contract: dict[str, Any]) -> dict[str, float | int | str]:
    model = contract.get("execution_model")
    if not isinstance(model, dict):
        raise ValueError("schema-2 plausibility contract has no execution_model")
    required = (
        "fps",
        "target_filter_tau_s",
        "max_joint_velocity_rad_s",
        "controller_velocity_limit_rad_s",
        "actions_per_chunk",
        "aggregate_fn_name",
    )
    missing = [name for name in required if name not in model]
    if missing:
        raise ValueError(f"execution_model lacks fields: {missing}")
    fps = float(model["fps"])
    tau = float(model["target_filter_tau_s"])
    velocity = float(model["max_joint_velocity_rad_s"])
    controller_velocity = float(model["controller_velocity_limit_rad_s"])
    actions = int(model["actions_per_chunk"])
    horizon = int(contract["horizon"])
    if (
        fps <= 0
        or tau < 0
        or velocity <= 0
        or controller_velocity <= 0
        or velocity > controller_velocity
        or actions <= 0
        or actions > horizon
        or model["aggregate_fn_name"] != "weighted_average"
    ):
        raise ValueError(
            "invalid execution_model values: "
            f"fps={fps}, tau={tau}, velocity={velocity}, "
            f"controller_velocity={controller_velocity}, actions={actions}, horizon={horizon}, "
            f"aggregate={model['aggregate_fn_name']}"
        )
    return model


def simulate_executed_actions(
    decoded_action: Any,
    observation_position: Any,
    execution_model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the follower EMA and target slew over an already-aggregated sequence.

    A frozen policy chunk has no older overlapping queue, so offline evaluation
    feeds it as the post-aggregation request. Live weighted averaging can be
    replayed by passing its resulting request sequence to this same function.
    """

    decoded = np.asarray(decoded_action, dtype=np.float64)
    state = np.asarray(observation_position, dtype=np.float64)
    if decoded.ndim < 2:
        raise ValueError(f"execution model expected (..., horizon, joints), got {decoded.shape}")
    joints = decoded.shape[-1]
    prefix = decoded.shape[:-2]
    try:
        state = np.broadcast_to(state, prefix + (joints,)).copy()
    except ValueError as error:
        raise ValueError(
            f"execution state shape {state.shape} cannot broadcast to {prefix + (joints,)}"
        ) from error

    fps = float(execution_model["fps"])
    tau = float(execution_model["target_filter_tau_s"])
    velocity = float(execution_model["max_joint_velocity_rad_s"])
    count = int(execution_model["actions_per_chunk"])
    if fps <= 0 or tau < 0 or velocity <= 0 or count <= 0 or count > decoded.shape[-2]:
        raise ValueError("invalid execution model")
    dt = 1.0 / fps
    alpha = 1.0 if tau == 0 else dt / (tau + dt)
    budget = velocity * dt
    filtered = state.copy()
    sent = state.copy()
    sent_steps = []
    saturated_steps = []
    for target in np.moveaxis(decoded[..., :count, :], -2, 0):
        filtered += alpha * (target - filtered)
        delta = filtered - sent
        saturated_steps.append(np.abs(delta) > budget + 1e-12)
        sent += np.clip(delta, -budget, budget)
        sent_steps.append(sent.copy())
    return np.stack(sent_steps, axis=-2), np.stack(saturated_steps, axis=-2)


def execution_metrics(
    decoded_action: Any,
    observation_position: Any,
    execution_model: dict[str, Any],
) -> dict[str, list[float]]:
    """Summarize the requests that the configured EMA and slew would send."""

    decoded = np.asarray(decoded_action, dtype=np.float64)
    state = np.asarray(observation_position, dtype=np.float64)
    prefix = decoded.shape[:-2]
    joints = decoded.shape[-1]
    state = np.broadcast_to(state, prefix + (joints,))
    sent, saturated = simulate_executed_actions(decoded, state, execution_model)
    previous = np.concatenate((state[..., None, :], sent[..., :-1, :]), axis=-2)
    steps = np.abs(sent - previous)
    displacement = np.abs(sent - state[..., None, :])
    reduce_axes = tuple(range(steps.ndim - 1))
    saturation_fraction = saturated.mean(axis=-2)
    saturation_axes = tuple(range(saturation_fraction.ndim - 1))
    return {
        "executed_step_abs_rad_per_joint": steps.max(axis=reduce_axes).tolist(),
        "executed_displacement_abs_rad_per_joint": displacement.max(
            axis=reduce_axes
        ).tolist(),
        "execution_slew_saturation_fraction_per_joint": (
            saturation_fraction.max(axis=saturation_axes).tolist()
            if saturation_axes
            else saturation_fraction.tolist()
        ),
    }


def compare_metrics(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
    *,
    primary_joints: int,
    limit_label: str,
) -> list[str]:
    """Compare common scalar/per-joint metrics without assigning their policy meaning."""

    findings: list[str] = []
    for metric, limit in thresholds.items():
        if metric not in metrics:
            continue
        values = metrics[metric]
        if isinstance(limit, list):
            if not isinstance(values, list) or len(values) != len(limit):
                raise ValueError(f"plausibility contract width mismatch for {metric}")
            for joint, (value, maximum) in enumerate(zip(values, limit, strict=True)):
                if joint < primary_joints and float(value) > float(maximum) + 1e-6:
                    findings.append(
                        f"{metric}[{joint}]={float(value):.8g} > "
                        f"{limit_label}={float(maximum):.8g}"
                    )
        elif float(values) > float(limit) + 1e-9:
            findings.append(
                f"{metric}={float(values):.8g} > {limit_label}={float(limit):.8g}"
            )
    return findings


def evaluate_chunk(
    normalized_action: Any,
    decoded_action: Any,
    observation_state: Any,
    contract: dict[str, Any],
    *,
    primary_joints: int = 6,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Evaluate one action chunk against the live subset of a contract."""

    normalized = None if normalized_action is None else np.asarray(normalized_action, dtype=np.float64)
    decoded = np.asarray(decoded_action, dtype=np.float64)
    state = np.asarray(observation_state, dtype=np.float64)
    if normalized is not None and normalized.ndim == 3 and normalized.shape[0] == 1:
        normalized = normalized[0]
    if decoded.ndim == 3 and decoded.shape[0] == 1:
        decoded = decoded[0]
    state = state.reshape(-1)
    expected = (int(contract["horizon"]), int(contract["joints"]))
    expected_state_width = int(contract.get("state_width", expected[1]))
    if expected_state_width < expected[1]:
        raise ValueError(
            f"plausibility contract state_width {expected_state_width} is smaller than "
            f"its {expected[1]} action joints"
        )
    normalized_required = any(
        name in contract["rejection_thresholds"]
        for name in NORMALIZED_PER_JOINT_METRICS + NORMALIZED_OVERALL_METRICS
    )
    normalized_required = normalized_required or any(
        name in contract.get("quality_thresholds", {})
        for name in NORMALIZED_PER_JOINT_METRICS + NORMALIZED_OVERALL_METRICS
    )
    if decoded.shape != expected or state.shape != (expected_state_width,):
        raise ValueError(
            "plausibility guard shape mismatch: "
            f"normalized={None if normalized is None else normalized.shape}, "
            f"decoded={decoded.shape}, state={state.shape}, "
            f"contract={expected}"
        )
    position_state = state[: expected[1]]
    if normalized_required and (normalized is None or normalized.shape != expected):
        raise ValueError(
            "plausibility guard normalized shape mismatch: "
            f"normalized={None if normalized is None else normalized.shape}, contract={expected}"
        )
    if (
        (normalized is not None and not np.isfinite(normalized).all())
        or not np.isfinite(decoded).all()
        or not np.isfinite(state).all()
    ):
        raise ValueError("plausibility guard received non-finite values")

    # GR00T's saved relative-action decoder clips to [-1, 1].  Applying that
    # same operation here makes the normalized metrics exactly comparable to
    # the inverse-decoded genuine demonstrations used to derive the contract.
    metrics: dict[str, Any] = {
        "adjacent_step_abs_rad_per_joint": np.abs(np.diff(decoded, axis=0))
        .max(axis=0)
        .tolist(),
        "first_target_distance_abs_rad_per_joint": np.abs(
            decoded[0] - position_state
        ).tolist(),
    }
    thresholds = contract["rejection_thresholds"]
    if contract.get("schema_version") == 2:
        model = validate_execution_model(contract)
        metrics.update(execution_metrics(decoded, position_state, model))
    if normalized_required:
        # GR00T's saved relative-action decoder clips to [-1, 1]. Applying the
        # same operation here matches its inverse-decoded demonstration contract.
        clipped = np.clip(normalized, -1.0, 1.0)
        endpoint = np.abs(clipped) >= 1.0 - 1e-6
        metrics.update(
            {
                "normalized_adjacent_step_abs_per_joint": np.abs(np.diff(clipped, axis=0))
                .max(axis=0)
                .tolist(),
                "normalized_endpoint_fraction_per_joint": endpoint.mean(axis=0).tolist(),
                "normalized_endpoint_fraction_overall": float(endpoint.mean()),
            }
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


def enforce_chunk(
    normalized_action: Any,
    decoded_action: Any,
    observation_state: Any,
    contract_path: str,
) -> dict[str, Any]:
    """Return metrics for a passing chunk or raise before it reaches the wire."""

    metrics, failures, warnings = evaluate_chunk(
        normalized_action,
        decoded_action,
        observation_state,
        load_contract(contract_path),
    )
    if failures:
        raise RuntimeError(
            "Tatbot chunk rejected by demonstration plausibility contract: " + "; ".join(failures)
        )
    return {
        "verdict": "review_no_arm_only" if warnings else "pass_no_arm_only",
        "metrics": metrics,
        "quality_warnings": warnings,
    }
