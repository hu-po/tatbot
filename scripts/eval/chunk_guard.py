#!/usr/bin/env python3
"""Fail-closed per-chunk checks for demonstration-derived policy contracts.

This module has no robot or camera dependencies.  The async policy server calls
it after postprocessing and before serializing any actions.  It deliberately
enforces only measurements available from one observation/chunk; repeated-input
variance and L1-to-demonstration remain offline trajectory-plausibility gates.
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
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported plausibility contract schema")
    thresholds = contract.get("rejection_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("plausibility contract has no rejection_thresholds")
    missing = [name for name in RAW_PER_JOINT_METRICS if name not in thresholds]
    if missing:
        raise ValueError(f"plausibility contract lacks live metrics: {missing}")
    return contract


def evaluate_chunk(
    normalized_action: Any,
    decoded_action: Any,
    observation_state: Any,
    contract: dict[str, Any],
    *,
    primary_joints: int = 6,
) -> tuple[dict[str, Any], list[str]]:
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
    normalized_required = any(
        name in contract["rejection_thresholds"]
        for name in NORMALIZED_PER_JOINT_METRICS + NORMALIZED_OVERALL_METRICS
    )
    if decoded.shape != expected or state.shape != (expected[1],):
        raise ValueError(
            "plausibility guard shape mismatch: "
            f"normalized={None if normalized is None else normalized.shape}, "
            f"decoded={decoded.shape}, state={state.shape}, "
            f"contract={expected}"
        )
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
        "first_target_distance_abs_rad_per_joint": np.abs(decoded[0] - state).tolist(),
    }
    thresholds = contract["rejection_thresholds"]
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
    failures: list[str] = []
    for metric in RAW_PER_JOINT_METRICS + NORMALIZED_PER_JOINT_METRICS:
        if metric not in thresholds:
            continue
        values = metrics[metric]
        limits = thresholds[metric]
        if len(values) != expected[1] or len(limits) != expected[1]:
            raise ValueError(f"plausibility contract width mismatch for {metric}")
        for joint, (value, maximum) in enumerate(zip(values, limits, strict=True)):
            if joint < primary_joints and value > float(maximum) + 1e-6:
                failures.append(f"{metric}[{joint}]={value:.8g} > demo_limit={maximum:.8g}")
    for metric in NORMALIZED_OVERALL_METRICS:
        if metric not in thresholds:
            continue
        value = float(metrics[metric])
        maximum = float(thresholds[metric])
        if value > maximum + 1e-9:
            failures.append(f"{metric}={value:.8g} > demo_limit={maximum:.8g}")
    return metrics, failures


def enforce_chunk(
    normalized_action: Any,
    decoded_action: Any,
    observation_state: Any,
    contract_path: str,
) -> dict[str, Any]:
    """Return metrics for a passing chunk or raise before it reaches the wire."""

    metrics, failures = evaluate_chunk(
        normalized_action,
        decoded_action,
        observation_state,
        load_contract(contract_path),
    )
    if failures:
        raise RuntimeError(
            "Tatbot chunk rejected by demonstration plausibility contract: " + "; ".join(failures)
        )
    return metrics
