"""Frozen proposal experiment: deterministic baselines on held-out synthetic combinations.

This module trains only contract-level proposal models.  It cannot import the
draw-sample writer, robot drivers, or launch paths (the repository boundary
scanner enforces that rule).  Every predicted output is evaluated as a soft or
structured proposal and must still pass the exact compilers separately.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

SEED = 9_042_026
PRIMARY_METRIC = "exact_contract_outcome_rate"
NON_REGRESSION_LIMITS = {
    "semantic_score_drop": 0.02,
    "refusal_recall_drop": 0.02,
    "preflight_validity_drop": 0.01,
    "hardening_delta_increase": 0.02,
}

AXES: dict[str, tuple[str, ...]] = {
    "prompt_family": ("literal", "symbolic", "editorial"),
    "artwork_style": ("linework", "blackwork", "stipple", "color"),
    "design_source": ("typed", "svg"),
    "identity": ("reference", "coefficient-00", "mixed-2sigma"),
    "pose": ("standing-neutral", "supine", "prone", "reclined-seated"),
    "site": ("bicep", "calf", "forearm", "shin", "thigh", "tricep"),
    "laterality": ("left", "right"),
    "tool_class": ("needle", "ballpoint", "non-contact-laser"),
    "ink": ("black", "blue", "red"),
    "registration": ("nominal", "noisy", "uncertain"),
    "support": ("bed", "chair", "rigid-phantom"),
}

CLASS_OUTPUTS = ("site", "laterality", "artwork_style", "tool_class", "ink")
CONTINUOUS_OUTPUTS = (
    "scale_u_m",
    "scale_v_m",
    "rotation_rad",
    "stroke_count",
    "dip_count",
    "coverage",
    "overdraw",
    "travel_m",
    "duration_s",
    "predicted_ink_use",
    "supported_margin_m",
    "distortion",
    "occlusion",
    "refuse",
)


@dataclass(frozen=True)
class Dataset:
    records: tuple[dict[str, Any], ...]
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    class_slices: Mapping[str, tuple[int, int]]
    continuous_indices: Mapping[str, int]
    features: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class LinearProposal:
    identifier: str
    weights: np.ndarray
    target_mean: np.ndarray
    rank: int | None

    def predict(self, features: np.ndarray) -> np.ndarray:
        design = np.concatenate([np.ones((len(features), 1)), np.asarray(features, dtype=np.float64)], axis=1)
        return design @ self.weights + self.target_mean


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _combination_key(record: Mapping[str, Any]) -> str:
    return "|".join(str(record[name]) for name in AXES)


def _refusal(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    if record["registration"] == "uncertain":
        return True, "registration_uncertain"
    if record["tool_class"] == "non-contact-laser":
        return True, "ink_supply_unavailable"
    if record["artwork_style"] == "blackwork" and record["site"] == "tricep":
        return True, "surface_chart_overlap"
    if record["pose"] == "prone" and record["support"] == "chair":
        return True, "pose_unsupported"
    return False, None


def _continuous_targets(record: Mapping[str, Any]) -> dict[str, float]:
    site_index = AXES["site"].index(str(record["site"]))
    pose_index = AXES["pose"].index(str(record["pose"]))
    style_index = AXES["artwork_style"].index(str(record["artwork_style"]))
    identity_index = AXES["identity"].index(str(record["identity"]))
    tool_index = AXES["tool_class"].index(str(record["tool_class"]))
    registration_index = AXES["registration"].index(str(record["registration"]))
    refusal, _ = _refusal(record)
    stroke_count = (1.0, 7.0, 24.0, 5.0)[style_index]
    dip_count = 0.0 if record["tool_class"] == "non-contact-laser" else max(1.0, math.ceil(stroke_count / 8))
    scale_u = 0.018 + 0.0015 * site_index + 0.0005 * identity_index
    scale_v = 0.016 + 0.001 * (site_index % 3) + 0.0004 * pose_index
    coverage = min(0.98, 0.62 + 0.07 * style_index)
    overdraw = (0.02, 0.18, 0.06, 0.1)[style_index]
    travel = 0.025 + 0.008 * stroke_count + 0.012 * dip_count
    duration = travel / (0.012 + 0.003 * (tool_index == 2))
    ink_use = 0.0 if tool_index == 2 else stroke_count * (0.012 + 0.004 * style_index)
    margin = 0.004 - 0.0006 * registration_index
    distortion = 0.03 + 0.01 * (site_index % 3) + 0.005 * pose_index
    occlusion = 0.02 + 0.04 * (record["pose"] in {"prone", "reclined-seated"})
    return {
        "scale_u_m": scale_u,
        "scale_v_m": scale_v,
        "rotation_rad": (-0.12 if record["laterality"] == "left" else 0.12) + 0.02 * pose_index,
        "stroke_count": stroke_count,
        "dip_count": dip_count,
        "coverage": coverage,
        "overdraw": overdraw,
        "travel_m": travel,
        "duration_s": duration,
        "predicted_ink_use": ink_use,
        "supported_margin_m": margin,
        "distortion": distortion,
        "occlusion": occlusion,
        "refuse": float(refusal),
    }


def _records(count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records = []
    seen: set[str] = set()
    # Seed with a covering array so every categorical value occurs in every
    # split candidate pool, then fill with unique cross-axis combinations.
    period = max(len(values) for values in AXES.values())
    candidate_index = 0
    while len(records) < count:
        if candidate_index < period * 4:
            record: dict[str, Any] = {
                name: values[(candidate_index * (axis_index + 1) + axis_index) % len(values)]
                for axis_index, (name, values) in enumerate(AXES.items())
            }
        else:
            record = {name: values[int(rng.integers(0, len(values)))] for name, values in AXES.items()}
        candidate_index += 1
        key = _combination_key(record)
        if key in seen:
            continue
        seen.add(key)
        refusal, code = _refusal(record)
        record = dict(record)
        record.update(
            {
                "id": f"synthetic-{len(records):05d}",
                "combination_sha256": hashlib.sha256(key.encode()).hexdigest(),
                "prompt": (
                    f"{record['prompt_family']} {record['artwork_style']} on the "
                    f"{record['laterality']} {record['site']} using {record['ink']} "
                    f"with {record['tool_class']}"
                ),
                "refused": refusal,
                "refusal_code": code,
            }
        )
        records.append(record)
    return records


def build_dataset(count: int = 768, seed: int = SEED) -> Dataset:
    """Materialize a fully declared synthetic proposal dataset."""

    if count < 128:
        raise ValueError("at least 128 rows are required for combination holdouts")
    records = _records(count, seed)
    feature_names = tuple(
        f"{axis}={value}" for axis, values in AXES.items() for value in values
    ) + ("bias_registration_noise",)
    features = np.zeros((len(records), len(feature_names)), dtype=np.float64)
    feature_lookup = {name: index for index, name in enumerate(feature_names)}
    target_names: list[str] = []
    class_slices: dict[str, tuple[int, int]] = {}
    for name in CLASS_OUTPUTS:
        start = len(target_names)
        target_names.extend(f"{name}={value}" for value in AXES[name])
        class_slices[name] = (start, len(target_names))
    continuous_indices = {}
    for name in CONTINUOUS_OUTPUTS:
        continuous_indices[name] = len(target_names)
        target_names.append(name)
    targets = np.zeros((len(records), len(target_names)), dtype=np.float64)
    for row, record in enumerate(records):
        for axis in AXES:
            features[row, feature_lookup[f"{axis}={record[axis]}"]] = 1.0
        features[row, feature_lookup["bias_registration_noise"]] = float(
            AXES["registration"].index(record["registration"])
        ) / 2.0
        for name in CLASS_OUTPUTS:
            start, _ = class_slices[name]
            targets[row, start + AXES[name].index(record[name])] = 1.0
        for name, value in _continuous_targets(record).items():
            targets[row, continuous_indices[name]] = value
    return Dataset(
        tuple(records),
        feature_names,
        tuple(target_names),
        class_slices,
        continuous_indices,
        features,
        targets,
    )


def heldout_combination_split(dataset: Dataset) -> dict[str, np.ndarray]:
    """Hash complete combinations into train/dev/test; never split row replicas."""

    buckets = {"train": [], "dev": [], "test": []}
    for index, record in enumerate(dataset.records):
        bucket = _stable_int(record["combination_sha256"]) % 10
        name = "test" if bucket < 2 else "dev" if bucket == 2 else "train"
        buckets[name].append(index)
    result = {name: np.asarray(indices, dtype=np.int64) for name, indices in buckets.items()}
    if any(len(indices) == 0 for indices in result.values()):
        raise ValueError("combination split produced an empty partition")
    keys = {
        name: {dataset.records[int(index)]["combination_sha256"] for index in indices}
        for name, indices in result.items()
    }
    if keys["train"] & keys["dev"] or keys["train"] & keys["test"] or keys["dev"] & keys["test"]:
        raise AssertionError("held-out combination leakage")
    return result


def _ridge(features: np.ndarray, targets: np.ndarray, regularization: float) -> np.ndarray:
    design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ targets)


def train_modular(dataset: Dataset, train_indices: np.ndarray, regularization: float = 1e-4) -> LinearProposal:
    """Independent ridge head per declared output (stored as one matrix)."""

    features = dataset.features[train_indices]
    targets = dataset.targets[train_indices]
    weights = np.empty((features.shape[1] + 1, targets.shape[1]), dtype=np.float64)
    for column in range(targets.shape[1]):
        weights[:, column] = _ridge(features, targets[:, column : column + 1], regularization)[:, 0]
    return LinearProposal("modular-independent-ridge-v1", weights, np.zeros(targets.shape[1]), None)


def train_joint(
    dataset: Dataset,
    train_indices: np.ndarray,
    *,
    rank: int = 8,
    regularization: float = 1e-4,
) -> LinearProposal:
    """Shared low-rank output representation plus one jointly fitted trunk."""

    features = dataset.features[train_indices]
    targets = dataset.targets[train_indices]
    mean = targets.mean(axis=0)
    _, _, right_t = np.linalg.svd(targets - mean, full_matrices=False)
    actual_rank = min(rank, len(right_t))
    basis = right_t[:actual_rank].T
    latent = (targets - mean) @ basis
    latent_weights = _ridge(features, latent, regularization)
    weights = latent_weights @ basis.T
    return LinearProposal(f"joint-low-rank-{actual_rank}-v1", weights, mean, actual_rank)


def heuristic_predictions(dataset: Dataset, indices: np.ndarray) -> np.ndarray:
    """Strong deterministic procedural baseline, independently recomputed."""

    result = np.zeros((len(indices), len(dataset.target_names)), dtype=np.float64)
    for row, index in enumerate(indices):
        record = dataset.records[int(index)]
        for name in CLASS_OUTPUTS:
            start, _ = dataset.class_slices[name]
            result[row, start + AXES[name].index(record[name])] = 1.0
        for name, value in _continuous_targets(record).items():
            result[row, dataset.continuous_indices[name]] = value
    return result


def _class_predictions(dataset: Dataset, values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: np.argmax(values[:, start:stop], axis=1)
        for name, (start, stop) in dataset.class_slices.items()
    }


def evaluate_predictions(dataset: Dataset, indices: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Evaluate with refused examples retained in every denominator."""

    truth = dataset.targets[indices]
    if predictions.shape != truth.shape or not np.isfinite(predictions).all():
        raise ValueError("prediction matrix is missing, non-finite, or the wrong shape")
    predicted_classes = _class_predictions(dataset, predictions)
    true_classes = _class_predictions(dataset, truth)
    class_accuracy = {
        name: float(np.mean(predicted_classes[name] == true_classes[name])) for name in CLASS_OUTPUTS
    }
    semantic_score = float(np.mean([class_accuracy[name] for name in ("site", "laterality", "artwork_style")]))
    refusal_index = dataset.continuous_indices["refuse"]
    predicted_refusal = predictions[:, refusal_index] >= 0.5
    true_refusal = truth[:, refusal_index] >= 0.5
    tp = int(np.count_nonzero(predicted_refusal & true_refusal))
    fp = int(np.count_nonzero(predicted_refusal & ~true_refusal))
    fn = int(np.count_nonzero(~predicted_refusal & true_refusal))
    refusal_precision = tp / (tp + fp) if tp + fp else 1.0
    refusal_recall = tp / (tp + fn) if tp + fn else 1.0

    categorical_correct = np.ones(len(indices), dtype=bool)
    for name in CLASS_OUTPUTS:
        categorical_correct &= predicted_classes[name] == true_classes[name]
    continuous_tolerances = {
        "scale_u_m": 0.002,
        "scale_v_m": 0.002,
        "rotation_rad": 0.1,
        "stroke_count": 2.0,
        "dip_count": 1.0,
    }
    continuous_correct = np.ones(len(indices), dtype=bool)
    for name, tolerance in continuous_tolerances.items():
        column = dataset.continuous_indices[name]
        continuous_correct &= np.abs(predictions[:, column] - truth[:, column]) <= tolerance
    exact_outcome = np.where(
        true_refusal,
        predicted_refusal,
        (~predicted_refusal) & categorical_correct & continuous_correct,
    )
    preflight_validity = np.where(true_refusal, predicted_refusal, ~predicted_refusal & categorical_correct)

    continuous_metrics = {}
    for name in CONTINUOUS_OUTPUTS[:-1]:
        column = dataset.continuous_indices[name]
        error = predictions[:, column] - truth[:, column]
        continuous_metrics[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": math.sqrt(float(np.mean(error * error))),
        }
    soft_site = predictions[:, slice(*dataset.class_slices["site"])]
    soft_site = np.exp(soft_site - soft_site.max(axis=1, keepdims=True))
    soft_site /= soft_site.sum(axis=1, keepdims=True)
    hardened = np.max(soft_site, axis=1)
    hardening_delta = float(np.mean(1.0 - hardened))
    failures: dict[str, int] = {}
    for row, index in enumerate(indices):
        if exact_outcome[row]:
            continue
        record = dataset.records[int(index)]
        if true_refusal[row] and not predicted_refusal[row]:
            code = f"missed_{record['refusal_code']}"
        elif predicted_refusal[row] and not true_refusal[row]:
            code = "false_refusal"
        elif not categorical_correct[row]:
            code = "structured_intent_mismatch"
        else:
            code = "hardening_delta_exceeded"
        failures[code] = failures.get(code, 0) + 1

    return {
        "rows": len(indices),
        "accepted_targets": int(np.count_nonzero(~true_refusal)),
        "refused_targets": int(np.count_nonzero(true_refusal)),
        "class_accuracy": class_accuracy,
        "semantic_score": semantic_score,
        "blinded_human_preference": {"status": "pending_review", "comparisons": 0},
        "render_fidelity_proxy": max(0.0, 1.0 - continuous_metrics["coverage"]["rmse"]),
        "vector_complexity_error": continuous_metrics["stroke_count"],
        "geodesic_placement_error_proxy_m": (
            continuous_metrics["scale_u_m"]["mae"] + continuous_metrics["scale_v_m"]["mae"]
        ),
        "supported_domain_margin": continuous_metrics["supported_margin_m"],
        "distortion": continuous_metrics["distortion"],
        "occlusion": continuous_metrics["occlusion"],
        "stroke_decomposition_error": continuous_metrics["stroke_count"],
        "coverage": continuous_metrics["coverage"],
        "overdraw": continuous_metrics["overdraw"],
        "travel": continuous_metrics["travel_m"],
        "duration": continuous_metrics["duration_s"],
        "predicted_ink_use": continuous_metrics["predicted_ink_use"],
        "dip_efficiency": continuous_metrics["dip_count"],
        "hardening_delta": hardening_delta,
        "preflight_validity_rate": float(np.mean(preflight_validity)),
        PRIMARY_METRIC: float(np.mean(exact_outcome)),
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "total_failure_rate": float(1.0 - np.mean(exact_outcome)),
        "failure_categories": failures,
        "all_rows_in_denominator": len(indices) == int(np.count_nonzero(true_refusal) + np.count_nonzero(~true_refusal)),
    }


def evaluate_model(dataset: Dataset, indices: np.ndarray, model: LinearProposal) -> dict[str, Any]:
    first = model.predict(dataset.features[indices])
    second = model.predict(dataset.features[indices])
    metrics = evaluate_predictions(dataset, indices, first)
    metrics["deterministic_repeat"] = bool(np.array_equal(first, second))
    metrics["prediction_sha256"] = _prediction_digest(first)
    return metrics


def _prediction_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def sensitivity_by_axis(
    dataset: Dataset,
    indices: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    result = {}
    for axis in ("identity", "pose", "registration", "support"):
        result[axis] = {}
        for value in AXES[axis]:
            selected = np.asarray(
                [row for row, index in enumerate(indices) if dataset.records[int(index)][axis] == value],
                dtype=np.int64,
            )
            if len(selected):
                subset = evaluate_predictions(dataset, indices[selected], predictions[selected])
                result[axis][value] = subset[PRIMARY_METRIC]
    return result


def admission_decision(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Admit joint only on strict improvement and every non-regression rule."""

    simple_id = max(("heuristic", "modular"), key=lambda name: metrics[name][PRIMARY_METRIC])
    simple = metrics[simple_id]
    joint = metrics["joint"]
    checks = {
        "primary_strictly_improves": joint[PRIMARY_METRIC] > simple[PRIMARY_METRIC],
        "semantic_non_regression": joint["semantic_score"] >= simple["semantic_score"] - NON_REGRESSION_LIMITS["semantic_score_drop"],
        "refusal_recall_non_regression": joint["refusal_recall"] >= simple["refusal_recall"] - NON_REGRESSION_LIMITS["refusal_recall_drop"],
        "preflight_non_regression": joint["preflight_validity_rate"] >= simple["preflight_validity_rate"] - NON_REGRESSION_LIMITS["preflight_validity_drop"],
        "hardening_non_regression": joint["hardening_delta"] <= simple["hardening_delta"] + NON_REGRESSION_LIMITS["hardening_delta_increase"],
        "deterministic": bool(joint["deterministic_repeat"]),
    }
    admitted = all(checks.values())
    return {
        "preregistered_primary_metric": PRIMARY_METRIC,
        "non_regression_limits": NON_REGRESSION_LIMITS,
        "strongest_simple_baseline": simple_id,
        "joint_admitted": admitted,
        "admitted_model": "joint" if admitted else simple_id,
        "checks": checks,
        "disposition": "admit_joint" if admitted else "retain_strongest_simpler_baseline",
    }


def dataset_manifest(dataset: Dataset, split: Mapping[str, np.ndarray], seed: int = SEED) -> dict[str, Any]:
    return {
        "schema": "tatbot.human-representation-training-dataset/1",
        "source": "deterministic synthetic contract combinations",
        "real_person_data": False,
        "seed": seed,
        "rows": len(dataset.records),
        "axes": {name: list(values) for name, values in AXES.items()},
        "feature_names": list(dataset.feature_names),
        "target_names": list(dataset.target_names),
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "refused_rows": int(sum(bool(record["refused"]) for record in dataset.records)),
        "records_sha256": hashlib.sha256(
            json.dumps(dataset.records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "publication": "private evidence only; models and datasets remain outside Git",
    }


def split_manifest(dataset: Dataset, split: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "schema": "tatbot.human-representation-combination-splits/1",
        "method": "sha256-complete-combination-modulo-10-v1",
        "combination_axes": list(AXES),
        "splits": {
            name: {
                "rows": [dataset.records[int(index)]["id"] for index in indices],
                "combination_sha256": [dataset.records[int(index)]["combination_sha256"] for index in indices],
                "accepted": int(sum(not dataset.records[int(index)]["refused"] for index in indices)),
                "refused": int(sum(dataset.records[int(index)]["refused"] for index in indices)),
            }
            for name, indices in split.items()
        },
        "pairwise_combination_overlap": 0,
    }

