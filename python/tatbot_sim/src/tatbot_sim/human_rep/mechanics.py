"""Frozen non-human phantom contact models, instrument adapter, and fit gates.

Only calibrated force/displacement traces can qualify a compliant model.  The
synthetic helpers exercise the implementation hermetically, but their
calibration contract is deliberately unqualified and cannot be used to admit
``compliant-heightfield-v1``.
"""

from __future__ import annotations

import csv
import hashlib
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from tatbot_contracts.digest import sha256_file

from tatbot_sim.human_rep.contracts import ContractError, canonical_digest, validate_contract

CALIBRATION_SCHEMA = "tatbot.force-displacement-calibration/1"
TRACE_COLUMNS = (
    "time_s",
    "commanded_indentation_m",
    "measured_indentation_m",
    "measured_force_n",
    "temperature_c",
)


@dataclass(frozen=True)
class IndentationTrace:
    time_s: np.ndarray
    commanded_indentation_m: np.ndarray
    indentation_m: np.ndarray
    velocity_m_s: np.ndarray
    force_n: np.ndarray
    temperature_c: np.ndarray
    source_sha256: str
    batch_id: str
    rate_m_s: float
    angle_deg: float
    repeat: int
    lateral_drag: bool = False

    def __post_init__(self) -> None:
        arrays = (
            self.time_s,
            self.commanded_indentation_m,
            self.indentation_m,
            self.velocity_m_s,
            self.force_n,
            self.temperature_c,
        )
        if not arrays or any(np.asarray(value).shape != np.asarray(arrays[0]).shape for value in arrays):
            raise ContractError("tissue_model_uncalibrated", "$.trace", "trace arrays differ in shape")
        if np.asarray(self.time_s).ndim != 1 or len(self.time_s) < 4:
            raise ContractError("tissue_model_uncalibrated", "$.trace", "at least four scalar samples are required")
        if any(not np.isfinite(value).all() for value in arrays):
            raise ContractError("tissue_model_uncalibrated", "$.trace", "trace contains non-finite values")
        if np.any(np.diff(self.time_s) <= 0):
            raise ContractError("tissue_model_uncalibrated", "$.trace.time_s", "timestamps must strictly increase")


@dataclass(frozen=True)
class KelvinVoigtWinkler:
    stiffness_n_m: float
    damping_n_s_m: float
    covariance: np.ndarray
    training_domain: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        if not math.isfinite(self.stiffness_n_m) or self.stiffness_n_m <= 0:
            raise ContractError("contact_solver_unstable", "$.stiffness_n_m", "must be finite and positive")
        if not math.isfinite(self.damping_n_s_m) or self.damping_n_s_m < 0:
            raise ContractError("contact_solver_unstable", "$.damping_n_s_m", "must be finite and nonnegative")
        covariance = np.asarray(self.covariance)
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            raise ContractError("contact_solver_unstable", "$.covariance", "expected finite 2x2 covariance")

    @property
    def relaxation_time_s(self) -> float:
        return self.damping_n_s_m / self.stiffness_n_m

    def force(self, indentation_m: Any, velocity_m_s: Any) -> np.ndarray:
        indentation = np.asarray(indentation_m, dtype=np.float64)
        velocity = np.asarray(velocity_m_s, dtype=np.float64)
        if indentation.shape != velocity.shape or not np.isfinite(indentation).all() or not np.isfinite(velocity).all():
            raise ContractError("contact_solver_unstable", "$.state", "indentation and velocity must be matching finite arrays")
        if np.any(indentation < -1e-12):
            raise ContractError("contact_solver_unstable", "$.indentation_m", "negative contact indentation")
        return self.stiffness_n_m * np.maximum(indentation, 0.0) + self.damping_n_s_m * velocity

    def stored_energy_j(self, indentation_m: Any) -> np.ndarray:
        indentation = np.asarray(indentation_m, dtype=np.float64)
        if not np.isfinite(indentation).all() or np.any(indentation < -1e-12):
            raise ContractError("contact_solver_unstable", "$.indentation_m", "invalid energy state")
        return 0.5 * self.stiffness_n_m * np.maximum(indentation, 0.0) ** 2

    def dissipation_w(self, velocity_m_s: Any) -> np.ndarray:
        velocity = np.asarray(velocity_m_s, dtype=np.float64)
        if not np.isfinite(velocity).all():
            raise ContractError("contact_solver_unstable", "$.velocity_m_s", "invalid rate")
        return self.damping_n_s_m * velocity**2


@dataclass(frozen=True)
class RigidContact:
    """Admitted reference: geometry is fixed and force is not inferred."""

    identifier: str = "rigid-contact-v1"

    def displacement(self, force_n: Any) -> np.ndarray:
        force = np.asarray(force_n, dtype=np.float64)
        if not np.isfinite(force).all():
            raise ContractError("contact_solver_unstable", "$.force_n", "non-finite force")
        return np.zeros_like(force)

    def force(self, indentation_m: Any, velocity_m_s: Any) -> np.ndarray:
        del velocity_m_s
        indentation = np.asarray(indentation_m, dtype=np.float64)
        if np.any(np.abs(indentation) > 1e-12):
            raise ContractError(
                "tissue_model_uncalibrated",
                "$.indentation_m",
                "rigid reference does not infer force from nonzero indentation",
            )
        return np.zeros_like(indentation)


def calibration_contract(
    *,
    instrument_id: str,
    calibration_trace_sha256: str,
    resolution_force_n: float,
    resolution_displacement_m: float,
    bias_force_n: float,
    bias_displacement_m: float,
    drift_force_n_per_hour: float,
    drift_displacement_m_per_hour: float,
    synchronization_uncertainty_s: float,
    temperature_reference_c: float,
    temperature_force_n_per_c: float,
    temperature_displacement_m_per_c: float,
    repeatability_force_n: float,
    repeatability_displacement_m: float,
    uncertainty_force_n: float,
    uncertainty_displacement_m: float,
    qualification_basis: str,
    qualified: bool,
    captured_utc: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    document = {
        "schema": CALIBRATION_SCHEMA,
        "content_sha256": "0" * 64,
        "instrument_id": instrument_id,
        "calibration_trace_sha256": calibration_trace_sha256,
        "resolution": {"force_n": resolution_force_n, "displacement_m": resolution_displacement_m},
        "bias": {"force_n": bias_force_n, "displacement_m": bias_displacement_m},
        "drift_per_hour": {
            "force_n": drift_force_n_per_hour,
            "displacement_m": drift_displacement_m_per_hour,
        },
        "synchronization_uncertainty_s": synchronization_uncertainty_s,
        "temperature_reference_c": temperature_reference_c,
        "temperature_sensitivity": {
            "force_n_per_c": temperature_force_n_per_c,
            "displacement_m_per_c": temperature_displacement_m_per_c,
        },
        "repeatability": {
            "force_n": repeatability_force_n,
            "displacement_m": repeatability_displacement_m,
        },
        "uncertainty": {"force_n": uncertainty_force_n, "displacement_m": uncertainty_displacement_m},
        "qualification_basis": qualification_basis,
        "qualified": qualified,
        "captured_utc": captured_utc,
        "provenance": deepcopy(provenance),
    }
    document["content_sha256"] = canonical_digest(document)
    return validate_contract(document, expected_schema=CALIBRATION_SCHEMA)


def load_instrument_csv(
    path: str | Path,
    calibration: dict[str, Any],
    *,
    batch_id: str,
    rate_m_s: float,
    angle_deg: float,
    repeat: int,
    lateral_drag: bool = False,
    require_qualified: bool = True,
) -> IndentationTrace:
    """Read and calibrate one strict, synchronized instrument trace."""

    source = Path(path)
    contract = validate_contract(calibration, expected_schema=CALIBRATION_SCHEMA)
    if require_qualified and not contract["qualified"]:
        raise ContractError(
            "force_sensor_unqualified",
            "$.calibration.qualified",
            f"{contract['instrument_id']} is {contract['qualification_basis']}",
        )
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRACE_COLUMNS:
            raise ContractError(
                "tissue_model_uncalibrated",
                "$.trace.columns",
                f"expected {','.join(TRACE_COLUMNS)}",
            )
        rows = list(reader)
    try:
        values = {name: np.asarray([float(row[name]) for row in rows], dtype=np.float64) for name in TRACE_COLUMNS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("tissue_model_uncalibrated", "$.trace", str(exc)) from exc
    if len(rows) < 4:
        raise ContractError("tissue_model_uncalibrated", "$.trace", "fewer than four samples")
    temperature_delta = values["temperature_c"] - float(contract["temperature_reference_c"])
    indentation = (
        values["measured_indentation_m"]
        - float(contract["bias"]["displacement_m"])
        - temperature_delta * float(contract["temperature_sensitivity"]["displacement_m_per_c"])
    )
    force = (
        values["measured_force_n"]
        - float(contract["bias"]["force_n"])
        - temperature_delta * float(contract["temperature_sensitivity"]["force_n_per_c"])
    )
    velocity = np.gradient(indentation, values["time_s"], edge_order=2)
    return IndentationTrace(
        values["time_s"],
        values["commanded_indentation_m"],
        indentation,
        velocity,
        force,
        values["temperature_c"],
        sha256_file(source),
        batch_id,
        float(rate_m_s),
        float(angle_deg),
        int(repeat),
        lateral_drag,
    )


def fit_kelvin_voigt(
    traces: Sequence[IndentationTrace],
    calibration: dict[str, Any],
    *,
    require_qualified: bool = True,
) -> tuple[KelvinVoigtWinkler, dict[str, Any]]:
    """Fit only identifiable normal stiffness/damping parameters."""

    contract = validate_contract(calibration, expected_schema=CALIBRATION_SCHEMA)
    if require_qualified and not contract["qualified"]:
        raise ContractError("force_sensor_unqualified", "$.calibration.qualified", "physical reference required")
    if not traces:
        raise ContractError("tissue_model_uncalibrated", "$.traces", "no traces")
    indentation = np.concatenate([trace.indentation_m for trace in traces if not trace.lateral_drag])
    velocity = np.concatenate([trace.velocity_m_s for trace in traces if not trace.lateral_drag])
    force = np.concatenate([trace.force_n for trace in traces if not trace.lateral_drag])
    design = np.stack([np.maximum(indentation, 0.0), velocity], axis=1)
    scale = np.maximum(np.linalg.norm(design, axis=0), 1e-20)
    normalized = design / scale
    singular = np.linalg.svd(normalized, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-20))
    if np.linalg.matrix_rank(normalized) < 2 or condition > 1e5:
        raise ContractError("tissue_model_uncalibrated", "$.traces", f"parameters not identifiable; condition={condition:.6g}")
    parameters, _, _, _ = np.linalg.lstsq(design, force, rcond=None)
    if parameters[0] <= 0 or parameters[1] < 0:
        raise ContractError("contact_solver_unstable", "$.parameters", f"nonphysical fit {parameters.tolist()}")
    residual = force - design @ parameters
    dof = max(1, len(force) - 2)
    variance = float(residual @ residual) / dof
    covariance = variance * np.linalg.pinv(design.T @ design, rcond=1e-12)
    domain = {
        "indentation_m": (float(indentation.min()), float(indentation.max())),
        "velocity_m_s": (float(velocity.min()), float(velocity.max())),
        "angle_deg": (
            float(min(trace.angle_deg for trace in traces)),
            float(max(trace.angle_deg for trace in traces)),
        ),
        "temperature_c": (
            float(min(trace.temperature_c.min() for trace in traces)),
            float(max(trace.temperature_c.max() for trace in traces)),
        ),
    }
    model = KelvinVoigtWinkler(float(parameters[0]), float(parameters[1]), covariance, domain)
    return model, {
        "samples": len(force),
        "traces": len(traces),
        "condition_number": condition,
        "residual_rmse_n": math.sqrt(float(np.mean(residual * residual))),
        "parameters_identified": True,
    }


def detect_out_of_distribution(
    model: KelvinVoigtWinkler,
    *,
    indentation_m: float,
    velocity_m_s: float,
    angle_deg: float,
    temperature_c: float,
) -> dict[str, Any]:
    values = {
        "indentation_m": float(indentation_m),
        "velocity_m_s": float(velocity_m_s),
        "angle_deg": float(angle_deg),
        "temperature_c": float(temperature_c),
    }
    outside = {
        name: value
        for name, value in values.items()
        if value < model.training_domain[name][0] or value > model.training_domain[name][1]
    }
    return {
        "status": "out_of_distribution" if outside else "in_distribution",
        "outside": outside,
        "policy": "refuse-compliant-use-and-fall-back-to-rigid-contact-v1",
    }


def parameter_uncertainty_sweep(
    model: KelvinVoigtWinkler,
    indentation_m: np.ndarray,
    velocity_m_s: np.ndarray,
    *,
    seed: int,
    samples: int = 256,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    parameters = rng.multivariate_normal(
        [model.stiffness_n_m, model.damping_n_s_m],
        model.covariance,
        size=samples,
        check_valid="raise",
    )
    valid = parameters[(parameters[:, 0] > 0) & (parameters[:, 1] >= 0)]
    if not len(valid):
        raise ContractError("contact_solver_unstable", "$.uncertainty", "no physical posterior samples")
    force = valid[:, 0, None] * indentation_m[None, :] + valid[:, 1, None] * velocity_m_s[None, :]
    return {
        "seed": seed,
        "requested_samples": samples,
        "physical_samples": len(valid),
        "force_n_p05": np.quantile(force, 0.05, axis=0).astype(float).tolist(),
        "force_n_p50": np.quantile(force, 0.5, axis=0).astype(float).tolist(),
        "force_n_p95": np.quantile(force, 0.95, axis=0).astype(float).tolist(),
    }


def hysteresis_area(indentation_m: np.ndarray, force_n: np.ndarray) -> float:
    return float(abs(np.trapezoid(np.asarray(force_n, dtype=np.float64), np.asarray(indentation_m, dtype=np.float64))))


def evaluate_model(
    model: KelvinVoigtWinkler,
    traces: Sequence[IndentationTrace],
    *,
    truth: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    force_errors = []
    indentation_errors = []
    area_errors = []
    ood = 0
    for trace in traces:
        predicted = model.force(trace.indentation_m, trace.velocity_m_s)
        force_errors.append(predicted - trace.force_n)
        inferred_indentation = np.maximum(
            0.0,
            (trace.force_n - model.damping_n_s_m * trace.velocity_m_s) / model.stiffness_n_m,
        )
        indentation_errors.append(inferred_indentation - trace.indentation_m)
        measured_area = hysteresis_area(trace.indentation_m, trace.force_n)
        predicted_area = hysteresis_area(trace.indentation_m, predicted)
        area_errors.append(abs(predicted_area - measured_area) / max(measured_area, 1e-12))
        midpoint = len(trace.time_s) // 2
        if detect_out_of_distribution(
            model,
            indentation_m=float(trace.indentation_m[midpoint]),
            velocity_m_s=float(trace.velocity_m_s[midpoint]),
            angle_deg=trace.angle_deg,
            temperature_c=float(trace.temperature_c[midpoint]),
        )["status"] != "in_distribution":
            ood += 1
    force_error = np.concatenate(force_errors)
    indentation_error = np.concatenate(indentation_errors)
    forces = np.concatenate([trace.force_n for trace in traces])
    force_range = max(float(forces.max() - forces.min()), 1e-12)
    stress_time = np.linspace(0.0, 2.0, 2001)
    stress_indent = 0.001 * (1.0 - np.cos(2 * np.pi * stress_time))
    stress_velocity = np.gradient(stress_indent, stress_time)
    stress_force = model.force(stress_indent, stress_velocity)
    energy = model.stored_energy_j(stress_indent)
    dissipation = model.dissipation_w(stress_velocity)
    metrics = {
        "heldout_traces": len(traces),
        "force_rmse_n": math.sqrt(float(np.mean(force_error * force_error))),
        "force_rmse_fraction_of_range": math.sqrt(float(np.mean(force_error * force_error))) / force_range,
        "force_bias_fraction_of_range": abs(float(np.mean(force_error))) / force_range,
        "indentation_rmse_m": math.sqrt(float(np.mean(indentation_error * indentation_error))),
        "hysteresis_area_relative_error": float(np.mean(area_errors)),
        "relaxation_time_s": model.relaxation_time_s,
        "finite_stable_energy_2x": bool(
            np.isfinite(stress_force).all()
            and np.isfinite(energy).all()
            and np.isfinite(dissipation).all()
            and np.all(energy >= 0)
            and np.all(dissipation >= 0)
        ),
        "heldout_ood_count": ood,
    }
    if truth:
        truth_tau = float(truth["damping_n_s_m"]) / float(truth["stiffness_n_m"])
        metrics["relaxation_time_relative_error"] = abs(model.relaxation_time_s - truth_tau) / truth_tau
    else:
        metrics["relaxation_time_relative_error"] = None
    return metrics


def candidate_checkpoints(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "force_rmse": metrics["force_rmse_fraction_of_range"] <= 0.10,
        "force_bias": metrics["force_bias_fraction_of_range"] <= 0.05,
        "indentation_rmse": metrics["indentation_rmse_m"] <= 0.00025,
        "hysteresis_area": metrics["hysteresis_area_relative_error"] <= 0.15,
        "relaxation_time": (
            metrics["relaxation_time_relative_error"] is not None
            and metrics["relaxation_time_relative_error"] <= 0.15
        ),
        "finite_stable_energy_2x": bool(metrics["finite_stable_energy_2x"]),
        "ood_detection": metrics["heldout_ood_count"] == 0,
    }


def make_tissue_patch(
    *,
    constitutive_model: str,
    stiffness_n_m: float,
    damping_n_s_m: float,
    identified: bool,
    calibration_sha256: str,
    batch_sha256: str,
    rest_surface_sha256: str,
    trace_sha256: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    posterior = hashlib.sha256(f"{stiffness_n_m:.17g}|{damping_n_s_m:.17g}".encode()).hexdigest()
    document = {
        "schema": "tatbot.tissue-patch/1",
        "content_sha256": "0" * 64,
        "material_chart_sha256": hashlib.sha256(b"synthetic-material-chart-v1").hexdigest(),
        "rest_surface_sha256": rest_surface_sha256,
        "deformed_surface_sha256": hashlib.sha256(b"synthetic-deformed-surface-v1").hexdigest(),
        "layer_prior_fields": [
            {
                "id": "generic-synthetic-thickness",
                "role": "generic-prior",
                "unit": "m",
                "values_sha256": hashlib.sha256(b"synthetic-thickness-values-v1").hexdigest(),
                "uncertainty_sha256": hashlib.sha256(b"synthetic-thickness-uncertainty-v1").hexdigest(),
            }
        ],
        "constitutive_model": constitutive_model,
        "parameters": {
            "normal_stiffness": {
                "value": stiffness_n_m,
                "sigma": 0.0 if not identified else abs(stiffness_n_m) * 0.05,
                "unit": "N/m",
                "identified": identified,
            }
        },
        "uncertainty": {
            "method": "synthetic-posterior-not-for-admission",
            "posterior_sha256": posterior,
            "out_of_distribution_policy_sha256": hashlib.sha256(b"rigid-fallback-ood-v1").hexdigest(),
            "confidence": 0.0 if not identified else 0.95,
        },
        "damping": {
            "value": damping_n_s_m,
            "sigma": 0.0 if not identified else abs(damping_n_s_m) * 0.05,
            "unit": "N*s/m",
            "identified": identified,
        },
        "relaxation": {
            "value": damping_n_s_m / stiffness_n_m if stiffness_n_m > 0 else 0.0,
            "sigma": 0.0,
            "unit": "s",
            "identified": identified,
        },
        "friction": {"value": 0.0, "sigma": 0.0, "unit": "1", "identified": False},
        "boundary_conditions": {
            "kind": "synthetic-fixed-support",
            "definition_sha256": hashlib.sha256(b"synthetic-fixed-support-v1").hexdigest(),
        },
        "support_sha256": hashlib.sha256(b"synthetic-rigid-phantom-support-v1").hexdigest(),
        "tool_geometry_sha256": hashlib.sha256(b"synthetic-indenter-v1").hexdigest(),
        "indentation": {
            "unit": "m",
            "commanded_trace_sha256": trace_sha256,
            "estimated_trace_sha256": trace_sha256,
            "measured_trace_sha256": None,
        },
        "force": {
            "unit": "N",
            "commanded_trace_sha256": None,
            "estimated_trace_sha256": trace_sha256,
            "measured_trace_sha256": None,
        },
        "calibration_sha256": calibration_sha256,
        "batch_sha256": batch_sha256,
        "provenance": deepcopy(provenance),
    }
    document["content_sha256"] = canonical_digest(document)
    return validate_contract(document, expected_schema="tatbot.tissue-patch/1")


def trace_digest(traces: Iterable[IndentationTrace]) -> str:
    digest = hashlib.sha256()
    for trace in traces:
        digest.update(trace.source_sha256.encode())
        digest.update(b"\0")
    return digest.hexdigest()
