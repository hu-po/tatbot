from __future__ import annotations

import csv
import hashlib

import numpy as np
import pytest
from tatbot_sim.human_rep.contracts import ContractError, validate_contract
from tatbot_sim.human_rep.mechanics import (
    IndentationTrace,
    KelvinVoigtWinkler,
    RigidContact,
    calibration_contract,
    candidate_checkpoints,
    detect_out_of_distribution,
    evaluate_model,
    fit_kelvin_voigt,
    load_instrument_csv,
    make_tissue_patch,
    parameter_uncertainty_sweep,
    trace_digest,
)

WHEN = "2026-09-04T16:00:00Z"


def _calibration(*, qualified=False):
    return calibration_contract(
        instrument_id="synthetic-force-displacement-reference",
        calibration_trace_sha256="a" * 64,
        resolution_force_n=0.001,
        resolution_displacement_m=1e-6,
        bias_force_n=0.02,
        bias_displacement_m=2e-5,
        drift_force_n_per_hour=0.001,
        drift_displacement_m_per_hour=1e-6,
        synchronization_uncertainty_s=0.0001,
        temperature_reference_c=23.0,
        temperature_force_n_per_c=0.001,
        temperature_displacement_m_per_c=1e-6,
        repeatability_force_n=0.002,
        repeatability_displacement_m=2e-6,
        uncertainty_force_n=0.003,
        uncertainty_displacement_m=3e-6,
        qualification_basis="physical-reference" if qualified else "synthetic-fixture-only",
        qualified=qualified,
        captured_utc=WHEN,
        provenance={
            "producer": "p7-test",
            "version": "1",
            "created_utc": WHEN,
            "source_sha256": "b" * 64,
            "seed": 17,
        },
    )


def _trace(rate=0.001, *, k=800.0, c=4.0, batch="batch-a", repeat=0, noise=0.0):
    duration = 4.0
    time = np.linspace(0.0, duration, 801)
    phase = time / duration
    indentation = np.where(phase <= 0.5, phase * 0.004, (1.0 - phase) * 0.004)
    # Scale time to obtain independently varied loading rates.
    indentation *= rate / 0.001
    velocity = np.gradient(indentation, time, edge_order=2)
    rng = np.random.default_rng(repeat + int(rate * 1e7))
    force = k * indentation + c * velocity + rng.normal(0.0, noise, len(time))
    raw = hashlib.sha256(np.ascontiguousarray(np.stack([time, indentation, force])).tobytes()).hexdigest()
    return IndentationTrace(
        time,
        indentation,
        indentation,
        velocity,
        force,
        np.full_like(time, 23.0),
        raw,
        batch,
        rate,
        0.0,
        repeat,
    )


def test_kelvin_voigt_equation_energy_and_rigid_reference():
    model = KelvinVoigtWinkler(800.0, 4.0, np.eye(2) * 1e-6, {
        "indentation_m": (0.0, 0.003),
        "velocity_m_s": (-0.01, 0.01),
        "angle_deg": (0.0, 30.0),
        "temperature_c": (20.0, 25.0),
    })
    indentation = np.asarray([0.0, 0.001, 0.002])
    velocity = np.asarray([0.0, 0.001, -0.001])
    assert np.allclose(model.force(indentation, velocity), 800 * indentation + 4 * velocity)
    assert np.all(model.stored_energy_j(indentation) >= 0)
    assert np.all(model.dissipation_w(velocity) >= 0)
    rigid = RigidContact()
    assert np.array_equal(rigid.displacement([0, 1]), [0, 0])
    with pytest.raises(ContractError) as caught:
        rigid.force([0.001], [0])
    assert caught.value.code == "tissue_model_uncalibrated"


def test_fit_recovers_identifiable_parameters_and_uncertainty():
    traces = [_trace(rate, repeat=repeat, noise=0.0002) for rate in (0.0005, 0.001, 0.002) for repeat in range(3)]
    model, fit = fit_kelvin_voigt(traces, _calibration(), require_qualified=False)
    assert fit["parameters_identified"] is True
    assert model.stiffness_n_m == pytest.approx(800.0, rel=1e-3)
    assert model.damping_n_s_m == pytest.approx(4.0, rel=0.02)
    metrics = evaluate_model(model, traces, truth={"stiffness_n_m": 800.0, "damping_n_s_m": 4.0})
    checks = candidate_checkpoints(metrics)
    assert all(checks.values())
    sweep = parameter_uncertainty_sweep(
        model,
        np.asarray([0.0, 0.001]),
        np.asarray([0.0, 0.001]),
        seed=17,
        samples=64,
    )
    assert sweep["physical_samples"] > 0
    assert len(sweep["force_n_p50"]) == 2


def test_unqualified_instrument_refuses_physical_fit():
    with pytest.raises(ContractError) as caught:
        fit_kelvin_voigt([_trace()], _calibration(), require_qualified=True)
    assert caught.value.code == "force_sensor_unqualified"


def test_strict_instrument_adapter_applies_bias_and_temperature(tmp_path):
    path = tmp_path / "trace.csv"
    rows = [
        [index * 0.01, index * 1e-5, index * 1e-5 + 2e-5 + 2e-6, index * 0.01 + 0.02 + 0.002, 25.0]
        for index in range(8)
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "time_s",
            "commanded_indentation_m",
            "measured_indentation_m",
            "measured_force_n",
            "temperature_c",
        ))
        writer.writerows(rows)
    trace = load_instrument_csv(
        path,
        _calibration(),
        batch_id="batch-a",
        rate_m_s=0.001,
        angle_deg=0,
        repeat=0,
        require_qualified=False,
    )
    assert trace.indentation_m[0] == pytest.approx(0.0)
    assert trace.force_n[0] == pytest.approx(0.0)
    assert trace.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_ood_and_tissue_contract_keep_synthetic_candidate_unadmitted():
    traces = [_trace(0.001), _trace(0.002)]
    model, _ = fit_kelvin_voigt(traces, _calibration(), require_qualified=False)
    assert detect_out_of_distribution(
        model,
        indentation_m=0.05,
        velocity_m_s=0.0,
        angle_deg=0.0,
        temperature_c=23.0,
    )["status"] == "out_of_distribution"
    trace_sha = trace_digest(traces)
    document = make_tissue_patch(
        constitutive_model="rigid-contact-v1",
        stiffness_n_m=0.0,
        damping_n_s_m=0.0,
        identified=False,
        calibration_sha256=_calibration()["content_sha256"],
        batch_sha256="c" * 64,
        rest_surface_sha256="d" * 64,
        trace_sha256=trace_sha,
        provenance={
            "producer": "p7-test",
            "version": "1",
            "created_utc": WHEN,
            "source_sha256": trace_sha,
        },
    )
    assert document["constitutive_model"] == "rigid-contact-v1"
    assert document["parameters"]["normal_stiffness"]["identified"] is False
    validate_contract(document, expected_schema="tatbot.tissue-patch/1")


def test_synthetic_basis_cannot_claim_qualified():
    with pytest.raises(ContractError) as caught:
        calibration_contract(
            instrument_id="bad",
            calibration_trace_sha256="a" * 64,
            resolution_force_n=0,
            resolution_displacement_m=0,
            bias_force_n=0,
            bias_displacement_m=0,
            drift_force_n_per_hour=0,
            drift_displacement_m_per_hour=0,
            synchronization_uncertainty_s=0,
            temperature_reference_c=23,
            temperature_force_n_per_c=0,
            temperature_displacement_m_per_c=0,
            repeatability_force_n=0,
            repeatability_displacement_m=0,
            uncertainty_force_n=0,
            uncertainty_displacement_m=0,
            qualification_basis="synthetic-fixture-only",
            qualified=True,
            captured_utc=WHEN,
            provenance={"producer": "test", "version": "1", "created_utc": WHEN, "source_sha256": "b" * 64},
        )
    assert caught.value.code == "force_sensor_unqualified"
