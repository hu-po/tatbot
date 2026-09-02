from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from chunk_guard import evaluate_chunk, execution_metrics  # noqa: E402
from trajectory_plausibility import (  # noqa: E402
    _wait_for_actions,
    action_decode_contract,
    build_contract,
    evaluate_predictions,
    validate_postprocessor_binding,
    wire_features,
)


def test_fixture_metadata_does_not_claim_held_out_source() -> None:
    source = (Path(__file__).resolve().parents[1] / "eval" / "trajectory_plausibility.py").read_text()
    assert '"kind": "genuine no-arm observation fixture"' in source
    assert "genuine held-out no-arm observation fixture" not in source


def write_postprocessor(path: Path, horizon: int, joints: int) -> None:
    bounds = np.ones((horizon, joints), dtype=float)
    path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "groot_n1_7_action_decode_v1",
                        "config": {
                            "raw_stats": {
                                "relative_action": {
                                    "single_arm": {
                                        "min": (-bounds).tolist(),
                                        "max": bounds.tolist(),
                                    }
                                }
                            }
                        },
                    }
                ]
            }
        )
    )


def write_mixed_postprocessor(path: Path, horizon: int) -> None:
    arm = np.ones((horizon, 6), dtype=float)
    path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "groot_n1_7_action_decode_v1",
                        "config": {
                            "raw_stats": {
                                "relative_action": {
                                    "single_arm": {
                                        "min": (-arm).tolist(),
                                        "max": arm.tolist(),
                                    }
                                },
                                "action": {
                                    "single_arm": {
                                        "min": np.full(6, -9.0).tolist(),
                                        "max": np.full(6, 9.0).tolist(),
                                    },
                                    "left_carriage_joint": {
                                        "min": [0.01],
                                        "max": [0.02],
                                    },
                                },
                            }
                        },
                    }
                ]
            }
        )
    )


def write_standard_postprocessor(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "policy_postprocessor",
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "config": {"features": {"action": {"shape": [7]}},},
                        "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                    }
                ],
            }
        )
    )
    (path.parent / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(
        b"standard-action-stats"
    )


def demonstrations() -> dict[str, np.ndarray]:
    action = np.zeros((4, 4, 7), dtype=np.float64)
    for sample in range(4):
        action[sample, :, :6] = np.arange(4)[:, None] * 0.01 + sample * 0.001
    return {
        "state": action[:, 0].copy(),
        "action": action,
        "action_is_pad": np.zeros((4, 4), dtype=bool),
        "episode_index": np.zeros(4, dtype=np.int64),
        "frame_index": np.arange(4, dtype=np.int64),
    }


def test_genuine_demonstration_chunks_pass_their_rejection_envelope(tmp_path: Path) -> None:
    postprocessor = tmp_path / "policy_postprocessor.json"
    write_postprocessor(postprocessor, 4, 7)
    demo = demonstrations()
    contract = build_contract(demo, postprocessor, [tmp_path / "genuine-demo"])
    chunks = np.repeat(demo["action"][:, None], 3, axis=1)

    metrics, failures, warnings = evaluate_predictions(chunks, demo, contract, postprocessor)

    assert failures == []
    assert warnings == []
    assert np.allclose(metrics["repeated_first_std_rad_per_joint"], 0.0)
    assert contract["safety_warning"].startswith("rejection-only")
    assert contract["schema_version"] == 2
    assert "l1_to_demo_rad_per_joint" not in contract["rejection_thresholds"]
    assert "l1_to_demo_rad_per_joint" in contract["report_only_metrics"]


def test_oscillation_and_repeated_input_variance_are_rejected(tmp_path: Path) -> None:
    postprocessor = tmp_path / "policy_postprocessor.json"
    write_postprocessor(postprocessor, 4, 7)
    demo = demonstrations()
    contract = build_contract(demo, postprocessor, [tmp_path / "genuine-demo"])
    chunks = np.repeat(demo["action"][:, None], 3, axis=1)
    chunks[:, 0, :, 0] = [0.0, 0.8, -0.8, 0.8]
    chunks[:, 1, :, 0] = [0.8, -0.8, 0.8, -0.8]

    _, failures, _ = evaluate_predictions(chunks, demo, contract, postprocessor)

    assert any("adjacent_step_abs_rad_per_joint[0]" in failure for failure in failures)
    assert any("repeated_first_std_rad_per_joint[0]" in failure for failure in failures)


def test_wire_schema_has_no_robot_or_driver_object() -> None:
    rgbd = wire_features("groot_rgbd")
    act = wire_features("act_rgb")
    masked = wire_features("act_rgbd14_masked")

    assert rgbd["observation.state"]["names"][-1] == "left_carriage_joint.pos"
    assert rgbd["observation.images.wrist_upper_depth"]["shape"] == (480, 640, 3)
    assert "observation.images.wrist_upper_depth" not in act
    assert masked["observation.state"]["shape"] == (14,)
    assert masked["observation.state"]["names"][7:] == [
        name.replace(".pos", ".ext_eff") for name in masked["observation.state"]["names"][:7]
    ]
    assert masked["observation.images.wrist_upper_depth"]["shape"] == (480, 640, 1)
    assert masked["observation.images.wrist_upper_depth"]["info"]["is_depth_map"] is True
    assert all("ip_address" not in feature for feature in rgbd.values())


def test_standard_policy_contract_accepts_14_wide_state(tmp_path: Path) -> None:
    postprocessor = tmp_path / "policy_postprocessor.json"
    write_standard_postprocessor(postprocessor)
    demo = demonstrations()
    demo["state"] = np.concatenate([demo["state"], np.zeros_like(demo["state"])], axis=1)

    contract = build_contract(demo, postprocessor, [tmp_path / "genuine-demo"])
    chunks = np.repeat(demo["action"][:, None], 3, axis=1)
    metrics, failures, warnings = evaluate_predictions(chunks, demo, contract, postprocessor)

    assert failures == []
    assert warnings == []
    assert contract["state_width"] == 14
    assert np.allclose(metrics["repeated_first_std_rad_per_joint"], 0.0)

    live_metrics, live_failures, live_warnings = evaluate_chunk(
        None, demo["action"][0], demo["state"][0], contract
    )
    assert live_failures == []
    assert live_warnings == []
    assert len(live_metrics["first_target_distance_abs_rad_per_joint"]) == 7


def test_corrected_contract_decodes_arm_relative_and_carriage_absolute(tmp_path: Path) -> None:
    postprocessor = tmp_path / "policy_postprocessor.json"
    write_mixed_postprocessor(postprocessor, 4)
    low, high, relative = action_decode_contract(postprocessor, 4, 7)

    assert low.shape == high.shape == (4, 7)
    assert relative.tolist() == [True, True, True, True, True, True, False]
    assert np.allclose(low[:, -1], 0.01)
    assert np.allclose(high[:, -1], 0.02)

    demo = demonstrations()
    demo["state"][:, -1] = 0.015
    demo["action"][:, :, -1] = 0.015
    contract = build_contract(demo, postprocessor, [tmp_path / "genuine-demo"])
    assert contract["action_semantics"] == {
        "normalization": "groot_relative_minmax",
        "relative_joint_indices": [0, 1, 2, 3, 4, 5],
        "absolute_joint_indices": [6],
    }
    assert contract["reference"]["normalized_endpoint_fraction_per_joint"][-1] == 0.0


def test_standard_absolute_policy_uses_decoded_action_contract(tmp_path: Path) -> None:
    postprocessor = tmp_path / "policy_postprocessor.json"
    write_standard_postprocessor(postprocessor)
    demo = demonstrations()
    contract = build_contract(demo, postprocessor, [tmp_path / "genuine-demo"])
    chunks = np.repeat(demo["action"][:, None], 3, axis=1)

    metrics, failures, warnings = evaluate_predictions(chunks, demo, contract, postprocessor)

    assert failures == []
    assert warnings == []
    assert contract["action_semantics"] == {
        "normalization": "standard_absolute",
        "relative_joint_indices": [],
        "absolute_joint_indices": list(range(7)),
    }
    assert set(contract["postprocessor_artifacts_sha256"]) == {
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    assert "normalized_adjacent_step_abs_per_joint" not in contract["rejection_thresholds"]
    assert "normalized_endpoint_fraction_overall" not in metrics

    live_metrics, live_failures, live_warnings = evaluate_chunk(
        None, demo["action"][0], demo["state"][0], contract
    )
    assert live_failures == []
    assert live_warnings == []
    assert "normalized_endpoint_fraction_overall" not in live_metrics

    unsafe = chunks.copy()
    unsafe[:, :, :, 0] = [0.0, 0.8, -0.8, 0.8]
    _, failures, _ = evaluate_predictions(unsafe, demo, contract, postprocessor)
    assert any("adjacent_step_abs_rad_per_joint[0]" in failure for failure in failures)


def test_probe_contract_rejects_mismatched_external_processor_state(tmp_path: Path) -> None:
    postprocessor = tmp_path / "policy_postprocessor.json"
    write_standard_postprocessor(postprocessor)
    contract = build_contract(demonstrations(), postprocessor, [tmp_path / "genuine-demo"])

    assert validate_postprocessor_binding(contract, postprocessor) == contract[
        "postprocessor_artifacts_sha256"
    ]
    (tmp_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(
        b"different-action-stats"
    )

    with pytest.raises(ValueError, match="unnormalizer_processor.safetensors"):
        validate_postprocessor_binding(contract, postprocessor)


def test_live_chunk_guard_passes_demo_and_rejects_oscillation(tmp_path: Path) -> None:
    postprocessor = tmp_path / "policy_postprocessor.json"
    write_postprocessor(postprocessor, 4, 7)
    demo = demonstrations()
    contract = build_contract(demo, postprocessor, [tmp_path / "genuine-demo"])
    decoded = demo["action"][0]
    state = demo["state"][0]
    normalized = decoded - state

    metrics, failures, warnings = evaluate_chunk(normalized, decoded, state, contract)
    assert failures == []
    assert warnings == []
    assert metrics["normalized_endpoint_fraction_overall"] == 0.0

    unsafe = decoded.copy()
    unsafe[:, 0] = [0.0, 0.8, -0.8, 0.8]
    _, failures, _ = evaluate_chunk(unsafe - state, unsafe, state, contract)
    assert any("adjacent_step_abs_rad_per_joint[0]" in failure for failure in failures)


def test_execution_model_replays_filter_and_slew_after_aggregation() -> None:
    model = {
        "fps": 30.0,
        "target_filter_tau_s": 0.3,
        "max_joint_velocity_rad_s": 0.25,
        "actions_per_chunk": 60,
    }
    queued = np.zeros((100, 7))
    incoming = np.full((100, 7), 0.31118 / 0.7)
    decoded = 0.3 * queued + 0.7 * incoming
    state = np.zeros(7)

    metrics = execution_metrics(decoded, state, model)

    assert max(metrics["executed_step_abs_rad_per_joint"]) <= 0.25 / 30.0 + 1e-12
    assert metrics["executed_displacement_abs_rad_per_joint"][0] < 0.31118
    assert metrics["execution_slew_saturation_fraction_per_joint"][0] > 0.5


def test_exact_demo_l1_is_report_only(tmp_path: Path) -> None:
    postprocessor = tmp_path / "policy_postprocessor.json"
    write_standard_postprocessor(postprocessor)
    demo = demonstrations()
    contract = build_contract(demo, postprocessor, [tmp_path / "genuine-demo"])
    chunks = np.repeat(demo["action"][:, None], 3, axis=1)
    mismatched_truth = dict(demo)
    mismatched_truth["action"] = demo["action"].copy()
    mismatched_truth["action"][:, :, :6] += 1.0

    metrics, failures, warnings = evaluate_predictions(
        chunks, mismatched_truth, contract, postprocessor
    )

    assert np.allclose(metrics["l1_to_demo_rad_per_joint"][:6], 1.0)
    assert failures == []
    assert warnings == []


def test_retrospective_margin_routes_close_act_to_review_and_corrupt_act_to_reject(
    tmp_path: Path,
) -> None:
    postprocessor = tmp_path / "policy_postprocessor.json"
    write_standard_postprocessor(postprocessor)
    demo = demonstrations()
    contract = build_contract(demo, postprocessor, [tmp_path / "genuine-demo"])
    quality_limit = [1.0] * 7
    quality_limit[2] = 0.04578
    hard_limit = [1.0] * 7
    hard_limit[2] = 0.09156
    contract["quality_thresholds"] = {
        "first_target_distance_abs_rad_per_joint": quality_limit,
    }
    contract["rejection_thresholds"] = {
        "first_target_distance_abs_rad_per_joint": hard_limit,
    }
    state = np.zeros(7)
    close = np.zeros((4, 7))
    close[:, 2] = 0.05812

    _, failures, warnings = evaluate_chunk(None, close, state, contract)

    assert failures == []
    assert any("first_target_distance_abs_rad_per_joint[2]" in item for item in warnings)

    corrupt = close.copy()
    corrupt[:, 2] = 0.17182
    _, failures, warnings = evaluate_chunk(None, corrupt, state, contract)

    assert any("first_target_distance_abs_rad_per_joint[2]" in item for item in failures)
    assert any("first_target_distance_abs_rad_per_joint[2]" in item for item in warnings)


def test_wait_for_actions_polls_until_data_or_timeout() -> None:
    class DummyResponse:
        def __init__(self, data: bytes) -> None:
            self.data = data

    class DummyStub:
        def __init__(self, responses: list[bytes]) -> None:
            self.responses = responses
            self.calls = 0

        def GetActions(self, empty_arg: None) -> DummyResponse:  # noqa: N802
            self.calls += 1
            if self.responses:
                return DummyResponse(self.responses.pop(0))
            return DummyResponse(b"")

    payload = ["action1", "action2"]
    encoded = pickle.dumps(payload)

    # Polls until non-empty data is available
    stub_success = DummyStub([b"", b"", encoded])
    result = _wait_for_actions(stub_success, lambda: None, timeout_s=1.0)
    assert result == payload
    assert stub_success.calls == 3

    # Times out if data remains empty
    stub_timeout = DummyStub([b"", b""])
    with pytest.raises(TimeoutError, match="no action chunk within"):
        _wait_for_actions(stub_timeout, lambda: None, timeout_s=0.05)
