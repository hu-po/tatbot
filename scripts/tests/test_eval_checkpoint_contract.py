from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "scripts" / "eval" / "checkpoint_contract.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_contract", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _act_rgbd_checkpoint(tmp_path: Path, sidecar: dict | None = None) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {
                    "observation.state": {"shape": [14]},
                    "observation.images.wrist_upper": {"shape": [3, 480, 640]},
                    "observation.images.wrist_upper_depth": {"shape": [1, 480, 640]},
                },
            }
        )
    )
    if sidecar is not None:
        (checkpoint / MODULE.SIDECAR).write_text(json.dumps(sidecar))
    return checkpoint


def test_effort_is_live_without_a_sidecar(tmp_path: Path) -> None:
    contract = MODULE.load_contract(str(_act_rgbd_checkpoint(tmp_path)))
    assert contract["use_external_effort"] is True
    assert contract["mask_external_effort"] is False


def test_sidecar_declares_masked_effort(tmp_path: Path) -> None:
    checkpoint = _act_rgbd_checkpoint(tmp_path, {"mask_external_effort": True})
    contract = MODULE.load_contract(str(checkpoint))
    # The channels stay on the wire — the width is unchanged — but they carry
    # zeros, so a launcher that sends measured effort must refuse this policy.
    assert contract["state_size"] == 14
    assert contract["use_external_effort"] is True
    assert contract["mask_external_effort"] is True
    assert MODULE.fields(contract).split("|")[-1] == "1"


def test_stdin_config_accepts_explicit_sidecar_json(tmp_path: Path) -> None:
    checkpoint = _act_rgbd_checkpoint(tmp_path, {"mask_external_effort": True})
    result = subprocess.run(
        [
            str(MODULE_PATH),
            "--format",
            "fields",
            "--sidecar-json",
            (checkpoint / MODULE.SIDECAR).read_text(),
            "-",
        ],
        input=(checkpoint / "config.json").read_text(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().split("|")[-1] == "1"


def test_masked_sidecar_rejects_non_14_wide_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {"observation.state": {"shape": [7]}},
            }
        )
    )
    (checkpoint / MODULE.SIDECAR).write_text(json.dumps({"mask_external_effort": True}))

    try:
        MODULE.load_contract(str(checkpoint))
    except ValueError as error:
        assert "shape [14]" in str(error)
    else:
        raise AssertionError("invalid masked checkpoint was accepted")


def test_declare_stamps_every_checkpoint_and_reads_back(tmp_path: Path) -> None:
    run = tmp_path / "outputs" / "run"
    for step in ("010000", "020000"):
        _act_rgbd_checkpoint(run / step)
    MODULE.declare(run, mask_external_effort=True)
    for step in ("010000", "020000"):
        contract = MODULE.load_contract(str(run / step / "checkpoint"))
        assert contract["mask_external_effort"] is True


def test_rollout_launcher_passes_contract_mask_to_the_robot_client() -> None:
    # il_rollout_async.sh is the one rollout launcher (the synchronous
    # il_rollout.sh and il_compare_policies.sh were retired 2026-09-02: the
    # first could not run the flagship, the second could not pass arm_gate
    # past its first launch).
    async_ = (REPO / "scripts" / "il_rollout_async.sh").read_text()

    assert "--robot.mask_external_effort=$MASK_EXT_EFF_BOOL" in async_
    assert "launcher-controlled rollout option cannot be overridden" in async_
    assert "Teach the client to zero" not in async_
    assert not (REPO / "scripts" / "il_rollout.sh").exists()
    assert not (REPO / "scripts" / "il_compare_policies.sh").exists()
