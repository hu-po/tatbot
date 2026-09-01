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


def test_groot_rgbd_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "groot",
                "use_relative_actions": True,
                "input_features": {
                    "observation.state": {"shape": [7]},
                    "observation.images.wrist_upper": {"shape": [3, 480, 640]},
                    "observation.images.wrist_upper_depth": {"shape": [3, 480, 640]},
                },
            }
        )
    )

    assert MODULE.load_contract(str(checkpoint)) == {
        "policy_type": "groot",
        "use_relative_actions": True,
        "use_depth": True,
        "depth_encoding": "depth-v1",
        "state_size": 7,
        "use_external_effort": False,
        "mask_external_effort": False,
    }


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


def test_declare_stamps_every_checkpoint_and_reads_back(tmp_path: Path) -> None:
    run = tmp_path / "outputs" / "run"
    for step in ("010000", "020000"):
        _act_rgbd_checkpoint(run / step)
    MODULE.declare(run, mask_external_effort=True)
    for step in ("010000", "020000"):
        contract = MODULE.load_contract(str(run / step / "checkpoint"))
        assert contract["mask_external_effort"] is True


def test_sync_launcher_refuses_masked_effort_checkpoint(tmp_path: Path) -> None:
    checkpoint = _act_rgbd_checkpoint(tmp_path, {"mask_external_effort": True})
    result = subprocess.run(
        [str(REPO / "scripts" / "il_rollout.sh"), str(checkpoint), "1"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "masked to zero" in result.stderr


def test_sync_launcher_rejects_relative_groot_before_arm_gate(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "groot",
                "use_relative_actions": True,
                "input_features": {"observation.state": {"shape": [7]}},
            }
        )
    )
    result = subprocess.run(
        [str(REPO / "scripts" / "il_rollout.sh"), str(tmp_path), "1"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "full-chunk async inference" in result.stderr


def test_sync_launcher_has_no_legacy_default() -> None:
    result = subprocess.run(
        [str(REPO / "scripts" / "il_rollout.sh")],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "GR00T flagship" in result.stderr
