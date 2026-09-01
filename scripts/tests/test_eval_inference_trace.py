from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from inference_trace import write_inference_trace  # noqa: E402


def test_trace_preserves_observation_normalized_and_decoded(tmp_path: Path) -> None:
    result = write_inference_trace(
        tmp_path,
        timestep=7,
        observation={
            "wrist_upper": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
            "joint_0.pos": 0.25,
            "task": "laser off",
        },
        observation_state=np.arange(7, dtype=np.float32),
        normalized_action=np.zeros((1, 16, 7), dtype=np.float32),
        decoded_action=np.ones((16, 7), dtype=np.float32),
        fixed_noise_seed="1700",
    )

    npz = Path(result["npz"])
    metadata = json.loads(Path(result["metadata"]).read_text())
    loaded = np.load(npz, allow_pickle=False)

    assert hashlib.sha256(npz.read_bytes()).hexdigest() == result["npz_sha256"]
    assert loaded["observation_state"].shape == (7,)
    assert loaded["normalized_action"].shape == (1, 16, 7)
    assert loaded["decoded_action"].shape == (16, 7)
    assert np.array_equal(loaded["observation__wrist_upper"], np.arange(18).reshape(2, 3, 3))
    assert metadata["observation_metadata"]["task"] == "laser off"
    assert metadata["fixed_noise_seed"] == "1700"
