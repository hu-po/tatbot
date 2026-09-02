"""Phase 4 exit gate: a fresh clone generates and audits a deterministic
example dataset — one tiny episode through the real generate -> write path,
then the real audit reads it back by its metadata (no assumed shapes).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_generate_and_audit_one_episode(tmp_path, monkeypatch):
    out = tmp_path / "example-ds"
    import dataclasses

    from tatbot_sim import generate
    from tatbot_sim.distributions import DISTRIBUTIONS

    clean_source = {
        "repository": "example/tatbot",
        "revision": "a" * 40,
        "dirty": False,
    }
    monkeypatch.setattr(generate, "source_state", lambda: clean_source)

    # A NAMED recipe, shrunk to one tiny episode: the audit refuses datasets
    # assembled from bare flags, and this test wants that behavior kept.
    args = dataclasses.replace(
        DISTRIBUTIONS["paper-draw"].build_args(),
        out_dir=str(out),
        num_episodes=1,
        num_envs=1,
        horizon=120,
        seed=7,
        task="maze",
        # This test calls the engine directly. The named factory owns the
        # pre-import, per-shard calibration draw exercised in factory tests.
        tool_calibration_jitter=False,
        sim_backend="cpu",
    )
    generate.main(args)

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    run_meta = json.loads((out / "meta" / "run_meta.json").read_text())
    assert run_meta["schema_version"] == 2
    assert len(run_meta["software"]["revision_start"]) == 40
    assert run_meta["software"]["revision_end"] == run_meta["software"]["revision_start"]
    assert run_meta["config"]["seed"] == 7
    assert run_meta["tool"]["geometry_status"] == "contact-qualified"
    assert run_meta["tool"]["contact_geometry_status"] == "pivot-calibrated"
    assert run_meta["tool"]["body_pose_status"] == "axis-inferred"
    assert run_meta["tool"]["provisional_geometry_override"] is False

    # The real audit tool, driven by the dataset's own metadata.
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "sim_dataset_audit.py"),
         "--path", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
