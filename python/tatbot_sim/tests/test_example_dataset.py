"""Phase 4 exit gate: a fresh clone generates and audits a deterministic
example dataset — one tiny episode through the real generate -> write path,
then the real audit reads it back by its metadata (no assumed shapes).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.slow
def test_generate_and_audit_one_episode(tmp_path, monkeypatch, sim_profile):
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
    from wrist_cameras import describe

    cameras = tuple(camera.role for camera in describe(REPO))
    assert {name.removeprefix("observation.images.") for name in info["features"]
            if name.startswith("observation.images.")} == set(cameras) | {f"{c}_depth" for c in cameras}
    run_meta = json.loads((out / "meta" / "run_meta.json").read_text())
    assert run_meta["sensor_profile"]["name"] == "deployment"
    assert run_meta["schema_version"] == 2
    assert len(run_meta["software"]["revision_start"]) == 40
    assert run_meta["software"]["revision_end"] == run_meta["software"]["revision_start"]
    assert run_meta["config"]["seed"] == 7
    from tatbot_sim.tools import registry, workspace
    current_workspace = workspace()
    fitted_tool = registry().active_tool_id(REPO, workspace=current_workspace)
    measured_tip = (registry().tip_offset_m(current_workspace)
                    if fitted_tool == run_meta["tool"]["tool_id"] else None)
    if sim_profile == "public" or measured_tip is None:
        assert run_meta["tool"]["geometry_status"] == "nominal"
        assert run_meta["tool"]["contact_geometry_status"] == "unqualified"
        assert run_meta["tool"]["geometry_basis"] == "nominal-datasheet"
        assert run_meta["tool"]["geometry_warnings"]
    else:
        assert run_meta["tool"]["geometry_status"] == "contact-qualified"
        assert run_meta["tool"]["contact_geometry_status"] == "pivot-calibrated"
        assert run_meta["tool"]["body_pose_status"] == "axis-inferred"
        assert run_meta["tool"]["geometry_basis"] == "measured-pivot"
        assert run_meta["tool"]["qualification"] == "qualified"
        assert run_meta["tool"]["geometry_warnings"] == []

    # The real audit tool, driven by the dataset's own metadata.
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "sim_dataset_audit.py"),
         "--path", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
