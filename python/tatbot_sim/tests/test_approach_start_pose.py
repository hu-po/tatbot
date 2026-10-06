"""An episode that opens at the staged pose starts from a pose the whole
articulation can take.

The staged pose is the six arm joints (planning.STAGED_POSE). Since the tool
carriage became the seventh IK axis, the episode's opening pose has to seed
the carriage the way the expert's own approach segment does; handing the six
values to seven joints refused every batch that drew an approach, which on a
one-env run is the first batch a quarter of the time.

CPU backend, one tiny maze episode, the approach forced on: the same path
test_example_dataset takes, with the descent it leaves to chance.
"""

from __future__ import annotations

import dataclasses
import json

import pytest


@pytest.mark.slow
def test_an_approach_episode_opens_at_the_seeded_staged_pose(tmp_path, monkeypatch):
    from tatbot_sim import generate
    from tatbot_sim.distributions import DISTRIBUTIONS

    monkeypatch.setattr(generate, "source_state", lambda: {
        "repository": "example/tatbot", "revision": "a" * 40, "dirty": False,
    })
    out = tmp_path / "approach-ds"
    args = dataclasses.replace(
        DISTRIBUTIONS["paper-draw"].build_args(),
        out_dir=str(out),
        num_episodes=1,
        num_envs=1,
        horizon=120,
        seed=7,
        task="maze",
        tool_calibration_jitter=False,
        sim_backend="cpu",
    )
    args.dr.approach.prob = 1.0
    generate.main(args)

    run_meta = json.loads((out / "meta" / "run_meta.json").read_text())
    [episode] = run_meta["episodes"]
    assert episode["approach_frames"] > 0
