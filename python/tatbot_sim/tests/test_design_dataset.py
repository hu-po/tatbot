"""`--design FILE` end to end: one tiny plane design through generate, on the
CPU backend, into a dataset whose metadata names the design.

Slow, needs a render device and Node (the design is built through the
browser reader, as `tatbot design place` builds one).
"""

from __future__ import annotations

import dataclasses
import json

import pytest

SQUARE = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
          '<path d="M3 1 H9 V9 H1 V4 Z"/></svg>')


@pytest.mark.slow
def test_generate_draws_a_portable_design_and_records_it(tmp_path, monkeypatch):
    from tatbot_sim import generate
    from tatbot_sim.distributions import DISTRIBUTIONS
    from tatbot_sim.inkmap.design_build import artwork_from_svg, design_from_artwork, file_source, write_json

    record = artwork_from_svg(SQUARE, name="Notched square", size_mm=(6.0, 6.0),
                              source=file_source(SQUARE, path="notched.svg"))
    design = design_from_artwork(record, kind="plane", canvas_m=(0.1905, 0.2794), anchor_uv_m=(0.01, -0.01))
    design_path = write_json(tmp_path / "notched.json", design)

    monkeypatch.setattr(generate, "source_state", lambda: {
        "repository": "example/tatbot", "revision": "a" * 40, "dirty": False,
    })
    out = tmp_path / "design-ds"
    args = dataclasses.replace(
        DISTRIBUTIONS["paper-draw"].build_args(),
        out_dir=str(out), num_episodes=1, num_envs=1, horizon=2400, seed=7,
        design=str(design_path), tool_calibration_jitter=False, sim_backend="cpu",
    )
    generate.main(args)

    run_meta = json.loads((out / "meta" / "run_meta.json").read_text())
    assert run_meta["design"]["design_sha256"] == design["content_sha256"]
    assert run_meta["design"]["placement"] == "authored"
    assert run_meta["config"]["design"] == str(design_path)
    [episode] = run_meta["episodes"]
    assert episode["kind"] == "artwork"
    assert episode["artwork"] == {"design_id": "notched", "source_sha256": design["content_sha256"],
                                  "family": "portable-design", "split": "external"}
    assert episode["program"]["prompt"] == "draw the notched square tattoo design"
    assert episode["engaged"]
