"""Physical job inputs are checked before starting a licensed application."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import tatbot_cli  # noqa: F401 -- shared stdlib source root
from drawingbot.job import SCHEMA, generate, load_job, validate_job
from drawingbot.recipe import DEFAULTS, read_json


def job():
    return {"schema": SCHEMA, "id": "small-peony", "source": {
        "file": "source.png", "sha256": hashlib.sha256(b"image bytes").hexdigest(),
        "provenance": {"kind": "fixture", "identifier": "test", "license": None, "attribution": None, "generation": None}}, "size_mm": [16, 24], "pen_width_mm": .5,
        "pfm": "Sketch Sweeping Curves", "settings": {"Random Seed": 8, "Line Density": 35.0},
        "state": read_json(DEFAULTS / "state.json"), "drawing_set": read_json(DEFAULTS / "drawing-set.json")}


@pytest.mark.parametrize("field,value", [("size_mm", [float("nan"), 24]), ("size_mm", [16, 0]),
                                        ("size_mm", [True, 24]), ("pen_width_mm", float("inf")),
                                        ("pen_width_mm", -1), ("id", "../../escape")])
def test_invalid_physical_job_is_refused(field, value):
    value_job = job()
    value_job[field] = value
    with pytest.raises(ValueError):
        validate_job(value_job)


def test_job_requires_complete_explicit_state_and_seed():
    value = job()
    del value["settings"]["Random Seed"]
    with pytest.raises(ValueError, match="Random Seed"):
        validate_job(value)
    value = job()
    del value["state"]["export"]["multipassEnabled"]
    with pytest.raises(ValueError, match="all export settings"):
        validate_job(value)


def test_job_binds_source_bytes_and_load_does_not_mutate(tmp_path):
    path = tmp_path / "job.json"
    value = job()
    path.write_text(json.dumps(value))
    (tmp_path / "source.png").write_bytes(b"image bytes")
    loaded, source = load_job(path)
    assert loaded == value and source == tmp_path / "source.png"
    loaded["size_mm"][0] = 5
    assert load_job(path)[0]["size_mm"] == [16, 24]
    source.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="source bytes"):
        generate(path, tmp_path / "result", tmp_path / "nonexistent-app")
    assert not (tmp_path / "result").exists()


def test_existing_acquisition_is_never_overwritten(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job()))
    (tmp_path / "source.png").write_bytes(b"image bytes")
    out = tmp_path / "out"
    out.mkdir()
    (out / "result.json").write_text("retained")
    with pytest.raises(FileExistsError):
        generate(path, out, tmp_path / "nonexistent-app")
    assert (out / "result.json").read_text() == "retained"
