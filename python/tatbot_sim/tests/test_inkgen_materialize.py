"""Inkgen is a bounded pre-sim materializer, never an environment dependency."""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np
import pytest
from tatbot_sim.inkmap.designs import directory_artifacts
from tatbot_sim.inkmap.inkgen_materialize import (
    InkgenMaterializationError,
    materialize_inkgen_designs,
    trace_raster_svg,
)


def _png_base64(blank: bool = False, seed: int = 0) -> str:
    """A drawing that depends on its seed, the way a real generator's does.

    Distinct seeds must give distinct bytes, or the batch deduplicator will
    correctly collapse them and the test will be measuring the wrong thing.
    """
    image = np.full((96, 128, 3), 255, dtype=np.uint8)
    if not blank:
        cv2.circle(image, (42, 48), 24, (15, 15, 15), 5)
        rng = np.random.default_rng(seed)
        for _ in range(3):
            x0, y0, x1, y1 = (int(v) for v in rng.integers(8, 86, size=4))
            cv2.line(image, (x0, y0), (x1, y1), (25, 25, 25), 4)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(encoded).decode()


def test_materializer_writes_exact_raster_svg_and_provenance(tmp_path):
    calls = []

    def request(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {
            "png_base64": _png_base64(seed=payload["seed"]),
            "seed": payload["seed"],
            "prompt": f"service prompt {payload['subject']}",
            "model": "test-inkgen",
        }

    output = tmp_path / "materialized"
    manifest = materialize_inkgen_designs(
        output,
        ["a heron", "a fern"],
        count=3,
        seed=40,
        api_url="http://inkgen.test:8600",
        style="single needle",
        request_json=request,
    )
    # Seeds belong to the item, not to its position: the same key always draws
    # the same picture no matter what else is in the job.
    assert len({call[1]["seed"] for call in calls}) == 3
    assert [call[1]["subject"] for call in calls] == ["a heron", "a fern", "a heron"]
    assert len(manifest["artifacts"]) == 3
    assert len(list(output.glob("*.svg"))) == 3
    assert len(list(output.glob("*.png"))) == 3
    sidecar = json.loads(next(output.glob("0*.json")).read_text())
    assert sidecar["source"]["model"] == "test-inkgen"
    assert sidecar["source"]["trace"]["algorithm"] == "inkmap-vtracer-otsu-v2"
    with pytest.raises(ValueError, match="complete DBV3 acquisition"):
        directory_artifacts(output, (99.0, 99.0))


def test_tracer_rejects_blank_or_solid_outputs():
    # The tracer prepares source imagery only, so its refusals are the
    # shared builder's; InkgenMaterializationError now names the Inkgen exchange.
    from tatbot_sim.inkmap.design_build import DesignBuildError

    for value in (0, 255):
        image = np.full((64, 64, 3), value, dtype=np.uint8)
        with pytest.raises(DesignBuildError, match="coverage"):
            trace_raster_svg(image)


def test_shared_raster_conversion_preserves_painted_ring_and_empty_hole():
    from shapely import Point, Polygon, union_all
    from tatbot_sim.inkmap.artwork import make_artwork_record
    image = np.full((128, 128, 3), 255, np.uint8)
    cv2.circle(image, (64, 64), 40, (0, 0, 0), 12)
    svg, _ = trace_raster_svg(image)
    record = make_artwork_record(name="ring", original_svg=svg,
        source={"kind": "imported", "identifier": "ring-test", "license": None, "attribution": None, "generation": None},
        conversion={"adapter": "tatbot-svg-paint/1", "canvas_m": [.03, .03],
                    "semantic_intent": "ring", "width_m": .0003, "deposition": 1, "chord_error_m": .000005})
    paint = union_all([Polygon(e["points_m"]) for layer in record["program"]["layers"] for e in layer["elements"]])
    assert not paint.covers(Point(.015, .015))
    assert paint.covers(Point(.015, .003))
    assert paint.area > .0001  # Finite painted band, not a contour-only substitute.


def test_materializer_does_not_create_partial_directory_for_bad_arguments(tmp_path):
    output = tmp_path / "not-created"
    with pytest.raises(InkgenMaterializationError, match="subject"):
        materialize_inkgen_designs(
            output, [], count=1, seed=0, api_url="http://inkgen.test", request_json=lambda *_: {}
        )
    assert not output.exists()


def _seeded_service(fail_after=None, blank_subjects=()):
    state = {"calls": 0}

    def request(url, payload, timeout):
        state["calls"] += 1
        if fail_after is not None and state["calls"] > fail_after:
            raise InkgenMaterializationError("Inkgen request failed: connection refused")
        return {"png_base64": _png_base64(blank=payload["subject"] in blank_subjects,
                                          seed=payload["seed"]),
                "seed": payload["seed"], "prompt": f"p {payload['subject']}", "model": "test-inkgen"}

    return request, state


def test_an_interrupted_batch_resumes_with_unchanged_artifact_bytes(tmp_path):
    output = tmp_path / "materialized"
    subjects = ["a heron", "a fern", "a koi", "a moth"]
    stopping, first = _seeded_service(fail_after=2)
    with pytest.raises(InkgenMaterializationError, match="resume it"):
        materialize_inkgen_designs(output, subjects, count=4, seed=11,
                                   api_url="http://inkgen.test:8600", request_json=stopping,
                                   max_attempts=1)
    done = {path.name: path.read_bytes() for path in output.glob("*.svg")}
    assert 0 < len(done) < 4
    assert not (output / "manifest.json").exists()

    resuming, second = _seeded_service()
    manifest = materialize_inkgen_designs(output, subjects, count=4, seed=11,
                                          api_url="http://inkgen.test:8600", request_json=resuming)
    assert len(manifest["artifacts"]) == 4
    assert all((output / name).read_bytes() == data for name, data in done.items())
    assert second["calls"] == 4 - len(done)  # completed work is not asked for again
    with pytest.raises(ValueError, match="complete DBV3 acquisition"):
        directory_artifacts(output, (99.0, 99.0))


def test_a_short_job_is_never_admitted_as_a_complete_library(tmp_path):
    output = tmp_path / "materialized"
    refusing, _ = _seeded_service(blank_subjects={"a fern"})
    with pytest.raises(InkgenMaterializationError, match="1 refused"):
        materialize_inkgen_designs(output, ["a heron", "a fern"], count=2, seed=5,
                                   api_url="http://inkgen.test:8600", request_json=refusing)
    selection = json.loads((output / "selection.json").read_text())
    assert (selection["accepted"], selection["refused"], selection["complete"]) == (1, 1, False)
    with pytest.raises(ValueError, match="complete DBV3 acquisition"):
        directory_artifacts(output, (99.0, 99.0))


def test_a_replacement_budget_fills_a_refused_slot(tmp_path):
    """The blank draw is the first candidate's; its replacement has a new seed."""
    first = {}

    def request(url, payload, timeout):
        blank = False
        if payload["subject"] == "a fern":
            first.setdefault("fern", payload["seed"])
            blank = payload["seed"] == first["fern"]
        return {"png_base64": _png_base64(blank=blank, seed=payload["seed"]),
                "seed": payload["seed"], "prompt": "p", "model": "test-inkgen"}

    output = tmp_path / "materialized"
    manifest = materialize_inkgen_designs(output, ["a heron", "a fern"], count=2, seed=5,
                                          api_url="http://inkgen.test:8600", request_json=request,
                                          replacement_budget=2)
    assert len(manifest["artifacts"]) == 2
    assert manifest["report"]["refused"] == 1 and manifest["report"]["requested"] == 2
    with pytest.raises(ValueError, match="complete DBV3 acquisition"):
        directory_artifacts(output, (99.0, 99.0))


def test_a_missing_local_backend_makes_zero_public_requests(tmp_path, monkeypatch):
    """No `inkgen` role and no address: refuse, never reach for the public Space."""
    from tatbot_sim.inkmap import inkgen_client

    def never(*_args, **_kwargs):
        raise AssertionError("a bulk job must not contact or start any generator")

    monkeypatch.setattr(inkgen_client, "fleet_endpoint", lambda node_config=None: None)
    monkeypatch.setattr(inkgen_client, "health", never)
    monkeypatch.setattr(inkgen_client, "start", never)
    with pytest.raises(inkgen_client.InkgenBackendError, match="never falls back"):
        materialize_inkgen_designs(tmp_path / "unused", ["a heron"], count=1, seed=0,
                                   api_url="", autostart=True, request_json=never)
    assert not (tmp_path / "unused").exists()
