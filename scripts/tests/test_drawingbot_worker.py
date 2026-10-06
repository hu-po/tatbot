"""Actual protocol/process failure paths and immutable bundle verification; no licensed app required."""
from __future__ import annotations

import json
import selectors
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from drawingbot.artifacts import digest
from drawingbot.bridge import Bridge, stop_process_group
from drawingbot.pens import verify_pen_readback
from drawingbot.recipe import read_json, runtime_project, save_bundle, verify_bundle


def effective():
    requested = read_json(Path(__file__).resolve().parents[1] / 'lib/drawingbot/defaults/drawing-set.json')
    actual = {**requested, 'native_set_id': 0, 'pens': [{**requested['pens'][0], 'native_row_id': 0,
              'name': 'black', 'export_group': 'Ballpoint_black'}]}
    return {'drawing': {'size_mm': [16, 24], 'pen_width_mm': .5},
            'drawing_set': verify_pen_readback(requested, actual, .5)}


def test_partial_protocol_reply_obeys_deadline_and_kills_worker_group():
    code = ("import os,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "sys.stdout.write('{'); sys.stdout.flush(); time.sleep(60)")
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, start_new_session=True)
    bridge = Bridge.__new__(Bridge)
    bridge.process, bridge.pending = process, b""
    bridge.selector = selectors.DefaultSelector()
    bridge.selector.register(process.stdout, selectors.EVENT_READ)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            bridge._read(.2)
        assert time.monotonic() - started < 2
    finally:
        stop_process_group(process)
        bridge.selector.close()
        process.stdout.close()
    assert process.returncode is not None


def test_shutdown_kills_descendant_even_after_wrapper_exits(tmp_path):
    marker = tmp_path / "alive"
    child = f"import pathlib,time; time.sleep(2); pathlib.Path({str(marker)!r}).write_text('alive')"
    code = f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child!r}])"
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    process.wait(timeout=2)
    stop_process_group(process)
    time.sleep(2.1)
    assert not marker.exists()


def test_reported_task_error_fails_without_waiting_for_export(tmp_path):
    bridge = Bridge.__new__(Bridge)
    bridge.call = lambda *a, **k: {"taskError": "Image decoding failed", "batchDisabled": False}
    with pytest.raises(RuntimeError, match="Image decoding failed"):
        bridge._wait_export(tmp_path / "source.svg", 300)


def test_single_job_worker_refuses_reuse_before_touching_output(tmp_path):
    bridge = Bridge.__new__(Bridge)
    bridge.used = True
    with pytest.raises(RuntimeError, match="one acquisition"):
        bridge.export({}, tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_portable_recipe_detects_source_state_and_runtime_drift(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"source identity fixture")
    project = tmp_path / "native.json"
    project.write_text(json.dumps({"data": {"imagePath": str(source), "settings": {}}}))
    variant = {"id": "a", "source": "a", "source_asset": "image.png", "pfm": "Sketch Lines", "density": "low", "overrides": {}}
    identity = {"jar_sha256": "a", "bridge_sha256": "b", "java_version": "21"}
    bundle = tmp_path / "recipe"
    save_bundle(bundle, project, source, variant, {"export": {}},
                effective(), identity)
    assert verify_bundle(bundle, identity)["files"]["source.png"] == digest(source)
    assert read_json(bundle / "project.json")["data"]["imagePath"] == "source.png"
    resolved = runtime_project(bundle / "project.json", source, tmp_path / "runtime.json")
    assert read_json(resolved)["data"]["imagePath"] == str(source)
    assert read_json(bundle / "project.json")["data"]["imagePath"] == "source.png"
    with pytest.raises(ValueError, match="java_version"):
        verify_bundle(bundle, {**identity, "java_version": "22"})
    assert verify_bundle(bundle, {**identity, "java_version": "22"}, migrate_runtime=True)
    with pytest.raises(ValueError, match="jar_sha256"):
        verify_bundle(bundle, {**identity, "jar_sha256": "different"}, migrate_runtime=True)
    (bundle / "state.json").write_text('{}')
    with pytest.raises(ValueError, match="state.json"):
        verify_bundle(bundle)


def test_native_recipe_removes_cached_drawing_and_job_local_directories(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    project = tmp_path / "native.json"
    project.write_text(json.dumps({"data": {"imagePath": "/temporary/input.png", "settings": {
        "ui_state": {"tab": "live"}, "drawingState": {"uuid": "random", "geometries": [1, 2]},
        "batch_processing": {"inputFolder": "/temporary/input", "outputFolder": "/temporary/output", "format": "SVG"},
        "pfm": {"seed": 42}}}}))
    bundle = tmp_path / "bundle"
    save_bundle(bundle, project, source, {"id": "test"}, {},
                effective(), {})
    settings = read_json(bundle / "project.json")["data"]["settings"]
    assert settings == {"batch_processing": {"inputFolder": "", "outputFolder": "", "format": "SVG"},
                        "pfm": {"seed": 42}}
    assert read_json(project)["data"]["settings"]["drawingState"]["uuid"] == "random"
