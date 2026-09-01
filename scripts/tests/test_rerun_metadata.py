"""Tests for Python Rerun producer provenance."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/vision"))

from rerun_metadata import log_producer_metadata  # noqa: E402


class FakeRerun:
    class TextLog:
        def __init__(self, text):
            self.text = text

    def __init__(self):
        self.logged = []

    def log(self, entity, value, *, static=False):
        self.logged.append((entity, value.text, static))


def test_metadata_records_calibration_urdf_and_source_identity(tmp_path, monkeypatch):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"bundle_id": "bundle-123"}))
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot/>")
    monkeypatch.setenv("TATBOT_SOURCE_COMMIT", "abc123")
    rerun = FakeRerun()

    metadata = log_producer_metadata(
        rerun, "live surface", "recording-1", calibration, urdf
    )

    assert metadata["source_commit"] == "abc123"
    assert metadata["calibration_id"] == "bundle-123"
    assert metadata["urdf_sha256"] == hashlib.sha256(b"<robot/>").hexdigest()
    assert rerun.logged[0][0] == "session/producers/live_surface"
    assert rerun.logged[0][2] is True
