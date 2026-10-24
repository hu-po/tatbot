"""Tests for Python Rerun producer provenance."""

from __future__ import annotations

import hashlib
import json

from tatbot_rerun import log_producer_metadata  # noqa: E402


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


def test_watch_sink_exits_once_the_sink_is_gone_and_waits_through_a_timeout(monkeypatch):
    """Rerun 0.36 never redials: a flush naming a disconnected sink exits the producer for its supervisor; a timed-out
    flush (a hub still starting) does not."""
    import threading

    import tatbot_rerun

    errors = iter([RuntimeError("Flush timed out - not all log messages were sent"),
                   RuntimeError("gRPC connection gracefully disconnected, uri: rerun+http://hub:9876/proxy")])

    class Recording:
        def flush(self, *, timeout_sec):
            raise next(errors)

    exited, said = threading.Event(), []
    monkeypatch.setattr(tatbot_rerun.os, "_exit", lambda code: (said.append(code), exited.set()))
    tatbot_rerun.watch_sink(said.append, exit_code=1, period_s=0.01, recording=Recording())
    assert exited.wait(5.0) and said[-1] == 1 and "disconnected" in said[0]
