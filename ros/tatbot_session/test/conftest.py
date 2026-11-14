"""Every test here reads the station touch it writes itself: the ros node's live one (~/tatbot-ros/calib) would set the
dips' heading the executor's pen heading prefers (ArmExecutor._dip_heading). And every test's rig is the WXAI wrist with
its two D405s, whichever arm vision.toml mounts them on today (the xArm since 2026-11-08)."""
from __future__ import annotations

import os

import pytest

PINK_WRIST = """
schema_version = 1
[sync]
""" + "".join(f"""
[[cameras.realsense]]
name = "{name}"
serial = "{serial}"
role = "{role}"
group = "d405"
arm = "right"
owner_role = "ros"
optical_frame = "right/{prefix}_color_optical_frame"
depth_optical_frame = "right/{prefix}_depth_optical_frame"
[cameras.realsense.color]
stream = "color"
width = 640
height = 480
fps_num = 30
fps_den = 1
[cameras.realsense.depth]
stream = "depth"
width = 640
height = 480
fps_num = 30
fps_den = 1
""" for name, serial, role, prefix in (("realsense2", "000000000000", "wrist_2", "realsense"),
                                       ("realsense1", "000000000001", "wrist_1", "realsense_lower")))


@pytest.fixture(autouse=True)
def _no_live_station_touch(tmp_path, monkeypatch):
    from tatbot_motion import station

    monkeypatch.setattr(station, "touch_path", lambda arm: tmp_path / f"no-station-touch-{arm}.json")


@pytest.fixture(autouse=True)
def _pink_wrist_cameras(tmp_path, monkeypatch):
    import tatbot_description  # noqa: F401  (it puts scripts/lib, wrist_cameras' home, on the path)
    import wrist_cameras

    path = tmp_path / "vision.toml"
    path.write_text(PINK_WRIST)
    monkeypatch.setattr(wrist_cameras, "registry_path", lambda _repo: path)


_over_budget = []


def pytest_runtest_logreport(report):
    budget = float(os.environ.get("TATBOT_TEST_BUDGET_S", "60"))
    if (report.when == "call" and report.passed and "slow" not in report.keywords
            and budget > 0 and report.duration > budget):
        _over_budget.append((report.nodeid, report.duration))


def pytest_sessionfinish(session, exitstatus):
    if _over_budget:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        for nodeid, duration in _over_budget:
            reporter.write_line(f"{nodeid}: {duration:.1f}s exceeds the test budget; mark slow or make cheaper")
        session.exitstatus = max(int(exitstatus), 1)
