"""Every test here reads the station touch it writes itself: the ros node's live one (~/tatbot-ros/calib) would set the
dips' heading the executor's pen heading prefers (ArmExecutor._dip_heading)."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _no_live_station_touch(tmp_path, monkeypatch):
    from tatbot_motion import station

    monkeypatch.setattr(station, "touch_path", lambda arm: tmp_path / f"no-station-touch-{arm}.json")


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
