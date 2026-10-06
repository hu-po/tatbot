"""The package mock backend drives the same code paths the tests' fakes do.

Plan Phase 2 exit gate: a synthetic/mock backend passes the recovery, golden
and tool code paths with no vendor hardware. The interface is pinned against
the package's real driver call sites, so a driver-API drift breaks here, not
at connect time on the rig.
"""

from __future__ import annotations

from lerobot_robot_tatbot.mock_driver import MockDriver


def test_mock_driver_is_neutral_not_measured():
    d = MockDriver()
    jc = d.get_joint_characteristics()
    assert [c.friction_constant_term for c in jc] == [0.0] * 7
    assert [c.effort_correction for c in jc] == [1.0] * 7


def test_mock_driver_kinematic_state_round_trips():
    d = MockDriver()
    d.set_positions([0.1] * 7)
    assert d.get_positions() == [0.1] * 7
    assert d.get_error_information() == ""


def test_mock_covers_every_method_the_codebase_calls_on_a_driver():
    """Pin the mock against real call sites: every `driver.<name>(` and
    `self.driver.<name>(` in the package must exist on MockDriver — so a
    driver-API drift in recovery/goldens breaks HERE, not on the rig."""
    import re
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[1] / "src" / "lerobot_robot_tatbot"
    called = set()
    for f in pkg.glob("*.py"):
        if f.name == "mock_driver.py":
            continue
        for line in f.read_text().splitlines():
            code = line.split("#", 1)[0]
            for m in re.finditer(r"(?:\bdriver|\.driver)\.(\w+)\s*\(", code):
                # skip matches inside string literals (error messages)
                if '"' in code[:m.start()] or "'" in code[:m.start()]:
                    continue
                called.add(m.group(1))
    missing = sorted(n for n in called if not hasattr(MockDriver(), n))
    assert not missing, f"MockDriver lacks methods the code calls: {missing}"


def test_mock_connection_state_is_cleared_by_cleanup():
    d = MockDriver()
    assert not d.get_is_configured()
    d.configure()
    assert d.get_is_configured()
    d.cleanup()
    assert not d.get_is_configured()
