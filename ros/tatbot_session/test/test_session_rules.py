"""Decide rules (ros/README.md section 8)."""
from tatbot_session import rules


def ok(**kw):
    base = {"estop_ok": True, "controller_error": False, "joint_moved_rad": 0.0, "tip_moved_m": 0.0,
            "tool_changed": False}
    base.update(kw)
    return rules.continue_refusal(**base)


def test_continue_allowed_when_nothing_changed():
    assert ok() is None
    assert ok(joint_moved_rad=0.049, tip_moved_m=0.0049) is None


def test_continue_refusals_leave_land():
    assert "e-stop" in ok(estop_ok=False)
    assert "controller" in ok(controller_error=True)
    assert "moved" in ok(joint_moved_rad=0.051)
    assert "moved" in ok(tip_moved_m=0.0051)
    assert "tool" in ok(tool_changed=True)


def test_decisions_fit_what_the_arm_waits_for():
    assert rules.check("latched", rules.CONTINUE) is None
    assert rules.check("latched", rules.LAND) is None
    assert rules.check("latched", rules.SKIP) is not None
    assert rules.check("pause", rules.CONTINUE) is None
    assert rules.check("pause", rules.REDRAW) is not None
    assert rules.check("uncertain", rules.REDRAW) is None
    assert rules.check("uncertain", rules.SKIP) is None
    assert rules.check("uncertain", rules.LAND) is None
    assert rules.check("uncertain", rules.CONTINUE) is not None
    assert rules.check(None, rules.CONTINUE) is None
    assert rules.check(None, rules.SKIP) is not None
    assert rules.check("latched", 9) is not None


def test_resume_timeout():
    assert not rules.resume_expired(None, 1000.0, 300.0)  # still pressed
    assert not rules.resume_expired(0.0, 300.0, 300.0)
    assert rules.resume_expired(0.0, 300.1, 300.0)


def test_decide_constants_match_the_interface():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "tatbot_interfaces" / "srv" / "Decide.srv").read_text()
    for value, name in rules.NAMES.items():
        assert f"{name.upper()}={value}" in text
