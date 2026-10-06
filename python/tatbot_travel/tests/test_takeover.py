"""The blue arm's takeover refuses a joint measured past its limits, and then nothing lands the arm.

No hardware. The controller is a stand-in as a power cycle can leave it: healthy, idle, its joint 5 counted a full
turn low. The driver lease is the real one, at a path of the test's own.
"""

from __future__ import annotations

import math
import sys
import types

import pytest
from tatbot_travel import hardware

STAGED = [0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2]
MODES = types.SimpleNamespace(idle="idle", position="position")


def limit(low: float, high: float, tolerance: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(position_min=low, position_max=high, position_tolerance=tolerance)


class Controller:
    """A healthy, idle controller measuring ``positions``, with the blue arm's golden limits."""

    LIMITS = [limit(-math.pi, math.pi, 0.2), limit(0.0, math.pi, 0.2), limit(0.0, 2.356, 0.2),
              limit(-math.pi / 2, math.pi / 2, 0.4), limit(-math.pi / 2, math.pi / 2, 0.4),
              limit(-math.pi, math.pi, 0.4), limit(0.0, 0.04, 0.004)]

    def __init__(self, positions):
        self.positions = list(positions)
        self.commands: list[tuple] = []
        self.closed = False

    def configure(self, *args):
        pass

    def set_end_effector(self, *args):
        pass

    def get_is_configured(self):
        return True

    def get_error_information(self):
        return "No error"

    def get_all_positions(self):
        return list(self.positions)

    def get_joint_limits(self):
        return list(self.LIMITS)

    def set_all_modes(self, mode):
        self.commands.append(("modes", mode))

    def set_all_positions(self, *args):
        self.commands.append(("positions", *args))

    def cleanup(self):
        self.closed = True


@pytest.fixture
def blue_arm(tmp_path, monkeypatch):
    """A BlueArm without an e-stop monitor over a stand-in controller; the golden is not written."""
    sdk = types.ModuleType("trossen_arm")  # the recovery module imports the vendor SDK; nothing here calls it
    sdk.Mode, sdk.Model = MODES, types.SimpleNamespace(wxai_v0="wxai_v0")
    sdk.StandardEndEffector = types.SimpleNamespace(wxai_v0_leader="wxai_v0_leader")
    if "trossen_arm" not in sys.modules:
        monkeypatch.setitem(sys.modules, "trossen_arm", sdk)
    recovery, goldens = hardware.plugin("recovery"), hardware.plugin("goldens")
    monkeypatch.setattr(goldens, "apply_arm_golden", lambda *args: [])
    staged_to = []
    monkeypatch.setattr(recovery, "raise_arms_together", lambda arms, **kwargs: staged_to.append(arms) or True)

    def make(controller: Controller) -> hardware.BlueArm:
        arm = hardware.BlueArm.__new__(hardware.BlueArm)
        arm.estop_required, arm.estop_device, arm.estop = False, None, None
        arm.trossen = types.SimpleNamespace(TrossenArmDriver=lambda: controller, Mode=MODES,
                                            Model=types.SimpleNamespace(wxai_v0="wxai_v0"))
        arm.ip, arm.golden, arm.end_effector = "192.0.2.7", tmp_path / "leader.yaml", "wxai_v0_leader"
        arm.staged, arm.carriage = list(STAGED), 0.0
        arm.lease_path, arm.lease, arm.driver = tmp_path / "arm-driver.lock", None, None
        arm.staged_to = staged_to
        return arm

    return make


def test_a_joint_counted_a_full_turn_off_is_refused_and_the_arm_is_left_alone(blue_arm, monkeypatch):
    controller = Controller([0.018, 0.002, 0.001, -0.008, -0.008, 1.487 - 2 * math.pi, 0.0])
    arm = blue_arm(controller)
    with pytest.raises(RuntimeError, match=r"joint 5 at -4\.796 rad"):
        arm.connect()
    assert controller.commands == [] and controller.closed and arm.staged_to == []
    # The runner lands whatever connect() left: here, nothing. The recovery landing would only refuse joint 5
    # again (the next test).
    monkeypatch.setattr(hardware.BlueArm, "_recover", lambda self, recovery: pytest.fail("handed to the recovery"))
    assert arm.land() is None
    assert hardware.lease_free(arm.lease_path)


def test_the_recovery_landing_refuses_that_joint_too_and_exits_7(blue_arm, monkeypatch, caplog):
    # The recovery's own process over the same controller, through the plugin's land_arm: its session is closed
    # with nothing commanded, and it exits 7 (refused) rather than 1 (arm state unknown).
    recovery, driver_lease = hardware.plugin("recovery"), hardware.plugin("driver_lease")
    if not hardware.lease_free(driver_lease.HARDWARE_LEASE):  # land_arm takes the lease at its one fixed path
        pytest.skip("the arm-driver lease is held on this node")
    controller = Controller([0.018, 0.002, 0.001, -0.008, -0.008, 1.487 - 2 * math.pi, 0.0])
    monkeypatch.setattr(recovery.trossen_arm, "TrossenArmDriver", lambda: controller, raising=False)
    staged = ",".join(repr(v) for v in [*STAGED, 0.0])
    assert hardware.recovery_landing(["192.0.2.7", "--staged", staged]) == 7
    assert controller.commands == [] and controller.closed
    assert "joint 5 at -4.796 rad" in caplog.text and "turn joint 5 by hand to near 0" in caplog.text
    assert 7 in hardware.RECOVERY_EXITS


def test_joints_resting_on_their_stops_within_tolerance_are_taken_over(blue_arm):
    # Shoulder and elbow read a few mrad below 0 on their stops; the carriage stays where the run found it.
    controller = Controller([0.0, -0.0055, -0.0017, 0.0, 0.0, math.pi / 2, 0.0123])
    arm = blue_arm(controller)
    arm.connect()
    assert controller.commands[:2] == [("modes", "position"), ("positions", controller.positions, 0.5, False)]
    assert arm.carriage == 0.0123 and len(arm.staged_to) == 1
    assert arm.land() == "landed"
    assert controller.commands[-1] == ("modes", "idle") and controller.closed
    assert hardware.lease_free(arm.lease_path)


def test_beyond_limits_names_only_joints_past_the_tolerance():
    limits = Controller.LIMITS[:6]
    assert hardware.beyond_limits([0.0, -0.19, 0.0, 0.0, 0.0, math.pi + 0.39], limits) == []
    named = hardware.beyond_limits([0.0, -0.21, 0.0, 0.0, 0.0, -math.pi - 0.41], limits)
    assert [line.split(" at ")[0] for line in named] == ["joint 1", "joint 5"]
