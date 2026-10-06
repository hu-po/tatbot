"""A travel runner whose driver wedges as the vendor's does, for test_landing.py. No hardware.

    python wedged_runner.py SCENARIO ESTOP_DEVICE SUMMARY GOLDEN

The blue arm is taken through BlueArm.connect() -- the e-stop monitor, the lease, the watchdog, then the driver --
over a stand-in session whose calls answer at once, except the one SCENARIO wedges: it blocks in recv() holding
the GIL, as the vendor binding does reading a TCP reply that never comes.

    connect   get_error_information, the controller-error check inside connect()
    measured  get_all_positions, a position read in the run's loop
    land      set_all_modes(idle), in the staged landing
    beyond    cleanup(), as connect() closes a session it refuses: joint 5 is counted a full turn off, and the
              recovery landing's clamp is the move refused
    estop     nothing: the runner waits on the e-stop once it is pressed, then lands, the staged landing's blocking
              move taking its whole goal time
    unwatched nothing: the watchdog never comes up

Prints ``watchdog <pid>``, ``connected``, ``wedging <call>`` as a call blocks, and what land() reported. The
budgets come from the environment: CALL_BUDGET_S, LANDING_BUDGET_S. The test's stand-in SDK is first on
PYTHONPATH; the recovery landing the watchdog starts runs over that one.
"""

from __future__ import annotations

import ctypes
import functools
import math
import os
import socket
import sys
import time
import types
from pathlib import Path

import trossen_arm
from tatbot_travel import hardware, watchdog

STAGED = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966]
CARRIAGE = 0.0123  # where the run finds the carriage
LIBC = ctypes.PyDLL(None)  # a PyDLL call keeps the GIL, as the vendor binding's calls do
LIBC.recv.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
LIBC.recv.restype = ctypes.c_ssize_t


class Session:
    """The stand-in driver session: every call answers at once, but the one named in ``wedge``."""

    wedge: str | None = None
    move_scale = 0.0  # a blocking move takes this much of its goal time
    wrist = 0.0  # where joint 5 is measured
    opened = 0
    LIMITS = [types.SimpleNamespace(position_min=-math.pi, position_max=math.pi, position_tolerance=0.2)] * 6 + [
        types.SimpleNamespace(position_min=0.0, position_max=0.04, position_tolerance=0.004)]

    def __init__(self):
        Session.opened += 1
        self.silent, self.peer = socket.socketpair()  # nothing is ever written to it
        self.q = [0.0] * 5 + [self.wrist, CARRIAGE]

    def _answer(self, call: str) -> None:
        if call == self.wedge:
            print("wedging", call, flush=True)
            LIBC.recv(self.silent.fileno(), ctypes.create_string_buffer(1), 1, 0)  # never returns

    def configure(self, *args):
        self._answer("configure")

    def set_end_effector(self, end_effector):
        self._answer("set_end_effector")

    def get_is_configured(self):
        self._answer("get_is_configured")
        return True

    def get_error_information(self):
        self._answer("get_error_information")
        return "No error"

    def set_all_modes(self, mode):
        self._answer("set_all_modes")

    def get_all_positions(self):
        self._answer("get_all_positions")
        return list(self.q)

    def get_joint_limits(self):
        self._answer("get_joint_limits")
        return list(self.LIMITS)

    def set_all_positions(self, q, goal_time=2.0, blocking=True):
        self._answer("set_all_positions")
        self.q = list(q)
        if blocking:
            time.sleep(goal_time * self.move_scale)

    def cleanup(self):
        self._answer("cleanup")


def blue_arm(device: str, golden: Path) -> hardware.BlueArm:
    """A BlueArm as __init__ leaves it, without the rig's configuration: a documentation address, the relay's
    e-stop, an empty golden, the stand-in session."""
    arm = hardware.BlueArm.__new__(hardware.BlueArm)
    arm.estop_required, arm.estop_device, arm.ip, arm.golden = True, device, "192.0.2.7", golden
    arm.trossen = types.SimpleNamespace(Model=trossen_arm.Model, Mode=trossen_arm.Mode,
                                        StandardEndEffector=trossen_arm.StandardEndEffector, TrossenArmDriver=Session)
    arm.end_effector = trossen_arm.StandardEndEffector.wxai_v0_leader
    arm.staged, arm.carriage = list(STAGED), 0.0
    arm.lease_path = hardware.plugin("driver_lease").HARDWARE_LEASE
    arm.lease = arm.estop = arm.driver = arm.watch = arm.summary = None
    return arm


def announce_watchdog() -> None:
    start = watchdog.DriverWatch.start

    def started():
        watch = start()
        print("watchdog", watch.process.pid, flush=True)
        return watch

    watchdog.DriverWatch.start = started


def main() -> None:
    scenario, device, summary, golden = sys.argv[1:5]
    recovery = hardware.plugin("recovery")
    recovery.LANDING_DEADLINE_S = float(os.environ["LANDING_BUDGET_S"])
    watchdog.CALL_BUDGET_S = float(os.environ["CALL_BUDGET_S"])
    # connect()'s staged raise, quick: its goal time is a default argument, so the constant cannot shorten it
    recovery.raise_arms_together = functools.partial(recovery.raise_arms_together, goal_time=0.05)
    announce_watchdog()
    if scenario == "unwatched":
        watchdog._command = lambda runner, fd: [sys.executable, "-c", "raise SystemExit(1)"]
    arm = blue_arm(device, Path(golden))
    Session.wedge = {"connect": "get_error_information", "beyond": "cleanup"}.get(scenario)
    if scenario == "beyond":
        Session.wrist = 1.487 - 2 * math.pi
    try:
        arm.connect(Path(summary))
    except RuntimeError as refused:
        print("refused", refused, flush=True)
        print("sessions", Session.opened, flush=True)
        print(arm.land(), flush=True)
        return
    print("connected", flush=True)
    if scenario == "measured":
        Session.wedge = "get_all_positions"
        arm.measured()
    Session.wedge = "set_all_modes" if scenario == "land" else None
    if scenario == "estop":
        Session.move_scale = 1.0
        while not arm.estopped:
            time.sleep(0.01)
    print(arm.land(), flush=True)


if __name__ == "__main__":
    main()
