"""What an arm plugin does around a driver session.

The arm takes the exclusive driver lease, acquires the e-stop monitor before
anything moves, and releases both the same way on the way out. It is a mixin:
it was shared by the follower and the LeRobot leader teleoperator, which
descended from different LeRobot bases, until the leader plugin was removed
with the recording path on 2026-09-29.

It was copy-paste until 2026-09-09, and it had already drifted: the hold that
waits out a latched e-stop during disconnect polled every 50 ms on the follower
and every ``recovery.ESTOP_POLL_S`` on the leader. Nothing compared them. A fix
to one arm's e-stop path reaching only one arm is the failure this module
exists to prevent, and it is the path AGENTS.md says not to weaken.

Everything here needs ``self.config`` (``estop_device``, ``estop_required``),
``self.driver``, and ``self._estop``; the arm classes own the rest.
"""

from __future__ import annotations

import logging
import time

from lerobot_robot_tatbot import recovery
from lerobot_robot_tatbot.driver_lease import acquire as acquire_driver_lease
from lerobot_robot_tatbot.estop import EstopMonitor, acquire_estop, release_estop


class TatbotArmSession:
    """Driver lease and e-stop lifecycle shared by both arms.

    ``ARM_ROLE`` names the arm in operator-facing messages. Subclasses set it.
    """

    ARM_ROLE = "arm"

    _estop: EstopMonitor | None

    @property
    def _log(self) -> logging.Logger:
        """The subclass's own logger, so records keep the module names that
        operators and the run logs already filter on."""
        return logging.getLogger(type(self).__module__)

    def connect(self, calibrate: bool = True) -> None:
        if getattr(self, "_driver_lease", None) is not None:
            raise RuntimeError("this arm already owns a driver lease")
        try:
            self._driver_lease = acquire_driver_lease()
        except RuntimeError:
            self._driver_lease_refused = True
            raise
        self._driver_lease_refused = False
        try:
            self._connect_owned(calibrate=calibrate)
        except BaseException:
            # Retain ownership if driver cleanup fails; do not invite a
            # second process to replace a possibly live controller session.
            try:
                self.driver.cleanup()
            except Exception:
                self._log.exception("failed connection cleanup; retaining driver lease")
            else:
                self._driver_lease.close()
                self._driver_lease = None
            raise

    def acquire_estop_or_refuse(self) -> None:
        """Take the e-stop monitor before anything moves, or refuse to start.

        Called from each arm's ``configure()``, ahead of the staged-pose move.
        Both arms acquire the same underlying object, so the serial bytes
        always have exactly one consumer.
        """
        if self._estop is not None:
            release_estop(self._estop)
            self._estop = None
        if self.config.estop_required and not self.config.estop_device:
            # Reachable since the profile refactor: an unresolvable profile
            # defaults the device to "" while required stays True. Refuse —
            # never move arms unmonitored because a config file was absent.
            raise RuntimeError(
                "estop_required=True but no e-stop device resolved from the "
                "hardware profile — fix TATBOT_PROFILE/config/profiles, or "
                "set estop_required=False for an explicit hardware-free bench")
        if not self.config.estop_device:
            return
        self._estop = acquire_estop(
            self.config.estop_device, required=self.config.estop_required
        )
        if self._estop is None:
            return
        self._estop.wait_for_initial_state()
        if self._estop.engaged:
            state = self._estop.state.value
            release_estop(self._estop)
            self._estop = None
            raise RuntimeError(
                f"e-stop engaged ({state}) — twist-release "
                "the button (or reconnect the box) and retry"
            )

    def hold_while_estop_engaged(self) -> None:
        """Block until a latched e-stop is twist-released.

        The stock disconnect drives the arm (staged/sleep moves); never do that
        with the e-stop latched. Ctrl+C during the wait aborts disconnect —
        driver teardown idles the arm, which is the safe outcome.
        """
        if self._estop is None or not self._estop.engaged:
            return
        self._log.warning(
            "%s disconnect requested with e-stop engaged (%s): holding until "
            "twist-release", self.ARM_ROLE, self._estop.state.value,
        )
        while self._estop.engaged:
            time.sleep(recovery.ESTOP_POLL_S)

    def disconnect(self, land: bool = True) -> None:
        if getattr(self, "_driver_lease_refused", False):
            return  # A refused connection owns nothing to move or disconnect.
        self._disconnect_owned(land=land)
        lease = getattr(self, "_driver_lease", None)
        if lease is not None:
            if self.driver.get_is_configured():
                raise RuntimeError("disconnect left driver configured; retaining ownership")
            lease.close()
            self._driver_lease = None
