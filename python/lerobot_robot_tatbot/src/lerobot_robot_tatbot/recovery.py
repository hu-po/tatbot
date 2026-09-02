"""Emergency arm recovery: never leave an arm frozen.

When a session dies mid-motion (driver TCP/UDP fault, firmware trip, crash),
the graceful staged→sleep disconnect can't run because every driver call
raises. This module's park ritual handles that case: drop the wedged
connection, reconnect fresh with clear_error=True (which clears controller
fault state), take control softly at the current pose, then staged pose →
sleep pose → all motors idle → cleanup. Used by both plugins' disconnect
fallbacks and by the standalone tune session on any error.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import threading
import time

import trossen_arm

logger = logging.getLogger(__name__)

SLEEP_POSE_S = 3.0


def sleep_pose_for(staged_positions, gripper_index):
    """The sleep pose: every joint at zero EXCEPT the wrist roll (joint 5) and
    the carriage, which keep the staged pose's values. Since 2026-08-30 the
    staged pose IS the sleep pose with the wrist rolled +90 deg so the
    fiducial cube points up; an arm that rests cube-up, and lands without a
    90 deg wrist move at the very end, is what the operator asked for. The
    carriage goes to its staged (rest) value: nothing is gripped, and a
    session must not inherit a retracted carriage from the last one."""
    n = len(staged_positions)
    pose = [0.0] * n
    if n > 5:
        pose[5] = float(staged_positions[5])
    pose[gripper_index] = float(staged_positions[gripper_index])
    return pose
STAGED_POSE_S = 4.0
TAKEOVER_S = 0.5
RETRY_DELAY_S = 2.0
CONFIGURE_TIMEOUT_S = 5.0  # driver default is 20 s — far too long to hang a landing
STAGED_TOLERANCE_RAD = 0.15
LANDED_TOLERANCE_RAD = 0.20   # "did it actually reach the sleep pose?"
LANDING_DEADLINE_S = 45.0     # whole-landing budget across retries
HANG_ESCAPE_S = 10.0          # Ctrl+C is honoured again after this long
ESTOP_POLL_S = 0.02


# get_error_information() renders ErrorState as human-readable text, and a
# HEALTHY controller returns the literal string "No error" (ErrorState::none)
# — not an empty string. Treating any non-empty return as a fault aborted
# healthy sessions on 2026-08-20; keep this set in sync with the SDK.
CLEAN_ERROR_STRINGS = {"", "no error", "none", "error state: none"}


def controller_error(driver) -> str:
    """The controller's error string, or '' when the controller is healthy."""
    try:
        # Reading from an unconfigured driver raises and logs CRITICAL; skip
        # it (an arm we never connected has no error state to report).
        is_configured = getattr(driver, "get_is_configured", None)
        if is_configured is not None and not is_configured():
            return ""
        raw = str(driver.get_error_information() or "").strip()
    except Exception:
        return ""
    return "" if raw.lower() in CLEAN_ERROR_STRINGS else raw


def assert_controller_healthy(driver, expected_positions, name: str,
                              tol: float = STAGED_TOLERANCE_RAD) -> None:
    """Refuse to start (or continue) a session on a wedged controller.

    The 2026-08-20 second incident: the follower came up 'connected' but was
    still faulted from the previous crash — it accepted configure and a
    blocking staged move, tracked nothing, and died a minute in. This check
    turns that into an immediate, explicit abort: a reported firmware error
    or a staged pose that was commanded but not reached both mean the
    controller needs a power cycle, not a teleop session.
    """
    err = controller_error(driver)
    if err:
        raise RuntimeError(
            f"{name} controller reports a firmware error: {err!r} — "
            "power-cycle the arm before starting a session"
        )
    if expected_positions is not None:
        positions = list(driver.get_all_positions())
        worst = max(
            abs(p - e) for p, e in zip(positions, expected_positions, strict=True)
        )
        if worst > tol:
            raise RuntimeError(
                f"{name} did not reach the staged pose (worst joint off by "
                f"{worst:.2f} rad) — the controller is not executing motion "
                "commands; power-cycle the arm"
            )


class TrackingWatchdog:
    """Abort instead of silently dragging a non-tracking arm.

    Detects "commanded to move, didn't move" — and ONLY that. Tracking error
    alone is not the signal: ``max_relative_target`` clamps the sent target
    to present ± 0.5 rad, so the error pins at exactly 0.5 both when the
    controller is frozen AND when the operator simply outruns the follower's
    slew limit. The discriminator is physical progress: a healthy arm under
    a saturated clamp is still slewing at up to max_joint_velocity, while a
    wedged controller does not move at all. So the watchdog fires only when
    the error stays high for ``grace_s`` AND the measured position has moved
    less than ``min_progress_rad`` over that whole window.
    """

    def __init__(self, threshold_rad: float = 0.35, grace_s: float = 2.0,
                 min_progress_rad: float = 0.05):
        self.threshold = threshold_rad
        self.grace = grace_s
        self.min_progress = min_progress_rad
        self._since: float | None = None
        self._anchor: list[float] | None = None

    def update(self, worst_error_rad: float, positions, now: float,
               driver=None) -> None:
        if worst_error_rad < self.threshold:
            self._since = None
            self._anchor = None
            return
        anchor = self._anchor
        if self._since is None or anchor is None:  # error window opens; remember where we were
            self._since = now
            self._anchor = list(positions)
            return
        if now - self._since <= self.grace:
            return
        progress = max(
            abs(p - a) for p, a in zip(positions, anchor, strict=True)
        )
        if progress >= self.min_progress:
            # Lagging but moving — the operator is outrunning the slew
            # limiter. Healthy: restart the window from the current pose.
            self._since = now
            self._anchor = list(positions)
            return
        err = controller_error(driver) if driver is not None else ""
        raise RuntimeError(
            f"arm is not executing motion commands (worst joint error "
            f"{worst_error_rad:.2f} rad for {now - self._since:.1f} s while "
            f"moving only {progress:.3f} rad); firmware error: "
            f"{err or 'none reported'} — aborting session"
        )


class SigintShield:
    """Swallow Ctrl+C while the arm is landing, with a hang escape.

    A terminal Ctrl+C hits the whole foreground process group, and a second
    press during a landing would abort it mid-motion and leave the arm
    frozen holding its last position-mode target — the worst outcome. So
    repeats are swallowed while the landing runs. Two escapes remain: a
    press more than ``escape_s`` after the first is let through (a genuinely
    hung landing must stay interruptible), and the hardware e-stop is always
    live.

    Also clears SA_RESTART's inverse: a swallowed signal must not EINTR-abort
    the SDK's blocking socket reads, because the driver does not retry them.
    """

    def __init__(self, escape_s: float = HANG_ESCAPE_S):
        self.escape_s = escape_s
        self._first = 0.0
        self._prev = None
        self._installed = False

    def _handler(self, signum, frame):
        now = time.monotonic()
        if self._first == 0.0:
            self._first = now
            logger.warning(
                "landing in progress — Ctrl+C ignored (the arm idles in a few "
                "seconds; press again after %.0f s if it is stuck)",
                self.escape_s,
            )
            return
        if now - self._first > self.escape_s:
            logger.error(
                "landing appears hung — allowing Ctrl+C through; recover with "
                "scripts/il_recover_arm.sh"
            )
            raise KeyboardInterrupt

    def __enter__(self):
        try:  # signal handlers can only be installed on the main thread
            self._prev = signal.signal(signal.SIGINT, self._handler)
            signal.siginterrupt(signal.SIGINT, False)
            self._installed = True
        except (ValueError, OSError):
            logger.debug("SIGINT shield unavailable off the main thread")
        return self

    def __exit__(self, *exc):
        if self._installed and self._prev is not None:
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGINT, self._prev)
        return False


def _apply_golden(driver, name: str) -> list[str]:
    """Push the arm's golden (config/trossen/<name>.yaml) into the controller before any mode switch.

    A power-cycled controller boots with its own limits, and the follower's
    carriage rests on its stop at ~-4.6 mm, past the boot -4 mm limit: the
    controller idled motor 6 at connect and every landing attempt failed with
    "Robot input with modes different than configured modes" (2026-09-01).
    The golden carries the -6 mm limit the sessions run with.
    """
    try:
        from lerobot_robot_tatbot import goldens
        # callers name the arm "follower" or "follower@<address>"; the golden is per role
        role = name.split("@", 1)[0].strip().lower()
        if role not in ("follower", "leader"):
            logger.warning("%s landing: no golden for role %r (expected follower|leader)", name, role)
            return []
        path = goldens.config_dir() / f"{role}.yaml"
        if not path.exists():
            logger.warning("%s landing: golden %s missing; controller keeps its boot limits", name, path)
            return []
        return goldens.apply_arm_golden(driver, trossen_arm, path)
    except Exception as exc:  # a landing must never die on its own config
        logger.warning("%s golden not applied at landing: %s", name, exc)
        return []


def land_arm(ip: str, end_effector, staged_positions, *, name: str = "arm",
             gripper_index: int = 6, estop=None, attempts: int = 3,
             verify: bool = True) -> bool:
    """Bring an arm safely down: fresh session, clear error, home, sleep, idle.

    One implementation for both arms and every caller (plugin disconnect
    fallbacks, the standalone tune session, the cockpit's Recover). Uses a
    FRESH driver rather than the caller's: when a session dies mid-run its
    driver object can no longer command anything, but the controller itself
    usually still accepts a new connection.

    The gripper is held at its measured position throughout, so a gripped
    tool (the tattoo pen) is never ground against the position-mode effort
    saturation. Ctrl+C is shielded for the duration. The landing is verified
    rather than assumed: the arm joints must actually reach the sleep pose,
    otherwise this reports failure instead of claiming success.
    """
    if estop is not None and getattr(estop, "engaged", False):
        logger.error(
            "%s landing refused: e-stop is engaged (%s) — release it first, "
            "the arm must never be driven while latched",
            name, getattr(getattr(estop, "state", None), "value", "engaged"),
        )
        return False

    deadline = time.monotonic() + LANDING_DEADLINE_S
    with SigintShield():
        for attempt in range(1, attempts + 1):
            driver = trossen_arm.TrossenArmDriver()
            try:
                logger.warning(
                    "%s landing: attempt %d/%d over a fresh driver session",
                    name, attempt, attempts,
                )
                driver.configure(
                    trossen_arm.Model.wxai_v0, end_effector, ip, True,
                    CONFIGURE_TIMEOUT_S,
                )
                err = controller_error(driver)
                if err:
                    logger.warning("%s firmware error at landing: %s", name, err)
                applied = _apply_golden(driver, name)
                if applied:
                    logger.warning("%s landing: golden applied (%s)", name, ", ".join(applied))
                if err and applied:
                    # The fault was judged against the boot limits. The SDK clears
                    # errors only at configure, so reconnect once now that the
                    # golden limits are in the controller.
                    with contextlib.suppress(Exception):
                        driver.cleanup()
                    driver = trossen_arm.TrossenArmDriver()
                    driver.configure(
                        trossen_arm.Model.wxai_v0, end_effector, ip, True,
                        CONFIGURE_TIMEOUT_S,
                    )
                    err = controller_error(driver)
                    if err:
                        logger.warning("%s fault persists after the golden: %s", name, err)
                    else:
                        logger.warning("%s fault cleared after applying the golden limits", name)

                positions = list(driver.get_all_positions())
                staged = list(staged_positions)
                # The carriage goes to its staged (rest) value in every phase:
                # nothing is gripped (2026-08-30), and a retract left over from a
                # trip must not survive into the next session.
                sleep_pose = sleep_pose_for(staged, gripper_index)

                driver.set_all_modes(trossen_arm.Mode.position)
                live = [{"name": name, "driver": driver}]
                # Every phase is non-blocking and heartbeat-monitored. If the
                # e-stop engages mid-interpolation, override the target with
                # the measured pose, wait, then continue from there.
                for target, seconds in (
                    (positions, TAKEOVER_S),
                    (staged, STAGED_POSE_S),
                    (sleep_pose, SLEEP_POSE_S),
                ):
                    _run_monitored_phase(live, [target], seconds, estop)
                driver.set_all_modes(trossen_arm.Mode.idle)

                landed = True
                if verify:
                    final = list(driver.get_all_positions())
                    worst = max(
                        abs(f - s) for i, (f, s) in enumerate(zip(final, sleep_pose, strict=True))
                        if i != gripper_index
                    )
                    landed = worst <= LANDED_TOLERANCE_RAD
                    if not landed:
                        logger.error(
                            "%s landing did NOT reach the sleep pose (worst "
                            "joint off by %.2f rad) — the controller accepted "
                            "the commands but did not execute them",
                            name, worst,
                        )
                driver.cleanup()
                if landed:
                    logger.warning(
                        "%s landing complete: sleep pose, motors idle "
                        "(carriage at rest %.4f)", name, sleep_pose[gripper_index],
                    )
                    return True
            except KeyboardInterrupt:
                logger.error("%s landing interrupted — arm state unknown", name)
                with contextlib.suppress(Exception):
                    driver.cleanup()
                return False
            except Exception as e:
                logger.error("%s landing attempt %d failed: %s", name, attempt, e)
                with contextlib.suppress(Exception):
                    driver.cleanup()
            if time.monotonic() > deadline:
                logger.error("%s landing exceeded its %.0f s budget", name,
                             LANDING_DEADLINE_S)
                break
            if attempt < attempts:
                time.sleep(RETRY_DELAY_S)
    logger.error(
        "%s landing FAILED — the arm may still be holding position. "
        "Power-cycle it and run scripts/il_recover_arm.sh", name,
    )
    return False


def _freeze_live_arms(live) -> None:
    """Override any interpolation with a measured-pose position hold."""
    for arm in list(live):
        try:
            present = list(arm["driver"].get_all_positions())
            arm["driver"].set_all_modes(trossen_arm.Mode.position)
            arm["driver"].set_all_positions(present, 0.0, False)
        except Exception as exc:
            logger.error("%s e-stop hold failed: %s", arm["name"], exc)
            live.remove(arm)


def _run_monitored_phase(live, targets, seconds: float, estop=None) -> None:
    """Run one interpolation phase, pausing at the measured pose on e-stop."""
    target_by_name = {
        arm["name"]: target for arm, target in zip(live, targets, strict=True)
    }
    while True:
        if estop is not None and getattr(estop, "engaged", False):
            logger.warning(
                "E-STOP engaged (%s): lifecycle motion frozen",
                getattr(getattr(estop, "state", None), "value", "engaged"),
            )
            _freeze_live_arms(live)
            if not live:
                raise RuntimeError("no arm could enter the e-stop hold")
            while getattr(estop, "engaged", False):
                time.sleep(ESTOP_POLL_S)
            logger.warning("E-stop released: resuming lifecycle motion")

        for arm in list(live):
            try:
                arm["driver"].set_all_positions(
                    target_by_name[arm["name"]], seconds, False
                )
            except Exception as exc:
                logger.error("%s lifecycle move failed: %s", arm["name"], exc)
                live.remove(arm)
        if not live:
            raise RuntimeError("no arm remains controllable")
        if estop is None:
            # Explicit hardware-free bench/legacy callers retain the old
            # single sleep (and the existing fake-driver tests can collapse
            # it). Production callers always pass the shared monitor.
            time.sleep(seconds + 0.15)
            return
        started = time.monotonic()
        while time.monotonic() - started < seconds + 0.15:
            if estop is not None and getattr(estop, "engaged", False):
                break
            time.sleep(ESTOP_POLL_S)
        else:
            # Close the edge between the final poll and returning to the
            # caller (which may idle the arms after a landing).
            if not getattr(estop, "engaged", False):
                return


def land_arms_together(arms, verify: bool = True, estop=None) -> bool:
    """Land several arms SIMULTANEOUSLY over their existing driver sessions.

    ``arms``: a list of (name, driver, staged_positions, gripper_index).

    Threads cannot do this: the trossen_arm binding holds the GIL across its
    blocking calls (measured 2026-08-20 — two 3 s blocking calls in threads
    took 6.2 s wall, zero overlap), and constructing two drivers at once
    races on the SDK's logger name. So instead every move is issued
    NON-blocking to all arms in turn — each call just posts a UDP command and
    returns — and then a single sleep covers the whole fleet. Same total time
    as landing one arm, no threads, no GIL, no driver-lifecycle races.

    Each arm's gripper is held at its measured position throughout, so a
    gripped tool is never ground. Arms whose driver fails are reported and
    skipped so one dead session cannot strand the others.
    """
    live = []
    for name, driver, staged, grip_idx in arms:
        try:
            positions = list(driver.get_all_positions())
            staged_pose = list(staged)
            sleep_pose = sleep_pose_for(staged_pose, grip_idx)
            driver.set_all_modes(trossen_arm.Mode.position)
            live.append({
                "name": name, "driver": driver, "grip_idx": grip_idx,
                "start": positions, "staged": staged_pose, "sleep": sleep_pose,
            })
        except Exception as e:
            logger.error("%s cannot be landed over its session (%s)", name, e)
    if not live:
        return False

    logger.warning(
        "landing %s together — release the arms",
        " and ".join(a["name"] for a in live),
    )
    # Each phase: post the command to every arm (returns immediately), then
    # one sleep for the whole fleet.
    for pose_key, seconds in (("start", TAKEOVER_S),
                              ("staged", STAGED_POSE_S),
                              ("sleep", SLEEP_POSE_S)):
        try:
            _run_monitored_phase(
                live, [arm[pose_key] for arm in live], seconds, estop)
        except Exception as e:
            logger.error("landing move failed: %s", e)
            return False

    ok = True
    for arm in live:
        try:
            arm["driver"].set_all_modes(trossen_arm.Mode.idle)
            if verify:
                final = list(arm["driver"].get_all_positions())
                worst = max(
                    abs(f - s) for i, (f, s)
                    in enumerate(zip(final, arm["sleep"], strict=True))
                    if i != arm["grip_idx"]
                )
                if worst > LANDED_TOLERANCE_RAD:
                    logger.error(
                        "%s did NOT reach the sleep pose (worst joint off by "
                        "%.2f rad)", arm["name"], worst,
                    )
                    ok = False
                    continue
            logger.warning("%s landed: sleep pose, motors idle", arm["name"])
        except Exception as e:
            logger.error("%s failed to idle after landing: %s", arm["name"], e)
            ok = False
    return ok


def raise_arms_together(arms, goal_time: float = STAGED_POSE_S,
                        verify: bool = True, estop=None) -> bool:
    """Bring several arms to their staged poses SIMULTANEOUSLY.

    The startup mirror image of land_arms_together, and parallel for the same
    reason: the driver binding holds the GIL across blocking calls, so the
    move is posted NON-blocking to every arm and one wait covers them all.
    Without this, each plugin's connect() blocks through its own staged move
    and you watch the arms lift one after another on every session.

    ``arms``: (name, driver, staged_positions, gripper_index). Each arm's
    gripper is held at its measured position rather than driven to the staged
    value, so a tool already in the gripper is not squeezed in position mode.
    """
    live = []
    for name, driver, staged, grip_idx in arms:
        try:
            # the staged carriage value (rest) is part of the staged pose now
            target = list(staged)
            driver.set_all_modes(trossen_arm.Mode.position)
            live.append({"name": name, "driver": driver, "target": target,
                         "grip_idx": grip_idx})
        except Exception as e:
            logger.error("%s cannot be staged (%s)", name, e)
            raise
    if not live:
        return False

    logger.info(
        "raising %s together to the staged pose",
        " and ".join(a["name"] for a in live),
    )
    _run_monitored_phase(
        live, [arm["target"] for arm in live], goal_time, estop)

    if not verify:
        return True
    ok = True
    for arm in live:
        final = list(arm["driver"].get_all_positions())
        worst = max(
            abs(f - t) for i, (f, t)
            in enumerate(zip(final, arm["target"], strict=True))
            if i != arm["grip_idx"]
        )
        if worst > STAGED_TOLERANCE_RAD:
            logger.error(
                "%s did not reach the staged pose (worst joint off by %.2f "
                "rad) — the controller is not executing motion commands",
                arm["name"], worst,
            )
            ok = False
    return ok


class ArmGroup:
    """Process-wide set of connected arms, so lifts and landings are shared.

    lerobot connects the robot and the teleoperator one after the other, and
    disconnects them the same way. Each plugin left to itself therefore
    blocks through its own staged move and its own landing, and you watch
    the arms rise and retract one at a time on every session.

    This group lets whichever arm acts first move the whole fleet:
      * every arm registers when it connects with its staged move deferred,
      * the first arm actually used lifts them all together, then each
        finishes its own configuration,
      * the first arm disconnected lands them all together, and the rest
        then only release their hardware.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._members: dict[str, dict] = {}
        self._landed = False

    def register(
        self, name: str, plugin, driver, staged, gripper_index: int, estop=None
    ) -> None:
        with self._lock:
            self._members[name] = {
                "plugin": plugin, "driver": driver,
                "staged": list(staged), "grip": gripper_index, "pending": True,
                "estop": estop,
            }
            self._landed = False

    def unregister(self, name: str) -> None:
        with self._lock:
            self._members.pop(name, None)
            if not self._members:
                self._landed = False

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._members)

    def stage_pending(self) -> bool:
        """Lift every arm still awaiting its staged move — together."""
        with self._lock:
            pending = [(n, m) for n, m in self._members.items() if m["pending"]]
            for _, m in pending:
                m["pending"] = False
        if not pending:
            return True
        estop = next((m["estop"] for _, m in pending if m["estop"] is not None), None)
        ok = raise_arms_together([
            (n, m["driver"], m["staged"], m["grip"]) for n, m in pending
        ], estop=estop)
        for _, m in pending:  # per-arm verification + mode switches
            m["plugin"].finish_staging()
        return ok

    def restage_all(self) -> bool:
        """Re-stage every registered arm together — the safety-pause resume
        path: after a MotionSafetyError froze the arms mid-episode and the
        operator confirmed, lift both back to the staged pose before the
        next attempt. Same verified move as session start."""
        with self._lock:
            for m in self._members.values():
                m["pending"] = True
        return self.stage_pending()

    def land(self) -> bool:
        """Land every registered arm together, once per session."""
        with self._lock:
            if self._landed or not self._members:
                return self._landed
            self._landed = True
            members = list(self._members.items())
        estop = next((m["estop"] for _, m in members if m["estop"] is not None), None)
        return land_arms_together([
            (n, m["driver"], m["staged"], m["grip"]) for n, m in members
        ], estop=estop)

    def has_landed(self) -> bool:
        with self._lock:
            return self._landed


arm_group = ArmGroup()
