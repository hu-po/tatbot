"""The real blue arm and its wrist camera, for the travel runner on the arm node.

``BlueArm`` takes the arm the way every tatbot arm process does, with the
plugin's own modules: the exclusive driver lease, the e-stop monitor before
anything moves (a pressed button refuses the start), the leader's golden
controller config and end-effector preset, a controller-error check; it
raises the arm to its staged pose and lands it back there. Only those
modules are loaded: the plugin package's ``__init__`` pulls its robot classes
and the LeRobot release they are built on, which this environment replaces.

Every call on the session's driver is announced to a watchdog process
(``watchdog.py``), started before the driver is opened: a vendor driver call
can block forever in a TCP read once the controller's TCP server wedges, and
nothing in the process making that read can end it. A call that outlives its
budget ends the runner there, and the watchdog hands the arm to the recovery
landing below.

A landing that fails over the session's driver is handed to the plugin's
recovery landing (``recovery.land_arm``) run in a process of its own,
``python -m tatbot_travel.hardware``, with its own e-stop monitor and lease,
in its own process group, under GNU timeout and the budget of
`tatbot arm recover`. Once the controller's TCP server wedges, a vendor
driver can block forever in any TCP read after it connects (``configure()``'s
own handshake included), and nothing in the process making that read can end
it; another process can end that one.
It is not ``arm_recover`` itself: that reads only a serial e-stop, and the
rig's may be the relay's UDP stream, which the Python monitor reads.

``WristCamera`` opens the blue arm's D405 (serial from visiond's camera table)
in the stream visiond uses -- 640x480 YUYV colour at 30 Hz -- with depth
aligned to colour, and keeps the newest frame on a background thread.
"""

from __future__ import annotations

import contextlib
import fcntl
import importlib
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tomllib

from tatbot_travel import assets, watchdog

ARM_JOINTS = 6
log = logging.getLogger("travel.hardware")

# The recovery landing runs as scripts/il_recover_arm.sh runs arm_recover: GNU timeout ends it after
# recovery.LANDING_DEADLINE_S with SIGTERM, then SIGKILL this much later.
RECOVERY_KILL_AFTER_S = 2.0
# Past timeout's own end the runner stops waiting and ends the recovery's process group itself.
RECOVERY_SLACK_S = 3.0
# The recovery landing's exits (recovery_landing below, arm_recover's codes) and timeout's, as the runner reports them.
RECOVERY_EXITS = {
    1: "the recovery landing failed or did not reach the sleep pose",
    3: "the recovery refused before commanding anything: e-stop engaged or unavailable",
    6: "the recovery could not take the driver lease",
    7: "the recovery refused before commanding anything: a joint measured past its limits (power the "
       "controller off, turn that joint by hand to near 0 and power it on)",
    124: "the recovery landing outlived its budget and was terminated",
    137: "the recovery landing outlived its budget and was killed",
}


def beyond_limits(positions, limits) -> list[str]:
    """The arm joints measured past their limits by more than the controller's own position tolerance.

    Taking over such a joint first snaps it into its limits and then drives it to the staged pose. After a power
    cycle at the staged wrist roll the blue controller once counted joint 5 a full turn low (-4.80 rad where it
    had read +1.49), so that takeover would have turned the wrist, and its camera cable, a full extra turn.
    """
    return [f"joint {i} at {q:+.3f} rad (limits {j.position_min:+.3f}..{j.position_max:+.3f}, "
            f"tolerance {j.position_tolerance:.3f})"
            for i, (q, j) in enumerate(zip(positions, limits, strict=True))
            if not j.position_min - j.position_tolerance <= q <= j.position_max + j.position_tolerance]


def plugin(module: str):
    """One ``lerobot_robot_tatbot`` module from the repo checkout, without the package ``__init__``."""
    if "lerobot_robot_tatbot" not in sys.modules:
        package = types.ModuleType("lerobot_robot_tatbot")
        package.__path__ = [str(assets.repo_root() / "python" / "lerobot_robot_tatbot" / "src"
                                / "lerobot_robot_tatbot")]
        sys.modules["lerobot_robot_tatbot"] = package
    return importlib.import_module(f"lerobot_robot_tatbot.{module}")


class BlueArm:
    """The blue arm's driver session: lease, e-stop, golden config, staging, commands, landing."""

    def __init__(self, estop_required: bool = True):
        import trossen_arm

        # False only for a supervised run whose operator holds another stop (the arm's power rocker):
        # the plugin's own benches opt out the same way, estop_required=False.
        self.estop_required = estop_required

        paths, goldens = plugin("paths"), plugin("goldens")
        self.trossen = trossen_arm
        self.ip = paths.driver_default("leader_ip", "TATBOT_LEADER_IP")
        self.estop_device = paths.driver_default("estop_device", "TATBOT_ESTOP_DEVICE")
        self.golden = goldens.config_dir() / "leader.yaml"
        # The follower's staged arm pose, as the plugin's leader config reads it; no guessed copy.
        staged = goldens.load_tatbot_yaml().get("follower", {}).get("staged_positions") or []
        if len(staged) < ARM_JOINTS:
            raise RuntimeError("config/trossen/tatbot.yaml has no follower.staged_positions for the blue arm")
        self.staged = [float(v) for v in staged[:ARM_JOINTS]]
        self.end_effector = trossen_arm.StandardEndEffector.wxai_v0_leader
        self.lease_path = plugin("driver_lease").HARDWARE_LEASE  # the recovery landing takes it next
        self.lease = self.estop = self.driver = self.watch = self.summary = None
        self.carriage = 0.0

    def connect(self, summary: Path | None = None) -> None:
        """Take the arm: the e-stop, the lease, the watchdog, then the driver, every call of it bounded.

        ``summary`` is where the watchdog records the landing if it has to end this process (the run's summary).
        """
        recovery, goldens = plugin("recovery"), plugin("goldens")
        if not self.ip:
            raise RuntimeError("no blue-arm address in the hardware profile (driver.leader_ip)")
        if self.estop_required:
            self.estop = plugin("estop").acquire_estop(self.estop_device, required=True)
            self.estop.wait_for_initial_state()
            if self.estop.engaged:
                raise RuntimeError(f"e-stop engaged ({self.estop.state.value}): twist-release it and retry")
        else:
            log.warning("running WITHOUT the e-stop monitor: the operator's own stop is the only hardware stop")
        self.lease = plugin("driver_lease").acquire(self.lease_path)
        self.watch, self.summary = watchdog.DriverWatch.start(), None if summary is None else str(summary)
        self._tell_watchdog_how_to_land()
        self.driver = watchdog.WatchedDriver(self.trossen.TrossenArmDriver(), self.watch)
        self.driver.configure(self.trossen.Model.wxai_v0, self.end_effector, self.ip, True)
        goldens.apply_arm_golden(self.driver, self.trossen, self.golden)
        self.driver.set_end_effector(self.end_effector)
        error = recovery.controller_error(self.driver)
        if error:
            raise RuntimeError(f"controller reports {error!r}: power-cycle the arm")
        beyond = beyond_limits(list(self.driver.get_all_positions())[:ARM_JOINTS],
                               self.driver.get_joint_limits()[:ARM_JOINTS])
        if beyond:
            # Nothing was commanded and the motors are idle: close the session so that land() leaves the arm
            # alone. The recovery landing would only refuse the same joint (exit 7), so the watchdog is told
            # first: if the closing wedges, it ends this process and starts nothing.
            self.watch.withdraw(f"the blue arm measures {'; '.join(beyond)}", self.summary)
            self.driver.cleanup()
            self._close_driver()
            raise RuntimeError(f"the blue arm measures {'; '.join(beyond)}: a takeover would snap it into its "
                               "limits and drive it the long way round. A power cycle can leave a joint counted a "
                               "full turn off: switch the controller off, turn that joint by hand to near 0, "
                               "switch it on and retry")
        self.driver.set_all_modes(self.trossen.Mode.position)
        here = list(self.driver.get_all_positions())
        self.driver.set_all_positions(here, recovery.TAKEOVER_S, False)  # pin the target before any move
        self.carriage = here[ARM_JOINTS]  # the pen cradle rides the carriage: it stays where it is
        self._tell_watchdog_how_to_land()
        staged = [*self.staged, self.carriage]
        if not recovery.raise_arms_together([("travel", self.driver, staged, ARM_JOINTS)], estop=self.estop):
            raise RuntimeError("the arm did not reach its staged pose")

    def joint_limits(self, margin: float) -> tuple[np.ndarray, np.ndarray]:
        joints = self.driver.get_joint_limits()[:ARM_JOINTS]
        return (np.array([j.position_min for j in joints]) + margin,
                np.array([j.position_max for j in joints]) - margin)

    def measured(self) -> np.ndarray:
        return np.asarray(self.driver.get_all_positions(), dtype=float)[:ARM_JOINTS]

    @property
    def estopped(self) -> bool:
        return bool(self.estop is not None and self.estop.engaged)

    def command(self, q: np.ndarray, goal_time: float) -> None:
        self.driver.set_all_positions([*map(float, q), self.carriage], goal_time, False)

    def freeze(self) -> np.ndarray:
        """Hold exactly where the arm stands (an e-stop, a lost camera, skin too close)."""
        q = self.measured()
        self.driver.set_all_positions([*map(float, q), self.carriage], 0.0, False)
        return q

    def land(self) -> str | None:
        """Back to the staged pose and idle, or the recovery landing if that fails; never while e-stopped.

        What became of the arm: ``None`` when no session was taken, ``"landed"`` (staged and idled over this
        session), ``"recovered"`` (the recovery landing measured the sleep pose, motors idle), else ``"unknown"``.
        """
        recovery, estop = plugin("recovery"), plugin("estop")
        try:
            if self.driver is None:
                return None  # refused before taking the arm (e.g. e-stop pressed at start): nothing to land
            while self.estopped:
                time.sleep(recovery.ESTOP_POLL_S)
            try:
                self.driver.set_all_positions([*self.staged, self.carriage], 4.0, True)
                self.driver.set_all_modes(self.trossen.Mode.idle)
                self.driver.cleanup()
                return "landed"
            except Exception:
                log.exception("staged landing failed; landing the recovery way")
            try:
                self.driver.cleanup()
            except Exception:
                log.exception("driver cleanup failed")
            self._close_driver()  # this session's connection is gone before the recovery opens its own
            while self.estopped:
                time.sleep(recovery.ESTOP_POLL_S)
            self._release(estop)  # the recovery opens the e-stop and takes the lease for itself
            return self._recover(recovery)
        finally:
            self._close_driver()
            self._release(estop)

    def _close_driver(self) -> None:
        """Drop the session's driver inside a bounded call, then let the watchdog go: the vendor's destructor runs
        cleanup(), a TCP round trip unless cleanup() already ran."""
        driver, self.driver = self.driver, None
        watch, self.watch = self.watch, None
        if watch is not None:
            with watch.bounded(watchdog.DESTRUCTOR, watchdog.CALL_BUDGET_S):
                del driver  # the last reference to the session: its destructor runs here
            watch.close()

    def _release(self, estop) -> None:
        """The e-stop reader first, the driver lease last: a free lease says the arm and its e-stop are free.
        The monitor holds a share of the lease too, so the lock goes free only once both are closed."""
        estop.release_estop(self.estop)
        self.estop = None
        if self.lease is not None:
            self.lease.close()
            self.lease = None

    def _recover(self, recovery) -> str:
        """:func:`hand_over` this arm. Ctrl+C is shielded as the landing's own is: a press 10 s after the first ends
        it."""
        budget = recovery.LANDING_DEADLINE_S
        return hand_over(lambda: self._recovery_command(budget), budget, recovery.SigintShield())

    def _recovery_command(self, budget_s: float) -> list[str]:
        """:func:`recovery_command` for this arm and e-stop."""
        return recovery_command(**self._recovery(), budget_s=budget_s)

    def _recovery(self) -> dict:
        """This arm, its e-stop and lease, and this run's staged pose: the carriage lands where the run found it
        (the pen cradle rides it), not at the golden's rest."""
        return {"ip": self.ip, "pose": [*self.staged, self.carriage],
                "estop": self.estop_device if self.estop_required else None, "lease": str(self.lease_path)}

    def _tell_watchdog_how_to_land(self) -> None:
        """The recovery landing as :meth:`_recover` would start it now, for the watchdog to start if it has to end
        this process, and where the run's summary goes."""
        self.watch.handover(**self._recovery(), budget_s=plugin("recovery").LANDING_DEADLINE_S,
                            summary=self.summary)


def recovery_command(ip: str, pose: list[float], estop: str | None, lease: str, budget_s: float) -> list[str]:
    """:func:`recovery_landing` in this interpreter under GNU timeout: the arm at ``ip`` lands at ``pose`` (six joints,
    then the carriage) under ``estop`` (None: a run without one). Refused while ``lease`` is held."""
    timeout = shutil.which("timeout")
    if timeout is None:
        raise RuntimeError("GNU timeout is missing")
    if not lease_free(Path(lease)):
        raise RuntimeError(f"{lease} is still held")
    staged = ",".join(repr(float(v)) for v in pose)
    return [timeout, "--signal=TERM", f"--kill-after={RECOVERY_KILL_AFTER_S:g}s", f"{budget_s:g}s", sys.executable,
            "-m", "tatbot_travel.hardware", ip, "--staged", staged, *(["--estop", estop] if estop else [])]


def hand_over(command: Callable[[], list[str]], budget_s: float, shield=None) -> str:
    """The recovery landing ``command()`` builds, in a process group of its own, waited for within a hard budget.

    ``"recovered"`` only when it measured the sleep pose; expiry, a refusal, a failure or an interrupt is
    ``"unknown"``. ``shield`` guards the wait (the runner's shields Ctrl+C).
    """
    try:
        argv = command()
        log.warning("handing the arm to the recovery landing (%.0f s budget): %s", budget_s, shlex.join(argv))
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, process_group=0)
    except (RuntimeError, OSError) as refused:
        return unknown(f"the arm was not handed to the recovery landing: {refused}")
    try:
        with shield or contextlib.nullcontext():
            code = child.wait(timeout=budget_s + RECOVERY_KILL_AFTER_S + RECOVERY_SLACK_S)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as stopped:
        _end(child)
        why = "was interrupted" if isinstance(stopped, KeyboardInterrupt) else "outlived its budget"
        return unknown(f"the recovery landing {why}; its processes were terminated")
    if code != 0:
        return unknown(RECOVERY_EXITS.get(code, f"the recovery landing exited with {code}"))
    log.warning("the recovery landing measured the sleep pose, motors idle")
    return "recovered"


def recovery_landing(argv: list[str] | None = None) -> int:
    """The recovery landing's own process: ``recovery.land_arm`` over a fresh driver session for the blue arm,
    under an e-stop monitor and a driver lease of this process's own.

    Exits as arm_recover does: 0 at the sleep pose, measured; 1 not (the arm state is unknown); 3 the e-stop
    engaged or unavailable before any command; 6 the driver lease held elsewhere; 7 a joint measured past its
    limits, refused before any command.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m tatbot_travel.hardware", description=recovery_landing.__doc__)
    parser.add_argument("ip", help="the blue arm's controller")
    parser.add_argument("--staged", required=True, help="six joints and the carriage, comma-separated")
    parser.add_argument("--estop", help="the e-stop device; left out only for a run without one")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    driver_lease, estop, recovery = plugin("driver_lease"), plugin("estop"), plugin("recovery")
    try:
        monitor = estop.acquire_estop(args.estop, required=True) if args.estop else None
    except driver_lease.DriverBusyError as busy:
        return _refuse(6, busy)
    except RuntimeError as unavailable:
        return _refuse(3, unavailable)
    try:
        if monitor is not None and monitor.wait_for_initial_state() is not estop.EstopState.OK:
            return _refuse(3, f"e-stop engaged ({monitor.state.value})")
        staged = [float(v) for v in args.staged.split(",")]
        leader = recovery.trossen_arm.StandardEndEffector.wxai_v0_leader
        return 0 if recovery.land_arm(args.ip, leader, staged, name="leader", estop=monitor) else 1
    except driver_lease.DriverBusyError as busy:
        return _refuse(6, busy)
    except recovery.JointBeyondLimitsError as beyond:
        return _refuse(7, beyond)
    finally:
        estop.release_estop(monitor)


def lease_free(path: Path) -> bool:
    """No process holds the driver lease, this one included: an exclusive flock on a fresh open file description,
    let go at once. The recovery landing takes the lease for itself, so the runner hands over only a free one."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    finally:
        os.close(fd)
    return True


def _end(child: subprocess.Popen) -> None:
    """SIGTERM the recovery's process group (timeout and the landing), SIGKILL it after the grace, and reap it.
    The leader is signalled only while unreaped, so its group id cannot have been reused."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if child.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, sig)
        with contextlib.suppress(subprocess.TimeoutExpired):
            child.wait(timeout=RECOVERY_KILL_AFTER_S)


def _refuse(code: int, why) -> int:
    log.error("recovery landing refused before any command: %s", why)
    return code


def unknown(why: str) -> str:
    log.error("arm state UNKNOWN: %s. No landing was verified: support the arm before any controller power cycle",
              why)
    return "unknown"


def poe_camera_source(name: str) -> tuple[str, str]:
    """A fleet PoE camera's address and the environment variable holding its password, from visiond's table."""
    table = tomllib.loads((assets.repo_root() / "rust" / "visiond" / "config" / "vision.toml").read_text())
    for camera in table.get("cameras", {}).get("poe", []):
        if camera.get("name") == name:
            return str(camera["address"]), str(camera["password_env"])
    raise RuntimeError(f"no PoE camera {name!r} in vision.toml")


def rtsp_url(name: str, subtype: int = 1, env_file: Path | None = None) -> str:
    """The camera's RTSP URL (sub stream by default): the password from the environment, else the fleet's
    ``~/.config/tatbot/cameras.env``. Never log it."""
    from urllib.parse import quote

    address, password_env = poe_camera_source(name)
    password = os.environ.get(password_env)
    env_file = env_file or Path.home() / ".config" / "tatbot" / "cameras.env"
    if password is None and env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, _, value = line.strip().removeprefix("export ").partition("=")
            if key.strip() == password_env:
                password = value.strip().strip("'\"")
    if not password:
        raise RuntimeError(f"{password_env} is not set here (nor in {env_file})")
    return f"rtsp://admin:{quote(password, safe='')}@{address}:554/cam/realmonitor?channel=1&subtype={subtype}"


def left_wrist_serial() -> str:
    """The blue arm's wrist D405 serial, from visiond's camera table."""
    table = tomllib.loads((assets.repo_root() / "rust" / "visiond" / "config" / "vision.toml").read_text())
    cameras = table.get("cameras", {}).get("realsense", [])
    serials = [c["serial"] for c in cameras if c.get("arm") == "left"]
    if len(serials) != 1:
        raise RuntimeError(f"expected one left-arm RealSense in vision.toml, found {serials}")
    return str(serials[0])


class WristCamera:
    """The newest (RGB, depth in metres, monotonic time) from the blue arm's D405, on a background thread."""

    def __init__(self, serial: str | None = None, fps: int = 30):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial or left_wrist_serial())
        config.enable_stream(rs.stream.color, 640, 480, rs.format.yuyv, fps)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.latest: tuple[np.ndarray, np.ndarray, float] | None = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="wrist-d405", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        import cv2

        while not self.stop.is_set():
            try:
                frames = self.align.process(self.pipeline.wait_for_frames(1000))
                colour, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not colour or not depth:
                    continue
                # YUYV arrives as one uint16 per pixel: its two bytes are the Y and the shared U/V sample.
                yuyv = np.asanyarray(colour.get_data()).view(np.uint8).reshape(480, 640, 2)
                rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
                depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
            except RuntimeError:
                continue  # a frame timeout; the caller's staleness check decides what that means
            except Exception:
                log.exception("wrist camera frame dropped")
                continue
            with self.lock:
                self.latest = (rgb, depth_m, time.monotonic())

    def newest(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        with self.lock:
            return self.latest

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)
        self.pipeline.stop()


class SceneCamera:
    """The newest (RGB, monotonic time) from a fleet PoE camera's sub stream (704x480, ~16 fps), over RTSP,
    resized to what policies see (``SCENE_STREAM_SIZE``, the wrist stream's 640x480).

    A third-person view like the SO-101 package's scene camera. The camera also serves visiond; RTSP takes a
    second client. Frames are read continuously so the newest is never behind the network buffer.
    """

    def __init__(self, name: str = "camera5", subtype: int = 1):
        import cv2

        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")
        self.name = name
        self.capture = cv2.VideoCapture(rtsp_url(name, subtype), cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open {name}'s stream")
        self.latest: tuple[np.ndarray, float] | None = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name=f"scene-{name}", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        import cv2

        from tatbot_travel.camera import SCENE_STREAM_SIZE

        while not self.stop.is_set():
            ok, bgr = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if (rgb.shape[1], rgb.shape[0]) != SCENE_STREAM_SIZE:
                rgb = cv2.resize(rgb, SCENE_STREAM_SIZE, interpolation=cv2.INTER_AREA)
            with self.lock:
                self.latest = (rgb, time.monotonic())

    def newest(self) -> tuple[np.ndarray, float] | None:
        with self.lock:
            return self.latest

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)
        self.capture.release()


if __name__ == "__main__":
    sys.exit(recovery_landing())
