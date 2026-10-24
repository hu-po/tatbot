#!/usr/bin/env python3
"""Host load for the fleet viewer: CPU %, memory %, the hottest CPU/SoC zone and, on
the station, GPU % and GPU temperature, at 1 Hz; beside them the hosts this node reads
for others and, on the ros node, the robot.

What `tatbot-sysmon@<user>` runs (`scripts/viewer.sh sysmon`). Scalars go to
`system/<entity>/{cpu_percent,mem_percent,temp_c,gpu_percent,gpu_temp_c}` in the
standing recording. A host's entity is the first of SYSMON_ROLES it holds in
config/nodes.json, never its hostname; its legend is its `platform` there ("thor"),
else that role. Two more kinds of source ride along:

- A host with no checkout whose `load_via` names a role this node holds (the palette
  Pi, read by the station) is read over one ssh session: its /proc/stat line and
  thermal zones every REMOTE_PERIOD_S by shell builtins (REMOTE_LOOP), so it runs
  nothing but `sleep`, and nothing is installed or written there.
- On the ros node, `system/robot/`: the hottest motor of the arm the ROS stack drives
  (the controller's rotor temperatures in the newest flight-log record), read from
  the newest ros-stack run's files, never from the stack's graph or its control loop.
  TEMP only: the controller process's CPU is % of one core, and on the hosts' CPU
  axis (% of all cores) it read as more load than it is (owner, 2026-10-20).

Display only: it reads /proc, the kernel's thermal zones, the GPU's NVML counters and
those files, and touches nothing else.

A hub the SDK loses ends the process (exit 1), and systemd restarts it after the
unit's RestartSec, which redials and logs the static names again. The port is
no witness: rerun 0.36 never redials a severed stream, and a restarted hub
answers again within seconds, so the SDK's own flush is asked instead.
A node with no reporting role exits 3, which the unit does not restart.
Stdlib, plus rerun once it streams (imported by tatbot_rerun.start).
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/tatbot_contracts/src"))
import tatbot_rerun as tr  # noqa: E402
import tatbot_runlog  # noqa: E402
from tatbot_cli import nodes  # noqa: E402

SYSMON_ROLES = ("station", "ros", "estop-relay")
"""The entity name for a host: the first of these roles it holds."""
GPU_ROLES = ("station",)
"""Where gpu_percent and gpu_temp_c are logged (the station renders the viewer)."""
ROBOT = "robot"
ROBOT_ROLES = ("ros",)
"""Where system/robot/ is logged: the node whose run logs hold the ROS stack's runs."""
COLORS = {"estop-relay": (217, 89, 38), "station": (25, 158, 112), "ros": (57, 135, 229), ROBOT: (232, 123, 164)}
"""One colour per source in every pane (orange, green, blue, pink), so one legend keys the CPU and TEMP panes."""
LEGEND = {"mem_percent": "mem", "gpu_percent": "gpu", "gpu_temp_c": "gpu"}
"""What follows the label for the series the Demo panes do not show: CPU % and TEMP °C name their unit,
so cpu_percent and temp_c carry the label alone ("thor", "robot"), and the rest say what they are ("spark gpu")."""
EXIT_NO_ROLE = 3
PERIOD_S = 1.0
HUB_CHECK_S = 5.0
HUB_TIMEOUT_S = 2.0
NVML = "libnvidia-ml.so.1"
NVML_TEMPERATURE_GPU = 0
THERMAL = "/sys/class/thermal"

REMOTE_PERIOD_S = 2
REMOTE_READS = 1800
"""Readings per ssh session (an hour): a session whose peer vanished unnoticed ends by itself."""
REMOTE_STALE_S = 3.0 * REMOTE_PERIOD_S
REMOTE_RETRY_S = 30.0
REMOTE_LOOP = (f'i=0; while [ "$i" -lt {REMOTE_READS} ]; do i=$((i+1)); read -r l </proc/stat; echo "$l"; '
               'for z in /sys/class/thermal/thermal_zone*/temp; do read -r t 2>/dev/null <"$z" && echo "t $t"; done; '
               f'echo .; sleep {REMOTE_PERIOD_S} || break; done')
"""What a remote host runs: shell builtins, and one `sleep` a reading (without which it stops, never spins)."""
SSH = ("ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
       "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3")

FLIGHT_VERSION = b"2"
ROTOR_TEMP_AT = 8 + 4 * 7 * 7
"""Where a version-2 flight record keeps its seven float32 rotor temperatures: after `t` (float64) and seven
7-float arrays (ros/README.md "Flight log")."""
STALE_S = 5.0
"""A flight log not written for this long belongs to a stack that stopped."""


# --- readings (pure: they take the text the kernel wrote) ---------------------------


def cpu_times(stat: str) -> tuple[int, int]:
    """(busy, total) jiffies from /proc/stat's aggregate `cpu` line. iowait is idle
    time; guest time is already inside user and nice, so only the first eight count."""
    fields = [int(x) for x in stat.split("\n", 1)[0].split()[1:9]]
    idle = sum(fields[3:5])
    total = sum(fields)
    return total - idle, total


def cpu_percent(before: tuple[int, int], after: tuple[int, int]) -> float | None:
    """Busy share of every core between two samples, 0..100."""
    busy, total = after[0] - before[0], after[1] - before[1]
    return 100.0 * busy / total if total > 0 else None


def mem_percent(meminfo: str) -> float | None:
    """Memory in use, 0..100: what MemAvailable says cannot be had without swapping."""
    kib = {}
    for line in meminfo.splitlines():
        key, _, value = line.partition(":")
        if value.split():
            kib[key] = int(value.split()[0])
    total, available = kib.get("MemTotal"), kib.get("MemAvailable")
    if not total or available is None:
        return None
    return 100.0 * (total - available) / total


def _sysfs(path: Path) -> str:
    """A sysfs value, or "" when it cannot be read: a power-gated sensor (the Thor's
    gpu-thermal) answers EAGAIN, which a text-mode open() turns into a TypeError."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return ""
    try:
        return os.read(fd, 64).decode(errors="replace").strip()
    except OSError:
        return ""
    finally:
        os.close(fd)


def soc_zones(root: str = THERMAL) -> list[Path]:
    """The temp files of every thermal zone but the GPU's. The Thor names its zones
    tj/cpu/gpu/soc*-thermal (tj is the SoC junction, the hottest of all); the GB10's are
    ACPI `acpitz` zones whose device path names them (TSOC, TS0E, ..., TGPU, TUNC). The
    GPU is left out because the station reports its own temperature from NVML."""
    return [z / "temp" for z in Path(root).glob("thermal_zone*")
            if "gpu" not in (_sysfs(z / "type") + _sysfs(z / "device" / "path")).lower()]


def hottest_c(temps: list[Path]) -> float | None:
    """The hottest of these zones now, deg C (they report millidegrees); None when none reads."""
    readings = [int(raw) / 1000 for raw in map(_sysfs, temps) if raw.lstrip("-").isdigit()]
    return max(readings, default=None)


def remote_readings(lines):
    """(cpu_times or None, hottest zone deg C or None) for each reading REMOTE_LOOP writes."""
    stat, temps = None, []
    for line in lines:
        if line.startswith("cpu "):
            try:
                stat = cpu_times(line)
            except ValueError:
                stat = None
        elif line.startswith("t ") and line[2:].strip().lstrip("-").isdigit():
            temps.append(int(line[2:]) / 1000)
        elif line.strip() == ".":
            yield stat, max(temps, default=None)
            stat, temps = None, []


def newest_rotor_temps(path: Path) -> list[float] | None:
    """The rotor temperatures (deg C) in a flight log's newest whole record, or None without one."""
    try:
        with open(path, "rb") as fh:
            header = fh.readline(256)
            words = header.split()
            if len(words) < 3 or words[:2] != [b"tatbot-flight", FLIGHT_VERSION]:
                return None
            size = int(words[-1])
            records = (os.fstat(fh.fileno()).st_size - len(header)) // size
            if records < 1 or size < ROTOR_TEMP_AT + 28:
                return None
            fh.seek(len(header) + (records - 1) * size)
            record = fh.read(size)
    except (OSError, ValueError):
        return None
    return list(struct.unpack_from("<7f", record, ROTOR_TEMP_AT)) if len(record) == size else None


def hottest_motor_c(run: Path, now: float) -> float | None:
    """The hottest motor of every arm whose flight log in `run` is still being written. None once the
    stack stopped, and from a mock arm, which writes zeros."""
    temps = []
    for log in run.glob("*-flight.bin"):
        try:
            if now - log.stat().st_mtime > STALE_S:
                continue
        except OSError:
            continue
        temps += [t for t in newest_rotor_temps(log) or [] if math.isfinite(t) and t > 0]
    return max(temps, default=None)


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class Gpu:
    """GPU utilisation % and temperature, from NVML (what nvidia-smi reads, here
    without a process per sample; the GB10 reports utilisation, not memory).
    `present` is False where NVML does not open."""

    def __init__(self, nvml: str = NVML):
        self._lib, self._handle = self._open_nvml(nvml)

    @staticmethod
    def _open_nvml(name: str):
        try:
            lib = ctypes.CDLL(name)
            handle = ctypes.c_void_p()
            if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0:
                return lib, handle
        except (OSError, AttributeError):
            pass
        return None, None

    @property
    def present(self) -> bool:
        return self._lib is not None

    def percent(self) -> float | None:
        if self._lib is None:
            return None
        util = _Utilization()
        ok = self._lib.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(util)) == 0
        return float(util.gpu) if ok else None

    def temp_c(self) -> float | None:
        if self._lib is None:
            return None
        temp = ctypes.c_uint()
        ok = self._lib.nvmlDeviceGetTemperature(self._handle, NVML_TEMPERATURE_GPU, ctypes.byref(temp)) == 0
        return float(temp.value) if ok else None


# --- what it reports ------------------------------------------------------------------


def report_role(nmap: dict, node: str) -> str | None:
    """The role naming this host's entities, or None when it holds none of SYSMON_ROLES."""
    roles = nodes.roles_of(nmap, node)
    return next((role for role in SYSMON_ROLES if role in roles), None)


def label(nmap: dict, node: str, entity: str) -> str:
    """A source's legend: its host's `platform` in config/nodes.json ("thor"), else its entity."""
    return str(nmap.get(node, {}).get("platform") or entity)


def remote_hosts(nmap: dict, node: str) -> list[str]:
    """The hosts this node reads for others: each whose `load_via` names a role it holds."""
    held = set(nodes.roles_of(nmap, node))
    return [host for host, rec in nmap.items()
            if rec.get("load_via") in held and report_role(nmap, host) and nodes.ssh_target(nmap, host)]


def hub_address(url: str) -> tuple[str, int]:
    """(host, port) of a `rerun+http://HOST:PORT/proxy` URL."""
    parts = urlsplit(url)
    return parts.hostname or "127.0.0.1", parts.port or tr.PORT


def hub_answers(address: tuple[str, int], timeout: float = HUB_TIMEOUT_S) -> bool:
    try:
        with socket.create_connection(address, timeout=timeout):
            return True
    except OSError:
        return False


# --- the sources -----------------------------------------------------------------------


def _read(path: str) -> str:
    with open(path) as fh:
        return fh.read()


class Sampler:
    """This host, one reading per call: CPU over the time since the last call; memory, the
    hottest CPU/SoC zone and the GPU now. A series is named only where it can be read."""

    def __init__(self, gpu: Gpu | None, proc: str = "/proc", thermal: str = THERMAL):
        self.gpu, self.proc = gpu, proc
        self.zones = soc_zones(thermal)
        self.names = (["cpu_percent", "mem_percent"] + (["temp_c"] if self.zones else [])
                      + (["gpu_percent", "gpu_temp_c"] if gpu is not None else []))
        self._cpu = cpu_times(_read(f"{proc}/stat"))

    def sample(self) -> dict[str, float | None]:
        now = cpu_times(_read(f"{self.proc}/stat"))
        values = {"cpu_percent": cpu_percent(self._cpu, now),
                  "mem_percent": mem_percent(_read(f"{self.proc}/meminfo"))}
        self._cpu = now
        if self.zones:
            values["temp_c"] = hottest_c(self.zones)
        if self.gpu is not None:
            values["gpu_percent"] = self.gpu.percent()
            values["gpu_temp_c"] = self.gpu.temp_c()
        return values


class Remote:
    """A host read over one ssh session that runs REMOTE_LOOP: its CPU over each period and its
    hottest zone, each reading once; nothing once the newest is older than REMOTE_STALE_S. A
    session that ends is opened again REMOTE_RETRY_S after the last attempt."""

    names = ["cpu_percent", "temp_c"]

    def __init__(self, command: list[str]):
        self.command = command
        self._proc: subprocess.Popen | None = None
        self._retry_at = 0.0
        self._latest: tuple | None = None
        self._used: tuple | None = None

    def _pump(self, lines) -> None:
        for stat, hottest in remote_readings(lines):
            self._latest = (time.monotonic(), stat, hottest)

    def _open(self, now: float) -> None:
        self._retry_at = now + REMOTE_RETRY_S
        try:
            self._proc = subprocess.Popen(self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
        except OSError as exc:
            print(f"sysmon: cannot start {self.command[0]}: {exc}", file=sys.stderr)
            self._proc = None
            return
        threading.Thread(target=self._pump, args=(self._proc.stdout,), daemon=True).start()

    def sample(self) -> dict[str, float | None]:
        now = time.monotonic()
        if (self._proc is None or self._proc.poll() is not None) and now >= self._retry_at:
            self._open(now)
        latest, before = self._latest, self._used
        if latest is None or latest is before or now - latest[0] > REMOTE_STALE_S:
            return {}
        self._used = latest
        follows = before is not None and latest[0] - before[0] <= REMOTE_STALE_S and before[1] and latest[1]
        return {"cpu_percent": cpu_percent(before[1], latest[1]) if follows else None, "temp_c": latest[2]}

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()


class Robot:
    """system/robot/: the hottest motor of the arm the ROS stack drives, from its newest run's
    files (`<runs>/latest`)."""

    names = ["temp_c"]

    def __init__(self, runs: Path):
        self.runs = runs

    def sample(self) -> dict[str, float | None]:
        return {"temp_c": hottest_motor_c(self.runs / "latest", time.time())}


def sources(nmap: dict, node: str, role: str, gpu: Gpu | None) -> list[tuple[str, str, object]]:
    """(entity, legend label, sampler) for this host, the hosts it reads for others, and the robot."""
    found = [(role, label(nmap, node, role), Sampler(gpu))]
    for host in remote_hosts(nmap, node):
        entity = report_role(nmap, host)
        found.append((entity, label(nmap, host, entity), Remote([*SSH, nodes.ssh_target(nmap, host), REMOTE_LOOP])))
    if role in ROBOT_ROLES:
        found.append((ROBOT, ROBOT, Robot(tatbot_runlog.log_root() / "ros-stack")))
    return found


# --- the stream ----------------------------------------------------------------------


def log_series_names(rr, entity: str, label: str, names: list[str]) -> None:
    """Legend names ("thor", "spark gpu") and each source's one colour, once, static."""
    color = [COLORS[entity]] if entity in COLORS else None
    for name in names:
        legend = f"{label} {LEGEND[name]}" if name in LEGEND else label
        rr.log(f"system/{entity}/{name}", rr.SeriesLines(names=legend, colors=color), static=True)


def log_values(rr, readings: list[tuple[str, dict[str, float | None]]]) -> None:
    tr.set_capture_time(time.time_ns())
    for entity, values in readings:
        for name, value in values.items():
            if value is not None:
                rr.log(f"system/{entity}/{name}", rr.Scalars(value))


def hub_lost(rr) -> str | None:
    """Why the SDK can no longer deliver to the hub, or None while it can. A stream to a
    hub that went away stays severed even once a new hub listens on the port."""
    try:
        rr.get_global_data_recording().flush(timeout_sec=HUB_TIMEOUT_S)
    except RuntimeError as exc:
        return str(exc)
    return None


def stream(rr, found: list[tuple[str, str, object]], watch_hub: bool, duration_s: float) -> int:
    """Sample at PERIOD_S until `duration_s` (0 = until stopped); 1 when the hub is lost."""
    start = next_tick = next_hub = time.monotonic()
    while duration_s <= 0 or time.monotonic() - start < duration_s:
        # Never catch up a backlog after a suspend or a stall: one sample per period.
        next_tick = max(next_tick + PERIOD_S, time.monotonic())
        time.sleep(max(0.0, next_tick - time.monotonic()))
        log_values(rr, [(entity, sampler.sample()) for entity, _, sampler in found])
        if watch_hub and time.monotonic() >= next_hub:
            lost = hub_lost(rr)
            if lost:
                print(f"sysmon: lost the viewer hub ({lost}); exiting for a restart", file=sys.stderr)
                return 1
            next_hub = time.monotonic() + HUB_CHECK_S
    return 0


def parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tr.add_sink_args(ap)
    ap.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 (default) until stopped")
    ap.add_argument("--print-role", action="store_true",
                    help="print the role this node reports as and exit (3 when it has none)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    nmap = nodes.load(tr.REPO_ROOT)
    node = nodes.this_node(nmap)
    role = report_role(nmap, node)
    if role is None:
        print(f"sysmon: {node} holds none of the roles {', '.join(SYSMON_ROLES)} in config/nodes.json; "
              "nothing to report", file=sys.stderr)
        return EXIT_NO_ROLE
    if args.print_role:
        print(role)
        return 0
    connect = args.connect or (tr.default_connect() if not args.output else None)
    if not connect and not args.output:
        print("sysmon: no sink; pass --connect rerun+http://HOST:PORT/proxy or --output FILE.rrd", file=sys.stderr)
        return 2
    hub = hub_address(connect) if connect else None
    if hub is not None and not hub_answers(hub):
        print(f"sysmon: the viewer hub {hub[0]}:{hub[1]} is not answering", file=sys.stderr)
        return 1
    # One batch a second is all this producer makes: flush once a second, not every 8 ms.
    os.environ.setdefault("RERUN_FLUSH_TICK_SECS", "1")
    rr = tr.start(f"sysmon-{role}", connect=connect, output=args.output,
                  recording_id=args.recording_id or tr.STANDING_RECORDING, view_coordinates=False)
    gpu = Gpu() if role in GPU_ROLES else None
    found = sources(nmap, node, role, gpu if gpu is not None and gpu.present else None)
    for entity, name, sampler in found:
        log_series_names(rr, entity, name, sampler.names)
    try:
        return stream(rr, found, hub is not None, args.duration)
    finally:
        for _, _, sampler in found:
            if isinstance(sampler, Remote):
                sampler.close()


if __name__ == "__main__":
    sys.exit(main())
