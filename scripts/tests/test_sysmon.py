"""Host load for the demo page: what the kernel's counters say, under which role, and where it goes."""

from __future__ import annotations

import os
import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/lib"))

import tatbot_sysmon as sm  # noqa: E402
from tatbot_cli import nodes  # noqa: E402

# /proc/stat and /proc/meminfo as a 20-core aarch64 host writes them.
STAT_A = ("cpu  4705 150 1120 16250 520 0 90 0 30 0\n"
          "cpu0 235 7 56 812 26 0 4 0 0 0\nintr 1 2 3\nctxt 99\n")
STAT_B = ("cpu  4805 150 1170 16950 600 0 100 0 40 0\n"
          "cpu0 240 7 58 846 30 0 5 0 0 0\nintr 1 2 3\nctxt 120\n")
MEMINFO = ("MemTotal:       127535420 kB\nMemFree:         1611044 kB\nMemAvailable:   50331648 kB\n"
           "Buffers:          123456 kB\nHugePages_Total:       0\n")


def test_cpu_is_the_busy_share_of_every_core_with_iowait_idle_and_guest_not_counted_twice():
    before, after = sm.cpu_times(STAT_A), sm.cpu_times(STAT_B)
    # busy: user +100, system +50, softirq +10 = 160; idle +700, iowait +80; guest (+10) is inside user
    assert after[0] - before[0] == 160 and after[1] - before[1] == 940
    assert sm.cpu_percent(before, after) == pytest.approx(100 * 160 / 940)
    assert sm.cpu_percent(after, after) is None


def test_memory_in_use_is_what_memavailable_leaves():
    assert sm.mem_percent(MEMINFO) == pytest.approx(100 * (127535420 - 50331648) / 127535420)
    assert sm.mem_percent("MemTotal: 100 kB\n") is None


def _zones(root: Path, zones: list[tuple[str, str | None, str]]) -> Path:
    """A /sys/class/thermal tree: (type, ACPI device path or None, temp as the kernel writes it)."""
    for i, (kind, acpi, temp) in enumerate(zones):
        zone = root / f"thermal_zone{i}"
        zone.mkdir(parents=True)
        (zone / "type").write_text(kind + "\n")
        (zone / "temp").write_text(temp + "\n")
        if acpi:
            (zone / "device").mkdir()
            (zone / "device" / "path").write_text(acpi + "\n")
    return root


# The ros node (a Jetson Thor): tj is the junction; a power-gated gpu-thermal reads as nothing.
THOR = [("tj-thermal", None, "49531"), ("gpu-thermal", None, "91000"), ("cpu-thermal", None, "49718"),
        ("soc012-thermal", None, "47812"), ("soc345-thermal", None, "48843")]
# The station (a GB10): seven ACPI zones, told apart only by their device path.
GB10 = [("acpitz", "\\_TZ_.TSOC", "57300"), ("acpitz", "\\_TZ_.TS0P", "57400"),
        ("acpitz", "\\_TZ_.TGPU", "71000"), ("acpitz", "\\_TZ_.TUNC", "56000")]


def test_the_hottest_cpu_soc_zone_leaves_out_the_gpu_and_what_cannot_be_read(tmp_path):
    thor = sm.soc_zones(str(_zones(tmp_path / "thor", THOR)))
    assert sorted(p.parent.name for p in thor) == ["thermal_zone0", "thermal_zone2", "thermal_zone3", "thermal_zone4"]
    assert sm.hottest_c(thor) == pytest.approx(49.718)
    gb10 = sm.soc_zones(str(_zones(tmp_path / "gb10", GB10)))
    assert sm.hottest_c(gb10) == pytest.approx(57.4)
    (tmp_path / "thor/thermal_zone2/temp").write_text("")  # what an EAGAIN zone yields
    assert sm.hottest_c(thor) == pytest.approx(49.531)
    assert sm.hottest_c([tmp_path / "gone" / "temp"]) is None
    assert sm.soc_zones(str(tmp_path / "no-thermal")) == []


def test_the_sampler_reads_a_proc_tree_and_names_only_what_it_logs(tmp_path):
    (tmp_path / "stat").write_text(STAT_A)
    (tmp_path / "meminfo").write_text(MEMINFO)
    thermal = str(_zones(tmp_path / "thermal", THOR))
    sampler = sm.Sampler(None, proc=str(tmp_path), thermal=thermal)
    (tmp_path / "stat").write_text(STAT_B)
    values = sampler.sample()
    assert sampler.names == list(values) == ["cpu_percent", "mem_percent", "temp_c"]
    assert values["cpu_percent"] == pytest.approx(100 * 160 / 940)
    assert values["temp_c"] == pytest.approx(49.718)
    absent = sm.Gpu(nvml="libnvidia-ml-absent.so.1")
    assert not absent.present and absent.percent() is None and absent.temp_c() is None
    station = sm.Sampler(absent, proc=str(tmp_path), thermal=str(tmp_path / "no-thermal"))
    assert station.names == ["cpu_percent", "mem_percent", "gpu_percent", "gpu_temp_c"]
    assert [n for n in station.names if n not in sm.LEGEND] == ["cpu_percent"]  # its pane names the unit


NO_FLEET = "needs a described fleet (config/nodes.json)"   # the public export's reason, which its suite accepts


def _holders(nmap: dict, role: str) -> list[str]:
    """The nodes holding ``role``; without a fleet map, or one without the role, the test skips."""
    found = nodes.nodes_with(nmap, role)
    if not found:
        pytest.skip(NO_FLEET if not nmap else f"no {role} node in this fleet map")
    return found


def test_each_node_reports_under_its_role_never_its_name():
    nmap = nodes.load(REPO)
    if not nmap:
        pytest.skip(NO_FLEET)
    for role in sm.SYSMON_ROLES:
        for node in nodes.nodes_with(nmap, role):
            assert sm.report_role(nmap, node) == role
    quiet = [n for n in nmap if not set(nodes.roles_of(nmap, n)) & set(sm.SYSMON_ROLES)]
    assert all(sm.report_role(nmap, n) is None for n in quiet)
    assert sm.GPU_ROLES == ("station",)


def test_a_legend_is_the_hosts_platform_and_the_palette_pi_is_read_by_the_station():
    nmap = {"ros-host": {"roles": ["ros"], "platform": "thor", "ssh": "u@ros"},
            "station-host": {"roles": ["station"], "ssh": "u@station"},
            "board": {"roles": ["estop-relay"], "checkout": None, "platform": "rpi", "ssh": "u@board",
                      "load_via": "station"},
            "retired": {"roles": [], "load_via": "station", "ssh": "u@retired"}}
    assert sm.label(nmap, "ros-host", "ros") == "thor"
    assert sm.label(nmap, "station-host", "station") == "station"  # no platform: the role
    assert sm.remote_hosts(nmap, "station-host") == ["board"]
    assert sm.remote_hosts(nmap, "ros-host") == []
    found = sm.sources(nmap, "station-host", "station", None)
    assert [(entity, label, type(s).__name__) for entity, label, s in found] == [
        ("station", "station", "Sampler"), ("estop-relay", "rpi", "Remote")]
    assert found[1][2].command[-2:] == ["u@board", sm.REMOTE_LOOP]
    assert set(sm.COLORS) == {"estop-relay", "station", "ros", sm.ROBOT}


def test_the_fleet_map_names_each_plotted_host_by_its_platform():
    nmap = nodes.load(REPO)
    if not nmap:
        pytest.skip(NO_FLEET)
    for role in sm.SYSMON_ROLES:
        for node in nodes.nodes_with(nmap, role):
            assert nmap[node].get("platform"), f"{node} ({role}) has no platform for its legend"
    for host in nmap:
        if nmap[host].get("load_via"):
            assert nmap[host].get("checkout") is None, "a host with a checkout runs its own monitor"


def test_the_remote_reading_is_shell_builtins_and_stops_without_sleep():
    """With no command on PATH the loop still writes one whole reading, then ends instead of spinning."""
    done = subprocess.run(["env", "-i", "PATH=/nonexistent", "/bin/sh", "-c", sm.REMOTE_LOOP],
                          capture_output=True, text=True, timeout=10)
    readings = list(sm.remote_readings(done.stdout.splitlines(True)))
    assert len(readings) == 1 and readings[0][0] is not None
    assert "sleep" in done.stderr
    lines = ["cpu  100 0 100 800 0 0 0 0 0 0\n", "t 47400\n", "t -\n", ".\n",
             "cpu  150 0 150 1600 0 0 0 0 0 0\n", ".\n"]
    (first, hot), (second, cold) = sm.remote_readings(lines)
    assert hot == pytest.approx(47.4) and cold is None
    assert sm.cpu_percent(first, second) == pytest.approx(100 * 100 / 900)


def test_a_remote_host_reports_each_reading_once_and_nothing_once_stale(monkeypatch):
    remote = sm.Remote(["/bin/sh", "-c", sm.REMOTE_LOOP])
    try:
        deadline = time.monotonic() + 5
        while (values := remote.sample()) == {} and time.monotonic() < deadline:
            time.sleep(0.05)
        assert values["cpu_percent"] is None  # the first reading has nothing to difference
        assert remote.sample() == {}  # the same reading is not logged twice
    finally:
        remote.close()
    remote._latest = (time.monotonic() - sm.REMOTE_STALE_S - 1, (1, 2), 40.0)
    monkeypatch.setattr(sm, "REMOTE_RETRY_S", 1e9)
    assert remote.sample() == {}


def _flight(path: Path, rotor: list[list[float]], header: bytes = b"tatbot-flight 2 right realtime0=1.0 256\n"):
    """A version-2 flight log with one record per rotor-temperature row (ros/README.md "Flight log")."""
    body = b""
    for k, temps in enumerate(rotor):
        record = bytearray(256)
        struct.pack_into("<d", record, 0, float(k))
        struct.pack_into("<7f", record, sm.ROTOR_TEMP_AT, *temps)
        body += bytes(record)
    path.write_bytes(header + body + b"\0" * 100)  # and a record still being written


def test_the_robot_is_its_hottest_motor(tmp_path):
    run = tmp_path / "ros-stack" / "20261019T000000Z-node-0000"
    run.mkdir(parents=True)
    (tmp_path / "ros-stack" / "latest").symlink_to(run.name)
    _flight(run / "right-flight.bin", [[30, 40, 41, 35, 35, 31, 30], [28, 41, 43, 36, 36, 31, 30]])
    now = time.time()
    assert sm.newest_rotor_temps(run / "right-flight.bin") == [28, 41, 43, 36, 36, 31, 30]
    assert sm.hottest_motor_c(run, now) == 43
    _flight(run / "left-flight.bin", [[0] * 7])  # a mock arm writes zeros
    assert sm.hottest_motor_c(run, now) == 43
    assert sm.hottest_motor_c(run, now + sm.STALE_S + 1) is None  # the stack stopped
    _flight(run / "old-flight.bin", [[50] * 7], header=b"tatbot-flight 1 old 188\n")
    assert sm.newest_rotor_temps(run / "old-flight.bin") is None  # version 1 had no temperatures

    robot = sm.Robot(tmp_path / "ros-stack")
    assert robot.sample() == {"temp_c": sm.hottest_motor_c(run, time.time())}


def _sysmon(*args, node):
    env = {**os.environ, "TATBOT_NODE": node}
    return subprocess.run([sys.executable, str(REPO / "scripts/lib/tatbot_sysmon.py"), *args],
                          capture_output=True, text=True, timeout=30, env=env)


def test_a_node_without_a_reporting_role_refuses_with_3_and_a_station_says_station():
    nmap = nodes.load(REPO)
    station = _holders(nmap, "station")
    assert _sysmon("--print-role", node=station[0]).stdout.strip() == "station"
    refused = _sysmon("--print-role", node="no-such-node")
    assert refused.returncode == sm.EXIT_NO_ROLE and "nothing to report" in refused.stderr


def test_a_hub_that_does_not_answer_ends_the_process_before_it_streams():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert sm.hub_address(f"rerun+http://127.0.0.1:{port}/proxy") == ("127.0.0.1", port)
        assert sm.hub_answers(("127.0.0.1", port))
    assert not sm.hub_answers(("127.0.0.1", port), timeout=0.5)
    nmap = nodes.load(REPO)
    station = _holders(nmap, "station")
    gone = _sysmon("--connect", f"rerun+http://127.0.0.1:{port}/proxy", "--duration", "1", node=station[0])
    assert gone.returncode == 1 and "not answering" in gone.stderr


def test_it_streams_scalars_under_system_role_into_the_standing_recording(tmp_path):
    pytest.importorskip("rerun")
    nmap = nodes.load(REPO)
    ros = _holders(nmap, "ros")
    out = tmp_path / "sysmon.rrd"
    done = _sysmon("--output", str(out), "--duration", "2.5", node=ros[0])
    assert done.returncode == 0, done.stderr
    printed = subprocess.run([sys.executable, "-m", "rerun", "rrd", "print", str(out)],
                             capture_output=True, text=True, timeout=60).stdout
    assert '"tatbot",\n        "live-cameras"' in printed
    paths = set(re.findall(r"/system/[a-z_/]+", printed))
    temp = {"/system/ros/temp_c"} if sm.soc_zones() else set()
    robot = {"/system/robot/temp_c"}  # named even while no stack runs
    assert paths == {"/system/ros/cpu_percent", "/system/ros/mem_percent"} | temp | robot
    assert "/session/producers/sysmon-ros" in printed


def test_a_hub_the_sdk_lost_ends_the_loop_with_1_for_a_restart(monkeypatch):
    class Recording:
        severed = False

        def flush(self, timeout_sec):
            if self.severed:
                raise RuntimeError("gRPC connection severed")

    recording = Recording()
    rr = type("rr", (), {"get_global_data_recording": staticmethod(lambda: recording)})
    monkeypatch.setattr(sm, "PERIOD_S", 0.01)
    monkeypatch.setattr(sm, "HUB_CHECK_S", 0.0)
    monkeypatch.setattr(sm, "log_values", lambda *a: None)
    found = [("ros", "thor", type("Sampler", (), {"sample": lambda self: {}})())]
    assert sm.stream(rr, found, True, duration_s=0.1) == 0
    recording.severed = True
    assert sm.stream(rr, found, False, duration_s=0.1) == 0  # a file sink is never watched
    assert sm.stream(rr, found, True, duration_s=30) == 1


def _hub(binary: Path, port: int) -> subprocess.Popen:
    hub = subprocess.Popen([str(binary), "--serve-grpc", "--port", str(port), "--server-memory-limit", "64MB"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if sm.hub_answers(("127.0.0.1", port), timeout=0.1):
            return hub
        time.sleep(0.1)
    hub.kill()
    pytest.skip("a local rerun hub did not start")


def test_a_hub_restarted_behind_an_answering_port_still_ends_the_loop(monkeypatch):
    """rerun 0.36 never redials a severed stream, and the restarted hub answers on the
    same port, so only the SDK's flush can tell the stream is dead."""
    rerun_cli = pytest.importorskip("rerun_cli")
    binary = Path(rerun_cli.__file__).parent / "rerun"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(sm, "PERIOD_S", 0.05)
    monkeypatch.setattr(sm, "HUB_CHECK_S", 0.0)
    hub = _hub(binary, port)
    try:
        rr = sm.tr.start("sysmon-test", connect=f"rerun+http://127.0.0.1:{port}/proxy",
                         recording_id="live-cameras", view_coordinates=False)
        found = [("ros", "thor", sm.Sampler(None))]
        assert sm.stream(rr, found, True, duration_s=0.5) == 0
        hub.kill()
        hub.wait()
        hub = _hub(binary, port)
        assert sm.stream(rr, found, True, duration_s=20) == 1
    finally:
        hub.kill()
        hub.wait()
