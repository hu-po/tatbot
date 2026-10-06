"""`tatbot ros`: the backend's pure helpers and every verb's dry run (no ssh, no ROS)."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from cli_runner import REPO, tatbot
from tatbot_cli import nodes

BACKEND = REPO / "scripts" / "lib" / "tatbot_cli" / "tatbot_ros.py"
spec = importlib.util.spec_from_file_location("tatbot_ros_backend", BACKEND)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)

NMAP = nodes.load(REPO)
HAS_ROS_NODE = len(nodes.nodes_with(NMAP, "ros")) == 1
needs_fleet = pytest.mark.skipif(not HAS_ROS_NODE, reason="config/nodes.json names no single ros node")


def test_tip_adoption_imports_without_ros_or_site_packages():
    code = (f'import sys; sys.path.insert(0, {str(REPO / "scripts/lib")!r}); import probe_calibration_adopt; '
            'assert probe_calibration_adopt.apply({"arm": "right", "tool": "x"}, "right:\\n  tool_id: y\\n")[1]')
    subprocess.run([sys.executable, '-S', '-c', code], check=True, capture_output=True, text=True)


def test_rsync_filter_includes_exactly_the_files():
    text = backend.rsync_filter(["ros/a/x.py", "ros/a/b/y.py", "config/z.yaml"])
    lines = text.splitlines()
    assert lines[-1] == "- *"
    assert {"+ /ros/", "+ /ros/a/", "+ /ros/a/b/", "+ /config/"} <= set(lines)
    assert {"+ /ros/a/x.py", "+ /ros/a/b/y.py", "+ /config/z.yaml"} <= set(lines)
    assert len(lines) == 8


def test_changed_packages_from_itemized_rsync():
    itemized = "\n".join([
        ">f.st...... ros/tatbot_bridge/tatbot_bridge/page.py",
        "*deleting   ros/tatbot_rerun/test/old_test.py",
        ".d..t...... ros/tatbot_session/",
        ">f+++++++++ config/nodes.json",
        ">f.st...... ros/README.md",
        "cd+++++++++ ros/tatbot_new/",
        "*deleting   ros/tatbot_motion/tatbot_motion/__pycache__/plan.cpython-312.pyc",
        "*deleting   ros/tatbot_ink/test/.pytest_cache/",
    ])
    assert backend.changed_packages(itemized) == {"tatbot_bridge", "tatbot_rerun", "tatbot_new"}
    said = []
    runner = type("R", (), {"say": lambda self, text: said.append(text)})()
    assert backend.buildable({"tatbot_bridge", "tatbot_estop_relay", "tatbot_new"}, runner) == {"tatbot_bridge"}
    assert len(said) == 1 and "tatbot_new" in said[0]


def test_home_relative_root():
    home = str(Path("~").expanduser())
    assert backend.home_relative(f"{home}/tatbot-ros-lanes/x") == "~/tatbot-ros-lanes/x"
    assert backend.home_relative("/srv/tatbot-ros") == "/srv/tatbot-ros"
    assert backend.home_relative("~/tatbot-ros") == "~/tatbot-ros"


@needs_fleet
@pytest.mark.parametrize('action', ['status', 'load'])
def test_palette_dry_run_routes_measured_cap_declarations_to_the_ros_owner(action):
    args = [] if action == 'status' else ['M1=nighthawk_black', '--level-mm', 'M1=6']
    proc = tatbot('--dry-run', 'ros', 'palette', action, *args)
    assert proc.returncode == 0, proc.stderr
    assert '-m tatbot_cli.ros_palette '+action in proc.stdout
    if action == 'load':
        assert 'M1=nighthawk_black --level-mm M1=6' in proc.stdout


def up_ns(**kw):
    base = {"hardware": None, "estop": None, "relay_addr": None, "page": None, "arms": None, "no_touch": False,
            "rerun": False, "probe": False}
    return argparse.Namespace(**{**base, **kw})


def test_launch_args_are_the_stated_flags_only(monkeypatch):
    defaults = backend.stack_defaults()
    monkeypatch.setattr(backend, "stack_defaults", lambda: {**defaults, "machine_switch": "none"})
    assert backend.launch_args(up_ns()) == []
    assert backend.launch_args(up_ns(hardware="real", estop="serial", page="fixed", no_touch=True, rerun=True)) == [
        "hardware:=real", "estop_source:=serial", "page_source:=fixed", "touch:=false", "rerun:=true"]
    monkeypatch.setattr(backend, "relay_lan", lambda: "relay.example")
    assert "estop_relay_addr:=relay.example" in backend.launch_args(up_ns(estop="udp"))
    # stack.yaml's udp: a real arm resolves the relay without --estop; mock reads no e-stop
    assert backend.launch_args(up_ns(hardware="real")) == ["hardware:=real", "estop_relay_addr:=relay.example"]
    assert backend.launch_args(up_ns(hardware="mock")) == ["hardware:=mock"]
    assert "estop_relay_addr:=other" in backend.launch_args(up_ns(estop="udp", relay_addr="other"))
    # the station probe's relay is the same Pi: --probe names it, with or without the e-stop's udp
    assert backend.launch_args(up_ns(hardware="real", estop="serial", probe=True)) == [
        "hardware:=real", "estop_source:=serial", "probe:=true", "probe_relay_addr:=relay.example"]
    monkeypatch.setattr(backend, "relay_lan", lambda: None)
    with pytest.raises(backend.VerbError) as err:
        backend.launch_args(up_ns(estop="udp"))
    assert err.value.code == 2
    with pytest.raises(backend.VerbError):
        backend.launch_args(up_ns(estop="serial", probe=True))


class _Listing:
    """A Runner that answers every captured remote step with `out`."""
    dry = False

    def __init__(self, out=""):
        self.out, self.commands = out, []

    def step(self, label, argv, *, capture=False, stdin=None):
        self.commands.append(argv[-1])
        return 0, self.out


def test_pattern_names_one_installed_print_by_the_digits_its_sheet_prints():
    """`ros up --pattern` resolves the hex digits a print sheet shows against the prints installed on the ros node;
    none or several is refused before the stack restarts, and the stack gets the full pattern id."""
    target = argparse.Namespace(node="ros", shell=lambda command: ["ssh", "ros", command])
    full = "stencil-71f05c95e635" + "0" * 52
    steps = _Listing(f"{full}/\n")
    assert backend.installed_pattern(target, steps, "71f05c95e635") == full
    assert "ls -d stencil-71f05c95e635*/" in steps.commands[0]
    assert backend.installed_pattern(target, _Listing(f"{full}/\n"), full) == full
    with pytest.raises(backend.VerbError) as error:
        backend.installed_pattern(target, _Listing(""), "71f05c95e635")
    assert error.value.code == backend.EXIT_GATE and "no print is installed" in str(error.value)
    with pytest.raises(backend.VerbError, match="2 prints"):
        backend.installed_pattern(target, _Listing(f"{full}/\n{full[:-1]}1/\n"), "71f05c95e635")
    with pytest.raises(backend.VerbError) as error:
        backend.installed_pattern(target, _Listing(), "71f0; rm -rf")
    assert error.value.code == backend.EXIT_USAGE
    assert backend.launch_args(up_ns(pattern_id=full))[-1] == f"pattern_id:={full}"


@needs_fleet
@pytest.mark.parametrize("args", [
    ("deploy",), ("deploy", "--units", "--clean"), ("up", "--hardware", "mock", "--estop", "none"),
    ("up", "--hardware", "mock", "--estop", "none", "--pattern", "71f05c95e635"),
    ("down", "--router"), ("down", "--hold"), ("status",), ("touch",), ("jog", "--joint", "0", "--delta", "0.02"),
    ("decide", "right", "land"), ("logs", "last"), ("relay-install",), ("ready",)])
def test_every_verb_dry_runs_with_the_backend_plan(args):
    proc = tatbot("--dry-run", "--json", "ros", *args, env={"TATBOT_ROS_ROOT": "~/tatbot-ros-lanes/test"})
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["argv"][:2] == ["python3", str(BACKEND)]
    notes = [n for n in plan["notes"] if n.startswith("plan")]
    assert notes, plan["notes"]
    if args[0] not in ("relay-install", "down"):  # the units, not the root
        assert any("tatbot-ros-lanes/test" in n for n in notes)


@needs_fleet
def test_the_station_fix_is_the_d555s_on_the_ros_node_and_calib_runs_after_it():
    """The station fix is one step on the ros node (the D555 through the arm's registration), with no overhead scan
    elsewhere. `calib run` fixes the station, wakes the arm and calibrates, which refuses a tool without a contact
    model before anything moves; the stated --ee-tool rides along."""
    lane = {"TATBOT_ROS_ROOT": "~/tatbot-ros-lanes/test"}
    out = tatbot("--dry-run", "ros", "station", "--arm", "right", env=lane).stdout
    steps = [line.split("]")[0].split("[")[-1] for line in out.splitlines() if "plan   [" in line]
    assert steps == ["fix"] and "tatbot_calib station fix --arm right" in out and "vision tags scan" not in out
    proc = tatbot("--dry-run", "--ee-tool", "lutin-ballpoint-dot", "ros", "calib", "run", "--arm", "right", env=lane)
    assert proc.returncode == 0, proc.stderr
    steps = [line.split("]")[0].split("[")[-1] for line in proc.stdout.splitlines() if "plan   [" in line]
    assert steps == ["fix", "arms", "calibrate"], steps
    assert "calib run --arm right --station" in proc.stdout and "--tool lutin-ballpoint-dot" in proc.stdout
    assert "--against" not in proc.stdout
    proc = tatbot("--dry-run", "--ee-tool", "lutin-ballpoint-dot", "ros", "calib", "sweep", "--arm", "right", env=lane)
    assert proc.returncode == 0, proc.stderr
    steps = [line.split("]")[0].split("[")[-1] for line in proc.stdout.splitlines() if "plan   [" in line]
    assert steps == ["fix", "arms", "sweep"], steps
    assert "tatbot_calib sweep --arm right --station" in proc.stdout and "rpicam-still" in proc.stdout


def test_a_sweep_still_is_taken_on_the_camera_node_and_sent_into_the_run():
    """The ros node holds no key on the palette camera's node: this node takes the still there, fetches it and sends
    it into the run, and answers why not when a step fails."""
    class Steps(_Steps):
        def step(self, label, argv, capture=False, stdin=None):
            self.steps.append((label, argv))
            if self.fail_land and label == "fetch":
                raise backend.VerbError(1, "fetch failed (exit 1)")
            return 0, ""

    cam = argparse.Namespace(local=False, ssh="pi@camera", shell=lambda script: ["ssh", "pi@camera", script])
    ros = argparse.Namespace(local=False, ssh="me@ros")
    r = Steps("")
    assert backend.still(cam, ros, "/data/ros-calib/x/stills/00-park.jpg", r) == "ok"
    assert [label for label, _ in r.steps] == ["still", "fetch", "send"]
    assert "rpicam-still" in r.steps[0][1][-1] and r.steps[1][1][-2] == f"pi@camera:{backend.STILL_TMP}"
    assert r.steps[2][1][-1] == "me@ros:/data/ros-calib/x/stills/00-park.jpg"
    assert backend.still(cam, ros, "/x.jpg", Steps("", fail_land=True)).startswith("error: fetch failed")


@needs_fleet
def test_register_forwards_a_table_at_zero_and_leaves_unset_options_out():
    """A table at z 0.0 in the arm's base is a height, not an unset option (2026-09-29: `--table-z 0.0` was dropped
    and the holds planned over the configured page's 0.0334)."""
    out = tatbot("--dry-run", "ros", "register", "--arm", "left", "--center", "0.3", "-0.05", "--table-z", "0.0",
                 env={"TATBOT_ROS_ROOT": "~/tatbot-ros-lanes/test"}).stdout
    assert "register --arm left --center 0.3 -0.05 --table-z 0.0" in out
    assert "--prior" not in out and "--max-holds" not in out and "--holds" not in out


@needs_fleet
def test_chain_pools_the_named_registrations_on_the_ros_node_and_moves_nothing():
    proc = tatbot("--dry-run", "ros", "chain", "--arm", "left", "run-a", "run-b",
                  env={"TATBOT_ROS_ROOT": "~/tatbot-ros-lanes/test"})
    assert proc.returncode == 0, proc.stderr
    steps = [line.split("]")[0].split("[")[-1] for line in proc.stdout.splitlines() if "plan   [" in line]
    assert steps == ["chain"] and "tatbot_calib chain --arm left run-a run-b" in proc.stdout


def test_draw_and_compile_dry_run(tmp_path):
    from ros_program_fixtures import program as resource_program

    program = tmp_path / "program.json"
    program.write_text(json.dumps(resource_program()))
    proc = tatbot("--dry-run", "ros", "draw", str(program), "--arm", "right", "--from-op", "t0000")
    assert proc.returncode == 0, proc.stderr
    assert "client draw" in proc.stdout and "--from-op t0000" in proc.stdout and "[send]" in proc.stdout
    # the arm is checked (a landed arm is woken alone), and a complete draw is inspected, then landed
    assert proc.stdout.index("[arms]") < proc.stdout.index("client draw")
    assert "client inspect --arm right" in proc.stdout and "client land --arm right" in proc.stdout
    proc = tatbot("--dry-run", "ros", "draw", str(program), "--hold")
    assert proc.returncode == 0 and "client land" not in proc.stdout and "client inspect" in proc.stdout
    proc = tatbot("--dry-run", "ros", "draw", str(program), "--no-wake", "--hold")
    assert proc.returncode == 0 and "[arms]" not in proc.stdout and "client draw" in proc.stdout
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"schema": "tatbot.inkmap-design/1"}))
    proc = tatbot("--dry-run", "ros", "draw", str(design))
    assert proc.returncode == 0 and "compiling it first" in proc.stdout
    proc = tatbot("--dry-run", "ros", "compile", str(design), "-o", str(tmp_path / "out"))
    assert proc.returncode == 0 and "-m tatbot_ink compile" in proc.stdout
    proc = tatbot("--dry-run", "ros", "compile", str(design), "--stencil", str(tmp_path / "print"))
    assert proc.returncode == 0 and f"--stencil {tmp_path / 'print'}" in proc.stdout
    proc = tatbot("--dry-run", "ros", "draw", str(design), "--stencil", str(tmp_path / "print"))
    assert proc.returncode == 0 and f"--stencil {tmp_path / 'print'}" in proc.stdout
    proc = tatbot("--dry-run", "ros", "draw", str(program), "--stencil", str(tmp_path / "print"))
    assert "--stencil apply to artwork" in proc.stdout and "dry run exits 2" in proc.stdout


@pytest.mark.parametrize('document', [
    {'format': 'tatbot-program', 'version': 2, 'ops': []},
    {'format': 'tatbot-program', 'version': 1, 'ops': [{'op': 'dip'}]},
    {'format': 'tatbot-program', 'version': 1, 'ops': [{'op': 'tool_change'}]},
])
def test_program_refusal_precedes_any_send_or_wake(tmp_path, document):
    path = tmp_path / 'program.json'
    path.write_text(json.dumps(document))
    with pytest.raises(backend.VerbError) as error:
        backend.program_file(str(path), argparse.Namespace(at=None, width=None, speed=None), None)
    assert error.value.code == backend.EXIT_USAGE


@needs_fleet
def test_missing_program_is_a_usage_error_in_the_plan(tmp_path):
    proc = tatbot("--dry-run", "ros", "draw", str(tmp_path / "absent.json"))
    assert proc.returncode == 0  # the CLI's dry run plans; the backend's own dry run names the problem
    assert "the backend's dry run exits 2" in proc.stdout
    assert backend.main(["--dry-run", "draw", str(tmp_path / "absent.json")]) == 2


def test_lane_roots_get_their_own_units():
    for root, units in (("~/tatbot-ros", backend.UNITS),
                        ("~/tatbot-ros-lanes/ros-int.1", ("tatbot-ros-lane-ros-int-1-router.service",
                                                          "tatbot-ros-lane-ros-int-1.service"))):
        try:
            target = backend.Target(backend.fleet.this_node(NMAP) or "", root=root)
        except backend.VerbError:
            pytest.skip("this node is not in config/nodes.json")
        assert target.units == units and target.production == (root == "~/tatbot-ros")


def test_effective_settings_merge_flags_over_stack_yaml():
    d = backend.stack_defaults()
    base = {k: v for k, v in d.items() if not k.startswith("machine_")}
    assert set(base) == {"hardware", "estop", "relay_gpio", "probe_gpio", "page", "arms"} and all(base.values())
    assert set(d) - set(base) == {f"machine_{k}" for k in ("switch", "arm", "addr", "gpio", "port", "estop_port",
                                                           "period_s", "timeout_s", "confirm_s")}
    eff = backend.effective(up_ns(hardware="real", estop="udp"))
    assert eff.startswith("hardware=real estop=udp page=") and f"arms={d['arms']}" in eff


@needs_fleet
def test_lane_up_down_use_transient_units_and_production_the_installed_ones():
    lane = {"TATBOT_ROS_ROOT": "~/tatbot-ros-lanes/test"}
    out = tatbot("--dry-run", "ros", "up", "--hardware", "mock", env=lane).stdout
    assert "systemd-run" in out and "tatbot-ros-lane-test.service" in out and "LimitRTPRIO=95" in out
    assert "systemctl restart tatbot-ros" not in out and "hardware=mock estop=" in out
    out = tatbot("--dry-run", "ros", "down", env=lane).stdout
    assert "stop tatbot-ros-lane-test.service" in out and "tatbot-ros.service" not in out.replace("lane-test.service", "")
    out = tatbot("--dry-run", "ros", "up", env={"TATBOT_ROS_ROOT": "~/tatbot-ros"}).stdout
    assert "sudo systemctl restart tatbot-ros-router.service tatbot-ros.service" in out and "systemd-run" not in out


@needs_fleet
def test_deploy_records_pending_packages_before_anything_can_fail():
    out = tatbot("--dry-run", "ros", "deploy", "--no-build", env={"TATBOT_ROS_ROOT": "~/tatbot-ros-lanes/test"}).stdout
    steps = [line.split("]")[0].split("[")[-1] for line in out.splitlines() if "] " in line and "[" in line]
    assert steps.index("pending") == steps.index("sync") + 1 and "build" not in steps


@needs_fleet
def test_relay_install_copies_over_ssh_and_installs_the_udev_rule_for_a_pico(monkeypatch):
    if not nodes.nodes_with(NMAP, "estop-relay"):
        pytest.skip("no estop-relay node")
    out = tatbot("--dry-run", "ros", "relay-install", "--dest", "192.0.2.1:7640").stdout
    assert "rsync" not in out and "[copy 99-tatbot-estop.rules]" in out and "udevadm trigger" not in out
    defaults = backend.stack_defaults()
    monkeypatch.setattr(backend, "stack_defaults", lambda: {**defaults, "relay_gpio": ""})
    with contextlib.redirect_stdout(io.StringIO()) as pico:
        assert backend.main(["--dry-run", "relay-install", "--dest", "192.0.2.1:7640"]) == 0
    assert "udevadm trigger" in pico.getvalue() and "--device /dev/tatbot-estop --dest" in pico.getvalue()
    out = tatbot("--dry-run", "ros", "relay-install", "--dest", "192.0.2.1:7640", "--bind", "192.0.2.7").stdout
    assert "@ARGS@|--gpio GPIO17 --dest 192.0.2.1:7640 --bind 192.0.2.7|" in out
    assert backend.main(["--dry-run", "relay-install", "--dest", "192.0.2.1:7640", "--bind", "eth0"]) == 2


@needs_fleet
def test_relay_install_sends_to_every_reader_of_the_button(monkeypatch):
    """With the arm node's profile e-stop on the relay, one button stops both arms: the relay's default
    readers are the ros node's driver and every `estop` node, each at its own port, and the tattoo machine's
    switch on the Pi's own loopback."""
    if not nodes.nodes_with(NMAP, "estop-relay") or not all(NMAP[n].get("lan") for n in nodes.nodes_with(NMAP, "estop")):
        pytest.skip("no estop-relay node, or an estop node without a lan")
    ros_lan = NMAP[nodes.require_role(NMAP, "ros")]["lan"]
    monkeypatch.setattr(backend, "arm_estop_port", lambda: 7640)
    readers = ([f"{ros_lan}:{backend.UDP_PORT}"] + [f"{NMAP[n]['lan']}:7640" for n in nodes.nodes_with(NMAP, "estop")]
               + ["127.0.0.1:7643"])
    assert backend.estop_dests() == list(dict.fromkeys(readers))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert backend.main(["--dry-run", "relay-install"]) == 0
    assert "".join(f" --dest {d}" for d in dict.fromkeys(readers)) in out.getvalue()
    monkeypatch.setattr(backend, "arm_estop_port", lambda: None)   # a serial e-stop on the arm node
    assert backend.estop_dests() == [f"{ros_lan}:{backend.UDP_PORT}", "127.0.0.1:7643"]


@needs_fleet
def test_the_probe_relay_sends_to_the_ros_node_and_the_arm_node():
    """The probe's readers are wherever a stack drives an arm on the station: the ros node's driver, and the arm
    node's when the blue arm calibrates there. A --dest replaces them."""
    if not nodes.nodes_with(NMAP, "estop-relay"):
        pytest.skip("no estop-relay node")
    readers = [nodes.require_role(NMAP, "ros"), *nodes.nodes_with(NMAP, "arm")]
    expected = list(dict.fromkeys(f"{NMAP[n]['lan']}:{backend.PROBE_PORT}" for n in readers if NMAP[n].get("lan")))
    assert backend.probe_dests() == expected and expected
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert backend.main(["--dry-run", "relay-install", "--probe"]) == 0
    assert "--probe GPIO27" + "".join(f" --dest {d}" for d in expected) in out.getvalue()
    assert "tatbot-estop-relay.service " not in out.getvalue()   # the e-stop relay's unit is never touched
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert backend.main(["--dry-run", "relay-install", "--probe", "--dest", "192.0.2.1:7641"]) == 0
    assert "--probe GPIO27 --dest 192.0.2.1:7641|" in out.getvalue()


@needs_fleet
def test_the_machine_switch_takes_the_ros_nodes_commands_and_the_button_over_the_loopback(monkeypatch):
    """The machine relay is its own instance on the estop-relay node. It installs only once its line is recorded,
    and takes commands from the ros node alone; the e-stop relay always sends to it on the Pi's loopback. A stack
    with a pi switch resolves the relay's address at up."""
    if not nodes.nodes_with(NMAP, "estop-relay"):
        pytest.skip("no estop-relay node")
    defaults = backend.stack_defaults()
    monkeypatch.setattr(backend, "stack_defaults", lambda: {**defaults, "machine_gpio": ""})
    assert backend.main(["--dry-run", "relay-install", "--machine"]) == 2   # unwired fixture
    monkeypatch.setattr(backend, "stack_defaults", lambda: {**defaults, "machine_gpio": "GPIO22"})
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert backend.main(["--dry-run", "relay-install", "--machine"]) == 0
    ros_lan = NMAP[nodes.require_role(NMAP, "ros")]["lan"]
    assert f"@ARGS@|--machine GPIO22 --from {ros_lan} --listen 7642 --estop-port 7643 --timeout 0.2|" in out.getvalue()
    assert "tatbot-machine-relay.service" in out.getvalue() and "tatbot-estop-relay.service " not in out.getvalue()
    monkeypatch.setattr(backend, "stack_defaults", lambda: {**defaults, "machine_switch": "pi"})
    monkeypatch.setattr(backend, "relay_lan", lambda: "relay.example")
    assert "machine_addr:=relay.example" in backend.launch_args(up_ns(hardware="mock"))


@needs_fleet
def test_draw_passes_the_global_tool_and_cli_logs_are_local(tmp_path):
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"schema": "tatbot.inkmap-design/1"}))
    tool = "lutin-ballpoint-dot"
    proc = tatbot("--dry-run", "--ee-tool", tool, "ros", "draw", str(design))
    assert proc.returncode == 0, proc.stderr
    assert f"--ee-tool {tool}" in proc.stdout
    out = tatbot("--dry-run", "ros", "logs", "last", "--workflow", "ros-cli").stdout
    assert "[local]" in out and "ssh" not in out


@needs_fleet
def test_down_and_up_land_first_unless_hold():
    lane = {"TATBOT_ROS_ROOT": "~/tatbot-ros-lanes/test"}
    out = tatbot("--dry-run", "ros", "down", env=lane).stdout
    assert out.index("[arms]") < out.index("[stop]") and "client land --arm" in out
    out = tatbot("--dry-run", "ros", "down", "--hold", env=lane).stdout
    assert "[arms]" not in out and "--hold: stopping without landing" in out
    out = tatbot("--dry-run", "ros", "up", "--hardware", "real", "--estop", "none", "--page", "fixed", env=lane).stdout
    assert "[arms]" in out and "rocker switch is the only stop" in out
    out = tatbot("--dry-run", "ros", "up", "--hardware", "real", "--estop", "serial", env=lane).stdout
    assert "rocker switch" not in out


class _Steps:
    """A Runner that answers the probe and records the other steps; land steps fail when told to."""

    def __init__(self, probe: str, fail_land: bool = False, in_cap: str = ""):
        self.dry, self.probe, self.fail_land, self.steps, self.said = False, probe, fail_land, [], []
        self.in_cap = in_cap

    def say(self, text):
        self.said.append(text)

    def step(self, label, argv, capture=False, stdin=None):
        self.steps.append(label)
        if label.startswith("land") and self.fail_land:
            raise backend.VerbError(1, f"{label} failed (exit 1)")
        return 0, self.probe if label == "arms" else self.in_cap if label == "in-cap" else ""


def _probe(effective, safety):
    return f"effective={effective}\nclient={json.dumps({'safety': safety})}\n"


def test_land_before_stop_lands_each_arm_that_has_not_landed():
    try:
        target = backend.Target(backend.fleet.this_node(NMAP) or "", root="~/tatbot-ros-lanes/test")
    except backend.VerbError:
        pytest.skip("this node is not in config/nodes.json")
    hold = argparse.Namespace(hold=False)
    arm = {"landed": False, "latched": False, "latch_reason": "none", "estop_ok": True, "controller_error": False}
    r = _Steps(_probe("hardware=real estop=udp page=stencil arms=right", {"right": arm}))
    backend.land_before_stop(target, r, hold)
    assert r.steps == ["arms", "land right"] and any("right: landed=False" in t for t in r.said)
    r = _Steps(_probe("hardware=real estop=udp page=stencil arms=right", {"right": {**arm, "landed": True}}))
    backend.land_before_stop(target, r, hold)
    assert r.steps == ["arms"]
    r = _Steps(_probe("hardware=mock estop=none page=fixed arms=right", {"right": arm}))
    backend.land_before_stop(target, r, hold)
    assert r.steps == ["arms"]
    r = _Steps(_probe("hardware=real estop=udp page=stencil arms=right", {"right": {**arm, "controller_error": True}}))
    backend.land_before_stop(target, r, hold)
    assert r.steps == ["arms"] and any("refuses to land" in t for t in r.said)
    r = _Steps("")  # the stack is not running
    backend.land_before_stop(target, r, hold)
    assert r.steps == ["arms"]
    r = _Steps(_probe("hardware=real estop=udp page=stencil arms=right", {"right": arm}), fail_land=True)
    with pytest.raises(backend.VerbError, match="still up and holding"):
        backend.land_before_stop(target, r, hold)
    r = _Steps("")
    backend.land_before_stop(target, r, argparse.Namespace(hold=True))
    assert r.steps == ["in-cap"]
    # an arm stopped unlanded sags (9 mm at the palette, 2026-10-05): never with a tool that may be in a cap
    r = _Steps("", in_cap="~/.local/state/tatbot/in-cap-right.json\n")
    with pytest.raises(backend.VerbError, match="drop it into the cap") as refused:
        backend.land_before_stop(target, r, argparse.Namespace(hold=True))
    assert refused.value.code == backend.EXIT_GATE


def test_a_draw_wakes_its_landed_arm_alone_and_waits_for_it():
    """Not the stack: a restart would take the other arm down mid-goal (2026-09-30, two agents on one rig)."""
    try:
        target = backend.Target(backend.fleet.this_node(NMAP) or "", root="~/tatbot-ros-lanes/test")
    except backend.VerbError:
        pytest.skip("this node is not in config/nodes.json")
    arm = {"landed": False, "latched": False, "latch_reason": "none", "estop_ok": True, "controller_error": False}
    r = _Steps(_probe("hardware=real estop=none page=stencil arms=right", {"right": arm}))
    backend.wake(target, r, "right")
    assert r.steps == ["arms"]
    r = _Steps(_probe("hardware=real estop=none page=stencil arms=right", {"right": {**arm, "landed": True}}))
    backend.wake(target, r, "right")
    assert r.steps == ["arms", "wake right", "ready"] and any("landed and idle" in t for t in r.said)
    assert "restart" not in " ".join(r.said)
    r = _Steps("")  # no stack answering: the draw itself reports it
    backend.wake(target, r, "right")
    assert r.steps == ["arms"]


def test_a_calibration_lands_an_awake_arm_first_so_it_starts_from_rest():
    """A calibration wakes its arm at rest: one still awake where its last goal left it lands first, then is woken
    (2026-09-29: from a free-air touch 0.3 m out, every way to the station met joint_1's limit)."""
    try:
        target = backend.Target(backend.fleet.this_node(NMAP) or "", root="~/tatbot-ros-lanes/test")
    except backend.VerbError:
        pytest.skip("this node is not in config/nodes.json")
    arm = {"landed": False, "latched": False, "latch_reason": "none", "estop_ok": True, "controller_error": False}
    r = _Steps(_probe("hardware=real estop=udp page=stencil arms=right", {"right": arm}))
    backend.wake(target, r, "right", page=False, rest=True)
    assert r.steps == ["arms", "land right", "wake right", "ready"] and any("awake where" in t for t in r.said)
    r = _Steps(_probe("hardware=real estop=udp page=stencil arms=right", {"right": {**arm, "landed": True}}))
    backend.wake(target, r, "right", page=False, rest=True)
    assert r.steps == ["arms", "wake right", "ready"]
    r = _Steps("")  # no stack answering: the calibration itself reports it
    backend.wake(target, r, "right", page=False, rest=True)
    assert r.steps == ["arms"]


class _NotReady(_Steps):
    def step(self, label, argv, capture=False, stdin=None):
        if label == "ready":
            self.steps.append(label)
            raise backend.VerbError(1, "ready failed (exit 1)")
        return super().step(label, argv, capture, stdin)


def test_a_draw_that_cannot_get_ready_lands_the_arm_it_woke():
    try:
        target = backend.Target(backend.fleet.this_node(NMAP) or "", root="~/tatbot-ros-lanes/test")
    except backend.VerbError:
        pytest.skip("this node is not in config/nodes.json")
    arm = {"landed": True, "latched": False, "latch_reason": "none", "estop_ok": True, "controller_error": False}
    r = _NotReady(_probe("hardware=real estop=none page=stencil arms=right", {"right": arm}))
    with pytest.raises(backend.VerbError, match="ready failed"):
        backend.wake(target, r, "right")
    assert r.steps == ["arms", "wake right", "ready", "land right"]


class _Scripts(_Steps):
    """A _Steps that also keeps each step's command line."""

    def step(self, label, argv, capture=False, stdin=None):
        self.scripts = {**getattr(self, "scripts", {}), label: " ".join(argv)}
        return super().step(label, argv, capture, stdin)


@pytest.fixture
def local_target(monkeypatch):
    monkeypatch.setattr(backend.fleet, "load", lambda repo: {})
    monkeypatch.setattr(backend.fleet, "this_node", lambda nmap: "fixture")
    return backend.Target("fixture", root="~/tatbot-ros")


def test_a_fixed_page_is_never_waited_for_on_the_bus():
    """Waking a landed arm waits for a measured page only when the stencil tracker supplies it: a fixed page
    never reaches the bus, so waiting for one failed every draw after the first on a fixed-page stack."""
    assert backend.page_on_bus("hardware=real estop=udp page=stencil arms=right")
    assert not backend.page_on_bus("hardware=real estop=udp page=fixed arms=right")
    assert backend.page_on_bus("")   # a stack.env from before TATBOT_ROS_EFFECTIVE waits, as it always did
    assert "page_watch" in backend.ready_wait("~/tatbot-ros", "right", True)
    fixed = backend.ready_wait("~/tatbot-ros", "right", False)
    assert "page_watch" not in fixed and fixed.rstrip().endswith("exit 0")
    try:
        target = backend.Target(backend.fleet.this_node(NMAP) or "", root="~/tatbot-ros-lanes/test")
    except backend.VerbError:
        pytest.skip("this node is not in config/nodes.json")
    arm = {"landed": True, "latched": False, "latch_reason": "none", "estop_ok": True, "controller_error": False}
    for page, waits in (("fixed", False), ("stencil", True)):
        r = _Scripts(_probe(f"hardware=real estop=udp page={page} arms=right", {"right": arm}))
        backend.wake(target, r, "right")
        assert r.steps == ["arms", "wake right", "ready"] and ("page_watch" in r.scripts["ready"]) is waits
        assert "client wake --arm right" in r.scripts["wake right"]


def test_the_probes_go_on_when_systemd_does_not_answer():
    """A full system bus makes `systemctl is-active` print "Failed to ..."; that must not read as a stopped
    stack, or `up` restarts an arm that has not landed without landing it."""
    import subprocess

    check = backend.UNIT_UP.format(unit="x.service")
    fake = lambda out: f'systemctl() {{ echo "{out}"; }}; {check}'   # noqa: E731
    run = lambda out: subprocess.run(["bash", "-c", fake(out)]).returncode   # noqa: E731
    assert run("active") == 0
    assert run("Failed to retrieve unit state: No buffer space available") == 0
    assert run("inactive") != 0 and run("failed") != 0


