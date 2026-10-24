"""`tatbot ros`: the backend's pure helpers and every verb's dry run (no ssh, no ROS)."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
        "<f.st...... ros/tatbot_interfaces/action/Touch.action",
        "*deleting   ros/tatbot_rerun/test/old_test.py",
        ".d..t...... ros/tatbot_session/",
        ">f+++++++++ config/nodes.json",
        ">f.st...... ros/README.md",
        "cd+++++++++ ros/tatbot_new/",
        "*deleting   ros/tatbot_motion/tatbot_motion/__pycache__/plan.cpython-312.pyc",
        "*deleting   ros/tatbot_ink/test/.pytest_cache/",
    ])
    assert backend.changed_packages(itemized) == {"tatbot_bridge", "tatbot_interfaces", "tatbot_rerun", "tatbot_new"}
    # what the node marks new after the sync: the ros/ files rsync wrote (sent to a node, <f), not deletions,
    # directories or config
    assert backend.synced_sources(itemized) == ["ros/tatbot_bridge/tatbot_bridge/page.py",
                                                "ros/tatbot_interfaces/action/Touch.action", "ros/README.md"]
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
    monkeypatch.setattr(backend, "stack_defaults", lambda: {**defaults, "machine_switch": "none", "keypad_port": "0"})
    assert backend.launch_args(up_ns()) == []
    assert backend.launch_args(up_ns(hardware="real", estop="serial", page="fixed", no_touch=True, rerun=True)) == [
        "hardware:=real", "estop_source:=serial", "page_source:=fixed", "touch:=false", "rerun:=true"]
    assert backend.launch_args(up_ns(hardware="mock", surface="fitted")) == ["hardware:=mock", "page_surface:=fitted"]
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
    # a draw that ends in an error lands too (2026-10-18: a refused set-up left the arm awake over a forearm)
    draw_line = next(line for line in proc.stdout.splitlines() if "client draw" in line)
    assert "; rc=$?; " in draw_line and "&&" not in draw_line.split("client draw")[1]
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
    proc = tatbot("--dry-run", "ros", "compile", str(design), "--min-stroke-mm", "0", "--mark-width-mm", "0.8")
    assert proc.returncode == 0 and "--min-stroke-mm 0 " in proc.stdout and "--mark-width-mm 0.8" in proc.stdout
    proc = tatbot("--dry-run", "ros", "draw", str(design), "--stencil", str(tmp_path / "print"))
    assert proc.returncode == 0 and f"--stencil {tmp_path / 'print'}" in proc.stdout
    proc = tatbot("--dry-run", "ros", "draw", str(program), "--stencil", str(tmp_path / "print"))
    assert "--stencil apply to artwork" in proc.stdout and "dry run exits 2" in proc.stdout


@needs_fleet   # it resolves the ros node; the public export describes no fleet
def test_a_rehearsal_draws_the_program_in_the_air_without_its_inspection(tmp_path):
    from ros_program_fixtures import program as resource_program

    program = tmp_path / "program.json"
    program.write_text(json.dumps(resource_program()))
    proc = tatbot("--dry-run", "ros", "draw", str(program), "--rehearse", "--standoff-mm", "7")
    assert proc.returncode == 0, proc.stderr
    assert "rehearse: 1 strokes in the air at 7 mm, the machine off, 0 dip(s) left out" in proc.stdout
    assert "client draw" in proc.stdout and "client inspect" not in proc.stdout and "client land" in proc.stdout
    proc = tatbot("--dry-run", "ros", "draw", str(program), "--standoff-mm", "7")
    assert "--standoff-mm is a rehearsal's height" in proc.stdout and "dry run exits 2" in proc.stdout
    proc = tatbot("--dry-run", "ros", "draw", str(program), "--rehearse", "--standoff-mm", "1")
    assert "hover_height_m" in proc.stdout and "dry run exits 2" in proc.stdout


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


def test_the_keypad_takes_keys_from_the_station_alone(monkeypatch):
    defaults = {**backend.stack_defaults(), "machine_switch": "none", "estop": "serial"}
    monkeypatch.setattr(backend, "role_lan", lambda role: {"station": "192.0.2.4"}.get(role))
    for given, expected in (({}, ["keypad_from:=192.0.2.4"]),
                            ({"keypad_from": "192.0.2.5"}, []),   # stack.yaml names its own sender
                            ({"keypad_port": "0"}, [])):          # no keypad node
        monkeypatch.setattr(backend, "stack_defaults", lambda given=given: {**defaults, **given})
        assert backend.launch_args(up_ns()) == expected
    monkeypatch.setattr(backend, "role_lan", lambda role: None)   # no station: the node keeps to its own host
    monkeypatch.setattr(backend, "stack_defaults", lambda: defaults)
    assert backend.launch_args(up_ns()) == []


def test_effective_settings_merge_flags_over_stack_yaml():
    d = backend.stack_defaults()
    base = {k: v for k, v in d.items() if not k.startswith("machine_") and k != "keypad_from"}
    assert set(base) == {"hardware", "estop", "relay_gpio", "probe_gpio", "page", "pattern_id", "arms", "keypad_port"}
    assert all(base.values()) and d["keypad_from"] == ""   # the CLI resolves the station's lan
    assert set(d) - set(base) - {"keypad_from"} == {f"machine_{k}" for k in ("switch", "arm", "addr", "gpio", "port", "estop_port",
                                                           "period_s", "timeout_s", "confirm_s", "sim_endpoint")}
    eff = backend.effective(up_ns(hardware="real", estop="udp"))
    assert eff.startswith("hardware=real estop=udp page=") and f"arms={d['arms']}" in eff


@needs_fleet
def test_lane_up_down_use_transient_units_and_production_the_installed_ones():
    lane = {"TATBOT_ROS_ROOT": "~/tatbot-ros-lanes/test"}
    out = tatbot("--dry-run", "ros", "up", "--hardware", "mock", env=lane).stdout
    assert "systemd-run" in out and "tatbot-ros-lane-test.service" in out and "LimitRTPRIO=95" in out
    assert "systemctl restart tatbot-ros" not in out and "hardware=mock estop=" in out
    assert "rerun:=false (on stdin)" in out   # a lane never draws over the standing recording
    out = tatbot("--dry-run", "ros", "down", env=lane).stdout
    assert "stop tatbot-ros-lane-test.service" in out and "tatbot-ros.service" not in out.replace("lane-test.service", "")
    out = tatbot("--dry-run", "ros", "up", env={"TATBOT_ROS_ROOT": "~/tatbot-ros"}).stdout
    assert "sudo systemctl restart tatbot-ros-router.service tatbot-ros.service" in out and "systemd-run" not in out
    assert "rerun:=false" not in out


@needs_fleet
def test_deploy_records_pending_packages_before_anything_can_fail():
    out = tatbot("--dry-run", "ros", "deploy", "--no-build", env={"TATBOT_ROS_ROOT": "~/tatbot-ros-lanes/test"}).stdout
    steps = [line.split("]")[0].split("[")[-1] for line in out.splitlines() if "] " in line and "[" in line]
    assert steps.index("pending") == steps.index("sync") + 1 and "build" not in steps
    assert "rerun-sdk" not in steps           # a lane's stack runs without the bridge


def test_the_bridges_rerun_sdk_is_an_isolated_install_beside_the_apt_numpy():
    """rerun-sdk asks for numpy 2: in the ROS Python's user or system site it would shadow the apt numpy 1.26 the
    stack's cv2 is built against, so the deploy puts it alone in <root>/rerun-sdk, skipping a present pinned one."""
    script = backend.rerun_sdk(types.SimpleNamespace(root="$HOME/tatbot-ros"))
    check, install, warn = script.split(" || ")
    assert "rerun-sdk/rerun_sdk" in check and "('0.36.0', '21.0.0')" in check and "pip" not in check
    assert "--no-deps" in install and '--target "$HOME/tatbot-ros/rerun-sdk"' in install and "--user" not in install
    assert "rerun-sdk==0.36.0 pyarrow==21.0.0" in install and "numpy" not in install and "WARNING" in warn


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
def test_the_probe_relay_sends_to_the_ros_node():
    """The probe's reader is the ros node's driver, the one stack that drives an arm on the station. A --dest
    replaces it."""
    if not nodes.nodes_with(NMAP, "estop-relay"):
        pytest.skip("no estop-relay node")
    ros = nodes.require_role(NMAP, "ros")
    expected = [f"{NMAP[ros]['lan']}:{backend.PROBE_PORT}"]
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
    assert backend.land_before_stop(target, r, hold) == ["right"]
    assert r.steps == ["arms"] and any("lands after the restart" in t for t in r.said)
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


def _unanswered(effective, lock=None, client="{}"):
    """A running stack's probe with no SafetyState: what a deaf client or a hung session prints."""
    return f"effective={effective}\n" + (f"driver_lock={lock}\n" if lock else "") + f"client={client}\n"


def test_a_running_stack_whose_arms_do_not_answer_is_refused_while_a_driver_may_hold_one(monkeypatch):
    """'No SafetyState' is not 'nothing to land': after a deploy rebuilt tatbot_interfaces under a running stack its
    clients were deaf to it (rmw_zenoh keys topics by type hash, 2026-10-18), and a hung session's driver still holds
    the arm. Either way a restart drops an arm that has not landed. A stack whose arm never came up holds no driver
    lock and restarts as before (10-18 03:00: its spawner had timed out)."""
    try:
        target = backend.Target(backend.fleet.this_node(NMAP) or "", root="~/tatbot-ros-lanes/test")
    except backend.VerbError:
        pytest.skip("this node is not in config/nodes.json")
    hold = argparse.Namespace(hold=False)
    real = "hardware=real estop=udp page=stencil arms=right"
    for probe in (_unanswered(real, "held"), _unanswered(real, "held", '{"error": "timeout"}'),
                  _unanswered(real, "unknown", ""), _unanswered(real)):   # held, unreadable, or not reported
        r = _Steps(probe)
        with pytest.raises(backend.VerbError, match="no arm answered.*tatbot ros down --hold") as refused:
            backend.land_before_stop(target, r, hold)
        assert refused.value.code == backend.EXIT_GATE and r.steps == ["arms"]
    r = _Steps(_unanswered(real, "free"))   # no driver holds an arm: nothing can drop
    assert backend.land_before_stop(target, r, hold) == [] and r.steps == ["arms"]
    for effective in ("hardware=mock estop=none page=fixed arms=right", "hardware=fake estop=none arms=right",
                      "hardware=sim lane=7 config=/x/stack.yaml"):
        r = _Steps(_unanswered(effective, "held"))   # a stack that drives no arm
        assert backend.land_before_stop(target, r, hold) == [] and r.steps == ["arms"]
    with pytest.raises(backend.VerbError, match="hardware unknown"):
        backend.land_before_stop(target, _Steps(_unanswered("estop=udp arms=right", "held")), hold)
    defaults = backend.stack_defaults()
    for hardware, refused in (("real", True), ("mock", False)):   # no TATBOT_ROS_EFFECTIVE line: stack.yaml's
        monkeypatch.setattr(backend, "stack_defaults", lambda hardware=hardware: {**defaults, "hardware": hardware})
        r = _Steps(_unanswered("", "held"))
        with pytest.raises(backend.VerbError) if refused else contextlib.nullcontext():
            backend.land_before_stop(target, r, hold)
    r = _Steps(_unanswered(real, "held"))
    backend.land_before_stop(target, r, argparse.Namespace(hold=True))   # --hold: the operator saw it at rest
    assert r.steps == ["in-cap"]


class _Build(_Steps):
    """A Runner whose `probe` step answers what the node's install and .build-pending say."""

    def __init__(self, probe: str, seq: list, dry: bool = False):
        super().__init__(probe)
        self.dry, self.steps = dry, seq

    def step(self, label, argv, capture=False, stdin=None):
        self.steps.append(label)
        return 0, self.probe if label == "probe" else ""


def test_deploy_lands_and_stops_a_running_stack_before_rebuilding_its_message_types(monkeypatch):
    """A build of tatbot_interfaces (or a clean one) under a running stack would leave every client of the new
    install unable to land it, so the stack is landed and stopped first, while the old install still hears it, and
    never while a draw, touch or calibration runs. `tatbot ros setup`'s own deploy goes the same way."""
    seq, up = [], {"rc": 0}
    monkeypatch.setattr(backend, "down", lambda ns, r: seq.append(f"down hold={ns.hold}"))
    monkeypatch.setattr(backend, "ros_busy", lambda t: seq.append("busy?"))
    monkeypatch.setattr(backend.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=up["rc"]))
    t = types.SimpleNamespace(root="/r", host=lambda c: ["host", c], shell=lambda c: ["ssh", c], container=False,
                              units=("router", "stack"))
    plain, clean = argparse.Namespace(clean=False), argparse.Namespace(clean=True)
    for changed, probe, ns, expect in (
            ({"tatbot_interfaces"}, "installed\n", plain, ["probe", "busy?", "down hold=False", "build"]),
            ({"tatbot_session"}, "installed\ntatbot_interfaces\n", plain, ["probe", "busy?", "down hold=False", "build"]),
            ({"tatbot_session"}, "installed\n", plain, ["probe", "build"]),
            (set(), "installed\n", clean, ["busy?", "down hold=False", "clean", "probe", "build"])):
        seq.clear()
        backend.build(t, _Build(probe, seq), ns, set(changed))
        assert seq == expect, (changed, probe)
    up["rc"] = 1   # no stack running: nothing to stop
    seq.clear()
    backend.build(t, _Build("installed\n", seq), plain, {"tatbot_interfaces"})
    assert seq == ["probe", "build"]
    up["rc"] = 0
    monkeypatch.setattr(backend, "ros_busy", lambda t: (_ for _ in ()).throw(backend.VerbError(6, "a draw runs")))
    seq.clear()
    with pytest.raises(backend.VerbError, match="a draw runs"):
        backend.build(t, _Build("installed\n", seq), plain, {"tatbot_interfaces"})
    assert seq == ["probe"]   # neither landed, stopped nor built
    monkeypatch.setattr(backend.subprocess, "run", lambda *a, **k: pytest.fail("a dry run starts no process"))
    seq.clear()
    backend.build(t, _Build("", seq, dry=True), clean, {"tatbot_interfaces"})
    assert seq == ["clean", "build"]


def test_an_arm_whose_controller_dropped_is_landed_after_a_restart(monkeypatch):
    """2026-10-06: the controller closed its connection 49 s into a stroke and the 3RL's needles stayed in the
    skin; the stack that lost it cannot move it. A failed draw on such an arm restarts the stack and lands it (the
    session withdraws the pen first); any other failure leaves the arm holding for a decision."""
    try:
        target = backend.Target(backend.fleet.this_node(NMAP) or "", root="~/tatbot-ros-lanes/test")
    except backend.VerbError:
        pytest.skip("this node is not in config/nodes.json")
    restarts = []
    monkeypatch.setattr(backend, "restart", lambda t, r: restarts.append(t))
    arm = {"landed": False, "latched": True, "latch_reason": "controller_error", "estop_ok": True,
           "controller_error": True}
    r = _Steps(_probe("hardware=real estop=udp page=stencil arms=right", {"right": arm}))
    backend.recover_controller(target, r, "right")
    assert r.steps == ["arms", "arms", "ready", "land right"] and len(restarts) == 1
    assert any("withdrawing the pen and landing it" in t for t in r.said)
    r = _Steps(_probe("hardware=real estop=udp page=stencil arms=right", {"right": {**arm, "controller_error": False,
                                                                                    "latch_reason": "estop"}}))
    backend.recover_controller(target, r, "right")
    assert r.steps == ["arms"] and len(restarts) == 1


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




def test_a_tracked_page_is_reported_in_the_set_up_pages_frame():
    """A page turned -90 deg in the base (its x along base -y): a pose 2 mm further along base -y and 1 mm along +x
    stands +2 mm along the page's x and +1 mm along its y, turned by the heading difference."""
    placed = [[0.0, 1.0, 0.0, 0.300], [-1.0, 0.0, 0.0, 0.070], [0.0, 0.0, 1.0, 0.040], [0.0, 0.0, 0.0, 1.0]]
    moved = backend.page_moved(placed, [301.0, 68.0, 40.0], [0.0, 0.0, -88.5])
    assert moved == {"x_mm": 2.0, "y_mm": 1.0, "turn_deg": 1.5}


def test_the_page_report_gives_the_set_up_surface_and_says_when_the_stencil_is_not_seen(capsys):
    placed = [[0.0, 1.0, 0.0, 0.309], [-1.0, 0.0, 0.0, 0.072], [0.0, 0.0, 1.0, 0.034], [0.0, 0.0, 0.0, 1.0]]
    sample = {"base_xyz_mm": [306.0, 73.0, 42.0], "base_rpy_deg": [0.0, 0.0, -90.0], "sigma_mm": 4.0}
    out = "\n".join([f"setup={json.dumps({'run': 'r1', 'used': placed})}", json.dumps(sample),
                     json.dumps({"summary": {"pattern_id": "p", "seconds": 5.0, "measured": 1, "lost": 0}})])
    backend.print_page(backend.page_report(out))
    text = capsys.readouterr().out
    assert "surface 34 mm up" in text and "overhead alone reads 42" in text and "73 mm left" in text
    lost = backend.page_report(json.dumps({"summary": {"pattern_id": "p", "seconds": 5.0, "measured": 0, "lost": 5}}))
    backend.print_page(lost)
    assert capsys.readouterr().out.startswith("stencil NOT seen: 0 of 5")


def test_a_setup_whose_process_died_reads_as_stopped_at_its_step(tmp_path, capsys):
    """10-13: the station agent's tool timeout killed a setup mid-touch; its meta still said running."""
    run = tmp_path / "20261013T140923Z-node-9d63"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"status": "running", "started_at": "2026-10-13T14:09:23Z",
                                               "node": {"pid": 2 ** 22 + 12345}, "cmd": {"argv": ["x", "setup"]}}))
    (run / "console.log").write_text("\n".join([
        "# setup: restart with a fixed page", "# setup: register the arm to the D555", "hold 03: tags [3]",
        "# fit: 16 holds", "# setup: touch the station probe", "10:22:25 S1 search -p: no trigger within its retries"]))
    state = backend.setup_state(run)
    assert state["status"].startswith("stopped") and state["step"] == "touch the station probe"
    backend.print_setup_state(state)
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[1].startswith("  stopped at: touch the station probe -- 10:22:25")
    (run / "console.log").write_text("# setup: register the arm to the D555 3.8 min\n# setup: set the page up 3.3 min\n")
    meta = json.loads((run / "meta.json").read_text())
    (run / "meta.json").write_text(json.dumps({**meta, "status": "ok", "exit_code": 0,
                                               "ended_at": "2026-10-13T14:22:44Z"}))
    state = backend.setup_state(run)
    assert state["status"] == "ok" and state["minutes"] == 13.3 and state["done"][-1] == ("set the page up", 3.3)


def test_following_a_setup_prints_its_log_and_returns_its_exit_code(tmp_path, capsys):
    run = tmp_path / "20261013T151106Z-node-e0ff"
    run.mkdir()
    (run / "console.log").write_text("# setup: register the arm to the D555 3.8 min\n")
    (run / "meta.json").write_text(json.dumps({"status": "fail", "exit_code": 3, "started_at": "2026-10-13T15:11:06Z",
                                               "ended_at": "2026-10-13T15:20:00Z", "cmd": {"argv": ["x", "setup"]}}))
    assert backend.setup_follow(run) == 3
    assert "register the arm" in capsys.readouterr().out


def test_a_setup_that_may_run_starts_its_own_session(monkeypatch, capsys):
    """aedb043a returned at once with the dry run's line, exit 0, for every real `setup` (the station agent, 10-13)."""
    started = []
    monkeypatch.setattr(backend, "setup_runs",
                        lambda verb="setup": [] if not started else [Path("/nonexistent/20261013T1700Z-x")])
    monkeypatch.setattr(backend, "ros_busy", lambda t: None)
    monkeypatch.setattr(backend, "ros_target", lambda: None)
    monkeypatch.setattr(backend.subprocess, "Popen", lambda argv, **kw: started.append(argv))
    monkeypatch.setattr(backend.sys, "argv", ["tatbot_ros.py", "setup", "--detach"])
    monkeypatch.setitem(backend.DETACH_WATCH_S, "setup", 0.0)
    ns = argparse.Namespace(detach=True, verb="setup")
    assert backend.setup_detach(ns, backend.Runner(False)) == 0
    assert started and "--child" in started[0]
    assert "started in the background" in capsys.readouterr().out


def test_a_running_setup_asked_again_soon_answers_in_one_line(tmp_path, monkeypatch, capsys):
    """10-13: the station's agent asked --progress 41 times in the first minutes of one setup."""
    import os

    run = tmp_path / "20261013T171034Z-node-00ce"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"status": "running", "started_at": "2026-10-13T17:10:34Z",
                                               "node": {"pid": os.getpid()}, "cmd": {"argv": ["x", "setup"]}}))
    (run / "console.log").write_text("# setup: register the arm to the D555\nhold 03: tags [3]\n")
    monkeypatch.setattr(backend, "setup_runs", lambda verb="setup": [run])
    ns = argparse.Namespace(json=False, verb="setup")
    backend.setup_progress(ns, backend.Runner(False))
    first = capsys.readouterr().out.splitlines()
    assert first[0].startswith("calibration running") and "now: register the arm" in first[0]
    assert "stop here" in first[1]
    backend.setup_progress(ns, backend.Runner(False))
    again = capsys.readouterr().out.splitlines()
    assert len(again) == 1 and again[0].startswith("unchanged: still register the arm")


def test_a_drawing_reports_its_strokes_done(tmp_path, monkeypatch, capsys):
    """A draw runs in its own session like a setup; its progress counts strokes from the stack's own lines."""
    import os

    run = tmp_path / "20261013T200000Z-node-d0d0"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"status": "running", "started_at": "2026-10-13T20:00:00Z",
                                               "node": {"pid": os.getpid()}, "cmd": {"argv": ["x", "draw", "--child"]}}))
    (run / "console.log").write_text("12:04:04 [-] run_start  20261013T200005Z-node-1aea 157 ops arms ['right']\n"
                                     "12:11:17 [right] op_done s0001 \n12:13:17 [right] op_done s0002 \n")
    monkeypatch.setattr(backend, "setup_runs", lambda verb="setup": [run])
    backend.setup_progress(argparse.Namespace(json=False, verb="draw"), backend.Runner(False))
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("drawing running") and "2 of 157 strokes done" in out[0]


def _draw_run(tmp_path, log: str, *, status="running", pid=None, argv=("x", "draw", "--child", "prog.json")):
    import os

    run = tmp_path / "20261014T002916Z-node-bd15"
    run.mkdir(exist_ok=True)
    meta = {"status": status, "started_at": "2026-10-14T00:29:16Z", "node": {"pid": pid or os.getpid()},
            "cmd": {"argv": list(argv)}}
    if status != "running":
        meta.update(exit_code=1, ended_at="2026-10-14T00:29:34Z")
    (run / "meta.json").write_text(json.dumps(meta))
    (run / "console.log").write_text(log)
    return run


def test_a_paused_drawing_says_why_until_a_stroke_moves_on(tmp_path, capsys):
    """10-18: a draw paused off-plan for 16 min while nobody looked; the pause line is the stack's own words."""
    log = ("16:12:23 [-] run_start  20261018T201223Z-node-4704 115 ops arms ['right']\n"
           "16:17:54 [right] op_done s0007\n16:17:54 [right] 8/115 s0008 plan\n"
           "16:17:55 [right] pause s0008 track: the last strokes ended off the plan (s0006 2.4 mm). Hold it down\n"
           "16:17:55 [right] 8/115 s0008 paused\n")
    run = _draw_run(tmp_path, log, argv=("x", "draw", "--child", "/d/prepared/program.json", "--arm", "right",
                                         "--resume", "20261013T234612Z-node-bb82", "--from-op", "s0017"))
    state = backend.setup_state(run)
    assert state["held"].startswith("track: the last strokes ended off the plan") and state["ops_done"] == 1
    assert state["program"] == "/d/prepared/program.json" and state["resumes"] == "20261013T234612Z-node-bb82"
    backend.print_setup_state(state)
    assert "HELD: track: the last strokes" in capsys.readouterr().out
    (run / "console.log").write_text(log + "16:20:01 [right] 8/115 s0008 travel\n")
    assert backend.setup_state(run)["held"] is None


def test_a_drawing_that_ended_says_how_on_the_ros_node(tmp_path, capsys):
    log = ("20:06:57 [-] run_start  20261013T234628Z-node-3e58 430 ops arms ['right']\n"
           "20:08:44 [-] run_end  right: error: page moved 9.9 mm, 10.6 deg from the pose this run touched\n")
    state = backend.setup_state(_draw_run(tmp_path, log, status="fail"))
    assert state["ended"].startswith("error: page moved 9.9 mm") and state["held"] is None
    backend.print_setup_state(state)
    assert "ended on the ros node: error: page moved" in capsys.readouterr().out


def test_a_drawing_whose_land_was_refused_says_the_arm_holds_over_the_skin(tmp_path, capsys):
    """The land after the draw holds the arm over the skin when no lift off it plans (its sweep would cross the skin);
    the station called that arm "awake and holding (normal)" and said it lands after a stop."""
    log = ("20:06:57 [-] run_start  20261013T234628Z-node-3e58 4 ops arms ['right']\n"
           "20:08:44 [-] run_end  right: complete\n"
           '{"ok": false, "landed": [false], "message": "right: not landing: no lift off the skin plans from here"}\n')
    run = _draw_run(tmp_path, log, status="fail")
    state = backend.setup_state(run)
    assert state["ended"] == "complete" and state["not_landed"] == "right: not landing: no lift off the skin plans from here"
    backend.print_setup_state(state)
    assert "NOT LANDED, the arm holds over the skin: right: not landing" in capsys.readouterr().out
    (run / "console.log").write_text(log.replace('"ok": false, "landed": [false]', '"ok": true, "landed": [true]'))
    assert backend.setup_state(run)["not_landed"] is None


def test_a_detached_drawing_that_fails_at_once_says_so(tmp_path, monkeypatch, capsys):
    """10-14: three resumes failed 20 s in; `--detach` had said "started in the background" each time."""
    log = "20:29:31 [-] run_end  _: error: [Errno 2] No such file or directory: 'program.json'\n"
    run = _draw_run(tmp_path, log, status="fail")
    assert backend.detach_report(run, "draw") == 1
    out = capsys.readouterr().out
    assert out.startswith("the drawing did NOT start") and "No such file" in out


def test_a_detached_drawing_returns_once_the_ros_node_took_it(tmp_path, capsys):
    run = _draw_run(tmp_path, "12:04:04 [-] run_start  20261013T200005Z-node-1aea 157 ops arms ['right']\n")
    assert backend.detach_report(run, "draw") == 0
    assert "the ros node took it, 157 ops" in capsys.readouterr().out


def test_a_setup_resumes_at_the_step_it_names_and_keeps_the_stencil_restart(capsys, monkeypatch, local_target):
    """2026-10-16: three setups re-ran ten minutes of registration for a later step's failure."""
    monkeypatch.setattr(backend, "ros_target", lambda: local_target)
    assert backend.main(["--dry-run", "setup", "--hardware", "real", "--estop", "udp",
                         "--relay-addr", "relay.example", "--from", "gauge"]) == 0
    out = capsys.readouterr().out
    steps = [line.split("# setup: ", 1)[1] for line in out.splitlines()
             if line.startswith("# setup: ") and not line.endswith(" min")]
    assert steps == ["restart on the stencil", "measure the wrist gauge on the probe's top",
                     "restart on the drawing gains", "set the page up"]
    # the calibration on the calibration gains, the page on the role's own: a restart between (2026-10-22 benches)
    launches = [line for line in out.splitlines() if "ros2 launch tatbot_bringup stack.launch.py" in line]
    assert len(launches) == 2 and "tuning:=calibration" in launches[0] and "tuning:=" not in launches[1]
    assert backend.main(["--dry-run", "setup", "--hardware", "real", "--estop", "udp",
                         "--relay-addr", "relay.example", "--from", "page"]) == 0
    out = capsys.readouterr().out   # the page alone: one restart, on the drawing gains
    steps = [line.split("# setup: ", 1)[1] for line in out.splitlines()
             if line.startswith("# setup: ") and not line.endswith(" min")]
    launches = [line for line in out.splitlines() if "ros2 launch tatbot_bringup stack.launch.py" in line]
    assert steps == ["restart on the stencil", "set the page up"] and len(launches) == 1 and "tuning:=" not in launches[0]


def test_a_setup_keeps_the_wrist_cameras_registration_not_the_tags(capsys, monkeypatch, local_target):
    """2026-10-22: the agent's calibration adopted the wrist tags' registration, which put the station 9 mm low;
    the tags' now only aims the wrist's, which is the one kept."""
    monkeypatch.setattr(backend, "ros_target", lambda: local_target)
    backend.main(["--dry-run", "setup", "--hardware", "real", "--estop", "udp", "--relay-addr", "relay.example"])
    out = capsys.readouterr().out   # (the height step stops it here: this fleet names no relay node)
    steps = [line.split("# setup: ", 1)[1] for line in out.splitlines()
             if line.startswith("# setup: ") and not line.endswith(" min")]
    assert steps[:5] == ["restart with a fixed page", "register the arm to the D555", "restart on the stencil",
                         "register the D555 through the wrist", "touch the station probe"]
    overhead = next(line for line in out.splitlines() if "tatbot_calib overhead" in line)
    assert "--adopt" in overhead and "--pattern" in overhead
    backend.main(["--dry-run", "setup", "--hardware", "real", "--estop", "udp", "--relay-addr", "relay.example",
                  "--from", "overhead"])
    steps = [line.split("# setup: ", 1)[1] for line in capsys.readouterr().out.splitlines()
             if line.startswith("# setup: ") and not line.endswith(" min")]
    assert steps[:2] == ["restart on the stencil", "register the D555 through the wrist"]


def test_a_station_fix_asks_a_just_restarted_d555_owner_again(monkeypatch, local_target):
    """2026-10-22: the agent's setup fixed the station 9 s after the tags' adoption restarted the D555's owner."""
    monkeypatch.setattr(backend, "ros_target", lambda: local_target)
    monkeypatch.setattr(backend.time, "sleep", lambda s: None)
    codes, said = [5, 5, 0], []

    class Runner:
        dry = False

        def step(self, label, argv, **kw):
            code = codes.pop(0)
            if code:
                raise backend.VerbError(code, "no frame")
            return 0, '{"ok": true, "run_dir": "/runs/fix"}'

        def say(self, text):
            said.append(text)

    assert backend.station_fix("right", "", Runner()) == "/runs/fix"
    assert len(said) == 2 and "again" in said[0]
    codes[:] = [3]
    with pytest.raises(backend.VerbError) as error:   # a station the D555 cannot measure is not asked again
        backend.station_fix("right", "", Runner())
    assert error.value.code == 3 and not codes


def test_a_gauge_its_holds_do_not_agree_on_keeps_the_installed_one_and_the_setup_goes_on(monkeypatch, local_target):
    """2026-10-22: the station agent's setup stopped at the gauge (known to 1.17 mm, 0.25 wanted), its page unset."""
    monkeypatch.setattr(backend, "ros_target", lambda: local_target)
    ran, said = [], []
    r = SimpleNamespace(dry=False, say=said.append)
    monkeypatch.setattr(backend, "wake", lambda *a, **k: None)
    monkeypatch.setattr(backend, "run_cancellable", lambda *a: ran.append("page") or 0)

    def calib(ns, runner):
        ran.append(ns.action)
        if ns.action == "gauge":
            raise backend.VerbError(3, "gauge failed (exit 3)")
        return 0

    monkeypatch.setattr(backend, "calib", calib)
    ns = SimpleNamespace(arm="right", ee_tool=None, pattern=None)
    steps = {label: body for _, label, body in backend.setup_steps(ns, r, local_target, {})}
    assert steps["measure the wrist gauge on the probe's top"]() == 0 and "installed gauge stays" in said[-1]
    assert steps["set the page up"]() == 0 and ran == ["gauge", "page"]

    def broken(ns, runner):
        raise backend.VerbError(1, "gauge failed (exit 1)")

    monkeypatch.setattr(backend, "calib", broken)   # anything else still stops the setup
    with pytest.raises(backend.VerbError):
        steps["measure the wrist gauge on the probe's top"]()


def test_a_setup_says_the_minutes_of_the_steps_it_will_run():
    """2026-10-22: the station agent told a person "about 35 min" for a page step alone (2.4 min)."""
    assert backend.setup_minutes(None) == backend.setup_minutes("register") == sum(backend.SETUP_MIN.values())
    assert backend.setup_minutes("page") == backend.SETUP_MIN["page"]
    assert backend.setup_minutes("gauge") == backend.SETUP_MIN["gauge"] + backend.SETUP_MIN["page"]


def test_a_failed_setup_step_says_how_to_resume_there(monkeypatch):
    ns = argparse.Namespace(arm="right", hardware="real", estop="udp", relay_addr=None, pattern=None, surface=None,
                            arms=None, rerun=True)
    assert backend.resume_flags(ns) == ["--arm", "right", "--hardware", "real", "--estop", "udp", "--rerun"]


# --- sim lanes: the ros-sim owner, a container, never the ros node -------------------------------------------------
HAS_SIM_NODE = len(nodes.nodes_with(NMAP, "ros-sim")) == 1
needs_sim_fleet = pytest.mark.skipif(not (HAS_SIM_NODE and HAS_ROS_NODE), reason=(
    "config/nodes.json names no single ros-sim and ros node" if NMAP else "needs a described fleet (config/nodes.json)"))


def test_a_sim_lane_comes_only_from_the_lane_variable(monkeypatch):
    monkeypatch.delenv("TATBOT_ROS_LANE", raising=False)
    assert backend.sim_lane() is None and backend.sim_lane("real") is None
    with pytest.raises(backend.VerbError) as error:   # an implicit lane: the next `ros down` would stop the live stack
        backend.sim_lane("sim")
    assert error.value.code == backend.EXIT_USAGE and "TATBOT_ROS_LANE" in str(error.value)
    monkeypatch.setenv("TATBOT_ROS_LANE", "3")
    assert backend.sim_lane() == 3 and backend.sim_lane("sim") == 3
    for hardware in ("real", "fake", "mock"):   # a lane is never a way onto real or fake hardware
        with pytest.raises(backend.VerbError) as error:
            backend.sim_lane(hardware)
        assert error.value.code == backend.EXIT_USAGE
    for bad in ("40", "-1", "x"):
        monkeypatch.setenv("TATBOT_ROS_LANE", bad)
        with pytest.raises(backend.VerbError):
            backend.sim_lane()


@needs_sim_fleet
def test_a_sim_lane_targets_the_ros_sim_owner_in_its_container(monkeypatch):
    for name in ("TATBOT_ROS_NODE", "TATBOT_ROS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TATBOT_ROS_LANE", "2")
    t = backend.ros_target()
    assert t.node == nodes.require_role(NMAP, "ros-sim") != nodes.require_role(NMAP, "ros")
    assert (t.root_arg, t.domain, t.port, t.lane) == ("~/tatbot-ros-sim/2", "62", "7462", 2)
    assert t.container and not t.production and t.units == ("tatbot-ros-sim-2",)
    assert t.sim_endpoint == "tcp://127.0.0.1:7654"
    inner = t.shell("source $HOME/tatbot-ros-sim/2/env.sh && ros2 node list")[-1]
    assert inner.startswith("$HOME/tatbot-ros-sim/2/repo/containers/ros-sim/run.sh exec $HOME/tatbot-ros-sim/2 '")
    assert "run.sh" not in t.host("true")[-1]
    assert backend.stack_up(t) == "pgrep -f '[s]tack.launch.py' >/dev/null"   # never its own command line
    assert "systemctl" not in backend.unit_lines(t)
    monkeypatch.setenv("TATBOT_ROS_NODE", nodes.require_role(NMAP, "ros"))   # never the ros node
    with pytest.raises(backend.VerbError) as error:
        backend.ros_target()
    assert error.value.code == backend.EXIT_WRONG_NODE
    monkeypatch.delenv("TATBOT_ROS_LANE")
    monkeypatch.delenv("TATBOT_ROS_NODE")
    plain = backend.ros_target()
    assert not plain.container and plain.node == nodes.require_role(NMAP, "ros")
    assert backend.stack_up(plain) == backend.UNIT_UP.format(unit="tatbot-ros.service")


def sim_dry_run(capsys, *argv, lane: str | None = None, monkeypatch=None) -> tuple[int, str]:
    for name in ("TATBOT_ROS_NODE", "TATBOT_ROS_ROOT", "TATBOT_ROS_LANE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TATBOT_NODE", "operator.invalid")   # never the ros-sim owner, where --config must exist
    if lane is not None:
        monkeypatch.setenv("TATBOT_ROS_LANE", lane)
    code = backend.main(["--dry-run", *argv])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


@needs_sim_fleet
def test_up_hardware_sim_routes_to_the_ros_sim_owner_and_prints_the_container_command(capsys, monkeypatch):
    ros_ssh = nodes.ssh_target(NMAP, nodes.require_role(NMAP, "ros"))
    code, out = sim_dry_run(capsys, "up", "--hardware", "sim", "--config", "/srv/run/inputs/stack.yaml", lane="0",
                            monkeypatch=monkeypatch)
    assert code == 0, out
    assert "sim lane 0, container tatbot-ros-sim-0" in out
    assert "$HOME/tatbot-ros-sim/0/repo/containers/ros-sim/run.sh up $HOME/tatbot-ros-sim/0 --mount /srv/run/inputs" in out
    args = "hardware:=sim config:=/srv/run/inputs/stack.yaml sim_endpoint:=tcp://127.0.0.1:7650 rerun:=false"
    assert f"TATBOT_ROS_ARGS={args} (on stdin)" in out
    assert "tatbot-ci-job-*" in out and "label=tatbot.ci.run" in out   # CI first
    assert (ros_ssh or "\0") not in out and "systemd-run" not in out and "sudo" not in out
    code, out = sim_dry_run(capsys, "up", "--hardware", "sim", "--config", "/srv/x/stack.yaml", lane="5",
                            monkeypatch=monkeypatch)
    assert code == 0 and "ROS_DOMAIN_ID 65, router 127.0.0.1:7465, sim server tcp://127.0.0.1:7660" in out
    assert "machine relay udp://127.0.0.1:7661" in out


@needs_sim_fleet
@pytest.mark.parametrize("argv,lane", [
    (["up", "--hardware", "sim", "--config", "/x/stack.yaml"], None),             # no TATBOT_ROS_LANE
    (["up", "--hardware", "sim"], "0"),                                           # no --config
    (["up", "--hardware", "sim", "--config", "stack.yaml"], "0"),                 # not absolute
    (["up", "--hardware", "sim", "--config", "/x/stack.yaml", "--estop", "udp"], "0"),
    (["up", "--hardware", "sim", "--config", "/x/stack.yaml", "--rerun"], "0"),
    (["up", "--hardware", "real"], "1"),                                          # a lane is sim only
    (["up", "--config", "/x/stack.yaml"], None),                                  # --config is a lane's
    (["deploy", "--units"], "1"),
])
def test_sim_lane_misuse_is_refused_before_anything_runs(capsys, monkeypatch, argv, lane):
    code, out = sim_dry_run(capsys, *argv, lane=lane, monkeypatch=monkeypatch)
    assert code == backend.EXIT_USAGE, out
    assert "[" not in out.split("tatbot ros")[0]   # no step printed


@needs_sim_fleet
def test_a_sim_lanes_deploy_down_status_and_logs_stay_in_its_container(capsys, monkeypatch):
    code, out = sim_dry_run(capsys, "deploy", lane="1", monkeypatch=monkeypatch)
    assert code == 0, out
    assert "containers/ros-sim" in out and "run.sh build $HOME/tatbot-ros-sim/1" in out
    assert "source /opt/ros/jazzy/setup.bash" not in out and "sudo" not in out
    code, out = sim_dry_run(capsys, "down", lane="1", monkeypatch=monkeypatch)
    assert code == 0 and "run.sh down $HOME/tatbot-ros-sim/1" in out and "systemctl stop" not in out
    code, out = sim_dry_run(capsys, "status", lane="1", monkeypatch=monkeypatch)
    assert code == 0 and "run.sh exec $HOME/tatbot-ros-sim/1" in out and "tatbot-ros-sim-1=" in out
    code, out = sim_dry_run(capsys, "logs", "last", lane="1", monkeypatch=monkeypatch)
    assert code == 0 and "run.sh exec $HOME/tatbot-ros-sim/1" in out


def test_a_sim_lane_never_lands_on_the_ros_node_even_if_nodes_json_puts_it_there(monkeypatch):
    """Owner decision (2026-10-18): no sim lanes on the ros node. A nodes.json that gave the ros owner the
    ros-sim role too is refused, not obeyed."""
    for name in ("TATBOT_ROS_NODE", "TATBOT_ROS_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TATBOT_ROS_LANE", "0")
    monkeypatch.setattr(backend, "owner", lambda role: "same-node")
    with pytest.raises(backend.VerbError) as error:
        backend.ros_target()
    assert error.value.code == backend.EXIT_WRONG_NODE and "never runs on the ros node" in str(error.value)
