"""An isolated ROS graph for tatbot_session's end-to-end tests: its own rmw_zenohd on a free port, a
random ROS_DOMAIN_ID, logs under tmp, then robot_state_publisher, ros2_control_node with the given
description, the three controllers and tatbot_session: the session's part of stack.launch.py.
tatbot_bringup exec-depends on tatbot_session, so its config is read from the checkout."""
from __future__ import annotations

import contextlib
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

ZENOHD = Path("/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd")
STAGED = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0]  # config/trossen/tatbot.yaml staged_positions


def repo() -> Path:
    from tatbot_description import repo_root

    return repo_root(None)


def bringup_config(name: str) -> Path:
    return repo() / "ros" / "tatbot_bringup" / "config" / name


def executable(package: str, name: str) -> str:
    from ament_index_python.packages import get_package_prefix

    return str(Path(get_package_prefix(package)) / "lib" / package / name)


def free_port(kind=socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def mock_urdf() -> str:
    """The kinematic description plus a mock_components <ros2_control> block for the right arm."""
    from tatbot_description import names, robot_description

    joints = "".join(
        f'<joint name="{j}"><command_interface name="position"/><command_interface name="velocity"/>'
        f'<state_interface name="position"><param name="initial_value">{q}</param></state_interface>'
        f'<state_interface name="velocity"/><state_interface name="effort"/></joint>'
        for j, q in zip(names.joint_names("right"), STAGED, strict=True))
    states = "".join(f'<state_interface name="{s}"><param name="initial_value">{1 if s == "estop_ok" else 0}'
                     f'</param></state_interface>' for s in names.SAFETY_STATE_INTERFACES)
    commands = "".join(f'<command_interface name="{c}"><param name="initial_value">0</param></command_interface>'
                       for c in names.SAFETY_COMMAND_INTERFACES)
    block = (f'<ros2_control name="right_arm" type="system"><hardware><plugin>mock_components/GenericSystem'
             f'</plugin></hardware>{joints}<gpio name="right_safety">{commands}{states}</gpio></ros2_control>')
    return robot_description(repo(), arms=("right",)).replace("</robot>", block + "</robot>")


def base_stack(**changes) -> dict:
    stack = yaml.safe_load(bringup_config("stack.yaml").read_text())
    stack.update(registration={}, **changes)
    stack["page"]["source"] = "fixed"
    stack["touch"]["enabled"] = False
    stack["page"].setdefault("locate", {})["wrist"] = False
    stack["machine"]["switch"] = "sim"   # the committed stack switches the real arm's machine through the Pi
    return stack


def fast_motion(*, mock: bool = False) -> dict:
    """Temporary mock tuning; fake-driver contact/interlock sequences keep the original motion."""
    import tatbot_motion

    motion = tatbot_motion.load_motion()
    if mock:   # GenericSystem tracks instantly; fake-driver timing/interlock tests retain their motion.
        motion["touch"]["settle_s"] = 0.0
        for mode in ("press", "ride", "hover"):
            motion["pen"][mode]["settle_s"] = 0.0
        motion["approach"].update(omega_max_rad_s=0.8, angular_accel_rad_s2=0.8,
                                  final_m_s=0.020, lift_m_s=0.040)
        motion["corners"].update(pen_up_m_s2=0.1, touchdown_m_s2=0.030)
    return motion


class Graph:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        # The session's runtime record, which research evidence reads: never beside $TATBOT_REPO, where a
        # deployed workspace keeps its live stack's.
        self.runtime_record = tmp / "runtime.json"
        self.procs: list[subprocess.Popen] = []
        port = free_port()
        self.env = dict(os.environ, RMW_IMPLEMENTATION="rmw_zenoh_cpp", ROS_DOMAIN_ID=str(random.randint(100, 200)),
                        TATBOT_REPO=str(repo()), TATBOT_LOG_ROOT=str(tmp / "logs"), ROS_LOG_DIR=str(tmp / "ros"),
                        ZENOH_CONFIG_OVERRIDE=f'connect/endpoints=["tcp/127.0.0.1:{port}"];scouting/multicast/enabled=false')
        self.start([str(ZENOHD)], "router", ZENOH_CONFIG_OVERRIDE=
                   f'listen/endpoints=["tcp/127.0.0.1:{port}"];scouting/multicast/enabled=false')
        time.sleep(1.0)

    def up(self, urdf: str, stack: dict, *, motion: dict | None = None, sensing: bool = False) -> Graph:
        """robot_state_publisher, ros2_control_node, the spawners and tatbot_session; waits for /tatbot/safety."""
        (self.tmp / "rsp.yaml").write_text(yaml.safe_dump(
            {"robot_state_publisher": {"ros__parameters": {"robot_description": urdf}}}))
        from tatbot_session.runtime import configuration_digest, record_controller

        stack['controller_process_file'] = str(self.tmp / 'controller-process.json')
        stack['runtime_record'] = str(self.runtime_record)
        controller_bytes = bringup_config('controllers.yaml').read_bytes()
        controller_file = self.tmp / 'controllers.yaml'
        controller_file.write_bytes(controller_bytes)
        stack['runtime_configuration_sha256'] = configuration_digest(stack, urdf, controller_bytes)
        (self.tmp / "stack.yaml").write_text(yaml.safe_dump(stack))
        controllers = str(controller_file)
        # The controller manager subscribes before robot_state_publisher publishes its latched description:
        # a late joiner of a transient-local topic missed it once on the arm node (rmw_zenoh 0.2.10).
        controller = self.start([executable("controller_manager", "ros2_control_node"), "--ros-args", "--params-file", controllers,
                                 "-r", "/controller_manager/robot_description:=/robot_description"], "control")
        record_controller(stack['controller_process_file'], controller.pid)
        time.sleep(1.0)
        self.start([executable("robot_state_publisher", "robot_state_publisher"), "--ros-args",
                    "--params-file", str(self.tmp / "rsp.yaml")], "rsp")
        spawn = self.run([executable("controller_manager", "spawner"), "joint_state_broadcaster",
                          "right_arm_controller", "right_safety_controller", "--param-file", controllers,
                          "--controller-manager-timeout", "60"], timeout=90)
        assert spawn.returncode == 0, spawn.stdout + spawn.stderr
        session = [executable("tatbot_session", "session")]
        if motion is not None:
            (self.tmp / "motion.yaml").write_text(yaml.safe_dump(motion))
            session = [sys.executable, str(Path(__file__).with_name("fixture_session.py")),
                       str(self.tmp / "motion.yaml"), str(self.tmp / "stack.yaml")]
            if sensing:
                session.append("--sensing")
        self.start([*session, "--ros-args", "-p",
                    f"stack:={self.tmp / 'stack.yaml'}"], "session")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = self.client("status", "--json", timeout=20)
            if status.returncode == 0:
                return self
        raise AssertionError((self.tmp / "session.log").read_text())

    def start(self, argv, name: str, **env) -> subprocess.Popen:
        log = open(self.tmp / f"{name}.log", "w")  # noqa: SIM115 - lives as long as the process
        proc = subprocess.Popen(argv, env=dict(self.env, **env), stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.procs.append(proc)
        return proc

    def run(self, argv, timeout: float = 30.0) -> subprocess.CompletedProcess:
        return subprocess.run(argv, env=self.env, capture_output=True, text=True, timeout=timeout, check=False)

    def client(self, *args, timeout: float = 30.0) -> subprocess.CompletedProcess:
        """The session client to its end; past `timeout` its goal is cancelled (`end`), then TimeoutExpired. Killed
        instead, a timed-out touch kept the arm busy and failed the module's next tests."""
        proc = subprocess.Popen([executable("tatbot_session", "client"), *args], env=self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.end(proc)
            raise
        return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)

    def end(self, proc: subprocess.Popen) -> None:
        """A client still running: `client cancel` cancels every goal and the client gets a minute to see its own end.
        Not a signal, so the cleanup does not hang on the client under test handling one."""
        if proc.poll() is None:
            self.run([executable("tatbot_session", "client"), "cancel"])
            try:
                proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()

    def evidence(self, page: str, slot: str) -> subprocess.CompletedProcess:
        """`tatbot ros evidence`'s reader on this graph's logs and runtime record."""
        return self.run([sys.executable, "-m", "tatbot_session.research", "--page", page, "--slot", slot,
                         "--runtime-record", str(self.runtime_record)])

    def client_bg(self, *args, log: str) -> subprocess.Popen:
        stream = open(self.tmp / log, "w")  # noqa: SIM115 - closed by the OS with the client
        return subprocess.Popen([executable("tatbot_session", "client"), *args], env=self.env, stdout=stream,
                                stderr=subprocess.STDOUT, text=True)

    def wait_for(self, log: str, text: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if text in (self.tmp / log).read_text():
                return True
            time.sleep(0.2)
        return False

    def stop(self) -> None:
        # Keep the router alive until its clients have left: killing it alongside them made zenoh shutdown
        # wait for disconnected peers. Each process group owns only this fixture's descendants.
        for proc in reversed(self.procs):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for log in sorted(self.tmp.glob("*.log")):
            sys.stdout.write(f"--- {log.name}\n{log.read_text()[-3000:]}\n")


def square_program(tmp: Path, *, name: str, sides=(0.010,), pause: bool = False, speed_m_s: float = 0.020) -> Path:
    """Compile closed squares (the first centred, the next ones 15 mm above) with tatbot_ink; with pause,
    a pause op follows the first stroke (on a cartridge tool a colour change compiles to one)."""
    from tatbot_contracts.artwork import freeze_artwork
    from tatbot_contracts.paths import freeze_program
    from tatbot_ink import compile, write_program

    def artwork(side):
        element = {"id": "e", "kind": "path", "closed": True, "fill": False, "width_m": 0.0005, "deposition": 1,
                   "points_m": [[0, 0], [side, 0], [side, side], [0, side]]}
        program = {"canvas_m": {"width": side, "height": side},
                   "inks": [{"id": "black", "color_srgb": [0.0, 0.0, 0.0]}],
                   "layers": [{"id": "layer-0", "ink_id": "black", "elements": [element]}],
                   "negative_space_masks": []}
        program, _ = freeze_program(program, source_sha256='a'*64, name=name, adapter='dbv3-batik-paths/1')
        return freeze_artwork(program, name=name,
                              source={'kind': 'fixture', 'identifier': 'synthetic-e2e-square', 'license': None,
                                      'attribution': None, 'generation': None},
                              conversion={'adapter': 'dbv3-batik-paths/1', 'recipe_sha256': 'b'*64,
                                          'chord_error_m': .000005})

    directory = tmp / name
    directory.mkdir(parents=True, exist_ok=True)
    # Each fixture artwork is prepared independently through the single-pen
    # shorthand. The test explicitly binds their acquired identities to the
    # same configured assembly when composing several placements.
    import copy

    from tatbot_contracts.ros_program import validate_for_execution

    prepared = []
    for index, side in enumerate(sides):
        identity = f"p{index}"
        artwork_path = directory / (identity + ".json")
        artwork_path.write_text(json.dumps(artwork(side)))
        # Preparation binds the same temporary tuning that the fixture node reloads; keep the research
        # configuration digest check live rather than editing the prepared digest after compilation.
        from unittest.mock import patch

        motion = tmp / "motion.yaml"
        override = (patch("tatbot_ink.program.load_motion", return_value=yaml.safe_load(motion.read_text()))
                    if motion.exists() else contextlib.nullcontext())
        with override:
            value = compile(artwork_path, repo=repo(), speed_m_s=speed_m_s, at_m=[0.0, 0.015 * index])
        for op in value['ops']:
            if op['op'] == 'stroke':
                op['src']['placement'] = identity
        prepared.append(value)
    program = copy.deepcopy(prepared[0])
    program['ops'] = [program['ops'][0], *[op for value in prepared for op in value['ops'] if op['op'] == 'stroke']]
    for index, op in enumerate(program['ops']):
        op['id'] = ('t' if op['op'] == 'tool_change' else 's')+f'{index:04d}'
    program['preparation']['pen_bindings'] = [binding for value in prepared for binding in value['preparation']['pen_bindings']]
    validate_for_execution(program)
    if pause:
        first, *rest = program["ops"]
        program["ops"] = [first, {"op": "pause", "id": "p9001", "reason": "test pause"}, *rest]
    return write_program(program, directory / "program.json")


def last_json(text: str) -> dict:
    assert text.strip(), "no output"
    return json.loads(text.strip().splitlines()[-1])


def ledger(run_dir) -> list[dict]:
    return [json.loads(line) for line in (Path(run_dir) / "ledger.jsonl").read_text().splitlines()]
