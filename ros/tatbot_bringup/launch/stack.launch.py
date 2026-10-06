"""The whole arm-node stack: robot_state_publisher, controller_manager at 400 Hz with one TatbotArm
(or mock) system per arm, the controllers, tatbot_bridge, tatbot_session and optionally tatbot_rerun
and a stack-wide MCAP.

Every argument defaults to stack.yaml (ros/tatbot_bringup/config/stack.yaml) when empty; `tatbot ros
up` writes the ones the operator chose into ~/tatbot-ros/stack.env for the systemd unit. The launch
opens a `ros-stack` run (tatbot_runlog) and writes into it:
  stack.yaml              the effective configuration (arguments applied, the page's size, clear centre and
                          inner border edges from the installed print page.pattern_id names); every tatbot
                          node reads it through its `stack` (or `config`) parameter
  robot_description.urdf  what robot_state_publisher and controller_manager got
  ros/                    ROS_LOG_DIR of every node
  flight rings            the driver's `flight_path` is the run directory
  bag/                    the stack-wide MCAP when `record` is set
tatbot_session publishes its runtime identity outside the run, to `runtime_record`: by default
runtime.json in the workspace root holding the repo (~/tatbot-ros/runtime.json), which `tatbot ros
evidence` reads. A test harness points it into its own tmp dir, never at a live stack's.
"""
import signal
import sys
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    Shutdown,
)
from launch.event_handlers import OnProcessExit, OnProcessStart, OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from tatbot_description import STACK_YAML, load_stack, names, repo_root, robot_description
from tatbot_session.config import print_page
from tatbot_session.runtime import (
    configuration_digest,
    record_controller,
    workspace_description,
    workspace_record,
)

SHARE = Path(__file__).resolve().parents[1]  # share/tatbot_bringup (or the source package)

ARGUMENTS = {
    "config": ("", "stack.yaml path; '' = <repo>/ros/tatbot_bringup/config/stack.yaml"),
    "repo": ("", "repo checkout holding config/ and urdf/; '' = $TATBOT_REPO"),
    "runtime_record": ("", "where tatbot_session publishes its runtime identity; '' = <repo>/../runtime.json"),
    "arms": ("", "comma-separated arms; '' = stack.yaml arms (right)"),
    "hardware": ("", "mock | fake | real; '' = stack.yaml hardware"),
    "profile": ("", "config/profiles/<profile>.json for arm addresses; '' = stack.yaml profile"),
    "estop_source": ("", "none | serial | udp; '' = stack.yaml estop.source"),
    "estop_device": ("", "serial device; '' = stack.yaml estop.serial_device"),
    "estop_udp_port": ("", "UDP bind port; '' = stack.yaml estop.udp_port"),
    "estop_relay_addr": ("", "the relay's source address (udp); '' = stack.yaml estop.relay_addr"),
    "probe": ("", "true | false: read the station probe (PRB1); '' = stack.yaml probe.enabled"),
    "probe_relay_addr": ("", "the probe relay's source address; '' = stack.yaml probe.relay_addr"),
    "machine_addr": ("", "the machine relay's address; '' = stack.yaml machine.addr"),
    "page_source": ("", "stencil | fixed; '' = stack.yaml page.source"),
    "pattern_id": ("", "the installed print to draw on; '' = stack.yaml page.pattern_id"),
    "touch": ("", "true | false: page touches at run start; '' = stack.yaml touch.enabled"),
    "rerun": ("", "true | false: start tatbot_rerun; '' = stack.yaml rerun.enabled"),
    "record": ("", "true | false: stack-wide MCAP in the ros-stack run; '' = stack.yaml record"),
}


# 0, and death by SIGINT or SIGTERM as a signal or as the shell's 128 + signal.
CLEAN_EXITS = {0, -signal.SIGINT, -signal.SIGTERM, 128 + signal.SIGINT, 128 + signal.SIGTERM}


def _bool(text: str) -> bool:
    if text.lower() not in ("true", "false", "1", "0"):
        raise ValueError(f"expected true or false, got {text!r}")
    return text.lower() in ("true", "1")


# launch argument -> (stack.yaml key path, parser)
OVERRIDES = {
    "arms": (("arms",), lambda text: [arm.strip() for arm in text.split(",") if arm.strip()]),
    "hardware": (("hardware",), str),
    "profile": (("profile",), str),
    "estop_source": (("estop", "source"), str),
    "estop_device": (("estop", "serial_device"), str),
    "estop_udp_port": (("estop", "udp_port"), int),
    "estop_relay_addr": (("estop", "relay_addr"), str),
    "probe": (("probe", "enabled"), _bool),
    "probe_relay_addr": (("probe", "relay_addr"), str),
    "machine_addr": (("machine", "addr"), str),
    "page_source": (("page", "source"), str),
    "pattern_id": (("page", "pattern_id"), str),
    "touch": (("touch", "enabled"), _bool),
    "rerun": (("rerun", "enabled"), _bool),
    "record": (("record",), _bool),
}


def effective_stack(args: dict) -> dict:
    """stack.yaml (`config`, else the checkout's, the file the CLI and robot_description() read) with
    every non-empty launch argument applied, `config_path`: that file, from which tatbot_session
    re-reads page.trim at every goal, and `runtime_record`: where tatbot_session publishes its runtime
    identity, the argument, else the workspace root's."""
    stack = load_stack(args.get("repo") or None, args.get("config") or None)
    stack["config_path"] = str(Path(args["config"]).expanduser() if args.get("config")
                               else repo_root(args.get("repo") or None) / STACK_YAML)
    stack["runtime_record"] = str(Path(args["runtime_record"]).expanduser() if args.get("runtime_record")
                                  else workspace_record(repo_root(args.get("repo") or None)))
    for name, (path, parse) in OVERRIDES.items():
        if args.get(name):
            node = stack
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = parse(args[name])
    return stack


def registrations(stack: dict) -> dict:
    """{arm: registration path} for the arms whose adopted registration file exists."""
    found = {}
    for arm in stack["arms"]:
        path = (stack.get("registration") or {}).get(arm)
        if path and Path(path).expanduser().is_file():
            found[arm] = str(Path(path).expanduser())
    return found


def check_placement(stack: dict, placed: dict) -> None:
    """A real arm drawing on the camera-tracked stencil needs its adopted registration: without it the
    base would hang at the nominal URDF mount while the page arrives in the camera's world."""
    missing = [arm for arm in stack["arms"] if arm not in placed]
    if stack["hardware"] == "real" and stack["page"]["source"] == "stencil" and missing:
        raise RuntimeError("no adopted registration for " + ", ".join(
            f"{arm} ({(stack.get('registration') or {}).get(arm) or 'registration.' + arm + ' unset'})"
            for arm in missing) + ": run `tatbot ros deploy`, or use page_source fixed")


def _open_run(repo: Path, stack: dict):
    sys.path.insert(0, str(repo / "scripts" / "lib"))
    import tatbot_runlog

    meta = {"hardware": stack["hardware"], "arms": stack["arms"], "estop_source": stack["estop"]["source"],
            "page_source": stack["page"]["source"]}
    revision = repo / "REVISION"
    if revision.is_file():  # `tatbot ros deploy`: the sha, then dirty=0|1
        lines = revision.read_text().split()
        meta["revision"] = {"sha": lines[0] if lines else "", "dirty": "dirty=1" in lines}
    return tatbot_runlog.init(names.RUN_WORKFLOW_STACK, meta=meta, attach_logging=False)


def generate(context, *args, **kwargs):
    values = {name: LaunchConfiguration(name).perform(context) for name in ARGUMENTS}
    repo = repo_root(values["repo"] or None)
    stack = effective_stack(values)
    stack["page"] = print_page(repo, stack["page"])
    arms = tuple(stack["arms"])
    placed = registrations(stack)
    check_placement(stack, placed)
    run = _open_run(repo, stack)
    try:
        return _actions(run, repo, stack, arms, placed)
    except BaseException:
        run.finalize(1)
        raise


def _optional_nodes(stack: dict, arms: tuple, node_params: list) -> list:
    """The operator's numpad on this host (stack.yaml keypad) and the Rerun bridge, when configured."""
    keypad = stack.get("keypad") or {}
    return ([Node(package="tatbot_session", executable="keypad", name=names.KEYPAD_NODE, output="both",
                  parameters=node_params)] if keypad.get("device") and keypad.get("arm") in arms else []) + (
        [Node(package="tatbot_rerun", executable="rerun_bridge", name=names.RERUN_NODE, output="both",
              parameters=node_params)] if stack["rerun"]["enabled"] else [])


def _actions(run, repo: Path, stack: dict, arms: tuple, placed: dict) -> list:
    stack["flight_path"] = str(run.dir)
    stack['controller_process_file'] = str(run.dir / 'controller-process.json')
    stack_path = run.dir / "stack.yaml"
    description, stack['runtime_workspace_sha256'] = workspace_description(repo, lambda: robot_description(
        repo, arms=arms, hardware=stack["hardware"], registrations=placed, ros2_control=stack))
    (run.dir / "robot_description.urdf").write_text(description)
    controller_bytes = (SHARE / 'config' / 'controllers.yaml').read_bytes()
    controller_file = run.dir / 'controllers.yaml'
    controller_file.write_bytes(controller_bytes)
    controllers = str(controller_file)
    stack['runtime_configuration_sha256'] = configuration_digest(stack, description, controller_bytes)
    stack_path.write_text(yaml.safe_dump(stack, sort_keys=False))
    rt = stack["rt"]
    cm_params = {"thread_priority": int(rt["priority"])}
    if rt.get("cpus"):
        cm_params["cpu_affinity"] = [int(cpu) for cpu in rt["cpus"]]
    # `stack` (tatbot_session) and `config` (tatbot_bridge, tatbot_rerun) name the same file.
    node_params = [{"stack": str(stack_path), "config": str(stack_path)}]
    spawned = [names.JOINT_STATE_BROADCASTER]
    for arm in arms:
        spawned += [names.arm_controller(arm), names.safety_controller(arm)]

    controller_manager = Node(package="controller_manager", executable="ros2_control_node", output="both",
                              parameters=[controllers, cm_params],
                              remappings=[("~/robot_description", names.ROBOT_DESCRIPTION_TOPIC)])
    processes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="both",
             parameters=[{"robot_description": description}]),
        controller_manager,
        Node(package="controller_manager", executable="spawner", output="both",
             arguments=[*spawned, "--param-file", controllers, "--controller-manager-timeout", "60"]),
        Node(package="tatbot_bridge", executable="bridge", name=names.BRIDGE_NODE, output="both",
             parameters=node_params),
        Node(package="tatbot_session", executable="session", name=names.SESSION_NODE, output="both",
             parameters=node_params),
    ]
    processes += _optional_nodes(stack, arms, node_params)
    if stack.get("record"):
        processes.append(ExecuteProcess(output="both", cmd=[
            "ros2", "bag", "record", "--storage", "mcap", "--include-hidden-topics",
            "--output", str(run.dir / "bag"), "--topics", *names.bag_topics(arms)]))

    # The run's exit status: 1 when any process failed before shutdown began. A clean or stop-signal
    # exit is not a failure: systemd's stop signals every process of the unit, which can reach a node
    # before the launch sees its own SIGINT. controller_manager leaving, for any reason, takes the
    # whole launch (and tatbot-ros.service) down with it.
    state = {"stopping": False, "failed": []}

    def exited(event, context):
        if state["stopping"]:
            return None
        if event.returncode not in CLEAN_EXITS:
            state["failed"].append(f"{event.process_name} exit {event.returncode}")
            run.event("process_failed", process=event.process_name, returncode=event.returncode)
        if event.action is controller_manager:
            return [Shutdown(reason=f"controller_manager exited ({event.returncode})")]
        return None

    def shutdown(event, context):
        state["stopping"] = True
        run.finalize(1 if state["failed"] else 0)

    return [
        SetEnvironmentVariable("TATBOT_REPO", str(repo)),
        SetEnvironmentVariable("ROS_LOG_DIR", str(run.dir / "ros")),
        LogInfo(msg=f"ros-stack run {run.run_id}: hardware {stack['hardware']}, arms {','.join(arms)}, "
                    f"estop {stack['estop']['source']}, page {stack['page']['source']}"),
        LogInfo(msg="page geometry from {geometry}: {0:g} x {1:g} mm, clear centre {2:g} x {3:g} mm".format(
            *(round(v * 1000, 3) for v in (*stack["page"]["size_m"], *stack["page"]["clear_m"])), **stack["page"])),
        *(LogInfo(msg=f"{arm}: no adopted registration; base at the nominal urdf mount")
          for arm in arms if arm not in placed),
        *(RegisterEventHandler(OnProcessExit(target_action=process, on_exit=exited)) for process in processes),
        RegisterEventHandler(OnProcessStart(target_action=controller_manager,
                                           on_start=lambda event, context: record_controller(stack['controller_process_file'], event.pid))),
        RegisterEventHandler(OnShutdown(on_shutdown=shutdown)),
        *processes,
    ]


# reached through the ROS 2 launch system as the launch file entry point; not called directly
def generate_launch_description():
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=default, description=text)
          for name, (default, text) in ARGUMENTS.items()),
        OpaqueFunction(function=generate),
    ])
