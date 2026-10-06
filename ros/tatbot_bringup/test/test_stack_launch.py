"""stack.launch.py: its arguments are the contract's, and empty ones fall back to stack.yaml."""
import importlib.util
from pathlib import Path

import pytest
from tatbot_description import names

pytest.importorskip("launch_ros")
LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "stack.launch.py"
spec = importlib.util.spec_from_file_location("stack_launch", LAUNCH)
stack_launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stack_launch)


def test_arguments():
    assert tuple(stack_launch.ARGUMENTS) == (
        "config", "repo", "runtime_record", "arms", "hardware", "profile", "estop_source", "estop_device",
        "estop_udp_port", "estop_relay_addr", "probe", "probe_relay_addr", "machine_addr", "page_source", "pattern_id",
        "touch", "rerun", "record")
    assert all(default == "" for default, _ in stack_launch.ARGUMENTS.values())
    assert set(stack_launch.OVERRIDES) == set(stack_launch.ARGUMENTS) - {"config", "repo", "runtime_record"}


def test_empty_arguments_keep_stack_yaml():
    stack = stack_launch.effective_stack(dict.fromkeys(stack_launch.ARGUMENTS, ""))
    assert (stack["arms"], stack["hardware"], stack["estop"]["source"]) == (["right"], "mock", "udp")
    assert stack["record"] is False


def test_overrides():
    stack = stack_launch.effective_stack({
        "arms": "left,right", "hardware": "fake", "estop_source": "udp", "estop_udp_port": "7641",
        "estop_relay_addr": "192.0.2.9", "estop_device": "/dev/ttyACM3", "page_source": "fixed",
        "touch": "false", "rerun": "true", "record": "true", "profile": "bench", "pattern_id": "stencil-half"})
    assert stack["arms"] == ["left", "right"]
    assert stack["estop"] == {**stack["estop"], "source": "udp", "udp_port": 7641, "relay_addr": "192.0.2.9",
                              "serial_device": "/dev/ttyACM3"}
    assert (stack["hardware"], stack["profile"], stack["page"]["source"]) == ("fake", "bench", "fixed")
    assert stack["page"]["pattern_id"] == "stencil-half"
    assert (stack["touch"]["enabled"], stack["rerun"]["enabled"], stack["record"]) == (False, True, True)
    with pytest.raises(ValueError):
        stack_launch.effective_stack({"touch": "maybe"})


def test_registrations_only_existing(tmp_path):
    path = tmp_path / "reg.json"
    path.write_text("{}")
    stack = {"arms": ["left", "right"], "registration": {"right": str(path), "left": str(tmp_path / "missing.json")}}
    assert stack_launch.registrations(stack) == {"right": str(path)}


def test_bag_topics():
    topics = names.bag_topics(["right"])
    assert "/joint_states" in topics and "/tatbot/safety" in topics
    assert "/right_arm_controller/follow_joint_trajectory/_action/feedback" in topics
    assert len(set(topics)) == len(topics)


def test_config_argument_and_checkout_default(tmp_path):
    path = tmp_path / "stack.yaml"
    path.write_text("arms: [left]\nhardware: fake\n")
    record = str(tmp_path / "runtime.json")
    assert stack_launch.effective_stack({"config": str(path), "runtime_record": record}) == {
        "arms": ["left"], "hardware": "fake", "config_path": str(path), "runtime_record": record}
    default = stack_launch.effective_stack({})
    assert default["format"] == "tatbot-ros-stack"
    assert default["config_path"].endswith("ros/tatbot_bringup/config/stack.yaml")


def test_runtime_record_defaults_to_the_workspace_root(tmp_path):
    """A deployed repo is <root>/repo, so its stack's record is <root>/runtime.json (~/tatbot-ros/runtime.json)."""
    stack = stack_launch.effective_stack({"repo": str(tmp_path / "tatbot-ros" / "repo"),
                                          "config": str(LAUNCH.parents[1] / "config" / "stack.yaml")})
    assert stack["runtime_record"] == str(tmp_path / "tatbot-ros" / "runtime.json")


@pytest.mark.parametrize("hardware,source,raises", [
    ("real", "stencil", True), ("real", "fixed", False), ("mock", "stencil", False), ("fake", "stencil", False)])
def test_real_stencil_needs_a_registration(hardware, source, raises):
    stack = {"arms": ["right"], "hardware": hardware, "page": {"source": source},
             "registration": {"right": "/nonexistent/arm-registration-right-current.json"}}
    placed = stack_launch.registrations(stack)
    if raises:
        with pytest.raises(RuntimeError, match="right"):
            stack_launch.check_placement(stack, placed)
    else:
        stack_launch.check_placement(stack, placed)
    stack_launch.check_placement(stack, {"right": "/some/file.json"})
