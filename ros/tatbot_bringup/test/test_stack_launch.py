"""stack.launch.py: its arguments are the contract's, and empty ones fall back to stack.yaml."""
import importlib.util
from pathlib import Path

import pytest
import yaml
from tatbot_description import names

pytest.importorskip("launch_ros")
LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "stack.launch.py"
spec = importlib.util.spec_from_file_location("stack_launch", LAUNCH)
stack_launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stack_launch)


def test_arguments():
    assert tuple(stack_launch.ARGUMENTS) == (
        "config", "repo", "runtime_record", "arms", "hardware", "profile", "estop_source", "estop_device",
        "estop_udp_port", "estop_relay_addr", "probe", "probe_relay_addr", "machine_addr", "keypad_from", "page_source",
        "pattern_id", "page_surface", "touch", "rerun", "record", "sim_endpoint", "tuning")
    assert all(default == "" for default, _ in stack_launch.ARGUMENTS.values())
    assert set(stack_launch.OVERRIDES) == set(stack_launch.ARGUMENTS) - {"config", "repo", "runtime_record"}


def test_empty_arguments_keep_stack_yaml():
    stack = stack_launch.effective_stack(dict.fromkeys(stack_launch.ARGUMENTS, ""))
    expected = yaml.safe_load((LAUNCH.parents[1] / "config" / "stack.yaml").read_text())
    assert {key: value for key, value in stack.items() if key not in ("config_path", "runtime_record")} == expected


def test_overrides():
    stack = stack_launch.effective_stack({
        "arms": "left,right", "hardware": "fake", "estop_source": "udp", "estop_udp_port": "7641",
        "estop_relay_addr": "192.0.2.9", "estop_device": "/dev/ttyACM3", "page_source": "fixed",
        "touch": "false", "rerun": "true", "record": "true", "profile": "bench", "pattern_id": "stencil-half",
        "page_surface": "fitted", "keypad_from": "192.0.2.4"})
    assert stack["keypad"] == {**stack["keypad"], "from_addr": "192.0.2.4"}
    assert stack["arms"] == ["left", "right"]
    assert stack["estop"] == {**stack["estop"], "source": "udp", "udp_port": 7641, "relay_addr": "192.0.2.9",
                              "serial_device": "/dev/ttyACM3"}
    assert (stack["hardware"], stack["profile"], stack["page"]["source"]) == ("fake", "bench", "fixed")
    assert stack["page"]["pattern_id"] == "stencil-half"
    assert stack["page"]["surface"]["mode"] == "fitted"
    assert (stack["touch"]["enabled"], stack["rerun"]["enabled"], stack["record"]) == (False, True, True)
    with pytest.raises(ValueError):
        stack_launch.effective_stack({"touch": "maybe"})
    with pytest.raises(ValueError):
        stack_launch.effective_stack({"page_surface": "cylinder"})


def test_registrations_only_existing(tmp_path):
    path = tmp_path / "reg.json"
    path.write_text("{}")
    stack = {"arms": ["left", "right"], "registration": {"right": str(path), "left": str(tmp_path / "missing.json")}}
    assert stack_launch.registrations(stack) == {"right": str(path)}


def test_bag_topics():
    topics = names.bag_topics(["right"])
    assert "/joint_states" in topics and "/tatbot/safety" in topics
    assert "/right_arm_controller/follow_joint_trajectory/_action/feedback" in topics
    assert "/tatbot/draw/_action/feedback" in topics          # a replay's current stroke, phase and arc
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


def test_the_bridge_gets_its_isolated_sdk_and_never_the_helper_that_shadows_it(tmp_path, monkeypatch):
    """rerun-sdk from <stack root>/rerun-sdk (`tatbot ros deploy`) on the bridge's PYTHONPATH alone, and scripts/lib
    off it: its tatbot_rerun.py would shadow the bridge's own package."""
    repo, sdk = tmp_path / "repo", tmp_path / "rerun-sdk"
    (repo / "scripts" / "lib").mkdir(parents=True)
    (repo / "scripts" / "lib" / "tatbot_rerun.py").write_text("")
    (sdk / "rerun_sdk" / "rerun").mkdir(parents=True)
    (sdk / "rerun_sdk" / "rerun" / "__init__.py").write_text("")
    monkeypatch.setenv("PYTHONPATH", f"{repo / 'scripts' / 'lib'}:{tmp_path / 'contracts'}")
    env = stack_launch._rerun({"rerun": {"enabled": True}}, repo)
    assert env["PYTHONPATH"].split(":") == [str(sdk), str(sdk / "rerun_sdk"), str(tmp_path / "contracts")]
    assert stack_launch._rerun({"rerun": {"enabled": False}}, repo) is None
