"""stack.yaml carries its one format/version pair; controllers.yaml renders under the names module's names."""
import json
from pathlib import Path

import yaml
from tatbot_description import controllers, drawing_arm, names, repo_root

CONFIG = Path(__file__).resolve().parents[1] / "config"


def _rendered(arms=names.ARMS) -> dict:
    return yaml.safe_load(controllers.render_text(CONFIG / "controllers.yaml", arms))


def test_stack_yaml_format():
    stack = yaml.safe_load((CONFIG / "stack.yaml").read_text())
    assert (stack["format"], stack["version"]) == ("tatbot-ros-stack", 1)
    assert stack["estop"]["source"] in names.ESTOP_SOURCE
    assert set(stack["arms"]) <= set(names.ARMS)


def test_the_drawing_arm_is_the_first_arm_stack_yaml_names_and_it_is_the_xarm(tmp_path):
    """The one statement of which arm draws: the entry points' --arm defaults to it, and the CLI reads the same line
    (scripts/lib/tatbot_cli/stack.py). The rig's stack runs the xArm, so the committed file says so."""
    stack = yaml.safe_load((CONFIG / "stack.yaml").read_text())
    assert drawing_arm() == stack["arms"][0] == "xarm"
    other = tmp_path / "ros" / "tatbot_bringup" / "config" / "stack.yaml"
    other.parent.mkdir(parents=True)
    other.write_text(yaml.safe_dump({**stack, "arms": ["right", "xarm"]}))
    assert drawing_arm(tmp_path) == "right"


def test_the_stacks_arms_are_physical_arms_of_the_control_role_and_the_programs():
    from tatbot_contracts import ros_program

    arms = json.loads((repo_root() / "config" / "arms.json").read_text())["arms"]
    assert all(arms[arm]["control_role"] == "arm" for arm in names.ARMS)
    assert ros_program.ARMS == names.ARMS


def test_controllers_match_names():
    config = _rendered()
    assert config["controller_manager"]["ros__parameters"]["update_rate"] == names.CONTROL_RATE_HZ
    assert "per_arm" not in config
    for arm in names.ARMS:
        jtc = config[names.arm_controller(arm)]["ros__parameters"]
        assert tuple(jtc["joints"]) == names.joint_names(arm)
        assert tuple(jtc["command_interfaces"]) == names.JOINT_COMMAND_INTERFACES
        gpio = config[names.safety_controller(arm)]["ros__parameters"]
        assert gpio["gpios"] == [names.safety_gpio(arm)]
        assert tuple(gpio["command_interfaces"][names.safety_gpio(arm)]["interfaces"]) == names.SAFETY_COMMAND_INTERFACES
        assert tuple(gpio["state_interfaces"][names.safety_gpio(arm)]["interfaces"]) == names.SAFETY_STATE_INTERFACES


def test_only_the_driven_arms_get_controllers():
    manager = _rendered(("xarm",))["controller_manager"]["ros__parameters"]
    assert set(manager) == {"update_rate", names.JOINT_STATE_BROADCASTER, names.arm_controller("xarm"),
                            names.safety_controller("xarm")}


def test_jtc_never_fails_a_goal_on_tracking():
    """Every JTC tolerance is 0 (unchecked): the driver's stall guard is the tracking interlock."""
    config = _rendered()
    for arm in names.ARMS:
        jtc = config[names.arm_controller(arm)]["ros__parameters"]
        assert jtc["constraints"] == {"stopped_velocity_tolerance": 0.0, "goal_time": 0.0}
        assert tuple(jtc["state_interfaces"]) == ("position", "velocity")
        assert jtc["set_last_command_interface_value_as_state_on_activation"] is True
        assert jtc["allow_nonzero_velocity_at_trajectory_end"] is False


def test_units_and_env():
    systemd = CONFIG.parent / "systemd"
    stack_unit = (systemd / "tatbot-ros.service.in").read_text()
    for line in ("LimitRTPRIO=95", "LimitMEMLOCK=infinity", "KillSignal=SIGINT", "Requires=tatbot-ros-router.service",
                 "stack.launch.py $TATBOT_ROS_ARGS"):
        assert line in stack_unit
    assert 'ZENOH_CONFIG_OVERRIDE="$ZENOH_ROUTER_OVERRIDE"' in (systemd / "tatbot-ros-router.service.in").read_text()
    env = (CONFIG.parent / "scripts" / "env.sh.in").read_text()
    assert env.count("scouting/multicast/enabled=false") == 2
    assert 'connect/endpoints=[\\"$TATBOT_ROS_ROUTER\\"]' in env and 'listen/endpoints=[\\"$TATBOT_ROS_ROUTER\\"]' in env
