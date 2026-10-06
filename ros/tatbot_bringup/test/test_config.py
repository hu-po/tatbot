"""stack.yaml carries its one format/version pair; controllers.yaml uses the names module's names."""
from pathlib import Path

import yaml
from tatbot_description import names

CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_stack_yaml_format():
    stack = yaml.safe_load((CONFIG / "stack.yaml").read_text())
    assert (stack["format"], stack["version"]) == ("tatbot-ros-stack", 1)
    assert stack["estop"]["source"] in names.ESTOP_SOURCE
    assert set(stack["arms"]) <= set(names.ARMS)


def test_controllers_match_names():
    config = yaml.safe_load((CONFIG / "controllers.yaml").read_text())
    assert config["controller_manager"]["ros__parameters"]["update_rate"] == names.CONTROL_RATE_HZ
    for arm in names.ARMS:
        jtc = config[names.arm_controller(arm)]["ros__parameters"]
        assert tuple(jtc["joints"]) == names.joint_names(arm)
        assert tuple(jtc["command_interfaces"]) == names.JOINT_COMMAND_INTERFACES
        gpio = config[names.safety_controller(arm)]["ros__parameters"]
        assert gpio["gpios"] == [names.safety_gpio(arm)]
        assert tuple(gpio["command_interfaces"][names.safety_gpio(arm)]["interfaces"]) == names.SAFETY_COMMAND_INTERFACES
        assert tuple(gpio["state_interfaces"][names.safety_gpio(arm)]["interfaces"]) == names.SAFETY_STATE_INTERFACES


def test_jtc_never_fails_a_goal_on_tracking():
    """Every JTC tolerance is 0 (unchecked): the driver's stall guard is the tracking interlock."""
    config = yaml.safe_load((CONFIG / "controllers.yaml").read_text())
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
