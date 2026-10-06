"""Every cross-package ROS name in one place (ros/README.md "Names"). Pure Python, no ROS import.

C++ mirrors the GPIO part in tatbot_hardware/include/tatbot_hardware/names.hpp; a test keeps them equal.
"""
from __future__ import annotations

ARMS = ("left", "right")
DEFAULT_ARMS = ("right",)  # this stack never connects to the left arm unless told to
WORLD = "world"

REVOLUTE = tuple(f"joint_{i}" for i in range(6))
CARRIAGE = "left_carriage_joint"  # the tool rides the left finger carriage (prismatic, metres)


def joint_names(arm: str) -> tuple[str, ...]:
    """The seven controlled joints in controller order: joint_0..joint_5 (rad), then the carriage (m)."""
    return tuple(f"{arm}/{name}" for name in (*REVOLUTE, CARRIAGE))


def base_frame(arm: str) -> str:
    return f"{arm}/base_link"


def tool_mount_frame(arm: str) -> str:
    return f"{arm}/tool_mount"


def tcp_frame(arm: str) -> str:
    return f"{arm}/tcp"


def page_frame(pattern_id: str) -> str:
    return f"page/{pattern_id}"


def fixed_pattern_id(arm: str) -> str:
    """pattern_id of the stack.yaml nominal page when page.source is `fixed`."""
    return f"fixed_{arm}"


# --- ros2_control -------------------------------------------------------------------------
def hardware_name(arm: str) -> str:
    """The <ros2_control name=...> system of one arm."""
    return f"{arm}_arm"


def safety_gpio(arm: str) -> str:
    """The <gpio name=...> carrying the arm's safety state and requests."""
    return f"{arm}_safety"


JOINT_COMMAND_INTERFACES = ("position", "velocity")
JOINT_STATE_INTERFACES = ("position", "velocity", "effort")

# <arm>_safety state interfaces (doubles), in this order.
SAFETY_STATE_INTERFACES = (
    "estop_source",      # 0 none, 1 serial, 2 udp
    "estop_ok",          # 1 released and fresh (always 1 for none), else 0
    "estop_age_s",       # age of the last valid frame; -1 for none or before the first frame
    "probe_triggered",   # 0/1 (M2)
    "latched",           # 1 = holding the latched pose, ignoring commands
    "latch_reason",      # LATCH_* below
    "guard_mode",        # GUARD_* armed now
    "guard_tripped",     # 0/1 until the next accepted unlatch
    "trip_q0", "trip_q1", "trip_q2", "trip_q3", "trip_q4", "trip_q5",  # rad; NaN until a trip
    "trip_q6",           # carriage at the trip, m; NaN until a trip
    "unlatch_ack",       # last unlatch request id the driver processed
    "land_ack",          # last land request id the driver processed
    "landing",           # 0/1
    "landed",            # 1 = verified landed and idle
    "controller_error",  # 0/1
    "rt_period_max_ms",  # longest read-to-read period over the last second
)
# <arm>_safety command interfaces (doubles).
SAFETY_COMMAND_INTERFACES = (
    "guard_mode",  # level: GUARD_*
    "unlatch",     # request id: a new finite value different from the last one is one request
    "land",        # request id, same rule
)

# tatbot_hardware/TatbotArm <param>s (fake and real), all text; tatbot_description.hardware_params()
# fills them from stack.yaml, config/arms.json, config/profiles/<profile>.json, config/trossen/tatbot.yaml.
HARDWARE_PARAMETERS = (
    "arm", "sdk", "ip", "end_effector",
    "estop_source", "estop_device", "estop_timeout_s", "estop_udp_port", "estop_relay_addr", "estop_debounce_frames",
    # stack.yaml probe.*: the station probe's own relay (PRB1); relay_addr resolved like the e-stop's
    "probe_enabled", "probe_udp_port", "probe_timeout_s", "probe_relay_addr",
    "rt_priority", "rt_cpus", "base_frame", "tcp_frame",
    # stack.yaml safety.*, flattened with "_"
    "step_limit_rad", "step_limit_m", "feedback_warmup_s", "stall_error_rad", "stall_time_s", "stall_progress_rad",
    "over_velocity_rad_s", "over_velocity_m_s", "carriage_baseline_samples", "carriage_trip_ticks", "carriage_settle_ticks",
    "carriage_rebaseline_samples",
    "carriage_judge_below_rad_s", "carriage_retract_s", "tip_lag_trip_m", "tip_lag_hold_s", "tip_lag_arm_after_s",
    "contact_force_n", "contact_hold_s", "contact_baseline_s",
    "landing_takeover_s", "landing_staged_s", "landing_sleep_s", "landing_verify_rad", "landing_verify_carriage_m",
    "landing_budget_s",
    # config/trossen/tatbot.yaml follower; staged_positions is 7 comma-separated values
    "carriage_contact_cap_n", "carriage_contact_deflect_m", "carriage_retract_m", "staged_positions",
    "carriage_qualified",  # tatbot.yaml <controller role>: false for the leader, whose carriage never retracts
    "flight_path",  # the ros-stack run directory
)

ESTOP_SOURCE = {"none": 0, "serial": 1, "udp": 2}
LATCH_NONE = 0
LATCH_ESTOP = 1
LATCH_ESTOP_STALE = 2
LATCH_CARRIAGE_CONTACT = 3
LATCH_STALL = 4
LATCH_OVER_VELOCITY = 5
LATCH_GUARD_TIP_LAG = 6
LATCH_GUARD_PROBE = 7
LATCH_CONTROLLER_ERROR = 8
LATCH_DEACTIVATED = 9
LATCH_STEP_REFUSED = 10
GUARD_NONE = 0
GUARD_TIP_LAG = 1
GUARD_PROBE = 2

# --- controllers --------------------------------------------------------------------------
CONTROL_RATE_HZ = 400
JOINT_STATE_BROADCASTER = "joint_state_broadcaster"


def arm_controller(arm: str) -> str:
    """joint_trajectory_controller/JointTrajectoryController, position + velocity commands."""
    return f"{arm}_arm_controller"


def safety_controller(arm: str) -> str:
    """gpio_controllers/GpioCommandController over <arm>_safety."""
    return f"{arm}_safety_controller"


def follow_joint_trajectory(arm: str) -> str:
    return f"/{arm_controller(arm)}/follow_joint_trajectory"


def safety_commands_topic(arm: str) -> str:
    """control_msgs/msg/DynamicInterfaceGroupValues in; interface_groups=[<arm>_safety]."""
    return f"/{safety_controller(arm)}/commands"


def safety_states_topic(arm: str) -> str:
    """control_msgs/msg/DynamicInterfaceGroupValues out."""
    return f"/{safety_controller(arm)}/gpio_states"


# --- tatbot graph -------------------------------------------------------------------------
DRAW_ACTION = "/tatbot/draw"      # tatbot_interfaces/action/Draw
TOUCH_ACTION = "/tatbot/touch"    # tatbot_interfaces/action/Touch
LAND_ACTION = "/tatbot/land"      # tatbot_interfaces/action/Land
DECIDE_SERVICE = "/tatbot/decide"  # tatbot_interfaces/srv/Decide
EVENTS_TOPIC = "/tatbot/events"   # tatbot_interfaces/msg/Event
SAFETY_TOPIC = "/tatbot/safety"   # tatbot_interfaces/msg/SafetyState, one per arm
PAGE_TOPIC = "/tatbot/page"       # tatbot_interfaces/msg/Page
EE_TRACKING_FRAME = "{arm}/gripper_left"   # the frame the overhead fiducial tracker estimates (visiond)


def ee_topic(arm: str) -> str:
    """geometry_msgs/PoseWithCovarianceStamped: the overhead-tracked EE fiducial pose in <arm>/base_link."""
    return f"/tatbot/ee/{arm}"


KEYS = ("up", "down", "reset", "enter")   # the operator's keys: the pen trim a step up or down, back to 0; continue


def keys_topic(arm: str) -> str:
    """std_msgs/String, one of KEYS per press: the operator's numpad (tatbot_session.keypad) for this arm."""
    return f"/tatbot/keys/{arm}"


JOINT_STATES_TOPIC = "/joint_states"
ROBOT_DESCRIPTION_TOPIC = "/robot_description"

# Nodes.
SESSION_NODE = "tatbot_session"
BRIDGE_NODE = "tatbot_bridge"
RERUN_NODE = "tatbot_rerun"
KEYPAD_NODE = "tatbot_keypad"

# Run-log workflows (~/tatbot-logs/<workflow>/<run-id>/).
RUN_WORKFLOW_DRAW = "ros-draw"
RUN_WORKFLOW_TOUCH = "ros-touch"
RUN_WORKFLOW_STACK = "ros-stack"


def bag_topics(arms) -> tuple[str, ...]:
    """What an MCAP of the stack records: the stack-wide bag (launch `record`) and each run's bag/.

    JTC's controller_state carries the reference it tracks, so the executed goals are in the bag;
    the action feedback and status topics are hidden (`_action`) and need --include-hidden-topics.
    """
    per_arm = []
    for arm in arms:
        per_arm += [f"/{arm_controller(arm)}/controller_state", f"{follow_joint_trajectory(arm)}/_action/feedback",
                    f"{follow_joint_trajectory(arm)}/_action/status", safety_states_topic(arm), ee_topic(arm)]
    return (JOINT_STATES_TOPIC, "/tf", "/tf_static", ROBOT_DESCRIPTION_TOPIC, EVENTS_TOPIC, SAFETY_TOPIC, PAGE_TOPIC,
            *per_arm)
