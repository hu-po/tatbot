from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot_teleoperator_trossen.config_widowxai_leader import (
    WidowXAILeaderTeleopConfig,
)

from lerobot_robot_tatbot import paths


def _golden_leader_staged() -> list[float]:
    """The follower's staged arm pose from config/trossen/tatbot.yaml, with the
    leader's own gripper at 0 — or upstream's shape if the golden is unreadable."""
    try:
        from lerobot_robot_tatbot import goldens

        pose = goldens.load_tatbot_yaml().get("follower", {}).get("staged_positions")
        if pose and len(pose) == 7:
            return [float(v) for v in pose[:6]] + [0.0]
    except Exception:  # noqa: BLE001 - a missing golden must not break construction
        pass
    return [0.0] * 7


@TeleoperatorConfig.register_subclass("tatbot_leader_teleop")
@dataclass
class TatbotLeaderTeleopConfig(WidowXAILeaderTeleopConfig):
    # The leader stages to the SAME arm pose as the follower (the six joints
    # of config/trossen/tatbot.yaml follower.staged_positions), so the two
    # arms agree the moment tracking starts. Until 2026-08-30 the leader kept
    # upstream's un-rolled default while the follower staged with its wrist
    # rolled +90 deg, and the follower was dragged back to the leader's pose
    # through the max_relative_target clamp as soon as the session went live.
    staged_positions: list[float] = field(default_factory=_golden_leader_staged)

    """WidowX AI leader with tatbot golden-config loading and live tuning.

    The stock plugin runs the leader in pure gravity compensation on whatever
    friction/characteristic state the controller happens to hold (scratch RAM
    that reverts on power cycle). This variant writes the golden
    config/trossen/leader.yaml into the controller at every connect — so the
    tuned leader feel is what every recording session actually gets — and
    registers the leader-feel parameters with the in-process tuning cockpit.
    """

    # Leader address from the hardware profile (TATBOT_LEADER_IP env
    # override > profile driver stanza > empty, which fails at connect).
    ip_address: str = field(default_factory=lambda: paths.driver_default(
        "leader_ip", "TATBOT_LEADER_IP"))

    # Golden arm YAML loaded at connect. Empty = auto (config/trossen/
    # leader.yaml via repo checkout or $TATBOT_CONFIG_DIR); "-" disables.
    arm_config: str = ""

    # Shared hardware e-stop. TatbotLeader and TatbotFollower acquire one
    # process-wide serial monitor, so both arms freeze without competing for
    # heartbeat bytes. Empty string is the explicit hardware-free bench opt-out.
    estop_device: str = field(default_factory=lambda: paths.driver_default(
        "estop_device", "TATBOT_ESTOP_DEVICE"))
    estop_required: bool = True

    # Move every connected arm TOGETHER: each plugin defers its staged
    # move until the first arm is used, and the first arm disconnected
    # lands the whole fleet. lerobot connects and disconnects the robot and
    # the teleoperator one at a time, so without this you watch the arms
    # rise and retract one after the other on every session. Set False to
    # restore strictly per-arm behaviour.
    coordinated_arms: bool = True

    # Tuning cockpit (shared with the follower's server in this process).
    tuning_enabled: bool = True
    tuning_port: int = 8899

    # Load config/trossen/tatbot.yaml's `leader:` section at startup.
    use_tatbot_yaml: bool = True

    # EXPERIMENTAL viscous damping felt by the operator, Nm per rad/s, applied
    # as a commanded external effort opposing joint velocity.
    #
    # None of the three friction-compensation terms damp — they all push in
    # the direction of motion — so nothing in the firmware opposes the
    # stick-slip that makes fine, slow motion lurch. This adds the missing
    # term in software: effort = -leader_damping * velocity on the arm
    # joints (the gripper is left alone).
    #
    # 0.0 disables. Raise in small steps: it should make the arm feel
    # progressively heavier and smoother. If raising it makes the arm MORE
    # lurchy, the effort sign convention is inverted on this firmware —
    # set it back to 0 and flip DAMPING_SIGN in tatbot_leader.py.
    leader_damping: float = 0.0
