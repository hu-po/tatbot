import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from lerobot_robot_tatbot import recovery
from lerobot_robot_tatbot.estop import EstopState
from lerobot_robot_tatbot.tatbot_follower import TatbotFollower
from lerobot_robot_trossen.widowxai_follower import WidowXAIFollower

JOINTS = [f"joint_{i}" for i in range(7)]


class FakeEstop:
    def __init__(self, state=EstopState.OK):
        self.state = state

    @property
    def engaged(self):
        return self.state in (EstopState.PRESSED, EstopState.FAULT)


class FakeDriver:
    def __init__(self, positions=None):
        self.positions = list(positions or [0.0] * 7)
        self.commands = []
        self.configured = True

    def get_all_positions(self):
        return list(self.positions)

    def get_is_configured(self):
        return self.configured

    def set_all_modes(self, mode):
        self.commands.append(("all_modes", mode))

    def set_arm_modes(self, mode):
        self.commands.append(("arm_modes", mode))

    def set_all_positions(self, positions, goal_time, blocking):
        values = list(positions)
        self.commands.append(("all_positions", values, goal_time, blocking))
        if goal_time == 0:
            self.positions = values

    def set_arm_positions(self, positions, goal_time, blocking):
        values = list(positions)
        self.commands.append(("arm_positions", values, goal_time, blocking))
        if goal_time == 0:
            self.positions[:6] = values

    def set_all_external_efforts(self, efforts, goal_time, blocking):
        self.commands.append(("all_efforts", list(efforts), goal_time, blocking))

    def set_joint_position(self, joint, value, goal_time, blocking):
        self.commands.append(("joint_position", joint, value, goal_time, blocking))


def test_follower_hold_snaps_stale_command_and_filter_to_measured_pose():
    follower = object.__new__(TatbotFollower)
    follower.config = SimpleNamespace(joint_names=JOINTS)
    follower.driver = FakeDriver([0.1 * i for i in range(7)])
    follower.cameras = {}
    follower._estop = FakeEstop(EstopState.PRESSED)
    follower._estop_hold = None
    follower._cmd_target = dict.fromkeys(JOINTS[:6], 9.0)
    follower._filt_target = dict.fromkeys(JOINTS[:6], 8.0)
    follower._cmd_time = 0.0
    follower._scale_state = {"stale": True}
    follower._watchdog = recovery.TrackingWatchdog()
    follower._overforce_hold = dict.fromkeys(JOINTS, 7.0)
    follower._overforce_t = 1.0
    follower._overforce_n = 10
    follower._pose_history = deque([(1.0, dict.fromkeys(JOINTS, 6.0))])

    present = dict(zip(JOINTS, follower.driver.positions, strict=True))
    follower._enter_estop_hold(present)

    held = {j: present[j] for j in JOINTS[:6]}
    assert follower._cmd_target == held
    assert follower._filt_target == held
    assert follower._scale_state is None
    assert follower._overforce_hold is None
    assert follower._overforce_n == 0
    assert not follower._pose_history
    assert ("arm_positions", list(held.values()), 0.0, False) in follower.driver.commands

    follower.driver.positions[:6] = [0.2] * 6
    follower._estop.state = EstopState.OK
    released = dict(zip(JOINTS, follower.driver.positions, strict=True))
    follower._leave_estop_hold(released)
    assert follower._cmd_target == {j: released[j] for j in JOINTS[:6]}
    assert follower._filt_target == follower._cmd_target
    follower.driver.configured = False


def test_observation_only_cycle_still_enters_and_leaves_follower_hold(monkeypatch):
    """Recording/reset loops can observe without sending a new action."""
    follower = object.__new__(TatbotFollower)
    follower.config = SimpleNamespace(joint_names=JOINTS)
    follower.driver = FakeDriver([0.1 * i for i in range(7)])
    follower.cameras = {}
    follower._estop = FakeEstop(EstopState.PRESSED)
    follower._estop_hold = None
    follower._cmd_target = dict.fromkeys(JOINTS[:6], 9.0)
    follower._filt_target = dict.fromkeys(JOINTS[:6], 8.0)
    follower._cmd_time = 0.0
    follower._scale_state = {"stale": True}
    follower._watchdog = recovery.TrackingWatchdog()
    follower._overforce_hold = dict.fromkeys(JOINTS, 7.0)
    follower._overforce_t = 1.0
    follower._overforce_n = 10
    follower._pose_history = deque()
    follower._pending_stage = None

    # Name the LeRobot base, not a position in the MRO: TatbotFollower also
    # mixes in TatbotArmSession, so __mro__[1] is no longer the class that
    # defines get_observation, and patching it there silently misses.
    monkeypatch.setattr(WidowXAIFollower, "get_observation", lambda self: {"observed": True})

    assert follower.get_observation() == {"observed": True}
    assert follower._estop_hold is not None
    assert any(command[0] == "arm_positions" for command in follower.driver.commands)

    follower.driver.positions[:6] = [0.2] * 6
    follower._estop.state = EstopState.OK
    assert follower.get_observation() == {"observed": True}
    assert follower._estop_hold is None
    assert follower._cmd_target == dict.fromkeys(JOINTS[:6], 0.2)
    follower.driver.configured = False


def test_deferred_staging_never_starts_while_estop_is_engaged(monkeypatch):
    calls = []
    monkeypatch.setattr(recovery.arm_group, "stage_pending", lambda: calls.append(True))

    follower = object.__new__(TatbotFollower)
    follower.driver = FakeDriver()
    follower.driver.configured = False
    follower.cameras = {}
    follower._pending_stage = [1.0] * 7
    follower._estop = FakeEstop(EstopState.PRESSED)
    follower._ensure_staged()

    assert not calls

    follower._estop.state = EstopState.OK
    follower._ensure_staged()
    assert calls == [True]


def test_emergency_landing_refusal_still_tears_down_then_fails_the_run(monkeypatch):
    """A landing refused for a joint past its limits commanded nothing; the follower still
    disconnects its cameras and tears the session down, then fails the run with the refusal."""
    refusal = recovery.JointBeyondLimitsError("follower landing refused, nothing commanded: joint 5")

    def refuse(*args, **kwargs):
        raise refusal

    class DeadSession:
        def set_all_modes(self, mode):
            raise RuntimeError("TCP connection closed unexpectedly")

        def cleanup(self):
            pass

    class Camera:
        closed = False

        def disconnect(self):
            self.closed = True

    monkeypatch.setattr(recovery, "land_arm", refuse)
    follower = object.__new__(TatbotFollower)
    follower.disconnect = lambda: None  # Robot.__del__ would run the real one at gc
    follower.config = SimpleNamespace(coordinated_arms=False, ip_address="192.0.2.2",
                                      staged_positions=[0.0] * 7)
    follower.driver, follower._estop, follower.cameras = DeadSession(), None, {"wrist": Camera()}
    torn_down = []
    follower._teardown = lambda: torn_down.append(True)
    with pytest.raises(recovery.JointBeyondLimitsError) as raised:
        follower._disconnect_owned()
    assert raised.value is refusal
    assert torn_down == [True] and follower.cameras["wrist"].closed


def test_lifecycle_motion_freezes_mid_phase_then_reissues_target():
    driver = FakeDriver()
    estop = FakeEstop()

    def press_cycle():
        time.sleep(0.03)
        estop.state = EstopState.PRESSED
        time.sleep(0.08)
        estop.state = EstopState.OK

    operator = threading.Thread(target=press_cycle)
    operator.start()
    try:
        assert recovery.raise_arms_together(
            [("arm", driver, [1.0] * 7, 6)],
            goal_time=0.08,
            verify=False,
            estop=estop,
        )
    finally:
        operator.join()

    position_commands = [c for c in driver.commands if c[0] == "all_positions"]
    staged_target = [1.0] * 7  # the staged carriage value goes through (2026-08-30)
    targets = [c for c in position_commands if c[1] == staged_target]
    holds = [c for c in position_commands if c[2] == 0.0]
    assert len(targets) == 2, "target must be reissued after release"
    assert holds, "press must override interpolation with a measured-pose hold"


def test_follower_hold_retracts_the_carriage_only_once_staging_seated_it():
    """The e-stop hold freezes the arm at its measured pose and lifts the pen
    along its own axis (the carriage retract, 2026-08-30) — but only once
    staging has seated the carriage. With the E-stop pressed at connect,
    staging is deferred and the carriage is left where it is: the 2026-08-24
    crash was a write to a joint in the wrong state mid-hold, and the same
    rule (nothing is written to a joint the lifecycle has not claimed) is
    what keeps that from coming back in a new shape."""

    class FakeOutputDriver(FakeDriver):
        def get_robot_output(self):
            return SimpleNamespace(
                joint=SimpleNamespace(
                    all=SimpleNamespace(
                        positions=list(self.positions),
                        velocities=[0.0] * len(self.positions),
                        external_efforts=[0.0] * len(self.positions),
                    )
                )
            )

    follower = object.__new__(TatbotFollower)
    # Robot.__del__ calls the real disconnect(), whose e-stop hold would spin
    # forever against a fake that never releases — neutralize it for gc.
    follower.disconnect = lambda: None
    follower.config = SimpleNamespace(
        joint_names=JOINTS,
        carriage_rest_m=0.0,
        carriage_retract_m=0.040,
        carriage_contact_cap_n=15.0,
        carriage_cap_debounce=3,
        carriage_goal_time_s=0.5,
    )
    follower.driver = FakeOutputDriver([0.1 * i for i in range(7)])
    follower.cameras = {}
    follower._estop = FakeEstop(EstopState.PRESSED)
    follower._estop_hold = None
    follower._carriage_target = None
    follower._carriage_sent = None
    follower._contact_trip = False
    follower._contact_n = 0
    follower._pending_stage = list(follower.driver.positions)
    follower._cmd_target = None
    follower._filt_target = None
    follower._cmd_time = None
    follower._scale_state = None
    follower._watchdog = recovery.TrackingWatchdog()
    follower._overforce_hold = None
    follower._overforce_t = 0.0
    follower._overforce_n = 0
    follower._pose_history = deque()

    action = {f"{j}.pos": 9.0 for j in JOINTS}

    # Phase 1: hold before staging completed — freeze only, carriage untouched.
    sent = follower.send_action(dict(action))
    freeze = [c for c in follower.driver.commands if c[0] == "arm_positions"]
    assert len(freeze) == 1  # _enter_estop_hold's snap, nothing after it
    assert [c for c in follower.driver.commands if c[0] == "joint_position"] == []
    held = {f"{j}.pos": 0.1 * i for i, j in enumerate(JOINTS)}
    assert sent == held

    # Phase 2: the same press after staging seated the carriage — the hold
    # retracts the pen (one carriage write), arm joints still untouched, and
    # a second tick inside the hold writes nothing more.
    follower._estop_hold = None
    follower._carriage_target = 0.0
    follower._carriage_sent = 0.0
    sent = follower.send_action(dict(action))
    assert len(
        [c for c in follower.driver.commands if c[0] == "arm_positions"]
    ) == 2
    retracts = [c for c in follower.driver.commands if c[0] == "joint_position"]
    assert retracts == [("joint_position", TatbotFollower.GRIPPER, 0.040, 0.15, False)]
    assert sent == held
    follower.send_action(dict(action))
    assert len([c for c in follower.driver.commands if c[0] == "joint_position"]) == 1
    assert len([c for c in follower.driver.commands if c[0] == "arm_positions"]) == 2


def test_safety_guards_never_move_the_carriage():
    """A guard's rewrite of the goal reaches the ARM only.

    The carriage is the tool's contact axis and the safety layer's alone
    (2026-08-30): a retreat pose or a z-floor clamp that dragged it toward
    `present` would ride the pen back down its axis mid-retract, or lift it
    during ordinary drawing. Mirror image of the gripper-era invariant that a
    guard against pressing too hard must not let go.
    """
    gripper = JOINTS[6]
    follower = object.__new__(TatbotFollower)
    follower.config = SimpleNamespace(joint_names=JOINTS, z_floor_m=0.060)
    follower._z_floor_warned = 0.0
    follower._carriage_target = 0.040  # mid-retract

    # The action's carriage entry is whatever the leader/policy said; the
    # sent one is the target. The arm is commanded from a pose above the
    # floor to one below it, so the z-floor bisection has real work to do.
    commanded = {**dict.fromkeys(JOINTS[:6], 1.0), gripper: 0.011}
    present = {**dict.fromkeys(JOINTS[:6], 0.0), gripper: 0.0152}

    # --- the invariant itself ----------------------------------------------
    retreat = dict.fromkeys(JOINTS, 0.5)          # a full-pose rewrite
    guarded = follower._preserve_carriage(retreat, commanded)
    assert guarded[gripper] == 0.040, "guard must leave the carriage on its target"
    assert all(guarded[j] == 0.5 for j in JOINTS[:6]), "arm must still be rewritten"
    # before staging seats it there is no target, and the action passes through
    follower._carriage_target = None
    assert follower._preserve_carriage(retreat, commanded)[gripper] == commanded[gripper]
    follower._carriage_target = 0.040

    # --- the real z-floor, with the bisection actually running -------------
    # tool z falls as joint_0 rises, so a high command is below the floor and
    # the solve has to scale the arm back.
    follower._kinematics = lambda: object()
    # z falls as joint_0 rises: present (0.0) is at 0.100 m, well above the
    # 0.060 floor; the command (1.0) would reach 0.050 m, below it.
    follower._tool_z = lambda kin, pose: 0.10 - 0.05 * pose[JOINTS[0]]
    clamped = follower._apply_z_floor(commanded, present)
    assert clamped[gripper] == 0.040, "z-floor moved the carriage"
    assert clamped[JOINTS[0]] != commanded[JOINTS[0]], "arm should have been scaled"
    assert follower._tool_z(None, clamped) >= follower.config.z_floor_m - 1e-9

    # A command already above the floor passes through untouched.
    safe = {**dict.fromkeys(JOINTS[:6], 0.0), gripper: 0.011}
    assert follower._apply_z_floor(safe, present) == safe


# A workspace.yaml as the touch-off writes it in the mount frame (2026-08-30).
# The checked-in file is nulled until that touch-off happens on the arm, so
# the floor tests feed this one through the registry instead.
V2_WORKSPACE = {"right": {
    "tip_frame": "right/tool_mount", "carriage_m": 0.0,
    "pen_tip_offset_x": 0.0, "pen_tip_offset_y": 0.0, "pen_tip_offset_z": 0.060,
    "paper_plane_z": 0.0227, "paper_band_mm": None,
    "pivot_point_x": 0.38, "pivot_point_y": -0.23, "pivot_point_z": 0.0227,
    "ee_contact_z": None,
    "touchoff": {"utc": "2026-08-30T12:00:00Z", "session": "synthetic", "n_plate": 0,
                 "n_pad": 9, "cond": 8.0, "residual_mm": 1.0, "holdout_mm": None,
                 "spread_deg": 40.0, "note": ""}}}


def _floor_follower(monkeypatch, **overrides):
    """A follower with just enough wired up to exercise the workspace floor."""
    from lerobot_robot_tatbot import tool_registry
    reg = tool_registry.registry()
    # Keep the synthetic touch-off consistent with the generated tool block.
    tool_id = reg.active_tool_id(tool_registry.REPO)
    workspace = {"right": {**V2_WORKSPACE["right"], "tool_id": tool_id}}
    monkeypatch.setattr(reg, "read_workspace", lambda *a, **k: workspace)
    follower = object.__new__(TatbotFollower)
    cfg = {"joint_names": JOINTS, "z_floor_urdf": "urdf/tatbot.urdf",
           "z_floor_m": None, "z_floor_below_surface_m": 0.010}
    cfg.update(overrides)
    follower.config = SimpleNamespace(**cfg)
    follower._kin = None
    follower._z_floor_warned = 0.0
    follower._tool = tool_registry.stated_tool(tool_id)
    return follower


def test_workspace_floor_measures_the_tool_tip_not_the_gripper(monkeypatch):
    """The floor exists because "the policy is not responsible for depth", so
    it has to bound the thing that touches the work.

    It read ee_gripper_link until 2026-08-26 and never applied the tip offset,
    so its constant carried one pen's length by accident: 0.060 m suited a
    63.7 mm ballpoint and let a 136.3 mm laser's tip go ~73 mm deeper.
    """
    # the real 7th joint name: the floor maps it onto the URDF's carriage
    follower = _floor_follower(monkeypatch, joint_names=JOINTS[:6] + ["left_carriage_joint"])
    kin = follower._kinematics()
    assert kin is not None
    # The sim's pen-down pose over the pad (2026-08-30, tool axis 45 deg
    # forward-down in the carriage frame, wrist rolled +pi/2): the tip sits
    # well below the flange centre. This is the pose the floor protects; at
    # the all-zeros pose the tool points up-and-forward and the comparison
    # would say nothing.
    q = dict(zip(JOINTS[:6], [0.0, 1.037, 0.392, -0.141, 0.0, 1.5707963267948966], strict=True))
    tip = follower._tool_z(kin, q)
    base = kin.link_pose(
        "right/ee_gripper_link", {f"right/{j}": q[j] for j in JOINTS[:6]})[2, 3]
    # 19 mm at this pose: the 45 deg tool puts the tip ahead of the flange
    # more than below it, so the margin is modest — but it must be there.
    assert base - tip > 0.015, "floor is still reading the gripper base"
    # and the carriage lifts the tip along its axis: a retracted pen is higher
    lifted = follower._tool_z(kin, {**q, "left_carriage_joint": 0.040})
    assert lifted - tip > 0.02  # 40 mm of carriage travel at 45 deg to vertical


def test_workspace_floor_follows_the_measured_surface(monkeypatch):
    """The table moves -- it can sit above or below the robot base, and pads
    and skins have thickness -- so the deliberate constant is the clearance
    beneath the measured surface, not an absolute height that goes stale."""
    surface = V2_WORKSPACE["right"]["paper_plane_z"]

    follower = _floor_follower(monkeypatch, z_floor_below_surface_m=0.010)
    assert follower._kinematics() is not None
    assert abs(follower.config.z_floor_m - (surface - 0.010)) < 1e-9

    # A different clearance moves the floor by exactly that much, and nothing
    # assumes the surface is positive.
    deeper = _floor_follower(monkeypatch, z_floor_below_surface_m=0.025)
    assert deeper._kinematics() is not None
    assert abs(deeper.config.z_floor_m - (surface - 0.025)) < 1e-9


def test_required_workspace_floor_rejects_stale_touch_off(monkeypatch):
    from lerobot_robot_tatbot import tool_registry

    class Registry:
        def __init__(self, touched):
            self.touched = touched

        def read_workspace(self, repo):
            return {
                "right": {
                    "tool_id": "picosecond-laser-pen",
                    "touchoff": {
                        "utc": self.touched,
                        "session": "/evidence/touch.json",
                        "n_pad": 9,
                    },
                }
            }

    follower = object.__new__(TatbotFollower)
    follower.config = SimpleNamespace(z_floor_m=0.032, z_floor_max_age_s=4 * 3600)
    current = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    monkeypatch.setattr(tool_registry, "registry", lambda: Registry(current))
    follower._validate_floor_receipt()

    stale = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
    monkeypatch.setattr(tool_registry, "registry", lambda: Registry(stale))
    with pytest.raises(RuntimeError, match="touch-off is 5.0 h old"):
        follower._validate_floor_receipt()


def test_commissioning_velocity_limit_is_written_to_arm_controller() -> None:
    class LimitDriver(FakeDriver):
        def __init__(self):
            super().__init__()
            self.limits = [SimpleNamespace(velocity_max=9.4) for _ in JOINTS]

        def get_joint_limits(self):
            return self.limits

        def set_joint_limits(self, limits):
            self.limits = limits

    follower = object.__new__(TatbotFollower)
    follower.id = "test_follower"
    follower.config = SimpleNamespace(controller_velocity_limit=0.75)
    follower.driver = LimitDriver()
    follower._apply_controller_velocity_limit()

    assert [limit.velocity_max for limit in follower.driver.limits[:6]] == [0.75] * 6
    assert follower.driver.limits[6].velocity_max == 9.4


@pytest.mark.parametrize("mismatch", ["identity", "digest"])
def test_workspace_floor_refuses_a_urdf_built_for_another_tool(monkeypatch, mismatch):
    """The floor reads the tip out of the URDF's generated block. A block for
    the previous tool puts the tip where the tool is not, which is
    under-protection wearing the same reassuring log line."""
    follower = _floor_follower(monkeypatch)
    follower._tool = SimpleNamespace(
        tool_id="other-tool" if mismatch == "identity" else follower._tool.tool_id,
        sha256="deadbeefcafe" if mismatch == "digest" else follower._tool.sha256,
    )
    assert follower._kinematics() is None, "stale tool block must refuse"
    assert follower.config.z_floor_m is None
    assert follower.config.z_floor_below_surface_m is None


def test_unknown_tip_link_is_caught_by_membership_not_by_a_zero(monkeypatch):
    """Unknown links raise; floor setup also verifies the selected tip exists."""
    follower = _floor_follower(monkeypatch)
    kin = follower._kinematics()
    assert kin is not None
    with pytest.raises(ValueError, match='unknown URDF link'):
        kin.link_pose("right/no-such-link")
    assert "right/tattoo_needle" in kin.parent_of
