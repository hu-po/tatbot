"""Tests for the golden configs and the landing/recovery paths.

Runs entirely against fake drivers — no hardware, no lerobot imports.
"""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import trossen_arm
import yaml
from lerobot_robot_tatbot import goldens

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "config" / "trossen"
N = 7


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDriver:
    """Holds parameter groups like the real driver; same getter/setter names."""

    def __init__(self):
        self.motor_parameters = [
            {trossen_arm.Mode.position: _FakeMotor(kp, 8.0 if j < 3 else 1.0)}
            for j, kp in enumerate([120, 120, 120, 80, 40, 40, 20])
        ]
        self.joint_limits = [_FakeLimit() for _ in range(N)]
        self.characteristics = [_FakeCharacteristic() for _ in range(N)]
        self.algorithm = _FakeAlgo()

    def get_motor_parameters(self):
        return self.motor_parameters

    def set_motor_parameters(self, mp):
        self.motor_parameters = mp

    def get_joint_limits(self):
        return self.joint_limits

    def set_joint_limits(self, jl):
        self.joint_limits = list(jl)

    def get_joint_characteristics(self):
        return self.characteristics

    def set_joint_characteristics(self, jc):
        self.characteristics = list(jc)

    def get_algorithm_parameter(self):
        return self.algorithm

    def set_algorithm_parameter(self, ap):
        self.algorithm = ap


class _FakePID:
    def __init__(self, kp):
        self.kp, self.ki, self.kd, self.imax = float(kp), 0.0, 0.0, 0.0


class _FakeMotor:
    def __init__(self, pos_kp, vel_kp):
        self.position = _FakePID(pos_kp)
        self.velocity = _FakePID(vel_kp)


class _FakeLimit:
    def __init__(self):
        self.position_min, self.position_max = -3.14, 3.14
        self.position_tolerance = 0.2
        self.velocity_max, self.velocity_tolerance = 6.28, 0.0
        self.effort_max, self.effort_tolerance = 27.0, 5.4


class _FakeCharacteristic:
    def __init__(self):
        self.effort_correction = 1.0
        self.friction_transition_velocity = 0.02
        self.friction_constant_term = 0.0
        self.friction_coulomb_coef = 0.0
        self.friction_viscous_coef = 0.0
        self.position_offset = 0.0


class _FakeAlgo:
    def __init__(self):
        self.singularity_threshold = 0.0


@dataclass
class FakeFollowerConfig:
    ip_address: str = "192.0.2.2"
    loop_rate: int = 30
    min_time_to_move_multiplier: float = 3.0
    target_filter_tau: float = 0.0
    max_joint_velocity: float = 2.0
    max_relative_target: float = 0.5
    carriage_rest_m: float = 0.0
    carriage_retract_m: float = 0.032
    carriage_contact_cap_n: float = 15.0
    carriage_cap_debounce: int = 3
    carriage_goal_time_s: float = 0.5
    motion_scale: list = field(default_factory=lambda: [1.0] * 6)
    staged_positions: list = field(default_factory=lambda: [0.0] * 7)
    include_velocity: bool = False
    include_effort: bool = False
    include_external_effort: bool = True
    estop_device: str = ""
    estop_required: bool = False
    flight_log_dir: str = ""
    use_tatbot_yaml: bool = True

    def __post_init__(self):
        self.carriage_contact_cap_n = min(max(self.carriage_contact_cap_n, 2.0), 40.0)


# ---------------------------------------------------------------------------
# Goldens
# ---------------------------------------------------------------------------


def test_tatbot_yaml_merge_respects_cli_overrides(tmp_path):
    cfg = FakeFollowerConfig(target_filter_tau=0.3)  # "CLI override"
    section = {"target_filter_tau": 0.1, "carriage_contact_cap_n": 12.0, "bogus_key": 1}
    applied = goldens.apply_section(cfg, section)
    assert cfg.target_filter_tau == 0.3  # CLI wins
    assert cfg.carriage_contact_cap_n == 12.0  # default was untouched → yaml applies
    assert "carriage_contact_cap_n" in applied and "target_filter_tau" not in applied


def test_repo_tatbot_yaml_matches_defaults():
    """The checked-in tatbot.yaml must mirror the REAL dataclass defaults so
    the first golden load is a no-op (no surprise parameter jumps), and so
    the CLI-override heuristic (value != default) stays meaningful. Imported
    lazily: it pulls lerobot, which the rest of this module avoids."""
    from lerobot_robot_tatbot.config_tatbot_follower import TatbotFollowerConfig

    doc = yaml.safe_load((CONFIG_DIR / "tatbot.yaml").read_text())
    cfg = TatbotFollowerConfig(id="schema_check")
    for key, val in doc["follower"].items():
        if key == 'carriage_qualified':
            continue  # native-worker qualification is not a LeRobot configuration field
        assert getattr(cfg, key) == val, (
            f"tatbot.yaml {key}={val} but the dataclass default is "
            f"{getattr(cfg, key)} — change both together"
        )


def test_carriage_constants_match_cpp_teleop():
    """The C++ teleop carries its own copy of the carriage constants (rest,
    retract, contact cap) until it reads tatbot.yaml; a change on one side
    must move the other. scripts/check_tool_sync.py checks the same thing
    from the scripts side."""
    src = (REPO / "cpp" / "teleop" / "wxai_teleop.cpp").read_text()
    doc = yaml.safe_load((CONFIG_DIR / "tatbot.yaml").read_text())["follower"]
    for pattern, key in [
        (r"CARRIAGE_REST_M = ([\d.]+)", "carriage_rest_m"),
        (r"CARRIAGE_RETRACT_M = ([\d.]+)", "carriage_retract_m"),
        (r"CARRIAGE_CONTACT_CAP_N = ([\d.]+)", "carriage_contact_cap_n"),
    ]:
        m = re.search(pattern, src)
        assert m, f"could not find {key} in wxai_teleop.cpp"
        assert float(m.group(1)) == pytest.approx(doc[key]), (
            f"{key}: C++ has {m.group(1)}, tatbot.yaml has {doc[key]}"
        )


def test_follower_yaml_has_follower_end_effector():
    lead = yaml.safe_load((CONFIG_DIR / "leader.yaml").read_text())
    foll = yaml.safe_load((CONFIG_DIR / "follower.yaml").read_text())
    assert foll["end_effector"]["palm"]["mass"] != lead["end_effector"]["palm"]["mass"]
    # The goldens and the hardware profile must agree about which arm is which:
    # a mismatch here means one of them was edited alone.
    import json as _json

    profile = REPO / "config/profiles/tatbot.json"
    if profile.is_file():
        driver = _json.loads(profile.read_text()).get("driver", {})
        assert foll["manual_ip"] == driver.get("follower_ip")
        assert lead["manual_ip"] == driver.get("leader_ip")


# ---------------------------------------------------------------------------
# Robustness: golden apply, emergency park
# ---------------------------------------------------------------------------


def test_apply_arm_golden_via_setters():
    driver = FakeDriver()
    applied = goldens.apply_arm_golden(
        driver, trossen_arm, CONFIG_DIR / "follower.yaml")
    assert set(applied) == {
        "joint_characteristics", "joint_limits", "motor_parameters",
        "algorithm_parameter",
    }
    # friction table landed on the characteristic objects
    assert driver.characteristics[0].friction_constant_term == pytest.approx(0.24, abs=1e-4)
    assert driver.characteristics[6].friction_constant_term == pytest.approx(7.0, abs=1e-4)
    assert driver.characteristics[3].position_offset == 0.0
    # kp table landed in position mode; other modes untouched (fake has only position)
    mode = trossen_arm.Mode.position
    assert driver.motor_parameters[0][mode].position.kp == 120.0
    assert driver.motor_parameters[6][mode].position.kp == 20.0  # stock: the lead screw is self-locking, compliance was tried and reverted 2026-08-30
    # limits landed
    assert driver.joint_limits[6].effort_max == 200.0
    assert driver.algorithm.singularity_threshold == pytest.approx(0.01)


class _FakeLandDriver:
    """Stands in for a fresh TrossenArmDriver during a landing."""

    instances: list = []

    def __init__(self, fail_configures=0, lands=True, error="No error"):
        self.calls = []
        self.moves = []
        self.fail_configures = fail_configures
        self.configures = 0
        self.lands = lands
        self.error = error
        self.positions = [0.3, 0.9, 0.4, 0.5, 0.1, 0.2, 0.0175]  # gripper gripping
        _FakeLandDriver.instances.append(self)

    def configure(self, model, ee, ip, clear_error, timeout=None):
        self.configures += 1
        self.timeout = timeout
        if self.configures <= self.fail_configures:
            raise RuntimeError("controller still rebooting")
        self.calls.append(f"configure(clear_error={clear_error})")

    def get_is_configured(self):
        return True

    def get_error_information(self):
        return self.error

    def get_all_positions(self):
        return list(self.positions)

    def set_all_modes(self, mode):
        self.calls.append(f"modes:{mode}")

    def set_all_positions(self, positions, goal_time, blocking):
        self.calls.append(f"move(goal_time={goal_time})")
        self.moves.append(list(positions))
        if self.lands:
            self.positions = list(positions)  # the arm actually executes

    def cleanup(self):
        self.calls.append("cleanup")


@pytest.fixture
def land(monkeypatch):
    """Patch the driver constructor land_arm uses; return the instance list."""
    from lerobot_robot_tatbot import recovery

    _FakeLandDriver.instances = []
    monkeypatch.setattr(recovery, "RETRY_DELAY_S", 0.0)
    monkeypatch.setattr(recovery, "MEASUREMENT_SETTLE_S", 0.0)  # a fake's first reading is already live
    monkeypatch.setattr(recovery.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(recovery.trossen_arm, "TrossenArmDriver", _FakeLandDriver)
    return _FakeLandDriver.instances


def test_landing_sequence_returns_carriage_to_rest(land):
    """Take over at measured carriage, sweep staged, then close it at sleep."""
    from lerobot_robot_tatbot import recovery

    staged = [0.0, 1.047, 0.524, 0.628, 0.0, 0.0, 0.0]
    assert recovery.land_arm("192.0.2.2", object(), staged,
                             name="follower", gripper_index=6)
    drv = land[0]
    assert drv.calls == [
        "configure(clear_error=True)",
        f"modes:{trossen_arm.Mode.position}",
        f"move(goal_time={recovery.TAKEOVER_S})",
        f"move(goal_time={recovery.STAGED_POSE_S})",
        f"move(goal_time={recovery.SLEEP_POSE_S})",
        f"modes:{trossen_arm.Mode.idle}",
        "cleanup",
    ]
    # Takeover and staged sweep retain the measured retract; sleep closes it.
    assert drv.moves[0][6] == pytest.approx(0.0175)
    assert drv.moves[1][6] == pytest.approx(0.0175)
    assert drv.moves[2][6] == pytest.approx(0.0)
    assert drv.moves[-1][:6] == [0.0] * 6      # arm at sleep (staged wrist roll 0 here)
    assert drv.timeout == recovery.CONFIGURE_TIMEOUT_S


def test_landing_verifies_it_actually_landed(land):
    """A controller that accepts commands without executing them must be
    reported as a FAILED landing, not a success."""
    from lerobot_robot_tatbot import recovery

    _FakeLandDriver.instances = []
    import lerobot_robot_tatbot.recovery as rec
    rec.trossen_arm.TrossenArmDriver = lambda: _FakeLandDriver(lands=False)
    assert recovery.land_arm("192.0.2.2", object(), [0.0] * N,
                             name="follower", attempts=1) is False


def test_landing_rejects_a_carriage_that_did_not_return_to_rest(land):
    from lerobot_robot_tatbot import recovery

    class _StuckCarriage(_FakeLandDriver):
        def set_all_positions(self, positions, goal_time, blocking):
            carriage = self.positions[6]
            super().set_all_positions(positions, goal_time, blocking)
            self.positions[6] = carriage

    recovery.trossen_arm.TrossenArmDriver = _StuckCarriage
    assert recovery.land_arm("192.0.2.2", object(), [0.0] * N,
                             name="follower", attempts=1) is False


def test_landing_retries_then_succeeds(land):
    from lerobot_robot_tatbot import recovery

    _FakeLandDriver.instances = []
    import lerobot_robot_tatbot.recovery as rec
    made = []
    def factory():
        d = _FakeLandDriver(fail_configures=1 if not made else 0)
        made.append(d)
        return d
    rec.trossen_arm.TrossenArmDriver = factory
    assert recovery.land_arm("192.0.2.2", object(), [0.0] * N, name="leader")
    assert len(made) >= 2, "a failed attempt must use a FRESH driver session"


def test_landing_refuses_while_estop_engaged(land):
    """The emergency path must never drive a latched arm."""
    from lerobot_robot_tatbot import recovery

    class _Estop:
        engaged = True
        state = type("S", (), {"value": "pressed"})()

    assert recovery.land_arm("192.0.2.2", object(), [0.0] * N,
                             name="follower", estop=_Estop()) is False
    assert not land, "no driver session may be opened while e-stopped"


def test_sigint_shield_swallows_then_escapes():
    """First Ctrl+C during a landing is swallowed; a later one gets through
    so a genuinely hung landing stays interruptible."""
    from lerobot_robot_tatbot import recovery

    shield = recovery.SigintShield(escape_s=5.0)
    shield._handler(2, None)          # first press: swallowed
    assert shield._first > 0
    shield._first -= 1.0              # 1 s later: still swallowed
    shield._handler(2, None)
    shield._first -= 10.0             # past the escape window
    with pytest.raises(KeyboardInterrupt):
        shield._handler(2, None)


class _FakeHealthDriver:
    def __init__(self, error="No error", positions=None):
        self.error = error
        self.positions = positions if positions is not None else [0.0] * N

    def get_is_configured(self):
        return True

    def get_error_information(self):
        return self.error

    def get_all_positions(self):
        return list(self.positions)


def test_preflight_rejects_firmware_error():
    from lerobot_robot_tatbot import recovery

    drv = _FakeHealthDriver(error="joint 3 following error")
    with pytest.raises(RuntimeError, match="firmware error"):
        recovery.assert_controller_healthy(drv, [0.0] * N, "follower")


def test_preflight_rejects_unexecuted_staged_move():
    from lerobot_robot_tatbot import recovery

    staged = [0.0, 1.047, 0.524, 0.628, 0.0, 0.0, 0.0]
    frozen = [0.0] * N  # commanded staged, never moved
    drv = _FakeHealthDriver(positions=frozen)
    with pytest.raises(RuntimeError, match="did not reach the staged pose"):
        recovery.assert_controller_healthy(drv, staged, "follower")
    # healthy: at staged within tolerance
    drv2 = _FakeHealthDriver(positions=[v + 0.05 for v in staged])
    recovery.assert_controller_healthy(drv2, staged, "follower")


def test_tracking_watchdog_aborts_when_arm_does_not_move():
    """The observed freeze: commanded away from present, error pinned at the
    max_relative_target clamp, arm stationary."""
    from lerobot_robot_tatbot import recovery

    wd = recovery.TrackingWatchdog(threshold_rad=0.35, grace_s=2.0)
    drv = _FakeHealthDriver(error="velocity fault")
    frozen = [0.0, 1.047, 0.524, 0.6288, 0.0, 0.0]  # joint_3 stuck at staged
    wd.update(0.05, frozen, now=0.0)             # healthy
    wd.update(0.50, frozen, now=1.0)             # error window opens
    wd.update(0.50, frozen, now=2.5)             # inside grace
    with pytest.raises(RuntimeError, match="not executing motion.*velocity fault"):
        wd.update(0.50, frozen, now=3.2, driver=drv)


def test_tracking_watchdog_tolerates_fast_teleop():
    """A healthy arm outrun by the operator pins the SAME 0.5 rad error (the
    clamp saturates) but keeps moving — it must never abort."""
    from lerobot_robot_tatbot import recovery

    wd = recovery.TrackingWatchdog(threshold_rad=0.35, grace_s=2.0)
    pos = [0.0] * 6
    t = 0.0
    for _ in range(300):  # 10 s of sustained fast motion at 30 Hz
        t += 1 / 30
        pos = [p + 2.0 / 30 for p in pos]  # slewing at max_joint_velocity
        wd.update(0.5, pos, now=t)  # clamp saturated the whole time


def test_tracking_watchdog_resets_when_tracking_recovers():
    from lerobot_robot_tatbot import recovery

    wd = recovery.TrackingWatchdog(threshold_rad=0.35, grace_s=2.0)
    still = [0.0] * 6
    wd.update(0.50, still, now=0.0)
    wd.update(0.05, still, now=1.0)   # tracked again — window closes
    wd.update(0.50, still, now=10.0)  # new window
    wd.update(0.50, still, now=11.0)  # only 1 s in: no raise


def test_goldens_match_pinned_sdk_schema():
    """The arm goldens are also loaded by the C++ teleop via
    load_configs_from_file (SDK v1.8.5), whose YAML schema is strict: an
    unknown key fails the whole load. Keep the golden keys to exactly what
    the pinned driver's JointCharacteristic knows — position_offset is
    1.8.8+ only and broke this once already.
    """
    allowed = {f for f in dir(trossen_arm.JointCharacteristic)
               if not f.startswith("_")}
    for name in ("leader.yaml", "follower.yaml"):
        doc = yaml.safe_load((CONFIG_DIR / name).read_text())
        for i, jc in enumerate(doc["joint_characteristics"]):
            unknown = set(jc) - allowed
            assert not unknown, f"{name} joint {i}: {unknown} not in {allowed}"


def test_healthy_controller_string_is_not_an_error():
    """A healthy controller returns the literal string 'No error' from
    get_error_information() — not ''. Treating any non-empty string as a
    fault aborted every healthy session on 2026-08-20."""
    from lerobot_robot_tatbot import recovery

    for clean in ("No error", "no error", "  No error  ", "none", ""):
        drv = _FakeHealthDriver(error=clean)
        assert recovery.controller_error(drv) == "", clean
        # and the preflight must pass with a matching pose
        recovery.assert_controller_healthy(drv, [0.0] * N, "arm")

    real = _FakeHealthDriver(error="Joint limit exceeded")
    assert recovery.controller_error(real) == "Joint limit exceeded"
    with pytest.raises(RuntimeError, match="firmware error"):
        recovery.assert_controller_healthy(real, [0.0] * N, "arm")


def test_controller_error_skips_unconfigured_driver():
    """Reading error state off an arm we never connected must be silent."""
    from lerobot_robot_tatbot import recovery

    class _Unconfigured:
        def get_is_configured(self):
            return False

        def get_error_information(self):
            raise RuntimeError("This TrossenArmDriver is not configured")

    assert recovery.controller_error(_Unconfigured()) == ""


# ---------------------------------------------------------------------------
# Simultaneous landing
# ---------------------------------------------------------------------------


class _CoLandDriver:
    """Records the order and timing of landing commands."""

    def __init__(self, log, name, positions=None, fail_on=None):
        self.log = log
        self.name = name
        self.positions = positions or [0.3, 0.9, 0.4, 0.5, 0.1, 0.2, 0.0175]
        self.fail_on = fail_on
        self.moves = []

    def get_all_positions(self):
        return list(self.positions)

    def set_all_modes(self, mode):
        self.log.append((self.name, f"mode:{mode}"))

    def set_all_positions(self, positions, goal_time, blocking):
        if self.fail_on is not None and len(self.moves) == self.fail_on:
            raise RuntimeError("session dead")
        assert blocking is False, "coordinated landing must never block"
        self.log.append((self.name, f"move:{goal_time}"))
        self.moves.append(list(positions))
        self.positions = list(positions)


def test_coordinated_landing_interleaves_and_never_blocks(monkeypatch):
    """Each phase must be issued to EVERY arm before waiting, so the arms
    move together. Threads can't do this — the driver holds the GIL."""
    from lerobot_robot_tatbot import recovery

    monkeypatch.setattr(recovery.time, "sleep", lambda s: None)
    log = []
    staged = [0.0, 1.047, 0.524, 0.628, 0.0, 0.0, 0.0]
    arms = [("leader", _CoLandDriver(log, "leader"), staged, 6),
            ("follower", _CoLandDriver(log, "follower"), staged, 6)]
    assert recovery.land_arms_together(arms)

    moves = [entry for entry in log if entry[1].startswith("move:")]
    # phase-major order: both arms get phase 1 before either gets phase 2
    assert moves[0][0] != moves[1][0], "second arm did not get phase 1 first"
    assert moves[0][1] == moves[1][1], "arms are in different phases"
    assert moves[2][1] == moves[3][1] and moves[2][1] != moves[0][1]
    # carriage: keep measured through takeover/staged, then return to rest
    for _, drv, _, _ in arms:
        assert drv.moves[0][6] == pytest.approx(0.0175)
        assert drv.moves[1][6] == pytest.approx(0.0175)
        assert drv.moves[2][6] == pytest.approx(0.0)
        assert drv.moves[-1][:6] == [0.0] * 6


def test_coordinated_landing_survives_one_dead_arm(monkeypatch):
    from lerobot_robot_tatbot import recovery

    monkeypatch.setattr(recovery.time, "sleep", lambda s: None)
    log = []
    staged = [0.0] * N
    good = _CoLandDriver(log, "leader")
    dead = _CoLandDriver(log, "follower", fail_on=1)  # dies after phase 1
    recovery.land_arms_together([("leader", good, staged, 6),
                                 ("follower", dead, staged, 6)])
    # the healthy arm still completes all three phases
    assert len(good.moves) == 3
    assert good.moves[-1][:6] == [0.0] * 6


def test_coordinated_landing_reports_unexecuted_moves(monkeypatch):
    """An arm that accepts commands without moving must be reported."""
    from lerobot_robot_tatbot import recovery

    monkeypatch.setattr(recovery.time, "sleep", lambda s: None)

    class _Frozen(_CoLandDriver):
        def set_all_positions(self, positions, goal_time, blocking):
            self.log.append((self.name, f"move:{goal_time}"))
            self.moves.append(list(positions))  # accepted but never executed

    log = []
    frozen = _Frozen(log, "follower")
    assert recovery.land_arms_together(
        [("follower", frozen, [0.0] * N, 6)]) is False


def test_coordinated_landing_rejects_a_stuck_carriage(monkeypatch):
    from lerobot_robot_tatbot import recovery

    monkeypatch.setattr(recovery.time, "sleep", lambda s: None)

    class _StuckCarriage(_CoLandDriver):
        def set_all_positions(self, positions, goal_time, blocking):
            carriage = self.positions[6]
            super().set_all_positions(positions, goal_time, blocking)
            self.positions[6] = carriage

    stuck = _StuckCarriage([], "follower")
    assert recovery.land_arms_together(
        [("follower", stuck, [0.0] * N, 6)]) is False


def test_coordinated_lift_is_interleaved_and_holds_gripper(monkeypatch):
    """Startup mirror of the landing: every arm gets its staged move posted
    before anything waits, so they rise together."""
    from lerobot_robot_tatbot import recovery

    monkeypatch.setattr(recovery.time, "sleep", lambda s: None)
    log = []
    staged = [0.0, 1.047, 0.524, 0.628, 0.0, 0.0, 0.0]
    a = _CoLandDriver(log, "leader", positions=[0.0] * 6 + [0.021])
    b = _CoLandDriver(log, "follower", positions=[0.0] * 6 + [0.019])
    assert recovery.raise_arms_together(
        [("leader", a, staged, 6), ("follower", b, staged, 6)])

    moves = [e for e in log if e[1].startswith("move:")]
    assert len(moves) == 2 and moves[0][0] != moves[1][0]
    # the carriage is driven to the staged (rest) value: nothing is gripped
    # since 2026-08-30, and a retract left by a trip must not survive a lift
    assert a.moves[0][6] == pytest.approx(0.0)
    assert b.moves[0][6] == pytest.approx(0.0)
    assert a.moves[0][:6] == staged[:6]


def test_coordinated_lift_reports_arm_that_did_not_rise(monkeypatch):
    from lerobot_robot_tatbot import recovery

    monkeypatch.setattr(recovery.time, "sleep", lambda s: None)

    class _Frozen(_CoLandDriver):
        def set_all_positions(self, positions, goal_time, blocking):
            self.moves.append(list(positions))  # accepted, never executed

    frozen = _Frozen([], "follower", positions=[0.0] * 7)
    assert recovery.raise_arms_together(
        [("follower", frozen, [0.0, 1.047, 0.5, 0.6, 0.0, 0.0, 0.0], 6)]) is False


def test_coordinated_arms_on_by_default():
    """The follower shares the coordinated lift/land unless explicitly disabled."""
    import os
    os.environ.setdefault("TATBOT_CONFIG_DIR", str(CONFIG_DIR))
    from lerobot_robot_tatbot.config_tatbot_follower import TatbotFollowerConfig
    from lerobot_robot_tatbot.tatbot_follower import TatbotFollower

    # This test inspects class defaults and lifecycle methods; there is no
    # mounted tool or calibration fixture in the disconnected fake.
    follower = TatbotFollower(TatbotFollowerConfig(id="t", use_tool_registry=False))
    assert follower.config.coordinated_arms is True
    assert hasattr(follower, "finish_staging")
    assert hasattr(follower, "_ensure_staged")


class _GroupPlugin:
    """Minimal stand-in for a plugin registered with the ArmGroup."""

    def __init__(self, name, log):
        self.name = name
        self.log = log
        self.driver = _CoLandDriver(log, name)
        self.finished = 0

    def finish_staging(self):
        self.finished += 1
        self.log.append((self.name, "finish_staging"))


def test_arm_group_lifts_everyone_on_first_use(monkeypatch):
    """lerobot connects the arms one at a time; whichever is used first must
    lift the whole fleet together."""
    from lerobot_robot_tatbot import recovery

    monkeypatch.setattr(recovery.time, "sleep", lambda s: None)
    group = recovery.ArmGroup()
    log = []
    staged = [0.0, 1.047, 0.5, 0.6, 0.0, 0.0, 0.0]
    a, b = _GroupPlugin("leader", log), _GroupPlugin("follower", log)
    group.register("leader", a, a.driver, staged, 6)
    group.register("follower", b, b.driver, staged, 6)

    assert group.stage_pending()
    moves = [e for e in log if e[1].startswith("move:")]
    assert len(moves) == 2 and moves[0][0] != moves[1][0]  # interleaved
    assert a.finished == 1 and b.finished == 1

    # a second trigger is a no-op — arms must not be re-lifted
    group.stage_pending()
    assert a.finished == 1 and b.finished == 1


def test_arm_group_lands_everyone_once(monkeypatch):
    """The first disconnect lands the fleet; later ones must not re-land."""
    from lerobot_robot_tatbot import recovery

    monkeypatch.setattr(recovery.time, "sleep", lambda s: None)
    group = recovery.ArmGroup()
    log = []
    a, b = _GroupPlugin("leader", log), _GroupPlugin("follower", log)
    group.register("leader", a, a.driver, [0.0] * N, 6)
    group.register("follower", b, b.driver, [0.0] * N, 6)
    group.stage_pending()
    log.clear()

    group.land()
    assert group.has_landed()
    moves_first = len([e for e in log if e[1].startswith("move:")])
    assert moves_first == 6, "3 phases x 2 arms"
    group.land()  # second disconnect
    assert len([e for e in log if e[1].startswith("move:")]) == moves_first


def test_arm_group_unregister_clears_state():
    from lerobot_robot_tatbot import recovery

    group = recovery.ArmGroup()
    log = []
    a = _GroupPlugin("leader", log)
    group.register("leader", a, a.driver, [0.0] * N, 6)
    assert group.names() == ["leader"]
    group.unregister("leader")
    assert group.names() == [] and group.land() is False


class _FakeLimitDriver(_FakeLandDriver):
    """A controller that booted with its own limits and idled the carriage at connect."""

    def __init__(self, **kw):
        super().__init__(**kw)
        # the vendor's boot limits: the arm joints' as the golden has them, the carriage's -4 mm
        self.limits = [_FakeBootLimit(-3.14, 3.14) for _ in range(N - 1)] + [_FakeBootLimit()]
        self.positions[6] = -0.00467  # carriage on its stop, past the boot -4 mm limit

    def get_joint_limits(self):
        return self.limits

    def set_joint_limits(self, jl):
        self.limits = list(jl)
        self.calls.append("set_joint_limits")

    def get_error_information(self):
        # first session sees the boot-limit fault; the reconnect after the golden is clean
        return "Position limit exceeded: -0.004668 < -0.004000. Setting to idle." \
            if len(_FakeLandDriver.instances) == 1 else "No error"


class _FakeBootLimit:
    def __init__(self, lo=-0.004, hi=0.044):
        self.position_min = lo
        self.position_max = hi


def test_landing_applies_the_golden_limits_and_reconnects_to_clear_the_fault(land, tmp_path, monkeypatch):
    """The follower's carriage rests past the controller's boot limit; the landing must push
    the golden's -6 mm limit and reconnect so clear_error runs against it (2026-09-01)."""
    from lerobot_robot_tatbot import recovery

    limits = "\n".join(
        f"- position_min: {-0.006 if j == 6 else -3.14}\n  position_max: {0.04 if j == 6 else 3.14}" for j in range(N)
    )
    (tmp_path / "follower.yaml").write_text("joint_limits:\n" + limits + "\n")
    monkeypatch.setenv("TATBOT_CONFIG_DIR", str(tmp_path))
    _FakeLandDriver.instances = []
    monkeypatch.setattr(recovery.trossen_arm, "TrossenArmDriver", _FakeLimitDriver)
    staged = [0.0, 1.047, 0.524, 0.628, 0.0, 0.0, 0.0]
    # the recover script names the arm "follower@<ip>"; the golden is looked up by role
    assert recovery.land_arm("192.0.2.2", object(), staged, name="follower@192.0.2.2", gripper_index=6)
    first, second = _FakeLandDriver.instances[:2]
    assert first.calls[:2] == ["configure(clear_error=True)", "set_joint_limits"]
    assert first.limits[6].position_min == pytest.approx(-0.006)
    assert first.calls[-1] == "cleanup"
    assert second.calls[0] == "configure(clear_error=True)"
    assert f"modes:{trossen_arm.Mode.position}" in second.calls
    assert second.calls[-1] == "cleanup"


class _BootLimit:
    def __init__(self, lo, hi, tolerance=0.0):
        self.position_min, self.position_max = lo, hi
        self.position_tolerance = tolerance


class _FakeLimitedLandDriver(_FakeLandDriver):
    """A power-cycled controller: boot limits, no error while idle, carriage on
    its stop past the boot limit (the 2026-09-04 triple-recover case)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.positions[6] = -0.0047
        self.joint_limits = [_BootLimit(-3.14, 3.14)] * 6 + [_BootLimit(0.0, 0.040)]

    def get_joint_limits(self):
        return list(self.joint_limits)


def test_recovery_admits_rotary_feedback_tolerance_but_not_carriage_overtravel():
    from lerobot_robot_tatbot import recovery

    limits = [(-3.14, 3.14, 0.2)] * 6 + [(0.0, 0.040, 0.004)]
    positions = [0.0] * 7
    positions[1] = -3.15
    positions[6] = -0.001
    assert recovery._outside_limits(positions, limits) == [6]
    hold = recovery._clamp_to_limits(positions, limits)
    assert hold[1] == pytest.approx(-3.14 + recovery.LIMIT_MARGIN)
    assert hold[6] == pytest.approx(recovery.LIMIT_MARGIN)


def test_landing_applies_golden_when_measured_pose_is_outside_boot_limits(land, monkeypatch):
    """The controller reports no error while idle, so the fault-only golden
    path never fired; the takeover hold at -4.7 mm then tripped the boot
    -4 mm limit and idled the carriage. Now the measured pose is checked
    against the live limits, the golden is pushed, and the hold is legal."""
    from lerobot_robot_tatbot import recovery

    def fake_golden(driver, name):
        driver.joint_limits[6] = _BootLimit(-0.006, 0.040)
        return ["joint_limits"]

    monkeypatch.setattr(recovery, "_apply_golden", fake_golden)
    monkeypatch.setattr(recovery.trossen_arm, "TrossenArmDriver", _FakeLimitedLandDriver)
    staged = [0.0, 1.047, 0.524, 0.628, 0.0, 0.0, 0.0]
    assert recovery.land_arm("192.0.2.2", object(), staged,
                             name="follower", gripper_index=6, attempts=1)
    drv = land[0]
    assert drv.joint_limits[6].position_min == -0.006, "golden applied before takeover"
    # Takeover and staged sweep preserve the admitted measurement; sleep closes it.
    assert drv.moves[0][6] == pytest.approx(-0.0047)
    assert drv.moves[1][6] == pytest.approx(-0.0047)
    assert drv.moves[2][6] == pytest.approx(0.0)


def test_landing_clamps_hold_into_limits_without_a_golden(land, monkeypatch):
    """No golden available: the hold target must still be inside the limits
    the controller enforces rather than the raw measured value."""
    from lerobot_robot_tatbot import recovery

    monkeypatch.setattr(recovery, "_apply_golden", lambda driver, name: [])
    monkeypatch.setattr(recovery.trossen_arm, "TrossenArmDriver", _FakeLimitedLandDriver)
    assert recovery.land_arm("192.0.2.2", object(), [0.0] * N,
                             name="follower", gripper_index=6, attempts=1)
    drv = land[0]
    assert all(0.0 <= m[6] <= 0.040 for m in drv.moves)
    assert drv.moves[0][6] == pytest.approx(recovery.LIMIT_MARGIN)


def test_landing_leaves_a_legal_pose_alone(land):
    """A pose inside the limits is neither clamped nor golden-healed."""
    from lerobot_robot_tatbot import recovery

    class _Inside(_FakeLimitedLandDriver):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.positions[6] = 0.0175

    import lerobot_robot_tatbot.recovery as rec
    rec.trossen_arm.TrossenArmDriver = _Inside
    assert recovery.land_arm("192.0.2.2", object(), [0.0] * N,
                             name="follower", gripper_index=6, attempts=1)
    assert land[0].moves[0][6] == pytest.approx(0.0175)
    assert land[0].moves[1][6] == pytest.approx(0.0175)
    assert land[0].moves[2][6] == pytest.approx(0.0)


def _golden_limits():
    """The goldens' limits: the vendor's, but for the follower carriage's -6 mm."""
    return [_BootLimit(-math.pi, math.pi, 0.2), _BootLimit(0.0, math.pi, 0.2), _BootLimit(0.0, 2.356, 0.2),
            _BootLimit(-math.pi / 2, math.pi / 2, 0.4), _BootLimit(-math.pi / 2, math.pi / 2, 0.4),
            _BootLimit(-math.pi, math.pi, 0.4), _BootLimit(-0.006, 0.040, 0.004)]


LEADER_STAGED = [0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2, 0.0]


class _FakeWrappedWristDriver(_FakeLandDriver):
    """The blue controller power-cycled at the staged wrist roll (2026-09-29): healthy and
    idle, the golden's limits, joint 5 counted a full turn low (-4.796 where it read +1.487)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.positions = [0.018, 0.002, 0.001, -0.008, -0.008, 1.487 - 2 * math.pi, 0.0]
        self.joint_limits = _golden_limits()

    def get_joint_limits(self):
        return list(self.joint_limits)


def test_landing_refuses_a_joint_counted_a_full_turn_off_and_commands_nothing(land, monkeypatch):
    """Clamping joint 5 into its limits would snap the wrist 95 deg to -pi, and staging would
    then turn it and its camera cable a full extra turn. Refused instead: no golden, no mode
    switch, no target, its one session closed, no retry."""
    from lerobot_robot_tatbot import recovery

    golden = []
    monkeypatch.setattr(recovery, "_apply_golden", lambda driver, name: golden.append(name) or ["joint_limits"])
    monkeypatch.setattr(recovery.trossen_arm, "TrossenArmDriver", _FakeWrappedWristDriver)
    with pytest.raises(recovery.JointBeyondLimitsError, match=r"joint 5 at -4\.796 rad") as refused:
        recovery.land_arm("192.0.2.3", object(), LEADER_STAGED, name="leader")
    assert [driver.calls for driver in land] == [["configure(clear_error=True)", "cleanup"]]
    assert golden == []
    message = str(refused.value).lower()
    assert "joint 0" not in message
    for step in ("power the controller off", "turn joint 5 by hand to near 0", "power it on"):
        assert step in message


def test_landing_takes_over_joints_resting_within_their_tolerance(land, monkeypatch):
    """Shoulder and elbow resting a few mrad below 0 on their stops are inside the
    controller's tolerance: held just inside the limit, as before, and landed."""
    from lerobot_robot_tatbot import recovery

    class _OnStops(_FakeWrappedWristDriver):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.positions = [0.0, -0.0055, -0.0017, 0.0, 0.0, math.pi / 2, 0.0]

    monkeypatch.setattr(recovery.trossen_arm, "TrossenArmDriver", _OnStops)
    assert recovery.land_arm("192.0.2.3", object(), LEADER_STAGED, name="leader", attempts=1)
    takeover = land[0].moves[0]
    assert takeover[1] == pytest.approx(recovery.LIMIT_MARGIN)
    assert takeover[2] == pytest.approx(recovery.LIMIT_MARGIN)
    assert takeover[5] == pytest.approx(math.pi / 2)


def test_landing_judges_the_joints_again_under_the_golden_it_pushed_for_the_carriage(land, monkeypatch):
    """The carriage past its boot limit still brings in the golden; a joint inside the boot
    band but past the golden's is refused before anything is commanded."""
    from lerobot_robot_tatbot import recovery

    class _WideBoot(_FakeLimitedLandDriver):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.positions[5] = -3.7
            self.joint_limits = [_BootLimit(-4.0, 4.0, 0.4) for _ in range(6)] + [_BootLimit(0.0, 0.040)]

    def golden(driver, name):
        driver.joint_limits = _golden_limits()
        return ["joint_limits"]

    monkeypatch.setattr(recovery, "_apply_golden", golden)
    monkeypatch.setattr(recovery.trossen_arm, "TrossenArmDriver", _WideBoot)
    with pytest.raises(recovery.JointBeyondLimitsError, match=r"joint 5 at -3\.700 rad"):
        recovery.land_arm("192.0.2.2", object(), [0.0] * N, name="follower")
    assert land[0].joint_limits[6].position_min == -0.006, "the golden came in for the carriage"
    assert land[0].calls == ["configure(clear_error=True)", "cleanup"]


def test_landing_judges_a_live_measurement_not_the_sdk_default(land, monkeypatch):
    """Until the daemon hears from the controller, get_all_positions() is the SDK's zero
    default, inside every limit; the guard waits for the real reading."""
    from lerobot_robot_tatbot import recovery

    class _Booting(_FakeWrappedWristDriver):
        reads = 0

        def get_all_positions(self):
            self.reads += 1
            return [0.0] * N if self.reads <= 2 else super().get_all_positions()

    monkeypatch.setattr(recovery.trossen_arm, "TrossenArmDriver", _Booting)
    with pytest.raises(recovery.JointBeyondLimitsError):
        recovery.land_arm("192.0.2.3", object(), LEADER_STAGED, name="leader")
    assert land[0].reads == 3


