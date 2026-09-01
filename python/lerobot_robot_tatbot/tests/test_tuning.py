"""Tests for the tuning registry, golden configs, and cockpit server.

Runs entirely against a FakeDriver — no hardware, no lerobot imports.
"""

import json
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import trossen_arm
import yaml
from lerobot_robot_tatbot import goldens, params
from lerobot_robot_tatbot.params import (
    REST_VELOCITY,
    TuningShared,
)

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "config" / "trossen"
N = 7


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDriver:
    """Stores vectors like the real driver; same getter/setter names."""

    def __init__(self):
        self.friction_constant_terms = [0.24, 0.08, 0.16, -0.01, 0.04, 0.06, 7.0]
        self.friction_coulomb_coefs = [0.0, 0.1, 0.12, 0.02, 0.02, 0.0, 0.0]
        self.friction_viscous_coefs = [0.0] * N
        self.friction_transition_velocities = [0.04] * 3 + [0.01] * 3 + [1e-4]
        self.effort_corrections = [1.1, 1.1, 1.1, 1.25, 1.15, 1.15, 1.15]
        self.motor_parameters = [
            {trossen_arm.Mode.position: _FakeMotor(kp, 8.0 if j < 3 else 1.0)}
            for j, kp in enumerate([120, 120, 120, 80, 40, 40, 20])
        ]
        self.joint_limits = [_FakeLimit() for _ in range(N)]
        self.characteristics = [_FakeCharacteristic() for _ in range(N)]
        self.algorithm = _FakeAlgo()

    def _vec(name):  # noqa: N805 — descriptor factory
        def get(self):
            return list(getattr(self, name))

        def set_(self, vals):
            setattr(self, name, [float(v) for v in vals])

        return get, set_

    get_friction_constant_terms, set_friction_constant_terms = _vec("friction_constant_terms")
    get_friction_coulomb_coefs, set_friction_coulomb_coefs = _vec("friction_coulomb_coefs")
    get_friction_viscous_coefs, set_friction_viscous_coefs = _vec("friction_viscous_coefs")
    get_friction_transition_velocities, set_friction_transition_velocities = _vec(
        "friction_transition_velocities")
    get_effort_corrections, set_effort_corrections = _vec("effort_corrections")

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
    carriage_retract_m: float = 0.040
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


class FakeRobot:
    min_time_to_move = 0.1


def drain(shared, arm, velocities, max_ticks=400):
    """Run apply_pending until nothing is pending (ramped params need many
    ticks). Returns the simulated seconds elapsed."""
    t = 0.0
    for _ in range(max_ticks):
        t += 1 / 30
        shared.apply_pending(arm, velocities, now=t)
        if not shared.pending:
            break
    return t


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.fixture
def rig():
    driver = FakeDriver()
    config = FakeFollowerConfig()
    robot = FakeRobot()
    shared = TuningShared()
    for p in params.build_leader_params(driver, trossen_arm):
        shared.registry[p.name] = p
    for p in params.build_follower_params(driver, config, trossen_arm, robot):
        shared.registry[p.name] = p
    return driver, config, robot, shared


def test_registry_covers_the_guide(rig):
    _, _, _, shared = rig
    names = set(shared.registry)
    for expected in [
        "leader_friction_constant_term", "leader_friction_viscous_coef",
        "leader_effort_correction", "follower_position_kp",
        "follower_velocity_kp", "goal_time_multiplier", "target_filter_tau",
        "max_joint_velocity", "max_relative_target", "motion_scale",
        "carriage_contact_cap_n", "carriage_retract_m",
        "follower_velocity_max", "follower_velocity_tolerance",
        "follower_position_tolerance", "follower_effort_max",
    ]:
        assert expected in names, expected


def test_live_param_applies_on_loop(rig):
    driver, config, _, shared = rig
    shared.request("carriage_contact_cap_n", 20.0)
    assert config.carriage_contact_cap_n == 15.0  # server thread never writes directly
    shared.apply_pending("follower", [0.0] * N)
    assert config.carriage_contact_cap_n == 20.0


def test_clamping(rig):
    _, config, _, shared = rig
    shared.request("carriage_contact_cap_n", 500.0)
    shared.apply_pending("follower", [0.0] * N)
    assert config.carriage_contact_cap_n == 40.0  # registry range cap


def test_held_still_gates_on_motion(rig):
    driver, _, _, shared = rig
    shared.request("follower_position_kp", [130.0] * N)
    moving = [0.0] * 6 + [REST_VELOCITY * 3]
    shared.apply_pending("follower", moving)
    assert driver.motor_parameters[0][trossen_arm.Mode.position].position.kp == 120
    assert "follower_position_kp" in shared.waiting
    drain(shared, "follower", [0.0] * N)
    assert driver.motor_parameters[0][trossen_arm.Mode.position].position.kp == 130
    assert not shared.pending


def test_per_joint_edit_merges(rig):
    driver, _, _, shared = rig
    shared.request_joint("leader_friction_viscous_coef", 3, 0.25)
    drain(shared, "leader", [0.0] * N)  # viscous is ramped
    vec = driver.get_friction_viscous_coefs()
    assert vec[3] == pytest.approx(0.25) and vec[0] == 0.0


def test_goal_time_multiplier_recomputes(rig):
    _, config, robot, shared = rig
    shared.request("goal_time_multiplier", 1.5)
    shared.apply_pending("follower", [0.0] * N)
    assert robot.min_time_to_move == pytest.approx(1.5 / 30)


def test_effort_correction_respects_firmware_range(rig):
    driver, _, _, shared = rig
    shared.request("leader_effort_correction", [0.05] * N)  # below fw min 0.2
    drain(shared, "leader", [0.0] * N)
    assert all(v == pytest.approx(0.2) for v in driver.get_effort_corrections())


def test_session_params_rejected(rig):
    _, _, _, shared = rig
    with pytest.raises(KeyError):
        shared.request("loop_rate", 60)


# ---------------------------------------------------------------------------
# Goldens
# ---------------------------------------------------------------------------


def test_arm_golden_roundtrip(tmp_path, rig):
    driver, _, _, shared = rig
    src = CONFIG_DIR / "follower.yaml"
    assert src.exists(), "run gen: config/trossen/follower.yaml missing"
    dst = tmp_path / "follower.yaml"
    shutil.copy(src, dst)

    values = {
        "follower_position_kp": [111.0] * N,
        "follower_velocity_tolerance": [1.25] * N,
    }
    goldens.update_arm_golden(dst, values, "follower")
    doc = yaml.safe_load(dst.read_text())
    assert doc["motor_parameters"][0]["position"]["position"]["kp"] == 111.0
    # untouched modes keep their original gains
    assert doc["motor_parameters"][0]["idle"]["position"]["kp"] == 0
    assert all(
        jl["velocity_tolerance"] == 1.25 for jl in doc["joint_limits"]
    )
    # untouched fields survive byte-identical in value
    orig = yaml.safe_load(src.read_text())
    assert doc["joint_characteristics"] == orig["joint_characteristics"]
    assert doc["end_effector"] == orig["end_effector"]


def test_leader_golden_roundtrip(tmp_path):
    src = CONFIG_DIR / "leader.yaml"
    dst = tmp_path / "leader.yaml"
    shutil.copy(src, dst)
    goldens.update_arm_golden(
        dst, {"leader_friction_viscous_coef": [0.1] * N}, "leader")
    doc = yaml.safe_load(dst.read_text())
    assert all(
        jc["friction_viscous_coef"] == 0.1
        for jc in doc["joint_characteristics"]
    )


def test_tatbot_yaml_merge_respects_cli_overrides(tmp_path):
    cfg = FakeFollowerConfig(target_filter_tau=0.3)  # "CLI override"
    section = {"target_filter_tau": 0.1, "carriage_contact_cap_n": 12.0, "bogus_key": 1}
    applied = goldens.apply_section(cfg, section)
    assert cfg.target_filter_tau == 0.3  # CLI wins
    assert cfg.carriage_contact_cap_n == 12.0  # default was untouched → yaml applies
    assert "carriage_contact_cap_n" in applied and "target_filter_tau" not in applied


def test_tatbot_yaml_save_and_reload(tmp_path):
    cfg = FakeFollowerConfig(carriage_contact_cap_n=12.0, motion_scale=[0.5] * 6)
    goldens.save_tatbot_yaml(cfg, {"enabled": True, "port": 8899}, tmp_path)
    doc = yaml.safe_load((tmp_path / "tatbot.yaml").read_text())
    assert doc["follower"]["carriage_contact_cap_n"] == 12.0
    assert doc["follower"]["carriage_rest_m"] == 0.0
    assert doc["follower"]["motion_scale"] == [0.5] * 6
    fresh = FakeFollowerConfig()
    goldens.apply_section(fresh, doc["follower"])
    assert fresh.carriage_contact_cap_n == 12.0


def test_repo_tatbot_yaml_matches_defaults():
    """The checked-in tatbot.yaml must mirror the REAL dataclass defaults so
    the first golden load is a no-op (no surprise parameter jumps), and so
    the CLI-override heuristic (value != default) stays meaningful. Imported
    lazily: it pulls lerobot, which the rest of this module avoids."""
    from lerobot_robot_tatbot.config_tatbot_follower import TatbotFollowerConfig

    doc = yaml.safe_load((CONFIG_DIR / "tatbot.yaml").read_text())
    cfg = TatbotFollowerConfig(id="schema_check")
    for key, val in doc["follower"].items():
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
# HTTP server (end to end against the fake rig)
# ---------------------------------------------------------------------------


def _http(port, path, body=None):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


@pytest.fixture
def server(rig, tmp_path, monkeypatch):
    from lerobot_robot_tatbot import tuning_server as ts

    driver, config, robot, _ = rig
    monkeypatch.setattr(ts, "_singleton", None)
    for name in ("leader.yaml", "follower.yaml", "tatbot.yaml"):
        shutil.copy(CONFIG_DIR / name, tmp_path / name)

    srv = ts.TuningServer(port=0)  # ephemeral port
    srv.cfg_dir = tmp_path
    srv.register(
        "leader", params.build_leader_params(driver, trossen_arm), config)
    srv.register(
        "follower",
        params.build_follower_params(driver, config, trossen_arm, robot),
        config, robot=robot,
    )
    assert srv._httpd is not None
    port = srv._httpd.server_address[1]
    srv.shared.publish_values()
    srv.shared.publish("follower", {
        "joints": [f"j{i}" for i in range(N)],
        "positions": [0.1] * N, "velocities": [0.0] * N,
        "external_efforts": [0.0] * N, "sent": [0.1] * N,
        "tracking_error": [0.0] * N, "contact_force_n": 2.0,
        "contact_cap_n": 15.0, "carriage_target_m": 0.0, "goal_time": 0.1, "estop": "ok",
    })
    yield srv, port, rig
    srv.close()


def test_http_registry(server):
    srv, port, _ = server
    data = _http(port, "/api/registry")
    assert set(data["arms"]) == {"leader", "follower"}
    byname = {p["name"]: p for p in data["params"]}
    assert byname["carriage_contact_cap_n"]["value"] == 15.0
    assert byname["carriage_contact_cap_n"]["golden"] == 15.0
    assert byname["follower_position_kp"]["per_joint"] == N


def test_http_set_save_revert(server):
    srv, port, (driver, config, robot, _) = server
    shared = srv.shared
    _http(port, "/api/param", {"name": "carriage_contact_cap_n", "value": 20})
    shared.apply_pending("follower", [0.0] * N)
    shared.publish_values()
    assert config.carriage_contact_cap_n == 20.0
    assert "carriage_contact_cap_n" in dict(shared.dirty())

    saved = _http(port, "/api/save", {})["saved"]
    assert any(s.endswith("tatbot.yaml") for s in saved)
    doc = yaml.safe_load((srv.cfg_dir / "tatbot.yaml").read_text())
    assert doc["follower"]["carriage_contact_cap_n"] == 20.0
    assert not shared.dirty()  # saved value is the new golden

    _http(port, "/api/param", {"name": "carriage_contact_cap_n", "value": 8})
    shared.apply_pending("follower", [0.0] * N)
    shared.publish_values()
    reverted = _http(port, "/api/revert", {})["reverted"]
    assert "carriage_contact_cap_n" in reverted
    shared.apply_pending("follower", [0.0] * N)
    assert config.carriage_contact_cap_n == 20.0  # back to (new) golden


def test_http_per_joint_and_errors(server):
    srv, port, (driver, config, robot, _) = server
    shared = srv.shared
    _http(port, "/api/param",
          {"name": "leader_friction_viscous_coef", "joint": 2, "value": 0.5})
    drain(shared, "leader", [0.0] * N)  # viscous is ramped
    assert driver.get_friction_viscous_coefs()[2] == pytest.approx(0.5)

    with pytest.raises(urllib.error.HTTPError) as e:
        _http(port, "/api/param", {"name": "nope", "value": 1})
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        _http(port, "/api/recover", {})  # not standalone
    assert e.value.code == 409


def test_http_capture_pose(server):
    srv, port, (driver, config, robot, _) = server
    out = _http(port, "/api/capture_pose", {})
    assert out["staged_positions"] == [0.1] * N
    assert config.staged_positions == [0.1] * N


def test_http_cockpit_page(server):
    srv, port, _ = server
    req = urllib.request.Request(f"http://127.0.0.1:{port}/")
    with urllib.request.urlopen(req, timeout=5) as r:
        body = r.read().decode()
    assert "tatbot tuning" in body and "/api/stream" in body


# ---------------------------------------------------------------------------
# Robustness: guarded driver bindings, golden apply, emergency park
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


def test_driver_guard_caches_and_damps():
    calls = {"get": 0, "fail": False}

    def get_raw():
        calls["get"] += 1
        if calls["fail"]:
            raise RuntimeError("TCP connection closed unexpectedly")
        return [1.0] * N

    stored = {}
    get_fn, set_fn = params._guarded(
        "test_param", get_raw, lambda v: stored.update(v=v))

    # cache: raw read happens exactly once for repeated gets
    assert get_fn() == [1.0] * N
    assert get_fn() == [1.0] * N
    assert calls["get"] == 1

    # damping: a dead link is retried MAX_FAILS times, then never re-read
    calls["fail"] = True
    calls["get"] = 0
    get2, set2 = params._guarded("test_param2", get_raw, lambda v: stored.update(v=v))
    for _ in range(10):
        assert get2() is None
    assert calls["get"] == params._DriverGuard.MAX_FAILS

    # a successful write repopulates the cache and resets the damper
    set2([2.0] * N)
    assert get2() == [2.0] * N
    assert stored["v"] == [2.0] * N


def test_register_survives_dead_link(rig):
    """A follower whose config link is down registers with missing goldens
    instead of raising or spamming."""
    from lerobot_robot_tatbot import tuning_server as ts

    driver, config, robot, _ = rig

    def boom(*a, **k):
        raise RuntimeError("TCP connection closed unexpectedly")

    driver.get_motor_parameters = boom
    driver.get_joint_limits = boom
    srv = ts.TuningServer(port=0)
    shared = srv.register(
        "follower",
        params.build_follower_params(driver, config, trossen_arm, robot),
        config, robot=robot,
    )
    # plugin-side params still have goldens; driver-side ones are absent
    assert "carriage_contact_cap_n" in shared.golden
    assert "follower_position_kp" not in shared.golden
    # publish_values must not raise on the dead link
    shared.publish_values()
    assert "carriage_contact_cap_n" in shared.snapshot["values"]
    srv.close()


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
    monkeypatch.setattr(recovery.trossen_arm, "TrossenArmDriver", _FakeLandDriver)
    return _FakeLandDriver.instances


def test_landing_sequence_and_gripper_hold(land):
    """Home -> sleep -> idle over a fresh session, gripper held throughout so
    a gripped tool is never ground at the position-mode saturation."""
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
    # the takeover pins the measured pose (carriage where it was found); the
    # staged and sleep phases put the carriage at its staged (rest) value
    assert drv.moves[0][6] == pytest.approx(0.0175)
    assert all(m[6] == pytest.approx(0.0) for m in drv.moves[1:])
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
    """The arm goldens are also loaded by the C++ cockpit via
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
# Self-drive protection: per-joint ranges, ramping, instability auto-revert
# ---------------------------------------------------------------------------


def test_friction_constant_has_safe_per_joint_ranges(rig):
    """A wrist joint tuned at 0.04 Nm must not reach gripper-scale values:
    that is what self-drove joint 4 into the firmware velocity trip."""
    _, _, _, shared = rig
    p = shared.registry["leader_friction_constant_term"]
    assert p.limits is not None
    for j in range(6):  # arm joints, Nm
        lo, hi = p.joint_limits(j)
        assert hi <= 1.0, f"joint {j} ceiling {hi} allows self-drive"
    lo, hi = p.joint_limits(6)  # gripper carriage, N (golden is 7.0)
    assert hi >= 7.0
    # clamping is per joint, not shared
    clamped = p.clamp([12.0] * 7)
    assert clamped[:6] == [1.0] * 6
    assert clamped[6] == 12.0


def test_destabilizing_params_are_ramped(rig):
    _, _, _, shared = rig
    for name in ("leader_friction_constant_term", "leader_friction_coulomb_coef",
                 "leader_effort_correction", "follower_position_kp"):
        assert shared.registry[name].ramp_seconds, f"{name} steps instantly"


def test_ramp_walks_to_target_over_time(rig):
    """A slider jump becomes a smooth approach, not a torque step."""
    driver, _, _, shared = rig
    driver.set_friction_constant_terms([0.04] * N)
    shared.publish_values()
    shared.request("leader_friction_constant_term", [0.9] * N)

    t = 0.0
    first = None
    for _ in range(200):  # up to ~6.6 s at 30 Hz
        t += 1 / 30
        shared.apply_pending("leader", [0.0] * N, now=t)
        value = driver.get_friction_constant_terms()[0]
        if first is None:
            first = value
        if not shared.pending:
            break
    # first tick moved only a slice of the range, not the whole jump
    assert first - 0.04 < 0.05, f"stepped {first - 0.04:.3f} in one tick"
    assert driver.get_friction_constant_terms()[0] == pytest.approx(0.9, abs=1e-6)
    # ramp_seconds is the full-range traverse time; 0.04 -> 0.9 covers 57%
    # of the (-0.5, 1.0) range, so ~1.7 s of the 3 s budget.
    assert 1.2 < t < 2.5, f"ramp took {t:.1f}s (expected ~1.7s)"


def test_instability_reverts_the_last_change(rig):
    """The runaway case: raise friction, joint accelerates, value goes back."""
    driver, _, _, shared = rig
    driver.set_friction_constant_terms([0.04] * N)
    shared.publish_values()
    shared.request("leader_friction_constant_term", [0.9] * N)
    shared.apply_pending("leader", [0.0] * N, now=1.0)
    assert driver.get_friction_constant_terms()[0] > 0.04  # ramp underway

    reverted = shared.note_instability("leader", "leader joint hit 9.6 rad/s", now=1.5)
    assert reverted == "leader_friction_constant_term"
    assert driver.get_friction_constant_terms() == [0.04] * N
    assert not shared.pending  # in-flight ramp cancelled, not resumed

    # ramp must not restart on later ticks
    shared.apply_pending("leader", [0.0] * N, now=2.0)
    assert driver.get_friction_constant_terms() == [0.04] * N


def test_instability_ignores_stale_changes(rig):
    """Fast hand motion long after a tuning change must not revert anything."""
    driver, _, _, shared = rig
    shared.request("leader_friction_viscous_coef", [0.1] * N)
    shared.apply_pending("leader", [0.0] * N, now=1.0)
    assert shared.note_instability("leader", "fast move", now=99.0) is None
    assert shared.note_instability("follower", "no changes here", now=1.1) is None


# ---------------------------------------------------------------------------
# Leader damping (the term the firmware friction model cannot provide)
# ---------------------------------------------------------------------------


@dataclass
class FakeLeaderConfig:
    ip_address: str = "192.0.2.3"
    leader_damping: float = 0.0
    joint_names: list = field(default_factory=lambda: [f"j{i}" for i in range(N)])
    staged_positions: list = field(default_factory=lambda: [0.0] * N)
    use_tatbot_yaml: bool = True


class _DampingDriver:
    def __init__(self):
        self.arm_efforts = None
        self.writes = 0

    def set_arm_external_efforts(self, efforts, goal_time, blocking):
        self.arm_efforts = list(efforts)
        self.writes += 1


def _damp(cfg, drv, velocities):
    """Drive TatbotLeader._apply_damping without constructing the plugin."""
    from lerobot_robot_tatbot.tatbot_leader import TatbotLeader

    obj = TatbotLeader.__new__(TatbotLeader)
    obj.config = cfg
    obj.driver = drv
    obj._damping_written = getattr(_damp, "_written", False)
    TatbotLeader._apply_damping(obj, velocities)
    _damp._written = obj._damping_written
    return obj


def test_damping_opposes_velocity():
    from lerobot_robot_tatbot.tatbot_leader import ARM_JOINTS, DAMPING_SIGN

    cfg = FakeLeaderConfig(leader_damping=0.1)
    drv = _DampingDriver()
    _damp._written = False
    _damp(cfg, drv, [1.0, -2.0, 0.0, 0.5, 0.0, 0.0, 9.9])
    assert len(drv.arm_efforts) == ARM_JOINTS  # gripper untouched
    # effort must oppose motion: opposite sign to velocity
    assert abs(drv.arm_efforts[0]) == pytest.approx(0.1)
    assert abs(drv.arm_efforts[1]) == pytest.approx(0.2)
    assert drv.arm_efforts[2] == 0.0
    # Magnitude scales with speed and the sign is consistent per direction.
    # Which sign OPPOSES motion is a firmware convention, fixed on hardware
    # 2026-08-20 (DAMPING_SIGN); it must at least flip with velocity.
    assert drv.arm_efforts[0] * drv.arm_efforts[1] < 0
    assert drv.arm_efforts[0] == pytest.approx(DAMPING_SIGN * 0.1)


def test_damping_is_capped():
    from lerobot_robot_tatbot.tatbot_leader import DAMPING_EFFORT_CAP

    cfg = FakeLeaderConfig(leader_damping=0.5)
    drv = _DampingDriver()
    _damp._written = False
    _damp(cfg, drv, [100.0] * N)  # absurd velocity
    assert all(abs(e) <= DAMPING_EFFORT_CAP for e in drv.arm_efforts)


def test_damping_zero_is_a_noop_then_releases_once():
    cfg = FakeLeaderConfig(leader_damping=0.0)
    drv = _DampingDriver()
    _damp._written = False
    _damp(cfg, drv, [1.0] * N)
    assert drv.writes == 0, "disabled damping must not touch the driver"

    cfg.leader_damping = 0.2          # enable
    _damp(cfg, drv, [1.0] * N)
    assert drv.writes == 1
    cfg.leader_damping = 0.0          # disable again
    _damp(cfg, drv, [1.0] * N)
    assert drv.writes == 2 and drv.arm_efforts == [0.0] * 6  # released once
    _damp(cfg, drv, [1.0] * N)
    assert drv.writes == 2, "release happens once, not every tick"


def test_damping_is_a_registry_param():
    drv, cfg = FakeDriver(), FakeLeaderConfig()
    names = {p.name: p for p in params.build_leader_params(drv, trossen_arm, cfg)}
    assert "leader_damping" in names
    p = names["leader_damping"]
    assert p.arm == "leader" and p.persist == "tatbot.yaml" and p.ramp_seconds
    # absent when the config predates the field
    plain = {p.name for p in params.build_leader_params(drv, trossen_arm, None)}
    assert "leader_damping" not in plain


def test_tatbot_yaml_roundtrips_leader_damping(tmp_path):
    fcfg = FakeFollowerConfig()
    lcfg = FakeLeaderConfig(leader_damping=0.12)
    goldens.save_tatbot_yaml(fcfg, {"enabled": True}, tmp_path, leader_config=lcfg)
    doc = yaml.safe_load((tmp_path / "tatbot.yaml").read_text())
    assert doc["leader"]["leader_damping"] == 0.12
    fresh = FakeLeaderConfig()
    goldens.apply_section(fresh, doc["leader"])
    assert fresh.leader_damping == 0.12


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
    # carriage: measured at the takeover, then the staged (rest) value
    for _, drv, _, _ in arms:
        assert drv.moves[0][6] == pytest.approx(0.0175)
        assert all(m[6] == pytest.approx(0.0) for m in drv.moves[1:])
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
    """Every teleop entry point (record, teleoperate, tune) must share the
    coordinated lift/land — it is on unless explicitly disabled."""
    import os
    os.environ.setdefault("TATBOT_CONFIG_DIR", str(CONFIG_DIR))
    from lerobot_robot_tatbot.config_tatbot_follower import TatbotFollowerConfig
    from lerobot_robot_tatbot.config_tatbot_leader import TatbotLeaderTeleopConfig
    from lerobot_robot_tatbot.tatbot_follower import TatbotFollower
    from lerobot_robot_tatbot.tatbot_leader import TatbotLeader

    follower = TatbotFollower(TatbotFollowerConfig(id="t"))
    leader = TatbotLeader(TatbotLeaderTeleopConfig(id="t"))
    assert follower.config.coordinated_arms is True
    assert leader.config.coordinated_arms is True
    for rig in (follower, leader):
        assert hasattr(rig, "finish_staging")
        assert hasattr(rig, "_ensure_staged")


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
