"""Physical policy bindings and flight evidence, without a controller connection."""
import csv
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import trossen_arm
from lerobot_robot_tatbot import tool_registry, warn_throttle
from lerobot_robot_tatbot.tatbot_follower import TatbotFollower

JOINTS = [f"joint_{i}" for i in range(6)] + ["left_carriage_joint"]


def test_config_address_default_follows_physical_arm(monkeypatch):
    from lerobot_robot_tatbot.config_tatbot_follower import TatbotFollowerConfig
    monkeypatch.setenv('TATBOT_LEADER_IP', '192.0.2.10')
    monkeypatch.setenv('TATBOT_FOLLOWER_IP', '192.0.2.20')
    assert TatbotFollowerConfig(physical_arm='left').ip_address == '192.0.2.10'
    assert TatbotFollowerConfig().ip_address == '192.0.2.20'
    with pytest.raises(ValueError, match='physical_arm'):
        TatbotFollowerConfig(physical_arm='unknown')


@pytest.mark.parametrize('arm, preset', [('left', 'wxai_v0_leader'), ('right', 'wxai_v0_follower')])
def test_connection_uses_selected_physical_preset_before_configuration(arm, preset):
    calls = []

    class Offline(TatbotFollower):
        @property
        def is_connected(self):
            return False

        @property
        def is_calibrated(self):
            return True

        def configure(self):
            calls.append('guarded configure')

        def __del__(self):
            pass

    robot = object.__new__(Offline)
    robot.config = SimpleNamespace(physical_arm=arm, ip_address='192.0.2.10')
    robot.driver = SimpleNamespace(configure=lambda **kw: calls.append(kw))
    robot.cameras = {'wrist': SimpleNamespace(connect=lambda: calls.append('camera'))}
    robot._connect_owned()
    assert calls == [{'model': trossen_arm.Model.wxai_v0,
                      'end_effector': getattr(trossen_arm.StandardEndEffector, preset),
                      'serv_ip': '192.0.2.10', 'clear_error': True}, 'camera', 'guarded configure']


def test_left_floor_uses_its_own_tool_block_joints_and_receipt(monkeypatch):
    reg = tool_registry.registry()
    workspace = reg.read_workspace(tool_registry.REPO)
    workspace['left']['paper_plane_z'] = 0.045
    workspace['left']['touchoff'] = {'n_pad': 3, 'utc': datetime.now(UTC).isoformat()}
    workspace['right']['paper_plane_z'] = 100
    workspace['right']['touchoff'] = {'n_pad': 0}
    monkeypatch.setattr(reg, 'read_workspace', lambda *a: workspace)
    robot = object.__new__(TatbotFollower)
    robot.config = SimpleNamespace(physical_arm='left', joint_names=JOINTS,
                                   z_floor_urdf='urdf/tatbot.urdf', z_floor_m=None,
                                   z_floor_below_surface_m=.01, z_floor_max_age_s=3600)
    robot._kin = None
    robot._tool = reg.load_tool(workspace['left']['tool_id'], tool_registry.REPO)
    kin = robot._kinematics()
    assert kin is not None
    assert robot.config.z_floor_m == pytest.approx(.035)
    assert all(name.startswith('left/') for name in robot._kin_joints)
    robot._validate_floor_receipt()
    pose = dict.fromkeys(JOINTS, 0.)
    q = {f'left/{joint}': value for joint, value in pose.items()}
    assert robot._tool_z(kin, pose) == pytest.approx(kin.link_pose('left/tattoo_needle', q)[2, 3])
    workspace['left']['touchoff']['n_pad'] = 0
    workspace['right']['touchoff']['n_pad'] = 9
    with pytest.raises(RuntimeError, match='no paper-pad touch'):
        robot._validate_floor_receipt()


def test_emergency_landing_keeps_selected_preset_and_golden(monkeypatch):
    from lerobot_robot_tatbot import recovery
    calls = []
    monkeypatch.setattr(recovery, 'land_arm', lambda *a, **kw: calls.append((a, kw)) or True)
    robot = object.__new__(TatbotFollower)
    robot.config = SimpleNamespace(physical_arm='left', ip_address='192.0.2.10', staged_positions=[0.] * 7)
    robot._estop = None
    assert robot._emergency_landing() is None
    args, kwargs = calls[0]
    assert args[1] == trossen_arm.StandardEndEffector.wxai_v0_leader
    assert kwargs['name'] == 'leader'


def test_repeated_staging_preserves_rows_and_one_throttle(tmp_path, monkeypatch):
    from lerobot_robot_tatbot import runlog_shim
    installs = []
    monkeypatch.setattr(warn_throttle, 'install', lambda: installs.append(object()) or installs[-1])
    monkeypatch.setattr(runlog_shim, 'artifact', lambda *a, **kw: None)
    path = tmp_path / 'flight.csv'
    robot = object.__new__(TatbotFollower)
    robot.id = 'offline'
    robot.config = SimpleNamespace(id='offline', joint_names=JOINTS)
    robot._flight_log_path = lambda: path
    robot._flight_log = robot._flight_writer = robot._clamp_throttle = None
    robot._motion_watchdog = SimpleNamespace(reset=lambda *a: None)
    try:
        robot._finish_staging_runtime()
        handle = robot._flight_log
        for row in range(50):
            robot._flight_writer.writerow([row])
        robot._finish_staging_runtime()
        robot._flight_writer.writerow(['after staging'])
        assert robot._flight_log is handle
        assert len(installs) == 1
        with Path(path).open() as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == 52 and rows[1] == ['0'] and rows[-1] == ['after staging']
    finally:
        if robot._flight_log is not None:
            robot._flight_log.close()
