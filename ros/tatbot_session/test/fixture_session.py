"""Test-only session entry point: temporary motion.yaml, optionally deterministic wrist observations.

The ROS actions, controller, executor, locate search/fusion, gauge holds and surface fit stay real. Camera
capture/marker decoding and depth sensing are fixtures; their image algorithms have separate unit tests.
Never installed as a ROS executable, and refuses physical hardware before starting the node.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


class Camera:
    def grab_both(self, count, **kwargs):
        return [np.zeros((8, 8, 3), np.uint8)] * count, []

    def close(self):
        pass


def install_sensing(stack):
    from tatbot_cli import ros_depth
    from tatbot_session import drawrun, gauge, sim
    from tatbot_session import inspect as ins
    from tatbot_session.executor import ArmExecutor
    from test_sim_gauge import cylinder

    sensor = sim.TruthGauge(*cylinder(drawrun.fixed_page(stack, "right")[0], step_m=0.002)[:2])
    sensor.wrist = "real"   # include camera locate as well as the simulator gauge's real ray casts
    gauge.sensing = lambda stack, ex: sensor
    ArmExecutor.wrist_camera = lambda self, serial: Camera()
    # The sensor boundary supplies a decoded marker observation; no planner or setup decision is replaced.
    ArmExecutor._view_read = staticmethod(lambda cam, frames, *args: (frames[0], {
        "tx_m": 0.0, "ty_m": 0.0, "theta_rad": 0.0, "sigma": [0.0001] * 3, "sides": [0, 1, 2, 3]}, ""))
    ros_depth.retain = lambda *args: (None, {"reason": "fixture depth supplied by TruthGauge"}, None)
    ros_depth.base_points = lambda cam, depths, base_from_depth: (np.empty((0, 3)), None)   # nor seeds the first hold
    aim = ins.aim

    def fixture_aim(*args, azimuths=None, **kwargs):
        # One known view of this fixed fixture. Solve and validate it normally; the full candidate-grid
        # chooser has its own inspect tests. Searching thousands of other views here cost 26 s.
        if "tilts_deg" in kwargs:   # a known top-down candidate; retain every production posture/clearance bound
            grid = {**kwargs, "tilts_deg": (10,), "rolls_deg": (180,)}
            return aim(*args, azimuths=np.radians([300]) if azimuths is None else azimuths, **grid)
        return aim(*args, azimuths=np.radians([345]) if azimuths is None else azimuths,
                   tilts_deg=(50,), distances_m=(0.24,), rolls_deg=(135,), **kwargs)

    ins.aim = fixture_aim


def main():
    import tatbot_motion
    from tatbot_motion import station
    from tatbot_session import gauge

    parser = argparse.ArgumentParser()
    parser.add_argument("motion", type=Path)
    parser.add_argument("stack", type=Path)
    parser.add_argument("--sensing", action="store_true")
    args, ros_args = parser.parse_known_args()
    stack = yaml.safe_load(args.stack.read_text())
    assert stack["hardware"] in ("mock", "fake"), "test entry point refuses physical hardware"
    tatbot_motion.motion_path = lambda: args.motion
    gauge.CALIB = args.stack.parent
    station.touch_path = lambda arm: args.stack.parent / f"no-station-touch-{arm}.json"
    if args.sensing:
        assert stack["hardware"] == "mock" and stack["sensing"]["wrist"] == "real"
        install_sensing(stack)
    from tatbot_session.node import main as session

    return session(ros_args)


if __name__ == "__main__":
    raise SystemExit(main())
