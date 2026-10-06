"""Probe-station calibration.

- station: the palette's pose in an arm's frame, measured from its roof tag by the overhead D555 after a run
  starts, and the rules that keep it fresh (principle 8): an older measurement is refused, and a station that
  moved between two measurements invalidates what was taken in between.
- cli: `ros2 run tatbot_calib station fix`, which `tatbot ros station` drives.
- tool: the arm's fitted tool and its contact model, refused before any motion when the station cannot probe it.
- halo, solve, vision, program: the touches, the fit and `ros2 run tatbot_calib calib run`; adopt: the gate and
  the workspace.yaml edit; register: an arm's registration to the D555.
- sweep: the tip contact-free, from joint-6 turns the palette camera watches (`ros2 run tatbot_calib sweep`).
"""
