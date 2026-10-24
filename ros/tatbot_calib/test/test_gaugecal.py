"""The wrist gauge on the probe body's top (tatbot_calib.gaugecal): the known height along the palette's up, and
the installed tip moved along the plane's normal by the holds' mean correction."""
from __future__ import annotations

import numpy as np
import pytest
from tatbot_calib import gaugecal, station


def test_the_known_height_is_along_the_palettes_up_from_the_body_top():
    palette = np.eye(4)
    ball = np.array([0.30, 0.20, 0.0856])
    tcp = ball - [0.0, 0.0, station.BODY_TOP_BELOW_BALL_M] + [0.014, 0.0, 0.012]
    assert gaugecal.known_height(tcp, palette, ball) == pytest.approx(0.012)
    poses = gaugecal.hold_poses(ball, palette, 180.0, 0.0)
    assert [gaugecal.known_height(p[:3, 3], palette, ball) for p in poses] == pytest.approx(list(gaugecal.HEIGHTS_M))
    assert all(np.hypot(*(p[:2, 3] - ball[:2])) == pytest.approx(gaugecal.OFFSET_M) for p in poses)


def test_the_tip_moves_by_the_mean_correction_along_the_normal():
    rows = [{"known_m": 0.006, "read_m": 0.0052, "normal_cam": [0.0, 0.0, -1.0]},
            {"known_m": 0.008, "read_m": 0.0071, "normal_cam": [0.0, 0.0, -1.0]}]
    delta, spread, normal = gaugecal.correction(rows)
    assert delta == pytest.approx(0.00085) and spread == pytest.approx(0.00005)
    cal = {"tip_cam_m": [0.0, 0.0, 0.15], "cone_end_cam_m": [0.0, 0.0, 0.145], "axis_cam": [0.0, 0.0, 1.0],
           "tool": "lutin-11rl-wormhole", "note": "another cartridge's fit"}
    out = gaugecal.corrected(cal, delta, normal, "lutin-11rl-wormhole", "run", spread, 2)
    assert out["tip_cam_m"][2] == pytest.approx(0.15 - 0.00085)    # farther from the surface: toward the camera
    assert out["measured_with"] == "lutin-11rl-wormhole" and "note" not in out
    assert out["tip_length_m"] == pytest.approx(0.005 - 0.00085)


def test_a_hold_is_tried_where_the_camera_looks_across_the_probe_not_past_the_pen():
    """2026-10-22: holds whose wrist camera stood 18-63 deg round the probe from the hold read its metal top 2-7 mm
    deep or not at all; 103-156 deg round, within ~1 mm."""
    import math

    top, up = np.zeros(3), np.array([0.0, 0.0, 1.0])
    pose = np.eye(4)
    pose[:3, 3] = (0.014, 0.0, 0.02)                         # the hold, 14 mm toward +x over the top
    behind, across = np.eye(4), np.eye(4)
    behind[:3, 3] = (0.10, 0.0, 0.10)                        # the camera out past the hold's side
    turned = math.radians(155.0)
    across[:3, 3] = (0.11 * math.cos(turned) - 0.014, 0.11 * math.sin(turned), 0.10)
    assert gaugecal.view_round_deg(pose, behind, top, up) < 1.0 < gaugecal.VIEW_ROUND_DEG
    assert gaugecal.view_round_deg(pose, across, top, up) == pytest.approx(155.0, abs=0.5)



def test_a_tight_correction_past_three_mm_is_not_adopted(tmp_path):
    """2026-10-22: a setup left on the wrist tags' registration read the station's height ~9 mm off, and its gauge
    holds agreed on +8.75 +- 0.09 mm, which was adopted: the known heights were wrong, not the gauge."""
    from types import SimpleNamespace

    said = []
    cal = SimpleNamespace(say=said.append, run=SimpleNamespace(dir=tmp_path))
    rows = [{"known_m": 0.011 + d, "read_m": 0.011 - 0.00875, "normal_cam": [0.0, 0.0, -1.0]}
            for d in (0.0, 0.0001, -0.0001, 0.00005)]
    installed = tmp_path / "wrist-gauge-right.json"
    installed.write_text('{"tip_cam_m": [0.0, 0.0, 0.15]}')
    assert gaugecal.finish(cal, rows, {"tip_cam_m": [0.0, 0.0, 0.15]}, installed, "a-tool", adopt=True) == 3
    assert "not adopted" in said[-1] and installed.read_text() == '{"tip_cam_m": [0.0, 0.0, 0.15]}'
