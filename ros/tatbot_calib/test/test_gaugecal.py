"""The wrist gauge on the probe body's top (tatbot_calib.gaugecal): the known height along the palette's up, and
the installed tip moved along its ray from the camera until its height changes by the holds' mean correction."""
from __future__ import annotations

import json
import math

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


def test_off_the_camera_axis_the_tip_moves_along_its_ray_keeping_its_pixel():
    """The xArm's wrist_2 sees its tip 47.5 deg off the plane's normal (2026-11-10): moved along the normal, a -1.73 mm
    correction set the height right but put the tip 1.9 mm sideways, the ink's way on every lab cell."""
    normal = np.array([-0.7287, 0.2207, -0.6483])
    normal /= np.linalg.norm(normal)
    cal = {"tip_cam_m": [0.00606, -0.00036, 0.15559], "cone_end_cam_m": [0.00166, 0.00097, 0.15168],
           "axis_cam": (-normal).tolist(), "measured_with": "lutin-3rl-bugpin", "source": "a wristcam fit"}
    out = gaugecal.corrected(cal, -0.00173, normal, "lutin-3rl-bugpin", "run", 0.0001, 11)
    before, after = np.asarray(cal["tip_cam_m"]), np.asarray(out["tip_cam_m"])
    assert (after - before) @ normal == pytest.approx(-0.00173)                  # the height moves by the correction
    assert after / np.linalg.norm(after) == pytest.approx(before / np.linalg.norm(before))   # on the same pixel
    along_normal = before - 0.00173 * normal                                    # where the old rule put it
    sideways = after - along_normal
    assert np.linalg.norm(sideways) == pytest.approx(0.00173 * math.tan(math.radians(47.5)), rel=0.05)
    with pytest.raises(ValueError, match="grazing"):
        gaugecal.corrected(cal, -0.00173, np.cross(before / np.linalg.norm(before), [0.0, 1.0, 0.0]), "t", "r", 0.0, 3)


def test_an_adopted_gauge_is_rederived_from_its_own_holds_along_its_ray(tmp_path):
    normal = np.array([-0.7287, 0.2207, -0.6483])
    normal /= np.linalg.norm(normal)
    rows = [{"known_m": 0.011 + k * 0.002, "read_m": 0.011 + k * 0.002 + 0.0017 + 0.0001 * (-1) ** k,
             "normal_cam": normal.tolist()} for k in range(4)]
    run = tmp_path / "a-gauge-run"
    run.mkdir()
    (run / "candidate.json").write_text(json.dumps({"rows": rows}))
    delta, _, mean_normal = gaugecal.correction(rows)
    start = {"tip_cam_m": [0.00606, -0.00036, 0.15559], "measured_with": "lutin-3rl-bugpin", "source": "a wristcam fit"}
    installed = {**start, "tip_cam_m": (np.asarray(start["tip_cam_m"]) + delta * mean_normal).tolist(),
                 "run": run.name, "utc": "2026-11-10T17:04:21Z", "source": "known heights"}   # the old rule's adoption
    out = gaugecal.rederive(installed, run)
    expected = gaugecal.corrected(start, delta, mean_normal, "lutin-3rl-bugpin", run.name, 0.0, 4)["tip_cam_m"]
    assert out["tip_cam_m"] == pytest.approx(expected) and out["run"] == run.name and out["source"] == "known heights"
    with pytest.raises(ValueError, match="adopted by"):
        gaugecal.rederive({**installed, "run": "another"}, run)


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
    cal = SimpleNamespace(say=said.append, run=SimpleNamespace(dir=tmp_path), tip0=None)
    rows = [{"known_m": 0.011 + d, "read_m": 0.011 - 0.00875, "normal_cam": [0.0, 0.0, -1.0]}
            for d in (0.0, 0.0001, -0.0001, 0.00005)]
    installed = tmp_path / "wrist-gauge-right.json"
    installed.write_text('{"tip_cam_m": [0.0, 0.0, 0.15]}')
    assert gaugecal.finish(cal, rows, {"tip_cam_m": [0.0, 0.0, 0.15]}, installed, "a-tool", adopt=True) == 3
    assert "not adopted" in said[-1] and installed.read_text() == '{"tip_cam_m": [0.0, 0.0, 0.15]}'


def test_too_few_holds_in_dark_frames_say_the_probe_needs_light(tmp_path):
    """2026-10-25 at dusk: 0 of 8 holds read the probe's top in frames of median V 13; the message said only that
    none read, so a person at the station could not tell that a light was the fix."""
    from types import SimpleNamespace

    said = []
    cal = SimpleNamespace(say=said.append, run=SimpleNamespace(dir=tmp_path))
    args = ([], {}, tmp_path / "wrist-gauge-right.json", "lutin-3rl-bugpin", True)
    assert gaugecal.finish(cal, *args, dark_v=13.0) == 3
    assert "dark (median V 13 of 255)" in said[-1] and "light the probe's top" in said[-1]
    assert gaugecal.finish(cal, *args, dark_v=63.0) == 3 and "dark" not in said[-1]
    assert gaugecal.finish(cal, *args) == 3 and "dark" not in said[-1]



def test_a_loose_correction_from_few_holds_is_adopted_with_its_error(tmp_path):
    """2026-11-09: the xArm's D405-1 read the probe's top from 3 holds, -1.87 +- 0.27 mm; the mounts are as rigid
    as they can be, so the best estimate is adopted and its error recorded, not refused."""
    import json
    from types import SimpleNamespace

    said = []
    cal = SimpleNamespace(say=said.append, run=SimpleNamespace(dir=tmp_path), tip0=np.array([0.001, -0.002, 0.07]))
    rows = [{"known_m": 0.013, "read_m": 0.013 + d, "normal_cam": [0.0, 0.0, -1.0]} for d in (0.00158, 0.00241, 0.00161)]
    installed = tmp_path / "wrist-gauge-xarm-wrist_1.json"
    installed.write_text('{"tip_cam_m": [0.0, 0.0, 0.15]}')
    derived = {"tip_cam_m": [0.0, 0.0, 0.15], "derived": [{"run": "a-sweep"}], "tip_mount_m": [0.0, 0.0, 0.07]}
    assert gaugecal.finish(cal, rows, derived, installed, "a-tool", adopt=True) == 0
    assert "adopted" in said[-1] and "known to 0.27" in said[-1]
    adopted = json.loads(installed.read_text())
    assert adopted["tip_cam_m"] != [0.0, 0.0, 0.15]
    # measured, so no longer derived from the cartridge's tip, and it stands for the tip the holds were commanded with
    assert "derived" not in adopted and adopted["tip_mount_m"] == [0.001, -0.002, 0.07]
