"""gauge_derive: a wrist gauge's tip follows the fitted cartridge's tip through its camera's hand-eye, replayed on
the 3RL -> 7RL swap of 2026-11-10, where the session camera's gauge was also measured again."""
from __future__ import annotations

import json
import math

import gauge_derive
import pytest

# The session camera's gauge before the swap (measured with the 3RL), its hand-eye from that morning's `calib
# wristcam`, the tip the sweep adopted for the 7RL (across its axis only), and the gauge `calib gauge` then measured.
BEFORE = {"tip_cam_m": [0.0048783, 2.58e-05, 0.1570501], "cone_end_cam_m": [0.0016647, 0.0009743, 0.1516806],
          "axis_cam": [0.7287099, -0.2206857, 0.6482898], "normal_cam": [-0.7287099, 0.2206857, -0.6482898],
          "tip_length_m": 0.0060322, "measured_with": "lutin-3rl-bugpin", "tool": "lutin-3rl-bugpin",
          "utc": "2026-11-10T17:04:21Z", "run": "20261110T165856Z-node-8676", "correction_m": -0.0017322}
MOUNT_FROM_CAMERA = [[0.6848225, 0.2345825, -0.6899197, 0.1060959], [0.000178, 0.9467148, 0.3220731, -0.0525109],
                     [0.7287099, -0.2206857, 0.6482898, -0.032699], [0.0, 0.0, 0.0, 1.0]]
TIP_3RL, TIP_7RL = [0.002817, -0.002736, 0.070932], [0.002582, -0.002697, 0.070932]
MEASURED_TIP, MEASURED_M, MEASURED_SE_M = [0.0048734, 2.57e-05, 0.1568914], 0.000106, 0.000126
SWEEP = "20261110T193725Z-node-5042"


def _derived(gauge=BEFORE, then=TIP_3RL, now=TIP_7RL):
    return gauge_derive.derive(gauge, then, now, [row[:3] for row in MOUNT_FROM_CAMERA[:3]], tool_id="lutin-7rl",
                               run=SWEEP, utc="2026-11-10T19:46:46Z")


def test_the_derived_gauge_agrees_with_the_one_measured_after_the_swap():
    """The measured gauge moved +0.11 +- 0.13 mm along the plane's normal; the tip moved 0.24 mm across the pen, so
    the derived one keeps its height, inside that error. Across the normal they stand 0.35 mm apart: the tip's own
    move plus the measurement's slide along the tip's ray, neither of which the holds can see."""
    derived = _derived()
    assert gauge_derive.along_normal_mm(BEFORE, derived) == pytest.approx(0.0, abs=0.001)
    apart = [d - m for d, m in zip(derived["tip_cam_m"], MEASURED_TIP, strict=True)]
    along = sum(a * n for a, n in zip(apart, BEFORE["normal_cam"], strict=True))
    assert abs(along) == pytest.approx(MEASURED_M, abs=2e-6) and abs(along) < MEASURED_SE_M
    across = math.sqrt(sum(a * a for a in apart) - along * along)
    assert across * 1000 == pytest.approx(0.35, abs=0.02)
    moved = math.dist(derived["tip_cam_m"], BEFORE["tip_cam_m"])
    assert moved == pytest.approx(math.dist(TIP_3RL, TIP_7RL), abs=1e-9)   # a rotation: as far as the tip moved


def test_a_derived_gauge_says_so_and_keeps_what_was_measured():
    derived = _derived()
    step = derived["derived"][-1]
    assert step["run"] == SWEEP and step["from_tool"] == "lutin-3rl-bugpin"
    assert step["tip_moved_mount_mm"] == [-0.235, 0.039, 0.0]
    assert derived["measured_with"] == derived["tool"] == "lutin-7rl" and derived["tip_mount_m"] == TIP_7RL
    assert derived["run"] == BEFORE["run"] and derived["correction_m"] == BEFORE["correction_m"]
    assert derived["axis_cam"] == BEFORE["axis_cam"] and derived["tip_length_m"] == BEFORE["tip_length_m"]
    # the pen's cone end moves with its tip
    for key in ("tip_cam_m", "cone_end_cam_m"):
        assert [d - b for d, b in zip(derived[key], BEFORE[key], strict=True)] == pytest.approx(
            [v / 1000 for v in step["tip_moved_cam_mm"]], abs=1e-6)
    again = _derived(derived, gauge_derive.tip_before(derived, "lutin-7rl", [9.0, 9.0, 9.0], None), TIP_3RL)
    assert [s["run"] for s in again["derived"]] == [SWEEP, SWEEP]
    assert again["tip_cam_m"] == pytest.approx(BEFORE["tip_cam_m"], abs=1e-9)   # back where it was measured


def test_a_longer_tip_stands_that_much_farther_along_the_pen():
    """A tip adopted 2 mm longer reads every surface 2 mm nearer: the tip moves down the pen's axis in the camera."""
    longer = [TIP_3RL[0], TIP_3RL[1], TIP_3RL[2] + 0.002]
    derived = _derived(now=longer)
    moved = [d - b for d, b in zip(derived["tip_cam_m"], BEFORE["tip_cam_m"], strict=True)]
    assert moved == pytest.approx([0.002 * a for a in BEFORE["axis_cam"]], abs=1e-9)
    assert gauge_derive.along_normal_mm(BEFORE, derived) == pytest.approx(-2.0, abs=1e-3)


def test_the_hand_eye_is_the_files_or_the_wristcam_run_its_gauge_runs_lead_back_to(tmp_path):
    rows = [row[:3] for row in MOUNT_FROM_CAMERA[:3]]
    assert gauge_derive.hand_eye({"mount_from_camera": MOUNT_FROM_CAMERA}, tmp_path) == rows
    kept = "known heights over the probe body's top (tatbot ros calib gauge); the pen's axis and cone end kept from "
    chain = {"20261110T201440Z-node-8a76": {"gauge": {"source": kept + "20261110T165856Z-node-8676"}},
             "20261110T165856Z-node-8676": {"gauge": {"source": kept + "20261110T165413Z-node-292a"}},
             "20261110T165413Z-node-292a": {"mount_from_camera": MOUNT_FROM_CAMERA, "gauge": {"source": "wrist hand-eye"}}}
    for run, candidate in chain.items():
        (tmp_path / run).mkdir()
        (tmp_path / run / "candidate.json").write_text(json.dumps(candidate))
    assert gauge_derive.hand_eye({"run": "20261110T201440Z-node-8a76"}, tmp_path) == rows
    (tmp_path / "20261110T165413Z-node-292a" / "candidate.json").unlink()   # the hand-eye run pruned
    with pytest.raises(gauge_derive.NotDerivedError, match="calib wristcam --adopt"):
        gauge_derive.hand_eye({"run": "20261110T201440Z-node-8a76"}, tmp_path)


def test_a_gauge_that_records_no_tip_stands_for_the_installed_one_only_when_measured_with_it_since():
    installed, since = [0.001, 0.002, 0.07], "2026-11-10T19:46:46Z"
    gauge = {"measured_with": "lutin-7rl", "utc": "2026-11-10T20:18:57Z"}
    assert gauge_derive.tip_before(gauge, "lutin-7rl", installed, since) == installed
    assert gauge_derive.tip_before({**gauge, "tip_mount_m": TIP_3RL}, "another", installed, since) == TIP_3RL
    with pytest.raises(gauge_derive.NotDerivedError, match="measured with lutin-7rl"):
        gauge_derive.tip_before(gauge, "lutin-3rl-bugpin", installed, since)
    with pytest.raises(gauge_derive.NotDerivedError, match="before the installed tip"):
        gauge_derive.tip_before({**gauge, "utc": "2026-11-10T17:04:21Z"}, "lutin-7rl", installed, since)
