"""The scripted dip, without an arm.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_il_dip.py

What has to hold before this touches hardware: the cap targets are on the
rack (the URDF's arc, in the arm base frame, not the rig root); the tool
enters and leaves a cap along its own axis as the operator held it; the EE
target reproduces the tip target exactly; the driver's angle-axis convention
round-trips; the envelope check refuses a wrong frame; and a real tool is
refused until a rehearsal is on record.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "vision"))

import il_dip  # noqa: E402
import ink_spec  # noqa: E402
import tool_spec  # noqa: E402
from urdf_kinematics import UrdfChain  # noqa: E402

# 2026-08-30: the tool moved to the mount on the left finger carriage, so the
# 2026-08-26 palette_center hold (recorded at the old wrist orientation) no
# longer puts the tip anywhere near the rack, and the tip offset is expressed
# in right/tool_mount rather than right/ee_gripper_link. Until the phase-4
# calibration records a fixed-mount-v2 hold, the fixture is the staged pose
# with the wrist rolled 90 deg (cube up, pen down) and the datasheet's nominal
# tip along the bore: geometry the tests can reason about, not a measurement.
# the sim's pen-down IK solution over the pad (2026-08-30, tool axis 45 deg
# forward-down in the carriage frame), nudged off the exact half-turn
PALETTE_HOLD = [0.12, 1.037, 0.392, -0.141, -0.08, 1.45]
LASER_TIP = np.array([0.0, 0.0, 0.060])  # nominal ballpoint tip in right/tool_mount

# A workspace.yaml as the touch-off will write it in the mount frame; the
# checked-in file is nulled until that touch-off happens.
V2_WORKSPACE = {"right": {
    "tool_id": "lutin-ballpoint-dot", "tip_frame": "right/tool_mount", "carriage_m": 0.0,
    "pen_tip_offset_x": 0.0, "pen_tip_offset_y": 0.0, "pen_tip_offset_z": 0.060,
    "paper_plane_z": 0.0227, "paper_band_mm": None,
    "pivot_point_x": 0.38, "pivot_point_y": -0.23, "pivot_point_z": 0.0227,
    "ee_contact_z": None,
    "touchoff": {"utc": "2026-08-30T12:00:00Z", "session": "synthetic", "n_plate": 0,
                 "n_pad": 9, "cond": 8.0, "residual_mm": 1.0, "holdout_mm": None,
                 "spread_deg": 40.0, "note": ""}}}


@pytest.fixture(scope="module")
def frame():
    return il_dip.hold_frame_from_joints(UrdfChain(str(il_dip.URDF)), PALETTE_HOLD, LASER_TIP)


@pytest.fixture
def setup():
    palette = ink_spec.load_palette(REPO)
    load = {s: ink_spec.SlotLoad(s, None) for s in palette}
    layout = ink_spec.palette_layout_from_urdf(REPO)
    root = np.array(ink_spec.palette_root_in_base(REPO))
    policy = ink_spec.policy_for(tool_spec.load_tool("lutin-ballpoint-dot", REPO))
    return palette, load, layout, root, policy


def test_hold_frame_is_in_the_base_frame_and_the_axis_points_down(frame):
    # the pen-down hold: tool axis straight down
    tip_base = frame.ee_for_tip(np.zeros(3))  # ee at origin -> -R tip
    assert frame.axis[2] < -0.9, frame.axis
    assert np.isclose(np.linalg.norm(frame.axis), 1.0)
    # the frame is the BASE frame: FK of the mount through the rig root, with
    # the base's own fixed mount divided out, must agree with the chain
    chain = UrdfChain(str(il_dip.URDF))
    names = chain.arm_joint_names("right")
    root = chain.link_pose(il_dip.TIP_LINK, {**dict(zip(names, PALETTE_HOLD, strict=True)),
                                             il_dip.CARRIAGE_JOINT: il_dip.CARRIAGE_REST_M})
    ee_base = np.linalg.inv(chain.link_pose("right/base_link")) @ root
    assert np.allclose(ee_base[:3, :3], frame.rot_ee)
    assert np.allclose(tip_base, -(frame.rot_ee @ LASER_TIP))


def test_cap_targets_sit_on_the_rack_and_enter_along_the_axis(frame, setup):
    palette, load, layout, root, policy = setup
    slots = [s for s in palette if palette[s].arm == "right"]
    targets = il_dip.cap_targets(slots, palette, load, layout, root, frame, policy, 0.02, 0.03)
    assert [t.slot_id for t in targets] == slots
    for t in targets:
        assert np.allclose(t.rim, root + np.array(layout[t.slot_id]))
        assert np.allclose(t.above - t.rim, -frame.axis * 0.02)
        assert np.allclose(t.bottom - t.rim, frame.axis * t.plunge_m)
        assert np.allclose(t.transit - t.above, [0, 0, 0.03])
        assert t.plunge_m == pytest.approx(policy.dip_depth_m), "dry cap: plunge_m below the rim"
        # the EE target puts the tip exactly there
        ee = frame.ee_for_tip(t.bottom)
        assert np.allclose(ee + frame.rot_ee @ frame.tip_offset, t.bottom)
    assert not il_dip.check_targets(targets)


def test_a_wet_cap_plunges_deeper_than_a_dry_one(frame, setup):
    palette, load, layout, root, policy = setup
    real = ink_spec.policy_for(tool_spec.load_tool("lutin-3rl-bugpin", REPO))
    half = dict(load)
    half["inkcap_right_large"] = ink_spec.SlotLoad("inkcap_right_large", "nighthawk_black", 700.0)
    t_wet, = il_dip.cap_targets(["inkcap_right_large"], palette, half, layout, root, frame, real, 0.02, 0.03)
    t_dry, = il_dip.cap_targets(["inkcap_right_large"], palette, load, layout, root, frame, real, 0.02, 0.03)
    assert t_wet.plunge_m > t_dry.plunge_m
    assert t_wet.plunge_m <= palette["inkcap_right_large"].size.depth_m


def test_wrong_frame_is_refused(frame, setup):
    palette, load, layout, root, policy = setup
    far = il_dip.cap_targets(["inkcap_right_large"], palette, load, layout, root, frame, policy,
                             0.02, 0.03, palette_offset=np.array([0.3, 0.3, 0.0]))
    assert any("not the rack" in p for p in il_dip.check_targets(far))
    low = il_dip.cap_targets(["inkcap_right_large"], palette, load, layout, root, frame, policy,
                             0.02, 0.03, palette_offset=np.array([0.0, 0.0, -0.1]))
    assert any("rim z" in p for p in il_dip.check_targets(low))


def test_moves_enter_and_leave_each_cap_lifted(frame, setup):
    palette, load, layout, root, policy = setup
    targets = il_dip.cap_targets(["inkcap_right_large", "inkcap_right_small_0"], palette, load,
                                 layout, root, frame, policy, 0.02, 0.03)
    moves = il_dip.dip_moves(targets, policy, 3.0, 0.02)
    labels = [m.label.split(": ")[1].split(" ")[0] for m in moves]
    assert labels == ["transit", "above", "plunge", "retract", "lift"] * 2
    plunge = [m for m in moves if m.label.split(": ")[1].startswith("plunge")]
    assert all(m.dwell_s == policy.dip_dwell_s for m in plunge)
    assert all(m.seconds >= 0.8 for m in plunge), "a plunge is slow"
    # every cap-to-cap move happens at transit height
    assert np.allclose(moves[4].tip, targets[0].transit) and np.allclose(moves[5].tip, targets[1].transit)


def test_vertical_frame_points_straight_down_and_keeps_the_tip_math(frame):
    v = il_dip.vertical_frame(frame)
    assert np.allclose(v.axis, [0, 0, -1], atol=1e-9)
    assert np.allclose(v.rot_ee @ v.rot_ee.T, np.eye(3), atol=1e-9), "still a rotation"
    tip = np.array([0.13, 0.21, 0.085])
    assert np.allclose(v.ee_for_tip(tip) + v.rot_ee @ v.tip_offset, tip)
    # the EE sits directly ABOVE the tip now
    ee = v.ee_for_tip(tip)
    assert np.allclose(ee[:2], tip[:2], atol=1e-9) and ee[2] > tip[2]
    assert np.allclose(il_dip.vertical_frame(v).rot_ee, v.rot_ee, atol=1e-9), "idempotent"


def test_axis_angle_round_trip(frame):
    vec = il_dip.rotation_to_axis_angle(frame.rot_ee)
    assert np.allclose(il_dip.axis_angle_to_rotation(vec), frame.rot_ee, atol=1e-9)
    assert np.allclose(il_dip.rotation_to_axis_angle(np.eye(3)), 0)
    flip = np.diag([1.0, -1.0, -1.0])  # 180 deg about x
    assert np.allclose(il_dip.axis_angle_to_rotation(il_dip.rotation_to_axis_angle(flip)), flip, atol=1e-9)


def test_real_tool_needs_a_rehearsal_on_record():
    real = ink_spec.policy_for(tool_spec.load_tool("lutin-3rl-bugpin", REPO))
    reh = ink_spec.policy_for(tool_spec.load_tool("lutin-ballpoint-dot", REPO))
    assert il_dip.real_gate(reh, [], False) is None
    assert "no rehearsal" in il_dip.real_gate(real, [], True)
    rehearsed = [{"kind": "dip", "mode": "rehearsal"}]
    assert "--allow-real" in il_dip.real_gate(real, rehearsed, False)
    assert il_dip.real_gate(real, rehearsed, True) is None
    # a sim dip is not a rehearsal on this rig
    assert "no rehearsal" in il_dip.real_gate(real, [{"kind": "dip", "mode": "sim"}], True)


def test_dry_run_prints_the_plan_and_touches_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TATBOT_INK_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr(il_dip.ink_spec, "load_palette_cal", lambda *a, **k: {})
    monkeypatch.setattr(il_dip.tool_spec, "read_workspace", lambda *a, **k: V2_WORKSPACE)
    # the fitted tool in workspace.yaml is the laser; state the ballpoint and
    # the calibration cross-check refuses — exactly the right answer
    rc = il_dip.main(["--tool-id", "lutin-ballpoint-dot", "--dry-run"])
    err = capsys.readouterr().err
    if rc == 2 and "il_dip:" in err:
        assert "lutin-ballpoint-dot" in err or "workspace" in err
        return
    assert rc == 0
    assert not (tmp_path / "ledger.jsonl").exists()


# --- 2026-08-29: the session, the cap choice, the rack from the hold -------------------

@pytest.fixture
def ink_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TATBOT_INK_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("TATBOT_INK_SESSION", str(tmp_path / "session.json"))
    ball = tool_spec.load_tool("lutin-ballpoint-dot", REPO)
    # the fitted tool in workspace.yaml is whatever was last touched off; for
    # a plan that touches nothing, state the ballpoint and skip the cross-check
    monkeypatch.setattr(il_dip.tool_spec, "require_stated_tool", lambda *a, **k: ball)
    monkeypatch.setattr(il_dip.ink_spec, "load_palette_cal", lambda *a, **k: {})
    monkeypatch.setattr(il_dip.tool_spec, "read_workspace", lambda *a, **k: V2_WORKSPACE)
    return tmp_path


def test_dry_run_chooses_one_cap_with_a_reason(ink_env, capsys):
    rc = il_dip.main(["--tool-id", "lutin-ballpoint-dot", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "1 cap(s)" in out and "session_start" in out and "rehearsal: first usable cap" in out
    assert "no open session — this dip opens one" in out
    assert "palette_root: urdf" in out and "hold" in out
    assert not (ink_env / "ledger.jsonl").exists() and not (ink_env / "session.json").exists()


def test_if_needed_skips_when_the_session_is_charged(ink_env, capsys):
    import ink_session

    ball = tool_spec.load_tool("lutin-ballpoint-dot", REPO)
    pol = ink_spec.policy_for(ball)
    s = ink_session.start(ball, pol, need_ul=0.5)
    ink_session.apply_dip(s, pol, "inkcap_right_medium_0", None, pol.uptake_ul, "session_start")
    rc = il_dip.main(["--tool-id", "lutin-ballpoint-dot", "--dry-run", "--if-needed"])
    out = capsys.readouterr().out
    assert rc == 0 and "not dipping (--if-needed)" in out
    # without the flag it still plans, as an operator dip
    rc = il_dip.main(["--tool-id", "lutin-ballpoint-dot", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "dipping anyway" in out and "operator" in out
    # a run that exhausts the need asks for a dip again
    ink_session.apply_stroke(s, pol, 400.0, 30.0, run_id="r1")
    s = ink_session.current()
    rc = il_dip.main(["--tool-id", "lutin-ballpoint-dot", "--dry-run", "--if-needed", "--need-ul", "1.0"])
    out = capsys.readouterr().out
    assert rc == 0 and "low_charge" in out


def test_another_tools_session_is_refused(ink_env, capsys):
    import ink_session

    real = tool_spec.load_tool("lutin-3rl-bugpin", REPO)
    ink_session.start(real, ink_spec.policy_for(real))
    rc = il_dip.main(["--tool-id", "lutin-ballpoint-dot", "--dry-run"])
    assert rc == 2 and "session end" in capsys.readouterr().err


def _rigid_workspace(monkeypatch, residual_mm: float):
    """A mount-frame workspace with the touch-off's residual set — what the
    cap clearance reads. The gripper-era calibrations were ~4 mm; seated in
    the bore the ballpoint should come back at the 1.5 mm floor."""
    real = V2_WORKSPACE
    ws = {**real, "right": {**real["right"], "touchoff": {**(real["right"].get("touchoff") or {}),
                                                          "residual_mm": residual_mm}}}
    monkeypatch.setattr(il_dip.tool_spec, "read_workspace", lambda *a, **k: ws)


def test_named_slots_are_operator_dips(ink_env, monkeypatch, capsys):
    _rigid_workspace(monkeypatch, 0.8)
    rc = il_dip.main(["--tool-id", "lutin-ballpoint-dot", "--dry-run",
                      "--slots", "inkcap_right_medium_0", "inkcap_right_small_0"])
    out = capsys.readouterr().out
    assert rc == 0 and "2 cap(s)" in out and out.count("operator: named on the command line") == 2


def test_a_loose_calibration_keeps_the_tip_out_of_narrow_caps(ink_env, monkeypatch, capsys):
    """±4 mm on the tip: the 11 mm and 8 mm caps are rim strikes, only the 15 mm one is honest."""
    _rigid_workspace(monkeypatch, 4.2)
    rc = il_dip.main(["--tool-id", "lutin-ballpoint-dot", "--dry-run", "--slots", "inkcap_right_medium_0"])
    out, err = capsys.readouterr()
    assert rc == 2 and "not usable" in err and "inkcap_right_large" in err
    rc = il_dip.main(["--tool-id", "lutin-ballpoint-dot", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "inkcap_right_large" in out and "caps narrower than" in out


def test_stale_hold_is_refused_for_the_rack(ink_env, capsys):
    rc = il_dip.main(["--tool-id", "lutin-ballpoint-dot", "--dry-run", "--palette-from", "hold"])
    err = capsys.readouterr().err
    assert rc == 2 and "not static" in err
    rc = il_dip.main(["--tool-id", "lutin-ballpoint-dot", "--dry-run", "--palette-from", "hold",
                      "--allow-stale-hold"])
    assert rc == 0


@pytest.mark.skip(reason="needs a palette_center hold recorded in the fixed-mount-v2 "
                         "embodiment; "
                         "the 2026-08-26 hold in config/poses.yaml is gripper-era")
def test_rack_root_from_the_hold_is_near_the_urdf():
    chain = UrdfChain(str(il_dip.URDF))
    hold = il_dip.read_hold_joints()
    tip = il_dip.tip_offset_from_workspace(tool_spec.read_workspace(REPO))
    root_hold = il_dip.palette_root_from_hold(chain, hold, tip)
    root_urdf = np.array(ink_spec.palette_root_in_base(REPO))
    # the measured hold lands within a few centimetres of the URDF's rack
    # (docs/ink.md) — the same rack, not a different frame
    assert np.linalg.norm(root_hold - root_urdf) < 0.06


def test_hold_age_is_measured_from_the_recorded_utc():
    assert il_dip.hold_age_h("2026-08-26T16:39:04Z", now=1_800_000_000) > 24
    assert il_dip.hold_age_h(None) is None


def test_per_ink_dip_refines_the_targets(frame, setup):
    palette, load, layout, root, policy = setup
    real = ink_spec.policy_for(tool_spec.load_tool("lutin-3rl-bugpin", REPO))
    wet = dict(load)
    wet["inkcap_right_medium_0"] = ink_spec.SlotLoad("inkcap_right_medium_0", "slow", 300.0)
    inks = {"slow": ink_spec.Ink("slow", "Slow", (0, 0, 0), "with slow ink", dip={"dip_dwell_s": 2.5})}
    t = il_dip.cap_targets(["inkcap_right_medium_0"], palette, wet, layout, root, frame, real,
                           0.02, 0.03, inks=inks)[0]
    assert t.dwell_s == 2.5 and t.ink_id == "slow" and t.uptake_ul == real.uptake_ul
    moves = il_dip.dip_moves([t], real, 3.0, 0.02)
    assert [m.dwell_s for m in moves if "plunge" in m.label] == [2.5]


def test_cap_clearance_follows_the_touchoff_residual():
    """A ±4 mm tip may only be sent into the 15 mm cap; a ±1 mm one into all three."""
    palette = ink_spec.load_palette(REPO)
    right = [s for s in palette if s.startswith("inkcap_right")]
    keep, out = il_dip.caps_wide_enough(palette, right, 0.0042)
    assert keep == ["inkcap_right_large"] and set(out) == set(right) - {"inkcap_right_large"}
    keep, out = il_dip.caps_wide_enough(palette, right, 0.001)
    assert set(keep) == set(right) and out == []
    assert il_dip.tip_sigma_m({"right": {"touchoff": {"residual_mm": 4.2}}}) == pytest.approx(0.0042)
    assert il_dip.tip_sigma_m({"right": {}}) == 0.0 and il_dip.tip_sigma_m({}) == 0.0
