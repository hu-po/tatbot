"""Pin the touch-off solve, its refusal gates, and the workspace roundtrip.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_il_touchoff.py

The most important behaviour is the refusal: four near-vertical touches give a
beautiful residual and a wrong constant, and the tool must write nothing. The
second most important is that what il_touchoff.py writes, the analyzer's
hand-rolled two-level parser reads back — the two ends of workspace.yaml must
never drift apart.
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "vision"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import il_analyze_rollout  # noqa: E402
import il_touchoff  # noqa: E402
import tool_spec  # noqa: E402
from calib_synth import NUM_JOINTS, write_wxtl  # noqa: E402

P_TRUE = np.array([0.012, -0.004, 0.087])   # pen tip in the EE frame
PLANE_TRUE = 0.041                          # plate height, base frame


def touch_pose(rotation_vector, rng):
    """EE pose whose tip (at P_TRUE) lies exactly on the plate."""
    from solve_robot_world import vector_to_rotation
    pose = np.eye(4)
    pose[:3, :3] = vector_to_rotation(np.asarray(rotation_vector, float))
    pose[:2, 3] = rng.uniform(-0.15, 0.15, 2)
    pose[2, 3] = PLANE_TRUE - pose[2, :3] @ P_TRUE
    return pose


def varied_poses(count=8):
    rng = np.random.default_rng(11)
    return [touch_pose(rng.normal(0, 0.5, 3), rng) for _ in range(count)]


def test_solve_recovers_tip_and_plane():
    fit = il_touchoff.solve_plate(varied_poses())
    assert np.allclose(fit["p"], P_TRUE, atol=1e-9)
    assert abs(fit["plane_z"] - PLANE_TRUE) < 1e-9
    assert fit["cond"] < il_touchoff.COND_MAX
    assert fit["rms_mm"] < 0.001
    assert il_touchoff.holdout_residual_mm(varied_poses()) < 0.001


def test_vertical_touches_are_unidentifiable():
    """All touches near one orientation: tiny residual, huge condition number.
    This is the trap the gate exists for."""
    rng = np.random.default_rng(5)
    poses = [touch_pose(rng.normal(0, 0.005, 3), rng) for _ in range(8)]
    fit = il_touchoff.solve_plate(poses)
    assert fit["rms_mm"] < 0.01, "the fit LOOKS excellent"
    assert fit["cond"] > il_touchoff.COND_MAX, "and must still be refused"
    assert fit["spread_deg"] < 2.0


def test_workspace_roundtrip_through_analyzer_parser(tmp_path, monkeypatch):
    """What the writer renders, il_analyze_rollout.load_workspace must read."""
    right = {
        "pen_tip_offset_x": 0.012, "pen_tip_offset_y": -0.004,
        "pen_tip_offset_z": 0.087, "paper_plane_z": 0.0405,
        "paper_band_mm": 4.20, "ee_contact_z": None,
        "touchoff": {"utc": "2026-08-21T20:00:00Z", "session": "s", "n_plate": 8,
                     "n_pad": 3, "cond": 4.2, "residual_mm": 0.1,
                     "holdout_mm": 0.2, "spread_deg": 55.0, "note": ""},
    }
    text = il_touchoff.render_workspace(right)
    workspace = tmp_path / "config" / "workspace.yaml"
    workspace.parent.mkdir()
    workspace.write_text(text)

    monkeypatch.setattr(il_analyze_rollout, "REPO", tmp_path)
    parsed = il_analyze_rollout.load_workspace()["right"]
    assert parsed["pen_tip_offset_x"] == 0.012
    assert parsed["pen_tip_offset_y"] == -0.004
    assert parsed["pen_tip_offset_z"] == 0.087
    assert parsed["paper_plane_z"] == 0.0405
    assert parsed["paper_band_mm"] == 4.2
    assert parsed["ee_contact_z"] is None



def test_labels_from_events():
    touches = [
        {"start_unix": 100.0, "end_unix": 101.0, "label": "plate"},
        {"start_unix": 110.0, "end_unix": 111.0, "label": "plate"},
        {"start_unix": 120.0, "end_unix": 121.0, "label": "plate"},
    ]
    events = [
        {"start_unix": 100.2, "end_unix": 100.9, "kinds": ["touch"],
         "text": "touching the pad now"},
        {"start_unix": 112.0, "end_unix": 113.0, "kinds": ["discard"],
         "text": "scratch that"},
    ]
    il_touchoff.label_touches(touches, events)
    assert [t["label"] for t in touches] == ["pad", "plate"]
    assert len(touches) == 2, "the discarded touch is gone"


def test_cli_refuses_uniform_touches_and_writes_nothing(tmp_path):
    """End to end through the real URDF: eight touches at one wrist pose must
    exit 2 and leave workspace.yaml untouched."""
    q = np.zeros(NUM_JOINTS)
    q[:6] = [0.1, -0.5, 0.6, 0.0, 0.4, 0.0]
    contact = np.full(NUM_JOINTS, 0.05)
    contact[3] = 2.0
    free = np.full(NUM_JOINTS, 0.05)
    poses, efforts = [], []
    for i in range(8):
        wiggle = q.copy()
        # Base yaw moves the touch point in x/y but provably leaves the
        # world-z row of the EE rotation unchanged: varied position, identical
        # identifiability — exactly the trap.
        wiggle[0] += 0.05 * i
        poses += [wiggle + 0.3, wiggle]   # free travel, then touch
        efforts += [free, contact]
    session = tmp_path / "session"
    session.mkdir()
    write_wxtl(session / "teleop.wxtl", poses, efforts)
    workspace = tmp_path / "workspace.yaml"
    workspace.write_text("right:\n  pen_tip_offset_z: null\n")
    before = workspace.read_text()

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "il_touchoff.py"), str(session),
         "--tool-id", "lutin-ballpoint-dot",
         "--write", "--workspace", str(workspace)],
        capture_output=True, text=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "REFUSED" in result.stdout
    assert workspace.read_text() == before, "refusal must write nothing"


def test_cli_requires_a_stated_tool_id(tmp_path):
    """--tool-id used to fall back to whatever workspace.yaml named, which is
    the PREVIOUS tool — the one thing a swap invalidates. On 2026-08-26 that
    gated a laser-pen session against the ballpoint's datasheet and refused a
    correct 126 mm fit for landing 68.8 mm from the wrong tip. Omitting it
    must fail loudly instead of guessing, and an unknown name must say so."""
    session = tmp_path / "session"
    session.mkdir()
    q = np.zeros(NUM_JOINTS)
    q[:6] = [0.1, -0.5, 0.6, 0.0, 0.4, 0.0]
    contact = np.full(NUM_JOINTS, 0.05)
    contact[3] = 2.0
    write_wxtl(session / "teleop.wxtl", [q + 0.3, q], [np.full(NUM_JOINTS, 0.05), contact])

    missing = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "il_touchoff.py"), str(session)],
        capture_output=True, text=True)
    assert missing.returncode != 0
    assert "--tool-id" in missing.stderr

    unknown = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "il_touchoff.py"), str(session),
         "--tool-id", "no-such-pen"],
        capture_output=True, text=True)
    assert unknown.returncode != 0
    # The remedy is the list of real names, not a traceback.
    assert "no-such-pen" in unknown.stderr and "known tools" in unknown.stderr
    assert "Traceback" not in unknown.stderr


def pivot_poses(rotation_scale, count, rng, slip_indices=()):
    """Synthetic planted-tip roll: R_i varied, t_i = P - R_i p (+ slip)."""
    from solve_robot_world import vector_to_rotation
    pivot_true = np.array([0.21, -0.04, 0.041])
    poses = []
    for index in range(count):
        pose = np.eye(4)
        pose[:3, :3] = vector_to_rotation(rng.normal(0, rotation_scale, 3))
        pose[:3, 3] = pivot_true - pose[:3, :3] @ P_TRUE
        if index in slip_indices:
            pose[:3, 3] += np.array([0.008, 0.0, 0.004])   # tip slid 9 mm
        poses.append(pose)
    return poses, pivot_true


def test_pivot_recovers_full_3d_tip():
    rng = np.random.default_rng(3)
    poses, pivot_true = pivot_poses(0.5, 120, rng)
    fit = il_touchoff.solve_pivot(poses)
    assert np.allclose(fit["p"], P_TRUE, atol=1e-9)
    assert np.allclose(fit["pivot"], pivot_true, atol=1e-9)
    assert fit["spread_deg"] > il_touchoff.PIVOT_SPREAD_MIN_DEG
    assert fit["rms_mm"] < 0.001
    assert fit["cond"] < il_touchoff.COND_MAX


def test_pivot_without_rotation_is_unidentifiable():
    """Tip planted but wrist barely rolled: tiny residual, and the spread
    gate is what stands between that and a wrong constant."""
    rng = np.random.default_rng(4)
    poses, _ = pivot_poses(0.02, 120, rng)
    fit = il_touchoff.solve_pivot(poses)
    assert fit["rms_mm"] < 0.01
    assert fit["spread_deg"] < il_touchoff.PIVOT_SPREAD_MIN_DEG


def test_pivot_trim_drops_slips():
    rng = np.random.default_rng(5)
    poses, _ = pivot_poses(0.5, 150, rng, slip_indices=range(140, 150))
    fit = il_touchoff.solve_pivot_trimmed(poses)
    assert fit["dropped"] == 10, "the 10 slid samples must be trimmed"
    assert np.allclose(fit["p"], P_TRUE, atol=1e-6)
    assert fit["rms_mm"] < 0.01


def test_residual_gate_widens_only_for_blunt_tools():
    """The planted-tip solve assumes one contact point holds still. A tool
    whose contact is a disc of radius r cannot: tilting walks the contact
    across the face, ~r/sqrt(2) rms. So the gate is per-tool.

    The point is that this must NOT quietly loosen the pens. A 0.5 mm
    ballpoint and a 3RL needle both stay at the original 1.5 mm; only the
    laser's 15.4 mm lens face buys headroom, and only as much as its own
    geometry accounts for.
    """
    gates = {t: il_touchoff.residual_gate_mm(tool_spec.load_tool(t, REPO))
             for t in ("lutin-3rl-bugpin", "lutin-ballpoint-dot",
                       "picosecond-laser-pen")}
    assert gates["lutin-3rl-bugpin"] == il_touchoff.PIVOT_RESIDUAL_MAX_MM
    # The ballpoint's widening is its SEAT, not its ball: the printed mount's
    # bore is a ~33 mm clearance around a 29 mm body, the clamp locates the
    # machine, and sweep-20260831_082526 measured the contact migrating with
    # wrist pose at 3.0 mm rms. Its datasheet owns that as seat_residual_m.
    assert gates["lutin-ballpoint-dot"] == pytest.approx(
        tool_spec.load_tool("lutin-ballpoint-dot", REPO).seat_residual_m * 1000)

    laser = tool_spec.load_tool("picosecond-laser-pen", REPO)
    expected = laser.tip_radius_m * 1000 / math.sqrt(2)
    assert gates["picosecond-laser-pen"] == expected
    # The 2026-08-26 nine-hold capture measured 4.256 mm; it must clear this
    # gate, and a gross slip at twice that must not.
    assert 4.256 < gates["picosecond-laser-pen"] < 8.512

    # Never below the floor, whatever a datasheet claims, and no spec at all
    # falls back to the floor rather than to "anything goes".
    assert il_touchoff.residual_gate_mm(None) == il_touchoff.PIVOT_RESIDUAL_MAX_MM
    assert all(g >= il_touchoff.PIVOT_RESIDUAL_MAX_MM for g in gates.values())


def test_a_hold_with_the_carriage_pushed_off_rest_is_not_a_touch():
    """The mount rides the carriage: a hold whose carriage lifted was one
    where the pen was driven up its own axis (the contact cap yielding), and
    the tip was not where the FK at rest says. Six-joint holds from older
    fusers are taken at rest and cannot trip this."""
    rest = il_touchoff.CARRIAGE_REST_M
    holds = [[0.1, 1.2, 1.3, -1.2, 0.0, 1.5, rest],
             [0.1, 1.2, 1.3, -1.2, 0.0, 1.5, rest + 0.0002],
             [0.1, 1.2, 1.3, -1.2, 0.0, 1.5, rest + 0.003],
             [0.1, 1.2, 1.3, -1.2, 0.0, 1.5]]
    pushed = il_touchoff.carriage_off_rest(holds)
    assert pushed == [(2, pytest.approx(rest + 0.003))]
