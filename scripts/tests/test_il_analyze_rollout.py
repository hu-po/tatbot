"""Unit tests for scripts/il_analyze_rollout.py.

Tests data-transforming functions: CSV parser, workspace parser, kinematics transforms,
path quality metrics (SPARC, LDJ, hull area, turn stats), dwell/lift/descent stats,
run config & checks builder, report/compare formatters, ink debiting, and helper functions.

    uvx --with pytest --with numpy --with scipy pytest -q scripts/tests/test_il_analyze_rollout.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "vision"))

import il_analyze_rollout as ana  # noqa: E402
from urdf_kinematics import UrdfChain  # noqa: E402

# --- Helper to create a realistic flight CSV ---------------------------------


def make_synthetic_flight_csv(
    csv_path: Path,
    n_rows: int = 100,
    dt: float = 1.0 / 30.0,
    z_min: float = 0.005,
    z_max: float = 0.05,
    clamped: bool = False,
) -> Path:
    """Generates a valid CSV with all expected columns for il_analyze_rollout."""
    fieldnames = [
        "t_mono",
        "pos_joint_0",
        "pos_joint_1",
        "pos_joint_2",
        "pos_joint_3",
        "pos_joint_4",
        "pos_joint_5",
        "pos_left_carriage_joint",
        "eff_joint_0",
        "eff_joint_1",
        "eff_joint_2",
        "eff_joint_3",
        "eff_joint_4",
        "eff_joint_5",
        "eff_left_carriage_joint",
        "contact_force_n",
        "raw_joint_0",
        "raw_joint_1",
        "raw_joint_2",
        "raw_joint_3",
        "raw_joint_4",
        "raw_joint_5",
    ]

    t = 0.0
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_rows):
            frac = i / float(n_rows)
            # z decreases from high to low to simulate descent
            z_curr = z_max - (z_max - z_min) * frac
            raw_j0 = "1.0" if clamped else "0.1"
            row = {
                "t_mono": str(t),
                "pos_joint_0": "0.1",
                "pos_joint_1": "0.2",
                "pos_joint_2": "-0.3",
                "pos_joint_3": "0.4",
                "pos_joint_4": "0.5",
                "pos_joint_5": str(z_curr),
                "pos_left_carriage_joint": "0.0",
                "eff_joint_0": "1.0",
                "eff_joint_1": "1.5",
                "eff_joint_2": "2.0",
                "eff_joint_3": "0.5",
                "eff_joint_4": "0.2",
                "eff_joint_5": "0.1",
                "eff_left_carriage_joint": "5.0",
                "contact_force_n": "12.5",
                "raw_joint_0": raw_j0,
                "raw_joint_1": "0.2",
                "raw_joint_2": "-0.3",
                "raw_joint_3": "0.4",
                "raw_joint_4": "0.5",
                "raw_joint_5": str(z_curr),
            }
            writer.writerow(row)
            t += dt

    return csv_path


# --- Test load_rows -----------------------------------------------------------


def test_load_rows(tmp_path):
    csv_file = tmp_path / "test.csv"
    make_synthetic_flight_csv(csv_file, n_rows=5)
    rows = ana.load_rows(csv_file)
    assert len(rows) == 5
    assert "t_mono" in rows[0]
    assert "pos_joint_0" in rows[0]


# --- Test pen_path ------------------------------------------------------------


def test_pen_path(tmp_path):
    csv_file = tmp_path / "test.csv"
    make_synthetic_flight_csv(csv_file, n_rows=10)
    rows = ana.load_rows(csv_file)
    chain = UrdfChain(str(REPO / "urdf" / "tatbot.urdf"))
    names = chain.arm_joint_names("right")

    path_no_offset = ana.pen_path(rows, chain, names, tip_offset_m=None)
    assert path_no_offset.shape == (10, 3)

    offset = [0.01, -0.02, 0.05]
    path_with_offset = ana.pen_path(rows, chain, names, tip_offset_m=offset)
    assert path_with_offset.shape == (10, 3)
    assert not np.allclose(path_no_offset, path_with_offset)


# --- Test resample_arclength --------------------------------------------------


def test_resample_arclength():
    # Short path (< step_mm * 4) -> returns xy[:1]
    xy_short = np.array([[0.0, 0.0], [0.1, 0.1]])
    res_short = ana.resample_arclength(xy_short, step_mm=1.0)
    assert len(res_short) == 1
    assert np.allclose(res_short[0], xy_short[0])

    # Longer straight path (10 mm) with step_mm=0.5
    xy_long = np.array([[0.0, 0.0], [10.0, 0.0]])
    res_long = ana.resample_arclength(xy_long, step_mm=0.5)
    assert len(res_long) == 20  # 0.0 to 9.5 in 0.5 steps
    assert np.allclose(res_long[:, 1], 0.0)
    assert res_long[0, 0] == pytest.approx(0.0)
    assert res_long[-1, 0] == pytest.approx(9.5)


# --- Test turn_stats ----------------------------------------------------------


def test_turn_stats():
    # Collinear points: 0 turn angle, 0 reversal
    xy_straight = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    mean_ang, rev_pct = ana.turn_stats(xy_straight)
    assert mean_ang == pytest.approx(0.0)
    assert rev_pct == pytest.approx(0.0)

    # 180-degree reversal: forward then backward
    xy_reversal = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    mean_ang, rev_pct = ana.turn_stats(xy_reversal)
    assert mean_ang == pytest.approx(180.0, abs=1e-3)
    assert rev_pct == pytest.approx(100.0)

    # Too short or tiny steps (< MIN_STEP_MM) -> nan, nan
    xy_tiny = np.array([[0.0, 0.0], [0.01, 0.0]])
    ang, rev = ana.turn_stats(xy_tiny)
    assert np.isnan(ang) and np.isnan(rev)


# --- Test hull_area -----------------------------------------------------------


def test_hull_area():
    # Less than 3 points
    assert ana.hull_area(np.array([[0.0, 0.0], [1.0, 1.0]])) == 0.0

    # 3 Collinear points
    assert ana.hull_area(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])) == 0.0

    # 10x10 square
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]])
    assert ana.hull_area(square) == pytest.approx(100.0)


# --- Test load_workspace -----------------------------------------------------


def test_load_workspace(tmp_path, monkeypatch):
    # Non-existent file
    monkeypatch.setattr(ana, "REPO", tmp_path)
    assert ana.load_workspace() == {}

    # Mock workspace.yaml
    ws_dir = tmp_path / "config"
    ws_dir.mkdir(parents=True)
    ws_file = ws_dir / "workspace.yaml"
    ws_content = """# Comment line
right:
  tool_id: "lutin-ballpoint-dot"
  paper_plane_z: 0.005945
  paper_band_mm: null
  note:
  touchoff:
    ignore_me: 123
"""
    ws_file.write_text(ws_content)

    ws = ana.load_workspace()
    assert "right" in ws
    assert ws["right"]["tool_id"] == "lutin-ballpoint-dot"
    assert ws["right"]["paper_plane_z"] == 0.005945
    assert ws["right"]["paper_band_mm"] is None
    assert ws["right"]["note"] is None
    # Nested blocks under touchoff at indent 4 are ignored
    assert "ignore_me" not in ws["right"]


# --- Test sparc ---------------------------------------------------------------


def test_sparc():
    # Short speed array (< 16) -> nan
    assert np.isnan(ana.sparc(np.ones(10), fs=30.0))

    # All zero speed array -> nan
    assert np.isnan(ana.sparc(np.zeros(20), fs=30.0))

    # Smooth sinusoidal speed profile
    t = np.linspace(0, 1, 100)
    speed = np.sin(np.pi * t)
    val = ana.sparc(speed, fs=100.0)
    assert isinstance(val, float)
    assert not np.isnan(val)
    assert val < 0  # SPARC is negative


# --- Test log_dimensionless_jerk ---------------------------------------------


def test_log_dimensionless_jerk():
    # Short path (< 8) -> nan
    assert np.isnan(ana.log_dimensionless_jerk(np.zeros((5, 3)), dt=0.033))

    # Zero velocity -> nan
    assert np.isnan(ana.log_dimensionless_jerk(np.zeros((10, 3)), dt=0.033))

    # Moving path
    t = np.linspace(0, 1, 50)[:, None]
    p3 = np.hstack([t, np.zeros_like(t), np.zeros_like(t)])
    val = ana.log_dimensionless_jerk(p3, dt=0.02)
    assert isinstance(val, float)
    assert not np.isnan(val)


# --- Test pad_dwell -----------------------------------------------------------


def test_pad_dwell():
    z = np.linspace(0.0, 10.0, 100)
    res = ana.pad_dwell(z, pad_play=5.0)
    assert "hover_mm" in res
    assert "floor_mm" in res
    assert "drop_mm" in res
    assert "dwell_pct" in res
    assert res["floor_mm"] <= res["hover_mm"]


# --- Test lift_stats ----------------------------------------------------------


def test_lift_stats():
    # Constant floor z -> 0 lifts
    z_flat = np.ones(100) * 2.0
    t = np.linspace(0, 10, 100)
    res_flat = ana.lift_stats(z_flat, t)
    assert res_flat["lift_events"] == 0
    assert res_flat["lift_s"] == 0.0

    # z with a sustained lift (> 6 mm clearance, > 0.3 s duration)
    z_lift = np.ones(100) * 2.0
    z_lift[20:50] = 15.0  # lifted for 3 seconds
    res_lift = ana.lift_stats(z_lift, t, clearance_mm=6.0, min_s=0.3)
    assert res_lift["lift_events"] == 1
    assert res_lift["lift_s"] > 2.5


# --- Test descent_stats -------------------------------------------------------


def test_descent_stats():
    # Starts low (z[0] < band + min_start_mm) -> returns {"descent": None}
    z_low = np.array([5.0, 4.0, 3.0, 2.0])
    t_low = np.array([0.0, 0.1, 0.2, 0.3])
    assert ana.descent_stats(z_low, t_low, floor_mm=2.0)["descent"] is None

    # Starts high, never reaches band -> reached_band=False
    z_high_stay = np.array([50.0, 45.0, 40.0, 35.0])
    t_stay = np.array([0.0, 0.5, 1.0, 1.5])
    res_never = ana.descent_stats(z_high_stay, t_stay, floor_mm=2.0, min_start_mm=20.0)
    d_never = res_never["descent"]
    assert isinstance(d_never, dict)
    assert d_never["reached_band"] is False

    # Starts high, descends into band
    z_desc = np.array([50.0, 30.0, 10.0, 3.0, 2.0])
    t_desc = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    res_desc = ana.descent_stats(z_desc, t_desc, floor_mm=2.0, min_start_mm=20.0)
    d_desc = res_desc["descent"]
    assert isinstance(d_desc, dict)
    assert d_desc["reached_band"] is True
    assert d_desc["time_to_band_s"] == 1.5


# --- Test _run_config and build_checks ---------------------------------------


def test_run_config_and_build_checks():
    meta = {
        "key": {
            "fps": "30",
            "grip_force": "33.0",
            "policy": "test-policy",
            "refill_budget_ms": "100",
            "infer_ms": "80",
        }
    }
    cfg = ana._run_config(meta)
    assert cfg["fps"] == 30
    assert cfg["grip_force"] == 33.0
    assert cfg["policy"] == "test-policy"

    m = {
        "loop_hz_mean": 29.5,
        "stall_ticks_lost_pct": 0,
        "stall_count": 0,
        "stall_ms_median": 0,
        "grip_cmd_peak_n": 30.0,
        "grip_meas_peak_n": 35.0,
        "clamp_ticks_post_settle": 5,
    }
    geom = {"contact_basis": "touchoff_tip", "valid": True}

    checks = ana.build_checks(m, cfg, geom)
    assert len(checks) == 5
    check_names = {c["name"] for c in checks}
    assert "loop_rate" in check_names
    assert "grip_cmd_ceiling" in check_names
    assert "grip_measured" in check_names
    assert "clamp_post_settle" in check_names

    # Check failing condition for grip_cmd_ceiling
    m_fail = dict(m, grip_cmd_peak_n=40.0)
    checks_fail = ana.build_checks(m_fail, cfg, geom)
    cmd_check = next(c for c in checks_fail if c["name"] == "grip_cmd_ceiling")
    assert cmd_check["status"] == "fail"


# --- Test helpers: _wrap, _node_from_run_id, _short_host ---------------------


def test_helpers():
    wrapped = ana._wrap("hello world test wrapping string", 12)
    assert len(wrapped) > 1

    assert ana._node_from_run_id("20260821T164233Z-nodea-a3f1") == "nodea"
    assert ana._node_from_run_id("invalid-id") is None

    assert ana._short_host("nodea.local.domain") == "nodea"
    assert ana._short_host(None) is None


# --- Test analyze -------------------------------------------------------------


def test_analyze(tmp_path, monkeypatch):
    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    csv_file = run_dir / "flight-001.csv"
    make_synthetic_flight_csv(csv_file, n_rows=100, dt=0.1, clamped=True)

    meta = {"run_id": "run_01", "key": {"max_relative_target": 0.5}}
    (run_dir / "meta.json").write_text(json.dumps(meta))

    a = ana.analyze(run_dir, settle=1.0, paper_z=10.0)
    assert a["schema"] == "tatbot.rollout.analysis/1"
    assert a["run_id"] == "run_01"
    assert a["geometry"]["contact_basis"] == "external"
    assert a["geometry"]["plane_z_mm"] == 10.0
    assert "metrics" in a
    assert a["metrics"]["ticks"] == 100

    # Prepare mocked REPO structure with urdf for workspace tests
    monkeypatch.setattr(ana, "REPO", tmp_path)
    (tmp_path / "urdf").mkdir(parents=True)
    shutil.copy(REPO / "urdf" / "tatbot.urdf", tmp_path / "urdf" / "tatbot.urdf")

    ws_dir = tmp_path / "config"
    ws_dir.mkdir(parents=True)
    (ws_dir / "workspace.yaml").write_text("""
right:
  tip_frame: right/tool_mount
  pen_tip_offset_x: 0.0
  pen_tip_offset_y: 0.0
  pen_tip_offset_z: 0.05
  paper_plane_z: 0.01
""")
    a_ws = ana.analyze(run_dir, settle=1.0)
    assert a_ws["geometry"]["contact_basis"] == "touchoff_tip"

    # Test touchoff_ee contact basis via workspace
    (ws_dir / "workspace.yaml").write_text("""
right:
  ee_contact_z: 0.01
""")
    a_ee = ana.analyze(run_dir, settle=1.0)
    assert a_ee["geometry"]["contact_basis"] == "touchoff_ee"

    # Test inferred contact basis
    (ws_dir / "workspace.yaml").write_text("""
right: {}
""")
    a_inf = ana.analyze(run_dir, settle=1.0)
    assert a_inf["geometry"]["contact_basis"] == "inferred"

    # Test short CSV error
    short_dir = tmp_path / "run_short"
    short_dir.mkdir()
    short_csv = short_dir / "flight-short.csv"
    make_synthetic_flight_csv(short_csv, n_rows=10)
    with pytest.raises(SystemExit, match="only 10 rows"):
        ana.analyze(short_dir)

    # Test directory with no CSVs
    empty_dir = tmp_path / "run_empty"
    empty_dir.mkdir()
    with pytest.raises(SystemExit, match="no flight CSV"):
        ana.analyze(empty_dir)


# --- Test print_report and compare -------------------------------------------


def test_print_report_and_compare(tmp_path, capsys):
    run_dir = tmp_path / "run_report"
    run_dir.mkdir()
    csv_file = run_dir / "flight-001.csv"
    make_synthetic_flight_csv(csv_file, n_rows=100, dt=0.1)

    a: dict[str, Any] = ana.analyze(run_dir, settle=1.0, paper_z=5.0)
    ana.print_report(a)
    captured = capsys.readouterr()
    assert "window t=1..10s" in captured.out
    assert "geometry" in captured.out

    # Test print_report when descent never reached band
    a_no_reach: dict[str, Any] = dict(a)
    m_no_reach = dict(a["metrics"])
    m_no_reach["descent"] = {"start_mm": 50.0, "reached_band": False}
    a_no_reach["metrics"] = m_no_reach
    ana.print_report(a_no_reach)

    # Write analysis.json files for compare test
    a_file1 = tmp_path / "analysis1.json"
    a_file2 = tmp_path / "analysis2.json"
    a_file3 = tmp_path / "analysis3.json"
    a_file1.write_text(json.dumps(a))

    a2: dict[str, Any] = dict(a, run_id="run_02")
    m2 = dict(a["metrics"])
    m2["z_min_mm"] = 1.0
    m2["loop_hz_mean"] = 28.0
    a2["metrics"] = m2

    a3: dict[str, Any] = dict(a, run_id="run_03")
    m3 = dict(a["metrics"])
    m3["z_min_mm"] = 2.0
    m3["loop_hz_mean"] = 30.0
    a3["metrics"] = m3

    a_file2.write_text(json.dumps(a2))
    a_file3.write_text(json.dumps(a3))

    ana.compare([a_file1, a_file2, a_file3])
    comp_captured = capsys.readouterr()
    assert "SPARC" in comp_captured.out
    assert "dwell" in comp_captured.out

    # Compare with inferred plane (contact% withheld)
    a_inf: dict[str, Any] = dict(a, run_id="run_inf")
    g_inf = dict(a["geometry"])
    g_inf["valid"] = False
    a_inf["geometry"] = g_inf
    a_inf_file = tmp_path / "analysis_inf.json"
    a_inf_file.write_text(json.dumps(a_inf))
    ana.compare([a_file1, a_inf_file])
    captured_inf = capsys.readouterr()
    assert "contact% withheld" in captured_inf.out

    # Compare with empty list
    with pytest.raises(SystemExit, match="no readable analysis.json files"):
        ana.compare([tmp_path / "nonexistent.json"])


# --- Test main entry point options ------------------------------------------


def test_main_cli(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run_cli"
    run_dir.mkdir()
    csv_file = run_dir / "flight-001.csv"
    make_synthetic_flight_csv(csv_file, n_rows=100, dt=0.1)

    # Test main with --json
    monkeypatch.setattr(
        sys, "argv", ["il_analyze_rollout.py", str(run_dir), "--json", "--settle", "1.0"]
    )
    rc = ana.main()
    assert rc == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    assert res["schema"] == "tatbot.rollout.analysis/1"

    # Test main with no args
    monkeypatch.setattr(sys, "argv", ["il_analyze_rollout.py"])
    with pytest.raises(SystemExit):
        ana.main()

    # Test main with --compare
    a_file = run_dir / "analysis.json"
    a_file.write_text(json.dumps(res))
    monkeypatch.setattr(sys, "argv", ["il_analyze_rollout.py", "--compare", str(a_file)])
    rc_comp = ana.main()
    assert rc_comp == 0
